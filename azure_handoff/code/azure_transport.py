"""azure_transport.py -- the ONLY thing that differs from the frozen OpenRouter pipeline.

Balance-9 P3 was run against OpenRouter. The account was stopped mid-experiment (HTTP 402 then 401),
which left the llama half of the P3 matrix incomplete. This module replaces exactly one function --
`balance9_runner.transport_call` -- with an Azure AI Foundry call, and changes nothing else. The
prompts, the payoff tables, the RNG streams, the tie rule, the retry/repair/override logic and the
record schema all come from the byte-identical vendored modules next to this file.

Three things this module refuses to do, because each of them would silently turn a replication into a
different experiment:

  * IT NEVER GUESSES THE ROUTE. Azure exposes chat completions at two incompatible URLs (the Azure
    OpenAI `/openai/deployments/{d}/chat/completions` shape and the Azure AI model-inference
    `/chat/completions` shape). Trying one and falling back to the other on a 404 is the kind of
    "helpful" fallback that ends with nobody knowing which service answered. The route is decided
    once, from the endpoint host or from an explicit `AZURE_AI_ROUTE`, recorded in the manifest and
    stamped on EVERY attempt record. An endpoint whose route cannot be decided is a startup error.

  * IT NEVER ACCEPTS A DIFFERENT MODEL. The first live response is checked against the expected model
    identity and the whole run aborts on a mismatch, before any raw record exists. A quantised or
    community rebuild of Llama-3.3-70B is a different measurement instrument, and substituting one
    for the other silently is precisely the failure this package exists to avoid. The check can be
    overridden, but only by naming the exact accepted string in `AZURE_AI_ALLOW_MODEL_NAME`, which is
    then recorded as an override in the manifest and in the final report.

  * IT NEVER PRINTS, LOGS OR HASHES A CREDENTIAL. Errors are redacted through `redact()` before they
    reach a log, a record or a traceback. Note that a hash is not a redaction: a hash of a secret is
    a verifier for that secret, so credentials are not hashed either.

The retry shape (7 attempts, exponential, jittered, capped at 30s) is copied from the frozen
`balance9_runner.transport_call` so the transport-attempt counter keeps the meaning it has in the
OpenRouter half of the corpus.
"""

from __future__ import annotations

import json
import os
import random
import re
import ssl
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from contextvars import ContextVar
from pathlib import Path
from urllib.parse import urlparse, urlunparse

# ---------------------------------------------------------------------------------------------
# frozen transport constants -- these mirror balance9_runner, and gate A-T1 proves it
# ---------------------------------------------------------------------------------------------
TRANSPORT_RETRIES = 6              # == balance9_runner.TRANSPORT_RETRIES
BACKOFF_CAP_S = 30.0
TIMEOUT_S = float(os.environ.get("AZURE_AI_TIMEOUT_S", "180"))

# HTTP statuses that must NOT be retried: retrying an auth or quota failure just burns the budget
# and buries the real cause under six identical lines. 402/401 are exactly how the OpenRouter run
# died, so they are named explicitly rather than left to a generic 4xx rule.
HARD_HTTP = {400, 401, 402, 403, 404, 405, 501}
# 429 and 5xx are retryable; so is a transport-level error with no status at all.

EXPECTED_MODEL_TOKENS = ("llama", "3.3", "70b")
FORBIDDEN_MODEL_TOKENS = ("awq", "gptq", "gguf", "int4", "int8", "fp8", "-q4", "-q8",
                          "bnb", "nf4", "quant", "turbo", "draft", "speculative")

DEFAULT_API_VERSION = "2024-05-01-preview"


class AzureTransportError(RuntimeError):
    """A hard transport failure that must stop the whole run. Message is already redacted.

    This is deliberately NOT `balance9_runner.TechnicalFailure`. The frozen driver catches
    TechnicalFailure and re-runs the whole market seed up to three times; that is the right response
    to a flaky completion and the wrong response to a dead account. HTTP 401/402 is how the
    OpenRouter run actually died, and three silent seed re-runs against a revoked key would have
    turned one legible failure into a wall of noise. So a hard HTTP status escapes the seed loop and
    stops the run, and only the "exhausted all transport attempts" case is downgraded to
    TechnicalFailure so the frozen whole-seed-rerun policy still applies to genuine flakiness.
    """


class AzureConfigError(RuntimeError):
    """A configuration failure. Raised at startup or on the first live response, never guessed
    around, and always fatal to the run."""


def _technical_failure(msg):
    """`balance9_runner.TechnicalFailure`, resolved lazily so this module has no import-time
    dependency on the frozen runner (and so the offline tests can import it on its own)."""
    try:
        import balance9_runner as _R9                   # noqa: PLC0415
        return _R9.TechnicalFailure(msg)
    except Exception:                                   # noqa: BLE001
        return AzureTransportError(msg)


# ---------------------------------------------------------------------------------------------
# redaction -- applied to every string that can reach a log, a record or a traceback
# ---------------------------------------------------------------------------------------------
_SECRET_PATTERNS = [
    re.compile(r"(?i)\b(api[-_ ]?key|authorization|bearer|ocp-apim-subscription-key)\b\s*[:=]\s*\S+"),
    re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}"),  # JWT
    re.compile(r"\bsk-[A-Za-z0-9\-_]{16,}"),
    re.compile(r"\b[0-9a-f]{32}\b"),          # classic Azure 32-hex key
    re.compile(r"\b[A-Za-z0-9]{60,}\b"),      # long opaque token
]

_LIVE_SECRETS: list[str] = []                 # populated at startup; never written anywhere


def register_secret(value: str) -> None:
    """Remember a literal secret so `redact` can remove it even if no pattern matches it."""
    v = (value or "").strip()
    if len(v) >= 8:
        _LIVE_SECRETS.append(v)


def redact(text) -> str:
    s = str(text)
    for lit in _LIVE_SECRETS:
        s = s.replace(lit, "***REDACTED***")
    for pat in _SECRET_PATTERNS:
        s = pat.sub("***REDACTED***", s)
    return s


# ---------------------------------------------------------------------------------------------
# per-decision context -- set by the wrapper in to_run.py, read here, never mutated by frozen code
# ---------------------------------------------------------------------------------------------
CTX: ContextVar[dict] = ContextVar("azure_decision_ctx", default={})


def set_context(**kw) -> None:
    CTX.set(dict(kw))


def get_context() -> dict:
    try:
        return dict(CTX.get())
    except LookupError:
        return {}


# ---------------------------------------------------------------------------------------------
# configuration -- decided once, recorded, never re-decided mid-run
# ---------------------------------------------------------------------------------------------
class AzureConfig:
    def __init__(self, endpoint, deployment, api_version, route, auth_mode,
                 expected_model, allow_model_name):
        self.endpoint = endpoint
        self.deployment = deployment
        self.api_version = api_version
        self.route = route                    # "azure-openai" | "azure-ai-model-inference"
        self.auth_mode = auth_mode            # "api-key" | "default-azure-credential"
        self.expected_model = expected_model
        self.allow_model_name = allow_model_name
        self.host = urlparse(endpoint).hostname or ""
        self.region = None                    # filled from the first response header
        self.observed_model = None
        self.content_filter_seen = None
        self._token = None
        self._token_exp = 0.0
        self._lock = threading.Lock()
        self._cred = None

    # -- URL ----------------------------------------------------------------------------------
    def url(self) -> str:
        p = urlparse(self.endpoint)
        base = p.path.rstrip("/")
        if self.route == "azure-openai":
            path = f"{base}/openai/deployments/{self.deployment}/chat/completions"
        else:
            path = f"{base}/chat/completions" if not base.endswith("/chat/completions") else base
        return urlunparse((p.scheme, p.netloc, path, "", f"api-version={self.api_version}", ""))

    # -- auth ---------------------------------------------------------------------------------
    def _bearer(self) -> str:
        """A DefaultAzureCredential token, cached until 5 minutes before expiry."""
        with self._lock:
            if self._token and time.time() < self._token_exp - 300:
                return self._token
            if self._cred is None:
                try:
                    from azure.identity import DefaultAzureCredential  # noqa: PLC0415
                except ImportError as exc:
                    raise AzureConfigError(
                        "AZURE_AI_API_KEY is not set and azure-identity is not installed; "
                        "either set the key or `pip install azure-identity`") from exc
                self._cred = DefaultAzureCredential()
            scope = os.environ.get("AZURE_AI_TOKEN_SCOPE",
                                   "https://cognitiveservices.azure.com/.default")
            tok = self._cred.get_token(scope)
            self._token, self._token_exp = tok.token, float(tok.expires_on)
            register_secret(self._token)
            return self._token

    def headers(self) -> dict:
        h = {"Content-Type": "application/json",
             "User-Agent": "cwe-balance9-azure-handoff/1.0"}
        if self.auth_mode == "api-key":
            key = os.environ["AZURE_AI_API_KEY"].strip()
            if self.route == "azure-openai":
                h["api-key"] = key
            else:
                h["Authorization"] = f"Bearer {key}"
                h["api-key"] = key            # model-inference accepts either; send both, log neither
        else:
            h["Authorization"] = f"Bearer {self._bearer()}"
        if self.route != "azure-openai":
            h["extra-parameters"] = "pass-through"
        return h

    # -- what goes in the manifest (credentials are structurally absent, not stripped) --------
    def public(self) -> dict:
        return dict(
            endpoint_host=self.host,
            endpoint=self._endpoint_public(),
            deployment=self.deployment,
            api_version=self.api_version,
            route=self.route,
            request_url=self._url_public(),
            auth_mode=self.auth_mode,
            token_scope=(os.environ.get("AZURE_AI_TOKEN_SCOPE",
                                        "https://cognitiveservices.azure.com/.default")
                         if self.auth_mode != "api-key" else None),
            expected_model=self.expected_model,
            model_name_override=self.allow_model_name,
            observed_model=self.observed_model,
            observed_region=self.region,
            timeout_s=TIMEOUT_S,
            transport_retries=TRANSPORT_RETRIES,
            hard_http=sorted(HARD_HTTP),
        )

    def _endpoint_public(self) -> str:
        p = urlparse(self.endpoint)
        return urlunparse((p.scheme, p.netloc, p.path, "", "", ""))

    def _url_public(self) -> str:
        return self.url()


def _decide_route(endpoint: str) -> str:
    """Explicit route selection. An undecidable endpoint is an error, never a guess."""
    forced = os.environ.get("AZURE_AI_ROUTE", "").strip().lower()
    if forced:
        if forced not in ("azure-openai", "azure-ai-model-inference"):
            raise AzureConfigError(
                f"AZURE_AI_ROUTE={forced!r} is not one of "
                "'azure-openai' | 'azure-ai-model-inference'")
        return forced
    host = (urlparse(endpoint).hostname or "").lower()
    path = (urlparse(endpoint).path or "").lower()
    if "openai.azure.com" in host or "/openai" in path:
        return "azure-openai"
    if any(h in host for h in ("services.ai.azure.com", "inference.ai.azure.com",
                               "models.ai.azure.com", "cognitiveservices.azure.com")):
        return "azure-ai-model-inference"
    raise AzureConfigError(
        f"cannot decide the Azure route from endpoint host {host!r}. This is deliberate: guessing "
        "between the Azure OpenAI and Azure AI model-inference URL shapes would make the manifest a "
        "fiction. Set AZURE_AI_ROUTE to 'azure-openai' or 'azure-ai-model-inference'.")


def load_config() -> AzureConfig:
    ep = os.environ.get("AZURE_AI_ENDPOINT", "").strip().rstrip("/")
    if not ep:
        raise AzureConfigError("AZURE_AI_ENDPOINT is not set")
    if not ep.startswith("https://"):
        raise AzureConfigError("AZURE_AI_ENDPOINT must be an https:// URL")
    dep = os.environ.get("AZURE_AI_DEPLOYMENT", "").strip()
    if not dep:
        raise AzureConfigError("AZURE_AI_DEPLOYMENT is not set")
    ver = os.environ.get("AZURE_AI_API_VERSION", "").strip() or DEFAULT_API_VERSION
    route = _decide_route(ep)
    key = os.environ.get("AZURE_AI_API_KEY", "").strip()
    if key:
        register_secret(key)
        auth = "api-key"
    else:
        auth = "default-azure-credential"
    return AzureConfig(ep, dep, ver, route, auth,
                       expected_model=os.environ.get("AZURE_AI_MODEL_EXPECTED",
                                                     "Llama-3.3-70B-Instruct").strip(),
                       allow_model_name=os.environ.get("AZURE_AI_ALLOW_MODEL_NAME", "").strip()
                       or None)


# ---------------------------------------------------------------------------------------------
# model identity -- checked once, on the first live response, before any raw record exists
# ---------------------------------------------------------------------------------------------
def check_model_identity(cfg: AzureConfig, observed: str) -> dict:
    obs = (observed or "").strip()
    low = obs.lower()
    verdict = dict(observed=obs, expected=cfg.expected_model,
                   override=cfg.allow_model_name, accepted=False, reason="")
    if cfg.allow_model_name:
        verdict["accepted"] = (obs == cfg.allow_model_name)
        verdict["reason"] = ("explicit AZURE_AI_ALLOW_MODEL_NAME override matched"
                             if verdict["accepted"] else
                             f"AZURE_AI_ALLOW_MODEL_NAME={cfg.allow_model_name!r} does not equal "
                             f"the observed model {obs!r}")
        return verdict
    bad = [t for t in FORBIDDEN_MODEL_TOKENS if t in low]
    if bad:
        verdict["reason"] = (f"the served model name {obs!r} contains {bad}, which marks a "
                             "quantised/derived build; that is a different instrument")
        return verdict
    missing = [t for t in EXPECTED_MODEL_TOKENS if t not in low]
    if missing:
        verdict["reason"] = (f"the served model name {obs!r} is missing {missing}; expected an "
                             f"exact {cfg.expected_model}. If your deployment legitimately reports "
                             "a custom name, set AZURE_AI_ALLOW_MODEL_NAME to that exact string.")
        return verdict
    verdict["accepted"] = True
    verdict["reason"] = "served model name matches Llama-3.3-70B-Instruct"
    return verdict


# ---------------------------------------------------------------------------------------------
# the attempt log -- append-only, fsync'd, one JSON object per transport attempt
# ---------------------------------------------------------------------------------------------
class AttemptLog:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.n = 0

    def write(self, rec: dict) -> None:
        line = json.dumps(rec, ensure_ascii=False) + "\n"
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line)
                fh.flush()
                os.fsync(fh.fileno())
            self.n += 1


# ---------------------------------------------------------------------------------------------
# the call
# ---------------------------------------------------------------------------------------------
_SSL_CTX = ssl.create_default_context()


def _post(cfg: AzureConfig, body: dict):
    """One HTTP POST. Returns (status, headers, parsed_json). Raises on HTTP or transport error."""
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(cfg.url(), data=data, headers=cfg.headers(), method="POST")
    with urllib.request.urlopen(req, timeout=TIMEOUT_S, context=_SSL_CTX) as resp:
        payload = resp.read().decode("utf-8", "replace")
        return resp.status, dict(resp.headers), json.loads(payload)


class AzureTransport:
    """Callable with the exact signature of `balance9_runner.transport_call`."""

    def __init__(self, cfg: AzureConfig, attempt_log: AttemptLog, on_identity=None,
                 dry_run_hook=None):
        self.cfg = cfg
        self.log = attempt_log
        self.on_identity = on_identity
        self.dry_run_hook = dry_run_hook          # offline tests inject a fake HTTP layer here
        self._identity_done = threading.Event()
        self._identity_lock = threading.Lock()
        self.calls = 0
        self.attempts = 0
        self.retries = 0
        self._stats_lock = threading.Lock()

    # -- identity ---------------------------------------------------------------------------
    def _verify_identity(self, obj, headers):
        if self._identity_done.is_set():
            return
        with self._identity_lock:
            if self._identity_done.is_set():
                return
            observed = str(obj.get("model") or "")
            self.cfg.observed_model = observed
            self.cfg.region = (headers.get("x-ms-region") or headers.get("x-ms-deployment-name")
                               or headers.get("azureml-model-deployment") or None)
            verdict = check_model_identity(self.cfg, observed)
            verdict["response_headers_kept"] = {
                k: v for k, v in headers.items()
                if k.lower() in ("x-ms-region", "x-ms-client-request-id", "x-ratelimit-limit-requests",
                                 "x-ratelimit-remaining-requests", "x-ratelimit-limit-tokens",
                                 "x-ratelimit-remaining-tokens", "azureml-model-deployment",
                                 "x-ms-deployment-name", "apim-request-id", "x-envoy-upstream-service-time")}
            if self.on_identity:
                self.on_identity(verdict)
            self._identity_done.set()
            if not verdict["accepted"]:
                raise AzureConfigError(
                    "MODEL IDENTITY CHECK FAILED -- nothing has been written. "
                    + verdict["reason"])

    # -- payload ----------------------------------------------------------------------------
    def body(self, msgs):
        """The frozen payload: temperature 0, strict JSON object, no other sampling parameter.

        `model` is the Azure DEPLOYMENT name, not the OpenRouter slug. That substitution is the whole
        point of the package and is recorded on every attempt as (openrouter_model_id, deployment).
        """
        b = {"messages": msgs, "temperature": 0,
             "response_format": {"type": "json_object"}}
        if self.cfg.route == "azure-openai":
            # the deployment is already in the URL; Azure OpenAI ignores/echoes `model`
            b["model"] = self.cfg.deployment
        else:
            b["model"] = self.cfg.deployment
        return b

    # -- the transport_call replacement -------------------------------------------------------
    def __call__(self, model_id, msgs, attempts_box):
        """(raw, usage, n_transport_attempts) -- byte-compatible with balance9_runner.transport_call."""
        ctx = get_context()
        call_uid = uuid.uuid4().hex[:16]
        body = self.body(msgs)
        last_err = None
        with self._stats_lock:
            self.calls += 1

        for k in range(TRANSPORT_RETRIES + 1):
            attempts_box[0] += 1
            with self._stats_lock:
                self.attempts += 1
                if k:
                    self.retries += 1
            t0 = time.time()
            rec = dict(
                ts=time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z",
                call_uid=call_uid, transport_attempt=k + 1,
                run_key=ctx.get("run_key"), decision_key=ctx.get("decision_key"),
                seed=ctx.get("seed"),
                interface=ctx.get("interface"), policy=ctx.get("policy"),
                model_alias=ctx.get("model_alias"), round=ctx.get("round"), j=ctx.get("j"),
                openrouter_model_id=model_id,
                deployment=self.cfg.deployment, route=self.cfg.route,
                api_version=self.cfg.api_version, endpoint_host=self.cfg.host,
                auth_mode=self.cfg.auth_mode,
                n_messages=len(msgs),
            )
            try:
                if self.dry_run_hook is not None:
                    status, headers, obj = self.dry_run_hook(self.cfg, body, k)
                else:
                    status, headers, obj = _post(self.cfg, body)
                self._verify_identity(obj, headers)
                choices = obj.get("choices") or []
                if not choices:
                    raise AzureTransportError(
                        f"response contained no choices (finish info: "
                        f"{redact(json.dumps(obj)[:300])})")
                ch0 = choices[0]
                content = (ch0.get("message") or {}).get("content")
                finish = ch0.get("finish_reason")
                if content is None:
                    # A content filter returns 200 with a null message. Reporting that as an empty
                    # completion would let a filtered decision be scored as a schema failure, so it
                    # is raised as a transport error carrying the filter verdict.
                    self.cfg.content_filter_seen = ch0.get("content_filter_results") or \
                        obj.get("prompt_filter_results")
                    raise AzureTransportError(
                        f"null completion content (finish_reason={finish!r}, "
                        f"content_filter={redact(json.dumps(self.cfg.content_filter_seen)[:300])})")
                u = obj.get("usage") or {}
                usage = {"prompt_tokens": int(u.get("prompt_tokens") or 0),
                         "completion_tokens": int(u.get("completion_tokens") or 0),
                         "total_tokens": int(u.get("total_tokens") or 0)}
                rec.update(ok=True, http_status=status, latency_s=round(time.time() - t0, 3),
                           finish_reason=finish, response_model=str(obj.get("model") or ""),
                           response_id=str(obj.get("id") or ""),
                           system_fingerprint=obj.get("system_fingerprint"),
                           prompt_tokens=usage["prompt_tokens"],
                           completion_tokens=usage["completion_tokens"],
                           total_tokens=usage["total_tokens"],
                           content_filter=ch0.get("content_filter_results"),
                           prompt_filter=obj.get("prompt_filter_results"),
                           apim_request_id=headers.get("apim-request-id"),
                           ms_region=headers.get("x-ms-region"),
                           upstream_ms=headers.get("x-envoy-upstream-service-time"),
                           rl_remaining_requests=headers.get("x-ratelimit-remaining-requests"),
                           rl_remaining_tokens=headers.get("x-ratelimit-remaining-tokens"),
                           retry_after=headers.get("retry-after"))
                self.log.write(rec)
                return content, usage, k + 1

            except AzureConfigError:
                rec.update(ok=False, http_status=None, latency_s=round(time.time() - t0, 3),
                           error_class="AzureConfigError", error=redact(sys.exc_info()[1]),
                           retryable=False)
                self.log.write(rec)
                raise

            except urllib.error.HTTPError as exc:
                try:
                    detail = exc.read().decode("utf-8", "replace")[:1000]
                except Exception:                              # noqa: BLE001
                    detail = ""
                hard = exc.code in HARD_HTTP
                rec.update(ok=False, http_status=int(exc.code), reason=str(exc.reason),
                           latency_s=round(time.time() - t0, 3),
                           error_class="HTTPError", error=redact(detail),
                           retry_after=(exc.headers or {}).get("retry-after") if exc.headers else None,
                           retryable=not hard)
                self.log.write(rec)
                last_err = AzureTransportError(
                    f"HTTP {exc.code} {exc.reason}: {redact(detail)}")
                if hard:
                    # "non-retryable" is the sentinel balance9_runner.transport_call looks for; the
                    # phrase is preserved so the frozen failure semantics are unchanged.
                    raise AzureTransportError(
                        f"non-retryable HTTP {exc.code} {exc.reason}: {redact(detail)}") from None

            except Exception as exc:                           # noqa: BLE001
                rec.update(ok=False, http_status=None, latency_s=round(time.time() - t0, 3),
                           error_class=type(exc).__name__, error=redact(exc), retryable=True)
                self.log.write(rec)
                last_err = exc

            if k < TRANSPORT_RETRIES:
                time.sleep(min(BACKOFF_CAP_S, 1.0 * (2.0 ** k)) * (0.5 + random.random()))

        # Exhausted the retry budget on retryable errors. This is ordinary flakiness, so it is
        # reported with the frozen exception type and the frozen wording, which keeps the driver's
        # whole-seed-rerun policy (section 5 rule 5) working exactly as it did on OpenRouter.
        raise _technical_failure(
            f"transport failed after {TRANSPORT_RETRIES + 1} attempts: {redact(last_err)}")

    def stats(self) -> dict:
        with self._stats_lock:
            return dict(calls=self.calls, transport_attempts=self.attempts,
                        transport_retries=self.retries, attempt_records=self.log.n)
