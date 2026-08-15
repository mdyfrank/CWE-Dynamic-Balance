"""ha_transport.py -- the network boundary, and the only file in the package that touches it.

Two models are run against a backend the collaborator supplies. The backend may be Azure OpenAI,
Azure AI Foundry / model inference, or any OpenAI-compatible endpoint. Which one is decided by
configuration, recorded in the manifest, and stamped on every attempt record.

WHAT THIS MODULE REFUSES TO DO
------------------------------
  * IT NEVER GUESSES THE ROUTE. The three backends expose incompatible URL shapes. Trying one and
    falling back on a 404 is the kind of helpfulness that ends with nobody knowing which service
    produced the data. The route is decided once from an explicit setting or from the endpoint host,
    and an endpoint whose route cannot be decided is a startup error, not a coin flip.

  * IT NEVER ACCEPTS A DIFFERENT MODEL. Each alias declares the tokens its served model name must
    contain and the tokens that disqualify it. A quantised or community rebuild of Llama-3.3-70B is a
    different measurement instrument; substituting one silently would make a two-model comparison a
    comparison of unknowns. The check runs on the first live response, per alias, before any raw
    record for that alias exists, and the run aborts on mismatch. It can be overridden only by naming
    the exact accepted string, which is then recorded as an override in the manifest.

  * IT NEVER PRINTS, LOGS OR HASHES A CREDENTIAL. Every string that can reach a log, a record or a
    traceback goes through `redact` first. A hash is not a redaction -- a hash of a secret is a
    verifier for that secret -- so credentials are not hashed either.

  * IT NEVER CONFLATES A TRANSPORT RETRY WITH AN ECONOMIC ONE. This module counts HTTP attempts.
    Schema repairs are counted by the runner. The offered reconsideration turn in arm A3 is an
    experimental treatment counted separately again. Three counters, three meanings; merging any two
    of them would turn an infrastructure artefact into a finding.

CONFIGURATION -- environment variables only, never a committed file
------------------------------------------------------------------
  HA_ENDPOINT              https://... base endpoint (required)
  HA_API_KEY               API key (optional; falls back to DefaultAzureCredential on Azure routes)
  HA_ROUTE                 azure-openai | azure-ai-model-inference | openai-compatible (optional)
  HA_API_VERSION           Azure api-version (default 2024-05-01-preview)
  HA_DEPLOYMENT_GEMMA      deployment / model id serving Gemma 3 27B
  HA_DEPLOYMENT_LLAMA      deployment / model id serving Llama 3.3 70B
  HA_ENDPOINT_GEMMA        per-alias endpoint override (optional; some tenants split deployments)
  HA_ENDPOINT_LLAMA        per-alias endpoint override (optional)
  HA_ALLOW_MODEL_GEMMA     exact served model name to accept instead of the token check (optional)
  HA_ALLOW_MODEL_LLAMA     same, for llama (optional)
  HA_TIMEOUT_S             per-request timeout, default 180
  HA_TOKEN_SCOPE           AAD scope when using DefaultAzureCredential

No key, endpoint, deployment name or model credential is hard-coded anywhere in this package, and
`public()` is structurally incapable of emitting one: it builds a fresh dict of named fields rather
than filtering a dict that contains the secret.
"""

from __future__ import annotations

import hashlib
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

TRANSPORT_RETRIES = 6                 # 7 attempts total; same shape as the frozen OpenRouter pipeline
BACKOFF_CAP_S = 30.0
DEFAULT_API_VERSION = "2024-05-01-preview"
ROUTES = ("azure-openai", "azure-ai-model-inference", "openai-compatible")

# Statuses that must NOT be retried. Retrying an auth or quota failure burns the budget and buries the
# real cause under six identical lines. 401/402 are exactly how the earlier OpenRouter run died, so
# they are named rather than left to a generic 4xx rule.
HARD_HTTP = {400, 401, 402, 403, 404, 405, 501}

MODEL_SPECS = {
    "gemma": {"expected": "gemma-3-27b-it",
              "require": ("gemma", "27b"),
              "env_deployment": "HA_DEPLOYMENT_GEMMA",
              "env_endpoint": "HA_ENDPOINT_GEMMA",
              "env_allow": "HA_ALLOW_MODEL_GEMMA"},
    "llama": {"expected": "Llama-3.3-70B-Instruct",
              "require": ("llama", "3.3", "70b"),
              "env_deployment": "HA_DEPLOYMENT_LLAMA",
              "env_endpoint": "HA_ENDPOINT_LLAMA",
              "env_allow": "HA_ALLOW_MODEL_LLAMA"},
}

FORBIDDEN_MODEL_TOKENS = ("awq", "gptq", "gguf", "int4", "int8", "fp8", "-q4", "-q8",
                          "bnb", "nf4", "quant", "turbo", "draft", "speculative")


class TransportError(RuntimeError):
    """A transport failure. Message is already redacted."""


class ConfigError(RuntimeError):
    """A configuration failure. Raised at startup or on the first live response, never guessed
    around, and always fatal to the run."""


class ModelIdentityError(ConfigError):
    """The endpoint served a model that is not the one the manifest claims."""


# ==================================================================================================
# redaction
# ==================================================================================================
_SECRET_PATTERNS = [
    re.compile(r"(?i)\b(api[-_ ]?key|authorization|bearer|ocp-apim-subscription-key)\b\s*[:=]\s*\S+"),
    re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}"),   # JWT
    re.compile(r"\bsk-[A-Za-z0-9\-_]{16,}"),
    re.compile(r"\b[0-9a-f]{32}\b"),
    re.compile(r"\b[A-Za-z0-9]{60,}\b"),
]
_LIVE_SECRETS: list = []


def register_secret(value: str) -> None:
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


# ==================================================================================================
# per-decision context
# ==================================================================================================
CTX: ContextVar = ContextVar("ha_decision_ctx", default={})


def set_context(**kw) -> None:
    CTX.set(dict(kw))


def get_context() -> dict:
    try:
        return dict(CTX.get())
    except LookupError:
        return {}


# ==================================================================================================
# configuration
# ==================================================================================================
def decide_route(endpoint: str, forced: str = "") -> str:
    f = (forced or os.environ.get("HA_ROUTE", "")).strip().lower()
    if f:
        if f not in ROUTES:
            raise ConfigError(f"HA_ROUTE={f!r} is not one of {ROUTES}")
        return f
    host = (urlparse(endpoint).hostname or "").lower()
    path = (urlparse(endpoint).path or "").lower()
    if "openai.azure.com" in host or "/openai" in path:
        return "azure-openai"
    if any(h in host for h in ("services.ai.azure.com", "inference.ai.azure.com",
                               "models.ai.azure.com", "cognitiveservices.azure.com")):
        return "azure-ai-model-inference"
    if "openai.com" in host:
        return "openai-compatible"
    raise ConfigError(
        f"cannot decide the backend route from endpoint host {host!r}. This is deliberate: guessing "
        f"between the {len(ROUTES)} incompatible URL shapes would make the manifest a fiction. Set "
        f"HA_ROUTE to one of {ROUTES}.")


class ModelEndpoint:
    """One alias bound to one deployment on one endpoint."""

    def __init__(self, alias: str, endpoint: str, deployment: str, route: str, api_version: str,
                 auth_mode: str, expected: str, require: tuple, allow_name: str = None):
        self.alias = alias
        self.endpoint = endpoint.rstrip("/")
        self.deployment = deployment
        self.route = route
        self.api_version = api_version
        self.auth_mode = auth_mode
        self.expected = expected
        self.require = require
        self.allow_name = allow_name
        self.host = urlparse(self.endpoint).hostname or ""
        self.observed_model = None
        self.region = None
        self._token = None
        self._token_exp = 0.0
        self._cred = None
        self._lock = threading.Lock()

    def url(self) -> str:
        p = urlparse(self.endpoint)
        base = p.path.rstrip("/")
        if self.route == "azure-openai":
            path = f"{base}/openai/deployments/{self.deployment}/chat/completions"
            return urlunparse((p.scheme, p.netloc, path, "", f"api-version={self.api_version}", ""))
        if self.route == "azure-ai-model-inference":
            path = base if base.endswith("/chat/completions") else f"{base}/chat/completions"
            return urlunparse((p.scheme, p.netloc, path, "", f"api-version={self.api_version}", ""))
        path = base if base.endswith("/chat/completions") else f"{base}/chat/completions"
        return urlunparse((p.scheme, p.netloc, path, "", "", ""))

    def _bearer(self) -> str:
        with self._lock:
            if self._token and time.time() < self._token_exp - 300:
                return self._token
            if self._cred is None:
                try:
                    from azure.identity import DefaultAzureCredential      # noqa: PLC0415
                except ImportError as exc:
                    raise ConfigError(
                        "HA_API_KEY is not set and azure-identity is not installed; either set the "
                        "key or `pip install azure-identity`") from exc
                self._cred = DefaultAzureCredential()
            scope = os.environ.get("HA_TOKEN_SCOPE",
                                   "https://cognitiveservices.azure.com/.default")
            tok = self._cred.get_token(scope)
            self._token, self._token_exp = tok.token, float(tok.expires_on)
            register_secret(self._token)
            return self._token

    def headers(self) -> dict:
        h = {"Content-Type": "application/json", "User-Agent": "cwe-hidden-action/1.0"}
        if self.auth_mode == "api-key":
            key = os.environ["HA_API_KEY"].strip()
            if self.route == "azure-openai":
                h["api-key"] = key
            elif self.route == "azure-ai-model-inference":
                h["Authorization"] = f"Bearer {key}"
                h["api-key"] = key
            else:
                h["Authorization"] = f"Bearer {key}"
        else:
            h["Authorization"] = f"Bearer {self._bearer()}"
        if self.route == "azure-ai-model-inference":
            h["extra-parameters"] = "pass-through"
        return h

    def public(self) -> dict:
        """Built field by field. A credential cannot appear here by omission of a filter."""
        p = urlparse(self.endpoint)
        return {"alias": self.alias, "endpoint_host": self.host,
                "endpoint": urlunparse((p.scheme, p.netloc, p.path, "", "", "")),
                "deployment": self.deployment, "route": self.route,
                "api_version": self.api_version if self.route != "openai-compatible" else None,
                "request_url": self.url(), "auth_mode": self.auth_mode,
                "expected_model": self.expected, "required_tokens": list(self.require),
                "model_name_override": self.allow_name,
                "observed_model": self.observed_model, "observed_region": self.region}


def load_endpoints(aliases=("gemma", "llama")) -> dict:
    base_ep = os.environ.get("HA_ENDPOINT", "").strip().rstrip("/")
    key = os.environ.get("HA_API_KEY", "").strip()
    if key:
        register_secret(key)
    api_version = os.environ.get("HA_API_VERSION", "").strip() or DEFAULT_API_VERSION
    out, missing = {}, []
    for alias in aliases:
        spec = MODEL_SPECS[alias]
        ep = os.environ.get(spec["env_endpoint"], "").strip().rstrip("/") or base_ep
        dep = os.environ.get(spec["env_deployment"], "").strip()
        if not ep:
            missing.append(f"HA_ENDPOINT (or {spec['env_endpoint']}) for alias {alias!r}")
            continue
        if not ep.startswith("https://"):
            raise ConfigError(f"endpoint for alias {alias!r} must be an https:// URL")
        if not dep:
            missing.append(f"{spec['env_deployment']} for alias {alias!r}")
            continue
        route = decide_route(ep)
        auth = "api-key" if key else "default-azure-credential"
        if auth != "api-key" and route == "openai-compatible":
            raise ConfigError(
                "an OpenAI-compatible endpoint needs HA_API_KEY; DefaultAzureCredential is only "
                "meaningful on the Azure routes")
        out[alias] = ModelEndpoint(
            alias=alias, endpoint=ep, deployment=dep, route=route, api_version=api_version,
            auth_mode=auth, expected=spec["expected"], require=spec["require"],
            allow_name=os.environ.get(spec["env_allow"], "").strip() or None)
    if missing:
        raise ConfigError(
            "missing backend configuration: " + "; ".join(missing) +
            ". Nothing is hard-coded and nothing is guessed -- see the README for the exact "
            "environment variables. No API call has been made.")
    return out


# ==================================================================================================
# model identity
# ==================================================================================================
def check_model_identity(ep: ModelEndpoint, observed: str) -> dict:
    obs = (observed or "").strip()
    low = obs.lower()
    v = {"alias": ep.alias, "observed": obs, "expected": ep.expected,
         "override": ep.allow_name, "accepted": False, "reason": ""}
    if ep.allow_name:
        v["accepted"] = (obs == ep.allow_name)
        v["reason"] = ("explicit override matched" if v["accepted"] else
                       f"override {ep.allow_name!r} does not equal the observed model {obs!r}")
        return v
    bad = [t for t in FORBIDDEN_MODEL_TOKENS if t in low]
    if bad:
        v["reason"] = (f"the served model name {obs!r} contains {bad}, which marks a quantised or "
                       "derived build; that is a different instrument")
        return v
    miss = [t for t in ep.require if t not in low]
    if miss:
        v["reason"] = (f"the served model name {obs!r} is missing {miss}; expected {ep.expected!r}. "
                       f"If your deployment legitimately reports a custom name, set "
                       f"{MODEL_SPECS[ep.alias]['env_allow']} to that exact string.")
        return v
    v["accepted"] = True
    v["reason"] = f"served model name matches {ep.expected}"
    return v


# ==================================================================================================
# attempt log
# ==================================================================================================
class AttemptLog:
    """Append-only, fsync'd, one JSON object per transport attempt."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.n = 0

    def write(self, rec: dict) -> None:
        line = json.dumps(rec, ensure_ascii=False, default=str) + "\n"
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line)
                fh.flush()
                os.fsync(fh.fileno())
            self.n += 1


# ==================================================================================================
# the call
# ==================================================================================================
_SSL_CTX = ssl.create_default_context()


def _post(ep: ModelEndpoint, body: dict, timeout: float):
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(ep.url(), data=data, headers=ep.headers(), method="POST")
    with urllib.request.urlopen(req, timeout=timeout, context=_SSL_CTX) as resp:
        payload = resp.read().decode("utf-8", "replace")
        return resp.status, dict(resp.headers), json.loads(payload)


class Transport:
    """`transport(alias, messages) -> (content, usage, n_attempts)`."""

    def __init__(self, endpoints: dict, attempt_log: AttemptLog, on_identity=None,
                 http_hook=None, timeout_s: float = None):
        self.endpoints = endpoints
        self.log = attempt_log
        self.on_identity = on_identity
        self.http_hook = http_hook        # offline tests and mock mode inject a fake HTTP layer here
        self.timeout_s = float(timeout_s if timeout_s is not None
                               else os.environ.get("HA_TIMEOUT_S", "180"))
        self._identity_done = {}
        self._identity_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self.calls = 0
        self.attempts = 0
        self.retries = 0
        self.tokens_in = 0
        self.tokens_out = 0

    # -- identity, once per alias, before any raw record for that alias exists ------------------
    def _verify_identity(self, ep: ModelEndpoint, obj, headers):
        if self._identity_done.get(ep.alias):
            return
        with self._identity_lock:
            if self._identity_done.get(ep.alias):
                return
            observed = str(obj.get("model") or "")
            ep.observed_model = observed
            ep.region = (headers.get("x-ms-region") or headers.get("azureml-model-deployment")
                         or headers.get("x-ms-deployment-name") or None)
            verdict = check_model_identity(ep, observed)
            if self.on_identity:
                self.on_identity(verdict)
            self._identity_done[ep.alias] = True
            if not verdict["accepted"]:
                raise ModelIdentityError(
                    f"MODEL IDENTITY CHECK FAILED for alias {ep.alias!r} -- no raw record has been "
                    f"written for it. {verdict['reason']}")

    def body(self, ep: ModelEndpoint, msgs) -> dict:
        """Temperature 0, strict JSON object, no other sampling parameter."""
        return {"messages": msgs, "temperature": 0, "model": ep.deployment,
                "response_format": {"type": "json_object"}}

    def __call__(self, alias: str, msgs) -> tuple:
        ep = self.endpoints[alias]
        ctx = get_context()
        call_uid = uuid.uuid4().hex[:16]
        body = self.body(ep, msgs)
        last_err = None
        with self._stats_lock:
            self.calls += 1

        for k in range(TRANSPORT_RETRIES + 1):
            with self._stats_lock:
                self.attempts += 1
                if k:
                    self.retries += 1
            t0 = time.time()
            rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z",
                   "call_uid": call_uid, "transport_attempt": k + 1,
                   "run_key": ctx.get("run_key"), "decision_key": ctx.get("decision_key"),
                   "seed": ctx.get("seed"), "arm": ctx.get("arm"), "policy": ctx.get("policy"),
                   "model_alias": alias, "round": ctx.get("round"), "j": ctx.get("j"),
                   "attempt_kind": ctx.get("attempt_kind"),
                   "deployment": ep.deployment, "route": ep.route,
                   "endpoint_host": ep.host, "auth_mode": ep.auth_mode,
                   "n_messages": len(msgs)}
            try:
                if self.http_hook is not None:
                    status, headers, obj = self.http_hook(ep, body, k)
                else:
                    status, headers, obj = _post(ep, body, self.timeout_s)
                self._verify_identity(ep, obj, headers)
                choices = obj.get("choices") or []
                if not choices:
                    raise TransportError(
                        f"response contained no choices: {redact(json.dumps(obj)[:300])}")
                ch0 = choices[0]
                content = (ch0.get("message") or {}).get("content")
                finish = ch0.get("finish_reason")
                if content is None:
                    # A content filter returns 200 with a null message. Reporting that as an empty
                    # completion would let a filtered decision be scored as a schema failure.
                    raise TransportError(
                        f"null completion content (finish_reason={finish!r}, content_filter="
                        f"{redact(json.dumps(ch0.get('content_filter_results'))[:300])})")
                u = obj.get("usage") or {}
                usage = {"prompt_tokens": int(u.get("prompt_tokens") or 0),
                         "completion_tokens": int(u.get("completion_tokens") or 0),
                         "total_tokens": int(u.get("total_tokens") or 0)}
                with self._stats_lock:
                    self.tokens_in += usage["prompt_tokens"]
                    self.tokens_out += usage["completion_tokens"]
                rec.update(ok=True, http_status=status, latency_s=round(time.time() - t0, 3),
                           finish_reason=finish, response_model=str(obj.get("model") or ""),
                           response_id=str(obj.get("id") or ""),
                           prompt_tokens=usage["prompt_tokens"],
                           completion_tokens=usage["completion_tokens"],
                           total_tokens=usage["total_tokens"],
                           ms_region=headers.get("x-ms-region"),
                           apim_request_id=headers.get("apim-request-id"),
                           rl_remaining_requests=headers.get("x-ratelimit-remaining-requests"),
                           rl_remaining_tokens=headers.get("x-ratelimit-remaining-tokens"),
                           retry_after=headers.get("retry-after"))
                self.log.write(rec)
                return content, usage, k + 1

            except ConfigError:
                rec.update(ok=False, http_status=None, latency_s=round(time.time() - t0, 3),
                           error_class=type(sys.exc_info()[1]).__name__,
                           error=redact(sys.exc_info()[1]), retryable=False)
                self.log.write(rec)
                raise

            except urllib.error.HTTPError as exc:
                try:
                    detail = exc.read().decode("utf-8", "replace")[:1000]
                except Exception:                                     # noqa: BLE001
                    detail = ""
                hard = exc.code in HARD_HTTP
                rec.update(ok=False, http_status=int(exc.code), reason=str(exc.reason),
                           latency_s=round(time.time() - t0, 3), error_class="HTTPError",
                           error=redact(detail), retryable=not hard,
                           retry_after=(exc.headers or {}).get("retry-after") if exc.headers else None)
                self.log.write(rec)
                if hard:
                    raise TransportError(
                        f"non-retryable HTTP {exc.code} {exc.reason}: {redact(detail)}") from None
                last_err = TransportError(f"HTTP {exc.code} {exc.reason}: {redact(detail)}")

            except Exception as exc:                                  # noqa: BLE001
                rec.update(ok=False, http_status=None, latency_s=round(time.time() - t0, 3),
                           error_class=type(exc).__name__, error=redact(exc), retryable=True)
                self.log.write(rec)
                last_err = exc

            if k < TRANSPORT_RETRIES:
                time.sleep(min(BACKOFF_CAP_S, 1.0 * (2.0 ** k)) * (0.5 + random.random()))

        raise TransportError(
            f"transport failed after {TRANSPORT_RETRIES + 1} attempts: {redact(last_err)}")

    def stats(self) -> dict:
        with self._stats_lock:
            return {"calls": self.calls, "transport_attempts": self.attempts,
                    "transport_retries": self.retries, "attempt_records": self.log.n,
                    "prompt_tokens": self.tokens_in, "completion_tokens": self.tokens_out}

    def public(self) -> dict:
        return {"transport_retries": TRANSPORT_RETRIES, "hard_http": sorted(HARD_HTTP),
                "timeout_s": self.timeout_s,
                "endpoints": {a: e.public() for a, e in self.endpoints.items()}}


# ==================================================================================================
# mock backend -- no network, no key, deterministic
# ==================================================================================================
class MockEndpoint(ModelEndpoint):
    def __init__(self, alias):
        spec = MODEL_SPECS[alias]
        super().__init__(alias=alias, endpoint="https://mock.invalid", deployment=f"mock-{alias}",
                         route="openai-compatible", api_version="mock", auth_mode="mock",
                         expected=spec["expected"], require=spec["require"],
                         allow_name=spec["expected"])

    def headers(self):
        return {"Content-Type": "application/json"}


def mock_endpoints(aliases=("gemma", "llama")) -> dict:
    return {a: MockEndpoint(a) for a in aliases}


def make_mock_hook(behaviour="rational", fail_rate: float = 0.0, seed: int = 0):
    """A fake HTTP layer.

    The mock merchant is NOT a model of an LLM and is not evidence about one. It exists to exercise
    the pipeline end to end -- resume, atomicity, schema repair, accounting, the analyser and the
    validator -- with no key and no network, so that everything except the model's judgement is
    already known to work before a single paid call is made.

    `behaviour`:
      rational    reads the published penalty from the prompt and picks a plausible response to it
      honest      always index 0
      greedy      always index 20
      noisy       uniform over the grid
      malformed   returns unparseable text on the first attempt of each decision, then valid JSON,
                  so the schema-repair path is exercised by every offline run
    """
    rng = random.Random(seed)
    state = {"seen": {}}

    def hook(ep, body, attempt_k):
        sysmsg = body["messages"][0]["content"]
        usermsg = body["messages"][-1]["content"]
        if fail_rate and rng.random() < fail_rate and attempt_k < 2:
            raise urllib.error.HTTPError(ep.url(), 503, "mock unavailable", {}, None)
        # Read the economic payload from the last user message that parses as a JSON object, not
        # from the last message outright. A schema-repair turn appends a plain-text instruction, so
        # `messages[-1]` carries no payload; a mock that read only that would fall back to its
        # defaults and answer a different question. Under the previous version kappa defaulted to
        # zero on every repair and the mock returned the maximum action, so `malformed` did not
        # merely exercise the repair path -- it changed the answers, and the four arms diverged for
        # reasons that had nothing to do with what the arms show a merchant. A repair carries no new
        # economic information, so the reply after a repair must be the reply the payload implies.
        u, payload = {}, None
        for msg in reversed(body["messages"]):
            if msg.get("role") != "user":
                continue
            try:
                cand = json.loads(msg["content"])
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(cand, dict) and cand:
                u, payload = cand, msg["content"]
                break
        kappa = float(u.get("published_penalty_kappa", 0.0) or 0.0)
        tau = float(u.get("published_penalty_tau", 0.2) or 0.2)
        b = float(u.get("your_baseline_adverse_rate_b", 0.05) or 0.05)
        if behaviour == "honest":
            idx = 0
        elif behaviour == "greedy":
            idx = 20
        elif behaviour == "noisy":
            idx = rng.randint(0, 20)
        else:
            # "rational": aim just under the published threshold when the penalty bites, else high.
            # cs = 0.6 in the frozen spec; f such that b + 0.6 f ~ tau.
            if kappa <= 0.0:
                idx = 20
            else:
                f_star = max(0.0, (tau - b) / 0.6)
                idx = int(round(min(1.0, f_star) * 20))
                if kappa < 1.0:
                    idx = min(20, idx + 6)
        # Fire the malformed reply once per DISTINCT decision, and never on a repair prompt.
        # Keying on a prefix does not work: the system block is byte-identical across arms by
        # design and the payload's opening keys are fixed, so prefixes collide and the malformed
        # branch fires on whichever arm happens to have a long, varying prefix (A0, which carries
        # the payoff table) and on no other -- leaving the schema-repair path untested on exactly
        # the three arms the study is about. A repair prompt is plain text rather than a JSON
        # payload, which is what `is_fresh` detects; without that guard the repair would itself be
        # malformed, the budget would drain, and every cell would fail.
        is_fresh = payload is not None and usermsg == payload
        key = hashlib.sha256((sysmsg + "\x00" + (payload or usermsg)).encode("utf-8")).hexdigest()
        n_seen = state["seen"].get(key, 0)
        state["seen"][key] = n_seen + 1
        if behaviour == "malformed" and is_fresh and n_seen == 0:
            content = "I would choose a moderate level of exaggeration."
        else:
            content = json.dumps({"exaggeration_index": int(idx), "reasoning": "mock"})
        ntok = max(1, len(sysmsg) // 4)
        return 200, {"x-ms-region": "mock"}, {
            "id": f"mock-{uuid.uuid4().hex[:8]}", "model": ep.expected,
            "choices": [{"message": {"role": "assistant", "content": content},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": ntok, "completion_tokens": 12, "total_tokens": ntok + 12}}

    return hook


# ==================================================================================================
def selftest() -> dict:
    """Everything checkable without a network: routing, redaction, identity, mock round trip."""
    out = {}

    routes = {}
    for host, expect in [("https://x.openai.azure.com", "azure-openai"),
                         ("https://x.services.ai.azure.com/models", "azure-ai-model-inference"),
                         ("https://api.openai.com/v1", "openai-compatible")]:
        routes[host] = decide_route(host)
    try:
        decide_route("https://example.com/v1")
        routes["undecidable"] = "GUESSED -- BUG"
    except ConfigError:
        routes["undecidable"] = "raised ConfigError (correct)"
    out["routing"] = {"rows": routes,
                      "pass": all(routes[h] == e for h, e in
                                  [("https://x.openai.azure.com", "azure-openai"),
                                   ("https://x.services.ai.azure.com/models",
                                    "azure-ai-model-inference"),
                                   ("https://api.openai.com/v1", "openai-compatible")])
                      and routes["undecidable"].startswith("raised")}

    register_secret("abcdef0123456789abcdef0123456789")
    probe = ("key=abcdef0123456789abcdef0123456789 and api-key: SUPERSECRETVALUE and "
             "sk-0123456789abcdefghij")
    red = redact(probe)
    out["redaction"] = {"pass": "abcdef0123456789abcdef0123456789" not in red
                        and "SUPERSECRETVALUE" not in red and "sk-0123456789abcdefghij" not in red,
                        "redacted": red}

    ep = MockEndpoint("llama")
    ep.allow_name = None
    cases = {"Llama-3.3-70B-Instruct": True, "llama-3.3-70b-instruct-awq": False,
             "Meta-Llama-3.1-8B-Instruct": False, "gpt-4o": False}
    ident = {k: check_model_identity(ep, k)["accepted"] for k in cases}
    out["model_identity"] = {"pass": ident == cases, "rows": ident}

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        log = AttemptLog(Path(td) / "attempts.jsonl")
        tr = Transport(mock_endpoints(), log, http_hook=make_mock_hook("rational"))
        msgs = [{"role": "system", "content": "rules"},
                {"role": "user", "content": json.dumps(
                    {"published_penalty_kappa": 4.0, "published_penalty_tau": 0.2,
                     "your_baseline_adverse_rate_b": 0.05})}]
        c1, u1, k1 = tr("gemma", msgs)
        c2, u2, k2 = tr("llama", msgs)
        st = tr.stats()
        out["mock_roundtrip"] = {
            "pass": (json.loads(c1)["exaggeration_index"] == 5 and k1 == 1 and k2 == 1
                     and st["calls"] == 2 and st["attempt_records"] == 2
                     and u1["total_tokens"] > 0),
            "content": c1, "stats": st}

        log2 = AttemptLog(Path(td) / "attempts2.jsonl")
        tr2 = Transport(mock_endpoints(), log2,
                        http_hook=make_mock_hook("rational", fail_rate=1.0, seed=1))
        _, _, k3 = tr2("gemma", msgs)
        out["retry_path"] = {"pass": k3 >= 2, "attempts_used": k3,
                             "note": "a 503 is retried; the attempt counter is the TRANSPORT "
                                     "counter and is never merged with schema repairs or with the "
                                     "A3 reconsideration turn"}

    out["all_pass"] = all(v["pass"] for v in out.values() if isinstance(v, dict) and "pass" in v)
    return out


if __name__ == "__main__":
    r = selftest()
    for k, v in r.items():
        if isinstance(v, dict) and "pass" in v:
            print(f"[{'PASS' if v['pass'] else 'FAIL'}] {k}")
    print("all_pass:", r["all_pass"])
    if not r["all_pass"]:
        print(json.dumps(r, indent=2, default=str)[:3000])
        sys.exit(1)
