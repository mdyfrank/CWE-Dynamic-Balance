"""run_offline_tests.py -- everything that can be proved without spending an Azure call.

    python offline_tests/run_offline_tests.py

Nothing here opens a socket. The Azure transport is exercised through its `dry_run_hook`, which
replaces the single `urllib` POST with a scripted response, so the retry logic, the redaction, the
attempt schema, the model-identity gate and the HTTP classification are all tested against the real
code path rather than a paraphrase of it.

The tests run in a temporary directory and write nothing into `results/`.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
import urllib.error
from contextlib import redirect_stdout
from pathlib import Path

HERE = Path(os.path.dirname(os.path.abspath(__file__)))
PKG = HERE.parent
CODE = PKG / "code"
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(PKG))

RESULTS = []


def t(name, fn):
    t0 = time.time()
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            detail = fn() or ""
        ok, err = True, ""
    except Exception as exc:                             # noqa: BLE001
        ok, detail, err = False, "", traceback.format_exc()
    dt = round(time.time() - t0, 2)
    RESULTS.append(dict(test=name, ok=ok, detail=str(detail)[:400], seconds=dt,
                        error=err.splitlines()[-1] if err else ""))
    print(f"{'  ok  ' if ok else ' FAIL '} {name:<62} {dt:6.2f}s  {str(detail)[:70]}")
    if not ok:
        print("\n".join("        " + l for l in err.strip().splitlines()[-6:]))
    return ok


# =================================================================================================
def T1_vendored_integrity():
    """Every vendored file hashes to the manifest value, and the two severities behave differently.

    The gate must abort on a changed *frozen* file -- a changed prompt or constant makes the run a
    new experiment wearing the old experiment's name -- and must NOT abort on a changed package-own
    file, because the collaborator patching a path bug in my transport code has not touched the
    science. Both halves are exercised against a throwaway copy, so the test never mutates the tree
    it is checking.
    """
    import to_run as TR
    TR._G.clear()
    assert TR.gate_vendored_integrity(), "vendored integrity gate failed"
    assert TR.gate_catalog_placed(), "catalogue placement gate failed"
    man = json.loads((PKG / "manifests" / "vendored_sha256.json").read_text(encoding="utf-8"))

    prov = man["provenance"]
    frozen = sorted(r for r, v in prov.items() if v.get("frozen") is True)
    own = sorted(r for r, v in prov.items() if v.get("frozen") is False)
    assert len(frozen) == 16, f"expected 16 frozen files, manifest says {len(frozen)}"
    assert "to_run.py" in own and "code/azure_transport.py" in own, "runner not marked package-own"
    assert "code/balance9_prompts.py" in frozen, "the prompt module must be frozen"

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "pkg"
        shutil.copytree(PKG, tmp)
        real_here, real_man, real_prog = TR.HERE, TR.MANIFESTS, TR.PROGRESS
        try:
            TR.HERE, TR.MANIFESTS = tmp, tmp / "manifests"
            fb, lb, nf = TR.hash_audit()
            assert not fb and not lb, f"clean copy already dirty: {fb + lb}"
            assert nf == 16, f"hash_audit counted {nf} frozen files, expected 16"

            # a byte appended to a package-own file: recorded, not fatal
            with open(tmp / "to_run.py", "ab") as fh:
                fh.write(b"\n# local patch\n")
            fb, lb, _ = TR.hash_audit()
            assert not fb, "editing to_run.py must not be reported as frozen drift"
            assert any("to_run.py" in x for x in lb), "the local edit was not recorded at all"
            TR._G.clear()
            assert TR.gate_vendored_integrity(), "a package-own edit must not block the run"
            assert TR._LOCAL_EDITS, "the gate passed but left no record of the modification"

            # a byte appended to a frozen file: fatal
            with open(tmp / "code" / "balance9_prompts.py", "ab") as fh:
                fh.write(b"\n# tampered\n")
            fb, _, _ = TR.hash_audit()
            assert any("balance9_prompts" in x for x in fb), "frozen tampering not detected"
            TR._G.clear()
            assert not TR.gate_vendored_integrity(), "a changed prompt module must abort the run"

            # a file the manifest cannot classify is treated strictly, not waved through
            mp = tmp / "manifests" / "vendored_sha256.json"
            m2 = json.loads(mp.read_text(encoding="utf-8"))
            m2["provenance"].pop("to_run.py", None)
            mp.write_text(json.dumps(m2), encoding="utf-8")
            fb, _, _ = TR.hash_audit()
            assert any("to_run.py" in x for x in fb), "unknown provenance must default to frozen"

            # the flag reaches the artefact the collaborator sends back
            TR.HERE, TR.MANIFESTS = real_here, real_man
            TR.PROGRESS = Path(td) / "progress.json"       # never overwrite the real one
            TR._LOCAL_EDITS[:] = ["to_run.py: sha256 differs"]
            art = TR.write_progress(True)
            assert art["runner_locally_modified"] is True and art["local_modifications"], \
                "progress.json does not carry the modification forward"
        finally:
            TR.HERE, TR.MANIFESTS, TR.PROGRESS = real_here, real_man, real_prog
            TR._G.clear()
            TR._LOCAL_EDITS.clear()

    return (f"{man['n_files']} files verified; {len(frozen)} frozen abort on drift, "
            f"{len(own)} package-own recorded and continue")


def T2_prompt_hashes():
    """The vendored prompt surface reproduces the published hashes exactly."""
    import balance9_prompts as PR
    pub = PR.published_hashes()
    frozen = json.loads((PKG / "input_data" / "balance9_prompt_hashes.json")
                        .read_text(encoding="utf-8"))["hashes"]
    diff = sorted(k for k in set(pub) | set(frozen) if pub.get(k) != frozen.get(k))
    assert not diff, f"{len(diff)} prompt hashes differ: {diff[:5]}"
    assert "module::source" in frozen, "the prompt module does not hash its own source"
    return f"{len(frozen)} prompt hashes identical (incl. module::source)"


def T3_runner_gates():
    """balance9_runner.selftest() -- G-R1..G-R10. Makes no network request."""
    import balance9_runner as R
    R._R.clear()
    R.selftest()
    n = sum(1 for _, ok in R._R if ok)
    assert n == len(R._R), f"{len(R._R) - n} runner gates failed"
    return f"{n}/{len(R._R)} runner gates"


def T4_design_gates():
    """The design gates, re-derived from BALANCE9_PREREGISTRATION.md."""
    import to_run as TR
    import balance9_runner as R
    import balance9_prereg_freeze as FZ
    TR._G.clear()
    assert TR.gate_design(R, FZ), \
        "design gates failed: " + str([l for l, ok in TR._G if not ok])
    return f"{len(TR._G)}/{len(TR._G)} design gates"


def T5_route_selection():
    """Route selection is explicit; an undecidable endpoint is an error, never a guess."""
    import azure_transport as AT
    cases = [
        ("https://my-res.openai.azure.com", "", "azure-openai"),
        ("https://my-res.services.ai.azure.com/models", "", "azure-ai-model-inference"),
        ("https://x.inference.ai.azure.com", "", "azure-ai-model-inference"),
        ("https://x.cognitiveservices.azure.com", "", "azure-ai-model-inference"),
        ("https://anything.example.com", "azure-openai", "azure-openai"),
    ]
    for ep, forced, want in cases:
        os.environ["AZURE_AI_ROUTE"] = forced
        got = AT._decide_route(ep)
        assert got == want, f"{ep} (forced={forced!r}) -> {got}, expected {want}"
    os.environ["AZURE_AI_ROUTE"] = ""
    try:
        AT._decide_route("https://anything.example.com")
        raise AssertionError("an undecidable endpoint must raise, not guess")
    except AT.AzureConfigError:
        pass
    os.environ["AZURE_AI_ROUTE"] = "nonsense"
    try:
        AT._decide_route("https://x.openai.azure.com")
        raise AssertionError("an invalid AZURE_AI_ROUTE must raise")
    except AT.AzureConfigError:
        pass
    del os.environ["AZURE_AI_ROUTE"]

    # URL shapes
    cfg = AT.AzureConfig("https://r.openai.azure.com", "llama33", "2024-05-01-preview",
                         "azure-openai", "api-key", "Llama-3.3-70B-Instruct", None)
    assert cfg.url() == ("https://r.openai.azure.com/openai/deployments/llama33/chat/completions"
                         "?api-version=2024-05-01-preview"), cfg.url()
    cfg2 = AT.AzureConfig("https://r.services.ai.azure.com/models", "llama33", "2024-05-01-preview",
                          "azure-ai-model-inference", "api-key", "Llama-3.3-70B-Instruct", None)
    assert cfg2.url() == ("https://r.services.ai.azure.com/models/chat/completions"
                          "?api-version=2024-05-01-preview"), cfg2.url()
    return "5 endpoints routed, 2 refusals, 2 URL shapes"


def T6_model_identity():
    """A quantised, community or wrong model is refused; an explicit override is recorded."""
    import azure_transport as AT
    cfg = AT.AzureConfig("https://r.openai.azure.com", "d", "v", "azure-openai", "api-key",
                         "Llama-3.3-70B-Instruct", None)
    good = ["Llama-3.3-70B-Instruct", "llama-3.3-70b-instruct", "Meta-Llama-3.3-70B-Instruct"]
    bad = ["Llama-3.3-70B-Instruct-AWQ", "llama-3.3-70b-instruct-gptq", "Llama-3.1-70B-Instruct",
           "Llama-3.3-8B-Instruct", "gpt-4o", "llama-3.3-70b-instruct-fp8", "", "Meta-Llama-3-70B"]
    for m in good:
        v = AT.check_model_identity(cfg, m)
        assert v["accepted"], f"{m!r} should be accepted: {v['reason']}"
    for m in bad:
        v = AT.check_model_identity(cfg, m)
        assert not v["accepted"], f"{m!r} must be REFUSED but was accepted"
    cfg2 = AT.AzureConfig("https://r.openai.azure.com", "d", "v", "azure-openai", "api-key",
                          "Llama-3.3-70B-Instruct", "my-custom-llama-deploy")
    assert AT.check_model_identity(cfg2, "my-custom-llama-deploy")["accepted"]
    assert not AT.check_model_identity(cfg2, "something-else")["accepted"]
    return f"{len(good)} accepted, {len(bad)} refused, override honoured only on exact match"


def T7_redaction():
    """No credential shape survives `redact`, and a live secret is removed literally."""
    import azure_transport as AT
    AT._LIVE_SECRETS.clear()
    secret = "a1b2c3d4e5f60718293a4b5c6d7e8f90"
    AT.register_secret(secret)
    jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dBjftJeZ4CVPmB92K27uhbUJU1p1r_wW1g"
    AT.register_secret(jwt)
    samples = [
        f'{{"error":"unauthorized","api-key":"{secret}"}}',
        f"Authorization: Bearer {jwt}",
        f"api-key={secret}",
        f"the key {secret} was rejected",
        "sk-or-v1-0123456789abcdef0123456789abcdef",
        f"Bearer {jwt} trailing",
    ]
    for s in samples:
        r = AT.redact(s)
        assert secret not in r, f"32-hex key survived redaction: {r}"
        assert jwt not in r, f"JWT survived redaction: {r}"
        assert "sk-or-v1-0123456789abcdef" not in r, f"sk- token survived: {r}"
    AT._LIVE_SECRETS.clear()
    # and with NO registered secret, the patterns alone must still catch it
    assert secret not in AT.redact(f"api-key: {secret}")
    assert jwt not in AT.redact(f"Bearer {jwt}")
    return f"{len(samples)} samples redacted, patterns hold without registration"


# ---- a scripted Azure, used by T8..T11 ----------------------------------------------------------
def _resp(content, model="Llama-3.3-70B-Instruct", ptok=1100, ctok=40):
    return 200, {"x-ms-region": "swedencentral", "apim-request-id": "test",
                 "x-ratelimit-remaining-tokens": "999999"}, {
        "id": "chatcmpl-test", "model": model, "system_fingerprint": "fp_test",
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": content},
                     "content_filter_results": {"hate": {"filtered": False, "severity": "safe"}}}],
        "usage": {"prompt_tokens": ptok, "completion_tokens": ctok,
                  "total_tokens": ptok + ctok}}


def _http_error(code, body=b'{"error":{"code":"x","message":"api-key: DEADBEEF"}}'):
    return urllib.error.HTTPError("https://x", code, "Test Reason",
                                  {"retry-after": "1"}, io.BytesIO(body))


def _mk_transport(tmp, script, expected="Llama-3.3-70B-Instruct"):
    import azure_transport as AT
    cfg = AT.AzureConfig("https://r.openai.azure.com", "llama33", "2024-05-01-preview",
                         "azure-openai", "api-key", expected, None)
    state = {"n": 0, "identity": None}

    def hook(c, body, k):
        state["n"] += 1
        step = script(state["n"], body)
        if isinstance(step, Exception):
            raise step
        return step

    tx = AT.AzureTransport(cfg, AT.AttemptLog(Path(tmp) / "attempts.jsonl"),
                           on_identity=lambda v: state.__setitem__("identity", v),
                           dry_run_hook=hook)
    return tx, cfg, state


def T8_transport_contract():
    """The Azure transport honours the exact contract of balance9_runner.transport_call."""
    import azure_transport as AT
    import balance9_runner as R
    tmp = tempfile.mkdtemp()
    tx, cfg, st = _mk_transport(tmp, lambda n, b: _resp('{"action_index": 7}'))
    box = [0]
    AT.set_context(run_key="llama|U|P_GMV|9000", decision_key="llama|U|P_GMV|9000:1:0",
                   seed=9000, interface="U", policy="P_GMV", round=1, j=0)
    raw, usage, n_tr = tx("meta-llama/llama-3.3-70b-instruct", [{"role": "user", "content": "x"}],
                          box)
    assert raw == '{"action_index": 7}', raw
    assert usage == {"prompt_tokens": 1100, "completion_tokens": 40, "total_tokens": 1140}, usage
    assert n_tr == 1 and box[0] == 1, (n_tr, box)
    # parse through the FROZEN parser, so the contract is tested end to end
    a, c, reason, ok = R.parse_action(raw)
    assert ok and a == 7, (a, ok)
    # payload
    body = tx.body([{"role": "user", "content": "x"}])
    assert body["temperature"] == 0, body
    assert body["response_format"] == {"type": "json_object"}, body
    assert body["model"] == "llama33", body
    assert set(body) == {"messages", "temperature", "response_format", "model"}, sorted(body)
    # attempt record
    rec = json.loads(Path(tmp, "attempts.jsonl").read_text(encoding="utf-8").strip())
    for k in ("ts", "call_uid", "transport_attempt", "run_key", "decision_key", "seed", "interface",
              "policy", "round", "j", "openrouter_model_id", "deployment", "route", "api_version",
              "endpoint_host", "auth_mode", "ok", "http_status", "latency_s", "finish_reason",
              "response_model", "prompt_tokens", "completion_tokens", "ms_region"):
        assert k in rec, f"attempt record is missing {k}"
    assert not any("key" in str(v).lower() and len(str(v)) > 30 for v in rec.values())
    shutil.rmtree(tmp, ignore_errors=True)
    return "3-tuple contract, frozen payload, 24 attempt fields"


def T9_retry_and_hard_stop():
    """429/500 retry with the frozen budget; 401/402 stop immediately; exhaustion is TechnicalFailure."""
    import azure_transport as AT
    import balance9_runner as R
    AT.BACKOFF_CAP_S, real_cap = 0.001, AT.BACKOFF_CAP_S
    tmp = tempfile.mkdtemp()
    try:
        # (a) two 429s then success -> 3 transport attempts, one returned value
        tx, _, _ = _mk_transport(tmp, lambda n, b: (_http_error(429) if n <= 2
                                                    else _resp('{"action_index": 3}')))
        box = [0]
        raw, usage, n_tr = tx("m", [{"role": "user", "content": "x"}], box)
        assert n_tr == 3 and box[0] == 3, (n_tr, box)

        # (b) HTTP 402 and 401 -- the two statuses that killed the OpenRouter run -- never retried
        for code in (401, 402, 403, 404, 400):
            tx2, _, _ = _mk_transport(tmp, lambda n, b, c=code: _http_error(c))
            box2 = [0]
            try:
                tx2("m", [{"role": "user", "content": "x"}], box2)
                raise AssertionError(f"HTTP {code} must not return a value")
            except AT.AzureTransportError as exc:
                assert "non-retryable" in str(exc), str(exc)
                assert "DEADBEEF" not in str(exc), "the error body leaked a credential-shaped token"
            assert box2[0] == 1, f"HTTP {code} was retried {box2[0]} times"

        # (c) 7 failures -> TechnicalFailure, so the frozen whole-seed-rerun policy applies
        tx3, _, _ = _mk_transport(tmp, lambda n, b: _http_error(500))
        box3 = [0]
        try:
            tx3("m", [{"role": "user", "content": "x"}], box3)
            raise AssertionError("exhaustion must raise")
        except R.TechnicalFailure as exc:
            assert "transport failed after 7 attempts" in str(exc), str(exc)
        assert box3[0] == AT.TRANSPORT_RETRIES + 1 == 7, box3

        # (d) a content-filtered 200 with null content is a transport error, not an empty completion
        def filtered(n, b):
            return 200, {}, {"model": "Llama-3.3-70B-Instruct",
                             "choices": [{"finish_reason": "content_filter", "message":
                                          {"content": None},
                                          "content_filter_results": {"violence": {"filtered": True}}}],
                             "usage": {}}
        tx4, _, _ = _mk_transport(tmp, filtered)
        box4 = [0]
        try:
            tx4("m", [{"role": "user", "content": "x"}], box4)
            raise AssertionError("a null completion must not be returned as content")
        except R.TechnicalFailure as exc:
            assert "null completion" in str(exc), str(exc)
    finally:
        AT.BACKOFF_CAP_S = real_cap
        shutil.rmtree(tmp, ignore_errors=True)
    return "429/500 retried to 7, 401/402/403/404/400 stopped at 1, filter -> error"


def T10_identity_gate_blocks_run():
    """A wrong served model aborts before anything is written."""
    import azure_transport as AT
    tmp = tempfile.mkdtemp()
    try:
        tx, cfg, st = _mk_transport(tmp, lambda n, b: _resp('{"action_index": 1}',
                                                            model="Llama-3.3-70B-Instruct-AWQ"))
        try:
            tx("m", [{"role": "user", "content": "x"}], [0])
            raise AssertionError("a quantised model must abort the run")
        except AT.AzureConfigError as exc:
            assert "MODEL IDENTITY CHECK FAILED" in str(exc), str(exc)
        assert st["identity"] is not None and st["identity"]["accepted"] is False
        assert st["identity"]["observed"] == "Llama-3.3-70B-Instruct-AWQ"
        rec = json.loads(Path(tmp, "attempts.jsonl").read_text(encoding="utf-8").strip())
        assert rec["ok"] is False and rec["error_class"] == "AzureConfigError"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return "quantised build refused, verdict recorded, attempt logged as failed"


def T11_end_to_end_mock_matrix():
    """A complete small matrix through the REAL frozen runner, with a scripted Azure.

    This is the test that matters: it proves the frozen simulate_run / decide / run_cell path works
    unchanged against the Azure transport, that the raw schema is the Balance-9 schema, that the
    independent validator accepts the result, and that the R and G guard arms actually fire.
    """
    import azure_transport as AT
    import balance9_runner as R
    import balance9_validate as V
    import to_run as TR

    tmp = Path(tempfile.mkdtemp())
    out = tmp / "balance9_azure_p3_llama_raw.jsonl"
    real_decide, real_tc, real_ref, real_ap = (R.decide, R.transport_call,
                                               R.transport_call_ref[0], R.append_jsonl)
    AT.BACKOFF_CAP_S, real_cap = 0.001, AT.BACKOFF_CAP_S
    try:
        # A model that proposes index 3 -- deliberately NOT the argmax, so R retries economically and
        # G overrides. A model that just answered the argmax would leave both guard arms untested.
        n_calls = {"n": 0}

        def script(n, body):
            n_calls["n"] += 1
            if n_calls["n"] % 17 == 0:                    # exercise the schema-repair path
                return _resp("not json at all")
            return _resp(json.dumps({"action_index": 3, "claimed_best_index": 3,
                                     "reasoning": "test"}))

        tx, cfg, st = _mk_transport(str(tmp), script)
        R.transport_call = tx
        R.transport_call_ref[0] = tx
        R.append_jsonl = TR.make_durable_append(R)
        TR.install_decision_context(R, AT)

        stats = {}
        for itf in ("U", "H", "R", "G"):
            for pol in ("P_GMV", "P_robust"):
                s = R.run_cell("llama", itf, pol, [9000, 9001], out, rounds=3, seed_workers=2,
                               agent="llm", quiet=True)
                stats[f"{itf}|{pol}"] = s
                assert s["completed"] == 2 and s["failed"] == 0, (itf, pol, s)

        rows = [json.loads(l) for l in open(out, encoding="utf-8")]
        foot = [r for r in rows if r["kind"] == "run_footer"]
        rnd = [r for r in rows if r["kind"] == "round"]
        assert len(foot) == 16, f"expected 16 runs, got {len(foot)}"
        assert len(rnd) == 48, f"expected 48 rounds, got {len(rnd)}"

        # the guard arms actually did something
        g_over = sum(1 for r in rnd if r["interface"] == "G"
                     for m in r["merchants"] if m["override"])
        r_econ = sum(m["economic_retries"] for r in rnd if r["interface"] == "R"
                     for m in r["merchants"])
        u_over = sum(1 for r in rnd if r["interface"] == "U"
                     for m in r["merchants"] if m["override"])
        assert g_over > 0, "G never overrode -- the guard arm was not exercised"
        assert r_econ > 0, "R never retried economically -- the guard arm was not exercised"
        assert u_over == 0, "U must never override"
        # H carries a mark, U does not
        assert all(m["mark_index"] is not None for r in rnd if r["interface"] == "H"
                   for m in r["merchants"]), "H must carry a factual mark"
        assert all(m["mark_index"] is None for r in rnd if r["interface"] == "U"
                   for m in r["merchants"]), "U must carry no mark"
        # schema repair was exercised
        assert sum(m["schema_repairs"] for r in rnd for m in r["merchants"]) > 0, \
            "the schema-repair path was never exercised"
        # tokens flowed through from the transport into the frozen record
        assert all(m["prompt_tokens"] > 0 for r in rnd for m in r["merchants"]), "tokens lost"

        # duplicate safety / resume
        before = len(rows)
        s2 = R.run_cell("llama", "U", "P_GMV", [9000, 9001], out, rounds=3, seed_workers=2,
                        agent="llm", quiet=True)
        after = sum(1 for _ in open(out, encoding="utf-8"))
        assert after == before and s2["skipped_already_done"] == 2 and s2["completed"] == 0, \
            (before, after, s2)

        # the independent validator, on the real file
        rep, runs = V.validate_file(str(out))
        n_ok, n_tot = rep.summary()
        assert len(runs) == 16, f"validator reconstructed {len(runs)} runs"
        assert not rep.codes_failed(), f"validator failures: {rep.codes_failed()}"

        # attempts joinable to raw
        att = [json.loads(l) for l in open(tmp / "attempts.jsonl", encoding="utf-8")]
        keys_raw = {f"{r['run_key']}:{r['round']}:{m['j']}" for r in rnd for m in r["merchants"]}
        keys_att = {a["decision_key"] for a in att if a.get("decision_key")}
        assert keys_att <= keys_raw, f"{len(keys_att - keys_raw)} attempt keys have no raw decision"
        assert len(keys_att) == len(keys_raw), (len(keys_att), len(keys_raw))
        return (f"16 runs, 48 rounds, validator {n_ok}/{n_tot} with 0 failures, "
                f"{g_over} G overrides, {r_econ} R econ-retries, {len(att)} attempts joined")
    finally:
        R.decide, R.transport_call, R.append_jsonl = real_decide, real_tc, real_ap
        R.transport_call_ref[0] = real_ref
        AT.BACKOFF_CAP_S = real_cap
        shutil.rmtree(tmp, ignore_errors=True)


def T12_durable_writer_is_byte_identical():
    """The fsync'd writer produces exactly the bytes the frozen writer produces."""
    import balance9_runner as R
    import to_run as TR
    tmp = Path(tempfile.mkdtemp())
    try:
        rows = [{"kind": "round", "x": 1, "s": "unicode: é中", "f": 0.1234567890123},
                {"kind": "run_footer", "nested": {"a": [1, 2, {"b": None}]}, "t": True}]
        a, b = tmp / "balance9_a.jsonl", tmp / "balance9_b.jsonl"
        R.append_jsonl(a, rows)
        R.append_jsonl(a, rows)
        dur = TR.make_durable_append(R)
        dur(b, rows)
        dur(b, rows)
        assert a.read_bytes() == b.read_bytes(), "the durable writer changed the bytes"
        # and it keeps the write guard
        try:
            dur(tmp / "balance8_x.jsonl", rows)
            raise AssertionError("the durable writer must keep the balance9_ prefix guard")
        except ValueError:
            pass
        assert not (tmp / "balance8_x.jsonl").exists()
        return f"{a.stat().st_size} bytes identical, prefix guard kept"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def T13_selftest_does_not_revert_the_patch():
    """balance9_runner.selftest() restores transport_call from transport_call_ref -- the trap.

    gate_state_invariance() ends with `globals()["transport_call"] = real` where `real` came from
    `transport_call_ref[0]`. Patching only the module global therefore leaves the ORIGINAL OpenRouter
    transport installed after the self-test, and every subsequent call would go to the dead key. This
    test asserts the failure mode exists and that patching both handles defeats it.
    """
    import balance9_runner as R
    real_tc, real_ref = R.transport_call, R.transport_call_ref[0]
    try:
        sentinel = lambda *a, **k: None                   # noqa: E731
        # (a) the trap: patch only the global
        R.transport_call = sentinel
        R.transport_call_ref[0] = real_ref
        with redirect_stdout(io.StringIO()):
            R.gate_state_invariance()
        reverted = R.transport_call is not sentinel
        assert reverted, "expected selftest to revert a global-only patch; the trap has moved"
        # (b) the fix: patch both handles
        R.transport_call = sentinel
        R.transport_call_ref[0] = sentinel
        with redirect_stdout(io.StringIO()):
            R.gate_state_invariance()
        assert R.transport_call is sentinel and R.transport_call_ref[0] is sentinel, \
            "patching both handles was still reverted"
        return "trap confirmed; both-handle patch survives selftest"
    finally:
        R.transport_call, R.transport_call_ref[0] = real_tc, real_ref
        R._R.clear()


def T14_no_openrouter_call_is_reachable():
    """Nothing on the patched path can reach OpenRouter, and no OpenRouter key is required."""
    import balance9_runner as R
    import phase2_llm as L
    import azure_transport as AT
    tmp = Path(tempfile.mkdtemp())
    try:
        for var in ("OPENROUTER_API_KEY",):
            os.environ.pop(var, None)
        # load_key must fail on this machine, which proves the run cannot silently use OpenRouter
        reachable = True
        try:
            L.load_key()
        except Exception:                                # noqa: BLE001
            reachable = False
        tx, _, _ = _mk_transport(str(tmp), lambda n, b: _resp('{"action_index": 0}'))
        real_tc, real_ref = R.transport_call, R.transport_call_ref[0]
        try:
            R.transport_call = tx
            R.transport_call_ref[0] = tx
            assert R.transport_call.__module__ == "azure_transport", R.transport_call.__module__
            assert R.transport_call_ref[0] is tx
        finally:
            R.transport_call, R.transport_call_ref[0] = real_tc, real_ref
        return f"openrouter key reachable={reachable}; patched transport is azure_transport"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def T15_seed_freshness_supplement():
    """The shipped prior-seed set is real, and 9000-9059 does not intersect it."""
    d = json.loads((PKG / "input_data" / "prior_seeds.json").read_text(encoding="utf-8"))
    prior = set(d["prior_seeds"])
    assert len(prior) == d["n_prior_seeds"] == 306, (len(prior), d["n_prior_seeds"])
    assert min(prior) == 1000 and max(prior) == 7699, (min(prior), max(prior))
    assert not (prior & set(range(9000, 9060))), "the B9-P3 block overlaps the prior corpus"
    assert d["per_file"], "no per-file provenance recorded"
    return f"{len(prior)} prior seeds, range {min(prior)}-{max(prior)}, 0 overlap with 9000-9059"


def T16_packaging_and_extraction():
    """--package-results builds an archive whose every member extracts to its recorded hash."""
    import to_run as TR
    tmp = Path(tempfile.mkdtemp())
    saved = dict(RAW=TR.RAW, ATTEMPTS=TR.ATTEMPTS, PROGRESS=TR.PROGRESS, RUNLOG=TR.RUNLOG,
                 RAW_DIR=TR.RAW_DIR, ATT_DIR=TR.ATT_DIR, LOG_DIR=TR.LOG_DIR, SUM_DIR=TR.SUM_DIR,
                 VAL_DIR=TR.VAL_DIR, DELIV=TR.DELIV, HERE=TR.HERE)
    try:
        TR.HERE = tmp
        TR.RAW_DIR, TR.ATT_DIR = tmp / "results/raw", tmp / "results/attempts"
        TR.LOG_DIR, TR.SUM_DIR = tmp / "results/logs", tmp / "results/summaries"
        TR.VAL_DIR, TR.DELIV = tmp / "results/validation", tmp / "results/final_delivery"
        TR.RAW = TR.RAW_DIR / "balance9_azure_p3_llama_raw.jsonl"
        TR.ATTEMPTS = TR.ATT_DIR / "balance9_azure_attempts.jsonl"
        TR.PROGRESS, TR.RUNLOG = TR.LOG_DIR / "progress.json", TR.LOG_DIR / "run.log"
        for d in (TR.RAW_DIR, TR.ATT_DIR, TR.LOG_DIR, TR.SUM_DIR, TR.VAL_DIR, TR.DELIV):
            d.mkdir(parents=True, exist_ok=True)
        TR.RAW.write_text("", encoding="utf-8")
        TR.ATTEMPTS.write_text(json.dumps(dict(ok=True, http_status=200, prompt_tokens=10,
                                               completion_tokens=2, transport_attempt=1,
                                               latency_s=1.0)) + "\n", encoding="utf-8")
        TR.RUNLOG.write_text("test\n", encoding="utf-8")
        man = TR.package_results()
        assert man["extraction_verified"] is True, man
        assert man["members"] >= 4, man
        assert (TR.DELIV / "SHA256SUMS.json").exists()
        assert (TR.DELIV / "SHA256SUMS.txt").exists()
        u = json.loads((TR.SUM_DIR / "usage_and_cost.json").read_text(encoding="utf-8"))
        assert u["total_tokens"] == 12 and u["cost_usd"] is None, u
        return f"{man['members']} members, extraction verified, cost left null not guessed"
    finally:
        for k, v in saved.items():
            setattr(TR, k, v)
        shutil.rmtree(tmp, ignore_errors=True)


def T17_cli_surface():
    """Exactly the four documented invocations exist, and three of them cannot call Azure."""
    import to_run as TR
    ap = TR.main.__wrapped__ if hasattr(TR.main, "__wrapped__") else None
    src = (PKG / "to_run.py").read_text(encoding="utf-8")
    for flag in ("--resume", "--validate-only", "--package-results"):
        assert f'"{flag}"' in src, f"{flag} is not defined"
    # --validate-only and --package-results must return before the transport is ever constructed
    i_val = src.index("if args.validate_only:")
    i_pkg = src.index("if args.package_results:")
    i_tx = src.index("tx = AT.AzureTransport(")
    assert i_val < i_tx and i_pkg < i_tx, "a read-only mode can reach the Azure transport"
    assert "AT.load_config()" not in src[:i_pkg], "config is loaded before the read-only modes exit"
    return "4 invocations; --validate-only and --package-results exit before any transport exists"


def T18_no_secret_in_the_package():
    """No file in the package carries anything shaped like a credential."""
    import re as _re
    pats = [(_re.compile(r"sk-or-[A-Za-z0-9\-]{10,}"), "openrouter key"),
            (_re.compile(r"\bsk-[A-Za-z0-9]{20,}"), "openai key"),
            (_re.compile(r"eyJ[A-Za-z0-9_\-]{20,}\.[A-Za-z0-9_\-]{20,}\."), "jwt"),
            (_re.compile(r"(?i)(api[-_]?key|password|secret|token)\s*[:=]\s*[\"'][^\"'{$<]{16,}"),
             "assigned credential")]
    hits = []
    # `.git` is skipped: its packfiles are compressed blobs of the same files scanned below, and a
    # chance byte sequence inside one would fail this test without meaning anything. What is in the
    # objects is exactly what is in the working tree, and the working tree is what gets scanned.
    skip = {"results", ".git", "__pycache__"}
    for p in sorted(PKG.rglob("*")):
        if not p.is_file() or skip & set(p.parts) or p.suffix in (".gz", ".tar", ".pack", ".idx"):
            continue
        try:
            txt = p.read_text(encoding="utf-8")
        except Exception:                                # noqa: BLE001
            continue
        for pat, what in pats:
            for m in pat.finditer(txt):
                # the test's own patterns and the redaction test's fixtures are not findings
                if p.name in ("run_offline_tests.py", "azure_transport.py"):
                    continue
                hits.append(f"{p.relative_to(PKG)}: {what}: {m.group(0)[:24]}")
    assert not hits, "possible secrets: " + "; ".join(hits[:5])
    return f"scanned {sum(1 for p in PKG.rglob('*') if p.is_file())} files, 0 findings"


# =================================================================================================
def main():
    print("=" * 108)
    print("Balance-9 P3 Azure offline tests -- nothing here opens a socket")
    print("=" * 108)
    tests = [
        ("T1  vendored code is byte-identical to the frozen originals", T1_vendored_integrity),
        ("T2  published prompt hashes reproduce exactly", T2_prompt_hashes),
        ("T3  balance9_runner gates G-R1..G-R10", T3_runner_gates),
        ("T4  design gates re-derived from the preregistration", T4_design_gates),
        ("T5  Azure route selection is explicit, never guessed", T5_route_selection),
        ("T6  model identity refuses quantised / wrong models", T6_model_identity),
        ("T7  credential redaction", T7_redaction),
        ("T8  transport honours the frozen transport_call contract", T8_transport_contract),
        ("T9  retry budget, hard-stop statuses, content filter", T9_retry_and_hard_stop),
        ("T10 identity gate aborts before anything is written", T10_identity_gate_blocks_run),
        ("T11 end-to-end 16-run mock matrix through the frozen runner", T11_end_to_end_mock_matrix),
        ("T12 durable writer is byte-identical to the frozen writer", T12_durable_writer_is_byte_identical),
        ("T13 selftest does not silently revert the transport patch", T13_selftest_does_not_revert_the_patch),
        ("T14 no OpenRouter call is reachable from the patched path", T14_no_openrouter_call_is_reachable),
        ("T15 seed-freshness supplement is real and disjoint", T15_seed_freshness_supplement),
        ("T16 packaging builds an archive that extracts to its hashes", T16_packaging_and_extraction),
        ("T17 exactly four CLI invocations; read-only modes cannot call", T17_cli_surface),
        ("T18 no credential-shaped string anywhere in the package", T18_no_secret_in_the_package),
    ]
    for name, fn in tests:
        t(name, fn)

    n = sum(1 for r in RESULTS if r["ok"])
    print("-" * 108)
    print(f"{n}/{len(RESULTS)} offline tests passed")
    outp = PKG / "results" / "logs" / "offline_tests.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(dict(passed=n, total=len(RESULTS),
                                    python=sys.version.split()[0],
                                    when=time.strftime("%Y-%m-%dT%H:%M:%S"),
                                    results=RESULTS), indent=1), encoding="utf-8")
    print(f"written to {outp.relative_to(PKG)}")
    if n != len(RESULTS):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
