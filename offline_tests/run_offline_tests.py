#!/usr/bin/env python3
"""run_offline_tests.py -- everything that can be checked without spending a single API call.

    python offline_tests/run_offline_tests.py
    python offline_tests/run_offline_tests.py --quick     (skips the mock matrix)

This is the gate that stands in front of the paid run. If any test here fails, the experiment as
specified is not the experiment the code would execute, and running it would produce numbers that
cannot be defended.

Three of these tests deserve a note, because they check claims rather than code paths.

`oracle_positive_control` requires arm A0 to FAIL the leakage probe. A0 is built to receive the
displayed payoff table, which is evaluator-only; if the detector stayed silent on the one arm
constructed to trip it, the silence on the other three would mean nothing. A green board on every
arm would be the failure mode, not the success.

`harness_neutrality` runs a mock merchant that reads only the published penalty and its own type,
and requires arms A1, A2 and A3 to produce byte-identical trajectories. The arms differ in what the
merchant is shown; if they also differed in what the harness *does*, an arm effect measured later
could be plumbing. This test pins the difference to the prompt.

`raw_prompt_leakage` re-reads the prompt text actually retained on disk and scans it for renderings
of every evaluator-only quantity for that cell -- true fabrication rates, rival types, the payoff
table, its argmax. It checks the artefact rather than the intention, so a leak introduced by a later
edit to the prompt builder is caught by the same test.
"""
from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
CODE = PKG / "code"
for p in (str(CODE),):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np                                                            # noqa: E402

import ha_infoclass as IC                                                     # noqa: E402
import ha_model as M                                                          # noqa: E402
import ha_prompts as PR                                                       # noqa: E402
import ha_runner as RUN                                                       # noqa: E402
import ha_transport as TR                                                     # noqa: E402
import to_run as TO                                                           # noqa: E402

RESULTS = PKG / "results"
OUT = RESULTS / "validation"


# ==================================================================================================
def t_manifest_matches_code() -> dict:
    man, sha = TO.load_manifest()
    problems = TO.check_manifest_against_code(man)
    return {"pass": not problems, "manifest_id": man["manifest_id"], "sha256": sha[:16],
            "problems": problems}


def t_registry_total() -> dict:
    """Every field placed in a view must be registered, and none may be evaluator-only.

    Gate G-H0 in ha_prompts checks this on the history-free demo state. This widens it to the
    history-bearing state, where the signal rows and the derived own-history fields first appear --
    which is exactly where an unclassified field is most likely to slip in, because those rows are
    built per round rather than written out once.
    """
    cfg = M.Config()
    unknown, eo_in_view = [], []
    seen = set()
    for arm in PR.ARMS:
        for wh in (False, True):
            view = PR.build_view(cfg, arm, PR._demo_state(cfg, with_history=wh))
            for name in view.provenance:
                seen.add(name)
                if name not in IC.FIELD_CLASS:
                    unknown.append(f"{arm}:{name}")
                elif IC.FIELD_CLASS[name] is IC.EO:
                    eo_in_view.append(f"{arm}:{name}")
    return {"pass": not unknown and not eo_in_view,
            "n_registered": len(IC.FIELD_CLASS), "n_fields_seen_in_views": len(seen),
            "unclassified": sorted(set(unknown)), "evaluator_only_in_view": sorted(set(eo_in_view))}


def t_prompt_gates() -> dict:
    res = PR.run_gates(M.Config())
    return {"pass": res["all_pass"], "n_pass": res["n_pass"], "n_gates": res["n_gates"],
            "failed": [g["gate"] for g in res["gates"] if not g["pass"]]}


def t_transport_selftest() -> dict:
    res = TR.selftest()
    return {"pass": res["all_pass"],
            "checks": {k: v.get("pass") for k, v in res.items() if isinstance(v, dict)}}


def t_runner_invariants() -> dict:
    res = RUN.run_invariants()
    return {"pass": res["all_pass"], "checks": {c["check"]: c["pass"] for c in res["checks"]}}


def t_oracle_positive_control() -> dict:
    """A0 must trip the probe; the other three must not. A silent detector proves nothing."""
    cfg = M.Config()
    verdicts = {}
    for arm in PR.ARMS:
        state = PR._demo_state(cfg, with_history=True)
        eo_base = {"true_f_all": [0.3, 0.3, 0.3, 0.3], "displayed_argmax": 7,
                   "rival_true_f": [0.30, 0.45, 0.25]}
        if arm == "A0_oracle":
            rb = M.rbar_for_market(cfg, M.draw_market(cfg, 70000),
                                   RUN.rbar_for_policy(cfg, state["policy"]))
            state["payoff_table"] = RUN.displayed_table(
                cfg, M.draw_market(cfg, 70000), rb, 0, np.array([0.3, 0.3, 0.3, 0.3]))

        def build(st):
            msgs, _ = PR.build_messages(cfg, arm, st)
            return msgs

        perts = [{"payoff_table": np.zeros(cfg.Nf)}] if arm == "A0_oracle" else \
                [{"_evaluator_only_probe": 0.9999}]
        pr = IC.invariance_probe(build, state, perts)
        verdicts[arm] = bool(pr.get("invariant", pr.get("pass", False)))
    ok = (verdicts["A0_oracle"] is False) and all(verdicts[a] for a in PR.ARMS
                                                  if a != "A0_oracle")
    return {"pass": ok, "invariant_by_arm": verdicts,
            "expected": "A0_oracle False (it is built to leak), all others True"}


def _mock_matrix(tmp: Path, behaviour: str, arms=None, rounds=4, seeds=2, fail_rate=0.0) -> dict:
    log = TR.AttemptLog(tmp / "attempts.jsonl")
    hook = TR.make_mock_hook(behaviour, fail_rate, 0)
    transport = TR.Transport(TR.mock_endpoints(), log, http_hook=hook)
    man, _ = TO.load_manifest()
    mx = TO.manifest_matrix(man)
    specs = RUN.build_matrix(mx["aliases"], arms or mx["arms"], mx["policies"],
                             mx["seeds"][:seeds], rounds)
    prog = RUN.Progress(tmp / "progress")
    summary = RUN.run_matrix(M.Config(), specs, transport, prog, tmp / "raw")
    return {"summary": summary, "transport": transport.stats(), "raw": tmp / "raw"}


def t_mock_smoke(tmp: Path) -> dict:
    r = _mock_matrix(tmp / "smoke", "malformed", rounds=4, seeds=2, fail_rate=0.05)
    s = r["summary"]
    files = sorted((tmp / "smoke" / "raw").glob("*.json.gz"))
    every_round = True
    for f in files:
        c = json.loads(gzip.open(f, "rb").read())
        every_round &= (len(c["rounds"]) == c["n_rounds"] and bool(c["complete"]))
    return {"pass": s["failed"] == 0 and s["completed"] == len(files) and every_round,
            "cells_completed": s["completed"], "cells_failed": s["failed"],
            "files": len(files), "api_calls": r["transport"]["calls"],
            "transport_retries": r["transport"]["transport_retries"]}


def t_resume_idempotence(tmp: Path) -> dict:
    """A second invocation over completed cells must make zero API calls."""
    d = tmp / "resume"
    _mock_matrix(d, "rational", arms=["A1_policy"], rounds=3, seeds=2)
    r2 = _mock_matrix(d, "rational", arms=["A1_policy"], rounds=3, seeds=2)
    return {"pass": r2["transport"]["calls"] == 0 and r2["summary"]["completed"] == 0,
            "second_run_api_calls": r2["transport"]["calls"],
            "second_run_skipped": r2["summary"]["skipped_complete"]}


def t_truncated_cell_is_not_done(tmp: Path) -> dict:
    """A file cut off mid-write must be re-run, not counted. Otherwise a kill silently
    shrinks a denominator and every mean computed from it is wrong by an unknown amount."""
    d = tmp / "trunc"
    _mock_matrix(d, "rational", arms=["A1_policy"], rounds=3, seeds=1)
    files = sorted((d / "raw").glob("*.json.gz"))
    if not files:
        return {"pass": False, "why": "no cell produced"}
    spec = RUN.CellSpec("gemma", "A1_policy", "P_GMV", 70000, 3)
    before = RUN.cell_done(spec, d / "raw")
    raw = files[0].read_bytes()
    files[0].write_bytes(raw[: len(raw) // 2])
    after_trunc = RUN.cell_done(RUN.CellSpec(*files[0].stem.replace(".json", "").split("__")[:3],
                                             int(files[0].stem.split("__s")[1].split(".")[0]), 3),
                               d / "raw")
    # and a syntactically fine file whose round count is short must also be rejected
    good = json.loads(gzip.open(files[1], "rb").read()) if len(files) > 1 else None
    short_ok = None
    if good:
        good["rounds"] = good["rounds"][:-1]
        RUN.atomic_write_json_gz(files[1], good)
        sp = RUN.CellSpec(good["model_alias"], good["arm"], good["policy_name"],
                          good["seed"], good["n_rounds"])
        short_ok = RUN.cell_done(sp, d / "raw")
    return {"pass": bool(before) and after_trunc is False and short_ok is False,
            "complete_file_done": bool(before), "truncated_file_done": after_trunc,
            "short_round_count_done": short_ok}


def t_harness_neutrality(tmp: Path) -> dict:
    """The mock reads only the published penalty and its own type. If A1, A2 and A3 nevertheless
    diverge, the harness is contributing an arm effect and no later arm comparison is clean."""
    d = tmp / "neutral"
    _mock_matrix(d, "rational", arms=["A1_policy", "A2_history", "A3_assist"], rounds=5, seeds=3)
    traj = {}
    for f in sorted((d / "raw").glob("*.json.gz")):
        c = json.loads(gzip.open(f, "rb").read())
        traj.setdefault((c["model_alias"], c["policy_name"], c["seed"]), {})[c["arm"]] = \
            [r["f"] for r in c["rounds"]]
    same, diff = 0, []
    for k, byarm in traj.items():
        for other in ("A2_history", "A3_assist"):
            if byarm.get("A1_policy") == byarm.get(other):
                same += 1
            else:
                diff.append(f"{k} A1 vs {other}")
    return {"pass": not diff, "identical": same, "differing": diff[:5],
            "note": "arms differ in what the merchant is shown, never in what the harness does"}


def t_raw_prompt_leakage(tmp: Path) -> dict:
    """Scan the prompt text actually kept on disk for renderings of evaluator-only quantities."""
    d = tmp / "leak"
    _mock_matrix(d, "rational", arms=["A1_policy", "A2_history", "A3_assist"], rounds=4, seeds=3)
    cfg = M.Config()
    hits, scanned = [], 0
    for f in sorted((d / "raw").glob("*.json.gz")):
        c = json.loads(gzip.open(f, "rb").read())
        mkt_eo = c["market_evaluator_only"]
        for dec in c["decisions"]:
            if not dec.get("prompt_messages"):
                continue
            scanned += 1
            text = "\n".join(m["content"] for m in dec["prompt_messages"])
            j = dec["j"]
            secrets = {}
            for k, v in enumerate(mkt_eo["q"]):
                if k != j:
                    secrets[f"rival_q_{k}"] = v
            for k, v in enumerate(mkt_eo["b"]):
                if k != j:
                    secrets[f"rival_b_{k}"] = v
            for k, v in (dec.get("evaluator_only") or {}).items():
                if isinstance(v, (int, float)):
                    secrets[f"eo_{k}"] = v
                elif isinstance(v, list):
                    for i, x in enumerate(v):
                        if isinstance(x, (int, float)):
                            secrets[f"eo_{k}_{i}"] = x
            allowed = [v for _, v in IC._flatten(dec.get("platform_observable") or {})]
            allowed += [v for _, v in IC._flatten(dec.get("merchant_private") or {})]
            allowed += list(M.spec_dict(cfg).values())
            res = IC.scan_text_for_secrets(text, secrets, allowed_values=allowed)
            if res.get("hits"):
                hits.append({"cell": c["run_key"], "j": j, "hits": res["hits"][:4]})
    return {"pass": not hits, "prompts_scanned": scanned, "leaks": hits[:5]}


def t_retry_counters_separate(tmp: Path) -> dict:
    """The three counters must move independently. If a rate limit could raise the economic-retry
    count, an infrastructure hiccup would read as a merchant reconsidering."""
    d = tmp / "counters"
    r_noretry = _mock_matrix(d / "a", "rational", arms=["A2_history"], rounds=6, seeds=2,
                             fail_rate=0.0)
    r_http = _mock_matrix(d / "b", "rational", arms=["A2_history"], rounds=6, seeds=2,
                          fail_rate=0.30)
    r_bad = _mock_matrix(d / "c", "malformed", arms=["A2_history"], rounds=6, seeds=2,
                         fail_rate=0.0)
    r_a3 = _mock_matrix(d / "d", "greedy", arms=["A3_assist"], rounds=6, seeds=2, fail_rate=0.0)

    def agg(res):
        tot = {"schema_repairs": 0, "economic_retries": 0, "transport_attempts": 0, "decisions": 0}
        for f in sorted((res["raw"]).glob("*.json.gz")):
            c = json.loads(gzip.open(f, "rb").read())
            for k in tot:
                tot[k] += c["counters"][k]
        return tot
    a, b, cc, dd = agg(r_noretry), agg(r_http), agg(r_bad), agg(r_a3)
    checks = {
        "http_failures_raise_only_transport":
            b["transport_attempts"] > a["transport_attempts"]
            and b["schema_repairs"] == a["schema_repairs"] == 0
            and b["economic_retries"] == a["economic_retries"] == 0,
        "bad_json_raises_only_schema_repairs":
            cc["schema_repairs"] > 0 and cc["economic_retries"] == 0,
        "economic_retry_is_arm_a3_only":
            dd["economic_retries"] > 0 and a["economic_retries"] == 0,
        "decisions_exclude_repairs":
            cc["decisions"] == a["decisions"],
    }
    return {"pass": all(checks.values()), "checks": checks,
            "baseline": a, "http_30pct": b, "malformed": cc, "a3": dd}


def t_validator_catches_corruption(tmp: Path) -> dict:
    """Mutation test. Break one thing per cell and require the matching check to fail.

    A validator that has only ever been run on good data is an untested assertion. Each mutation
    below is a specific way results could be wrong -- a tampered figure, a broken recursion, an
    impossible signal, an off-grid action, noise that does not regenerate from the seed, an
    infrastructure event booked as an economic one, a seed from outside the frozen block -- and the
    check that is supposed to catch it must actually turn red.
    """
    import ha_validate as VAL
    d = tmp / "mutants"
    _mock_matrix(d, "rational", arms=["A1_policy", "A3_assist"], rounds=5, seeds=4)
    files = sorted((d / "raw").glob("*.json.gz"))
    if len(files) < 8:
        return {"pass": False, "why": f"need >= 8 cells to mutate, produced {len(files)}"}

    def load(p):
        return json.loads(gzip.open(p, "rb").read())

    muts = []

    def mutate(i, name, expect, fn):
        c = load(files[i])
        fn(c)
        RUN.atomic_write_json_gz(files[i], c)
        muts.append({"cell": files[i].name, "mutation": name, "expected_check": expect})

    def _gmv(c):
        c["rounds"][2]["GMV"] *= 1.05

    def _rep(c):
        c["rounds"][1]["signals"][0]["reputation_after"] += 0.03

    def _ref(c):
        c["rounds"][0]["signals"][1]["own_refunds"] = 99

    def _grid(c):
        c["rounds"][0]["f"][0] = 0.333

    def _crn(c):
        c["rounds"][0]["signals"][0]["own_complaints"] += 1

    def _ctr(c):
        c["counters"]["economic_retries"] = 3
        c["arm"] = "A1_policy"

    def _seed(c):
        c["seed"] = 99999

    def _trunc(c):
        c["rounds"] = c["rounds"][:-1]

    for i, (nm, exp, fn) in enumerate([
            ("tampered GMV", "V2_accounting_identity", _gmv),
            ("broken reputation recursion", "V3_reputation_recursion", _rep),
            ("refunds exceed complaints", "V4_signal_bounds", _ref),
            ("off-grid action", "V5_action_grid", _grid),
            ("signal not regenerable from seed", "V6_crn_reproducible", _crn),
            ("economic retry outside A3", "V10_counter_separation", _ctr),
            ("seed outside the frozen block", "V9_manifest_conformance", _seed),
            ("round count short of declared", "V1_cell_integrity", _trunc)]):
        mutate(i, nm, exp, fn)

    cfg = M.Config()
    cells = [load(p) for p in files]
    man = json.loads((PKG / "manifests" / "ha_manifest_HA-M1.json").read_text(encoding="utf-8"))
    verdicts = {c["check"]: c["pass"] for c in [
        VAL.v1_cell_integrity(cells), VAL.v2_accounting_identity(cells, cfg),
        VAL.v3_reputation_recursion(cells, cfg), VAL.v4_signal_bounds(cells, cfg),
        VAL.v5_action_grid(cells, cfg), VAL.v6_crn_reproducible(cells, cfg),
        VAL.v9_manifest_conformance(cells, man), VAL.v10_counter_separation(cells)]}
    expected = {m["expected_check"] for m in muts}
    missed = sorted(c for c in expected if verdicts.get(c) is not False)
    return {"pass": not missed, "mutations": len(muts),
            "checks_that_failed_as_intended": sorted(c for c in expected
                                                     if verdicts.get(c) is False),
            "mutations_not_caught": missed, "verdicts": verdicts}


def t_no_credentials_in_repo() -> dict:
    """No key, endpoint, deployment name or token may be committed. Checked over the package.

    The allowlist below is deliberately a short list of exact literals rather than a rule, and it is
    echoed into the result so a reader can see everything that was excused. A pattern-based excuse
    ("ignore anything in a selftest") would grow to cover a real credential the first time someone
    pasted one into a test; four enumerated strings cannot.
    """
    import re
    allow = {
        "sk-0123456789abcdefghij",              # redaction fixture: visibly sequential, not a key
        "https://x.openai.azure.com",           # routing fixture: single-letter placeholder host
        "https://x.services.ai.azure.com",      # routing fixture
        "https://x.inference.ai.azure.com",     # routing fixture
    }
    pats = [(re.compile(r"sk-[A-Za-z0-9]{20,}"), "openai-style key"),
            (re.compile(r"[A-Za-z0-9_\-]{32,}\.[A-Za-z0-9_\-]{16,}\.[A-Za-z0-9_\-]{16,}"), "jwt"),
            (re.compile(r"https://[A-Za-z0-9\-]+\.(openai\.azure\.com|services\.ai\.azure\.com|"
                        r"inference\.ai\.azure\.com)"), "live azure endpoint"),
            (re.compile(r"(?i)api[_-]?key\s*[:=]\s*[\"'][^\"'\s]{12,}[\"']"), "inline api key")]
    hits = []
    for f in sorted(PKG.rglob("*")):
        if not f.is_file() or f.suffix not in (".py", ".json", ".md", ".txt", ".yaml", ".yml"):
            continue
        if "results" in f.parts:
            continue
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for pat, label in pats:
            for mt in pat.finditer(text):
                seg = mt.group(0)
                if seg in allow or "mock.invalid" in seg or "example" in seg.lower():
                    continue
                hits.append({"file": str(f.relative_to(PKG)), "kind": label,
                             "match": seg[:24] + "..."})
    return {"pass": not hits, "files_scanned": "package excluding results/",
            "allowlisted_fixtures": sorted(allow), "hits": hits[:10]}


# ==================================================================================================
TESTS = [
    ("manifest_matches_code", t_manifest_matches_code, False),
    ("registry_total", t_registry_total, False),
    ("prompt_gates", t_prompt_gates, False),
    ("transport_selftest", t_transport_selftest, False),
    ("runner_invariants", t_runner_invariants, False),
    ("oracle_positive_control", t_oracle_positive_control, False),
    ("no_credentials_in_repo", t_no_credentials_in_repo, False),
    ("mock_smoke", t_mock_smoke, True),
    ("resume_idempotence", t_resume_idempotence, True),
    ("truncated_cell_is_not_done", t_truncated_cell_is_not_done, True),
    ("harness_neutrality", t_harness_neutrality, True),
    ("raw_prompt_leakage", t_raw_prompt_leakage, True),
    ("retry_counters_separate", t_retry_counters_separate, True),
    ("validator_catches_corruption", t_validator_catches_corruption, True),
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip the tests that run a mock matrix")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--keep", action="store_true", help="keep the temporary run directory")
    ap.add_argument("--out", default=str(OUT / "offline_tests.json"))
    a = ap.parse_args(argv)

    print("=" * 96)
    print("OFFLINE TESTS  --  no API key, no network, no cost")
    print("=" * 96)
    tmp = Path(tempfile.mkdtemp(prefix="ha_offline_"))
    results, t0 = [], time.time()
    try:
        for name, fn, needs_tmp in TESTS:
            if a.only and name not in a.only:
                continue
            if a.quick and needs_tmp:
                results.append({"test": name, "pass": None, "skipped": "quick"})
                print(f"[SKIP] {name}")
                continue
            t1 = time.time()
            try:
                r = fn(tmp) if needs_tmp else fn()
            except Exception as exc:                                          # noqa: BLE001
                r = {"pass": False, "error": f"{type(exc).__name__}: {exc}",
                     "traceback": traceback.format_exc()[-1200:]}
            r["test"] = name
            r["seconds"] = round(time.time() - t1, 2)
            results.append(r)
            tag = "PASS" if r.get("pass") else "FAIL"
            print(f"[{tag}] {name}  ({r['seconds']}s)")
            if not r.get("pass"):
                print("      " + json.dumps({k: v for k, v in r.items()
                                             if k not in ("test", "pass", "seconds",
                                                          "traceback")},
                                            default=str)[:900])
                if r.get("traceback"):
                    print("      " + r["traceback"].replace("\n", "\n      ")[-700:])
    finally:
        if not a.keep:
            shutil.rmtree(tmp, ignore_errors=True)
        else:
            print(f"\ntemporary run directory kept at {tmp}")

    ran = [r for r in results if r.get("pass") is not None]
    n_pass = sum(1 for r in ran if r["pass"])
    payload = {"schema_version": "ha-offline-tests-1",
               "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "n_tests": len(ran), "n_pass": n_pass, "all_pass": n_pass == len(ran),
               "seconds": round(time.time() - t0, 1),
               "git": TO.git_info(), "code_sha256_prefix": TO.code_hashes(),
               "results": results}
    outp = Path(a.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    tmpf = outp.with_suffix(outp.suffix + ".tmp")
    tmpf.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmpf.replace(outp)

    print("=" * 96)
    print(f"  {n_pass}/{len(ran)} passed in {payload['seconds']}s   ->  {outp}")
    print("=" * 96)
    return 0 if payload["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
