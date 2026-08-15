#!/usr/bin/env python3
"""ha_validate.py -- an independent check on everything the analyser claims.

Run from the root of the repository:

    python code/ha_validate.py                                 # results/raw + results/summaries
    python code/ha_validate.py --raw results/raw_smoke --summaries results/summaries/smoke

This deliberately does NOT import ha_analyze. Where it checks a number the analyser produced, it
recomputes that number from the raw cells with its own arithmetic and compares. A validator that
called the analyser's functions would only prove the analyser is deterministic.

It does import ha_model and ha_runner, and that is the point rather than a compromise: the strongest
check available is V6, which rebuilds every complaint, refund and audit count in every cell from the
market seed alone and requires an exact match against what was written to disk. If it passes, the
noise in the results is the noise the protocol specifies -- not merely self-consistent, but
reproducible from a single integer by anyone who has the code.

The checks, and what a failure of each would mean:

  V1  cell integrity        a cell is complete and its round count is exactly its declared horizon
  V2  accounting identity   demand, traffic, profit and GMV satisfy the model's own equations
  V3  reputation recursion  r' = clip(r + eta(1-r) - P) with P from the published policy
  V4  signal bounds         refunds <= complaints <= sampled transactions; flags <= audited
  V5  action grid           every action is a point on the declared 21-point grid
  V6  CRN reproducibility   signals rebuild exactly from the market seed
  V7  metric recomputation  the analyser's per-cell numbers, recomputed independently
  V8  benchmark alignment   ratios use the same seed and policy they claim to
  V9  manifest conformance  the cells on disk are the cells the manifest specifies
  V10 counter separation    decisions, repairs, economic retries and transport attempts reconcile
  V11 prompt hygiene        no evaluator-only quantity appears in any retained prompt
  V12 claim traceability    every summary field is backed by cells that exist

Exit code is non-zero if any check fails, so this can gate a delivery.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np                                                            # noqa: E402

import ha_infoclass as IC                                                     # noqa: E402
import ha_model as M                                                          # noqa: E402
import ha_runner as RUN                                                       # noqa: E402

PKG = HERE.parent
RAW = PKG / "results" / "raw"
SUMM = PKG / "results" / "summaries"
SOLVER = PKG / "results" / "solver"
OUT = PKG / "results" / "validation"

TOL = 1e-9
TOL_LOOSE = 1e-6
TAIL_ROUNDS = 20


def _fail(rows, cell, what, detail):
    rows.append({"cell": cell, "what": what, "detail": detail})


# ==================================================================================================
def v1_cell_integrity(cells) -> dict:
    bad = []
    for c in cells:
        if not c.get("complete"):
            _fail(bad, c["run_key"], "not marked complete", {})
        if len(c.get("rounds", [])) != c.get("n_rounds"):
            _fail(bad, c["run_key"], "round count mismatch",
                  {"rounds": len(c.get("rounds", [])), "declared": c.get("n_rounds")})
        if c.get("schema_version") != "ha-cell-1":
            _fail(bad, c["run_key"], "unexpected schema", {"v": c.get("schema_version")})
        rs = [r["round"] for r in c["rounds"]]
        if rs != list(range(1, len(rs) + 1)):
            _fail(bad, c["run_key"], "round indices not 1..n", {})
    return {"check": "V1_cell_integrity", "pass": not bad, "n_violations": len(bad),
            "violations": bad[:10]}


def v2_accounting_identity(cells, cfg) -> dict:
    """Recompute demand, traffic, profit and GMV from the recorded actions and reputations."""
    bad = []
    for c in cells:
        mkt = M.draw_market(cfg, int(c["seed"]))
        for r in c["rounds"]:
            f = np.array(r["f"], dtype=float)
            rb = np.array(r["r_before"], dtype=float)
            u = cfg.alpha * (mkt.q + (1 - mkt.q) * f) + cfg.beta * rb - cfg.gamma * mkt.p
            den = math.exp(cfg.w0) + float(np.exp(u).sum())
            s = np.exp(u) / den
            Q = cfg.Q0 * math.exp(-cfg.lam * float(f.mean()))
            y = Q * s
            profit = cfg.margin_frac * mkt.p * y
            gmv = float(Q * float((mkt.p * s).sum()))
            for name, got, want in (("share", r["share"], s), ("sales", r["sales"], y),
                                    ("profit", r["profit"], profit)):
                if not np.allclose(np.array(got, dtype=float), want, atol=TOL_LOOSE):
                    _fail(bad, c["run_key"], f"{name} mismatch at round {r['round']}",
                          {"recorded": list(np.round(got, 8)), "recomputed": list(np.round(want, 8))})
            if abs(float(r["Q"]) - Q) > TOL_LOOSE:
                _fail(bad, c["run_key"], f"Q mismatch at round {r['round']}",
                      {"recorded": r["Q"], "recomputed": Q})
            if abs(float(r["GMV"]) - gmv) > TOL_LOOSE:
                _fail(bad, c["run_key"], f"GMV mismatch at round {r['round']}",
                      {"recorded": r["GMV"], "recomputed": gmv})
    return {"check": "V2_accounting_identity", "pass": not bad, "n_violations": len(bad),
            "violations": bad[:6]}


def v3_reputation_recursion(cells, cfg) -> dict:
    bad = []
    for c in cells:
        kappa, tau, kappa_a, tau_a = (float(x) for x in c["policy"])
        for r in c["rounds"]:
            for j, s in enumerate(r["signals"]):
                pen = M.penalty_from_signal(kappa, tau, float(s["own_complaint_rate"]),
                                            kappa_a, tau_a,
                                            float(s["own_audit_flags"]) / cfg.N_audit)
                if abs(pen - float(s["own_penalty_applied"])) > TOL_LOOSE:
                    _fail(bad, c["run_key"], f"penalty mismatch r{r['round']} m{j}",
                          {"recorded": s["own_penalty_applied"], "recomputed": pen})
                rn = M.reputation_update(cfg, float(s["reputation_before"]), pen)
                if abs(rn - float(s["reputation_after"])) > TOL_LOOSE:
                    _fail(bad, c["run_key"], f"reputation mismatch r{r['round']} m{j}",
                          {"recorded": s["reputation_after"], "recomputed": rn})
            if r["round"] < len(c["rounds"]):
                nxt = c["rounds"][r["round"]]
                if not np.allclose(r["r_after"], nxt["r_before"], atol=TOL):
                    _fail(bad, c["run_key"], f"reputation not carried into round {r['round'] + 1}",
                          {"r_after": r["r_after"], "next_r_before": nxt["r_before"]})
    return {"check": "V3_reputation_recursion", "pass": not bad, "n_violations": len(bad),
            "violations": bad[:6]}


def v4_signal_bounds(cells, cfg) -> dict:
    bad = []
    for c in cells:
        for r in c["rounds"]:
            for j, s in enumerate(r["signals"]):
                D, ref, C = s["own_complaints"], s["own_refunds"], s["own_audit_flags"]
                if not (0 <= D <= cfg.N_obs):
                    _fail(bad, c["run_key"], "complaints out of range", {"D": D})
                if not (0 <= ref <= D):
                    _fail(bad, c["run_key"], "refunds exceed complaints", {"D": D, "refunds": ref})
                if not (0 <= C <= cfg.N_audit):
                    _fail(bad, c["run_key"], "audit flags out of range", {"C": C})
                if abs(s["own_complaint_rate"] - D / cfg.N_obs) > TOL:
                    _fail(bad, c["run_key"], "complaint rate not D/N_obs", {"D": D})
    return {"check": "V4_signal_bounds", "pass": not bad, "n_violations": len(bad),
            "violations": bad[:6]}


def v5_action_grid(cells, cfg) -> dict:
    grid = np.round(cfg.fgrid, 10)
    bad = []
    for c in cells:
        for r in c["rounds"]:
            for j, f in enumerate(r["f"]):
                if not np.any(np.abs(grid - round(float(f), 10)) < 1e-9):
                    _fail(bad, c["run_key"], f"action off grid at round {r['round']}", {"f": f})
    return {"check": "V5_action_grid", "pass": not bad, "n_violations": len(bad),
            "violations": bad[:6]}


def v6_crn_reproducible(cells, cfg) -> dict:
    """Rebuild every signal from the market seed. The single strongest check in this file.

    If it holds, the entire noise process in the results is regenerable from one integer, so the
    pairing across arms is not an assertion in a protocol document but a property of the data.
    """
    bad, n_checked = [], 0
    for c in cells:
        mkt = M.draw_market(cfg, int(c["seed"]))
        crn = RUN.draw_crn(cfg, int(c["seed"]), int(c["n_rounds"]))
        for r in c["rounds"]:
            t = r["round"] - 1
            for j, s in enumerate(r["signals"]):
                z = RUN.signal_from_crn(cfg, crn, t, j, float(s["own_action"]), float(mkt.b[j]))
                n_checked += 1
                for k in ("own_complaints", "own_refunds", "own_audit_flags"):
                    if int(z[k]) != int(s[k]):
                        _fail(bad, c["run_key"], f"{k} not reproducible r{r['round']} m{j}",
                              {"recorded": s[k], "rebuilt": z[k], "f": s["own_action"]})
    return {"check": "V6_crn_reproducible", "pass": not bad, "n_signals_checked": n_checked,
            "n_violations": len(bad), "violations": bad[:6]}


def v7_metric_recomputation(cells, cfg, metrics) -> dict:
    """Recompute the analyser's per-cell numbers with independent arithmetic."""
    by_key = {m["run_key"]: m for m in metrics}
    bad, n = [], 0
    for c in cells:
        m = by_key.get(c["run_key"])
        if m is None:
            _fail(bad, c["run_key"], "cell has no metric record", {})
            continue
        n += 1
        rounds = c["rounds"]
        tail_n = min(TAIL_ROUNDS, len(rounds))
        tail = rounds[-tail_n:]
        gmv = sum(float(r["GMV"]) for r in tail) / tail_n
        if abs(gmv - float(m["realized_GMV_tail"])) > TOL_LOOSE:
            _fail(bad, c["run_key"], "realized_GMV_tail mismatch",
                  {"analyser": m["realized_GMV_tail"], "validator": gmv})
        fs = [f for r in tail for f in r["f"]]
        mean_f = sum(fs) / len(fs)
        if abs(mean_f - float(m["realized_mean_f_tail"])) > TOL_LOOSE:
            _fail(bad, c["run_key"], "realized_mean_f_tail mismatch",
                  {"analyser": m["realized_mean_f_tail"], "validator": mean_f})
        prof = sum(sum(r["profit"]) for r in tail) / tail_n
        if abs(prof - float(m["realized_profit_tail_total"])) > TOL_LOOSE:
            _fail(bad, c["run_key"], "realized_profit_tail_total mismatch",
                  {"analyser": m["realized_profit_tail_total"], "validator": prof})
        # exploitability, recomputed from the modal tail profile
        mkt = M.draw_market(cfg, int(c["seed"]))
        rg = RUN.rbar_for_policy(cfg, tuple(float(x) for x in c["policy"]))
        rbar_m = M.rbar_for_market(cfg, mkt, rg)
        mod = []
        for j in range(int(c["m"])):
            col = [round(float(r["f"][j]), 6) for r in tail]
            mod.append(max(set(col), key=col.count))
        fidx = np.array([int(round(v * (cfg.Nf - 1))) for v in mod], dtype=int)
        eps = M.exploitability(cfg, mkt, rbar_m, fidx)
        if abs(float(np.mean(eps)) - float(m["stationary_exploitability_mean"])) > TOL_LOOSE:
            _fail(bad, c["run_key"], "stationary_exploitability_mean mismatch",
                  {"analyser": m["stationary_exploitability_mean"],
                   "validator": float(np.mean(eps))})
        is_nash = bool(np.max(eps) <= 1e-9)
        if is_nash != bool(m["is_exact_nash_at_tail_profile"]):
            _fail(bad, c["run_key"], "Nash verdict mismatch",
                  {"analyser": m["is_exact_nash_at_tail_profile"], "validator": is_nash})
    return {"check": "V7_metric_recomputation", "pass": not bad, "n_cells_recomputed": n,
            "n_violations": len(bad), "violations": bad[:6]}


def v8_benchmark_alignment(metrics, bench) -> dict:
    """A ratio must divide by the benchmark for its own seed and its own policy."""
    bad = []
    if not bench:
        return {"check": "V8_benchmark_alignment", "pass": None, "why": "no benchmark file"}
    per, named = bench.get("per_seed", {}), bench.get("named_per_seed", {})
    block = set(bench.get("seed_block", {}).get("seeds") or [])
    lo, hi = (min(block), max(block)) if block else (None, None)
    for m in metrics:
        s = str(m["seed"])
        if lo is not None and not (lo <= m["seed"] <= hi):
            _fail(bad, m["run_key"], "seed outside the benchmark block", {"seed": m["seed"]})
        if s not in per:
            _fail(bad, m["run_key"], "no benchmark for this seed", {"seed": s})
            continue
        if m.get("G_FB") is not None and abs(m["G_FB"] - float(per[s]["G_FB"])) > TOL_LOOSE:
            _fail(bad, m["run_key"], "G_FB does not match the solver for this seed",
                  {"metric": m["G_FB"], "solver": per[s]["G_FB"]})
        nb = (named.get(s) or {}).get(m["policy"])
        if nb and m.get("G_NASH_same_policy_worst_eq") is not None:
            if abs(m["G_NASH_same_policy_worst_eq"] - float(nb["gmv_worst_eq"])) > TOL_LOOSE:
                _fail(bad, m["run_key"], "G_NASH does not match the solver for this seed+policy",
                      {"metric": m["G_NASH_same_policy_worst_eq"], "solver": nb["gmv_worst_eq"]})
        for num, den, name in ((m.get("realized_GMV_tail"), m.get("G_FB"),
                                "ratio_realizedGMV_over_G_FB"),
                               (m.get("realized_GMV_tail"), m.get("G_SB_P_pessimistic"),
                                "ratio_realizedGMV_over_G_SB_P")):
            if num is not None and den:
                if abs(num / den - float(m[name])) > TOL_LOOSE:
                    _fail(bad, m["run_key"], f"{name} is not numerator/denominator",
                          {"stated": m[name], "recomputed": num / den})
    return {"check": "V8_benchmark_alignment", "pass": not bad, "n_violations": len(bad),
            "violations": bad[:6]}


def v9_manifest_conformance(cells, manifest) -> dict:
    d = manifest["design"]
    sb = d["seed_block"]
    allowed_seeds = set(range(sb["start"], sb["start"] + sb["count"]))
    bad, seen = [], set()
    for c in cells:
        if c["arm"] not in d["arms"]:
            _fail(bad, c["run_key"], "arm not in the manifest", {"arm": c["arm"]})
        if c["policy_name"] not in d["policies"]:
            _fail(bad, c["run_key"], "policy not in the manifest", {"policy": c["policy_name"]})
        else:
            p = d["policies"][c["policy_name"]]
            want = [p["kappa"], p["tau"], p["kappa_a"], p["tau_a"]]
            if [float(x) for x in c["policy"]] != [float(x) for x in want]:
                _fail(bad, c["run_key"], "policy values differ from the manifest",
                      {"cell": c["policy"], "manifest": want})
        if int(c["seed"]) not in allowed_seeds:
            _fail(bad, c["run_key"], "seed outside the frozen block", {"seed": c["seed"]})
        if int(c["m"]) != int(d["merchants_per_market"]):
            _fail(bad, c["run_key"], "merchant count differs", {"m": c["m"]})
        key = (c["model_alias"], c["arm"], c["policy_name"], c["seed"])
        if key in seen:
            _fail(bad, c["run_key"], "duplicate cell", {})
        seen.add(key)
    horizons = sorted({int(c["n_rounds"]) for c in cells})
    return {"check": "V9_manifest_conformance", "pass": not bad, "n_violations": len(bad),
            "n_cells": len(cells), "distinct_horizons": horizons,
            "note": ("all cells share one horizon" if len(horizons) == 1 else
                     "MIXED horizons: cells of different lengths must not be pooled without saying "
                     "so, because the tail window then covers a different fraction of each run"),
            "violations": bad[:6]}


def v10_counter_separation(cells) -> dict:
    """decisions = rounds*m + economic_retries, and the three counters never borrow from each
    other. If this fails, an infrastructure event has been recorded as an economic one."""
    bad = []
    for c in cells:
        k = c["counters"]
        expect = int(c["n_rounds"]) * int(c["m"]) + int(k["economic_retries"])
        if int(k["decisions"]) != expect:
            _fail(bad, c["run_key"], "decision count does not reconcile",
                  {"decisions": k["decisions"], "expected": expect,
                   "rounds*m": int(c["n_rounds"]) * int(c["m"]),
                   "economic_retries": k["economic_retries"]})
        if int(k["transport_attempts"]) < int(k["decisions"]):
            _fail(bad, c["run_key"], "fewer transport attempts than decisions",
                  {"attempts": k["transport_attempts"], "decisions": k["decisions"]})
        if c["arm"] != "A3_assist" and int(k["economic_retries"]) != 0:
            _fail(bad, c["run_key"], "economic retry outside arm A3",
                  {"arm": c["arm"], "economic_retries": k["economic_retries"]})
        if int(k["economic_retries_changed"]) > int(k["economic_retries"]):
            _fail(bad, c["run_key"], "more retries changed than offered", {})
        n_dec = len(c.get("decisions", []))
        if n_dec != int(k["decisions"]):
            _fail(bad, c["run_key"], "recorded decision rows do not match the counter",
                  {"rows": n_dec, "counter": k["decisions"]})
    return {"check": "V10_counter_separation", "pass": not bad, "n_violations": len(bad),
            "violations": bad[:6]}


def v11_prompt_hygiene(cells, cfg) -> dict:
    """Independently scan retained prompts for evaluator-only quantities."""
    bad, n = [], 0
    spec_vals = list(M.spec_dict(cfg).values())
    for c in cells:
        eo_mkt = c["market_evaluator_only"]
        for dec in c.get("decisions", []):
            if not dec.get("prompt_messages"):
                continue
            n += 1
            j = int(dec["j"])
            text = "\n".join(mm["content"] for mm in dec["prompt_messages"])
            secrets = {}
            for k, v in enumerate(eo_mkt["q"]):
                if k != j:
                    secrets[f"rival_true_quality_{k}"] = v
            for k, v in enumerate(eo_mkt["b"]):
                if k != j:
                    secrets[f"rival_baseline_{k}"] = v
            for k, v in (dec.get("evaluator_only") or {}).items():
                if isinstance(v, (int, float)):
                    secrets[f"eo_{k}"] = v
                elif isinstance(v, list):
                    for i, x in enumerate(v):
                        if isinstance(x, (int, float)):
                            secrets[f"eo_{k}[{i}]"] = x
            allowed = [v for _, v in IC._flatten(dec.get("platform_observable") or {})]
            allowed += [v for _, v in IC._flatten(dec.get("merchant_private") or {})]
            allowed += spec_vals
            res = IC.scan_text_for_secrets(text, secrets, allowed_values=allowed)
            if res.get("hits"):
                _fail(bad, c["run_key"], "evaluator-only value rendered in a prompt",
                      {"j": j, "hits": res["hits"][:3]})
            if dec.get("prompt_hash") and IC.prompt_hash(dec["prompt_messages"]) != \
                    dec["prompt_hash"]:
                _fail(bad, c["run_key"], "prompt hash does not match the retained text", {})
    return {"check": "V11_prompt_hygiene", "pass": not bad, "prompts_scanned": n,
            "n_violations": len(bad), "violations": bad[:6],
            "note": "Only arm A0_oracle is permitted to carry the payoff table, and it is declared "
                    "evaluator-only there rather than scanned as a secret."}


def v12_claim_traceability(summary, metrics) -> dict:
    """Every group the summary reports must be backed by cells that exist on disk."""
    bad = []
    have = set()
    for m in metrics:
        have.add((m["model"], m["arm"], m["policy"]))
    for key, g in (summary.get("groups") or {}).items():
        parts = tuple(key.split("|"))
        if parts not in have:
            _fail(bad, key, "summary group has no backing cells", {})
            continue
        n = len([m for m in metrics if (m["model"], m["arm"], m["policy"]) == parts])
        if n != g["n_cells"]:
            _fail(bad, key, "group cell count differs from the metric records",
                  {"summary": g["n_cells"], "metrics": n})
        seeds = sorted(m["seed"] for m in metrics
                       if (m["model"], m["arm"], m["policy"]) == parts)
        if seeds != sorted(g["seeds"]):
            _fail(bad, key, "group seed list differs", {})
    for lab, con in (summary.get("contrasts") or {}).items():
        if con.get("n_pairs", 0) < 2 and con.get("p_two_sided") is not None:
            _fail(bad, lab, "p-value reported on fewer than two pairs", con)
    ev = summary.get("equilibrium_vs_stability") or {}
    n_nash = len([m for m in metrics if m.get("is_exact_nash_at_tail_profile")])
    if ev and int(ev.get("n_exact_nash_at_tail_profile", -1)) != n_nash:
        _fail(bad, "equilibrium_vs_stability", "Nash count differs from the metric records",
              {"summary": ev.get("n_exact_nash_at_tail_profile"), "metrics": n_nash})
    return {"check": "V12_claim_traceability", "pass": not bad, "n_violations": len(bad),
            "violations": bad[:6]}


# ==================================================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(RAW))
    ap.add_argument("--summaries", default=str(SUMM))
    ap.add_argument("--benchmarks", default=str(SOLVER / "ha_benchmarks.json"))
    ap.add_argument("--manifest", default=str(PKG / "manifests" / "ha_manifest_HA-M1.json"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    print("=" * 96)
    print("INDEPENDENT VALIDATION")
    print("=" * 96)
    raw_dir, sum_dir = Path(a.raw), Path(a.summaries)
    files = sorted(glob.glob(str(raw_dir / "*.json.gz")))
    if not files:
        print(f"no cells in {raw_dir}")
        return 2
    cfg = M.Config()
    cells = []
    for f in files:
        try:
            cells.append(json.loads(gzip.open(f, "rb").read()))
        except (OSError, EOFError, json.JSONDecodeError) as exc:
            print(f"  UNREADABLE {Path(f).name}: {exc}")
    print(f"raw cells      {len(cells)} from {raw_dir}")

    mp = sum_dir / "ha_metrics_cells.json.gz"
    sp = sum_dir / "ha_metrics_summary.json"
    metrics = json.loads(gzip.open(mp, "rb").read()) if mp.exists() else []
    summary = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else {}
    print(f"metric records {len(metrics)} from {sum_dir}")
    bench = json.loads(Path(a.benchmarks).read_text(encoding="utf-8")) \
        if Path(a.benchmarks).exists() else {}
    man = json.loads(Path(a.manifest).read_text(encoding="utf-8"))

    t0 = time.time()
    checks = [
        v1_cell_integrity(cells),
        v2_accounting_identity(cells, cfg),
        v3_reputation_recursion(cells, cfg),
        v4_signal_bounds(cells, cfg),
        v5_action_grid(cells, cfg),
        v6_crn_reproducible(cells, cfg),
        v9_manifest_conformance(cells, man),
        v10_counter_separation(cells),
        v11_prompt_hygiene(cells, cfg),
    ]
    if metrics:
        checks += [v7_metric_recomputation(cells, cfg, metrics),
                   v8_benchmark_alignment(metrics, bench)]
    else:
        checks += [{"check": "V7_metric_recomputation", "pass": None, "why": "no metric records"},
                   {"check": "V8_benchmark_alignment", "pass": None, "why": "no metric records"}]
    if summary and metrics:
        checks.append(v12_claim_traceability(summary, metrics))
    else:
        checks.append({"check": "V12_claim_traceability", "pass": None, "why": "no summary"})

    checks.sort(key=lambda c: int(c["check"].split("_")[0][1:]))
    print()
    for c in checks:
        tag = "PASS" if c["pass"] else ("SKIP" if c["pass"] is None else "FAIL")
        extra = ""
        if c.get("n_signals_checked"):
            extra = f"  ({c['n_signals_checked']:,} signals rebuilt)"
        elif c.get("prompts_scanned"):
            extra = f"  ({c['prompts_scanned']} prompts scanned)"
        elif c.get("n_cells_recomputed"):
            extra = f"  ({c['n_cells_recomputed']} cells recomputed)"
        print(f"[{tag}] {c['check']}{extra}")
        if c["pass"] is False:
            print(f"       {c['n_violations']} violations")
            for v in c["violations"][:3]:
                print(f"       - {v['cell']}: {v['what']} {json.dumps(v['detail'], default=str)[:160]}")
        elif c["pass"] is None:
            print(f"       {c.get('why', '')}")

    ran = [c for c in checks if c["pass"] is not None]
    n_pass = sum(1 for c in ran if c["pass"])
    payload = {"schema_version": "ha-validation-1",
               "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "raw_dir": str(raw_dir), "summaries_dir": str(sum_dir),
               "n_cells": len(cells), "n_metric_records": len(metrics),
               "n_checks_run": len(ran), "n_pass": n_pass, "all_pass": n_pass == len(ran),
               "seconds": round(time.time() - t0, 1), "checks": checks}
    outp = Path(a.out) if a.out else (OUT / ("validation_smoke.json" if "smoke" in str(raw_dir)
                                             else "validation.json"))
    outp.parent.mkdir(parents=True, exist_ok=True)
    tmp = outp.with_suffix(outp.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(outp)
    print("=" * 96)
    print(f"  {n_pass}/{len(ran)} checks passed in {payload['seconds']}s  ->  {outp}")
    print("=" * 96)
    return 0 if payload["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
