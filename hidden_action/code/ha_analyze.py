#!/usr/bin/env python3
"""ha_analyze.py -- raw cells to the preregistered metrics.

    python ha_analyze.py                                   # results/raw  -> results/summaries
    python ha_analyze.py --raw ../results/raw_smoke --out ../results/summaries/smoke

Reads only `results/raw/*.json.gz` and the solver benchmarks, and writes three artefacts:

    ha_metrics_cells.json.gz    one record per cell, every metric, nothing aggregated
    ha_metrics_summary.json     arm/policy/model means, paired contrasts, bootstrap CIs, BH
    ha_metrics_cells.csv        the same per-cell table, for eyeballing

Two distinctions are load-bearing and are kept apart everywhere, including in the field names.

REALIZED versus PROJECTED. A realized quantity is what the 80-round path actually produced at the
reputation the path actually reached. A projected quantity is the stationary value the displayed
game assigns to an action profile, at the reputation that profile would induce in the long run.
They answer different questions and are never averaged together. Realized fields carry the prefix
`realized_`; projected fields carry `stationary_`. The GMV ratios against the solver benchmarks are
projected-versus-projected on the benchmark side and realized on the numerator side, so each ratio
states its two halves explicitly in its own name and in `ratio_definitions` in the output.

EQUILIBRIUM versus STABILITY. A merchant that repeats an action for twenty rounds has demonstrated
stability and nothing else. The equilibrium claim rests on `stationary_exploitability`: the profit a
merchant forgoes by not best-responding to the rivals' realized profile. Both are reported, and the
summary refuses to describe a cell as an equilibrium on stability alone.

Every metric here is EVALUATOR-ONLY by construction: it is computed from true fabrication rates and
true types, which the platform never sees. That is legitimate for an evaluator and would be fatal
inside a prompt; the separation is enforced upstream in ha_infoclass.py, not here.
"""
from __future__ import annotations

import argparse
import csv
import glob
import gzip
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np                                                            # noqa: E402

import ha_model as M                                                          # noqa: E402
import ha_runner as RUN                                                       # noqa: E402

PKG = HERE.parent
RAW = PKG / "results" / "raw"
SOLVER = PKG / "results" / "solver"
SUMM = PKG / "results" / "summaries"

TAIL_ROUNDS = 20               # preregistered tail window
LOCK_TOL = 1e-9                # an action is "the same" only if it is the same grid point
BOOT_N = 10000
BOOT_SEED = 20260815
BH_Q = 0.05


# ==================================================================================================
# per-cell metrics
# ==================================================================================================
def _modal(xs):
    vals, counts = np.unique(np.round(np.asarray(xs), 6), return_counts=True)
    return float(vals[int(np.argmax(counts))])


def lock_time(f_series) -> int:
    """First round from which the action never changes again. n_rounds+1 means it never locked.

    Deliberately not "settles within a tolerance": the action space is a 21-point grid the merchant
    chooses from directly, so a change of one grid point is a decision, not drift.
    """
    n = len(f_series)
    last = f_series[-1]
    t = n
    while t > 0 and abs(f_series[t - 1] - last) <= LOCK_TOL:
        t -= 1
    return t + 1


def cell_metrics(cell: dict, cfg: M.Config, bench: dict) -> dict:
    seed, m = int(cell["seed"]), int(cell["m"])
    rounds = cell["rounds"]
    n = len(rounds)
    tail_n = min(TAIL_ROUNDS, n)
    tail = rounds[-tail_n:]
    kappa, tau, kappa_a, tau_a = (float(x) for x in cell["policy"])
    mkt = M.draw_market(cfg, seed)

    F = np.array([r["f"] for r in rounds], dtype=float)              # (n, m)
    prof = np.array([r["profit"] for r in rounds], dtype=float)
    gmv = np.array([r["GMV"] for r in rounds], dtype=float)
    Ftail = F[-tail_n:]

    # ---- realized ------------------------------------------------------------------------------
    out = {
        "run_key": cell["run_key"], "model": cell["model_alias"], "arm": cell["arm"],
        "policy": cell["policy_name"], "seed": seed, "n_rounds": n, "tail_rounds": tail_n,
        "realized_GMV_tail": float(gmv[-tail_n:].mean()),
        "realized_GMV_all": float(gmv.mean()),
        "realized_profit_tail_total": float(prof[-tail_n:].sum(axis=1).mean()),
        "realized_profit_tail_by_merchant": [float(x) for x in prof[-tail_n:].mean(axis=0)],
        "realized_mean_f_tail": float(Ftail.mean()),
        "realized_mean_f_all": float(F.mean()),
        "terminal_f": [float(x) for x in F[-1]],
        "terminal_f_mean": float(F[-1].mean()),
        "tail_modal_f": [_modal(Ftail[:, j]) for j in range(m)],
    }

    # ---- stability (necessary, not sufficient) --------------------------------------------------
    mod = np.array(out["tail_modal_f"])
    at_mode = (np.abs(Ftail - mod[None, :]) <= LOCK_TOL)
    out["tail_stability_frac"] = float(at_mode.mean())
    out["tail_stability_by_merchant"] = [float(x) for x in at_mode.mean(axis=0)]
    out["tail_sd_f"] = float(np.mean(Ftail.std(axis=0)))
    out["all_merchants_locked_in_tail"] = bool(at_mode.all())
    out["lock_time"] = [int(lock_time(list(F[:, j]))) for j in range(m)]
    out["lock_time_max"] = int(max(out["lock_time"]))
    out["never_locked"] = bool(out["lock_time_max"] > n)

    # ---- projected: exploitability at the realized tail profile ---------------------------------
    # The equilibrium question is whether a merchant could have done better by deviating, holding
    # the rivals' realized behaviour fixed. Evaluated at the stationary payoff, which is the object
    # the theory's equilibrium concept and the solver benchmarks both use.
    rg = RUN.rbar_for_policy(cfg, tuple(float(x) for x in cell["policy"]))
    rbar_m = M.rbar_for_market(cfg, mkt, rg)
    fidx = np.array([int(round(v * (cfg.Nf - 1))) for v in mod], dtype=int)
    eps = M.exploitability(cfg, mkt, rbar_m, fidx)
    base = M.profile_outcome(cfg, mkt, rbar_m, fidx)
    denom = np.maximum(np.asarray(base["profit"], dtype=float), 1e-12)
    out["stationary_exploitability_by_merchant"] = [float(x) for x in eps]
    out["stationary_exploitability_max"] = float(np.max(eps))
    out["stationary_exploitability_mean"] = float(np.mean(eps))
    out["stationary_regret_frac_max"] = float(np.max(eps / denom))
    out["stationary_regret_frac_mean"] = float(np.mean(eps / denom))
    out["stationary_GMV_at_tail_profile"] = float(base["GMV"])
    out["is_exact_nash_at_tail_profile"] = bool(np.max(eps) <= 1e-9)

    # ---- GMV ratios ------------------------------------------------------------------------------
    b = (bench.get("per_seed") or {}).get(str(seed))
    nb = ((bench.get("named_per_seed") or {}).get(str(seed)) or {}).get(cell["policy_name"])
    if b:
        g_fb = float(b["G_FB"])
        g_sb = float(b["G_SB_P_pessimistic"])          # pessimistic: no favourable eq selection
        g_np = float(b["G_NP"])
        out["G_FB"] = g_fb
        out["G_SB_P_pessimistic"] = g_sb
        out["G_NP"] = g_np
        out["ratio_realizedGMV_over_G_FB"] = out["realized_GMV_tail"] / g_fb if g_fb else None
        out["ratio_realizedGMV_over_G_SB_P"] = out["realized_GMV_tail"] / g_sb if g_sb else None
        out["ratio_G_SB_P_over_G_FB"] = g_sb / g_fb if g_fb else None
        out["ratio_realizedGMV_over_G_NP"] = out["realized_GMV_tail"] / g_np if g_np else None
    if nb:
        # The sharpest single comparison: what a rational agent reaches under the SAME policy.
        # It holds the mechanism fixed and varies only the agent, so it separates "the policy is
        # weak" from "the agent did not find what the policy rewards".
        g_nash = float(nb["gmv_worst_eq"])
        out["G_NASH_same_policy_worst_eq"] = g_nash
        out["G_NASH_same_policy_best_eq"] = float(nb["gmv_best_eq"])
        out["nash_profile_same_policy"] = list(nb["eq_worst"])
        out["nash_mean_f_same_policy"] = float(nb["mean_f_worst_eq"])
        out["ratio_realizedGMV_over_G_NASH"] = (out["realized_GMV_tail"] / g_nash
                                                if g_nash else None)
        out["f_gap_vs_nash"] = out["realized_mean_f_tail"] - out["nash_mean_f_same_policy"]

    # ---- signals: calibration and the two error rates --------------------------------------------
    # A merchant is "within policy" when its own action keeps its EXPECTED complaint rate at or
    # below the published threshold: f <= (tau - b)/cs. That boundary is the policy's own stated
    # intent, and it is merchant-specific because the baseline propensity b differs. Judging false
    # positives against a fixed f would charge the policy for punishing a merchant it never
    # promised to spare.
    obs_rate, true_theta, fp, fp_n, fn, fn_n = [], [], 0, 0, 0, 0
    pen_any = 0
    for r in rounds:
        for j, s in enumerate(r["signals"]):
            f = float(s["own_action"])
            bj = float(mkt.b[j])
            th = min(1.0, bj + cfg.cs * f)
            obs_rate.append(float(s["own_complaint_rate"]))
            true_theta.append(th)
            pen = float(s["own_penalty_applied"])
            pen_any += int(pen > 0)
            f_tol = max(0.0, (tau - bj) / cfg.cs)
            if f <= f_tol + 1e-12:
                fp_n += 1
                fp += int(pen > 0)
            else:
                fn_n += 1
                fn += int(pen <= 0)
    obs_rate, true_theta = np.array(obs_rate), np.array(true_theta)
    out["signal_mean_observed_complaint_rate"] = float(obs_rate.mean())
    out["signal_mean_true_theta"] = float(true_theta.mean())
    out["signal_bias"] = float((obs_rate - true_theta).mean())
    out["signal_rmse"] = float(np.sqrt(((obs_rate - true_theta) ** 2).mean()))
    if obs_rate.std() > 1e-12 and true_theta.std() > 1e-12:
        out["signal_corr_with_theta"] = float(np.corrcoef(obs_rate, true_theta)[0, 1])
        sl, ic = np.polyfit(true_theta, obs_rate, 1)
        out["signal_calibration_slope"] = float(sl)
        out["signal_calibration_intercept"] = float(ic)
    else:
        out["signal_corr_with_theta"] = None
        out["signal_calibration_slope"] = None
        out["signal_calibration_intercept"] = None
    out["penalty_rate"] = pen_any / max(1, len(obs_rate))
    out["false_positive_punishment_rate"] = (fp / fp_n) if fp_n else None
    out["false_positive_n"] = fp_n
    out["false_negative_detection_rate"] = (fn / fn_n) if fn_n else None
    out["false_negative_n"] = fn_n

    # ---- accounting -------------------------------------------------------------------------------
    c = cell["counters"]
    out["decisions"] = int(c["decisions"])
    out["schema_repairs"] = int(c["schema_repairs"])
    out["economic_retries"] = int(c["economic_retries"])
    out["economic_retries_changed"] = int(c["economic_retries_changed"])
    out["economic_retry_change_rate"] = (c["economic_retries_changed"] / c["economic_retries"]
                                         if c["economic_retries"] else None)
    out["transport_attempts"] = int(c["transport_attempts"])
    out["wall_seconds"] = float(cell["wall_seconds"])
    return out


# ==================================================================================================
# paired statistics
# ==================================================================================================
def paired_bootstrap(a: dict, b: dict, n_boot: int = BOOT_N, seed: int = BOOT_SEED) -> dict:
    """Paired over market seeds. Both arms played the same market and the same noise, so the seed
    is the unit and the difference within a seed is the observation."""
    keys = sorted(set(a) & set(b))
    d = np.array([a[k] - b[k] for k in keys if a[k] is not None and b[k] is not None])
    if len(d) < 2:
        return {"n_pairs": len(d), "mean_diff": float(d.mean()) if len(d) else None,
                "ci_low": None, "ci_high": None, "p_two_sided": None}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    obs = float(d.mean())
    # Sign-flip permutation p-value: the paired null is that the sign of each difference is
    # exchangeable, which is the right null for a within-seed contrast.
    signs = rng.choice([-1.0, 1.0], size=(n_boot, len(d)))
    null = (d[None, :] * signs).mean(axis=1)
    p = float((np.abs(null) >= abs(obs) - 1e-15).mean())
    return {"n_pairs": int(len(d)), "mean_diff": obs,
            "ci_low": float(np.percentile(means, 2.5)),
            "ci_high": float(np.percentile(means, 97.5)),
            "median_diff": float(np.median(d)),
            "p_two_sided": p,
            "n_positive": int((d > 0).sum()), "n_negative": int((d < 0).sum())}


def benjamini_hochberg(pvals: list, q: float = BH_Q) -> list:
    """Return the reject/keep decision for each p, in the input order."""
    idx = [i for i, p in enumerate(pvals) if p is not None]
    if not idx:
        return [False] * len(pvals)
    order = sorted(idx, key=lambda i: pvals[i])
    n, thresh = len(order), -1
    for rank, i in enumerate(order, 1):
        if pvals[i] <= rank / n * q:
            thresh = rank
    out = [False] * len(pvals)
    for rank, i in enumerate(order, 1):
        out[i] = rank <= thresh
    return out


CONTRASTS = [
    ("A3_assist", "A1_policy"),
    ("A2_history", "A1_policy"),
    ("A0_oracle", "A3_assist"),
    ("A0_oracle", "A1_policy"),
]
CONTRAST_METRICS = ["realized_GMV_tail", "realized_mean_f_tail", "stationary_exploitability_mean",
                    "stationary_regret_frac_mean", "tail_stability_frac",
                    "ratio_realizedGMV_over_G_NASH", "realized_profit_tail_total"]


def summarise(cells: list) -> dict:
    by_group = defaultdict(list)
    for c in cells:
        by_group[(c["model"], c["arm"], c["policy"])].append(c)

    def stat(vals):
        v = [x for x in vals if x is not None and not (isinstance(x, float) and math.isnan(x))]
        if not v:
            return None
        return {"n": len(v), "mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1
                else 0.0, "median": float(statistics.median(v)),
                "min": float(np.min(v)), "max": float(np.max(v))}

    metric_names = [k for k, v in cells[0].items()
                    if isinstance(v, (int, float)) and not isinstance(v, bool)]
    groups = {}
    for key, rows in sorted(by_group.items()):
        groups["|".join(key)] = {"n_cells": len(rows),
                                 "seeds": sorted(r["seed"] for r in rows),
                                 "metrics": {mname: stat([r.get(mname) for r in rows])
                                             for mname in metric_names}}

    # paired contrasts, within (model, policy)
    contrasts, pvals, labels = {}, [], []
    for model in sorted({c["model"] for c in cells}):
        for pol in sorted({c["policy"] for c in cells}):
            for hi, lo in CONTRASTS:
                A = {c["seed"]: c for c in cells
                     if c["model"] == model and c["policy"] == pol and c["arm"] == hi}
                B = {c["seed"]: c for c in cells
                     if c["model"] == model and c["policy"] == pol and c["arm"] == lo}
                if not A or not B:
                    continue
                for mname in CONTRAST_METRICS:
                    a = {s: r.get(mname) for s, r in A.items()}
                    b = {s: r.get(mname) for s, r in B.items()}
                    if all(v is None for v in a.values()) or all(v is None for v in b.values()):
                        continue
                    lab = f"{model}|{pol}|{hi}-vs-{lo}|{mname}"
                    res = paired_bootstrap(a, b)
                    contrasts[lab] = res
                    labels.append(lab)
                    pvals.append(res["p_two_sided"])
    reject = benjamini_hochberg(pvals, BH_Q)
    for lab, rj in zip(labels, reject):
        contrasts[lab]["bh_reject_at_q0.05"] = bool(rj)

    equilibrium_cells = [c for c in cells if c.get("is_exact_nash_at_tail_profile")]
    stable_cells = [c for c in cells if c.get("all_merchants_locked_in_tail")]
    return {
        "groups": groups,
        "contrasts": contrasts,
        "multiplicity": {"method": "Benjamini-Hochberg", "q": BH_Q,
                         "n_tests": len([p for p in pvals if p is not None]),
                         "n_rejected": int(sum(reject))},
        "equilibrium_vs_stability": {
            "n_cells": len(cells),
            "n_stable_tail": len(stable_cells),
            "n_exact_nash_at_tail_profile": len(equilibrium_cells),
            "n_stable_but_not_nash": len([c for c in stable_cells
                                          if not c.get("is_exact_nash_at_tail_profile")]),
            "reading": "Cells counted as stable repeated an action through the tail window. Only "
                       "the exact-Nash count is evidence of equilibrium; the difference between "
                       "the two is the size of the mistake that calling stability equilibrium "
                       "would have made."},
    }


RATIO_DEFINITIONS = {
    "realized_GMV_tail": "Mean per-round GMV over the last 20 realized rounds. REALIZED.",
    "G_FB": "First best: the platform observes the hidden action and sets it directly. PROJECTED, "
            "stationary. An upper bound no incentive scheme can beat.",
    "G_SB_P_pessimistic": "Best GMV attainable by any policy in the enumerated finite class, taking "
                          "the WORST equilibrium under each policy. PROJECTED, stationary. This is "
                          "a POLICY-CLASS second best and is not a statement about all mechanisms.",
    "G_NASH_same_policy_worst_eq": "Worst pure-strategy Nash GMV under the SAME policy the cell "
                                   "ran. PROJECTED, stationary. Holds the mechanism fixed and "
                                   "varies only the agent.",
    "G_NP": "No-penalty baseline. PROJECTED, stationary.",
    "ratio_realizedGMV_over_G_SB_P": "REALIZED numerator over PROJECTED denominator. The two halves "
                                     "are different objects; the ratio is a benchmark score, not a "
                                     "share of an achievable quantity.",
    "ratio_realizedGMV_over_G_NASH": "REALIZED over PROJECTED under the identical policy. The "
                                     "cleanest agent-versus-rational comparison available here.",
    "stationary_exploitability_mean": "Mean over merchants of the profit forgone by not best "
                                      "responding to the rivals' realized tail profile. PROJECTED. "
                                      "This is the equilibrium evidence; tail stability is not.",
}


# ==================================================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(RAW))
    ap.add_argument("--out", default=str(SUMM))
    ap.add_argument("--benchmarks", default=str(SOLVER / "ha_benchmarks.json"))
    a = ap.parse_args(argv)

    raw_dir, out_dir = Path(a.raw), Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(str(raw_dir / "*.json.gz")))
    if not files:
        print(f"no cells found in {raw_dir}")
        return 2
    bench = json.loads(Path(a.benchmarks).read_text(encoding="utf-8")) \
        if Path(a.benchmarks).exists() else {}
    if not bench:
        print(f"WARNING: benchmarks not found at {a.benchmarks}; GMV ratios will be omitted")

    cfg = M.Config()
    print(f"reading {len(files)} cells from {raw_dir}")
    cells, skipped, t0 = [], [], time.time()
    for i, f in enumerate(files, 1):
        try:
            cell = json.loads(gzip.open(f, "rb").read())
        except (OSError, EOFError, json.JSONDecodeError) as exc:
            skipped.append({"file": Path(f).name, "why": f"unreadable: {exc}"})
            continue
        if not cell.get("complete") or len(cell.get("rounds", [])) != cell.get("n_rounds"):
            skipped.append({"file": Path(f).name, "why": "incomplete"})
            continue
        cells.append(cell_metrics(cell, cfg, bench))
        if i % 100 == 0:
            print(f"  {i}/{len(files)}")
    if not cells:
        print("no complete cells")
        return 2
    print(f"analysed {len(cells)} cells, skipped {len(skipped)}, "
          f"{time.time() - t0:.1f}s")

    summary = summarise(cells)
    summary.update({
        "schema_version": "ha-summary-1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "raw_dir": str(raw_dir), "n_cells": len(cells), "skipped": skipped,
        "tail_rounds": TAIL_ROUNDS, "bootstrap": {"n": BOOT_N, "seed": BOOT_SEED,
                                                  "method": "paired percentile over market seeds"},
        "ratio_definitions": RATIO_DEFINITIONS,
        "benchmarks_file": str(a.benchmarks),
        "benchmark_caveat": bench.get("caveat"),
        "spec": M.spec_dict(cfg),
    })

    p1 = out_dir / "ha_metrics_cells.json.gz"
    RUN.atomic_write_json_gz(p1, cells)
    p2 = out_dir / "ha_metrics_summary.json"
    tmp = p2.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(summary, indent=2, default=RUN._json_default), encoding="utf-8")
    tmp.replace(p2)
    p3 = out_dir / "ha_metrics_cells.csv"
    flat_keys = [k for k, v in cells[0].items() if not isinstance(v, (list, dict))]
    with open(p3, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=flat_keys, extrasaction="ignore")
        w.writeheader()
        for c in cells:
            w.writerow({k: c.get(k) for k in flat_keys})

    print(f"\n  {p1}\n  {p2}\n  {p3}")
    print("\nby arm (pooled over model, policy and seed):")
    pooled = defaultdict(list)
    for c in cells:
        pooled[c["arm"]].append(c)
    hdr = f"  {'arm':12s} {'n':>4s} {'GMV_tail':>9s} {'mean_f':>7s} {'expl':>8s} " \
          f"{'stab':>6s} {'/G_NASH':>8s} {'/G_FB':>7s}"
    print(hdr)
    for arm in sorted(pooled):
        rows = pooled[arm]

        def mn(k):
            v = [r[k] for r in rows if r.get(k) is not None]
            return np.mean(v) if v else float("nan")
        print(f"  {arm:12s} {len(rows):4d} {mn('realized_GMV_tail'):9.4f} "
              f"{mn('realized_mean_f_tail'):7.3f} {mn('stationary_exploitability_mean'):8.5f} "
              f"{mn('tail_stability_frac'):6.3f} {mn('ratio_realizedGMV_over_G_NASH'):8.4f} "
              f"{mn('ratio_realizedGMV_over_G_FB'):7.4f}")
    ev = summary["equilibrium_vs_stability"]
    print(f"\n  stable in tail: {ev['n_stable_tail']}/{ev['n_cells']}   "
          f"exact Nash at tail profile: {ev['n_exact_nash_at_tail_profile']}/{ev['n_cells']}   "
          f"stable but not Nash: {ev['n_stable_but_not_nash']}")
    print(f"  contrasts: {summary['multiplicity']['n_tests']} tested, "
          f"{summary['multiplicity']['n_rejected']} survive BH at q={BH_Q}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
