#!/usr/bin/env python3
"""ha_dp_merge.py -- fold the dynamic-DP shards into one artefact, without inventing coverage.

    python code/ha_dp_merge.py results/solver/_dp_shard_*.json

`ha_dynamic_dp.py` costs about two CPU-hours per (seed, policy, delta), so the run is split: one
shard sweeps all 60 seeds at the headline delta and pays for the 80-round backward induction, a
second sweeps a 15-seed subset at the neighbouring deltas and skips it. Concatenating the two would
produce an aggregate whose denominator silently changes from block to block, which is exactly the
kind of quiet mis-statement the rest of this package exists to prevent.

So the merge is per-BLOCK, not per-file. Every `gamma_*` key gets its own `n_seeds`, its own seed
list, and its own aggregate recomputed from the merchant records that actually exist for it. Where
two shards report the same (seed, merchant, block) the values must agree bit for bit -- they were
produced by the same deterministic code from the same seed, so any difference means a shard is stale
and the merge refuses rather than picks a winner.

No API, no network, pure stdlib.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
OUT = PKG / "results" / "solver" / "ha_dynamic_equilibrium.json"

# fields that are reduced with max / min / sum over merchant records; anything not listed is carried
# per merchant but not aggregated, deliberately -- an aggregate nobody defined is worse than none
MAXES = ("eps_at_x0", "rel_at_x0", "eps_max", "rel_max", "eps_max_reachable", "rel_max_reachable",
         "eps_mean_under_stationary", "rel_mean_under_stationary")
COUNTS = ("build_then_exploit", "first_round_action_differs_from_f_star")


def _agg(recs: list) -> dict:
    """Aggregate one block over the merchant records that exist for it."""
    out = {"n_merchant_cases": len(recs)}
    for k in MAXES:
        v = [r[k] for r in recs if k in r]
        if v:
            out[f"max_{k}"] = float(max(v))
    if any("eps_min" in r for r in recs):
        out["min_eps"] = float(min(r["eps_min"] for r in recs if "eps_min" in r))
    out["n_cases_with_gain_at_x0"] = int(sum(1 for r in recs if r.get("eps_at_x0", 0.0) > 1e-12))
    out["n_cases_with_gain_anywhere_reachable"] = int(
        sum(1 for r in recs if r.get("eps_max_reachable", 0.0) > 1e-12))
    for k in COUNTS:
        if any(k in r for r in recs):
            out[f"n_{k}"] = int(sum(1 for r in recs if r.get(k)))
    if any("max_distinct_actions_over_states" in r for r in recs):
        out["max_distinct_actions_over_states"] = int(
            max(r["max_distinct_actions_over_states"] for r in recs
                if "max_distinct_actions_over_states" in r))
    if any("converged" in r for r in recs):
        out["all_converged"] = bool(all(r.get("converged", True) for r in recs))
        out["max_residual"] = float(max(r.get("eps_residual", 0.0) for r in recs))
    # `level_residual` postdates most of this artefact: the records solved before the eps-only
    # stopping rule was corrected do not carry it. A max over only the records that HAVE it would be
    # a residual for the whole block computed from a fraction of it -- the exact denominator-swap
    # this merge exists to prevent -- so the count travels with the number and the flag says whether
    # the block is covered. A reader who needs a converged denominator on the uncovered records
    # should use the two-route agreement check in ha_validate.py, which covers all of them.
    lv = [r["level_residual"] for r in recs if "level_residual" in r]
    if lv:
        out["max_level_residual"] = float(max(lv))
        out["n_with_level_residual"] = len(lv)
        out["level_residual_covers_the_block"] = bool(len(lv) == len(recs))
    return out


def merge(paths: list, out_path: Path, supersede: list = None) -> dict:
    """`supersede` names shards allowed to WIN a conflict instead of causing the merge to refuse.

    Refusing on conflict is the right default and it has one legitimate exception: a bug in the
    solver is fixed and the instances it touched are re-solved. The old values are then not stale by
    accident, they are wrong on purpose, and the merge must be able to say so. Every field a
    superseding shard replaces is recorded in `superseded` with both values, so the artefact carries
    the correction rather than quietly presenting the new number as if it had always been there.
    """
    supersede = set(supersede or [])
    shards, res = [], {"schema_version": "ha-dynamic-equilibrium-merged-1",
                       "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                       "script": "code/ha_dp_merge.py", "policies": {}}
    pols: dict = {}
    conflicts, replaced = [], []
    # superseding shards are folded in last, so the record they replace is already present and the
    # `superseded` entry can quote the value that was there rather than only the one that won
    for p in sorted(paths, key=lambda q: Path(q).name in supersede):
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        wins = Path(p).name in supersede
        shards.append({"file": str(Path(p).name), "generated": d.get("generated"),
                       "deltas": d.get("deltas"),
                       "policies": sorted(d.get("policies", {})),
                       # a repair shard is a partial seed record by construction, so its provenance
                       # has to travel with it: which instances it covers, and whether it was given
                       # the authority to overwrite what was already there
                       "repair": d.get("repair"), "supersedes": bool(wins),
                       "n_seeds": {k: v.get("n_seeds") for k, v in d.get("policies", {}).items()}})
        for key in ("horizon", "r_init", "state_space", "spec"):
            if key in d:
                res.setdefault(key, d[key])
        for pname, blk in d.get("policies", {}).items():
            tgt = pols.setdefault(pname, {"policy": blk.get("policy"), "seeds": {}})
            for row in blk.get("per_seed", []):
                s = int(row["seed"])
                cur = tgt["seeds"].get(s)
                if cur is None:
                    tgt["seeds"][s] = json.loads(json.dumps(row))
                    continue
                if row.get("skipped") or cur.get("skipped"):
                    continue
                # paired by merchant INDEX, not by position: a repair shard carries only the
                # instances that were re-solved, so `zip` over two lists of different lengths would
                # write merchant 2's corrected values on top of merchant 0's record
                byidx = {a["merchant"]: a for a in cur["merchants"]}
                for b in row["merchants"]:
                    a = byidx.get(b["merchant"])
                    if a is None:
                        cur["merchants"].append(json.loads(json.dumps(b)))
                        cur["merchants"].sort(key=lambda r: r["merchant"])
                        continue
                    for k, v in b.items():
                        if k not in a:
                            a[k] = v
                        elif a[k] != v and wins:
                            replaced.append({"seed": s, "merchant": b.get("merchant"), "field": k,
                                             "by": Path(p).name, "was": a[k], "now": v})
                            a[k] = v
                        elif a[k] != v:
                            conflicts.append({"seed": s, "merchant": b.get("merchant"), "field": k})

    for pname, tgt in pols.items():
        rows = [tgt["seeds"][s] for s in sorted(tgt["seeds"])]
        good = [r for r in rows if not r.get("skipped")]
        blocks = sorted({k for r in good for m in r["merchants"] for k in m
                         if k.startswith("gamma_")})
        agg, cov = {}, {}
        for b in blocks:
            recs = [m[b] for r in good for m in r["merchants"] if b in m]
            seeds_b = sorted({r["seed"] for r in good if any(b in m for m in r["merchants"])})
            agg[b] = _agg(recs)
            cov[b] = {"n_seeds": len(seeds_b), "seeds": [seeds_b[0], seeds_b[-1]] if seeds_b else [],
                      "complete_over_the_seed_block": len(seeds_b) == len(good)}
        res["policies"][pname] = {"policy": tgt["policy"], "n_seeds": len(good),
                                  "n_skipped": len(rows) - len(good),
                                  "coverage_by_block": cov, "aggregate": agg, "per_seed": rows}

    res["shards"] = shards
    res["merge_conflicts"] = conflicts
    res["superseded"] = replaced
    res["note"] = ("every gamma_* block carries its own n_seeds: the 80-round block and the delta "
                   "blocks do NOT cover the same seeds, and "
                   "`coverage_by_block.complete_over_the_seed_block` is the flag to read before "
                   "quoting any aggregate")
    if conflicts:
        raise SystemExit(f"REFUSING to merge: {len(conflicts)} conflicting fields, e.g. "
                         f"{conflicts[:3]} -- a shard is stale, re-run it")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("shards", nargs="+")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--supersede", default="", help="comma-separated shard FILENAMES that win a "
                                                    "conflict instead of causing a refusal")
    a = ap.parse_args(argv)
    res = merge(a.shards, Path(a.out), [v for v in a.supersede.split(",") if v])
    if res["superseded"]:
        by = {r["by"] for r in res["superseded"]}
        print(f"superseded {len(res['superseded'])} fields from {sorted(by)}")
    for pname, blk in res["policies"].items():
        print(f"{pname}: {blk['n_seeds']} seeds, {blk['n_skipped']} skipped")
        for b, c in blk["coverage_by_block"].items():
            g = blk["aggregate"][b]
            flag = "" if c["complete_over_the_seed_block"] else "   <- PARTIAL"
            print(f"   {b:<20} {c['n_seeds']:>3} seeds  {g['n_merchant_cases']:>4} cases   "
                  f"max rel@x0 {g.get('max_rel_at_x0', float('nan')):.4%}   "
                  f"max rel reachable {g.get('max_rel_max_reachable', float('nan')):.4%}{flag}")
    p = Path(a.out).resolve()
    print(f"wrote {p.relative_to(PKG) if p.is_relative_to(PKG) else p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
