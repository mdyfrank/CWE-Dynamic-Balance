#!/usr/bin/env python3
"""to_run.py -- the one entry point for the hidden-action experiment.

    python to_run.py --mode smoke          mock backend, no key, no network, tiny matrix
    python to_run.py --mode main --plan    print the matrix and the cost, write nothing
    python to_run.py --mode main           the preregistered matrix, resumable
    python to_run.py --mode main --resume  identical to the line above; kept because operators
                                           expect the word to exist

There is deliberately no second script. Every knob that could change what the experiment *is* --
the arms, the policies, the seed block, the horizon -- is read from the frozen manifest, and the
manifest's sha256 is written into the provenance record of every run. A flag can narrow the matrix
for a smoke test or a restart, but nothing on the command line can widen it past the manifest or
alter the design, so a run either matches the preregistration or is visibly a subset of it.

Resume is by cell, not by round. A cell is `(model, arm, policy, market seed)` played for the full
horizon, written once, atomically, at the end. `cell_done` re-reads and re-parses the file and
checks that the completion marker is present and the round count is exactly right, so a file
truncated by a kill signal is re-run rather than silently shrinking a denominator. Completed cells
are never re-called: the skip happens before the transport is touched.

Three failure counters exist and are never added together.

    transport_attempts   HTTP-level. A 503 and a retry. Says nothing about the merchant.
    schema_repairs       the reply would not parse; the merchant is asked again for JSON only,
                         with no new economic information. Budget 2, then the cell fails.
    economic_retries     an arm-A3 treatment: the platform offers a second decision opportunity.
                         This one IS an economic event and belongs in the results.

Conflating them is how a rate limit turns into a finding about model behaviour.

An action is never invented. If a reply cannot be parsed after the repair budget the cell fails and
is recorded as failed. Nothing defaults to honesty, to the previous round, or to the middle of the
grid, because each of those defaults is itself an economic claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np                                                            # noqa: E402

import ha_model as M                                                          # noqa: E402
import ha_prompts as PR                                                       # noqa: E402
import ha_runner as RUN                                                       # noqa: E402
import ha_transport as TR                                                     # noqa: E402

PKG = HERE.parent
MANIFEST_PATH = PKG / "manifests" / "ha_manifest_HA-M1.json"
RESULTS = PKG / "results"
PROGRESS_DIR = RESULTS / "progress"
ATTEMPTS_DIR = RESULTS / "attempts"

SMOKE_SEEDS = 2
SMOKE_ROUNDS = 4


# ==================================================================================================
# manifest
# ==================================================================================================
def load_manifest(path: Path = MANIFEST_PATH) -> tuple:
    """Return (manifest, sha256). The hash is of the file's bytes, so any edit is visible."""
    raw = path.read_bytes()
    return json.loads(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest()


def manifest_matrix(man: dict, tier: str = "A_full") -> dict:
    d = man["design"]
    t = man["tiers"][tier]
    sb = d["seed_block"]
    seeds = list(range(sb["start"], sb["start"] + int(t["seeds"])))
    return {"aliases": list(d["models"]), "arms": list(d["arms"]),
            "policies": list(d["policies"].keys()), "seeds": seeds,
            "rounds": int(t["rounds"]), "tier": tier}


def check_manifest_against_code(man: dict) -> list:
    """The manifest describes the design; the code implements it. Disagreement is a defect.

    Every mismatch found here would otherwise become a silent divergence between what was
    preregistered and what actually ran, discoverable only by someone reading both.
    """
    problems = []
    d = man["design"]
    if set(d["arms"]) != set(PR.ARMS):
        problems.append(f"arms: manifest {d['arms']} vs ha_prompts.ARMS {list(PR.ARMS)}")
    for name, p in d["policies"].items():
        coded = RUN.POLICY_NAMES.get(name)
        want = (p["kappa"], p["tau"], p["kappa_a"], p["tau_a"])
        if coded is None:
            problems.append(f"policy {name} is in the manifest but not in ha_runner.POLICY_NAMES")
        elif tuple(float(x) for x in coded) != tuple(float(x) for x in want):
            problems.append(f"policy {name}: manifest {want} vs code {coded}")
    cfg = M.Config()
    if int(d["merchants_per_market"]) != int(cfg.m):
        problems.append(f"m: manifest {d['merchants_per_market']} vs Config.m {cfg.m}")
    if int(d["action_grid"]["n"]) != int(cfg.Nf):
        problems.append(f"action grid: manifest {d['action_grid']['n']} vs Config.Nf {cfg.Nf}")
    if float(d["r_init"]) != float(RUN.R_INIT):
        problems.append(f"r_init: manifest {d['r_init']} vs ha_runner.R_INIT {RUN.R_INIT}")
    if int(d["rounds"]) > RUN.CRN_MAX_ROUNDS:
        problems.append(f"rounds {d['rounds']} exceeds CRN_MAX_ROUNDS {RUN.CRN_MAX_ROUNDS}")
    for tier, t in man["tiers"].items():
        if tier == "note":
            continue
        n = 2 * len(d["arms"]) * len(d["policies"]) * int(t["seeds"])
        if int(t["cells"]) != n:
            problems.append(f"tier {tier}: cells says {t['cells']}, design implies {n}")
        if int(t["base_decisions"]) != n * int(t["rounds"]) * int(d["merchants_per_market"]):
            problems.append(f"tier {tier}: base_decisions inconsistent with cells x rounds x m")
    sb = d["seed_block"]
    if sb["start"] + sb["count"] - 1 != sb["end_inclusive"]:
        problems.append("seed_block: end_inclusive does not match start + count - 1")
    return problems


# ==================================================================================================
# provenance
# ==================================================================================================
def git_info() -> dict:
    def g(*args):
        try:
            return subprocess.run(["git", *args], cwd=str(PKG), capture_output=True, text=True,
                                  timeout=15).stdout.strip() or None
        except Exception:                                                     # noqa: BLE001
            return None
    return {"commit": g("rev-parse", "HEAD"), "branch": g("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(g("status", "--porcelain"))}


def code_hashes() -> dict:
    out = {}
    for name in ("ha_model.py", "ha_infoclass.py", "ha_prompts.py", "ha_transport.py",
                 "ha_runner.py", "to_run.py"):
        p = HERE / name
        if p.exists():
            out[name] = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    return out


def provenance(mode: str, man: dict, man_sha: str, matrix: dict, transport) -> dict:
    return {
        "schema_version": "ha-provenance-1",
        "mode": mode,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "manifest_id": man["manifest_id"],
        "manifest_sha256": man_sha,
        "manifest_path": str(MANIFEST_PATH.relative_to(PKG.parent)),
        "matrix": {k: (v if k != "seeds" else {"n": len(v), "first": v[0], "last": v[-1]})
                   for k, v in matrix.items()},
        "git": git_info(),
        "code_sha256_prefix": code_hashes(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "spec": M.spec_dict(M.Config()),
        "prompt_hashes": PR.prompt_hashes(M.Config()),
        "crn": {"base": RUN.CRN_BASE, "max_rounds": RUN.CRN_MAX_ROUNDS},
        "retry_semantics": {
            "transport_attempts": f"HTTP level, up to {TR.TRANSPORT_RETRIES}, not an economic event",
            "schema_repairs": f"unparseable reply, budget {RUN.SCHEMA_REPAIR_BUDGET}, no new "
                              f"economic information, not an economic event",
            "economic_retries": "arm A3 treatment, a genuine second decision opportunity",
            "cell_retries": RUN.CELL_RETRY_BUDGET},
        "transport": transport.public(),
    }


# ==================================================================================================
# planning
# ==================================================================================================
def plan(matrix: dict, man: dict, out_root: Path) -> dict:
    specs = RUN.build_matrix(matrix["aliases"], matrix["arms"], matrix["policies"],
                             matrix["seeds"], matrix["rounds"])
    done = [s for s in specs if RUN.cell_done(s, out_root)]
    cfg = M.Config()
    n_todo = len(specs) - len(done)
    base = n_todo * matrix["rounds"] * cfg.m
    a3 = [s for s in specs if s.arm == "A3_assist" and not RUN.cell_done(s, out_root)]
    retry_ub = len(a3) * max(0, matrix["rounds"] - 2) * cfg.m
    return {"cells_total": len(specs), "cells_done": len(done), "cells_todo": n_todo,
            "base_decisions": base, "economic_retry_upper_bound": retry_ub,
            "api_calls_range": [base, base + retry_ub],
            "schema_repair_headroom": base * RUN.SCHEMA_REPAIR_BUDGET,
            "by_arm": {a: sum(1 for s in specs if s.arm == a and not RUN.cell_done(s, out_root))
                       for a in matrix["arms"]},
            "by_model": {a: sum(1 for s in specs if s.alias == a and not RUN.cell_done(s, out_root))
                         for a in matrix["aliases"]}}


def print_plan(p: dict, matrix: dict, man_sha: str, mode: str, out_root: Path) -> None:
    print(f"  mode              {mode}")
    print(f"  tier              {matrix['tier']}")
    print(f"  manifest sha256   {man_sha[:32]}...")
    print(f"  models            {', '.join(matrix['aliases'])}")
    print(f"  arms              {', '.join(matrix['arms'])}")
    print(f"  policies          {', '.join(matrix['policies'])}")
    print(f"  seeds             {len(matrix['seeds'])}  "
          f"({matrix['seeds'][0]}-{matrix['seeds'][-1]})")
    print(f"  rounds            {matrix['rounds']}")
    print(f"  output            {out_root}")
    print()
    print(f"  cells             {p['cells_total']}  "
          f"(done {p['cells_done']}, to run {p['cells_todo']})")
    print(f"  decisions         {p['base_decisions']:,}")
    print(f"  economic retries  <= {p['economic_retry_upper_bound']:,}  (arm A3, trigger-gated)")
    print(f"  API calls         {p['api_calls_range'][0]:,} .. {p['api_calls_range'][1]:,}")
    print(f"  cells to run      by arm {p['by_arm']}")
    print(f"                    by model {p['by_model']}")


# ==================================================================================================
# run
# ==================================================================================================
def build_transport(mode: str, args, prov_sink: list):
    ATTEMPTS_DIR.mkdir(parents=True, exist_ok=True)
    log = TR.AttemptLog(ATTEMPTS_DIR / ("attempts_smoke.jsonl" if mode == "smoke"
                                        else "attempts.jsonl"))
    if mode == "smoke":
        eps = TR.mock_endpoints(tuple(args.models)) if args.models else TR.mock_endpoints()
        hook = TR.make_mock_hook(args.mock_behaviour, args.mock_fail_rate, args.mock_seed)
        return TR.Transport(eps, log, http_hook=hook, on_identity=prov_sink.append)
    eps = TR.load_endpoints(tuple(args.models) if args.models else ("gemma", "llama"))
    return TR.Transport(eps, log, on_identity=prov_sink.append)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("smoke", "main"), default="smoke",
                    help="smoke uses the mock backend and never touches the network")
    ap.add_argument("--tier", choices=("A_full", "B_reduced"), default="A_full")
    ap.add_argument("--plan", action="store_true", help="print the matrix and cost, write nothing")
    ap.add_argument("--resume", action="store_true",
                    help="accepted and ignored; every run resumes by construction")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--policies", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="run at most N cells this invocation")
    ap.add_argument("--out", default=None, help="raw output root; defaults to results/raw")
    ap.add_argument("--stop-on-fail", action="store_true")
    ap.add_argument("--skip-invariants", action="store_true",
                    help="not recommended; the invariants are cheap and catch silent noise bugs")
    ap.add_argument("--mock-behaviour", default="rational",
                    choices=("rational", "honest", "greedy", "noisy", "malformed"))
    ap.add_argument("--mock-fail-rate", type=float, default=0.0)
    ap.add_argument("--mock-seed", type=int, default=0)
    args = ap.parse_args(argv)

    print("=" * 96)
    print("HIDDEN-ACTION EXPERIMENT  --  to_run.py")
    print("=" * 96)

    man, man_sha = load_manifest()
    problems = check_manifest_against_code(man)
    if problems:
        print("\nMANIFEST DOES NOT MATCH THE CODE:")
        for p in problems:
            print(f"  - {p}")
        print("\nRefusing to run. The preregistration and the implementation must agree.")
        return 2
    print(f"\nmanifest {man['manifest_id']} agrees with the code "
          f"({len(man['design']['arms'])} arms, {len(man['design']['policies'])} policies)")

    matrix = manifest_matrix(man, args.tier)
    if args.mode == "smoke":
        matrix["seeds"] = matrix["seeds"][:SMOKE_SEEDS]
        matrix["rounds"] = SMOKE_ROUNDS
        matrix["tier"] = "smoke"

    # Narrowing only. A flag may select a subset of the manifest; it may never add to it.
    for key, val, allowed in (("aliases", args.models, matrix["aliases"]),
                              ("arms", args.arms, matrix["arms"]),
                              ("policies", args.policies, matrix["policies"])):
        if val:
            extra = [v for v in val if v not in allowed]
            if extra:
                print(f"\n{key}: {extra} are not in the manifest. The command line can narrow the "
                      f"matrix, never widen it.")
                return 2
            matrix[key] = list(val)
    if args.seeds:
        extra = [s for s in args.seeds if s not in matrix["seeds"]]
        if extra:
            print(f"\nseeds {extra} are outside the frozen block "
                  f"{man['design']['seed_block']['start']}-"
                  f"{man['design']['seed_block']['end_inclusive']}.")
            return 2
        matrix["seeds"] = list(args.seeds)
    if args.rounds:
        if args.rounds > matrix["rounds"]:
            print(f"\nrounds {args.rounds} exceeds the tier horizon {matrix['rounds']}.")
            return 2
        matrix["rounds"] = args.rounds

    out_root = Path(args.out) if args.out else (RESULTS / ("raw_smoke" if args.mode == "smoke"
                                                           else "raw"))
    out_root.mkdir(parents=True, exist_ok=True)

    if not args.skip_invariants:
        print("\ninvariants")
        inv = RUN.run_invariants()
        for c in inv["checks"]:
            print(f"  [{'PASS' if c['pass'] else 'FAIL'}] {c['check']}")
        if not inv["all_pass"]:
            print("\nInvariants failed. The noise process is not what the theory says it is.")
            return 2

    print("\nplan")
    p = plan(matrix, man, out_root)
    print_plan(p, matrix, man_sha, args.mode, out_root)
    if args.plan:
        print("\n--plan given: nothing was written.")
        return 0

    if args.mode == "main":
        missing = [v for v in ("HA_ENDPOINT", "HA_API_KEY") if not os.environ.get(v)]
        if missing and not os.environ.get("HA_ENDPOINT_GEMMA"):
            print(f"\nmissing environment: {', '.join(missing)}")
            print("See README.md. Credentials are never stored in this repository.")
            return 2

    identity = []
    transport = build_transport(args.mode, args, identity)
    progress = RUN.Progress(PROGRESS_DIR if args.mode == "main" else PROGRESS_DIR / "smoke")
    prov = provenance(args.mode, man, man_sha, matrix, transport)
    prov["plan"] = p
    prov_path = progress.root / f"provenance_{args.mode}.json"
    prov_path.parent.mkdir(parents=True, exist_ok=True)
    prov_path.write_text(json.dumps(prov, indent=2, default=RUN._json_default), encoding="utf-8")
    print(f"\nprovenance -> {prov_path}")

    specs = RUN.build_matrix(matrix["aliases"], matrix["arms"], matrix["policies"],
                             matrix["seeds"], matrix["rounds"])
    if args.limit:
        todo = [s for s in specs if not RUN.cell_done(s, out_root)]
        specs = [s for s in specs if RUN.cell_done(s, out_root)] + todo[:args.limit]
        print(f"--limit {args.limit}: this invocation will attempt at most {args.limit} cells")

    print(f"\nrunning {args.mode} ...")
    t0 = time.time()
    summary = RUN.run_matrix(M.Config(), specs, transport, progress, out_root,
                             stop_on_fail=args.stop_on_fail)
    summary["identity_verdicts"] = identity
    summary["manifest_sha256"] = man_sha
    summary["mode"] = args.mode
    summary["out_root"] = str(out_root)
    progress.snapshot(summary)

    print("\n" + "=" * 96)
    print(f"  cells      {summary['completed']} completed, {summary['failed']} failed, "
          f"{summary['skipped_complete']} already done")
    print(f"  transport  {summary['transport']}")
    print(f"  wall       {time.time() - t0:.1f} s")
    print(f"  raw        {out_root}")
    print(f"  progress   {progress.root}")
    if summary["failed"]:
        print(f"\n  failures ({len(summary['failures'])}):")
        for f in summary["failures"][:10]:
            print(f"    {f['run_key']}: {f['error'][:120]}")
        frac = summary["failed"] / max(1, summary["attempted"])
        if frac > 0.05:
            print(f"\n  {frac:.1%} of attempted cells failed, above the preregistered 5% abort "
                  f"threshold. Report the cause rather than analysing the remainder.")
    print("=" * 96)
    if args.mode == "smoke":
        print("\nsmoke complete. Next, from the root of the repository:")
        print("  python code/ha_analyze.py --raw results/raw_smoke "
              "--out results/summaries/smoke")
        print("  python code/ha_validate.py --raw results/raw_smoke "
              "--summaries results/summaries/smoke")
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
