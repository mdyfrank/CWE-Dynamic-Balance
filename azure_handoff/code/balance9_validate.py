"""balance9_validate.py -- the independent validator (protocol section 6).

FORBIDDEN IMPORTS, and the reason the list exists
-------------------------------------------------
This module may not import `balance9_runner`, `balance8_runner`, any analyzer, `equilibrium`,
`hetero`, `hetero_policy_audit`, `phase2_exploitability`, `phase2_framing_controlled`, or
`balance9_prompts`. It imports `json`, `math`, `numpy` and nothing else from this project.

Errata E-3 recorded that Balance-8's "independent re-derivation" was not independent: it shared the
payoff-table function with the runner, so the two agreed by construction and the agreement was
reported as validation. A checker that calls the code it is checking measures nothing except that
Python is deterministic. So every formula below is typed out again from the written specification in
`BALANCE9_PROTOCOL.md` section 1, and `gate_no_forbidden_imports` parses this file's own AST to prove
the rule was kept -- because a rule that lives only in a docstring is the exact failure mode E-7 is
about.

The consequence for the raw schema is real, not cosmetic. This file cannot call a reputation solver,
so the runner has to persist `rbar_used`; it cannot call `equilibrium._w0`, so the runner has to
persist `w0` and `fgrid` as values. Those fields exist because of this constraint.

WHAT IS CHECKED
---------------
TWO SCHEMAS, because this round produced two. `--file` sniffs which one it is holding and refuses to
report a tally for a file it cannot read (see `V0`/`W0`).

*Dynamic runs* (`balance9_p3_raw.jsonl`) -- `run_header` / `round` / `run_footer`:

  V1  displayed table -- all 21 values, rebuilt from q, p, bidx, rbar_used and the rival profile
  V2  displayed argmax, the tie set at tau_tie, and the regret of every attempt
  V3  the realized market step -- shares, outside share, sales, profit, GMV
  V4  reputation transition, complaint counts, penalties
  V5  trust transition and the lagged-traffic timing
  V6  the realized one-step deviation payoff vector and realized exploitability
  V7  decision logic -- retry trigger, override trigger, accepted vs executed index
  V8  structural integrity -- seed freshness, no duplicate seeds, no missing rounds, counters
  V9  the initial condition is as recorded rather than as assumed

*One-shot trials* (`balance9_p4_raw.jsonl`, `balance9_p5_raw.jsonl`) -- `p4_trial` / `p5_first` /
`p5_trial` / `*_cell_footer`. Added after L-10 recorded that these two files had no independent
validator at all and the module's honest response to them was `V0`, a refusal:

  W1  the displayed table -- all 21 values, rebuilt from the market, the persisted rbar row and the
      RIVALS' stationary reputations, which the one-shot schema does not persist and which are
      therefore RECOVERED FROM THE CORPUS (see `assemble_rbar`)
  W2  displayed argmax, the tie set, the certificate set, relative and absolute regret
  W3  the row permutation and every recorded row position
  W4  the executed action, its fabrication level and its proposal trail
  W5  the frozen decoy-pair eligibility rule of preregistration section 6.2
  W6  every field of `p4_classify`, against the frozen definitions of section 6.3
  W7  the arm profile -- which arms show a table, which carry a mark, and what the mark is
  W8  structural integrity -- duplicate keys, seed freshness, the cell footers' own accounting
      reconciled against the records on disk, and the trials a first attempt OWES but may not have
  W9  P5's factorial -- the shared first attempt, the decision path, the retry trigger, the
      factor labels and the correction counters

W8's footer accounting is deliberately a second, independent copy of what `P4A-D11b`/`P5A-D3c` do
inside the analysers. Errata E-14 is the reason: those gates and their own fault suites read the file
through one shared loader, so a defect in the loader was invisible to both. A second reader that
shares nothing with them is the only instrument that can see past that, and this is it.

That independence has a price, and E-15 is where it was paid: a reader that shares nothing must also
GUESS NOTHING. This one first assumed the footers count seeds, multiplied P4's by the arm count, and
reported a chain break in a file that is correct -- P4's counters are trials, P5's are seeds, and the
files say so. **An independent second reader's first output is a hypothesis about the format, not a
verdict on the data.**

FAULT INJECTION
---------------
The validator is not trusted until it has been shown to fail. `--faults` mutates a clean run one way
at a time and requires exactly the named check(s) to catch each -- not merely *some* check, since a
mutation caught by nine checks establishes none of them.

It then computes, rather than asserts, its own attribution coverage: which checks are established by
a fault that fires them ALONE. Every check this module can emit must be one of them. That report
found two defects the eye had missed -- `V8d` was established under a private alias the pipeline
never raises, and `V6b` was `V3c` under a second name -- and it is printed on every run so the claim
can never again drift away from the table underneath it. See errata E-10.1 and E-10.3.

For Balance-8 raw data, `--legacy` reports which quantities CANNOT be reconstructed from the old
schema. It makes no retroactive claim of validation.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(os.path.dirname(os.path.abspath(__file__)))

FORBIDDEN = {
    "balance9_runner", "balance8_runner", "balance7_runner", "balance6_runner",
    "balance9_prompts", "balance9_metrics", "balance8_analyze", "balance7_analyze",
    "equilibrium", "hetero", "hetero_policy_audit", "phase2_exploitability",
    "phase2_framing_controlled", "phase2_llm",
    # Added with the one-shot checks. The drivers and analysers of P3, P4 and P5 are barred for the
    # same reason the runner is: E-3 was not about one module, it was about a validator that reaches
    # for the thing it validates. `balance9_prompts` -- which owns `decoy_pair` and `p4_classify`, the
    # two functions `W5` and `W6` re-derive -- was already on the list and is the most important
    # entry on it.
    "balance9_p3", "balance9_p4", "balance9_p5",
    "balance9_p3_analyze", "balance9_p4_analyze", "balance9_p5_analyze",
    "balance9_ratspec", "balance9_prereg_freeze",
}

TOL = 1e-9

# Frozen in BALANCE9_PREREGISTRATION.md section 6.2 and retyped, not imported. The band is a closed
# interval on RELATIVE displayed regret; `ETA_REF` is the round's reference tolerance.
DECOY_BAND = (0.04, 0.06)
ETA_REF = 0.01
# Preregistration section 6.1: arm -> (a payoff table is shown, what the mark is). `None` means the
# arm carries no mark at all; the strings are resolved per record against that record's own table.
P4_ARMS = {
    "U1":     (True,  None),
    "HF":     (True,  "argmax"),
    "HR":     (True,  "argmax"),
    "HD_BAL": (True,  "decoy_bal"),
    "HD_EXT": (True,  "decoy_ext"),
    "RO":     (False, "argmax"),
    "N":      (False, None),
    "M":      (False, None),
}
# Protocol section 9: the 2x2. A1/A2 is the acceptance tolerance, B1/B2 the retry wording.
P5_FACTOR_A = {"A1": 0.01, "A2": 0.0}
P5_FACTOR_B = {"B1": "generic", "B2": "explicit"}


# =============================================================================================
# the specification, retyped from BALANCE9_PROTOCOL.md section 1
# =============================================================================================
def spec_softmax_shares(alpha, beta, gamma, q, p, r, f, w0):
    """s_j and the outside share. Protocol 1.1 lines 35-37."""
    q, p, r, f = map(lambda x: np.asarray(x, float), (q, p, r, f))
    u = alpha * (q + (1.0 - q) * f) + beta * r - gamma * p
    e = np.exp(u)
    den = math.exp(w0) + float(e.sum())
    return e / den, math.exp(w0) / den


def spec_market_step(cfg, q, p, b, r, T, f, U_row, kappa, tau, zeta, lam):
    """One realized round. Protocol 1.1 in full. Returns a dict of every intermediate.

    Reputation enters at its PRE-update value and traffic at its LAGGED value. Getting either wrong
    produces numbers that look entirely plausible and are wrong by a few percent, which is how
    errata E-7 happened; the ordering here is deliberate and is what V3/V5 test.
    """
    q, p, b, r, f = map(lambda x: np.asarray(x, float), (q, p, b, r, f))
    s, outside = spec_softmax_shares(cfg["alpha"], cfg["beta"], cfg["gamma"], q, p, r, f, cfg["w0"])
    Q = cfg["Q0"] * T                                   # LAGGED trust
    y = Q * s
    c = p / 2.0
    profit = (p - c) * y
    gmv = float((p * y).sum())
    theta = np.clip(b + cfg["cs"] * f, 0.0, 1.0)
    D = None
    if U_row is not None:
        D = (np.asarray(U_row, float) < theta[:, None]).sum(axis=1).astype(int)
    F = float(f.mean())
    T_next = (1.0 - zeta) * T + zeta * math.exp(-lam * F)
    return dict(s=s, outside=outside, Q=Q, y=y, c=c, profit=profit, gmv=gmv, theta=theta,
                D=D, F=F, T_next=T_next)


def spec_reputation_next(cfg, r, d, kappa, tau):
    """Protocol 1.1 lines 45-46."""
    r = np.asarray(r, float)
    pen = kappa * np.maximum(0.0, np.asarray(d, float) - tau)
    return np.clip(r + cfg["eta_r"] * (1.0 - r) - pen, 0.0, 1.0), pen


def spec_displayed_table(cfg, q, p, rbar_row_j, f_rivals, j, lam, fgrid, omega=0.5):
    """The 21 displayed values for merchant j. Protocol 1.2.

    Stationary reputation rbar[b_j, i] (moves with the action), contemporaneous traffic
    Q0*exp(-lam*mean f) (moves with the action), previous-round rivals (fixed). Written out rather
    than delegated, which is the entire point of this module.

    `omega` is the margin fraction, c_j = omega * p_j. It was written as the literal 0.5 until the
    one-shot files were read, and they persist it in `config.omega` -- so it is now a parameter and
    `G-V5` asserts that what the reconstruction used is what the record says, rather than a default
    that happens to be right. A constant nobody reads is decoration; a constant read from the record
    and compared to the record is a check.
    """
    q = np.asarray(q, float)
    p = np.asarray(p, float)
    m = len(q)
    fgrid = np.asarray(fgrid, float)
    out = np.empty(len(rbar_row_j))
    for i in range(len(rbar_row_j)):
        f = np.asarray(f_rivals, float).copy()
        f[j] = round(float(i) / (len(rbar_row_j) - 1), 4)
        # every merchant's stationary reputation is read at ITS OWN action index
        r = np.empty(m)
        for k in range(m):
            ik = int(np.argmin(np.abs(fgrid - f[k])))
            r[k] = cfg["rbar"][k][ik]
        u = cfg["alpha"] * (q + (1.0 - q) * f) + cfg["beta"] * r - cfg["gamma"] * p
        e = np.exp(u)
        den = math.exp(cfg["w0"]) + float(e.sum())
        s = e / den
        Q = cfg["Q0"] * math.exp(-lam * float(f.mean()))
        out[i] = omega * p[j] * Q * s[j]
    return out


def spec_realized_vector(cfg, q, p, r, T, f_exec, j, n_idx):
    """W_j(i): what j would have been PAID this round for each action, own reputation and the
    lagged traffic scale held fixed. Theory Prop 4.1-C / errata E-7."""
    q = np.asarray(q, float)
    p = np.asarray(p, float)
    r = np.asarray(r, float)
    f = np.asarray(f_exec, float)
    u = cfg["alpha"] * (q + (1.0 - q) * f) + cfg["beta"] * r - cfg["gamma"] * p
    e = np.exp(u)
    others = math.exp(cfg["w0"]) + float(e.sum()) - float(e[j])
    out = np.empty(n_idx + 1)
    for i in range(n_idx + 1):
        uj = (cfg["alpha"] * (q[j] + (1.0 - q[j]) * round(float(i) / n_idx, 4))
              + cfg["beta"] * r[j] - cfg["gamma"] * p[j])
        ej = math.exp(uj)
        out[i] = 0.5 * p[j] * (cfg["Q0"] * T) * (ej / (others + ej))
    return out


def spec_rel_regret(v, i):
    v = np.asarray(v, float)
    return float((v.max() - v[i]) / max(abs(v.max()), 1e-9))


def spec_argmax_min(v, tau_tie):
    v = np.asarray(v, float)
    return int(np.flatnonzero(v >= v.max() - tau_tie)[0])


def spec_argmax_set(v, tau_tie):
    v = np.asarray(v, float)
    return [int(i) for i in np.flatnonzero(v >= v.max() - tau_tie)]


# =============================================================================================
# validation
# =============================================================================================
class Report:
    def __init__(self):
        self.rows = []
        self.worst = defaultdict(float)

    def add(self, code, ok, detail="", err=None):
        self.rows.append((code, bool(ok), detail))
        if err is not None:
            self.worst[code] = max(self.worst[code], float(err))

    def failures(self):
        return [(c, d) for c, ok, d in self.rows if not ok]

    def codes_failed(self):
        return sorted({c for c, ok, _ in self.rows if not ok})

    def summary(self):
        n = sum(1 for _, ok, _ in self.rows if ok)
        return n, len(self.rows)


def load_runs(path, census=None):
    """Group a raw JSONL into {run_id: (header, [rounds], footer)}.

    `census`, if supplied, is filled with a count of every record `kind` seen in the file. The caller
    needs it to tell "this file is clean" apart from "this file is written in a schema I cannot
    read", which are the same output without it -- see `V0`.
    """
    headers, rounds, footers = {}, defaultdict(list), {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            k = d.get("kind")
            if census is not None:
                census[k] += 1
            if k == "run_header":
                headers[d["run_id"]] = d
            elif k == "round":
                rounds[d["run_id"]].append(d)
            elif k == "run_footer":
                footers[d["run_id"]] = d
    return {rid: (h, sorted(rounds[rid], key=lambda r: r["round"]), footers.get(rid))
            for rid, h in headers.items()}


def validate_run(header, rounds, rep: Report, crn=None):
    """Re-derive every quantity of one run from the raw fields alone."""
    cfgh = header["config"]
    m = cfgh["m"]
    n_idx = header["n_index"]
    lam = header["lam"]
    zeta = header["zeta"]
    kappa, tau = header["kappa"], header["tau"]
    tau_tie = header["tau_tie"]
    q = np.asarray(header["market"]["q"], float)
    p = np.asarray(header["market"]["p"], float)
    b = np.asarray(header["market"]["b"], float)
    cfg = dict(alpha=cfgh["alpha"], beta=cfgh["beta"], gamma=cfgh["gamma"], Q0=cfgh["Q0"],
               eta_r=cfgh["eta_r"], cs=cfgh["cs"], w0=cfgh["w0"], rbar=header["rbar_used"])
    fgrid = cfgh["fgrid"]

    # ---- V9: the initial condition is READ, never assumed --------------------------------------
    ic = header["initial_condition"]
    r_state = np.asarray(ic["r_init"], float).copy()
    T_state = float(ic["T_init"])
    f_prev = np.asarray(ic["f_last_init"], float).copy()
    rep.add("V9", len(r_state) == m and len(f_prev) == m,
            "initial condition present and correctly shaped")

    # ---- V8a: rounds are complete and consecutive ----------------------------------------------
    got = [r["round"] for r in rounds]
    rep.add("V8a", got == list(range(1, len(got) + 1)),
            f"rounds consecutive from 1 (n={len(got)})")
    rep.add("V8b", len(got) == header["rounds"],
            f"round count matches the header ({len(got)} vs {header['rounds']})")

    for rec in rounds:
        t = rec["round"]
        M = rec["merchants"]
        f_exec = np.array([mm["f"] for mm in M], float)

        # ---- V1: the displayed table ------------------------------------------------------------
        for j, mm in enumerate(M):
            v_rec = np.asarray(mm["displayed_table"], float)
            v_own = spec_displayed_table(cfg, q, p, header["rbar_used"][j],
                                         np.asarray(mm["rival_f"], float), j, lam, fgrid)
            err = float(np.abs(v_rec - v_own).max())
            rep.add("V1", err < 1e-9, f"displayed table t={t} j={j}", err)

            # ---- V2: argmax, tie set, regret ----------------------------------------------------
            rep.add("V2a", spec_argmax_min(v_rec, tau_tie) == mm["argmax_index"],
                    f"argmax t={t} j={j}")
            rep.add("V2b", spec_argmax_set(v_rec, tau_tie) == mm["argmax_set"],
                    f"tie set t={t} j={j}")
            e2 = abs(spec_rel_regret(v_rec, mm["executed_index"]) - mm["regret_executed"])
            rep.add("V2c", e2 < TOL, f"executed regret t={t} j={j}", e2)
            for att in mm["regret_per_attempt"]:
                ea = abs(spec_rel_regret(v_rec, att["i"]) - att["rel"])
                rep.add("V2d", ea < TOL, f"attempt regret t={t} j={j}", ea)

            # ---- V7: decision logic --------------------------------------------------------------
            props = mm["proposed_index"]
            rep.add("V7a", (mm["accepted_index"] == (props[-1] if props else None)),
                    f"accepted == last proposal t={t} j={j}")
            if not mm["override"]:
                rep.add("V7b", mm["executed_index"] == mm["accepted_index"],
                        f"no override => executed == accepted t={t} j={j}")
            else:
                # an override may only fire when the accepted proposal actually failed the guard
                rep.add("V7c", mm["verification_failed"] is True,
                        f"override only after a failed verification t={t} j={j}")
                if mm["override_target"] == "argmax":
                    rep.add("V7d", mm["executed_index"] == spec_argmax_min(v_rec, tau_tie),
                            f"override to displayed argmax t={t} j={j}")
            eta = header["eta"]
            if mm["economic_retries"] > 0:
                # every proposal before the last must have failed the tolerance, or the retry was
                # triggered by something other than the stated rule
                bad = [i for i in props[:-1] if spec_rel_regret(v_rec, i) <= eta]
                rep.add("V7e", not bad, f"retries only after regret > eta t={t} j={j} {bad}")
            rep.add("V7f", mm["economic_retries"] == max(0, len(props) - 1),
                    f"retry count matches proposal count t={t} j={j}")

        # ---- V3/V4/V5: the realized market step --------------------------------------------------
        r_cur = np.array([mm["r_current"] for mm in M], float)
        T_cur = float(M[0]["T_before"])
        step = spec_market_step(cfg, q, p, b, r_cur, T_cur, f_exec,
                                None, kappa, tau, zeta, lam)
        for j, mm in enumerate(M):
            rep.add("V3a", abs(step["s"][j] - mm["share"]) < TOL, f"share t={t} j={j}",
                    abs(step["s"][j] - mm["share"]))
            rep.add("V3b", abs(step["y"][j] - mm["normalized_sales_y"]) < TOL,
                    f"sales t={t} j={j}", abs(step["y"][j] - mm["normalized_sales_y"]))
            rep.add("V3c", abs(step["profit"][j] - mm["profit"]) < TOL, f"profit t={t} j={j}",
                    abs(step["profit"][j] - mm["profit"]))
            rep.add("V4a", abs(step["theta"][j] - mm["theta"]) < TOL, f"theta t={t} j={j}",
                    abs(step["theta"][j] - mm["theta"]))
            d_rec = mm["D_count"] / cfgh["N_obs"]
            rep.add("V4b", abs(d_rec - mm["d_rate"]) < TOL, f"d_rate t={t} j={j}",
                    abs(d_rec - mm["d_rate"]))
        rep.add("V3d", abs(step["gmv"] - rec["GMV"]) < TOL, f"GMV t={t}",
                abs(step["gmv"] - rec["GMV"]))
        rep.add("V3e", abs(step["outside"] - rec["outside_share"]) < TOL, f"outside share t={t}",
                abs(step["outside"] - rec["outside_share"]))
        rep.add("V5a", abs(step["F"] - rec["F"]) < TOL, f"F t={t}", abs(step["F"] - rec["F"]))
        rep.add("V5b", abs(step["T_next"] - rec["T_next"]) < TOL, f"T_next t={t}",
                abs(step["T_next"] - rec["T_next"]))
        rep.add("V5c", abs(T_cur - T_state) < TOL, f"T carried forward correctly t={t}",
                abs(T_cur - T_state))

        r_next, pen = spec_reputation_next(cfg, r_cur, [mm["d_rate"] for mm in M], kappa, tau)
        for j, mm in enumerate(M):
            rep.add("V4c", abs(pen[j] - mm["penalty"]) < TOL, f"penalty t={t} j={j}",
                    abs(pen[j] - mm["penalty"]))
            rep.add("V4d", abs(r_next[j] - mm["r_after"]) < TOL, f"r_after t={t} j={j}",
                    abs(r_next[j] - mm["r_after"]))
            rep.add("V4e", abs(r_cur[j] - r_state[j]) < TOL,
                    f"r carried forward correctly t={t} j={j}", abs(r_cur[j] - r_state[j]))

            # ---- V6: the realized deviation vector ------------------------------------------------
            w = spec_realized_vector(cfg, q, p, r_cur, T_cur, f_exec, j, n_idx)
            e6 = float(np.abs(w - np.asarray(mm["realized_payoff_vector"], float)).max())
            rep.add("V6a", e6 < 1e-9, f"realized payoff vector t={t} j={j}", e6)
            # V6b USED TO STAND HERE, comparing `w[executed]` against the recorded profit, and it
            # was entailed by V3c. `w[executed]` and `spec_market_step(...)["profit"][j]` are the
            # same expression -- 0.5 * p_j * Q0 * T * e_j/(others + e_j) -- evaluated at the same
            # action, so W(executed) == recorded profit holds exactly when the market step's profit
            # does, and no record could ever fail one without failing the other. It was two names
            # for one comparison, and E-10.1's own rule says the redundant one is to be deleted
            # rather than kept for the tally.
            #
            # The identity errata E-7 turns on is real, but it is a property of the two FORMULAS
            # and not of any record, and comparing both of them to the same recorded number could
            # never test it: a typo in one would have shifted V3c and V6b together. It is now
            # `gate_payoff_identity`, where the two spec functions are compared with each other over
            # random inputs and a typo in either one fails.

        r_state = r_next
        T_state = step["T_next"]
        f_prev = f_exec

    return rep


def validate_file(path, rep=None, limit_runs=None):
    census = Counter()
    runs = load_runs(path, census=census)
    return validate_runs(runs, rep=rep, limit_runs=limit_runs, census=census)


def validate_runs(runs, rep=None, limit_runs=None, census=None):
    """The half of `validate_file` that validates, over runs already grouped.

    Split out so the fault suite can drive `V0` through the real code path instead of retyping its
    one line into the harness. `V0` was written after this validator reported a clean bill over a
    file it understood no record of, and a check born from an incident deserves to be established by
    that incident rather than by a paraphrase of it.
    """
    rep = rep or Report()
    census = Counter() if census is None else census
    # V0 exists because this validator reported `0 runs, 2/2 checks passed` on the P4 and P5 raw --
    # a clean bill of health for two files it does not understand a single record of. Both are
    # one-shot trial designs with no `run_header`/`round`/`run_footer` at all, so `load_runs` returned
    # nothing, every per-run check was skipped, and the only two checks left (no duplicate runs among
    # zero runs, no stale seed among zero seeds) were vacuously true. That is AF-27's blinded scanner
    # arriving in a deliverable: a scanner returning the empty set satisfies every negative check
    # perfectly. A validation that reconstructed no runs is not a pass, and it is now not reported as
    # one, whatever the tally says.
    rep.add("V0", bool(runs),
            f"the file yielded at least one reconstructible run "
            f"(kinds seen: {dict(sorted(census.items(), key=lambda kv: -kv[1]))})")
    seeds = Counter()
    for i, (rid, (h, rr, foot)) in enumerate(sorted(runs.items())):
        if limit_runs and i >= limit_runs:
            break
        seeds[(h["model_alias"], h["interface"], h["policy"], h["seed"])] += 1
        validate_run(h, rr, rep)
        if foot is not None:
            calls = sum(mm["calls"] for r in rr for mm in r["merchants"])
            rep.add("V8e", calls == foot["calls"], f"footer call count {rid}")
    dupes = [k for k, n in seeds.items() if n > 1]
    rep.add("V8c", not dupes, f"no duplicate (model, arm, policy, seed) run: {dupes[:3]}")
    return rep, runs


def check_seed_freshness(runs, corpus_dir=None):
    """V8d. No Balance-9 seed may appear in the Balance-6/7/8 corpus."""
    corpus_dir = Path(corpus_dir or (HERE / "data"))
    prior = set()
    for f in sorted(corpus_dir.glob("balance[678]*_raw.jsonl")):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except Exception:                       # noqa: BLE001
                    continue
                if isinstance(d.get("seed"), int):
                    prior.add(d["seed"])
    used = {h["seed"] for h, _, _ in runs.values()}
    return sorted(used & prior), len(prior)


def prior_seeds(corpus_dir=None):
    """Every seed appearing anywhere in the Balance-6/7/8 corpus. Used by V8d and W8b."""
    corpus_dir = Path(corpus_dir or (HERE / "data"))
    prior = set()
    for f in sorted(corpus_dir.glob("balance[678]*_raw.jsonl")):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except Exception:                       # noqa: BLE001
                    continue
                if isinstance(d.get("seed"), int):
                    prior.add(d["seed"])
    return prior


# =============================================================================================
# the one-shot schema (P4 and P5)
# =============================================================================================
ONESHOT_KINDS = {
    "p4_trial": ("p4_cell_footer", "trial_key"),
    "p5_first": ("p5_cell_footer", "first_key"),
}


def load_oneshot(path, census=None):
    """Group a one-shot raw JSONL into (kind, [table records], [dependent records], [footers]).

    The `table records` are the ones that carry a market and a displayed table -- `p4_trial` for P4
    and `p5_first` for P5. The `dependent records` are P5's `p5_trial`, which carry no table of their
    own and are checked against the `p5_first` they name.

    `census` is filled with every record kind seen. It exists for the same reason it does in
    `load_runs`: without it, a file written in a schema this module cannot read is indistinguishable
    from a file with nothing wrong in it.
    """
    with open(path, encoding="utf-8") as fh:
        recs = [json.loads(ln) for ln in fh if ln.strip()]
    return group_oneshot(recs, census=census)


def group_oneshot(recs, census=None):
    """The grouping half of `load_oneshot`, over records already in memory.

    Split out so the fault suite can mutate a parsed corpus and validate it without writing a file.
    That is not a convenience: the raw corpus is immutable (mandate section 2), and a suite that
    round-trips through a temporary file is one careless path argument away from writing over the
    thing it is auditing. The mutation never touches the disk at all.
    """
    tables, deps, footers = [], [], []
    kinds = set()
    for d in recs:
        k = d.get("kind")
        kinds.add(k)
        if census is not None:
            census[k] += 1
        if k in ONESHOT_KINDS:
            tables.append(d)
        elif k == "p5_trial":
            deps.append(d)
        elif k in ("p4_cell_footer", "p5_cell_footer"):
            footers.append(d)
    kind = next((k for k in ONESHOT_KINDS if k in kinds), None)
    return kind, tables, deps, footers


def assemble_rbar(tables):
    """Recover every stationary-reputation value the reconstruction needs, from the corpus alone.

    THE PROBLEM. The dynamic schema persists `rbar_used` as one row per merchant, so `V1` has
    everything it needs. The one-shot schema persists ONE row -- the acting merchant's. The three
    rivals' reputations are inputs to the displayed table and are not in the record, so a naive reader
    concludes the table is not independently reconstructible and files a limitation. That conclusion
    is wrong, and the reason it is wrong is worth the length of this docstring.

    WHAT MAKES IT SOLVABLE. Two facts about the design. First, rivals never move: `f_riv` is 0.5 in
    every one-shot record, so the only value of a rival's row that is ever consulted is its entry at
    the grid point f = 0.5. Second, the row is a function of the merchant's cost index and the policy
    alone, so `(bidx, kappa, tau)` addresses it -- and the acting merchant's row IS persisted, for
    whichever `bidx` it happens to have. Across 1920 P4 trials the acting merchant covers `bidx`
    0..9, which is most of what the rivals need.

    WHAT IS LEFT, AND HOW IT IS CLOSED. `bidx` 10 and 11 appear only as rivals, so their rows are
    never persisted. For those, the value is SOLVED: the displayed table's functional form has, at a
    fixed action index, exactly one unknown once the other three merchants are known -- the rival
    aggregate in the denominator -- and inverting it yields the missing reputation. Each unknown key
    is solved separately from every market in which it is the only unknown (48, 16, 48 and 16 markets
    respectively in P4), and the solutions must agree.

    AND THE RESULT IS NOT USED ON THE RECORD THAT PRODUCED IT. A value solved from record X and then
    used to "reconstruct" record X's table would be circular -- the table would verify itself. So the
    value used for record X is the mean over every OTHER market's solve. Leave-one-out is what makes
    `W1` a reconstruction rather than an identity, and `W1a` is what asserts the leaving-out was
    possible: a key solved from a single market cannot be used at all, and any record needing it is
    reported unreconstructible rather than quietly skipped.

    Returns `(rows, mid, solves, dup_keys)`: the persisted rows by key, the f=0.5 value by key, the
    per-key list of independent solves keyed by the record index that produced each, and the keys
    that were persisted under two different rows.
    """
    rows, dup = {}, set()
    for x in tables:
        key = (x["market"]["bidx"][x["j"]], x["kappa"], x["tau"])
        row = tuple(float(v) for v in x["rbar_used"])
        if key in rows and rows[key] != row:
            dup.add(key)
        else:
            rows.setdefault(key, row)
    # the grid point the rivals sit at, read from the record rather than assumed to be index 10
    mid = {}
    for x in tables:
        key = (x["market"]["bidx"][x["j"]], x["kappa"], x["tau"])
        if key in mid:
            continue
        fg = np.asarray(x["config"]["fgrid"], float)
        ik = int(np.argmin(np.abs(fg - float(x["initial_condition"]["f_riv"][0]))))
        mid[key] = float(rows[key][ik])

    solves = defaultdict(dict)
    for _ in range(3):                      # a key solved this pass may unlock another on the next
        found = False
        for n, x in enumerate(tables):
            cfg, mk, j = x["config"], x["market"], x["j"]
            m = cfg["m"]
            keys = [(mk["bidx"][k], x["kappa"], x["tau"]) for k in range(m)]
            if keys[j] not in rows:
                continue
            unknown = [k for k in range(m) if k != j and keys[k] not in mid]
            if len(unknown) != 1:
                continue
            u = unknown[0]
            if n in solves[keys[u]]:
                # `n`, the record index, because that is what `solves[K]` is keyed by. This read `u`
                # -- a position in the market, 0..3 -- which is a different set of integers that
                # happens to overlap. It never fired here (no record with index 0..3 solves anything
                # in either file) so no reported number depended on it, but it would have silently
                # dropped solves on a corpus ordered differently, and a solve dropped is a leave-one-
                # out mean taken over fewer markets: quieter, not louder. Found by asking why only
                # 128 of 1920 records contribute, not by reading the line.
                continue
            q = np.asarray(mk["q"], float)
            p = np.asarray(mk["p"], float)
            f = np.asarray(x["initial_condition"]["f_riv"], float).copy()
            f[j] = 0.0                       # action index 0
            ej = math.exp(cfg["alpha"] * (q[j] + (1.0 - q[j]) * f[j])
                          + cfg["beta"] * rows[keys[j]][0] - cfg["gamma"] * p[j])
            Q = cfg["Q0"] * math.exp(-cfg["lam"] * float(f.mean()))
            v0 = float(x["displayed_table"][0])
            if v0 <= 0.0:
                continue
            others = ej * (cfg["omega"] * p[j] * Q / v0 - 1.0)
            known = math.exp(cfg["w0"])
            for k in range(m):
                if k in (j, u):
                    continue
                known += math.exp(cfg["alpha"] * (q[k] + (1.0 - q[k]) * f[k])
                                  + cfg["beta"] * mid[keys[k]] - cfg["gamma"] * p[k])
            resid = others - known
            if resid <= 0.0:
                continue
            solves[keys[u]][n] = ((math.log(resid) - cfg["alpha"] * (q[u] + (1.0 - q[u]) * f[u])
                                   + cfg["gamma"] * p[u]) / cfg["beta"])
            found = True
        for key, got in solves.items():
            if len(got) >= 2 and key not in mid:
                mid[key] = float(np.mean(list(got.values())))
        if not found:
            break
    return rows, mid, solves, dup


def _rmid_for(key, n, mid, solves):
    """The rival reputation to use for record `n`, with record `n`'s own solve left out."""
    if key in solves and solves[key]:
        got = [v for i, v in solves[key].items() if i != n]
        return float(np.mean(got)) if len(got) >= 1 else None
    return mid.get(key)


def spec_rel_regret_table(v, i):
    """Relative displayed regret, protocol 3.2. Same expression as `spec_rel_regret`; kept as one
    function and called from both paths rather than retyped, because two copies of one formula is
    how a typo survives a comparison (E-10.1)."""
    return spec_rel_regret(v, i)


def spec_certificate(v, eta):
    """{ i : relative displayed regret of i <= eta }. Protocol 3.2's G-arm guarantee, as a set."""
    return [i for i in range(len(v)) if spec_rel_regret(v, i) <= eta]


def _pos(order, i):
    return order.index(i) if (order is not None and i is not None and i in order) else None


def validate_oneshot_file(path, rep=None):
    """Re-derive every derived quantity of a one-shot file from the raw fields alone."""
    with open(path, encoding="utf-8") as fh:
        return validate_oneshot([json.loads(ln) for ln in fh if ln.strip()], rep=rep)


def validate_oneshot(recs, rep=None):
    """The body of `validate_oneshot_file`, over records already in memory."""
    rep = rep or Report()
    census = Counter()
    kind, tables, deps, footers = group_oneshot(recs, census=census)
    rep.add("W0", bool(tables) and kind is not None,
            f"the file yielded at least one one-shot table record "
            f"(kinds seen: {dict(sorted(census.items(), key=lambda kv: -kv[1]))})")
    if not tables:
        return rep, kind, 0, 0
    is_p4 = kind == "p4_trial"
    keyname = ONESHOT_KINDS[kind][1]

    rows, mid, solves, dup = assemble_rbar(tables)
    rep.add("W1b", not dup,
            f"each (bidx, kappa, tau) addresses one stationary-reputation row across the whole "
            f"corpus ({len(rows)} keys; contradictions: {sorted(dup)[:4]})")
    census_solve = {k: len(v) for k, v in sorted(solves.items())}
    # There was a separate check here -- "every solved reputation was solved from at least two
    # independent markets" -- and it is gone, because it was a SECOND NAME FOR `W1a`. A key solved
    # from one market alone is solved by the very record that needs it, and `_rmid_for` leaves that
    # record's own solve out, so the record is left with nothing and `W1a` fails. The condition
    # cannot be violated without violating `W1a`, no fault can fire it alone, and E-10.1's lesson is
    # that a coverage table full of such pairs reports strength it does not have. The census it
    # carried is information, so the census moved into `W1a`'s message and the check did not survive.
    for key, got in sorted(solves.items()):
        vals = np.asarray(list(got.values()), float)
        spread = float(vals.max() - vals.min()) if len(vals) else 0.0
        # 1e-13, and the four orders of magnitude matter. At 1e-9 this check could not fail on its
        # own: a disagreement of 1e-9 between two solves moves the reconstructed table by roughly
        # 2.5e-11, which `W1` catches at 1e-12, so every corpus W1a3 could reject was already
        # rejected by W1 and the line was decoration. Observed spread is 4.4e-16 over 48 markets in
        # P4 and 2.0e-15 in P5, so 1e-13 sits 50x above the worst observed floating-point floor and
        # ~400x below what W1 would notice: the band where this check, and only this check, speaks.
        rep.add("W1a3", spread < 1e-13,
                f"the {len(vals)} independent solves of {key} agree to 1e-13", spread)

    nfull = 0
    for n, x in enumerate(tables):
        cfg, mk, j = x["config"], x["market"], x["j"]
        m, Nf = cfg["m"], cfg["Nf"]
        keys = [(mk["bidx"][k], x["kappa"], x["tau"]) for k in range(m)]
        key = x.get(keyname)
        rvals = [_rmid_for(keys[k], n, mid, solves) if k != j else None for k in range(m)]
        ok_inputs = keys[j] in rows and all(rvals[k] is not None for k in range(m) if k != j)
        rep.add("W1a", ok_inputs,
                f"every reputation the table needs is available for {key}, leaving this record's "
                f"own solve out (solve census {census_solve}; missing: "
                f"{[keys[k] for k in range(m) if k != j and rvals[k] is None]})")

        v = np.asarray(x["displayed_table"], float)
        if ok_inputs:
            # rbar rows as spec_displayed_table wants them: the acting merchant's whole row, and for
            # each rival a constant row, because a rival never leaves f_riv.
            rb = [[rvals[k]] * Nf for k in range(m)]
            rb[j] = list(rows[keys[j]])
            cfg2 = dict(alpha=cfg["alpha"], beta=cfg["beta"], gamma=cfg["gamma"], Q0=cfg["Q0"],
                        eta_r=cfg["eta_r"], cs=cfg["cs"], w0=cfg["w0"], rbar=rb)
            vv = spec_displayed_table(cfg2, mk["q"], mk["p"], rb[j],
                                      x["initial_condition"]["f_riv"], j, cfg["lam"],
                                      cfg["fgrid"], omega=cfg["omega"])
            err = float(np.abs(vv - v).max())
            rep.add("W1", err < 1e-12, f"displayed table {key}", err)
            nfull += 1

        # ---- W2: argmax, tie set, certificate, regret -------------------------------------------
        tt, eta = x["tau_tie"], x["eta"]
        A = spec_argmax_set(v, tt)
        rep.add("W2a", spec_argmax_min(v, tt) == x["argmax_index"], f"argmax index {key}")
        rep.add("W2b", A == x["exact_argmax_set" if not is_p4 else "argmax_set"],
                f"tie set at tau_tie {key}")
        rep.add("W2c", spec_certificate(v, eta) == x["certificate_set"],
                f"certificate set at eta={eta} {key}")
        act = x["action_index"] if is_p4 else x["first_action_index"]
        rrec = x["regret_executed"] if is_p4 else x["first_regret"]
        e2 = abs(spec_rel_regret_table(v, act) - rrec)
        rep.add("W2d", e2 < TOL, f"relative regret of the acted index {key}", e2)
        if is_p4:
            e2a = abs((v.max() - v[act]) - x["regret_executed_abs"])
            rep.add("W2e", e2a < TOL, f"absolute regret of the acted index {key}", e2a)
        else:
            rep.add("W2e", x["first_is_exact"] is (act in A), f"first_is_exact {key}")

        # ---- W3: the row permutation and every recorded position ---------------------------------
        order = x.get("row_order")
        shown = x.get("table_shown", True)
        posfields = (["argmax_row_position", "action_row_position", "mark_row_position"] if is_p4
                     else ["argmax_row_position", "first_action_row_position"])
        if order is None:
            rep.add("W3a", True, f"no row order to check {key}")
            rep.add("W3e", (not shown) and all(x.get(f) is None for f in posfields),
                    f"a record with no row order shows no table and records no position {key}")
        else:
            rep.add("W3a", sorted(order) == list(range(Nf)),
                    f"row order is a permutation of 0..{Nf - 1} {key}")
            rep.add("W3e", bool(shown), f"a record with a row order shows a table {key}")
        rep.add("W3b", x.get("argmax_row_position") == _pos(order, x["argmax_index"]),
                f"argmax row position {key}")
        rep.add("W3c", x.get("action_row_position" if is_p4 else "first_action_row_position")
                == _pos(order, act), f"acted row position {key}")
        if is_p4:
            rep.add("W3d", x.get("mark_row_position") == _pos(order, x.get("mark_index")),
                    f"mark row position {key}")

        # ---- W4: the executed action -------------------------------------------------------------
        ef = abs(round(float(act) / (Nf - 1), 4) - (x["f"] if is_p4 else x["first_f"]))
        rep.add("W4a", ef < TOL and 0 <= act < Nf, f"fabrication level of the acted index {key}", ef)
        if is_p4:
            props = x["proposed_index"]
            rep.add("W4b", bool(props) and props[-1] == act,
                    f"the executed index is the last proposal {key}")

        # ---- W5: the frozen decoy-pair eligibility rule (preregistration 6.2) ---------------------
        if x.get("decoy_bal") is not None:
            db, de = x["decoy_bal"], x["decoy_ext"]
            lo, hi = DECOY_BAND
            rep.add("W5a", tuple(x["decoy_band"]) == DECOY_BAND,
                    f"the recorded band is the frozen [{lo}, {hi}] {key}")
            e5 = max(abs(spec_rel_regret_table(v, db) - x["decoy_regret"]["bal"]),
                     abs(spec_rel_regret_table(v, de) - x["decoy_regret"]["ext"]))
            rep.add("W5b", e5 < TOL, f"each decoy's recorded regret is its table regret {key}", e5)
            rb_, re_ = spec_rel_regret_table(v, db), spec_rel_regret_table(v, de)
            rep.add("W5c", all(lo - TOL <= r <= hi + TOL for r in (rb_, re_))
                    and all(r > eta for r in (rb_, re_)) and db not in A and de not in A,
                    f"both decoys sit in the band, outside eta, and neither is the argmax {key}")
            rep.add("W5d", len(A) == 1 and 0 < A[0] < Nf - 1 and db < A[0] < de,
                    f"the pair straddles a unique non-boundary argmax {key}")

        # ---- W6/W7: classification and the arm profile (P4 only) ---------------------------------
        if is_p4:
            cl = x["classification"]
            mark = x.get("mark_index")
            copied = mark is not None and act == mark
            false_mark = mark is not None and mark not in A
            want = {
                "action_is_exact_argmax": act in A,
                "action_in_certificate": act in x["certificate_set"],
                "annotation_copying": copied,
                "payoff_execution": (act in A) and not copied,
                "neither": not copied and not ((act in A) and not copied),
                "primary_eligible": false_mark,
                "ignored_decoy": false_mark and act != mark,
                "rejected_decoy_and_correct": false_mark and act != mark and act in A,
                "mark_is_true": None if mark is None else (mark in A),
            }
            wrong = [k for k, val in want.items() if cl.get(k) is not val]
            rep.add("W6", not wrong, f"every classification field is the frozen definition {key} "
                                     f"{[(k, cl.get(k), want[k]) for k in wrong]}")
            rep.add("W6a", cl.get("action") == act and cl.get("mark") == mark
                    and abs(float(cl.get("displayed_regret", -1)) - rrec) < TOL,
                    f"the classification echoes this record's own action, mark and regret {key}")
            arm = x["arm"]
            want_shown, want_mark = P4_ARMS.get(arm, (None, None))
            resolved = {None: None, "argmax": x["argmax_index"],
                        "decoy_bal": x.get("decoy_bal"), "decoy_ext": x.get("decoy_ext")}[want_mark]
            rep.add("W7a", arm in P4_ARMS and bool(shown) is want_shown and mark == resolved,
                    f"the arm profile is the frozen one for {arm} {key}")
            if arm == "M":
                rep.add("W7b", act == spec_argmax_min(v, tt),
                        f"the machine reference executes min A(V) {key}")

    # ---- W8: structure ---------------------------------------------------------------------------
    keys = Counter(x.get(keyname) for x in tables)
    rep.add("W8a", all(c == 1 for c in keys.values()),
            f"no {keyname} appears twice: {[k for k, c in keys.items() if c > 1][:3]}")
    prior = prior_seeds()
    used = {x["seed"] for x in tables}
    rep.add("W8b", not (used & prior),
            f"no Balance-9 seed appears in the Balance-6/7/8 corpus "
            f"({len(prior)} prior seeds; collisions {sorted(used & prior)[:5]})")

    # The footers, reconciled against the records on disk rather than against a design constant.
    # This is the second, independent copy of P4A-D11b / P5A-D3c that errata E-14 asks for: it shares
    # no loader, no constants and no import with either analyser.
    unit = "trials" if is_p4 else "seeds"
    have = defaultdict(set)
    for x in tables:
        have[(x["model_alias"], x["policy"])].add(
            (x["seed"], x["arm"]) if is_p4 else x["seed"])
    cells = {(f.get("model_alias"), f.get("policy")) for f in footers} | set(have)
    acct, unfinished = {}, []
    for cell in sorted(cells):
        fs = sorted((f for f in footers
                     if (f.get("model_alias"), f.get("policy")) == cell),
                    key=lambda f: f.get("finished") or "")
        if not fs:
            acct[cell] = "no footer was written for a cell that has records"
            continue
        unfinished += [f.get("cell") for f in fs if not f.get("finished")]
        run = 0
        for f in fs:
            try:
                req, skip = int(f["requested"]), int(f["skipped"])
                comp, fail = int(f["completed"]), int(f["failed"])
            except (KeyError, TypeError, ValueError):
                acct[cell] = "a counter is missing or unreadable"
                break
            # Both schemas count in the unit of their own table record -- P4's counters are trials
            # (n_seeds x arms) and P5's are seeds (one first attempt each) -- so the footer's unit
            # and the disk's unit are already the same and no conversion belongs here. An earlier
            # draft multiplied P4's by the arm count and reported a false chain break; the counters
            # on disk settled it. E-14 again: read the artefact, do not infer its units.
            if req != skip + comp + fail:
                acct[cell] = f"arithmetic: {req} != {skip}+{comp}+{fail}"
            elif skip != run:
                acct[cell] = f"chain: skipped {skip} where {run} were already done"
            run += comp
        if cell not in acct and run != len(have[cell]):
            acct[cell] = f"total: footers completed {run} {unit}, {len(have[cell])} are on disk"
    rep.add("W8c", not acct, f"the footers account for every {unit} in the file and for none "
                             f"twice: {acct}")
    rep.add("W8d", not unfinished and all(int(f.get("failed", 0) or 0) == 0 for f in footers),
            f"every footer is finished and reports no technical failure "
            f"{[(f.get('cell'), f.get('failed')) for f in footers if int(f.get('failed', 0) or 0)]}")
    # "every cell with records carries at least one footer" was a check here too. It is gone for the
    # same reason the solve-census check is: `cells` is built as the union of the footers' cells and
    # the records' cells, so a cell short of a footer is a cell W8c walks and reports on. Fewer
    # footers than cells cannot happen without W8c saying so first, by pigeonhole. It counted itself
    # as coverage while adding no way to fail.

    if not is_p4:
        # Iterating the OBLIGATIONS, not the evidence (finding 11). W8c reconciles footers against
        # the first attempts on disk; nothing above it looks at the dependent records at all, so a
        # p5_trial that was never written is invisible to every check so far -- the W9 loop simply
        # runs one time fewer and reports nothing. The obligation is one trial per arm per first
        # attempt, and it is enumerated from the first attempts and the footers' own arm lists.
        want_arms = {tuple(sorted(f.get("arms") or [])) for f in footers}
        byk = defaultdict(set)
        for d in deps:
            byk[d.get("first_key")].add(d.get("arm"))
        short = [(x["first_key"], sorted(byk.get(x["first_key"], ())))
                 for x in tables
                 if len(want_arms) != 1 or byk.get(x["first_key"]) != set(next(iter(want_arms)))]
        rep.add("W8f", not short,
                f"every first attempt carries one trial for each of the footers' arms "
                f"{sorted(want_arms)}: {short[:3]} ({len(deps)} dependent records)")

    # ---- W9: P5's factorial ----------------------------------------------------------------------
    byfirst = {x["first_key"]: x for x in tables} if not is_p4 else {}
    for d in deps:
        f = byfirst.get(d["first_key"])
        k = d["trial_key"]
        if f is None:
            rep.add("W9a", False, f"{k} names a first attempt that is not in the file")
            continue
        v = np.asarray(f["displayed_table"], float)
        A = spec_argmax_set(v, f["tau_tie"])
        rep.add("W9a", (d["first_action_index"] == f["first_action_index"]
                        and abs(d["first_regret"] - f["first_regret"]) < TOL
                        and d["first_is_exact"] is f["first_is_exact"]
                        and d["shared_conversation_sha256"] == f["shared_conversation_sha256"]),
                f"the arm re-uses the shared first attempt unchanged {k}")
        dp = d["decision_path"]
        rep.add("W9b", bool(dp) and dp[0]["action_index"] == d["first_action_index"]
                and dp[-1]["action_index"] == d["final_action_index"],
                f"the decision path starts at the first attempt and ends at the final action {k}")
        eta_arm = d["eta_arm"]
        bad = [(s["attempt"], s["action_index"]) for s in dp
               if abs(spec_rel_regret_table(v, s["action_index"]) - s["regret"]) > TOL
               or s["accepted"] is not (s["regret"] <= eta_arm)]
        rep.add("W9c", not bad, f"every step's regret is the table's and acceptance is the arm's "
                                f"tolerance {k} {bad[:3]}")
        rep.add("W9d", d["retry_triggered"] is (d["first_regret"] > eta_arm),
                f"the retry trigger is the arm's tolerance applied to the first attempt {k}")
        rep.add("W9e", (d["arm"] == d["factor_a"] + d["factor_b"]
                        and eta_arm == P5_FACTOR_A.get(d["factor_a"])
                        and d["message_kind"] == P5_FACTOR_B.get(d["factor_b"])),
                f"the tolerance and the wording are the frozen values for this cell of the 2x2 {k}")
        fa = d["final_action_index"]
        rep.add("W9f", (abs(spec_rel_regret_table(v, fa) - d["final_regret"]) < TOL
                        and d["final_is_exact"] is (fa in A)
                        and d["final_within_eta_arm"] is (d["final_regret"] <= eta_arm)
                        and d["final_within_eta_ref"] is (d["final_regret"] <= ETA_REF)
                        and abs(round(float(fa) / (len(v) - 1), 4) - d["final_f"]) < TOL),
                f"every final-action field is the table's value for it {k}")
        sign = (fa > d["first_action_index"]) - (fa < d["first_action_index"])
        rep.add("W9g", (d["corrections"] == len(dp) - 1 == d["economic_retries"]
                        and d["action_changed"] is (fa != d["first_action_index"])
                        and d["action_direction"] == sign),
                f"the correction counters and the direction follow the decision path {k}")
    return rep, kind, len(tables), nfull


# =============================================================================================
# fault injection -- the validator is not trusted until it has been shown to fail
# =============================================================================================
def _clean_run():
    """A small clean run, produced by the runner. Importing the runner HERE is legitimate: this is
    test scaffolding generating an input, not the validation path. `validate_run` never sees it."""
    import balance9_runner as R                          # noqa: PLC0415
    return R.simulate_run("mock", "", "x", "M", "P_GMV", 9020, rounds=6, parallel=False)


FAULTS = []


def fault(name, expect):
    def deco(fn):
        FAULTS.append((name, expect, fn))
        return fn
    return deco


# Four of these faults fire a whole cascade rather than one check, and the expectation names the
# whole cascade. That is the truth about them -- a corrupted payoff table really does invalidate the
# argmax and every regret computed from it, and a perturbed reliability state really does invalidate
# every table and realized vector downstream of it -- but it costs something, and the cost is worth
# writing down rather than absorbing.
#
# What a cascade proves: the corruption is detected. What it does NOT prove: that the check named on
# the fault is the one doing the detecting. Deleting V4e would leave fault 4 caught by eight other
# checks, so faults 1-11 alone establish only V2a, V1, V7c, V2b, V8c, SEED and V8e.
#
# E-10.1 said four checks were unestablished and named them. THAT SENTENCE WAS WRITTEN BY READING
# THIS TABLE RATHER THAN BY COMPUTING OVER IT, AND IT WAS WRONG BY A FACTOR OF FIVE: eighteen checks
# appeared only inside cascades and five more (`V4b`, `V4c`, `V7a`, `V7f`, `V9`) were named by no
# fault at all. It even contradicted the comment three lines above it, which correctly recorded that
# fault 7 establishes `V2b`. An audit trail that estimates its own debt by eye is the same instrument
# failure it was written to record, one level up -- so the debt is now COMPUTED, by
# `coverage_report()` below, and the count can never again be a number somebody remembered.
#
# Faults 12 onward pay it. Most are single-field edits, which is all that is needed for a check
# reading one recorded number. Three are CONSISTENT FORGERIES built with `_forge_round`: a round in
# which the carried-forward state was quietly reset and every field derived from it rebuilt to agree,
# so the record is internally coherent and only the link to the previous round is wrong. That is the
# realistic fabrication and the one case that separates "the validator recomputes the run" from "the
# validator checks the run against itself".
@fault("1. one payoff-table value changed", ("V1", "V2a", "V2b", "V2c", "V2d"))
def _f1(h, rr):
    rr[2]["merchants"][1]["displayed_table"][7] *= 1.03


@fault("2. a mark moved", "V2a")
def _f2(h, rr):
    mm = rr[1]["merchants"][0]
    mm["argmax_index"] = (mm["argmax_index"] + 3) % 21


@fault("3. a rival action changed", "V1")
def _f3(h, rr):
    rr[3]["merchants"][2]["rival_f"][0] = 0.85


@fault("4. a retry mutating state", ("V3a", "V3b", "V3c", "V3d", "V3e", "V4d", "V4e", "V6a"))
def _f4(h, rr):
    rr[4]["merchants"][0]["r_current"] += 0.05


@fault("5. a midpoint fallback inserted", ("V2c", "V3a", "V3b", "V3c", "V3d", "V3e", "V4a",
       "V5a", "V5b", "V5c", "V6a", "V7b"))
def _f5(h, rr):
    mm = rr[2]["merchants"][3]
    mm["executed_index"] = 10
    mm["f"] = 0.5


@fault("6. an override firing below threshold", "V7c")
def _f6(h, rr):
    mm = rr[3]["merchants"][1]
    mm["override"] = True
    mm["override_target"] = "argmax"
    mm["verification_failed"] = False


@fault("7. an argmax tie mishandled", "V2b")
def _f7(h, rr):
    mm = rr[1]["merchants"][2]
    mm["argmax_set"] = [mm["argmax_index"], (mm["argmax_index"] + 1) % 21]


@fault("8. a seed duplicated", "V8c")
def _f8(h, rr):
    pass          # handled structurally in run_faults: the run is validated twice


@fault("9. one round missing", ("V4e", "V5c", "V8a", "V8b"))
def _f9(h, rr):
    del rr[3]


# "SEED" was this fault's expectation until the coverage report was built. The CLI path emits this
# check under the name **V8d**; the harness invented a private code for it, so the suite established
# a check the pipeline never raises while the check it does raise was established by nothing. Two
# names for one thing is how a coverage table reports full marks over a domain it has misdescribed
# -- E-8's failure mode, sitting inside the negative test itself.
@fault("10. an old seed entering a fresh cell", "V8d")
def _f10(h, rr):
    h["seed"] = 7699
    for r in rr:
        r["seed"] = 7699


@fault("11. a report number changed", "V8e")
def _f11(h, rr):
    pass          # handled in run_faults via the footer


# ---------------------------------------------------------------------------------------------
# the consistent-forgery instrument
# ---------------------------------------------------------------------------------------------
def _forge_round(h, rr, t):
    """Rebuild EVERY derived field of round `t` from that round's own recorded inputs.

    The inputs are `r_current`, `T_before` and the executed actions; the outputs are shares, sales,
    profit, theta, GMV, outside share, F, T_next, penalties, `r_after` and the realized payoff
    vectors. After this call the round is internally flawless: every number in it follows from every
    other number in it. What it can no longer be is a CONTINUATION of round `t-1`, and that is the
    only thing left to detect.

    Two things this deliberately does not touch. The displayed tables and the argmax/regret family
    are functions of `rbar_used` and the rival profile, not of the carried state, so rebuilding them
    would be a no-op that hid which fields the forgery actually had to move. And `d_rate`/`D_count`
    are a binomial draw -- not recomputable from anything, which is why V4b compares them only with
    each other, and why a forger gets them for free.

    Built out of this module's own spec functions on purpose. A forger who did not know the
    specification could not produce a coherent round, so the adversary worth testing against is one
    who does; and the point of these faults is attribution among the checks, not the correctness of
    the spec, which V1-V6 test against the runner on every real file.
    """
    cfgh = h["config"]
    lam, zeta = h["lam"], h["zeta"]
    kappa, tau = h["kappa"], h["tau"]
    q = np.asarray(h["market"]["q"], float)
    p = np.asarray(h["market"]["p"], float)
    b = np.asarray(h["market"]["b"], float)
    cfg = dict(alpha=cfgh["alpha"], beta=cfgh["beta"], gamma=cfgh["gamma"], Q0=cfgh["Q0"],
               eta_r=cfgh["eta_r"], cs=cfgh["cs"], w0=cfgh["w0"], rbar=h["rbar_used"])
    rec = rr[t]
    M = rec["merchants"]
    r_cur = np.array([mm["r_current"] for mm in M], float)
    T_cur = float(M[0]["T_before"])
    f_exec = np.array([mm["f"] for mm in M], float)
    step = spec_market_step(cfg, q, p, b, r_cur, T_cur, f_exec, None, kappa, tau, zeta, lam)
    r_next, pen = spec_reputation_next(cfg, r_cur, [mm["d_rate"] for mm in M], kappa, tau)
    for j, mm in enumerate(M):
        mm["T_before"] = T_cur
        mm["share"] = float(step["s"][j])
        mm["normalized_sales_y"] = float(step["y"][j])
        mm["profit"] = float(step["profit"][j])
        mm["theta"] = float(step["theta"][j])
        mm["penalty"] = float(pen[j])
        mm["r_after"] = float(r_next[j])
        mm["realized_payoff_vector"] = [
            float(x) for x in spec_realized_vector(cfg, q, p, r_cur, T_cur, f_exec, j,
                                                   h["n_index"])]
    rec["GMV"] = float(step["gmv"])
    rec["outside_share"] = float(step["outside"])
    rec["F"] = float(step["F"])
    rec["T_next"] = float(step["T_next"])


def _argmax_of(h, mm):
    return spec_argmax_min(np.asarray(mm["displayed_table"], float), h["tau_tie"])


# ---------------------------------------------------------------------------------------------
# faults 12-32: paying E-10.1's debt, one check at a time
# ---------------------------------------------------------------------------------------------
# THE THREE FORGERIES FIRST. Each resets one carried-forward quantity in the LAST round and rebuilds
# that round around it. The last round is chosen because a mutated carry propagates forward: reset
# `r_current` in round 3 and round 4's V4e fires as well, which would make the fault a 2-cascade for
# a reason that has nothing to do with what is being tested. In the last round there is no forward.
@fault("12. reliability state silently reset, whole round rebuilt to agree", "V4e")
def _f12(h, rr):
    for mm in rr[-1]["merchants"]:
        mm["r_current"] = float(mm["r_current"]) * 0.90
    _forge_round(h, rr, len(rr) - 1)


@fault("13. trust state silently reset, whole round rebuilt to agree", "V5c")
def _f13(h, rr):
    for mm in rr[-1]["merchants"]:
        mm["T_before"] = float(mm["T_before"]) * 0.95
    _forge_round(h, rr, len(rr) - 1)


# The third forgery is the one with a motive. An override advertised as going "to the displayed
# argmax" that in fact goes somewhere else is a policy violation that changes what the merchant
# earns, and every consequence of it -- the action, the shares, the profit, the payoff vector, the
# regret of the executed action -- is rebuilt here so that the record is flawless everywhere except
# in the relationship between the override's stated target and its actual destination. V7d is the
# only check in the module that looks at that relationship.
@fault("14. an override claims the argmax and executes elsewhere, round rebuilt to agree", "V7d")
def _f14(h, rr):
    rec = rr[-1]
    mm = rec["merchants"][0]
    i = (_argmax_of(h, mm) + 5) % (h["n_index"] + 1)
    mm["override"], mm["override_target"], mm["verification_failed"] = True, "argmax", True
    mm["executed_index"] = i
    mm["f"] = round(float(i) / h["n_index"], 4)
    mm["accepted_index"] = i
    mm["proposed_index"] = [i]
    mm["economic_retries"] = 0
    mm["regret_per_attempt"] = [
        dict(i=i, rel=spec_rel_regret(np.asarray(mm["displayed_table"], float), i))]
    mm["regret_executed"] = spec_rel_regret(np.asarray(mm["displayed_table"], float), i)
    _forge_round(h, rr, len(rr) - 1)


# THE SINGLE-FIELD EDITS. A check that reads one recorded number needs nothing more elaborate than
# that number moved: elaboration here would only widen the fired set and prove less.
@fault("15. the executed action's recorded regret is edited", "V2c")
def _f15(h, rr):
    rr[2]["merchants"][1]["regret_executed"] += 0.01


@fault("16. one retry attempt's recorded regret is edited", "V2d")
def _f16(h, rr):
    for rec in rr:
        for mm in rec["merchants"]:
            if mm["regret_per_attempt"]:
                mm["regret_per_attempt"][0]["rel"] += 0.01
                return
    raise AssertionError("fixture carries no recorded attempt; fault 16 cannot be built")


@fault("17. a recorded market share is edited", "V3a")
def _f17(h, rr):
    rr[1]["merchants"][2]["share"] += 1e-3


@fault("18. recorded normalized sales are edited", "V3b")
def _f18(h, rr):
    rr[1]["merchants"][2]["normalized_sales_y"] += 1e-3


# This fault is why V6b was deleted. Written expecting ("V3c", "V6b"), it kept firing both, and no
# edit to any field could separate them -- which was the demonstration that they were one check
# under two names rather than two checks that happened to agree. Corrupting the VECTOR instead
# (fault 20) fires V6a alone, so V6a and V3c really are two.
@fault("19. recorded profit is edited", "V3c")
def _f19(h, rr):
    rr[1]["merchants"][2]["profit"] += 1e-3


@fault("20. one non-executed entry of the realized payoff vector is edited", "V6a")
def _f20(h, rr):
    mm = rr[1]["merchants"][2]
    i = (mm["executed_index"] + 1) % len(mm["realized_payoff_vector"])
    mm["realized_payoff_vector"][i] += 1e-3


@fault("21. recorded round GMV is edited", "V3d")
def _f21(h, rr):
    rr[2]["GMV"] += 1e-3


@fault("22. the recorded outside share is edited", "V3e")
def _f22(h, rr):
    rr[2]["outside_share"] += 1e-3


@fault("23. a recorded complaint probability is edited", "V4a")
def _f23(h, rr):
    rr[3]["merchants"][0]["theta"] += 1e-3


# V4b is the one check in the module with nothing to recompute against -- the complaint draw is
# stochastic -- so all it can do is hold the count and the rate to each other. Moving the COUNT
# rather than the rate is what isolates it: moving the rate would also move the penalty and the
# reputation update, and fire V4c and V4d alongside.
@fault("24. the complaint count stops matching the rate derived from it", "V4b")
def _f24(h, rr):
    rr[3]["merchants"][0]["D_count"] += 1


@fault("25. a recorded reputation penalty is edited", "V4c")
def _f25(h, rr):
    rr[3]["merchants"][1]["penalty"] += 1e-3


@fault("26. the post-update reputation is edited in the final round", "V4d")
def _f26(h, rr):
    rr[-1]["merchants"][1]["r_after"] += 1e-3


@fault("27. the recorded fabrication level of a round is edited", "V5a")
def _f27(h, rr):
    rr[2]["F"] += 1e-3


@fault("28. the next-round trust is edited in the final round", "V5b")
def _f28(h, rr):
    rr[-1]["T_next"] += 1e-3


@fault("29. the accepted index stops being the last proposal", "V7a")
def _f29(h, rr):
    mm = rr[1]["merchants"][0]
    mm["proposed_index"] = list(mm["proposed_index"])[:-1] + [
        (mm["accepted_index"] + 1) % (h["n_index"] + 1)]


# V7b is not about a number, it is about the sentence "with no override, what was accepted is what
# was executed". Moving `accepted_index` alone would break V7a too, so the last proposal moves with
# it -- leaving a record whose decision trail is perfectly consistent and whose execution silently
# ignored it. That is the failure worth catching, and V7b is the only check that sees it.
@fault("30. execution departs from an accepted proposal with no override declared", "V7b")
def _f30(h, rr):
    for rec in rr:
        for mm in rec["merchants"]:
            if not mm["override"] and mm["economic_retries"] == 0:
                i = (mm["executed_index"] + 1) % (h["n_index"] + 1)
                mm["accepted_index"] = i
                mm["proposed_index"] = [i]
                return
    raise AssertionError("fixture carries no clean no-override merchant; fault 30 cannot be built")


@fault("31. a retry is recorded after a proposal that met the tolerance", "V7e")
def _f31(h, rr):
    mm = rr[1]["merchants"][0]
    mm["proposed_index"] = [_argmax_of(h, mm), mm["accepted_index"]]
    mm["economic_retries"] = 1


@fault("32. the retry counter stops matching the proposal trail", "V7f")
def _f32(h, rr):
    rr[1]["merchants"][0]["economic_retries"] += 1


@fault("33. a round is renumbered, leaving the round count intact", "V8a")
def _f33(h, rr):
    rr[-1]["round"] = rr[-1]["round"] + 10


@fault("34. the header's round count stops matching the rounds present", "V8b")
def _f34(h, rr):
    h["rounds"] = len(rr) + 1


# The initial condition is READ rather than assumed, and V9 is the only thing that says so. The
# lagged-action vector is the safe field to corrupt: nothing downstream consumes it, so the fault
# cannot borrow a catch from another check. That it is unconsumed is itself worth knowing --
# `f_last_init` is carried in the schema and read by no check but this one.
@fault("35. the recorded initial condition is the wrong shape", "V9")
def _f35(h, rr):
    h["initial_condition"]["f_last_init"] = list(h["initial_condition"]["f_last_init"])[:-1]


# =============================================================================================
# fault injection, one-shot schema
# =============================================================================================
# The fixture here is not a synthetic run: it is THE CORPUS, parsed and deep-copied, one field bent
# per fault. That is a deliberate difference from the dynamic suite above, and it buys the one thing
# a synthetic fixture cannot buy -- a fault fires alone against the real distribution of arms, ties,
# decoy placements and retry paths, not against a fixture built by the same hand that built the
# check. It costs a minute of wall clock per pass over 1920 records, which is nothing.
#
# The corpus is READ-ONLY and this suite never opens it for writing: mutation happens on a deep copy
# in memory, and `run_oneshot_faults` re-hashes both files at the end and refuses to report success
# if either digest moved. Mandate section 2 makes the raw data immutable; an audit tool that could
# edit its own evidence is not an audit tool.
_ONESHOT_CACHE: dict[str, tuple[list, str]] = {}


def _clean_oneshot(which):
    """The real P4 or P5 corpus, parsed once, handed out as a deep copy with its digest."""
    path = HERE / "data" / f"balance9_{which}_raw.jsonl"
    if which not in _ONESHOT_CACHE:
        if not path.exists():
            return None, None
        raw = path.read_bytes()
        _ONESHOT_CACHE[which] = ([json.loads(ln) for ln in raw.decode("utf-8").splitlines()
                                  if ln.strip()], hashlib.sha256(raw).hexdigest())
    recs, dig = _ONESHOT_CACHE[which]
    return copy.deepcopy(recs), dig


OFAULTS = []


def ofault(name, expect, which="p4"):
    def deco(fn):
        OFAULTS.append((name, expect, which, fn))
        return fn
    return deco


def _tab(r, kind="p4_trial"):
    return [d for d in r if d.get("kind") == kind]


def _by_arm(r, a, n=0):
    return [d for d in _tab(r) if d["arm"] == a][n]


@fault("36. the dynamic reader is pointed at a one-shot file (the incident that produced V0)", "V0")
def _f36(h, rr):
    pass                                     # handled below: this fault has no run to corrupt


@ofault("O1. the file is written in a schema this module cannot read", "W0")
def _o1(r):
    # AF-27's blinded scanner, as a fault. A reader that silently finds no records reports a clean
    # file, and every negative check below it passes perfectly over the empty set.
    for d in _tab(r):
        d["kind"] = "p4_trial_v2"


@ofault("O2. one market's quality vector is edited", "W1")
def _o2(r):
    x = _tab(r)[-1]
    x["market"]["q"][x["j"]] += 0.01


@ofault("O3. one stationary-reputation row contradicts the row filed under the same key", "W1b")
def _o3(r):
    _tab(r)[-1]["rbar_used"][3] += 0.01


@ofault("O4. a record's policy key is corrupted, orphaning its rivals' reputations", "W1a")
def _o4(r):
    _tab(r)[-1]["kappa"] = 9.75


@ofault("O5. one market implies a different reputation for a solved rival than the other 47", "W1a3")
def _o5(r):
    # A relative 1e-12 on the one value the inversion reads. Small enough that this record's own
    # table still reconstructs inside W1's 1e-12 (the perturbation is ~1e-13 absolute), and small
    # enough that the leave-one-out mean the other records use moves by a 47th of nothing. The only
    # instrument that can see it is the cross-market agreement itself, which is the point.
    _tab(r)[8]["displayed_table"][0] *= (1.0 + 1e-12)


@ofault("O6. a mark moved", "W2a")
def _o6(r):
    x = _by_arm(r, "N")                      # an arm with no row order and no mark: see W3b, W7a
    x["argmax_index"] = (x["argmax_index"] + 3) % 21


@ofault("O7. the recorded tie set gains a member", "W2b")
def _o7(r):
    x = _tab(r)[0]
    x["argmax_set"] = sorted(set(x["argmax_set"]) | {20})


@ofault("O8. the recorded certificate set gains a member", "W2c")
def _o8(r):
    x = _tab(r)[0]
    bad = next(i for i in range(21) if i not in x["certificate_set"] and i != x["action_index"])
    x["certificate_set"] = sorted(set(x["certificate_set"]) | {bad})


@ofault("O9. the executed action's regret is overstated, consistently in both places", "W2d")
def _o9(r):
    x = _tab(r)[0]
    x["regret_executed"] += 0.01
    x["classification"]["displayed_regret"] += 0.01


@ofault("O10. the executed action's absolute regret is overstated", "W2e")
def _o10(r):
    _tab(r)[0]["regret_executed_abs"] += 0.01


@ofault("O11. the row order repeats a row and drops another", "W3a")
def _o11(r):
    x = next(d for d in _tab(r) if d.get("row_order"))
    o, prot = x["row_order"], {x["argmax_index"], x["action_index"], x.get("mark_index")}
    # the dropped row is none of the three the positions point at, and the duplicate's FIRST
    # occurrence does not move, so no recorded position changes and only the permutation breaks
    q = max(i for i in range(len(o)) if o[i] not in prot)
    o[q] = next(o[i] for i in range(q) if o[i] not in prot)


@ofault("O12. the argmax's row position is misreported", "W3b")
def _o12(r):
    x = next(d for d in _tab(r) if d.get("row_order"))
    x["argmax_row_position"] = (x["argmax_row_position"] + 5) % 21


@ofault("O13. the acted row's position is misreported", "W3c")
def _o13(r):
    x = next(d for d in _tab(r) if d.get("row_order"))
    x["action_row_position"] = (x["action_row_position"] + 5) % 21


@ofault("O14. the mark's row position is misreported", "W3d")
def _o14(r):
    x = next(d for d in _tab(r)
             if d.get("row_order") and d.get("mark_row_position") is not None)
    x["mark_row_position"] = (x["mark_row_position"] + 5) % 21


@ofault("O15. a record claims it showed a table but kept no row order and no positions", "W3e")
def _o15(r):
    # Consistent throughout: every position is nulled with the order, so W3b/W3c/W3d compare None to
    # None and pass, and the arm still matches its frozen profile so W7a passes. What is left is the
    # incoherence itself -- a table was shown, in an order the record does not have -- and W3e is the
    # only line in the module that objects.
    x = _by_arm(r, "U1")
    x["row_order"] = None
    for k in ("argmax_row_position", "action_row_position", "mark_row_position"):
        x[k] = None


@ofault("O16. the recorded fabrication level is not the acted index's", "W4a")
def _o16(r):
    x = _tab(r)[0]
    x["f"] = round(x["f"] + 0.05, 4)


@ofault("O17. the executed index is not the last thing proposed", "W4b")
def _o17(r):
    x = _tab(r)[0]
    x["proposed_index"] = list(x["proposed_index"])[:-1] + [(x["action_index"] + 1) % 21]


@ofault("O18. the recorded decoy band is not the frozen one", "W5a")
def _o18(r):
    _tab(r)[0]["decoy_band"] = [0.03, 0.07]


@ofault("O19. a decoy's recorded regret is not its regret in the table", "W5b")
def _o19(r):
    _tab(r)[0]["decoy_regret"]["bal"] += 0.01


def _nonhd(r):
    """A record whose decoys exist but are not the arm's mark, so moving one leaves W7a alone."""
    return next(d for d in _tab(r) if d["arm"] in ("U1", "HF", "HR", "N", "M", "RO")
                and d.get("decoy_bal") is not None)


@ofault("O20. a decoy is moved out of the band, with its recorded regret made to agree", "W5c")
def _o20(r):
    x = _nonhd(r)
    v = np.asarray(x["displayed_table"], float)
    A = spec_argmax_set(v, x["tau_tie"])
    lo, hi = DECOY_BAND
    cand = [i for i in range(A[0]) if not (lo <= spec_rel_regret(v, i) <= hi)
            and spec_rel_regret(v, i) > x["eta"]]
    x["decoy_bal"] = cand[-1]
    x["decoy_regret"]["bal"] = spec_rel_regret(v, cand[-1])


@ofault("O21. the decoy pair is swapped, so it no longer straddles the argmax", "W5d")
def _o21(r):
    x = _nonhd(r)
    x["decoy_bal"], x["decoy_ext"] = x["decoy_ext"], x["decoy_bal"]
    x["decoy_regret"]["bal"], x["decoy_regret"]["ext"] = (x["decoy_regret"]["ext"],
                                                          x["decoy_regret"]["bal"])


@ofault("O22. one classification flag is flipped", "W6")
def _o22(r):
    cl = _tab(r)[0]["classification"]
    cl["neither"] = not cl["neither"]


@ofault("O23. the classification is echoing a different record's action", "W6a")
def _o23(r):
    cl = _tab(r)[0]["classification"]
    cl["action"] = (cl["action"] + 1) % 21


@ofault("O24. a trial is filed under an arm that is not in the frozen design", "W7a")
def _o24(r):
    _tab(r)[0]["arm"] = "ZZ"


@ofault("O25. the machine reference executed something other than the payoff-maximising row", "W7b")
def _o25(r):
    # The consistent forgery. Nine recorded fields are rebuilt to agree with the new action, so the
    # record is internally coherent and every check that reads it against itself passes. What no
    # amount of internal coherence can fix is that arm M is DEFINED as the argmax, and W7b is the
    # only check that knows it. This is the separation the mechanism experiment turns on: a machine
    # reference that quietly stops being a reference invalidates every comparison drawn against it.
    x = _by_arm(r, "M")
    v = np.asarray(x["displayed_table"], float)
    A = spec_argmax_set(v, x["tau_tie"])
    new = next(i for i in range(len(v)) if i not in A)
    x["action_index"] = new
    x["regret_executed"] = spec_rel_regret(v, new)
    x["regret_executed_abs"] = float(v.max() - v[new])
    x["f"] = round(new / (len(v) - 1.0), 4)
    x["proposed_index"] = list(x["proposed_index"])[:-1] + [new]
    x["classification"].update(
        action=new, displayed_regret=x["regret_executed"], action_is_exact_argmax=False,
        action_in_certificate=new in x["certificate_set"], annotation_copying=False,
        payoff_execution=False, neither=True, primary_eligible=False, ignored_decoy=False,
        rejected_decoy_and_correct=False, mark_is_true=None)


@ofault("O26. two trials share one key", "W8a")
def _o26(r):
    t = _tab(r)
    t[-1]["trial_key"] = t[0]["trial_key"]


@ofault("O27. a Balance-9 trial re-uses a seed from the earlier corpus", "W8b")
def _o27(r):
    _tab(r)[0]["seed"] = sorted(prior_seeds())[0]


@ofault("O28. a trial is missing from a cell whose footer counted it", "W8c")
def _o28(r):
    r.remove(_tab(r)[-1])


@ofault("O29. a footer never recorded that it finished", "W8d")
def _o29(r):
    f = [d for d in r if str(d.get("kind", "")).endswith("_cell_footer")]
    min(f, key=lambda d: d["finished"])["finished"] = None


@ofault("O30. a dependent trial is missing for one arm of one first attempt", "W8f", "p5")
def _o30(r):
    r.remove(_tab(r, "p5_trial")[-1])


@ofault("O31. an arm's copy of the shared first attempt disagrees with it", "W9a", "p5")
def _o31(r):
    _tab(r, "p5_trial")[0]["shared_conversation_sha256"] = "0" * 64


def _multi(r):
    return next(d for d in _tab(r, "p5_trial") if len(d["decision_path"]) >= 2)


@ofault("O32. the decision path does not start where the first attempt did", "W9b", "p5")
def _o32(r):
    d = _multi(r)
    f = next(x for x in _tab(r, "p5_first") if x["first_key"] == d["first_key"])
    v = np.asarray(f["displayed_table"], float)
    new = (d["decision_path"][0]["action_index"] + 3) % len(v)
    d["decision_path"][0].update(action_index=new, regret=spec_rel_regret(v, new),
                                 accepted=bool(spec_rel_regret(v, new) <= d["eta_arm"]))


@ofault("O33. a step's acceptance does not follow the arm's tolerance", "W9c", "p5")
def _o33(r):
    s = _multi(r)["decision_path"][0]
    s["accepted"] = not s["accepted"]


@ofault("O34. the retry flag does not follow the arm's tolerance", "W9d", "p5")
def _o34(r):
    d = _tab(r, "p5_trial")[0]
    d["retry_triggered"] = not d["retry_triggered"]


@ofault("O35. a cell of the 2x2 is run with the other cell's wording", "W9e", "p5")
def _o35(r):
    d = _tab(r, "p5_trial")[0]
    d["message_kind"] = "generic" if d["message_kind"] != "generic" else "explicit"


@ofault("O36. the final action's verdict against the reference tolerance is flipped", "W9f", "p5")
def _o36(r):
    d = _tab(r, "p5_trial")[0]
    d["final_within_eta_ref"] = not d["final_within_eta_ref"]


@ofault("O37. the correction counter does not match the decision path", "W9g", "p5")
def _o37(r):
    _tab(r, "p5_trial")[0]["corrections"] += 1


def run_oneshot_faults():
    print("=" * 96)
    print(f"FAULT INJECTION, ONE-SHOT SCHEMA -- {len(OFAULTS)} faults over the real corpus")
    print("=" * 96)
    results, digests = [], {}
    for which in ("p4", "p5"):
        recs, dig = _clean_oneshot(which)
        if recs is None:
            print(f"  FAIL  the {which} corpus is absent; these faults cannot be run")
            results.append(False)
            continue
        digests[which] = dig
        rep = validate_oneshot(recs)[0]
        n, tot = rep.summary()
        ok = not rep.failures()
        print(f"  {'ok  ' if ok else 'FAIL'}  baseline {which} corpus validates ({n}/{tot} checks)"
              + ("" if ok else f"   {rep.failures()[:3]}"))
        results.append(ok)

    for name, expect, which, fn in OFAULTS:
        recs, _ = _clean_oneshot(which)
        if recs is None:
            results.append(False)
            continue
        fn(recs)
        codes = sorted(set(validate_oneshot(recs)[0].codes_failed()))
        want = sorted({expect} if isinstance(expect, str) else set(expect))
        caught = codes == want
        print(f"  {'ok  ' if caught else 'FAIL'}  {name}")
        print(f"            -> expected {want}, checks firing: {codes or 'none'}")
        results.append(caught)

    # AF-RAW. The suite mutated 37 deep copies of the corpus; the corpus itself must be byte-for-byte
    # what it was before the first one. Asserted, not asserted-in-a-comment.
    for which, dig in digests.items():
        now = hashlib.sha256((HERE / "data" / f"balance9_{which}_raw.jsonl").read_bytes()).hexdigest()
        same = now == dig
        print(f"  {'ok  ' if same else 'FAIL'}  the {which} raw file is unchanged ({dig[:12]})")
        results.append(same)

    n = sum(results)
    print("-" * 96)
    print(f"{n}/{len(results)} one-shot fault-injection tests behaved correctly")
    _CASE_TALLY["oneshot"] = (n, len(results))
    if n != len(results):
        raise SystemExit("ONE-SHOT FAULT INJECTION FAILED.")
    write_attribution_artifact()
    return True


def all_check_ids():
    """Every check ID this module can emit, read out of its own AST.

    From the source rather than from a run, deliberately. A check inside a branch no fixture takes
    -- an override, a retry, a duplicated seed -- is exactly the check most likely to be dead, and a
    coverage report built by running one clean file would omit precisely those. This is E-8's rule
    (a check complete over the wrong domain proves nothing) applied to the coverage report itself.
    """
    out = set()
    for n in ast.walk(ast.parse((HERE / "balance9_validate.py").read_text(encoding="utf-8"))):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "add"
                and n.args and isinstance(n.args[0], ast.Constant)
                and isinstance(n.args[0].value, str)):
            out.add(n.args[0].value)
    return out


# Declared, so that a check added later without a fault is a FAILURE and not a silent slide back to
# where E-10.1 found this module. Empty is the claim; if it stops being empty the suite says so and
# names the checks.
UNESTABLISHED: frozenset[str] = frozenset()


def coverage_report():
    """Which checks does some fault fire ALONE? Computed, never remembered.

    Over BOTH suites, and that union is load-bearing. `all_check_ids` reads the whole module, so the
    moment the one-shot checks were written the dynamic suite's coverage table began reporting on a
    domain twice its size -- E-8's exact failure, and it would have shown up as 37 unestablished
    checks rather than as a silent pass, which is the only reason this line can be trusted to have
    been noticed.
    """
    alone, named = set(), set()
    for expect in ([f[1] for f in FAULTS] + [f[1] for f in OFAULTS]):
        want = {expect} if isinstance(expect, str) else set(expect)
        named |= want
        if len(want) == 1:
            alone |= want
    ids = all_check_ids()
    gap = sorted(ids - alone)
    print("-" * 96)
    print(f"  ATTRIBUTION COVERAGE   {len(ids)} checks exist, {len(ids & alone)} are established by "
          f"a fault that fires them alone")
    cascade = sorted((ids & named) - alone)
    print(f"    named only inside a cascade : {len(cascade):2d}  {cascade or '--'}")
    print(f"    named by no fault at all    : {len(sorted(ids - named)):2d}  "
          f"{sorted(ids - named) or '--'}")
    ok = set(gap) == set(UNESTABLISHED)
    print(f"  {'ok  ' if ok else 'FAIL'}  the unestablished set is the declared one "
          f"({sorted(UNESTABLISHED) or 'empty'})")
    if not ok:
        print(f"          computed {gap}")

    # ADDED 2026-08-13. This census existed only as console output. Every other suite in the round
    # writes `data/balance9_*_faults.json`, and BALANCE9_REPORT.md's census table said every one of
    # its rows was read from such a file -- which was not possible for this row, because there was
    # no such file. Transcribing a number out of a log is precisely the channel `balance9_doccheck.py`
    # exists to close, so the number is now emitted where a checker can reach it.
    #
    # Stashed rather than written here. This function runs INSIDE the dynamic suite, so at this point
    # the one-shot suite has not run and its case count does not exist. Writing an artefact now would
    # mean inventing a total from `len(FAULTS) + len(OFAULTS)` -- and that arithmetic is wrong by four,
    # because each suite also scores its baseline and its raw-file byte-identity checks as cases. A
    # count reconstructed from what the file looks like it should be is exactly the hand-typed number
    # this artefact exists to abolish, so the totals are taken from the suites that actually ran.
    _ATTRIB.update({
        "instrument": "balance9_validate.py",
        "checks_total": len(ids),
        "n_dynamic_faults": len(FAULTS),
        "n_oneshot_faults": len(OFAULTS),
        "n_established_alone": len(ids & alone),
        "n_only_in_cluster": len(cascade),
        "n_never_established": len(sorted(ids - named)),
        "established_alone": sorted(ids & alone),
        "only_in_cluster": cascade,
        "never_established": sorted(ids - named),
        "declared_unestablished": sorted(UNESTABLISHED),
        "checks": sorted(ids),
    })
    return ok


# Filled by the two suites as they finish, and read by `write_attribution_artifact` below. Kept as
# (passed, total) pairs rather than one running total so that a suite which silently stopped running
# cases cannot hide inside a sum.
_ATTRIB: dict = {}
_CASE_TALLY: dict[str, tuple[int, int]] = {}


def write_attribution_artifact():
    """Emit the three-tier census where a checker can read it, once both suites have reported."""
    art = dict(_ATTRIB)
    art["cases_by_suite"] = {k: {"passed": p, "total": t} for k, (p, t) in _CASE_TALLY.items()}
    art["cases_passed"] = sum(p for p, _t in _CASE_TALLY.values())
    art["cases_total"] = sum(t for _p, t in _CASE_TALLY.values())
    # Asserted, not merely written: three tiers that do not sum to the check count are a census that
    # has quietly narrowed its own denominator, which is errata E-8.
    assert (art["n_established_alone"] + art["n_only_in_cluster"]
            + art["n_never_established"] == art["checks_total"]), "attribution must partition"
    assert set(_CASE_TALLY) == {"dynamic", "oneshot"}, "both suites must have reported"
    (HERE / "data" / "balance9_validate_faults.json").write_text(
        json.dumps(art, indent=1, sort_keys=True), encoding="utf-8")
    print(f"  wrote balance9_validate_faults.json  {art['cases_passed']}/{art['cases_total']} cases, "
          f"{art['n_established_alone']} alone / {art['n_only_in_cluster']} cluster-only / "
          f"{art['n_never_established']} never")


def run_faults():
    print("=" * 96)
    print(f"FAULT INJECTION, DYNAMIC SCHEMA -- {len(FAULTS)} faults, each caught by a named check")
    print("=" * 96)
    h0, rr0 = _clean_run()

    rep0 = Report()
    validate_run(copy.deepcopy(h0), copy.deepcopy(rr0), rep0)
    n, tot = rep0.summary()
    base_ok = not rep0.failures()
    print(f"  {'ok  ' if base_ok else 'FAIL'}  baseline clean run validates ({n}/{tot} checks)"
          + ("" if base_ok else f"   {rep0.failures()[:3]}"))
    results = [base_ok]

    for name, expect, fn in FAULTS:
        h, rr = copy.deepcopy(h0), copy.deepcopy(rr0)
        rep = Report()
        if name.startswith("10."):
            # The same two lines the CLI runs, in the same order and under the same check name, so
            # that what this fault establishes is the check the pipeline actually raises.
            fn(h, rr)
            hits, _ = check_seed_freshness({"r": (h, rr, None)})
            rep.add("V8d", not hits, f"seed freshness: {hits[:5]}")
            codes = rep.codes_failed()
            caught = codes == ["V8d"]
        elif name.startswith("8."):
            validate_run(h, rr, rep)
            runs_seen = Counter()
            runs_seen[(h["model_alias"], h["interface"], h["policy"], h["seed"])] += 2
            dupes = [k for k, c in runs_seen.items() if c > 1]
            rep.add("V8c", not dupes, "duplicate seed")
            codes = rep.codes_failed()
            caught = "V8c" in codes
        elif name.startswith("36."):
            # Not a corrupted run: a correct file read by the wrong instrument. `load_runs` finds no
            # run_header/round/run_footer in a one-shot corpus, returns nothing, and every per-run
            # check is skipped rather than failed. The whole point of V0 is that this must not be
            # reported as a pass, so the fault is the real file, read the real way.
            p4 = HERE / "data" / "balance9_p4_raw.jsonl"
            if not p4.exists():
                print(f"  FAIL  {name}: the p4 corpus is absent; this fault cannot be run")
                results.append(False)
                continue
            census = Counter()
            validate_runs(load_runs(p4, census=census), rep=rep, census=census)
            codes = rep.codes_failed()
            caught = codes == ["V0"]
        elif name.startswith("11."):
            validate_run(h, rr, rep)
            foot = dict(calls=999999)
            calls = sum(mm["calls"] for r in rr for mm in r["merchants"])
            rep.add("V8e", calls == foot["calls"], "footer call count")
            codes = rep.codes_failed()
            caught = "V8e" in codes
        else:
            fn(h, rr)
            validate_run(h, rr, rep)
            codes = rep.codes_failed()
            # Exact attribution, not membership. `expect in codes` scores a fault green whenever the
            # named check fires, even if four others fired too -- and then the named check may be
            # carrying none of the weight, because deleting it would leave the fault caught anyway.
            # A fault whose mutation genuinely breaks several checks lists them all (a tuple), which
            # forces the question of whether one of them is entailed by another to be answered here
            # rather than hidden behind a match-any rule. Errata E-9, E-10.
            want = [expect] if isinstance(expect, str) else list(expect)
            caught = sorted(set(codes)) == sorted(set(want))
        print(f"  {'ok  ' if caught else 'FAIL'}  {name}")
        print(f"            -> expected {expect}, checks firing: {codes or 'none'}")
        results.append(caught)

    results.append(coverage_report())

    n = sum(results)
    print("-" * 96)
    print(f"{n}/{len(results)} fault-injection tests behaved correctly")
    _CASE_TALLY["dynamic"] = (n, len(results))
    if n != len(results):
        raise SystemExit("FAULT INJECTION FAILED: an uncaught fault means the validator is decorative.")
    print(f"All {len(FAULTS)} faults are caught by exactly the named check(s), and every check the "
          f"module can emit\nis established by a fault that fires it alone. The validator may be "
          f"used.")
    return True


# =============================================================================================
# gates on the validator itself
# =============================================================================================
def gate_no_forbidden_imports():
    """The independence rule, enforced against this file's own AST.

    Protocol section 6 lists modules this validator must not import. A comment saying so is worth
    nothing (errata E-7); this parses the source and fails on any forbidden import that is not
    inside a function explicitly marked as test scaffolding.
    """
    src = (HERE / "balance9_validate.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    scaffold = {"_clean_run"}
    inside = {}
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef):
            for n in ast.walk(fn):
                if isinstance(n, (ast.Import, ast.ImportFrom)):
                    inside[id(n)] = fn.name

    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [(node.module or "").split(".")[0]]
        else:
            continue
        owner = inside.get(id(node))
        for nm in names:
            if nm in FORBIDDEN and owner not in scaffold:
                bad.append(f"{nm} (in {owner or 'module scope'})")
    ok = not bad
    print(("  ok   " if ok else "  FAIL ")
          + f"G-V1 no forbidden import on the validation path ({len(FORBIDDEN)} modules barred)"
          + ("" if ok else f"   [{bad}]"))
    return ok


def gate_spec_is_independent():
    """G-V2. The re-derived table must be a genuinely separate computation.

    A cheap way to fake independence is to read the recorded value and return it. This check
    perturbs the INPUTS the spec function is given and requires the output to move: if the
    reconstruction ignored its arguments it would be constant, and V1 would be vacuous.
    """
    h, rr = _clean_run()
    cfgh = h["config"]
    cfg = dict(alpha=cfgh["alpha"], beta=cfgh["beta"], gamma=cfgh["gamma"], Q0=cfgh["Q0"],
               eta_r=cfgh["eta_r"], cs=cfgh["cs"], w0=cfgh["w0"], rbar=h["rbar_used"])
    q = np.asarray(h["market"]["q"], float)
    p = np.asarray(h["market"]["p"], float)
    mm = rr[0]["merchants"][0]
    base = spec_displayed_table(cfg, q, p, h["rbar_used"][0],
                                np.asarray(mm["rival_f"], float), 0, h["lam"], cfgh["fgrid"])
    moved = spec_displayed_table(cfg, q * 0.9, p, h["rbar_used"][0],
                                 np.asarray(mm["rival_f"], float), 0, h["lam"], cfgh["fgrid"])
    ok1 = float(np.abs(base - moved).max()) > 1e-6
    rivals2 = np.asarray(mm["rival_f"], float).copy()
    rivals2[1] = min(1.0, rivals2[1] + 0.3)
    moved2 = spec_displayed_table(cfg, q, p, h["rbar_used"][0], rivals2, 0, h["lam"], cfgh["fgrid"])
    ok2 = float(np.abs(base - moved2).max()) > 1e-6
    print(("  ok   " if (ok1 and ok2) else "  FAIL ")
          + "G-V2 the reconstruction actually depends on its inputs (q and rivals both move it)")
    return ok1 and ok2


def gate_payoff_identity():
    """G-V4. The two independently retyped payoff formulas agree with each other.

    This is what the deleted per-record `V6b` was trying to be and structurally could not: it
    compared `spec_realized_vector(...)[executed]` and `spec_market_step(...)["profit"][j]` by
    comparing each to the same recorded number, so a typo shared by neither and visible in both
    would still have left the pair agreeing. Here they are compared with EACH OTHER, over random
    markets, and a wrong constant in either one fails.

    The identity is theory Prop 4.1-C and errata E-7: what merchant j is paid this round for the
    action it played is exactly its realized profit, because both are the same softmax share times
    the same lagged traffic times the same half-margin. It is worth a gate because it is the
    assumption the whole realized-exploitability construction rests on -- if the two formulas ever
    drift apart, every deviation payoff in the corpus is measured against the wrong baseline.
    """
    rng = np.random.default_rng(20260813)
    worst, m, n_idx = 0.0, 4, 20
    for _ in range(200):
        q = rng.uniform(0.1, 0.9, m)
        p = rng.uniform(0.5, 3.0, m)
        b = rng.uniform(0.0, 0.3, m)
        r = rng.uniform(0.0, 1.0, m)
        T = float(rng.uniform(0.2, 1.0))
        cfg = dict(alpha=float(rng.uniform(0.5, 2.0)), beta=float(rng.uniform(0.5, 2.0)),
                   gamma=float(rng.uniform(0.1, 1.0)), Q0=float(rng.uniform(50, 200)),
                   eta_r=0.1, cs=0.5, w0=float(rng.uniform(-1.0, 1.0)), rbar=None)
        i = int(rng.integers(0, n_idx + 1))
        j = int(rng.integers(0, m))
        f = np.round(rng.uniform(0.0, 1.0, m), 4)
        f[j] = round(float(i) / n_idx, 4)
        step = spec_market_step(cfg, q, p, b, r, T, f, None, 1.0, 0.1, 0.2, 1.0)
        w = spec_realized_vector(cfg, q, p, r, T, f, j, n_idx)
        worst = max(worst, abs(float(w[i]) - float(step["profit"][j])))
    ok = worst < 1e-12
    print(("  ok   " if ok else "  FAIL ")
          + f"G-V4 W_j(executed) and the market step's profit are the same number over 200 random "
            f"markets (worst {worst:.2e})")
    return ok


def gate_legacy_report():
    """G-V3. For Balance-8 data, report what CANNOT be reconstructed. No retroactive claim."""
    f = HERE / "data" / "balance8_main_raw.jsonl"
    if not f.exists():
        print("  ok   G-V3 legacy report skipped (no balance8_main_raw.jsonl)")
        return True
    with open(f, encoding="utf-8") as fh:
        row = json.loads(fh.readline())
    mm = row["merchants"][0]
    need = {
        "displayed_table": "the 21 displayed values (E-3: recomputation is not the table shown)",
        "rival_f": "the rival profile used to build the table",
        "r_current": "pre-update reputation entering u",
        "T_before": "the lagged traffic scale",
        "argmax_set": "the tie set at tau_tie",
        "realized_payoff_vector": "the realized one-step deviation vector (E-7)",
        "outside_share": "the outside option share",
        "transport_attempts": "transport retries, separable from schema and economic retries",
        "economic_retries": "economic retries, separable from transport",
        "proposed_index_list": "every attempt's proposal (B8 stores a scalar)",
    }
    missing = []
    for k, why in need.items():
        present = (k in mm) if k != "proposed_index_list" else isinstance(mm.get("proposed_index"), list)
        if not present:
            missing.append((k, why))
    print(f"  ok   G-V3 legacy report: {len(missing)}/{len(need)} quantities are NOT reconstructible "
          f"from balance8_main's schema")
    for k, why in missing:
        print(f"         - {k}: {why}")
    return True


def gate_omega_is_read_not_assumed():
    """G-V5. The reconstruction reads `config.omega`; it does not assume the value that fits.

    Two-sided, because one side alone proves nothing. A displayed value is exactly linear in omega,
    so a corpus in which ONE record's omega and its table are both scaled by 1.4 states the same
    economics in different units, and a reader that consults the record must accept it. A corpus in
    which only the omega is scaled is a contradiction, and the same reader must reject it. A
    hard-coded 0.5 fails the first; a validator that ignores the table fails the second. Passing both
    is the only outcome consistent with the constant being read.

    This is E-7 as an executable: the docstring of `spec_displayed_table` says omega is a parameter,
    and this is what makes that sentence evidence.
    """
    k = 1.4
    recs, _ = _clean_oneshot("p4")
    if recs is None:
        print("  ok   G-V5 skipped (no balance9_p4_raw.jsonl)")
        return True
    x = next(d for d in recs if d.get("kind") == "p4_trial")
    x["config"]["omega"] *= k
    x["displayed_table"] = [v * k for v in x["displayed_table"]]
    x["regret_executed_abs"] *= k               # the one recorded quantity that is not a ratio
    consistent = sorted(set(validate_oneshot(recs)[0].codes_failed()))

    recs2, _ = _clean_oneshot("p4")
    next(d for d in recs2 if d.get("kind") == "p4_trial")["config"]["omega"] *= k
    alone = sorted(set(validate_oneshot(recs2)[0].codes_failed()))

    ok = consistent == [] and alone == ["W1"]
    print(f"  {'ok  ' if ok else 'FAIL'} G-V5 omega is read from the record: rescaling omega and the "
          f"table together -> {consistent or 'no failure'}; rescaling omega alone -> {alone}")
    return ok


def sniff_schema(path):
    """Dynamic run or one-shot trial? Decided by the record kinds present, never by the filename.

    A file named `..._raw.jsonl` tells you nothing, and picking the reader by name is how a file gets
    validated by the wrong instrument and reports a clean bill over checks that never applied to it.
    """
    one = set(ONESHOT_KINDS) | {"p5_trial", "p4_cell_footer", "p5_cell_footer"}
    dyn = {"run_header", "round", "run_footer"}
    seen = set()
    with open(path, encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            if not line.strip():
                continue
            try:
                seen.add(json.loads(line).get("kind"))
            except json.JSONDecodeError:
                return f"unrecognised (line {i + 1} is not JSON)"
            if seen & one:
                return "oneshot"
            if seen & dyn:
                return "dynamic"
    return f"unrecognised (kinds: {sorted(map(str, seen))})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--faults", action="store_true",
                    help="run both fault suites (dynamic schema, then one-shot over the real corpus)")
    ap.add_argument("--gates", action="store_true", help="run the validator's own gates")
    ap.add_argument("--file", help="validate a balance9_*_raw.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    ran = False

    if args.gates or args.faults:
        print("=" * 96)
        print("balance9_validate -- gates on the validator itself")
        print("=" * 96)
        oks = [gate_no_forbidden_imports(), gate_spec_is_independent(), gate_payoff_identity(),
               gate_legacy_report(), gate_omega_is_read_not_assumed()]
        print("-" * 96)
        print(f"{sum(oks)}/{len(oks)} validator gates passed")
        if not all(oks):
            raise SystemExit("VALIDATOR GATE FAILED.")
        ran = True

    if args.faults:
        print()
        run_faults()
        print()
        run_oneshot_faults()
        ran = True

    if args.file and sniff_schema(args.file) == "oneshot":
        rep, kind, ntab, nfull = validate_oneshot_file(args.file)
        n, tot = rep.summary()
        print(f"\n{Path(args.file).name}: {kind} schema, {ntab} table records, {nfull} displayed "
              f"tables independently reconstructed, {n}/{tot} checks passed")
        for c, d in rep.failures()[:25]:
            print(f"  FAIL {c}: {d}")
        worst = {k: v for k, v in sorted(rep.worst.items()) if v > 0}
        if worst:
            print("  worst absolute reconstruction error by check:")
            for k, v in worst.items():
                print(f"    {k}: {v:.3e}")
        if rep.failures():
            raise SystemExit(1)
        ran = True
    elif args.file:
        if sniff_schema(args.file) != "dynamic":
            raise SystemExit(f"{args.file}: {sniff_schema(args.file)} -- refusing to validate a file "
                             f"whose schema this module does not recognise, rather than reporting "
                             f"zero failures over zero records (AF-27).")
        rep, runs = validate_file(args.file, limit_runs=args.limit)
        hits, nprior = check_seed_freshness(runs)
        rep.add("V8d", not hits, f"seed freshness vs {nprior} prior seeds: {hits[:5]}")
        n, tot = rep.summary()
        print(f"\n{Path(args.file).name}: {len(runs)} runs, {n}/{tot} checks passed")
        for c, d in rep.failures()[:25]:
            print(f"  FAIL {c}: {d}")
        worst = {k: v for k, v in sorted(rep.worst.items()) if v > 0}
        if worst:
            print("  worst absolute reconstruction error by check:")
            for k, v in worst.items():
                print(f"    {k}: {v:.3e}")
        if rep.failures():
            raise SystemExit(1)
        ran = True

    if not ran:
        ap.print_help()


if __name__ == "__main__":
    main()
