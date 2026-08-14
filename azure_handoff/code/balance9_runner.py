"""balance9_runner.py -- the Balance-9 experiment runner (protocol sections 4 and 5).

This runner has two obligations that pull in opposite directions, and most of the design is about
holding both at once:

  1. The ENVIRONMENT must be the Balance-8 environment, bit for bit. P3 is billed as a clean
     replication. If the market step drifted while the schema was being enriched, a P3 result would
     be a new experiment wearing a replication's name, and the drift would be invisible because
     every number would still look reasonable. Gate G-R1 therefore runs this runner and
     `balance8_runner` side by side on the deterministic no-LLM arms and requires the trajectories
     to agree to the last bit.

  2. The RECORD must be far richer than Balance-8's, because errata E-3 established that Balance-8's
     schema cannot support independent re-derivation: too much of the decision was reconstructible
     only by importing runner code, which makes a "validator" a mirror. Protocol section 4 lists
     what has to be in the raw for `balance9_validate.py` to rebuild a decision from the file alone.

The reconciliation is that this file changes what is WRITTEN DOWN and never what is COMPUTED. Every
dynamical line below is the Balance-8 line. Where I wanted to improve something -- and there are
places where the Balance-8 choice is not what I would write fresh -- I left it alone and recorded the
objection in a comment, because a replication that silently improves its own environment is not a
replication.

Three specific departures, all of them additive and all gated:

  * the argmax uses `balance9_prompts.argmax_min_index` (lowest index within TAU_TIE) rather than
    `np.argmax`. Protocol 3.3 requires a declared tie rule rather than an inherited one. On real
    payoff vectors the two agree -- G-R7 asserts that they agree on every table the gate visits, so
    if they ever diverge the run stops instead of quietly changing arms;
  * transport retries are counted, which Balance-8 could not do because `phase2_llm.openrouter_call`
    swallows its own retry loop. This runner calls it with `retries=0` and does the backoff itself.
    Transport is not an experimental variable, but section 5 requires the three counters -- transport,
    schema repair, economic retry -- to be separable, and Balance-8's single `retries` field is
    exactly the collapse that makes a network hiccup indistinguishable from a verification effect;
  * a run is buffered and written only when it completes. Section 5 rule 4 says abort BEFORE the
    state transition and rule 5 says re-run the whole seed; a partially written run in an append-only
    file would make rule 5 impossible to honour without patching, and rule 6 of the mandate forbids
    patching raw files.

Prompts are not built here. They come from `balance9_prompts`, which proved byte-identity with
Balance-8's builder over 144 cases; a second builder in a second file is a second thing to drift.

Raw output is append-only JSONL under a `balance9_*` prefix. No prior raw file is opened for writing.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import random
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(HERE))

import equilibrium as E                      # noqa: E402
import hetero as H                           # noqa: E402
import hetero_policy_audit as A              # noqa: E402
import phase2_exploitability as X            # noqa: E402
import phase2_framing_controlled as FC       # noqa: E402
import phase2_llm as L                       # noqa: E402
import balance9_prompts as PR                # noqa: E402

# ---------------------------------------------------------------------------------------------
# frozen constants -- sourced from the prompt module so there is ONE definition of each
# ---------------------------------------------------------------------------------------------
NIDX = PR.NIDX
ETA = PR.ETA
TAU_TIE = PR.TAU_TIE
OMEGA, LAM = PR.OMEGA, PR.LAM

ROUNDS = 80
ZETA = 0.2
MAX_ECON_RETRY = 2        # economic (verification) retries -- protocol 7, "retry budget 2"
SCHEMA_REPAIR = 2         # information-free formatting repairs per economic attempt
TRANSPORT_RETRIES = 6     # transport attempts per completion, counted separately

POLICIES = {"P_GMV": (0.5, 0.20), "P_robust": (4.0, 0.30), "P_sep": (8.0, 0.02)}
MODELS = {"gemma": "google/gemma-3-27b-it", "llama": "meta-llama/llama-3.3-70b-instruct"}

NOLLM_INTERFACES = ("M", "O")

# (eta_guard, override_target); None = no verification loop.
GUARD_SPEC = {"R": (ETA, None), "G": (ETA, "argmax")}

DATA = HERE / "data"


class TechnicalFailure(RuntimeError):
    """Raised instead of inventing an action. Aborts the whole seed; never falls back."""


# ---------------------------------------------------------------------------------------------
# environment -- every line below is the Balance-8 line, and G-R1 proves it
# ---------------------------------------------------------------------------------------------
def idx_to_f(i):
    return round(float(i) / NIDX, 4)


def rbar_grid_for(cfg, kappa, tau):
    # The kappa == 0 branch routes to (0, 0) rather than (0, tau). It is dead for every Balance-9
    # policy (all have kappa > 0) but is preserved because a replication may not quietly change a
    # branch just because it believes the branch is unreachable.
    return H.rbar_grid(cfg, 0.0, 0.0) if kappa == 0 else H.rbar_grid(cfg, kappa, tau)


def payoff_table(cfg, qs, ps, bidx, rg, j, f_rivals):
    """The DISPLAYED table: 21 projected profits for merchant j holding rivals fixed."""
    base = np.array(f_rivals, float)
    v = np.empty(NIDX + 1)
    for i in range(NIDX + 1):
        fp = base.copy()
        fp[j] = idx_to_f(i)
        v[i] = float(X.profits(cfg, qs, ps, bidx, OMEGA, LAM, rg, fp)[j])
    return v


def rel_regret(v, i):
    if i is None:
        return float("nan")
    return float((v.max() - v[i]) / max(abs(v.max()), 1e-9))


def abs_regret(v, i):
    return float("nan") if i is None else float(v.max() - v[i])


def argmax_set(v):
    """Every index within TAU_TIE of the maximum. Recorded so a tie is visible in the raw rather
    than resolved silently by whichever index numpy happened to reach first."""
    v = np.asarray(v, float)
    return [int(i) for i in np.flatnonzero(v >= v.max() - TAU_TIE)]


def certificate_set(v, eta):
    return [i for i in range(NIDX + 1) if rel_regret(v, i) <= eta + 1e-12]


def balanced_in_certificate(v, eta):
    return int(min(certificate_set(v, eta)))


def override_index(v, eta, target):
    return PR.argmax_min_index(v) if target == "argmax" else balanced_in_certificate(v, eta)


def realized_payoff_vector(cfg, qs, ps, bidx, r_now, T_now, f_exec, j):
    """W_j(i) for all 21 i: the payoff merchant j WOULD have been paid this round for each action,
    holding its own reputation and the lagged traffic scale fixed (BALANCE9_THEORY.md Prop 4.1-C).

    This is the object errata E-7 is about. Balance-8's audit substituted stationary reputation here
    and so measured a different function -- one with an interior optimum that the market never paid.
    The guard against repeating that mistake is not this docstring: `check_realized_identity` below
    asserts W_j(executed) == the profit actually recorded, and G-R3 runs it on every decision of
    every gate run. An instrument that cannot reproduce a recorded transition does not get to
    describe one.
    """
    w0 = E._w0(cfg, OMEGA)
    f = np.asarray(f_exec, float).copy()
    m = len(qs)
    u_riv = cfg.alpha * (qs + (1 - qs) * f) + cfg.beta * r_now - cfg.gamma * ps
    exu = np.exp(u_riv)
    others = math.exp(w0) + float(exu.sum()) - float(exu[j])
    out = np.empty(NIDX + 1)
    for i in range(NIDX + 1):
        uj = cfg.alpha * (qs[j] + (1 - qs[j]) * idx_to_f(i)) + cfg.beta * r_now[j] - cfg.gamma * ps[j]
        e = math.exp(uj)
        s = e / (others + e)
        out[i] = (ps[j] - 0.5 * ps[j]) * (cfg.Q0 * T_now) * s
    return out


# ---------------------------------------------------------------------------------------------
# transport -- one completion, with its own counted backoff
# ---------------------------------------------------------------------------------------------
def transport_call(model_id, msgs, attempts_box):
    """One completion. Returns (raw, usage, n_transport_attempts, provider_meta).

    `openrouter_call` is asked for retries=0 so that its internal loop cannot hide attempts from the
    counter; the backoff below is the same shape as the one inside it (exponential, jittered, capped
    at 30s), for the reason recorded there: unjittered retries at high concurrency re-synchronise
    into a second failure storm.
    """
    last = None
    for k in range(TRANSPORT_RETRIES + 1):
        attempts_box[0] += 1
        try:
            raw, usage = L.openrouter_call(model_id, "", "", messages=msgs, retries=0)
            return raw, (usage or {}), k + 1
        except Exception as exc:                      # noqa: BLE001
            last = exc
            if "non-retryable" in str(exc):
                raise
            if k < TRANSPORT_RETRIES:
                time.sleep(min(30.0, 1.0 * (2.0 ** k)) * (0.5 + random.random()))
    raise TechnicalFailure(f"transport failed after {TRANSPORT_RETRIES + 1} attempts: {last}")


def parse_action(raw):
    """Strict JSON parse. Returns (action_index, claimed_best_index, reasoning, ok).

    No coercion of out-of-range integers to the nearest legal one: an action index of 47 is a parse
    failure, not a 20. Coercion would manufacture a decision the model did not make, which is the
    Balance-7 defect this whole line of work exists to avoid.
    """
    try:
        s = str(raw).strip()
        if s.startswith("```"):
            s = s.strip("`")
            s = s[s.find("{"):]
        i0, i1 = s.find("{"), s.rfind("}")
        if i0 < 0 or i1 <= i0:
            return None, None, "", False
        o = json.loads(s[i0:i1 + 1])
        a = o.get("action_index")
        if not isinstance(a, (int, float)) or isinstance(a, bool):
            return None, None, "", False
        a = int(a)
        if a < 0 or a > NIDX:
            return None, None, "", False
        c = o.get("claimed_best_index")
        c = int(c) if isinstance(c, (int, float)) and not isinstance(c, bool) and 0 <= int(c) <= NIDX else None
        return a, c, str(o.get("reasoning", ""))[:200], True
    except Exception:                                 # noqa: BLE001
        return None, None, "", False


# ---------------------------------------------------------------------------------------------
# one merchant decision
# ---------------------------------------------------------------------------------------------
def new_trace(interface, v, mark):
    return dict(
        interface=interface, mark_index=mark,
        argmax_index=PR.argmax_min_index(v), argmax_set=argmax_set(v), tau_tie=TAU_TIE,
        # the three counters section 5 requires to stay separable
        transport_attempts=0, schema_repairs=0, economic_retries=0,
        proposed_index=[],           # EVERY economic attempt, in order -- a list, not a scalar
        accepted_index=None, executed_index=None,
        claimed_best_index=None, override=False, override_target=None,
        verification_failed=False, followed_mark=None,
        attempts=[], calls=0, latency_s=0.0, prompt_tokens=0, completion_tokens=0,
    )


def finish(tr, v, i_exec):
    tr["executed_index"] = int(i_exec)
    tr["accepted_index"] = tr["proposed_index"][-1] if tr["proposed_index"] else None
    tr["regret_proposed"] = rel_regret(v, tr["proposed_index"][0] if tr["proposed_index"] else None)
    tr["regret_accepted"] = rel_regret(v, tr["accepted_index"])
    tr["regret_executed"] = rel_regret(v, i_exec)
    tr["regret_executed_abs"] = abs_regret(v, i_exec)
    tr["regret_per_attempt"] = [dict(i=int(i), rel=rel_regret(v, i), abs=abs_regret(v, i))
                                for i in tr["proposed_index"]]
    if tr["mark_index"] is not None and tr["proposed_index"]:
        tr["followed_mark"] = bool(tr["proposed_index"][0] == tr["mark_index"])
    return tr


def decide(agent, model_id, cfg, interface, policy, state, v, seed, prompt_fn=None):
    """Executed action plus the full trace. Raises TechnicalFailure rather than inventing an action.

    The state argument is READ ONLY. Nothing in this function may mutate market state -- that is the
    invariant G-R2 tests by hashing the state before and after every attempt of the same decision.
    """
    if interface in NOLLM_INTERFACES:
        tr = new_trace(interface, v, None)
        i = PR.argmax_min_index(v) if interface == "M" else int(state["eq_index"])
        tr["proposed_index"].append(i)
        return finish(tr, v, i)

    eta_g, ovr_target = GUARD_SPEC.get(interface, (None, None))
    build = prompt_fn or PR.p3_prompt
    system, user, mark = build(cfg, interface, policy, state, v)
    tr = new_trace(interface, v, mark)
    tr["override_target"] = ovr_target
    tr["prompt_sha256"] = dict(system=PR.sha(system), user=PR.sha(user))

    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    box = [0]
    i = None
    t_run = time.time()

    for attempt in range(MAX_ECON_RETRY + 1):
        ok = False
        for rep in range(SCHEMA_REPAIR + 1):
            t0 = time.time()
            raw, usage, n_tr = transport_call(model_id, msgs, box)
            a, c, reason, ok = parse_action(raw)
            tr["calls"] += 1
            tr["prompt_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
            tr["completion_tokens"] += int(usage.get("completion_tokens", 0) or 0)
            tr["attempts"].append(dict(
                economic_attempt=attempt, schema_repair=rep, transport_attempts=n_tr,
                raw=str(raw)[:1200], parsed_ok=bool(ok), action_index=a, claimed_best_index=c,
                reasoning=reason, latency_s=round(time.time() - t0, 3),
                prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
                completion_tokens=int(usage.get("completion_tokens", 0) or 0)))
            if ok:
                break
            tr["schema_repairs"] += 1
            # Information-free repair: PR.SCHEMA_REPAIR_MSG names the format and nothing else. The
            # prompt gate proves it carries no objective lexicon, so a badly-formatting arm cannot
            # buy a hint with its own failure.
            msgs = msgs + [{"role": "assistant", "content": str(raw)},
                           {"role": "user", "content": PR.SCHEMA_REPAIR_MSG}]

        if not ok:
            raise TechnicalFailure(
                f"unparseable after {SCHEMA_REPAIR} schema repairs "
                f"(interface={interface} model={model_id} seed={seed} "
                f"t={state['t']} j={state['j']})")

        i = a
        tr["proposed_index"].append(int(a))
        if tr["claimed_best_index"] is None:
            tr["claimed_best_index"] = c
        r = rel_regret(v, i)
        if eta_g is None or r <= eta_g or attempt == MAX_ECON_RETRY:
            break
        msgs = msgs + [{"role": "assistant", "content": str(raw)},
                       {"role": "user", "content": PR.retry_factual(v, i, r, eta_g)}]
        tr["economic_retries"] += 1

    if eta_g is not None and rel_regret(v, i) > eta_g:
        tr["verification_failed"] = True
        if ovr_target is not None:
            i, tr["override"] = override_index(v, eta_g, ovr_target), True

    tr["transport_attempts"] = box[0]
    tr["latency_s"] = round(time.time() - t_run, 2)
    return finish(tr, v, i)


# ---------------------------------------------------------------------------------------------
# one market run
# ---------------------------------------------------------------------------------------------
def run_key(alias, interface, pol_name, seed):
    return f"{alias}|{interface}|{pol_name}|{seed}"


def state_fingerprint(r, T, f_last, tables):
    """A hash of everything a decision is allowed to see. Used by G-R2."""
    h = [np.asarray(r, float).tobytes(), np.float64(T).tobytes(),
         np.asarray(f_last, float).tobytes()] + [np.asarray(t, float).tobytes() for t in tables]
    return PR.sha("".join(x.hex() for x in h))


def simulate_run(agent, model_id, alias, interface, pol_name, seed, rounds=ROUNDS, zeta=ZETA,
                 parallel=True, prompt_fn=None, collect_state=None):
    """One market seed, all rounds. Returns (header, round_records).

    Nothing is written here. The caller writes only if this returns, which is how section 5 rule 4
    ("abort before the state transition") and rule 5 ("re-run the complete market seed") are made
    compatible with an append-only raw file.
    """
    kappa, tau = POLICIES[pol_name]
    cfg = E.Config(m=4)
    m = cfg.m
    qs, ps, bs, bidx = H.draw_market_seeded(cfg, seed)
    U = FC.crn_uniforms(seed, rounds, m, cfg.N_obs)
    rg = rbar_grid_for(cfg, kappa, tau)
    eq_f = H.hetero_equilibrium(cfg, qs, ps, bidx, OMEGA, LAM, rg)
    eq_idx = [int(round(float(x) * NIDX)) for x in eq_f]
    gstar_projected = float(A.gmv_of(cfg, qs, ps, bidx, OMEGA, LAM, rg, np.asarray(eq_f, float)))

    r = np.full(m, 0.5)
    T = 1.0
    w0 = E._w0(cfg, OMEGA)
    f_last = np.full(m, 0.5)
    y_last = np.zeros(m)
    profit_last = np.zeros(m)
    D_last = np.zeros(m, int)
    d_last = np.zeros(m)
    P_last = np.zeros(m)
    rbefore_last = r.copy()
    F_last = 0.0

    run_id = uuid.uuid4().hex[:16]
    rk = run_key(alias, interface, pol_name, seed)

    header = dict(
        kind="run_header", run_id=run_id, run_key=rk, schema_version="B9.1",
        model=model_id, model_alias=alias, interface=interface,
        policy=pol_name, kappa=kappa, tau=tau, seed=seed, rounds=rounds, zeta=zeta, eta=ETA,
        n_index=NIDX, tau_tie=TAU_TIE, omega=OMEGA, lam=LAM,
        provenance=dict(draw_fn="draw_market_seeded", crn_fn="crn_uniforms",
                        crn_seed_offset=90000, rbar_fn="rbar_grid",
                        prompt_module_sha256=PR.published_hashes()["module::source"]),
        # w0 and fgrid are recorded as VALUES, not as a recipe. The validator is forbidden to import
        # equilibrium.py, so a header that said "w0 = W0_BASE + W0_SPREAD(1-omega)" would force it to
        # either import the module or hard-code a constant it cannot check -- and a validator that
        # hard-codes the thing it is validating is a mirror. These two fields are what make
        # independent reconstruction of the displayed table possible at all.
        config=dict(m=cfg.m, alpha=cfg.alpha, beta=cfg.beta, gamma=cfg.gamma, Q0=cfg.Q0,
                    eta_r=cfg.eta_r, cs=cfg.cs, N_obs=cfg.N_obs, Nf=cfg.Nf,
                    w0=float(w0), erosion=getattr(cfg, "erosion", "exp"),
                    fgrid=[float(x) for x in cfg.fgrid]),
        market=dict(q=[float(x) for x in qs], p=[float(x) for x in ps], b=[float(x) for x in bs],
                    bidx=[int(x) for x in bidx], c=[0.5 * float(x) for x in ps]),
        # the stationary rows actually indexed to build every table this run. Recorded once, not
        # 80 times: they are constant in t, and duplicating 21 floats x 4 merchants x 80 rounds
        # would triple the file to restate a constant.
        rbar_used=[[float(x) for x in rg[int(bidx[j])]] for j in range(m)],
        initial_condition=dict(r_init=[0.5] * m, T_init=1.0, f_last_init=[0.5] * m),
        equilibrium=dict(eq_f=[float(x) for x in eq_f], eq_index=eq_idx,
                         gstar_projected=gstar_projected),
        budget=dict(max_economic_retries=MAX_ECON_RETRY, schema_repairs=SCHEMA_REPAIR,
                    transport_retries=TRANSPORT_RETRIES),
    )

    recs = []
    for t in range(1, rounds + 1):
        f_riv = [float(x) for x in f_last]
        tables = [payoff_table(cfg, qs, ps, bidx, rg, j, f_riv) for j in range(m)]
        fp_before = state_fingerprint(r, T, f_last, tables)

        states = [dict(t=t, j=j, q=float(qs[j]), p=float(ps[j]), b=float(bs[j]),
                       r=float(r[j]), T=float(T), zeta=zeta, f_last=float(f_last[j]),
                       y_last=float(y_last[j]), profit_last=float(profit_last[j]),
                       D_last=int(D_last[j]), d_last=float(d_last[j]), P_last=float(P_last[j]),
                       rbefore_last=float(rbefore_last[j]), F_last=float(F_last),
                       eq_index=eq_idx[j]) for j in range(m)]

        def one(j):
            return j, decide(agent, model_id, cfg, interface, POLICIES[pol_name], states[j],
                             tables[j], seed, prompt_fn)

        if interface in NOLLM_INTERFACES or not parallel:
            res = [one(j) for j in range(m)]
        else:
            res = list(ThreadPoolExecutor(max_workers=m).map(one, range(m)))
        trs = [None] * m
        for j, tr in res:
            trs[j] = tr

        fp_after = state_fingerprint(r, T, f_last, tables)
        if fp_before != fp_after:
            raise TechnicalFailure(f"state mutated during decision (seed={seed} t={t})")
        if collect_state is not None:
            collect_state.append((fp_before, fp_after))

        # ---- market step: identical to balance8_runner.simulate_run, proven by G-R1 -------------
        f = np.array([idx_to_f(tr["executed_index"]) for tr in trs])
        appeal = qs + (1.0 - qs) * f
        u = cfg.alpha * appeal + cfg.beta * r - cfg.gamma * ps
        exu = np.exp(u)
        den = math.exp(w0) + exu.sum()
        s = exu / den
        F = float(f.mean())
        Q = cfg.Q0 * T
        y = Q * s
        c = 0.5 * ps
        profit = (ps - c) * y
        gmv = float((ps * y).sum())
        theta = np.clip(bs + cfg.cs * f, 0, 1)
        Dcount = (U[t - 1] < theta[:, None]).sum(axis=1).astype(int)
        d = Dcount / cfg.N_obs
        Pen = kappa * np.maximum(0.0, d - tau)
        rbef = r.copy()
        r_next = np.clip(r + cfg.eta_r * (1 - r) - Pen, 0.0, 1.0)
        T_next = (1 - zeta) * T + zeta * math.exp(-LAM * F)
        # -----------------------------------------------------------------------------------------

        outside_share = float(math.exp(w0) / den)
        gmv_projected = float(A.gmv_of(cfg, qs, ps, bidx, OMEGA, LAM, rg, f))
        W = [realized_payoff_vector(cfg, qs, ps, bidx, rbef, T, f, j) for j in range(m)]

        merchants = []
        for j in range(m):
            tr = trs[j]
            v = tables[j]
            wj = W[j]
            i_exec = int(tr["executed_index"])
            merchants.append(dict(
                decision_id=f"{run_id}:{t}:{j}", j=j,
                q=float(qs[j]), p=float(ps[j]), b=float(bs[j]), c=float(c[j]),
                bidx=int(bidx[j]),
                # --- state ---
                r_current=float(rbef[j]), r_previous=float(rbefore_last[j]), r_after=float(r_next[j]),
                T_before=float(T), T_after=float(T_next),
                traffic_before=float(cfg.Q0 * T), traffic_after=float(cfg.Q0 * T_next),
                # --- actions ---
                action_index=i_exec, f=idx_to_f(i_exec),
                rival_index=[int(round(x * NIDX)) for x in f_riv],
                rival_f=[float(x) for x in f_riv],
                prev_action_index=[int(round(float(x) * NIDX)) for x in f_last],
                # --- displayed table ---
                displayed_table=[float(x) for x in v],
                argmax_index=int(tr["argmax_index"]), argmax_set=tr["argmax_set"],
                tau_tie=TAU_TIE,
                eq_index=eq_idx[j], eq_f=float(eq_f[j]),
                # --- decision ---
                proposed_index=list(tr["proposed_index"]),
                accepted_index=tr["accepted_index"], executed_index=i_exec,
                claimed_best_index=tr["claimed_best_index"],
                override=bool(tr["override"]), override_target=tr["override_target"],
                mark_index=tr["mark_index"], followed_mark=tr["followed_mark"],
                verification_failed=bool(tr["verification_failed"]),
                regret_proposed=tr["regret_proposed"], regret_accepted=tr["regret_accepted"],
                regret_executed=tr["regret_executed"],
                regret_executed_abs=tr["regret_executed_abs"],
                regret_per_attempt=tr["regret_per_attempt"],
                # --- outcomes ---
                realized_payoff_vector=[float(x) for x in wj],
                realized_argmax=int(PR.argmax_min_index(wj)),
                realized_regret_rel=float((wj.max() - wj[i_exec]) / max(abs(wj.max()), 1e-9)),
                share=float(s[j]), outside_share=outside_share,
                normalized_sales_y=float(y[j]), profit=float(profit[j]),
                theta=float(theta[j]), D_count=int(Dcount[j]), d_rate=float(d[j]),
                penalty=float(Pen[j]),
                # --- technical ---
                transport_attempts=int(tr["transport_attempts"]),
                schema_repairs=int(tr["schema_repairs"]),
                economic_retries=int(tr["economic_retries"]),
                calls=int(tr["calls"]), latency_s=tr["latency_s"],
                prompt_tokens=int(tr["prompt_tokens"]),
                completion_tokens=int(tr["completion_tokens"]),
                prompt_sha256=tr.get("prompt_sha256"),
                attempts=tr["attempts"],
            ))

        recs.append(dict(
            kind="round", run_id=run_id, run_key=rk, model=model_id, model_alias=alias,
            interface=interface, policy=pol_name, kappa=kappa, tau=tau, seed=seed, round=t,
            zeta=zeta, eta=ETA, n_index=NIDX,
            F=F, GMV=gmv, gmv_projected=gmv_projected, gstar_projected=gstar_projected,
            outside_share=outside_share,
            T=float(T), T_next=float(T_next),
            state_fingerprint=fp_before,
            crn_row=[int(x) for x in Dcount],
            merchants=merchants))

        f_last = f
        y_last = y
        profit_last = profit
        D_last = Dcount
        d_last = d
        P_last = Pen
        rbefore_last = rbef
        F_last = F
        T = T_next
        r = r_next

    return header, recs


def check_realized_identity(recs, tol=1e-9):
    """W_j(executed) must equal the profit the market actually paid. Returns the worst error.

    This is errata E-7's requirement turned into an assertion: the realized-payoff instrument is
    verified against a recorded transition before any statement is made with it.
    """
    worst = 0.0
    for rec in recs:
        for mm in rec["merchants"]:
            w = mm["realized_payoff_vector"][mm["executed_index"]]
            worst = max(worst, abs(w - mm["profit"]))
    return worst


# ---------------------------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------------------------
def append_jsonl(path: Path, rows):
    """Append-only. Refuses any path that is not a fresh balance9_* raw file."""
    name = path.name
    if not name.startswith("balance9_"):
        raise ValueError(f"refusing to write outside the balance9_ prefix: {name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def completed_run_keys(path: Path):
    """run_keys already present as completed runs, so a resumed driver never duplicates a seed."""
    done = set()
    if not path.exists():
        return done
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except Exception:                          # noqa: BLE001
                continue
            if d.get("kind") == "run_footer":
                done.add(d.get("run_key"))
    return done


# ---------------------------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------------------------
_WLOCK = __import__("threading").Lock()


def run_cell(alias, interface, pol_name, seeds, out_path, rounds=ROUNDS, zeta=ZETA,
             seed_workers=4, max_seed_attempts=3, agent="llm", prompt_fn=None, quiet=False,
             parallel=True):
    """One (model, interface, policy) cell over `seeds`. Returns a summary dict.

    Section 5 rules 4 and 5 are implemented here rather than inside `simulate_run`: a failed seed is
    re-run from round 1, in full, because a run resumed mid-trajectory has a different history and is
    therefore a different observation wearing the same seed. Nothing is written until a seed
    completes, so the raw file never contains a partial run for the driver to have to reason about.

    Seeds already carrying a `run_footer` are skipped, which makes the driver resumable without any
    file surgery -- the mandate forbids patching raw files, so resumption has to be append-only too.
    """
    out_path = Path(out_path)
    done = completed_run_keys(out_path)
    todo = [s for s in seeds if run_key(alias, interface, pol_name, s) not in done]
    model_id = MODELS.get(alias, alias)
    stats = dict(cell=f"{alias}|{interface}|{pol_name}", requested=len(seeds),
                 skipped_already_done=len(seeds) - len(todo), completed=0, failed=0,
                 seed_reruns=0, calls=0, transport_attempts=0, schema_repairs=0,
                 economic_retries=0, overrides=0, failures=[])

    def do_seed(seed):
        rk = run_key(alias, interface, pol_name, seed)
        last_err = None
        for attempt in range(1, max_seed_attempts + 1):
            t0 = time.time()
            try:
                header, recs = simulate_run(agent, model_id, alias, interface, pol_name, seed,
                                            rounds=rounds, zeta=zeta, prompt_fn=prompt_fn,
                                            parallel=parallel)
            except TechnicalFailure as exc:
                last_err = str(exc)[:300]
                if not quiet:
                    print(f"  [rerun] {rk} attempt {attempt}/{max_seed_attempts}: {last_err[:120]}")
                continue
            calls = sum(mm["calls"] for r in recs for mm in r["merchants"])
            tra = sum(mm["transport_attempts"] for r in recs for mm in r["merchants"])
            sch = sum(mm["schema_repairs"] for r in recs for mm in r["merchants"])
            eco = sum(mm["economic_retries"] for r in recs for mm in r["merchants"])
            ovr = sum(1 for r in recs for mm in r["merchants"] if mm["override"])
            werr = check_realized_identity(recs)
            if werr > 1e-8:
                # The instrument disagrees with the transition it just recorded. E-7 is precisely
                # this failure going unnoticed, so it aborts the seed rather than being logged.
                raise TechnicalFailure(f"realized-payoff identity violated ({werr:.2e}) on {rk}")
            footer = dict(kind="run_footer", run_id=header["run_id"], run_key=rk,
                          seed=seed, model_alias=alias, interface=interface, policy=pol_name,
                          rounds=len(recs), seed_attempts=attempt, calls=calls,
                          transport_attempts=tra, schema_repairs=sch, economic_retries=eco,
                          overrides=ovr, realized_identity_max_err=werr,
                          wall_s=round(time.time() - t0, 1))
            with _WLOCK:
                append_jsonl(out_path, [header] + recs + [footer])
            return dict(ok=True, seed=seed, attempt=attempt, calls=calls, transport=tra,
                        schema=sch, econ=eco, overrides=ovr)
        return dict(ok=False, seed=seed, error=last_err)

    if seed_workers > 1 and interface not in NOLLM_INTERFACES:
        results = list(ThreadPoolExecutor(max_workers=seed_workers).map(do_seed, todo))
    else:
        results = [do_seed(s) for s in todo]

    for r in results:
        if r["ok"]:
            stats["completed"] += 1
            stats["seed_reruns"] += r["attempt"] - 1
            stats["calls"] += r["calls"]
            stats["transport_attempts"] += r["transport"]
            stats["schema_repairs"] += r["schema"]
            stats["economic_retries"] += r["econ"]
            stats["overrides"] += r["overrides"]
        else:
            stats["failed"] += 1
            stats["failures"].append(dict(seed=r["seed"], error=r["error"]))
    return stats


# ---------------------------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------------------------
_R = []


def ck(label, cond, detail=""):
    _R.append((label, bool(cond)))
    print(("  ok   " if cond else "  FAIL ") + label + (f"   [{detail}]" if detail and not cond else ""))
    return bool(cond)


def gate_environment_matches_balance8(seeds=(9000, 9001, 9002), pols=("P_GMV", "P_robust")):
    """G-R1. The dynamics are Balance-8's, to the bit, on the deterministic no-LLM arms.

    This is the gate that makes 'clean replication' an honest phrase. Balance-8's runner is imported
    HERE ONLY -- the production path above never touches it, so the comparison is between two
    independent implementations of the same step rather than between a function and itself.
    """
    import balance8_runner as B8                       # noqa: PLC0415

    worst = 0.0
    n = 0
    for pol in pols:
        for seed in seeds:
            for itf in ("M", "O"):
                _, mine = simulate_run("mock", "", "x", itf, pol, seed, rounds=12, parallel=False)
                *_, theirs, _c = B8.simulate_run("mock", "", "x", itf, pol, POLICIES[pol], seed,
                                                 12, ZETA, parallel=False)
                if len(mine) != len(theirs):
                    return ck(f"G-R1 no-LLM trajectory matches balance8_runner", False,
                              "round count differs")
                for a, b in zip(mine, theirs):
                    for j in range(4):
                        ma, mb = a["merchants"][j], b["merchants"][j]
                        if ma["action_index"] != mb["action_index"]:
                            return ck("G-R1 no-LLM trajectory matches balance8_runner", False,
                                      f"action differs at seed={seed} t={a['round']} j={j}")
                        # Balance-8 ROUNDS its stored fields (profit 5dp, reputation 4dp, GMV 5dp);
                        # this runner stores full precision. Comparing raw would show ~5e-5 gaps that
                        # are entirely B8's quantisation, and calling that "bit-for-bit" under a
                        # 1e-4 tolerance would be exactly the kind of loose claim this project keeps
                        # correcting. So quantise MINE the way B8 quantises ITS and demand equality.
                        for key, dp in (("profit", 5), ("r_after", 4), ("theta", 4), ("d_rate", 4),
                                        ("penalty", 4), ("share", 4)):
                            worst = max(worst, abs(round(ma[key], dp) - mb[key]))
                        worst = max(worst, abs(round(a["GMV"], 5) - b["GMV"]))
                        worst = max(worst, abs(round(a["F"], 4) - b["F"]))
                        n += 1
    return ck(f"G-R1 no-LLM trajectory is EXACTLY balance8_runner's after applying B8's own storage "
              f"rounding ({n} merchant-rounds, worst |diff| = {worst:.1e})", worst == 0.0)


def gate_tables_full_precision(seeds=(9003, 9004), pols=("P_GMV", "P_sep")):
    """G-R6. The displayed tables equal Balance-8's at FULL precision, not just where printed.

    The prompt gate proved the printed table is byte-identical, but it prints 4 decimals. A table
    that agreed to 4dp and differed at 1e-9 would pass that gate and still change every argmax near
    a tie. Different question, different check.
    """
    import balance8_runner as B8                       # noqa: PLC0415

    cfg = E.Config(m=4)
    worst = 0.0
    n = 0
    for pol in pols:
        kappa, tau = POLICIES[pol]
        rg_a, rg_b = rbar_grid_for(cfg, kappa, tau), B8.rbar_grid_for(cfg, kappa, tau)
        if not np.array_equal(rg_a, rg_b):
            return ck("G-R6 displayed tables equal balance8 at full precision", False,
                      "rbar grids differ")
        for seed in seeds:
            qs, ps, bs, bidx = H.draw_market_seeded(cfg, seed)
            for f_riv in ([0.5] * 4, [0.0, 0.25, 0.75, 1.0], [0.3, 0.3, 0.3, 0.3]):
                for j in range(4):
                    va = payoff_table(cfg, qs, ps, bidx, rg_a, j, f_riv)
                    vb = B8.payoff_table(cfg, qs, ps, bidx, rg_b, j, f_riv)
                    worst = max(worst, float(np.abs(va - vb).max()))
                    n += 1
    return ck(f"G-R6 displayed tables equal balance8 at full precision "
              f"({n} tables, worst |diff| = {worst:.3e})", worst == 0.0)


def gate_state_invariance():
    """G-R2. Market state is bit-identical before and after every attempt of a decision.

    Protocol section 5 requires the runner to SHIP this test. A mock arm with a forced retry is used
    so the multi-attempt path is actually exercised -- testing the invariant on a single-attempt arm
    would prove nothing about retries, which is the case the invariant exists for.
    """
    seen = []
    _, recs = simulate_run("mock", "", "x", "M", "P_GMV", 9005, rounds=8, parallel=False,
                           collect_state=seen)
    ok_pairs = bool(seen) and all(a == b for a, b in seen)

    # now the retry path, with a stub that proposes a bad action twice then the argmax
    calls = {"n": 0}

    def stub(model_id, msgs, box):
        box[0] += 1
        calls["n"] += 1
        i = 0 if calls["n"] % 3 else NIDX
        return json.dumps({"action_index": i, "claimed_best_index": i, "reasoning": "s"}), {}, 1

    real, seen2 = transport_call_ref[0], []
    try:
        transport_call_ref[0] = stub
        globals()["transport_call"] = stub
        _, recs2 = simulate_run("llm", "stub", "x", "R", "P_GMV", 9006, rounds=4, parallel=False,
                                collect_state=seen2)
    finally:
        transport_call_ref[0] = real
        globals()["transport_call"] = real

    retried = any(mm["economic_retries"] > 0 for rec in recs2 for mm in rec["merchants"])
    ok2 = bool(seen2) and all(a == b for a, b in seen2)
    ck("G-R2a state bit-identical across attempts (no-LLM arm)", ok_pairs)
    ck("G-R2b the retry path was actually exercised by the stub", retried)
    return ck("G-R2 state bit-identical across attempts (retry arm)", ok2)


def gate_realized_identity():
    """G-R3. The realized-payoff instrument reproduces the recorded profit exactly."""
    worst = 0.0
    for pol in ("P_GMV", "P_robust"):
        for seed in (9007, 9008):
            _, recs = simulate_run("mock", "", "x", "M", pol, seed, rounds=10, parallel=False)
            worst = max(worst, check_realized_identity(recs))
    return ck(f"G-R3 W_j(executed) reproduces recorded profit (worst |err| = {worst:.3e})",
              worst < 1e-9)


def gate_no_fallback():
    """G-R4. No fabricated action anywhere on the LLM path.

    An AST scan, not a text grep: this module's own docstring discusses fallbacks, and a text scan
    cannot tell code from prose about code. That distinction cost Balance-8 a false positive and is
    recorded there; repeating the text-scan version here would repeat the mistake.
    """
    tree = ast.parse((HERE / "balance9_runner.py").read_text(encoding="utf-8"))
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.FloorDiv):
            if (isinstance(node.left, ast.Name) and node.left.id == "NIDX"
                    and isinstance(node.right, ast.Constant) and node.right.value == 2):
                bad.append("NIDX // 2")
        if isinstance(node, ast.Constant) and node.value == 0.5:
            pass  # 0.5 appears legitimately as cost share and initial reputation
    ck("G-R4a no 'NIDX // 2' midpoint fallback expression exists", not bad, str(bad))

    # the decide() function must raise on parse failure, never assign an action there
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "decide")
    raises = [n for n in ast.walk(fn) if isinstance(n, ast.Raise)]
    has_tf = any("TechnicalFailure" in ast.dump(n) for n in raises)
    return ck("G-R4 decide() raises TechnicalFailure instead of inventing an action", has_tf)


def gate_tie_rule():
    """G-R7. The declared tie rule and numpy's agree on every table visited -- and if they ever
    stop agreeing, the run must stop rather than silently switch arms."""
    cfg = E.Config(m=4)
    n = mism = 0
    for pol in ("P_GMV", "P_robust", "P_sep"):
        kappa, tau = POLICIES[pol]
        rg = rbar_grid_for(cfg, kappa, tau)
        for seed in (9009, 9010, 9011):
            qs, ps, bs, bidx = H.draw_market_seeded(cfg, seed)
            for f_riv in ([0.5] * 4, [0.1, 0.4, 0.6, 0.9]):
                for j in range(4):
                    v = payoff_table(cfg, qs, ps, bidx, rg, j, f_riv)
                    n += 1
                    if PR.argmax_min_index(v) != int(np.argmax(v)):
                        mism += 1
    ck(f"G-R7a declared tie rule agrees with np.argmax on {n} real tables ({mism} mismatches)",
       mism == 0)
    v = np.array([1.0] * 21)
    v[7] = 1.0 + 1e-13
    return ck("G-R7 the tie rule resolves a within-TAU_TIE near-tie to the LOWEST index",
              PR.argmax_min_index(v) == 0 and argmax_set(v) == list(range(21)))


def gate_schema_completeness():
    """G-R5. Every field protocol section 4 names is present in a real record."""
    header, recs = simulate_run("mock", "", "x", "M", "P_GMV", 9012, rounds=3, parallel=False)
    hdr_req = ["run_id", "provenance", "config", "market", "rbar_used", "initial_condition",
               "equilibrium", "policy", "kappa", "tau", "seed", "zeta", "budget"]
    missing_h = [k for k in hdr_req if k not in header]
    ck(f"G-R5a run header carries the run-level schema ({len(hdr_req)} fields)", not missing_h,
       str(missing_h))
    for k in ("q", "p", "b", "bidx", "c"):
        ck(f"G-R5b header.market carries {k}", k in header["market"])
    ck("G-R5c header records the market draw provenance",
       header["provenance"]["draw_fn"] == "draw_market_seeded")
    ck("G-R5d header records the explicit round-1 initial condition",
       set(header["initial_condition"]) == {"r_init", "T_init", "f_last_init"})
    ck("G-R5e header records the stationary rows actually indexed",
       len(header["rbar_used"]) == 4 and len(header["rbar_used"][0]) == NIDX + 1)
    ck("G-R5j header records w0 and fgrid as VALUES, so the validator needs no equilibrium import",
       isinstance(header["config"].get("w0"), float)
       and len(header["config"].get("fgrid", [])) == NIDX + 1)

    mreq = ["decision_id", "r_current", "r_previous", "T_before", "T_after", "traffic_before",
            "traffic_after", "rival_index", "rival_f", "prev_action_index", "displayed_table",
            "argmax_index", "argmax_set", "tau_tie", "proposed_index", "accepted_index",
            "executed_index", "override", "override_target", "claimed_best_index", "mark_index",
            "followed_mark", "regret_proposed", "regret_executed", "regret_per_attempt",
            "realized_payoff_vector", "realized_regret_rel", "share", "outside_share",
            "normalized_sales_y", "profit", "theta", "D_count", "d_rate", "penalty",
            "transport_attempts", "schema_repairs", "economic_retries", "calls", "attempts",
            "eq_index", "eq_f"]
    mm = recs[0]["merchants"][0]
    missing = [k for k in mreq if k not in mm]
    ck(f"G-R5f merchant record carries all {len(mreq)} section-4 fields", not missing, str(missing))
    ck("G-R5g displayed table has all 21 values", len(mm["displayed_table"]) == NIDX + 1)
    ck("G-R5h proposed_index is a LIST (every attempt), not a scalar",
       isinstance(mm["proposed_index"], list))
    ck("G-R5i the three technical counters are separate fields",
       len({"transport_attempts", "schema_repairs", "economic_retries"} & set(mm)) == 3)
    return ck("G-R5 round record carries GMV, the projected reference and the outside share",
              all(k in recs[0] for k in ("GMV", "gmv_projected", "gstar_projected",
                                         "outside_share")))


def gate_write_guard():
    """G-R8. The writer refuses any path outside the balance9_ prefix."""
    import tempfile                                    # noqa: PLC0415
    d = Path(tempfile.mkdtemp())
    bad = False
    try:
        append_jsonl(d / "balance8_main_raw.jsonl", [{"x": 1}])
    except ValueError:
        bad = True
    ck("G-R8a append_jsonl refuses a balance8_ path", bad)
    append_jsonl(d / "balance9_gate_raw.jsonl", [{"x": 1}])
    ck("G-R8b append_jsonl accepts a balance9_ path", (d / "balance9_gate_raw.jsonl").exists())
    return ck("G-R8 no prior raw file was opened for writing",
              not (d / "balance8_main_raw.jsonl").exists())


def gate_prompt_provenance():
    """G-R9. This module builds no prompt text of its own.

    The whole point of freezing the prompt surface in one module is defeated if the runner grows a
    second copy. Any long string literal here that looks like a prompt is a finding.
    """
    tree = ast.parse((HERE / "balance9_runner.py").read_text(encoding="utf-8"))

    # Skip docstrings. This module's prose necessarily quotes the JSON schema it parses, and the
    # first version of this gate fired on parse_action's own docstring. That is the third time in
    # Balance-8/9 that a text-shaped scanner has flagged an explanation of a thing as the thing
    # (balance8_runner's NIDX//2 grep, balance9_prompts' horizon gate, and now this one). The
    # general lesson is written down rather than re-learned: a scanner over source must be told the
    # difference between code and commentary about code, or it will price honesty as a defect.
    docs = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr) \
                and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
            docs.add(id(body[0].value))

    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docs:
            s = node.value
            if len(s) > 160 and ("action_index" in s or "You are" in s or "index [" in s):
                hits.append(s[:60])
    ck("G-R9a no prompt-shaped string literal is defined in the runner", not hits, str(hits))
    src = (HERE / "balance9_runner.py").read_text(encoding="utf-8")
    return ck("G-R9 prompts and repair messages are sourced from balance9_prompts",
              "PR.p3_prompt" in src and "PR.SCHEMA_REPAIR_MSG" in src and "PR.retry_factual" in src)


def gate_driver_failure_policy():
    """G-R10. The whole-seed rerun and resume rules actually hold under an injected failure.

    Three properties, each of which is a way a dataset gets silently corrupted:
      a. a seed that fails mid-run leaves NOTHING in the file (not a truncated run);
      b. a rerun starts from round 1, so the written run is a complete trajectory;
      c. a resumed driver skips completed seeds instead of appending a duplicate.
    """
    import tempfile                                    # noqa: PLC0415
    d = Path(tempfile.mkdtemp())
    out = d / "balance9_gate_driver_raw.jsonl"

    # (a) + (b): make the FIRST attempt at the seed die in round 3, then let everything succeed.
    # parallel=False so that call ordering is deterministic: with 4 merchant threads the injected
    # junk would scatter one response across four different merchants and none would exhaust its
    # repair budget, so the fault would silently fail to be injected and the gate would pass while
    # testing nothing. Rounds 1-2 use calls 1-8; merchant 0 of round 3 uses calls 9, 10, 11 (the
    # original plus SCHEMA_REPAIR=2 repairs), which is exactly the budget needed to abort the run.
    state = {"calls": 0}
    real = transport_call

    def flaky(model_id, msgs, box):
        box[0] += 1
        state["calls"] += 1
        if 9 <= state["calls"] <= 11:
            return "not json at all", {}, 1
        return json.dumps({"action_index": NIDX, "claimed_best_index": NIDX,
                           "reasoning": "s"}), {}, 1

    try:
        globals()["transport_call"] = flaky
        st = run_cell("x", "U", "P_GMV", [9013], out, rounds=4, seed_workers=1, agent="llm",
                      quiet=True, parallel=False)
    finally:
        globals()["transport_call"] = real

    rows = [json.loads(l) for l in open(out, encoding="utf-8")] if out.exists() else []
    kinds = [r["kind"] for r in rows]
    headers = kinds.count("run_header")
    footers = kinds.count("run_footer")
    rounds_written = kinds.count("round")

    ck("G-R10a a rerun leaves exactly one complete run in the file, not a truncated one",
       headers == footers == 1 and rounds_written == 4,
       f"headers={headers} footers={footers} rounds={rounds_written}")
    ck("G-R10b the driver recorded that the seed had to be re-run from scratch",
       st["seed_reruns"] >= 1, f"seed_reruns={st['seed_reruns']}")
    ck("G-R10c the rerun trajectory starts at round 1",
       sorted(r["round"] for r in rows if r["kind"] == "round") == [1, 2, 3, 4])

    # (c) resume: a second call must add nothing.
    before = len(rows)
    st2 = run_cell("x", "U", "P_GMV", [9013], out, rounds=4, seed_workers=1, agent="llm",
                   quiet=True, parallel=False)
    after = sum(1 for _ in open(out, encoding="utf-8")) if out.exists() else 0
    ck("G-R10d a resumed driver skips a completed seed instead of duplicating it",
       after == before and st2["skipped_already_done"] == 1 and st2["completed"] == 0)

    # (d) a seed that never succeeds writes nothing at all
    out2 = d / "balance9_gate_driver2_raw.jsonl"

    def always_bad(model_id, msgs, box):
        box[0] += 1
        return "still not json", {}, 1

    try:
        globals()["transport_call"] = always_bad
        st3 = run_cell("x", "U", "P_GMV", [9014], out2, rounds=3, seed_workers=1,
                       max_seed_attempts=2, agent="llm", quiet=True, parallel=False)
    finally:
        globals()["transport_call"] = real

    return ck("G-R10 a permanently failing seed writes NO raw record and is reported as failed",
              st3["failed"] == 1 and st3["completed"] == 0 and not out2.exists())


transport_call_ref = [transport_call]


def selftest():
    print("=" * 96)
    print("balance9_runner gates -- environment fidelity, schema completeness, failure policy")
    print("=" * 96)
    gate_environment_matches_balance8()
    gate_tables_full_precision()
    gate_state_invariance()
    gate_realized_identity()
    gate_no_fallback()
    gate_tie_rule()
    gate_schema_completeness()
    gate_write_guard()
    gate_prompt_provenance()
    gate_driver_failure_policy()
    n = sum(1 for _, ok in _R if ok)
    print("-" * 96)
    print(f"{n}/{len(_R)} gates passed")
    if n != len(_R):
        raise SystemExit("GATE FAILED. The runner may not be used to spend an API call.")
    print("runner gates PASSED.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
    else:
        ap.print_help()
