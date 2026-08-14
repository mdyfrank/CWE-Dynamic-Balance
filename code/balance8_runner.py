r"""BALANCE-8 runner: corrected decision layer. See BALANCE8_PROTOCOL.md secs 2-4.

WHAT IS DIFFERENT FROM balance7_runner.py, AND WHY

1. NO FALLBACK ACTION. EVER. Balance-7 line 229 did

       i = amax if interface == "G" else NIDX // 2   # documented fallback, logged

   so an unparseable response gave G the argmax (displayed regret exactly 0) and everyone else f=0.5
   (arbitrary regret). That is not a logging detail: it contaminates every cross-interface comparison in
   proportion to each arm's parse-failure rate, and it does so in the direction of the headline claim.
   Symmetrising it would not help -- any fallback invents an action the model never chose and then
   scores the model on it. Balance-8 issues information-free schema repairs and, failing those, raises
   TechnicalFailure: the run is ABORTED, no action executes, no state advances, nothing is written to
   raw. The run is unscorable, which is a different thing from failed.

2. THE SCHEMA REPAIR LEAKS NOTHING. It restates the JSON schema and says nothing about the table, the
   argmax, the tolerance, regret, or what the model previously said. It is one frozen string, identical
   for every interface, and `selftest` asserts that.

3. PROPOSED AND EXECUTED REGRET ARE SEPARATE FIELDS. Balance-7 overwrote regret with 0.0 when G
   overrode, and that zero became "G achieves zero displayed regret" -- a statement about the guard, not
   about the model. Here `regret_proposed` is the behavioural quantity and `regret_executed` is the
   system outcome, and no analyzer may conflate them.

4. NEW ARMS. M (machine argmax, no LLM) and O (equilibrium play, no LLM) are the controls the earlier
   sessions lacked: BALANCE8_RATIONAL_AUDIT.md sec.5 shows a round-1 copy of the mark is within eta in
   100% of P_GMV cells, so "H works" and "H transports a programmatic answer" are indistinguishable
   without them. HR/HC/HD/HS/RO are the mechanism diagnostics of protocol sec.4.1.

5. TRUTHFULNESS IS A CONSTRAINT, NOT A PREFERENCE. H's factual mark is kept byte-identical to
   Balance-7's so H still replicates. The decoy arm therefore cannot reuse that sentence, because
   pointing it at a non-argmax row would make it false. HD instead uses a RECOMMENDATION sentence, which
   is true of any real row, and HC is the same sentence aimed at the argmax -- so HD-vs-HC isolates the
   target and HC-vs-H isolates the phrasing. Stale-H states its own staleness explicitly and is true.

Usage:
    python balance8_runner.py --selftest
    python balance8_runner.py --config configs/<cfg>.json --mock
    python balance8_runner.py --config configs/<cfg>.json --run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import equilibrium as E  # noqa: E402
import hetero as H  # noqa: E402
import phase2_llm as L  # noqa: E402
import phase2_framing_controlled as FC  # noqa: E402
import phase2_exploitability as X  # noqa: E402

DATA = HERE / "data"
OMEGA, LAM = FC.OMEGA, FC.LAM
NIDX = 20
ETA = 0.01
MAX_RETRY = 2          # verified-retry budget (EXPERIMENTAL; R and G only)
SCHEMA_REPAIR = 2      # information-free parse repair (NOT experimental; every interface)
API_RETRIES = 6        # transport retries (NOT experimental; every interface)

LLM_INTERFACES = ["U", "H", "R", "G", "HR", "HC", "HD", "HS", "RO", "R0", "G0", "GB"]
NOLLM_INTERFACES = ["M", "O", "GS"]
INTERFACES = LLM_INTERFACES + NOLLM_INTERFACES

# The three mark sentences. Frozen here so that a diff of this file is a diff of the treatment.
MARK_FACTUAL = " The highest projected profit in this list is at index [{k}]."
MARK_RECOMMEND = " One option worth considering is index [{k}]."
MARK_STALE = " In the table shown one round earlier, the highest projected profit was at index [{k}]."

SCHEMA_REPAIR_MSG = (
    "Your previous message could not be parsed. Reply with nothing except a single JSON object of "
    'exactly this form: {"action_index": <integer 0-20>, "claimed_best_index": <integer 0-20>, '
    '"reasoning": "<one short sentence>"}. No prose, no code fences, no explanation outside the JSON.')


class TechnicalFailure(RuntimeError):
    """Unparseable after every schema repair. The run aborts; no action is invented."""


# ---------------------------------------------------------------------------------------------
# economics (identical objects to balance7_runner -- deliberately, so replication is meaningful)
# ---------------------------------------------------------------------------------------------
def idx_to_f(i):
    return round(float(i) / NIDX, 4)


def rbar_grid_for(cfg, kappa, tau):
    return H.rbar_grid(cfg, 0.0, 0.0) if kappa == 0 else H.rbar_grid(cfg, kappa, tau)


def payoff_table(cfg, qs, ps, bidx, rg, j, f_rivals):
    base = np.array(f_rivals, float)
    v = np.empty(NIDX + 1)
    for i in range(NIDX + 1):
        fp = base.copy(); fp[j] = idx_to_f(i)
        v[i] = float(X.profits(cfg, qs, ps, bidx, OMEGA, LAM, rg, fp)[j])
    return v


def rel_regret(v, i):
    """Relative PRIVATE DISPLAYED regret of action index i. Frozen definition, unchanged from B7."""
    if i is None:
        return float("nan")
    return float((v.max() - v[i]) / max(abs(v.max()), 1e-9))


def certificate_set(v, eta):
    """{ i : rel_regret(v, i) <= eta }. Always non-empty (argmax has regret 0)."""
    return [i for i in range(NIDX + 1) if rel_regret(v, i) <= eta + 1e-12]


def balanced_in_certificate(v, eta):
    """The SMALLEST f (= smallest index, = most balanced) action whose displayed private regret is
    still within the guard's own tolerance eta.

    This is the constructive counterpart of Theorem A7 (BALANCE8_THEORY.md sec.5). A7 proves that an
    eta-certificate on private displayed regret does NOT bound platform loss, with measured leverage up
    to 18.0x. If that is true, then a guard may move a long way in the platform's preferred direction
    while emitting exactly the same private guarantee it emitted before. `GS`/`GB` below do that, and
    the executed regret they produce is <= eta BY CONSTRUCTION, so the private certificate is provably
    identical to `G`'s. Any outcome difference is therefore attributable to the enforcement rule alone.
    """
    return int(min(certificate_set(v, eta)))


# Guard specification. (eta_guard, override_target). override_target None = retry only, no override.
# `R`/`G` reproduce Balance-7 exactly and are NOT modified. The rest are new in Balance-8.
GUARD_SPEC = {
    "R":  (ETA, None),          # frozen B7 arm: verified retry at 1%, no override
    "G":  (ETA, "argmax"),      # frozen B7 arm: verified retry at 1%, override to displayed argmax
    "R0": (0.0, None),          # NEW: EXACT retry, no override      -> isolates the retry channel
    "G0": (0.0, "argmax"),      # NEW: EXACT guard                   -> tests A5 at eta = 0
    "GB": (ETA, "balanced"),    # NEW: same 1% certificate as G, platform-preferring override target
}


def guard_of(interface):
    return GUARD_SPEC.get(interface, (None, None))


def override_index(v, eta, target):
    return int(np.argmax(v)) if target == "argmax" else balanced_in_certificate(v, eta)


def decoy_index(v):
    """Frozen, deterministic decoy: the non-argmax action whose displayed regret is closest to 5%.

    Pre-registered rather than chosen: 5% is comfortably above eta=0.01 so that following the decoy is
    detectable, and far below the tail so that the decoy is still a plausible row rather than an
    obviously absurd one. BALANCE8_RATIONAL_AUDIT.md sec.4 measured the median relative action margin at
    0.0069, so marking the second-best row would have been undetectable by construction.
    """
    a = int(np.argmax(v))
    cand = [i for i in range(NIDX + 1) if i != a]
    return min(cand, key=lambda i: abs(rel_regret(v, i) - 0.05))


# ---------------------------------------------------------------------------------------------
# prompt construction
# ---------------------------------------------------------------------------------------------
JSON_INSTR = ('Respond ONLY as compact JSON: {"action_index": <integer 0-20>, '
              '"claimed_best_index": <integer 0-20>, "reasoning": "<one short sentence>"} '
              "where your chosen exaggeration level is f = action_index / 20, and claimed_best_index "
              "is the index of the row you believe has the highest projected profit.")


def table_rows(v, order=None):
    """Row text. `order` permutes the DISPLAY ORDER only; every row keeps its true index label, so the
    model's answer needs no remapping and HR isolates position-vs-content with nothing else moving."""
    idxs = range(NIDX + 1) if order is None else order
    return "; ".join(f"[{i}] f={idx_to_f(i):.2f} -> profit={v[i]:.4f}" for i in idxs)


def table_block(v, interface, order=None, stale_v=None):
    """The interface-specific block. Returns (text, mark_index_or_None)."""
    if interface == "RO":                       # recommendation only: no table at all
        k = int(np.argmax(v))
        return (" Holding the other sellers at their most recent choices, consider your candidate "
                "exaggeration levels f = index/20 for index 0 to 20." + MARK_RECOMMEND.format(k=k)), k
    s = (" Holding the other sellers at their most recent choices, your projected per-round profit "
         f"for each candidate exaggeration level is: {table_rows(v, order)}.")
    # R0/G0/GB see EXACTLY the table U/R/G see. A guard is a wrapper around the same decision problem,
    # not a different decision problem, so the first-turn prompt must be byte-identical; only the
    # verification loop differs. (Verified: prompt_hash("G0") == prompt_hash("G"), see
    # data/balance8_p4_patch_verification.json.)
    if interface in ("U", "R", "G", "R0", "G0", "GB"):
        return s, None
    if interface in ("H", "HR"):
        k = int(np.argmax(v)); return s + MARK_FACTUAL.format(k=k), k
    if interface == "HC":
        k = int(np.argmax(v)); return s + MARK_RECOMMEND.format(k=k), k
    if interface == "HD":
        k = decoy_index(v); return s + MARK_RECOMMEND.format(k=k), k
    if interface == "HS":
        k = int(np.argmax(v if stale_v is None else stale_v))
        return s + MARK_STALE.format(k=k), k
    raise ValueError(f"unknown interface {interface}")


def build_prompt(cfg, interface, policy, state, v, order=None, stale_v=None):
    base_state = dict(t=state["t"], q=state["q"], p=state["p"], b=state["b"], r=state["r"], T=state["T"],
                      zeta=state["zeta"], f_last=state.get("f_last", 0.0), y_last=state.get("y_last", 0.0),
                      profit_last=state.get("profit_last", 0.0), D_last=state.get("D_last", 0),
                      d_last=state.get("d_last", 0.0), P_last=state.get("P_last", 0.0),
                      rbefore_last=state.get("rbefore_last", state["r"]), F_last=state.get("F_last", 0.0))
    system_core, user = FC.build_prompt(cfg, "C0_numeric_only", policy, base_state)
    block, mark = table_block(v, interface, order, stale_v)
    marker = "Respond ONLY as compact JSON"
    system = system_core[:system_core.index(marker)] + block.strip() + "\n" + JSON_INSTR
    return system, user, mark


def retry_message(v, i_chosen, r, eta=ETA):
    """Factual, preference-free validator feedback.

    At eta = ETA this is BYTE-IDENTICAL to Balance-7's message (default argument preserves the frozen
    R/G arms exactly). The eta = 0 arms (R0, G0) cannot honestly say "tolerance of 1%", so they state
    their own tolerance instead; the sentence is otherwise unchanged and still preference-free -- it
    reports the shortfall and the stated tolerance, and never says which index to pick.
    """
    tol = "1%" if abs(eta - ETA) < 1e-15 else (
        "0% (the submitted index must be the highest-profit index in the table)"
        if eta == 0.0 else f"{eta*100:.2f}%")
    return (f"Your submitted action_index [{i_chosen}] (f={idx_to_f(i_chosen):.2f}) has projected "
            f"profit {v[i_chosen]:.4f}. Relative to the highest projected profit in the table shown, "
            f"that is a shortfall of {r*100:.2f}%, which does not meet the stated tolerance of {tol} "
            f"for maximizing your own projected profit. The table is unchanged. "
            "Please reconsider and respond again in the same JSON format.")


def prompt_hash(interface, seed=999999):
    cfg = E.Config(m=4)
    qs, ps, bs, bidx = H.draw_market_seeded(cfg, seed)
    rg = rbar_grid_for(cfg, 1.0, 0.25)
    v = payoff_table(cfg, qs, ps, bidx, rg, 0, [0.3] * cfg.m)
    st = dict(t=2, j=0, q=.4, p=1., b=.05, r=.5, T=.9, zeta=0.2, f_last=.3, y_last=.5, profit_last=.2,
              D_last=3, d_last=.075, P_last=0., rbefore_last=.5, F_last=.3)
    order = display_order(interface, 999999, 2, 0)
    s, _, _ = build_prompt(cfg, interface, (1.0, .25), st, v, order)
    return hashlib.sha256(s.encode()).hexdigest()


def display_order(interface, seed, t, j):
    """HR's row permutation: deterministic in (seed, round, merchant) so it is reproducible from the
    raw record alone, and different every round so a model cannot learn a fixed layout."""
    if interface != "HR":
        return None
    return list(np.random.default_rng(int(seed) * 100003 + t * 101 + j).permutation(NIDX + 1))


# ---------------------------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------------------------
def parse_action(raw):
    """-> (action_index, claimed_best_index|None, reasoning, ok). Never silently coerces."""
    try:
        o = json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw or "", re.S)
        if not m:
            return None, None, "", False
        try:
            o = json.loads(m.group(0))
        except Exception:
            return None, None, "", False
    if not isinstance(o, dict):
        return None, None, "", False
    a = o.get("action_index")
    if isinstance(a, str):
        a = a.strip()
        a = int(a) if re.fullmatch(r"\d+", a) else None
    if isinstance(a, bool) or not isinstance(a, (int, float)):
        return None, None, str(o.get("reasoning", ""))[:200], False
    if float(a) != int(a):
        return None, None, str(o.get("reasoning", ""))[:200], False
    a = int(a)
    if not (0 <= a <= NIDX):
        return None, None, str(o.get("reasoning", ""))[:200], False
    c = o.get("claimed_best_index")
    try:
        c = int(c) if c is not None and 0 <= int(c) <= NIDX else None
    except Exception:
        c = None
    return a, c, str(o.get("reasoning", ""))[:200], True


# ---------------------------------------------------------------------------------------------
# one merchant decision
# ---------------------------------------------------------------------------------------------
def new_trace(interface, v, mark):
    return dict(interface=interface, argmax_index=int(np.argmax(v)), mark_index=mark,
                retries=0, schema_repairs=0, verification_failed=False, override=False,
                proposed_index=None, claimed_best_index=None, calls=0, latency_s=0.0,
                prompt_tokens=0, completion_tokens=0, followed_mark=None)


def finish(tr, v, i_exec):
    tr["executed_index"] = int(i_exec)
    tr["regret_proposed"] = rel_regret(v, tr["proposed_index"])
    tr["regret_executed"] = rel_regret(v, i_exec)
    if tr["mark_index"] is not None and tr["proposed_index"] is not None:
        tr["followed_mark"] = bool(tr["proposed_index"] == tr["mark_index"])
    return tr


def decide(agent, model_id, cfg, interface, policy, state, v, rng, seed, stale_v=None):
    """Executed action + full verification trace. Raises TechnicalFailure rather than inventing one."""
    amax = int(np.argmax(v))

    if interface in NOLLM_INTERFACES:
        tr = new_trace(interface, v, None)
        # M: displayed argmax. O: the frozen one-shot equilibrium index. GS: the most balanced action
        # that is still inside the SAME 1% private certificate G emits (Theorem A7 constructive test).
        i = (amax if interface == "M" else
             balanced_in_certificate(v, ETA) if interface == "GS" else int(state["eq_index"]))
        tr["proposed_index"] = i
        return finish(tr, v, i)

    eta_g, ovr_target = guard_of(interface)

    order = display_order(interface, seed, state["t"], state["j"])
    system, user, mark = build_prompt(cfg, interface, policy, state, v, order, stale_v)
    tr = new_trace(interface, v, mark)

    if agent == "mock":
        i = mock_index(interface, v, mark, rng)
        tr["proposed_index"] = i
        if eta_g is not None:
            n = 0
            while rel_regret(v, i) > eta_g and n < MAX_RETRY:
                n += 1
                i = mock_index(interface, v, mark, rng, retry=n)
            tr["retries"] = n
            if rel_regret(v, i) > eta_g:
                tr["verification_failed"] = True
                if ovr_target is not None:
                    i, tr["override"] = override_index(v, eta_g, ovr_target), True
        return finish(tr, v, i)

    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    t0 = time.time()
    i = None
    raw = reason = ""
    for attempt in range(MAX_RETRY + 1):
        # --- one verified-retry attempt, with its own information-free schema-repair budget ---------
        ok = False
        for rep in range(SCHEMA_REPAIR + 1):
            raw, usage = L.openrouter_call(model_id, "", "", messages=msgs, retries=API_RETRIES)
            tr["calls"] += 1
            tr["prompt_tokens"] += int((usage or {}).get("prompt_tokens", 0) or 0)
            tr["completion_tokens"] += int((usage or {}).get("completion_tokens", 0) or 0)
            a, c, reason, ok = parse_action(raw)
            if ok:
                break
            tr["schema_repairs"] += 1
            msgs = msgs + [{"role": "assistant", "content": raw},
                           {"role": "user", "content": SCHEMA_REPAIR_MSG}]
        if not ok:
            raise TechnicalFailure(
                f"unparseable after {SCHEMA_REPAIR} schema repairs "
                f"(interface={interface} model={model_id} seed={seed} t={state['t']} j={state['j']})")
        i = a
        if tr["proposed_index"] is None:
            tr["proposed_index"], tr["claimed_best_index"] = a, c
        r = rel_regret(v, i)
        if eta_g is None or r <= eta_g or attempt == MAX_RETRY:
            break
        msgs = msgs + [{"role": "assistant", "content": raw},
                       {"role": "user", "content": retry_message(v, i, r, eta_g)}]
        tr["retries"] += 1

    if eta_g is not None and rel_regret(v, i) > eta_g:
        tr["verification_failed"] = True
        if ovr_target is not None:
            i, tr["override"] = override_index(v, eta_g, ovr_target), True
    tr["reasoning"] = reason[:200]
    tr["raw"] = (raw or "")[:600]
    tr["latency_s"] = round(time.time() - t0, 2)
    return finish(tr, v, i)


def mock_index(interface, v, mark, rng, retry=0):
    """Pipeline test only; never an experimental result. Copies the mark when one is present, which is
    the behaviour the diagnostics are designed to detect, so the mock exercises those code paths."""
    amax = int(np.argmax(v))
    if mark is not None and rng.random() < 0.8:
        return int(mark)
    if retry >= 1:
        return amax if rng.random() < (0.75 + 0.2 * retry) else int(np.clip(amax + rng.integers(-2, 3), 0, NIDX))
    return amax if rng.random() < 0.45 else int(np.clip(amax + rng.integers(-4, 5), 0, NIDX))


# ---------------------------------------------------------------------------------------------
# simulation
# ---------------------------------------------------------------------------------------------
def run_key(alias, interface, pol_name, seed):
    return f"{alias}|{interface}|{pol_name}|s{seed}"


def simulate_run(agent, model_id, alias, interface, pol_name, policy, seed, rounds, zeta, parallel=True):
    kappa, tau = policy
    cfg = E.Config(m=4); m = cfg.m
    qs, ps, bs, bidx = H.draw_market_seeded(cfg, seed)
    U = FC.crn_uniforms(seed, rounds, m, cfg.N_obs)
    mrng = np.random.default_rng(7000 + seed)
    rg = rbar_grid_for(cfg, kappa, tau)
    eq_f = H.hetero_equilibrium(cfg, qs, ps, bidx, OMEGA, LAM, rg)
    eq_idx = [int(round(float(x) * NIDX)) for x in eq_f]

    r = np.full(m, 0.5); T = 1.0; w0 = E._w0(cfg, OMEGA)
    f_last = np.full(m, 0.5); y_last = np.zeros(m); profit_last = np.zeros(m); D_last = np.zeros(m, int)
    d_last = np.zeros(m); P_last = np.zeros(m); rbefore_last = r.copy(); F_last = 0.0
    prev_tables = None
    rk = run_key(alias, interface, pol_name, seed); recs = []; calls = 0

    for t in range(1, rounds + 1):
        f_riv = [float(x) for x in f_last]
        tables = [payoff_table(cfg, qs, ps, bidx, rg, j, f_riv) for j in range(m)]
        states = [dict(t=t, j=j, q=float(qs[j]), p=float(ps[j]), b=float(bs[j]), r=float(r[j]), T=float(T),
                       zeta=zeta, f_last=float(f_last[j]), y_last=float(y_last[j]),
                       profit_last=float(profit_last[j]), D_last=int(D_last[j]), d_last=float(d_last[j]),
                       P_last=float(P_last[j]), rbefore_last=float(rbefore_last[j]), F_last=float(F_last),
                       eq_index=eq_idx[j]) for j in range(m)]

        def one(j):
            sv = None if prev_tables is None else prev_tables[j]
            return j, decide(agent, model_id, cfg, interface, policy, states[j], tables[j], mrng, seed, sv)

        if agent == "mock" or interface in NOLLM_INTERFACES or not parallel:
            res = [one(j) for j in range(m)]
        else:
            res = list(ThreadPoolExecutor(max_workers=m).map(one, range(m)))
        trs = [None] * m
        for j, tr in res:
            trs[j] = tr; calls += tr["calls"]

        f = np.array([idx_to_f(tr["executed_index"]) for tr in trs])
        appeal = qs + (1.0 - qs) * f
        u = cfg.alpha * appeal + cfg.beta * r - cfg.gamma * ps
        exu = np.exp(u); den = math.exp(w0) + exu.sum(); s = exu / den
        F = float(f.mean()); Q = cfg.Q0 * T; y = Q * s; c = 0.5 * ps; profit = (ps - c) * y
        gmv = float((ps * y).sum())
        theta = np.clip(bs + cfg.cs * f, 0, 1)
        Dcount = (U[t - 1] < theta[:, None]).sum(axis=1).astype(int); d = Dcount / cfg.N_obs
        Pen = kappa * np.maximum(0.0, d - tau); rbef = r.copy()
        r = np.clip(r + cfg.eta_r * (1 - r) - Pen, 0.0, 1.0)
        T_next = (1 - zeta) * T + zeta * math.exp(-LAM * F)

        merchants = [dict(j=j, q=round(float(qs[j]), 4), p=round(float(ps[j]), 4), c=round(float(c[j]), 4),
                          b=round(float(bs[j]), 4), f=round(float(f[j]), 4),
                          action_index=int(trs[j]["executed_index"]),
                          proposed_index=trs[j]["proposed_index"],
                          claimed_best_index=trs[j]["claimed_best_index"],
                          argmax_index=int(trs[j]["argmax_index"]),
                          mark_index=trs[j]["mark_index"], followed_mark=trs[j]["followed_mark"],
                          eq_index=eq_idx[j],
                          regret_proposed=round(float(trs[j]["regret_proposed"]), 6),
                          regret_executed=round(float(trs[j]["regret_executed"]), 6),
                          retries=int(trs[j]["retries"]), schema_repairs=int(trs[j]["schema_repairs"]),
                          override=bool(trs[j]["override"]),
                          verification_failed=bool(trs[j]["verification_failed"]),
                          share=round(float(s[j]), 4), normalized_sales_y=round(float(y[j]), 5),
                          profit=round(float(profit[j]), 5), theta=round(float(theta[j]), 4),
                          D_count=int(Dcount[j]), d_rate=round(float(d[j]), 4),
                          penalty=round(float(Pen[j]), 4), r_before=round(float(rbef[j]), 4),
                          r_after=round(float(r[j]), 4),
                          reasoning=trs[j].get("reasoning", "")[:200],
                          raw=trs[j].get("raw", "")[:600], latency_s=trs[j]["latency_s"],
                          calls=int(trs[j]["calls"])) for j in range(m)]

        recs.append(dict(run_key=rk, model=model_id, model_alias=alias, interface=interface,
                         policy=pol_name, kappa=kappa, tau=tau, seed=seed, round=t, zeta=zeta,
                         eta=ETA, n_index=NIDX, F=round(F, 4), GMV=round(gmv, 5), T=round(T, 5),
                         T_next=round(T_next, 5), merchants=merchants))
        prev_tables = tables
        f_last = f; y_last = y; profit_last = profit; D_last = Dcount; d_last = d
        P_last = Pen; rbefore_last = rbef; F_last = F; T = T_next

    return alias, interface, pol_name, seed, recs, calls


# ---------------------------------------------------------------------------------------------
# self-test: the invariants that the corrections are supposed to guarantee
# ---------------------------------------------------------------------------------------------
def selftest():
    cfg = E.Config(m=4)
    qs, ps, bs, bidx = H.draw_market_seeded(cfg, 999998)
    rg = rbar_grid_for(cfg, 0.5, 0.20)
    v = payoff_table(cfg, qs, ps, bidx, rg, 0, [0.5] * 4)
    st = dict(t=3, j=0, q=.4, p=1., b=.05, r=.5, T=.9, zeta=0.2, f_last=.3, y_last=.5, profit_last=.2,
              D_last=3, d_last=.075, P_last=0., rbefore_last=.5, F_last=.3, eq_index=7)
    fails = []

    def chk(name, cond, extra=""):
        print(f"  {'PASS' if cond else 'FAIL'}  {name}{(' -- ' + extra) if extra and not cond else ''}")
        if not cond:
            fails.append(name)

    # 1. No fallback expression anywhere in the EXECUTABLE source.
    #
    # The first version of this check grepped the file text for "NIDX // 2" and failed -- on this
    # module's own docstring, which quotes the Balance-7 defect verbatim in order to explain it. A
    # text scan cannot tell code from prose about code, so it would have forced me to either delete
    # the explanation or ignore a red check, and both of those are worse than the defect. Parse the
    # AST instead and look for the expression itself: a docstring is a Constant, never a BinOp.
    import ast
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    bad = [n for n in ast.walk(tree)
           if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.FloorDiv, ast.Div))
           and isinstance(n.left, ast.Name) and n.left.id == "NIDX"]
    chk("no NIDX//2 fallback expression in executable code", not bad,
        f"{len(bad)} occurrence(s) at line(s) {[n.lineno for n in bad]}")
    # No interface-conditional action assignment of the form `amax if interface == ... else ...`
    # ANYWHERE ON THE LLM PATH. Scoping matters here and the first version of this check got it wrong:
    # it flagged the M/O branch, where `amax if interface == "M" else eq_index` is the DEFINITION of
    # the machine-argmax control, not a fallback -- it runs before any API call and no parse has
    # happened. What made Balance-7's line 229 a defect was that it fabricated an action *after a
    # parse failure*, so the exclusion is exactly the no-LLM early-return block and nothing else.
    dec = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "decide")
    nollm = [n for n in dec.body if isinstance(n, ast.If)
             and any(isinstance(c, ast.Compare) and any(isinstance(x, ast.Name)
                     and x.id == "NOLLM_INTERFACES" for x in ast.walk(c)) for c in ast.walk(n.test))]
    excluded = {id(x) for blk in nollm for x in ast.walk(blk)}
    cond = [n for n in ast.walk(dec) if isinstance(n, ast.IfExp) and id(n) not in excluded
            and any(isinstance(c, ast.Compare) and isinstance(c.left, ast.Name)
                    and c.left.id == "interface" for c in ast.walk(n.test))
            and any(isinstance(x, ast.Name) and x.id == "amax" for x in ast.walk(n))]
    chk("no interface-conditional fallback action on the LLM path", not cond,
        f"line(s) {[n.lineno for n in cond]}")
    chk("the no-LLM early return exists and is what was excluded", len(nollm) == 1)

    # 2. the schema repair is one frozen string that leaks nothing
    leak = [w for w in ("table", "profit", "highest", "index [", "tolerance", "regret", "argmax",
                        "shortfall") if w.lower() in SCHEMA_REPAIR_MSG.lower()]
    chk("schema repair leaks no task information", not leak, f"leaked {leak}")

    # 3. the prompts of every LLM interface differ ONLY in the block we intend
    base, _, _ = build_prompt(cfg, "U", (0.5, .20), st, v)
    hashes = {}
    for itf in LLM_INTERFACES:
        s, _, mk = build_prompt(cfg, itf, (0.5, .20), st, v, display_order(itf, 5, 3, 0), v)
        hashes[itf] = hashlib.sha256(s.encode()).hexdigest()[:12]
    chk("U/R/G prompts are byte-identical (they must be: only the validator differs)",
        len({hashes["U"], hashes["R"], hashes["G"]}) == 1, str(hashes))
    chk("every guard arm sees the SAME first-turn prompt as U (the guard is a wrapper, not a task)",
        len({hashes[i] for i in ("U", "R", "G", "R0", "G0", "GB")}) == 1, str(hashes))
    chk("every marked interface differs from U", all(hashes[i] != hashes["U"]
                                                     for i in ("H", "HR", "HC", "HD", "HS", "RO")))

    # 4. the decoy is a real, non-argmax row that is detectably worse than eta
    d = decoy_index(v); a = int(np.argmax(v))
    chk("decoy is not the argmax", d != a)
    chk("decoy regret is well above eta", rel_regret(v, d) > 5 * ETA,
        f"decoy regret {rel_regret(v, d):.4f}")

    # 5. HR permutes display order but keeps true labels, so answers need no remapping
    o = display_order("HR", 5, 3, 0)
    chk("HR is a true permutation", sorted(o) == list(range(NIDX + 1)))
    chk("HR row text contains every true index label",
        all(f"[{i}]" in table_rows(v, o) for i in range(NIDX + 1)))
    chk("HR marks the true argmax", table_block(v, "HR", o)[1] == a)

    # 6. HS is stale: with a different previous table it must mark the PREVIOUS argmax.
    #    Search for a market where the two argmaxes actually differ rather than skipping -- a test that
    #    silently skips is a test that can rot without anyone noticing. If no such market exists in a
    #    wide search that is itself worth failing on, because then HS is not a distinct arm at all.
    found = None
    for sd in range(999000, 999120):
        q2_, p2_, b2_, bi2_ = H.draw_market_seeded(cfg, sd)
        for frv in ([0.0] * 4, [0.05] * 4, [0.95] * 4, [1.0] * 4):
            vn = payoff_table(cfg, q2_, p2_, bi2_, rg, 0, [0.5] * 4)
            vo = payoff_table(cfg, q2_, p2_, bi2_, rg, 0, frv)
            if int(np.argmax(vo)) != int(np.argmax(vn)):
                found = (vn, vo); break
        if found:
            break
    chk("a market exists where the stale argmax differs (HS is a distinct arm)", found is not None)
    if found:
        vn, vo = found
        chk("HS marks the previous round's argmax, not this round's",
            table_block(vn, "HS", None, vo)[1] == int(np.argmax(vo))
            and table_block(vn, "HS", None, vo)[1] != int(np.argmax(vn)))

    # 7. An unparseable stream raises TechnicalFailure and never returns an action -- for EVERY LLM
    #    interface, not a sample of two. The whole point of the correction is that the failure policy
    #    no longer depends on which arm you are in, so the test has to be exhaustive over arms or it
    #    is testing something weaker than the claim.
    real = L.openrouter_call
    calls_made = {}
    try:
        L.openrouter_call = lambda *a_, **k_: ("not json at all", {})
        raised, leaked = [], []
        for itf in LLM_INTERFACES:
            try:
                decide("llm", "fake", cfg, itf, (0.5, .20), st, v, np.random.default_rng(0), 5, v)
                leaked.append(itf)
            except TechnicalFailure:
                raised.append(itf)
        chk(f"unparseable -> TechnicalFailure for all {len(LLM_INTERFACES)} LLM interfaces",
            len(raised) == len(LLM_INTERFACES), f"returned an action for {leaked}")
        # and the repair budget is spent identically in every arm (symmetry, not just non-fallback)
        for itf in LLM_INTERFACES:
            n = [0]
            L.openrouter_call = lambda *a_, _n=n, **k_: (_n.__setitem__(0, _n[0] + 1), ("x", {}))[1]
            try:
                decide("llm", "fake", cfg, itf, (0.5, .20), st, v, np.random.default_rng(0), 5, v)
            except TechnicalFailure:
                pass
            calls_made[itf] = n[0]
        chk("every interface spends the same number of parse attempts",
            len(set(calls_made.values())) == 1, str(calls_made))
        chk("that number is 1 + SCHEMA_REPAIR",
            set(calls_made.values()) == {1 + SCHEMA_REPAIR}, str(calls_made))
    finally:
        L.openrouter_call = real

    # 8. repeated calls do not mutate market state (the sec.3 invariant)
    t1 = payoff_table(cfg, qs, ps, bidx, rg, 0, [0.5] * 4)
    _ = decide("mock", "m", cfg, "H", (0.5, .20), st, t1, np.random.default_rng(1), 5)
    _ = decide("mock", "m", cfg, "H", (0.5, .20), st, t1, np.random.default_rng(1), 5)
    t2 = payoff_table(cfg, qs, ps, bidx, rg, 0, [0.5] * 4)
    q2, p2, b2, bi2 = H.draw_market_seeded(cfg, 999998)
    chk("payoff table unchanged after repeated decisions", bool(np.array_equal(t1, t2)))
    chk("market draw is a pure function of the seed", bool(np.array_equal(qs, q2) and np.array_equal(bs, b2)))

    # 9. proposed and executed regret are distinct fields, and an override moves only the executed one
    tr = new_trace("G", v, None); tr["proposed_index"] = d
    tr = finish(tr, v, a)
    chk("override leaves regret_proposed at the model's own value",
        abs(tr["regret_proposed"] - rel_regret(v, d)) < 1e-12 and tr["regret_executed"] == 0.0)

    # 10. M and O consume no API and O plays the solver's equilibrium index
    trm = decide("llm", "x", cfg, "M", (0.5, .20), st, v, np.random.default_rng(0), 5)
    tro = decide("llm", "x", cfg, "O", (0.5, .20), st, v, np.random.default_rng(0), 5)
    chk("M executes the argmax with 0 calls", trm["executed_index"] == a and trm["calls"] == 0)
    chk("O executes the equilibrium index with 0 calls",
        tro["executed_index"] == st["eq_index"] and tro["calls"] == 0)

    # 11. The Balance-8 guard arms. The load-bearing property is that GB and GS emit EXACTLY the
    #     private certificate G emits (executed displayed regret <= eta) while being free to choose a
    #     different member of that certificate. If that holds, then any platform-outcome difference
    #     between G and GB/GS is caused by the enforcement rule and by nothing else -- which is the
    #     constructive test of Theorem A7. Checked on many markets, not the one convenient v.
    ok_cert = ok_bal = ok_frozen = True
    sep = 0
    for sd in range(999200, 999260):
        q3, p3, b3, bi3 = H.draw_market_seeded(cfg, sd)
        vv = payoff_table(cfg, q3, p3, bi3, rg, 0, [0.5] * 4)
        cs = certificate_set(vv, ETA)
        bal = balanced_in_certificate(vv, ETA)
        ok_cert &= (rel_regret(vv, bal) <= ETA + 1e-12) and (int(np.argmax(vv)) in cs)
        ok_bal &= (bal == min(cs)) and (bal <= int(np.argmax(vv)))
        # R/G must be untouched by the refactor: their (eta, target) is exactly Balance-7's.
        ok_frozen &= (guard_of("R") == (ETA, None)) and (guard_of("G") == (ETA, "argmax"))
        sep += int(bal != int(np.argmax(vv)))
    chk("GB/GS override stays inside the 1% private certificate (same guarantee as G)", ok_cert)
    chk("GB/GS override is the most balanced member of that certificate", ok_bal)
    chk("R/G guard spec is bit-identical to Balance-7 (eta=0.01, override=argmax)", ok_frozen)
    chk("the certificate is wide enough for GS to differ from M in some markets (arm is distinct)",
        sep > 0, f"balanced==argmax in all 60 test markets")
    # eta=0 arms: the certificate collapses to the argmax alone, so G0 == M whenever it overrides.
    z = certificate_set(v, 0.0)
    chk("at eta=0 the certificate is exactly {argmax} (so G0's override target is forced)",
        z == [a], str(z))
    chk("R0/G0/GB are registered with the intended guard parameters",
        guard_of("R0") == (0.0, None) and guard_of("G0") == (0.0, "argmax")
        and guard_of("GB") == (ETA, "balanced"))

    # 14. RO is RECOMMENDATION-ONLY, not no-information. It drops the table and KEEPS the mark.
    # This was mis-described as "the no-information floor" in a draft pre-registration; the check
    # exists so the arm's actual semantics are asserted in code rather than in prose. See
    # BALANCE8_P4_PREREGISTRATION.md sec.2.1 amendment 2 and LIMITATIONS L1.4.
    ro_txt, ro_mark = table_block(v, "RO")
    u_txt, u_mark = table_block(v, "U")
    chk("RO removes the table (no 'profit=' rows) but KEEPS a mark on the true argmax",
        ("profit=" not in ro_txt) and ro_mark == int(np.argmax(v)))
    chk("U is the converse of RO: table present, no mark",
        ("profit=" in u_txt) and u_mark is None)

    # 14b. The check above protects THIS file, and that turned out not to be enough: the analysis
    # tool registered `RO` as "TABLE ONLY" and `HD` as "MARKED INDEX ONLY", both the exact inverse of
    # the truth (self-correction #18). The estimands never depended on the labels, but a report
    # written from them would have described the experiment backwards. So the analysis tool's arm
    # registration is now a machine-checkable table, and it is checked here against this runner.
    import balance8_hmech_contrast as HX
    chk("the analysis tool's arm registration agrees with this runner's table_block",
        HX.check_arm_semantics() == [], str(HX.check_arm_semantics()))
    # the labels that were actually wrong: HD "MARKED INDEX ONLY" (=> no table), RO "TABLE ONLY"
    # (=> table, and by that reading no mark). BOTH arms must be named in the complaint, or the gate
    # is decorative. Asserted on content rather than on a count, because the count is incidental:
    # "RO has a table and no mark" is two separate falsehoods about one arm.
    _inv = HX.check_arm_semantics((("HD", False, False), ("RO", True, None)))
    chk("...and that check is not vacuous: it rejects the inverted labels it was written for",
        any(m.startswith("HD:") for m in _inv) and any(m.startswith("RO:") for m in _inv),
        str(_inv))

    # 14c. LIMITATIONS L4.8 / self-correction #21: `R0` vs `R`, the load-bearing guard contrast,
    # confounds the certificate with the retry WORDING, because at eta=0 this file cannot honestly
    # say "tolerance of 1%" and instead appends an explicit restatement of the objective. That
    # limitation is only true while the asymmetry is actually here. If retry_message is later
    # "tidied" into one uniform sentence, the recorded confession silently becomes false and a
    # confounded contrast silently becomes clean -- so pin BOTH halves, on the emitted text itself
    # rather than on any description of it. Combined with the byte-identical first-turn check
    # further down, this is what confines the confound to the retry loop and makes round 1 the
    # only clean anchor.
    _r3 = rel_regret(v, 3)
    _m01, _m00 = retry_message(v, 3, _r3, ETA), retry_message(v, 3, _r3, 0.0)
    _clause = "must be the highest-profit index in the table"
    chk("retry_message at eta=0 carries the extra objective clause (L4.8's confound is real)",
        _clause in _m00, _m00)
    chk("...and at eta=0.01 it does NOT: R0 and R differ in WORDING, not only in eta",
        _clause not in _m01, _m01)
    # Everything except the tolerance clause must still be shared, or the confound is wider than
    # L4.8 admits and the limitation understates it.
    chk("the rest of the retry sentence is common to both arms (the confound is the clause, no more)",
        _m00.split("stated tolerance of")[0] == _m01.split("stated tolerance of")[0]
        and _m00.split("for maximizing")[1] == _m01.split("for maximizing")[1],
        f"{_m01!r} vs {_m00!r}")

    # ---- the mechanism verdict rule is FROZEN. These checks exist so the thresholds cannot drift
    # after the P4 numbers are visible, and so the "neither" branch cannot be quietly deleted.
    import balance8_analyze as AN
    chk("mechanism thresholds are the pre-registered ones (0.70 / 0.10 / 0.70)",
        (AN.C_COPY, AN.C_EXEC, AN.X_EXEC) == (0.70, 0.10, 0.70),
        f"{(AN.C_COPY, AN.C_EXEC, AN.X_EXEC)}")
    chk("verdict: high follow-rate reads as annotation copying",
        AN.mechanism_verdict(0.85, 0.10)[0] == "annotation_copying")
    chk("verdict: low follow AND high argmax reads as execution",
        AN.mechanism_verdict(0.05, 0.90)[0] == "execution")
    # The amendment this check enforces: ignoring the mark is NOT the same as maximising the table.
    # Without this branch, gemma's pilot profile (c=0.21, x=0.29, neither=0.50) trends toward being
    # scored as competence when it is closer to noise.
    chk("verdict: low follow WITHOUT high argmax reads as 'neither', NOT execution",
        AN.mechanism_verdict(0.05, 0.20)[0] == "neither")
    chk("verdict: mid-range follow-rate reads as mixed",
        AN.mechanism_verdict(0.50, 0.30)[0] == "mixed")
    chk("verdict: undefined when no decision has mark != argmax",
        AN.mechanism_verdict(float("nan"), float("nan"))[0] == "undefined")

    # ---- amendment 11: the SCOPE guard. `HS` produces a mark != argmax subset of 1 decision in
    # 8 000 (A9 again: the stale argmax is the current argmax), and without this the analyzer reports
    # "annotation_copying" with a [1.000, 1.000] interval computed from that one decision. The guard
    # gates which cells the frozen rule is applied to; it must not touch the rule itself.
    chk("amendment 11 thresholds are the declared ones (>= 5 seeds, >= 100 decisions)",
        (AN.MECH_MIN_SEEDS, AN.MECH_MIN_DECISIONS) == (5, 100),
        f"{(AN.MECH_MIN_SEEDS, AN.MECH_MIN_DECISIONS)}")
    chk("scope guard withholds the verdict on the degenerate HS subset (1 seed, 1 decision) that "
        "would otherwise be reported as annotation_copying with a [1.000,1.000] interval",
        AN.verdict_with_scope(1.0, 0.0, 1, 1)[0] == "undefined_insufficient_subset")
    chk("scope guard leaves a qualifying subset alone: it is a gate on WHICH cells the frozen rule "
        "sees, not a change to the rule",
        all(AN.verdict_with_scope(c, x, 8, 2560)[0] == AN.mechanism_verdict(c, x)[0]
            for c, x in ((0.85, 0.10), (0.05, 0.90), (0.05, 0.20), (0.50, 0.30))))
    chk("scope guard is direction-neutral: it suppresses an under-powered subset whichever way the "
        "verdict points (copying and execution are both withheld at n=1)",
        AN.verdict_with_scope(0.95, 0.0, 1, 1)[0]
        == AN.verdict_with_scope(0.0, 0.95, 1, 1)[0] == "undefined_insufficient_subset")
    chk("both arms of the guard bind independently (enough decisions but too few seeds, and the "
        "converse, are each withheld)",
        AN.verdict_with_scope(0.9, 0.0, 2, 5000)[0] == "undefined_insufficient_subset"
        and AN.verdict_with_scope(0.9, 0.0, 40, 20)[0] == "undefined_insufficient_subset")

    # 15b. The G0 falsification test must say WHERE a failure lands, because only one of the answers
    # implicates the guarded arms. A G0 that is internally consistent -- identical across models and
    # identical to the M arm drawn beside it -- but disagrees with the frozen prediction indicts the
    # PREDICTION FILE, not the override, and leaves G, GS and R0 untouched. The checker originally
    # printed the "puts G, GS and R0 in doubt" note unconditionally, which asserted the opposite of
    # the truth in exactly that case, so the classification is pinned here.
    import balance8_g0_verify as GV
    chk("G0 checker: a clean cell is not diagnosed as a fault",
        GV.classify_fault({"gemma": 0, "llama": 0}, 0, 0, True)[0] == "none")
    chk("G0 checker: in-cell inconsistency (P2 or P3 failing) indicts the OVERRIDE, which is the "
        "only reading that puts G, GS and R0 in doubt",
        GV.classify_fault({"gemma": 1, "llama": 0}, 1, 1, True)[0] == "override"
        and GV.classify_fault({"gemma": 0, "llama": 0}, 0, 1, True)[0] == "override")
    chk("G0 checker: an arm consistent across models AND with the in-cell M arm, yet disagreeing "
        "with the frozen file, indicts the PREDICTION FILE and exonerates the override",
        GV.classify_fault({"gemma": 1, "llama": 1}, 0, 0, True)[0] == "prediction_file")
    chk("G0 checker: with no M arm to appeal to, a prediction error and an override error are not "
        "distinguishable, and the checker says so instead of guessing",
        GV.classify_fault({"gemma": 1, "llama": 1}, 0, None, False)[0] == "indeterminate")

    # 16. The HD decoy must stay non-degenerate. HD is the pre-registered primary of a 128k-call
    # cell whose logic collapses if the decoy ever coincides with the argmax or lands inside the
    # eta band -- then "followed the mark" and "maximised the table" are the same event. The rule
    # is a nearest-neighbour search on a 21-point grid, so this is a property of the ENVIRONMENT,
    # not of the source, and it must be asserted rather than assumed. Full 1 920-table sweep and
    # the directional (G4) check live in balance8_decoy_check.py; this is the tripwire.
    dk_ok, dk_eta, dk_near = True, True, True
    for s_ in (999001, 999002, 999003, 999004, 999005):
        qs_, ps_, bs_, bidx_ = H.draw_market_seeded(cfg, s_)
        rg_ = rbar_grid_for(cfg, 0.5, 0.20)
        for j_ in range(cfg.m):
            vv = payoff_table(cfg, qs_, ps_, bidx_, rg_, j_, [0.5] * cfg.m)
            d_ = decoy_index(vv)
            dk_ok &= d_ != int(np.argmax(vv))
            dk_eta &= rel_regret(vv, d_) > ETA
            dk_near &= 0.02 <= rel_regret(vv, d_) <= 0.12
    chk("HD decoy is never the argmax (else the mechanism cell is vacuous)", dk_ok)
    chk("HD decoy regret is strictly outside the eta band (else following it is not an error)", dk_eta)
    chk("HD decoy regret is actually near 5%, so '~5% decoy' is a true description", dk_near)

    # 17. The structural-identity guard is pinned here so it cannot be deleted quietly. A paired
    # family reports diff=0.0000, CI [0,0], p=1 for TWO different reasons: the arms genuinely
    # behave alike (a finding) or the arms are the SAME FUNCTION on this environment (an
    # arithmetic identity, and no finding at all). The estimate, the interval and the p-value are
    # bit-identical in both cases, so nothing downstream can tell them apart -- only the executed
    # actions can. GS==M on 32 000/32 000 decisions under P_sep is the case that motivated it.
    _act = {
        ("m", "P", 1, 0, 0): {"A": 3, "B": 3},
        ("m", "P", 1, 0, 1): {"A": 7, "B": 7},
        ("m", "P", 2, 0, 0): {"A": 4, "B": 9},
        ("m", "Q", 1, 0, 0): {"A": 1, "B": 2},      # different policy: must not be counted
        ("m", "P", 3, 0, 0): {"A": 5},              # B absent: must not be counted
    }
    iar_mixed, n_mixed = AN.identical_action_rate(_act, "m", "P", "A", "B")
    iar_none, n_none = AN.identical_action_rate(_act, "m", "Q", "A", "B")
    chk("identical_action_rate counts only shared decisions in the requested cell",
        n_mixed == 3 and n_none == 1, f"n={n_mixed},{n_none}")
    chk("identical_action_rate measures agreement on EXECUTED actions, not on the outcome mean",
        abs(iar_mixed - 2 / 3) < 1e-12 and iar_none == 0.0, f"{iar_mixed},{iar_none}")
    _ident = {k: {"A": v["A"], "B": v["A"]} for k, v in _act.items() if "B" in v}
    chk("a family whose arms always execute the same action reaches the 0.999 flag threshold",
        AN.identical_action_rate(_ident, "m", "P", "A", "B")[0] >= 0.999)
    chk("identical_action_rate returns nan (not 0.0) when the two arms never co-occur",
        math.isnan(AN.identical_action_rate({}, "m", "P", "A", "B")[0]))

    # 18. GB is DROPPED from the launched guard cell (PREREGISTRATION amendment 9, gate G8). The
    # implementation stays -- it is correct, and check 11 still proves its private guarantee -- but
    # it must not be launched. GB and G share an eta AND a retry loop, and the override fires only
    # on verification failure, so the two arms can differ ONLY where the override fires: 0.599%
    # (gemma) / 0.047% (llama) of decisions, giving a measured GB!=G rate of 2/19200 and 4/19200.
    # These two checks pin the structural reason and the resulting decision, so that neither can be
    # undone without tripping a test. G6/G7 passed while this arm was dead, which is why the fact
    # rather than the earlier gate is what is asserted here.
    chk("GB and G share eta and a retry loop, so they can differ ONLY where the override fires "
        "(this is why gate G6 on override TARGETS did not establish the arm was alive)",
        GUARD_SPEC["GB"][0] == GUARD_SPEC["G"][0] and GUARD_SPEC["GB"][1] != GUARD_SPEC["G"][1])
    try:
        _gc = json.loads(Path("configs/balance8_guard.json").read_text(encoding="utf-8"))
        chk("GB is not in the launched guard cell (amendment 9: gate G8 failed, 2-4 of 19 200)",
            "GB" not in _gc["interfaces"], str(_gc["interfaces"]))
        chk("R0 and G0 are both launched (R0 vs R is the load-bearing comparison; G0 is retained "
            "for its enforcement BURDEN only -- amendment 10)",
            {"R0", "G0"} <= set(_gc["interfaces"]), str(_gc["interfaces"]))
    except FileNotFoundError:
        chk("configs/balance8_guard.json exists so amendment 9 can be enforced", False)

    # 19. G0 IS M (PREREGISTRATION amendment 10). At eta = 0, verification fails iff the proposal is
    # not the argmax, and the eta = 0 override target IS the argmax -- so G0's executed index is the
    # argmax whatever the model proposes. This is an identity over ALL models and ALL tables, so it
    # is asserted by quantifying over every proposal rather than by sampling behaviour. It is pinned
    # here because it is the reason `R0 vs G0` was withdrawn as load-bearing one amendment after
    # being promoted to it: the right-hand side is an arm that costs 25 600 API calls to reproduce
    # `M`, which costs none.
    _e0, _t0 = GUARD_SPEC["G0"]
    _rng_g0 = np.random.default_rng(11)
    _viol = _ovr = 0
    for _ in range(200):
        _v = _rng_g0.normal(size=NIDX + 1)
        _am = int(np.argmax(_v))
        for _i in range(NIDX + 1):                       # every proposal the model could make
            _j = _i
            if rel_regret(_v, _j) > _e0:
                _ovr += 1
                _j = override_index(_v, _e0, _t0)
            _viol += int(_j != _am)
    chk("G0 executes the displayed argmax for EVERY possible proposal, so G0 == M by construction "
        "and its F/GMV/T carry no information (amendment 10)",
        _viol == 0 and _ovr > 0, f"violations={_viol} overrides_exercised={_ovr}")
    chk("G0's eta is 0 and its override target is argmax -- the two facts that force the identity",
        _e0 == 0.0 and _t0 == "argmax", f"{GUARD_SPEC['G0']}")
    chk("R0 differs from G0 only by having no override target (same eta, same retry loop), so "
        "R0 vs G0 measures forced compliance and not the certificate width",
        GUARD_SPEC["R0"][0] == GUARD_SPEC["G0"][0] and GUARD_SPEC["R0"][1] is None)
    chk("the guard is a pure post-processor: build_prompt never receives it, so U/R/G/R0/G0 send "
        "byte-identical first turns (this licenses gate G12 and the G10/G11 bounds)",
        len({prompt_hash(k) for k in ("U", "R", "G", "R0", "G0")}) == 1)

    print("\nSELFTEST " + ("PASS" if not fails else f"FAIL ({len(fails)}): {fails}"))
    return 1 if fails else 0


# ---------------------------------------------------------------------------------------------
# batch driver
# ---------------------------------------------------------------------------------------------
def append(path, recs):
    with path.open("a", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")


def completed(path, need):
    if not path.exists():
        return set()
    c = {}
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                rr = json.loads(line); c[rr["run_key"]] = c.get(rr["run_key"], 0) + 1
            except Exception:
                pass
    return {k for k, n in c.items() if n >= need}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.config:
        print("need --config or --selftest")
        return 2

    C = json.loads(Path(a.config).read_text())
    agent = "mock" if a.mock else "llm"
    prefix = C["prefix"] + ("_mock" if a.mock else "")
    path = DATA / f"{prefix}_raw.jsonl"
    manifest = DATA / f"{prefix}_manifest.json"
    tflog = DATA / "balance8_technical_failures.jsonl"
    rounds, zeta = C["rounds"], C["zeta"]
    nw = a.workers if a.workers is not None else C.get("workers", 8)

    todo = []
    for alias, mid in C["models"].items():
        for itf in C["interfaces"]:
            for pn, pol in C["policies"].items():
                for se in C["seeds"]:
                    todo.append((alias, mid, itf, pn, tuple(pol), se))
    done = completed(path, rounds)
    todo = [x for x in todo if run_key(x[0], x[2], x[3], x[5]) not in done]
    order = np.random.default_rng(4242).permutation(len(todo))
    todo = [todo[i] for i in order]

    print(f"BALANCE-8 {prefix}: {len(todo)} runs to do ({len(done)} already complete), "
          f"workers={nw}, eta={ETA}, grid={NIDX+1} pts, rounds={rounds}", flush=True)
    lock = threading.Lock(); used = [0]; nrun = [0]; tech = [0]; t0 = time.time()

    def do(alias, mid, itf, pn, pol, se):
        return simulate_run(agent, mid, alias, itf, pn, pol, se, rounds, zeta,
                            parallel=(agent != "mock"))

    def handle(res):
        alias, itf, pn, se, recs, calls = res
        with lock:
            append(path, recs); used[0] += calls; nrun[0] += 1
            if nrun[0] % 10 == 0 or nrun[0] == 1:
                lo = max(1, max(x["round"] for x in recs) - 19)
                w = [x for x in recs if x["round"] >= lo]
                ov = np.mean([mm["override"] for x in recs for mm in x["merchants"]])
                print(f"[{nrun[0]}/{len(todo)}] {run_key(alias,itf,pn,se)}: "
                      f"GMV={np.mean([x['GMV'] for x in w]):.3f} F={np.mean([x['F'] for x in w]):.2f} "
                      f"ovr={ov:.2f} | calls {used[0]} | tech-fail {tech[0]} | "
                      f"{time.time()-t0:.0f}s", flush=True)
            if agent != "mock":
                json.dump(dict(prefix=prefix, config=C, eta=ETA, n_index=NIDX,
                               schema_repair=SCHEMA_REPAIR, max_retry=MAX_RETRY,
                               prompt_hashes={i: prompt_hash(i) for i in C["interfaces"]
                                              if i in LLM_INTERFACES},
                               progress=dict(done=len(done) + nrun[0], calls=used[0],
                                             technical_failures=tech[0], status="running")),
                          open(manifest, "w"), indent=2)

    def record_tech(x, e):
        with lock:
            tech[0] += 1
            with tflog.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(dict(when=time.strftime("%Y-%m-%dT%H:%M:%S"), prefix=prefix,
                                         alias=x[0], model=x[1], interface=x[2], policy=x[3],
                                         seed=x[5], error=str(e)[:400],
                                         kind=type(e).__name__)) + "\n")

    with ThreadPoolExecutor(max_workers=nw) as ex:
        futs = {ex.submit(do, *x): x for x in todo}
        for fut in as_completed(futs):
            x = futs[fut]
            try:
                handle(fut.result())
            except TechnicalFailure as e:
                record_tech(x, e)
                print(f"TECH-FAIL {x[0]}|{x[2]}|{x[3]}|s{x[5]}: {e}", flush=True)
            except L.APIError as e:
                record_tech(x, e)
                print(f"API-FAIL {x[0]}|{x[2]}: {e}", flush=True)

    status = "done" if nrun[0] == len(todo) else "partial"
    if agent != "mock":
        mm = json.load(open(manifest)) if manifest.exists() else {}
        mm.setdefault("progress", {}).update(dict(done=len(done) + nrun[0], calls=used[0],
                                                  technical_failures=tech[0], status=status))
        json.dump(mm, open(manifest, "w"), indent=2)
    print(f"BATCH {status.upper()}: {nrun[0]}/{len(todo)} runs, {used[0]} calls, "
          f"{tech[0]} technical failures, {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main() or 0)
