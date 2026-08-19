#!/usr/bin/env python3
"""recompute_dynamic_dp.py -- independent checks on the dynamic best-response solver.

    python offline_tests/recompute_dynamic_dp.py
    python offline_tests/recompute_dynamic_dp.py --only open_loop_certificate

`code/ha_dynamic_dp.py` answers the question §10 of the theory document asks: with rivals frozen at
the enumerated profile f*, does merchant i want to deviate once time-varying, state-dependent
strategies are allowed? It answers it with a Bellman recursion on 923,521 states, and it is fast only
because the joint kernel factorises and every sweep is a batched gemm against a 155 MB reward tensor.
That is precisely the kind of code that can be confidently, silently wrong.

Six checks, in increasing order of how much they would embarrass the result if they failed:

  sweep_matches_naive_dp     shrink the model until the joint kernel fits in memory as an explicit
                             matrix, build it by outer products, and require the optimised sweep to
                             agree to 1e-13. Tests the mode products, the flattening and the reward
                             tensor together.
  policy_value_forward       at full size, recompute V^{f*}(x_0) by pushing the state distribution
                             forward 80 rounds and summing expected rewards -- no recursion, no
                             maximisation, no 155 MB tensor.
  open_loop_certificate      at full size, exhibit explicit non-stationary strategies that beat the
                             constant action f*_i, by forward simulation. This is a constructive
                             lower bound on eps_dyn that uses no dynamic program at all: if it is
                             positive, "f* is not an equilibrium of Gamma_80" is proved independently
                             of every optimisation in ha_dynamic_dp.py. The scan runs over all 80
                             switch rounds, so it also reports HOW MUCH of the violation is end-game
                             unravelling rather than a substantive incentive.
  persistent_deviation       the same question with the horizon removed. In Gamma_delta there is no
                             final round, so no backward-induction artefact is available; a constant
                             deviation "play a forever" is evaluated from x_0 under delta = 0.95. A
                             positive gain here cannot be explained away as an end-game effect, and
                             it is the honest version of the claim.
  population_certificate     both of the above over the whole seed block, plus the patience threshold
                             delta*, the exact delta -> 1 criterion cross-validated against §9.5 by a
                             completely different route, the cost of the b-discretisation that f* is
                             enumerated under, and §2.4's 31-state reduction -- which brackets every
                             bound here from above and the full solver from below.
  gmv_consequence            the same objection priced in GMV: the four benchmark ratios recomputed
                             under each restriction in turn, and the profile merchants actually
                             settle on once constant strategies are scored exactly rather than by
                             the stationary plug-in.

No API key, no network, no cost.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
CODE = PKG / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import numpy as np                                                            # noqa: E402

import ha_model as M                                                          # noqa: E402
from ha_dynamics_audit import kernel                                          # noqa: E402
from ha_dynamic_dp import Case, NAMED, R_INIT, T_HORIZON                      # noqa: E402

DP = PKG / "results" / "solver" / "ha_dynamic_equilibrium.json"
BENCH = PKG / "results" / "solver" / "ha_benchmarks.json"
OUT = PKG / "results" / "validation" / "recompute_dynamic_dp.json"


class Lite:
    """Merchant i's problem with rivals frozen -- the small fields of `ha_dynamic_dp.Case`, only.

    `Case` allocates two (21, 31, 31, 31, 31) tensors, 310 MB, because the Bellman sweep streams them.
    Nothing below ever forms a joint state: every check works on marginals and on the exact law of the
    rivals' aggregate. Rebuilding the kernels without the tensors is what makes a best-response
    ITERATION affordable -- `t_gmv_consequence` walks a path through profile space, and at 310 MB and
    0.35 s per node it could not. `t_gmv_consequence` also asserts field-by-field that the two agree,
    so this is a cheaper constructor and not a second model.
    """

    def __init__(self, cfg: M.Config, mkt: M.Market, i: int, fprof, kappa: float, tau: float):
        self.cfg, self.mkt, self.i = cfg, mkt, i
        self.fstar = np.asarray(fprof, dtype=int)
        self.others = [k for k in range(mkt.m) if k != i]
        drate = np.arange(cfg.N_obs + 1) / cfg.N_obs
        pen_d = kappa * np.maximum(0.0, drate - tau)
        self.T_riv = [kernel(cfg, M.complaint_pmf(cfg, cfg.fgrid[self.fstar[k]], mkt.b[k]), pen_d)
                      for k in self.others]
        self.T_i = np.stack([kernel(cfg, M.complaint_pmf(cfg, f, mkt.b[i]), pen_d)
                             for f in cfg.fgrid])
        self.Ei = np.exp(cfg.alpha * (mkt.q[i] + (1 - mkt.q[i]) * cfg.fgrid[:, None])
                         + cfg.beta * cfg.rgrid[None, :] - cfg.gamma * mkt.p[i])
        fo = float(cfg.fgrid[self.fstar[self.others]].sum())
        self.Q = cfg.Q0 * np.exp(-cfg.lam * (cfg.fgrid + fo) / mkt.m)
        self.astar = int(self.fstar[i])


# ==================================================================================================
def _reward_table(case: Case, T: int) -> np.ndarray:
    """C[t, a, r] = merchant i's expected margin in round t if it plays a and its reputation is r.

    Merchant i's own action never enters the rivals' kernels, so the rivals' reputations evolve the
    same way whatever i does, and under any OPEN-LOOP strategy the four reputations stay independent:
    the joint law is a product of marginals at every round. Merchant i's payoff then needs only the
    exact law of the scalar Z = sum over rivals of exp(u_k), which is the convolution of three
    31-atom laws -- 29,791 atoms, summed directly rather than sampled.

    Building this table costs one pass; afterwards ANY open-loop sequence is worth a chain of 31-wide
    dot products, which is what makes the exhaustive scans below affordable. Nothing here recurses
    backwards and nothing here maximises -- it shares no code path with the Bellman sweep.
    """
    cfg, mkt, i = case.cfg, case.mkt, case.i
    R, Nf, e0 = cfg.R, cfg.Nf, math.exp(cfg.w0)
    r0 = int(round(R_INIT * (R - 1)))

    zgrid = np.zeros(1)
    for k in case.others:
        e = np.exp(cfg.alpha * (mkt.q[k] + (1 - mkt.q[k]) * cfg.fgrid[case.fstar[k]])
                   + cfg.beta * cfg.rgrid - cfg.gamma * mkt.p[k])
        zgrid = np.add.outer(zgrid, e).ravel()

    pr = []
    for _ in case.others:
        v = np.zeros(R)
        v[r0] = 1.0
        pr.append(v)
    W = np.empty((T, zgrid.size))
    for t in range(T):
        w = np.ones(1)
        for v in pr:
            w = np.outer(w, v).ravel()
        W[t] = w
        pr = [v @ case.T_riv[j] for j, v in enumerate(pr)]

    C = np.empty((T, Nf, R))
    coef = cfg.margin_frac * float(mkt.p[i])
    for a in range(Nf):
        ei = case.Ei[a][:, None]
        C[:, a, :] = (coef * case.Q[a]) * (W @ (ei / (e0 + ei + zgrid[None, :])).T)
    return C


def _value(case: Case, actions, C: np.ndarray, delta: float = 1.0) -> float:
    """Value at x_0 of an open-loop action sequence, read off the table."""
    R = case.cfg.R
    pi_ = np.zeros(R)
    pi_[int(round(R_INIT * (R - 1)))] = 1.0
    tot, d = 0.0, 1.0
    for t, a in enumerate(actions):
        a = int(a)
        tot += d * float(pi_ @ C[t, a])
        pi_ = pi_ @ case.T_i[a]
        d *= delta
    return tot


def _tail_table(case: Case, T0: int = 250):
    """(C, c_inf, residual): the transient reward table, plus its limit once the rivals have settled.

    C[t] is only needed while the rivals' law is still moving. Once W_t has converged to the rivals'
    stationary product law the per-round reward depends on t through nothing at all, so the tail of
    any constant-action value is the exact 31-state resolvent (I - delta T_i[a])^{-1} c_inf[a]. That
    removes the horizon from the calculation entirely: delta = 0.999 costs the same as delta = 0.9,
    and the only error left is how far W_{T0} still is from its limit, which is returned so the
    caller can check it rather than trust it.
    """
    cfg, mkt, i = case.cfg, case.mkt, case.i
    R, Nf, e0 = cfg.R, cfg.Nf, math.exp(cfg.w0)
    r0 = int(round(R_INIT * (R - 1)))

    zgrid = np.zeros(1)
    for k in case.others:
        e = np.exp(cfg.alpha * (mkt.q[k] + (1 - mkt.q[k]) * cfg.fgrid[case.fstar[k]])
                   + cfg.beta * cfg.rgrid - cfg.gamma * mkt.p[k])
        zgrid = np.add.outer(zgrid, e).ravel()

    pr = []
    for _ in case.others:
        v = np.zeros(R)
        v[r0] = 1.0
        pr.append(v)
    W = np.empty((T0 + 1, zgrid.size))
    for t in range(T0 + 1):
        w = np.ones(1)
        for v in pr:
            w = np.outer(w, v).ravel()
        W[t] = w
        pr = [v @ case.T_riv[j] for j, v in enumerate(pr)]
    resid = float(np.abs(W[T0] - W[T0 - 1]).sum())

    C = np.empty((T0, Nf, R))
    cinf = np.empty((Nf, R))
    coef = cfg.margin_frac * float(mkt.p[i])
    for a in range(Nf):
        ei = case.Ei[a][:, None]
        S = (ei / (e0 + ei + zgrid[None, :])).T                  # (Z, R)
        C[:, a, :] = (coef * case.Q[a]) * (W[:T0] @ S)
        cinf[a] = (coef * case.Q[a]) * (W[T0] @ S)
    return C, cinf, resid


def _settle(case, tol: float = 1e-13, cap: int = 400) -> int:
    """First round by which every rival marginal has stopped moving, at this kernel's noise floor.

    `_tail_table`'s default T_0 = 250 is a safe constant, but it is 2-3x more transient than these
    kernels need, and `t_gmv_consequence` pays that cost at every node of a best-response path. The
    marginals here fall to ~2e-14 by t ~ 100 and then stop improving -- 31 entries at one ulp each --
    so `tol` must sit above that floor or this never returns. The chosen T_0 is reported alongside the
    residual it actually achieved, so a bad choice is visible rather than silent.
    """
    r0, out = int(round(R_INIT * (case.cfg.R - 1))), 1
    for T in case.T_riv:
        v = np.zeros(case.cfg.R)
        v[r0] = 1.0
        t = cap
        for t in range(1, cap + 1):
            w = v @ T
            d = float(np.abs(w - v).sum())
            v = w
            if d < tol:
                break
        out = max(out, t)
    return out


def _gmv_tables(case, T0: int, actions):
    """(G[t, j, r], g_inf[j, r], residual): PLATFORM GMV, not merchant i's margin.

    `ha_dynamic_dp.Case.GMV` is Q * (sum_k p_k exp(u_k)) / den over ALL m merchants -- the object the
    benchmark file's G_FB, G_NP and G_SB are made of. Merchant i's action moves it through three
    channels at once: its own share p_i e_i / den, the traffic factor Q (which falls in the MEAN
    fabrication, so a deviation shrinks the pie for everyone), and its own reputation path. All three
    are exact here, because the rivals enter only through the joint law of the pair
    (z, A) = (sum exp(u_k), sum p_k exp(u_k)), which is enumerated over all 29,791 atoms.

    Only `actions` are tabulated: the check needs f*_i and one deviation, not all 21.
    """
    cfg, mkt, i = case.cfg, case.mkt, case.i
    R, e0 = cfg.R, math.exp(cfg.w0)
    r0 = int(round(R_INIT * (R - 1)))

    zgrid, agrid = np.zeros(1), np.zeros(1)
    for k in case.others:
        e = np.exp(cfg.alpha * (mkt.q[k] + (1 - mkt.q[k]) * cfg.fgrid[case.fstar[k]])
                   + cfg.beta * cfg.rgrid - cfg.gamma * mkt.p[k])
        zgrid = np.add.outer(zgrid, e).ravel()
        agrid = np.add.outer(agrid, float(mkt.p[k]) * e).ravel()

    pr = []
    for _ in case.others:
        v = np.zeros(R)
        v[r0] = 1.0
        pr.append(v)
    W = np.empty((T0 + 1, zgrid.size))
    for t in range(T0 + 1):
        w = np.ones(1)
        for v in pr:
            w = np.outer(w, v).ravel()
        W[t] = w
        pr = [v @ case.T_riv[j] for j, v in enumerate(pr)]
    resid = float(np.abs(W[T0] - W[T0 - 1]).sum())

    pi_ = float(mkt.p[i])
    G = np.empty((T0, len(actions), R))
    ginf = np.empty((len(actions), R))
    for j, a in enumerate(actions):
        ei = case.Ei[int(a)]
        den = e0 + zgrid[:, None] + ei[None, :]                  # (Z, R)
        tot = case.Q[int(a)] * ((W @ (agrid[:, None] / den)) + pi_ * ei[None, :] * (W @ (1.0 / den)))
        G[:, j, :] = tot[:T0]
        ginf[j] = tot[T0]
    return G, ginf, resid


def _stationary_law(case, a: int, n: int = 20000):
    """i's own reputation law under a constant action, iterated from the experiment's r_0 = 0.5."""
    R = case.cfg.R
    v = np.zeros(R)
    v[int(round(R_INIT * (R - 1)))] = 1.0
    resid = float("inf")
    for _ in range(n):
        w = v @ case.T_i[int(a)]
        w /= w.sum()
        resid = float(np.abs(w - v).sum())
        v = w
        if resid < 1e-15:
            break
    return v, resid


def _path_average(case, a: int, G, ginf, j: int, delta: float) -> float:
    """(1 - delta) sum_t delta^t E[GMV_t] from x_0 -- the per-round GMV the platform actually books.

    Normalised so it is directly comparable with the stationary GMV the benchmark file reports: a
    profile whose reputations start where they end has the two equal, and any difference is exactly
    what the transient is worth. The tail past T_0 is the exact resolvent, so no horizon is imposed.
    """
    R, T0 = case.cfg.R, G.shape[0]
    mu = np.zeros(R)
    mu[int(round(R_INIT * (R - 1)))] = 1.0
    tot, d = 0.0, 1.0
    for t in range(T0):
        tot += d * float(mu @ G[t, j])
        mu = mu @ case.T_i[int(a)]
        d *= delta
    tot += d * float(mu @ np.linalg.solve(np.eye(R) - delta * case.T_i[int(a)], ginf[j]))
    return (1.0 - delta) * tot


def _constant_action_values(case: Case, C, cinf, delta: float) -> np.ndarray:
    """Discounted value at x_0 of every constant action, transient summed and tail solved exactly."""
    R, Nf = case.cfg.R, case.cfg.Nf
    T0 = C.shape[0]
    I = np.eye(R)
    out = np.empty(Nf)
    for a in range(Nf):
        mu = np.zeros(R)
        mu[int(round(R_INIT * (R - 1)))] = 1.0
        tot, d = 0.0, 1.0
        for t in range(T0):
            tot += d * float(mu @ C[t, a])
            mu = mu @ case.T_i[a]
            d *= delta
        tot += d * float(mu @ np.linalg.solve(I - delta * case.T_i[a], cinf[a]))
        out[a] = tot
    return out


def _average_reward_values(case: Case, cinf, n: int = 20000, start: str = "r0"):
    """The exact delta -> 1 criterion: long-run average reward per round, for each constant action.

    Taking delta = 0.9999 and calling it "the patient limit" is not good enough here. The transient
    still carries weight 1 - delta^{T_0} ~ 2.5% of the value, and the incentive gaps being adjudicated
    are of order 0.01% of profit, so the discounted criterion cannot resolve them. The delta -> 1
    limit of the discounted average is the Cesaro limit pi(a).c_inf(a), which involves no delta at
    all -- so it is computed directly.

    `start` decides which recurrent class the average is taken over, and after 9.1 that is not a
    detail: pi is reached by iterating from the experiment's own initial condition r_0 = 0.5
    (start="r0") or from the uniform law that `_stationary_mean` uses (start="uniform"). Where the
    kernel has several recurrent classes the two disagree, and the disagreement is a finding.
    """
    R, Nf, T = case.cfg.R, case.cfg.Nf, case.T_i
    if start == "uniform":
        P = np.full((Nf, R), 1.0 / R)
    else:
        P = np.zeros((Nf, R))
        P[:, int(round(R_INIT * (R - 1)))] = 1.0
    # all 21 chains advanced together; stopped on the residual rather than a fixed budget, because
    # 9.1 showed the mixing time varies by eleven orders of magnitude across this kernel family
    resid = float("inf")
    for k in range(n):
        Q = np.einsum("ar,arc->ac", P, T)
        Q /= Q.sum(axis=1, keepdims=True)
        resid = float(np.abs(Q - P).sum(axis=1).max())
        P = Q
        if resid < 1e-15 and k > 2:
            break
    return np.einsum("ar,ar->a", P, cinf), resid


def _reduced_state_values(case, C, cinf, delta: float, horizon: int = None,
                          cap: int = 20000, tol: float = 1e-13) -> dict:
    """The exact optimum over strategies measurable in i's OWN reputation -- §2.4's 31-state MDP.

    §2.4 promises this as the cross-check whose gap to the full 31^4 solution measures what
    conditioning on rivals is worth, and it is the rung that turns §10's bounds into a sandwich:
    every open-loop deviation is (t, r_i)-measurable, so this is an upper bound on C_2; and every
    (t, r_i)-measurable strategy is (t, x)-measurable, so it is a lower bound on C_3. It is exact
    rather than heuristic because i's transition does not depend on x_{-i} and the rivals evolve
    exogenously: averaging their law into the stage reward loses nothing for a strategy that was
    never going to look at them.

    The reward is NOT time-homogeneous here -- that is the whole difference from the solver, which
    carries the rivals as state variables and therefore has a stationary reward on 923,521 states.
    So the transient is a genuine time-varying backward induction over C[t], and only the tail, once
    the rivals have settled, is a stationary 31-state MDP.
    """
    R, ast, T_i = case.cfg.R, case.astar, case.T_i
    n, resid = 0, 0.0
    if horizon is None:
        V, P, prev, resid = np.zeros(R), np.zeros(R), None, float("inf")
        for n in range(1, cap + 1):
            V = np.max(cinf + delta * (T_i @ V), axis=0)
            P = cinf[ast] + delta * (T_i[ast] @ P)
            e = V - P
            if prev is not None:
                resid = float(np.abs(e - prev).max())
                if resid < tol:
                    break
            prev = e
        rng = range(C.shape[0] - 1, -1, -1)
    else:
        V, P = np.zeros(R), np.zeros(R)
        rng = range(horizon - 1, -1, -1)
    for t in rng:
        V = np.max(C[t] + delta * (T_i @ V), axis=0)
        P = C[t, ast] + delta * (T_i[ast] @ P)
    eps, r0 = V - P, int(round(R_INIT * (R - 1)))
    return {"eps_at_r0": float(eps[r0]), "V_f_star_at_r0": float(P[r0]),
            "rel_at_r0": float(eps[r0] / P[r0]) if P[r0] > 0 else float("nan"),
            "eps_max_over_own_reputation": float(eps.max()),
            "rel_max_over_own_reputation": float((eps / np.maximum(P, 1e-15)).max()),
            "iterations": n, "residual": resid}


def _patience_threshold(case: Case, C, cinf, grid) -> dict:
    """Smallest delta on `grid` from which f*_i is best at that delta AND at every larger one.

    "Threshold" presumes optimality is monotone in patience, which is intuitive but not a theorem
    here: the transient advantage of a corner deviation and the stationary penalty it pays are both
    non-linear in delta. So the threshold is defined as the start of the maximal optimal SUFFIX of
    the grid -- which is correct whether or not monotonicity holds -- and `monotone_in_delta` records
    whether the naive definition would have given the same answer.
    """
    rows = []
    for dl in sorted(grid):
        v = _constant_action_values(case, C, cinf, dl)
        rows.append((dl, int(np.argmax(v)), float(v.max() / v[case.astar] - 1.0)))
    ok = [r[1] == case.astar for r in rows]
    star, j = None, len(rows)
    while j > 0 and ok[j - 1]:
        j -= 1
        star = rows[j][0]
    first_ok = next((r[0] for r, o in zip(rows, ok) if o), None)
    return {"delta_star": star,
            "monotone_in_delta": bool(star is None or star == first_ok),
            "always_optimal": all(ok),
            "never_optimal": not any(ok),
            "worst_relative_gain": max((r[2] for r in rows), default=0.0),
            "worst_at_delta": max(rows, key=lambda r: r[2])[0] if rows else None,
            "best_action_by_delta": {f"{r[0]:g}": r[1] for r in rows},
            "relative_gain_by_delta": {f"{r[0]:g}": r[2] for r in rows}}


def _one_shot_scan(case: Case, C: np.ndarray, T: int, delta: float = 1.0):
    """Value of every open-loop one-shot deviation: play f*_i always, except action a in round t.

    mu[t] is the law of i's own reputation at round t under f*_i; B[t][r] is the value from round t
    onward given reputation r and f*_i thereafter. Then switching to a at t is worth
        (prefix under f*) + mu[t].C[t,a] + delta * (mu[t] T_i[a]).B[t+1],
    so the whole (T, Nf) surface costs O(T*Nf) once mu and B are built. Returns (values, base).
    """
    R, Nf = case.cfg.R, case.cfg.Nf
    ast = case.astar
    mu = np.zeros((T, R))
    mu[0, int(round(R_INIT * (R - 1)))] = 1.0
    for t in range(T - 1):
        mu[t + 1] = mu[t] @ case.T_i[ast]
    B = np.zeros((T + 1, R))
    for t in range(T - 1, -1, -1):
        B[t] = C[t, ast] + delta * (case.T_i[ast] @ B[t + 1])
    pref = np.zeros(T + 1)                    # discounted reward accumulated strictly before t
    for t in range(T):
        pref[t + 1] = pref[t] + (delta ** t) * float(mu[t] @ C[t, ast])
    vals = np.empty((T, Nf))
    for t in range(T):
        dt = delta ** t
        nxt = mu[t] @ case.T_i                # (Nf, R): law at t+1 after each action at t
        vals[t] = pref[t] + dt * (mu[t] @ C[t].T + delta * (nxt @ B[t + 1]))
    return vals, float(pref[T])


def t_sweep_matches_naive_dp(_args) -> dict:
    """Shrink the model until the 4-merchant joint kernel is an explicit matrix, then compare."""
    cfg = replace(M.Config(), R=5, Nf=5, N_obs=6)
    mkt = M.draw_market(cfg, 70000)
    fstar = np.array([1, 2, 3, 2], dtype=int)
    kappa, tau = NAMED["P_SB_uniform"]
    c = Case(cfg, mkt, 0, fstar, kappa, tau)
    R, Nf, m = cfg.R, cfg.Nf, mkt.m
    n = R ** m

    # the naive joint kernel: one explicit (n, n) matrix per action of merchant 0
    K = np.zeros((Nf, n, n))
    idx = np.array(np.unravel_index(np.arange(n), (R,) * m)).T
    for a in range(Nf):
        rows = [c.T_i[a]] + list(c.T_riv)
        for s in range(n):
            row = np.ones(1)
            for d in range(m):
                row = np.outer(row, rows[d][idx[s, d]]).ravel()
            K[a, s] = row

    rng = np.random.default_rng(7)
    worst_sweep, worst_eval = 0.0, 0.0
    for _ in range(3):
        V = rng.normal(size=(R,) * m)
        Vf = V.ravel()
        # naive Bellman: max over actions of reward + K V, state by state
        naive = np.empty(n)
        for s in range(n):
            best = -np.inf
            for a in range(Nf):
                rwd = c.RWD[(a,) + tuple(idx[s])]
                best = max(best, rwd + float(K[a, s] @ Vf))
            naive[s] = best
        fast, _C = c._sweep(V, 1.0)
        worst_sweep = max(worst_sweep, float(np.abs(fast.ravel() - naive).max()))

        naive_e = np.empty(n)
        for s in range(n):
            naive_e[s] = c.RWD[(c.astar,) + tuple(idx[s])] + float(K[c.astar, s] @ Vf)
        worst_eval = max(worst_eval, float(np.abs(c._eval(V, 1.0).ravel() - naive_e).max()))

    return {"pass": worst_sweep < 1e-13 and worst_eval < 1e-13,
            "reduced_spec": {"R": R, "Nf": Nf, "N_obs": cfg.N_obs, "states": n},
            "max_abs_difference_bellman_max": worst_sweep,
            "max_abs_difference_policy_evaluation": worst_eval,
            "note": "the optimised sweep is a batched gemm against a flattened reward tensor; this "
                    "compares it with an explicitly constructed joint transition matrix"}


def t_policy_value_forward(args) -> dict:
    """V^{f*}(x_0) over 80 rounds, by forward simulation instead of backward recursion."""
    cfg = M.Config()
    kappa, tau = NAMED[args.policy]
    rg = M.rbar_grid(cfg, kappa, tau)
    mkt = M.draw_market(cfg, args.seed)
    eqs = M.pure_nash(cfg, mkt, rg[mkt.bidx, :])
    if len(eqs) != 1:
        return {"pass": None, "why": f"seed {args.seed} has {len(eqs)} pure equilibria"}
    fstar = np.array(eqs[0], dtype=int)
    c = Case(cfg, mkt, args.merchant, fstar, kappa, tau)

    C = _reward_table(c, T_HORIZON)
    fwd = _value(c, [c.astar] * T_HORIZON, C)

    P = np.zeros(c.shape)                     # the module's own backward policy evaluation
    for _ in range(T_HORIZON):
        P = c._eval(P, 1.0)
    back = float(P[c.x0])
    d = abs(fwd - back)
    return {"pass": bool(d < 1e-9), "seed": args.seed, "merchant": args.merchant,
            "policy": args.policy, "f_star": [int(v) for v in fstar],
            "V_f_star_at_x0_forward": fwd, "V_f_star_at_x0_backward": back, "delta": d,
            "note": "80 rounds of a product law pushed forward, against the module's recursion"}


def _equilibrium(args):
    cfg = M.Config()
    kappa, tau = NAMED[args.policy]
    rg = M.rbar_grid(cfg, kappa, tau)
    mkt = M.draw_market(cfg, args.seed)
    eqs = M.pure_nash(cfg, mkt, rg[mkt.bidx, :])
    if len(eqs) != 1:
        return cfg, mkt, kappa, tau, None
    return cfg, mkt, kappa, tau, np.array(eqs[0], dtype=int)


def t_open_loop_certificate(args) -> dict:
    """A constructive lower bound on eps_dyn that uses no dynamic program at all.

    Take the constant action f*_i and change it in exactly one round. If some such sequence is worth
    more than the constant one, merchant i has a strictly profitable deviation in Gamma_80 and the
    stationary profile is not an equilibrium of it -- whatever the Bellman code says. The bound is
    deliberately weak: it searches one round out of eighty, over 21 actions, by forward simulation.
    Any gap between it and ha_dynamic_dp.py's eps_dyn is what conditioning on the state and on the
    other 79 rounds is worth.

    Because the scan covers every switch round, it also answers the obvious objection. A deviation in
    round 80 is a backward-induction artefact: the last round has no reputational future, so of
    course fabrication pays there. `endgame_share` reports how much of the best deviation is bought
    in the last five rounds, and `best_gain_from_an_early_switch` reports what is left if the switch
    is confined to the first half of the game.

    Where the solver's own output is on disk, its reported on-path action sequence is evaluated the
    same way, which turns that sequence into a certificate as well.
    """
    cfg, mkt, kappa, tau, fstar = _equilibrium(args)
    if fstar is None:
        return {"pass": None, "why": f"seed {args.seed} has no unique pure equilibrium"}

    T, half, tail = T_HORIZON, T_HORIZON // 2, T_HORIZON - 5
    rows, cases, worst_scan = [], [], 0.0
    for i in range(mkt.m):
        c = Case(cfg, mkt, i, fstar, kappa, tau)
        cases.append(c)
        C = _reward_table(c, T)
        vals, base = _one_shot_scan(c, C, T)

        # the scan collapses O(T*Nf) simulations into prefix/suffix algebra; spot-check it against
        # the sequences it claims to summarise, because every number below rests on it
        for tt, aa in ((0, 20), (0, 3), (T // 2, 12), (T - 3, 20), (T - 1, 20), (T - 1, 0)):
            seq = [c.astar] * T
            seq[tt] = aa
            worst_scan = max(worst_scan, abs(_value(c, seq, C) - float(vals[tt, aa])))

        gain = vals - base
        t_b, a_b = np.unravel_index(int(np.argmax(gain)), gain.shape)
        early = gain[:half].max()
        rows.append({
            "merchant": i, "f_star": int(fstar[i]), "V_constant": base,
            "best_one_shot_gain": float(gain.max()), "best_switch_round": int(t_b) + 1,
            "best_switch_action": int(a_b),
            "relative": float(gain.max() / base),
            "best_gain_from_an_early_switch": float(early),
            "early_switch_round": int(np.argmax(gain[:half].max(axis=1))) + 1,
            "early_relative": float(early / base),
            "endgame_share_of_best": float(gain[tail:].max() / gain.max()) if gain.max() > 0 else None,
            "gain_by_round_max_over_actions": [float(v) for v in gain.max(axis=1)],
        })

    dp = json.loads(DP.read_text(encoding="utf-8")) if DP.exists() else None
    against_dp = None
    if dp:
        blk = (dp.get("policies") or {}).get(args.policy) or {}
        rec = next((r for r in blk.get("per_seed", []) if r.get("seed") == args.seed), None)
        if rec and not rec.get("skipped"):
            against_dp = []
            for i, mrec in enumerate(rec.get("merchants", [])):
                g8 = mrec.get("gamma_80")
                if not g8 or not g8.get("actions_at_x0_by_round"):
                    continue
                c = cases[i]
                v_seq = _value(c, g8["actions_at_x0_by_round"], _reward_table(c, T))
                ol = v_seq - rows[i]["V_constant"]
                against_dp.append({
                    "merchant": i, "dp_V_at_x0": g8["V_at_x0"], "dp_eps_at_x0": g8["eps_at_x0"],
                    "open_loop_value_of_dp_on_path_actions": v_seq, "open_loop_eps": ol,
                    "dp_eps_is_at_least_open_loop": bool(g8["eps_at_x0"] >= ol - 1e-9)})

    positive = [r for r in rows if r["best_one_shot_gain"] > 1e-12]
    ok = (bool(positive) and worst_scan < 1e-11
          and (against_dp is None or all(r["dp_eps_is_at_least_open_loop"] for r in against_dp)))
    return {"pass": ok, "seed": args.seed, "policy": args.policy,
            "f_star": [int(v) for v in fstar],
            "n_merchants_with_a_profitable_one_round_deviation": len(positive),
            "max_relative_lower_bound": max(r["relative"] for r in rows),
            "max_relative_from_an_early_switch": max(r["early_relative"] for r in rows),
            "scan_vs_direct_simulation_max_abs_error": worst_scan,
            "per_merchant": rows, "against_solver_output": against_dp,
            "note": "a strictly positive lower bound proves f* is not an equilibrium of Gamma_80 "
                    "without using the dynamic program; the solver's eps must be at least this large"}


def t_persistent_deviation(args) -> dict:
    """The same violation with the horizon removed, so it cannot be an end-game artefact.

    Gamma_delta announces no final round (gate G-H4), so backward induction from a terminal date is
    unavailable and the last-round unravelling that drives `open_loop_certificate` cannot occur. The
    deviation tested here is the crudest one that survives that: play a CONSTANT action a != f*_i
    forever, starting from r_0 = 0.5, discounted at delta. No state dependence, no timing, no
    dynamic program -- if this is profitable then f* is not a Nash equilibrium of Gamma_delta, and
    the reason is R2 alone (the stationary surrogate charges the deviator its long-run reputation
    immediately, while a merchant starting at r_0 = 0.5 spends its first rounds well above the
    reputation its deviation will eventually earn).

    The horizon is truncated at T rounds, which for delta = 0.95 leaves delta^T / (1 - delta) times
    the per-round margin unaccounted -- six orders of magnitude below the gaps reported here, and it
    biases both branches identically.
    """
    cfg, mkt, kappa, tau, fstar = _equilibrium(args)
    if fstar is None:
        return {"pass": None, "why": f"seed {args.seed} has no unique pure equilibrium"}
    T, delta = args.horizon, args.delta

    rows = []
    for i in range(mkt.m):
        c = Case(cfg, mkt, i, fstar, kappa, tau)
        C = _reward_table(c, T)
        vals = np.array([_value(c, [a] * T, C, delta) for a in range(cfg.Nf)])
        base = float(vals[c.astar])
        a_b = int(np.argmax(vals))
        rows.append({"merchant": i, "f_star": int(fstar[i]), "V_constant_f_star": base,
                     "best_constant_action": a_b, "V_best_constant": float(vals[a_b]),
                     "gain": float(vals[a_b] - base), "relative": float(vals[a_b] / base - 1.0),
                     "deviation_is_more_honest": bool(a_b < c.astar),
                     "runner_up_action": int(np.argsort(vals)[-2]),
                     "margin_over_runner_up": float(vals[a_b] - np.sort(vals)[-2]),
                     "value_by_constant_action": [float(v) for v in vals]})

    pos = [r for r in rows if r["gain"] > 1e-12]
    trunc = delta ** T / (1.0 - delta)
    return {"pass": True, "seed": args.seed, "policy": args.policy, "delta": delta, "rounds": T,
            "truncation_bound_per_unit_margin": trunc,
            "f_star": [int(v) for v in fstar],
            "n_merchants_with_a_profitable_constant_deviation": len(pos),
            "max_relative_gain": max(r["relative"] for r in rows),
            "verdict": ("f* is NOT a Nash equilibrium of Gamma_delta even in constant strategies"
                        if pos else
                        "no constant deviation pays; any Gamma_delta violation needs state "
                        "dependence or timing"),
            "per_merchant": rows,
            "note": "horizon-free: no final round exists, so this cannot be backward-induction "
                    "unravelling"}


def t_population_certificate(args) -> dict:
    """Both certificates over the whole seed population, so neither claim rests on one market.

    Cheap enough to sweep because `_reward_table` amortises: once the table is built, a constant
    deviation costs 300 dot products of length 31 and the full one-shot surface costs O(T*Nf).
    """
    lo, hi = (int(v) for v in args.seeds.split("-"))
    T = T_HORIZON
    grid = [float(v) for v in args.delta_grid.split(",")]
    per_seed, skipped, worst_resid = [], [], 0.0
    for seed in range(lo, hi + 1):
        a2 = argparse.Namespace(**{**vars(args), "seed": seed})
        cfg, mkt, kappa, tau, fstar = _equilibrium(a2)
        if fstar is None:
            skipped.append(seed)
            continue
        # the equilibrium f* is enumerated from rbar_grid()[bidx], i.e. with each merchant's base
        # complaint rate SNAPPED onto B_GRID; the runner (ha_model.sample_signal) and the dynamic
        # solver both use the exact draw mkt.b. Both reputation processes are therefore built, so the
        # discretisation can be measured instead of inherited.
        mkt_b = replace(mkt, b=M.B_GRID[mkt.bidx])
        rec = {"seed": seed, "f_star": [int(v) for v in fstar],
               "max_abs_b_discretisation": float(np.abs(mkt.b - mkt_b.b).max()), "merchants": []}
        for i in range(mkt.m):
            c = Case(cfg, mkt, i, fstar, kappa, tau)
            vals, base = _one_shot_scan(c, _reward_table(c, T), T)
            gain = vals - base
            C, cinf, resid = _tail_table(c, args.tail_t0)
            worst_resid = max(worst_resid, resid)
            pt = _patience_threshold(c, C, cinf, grid)
            g_ex, res_ex = _average_reward_values(c, cinf, args.avg_iters)

            # §2.4's 31-state reduction, both games. Three inequalities must hold by construction --
            # they are asserted rather than assumed, because each pair is computed by unrelated
            # arithmetic and a violation would mean one of the two routes is wrong.
            cv = _constant_action_values(c, C, cinf, args.delta)
            red80 = _reduced_state_values(c, C, cinf, 1.0, horizon=T)
            redd = _reduced_state_values(c, C, cinf, args.delta)
            sand = {
                "gamma_80_reduced_ge_one_shot":
                    float(red80["eps_at_r0"] - gain.max()),
                "gamma_delta_reduced_ge_best_constant":
                    float(redd["eps_at_r0"] - (cv.max() - cv[c.astar])),
                "policy_value_two_routes":
                    float(abs(redd["V_f_star_at_r0"] - cv[c.astar])),
            }

            cb = Case(cfg, mkt_b, i, fstar, kappa, tau)
            _, cinf_b, resid_b = _tail_table(cb, args.tail_t0)
            worst_resid = max(worst_resid, resid_b)
            g_bk, res_bk = _average_reward_values(cb, cinf_b, args.avg_iters)

            rec["merchants"].append({
                "merchant": i, "f_star": int(fstar[i]),
                "g80_best_relative": float(gain.max() / base),
                "g80_best_round": int(np.unravel_index(int(np.argmax(gain)), gain.shape)[0]) + 1,
                "g80_early_relative": float(gain[:T // 2].max() / base),
                "delta_star": pt["delta_star"],
                "monotone_in_delta": pt["monotone_in_delta"],
                "worst_constant_gain_relative": pt["worst_relative_gain"],
                "worst_at_delta": pt["worst_at_delta"],
                "best_action_by_delta": pt["best_action_by_delta"],
                # the exact delta -> 1 criterion, on both reputation processes
                "avg_gain_relative_exact_b": float(g_ex.max() / g_ex[c.astar] - 1.0),
                "avg_best_action_exact_b": int(np.argmax(g_ex)),
                "avg_gain_relative_bucketed_b": float(g_bk.max() / g_bk[cb.astar] - 1.0),
                "avg_best_action_bucketed_b": int(np.argmax(g_bk)),
                "stationary_residual": max(res_ex, res_bk),
                "reduced_state_gamma_80": red80,
                "reduced_state_gamma_delta": redd,
                "sandwich": sand,
            })
        per_seed.append(rec)

    allm = [m for r in per_seed for m in r["merchants"]]
    late = [m for m in allm if m["g80_best_round"] > T - 5]
    by_delta = {}
    for dl in grid:
        key = f"{dl:g}"
        bad = [m for m in allm if m["best_action_by_delta"][key] != m["f_star"]]
        corner = [m for m in bad if m["best_action_by_delta"][key] == cfg.Nf - 1]
        by_delta[key] = {"n_merchants_wanting_a_different_constant_action": len(bad),
                         "of_which_want_full_fabrication": len(corner),
                         "share": len(bad) / len(allm)}
    ds = [m["delta_star"] for m in allm if m["delta_star"] is not None]
    # cross-validation: as delta -> 1 the discounted criterion collapses onto the long-run average,
    # which is exactly the stationary expected payoff that section 9.5 tested by a completely
    # different route (power-iterated stationary laws contracted against a 31^4 payoff tensor). The
    # two must agree on WHICH markets fail, or one of them is wrong.
    cross = None
    aud = PKG / "results" / "solver" / "ha_mixing_audit.json"
    if aud.exists():
        blk = json.loads(aud.read_text(encoding="utf-8"))["part_B_payoff_restrictions"]
        pb = blk.get(args.policy, {})
        s95 = {r["seed"]: r["eps_expectation_rel_max"] for r in pb.get("per_seed", [])}
        mine = {r["seed"]: max(m["avg_gain_relative_bucketed_b"] for m in r["merchants"])
                for r in per_seed}
        both = sorted(set(s95) & set(mine))
        worst = max((abs(s95[s] - mine[s]) for s in both), default=0.0)
        cross = {"seeds_compared": len(both),
                 "applicable": bool(both),
                 "section_9_5_failures": sorted(s for s in both if s95[s] > 1e-9),
                 "my_bucketed_b_failures": sorted(s for s in both if mine[s] > 1e-9),
                 "agree_on_the_failing_set":
                     {s for s in both if s95[s] > 1e-9} == {s for s in both if mine[s] > 1e-9},
                 "max_abs_difference_in_relative_gap": worst,
                 "agree_numerically": bool(both) and bool(worst < 1e-9),
                 "note": "two unrelated routes to the same delta -> 1 limit: 9.5 power-iterates "
                         "stationary laws and contracts them against a 31^4 payoff tensor; this "
                         "convolves the rivals' aggregate exactly and takes a Cesaro limit. They "
                         "agree only when BOTH use the bucketed b that rbar_grid enumerates f* "
                         "with -- which is the point of the next block"}

    # how much the b-bucketing is worth, measured rather than inherited
    flip = [m for m in allm
            if (m["avg_gain_relative_exact_b"] > 1e-9) != (m["avg_gain_relative_bucketed_b"] > 1e-9)]
    bdisc = {
        "max_abs_b_discretisation": max(r["max_abs_b_discretisation"] for r in per_seed),
        "n_merchant_instances_whose_verdict_flips": len(flip),
        "n_exact_b_failures": sum(1 for m in allm if m["avg_gain_relative_exact_b"] > 1e-9),
        "n_bucketed_b_failures": sum(1 for m in allm if m["avg_gain_relative_bucketed_b"] > 1e-9),
        "n_seeds_with_an_exact_b_failure":
            sum(1 for r in per_seed
                if any(m["avg_gain_relative_exact_b"] > 1e-9 for m in r["merchants"])),
        "n_seeds_with_a_bucketed_b_failure":
            sum(1 for r in per_seed
                if any(m["avg_gain_relative_bucketed_b"] > 1e-9 for m in r["merchants"])),
        "max_exact_b_gain": max(m["avg_gain_relative_exact_b"] for m in allm),
        "max_bucketed_b_gain": max(m["avg_gain_relative_bucketed_b"] for m in allm),
        "max_abs_gain_difference": max(abs(m["avg_gain_relative_exact_b"]
                                           - m["avg_gain_relative_bucketed_b"]) for m in allm),
        "worst_stationary_residual": max(m["stationary_residual"] for m in allm),
        "note": "f* is enumerated from rbar_grid()[bidx], which snaps each merchant's base complaint "
                "rate onto B_GRID; ha_model.sample_signal and ha_dynamic_dp both use the exact draw. "
                "These are different games and this measures the difference",
    }
    # the cross-validation only bites on seeds section 9.5 also covers, so on a HELD-OUT block it
    # would pass vacuously. The self-checks that survive out of sample -- the rivals' law and the
    # merchant's own law having actually converged -- are therefore part of the verdict too.
    stat_ok = max(m["stationary_residual"] for m in allm) if allm else 1.0
    # the reduced-state rung: §2.4 promised it, and it brackets everything else
    sw = {
        "worst_gamma_80_reduced_minus_one_shot":
            min(m["sandwich"]["gamma_80_reduced_ge_one_shot"] for m in allm) if allm else -1.0,
        "worst_gamma_delta_reduced_minus_best_constant":
            min(m["sandwich"]["gamma_delta_reduced_ge_best_constant"] for m in allm)
            if allm else -1.0,
        "worst_policy_value_disagreement_between_two_routes":
            max(m["sandwich"]["policy_value_two_routes"] for m in allm) if allm else 1.0,
        "gamma_80_max_relative_eps": max(m["reduced_state_gamma_80"]["rel_at_r0"] for m in allm)
        if allm else None,
        "gamma_80_max_relative_eps_over_own_reputation":
            max(m["reduced_state_gamma_80"]["rel_max_over_own_reputation"] for m in allm)
            if allm else None,
        "gamma_delta_max_relative_eps": max(m["reduced_state_gamma_delta"]["rel_at_r0"]
                                            for m in allm) if allm else None,
        "gamma_delta_max_relative_eps_over_own_reputation":
            max(m["reduced_state_gamma_delta"]["rel_max_over_own_reputation"] for m in allm)
            if allm else None,
        "n_instances_with_a_reduced_state_gain_at_x0":
            sum(1 for m in allm if m["reduced_state_gamma_delta"]["eps_at_r0"] > 1e-12),
        "all_value_iterations_converged":
            bool(all(m["reduced_state_gamma_delta"]["residual"] < 1e-13 for m in allm)),
        "note": "eps here is the optimum over (t, r_i)-measurable strategies: an upper bound on "
                "every open-loop deviation above and a LOWER bound on the full 31^4 best response "
                "in ha_dynamic_equilibrium.json. The gap to the latter is what conditioning on "
                "rivals' reputations is worth, which is the quantity §2.4 promised",
    }
    sandwich_ok = bool(sw["worst_gamma_80_reduced_minus_one_shot"] > -1e-12
                       and sw["worst_gamma_delta_reduced_minus_best_constant"] > -1e-12
                       and sw["worst_policy_value_disagreement_between_two_routes"] < 1e-9
                       and sw["all_value_iterations_converged"])
    return {
        "pass": bool(allm and worst_resid < 1e-9 and stat_ok < 1e-9 and sandwich_ok
                     and (cross is None or not cross["applicable"] or cross["agree_numerically"])),
        "cross_validation_applicable": bool(cross and cross["applicable"]),
        "reduced_state_sandwich": sw,
        "seeds": args.seeds, "policy": args.policy, "delta_grid": grid,
        "n_seeds_with_unique_equilibrium": len(per_seed), "seeds_skipped": skipped,
        "n_merchant_instances": len(allm),
        "rival_law_convergence_residual_at_T0": worst_resid,
        "gamma_80": {
            "all_have_a_profitable_one_shot_deviation":
                bool(all(m["g80_best_relative"] > 1e-12 for m in allm)),
            "best_switch_round_is_in_the_last_five": f"{len(late)}/{len(allm)}",
            "max_relative_gain": max(m["g80_best_relative"] for m in allm),
            "max_relative_gain_from_a_first_half_switch":
                max(m["g80_early_relative"] for m in allm),
            "mean_relative_gain_from_a_first_half_switch":
                float(np.mean([m["g80_early_relative"] for m in allm])),
        },
        "gamma_delta_constant_strategies": {
            "by_delta": by_delta,
            "f_star_optimal_in_the_exact_patient_limit":
                f"{sum(1 for m in allm if m['avg_gain_relative_exact_b'] <= 1e-9)}/{len(allm)}",
            "delta_star_min": min(ds) if ds else None,
            "delta_star_max": max(ds) if ds else None,
            "delta_star_median": float(np.median(ds)) if ds else None,
            "n_never_optimal_on_grid": sum(1 for m in allm if m["delta_star"] is None),
            "n_non_monotone_in_delta": sum(1 for m in allm if not m["monotone_in_delta"]),
            "worst_relative_gain_anywhere_on_grid": max(m["worst_constant_gain_relative"]
                                                        for m in allm),
        },
        "cross_validation_against_section_9_5": cross,
        "b_discretisation_sensitivity": bdisc,
        "per_seed": per_seed,
        "note": "the Gamma_80 violation is universal but concentrated in the final rounds; the "
                "Gamma_delta sweep is horizon-free and restricted to constant strategies, which is "
                "exactly the class the stationary surrogate optimises over, so any failure there is "
                "R2 (transient vs stationary evaluation) and nothing else",
    }


def _c1_node(cfg, mkt, kappa, tau, f, i, cache, t0cap):
    """Cached (Lite, reward table, tail, T_0) for merchant i facing rivals f_{-i}."""
    key = (i, tuple(int(v) for v in np.delete(np.asarray(f), i)))
    if key not in cache:
        lc = Lite(cfg, mkt, i, f, kappa, tau)
        T0 = min(t0cap, _settle(lc) + 5)
        C, cinf, resid = _tail_table(lc, T0)
        cache[key] = (lc, C, cinf, T0, resid)
    return cache[key]


def _c1_equilibrium(cfg, mkt, kappa, tau, f0, delta, cache, t0cap, max_iter=30):
    """Nash equilibrium of the game whose strategies are CONSTANT actions and whose payoff is the
    exact discounted value from x_0 -- the class C_1 profile, the honest dynamic analogue of f*.

    f* is the fixed point of the same best-response map scored by the STATIONARY plug-in instead. So
    wherever this lands somewhere else, the distance is R2 and R3 priced in units the platform cares
    about, and the GMV there is what the marketplace would actually book. Simultaneous best response
    can cycle; that is reported rather than smoothed away, because a cycle is itself a finding about
    the surrogate (it means no constant-strategy profile survives exact discounted evaluation).
    """
    f = np.asarray(f0, dtype=int).copy()
    seen, path, worst = {tuple(f)}, [f.tolist()], 0.0
    for it in range(1, max_iter + 1):
        nxt = f.copy()
        for i in range(mkt.m):
            lc, C, cinf, _, resid = _c1_node(cfg, mkt, kappa, tau, f, i, cache, t0cap)
            worst = max(worst, resid)
            nxt[i] = int(np.argmax(_constant_action_values(lc, C, cinf, delta)))
        if np.array_equal(nxt, f):
            return f, "fixed_point", it, path, worst
        f = nxt
        if tuple(f) in seen:
            return f, "cycle", it, path, worst
        seen.add(tuple(f))
        path.append(f.tolist())
    return f, "iteration_cap", max_iter, path, worst


def t_gmv_consequence(args) -> dict:
    """What the dynamic objection is worth in GMV -- the only units the platform question is asked in.

    Sections 9 and 10 establish that f* can fail as a dynamic best response. That is a statement about
    incentives; it becomes a statement about the marketplace only once it is priced. Four GMV numbers
    are computed for the same markets, differing ONLY in which restriction is lifted:

      plug-in stationary   the benchmark file's own number at this policy: GMV(r-bar) at f*.  [C_0]
      exact stationary     E[GMV(r)] at f* under the exact stationary law -- R3 (Jensen) removed.
      discounted average   (1-delta) sum delta^t E[GMV_t] from r_0 = 0.5 at f* -- R2 removed too.
      C_1 equilibrium      the same, at the profile that is a Nash equilibrium under that criterion
                           rather than under the plug-in -- R1 partially removed (constant strategies,
                           exact evaluation), which is the profile merchants would actually settle on.

    Each is divided by the same per-seed first best G_FB, so the headline "the second best reaches
    70.9% of first best" can be restated under each restriction and the reader can see which
    restriction the number depends on. Two exactness guards run first: `Lite` must reproduce `Case`
    field by field, and `_gmv_tables` must reproduce `ha_dynamic_dp.Case.GMV` contracted against the
    rivals' law -- otherwise every number below is measuring the wrong game.
    """
    lo, hi = (int(v) for v in args.seeds.split("-"))
    delta = float(args.delta)
    deltas = sorted({float(v) for v in args.gmv_deltas.split(",")} | {delta})
    cfg = M.Config()
    kappa, tau = NAMED[args.policy]
    rg = M.rbar_grid(cfg, kappa, tau)

    # ---- guard 1 and 2: Lite == Case, and my GMV table == Case.GMV contracted --------------------
    mkt0 = M.draw_market(cfg, lo)
    f0 = np.array(M.pure_nash(cfg, mkt0, rg[mkt0.bidx, :])[0], dtype=int)
    ref, lit = Case(cfg, mkt0, 0, f0, kappa, tau), Lite(cfg, mkt0, 0, f0, kappa, tau)
    gerr = max(float(np.abs(np.asarray(getattr(ref, k)) - np.asarray(getattr(lit, k))).max())
               for k in ("T_i", "Ei", "Q"))
    gerr = max(gerr, max(float(np.abs(a - b).max()) for a, b in zip(ref.T_riv, lit.T_riv)))
    acts0 = sorted({int(f0[0]), 0, cfg.Nf - 1})
    Gt, _, _ = _gmv_tables(lit, 12, acts0)
    pr = [np.zeros(cfg.R) for _ in lit.others]
    for v in pr:
        v[int(round(R_INIT * (cfg.R - 1)))] = 1.0
    gmv_err = 0.0
    for t in range(12):
        for j, a in enumerate(acts0):
            ex = np.einsum("rabc,a,b,c->r", ref.GMV[a], pr[0], pr[1], pr[2])
            gmv_err = max(gmv_err, float(np.abs(ex - Gt[t, j]).max()))
        pr = [v @ lit.T_riv[k] for k, v in enumerate(pr)]
    del ref

    bench = json.loads(BENCH.read_text(encoding="utf-8")) if BENCH.exists() else {"per_seed": {}}
    per_seed, skipped, worst_resid, worst_stat, prof_err = [], [], 0.0, 0.0, 0.0
    for seed in range(lo, hi + 1):
        mkt = M.draw_market(cfg, seed)
        eqs = M.pure_nash(cfg, mkt, rg[mkt.bidx, :])
        bs = bench["per_seed"].get(str(seed))
        if len(eqs) != 1 or bs is None:
            skipped.append(seed)
            continue
        fstar = np.array(eqs[0], dtype=int)
        row = next((r for r in bs["policy_rows"]
                    if abs(r["kappa"] - kappa) < 1e-12 and abs(r["tau"] - tau) < 1e-12), None)
        if row is None:
            skipped.append(seed)
            continue
        cache = {}

        def gmv_at(prof, delta_=delta):
            """(stationary E[GMV], discounted average GMV) at an arbitrary constant profile.

            Computed from merchant 0's decomposition; `profile_consistency` below recomputes it from
            merchant 1's and requires the two to agree, which they can only do if the (z, A) law and
            the traffic factor are right, since the two decompositions share no arithmetic.
            """
            out = []
            for i in (0, 1):
                lc, _, _, T0, rs = _c1_node(cfg, mkt, kappa, tau, prof, i, cache, args.tail_t0)
                a = int(prof[i])
                G, gi, rr = _gmv_tables(lc, T0, [a])
                pi_, sres = _stationary_law(lc, a)
                out.append((float(pi_ @ gi[0]), _path_average(lc, a, G, gi, 0, delta_), rs, rr, sres))
            return out

        (st0, dy0, r0a, r0b, s0), (st1, dy1, _, _, s1) = gmv_at(fstar)
        prof_err = max(prof_err, abs(st0 - st1), abs(dy0 - dy1))
        worst_resid = max(worst_resid, r0a, r0b)
        worst_stat = max(worst_stat, s0, s1)

        # the C_1 equilibrium at several patience levels, sharing one cache: which profile merchants
        # settle on is exactly the delta question section 10.2 asks, and the GMV it implies is the
        # answer in platform units
        by_delta = {}
        for dl in deltas:
            fd, sd, itd, pathd, wrd = _c1_equilibrium(cfg, mkt, kappa, tau, fstar, dl,
                                                      cache, args.tail_t0)
            worst_resid = max(worst_resid, wrd)
            (std, dyd, _, _, sdd), _ = gmv_at(fd, dl)
            worst_stat = max(worst_stat, sdd)
            by_delta[f"{dl:g}"] = {
                "profile": [int(v) for v in fd], "status": sd, "iterations": itd,
                "equals_f_star": bool(np.array_equal(fd, fstar)),
                "mean_f": float(cfg.fgrid[fd].mean()),
                "gmv_exact_stationary": std, "gmv_discounted_average": dyd,
                "ratio_to_FB": dyd / float(bs["G_FB"]),
                "path_length": len(pathd),
            }
        main = by_delta[f"{delta:g}"]
        fc1, status, iters = np.array(main["profile"], dtype=int), main["status"], main["iterations"]
        stc, dyc = main["gmv_exact_stationary"], main["gmv_discounted_average"]

        # unilateral deviation: one merchant moves, the other three hold f* -- a strictly weaker
        # (and always available) counterfactual than the C_1 equilibrium, so it brackets the damage
        uni = []
        for i in range(mkt.m):
            lc, C, cinf, T0, _ = _c1_node(cfg, mkt, kappa, tau, fstar, i, cache, args.tail_t0)
            ad = int(np.argmax(_constant_action_values(lc, C, cinf, delta)))
            G, gi, _ = _gmv_tables(lc, T0, [ad])
            uni.append({"merchant": i, "f_star": int(fstar[i]), "best_constant_action": ad,
                        "gmv_discounted_average": _path_average(lc, ad, G, gi, 0, delta)})

        fb = float(bs["G_FB"])
        per_seed.append({
            "seed": seed, "f_star": [int(v) for v in fstar], "G_FB": fb,
            "gmv_plugin_stationary": float(row["gmv_best_eq"]),
            "benchmark_eq_matches": [int(v) for v in row["eq_best"]] == [int(v) for v in fstar],
            "gmv_exact_stationary": st0,
            "gmv_discounted_average": dy0,
            "c1_profile": [int(v) for v in fc1], "c1_status": status, "c1_iterations": iters,
            "c1_equals_f_star": bool(np.array_equal(fc1, fstar)),
            "c1_by_delta": by_delta,
            "gmv_c1_exact_stationary": stc,
            "gmv_c1_discounted_average": dyc,
            "mean_f_star": float(cfg.fgrid[fstar].mean()),
            "mean_f_c1": float(cfg.fgrid[fc1].mean()),
            "worst_unilateral_deviation_gmv": min(u["gmv_discounted_average"] for u in uni),
            "unilateral": uni,
        })

    if not per_seed:
        return {"pass": None, "seeds": args.seeds,
                "why": "no seed in this block has both a unique pure equilibrium and a first-best "
                       "entry in results/solver/ha_benchmarks.json; the GMV ratios are defined "
                       "against that file's G_FB and are not recomputed here",
                "seeds_skipped": skipped}

    def agg(key):
        v = [r[key] for r in per_seed]
        return {"mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v)),
                "mean_ratio_to_FB": float(np.mean([r[key] / r["G_FB"] for r in per_seed]))}

    n_moved = sum(1 for r in per_seed if not r["c1_equals_f_star"])
    return {
        "pass": bool(gerr < 1e-15 and gmv_err < 1e-12 and prof_err < 1e-12
                     and all(r["benchmark_eq_matches"] for r in per_seed)),
        "seeds": args.seeds, "policy": args.policy, "kappa": kappa, "tau": tau, "delta": delta,
        "deltas": deltas, "n_seeds": len(per_seed), "seeds_skipped": skipped,
        "guards": {
            "lite_vs_case_max_abs_field_error": gerr,
            "gmv_table_vs_case_GMV_max_abs_error": gmv_err,
            "profile_gmv_from_two_decompositions_max_abs_error": prof_err,
            "rival_law_residual_at_T0": worst_resid,
            "own_stationary_residual": worst_stat,
            "benchmark_equilibrium_agrees_on_every_seed":
                bool(all(r["benchmark_eq_matches"] for r in per_seed)),
        },
        "gmv": {
            "plugin_stationary_at_f_star": agg("gmv_plugin_stationary"),
            "exact_stationary_at_f_star": agg("gmv_exact_stationary"),
            "discounted_average_at_f_star": agg("gmv_discounted_average"),
            "discounted_average_at_c1_equilibrium": agg("gmv_c1_discounted_average"),
            "exact_stationary_at_c1_equilibrium": agg("gmv_c1_exact_stationary"),
            "first_best": {"mean": float(np.mean([r["G_FB"] for r in per_seed]))},
        },
        "c1": {
            "n_seeds_where_the_c1_equilibrium_differs_from_f_star": n_moved,
            "share": n_moved / max(1, len(per_seed)),
            "status_counts": {s: sum(1 for r in per_seed if r["c1_status"] == s)
                              for s in sorted({r["c1_status"] for r in per_seed})},
            "mean_f_star": float(np.mean([r["mean_f_star"] for r in per_seed])),
            "mean_f_c1": float(np.mean([r["mean_f_c1"] for r in per_seed])),
            "n_seeds_at_the_fabrication_corner":
                sum(1 for r in per_seed if r["mean_f_c1"] > 0.999),
            "worst_gmv_loss_relative": max((r["gmv_discounted_average"]
                                            - r["gmv_c1_discounted_average"])
                                           / r["gmv_discounted_average"] for r in per_seed),
            "mean_gmv_loss_relative": float(np.mean(
                [(r["gmv_discounted_average"] - r["gmv_c1_discounted_average"])
                 / r["gmv_discounted_average"] for r in per_seed])),
        },
        "c1_by_delta": {
            f"{dl:g}": {
                "n_seeds_where_the_c1_equilibrium_differs_from_f_star":
                    sum(1 for r in per_seed if not r["c1_by_delta"][f"{dl:g}"]["equals_f_star"]),
                "n_seeds_at_the_fabrication_corner":
                    sum(1 for r in per_seed if r["c1_by_delta"][f"{dl:g}"]["mean_f"] > 0.999),
                "status_counts": {s: sum(1 for r in per_seed
                                         if r["c1_by_delta"][f"{dl:g}"]["status"] == s)
                                  for s in sorted({r["c1_by_delta"][f"{dl:g}"]["status"]
                                                   for r in per_seed})},
                "mean_f": float(np.mean([r["c1_by_delta"][f"{dl:g}"]["mean_f"] for r in per_seed])),
                "mean_gmv": float(np.mean([r["c1_by_delta"][f"{dl:g}"]["gmv_discounted_average"]
                                           for r in per_seed])),
                "mean_ratio_to_FB": float(np.mean([r["c1_by_delta"][f"{dl:g}"]["ratio_to_FB"]
                                                   for r in per_seed])),
                "worst_ratio_to_FB": float(np.min([r["c1_by_delta"][f"{dl:g}"]["ratio_to_FB"]
                                                   for r in per_seed])),
            } for dl in deltas},
        "per_seed": per_seed,
        "note": "the ratios to first best are the paper's headline restated under each restriction; "
                "the plug-in row must reproduce the benchmark file exactly, because it IS the "
                "benchmark file, and it is carried only so the other three rows have a baseline "
                "computed by the same code path they are",
    }


# ==================================================================================================
TESTS = [
    ("sweep_matches_naive_dp", t_sweep_matches_naive_dp),
    ("policy_value_forward", t_policy_value_forward),
    ("open_loop_certificate", t_open_loop_certificate),
    ("persistent_deviation", t_persistent_deviation),
    ("population_certificate", t_population_certificate),
    ("gmv_consequence", t_gmv_consequence),
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=70000)
    ap.add_argument("--seeds", default="70000-70059")
    ap.add_argument("--merchant", type=int, default=0)
    ap.add_argument("--policy", default="P_SB_uniform")
    ap.add_argument("--delta", type=float, default=0.95)
    ap.add_argument("--horizon", type=int, default=300)
    ap.add_argument("--tail-t0", type=int, default=250)
    ap.add_argument("--avg-iters", type=int, default=20000)
    ap.add_argument("--delta-grid",
                    default="0.8,0.85,0.9,0.925,0.95,0.96,0.97,0.98,0.99,0.995,0.999,0.9999")
    ap.add_argument("--gmv-deltas", default="0.9,0.95,0.99")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)

    print("=" * 96)
    print("RECOMPUTE  --  the dynamic best-response solver, by independent routes")
    print("=" * 96)
    results, t0 = [], time.time()
    for name, fn in TESTS:
        if a.only and name not in a.only:
            continue
        t1 = time.time()
        try:
            r = fn(a)
        except Exception as exc:                                              # noqa: BLE001
            import traceback
            r = {"pass": False, "error": f"{type(exc).__name__}: {exc}",
                 "traceback": traceback.format_exc()[-1200:]}
        r["check"] = name
        r["seconds"] = round(time.time() - t1, 2)
        results.append(r)
        tag = "PASS" if r.get("pass") else ("SKIP" if r.get("pass") is None else "FAIL")
        print(f"[{tag}] {name}  ({r['seconds']}s)")
        if r.get("pass") is False:
            print("      " + json.dumps({k: v for k, v in r.items()
                                         if k not in ("check", "pass", "seconds", "traceback")},
                                        default=str)[:1200])
            if r.get("traceback"):
                print("      " + r["traceback"].replace("\n", "\n      ")[-700:])

    ran = [r for r in results if r.get("pass") is not None]
    n_pass = sum(1 for r in ran if r["pass"])
    payload = {"schema_version": "ha-recompute-dynamic-dp-1",
               "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "n_checks": len(ran), "n_pass": n_pass, "all_pass": n_pass == len(ran),
               "seconds": round(time.time() - t0, 1), "results": results}
    outp = Path(a.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    tmp = outp.with_suffix(outp.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(outp)
    print("=" * 96)
    print(f"  {n_pass}/{len(ran)} passed in {payload['seconds']}s   ->  {outp}")
    print("=" * 96)
    return 0 if payload["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
