"""ha_dynamics_audit.py -- does the stationary surrogate describe the dynamic model?

The frozen corpus evaluates every benchmark inside the restricted stationary game G(theta,kappa,tau)
of theory/HIDDEN_ACTION_THEORY.md section 2.3. That game is obtained from the dynamic game by three
independent restrictions:

    R1  strategies are constant in t and in the state          -- tested in ha_dynamic_dp.py
    R2  payoffs are read off the STATIONARY reputation         -- tested here (part B2)
    R3  utility uses rbar = E[r] rather than E[u(r)]           -- tested here (part B1)

This module tests R2 and R3, and first establishes the precondition both of them need: that the
reputation chain a fixed action induces actually HAS a unique stationary distribution and actually
reaches it. `ha_model._stationary_mean` power-iterates from the uniform distribution for 300 steps
and returns `pi @ rgrid` without ever checking that the limit is unique or that 300 steps is enough.
If some kernel has two recurrent classes, that function silently returns a number that depends on the
initial uniform guess and on nothing else -- it would still be a float, it would still be
deterministic, and every downstream benchmark would inherit it. Part A checks it.

Nothing here presumes the answer. A kernel that fails ergodicity is reported, not dropped; a Jensen
error that is large is reported as a bias in the headline benchmarks.

Output: results/solver/ha_mixing_audit.json.  No API, no network, pure numpy.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from math import gcd
from pathlib import Path

import numpy as np

import ha_model as M

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
OUT = PKG / "results" / "solver" / "ha_mixing_audit.json"

SEEDS = list(range(70000, 70060))
# the three policies every downstream claim is made under
NAMED = {"P_GMV": (0.5, 0.20), "P_robust": (4.0, 0.30), "P_SB_uniform": (2.0, 0.30)}
R_INIT = 0.5                     # ha_runner.R_INIT; on the 31-point grid this is exactly index 15
T_HORIZON = 80                   # ha_runner.DEFAULT_ROUNDS
TV_CHECKPOINTS = (20, 40, 80, 300)


# ==================================================================================================
# part A -- is the reputation chain ergodic, and does 300 iterations reach its limit?
# ==================================================================================================
def kernel(cfg: M.Config, pmf_d: np.ndarray, pen_d: np.ndarray) -> np.ndarray:
    """The 31x31 row-stochastic kernel that `ha_model._stationary_mean` builds and discards.

    Identical construction, including the `np.round` snap onto the grid, which is inherited from
    `equilibrium._stationary_mean` and deliberately unchanged.
    """
    R, rgrid = cfg.R, cfg.rgrid
    rp = np.clip(rgrid[:, None] + cfg.eta_r * (1.0 - rgrid[:, None]) - pen_d[None, :], 0.0, 1.0)
    idx = np.clip(np.round(rp * (R - 1)).astype(int), 0, R - 1)
    T = np.zeros((R, R))
    rows = np.repeat(np.arange(R), pen_d.size)
    np.add.at(T, (rows, idx.ravel()), np.tile(pmf_d, R))
    return T


def _reach(T: np.ndarray) -> np.ndarray:
    """Boolean reachability closure of the support graph."""
    A = T > 0
    reach = A | np.eye(len(T), dtype=bool)
    for _ in range(int(np.ceil(np.log2(max(2, len(T)))) + 1)):
        reach = reach | (reach @ reach)
    return reach


def classes(T: np.ndarray):
    """Communicating classes, and which of them are closed (= recurrent, the chain being finite)."""
    reach = _reach(T)
    comm = reach & reach.T
    n, seen, out = len(T), np.zeros(len(T), bool), []
    for i in range(n):
        if seen[i]:
            continue
        members = np.nonzero(comm[i])[0]
        seen[members] = True
        closed = bool(np.all(np.isin(np.nonzero(T[members].sum(axis=0) > 0)[0], members)))
        out.append((members, closed))
    return out


def period(T: np.ndarray, members: np.ndarray) -> int:
    """Period of a communicating class: gcd over edges of (dist[i] + 1 - dist[j])."""
    pos = {int(v): k for k, v in enumerate(members)}
    dist = {int(members[0]): 0}
    stack, d = [int(members[0])], 0
    while stack:
        i = stack.pop()
        for j in np.nonzero(T[i] > 0)[0]:
            j = int(j)
            if j not in pos:
                continue
            if j not in dist:
                dist[j] = dist[i] + 1
                stack.append(j)
            d = gcd(d, abs(dist[i] + 1 - dist[j]))
    return d if d > 0 else 1


def audit_kernel(T: np.ndarray, rgrid: np.ndarray) -> dict:
    """Every property `_stationary_mean` assumes without checking."""
    R = len(T)
    cls = classes(T)
    rec = [m for m, closed in cls if closed]
    n_rec = len(rec)

    # the stationary vector the model actually uses: 300 power iterations from uniform
    pi_model = np.full(R, 1.0 / R)
    for _ in range(300):
        pi_model = pi_model @ T

    # the one the EXPERIMENT would reach: every merchant starts at r = 0.5, not at uniform
    r0 = int(round(R_INIT * (R - 1)))
    pi_r0 = np.zeros(R)
    pi_r0[r0] = 1.0
    for _ in range(300):
        pi_r0 = pi_r0 @ T

    # an independent one: left eigenvector for eigenvalue 1
    w, v = np.linalg.eig(T.T)
    k = int(np.argmin(np.abs(w - 1.0)))
    pi_eig = np.real(v[:, k])
    pi_eig = pi_eig / pi_eig.sum() if abs(pi_eig.sum()) > 1e-12 else pi_eig
    pi_gap = float(np.abs(pi_model - pi_eig).sum()) if n_rec == 1 else float("nan")

    ev = np.sort(np.abs(np.linalg.eigvals(T)))[::-1]
    slem = float(ev[1]) if R > 1 else 0.0
    # a SLEM within 1e-9 of 1 makes the eigenvector cross-check meaningless: eigenvalue 1 is then
    # numerically degenerate and `pi_eig` is an arbitrary vector from a near-2-dimensional eigenspace
    degenerate = bool(slem > 1.0 - 1e-9)
    if degenerate:
        pi_gap = float("nan")

    # Dobrushin ergodic coefficient: 1 - min over row pairs of the overlap mass
    ov = np.minimum(T[:, None, :], T[None, :, :]).sum(axis=2)
    dobrushin = float(1.0 - ov.min())

    tv = {}
    for n in TV_CHECKPOINTS:
        Tn = np.linalg.matrix_power(T, n)
        tv[str(n)] = float(0.5 * np.abs(Tn - pi_model[None, :]).sum(axis=1).max())

    per = [period(T, m) for m in rec]
    rb_u, rb_0 = float(pi_model @ rgrid), float(pi_r0 @ rgrid)
    return dict(n_recurrent=n_rec, unique_pi=bool(n_rec == 1),
                irreducible=bool(len(cls) == 1 and cls[0][1]),
                aperiodic=bool(all(p == 1 for p in per)) if per else False,
                periods=[int(p) for p in per],
                transient_states=int(R - sum(len(m) for m in rec)),
                slem=slem, slem_degenerate=degenerate, dobrushin=dobrushin,
                pi_l1_gap_vs_eig=pi_gap,
                rbar_model=rb_u, rbar_from_r_init=rb_0, rbar_init_gap=rb_u - rb_0,
                converged_in_300=bool(tv["300"] <= 1e-9),
                pi_l1_gap_uniform_vs_r_init=float(np.abs(pi_model - pi_r0).sum()), tv=tv)


def _nanmax(vals: list) -> float:
    """max ignoring the nans that a non-ergodic kernel legitimately produces; nan if all are nan."""
    v = [x for x in vals if not math.isnan(x)]
    return float(max(v)) if v else float("nan")


def part_a(cfg: M.Config, verbose: bool) -> dict:
    t0 = time.time()
    drate = np.arange(cfg.N_obs + 1) / cfg.N_obs
    # kappa = 0 makes tau irrelevant: 1 + 7*8 = 57 distinct penalty vectors
    pols = sorted({(k, t if k > 0 else 0.0) for (k, t) in M.POLICY_CLASS})
    pmfs = {(bi, fi): M.complaint_pmf(cfg, f, b)
            for bi, b in enumerate(M.B_GRID) for fi, f in enumerate(cfg.fgrid)}

    per_policy, offenders, all_rows = [], [], []
    for (kap, tau) in pols:
        pen_d = kap * np.maximum(0.0, drate - tau)
        rows = []
        for (bi, fi), pmf in pmfs.items():
            a = audit_kernel(kernel(cfg, pmf, pen_d), cfg.rgrid)
            a.update(kappa=kap, tau=tau, b_index=bi, f_index=fi)
            rows.append(a)
            if not a["unique_pi"] or not a["aperiodic"] or not a["converged_in_300"]:
                offenders.append(dict(a, tv=a["tv"]))
        all_rows += rows
        per_policy.append(dict(
            kappa=kap, tau=tau, n_kernels=len(rows),
            n_unique_pi=int(sum(r["unique_pi"] for r in rows)),
            n_irreducible=int(sum(r["irreducible"] for r in rows)),
            n_aperiodic=int(sum(r["aperiodic"] for r in rows)),
            n_converged_in_300=int(sum(r["converged_in_300"] for r in rows)),
            max_transient_states=int(max(r["transient_states"] for r in rows)),
            max_slem=float(max(r["slem"] for r in rows)),
            max_dobrushin=float(max(r["dobrushin"] for r in rows)),
            max_pi_l1_gap_vs_eig=_nanmax([r["pi_l1_gap_vs_eig"] for r in rows]),
            max_abs_rbar_init_gap=float(max(abs(r["rbar_init_gap"]) for r in rows)),
            max_tv={str(n): float(max(r["tv"][str(n)] for r in rows)) for n in TV_CHECKPOINTS}))
        if verbose:
            p = per_policy[-1]
            print(f"  kappa={kap:<5} tau={tau:<5} unique_pi {p['n_unique_pi']}/{p['n_kernels']}  "
                  f"SLEM<={p['max_slem']:.4f}  TV80<={p['max_tv']['80']:.2e}  "
                  f"[{time.time() - t0:.0f}s]")

    n = len(all_rows)
    # relaxation time from the SLEM: rounds to contract a TV unit by e
    slems = np.array([r["slem"] for r in all_rows])
    ok = slems < 1.0 - 1e-12
    trelax = np.where(ok, 1.0 / np.maximum(1e-12, -np.log(np.maximum(slems, 1e-300))), np.inf)
    gaps = np.array([abs(r["rbar_init_gap"]) for r in all_rows])
    worst = all_rows[int(np.argmax(gaps))]
    # the population that matters for the benchmarks: everything the platform can actually deploy
    live = [r for r in all_rows if r["kappa"] > 0]
    return dict(
        n_kernels=n, n_distinct_penalty_vectors=len(pols),
        n_b=len(M.B_GRID), n_f=cfg.Nf,
        summary=dict(
            unique_pi=int(sum(r["unique_pi"] for r in all_rows)),
            irreducible=int(sum(r["irreducible"] for r in all_rows)),
            aperiodic=int(sum(r["aperiodic"] for r in all_rows)),
            converged_in_300=int(sum(r["converged_in_300"] for r in all_rows)),
            converged_in_300_kappa_positive=int(sum(r["converged_in_300"] for r in live)),
            n_kappa_positive=len(live),
            n_with_transient_states=int(sum(r["transient_states"] > 0 for r in all_rows)),
            max_transient_states=int(max(r["transient_states"] for r in all_rows)),
            max_slem=float(slems.max()), median_slem=float(np.median(slems)),
            n_slem_degenerate=int(sum(r["slem_degenerate"] for r in all_rows)),
            max_relaxation_time_rounds_nondegenerate=float(np.max(trelax[ok])) if ok.any() else float("inf"),
            max_dobrushin=float(max(r["dobrushin"] for r in all_rows)),
            max_pi_l1_gap_vs_eig=_nanmax([r["pi_l1_gap_vs_eig"] for r in all_rows]),
            max_abs_rbar_init_gap=float(gaps.max()),
            max_abs_rbar_init_gap_kappa_positive=float(max(abs(r["rbar_init_gap"]) for r in live)),
            worst_rbar_init_gap_at=dict(kappa=worst["kappa"], tau=worst["tau"],
                                        b_index=worst["b_index"], f_index=worst["f_index"],
                                        rbar_uniform_start=worst["rbar_model"],
                                        rbar_r_init_start=worst["rbar_from_r_init"]),
            max_tv={str(k): float(max(r["tv"][str(k)] for r in all_rows)) for k in TV_CHECKPOINTS},
            worst_tv_at_horizon_80=float(max(r["tv"]["80"] for r in all_rows))),
        offenders=offenders[:50], n_offenders=len(offenders),
        per_policy=per_policy, seconds=round(time.time() - t0, 1))


# ==================================================================================================
# part B -- the plug-in payoff against the honest expectation, and 80 rounds against infinity
# ==================================================================================================
def pi_tables(cfg: M.Config, kappa: float, tau: float):
    """pi[b_index, f_index, :] and the kernels, for one policy."""
    drate = np.arange(cfg.N_obs + 1) / cfg.N_obs
    pen_d = kappa * np.maximum(0.0, drate - tau)
    pis = np.zeros((len(M.B_GRID), cfg.Nf, cfg.R))
    Ts = np.zeros((len(M.B_GRID), cfg.Nf, cfg.R, cfg.R))
    for bi, b in enumerate(M.B_GRID):
        for fi, f in enumerate(cfg.fgrid):
            T = kernel(cfg, M.complaint_pmf(cfg, f, b), pen_d)
            Ts[bi, fi] = T
            p = np.full(cfg.R, 1.0 / cfg.R)
            for _ in range(300):
                p = p @ T
            pis[bi, fi] = p
    return pis, Ts


def _expu(cfg: M.Config, mkt: M.Market, fidx: np.ndarray) -> list:
    """exp(u_k(r_k)) as a 31-vector per merchant, at the given action profile."""
    f = cfg.fgrid[np.asarray(fidx, dtype=int)]
    return [np.exp(cfg.alpha * (mkt.q[k] + (1 - mkt.q[k]) * f[k]) + cfg.beta * cfg.rgrid
                   - cfg.gamma * mkt.p[k]) for k in range(mkt.m)]


def _tensors(cfg: M.Config, mkt: M.Market, fidx: np.ndarray):
    """GMV and per-merchant profit as functions of the JOINT reputation state (31^m tensor)."""
    m, R = mkt.m, cfg.R
    E = _expu(cfg, mkt, fidx)

    def ax(k):
        sh = [1] * m
        sh[k] = R
        return tuple(sh)

    den = np.full((R,) * m, math.exp(cfg.w0))
    for k in range(m):
        den = den + E[k].reshape(ax(k))
    Q = cfg.Q0 * math.exp(-cfg.lam * float(cfg.fgrid[np.asarray(fidx, dtype=int)].mean()))
    shares = [E[k].reshape(ax(k)) / den for k in range(m)]
    gmv = Q * sum(mkt.p[k] * shares[k] for k in range(m))
    profit = [cfg.margin_frac * mkt.p[k] * Q * shares[k] for k in range(m)]
    return gmv, profit


def _outer(vs: list) -> np.ndarray:
    w = vs[0]
    for v in vs[1:]:
        w = np.multiply.outer(w, v)
    return w


def _mix_weights(paths: list) -> np.ndarray:
    """(1/T) sum_t (x)_j p_{j,t}, built as one BLAS matmul instead of T outer products.

    paths[j] is (T, R). The time-averaged joint law is a MIXTURE of T product measures, not a product
    measure, so it has to be formed explicitly -- but (a (x) b) (x) (c (x) d) reshapes to an outer
    product of two (T, R^2) matrices, so the whole mixture is one T-contraction.
    """
    Tn, R = paths[0].shape
    A = np.einsum("ti,tj->tij", paths[0], paths[1]).reshape(Tn, R * R)
    B = np.einsum("ti,tj->tij", paths[2], paths[3]).reshape(Tn, R * R)
    return (A.T @ B).reshape(R, R, R, R) / Tn


def part_b(cfg: M.Config, seeds: list, verbose: bool) -> dict:
    t0 = time.time()
    out = {}
    r0 = int(round(R_INIT * (cfg.R - 1)))
    for pname, (kap, tau) in NAMED.items():
        pis, Ts = pi_tables(cfg, kap, tau)
        rg = M.rbar_grid(cfg, kap, tau)
        rows = []
        for seed in seeds:
            mkt = M.draw_market(cfg, seed)
            rbar_m = rg[mkt.bidx, :]
            eqs = M.pure_nash(cfg, mkt, rbar_m)
            if len(eqs) != 1:
                rows.append(dict(seed=seed, n_pure_nash=len(eqs), skipped=True))
                continue
            fidx = np.array(eqs[0], dtype=int)
            pi_j = [pis[mkt.bidx[k], fidx[k]] for k in range(mkt.m)]

            gmv_t, profit_t = _tensors(cfg, mkt, fidx)
            W_stat = _outer(pi_j)

            # --- R3: plug-in GMV(rbar) against E[GMV(r)] under the joint stationary law -----------
            plug = M.profile_outcome(cfg, mkt, rbar_m, fidx)
            g_plug = float(plug["GMV"])
            g_stat = float((gmv_t * W_stat).sum())

            # independence is a THEOREM here (the kernel factorises across merchants), so what is
            # measured is the residual of that factorisation, not an assumption
            marg = [W_stat.sum(axis=tuple(a for a in range(mkt.m) if a != k)) for k in range(mkt.m)]
            indep_resid = float(np.abs(W_stat - _outer(marg)).sum())

            # --- R2: the transient path from r0 = 0.5 over the experiment's own horizon ----------
            paths = []
            for k in range(mkt.m):
                p = np.zeros(cfg.R)
                p[r0] = 1.0
                seq = np.zeros((T_HORIZON, cfg.R))
                Tk = Ts[mkt.bidx[k], fidx[k]]
                for t in range(T_HORIZON):
                    seq[t] = p
                    p = p @ Tk
                paths.append(seq)
            W_path = _mix_weights(paths)
            g_path = float((gmv_t * W_path).sum())
            g_by_t = [float((gmv_t * _outer([paths[k][t] for k in range(mkt.m)])).sum())
                      for t in (0, 4, 9, 19, 39, 79)]

            # --- does the plug-in NE survive being scored with expectations? ---------------------
            # exploitability of f* under E[profit], with the deviator's OWN stationary law moved too
            eps_exp, eps_plug = np.zeros(mkt.m), M.exploitability(cfg, mkt, rbar_m, fidx)
            base_exp = np.zeros(mkt.m)
            for k in range(mkt.m):
                base_exp[k] = float((profit_t[k] * W_stat).sum())
                best = base_exp[k]
                for d in range(cfg.Nf):
                    if d == fidx[k]:
                        continue
                    c = fidx.copy()
                    c[k] = d
                    _, pr_d = _tensors(cfg, mkt, c)
                    pv = list(pi_j)
                    pv[k] = pis[mkt.bidx[k], d]
                    best = max(best, float((pr_d[k] * _outer(pv)).sum()))
                eps_exp[k] = best - base_exp[k]

            rows.append(dict(
                seed=seed, f_star=[int(v) for v in fidx],
                GMV_plugin_rbar=g_plug, GMV_stationary_expectation=g_stat,
                GMV_path_mean_80=g_path, GMV_by_round=g_by_t,
                jensen_abs=g_plug - g_stat, jensen_rel=(g_plug - g_stat) / g_stat,
                transient_abs=g_path - g_stat, transient_rel=(g_path - g_stat) / g_stat,
                independence_residual_l1=indep_resid,
                eps_plugin_max=float(eps_plug.max()),
                eps_expectation_max=float(eps_exp.max()),
                eps_expectation_rel_max=float((eps_exp / np.maximum(base_exp, 1e-15)).max()),
                still_nash_under_expectation=bool(eps_exp.max() <= 1e-12)))
        good = [r for r in rows if not r.get("skipped")]
        ja = np.array([r["jensen_rel"] for r in good])
        ta = np.array([r["transient_rel"] for r in good])
        out[pname] = dict(
            policy=[kap, tau], n_seeds=len(good), n_skipped=len(rows) - len(good),
            GMV_plugin_mean=float(np.mean([r["GMV_plugin_rbar"] for r in good])),
            GMV_stationary_mean=float(np.mean([r["GMV_stationary_expectation"] for r in good])),
            GMV_path80_mean=float(np.mean([r["GMV_path_mean_80"] for r in good])),
            jensen_rel_mean=float(ja.mean()), jensen_rel_absmax=float(np.abs(ja).max()),
            jensen_sign_positive=int((ja > 0).sum()), jensen_sign_negative=int((ja < 0).sum()),
            transient_rel_mean=float(ta.mean()), transient_rel_absmax=float(np.abs(ta).max()),
            independence_residual_max=float(max(r["independence_residual_l1"] for r in good)),
            n_still_nash_under_expectation=int(sum(r["still_nash_under_expectation"] for r in good)),
            eps_expectation_rel_max=float(max(r["eps_expectation_rel_max"] for r in good)),
            per_seed=rows)
        if verbose:
            o = out[pname]
            print(f"  {pname:<13} plug-in {o['GMV_plugin_mean']:.4f}  stationary "
                  f"{o['GMV_stationary_mean']:.4f}  path80 {o['GMV_path80_mean']:.4f}  "
                  f"Jensen {o['jensen_rel_mean']:+.2%}  NE survives "
                  f"{o['n_still_nash_under_expectation']}/{o['n_seeds']}  [{time.time() - t0:.0f}s]")
    return out


# ==================================================================================================
# part C -- do the headline benchmarks move when the solver's initial guess is replaced by the
#           initial condition the experiment actually uses?
# ==================================================================================================
def rbar_grid_from_r_init(cfg: M.Config, kappa: float, tau: float) -> np.ndarray:
    """`ha_model.rbar_grid`, but the power iteration starts at delta_{r=0.5} instead of at uniform.

    For an ergodic kernel the two agree and this is a no-op. For the 2,733 kernels part A finds that
    have not converged after 300 iterations -- and the 315 that have three recurrent classes and so
    never will -- the difference is exactly the arbitrariness in the published rbar table.
    """
    drate = np.arange(cfg.N_obs + 1) / cfg.N_obs
    pen_d = kappa * np.maximum(0.0, drate - tau)
    r0 = int(round(R_INIT * (cfg.R - 1)))
    out = np.zeros((len(M.B_GRID), cfg.Nf))
    for bi, b in enumerate(M.B_GRID):
        for fi, f in enumerate(cfg.fgrid):
            T = kernel(cfg, M.complaint_pmf(cfg, f, b), pen_d)
            p = np.zeros(cfg.R)
            p[r0] = 1.0
            for _ in range(300):
                p = p @ T
            out[bi, fi] = p @ cfg.rgrid
    return out


def _bench(cfg: M.Config, seeds: list, table) -> dict:
    """G_FB, G_NP and the uniform second best under a given rbar table constructor."""
    cache = {}

    def get(k, t):
        key = (k, t if k > 0 else 0.0)
        if key not in cache:
            cache[key] = table(cfg, *key)
        return cache[key]

    pols = list(M.POLICY_CLASS)
    fb, npv = np.zeros(len(seeds)), np.zeros(len(seeds))
    gw = np.zeros((len(seeds), len(pols)))
    for si, seed in enumerate(seeds):
        mkt = M.draw_market(cfg, seed)
        rm0 = get(0.0, 0.0)[mkt.bidx, :]
        en0 = M.enumerate_profiles(cfg, mkt, rm0)
        g_fb = float(en0["GMV"].max())
        ne0 = M.pure_nash(cfg, mkt, rm0, en0)
        npv[si] = max(float(en0["GMV"][e]) for e in ne0) if ne0 else np.nan
        for pi_, (k, t) in enumerate(pols):
            rm = get(k, t)[mkt.bidx, :]
            en = M.enumerate_profiles(cfg, mkt, rm)
            g_fb = max(g_fb, float(en["GMV"].max()))
            ne = M.pure_nash(cfg, mkt, rm, en)
            gw[si, pi_] = min(float(en["GMV"][e]) for e in ne) if ne else np.nan
        fb[si] = g_fb
    col = gw.mean(axis=0)
    i = int(np.argmax(col))
    return dict(G_FB=float(fb.mean()), G_NP=float(np.nanmean(npv)),
                G_SB_uniform=float(col[i]), SB_policy=[float(v) for v in pols[i]],
                ratio_SB_over_FB=float(col[i] / fb.mean()),
                implementation_power=float((col[i] - np.nanmean(npv)) / (fb.mean() - np.nanmean(npv))))


def part_c(cfg: M.Config, seeds: list, verbose: bool) -> dict:
    t0 = time.time()
    if verbose:
        print("  published table (power iteration from the uniform distribution) ...")
    pub = _bench(cfg, seeds, lambda c, k, t: M.rbar_grid(c, k, t))
    if verbose:
        print(f"    G_FB {pub['G_FB']:.6f}  G_NP {pub['G_NP']:.6f}  "
              f"G_SB_unif {pub['G_SB_uniform']:.6f}  [{time.time() - t0:.0f}s]")
        print("  same thing started at the experiment's own r_init = 0.5 ...")
    alt = _bench(cfg, seeds, rbar_grid_from_r_init)
    if verbose:
        print(f"    G_FB {alt['G_FB']:.6f}  G_NP {alt['G_NP']:.6f}  "
              f"G_SB_unif {alt['G_SB_uniform']:.6f}  [{time.time() - t0:.0f}s]")
    return dict(
        published_uniform_start=pub, r_init_start=alt,
        shift={k: alt[k] - pub[k] for k in ("G_FB", "G_NP", "G_SB_uniform", "ratio_SB_over_FB",
                                            "implementation_power")},
        same_SB_policy=bool(pub["SB_policy"] == alt["SB_policy"]),
        note="If these two columns agree to the reported precision, the non-ergodic kernels of part A "
             "do not contaminate any headline number, and the stationary solver's undocumented "
             "dependence on its uniform initial guess is a defect of presentation rather than of "
             "measurement. If they disagree, the published benchmarks are the ones that are wrong.",
        seconds=round(time.time() - t0, 1))


# ==================================================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="70000-70059")
    ap.add_argument("--part", choices=["a", "b", "c", "all"], default="all")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    lo, hi = (args.seeds.split("-") + [None])[:2]
    seeds = list(range(int(lo), int(hi) + 1)) if hi else [int(lo)]

    cfg = M.Config()
    verbose = not args.quiet
    res = dict(generated=time.strftime("%Y-%m-%dT%H:%M:%S"),
               script="code/ha_dynamics_audit.py",
               question="does the restricted stationary game describe the dynamic model?",
               restrictions_tested=dict(R2="stationary vs 80-round transient path",
                                        R3="plug-in GMV(rbar) vs E[GMV(r)]",
                                        ergodicity="does a stationary distribution exist and is it "
                                                   "reached in the 300 iterations the model uses"),
               spec=M.spec_dict(cfg), seed_block=dict(name="HA-M1", n=len(seeds)),
               r_init=R_INIT, horizon=T_HORIZON)

    if OUT.exists():                       # keep the parts that are not being re-run
        res = {**json.loads(OUT.read_text()), **res}
    if args.part in ("a", "all"):
        if verbose:
            print("PART A -- ergodicity of the reputation kernels")
        res["part_A_kernel_audit"] = part_a(cfg, verbose)
    if args.part in ("b", "all"):
        if verbose:
            print("PART B -- plug-in vs expectation, stationary vs 80 rounds")
        res["part_B_payoff_restrictions"] = part_b(cfg, seeds, verbose)
    if args.part in ("c", "all"):
        if verbose:
            print("PART C -- do the headline benchmarks depend on the solver's initial guess?")
        res["part_C_benchmark_sensitivity"] = part_c(cfg, seeds, verbose)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {OUT.relative_to(PKG)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
