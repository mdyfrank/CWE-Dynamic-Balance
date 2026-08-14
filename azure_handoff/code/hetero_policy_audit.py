"""NO-API audit: is the LLM-study rational benchmark policy correct under HETEROGENEOUS real-catalog types?

The old benchmark P_R_sym=(kappa=2,tau=.15) was chosen on the HOMOGENEOUS representative cell
(q=.4,p=1,b=.05). The LLM markets use heterogeneous catalog (q_j,p_j) and synthetic b_j. Here we compute
the heterogeneous rational equilibrium GMV over a large FIXED market sample (seeds 500000..500399, disjoint
from all LLM seeds) for every policy on the evaluated grid, and distinguish:
  - P_R_sym         : homogeneous representative-cell optimum (existing = (2,.15)).
  - P_R_het_global  : ONE fixed policy maximizing EXPECTED rational GMV across the market distribution.
  - P_R_het_market  : per-market optimum (diagnostic upper bound; NOT an implementable fixed policy).

We also audit the heterogeneous equilibrium solver: multi-init / multi-order convergence, best-response
residual (never label Nash if residual>0), multiplicity, and cycles. Writes HETERO_POLICY_AUDIT.md,
data/hetero_policy_audit.{csv,json}, figs/hetero_policy_surface.png. Pure numpy, NO API.
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import equilibrium as E
import hetero as H

POOL = H.POOL
B_GRID = H.B_GRID
OMEGA, LAM, M = 0.5, 1.0, 4                 # NB: lambda=1.0 (the erosion cell used by the LLM study),
                                            # not hetero.py's 0.8 rho* illustration.
MARKET_SEED0 = 500000                       # rational-policy-selection market seeds (frozen, disjoint)
POLICIES = [(0.0, 0.0)] + [(k, t) for k in (1.0, 2.0, 4.0) for t in (0.10, 0.15, 0.25)]
P_R_SYM = (2.0, 0.15)


def draw_market_seeded(seed):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(POOL), size=M, replace=False)
    qs = np.array([POOL[i]["q"] for i in idx], float)
    ps = np.array([POOL[i]["p_norm"] for i in idx], float)
    bs = np.clip(rng.beta(2, 5, M) * 0.15, 0.01, 0.15)
    bidx = np.array([int(np.argmin(np.abs(B_GRID - b))) for b in bs])
    return qs, ps, bidx


def best_index(cfg, qs, ps, bidx, omega, lam, rg, fi, j):
    """Global best-response f-index for merchant j on the full grid, rivals fixed at fi."""
    w0 = E._w0(cfg, omega); fgrid = cfg.fgrid
    r_oth = np.array([rg[bidx[k], fi[k]] for k in range(len(qs))])
    u_oth = cfg.alpha * (qs + (1 - qs) * fgrid[fi]) + cfg.beta * r_oth - cfg.gamma * ps
    sum_exp_oth = np.exp(u_oth).sum() - np.exp(u_oth[j])
    f_sum_oth = fgrid[fi].sum() - fgrid[fi[j]]
    u_j = cfg.alpha * (qs[j] + (1 - qs[j]) * fgrid) + cfg.beta * rg[bidx[j]] - cfg.gamma * ps[j]
    F = (f_sum_oth + fgrid) / len(qs)
    den = np.exp(w0) + sum_exp_oth + np.exp(u_j)
    s_j = np.exp(u_j) / den
    prof = (0.5 * ps[j]) * cfg.Q0 * E.g_traffic(F, lam, cfg) * s_j      # margin c_j = 0.5 p_j
    return int(np.argmax(prof))


def hetero_eq(cfg, qs, ps, bidx, omega, lam, rg, init=None, order=None, iters=400):
    """Gauss-Seidel best response. Returns (f_profile, converged, sweeps, residual, cycled)."""
    m = len(qs)
    fi = (np.zeros(m, int) if init is None else np.array(init, int)).copy()
    order = list(range(m)) if order is None else list(order)
    seen = set(); cycled = False; converged = False; sweeps = 0
    for it in range(iters):
        sweeps = it + 1
        changed = False
        for j in order:
            b = best_index(cfg, qs, ps, bidx, omega, lam, rg, fi, j)
            if b != fi[j]:
                fi[j] = b; changed = True
        if not changed:
            converged = True; break
        key = tuple(fi)
        if key in seen:
            cycled = True; break
        seen.add(key)
    # best-response residual at the returned profile (0 => unilateral BR fixed point on the grid)
    residual = sum(1 for j in range(m) if best_index(cfg, qs, ps, bidx, omega, lam, rg, fi, j) != fi[j])
    return cfg.fgrid[fi], converged, sweeps, residual, cycled


def gmv_of(cfg, qs, ps, bidx, omega, lam, rg, fprof):
    w0 = E._w0(cfg, omega)
    fi = np.array([int(np.argmin(np.abs(cfg.fgrid - f))) for f in fprof])
    rr = np.array([rg[bidx[j], fi[j]] for j in range(len(qs))])
    u = cfg.alpha * (qs + (1 - qs) * fprof) + cfg.beta * rr - cfg.gamma * ps
    den = np.exp(w0) + np.exp(u).sum()
    s = np.exp(u) / den
    Q = cfg.Q0 * float(E.g_traffic(fprof.mean(), lam, cfg))
    return float(Q * (ps * s).sum())


def boot_ci(x, B=10000, seed=11):
    x = np.asarray(x, float); rng = np.random.default_rng(seed); n = len(x)
    mu = x[rng.integers(0, n, size=(B, n))].mean(1)
    return float(x.mean()), float(np.percentile(mu, 2.5)), float(np.percentile(mu, 97.5))


def pol_label(k, t):
    return "P0" if k == 0 else f"k{k:g}_t{t:g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", type=int, default=400)
    ap.add_argument("--mult-subset", type=int, default=60, help="markets for the multi-init/order audit")
    args = ap.parse_args()
    cfg = E.Config()
    seeds = [MARKET_SEED0 + i for i in range(args.markets)]
    print(f"hetero policy audit: {len(seeds)} markets (seeds {seeds[0]}..{seeds[-1]}), "
          f"{len(POLICIES)} policies, omega={OMEGA}, lambda={LAM}")

    # precompute rbar grids per policy
    rg = {}
    for (k, t) in POLICIES:
        rg[(k, t)] = H.rbar_grid(cfg, 0.0, 0.0) if k == 0 else H.rbar_grid(cfg, k, t)
    print("precomputed rbar grids")

    # main sweep: default all-honest init, natural order (matches hetero.py behavior)
    gmv = {p: [] for p in POLICIES}
    nonconv = {p: 0 for p in POLICIES}; nonzero_resid = {p: 0 for p in POLICIES}
    rows = []
    for si, s in enumerate(seeds):
        qs, ps, bidx = draw_market_seeded(s)
        row = {"seed": s}
        for (k, t) in POLICIES:
            f, conv, sw, res, cyc = hetero_eq(cfg, qs, ps, bidx, OMEGA, LAM, rg[(k, t)])
            g = gmv_of(cfg, qs, ps, bidx, OMEGA, LAM, rg[(k, t)], f)
            gmv[(k, t)].append(g)
            if not conv:
                nonconv[(k, t)] += 1
            if res > 0:
                nonzero_resid[(k, t)] += 1
            row[pol_label(k, t)] = round(g, 5)
        rows.append(row)
        if (si + 1) % 100 == 0:
            print(f"  {si+1}/{len(seeds)} markets")

    # expected GMV by policy + CI (unit = market)
    summary = {}
    for p in POLICIES:
        m, lo, hi = boot_ci(gmv[p])
        summary[pol_label(*p)] = dict(kappa=p[0], tau=p[1], mean=m, lo=lo, hi=hi,
                                      nonconv=nonconv[p], nonzero_residual=nonzero_resid[p])
    P_R_het = max(POLICIES, key=lambda p: summary[pol_label(*p)]["mean"])

    # per-market optimum distribution
    G = np.array([[gmv[p][i] for p in POLICIES] for i in range(len(seeds))])  # markets x policies
    per_market_opt = [POLICIES[j] for j in G.argmax(axis=1)]
    from collections import Counter
    opt_freq = Counter(pol_label(*p) for p in per_market_opt)

    # regret of using P_R_sym vs P_R_het_global (paired across markets)
    i_sym = POLICIES.index(P_R_SYM); i_het = POLICIES.index(P_R_het)
    regret = G[:, i_het] - G[:, i_sym]
    rg_m, rg_lo, rg_hi = boot_ci(regret)

    # robustness: split-half argmax
    half = len(seeds) // 2
    A = G[:half].mean(0); Bm = G[half:].mean(0)
    het_A = POLICIES[int(A.argmax())]; het_B = POLICIES[int(Bm.argmax())]

    # multi-init / multi-order convergence + multiplicity audit on a subset
    mult_seeds = seeds[:args.mult_subset]
    inits = {"zero": [0, 0, 0, 0], "one": [cfg.Nf - 1] * M, "mid": [cfg.Nf // 2] * M}
    rng = np.random.default_rng(777)
    rand_inits = [rng.integers(0, cfg.Nf, M).tolist() for _ in range(3)]
    orders = [None, [3, 2, 1, 0], [1, 3, 0, 2]]
    mult_disagree = 0; mult_total = 0; any_cycle = 0; any_nonconv = 0; any_resid = 0
    gmv_spread = []
    for s in mult_seeds:
        qs, ps, bidx = draw_market_seeded(s)
        for p in POLICIES:
            gs = []
            for init in list(inits.values()) + rand_inits:
                for order in orders:
                    f, conv, sw, res, cyc = hetero_eq(cfg, qs, ps, bidx, OMEGA, LAM, rg[p], init=init, order=order)
                    gs.append(gmv_of(cfg, qs, ps, bidx, OMEGA, LAM, rg[p], f))
                    any_cycle += cyc; any_nonconv += (not conv); any_resid += (res > 0)
            mult_total += 1
            spread = max(gs) - min(gs)
            gmv_spread.append(spread)
            if spread > 1e-6:
                mult_disagree += 1

    # ---- write CSV (per-market GMV by policy) ----
    (HERE / "data").mkdir(exist_ok=True)
    with open(HERE / "data" / "hetero_policy_audit.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["seed"] + [pol_label(*p) for p in POLICIES])
        w.writeheader(); w.writerows(rows)

    audit = dict(
        markets=len(seeds), market_seeds=[seeds[0], seeds[-1]], omega=OMEGA, lam=LAM,
        policies=[pol_label(*p) for p in POLICIES], summary=summary,
        P_R_sym=pol_label(*P_R_SYM), P_R_het_global=pol_label(*P_R_het),
        P_R_het_global_kt=list(P_R_het),
        regret_sym_vs_het=dict(mean=rg_m, lo=rg_lo, hi=rg_hi),
        per_market_opt_freq=dict(opt_freq),
        split_half=dict(A=pol_label(*het_A), B=pol_label(*het_B)),
        convergence=dict(mult_markets=len(mult_seeds), mult_cells=mult_total,
                         cells_with_gmv_disagreement=mult_disagree,
                         max_gmv_spread=float(max(gmv_spread)) if gmv_spread else 0.0,
                         any_cycle=int(any_cycle), any_nonconverged=int(any_nonconv),
                         any_nonzero_residual=int(any_resid)),
    )
    json.dump(audit, open(HERE / "data" / "hetero_policy_audit.json", "w"), indent=2)

    # ---- figure: expected GMV by policy (bar + CI) ----
    labels = [pol_label(*p) for p in POLICIES]
    means = [summary[l]["mean"] for l in labels]
    los = [summary[l]["mean"] - summary[l]["lo"] for l in labels]
    his = [summary[l]["hi"] - summary[l]["mean"] for l in labels]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), dpi=150)
    xs = np.arange(len(labels))
    cols = ["#999" if l == "P0" else ("#b00" if l == pol_label(*P_R_SYM) else
            ("#1d7a4c" if l == pol_label(*P_R_het) else "#5b8ca5")) for l in labels]
    ax[0].bar(xs, means, yerr=[los, his], capsize=3, color=cols)
    ax[0].set_xticks(xs); ax[0].set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax[0].set_ylabel("expected rational GMV (het. markets)", fontsize=9)
    ax[0].set_title(f"P_R_het_global = {pol_label(*P_R_het)} (green); P_R_sym = {pol_label(*P_R_SYM)} (red)",
                    fontsize=9)
    ax[0].grid(alpha=0.2, axis="y")
    # per-market optimum frequency
    of_labels = labels
    of_vals = [opt_freq.get(l, 0) for l in of_labels]
    ax[1].bar(np.arange(len(of_labels)), of_vals, color="#5b8ca5")
    ax[1].set_xticks(np.arange(len(of_labels))); ax[1].set_xticklabels(of_labels, rotation=60, ha="right", fontsize=7)
    ax[1].set_ylabel("# markets where policy is per-market optimum", fontsize=9)
    ax[1].set_title("per-market optimum P_R_het_market (heterogeneity diagnostic)", fontsize=9)
    ax[1].grid(alpha=0.2, axis="y")
    fig.suptitle(f"Heterogeneous rational policy audit ({len(seeds)} catalog markets, omega={OMEGA}, lambda={LAM})",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    (HERE / "figs").mkdir(exist_ok=True)
    fig.savefig(HERE / "figs" / "hetero_policy_surface.png"); plt.close(fig)

    # ---- console summary ----
    print("\nEXPECTED rational GMV by policy (mean [95% CI], nonconv, nonzeroResid):")
    for l in labels:
        s = summary[l]
        print(f"  {l:>10}: {s['mean']:.4f} [{s['lo']:.4f},{s['hi']:.4f}]  nonconv={s['nonconv']} resid={s['nonzero_residual']}")
    print(f"\nP_R_sym        = {pol_label(*P_R_SYM)}  (mean GMV {summary[pol_label(*P_R_SYM)]['mean']:.4f})")
    print(f"P_R_het_global = {pol_label(*P_R_het)}  (mean GMV {summary[pol_label(*P_R_het)]['mean']:.4f})")
    print(f"regret(P_R_sym vs P_R_het_global) = {rg_m:+.4f} [{rg_lo:+.4f},{rg_hi:+.4f}] (per-market paired)")
    print(f"split-half argmax: A={pol_label(*het_A)}  B={pol_label(*het_B)}  "
          f"{'STABLE' if het_A==het_B==P_R_het else 'UNSTABLE'}")
    print(f"per-market optimum freq: {dict(opt_freq)}")
    print(f"convergence audit ({len(mult_seeds)} markets x {len(POLICIES)} policies x "
          f"{len(list(inits.values())+rand_inits)} inits x {len(orders)} orders): "
          f"cells_with_disagreement={mult_disagree}/{mult_total}, max_gmv_spread={max(gmv_spread):.2e}, "
          f"cycles={any_cycle}, nonconv={any_nonconv}, nonzero_resid={any_resid}")
    print(f"\nDECISION: {'RETAIN (2,.15) as heterogeneous-distribution optimum' if tuple(P_R_het)==P_R_SYM else 'USE corrected P_R_het_global='+pol_label(*P_R_het)}")
    print("wrote data/hetero_policy_audit.{csv,json}, figs/hetero_policy_surface.png")


if __name__ == "__main__":
    main()
