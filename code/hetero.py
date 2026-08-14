"""Phase 1C -- heterogeneous rational benchmark: the fabrication DISTRIBUTION rho*.

Theory tie-in (notes.md sec 7). The symmetric model predicts a single fabrication level; the real world
is heterogeneous, so the benchmark that is actually comparable to LLM merchants is a *distribution*
rho* over per-merchant fabrication. We draw m-merchant markets whose (quality q_j, price p_j) come from
the real catalog type pool (catalog_types.json) and whose baseline complaint propensity b_j is drawn
synthetically, then solve a restricted heterogeneous Nash equilibrium (Gauss-Seidel best response to a
fixed point). We report rho* under {no penalty} vs {second-best threshold penalty}, and how the
equilibrium fabrication correlates with a merchant's type.

Predictions from the payoff structure:
  * a_j = q_j+(1-q_j)f_j, so HIGH-quality merchants gain less appeal per unit fabrication => fabricate
    LESS (they have a smaller honest gap to fill);
  * LOW baseline-b merchants generate fewer complaints per unit fabrication => face a weaker penalty
    => fabricate MORE under a penalty.
Pure numpy, no API.
"""
from __future__ import annotations
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

POOL = json.loads((HERE / "data" / "catalog_types.json").read_text())["pool"]
OMEGA, LAM, M_PER_MARKET, N_MARKETS = 0.5, 0.8, 4, 400
KAPPA, TAU = 2.0, 0.15               # second-best threshold penalty (from E6/E7)
B_GRID = np.linspace(0.01, 0.15, 15)
OUT = HERE / "figs" / "e9_hetero.png"
RNG = np.random.default_rng(1)


def rbar_grid(cfg, kappa, tau):
    """rbar[b_idx, f_idx] stationary mean reputation for baseline b fabricating f under (kappa,tau)."""
    return np.array([[E.rbar_scalar(cfg, f, b, kappa, tau) for f in cfg.fgrid] for b in B_GRID])


def hetero_equilibrium(cfg, qs, ps, bidx, omega, lam, rg, iters=60):
    """Gauss-Seidel best response to a symmetric-restricted heterogeneous Nash fixed point."""
    m = len(qs); w0 = E._w0(cfg, omega); fgrid = cfg.fgrid
    fi = np.zeros(m, dtype=int)
    margin = 0.5 * ps
    for _ in range(iters):
        changed = False
        for j in range(m):
            r_others = np.array([rg[bidx[k], fi[k]] for k in range(m)])
            u_others = cfg.alpha * (qs + (1 - qs) * fgrid[fi]) + cfg.beta * r_others - cfg.gamma * ps
            sum_exp_oth = np.exp(u_others).sum() - np.exp(u_others[j])
            f_sum_oth = fgrid[fi].sum() - fgrid[fi[j]]
            u_j = cfg.alpha * (qs[j] + (1 - qs[j]) * fgrid) + cfg.beta * rg[bidx[j]] - cfg.gamma * ps[j]
            F = (f_sum_oth + fgrid) / m
            den = np.exp(w0) + sum_exp_oth + np.exp(u_j)
            s_j = np.exp(u_j) / den
            prof = margin[j] * cfg.Q0 * E.g_traffic(F, lam, cfg) * s_j
            b = int(np.argmax(prof))
            if b != fi[j]:
                changed = True; fi[j] = b
        if not changed:
            break
    return fgrid[fi]


def draw_market_rng(cfg, rng):
    """Draw a market from an EXPLICIT rng. No global state is read or written.

    Balance-6's runner did `H.RNG = default_rng(1000+seed)` and then called draw_market(cfg)
    from inside a run-level thread pool, so a second thread could reassign the module global
    between the assignment and the read and hand a run the wrong seed's market. Forensics
    (balance7_rngforensics.py) show the race never fired in Balance-6 -- all 35,200 records
    match a single-threaded recomputation -- but the window is real, so Balance-7 threads the
    generator explicitly instead. The draw ORDER is unchanged, so for a given generator state
    this returns exactly what the old code returned.
    """
    idx = rng.choice(len(POOL), size=M_PER_MARKET, replace=False)
    qs = np.array([POOL[i]["q"] for i in idx])
    ps = np.array([POOL[i]["p_norm"] for i in idx])
    bs = np.clip(rng.beta(2, 5, M_PER_MARKET) * 0.15, 0.01, 0.15)
    bidx = np.array([int(np.argmin(np.abs(B_GRID - b))) for b in bs])
    return qs, ps, bs, bidx


def draw_market_seeded(cfg, seed, base=1000):
    """Thread-safe market draw for market `seed`. This is the Balance-7 entry point."""
    return draw_market_rng(cfg, np.random.default_rng(base + seed))


def draw_market(cfg):
    """Legacy global-RNG draw, kept so Balance-6 artifacts stay bit-reproducible.

    DEPRECATED for concurrent use: it reads the module-level RNG. New code must call
    draw_market_seeded / draw_market_rng.
    """
    return draw_market_rng(cfg, RNG)


def main():
    cfg = E.Config()
    rg0 = rbar_grid(cfg, 0.0, 0.0)
    rgp = rbar_grid(cfg, KAPPA, TAU)

    Qs, Bs, F0, FP = [], [], [], []
    for _ in range(N_MARKETS):
        qs, ps, bs, bidx = draw_market(cfg)
        F0.append(hetero_equilibrium(cfg, qs, ps, bidx, OMEGA, LAM, rg0))
        FP.append(hetero_equilibrium(cfg, qs, ps, bidx, OMEGA, LAM, rgp))
        Qs.append(qs); Bs.append(bs)
    Qs = np.concatenate(Qs); Bs = np.concatenate(Bs)
    F0 = np.concatenate(F0); FP = np.concatenate(FP)

    fig, ax = plt.subplots(1, 3, figsize=(9.6, 3.1), dpi=150)
    bins = np.linspace(0, 1, 22)
    ax[0].hist(F0, bins=bins, color="#b00", alpha=0.6, label=f"no penalty (mean {F0.mean():.2f})")
    ax[0].hist(FP, bins=bins, color="#1d7a4c", alpha=0.6, label=f"penalty (mean {FP.mean():.2f})")
    ax[0].set_xlabel("per-merchant fabrication f", fontsize=9); ax[0].set_ylabel("count", fontsize=9)
    ax[0].set_title(r"benchmark distribution $\rho^\star$", fontsize=10)
    ax[0].legend(fontsize=7, frameon=False)

    def binned(x, y, nb=6):
        edges = np.linspace(x.min(), x.max(), nb + 1); c = 0.5 * (edges[:-1] + edges[1:])
        mu = [y[(x >= edges[i]) & (x < edges[i + 1] + (i == nb - 1) * 1e-9)].mean() for i in range(nb)]
        return c, mu
    cq0, mq0 = binned(Qs, F0); cqp, mqp = binned(Qs, FP)
    ax[1].plot(cq0, mq0, "-o", c="#b00", ms=3, label="no penalty")
    ax[1].plot(cqp, mqp, "-o", c="#1d7a4c", ms=3, label="penalty")
    ax[1].set_xlabel("merchant quality  $q_j$", fontsize=9); ax[1].set_ylabel("mean fabrication", fontsize=9)
    ax[1].set_title("higher quality $\\to$ less fabrication", fontsize=9.5)
    ax[1].legend(fontsize=7, frameon=False)
    cb0, mb0 = binned(Bs, F0); cbp, mbp = binned(Bs, FP)
    ax[2].plot(cb0, mb0, "-o", c="#b00", ms=3, label="no penalty")
    ax[2].plot(cbp, mbp, "-o", c="#1d7a4c", ms=3, label="penalty")
    ax[2].set_xlabel(r"baseline complaint prop. $b_j$", fontsize=9); ax[2].set_ylabel("mean fabrication", fontsize=9)
    ax[2].set_title("baseline $b_j$: weak/noisy effect", fontsize=9.5)
    ax[2].legend(fontsize=7, frameon=False)
    for a in ax:
        a.tick_params(labelsize=7); a.grid(alpha=0.2)
    fig.suptitle(f"E9  heterogeneous rational benchmark  (real-catalog types, m={M_PER_MARKET}, "
                 f"{N_MARKETS} markets, omega={OMEGA}, lambda={LAM})", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    OUT.parent.mkdir(parents=True, exist_ok=True); fig.savefig(OUT); plt.close(fig)

    def corr(a, b):
        if a.std() < 1e-12 or b.std() < 1e-12:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])
    print(f"rho* no-penalty: mean {F0.mean():.3f}, sd {F0.std():.3f}, frac at f>0.9 = {(F0>0.9).mean():.2f}")
    print(f"rho* penalty   : mean {FP.mean():.3f}, sd {FP.std():.3f}, frac at f<0.2 = {(FP<0.2).mean():.2f}")
    print(f"corr(q, f) no-pen {corr(Qs,F0):+.3f} | penalty {corr(Qs,FP):+.3f}   (expect negative)")
    print(f"corr(b, f) no-pen {corr(Bs,F0):+.3f} | penalty {corr(Bs,FP):+.3f}   (expect negative under penalty)")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
