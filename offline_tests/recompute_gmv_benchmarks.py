#!/usr/bin/env python3
"""Independent recomputation of the two headline GMV benchmarks, G_FB and G_SB_uniform.

    python offline_tests/recompute_gmv_benchmarks.py

This deliberately imports NOTHING from `code/`. Every constant is written out again from the frozen
spec and every formula is implemented again from the model definition, so agreement with
`results/solver/ha_benchmarks.json` is evidence about the numbers rather than evidence that
`ha_benchmarks.py` is deterministic. The only shared input is `input_data/catalog_types.json`, which
is data, and the seed block, which is a design choice.

Two levels are checked:

  L1  re-aggregate from the committed per-seed records
      G_FB       = mean over seeds of per_seed[s].G_FB
      G_SB_unif  = max over policies of (mean over seeds of that policy's worst-equilibrium GMV)
      This catches an aggregation that does not match the per-seed evidence it claims to summarise.

  L2  rebuild everything from catalog_types.json + the seed block + the frozen constants
      This catches a per-seed record that does not follow from the model.

L1 must agree exactly (float64 reproduction of the same reduction). L2 must agree to 1e-9.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PKG = HERE.parent

# ------------------------------------------------------------------------------------------------
# the frozen spec, written out again rather than imported
# ------------------------------------------------------------------------------------------------
M_MERCH = 4
ALPHA, BETA, GAMMA = 3.0, 1.5, 1.0
Q0 = 1.0
ETA_R = 0.20
CS = 0.60
N_OBS = 40
NF = 21
R_GRID_N = 31
OMEGA = 0.50
LAM = 1.00
W0_BASE, W0_SPREAD = -1.0, 4.0
MARGIN_FRAC = 0.5

W0 = W0_BASE + W0_SPREAD * (1.0 - OMEGA)          # = 1.0
FGRID = np.linspace(0.0, 1.0, NF)                 # 0, 0.05, ..., 1.00
RGRID = np.linspace(0.0, 1.0, R_GRID_N)
B_GRID = np.linspace(0.01, 0.15, 15)

KAPPA_GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
TAU_GRID = (0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40)
POLICY_CLASS = [(float(k), float(t)) for k in KAPPA_GRID for t in TAU_GRID]   # 64, k outer

SEEDS = list(range(70000, 70060))                 # seed block HA-M1
MARKET_BASE = 1000
TIE_TOL = 1e-12


# ------------------------------------------------------------------------------------------------
def draw_market(seed: int):
    """q, p from the catalogue pool; b beta(2,5)*0.15 clipped, then snapped to B_GRID."""
    pool = json.loads((PKG / "input_data" / "catalog_types.json").read_text())["pool"]
    rng = np.random.default_rng(MARKET_BASE + seed)
    idx = rng.choice(len(pool), size=M_MERCH, replace=False)
    q = np.array([pool[i]["q"] for i in idx], dtype=float)
    p = np.array([pool[i]["p_norm"] for i in idx], dtype=float)
    b = np.clip(rng.beta(2, 5, M_MERCH) * 0.15, 0.01, 0.15)
    bidx = np.array([int(np.argmin(np.abs(B_GRID - v))) for v in b])
    return q, p, b, bidx


def binom_pmf(n: int, pr: float) -> np.ndarray:
    pr = min(max(float(pr), 1e-12), 1 - 1e-12)
    k = np.arange(n + 1)
    logc = (np.log(np.arange(1, n + 1)).sum()
            - np.array([np.log(np.arange(1, i + 1)).sum() for i in k])
            - np.array([np.log(np.arange(1, n - i + 1)).sum() for i in k]))
    return np.exp(logc + k * math.log(pr) + (n - k) * math.log1p(-pr))


def stationary_mean(pmf_d: np.ndarray, pen_d: np.ndarray) -> float:
    """Stationary mean of r' = clip(r + eta_r(1-r) - P, 0, 1) on the 31-point grid."""
    T = np.zeros((R_GRID_N, R_GRID_N))
    for i, r in enumerate(RGRID):
        rp = np.clip(r + ETA_R * (1 - r) - pen_d, 0.0, 1.0)
        j = np.clip(np.round(rp * (R_GRID_N - 1)).astype(int), 0, R_GRID_N - 1)
        np.add.at(T[i], j, pmf_d)
    pi = np.full(R_GRID_N, 1.0 / R_GRID_N)
    for _ in range(300):
        pi = pi @ T
    return float(pi @ RGRID)


def rbar_grid(kappa: float, tau: float) -> np.ndarray:
    """rbar[b_index, f_index]: stationary mean reputation the policy induces."""
    drate = np.arange(N_OBS + 1) / N_OBS
    pen_d = kappa * np.maximum(0.0, drate - tau)
    out = np.zeros((len(B_GRID), NF))
    for bi, b in enumerate(B_GRID):
        for fi, f in enumerate(FGRID):
            th = min(max(b + CS * f, 0.0), 1.0)
            out[bi, fi] = stationary_mean(binom_pmf(N_OBS, th), pen_d)
    return out


def enumerate_profiles(q, p, rbar_m):
    """All 21^4 joint profiles by broadcasting. Returns (GMV array, list of profit arrays)."""
    shape = (NF,) * M_MERCH
    exp_u = [np.exp(ALPHA * (q[j] + (1 - q[j]) * FGRID) + BETA * rbar_m[j] - GAMMA * p[j])
             for j in range(M_MERCH)]

    def ax(j):
        sh = [1] * M_MERCH
        sh[j] = NF
        return tuple(sh)

    den = np.full(shape, math.exp(W0))
    fsum = np.zeros(shape)
    for j in range(M_MERCH):
        den = den + exp_u[j].reshape(ax(j))
        fsum = fsum + FGRID.reshape(ax(j))
    Q = Q0 * np.exp(-LAM * fsum / M_MERCH)
    shares = [exp_u[j].reshape(ax(j)) / den for j in range(M_MERCH)]
    profit = [MARGIN_FRAC * p[j] * Q * shares[j] for j in range(M_MERCH)]
    gmv = Q * sum(p[j] * shares[j] for j in range(M_MERCH))
    return gmv, profit


def pure_nash_mask(profit):
    """Profiles with no strictly profitable unilateral deviation."""
    ok = np.ones((NF,) * M_MERCH, dtype=bool)
    for j in range(M_MERCH):
        ok &= profit[j] >= profit[j].max(axis=j, keepdims=True) - TIE_TOL
    return ok


# ------------------------------------------------------------------------------------------------
def level1(ref: dict) -> list:
    """Re-aggregate from the committed per-seed records."""
    out = []
    per = ref["per_seed"]

    fb = np.array([per[str(s)]["G_FB"] for s in SEEDS])
    out.append(("L1 G_FB = mean over 60 seeds of per_seed.G_FB",
                float(fb.mean()), ref["aggregate"]["G_FB"]["mean"]))

    # worst-equilibrium GMV, seeds x 64 policies, in the stored policy_rows order
    gw = np.array([[r["gmv_worst_eq"] for r in per[str(s)]["policy_rows"]] for s in SEEDS])
    col = gw.mean(axis=0)
    i = int(np.argmax(col))
    rows0 = per[str(SEEDS[0])]["policy_rows"]
    pol = (rows0[i]["kappa"], rows0[i]["tau"])
    ref_uni = ref["aggregate"]["G_SB_uniform"]["pessimistic"]
    out.append((f"L1 G_SB_uniform = max_policy mean_seed (argmax policy = kappa {pol[0]}, tau {pol[1]})",
                float(col[i]), ref_uni["G"]))
    out.append(("L1 argmax policy matches the stored one", float(pol[0]), ref_uni["policy"][0]))
    out.append(("L1 argmax threshold matches the stored one", float(pol[1]), ref_uni["policy"][1]))
    return out


def level2(ref: dict, verbose: bool) -> list:
    """Rebuild from catalog_types.json + seeds + constants."""
    t0 = time.time()
    tables = {}
    for (k, t) in POLICY_CLASS:
        key = (k, t if k else 0.0)          # tau is irrelevant when kappa is 0
        if key not in tables:
            tables[key] = rbar_grid(*key)
    tables[(0.0, 0.0)] = tables.get((0.0, 0.0), rbar_grid(0.0, 0.0))
    if verbose:
        print(f"  reputation tables: {len(tables)} distinct, {time.time() - t0:.1f}s")

    fb_per_seed = np.zeros(len(SEEDS))
    gw = np.zeros((len(SEEDS), len(POLICY_CLASS)))
    for si, s in enumerate(SEEDS):
        q, p, b, bidx = draw_market(s)

        gmv0, _ = enumerate_profiles(q, p, tables[(0.0, 0.0)][bidx, :])
        g_fb = float(gmv0.max())

        for pi_, (k, t) in enumerate(POLICY_CLASS):
            rbar_m = tables[(k, t if k else 0.0)][bidx, :]
            gmv, profit = enumerate_profiles(q, p, rbar_m)
            if float(gmv.max()) > g_fb + 1e-12:
                g_fb = float(gmv.max())
            ne = pure_nash_mask(profit)
            gw[si, pi_] = gmv[ne].min() if ne.any() else np.nan
        fb_per_seed[si] = g_fb
        if verbose and ((si + 1) % 10 == 0 or si == 0):
            print(f"  seed {s} ({si + 1}/{len(SEEDS)})  G_FB={g_fb:.4f}  [{time.time() - t0:.0f}s]")

    col = gw.mean(axis=0)
    i = int(np.argmax(col))
    ref_uni = ref["aggregate"]["G_SB_uniform"]["pessimistic"]
    return [
        ("L2 G_FB rebuilt from catalogue + seeds",
         float(fb_per_seed.mean()), ref["aggregate"]["G_FB"]["mean"]),
        ("L2 G_SB_uniform rebuilt from catalogue + seeds", float(col[i]), ref_uni["G"]),
        ("L2 argmax policy kappa", float(POLICY_CLASS[i][0]), ref_uni["policy"][0]),
        ("L2 argmax policy tau", float(POLICY_CLASS[i][1]), ref_uni["policy"][1]),
        ("L2 ratio G_SB_uniform / G_FB",
         float(col[i] / fb_per_seed.mean()), ref_uni["ratio_over_FB"]),
    ]


def main() -> int:
    ref = json.loads((PKG / "results" / "solver" / "ha_benchmarks.json").read_text())
    quick = "--l1-only" in sys.argv
    print("=" * 96)
    print("INDEPENDENT RECOMPUTATION OF G_FB AND G_SB_uniform")
    print("=" * 96)
    print(f"reference artefact generated {ref['generated']},  seed block "
          f"{ref['seed_block']['name']} {ref['seed_block']['seeds']} n={ref['seed_block']['n']}")
    print(f"policy class {ref['policy_class']['n_settings']} settings, "
          f"{ref['enumeration']['joint_profiles_per_seed_policy']:,} profiles per (seed, policy)\n")

    checks = level1(ref)
    if not quick:
        print("rebuilding from input_data/catalog_types.json ...")
        checks += level2(ref, verbose=True)
        print()

    worst = 0.0
    for name, got, want in checks:
        d = abs(got - want)
        worst = max(worst, d)
        flag = "PASS" if d <= 1e-9 else "FAIL"
        print(f"[{flag}] {name}\n         recomputed {got!r}\n         artefact   {want!r}\n"
              f"         |diff|     {d:.3e}")
    print("=" * 96)
    ok = worst <= 1e-9
    print(f"  {'all checks passed' if ok else 'MISMATCH'};  largest |diff| = {worst:.3e}")
    print("=" * 96)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
