#!/usr/bin/env python3
"""ha_dp_policy_probe.py -- what the dynamic best response actually DOES, not just what it is worth.

    python code/ha_dp_policy_probe.py --seeds 70000,70002 --delta 0.95

`ha_dynamic_dp.py` reports the size of the deviation gain, because that is what decides whether f*
is an equilibrium. It does not retain the argmax: `discounted()` deletes the (21, 31, 31^3) candidate
array on every sweep, since keeping it would put 155 MB per iteration against 6,000 iterations. So
the SHAPE of the deviation -- which action, at which reputation, in which direction -- is absent from
the artefact, and that shape is the whole content of the build-then-exploit story R1 exists to flag.

This recovers it on a handful of instances by rerunning the value iteration and keeping the argmax of
the last sweep only. Three things are reported that the aggregate cannot say:

  cash_in_slope         is the best-response action INCREASING in the merchant's own reputation?
                        "Build a reputation, then cash it in" is exactly a(r_i) rising in r_i: the
                        better the stock, the more it pays to spend. A flat or falling map means the
                        gain, whatever its size, is not the build-then-exploit mechanism.
  first_deviation_state Gamma_delta has no rounds, so "first" cannot mean time. It means the
                        reachable state NEAREST the experiment's own start x_0 = (0.5,...,0.5) at
                        which the best response differs from f*_i -- i.e. how far the market has to
                        drift before the profile stops being self-enforcing.
  action_map_slice      the argmax along i's own reputation with rivals held at r_0, printed in full,
                        so a reader can see the map rather than a summary statistic of it.

A probe, not a sweep: it covers the instances named on the command line and says so. No API, no
network, pure numpy.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import ha_model as M
from ha_dynamic_dp import Case, NAMED, R_INIT, T_HORIZON, VI_MAX, VI_TOL

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
OUT = PKG / "results" / "solver" / "ha_dp_policy_probe.json"


def _argmax_policy(case: Case, delta: float, tol: float = VI_TOL, cap: int = VI_MAX):
    """Value iteration on Gamma_delta, keeping the argmax of the final sweep only.

    Identical recursion to `Case.discounted`; the only difference is that the candidate array is
    argmaxed before it is dropped. `eps` is returned as well, so the probe can assert it reproduces
    the number the solver already published rather than quietly disagreeing with it.

    "Identical" has to include the stopping rule, and for a while it did not. Both loops stopped on
    `eps = V - P` alone, which is the right test for the numerator and no test at all for the
    denominator: V and P share the 1/(1-delta) level term that converges at rate delta, so their
    difference can be stationary to 1e-11 while P is still tens of percent below its fixed point.
    See `Case.discounted`, which carries the full account. The level condition below is the
    contraction bound delta/(1-delta) * ||P_n - P_{n-1}||.

    It happens that the three instances this probe has published are non-corner cases, where `eps`
    is the slow object and the level condition was already implied -- their numbers do not move. The
    rule is corrected anyway: a probe that agrees with the solver by accident is not a check.
    """
    V = np.zeros(case.shape)
    P = np.zeros(case.shape)
    eps_prev, n, resid, arg = None, 0, float("inf"), None
    lvl, fac = float("inf"), delta / (1.0 - delta) if delta < 1.0 else float("inf")
    for n in range(1, cap + 1):
        V, C = case._sweep(V, delta)                                          # noqa: SLF001
        arg = np.argmax(C, axis=0).reshape(case.shape)
        del C
        P_prev, P = P, case._eval(P, delta)                                   # noqa: SLF001
        eps = V - P
        if eps_prev is not None:
            resid = float(np.abs(eps - eps_prev).max())
            lvl = float(np.abs(P - P_prev).max()) * fac
            if resid < tol and lvl < tol:
                break
        eps_prev = eps
    return arg, V - P, P, n, resid, lvl


def _describe(case: Case, arg: np.ndarray, eps: np.ndarray, P: np.ndarray) -> dict:
    R, m, ast = case.cfg.R, case.mkt.m, case.astar
    r0 = int(round(R_INIT * (R - 1)))
    mask = case.reach_mask()
    dev = (arg != ast) & mask

    # nearest deviating state to x_0, in L1 on grid indices -- "how far must the market drift"
    idx = np.argwhere(dev)
    first = None
    if idx.size:
        d = np.abs(idx - r0).sum(axis=1)
        j = int(np.argmin(d))
        first = {"state": [int(v) for v in idx[j]],
                 "reputations": [float(case.cfg.rgrid[v]) for v in idx[j]],
                 "l1_grid_distance_from_x0": int(d[j]),
                 "action_there": int(arg[tuple(idx[j])]),
                 "f_star_i": ast,
                 "eps_there": float(eps[tuple(idx[j])])}

    # the map along i's own reputation, rivals held at r_0
    sl = tuple([slice(None)] + [r0] * (m - 1))
    line = [int(v) for v in arg[sl]]
    own_reach = [int(v) for v in np.nonzero(mask[sl])[0]]
    lr = [line[k] for k in own_reach]
    slope = float(np.polyfit(own_reach, lr, 1)[0]) if len(set(own_reach)) > 1 else float("nan")
    return {
        "f_star_i": ast,
        "action_at_x0": int(arg[(r0,) * m]),
        "action_at_x0_differs": bool(int(arg[(r0,) * m]) != ast),
        "n_distinct_actions_over_reachable_states": int(len(np.unique(arg[mask]))),
        "frac_reachable_states_where_action_differs": float(dev[mask].mean()),
        "first_deviation_state": first,
        "action_map_slice_own_reputation": line,
        "action_map_slice_reachable_indices": own_reach,
        "cash_in_slope": slope,
        "cash_in": bool(slope > 1e-9),
        "action_at_lowest_reachable_own_reputation": lr[0] if lr else None,
        "action_at_highest_reachable_own_reputation": lr[-1] if lr else None,
        "eps_at_x0": float(eps[(r0,) * m]), "V_at_x0": float(P[(r0,) * m]),
        "rel_at_x0": float(eps[(r0,) * m] / P[(r0,) * m]) if P[(r0,) * m] > 0 else float("nan"),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="70000")
    ap.add_argument("--merchants", default="all")
    ap.add_argument("--policy", default="P_SB_uniform")
    ap.add_argument("--delta", type=float, default=0.95)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)

    seeds = [int(v) for v in a.seeds.replace(",", " ").split()]
    cfg = M.Config()
    kappa, tau = NAMED[a.policy]
    rg = M.rbar_grid(cfg, kappa, tau)
    rows, t0 = [], time.time()
    for seed in seeds:
        mkt = M.draw_market(cfg, seed)
        eqs = M.pure_nash(cfg, mkt, rg[mkt.bidx, :])
        if len(eqs) != 1:
            rows.append({"seed": seed, "skipped": True, "n_pure_nash": len(eqs)})
            continue
        fstar = np.array(eqs[0], dtype=int)
        ms = range(mkt.m) if a.merchants == "all" else [int(v) for v in a.merchants.split(",")]
        rec = {"seed": seed, "f_star": [int(v) for v in fstar], "merchants": []}
        for i in ms:
            c = Case(cfg, mkt, i, fstar, kappa, tau)
            arg, eps, P, n, resid, lvl = _argmax_policy(c, a.delta)
            d = _describe(c, arg, eps, P)
            d.update(merchant=i, iterations=n, eps_residual=resid, level_residual=lvl,
                     converged=bool(resid < VI_TOL and lvl < VI_TOL))
            rec["merchants"].append(d)
            print(f"  seed {seed} m{i}  f*={c.astar}  a(x0)={d['action_at_x0']}  "
                  f"slope={d['cash_in_slope']:+.3f}  distinct={d['n_distinct_actions_over_reachable_states']}  "
                  f"rel@x0={d['rel_at_x0']:.3%}  [{time.time() - t0:.0f}s]")
            del c, arg, eps, P
        rows.append(rec)

    allm = [m for r in rows if not r.get("skipped") for m in r["merchants"]]
    payload = {
        "schema_version": "ha-dp-policy-probe-1",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script": "code/ha_dp_policy_probe.py", "policy": a.policy, "kappa": kappa, "tau": tau,
        "delta": a.delta, "horizon_note": f"Gamma_delta only; Gamma_{T_HORIZON} argmax is already in "
                                          f"ha_dynamic_equilibrium.json as actions_at_x0_by_round",
        "coverage": {"seeds": seeds, "merchants": a.merchants, "n_instances": len(allm),
                     "is_a_probe_not_a_sweep": True},
        "summary": {
            "n_with_cash_in_slope_positive": int(sum(1 for m in allm if m["cash_in"])),
            "n_with_a_different_action_at_x0": int(sum(1 for m in allm
                                                       if m["action_at_x0_differs"])),
            "max_distinct_actions_over_reachable_states":
                int(max((m["n_distinct_actions_over_reachable_states"] for m in allm), default=0)),
            "min_l1_distance_from_x0_to_a_deviating_state":
                min((m["first_deviation_state"]["l1_grid_distance_from_x0"] for m in allm
                     if m["first_deviation_state"]), default=None),
            "all_converged": bool(all(m["converged"] for m in allm)),
            "max_eps_residual": float(max((m["eps_residual"] for m in allm), default=0.0)),
            "max_level_residual": float(max((m["level_residual"] for m in allm), default=0.0)),
        },
        "per_seed": rows,
        "seconds": round(time.time() - t0, 1),
    }
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {p.relative_to(PKG) if p.resolve().is_relative_to(PKG) else p}  "
          f"[{payload['seconds']}s]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
