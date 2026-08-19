"""ha_dynamic_dp.py -- is the stationary profile an equilibrium of the DYNAMIC game?

Restriction R1 of theory/HIDDEN_ACTION_THEORY.md section 2.3 confines merchants to actions that are
constant in time and in the state. Every benchmark in the frozen corpus is computed under it. R1 is
exactly the restriction that rules out the deviation everyone suspects: build a reputation while
honest, then cash it in. This module removes R1 and asks what it was hiding.

Fix rivals at the enumerated pure-strategy profile f* of G(theta,kappa,tau). Merchant i is then facing
a Markov decision process, and the question is whether the constant action f*_i solves it. The answer
is a number:

    eps_dyn(x) = V_i^BR(x) - V_i^{f*}(x) >= 0

reported at the initial state the experiment actually starts from, at its maximum over the whole
state space, and in expectation under the equilibrium's own stationary law.

THE STATE SPACE IS THE JOINT ONE. Merchant i's share has every rival's reputation in its logit
denominator, so i's reward is not separable in (r_i, r_-i) and a best response may condition on
rivals. Solving the 31-state problem in r_i alone would give a LOWER bound on eps_dyn, and a lower
bound of zero certifies nothing. So the DP runs on all 31^4 = 923,521 states. That is affordable
because rivals hold constant actions, which makes the transition kernel a product of four fixed
31x31 matrices and every Bellman sweep a sequence of mode products.

Two horizons, because the experiment and its subjects disagree about which game is being played
(section 2.2). Gamma_80 is the 80-round truncation the runner executes; Gamma_delta is the
indefinite-horizon game the prompt describes and gate G-H4 protects. Backward induction from a known
last round produces end-game exploitation that no subject of this experiment has been told to expect,
so both are computed and neither is presented as the other.

Output: results/solver/ha_dynamic_equilibrium.json.  No API, no network, pure numpy.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

import ha_model as M
from ha_dynamics_audit import kernel

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
OUT = PKG / "results" / "solver" / "ha_dynamic_equilibrium.json"

NAMED = {"P_SB_uniform": (2.0, 0.30), "P_GMV": (0.5, 0.20), "P_robust": (4.0, 0.30)}
R_INIT = 0.5
T_HORIZON = 80
DELTAS = (0.90, 0.95, 0.99)
VI_MAX = 6000
VI_TOL = 1e-11          # eps_dyn itself is O(1e-3..1e-1), so this is eight orders below the signal


def _mode(T: np.ndarray, V: np.ndarray, axis: int) -> np.ndarray:
    """(T V) contracted along `axis`, result put back on `axis`."""
    return np.moveaxis(np.tensordot(T, V, axes=([1], [axis])), 0, axis)


class Case:
    """Merchant i's decision problem with rivals frozen at f*_{-i}. Axis 0 is always merchant i."""

    def __init__(self, cfg: M.Config, mkt: M.Market, i: int, fstar: np.ndarray,
                 kappa: float, tau: float):
        self.cfg, self.mkt, self.i, self.fstar = cfg, mkt, i, np.asarray(fstar, dtype=int)
        R, Nf, m = cfg.R, cfg.Nf, mkt.m
        self.others = [k for k in range(m) if k != i]
        drate = np.arange(cfg.N_obs + 1) / cfg.N_obs
        pen_d = kappa * np.maximum(0.0, drate - tau)

        # kernels: rivals fixed, merchant i one per candidate action
        self.T_riv = [kernel(cfg, M.complaint_pmf(cfg, cfg.fgrid[self.fstar[k]], mkt.b[k]), pen_d)
                      for k in self.others]
        self.T_i = np.stack([kernel(cfg, M.complaint_pmf(cfg, f, mkt.b[i]), pen_d)
                             for f in cfg.fgrid])

        # exp(u) for the frozen rivals, and the parts of the logit that do not involve i
        Ek = {k: np.exp(cfg.alpha * (mkt.q[k] + (1 - mkt.q[k]) * cfg.fgrid[self.fstar[k]])
                        + cfg.beta * cfg.rgrid - cfg.gamma * mkt.p[k]) for k in self.others}
        S = np.full((R,) * (m - 1), math.exp(cfg.w0))      # outside option + rivals' exp(u)
        A = np.zeros((R,) * (m - 1))                       # sum_k p_k exp(u_k) over rivals
        for a, k in enumerate(self.others):
            sh = [1] * (m - 1)
            sh[a] = R
            S = S + Ek[k].reshape(sh)
            A = A + mkt.p[k] * Ek[k].reshape(sh)
        self.S, self.A = S[None, ...], A[None, ...]

        # exp(u_i) per candidate action, and the traffic factor it implies
        self.Ei = np.exp(cfg.alpha * (mkt.q[i] + (1 - mkt.q[i]) * cfg.fgrid[:, None])
                         + cfg.beta * cfg.rgrid[None, :] - cfg.gamma * mkt.p[i])   # (Nf, R)
        fo = float(cfg.fgrid[self.fstar[self.others]].sum())
        self.Q = cfg.Q0 * np.exp(-cfg.lam * (cfg.fgrid + fo) / m)                  # (Nf,)

        # the stacked reward, built once: (Nf, R, R, R, R). Kept flat as (Nf, R, R^(m-1)) as well,
        # because every Bellman sweep streams all 155 MB of it and the reshape must be a free view.
        e = self.Ei[:, :, None, None, None]
        den = self.S[None, ...] + e
        self.RWD = np.ascontiguousarray(
            (cfg.margin_frac * mkt.p[i] * self.Q[:, None, None, None, None]) * e / den)
        self.GMV = self.Q[:, None, None, None, None] * (self.A[None, ...] + mkt.p[i] * e) / den
        self.RWDf = self.RWD.reshape(Nf, R, -1)
        self.astar = int(self.fstar[i])
        self.shape = (R,) * m
        self.x0 = (15,) * m

    def _riv(self, V: np.ndarray) -> np.ndarray:
        for a in range(len(self.others)):
            V = _mode(self.T_riv[a], V, a + 1)
        return np.ascontiguousarray(V)

    def _sweep(self, V: np.ndarray, disc: float, out: np.ndarray = None) -> np.ndarray:
        """One Bellman maximisation over all 21 actions on all 31^4 states.

        `T_i @ W` with T_i of shape (Nf, R, R) and W flattened to (R, R^(m-1)) is a single batched
        gemm, which is what makes the full joint state space affordable at all.
        """
        R, Nf = self.cfg.R, self.cfg.Nf
        C = np.matmul(self.T_i, self._riv(V).reshape(R, -1))      # (Nf, R, R^(m-1))
        if disc != 1.0:
            C *= disc
        C += self.RWDf
        return np.max(C, axis=0, out=out).reshape(self.shape), C

    def _eval(self, P: np.ndarray, disc: float) -> np.ndarray:
        """The same sweep with the action frozen at f*_i -- policy evaluation, not maximisation."""
        return self.RWD[self.astar] + disc * _mode(self.T_i[self.astar], self._riv(P), 0)

    # ------------------------------------------------------------------------------------------
    def finite(self, T: int = T_HORIZON, track: bool = True) -> dict:
        """Backward induction on Gamma_T: exact, no approximation, all 31^4 states."""
        V = np.zeros(self.shape)          # best response
        P = np.zeros(self.shape)          # value of holding f*_i
        flat0 = int(np.ravel_multi_index(self.x0[1:], self.shape[1:]))
        acts, n_state_dep = [], []
        # a full argmax over 923,521 states costs as much as the sweep itself, so the state-dependence
        # diagnostic is taken at three rounds and the on-path action -- one column of C -- at every one
        probe = {T, max(1, T // 2), 1}
        for t in range(T, 0, -1):
            V, C = self._sweep(V, 1.0)
            acts.append(int(np.argmax(C[:, self.x0[0], flat0])))
            n_state_dep.append(int(np.unique(C.argmax(axis=0)).size) if t in probe else -1)
            del C
            P = self._eval(P, 1.0)
        acts = acts[::-1]
        n_state_dep = n_state_dep[::-1]
        return dict(V=V, P=P, actions_at_x0_by_round=acts,
                    n_distinct_actions_by_round=[v for v in n_state_dep if v >= 0])

    def discounted(self, delta: float, tol: float = VI_TOL, cap: int = VI_MAX) -> dict:
        """Value iteration on Gamma_delta, run in lockstep for V^BR and V^{f*}.

        The two value functions share the large 1/(1-delta) level term, which is exactly the part
        that converges at rate delta. Their DIFFERENCE converges at the chain's mixing rate instead,
        so the stopping rule is applied to eps = V - P and the residual is reported rather than
        assumed negligible.
        """
        V = np.zeros(self.shape)
        P = np.zeros(self.shape)
        eps_prev, n, resid = None, 0, float("inf")
        for n in range(1, cap + 1):
            V, C = self._sweep(V, delta)
            del C
            P = self._eval(P, delta)
            eps = V - P
            if eps_prev is not None:
                resid = float(np.abs(eps - eps_prev).max())
                if resid < tol:
                    break
            eps_prev = eps
        return dict(V=V, P=P, iterations=n, eps_residual=resid, converged=bool(resid < tol))

    # ------------------------------------------------------------------------------------------
    def stationary_law(self) -> np.ndarray:
        """Equilibrium joint law: the product of the four merchants' own stationary distributions."""
        w = None
        Ts = [self.T_i[self.astar]] + self.T_riv
        for T in Ts:
            p = np.full(self.cfg.R, 1.0 / self.cfg.R)
            for _ in range(300):
                p = p @ T
            w = p if w is None else np.multiply.outer(w, p)
        return w

    def reach_mask(self) -> np.ndarray:
        """States reachable from x0 = (0.5,...,0.5) under ANY action of i and f* for the rivals."""
        R = self.cfg.R
        sets = []
        sup_i = (self.T_i.sum(axis=0) > 0)
        for T in [sup_i] + [t > 0 for t in self.T_riv]:
            seen, stack = {15}, [15]
            while stack:
                a = stack.pop()
                for b in np.nonzero(T[a])[0]:
                    if int(b) not in seen:
                        seen.add(int(b))
                        stack.append(int(b))
            v = np.zeros(R, bool)
            v[sorted(seen)] = True
            sets.append(v)
        mask = sets[0][:, None, None, None] & sets[1][None, :, None, None] \
            & sets[2][None, None, :, None] & sets[3][None, None, None, :]
        return mask


def _report(case: Case, V: np.ndarray, P: np.ndarray, w: np.ndarray, mask: np.ndarray) -> dict:
    eps = V - P
    x0 = (15,) * case.mkt.m
    pos = eps > 1e-12
    return dict(
        eps_at_x0=float(eps[x0]), V_at_x0=float(P[x0]),
        rel_at_x0=float(eps[x0] / P[x0]) if P[x0] > 0 else float("nan"),
        eps_max=float(eps.max()), eps_max_reachable=float(eps[mask].max()),
        rel_max=float((eps / np.maximum(P, 1e-15)).max()),
        rel_max_reachable=float((eps[mask] / np.maximum(P[mask], 1e-15)).max()),
        eps_mean_under_stationary=float((eps * w).sum()),
        rel_mean_under_stationary=float((eps * w).sum() / (P * w).sum()),
        eps_min=float(eps.min()),
        frac_states_with_gain=float(pos.mean()),
        frac_reachable_states_with_gain=float(pos[mask].mean()),
        n_reachable=int(mask.sum()))


def solve_seed(cfg: M.Config, seed: int, kappa: float, tau: float, rg: np.ndarray,
               deltas: tuple, do_finite: bool, verbose: bool) -> dict:
    mkt = M.draw_market(cfg, seed)
    rbar_m = rg[mkt.bidx, :]
    eqs = M.pure_nash(cfg, mkt, rbar_m)
    if len(eqs) != 1:
        return dict(seed=seed, n_pure_nash=len(eqs), skipped=True)
    fstar = np.array(eqs[0], dtype=int)
    out = dict(seed=seed, f_star=[int(v) for v in fstar],
               f_star_levels=[float(cfg.fgrid[v]) for v in fstar],
               eps_static_max=float(M.exploitability(cfg, mkt, rbar_m, fstar).max()),
               merchants=[])
    for i in range(mkt.m):
        c = Case(cfg, mkt, i, fstar, kappa, tau)
        w, mask = c.stationary_law(), c.reach_mask()
        rec = dict(merchant=i, q=float(mkt.q[i]), p=float(mkt.p[i]), b=float(mkt.b[i]),
                   f_star=int(fstar[i]))
        if do_finite:
            f = c.finite()
            rec["gamma_80"] = _report(c, f["V"], f["P"], w, mask)
            a = f["actions_at_x0_by_round"]
            rec["gamma_80"].update(
                action_at_x0_round_1=a[0], action_at_x0_round_T=a[-1],
                action_at_x0_distinct=len(set(a)),
                build_then_exploit=bool(a[-1] > a[0]),
                first_round_action_differs_from_f_star=bool(a[0] != int(fstar[i])),
                max_distinct_actions_over_states=int(max(f["n_distinct_actions_by_round"])),
                actions_at_x0_by_round=a)
            del f
        for d in deltas:
            r = c.discounted(d)
            rec[f"gamma_delta_{d}"] = dict(**_report(c, r["V"], r["P"], w, mask),
                                           iterations=r["iterations"],
                                           eps_residual=r["eps_residual"],
                                           converged=r["converged"])
            del r
        out["merchants"].append(rec)
        del c
    if verbose:
        g = [m for m in out["merchants"]]
        key = "gamma_80" if do_finite else f"gamma_delta_{deltas[0]}"
        print(f"    seed {seed} f*={out['f_star']}  max eps_dyn@x0 "
              f"{max(m[key]['eps_at_x0'] for m in g):.3e}  "
              f"max rel {max(m[key]['rel_max_reachable'] for m in g):.3%}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="70000-70059")
    ap.add_argument("--policies", default="P_SB_uniform")
    ap.add_argument("--deltas", default="0.90,0.95,0.99")
    ap.add_argument("--no-finite", action="store_true")
    ap.add_argument("--out", default=None, help="shard path; parallel runs must not share one")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    # resolved, because the closing log line reports the path relative to the package and a shard
    # named with a `..` prefix from inside `code/` is not literally a subpath of it. That raised
    # AFTER a two-hour block had already been written to disk, so the run looked like a failure and
    # the driver aborted the remaining blocks over a cosmetic string.
    out = (Path(args.out).resolve() if args.out else OUT)
    lo, hi = (args.seeds.split("-") + [None])[:2]
    seeds = list(range(int(lo), int(hi) + 1)) if hi else [int(lo)]
    deltas = tuple(float(v) for v in args.deltas.split(",") if v)
    pols = [p for p in args.policies.split(",") if p]
    cfg, verbose = M.Config(), not args.quiet

    res = json.loads(out.read_text()) if out.exists() else {}
    res.update(generated=time.strftime("%Y-%m-%dT%H:%M:%S"), script="code/ha_dynamic_dp.py",
               question="does the stationary profile survive as an equilibrium of the dynamic game?",
               state_space=dict(joint=cfg.R ** cfg.m, per_merchant=cfg.R, note="full joint state; the "
                                "31-state reduction in r_i alone would only lower-bound eps_dyn"),
               horizon=T_HORIZON, r_init=R_INIT, deltas=list(deltas), spec=M.spec_dict(cfg))
    res.setdefault("policies", {})

    t0 = time.time()
    for pname in pols:
        kappa, tau = NAMED[pname]
        if verbose:
            print(f"  policy {pname} (kappa={kappa}, tau={tau})")
        rg = M.rbar_grid(cfg, kappa, tau)
        rows = [solve_seed(cfg, s, kappa, tau, rg, deltas, not args.no_finite, verbose)
                for s in seeds]
        good = [r for r in rows if not r.get("skipped")]
        agg = {}
        for key in (["gamma_80"] if not args.no_finite else []) + [f"gamma_delta_{d}" for d in deltas]:
            allm = [m[key] for r in good for m in r["merchants"]]
            agg[key] = dict(
                n_merchant_cases=len(allm),
                max_eps_at_x0=float(max(a["eps_at_x0"] for a in allm)),
                max_rel_at_x0=float(max(a["rel_at_x0"] for a in allm)),
                max_eps_reachable=float(max(a["eps_max_reachable"] for a in allm)),
                max_rel_reachable=float(max(a["rel_max_reachable"] for a in allm)),
                max_rel_under_stationary=float(max(a["rel_mean_under_stationary"] for a in allm)),
                n_cases_with_gain_at_x0=int(sum(a["eps_at_x0"] > 1e-12 for a in allm)),
                n_cases_with_gain_anywhere_reachable=int(
                    sum(a["eps_max_reachable"] > 1e-12 for a in allm)),
                min_eps=float(min(a["eps_min"] for a in allm)))
            if key == "gamma_80":
                agg[key].update(
                    n_build_then_exploit=int(sum(a["build_then_exploit"] for a in allm)),
                    n_first_round_differs=int(sum(a["first_round_action_differs_from_f_star"]
                                                  for a in allm)),
                    max_distinct_actions_over_states=int(
                        max(a["max_distinct_actions_over_states"] for a in allm)))
            else:
                agg[key]["all_converged"] = bool(all(a["converged"] for a in allm))
                agg[key]["max_residual"] = float(max(a["eps_residual"] for a in allm))
        res["policies"][pname] = dict(policy=[kappa, tau], n_seeds=len(good),
                                      n_skipped=len(rows) - len(good), aggregate=agg,
                                      per_seed=rows)
        if verbose:
            print(f"    {pname} done [{time.time() - t0:.0f}s]")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    shown = out.relative_to(PKG) if out.is_relative_to(PKG) else out
    print(f"wrote {shown}  [{time.time() - t0:.0f}s]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
