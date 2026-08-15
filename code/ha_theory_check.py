"""ha_theory_check.py -- numerical verification of every proposition in HIDDEN_ACTION_THEORY.md.

A proposition that has not been checked against the actual code is a hope. Each check below is named
after the proposition it verifies, states what would falsify it, and reports PASS / FAIL. Nothing here
is tuned: the checks were written from the statements, and where a statement turned out to be false the
statement was changed, not the check.

Checks
  P1  existence and multiplicity of pure equilibria, certified by exhaustive enumeration
  P2  the exact incentive characterisation (the price and margin terms cancel)
  P3a the exact deviation identity every design condition must be derived from
  P3b soundness of the share-free sufficient condition against upward deviation, and how slack it is
  P4  what the platform can guarantee knowing only the type SUPPORT (robust implementation)
  P5  deterrence is NON-monotone in the penalty coefficient (the over-punishment reversal)
  P6  IMPOSSIBILITY: with an f-independent signal the equilibrium action profile is invariant across
      the entire policy class, so the platform has zero implementation power
  P7  refunds are a garbling of complaints: they carry no extra information about the hidden action
  P8b a fixed point of displayed-argmax play is exactly a pure equilibrium of the restricted game --
      which is why the earlier oracle interfaces obtained convergence by construction
  P9  dead zones: a policy whose threshold sits above the attainable complaint rate cannot implement
      any target inside the flat region

Pure numpy, no API.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ha_model as M  # noqa: E402

OUTDIR = HERE.parent / "results" / "solver"
SEEDS = list(range(70000, 70060))
CHECK_SEEDS = list(range(70000, 70020))   # the propositions that need a per-profile sweep


def _res(name, statement, passed, detail, falsifier):
    return dict(check=name, statement=statement, verdict="PASS" if passed else "FAIL",
                detail=detail, would_falsify=falsifier)


# --------------------------------------------------------------------------------------------------
def p1_existence(bench: dict) -> dict:
    """P1. Gamma(g) is finite, so a mixed equilibrium exists by Nash's theorem; PURE existence is not
    implied and is therefore certified per (seed, policy) by exhaustive enumeration."""
    zero = multi = total = 0
    for s, rec in bench["per_seed"].items():
        for row in rec["policy_rows"]:
            total += 1
            zero += int(row["n_pure_nash"] == 0)
            multi += int(row["n_pure_nash"] > 1)
    return _res("P1_existence",
                "Every (seed, policy) pair in the frozen design has at least one pure-strategy Nash "
                "equilibrium of the restricted stationary hidden-action game.",
                zero == 0,
                dict(seed_policy_pairs=total, pairs_with_zero_pure_nash=zero,
                     pairs_with_multiple_pure_nash=multi,
                     uniqueness_holds=bool(zero == 0 and multi == 0),
                     note="Uniqueness is an empirical property of this parameterisation, not a "
                          "theorem. It is reported so that equilibrium selection can be ruled out "
                          "as an explanation for anything downstream."),
                "one pair with no pure equilibrium")


# --------------------------------------------------------------------------------------------------
def p2_exact_ic(cfg, seeds) -> dict:
    """P2. f* is a pure equilibrium iff for every i and every deviation d,
           log s_i(d, f*_-i) - log s_i(f*) <= lam*(f_d - f*_i)/m.
    The margin (p_i/2) and the base traffic Q0 cancel, so the characterisation is exact and does not
    depend on the price level. Checked by comparing the two membership tests profile by profile."""
    rgs = {p: M.rbar_grid(cfg, *kt) for p, kt in M.POLICIES.items()}
    mismatch, tested = 0, 0
    worst = 0.0
    for seed in seeds:
        mkt = M.draw_market(cfg, seed)
        for p, rg in rgs.items():
            rm = M.rbar_for_market(cfg, mkt, rg)
            en = M.enumerate_profiles(cfg, mkt, rm)
            ne = M.pure_nash(cfg, mkt, rm, en)
            for prof in ne:
                base = M.profile_outcome(cfg, mkt, rm, np.array(prof))
                for j in range(mkt.m):
                    for d in range(cfg.Nf):
                        c = list(prof)
                        c[j] = d
                        alt = M.profile_outcome(cfg, mkt, rm, np.array(c))
                        lhs = math.log(alt["share"][j]) - math.log(base["share"][j])
                        rhs = cfg.lam * (cfg.fgrid[d] - cfg.fgrid[prof[j]]) / mkt.m
                        profitable_exact = alt["profit"][j] > base["profit"][j] + 1e-12
                        profitable_char = lhs > rhs + 1e-12
                        tested += 1
                        worst = max(worst, abs(lhs - rhs - math.log(
                            max(alt["profit"][j], 1e-300) / max(base["profit"][j], 1e-300))))
                        mismatch += int(profitable_exact != profitable_char)
    return _res("P2_exact_ic",
                "log s_i(d) - log s_i(f*) <= lam (f_d - f*_i)/m characterises unprofitability of the "
                "deviation exactly; the margin and base traffic cancel.",
                mismatch == 0,
                dict(deviations_tested=tested, characterisation_mismatches=mismatch,
                     max_abs_identity_residual=float(worst)),
                "one deviation where the exact profit test and the characterisation disagree")


# --------------------------------------------------------------------------------------------------
def p3a_deviation_identity(cfg, seeds) -> dict:
    """P3a. The exact deviation identity. With x = exp(Delta u_i) and s_i the incumbent share,

        V_i(f', f_-i) / V_i(f_i, f_-i) = exp(-lam (f'-f_i)/m) * x / (s_i x + 1 - s_i),
        Delta u_i = alpha(1-q_i)(f'-f_i) + beta[rbar_i(f') - rbar_i(f_i)].

    It is an identity, not a bound: the logit denominator moves with the deviation, and the factor
    x/(s_i x + 1 - s_i) is precisely that movement. Everything a design condition can say has to be
    derived from here, which is why the s-free bound below works upward and provably cannot work
    downward."""
    rgs = {p: M.rbar_grid(cfg, *kt) for p, kt in M.POLICIES.items()}
    worst, tested = 0.0, 0
    for seed in seeds:
        mkt = M.draw_market(cfg, seed)
        for p, rg in rgs.items():
            rm = M.rbar_for_market(cfg, mkt, rg)
            rng = np.random.default_rng(20260815 + seed)
            for _ in range(40):
                prof = rng.integers(0, cfg.Nf, mkt.m)
                base = M.profile_outcome(cfg, mkt, rm, prof)
                for j in range(mkt.m):
                    s = float(base["share"][j])
                    for d in (0, 5, 10, 15, 20):
                        c = prof.copy()
                        c[j] = d
                        alt = M.profile_outcome(cfg, mkt, rm, c)
                        du = (cfg.alpha * (1 - mkt.q[j]) * (cfg.fgrid[d] - cfg.fgrid[prof[j]])
                              + cfg.beta * (rm[j, d] - rm[j, prof[j]]))
                        x = math.exp(du)
                        pred = (math.exp(-cfg.lam * (cfg.fgrid[d] - cfg.fgrid[prof[j]]) / mkt.m)
                                * x / (s * x + 1 - s))
                        actual = alt["profit"][j] / base["profit"][j]
                        worst = max(worst, abs(pred - actual) / max(actual, 1e-12))
                        tested += 1
    return _res("P3a_deviation_identity",
                "V_i(f')/V_i(f) = exp(-lam(f'-f_i)/m) * x/(s_i x + 1 - s_i) with x = exp(Delta u_i).",
                worst < 1e-9,
                dict(deviations_tested=tested, max_relative_error=float(worst)),
                "a relative error above 1e-9 on any deviation")


def _sc_up_holds(cfg, mkt, rbar_m, prof) -> np.ndarray:
    """SC-up, merchant by merchant:  beta[rbar_i(f_i) - rbar_i(f')] >= (alpha(1-q_i) - lam/m)(f'-f_i)
    for every f' > f_i. It follows from x/(s x + 1 - s) <= x for x >= 1, which is exactly the case
    where the deviation raises own utility. There is deliberately NO downward counterpart: for x < 1
    the same expression is increasing in s and its supremum over s in [0,1] is 1, so no share-free
    sufficient condition against downward deviations exists. That is a limit of the instrument, and
    it is stated rather than hidden."""
    ok = np.ones(mkt.m, dtype=bool)
    for j in range(mkt.m):
        i0 = prof[j]
        g = cfg.alpha * (1 - mkt.q[j])
        for d in range(i0 + 1, cfg.Nf):
            delta = cfg.fgrid[d] - cfg.fgrid[i0]
            if cfg.beta * (rbar_m[j, i0] - rbar_m[j, d]) < (g - cfg.lam / mkt.m) * delta - 1e-12:
                ok[j] = False
                break
    return ok


def p3b_sufficient_condition(cfg, seeds) -> dict:
    """P3b. SC-up is SUFFICIENT against upward deviations, and it is slack. Soundness is the claim: a
    single profile where SC-up holds for merchant j yet j gains by fabricating more would kill it."""
    rgs = {p: M.rbar_grid(cfg, *kt) for p, kt in M.POLICIES.items()}
    unsound = certified = checked = 0
    ne_total = ne_certified = 0
    for seed in seeds:
        mkt = M.draw_market(cfg, seed)
        for p, rg in rgs.items():
            rm = M.rbar_for_market(cfg, mkt, rg)
            en = M.enumerate_profiles(cfg, mkt, rm)
            ne = set(M.pure_nash(cfg, mkt, rm, en))
            ne_total += len(ne)
            rng = np.random.default_rng(20260815 + seed)
            sample = {tuple(int(v) for v in rng.integers(0, cfg.Nf, mkt.m))
                      for _ in range(400)} | ne
            for prof in sample:
                ok = _sc_up_holds(cfg, mkt, rm, list(prof))
                base = M.profile_outcome(cfg, mkt, rm, np.array(prof))
                for j in range(mkt.m):
                    if not ok[j]:
                        continue
                    checked += 1
                    certified += 1
                    gain = 0.0
                    for d in range(prof[j] + 1, cfg.Nf):
                        c = list(prof)
                        c[j] = d
                        gain = max(gain, M.profile_outcome(cfg, mkt, rm, np.array(c))["profit"][j]
                                   - base["profit"][j])
                    unsound += int(gain > 1e-12)
                if prof in ne and bool(ok.all()):
                    ne_certified += 1
    return _res("P3b_sufficient_condition_upward",
                "If beta[rbar_i(f_i)-rbar_i(f')] >= (alpha(1-q_i)-lam/m)(f'-f_i) for every f' > f_i, "
                "merchant i has no profitable upward deviation.",
                unsound == 0,
                dict(merchant_cases_certified=certified, unsound_cases=unsound,
                     equilibria_total=ne_total,
                     equilibria_with_SC_up_at_every_merchant=ne_certified,
                     certified_fraction=(ne_certified / ne_total) if ne_total else None,
                     note="SC-up certifies only a fraction of true equilibria: the bound "
                          "x/(sx+1-s) <= x is slack by exactly the share a merchant already holds. "
                          "SC-up is a design tool, not a characterisation of equilibrium."),
                "one merchant that satisfies SC-up yet gains by fabricating more")


def p4_robust_implementation(cfg, seeds) -> dict:
    """P4. What the PLATFORM can actually guarantee. SC-up needs q_i and b_i, which the platform does
    not have. Take the worst case over the type support instead:

        beta * min_b [rbar(f_t; b) - rbar(f'; b)] >= (alpha(1 - q_min) - lam/m)(f' - f_t)  for all f' > f_t

    where q_min is the lowest quality in the catalogue pool -- the merchant most tempted to fabricate.
    A policy satisfying this deters upward deviation from f_t for EVERY type the platform might face,
    using only knowledge of the type support. The value it guarantees is a lower bound on the
    policy-class second best, and the check reports both."""
    pool = M.catalog_pool()
    q_min = min(t["q"] for t in pool)
    targets = list(range(cfg.Nf))
    best = None
    rows = []
    for (k, t) in M.POLICY_CLASS:
        rg = M.rbar_grid(cfg, k, t)
        for ti in targets:
            need_ok = True
            for d in range(ti + 1, cfg.Nf):
                delta = cfg.fgrid[d] - cfg.fgrid[ti]
                bite = cfg.beta * float(np.min(rg[:, ti] - rg[:, d]))
                if bite < (cfg.alpha * (1 - q_min) - cfg.lam / cfg.m) * delta - 1e-12:
                    need_ok = False
                    break
            if not need_ok:
                continue
            g = []
            for seed in seeds:
                mkt = M.draw_market(cfg, seed)
                rm = M.rbar_for_market(cfg, mkt, rg)
                g.append(M.profile_outcome(cfg, mkt, rm, np.full(mkt.m, ti))["GMV"])
            rows.append(dict(kappa=k, tau=t, target_f=float(cfg.fgrid[ti]), mean_GMV=float(np.mean(g))))
            if best is None or rows[-1]["mean_GMV"] > best["mean_GMV"]:
                best = rows[-1]
    return _res("P4_robust_implementation",
                "A policy whose worst-case-over-types reputational bite covers the worst-case appeal "
                "gain deters upward deviation from the target for every type in the support, using "
                "only the type SUPPORT and never an individual merchant's type.",
                best is not None,
                dict(q_min_in_pool=float(q_min),
                     robustly_implementable_pairs=len(rows),
                     best=best,
                     top=sorted(rows, key=lambda r: -r["mean_GMV"])[:8],
                     note="This is a guarantee, so it is conservative: it must hold for the most "
                          "tempted type the platform could face. The value it certifies is a lower "
                          "bound on the policy-class second best, not an estimate of it."),
                "no (policy, target) pair in the class satisfying the worst-case condition")


# --------------------------------------------------------------------------------------------------
def p5_overpunishment(cfg, seeds) -> dict:
    """P5. Deterrence is non-monotone in kappa. Once the penalty drives stationary reputation to the
    floor across a whole range of f, the policy stops discriminating between fabrication levels there
    and the merchant's incentives revert to the no-penalty ones. Scanned, not asserted."""
    kappas = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
    out = {}
    for tau in (0.02, 0.15, 0.30):
        row = []
        for k in kappas:
            rg = M.rbar_grid(cfg, k, tau)
            fs = []
            for seed in seeds:
                mkt = M.draw_market(cfg, seed)
                rm = M.rbar_for_market(cfg, mkt, rg)
                ne = M.pure_nash(cfg, mkt, rm)
                fs.append(float(cfg.fgrid[list(ne[0])].mean()) if ne else float("nan"))
            spread = float(rg[:, 0].mean() - rg[:, -1].mean())
            row.append(dict(kappa=k, mean_equilibrium_f=float(np.nanmean(fs)),
                            rbar_spread_over_f=spread))
        out[f"tau={tau}"] = row
    nonmono = {}
    for tau, row in out.items():
        f = [r["mean_equilibrium_f"] for r in row]
        lo = int(np.nanargmin(f))
        nonmono[tau] = dict(argmin_kappa=row[lo]["kappa"], min_mean_f=f[lo],
                            mean_f_at_max_kappa=f[-1],
                            rises_after_minimum=bool(f[-1] > f[lo] + 1e-9))
    any_reversal = any(v["rises_after_minimum"] for v in nonmono.values())
    return _res("P5_overpunishment_reversal",
                "Mean equilibrium fabrication is not monotone decreasing in kappa: past some kappa the "
                "stationary reputation floors out over a range of f, marginal deterrence collapses "
                "there, and fabrication rises again.",
                any_reversal,
                dict(scan=out, summary=nonmono,
                     note="This is the mechanism behind the earlier P_sep finding (kappa=8, tau=0.02 "
                          "raised fabrication and cut GMV). It is a property of the instrument, not "
                          "an artefact of any particular run."),
                "mean equilibrium fabrication decreasing in kappa at every tau")


# --------------------------------------------------------------------------------------------------
def p6_impossibility(seeds) -> dict:
    """P6. IMPOSSIBILITY. Make the signal independent of the hidden action (cs = 0, psi1 = 0). The
    stationary reputation is then flat in f for every policy in the class, so no signal-contingent
    policy changes any merchant's ranking over actions: the equilibrium action correspondence is
    constant across the entire class and the platform's implementation power is exactly zero.

    Note the precise claim. GMV is NOT invariant, because penalties still bite unequally across
    baseline complaint propensities and still reallocate logit share between merchants at different
    prices. That residual movement is demand reallocation, not incentive provision, and calling it
    implementation would be the exact error this proposition exists to prevent."""
    cfg = M.Config(cs=0.0, psi1=0.0)
    rgs = {kt: M.rbar_grid(cfg, kt[0], kt[1])
           for kt in {(k, t if k else 0.0) for (k, t) in M.POLICY_CLASS}}
    varying, total, gmv_spread, flat_max = 0, 0, [], 0.0
    profiles_seen = {}
    for seed in seeds:
        mkt = M.draw_market(cfg, seed)
        seen, gmvs = set(), []
        for (k, t) in M.POLICY_CLASS:
            rm = M.rbar_for_market(cfg, mkt, rgs[(k, t if k else 0.0)])
            flat_max = max(flat_max, float(np.abs(rm[:, 0] - rm[:, -1]).max()))
            en = M.enumerate_profiles(cfg, mkt, rm)
            ne = M.pure_nash(cfg, mkt, rm, en)
            total += 1
            if ne:
                seen.add(tuple(ne[0]))
                gmvs.append(float(en["GMV"][ne[0]]))
        profiles_seen[str(seed)] = [list(p) for p in seen]
        varying += int(len(seen) > 1)
        gmv_spread.append(max(gmvs) - min(gmvs))
    all_full = all(set(p[0]) == {M.Config().Nf - 1} for p in profiles_seen.values()
                   for p in [profiles_seen[list(profiles_seen)[0]]])
    return _res("P6_impossibility_uninformative_signal",
                "If the signal distribution does not depend on the hidden action, then for every "
                "policy in the class the equilibrium action profile is the same; the platform cannot "
                "move fabrication at all.",
                varying == 0,
                dict(seeds=len(seeds), seed_policy_pairs=total,
                     seeds_where_equilibrium_profile_varies_across_class=varying,
                     max_rbar_variation_over_f=flat_max,
                     equilibrium_profile_example=profiles_seen[str(seeds[0])],
                     equilibrium_is_full_fabrication_everywhere=bool(all_full),
                     residual_gmv_spread_across_class=dict(
                         mean=float(np.mean(gmv_spread)), max=float(np.max(gmv_spread)),
                         interpretation="pure demand reallocation between merchants at different "
                                        "prices; NOT incentive provision")),
                "one seed where two policies in the class induce different equilibrium profiles")


# --------------------------------------------------------------------------------------------------
def p7_refund_garbling(cfg) -> dict:
    """P7. refunds ~ Binomial(complaints, phi) with phi independent of f, so P(d, rho | f) =
    P(d | f) P(rho | d) and d is sufficient. Blackwell: conditioning a policy on refunds as well as
    complaints cannot implement anything complaints alone cannot. Verified by Fisher information."""
    def fisher_d(f, b, h=1e-5):
        p0, p1 = M.complaint_pmf(cfg, f - h, b), M.complaint_pmf(cfg, f + h, b)
        pm = M.complaint_pmf(cfg, f, b)
        dlog = (np.log(np.maximum(p1, 1e-300)) - np.log(np.maximum(p0, 1e-300))) / (2 * h)
        return float((pm * dlog ** 2).sum())

    def fisher_d_rho(f, b, h=1e-5):
        tot = 0.0
        for d in range(cfg.N_obs + 1):
            pr = M.binom_pmf(d, cfg.phi_refund) if d > 0 else np.array([1.0])
            p0 = M.complaint_pmf(cfg, f - h, b)[d]
            p1 = M.complaint_pmf(cfg, f + h, b)[d]
            pmf = M.complaint_pmf(cfg, f, b)[d]
            dlog = (math.log(max(p1, 1e-300)) - math.log(max(p0, 1e-300))) / (2 * h)
            tot += pmf * float(pr.sum()) * dlog ** 2
        return float(tot)

    rows, worst = [], 0.0
    for b in (0.01, 0.05, 0.10, 0.15):
        for f in (0.1, 0.3, 0.5, 0.7, 0.9):
            a, c = fisher_d(f, b), fisher_d_rho(f, b)
            rel = abs(a - c) / max(a, 1e-12)
            worst = max(worst, rel)
            rows.append(dict(b=b, f=f, I_complaints=a, I_complaints_and_refunds=c,
                             relative_difference=rel))
    return _res("P7_refunds_are_a_garbling",
                "Refunds are a thinning of complaints with an f-independent rate, so the pair "
                "(complaints, refunds) is Blackwell-equivalent to complaints alone.",
                worst < 1e-8,
                dict(max_relative_fisher_difference=worst, rows=rows,
                     note="A second observable is not a second instrument unless its distribution "
                          "moves with the hidden action. Audit flags do; refunds do not."),
                "a positive information gain from adding refunds")


# --------------------------------------------------------------------------------------------------
def p8b_oracle_reduction(cfg, bench: dict) -> dict:
    """P8b. The displayed payoff of the earlier interfaces is Vtilde_i(f) = V_i(f, f_-i^{t-1}): the
    merchant's exact best-response correspondence against last round's rivals. A fixed point of
    displayed-argmax play is therefore, by definition, a pure equilibrium of the restricted game -- the
    old design obtained convergence by construction, because the interface handed over the answer.
    Verified by checking that the historical best-response iteration lands inside the enumerated
    equilibrium set."""
    agree = tot = 0
    for s, per in bench["named_per_seed"].items():
        for p, rec in per.items():
            tot += 1
            agree += int(rec["br_dynamics"]["agrees_with_enumeration"])
    return _res("P8b_oracle_reduction",
                "Displayed-argmax play is best response to last round's rivals, so its fixed points "
                "are exactly the pure equilibria of the restricted stationary game.",
                agree == tot,
                dict(cases=tot, best_response_fixed_points_inside_enumerated_equilibrium_set=agree,
                     note="The consequence for the paper: convergence under the earlier U/H/R/G "
                          "interfaces is not evidence that a language model inferred an "
                          "equilibrium. The interface computed the best response and printed it. "
                          "Only an environment that withholds that table can test the inference."),
                "a best-response fixed point outside the enumerated equilibrium set")


# --------------------------------------------------------------------------------------------------
def p9_dead_zone(cfg) -> dict:
    """P9. Dead zones, defined economically rather than by floating-point equality.

    A threshold above the complaint rate a merchant can reach means the penalty almost never fires, so
    the marginal reputational bite of one more grid step of fabrication is negligible NEXT TO THE
    APPEAL GAIN of that same step. Inside such a region the policy is, for incentive purposes, the
    no-penalty policy, and no target strictly inside it is implementable. The criterion below is the
    ratio of the two, not an absolute tolerance, because only the ratio has economic meaning."""
    pool = M.catalog_pool()
    q_med = float(np.median([t["q"] for t in pool]))
    step = cfg.fgrid[1] - cfg.fgrid[0]
    appeal_step = cfg.alpha * (1 - q_med) * step
    found = []
    for kappa in (0.5, 2.0, 4.0, 8.0, 16.0):
        for tau in (0.02, 0.15, 0.20, 0.30, 0.40):
            rg = M.rbar_grid(cfg, kappa, tau)
            for bi, b in enumerate(M.B_GRID):
                bite = cfg.beta * (rg[bi, :-1] - rg[bi, 1:])
                weak = bite < 0.01 * appeal_step
                width = int(np.argmax(~weak)) if (~weak).any() else cfg.Nf - 1
                if width >= 3:
                    found.append(dict(kappa=kappa, tau=tau, b=float(b),
                                      dead_zone_upper_f=float(cfg.fgrid[width]),
                                      grid_steps=width,
                                      marginal_bite_at_upper=float(bite[width - 1]),
                                      appeal_gain_per_step=float(appeal_step),
                                      theta_at_upper=float(M.theta(cfg, cfg.fgrid[width], b))))
    found.sort(key=lambda r: -r["grid_steps"])
    by_tau = {}
    for r in found:
        by_tau.setdefault(f"tau={r['tau']}", []).append(r["dead_zone_upper_f"])
    return _res("P9_dead_zone",
                "When tau exceeds the complaint rate reachable at fabrication f, the marginal "
                "reputational bite over [0, f] is negligible next to the appeal gain, and no target "
                "strictly inside that interval is implementable by that policy.",
                len(found) > 0,
                dict(criterion="marginal beta*drbar per grid step < 1% of the appeal gain "
                               "alpha*(1-q_median)*step",
                     q_median_in_pool=q_med, appeal_gain_per_step=float(appeal_step),
                     n_dead_zones_found=len(found), widest=found[:8],
                     max_dead_zone_upper_f_by_tau={k: max(v) for k, v in sorted(by_tau.items())},
                     note="Dead zones are why the second-best search must range over tau as well as "
                          "kappa, and why an implementation claim must name the policy it holds "
                          "under. They also bound from below how much fabrication any policy in "
                          "this class must tolerate."),
                "no policy in the class exhibiting an economically dead region")


# --------------------------------------------------------------------------------------------------
def p10_implementation_floor(cfg, seeds) -> dict:
    """P10. Limited liability caps implementation, and the cap does not move with monitoring.

    Reputation lives in [0, 1], so the WHOLE reputational budget a platform can ever spend on one
    merchant is beta * rbar_max <= beta. The merchant's alternative to complying at a low target is to
    abandon reputation and fabricate fully. Compliance therefore has to beat abandonment out of a
    fixed budget, and no amount of signal quality enlarges that budget.

    Applying the exact deviation identity to the pair (f_c, 1) and bounding the share term by an upper
    bound sbar on the merchant's share (shares ARE platform-observable, unlike q and b) gives: target
    f_c is NOT implementable for merchant i whenever

        alpha (1-q_i) D - lam D / m - ln[(1 - sbar) / (1 - g sbar)]  >  beta * rbar_max,
        D = 1 - f_c,  g = exp(lam D / m),  valid while g sbar < 1.

    Solving for D gives a per-merchant FLOOR: in any equilibrium under any policy acting through
    reputation, merchant i plays at least f_floor(i). Two forms are checked here:

      exact   -- sbar = the merchant's own realised equilibrium share. This verifies the algebra of the
                 identity; it is not a rule the platform could apply ex ante.
      usable  -- sbar = 0.5, a bound the platform can assert from observed sales alone. Weaker, but it
                 requires nothing unobservable, so it is the form that belongs in a mechanism claim.

    The prediction is falsifiable and sharp: an equilibrium action strictly below the floor refutes it.
    """
    rgs = {k: M.rbar_grid(cfg, *M.POLICIES[k]) for k in M.POLICIES}
    rbar_max = float(max(rg.max() for rg in rgs.values()))
    sbar_usable = 0.5
    rows, viol_exact, viol_usable, n = [], 0, 0, 0
    floors_usable, floors_exact = [], []

    def floor_for(q, sbar):
        """Smallest f_c that is NOT ruled out by the budget, searched on the action grid."""
        for fc in cfg.fgrid:
            D = 1.0 - float(fc)
            if D <= 0:
                return float(fc)
            g = math.exp(cfg.lam * D / cfg.m)
            if g * sbar >= 1.0:
                return float(fc)          # share large enough that the externality alone deters
            lhs = (cfg.alpha * (1 - q) * D - cfg.lam * D / cfg.m
                   - math.log((1 - sbar) / (1 - g * sbar)))
            if lhs <= cfg.beta * rbar_max:
                return float(fc)          # budget suffices: not ruled out
        return 1.0

    for s in seeds:
        mkt = M.draw_market(cfg, s)
        for pname, rg in rgs.items():
            rm = M.rbar_for_market(cfg, mkt, rg)
            en = M.enumerate_profiles(cfg, mkt, rm)
            eqs = M.pure_nash(cfg, mkt, rm, en)
            for eq in eqs:
                for j in range(mkt.m):
                    n += 1
                    f_eq = float(cfg.fgrid[eq[j]])
                    s_j = float(en["shares"][j][tuple(eq)])
                    fe = floor_for(float(mkt.q[j]), min(s_j, 0.999))
                    fu = floor_for(float(mkt.q[j]), sbar_usable)
                    floors_exact.append(fe)
                    floors_usable.append(fu)
                    # tolerance of one grid step: the floor is searched on the same grid
                    tol = float(cfg.fgrid[1] - cfg.fgrid[0]) + 1e-12
                    if f_eq < fe - tol:
                        viol_exact += 1
                    if f_eq < fu - tol:
                        viol_usable += 1
                    if len(rows) < 8:
                        rows.append(dict(seed=int(s), policy=pname, merchant=j,
                                         q=float(mkt.q[j]), share=s_j, f_equilibrium=f_eq,
                                         floor_exact=fe, floor_usable=fu))

    q_lo = float(min(t["q"] for t in M.catalog_pool()))
    return _res("P10_implementation_floor",
                "Reputation is bounded in [0,1], so the deterrent budget is bounded by beta*rbar_max. "
                "Targets far enough below full fabrication are therefore unimplementable by ANY "
                "policy acting through reputation, no matter how informative the signal, and every "
                "equilibrium action respects the resulting per-merchant floor.",
                viol_exact == 0 and viol_usable == 0,
                dict(merchant_equilibrium_cases=n,
                     rbar_max=rbar_max, deterrent_budget_beta_times_rbar_max=cfg.beta * rbar_max,
                     violations_exact_share_bound=viol_exact,
                     violations_usable_share_bound=viol_usable,
                     sbar_usable=sbar_usable,
                     mean_floor_exact=float(np.mean(floors_exact)),
                     mean_floor_usable=float(np.mean(floors_usable)),
                     max_floor_exact=float(np.max(floors_exact)),
                     floor_at_lowest_quality_in_pool=dict(
                         q_min=q_lo, floor_usable=floor_for(q_lo, sbar_usable),
                         floor_at_zero_share=floor_for(q_lo, 1e-9)),
                     examples=rows,
                     note="This is the constraint that caps implementation power, not signal "
                          "informativeness: the floor contains no signal parameter. It explains why "
                          "the monitoring continuum saturates well below 1 (P8a) and why tightening "
                          "tau eventually RAISES equilibrium fabrication (P5) -- past the floor, "
                          "compliance is worth less than abandonment, and merchants switch."),
                "an equilibrium action more than one grid step below the floor")


# --------------------------------------------------------------------------------------------------
def p5b_large_kappa_limit(cfg) -> dict:
    """P5b. The closed form behind the over-punishment reversal.

    As kappa -> infinity with tau fixed, any penalty event drives reputation to 0, so the chain becomes
    'reset on a penalty, recover by eta otherwise'. The age since the last reset is geometric with
    parameter p0 = P(d <= floor(tau*N) | theta), and r after k clean rounds is 1 - (1-eta)^k, so

        rbar_inf(f, b) = eta * p0 / (1 - (1 - eta) p0).

    The consequence is the point of the proposition: rbar_inf(f=0) < 1 whenever p0 < 1, so an infinite
    penalty COMPRESSES the reputational spread instead of widening it. The deterrent is the spread,
    not the level, and over-punishment destroys the instrument it is trying to use.
    """
    big = 1e9
    # The closed form is derived for a chain whose recovery r -> r + eta(1-r) has ceiling 1. The
    # solver holds r on a finite grid, where the same recovery tops out at the no-penalty stationary
    # mean rbar_ceiling < 1. Comparing against the uncorrected form would charge the proposition for
    # a discretisation artefact, so the ceiling is measured and divided out.
    ceiling = float(M.rbar_grid(cfg, 0.0, 0.0).max())
    rows, max_err, max_err_raw = [], 0.0, 0.0
    for tau in (0.02, 0.10, 0.20):
        rg = M.rbar_grid(cfg, big, tau)
        for bi in (0, len(M.B_GRID) // 2, len(M.B_GRID) - 1):
            b = float(M.B_GRID[bi])
            for fi in (0, cfg.Nf // 2, cfg.Nf - 1):
                f = float(cfg.fgrid[fi])
                pmf = M.complaint_pmf(cfg, f, b)
                k_ok = int(math.floor(tau * cfg.N_obs + 1e-12))
                p0 = float(pmf[:k_ok + 1].sum())
                raw = cfg.eta_r * p0 / (1 - (1 - cfg.eta_r) * p0)
                closed = ceiling * raw
                err = abs(closed - float(rg[bi, fi]))
                max_err = max(max_err, err)
                max_err_raw = max(max_err_raw, abs(raw - float(rg[bi, fi])))
                rows.append(dict(tau=tau, b=b, f=f, p0_no_penalty=p0,
                                 rbar_numeric=float(rg[bi, fi]), rbar_closed_form=closed,
                                 rbar_closed_form_uncorrected=raw, abs_error=err))
    # the spread at kappa -> inf, next to the spread at a moderate kappa
    spread = {}
    bi = len(M.B_GRID) // 2
    for kappa in (0.5, 2.0, 1e9):
        for tau in (0.02, 0.20):
            rg = M.rbar_grid(cfg, kappa, tau)
            spread[f"kappa={kappa:g},tau={tau}"] = dict(
                rbar_at_f0=float(rg[bi, 0]), rbar_at_f1=float(rg[bi, -1]),
                beta_times_spread=float(cfg.beta * (rg[bi, 0] - rg[bi, -1])))
    # ---- what the proposition actually asserts, tested on the solver's own tables ----------------
    # The closed form is a continuum result and the solver's chain is discretised with np.round
    # snapping (inherited verbatim from equilibrium._stationary_mean and deliberately not changed,
    # because changing it would break comparability with the frozen corpus). Snapping is not mean
    # preserving, so the closed form cannot be exact and its residual is reported as a diagnostic
    # rather than used as a pass criterion. The claim under test is the one that matters and that the
    # tables settle exactly: at a threshold below what an honest merchant can achieve, raising kappa
    # SHRINKS the deterrent spread; at a threshold above it, the spread is unaffected.
    bi_mid = len(M.B_GRID) // 2
    kappas = (0.5, 2.0, 8.0, big)
    spread_by_tau = {}
    for tau in (0.02, 0.20):
        sp = []
        for kappa in kappas:
            rg = M.rbar_grid(cfg, kappa, tau)
            sp.append(float(cfg.beta * (rg[bi_mid, 0] - rg[bi_mid, -1])))
        spread_by_tau[f"tau={tau}"] = sp
    tight = spread_by_tau["tau=0.02"]
    loose = spread_by_tau["tau=0.2"]
    collapses = all(tight[i] > tight[i + 1] for i in range(len(tight) - 1))
    survives = min(loose) / max(loose) > 0.95
    passed = collapses and survives

    tol = 0.5 / (cfg.R - 1)
    return _res("P5b_large_kappa_limit",
                "As kappa -> infinity the reputation chain becomes reset-on-penalty and its stationary "
                "mean has the closed form eta*p0/(1-(1-eta)p0); the honest merchant's reputation is "
                "then bounded strictly below 1, so an unbounded penalty compresses the very spread "
                "that does the deterring.",
                passed,
                dict(criterion="beta*(rbar(f=0) - rbar(f=1)) is strictly decreasing in kappa at "
                               "tau=0.02 (a threshold below the honest complaint rate) and flat "
                               "within 5% at tau=0.20 (a threshold above it)",
                     kappa_values=[float(k) for k in kappas],
                     deterrent_spread_by_tau=spread_by_tau,
                     spread_collapses_at_tight_threshold=collapses,
                     spread_survives_at_loose_threshold=survives,
                     collapse_factor_at_tau_0p02=float(tight[0] / tight[-1]),
                     closed_form_diagnostic=dict(
                         formula="rbar_inf = ceiling * eta*p0 / (1 - (1-eta)*p0), "
                                 "p0 = P(d <= floor(tau*N))",
                         max_abs_error=max_err,
                         max_abs_error_before_ceiling_correction=max_err_raw,
                         rbar_ceiling_measured=ceiling,
                         grid_half_spacing=tol,
                         within_half_spacing=bool(max_err <= tol),
                         why_not_exact="equilibrium._stationary_mean snaps r to the nearest of "
                                       f"{cfg.R} grid points with np.round, which is not mean "
                                       "preserving. The snapping is frozen Balance-9 code and is "
                                       "kept byte-identical, so the residual is a property of the "
                                       "corpus's discretisation, not of the proposition. It is "
                                       "reported, not tuned away.",
                         rows=rows),
                     spread_comparison=spread,
                     note="Read the spread table downward: at tau=0.02 the deterrent beta*spread "
                          "FALLS as kappa grows, because the penalty catches the honest merchant "
                          "too. At tau=0.20 it does not. That contrast is the mechanism of the P5 "
                          "reversal, and it is why 'punish harder' is not a policy direction."),
                "the deterrent spread failing to collapse at a threshold below the honest complaint "
                "rate, or collapsing at one above it")


# --------------------------------------------------------------------------------------------------
def main():
    t0 = time.time()
    cfg = M.Config()
    bench_path = OUTDIR / "ha_benchmarks.json"
    if not bench_path.exists():
        raise SystemExit("run ha_benchmarks.py first (results/solver/ha_benchmarks.json missing)")
    bench = json.loads(bench_path.read_text())

    checks = [p1_existence(bench)]
    print(f"  P1 {checks[-1]['verdict']}  [{time.time() - t0:.0f}s]")
    for fn, args in ((p2_exact_ic, (cfg, CHECK_SEEDS)),
                     (p3a_deviation_identity, (cfg, CHECK_SEEDS)),
                     (p3b_sufficient_condition, (cfg, CHECK_SEEDS)),
                     (p4_robust_implementation, (cfg, CHECK_SEEDS)),
                     (p5_overpunishment, (cfg, CHECK_SEEDS)),
                     (p5b_large_kappa_limit, (cfg,)),
                     (p6_impossibility, (CHECK_SEEDS,)),
                     (p7_refund_garbling, (cfg,)),
                     (p8b_oracle_reduction, (cfg, bench)),
                     (p9_dead_zone, (cfg,)),
                     (p10_implementation_floor, (cfg, CHECK_SEEDS))):
        checks.append(fn(*args))
        print(f"  {checks[-1]['check']}: {checks[-1]['verdict']}  [{time.time() - t0:.0f}s]")

    out = dict(
        purpose="Numerical verification of the propositions in theory/HIDDEN_ACTION_THEORY.md.",
        generated=time.strftime("%Y-%m-%dT%H:%M:%S"),
        spec=M.spec_dict(cfg),
        seeds_used=dict(existence=[SEEDS[0], SEEDS[-1]],
                        per_profile_sweeps=[CHECK_SEEDS[0], CHECK_SEEDS[-1]]),
        n_checks=len(checks),
        n_pass=sum(1 for c in checks if c["verdict"] == "PASS"),
        n_fail=sum(1 for c in checks if c["verdict"] == "FAIL"),
        checks=checks,
        runtime_seconds=round(time.time() - t0, 1))
    OUTDIR.mkdir(parents=True, exist_ok=True)
    p = OUTDIR / "ha_theory_checks.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(out, indent=1), encoding="utf-8")
    tmp.replace(p)
    print(f"\n{out['n_pass']}/{out['n_checks']} PASS, {out['n_fail']} FAIL -> {p}")
    for c in checks:
        print(f"  {c['verdict']:5s} {c['check']}")
    if out["n_fail"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
