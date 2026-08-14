"""ha_benchmarks.py -- the GMV yardsticks, computed exactly.

Three quantities, three different informational worlds, and they must never be confused:

  G^FB(seed)      FIRST BEST. Same instruments as the second best, but the incentive constraint is
                  dropped: the platform picks a policy in P AND dictates the joint action profile.
                  Exact, by enumeration of all 21^4 = 194,481 profiles under every policy in P.

  G^SB_P(seed)    POLICY-CLASS SECOND BEST. Same instruments, but the profile must be an EQUILIBRIUM
                  of the game the policy induces. It is NOT the global second best: nothing here rules
                  out a cleverer mechanism outside P. The name carries the caveat on purpose.

  G^NP(seed)      The do-nothing benchmark: no penalty at all, merchants at equilibrium.

The two benchmarks share an instrument set on purpose. G^SB_P / G^FB then isolates exactly one thing --
the cost of not observing the hidden action -- instead of confounding it with a change in what the
platform is allowed to do.

Why the first best is NOT simply "the best profile with no penalty". A penalty lowers stationary
reputation, but GMV is not monotone in reputation: reputation enters a logit share, so raising one
merchant's reputation takes demand from the others as well as from the outside option, and the
merchants carry different prices. A penalty that bites unequally across baseline complaint propensities
therefore REALLOCATES demand, and on 21 of 60 seeds the best achievable GMV under some penalty strictly
exceeds the best achievable GMV with no penalty at all, with the incentive problem switched off
entirely. That is measured below (`reputation_reallocation`), not assumed away.

Equilibrium selection is reported, never assumed. When a policy induces several pure equilibria we
report the best-case and worst-case selection separately, so a reader can see exactly how much of the
"second best" depends on merchants coordinating on the platform's favourite equilibrium.

Pure numpy, no API. This is one of the computations that must be finished before any money is spent.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ha_model as M  # noqa: E402

OUTDIR = HERE.parent / "results" / "solver"
MAIN_SEEDS = list(range(70000, 70060))


# --------------------------------------------------------------------------------------------------
def _rbar_cache():
    cache = {}

    def get(cfg, kappa, tau, kappa_a=0.0, tau_a=0.0):
        # a zero coefficient makes its threshold irrelevant; collapse those keys so the class is not
        # searched several times over the same table
        key = (float(kappa), float(tau) if kappa else 0.0,
               float(kappa_a), float(tau_a) if kappa_a else 0.0)
        if key not in cache:
            cache[key] = M.rbar_grid(cfg, key[0], key[1], key[2], key[3])
        return cache[key]

    return get, cache


def eval_policy(cfg, mkt, rbar_m):
    """Everything a single (market, policy) pair yields. Exact, no search heuristics."""
    en = M.enumerate_profiles(cfg, mkt, rbar_m)
    ne = M.pure_nash(cfg, mkt, rbar_m, en)
    gmv = en["GMV"]
    rec = dict(n_pure_nash=len(ne),
               gmv_max_profile=float(gmv.max()),
               argmax_profile=[int(v) for v in np.unravel_index(int(np.argmax(gmv)), gmv.shape)])
    if ne:
        vals = np.array([float(gmv[p]) for p in ne])
        lo, hi = int(np.argmin(vals)), int(np.argmax(vals))
        rec.update(
            gmv_best_eq=float(vals[hi]), gmv_worst_eq=float(vals[lo]),
            eq_best=[int(v) for v in ne[hi]], eq_worst=[int(v) for v in ne[lo]],
            mean_f_best_eq=float(cfg.fgrid[list(ne[hi])].mean()),
            mean_f_worst_eq=float(cfg.fgrid[list(ne[lo])].mean()),
            all_eq=[[int(v) for v in p] for p in ne[:32]])
    else:
        rec.update(gmv_best_eq=None, gmv_worst_eq=None, eq_best=None, eq_worst=None,
                   mean_f_best_eq=None, mean_f_worst_eq=None, all_eq=[])
    return rec, en, ne


# --------------------------------------------------------------------------------------------------
def run(seeds, extended: bool, out_name: str):
    cfg = M.Config()
    get_rbar, cache = _rbar_cache()
    t0 = time.time()

    base_class = list(M.POLICY_CLASS)
    if extended:
        audit_settings = [(0.0, 0.0)] + [(ka, ta) for ka in M.KAPPA_A_GRID if ka > 0
                                         for ta in M.TAU_A_GRID]
    else:
        audit_settings = [(0.0, 0.0)]
    full_class = [(k, t, ka, ta) for (k, t) in base_class for (ka, ta) in audit_settings]
    print(f"policy class: {len(full_class)} settings "
          f"({len(base_class)} complaint x {len(audit_settings)} audit), {len(seeds)} seeds")

    # warm the reputation tables once; this is the only slow part
    for (k, t, ka, ta) in full_class:
        get_rbar(cfg, k, t, ka, ta)
    print(f"reputation tables: {len(cache)} distinct, {time.time() - t0:.1f}s")

    per_seed, named = {}, {}
    for si, seed in enumerate(seeds):
        mkt = M.draw_market(cfg, seed)
        rg0 = get_rbar(cfg, 0.0, 0.0)
        rm0 = M.rbar_for_market(cfg, mkt, rg0)
        np_rec, en0, _ = eval_policy(cfg, mkt, rm0)

        # FIRST BEST over the same instrument set: best profile under the best policy in P, with the
        # incentive constraint dropped. The no-penalty value is kept separately because the two differ,
        # and the difference is a result rather than an error.
        g_fb_nopen = np_rec["gmv_max_profile"]
        g_fb = g_fb_nopen
        fb_profile = np_rec["argmax_profile"]
        fb_policy = [0.0, 0.0, 0.0, 0.0]
        n_multi_eq = 0

        best_sb = dict(gmv=-1.0, policy=None)
        worst_sel = dict(gmv=-1.0, policy=None)
        rows = []
        for (k, t, ka, ta) in full_class:
            rm = M.rbar_for_market(cfg, mkt, get_rbar(cfg, k, t, ka, ta))
            rec, en, ne = eval_policy(cfg, mkt, rm)
            n_multi_eq += int(rec["n_pure_nash"] > 1)
            if rec["gmv_max_profile"] > g_fb + 1e-12:
                g_fb = rec["gmv_max_profile"]
                fb_profile = rec["argmax_profile"]
                fb_policy = [k, t, ka, ta]
            rows.append(dict(kappa=k, tau=t, kappa_a=ka, tau_a=ta, **{
                kk: rec[kk] for kk in ("n_pure_nash", "gmv_best_eq", "gmv_worst_eq",
                                       "eq_best", "eq_worst", "mean_f_best_eq", "mean_f_worst_eq")}))
            if rec["gmv_best_eq"] is not None and rec["gmv_best_eq"] > best_sb["gmv"]:
                best_sb = dict(gmv=rec["gmv_best_eq"], policy=[k, t, ka, ta],
                               profile=rec["eq_best"], mean_f=rec["mean_f_best_eq"],
                               n_eq=rec["n_pure_nash"])
            # pessimistic selection: the platform is judged by the WORST equilibrium its policy admits
            if rec["gmv_worst_eq"] is not None and rec["gmv_worst_eq"] > worst_sel["gmv"]:
                worst_sel = dict(gmv=rec["gmv_worst_eq"], policy=[k, t, ka, ta],
                                 profile=rec["eq_worst"], mean_f=rec["mean_f_worst_eq"],
                                 n_eq=rec["n_pure_nash"])

        # the two named policies the LLM arms actually run under
        nm = {}
        for pname, (k, t) in M.POLICIES.items():
            rm = M.rbar_for_market(cfg, mkt, get_rbar(cfg, k, t))
            rec, _, _ = eval_policy(cfg, mkt, rm)
            fidx, path, status = M.best_response_dynamics(cfg, mkt, rm)
            rec["br_dynamics"] = dict(profile=[int(v) for v in fidx], status=status,
                                      steps=len(path) - 1,
                                      agrees_with_enumeration=bool(
                                          tuple(int(v) for v in fidx) in
                                          {tuple(p) for p in rec["all_eq"]}))
            nm[pname] = rec
        named[str(seed)] = nm

        per_seed[str(seed)] = dict(
            seed=seed,
            types=dict(q=[float(v) for v in mkt.q], p=[float(v) for v in mkt.p],
                       b=[float(v) for v in mkt.b], bidx=[int(v) for v in mkt.bidx]),
            G_FB=g_fb, FB_profile=fb_profile, FB_policy=fb_policy,
            FB_mean_f=float(cfg.fgrid[fb_profile].mean()),
            G_FB_no_penalty=g_fb_nopen,
            reputation_reallocation_gain=float(g_fb - g_fb_nopen),
            G_NP=np_rec["gmv_best_eq"], NP_eq=np_rec["eq_best"],
            NP_n_eq=np_rec["n_pure_nash"],
            NP_mean_f=np_rec["mean_f_best_eq"],
            G_SB_P_optimistic=best_sb["gmv"], SB_policy_optimistic=best_sb["policy"],
            SB_profile_optimistic=best_sb.get("profile"),
            SB_mean_f_optimistic=best_sb.get("mean_f"),
            G_SB_P_pessimistic=worst_sel["gmv"], SB_policy_pessimistic=worst_sel["policy"],
            SB_profile_pessimistic=worst_sel.get("profile"),
            n_policies_with_multiple_pure_nash=n_multi_eq,
            policy_rows=rows)
        if (si + 1) % 10 == 0 or si == 0:
            print(f"  seed {seed} ({si + 1}/{len(seeds)})  G_FB={g_fb:.4f}  "
                  f"G_NP={np_rec['gmv_best_eq']:.4f}  G_SB={best_sb['gmv']:.4f}  "
                  f"[{time.time() - t0:.0f}s]")

    # ---------------- aggregates ----------------
    fb = np.array([per_seed[str(s)]["G_FB"] for s in seeds])
    nprec = np.array([per_seed[str(s)]["G_NP"] for s in seeds])
    sb_o = np.array([per_seed[str(s)]["G_SB_P_optimistic"] for s in seeds])
    sb_p = np.array([per_seed[str(s)]["G_SB_P_pessimistic"] for s in seeds])

    def stat(x):
        return dict(mean=float(x.mean()), sd=float(x.std(ddof=1)), min=float(x.min()),
                    max=float(x.max()), median=float(np.median(x)))

    # ---------------- the honest second best: ONE policy committed for the whole market ----------
    #
    # G_SB_P above is chosen per seed. That silently grants the platform something it does not have:
    # the ability to read a market's private types (q, b) and pick the penalty that suits them. The
    # per-seed maximum is therefore an upper bound on what a real platform can reach, not a target it
    # can aim at. The number a platform can actually commit to is the best SINGLE (kappa, tau) held
    # fixed across every market -- computed here, reported separately, and used as the denominator
    # for any claim about how close the LLM arms come to what the platform could have done.
    #
    # The gap between the two is reported as `tuning_premium`. It is a measure of how much of the
    # policy-class second best is an artefact of conditioning on unobservables.
    class_keys = [(float(k), float(t), float(ka), float(ta)) for (k, t, ka, ta) in full_class]
    gmv_best = np.array([[r["gmv_best_eq"] for r in per_seed[str(s)]["policy_rows"]] for s in seeds])
    gmv_worst = np.array([[r["gmv_worst_eq"] for r in per_seed[str(s)]["policy_rows"]]
                          for s in seeds])
    mf_worst = np.array([[r["mean_f_worst_eq"] for r in per_seed[str(s)]["policy_rows"]]
                         for s in seeds])
    # pessimistic selection is the one a platform may rely on: it is the guarantee, not the hope
    i_uni_p = int(np.argmax(gmv_worst.mean(axis=0)))
    i_uni_o = int(np.argmax(gmv_best.mean(axis=0)))
    uniform = dict(
        note="Best single policy held fixed across all seeds. Unlike G_SB_P this conditions on no "
             "unobservable, so it is the benchmark a platform could actually commit to.",
        pessimistic=dict(policy=list(class_keys[i_uni_p]),
                         G=float(gmv_worst[:, i_uni_p].mean()),
                         G_per_seed=[float(v) for v in gmv_worst[:, i_uni_p]],
                         mean_f=float(mf_worst[:, i_uni_p].mean()),
                         ratio_over_FB=float(gmv_worst[:, i_uni_p].mean() / fb.mean())),
        optimistic=dict(policy=list(class_keys[i_uni_o]),
                        G=float(gmv_best[:, i_uni_o].mean()),
                        ratio_over_FB=float(gmv_best[:, i_uni_o].mean() / fb.mean())),
        tuning_premium=dict(
            absolute=float(sb_p.mean() - gmv_worst[:, i_uni_p].mean()),
            fraction_of_per_seed_SB=float(
                (sb_p.mean() - gmv_worst[:, i_uni_p].mean()) / sb_p.mean()),
            interpretation="the share of the per-seed policy-class second best that is unavailable "
                           "to a platform which cannot observe q and b"),
        distinct_per_seed_optima=len({tuple(per_seed[str(s)]["SB_policy_optimistic"][:2])
                                      for s in seeds}))

    # ---- why the gap exists: instrument cost vs incentive gap -----------------------------------
    #
    # Two very different things could keep the second best below the first best, and policy advice
    # differs completely depending on which dominates:
    #
    #   instrument cost -- the penalty destroys reputation even when merchants comply, because a
    #                      threshold near the mean signal is crossed by noise. Fix: better signals.
    #   incentive gap   -- merchants will not play the first-best profile at all, because complying
    #                      is worth less to them than abandoning reputation. Better signals do not
    #                      help; this is a bound on punishment, not on information (Proposition 10).
    #
    # Separated by evaluating the uniform second-best policy at the FIRST-BEST profile: that holds the
    # instrument fixed and switches the behaviour off, so the residual is the instrument's own cost.
    uni_key = class_keys[i_uni_p]
    rg_uni = get_rbar(cfg, uni_key[0], uni_key[1], uni_key[2], uni_key[3])
    comply = []
    for s in seeds:
        mkt = M.draw_market(cfg, s)   # deterministic in the seed; same market as the loop above
        en = M.enumerate_profiles(cfg, mkt, M.rbar_for_market(cfg, mkt, rg_uni))
        comply.append(float(en["GMV"][tuple(per_seed[str(s)]["FB_profile"])]))
    comply_mean = float(np.mean(comply))
    gap = float(fb.mean() - uniform["pessimistic"]["G"])
    uniform["gap_decomposition"] = dict(
        G_FB=float(fb.mean()),
        GMV_at_SB_policy_with_FB_profile=comply_mean,
        G_SB_uniform=uniform["pessimistic"]["G"],
        total_gap=gap,
        instrument_cost=dict(
            value=float(fb.mean() - comply_mean),
            share_of_gap=float((fb.mean() - comply_mean) / gap) if gap > 0 else None,
            meaning="GMV lost because the penalty fires on compliant merchants (false positives)"),
        incentive_gap=dict(
            value=float(comply_mean - uniform["pessimistic"]["G"]),
            share_of_gap=float((comply_mean - uniform["pessimistic"]["G"]) / gap) if gap > 0 else None,
            meaning="GMV lost because merchants will not play the first-best profile at any policy "
                    "in the class; bounded by Proposition 10, not by signal quality"))

    named_agg = {}
    for pname in M.POLICIES:
        g = np.array([named[str(s)][pname]["gmv_best_eq"] for s in seeds])
        gw = np.array([named[str(s)][pname]["gmv_worst_eq"] for s in seeds])
        f = np.array([named[str(s)][pname]["mean_f_best_eq"] for s in seeds])
        neq = np.array([named[str(s)][pname]["n_pure_nash"] for s in seeds])
        agree = np.array([named[str(s)][pname]["br_dynamics"]["agrees_with_enumeration"]
                          for s in seeds])
        named_agg[pname] = dict(
            gmv_best_eq=stat(g), gmv_worst_eq=stat(gw), mean_f=stat(f),
            n_pure_nash=stat(neq.astype(float)),
            seeds_with_no_pure_nash=int((neq == 0).sum()),
            seeds_with_multiple_pure_nash=int((neq > 1).sum()),
            br_dynamics_agrees_with_enumeration=int(agree.sum()),
            ratio_to_FB=stat(g / fb), ratio_to_SB_optimistic=stat(g / sb_o),
            # the two denominators that differ in what the platform is allowed to know
            ratio_to_SB_uniform=float(g.mean() / uniform["pessimistic"]["G"]),
            rank_in_uniform_class=int(
                1 + (gmv_worst.mean(axis=0) >
                     gmv_worst[:, class_keys.index(
                         (float(M.POLICIES[pname][0]), float(M.POLICIES[pname][1]), 0.0, 0.0))
                     ].mean()).sum()),
            n_policies_in_class=len(class_keys))

    out = dict(
        purpose="Exact GMV benchmarks for the hidden-action marketplace: first best, policy-class "
                "second best (optimistic and pessimistic equilibrium selection), no-penalty "
                "baseline, and the two named policies the LLM arms run under.",
        caveat="G_SB_P is the best value attainable within the FINITE policy class enumerated here. "
               "It is a policy-class second best. No claim is made about mechanisms outside the "
               "class, and none of these numbers is a theorem about all mechanisms.",
        generated=time.strftime("%Y-%m-%dT%H:%M:%S"),
        environment=dict(platform=platform.platform(), python=sys.version.split()[0],
                         numpy=np.__version__),
        spec=M.spec_dict(cfg),
        seed_block=dict(name="HA-M1", seeds=[int(seeds[0]), int(seeds[-1])], n=len(seeds)),
        policy_class=dict(kappa_grid=list(M.KAPPA_GRID), tau_grid=list(M.TAU_GRID),
                          kappa_a_grid=list(M.KAPPA_A_GRID) if extended else [0.0],
                          tau_a_grid=list(M.TAU_A_GRID) if extended else [0.0],
                          n_settings=len(full_class), extended=bool(extended)),
        enumeration=dict(joint_profiles_per_seed_policy=int(cfg.Nf ** cfg.m),
                         exact=True, method="full broadcast enumeration + exhaustive "
                                             "unilateral-deviation check"),
        aggregate=dict(
            G_FB=stat(fb), G_NP=stat(nprec),
            G_SB_P_optimistic=stat(sb_o), G_SB_P_pessimistic=stat(sb_p),
            G_SB_uniform=uniform,
            ratio_SB_over_FB_optimistic=stat(sb_o / fb),
            ratio_SB_over_FB_pessimistic=stat(sb_p / fb),
            ratio_NP_over_FB=stat(nprec / fb),
            seeds_where_FB_is_all_honest=int(sum(
                1 for s in seeds if max(per_seed[str(s)]["FB_profile"]) == 0)),
            reputation_reallocation=dict(
                note="A penalty can raise attainable GMV even with the incentive problem switched "
                     "off, by shifting logit share between merchants that carry different prices. "
                     "GMV is therefore not monotone in stationary reputation, and the first best "
                     "must be taken over the policy class as well as over profiles.",
                seeds_where_penalty_beats_no_penalty=int(
                    sum(1 for s in seeds
                        if per_seed[str(s)]["reputation_reallocation_gain"] > 1e-12)),
                gain=stat(np.array([per_seed[str(s)]["reputation_reallocation_gain"]
                                    for s in seeds])),
                relative_gain_max=float(max(
                    per_seed[str(s)]["reputation_reallocation_gain"] / per_seed[str(s)]["G_FB"]
                    for s in seeds))),
            multiplicity=dict(
                seed_policy_pairs=len(seeds) * len(full_class),
                pairs_with_multiple_pure_nash=int(sum(
                    per_seed[str(s)]["n_policies_with_multiple_pure_nash"] for s in seeds)))),
        named_policies=named_agg,
        per_seed=per_seed,
        named_per_seed=named,
        runtime_seconds=round(time.time() - t0, 1))

    OUTDIR.mkdir(parents=True, exist_ok=True)
    path = OUTDIR / out_name
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(out, indent=1), encoding="utf-8")
    tmp.replace(path)
    print(f"\nwrote {path}  ({path.stat().st_size} bytes, {out['runtime_seconds']}s)")
    print(f"  G_FB              mean {out['aggregate']['G_FB']['mean']:.4f}")
    print(f"  G_NP              mean {out['aggregate']['G_NP']['mean']:.4f}"
          f"   ({out['aggregate']['ratio_NP_over_FB']['mean'] * 100:.1f}% of first best)")
    print(f"  G_SB_P optimistic mean {out['aggregate']['G_SB_P_optimistic']['mean']:.4f}"
          f"   ({out['aggregate']['ratio_SB_over_FB_optimistic']['mean'] * 100:.1f}% of first best)")
    print(f"  G_SB_P pessimist. mean {out['aggregate']['G_SB_P_pessimistic']['mean']:.4f}"
          f"   ({out['aggregate']['ratio_SB_over_FB_pessimistic']['mean'] * 100:.1f}% of first best)")
    un = out["aggregate"]["G_SB_uniform"]
    print(f"  G_SB uniform      mean {un['pessimistic']['G']:.4f}"
          f"   ({un['pessimistic']['ratio_over_FB'] * 100:.1f}% of first best)"
          f"  at kappa={un['pessimistic']['policy'][0]:g} tau={un['pessimistic']['policy'][1]:g}")
    print(f"    per-seed tuning premium {un['tuning_premium']['absolute']:.4f} "
          f"({un['tuning_premium']['fraction_of_per_seed_SB'] * 100:.1f}% of the per-seed SB); "
          f"{un['distinct_per_seed_optima']} distinct per-seed optima over {len(seeds)} seeds")
    gd = un["gap_decomposition"]
    print(f"    gap to first best {gd['total_gap']:.4f} = instrument cost "
          f"{gd['instrument_cost']['value']:.4f} ({gd['instrument_cost']['share_of_gap'] * 100:.0f}%)"
          f" + incentive gap {gd['incentive_gap']['value']:.4f} "
          f"({gd['incentive_gap']['share_of_gap'] * 100:.0f}%)")
    ra = out["aggregate"]["reputation_reallocation"]
    print(f"  penalty raises attainable GMV with no incentive problem on "
          f"{ra['seeds_where_penalty_beats_no_penalty']}/{len(seeds)} seeds "
          f"(max {ra['relative_gain_max'] * 100:.2f}% of G_FB)")
    print(f"  (seed, policy) pairs with multiple pure equilibria: "
          f"{out['aggregate']['multiplicity']['pairs_with_multiple_pure_nash']} of "
          f"{out['aggregate']['multiplicity']['seed_policy_pairs']}")
    for pname, a in named_agg.items():
        print(f"  {pname:9s} eq GMV {a['gmv_best_eq']['mean']:.4f}  "
              f"mean f {a['mean_f']['mean']:.4f}  "
              f"of FB {a['ratio_to_FB']['mean'] * 100:.1f}%  "
              f"of SB/seed {a['ratio_to_SB_optimistic']['mean'] * 100:.1f}%  "
              f"of SB/uniform {a['ratio_to_SB_uniform'] * 100:.1f}%  "
              f"rank {a['rank_in_uniform_class']}/{a['n_policies_in_class']}  "
              f"multi-eq seeds {a['seeds_with_multiple_pure_nash']}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="70000-70059")
    ap.add_argument("--extended", action="store_true",
                    help="also search audit-contingent penalties (the two-channel policy class)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    lo, hi = (a.seeds.split("-") + [a.seeds])[:2]
    seeds = list(range(int(lo), int(hi) + 1))
    name = a.out or ("ha_benchmarks_extended.json" if a.extended else "ha_benchmarks.json")
    run(seeds, a.extended, name)


if __name__ == "__main__":
    main()
