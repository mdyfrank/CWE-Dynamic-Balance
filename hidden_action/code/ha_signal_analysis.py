"""ha_signal_analysis.py -- how much the platform's signals actually say about the hidden action.

Everything the platform can do rests on one question: does the observable move with the thing it is
not allowed to see? This module answers it three ways.

  1. INFORMATIVENESS. Fisher information of the complaint count and of the audit-flag count about the
     fabrication level, and the KL divergence between adjacent grid steps. This is the ceiling on any
     inference, the platform's or a merchant's.

  2. CALIBRATION AND ERROR RATES. The published rule punishes when the complaint rate crosses tau.
     Treated as a detector of "this merchant fabricates at least f_ref", that rule has a false-positive
     rate on honest sellers, a recall on fabricators, a precision under the equilibrium action
     distribution, and an AUC. All four are computed exactly from the signal distributions -- no
     simulation, no sampling error -- and all four are reported whether or not they flatter the design.

  3. THE MONITORING CONTINUUM. The whole point of the reformulation is that the earlier oracle
     interfaces sit at one end of a continuum and an uninformative signal sits at the other. Scanning
     (N_obs, cs) traces the path between them: the policy-class second best as a fraction of first
     best should climb toward 1 as monitoring sharpens and fall to the no-penalty value as the signal
     stops depending on f. If it does not, the bridge in the theory document is wrong.

Pure numpy, no API.
"""

from __future__ import annotations

import argparse
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
F_REF = 0.50          # preregistered label boundary: "high fabrication" means f >= 0.50
SCAN_SEEDS = list(range(70000, 70020))


# --------------------------------------------------------------------------------------------------
def informativeness(cfg) -> dict:
    """Fisher information and adjacent-step KL for each channel. The audit channel is included because
    it is the only added observable whose distribution actually moves with f (Proposition 7)."""
    step = float(cfg.fgrid[1] - cfg.fgrid[0])

    def fisher(pmf_fn, f, h=1e-5):
        p0, p1, pm = pmf_fn(f - h), pmf_fn(f + h), pmf_fn(f)
        dlog = (np.log(np.maximum(p1, 1e-300)) - np.log(np.maximum(p0, 1e-300))) / (2 * h)
        return float((pm * dlog ** 2).sum())

    def kl(pmf_fn, f):
        p, q = pmf_fn(f), pmf_fn(min(f + step, 1.0))
        p = np.maximum(p, 1e-300)
        q = np.maximum(q, 1e-300)
        return float((p * np.log(p / q)).sum())

    rows = []
    for b in (0.01, 0.05, 0.10, 0.15):
        for f in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
            cf = (lambda x, b=b: M.complaint_pmf(cfg, min(max(x, 0.0), 1.0), b))
            af = (lambda x: M.audit_pmf(cfg, min(max(x, 0.0), 1.0)))
            rows.append(dict(b=b, f=f,
                             fisher_complaints=fisher(cf, f), fisher_audit=fisher(af, f),
                             kl_complaints_next_step=kl(cf, f), kl_audit_next_step=kl(af, f)))
    fc = np.array([r["fisher_complaints"] for r in rows])
    fa = np.array([r["fisher_audit"] for r in rows])
    return dict(rows=rows, grid_step=step,
                fisher_complaints=dict(mean=float(fc.mean()), min=float(fc.min()),
                                       max=float(fc.max())),
                fisher_audit=dict(mean=float(fa.mean()), min=float(fa.min()), max=float(fa.max())),
                audit_share_of_total_information=float(fa.mean() / (fa.mean() + fc.mean())),
                note="Information is highest near f = 0, where the complaint rate is small and a "
                     "single extra complaint is surprising, and falls as the rate grows. The "
                     "platform therefore discriminates best exactly where merchants fabricate "
                     "least -- the opposite of where it needs to.")


# --------------------------------------------------------------------------------------------------
def detector_quality(cfg, f_dist: np.ndarray, policies) -> dict:
    """Exact error rates of the published rule, read as a detector of 'f >= F_REF'.

    `f_dist` is the action distribution the detector is judged against -- the equilibrium fabrication
    distribution from the solver, not a uniform prior, because precision is only meaningful against
    the population the platform actually faces.
    """
    b_prior = np.full(len(M.B_GRID), 1.0 / len(M.B_GRID))
    hi = cfg.fgrid >= F_REF
    out = []
    for (kappa, tau) in policies:
        # P(punished | f) marginalising over the baseline propensity
        p_pun = np.zeros(cfg.Nf)
        for fi, f in enumerate(cfg.fgrid):
            acc = 0.0
            for bi, b in enumerate(M.B_GRID):
                pmf = M.complaint_pmf(cfg, f, b)
                rate = np.arange(cfg.N_obs + 1) / cfg.N_obs
                acc += b_prior[bi] * float(pmf[rate > tau].sum())
            p_pun[fi] = acc
        w = f_dist / f_dist.sum()
        p_hi = float(w[hi].sum())
        tpr = float((w[hi] * p_pun[hi]).sum() / max(p_hi, 1e-12))
        fpr = float((w[~hi] * p_pun[~hi]).sum() / max(1 - p_hi, 1e-12))
        fpr_honest = float(p_pun[0])
        marg = float((w * p_pun).sum())
        prec = float((w[hi] * p_pun[hi]).sum() / max(marg, 1e-12))
        # AUC of the raw complaint rate as a score for the same label
        score = np.zeros((cfg.Nf, cfg.N_obs + 1))
        for fi, f in enumerate(cfg.fgrid):
            for bi, b in enumerate(M.B_GRID):
                score[fi] += b_prior[bi] * M.complaint_pmf(cfg, f, b)
        pos = (w[hi, None] * score[hi]).sum(0)
        neg = (w[~hi, None] * score[~hi]).sum(0)
        pos = pos / max(pos.sum(), 1e-12)
        neg = neg / max(neg.sum(), 1e-12)
        cum_neg = np.concatenate([[0.0], np.cumsum(neg)[:-1]])
        auc = float((pos * (cum_neg + 0.5 * neg)).sum())
        out.append(dict(kappa=kappa, tau=tau,
                        p_punished_marginal=marg,
                        false_positive_rate_on_fully_honest=fpr_honest,
                        false_positive_rate_below_ref=fpr,
                        recall_at_or_above_ref=tpr,
                        precision=prec,
                        false_negative_rate=1.0 - tpr,
                        auc_complaint_rate=auc))
    return dict(f_ref=F_REF, rows=out,
                note="A false positive here is a merchant punished for noise. It is reported per "
                     "policy because it is the cost side of every deterrence number in this "
                     "package, and it does not appear anywhere in the GMV objective.")


def posterior_calibration(cfg, f_dist: np.ndarray) -> dict:
    """E[f | complaint count] under the equilibrium action distribution and a uniform prior over the
    baseline propensity. This is the sharpest inference ANY observer of the signal can make, so it also
    bounds what a merchant could learn about a rival, and what a signal-derived hint can honestly say."""
    w = f_dist / f_dist.sum()
    b_prior = np.full(len(M.B_GRID), 1.0 / len(M.B_GRID))
    lik = np.zeros((cfg.Nf, cfg.N_obs + 1))
    for fi, f in enumerate(cfg.fgrid):
        for bi, b in enumerate(M.B_GRID):
            lik[fi] += b_prior[bi] * M.complaint_pmf(cfg, f, b)
    joint = w[:, None] * lik
    marg = joint.sum(0)
    post = joint / np.maximum(marg, 1e-300)
    e_f = (cfg.fgrid[:, None] * post).sum(0)
    var_f = ((cfg.fgrid[:, None] - e_f) ** 2 * post).sum(0)
    prior_var = float((w * (cfg.fgrid - (w * cfg.fgrid).sum()) ** 2).sum())
    post_var = float((marg * var_f).sum())
    return dict(prior_mean=float((w * cfg.fgrid).sum()), prior_variance=prior_var,
                expected_posterior_variance=post_var,
                variance_reduction=1.0 - post_var / max(prior_var, 1e-12),
                curve=[dict(complaints=int(d), p_observed=float(marg[d]),
                            posterior_mean_f=float(e_f[d]), posterior_sd_f=float(math.sqrt(var_f[d])))
                       for d in range(0, cfg.N_obs + 1, 2)],
                note="One round of 40 audited transactions leaves substantial residual uncertainty "
                     "about the hidden action. Any risk hint the platform shows a merchant must be "
                     "phrased at this resolution, which is why the assistance arm states signal "
                     "levels and policy consequences rather than an inferred fabrication rate.")


# --------------------------------------------------------------------------------------------------
def monitoring_continuum(seeds, n_obs_grid, cs_grid) -> dict:
    """The bridge of Proposition 8, computed rather than argued.

    cs -> 0 is Proposition 6: the signal stops depending on the hidden action and the second best
    collapses onto the no-penalty value. N_obs -> large sharpens the complaint rate toward its mean and
    pushes the second best toward first best -- the perfect-monitoring limit in which the displayed
    payoff table of the earlier interfaces becomes constructible and the oracle experiments are the
    right model of the world.
    """
    rows = []
    t0 = time.time()
    for n_obs in n_obs_grid:
        for cs in cs_grid:
            cfg = M.Config(N_obs=int(n_obs), cs=float(cs))
            cache = {}

            def rb(k, t):
                key = (k, t if k else 0.0)
                if key not in cache:
                    cache[key] = M.rbar_grid(cfg, key[0], key[1])
                return cache[key]

            fb, npen, sb, meanf = [], [], [], []
            for seed in seeds:
                mkt = M.draw_market(cfg, seed)
                best_fb = -1.0
                best_sb = -1.0
                best_f = None
                g_np = None
                for (k, t) in M.POLICY_CLASS:
                    rm = M.rbar_for_market(cfg, mkt, rb(k, t))
                    en = M.enumerate_profiles(cfg, mkt, rm)
                    best_fb = max(best_fb, float(en["GMV"].max()))
                    ne = M.pure_nash(cfg, mkt, rm, en)
                    if not ne:
                        continue
                    g = float(en["GMV"][ne[0]])
                    if k == 0.0 and g_np is None:
                        g_np = g
                    if g > best_sb:
                        best_sb = g
                        best_f = float(cfg.fgrid[list(ne[0])].mean())
                fb.append(best_fb)
                npen.append(g_np)
                sb.append(best_sb)
                meanf.append(best_f)
            fb, npen, sb = np.array(fb), np.array(npen), np.array(sb)
            rows.append(dict(N_obs=int(n_obs), cs=float(cs),
                             G_FB=float(fb.mean()), G_NP=float(npen.mean()),
                             G_SB_P=float(sb.mean()),
                             ratio_SB_over_FB=float((sb / fb).mean()),
                             ratio_NP_over_FB=float((npen / fb).mean()),
                             implementation_power=float(((sb - npen) / (fb - npen)).mean()),
                             mean_f_at_SB=float(np.mean(meanf))))
            print(f"  N_obs={n_obs:4d} cs={cs:.2f}  SB/FB={rows[-1]['ratio_SB_over_FB']:.4f}  "
                  f"power={rows[-1]['implementation_power']:.4f}  "
                  f"f*={rows[-1]['mean_f_at_SB']:.3f}  [{time.time() - t0:.0f}s]")
    return dict(seeds=[seeds[0], seeds[-1]], n_seeds=len(seeds),
                N_obs_grid=[int(v) for v in n_obs_grid], cs_grid=[float(v) for v in cs_grid],
                rows=rows,
                implementation_power_definition="(G_SB_P - G_NP) / (G_FB - G_NP): the fraction of the "
                                                "gap between doing nothing and first best that a "
                                                "signal-contingent policy in the finite class can "
                                                "actually close. 0 at cs = 0 by Proposition 6.",
                note="This scan is the quantitative form of the claim that the earlier oracle "
                     "interfaces are the perfect-monitoring endpoint of the same model, not a "
                     "different model.")


# --------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-continuum", action="store_true")
    a = ap.parse_args()
    cfg = M.Config()
    t0 = time.time()

    bench_path = OUTDIR / "ha_benchmarks.json"
    if not bench_path.exists():
        raise SystemExit("run ha_benchmarks.py first")
    bench = json.loads(bench_path.read_text())

    # the equilibrium action distribution the detector is judged against
    f_counts = np.zeros(cfg.Nf)
    for s, per in bench["named_per_seed"].items():
        for p, rec in per.items():
            for i in rec["eq_best"]:
                f_counts[i] += 1

    out = dict(
        purpose="Signal informativeness, detector error rates, posterior calibration, and the "
                "monitoring continuum linking the oracle interfaces to the uninformative-signal "
                "impossibility result.",
        generated=time.strftime("%Y-%m-%dT%H:%M:%S"),
        spec=M.spec_dict(cfg),
        equilibrium_action_distribution=dict(
            source="results/solver/ha_benchmarks.json, named policies, best equilibrium",
            counts=[int(v) for v in f_counts],
            mean_f=float((f_counts * cfg.fgrid).sum() / f_counts.sum())),
        informativeness=informativeness(cfg),
        detector=detector_quality(cfg, f_counts, list(M.POLICIES.values()) + [
            (1.0, 0.05), (1.0, 0.15), (4.0, 0.02), (8.0, 0.02), (16.0, 0.40)]),
        calibration=posterior_calibration(cfg, f_counts))

    if not a.skip_continuum:
        print("monitoring continuum scan:")
        out["monitoring_continuum"] = monitoring_continuum(
            SCAN_SEEDS, (10, 20, 40, 80, 160), (0.0, 0.15, 0.30, 0.60, 1.00))

    out["runtime_seconds"] = round(time.time() - t0, 1)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    p = OUTDIR / "ha_signal_analysis.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(out, indent=1), encoding="utf-8")
    tmp.replace(p)
    print(f"\nwrote {p} ({p.stat().st_size} bytes, {out['runtime_seconds']}s)")
    for r in out["detector"]["rows"][:2]:
        print(f"  policy (kappa={r['kappa']}, tau={r['tau']}): "
              f"FPR(honest) {r['false_positive_rate_on_fully_honest']:.4f}  "
              f"recall {r['recall_at_or_above_ref']:.4f}  precision {r['precision']:.4f}  "
              f"AUC {r['auc_complaint_rate']:.4f}")
    print(f"  posterior variance reduction from one round: "
          f"{out['calibration']['variance_reduction']:.4f}")


if __name__ == "__main__":
    main()
