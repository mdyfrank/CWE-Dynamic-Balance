"""Phase 1B -- restricted symmetric numerical FALSIFICATION model.

Purpose (per notes.md sec 6): do NOT try to "prove the economics." Build the smallest discrete,
symmetric, restricted-policy market that can *falsify* the two hypotheses, then run a wide (omega,
lambda) sweep against pass/fail criteria written BEFORE looking at the numbers.

Platform objective = LONG-RUN GMV only (notes.md sec 3). Fabrication lifts CURRENT conversion but,
through the trust-erosion feedback Q_{t+1}=Q0 g(F_t), lowers FUTURE demand -- so GMV already prices
in "misled buyers purchase less later." We do NOT add a separate consumer-harm term (that would
double-count the same loss). Consumer harm H, complaint rate, false-positive penalty rate, and
consumer abstention are REPORTED metrics, never optimized.

What is modelled (v1 defaults; see notes.md secs 2-3):
  * m symmetric merchants, common type (q, p, c); price fixed, margin = p - c.
  * Restricted stationary policy: a *constant* fabrication level f in [0,1]. Each merchant sits at the
    stationary MEAN reputation rbar(f; kappa, tau) induced by its own complaint process -- this
    reduces the merchant's problem to a stationary (steady-state average) best-response in constant f
    (appeal gain vs penalty drag vs erosion); no dynamic-programming/Bellman solve is claimed.
  * Advertised appeal a = q + (1-q) f. Logit shares with an outside option; catalog overlap omega
    modulates the outside utility, so high omega => small outside share => strong business-stealing.
  * Two-stage demand Q = Q0 * g(F), F = mean fabrication, g = exp(-lambda F) (v1) or max(0,1-lambda F).
  * Complaint signal D ~ Binomial(N_obs, theta(f, b))/N_obs -- a DISTRIBUTION, not an invertible mean;
    threshold penalty P = kappa * max(0, D - tau) hits reputation, which recovers toward 1.

Two symmetric fabrication levels to compare:
  * Private equilibrium f_NE  = fixed point of best-response (each merchant maximises OWN long-run
    profit, ignoring the erosion it inflicts on rivals + the whole category).
  * Platform optimum f_GMV = argmax_f long-run GMV(all play f). NOT "social welfare"; NOT assumed to be
    zero -- under weak erosion f_GMV>0 is efficient fabrication, under strong erosion f_GMV=0.

H1 asks: is F^NE > f_GMV over a NON-degenerate (omega, lambda) region?  (expected: yes in the erosion
         regime, vanishing in the weak-erosion / market-expansion regime).
H2 asks: does an optimised second-best penalty (kappa*, tau*) RAISE long-run GMV above no-penalty, and
         how much of the fabrication gap (F^NE - f_GMV) does it close?

No API, no external data. Pure numpy + matplotlib.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ADOPT_MARGIN = 1e-4   # platform adopts a penalty only if it beats do-nothing GMV by this much


# --------------------------------------------------------------------------------------------------
@dataclass
class Config:
    m: int = 4                 # merchants
    q: float = 0.40            # honest quality (symmetric)
    p: float = 1.00            # price
    c: float = 0.50            # marginal cost -> margin 0.50
    alpha: float = 3.0         # appeal weight in logit
    beta: float = 1.5          # reputation weight in logit
    gamma: float = 1.0         # price weight in logit
    Q0: float = 1.0            # base traffic
    eta_r: float = 0.20        # reputation recovery toward 1
    b: float = 0.05            # baseline complaint propensity (per sale)
    cs: float = 0.60           # fabrication -> complaint slope: theta(f) = clip(b + cs f, 0, 1)
    xi: float = 0.0            # ORGANIC complaint->reputation damage (0 = notes.md v1: platform penalty
                               # is the ONLY reputational force). xi>0 = the market disciplines on its
                               # own and the platform (kappa,tau) is an above-threshold amplifier.
    N_obs: int = 40            # transactions/merchant/round (complaint-signal noise / separability)
    signal_conc: float = None  # complaint model: None -> Binomial; float -> Beta-Binomial concentration
                               # s=alpha+beta (small s = MORE overdispersion, models an unknown baseline b;
                               # s->inf recovers the Binomial). Lowers honest-vs-fab separability.
    ell0: float = 0.50         # linear realized-harm slope for the REPORTED metric ell(f)=ell0 f
    delta_P: float = 0.95      # platform discount (documented; steady-state GMV folds it into a scale)
    W0_BASE: float = -1.0      # outside utility w0 = W0_BASE + W0_SPREAD*(1-omega)
    W0_SPREAD: float = 4.0     # omega=1: head-to-head (small outside);  omega=0: niche (big outside)
    Nf: int = 21
    R: int = 31
    n_omega: int = 13
    n_lambda: int = 13
    lambda_max: float = 3.0
    kappa_grid: tuple = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)
    tau_grid: tuple = (0.05, 0.10, 0.15, 0.25)
    erosion: str = "exp"       # "exp" -> e^{-lambda F};  "linear" -> max(0, 1-lambda F)
    appeal_form: str = "linear"  # a=q+(1-q)*h(f): h=f (linear), sqrt(f) (concave), f^2 (convex)

    fgrid: np.ndarray = field(init=False)
    rgrid: np.ndarray = field(init=False)

    def __post_init__(self):
        self.fgrid = np.linspace(0.0, 1.0, self.Nf)
        self.rgrid = np.linspace(0.0, 1.0, self.R)

    @property
    def margin(self) -> float:
        return self.p - self.c


def g_traffic(F, lam, cfg: Config):
    if cfg.erosion == "linear":
        return np.maximum(0.0, 1.0 - lam * np.asarray(F))
    return np.exp(-lam * np.asarray(F))


# --------------------------------------------------------------------------------------------------
# reputation: stationary MEAN under the complaint process, given (f, kappa, tau)
# --------------------------------------------------------------------------------------------------
def _binom_pmf(n: int, theta: float) -> np.ndarray:
    theta = min(max(theta, 1e-9), 1 - 1e-9)
    k = np.arange(n + 1)
    logc = np.array([math.lgamma(n + 1) - math.lgamma(kk + 1) - math.lgamma(n - kk + 1) for kk in k])
    return np.exp(logc + k * math.log(theta) + (n - k) * math.log(1 - theta))


def _lgamma_arr(x):
    return np.array([math.lgamma(float(v)) for v in x])


def _betabinom_pmf(n: int, mean: float, conc: float) -> np.ndarray:
    """Beta-Binomial: complaint rate ~ Beta(mean*conc, (1-mean)*conc), count ~ Binomial(n, rate).
    Overdispersion (relative to Binomial) grows as conc shrinks -- a faithful 'unknown baseline b'."""
    mean = min(max(mean, 1e-6), 1 - 1e-6)
    a, b = mean * conc, (1 - mean) * conc
    k = np.arange(n + 1)
    logC = math.lgamma(n + 1) - _lgamma_arr(k + 1) - _lgamma_arr(n - k + 1)
    logBnum = _lgamma_arr(k + a) + _lgamma_arr(n - k + b) - math.lgamma(n + a + b)
    logBden = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    p = np.exp(logC + logBnum - logBden)
    return p / p.sum()


def complaint_pmf(cfg: "Config", mean: float) -> np.ndarray:
    """Complaint-count distribution for a merchant whose expected complaint rate is `mean`."""
    if getattr(cfg, "signal_conc", None) is None:
        return _binom_pmf(cfg.N_obs, mean)
    return _betabinom_pmf(cfg.N_obs, mean, cfg.signal_conc)


def _threshold(kappa: float, tau: float):
    return lambda D: kappa * np.maximum(0.0, D - tau)


def _stationary_mean(cfg: Config, theta: float, P: np.ndarray) -> float:
    """Stationary mean reputation of the Markov chain r'=clip(r+eta_r(1-r)-P(count), 0, 1),
    where the complaint count ~ Binomial(N_obs, theta) and P is the per-count penalty vector."""
    R, rgrid = cfg.R, cfg.rgrid
    w = complaint_pmf(cfg, theta)
    T = np.zeros((R, R))
    for i, r in enumerate(rgrid):
        rp = np.clip(r + cfg.eta_r * (1 - r) - P, 0.0, 1.0)
        idx = np.clip(np.round(rp * (R - 1)).astype(int), 0, R - 1)
        np.add.at(T[i], idx, w)
    pi = np.full(R, 1.0 / R)
    for _ in range(300):
        pi = pi @ T
    return float(pi @ rgrid)


def rbar_table(cfg: Config, kappa: float, tau: float, penalty_fn=None) -> np.ndarray:
    """rbar[fi] = stationary mean reputation for a merchant that always fabricates fgrid[fi].
    penalty_fn(Dhat)->penalty array overrides the default threshold (used by the penalty-class study)."""
    Dhat = np.arange(cfg.N_obs + 1) / cfg.N_obs
    pf = penalty_fn if penalty_fn is not None else _threshold(kappa, tau)
    out = np.zeros(cfg.Nf)
    for fi, f in enumerate(cfg.fgrid):
        P = cfg.xi * Dhat + pf(Dhat)
        out[fi] = _stationary_mean(cfg, cfg.b + cfg.cs * f, P)
    return out


def rbar_scalar(cfg: Config, f: float, b: float, kappa: float = 0.0, tau: float = 0.0,
                penalty_fn=None) -> float:
    """Stationary mean reputation for a single heterogeneous merchant of baseline b fabricating f."""
    Dhat = np.arange(cfg.N_obs + 1) / cfg.N_obs
    pf = penalty_fn if penalty_fn is not None else _threshold(kappa, tau)
    P = cfg.xi * Dhat + pf(Dhat)
    return _stationary_mean(cfg, min(max(b + cfg.cs * f, 0.0), 1.0), P)


def false_positive_rate(cfg: Config, kappa: float, tau: float) -> float:
    """P(an HONEST seller's complaint rate crosses tau) -> fraction of honest mass that gets penalised."""
    if kappa == 0.0:
        return 0.0
    Dhat = np.arange(cfg.N_obs + 1) / cfg.N_obs
    w = complaint_pmf(cfg, cfg.b)             # honest seller: expected rate = baseline b
    return float(w[Dhat > tau].sum())


# --------------------------------------------------------------------------------------------------
# best-response, symmetric equilibrium, and the symmetric metrics (GMV objective + reported extras)
# --------------------------------------------------------------------------------------------------
def _w0(cfg: Config, omega: float) -> float:
    return cfg.W0_BASE + cfg.W0_SPREAD * (1.0 - omega)


def appeal(cfg: Config, f):
    """Advertised appeal a = q + (1-q) h(f); h sets how fabrication fills the honest gap."""
    if cfg.appeal_form == "concave":
        h = np.sqrt(np.asarray(f, dtype=float))
    elif cfg.appeal_form == "convex":
        h = np.asarray(f, dtype=float) ** 2
    else:
        h = np.asarray(f, dtype=float)
    return cfg.q + (1 - cfg.q) * h


def _util(cfg: Config, f, rbar_f):
    return cfg.alpha * appeal(cfg, f) + cfg.beta * rbar_f - cfg.gamma * cfg.p


def best_response_idx(cfg: Config, rbar: np.ndarray, omega: float, lam: float, f_riv_idx: int) -> int:
    w0 = _w0(cfg, omega)
    f_riv = cfg.fgrid[f_riv_idx]
    u_own = _util(cfg, cfg.fgrid, rbar)
    u_riv = _util(cfg, f_riv, rbar[f_riv_idx])
    den = math.exp(w0) + np.exp(u_own) + (cfg.m - 1) * math.exp(u_riv)
    s_own = np.exp(u_own) / den
    F = (cfg.fgrid + (cfg.m - 1) * f_riv) / cfg.m
    profit = cfg.margin * cfg.Q0 * g_traffic(F, lam, cfg) * s_own
    return int(np.argmax(profit))


def equilibrium(cfg: Config, rbar: np.ndarray, omega: float, lam: float):
    """Return (f_NE_index, list_of_fixed_point_indices). Multiplicity = len(list)."""
    fps = [i for i in range(cfg.Nf) if best_response_idx(cfg, rbar, omega, lam, i) == i]
    i, seen = 0, set()
    for _ in range(200):
        j = best_response_idx(cfg, rbar, omega, lam, i)
        if j == i:
            break
        if j in seen:
            i = max(i, j)
            break
        seen.add(i)
        i = j
    return i, fps


def metrics(cfg: Config, rbar: np.ndarray, omega: float, lam: float, f_idx: int) -> dict:
    """All merchants play fgrid[f_idx] and sit at rbar[f_idx]. GMV is the objective; the rest report."""
    w0 = _w0(cfg, omega)
    f = cfg.fgrid[f_idx]
    u = _util(cfg, f, rbar[f_idx])
    den = math.exp(w0) + cfg.m * math.exp(u)
    s = math.exp(u) / den
    S_in = cfg.m * s
    s0 = math.exp(w0) / den                            # consumer abstention (outside share)
    Q = cfg.Q0 * float(g_traffic(f, lam, cfg))
    GMV = cfg.p * Q * S_in                             # <-- the platform objective
    H = Q * S_in * (cfg.ell0 * f)                      # reported: misled mass * realized harm/buyer
    complaint = cfg.b + cfg.cs * f                     # reported: complaint rate per sale
    return dict(GMV=GMV, H=H, S_in=S_in, abstain=s0, complaint=complaint, Q=Q, f=f)


def gmv_optimal(cfg: Config, rbar: np.ndarray, omega: float, lam: float) -> int:
    gs = np.array([metrics(cfg, rbar, omega, lam, i)["GMV"] for i in range(cfg.Nf)])
    return int(np.argmax(gs))


def second_best(cfg, rbar0, rbar_pen, penalties, omega, lam, ne_i):
    """argmax over the penalty grid of long-run GMV at the induced equilibrium; do-nothing on ties."""
    G0 = metrics(cfg, rbar0, omega, lam, ne_i)["GMV"]
    best = (G0, 0.0, cfg.tau_grid[0], ne_i)
    for (k, t) in penalties:
        if k == 0.0:
            continue
        rb = rbar_pen[(k, t)]
        ei, _ = equilibrium(cfg, rb, omega, lam)
        G = metrics(cfg, rb, omega, lam, ei)["GMV"]
        if G > best[0] + ADOPT_MARGIN:
            best = (G, k, t, ei)
    return best   # (GMV, kappa*, tau*, eq_index)


# --------------------------------------------------------------------------------------------------
# the (omega, lambda) sweep
# --------------------------------------------------------------------------------------------------
def run_sweep(cfg: Config, verbose: bool = True):
    omegas = np.linspace(0.0, 1.0, cfg.n_omega)
    lams = np.linspace(0.0, cfg.lambda_max, cfg.n_lambda)
    rbar0 = rbar_table(cfg, 0.0, cfg.tau_grid[0])
    penalties = [(k, t) for k in cfg.kappa_grid for t in cfg.tau_grid]
    rbar_pen = {(k, t): (rbar0 if k == 0 else rbar_table(cfg, k, t)) for (k, t) in penalties}
    if verbose:
        print(f"precomputed rbar for {len(penalties)} penalty settings")

    shape = (cfg.n_lambda, cfg.n_omega)
    f_ne = np.zeros(shape); f_gmv = np.zeros(shape); mult = np.zeros(shape)
    dGMV = np.zeros(shape); kstar = np.zeros(shape); tstar = np.zeros(shape)
    gap_closed = np.full(shape, np.nan); collapse = np.zeros(shape)
    H_sb = np.zeros(shape); fp_sb = np.zeros(shape); abstain_sb = np.zeros(shape)

    for li, lam in enumerate(lams):
        for oi, omega in enumerate(omegas):
            ne_i, fps = equilibrium(cfg, rbar0, omega, lam)
            gi = gmv_optimal(cfg, rbar0, omega, lam)
            f_ne[li, oi] = cfg.fgrid[ne_i]; f_gmv[li, oi] = cfg.fgrid[gi]; mult[li, oi] = len(fps)
            G0 = metrics(cfg, rbar0, omega, lam, ne_i)["GMV"]
            Gsb, kb, tb, eib = second_best(cfg, rbar0, rbar_pen, penalties, omega, lam, ne_i)
            dGMV[li, oi] = Gsb - G0; kstar[li, oi] = kb; tstar[li, oi] = tb
            msb = metrics(cfg, rbar_pen[(kb, tb)], omega, lam, eib)
            H_sb[li, oi] = msb["H"]; abstain_sb[li, oi] = msb["abstain"]
            fp_sb[li, oi] = false_positive_rate(cfg, kb, tb)
            collapse[li, oi] = 1.0 if msb["S_in"] < 0.02 else 0.0
            gap = f_ne[li, oi] - f_gmv[li, oi]
            if gap > 1e-9:
                gap_closed[li, oi] = (f_ne[li, oi] - cfg.fgrid[eib]) / gap

    return dict(omegas=omegas, lams=lams, f_ne=f_ne, f_gmv=f_gmv, gap=f_ne - f_gmv, mult=mult,
                dGMV=dGMV, kstar=kstar, tstar=tstar, gap_closed=gap_closed, collapse=collapse,
                H_sb=H_sb, fp_sb=fp_sb, abstain_sb=abstain_sb)


# --------------------------------------------------------------------------------------------------
# erosion scan at fixed omega -- the notes.md sec 3 headline: efficient fabrication -> honesty
# --------------------------------------------------------------------------------------------------
def erosion_scan(cfg: Config, omega: float, lams: np.ndarray):
    rbar0 = rbar_table(cfg, 0.0, cfg.tau_grid[0])
    penalties = [(k, t) for k in cfg.kappa_grid for t in cfg.tau_grid]
    rbar_pen = {(k, t): (rbar0 if k == 0 else rbar_table(cfg, k, t)) for (k, t) in penalties}
    rows = []
    for lam in lams:
        ne_i, _ = equilibrium(cfg, rbar0, omega, lam)
        gi = gmv_optimal(cfg, rbar0, omega, lam)
        G0 = metrics(cfg, rbar0, omega, lam, ne_i)["GMV"]
        Gsb, kb, tb, eib = second_best(cfg, rbar0, rbar_pen, penalties, omega, lam, ne_i)
        m_ne = metrics(cfg, rbar0, omega, lam, ne_i)
        m_sb = metrics(cfg, rbar_pen[(kb, tb)], omega, lam, eib)
        rows.append(dict(lam=float(lam), f_ne=cfg.fgrid[ne_i], f_gmv=cfg.fgrid[gi],
                         f_sb=cfg.fgrid[eib], kstar=kb, tstar=tb,
                         gmv_ne=m_ne["GMV"], gmv_sb=Gsb, H_ne=m_ne["H"], H_sb=m_sb["H"],
                         complaint_ne=m_ne["complaint"], complaint_sb=m_sb["complaint"]))
    return rows


def make_erosion_figure(rows, cfg: Config, omega: float, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    la = np.array([r["lam"] for r in rows])
    fig, ax = plt.subplots(1, 3, figsize=(9.6, 3.2), dpi=150)

    fg = np.array([r["f_gmv"] for r in rows])
    # shade the efficient-fabrication regime (f_GMV > 0)
    ax[0].fill_between(la, 0, 1.03, where=fg > 0.01, color="#f0e2c0", alpha=0.5,
                       label="efficient-fab. regime")
    ax[0].plot(la, [r["f_ne"] for r in rows], "--", c="#b00", label=r"$F^{\mathrm{NE}}$ (private)")
    ax[0].plot(la, fg, "-", c="#333", label=r"$f^{\mathrm{GMV}}$ (platform opt.)")
    ax[0].plot(la, [r["f_sb"] for r in rows], "-o", c="#1d7a4c", ms=3, label="second-best F")
    ax[0].set_xlabel(r"erosion  $\lambda$", fontsize=9); ax[0].set_ylabel("fabrication F", fontsize=9)
    ax[0].set_title("efficient fabrication $\\to$ honesty", fontsize=10)
    ax[0].set_ylim(-0.03, 1.05); ax[0].legend(fontsize=6.5, frameon=False, loc="upper right")

    ax[1].plot(la, [r["gmv_ne"] for r in rows], "--", c="#b00", label="no penalty")
    ax[1].plot(la, [r["gmv_sb"] for r in rows], "-o", c="#1d7a4c", ms=3, label="second-best")
    ax[1].set_xlabel(r"erosion  $\lambda$", fontsize=9); ax[1].set_ylabel("long-run GMV", fontsize=9)
    ax[1].set_title("penalty rescues long-run GMV", fontsize=10)
    ax[1].legend(fontsize=7, frameon=False)

    ax[2].plot(la, [r["kstar"] for r in rows], "-o", c="#5b8ca5", ms=3)
    ax[2].set_xlabel(r"erosion  $\lambda$", fontsize=9)
    ax[2].set_ylabel(r"second-best penalty  $\kappa^\star$", fontsize=9)
    ax[2].set_title("optimal penalty grows with erosion", fontsize=10)

    for a in ax:
        a.tick_params(labelsize=7); a.grid(alpha=0.2)
    fig.suptitle(f"Erosion scan at omega={omega:.2f}  (m={cfg.m}, g={cfg.erosion}, xi={cfg.xi})",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out); plt.close(fig)
    print(f"wrote {out}")


# --------------------------------------------------------------------------------------------------
# pre-registered pass/fail criteria (notes.md sec 6) -- fixed before reading the numbers
# --------------------------------------------------------------------------------------------------
def evaluate(res: dict, cfg: Config, gap_tol: float = 0.05):
    gap = res["gap"]
    over = gap > gap_tol
    frac_region = float(over.mean())
    helped = (res["dGMV"] > 1e-9) & over
    frac_helped = float(helped[over].mean()) if over.any() else 0.0
    gc = res["gap_closed"][over & np.isfinite(res["gap_closed"])]
    mean_gap_closed = float(np.mean(gc)) if gc.size else float("nan")
    multi = float((res["mult"] > 1).mean())
    collapse = float(res["collapse"].mean())
    fne = res["f_ne"]; fgmv = res["f_gmv"]

    L = ["=" * 78, "PHASE 1B  pass/fail  (criteria fixed before the run; notes.md sec 6)", "=" * 78]
    L.append(f"[1] REGION not point : {frac_region*100:5.1f}% of cells have F_NE - f_GMV > {gap_tol}"
             f"   -> {'PASS' if frac_region > 0.20 else 'fail'}")
    L.append(f"    max gap {gap.max():.3f}; f_GMV in [{fgmv.min():.2f},{fgmv.max():.2f}] "
             f"(efficient-fab. cells f_GMV>0: {(fgmv>0.01).mean()*100:.0f}%)")
    L.append(f"[2] SECOND-BEST helps: on over-fab cells, long-run GMV rises in {frac_helped*100:5.1f}%"
             f"   -> {'PASS' if frac_helped > 0.5 else 'fail'}")
    L.append(f"    mean fraction of the fabrication gap CLOSED by (kappa*,tau*) = {mean_gap_closed*100:5.1f}%")
    L.append(f"[4] PATHOLOGIES      : multiple equilibria {multi*100:4.1f}%; penalty-collapse {collapse*100:4.1f}%")
    L.append("-" * 78)
    L.append(f"    diag F_NE in [{fne.min():.2f},{fne.max():.2f}] "
             f"(interior {((fne>0.01)&(fne<0.99)).mean()*100:.0f}%);  "
             f"kappa* mean {res['kstar'].mean():.2f}, tau* mean {res['tstar'].mean():.3f}")
    L.append(f"    REPORTED @ second-best: harm H mean {res['H_sb'].mean():.3f}, "
             f"false-pos rate mean {res['fp_sb'].mean():.3f}, abstention mean {res['abstain_sb'].mean():.3f}")
    L.append("=" * 78)
    return "\n".join(L)


# --------------------------------------------------------------------------------------------------
def make_figure(res: dict, cfg: Config, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    om, la = res["omegas"], res["lams"]
    ext = [om[0], om[-1], la[0], la[-1]]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), dpi=150)

    def panel(ax, Z, title, cmap, vmin=None, vmax=None, cbar=""):
        im = ax.imshow(Z, origin="lower", aspect="auto", extent=ext, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(r"catalog overlap  $\omega$", fontsize=9)
        ax.set_ylabel(r"erosion  $\lambda$", fontsize=9)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label(cbar, fontsize=8); cb.ax.tick_params(labelsize=7); ax.tick_params(labelsize=7)

    panel(axes[0, 0], res["f_gmv"], r"platform optimum  $f^{\mathrm{GMV}}$", "viridis", 0, 1, "fab.")
    panel(axes[0, 1], res["gap"], r"over-fabrication  $F^{\mathrm{NE}}-f^{\mathrm{GMV}}$", "Reds", 0, 1, "gap")
    panel(axes[1, 0], res["dGMV"], r"second-best gain  $\Delta$GMV$(\kappa^\star,\tau^\star)$", "Greens", 0, None, r"$\Delta$GMV")
    panel(axes[1, 1], np.clip(res["gap_closed"], 0, 1), r"fraction of gap closed", "Blues", 0, 1, "closed")

    fig.suptitle(f"Phase 1B sweep -- long-run GMV objective  (m={cfg.m}, g={cfg.erosion}, xi={cfg.xi})",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out); plt.close(fig)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--erosion", choices=["exp", "linear"], default="exp")
    ap.add_argument("--m", type=int, default=4)
    ap.add_argument("--xi", type=float, default=0.0, help="organic complaint->reputation damage (0=v1)")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--erosion-scan", action="store_true",
                    help="scan lambda at fixed omega (efficient-fabrication -> honesty transition)")
    ap.add_argument("--omega", type=float, default=0.30, help="fixed overlap for --erosion-scan")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = Config(erosion=args.erosion, m=args.m, xi=args.xi)
    tag = f"_organic{args.xi:g}" if args.xi > 0 else ""
    if args.quick:
        cfg.n_omega = cfg.n_lambda = 7; cfg.Nf = 15; cfg.R = 21; cfg.N_obs = 25; cfg.__post_init__()

    if args.erosion_scan:
        out = Path(args.out) if args.out else HERE / f"fig_erosion{tag}.png"
        lams = np.linspace(0.0, 1.6, 17)
        print(f"erosion scan at omega={args.omega}, xi={cfg.xi}")
        rows = erosion_scan(cfg, args.omega, lams)
        print(f"{'lam':>5} {'F_NE':>6} {'f_GMV':>6} {'F_sb':>6} {'kappa*':>7} {'GMV_ne':>7} {'GMV_sb':>7} {'H_sb':>6}")
        for r in rows:
            print(f"{r['lam']:5.2f} {r['f_ne']:6.2f} {r['f_gmv']:6.2f} {r['f_sb']:6.2f} "
                  f"{r['kstar']:7.2f} {r['gmv_ne']:7.3f} {r['gmv_sb']:7.3f} {r['H_sb']:6.3f}")
        make_erosion_figure(rows, cfg, args.omega, out)
        return

    out = Path(args.out) if args.out else HERE / f"fig_phase1b{tag}.png"
    print(f"Config: m={cfg.m}, erosion={cfg.erosion}, xi={cfg.xi}, grid={cfg.n_omega}x{cfg.n_lambda}")
    res = run_sweep(cfg)
    print(evaluate(res, cfg))
    make_figure(res, cfg, out)


if __name__ == "__main__":
    main()
