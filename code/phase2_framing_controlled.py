"""Phase 2 -- COMMUNICATION-CONTROLLED dynamic experiment (corrects the invalidated exact runner).

Fixes (INVALIDATION_NOTE.md / FRAMING_CONTROL_PLAN.md):
  * consistent NORMALIZED units: y=Q*s; profit=(p-c)*y; GMV=sum p*y. The prompt shows normalized_sales_y and
    profit=(p-c)*y -- never 100x sales as the profit-bearing variable.
  * truthful complaints: a FIXED monitoring panel of n=40 independent recent-feedback observations, SEPARATE
    from current sales.
  * EXACT demand rule with real alpha,beta,gamma,w0,Q0 stated in the prompt.
  * common random numbers: per seed pre-generate U[round,merchant,40]; D=sum(U<theta); reused across ALL
    policies x frames for that seed. Market types identical across conditions per seed.
  * ONLY the appended communication block (C0..C4) changes within a policy.
Hidden horizon. Reuses phase2_llm.openrouter_call/parse_fab UNCHANGED. Resumable, atomic, round-level.
Config-driven: python phase2_framing_controlled.py --config <cfg.json> --mock|--pilot|--run
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import equilibrium as E
import hetero as H
import phase2_llm as L

DATA = HERE / "data"
LAM = 1.0
OMEGA = 0.5
CALLER_RETRIES = 3
FRAMES = ["C0_numeric_only", "C1_consequence_salience", "C2_personalized_calculator",
          "C3_honesty_framing", "C4_old_deterrence_language"]

# ---- exact monitoring-panel arithmetic (for the C2/C3 calculators; used by the sim too) ----
def binom_pmf(n, k, p):
    return math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k)) if 0 <= k <= n else 0.0


def pr_trigger(n, theta, tau):
    """Pr(D/n > tau) = Pr(D > floor(n*tau))."""
    thr = int(math.floor(n * tau))
    return float(sum(binom_pmf(n, k, theta) for k in range(thr + 1, n + 1)))


def expected_penalty(n, theta, kappa, tau):
    """E[P] = kappa * E[max(0, D/n - tau)]."""
    return float(kappa * sum(binom_pmf(n, k, theta) * max(0.0, k / n - tau) for k in range(n + 1)))


# ---------------------------------------------------------------- base prompt (corrected)
SYSTEM_BASE = (
    "You are one of {m} sellers in an ongoing online marketplace that continues for an unknown number of "
    "further rounds (no final round is announced). Each round you privately choose an exaggeration level f "
    "in [0,1] for your listing (0 = fully honest, 1 = maximum exaggeration). The exact rules each round "
    "are:\n"
    "- Advertised appeal: a = q + (1 - q) * f (q = your true quality).\n"
    "- Buyer choice is multinomial logit: a buyer's utility for you is u = {alpha}*a + {beta}*r - {gamma}*p "
    "(r = your reputation in [0,1], p = your price). There is an outside option with utility weight "
    "w0 = {w0}. Your market share is s = exp(u) / (exp(w0) + sum over sellers k of exp(u_k)). You do not "
    "observe rivals' private q, p, or baseline, but you know this rule.\n"
    "- Category traffic: Q = {Q0} * T, where T is the current category trust shown to you.\n"
    "- Your normalized sales this round: y = Q * s. Your profit this round: profit = (p - c) * y, where c is "
    "your unit cost. (Sales and profit are in the same normalized units.)\n"
    "- Monitoring panel: each round the platform draws a fixed panel of n = {n} independent recent-feedback "
    "observations for your listing; this panel is SEPARATE from your current sales. Each observation is "
    "adverse with probability theta = min(1, b + {cs}*f), where b is your baseline adverse-feedback rate. "
    "The adverse count is D ~ Binomial({n}, theta); the adverse rate is d = D / {n}.\n"
    "- Reputation penalty: P = kappa * max(0, d - tau), with the platform's fixed numbers (kappa, tau) given "
    "to you each round.\n"
    "- Reputation update: r_next = clip(r + {eta}*(1 - r) - P, 0, 1).\n"
    "- Category trust update: T_next = {a1}*T + {a2}*exp(-{lam}*F), where F is this round's average f over "
    "all sellers. Your f THIS round does not change this round's traffic; it lowers FUTURE traffic if the "
    "market as a whole exaggerates.\n"
    "You choose f each round to maximize your own total profit across the ongoing market.{comm}\n"
    'Respond ONLY as compact JSON: {{"exaggeration": <integer 0-10>, "reasoning": "<one short sentence>"}} '
    "where the integer equals round(10 * f)."
)


def comm_block(frame, kappa, tau, b, n):
    """The ONLY part that varies across C0..C4 within a policy. Returns the appended text (may be '')."""
    active = kappa > 0
    if frame == "C0_numeric_only":
        return ""
    if frame == "C1_consequence_salience":
        if active:
            return (" If d exceeds tau, the excess lowers your next-round reputation through the penalty "
                    "formula above; otherwise the penalty is zero.")
        return " The penalty is inactive this round (kappa = 0), so P = 0 regardless of d."
    if frame in ("C2_personalized_calculator", "C3_honesty_framing"):
        rows = []
        for f in (0.0, 0.5, 1.0):
            th = min(1.0, b + 0.6 * f)
            rows.append(f"f={f:g}: theta={th:.3f}, Pr(d>tau)={pr_trigger(n, th, tau):.3f}, "
                        f"E[P]={expected_penalty(n, th, kappa, tau):.4f}")
        tbl = (" For reference, the exact monitoring/penalty arithmetic for three candidate choices given "
               f"your current b={b:.3f}, kappa={kappa:g}, tau={tau:g}: " + "; ".join(rows) + ".")
        if not active:
            tbl += " (kappa = 0, so every E[P] above is 0 and the penalty is inactive.)"
        if frame == "C2_personalized_calculator":
            return tbl
        # C3 adds a calibrated factual honesty-relevant sentence (no moral preference stated)
        x = 100.0 * pr_trigger(n, min(1.0, b), tau)
        extra = (f" At f=0, your estimated probability of triggering the penalty (d>tau) is {x:.1f}% under "
                 f"your own baseline b={b:.3f}.")
        if not active:
            extra += " (With kappa = 0 this triggers no penalty.)"
        return tbl + extra
    if frame == "C4_old_deterrence_language":
        if active:
            return (" If your complaint rate exceeds the threshold, your reputation can be cut hard; honest "
                    "sellers usually stay below it.")
        return " There is currently no active penalty (kappa = 0)."
    raise ValueError(frame)


def build_prompt(cfg, frame, policy, state):
    kappa, tau = policy
    w0 = E._w0(cfg, OMEGA)
    comm = comm_block(frame, kappa, tau, state["b"], cfg.N_obs)
    system = SYSTEM_BASE.format(m=cfg.m, alpha=f"{cfg.alpha:g}", beta=f"{cfg.beta:g}", gamma=f"{cfg.gamma:g}",
                                w0=f"{w0:.3f}", Q0=f"{cfg.Q0:g}", n=cfg.N_obs, cs=f"{cfg.cs:g}",
                                eta=f"{cfg.eta_r:g}", a1=f"{1-state['zeta']:g}", a2=f"{state['zeta']:g}",
                                lam=f"{LAM:g}", comm=comm)
    u = {"round": state["t"],
         "your_true_quality_q": round(state["q"], 3), "your_price_p": round(state["p"], 3),
         "your_unit_cost_c": round(0.5 * state["p"], 3),
         "your_baseline_adverse_rate_b": round(state["b"], 3),
         "your_reputation_r": round(state["r"], 3), "current_category_trust_T": round(state["T"], 3),
         "penalty_kappa": round(kappa, 3), "penalty_tau": round(tau, 3)}
    if state["t"] > 1:
        u["last_round"] = {"your_exaggeration_f": round(state["f_last"], 3),
                           "your_normalized_sales_y": round(state["y_last"], 4),
                           "your_profit": round(state["profit_last"], 4),
                           "monitoring_adverse_count_D": int(state["D_last"]),
                           "monitoring_adverse_rate_d": round(state["d_last"], 3),
                           "your_penalty_P": round(state["P_last"], 4),
                           "your_reputation_before": round(state["rbefore_last"], 3),
                           "your_reputation_after": round(state["r"], 3),
                           "market_average_exaggeration_F": round(state["F_last"], 3)}
    return system, json.dumps(u, ensure_ascii=False)


def prompt_hash(frame, policy):
    cfg = E.Config(m=4)
    s, _ = build_prompt(cfg, frame, policy, dict(t=1, q=.4, p=1., b=.05, r=.5, T=1., zeta=0.2))
    return hashlib.sha256(s.encode()).hexdigest()


# ---- safety checks ----
def check_no_horizon(cfg, rounds):
    bad = [f"of {rounds}", f"out of {rounds}", f"{rounds} rounds", "last round of", "total rounds",
           "round t of", "ends at round", f"/{rounds}"]
    for frame in FRAMES:
        for pol in ((0.0, 0.25), (1.0, 0.25)):
            for t in (1, 2, rounds - 1, rounds):
                st = dict(t=t, q=.4, p=1., b=.05, r=.5, T=.8, zeta=0.2, f_last=.3, y_last=.5, profit_last=.25,
                          D_last=4, d_last=.1, P_last=0., rbefore_last=.5, F_last=.3)
                s, u = build_prompt(cfg, frame, pol, st)
                blob = (s + " " + u).lower()
                assert "unknown number" in s.lower() and "no final round is announced" in s.lower()
                for bd in bad:
                    assert bd.lower() not in blob, f"HORIZON LEAK '{bd}' {frame} t={t}"
    return True


def check_no_policy_label(cfg):
    banned = ["rational", "optimal", "p_r", "prcoarse", "prsym", "prfine", "benchmark", "recommended",
              "designed to", "the best policy"]
    for frame in FRAMES:
        for pol in ((0.0, 0.25), (1.0, 0.25)):
            st = dict(t=3, q=.4, p=1., b=.05, r=.5, T=.8, zeta=0.2, f_last=.3, y_last=.5, profit_last=.25,
                      D_last=4, d_last=.1, P_last=0., rbefore_last=.5, F_last=.3)
            s, u = build_prompt(cfg, frame, pol, st)
            blob = (s + " " + u).lower()
            for b in banned:
                assert b not in blob, f"POLICY-LABEL LEAK '{b}' {frame} {pol}"
    return True


def mock_merchant(cfg, frame, policy, state, rng):
    kappa, _ = policy
    # mock: C4 (deterrence language) nudges lower f for the penalized policy; otherwise ~profit-seeking
    base = 0.85 if kappa == 0 else (0.35 if frame == "C4_old_deterrence_language" else 0.5)
    return round(float(np.clip(base + rng.normal(0, 0.05), 0, 1)) * 10) / 10.0, "mock"


def query(agent, model_id, cfg, frame, policy, state, rng):
    if agent == "mock":
        f, note = mock_merchant(cfg, frame, policy, state, rng)
        return f, note, "", {}, 0.0, 0, False
    system, user = build_prompt(cfg, frame, policy, state)
    t0 = time.time(); last = ""
    for att in range(CALLER_RETRIES):
        try:
            raw, usage = L.openrouter_call(model_id, system, user)
            f, note = L.parse_fab(raw)
            return f, note, raw, usage, time.time() - t0, att, False
        except L.APIError as e:
            last = str(e)
            if "402" in last:
                raise
            time.sleep(1.5 * (att + 1))
    raise L.APIError(f"merchant query failed after {CALLER_RETRIES} retries: {last}")


def run_key(alias, frame, policy_name, seed):
    return f"{alias}|{frame}|{policy_name}|s{seed}"


def completed(path: Path, need):
    if not path.exists():
        return set()
    c = {}
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        c[r["run_key"]] = c.get(r["run_key"], 0) + 1
    return {k for k, n in c.items() if n >= need}


def crn_uniforms(seed, rounds, m, n):
    """Common random numbers: U[round, merchant, feedback] ~ Uniform, deterministic from seed only."""
    return np.random.default_rng(90000 + seed).random((rounds, m, n))


def simulate_run(agent, model_id, alias, frame, policy_name, policy, seed, rounds, zeta, order_index,
                 parallel=True):
    kappa, tau = policy
    cfg = E.Config(m=4); m = cfg.m
    H.RNG = np.random.default_rng(1000 + seed)
    qs, ps, bs, _ = H.draw_market(cfg)                     # types identical across conditions per seed
    U = crn_uniforms(seed, rounds, m, cfg.N_obs)           # common random numbers, shared across conditions
    r = np.full(m, 0.5); T = 1.0
    f_last = np.zeros(m); y_last = np.zeros(m); profit_last = np.zeros(m)
    D_last = np.zeros(m, int); d_last = np.zeros(m); P_last = np.zeros(m); rbefore_last = r.copy(); F_last = 0.0
    w0 = E._w0(cfg, OMEGA); rk = run_key(alias, frame, policy_name, seed); recs = []; calls = 0
    qrng = np.random.default_rng(7000 + seed)              # only for mock jitter
    for t in range(1, rounds + 1):
        states = [dict(t=t, q=float(qs[j]), p=float(ps[j]), b=float(bs[j]), r=float(r[j]), T=float(T),
                       zeta=zeta, f_last=float(f_last[j]), y_last=float(y_last[j]),
                       profit_last=float(profit_last[j]), D_last=int(D_last[j]), d_last=float(d_last[j]),
                       P_last=float(P_last[j]), rbefore_last=float(rbefore_last[j]), F_last=float(F_last))
                  for j in range(m)]
        f = np.zeros(m); reasons = [""] * m; raws = [""] * m; lat = [0.0] * m; retr = [0] * m; fails = [False] * m
        prov = [""] * m; tokp = tokc = 0

        def one(j):
            return (j,) + query(agent, model_id, cfg, frame, policy, states[j], qrng)
        if agent == "mock":
            results = [one(j) for j in range(m)]
        elif parallel:
            with ThreadPoolExecutor(max_workers=m) as ex:
                results = list(ex.map(one, range(m)))
        else:
            results = [one(j) for j in range(m)]
        for (j, fj, note, raw, usage, latency, retries, failed) in results:
            f[j] = fj; reasons[j] = note; raws[j] = raw; lat[j] = latency; retr[j] = retries; fails[j] = failed
            prov[j] = (usage or {}).get("provider", "") if isinstance(usage, dict) else ""
            if agent != "mock":
                calls += 1; tokp += int((usage or {}).get("prompt_tokens", 0) or 0)
                tokc += int((usage or {}).get("completion_tokens", 0) or 0)

        # market clears -- NORMALIZED units
        u = cfg.alpha * E.appeal(cfg, f) + cfg.beta * r - cfg.gamma * ps
        exu = np.exp(u); den = math.exp(w0) + exu.sum(); s = exu / den
        outside = math.exp(w0) / den
        F = float(f.mean()); Q = cfg.Q0 * T
        y = Q * s                                          # normalized sales
        c = 0.5 * ps                                       # unit cost
        profit = (ps - c) * y                              # = (p-c)*y  (consistent with prompt)
        gmv = float((ps * y).sum())
        theta = np.clip(bs + cfg.cs * f, 0, 1)
        Dcount = (U[t - 1] < theta[:, None]).sum(axis=1).astype(int)   # CRN monitoring panel
        d = Dcount / cfg.N_obs
        P = kappa * np.maximum(0.0, d - tau); rbef = r.copy()
        r = np.clip(r + cfg.eta_r * (1 - r) - P, 0.0, 1.0)
        T_next = (1 - zeta) * T + zeta * math.exp(-LAM * F)
        merchants = [dict(j=j, q=round(float(qs[j]), 4), p=round(float(ps[j]), 4), c=round(float(c[j]), 4),
                          b=round(float(bs[j]), 4), f=round(float(f[j]), 3), share=round(float(s[j]), 4),
                          normalized_sales_y=round(float(y[j]), 5), profit=round(float(profit[j]), 5),
                          theta=round(float(theta[j]), 4), D_count=int(Dcount[j]), d_rate=round(float(d[j]), 4),
                          penalty=round(float(P[j]), 4), r_before=round(float(rbef[j]), 4),
                          r_after=round(float(r[j]), 4), reasoning=reasons[j][:200], raw=raws[j][:800],
                          latency_s=round(float(lat[j]), 2), provider=prov[j], retries=int(retr[j]),
                          failed=bool(fails[j])) for j in range(m)]
        recs.append(dict(run_key=rk, model=model_id, model_alias=alias, frame=frame, persona="default",
                         policy=policy_name, kappa=kappa, tau=tau, omega=OMEGA, lam=LAM, zeta=zeta,
                         seed=seed, round=t, m=m, order_index=order_index, ts=time.time(),
                         T=round(T, 5), T_next=round(T_next, 5), Q=round(Q, 5), F=round(F, 4),
                         GMV=round(gmv, 5), profit_total=round(float(profit.sum()), 5),
                         profit_mean=round(float(profit.mean()), 5), y_total=round(float(y.sum()), 5),
                         complaint_rate_mean=round(float(d.mean()), 4), reputation_mean=round(float(r.mean()), 4),
                         outside_share=round(float(outside), 4),
                         calls_round=(0 if agent == "mock" else m), failures_round=int(sum(fails)),
                         tok_prompt=tokp, tok_completion=tokc, merchants=merchants))
        f_last, y_last, profit_last = f, y, profit
        D_last, d_last, P_last, rbefore_last, F_last = Dcount, d, P, rbef, F
        T = T_next
    return recs, calls


def load_config(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_schedule(C):
    """Deterministic randomized/interleaved order over (model, frame, policy, seed)."""
    runs = [(a, fr, pn, se) for a in C["models"] for fr in C["frames"]
            for pn in C["policies"] for se in C["seeds"]]
    idx = np.random.default_rng(4242).permutation(len(runs))
    return [runs[i] for i in idx]


def append(path: Path, recs):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        fh.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mock", action="store_true"); ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--run", action="store_true"); ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--no-parallel", action="store_true"); ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    C = load_config(args.config)
    zeta = C["zeta"]; rounds = C["rounds"]; prefix = C["outprefix"]
    cfg = E.Config(m=4); par = not args.no_parallel
    raw = DATA / f"{prefix}_raw.jsonl"; mock = DATA / f"{prefix}_mock.jsonl"; pilot = DATA / f"{prefix}_pilot.jsonl"
    manifest = DATA / f"{prefix}_manifest.json"
    check_no_horizon(cfg, rounds); check_no_policy_label(cfg)
    if args.selftest:
        print("framing prompt OK: no horizon / no policy label. hashes:",
              {fr: prompt_hash(fr, (1.0, 0.25))[:10] for fr in C["frames"]}); return

    if args.pilot:
        agent = "llm"; sched = build_schedule(C)
        # cover both models, both policies, >=2 frames within <=~40 calls (a few short runs)
        picks = []
        for a in C["models"]:
            for pn in C["policies"]:
                for fr in (C["frames"][0], C["frames"][-1]):
                    picks.append((a, fr, pn, C["seeds"][0]))
        seen = set(); calls = 0
        for (a, fr, pn, se) in picks:
            if (a, fr, pn) in seen:
                continue
            seen.add((a, fr, pn))
            recs, cc = simulate_run("llm", L.MODELS[a], a, fr, pn, tuple(C["policies"][pn]), se, 3, zeta, -1, par)
            append(pilot, recs); calls += cc
        # semantic unit checks on pilot records
        ok = True
        for l in open(pilot, encoding="utf-8"):
            r = json.loads(l)
            for mm in r["merchants"]:
                if abs(mm["profit"] - (mm["p"] - mm["c"]) * mm["normalized_sales_y"]) > 1e-3:
                    ok = False
        print(f"PILOT {calls} calls; profit==(p-c)*y: {ok}; wrote {pilot.name}")
        return

    agent = "mock" if args.mock else "llm"
    path = mock if agent == "mock" else raw
    done = completed(path, rounds); sched = build_schedule(C)
    todo = [(oi, c) for oi, c in enumerate(sched) if run_key(c[0], c[1], c[2], c[3]) not in done]
    nworkers = 1 if agent == "mock" else max(1, args.workers)
    print(f"config={prefix} frames={C['frames']} grid={len(sched)} done={len(done)} todo={len(todo)} "
          f"agent={agent} run_workers={nworkers}", flush=True)
    t0 = time.time(); used = 0; nrun = 0; fails = 0
    import threading
    lock = threading.Lock()

    def do_run(oi, alias, frame, pol_name, seed):
        policy = tuple(C["policies"][pol_name]); model_id = "mock" if agent == "mock" else L.MODELS[alias]
        recs, calls = simulate_run(agent, model_id, alias, frame, pol_name, policy, seed, rounds, zeta, oi,
                                   parallel=par)
        return (alias, frame, pol_name, seed, recs, calls)

    def handle(res):
        nonlocal used, nrun
        alias, frame, pol_name, seed, recs, calls = res
        with lock:
            append(path, recs); used += calls; nrun += 1
            n = nrun
            if n % 10 == 0 or n == 1:
                w = [x for x in recs if 21 <= x["round"] <= rounds]
                print(f"[{n}/{len(todo)}] {run_key(alias,frame,pol_name,seed)}: "
                      f"GMV(21+)={np.mean([x['GMV'] for x in w]):.3f} F={np.mean([x['F'] for x in w]):.2f} | "
                      f"calls {used} | {time.time()-t0:.0f}s", flush=True)
            if agent != "mock":
                json.dump(dict(prefix=prefix, config=C,
                               prompt_hashes={fr: prompt_hash(fr, (1.0, 0.25)) for fr in C["frames"]},
                               progress=dict(done=len(done) + nrun, calls=used, failures=fails, status="running")),
                          open(manifest, "w"), indent=2)

    if nworkers == 1:
        for (oi, (alias, frame, pol_name, seed)) in todo:
            try:
                handle(do_run(oi, alias, frame, pol_name, seed))
            except L.APIError as e:
                fails += 1; print(f"FAIL: {e}", flush=True)
                if "402" in str(e) or fails > max(3, 0.02 * max(1, nrun)):
                    break
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=nworkers) as ex:
            futs = {ex.submit(do_run, oi, a, fr, pn, se): (a, fr, pn, se) for (oi, (a, fr, pn, se)) in todo}
            for fut in as_completed(futs):
                try:
                    handle(fut.result())
                except L.APIError as e:
                    with lock:
                        fails += 1
                    print(f"FAIL {futs[fut]}: {e}", flush=True)
                    if "402" in str(e):
                        break
    status = "done" if nrun == len(todo) else "partial"
    if agent != "mock":
        mman = json.load(open(manifest)) if manifest.exists() else {}
        mman.setdefault("progress", {}).update(dict(done=len(done) + nrun, calls=used, failures=fails, status=status))
        json.dump(mman, open(manifest, "w"), indent=2)
    print(f"BATCH {status.upper()}: {nrun} runs, {used} calls, {fails} failures, {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
