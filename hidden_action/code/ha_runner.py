"""ha_runner.py -- the experiment driver.

One CELL is one (model, arm, policy, market seed): a market of 4 merchants played for `rounds`
rounds, with every merchant's action chosen by the model each round. The matrix is

    2 models  x  4 arms  x  2 policies  x  60 seeds  =  960 cells
    960 cells x 80 rounds x 4 merchants             =  307,200 decisions

A cell is the unit of atomicity and the unit of resume: it is written once, in full, to its own file,
and a cell whose file exists is never called again. There is no partial-cell state to reconcile,
which is what makes resume trivially correct rather than merely tested.

COMMON RANDOM NUMBERS
---------------------
All latent draws for a market seed -- the 40 complaint uniforms, the refund uniforms and the 5 audit
uniforms for every (round, merchant) -- are generated once from the seed alone and reused across
every arm, policy and model. Two consequences, both wanted:

  * comparisons are PAIRED at the level of the noise, not merely of the market draw, so an arm
    difference cannot be a difference in luck;
  * the coupling is MONOTONE in f: with the uniforms held fixed, raising a merchant's fabrication can
    only weakly raise its complaint count. Realised signals therefore move in the direction the model
    says they should, seed by seed, and a violation would be a bug rather than noise.

THREE RETRY COUNTERS, NEVER MERGED
----------------------------------
  transport_attempts   HTTP-level. Counted in ha_transport. A 503 is not a decision.
  schema_repairs       the reply did not parse; the model is asked again for the same decision with
                       an information-free repair message. Counted here. Not a treatment.
  economic_retries     arm A3 only: the platform offers a reconsideration turn because the merchant's
                       OWN published record crossed the published threshold twice running. This is a
                       treatment, it is part of the mechanism being studied, and it is counted here
                       and reported separately.

Merging any two of these would let an infrastructure artefact be read as an economic finding. They
are separate fields in every record and separate columns in every summary.

WHAT IS NEVER DONE
------------------
An action is never invented. If a reply cannot be parsed after the repair budget, the cell fails and
is recorded as failed; no default, no midpoint, no last-round carry-forward. A run with 3% invented
actions is not a run with a 3% error rate, it is a run whose central quantity is partly fiction.
"""

from __future__ import annotations

import gzip
import json
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import ha_infoclass as IC                                          # noqa: E402
import ha_model as M                                               # noqa: E402
import ha_prompts as PR                                            # noqa: E402
import ha_transport as TR                                          # noqa: E402

PKG = HERE.parent
RESULTS = PKG / "results"
RAW = RESULTS / "raw"
LOGS = RESULTS / "logs"
PROGRESS = RESULTS / "progress"
CACHE = RESULTS / "cache"

CRN_BASE = 500000              # distinct from the market draw base (1000) so the streams cannot alias
CRN_MAX_ROUNDS = 120           # fixed draw horizon: a short run is a noise-prefix of a long one
R_INIT = 0.5                   # every merchant starts at neutral reputation
SCHEMA_REPAIR_BUDGET = 2       # information-free parse repairs per decision
CELL_RETRY_BUDGET = 2          # whole-cell re-runs on transient failure
DEFAULT_ROUNDS = 80

POLICY_NAMES = {"P_GMV": (0.5, 0.20, 0.0, 0.0), "P_robust": (4.0, 0.30, 0.0, 0.0)}


class CellFailure(RuntimeError):
    """This cell could not be completed. Recorded, never patched over."""


# ==================================================================================================
# common random numbers
# ==================================================================================================
def draw_crn(cfg: M.Config, seed: int, rounds: int) -> dict:
    """Uniforms for one market seed, drawn at a FIXED horizon and then sliced.

    The draw is deliberately not a function of `rounds`. If the shapes were sized to the requested
    horizon, a 40-round run would consume the generator in a different order from an 80-round run and
    would land on different noise, so the short run would not be the prefix of the long one and the
    two could not be pooled or compared. Drawing `CRN_MAX_ROUNDS` once and slicing makes a shorter
    horizon exactly a truncation of the longer trajectory's noise -- which is what lets the
    preregistered reduced tier remain a strict subset of the full tier rather than a different
    experiment. `check_crn_horizon_prefix` holds us to it.
    """
    if rounds > CRN_MAX_ROUNDS:
        raise ValueError(f"rounds={rounds} exceeds CRN_MAX_ROUNDS={CRN_MAX_ROUNDS}; raising the "
                         f"constant changes every stream and invalidates comparability with runs "
                         f"already on disk")
    rng = np.random.default_rng(CRN_BASE + seed)
    full = {"complaint": rng.random((CRN_MAX_ROUNDS, cfg.m, cfg.N_obs)),
            "refund": rng.random((CRN_MAX_ROUNDS, cfg.m, cfg.N_obs)),
            "audit": rng.random((CRN_MAX_ROUNDS, cfg.m, cfg.N_audit))}
    return {k: v[:rounds] for k, v in full.items()}


def signal_from_crn(cfg: M.Config, crn: dict, t: int, j: int, f: float, b: float) -> dict:
    """One merchant's observable signals for one round, from the shared uniforms."""
    th = float(np.clip(b + cfg.cs * f, 0.0, 1.0))
    adverse = crn["complaint"][t, j] < th
    D = int(adverse.sum())
    refunds = int((crn["refund"][t, j][adverse] < cfg.phi_refund).sum()) if D else 0
    ps = float(np.clip(cfg.psi0 + cfg.psi1 * f, 0.0, 1.0))
    C = int((crn["audit"][t, j] < ps).sum())
    return {"own_complaints": D, "own_complaint_rate": D / cfg.N_obs, "own_refunds": refunds,
            "own_audit_flags": C, "own_listings_audited": cfg.N_audit,
            "own_transactions_sampled": cfg.N_obs}


# ==================================================================================================
# stationary reputation tables -- cached on disk, identical for every cell with the same policy
# ==================================================================================================
_RBAR_MEM = {}


def rbar_for_policy(cfg: M.Config, policy) -> np.ndarray:
    key = tuple(float(x) for x in policy)
    if key in _RBAR_MEM:
        return _RBAR_MEM[key]
    CACHE.mkdir(parents=True, exist_ok=True)
    fp = CACHE / ("rbar_" + "_".join(f"{v:g}" for v in key) + ".npy")
    if fp.exists():
        arr = np.load(fp)
    else:
        arr = M.rbar_grid(cfg, key[0], key[1], key[2], key[3])
        tmp = fp.with_suffix(".npy.tmp")
        # np.save appends ".npy" to a *path* whose suffix is not already .npy, so saving to
        # "....npy.tmp" silently produces "....npy.tmp.npy" and the rename below then fails on a
        # file that was never created. Handing it an open handle suppresses that rewriting.
        with open(tmp, "wb") as fh:
            np.save(fh, arr)
            fh.flush()
            os.fsync(fh.fileno())
        tmp.replace(fp)
    _RBAR_MEM[key] = arr
    return arr


def displayed_table(cfg: M.Config, mkt: M.Market, rbar_m: np.ndarray, j: int,
                    f_rivals: np.ndarray) -> np.ndarray:
    """EVALUATOR-ONLY. Merchant j's exact stationary payoff against the rivals' last actions.

    This is the object arm A0 is handed and the object Proposition 12 is about. It needs every
    merchant's private type and the rivals' true current actions; no platform can build it.
    """
    v = np.empty(cfg.Nf)
    base = np.array(f_rivals, dtype=float)
    for i in range(cfg.Nf):
        fp = base.copy()
        fp[j] = cfg.fgrid[i]
        fidx = np.array([int(round(x * (cfg.Nf - 1))) for x in fp])
        v[i] = M.profile_outcome(cfg, mkt, rbar_m, fidx)["profit"][j]
    return v


# ==================================================================================================
# cells
# ==================================================================================================
@dataclass
class CellSpec:
    alias: str
    arm: str
    policy_name: str
    seed: int
    rounds: int = DEFAULT_ROUNDS

    @property
    def policy(self):
        return POLICY_NAMES[self.policy_name]

    @property
    def key(self) -> str:
        return f"{self.alias}__{self.arm}__{self.policy_name}__s{self.seed}"


def atomic_write_json_gz(path: Path, obj) -> int:
    """Write, fsync, then rename. A reader never sees a half-written cell."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    data = json.dumps(obj, ensure_ascii=False, default=_json_default).encode("utf-8")
    # fsync must be called on the handle the bytes were written through. Reopening the file
    # read-only to sync it raises EBADF on Windows, and -- worse on the platforms where it does not
    # raise -- syncs nothing, so the durability the rename is supposed to provide is imaginary.
    # mtime=0 keeps the container byte-identical for identical content, so a cell file can be
    # hashed and compared across machines.
    with open(tmp, "wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=6, mtime=0) as gz:
            gz.write(data)
        raw.flush()
        os.fsync(raw.fileno())
    tmp.replace(path)
    return path.stat().st_size


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def read_json_gz(path: Path):
    with gzip.open(path, "rb") as fh:
        return json.loads(fh.read().decode("utf-8"))


def cell_path(spec: CellSpec, root: Path = None) -> Path:
    return (root or RAW) / f"{spec.key}.json.gz"


def cell_done(spec: CellSpec, root: Path = None) -> bool:
    """A cell counts as done only if its file parses and carries the completion marker.

    Existence alone is not enough: a file truncated by a kill signal would otherwise be treated as a
    completed cell forever, and the missing rounds would silently shrink the denominator.
    """
    p = cell_path(spec, root)
    if not p.exists() or p.stat().st_size == 0:
        return False
    try:
        obj = read_json_gz(p)
    except (OSError, EOFError, json.JSONDecodeError, gzip.BadGzipFile):
        return False
    return bool(obj.get("complete")) and len(obj.get("rounds", [])) == obj.get("n_rounds")


# ==================================================================================================
# one decision
# ==================================================================================================
def one_decision(cfg, spec, transport, state, evaluator_only, counters, attempt_kind="initial"):
    """Ask the model once, repairing only parse failures. Returns (decision, record)."""
    msgs, meta = PR.build_messages(cfg, spec.arm, state)
    view = meta["view"]
    j, t = int(state["j"]), int(state["t"])
    TR.set_context(run_key=spec.key, decision_key=f"{spec.key}|t{t}|j{j}|{attempt_kind}",
                   seed=spec.seed, arm=spec.arm, policy=spec.policy_name,
                   model_alias=spec.alias, round=t, j=j, attempt_kind=attempt_kind)
    convo = list(msgs)
    repairs = 0
    last_err = None
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    n_attempts = 0
    while repairs <= SCHEMA_REPAIR_BUDGET:
        content, usage, k = transport(spec.alias, convo)
        n_attempts += k
        for kk in usage_total:
            usage_total[kk] += usage.get(kk, 0)
        try:
            dec = PR.parse_reply(content)
            break
        except PR.ParseFailure as exc:
            last_err = str(exc)
            repairs += 1
            counters["schema_repairs"] += 1
            if repairs > SCHEMA_REPAIR_BUDGET:
                raise CellFailure(
                    f"unparseable after {SCHEMA_REPAIR_BUDGET} schema repairs at "
                    f"{spec.key} t={t} j={j}: {last_err}") from None
            convo = convo + [{"role": "assistant", "content": str(content)[:800]},
                             {"role": "user", "content": PR.SCHEMA_REPAIR_MSG}]
    rec = IC.build_record(
        view=view, evaluator_only=evaluator_only,
        decision={"index": dec["index"], "f": dec["f"], "reasoning": dec["reasoning"][:200],
                  "attempt_kind": attempt_kind},
        msgs=msgs,
        meta={"round": t, "j": j, "arm": spec.arm, "model_alias": spec.alias,
              "policy": spec.policy_name, "schema_repairs": repairs,
              "transport_attempts": n_attempts, "usage": usage_total,
              "declassified": meta["declassified"], "notices": meta["notices"],
              "retry_offered": meta["retry_offered"]})
    # The full prompt text is reproducible from the recorded state by ha_prompts, and 307,200 copies
    # of a 3 kB system message is a 900 MB artefact nobody will read. The HASH is kept on every
    # decision; the TEXT is kept for the first decision of each cell as a spot-check anchor.
    if not (t == 1 and j == 0):
        rec.pop("prompt_messages", None)
    return dec, rec


# ==================================================================================================
# one cell
# ==================================================================================================
def run_cell(cfg: M.Config, spec: CellSpec, transport, out_root: Path = None,
             on_round=None) -> dict:
    mkt = M.draw_market(cfg, spec.seed)
    crn = draw_crn(cfg, spec.seed, spec.rounds)
    rbar_m = None
    if spec.arm == "A0_oracle":
        rbar_m = M.rbar_for_market(cfg, mkt, rbar_for_policy(cfg, spec.policy))
    kappa, tau, kappa_a, tau_a = (float(x) for x in spec.policy)

    r = np.full(cfg.m, R_INIT)
    f_prev = np.zeros(cfg.m)
    Q_last = cfg.Q0
    hist = [[] for _ in range(cfg.m)]
    counters = {"schema_repairs": 0, "economic_retries": 0, "economic_retries_changed": 0,
                "decisions": 0, "transport_attempts": 0}
    rounds_out, decisions_out = [], []
    t_start = time.time()

    for t in range(1, spec.rounds + 1):
        f_now = np.zeros(cfg.m)
        for j in range(cfg.m):
            state = {"t": t, "j": j, "q": mkt.q, "p": mkt.p, "b": mkt.b, "r": r,
                     "Q_last": Q_last, "policy": spec.policy, "seed": spec.seed,
                     "history": hist[j]}
            eo = {"true_f_all": [float(x) for x in f_prev], "seed": int(spec.seed),
                  "rival_quality": [float(v) for k, v in enumerate(mkt.q) if k != j],
                  "rival_baseline_complaint": [float(v) for k, v in enumerate(mkt.b) if k != j],
                  "rival_true_f": [float(v) for k, v in enumerate(f_prev) if k != j]}
            if spec.arm == "A0_oracle":
                v = displayed_table(cfg, mkt, rbar_m, j, f_prev)
                state["payoff_table"] = v
                eo["displayed_payoff_table"] = [float(x) for x in v]
                eo["displayed_argmax"] = int(np.argmax(v))
            dec, rec = one_decision(cfg, spec, transport, state, eo, counters, "initial")
            counters["decisions"] += 1
            counters["transport_attempts"] += rec["transport_attempts"]

            # ---- economic retry: a TREATMENT, offered only by arm A3 and only on the merchant's
            # own published record. Both decisions are kept; the second is the one that plays.
            if spec.arm == "A3_assist" and PR.retry_available(state):
                counters["economic_retries"] += 1
                state_retry = dict(state)
                state_retry["_retry"] = True
                dec2, rec2 = one_decision(cfg, spec, transport, state_retry, eo, counters, "economic_retry")
                rec2["decision"]["superseded_index"] = dec["index"]
                counters["decisions"] += 1
                counters["transport_attempts"] += rec2["transport_attempts"]
                if dec2["index"] != dec["index"]:
                    counters["economic_retries_changed"] += 1
                decisions_out.append(rec)
                decisions_out.append(rec2)
                dec = dec2
            else:
                decisions_out.append(rec)
            f_now[j] = dec["f"]

        # ---- realise the round -------------------------------------------------------------
        u = cfg.alpha * (mkt.q + (1 - mkt.q) * f_now) + cfg.beta * r - cfg.gamma * mkt.p
        den = math.exp(cfg.w0) + float(np.exp(u).sum())
        s = np.exp(u) / den
        Q = cfg.Q0 * math.exp(-cfg.lam * float(f_now.mean()))
        y = Q * s
        profit = cfg.margin_frac * mkt.p * y
        gmv = float(Q * float((mkt.p * s).sum()))

        r_next = np.zeros(cfg.m)
        sig_rows = []
        for j in range(cfg.m):
            z = signal_from_crn(cfg, crn, t - 1, j, float(f_now[j]), float(mkt.b[j]))
            pen = M.penalty_from_signal(kappa, tau, z["own_complaint_rate"], kappa_a, tau_a,
                                        z["own_audit_flags"] / cfg.N_audit)
            r_next[j] = M.reputation_update(cfg, float(r[j]), pen)
            row = {"round": t, "own_action": float(f_now[j]),
                   "own_sales_volume": float(y[j]), "own_profit": float(profit[j]),
                   "own_penalty_applied": float(pen),
                   "reputation_before": float(r[j]), "reputation_after": float(r_next[j]), **z}
            hist[j].append(row)
            sig_rows.append(row)

        rounds_out.append({
            "round": t, "f": [float(x) for x in f_now], "mean_f": float(f_now.mean()),
            "r_before": [float(x) for x in r], "r_after": [float(x) for x in r_next],
            "share": [float(x) for x in s], "Q": float(Q),
            "sales": [float(x) for x in y], "profit": [float(x) for x in profit], "GMV": gmv,
            "signals": sig_rows})
        if on_round:
            on_round(spec, t, rounds_out[-1])
        r, f_prev, Q_last = r_next, f_now, Q

    cell = {
        "schema_version": "ha-cell-1",
        "run_key": spec.key, "model_alias": spec.alias, "arm": spec.arm,
        "policy_name": spec.policy_name, "policy": [float(x) for x in spec.policy],
        "seed": int(spec.seed), "n_rounds": spec.rounds, "m": cfg.m,
        "r_init": R_INIT, "crn_base": CRN_BASE,
        "market_evaluator_only": {"q": [float(x) for x in mkt.q], "p": [float(x) for x in mkt.p],
                                  "b": [float(x) for x in mkt.b],
                                  "bidx": [int(x) for x in mkt.bidx]},
        "counters": counters,
        "rounds": rounds_out,
        "decisions": decisions_out,
        "wall_seconds": round(time.time() - t_start, 2),
        "spec": M.spec_dict(cfg),
        "prompt_surface": {"history_window": PR.HISTORY_WINDOW,
                           "rules_sha256": PR.prompt_hashes(cfg)["rules_block_sha256"]},
        "complete": True,
    }
    p = cell_path(spec, out_root)
    size = atomic_write_json_gz(p, cell)
    cell["_bytes"] = size
    cell["_path"] = str(p)
    return cell


# ==================================================================================================
# the matrix
# ==================================================================================================
def build_matrix(aliases, arms, policies, seeds, rounds=DEFAULT_ROUNDS) -> list:
    return [CellSpec(alias=a, arm=arm, policy_name=pn, seed=int(s), rounds=rounds)
            for a in aliases for arm in arms for pn in policies for s in seeds]


class Progress:
    """Append-only event log plus an atomically replaced state file."""

    def __init__(self, root: Path = None):
        self.root = root or PROGRESS
        self.root.mkdir(parents=True, exist_ok=True)
        self.events = self.root / "progress.jsonl"
        self.state = self.root / "state.json"
        self.errors = self.root / "errors.jsonl"

    def _append(self, path: Path, rec: dict):
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False, default=_json_default) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def event(self, **kw):
        self._append(self.events, {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **kw})

    def error(self, **kw):
        self._append(self.errors, {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **kw})

    def snapshot(self, obj: dict):
        tmp = self.state.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")
        tmp.replace(self.state)


def run_matrix(cfg: M.Config, specs, transport, progress: Progress, out_root: Path = None,
               stop_on_fail: bool = False) -> dict:
    todo = [s for s in specs if not cell_done(s, out_root)]
    done_already = len(specs) - len(todo)
    summary = {"n_cells": len(specs), "skipped_complete": done_already,
               "attempted": 0, "completed": 0, "failed": 0, "failures": [],
               "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    progress.event(kind="matrix_start", n_cells=len(specs), todo=len(todo),
                   skipped_complete=done_already)
    t0 = time.time()
    for i, spec in enumerate(todo, 1):
        summary["attempted"] += 1
        last_exc = None
        for attempt in range(CELL_RETRY_BUDGET + 1):
            try:
                cell = run_cell(cfg, spec, transport, out_root)
                summary["completed"] += 1
                progress.event(kind="cell_done", run_key=spec.key, i=i, of=len(todo),
                               bytes=cell["_bytes"], wall_s=cell["wall_seconds"],
                               counters=cell["counters"], cell_attempt=attempt + 1)
                last_exc = None
                break
            except (TR.ConfigError, TR.ModelIdentityError) as exc:
                # Configuration is not flaky. Retrying it would burn the budget and bury the cause.
                progress.error(kind="fatal_config", run_key=spec.key, error=TR.redact(exc))
                raise
            except Exception as exc:                                        # noqa: BLE001
                last_exc = exc
                progress.error(kind="cell_attempt_failed", run_key=spec.key,
                               cell_attempt=attempt + 1, error_class=type(exc).__name__,
                               error=TR.redact(exc),
                               traceback=TR.redact(traceback.format_exc())[:2000])
        if last_exc is not None:
            summary["failed"] += 1
            summary["failures"].append({"run_key": spec.key, "error": TR.redact(last_exc)})
            progress.event(kind="cell_failed", run_key=spec.key,
                           error_class=type(last_exc).__name__)
            if stop_on_fail:
                break
        summary["elapsed_s"] = round(time.time() - t0, 1)
        summary["transport"] = transport.stats()
        progress.snapshot(summary)
    summary["elapsed_s"] = round(time.time() - t0, 1)
    summary["transport"] = transport.stats()
    summary["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    progress.snapshot(summary)
    progress.event(kind="matrix_end", **{k: summary[k] for k in
                                         ("completed", "failed", "skipped_complete", "elapsed_s")})
    return summary


# ==================================================================================================
# invariants -- checked offline, on mock data, before any paid call
# ==================================================================================================
def check_crn_monotone(cfg: M.Config, seed: int = 70000, rounds: int = 5) -> dict:
    """With the uniforms fixed, a higher f can only weakly raise the complaint and audit counts."""
    crn = draw_crn(cfg, seed, rounds)
    bad = []
    for j in range(cfg.m):
        for t in range(rounds):
            prev_D, prev_C = -1, -1
            for f in np.linspace(0, 1, 21):
                z = signal_from_crn(cfg, crn, t, j, float(f), 0.05)
                if z["own_complaints"] < prev_D or z["own_audit_flags"] < prev_C:
                    bad.append((j, t, float(f), z["own_complaints"], prev_D,
                                z["own_audit_flags"], prev_C))
                prev_D, prev_C = z["own_complaints"], z["own_audit_flags"]
    return {"check": "crn_monotone_in_f", "pass": not bad, "violations": bad[:10],
            "n_violations": len(bad)}


def check_crn_shared(cfg: M.Config, seed: int = 70000) -> dict:
    """The latent draws depend on the market seed alone -- not on arm, policy or model."""
    a = draw_crn(cfg, seed, 3)
    b = draw_crn(cfg, seed, 3)
    c = draw_crn(cfg, seed + 1, 3)
    same = all(np.array_equal(a[k], b[k]) for k in a)
    diff = any(not np.array_equal(a[k], c[k]) for k in a)
    return {"check": "crn_shared_across_arms", "pass": bool(same and diff),
            "identical_for_same_seed": bool(same), "differs_across_seeds": bool(diff)}


def check_signal_distribution(cfg: M.Config, seed: int = 70000, rounds: int = 400) -> dict:
    """The CRN construction must reproduce the Binomial means the theory and the prompt state.

    This draws its own uniforms rather than calling `draw_crn`, for two reasons: the sample it needs
    is longer than any experimental horizon, and reusing the experiment's own stream to validate the
    experiment's distribution would cap the test at `CRN_MAX_ROUNDS`. The mapping under test is
    `signal_from_crn`, which is shared; only the source of the uniforms differs.
    """
    rng = np.random.default_rng(CRN_BASE + seed)
    crn = {"complaint": rng.random((rounds, cfg.m, cfg.N_obs)),
           "refund": rng.random((rounds, cfg.m, cfg.N_obs)),
           "audit": rng.random((rounds, cfg.m, cfg.N_audit))}
    rows = []
    ok = True
    for f in (0.0, 0.25, 0.5, 1.0):
        b = 0.05
        D = [signal_from_crn(cfg, crn, t, 0, f, b)["own_complaints"] for t in range(rounds)]
        C = [signal_from_crn(cfg, crn, t, 0, f, b)["own_audit_flags"] for t in range(rounds)]
        exp_D = cfg.N_obs * min(1.0, b + cfg.cs * f)
        exp_C = cfg.N_audit * min(1.0, cfg.psi0 + cfg.psi1 * f)
        se_D = math.sqrt(max(1e-12, exp_D * (1 - min(1.0, b + cfg.cs * f)) / rounds))
        good = abs(np.mean(D) - exp_D) < max(0.2, 4 * se_D) and abs(np.mean(C) - exp_C) < 0.25
        ok &= good
        rows.append({"f": f, "mean_D": round(float(np.mean(D)), 3), "expected_D": round(exp_D, 3),
                     "mean_C": round(float(np.mean(C)), 3), "expected_C": round(exp_C, 3),
                     "ok": bool(good)})
    return {"check": "signal_distribution_matches_spec", "pass": bool(ok), "rows": rows}


def check_crn_horizon_prefix(cfg: M.Config, seed: int = 70000) -> dict:
    """A 40-round draw must be exactly the first 40 rounds of an 80-round draw.

    This is what makes the preregistered reduced tier a truncation of the full tier rather than an
    independent experiment on different noise. Without it, choosing the short tier for budget reasons
    would silently break the pairing that every comparison in this study rests on.
    """
    short, long = draw_crn(cfg, seed, 40), draw_crn(cfg, seed, 80)
    same = {k: bool(np.array_equal(short[k], long[k][:40])) for k in short}
    return {"check": "crn_horizon_prefix", "pass": all(same.values()),
            "detail": {"streams": same, "short_rounds": 40, "long_rounds": 80,
                       "crn_max_rounds": CRN_MAX_ROUNDS}}


def run_invariants(cfg: M.Config = None) -> dict:
    cfg = cfg or M.Config()
    checks = [check_crn_shared(cfg), check_crn_monotone(cfg), check_signal_distribution(cfg),
              check_crn_horizon_prefix(cfg)]
    return {"all_pass": all(c["pass"] for c in checks), "checks": checks}


if __name__ == "__main__":
    r = run_invariants()
    for c in r["checks"]:
        print(f"[{'PASS' if c['pass'] else 'FAIL'}] {c['check']}")
        if not c["pass"]:
            print("   ", json.dumps(c, default=str)[:800])
    print("all_pass:", r["all_pass"])
    sys.exit(0 if r["all_pass"] else 1)
