"""to_run.py -- the only runner in this package.

    python azure_handoff/to_run.py                  start
    python azure_handoff/to_run.py --resume         continue an interrupted run
    python azure_handoff/to_run.py --validate-only  validate what is already on disk
    python azure_handoff/to_run.py --package-results build the final delivery archive

WHAT THIS RUNS
    A fresh Azure replication of the Balance-9 P3 llama matrix:
    2 policies (P_GMV, P_robust) x 4 LLM interfaces (U, H, R, G) x 60 frozen seeds (9000-9059)
    x 80 rounds x 4 merchants = 480 runs. Nothing else.

WHAT IT DOES NOT DO
    It does not touch, read for writing, append to, or merge with any OpenRouter result. The Azure
    raw file is a separate file with its own name, and the two corpora are never concatenated by
    anything in this package. The OpenRouter llama cells stay exactly as incomplete as they are; the
    Azure cells are a standalone replication that must be reported as such.

HOW IT STAYS THE SAME EXPERIMENT
    Every module in `code/` is a byte-identical copy of the file that produced the OpenRouter half,
    verified against `manifests/vendored_sha256.json` at startup. Exactly one function is replaced --
    `balance9_runner.transport_call` -- and the replacement has the same signature, the same retry
    shape, the same non-retryable semantics and the same return contract. The prompts, the payoff
    tables, the RNG streams, the tie rule, the economic-retry and schema-repair budgets, the override
    rule and the record schema are all the frozen ones.

    The frozen gates run BEFORE the first call, exactly as `balance9_p3.py` requires: the ten runner
    gates (G-R1..G-R10) and the design gates parsed out of BALANCE9_PREREGISTRATION.md. None of them
    makes a network request.

THE LAUNCH ORDER IS FROZEN AND SEQUENTIAL
    Preregistration 5.1 freezes wave 1 = M, O, U, H; wave 2 = R; wave 3 = G, "so that a truncated run
    is a preregistered subset rather than a choice made under time pressure". M and O call no model
    and are not part of this package, so the LLM restriction of that order is U, H -> R -> G, and it
    is enforced, not documented: wave n+1 does not start until every cell of wave n has all 60 seeds.
    If the budget runs out mid-way, what you have is a preregistered prefix.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tarfile
import time
import traceback
from pathlib import Path

HERE = Path(os.path.dirname(os.path.abspath(__file__)))
CODE = HERE / "code"
INPUT = HERE / "input_data"
MANIFESTS = HERE / "manifests"
RESULTS = HERE / "results"
RAW_DIR, ATT_DIR = RESULTS / "raw", RESULTS / "attempts"
LOG_DIR, SUM_DIR = RESULTS / "logs", RESULTS / "summaries"
VAL_DIR, DELIV = RESULTS / "validation", RESULTS / "final_delivery"

# The write guard in balance9_runner.append_jsonl refuses any filename outside the balance9_ prefix,
# which is why the Azure raw file is named this way rather than "azure_raw.jsonl".
RAW = RAW_DIR / "balance9_azure_p3_llama_raw.jsonl"
ATTEMPTS = ATT_DIR / "balance9_azure_attempts.jsonl"
PROGRESS = LOG_DIR / "progress.json"
RUNLOG = LOG_DIR / "run.log"

sys.path.insert(0, str(CODE))

# -------------------------------------------------------------------------------------------------
# the design, as this runner believes it -- every line is re-checked against the frozen document
# -------------------------------------------------------------------------------------------------
MODEL_ALIAS = "llama"
POLICIES = ("P_GMV", "P_robust")
WAVES = ((1, ("U", "H")), (2, ("R",)), (3, ("G",)))     # the LLM restriction of preregistration 5.1
SEED_BLOCK = "B9-P3"
ROUNDS = 80
N_SEEDS = 60
N_MERCHANTS = 4

_G: list[tuple[str, bool]] = []
# package-own files found modified by gate A-V1; non-blocking, but carried into every artefact so the
# returned results can never quietly claim they came from unmodified scaffolding
_LOCAL_EDITS: list[str] = []


def ck(label, cond, detail=""):
    _G.append((label, bool(cond)))
    txt = ("  ok   " if cond else "  FAIL ") + label
    if detail and not cond:
        txt += f"   [{str(detail)[:200]}]"
    log(txt)
    return bool(cond)


def log(msg):
    line = str(msg).encode("ascii", "replace").decode("ascii")
    print(line, flush=True)
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(RUNLOG, "a", encoding="utf-8") as fh:
            fh.write(time.strftime("%Y-%m-%dT%H:%M:%S ") + line + "\n")
    except Exception:                                    # noqa: BLE001
        pass


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# -------------------------------------------------------------------------------------------------
# gate A-V: the vendored code is the code that produced the OpenRouter half
# -------------------------------------------------------------------------------------------------
def hash_audit():
    """Compare every file in the build manifest against its recorded hash.

    Returns (frozen_bad, local_bad, n_frozen). Split out from the gate so that `--validate-only` and
    `--package-results` can report the same fact without needing the start-up gates to have run: the
    delivered results should state whether the runner was modified regardless of which command
    produced them.
    """
    man = json.loads((MANIFESTS / "vendored_sha256.json").read_text(encoding="utf-8"))
    prov = man.get("provenance", {})
    frozen_bad, local_bad, n_frozen = [], [], 0
    for rel, want in sorted(man["files"].items()):
        is_frozen = bool(prov.get(rel, {}).get("frozen", True))    # unknown -> strict
        n_frozen += is_frozen
        p = HERE / rel
        if not p.exists():
            why = f"{rel}: MISSING"
        elif sha256_file(p) != want:
            why = f"{rel}: sha256 differs"
        else:
            continue
        (frozen_bad if is_frozen else local_bad).append(why)
    return frozen_bad, local_bad, n_frozen


def gate_vendored_integrity():
    """Every frozen module hashes to the value recorded when the package was built.

    This is the load-bearing check of the whole package. If someone "fixes" a prompt or a constant
    on the collaborator's machine, the Azure cells stop being a replication of the OpenRouter cells
    and become a new experiment with the old experiment's name. A byte hash is the only statement
    about that which cannot be made accidentally true.

    The check is deliberately split by severity, because the two kinds of file here carry different
    claims:

      frozen=True   the 16 vendored files -- experiment code, prompts, catalogue, preregistration.
                    These came byte-identical from the repository that produced the OpenRouter half.
                    A mismatch means the run would no longer be a replication, so it ABORTS.

      frozen=False  files written for this package -- to_run.py, azure_transport.py, prior_seeds.json,
                    the offline tests, requirements.txt. These are scaffolding around the experiment,
                    not the experiment. If the collaborator has to patch a path bug or an Azure route
                    quirk in my transport code at 2am, blocking the run over it would be theatre, not
                    integrity: the science is unchanged. So a mismatch is RECORDED -- printed, written
                    to run.log, carried into progress.json, validation.json and the delivery manifest
                    -- and the run proceeds. Recorded, never silent; the returned results say plainly
                    that the runner was locally modified, and I can diff it on receipt.

    A file the manifest cannot classify is treated as frozen. Unknown provenance gets the strict rule.
    """
    frozen_bad, local_bad, n_frozen = hash_audit()
    _LOCAL_EDITS.clear()
    _LOCAL_EDITS.extend(local_bad)
    ok = ck(f"A-V1 all {n_frozen} frozen vendored files are byte-identical to the originals that "
            f"produced the OpenRouter half", not frozen_bad, "; ".join(frozen_bad[:6]))
    if local_bad:
        log(f"  NOTE  {len(local_bad)} package-own file(s) differ from the build manifest: "
            f"{', '.join(local_bad)}")
        log( "        These are this package's own scaffolding, not frozen experiment code, so the "
             "run continues.")
        log( "        The modification is recorded in progress.json and in the returned results.")
    return ok


def gate_catalog_placed():
    """`hetero.py` reads its catalogue from its OWN directory at import time.

    The catalogue is shipped once, in input_data/, and copied to code/data/ here rather than being
    duplicated in the repository, so there is exactly one file whose hash means anything. The copy is
    verified against the manifest, logged, and refused if it disagrees -- it is a deterministic
    placement step, not a fallback.
    """
    src = INPUT / "catalog_types.json"
    dst = CODE / "data" / "catalog_types.json"
    man = json.loads((MANIFESTS / "vendored_sha256.json").read_text(encoding="utf-8"))
    want = man["files"]["input_data/catalog_types.json"]
    if sha256_file(src) != want:
        return ck("A-V2 the shipped catalogue matches the manifest", False, "input_data copy differs")
    if not dst.exists() or sha256_file(dst) != want:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())
        log(f"  ..   placed catalog_types.json into {dst.relative_to(HERE)}")
    return ck("A-V2 catalog_types.json is in place for hetero.py and matches the manifest",
              sha256_file(dst) == want)


# -------------------------------------------------------------------------------------------------
# gate A-D: the design, re-derived from BALANCE9_PREREGISTRATION.md
# -------------------------------------------------------------------------------------------------
def parse_launch_order(text):
    """The frozen launch order, anchored on the sentence rather than on a line number."""
    i = text.find("**Launch order, frozen now**")
    if i < 0:
        return None
    tail = text[i:i + 700]
    waves = []
    for m in re.finditer(r"^\s*\d+\.\s*wave\s*(\d+)\s*[—-]\s*(.*)$", tail, re.M):
        arms = tuple(re.findall(r"`([A-Z])`", m.group(2)))
        if arms:
            waves.append((int(m.group(1)), arms))
    return tuple(waves) or None


def gate_design(R, FZ):
    doc = (INPUT / "BALANCE9_PREREGISTRATION.md").read_text(encoding="utf-8")

    parsed = parse_launch_order(doc)
    full = ((1, ("M", "O", "U", "H")), (2, ("R",)), (3, ("G",)))
    ck("A-D1 the launch order in the document is the frozen wave order",
       parsed == full, f"document says {parsed}")
    # This package runs the LLM restriction of that order. Proving it is a RESTRICTION -- same waves,
    # same relative order, only the no-LLM arms dropped -- is what keeps "preregistered subset" true.
    ck("A-D2 the waves run here are the launch order with the no-LLM arms removed, order preserved",
       WAVES == tuple((n, tuple(a for a in arms if a not in R.NOLLM_INTERFACES))
                      for n, arms in full if any(a not in R.NOLLM_INTERFACES for a in arms)),
       str(WAVES))
    ck("A-D3 the four interfaces run here are exactly the LLM interfaces of the design",
       tuple(a for _, w in WAVES for a in w) == ("U", "H", "R", "G"))

    row = next((l for l in doc.splitlines() if l.startswith("| Models |")), "")
    ck("A-D4 the llama model id is the preregistered one",
       f"`{MODEL_ALIAS}` = `{R.MODELS[MODEL_ALIAS]}`" in row
       and R.MODELS[MODEL_ALIAS] == "meta-llama/llama-3.3-70b-instruct", row[:120])

    prow = next((l for l in doc.splitlines() if l.startswith("| Policies |")), "")
    ck("A-D5a P_GMV = (0.5, 0.20) in the document and in the runner",
       "`P_GMV` = (κ=0.5, τ=0.20)" in prow and R.POLICIES["P_GMV"] == (0.5, 0.20), prow[:160])
    ck("A-D5b P_robust = (4.0, 0.30) in the document and in the runner",
       "`P_robust` = (κ=**4.0**, τ=**0.30**)" in prow
       and R.POLICIES["P_robust"] == (4.0, 0.30), prow[:160])
    ck("A-D5c P_sep is NOT substituted into this experiment",
       "`P_sep`" in prow and "are **not** substituted" in prow and "P_sep" not in POLICIES)

    lo, hi = FZ.SEED_BLOCKS[SEED_BLOCK]
    ck(f"A-D6 the seed block is {SEED_BLOCK} = {lo}-{hi}, exactly {N_SEEDS} seeds",
       hi - lo + 1 == N_SEEDS and f"| **{SEED_BLOCK}** | {lo}–{hi} |" in doc
       and (lo, hi) == (9000, 9059))
    ck(f"A-D7 the design is {N_SEEDS} fresh paired seeds x {ROUNDS} rounds x {N_MERCHANTS} merchants",
       f"{N_SEEDS} fresh paired market seeds × {ROUNDS} rounds × {N_MERCHANTS} merchants" in doc
       and R.ROUNDS == ROUNDS)

    rrow = next((l for l in doc.splitlines() if l.startswith("| Retry budget |")), "")
    ck("A-D8 the retry budget is 2 economic retries at eta = 0.01 with 2 schema repairs",
       "2 economic retries" in rrow and "η = 0.01" in rrow and "schema repair 2" in rrow
       and (R.MAX_ECON_RETRY, R.SCHEMA_REPAIR, R.ETA) == (2, 2, 0.01), rrow[:160])
    ck("A-D9 prompts come from balance9_prompts.p3_prompt and nowhere else",
       "`balance9_prompts.p3_prompt`" in doc and R.PR.p3_prompt.__module__ == "balance9_prompts")
    ck("A-D10 R verifies without override; G verifies and overrides to the argmax",
       R.GUARD_SPEC == {"R": (R.ETA, None), "G": (R.ETA, "argmax")}, str(R.GUARD_SPEC))
    ck("A-D11 M and O are the no-LLM arms and are not run by this package",
       R.NOLLM_INTERFACES == ("M", "O")
       and not set(a for _, w in WAVES for a in w) & set(R.NOLLM_INTERFACES))

    # the prompt surface must hash to the published values
    pub = R.PR.published_hashes()
    frozen = json.loads((INPUT / "balance9_prompt_hashes.json").read_text(encoding="utf-8"))["hashes"]
    diff = sorted(k for k in set(pub) | set(frozen) if pub.get(k) != frozen.get(k))
    ck(f"A-D12 all {len(frozen)} published prompt hashes reproduce exactly", not diff,
       f"{len(diff)} differ, e.g. {diff[:4]}")
    return all(ok for _, ok in _G)


# -------------------------------------------------------------------------------------------------
# gate A-T: the transport substitution is total, and survived the frozen gates
# -------------------------------------------------------------------------------------------------
def gate_transport_substitution(R, AT, tx):
    """The single most dangerous thing in this package, so it gets its own gate.

    `balance9_runner` keeps a SECOND handle on the original transport in `transport_call_ref[0]`,
    and `gate_state_invariance` restores from it in a finally-block. A patch applied only to the
    module global is therefore silently reverted by the runner's own self-test, and every subsequent
    call would go to the dead OpenRouter key. Both handles are patched, and the patch is re-asserted
    after the self-test rather than assumed.
    """
    ck("A-T1 the Azure retry shape equals the frozen transport retry shape",
       AT.TRANSPORT_RETRIES == R.TRANSPORT_RETRIES == 6)
    ck("A-T2 balance9_runner.transport_call is the Azure transport",
       R.transport_call is tx, str(R.transport_call))
    ck("A-T3 the runner's second handle transport_call_ref[0] is also the Azure transport",
       R.transport_call_ref[0] is tx, str(R.transport_call_ref[0]))
    ck("A-T4 phase2_llm.openrouter_call is unreachable from the patched path",
       R.transport_call.__module__ != "balance9_runner")
    return all(ok for _, ok in _G)


# -------------------------------------------------------------------------------------------------
# durable append -- byte-identical to balance9_runner.append_jsonl, plus flush+fsync
# -------------------------------------------------------------------------------------------------
def make_durable_append(R):
    """The frozen writer is append-only but not fsync'd; a power cut can leave a torn final line.

    The replacement writes exactly the same bytes -- `json.dumps(r, ensure_ascii=False) + "\\n"`, the
    same guard, the same open mode -- and adds flush + fsync. Offline test T7 asserts the two writers
    produce identical files, so this is a durability change and not a format change.
    """
    def durable_append_jsonl(path: Path, rows):
        path = Path(path)
        name = path.name
        if not name.startswith("balance9_"):
            raise ValueError(f"refusing to write outside the balance9_ prefix: {name}")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
    return durable_append_jsonl


# -------------------------------------------------------------------------------------------------
# per-decision context, so every attempt record knows which decision it belongs to
# -------------------------------------------------------------------------------------------------
def install_decision_context(R, AT):
    """Wrap `decide` so the transport can stamp identifiers onto every attempt record.

    A pass-through wrapper, not an edit: the frozen function is called with the frozen arguments and
    its return value is returned unchanged. The context is a ContextVar, and `decide` and
    `transport_call` always run in the same worker thread of the same ThreadPoolExecutor, so the
    value a transport reads is always the one its own decision set.

    WHICH IDENTIFIERS ARE AVAILABLE, AND WHY NOT run_id. `simulate_run` mints `run_id` internally and
    never passes it down; `decide` receives only (interface, policy, state, seed), and `state` carries
    the round `t` and the merchant `j`. Reaching in for `run_id` would mean editing frozen code, so
    instead the attempt log keys on (run_key, round, j), which is unique by construction, and
    `results/logs/run_id_map.json` carries run_key -> run_id read back out of the raw footers. Both
    identifiers are therefore recoverable and neither required touching the runner.
    """
    inner = R.decide
    policy_name = {v: k for k, v in R.POLICIES.items()}

    def decide_with_context(agent, model_id, cfg, interface, policy, state, v, seed,
                            prompt_fn=None):
        # `policy` arrives as the (kappa, tau) tuple; the name is what the records are keyed by.
        pol = policy_name.get(tuple(policy) if isinstance(policy, (list, tuple)) else policy,
                              str(policy))
        rk = R.run_key(MODEL_ALIAS, interface, pol, seed)
        t, j = state.get("t"), state.get("j")
        AT.set_context(run_key=rk, decision_key=f"{rk}:{t}:{j}", seed=seed, interface=interface,
                       policy=pol, model_alias=MODEL_ALIAS, round=t, j=j)
        return inner(agent, model_id, cfg, interface, policy, state, v, seed, prompt_fn=prompt_fn)

    R.decide = decide_with_context
    return inner


# -------------------------------------------------------------------------------------------------
# progress
# -------------------------------------------------------------------------------------------------
def seeds():
    return list(range(9000, 9000 + N_SEEDS))


def cells_of(arms):
    return [(MODEL_ALIAS, i, p) for i in arms for p in POLICIES]


def scan(path=RAW):
    """Completed seeds per cell and technical totals, read from the raw file's footers.

    Footers rather than headers, because a footer is what `balance9_runner.run_cell` itself treats as
    "this seed is done"; two different definitions of complete in one pipeline is how a duplicate
    seed gets written.
    """
    done, tot = {}, dict(footers=0, rounds=0, calls=0, transport_attempts=0, transport_retries=0,
                         schema=0, econ=0, overrides=0, reruns=0, wall_s=0.0)
    if not Path(path).exists():
        return done, tot
    with open(path, "rb") as fh:
        for bline in fh:
            if not bline.endswith(b"\n"):
                break
            line = bline.decode("utf-8", "replace")
            if '"run_footer"' not in line:
                continue
            try:
                d = json.loads(line)
            except Exception:                            # noqa: BLE001
                continue
            if d.get("kind") != "run_footer":
                continue
            done.setdefault((d["model_alias"], d["interface"], d["policy"]), set()).add(int(d["seed"]))
            tot["footers"] += 1
            tot["rounds"] += int(d.get("rounds", 0))
            tot["calls"] += int(d.get("calls", 0))
            tot["transport_attempts"] += int(d.get("transport_attempts", 0))
            tot["transport_retries"] += max(0, int(d.get("transport_attempts", 0))
                                            - int(d.get("calls", 0)))
            tot["schema"] += int(d.get("schema_repairs", 0))
            tot["econ"] += int(d.get("economic_retries", 0))
            tot["overrides"] += int(d.get("overrides", 0))
            tot["reruns"] += max(0, int(d.get("seed_attempts", 1)) - 1)
            tot["wall_s"] += float(d.get("wall_s", 0.0) or 0.0)
    return done, tot


def wave_status(no, arms, done):
    want = set(seeds())
    rows = [dict(cell="|".join(c), have=len(done.get(c, set()) & want), want=len(want),
                 missing=sorted(want - done.get(c, set()))[:5]) for c in cells_of(arms)]
    return dict(wave=no, arms=list(arms), cells=rows,
                complete=all(r["have"] == r["want"] for r in rows))


def write_progress(gates_ok, azure_public=None, extra=None):
    done, tot = scan()
    waves = [wave_status(n, a, done) for n, a in WAVES]
    art = dict(
        experiment="B9-P3-AZURE", provider="azure-ai-foundry", model_alias=MODEL_ALIAS,
        seed_block=SEED_BLOCK, seeds=[seeds()[0], seeds()[-1]], n_seeds=N_SEEDS, rounds=ROUNDS,
        merchants=N_MERCHANTS, policies=list(POLICIES),
        launch_order=[[n, list(a)] for n, a in WAVES],
        runs_target=len(POLICIES) * 4 * N_SEEDS, runs_done=tot["footers"],
        gates_passed=bool(gates_ok), n_gates=len(_G),
        runner_locally_modified=bool(_LOCAL_EDITS), local_modifications=list(_LOCAL_EDITS),
        cells_total=sum(len(cells_of(a)) for _, a in WAVES),
        cells_complete=sum(1 for w in waves for r in w["cells"] if r["have"] == r["want"]),
        waves=waves, totals=tot, azure=azure_public,
        raw_path=str(RAW.relative_to(HERE)),
        raw_bytes=RAW.stat().st_size if RAW.exists() else 0,
        attempts_path=str(ATTEMPTS.relative_to(HERE)),
        attempts_bytes=ATTEMPTS.stat().st_size if ATTEMPTS.exists() else 0,
        updated=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    if extra:
        art.update(extra)
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    tmp = PROGRESS.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(art, indent=1, sort_keys=True), encoding="utf-8")
    tmp.replace(PROGRESS)                                # atomic: never a half-written progress file
    return art


# -------------------------------------------------------------------------------------------------
# summaries
# -------------------------------------------------------------------------------------------------
def build_summaries():
    """Per-run and per-cell summaries, plus calls/tokens/latency/cost and retry/override totals.

    Read back out of the written raw file rather than accumulated in memory, so the summary describes
    the data that exists rather than the run that was intended.
    """
    SUM_DIR.mkdir(parents=True, exist_ok=True)
    per_run, per_cell, id_map = [], {}, {}
    if RAW.exists():
        with open(RAW, encoding="utf-8") as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except Exception:                        # noqa: BLE001
                    continue
                if d.get("kind") == "run_footer":
                    id_map[d.get("run_key")] = d.get("run_id")
                    per_run.append({k: d.get(k) for k in
                                    ("run_id", "run_key", "seed", "model_alias", "interface",
                                     "policy", "rounds", "seed_attempts", "calls",
                                     "transport_attempts", "schema_repairs", "economic_retries",
                                     "overrides", "realized_identity_max_err", "wall_s")})
                elif d.get("kind") == "round":
                    c = per_cell.setdefault(f"{d['model_alias']}|{d['interface']}|{d['policy']}",
                                            dict(rounds=0, decisions=0, calls=0, prompt_tokens=0,
                                                 completion_tokens=0, latency_s=0.0, n_latency=0,
                                                 verif_failed=0, overrides=0, followed_mark=0,
                                                 mark_defined=0, seeds=set()))
                    c["rounds"] += 1
                    c["seeds"].add(d["seed"])
                    for m in d.get("merchants") or []:
                        c["decisions"] += 1
                        c["calls"] += int(m.get("calls", 0))
                        c["verif_failed"] += int(bool(m.get("verification_failed")))
                        c["overrides"] += int(bool(m.get("override")))
                        if m.get("followed_mark") is not None:
                            c["mark_defined"] += 1
                            c["followed_mark"] += int(bool(m["followed_mark"]))
                        for a in m.get("attempts") or []:
                            c["prompt_tokens"] += int(a.get("prompt_tokens") or 0)
                            c["completion_tokens"] += int(a.get("completion_tokens") or 0)
                            if isinstance(a.get("latency_s"), (int, float)):
                                c["latency_s"] += a["latency_s"]
                                c["n_latency"] += 1

    cells = {}
    for k, c in sorted(per_cell.items()):
        tt = c["prompt_tokens"] + c["completion_tokens"]
        cells[k] = dict(
            runs=len(c["seeds"]), rounds=c["rounds"], decisions=c["decisions"], calls=c["calls"],
            prompt_tokens=c["prompt_tokens"], completion_tokens=c["completion_tokens"],
            total_tokens=tt,
            mean_tokens_per_call=round(tt / c["calls"], 2) if c["calls"] else 0.0,
            mean_latency_s=round(c["latency_s"] / c["n_latency"], 3) if c["n_latency"] else 0.0,
            verification_failed=c["verif_failed"], overrides=c["overrides"],
            followed_mark=c["followed_mark"], mark_defined=c["mark_defined"],
        )
    (SUM_DIR / "per_run.json").write_text(
        json.dumps(sorted(per_run, key=lambda r: (r["interface"], r["policy"], r["seed"])),
                   indent=1), encoding="utf-8")
    (SUM_DIR / "per_cell.json").write_text(json.dumps(cells, indent=1, sort_keys=True),
                                           encoding="utf-8")
    # The attempt log keys on (run_key, round, j) because `decide` never sees `run_id`; this map is
    # what makes an attempt record joinable to the run_id in the raw header and footer.
    (LOG_DIR / "run_id_map.json").write_text(
        json.dumps(dict(join_key=["run_key", "round", "j"],
                        note="attempts[].decision_key == f'{run_key}:{round}:{j}'; the raw record's "
                             "merchants[].decision_id == f'{run_id}:{round}:{j}'",
                        run_key_to_run_id=dict(sorted(id_map.items()))), indent=1),
        encoding="utf-8")

    # attempts / retries / HTTP / cost
    att = dict(records=0, ok=0, failed=0, by_status={}, by_error={}, retries=0,
               prompt_tokens=0, completion_tokens=0, latency_s=0.0, n_latency=0,
               content_filtered=0)
    if ATTEMPTS.exists():
        with open(ATTEMPTS, encoding="utf-8") as fh:
            for line in fh:
                try:
                    a = json.loads(line)
                except Exception:                        # noqa: BLE001
                    continue
                att["records"] += 1
                if a.get("ok"):
                    att["ok"] += 1
                    att["prompt_tokens"] += int(a.get("prompt_tokens") or 0)
                    att["completion_tokens"] += int(a.get("completion_tokens") or 0)
                else:
                    att["failed"] += 1
                    att["by_error"][a.get("error_class") or "?"] = \
                        att["by_error"].get(a.get("error_class") or "?", 0) + 1
                    if "content_filter" in str(a.get("error", "")):
                        att["content_filtered"] += 1
                s = str(a.get("http_status"))
                att["by_status"][s] = att["by_status"].get(s, 0) + 1
                if int(a.get("transport_attempt") or 1) > 1:
                    att["retries"] += 1
                if isinstance(a.get("latency_s"), (int, float)):
                    att["latency_s"] += a["latency_s"]
                    att["n_latency"] += 1
    tt = att["prompt_tokens"] + att["completion_tokens"]
    rate_in = float(os.environ.get("AZURE_PRICE_PER_1M_INPUT", "0") or 0)
    rate_out = float(os.environ.get("AZURE_PRICE_PER_1M_OUTPUT", "0") or 0)
    usage = dict(
        successful_calls=att["ok"], transport_attempts=att["records"],
        transport_retries=att["retries"], failed_attempts=att["failed"],
        http_status_counts=att["by_status"], error_class_counts=att["by_error"],
        content_filtered_completions=att["content_filtered"],
        prompt_tokens=att["prompt_tokens"], completion_tokens=att["completion_tokens"],
        total_tokens=tt,
        mean_latency_s=round(att["latency_s"] / att["n_latency"], 3) if att["n_latency"] else 0.0,
        price_per_1m_input=rate_in or None, price_per_1m_output=rate_out or None,
        cost_usd=(round(att["prompt_tokens"] / 1e6 * rate_in
                        + att["completion_tokens"] / 1e6 * rate_out, 2)
                  if (rate_in or rate_out) else None),
        cost_note=("set AZURE_PRICE_PER_1M_INPUT and AZURE_PRICE_PER_1M_OUTPUT from your own Azure "
                   "rate card to have this filled in; it is left null rather than guessed"),
    )
    (SUM_DIR / "usage_and_cost.json").write_text(json.dumps(usage, indent=1), encoding="utf-8")

    ovr = {k: dict(verification_failed=v["verification_failed"], overrides=v["overrides"],
                   decisions=v["decisions"],
                   override_rate=round(v["overrides"] / v["decisions"], 6) if v["decisions"] else 0.0)
           for k, v in cells.items()}
    retries = {r["run_key"]: dict(seed_attempts=r["seed_attempts"],
                                  transport_retries=max(0, (r["transport_attempts"] or 0)
                                                        - (r["calls"] or 0)),
                                  schema_repairs=r["schema_repairs"],
                                  economic_retries=r["economic_retries"])
               for r in per_run if (r["seed_attempts"] or 1) > 1
               or (r["transport_attempts"] or 0) > (r["calls"] or 0)
               or r["schema_repairs"] or r["economic_retries"]}
    (SUM_DIR / "retries_and_overrides.json").write_text(
        json.dumps(dict(per_cell_overrides=ovr, runs_with_retries=retries,
                        n_runs_with_retries=len(retries)), indent=1, sort_keys=True),
        encoding="utf-8")
    return dict(per_run=len(per_run), cells=len(cells), usage=usage)


# -------------------------------------------------------------------------------------------------
# validation
# -------------------------------------------------------------------------------------------------
def run_validation():
    """The independent validator, on the Azure raw file, plus the seed-freshness supplement.

    `balance9_validate.check_seed_freshness` globs `data/balance[678]*_raw.jsonl` for the prior
    corpus. In this package that glob matches nothing, so the check would pass vacuously with
    nprior = 0. The shipped `input_data/prior_seeds.json` carries the 306 distinct seeds actually
    used in Balance-6/7/8, and the overlap is recomputed here so the check has something to be about.
    """
    VAL_DIR.mkdir(parents=True, exist_ok=True)
    import balance9_validate as V                        # noqa: PLC0415
    frozen_bad, local_bad, n_frozen = hash_audit()
    out = dict(validated_file=str(RAW.relative_to(HERE)),
               validator_module_sha256=sha256_file(CODE / "balance9_validate.py"),
               code_integrity=dict(
                   n_frozen_files=n_frozen,
                   frozen_files_intact=not frozen_bad,
                   frozen_mismatches=frozen_bad,
                   runner_locally_modified=bool(local_bad),
                   local_modifications=local_bad,
                   note="frozen files are the vendored experiment code and inputs; a mismatch there "
                        "invalidates the replication. Local modifications are to this package's own "
                        "runner/transport scaffolding and are reported, not treated as failures."))
    if not RAW.exists():
        out["error"] = "no raw file to validate"
        (VAL_DIR / "validation.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
        log("  !!   no raw file to validate")
        return out

    log(f"validating {RAW.name} with the independent validator ...")
    rep, runs = V.validate_file(str(RAW))
    n_ok, n_tot = rep.summary()
    out["validator"] = dict(
        checks_passed=n_ok, checks_total=n_tot, runs_reconstructed=len(runs),
        codes_failed=rep.codes_failed(),
        failures=[dict(code=c, detail=str(d)[:400]) for c, d in rep.failures()[:200]],
        worst_error={k: v for k, v in sorted(rep.worst.items())})
    (VAL_DIR / "validator_rows.txt").write_text(
        "".join(f"{'ok  ' if ok else 'FAIL'} {c:6s} {str(d)[:300]}\n" for c, ok, d in rep.rows),
        encoding="utf-8")

    # V8d as the validator runs it here: vacuous, and reported as vacuous rather than as a pass.
    try:
        vac = V.check_seed_freshness(runs)
        out["v8d_as_run_in_this_package"] = dict(
            result=str(vac)[:300],
            vacuous=True,
            why="check_seed_freshness globs data/balance[678]*_raw.jsonl relative to the validator's "
                "own directory. This package ships no Balance-6/7/8 raw data, so the prior set is "
                "empty and the check cannot fail. It is recorded, then superseded by the supplement "
                "below.")
    except Exception as exc:                             # noqa: BLE001
        out["v8d_as_run_in_this_package"] = dict(error=f"{type(exc).__name__}: {exc}")

    prior = set(json.loads((INPUT / "prior_seeds.json").read_text(encoding="utf-8"))["prior_seeds"])
    used = set()
    with open(RAW, encoding="utf-8") as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except Exception:                            # noqa: BLE001
                continue
            if isinstance(d.get("seed"), int):
                used.add(d["seed"])
    overlap = sorted(used & prior)
    out["seed_freshness_supplement"] = dict(
        reason="the real V8d check, using the 306 prior Balance-6/7/8 seeds shipped in "
               "input_data/prior_seeds.json instead of an empty glob",
        n_prior_seeds=len(prior), prior_range=[1000, 7699],
        n_seeds_used=len(used),
        seeds_used_range=[min(used), max(used)] if used else None,
        seeds_outside_block=sorted(s for s in used if not 9000 <= s <= 9059),
        overlap_with_prior=overlap, fresh=not overlap)

    out["expected"] = dict(runs=len(POLICIES) * 4 * N_SEEDS, rounds_per_run=ROUNDS,
                           merchants=N_MERCHANTS, seed_block=[9000, 9059],
                           cells=[f"{MODEL_ALIAS}|{i}|{p}" for _, w in WAVES for i in w
                                  for p in POLICIES])
    done, tot = scan()
    out["observed"] = dict(runs=tot["footers"], rounds=tot["rounds"], calls=tot["calls"],
                           cells={"|".join(k): len(v) for k, v in sorted(done.items())})
    out["complete"] = (tot["footers"] == out["expected"]["runs"]
                       and all(len(done.get((MODEL_ALIAS, i, p), set())) == N_SEEDS
                               for _, w in WAVES for i in w for p in POLICIES))
    out["verdict"] = ("FAIL" if frozen_bad else                     # frozen code changed: not a replication
                      "PASS" if (out["complete"] and not rep.codes_failed()
                                 and out["seed_freshness_supplement"]["fresh"]) else
                      "INCOMPLETE" if not out["complete"] else "FAIL")
    (VAL_DIR / "validation.json").write_text(json.dumps(out, indent=1, default=str),
                                             encoding="utf-8")
    log(f"  ..   validator {n_ok}/{n_tot} checks, {len(runs)} runs reconstructed"
        + (f", FAILED codes {rep.codes_failed()}" if rep.codes_failed() else ""))
    log(f"  ..   runs {out['observed']['runs']}/{out['expected']['runs']}, seed freshness: "
        f"{'FRESH' if out['seed_freshness_supplement']['fresh'] else 'OVERLAP'} -> {out['verdict']}")
    return out


# -------------------------------------------------------------------------------------------------
# packaging
# -------------------------------------------------------------------------------------------------
def package_results():
    DELIV.mkdir(parents=True, exist_ok=True)
    build_summaries()
    run_validation()

    # (path, name inside the archive). Arcnames are built from the directory's ROLE rather than from
    # `relative_to(HERE)`, so the archive layout does not depend on where the package happens to sit.
    members: list[tuple[Path, str]] = []
    for d, role, pat in ((RAW_DIR, "raw", "*.jsonl"), (ATT_DIR, "attempts", "*.jsonl"),
                         (LOG_DIR, "logs", "*"), (SUM_DIR, "summaries", "*.json"),
                         (VAL_DIR, "validation", "*.json"), (MANIFESTS, "manifests", "*.json")):
        for p in sorted(Path(d).glob(pat)):
            if p.is_file():
                members.append((p, f"balance9_azure_p3_llama/{role}/{p.name}"))

    hashes = {arc: dict(sha256=sha256_file(p), bytes=p.stat().st_size) for p, arc in members}
    hp = DELIV / "SHA256SUMS.json"
    hp.write_text(json.dumps(hashes, indent=1, sort_keys=True), encoding="utf-8")
    (DELIV / "SHA256SUMS.txt").write_text(
        "".join(f"{v['sha256']}  {k}\n" for k, v in sorted(hashes.items())), encoding="utf-8")

    arc = DELIV / "balance9_azure_p3_llama_results.tar.gz"
    with tarfile.open(arc, "w:gz") as tf:
        for p, name in members + [(hp, "balance9_azure_p3_llama/SHA256SUMS.json")]:
            tf.add(p, arcname=name)

    # Extraction test: a hash of an archive nobody has opened is a hash of a hope.
    ok, n_members, why = True, 0, ""
    try:
        with tarfile.open(arc, "r:gz") as tf:
            names = set(tf.getnames())
            n_members = len(names)
            for _, name in members:
                if name not in names:
                    ok, why = False, f"missing member {name}"
                    break
                h = hashlib.sha256()
                ex = tf.extractfile(name)
                for chunk in iter(lambda: ex.read(1 << 20), b""):
                    h.update(chunk)
                if h.hexdigest() != hashes[name]["sha256"]:
                    ok, why = False, f"hash mismatch on {name}"
                    break
    except Exception as exc:                             # noqa: BLE001
        ok, why = False, f"{type(exc).__name__}: {exc}"
        log(f"  !!   archive verification raised {why}")

    man = dict(archive=arc.name, archive_bytes=arc.stat().st_size,
               archive_sha256=sha256_file(arc), members=n_members,
               extraction_verified=ok, extraction_failure=why or None,
               root_dir_in_archive="balance9_azure_p3_llama/",
               extract_with="tar -xzf balance9_azure_p3_llama_results.tar.gz",
               built=time.strftime("%Y-%m-%dT%H:%M:%S"))
    (DELIV / "ARCHIVE.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    log(f"packaged {n_members} files -> {arc.name} "
        f"({arc.stat().st_size / 1e6:.1f} MB, extraction {'VERIFIED' if ok else 'FAILED'})")
    if not ok:
        raise SystemExit("ARCHIVE VERIFICATION FAILED. Do not ship this archive.")
    return man


# -------------------------------------------------------------------------------------------------
# the run
# -------------------------------------------------------------------------------------------------
def run_waves(R, tx, cfg, seed_workers, deadline):
    done, _ = scan()
    for wno, arms in WAVES:
        # sequential-launch rule: wave n+1 does not start until wave n has all 60 seeds in every cell
        for prev, parms in WAVES:
            if prev >= wno:
                break
            st = wave_status(prev, parms, scan()[0])
            if not st["complete"]:
                short = [r["cell"] for r in st["cells"] if r["have"] < r["want"]]
                log(f"[stop] wave {wno} may not start: wave {prev} is incomplete "
                    f"({len(short)} cells short: {short}). The launch order is frozen.")
                return False
        for cell in cells_of(arms):
            alias, itf, pol = cell
            have = scan()[0].get(cell, set())
            left = [s for s in seeds() if s not in have]
            if not left:
                log(f"[wave {wno}] {alias}|{itf}|{pol}: already complete (60/60)")
                continue
            if deadline and time.time() > deadline:
                log("[budget] wall-clock budget reached; stopping cleanly between cells")
                return False
            log(f"[wave {wno}] {alias}|{itf}|{pol}: {len(left)} seeds to run")
            t0 = time.time()
            st = R.run_cell(alias, itf, pol, left, RAW, rounds=ROUNDS,
                            seed_workers=seed_workers, agent="llm", quiet=False)
            st["wall_s"] = round(time.time() - t0, 1)
            log(f"    +{st['completed']}/{len(left)} seeds  calls={st['calls']}  "
                f"reruns={st['seed_reruns']}  failed={st['failed']}  {st['wall_s']}s")
            for f in st["failures"][:3]:
                log(f"    ! seed {f['seed']}: {str(f['error'])[:200]}")
            write_progress(True, cfg.public(), dict(last_cell="|".join(cell), last_chunk=st,
                                                    transport=tx.stats()))
    return True


def main():
    ap = argparse.ArgumentParser(
        description="Azure replication of the Balance-9 P3 llama matrix (480 runs).")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--resume", action="store_true",
                   help="continue an interrupted run (skips seeds that already have a footer)")
    g.add_argument("--validate-only", action="store_true",
                   help="validate the results already on disk; makes no API call")
    g.add_argument("--package-results", action="store_true",
                   help="build summaries, validate, and write the final delivery archive")
    ap.add_argument("--seed-workers", type=int, default=int(os.environ.get("AZURE_SEED_WORKERS", "6")),
                    help="concurrent market seeds (default 6)")
    ap.add_argument("--max-seconds", type=float, default=0.0,
                    help="stop cleanly between cells after this many seconds (0 = no limit)")
    args = ap.parse_args()

    for d in (RAW_DIR, ATT_DIR, LOG_DIR, SUM_DIR, VAL_DIR, DELIV, MANIFESTS):
        d.mkdir(parents=True, exist_ok=True)

    if args.validate_only:
        log("=" * 96)
        log("validate-only: no Azure request will be made")
        log("=" * 96)
        build_summaries()
        rep = run_validation()
        log(f"verdict: {rep.get('verdict', 'NO RESULTS YET')}  "
            + json.dumps(rep.get("observed", {}), sort_keys=True))
        log(f"validation written to {(VAL_DIR / 'validation.json').relative_to(HERE)}")
        return

    if args.package_results:
        log("=" * 96)
        log("package-results: no Azure request will be made")
        log("=" * 96)
        man = package_results()
        log(json.dumps(man, indent=1))
        return

    # ---- gates, all offline -------------------------------------------------------------------
    log("=" * 96)
    log("Balance-9 P3 llama -- Azure replication")
    log("=" * 96)
    gate_vendored_integrity()
    gate_catalog_placed()
    if not all(ok for _, ok in _G):
        raise SystemExit("INTEGRITY GATE FAILED. The vendored code is not the frozen code.")

    import azure_transport as AT                         # noqa: PLC0415
    import balance9_runner as R                          # noqa: PLC0415
    import balance9_prereg_freeze as FZ                  # noqa: PLC0415

    if not gate_design(R, FZ):
        write_progress(False)
        raise SystemExit("DESIGN GATE FAILED. The runner's constants are not the document's.")

    try:
        cfg = AT.load_config()
    except AT.AzureConfigError as exc:
        # The collaborator's most likely first failure. A traceback here would be a bad first
        # experience for an error that is entirely about four environment variables.
        missing = [v for v in ("AZURE_AI_ENDPOINT", "AZURE_AI_DEPLOYMENT")
                   if not os.environ.get(v, "").strip()]
        log("")
        log("AZURE CONFIGURATION INCOMPLETE -- no request was made, nothing was written.")
        log(f"  {exc}")
        if missing:
            log(f"  not set: {', '.join(missing)}")
        log("")
        log("  export AZURE_AI_ENDPOINT=\"https://<resource>.services.ai.azure.com/models\"")
        log("  export AZURE_AI_DEPLOYMENT=\"<your Llama-3.3-70B-Instruct deployment name>\"")
        log("  export AZURE_AI_API_VERSION=\"2024-05-01-preview\"      # optional, this is the default")
        log("  export AZURE_AI_API_KEY=\"<key>\"                        # omit to use DefaultAzureCredential")
        log("  export AZURE_AI_ROUTE=\"azure-openai\"                   # only if the host is unusual")
        raise SystemExit(2) from None

    log(f"azure route      : {cfg.route}")
    log(f"azure endpoint   : {cfg.host}")
    log(f"azure deployment : {cfg.deployment}")
    log(f"azure api-version: {cfg.api_version}")
    log(f"azure auth       : {cfg.auth_mode}")

    ident_path = LOG_DIR / "model_identity.json"

    def on_identity(v):
        ident_path.write_text(json.dumps(v, indent=1, default=str), encoding="utf-8")
        log(f"model identity   : observed={v['observed']!r} accepted={v['accepted']} -- {v['reason']}")

    tx = AT.AzureTransport(cfg, AT.AttemptLog(ATTEMPTS), on_identity=on_identity)

    # the substitution -- BOTH handles, then the frozen gates, then re-assert
    R.transport_call = tx
    R.transport_call_ref[0] = tx
    R.append_jsonl = make_durable_append(R)
    install_decision_context(R, AT)

    log("-" * 96)
    R.selftest()                                          # G-R1..G-R10, no network request
    log("-" * 96)
    R.transport_call = tx                                 # selftest's finally-blocks restore from
    R.transport_call_ref[0] = tx                          # transport_call_ref; re-assert, never assume
    if not gate_transport_substitution(R, AT, tx):
        raise SystemExit("TRANSPORT GATE FAILED. The Azure transport is not installed.")

    n = sum(1 for _, ok in _G if ok)
    log(f"{n}/{len(_G)} package gates passed")

    # ---- resume policy ------------------------------------------------------------------------
    done, tot = scan()
    if tot["footers"] and not args.resume:
        raise SystemExit(
            f"{RAW.name} already contains {tot['footers']} completed runs. Pass --resume to "
            "continue (completed seeds are skipped and never duplicated). Nothing was written.")
    if args.resume:
        log(f"resuming: {tot['footers']}/{len(POLICIES) * 4 * N_SEEDS} runs already complete")

    write_progress(True, cfg.public(), dict(transport=tx.stats()))
    (MANIFESTS / "azure_environment.json").write_text(
        json.dumps(dict(azure=cfg.public(), python=sys.version.split()[0],
                        started=time.strftime("%Y-%m-%dT%H:%M:%S"),
                        seed_workers=args.seed_workers), indent=1), encoding="utf-8")

    deadline = time.time() + args.max_seconds if args.max_seconds > 0 else 0.0
    finished = False
    try:
        finished = run_waves(R, tx, cfg, args.seed_workers, deadline)
    except SystemExit:
        raise
    except BaseException as exc:                          # noqa: BLE001
        msg = AT.redact(traceback.format_exc())
        log("=" * 96)
        log(f"RUN STOPPED: {AT.redact(exc)}")
        (LOG_DIR / "fatal.log").write_text(msg, encoding="utf-8")
        write_progress(True, cfg.public(), dict(fatal=AT.redact(exc)[:500], transport=tx.stats()))
        raise SystemExit(1) from None

    art = write_progress(True, cfg.public(), dict(transport=tx.stats()))
    build_summaries()
    log("=" * 96)
    log(f"runs {art['runs_done']}/{art['runs_target']}   cells "
        f"{art['cells_complete']}/{art['cells_total']}   raw {art['raw_bytes'] / 1e6:.1f} MB")
    log(f"transport: {json.dumps(tx.stats(), sort_keys=True)}")
    if finished and art["runs_done"] == art["runs_target"]:
        log("matrix COMPLETE -- now run:  python azure_handoff/to_run.py --package-results")
    else:
        log("matrix INCOMPLETE -- rerun with --resume to continue")


if __name__ == "__main__":
    main()
