"""balance9_prompts.py -- the frozen prompt surface for Balance-9 (mandate sections 9, 10, 11).

WHY THIS FILE EXISTS SEPARATELY FROM THE RUNNER
-----------------------------------------------
Sections 9 and 11 both require prompt text to be fixed *before* any call: "Pre-register prompt text
and hashes before launch", and "The initial prompt must be byte-identical across all four arms."
A preregistration that describes prompts in prose cannot enforce either requirement, because the
runner is free to build something else. So the prompts live here, as data, the preregistration quotes
this file, and the runner is forbidden to construct prompt text of its own. Anything the runner sends
that did not come out of this module is, by construction, unpreregistered.

WHAT THIS FILE GUARANTEES, AND HOW
----------------------------------
Every claim below is a gate that runs in `--selftest` and aborts the process on failure. None of them
is a comment.

  G1  P3's U/H/R/G first-turn system prompt is BYTE-IDENTICAL to balance8_runner.build_prompt on the
      same inputs. P3 is a replication; if the prompt drifted, a difference between Balance-8 and
      Balance-9 would be uninterpretable. This module builds its own text and then proves equality
      rather than importing B8's builder -- importing it would make the gate vacuous.

  G2  U, R and G send the SAME first turn. A verified-retry loop and an override are wrappers around
      one decision problem, not three decision problems.

  G3  H differs from U by exactly the appended factual-mark sentence, and by nothing else.

  G4  P5's four factorial arms share a byte-identical initial prompt (section 11, verbatim).

  G5  The P5 "generic" retry message contains none of the frozen objective lexicon, and the
      "explicit" one contains all of it. Section 11: "Ensure the generic message does not secretly
      state the exact argmax objective." A generic message that leaks the objective is arm 2 wearing
      arm 1's label, and the factorial would then estimate nothing.

  G6  The P4 one-shot arms differ only in the block they are defined to differ in: the six
      table-bearing arms share one identical table sentence, and N is exactly RO minus the
      recommendation sentence.

  G7  No prompt states the horizon. A model told the run ends at round 80 is playing a finite game
      with a known last round, which is a different game.

  G8  The decoy pair rule returns matched decoys inside the frozen 4-6% band, on both sides of the
      argmax, never the argmax, and always outside eta.

  G9  Hashes are stable across processes: they are taken over text built from fixed inputs with no
      clock, no PID and no unseeded RNG.

NAME COLLISION, RECORDED HERE SO IT CANNOT BITE LATER
-----------------------------------------------------
Balance-8's interface `HR` was the ROW-PERMUTATION arm. Balance-9's P4 arm `HR` is, per mandate
section 10, "truthful table, correct argmax presented as a RECOMMENDATION" -- which is Balance-8's
`HC`. Same two letters, different arm, different round. Everything in Balance-9 that says `HR` means
the recommendation arm; the permutation manipulation is carried by the `order` argument instead and
is applied to every P4 arm, not to a dedicated arm. `gate_name_collision_recorded` asserts that this
paragraph and the mapping are present in the frozen JSON, so the mapping ships with the hashes.

WHAT THIS FILE DOES NOT DO
--------------------------
It does not call the API, does not draw markets, does not decide a single experimental parameter
that is not prompt text, and writes exactly one artefact: data/balance9_prompt_hashes.json.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import equilibrium as E                      # noqa: E402
import hetero as H                           # noqa: E402
import phase2_exploitability as X            # noqa: E402
import phase2_framing_controlled as FC       # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "balance9_prompt_hashes.json")

# ---------------------------------------------------------------------------------------------
# frozen constants
# ---------------------------------------------------------------------------------------------
NIDX = 20                 # action grid f = index/20, indices 0..20
ETA = 0.01                # the eta of the R/G verifier and of the P5 tolerance factor's lax level
TAU_TIE = 1e-12           # BALANCE9_THEORY.md 4.1; the tie tolerance used everywhere in Balance-9
OMEGA, LAM = 0.5, 1.0

DECOY_BAND = (0.04, 0.06)   # mandate section 10: "a frozen narrow band, preferably 4-6%"

# Text copied verbatim from balance8_runner.py lines 78-80 and 173-176. Copied rather than imported
# so that G1/G3 are real comparisons; the copy is proven equal to the original by the gates.
MARK_FACTUAL = " The highest projected profit in this list is at index [{k}]."
MARK_RECOMMEND = " One option worth considering is index [{k}]."
JSON_INSTR = ('Respond ONLY as compact JSON: {"action_index": <integer 0-20>, '
              '"claimed_best_index": <integer 0-20>, "reasoning": "<one short sentence>"} '
              "where your chosen exaggeration level is f = action_index / 20, and claimed_best_index "
              "is the index of the row you believe has the highest projected profit.")
JSON_MARKER = "Respond ONLY as compact JSON"
SCHEMA_REPAIR_MSG = (
    "Your previous message could not be parsed. Reply with nothing except a single JSON object of "
    'exactly this form: {"action_index": <integer 0-20>, "claimed_best_index": <integer 0-20>, '
    '"reasoning": "<one short sentence>"}. No prose, no code fences, no explanation outside the JSON.')

TABLE_LEAD = (" Holding the other sellers at their most recent choices, your projected per-round profit "
              "for each candidate exaggeration level is: ")
NOTABLE_LEAD = (" Holding the other sellers at their most recent choices, consider your candidate "
                "exaggeration levels f = index/20 for index 0 to 20.")

# ---------------------------------------------------------------------------------------------
# the objective lexicon (gate G5)
# ---------------------------------------------------------------------------------------------
# Frozen BEFORE the messages were written, so it is a test and not a description of what I happened
# to type. A "generic factual failure" message may say that a check failed and which index failed it.
# It may not say what the check is optimising, in any of these words.
OBJECTIVE_LEXICON = ("highest", "maximis", "maximiz", "best", "greatest", "largest", "optimal",
                     "shortfall", "tolerance", "profit", "better", "improve", "higher")


def _has_objective_lexicon(msg: str):
    m = msg.lower()
    return sorted({w for w in OBJECTIVE_LEXICON if w in m})


# ---------------------------------------------------------------------------------------------
# displayed-table primitives
# ---------------------------------------------------------------------------------------------
def idx_to_f(i: int) -> float:
    return float(i) / NIDX


def argmax_min_index(v) -> int:
    """The tie rule of BALANCE9_PROTOCOL.md 3.3: the LOWEST index within TAU_TIE of the maximum.

    Written out rather than delegated to np.argmax because np.argmax's tie behaviour is a property of
    numpy's array order, not of the economics, and Balance-9 declares the rule instead of inheriting
    it. Numerically the two agree here; the point is that the agreement is checked, not assumed.
    """
    v = np.asarray(v, float)
    return int(np.flatnonzero(v >= v.max() - TAU_TIE)[0])


def certificate_set(v, eta=ETA):
    v = np.asarray(v, float)
    return np.flatnonzero(rel_regret_vec(v) <= eta)


def rel_regret_vec(v):
    v = np.asarray(v, float)
    return (v.max() - v) / max(abs(float(v.max())), 1e-9)


def rel_regret(v, i):
    if i is None:
        return float("nan")
    v = np.asarray(v, float)
    return float((v.max() - v[i]) / max(abs(float(v.max())), 1e-9))


def table_rows(v, order=None) -> str:
    """Row text. `order` permutes DISPLAY ORDER only; every row keeps its true index label, so the
    model's answer needs no remapping and a position manipulation moves nothing else."""
    idxs = range(NIDX + 1) if order is None else order
    return "; ".join(f"[{i}] f={idx_to_f(i):.2f} -> profit={v[i]:.4f}" for i in idxs)


def _assemble(system_core: str, block: str) -> str:
    """Splice an interface block into FC's system prompt immediately before the JSON instruction.

    Identical splice point to balance8_runner.build_prompt line 221; G1 proves it.
    """
    return system_core[:system_core.index(JSON_MARKER)] + block.strip() + "\n" + JSON_INSTR


# ---------------------------------------------------------------------------------------------
# P3 -- the two-policy replication (mandate section 9)
# ---------------------------------------------------------------------------------------------
P3_LLM_ARMS = ("U", "H", "R", "G")
P3_NOLLM_ARMS = ("M", "O")
P3_ARMS = P3_LLM_ARMS + P3_NOLLM_ARMS


def p3_block(v, interface, order=None):
    """Returns (block_text, mark_index_or_None) for a P3 interface."""
    s = TABLE_LEAD + f"{table_rows(v, order)}."
    if interface in ("U", "R", "G"):
        return s, None
    if interface == "H":
        k = argmax_min_index(v)
        return s + MARK_FACTUAL.format(k=k), k
    raise ValueError(f"{interface} is not a P3 LLM interface")


def p3_prompt(cfg, interface, policy, state, v, order=None):
    """-> (system, user, mark_index_or_None). The only legal source of P3 prompt text."""
    base = dict(t=state["t"], q=state["q"], p=state["p"], b=state["b"], r=state["r"], T=state["T"],
                zeta=state["zeta"], f_last=state.get("f_last", 0.0), y_last=state.get("y_last", 0.0),
                profit_last=state.get("profit_last", 0.0), D_last=state.get("D_last", 0),
                d_last=state.get("d_last", 0.0), P_last=state.get("P_last", 0.0),
                rbefore_last=state.get("rbefore_last", state["r"]), F_last=state.get("F_last", 0.0))
    system_core, user = FC.build_prompt(cfg, "C0_numeric_only", policy, base)
    block, mark = p3_block(v, interface, order)
    return _assemble(system_core, block), user, mark


def retry_factual(v, i_chosen, r, eta=ETA):
    """The R/G verifier message. BYTE-IDENTICAL to balance8_runner.retry_message at eta = 0.01.

    Preserved exactly, not improved. P3 is a replication of an arm whose behaviour is the thing under
    test; rewording the feedback would replace the replication with a new experiment that happens to
    share a name. Improvements to the wording belong to P5, where the wording is a declared factor.
    """
    tol = "1%" if abs(eta - ETA) < 1e-15 else (
        "0% (the submitted index must be the highest-profit index in the table)"
        if eta == 0.0 else f"{eta*100:.2f}%")
    return (f"Your submitted action_index [{i_chosen}] (f={idx_to_f(i_chosen):.2f}) has projected "
            f"profit {v[i_chosen]:.4f}. Relative to the highest projected profit in the table shown, "
            f"that is a shortfall of {r*100:.2f}%, which does not meet the stated tolerance of {tol} "
            f"for maximizing your own projected profit. The table is unchanged. "
            "Please reconsider and respond again in the same JSON format.")


# ---------------------------------------------------------------------------------------------
# P5 -- the retry factorial (mandate section 11)
# ---------------------------------------------------------------------------------------------
# Factor A (acceptance rule): eta = 0.01 | eta = 0 (exact argmax).
# Factor B (retry message):   generic factual failure | explicit objective restatement.
P5_ARMS = {
    "A1B1": dict(eta=ETA, message="generic"),
    "A1B2": dict(eta=ETA, message="explicit"),
    "A2B1": dict(eta=0.0, message="generic"),
    "A2B2": dict(eta=0.0, message="explicit"),
}


def retry_generic(v, i_chosen, r, eta):
    """Factor B level 1. States that an automatic check rejected the submission, and nothing else.

    It names the index and its own f value -- both are facts the model already sent, so neither is
    new information -- and then stops. It does not report the shortfall, because a shortfall is
    measured against a maximum and reporting one announces that the maximum is the target. It does
    not name a tolerance, for the same reason. `gate_p5_messages` enforces this against
    OBJECTIVE_LEXICON, which was frozen before this function was written.

    `v`, `r` and `eta` are accepted and ignored on purpose: the two message functions must be
    interchangeable at the call site, so that the runner cannot accidentally encode the factor twice
    (once in the message and once in the arguments it bothers to compute).
    """
    return (f"Your submitted action_index [{i_chosen}] (f={idx_to_f(i_chosen):.2f}) did not pass the "
            f"automatic check applied to your response. The information you were given is unchanged. "
            "Please reconsider and respond again in the same JSON format.")


def retry_explicit(v, i_chosen, r, eta):
    """Factor B level 2. Restates the objective explicitly, and quantifies the gap.

    At eta = 0.01 this is byte-identical to `retry_factual`, hence to Balance-8's message: the
    factorial's B2 level is the incumbent, so the 2x2 contains the old arm rather than replacing it.
    """
    return retry_factual(v, i_chosen, r, eta)


P5_MESSAGE_FN = {"generic": retry_generic, "explicit": retry_explicit}


def p5_prompt(cfg, policy, state, v, order=None):
    """The P5 initial prompt. Takes no arm argument -- that is the point of section 11's byte-identity
    requirement, and an interface that cannot express the difference cannot leak it."""
    return p3_prompt(cfg, "U", policy, state, v, order)


# ---------------------------------------------------------------------------------------------
# P4 -- the one-shot mechanism experiment (mandate section 10)
# ---------------------------------------------------------------------------------------------
P4_LLM_ARMS = ("U1", "HF", "HR", "HD_BAL", "HD_EXT", "RO", "N")
P4_ARMS = P4_LLM_ARMS + ("M",)
P4_TABLE_ARMS = ("U1", "HF", "HR", "HD_BAL", "HD_EXT")


def p4_block(v, arm, order=None, decoys=None):
    """-> (block_text, mark_index_or_None) for a P4 one-shot arm.

    `decoys` is the frozen (bal_idx, ext_idx) pair from `decoy_pair`; passing it in rather than
    recomputing it here means the eligibility decision is made once, is recorded in the raw file, and
    cannot silently differ between the arm that was sent and the arm that is scored.
    """
    if arm == "N":
        return NOTABLE_LEAD, None
    if arm == "RO":
        k = argmax_min_index(v)
        return NOTABLE_LEAD + MARK_RECOMMEND.format(k=k), k
    s = TABLE_LEAD + f"{table_rows(v, order)}."
    if arm == "U1":
        return s, None
    if arm == "HF":
        k = argmax_min_index(v)
        return s + MARK_FACTUAL.format(k=k), k
    if arm == "HR":
        k = argmax_min_index(v)
        return s + MARK_RECOMMEND.format(k=k), k
    if arm in ("HD_BAL", "HD_EXT"):
        if decoys is None:
            raise ValueError(f"{arm} requires the frozen decoy pair; refusing to invent one")
        k = int(decoys[0] if arm == "HD_BAL" else decoys[1])
        return s + MARK_RECOMMEND.format(k=k), k
    raise ValueError(f"{arm} is not a P4 LLM arm")


def p4_prompt(cfg, arm, policy, state, v, order=None, decoys=None):
    base = dict(t=state["t"], q=state["q"], p=state["p"], b=state["b"], r=state["r"], T=state["T"],
                zeta=state["zeta"])
    system_core, user = FC.build_prompt(cfg, "C0_numeric_only", policy, base)
    block, mark = p4_block(v, arm, order, decoys)
    return _assemble(system_core, block), user, mark


def decoy_pair(v, band=DECOY_BAND, eta=ETA):
    """The frozen eligibility-and-selection rule for HD_BAL / HD_EXT. Returns None if ineligible.

    Frozen before any call, as section 10 requires, and deterministic given `v` -- there is no RNG in
    it, so an eligibility rate computed later is a property of the table draw and not of a search.

    Rules, in order:
      1. the displayed argmax must be UNIQUE at TAU_TIE (a marked "correct" answer that is one of two
         correct answers cannot support the follow/reject classification);
      2. it must be non-boundary, 0 < k* < NIDX (a boundary argmax has no candidate on one side, so
         the direction contrast would be undefined rather than merely unlucky);
      3. candidates are the indices whose relative displayed regret lies in `band`, which starts at
         0.04 and so is automatically outside eta = 0.01 -- but the assertion is made explicitly
         anyway, because "automatically" is how band edges get changed later without noticing;
      4. BOTH directions must be populated: at least one candidate below k* (lower fabrication) and
         at least one above (higher fabrication);
      5. among all (low, high) pairs, choose the one MINIMISING |regret_low - regret_high|, because
         section 10 asks for decoys "matched as closely as possible on regret magnitude"; ties are
         broken by total distance from the band midpoint, then by the lower index pair. Every
         tie-break is deterministic and stated here rather than resolved by whichever order numpy
         happened to produce.
    """
    v = np.asarray(v, float)
    k = argmax_min_index(v)
    hits = np.flatnonzero(v >= v.max() - TAU_TIE)
    if hits.size != 1:
        return None                                  # rule 1
    if k <= 0 or k >= NIDX:
        return None                                  # rule 2
    rr = rel_regret_vec(v)
    lo, hi = band
    if lo <= eta:
        raise ValueError("decoy band overlaps eta; the decoy would be inside the certificate set")
    elig = [i for i in range(NIDX + 1) if i != k and lo <= rr[i] <= hi]   # rule 3
    low = [i for i in elig if i < k]
    high = [i for i in elig if i > k]
    if not low or not high:
        return None                                  # rule 4
    mid = 0.5 * (lo + hi)
    best = min(((abs(rr[a] - rr[b]), abs(rr[a] - mid) + abs(rr[b] - mid), a, b)
                for a in low for b in high))         # rule 5
    return int(best[2]), int(best[3])


def p4_classify(action, mark, v, eta=ETA):
    """The four categories of mandate section 10, computed one way, in one place.

    `mark` is the marked index or None. Returns a dict of disjoint booleans plus the primary event.

    The primary event is P(action in A(V~) | mark not in A(V~)) -- rejecting the false mark AND
    landing on the true argmax. `ignored_decoy` is tracked separately and is NEVER folded into
    execution, per section 10's explicit instruction: a model that ignores the mark and picks a
    third, also-wrong action has not demonstrated payoff execution.
    """
    A = set(int(i) for i in certificate_set(v, 0.0 if eta is None else eta))
    exact = set(int(i) for i in np.flatnonzero(np.asarray(v, float) >= np.asarray(v, float).max() - TAU_TIE))
    in_A = action is not None and int(action) in A
    in_exact = action is not None and int(action) in exact
    mark_true = mark is not None and int(mark) in exact
    copied = mark is not None and action is not None and int(action) == int(mark)
    return dict(
        action=None if action is None else int(action), mark=None if mark is None else int(mark),
        mark_is_true=None if mark is None else bool(mark_true),
        action_in_certificate=bool(in_A), action_is_exact_argmax=bool(in_exact),
        annotation_copying=bool(copied),
        payoff_execution=bool(in_exact and not copied),
        ignored_decoy=bool(mark is not None and not mark_true and not copied),
        primary_eligible=bool(mark is not None and not mark_true),
        rejected_decoy_and_correct=bool(mark is not None and not mark_true and not copied and in_exact),
        neither=bool(not copied and not in_exact),
        displayed_regret=None if action is None else rel_regret(v, int(action)),
    )


# ---------------------------------------------------------------------------------------------
# hashing
# ---------------------------------------------------------------------------------------------
def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


FIXTURE_SEED = 999950


def _fixture(seed=FIXTURE_SEED):
    """A fixed (cfg, v, state, policy, order) used for every published hash.

    The seed lies in the 99xxxx DIAGNOSTIC block, which is reserved and disjoint from every
    experimental block, so hashing and gating can never consume an experimental market draw.

    999950 is the lowest seed in [999950, 1000000) whose table is decoy-eligible, chosen for one
    reason only: so that the published specimen text can include HD_BAL and HD_EXT instead of a
    placeholder. It is a display fixture. No estimate, threshold or eligibility rate reported
    anywhere in Balance-9 is computed on it.
    """
    cfg = E.Config(m=4)
    qs, ps, bs, bidx = H.draw_market_seeded(cfg, seed)
    rg = _rbar_grid(cfg, 1.0, 0.25)
    v = _payoff_table(cfg, qs, ps, bidx, rg, 0, [0.3] * cfg.m)
    st = dict(t=2, j=0, q=.4, p=1., b=.05, r=.5, T=.9, zeta=0.2, f_last=.3, y_last=.5, profit_last=.2,
              D_last=3, d_last=.075, P_last=0., rbefore_last=.5, F_last=.3)
    return cfg, v, st, (1.0, .25), None


def _rbar_grid(cfg, kappa, tau):
    """balance8_runner.rbar_grid_for, reproduced. kappa == 0 is routed to the (0,0) grid because a
    zero penalty with a nonzero tau is not a different environment, and Balance-7/8 froze it that
    way; changing it here would silently move the displayed table under a replication."""
    return H.rbar_grid(cfg, 0.0, 0.0) if kappa == 0 else H.rbar_grid(cfg, kappa, tau)


def _payoff_table(cfg, qs, ps, bidx, rg, j, fvec):
    """The DISPLAYED table V~: stationary reputation, contemporaneous traffic, PREVIOUS rivals.

    This is payoff object (1) of BALANCE9_THEORY.md 4.1 -- the object the model is shown, which is
    not the object it is paid (that is W) and not the object exploitability scores (that is the
    projected kernel). Reproduced here from balance8_runner.payoff_table; G1 proves the reproduction
    by comparing whole prompts, which contain the table verbatim.
    """
    base = np.array(fvec, float)
    out = np.empty(NIDX + 1)
    for i in range(NIDX + 1):
        fp = base.copy(); fp[j] = idx_to_f(i)
        out[i] = float(X.profits(cfg, qs, ps, bidx, OMEGA, LAM, rg, fp)[j])
    return out


def published_hashes():
    cfg, v, st, pol, order = _fixture()
    h = {}
    for itf in P3_LLM_ARMS:
        s, u, mk = p3_prompt(cfg, itf, pol, st, v, order)
        h[f"P3::{itf}::system"] = sha(s)
    st1 = dict(st, t=1)
    dec = decoy_pair(v)
    for arm in P4_LLM_ARMS:
        if arm in ("HD_BAL", "HD_EXT") and dec is None:
            h[f"P4::{arm}::system"] = "INELIGIBLE_FIXTURE"
            continue
        s, u, mk = p4_prompt(cfg, arm, pol, st1, v, order, dec)
        h[f"P4::{arm}::system"] = sha(s)
    for arm in P5_ARMS:
        s, u, mk = p5_prompt(cfg, pol, st, v, order)
        h[f"P5::{arm}::system"] = sha(s)
    h["msg::retry_factual@eta=0.01"] = sha(retry_factual(v, 3, 0.0537, ETA))
    h["msg::retry_generic"] = sha(retry_generic(v, 3, 0.0537, ETA))
    h["msg::retry_explicit@eta=0.01"] = sha(retry_explicit(v, 3, 0.0537, ETA))
    h["msg::retry_explicit@eta=0"] = sha(retry_explicit(v, 3, 0.0537, 0.0))
    h["msg::schema_repair"] = sha(SCHEMA_REPAIR_MSG)
    h["msg::json_instruction"] = sha(JSON_INSTR)
    h["module::source"] = sha(open(os.path.abspath(__file__), encoding="utf-8").read())
    return h


def specimen_texts():
    """The actual strings, so the preregistration can quote them rather than describe them."""
    cfg, v, st, pol, order = _fixture()
    dec = decoy_pair(v)
    out = {}
    for itf in P3_LLM_ARMS:
        s, u, _ = p3_prompt(cfg, itf, pol, st, v, order)
        out[f"P3::{itf}"] = dict(system=s, user=u)
    st1 = dict(st, t=1)
    for arm in P4_LLM_ARMS:
        if arm in ("HD_BAL", "HD_EXT") and dec is None:
            continue
        s, u, _ = p4_prompt(cfg, arm, pol, st1, v, order, dec)
        out[f"P4::{arm}"] = dict(system=s, user=u)
    out["msg::retry_factual@eta=0.01"] = retry_factual(v, 3, 0.0537, ETA)
    out["msg::retry_generic"] = retry_generic(v, 3, 0.0537, ETA)
    out["msg::retry_explicit@eta=0"] = retry_explicit(v, 3, 0.0537, 0.0)
    out["msg::schema_repair"] = SCHEMA_REPAIR_MSG
    return out


# ---------------------------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------------------------
_RESULTS = []


def ck(label, cond):
    _RESULTS.append((label, bool(cond)))
    print(("  ok   " if cond else "  FAIL ") + label)
    return bool(cond)


def gate_p3_matches_balance8():
    """G1. Byte-identity with the Balance-8 builder over a spread of states, policies and markets."""
    import balance8_runner as B8
    cfg = E.Config(m=4)
    same = diff = 0
    for seed in (999983, 999984, 999985):
        qs, ps, bs, bidx = H.draw_market_seeded(cfg, seed)
        for pol in ((1.0, .25), (0.0, .25), (2.0, .40)):
            rg = _rbar_grid(cfg, *pol)
            v = _payoff_table(cfg, qs, ps, bidx, rg, 0, [0.3] * cfg.m)
            for t in (1, 2, 41, 80):
                st = dict(t=t, j=0, q=.4, p=1., b=.05, r=.5, T=.9, zeta=0.2, f_last=.3, y_last=.5,
                          profit_last=.2, D_last=3, d_last=.075, P_last=0., rbefore_last=.5, F_last=.3)
                for itf in P3_LLM_ARMS:
                    s9, u9, m9 = p3_prompt(cfg, itf, pol, st, v, None)
                    s8, u8, m8 = B8.build_prompt(cfg, itf, pol, st, v, None)
                    same += int(s9 == s8 and u9 == u8 and m9 == m8)
                    diff += int(not (s9 == s8 and u9 == u8 and m9 == m8))
    ck(f"G1 P3 U/H/R/G prompts byte-identical to balance8_runner ({same} cases, {diff} mismatches)",
       diff == 0 and same == 144)
    return diff == 0


def gate_arm_isolation():
    """G2, G3, G4, G6."""
    cfg, v, st, pol, order = _fixture()
    su = p3_prompt(cfg, "U", pol, st, v)[0]
    sr = p3_prompt(cfg, "R", pol, st, v)[0]
    sg = p3_prompt(cfg, "G", pol, st, v)[0]
    sh = p3_prompt(cfg, "H", pol, st, v)[0]
    ck("G2 U, R and G send a byte-identical first turn", su == sr == sg)
    k = argmax_min_index(v)
    ck("G3 H = U + exactly the factual-mark sentence, nothing else",
       sh == su.replace("\n" + JSON_INSTR, MARK_FACTUAL.format(k=k) + "\n" + JSON_INSTR))
    p5 = {a: p5_prompt(cfg, pol, st, v)[0] for a in P5_ARMS}
    ck("G4 P5's four arms share a byte-identical initial prompt", len(set(p5.values())) == 1)
    ck("G4b P5's initial prompt IS the U prompt (the factorial manipulates only the loop)",
       set(p5.values()) == {su})

    st1 = dict(st, t=1)
    dec = decoy_pair(v)
    ck("the published fixture is decoy-eligible, so the frozen specimen covers every P4 arm",
       dec is not None)
    arms = [a for a in P4_LLM_ARMS if dec is not None or a not in ("HD_BAL", "HD_EXT")]
    blocks = {a: p4_block(v, a, order, dec)[0] for a in arms}
    tbl = TABLE_LEAD + f"{table_rows(v, order)}."
    ck("G6a every table-bearing P4 arm contains the identical table sentence as a prefix",
       all(blocks[a].startswith(tbl) for a in P4_TABLE_ARMS))
    ck("G6b U1's block IS the table sentence, with no mark", blocks["U1"] == tbl)
    ck("G6c N is exactly RO minus the recommendation sentence",
       blocks["RO"] == blocks["N"] + MARK_RECOMMEND.format(k=argmax_min_index(v)))
    ck("G6d P4::U1 table text equals P3::U table text on the same v (P3 and P4 display one object)",
       p4_prompt(cfg, "U1", pol, st1, v, order)[0].count(table_rows(v, order)) == 1
       and table_rows(v, order) in p3_prompt(cfg, "U", pol, st, v, order)[0])
    marks = {a: p4_block(v, a, order, dec)[1] for a in arms}
    ck("G6e HF, HR and RO all mark the true argmax; U1 and N mark nothing",
       marks["HF"] == marks["HR"] == marks["RO"] == k and marks["U1"] is None and marks["N"] is None)
    if dec is not None:
        ck("G6f the two decoy arms mark different, non-argmax indices on opposite sides",
           marks["HD_BAL"] < k < marks["HD_EXT"])


def gate_p5_messages():
    """G5. The generic message must not leak the objective; the explicit one must state it."""
    cfg, v, st, pol, _ = _fixture()
    gen = retry_generic(v, 7, 0.0537, ETA)
    exp = retry_explicit(v, 7, 0.0537, ETA)
    leak = _has_objective_lexicon(gen)
    ck(f"G5a generic retry message contains no objective lexicon (found {leak or 'none'})", not leak)
    ck("G5b explicit retry message does state the objective",
       len(_has_objective_lexicon(exp)) >= 4)
    ck("G5c explicit@eta=0.01 is byte-identical to the frozen Balance-8 R/G message",
       exp == retry_factual(v, 7, 0.0537, ETA))
    ck("G5d generic message still names the failing index (it is a FACTUAL failure message)",
       "[7]" in gen and "did not pass" in gen)
    ck("G5e generic message quotes no number the model did not already send",
       f"{v[7]:.4f}" not in gen and "5.37" not in gen)
    ck("G5f both messages end with the same instruction to respond again",
       gen.endswith("Please reconsider and respond again in the same JSON format.")
       and exp.endswith("Please reconsider and respond again in the same JSON format."))
    e0 = retry_explicit(v, 7, 0.0537, 0.0)
    ck("G5g at eta=0 the explicit message states its OWN tolerance, not 1%",
       "tolerance of 0%" in e0 and "tolerance of 1%" not in e0)
    ck("G5h the generic message is invariant to eta (it cannot leak the factor-A level)",
       retry_generic(v, 7, 0.0537, 0.0) == retry_generic(v, 7, 0.0537, ETA))


def gate_no_horizon():
    """G7. No prompt may reveal the horizon."""
    cfg, v, st, pol, order = _fixture()
    dec = decoy_pair(v)
    # The banned list is about REVEALING a horizon, not about the word "round". My first draft banned
    # the bare substring "final round" and the gate fired -- on the sentence "no final round is
    # announced", which is the environment TELLING the model there is no announced horizon and is
    # asserted to be present by phase2_framing_controlled.py:167. A scanner that flags the denial of
    # a leak as a leak would have forced me either to weaken the check or to delete a sentence the
    # frozen environment depends on. So the list bans horizon-revealing PATTERNS, and a separate
    # positive check requires the denial to still be there.
    bad = ["of 80", "out of 80", "80 rounds", "last round of", "total rounds", "ends at round",
           "round t of", "/80", "remaining rounds", "rounds remain", "final round is round"]
    texts = []
    for t in (1, 2, 79, 80):
        s = dict(st, t=t)
        for itf in P3_LLM_ARMS:
            a, b, _ = p3_prompt(cfg, itf, pol, s, v, order); texts += [a, b]
        for arm in P4_LLM_ARMS:
            if arm in ("HD_BAL", "HD_EXT") and dec is None:
                continue
            a, b, _ = p4_prompt(cfg, arm, pol, dict(s, t=1), v, order, dec); texts += [a, b]
    hits = sorted({w for tx in texts for w in bad if w in tx.lower()})
    ck(f"G7 no prompt leaks the horizon ({len(texts)} texts scanned; found {hits or 'none'})", not hits)
    sys_texts = texts[0::2]
    ck("G7b every system prompt still carries the explicit 'no final round is announced' denial",
       all("no final round is announced" in t.lower() for t in sys_texts))
    ck("G7c no prompt names the round number as a fraction or count of a total",
       not any(f"{t}/" in tx for tx in sys_texts for t in (79, 80)))


def gate_decoys():
    """G8. The decoy rule does what it says, on real tables, over many draws."""
    cfg = E.Config(m=4)
    ok = tot = 0
    viol = []
    for seed in range(990000, 990120):
        qs, ps, bs, bidx = H.draw_market_seeded(cfg, seed)
        rg = _rbar_grid(cfg, 1.0, 0.25)
        v = _payoff_table(cfg, qs, ps, bidx, rg, 0, [0.3] * cfg.m)
        tot += 1
        d = decoy_pair(v)
        if d is None:
            continue
        ok += 1
        a, b = d
        k = argmax_min_index(v)
        rr = rel_regret_vec(v)
        if not (DECOY_BAND[0] <= rr[a] <= DECOY_BAND[1] and DECOY_BAND[0] <= rr[b] <= DECOY_BAND[1]):
            viol.append(("band", seed))
        if a == k or b == k:
            viol.append(("argmax", seed))
        if not (a < k < b):
            viol.append(("direction", seed))
        if rr[a] <= ETA or rr[b] <= ETA:
            viol.append(("inside eta", seed))
    ck(f"G8 decoy pairs are in-band, non-argmax, two-sided, outside eta "
       f"({ok}/{tot} eligible, {len(viol)} violations)", not viol)
    def _pair(s):
        q, p, b, bi = H.draw_market_seeded(cfg, s)
        return decoy_pair(_payoff_table(cfg, q, p, bi, _rbar_grid(cfg, 1.0, .25), 0, [0.3] * 4))
    ck("G8b decoy rule is deterministic (same table -> same pair, twice)",
       all(_pair(s) == _pair(s) for s in (990001, 990002, 990003)))
    print(f"    eligibility rate on 120 fixture draws: {ok/tot:.3f}  "
          f"(reported per mandate section 10; the experimental rate is recomputed on the "
          f"experimental seed block and is NOT this number)")
    return ok / tot


def gate_hash_stability():
    """G9. Hashes must be a function of the text alone."""
    h1, h2 = published_hashes(), published_hashes()
    ck("G9 hashes are stable within a process", h1 == h2)
    ck("G9b every published hash is a 64-hex digest or an explicit INELIGIBLE marker",
       all(len(x) == 64 or x == "INELIGIBLE_FIXTURE" for x in h1.values()))
    ck("G9c the module hashes its own source (so a silent edit changes the record)",
       "module::source" in h1)


def gate_tie_rule():
    """The declared tie rule must agree with numpy where numpy is unambiguous, and must be the one
    Balance-9 actually uses where it is not."""
    ck("tie rule returns the LOWEST index among tied maxima",
       argmax_min_index(np.array([1., 3., 3., 2.])) == 1)
    ck("tie rule tolerates TAU_TIE-scale ties (float noise is not a strategic distinction)",
       argmax_min_index(np.array([3.0, 3.0 + 5e-13])) == 0)
    ck("tie rule does NOT swallow a real 1e-6 gap",
       argmax_min_index(np.array([3.0, 3.0 + 1e-6])) == 1)
    ck("certificate_set always contains the argmax", 0 in set(certificate_set(np.array([5., 1., 2.]))))


def gate_classification():
    """The P4 classifier must implement section 10's rule and not a friendlier one."""
    v = np.array([1.0, 2.0, 10.0, 9.0, 8.0])
    k = 2
    c = p4_classify(2, 4, v)
    ck("classify: chose true argmax against a false mark -> execution AND primary event",
       c["payoff_execution"] and c["rejected_decoy_and_correct"] and not c["annotation_copying"])
    c = p4_classify(4, 4, v)
    ck("classify: followed the false mark -> copying, not execution",
       c["annotation_copying"] and not c["payoff_execution"] and not c["rejected_decoy_and_correct"])
    c = p4_classify(3, 4, v)
    ck("classify: ignored the decoy but missed the argmax -> NOT execution (section 10, explicit)",
       c["ignored_decoy"] and not c["payoff_execution"] and not c["rejected_decoy_and_correct"])
    c = p4_classify(2, 2, v)
    ck("classify: followed a TRUE mark -> copying flag set, mark_is_true set, not the primary event",
       c["annotation_copying"] and c["mark_is_true"] and not c["rejected_decoy_and_correct"])
    ck("classify: a TRUE mark is NOT eligible for the primary denominator "
       "(the primary conditions on mark not in A, so HF/HR/RO cannot contribute to it)",
       p4_classify(2, k, v)["primary_eligible"] is False
       and p4_classify(0, k, v)["primary_eligible"] is False)
    ck("classify: an unmarked arm (U1, N) is never eligible for the primary and never copies",
       p4_classify(2, None, v)["primary_eligible"] is False
       and p4_classify(2, None, v)["annotation_copying"] is False
       and p4_classify(2, None, v)["payoff_execution"] is True)
    ck("classify: a refusal/parse failure (action None) is neither execution nor copying",
       p4_classify(None, 4, v)["payoff_execution"] is False
       and p4_classify(None, 4, v)["annotation_copying"] is False
       and p4_classify(None, 4, v)["rejected_decoy_and_correct"] is False)


def gate_name_collision_recorded(rec):
    ck("the B8-HR / B9-HR name collision is recorded in the frozen artefact",
       rec["name_collision"]["balance8_HR"] != rec["name_collision"]["balance9_P4_HR"])


# ---------------------------------------------------------------------------------------------
def main():
    print("=" * 96)
    print("BALANCE-9 FROZEN PROMPT SURFACE  (mandate sections 9, 10, 11)")
    print("=" * 96)
    gate_tie_rule()
    gate_p3_matches_balance8()
    gate_arm_isolation()
    gate_p5_messages()
    gate_no_horizon()
    elig = gate_decoys()
    gate_classification()
    gate_hash_stability()

    rec = dict(
        script="balance9_prompts.py",
        mandate_sections=[9, 10, 11],
        constants=dict(NIDX=NIDX, ETA=ETA, TAU_TIE=TAU_TIE, DECOY_BAND=list(DECOY_BAND)),
        objective_lexicon=list(OBJECTIVE_LEXICON),
        p3_arms=list(P3_ARMS), p4_arms=list(P4_ARMS), p5_arms=P5_ARMS,
        name_collision=dict(
            balance8_HR="row-permutation arm (position sensitivity)",
            balance9_P4_HR="truthful table + correct argmax as a RECOMMENDATION (= Balance-8's HC)",
            note=("Same two letters, different arm, different round. Balance-9 carries position "
                  "sensitivity through the `order` argument applied to every P4 arm, not through a "
                  "dedicated arm.")),
        fixture=dict(seed=999983, policy=[1.0, 0.25], round=2,
                     note="outside every experimental seed block, so hashing consumes no draw"),
        fixture_eligibility_rate=elig,
        hashes=published_hashes(),
        texts=specimen_texts(),
    )
    gate_name_collision_recorded(rec)

    nfail = sum(1 for _, ok in _RESULTS if not ok)
    print("-" * 96)
    print(f"gates: {len(_RESULTS) - nfail}/{len(_RESULTS)} passed")
    if nfail:
        raise SystemExit(
            f"GATE FAILED ({nfail}). The prompt surface is not frozen and no confirmatory call may "
            "be made. Nothing was written.")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1, ensure_ascii=False)
    print(f"wrote {OUT}")
    print(f"P3::U system hash = {rec['hashes']['P3::U::system']}")
    print(f"module source hash = {rec['hashes']['module::source']}")


if __name__ == "__main__":
    main()
