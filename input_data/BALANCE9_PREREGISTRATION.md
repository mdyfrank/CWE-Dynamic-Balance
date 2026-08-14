# BALANCE-9 PREREGISTRATION

**Status:** frozen. Written before the first confirmatory API call of Balance-9.
**Governs:** P3 (mandate §9), P4 (§10), P5 (§11), and — if reached — P6 (§12).
**Companion documents:** `BALANCE9_PLAN.md` (what gets done and in what order),
`BALANCE9_PROTOCOL.md` (how it gets done), `BALANCE9_THEORY.md` (what the objects are),
`BALANCE9_ERRATA.md` (what was wrong before), `data/balance9_metric_spec.json` (what the numbers mean),
`data/balance9_prompt_hashes.json` (what the models will actually be sent).

---

## 0. What this document is for, and what would falsify it

A preregistration earns its keep only if it can be violated. Every section below is written so that a
reader holding this file and the raw data can check, without my help, whether the analysis I report
is the analysis I promised. Concretely:

* every metric is named by its **ID** in `data/balance9_metric_spec.json`, never by an English
  phrase. In this corpus the phrase "relative exploitability" denotes two different numbers whose
  means differ by 1.86×, so an English name is not an identifier;
* every hypothesis has a direction, a test, a unit of inference, and a family;
* every threshold that decides an outcome is quoted from the mandate, or, where the mandate uses a
  word like "often" or "most", is given a number **here**, before the data exist;
* the prompts are not described, they are hashed, and the hashes are already on disk;
* the seed blocks are fixed and verified disjoint from every prior round;
* the analyses I am **not** allowed to run are listed in §12.

If the eventual `BALANCE9_RESULTS.md` reports a primary test that is not in §5–§8, or reports a
primary test from §5–§8 with a different denominator, window, or unit of inference, this document has
been violated and the result should be read as exploratory.

**A negative result is an acceptable outcome of every hypothesis below.** Mandate §16 supplies
outcome labels for failure (E, G) as well as for success. None of the thresholds here may be moved
after data are observed; §13 is the amendment log, and it is designed so that any change is visible.

---

## 1. Facts established before freezing (not assumptions)

| Fact | How established | Value |
|---|---|---|
| API reachable | one probe call, `google/gemma-3-27b-it` | ok, 1.8 s, $3.9e-06 |
| Key source | `phase2_llm.load_key()` falls back to `KDD_KEYFILE` | file present outside this worktree |
| Highest seed ever used in this corpus | scan of every `data/*.jsonl` (`seed`, `market_seed`, `table_seed`) and every `data/*manifest*.json` | **7699** |
| Prompt surface | `balance9_prompts.py`, 40/40 gates | hashes in `data/balance9_prompt_hashes.json` |
| P3 prompts equal Balance-8's | gate G1, 144 cases | 0 mismatches |
| Reliability criterion | `balance9_metrics.py`, 44/44 selftest, 3 gates at 0 mismatches | `data/balance9_metric_spec.json` |
| Decoy eligibility rate (diagnostic block) | 120 draws, seeds 990000–990119 | **0.192** |
| Temperature | `phase2_llm.openrouter_call` payload | `0` (env-overridable; not overridden) |
| Response format | same | `{"type": "json_object"}` |

**Temperature 0 is not determinism.** No `seed` parameter is sent, and provider-side routing and
batching can change a completion between two byte-identical requests. Every design below that needs
two arms to share a response therefore **shares the actual call**, rather than assuming that
identical prompts reproduce identical text. This is why P5 is built the way §7 describes.

---

## 2. Frozen seed blocks

Disjoint by construction, and disjoint from all prior rounds because the highest seed ever used
previously is 7699.

| Block | Range | Use | Unit |
|---|---|---|---|
| **B9-P2** | 50000–50599 | P2 rational audits (mandate §8). **No LLM call ever touches this block.** | market seed |
| **B9-P3** | 9000–9059 | P3 confirmatory replication (60 market seeds) | market seed |
| **B9-P3X** | 9060–9179 | P3 extension, only if a complete additional wave finishes | market seed |
| **B9-P4** | 20000–20999 | P4 one-shot mechanism (scan ascending, take first 60 **eligible** per policy) | table/market seed |
| **B9-P5** | 30000–30059 | P5 retry factorial (60 seeds) | table/market seed |
| **B9-P6** | 40000–40059 | P6 system-aware guard, only if the P2 rational gates pass | market seed |
| **B9-DIAG** | 99xxxx | gates, fixtures, eligibility measurement, pipeline tests | never analysed |

Rules:

1. **Seeds are paired across every arm, interface, policy and model.** The same 60 market draws and
   the same complaint CRNs are used everywhere within an experiment, so every comparison is paired.
2. A seed that enters one experiment never enters another.
3. B9-P3X exists so that an extension, if it happens, is a preregistered extension rather than a
   top-up chosen after seeing that 60 seeds nearly reached significance. **If B9-P3X is used, both
   the 60-seed and the 180-seed results are reported, and the 60-seed result is the primary one.**
4. B9-DIAG results never appear in a confirmatory table. The 0.192 eligibility rate in §1 is a
   B9-DIAG number and is **not** the P4 eligibility rate; that one is recomputed on B9-P4 and
   reported separately.
5. **B9-P2 is split, in advance, into three contiguous sub-ranges.** These are *designations inside
   one block*, not separate blocks, because a sub-range of a block cannot be "disjoint" from the
   block that contains it and a disjointness gate that was asked to prove otherwise would be a gate
   built to pass:

   | Split | Range | Size | May be inspected |
   |---|---|---|---|
   | `TRAIN` | 50000–50299 | 300 | freely; this is where search happens |
   | `VALIDATION` | 50300–50449 | 150 | **only after** the candidate under test is frozen |
   | `TEST` | 50450–50599 | 150 | **once**, after `VALIDATION` has been read, and never again |

   The exact transition-graph certificates of mandate §8.1 are computed on the **first 100 seeds of
   `TRAIN` (50000–50099)**, designated `TRAIN-CERT`, for every policy audited. This subset is named
   now, before any graph is built, so that "the markets where the certificate happened to be clean"
   can never become the certification set.

---

## 3. Metric IDs

Every confirmatory quantity below is one of these. The definitions, thresholds, windows, tie rules,
denominators, payoff objects and missing-data behaviour are in `data/balance9_metric_spec.json` and
are not restated here, because a restatement is a second definition.

| ID | What it is | Note |
|---|---|---|
| `M1a_eps_max_rel_L6` | per-round relative restricted exploitability, **Balance-6 denominator** (payoff at the played profile) | unbounded improvement ratio |
| `M1b_eps_max_rel_L7` | same numerator, **Balance-7 denominator** (best deviation) | normalised regret in [0,1) |
| `M2_state_balance` | per-round balance indicator | conjunction of the exploitability and GMV legs |
| `M3_stability` | tail drift of `F` | |
| `M4_sustained_balance` | **strict** sustained balance: ≥10 consecutive OK rounds | explicitly *not* "mean F over rounds 61–80" |
| `M5a_t_first_sustained_L6` / `M5b_t_first_streak_L7` | time to first sustained balance | differ by exactly 9 rounds by construction; verified on 81 runs |
| `M6_gmv_ratio_projected` | projected `GMV/G*` | **not** bounded above by 1 |
| `M7_gmv_ratio_actual` | actual `GMV/G*` — replay of the rational profile through the real dynamics under the run's own CRNs at the same round | new in Balance-9; E-7-gated |
| `M8a_tail_eps_mean_L6` / `M8b_tail_eps_median_L7` | tail exploitability statistics | mean vs median, different windows |
| `M9_round1_exact_argmax` | round-1 exact-argmax rate | reported separately, per §13 of the mandate |
| `M10_tail_exact_argmax` | tail exact-argmax rate | |
| `M11_per_seed_trajectory` | per-seed trajectory outcome | the unit of inference |
| `M12_pooled_decision_proportion` | pooled decision proportion | **descriptive only — may never carry an interval or a p-value** |
| `VALIDITY` | technical-validity filter | Level-A leg A4 |

Both `M1a` and `M1b` are computed and reported for every confirmatory cell. Neither is called "the"
exploitability. `M6` and `M7` are both reported; `M7` is never substituted for `M6` in Level-A leg A3,
because the rule names the projected object.

---

## 4. The reliability standard (Level-A), carried unchanged

| Leg | Condition |
|---|---|
| **A1** | strict sustained-balance probability (`M4`) ≥ **0.80**, computed over market seeds, never over rounds |
| **A2** | median tail relative restricted exploitability ≤ **0.02** |
| **A3** | mean **projected** `GMV/G*` (`M6`) ≥ **0.95** |
| **A4** | technically valid runs only (`VALIDITY`), exclusion rate reported |
| **A5** | uncertainty intervals reported — Wilson for A1, bootstrap over seeds for A2 and A3 |

A1–A3 are evaluated under **both** exploitability definitions. A verdict whose interval straddles its
threshold is marked `PASS*` / `fail*` and never reported flat. These thresholds are the originals and
are not adjustable by this document or by any later one.

For reference, the same rule applied to the **`balance7_main`** prefix (480 runs = 2 models × 4
interfaces `U/H/R/G` × 2 policies × 30 seeds) gives, identically under both exploitability
definitions and on the same 12 cells:

| Verdict | Cells | Meaning |
|---|---|---|
| `PASS` | **7** / 16 | all three legs clear, intervals clear of the threshold |
| `PASS*` | **5** / 16 | point estimate clears, **interval straddles a threshold** |
| `fail` / `fail*` | **4** / 16 | at least one leg fails |

**This table is quoted as 7 + 5, never as "12/16".** Collapsing the two rows would promote five
marginal cells to full passes and so break rule A5 — stated two paragraphs above — which requires
that a straddling interval be reported as PASS-marginal "never as PASS full stop". A prior draft of
this document did quote the flat 12/16; it was corrected before any Balance-9 confirmatory call was
made, and the correction is recorded in §13.1 rather than §13, because no data had been observed.

Two further scope facts about this reference number, both verified against the raw corpus rather
than recalled:

- it comes from **Balance-7**, not Balance-8. **Balance-8 never ran `P_robust` at all** — every
  `balance8_*_raw.jsonl` is `P_GMV` or `P_sep` only. That absence is itself part of why the mandate
  demands a two-policy confirmation, and it means no Balance-8 cell can corroborate the
  `P_robust` column above;
- the counts are **not** a prediction and **not** a baseline for comparison. P3's cells are fresh,
  paired across both policies, and scored on their own. Nothing in §5 compares a P3 cell to a
  Balance-7 cell.

---

## 5. P3 — clean two-policy replication (mandate §9)

### 5.1 Design

2 model families × 2 policies × 6 interfaces × 60 fresh paired market seeds × 80 rounds × 4 merchants.

| | |
|---|---|
| Models | `gemma` = `google/gemma-3-27b-it`, `llama` = `meta-llama/llama-3.3-70b-instruct` |
| Policies | `P_GMV` = (κ=0.5, τ=0.20), `P_robust` = (κ=**4.0**, τ=**0.30**), both read from the raw corpus. `P_sep` = (8.0, 0.02) and `P_safe` are **not** substituted |
| Interfaces | `U`, `H`, `R`, `G` (LLM); `M`, `O` (no LLM) |
| Seeds | B9-P3, paired across every cell |
| Retry budget | 2 economic retries (`R`, `G`), η = 0.01; schema repair 2, information-free, every interface |
| Prompts | `balance9_prompts.p3_prompt`, hashes in `data/balance9_prompt_hashes.json` |

**Launch order, frozen now** (the plan's triage §8, made concrete so that a truncated run is a
preregistered subset rather than a choice made under time pressure):

1. wave 1 — `M`, `O` (no API), then `U`, `H`;
2. wave 2 — `R`;
3. wave 3 — `G`.

A wave is analysed only when every cell in it is complete under `BALANCE9_PROTOCOL.md` §11. **An
incomplete cell is never analysed, never partially reported, and never compared to a complete one.**

### 5.2 Primary family F1

Unit of inference: **market seed**. All tests are paired at the seed level. Outcome variable:
`M4_sustained_balance` (a per-seed binary), scored under `M1b` for the primary and re-reported under
`M1a`. Test: exact McNemar (binomial on discordant pairs). Direction as stated. Family-wise
correction: **Holm within F1**, which contains **16 tests** — 4 hypotheses × 4 (model × policy) cells.

| ID | Hypothesis (per model × policy cell) | Direction |
|---|---|---|
| **H1** | `M4(H) > M4(U)` — a correct factual annotation raises strict sustained balance | one-sided |
| **H2** | `M4(R) > M4(U)` — a verified retry loop raises it | one-sided |
| **H3** | `M4(G) > M4(R)` — the override adds reliability the retry loop does not deliver | one-sided |
| **H4** | `M4(U) < M4(M)` — the unaided model is below the machine argmax reference | one-sided |

The directions are one-sided because each is a claim about a *mechanism adding reliability*, and the
opposite finding (an interface that makes things worse) would refute the mechanism just as
decisively; a two-sided test would dilute power to protect against a direction no one has proposed.
**A significant effect in the reverse direction is reported as a refutation, in bold, in the primary
table** — one-sided testing is not permission to ignore the other tail.

### 5.3 Secondary comparisons (reported with intervals, not Holm-corrected within F1)

* each LLM arm vs `M`, and each LLM arm vs `O`, on `M4`, `M8b`, `M6`, `M7`;
* `P_GMV` vs `P_robust` within interface, **explicitly acknowledging weak action separation**:
  `BALANCE9_THEORY.md` §1.6 shows the reputation gradient is destroyed at both ends of the penalty
  scale, so a harsher policy is not automatically a stronger incentive, and a null here is a fact
  about the policy pair, not about policy responsiveness in general;
* `M9_round1_exact_argmax` and `M10_tail_exact_argmax`, reported separately per mandate §13;
* type responsiveness, retries, overrides, technical repairs, failures, calls, tokens, latency.

### 5.4 `H` vs `M` is an empirical adherence measurement, not an identity

Errata E-1: within-arm override rate 0.65222 versus cross-arm agreement at the same index 0.64975 —
the two diverge because the trajectories diverge. Preregistered as: report `H`'s **empirical action
agreement with `M`'s action at the same (seed, round, merchant) index**, and separately `H`'s
adherence to its own mark. Neither is asserted to be 1 by construction, and no test treats them as
the same quantity.

### 5.5 `G` vs `R`: the equivalence margin, frozen before analysis

Failure to reject is not equivalence. The equivalence claim, if made, is about the per-seed
sustained-balance rate `M4`, and the margin is fixed **now**:

> **Δ = 0.05** in absolute sustained-balance probability.

Justification, recorded before seeing the data: Level-A's A1 threshold is 0.80, and a five-point
difference is one quarter of the distance from that threshold to certainty; declaring a pair
"equivalent" while allowing them to differ by more than that would make the word useless. A margin
tighter than 0.05 is not measurable at n = 60 — the Wilson half-width at p ≈ 0.9, n = 60 is about
0.07 — so a tighter margin would guarantee an inconclusive answer and pretend to rigour.

Procedure: paired TOST on the seed-level difference, using the exact paired binomial. **Pre-committed
three-way verdict:**

* **equivalent** if the 90% CI for the paired difference lies entirely inside (−0.05, +0.05);
* **different** if H3 rejects;
* **inconclusive** otherwise — and specifically, if the CI half-width exceeds Δ, the report says
  *"the design cannot distinguish equivalence from a 5-point difference"* and states the achieved
  precision. It does not say "no difference".

### 5.6 Power, stated in advance

With 60 paired seeds and Holm over 16 tests, McNemar has good power only for large effects. Balance-8
observed U-vs-G gaps between 0.13 and 1.00 in the sustained-balance rate, which are large. Effects
below roughly **0.10** in absolute rate are unlikely to survive correction. Accordingly: **any null in
F1 whose observed effect is under 0.10 is reported as "not powered", not as "no effect".** This
sentence exists so that I cannot later present an underpowered null as a finding.

---

## 6. P4 — one-shot mechanism experiment (mandate §10)

This is the experiment that mandate §17 names as mandatory alongside P3: it "separates correct
computation from supplied-answer following". It is the reason the primary endpoint below is a
conjunction and not a rejection rate.

### 6.1 Design

| | |
|---|---|
| Arms | `U1`, `HF`, `HR`, `HD_BAL`, `HD_EXT`, `RO`, `N` (LLM) + `M` (reference) |
| Cells | 2 models × 2 policies |
| Tables | 60 **eligible** table/market seeds per policy, scanned ascending from B9-P4, shared across models |
| Round | one shot, `t = 1`, no dynamics, no history |
| Prompts | `balance9_prompts.p4_prompt`, hashes on disk |
| Row order | randomised per (seed, arm) with a local RNG; **true index labels retained**, so no remapping |

**Arm `HR` is the recommendation arm** (mandate §10). It is *not* Balance-8's `HR`, which was the row
permutation arm. The collision is recorded in `data/balance9_prompt_hashes.json` under
`name_collision` so that no later reader can conflate them.

### 6.2 Eligibility, frozen before any call

Implemented in `balance9_prompts.decoy_pair`, deterministic, no RNG:

1. displayed argmax unique at τ_tie = 1e-12;
2. argmax non-boundary (0 < k* < 20);
3. candidates have relative displayed regret in the frozen band **[0.04, 0.06]** — outside η = 0.01;
4. **both directions must be populated**: at least one candidate below k* (lower fabrication) and one
   above (higher fabrication);
5. the pair minimising |regret_low − regret_high| is chosen; ties broken by distance from the band
   midpoint, then by lower index.

The **eligibility rate on B9-P4 is reported**. The diagnostic-block rate is 0.192, so roughly 320
draws are expected to yield 60 eligible tables; the scan count is recorded.

**Scope limit, stated before results:** eligibility selects tables that *have* a matched two-sided
decoy in a 2-point regret band. P4's estimates describe that population, not all tables. This is
inherent to the mandate's design and is a limitation, not a defect.

### 6.3 Primary endpoint

```
primary = P( action ∈ A(Ṽ)  |  mark ∉ A(Ṽ) )
```

where `A(Ṽ) = { i : Ṽ(i) ≥ max Ṽ − τ_tie }`. Denominator: the conflict trials, i.e. arms `HD_BAL` and
`HD_EXT` pooled. Numerator: trials on which the model **both** declined the false mark **and** landed
on the true argmax. Implemented once, in `balance9_prompts.p4_classify`, field
`rejected_decoy_and_correct`.

This is deliberately **not** `P(action ≠ mark | mark ∉ A(Ṽ))`. A model that ignores the mark and picks
a third, also-wrong action has not demonstrated payoff execution. `ignored_decoy` is tracked
separately and is never folded into execution.

**Mandate threshold (Outcome B): ≥ 0.70.** The outcome label is decided by the point estimate against
0.70, which is the mandate's literal wording. The 95% cluster-bootstrap lower bound is reported
beside it so a reader can see whether the verdict is secure; if the point estimate clears 0.70 while
the lower bound does not, the report says so in the same sentence as the claim.

### 6.4 Primary family F2

Unit of inference: **table/market seed** (clustered; models and policies are within-seed factors).
Policies are pooled for the primary and reported as a moderator — policy is not P4's question.
Family: **10 tests** = 5 hypotheses × 2 models. Holm within F2. Tests are paired at the seed level:
exact McNemar for binary arm contrasts, paired cluster bootstrap for rates.

| ID | Hypothesis | Direction | Reads on |
|---|---|---|---|
| **H5** | accuracy on conflict trials < accuracy on `U1` (an unmarked table) — a false mark *degrades* correct selection | one-sided | supplied-answer following |
| **H6** | `P(action = d | HD arm) > P(action = d | U1)` for the same table and the same index d — marking an index raises the probability of choosing it | one-sided | annotation influence, causally identified |
| **H7** | accuracy on `HF` > accuracy on `U1` — a correct factual mark raises accuracy | one-sided | annotation transport |
| **H8** | `P(argmax | RO) > P(argmax | N)` — a recommendation without a table still moves behaviour | one-sided | recommendation compliance |
| **H9** | `P(follow | HD_BAL) ≠ P(follow | HD_EXT)` — directional asymmetry between a lower- and a higher-fabrication decoy | **two-sided** | directional preference |

H9 is two-sided because both directions are scientifically interesting and neither was predicted:
a model that preferentially follows the *balanced* decoy and one that preferentially follows the
*extractive* decoy imply opposite things about what the annotation is doing.

H6 is the sharpest test in the experiment and the reason `U1` exists: it compares the same index on
the same table with and without a mark, so a positive result cannot be explained by that index being
attractive on its own.

### 6.5 Secondary

`HF` vs `HR` (fact vs recommendation framing); `U1` vs `N` (effect of table removal); `M` reference;
position sensitivity from the row-order randomisation; per-model differences; the four-way
classification (annotation copying / payoff execution / neither / mixed) as a descriptive table.

### 6.6 Scope limit, stated before results

P4 may support claims about annotation influence, independent table use, directional preference and
recommendation compliance. It may **not** support a claim of strategic convergence, and no sentence
in any Balance-9 report will use it that way.

---

## 7. P5 — retry factorial (mandate §11)

### 7.1 Design, and why the first attempt is shared

Balance-8's `R0`-vs-`R` contrast moved the tolerance *and* the retry wording together. The 2×2:

| | B1: generic factual failure | B2: explicit objective restatement |
|---|---|---|
| **A1: η = 0.01** | `A1B1` | `A1B2` |
| **A2: η = 0 (exact)** | `A2B1` | `A2B2` |

The initial prompt is byte-identical across all four arms — gate G4 proves it, and the design goes
further: **one first attempt is drawn per (seed, model, policy) and shared by all four arms.** Two
reasons, both recorded before the run:

1. temperature 0 is not determinism (§1), so four separately-drawn first attempts would differ by
   provider noise and the "byte-identical initial prompt" requirement would be satisfied in text and
   violated in fact;
2. §11 asks for the second-attempt exact rate "conditional on the **same** first-attempt failure
   set". With a shared first attempt that set is literally the same set, not a matched one.

Factor A then acts as a trigger rule on the shared first attempt: the η = 0.01 arms retry iff the
first action's displayed regret exceeds 0.01; the η = 0 arms retry iff it is not the exact argmax.
Factor B determines the message sent. Retry budget 2, as in P3, so factor A also acts on the
*acceptance* of the second attempt — which is what makes the interaction estimable.

### 7.2 The generic message is checked, not asserted

`balance9_prompts.retry_generic` is verified against a lexicon frozen *before the message was
written* (`OBJECTIVE_LEXICON`: highest, maximis, maximiz, best, greatest, largest, optimal,
shortfall, tolerance, profit, better, improve, higher). Gate G5a passes with zero hits; G5b requires
the explicit message to contain at least four. The generic message reports no shortfall and names no
tolerance, because a shortfall is measured against a maximum and quoting one announces the objective.
It is also **invariant to η** (G5h), so it cannot leak factor A's level.

`retry_explicit` at η = 0.01 is byte-identical to Balance-8's message (G5c): the factorial contains
the incumbent arm rather than replacing it.

### 7.3 Primary family F3

Unit of inference: **table/market seed**. Family: **6 tests** = 3 hypotheses × 2 models. Holm within
F3. Policies pooled for the primary, reported as a moderator.

| ID | Hypothesis | Direction | Estimand |
|---|---|---|---|
| **H10** | main effect of tolerance on the final exact rate | one-sided (η = 0 higher) | full sample |
| **H11** | main effect of explicit wording on the second-attempt exact rate | one-sided (explicit higher) | **common subset**: seeds where both η levels trigger a retry, i.e. first-attempt regret > 0.01 |
| **H12** | tolerance × wording interaction | two-sided | common subset, difference-in-differences |

Also measured, reported without correction: first-attempt exact rate; first-attempt within-η rate;
probability of retry; final within-η rate; number of corrections; action-direction change; calls,
tokens, latency.

**Later dynamic states are never pooled to identify this effect.** The one-shot matched-table design
is the causal instrument. A smaller fresh end-to-end dynamic confirmation runs afterwards only if
platform consequences need establishing, and if it runs it is reported as a separate experiment with
its own seeds.

---

## 8. P6 — system-aware guard (mandate §12), conditional

P6 runs **only** if the P2 rational gates pass and the threshold and platform reference are frozen and
validated first. If it runs:

* the old no-LLM "most balanced member of the certificate" rule is called **`MBM`**, never "the dual
  guard";
* a genuine dual guard `DG` is defined only after its guarantee is proved, and **only the one-step
  guarantee actually enforced** is claimed;
* the acceptance rule (private displayed regret ≤ η **and** one-step projected `GMV/G_ref` ≥ a frozen
  threshold), the platform reference, the rival convention, unilateral-vs-joint, projected-vs-realized,
  tie handling and the safe fallback are all specified in `BALANCE9_PROTOCOL.md` before the first call;
* if reliable outcomes require frequent overrides, the result is classified as **programmatic
  enforcement**, not as model capability.

Its hypotheses are not preregistered here because its reference object is not yet frozen. **If P6 runs,
a preregistration amendment is written and hashed before its first call** (§13). If that amendment does
not exist, P6's results are exploratory by definition.

---

## 9. Statistical rules that apply to every experiment above

* **The market seed is the unit of inference.** For dynamic experiments, rounds and merchant
  decisions are never independent observations. For one-shot diagnostics, inference clusters by
  market/table seed, not by repeated model response.
* Paired seed-level effects; paired bootstrap CIs (10,000 resamples over seeds); paired sign-flip
  permutation tests; exact paired binary inference or McNemar; **Wilson** (never Wald) for standalone
  rates; Holm within the frozen families F1, F2, F3; effect sizes always; per-seed outputs retained.
* Reported separately and always: **round 1** (`M9`), **tail window** (`M10`, `M8a`, `M8b`),
  **per-seed trajectory** (`M11`), and pooled decisions **as descriptive only** (`M12`, which may
  never carry an interval or a p-value).
* **Missing data / technical failure:** the symmetric six-rule policy of `BALANCE9_PROTOCOL.md` §5
  applies identically to every LLM arm. Runs with an unresolved technical failure are excluded by
  `VALIDITY` and the exclusion rate is reported per cell (leg A4). Exclusion is never conditioned on
  the outcome.
* **Kept separate, never pooled:** pilot; confirmation; mechanism diagnostics; retry factorial;
  policy selection; rational validation; rational test; system-guard experiments; and all Balance-6,
  Balance-7 and Balance-8 data.

---

## 10. Outcome rules from mandate §16, with every vague word given a number now

The mandate's outcome labels use words like "often" and "most". Numbers are assigned here, before the
data exist, so that the outcome cannot be selected by choosing a reading afterwards. Outcomes are
**not mutually exclusive**; A and H can both hold, and on Balance-8 evidence both are live.

| Outcome | Mandate condition | Number frozen here |
|---|---|---|
| **A** | `H` replicates near-perfect conformance under both policies while decoy marks are "often followed" | `H` passes Level-A in all 4 cells **and** decoy-following rate ≥ **0.50** on conflict trials |
| **B** | `H` replicates and models reject matched false marks while selecting the true argmax on ≥ 70% of conflict trials | the primary endpoint of §6.3 ≥ **0.70** (point estimate; LCB reported beside it) |
| **C** | `R`/exact retry succeeds but "most" first-attempt errors are repaired by the loop | loop share = (final within-η − first-attempt within-η) / (1 − first-attempt within-η) > **0.50** |
| **D** | only `G0`, `DG` or `M` reaches the standard | the Level-A pass set contains no unaided LLM arm: `U`, `H` and `R` all fail in all 4 cells |
| **E** | the clean replication fails under `P_robust`, or fails the original criterion | any `P_robust` cell that passes under `P_GMV` fails Level-A |
| **F** | a practical `P_safe` improves held-out robustness within the 5% equilibrium-GMV cost | stated **only** if both rational VALIDATION and TEST pass |
| **G** | no practical separating policy exists | the §8.3 search returns no admissible policy with separation above its frozen bar |
| **H** | private regret stays small while platform losses vary materially | median executed displayed regret ≤ **0.01** while the max−min spread of mean `M7` across arms within a cell exceeds **0.05** |

The 0.05 in Outcome H is not invented for the occasion: 5% of equilibrium GMV is already the frozen
admissibility budget of the §8.3 policy search, so it is this corpus's existing scale for "material".

---

## 11. Prompts

The full specimen texts are stored, not paraphrased, in `data/balance9_prompt_hashes.json` under
`texts`, together with their SHA-256 digests under `hashes`, the frozen objective lexicon, the decoy
band, the tie tolerance, and the source hash of `balance9_prompts.py` itself. The runner is forbidden
to construct prompt text; anything sent that did not come out of that module is unpreregistered by
construction.

Published on the fixture (seed 999950, policy (1.0, 0.25), round 2 for P3 / round 1 for P4 — a
B9-DIAG display fixture that enters no analysis):

| Key | SHA-256 |
|---|---|
| `P3::U::system` | `a83948317db9561afb39532afd3937a31b554c53b07ee2465382314c06c03c3b` |
| `balance9_prompts.py` source | `9e19cfdc6a41ae397ee356ac579bace8f50518f71649a7068569258b03e8652f` |

The remaining keys — `P3::H`, `P3::R`, `P3::G`, the seven `P4::*` arms, the four `P5::*` arms, and the
four message digests — are in the JSON. `P3::R` and `P3::G` equal `P3::U` by gate G2; the four `P5`
entries are all equal to `P3::U` by gate G4b. That equality is the preregistered claim, and it is
checkable from the file.

**Verbatim, the three sentences that carry the manipulations:**

* factual mark (`H`, `HF`): `" The highest projected profit in this list is at index [{k}]."`
* recommendation (`HR`, `HD_BAL`, `HD_EXT`, `RO`): `" One option worth considering is index [{k}]."`
* generic retry (`P5` factor B level 1): `"Your submitted action_index [{i}] (f={f}) did not pass the automatic check applied to your response. The information you were given is unchanged. Please reconsider and respond again in the same JSON format."`

---

## 12. Analyses I am not permitted to run

Listed so that doing one is visibly a violation rather than a judgement call.

1. Reporting `M12_pooled_decision_proportion` with a confidence interval or a p-value.
2. Treating rounds or merchant-decisions as independent observations in any dynamic experiment.
3. Substituting `M7` for `M6` in Level-A leg A3, or reporting one exploitability metric without its ID.
4. Redefining sustained balance as a tail mean, a fraction of OK rounds, or anything other than `M4`.
5. Using a Wald interval on a rate at or near 0 or 1.
6. Adding seeds to a cell after seeing its p-value, other than the preregistered B9-P3X wave, which
   is reported alongside — never instead of — the 60-seed primary.
7. Analysing an incomplete cell, or comparing an incomplete cell to a complete one.
8. Reading "failure to reject" as equivalence anywhere, including in the `G`-vs-`R` comparison, where
   the only admissible equivalence claim is the frozen TOST of §5.5.
9. Reporting `H`-vs-`M` as an arithmetic identity.
10. Pooling later dynamic states to identify the P5 factorial effect.
11. Claiming strategic convergence from P4.
12. Claiming long-run GMV safety from a one-step guard guarantee.
13. Moving any threshold in §4 or §10 after data are observed, in either direction.
14. Quoting a Balance-8 number that Balance-9 has not reproduced. Concretely, for §17 item 24
    (calls, tokens, latency): the manifest field `progress.calls` is a stale checkpoint counter, not
    a call count — `data/balance8_main_manifest.json` reads **5,279** while the per-merchant `calls`
    field summed over `balance8_main_raw.jsonl` gives **159,850**. The correct figure is the one
    Balance-8's own `balance8_call_audit.json` already derives from the raw records (errata E-5,
    corpus total 429,215), and it is reproduced here independently: my scan returns 159,850 for the
    `main` cell, equal to the audit's `main` total. Balance-9 reads call counts from raw records,
    never from a manifest counter.

---

## 13. Amendments

Any change to this document after the first confirmatory call is an amendment. An amendment must be
added here, with its date, the exact change, the reason, and **whether any data had been observed at
the time**. An amendment made after data are observed converts every affected test to exploratory.

| # | Date | Change | Data observed? | Reason |
|---|---|---|---|---|
| **A-1** | 2026-08-13 | §2 gains the **B9-P2** block (50000–50599) and rule 5, which splits it into `TRAIN` / `VALIDATION` / `TEST` and names `TRAIN-CERT`. | **No.** Zero confirmatory calls have been made; no rational audit had been run when this was written; no graph, policy score or split outcome had been computed. | The document allocated seed blocks for P3–P6 and for diagnostics but **none for the P2 rational audits**, although mandate §8.4 requires disjoint rational TRAIN/VALIDATION/TEST splits. Discovered while building `balance9_graph.py`, before its first execution. |
| **A-2** | 2026-08-13 | `BALANCE9_PROTOCOL.md` §14.7 gains the sentence fixing the **GMV-ratio denominator** to `G*_{P_GMV}` on the same market; `balance9_ratspec.SEL_DENOM` = `gstar_pgmv`. | **Yes**, in part — and the part matters. The §8.1 transition-graph certificates and the η-tolerance sweep on `TRAIN-CERT` existed (03:39). **No policy had been scored under either candidate denominator**: the §8.2 audit that first computes a GMV ratio ran seven minutes later (03:46), so no ranking existed to read the answer off. *Consequence:* no §8.4 test is reclassified, because the data in hand contained no instance of the comparison A-2 decides. | §14.7 wrote "GMV ratio" three times without naming a denominator. The two available — the policy's **own** equilibrium GMV, and `G*_{P_GMV}` on the **same market** — order candidates differently, so an unfixed denominator is a free parameter that could be settled after seeing which ordering was preferred. The own-GMV denominator rewards a candidate for being bad at equilibrium, which is the failure admissibility rule S1 exists to exclude. |
| **A-3** | 2026-08-13 | `BALANCE9_PROTOCOL.md` §14.6 gains the definition of **the searched class** — the 13 × 11 = 143-point (κ, τ) lattice in `balance9_ratspec.SEARCH_KAPPA` / `SEARCH_TAU` — and the rule that only the `TRAIN` shortlist is carried to `VALIDATION`. | **Yes.** The §8.2 audit of `P_GMV`, `P_robust` and `P_sep` on `TRAIN-CERT` had completed (03:46) and all ten of its quantities were known, including that `P_robust` slightly outperforms `P_GMV`. No other member of the class had been evaluated on any split. *Consequence:* under the rule at the head of this section, **the §8.3 result is reported as exploratory with respect to the choice of class.** The TRAIN → shortlist → VALIDATION procedure *within* the class remains confirmatory as written. | §14.6 pre-commits a sentence ending "in the searched class" while no frozen document said what that class was, so any negative result could afterwards be re-described as having searched something else. The lattice is regular and exhaustive with no adaptive refinement, and check `C9q3` requires it to contain all three §14.5 policies — a class that omitted the incumbents could report "nothing admissible" without ever evaluating the policies under discussion. |
| **A-4** | 2026-08-13 | `BALANCE9_PROTOCOL.md` §14.7 gains the definition of the **`P_safe` candidate class**, identified with the class A-3 had already frozen; `balance9_ratspec.PSAFE_CLASS` = `A3_GRID`. | **Yes.** All of §8.3 had run (04:08): the `TRAIN` equilibrium GMV ratio, fabrication, outside share and action separation of all 143 candidates were known, as were the `VALIDATION` rows of the three shortlisted ones. **The statistic that selects `P_safe` — the mean worst *joint* η-acceptable GMV ratio — had been computed for no member of the class** except the three §14.5 policies, and those on `TRAIN-CERT` only. *Consequence:* **§8.4's selection is reported as exploratory with respect to the choice of candidate class.** The freeze-before-`VALIDATION` step and the read-once discipline on both held-out splits remain confirmatory. | §14.7 says "for every candidate" and "restrict to candidates" without naming a candidate set — the same undefined referent A-3 removed from §14.6. "We selected the most robust policy" says nothing until the set it was selected from is fixed. A-4 does not invent a class: it identifies §14.7's unnamed set with the lattice already frozen at 03:55, because inventing a fresh class *after* seeing §8.3's frontier is precisely the move this rule exists to prevent. |
| **A-5** | 2026-08-13 | `BALANCE9_PROTOCOL.md` §14.7 gains the **aggregation** of the per-market ratios into the per-candidate number the selection rule compares — `mean_of_ratios`, not ratio-of-means — and the full ordered key list `(worst_joint_ratio_pgmv, gmv_ratio_vs_pgmv, class_index)` that makes the rule a function. `balance9_ratspec.SEL_AGG` and `SEL_KEYS`. | **Yes**, and asymmetrically — which is the part worth stating. §8.3 computed the `mean_of_ratios` equilibrium leg for all 143 candidates on `TRAIN` (its rule S1 *is* that quantity), so the shortlist this reading produces is already known; the ratio-of-means reading has been computed for **no** candidate. I am therefore choosing between a reading whose consequence I know and one whose consequence I do not. **The statistic the rule selects on — the mean worst joint η-acceptable GMV ratio — is still computed for no member of the class**, so knowing the membership of the filter does not reveal which candidate the filter admits as the winner. *Consequence:* §8.4's selection is already exploratory with respect to the candidate class under A-4, and A-5 does not extend that label, because a choice that cannot be responsive to the ranking cannot have been made to obtain one. The freeze-before-`VALIDATION` step and the read-once discipline remain confirmatory. | §14.7 says "mean equilibrium GMV" and "mean … GMV ratio" without saying whether the mean is taken before or after the division. The two readings can admit different candidates: ratio-of-means lets a few large-`G*` markets outvote the rest, which is precisely the market-weighting freedom the paired per-market design exists to remove. Left unfixed it is a free parameter inside a preregistered rule, settled by whoever writes the selection code. Fixing it *to §14.6's existing arithmetic* is the choice that adds no new convention; adopting the other reading would have made one rule perform two different aggregations on its two legs. The key list is added for a second reason: as written the rule can return two answers, and an implementation that silently picks one would be the undocumented step this whole section is built to exclude. |
| **A-6** | 2026-08-13 | `BALANCE9_PROTOCOL.md` §14.7 gains the definition of **"replicates on a held-out split"** (R1 advantage under A-5's aggregation, R2 the 0.95 admissibility floor re-applied on the held-out markets, R3 paired per-market win rate ≥ 0.60 with ties counted as losses), the pre-committed failure sentence, and the disclosure that **no minimum effect size is preregistered**. `balance9_ratspec.HELDOUT_POLICIES`, `REP_WIN_MIN`, `REP_NEGLIGIBLE`. | **Yes**, but this is the earliest amendment in this table relative to the data it governs. Observed: everything through §8.3, i.e. `TRAIN` equilibrium quantities for all 143 candidates. **Not observed: the selection statistic for any candidate on any split — `TRAIN` included — because §8.4 had not been run when this was written; no `P_safe` existed; and neither held-out split had been read.** The quantity R1 and R3 are stated over therefore did not exist in any form for the policy they judge. *Consequence:* none of §8.4's held-out tests is reclassified. The exploratory label A-4 attaches to the choice of candidate class stands unchanged; the replication verdict itself is confirmatory, because its criterion predates the number it is applied to. | §14.7 makes the whole LLM `P_safe` experiment conditional on the advantage "replicating on both held-out splits" and never says what replication is — the same undefined referent A-3, A-4 and A-5 each removed, and the costliest instance of it, because this is the word that decides whether a result is written up as a success. Left open until the splits were read, it would have been settled by whatever the numbers supported: a mean gap of 10⁻⁹ is an advantage under one reading and noise under another. R3 exists because R1 is a statement about an average and one extreme market can carry an average. The absence of a minimum effect size is disclosed rather than fixed, because every floor available to write was either arbitrary or read off a gap already measured on `TRAIN-CERT` in §8.2 — and `REP_NEGLIGIBLE` is a labelling rule that changes the words and never the verdict. |

| **A-7** | 2026-08-13 | `BALANCE9_PROTOCOL.md` §10 gains the standing **reporting rule for multi-condition verdicts**: whenever a result takes the form "*n* of *N* candidates cleared *k* conditions", the pairwise rank correlations of those *k* conditions over the candidate set are reported beside it. | **Yes**, and fully — this is the most data-exposed amendment in the table, and it is also the only one that cannot be used by the person who wrote it. Observed: all of §8.3, including the finding that produced it. §8.4's `TRAIN` sweep was mid-flight (40 of 300 markets); its per-candidate selection statistic had been aggregated for no candidate and read for none, and neither held-out split had been touched. *Consequence:* **no test is reclassified, and the set of affected tests is empty.** A-7 moves no threshold, changes no metric, adds and removes no candidate, and can flip no verdict — it can only add a measured number *beside* a verdict the frozen rule already decided. A rule with no outcome in its range cannot have been adopted to obtain one. The direction is worth recording too: A-7's first application makes this corpus's own published negative result read **weaker**, not stronger, which is the opposite of what a motivated post-data disclosure rule would do. | §8.3's headline — *0 of 143 candidates cleared all five conditions* — reads as five independent hurdles, and on the scored grid it was closer to two: S2 ⊆ S1 exactly (39 of 39), and S3 tracks fabrication at Spearman **ρ = +0.9449**, so "separating" and "low-fabricating" were near-antonyms by construction of the metrics rather than by any fact about the class. The negative result stands — the conditions, thresholds and split order were frozen by A-3 before a candidate was scored — but its *reading* narrows, and a reader given only the count could not have narrowed it. The collinearity is computable in a few lines from the scored grid and **not** computable before scoring, so no pre-data gate could have caught it; that is exactly why the countermeasure is a reporting duty rather than a completion gate, and why it is filed here as an amendment instead of quietly fixed in a report. Errata **E-11**; the correlations themselves are recomputed on every `balance9_doccheck.py` run rather than transcribed. |

**A-2, A-3 and A-4 were missing from this table until they were caught by a repaired `C8d`.** All
three were written into `BALANCE9_PROTOCOL.md` §14 as prose and never given a ledger row, and the
check that is supposed to keep the two ledgers in step read only in the direction preregistration →
protocol, so a protocol-only amendment was outside its domain in both senses. It reported 84/84 with
three amendments unrecorded. `C8d` and its new twin `C8e` now take **every** `A-n` mentioned anywhere
in either document and require a row in **both** tables; the reverse-direction fault is `AF-32`. The
rows above were reconstructed from the recorded modification times of the audit outputs in `data/`,
not from memory, which is why they carry clock times. This is errata **E-10 §9**.

**Why A-1 is an amendment and not a §13.1 correction.** §13.1 is for errors found *before* the
document was hashed. This one was found after. The distinction is not cosmetic: the pre-freeze
corrections changed sentences that were **false**, whereas this adds a section that was **missing**,
and a reader checking the published hash would otherwise find a document that no longer matches it
with no record of why. The honest form of "I forgot something" is a dated row, not a silent edit.

A-1 is a **structural** amendment: it allocates seeds and names splits. It does not touch a
threshold, an endpoint, a metric, a hypothesis or a decision rule — the §8.3 and §8.4 admissibility
numbers it will be used with were already fixed by the mandate and are restated unchanged in
`BALANCE9_PROTOCOL.md` §14. If a later amendment ever needs to move one of those numbers, it
converts the affected test to exploratory under the rule stated at the top of this section, and no
amendment in this document may exempt itself from that rule.

### 13.1 Pre-freeze corrections

Corrections made **before** the gate first passed and **before any Balance-9 confirmatory call**.
They are logged separately from §13 because nothing was at stake yet: no Balance-9 datum existed, so
no test could have been steered by an outcome. They are logged at all because a preregistration that
quietly repairs itself is indistinguishable from one that quietly rewrites itself.

| # | Section | Was | Is | How it was caught |
|---|---|---|---|---|
| PF-1 | §5.1 | `P_robust` = (κ=1.0, τ=0.25) | `P_robust` = (κ=**4.0**, τ=**0.30**) | Corpus scan. (1.0, 0.25) is **Balance-6's `Pstar`** — a real tuple filed under the wrong name, which is why it read as plausible. Now gated by `C5k`. |
| PF-2 | §4 | figure attributed to **Balance-8** | attributed to **`balance7_main`** | Every `balance8_*_raw.jsonl` is `P_GMV`/`P_sep`; Balance-8 never ran `P_robust`, so it cannot have produced a 2-policy table. Now gated by `C5l`. |
| PF-3 | §4 | "12/16 cells passing" | **7 `PASS` + 5 `PASS*`**, quoted separately | Re-derived from `level_A_by_cell`. The flat count silently promoted 5 interval-straddling cells to full passes, contradicting rule A5 in the same section. Now gated by `C5m`. |

PF-3 is the one worth remembering: PF-1 and PF-2 were wrong facts, but PF-3 was a *correct
arithmetic* statement that misrepresented its own evidence. The number 12 was real. It was reported
in a way this document's own rule forbids, and only re-deriving it from the stored per-cell verdicts
exposed that. Each of the three now has a check that fails the gate, because a corrected sentence
with no check behind it is worth no more than the sentence it replaced (errata E-7).

---

## 14. Integrity

`balance9_prereg_freeze.py` computes the SHA-256 of this file, verifies that the three manipulation
sentences quoted in §11 are byte-identical to the constants in `balance9_prompts.py`, verifies that
every metric ID named in §3 exists in `data/balance9_metric_spec.json`, verifies that the seed blocks
in §2 do not intersect each other or any seed used in a prior round, and writes
`data/balance9_preregistration.json`. **No confirmatory call may be made until that script passes**
(gate `G-prereg`, `BALANCE9_PLAN.md` §4).

No paper file was read or written in the production of this document. Nothing has been committed or
pushed.
