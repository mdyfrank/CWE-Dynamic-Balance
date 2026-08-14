# BALANCE9_PROTOCOL.md

Experimental protocol for Balance-9. This document fixes **how** the round is run: the environment,
the interfaces, the raw schema, the technical-failure policy, the validator's obligations, the three
experiment designs, and the statistical rules. It fixes design, not hypotheses — those are in
`BALANCE9_PREREGISTRATION.md`, which is written and hashed before the first confirmatory call.

Nothing in this file may be edited after the preregistration is hashed. If a design defect is found
after that point it is recorded as an amendment with a timestamp, never by silent edit.

---

## 1. Environment

Unchanged from Balance-6/7/8, and deliberately so: the point of Balance-9 is a clean replication of a
frozen environment, not a new environment.

| Quantity | Value | Source |
|---|---|---|
| merchants `m` | 4 | `equilibrium.Config(m=4)` |
| action grid | 21 points, `f = i/20`, `i ∈ {0,…,20}` | `balance9_runner.idx_to_f` |
| rounds per run | 80 | mandate §9 |
| outside-option weight `ω` | 0.5 | `OMEGA` |
| trust erosion `λ` | 1.0 | `LAM` |
| trust inertia `ζ` | as recorded per run | raw field `zeta` |
| initial condition | `r_j(1) = 0.5` ∀j, `T(1) = 1.0`, `f_last(1)` set explicitly | mandate §5 requires round 1's initial condition be recorded, not inferred |
| complaint draw | `U[t, j, n] ~ Uniform(0,1)`, `default_rng(90000 + seed)`, shared across all arms of a seed | `phase2_framing_controlled.crn_uniforms` |
| market draw | `hetero.draw_market_seeded(cfg, seed)` — **never** the global-RNG `draw_market` | discrepancy D5, `data/balance9_metric_spec.json` |

### 1.1 Market step (the realized dynamics)

For round `t`, with state `r(t)` (dynamic reputation), `T(t)` (trust), and executed actions `f(t)`:

```
appeal_j   = q_j + (1 - q_j) f_j(t)
u_j        = α·appeal_j + β·r_j(t) − γ·p_j
s_j        = exp(u_j) / ( exp(w0(ω)) + Σ_k exp(u_k) )
Q(t)       = Q0 · T(t)                      # LAGGED trust
y_j        = Q(t) · s_j
profit_j   = (p_j − c_j) · y_j,  c_j = p_j/2
GMV(t)     = Σ_j p_j · y_j
θ_j        = clip(b_j + c_s·f_j(t), 0, 1)
D_j        = #{ n : U[t,j,n] < θ_j }
d_j        = D_j / N_obs
Pen_j      = κ · max(0, d_j − τ)
r_j(t+1)   = clip( r_j(t) + η_r (1 − r_j(t)) − Pen_j, 0, 1 )
T(t+1)     = (1 − ζ) T(t) + ζ exp(−λ F(t)),   F(t) = mean_j f_j(t)
```

Reputation enters `u_j` at its **pre-update** value and traffic at its **lagged** value. That timing
is the whole content of errata E-7 and of `BALANCE9_THEORY.md` §1, and any function claiming to
implement it must prove it against a recorded transition before its output is used.

### 1.2 The displayed table

Merchant `j` at round `t` is shown 21 payoff values

```
Ṽ_j(i) = profits( f_{-j} = f(t−1),  f_j = i/20 )_j
```

computed by `phase2_exploitability.profits`: **stationary** reputation `rg[b_j, i]`,
**contemporaneous** traffic `Q0·g(mean f)`, and **previous-round** rival actions. This is a
different object from §1.1's realized payoff in two independent ways, and neither difference is a
multiplicative constant, so neither cancels from an argmax (theory §1, Prop 1.1–1.2).

The displayed table is persisted in full at full precision (errata E-3). It is never recomputed
downstream from state, because recomputation is what let a previous round report a table that was
never shown.

---

## 2. Policies

| Name | `κ` | `τ` | Role |
|---|---|---|---|
| `P_GMV` | as frozen in Balance-6 | as frozen | the GMV-optimal penalty policy |
| `P_robust` | as frozen | as frozen | the robustness-oriented policy |

`P_sep` and `P_safe` are **not** substituted into the confirmatory replication under any
circumstance. `P_sep` may appear only as a legacy extreme diagnostic in the policy audit, clearly
labelled as such; `P_safe`, if it exists at all, is a P6 object.

The exact `(κ, τ)` values are read from the frozen policy table at launch and written into the
preregistration and into every raw record, so that a policy named in a report can be checked against
the policy that ran.

---

## 3. Interfaces

Six arms. Four call the model; two do not.

| ID | LLM? | Definition |
|---|---|---|
| `U` | yes | Unlabeled full payoff table. All 21 rows, no annotation, no indication of which is best. |
| `H` | yes | Full table **plus a correct factual argmax annotation**. The annotation states the true displayed argmax as a fact, not a recommendation. |
| `R` | yes | Full table, no annotation. After the model proposes, a verifier computes the displayed relative regret. If it exceeds `η = 0.01`, the model is given a **factual** regret message and asked again, up to **two** economic retries. **No override.** |
| `G` | yes | Exactly `R`, and then: if the final proposal's regret still exceeds `η`, the platform **overrides** to the displayed argmax. |
| `M` | no | Execute the displayed argmax directly. |
| `O` | no | Execute the exact restricted-game best response / equilibrium reference. |

### 3.1 What `R` does and does not guarantee

`R`'s accepted action is **the last proposal the model made**, whatever its regret. The retry loop
exits when the budget is exhausted, not when the regret condition is met. Therefore:

> **`R` carries no accepted-action guarantee and no executed-action guarantee.** Any sentence of the
> form "the verified arms are within η" is true of `G` and false of `R`.

This is `BALANCE9_THEORY.md` Prop 5.2, and it is restated here because it is a property of the
*protocol*, not of a result. `R` is retained precisely because it isolates what the model does when
told it is wrong but not made right.

### 3.2 Guarantee ledger

Every guard statement in this round must name both its arm and its metric. Abbreviating the table
below is how E-2 happened.

| Arm | Quantity guaranteed | On which payoff object | Bound |
|---|---|---|---|
| `M` | executed action | displayed `Ṽ` | exactly the displayed argmax (tie rule below) |
| `G` | executed action | displayed `Ṽ` | relative regret ≤ `η = 0.01` |
| `R` | — | — | none |
| `U`, `H` | — | — | none |
| any arm | realized one-step `W` | realized | **none.** Measured mean realized shortfall of the displayed argmax is 46.3%. |

### 3.3 Tie rule

The displayed argmax is a **set**:

```
A(Ṽ) = { i : Ṽ(i) ≥ max Ṽ − τ_tie },   τ_tie = 1e-12
```

`np.argmax` is never used to define a decision or a score. On a table with two exactly equal maxima
it returns the lower index by array order — a choice made by numpy, not by economics. Every raw
record stores the whole set. "Executed the argmax" means membership in `A(Ṽ)`.

For `M`, when `|A| > 1`, the executed index is `min A`, declared here in advance and recorded in the
raw as `tie_rule="min_index"` so the arbitrariness is visible rather than hidden.

---

## 4. Raw schema (mandate §5)

Every merchant-decision record must let the validator reconstruct the decision **without importing
runner code**. Fields, grouped:

**Market primitives** — `q`, `p`, `b` at full precision; `bidx` (the discrete type index); market
draw provenance (`draw_fn="draw_market_seeded"`, `seed`); policy `(κ, τ)` and policy name.

**State** — `r_current`, `r_previous`; the stationary reputation row `rbar_used` actually indexed to
build the table; `T_before`, `T_after`; `traffic_before`, `traffic_after`; `zeta`; `round`; and for
round 1 the **explicit** initial condition (`r_init`, `T_init`, `f_last_init`) rather than a
convention the validator must know.

**Actions** — the full rival action vector used to construct the table; the current and previous
action vectors; all as grid **indices** `0–20` plus their float images. Internally, indices only:
floats are never dictionary keys.

**Displayed table** — all 21 actions; all 21 payoff values at full precision; `argmax_set`
(including ties); `tau_tie`; the displayed absolute and relative regret of every attempt.

**Decision** — `proposed_index` for **every** economic attempt (a list, not a scalar);
`accepted_index` (the final proposal); `executed_index` (after any override); `override` flag and
`override_target`; `claimed_best_index` if the arm elicits one; `mark_index` and `followed_mark`
where a mark exists.

**Outcomes** — realized one-step deviation payoff vector where computed; realized exploitability;
projected GMV; actual GMV; equilibrium reference values (`eq_index`, `eq_f`, `G*_projected`,
`G*_actual`); demand shares including the outside share; penalties; complaint counts and the CRN
provenance sufficient to regenerate them.

**Technical** — all prompt hashes; the raw model response for **every** attempt; parse/repair
status; transport attempt count; schema-repair count; economic retry count — the three counted
**separately**; latency and token usage per API attempt; provider and model metadata; run ID and
decision ID.

Raw files are append-only JSONL under a new `balance9_*` prefix. No prior raw file is patched.

---

## 5. Technical-failure policy (symmetric across all LLM arms)

Applied identically to `U`, `H`, `R`, `G`. Asymmetry here would make arm comparisons meaningless,
because the arm with the more forgiving failure policy would look more capable.

1. Retry identical transport requests with bounded exponential backoff.
2. Permit the same formatting-only schema repair in every arm.
3. **Never reveal argmax or regret during formatting repair.** A repair prompt that leaks the answer
   converts a parse failure into a free hint, and it would do so only in the arms that happen to
   parse badly.
4. If still unsuccessful, **abort before the state transition**.
5. Re-run the complete market seed later — the whole seed, not the failed round, because a resumed
   run has a different history.
6. **Never substitute** `f = 0.5`, the argmax, the previous action, or any arm-specific fallback.

Counted separately in the raw and reported separately: transport retries; schema-format repairs;
economic verification retries; programmatic overrides. Collapsing these into one "retry" number is
how a verification effect and a network hiccup become indistinguishable.

**Invariant test.** The runner ships a test proving that market state is *bit-identical* before and
after every attempt of the same decision: reputation, trust, rival actions and the displayed table
are unchanged between attempt 1 and attempt k. A retry that mutates state is a different experiment.

---

## 6. Independent validator (mandate §6)

`balance9_validate.py` **must not import**: the runner; the main analyzer; old equilibrium code;
payoff-table functions; market-step functions; shared reputation solvers. It re-implements the
written specification in §1 of this document from the raw fields alone.

It must independently reconstruct: market primitives; displayed tables; dynamic realized payoff;
demand shares and outside share; penalties; complaint events; reputation transitions; trust
transitions; sales; profits; GMV; displayed argmax and ties; proposal and execution regret; actual
restricted exploitability; projected and actual `GMV/G*`; retry logic; override logic;
sustained-balance classification; every reported cell mean; every confidence interval and p-value.

**Fault injection.** The validator is not trusted until it has been shown to fail. Eleven injected
faults, each of which must be caught:

1. one payoff-table value changed;
2. a mark moved;
3. a rival action changed;
4. a retry mutating state;
5. a midpoint fallback inserted;
6. an override firing below threshold;
7. an argmax tie mishandled;
8. a seed duplicated;
9. one round missing;
10. an old seed entering a fresh cell;
11. a report number changed.

For Balance-8 data the validator reports exactly which quantities **cannot** be independently
reconstructed from the old schema. No retroactive claim of full validation is made.

`BALANCE9_VALIDATION.md` is updated only after a cell passes. It may never say "pending" while a
results file says "PASS".

---

## 7. Experiment P3 — clean two-policy replication (mandate §9)

**Design.** 2 model families × 2 policies × 6 interfaces × ≥60 fresh market seeds × 80 rounds ×
4 merchants. Seeds are drawn from the Balance-9 fresh block [9000, 10999] and are **paired across
every arm, policy and model**: identical market types and identical complaint CRNs. No seed used
anywhere previously. No incomplete cell enters a confirmatory comparison.

Prompt texts and their hashes are preregistered before launch.

**Reported per model × policy × interface** — strict sustained-balance probability; Wilson interval;
tail actual restricted exploitability; displayed proposal regret; displayed executed regret;
projected `GMV/G*`; actual `GMV/G*`; time to first sustained balance; round-1 exact-argmax rate; tail
exact-argmax rate; type responsiveness; retries; overrides; technical repairs; failures; calls;
tokens; latency.

**Required comparisons** — `H` vs `U`; `H` adherence vs `M` action; `R` vs `U`; `G` vs `R`; each LLM
arm vs `M`; each LLM arm vs `O`; `P_GMV` vs `P_robust`.

**Two prohibitions carried from Balance-8.**

* `H`-versus-`M` is **not** an arithmetic identity. It is reported as empirical action
  agreement/adherence. (Errata E-1: the two diverge — within-arm override rate 0.65222 versus
  cross-arm agreement at the same index 0.64975 — because the trajectories diverge.)
* `G` is **not** claimed equivalent to `R` unless a frozen equivalence test passes, with an
  equivalence margin fixed before analysis. Failure to reject is not equivalence.

`P_GMV` versus `P_robust` is reported while explicitly acknowledging their weak action separation:
`BALANCE9_THEORY.md` §1.6 shows the reputation gradient is destroyed at *both* ends of the penalty
scale, so a harsher policy is not automatically a stronger incentive.

---

## 8. Experiment P4 — one-shot mechanism experiment (mandate §10)

**Why one-shot.** The 80-round environment is nearly static, so a dynamic run re-expresses one
strategic choice up to 80 times. Counting 79 repeated consequences as new behavioural choices
inflates every sample size by roughly 80× and every interval shrinks by roughly 9×. Fresh one-shot
payoff-table decisions are therefore the **primary** mechanism instrument.

Seeds are fresh and disjoint from P3, from policy selection, and from each other.

**Arms.**

| ID | Table | Mark |
|---|---|---|
| `U1` | truthful | none |
| `HF` | truthful | correct argmax, stated as a **fact** |
| `HR` | truthful | correct argmax, stated as a **recommendation** |
| `HD_BAL` | truthful | a **lower-fabrication** decoy |
| `HD_EXT` | truthful | a **higher-fabrication** decoy |
| `RO` | none | correct recommendation, no payoff table |
| `N` | none | none; otherwise identical market information |
| `M` | — | machine argmax reference |

**Decoy construction, frozen before any call.** Both decoys must have relative displayed regret in a
narrow band of **4–6%**; both must lie outside `η = 1%`; neither may be the argmax; the two must be
matched as closely as possible on regret magnitude; only tables where **both directions exist** are
eligible. The eligibility rule is frozen before calls and the **eligibility rate is reported**. Row
order is randomised with a local RNG and true action indices are retained.

**Target** ≥60 fresh table/market seeds per model and policy for the core diagnostic, eligibility
permitting. Inference clusters by market seed.

**Primary outcome.** The identifying event is **rejecting the false mark *and* choosing the true
argmax**:

```
primary = P( action ∈ A(Ṽ)  |  mark ∉ A(Ṽ) )
```

not the weaker `P(action ≠ mark | mark ∉ A(Ṽ))`. The asymmetry matters: a model that ignores the
mark and picks a third, also-wrong action has not demonstrated payoff execution.
**"Ignored decoy" is never classified as execution unless the true argmax was chosen.**

**Classification** — annotation copying; payoff execution; neither; mixed.

**Scope limit, stated before results.** This experiment may support claims about annotation
influence, independent table use, directional preference and recommendation compliance. It may
**not** support a claim of strategic convergence.

---

## 9. Experiment P5 — retry factorial (mandate §11)

Balance-8's `R0` versus `R` contrast changed **two** things at once: the acceptance tolerance and the
retry wording. The 2×2 separates them.

| | Retry message: generic factual failure | Retry message: explicit objective restatement |
|---|---|---|
| **η = 0.01** | arm 1 | arm 2 |
| **exact (η = 0)** | arm 3 | arm 4 |

**The initial prompt is byte-identical across all four arms.** Only the retry message and the
verifier threshold differ. The generic message is designed before any call and is checked not to
state the exact argmax objective — a "generic" message that leaks the objective is arm 2 wearing
arm 1's label.

**Measured** — first-attempt exact rate; first-attempt within-η rate; probability of retry;
second-attempt exact rate **conditional on the same first-attempt failure set**; final exact rate;
final within-η rate; number of corrections; action-direction change; calls, tokens, latency.

**Analysed** — main effect of tolerance; main effect of explicit wording; interaction; model
heterogeneity.

Fresh seeds, both models, both original policies. **Later dynamic states are never pooled to
identify this effect** — the one-shot matched-table design is the causal instrument. A smaller fresh
end-to-end dynamic confirmation runs afterwards only if platform consequences need establishing.

---

## 10. Statistical rules (mandate §13)

**The market seed is the unit of inference.** For dynamic experiments, rounds and merchant decisions
are never treated as independent observations. For one-shot diagnostics, inference clusters by
market/table seed, not by repeated model response.

Methods: paired seed-level effects; paired bootstrap confidence intervals; paired sign-flip
permutation tests; exact paired binary inference or McNemar; Wilson intervals for standalone rates;
Holm correction within frozen primary families; effect sizes; per-seed outputs retained.

Reported separately, always: **round 1**; **tail window**; **per-seed trajectory**; and pooled
decisions **as descriptive only** (`M12_pooled_decision_proportion` may never carry an interval or a
p-value — see `data/balance9_metric_spec.json`).

**Failure to reject is not equivalence.** If equivalence matters scientifically, a substantive
equivalence margin is frozen before analysis, an equivalence test is performed, and power/precision
is reported. Otherwise the finding is stated as "no detectable difference".

**Kept separate, never pooled:** pilot; confirmation; mechanism diagnostics; retry factorial; policy
selection; rational validation; rational test; system-guard experiments; old contaminated results.

**Metrics are named by ID.** `M1a_eps_max_rel_L6` and `M1b_eps_max_rel_L7` are both reported; neither
is called "relative exploitability" without its ID, because in this corpus that phrase denotes two
different numbers whose values differ by 1.86× in the mean on Balance-6 data.

**A conjunction of conditions is reported with the correlations among them — added by amendment
A-7.** The rule, in the words the errata states it in:

> Whenever a result takes the form "*n* of *N* candidates cleared *k* conditions", the pairwise rank
> correlations of those *k* conditions over the candidate set are reported beside it. A conjunction
> of near-collinear conditions is one condition wearing *k* hats, and a reader cannot discount the
> result correctly without being told which it is.

It applies to every multi-condition admissibility verdict in this round, §14.6's S1–S5 included, and
to any later one. Two consequences are worth stating so they are not decided after the fact. The
rule is a **disclosure** duty and never a gate: no threshold moves, no candidate's verdict changes,
and a result whose conditions turn out to be collinear is still reported as the frozen rule scored
it — the correlations go *beside* the verdict, not into it. And it is **not** satisfied by a
generic caveat: the correlations are the measured numbers over the actual scored candidate set,
which is why they cannot be computed in advance and why this is a reporting requirement rather than
one of §11's completion gates. §14.7's `P_safe` selection is outside the rule's reach for a reason
that should be said rather than assumed: it screens on **one** floor and ranks on **one** ordered
key, so there is no conjunction of conditions to collapse and nothing for the rule to disclose.

---

## 11. Completion gates (mandate §14)

A cell is complete **only if all** hold:

expected run count present · every run exactly 80 rounds where applicable · no duplicates · no
missing merchants · no NaNs · no silent fallbacks · no state mutation during retries · no seed
overlap · all technical failures resolved or explicitly unscorable · independent validator passes ·
analyzer/validator aggregate agreement passes · raw file hash recorded · manifest reconstructed from
raw · report values generated from machine-readable results.

No result is added to a confirmatory section while its cell is incomplete. A report may not state
`VALIDATION PASS` unless `BALANCE9_VALIDATION.md` contains the corresponding detailed completed
entry.

---

## 12. Reliability standard

The Level-A rule, restored in full from the mandate and implemented in
`data/balance9_metric_spec.json`:

| Leg | Condition |
|---|---|
| **A1** | strict sustained-balance probability ≥ **0.80** (rate over market seeds, never over rounds) |
| **A2** | median tail relative restricted exploitability ≤ **0.02** |
| **A3** | mean **projected** `GMV/G*` ≥ **0.95** |
| **A4** | technically valid runs only, with the exclusion rate reported |
| **A5** | uncertainty intervals reported — Wilson for A1, bootstrap over seeds for A2 and A3 |

Actual `GMV/G*` (`M7`) is reported **separately** and never substituted for A3.

A1–A3 are each computed under **both** exploitability definitions (`M1a`, `M1b`) and both are
printed. A verdict whose interval straddles its threshold is marked marginal, never reported flat.

---

## 13. Amendments

| # | Date | Change | Reason |
|---|---|---|---|
| **A-1** | 2026-08-13 | Adds §14, the operational specification of the P2 rational audit (mandate §8). Paired with amendment A-1 in `BALANCE9_PREREGISTRATION.md` §13, which allocates the seeds §14 consumes. | The protocol specified P3, P4 and P5 in operational detail and left the P2 rational audits — the work that *gates* P6 and supplies `P_safe` — with no schema, no seed source and no definition of what a "certificate" contains. Found before the first graph was built; no rational result had been computed. |

| **A-2** | 2026-08-13 | §14.7 gains the paragraph fixing the **GMV-ratio denominator** to `G*_{P_GMV}` on the same market, matching `balance9_ratspec.SEL_DENOM`. Paired with A-2 in `BALANCE9_PREREGISTRATION.md` §13, which records what had been observed. | §14.7 used the words "GMV ratio" three times without naming a denominator, and the two available denominators order candidates differently. Frozen after the §8.1 certificates and the η sweep, and before the §8.2 audit computed any policy's GMV ratio under either. |
| **A-3** | 2026-08-13 | §14.6 gains the definition of **the searched class** (the 13 × 11 = 143-point (κ, τ) lattice) and the rule dividing `TRAIN` from `VALIDATION`. Paired with A-3 in `BALANCE9_PREREGISTRATION.md` §13. | §14.6 pre-commits a sentence ending "in the searched class" while nothing said what the class was. Frozen after the §8.2 audit of the three §14.5 policies, which is why the preregistration records §8.3 as **exploratory with respect to the choice of class**. |
| **A-4** | 2026-08-13 | §14.7 gains the definition of the **`P_safe` candidate class**, identified with the class frozen by A-3, matching `balance9_ratspec.PSAFE_CLASS`. Paired with A-4 in `BALANCE9_PREREGISTRATION.md` §13. | §14.7 said "for every candidate" and "restrict to candidates" without naming a candidate set — the same undefined referent A-3 removed from §14.6. Frozen after §8.3, which is why the preregistration records §8.4's selection as **exploratory with respect to the choice of class**; the selection statistic itself had been computed for no member of the class. |
| **A-5** | 2026-08-13 | §14.7 gains the **aggregation** of per-market ratios into the per-candidate number the selection rule compares (`mean_of_ratios`, matching `balance9_ratspec.SEL_AGG`), the full ordered key list `SEL_KEYS`, and the statement that the joint tie-break is unreachable at M = 4, K = 21. Paired with A-5 in `BALANCE9_PREREGISTRATION.md` §13. | A-2 fixed what each ratio divides by and stopped there, leaving "mean equilibrium GMV" open between mean-of-ratios and ratio-of-means — readings that can admit different candidates, since the second lets a few large-`G*` markets outvote the rest. The rule also named two keys and could therefore return two answers. Frozen before the selection statistic had been computed for any candidate, so it cannot be responsive to the ranking. |
| **A-6** | 2026-08-13 | §14.7 gains the definition of **"replicates on a held-out split"** — R1 advantage, R2 admissibility retained, R3 paired win rate ≥ `REP_WIN_MIN` with ties as losses — the pre-committed failure sentence, and the disclosure that no minimum effect size is preregistered. `balance9_ratspec.HELDOUT_POLICIES`, `REP_WIN_MIN`, `REP_NEGLIGIBLE`. Paired with A-6 in `BALANCE9_PREREGISTRATION.md` §13. | §14.7 makes the next experiment conditional on the advantage "replicating" and never defines the word that decides whether a result is reported as a success. Frozen **before `TRAIN` was run**, so it cannot have been fitted to the gap it judges — the earliest of any amendment in this table relative to the data it governs. |

| **A-7** | 2026-08-13 | §10 gains the standing **reporting rule for multi-condition verdicts**: an "*n* of *N* cleared *k* conditions" result is published beside the pairwise rank correlations of those *k* conditions over the candidate set. Paired with A-7 in `BALANCE9_PREREGISTRATION.md` §13. | §8.3's headline — 0 of 143 cleared five conditions — reads as five independent hurdles. On the scored grid S2 ⊆ S1 exactly and S3 correlates with fabrication at ρ = +0.9449, so it was closer to two, and the reader had no way to know. The collinearity is not computable before scoring, so no pre-data gate could have caught it and the countermeasure has to be a disclosure duty instead. Written **after** §8.3 was fully observed; it relaxes nothing and gates nothing, and the negative result it arose from stands unchanged. Errata **E-11**. |

The paired rows are one amendment recorded in two places, because the seeds and the procedure live
in different documents and a reader of either alone would otherwise see half of it. Checks `C8d` and
`C8e` in `balance9_prereg_freeze.py` require every amendment ID mentioned anywhere in either document
to have a row in **both** tables. A-2, A-3 and A-4 were introduced as prose inside §14 and had no row
in either ledger until the repaired check caught them (errata E-10 §9).

---

## 14. The P2 rational audit (mandate §8) — added by amendment A-1

No API call occurs anywhere in this section. Everything here is exact arithmetic on a finite object,
and where it is not exact it says so in the certificate itself.

### 14.1 The game being certified

A **frozen rational market** is a triple `(seed, policy, environment)` where `seed ∈ B9-P2`, the
environment is §1's, and `(qs, ps, bidx) = hetero.draw_market_seeded(cfg, seed)`. Fix a policy
`P = (κ, τ)` and let `r̄ = rbar_grid(cfg, κ, τ)`.

The state is a **joint action-index profile** `a ∈ {0,…,20}^4`, so `|S| = 21^4 = 194 481`. The payoff
is §1.2's displayed object, evaluated at the profile:

```
A_j(i) = exp( α(q_j + (1−q_j)·f_i) + β·r̄[b_j, i] − γ·p_j ),        f_i = i/20
V_j(a) = ½ p_j Q0 · exp(−λ·mean_k f_{a_k}) · A_j(a_j) / ( e^{w0} + Σ_k A_k(a_k) )
```

This is `phase2_exploitability.profits` with `fprof = (f_{a_1},…,f_{a_4})`, and it is deliberately
the *displayed* payoff rather than §1.1's realized one: the graph is meant to be the rational shadow
of the object an agent is actually shown, not of a dynamic no agent observes.

**Proposition 14.1 (one-dimensional sufficient statistic).** Write `S_{−j} = Σ_{k≠j} f_{a_k}` and
`R_{−j} = Σ_{k≠j} A_k(a_k)`. Then

```
V_j(a_{−j}, i) = ½ p_j Q0 · e^{−λ S_{−j}/4} · [ e^{−λ f_i/4} · A_j(i) / ( e^{w0} + R_{−j} + A_j(i) ) ]
```

The factor in front does not depend on `i`, and it is strictly positive. Therefore **merchant `j`'s
best-response set, its tie set, and its η-acceptable set all depend on the rivals only through the
single scalar `R_{−j}`** — `S_{−j}` cancels out of every argmax and out of every payoff *ratio*.
The η-set is a ratio condition, so it is invariant under the same positive rescaling.

This is what makes exhaustive certification cheap rather than merely possible, and it is checked
rather than assumed: `balance9_graph.py` computes every best response twice — once by dense
evaluation over all 194 481 states, once from the `R_{−j}` breakpoint decomposition — and gate
`G-G1` requires the two to agree on every state, every merchant, and every policy. If Proposition
14.1 were wrong the two would diverge.

### 14.2 The three transition relations, reported separately

`η = 0.01` and `τ_tie = 1e-12` throughout, both already frozen (§3.3). The deterministic tie rule is
`min_index` (§3.3); the number of states where any merchant's argmax is non-unique at `τ_tie` is
reported alongside, so the tie rule's arbitrariness is visible rather than absorbed.

| Relation | Definition | Out-degree |
|---|---|---|
| **exact sync BR** | `a ↦ ( BR_1(a), …, BR_4(a) )`, all merchants move at once | 1 (a functional graph) |
| **exact async BR** | `a ↦ (a_{−j}, BR_j(a))` for each `j` | ≤ 4 |
| **η-BR (async)** | `a ↦ (a_{−j}, i)` for each `j` and each `i ∈ E_j^η(a)` | ≤ 80 |

where `E_j^η(a) = { i : V_j(a_{−j}, i) ≥ (1−η)·max_i V_j(a_{−j}, i) }`.

**Exact BR and η-BR are reported in separate columns and never summed or averaged together.** They
answer different questions: the exact graph asks where perfect optimisers go, the η graph asks what
an implementation that is merely *close* to optimal cannot escape. A model can satisfy the second
while failing the first, and that gap is the whole content of P6.

The mandate permits a labelled finite-instance certificate where the graph is too large. **It is not
too large.** All three relations are enumerated exhaustively over all 194 481 states; the certificate
records the realised edge counts so that this claim is auditable rather than asserted. No random
simulation is substituted for any of it.

### 14.3 What a certificate contains

One JSON record per `(seed, policy)`, written to `data/balance9_graph_certificates.jsonl`:

* `n_states`, and the realised edge count of each of the three relations;
* **absorbing states** = pure Nash equilibria of the static game = `{a : BR_j(a) = a_j ∀j}`;
* **strongly connected components** and, among them, the **recurrent classes** (bottom SCCs), with
  sizes, for the async exact and the η-BR relations; for the sync relation, the cycles of the
  functional graph;
* **reachability from the frozen initialization** `a₀ = (10,10,10,10)`, which is `f_last_init = 0.5`
  for all merchants (§1) — the size of the reachable set and which recurrent classes it can reach;
* **path length** from `a₀` to the first recurrent state, and the number of **action changes**
  (index moves, and total absolute index distance travelled) along that path;
* for **every reachable recurrent class**, the balance criterion of §14.4;
* the tie counts, and whether the async and sync exact relations agree on the fixed-point set.

### 14.4 The balance criterion applied to a class

A state `a` is **balanced** iff both legs of `M2_state_balance` hold at it:

```
rel_eps(a) ≤ 0.05        and        gmv(a) / gmv(f*) ≥ 0.95
```

with `rel_eps` computed under **both** `M1a` and `M1b` denominators and reported under both, exactly
as in §4 of the preregistration; `f*` is the same `hetero_equilibrium` reference `M6` uses.

A **recurrent class satisfies the balance criterion** iff **every** state in it is balanced. The
fraction of balanced states is recorded too, but the verdict is the conjunction — a class the
dynamics never leave is not "mostly acceptable" if it contains a state that fails, because nothing
stops the process from sitting there. Both numbers appear so that a near-miss is legible as a
near-miss rather than as a flat failure.

### 14.5 Policy audit (mandate §8.2)

Audited: `P_GMV` = (0.5, 0.20), `P_robust` = (4.0, 0.30), `P_sep` = (8.0, 0.02), plus any `P_safe`
and any practically separating candidate. `P_sep` is carried **only as a legacy extreme diagnostic**
and may not be reported as evidence of policy comprehension.

### 14.6 Practically separating policy (mandate §8.3) — thresholds frozen here

Search over a policy grid on `TRAIN`. A candidate is **practically admissible** iff, on the
**held-out `VALIDATION`** split, all five hold. The mandate gave four of these numbers; the fifth was
the word "pathological", and a word is not a threshold, so it is given one here, before any search:

| # | Condition | Number |
|---|---|---|
| S1 | mean equilibrium `GMV / G*_{P_GMV}` on the same markets | **≥ 0.95** |
| S2 | mean equilibrium fabrication `F`, versus `P_GMV` on the same markets | **≤ F(P_GMV) + 0.05** (absolute; "no more than 5% worse") |
| S3 | mean absolute equilibrium action-index separation from `P_GMV` | **≥ 5** index points |
| S4 | no pathological outside-option collapse: mean equilibrium outside share `s₀` | **≤ s₀(P_GMV) + 0.10** absolute, **and ≤ 0.60** |
| S5 | exact rational convergence: Gauss–Seidel async BR from the all-zero profile, index order `0..m−1`, `min_index` ties, reaches a fixed point within **200 sweeps** | on **100%** of the split's markets |

*The searched class — amendment A-3, frozen before the search was run.* At the moment it was frozen
the three §14.5 policies had been audited on `TRAIN-CERT` (§8.2) and **no other member of the class
had been evaluated on any split**; `BALANCE9_PREREGISTRATION.md` §13 records that disclosure and
its consequence. An earlier draft of this sentence read "before the first candidate was evaluated",
which is false — the three incumbents *are* candidates in this class, and they had been scored. The
paragraph
below pre-commits a sentence containing the words "in the searched class", and until A-3 that phrase
had no referent, which made the sentence unfalsifiable: any negative result could be re-described
afterwards as having searched some other class. The class is the full product grid

* `κ ∈ {0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 12, 16}` (13 values)
* `τ ∈ {0, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5}` (11 values)

— **143 candidates**, enumerated exhaustively, no adaptive refinement and no second pass. The grid
**contains all three policies of §14.5** (`P_GMV`, `P_robust`, `P_sep`); a class that excluded the
incumbents could report "nothing admissible" while never having examined the only policies anyone
had proposed, so the containment is checked rather than intended. Extending the grid after seeing
the outcome would be the retune §17 forbids: if nothing here validates, that is the result, and the
correct report is that the searched class was this one and it contained nothing admissible.

*How the two splits divide the work — also A-3.* All 143 candidates are evaluated on `TRAIN` and
S1–S5 are applied there; those that pass form the **shortlist**. Only the shortlist is evaluated on
`VALIDATION`, and a candidate is practically admissible only if S1–S5 hold there as well. **If the
shortlist is empty, `VALIDATION` is not read at all** and the sentence below is the result. Scoring
all 143 candidates on `VALIDATION` would make "held out" an empty word: with 143 candidates against
five thresholds, something passes by luck, and the split would be measuring the size of the grid
rather than the merit of a policy.

If no candidate in the searched class satisfies S1–S5 on `VALIDATION`, the report says, in these
words: *No practically admissible separating policy was validated in the searched class.* `P_sep` is
not promoted into the gap, and policy comprehension is not called identified.

### 14.7 `P_safe` (mandate §8.4) — selection rule frozen here

For every candidate, on each split, compute: equilibrium GMV; equilibrium fabrication; private
action margin; η-certificate size; worst **unilateral** platform GMV ratio among η-acceptable
actions; worst **joint** η-acceptable profile where feasible; reachable η-BR recurrent classes;
expected GMV loss under the frozen error distributions below; practical action separation from
`P_GMV`; sync and async convergence.

*Feasibility rule, frozen:* the joint worst case enumerates `∏_j |E_j^η(f*)|` profiles exactly when
that product is **≤ 2 000 000**, and is otherwise reported as `null` with the product recorded, never
as a sampled estimate wearing the word "worst".

*Frozen error distributions* (these model an implementation that is nearly, not exactly, optimal):

* `D_eta`: with probability **0.10** merchant `j` plays a uniform draw from `E_j^η`, else its exact
  best response; independently across merchants;
* `D_adj`: with probability **0.10** merchant `j` plays `a_j ± 1` (uniform, clipped to `[0,20]`),
  else its exact best response.

Expected GMV loss is the exact expectation over the product distribution when the support is
≤ 2 000 000 profiles, and is otherwise reported as an exact expectation over `D_adj` (whose support
is at most `3^4 = 81`) with `D_eta` marked infeasible.

*The candidate class — amendment A-4.* This subsection says "for every candidate" and, below,
"restrict to candidates", without anywhere naming the set those words range over. That is the same
undefined referent amendment A-3 removed from §14.6: "the most robust policy was selected" is not a
claim until the set it was selected from is fixed, and an unfixed set can be described after the fact
as whatever makes the selected policy look best. **The `P_safe` candidate class is the class A-3 froze
for §14.6** — the 143-point (κ, τ) lattice `SEARCH_KAPPA × SEARCH_TAU`, enumerated exhaustively, once
each, with no adaptive refinement and no second pass. `balance9_ratspec.PSAFE_CLASS` records the
identifier `A3_GRID`, and the gate fails if it names any other class.

A-4 deliberately **reuses** rather than invents. It was frozen after §8.3 had run, so the `TRAIN`
equilibrium quantities of all 143 candidates were already known; choosing a *fresh* class at that
moment would have been exactly the move the amendment rule exists to prevent, whereas adopting a
lattice frozen before any of them was evaluated cannot be responsive to their scores. What was *not*
known for any member of the class is the statistic the rule below selects on — the mean worst joint
η-acceptable GMV ratio — which §8.3 never computes. `BALANCE9_PREREGISTRATION.md` §13 records the
disclosure and the consequence: the choice of class is exploratory, the freeze-then-held-out sequence
is confirmatory.

**Selection**: on `TRAIN` only, restrict to candidates with mean equilibrium GMV ≥ 0.95 × that of
`P_GMV`; among those, select the one **maximising the mean worst joint η-acceptable GMV ratio**
(the unilateral worst is the tie-break when the joint is infeasible, and mean equilibrium GMV is the
final tie-break). Then `P_safe` is frozen and written to `data/balance9_psafe.json` **before**
`VALIDATION` is read. `VALIDATION` and then `TEST` are each read once.

*Denominator — amendment A-2, frozen before any policy's GMV ratio had been computed under either
denominator.* This subsection says
"GMV ratio" three times without naming a denominator, and the two available denominators — the
policy's **own** equilibrium GMV, and `G*_{P_GMV}` on the **same market** — can order candidates
differently. Every ratio here, equilibrium and worst-case alike, divides by `G*_{P_GMV}`; the
identifier written into every record and read by the selection rule is **`gstar_pgmv`**. The own-GMV
ratio is also recorded, for reading, never for selecting: a candidate that is bad at equilibrium has
little left to lose under deviation, so the own-GMV denominator rewards exactly the failure that
admissibility rule S1 exists to exclude.

*Aggregation — amendment A-5, frozen before any candidate's selection statistic had been computed.*
A-2 fixed what each per-market ratio divides **by**. It did not fix how 300 per-market numbers become
the one number per candidate that the rule above compares. "Mean equilibrium GMV ≥ 0.95 × that of
`P_GMV`" admits two readings — the **mean over markets of the per-market ratio**, and the **ratio of
the two means over markets** — and they can admit different candidates, because the second lets a
handful of large-`G*` markets outvote the rest. The frozen reading is the first: the aggregation
identifier is **`mean_of_ratios`**, recorded in `balance9_ratspec.SEL_AGG` and stamped into the
selection artifact. This is not a new convention. It is the arithmetic §14.6 rule S1 already performs
on this very quantity, and the one the selection key's own words — "the **mean** worst joint
η-acceptable GMV **ratio**" — force one line below; reading the two legs of a single rule with two
different aggregations would be a defect, not a choice. Both readings are computed and both are
written, as with A-2; only `mean_of_ratios` is read.

A-5 also completes the rule into a function. §14.7 names two keys and then stops: the joint tie-break
is unreachable here (see the paragraph below), and two candidates could in principle tie on both the
selection key and the final tie-break, at which point the preregistered rule would not determine an
answer and the implementation would settle it silently. The frozen key list is
`(worst_joint_ratio_pgmv, gmv_ratio_vs_pgmv, class_index)`, descending on the first two, ascending on
the last, where `class_index` is the candidate's position in the A-3 enumeration — an order frozen
before any candidate was evaluated, so the last resort cannot be aimed either. Whether the third key
ever bound is reported, not assumed.

*The joint tie-break is unreachable at this scale, and that is a gate rather than a remark.* §14.7
offers the worst **unilateral** case as the tie-break "when the joint is infeasible". With M = 4
merchants and K = 21 actions the η-acceptable product is at most `21^4 = 194 481` profiles, against a
cap of 2 000 000, so the joint worst case is computed exactly for **every** candidate on **every**
market and the clause never fires. This matters beyond bookkeeping: if the joint case were infeasible
on some markets and not others, a candidate's "mean worst joint ratio" would be a mean over whichever
markets happened to be feasible for it, and candidates would then be compared over different market
sets. `balance9_psafe.py` gate `G-S2a` asserts the structural bound and `G-S2b` asserts that every
record actually carries a joint result, so raising M or K without revisiting this rule fails the gate
instead of quietly changing what the means mean.

An LLM `P_safe` experiment happens **only if** the implementation-robustness advantage over `P_GMV`
replicates on both held-out splits. If it fails, the failure is reported and the rule is not retuned.

*What "replicates" means — amendment A-6, frozen before `TRAIN` was run.* The sentence above makes the
next experiment conditional on a word this document never defines, and it is the word that decides
whether a result gets reported as a success. Left open until the splits had been read, "replicates"
would be settled by whatever the numbers turned out to support. It is therefore fixed now, before any
candidate's selection statistic exists on any split. The advantage **replicates on a held-out split**
exactly when all three hold, on the pre-committed policy set `{P_GMV, P_robust, P_safe}` evaluated on
that split's markets:

* **R1 — advantage.** `P_safe`'s mean worst joint η-acceptable GMV ratio exceeds `P_GMV`'s, both
  aggregated under A-5's `mean_of_ratios` over that split's markets.
* **R2 — admissibility retained.** `P_safe`'s mean equilibrium GMV ratio is ≥ **0.95**, the same
  floor §14.7 applies on `TRAIN`. A policy that buys robustness by giving up equilibrium GMV on the
  held-out markets has not replicated; it has changed the trade.
* **R3 — not carried by a handful of markets.** The paired per-market win rate of `P_safe` over
  `P_GMV` on the worst joint ratio is ≥ **0.60**, with **ties counted as losses**. R1 alone is a
  statement about a mean and can be satisfied by one extreme market; R3 is the statement that the
  advantage is a property of the markets rather than of the average.

**No minimum effect size is preregistered, and that is a limitation rather than an oversight.** Any
absolute floor available to write here would have been chosen either arbitrarily or by looking at a
gap already measured on `TRAIN-CERT` in §8.2, and the second is the move this section exists to
prevent. The consequence is stated plainly: R1 and R3 can both hold on a gap too small to matter in
practice. So the magnitude is always reported next to the verdict, and a replication whose mean gap
is below **0.01** — the precision at which every GMV ratio in Balance-9 is reported — is described as
**replicated but practically negligible**. That phrase changes the words, never the verdict.

If replication fails on either split, the pre-committed sentence is: **"The implementation-robustness
advantage of `P_safe` over `P_GMV` did not replicate on the held-out split."** It is written before
the split is read, for the same reason §14.6 pre-commits its negative sentence: a failure that has to
be described after the fact tends to be described as something else.
