# Preregistration and Protocol — Hidden-Action Experiment HA-M1

**Frozen 2026-08-15, before any LLM API call.**
Machine-readable counterpart: `manifests/ha_manifest_HA-M1.json`, whose sha256 is written into the
provenance record of every run. Any edit to the manifest changes that hash and is therefore visible
in the results.

---

## 1. Question

The platform does not observe merchants' true fabrication behaviour, and it cannot construct the
counterfactual payoff table across fabrication rates, because building that table requires each
merchant's private honest quality. It can publish a policy and return feedback assembled only from
what it observes.

> Can a platform, using only observable signals, bring LLM merchants to a stable hidden-action
> equilibrium — and what fraction of attainable GMV does that reach?

## 2. Why the earlier design cannot answer it

Documented in full in `theory/ORACLE_DEPENDENCY_AUDIT.md`. In short, three earlier instruments each
require a capability no real platform has:

- the **correct-argmax annotation** requires knowing the true fabrication rate and true quality of
  every merchant;
- the **displayed-regret verifier** requires the displayed payoff table, which requires the same;
- the **terminal override** requires the platform to set the merchant's action directly, which is
  the negation of hidden action.

These are retained as **oracle upper-bound controls only** (arm `A0_oracle`), never as primary
evidence.

## 3. Model

Merchant `i` chooses a fabrication rate `f_i` on a 21-point grid `{0, 0.05, …, 1.00}`. The platform
never observes it. Honest quality `q_i` and baseline complaint propensity `b_i` are
**merchant-private** — the decisive modelling choice, and exactly what makes the displayed payoff
table impossible for the platform to construct.

Observable signals per merchant per round:

```
complaints   d ~ Binomial(N_obs = 40, θ),   θ = clip(b_i + c_s·f_i, 0, 1),   c_s = 0.60
refunds      ρ | d ~ Binomial(d, φ = 0.50)                    a thinning: a Blackwell garbling,
                                                              so it carries no extra information
audit flags  c ~ Binomial(N_audit = 5, ψ),  ψ = clip(ψ₀ + ψ₁·f_i, 0, 1),  ψ₀ = 0.02, ψ₁ = 0.25
```

Signal-contingent policy and reputation:

```
P  = κ·max(0, d/N_obs − τ) + κ_a·max(0, c/N_audit − τ_a)
r' = clip(r + η_r(1 − r) − P, 0, 1),   η_r = 0.20
```

Frozen market constants: `m = 4`, `α = 3.0`, `β = 1.5`, `γ = 1.0`, `Q₀ = 1.0`, `λ = 1.00`,
`ω = 0.50`, `w₀ = 1.0`, `margin_frac = 0.5`, reputation grid `R = 31`.

Two policies are run: `P_GMV = (κ 0.5, τ 0.20)` and `P_robust = (κ 4.0, τ 0.30)`.

## 4. Design

- **Unit of inference: the market seed.** Every arm, policy and model plays the identical market
  draw and the identical uniform noise.
- **Seed block HA-M1 = 70000–70059** (60 seeds), verified unused by any earlier study in this
  repository line. Market draws use base 1000; common random numbers use base 500000; the streams
  cannot alias.
- **Arms**: `A0_oracle` (upper-bound control), `A1_policy`, `A2_history`, `A3_assist`.
- **80 rounds**, 4 merchants, temperature 0, initial reputation 0.5.
- **960 cells** = 2 models × 4 arms × 2 policies × 60 seeds.

### Reduced tier, fixed in advance

`B_reduced` = 40 seeds (70000–70039) × 40 rounds = 640 cells. It is a **strict subset** of the full
tier: the seeds are a prefix of the block, and the common random numbers are drawn at a fixed
horizon and sliced, so a 40-round trajectory is exactly the 80-round trajectory truncated
(invariant `crn_horizon_prefix`). Selecting it is permitted **for budget or wall-clock reasons
only**, must be decided and recorded before the analyser is run, and is never a result-dependent
choice.

## 5. Information classes

Enforced by `ha_infoclass.py` through construction, not convention: a field reaches a prompt only
via `MerchantView.put`, which raises on an unregistered field, on an evaluator-only field, and on a
field owned by another merchant.

In the main experiment merchants **never** see: the counterfactual payoff table (except `A0_oracle`,
by design); expected payoff per candidate fabrication rate; any argmax or recommended rate; regret
computed from latent `f` or `q`; or any evaluator-only ground truth.

**Declared visible channels.** The published traffic index `Q = Q₀·exp(−λ·f̄)` is invertible in the
market-average action; it is retained because without it the four merchants face independent
decision problems and *equilibrium* would be an empty word, and the `λ/m` term of the deviation
identity already accounts for it. Each merchant sees its own `q_i` and `b_i`; that is precisely why
the platform cannot rebuild the payoff table.

**Positive control.** Arm `A0_oracle` is built to receive an evaluator-only object and **must fail**
the leakage invariance probe. A detector that never fires on the arm constructed to trip it has not
been shown to work. Offline test `oracle_positive_control` requires `A0 = False` and all other arms
`True`.

## 6. Primary metrics

Frozen. Full list in the manifest under `primary_metrics`.

1. **Hidden-action equilibrium regret / exploitability** — profit forgone by not best-responding to
   the rivals' realized tail profile, at the stationary payoff. *This is the equilibrium evidence.*
2. **Tail action stability** over the last 20 rounds, **lock time**, terminal fabrication
   distribution. *Necessary, not sufficient — never reported as equilibrium.*
3. **Realized merchant profit** and **realized platform GMV**.
4. **GMV ratios**: `G_LLM / G^SB_P`, `G^SB_P / G^FB`, `G_LLM / G^FB`, and `G_LLM / G^NASH` under the
   identical policy.
5. **Signal calibration**: observed complaint rate against latent `θ`; slope, intercept,
   correlation, bias, RMSE.
6. **False-positive punishment** and **false-negative detection** rates, against the policy's own
   stated tolerance `f ≤ (τ − b_i)/c_s` — merchant-specific, because `b_i` differs, and judging
   against a fixed threshold would charge the policy for punishing a merchant it never promised to
   spare.
7. **Oracle minus non-oracle arm gap.**

### GMV benchmarks, and what each is

| Symbol | Definition | Kind |
|---|---|---|
| `G^FB` | First best: the platform observes the hidden action and sets it directly | projected, stationary |
| `G^SB_P` | Best GMV attainable by **any policy in the enumerated finite class**, at the worst equilibrium under each policy | projected, stationary |
| `G^NASH(π)` | Worst pure-strategy Nash GMV under the **same policy the cell ran** | projected, stationary |
| `G^NP` | No-penalty baseline | projected, stationary |
| `G_LLM` | Mean per-round GMV over the last 20 realized rounds | **realized** |

`G^SB_P` is a **policy-class second best**. It is never called a global second best, and none of
these numbers is a theorem about all mechanisms. Exact enumeration of all `21⁴ = 194,481` joint
profiles per (seed, policy) is feasible and is used — measured at roughly 15 ms per pair by numpy
broadcasting, which is why the second-best search over 640 policies × 60 seeds is exact rather than
sampled.

**Realized and projected quantities are never averaged together.** Field names carry the prefix
`realized_` or `stationary_`, and every ratio states both halves in `ratio_definitions` in the
analyser output.

## 7. Comparison and stopping rules

- **Pairing** by market seed; the within-seed difference is the observation.
- **Intervals**: paired bootstrap over seeds, 10,000 resamples, seed 20260815, percentile method.
- **p-values**: sign-flip permutation, the correct null for a within-seed contrast.
- **Multiplicity**: Benjamini–Hochberg at `q = 0.05` across the preregistered contrasts.
- **Primary contrasts**: `A3−A1`, `A2−A1`, `A0−A3`, `A0−A1`, each within model and policy.
- **No interim peeking.** Operational health is read from the progress files, never from metric
  summaries.
- **Cell failure**: at most two retries; a still-failing cell is excluded and the exclusion count is
  reported next to every affected metric.
- **Abort**: if more than 5% of cells fail, the run stops and the cause is reported rather than the
  remainder analysed.
- **No threshold adjustment.** If the evidence is negative it is reported as negative.

## 8. What this design cannot show

- Equilibrium cannot be inferred from a stable fabrication rate. The analyser reports the stable
  count and the exact-Nash count side by side, plus the count of cells that are stable but **not**
  Nash — the size of the mistake that conflating them would have made.
- No claim about mechanisms outside the enumerated policy class.
- No claim of autonomous learning or strategic adaptation without direct evidence from an
  environment with identifiable strategic interaction.
- No reading of a ratio at or above 1 as "reached the theoretical maximum GMV".

## 9. Traceability

Every reported conclusion resolves along a fixed chain:

```
results/raw/*.json.gz  →  ha_analyze.py  →  results/summaries/ha_metrics_summary.json
                       →  ha_validate.py →  results/validation/validation.json  →  report claim
```

`ha_validate.py` does not import the analyser; where it checks an analyser number it recomputes it
independently. Its strongest check, `V6`, rebuilds every complaint, refund and audit count in every
cell **from the market seed alone** and requires an exact match — so the pairing across arms is a
property of the data rather than an assertion in this document. The validator is itself
mutation-tested by offline test `validator_catches_corruption`, which breaks one thing per cell and
requires the matching check to turn red.
