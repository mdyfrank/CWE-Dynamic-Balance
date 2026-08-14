# Hidden-Action Experiment — Run Package

Platform-mediated implementation of a hidden-action equilibrium under signal-based reputation and
penalty design.

The platform does **not** observe merchants' true fabrication behaviour and **cannot** display
counterfactual payoffs across fabrication rates. It can publish a policy and return feedback built
only from what it observes. This package tests whether that is enough to bring LLM merchants to a
stable hidden-action equilibrium, and measures what fraction of attainable GMV is reached.

---

## 1. Environment

Set these before running. Nothing here is stored in the repository.

| Variable | Required | Meaning |
|---|---|---|
| `HA_ENDPOINT` | yes | Base URL of your inference endpoint |
| `HA_API_KEY` | yes | Key or token for that endpoint |
| `HA_DEPLOYMENT_GEMMA` | yes | Deployment/model name serving **Gemma 3 27B** |
| `HA_DEPLOYMENT_LLAMA` | yes | Deployment/model name serving **Llama 3.3 70B** |
| `HA_ROUTE` | no | `azure-openai`, `azure-ai-model-inference`, or `openai-compatible`. Inferred from the endpoint if unset |
| `HA_API_VERSION` | no | Azure API version, when the route needs one |
| `HA_ENDPOINT_GEMMA`, `HA_ENDPOINT_LLAMA` | no | Per-model endpoints, if the two models live in different places |
| `HA_TIMEOUT_S` | no | Per-request timeout, default 180 |
| `HA_ALLOW_MODEL_GEMMA`, `HA_ALLOW_MODEL_LLAMA` | no | Accept a served model name that differs from the expected one. **Use only deliberately** |

```bash
export HA_ENDPOINT="https://<your-resource>.openai.azure.com"
export HA_API_KEY="<your-key>"
export HA_DEPLOYMENT_GEMMA="<your-gemma-3-27b-deployment>"
export HA_DEPLOYMENT_LLAMA="<your-llama-3.3-70b-deployment>"
```

The transport verifies on the first call of each model that the served model name matches the one
expected, and **aborts before writing any result for that model** if it does not. There is no silent
substitution: a run that says Gemma 3 27B was produced by something that identified itself as Gemma
3 27B. If your provider reports a name that does not contain the expected tokens, set the
corresponding `HA_ALLOW_MODEL_*` variable explicitly — that choice is recorded in the run
provenance.

---

## 2. Run

```bash
# 0. everything checkable without an API key -- run this first, it is free
python hidden_action/offline_tests/run_offline_tests.py

# 1. see the matrix and the cost, write nothing
python hidden_action/code/to_run.py --mode main --plan

# 2. rehearse the whole pipeline on a mock backend -- no key, no network
python hidden_action/code/to_run.py --mode smoke

# 3. the real run
python hidden_action/code/to_run.py --mode main
```

**Resume**: just run step 3 again. Resume is by cell, and a completed cell is never re-called — the
skip happens before the transport is touched, so a resumed run costs nothing for work already done.
Interrupt with Ctrl-C at any time; at most the one cell in flight is lost.

Useful narrowing flags — they can only ever select a **subset** of the frozen manifest, never add to
it:

```bash
--models gemma            --arms A1_policy A3_assist
--policies P_GMV          --seeds 70000 70001
--limit 20                --stop-on-fail
--tier B_reduced          # the preregistered reduced tier; see section 5
```

---

## 3. Analyse and validate

```bash
python hidden_action/code/ha_analyze.py
python hidden_action/code/ha_validate.py
```

`ha_analyze.py` turns raw cells into the preregistered metrics. `ha_validate.py` re-derives them
independently — it does not import the analyser — and additionally rebuilds every complaint, refund
and audit count in every cell **from the market seed alone**, requiring an exact match. It exits
non-zero if any check fails, so it can gate a delivery.

---

## 4. What is where

```
hidden_action/
  code/
    ha_model.py          the market, signals, policy, stationary reputation, exact enumeration
    ha_infoclass.py      the information firewall: three data classes, enforced by construction
    ha_prompts.py        the frozen prompt surface for the four arms, plus 11 leakage gates
    ha_transport.py      multi-backend HTTP, retries, redaction, model-identity check, mock backend
    ha_runner.py         one cell = one (model, arm, policy, seed) played to the horizon
    to_run.py            THE entry point
    ha_analyze.py        metrics
    ha_validate.py       independent validation
    ha_benchmarks.py     G^FB, policy-class G^SB_P, per-seed benchmarks
    ha_theory_check.py   numerical checks of the propositions
    ha_signal_analysis.py signal informativeness
  manifests/
    ha_manifest_HA-M1.json    the frozen design; its sha256 is written into every run's provenance
  theory/
    HIDDEN_ACTION_THEORY.md   the model, the equilibrium concept, the propositions and proofs
    ORACLE_DEPENDENCY_AUDIT.md which earlier results relied on capabilities a real platform lacks
  offline_tests/
    run_offline_tests.py      14 tests, no key, no network, no cost
  results/
    raw/                 one gzipped JSON per cell
    attempts/            one JSON line per transport attempt
    progress/            progress.jsonl, state.json, errors.jsonl, provenance_*.json
    solver/              benchmarks, theory checks, signal analysis
    summaries/           analyser output
    validation/          validator and offline-test output
```

---

## 5. The design, in one screen

**Four arms.** All four share a byte-identical rules block; they differ only in what is appended.

| Arm | The merchant additionally sees |
|---|---|
| `A0_oracle` | The exact stationary payoff across all 21 fabrication rates. **Not implementable by a real platform.** An upper-bound control, kept to bound what the information is worth |
| `A1_policy` | Nothing beyond the published policy and its own state |
| `A2_history` | Its own observable history window |
| `A3_assist` | Platform assistance built **only** from platform-observable fields: policy explanation, the arithmetic of its own last penalty, threshold warnings, a compliance hint, and one economic retry |

**Three information classes**, enforced in code rather than by convention. A field reaches a prompt
only through `MerchantView.put`, which raises on an unregistered field, on an evaluator-only field,
and on another merchant's private field.

- `merchant_private` — own quality `q_i`, own baseline complaint propensity `b_i`, own history
- `platform_observable` — sampled transactions, complaints, refunds, audit flags, reputation,
  penalty, published policy parameters, category traffic index
- `evaluator_only` — **true fabrication of anyone**, rivals' types, the payoff table, its argmax,
  regret, the latent probabilities `θ` and `ψ`, every benchmark

**Two channels are visible by construction**, and are declared rather than discovered. The published
traffic index `Q = Q₀·exp(−λ·f̄)` is invertible in the market-average action — it is retained because
without it the four merchants face independent decision problems and the word *equilibrium* would be
empty. And each merchant sees its own `q_i` and `b_i`, which is precisely *why* the platform cannot
rebuild the payoff table, and so is the substance of the hidden-action assumption rather than a leak.

**Scale.** 2 models × 4 arms × 2 policies × 60 market seeds = **960 cells**, 80 rounds, 4 merchants
= **307,200 decisions**; with arm-A3 economic retries the API call count lands between **307,200 and
382,080**. A reduced tier (40 seeds × 40 rounds = 640 cells, 102,400 decisions) is preregistered in
the manifest. It is a **strict subset**: seeds are a prefix of the block, and the common random
numbers are drawn at a fixed horizon and sliced, so a 40-round trajectory is exactly the 80-round
trajectory truncated. Choosing it is permitted for budget reasons only, must be decided before the
analyser is run, and never becomes a different experiment.

---

## 6. Three retry counters, never added together

| Counter | Cause | Economic event? |
|---|---|---|
| `transport_attempts` | HTTP failure, rate limit, timeout | **No** |
| `schema_repairs` | Reply would not parse; asked again for JSON only, **no new economic information**. Budget 2 | **No** |
| `economic_retries` | Arm A3 offers a second decision opportunity | **Yes** |

Conflating them is how a rate limit becomes a finding about model behaviour. `V10` in the validator
checks they reconcile and that no economic retry appears outside arm A3.

**An action is never invented.** If a reply cannot be parsed after the repair budget, the cell fails
and is recorded as failed. Nothing defaults to honesty, to the previous round, or to the middle of
the grid — each of those defaults would itself be an economic claim.

---

## 7. What this design cannot show

Stated here so it cannot be quietly forgotten at writing-up time.

- A stable fabrication rate is **not** equilibrium evidence. Stability is necessary, not sufficient;
  the exploitability metric carries that burden. The analyser reports both counts side by side and
  the difference between them.
- Every second-best quantity is relative to the enumerated finite policy class. It is a
  **policy-class second best**, never a global one.
- A GMV ratio near or above 1 does **not** mean the theoretical maximum GMV was reached; the
  numerator is realized and the denominator is projected, and `ratio_definitions` in the analyser
  output says so for every ratio.
- Autonomous learning and strategic adaptation are not claimed without direct evidence from an
  environment with identifiable strategic interaction.

---

## 8. Cost and failure handling

Token and call counts accumulate in `results/progress/state.json` under `transport`, and every
attempt is a line in `results/attempts/attempts.jsonl` with its timing, status and token usage.

A cell is retried at most twice. A cell that still fails is recorded and excluded, and the exclusion
count is reported next to every affected metric. **If more than 5% of cells fail the run stops** —
the preregistered abort rule — and the cause is reported rather than the remainder analysed.

Configuration errors (`ConfigError`, `ModelIdentityError`) are never retried: they are not flaky, and
retrying would burn the budget and bury the cause.
