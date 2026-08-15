# Result Inventory and Status — Hidden-Action Experiment HA-M1

**As of 2026-08-15.** Every row is either `DONE` with a named artefact on disk, or `TO RUN` with the
exact cell count, seed block, manifest and target output path it will produce. Nothing is listed as
done because the code for it exists; a row is `DONE` only when the artefact is in `results/`.

Manifest under which all of this is frozen: `manifests/ha_manifest_HA-M1.json`,
sha256 `adf8a76ebc8e337093c5b5e4cc82a456f984d67098fa3a3913b2fc4d617eae6b`. That hash is written into
the provenance record of every run, so a silently edited manifest is visible in the results. It is a
hash of the file's bytes, and `.gitattributes` pins `eol=lf`, so a checkout on Windows reproduces it
too; without that pin a CRLF checkout would change this value and every `code_sha256_prefix` in the
provenance record without a single character of content having changed.

---

## 1. Non-LLM work — complete

These need no API key, no network and no budget. All of it has been run; the artefacts are in this
branch.

| # | Item | Status | Artefact | Headline |
|---|---|---|---|---|
| N1 | Exact equilibrium enumeration, 64-policy class × 60 seeds | **DONE** | `results/solver/ha_benchmarks.json` | 194,481 joint profiles per (seed, policy), exact; 3,840 pairs, 0 with no pure equilibrium, 0 with multiple |
| N2 | Extended two-channel policy class, 640 settings × 60 seeds | **DONE** | `results/solver/ha_benchmarks_extended.json.gz` (776 KB gz, 15.9 MB raw), `.log` | 38,400 pairs; 732.7 s |
| N3 | First best `G^FB` | **DONE** | both benchmark files, `aggregate.G_FB` | mean 0.8978 |
| N4 | No-penalty floor `G^NP` | **DONE** | same, `aggregate.G_NP` | mean 0.3491 = 38.9% of first best |
| N5 | Policy-class second best `G^SB_P`, per-seed tuned | **DONE** | same, `aggregate.G_SB_P_pessimistic` | 0.7035 (64-policy) → 0.7084 (640-policy) = 78.7% of first best |
| N6 | Committable second best `G^SB_uniform`, one policy for all seeds | **DONE** | same, `aggregate.G_SB_uniform` | 0.6326 (64) → 0.6366 (640) at κ=2, τ=0.3 |
| N7 | Gap decomposition | **DONE** | same | gap to first best 0.2612 = **8% instrument cost + 92% incentive gap** |
| N8 | Twelve theory propositions, numerically checked | **DONE** | `results/solver/ha_theory_checks.json` | **12/12 PASS**, 74.8 s |
| N9 | Signal informativeness, calibration, detector rates, monitoring continuum | **DONE** | `results/solver/ha_signal_analysis.json` | complaint-count posterior cuts variance in `f` by 93.8%; audit Fisher information is 1.1–14.1% of the complaint channel's (median 3.8%) across the 24 `(b, f)` cells |
| N10 | Oracle-dependency audit of the earlier design | **DONE** | `theory/ORACLE_DEPENDENCY_AUDIT.md` | three instruments identified as unimplementable, cited to frozen line numbers |
| N11 | Hidden-action theory with proofs | **DONE** | `theory/HIDDEN_ACTION_THEORY.md` | P1–P10, each with a falsifier and a traceability key |
| N12 | Offline test suite | **DONE** | `results/validation/offline_tests.json` | **14/14 PASS**, 143.9 s |
| N13 | Mock end-to-end rehearsal | **DONE** | regenerable by `--mode smoke` | 32/32 cells, resume makes 0 calls |
| N14 | Analyser and validator exercised on mock data | **DONE** | `results/validation/validation_smoke.json` | **12/12 PASS**; 512 signals rebuilt from seeds alone |

### The twelve propositions and what would falsify each

All twelve verdicts are `PASS` in `ha_theory_checks.json`; each record carries its own
`would_falsify` string, so a reader can check the test was capable of failing.

| Check | Statement, in one line |
|---|---|
| `P1_existence` | Every (seed, policy) pair has a pure-strategy Nash equilibrium |
| `P2_exact_ic` | The incentive-compatibility characterisation is exact on the finite grid |
| `P3a_deviation_identity` | The deviation identity holds term by term, including the `λ/m` traffic term |
| `P3b_sufficient_condition_upward` | The sufficient condition for no upward deviation, and its slack |
| `P4_robust_implementation` | Implementation survives the stated perturbation |
| `P5_overpunishment_reversal` | Past a threshold, more punishment lowers GMV |
| `P5b_large_kappa_limit` | The κ→∞ limit behaves as the proposition says |
| `P6_impossibility_uninformative_signal` | With `c_s = 0` no policy in the class improves on no penalty |
| `P7_refunds_are_a_garbling` | Refunds add zero information over complaints |
| `P8b_oracle_reduction` | The old displayed table is the best-response payoff; its fixed points are Nash — 120/120 |
| `P9_dead_zone` | Regions where the signal cannot separate actions |
| `P10_implementation_floor` | A floor below which no policy in the class can push behaviour |

`P6` is the impossibility result mandated for the completely uninformative case; `P8b` is what
connects the old displayed-game results to the new formulation.

---

## 2. LLM experiments — `TO RUN`

Nothing in this section has been run. No API call has been made under manifest HA-M1.

### Tier A_full — the preregistered primary tier

| Field | Value |
|---|---|
| Status | **TO RUN** |
| Cells | **960** = 2 models × 4 arms × 2 policies × 60 seeds |
| Seed block | **HA-M1 = 70000–70059**, verified unused by any earlier study in this repository line |
| Rounds × merchants | 80 × 4 |
| Decisions | **307,200** |
| API calls | **307,200 – 382,080** (the upper end is arm A3's economic retries) |
| Models | Gemma 3 27B, Llama 3.3 70B; temperature 0 |
| Manifest | `manifests/ha_manifest_HA-M1.json` @ `adf8a76e…` |
| Command | `python code/to_run.py --mode main` |
| Target raw output | `results/raw/*.json.gz`, one per cell, 960 files |
| Target progress | `results/progress/{progress.jsonl, state.json, errors.jsonl, provenance_*.json}` |
| Target attempts | `results/attempts/attempts.jsonl`, one line per transport attempt |
| Then | `ha_analyze.py` → `results/summaries/`, `ha_validate.py` → `results/validation/validation.json` |

### Tier B_reduced — the preregistered fallback

| Field | Value |
|---|---|
| Status | **TO RUN, and only if the budget requires it** |
| Cells | **640** = 2 × 4 × 2 × 40 seeds |
| Seed block | 70000–70039, a **prefix** of HA-M1 |
| Rounds | 40 |
| Decisions | **102,400** |
| Command | `python code/to_run.py --mode main --tier B_reduced` |

B_reduced is a **strict subset** of A_full, not a different experiment: the seeds are a prefix, and
the common random numbers are drawn at a fixed horizon of 120 rounds and sliced, so a 40-round
trajectory is exactly the 80-round trajectory truncated. The runner enforces this as invariant
`crn_horizon_prefix`. Choosing it is permitted **for budget or wall-clock reasons only**, must be
recorded before the analyser is run, and is never a result-dependent choice.

### The four arms

| Arm | Cells | What the merchant additionally sees | Role |
|---|---|---|---|
| `A0_oracle` | 240 | The exact stationary payoff across all 21 fabrication rates | **Upper-bound control only.** Not implementable by a real platform, never primary evidence |
| `A1_policy` | 240 | Nothing beyond the published policy and its own state | Baseline |
| `A2_history` | 240 | Its own observable history window | Treatment |
| `A3_assist` | 240 | Platform assistance built only from platform-observable fields | Treatment |

---

## 3. What the LLM run will and will not be able to establish

Written now, before any result exists, so it cannot be quietly relaxed later.

**It can establish**, if the data support it:

- whether the tail fabrication profile is an approximate equilibrium — the **exploitability** metric
  carries this, not tail stability;
- the fraction of `G^SB_P` and of `G^FB` that realized GMV reaches, under each policy and arm;
- whether observable-history feedback (A2) or signal-based assistance (A3) moves behaviour relative
  to policy alone (A1), paired by market seed;
- how much of the A0–A3 gap is attributable to information the platform cannot supply;
- whether the platform's own detector is well calibrated against the latent complaint probability,
  and its false-positive and false-negative rates against its **merchant-specific** stated tolerance.

**It cannot establish**, whatever the numbers say:

- equilibrium from a stable fabrication rate. The analyser reports the stable count and the
  exact-Nash count side by side, plus the count of cells that are stable but **not** Nash. On the
  32 mock cells that difference is already **32 stable, 0 exact Nash, 32 stable-but-not-Nash** — the
  size of the mistake that conflating the two would have made;
- anything about mechanisms outside the enumerated finite policy class. Every second-best number is
  a **policy-class** second best;
- that a GMV ratio at or above 1 means the theoretical maximum was reached — the numerator is
  realized and the denominator is projected, and `ratio_definitions` in the analyser output says so
  for every ratio;
- autonomous learning or strategic adaptation, without direct evidence from an environment with
  identifiable strategic interaction.

---

## 4. Status of the earlier handoff branch

| Field | Value |
|---|---|
| Branch | `azure-llm-handoff` @ `4e66dab7` |
| Status | **UNKNOWN** |

**Evidence for `UNKNOWN` rather than `INCOMPLETE`.** All six `results/` subdirectories on that branch
contain only a 0-byte `.gitkeep`, and there are no commits after 2026-08-14T05:07:08Z. But that
package's own README instructed the collaborator to return the ~80 MB result archive **out of band**
rather than by pushing it. So the absence of results in git cannot distinguish "never run" from "run
and returned by file transfer". Branch existence is not completion, and neither is branch emptiness
evidence of non-completion.

**Nothing on that branch was modified.** It was not rebased, deleted, force-pushed or written to. Its
source files are read-only inputs to `theory/ORACLE_DEPENDENCY_AUDIT.md`, which cites them by
unchanged line number. The present work is on an independent branch with a different manifest, a
different seed block, a different checkpoint namespace and a different output path, so the two runs
cannot collide even if they are executed on the same machine at the same time.

| Dimension | `azure-llm-handoff` | This branch |
|---|---|---|
| Manifest | balance-9 manifest | `ha_manifest_HA-M1.json` |
| Seed block | earlier blocks | 70000–70059 |
| Output path | `www_project/results/` | `results/` |
| Checkpoint | that package's state file | `results/progress/state.json` |

---

## 5. Traceability

Every reported conclusion resolves along one fixed chain, and each arrow is a file on disk:

```
results/raw/*.json.gz
  → code/ha_analyze.py    → results/summaries/ha_metrics_summary.json
  → code/ha_validate.py   → results/validation/validation.json
  → a numbered claim in REPORT.md
```

`ha_validate.py` deliberately does not import the analyser. Where it checks an analyser number it
recomputes that number independently. Its strongest check, `V6`, rebuilds every complaint, refund
and audit count in every cell **from the market seed alone** and requires an exact match — so the
claim that arms are paired on identical noise is a property of the data rather than an assertion in
a document. The validator is itself mutation-tested by offline test `validator_catches_corruption`,
which breaks one thing per cell and requires the matching check to turn red.

| Validator check | What it re-derives |
|---|---|
| V1 | Cell integrity and schema |
| V2 | Accounting identity |
| V3 | Reputation recursion |
| V4 | Signal bounds |
| V5 | Action grid membership |
| V6 | **Every signal rebuilt from the market seed alone** |
| V7 | Analyser metrics, recomputed independently |
| V8 | Benchmark alignment against the solver |
| V9 | Manifest conformance |
| V10 | The three retry counters reconcile; no economic retry outside A3 |
| V11 | Prompt hygiene |
| V12 | Claim traceability |

---

## 6. Reproducing the non-LLM tier from scratch

No key, no network, no cost. Wall-clock on a laptop in brackets.

```bash
python code/ha_benchmarks.py --seeds 70000-70059              # 64 policies  [ 75 s]
python code/ha_benchmarks.py --seeds 70000-70059 --extended   # 640 policies [733 s]
python code/ha_theory_check.py                                # 12 checks    [ 75 s]
python code/ha_signal_analysis.py                             #              [826 s]
python offline_tests/run_offline_tests.py                     # 14 tests     [145 s]
python code/to_run.py --mode smoke                            # 32 cells     [1.3 s]
```

The first four are ordered: `ha_theory_check.py` and `ha_signal_analysis.py` both read
`ha_benchmarks.json` and exit with a message if it is missing. Total, about 33 minutes.

Almost all of the offline suite's 145 s is deliberate retry backoff in `mock_smoke` (60 s) and
`retry_counters_separate` (75 s), which drive a mock backend that returns malformed responses on
purpose. The smoke run itself is 1.3 s because nothing there has to fail and be retried.

`ha_benchmarks_extended.json.gz` is committed compressed because it is 15.9 MB raw and nothing reads
it programmatically; `ha_benchmarks.json` is committed uncompressed because the analyser, the
validator, the theory checker and the signal analyser all open it by that exact path.

The accompanying `ha_benchmarks_extended.log` is the record of that run and is left exactly as it was
written, so its last line still names an absolute path containing `hidden_action\` and the
uncompressed `.json` filename — both from before the package moved to the branch root and the
artefact was gzipped. Rewriting a log to agree with a later reorganisation would make the provenance
worse, not better. It is the only stale path anywhere in the tree; everything the code opens is
resolved relative to the file that opens it.
