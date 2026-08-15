# Azure run: hidden-action experiment HA-M1

960 cells (2 models x 4 arms x 2 policies x 60 seeds x 80 rounds x 4 merchants) against
**Gemma 3 27B** and **Llama 3.3 70B**. Expect roughly **307,200 to 382,080 calls** — the spread is
arm A3's economic retries. See `manifests/ha_manifest_HA-M1.json` for the frozen design and
`PROTOCOL.md` for what the experiment is.

All commands below are run from the root of this repository. Everything that could be checked
without a key has been checked already and the evidence is committed under `results/validation/`,
so there is nothing to verify before you start.

## 1. Install

```bash
pip install -r requirements.txt
```

Python 3.11+. The only dependency is numpy; HTTP goes through the standard library.

## 2. Set your credentials

```bash
export HA_ENDPOINT="https://<your-resource>.openai.azure.com"
export HA_API_KEY="<your-key>"
export HA_DEPLOYMENT_GEMMA="<your-gemma-3-27b-deployment-name>"
export HA_DEPLOYMENT_LLAMA="<your-llama-3.3-70b-deployment-name>"
```

On Windows PowerShell use `$env:HA_ENDPOINT = "..."`.

If your endpoint host is neither `*.openai.azure.com` nor `*.services.ai.azure.com`, also set
`HA_ROUTE` to `azure-openai`, `azure-ai-model-inference` or `openai-compatible`. The runner will tell
you if it cannot decide; it will not guess. Set `HA_API_VERSION` if your route needs one, and
`HA_ENDPOINT_GEMMA` / `HA_ENDPOINT_LLAMA` if the two models live in different places.

The transport checks on the first call of each model that the served model name matches the one
expected, and aborts before writing any result for that model if it does not. If your provider
reports a name that does not contain the expected tokens, set `HA_ALLOW_MODEL_GEMMA` or
`HA_ALLOW_MODEL_LLAMA` explicitly — that choice is recorded in the run provenance.

Nothing here is stored in this repository.

## 3. Start

```bash
python code/to_run.py --mode main
```

To continue after any interruption, run the same command again. Completed cells are skipped before
the transport is touched, so a resumed run costs nothing for work already done. Ctrl-C is safe; at
most the one cell in flight is lost.

Add `--plan` to print the matrix and the call count and exit without writing anything, if you want
to see the size of the job before committing to it.

## 4. Progress

Live: `results/progress/progress.jsonl` and `results/progress/state.json`
(cells done / target, per-arm counts, transport and token totals).
Errors: `results/progress/errors.jsonl`.

## 5. Results

- raw: `results/raw/{model}__{arm}__{policy}__s{seed}.json.gz`, one per cell, 960 files
- per-call log: `results/attempts/attempts.jsonl`
- summaries: `results/summaries/`

## 6. Validate and package

```bash
python code/ha_analyze.py       # writes results/summaries/
python code/ha_validate.py      # writes results/validation/validation.json, non-zero exit on failure
```

`ha_validate.py` does not import the analyser; it re-derives every number independently and rebuilds
every complaint, refund and audit count from the market seed alone.

Send back everything in `results/raw/` and `results/summaries/`. Expect roughly 100 MB; send it by
file transfer or shared drive rather than committing it to git.

## 7. Input data

Not compressed; no extraction step is needed. Everything the run reads is already in `input_data/`,
`manifests/` and `code/`.

## 8. If the budget is short

```bash
python code/to_run.py --mode main --tier B_reduced    # 640 cells, 40 seeds x 40 rounds
```

`B_reduced` is preregistered and is a strict subset of the full tier, not a different experiment.
Choose it for budget or wall-clock reasons only, and say so before the analyser is run.

---

**One rule.** These are hidden-action results and they are a standalone study. Do not append them to,
or merge them with, the earlier displayed-payoff results on `azure-llm-handoff`. The manifest, the
seed block and the output paths are all different so that the two cannot collide.
