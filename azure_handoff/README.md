# Azure run: Balance-9 P3 llama matrix

480 runs (2 policies x 4 interfaces x 60 seeds x 80 rounds x 4 merchants) against Azure AI Foundry
`Llama-3.3-70B-Instruct`. Expect roughly **168,000 calls, 205M tokens, 5-6 hours** at the default
concurrency. See `manifests/cost_and_runtime_estimate.json` for the cost arithmetic.

## 1. Install

```bash
pip install -r azure_handoff/requirements.txt
```

Python 3.11+.

## 2. Set your Azure credentials

```bash
export AZURE_AI_ENDPOINT="https://<your-resource>.services.ai.azure.com/models"
export AZURE_AI_DEPLOYMENT="<your-Llama-3.3-70B-Instruct-deployment-name>"
export AZURE_AI_API_VERSION="2024-05-01-preview"
export AZURE_AI_API_KEY="<your-key>"        # omit to use DefaultAzureCredential instead
```

On Windows PowerShell use `$env:AZURE_AI_ENDPOINT = "..."`.

If your endpoint host is neither `*.openai.azure.com` nor `*.services.ai.azure.com`, also set
`AZURE_AI_ROUTE` to `azure-openai` or `azure-ai-model-inference`. The runner will tell you if it
cannot decide; it will not guess.

Optional, so the cost summary is filled in instead of left null:
`AZURE_PRICE_PER_1M_INPUT`, `AZURE_PRICE_PER_1M_OUTPUT`.

## 3. Start

```bash
python azure_handoff/to_run.py
```

To continue after any interruption:

```bash
python azure_handoff/to_run.py --resume
```

Completed seeds are skipped and never duplicated, so `--resume` is safe to repeat.

## 4. Progress

Live: `azure_handoff/results/logs/run.log` and `azure_handoff/results/logs/progress.json`
(`runs_done` / `runs_target`, per-cell counts, transport totals).

## 5. Results

- raw: `azure_handoff/results/raw/balance9_azure_p3_llama_raw.jsonl`
- per-call log: `azure_handoff/results/attempts/balance9_azure_attempts.jsonl`
- summaries: `azure_handoff/results/summaries/`

## 6. Validate and package

```bash
python azure_handoff/to_run.py --validate-only      # writes results/validation/validation.json
python azure_handoff/to_run.py --package-results    # writes results/final_delivery/
```

Send back everything in `azure_handoff/results/final_delivery/` — in particular
`balance9_azure_p3_llama_results.tar.gz` and `SHA256SUMS.json`. Expect roughly 80 MB; send it by file
transfer or shared drive rather than committing it to git.

## 7. Input data

Not compressed; no extraction step is needed. Everything the run reads is already in
`azure_handoff/input_data/` and `azure_handoff/code/`.

---

**One rule.** These Azure results are a standalone replication. Do not append them to, or merge them
with, the existing OpenRouter results.
