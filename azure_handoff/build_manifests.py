"""build_manifests.py -- run ONCE, at package build time, in the source repository.

Records the SHA-256 of every vendored file so `to_run.py` can prove on the collaborator's machine
that the code about to spend money is the code that produced the OpenRouter half of the corpus.

This is a build tool, not part of the run. The collaborator never needs it.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

HERE = Path(os.path.dirname(os.path.abspath(__file__)))
SRC = HERE.parent

VENDORED = [
    # frozen experiment code -- byte-identical copies, hashed here and re-checked at run time
    ("code/equilibrium.py", "equilibrium.py"),
    ("code/hetero.py", "hetero.py"),
    ("code/hetero_policy_audit.py", "hetero_policy_audit.py"),
    ("code/phase2_exploitability.py", "phase2_exploitability.py"),
    ("code/phase2_framing_controlled.py", "phase2_framing_controlled.py"),
    ("code/phase2_llm.py", "phase2_llm.py"),
    ("code/balance8_runner.py", "balance8_runner.py"),
    ("code/balance9_prompts.py", "balance9_prompts.py"),
    ("code/balance9_runner.py", "balance9_runner.py"),
    ("code/balance9_ratspec.py", "balance9_ratspec.py"),
    ("code/balance9_prereg_freeze.py", "balance9_prereg_freeze.py"),
    ("code/balance9_validate.py", "balance9_validate.py"),
    # frozen inputs
    ("input_data/catalog_types.json", "data/catalog_types.json"),
    ("input_data/balance9_prompt_hashes.json", "data/balance9_prompt_hashes.json"),
    ("input_data/BALANCE9_PREREGISTRATION.md", "BALANCE9_PREREGISTRATION.md"),
    ("input_data/BALANCE9_PROTOCOL.md", "BALANCE9_PROTOCOL.md"),
]
# written by this package, hashed for completeness but not compared against a source file
PACKAGE_OWN = ["to_run.py", "code/azure_transport.py", "input_data/prior_seeds.json",
               "offline_tests/run_offline_tests.py", "requirements.txt"]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    files, provenance, drift = {}, {}, []
    for rel, src_rel in VENDORED:
        p, s = HERE / rel, SRC / src_rel
        if not p.exists():
            raise SystemExit(f"missing vendored file: {rel}")
        h = sha256_file(p)
        files[rel] = h
        provenance[rel] = dict(source=src_rel, sha256=h, bytes=p.stat().st_size, frozen=True,
                               identical_to_source=(s.exists() and sha256_file(s) == h))
        if not provenance[rel]["identical_to_source"]:
            drift.append(rel)
    for rel in PACKAGE_OWN:
        p = HERE / rel
        if p.exists():
            files[rel] = sha256_file(p)
            provenance[rel] = dict(source="written for this package", sha256=files[rel],
                                   bytes=p.stat().st_size, frozen=False,
                                   identical_to_source=None)
    if drift:
        raise SystemExit(f"vendored copies differ from their sources: {drift}")

    (HERE / "manifests" / "vendored_sha256.json").write_text(json.dumps(dict(
        purpose="Byte identity of every file this package runs. to_run.py re-checks these before "
                "any Azure request; a mismatch aborts the run, because a changed prompt or constant "
                "turns a replication into a different experiment wearing the same name.",
        built=time.strftime("%Y-%m-%dT%H:%M:%S"),
        n_files=len(files), files=files, provenance=provenance), indent=1, sort_keys=True),
        encoding="utf-8")

    (HERE / "manifests" / "experiment_manifest.json").write_text(json.dumps(dict(
        experiment="B9-P3-AZURE",
        title="Fresh Azure replication of the Balance-9 P3 llama matrix",
        provider="Azure AI Foundry",
        model=dict(expected="Llama-3.3-70B-Instruct",
                   openrouter_counterpart="meta-llama/llama-3.3-70b-instruct",
                   substitution_forbidden=["quantised builds (AWQ/GPTQ/GGUF/int4/int8/fp8)",
                                           "community rebuilds", "any other parameter count",
                                           "any other Llama version"],
                   identity_checked_on="the first live response, before any raw record is written"),
        design=dict(policies=["P_GMV", "P_robust"], interfaces=["U", "H", "R", "G"],
                    seed_block="B9-P3", seeds=[9000, 9059], n_seeds=60, rounds=80, merchants=4,
                    runs=480,
                    launch_order=[[1, ["U", "H"]], [2, ["R"]], [3, ["G"]]],
                    launch_order_note="the LLM restriction of preregistration 5.1 wave order "
                                      "(M, O, U, H | R | G); M and O call no model and are not run"),
        constants=dict(NIDX=20, ETA=0.01, TAU_TIE=1e-12, OMEGA=0.5, LAM=1.0, ZETA=0.2,
                       MAX_ECON_RETRY=2, SCHEMA_REPAIR=2, TRANSPORT_RETRIES=6,
                       P_GMV=[0.5, 0.20], P_robust=[4.0, 0.30]),
        separation=dict(
            openrouter_results="untouched; this package neither reads them for writing nor merges "
                               "with them",
            azure_output="results/raw/balance9_azure_p3_llama_raw.jsonl",
            merge_policy="Azure runs must be reported as a standalone replication. Do NOT append "
                         "them to the incomplete OpenRouter llama cells."),
        join_keys=dict(raw="run_key + round + merchants[].j",
                       attempts="run_key + round + j (decision_key)",
                       run_id_map="results/logs/run_id_map.json"),
        environment=dict(built_on=platform.platform(), python=sys.version.split()[0]),
        built=time.strftime("%Y-%m-%dT%H:%M:%S"),
    ), indent=1), encoding="utf-8")

    print(f"vendored_sha256.json: {len(files)} files, {len(drift)} drifted")
    for rel in sorted(files):
        print(f"  {files[rel][:16]}  {rel}")


if __name__ == "__main__":
    main()
