"""balance9_prereg_freeze.py -- gate G-prereg.

BALANCE9_PREREGISTRATION.md section 14 promises that this script exists and that no confirmatory call
may be made until it passes. This is that script. It is the last thing that runs before Balance-9 is
allowed to spend an API call on a confirmatory arm.

A preregistration is worth exactly as much as the difference between what it says and what the code
does. So this file does not check that the document is well written; it checks that every number,
sentence and identifier in it is the same object the code will actually use:

  C1  the document hashes to a recorded digest (so a later edit is detectable);
  C2  the three manipulation sentences quoted in section 11 are byte-identical to the constants in
      balance9_prompts.py -- a preregistration that paraphrases its own stimulus has preregistered
      nothing;
  C3  the published prompt hashes in the document and in data/balance9_prompt_hashes.json are equal
      to a FRESH recomputation from the module, not merely to each other;
  C4  every metric ID named in section 3 exists in data/balance9_metric_spec.json, and every metric
      in the spec is named in section 3 -- a spec entry the preregistration forgot is a metric that
      could be introduced later and called preregistered;
  C5  the frozen numbers in the document (Level-A legs, eta, tau_tie, decoy band, the outcome
      thresholds of section 10, the equivalence margin) equal the constants in the code;
  C5k/C5l/C5m  every claim the document makes ABOUT PRIOR DATA is re-derived from the raw corpus:
      the policy tuples, the prefix credited with a legacy figure, and the per-cell Level-A counts.
      These exist because three such claims survived the first 42 checks. The gate compared the
      document to the code and to itself, and a plausible-but-wrong statement about Balance-7 is
      consistent with both -- C5a-C5j could not have caught any of them. Section 13.1 is exempt as a
      quotation zone (it must be able to quote retracted wording) and is separately required to
      preserve what it retracts, so the exemption cannot become an eraser. Negative test:
      _b9_prereg_faults.py, 7/7, which is what revealed that the first draft of C5k read only the
      last occurrence of each policy name and was therefore masked by the correction log itself;
  C6  the seed blocks of section 2 are mutually disjoint AND disjoint from every seed used anywhere
      in Balance-6/7/8, established by rescanning the raw corpus rather than by trusting section 1;
  C7  the family sizes stated for Holm correction match the hypothesis counts in the document;
  C8  the amendment table is empty, or every amendment row records whether data had been observed.

On failure nothing is written and the process exits non-zero, so a build step that runs this before
the runner cannot proceed on a stale or contradicted preregistration.

Writes exactly one artefact: data/balance9_preregistration.json.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(HERE))

import balance9_prompts as PR  # noqa: E402
import balance9_ratspec as RS  # noqa: E402
# imported for C9u3 only: the environment, not the spec, decides how many merchants a market has,
# and the only honest way to ask is to draw one and count.
import equilibrium as EQ  # noqa: E402
import hetero as HET  # noqa: E402

DOC = HERE / "BALANCE9_PREREGISTRATION.md"
PROTO = HERE / "BALANCE9_PROTOCOL.md"
SPEC = HERE / "data" / "balance9_metric_spec.json"
PROMPTS = HERE / "data" / "balance9_prompt_hashes.json"
OUT = HERE / "data" / "balance9_preregistration.json"

SEED_BLOCKS = {
    "B9-P2": RS.P2_BLOCK,
    "B9-P3": (9000, 9059),
    "B9-P3X": (9060, 9179),
    "B9-P4": (20000, 20999),
    "B9-P5": (30000, 30059),
    "B9-P6": (40000, 40059),
}
DIAG_MIN = 990000          # the B9-DIAG block; reserved, never analysed

_R = []


def ck(label, cond, detail=""):
    """Record and print one check.

    `detail` is quoted document text and can therefore contain anything the document contains --
    including the subscript in `a₀`, which a cp936 console cannot encode. A gate that dies with a
    UnicodeEncodeError while reporting a failure has converted a legible failure into a crash, so
    the detail is forced to ASCII before it reaches stdout. The check's verdict never depends on it.
    """
    _R.append((label, bool(cond)))
    if detail and not cond:
        detail = str(detail).encode("ascii", "replace").decode("ascii")
    print(("  ok   " if cond else "  FAIL ") + label + (f"   [{detail}]" if detail and not cond else ""))
    return bool(cond)


# ---------------------------------------------------------------------------------------------
def prior_seeds():
    """Every market seed used anywhere in Balance-6/7/8. Rescanned, not remembered.

    Section 1 of the preregistration asserts the highest prior seed is 7699. That assertion is
    exactly the kind of thing that is true when written and false three experiments later, so it is
    re-derived here from the files rather than carried forward as a constant.
    """
    seen = set()
    for f in glob.glob(str(HERE / "data" / "*.jsonl")):
        if "balance9" in os.path.basename(f):
            continue
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                for k in ("seed", "market_seed", "table_seed"):
                    v = d.get(k)
                    if isinstance(v, int):
                        seen.add(v)
    for f in glob.glob(str(HERE / "data" / "*.json")):
        if "balance9" in os.path.basename(f):
            continue
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue

        def walk(o, depth=0):
            if depth > 8:
                return
            if isinstance(o, dict):
                for k, v in o.items():
                    if k in ("seeds", "seed_list") and isinstance(v, list):
                        seen.update(x for x in v if isinstance(x, int))
                    else:
                        walk(v, depth + 1)
            elif isinstance(o, list):
                for v in o[:200]:
                    walk(v, depth + 1)
        walk(d)
    return seen


def corpus_by_prefix():
    """{raw prefix -> {policy names it actually contains}}, read from the raw JSONL.

    Used to check claims of the form "experiment X shows Y about policy Z". Balance-9's
    preregistration made exactly that kind of claim about a prefix that never ran the policy.
    """
    out = {}
    for f in sorted(glob.glob(str(HERE / "data" / "balance[678]*_raw.jsonl"))):
        pref = os.path.basename(f).replace("_raw.jsonl", "")
        pols = out.setdefault(pref, set())
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                p = d.get("policy") or d.get("policy_name")
                if p:
                    pols.add(p)
    return out


def corpus_policies():
    """{policy name -> {(kappa, tau), ...}} as actually recorded in Balance-6/7/8 raw rows.

    A set, not a scalar, on purpose: if a name were ever used for two different tuples the check
    must see that rather than silently take the first.
    """
    out = {}
    for f in sorted(glob.glob(str(HERE / "data" / "balance[678]*_raw.jsonl"))):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                p, k, t = (d.get("policy") or d.get("policy_name")), d.get("kappa"), d.get("tau")
                if p is not None and k is not None and t is not None:
                    out.setdefault(p, set()).add((round(float(k), 9), round(float(t), 9)))
    return out


def doc_policies(text):
    """{policy name -> SET of every (kappa, tau) tuple written for it in `text`}.

    A set of all occurrences, not one value per name. The first version of this returned a dict
    keyed by name, so the LAST occurrence silently won -- and the negative test caught the
    consequence: with the section 13.1 correction log quoting both the retracted tuple and the
    corrected one, the log's own correct value overwrote a deliberately corrupted section 5.1 and
    the check passed on a document that contained the exact error it was written to catch. A
    "last one wins" reader of a document that quotes itself is not a checker.

    Tolerates the bold markers and the 'kappa='/'tau=' labels, because the check must apply to the
    sentence as written, not to a sanitised version of it.
    """
    out = {}
    for name, inner in re.findall(r"`(P_[A-Za-z0-9_]+)`\s*=\s*\(([^)]*)\)", text):
        nums = re.findall(r"[-+]?\d+(?:\.\d+)?", inner.replace("*", ""))
        if len(nums) >= 2:
            out.setdefault(name, set()).add((round(float(nums[0]), 9), round(float(nums[1]), 9)))
    return out


def section(text, header):
    """The text of one '## n.' section, up to the next '## ' header."""
    i = text.find(header)
    if i < 0:
        return ""
    j = text.find("\n## ", i + len(header))
    return text[i:] if j < 0 else text[i:j]


def unwrapped(s):
    """Collapse all whitespace, so a prose check does not depend on where a line wraps.

    Markdown is hard-wrapped at 100 columns, so a sentence-level check written as a literal
    substring silently becomes a check on the position of a newline. C9b1 failed on its first run
    for exactly this reason: the committed sentence is real and present, but the wrap falls between
    'is not' and 'too large'. Typography is not the thing being certified.
    """
    return " ".join(str(s).split())


def subsection(text, header):
    """The text of one '### n.m' subsection, up to the next header of level 3 or shallower.

    Separate from `section` because '## 14.' is a substring of '### 14.1' and a naive splitter
    silently returns the six characters between them -- which is truthy, non-empty, and passes an
    existence check while failing every content check underneath it. That is how the first run of
    C9 reported 'section 14 exists' alongside sixteen failures saying it contained nothing.
    """
    i = text.find(header)
    if i < 0:
        return ""
    j = min([k for k in (text.find("\n### ", i + len(header)),
                         text.find("\n## ", i + len(header))) if k >= 0] or [-1])
    return text[i:] if j < 0 else text[i:j]


def main():
    print("=" * 96)
    print("GATE G-prereg  --  BALANCE9_PREREGISTRATION.md vs the code it preregisters")
    print("=" * 96)

    # These three carried no check id until the attribution-coverage report was built and found them
    # sitting in the domain as the bare words "the" and "no". A check with no id cannot be named by a
    # fault, so no fault could ever have established one, and `attribute()` would have scored a
    # firing against whatever its first word happened to be. Ids are not decoration here; they are
    # what makes a check addressable by the negative test.
    ck("C0a the preregistration exists", DOC.exists())
    ck("C0b the metric spec exists (section 7 must be discharged first)", SPEC.exists())
    ck("C0c the prompt hash file exists (the prompt surface must be frozen first)", PROMPTS.exists())
    if not (DOC.exists() and SPEC.exists() and PROMPTS.exists()):
        raise SystemExit("GATE FAILED: a prerequisite artefact is missing. Nothing was written.")

    text = DOC.read_text(encoding="utf-8")
    doc_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    spec = json.load(open(SPEC, encoding="utf-8"))
    pfile = json.load(open(PROMPTS, encoding="utf-8"))

    # -- C2: the stimulus sentences ------------------------------------------------------------
    ck("C2a the factual-mark sentence in section 11 is byte-identical to MARK_FACTUAL",
       PR.MARK_FACTUAL in text)
    ck("C2b the recommendation sentence in section 11 is byte-identical to MARK_RECOMMEND",
       PR.MARK_RECOMMEND in text)
    gen = PR.retry_generic(None, 3, 0.0537, PR.ETA)
    tail = gen.split(") ", 1)[1]
    ck("C2c the generic retry message quoted in section 11 is the message the code sends",
       tail in text and "Your submitted action_index [" in text)
    ck("C2d the objective lexicon in section 7.2 matches OBJECTIVE_LEXICON exactly",
       all(w in text for w in PR.OBJECTIVE_LEXICON)
       and len(re.findall(r"OBJECTIVE_LEXICON", text)) >= 1)
    ck("C2e the generic message still contains no objective lexicon (re-checked here, not trusted)",
       not [w for w in PR.OBJECTIVE_LEXICON if w in gen.lower()])

    # -- C3: prompt hashes ---------------------------------------------------------------------
    fresh = PR.published_hashes()
    ck("C3a the stored prompt hashes equal a fresh recomputation from the module",
       fresh == pfile["hashes"],
       str(sorted(k for k in fresh if fresh[k] != pfile["hashes"].get(k)))[:200])
    ck("C3b the P3::U hash quoted in section 11 is that hash", fresh["P3::U::system"] in text)
    ck("C3c the module source hash quoted in section 11 is current", fresh["module::source"] in text)
    ck("C3d U, R and G share one hash (the preregistered claim of section 11)",
       fresh["P3::U::system"] == fresh["P3::R::system"] == fresh["P3::G::system"])
    ck("C3e all four P5 arms share the U hash",
       {fresh[f"P5::{a}::system"] for a in PR.P5_ARMS} == {fresh["P3::U::system"]})
    ck("C3f H does NOT share the U hash (the annotation must actually change the prompt)",
       fresh["P3::H::system"] != fresh["P3::U::system"])

    # -- C4: metric IDs ------------------------------------------------------------------------
    ids = set(spec["metrics"].keys())
    named = {m for m in ids if m in text}
    ck(f"C4a every metric in the spec is named in section 3 ({len(named)}/{len(ids)})",
       named == ids, str(sorted(ids - named))[:200])
    doc_ids = set(re.findall(r"`(M\d+[ab]?_[A-Za-z0-9_]+)`", text))
    ck("C4b every metric ID quoted in the document exists in the spec",
       doc_ids <= ids, str(sorted(doc_ids - ids))[:200])
    ck("C4c the VALIDITY filter is named", "VALIDITY" in text and "level_A" in spec)

    # -- C5: the frozen numbers ----------------------------------------------------------------
    la = spec["level_A_thresholds"]
    ck("C5a Level-A A1 threshold in the document equals the spec (0.80)",
       abs(la["sustained_rate_min"] - 0.80) < 1e-12 and "≥ **0.80**" in text)
    ck("C5b Level-A A2 threshold equals the spec (0.02)",
       abs(la["median_tail_rel_exploit_max"] - 0.02) < 1e-12 and "≤ **0.02**" in text)
    ck("C5c Level-A A3 threshold equals the spec (0.95)",
       abs(la["mean_gmv_ratio_min"] - 0.95) < 1e-12 and "≥ **0.95**" in text)
    ck("C5d eta equals the module constant (0.01)", PR.ETA == 0.01 and "η = 0.01" in text)
    ck("C5e tau_tie equals the module constant (1e-12)",
       PR.TAU_TIE == 1e-12 and "1e-12" in text)
    ck("C5f the decoy band equals the module constant [0.04, 0.06]",
       list(PR.DECOY_BAND) == [0.04, 0.06] and "[0.04, 0.06]" in text)
    ck("C5g the Outcome-B threshold 0.70 is quoted from the mandate", "≥ **0.70**" in text)
    ck("C5h the Outcome-A and Outcome-C thresholds are numeric, not adjectival",
       "≥ **0.50**" in text and "> **0.50**" in text)
    ck("C5i the G-vs-R equivalence margin is frozen at 0.05", "**Δ = 0.05**" in text)
    ck("C5j the Outcome-H materiality threshold is frozen at 0.05", "exceeds **0.05**" in text)

    # -- C5k/C5l/C5m: claims the document makes about the LEGACY corpus -------------------------
    # Added after three errors survived a 42-check gate (preregistration section 13.1). All three
    # were statements ABOUT prior data that no check ever compared TO prior data: the gate verified
    # the document against the code and against itself, and a wrong policy tuple is consistent with
    # both. These re-derive the claims from the raw corpus instead.
    # Section 13.1 is a QUOTATION ZONE: its job is to record retracted wording verbatim, so it is
    # exempt from the claim checks and then separately required to still contain what it retracted.
    # Without the exemption the log's own corrected values mask errors elsewhere (see doc_policies).
    pf = section(text, "### 13.1")
    body = text.replace(pf, "") if pf else text

    corpus = corpus_policies()
    quoted = doc_policies(body)
    ck("C5k0 the document quotes tuples for both P3 policies",
       {"P_GMV", "P_robust"} <= set(quoted), f"quoted={sorted(quoted)}")
    bad = {n: (sorted(vs - corpus[n]), sorted(corpus[n])) for n, vs in quoted.items()
           if n in corpus and not vs <= corpus[n]}
    ck(f"C5k every policy tuple quoted outside the correction log matches the raw corpus "
       f"({sum(len(v) for v in quoted.values())} quoted, {len(corpus)} names in corpus)", not bad,
       "; ".join(f"{n}: doc says {v}, corpus says {c}" for n, (v, c) in bad.items()))
    ck("C5k1 the correction log still records the retracted P_robust tuple",
       bool(pf) and (1.0, 0.25) in doc_policies(pf).get("P_robust", set()))

    # C5l: the Level-A reference figure is attributed to a prefix, and that prefix must be one that
    # could actually have produced it. balance7_main ran both policies; no balance8 prefix ran
    # P_robust at all, so no balance8 prefix can source a two-policy table.
    ref_prefix = spec["gates"]["balance7_reproduction"]["prefix"]
    b8_pols = {p for pref, pols in corpus_by_prefix().items()
               if pref.startswith("balance8") for p in pols}
    ck("C5l0 no balance8 prefix ran P_robust (so none can source a 2-policy Level-A table)",
       "P_robust" not in b8_pols, f"balance8 policies = {sorted(b8_pols)}")
    seg = section(text, "## 4.")
    ck(f"C5l the Level-A reference figure names the prefix that produced it ({ref_prefix}) and "
       f"states Balance-8's omission", f"`{ref_prefix}`" in seg
       and "Balance-8 never ran `P_robust`" in seg)

    # C5m: re-derive the per-cell verdict counts rather than trusting the typed ones, and forbid the
    # flat pass total that collapses PASS* into PASS against the document's own rule A5.
    cells = spec["level_A_by_cell"]
    defs = ("b6", "b7")
    agree = all(c[d]["verdict"] == c[defs[0]]["verdict"] for c in cells.values() for d in defs)
    n_pass = sum(1 for c in cells.values() if all(c[d]["verdict"] == "PASS" for d in defs))
    n_marg = sum(1 for c in cells.values() if all(c[d]["verdict"] == "PASS*" for d in defs))
    n_fail = len(cells) - n_pass - n_marg
    ck("C5m0 the two exploitability definitions agree cell-by-cell, as section 4 claims", agree)
    ck(f"C5m section 4 quotes the re-derived counts split as {n_pass} PASS / {n_marg} PASS* / "
       f"{n_fail} fail",
       all(f"**{n}** / {len(cells)}" in seg for n in (n_pass, n_marg, n_fail)))
    # C5m1 scans the body (13.1 exempt, as above) so that the log may quote the retracted phrasing;
    # C5m2 then requires the log to still contain it, so the exemption cannot be used to delete the
    # record. (Same shape as the horizon gate in balance9_prompts.py, which once flagged the
    # sentence denying a horizon leak as a horizon leak.)
    flat = f"{n_pass + n_marg}/{len(cells)} cells passing"
    ck(f"C5m1 the document body nowhere collapses them into a bare '{flat}'",
       flat not in body and flat.replace("/", " / ") not in body)
    ck("C5m2 the 13.1 correction log still records the retracted wording verbatim",
       bool(pf) and flat in pf)

    # -- C6: seed blocks -----------------------------------------------------------------------
    prior = prior_seeds()
    hi = max(prior) if prior else -1
    ck(f"C6a the corpus scan reproduces the section-1 claim that the highest prior seed is 7699 "
       f"(found {hi})", hi == 7699)
    blocks = {k: set(range(a, b + 1)) for k, (a, b) in SEED_BLOCKS.items()}
    overlaps = [(a, b) for a in blocks for b in blocks if a < b and blocks[a] & blocks[b]]
    ck("C6b the Balance-9 seed blocks are mutually disjoint", not overlaps, str(overlaps))
    contaminated = {k: sorted(v & prior)[:5] for k, v in blocks.items() if v & prior}
    ck("C6c no Balance-9 seed was ever used in a prior round", not contaminated, str(contaminated))
    ck("C6d every block in section 2 appears in the document with its exact range",
       all(f"{a}–{b}" in text or f"{a}-{b}" in text for a, b in SEED_BLOCKS.values()))
    ck("C6e the diagnostic fixture seed is in the reserved B9-DIAG block, not an experimental one",
       PR.FIXTURE_SEED >= DIAG_MIN and not any(PR.FIXTURE_SEED in v for v in blocks.values()))

    # C6f-C6i: the B9-P2 splits (amendment A-1). These are sub-ranges INSIDE one block, so the
    # disjointness they must satisfy is different from C6b's: they must partition their parent
    # exactly -- no overlap AND no gap. A gap would be a pool of allocated-but-unassigned seeds,
    # which is precisely the reservoir a later "we also ran these" would draw on.
    lo2, hi2 = RS.P2_BLOCK
    parts = {k: set(range(a, b + 1)) for k, (a, b) in RS.SPLITS.items()}
    ck("C6f the three B9-P2 splits are mutually disjoint",
       not [(a, b) for a in parts for b in parts if a < b and parts[a] & parts[b]])
    union = set().union(*parts.values())
    ck("C6g the three splits partition B9-P2 exactly, with no unassigned seeds left over",
       union == set(range(lo2, hi2 + 1)),
       f"{len(union)} assigned vs {hi2 - lo2 + 1} in block")
    ck("C6h TRAIN-CERT is a subset of TRAIN, not a fourth split",
       set(RS.seeds("TRAIN-CERT")) <= parts["TRAIN"])
    ck("C6i section 2 states each split's range and size, and names TRAIN-CERT",
       all(f"{a}–{b}" in text and f"| {len(parts[k])} |" in text
           for k, (a, b) in RS.SPLITS.items()) and "TRAIN-CERT" in text)

    # -- C7: families --------------------------------------------------------------------------
    ck("C7a F1 declares 16 tests and lists 4 hypotheses over 4 cells",
       "**16 tests**" in text and all(f"**H{i}**" in text for i in (1, 2, 3, 4)))
    ck("C7b F2 declares 10 tests and lists 5 hypotheses over 2 models",
       "**10 tests**" in text and all(f"**H{i}**" in text for i in (5, 6, 7, 8, 9)))
    ck("C7c F3 declares 6 tests and lists 3 hypotheses over 2 models",
       "**6 tests**" in text and all(f"**H{i}**" in text for i in (10, 11, 12)))
    ck("C7d every hypothesis H1..H12 has a stated direction",
       len(re.findall(r"one-sided|two-sided", text)) >= 12)
    ck("C7e Holm is named for each family", text.count("Holm") >= 3)

    # -- C8: prohibitions and amendments --------------------------------------------------------
    ck("C8a the prohibited-analysis list has at least 12 numbered entries",
       len(re.findall(r"^\d+\. ", text.split("## 12.")[1].split("## 13.")[0], re.M)) >= 12)
    amend = text.split("## 13.")[1].split("## 14.")[0]
    rows = [r for r in re.findall(r"^\| .*\|$", amend, re.M) if "---" not in r]
    body = [r for r in rows if not r.startswith("| # ")]
    # The data-observed column must OPEN with a bold Yes or No. The first version of this check
    # accepted any cell containing the bare word, which a cell reading "no data were observed for
    # most of it" would also satisfy; the answer to "had you looked?" is one of two words, and it
    # belongs at the front of the cell where a reader cannot miss it.
    def observed_cell(row):
        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        return cells[3] if len(cells) > 3 else ""
    ck("C8b every amendment row opens its data-observed cell with a bold Yes or No",
       all("*(none)*" in r or re.match(r"^\*\*(Yes|No)\.?\*\*", observed_cell(r)) for r in body),
       str([observed_cell(r)[:40] for r in body])[:220])
    ck("C8c the document states that a post-data amendment makes affected tests exploratory",
       "converts every affected test to exploratory" in text)
    # C8d/C8e: an amendment that changes the procedure must appear in BOTH documents.
    #
    # The first version of C8d walked one way only -- it took the IDs out of THIS document's
    # amendment table and required each to appear somewhere in the protocol. Amendments A-2 and A-3
    # were introduced as prose inside protocol §14.6 and §14.7 and never given a row in either
    # ledger, so they were outside the check's domain in both senses: not in the table it read from,
    # and not required to be in the table it read to. It passed 84/84 with two amendments unrecorded,
    # while its own comment claimed they must appear in both documents. That is E-8's failure mode
    # (a check complete over the wrong domain) applied to the ledger whose whole purpose is to make
    # post-hoc changes visible -- the worst possible place for it.
    #
    # The domain is now every amendment ID that appears ANYWHERE in either document, including in
    # running prose, and the requirement is a ROW in both tables. An amendment mentioned in a
    # paragraph but absent from the ledgers is exactly the change a reader of the ledgers would
    # never learn about.
    ptext = PROTO.read_text(encoding="utf-8")
    pamend = ptext.split("## 13.")[1].split("## 14.")[0] if "## 13." in ptext else ""
    prows = [r for r in re.findall(r"^\| .*\|$", pamend, re.M) if "---" not in r]

    def table_ids(table_rows):
        out = set()
        for r in table_rows:
            m = re.match(r"^\|\s*\*\*(A-\d+)\*\*\s*\|", r)
            if m:
                out.add(m.group(1))
        return out

    mentioned = set(re.findall(r"\bA-(\d+)\b", text)) | set(re.findall(r"\bA-(\d+)\b", ptext))
    mentioned = {f"A-{n}" for n in mentioned}
    here, there = table_ids(body), table_ids(prows)
    ck("C8d every amendment ID mentioned in either document has a row in THIS document's table",
       not (mentioned - here), str(sorted(mentioned - here)))
    ck("C8e every amendment ID mentioned in either document has a row in the protocol's table",
       not (mentioned - there), str(sorted(mentioned - there)))

    # -- C9: protocol section 14 (amendment A-1) vs the frozen constants -------------------------
    # Same discipline as C2/C5: the document and the code that will consume it are compared, never
    # the document with itself. Every number below is one the rational audit will actually branch on.
    p14 = section(ptext, "## 14.")
    s = {n: subsection(p14, f"### 14.{n}") for n in range(1, 8)}
    ck("C9a the protocol carries a section 14 with all seven subsections",
       bool(p14) and all(s.values()), str([n for n in s if not s[n]]))
    # C9b anchors on the DEFINITIONAL form, not on the digits appearing somewhere in the section.
    # The first draft checked only for the substring "194 481"; the negative test then deleted the
    # definition and the check stayed green, because a later sentence about the gate happened to
    # repeat the number. A presence test over a whole section cannot tell a definition from an echo.
    ck("C9b section 14.1 defines the state count as |S| = 21^4 = 194 481",
       f"`|S| = 21^4 = {RS.N_STATES:,}`".replace(",", " ") in unwrapped(s[1])
       and RS.N_STATES == 21 ** 4)
    ck("C9b1 section 14.2 forecloses the 'too large' escape and the random-simulation substitute",
       "**It is not too large.**" in unwrapped(s[2])
       and "No random simulation is substituted for any of it." in unwrapped(s[2]))
    ck("C9c section 14.3 states the frozen initialization used for reachability",
       f"({RS.INIT_INDEX},{RS.INIT_INDEX},{RS.INIT_INDEX},{RS.INIT_INDEX})" in s[3].replace(" ", ""))
    # This used to carry `(RS.EPS_MAX, RS.GMV_MIN) == (0.05, 0.95)` as a third leg. That literal sat
    # in THIS file, so it pinned the two thresholds no better than they pinned themselves -- a
    # consistent forgery would have edited it along with everything else. C9u3 now anchors both to
    # the Balance-6 metric spec, an artefact that predates this round, which is strictly stronger;
    # errata E-10.1 says the entailed check is to be deleted rather than kept for the tally. What is
    # left here is the one thing C9u3 does not say: that the document states the numbers the code uses.
    ck("C9d section 14.4's balance criterion is M2's two legs at their frozen numbers",
       f"{RS.EPS_MAX}" in unwrapped(s[4]) and f"{RS.GMV_MIN}" in unwrapped(s[4]))
    ck("C9e section 14.5 audits exactly the policies named in balance9_ratspec, with their tuples",
       all(n in unwrapped(s[5]) and f"({k}, {t:.2f})" in unwrapped(s[5])
           for n, (k, t) in RS.POLICIES.items()))
    ck("C9f section 14.5 marks P_sep as a legacy diagnostic that is not evidence of comprehension",
       "legacy extreme diagnostic" in unwrapped(s[5]) and "policy comprehension" in unwrapped(s[5]))
    # Per-ROW, not per-section: '5' occurs in a section about five conditions whatever the
    # threshold is, so a section-wide substring test would pass on any number at all.
    srows = {m.group(1): m.group(0)
             for m in re.finditer(r"(?m)^\| (S\d) \|.*$", s[6])}
    for lbl, name, val in (("S1", "GMV floor", RS.SEP_GMV_MIN),
                           ("S2", "fabrication slack", RS.SEP_FAB_SLACK),
                           ("S3", "index separation", RS.SEP_INDEX_SEP_MIN),
                           ("S4", "outside slack", RS.SEP_OUTSIDE_SLACK),
                           ("S4", "outside cap", RS.SEP_OUTSIDE_ABS),
                           ("S5", "convergence sweeps", RS.SEP_CONV_SWEEPS)):
        # Accept either '0.1' or '0.10': the document is entitled to align its decimals, and a gate
        # that failed on trailing-zero formatting would be checking typography, not thresholds.
        forms = {f"{val:g}", f"{val:.2f}", str(int(val)) if float(val).is_integer() else ""}
        row = srows.get(lbl, "")
        # The frozen value used to sit INSIDE the bracket, which made the check's identity move
        # whenever the thing it checks moved: AF-64 sets SEP_OUTSIDE_ABS to 0.95 and the check that
        # fired came back as `C9g[S4 outside cap = 0.95]` while the clean run had emitted
        # `C9g[S4 outside cap = 0.6]`. Two names for one check is how E-10.3's `V8d` went unnoticed.
        # The value belongs in the detail, where it is reported and not used as an address.
        ck(f"C9g[{lbl} {name}] the frozen value {val:g} appears in its own row of the 14.6 table",
           bool(row) and "**" in row and any(f for f in forms if f and f in row),
           (row or "MISSING ROW")[:110])
    ck("C9g[S5 rate] section 14.6 requires convergence on 100% of the split",
       f"**{RS.SEP_CONV_RATE:.0%}**" in unwrapped(s[6]))
    ck("C9h section 14.6 pre-commits the exact sentence used when no candidate validates",
       "No practically admissible separating policy was validated in the searched class."
       in unwrapped(s[6]))
    ck("C9i section 14.7 states the joint-enumeration cap and forbids a sampled 'worst'",
       f"{RS.JOINT_ENUM_CAP:,}".replace(",", " ") in unwrapped(s[7])
       and 'never as a sampled estimate wearing the word "worst"' in unwrapped(s[7]))
    ck("C9j section 14.7 names both frozen error distributions and their shared slip probability",
       all(f"`{d}`" in unwrapped(s[7]) for d in RS.ERR_DISTS) and f"**{RS.ERR_P:.2f}**" in unwrapped(s[7]))
    ck("C9k section 14.7 requires P_safe to be frozen to disk before VALIDATION is read",
       "data/balance9_psafe.json" in unwrapped(s[7]) and "**before**" in unwrapped(s[7]))
    ck("C9l section 14.2 reports exact BR and eta-BR separately, as the mandate requires",
       "never summed or averaged together" in unwrapped(s[2]))
    ck("C9o section 14.1 makes Proposition 14.1 a gated claim rather than an assertion",
       "G-G1" in unwrapped(s[1]) and "computes every best response twice" in unwrapped(s[1]))
    # C9m and C9n used to sit here. They compared `RS.ETA`, `RS.TAU_TIE`, `RS.N_INDEX` and
    # `RS.N_STATES` against the runner and against arithmetic -- and the C9u anchor table below now
    # makes exactly those comparisons, for those constants and for every other one, over a domain
    # that cannot silently shrink. Keeping both would have left every eta or grid fault naming two
    # checks of which one is entailed by the other, and errata E-10.1 says the redundant check is to
    # be deleted rather than kept for the tally. Deleting them is not a loss of coverage: C9u1 fails
    # if any constant loses its declaration, which is a stronger statement than either made.
    # C9p is the two-sided lock on amendment A-2. Section 14.7 asked for a "GMV ratio" without
    # naming a denominator, and the two candidates order policies differently, so leaving the choice
    # to whoever writes the selection code would put a free parameter inside a preregistered rule.
    # Neither side may move alone: the identifier has to be in the document AND be the one the audit
    # stamps into every record, and the document has to still say why the other one is not used.
    ck(f"C9p[A-2] section 14.7 names the frozen selection denominator `{RS.SEL_DENOM}`",
       f"**`{RS.SEL_DENOM}`**" in unwrapped(s[7])
       and "amendment A-2, frozen before any policy's GMV ratio had been computed under either "
           "denominator" in unwrapped(s[7])
       and "for reading, never for selecting" in unwrapped(s[7]))
    # C9q is amendment A-3. Section 14.6 pre-commits a sentence about "the searched class", so the
    # class has to BE somewhere -- an undefined referent turns a falsifiable negative result into a
    # phrase that can be re-aimed after the outcome. Both grids and the candidate count are locked
    # against the document, and the grid is required to contain the incumbents.
    grid = RS.search_grid()
    u6 = unwrapped(s[6])
    # The freeze-time claim is checked in the exact form the chronology supports. An earlier draft
    # said "before the first candidate was evaluated", which is false: the three section 14.5
    # policies ARE candidates in this class and had been audited when A-3 was written. A check that
    # demanded the false sentence would have been enforcing the overstatement.
    ck(f"C9q1[A-3] section 14.6 defines the searched class as {len(grid)} candidates, and states "
       f"the freeze time in the form the chronology supports",
       f"**{len(grid)} candidates**" in u6
       and "amendment A-3, frozen before the search was run" in u6
       and "no other member of the class\nhad been evaluated on any split".replace("\n", " ") in u6
       and "no adaptive refinement" in u6)
    for axis, vals in (("kappa", RS.SEARCH_KAPPA), ("tau", RS.SEARCH_TAU)):
        want = "{" + ", ".join(f"{v:g}" for v in vals) + "}"
        ck(f"C9q2[A-3/{axis}] section 14.6 lists that axis exactly as balance9_ratspec has it "
           f"({len(vals)} values)", want in u6 and f"({len(vals)} values)" in u6, want)
    ck("C9q3[A-3] the searched class contains every policy section 14.5 audits",
       all(tuple(float(x) for x in p) in grid for p in RS.POLICIES.values())
       and "contains all three policies of §14.5" in u6,
       [p for p in RS.POLICIES if tuple(float(x) for x in RS.POLICIES[p]) not in grid])
    ck("C9q4[A-3] the searched class is enumerated once, with no duplicate candidate",
       len(set(grid)) == len(grid) == len(RS.SEARCH_KAPPA) * len(RS.SEARCH_TAU))
    # The first version of C9q5 read three phrases -- that a shortlist is formed, that an empty one
    # stops the VALIDATION read, and the reason -- but not the sentence that states the restriction
    # itself. Fault AF-30 replaced "Only the shortlist is evaluated on `VALIDATION`" with "Every
    # candidate is evaluated on `VALIDATION`" and the gate passed: the document preregistered the
    # opposite procedure while all three checked phrases stood, because a shortlist that is formed
    # and then ignored still satisfies "those that pass form the shortlist". The restriction is the
    # claim; it is now the first leg (errata E-10).
    ck("C9q5[A-3] section 14.6 restricts VALIDATION to the TRAIN shortlist, and forbids the read "
       "entirely if the shortlist is empty",
       "Only the shortlist is evaluated on\n`VALIDATION`".replace("\n", " ") in u6
       and "form the **shortlist**" in u6
       and "If the\nshortlist is empty, `VALIDATION` is not read at all".replace("\n", " ")
       in u6 and "something passes by luck" in u6)

    # C9r is amendment A-4: the P_safe candidate class. Section 14.7 ranged the words "for every
    # candidate" and "restrict to candidates" over a set it never named, which is A-3's defect in the
    # subsection next door. The lock is two-sided as usual, and it has an extra leg the others do not
    # need: the class must be the SAME OBJECT as the A-3 class, not an equal-looking copy, because
    # two literal lists could drift and then 14.7 would select from a set 14.6 had not searched.
    u7 = unwrapped(s[7])
    ck(f"C9r1[A-4] section 14.7 names the P_safe candidate class and identifies it with A-3's",
       "amendment A-4" in u7
       and "The `P_safe` candidate class is the class A-3 froze" in u7
       and f"`{RS.PSAFE_CLASS}`" in u7 and "no adaptive refinement and no second pass" in u7)
    # `psafe_class()` REFUSES to answer when the identifier is not A-3's. That is right for the spec
    # module and wrong for a gate: an exception raised inside a check's condition aborts the run, and
    # every check below it goes unevaluated while the transcript shows only the ones that had already
    # fired. A condition that raised is a condition that was not satisfied, so it is recorded as this
    # check failing and the remaining hundred still execute -- errata E-8, a check that never runs is
    # not a check, applied to the checks a fault happens to skip past.
    try:
        psafe_ok = RS.psafe_class() == RS.search_grid() and len(RS.psafe_class()) == len(grid)
        psafe_why = ""
    except Exception as exc:                                                        # noqa: BLE001
        psafe_ok, psafe_why = False, f"psafe_class() refused to answer: {exc}"
    ck("C9r2[A-4] balance9_ratspec returns the A-3 class itself for P_safe, candidate for candidate",
       psafe_ok, psafe_why)
    ck("C9r3[A-4] section 14.7 discloses that the class was frozen after 8.3 and says which part is "
       "exploratory",
       "frozen after §8.3" in u7 and "exploratory" in u7
       and "the freeze-then-held-out sequence\nis confirmatory".replace("\n", " ") in u7)

    # C9s is amendment A-5. A-2 fixed the denominator of each per-market ratio and stopped there, so
    # the step from 300 per-market numbers to the one number per candidate that the rule compares was
    # left unwritten -- and mean-of-ratios and ratio-of-means can admit different candidates. Two
    # separate things are locked here because they can fail separately: WHICH aggregation, and
    # whether the rule is a total order at all. A rule that returns two answers is not a
    # preregistered rule, whichever of the two the code happens to take.
    ck(f"C9s1[A-5] section 14.7 names the frozen aggregation `{RS.SEL_AGG}` and says what the "
       "rejected reading would have done",
       f"**`{RS.SEL_AGG}`**" in u7
       and "amendment A-5, frozen before any candidate's selection statistic had been computed" in u7
       and "ratio of the two means over markets" in u7
       and "outvote the rest" in u7)
    ck("C9s2[A-5] the aggregation is the one section 14.6 rule S1 already performs, not a new one",
       "the arithmetic §14.6 rule S1 already performs" in u7
       and "reading the two legs of a single rule with two different aggregations" in u7)
    ck("C9s3[A-5] section 14.7 states the full ordered key list, so the selection rule is a function",
       all(k in u7 for k in RS.SEL_KEYS)
       and "`(worst_joint_ratio_pgmv, gmv_ratio_vs_pgmv, class_index)`" in u7
       and "descending on the first two, ascending on the last" in u7
       and "position in the A-3 enumeration" in u7)
    ck("C9s4[A-5] the code's key list matches the document's, and names three distinct keys",
       RS.SEL_KEYS == ("worst_joint_ratio_pgmv", "gmv_ratio_vs_pgmv", "class_index")
       and len(set(RS.SEL_KEYS)) == 3)
    # The unreachability of section 14.7's joint tie-break is a structural fact about M and K, not an
    # observation about a run, so it belongs in the preregistration gate and not only in the audit.
    # If it ever became reachable, each candidate's mean would silently begin ranging over whichever
    # markets were feasible FOR THAT CANDIDATE, and the comparison would stop being paired.
    ck("C9s5[A-5] section 14.7 records that the joint tie-break is unreachable at this M and K, and "
       "it is",
       RS.N_STATES <= RS.JOINT_ENUM_CAP
       and f"`{RS.N_INDEX + 1}^{RS.M_MERCHANTS} = {RS.N_STATES:,}`".replace(",", " ") in u7
       and "the clause never fires" in u7
       and "candidates would then be compared over different market sets" in u7
       and "G-S2a" in u7 and "G-S2b" in u7)

    # C9t is amendment A-6, and it guards the single most abusable word in §14. Everything else in
    # this section decides how a number is computed; "replicates" decides whether the result of the
    # whole exercise is written up as a success. An undefined success criterion is not a weak
    # criterion, it is the absence of one, and it is worth more to an author than any threshold in
    # this file. Four legs, because the three conditions and the pre-committed failure sentence can
    # each disappear on their own.
    ck("C9t1[A-6] section 14.7 defines replication as three named conditions, not as a word",
       "**R1 — advantage.**" in u7 and "**R2 — admissibility retained.**" in u7
       and "**R3 — not carried by a handful of markets.**" in u7
       and "exactly when all three hold" in u7)
    ck(f"C9t2[A-6] R3's win rate and its tie convention are stated and match the code "
       f"({RS.REP_WIN_MIN})",
       f"≥ **{RS.REP_WIN_MIN:.2f}**" in u7 and "ties counted as losses" in u7
       and "R1 alone is a statement about a mean" in u7)
    # R2 re-applies the TRAIN floor. If the document quietly named a lower one for the held-out
    # splits, a policy could fail admissibility where it was selected and pass it where it was
    # tested, which is the trade §14.7 says it is not making.
    ck("C9t3[A-6] R2 re-applies the SAME admissibility floor the TRAIN selection used",
       f"≥ **{RS.PSAFE_GMV_FLOOR:.2f}**, the same" in u7
       and "floor §14.7 applies on `TRAIN`" in u7
       and "has not replicated; it has changed the trade" in u7)
    ck("C9t4[A-6] the absence of a minimum effect size is disclosed, and 0.01 is a label not a "
       "threshold",
       "**No minimum effect size is preregistered, and that is a limitation rather than an "
       "oversight.**" in u7
       and f"below **{RS.REP_NEGLIGIBLE:.2f}**" in u7
       and "That phrase changes the words, never the verdict." in u7)
    ck("C9t5[A-6] section 14.7 pre-commits the sentence used when replication fails",
       'the pre-committed sentence is: **"The implementation-robustness advantage of `P_safe` over '
       '`P_GMV` did not replicate on the held-out split."**' in u7
       and "written before the split is read" in u7)
    ck("C9t6[A-6] the held-out policy set is frozen, and contains the two policies compared",
       RS.HELDOUT_POLICIES == ("P_GMV", "P_robust", "P_safe")
       and "`{P_GMV, P_robust, P_safe}`" in u7
       and all(p in RS.POLICIES for p in RS.HELDOUT_POLICIES if p != "P_safe"))

    # -- C9u: what pins each frozen constant to its VALUE ------------------------------------------
    #
    # Every C9 check above compares the document against the code. That catches one side moving. It
    # cannot catch both sides moving together, and fault AF-54 demonstrated the gap on the worst
    # possible constant: `PSAFE_GMV_FLOOR`, the 95% admissibility floor mandate §8.4 states in words,
    # was lowered to 0.90 in `balance9_ratspec` AND in both sentences of §14.7 that quote it, and the
    # gate reported 101/101. Nothing in it had ever been told what that number is supposed to be --
    # only that two places should say the same thing.
    #
    # The repair is not another literal. It is to make the gate carry, for every public constant in
    # the frozen spec, a declaration of WHAT PINS IT, of which there are only four kinds:
    #
    #   mandate      the Balance-9 mandate fixes the number in words. The literal goes here, and a
    #                consistent forgery is caught because the mandate is not a file I can edit.
    #   environment  another artefact that predates Balance-9 already carries it -- the runner, the
    #                metric spec reconstructed from balance6_analyze.py. Checked by identity.
    #   structure    arithmetic or an invariant determines it; recomputed, never quoted.
    #   free         nothing outside the frozen documents fixes it. This is an honest category, not
    #                a failure: SEP_FAB_SLACK is a design choice and there is no external truth to
    #                compare it to. What holds a free constant still is the PUBLISHED HASH of the
    #                document that states it -- so C9u4 requires the value to actually appear in a
    #                hashed document, or the hash is pinning nothing.
    #
    # C9u1 is the part that keeps this alive: it ranges over the module's namespace, so a constant
    # added later without a declaration fails the gate instead of quietly joining the unpinned set.
    # That is errata E-8's lesson (a check must be complete over its domain) applied to the question
    # E-10.1 left open (a check must be the reason something fails).
    hashed = unwrapped(text) + " " + unwrapped(ptext)

    def _renderings(v):
        """Every way this value could reasonably be written in prose, for the C9u4 presence test."""
        if isinstance(v, bool):
            return {str(v)}
        if isinstance(v, float):
            out = {f"{v}", f"{v:g}", f"{v:.2f}", f"{v:.3f}"}
            if 0 < v <= 1:
                out |= {f"{v:.0%}", f"{v * 100:g}%"}
            return out
        if isinstance(v, int):
            return {str(v), f"{v:,}", f"{v:,}".replace(",", " ")}
        if isinstance(v, str):
            return {v}
        if isinstance(v, (tuple, list, set)):
            return set().union(*[_renderings(x) for x in v]) if v else {""}
        if isinstance(v, dict):
            return set().union(*[_renderings(x) for x in v.values()]) if v else {""}
        return {str(v)}

    def pol_anchor():
        """`RS.POLICIES`, with each incumbent's tuple replaced by the one the raw corpus records.

        `P_sep` is deliberately absent from the corpus -- it is a Balance-9 diagnostic that no prior
        round ran -- so its tuple falls through unpinned and is held only by the document hash, like
        any other free constant. That is stated here rather than left to be inferred from the fact
        that the check passes. A corpus name carrying two different tuples is not an anchor at all,
        so it resolves to a sentinel that cannot match and the check fails loudly.
        """
        out = {}
        for k in RS.POLICIES:
            vs = sorted(corpus.get(k, ()))
            out[k] = (RS.POLICIES[k] if not vs else
                      tuple(vs[0]) if len(vs) == 1 else ("AMBIGUOUS_IN_CORPUS", k))
        return out

    ANCHORS = {
        # -- mandate: the words of the Balance-9 mandate fix these, and I cannot edit the mandate ---
        # §8.4: "policies whose rational equilibrium GMV is at least 95% of `P_GMV`".
        "PSAFE_GMV_FLOOR": ("mandate", 0.95),
        # -- environment: an artefact older than Balance-9 already carries the number ---------------
        "ETA": ("environment", lambda: PR.ETA),
        "TAU_TIE": ("environment", lambda: PR.TAU_TIE),
        "N_INDEX": ("environment", lambda: PR.NIDX),
        "EPS_MAX": ("environment", lambda: spec["constants"]["EPS_MAX"]),
        "GMV_MIN": ("environment", lambda: spec["constants"]["GRATIO_MIN"]),
        "POLICIES": ("environment", pol_anchor),
        # the environment decides how many merchants a market has; the spec must follow it, not the
        # other way round. Drawing one frozen market is the only way to ask that question of the
        # code that will actually be run.
        "M_MERCHANTS": ("environment",
                        lambda: len(HET.draw_market_seeded(EQ.Config(), RS.SPLITS["TRAIN"][0])[0])),
        # -- structure: recomputed from something else in the spec, never quoted -------------------
        "N_STATES": ("structure", lambda: (RS.N_INDEX + 1) ** RS.M_MERCHANTS),
        "INIT_INDEX": ("structure", lambda: RS.N_INDEX // 2),          # f_last_init = 0.5
        "PSAFE_CLASS": ("structure", lambda: "A3_GRID"),               # the only value A-4 admits
        "SEL_AGG": ("structure", lambda: "mean_of_ratios"),            # = §14.6 S1's arithmetic
        "P2_BLOCK": ("structure", lambda: (min(lo for lo, _ in RS.SPLITS.values()),
                                           max(hi for _, hi in RS.SPLITS.values()))),
        # -- free: held still by the published document hash, and by nothing else ------------------
        # SPLITS, TRAIN_CERT and HELDOUT_POLICIES are design choices, so they belong here rather
        # than under `structure`. An anchor of `lambda: RS.SPLITS` would have been a check that
        # cannot fail (errata E-10), and pinning TRAIN_CERT to "the first hundred of TRAIN" would
        # have over-claimed: what is structural about it is that it lies INSIDE TRAIN, which C6
        # already establishes with the reason attached.
        "SPLITS": ("free", None), "TRAIN_CERT": ("free", None),
        "HELDOUT_POLICIES": ("free", None),
        "TIE_RULE": ("free", None), "SEP_GMV_MIN": ("free", None),
        "SEP_FAB_SLACK": ("free", None), "SEP_INDEX_SEP_MIN": ("free", None),
        "SEP_OUTSIDE_SLACK": ("free", None), "SEP_OUTSIDE_ABS": ("free", None),
        "SEP_CONV_SWEEPS": ("free", None), "SEP_CONV_RATE": ("free", None),
        "SEARCH_KAPPA": ("free", None), "SEARCH_TAU": ("free", None),
        "JOINT_ENUM_CAP": ("free", None), "ERR_P": ("free", None), "ERR_DISTS": ("free", None),
        "SEL_DENOM": ("free", None), "SEL_KEYS": ("free", None),
        "REP_WIN_MIN": ("free", None), "REP_NEGLIGIBLE": ("free", None),
    }
    live = {n for n in dir(RS) if n.isupper() and not n.startswith("_")}
    ck("C9u1 every frozen constant declares what pins its value, and every declaration names a "
       "live constant",
       live == set(ANCHORS), str(sorted(live ^ set(ANCHORS))))
    mand = {n: e for n, (a, e) in ANCHORS.items() if a == "mandate"}
    ck("C9u2 every mandate-anchored constant equals the number the mandate states in words",
       all(getattr(RS, n) == e for n, e in mand.items()),
       str({n: getattr(RS, n) for n, e in mand.items() if getattr(RS, n) != e}))
    derived = {n: e for n, (a, e) in ANCHORS.items() if a in ("environment", "structure")}
    bad_d = {n: (getattr(RS, n), e()) for n, e in derived.items() if getattr(RS, n) != e()}
    ck("C9u3 every environment- or structure-anchored constant equals what its anchor computes",
       not bad_d, str(bad_d))
    free = [n for n, (a, _) in ANCHORS.items() if a == "free"]
    unstated = [n for n in free
                if not any(r and r in hashed for r in _renderings(getattr(RS, n)))]
    ck("C9u4 every free constant's value is written in a hashed document, since nothing else "
       f"pins it ({len(free)} free constants)",
       not unstated, str(unstated))

    # -- C9v: amendment A-7, the reporting rule for multi-condition verdicts --------------------
    #
    # THIS document's A-7 row says "`BALANCE9_PROTOCOL.md` §10 gains the standing reporting rule".
    # Without the three checks below that is a claim in one hashed file about the contents of
    # another, and deleting the paragraph from §10 would leave the ledger describing a rule the
    # protocol does not carry, at 103/103. That is E-7 exactly ("a docstring is not evidence") --
    # and it is worse here than in a docstring, because the sentence being taken on trust is the
    # countermeasure to a finding about this corpus's own published negative result.
    #
    # Three legs, none entailing another. §10 could carry the rule while dropping the clause that
    # makes it a disclosure; it could carry both while dropping §14.7's exclusion. The middle leg
    # is the one that does real work: the A-7 row claims "the set of affected tests is empty", and
    # that claim rests entirely on the rule having no verdict in its range. A protocol that quietly
    # let the rule gate a candidate would turn every §8.3-shaped verdict post-data-exploratory
    # while the ledger still said nothing was reclassified.
    # The rule is quoted inside a markdown blockquote, so the raw text carries "> " at every wrap
    # point and a literal substring test would be a test of where the paragraph happened to wrap.
    # Blockquote markers are stripped and whitespace collapsed first: the check is about what the
    # section says, not about its line breaks.
    p10 = re.sub(r"\s+", " ", re.sub(r"^> ?", "", section(ptext, "## 10."), flags=re.M))
    ck("C9v1[A-7] protocol section 10 carries the multi-condition reporting rule, stated over "
       "pairwise RANK correlations of the k conditions over the candidate set",
       "pairwise rank correlations of those *k* conditions over the candidate set are reported "
       "beside it" in p10 and "cleared *k* conditions" in p10)
    ck("C9v2[A-7] section 10 states that the rule is a disclosure and never a gate, which is what "
       "makes this document's 'no test is reclassified' true rather than asserted",
       "**disclosure** duty and never a gate" in p10 and "no threshold moves" in p10
       and "no candidate's verdict changes" in p10)
    ck("C9v3[A-7] section 10 pre-commits the reason 14.7's selection is outside the rule, so the "
       "exemption cannot be argued after a verdict is seen",
       "one** floor and ranks on **one** ordered key" in p10
       and "no conjunction of conditions to collapse" in p10)

    # C8f: an amendment whose disclosure cell answers Yes must state a consequence. The rule at the
    # head of section 13 says a post-data amendment converts affected tests to exploratory; without
    # this check the rule is a sentence that no row has to obey, and "Yes" could be recorded with no
    # visible effect anywhere -- a disclosure that costs nothing and therefore proves nothing.
    yes_rows = [r for r in body if observed_cell(r).startswith("**Yes")]
    ck("C8f every amendment row answering Yes states the consequence for the affected tests",
       all("*Consequence:*" in observed_cell(r) for r in yes_rows),
       f"{len(yes_rows)} Yes rows")

    # -- housekeeping ---------------------------------------------------------------------------
    ck("C10a no paper file is referenced", not re.search(r"\bmain\.tex\b", text))
    ck("C10b the document forbids quoting an unreproduced Balance-8 number",
       "Quoting a Balance-8 number that Balance-9 has not reproduced" in text)

    nfail = sum(1 for _, ok in _R if not ok)
    print("-" * 96)
    print(f"checks: {len(_R) - nfail}/{len(_R)} passed")
    if nfail:
        raise SystemExit(
            f"GATE G-prereg FAILED ({nfail}). The preregistration and the code disagree, so nothing "
            "is preregistered. No confirmatory call may be made. Nothing was written.")

    rec = dict(
        script="balance9_prereg_freeze.py",
        document="BALANCE9_PREREGISTRATION.md",
        document_sha256=doc_hash,
        document_bytes=len(text.encode("utf-8")),
        # C9u4 says a free constant is held still by the published hash of the document that states
        # it. Amendments A-2 through A-6 all live in the PROTOCOL, so recording only the
        # preregistration's hash would have left most of §14 pinned by a hash nobody wrote down --
        # the check would have been true of a document with no fixed identity. Both hashes and the
        # constants themselves are recorded, so a later consistent forgery has to disagree with a
        # file already on disk instead of merely with itself.
        protocol="BALANCE9_PROTOCOL.md",
        protocol_sha256=hashlib.sha256(PROTO.read_bytes()).hexdigest(),
        protocol_bytes=len(ptext.encode("utf-8")),
        rational_constants={n: (list(v) if isinstance(v, tuple) else
                                {k: list(x) for k, x in v.items()} if isinstance(v, dict) else v)
                            for n, v in sorted((n, getattr(RS, n)) for n in ANCHORS)},
        constant_anchors={n: a for n, (a, _) in sorted(ANCHORS.items())},
        prompt_module_sha256=fresh["module::source"],
        prompt_hashes=fresh,
        metric_spec_sha256=hashlib.sha256(SPEC.read_bytes()).hexdigest(),
        seed_blocks={k: list(v) for k, v in SEED_BLOCKS.items()},
        highest_prior_seed=hi,
        n_prior_seeds=len(prior),
        families=dict(F1=16, F2=10, F3=6),
        level_A=la,
        frozen_numbers=dict(eta=PR.ETA, tau_tie=PR.TAU_TIE, decoy_band=list(PR.DECOY_BAND),
                            equivalence_margin_G_vs_R=0.05, outcome_B_threshold=0.70,
                            outcome_A_decoy_following=0.50, outcome_C_loop_share=0.50,
                            outcome_H_materiality=0.05),
        checks_passed=len(_R),
        gate="G-prereg",
        # Re-derived from the raw corpus by C5k/C5l/C5m, not typed. Recorded so that a later round
        # can cite these without rescanning, and so that a disagreement is visible as a diff.
        corpus_policy_tuples={k: sorted(v) for k, v in corpus.items()},
        level_A_reference=dict(
            prefix=ref_prefix, cells=len(cells), n_pass=n_pass, n_pass_marginal=n_marg,
            n_fail=n_fail, definitions_agree=agree,
            note="quoted as 7 + 5, never as 12/16; rule A5 forbids reporting a straddling "
                 "interval as PASS full stop",
        ),
        prefreeze_corrections=["PF-1 P_robust tuple", "PF-2 prefix attribution",
                               "PF-3 PASS* collapsed into PASS"],
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1, ensure_ascii=False)
    print(f"wrote {OUT}")
    print(f"BALANCE9_PREREGISTRATION.md sha256 = {doc_hash}")
    print("G-prereg PASSED. Confirmatory calls are now permitted.")


if __name__ == "__main__":
    main()
