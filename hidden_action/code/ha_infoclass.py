"""ha_infoclass.py -- the information firewall.

The claim this package wants to make is a claim about what happens when the platform CANNOT see the
merchant's action. A claim like that is worth exactly as much as the mechanism that enforces it. In
the previous generation of this experiment the enforcement was a convention -- "we just don't put the
true f in the prompt" -- and a convention is not evidence, because a convention is one refactor away
from being false and nobody would notice.

So the three data classes are objects, not comments:

  MERCHANT_PRIVATE     owned by exactly one merchant. Own honest quality, own baseline complaint
                       propensity, own past actions, own realised profit. Merchant i may condition
                       on its own; merchant i seeing merchant j's is leakage.
  PLATFORM_OBSERVABLE  what the platform actually records: prices, sales, complaint counts, refunds,
                       audit flags, reputations, penalties applied, the published policy, the round
                       index, category aggregates. The platform may condition on all of it. A
                       merchant sees the PUBLISHED subset, which each arm declares.
  EVALUATOR_ONLY       exists only so a paper can be written: every merchant's true fabrication rate,
                       the latent complaint and audit probabilities, the stationary payoff tables,
                       the displayed argmax, regret, exploitability, and the benchmark GMVs. Nobody
                       inside the experiment may condition on any of it, ever.

Three properties are enforced mechanically rather than promised:

  1. TOTALITY. Every field written by any part of the package must be registered in exactly one
     class. `check_registry_covers` fails on an unregistered field, so adding a field to a record
     without classifying it breaks the offline tests instead of silently widening someone's
     information set.

  2. OWNERSHIP. Merchant-private fields carry an owner. A view built for merchant i that contains a
     private field owned by j != i raises. This is the check that catches the subtle version of the
     leak, where a rival's quality arrives inside an otherwise innocent "market summary".

  3. FUNCTIONAL INVARIANCE. The string scan below is a tripwire, not the proof, because a leak can be
     laundered through arithmetic: "your rival is a high-quality seller" contains no number at all.
     The real test is `invariance_probe`, which holds the published observables fixed, perturbs the
     hidden state, and requires the prompt to be byte-identical. A prompt that changes when only the
     hidden state changes is conditioning on the hidden state, whatever it looks like.

Nothing in this module is specific to an LLM. It is deliberately importable and testable with no API,
no network and no model.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass, field
from enum import Enum


class InfoClass(str, Enum):
    MERCHANT_PRIVATE = "merchant_private"
    PLATFORM_OBSERVABLE = "platform_observable"
    EVALUATOR_ONLY = "evaluator_only"


MP, PO, EO = InfoClass.MERCHANT_PRIVATE, InfoClass.PLATFORM_OBSERVABLE, InfoClass.EVALUATOR_ONLY


# ==================================================================================================
# the registry -- the single authority on which class a field belongs to
# ==================================================================================================
FIELD_CLASS: dict[str, InfoClass] = {
    # ---- merchant-private -----------------------------------------------------------------------
    "own_quality": MP,               # q_i. The merchant knows how good its product really is.
    "own_baseline_complaint": MP,    # b_i. Its own historical complaint propensity when honest.
    "own_margin_frac": MP,
    "own_action_history": MP,        # the f it actually chose in each past round
    "own_profit_history": MP,        # its own realised profit per round
    "own_last_action": MP,
    "own_last_profit": MP,
    "own_cumulative_profit": MP,

    # ---- platform-observable --------------------------------------------------------------------
    "round": PO,
    "horizon": PO,
    "policy_kappa": PO,
    "policy_tau": PO,
    "policy_kappa_audit": PO,
    "policy_tau_audit": PO,
    "policy_text": PO,
    "price": PO,                     # listed price, per merchant, public on the listing page
    "reputation": PO,                # displayed reputation score, public
    "own_sales_volume": PO,          # the seller dashboard shows the seller its own sales
    "own_complaints": PO,
    "own_complaint_rate": PO,
    "own_refunds": PO,
    "own_audit_flags": PO,
    "own_listings_audited": PO,
    "own_transactions_sampled": PO,
    "own_penalty_applied": PO,
    "own_signal_history": PO,        # the merchant's own past dashboard rows
    "rival_prices": PO,
    "rival_reputations": PO,
    "category_traffic_index": PO,    # Q, published as a category health index
    "category_mean_complaint_rate": PO,
    "category_mean_reputation": PO,
    "n_merchants": PO,
    "warning_flag": PO,              # A3: derived from own signals only
    "warning_text": PO,
    "compliance_hint": PO,
    "policy_explanation": PO,
    "retry_offered": PO,
    "retry_reason": PO,

    # ---- evaluator-only -------------------------------------------------------------------------
    "true_f": EO,                    # THE hidden action
    "true_f_all": EO,
    "rival_true_f": EO,
    "rival_quality": EO,
    "rival_baseline_complaint": EO,
    "theta_true": EO,                # latent complaint probability b + cs f
    "psi_true": EO,                  # latent audit-flag probability
    "displayed_payoff_table": EO,    # the object the oracle interfaces used to hand over
    "displayed_argmax": EO,
    "displayed_regret": EO,
    "stationary_rbar_table": EO,
    "stationary_profit_table": EO,
    "exploitability": EO,
    "best_response": EO,
    "nash_profiles": EO,
    "G_FB": EO,
    "G_SB_P": EO,
    "G_SB_uniform": EO,
    "market_share": EO,              # realised share; a real platform could compute it, but it is
                                     # a sufficient statistic for the rivals' actions here, so it is
                                     # withheld from merchants and kept for the evaluator only
    "realised_gmv": EO,
    "seed": EO,
}


def field_class(name: str) -> InfoClass:
    try:
        return FIELD_CLASS[name]
    except KeyError:
        raise KeyError(
            f"field {name!r} is not registered in ha_infoclass.FIELD_CLASS. Every field that reaches "
            "a prompt, a record or a log must be classified merchant_private / platform_observable / "
            "evaluator_only. Refusing to guess: guessing is how an oracle gets back in.") from None


def check_registry_covers(names) -> list:
    """Return the unregistered field names. The offline tests require this to be empty."""
    return sorted({n for n in names if n not in FIELD_CLASS})


# ==================================================================================================
# views
# ==================================================================================================
@dataclass
class MerchantView:
    """Everything merchant i is allowed to condition on, and a proof of which class each field came
    from. Construction is the enforcement point: a field lands here only through `put`."""

    merchant_index: int
    arm: str
    private: dict = field(default_factory=dict)
    observable: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)

    def put(self, name: str, value, owner: int | None = None):
        cls = field_class(name)
        if cls is EO:
            raise LeakageError(
                f"attempted to place evaluator-only field {name!r} into the view of merchant "
                f"{self.merchant_index}. This is the exact failure the package exists to prevent.")
        if cls is MP:
            if owner is None:
                owner = self.merchant_index
            if owner != self.merchant_index:
                raise LeakageError(
                    f"merchant-private field {name!r} is owned by merchant {owner} but was placed "
                    f"into the view of merchant {self.merchant_index}")
            self.private[name] = value
        else:
            self.observable[name] = value
        self.provenance[name] = cls.value
        return self

    def as_dict(self) -> dict:
        return {"merchant_index": self.merchant_index, "arm": self.arm,
                "merchant_private": self.private, "platform_observable": self.observable,
                "provenance": self.provenance}

    def all_items(self):
        yield from self.private.items()
        yield from self.observable.items()


class LeakageError(RuntimeError):
    """Raised when a value crosses an information boundary. Always fatal: a run that leaked is not a
    hidden-action run, and continuing would produce a number that looks like evidence and is not."""


# ==================================================================================================
# the string tripwire
# ==================================================================================================
# Renderings coarser than two decimals are deliberately NOT treated as evidence. A quantity on the
# 0.05 action grid rendered at one decimal is "0.4", which collides with half the constants in any
# marketplace prompt; counting that as a leak would bury the real hits under noise and the scan would
# stop being read. Two decimals or finer is the point at which a match identifies a value rather than
# a neighbourhood. Coarser laundering -- "your rival is aggressive" -- is not a string problem at all,
# and is what `invariance_probe` exists to catch.
_NUM_FORMATS = ("{:.2f}", "{:.3f}", "{:.4f}", "{:g}")
_PCT_FORMATS = ("{:.0f}", "{:.1f}", "{:.2f}")


def _renderings(v, include_percent: bool = False) -> set:
    """Ways a float could identifiably surface in text."""
    out = set()
    try:
        x = float(v)
    except (TypeError, ValueError):
        s = str(v).strip()
        return {s} if len(s) >= 3 else set()
    if x != x:                       # NaN
        return out
    for fmt in _NUM_FORMATS:
        try:
            out.add(fmt.format(x))
        except (ValueError, TypeError):
            pass
    if include_percent:
        for fmt in _PCT_FORMATS:
            out.add(fmt.format(x * 100.0))
    return {s for s in out if len(s.replace("-", "").replace(".", "")) > 0}


def _flatten(obj, prefix="") -> list:
    """(path, scalar) pairs for an arbitrarily nested record."""
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out += _flatten(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out += _flatten(v, f"{prefix}[{i}]")
    else:
        out.append((prefix, obj))
    return out


def scan_text_for_secrets(text: str, secrets: dict, allowed_values=None) -> dict:
    """Look for evaluator-only or foreign-private values rendered into `text`.

    Honest about its own weakness. A forbidden value that happens to equal a value the merchant is
    legitimately shown -- a fabrication rate of 0.50 in a market where some price is also 0.50 --
    cannot be distinguished by string matching from a real leak. Those cases are returned separately
    as `coincident`, not silently dropped and not counted as hits, and the invariance probe below is
    what actually decides them.
    """
    # Percent renderings are only plausible if the text speaks in percent at all. These prompts state
    # every quantity as a fraction, so the check is normally inert -- but it is written as a condition
    # on the text rather than assumed away, so it starts working the day a prompt does use "%".
    pct = "%" in text
    allowed = set()
    for v in (allowed_values or []):
        allowed |= _renderings(v, include_percent=pct)
    hits, coincident = [], []
    for path, val in _flatten(secrets):
        if isinstance(val, bool) or val is None:
            continue
        for r in _renderings(val, include_percent=pct):
            if len(r.replace("-", "").replace(".", "").lstrip("0")) == 0:
                continue          # 0, 0.0, 0.00 ... too common to be evidence of anything
            if not re.search(r"(?<![\d.])" + re.escape(r) + r"(?![\d])", text):
                continue
            (coincident if r in allowed else hits).append(
                {"path": path, "value": val, "rendering": r})
            break
    return {"leaks": hits, "coincident": coincident, "clean": not hits}


# ==================================================================================================
# the invariance probe -- the test that actually proves the point
# ==================================================================================================
def prompt_hash(msgs) -> str:
    return hashlib.sha256(
        json.dumps(msgs, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def invariance_probe(build_fn, base_state: dict, perturbations: list) -> dict:
    """Require the prompt to be blind to the hidden state.

    `build_fn(state) -> messages`. Each perturbation is a dict merged onto `base_state` that changes
    ONLY evaluator-only quantities and leaves every published observable exactly as it was. If the
    prompt hash moves, the builder is reading something it must not read, and the probe names the
    perturbation that exposed it.

    The converse is not claimed. Invariance across the perturbations tested is evidence, not proof,
    that no functional dependence exists; the set of perturbations is therefore reported alongside
    the verdict so a reader can judge its coverage instead of trusting the word "invariant".
    """
    base = prompt_hash(build_fn(base_state))
    rows, bad = [], []
    for k, pert in enumerate(perturbations):
        st = copy.deepcopy(base_state)
        st.update(copy.deepcopy(pert))
        h = prompt_hash(build_fn(st))
        ok = (h == base)
        rows.append({"index": k, "perturbation": pert, "hash": h[:16], "invariant": ok})
        if not ok:
            bad.append(k)
    return {"base_hash": base[:16], "n_perturbations": len(perturbations),
            "n_violations": len(bad), "violations": bad, "rows": rows,
            "pass": not bad}


# ==================================================================================================
# record assembly -- the only sanctioned way to write a raw log line
# ==================================================================================================
def build_record(view: MerchantView, evaluator_only: dict, decision: dict,
                 msgs, meta: dict) -> dict:
    """Assemble one raw record with the three classes kept in labelled, disjoint blocks.

    Evaluator-only quantities ARE written to the raw log -- the analysis needs them -- but they are
    written beside the prompt, never inside it, and the record carries the prompt hash so an auditor
    can recompute what the model actually saw without trusting this function's word for it.
    """
    unregistered = check_registry_covers(list(evaluator_only.keys()))
    if unregistered:
        raise KeyError(f"unregistered evaluator-only fields: {unregistered}")
    misclassified = [k for k in evaluator_only if field_class(k) is not EO]
    if misclassified:
        raise LeakageError(
            f"fields {misclassified} were passed as evaluator_only but are registered otherwise; "
            "the registry is the authority, so this is a bug in the caller, not in the registry")
    return {
        **meta,
        "prompt_hash": prompt_hash(msgs),
        "prompt_messages": msgs,
        "merchant_private": view.private,
        "platform_observable": view.observable,
        "evaluator_only": evaluator_only,
        "provenance": view.provenance,
        "decision": decision,
    }


def registry_report() -> dict:
    by = {c.value: sorted(k for k, v in FIELD_CLASS.items() if v is c) for c in InfoClass}
    return {"n_fields": len(FIELD_CLASS), "by_class": by,
            "counts": {k: len(v) for k, v in by.items()}}
