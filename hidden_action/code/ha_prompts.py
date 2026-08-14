"""ha_prompts.py -- the frozen prompt surface for the hidden-action experiment.

The prompts live here, as data, so the preregistration can quote a file rather than describe an
intention, and so the runner is forbidden to compose text of its own. Anything sent that did not come
out of this module is by construction unpreregistered.

THE FOUR ARMS
-------------
  A0_oracle    The upper-bound control. The merchant is handed the full counterfactual payoff table
               over all 21 candidate fabrication levels, holding rivals fixed -- exactly the object
               the Balance-6..9 U/H/R/G interfaces handed over. Building that table needs every
               merchant's private type and the rivals' true current actions, so NO PLATFORM COULD
               EVER BUILD IT. A0 is retained only to measure what is lost by not having it, and it
               is the one arm that deliberately breaks the information firewall. The breach is
               declared, logged and asserted (gate G-H3 requires A0 to FAIL the leakage probe; a
               leak detector that never fires on an arm designed to leak is not a detector).

  A1_policy    Published rules and the current published state. The merchant knows the mechanics of
               the market, its own type, the committed (kappa, tau), its own current dashboard, the
               rivals' public prices and reputations, and the lagged category traffic index. It gets
               no history and no help.

  A2_history   A1 plus the merchant's own observable history: its past actions, past realised
               profits, and its past dashboard rows. This is the arm that asks whether a merchant can
               learn a policy response from its own experience alone.

  A3_assist    A2 plus platform assistance computed from PLATFORM-OBSERVABLE FIELDS ONLY: a risk
               warning when the merchant's own observed complaint or audit rate crossed the published
               threshold, an arithmetic explanation of what the published rule did to it, a
               compliance hint, and -- when its own observed rate has been over threshold twice in a
               row -- one offered reconsideration turn. The platform never overrides the choice, and
               the assistance is never a verifier: it reports what the rule DID, never what the
               merchant SHOULD have picked, and it never evaluates the chosen action against a
               payoff table. That distinction is the whole difference between A3 and the withdrawn
               R/G arms, and gate G-H7 enforces it lexically.

WHAT EVERY ARM SHARES
---------------------
The rules block is byte-identical across all four arms (gate G-H5). The arms differ by appended
blocks and by nothing else, so a difference between arms is a difference in information, not a
difference in wording.

TWO THINGS DELIBERATELY VISIBLE, RECORDED HERE SO THEY ARE NOT DISCOVERED LATER AS FLAWS
---------------------------------------------------------------------------------------
  1. The published category traffic index is a deterministic, invertible function of the market-
     average action: Q = Q0 * exp(-lam * F). A merchant that inverts it recovers the LAGGED market
     average F to within the rounding of the published index. This is intended. Without it -- and
     without the published rival reputations -- the four merchants would be solving four independent
     decision problems, "equilibrium" would be an empty word, and no claim about strategic
     interaction could be identified from the data. It reveals an average, never an individual
     action, and no policy in this package conditions on it.
  2. The merchant sees its own baseline complaint propensity b and its own quality q. Both are
     MERCHANT-PRIVATE, which is exactly why the platform cannot reconstruct the payoff table: the
     table needs b and q for all four merchants, and the platform has none of them.

No API, no network. Importable and fully testable offline.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import ha_infoclass as IC                                        # noqa: E402
import ha_model as M                                             # noqa: E402

ARMS = ("A0_oracle", "A1_policy", "A2_history", "A3_assist")
NON_ORACLE_ARMS = ("A1_policy", "A2_history", "A3_assist")
NIDX = 20                     # action index 0..20, f = index / 20  (== M.Config.Nf - 1)
HISTORY_WINDOW = 8            # rounds of own history shown in A2/A3; frozen before any run

# ---------------------------------------------------------------------------------------------
# the rules block -- byte-identical across arms
# ---------------------------------------------------------------------------------------------
SYSTEM_RULES = (
    "You are one of {m} sellers in an ongoing online marketplace that continues for an unknown number "
    "of further rounds (no final round is announced). Each round you privately choose an exaggeration "
    "level f in [0,1] for your listing (0 = fully honest, 1 = maximum exaggeration). Your choice of f "
    "is never observed by the platform or by the other sellers. The exact rules each round are:\n"
    "- Advertised appeal: a = q + (1 - q) * f, where q is your true quality.\n"
    "- Buyer choice is multinomial logit: a buyer's utility for seller k is u_k = {alpha}*a_k + "
    "{beta}*r_k - {gamma}*p_k, where r_k is that seller's displayed reputation in [0,1] and p_k is the "
    "price. There is an outside option with utility weight w0 = {w0}. Your market share is "
    "s = exp(u) / (exp(w0) + sum over sellers k of exp(u_k)). You do not observe rivals' true quality "
    "or their exaggeration, but you know this rule.\n"
    "- Category traffic: Q = {Q0} * exp(-{lam} * F), where F is the average exaggeration across all "
    "{m} sellers in the current round. Your own exaggeration therefore lowers total category traffic "
    "for everyone, including you, by a small amount.\n"
    "- Your sales this round: y = Q * s. Your profit this round: profit = {margin} * p * y.\n"
    "- Complaint monitoring: each round the platform samples a fixed panel of n = {n} recent "
    "transactions for your listing, separate from your current sales. Each sampled transaction is "
    "adverse with probability theta = min(1, b + {cs}*f), where b is your own baseline adverse-feedback "
    "rate. The adverse count is D ~ Binomial({n}, theta) and the observed complaint rate is d = D / {n}.\n"
    "- Refunds: each adverse transaction turns into a refund independently with probability {phi}.\n"
    "- Content audit: each round the platform also samples {na} of your listings for review. Each "
    "sampled listing is flagged with probability psi = min(1, {psi0} + {psi1}*f). The flag count is "
    "C ~ Binomial({na}, psi) and the observed audit rate is c = C / {na}.\n"
    "- Published penalty rule: P = kappa * max(0, d - tau) + kappa_a * max(0, c - tau_a). The platform "
    "publishes (kappa, tau, kappa_a, tau_a) in advance and applies the rule to the observed counts "
    "only. It never observes your f, your q or your b.\n"
    "- Reputation update: r_next = clip(r + {eta}*(1 - r) - P, 0, 1).\n"
    "You choose f each round to maximize your own total profit across the ongoing market."
)

JSON_INSTR = ('Respond ONLY as compact JSON: {"exaggeration_index": <integer 0-20>, '
              '"reasoning": "<one short sentence>"} where your chosen exaggeration level is '
              "f = exaggeration_index / 20.")

SCHEMA_REPAIR_MSG = (
    "Your previous message could not be parsed. Reply with nothing except a single JSON object of "
    'exactly this form: {"exaggeration_index": <integer 0-20>, "reasoning": "<one short sentence>"}. '
    "No prose, no code fences, no explanation outside the JSON.")

# ---- A0 only ---------------------------------------------------------------------------------
A0_TABLE_LEAD = (" Holding the other sellers at their most recent choices, your projected per-round "
                 "profit for each candidate exaggeration level is: ")
A0_DECLASSIFIED_FIELDS = ("displayed_payoff_table", "rival_true_f", "rival_quality",
                          "rival_baseline_complaint")

# ---- A3 only ---------------------------------------------------------------------------------
A3_HEADER = " The platform provides the following notices, computed from your published account record."
A3_POLICY_EXPLANATION = (
    " Policy arithmetic on your own last observed record: your complaint rate was d = {d:.3f} against "
    "the published threshold tau = {tau:g}, giving kappa * max(0, d - tau) = {pd:.4f}.")
A3_AUDIT_EXPLANATION = (
    " Your audit rate was c = {c:.3f} against tau_a = {ta:g}, giving "
    "kappa_a * max(0, c - tau_a) = {pa:.4f}.")
A3_TOTAL = " The total penalty applied to your reputation was P = {P:.4f}."
A3_WARNING = (" Notice: your observed complaint rate has been above the published threshold in "
              "{k} of your last {n} recorded rounds.")
# Counted over LISTINGS, because that is the unit the audit samples. The earlier wording counted
# flags against rounds and could print "3 of your last 2", which is not a sentence a platform would
# ever send and would have taught the model that the notices are unreliable.
A3_AUDIT_WARNING = (" Notice: {k} of the {n} listings sampled from your account over your last {w} "
                    "recorded rounds were flagged for review.")
A3_HINT = (" Reminder: the penalty is applied to observed rates only. Listings whose observed "
           "complaint and audit rates stay at or below the published thresholds incur no penalty.")
A3_RETRY = (" Your account has been over the published complaint threshold in each of the last two "
            "recorded rounds. You may submit your decision for this round again. The rules, the "
            "published thresholds and your record are unchanged.")

# Gate G-H7. Frozen BEFORE the assistance text was written, so it is a test rather than a description
# of what happened to be typed. Platform assistance may state what the rule DID. It may not name an
# optimum, a shortfall against one, or a preferred action.
FORBIDDEN_ASSIST_LEXICON = ("highest", "maximis", "maximiz", "best", "optimal", "shortfall",
                            "tolerance", "you should", "we recommend", "recommended", "better off",
                            "instead choose", "argmax", "profit-maximis", "profit-maximiz")

# Gate G-H4. No prompt may disclose the horizon. A model told the run ends at round 80 is playing a
# finite game with a known last round, which is a different game with different equilibria.
# The tokens are DISCLOSURES, not the word "round": the rules block must be free to say that no final
# round is announced, and the gate separately requires that denial to be present.
HORIZON_TOKENS = ("of 80", "out of 80", "80 rounds", "80 more", "/80", "ends at round",
                  "rounds remain", "rounds left", "the final round is", "the last round is",
                  "this is the final", "this is the last", "total number of rounds")
HORIZON_DENIAL = "no final round is announced"


# ==================================================================================================
# helpers
# ==================================================================================================
def idx_to_f(i) -> float:
    return round(float(i) / NIDX, 4)


def f_to_idx(f) -> int:
    return int(round(float(f) * NIDX))


def rules_block(cfg: M.Config) -> str:
    return SYSTEM_RULES.format(
        m=cfg.m, alpha=f"{cfg.alpha:g}", beta=f"{cfg.beta:g}", gamma=f"{cfg.gamma:g}",
        w0=f"{cfg.w0:.3f}", Q0=f"{cfg.Q0:g}", lam=f"{cfg.lam:g}", margin=f"{cfg.margin_frac:g}",
        n=cfg.N_obs, cs=f"{cfg.cs:g}", phi=f"{cfg.phi_refund:g}", na=cfg.N_audit,
        psi0=f"{cfg.psi0:g}", psi1=f"{cfg.psi1:g}", eta=f"{cfg.eta_r:g}")


def _round_row(row: dict) -> dict:
    """One dashboard row as the merchant is shown it. Every entry is platform-observable."""
    return {"round": int(row["round"]),
            "your_exaggeration_f": round(float(row["own_action"]), 3),
            "your_sales_y": round(float(row["own_sales_volume"]), 4),
            "your_profit": round(float(row["own_profit"]), 4),
            "complaints_D": int(row["own_complaints"]),
            "complaint_rate_d": round(float(row["own_complaint_rate"]), 3),
            "refunds": int(row["own_refunds"]),
            "audit_flags_C": int(row["own_audit_flags"]),
            "audit_rate_c": round(float(row["own_audit_flags"]) / max(1, int(row["own_listings_audited"])), 3),
            "penalty_P": round(float(row["own_penalty_applied"]), 4),
            "reputation_before": round(float(row["reputation_before"]), 3),
            "reputation_after": round(float(row["reputation_after"]), 3)}


# ==================================================================================================
# assistance -- A3 only, and computable from platform-observable fields alone
# ==================================================================================================
def assistance_block(state: dict, cfg: M.Config) -> tuple:
    """(text, notices). Every input is a published account field; none is a latent quantity.

    Returns the notices separately so the runner can log WHICH notice fired without re-parsing text.
    """
    pol = state["policy"]
    kappa, tau = float(pol[0]), float(pol[1])
    kappa_a = float(pol[2]) if len(pol) > 2 else 0.0
    tau_a = float(pol[3]) if len(pol) > 3 else 0.0
    hist = state.get("history") or []
    notices = {"warning": False, "audit_warning": False, "explanation": False, "hint": True,
               "retry_offered": False}
    parts = [A3_HEADER]

    # An audit channel with kappa_a = 0 carries no penalty, and the merchant already knows its own f
    # exactly, so an audit notice under kappa_a = 0 would tell it nothing it did not know about a
    # consequence that does not exist. Assistance that is noise is not neutral: it teaches the model
    # that the notices are not worth reading. So the audit sentences appear only when the audit term
    # is actually live.
    audit_live = kappa_a > 0.0
    if hist:
        last = hist[-1]
        d = float(last["own_complaint_rate"])
        c = float(last["own_audit_flags"]) / max(1, int(last["own_listings_audited"]))
        pd = kappa * max(0.0, d - tau)
        pa = kappa_a * max(0.0, c - tau_a)
        parts.append(A3_POLICY_EXPLANATION.format(d=d, tau=tau, pd=pd))
        if audit_live:
            parts.append(A3_AUDIT_EXPLANATION.format(c=c, ta=tau_a, pa=pa))
        parts.append(A3_TOTAL.format(P=float(last["own_penalty_applied"])))
        notices["explanation"] = True

        win = hist[-HISTORY_WINDOW:]
        over = sum(1 for h in win if float(h["own_complaint_rate"]) > tau)
        if over:
            parts.append(A3_WARNING.format(k=over, n=len(win)))
            notices["warning"] = True
        flagged = sum(int(h["own_audit_flags"]) for h in win)
        sampled = sum(int(h["own_listings_audited"]) for h in win)
        if audit_live and flagged:
            parts.append(A3_AUDIT_WARNING.format(k=flagged, n=sampled, w=len(win)))
            notices["audit_warning"] = True

    parts.append(A3_HINT)
    return "".join(parts), notices


def retry_available(state: dict) -> bool:
    """One reconsideration turn, triggered by the merchant's own PUBLISHED record only.

    This is an ECONOMIC retry: a second decision opportunity offered by the platform. It is counted
    separately from a transport retry (an HTTP failure) and from a schema repair (an unparseable
    reply), because conflating the three is how a technical artefact becomes an economic finding.
    """
    pol = state["policy"]
    tau = float(pol[1])
    hist = state.get("history") or []
    if len(hist) < 2:
        return False
    return all(float(h["own_complaint_rate"]) > tau for h in hist[-2:])


# ==================================================================================================
# the builder
# ==================================================================================================
def build_view(cfg: M.Config, arm: str, state: dict) -> IC.MerchantView:
    """Assemble what merchant j is allowed to see, through the firewall.

    Every `put` is checked against the registry, so a field that is not classified, or that belongs to
    another merchant, raises here rather than reaching a model.
    """
    j = int(state["j"])
    view = IC.MerchantView(merchant_index=j, arm=arm)
    pol = state["policy"]
    view.put("round", int(state["t"]))
    view.put("n_merchants", int(cfg.m))
    view.put("policy_kappa", float(pol[0]))
    view.put("policy_tau", float(pol[1]))
    view.put("policy_kappa_audit", float(pol[2]) if len(pol) > 2 else 0.0)
    view.put("policy_tau_audit", float(pol[3]) if len(pol) > 3 else 0.0)
    view.put("price", float(state["p"][j]))
    view.put("reputation", float(state["r"][j]))
    view.put("rival_prices", [float(v) for k, v in enumerate(state["p"]) if k != j])
    view.put("rival_reputations", [float(v) for k, v in enumerate(state["r"]) if k != j])
    view.put("category_traffic_index", round(float(state["Q_last"]), 3))
    view.put("own_quality", float(state["q"][j]), owner=j)
    view.put("own_baseline_complaint", float(state["b"][j]), owner=j)
    view.put("own_margin_frac", float(cfg.margin_frac), owner=j)
    hist = state.get("history") or []
    if arm in ("A2_history", "A3_assist") and hist:
        rows = [_round_row(h) for h in hist[-HISTORY_WINDOW:]]
        view.put("own_signal_history", rows)
        view.put("own_action_history", [r["your_exaggeration_f"] for r in rows], owner=j)
        view.put("own_profit_history", [r["your_profit"] for r in rows], owner=j)
    elif hist:
        view.put("own_signal_history", [_round_row(hist[-1])])
        view.put("own_last_action", _round_row(hist[-1])["your_exaggeration_f"], owner=j)
        view.put("own_last_profit", _round_row(hist[-1])["your_profit"], owner=j)
    return view


def build_messages(cfg: M.Config, arm: str, state: dict) -> tuple:
    """(messages, meta). `messages` is the OpenAI chat list actually sent.

    A0 is the only path that consults `state['payoff_table']`, and it records the declassification in
    `meta['declassified']` so the raw log shows on its face which arm broke the firewall.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
    view = build_view(cfg, arm, state)
    system = rules_block(cfg)
    meta = {"arm": arm, "declassified": [], "notices": None, "retry_offered": False}

    if arm == "A0_oracle":
        v = np.asarray(state["payoff_table"], dtype=float)
        rows = "; ".join(f"[{i}] f={idx_to_f(i):.2f} -> profit={v[i]:.4f}" for i in range(NIDX + 1))
        system = system + A0_TABLE_LEAD + rows + "."
        meta["declassified"] = list(A0_DECLASSIFIED_FIELDS)
    elif arm == "A3_assist":
        txt, notices = assistance_block(state, cfg)
        system = system + txt
        meta["notices"] = notices
        if retry_available(state):
            meta["retry_offered"] = True

    system = system + "\n" + JSON_INSTR
    user = json.dumps(user_payload(view), ensure_ascii=False)
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return msgs, {**meta, "view": view}


def user_payload(view: IC.MerchantView) -> dict:
    """The user turn: the view, rendered. Nothing is added here that did not pass the firewall."""
    o, pv = view.observable, view.private
    u = {"round": o["round"],
         "your_true_quality_q": round(pv["own_quality"], 3),
         "your_price_p": round(o["price"], 3),
         "your_baseline_adverse_rate_b": round(pv["own_baseline_complaint"], 3),
         "your_reputation_r": round(o["reputation"], 3),
         "rival_prices": [round(v, 3) for v in o["rival_prices"]],
         "rival_reputations": [round(v, 3) for v in o["rival_reputations"]],
         "category_traffic_index": o["category_traffic_index"],
         "published_penalty_kappa": o["policy_kappa"],
         "published_penalty_tau": o["policy_tau"],
         "published_audit_penalty_kappa_a": o["policy_kappa_audit"],
         "published_audit_penalty_tau_a": o["policy_tau_audit"]}
    if "own_signal_history" in o:
        u["your_recorded_rounds"] = o["own_signal_history"]
    return u


def retry_message() -> str:
    return A3_RETRY.strip()


# ==================================================================================================
# parsing -- the only sanctioned reader of a model reply
# ==================================================================================================
class ParseFailure(RuntimeError):
    """Unparseable after every schema repair. The run stops; an action is never invented."""


def parse_reply(text: str) -> dict:
    """Extract {'index', 'f', 'reasoning'} or raise. Never guesses an index from prose."""
    if text is None:
        raise ParseFailure("empty reply")
    s = str(text).strip()
    if s.startswith("```"):
        s = s.strip("`")
        s = s[s.index("\n") + 1:] if "\n" in s else s
    a, b = s.find("{"), s.rfind("}")
    if a < 0 or b <= a:
        raise ParseFailure(f"no JSON object in reply: {s[:200]!r}")
    try:
        obj = json.loads(s[a:b + 1])
    except json.JSONDecodeError as exc:
        raise ParseFailure(f"invalid JSON: {exc}") from None
    if "exaggeration_index" not in obj:
        raise ParseFailure(f"missing exaggeration_index: {sorted(obj)[:8]}")
    try:
        i = int(obj["exaggeration_index"])
    except (TypeError, ValueError):
        raise ParseFailure(f"exaggeration_index is not an integer: {obj['exaggeration_index']!r}") from None
    if not 0 <= i <= NIDX:
        raise ParseFailure(f"exaggeration_index {i} out of range 0..{NIDX}")
    return {"index": i, "f": idx_to_f(i), "reasoning": str(obj.get("reasoning", ""))[:400]}


# ==================================================================================================
# gates
# ==================================================================================================
DEMO_POLICY = (4.0, 0.20, 0.0, 0.0)
DEMO_R0 = (0.52, 0.62, 0.48, 0.71)
# (own f, own complaint count D, own audit flags C) for the two recorded rounds. Counts are FIXED
# rather than sampled so the fixture is reproducible byte-for-byte in any process.
DEMO_ROUNDS = ((0.35, 9, 1), (0.40, 11, 2))
DEMO_RIVAL_F = (0.30, 0.45, 0.25)


def _demo_state(cfg: M.Config, seed: int = 999999, t: int = 5, j: int = 0,
                hidden=None, with_history: bool = True) -> dict:
    """A fixed state used by every gate and by the preregistered prompt hashes.

    The recorded rounds are COMPUTED from the same arithmetic the rules block states -- logit demand,
    traffic, the published penalty, the reputation recursion -- with the random counts pinned. A
    fixture that violated the rules printed above it would be the first thing a careful reader tried
    to check and the first thing that would make them distrust the rest, so gate G-H11 recomputes it.
    """
    mkt = M.draw_market(cfg, seed)
    hidden = hidden or {}
    kappa, tau = DEMO_POLICY[0], DEMO_POLICY[1]
    r = np.array(DEMO_R0, dtype=float)
    hist, Q_last = [], cfg.Q0
    for k, (fj, D, C) in enumerate(DEMO_ROUNDS):
        f = np.array([fj, *DEMO_RIVAL_F], dtype=float)
        u = cfg.alpha * (mkt.q + (1 - mkt.q) * f) + cfg.beta * r - cfg.gamma * mkt.p
        den = math.exp(cfg.w0) + float(np.exp(u).sum())
        s = np.exp(u) / den
        Q = cfg.Q0 * math.exp(-cfg.lam * float(f.mean()))
        y = float(Q * s[j])
        d = D / cfg.N_obs
        pen = M.penalty_from_signal(kappa, tau, d, DEMO_POLICY[2], DEMO_POLICY[3], C / cfg.N_audit)
        r_before = float(r[j])
        r_after = M.reputation_update(cfg, r_before, pen)
        hist.append({"round": t - len(DEMO_ROUNDS) + k, "own_action": fj,
                     "own_sales_volume": round(y, 6),
                     "own_profit": round(cfg.margin_frac * float(mkt.p[j]) * y, 6),
                     "own_complaints": D, "own_complaint_rate": d,
                     "own_refunds": int(D // 2), "own_audit_flags": C,
                     "own_listings_audited": cfg.N_audit,
                     "own_penalty_applied": round(pen, 6),
                     "reputation_before": round(r_before, 6),
                     "reputation_after": round(r_after, 6)})
        r = r.copy()
        r[j] = r_after
        Q_last = Q
    st = {"t": t, "j": j, "q": mkt.q, "p": mkt.p, "b": mkt.b,
          "r": r if with_history else np.array(DEMO_R0, dtype=float),
          "Q_last": round(Q_last, 3), "policy": DEMO_POLICY, "seed": seed,
          "payoff_table": hidden.get("payoff_table", np.linspace(0.10, 0.22, NIDX + 1)),
          "rival_true_f": hidden.get("rival_true_f", list(DEMO_RIVAL_F)),
          "true_f": hidden.get("true_f", 0.35),
          "history": hist if with_history else []}
    return st


def _hidden_perturbations(cfg: M.Config) -> list:
    """Changes to EVALUATOR-ONLY state only. Every published observable is held fixed."""
    return [
        {"true_f": 0.95},
        {"true_f": 0.0},
        {"rival_true_f": [0.0, 0.0, 0.0]},
        {"rival_true_f": [1.0, 1.0, 1.0]},
        {"payoff_table": list(np.linspace(0.22, 0.10, NIDX + 1))},      # argmax moved to index 0
        {"payoff_table": list(np.full(NIDX + 1, 0.5))},                  # table flattened
        {"payoff_table": list(np.linspace(0.10, 0.22, NIDX + 1)), "true_f": 0.5,
         "rival_true_f": [0.9, 0.1, 0.9]},
    ]


def gate_registry(cfg: M.Config) -> dict:
    """Every field any arm places in a view is registered and correctly classified."""
    bad = []
    for arm in ARMS:
        v = build_view(cfg, arm, _demo_state(cfg))
        for name in v.provenance:
            if name not in IC.FIELD_CLASS:
                bad.append((arm, name, "unregistered"))
            elif IC.FIELD_CLASS[name] is IC.EO:
                bad.append((arm, name, "evaluator_only in a view"))
    return {"gate": "G-H0_registry", "pass": not bad, "violations": bad}


def gate_shared_rules(cfg: M.Config) -> dict:
    """The rules block is byte-identical across arms; arms differ by appended blocks only."""
    st = _demo_state(cfg)
    base = rules_block(cfg)
    rows, ok = {}, True
    for arm in ARMS:
        msgs, _ = build_messages(cfg, arm, st)
        sysmsg = msgs[0]["content"]
        starts = sysmsg.startswith(base)
        rows[arm] = {"starts_with_shared_rules": starts,
                     "appended_chars": len(sysmsg) - len(base),
                     "system_hash": hashlib.sha256(sysmsg.encode()).hexdigest()[:16]}
        ok &= starts
    ok &= rows["A1_policy"]["appended_chars"] == rows["A2_history"]["appended_chars"]
    return {"gate": "G-H5_shared_rules", "pass": bool(ok), "arms": rows,
            "note": "A1 and A2 must share the SAME system text: they differ only in the user turn, "
                    "which is where history lives. A0 and A3 append their declared blocks."}


def gate_no_horizon(cfg: M.Config) -> dict:
    bad, missing_denial = [], []
    for arm in ARMS:
        for t in (1, 2, 40, 79, 80):
            for wh in (True, False):
                msgs, _ = build_messages(cfg, arm, _demo_state(cfg, t=t, with_history=wh))
                blob = (msgs[0]["content"] + " " + msgs[1]["content"]).lower()
                for tok in HORIZON_TOKENS:
                    if tok in blob:
                        bad.append((arm, t, tok))
                if HORIZON_DENIAL not in blob:
                    missing_denial.append((arm, t))
    return {"gate": "G-H4_no_horizon", "pass": not bad and not missing_denial,
            "violations": bad, "missing_denial": missing_denial,
            "note": "two requirements, not one: no prompt may state the horizon, and every prompt "
                    "must carry the explicit statement that no final round is announced."}


def gate_assist_lexicon(cfg: M.Config) -> dict:
    """A3's assistance may say what the rule DID; it may never name an optimum or a preference."""
    hits = []
    for pol in ((0.5, 0.20, 0.0, 0.0), (4.0, 0.30, 0.1, 0.2), (0.0, 0.02, 0.0, 0.0)):
        st = _demo_state(cfg)
        st["policy"] = pol
        txt, _ = assistance_block(st, cfg)
        blob = (txt + " " + retry_message()).lower()
        hits += [(pol, w) for w in FORBIDDEN_ASSIST_LEXICON if w in blob]
    return {"gate": "G-H7_assist_lexicon", "pass": not hits, "violations": hits,
            "lexicon": list(FORBIDDEN_ASSIST_LEXICON)}


def gate_leakage(cfg: M.Config) -> dict:
    """String tripwire plus functional invariance, per arm.

    A1/A2/A3 must pass both. A0 must FAIL the invariance probe: it is the positive control that shows
    the probe can detect an oracle at all.
    """
    out, ok = {}, True
    for arm in ARMS:
        st = _demo_state(cfg)
        msgs, meta = build_messages(cfg, arm, st)
        text = msgs[0]["content"] + "\n" + msgs[1]["content"]
        secrets = {"true_f": st["true_f"], "rival_true_f": st["rival_true_f"],
                   "rival_quality": [float(v) for k, v in enumerate(st["q"]) if k != st["j"]],
                   "rival_baseline_complaint": [float(v) for k, v in enumerate(st["b"])
                                                if k != st["j"]],
                   "payoff_table": list(np.asarray(st["payoff_table"], dtype=float))}
        # Everything the merchant is legitimately shown, flattened to scalars including the values
        # nested inside dashboard rows, PLUS the published model constants that the rules block
        # states out loud (alpha, beta, cs, psi1, ...). A forbidden value that equals one of these is
        # `coincident`, not a hit: string matching genuinely cannot tell the two apart, and a scan
        # that reported psi1 = 0.25 as a leak every time some latent quantity also equalled 0.25
        # would be noise that nobody reads. The invariance probe is what covers those cases.
        allowed = [v for _, v in IC._flatten(meta["view"].as_dict())
                   if isinstance(v, (int, float)) and not isinstance(v, bool)]
        allowed += [v for _, v in IC._flatten(M.spec_dict(cfg))
                    if isinstance(v, (int, float)) and not isinstance(v, bool)]
        scan = IC.scan_text_for_secrets(text, secrets, allowed_values=allowed)

        def build(state, _arm=arm):
            return build_messages(cfg, _arm, state)[0]

        probe = IC.invariance_probe(build, st, _hidden_perturbations(cfg))
        expect_leak = (arm == "A0_oracle")
        arm_ok = (probe["pass"] and scan["clean"]) if not expect_leak else (not probe["pass"])
        out[arm] = {"expect_leak": expect_leak, "string_scan": scan,
                    "invariance": {k: probe[k] for k in
                                   ("base_hash", "n_perturbations", "n_violations", "pass")},
                    "verdict_ok": bool(arm_ok)}
        ok &= arm_ok
    return {"gate": "G-H1_G-H2_G-H3_leakage", "pass": bool(ok),
            "verdicts": {a: {"ok": out[a]["verdict_ok"], "expect_leak": out[a]["expect_leak"],
                             "n_string_hits": len(out[a]["string_scan"]["leaks"]),
                             "n_coincident": len(out[a]["string_scan"]["coincident"]),
                             "invariance_pass": out[a]["invariance"]["pass"]} for a in out},
            "arms": out,
            "note": "A0 is REQUIRED to fail: a leak detector that never fires on an arm built to "
                    "leak has not been shown to work."}


def gate_hash_stability(cfg: M.Config) -> dict:
    """Hashes must be reproducible across processes: fixed inputs, no clock, no unseeded RNG."""
    h = {}
    for arm in ARMS:
        for wh in (True, False):
            msgs, _ = build_messages(cfg, arm, _demo_state(cfg, with_history=wh))
            h[f"{arm}|history={int(wh)}"] = IC.prompt_hash(msgs)
    again = {}
    for arm in ARMS:
        for wh in (True, False):
            msgs, _ = build_messages(cfg, arm, _demo_state(cfg, with_history=wh))
            again[f"{arm}|history={int(wh)}"] = IC.prompt_hash(msgs)
    return {"gate": "G-H6_hash_stability", "pass": h == again,
            "hashes": {k: v[:16] for k, v in sorted(h.items())}}


def gate_parse() -> dict:
    good = ['{"exaggeration_index": 7, "reasoning": "x"}',
            '```json\n{"exaggeration_index": 0, "reasoning": "y"}\n```',
            'sure: {"exaggeration_index": 20, "reasoning": "z"} done']
    bad = ["", "no json here", '{"exaggeration_index": 21}', '{"exaggeration_index": -1}',
           '{"exaggeration_index": "seven"}', '{"reasoning": "no index"}', "{bad json,,}"]
    rows = []
    ok = True
    for s in good:
        try:
            rows.append({"text": s[:40], "parsed": parse_reply(s), "expected": "parse"})
        except ParseFailure as exc:
            rows.append({"text": s[:40], "error": str(exc), "expected": "parse"})
            ok = False
    for s in bad:
        try:
            parse_reply(s)
            rows.append({"text": s[:40], "expected": "reject", "got": "parsed"})
            ok = False
        except ParseFailure:
            rows.append({"text": s[:40], "expected": "reject", "got": "rejected"})
    return {"gate": "G-H8_parse", "pass": ok, "cases": rows}


def gate_retry_trigger(cfg: M.Config) -> dict:
    """The economic retry fires on the merchant's own published record, and on nothing else."""
    st = _demo_state(cfg)
    st["policy"] = (4.0, 0.20, 0.0, 0.0)
    on = retry_available(st)                       # both recorded rates 0.225, 0.275 > 0.20
    st2 = json.loads(json.dumps({k: v for k, v in st.items() if k in ("policy", "history")}))
    st2["policy"] = (4.0, 0.40, 0.0, 0.0)
    off = retry_available(st2)                     # same record, higher published threshold
    st3 = {"policy": (4.0, 0.20, 0.0, 0.0), "history": st["history"][-1:]}
    short = retry_available(st3)                   # only one recorded round
    st4 = {"policy": (4.0, 0.20, 0.0, 0.0), "history": [
        dict(st["history"][0], own_complaint_rate=0.05), st["history"][1]]}
    mixed = retry_available(st4)
    ok = on and not off and not short and not mixed
    return {"gate": "G-H9_retry_trigger", "pass": bool(ok),
            "over_threshold_twice": on, "threshold_raised": off,
            "single_round_history": short, "one_round_below": mixed,
            "note": "an economic retry is a second DECISION opportunity; it is counted separately "
                    "from transport retries and from schema repairs."}


def gate_a0_is_an_oracle(cfg: M.Config) -> dict:
    """A0 must actually depend on unobservable state -- otherwise it is not an upper bound at all."""
    st = _demo_state(cfg)
    h1 = IC.prompt_hash(build_messages(cfg, "A0_oracle", st)[0])
    st2 = dict(st)
    st2["payoff_table"] = list(np.linspace(0.22, 0.10, NIDX + 1))
    h2 = IC.prompt_hash(build_messages(cfg, "A0_oracle", st2)[0])
    return {"gate": "G-H10_a0_depends_on_table", "pass": h1 != h2,
            "note": "the table shown to A0 needs every merchant's private type and the rivals' true "
                    "current actions; no platform can construct it. A0 is a control, not a design."}


def gate_fixture_consistent(cfg: M.Config) -> dict:
    """The demo record must obey the rules the prompt prints above it.

    Recomputed here from the stated recursion rather than trusted from the constructor, because the
    prompt hashes that go into the preregistration are taken over this fixture.
    """
    st = _demo_state(cfg)
    kappa, tau = st["policy"][0], st["policy"][1]
    rows, ok = [], True
    for h in st["history"]:
        pen_expect = kappa * max(0.0, float(h["own_complaint_rate"]) - tau)
        r_expect = M.reputation_update(cfg, float(h["reputation_before"]), pen_expect)
        d_expect = h["own_complaints"] / cfg.N_obs
        profit_expect = cfg.margin_frac * float(st["p"][st["j"]]) * float(h["own_sales_volume"])
        good = (abs(pen_expect - h["own_penalty_applied"]) < 1e-6
                and abs(r_expect - h["reputation_after"]) < 1e-6
                and abs(d_expect - h["own_complaint_rate"]) < 1e-12
                and abs(profit_expect - h["own_profit"]) < 1e-5)
        ok &= good
        rows.append({"round": h["round"], "penalty_recomputed": round(pen_expect, 6),
                     "penalty_recorded": h["own_penalty_applied"],
                     "reputation_recomputed": round(r_expect, 6),
                     "reputation_recorded": h["reputation_after"],
                     "profit_recomputed": round(profit_expect, 6),
                     "profit_recorded": h["own_profit"], "consistent": bool(good)})
    return {"gate": "G-H11_fixture_consistent", "pass": bool(ok), "rows": rows}


def run_gates(cfg: M.Config = None) -> dict:
    cfg = cfg or M.Config()
    gates = [gate_registry(cfg), gate_shared_rules(cfg), gate_no_horizon(cfg),
             gate_assist_lexicon(cfg), gate_leakage(cfg), gate_hash_stability(cfg),
             gate_parse(), gate_retry_trigger(cfg), gate_a0_is_an_oracle(cfg),
             gate_fixture_consistent(cfg)]
    return {"all_pass": all(g["pass"] for g in gates),
            "n_pass": sum(1 for g in gates if g["pass"]), "n_gates": len(gates),
            "gates": gates}


def prompt_hashes(cfg: M.Config = None) -> dict:
    """The preregistered prompt surface: one hash per arm, computed from a fixed demo state."""
    cfg = cfg or M.Config()
    out = {"spec": M.spec_dict(cfg), "history_window": HISTORY_WINDOW,
           "action_grid": {"n_index": NIDX + 1, "f": "index/20"},
           "json_instruction_sha256": hashlib.sha256(JSON_INSTR.encode()).hexdigest(),
           "rules_block_sha256": hashlib.sha256(rules_block(cfg).encode()).hexdigest(),
           "rules_block_chars": len(rules_block(cfg)), "arms": {}}
    for arm in ARMS:
        for wh in (True, False):
            msgs, meta = build_messages(cfg, arm, _demo_state(cfg, with_history=wh))
            out["arms"][f"{arm}|history={int(wh)}"] = {
                "prompt_sha256": IC.prompt_hash(msgs),
                "system_sha256": hashlib.sha256(msgs[0]["content"].encode()).hexdigest(),
                "system_chars": len(msgs[0]["content"]),
                "user_chars": len(msgs[1]["content"]),
                "declassified": meta["declassified"]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--emit", default="")
    ap.add_argument("--show", default="")
    a = ap.parse_args()
    cfg = M.Config()
    if a.show:
        msgs, meta = build_messages(cfg, a.show, _demo_state(cfg))
        print("=" * 100)
        print(msgs[0]["content"])
        print("-" * 100)
        print(msgs[1]["content"])
        print("=" * 100)
        print("declassified:", meta["declassified"], "notices:", meta["notices"],
              "retry_offered:", meta["retry_offered"])
        return
    res = run_gates(cfg)
    for g in res["gates"]:
        print(f"[{'PASS' if g['pass'] else 'FAIL'}] {g['gate']}")
        if not g["pass"]:
            print("   ", json.dumps({k: v for k, v in g.items()
                                     if k not in ("gate", "pass")}, default=str)[:1200])
    print(f"\n{res['n_pass']}/{res['n_gates']} gates pass")
    if a.emit:
        p = Path(a.emit)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {"gates": res, "hashes": prompt_hashes(cfg)}
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        tmp.replace(p)
        print("wrote", p)
    if not res["all_pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
