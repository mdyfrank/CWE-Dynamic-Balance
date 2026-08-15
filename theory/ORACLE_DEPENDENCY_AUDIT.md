# Oracle-Dependency Audit

**Scope.** Every quantitative claim in the existing corpus (Balance-6 through Balance-9, including the
Azure P3 replication package) is examined for dependence on platform capabilities that a real
marketplace does not have. Where a claim depends on such a capability, this document states which
capability, where in the code it is exercised, what the claim can still support, and what it can no
longer support.

**Result in one sentence.** The four LLM interfaces `U`, `H`, `R`, `G` all read from a payoff table
whose construction requires the platform to know every merchant's private type and every rival's true
hidden action; that table is not merely unavailable in practice, it is the object whose absence
*defines* the hidden-action problem, so all four interfaces are reclassified as **oracle upper-bound
controls** and are removed from the evidentiary base for any claim about autonomous learning,
strategic adaptation, or convergence.

---

## 1. The five oracle capabilities

The mandate names five capabilities. Each is defined here precisely, located in the frozen code, and
given a verdict.

| # | Capability | Formal statement | Verdict under hidden action |
|---|---|---|---|
| O1 | Platform observes true fabrication and true quality | platform's information set contains \(f_{jt}\) and \(q_j\) for all \(j\) | **Unavailable.** \(f\) is the hidden action by construction; \(q_j, b_j\) are merchant-private types. |
| O2 | Platform can construct the full fabrication→payoff counterfactual table | platform can evaluate \(\tilde V_j(i)\) for all 21 grid points | **Unavailable.** Derived from O1; see §2. |
| O3 | Platform knows the displayed argmax | platform can compute \(\arg\max_i \tilde V_j(i)\) | **Unavailable.** Derived from O2. |
| O4 | Platform can compute regret against the argmax | platform can evaluate \((\max_i \tilde V_j(i) - \tilde V_j(i^{\text{chosen}}))/\max_i \tilde V_j(i)\) | **Unavailable.** Derived from O2. |
| O5 | Platform can directly override the merchant's action | platform replaces the submitted \(f\) with one of its own choosing before execution | **Unavailable** as a *mechanism*; retained only as an upper-bound *control*. See §6. |

O2–O4 are all consequences of O1. O5 is logically independent: a platform could in principle have
contract rights to force a listing edit. It is nonetheless excluded from the main experiment, for the
reason given in §6.

---

## 2. Where the oracle enters the code

The single point of entry is the displayed payoff table.

```
www_project/balance9_payoff.py:110
def displayed_table(q, p, bidx, rg, j, f_rivals):
    """V~_j(i): the 21 numbers actually shown to merchant j."""
```

Its arguments are, in order:

| Argument | Meaning | Information class under hidden action |
|---|---|---|
| `q` | honest quality of **all m merchants** | `merchant_private` (own only); rivals' \(q\) is not even merchant-observable |
| `p` | price of all merchants | `platform_observable` — this one is legitimate |
| `bidx` | baseline complaint-propensity index of all merchants | `merchant_private` |
| `rg` | stationary reputation table \(\bar r(b, f)\) | requires `bidx` and the full \(f\) grid |
| `j` | which merchant the table is for | legitimate |
| `f_rivals` | **true fabrication of every rival in the previous round** | `evaluator_only` — this is the hidden action |

Three of the six arguments are unobservable. The body (lines 120–129) computes, for each candidate
index \(i\),

\[
u_k = \alpha\bigl(q_k + (1-q_k)f_k\bigr) + \beta\,\bar r\bigl(b_k, f_k\bigr) - \gamma p_k,
\qquad
s = \frac{e^{u}}{e^{w_0} + \sum_k e^{u_k}},
\qquad
\tilde V_j(i) = \text{margin}\cdot p_j \cdot Q_0 e^{-\lambda \bar f}\, s_j .
\]

Every one of \(q_k\), \(b_k\), \(f_k\) for \(k \ne j\) appears. There is no reformulation that removes
them: the logit share is a function of the *whole profile*, and the traffic factor \(Q_0e^{-\lambda\bar f}\)
is a function of the *mean true fabrication of the market*. A platform that could compute
\(\tilde V_j\) would already know the answer to the question the mechanism is supposed to solve.

**This is not a fixable engineering limitation.** It is the definition of the problem. Any experiment
that shows a merchant the table has already assumed away hidden action.

---

## 3. Interface `H` — why the correct-argmax annotation cannot be implemented

```
www_project/balance9_prompts.py:186
    if interface == "H":
        k = argmax_min_index(v)
        return s + MARK_FACTUAL.format(k=k), k
```

`H` appends a sentence naming index `k` as the profit-maximising choice. The annotation is *factual*
by construction — `argmax_min_index(v)` reads the same vector `v` the merchant is shown.

Under hidden action the platform cannot produce `k` because:

1. **It cannot produce `v`** (§2). Every intermediate quantity requires \(q_{-j}, b_{-j}, f_{-j}\).
2. **Even a self-referential version fails.** Suppose the platform tried to annotate using only what
   it observes — say, by regressing observed sales on observed complaint rates. The resulting
   "recommended index" would not be the argmax of the merchant's true payoff, because the merchant's
   payoff depends on its own private \(q_j\) (through \(\alpha(1-q_j)f_j\), the appeal gain from
   fabrication, which is *larger for low-quality merchants*). Two merchants with identical observable
   histories and different \(q_j\) have different argmaxes. An annotation that ignores this is not a
   "noisy H"; it is a different and possibly harmful message.
3. **The annotation is not a mechanism.** Even granting the platform a miracle oracle, telling a
   merchant its argmax is not an incentive instrument — it changes no payoff. `H` measures whether the
   model can *read*, not whether the platform can *implement*.

**Conclusion.** `H` is retained only as the **information-saturated upper bound**: the fraction of
decisions a model gets right when the answer is written in the prompt. It bounds from above what any
non-oracle interface can achieve on *action-selection accuracy*, and it bounds nothing about
equilibrium implementation.

---

## 4. Interface `R` — why the displayed-regret verifier cannot be implemented

```
www_project/balance9_runner.py:93
GUARD_SPEC = {"R": (ETA, None), "G": (ETA, "argmax")}

www_project/balance9_runner.py:333
    if eta_g is not None and rel_regret(v, i) > eta_g:
        tr["verification_failed"] = True
```

and the message actually sent back:

```
www_project/balance9_prompts.py:214
    return (f"Your submitted action_index [{i_chosen}] (f={idx_to_f(i_chosen):.2f}) has projected "
            f"profit {v[i_chosen]:.4f}. Relative to the highest projected profit in the table shown, "
            f"that is a shortfall of {r*100:.2f}%, ...")
```

The verifier does three impossible things in one message:

1. It **evaluates the merchant's chosen action** — `v[i_chosen]` — which requires \(\tilde V_j\).
2. It **compares against the maximum** — `rel_regret(v, i)` divides by `v.max()`, which requires the
   entire table, not just one entry.
3. It **reveals the shortfall as a number**, which is a sufficient statistic for the direction of the
   argmax and, iterated, locates it exactly. `R` is `H` delivered in instalments.

A real platform can reject a listing for an *observable* reason — a complaint spike, an audit flag, a
contradiction between the listing text and the product record. It cannot reject a listing for
"leaving projected profit on the table", because it cannot see the projection and, more fundamentally,
because the merchant's projected profit is not the platform's business.

**A weaker, implementable analogue exists and is the basis of the new `A3_assist` arm:** a retry
triggered by an *observable* signal, whose message contains only `platform_observable` fields (current
complaint rate, distance to the penalty threshold \(\tau\), penalty incurred, reputation trajectory).
This is a genuinely different intervention and is preregistered as such — it is *not* a replication of
`R` and its results may not be compared to `R`'s as if they were the same arm.

**Conclusion.** `R` is retained only as the **verified-retry upper bound**.

---

## 5. Interface `G` — why the terminal override cannot be implemented

```
www_project/balance9_runner.py:152
def override_index(v, eta, target):
    return PR.argmax_min_index(v) if target == "argmax" else balanced_in_certificate(v, eta)

www_project/balance9_runner.py:336
        i, tr["override"] = override_index(v, eta_g, ovr_target), True
```

`G` discards the merchant's action and substitutes the displayed argmax. Two separate objections:

1. **It needs the oracle.** The substituted index *is* `argmax_min_index(v)` — §2 applies verbatim.
2. **It is not a game.** If the platform sets the action, the merchant's action is not a choice and
   there is no incentive-compatibility question to answer. An outcome produced by override is a
   statement about the *platform's* optimisation, not about merchant behaviour. Reporting a `G`
   outcome alongside `U` outcomes as if both were "what merchants did" conflates a controlled variable
   with a measured one.

Note that a platform *could* have contract rights to force a listing edit; delisting and forced-edit
powers exist in real marketplaces. But such powers must be triggered by an observable, and the target
of the edit must be describable in observable terms ("remove the unverifiable claim"), not
("set \(f = 0.35\)"). A forced-edit mechanism triggered by audit flags is a legitimate future
mechanism; it is not `G`, and it is not in this study.

**Conclusion.** `G` is retained only as the **enforced-compliance ceiling**: the GMV the market
reaches when the platform simply *takes* the action it wants. It is the empirical analogue of
\(G^{\mathrm{FB}}\) restricted to the displayed game.

---

## 6. Interfaces `M` and `O` — the two non-LLM reference agents

```
www_project/balance9_runner.py:276
    i = PR.argmax_min_index(v) if interface == "M" else int(state["eq_index"])
```

`M` plays the displayed argmax mechanically; `O` plays a precomputed equilibrium index. Both are
oracle agents by construction and were always labelled as reference series rather than findings. No
change of status is required; they are noted here for completeness.

---

## 7. The result that reframes everything: the displayed table *is* the best-response map

This is the sharpest statement of the problem and it is now a verified proposition (**P8b**,
`results/solver/ha_theory_checks.json`, 120/120 cases).

Compare `displayed_table` (§2) with the stationary payoff \(V_j(f_j, f_{-j})\) used by the
equilibrium solver. They are the same function. Therefore

\[
\tilde V_j^{(t)}(i) \;=\; V_j\bigl(f(i),\, f_{-j}^{\,t-1}\bigr),
\]

i.e. **the displayed table is merchant \(j\)'s exact payoff against last round's rivals**, and its
argmax is merchant \(j\)'s exact best response. Consequently:

- A fixed point of "everyone plays the displayed argmax" is, *by definition*, a pure-strategy Nash
  equilibrium of the restricted game.
- The dynamic implemented by `U`/`H`/`R`/`G` is **best-response dynamics** (equivalently, Cournot
  adjustment / one-step fictitious play), executed by the interface, not by the model.
- Numerical check: over 120 (seed, policy) cases, every fixed point of best-response dynamics lay
  inside the independently enumerated pure-equilibrium set. Zero exceptions.

**Therefore the earlier observation that "LLM merchants converge" is, at the level of the mechanism,
a statement about the interface's arithmetic.** The genuine and still-interesting empirical content
of the old experiments is *adherence*: how reliably a language model executes a best response that
has been computed for it, and how that reliability varies with framing, policy, and model. That is a
real and reportable finding. It is not evidence of strategic reasoning under uncertainty, and this
audit withdraws any sentence that presented it as such.

---

## 8. Claim-by-claim disposition

| Claim in the existing corpus | Depends on | Disposition |
|---|---|---|
| Displayed-game equilibria exist, are computable and (here) unique | O1–O2 for *construction*, none for *validity* | **RETAINED, unchanged.** It is a correct statement about a well-defined game. Scope note added: the game is the displayed game. |
| Dominance-solvability / strategic-content audit of the displayed game | O1–O2 | **RETAINED, unchanged.** Correct statement about the same well-defined game; it is in fact the reason the displayed game is a weak test-bed, and that reading is strengthened, not weakened. |
| Policy comparison \(P_{\mathrm{GMV}}\) vs \(P_{\mathrm{robust}}\) in the displayed game | O1–O2 | **RETAINED with scope restriction.** Valid as a comparison of policies *under oracle information*. Must not be quoted as a policy recommendation for a real platform. |
| Model-level adherence, framing sensitivity, retry effects (`U` vs `H` vs `R` vs `G`) | O1–O5 | **RETAINED, RELABELLED.** These are measurements of *instruction-following and arithmetic reliability under a fully revealed objective*. Every reporting sentence is rewritten accordingly. |
| "LLM merchants converge to equilibrium" | O2, O3 | **WITHDRAWN as stated.** Replaced by: the interface computes the best response; the observed convergence is a property of best-response dynamics (P8b). Retained sub-claim: models execute the computed best response at a measurable and model-dependent rate. |
| "LLM merchants learn / adapt strategically" | O2, O3 | **WITHDRAWN.** No environment in the old corpus contains identifiable strategic interaction that the model had to infer; the rival profile was supplied. |
| "M6 ≈ 1, so the market reaches (near-)maximal GMV" | O1–O2 and a denominator error | **WITHDRAWN as stated.** M6 is a ratio of a *projected* equilibrium quantity to a *projected* benchmark, both computed inside the displayed game. It is retained under the name **projected-equilibrium benchmark**, never as "theoretical maximum GMV", and never mixed with realised GMV. See `HIDDEN_ACTION_THEORY.md` §7 for the three ratios that replace it. |
| Reported limitations of Balance-8/9 | — | **RETAINED and extended**, not replaced. |

---

## 9. What the reclassified arms are still good for

Retaining `H`, `R`, `G` as oracle controls is not a consolation prize; they pin down the top of the
scale that the new arms are measured against.

| Arm | New name | What it bounds |
|---|---|---|
| `H` | information-saturated control | the ceiling on *action-selection accuracy* when the correct action is stated |
| `R` | verified-retry control | the ceiling on accuracy when errors are detected and returned for correction |
| `G` | enforced-compliance control | the GMV ceiling when the platform simply sets the action; the displayed-game analogue of \(G^{\mathrm{FB}}\) |
| `U` | revealed-table control | accuracy when the full counterfactual is shown but not annotated |

The new experiment's four arms (`A0_oracle`, `A1_policy`, `A2_history`, `A3_assist`) are designed so
that `A0_oracle` reproduces the old `U` condition on the *new* seed block, giving a paired bridge
between corpora that does not require reusing a single old data point.

---

## 10. Traceability

Every statement above is checkable from artefacts in this branch:

| Statement | Artefact |
|---|---|
| Displayed table needs \(q_{-j}, b_{-j}, f_{-j}\) | `www_project/balance9_payoff.py:110-129` (frozen, unmodified) |
| `H` marks `argmax_min_index(v)` | `www_project/balance9_prompts.py:186-188` |
| `R` reports `v[i_chosen]` and the relative shortfall | `www_project/balance9_prompts.py:204-218`, `balance9_runner.py:333` |
| `G` substitutes the argmax | `www_project/balance9_runner.py:93,152-153,336` |
| Displayed table = best-response payoff; fixed points are Nash | `code/ha_theory_check.py::p8b_oracle_reduction`; `results/solver/ha_theory_checks.json` check `P8b_oracle_reduction`, 120/120 |
| Pure equilibria exist and are unique here | same file, check `P1_existence`: 3840 pairs, 0 with zero, 0 with multiple |
| GMV benchmarks that replace M6 | `results/solver/ha_benchmarks.json`, `results/solver/ha_benchmarks_extended.json.gz` |

**Nothing in the old branches was modified.** The frozen source files are read-only inputs to this
audit; the citations above are to their unchanged contents.
