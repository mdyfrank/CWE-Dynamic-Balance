# Platform-Mediated Implementation of a Hidden-Action Equilibrium

**What changed.** In the earlier corpus the platform could see every merchant's fabrication rate and
print the resulting payoff table. Here it cannot. Fabrication is a hidden action; honest quality and
baseline complaint propensity are private types; the platform sees only sales, complaints, refunds,
audit flags and reputation, and acts only through a published, signal-contingent penalty. Every
definition, benchmark and proposition below is rebuilt on that information structure.

**Status of each claim.** Propositions marked **[proved]** are proved here. Propositions marked
**[verified]** are proved here *and* checked numerically. Statements marked **[computational]** are
properties of this parameterisation, not theorems, and are labelled as such wherever they appear.
All numerical checks live in `code/ha_theory_check.py` and their output in
`results/solver/ha_theory_checks.json` (12/12 PASS at the time of writing).

---

## 1. Primitives

A market is \(m = 4\) merchants and an outside option.

| Object | Symbol | Range | Who knows it |
|---|---|---|---|
| honest quality | \(q_i\) | catalogue pool, \([0.20, 0.95]\) | merchant \(i\) only |
| price | \(p_i\) | catalogue pool | everyone |
| baseline complaint propensity | \(b_i\) | \(\{0.01,\dots,0.15\}\), 15 points | merchant \(i\) only |
| **fabrication rate (the hidden action)** | \(f_i\) | \(F=\{0,0.05,\dots,1.00\}\), \(|F| = 21\) | merchant \(i\) only |
| reputation | \(r_i\) | 31-point grid on \([0,1]\) | everyone |
| policy | \((\kappa,\tau,\kappa_a,\tau_a)\) | finite class \(\mathcal P\) | published in advance |

Within round \(t\):

1. Merchant \(i\) chooses \(f_{it}\in F\), knowing its own type and its own history.
2. **Demand.** With \(\bar f = \frac1m\sum_k f_k\),
   \[
   u_i = \alpha\bigl(q_i + (1-q_i)f_i\bigr) + \beta r_i - \gamma p_i,\qquad
   s_i = \frac{e^{u_i}}{e^{w_0} + \sum_k e^{u_k}},\qquad
   Q = Q_0 e^{-\lambda \bar f},
   \]
   sales \(y_i = Q s_i\), merchant profit \(\pi_i = \text{margin}\cdot p_i y_i\), platform
   \(\text{GMV} = Q\sum_k p_k s_k\).
   Fabrication raises own appeal and shrinks the whole market: that tension is the entire economics.
3. **Signals.** \(z_{it}=(y_{it}, d_{it}, \rho_{it}, c_{it})\) with
   \[
   d_{it}\sim\text{Bin}(N_{\text{obs}},\,\theta_i),\quad \theta_i = \text{clip}(b_i + c_s f_i,0,1);
   \qquad
   \rho_{it}\mid d_{it}\sim\text{Bin}(d_{it},\phi);
   \]
   \[
   c_{it}\sim\text{Bin}(N_{\text{audit}},\,\psi_i),\quad \psi_i=\text{clip}(\psi_0+\psi_1 f_i,0,1).
   \]
4. **Penalty and reputation.**
   \[
   P_i = \kappa\Bigl(\tfrac{d_i}{N_{\text{obs}}}-\tau\Bigr)_{\!+} +
         \kappa_a\Bigl(\tfrac{c_i}{N_{\text{audit}}}-\tau_a\Bigr)_{\!+},
   \qquad
   r_i' = \text{clip}\bigl(r_i + \eta_r(1-r_i) - P_i,\,0,\,1\bigr).
   \]

Frozen constants (identical to Balance-9 except the three signal channels, which are new):
\(\alpha=3.0,\ \beta=1.5,\ \gamma=1.0,\ Q_0=1,\ \eta_r=0.20,\ c_s=0.60,\ N_{\text{obs}}=40,\
\lambda=1.0,\ \omega=0.5\Rightarrow w_0=1.0,\ \text{margin}=0.5p_i,\
\phi=0.50,\ N_{\text{audit}}=5,\ \psi_0=0.02,\ \psi_1=0.25\).

### 1.1 The information partition, which is enforced in code

| class | contents |
|---|---|
| `merchant_private` | own \(q_i, b_i\); own action history; own realised profit |
| `platform_observable` | \(p_i, y_i, d_i, \rho_i, c_i, r_i, P_i\), round index, published \((\kappa,\tau)\), category traffic index |
| `evaluator_only` | true \(f\) of anyone, rivals' \(q,b\), the displayed payoff table, its argmax, displayed regret, \(\theta_i\) |

Nothing in `evaluator_only` may reach a prompt. This is tested, not asserted
(`offline_tests/`, leakage tests).

---

## 2. The stationary restriction, stated before it is used

Holding a merchant's action fixed at \(f\), its reputation follows a Markov chain on the 31-point
grid with kernel induced by the signal distribution. Write \(\bar r(f;b,\kappa,\tau)\) for the
stationary mean of that chain, and define the **restricted stationary game**
\(\mathcal G(\theta,\kappa,\tau)\): players \(1..m\), actions \(F\), payoffs

\[
V_i(f) = \text{margin}\cdot p_i\cdot Q_0 e^{-\lambda\bar f}\cdot s_i(f),
\qquad
u_k = \alpha\bigl(q_k+(1-q_k)f_k\bigr) + \beta\,\bar r(f_k;b_k) - \gamma p_k .
\]

> **Boundary.** \(\mathcal G\) restricts merchants to constant actions and evaluates them at the
> stationary distribution of their own reputation chain. It is *not* the full dynamic game. It is the
> object the frozen corpus computed, so keeping it makes the old and new results commensurable, and
> every benchmark below is a benchmark *of \(\mathcal G\)*. Claims about the full game are confined to
> Proposition 1 and are correspondingly weaker.

---

## 3. Equilibrium

**Definition 1 (Hidden-action stationary Markov equilibrium).** In the full dynamic game, merchant
\(i\)'s information set at \(t\) is
\(I_{it}=\bigl(q_i,b_i,\ \{f_{is},y_{is},\pi_{is},d_{is},\rho_{is},c_{is},r_{is},P_{is}\}_{s<t},\ r_{it},\ (\kappa,\tau)\bigr)\)
— note it contains no rival action. A profile of Markov strategies \(\sigma^*\) with belief system
\(\mu\) is a *hidden-action stationary Markov equilibrium* if (i) \(\sigma_i^*\) maximises \(i\)'s
expected discounted payoff given \(\sigma^*_{-i}\) and \(\mu_i\), and (ii) \(\mu_i\) is obtained by
Bayes' rule from \(i\)'s own observations wherever possible and agrees with the equilibrium
distribution of \(f_{-i}\) on path.

Because rivals' actions are never observed, condition (ii) has bite: a merchant's belief about
\(f_{-i}\) is disciplined only by what its own signals reveal, and \(d_i\) depends on \(f_{-i}\) not
at all. Rivals' actions enter \(i\)'s payoff *only* through realised sales.

### Proposition 1 (existence) **[proved]**

*In the finite model — finite type space, \(|F|=21\), \(|R|=31\), and either a finite horizon \(T\) or
discounting \(\delta<1\) — a hidden-action stationary Markov equilibrium exists in behaviour
strategies.*

**Proof.** The state space \(R^m\times\Theta\) is finite and the action set \(F\) is finite. The
transition kernel is explicit and everywhere well defined: it is the product of binomial pmfs
composed with the deterministic map \(r\mapsto\text{clip}(r+\eta_r(1-r)-P,0,1)\). Stage payoffs are
bounded. For the finite-horizon case, apply backward induction, using Nash's theorem on the finite
stage game at each date; for the discounted infinite-horizon case, the model is a finite discounted
stochastic game, for which a stationary Markov perfect equilibrium in mixed strategies exists (Fink
1964; Takahashi 1964). ∎

**What this does not say.** It does not assert a *pure* equilibrium, and it does not assert
uniqueness. Both are separate questions and only one of them has a clean answer here.

### Proposition 1′ (pure existence and uniqueness) **[computational]**

*In every one of the 3,840 (seed, policy) pairs of the frozen design, \(\mathcal G(\theta,\kappa,\tau)\)
has **exactly one** pure-strategy Nash equilibrium.*

Verified by exhaustive enumeration of all \(21^4 = 194{,}481\) joint profiles and an exhaustive
unilateral-deviation test at tolerance \(10^{-12}\); extended to 38,400 pairs in the two-channel
class, again with zero exceptions.

> This is **not a theorem**. Finite games need not have pure equilibria, and nothing here rules out a
> parameterisation with several. It is reported because it removes equilibrium selection as a
> competing explanation for anything measured downstream: where "the equilibrium" is referred to in
> the experimental sections, it is well defined by computation, not by assumption.

---

## 4. The exact deviation identity, and everything that follows from it

Fix \(i\) and \(f_{-i}\). Write \(s_i\) for \(i\)'s share at \(f_i\), and for a candidate \(f'\)

\[
\Delta f = f'-f_i,\qquad
\Delta u_i = \alpha(1-q_i)\Delta f + \beta\bigl[\bar r_i(f') - \bar r_i(f_i)\bigr],\qquad
x = e^{\Delta u_i}.
\]

### Proposition 2 (exact deviation identity) **[verified]**

\[
\boxed{\ \frac{V_i(f',f_{-i})}{V_i(f_i,f_{-i})}
 = e^{-\lambda\Delta f/m}\cdot\frac{x}{s_i x + 1 - s_i}\ }
\]

**Proof.** \(V_i = C e^{-\lambda\bar f}s_i\) with \(C=\text{margin}\cdot p_i Q_0\). The traffic factor
contributes \(e^{-\lambda\Delta f/m}\) since only \(f_i\) moves and \(\bar f\) carries weight \(1/m\).
For the share, let \(a = e^{u_i(f_i)}\) and \(U = e^{w_0}+\sum_{k\ne i}e^{u_k}\), both free of \(f_i\).
Then \(s_i = a/(U+a)\), so \(U = a(1-s_i)/s_i\). Since \(u_i(f')-u_i(f_i)=\Delta u_i\) — the term
\(\alpha q_i\) cancels — we have \(e^{u_i(f')} = ax\), and

\[
s_i(f') = \frac{ax}{U+ax} = \frac{ax}{a(1-s_i)/s_i + ax} = \frac{s_i x}{1-s_i+s_i x}. \qquad\square
\]

*Checked:* 32,000 deviations, maximum relative error \(1.35\times10^{-15}\).

Two features of the identity drive everything below. Only \(\Delta u_i\) is policy-sensitive, and the
policy reaches it through exactly one term, \(\beta\Delta\bar r_i\) — the platform's **only**
instrument. And the share enters through \(h(s)=x/(sx+1-s)\), whose monotonicity in \(s\) flips sign
with \(x\); that sign flip is the source of the asymmetry in §4.2.

### Proposition 3 (exact incentive compatibility) **[verified]**

*\(f^*\) is a Nash equilibrium of \(\mathcal G\) iff for every \(i\) and every \(f'\ne f_i^*\), with
\(g=e^{\lambda\Delta f/m}\):*

\[
\text{either } g\,s_i \ge 1,\qquad\text{or}\qquad x \le \frac{g(1-s_i)}{1-g\,s_i}.
\]

**Proof.** No profitable deviation means \(e^{-\lambda\Delta f/m}h(s_i)\le1\), i.e.
\(x\le g(s_ix+1-s_i)\), i.e. \(x(1-gs_i)\le g(1-s_i)\). If \(1-gs_i\le0\) the left side is
non-positive and the right side positive, so it holds for every \(x>0\); otherwise divide. ∎

*Checked:* 3,360 deviations, 0 mismatches against direct payoff comparison, maximum identity residual
\(5.55\times10^{-16}\).

**Corollary 3.1 (the incumbency shield).** For an upward deviation of size \(\Delta f>0\), a merchant
with \(s_i\ge e^{-\lambda\Delta f/m}\) never gains, whatever the appeal gain and whatever the policy.
A merchant large enough internalises the market shrinkage it causes. At \(\lambda=1,m=4\) the
threshold is \(0.978\) for one grid step and \(0.779\) for a jump from \(f=0\) to \(f=1\) — so the
shield is real but essentially never binds in a four-merchant market. It is recorded because it is
the exact boundary at which the platform's instrument becomes unnecessary.

### 4.1 A sufficient condition the platform can act on

### Proposition 4 (SC-up) **[verified]**

*If for all \(f'>f_i\)*
\[
\beta\bigl[\bar r_i(f_i)-\bar r_i(f')\bigr]\;\ge\;\Bigl(\alpha(1-q_i)-\frac{\lambda}{m}\Bigr)(f'-f_i),
\]
*then no upward deviation from \(f_i\) is profitable, **for any share** \(s_i\).*

**Proof.** For \(x\ge1\), \(s_ix+1-s_i\ge1\), so \(h(s_i)\le x\) and
\(V_i(f')/V_i(f_i)\le e^{-\lambda\Delta f/m}x\). The displayed inequality is exactly
\(\Delta u_i\le\lambda\Delta f/m\), which makes that bound \(\le1\). If instead \(x<1\) the deviation
is unprofitable outright, since then both factors are below 1. ∎

*Checked:* 18,787 certified merchant cases, 0 unsound. **But only 25 of 40 equilibria (62.5%) are
certified.** The bound \(h(s)\le x\) is slack by precisely the share the merchant already holds, so
SC-up is a *design* rule, deliberately conservative, not a characterisation. Proposition 3 is the
characterisation.

### 4.2 Why there is no downward counterpart — an obstruction, not a gap

### Proposition 5 (no share-free downward condition) **[proved]**

*No condition on \((\alpha,\beta,\lambda,m,q_i,\bar r_i(\cdot))\) alone — that is, free of \(s_i\) —
can be sufficient to rule out a profitable downward deviation.*

**Proof.** Let \(f'<f_i\), so \(\Delta f<0\) and \(e^{-\lambda\Delta f/m}>1\). We have
\(\partial h/\partial s = x(1-x)/(1+s(x-1))^2\), which for \(x<1\) is strictly positive; hence \(h\) is
increasing in \(s\) with \(\sup_{s\in[0,1]}h(s)=h(1)=1\). Any bound not referring to \(s\) must hold
at that supremum, and yields
\(V_i(f')/V_i(f_i)\le e^{-\lambda\Delta f/m}\cdot 1 > 1\) — vacuous. ∎

**Reading.** Deviating *downward* expands the market, and the deviator keeps only \(s_i\) of the
expansion while paying the full appeal cost. Whether honesty pays is therefore irreducibly a question
of how big you already are. Since the platform cannot see types, and shares depend on types, downward
incentives cannot be certified type-free. This is a genuine structural asymmetry of the model, and it
is why the mechanism results below are all stated as deterrents against *upward* deviation.

### 4.3 What the platform can guarantee knowing only the type support

### Proposition 6 (robust implementation) **[verified]**

*Let \(q_{\min}=\min\text{supp}(q)\) and let \(b\) range over its support. If*
\[
\beta\cdot\min_{b}\bigl[\bar r(f_t;b)-\bar r(f';b)\bigr]\;\ge\;
\Bigl(\alpha(1-q_{\min})-\frac{\lambda}{m}\Bigr)(f'-f_t)\quad\text{for all } f'>f_t,
\]
*then \((\kappa,\tau)\) deters upward deviation from \(f_t\) for **every** type in the support.*

**Proof.** SC-up with the worst-case quality (largest appeal gain) and the worst-case baseline
(smallest reputational bite). ∎

**And here is the honest consequence.** Over the 64-policy class, 66 (policy, target) pairs are
robustly implementable, and the best of them is \((\kappa=0.5,\tau=0.40)\) implementing
\(f_t=0.90\) at mean GMV **0.373** — against a no-penalty baseline of **0.349**. Measured as a
*guarantee against the whole type support*, the platform's mechanism is worth about 2.4 GMV points
out of a first best of 0.898. Every larger number in this document, including the 70.5% second best,
relies on the *realised* type distribution rather than on a worst-case guarantee. That distinction
belongs in the paper's limitations, not in a footnote.

---

## 5. What bounds the platform: information, or punishment?

Two candidate explanations for why the second best falls short of the first best, with opposite
policy implications. The evidence separates them decisively.

### Proposition 7 (impossibility under uninformative signals) **[verified]**

*If the signal distribution does not depend on \(f\) (\(c_s=0\) and \(\psi_1=0\)), then \(\bar r_i\) is
constant in \(f\), \(\Delta u_i=\alpha(1-q_i)\Delta f\) contains no policy term, and no element of
\(\mathcal P\) changes the equilibrium set. The platform's implementation power is exactly zero.*

**Proof.** \(P(z\mid q,f)=P(z\mid q)\) makes the reputation kernel independent of \(f\), so
\(\bar r_i(\cdot)\) is a constant function and drops out of \(\Delta u_i\). The right-hand side of
Proposition 3 then involves no policy parameter. ∎

*Checked:* 20 seeds \(\times\) 64 policies; the variation of \(\bar r\) over \(f\) is exactly \(0.0\);
the unique equilibrium is \((f=1,1,1,1)\) under **every** policy.

> **A trap worth naming.** Even here, GMV varies by 0.0199 across the policy class, because penalties
> still bite differently across \(b\) and reallocate demand between differently-priced merchants.
> A naive "policy changes GMV, therefore the policy works" test would report success in a world where
> the mechanism is provably powerless. This is why implementation power is measured as
> \((G^{\mathrm{SB}}-G^{\mathrm{NP}})/(G^{\mathrm{FB}}-G^{\mathrm{NP}})\) and never as a raw GMV
> difference.

### Proposition 8 (refunds are a garbling) **[verified]**

*\(\rho\mid d\sim\text{Bin}(d,\phi)\) with \(\phi\) free of \(f\) makes \((d,\rho)\) a garbling of
\(d\): \(d\) is sufficient for \(f\), the Fisher information about \(f\) is unchanged, and every
decision rule using \(\rho\) is weakly dominated by one using \(d\) alone.*

**Proof.** \(P(d,\rho\mid f)=P(d\mid f)P(\rho\mid d)\) and the second factor is free of \(f\), so for
any \(f,f'\) the likelihood ratio is \(P(d\mid f)/P(d\mid f')\). Sufficiency, equality of Fisher
information, and Blackwell dominance all follow. ∎

*Checked:* maximum relative Fisher difference \(1.83\times10^{-15}\).

Contrast the audit channel, \(c\sim\text{Bin}(N_a,\psi_0+\psi_1f)\) with \(\psi_1>0\): genuinely
informative, and genuinely a second instrument. **Counting observable fields is not counting
instruments.** A dashboard with twenty columns may carry one.

### Proposition 9 (over-punishment destroys the instrument) **[verified]**

*The deterrent is the reputational **spread** \(\beta[\bar r(f_{\text{low}})-\bar r(f_{\text{high}})]\),
not the penalty level. As \(\kappa\to\infty\) with \(\tau\) fixed below the honest complaint rate, the
spread collapses to zero and the unique equilibrium returns to full fabrication.*

**Proof sketch and closed form.** As \(\kappa\to\infty\) any penalty event drives \(r\) to 0, so the
chain becomes "reset on a penalty, recover by \(\eta_r\) otherwise". The age since the last reset is
geometric with parameter \(p_0=P(d\le\lfloor\tau N\rfloor)\), and \(r\) after \(k\) clean rounds is
\(1-(1-\eta_r)^k\), giving
\[
\bar r_\infty(f,b)=\frac{\eta_r\,p_0(f,b)}{1-(1-\eta_r)p_0(f,b)} .
\]
When \(\tau\) is below what an honest merchant can achieve, \(p_0(0,b)\ll1\), so the *honest*
merchant's reputation is crushed along with the fabricator's and the spread vanishes. ∎

*Checked* on the solver's own tables, at the median baseline, \(\beta\times\)spread:

| \(\kappa\) | 0.5 | 2 | 8 | \(\to\infty\) |
|---|---|---|---|---|
| \(\tau=0.02\) | 1.278 | 0.595 | 0.056 | **0.011** |
| \(\tau=0.20\) | 1.399 | 1.401 | 1.397 | 1.381 |

A 116-fold collapse at the tight threshold; nothing at the loose one. Equilibrium fabrication
correspondingly falls and then rises in \(\kappa\), with the minimum at \(\kappa\approx0.5\text{–}2\)
and a return to \(f=1\) at \(\tau=0.02,\kappa=64\).

*Caveat, recorded rather than smoothed:* the closed form is a continuum result, while the solver
discretises \(r\) with `np.round` snapping onto 31 points — inherited verbatim from
`equilibrium._stationary_mean` and deliberately unchanged, since altering it would break
comparability with the frozen corpus. Snapping is not mean-preserving, so the closed form matches
only to \(3.2\times10^{-2}\), above the half-spacing of \(1.7\times10^{-2}\). The pass criterion is
therefore the spread behaviour above, computed from the solver's own tables; the closed form is
reported as explanation with its residual stated.

### Proposition 10 (limited liability caps implementation) **[verified]**

This is the result that answers the question in the section title.

*Reputation lives in \([0,1]\), so the entire deterrent budget the platform can ever spend on one
merchant is \(\beta\,\bar r_{\max}\le\beta\). Let \(D=1-f_c\) and \(g=e^{\lambda D/m}\), and let
\(\bar s\) be any upper bound on merchant \(i\)'s share with \(g\bar s<1\). Then the target \(f_c\) is
**not implementable** for merchant \(i\) — under any policy acting through reputation, however
informative the signal — whenever*

\[
\boxed{\ \alpha(1-q_i)D - \frac{\lambda D}{m} - \ln\!\frac{1-\bar s}{1-g\bar s} \;>\; \beta\,\bar r_{\max}. }
\]

**Proof.** Compare complying at \(f_c\) with abandoning to \(f'=1\). By Proposition 2 the gain is
\(e^{-\lambda D/m}\,h(s_i)\) with \(h(s)=x/(sx+1-s)\) and
\(x=\exp\{\alpha(1-q_i)D+\beta[\bar r_i(1)-\bar r_i(f_c)]\}\). Since \(\bar r_i(1)\ge0\) and
\(\bar r_i(f_c)\le\bar r_{\max}\), we have \(x\ge x_{\min}=\exp\{\alpha(1-q_i)D-\beta\bar r_{\max}\}\).
For \(x>1\), \(h\) is decreasing in \(s\), so \(s_i\le\bar s\) gives \(h(s_i)\ge h(\bar s)\).
Abandonment is therefore strictly profitable whenever
\(e^{-\lambda D/m}x_{\min}/(\bar s x_{\min}+1-\bar s)>1\); rearranging and taking logarithms — valid
because \(1-g\bar s>0\) — gives the display. ∎

**Corollary 10.1 (implementation floor).** Solving for \(D\) gives a per-merchant floor
\(f_{\text{floor}}(q_i,\bar s)\) below which no target is implementable. In any equilibrium, under any
policy in any class acting through reputation, merchant \(i\) plays at least \(f_{\text{floor}}\).

*Checked:* 160 merchant-equilibrium cases, **0 violations** of either the exact form (\(\bar s=\)
realised share) or the platform-usable form (\(\bar s=0.5\), assertable from observed sales alone).
Deterrent budget \(\beta\bar r_{\max}=1.405\). For the lowest quality in the catalogue pool
(\(q=0.20\)) the floor is **0.35** at vanishing share and **0.25** at \(\bar s=0.5\).

**Why this is the key result.** *The floor contains no signal parameter.* It is not a statement about
what the platform can learn; it is a statement about how much punishment reputation can hold. Three
otherwise puzzling facts follow at once:

1. **The monitoring continuum saturates** (§6): scanning \((N_{\text{obs}},c_s)\) up to near-perfect
   monitoring raises implementation power only to \(\approx0.63\), never toward 1.
2. **Tightening \(\tau\) raises fabrication.** At near-perfect monitoring, equilibrium mean \(f\) is
   0.33 at \(\tau=0.40\) but 0.57 at \(\tau=0.10\) and 0.73 at \(\tau=0.02\): past the floor,
   compliance is worth less than abandonment and merchants switch. Per-seed equilibria at intermediate
   \(\tau\) are visibly **bimodal** — some merchants just below the threshold, others abandoned near
   \(f=0.6\).
3. **The first-best gap is 92% incentive, 8% instrument** (§6). Better signals address the 8%.

### 5.1 Proposition 11 (dead zones) **[verified]**

*If \(\tau\) exceeds the complaint rate reachable at fabrication \(f\), then over \([0,f]\) the
marginal reputational bite per grid step is negligible next to the appeal gain of that step, and no
target strictly inside the interval is implementable by that policy.*

Measured with an economic criterion (marginal \(\beta\Delta\bar r\) per step below 1% of
\(\alpha(1-q_{\text{med}})\times\text{step}\), \(q_{\text{med}}=0.45\), appeal gain 0.0825 per step):
62 dead zones, the widest reaching \(f=0.35\) at \((\kappa=0.5,\tau=0.40)\). This is why a
second-best search must range over \(\tau\), not only \(\kappa\), and why any implementation claim
must name the policy it holds under.

---

## 6. GMV benchmarks — four denominators that must never be mixed

All exact: \(21^4=194{,}481\) joint profiles enumerated per (seed, policy), \(\approx15\) ms each, so
no benchmark rests on a heuristic search. Feasibility was **measured**, not assumed.

| Symbol | Definition | What the platform is allowed to know | Mean |
|---|---|---|---|
| \(G^{\mathrm{FB}}\) | best profile under the best policy in \(\mathcal P\); IC dropped | observes and sets \(f\) | **0.8978** |
| \(G^{\mathrm{SB}}_{\mathcal P,\text{seed}}\) | best IC outcome, policy chosen **per market** | tunes to unobservable types | **0.7035** (78.2%) |
| \(G^{\mathrm{SB}}_{\mathcal P,\text{unif}}\) | best IC outcome, **one** policy for all markets | nothing unobservable | **0.6326** (70.5%) |
| \(G^{\mathrm{NP}}\) | equilibrium with no penalty | — | **0.3491** (38.9%) |

**\(G^{\mathrm{SB}}_{\mathcal P,\text{unif}}\) is the honest denominator.** The per-seed second best
silently grants the platform the ability to read \(q\) and \(b\) and choose the penalty that suits
them: across 60 seeds the per-seed optimum takes **25 distinct** \((\kappa,\tau)\) values. The gap
between the two — a **tuning premium of 10.1%** of the per-seed second best — is an artefact of
conditioning on unobservables, and any claim about how close a merchant population comes to "what the
platform could have done" must use the uniform figure.

Against it, the two named policies do well: \(P_{\mathrm{GMV}}\) reaches **97.1%** and
\(P_{\mathrm{robust}}\) **97.3%** of the uniform second best, ranking 8th and 7th of 64. Against the
per-seed figure the same policies read 87.5% and 87.6%. Both are correct; they answer different
questions; the report states which.

**\(G^{\mathrm{SB}}_{\mathcal P,\text{unif}}\) is selected in sample, and that is worth 2.1 points.**
The single policy \((\kappa,\tau)=(2.0,0.30)\) is chosen by maximising the mean over the same 60 seeds
it is then scored on, so 0.6326 is an in-sample optimum, not an out-of-sample guarantee.
Leave-one-out — choose the policy on 59 seeds, score it on the held-out one — gives **0.6140**, i.e.
**68.4%** of \(G^{\mathrm{FB}}\) rather than 70.5%; the selection premium is **0.0186**, about 2.9% of
the benchmark. The cause is a near-tie at the top of the class: \((2.0,0.30)\) scores 0.632578 and
\((16.0,0.40)\) scores 0.632465, a gap of \(1.1\times10^{-4}\), so the argmax moves between three
policies across the 60 folds. **Where a claim needs a number the platform could have committed to
without seeing these markets, 68.4% is the defensible one**; 70.5% is the value of the best policy in
the class given this seed block, and is reported as such. Both are recomputed by
`offline_tests/recompute_gmv_benchmarks.py`.

**First best is not "no penalty".** A penalty can raise attainable GMV even with the incentive problem
switched off, because reputation enters a logit share and merchants carry different prices, so a
penalty biting unequally across \(b\) reallocates demand toward high-price merchants. This happens on
21 of 60 seeds, worth up to 3.83% of \(G^{\mathrm{FB}}\). \(G^{\mathrm{FB}}\) is therefore taken over
the policy class *and* profiles — the same instruments as the second best, with only IC dropped —
and \(G^{\mathrm{FB}}_{\text{no-penalty}}\) is retained as a separate field. **GMV is not monotone in
reputation**, and any argument assuming it is, is wrong in this model.

### 6.1 Decomposition of the first-best gap **[computational]**

Holding the uniform second-best policy fixed and switching merchant behaviour to the first-best
profile separates the two candidate causes:

| Component | Value | Share |
|---|---|---|
| total gap \(G^{\mathrm{FB}}-G^{\mathrm{SB}}_{\text{unif}}\) | 0.2652 | 100% |
| instrument cost (penalty fires on compliant merchants) | 0.0212 | **8%** |
| incentive gap (merchants will not play the first-best profile) | 0.2440 | **92%** |

Read with Proposition 10: the binding constraint is how much punishment reputation can hold, not how
much the platform can learn. **Better detection is not the lever here.**

### 6.2 The monitoring continuum **[computational]**

Implementation power \(\pi=(G^{\mathrm{SB}}_{\mathcal P}-G^{\mathrm{NP}})/(G^{\mathrm{FB}}-G^{\mathrm{NP}})\),
scanning \(N_{\text{obs}}\in\{10,20,40,80,160\}\times c_s\in\{0,0.15,0.30,0.60,1.00\}\), 20 seeds:

| \(N\backslash c_s\) | 0.00 | 0.15 | 0.30 | 0.60 | 1.00 |
|---|---|---|---|---|---|
| 10 | 0.011 | 0.153 | 0.359 | 0.564 | 0.571 |
| 40 | 0.018 | 0.411 | 0.581 | **0.619** | 0.613 |
| 160 | 0.024 | 0.546 | 0.619 | 0.632 | **0.635** |

The \(c_s=0\) column is Proposition 7 realised numerically — power \(\approx0.02\), all of it demand
reallocation, equilibrium \(f=1\) throughout. Moving right and down interpolates toward the oracle
endpoint but **saturates near 0.63**: the frozen spec (\(N=40, c_s=0.60\)) already sits at 0.619, so
essentially all of the available monitoring value is captured, and quadrupling the sample buys 0.013.
By Proposition 10 the remaining third of the gap is not purchasable with information at all.

### 6.3 Two channels are barely better than one **[computational]**

Extending the class to \(8\times8\) complaint settings \(\times\) 10 audit settings (640 policies,
38,400 (seed, policy) pairs) moves the per-seed second best from 0.7035 to **0.7084** (**+0.7%**) and
the committable uniform benchmark from 0.6326 to **0.6366** (**+0.6%**), the winner shifting from
\((\kappa,\tau)=(2,0.30)\) to \((\kappa,\tau,\kappa_a,\tau_a)=(2,0.30,0.05,0.0)\) — the *smallest*
non-zero audit penalty on the grid. The audit channel is genuinely informative (Proposition 8) and
still nearly worthless at these parameters, because it spends the same bounded reputational budget.

The gap decomposition barely moves: **8.1% instrument / 91.9% incentive**, against 8%/92% in the
64-policy class. That the split survives a tenfold enlargement of the instrument set is the strongest
available evidence that it is a property of the environment rather than of the grid.

---

## 7. The old displayed equilibrium, the oracle, and this one

### Proposition 12 (oracle reduction) **[verified]**

*The payoff table displayed by the earlier `U`/`H`/`R`/`G` interfaces is*
\(\tilde V^{(t)}_j(i)=V_j\bigl(f(i),f^{\,t-1}_{-j}\bigr)\) *— merchant \(j\)'s exact payoff against last
round's rivals. Its argmax is \(j\)'s exact best response, so a fixed point of displayed-argmax play
is by definition a pure equilibrium of \(\mathcal G\).*

**Proof.** Compare `balance9_payoff.displayed_table` with \(V_j\): the same logit, the same stationary
reputation lookup, the same traffic factor. The interface sweeps \(j\)'s own action with rivals held
at \(f^{t-1}_{-j}\). ∎

*Checked:* 120 cases, 120 best-response fixed points inside the independently enumerated equilibrium
set, 0 exceptions.

**Consequence, stated plainly.** Convergence under the old interfaces is best-response dynamics
executed by the interface. It is not evidence that a language model inferred an equilibrium. What the
old experiments do measure — reliably, and worth reporting — is *adherence*: how dependably a model
executes a best response that has been computed for it, and how that depends on framing, policy and
model. See `ORACLE_DEPENDENCY_AUDIT.md` for the claim-by-claim disposition.

**The three regimes.**

| Regime | Platform knows | Equilibrium object | GMV |
|---|---|---|---|
| oracle / displayed | \(f\), \(q\), \(b\) exactly | Nash of the displayed game; interface supplies the best response | \(\to G^{\mathrm{FB}}\) |
| hidden action (this work) | signals only | Definition 1; \(\bar r(\cdot)\) is the only instrument | \(G^{\mathrm{SB}}_{\mathcal P,\text{unif}}=70.5\%\) |
| uninformative signals | nothing about \(f\) | unique, \(f=1\) | \(G^{\mathrm{NP}}=38.9\%\) |

The old corpus sat at the top row and reported it as though it were the middle one. This document
places the middle row on its own footing.

---

## 8. Assumption boundaries and counterexamples

Stated as limits on what may be claimed, not as caveats to be skipped.

1. **The policy class is finite and small.** Everything called a second best is a
   **policy-class second best** over \(\{\kappa\}\times\{\tau\}\) (64, or 640 with the audit channel).
   No statement here is a theorem about all mechanisms. Transfers, menus, message-contingent contracts,
   bonding and entry fees are all outside the class and could do better.
2. **The stationary restriction** (§2) confines merchants to constant actions evaluated at the
   stationary reputation distribution. Non-stationary strategies — build reputation, then cash it in —
   are not in the strategy space and are not evaluated. This is a real gap between Proposition 1 and
   Propositions 2–12.
3. **Uniqueness is computational, not proved** (Proposition 1′). It holds in 38,400 pairs tested and
   may fail elsewhere.
4. **Robust implementation is nearly vacuous** (Proposition 6): guaranteed against the whole type
   support, the mechanism is worth 0.373 against 0.349. The headline 70.5% depends on the realised
   type distribution. Do not report the latter as a guarantee.
5. **SC-up certifies only 62.5% of equilibria** (Proposition 4) and there is provably no downward
   counterpart (Proposition 5). Type-free certification is one-sided, permanently.
6. **The signal model is parametric.** \(\theta=b+c_sf\) is linear and \(d\) is binomial with known
   \(N\). Overdispersion, correlated complaints, strategic complaining and merchant-side manipulation
   of \(d\) are all excluded, and each would weaken the mechanism further.
7. **A stable fabrication rate is not equilibrium evidence.** Stability is consistent with
   indifference, with a dead zone (Proposition 11), or with a model repeating its previous answer.
   Equilibrium claims in the experimental sections require exploitability and deviation-gain
   measurements, never a flat trajectory.
8. **\(M6 \approx 1\) never means "maximal GMV".** M6 is a ratio of a *projected* equilibrium quantity
   to a *projected* benchmark inside the displayed game. It is retained only as a projected-equilibrium
   benchmark, is never mixed with realised GMV, and is never described as reaching a theoretical
   maximum.

### Counterexamples on record

| Intuition | Refuted by | Where |
|---|---|---|
| "First best = no penalty, since penalties only lower reputation" | penalty reallocates demand across prices; helps on 21/60 seeds, up to 3.83% | §6 |
| "GMV is monotone in reputation" | same | §6 |
| "Punish harder to deter more" | spread collapses 116× as \(\kappa\to\infty\) at \(\tau=0.02\); \(f\to1\) | Prop 9 |
| "Lower the threshold to cut fabrication" | mean \(f\) rises 0.33 → 0.73 as \(\tau\) falls 0.40 → 0.02 | Prop 10 |
| "More observable fields = more instruments" | refunds add exactly zero Fisher information | Prop 8 |
| "Policy changes GMV, so the mechanism works" | GMV moves 0.0199 across policies even when signals are provably uninformative | Prop 7 |
| "Better detection closes the gap to first best" | 92% of the gap is incentive, and the floor has no signal parameter | §6.1, Prop 10 |
| "A share-free rule can certify honesty" | no downward share-free condition exists | Prop 5 |

---

## 9. Traceability

| Claim | Artefact | Key |
|---|---|---|
| Existence / uniqueness | `results/solver/ha_theory_checks.json` | `P1_existence` |
| Exact IC characterisation | same | `P2_exact_ic` |
| Deviation identity | same | `P3a_deviation_identity` |
| SC-up and its slack | same | `P3b_sufficient_condition_upward` |
| Robust implementation | same | `P4_robust_implementation` |
| Over-punishment reversal | same | `P5_overpunishment_reversal`, `P5b_large_kappa_limit` |
| Impossibility | same | `P6_impossibility_uninformative_signal` |
| Refund garbling | same | `P7_refunds_are_a_garbling` |
| Oracle reduction | same | `P8b_oracle_reduction` |
| Dead zones | same | `P9_dead_zone` |
| Implementation floor | same | `P10_implementation_floor` |
| Four GMV benchmarks, tuning premium, gap decomposition | `results/solver/ha_benchmarks.json` | `aggregate.G_SB_uniform` |
| Two-channel class | `results/solver/ha_benchmarks_extended.json.gz` | `aggregate` |
| Informativeness, detector rates, calibration, monitoring continuum | `results/solver/ha_signal_analysis.json` | — |

Reproduce with:

```
python code/ha_benchmarks.py --seeds 70000-70059
python code/ha_benchmarks.py --seeds 70000-70059 --extended
python code/ha_theory_check.py
python code/ha_signal_analysis.py
```

No LLM API is involved in any of it.
