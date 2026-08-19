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

## 2. Three games, kept apart on purpose

Most of the confusion available in this subject comes from using one symbol for three different
objects. This section defines all three before anything is computed, so that every later claim can
name the game it is a claim about.

| | \(\Gamma_{80}\) | \(\Gamma_\delta\) | \(\mathcal G(\theta,\kappa,\tau)\) |
|---|---|---|---|
| horizon | 80 rounds, hard stop | infinite, discount \(\delta\) | one shot |
| strategies | history-dependent | stationary Markov | one constant \(f_i\) |
| reputation | realised path from \(r_{i0}=0.5\) | realised path | replaced by \(\bar r(f_i)\) |
| what it is | the game the experiment **runs** | the game the merchants are **told** they are in | the game the frozen corpus **solves** |

### 2.1 \(\Gamma_{80}\) — the game the experiment actually runs

- **Players.** \(N=\{1,\dots,m\}\), \(m=4\). The platform is *not* a player: it commits to
  \((\kappa,\tau,\kappa_a,\tau_a)\) before \(t=1\), publishes it, and never moves again.
- **Horizon.** \(T=80\) rounds (40 in tier `B_reduced`), with no continuation value and no scrap value
  after \(T\).
- **Types.** \(\theta_i=(q_i,b_i)\) drawn once at \(t=0\) and fixed. Private to \(i\) throughout.
- **State.** \(x_t=(r_{1t},\dots,r_{mt})\in R^m\), \(|R|=31\), so \(|R^m|=31^4=923{,}521\).
  \(r_{i0}=0.5\) for all \(i\) (grid index 15).
- **Actions.** \(f_{it}\in F\), \(|F|=21\), chosen simultaneously, never observed by anyone else.
- **Merchant-private.** \(q_i,b_i\); own action history \(f_{i,<t}\); own realised profit history;
  own signal history \((d,\rho,c)_{i,<t}\).
- **Platform-observable.** \(p_k\) and \(r_{kt}\) for **every** \(k\) (both are published), own
  \((y,d,\rho,c,P)\), the round index, the lagged traffic index \(Q_{t-1}\), and \((\kappa,\tau,\kappa_a,\tau_a)\).
- **Evaluator-only.** Any merchant's true \(f\); rivals' \(q,b\); \(\theta_i=b_i+c_sf_i\); the
  displayed payoff table of the old interfaces, its argmax, and displayed regret.
- **Information set.** \(I_{it}=\bigl(q_i,b_i,\ \{f_{is},y_{is},\pi_{is},d_{is},\rho_{is},c_{is},P_{is}\}_{s<t},\ \{x_s\}_{s\le t},\ p,\ (\kappa,\tau,\kappa_a,\tau_a)\bigr)\).
- **Stage payoff.** \(\pi_{it}=\text{margin}\cdot p_i\,Q_t s_{it}\) with \(u,s,Q\) as in §1 evaluated at
  the *realised* \(r_{it}\), never at \(\bar r\). Platform \(\text{GMV}_t=Q_t\sum_k p_k s_{kt}\).
- **Signals and transition.** Exactly steps 3–4 of §1: \(d_{it}\sim\text{Bin}(N_{\text{obs}},\theta_i)\),
  \(\rho\mid d\sim\text{Bin}(d,\phi)\), \(c_{it}\sim\text{Bin}(N_a,\psi_i)\), then
  \(r_{i,t+1}=\text{clip}(r_{it}+\eta_r(1-r_{it})-P_{it},0,1)\). Conditional on the action profile the
  \(m\) signal draws are independent across merchants, so the kernel factorises:
  \(K(x'\mid x,f)=\prod_k K_k(r_k'\mid r_k,f_k)\).
- **Discounting.** None. The objective put to the merchants is total profit.

> **A rival's reputation is a censored signal of that rival's action.** Because
> \(r'=\text{clip}(r+\eta_r(1-r)-P,0,1)\) is invertible off the clip, a merchant observing
> \(r_{k,t}\) and \(r_{k,t+1}\) recovers \(P_{kt}\) exactly whenever neither bound binds, and hence a
> noisy signal of \(f_{kt}\) through \(d_{kt}\). Rivals' actions are unobserved but they are **not**
> uninformed-about: the published reputation vector leaks them, censored at the clip. §3's Definition
> 1 says beliefs are Bayesian "wherever possible", and this is what makes that clause non-empty.

### 2.2 \(\Gamma_\delta\) — the game the merchants are told they are in

Identical primitives, but the horizon is infinite and payoffs are discounted by \(\delta\in(0,1)\).
This is not an alternative modelling choice; it is what the prompt says. Gate **G-H4** forbids any
prompt from disclosing the horizon, and the rules block states that the market "continues for an
unknown number of further rounds (no final round is announced)" and asks for "total profit across the
ongoing market".

> **This gap is deliberate and it has to be reported.** The experiment truncates at 80 rounds a game
> its subjects believe has no announced end. Backward induction from a known last round is therefore
> the wrong model of the *subjects'* problem, and \(\Gamma_\delta\) is the right one; but \(\Gamma_{80}\)
> is what generates the data. Where the two disagree, both numbers are reported and the disagreement
> is the finding. \(\delta\) is not identified by the design — it is a free parameter of the analysis,
> so anything computed from \(\Gamma_\delta\) is reported over a range of \(\delta\), never at one
> flattering value.

### 2.3 \(\mathcal G(\theta,\kappa,\tau)\) — the restricted stationary surrogate

Holding a merchant's action fixed at \(f\), its reputation follows a Markov chain on the 31-point
grid with kernel induced by the signal distribution. Write \(\bar r(f;b,\kappa,\tau)\) for the
stationary mean of that chain, and define the **restricted stationary game**: players \(1..m\),
actions \(F\), payoffs

\[
V_i(f) = \text{margin}\cdot p_i\cdot Q_0 e^{-\lambda\bar f}\cdot s_i(f),
\qquad
u_k = \alpha\bigl(q_k+(1-q_k)f_k\bigr) + \beta\,\bar r(f_k;b_k) - \gamma p_k .
\]

\(\mathcal G\) is obtained from \(\Gamma_\delta\) by **three** restrictions, which are logically
independent and are worth separating because they fail for different reasons:

| | restriction | what would break it |
|---|---|---|
| **R1** | strategies are constant in \(t\) and in the state | build-then-exploit deviations |
| **R2** | payoffs are evaluated at the *stationary* reputation, not the transient path | slow mixing; 80 rounds too short |
| **R3** | \(u_k\) uses \(\bar r=\mathbb E[r]\) rather than \(\mathbb E[u_k(r)]\) — a plug-in, not an expectation | Jensen curvature of the logit share in \(r\) |

R3 is an inequality of known sign nowhere: \(s_i\) is convex in \(r_i\) below its inflection and
concave above, so \(\text{GMV}(\bar r)\) is neither an upper nor a lower bound on
\(\mathbb E[\text{GMV}(r)]\) a priori. R1 is the restriction Proposition 1 does not cover and §10
tests. R2 is quantified in §9.

> **Boundary.** \(\mathcal G\) is *not* the full dynamic game. It is the object the frozen corpus
> computed, so keeping it makes the old and new results commensurable, and every benchmark in §6 is a
> benchmark *of \(\mathcal G\)*. Claims about \(\Gamma_{80}\) or \(\Gamma_\delta\) are confined to
> Proposition 1 and to §§9–11, and are correspondingly weaker.

### 2.4 What is the sufficient state? — stated, not assumed

The reduction from \(\Gamma_\delta\) to \(\mathcal G\) is usually written as though merchant \(i\)'s
problem were a Markov decision process on its own reputation \(r_i\), 31 states. **It is not.** The
share

\[
s_i=\frac{e^{u_i(r_i,f_i)}}{e^{w_0}+\sum_k e^{u_k(r_k,f_k)}}
\]

has every rival's reputation in the denominator, so \(i\)'s stage payoff depends on the whole vector
\(x=(r_1,\dots,r_m)\) and the payoff-relevant state is \(R^m\), \(923{,}521\) points, not \(31\).
Two facts pull in opposite directions and both are needed:

1. \(i\)'s **transition** does not depend on \(x_{-i}\): \(d_i\) is drawn from \(b_i+c_sf_i\) alone.
   So \(x_{-i}\) is an exogenous, uncontrolled process from \(i\)'s point of view.
2. \(i\)'s **reward** is not separable in \((r_i,x_{-i})\), because they meet in the denominator.

Consequence, and it is the reason §10 is expensive rather than cheap: if rivals play *state-independent*
actions then \(x_{-i}\) is independent of \(r_i\), the expected-reward MDP on \(r_i\) alone reproduces
the correct **value** of any \(r_i\)-measurable strategy — but the supremum over \(r_i\)-measurable
strategies is only a **lower bound** on the supremum over \(x\)-measurable ones. A lower bound on the
deviation gain is worthless for certifying equilibrium: it can be zero when a profitable deviation
exists. §10 therefore solves the deviation problem on the full \(31^4\) state space, and the 31-state
reduction is retained only as a cross-check whose gap to the full solution measures how much
conditioning on rivals is worth.

The bound is one-sided, and it is worth being explicit about which side, because the answer turned
out to sit on the useful one. A lower bound cannot *certify* an equilibrium — \(\varepsilon_{2.5}=0\)
would say nothing. It can *refute* one: \(\varepsilon_{2.5}>0\) implies \(\varepsilon_3>0\)
immediately, since every \(r_i\)-measurable strategy is an \(x\)-measurable strategy. §10.3 finds it
strictly positive on every instance tested, so the refutation is carried by the cheap object and the
\(31^4\) program is left to answer the quantitative question §2.4 actually poses — *how much* is
conditioning on rivals worth — rather than the qualitative one.

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

All four are computed from a table of stationary reputations whose solver turns out to depend on its
own initial guess (§9.1). The dependence is worth \(-3.2\times10^{-4}\) on \(G^{\mathrm{FB}}\),
\(-3.8\times10^{-5}\) on \(G^{\mathrm{NP}}\) and **nothing at all** on
\(G^{\mathrm{SB}}_{\mathcal P,\text{unif}}\); the ratio below rises from 70.458% to 70.484% when it is
fixed. §9.2 gives the full table. The figures here are the published ones.

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
2. **The stationary restriction** (§2.3) confines merchants to constant actions evaluated at the
   stationary reputation distribution. It is three restrictions, not one, and they are now measured
   rather than flagged. **R2** (stationary rather than transient payoffs) costs 0.24% of GMV and is a
   start-up effect that is over by round 20 (§9.4). **R3** (plug-in \(\bar r\) rather than
   \(\mathbb E[u(r)]\)) costs \(1.7\times10^{-4}\) of GMV, but it changes an incentive conclusion:
   the profile Proposition 1′ calls the unique pure Nash equilibrium is **not** a Nash equilibrium of
   the expected-payoff stationary game on 12 of 60 seeds, with gains up to 0.22% of profit (§9.5).
   **R1** (constant, state-independent actions) is the one that excludes build-then-exploit, and §10
   measures it: it is the largest of the three by a wide margin. Merely scoring the *same* constant
   strategies correctly already flips the best action on 24.6% of merchant-instances at
   \(\delta=0.95\) (§10.2), and the constant-action equilibrium that results reaches **62.9% of
   first best against the 70.4% the corpus reports** (§10.5). Allowing the action to depend on the
   merchant's own reputation raises the deviation gain further on **223 of 240** instances (§10.3),
   and the 17 that survive are *exactly* the instances where \(f^*_i\) already sits at the fabrication
   corner and there is nothing above it to deviate to. \(f^*\) is an equilibrium of \(\mathcal G\); it
   is not one of \(\Gamma_\delta\). One caveat cuts against the framing rather than the finding: the
   deviations R1 excludes are **not** the ones it was written to exclude. Build-then-exploit does not
   occur in any instance — 0 of 240 in sample, 0 of 120 out of sample; what occurs is its mirror
   image, total fabrication at the *lowest* reputations and near-compliance above a threshold
   (§10.3), and that shape survives when the rivals' reputations are made visible too (§10.4). The
   restriction is doing real work, but not the work it was justified by.
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
| "The stationary solver has a well-defined answer" | 315 kernels have **three** recurrent classes; the reported \(\bar r\) is chosen by the initial guess | §9.1 |
| "Slow mixing invalidates the stationary payoffs" | worst TV to \(\pi\) at 80 rounds is 0.968, yet the mean is right to \(3.2\times10^{-3}\) | §9.2 |
| "\(f^*\) is *the* unique pure Nash equilibrium" | it is one of \(\mathcal G\); under expectations it is only a 0.22%-equilibrium, failing on 12/60 seeds | §9.5 |
| "\(f^*\) is the equilibrium of the game the runner plays" | \(f^*\) is enumerated from \(b\) **snapped to `B_GRID`**; the runner and the dynamic solver use the exact draw. The snap is \(\le4.9\times10^{-3}\), but an argmax is not Lipschitz and the two processes name different failing markets | §10.2 |
| "A restriction that barely moves GMV barely moves incentives" | R2 costs 0.24% of GMV and is over by round 20, yet it flips the best constant action on **24.6% of merchant-instances** at \(\delta=0.95\) | §9.4, §10.2 |
| "The stationary profile is what merchants would settle on" | the constant-action equilibrium under exact discounted payoffs differs from \(f^*\) on **41/60 seeds** and delivers **62.9%** of first best, not 70.4% | §10.5 |
| "Stopping value iteration when \(\varepsilon=V-P\) has settled is safe" | \(V\) and \(P\) share the \(c/(1-\delta)\) level, which converges at rate \(\delta\) while \(\varepsilon\) converges at the mixing rate; the level was short by \(3.4\times10^{-6}\) | §10.1 |
| "\(f^*\) at least survives the dynamic objection on *some* markets" | the 17 of 240 instances with zero own-state deviation gain are **exactly** the 17 with \(f^*_i=1\) — the sets coincide, in sample and held out. \(f^*\) survives only where it already prescribes total fabrication | §10.3 |
| "\(\Gamma_{80}\)'s equilibrium failure is a statement about the market" | the best switch round is in the last five on 224/240; confined to the first half the largest gain over all 240 instances is **0.0589%**. It is backward induction from a terminal round the merchants are never told about (G-H4) | §10.3 |
| "The state-contingent deviation is build-then-exploit — that is what R1 excludes" | it never occurs: 0 of 240 in sample, 0 of 120 out of sample. The own-reputation optimum sits at *total* fabrication over a block of the **lowest** reputations and returns to near \(f^*\) above a threshold — collapse, not cash-in — and in the **weakest** of the 59 collapse instances the corner still beats \(f^*\) by 23.6% of the state's whole action spread, so it is a preference and not round-off | §10.3, §10.4 |
| "A positive cash-in slope means cash-in" | 163 of 240 maps have a positive fitted slope and **none of those 163** exploits at high reputation — the slope is a rise *toward* \(f^*\) that stops there, fitted through a non-monotone argmax | §10.3 |

---

## 9. What the stationary surrogate costs: R2 and R3, measured **[computational]**

§2.3 listed three restrictions separating \(\mathcal G\) from \(\Gamma_\delta\). This section
measures two of them — R2 (stationary rather than transient evaluation) and R3 (plug-in \(\bar r\)
rather than \(\mathbb E[u(r)]\)) — and the ergodicity precondition that both silently require. R1 is
§10. Everything here is exact linear algebra on the model's own kernels; no simulation, no LLM.

The object under test is a single function. `ha_model._stationary_mean` builds the \(31\times31\)
reputation kernel, runs 300 power iterations from the **uniform** distribution, returns
\(\pi^\top r\), and discards the kernel. It never checks that the limit exists, that it is unique,
that 300 steps reach it, or that the uniform start is the right one. Each of those is a claim, and
each is now tested. `code/ha_dynamics_audit.py` rebuilds the identical kernel — same
\(r'=\text{clip}(r+\eta_r(1-r)-P,0,1)\), same `np.round` snap to the grid — and audits it.

### 9.1 The ergodicity audit: three of four assumptions fail somewhere

Over all \(57\times15\times21=17{,}955\) kernels the class of policies can produce:

| Property `_stationary_mean` assumes | Holds in |
|---|---|
| a unique stationary distribution | **17,640 / 17,955** |
| irreducibility on the 31-point grid | **0 / 17,955** |
| aperiodicity | 17,955 / 17,955 |
| 300 power iterations suffice (TV \(\le10^{-9}\)) | **15,222 / 17,955** |

Irreducibility fails everywhere and harmlessly: reputation drifts up by \(\eta_r(1-r)\) every round,
so low-\(r\) states are transient by construction (up to 28 of the 31 states are transient). That is
a property of the model, not an error.

The other two failures are real. **The 315 kernels with \(\kappa=0\) have three recurrent classes,
not one.** With no penalty the map \(r\mapsto\text{clip}(r+0.2(1-r))\) composed with the `np.round`
snap has three absorbing fixed points, at grid indices 28, 29, 30 — \(r\in\{0.9333,0.9667,1\}\) —
and which one the chain lands in is decided entirely by where it starts. The model's uniform start
splits mass \(\{28\!:\!0.9355,\ 29\!:\!0.0323,\ 30\!:\!0.0323\}\) and reports \(\bar r=0.936559\).
The experiment's own initial condition \(r_0=0.5\) follows the deterministic orbit
\(15\to18\to20\to22\to24\to25\to26\to27\to28\) and absorbs at \(\bar r=0.933333\). **These are
different numbers for the same object, and the model reports the one that does not correspond to its
own runner.**

Convergence at 300 iterations fails for 2,733 kernels. The diagnostics look alarming: the largest
non-degenerate second eigenvalue modulus implies a relaxation time of \(9.15\times10^{11}\) rounds;
the Dobrushin ergodic coefficient reaches 1.0, so it certifies no contraction anywhere in the class;
and the worst total-variation distance to \(\pi\) after 80 rounds is 0.968 — visually, no mixing at
all. (467 kernels have SLEM within \(10^{-9}\) of 1, making the unit eigenspace numerically
degenerate; `np.linalg.eig` returns an arbitrary vector from it, so those are excluded from the
eigenvector cross-check rather than allowed to produce a nonsense statistic.)

### 9.2 Why none of that propagates: the containment bound

Total variation is payoff-blind, and here that is the whole story. The slowly-mixing directions are
transitions among **adjacent near-absorbing states at the top of the grid**, and the payoff
functional is Lipschitz in \(r\). The bound that contains the entire §9.1 pathology, over every one
of the 17,955 kernels:

\[
\max\bigl|\bar r(\text{uniform start})-\bar r(r_0=0.5)\bigr| \;=\; 0.003226
\]

— one tenth of a single grid spacing \(1/30=0.0333\), attained at the worst \(\kappa=0\) kernel
described above. A chain can be arbitrarily far from \(\pi\) in TV and still give the right mean when
the mass it has not yet moved sits on states that differ in \(r\) by \(1/30\).

Restricting to the kernels that actually carry a result tightens this further:

| Kernel set (per policy) | unique \(\pi\) | converged in 300 | \(\max\|\bar r\) gap\(\|\) |
|---|---|---|---|
| all \((b,f)\), \(P_{\mathrm{GMV}}\) | 315/315 | 270/315 | 0.003226 |
| all \((b,f)\), \(P_{\mathrm{robust}}\) | 315/315 | 240/315 | 0.003226 |
| all \((b,f)\), \(P^{\mathrm{SB}}_{\text{unif}}\) | 315/315 | 240/315 | 0.003226 |
| **on the equilibrium path**, \(P_{\mathrm{GMV}}\) | 240/240 | 240/240 | **0** |
| **on the equilibrium path**, \(P_{\mathrm{robust}}\) | 240/240 | 116/240 | \(9.0\times10^{-6}\) |
| **on the equilibrium path**, \(P^{\mathrm{SB}}_{\text{unif}}\) | 240/240 | 230/240 | **0** |

Multiple recurrent classes occur **only** at \(\kappa=0\), which is not any named policy: at all three
named policies the stationary distribution is unique on every one of the 315 kernels, so the one
failure that is qualitative rather than quantitative never touches a reported number.

Non-convergence, by contrast, is common even on the equilibrium path — at \(P_{\mathrm{robust}}\)
**less than half** the equilibrium-path kernels have converged when `_stationary_mean` stops
(116/240), with SLEM up to 0.980. And it still does not matter: the resulting error in \(\bar r\) is
\(9.0\times10^{-6}\) there and exactly 0 at the other two policies. The 300-iteration cutoff is not
justified by the code, and happens to be enough.

**Verdict on the precondition: the assumption is false as stated and the error it causes is bounded
by a tenth of a grid point** — \(3.2\times10^{-3}\) anywhere in the class, \(9.0\times10^{-6}\) on
any equilibrium path.

#### What it costs the published benchmarks

The fix is free: start the power iteration at the runner's own \(r_0=0.5\) instead of at uniform.
Doing that and rebuilding §6 from scratch — all 64 policies, all 60 seeds, \(21^4\) profiles
enumerated per pair, twice:

| | published (uniform start) | corrected (\(r_0=0.5\)) | shift |
|---|---|---|---|
| \(G^{\mathrm{FB}}\) | 0.897803 | 0.897481 | \(-3.22\times10^{-4}\) |
| \(G^{\mathrm{NP}}\) | 0.349071 | 0.349033 | \(-3.78\times10^{-5}\) |
| \(G^{\mathrm{SB}}_{\mathcal P,\text{unif}}\) | 0.6325777 | 0.6325777 | \(-2.5\times10^{-13}\) |
| \(G^{\mathrm{SB}}/G^{\mathrm{FB}}\) | 70.458% | 70.484% | \(+0.025\) pp |
| implementation power | 0.51666 | 0.51699 | \(+3.4\times10^{-4}\) |
| argmax policy | \((2.0,0.30)\) | \((2.0,0.30)\) | unchanged |

The pattern is exactly what §9.1 predicts, which is the point of reporting it. The only benchmark
that moves at all is the one whose maximisation ranges over \(\kappa=0\) — the first best — and it
moves **down**, because the uniform start over-weights the two highest absorbing states and so
reports a reputation that is too generous by \(0.0032\) precisely where the penalty is switched off.
\(G^{\mathrm{SB}}_{\mathcal P,\text{unif}}\) is attained at \(\kappa=2.0\), where \(\pi\) is unique,
and does not move in thirteen digits.

**So the headline 70.5% was, if anything, a shade pessimistic**: correcting the defect raises it by
0.025 percentage points, and nothing else in §6 changes. The published numbers are retained, with
this table as the statement of what the initialisation is worth. `offline_tests/recompute_dynamics_audit.py`
checks not only the magnitudes but the **sign**, against the mechanism: a shift of the wrong sign, or
any movement in the second best, would mean the diagnosis was wrong even if the arithmetic
reproduced.

### 9.3 R3 — the plug-in is not an expectation, and the sign is not what one would guess

\(\text{GMV}(\bar r)\) versus \(\mathbb E[\text{GMV}(r)]\) under the **joint** stationary law, 60
seeds, exact:

| | \(P_{\mathrm{GMV}}\) | \(P_{\mathrm{robust}}\) | \(P^{\mathrm{SB}}_{\text{unif}}\) |
|---|---|---|---|
| \(\text{GMV}(\bar r)\) | 0.614007 | 0.615401 | **0.632578** |
| \(\mathbb E[\text{GMV}(r)]\) | 0.614053 | 0.615541 | 0.632684 |
| mean relative error | \(-0.0073\%\) | \(-0.0227\%\) | \(-0.0169\%\) |
| worst \(\|\)relative error\(\|\) | 0.0217% | 0.0724% | 0.0783% |
| seeds where plug-in *understates* | 56/60 | 53/60 | 51/60 |

The plug-in error is one part in \(10^4\) and, contrary to §2.3's warning that the sign is
indeterminate a priori, it is **negative on 51–56 of 60 seeds**: the equilibrium reputations sit in
the region where the logit share is convex in \(r\), so Jensen runs one way in practice even though
it need not. The direction is conservative — the paper's headline understates true expected GMV —
but the magnitude is too small to matter either way.

The joint stationary law is a product measure to machine precision: the residual
\(\|\Pi-\bigotimes_k\Pi_k\|_1\) is \(\le6.7\times10^{-11}\) on every seed and policy. That is not a
finding, it is a check on the code: the kernel factorises across merchants because complaint draws
are independent given actions, so independence here is a theorem, and the residual confirms the
implementation matches it.

**Consistency check.** The plug-in mean at \((\kappa,\tau)=(2.0,0.30)\) is 0.6325777, reproducing the
published \(G^{\mathrm{SB}}_{\mathcal P,\text{unif}}=0.632578\) of §6 to seven digits by an
independent code path.

### 9.4 R2 — the transient is a start-up effect, and it is over by round 20

The 80-round path from the runner's own \(r_0=0.5\), compared with the stationary value the paper
reports. Mean relative gap across 60 seeds, by round:

| round | 1 | 5 | 10 | 20 | 40 | 80 |
|---|---|---|---|---|---|---|
| \(P_{\mathrm{GMV}}\) | \(-4.35\%\) | \(-1.74\%\) | \(-0.52\%\) | \(-0.031\%\) | \(-0.000\%\) | \(-0.000\%\) |
| \(P_{\mathrm{robust}}\) | \(-4.25\%\) | \(-1.53\%\) | \(-0.11\%\) | \(-0.000\%\) | 0 | 0 |
| \(P^{\mathrm{SB}}_{\text{unif}}\) | \(-4.99\%\) | \(-1.61\%\) | \(-0.21\%\) | \(-0.002\%\) | 0 | 0 |

The worst single seed is \(-16.9\%\) at round 1 and \(-0.03\%\) by round 20. Averaged over all 80
rounds the path lies **0.22–0.26% below** the stationary value (worst seed 0.88%), and that entire
deficit is bought in the first ten rounds, while reputation climbs from 0.5 to its stationary level.

So R2 costs about a quarter of one percent of GMV, it is a level shift from the initial condition and
not a mixing failure, and §9.1's terrifying relaxation times never reach the payoff. **Reporting
stationary rather than 80-round-average GMV overstates by 0.24%**; where that matters the path figure
is now available.

### 9.5 What does **not** survive: the enumerated equilibrium is not an equilibrium under expectations

R3 is negligible for GMV *levels*. It is not negligible for *incentives*, and this is the finding
that changes a claim.

Proposition 1′ enumerates \(21^4\) profiles and reports a unique pure Nash equilibrium of
\(\mathcal G\) — a game whose payoffs are the plug-in \(V_i(\bar r)\). Rescoring exploitability with
\(\mathbb E_\Pi[\pi_i(r)]\) instead, moving the deviator's own stationary law with its deviation and
holding rivals' laws fixed:

| | \(P_{\mathrm{GMV}}\) | \(P_{\mathrm{robust}}\) | \(P^{\mathrm{SB}}_{\text{unif}}\) |
|---|---|---|---|
| still a Nash equilibrium under \(\mathbb E[\cdot]\) | 51/60 | 47/60 | **48/60** |
| worst deviation gain (% of profit) | 0.103% | 0.477% | 0.217% |
| mean gain among the failures | 0.029% | 0.205% | 0.109% |

**On 12 of 60 seeds at the headline policy, the profile the paper calls "the unique pure Nash
equilibrium" is not a Nash equilibrium of the expected-payoff stationary game.** The gains are small
— at most a fifth of a percent of profit — but they are strictly positive, and "unique pure Nash
equilibrium" is a statement that admits no exceptions. The correct statement is that \(f^*\) is the
unique pure NE **of \(\mathcal G\)**, and is a \(0.22\%\)-equilibrium of the stationary game scored
with expectations. Proposition 1′ is amended accordingly in §11.

This is a strictly weaker failure than R1's: it says the plug-in mis-ranks deviations that are nearly
tied, not that a fundamentally different strategy class wins. §10 tests the latter.

---

## 10. R1: what dropping constant strategies is worth **[computational]**

R1 is the last and largest of the three restrictions. \(\mathcal G\) lets merchant \(i\) choose one
number \(f_i\) and holds it for ever; \(\Gamma_\delta\) lets it choose a different action after every
history. The gap between those is where "build a reputation, then cash it in" lives, and it is the
one restriction Proposition 1 does not cover.

### 10.1 Four questions, not one

It is tempting to run the dynamic program, get a number, and call it *the* answer. That conflates two
independent axes, and the frozen corpus differs from \(\Gamma_\delta\) along both:

* the **scoring** — stationary plug-in \(V_i(\bar r)\), or the true discounted payoff from the actual
  initial condition \(r_0=0.5\);
* the **strategy class** — constant, or time-varying, or state-contingent.

§9 moved the first axis with the class held fixed. §10 moves the second with the scoring held
correct. That gives a nested ladder, and each rung is a strictly larger set of deviations, so the
exploitability \(\varepsilon\) is non-decreasing along it:

| | class of deviations available to \(i\) | scoring | isolates |
|---|---|---|---|
| \(\mathcal C_0\) | constant \(f_i\) | stationary, plug-in | the frozen corpus — \(\mathcal G\) |
| \(\mathcal C_1\) | constant \(f_i\) | discounted from \(x_0\), exact | **R2 alone** |
| \(\mathcal C_2\) | open-loop \(\{f_i^t\}_{t}\), state-independent | exact | + timing |
| \(\mathcal C_{2.5}\) | closed-loop on **own** reputation, \(f_i^t(r_i)\) | exact | + own-state contingency |
| \(\mathcal C_3\) | closed-loop \(f_i^t(x)\), \(x\in R^m\) | exact | + rival contingency = \(\Gamma_\delta\) |

\(\mathcal C_0\) and \(\mathcal C_1\) share a strategy set and differ only in the payoff functional;
\(\mathcal C_1\subset\mathcal C_2\subset\mathcal C_{2.5}\subset\mathcal C_3\) are genuine
enlargements. Reading the ladder upward is what makes the result interpretable: if \(\varepsilon\) is
already large at \(\mathcal C_1\), the surrogate's failure has nothing to do with dynamic strategy at
all and everything to do with how it scores; if \(\varepsilon\) only appears at \(\mathcal C_3\), the
failure is genuinely state-contingent and needs the rivals to be visible — the kind of thing R1 was
written to flag.

Where \(\varepsilon\) first becomes positive is not, however, the same question as *what the profitable
deviation does*, and it is worth separating them before the numbers arrive. A rung of the ladder is a
set of strategies; knowing that the optimum leaves that set says nothing about the direction it leaves
in. Build-then-exploit is one shape a state-contingent deviation could have — fabricate little while
the reputation stock is small, spend it once it is large — and its mirror image is another: hold
fabrication down *because* the stock is worth protecting, and abandon that restraint once the stock is
gone. The two are opposite in sign and mean opposite things for policy, and both are consistent with
any given \(\varepsilon_{2.5}>0\). §10.3's last block therefore reports the argmax as well as the
gain, and tests both hypotheses symmetrically rather than looking only for the one R1 names.

\(\mathcal C_{2.5}\) is the rung §2.4 promised. Conditioning on \(r_i\) alone is a 31-state MDP, so
it is solvable in milliseconds by value iteration on the same tail tables §10.2 already builds; and
because it is sandwiched — every open-loop deviation is available to it, and it is available to the
full \(31^4\) program — it converts the two cheap rungs from *suggestive* lower bounds into a
**rigorous bracket** around \(\mathcal C_3\). That matters for a practical reason: it means the
verdict on whether \(f^*\) survives R1 does not have to wait for, or depend on, the expensive solver.
If \(\varepsilon_{2.5}>0\) then \(\varepsilon_3>0\), full stop.

The methods are deliberately unrelated. \(\mathcal C_1\), \(\mathcal C_2\) and \(\mathcal C_{2.5}\)
are built from forward simulation of a product law and a \(31\times31\) resolvent, and are computed
in `offline_tests/recompute_dynamic_dp.py`, which imports no solver code; \(\mathcal C_3\) needs
backward induction on all \(31^4=923{,}521\) states (§2.4 explains why the 31-state reduction cannot
be *substituted* — only bracketed) and is `code/ha_dynamic_dp.py`. So the cheap rungs are independent
lower bounds on the expensive one, and §10.4 checks that the solver's \(\varepsilon\) is at least as
large as the certificates that need no solver. Where they disagree in sign, the cheap rungs win: they
are the ones with a closed-form tail.

Two facts make the cheap rungs exact rather than approximate. Merchant \(i\)'s action never enters a
rival's kernel, so under any open-loop strategy the four reputations stay independent and the joint
law is a product of marginals at every round; and \(i\)'s payoff depends on rivals only through the
scalar \(Z=\sum_{k\ne i}e^{u_k}\), whose law is the exact convolution of three 31-atom laws — 29,791
atoms, summed, not sampled. Once the rivals' law has settled, the tail of any constant-action value
is the \(31\times31\) resolvent \((I-\delta T_i(a))^{-1}\), which removes the horizon from the
calculation: \(\delta=0.999\) costs no more than \(\delta=0.9\).

That last step carries the one assumption in §§10.2–10.3, and it is load-bearing exactly where the
answer matters most. The transient is summed explicitly for \(T_0=250\) rounds and the resolvent
handles the rest, so the tail's weight is \(\delta^{T_0}\): negligible at \(\delta=0.9\)
(\(\sim10^{-11}\)), but **97.5% of the whole value at \(\delta=0.9999\)**. The patient limit is
therefore almost entirely a statement about the rivals' settled law \(W_{T_0}\), and after §9.1 —
where 2,733 kernels had not converged in 300 iterations — a one-step residual is not by itself
adequate evidence that it has settled. It is reported
(\(\lVert W_{T_0}-W_{T_0-1}\rVert_1\)), but the real check is §10.2's cross-validation: as
\(\delta\to1\) the discounted criterion collapses onto the long-run average, which is precisely the
stationary expected payoff §9.5 computed by power-iterating stationary laws and contracting them
against the payoff tensor. Two unrelated routes to the same limit must name the same failing markets,
and that is asserted as a check rather than hoped for.

One arithmetic point, because the 60-seed block caught it and a 3-seed probe had not. The reported
quantity is \(\varepsilon=V-P\), so it is tempting to iterate both and stop when \(\varepsilon\) has
settled. That is wrong. \(V\) and \(P\) share the level term \(\sim c/(1-\delta)\), which converges
at rate \(\delta\); their *difference* converges at the chain's mixing rate, which in these kernels
is far faster. Stopping on \(\varepsilon\) therefore returns a \(P\) whose **level** is still short
by \(O(\delta^n/(1-\delta))\) — invisible in \(\varepsilon\), and \(P\) is the denominator of every
relative figure quoted below. Measured against the resolvent, which has the level in closed form,
the iterated \(P\) was wrong by \(3.4\times10^{-6}\) in value units. Both tails are consequently
*solved*, not iterated: the policy value is one \(31\times31\) linear solve and the optimum is
Howard policy iteration, after which the two routes agree to \(1.8\times10^{-15}\). The guard that
caught this is retained at \(10^{-9}\).

### 10.2 \(\mathcal C_1\): \(f^*\) is not an equilibrium, it is a *patient* equilibrium

The first rung changes nothing about the strategy class. Merchant \(i\) still picks one number and
holds it for ever; only the scoring moves, from \(V_i(\bar r)\) to the exact discounted value from
the initial condition the experiment actually starts at, \(r_0=0.5\) for everyone. Any failure here
is **R2 and nothing else** — it cannot be build-then-exploit, because a constant action builds
nothing.

Sweeping \(\delta\) over a twelve-point grid, on all 60 seeds and all four merchants — 240
merchant-instances:

| \(\delta\) | instances preferring a *different constant* action | of which prefer \(f=1\) |
|---|---|---|
| 0.80 | 180 / 240 (75.0%) | 121 |
| 0.85 | 141 / 240 (58.8%) | 96 |
| 0.90 | 103 / 240 (42.9%) | 74 |
| 0.925 | 86 / 240 (35.8%) | 61 |
| **0.95** | **59 / 240 (24.6%)** | **40** |
| 0.96 | 51 / 240 (21.2%) | 31 |
| 0.97 | 49 / 240 (20.4%) | 29 |
| 0.98 | 39 / 240 (16.2%) | 19 |
| 0.99 | 25 / 240 (10.4%) | 5 |
| 0.995 | 22 / 240 (9.2%) | 1 |
| 0.999 | 20 / 240 (8.3%) | 0 |
| 0.9999 | 20 / 240 (8.3%) | 0 |

Three separate claims live in that table.

**\(f^*\) is not an equilibrium of the constant-strategy game unless merchants are patient.** The
median instance needs \(\delta^*=0.90\) before \(f^*\) becomes its best constant action; the worst
needs \(0.999\); and **20 of 240 never prefer \(f^*\) anywhere on the grid**, including at
\(\delta=0.9999\). The largest gain from a constant deviation anywhere on the grid is **48.2% of the
merchant's own discounted profit**. None of this is a plug-in artefact: it is the same strategy class
the surrogate optimises over, scored correctly.

**At impatient \(\delta\) the deviation is to the corner, not to the margin.** At \(\delta=0.90\),
74 of the 103 deviating instances want \(f=1\) — total fabrication. Reputation is a stock that takes
rounds to burn; a merchant discounting at 0.90 collects the demand now and is gone before the stock
matters. As \(\delta\to1\) the corner deviations disappear first (5 at \(\delta=0.99\), **none at
\(\delta=0.999\)**) and what remains is a residue of near-tied mis-rankings — the same phenomenon
§9.5 found by an unrelated route, arriving here as the \(\delta\to1\) limit of a different criterion.

**R2 is small for GMV and decisive for incentives, and §9.4 is not contradicted by this.** §9.4
measured R2 at 0.24% of GMV and called it a start-up effect over by round 20. That is a statement
about *levels*, and it stands. The table above is a statement about *rankings*: the same transient
that moves the level by a quarter of a percent moves the argmax on a quarter of all instances at the
headline \(\delta\). A restriction can be negligible in one currency and decisive in another. R2 is.

#### The \(b\)-discretisation: \(f^*\) solves a game the runner never plays

Chasing the \(\delta\to1\) cross-check above turned up a defect with nothing to do with dynamics.
`rbar_grid()` is tabulated on a grid of base complaint rates `B_GRID`, and `pure_nash` enumerates
\(f^*\) from `rbar_grid()[mkt.bidx]` — each merchant's drawn \(b_i\) **snapped to the nearest grid
point**. But `ha_model.sample_signal`, which the runner executes, and `ha_dynamic_dp.Case`, which
§10.4 solves, both use the exact draw \(b_i\). Those are different games. The snap is small — at most
\(4.9\times10^{-3}\) in \(b\) — but \(f^*\) is defined by an argmax over a 21-point grid, and an
argmax is not Lipschitz.

The consequence is not hypothetical: the two reputation processes **disagree about which markets
fail** in the patient limit. Scored on the bucketed \(b\), the \(\delta\to1\) criterion reproduces
§9.5's failing seed list exactly; scored on the exact \(b\) it does not, and neither list contains
the other. Both computations are correct — they answer the question for two different games, one of
which is enumerated and the other of which is run. It is recorded as a counterexample in §8 rather
than repaired, because repairing it means re-enumerating every benchmark in the frozen corpus.

### 10.3 \(\mathcal C_2\) and \(\mathcal C_{2.5}\): the horizon is an artefact, the state-contingency is not

Two questions have to be kept apart here. \(\Gamma_{80}\) has a last round, so *something* must fail
in it — at \(t=80\) reputation has no future and nothing restrains fabrication. That is a property of
the truncation. The question that matters is whether anything survives when the horizon is removed.

**\(\Gamma_{80}\) first, and it is mostly the end-game.** The open-loop one-shot certificate
(\(\mathcal C_2\): hold \(f^*\), switch to some other action at one round, revert) is strictly
positive on 223 of the 240 merchant-instances, with a largest gain of **2.00%** of the merchant's own
80-round profit. But the best switch round lies in the **last five rounds on 224 of 240** instances,
and if the switch is confined to the first half of the horizon the largest gain over all 240 collapses
to **0.0589%** and the mean to **0.0118%** — factors of 34 and 170. Lifting to \(\mathcal C_{2.5}\)
(own-reputation-contingent, still undiscounted, still 80 rounds) raises the maximum to **3.23%** at
\(x_0\) and **4.42%** over own reputation: larger, same shape. So \(\Gamma_{80}\)'s violation is
overwhelmingly terminal. That is exactly what gate G-H4 (§2.1) exists to contain — merchants are never
told the horizon, so the end-game is not available to them — and it is why the honest \(\Gamma_{80}\)
number to quote a platform is the first-half one, which is under a tenth of a percent.

**\(\Gamma_\delta\), where the horizon is gone, is a different story.** At \(\delta=0.95\), conditioning
only on the merchant's *own* reputation:

| | instances with a strictly positive gain at \(x_0\) | median | mean | 90th pct | max at \(x_0\) | max over own \(r_i\) |
|---|---|---|---|---|---|---|
| \(\mathcal C_1\) — constant | 59 / 240 | — | — | — | — | — |
| \(\mathcal C_{2.5}\) — \(f_i^t(r_i)\), \(\Gamma_{80}\) | 223 / 240 | 1.22% | 1.30% | — | 3.23% | 4.42% |
| \(\mathcal C_{2.5}\) — \(f_i^t(r_i)\), \(\Gamma_\delta\) | **223 / 240** | 0.203% | 1.30% | 6.15% | **13.47%** | **21.53%** |

and those 223 instances are spread over **all 60 of 60 seeds**. Conditioning on one scalar — the
merchant's own stock — takes the count from 59 to 223. The 164 instances added are precisely the ones
for which no constant action beats \(f^*\) but a contingent one does: the deviation that breaks the
profile is available *only* to a merchant who watches its own reputation, and R1 is what hides it.
It is tempting to go one step further and say that, since the only thing conditioned on is the
reputation stock, this must be build-then-exploit in its minimal form. That step does not follow and
the argmax refutes it — see below. Seed 70000
is the clean illustration: at \(\delta=0.95\) not one of its four merchants has a profitable constant
deviation, and all four have a strictly positive own-reputation-contingent one (0.064%, 0.369%,
0.276%, 0.177% at \(x_0\); up to 6.94% at the best own-reputation state).

**The 17 survivors are exactly the corner.** Every instance whose gain is exactly zero has
\(f^*_i=20\) — the top of the fabrication grid — and every instance with \(f^*_i=20\) has gain exactly
zero. The two sets coincide, in \(\mathcal C_2\) and in both \(\mathcal C_{2.5}\) columns. The reason
is not subtle: there is nothing above total fabrication to deviate to, and deviating downward does not
pay. So \(f^*\) survives R1 on precisely the instances where it already prescribes maximal fabrication
— that is, where its survival is worth nothing to the platform. Put the other way round: **on every
instance where the surrogate's prediction is interesting, it is wrong.**

**The bracket is checked, not assumed.** Three inequalities hold by construction, and each pair is
computed by unrelated arithmetic, so a violation would mean one of the two routes is broken rather
than that the theory is:

| inequality | worst over 240 instances |
|---|---|
| \(\varepsilon_{2.5}\ge\) best open-loop one-shot gain, \(\Gamma_{80}\) | \(-1.5\times10^{-14}\) |
| \(\varepsilon_{2.5}\ge\) best constant-action gain, \(\Gamma_\delta\) | \(-4.4\times10^{-15}\) |
| \(V_i(f^*)\) by resolvent vs by reduced-state recursion | \(4.0\times10^{-15}\) |

All three pass at round-off. Note the first is *tight*: on the worst instance the own-reputation
optimum ties the single best one-shot switch exactly, so the bracket is not vacuously slack somewhere
— the two routes meet.

**What this settles without the solver.** Every \((t,r_i)\)-measurable strategy is
\((t,x)\)-measurable, so \(\varepsilon_3\ge\varepsilon_{2.5}\) state by state. \(\varepsilon_{2.5}>0\)
on 223 of 240 instances therefore proves that \(f^*\) is **not** a subgame-perfect equilibrium of
\(\Gamma_\delta\) at \(\delta=0.95\). §10.4's dynamic program cannot overturn that; it can only report
how much *more* is available from watching rivals as well. This is the reason the verdict in §11 does
not rest on the expensive computation.

#### What the deviation actually does — and it is not build-then-exploit

Everything above is a statement about the *size* of \(\varepsilon_{2.5}\). R1 was written to exclude a
specific *shape*, and a gain of any size is consistent with a shape R1 never contemplated, so the
argmax is worth as much as the gain. On the full \(31^4\) program the argmax is a \((21,31,31^3)\)
array that `ha_dynamic_dp.py` deletes on every sweep because retaining it would cost 155 MB per
iteration against roughly 6,000 iterations; on §2.4's reduction it is 31 integers, so the same
question can be put to every instance at once rather than to a handful named on a command line. Three
maps are recorded per instance — the stationary shape once the rivals have settled, the shape at the
experiment's own first round, and the same for \(\Gamma_{80}\) — in `cash_in_shape`.

**Build-then-exploit does not occur.** Not rarely: not once, in any of the three blocks, across all
240 instances (60 seeds × 4 merchants).

| | \(\Gamma_\delta\) tail | \(\Gamma_\delta\) round 1 | \(\Gamma_{80}\) round 1 |
|---|---|---|---|
| **build-then-exploit** \(\;\pi(r_{\min})<f^*_i<\pi(r_{\max})\) | **0 / 240** | **0 / 240** | **0 / 240** |
| non-decreasing and not flat | 0 / 240 | 0 / 240 | 0 / 240 |
| builds at low \(r\) \(\;\pi(r_{\min})<f^*_i\) | 164 / 240 | 127 / 240 | 184 / 240 |
| exploits at high \(r\) \(\;\pi(r_{\max})>f^*_i\) | 18 / 240 | 27 / 240 | 0 / 240 |
| — of those, also at the corner at \(r_{\min}\) | 18 of 18 | 26 of 27 | — |
| at the corner at \(r_{\min}\) \(\;\pi(r_{\min})=20>f^*_i\) | 59 / 240 | 81 / 240 | 38 / 240 |
| collapse-then-behave (corner region a lower interval, then not) | 56 / 240 | 81 / 240 | 38 / 240 |
| states at the corner, mean (max) | 6.3 (31) | 6.6 (31) | 2.3 (30) |
| fitted slope \(>0\) | 163 / 240 | 136 / 240 | 173 / 240 |
| — of those, exploit at high \(r\) | 0 of 163 | 1 of 136 | 0 of 173 |
| mean fitted slope | \(-0.081\) | \(-0.129\) | \(-0.038\) |
| weakest corner margin at \(r_{\min}\), share of the action spread | 23.6% | 7.1% | 2.1% |

(\(f^*_i\) is itself the corner in 17 of the 240 instances, where the corner rows are undefined and
are not counted; \(164+59+17=240\) in the tail block.)

The zero in the first row is not a knife-edge. The two halves of R1's story are *anticorrelated*: in
the tail block all 163 instances with a positive fitted slope build at low \(r\) and **none** of them
exploits at high \(r\), while all 18 that exploit at high \(r\) are at the corner at low \(r\) and
have a negative slope. There is no instance in which the map crosses \(f^*_i\) from below and stays
above it.

What the maps do instead splits into two regimes, and the larger one is the *opposite* of cash-in:
164 of 240 fabricate **less** than \(f^*_i\) at the lowest reputations and rise toward \(f^*_i\) as
the stock grows — a positive slope that never crosses \(f^*_i\). The smaller regime, 59 of 240, is
the sharper one: it sits at *total* fabrication over a contiguous block of the **lowest** reputations
and then drops back to within a notch of \(f^*_i\) above a threshold. Seed 70000, merchant 3,
\(f^*_i=8\), at \(\delta=0.95\) once the rivals have settled:

```
r index  0  1  2  3  4  5  6  7  8  9 10 | 11 12 13 14 15 ... 28 29 30
action  20 20 20 20 20 20 20 20 20 20 20 |  8  7  8  7  7 ...  9  8  8
```

The switch is at \(r\approx0.37\), and the two sides of it are not the same kind of object. At the
lowest reputation the corner action beats \(f^*_i\) by 0.0192 in value units, which is 72% of the
entire spread of \(Q(\cdot,r)\) across all 21 actions at that state; the margin decays by an order of
magnitude across the corner region and is exactly zero at the switch. Above the switch the map wobbles
between 7, 8 and 9 with margins below \(2\times10^{-4}\) — those are near-ties between adjacent
actions and should not be read as anything. The corner region is not, and not only in this instance:
the *weakest* of the 59 corner instances still beats \(f^*_i\) at its lowest reputation by 23.6% of
that state's entire action spread. The artefact records the margin state by state beside the map for
exactly this reason: an argmax reports a winner whether it won by a mile or by round-off, and at
reputations where demand is nearly nil it would be easy to mistake an arithmetic accident for a
mechanism.

The mechanism this suggests is not *build a reputation and cash it in*; it is **reputation as a
hostage**. A merchant holding a stock protects it, because the threshold penalty
\(P=\kappa\max(0,d/N-\tau)\) is bounded above while the demand a reputation earns is not, so the
continuation value of restraint exceeds the one-round gain from fabricating. A merchant that has
already lost the stock has nothing left to protect, the continuation value of restraint collapses,
and total fabrication becomes strictly optimal. The two halves of R1's own story do both appear in
the data — some maps fabricate less than \(f^*_i\) at low reputation, others fabricate more at high
reputation — but they never appear in the same instance, and never in the order R1 predicts.

Two further properties of the maps matter for how the result may be summarised. First, 216 of the 240
tail maps are monotone in neither direction (7 are flat, 17 non-increasing, none non-decreasing), so
the linear slope in the table is a regression through a jagged map and not a description of it; no
threshold rule in \(r_i\) summarises the optimum, which is itself an argument against reading any of
these deviations as a simple strategy a merchant could be said to "follow". Second, the sign of that
slope and the sign of the mechanism disagree, and this is the trap the table is laid out to spring:
**163 of 240 tail maps have a positive fitted slope**, and a paper that reported that number alone
would have reported cash-in for a population in which cash-in does not happen once. The slope is
positive because the map climbs *toward* \(f^*_i\) from below; it stops there. The mean slope is
negative because the 59 collapse instances fall by twelve grid points at a stroke. Neither statistic
survives contact with the argmax, and both are in the artefact so that the reader can see the
disagreement rather than take the summary on trust.

This is a genuine correction to how §10 was framed, not a refinement of it. \(\varepsilon_{2.5}>0\)
refutes the equilibrium claim either way, so the verdict in §11 is unchanged; but the *reason* the
profile fails is the opposite of the one R1 was written to guard against, and a mechanism designer
who patched the model against build-then-exploit would have patched the wrong thing. §10.4 asks
whether watching rivals changes the shape as well as the size.

#### Held out

The block above chooses the strategy class after having seen the markets, so it was rerun unchanged on
30 seeds that appear nowhere else in this package, `71000`–`71029` (120 merchant-instances):

| | in sample (70000–70059) | held out (71000–71029) |
|---|---|---|
| instances with a \(\mathcal C_{2.5}\) gain at \(x_0\), \(\Gamma_\delta\) | 223 / 240 | 112 / 120 |
| zero-gain instances that are the fabrication corner | 17 / 17 | 8 / 8 |
| max relative gain at \(x_0\) | 13.47% | 14.93% |
| max relative gain over own reputation | 21.53% | 22.86% |
| \(\Gamma_{80}\) best switch in the last five rounds | 224 / 240 | 112 / 120 |
| \(\Gamma_{80}\) max gain from a first-half switch | 0.0589% | 0.0647% |
| \(\mathcal C_1\): want a different constant action at \(\delta=0.95\) | 24.6% | 22.5% |
| \(b\)-bucketing flips the \(\delta\to1\) verdict | 23 / 240 | 5 / 120 |
| **build-then-exploit maps** (tail · round 1 · \(\Gamma_{80}\)) | **0 / 240** · 0 / 240 · 0 / 240 | **0 / 120** · 0 / 120 · 0 / 120 |
| non-decreasing and not flat (tail · round 1 · \(\Gamma_{80}\)) | 0 / 240 · 0 / 240 · 0 / 240 | 0 / 120 · 0 / 120 · 0 / 120 |
| at the corner at \(r_{\min}\) (tail · round 1 · \(\Gamma_{80}\)) | 59 / 240 · 81 / 240 · 38 / 240 | 17 / 120 · 31 / 120 · 14 / 120 |
| collapse-then-behave (tail · round 1 · \(\Gamma_{80}\)) | 56 / 240 · 81 / 240 · 38 / 240 | 12 / 120 · 31 / 120 · 13 / 120 |
| mean fitted slope (tail · round 1 · \(\Gamma_{80}\)) | \(-0.081\) · \(-0.129\) · \(-0.038\) | \(-0.029\) · \(-0.070\) · \(-0.029\) |
| weakest corner margin at \(r_{\min}\), \(\Gamma_\delta\) tail | 23.6% | 64.2% |

Every sign and every qualitative claim reproduces, including the corner law exactly; the magnitudes
move by about a percentage point. The shape rows are the ones that were *not* designed on these
seeds in any sense — the two hypotheses were written down and tested symmetrically before the
held-out block was run — and the zero survives: not one build-then-exploit map, and not one
non-decreasing map, in 120 further instances. The mix shifts (the corner regime is rarer out of
sample, 17 / 120 against 59 / 240, which is why the mean slope is nearer zero), so the *frequency* of
the collapse is a property of the seed block and should not be quoted as a constant; its
*existence*, and the absence of its opposite, are not. One check does **not** transfer, and the
artefact says so instead of passing quietly: the cross-validation against §9.5 compares against `ha_mixing_audit.json`, which
covers only the 70000-block, so out of sample it records `cross_validation_applicable: false` rather
than reporting a vacuous agreement over an empty intersection.

### 10.4 \(\mathcal C_3\): the full program, where rivals are watched too

Everything to this point bounds \(\varepsilon_3\) from below without ever computing it. §10.3's
reduction lets merchant \(i\) condition on its own reputation while the rivals are integrated out over
their settled law; \(\mathcal C_3\) lets it condition on the whole vector \(R^m\), which is
\(\Gamma_\delta\) itself. Because every \((t,r_i)\)-measurable strategy is \((t,x)\)-measurable, the
reduction can only understate. Two questions survive that argument and neither can be settled by it:
**how much** the extra conditioning is worth, and whether the *shape* of the optimum changes once
rivals are visible. The second is the one §10.3 explicitly could not answer, because a merchant
watching only its own stock cannot, by construction, run a strategy that waits for a rival to
stumble.

Answering either means solving the unreduced program: \(31^4=923{,}521\) states, 21 actions, value
iteration with the stopping rule applied to \(\varepsilon=V-P\) rather than to \(V\) (§10.3's
stopping-rule trap — the two share the level \(c/(1-\delta)\) and only their difference converges at
the mixing rate), per merchant, per seed. `code/ha_dynamic_dp.py` does this; it is the only computation in this
package that is measured in hours rather than seconds, and it is the reason the verdict was
constructed so as not to depend on it.

#### The shape, where the reduction could not see it

`code/ha_dp_policy_probe.py` retains the argmax that the sweep discards and reports the map along
merchant \(i\)'s own reputation with the rivals pinned at \(r_0=0.5\) — the same one-dimensional
slice §10.3 computes, but cut out of the full four-dimensional optimum instead of out of a model in
which rivals were never a state. Three instances were chosen to cover the three regimes §10.3 found,
before the probe was run: a collapse instance, a near-\(f^*\) instance, and a build-toward-\(f^*\)
instance. All are merchant 3 at \(\delta=0.95\); 29 of the 31 own-reputation indices are reachable.

| | seed 70000 | seed 70002 | seed 70005 |
|---|---|---|---|
| \(f^*_i\) | 8 | 8 | 6 |
| action at \(x_0\), full program | 8 | **20** | 5 |
| action at \(x_0\), §10.3 reduction | 7 | **20** | 5 |
| corner region, full program | own-\(r\) 0–13 | own-\(r\) 0–28 | none |
| corner region, §10.3 reduction | own-\(r\) 0–10 | own-\(r\) 0–29 | none |
| fitted slope, full program | \(-0.614\) | \(+0.000\) | \(+0.029\) |
| reachable states where the action differs from \(f^*_i\) | 51.7% | 100.0% | 69.8% |
| nearest deviating state to \(x_0\) (\(L_1\) on grid indices) | 2 | 0 | 0 |
| action there | 20 | 20 | 5 |
| relative gain at \(x_0\) | 0.178% | 13.468% | 0.033% |

The shape survives the lifting of the restriction, and in the collapse instance it is *larger*: seed
70000's corner region grows from eleven states to fourteen once rivals are visible. Nothing in any of
the three maps rises through \(f^*_i\) and stays above it. The one instance with a positive slope,
seed 70005, is the build-toward-\(f^*\) regime again — the map sits at 5 against \(f^*_i=6\) and
touches 6 at scattered indices, which is a merchant fabricating *less* than the profile prescribes.

Seed 70002 is worth stating on its own, because it is the instance behind the headline number. At
\(x_0\) — all four reputations at 0.5, the published initial condition of every run in this package —
the best response to the enumerated profile is \(f_i=20\), **total fabrication, immediately**. Not
after a reputation has been accumulated, and not after the market has drifted somewhere unusual: the
nearest deviating state is \(x_0\) itself, and the action differs from \(f^*_i\) at 100% of reachable
states. That deviation is worth 13.468% of the merchant's own discounted value, and it is the same
13.468% the \(\mathcal C_{2.5}\) certificate reports for this instance: \(0.150541662\) in value
units by both routes, agreeing to nine significant figures. At the instance that produces the
headline number, watching the rivals is worth **nothing** at \(x_0\) — the reduction is not merely a
valid lower bound there, it is the answer.

Two cautions. The probe's slope is fitted over the reachable indices only and §10.3's over all 31, so
the two slope columns above are close but not the same statistic; the corner regions, which are read
off the maps, are directly comparable. And a probe is three instances, chosen to span regimes rather
than sampled — it can confirm that the reduction was not hiding a different shape, and it cannot
establish a frequency. The frequencies in §10.3 are from all 240, and out of sample from all 120.

#### One flag in the solver's output that must not be read as agreement

`ha_dynamic_dp.py`'s per-merchant record carries a field named `build_then_exploit`, inside its
`gamma_80` block, and it reads **true almost everywhere**. It does *not* corroborate anything above,
and the collision of names is unfortunate enough to state plainly. The flag is
`action_at_x0_round_T > action_at_x0_round_1` over `actions_at_x0_by_round`: the action **at the
fixed state \(x_0\)** as the round index runs to the terminal date. It measures movement in *time*
under a known end, not movement across *reputations*, and only the second bears on R1.

What the flag is picking up is the \(\Gamma_{80}\) end-game §7 warned about. Seed 70000's merchant 3
has the path \(7\) for 66 rounds, then \(8\) for 4, then \(20\) for the last 10 — the entire "rise"
is the terminal unravelling, and for 66 of the 80 rounds the merchant fabricates *less* than
\(f^*_i=8\). Across the first
completed block of 20 instances the flag is true 20 times, every path ends at the corner, and the
window in which the action exceeds \(f^*_i\) has median length 4 rounds; 19 of the 20 never exceed
\(f^*_i\) at any point in the first 40 rounds. Reading that flag as build-then-exploit would
reintroduce, as a finite-horizon artefact of a horizon §2 already declared unidentified, exactly the
mechanism the state-indexed argmax says does not occur.

### 10.5 What it costs the marketplace

Everything above is about incentives. The question the platform actually asks is in GMV, and the
translation is not automatic: a profile can be badly non-equilibrium and cost almost nothing, or
barely non-equilibrium and cost a great deal. So the same 60 markets are priced four times, differing
**only** in which restriction is lifted, each divided by the same per-seed first best \(G^{FB}\):

| what is lifted | mean GMV | % of first best |
|---|---|---|
| nothing — the corpus's own plug-in \(\mathrm{GMV}(\bar r)\) at \(f^*\) \(\;[\mathcal C_0]\) | 0.6326 | **70.35%** |
| R3: \(\mathbb E[\mathrm{GMV}(r)]\) at \(f^*\) under the exact stationary law | 0.6326 | **70.35%** |
| R3 + R2: discounted average from \(r_0=0.5\) at \(f^*\) | 0.6270 | **69.74%** |
| R3 + R2 + R1 (constant class): at the \(\mathcal C_1\) **equilibrium** | 0.5656 | **62.91%** |

The first two rows are the same to four figures — R3 does not move GMV, as §9.3 said. The third row
costs 0.6 points, which is §9.4's start-up transient. **The fourth row is the finding.** It is not a
different way of scoring \(f^*\); it is the profile merchants actually settle on once they are scored
correctly, and it is a different profile on **41 of 60 seeds**. Best-response iteration reached a
**fixed point on 60 of 60** — no cycles, no iteration caps — so the object being priced is a genuine
equilibrium of the constant-strategy game, not an artefact of a stalled search.

Mean fabrication rises from \(0.380\) at \(f^*\) to \(0.487\) at that equilibrium. The mean GMV loss
is **9.56%** and the worst seed loses **32.5%**.

It is patience-dependent, and monotonically so:

| \(\delta\) | \(\mathcal C_1\) equilibrium \(\neq f^*\) | mean \(f\) | % of first best | worst seed |
|---|---|---|---|---|
| 0.90 | 55 / 60 | 0.568 | **57.88%** | 38.83% |
| 0.95 | 41 / 60 | 0.487 | **62.91%** | 45.74% |
| 0.99 | 20 / 60 | 0.395 | **69.18%** | 52.29% |

So the corpus's 70.35% is recovered only in the patient limit. At the headline \(\delta=0.95\) the
marketplace attains **62.9% of first best, 7.4 points below what is reported**, and at \(\delta=0.90\)
just 57.9%. The honest statement of §6's headline is therefore conditional: *the uniform second-best
policy reaches 70.4% of first best **if merchants play the enumerated profile**, and 62.9% if they
play the constant-strategy equilibrium of the game they are actually in.*

Two caveats, both against the finding's favour and both stated because they bound it rather than
soften it. First, \(\mathcal C_1\) is still the **constant** class: it is a rung on the ladder, not
\(\Gamma_\delta\), and §10.3 shows state-contingent strategies do strictly better still — so 62.9%
is not a floor, and the true dynamic figure is not bracketed by this table. Second, the ratios use
`ha_benchmarks.json`'s \(G^{FB}\) rather than recomputing it, which is why the check asserts that its
enumerated equilibrium agrees with the one recomputed here on **every one of the 60 seeds** before
quoting any ratio.

Four exactness guards run before any of this is believed: the reduced `Lite` market object
reproduces the solver's `Case` field by field to **exactly 0.0**; the GMV table reproduces
`ha_dynamic_dp.Case.GMV` contracted against the rivals' law to \(1.8\times10^{-15}\); platform GMV
assembled from merchant 0's and merchant 1's decompositions — which share no arithmetic — agrees to
\(1.5\times10^{-12}\); and the rivals' and the merchant's own laws have converged to
\(6.1\times10^{-14}\) and \(9.9\times10^{-16}\).

---

## 11. Verdict

### 11.1 The question, answered

The question this section exists to answer: *when the platform cannot see fabrication and has only
noisy complaints and a reputation signal, does merchants' fabrication settle into a stationary
hidden-action equilibrium after many rounds, and what fraction of first-best GMV does that equilibrium
reach?* Three answers, in the order they were established, none of which needed an LLM API.

**Yes, a resting point exists — within the constant-action class.** Best-response iteration over
constant actions scored by the exact discounted payoff from the experiment's own initial condition
reached a **fixed point on 60 of 60 seeds**, with no cycles and no iteration caps (§10.5). So the
premise of the stationary narrative is not empty: there is a well-defined profile that merchants
restricted to a fixed fabrication rate would settle on.

**No, it is not the profile the corpus reports.** That fixed point differs from \(f^*\) on **41 of 60
seeds**. Mean fabrication is higher — \(0.380\) at \(f^*\), \(0.487\) at the fixed point — because
\(f^*\) is scored by a plug-in stationary payoff and the fixed point by the payoff a merchant starting
at \(r_0=0.5\) actually collects.

**It reaches 62.91% of first best, not the 70.35% reported**, at the headline \(\delta=0.95\); 57.88%
at \(\delta=0.90\) and 69.18% at \(\delta=0.99\). The published figure is recovered only in the
patient limit.

And then the resting point itself does not survive the next rung. Letting merchant \(i\)'s action
depend on nothing more than its **own** reputation pays strictly on **223 of 240** merchant-instances
and on all 60 of 60 seeds (§10.3). Since every \(r_i\)-measurable strategy is measurable with respect
to the full state, \(f^*\) is **not** a subgame-perfect equilibrium of \(\Gamma_\delta\). So the
complete answer to the mandate's question is: *within the constant class a stationary profile does
form, it is not the one on record, and it books 7.4 points less GMV; in the actual dynamic game no
stationary equilibrium claim is licensed at all.*

The one place \(f^*\) does survive is instructive rather than reassuring. The 17 instances with
exactly zero own-state deviation gain are **exactly** the 17 where \(f^*_i\) already sits at the top
of the fabrication grid — the sets coincide, in sample and on 30 held-out seeds. \(f^*\) is
unimprovable only where it already prescribes total fabrication.

### 11.2 What may be called what

The mandate for this section forbids the label "dynamic equilibrium" for anything whose one-shot
deviation condition has not been verified on the full sufficient state space. Applying that rule:

| statement | licensed | evidence |
|---|---|---|
| \(f^*\) is the unique pure Nash equilibrium of \(\mathcal G(\theta,\kappa,\tau)\) | **yes** | Prop 1′, exhaustive over \(21^4\) profiles |
| \(f^*\) is a Nash equilibrium of the stationary game scored with \(\mathbb E[\cdot]\) | **no** — a \(0.22\%\)-equilibrium, failing on 12/60 seeds | §9.5 |
| \(f^*\) is a Nash equilibrium of the constant-strategy discounted game | **no** at \(\delta\le0.99\); a *patient* equilibrium — 24.6% of instances deviate at \(\delta=0.95\) | §10.2 |
| \(f^*\) is a stationary or dynamic equilibrium of \(\Gamma_\delta\) | **no** — refuted on 223 of 240 instances | §10.3 |
| the \(\mathcal C_1\) fixed point is a dynamic equilibrium | **no** — it is a fixed point of the *constant* class only | §10.5 |
| the marketplace attains 70.35% of first best | **only if merchants play \(f^*\)** | §6, §10.5 |
| the marketplace attains 62.91% of first best | as the \(\mathcal C_1\) figure; **not** as a dynamic figure, and not as a lower bound on one | §10.5 |
| \(\Gamma_{80}\) shows merchants would fabricate more late in a repeated market | **no** — an end-game artefact; first-half gains max at 0.0589% | §10.3 |
| the deviation R1 hides is build-then-exploit | **no** — 0 of 240 in sample and 0 of 120 out of sample; the shape is the mirror image, total fabrication at the *lowest* reputations, and it survives on the full \(31^4\) program | §10.3, §10.4 |

Nothing in this package establishes what \(\Gamma_\delta\)'s equilibrium *is*. It establishes what it
is not. That asymmetry is inherent: a positive equilibrium claim needs a fixed point of a
state-contingent best-response map over four merchants, which is a strictly larger computation than
anything run here, whereas a refutation needs one profitable deviation.

### 11.3 Proposition 1′, amended

§9.5 promised this amendment; §10 makes it larger than §9.5 alone would have.

> **Proposition 1′ (pure existence and uniqueness), amended.** *In every one of the 3,840 (seed,
> policy) pairs of the frozen design, the restricted stationary surrogate
> \(\mathcal G(\theta,\kappa,\tau)\) — constant actions, stationary evaluation, plug-in reputation —
> has exactly one pure-strategy Nash equilibrium \(f^*\). This is a statement about \(\mathcal G\)
> and about no other game. Scored with \(\mathbb E_\Pi[\pi_i(r)]\) rather than the plug-in, \(f^*\) is
> only a \(0.22\%\)-equilibrium and fails outright on 12 of 60 seeds at the headline policy. Scored
> with the exact discounted payoff from \(r_0=0.5\) over the same constant class, it is not a Nash
> equilibrium unless merchants are patient: 24.6% of merchant-instances prefer a different constant
> action at \(\delta=0.95\), and 20 of 240 prefer one at every \(\delta\) tested up to 0.9999. Over
> strategies contingent on the merchant's own reputation it is not an equilibrium on 223 of 240
> instances, the exceptions being exactly those at the fabrication corner. \(f^*\) is therefore
> reported as the unique equilibrium* of \(\mathcal G\) *and never as the equilibrium of the dynamic
> game the experiment runs.*

The uniqueness half is untouched and the enumeration behind it stands. What is withdrawn is the
implicit bridge — the step from "unique equilibrium of the object we solved" to "what merchants would
do" — which was never separately verified and is false.

### 11.4 What survives

Most of the document. Propositions 2–12 are statements about the deviation identity, the information
content of the signal, and the limits of the punishment instrument; none of them asserts that \(f^*\)
is played, and none is affected. §5's impossibility, §5's over-punishment reversal, Proposition 8's
zero Fisher information for refunds and Proposition 10's implementation floor are all properties of
the mechanism, not of the profile. §6's four benchmarks survive as benchmarks: they were always
defined as optima over enumerated profiles, and an optimum does not require anyone to play it.

What does not survive is any sentence that reads the corpus's equilibrium GMV as the GMV a merchant
population would produce. That is now a conditional statement and §10.5 gives the number it is
conditional on.

### 11.5 Two deductions from 70.35%, which are not one deduction

Two independent corrections to the headline have now been measured, and they must not be silently
combined:

| correction | size | source |
|---|---|---|
| the uniform policy is chosen in sample on the seeds it is then scored on | 70.5% → **68.4%** out of sample | §6 |
| merchants play the constant-class equilibrium rather than \(f^*\) | 70.35% → **62.91%** | §10.5 |

They are independent in origin and would compound, but their composition is **not** the product: the
policy \((\kappa,\tau)\) would be re-selected under the \(\mathcal C_1\) criterion, and there is no
reason the argmax over the 64-policy class is the same one. §6 already shows the top of that class is
a near-tie moving across three policies over 60 leave-one-out folds. A single combined number is
therefore not reported. The chain is reported, and a reader who needs one figure should be told which
question it answers.

### 11.6 What would overturn this

Stated so that a disagreement can be settled by computation rather than argument.

* **Patience.** Everything above is at \(\delta=0.95\). The refutation weakens monotonically with
  patience — 20 of 240 instances at \(\delta=0.999\) against 59 at \(0.95\), and the \(\mathcal C_1\)
  GMV rises to 69.18% at \(\delta=0.99\). A defensible argument that merchants in this market are
  much more patient than 0.95 would shrink the finding substantially, though not to zero: the 20
  instances that deviate at every \(\delta\) on the grid are not a patience artefact.
* **The tail assumption.** \(\mathcal C_1\), \(\mathcal C_2\) and \(\mathcal C_{2.5}\) sum \(T_0=250\)
  rounds explicitly and close with a resolvent, so they assume the rivals' law has settled by \(T_0\).
  That assumption carries \(\sim10^{-11}\) of the value at \(\delta=0.9\) and **97.5% at
  \(\delta=0.9999\)** (§10.1). The patient-limit rows are the ones to distrust; the headline row is
  not.
* **The wrong \(f^*\).** \(f^*\) is enumerated from \(b\) snapped onto `B_GRID` while the runner and
  the solver both use the exact draw (§10.2). Re-enumerating the frozen corpus on the exact \(b\)
  would change *which* markets fail. It would not change that some do: the exact-\(b\) process fails
  on more instances than the bucketed one, not fewer.
* **A tighter upper rung.** \(\mathcal C_{2.5}\) is a lower bound on \(\varepsilon_3\), so the true
  exploitability of \(\Gamma_\delta\) is at least what is reported and possibly much more. §10.4
  measures the gap directly where the full program has been run. Nothing there can lower the verdict;
  it can only raise it.
* **A different strategy class for the platform.** Everything is inside the 64-point
  \(\{\kappa\}\times\{\tau\}\) class (§8, item 1). A mechanism with transfers, bonding or menus is
  outside it and might restore an equilibrium at a much better GMV. That is a different paper, and
  this one should not be read as ruling it out.
* **The shape, if the reduction is hiding it — partly closed, and not by an argument.** The finding
  that no deviation is build-then-exploit is read off §2.4's reduction, where the strategy may
  condition on own reputation but not on rivals'. A build-then-exploit deviation that requires
  watching a rival would be invisible to it. This is the one conclusion in §10 that the reduction
  bounds in the *wrong* direction — for the size of \(\varepsilon\) the reduction is a lower bound
  and can only understate, but for the shape it is simply a different question. §10.4's probe of the
  full \(31^4\) argmax was run for exactly this reason and did not overturn it: on the three instances
  probed the collapse shape survives and in one case grows, and nothing rises through \(f^*_i\).
  Three instances are not 240, so the residual risk is real but now bounded in kind rather than
  unexamined: what remains possible is that build-then-exploit exists at \(\mathcal C_3\) on
  instances the probe did not visit. Retaining the full argmax over all 240 would settle it and costs
  155 MB per iteration, which is why it has not been done. If it ever is and the maps do cash in, the
  sentence to withdraw is the one about the mechanism, not the one about the equilibrium.

---

## 12. Traceability

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
| Ergodicity / mixing audit of the stationary solver (§9.1–9.2) | `results/solver/ha_mixing_audit.json` | `part_A_kernel_audit` |
| Jensen and transient costs, NE under expectations (§9.3–9.5) | same | `part_B_payoff_restrictions` |
| Benchmark sensitivity to the solver's initial guess (§9.2) | same | `part_C_benchmark_sensitivity` |
| Independent recomputation of all of §9 | `results/validation/recompute_dynamics_audit.json` | `all_pass` |
| R1 dropped: the full \(31^4\) dynamic program, both horizons (§10.4) | `results/solver/ha_dynamic_equilibrium.json` | `policies.P_SB_uniform.aggregate` |
| Per-block seed coverage of that artefact — **read before quoting any aggregate** | same | `policies.*.coverage_by_block` |
| Rungs \(\mathcal C_1\), \(\mathcal C_2\), \(\mathcal C_{2.5}\), and the sandwich (§10.2–10.3) | `results/validation/recompute_dynamic_dp.json` | `results[].population_certificate` |
| GMV consequence of dropping R1 (§10.5) | same | `results[].gmv_consequence` |
| \(b\)-discretisation: \(f^*\) solves a game the runner never plays (§8) | same | `results[].population_certificate.b_discretisation_sensitivity` |
| Same certificate on 30 held-out seeds never used in tuning | `results/validation/recompute_dynamic_dp_heldout.json` | `all_pass` |
| Shape of the reduced-state deviation, all 240 instances, with the margin that produced each argmax (§10.3) | same | `results[].cash_in_shape.shape` |
| Shape of the deviation on the full \(31^4\) program — argmax, cash-in slope, first deviating state | `results/solver/ha_dp_policy_probe.json` | `summary`, `per_seed[].merchants[]` |

Reproduce with:

```
python code/ha_benchmarks.py --seeds 70000-70059
python code/ha_benchmarks.py --seeds 70000-70059 --extended
python code/ha_theory_check.py
python code/ha_signal_analysis.py
python code/ha_dynamics_audit.py --part all --seeds 70000-70059
python offline_tests/recompute_dynamics_audit.py

# section 10 -- R1. The solver costs about 13 min per seed on one core, so it is
# sharded five seeds at a time and run STRICTLY SEQUENTIALLY: concurrent numpy
# jobs contend for BLAS and made each Bellman sweep 12x slower, which is how the
# first cost estimate for this run came out an order of magnitude wrong. The
# merge keeps a separate n_seeds per gamma_* block so no aggregate silently
# changes its denominator between blocks.
for lo in 70000 70005 70010 ... 70055; do
  python code/ha_dynamic_dp.py --seeds $lo-$((lo+4)) --policies P_SB_uniform \
         --deltas 0.95 --out results/solver/_dp_shard_main_$lo.json
done
python code/ha_dynamic_dp.py --seeds 70000-70004 --policies P_SB_uniform \
       --deltas 0.90,0.99 --no-finite --out results/solver/_dp_shard_delta.json
python code/ha_dp_merge.py results/solver/_dp_shard_*.json
python code/ha_dp_policy_probe.py --seeds 70000,70002,70005 --merchants 3 --delta 0.95

# the solver-free rungs, and the same certificates out of sample
python offline_tests/recompute_dynamic_dp.py --seeds 70000-70059
python offline_tests/recompute_dynamic_dp.py --only population_certificate cash_in_shape \
       --seeds 71000-71029 --out results/validation/recompute_dynamic_dp_heldout.json
```

No LLM API is involved in any of it.
