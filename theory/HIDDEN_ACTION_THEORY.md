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
   **R1** (constant, state-independent actions) is the one that excludes build-then-exploit, and is
   tested in §10. Until §10, the gap between Proposition 1 and Propositions 2–12 is real.
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
| \(\mathcal C_3\) | closed-loop \(f_i^t(x)\), \(x\in R^m\) | exact | + state contingency = \(\Gamma_\delta\) |

\(\mathcal C_0\) and \(\mathcal C_1\) share a strategy set and differ only in the payoff functional;
\(\mathcal C_1\subset\mathcal C_2\subset\mathcal C_3\) are genuine enlargements. Reading the ladder
upward is what makes the result interpretable: if \(\varepsilon\) is already large at
\(\mathcal C_1\), the surrogate's failure has nothing to do with dynamic strategy at all and
everything to do with how it scores; if \(\varepsilon\) only appears at \(\mathcal C_3\), the failure
is exactly the build-then-exploit story R1 was written to flag.

The methods are deliberately unrelated. \(\mathcal C_1\) and \(\mathcal C_2\) are pure forward
simulations of a product law and are computed in `offline_tests/recompute_dynamic_dp.py`, which
imports no solver code; \(\mathcal C_3\) needs backward induction on all \(31^4=923{,}521\) states
(§2.4 explains why the 31-state reduction cannot be substituted) and is `code/ha_dynamic_dp.py`. So
the two cheap rungs are also independent lower bounds on the expensive one, and §10.4 checks that the
solver's \(\varepsilon\) is at least as large as the certificates that need no solver.

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

Reproduce with:

```
python code/ha_benchmarks.py --seeds 70000-70059
python code/ha_benchmarks.py --seeds 70000-70059 --extended
python code/ha_theory_check.py
python code/ha_signal_analysis.py
python code/ha_dynamics_audit.py --part all --seeds 70000-70059
python offline_tests/recompute_dynamics_audit.py
```

No LLM API is involved in any of it.
