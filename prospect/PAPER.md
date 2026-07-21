# PROSPECT: Latent Mental Simulation for Language Agents — What Works, What Doesn't, and How We Know

**Predictive Rollouts in Self-supervised Embeddings for Counterfactual Thinking**

*Draft for NeurIPS/ICML submission. A measured study: one confirmed prediction, one
mechanistically-explained self-refutation, and an adequately-powered null.*

> **Epistemic status — read first.** This document separates two tiers, and never blurs them:
> * **[MEASURED]** — executed on real hardware and reported as data: the world-model pilot
>   (§10.1), the latent-vs-linguistic comparison (§10.2), the power analysis (§10.3), and the
>   powered re-run (§10.4). Numbers come from `training/` and named Kaggle kernels.
> * **[PROPOSED]** — designed and mathematically specified but not evaluated: the Latent
>   Trajectory Token LLM interface (§3.8) and latent counterfactuals (§3.9) are implemented and
>   CPU-smoke-tested but never measured against a real LM; DreamerV3/MuZero/GPT-4 are **NOT
>   RUN** (§10.2) and are never assigned invented numbers.
>
> Nothing in this paper is a fabricated result. Where we could not run a baseline, the cell
> reads "NOT RUN" with the reason, not a placeholder value.

---

## Abstract

Large language models reason by autoregressive token prediction. They can *describe* a
hypothetical outcome, but they possess no explicit, queryable model of the world that can be
rolled forward before a decision is committed. Chain-of-Thought and its descendants simulate
in *language*: every imagined step is paid for in tokens, is unconstrained by dynamics, and is
verified only by the same network that produced it.

We propose **PROSPECT**, an architecture that gives a language agent an explicit **mental
simulator** operating entirely in a self-supervised latent space. PROSPECT couples an
**action-conditioned JEPA** (AC-JEPA) world model — which predicts future *representations*
rather than reconstructing observations — to a planner that scores multi-step latent rollouts,
and exposes the resulting trajectories to an LLM as **latent trajectory tokens** (LTTs), so
the model can condition on simulated futures *without verbalising them*.

Three components are, to our knowledge, novel in combination. **(1) AC-JEPA**: a JEPA
predictor conditioned on actions, trained with a multi-step rollout objective plus explicit
anti-collapse regularisation, since a collapsed latent renders planning vacuous. **(2)
Uncertainty-gated adaptive horizon**: rollouts extend only while ensemble epistemic uncertainty
stays under budget, attacking the compounding-error failure that limits learned simulators;
Prop. 1 gives a drift bound motivating the gate. **(3) Latent counterfactuals**: a `do`-operator
in representation space, at cost independent of token budget.

**Our measured pilot refutes (2), and the refutation is our main empirical result.** On a
self-contained gridworld (3 seeds), AC-JEPA is validated — one-step error ~8% relative, drift
*linear* over 8 steps, and ablating VICReg collapses the encoder exactly as predicted (35×
variance gap, $\mathcal{L}_{\text{pred}} \to 0$). But ensemble disagreement is
**anti-correlated** with true rollout error ($r = -0.71$, replicated across three independent
runs): disagreement *decays* monotonically as error *grows*. The mechanism is structural — the
contractive residual parameterisation that buys linear (not exponential) drift also pulls every
ensemble member toward a shared attractor. **The design choice that makes latent rollouts
stable is the one that destroys ensemble-disagreement as an uncertainty signal.** This
contradicts standard practice in pessimistic model-based RL, where ensemble disagreement is the
default epistemic proxy.

We report this rather than the result we designed for. On the central hypothesis — latent
simulation beats reasoning-in-language for planning — we conducted a properly powered test.
After an *a-priori* power analysis (§10.3) fixed a benchmark with a well-defined 0.25 null and
80% power to detect a 0.15 effect, latent planning was **statistically indistinguishable from
Chain-of-Thought and from chance** (§10.4). This is a *valid null*, not an inconclusive one: the
effect, if any, is $<0.15$. The single replicated positive is a **7.7–31× latency advantage**
over CoT/ToT — a real property, but not a planning result, since no method plans above chance
here. We do **not** claim parity with DreamerV3 or MuZero (not run; a same-day reimplementation
would be a strawman). A methodological thread runs through the work: across two studies we three
times designed a mechanism before verifying the measurement could detect it, and once got it
right — we treat computing the null and required sample size *before* running as the transferable
lesson.

---

## 1. Introduction

Deliberation in current LLM agents is *linguistic*. Chain-of-Thought (Wei et al., 2022),
Tree-of-Thoughts (Yao et al., 2023), and ReAct (Yao et al., 2022) all externalise reasoning
into the token stream. This has three structural costs:

1. **Cost scales with imagination.** Each simulated step costs tokens; a branching search costs
   tokens × branches × depth. Latent rollouts cost a forward pass through a small MLP.
2. **No dynamics constraint.** A language "simulation" can emit any continuation the LM finds
   probable. Nothing enforces that state transitions are *possible*.
3. **Self-verification.** The critic is the generator. Errors that the model cannot detect, it
   cannot correct.

Model-based RL solved a version of this: PlaNet, Dreamer, and MuZero plan by rolling a learned
latent model. But those systems are (a) task-specific, (b) trained from reward, and (c)
disconnected from language. JEPA (LeCun, 2022; Assran et al., 2023) supplies the missing
representational idea — predict in *representation space*, discarding unpredictable detail —
but is defined without actions and without planning.

**The gap.** No existing system provides an *action-conditioned, self-supervised, latent world
model that a language agent can query before answering*. PROSPECT is a concrete proposal for
that object.

### 1.1 Contributions

1. **AC-JEPA** [PROPOSED, §3.2–3.4]. An action-conditioned JEPA with a multi-step rollout
   objective and VICReg-style anti-collapse. We argue (§3.2.1) that anti-collapse is not
   hygiene but a *precondition for planning*: under collapse, all trajectories score alike.
2. ~~**Uncertainty-gated adaptive horizon**~~ → **REFUTED BY §10.1; replaced by:** *the
   stability/uncertainty tension* [MEASURED, §10.1/C2.1]. We proposed gating rollout length on
   ensemble disagreement, motivated by Prop. 1. Measurement shows disagreement is
   **anti-correlated** with error ($r=-0.71$, ×3 runs), because contractive latent dynamics
   collapse the ensemble onto a shared attractor. Prop. 1's *bound* stands; our *proxy* for it
   was wrong, and wrong for a reason the architecture itself causes. We report this as a
   cautionary result for pessimistic model-based RL, which defaults to this proxy.
3. **Latent Trajectory Tokens** [PROPOSED, §3.8]. A projection of rollout summaries into the
   LLM's embedding space, letting the LM condition on simulation without generating it. This
   is the LLM↔world-model bridge that MuZero/Dreamer lack and CoT approximates in text.
4. **Latent counterfactuals** [PROPOSED, §3.9]. `do(a_k := a')` applied to a latent trajectory,
   giving counterfactual contrasts without natural-language generation.
5. **A powered evaluation and its methodology** [MEASURED, §10.1–10.4]. We validate AC-JEPA,
   refute the gate, and — after an *a-priori power analysis* (§10.3) that our pilot's design
   lacked — report an adequately-powered **null** on the central hypothesis: latent planning is
   indistinguishable from CoT and from chance at 7×7/3B. The contribution is not the null but the
   power analysis that makes it *valid* rather than *inconclusive* — a distinction we get wrong
   three times and right once across two studies.

---

## 2. Background

**JEPA.** Given context $x$ and target $y$, JEPA trains a predictor to match the *embedding*
of $y$ from the embedding of $x$: $\min \|P(E(x)) - \mathrm{sg}[\bar E(y)]\|^2$, where
$\mathrm{sg}$ is stop-gradient and $\bar E$ an EMA target encoder. Unlike generative models
(VAE, Dreamer's RSSM), no pixel is reconstructed, so capacity is not spent on unpredictable
detail. The price is **collapse**: a constant encoder achieves zero loss.

**Model-based RL.** PlaNet/Dreamer learn latent dynamics with reconstruction; MuZero learns
dynamics with *no* reconstruction but supervises only value/policy/reward — its latent is
whatever those three signals require. AC-JEPA sits between: self-supervised like JEPA,
reconstruction-free like MuZero, but trained to be *predictively sufficient* for dynamics.

**Notation.** Observation $o_t \in \mathcal{O}$, action $a_t \in \mathcal{A}$, latent
$s_t \in \mathbb{R}^d$, horizon $H$, discount $\gamma$.

---

## 3. Method

```
                    ┌──────────────────────── PROSPECT ────────────────────────┐
   o_t ──► Encoder E_θ ──► s_t ─┬──────────────────────────────────────────┐   │
                                │                                          │   │
                                ▼                                          │   │
                    ┌─ Transition ensemble {T_φk} ◄── a_t ─── Planner ─────┤   │
                    │           │                                ▲         │   │
                    │           ▼                                │         │   │
                    │   ŝ_{t+1} ... ŝ_{t+H}   (latent rollout)   │         │   │
                    │           │                                │         │   │
                    │           ├──► Uncertainty u(s,a) ──► gate ┤ (halt)  │   │
                    │           │                                │         │   │
                    │           └──► Value V_ψ / Reward R_ψ ─────┘         │   │
                    │                       │                              │   │
                    │                       ▼                              │   │
                    │            Trajectory score J(τ)  ──► argmax τ*      │   │
                    └───────────────────────┬──────────────────────────────┘   │
                                            ▼                                  │
                             Latent Trajectory Tokens  z(τ*) ∈ R^{m×d_llm}     │
                                            │                                  │
   prompt ────────────────────────────► LLM (frozen) ──► response              │
                                            ▲                                  │
                                  action proposals a^(1..N) ────────────────────┘
```

### 3.1 State representation
$$s_t = E_\theta(o_t), \qquad \bar s_t = \bar E_{\bar\theta}(o_t), \qquad
\bar\theta \leftarrow \tau\bar\theta + (1-\tau)\theta$$
$\bar E$ is the EMA target encoder ($\tau \approx 0.996$), never updated by gradient.

### 3.2 The AC-JEPA objective

**One-step prediction.**
$$\mathcal{L}_{\text{pred}} = \mathbb{E}\big[\,\|T_\phi(s_t, a_t) - \mathrm{sg}[\bar E(o_{t+1})]\|_2^2\,\big]$$

#### 3.2.1 Anti-collapse is a planning precondition
If $E_\theta \equiv c$, then $\mathcal{L}_{\text{pred}} = 0$ and every trajectory has identical
value — **planning becomes a no-op**. We therefore regularise with VICReg terms (Bardes et al.,
2022) over a batch $S \in \mathbb{R}^{B\times d}$:

$$\mathcal{L}_{\text{var}} = \frac1d\sum_{j=1}^{d}\max\!\big(0,\ \gamma_v - \sqrt{\mathrm{Var}(S_{:,j}) + \epsilon}\big)$$
$$\mathcal{L}_{\text{cov}} = \frac1d\sum_{i \neq j} \big[C(S)\big]_{ij}^2, \qquad
C(S) = \tfrac{1}{B-1}(S - \bar S)^\top (S - \bar S)$$

$\mathcal{L}_{\text{var}}$ forces per-dimension standard deviation above $\gamma_v$ (hinge);
$\mathcal{L}_{\text{cov}}$ decorrelates dimensions, preventing informational collapse into a
subspace. **This is a falsifiable design claim**: ablating VICReg should collapse latent
variance and destroy planning (§11, A1) — measured in the pilot (§10.1).

### 3.3 Action-conditioned transition ensemble
$$\hat s_{t+1}^{(k)} = T_{\phi_k}(s_t, a_t) = s_t + f_{\phi_k}\big([\,s_t \,\|\, \psi(a_t)\,]\big),
\qquad k=1..K$$
Two choices matter. **Residual form** ($s_t + f(\cdot)$) biases toward slow dynamics and
markedly improves multi-step stability. **Action embedding** $\psi$: a learned embedding for
discrete $\mathcal{A}$; an MLP for continuous. The ensemble ($K{=}5$, different init + data
order) supplies epistemic uncertainty (§3.5) at negligible cost, as each $f_{\phi_k}$ is a small
MLP.

### 3.4 Multi-step rollout objective (latent consistency)
Training on one-step loss alone yields models that diverge under their own predictions
(exposure bias). We roll the model on its *own* output and supervise every step:
$$\hat s_{t+h} = T_\phi(\hat s_{t+h-1}, a_{t+h-1}), \qquad \hat s_t = s_t$$
$$\mathcal{L}_{\text{roll}} = \sum_{h=1}^{H} \lambda^{h-1}\,\big\|\hat s_{t+h} - \mathrm{sg}[\bar E(o_{t+h})]\big\|_2^2$$
with $\lambda \in (0,1]$ down-weighting distant steps. We anneal from teacher forcing to full
rollout (scheduled sampling) over training.

**Total objective.**
$$\boxed{\ \mathcal{L} = \mathcal{L}_{\text{roll}} \;+\; \alpha_v \mathcal{L}_{\text{var}} \;+\; \alpha_c \mathcal{L}_{\text{cov}} \;+\; \alpha_r \mathcal{L}_{\text{rew}}\ }$$
where $\mathcal{L}_{\text{rew}} = \mathbb{E}\|R_\psi(\hat s_{t+h}) - r_{t+h}\|^2$ grounds the
latent in task-relevant structure (optional; §11 A2 ablates it).

### 3.5 Uncertainty and the adaptive horizon

**Epistemic uncertainty** as ensemble disagreement:
$$u(s,a) = \frac{1}{K}\sum_{k=1}^{K}\big\|T_{\phi_k}(s,a) - \bar T(s,a)\big\|_2^2,
\qquad \bar T(s,a) = \frac1K\sum_k T_{\phi_k}(s,a)$$

**Adaptive horizon.** Rather than a fixed $H$, we roll while cumulative uncertainty stays
within budget $\beta$:
$$H^*(s_0, \mathbf{a}) = \max\Big\{ h \;:\; \sum_{i=0}^{h-1} u(\hat s_i, a_i) \le \beta \Big\}$$

This is the mechanism that makes latent simulation *safe to trust*: the model reports where its
own imagination stops being reliable. Prop. 1 shows why a fixed horizon cannot do this.

### 3.6 Trajectory evaluation (pessimistic)
$$J(\tau) = \sum_{h=0}^{H^*-1} \gamma^h\Big[\underbrace{R_\psi(\hat s_h)}_{\text{predicted reward}} - \underbrace{\alpha\, u(\hat s_h, a_h)}_{\text{uncertainty penalty}}\Big] \;+\; \gamma^{H^*} V_\psi(\hat s_{H^*})$$

The $-\alpha u$ term implements *pessimism under model error* (cf. MOPO/MOReL): the planner is
discouraged from exploiting regions where the model is merely confident-looking. Terminal
$V_\psi$ absorbs the truncated tail — necessary precisely because $H^*$ is adaptive.

### 3.7 Planner
**Discrete** $\mathcal{A}$: MCTS with the LLM's action proposals as priors (§3.8), PUCT
selection, value backup from $J$. **Continuous**: CEM — sample $N$ sequences from
$\mathcal{N}(\mu,\sigma^2)$, evaluate $J$ in latent, refit $(\mu,\sigma)$ to the elite set,
iterate. The pilot (§10.1) uses CEM over discrete actions via one-hot relaxation for
simplicity; both are implemented.

### 3.8 LLM interface: Latent Trajectory Tokens [novel]
The bridge is bidirectional.

**Downward (LLM → planner).** The LLM proposes $N$ candidate action sequences
$\mathbf{a}^{(1..N)}$ from the prompt; these seed CEM's initial distribution / MCTS priors. The
LLM contributes *semantic plausibility*; the world model contributes *dynamical feasibility*.

**Upward (planner → LLM).** Given the selected $\tau^*$ (and counterfactual contrasts), we
summarise the rollout and project it into the LM's embedding space:
$$z(\tau) = W\,\big[\,\mathrm{pool}(\hat s_0,\dots,\hat s_{H^*})\ \|\ J(\tau)\ \|\ \bar u(\tau)\,\big] \in \mathbb{R}^{m \times d_{\text{llm}}}$$
with $W$ a learned projection (LLM frozen; only $W$ trained, cf. prefix-tuning). These $m$
tokens are prepended to the prompt. **The LLM conditions on the simulation without the
simulation ever becoming text** — this is the concrete sense in which PROSPECT reasons about
futures at cost independent of token budget.

### 3.9 Latent counterfactuals [novel]
Given a factual trajectory $\tau = (s_0,a_0,\dots)$, the counterfactual under
$do(a_k := a')$ is
$$\tau' = \big(s_0,a_0,\dots,s_k,\ a',\ \hat s'_{k+1},\dots\big), \qquad
\hat s'_{k+1} = T_\phi(s_k, a')$$
with effect $\Delta(k, a') = J(\tau') - J(\tau)$. Because $s_k$ is *reused* from the factual
rollout, the intervention is minimal in Pearl's sense: only the intervened variable and its
descendants change. Evaluating $|\mathcal{A}|$ counterfactuals at every step costs
$O(H|\mathcal{A}|)$ MLP passes — microseconds — versus $O(H|\mathcal{A}|)$ *LLM generations*
for a text-based equivalent.

---

## 4. Theory

**Proposition 1 (rollout drift bound).** Let $T_\phi$ be $L$-Lipschitz in $s$, and suppose the
one-step prediction error is bounded, $\|T_\phi(s_t,a_t) - \bar E(o_{t+1})\| \le \varepsilon$
for all reachable $(s_t,a_t)$. Then the $h$-step rollout error satisfies
$$\|\hat s_{t+h} - \bar E(o_{t+h})\| \;\le\; \varepsilon \sum_{i=0}^{h-1} L^{i} \;=\;
\begin{cases}\varepsilon\,\dfrac{L^{h}-1}{L-1}, & L \neq 1\\[4pt] \varepsilon h, & L = 1\end{cases}$$

*Proof sketch.* Induction on $h$. Let $e_h = \|\hat s_{t+h} - \bar E(o_{t+h})\|$. Then
$e_{h+1} \le \|T(\hat s_{t+h},a) - T(\bar E(o_{t+h}),a)\| + \|T(\bar E(o_{t+h}),a) - \bar E(o_{t+h+1})\|
\le L e_h + \varepsilon$, with $e_0 = 0$. Unrolling gives the geometric sum. ∎

**Consequences.** (i) For $L>1$ error grows *exponentially* — a fixed horizon is unjustifiable
without knowing $L$. (ii) The residual parameterisation (§3.3) targets $L \approx 1$, giving
*linear* growth. (iii) The bound is worst-case and unobservable at test time; ensemble
disagreement $u$ is a *measurable proxy* for it, which is precisely why the gate (§3.5)
conditions on $u$ rather than on $h$. **This proposition motivates the adaptive horizon; it
does not prove the gate optimal.**

---

## 5. Algorithms

**Algorithm 1 — Training AC-JEPA**
```
Require: dataset D of trajectories (o_t, a_t, r_t); horizon H; EMA rate τ; weights α_v, α_c, α_r
 1: init θ (encoder), φ_1..φ_K (transition ensemble), ψ (reward/value); θ̄ ← θ
 2: for each training step do
 3:    sample batch of subsequences (o_t..o_{t+H}, a_t..a_{t+H-1}, r) ~ D
 4:    s_t ← E_θ(o_t)                                       # online encoder
 5:    for h = 1..H:  s̄_{t+h} ← Ē_θ̄(o_{t+h})               # target embeddings (stop-grad)
 6:    ŝ ← s_t ;  L_roll ← 0
 7:    for h = 1..H do
 8:        k ~ Uniform{1..K}                                # per-step ensemble member
 9:        ŝ ← T_{φ_k}(ŝ, a_{t+h-1})                        # roll on own prediction
10:        if scheduled_sampling_teacher_force(h): ŝ ← s̄_{t+h}
11:        L_roll ← L_roll + λ^{h-1} · ‖ŝ − sg[s̄_{t+h}]‖²
12:    L_var, L_cov ← VICReg(batch of s_t)                  # anti-collapse (§3.2.1)
13:    L_rew ← Σ_h ‖R_ψ(ŝ_h) − r_{t+h}‖²
14:    L ← L_roll + α_v L_var + α_c L_cov + α_r L_rew
15:    θ, φ, ψ ← Adam step on ∇L
16:    θ̄ ← τ θ̄ + (1−τ) θ                                   # EMA target update
17: return θ, φ, ψ
```

**Algorithm 2 — Latent Rollout Simulation (uncertainty-gated)**
```
Require: s_0; action sequence a_{0..H-1}; budget β; ensemble {T_φk}
 1: ŝ ← s_0 ; U ← 0 ; τ ← [s_0] ; h ← 0
 2: while h < H do
 3:    μ ← (1/K) Σ_k T_{φ_k}(ŝ, a_h)                       # ensemble mean
 4:    u ← (1/K) Σ_k ‖T_{φ_k}(ŝ, a_h) − μ‖²                # epistemic disagreement
 5:    if U + u > β: break                                  # ADAPTIVE HALT (§3.5)
 6:    U ← U + u ; ŝ ← μ ; append (ŝ, u) to τ ; h ← h + 1
 7: return τ, H* = h, U
```

**Algorithm 3 — Mental Simulation Planning (CEM)**
```
Require: s_0; iterations I; population N; elites M; priors from LLM (optional)
 1: (μ, σ) ← init_from_priors(LLM proposals) or (0, 1)
 2: for i = 1..I do
 3:    A ← {a^(n) ~ π(μ, σ)}_{n=1..N}                       # sample action sequences
 4:    for n = 1..N: τ^(n), H*_n ← Rollout(s_0, a^(n), β)   # Algorithm 2 (batched)
 5:        J_n ← Σ_h γ^h [R_ψ(ŝ_h) − α·u_h] + γ^{H*} V_ψ(ŝ_{H*})
 6:    E ← top-M sequences by J
 7:    (μ, σ) ← fit(E)                                      # refit proposal distribution
 8: return a* = μ, τ* = Rollout(s_0, μ, β), J*
```

**Algorithm 4 — LLM + World Model Inference**
```
Require: prompt p; observation o_0; frozen LLM; projection W
 1: s_0 ← E_θ(o_0)
 2: a^(1..N) ← LLM_propose(p, o_0)                          # semantic priors (downward)
 3: a*, τ*, J* ← Plan(s_0, priors = a^(1..N))               # Algorithm 3
 4: C ← {Δ(k,a') = J(τ') − J(τ*) : counterfactuals}         # Algorithm 5 (optional)
 5: z ← W([pool(τ*) ‖ J* ‖ ū(τ*) ‖ C])                      # Latent Trajectory Tokens (upward)
 6: response ← LLM(concat(z, embed(p)))                     # condition WITHOUT verbalising
 7: return response, a*, confidence = g(ū(τ*))
```

**Algorithm 5 — Latent Counterfactual Reasoning**
```
Require: factual τ = (s_0,a_0,..,s_{H*}); intervention set 𝒜'; step k
 1: J_fact ← Score(τ)
 2: for a' ∈ 𝒜' do
 3:    ŝ ← T_φ(s_k, a')                                     # do(a_k := a'), s_k REUSED (minimal)
 4:    τ' ← [s_0..s_k, ŝ] ⊕ Rollout(ŝ, a_{k+1..}, β)        # descendants re-simulated only
 5:    Δ(k,a') ← Score(τ') − J_fact
 6: return sorted {(a', Δ)}                                 # ranked counterfactual effects
```

---

## 6. Implementation [IMPLEMENTED]

```
prospect/
  world_model/
    encoder.py        # E_θ + EMA target Ē
    predictor.py      # JEPA losses: L_pred, VICReg (L_var, L_cov)
    transition.py     # action-conditioned residual ensemble {T_φk}
    rollout.py        # Algorithm 2 (uncertainty-gated, batched)
    uncertainty.py    # ensemble disagreement u(s,a)
    value_network.py  # R_ψ, V_ψ
    planner.py        # Algorithm 3 (CEM) [+ MCTS]
    trainer.py        # Algorithm 1
  llm/
    connector.py      # Latent Trajectory Tokens (projection W)
    planner_interface.py  # Algorithm 4 glue
  training/
    dataset.py        # self-contained gridworld (no external deps)
    train.py, evaluate.py
  configs/, experiments/
```

Key snippets appear in §6.1–6.4 of the repository README; the modules are runnable and are what
produced §10.1. Design notes worth stating: the transition is **residual** (Prop. 1 (ii)); the
target encoder is EMA-only (no gradient); VICReg is computed on the *online* encoder's batch;
and rollouts are **batched across the CEM population** so planning is one tensor op, not a loop.

---

## 7. Datasets

| Environment | Why appropriate | Feasible here? |
|---|---|---|
| **Self-contained gridworld** | Deterministic, discrete, tiny; ground-truth optimal plan computable by BFS → *exact* planning-accuracy metric. No dependency risk. | **Yes — pilot (§10.1)** |
| **MiniGrid / BabyAI** | Discrete, partial observability, language-conditioned goals; standard for latent planning; cheap. | Yes (T4) |
| **Procgen** | Procedural levels → tests generalisation/OOD directly. | Marginal |
| **MuJoCo** | Continuous control; standard for Dreamer/PlaNet comparison. | Marginal |
| **Atari (100k)** | Canonical MuZero/Dreamer benchmark; sample-efficiency comparison. | **No** (compute) |
| **ALFWorld / WebArena** | Language agents with real dynamics — the target setting for LTTs. | **No** (compute) |
| **Habitat / MineDojo / Minecraft** | Embodied, long-horizon; Voyager comparison. | **No** (compute) |

We state plainly: rows marked **No** are *proposed* evaluations. Reporting them as results
without the compute to run them would be fabrication.

---

## 8. Experimental methodology [PROPOSED]

**Baselines.** (a) Autoregressive LLM (direct answer); (b) Chain-of-Thought; (c)
Tree-of-Thoughts; (d) ReAct; (e) Voyager (skill library); (f) DreamerV3 (generative latent);
(g) MuZero (value-supervised latent); (h) **JEPA-noaction** (JEPA without action conditioning
— isolates contribution 1); (i) **PROSPECT-fixed-H** (isolates contribution 2).

**Metrics.** Planning accuracy (vs BFS-optimal where computable); task success; $h$-step latent
prediction error $\|\hat s_{t+h} - \bar E(o_{t+h})\|$; simulation consistency (agreement
between predicted and realised rollouts); sample efficiency (success vs environment steps);
inference latency; peak memory; generalisation (held-out layouts); OOD robustness (perturbed
dynamics).

**Controls the CANTOR post-mortem demands** (see companion report): ≥5 seeds, paired per-seed
deltas, effect sizes reported against seed SD, and *no claim from a single seed*.

---

## 9. Ablations [PROPOSED]

| # | Ablation | Isolates | Predicted effect |
|---|---|---|---|
| A1 | − VICReg | anti-collapse | latent variance → 0; planning ≈ random (§3.2.1) |
| A2 | − reward grounding | task-relevance of latent | prediction fine, planning degrades |
| A3 | − uncertainty gate (fixed $H$) | contribution 2 | long-horizon planning degrades (Prop. 1) |
| A4 | − ensemble ($K{=}1$) | uncertainty source | gate unavailable; pessimism off |
| A5 | − multi-step loss (1-step only) | §3.4 | rollout diverges (exposure bias) |
| A6 | − residual transition | Prop. 1 (ii) | $L>1$; exponential drift |
| A7 | rollout length $H \in \{1,3,5,10,20\}$ | horizon | inverted-U (drift vs foresight) |
| A8 | latent dim $d \in \{16,32,64,128\}$ | capacity | small $d$ collapses; large $d$ overfits |
| A9 | planner ∈ {random, CEM, MCTS} | search | CEM/MCTS ≫ random iff model accurate |
| A10 | action encoding ∈ {one-hot, embed, FiLM} | §3.3 | minor |

---

## 10. Results

### 10.1 Pilot — MEASURED [IMPLEMENTED]

Self-contained gridworld, 3 seeds, Tesla T4. **Two of our three testable claims fail.**

| seed | latent_std | −VICReg std | plan:gate | plan:fixed | plan:random | plan:−VICReg |
|---|---|---|---|---|---|---|
| 0 | 1.114 | 0.0321 | 0.217 | 0.233 | 0.217 | 0.200 |
| 1 | 1.094 | 0.0285 | 0.183 | 0.217 | 0.267 | 0.333 |
| 2 | 1.105 | 0.0329 | 0.233 | 0.183 | 0.233 | 0.333 |
| **mean** | **1.104 ± 0.010** | **0.0312 ± 0.0024** | 0.211 ± 0.025 | 0.211 ± 0.025 | 0.239 ± 0.025 | 0.289 ± 0.077 |

#### C1 — anti-collapse: **CONFIRMED** (replicated ×3)
A 35× latent-variance gap. The ablation reproduced §3.2.1's predicted mechanism *exactly*:
```
−VICReg:  L=0.0003  roll=0.0003  latent_std=0.031   <- loss ≈ 0 BECAUSE the encoder collapsed
full:     L=1.2780  roll=0.0435  latent_std=1.057
```
This is the paper's one cleanly confirmed prediction.

#### The AC-JEPA world model itself: **works**
Typical inter-state distance is $\sigma\sqrt{d} = 1.10\sqrt{64} \approx 8.9$, so the one-step
error of 0.731 is **~8% relative**, growing only to 0.869 over eight steps — *linear*, as the
residual parameterisation ($L\approx1$, Prop. 1(ii)) predicts. Rollouts are accurate and stable.

#### C2 — uncertainty-gated horizon: **REFUTED** (replicated ×3)
| h | 1 | 2 | 4 | 6 | 8 |
|---|---|---|---|---|---|
| latent error | 0.731 | 0.727 | 0.761 | 0.813 | 0.869 ↑ |
| ensemble $u$ | **0.0531** | 0.0292 | 0.0163 | 0.0120 | **0.0102** ↓ |

$$\mathrm{corr}(\text{error},\,u) = -0.713 \quad(\text{independent runs: } -0.741,\ -0.698)$$

Disagreement decays monotonically while error grows monotonically. Ensemble disagreement is not
merely uninformative — it is **anti-informative**. §3.5's gate, the contribution Prop. 1 exists
to motivate, rests on a proxy that points the wrong way. **Contribution 2 does not survive.**

#### C2.1 — Why: the stability/uncertainty tension [the paper's real finding]
The residual form (§3.3) is chosen to make $L \approx 1$, converting *exponential* drift into
*linear* drift (Prop. 1). But a contractive map pulls **every ensemble member toward the same
attractor**: as $h$ grows, $T_{\phi_1..K}$ converge to one another while the true trajectory
departs. Hence $u \downarrow$ exactly as error $\uparrow$.

> **The mechanism that makes latent rollouts stable is the mechanism that destroys
> ensemble-disagreement as an uncertainty signal.** These are not independent design choices.

This contradicts standard practice: ensemble disagreement is the default epistemic proxy in
pessimistic model-based RL (MOPO, MOReL). Our result says it is unsafe *specifically* in the
contractive-latent regime that stable rollouts require. We consider this the most useful thing
the pilot produced, and it is the inverse of what we set out to show.

#### C3 — latent planning: **UNTESTABLE on this task** (not a refutation)
Every condition sits at chance (0.25): gate 0.211, fixed-H 0.211, random 0.239, **α=0 with
H=10: 0.267**, and — decisively — the **collapsed** model plans at **0.289**, *above* the healthy
model. A latent with no information cannot outplan one that has information unless **neither
enters the score**. We tested and rejected three explanations in sequence:

1. *Untrained $V_\psi$* — a real bug (Alg. 1 omitted a value loss; $V_\psi$ was random yet used
   as the terminal bootstrap). Fixed → planning stayed at chance.
2. *Pessimism swamping reward* ($\alpha=1$, $u \sim 0.05$ vs $r = -0.01$) — plausible;
   $\alpha=0$ changed nothing.
3. *Horizon too short* — $H=10$ changed nothing.

The remaining explanation is the **task**: reward is $-0.01$/step with $+1$ only on goal-contact
(~4% of transitions). $R_\psi$ converges to $\approx -0.01$ everywhere and $V_\psi$ to a
near-constant, so $J(\tau)$ is **flat** — trajectories tie and CEM argmaxes noise. The gridworld
cannot resolve whether latent planning works. This is the same failure our companion study hit:
*the measurement, not the mechanism, is the binding constraint.*

#### 10.1.1 What a reviewer should take from §10.1
* AC-JEPA + VICReg: **validated** — accurate, stable, non-collapsed.
* Uncertainty-gated horizon: **refuted, with a mechanism** (§C2.1).
* Latent planning: **untestable on this pilot** (flat $J$); tested under proper power in §10.4,
  where it is a valid null. LTTs and latent counterfactuals: **untested** (implemented,
  CPU-smoke-tested, never evaluated against a real LM).
* Required next: a **dense-reward** environment (e.g. distance-shaped, or MiniGrid with shaped
  returns) where $J$ has gradient. Task admissibility must be established *before* the planning
  claim can be evaluated at all.

### 10.2 Latent vs linguistic simulation — MEASURED PILOT [IMPLEMENTED]

The paper's central hypothesis — latent simulation beats reasoning-in-language for planning —
is exactly PROSPECT vs Chain-of-Thought / Tree-of-Thoughts / direct LLM. We ran that
comparison on Qwen2.5-3B (T4), identical grids, identical metric (chosen first move on *a*
shortest path; ties counted), reward densified with potential-based shaping (Ng et al., 1999;
preserves the optimal policy) after §10.1 showed the sparse reward flattens $J(\tau)$.

**We present this as a PILOT / sanity check, not as a test of the hypothesis, because it is
statistically underpowered — see §10.2.1.**

| method | plan acc | ±sd | latency/decision | rel. latency |
|---|---|---|---|---|
| random | **0.444** | 0.064 | 0.000 s | — |
| LLM direct | 0.333 | 0.072 | 0.484 s | 1.0× |
| Chain-of-Thought | 0.375 | 0.150 | 0.498 s | 1.0× |
| Tree-of-Thoughts (k=3) | 0.389 | 0.173 | 1.985 s | 4.1× |
| **PROSPECT (latent CEM)** | 0.389 | 0.158 | **0.065 s** | **0.13×** |
| DreamerV3 | NOT RUN | — | — | faithful reimpl. = weeks; same-day = strawman |
| MuZero | NOT RUN | — | — | same |
| GPT-4 | NOT RUN | — | — | no API access |

**Two readings, one valid.**
* *Accuracy* (invalid — see §10.2.1): every method, PROSPECT and all three LLM baselines,
  is statistically indistinguishable from random (0.444). **No method demonstrates planning
  ability on this benchmark.** When a coin flip ties every method, the benchmark — not the
  methods — is what failed.
* *Latency* (valid): PROSPECT is **7.6× cheaper than CoT and 30× cheaper than ToT** per
  decision, and this gap is large relative to its variance. The architecture's cost argument
  holds; but cost is only interesting conditional on accuracy, which no method here establishes.

### 10.2.1 Why §10.2 is underpowered (and what a defensible version requires)

The benchmark cannot detect the effect it was built to measure. Three compounding causes:

1. **The null is high, not 0.25.** Counting ties (a fairness fix) raised chance from 0.25 to
   the empirical **0.444**: on 7×7 grids with 15% walls, a random state has on average ~1.8 of
   4 moves on some shortest path. The "improvement room" above chance is only ~0.56, and much
   of it is unreachable.
2. **Per-condition variance is large.** With $n=24$ grids × 3 seeds, a Bernoulli proportion at
   $p\approx0.4$ has SE $=\sqrt{p(1-p)/n}\approx0.10$ per seed. The observed sd (0.06–0.17)
   swamps every between-method gap (all within 0.11 of each other).
3. **No a-priori power analysis.** We sized the experiment by convenience (24×3), not by the
   sample needed to detect a plausible effect at the true null — the direct cause of (1)–(2).

**This is the third instance in our two studies of designing a mechanism before verifying the
measurement can see it** (cf. the companion CANTOR study, where the base model already solved
33–50% of items; and §10.1, where sparse reward flattened $J$). We treat the pattern itself as
a finding: *for planning/agentic evaluations, the null distribution and required sample size
must be computed before any method is run.* §10.3 does this; §10.4 re-runs on a benchmark
designed to the resulting spec.

### 10.3 Formal power analysis [MEASURED — `training/power.py`]

Run *before* any method is re-run. We (a) measured the empirical random-policy null on five
candidate benchmark designs, (b) selected one whose null is low and *well-defined*, and (c)
computed the required sample size via the two-proportion normal approximation, cross-checked
against a Monte-Carlo simulation (analytic $n$ reproduces 0.80 MC power — the calculation is
self-validating).

**(a) Null accuracy by design.** The pilot's flaw is visible in row 1: ties raise chance to
0.379. Filtering to states with **exactly one** optimal move fixes chance at 0.25 *by
construction*, verified empirically.

| design | random null | usable % | mean optimal moves |
|---|---|---|---|
| 7×7, w=0.15, ties kept (pilot) | 0.379 | 100% | 1.49 |
| 7×7, w=0.15, **unique-opt only** | **0.250** | 52% | 1.49 |
| **7×7, w=0.25, unique-opt only** | **0.250** | **65%** | 1.36 |
| 9×9, w=0.20, unique-opt only | 0.257 | 56% | 1.46 |
| 11×11, w=0.22, unique-opt only | 0.262 | 57% | 1.45 |

**(b) Chosen design:** 7×7, wall density 0.25, unique-optimal-move filter — null $p_0 = 0.250$,
65% grid yield.

**(c) Sample size** ($\alpha=0.05$ one-sided, power $=0.80$):

| target effect $\delta$ | $n$ (analytic) | MC power @ $n$ |
|---|---|---|
| +0.10 | 259 | 0.79 |
| **+0.15** | **120** | 0.80 |
| +0.20 | 70 | 0.79 |
| +0.25 | 46 | 0.82 |

**Pre-registered design:** to detect $\delta = +0.15$ (0.25 → 0.40), use **120 eval states =
40/seed × 3 seeds** at null 0.250. Contrast the pilot: 24/seed at null 0.444, per-seed
SE $\approx 0.10$ — undetectable by construction. A null result on the powered benchmark then
means *"no effect $\ge 0.15$,"* not *"the benchmark could not tell."*

### 10.4 Re-run on the powered benchmark [MEASURED]

Same five methods, §10.3 design (7×7, w=0.25, unique-optimal filter, 40 states × 3 seeds,
null $p_0=0.25$, 80% power to detect $\delta=0.15$). Random landed at 0.233 — confirming the
designed null.

| method | plan acc | 95% CI | latency/decision | rel. latency |
|---|---|---|---|---|
| random | 0.233 | ±0.114 | 0.000 s | — |
| LLM direct | 0.300 | ±0.075 | 0.483 s | 1.0× |
| Chain-of-Thought | 0.200 | ±0.049 | 0.495 s | 1.0× |
| Tree-of-Thoughts (k=3) | 0.200 | ±0.049 | 2.009 s | 4.2× |
| **PROSPECT (latent CEM)** | 0.250 | ±0.085 | **0.064 s** | **0.13×** |
| DreamerV3 / MuZero / GPT-4 | NOT RUN | — | — | (see §10.2) |

**Pre-registered verdict.** PROSPECT beats the null by $\Delta = +0.000$; the best LLM baseline
by $\Delta = +0.050$ — **neither reaches the $\delta=0.15$ detectable threshold.** PROSPECT and
CoT have overlapping CIs (indistinguishable). Every method sits within its CI of chance.

**This is a valid null result.** On a benchmark *designed a priori* to have 80% power to detect
a 0.15 effect at a well-defined 0.25 null, **no method — latent simulation or linguistic
reasoning — demonstrates a planning effect** on 7×7 gridworld at 3B scale. The conclusion is
"the effect, if any, is $<0.15$," not "the benchmark could not distinguish the methods" (the
pilot's fate, §10.2.1). We regard the *difference between these two statements* — and the power
analysis that separates them — as the section's methodological contribution.

**The one replicated positive signal is latency**, not accuracy: PROSPECT plans **7.7× faster
than CoT and 31× faster than ToT** (both the pilot and powered runs agree, tight variance),
consistent with §1's cost argument for dynamical over linguistic simulation. But latency is an
advantage only conditional on accuracy, which no method establishes here; a fast method
indistinguishable from chance is not a useful planner. We therefore make **no accuracy claim**
for PROSPECT and report the cost result as a property, not a win.

### 10.5 What the measured evaluation establishes (and does not)
* **Establishes:** AC-JEPA is an accurate, stable, non-collapsing world model (§10.1); the
  uncertainty gate is refuted with mechanism (§10.1/C2.1); latent rollouts are ~8–30× cheaper
  than linguistic reasoning (§10.4); and on a properly powered 7×7 benchmark, latent planning
  is statistically indistinguishable from CoT and from chance.
* **Does not establish:** anything about PROSPECT vs DreamerV3/MuZero (not run), about larger
  or richer environments, or about whether the null is *task-specific* (7×7 may be too simple
  to separate a learned latent planner from chance) or *architecture-specific*. §14 lists the
  environments (dense-reward continuous control; longer-horizon partial observability) where a
  powered test could still find an effect the gridworld cannot.

---

## 11. Related work

| Work | Latent model | Action-cond. | Reconstruction-free | Self-supervised | LLM-coupled | Planning |
|---|---|---|---|---|---|---|
| World Models (Ha & Schmidhuber, 2018) | ✓ | ✓ | ✗ (VAE) | ✓ | ✗ | ✓ |
| PlaNet / Dreamer / DreamerV3 | ✓ | ✓ | ✗ (recon) | partial | ✗ | ✓ |
| MuZero | ✓ | ✓ | ✓ | ✗ (value-sup.) | ✗ | ✓ (MCTS) |
| I-JEPA / V-JEPA | ✓ | **✗** | ✓ | ✓ | ✗ | **✗** |
| LeCun (2022) H-JEPA | proposed | proposed | ✓ | ✓ | ✗ | proposed |
| Decision Transformer | ✗ | ✓ | ✓ | ✓ | ~ | ✗ (no model) |
| CoT / ToT / GoT | ✗ (language) | ~ | — | ✗ | ✓ | ✓ (in text) |
| ReAct / Voyager / Generative Agents | ✗ | ✓ | — | ✗ | ✓ | ✓ (in text) |
| **PROSPECT** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ (latent) |

**Honest positioning.** LeCun's H-JEPA proposes an action-conditioned JEPA with planning; we do
not claim that idea. Our claim is narrower and concrete: a *specified, implemented*
instantiation with (i) an uncertainty-gated adaptive horizon grounded in Prop. 1, (ii) latent
trajectory tokens as an LLM interface, and (iii) latent counterfactuals. Against MuZero the
distinction is the self-supervised, reconstruction-free objective; against Dreamer, the absence
of reconstruction; against V-JEPA, actions and planning; against ToT/ReAct, simulation that is
dynamical rather than linguistic.

---

## 12. Neuroscience motivation

The architecture is a functional analogue, not a model, of a well-characterised system.

* **Hippocampal replay/preplay.** Rodent place cells replay past *and* pre-play unvisited
  future trajectories before choice points (Ólafsdóttir et al., 2015). This is trajectory
  rollout prior to action — Algorithm 2's biological referent.
* **Default Mode Network.** Prospection, autobiographical future-thinking, and simulation
  recruit a common DMN core (Buckner & Carroll, 2007) — the system's name derives from
  *prospection*.
* **Motor imagery.** Imagined movement activates motor cortex with preserved temporal
  structure, and *without execution* — simulation decoupled from action, i.e. our latent
  rollout without environment steps.
* **Predictive coding / active inference.** Cortex as hierarchical prediction with
  precision-weighted error (Rao & Ballard, 1999; Friston, 2010). Our uncertainty gate is a
  crude precision weighting: trust simulation only where predicted error is low.
* **Embodied cognition.** Simulation is grounded in sensorimotor representation, not symbols —
  the argument for latent rather than linguistic imagination.

**We resist over-claiming**: none of this is evidence *for* PROSPECT. It motivates the
factoring; only §10 can support it.

---

## 13. Limitations

1. **Compounding error** — bounded but not eliminated (Prop. 1). The gate *detects* drift; it
   does not *prevent* it. Long horizons remain hard.
2. **Latent drift / manifold departure.** Rollouts may leave the encoder's data manifold, where
   $R_\psi, V_\psi$ are undefined. Uncertainty is a proxy, not a guarantee.
3. **Collapse remains a live risk.** VICReg mitigates; it does not prove non-collapse.
4. **Distribution shift.** The model is accurate only where the data was.
5. **Planning cost.** CEM = $N \times I \times H$ transition passes. Cheap vs. an LLM, not free.
6. **Interpretability.** A latent trajectory is not human-readable; LTTs let the LLM *use*
   simulation but do not make it *inspectable* — arguably a safety regression vs CoT.
7. **Hallucination moves, not vanishes.** A confidently wrong transition is a hallucination in
   latent space, with no text for a human to catch it in.
8. **Untested at LLM scale.** The pilot is a gridworld. Whether LTTs help a real LM is open.

---

## 14. Future work
Hierarchical world models (temporal abstraction over options); multi-modal and video
prediction; robotics; memory-augmented planning (episodic retrieval seeding rollouts); tool use
as actions; self-improving agents — with the caveat that the companion CANTOR study shows
self-generated data cannot bootstrap below a data threshold; multi-agent simulation (modelling
other agents' latents); embodied deployment.

---

## 15. Conclusion

We proposed PROSPECT: an action-conditioned JEPA world model, an uncertainty-gated latent
rollout simulator, and an interface letting a frozen LLM condition on simulated futures without
verbalising them. The argument was that deliberation should be *dynamical* rather than
*linguistic* — a rollout constrained by learned dynamics is cheaper than a token stream and
constrained in a way a token stream is not.

The pilot supports part of that and dismantles part of it. **AC-JEPA works**: rollouts are
accurate (~8% one-step error) and drift *linearly*, and anti-collapse behaves exactly as §3.2.1
predicted — the ablation reproduced the theorised failure (loss $\to 0$ *because* the encoder
went constant) at a 35× variance gap. **The uncertainty gate does not work**, and its failure is
the paper's most useful contribution: ensemble disagreement is anti-correlated with rollout
error ($r=-0.71$, three independent runs) because the contractive dynamics that make rollouts
stable also collapse the ensemble onto a common attractor. *Stability and uncertainty-signal are
in tension by construction* — a caution for any pessimistic model-based method that gates on
disagreement.

**The planning claim was hard to test, and testing it honestly took three tries.** The pilot's
sparse reward flattened $J(\tau)$ so completely that a collapsed encoder planned as well as a
healthy one (§10.1); densifying the reward removed that confound but left a benchmark whose
tie-counted null was 0.44, larger than any between-method gap (§10.2); only an *a-priori power
analysis* (§10.3) produced a benchmark — unique-optimal-move states, null 0.25, 40 states × 3
seeds — with the power to detect a 0.15 effect. **Task admissibility and statistical power must
be established before a claim is testable; across two studies we designed the mechanism before
checking the measurement could see it three times, and computed the power first only once.**

We therefore make no claim of parity with DreamerV3 or MuZero (not run), and we now report the
central-hypothesis comparison as **measured**: on a benchmark designed *a priori* — via a formal
power analysis (§10.3) — to have 80% power to detect a 0.15 effect at a well-defined 0.25 null,
latent planning is **statistically indistinguishable from Chain-of-Thought and from chance**
(§10.4). This is a valid null, not an inconclusive one: the distinction rests entirely on the
power analysis, which is why we foreground it. The one replicated positive is a 7.7–31× latency
advantage — a real property, but not a planning result, since no method here plans above chance.

What we offer, then, is a specified and implemented architecture; one confirmed prediction
(anti-collapse); one mechanistically-explained self-refutation (the stability/uncertainty
tension); a measured, adequately-powered null on the central hypothesis at 7×7/3B scale; and a
running methodological thesis — repeated across two studies — that in agentic evaluation *the
null distribution and required sample size must be computed before the method is run*. We got
this wrong three times and right once; the paper is stronger for reporting both.

---

## References

Assran et al. (2023) *I-JEPA*. CVPR · Bardes et al. (2022) *VICReg*. ICLR · Buckner & Carroll
(2007) *Self-projection and the brain*. TiCS · Friston (2010) *The free-energy principle*. Nat
Rev Neurosci · Ha & Schmidhuber (2018) *World Models*. NeurIPS · Hafner et al. (2019) *PlaNet*.
ICML · Hafner et al. (2023) *DreamerV3* · Chen et al. (2021) *Decision Transformer*. NeurIPS ·
LeCun (2022) *A Path Towards Autonomous Machine Intelligence* · Ólafsdóttir et al. (2015)
*Hippocampal place cells construct reward-related sequences*. eLife · Rao & Ballard (1999)
*Predictive coding*. Nat Neurosci · Schrittwieser et al. (2020) *MuZero*. Nature · Wei et al.
(2022) *Chain-of-Thought*. NeurIPS · Yao et al. (2022) *ReAct*. ICLR · Yao et al. (2023) *Tree
of Thoughts*. NeurIPS · Yu et al. (2020) *MOPO*. NeurIPS · Wang et al. (2023) *Voyager* ·
Park et al. (2023) *Generative Agents*. UIST · Besta et al. (2024) *Graph of Thoughts*. AAAI
