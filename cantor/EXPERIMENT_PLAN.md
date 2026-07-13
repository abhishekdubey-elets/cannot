# Experimental Plan — "Remember or Relearn?"
### Locating Continual Improvement in LLM Agents Between Non-Parametric Memory and Parametric Weight Updates

*A concrete, ICML/CoLM-targeted experimental plan derived from Slice #2 of the CANTOR review
(replay-based consolidation, report §7.5, §10.7). This document specifies the claim, hypotheses,
datasets, baselines, metrics, ablations, expected result tables, protocol, and reviewer defenses.*

Companion to the reference implementation in [`cantor/`](cantor/) — the episodic/semantic/procedural
stores and the `sleep()` consolidation loop already exist; this plan specifies the science needed to
turn them into a paper.

---

## 1. One-line claim

> Continual improvement in an LLM agent decomposes cleanly into a **knowledge locus** (best served by
> non-parametric memory) and a **skill/policy locus** (which *requires* parametric weight updates), and a
> sleep-inspired **hybrid** that routes each type of experience to the right store dominates both pure
> strategies on end-of-stream accuracy **while retaining general capability** — which naive online
> weight updates destroy.

The paper's contribution is **not** "we built a memory system." It is a **controlled locus analysis**:
*which kinds of experience must change weights, which only need to be remembered, and what does the
Pareto frontier of (plasticity, stability, cost) look like when you factor them correctly?* No prior
work (Generative Agents, Voyager, MemGPT — memory only; ReST/STaR — batch self-training, no forgetting
analysis; classical CL — classification, not agentic LLMs) answers this cleanly.

---

## 2. Research questions and falsifiable hypotheses

| ID | Hypothesis | Falsified if... |
|---|---|---|
| **H1** (plasticity) | Consolidation improves task-family accuracy over a stream vs. a frozen agent. | Hybrid ≈ frozen baseline at end of stream. |
| **H2** (locus) | Non-parametric memory suffices for *knowledge reuse* (facts, exact-instance solutions) but **fails to generalize skills** to novel instances requiring a changed policy; parametric updates are required there. | Memory-only matches hybrid on the *novel-instance skill-transfer* split. |
| **H3** (stability) | Naive online LoRA on successes causes catastrophic forgetting of general capability; **prioritized + generative replay** prevents it at minimal plasticity cost. | Naive-LoRA shows no general-capability drop, OR replay fails to recover it. |
| **H4** (efficiency) | The hybrid reaches target accuracy at **lower cumulative inference cost** than always-deliberate, because consolidated skills route hard instances to System 1. | Hybrid's accuracy-per-dollar ≤ always-System-2. |

These are the four tables in §7. Each hypothesis has a designed experiment that can refute it.

---

## 3. Problem formalization

**Setting: domain-incremental continual learning for an agent.** A stream of tasks
`T = (t_1, t_2, ..., t_N)` drawn from a family distribution. The agent encounters each task online,
must produce an answer/trajectory immediately (no peeking ahead), and every `K` tasks (a "night")
runs an offline **consolidation** pass over its own accumulated episodic buffer.

**Three evaluation surfaces, measured after every night:**
1. **Online / running accuracy** — performance on the stream as it arrives (plasticity).
2. **Retention set** `R_i` — held-out instances from each *already-seen* task family (do we keep what we learned?).
3. **General-capability probe** `G` — a fixed benchmark *unrelated* to the stream (e.g., MMLU),
   held constant, to detect erosion of the base model (stability of general skill).

**What consolidation may update (the independent variable):**
- **Non-parametric:** semantic facts (exact-instance answers, provenance-tagged) + procedural skills
  (cached plans keyed by normalized task signature). No gradient.
- **Parametric:** eval-gated **LoRA** adapter updates trained on prioritized replay of successful
  self-generated trajectories, interleaved with **generative replay** of base-distribution samples
  to resist forgetting.
- **Hybrid (ours):** a router sends *knowledge-type* experience to non-parametric stores and
  *skill-type* experience (repeated task signatures with policy structure) to the LoRA update.

**Critical integrity constraint:** the agent's *internal* eval set used for eval-gating LoRA promotion
is drawn **only** from self-generated / episodic data and is **disjoint** from `R_i` and `G`. No test
leakage into the gating decision. (Reviewers will check this first — see §9.)

---

## 4. Datasets / task streams

Three streams, chosen so that (a) improvement-over-time is real and measurable, and (b) the
knowledge-vs-skill distinction is controllable.

| Stream | Purpose | Why it isolates the mechanism |
|---|---|---|
| **S-CTRL** — synthetic templated task family (e.g., a generator of multi-step tool/math problems where new *problem schemas* are introduced sequentially, with held-out *novel instances* of each schema) | Clean control; the H2/H3 proof | We control exactly what is "knowledge" (repeat instances) vs. "skill" (new instances of a learned schema). Lets us construct the decisive novel-instance-transfer split. |
| **AppWorld** or **τ-bench** (realistic tool-use agent tasks with repeated substructure) | External validity on real agentic tasks | Repeated API/tool patterns are consolidatable skills; realistic, adopted benchmarks reviewers respect. |
| **WebArena** (or ALFWorld/ScienceWorld for sequential-decision variety) | Second realistic domain, guards against single-benchmark overfit | Long-horizon, different action space; tests whether the locus finding generalizes across domains. |
| **MMLU** (fixed, never trained on) | The general-capability probe `G` | Standard, broad; a drop here is the catastrophic-forgetting signal for H3. |

**Primary result** on S-CTRL (mechanism) + AppWorld/τ-bench (realism). WebArena as the
generalization check. Report all; lead with the controlled + one realistic.

**Stream construction details to specify in the paper:** N (≈ 200–1000 tasks), schema-introduction
schedule, retention-set size per schema (≥ 50), night frequency K (ablated, §6), and 3+ stream
orderings per seed to control for curriculum-order effects.

---

## 5. Baselines

Strong, fair, and spanning the design space. Weak baselines are the fastest path to reject.

| # | Baseline | What it isolates / why reviewers demand it |
|---|---|---|
| B0 | **Frozen base agent** (no memory, no updates) | Floor. |
| B1 | **Raw-transcript RAG** (Generative-Agents-style recency+relevance+importance retrieval over past trajectories) | The "just remember text" baseline — the honest strong memory baseline. |
| B2 | **Structured non-parametric memory only** (CANTOR semantic + procedural, no LoRA) | The pure-memory arm of the locus question (tests H2). |
| B3 | **Naive online LoRA** (SFT on successes, no replay) | The catastrophic-forgetting arm (tests H3). |
| B4 | **Rehearsal LoRA** (experience replay: interleave stored past data) | Standard rehearsal-based CL baseline. |
| B5 | **EWC / L2-SP** (regularization-based CL) | Classic CL baseline reviewers expect to see. |
| B6 | **ReST / STaR-style self-training** (batch, offline, no episodic structure, no forgetting control) | The self-improvement baseline; tests whether episodic structure + eval-gating matters vs. plain self-distillation. |
| **B7** | **CANTOR-Hybrid (ours)** — prioritized multi-store replay + eval-gated LoRA + generative replay | The proposed method. |
| SKY | **i.i.d. oracle** (train on the *entire* stream offline, non-continually) | Upper-bound skyline; quantifies the continual-learning gap. |

**Models (open-weights, so LoRA is possible):** Llama-3.1-8B-Instruct as primary; Qwen-2.5-32B for a
scale check. Memory-only conditions (B1, B2) additionally reported on a frontier API model to show the
non-parametric findings transfer to models you *cannot* fine-tune (a nice practical angle: "if you
can't touch the weights, here's exactly what memory can and cannot buy you").

---

## 6. Metrics

Standard continual-learning metrics (Lopez-Paz & Ranzato, 2017; Chaudhry et al., 2018), plus cost.
Let `a_{i,j}` = accuracy on task-family `j`'s retention set after finishing stream position `i`.

- **Average Accuracy (ACC)** = mean over `j` of `a_{N,j}` — end-of-stream competence.
- **Backward Transfer (BWT)** = `(1/(N-1)) Σ_{j<N} (a_{N,j} − a_{j,j})`. **Negative = forgetting.** The headline stability number.
- **Forward Transfer (FWT)** = `(1/(N-1)) Σ_{j>1} (a_{j-1,j} − b_j)`, where `b_j` is base-model zero-shot on `j` — does prior consolidation help new tasks?
- **Novel-Instance Skill Transfer (NIST)** *(the decisive H2 metric)* = accuracy on *held-out novel instances* of learned schemas. Memory-only is predicted to lag here even when its exact-instance recall is perfect.
- **General-Capability Retention (GCR)** = `MMLU_after − MMLU_base` (should be ≈ 0 for a good method; strongly negative for B3).
- **Online cumulative accuracy / learning-curve area** — plasticity over the whole run.
- **Sample efficiency** — tasks-to-threshold (e.g., tasks to reach 80% schema accuracy).
- **Cost:** cumulative inference tokens + consolidation FLOPs (LoRA training + replay), and
  **accuracy-per-dollar**; plus **route mix** (fraction hitting System 1 vs. System 2 over time — H4's mechanism).

Report mean ± std over **≥ 3 seeds × 3 stream orderings**; significance via paired bootstrap;
Pareto plots for (ACC vs. GCR) and (ACC vs. cost).

---

## 7. Expected result tables (illustrative shapes — numbers hypothesized, not measured)

> **These numbers are placeholders indicating the *predicted pattern* that would support/refute each
> hypothesis.** They are not results. The paper reports measured values; here they show what "success"
> looks like and which cell decides each hypothesis.

### Table 1 — Main continual-learning result on S-CTRL (H1, H3)

| Method | ACC ↑ | BWT ↑ (0 = no forgetting) | FWT ↑ | GCR ↑ (0 = no erosion) | Acc/\$ ↑ |
|---|---|---|---|---|---|
| B0 Frozen | 0.41 | 0.00 | 0.00 | 0.00 | 1.0× |
| B1 Raw-RAG | 0.55 | −0.01 | +0.04 | 0.00 | 0.9× |
| B2 Memory-only | 0.63 | 0.00 | +0.06 | 0.00 | 1.1× |
| B3 Naive-LoRA | 0.71 | **−0.22** | +0.09 | **−0.14** | 0.7× |
| B4 Rehearsal-LoRA | 0.74 | −0.05 | +0.11 | −0.03 | 0.6× |
| B5 EWC | 0.68 | −0.04 | +0.07 | −0.02 | 0.6× |
| B6 ReST | 0.70 | −0.08 | +0.08 | −0.05 | 0.7× |
| **B7 Hybrid (ours)** | **0.79** | **−0.02** | **+0.14** | **−0.01** | **1.4×** |
| SKY i.i.d. oracle | 0.83 | — | — | −0.01 | — |

*Decisive cells:* B3's large negative BWT **and** GCR = the catastrophic-forgetting story (H3).
Hybrid recovering both while beating memory-only on ACC = the headline.

### Table 2 — The locus result (H2): knowledge vs. skill split on S-CTRL

| Method | Exact-instance recall ↑ | **NIST** (novel-instance skill transfer) ↑ | Gap |
|---|---|---|---|
| B2 Memory-only | **0.98** | 0.44 | **−0.54** |
| B3 Naive-LoRA | 0.66 | 0.69 | +0.03 |
| **B7 Hybrid (ours)** | **0.97** | **0.73** | −0.24 |

*This is the paper's intellectual core.* Memory-only nails exact recall but **cannot generalize the
skill** to unseen instances; weight updates are required for NIST. The hybrid gets both. If this table
does not show the memory-only NIST gap, **H2 is refuted and the paper's thesis collapses** — so run
this experiment *first* (§8, Phase 1) as a go/no-go.

### Table 3 — Realistic-domain generalization (external validity)

| Method | AppWorld/τ-bench ACC ↑ | BWT ↑ | GCR ↑ | Acc/\$ ↑ | WebArena ACC ↑ (transfer check) |
|---|---|---|---|---|---|
| B2 Memory-only | 0.52 | 0.00 | 0.00 | 1.1× | 0.31 |
| B4 Rehearsal-LoRA | 0.58 | −0.04 | −0.03 | 0.6× | 0.34 |
| **B7 Hybrid (ours)** | **0.62** | **−0.02** | **−0.01** | **1.3×** | **0.37** |

*Purpose:* show the S-CTRL locus finding survives on real agentic benchmarks and a second domain.

### Table 4 — Ablations (which mechanism carries the result)

| Variant | ACC ↑ | BWT ↑ | NIST ↑ | GCR ↑ | Note |
|---|---|---|---|---|---|
| **Full Hybrid** | **0.79** | −0.02 | **0.73** | −0.01 | — |
| − parametric (memory only) [= B2] | 0.63 | 0.00 | 0.44 | 0.00 | kills NIST → confirms H2 |
| − non-parametric (LoRA only) | 0.72 | −0.06 | 0.70 | −0.05 | loses exact recall + some stability |
| − generative replay (stored replay only) | 0.76 | −0.07 | 0.71 | −0.06 | worse GCR → generative replay matters |
| − prioritization (uniform replay) | 0.75 | −0.03 | 0.68 | −0.02 | salience gating gives a modest lift |
| − eval-gating (promote all updates) | 0.70 | −0.09 | 0.64 | −0.07 | bad updates leak in → quality control matters |
| − episodic salience gate (store everything) | 0.77 | −0.03 | 0.72 | −0.02 | memory-hygiene effect, smaller |

Plus two **curves**, not tables:
- **A8 — consolidation frequency K** (every 5 … 100 tasks): the "sleep schedule" — expect a U-shaped
  cost/quality tradeoff (too-frequent = expensive & unstable; too-rare = slow plasticity).
- **A9 — LoRA rank / update budget:** plasticity vs. forgetting vs. cost frontier.

---

## 8. Protocol and phasing (go/no-go gated)

| Phase | Work | Gate |
|---|---|---|
| **0. Harness** (~2 wk) | Wire a real backend into [`cantor/llm.py`](cantor/cantor/llm.py); add LoRA train/serve; build the CL evaluation loop (streams, retention sets, `G` probe, per-night eval); implement generative replay + eval-gating. | Harness reproduces B0 and B2 numbers stably across seeds. |
| **1. Mechanism (S-CTRL)** (~3 wk) | Table 2 first (the H2 go/no-go), then Table 1. | **If Table 2 shows no memory-only NIST gap → stop / re-scope.** This is the cheap experiment that de-risks the whole paper. |
| **2. Realism** (~4 wk) | AppWorld/τ-bench + WebArena (Table 3). | Locus finding replicates on ≥ 1 realistic domain. |
| **3. Ablations + scale** (~3 wk) | Table 4, curves A8/A9, Qwen-32B scale check, frontier-API memory-only condition. | Each claimed mechanism has an ablation that moves the decisive metric. |
| **4. Writing** (~2 wk) | Paper, release harness + streams as artifact (NeurIPS D&B-style reproducibility). | — |

**Compute estimate (order-of-magnitude):** 8B LoRA nights are cheap (minutes each on 1×A100);
N≈500 tasks × 3 orderings × 3 seeds × ~8 methods dominated by *inference* (agent rollouts), not
training. Budget ≈ a few thousand A100-hours total — feasible for an academic lab. Frontier-API
conditions add inference \$ but only for B1/B2.

---

## 9. Threats to validity and pre-emptive reviewer defenses

| Likely reviewer attack | Defense built into the design |
|---|---|
| "This is just RAG / prompt engineering." | Table 2: memory-only (incl. strong B1 RAG) **provably fails NIST**; the win requires weight updates. The controlled split makes this airtight. |
| "Catastrophic forgetting and replay are known." | Correct — the contribution is the **locus analysis** (which experience needs weights) + **agent's own trajectories as the prioritized replay source** + a unified plasticity/stability/cost evaluation for *agentic LLMs*, not inventing replay. Framed as such in the intro. |
| "Eval-gating leaks test labels." | Internal gating set is disjoint from `R_i` and `G` by construction (§3); we state and verify this, and report GCR on a benchmark never touched by any loop. |
| "Cherry-picked stream / curriculum order." | 3 stream orderings × 3 seeds, variance reported, order-effect ablation. |
| "Consolidation cost is hidden." | Cost metric includes LoRA + replay FLOPs; accuracy-per-dollar is a headline column. |
| "How far from just training i.i.d.?" | SKY oracle row quantifies the CL gap explicitly. |
| "Only works on a toy model." | Qwen-32B scale check + two realistic benchmarks. |
| "Self-training collapse / reward hacking over many nights." | Eval-gating + generative replay are exactly the guards; A6 (remove eval-gating) demonstrates the failure mode we prevent. |

---

## 10. Why this lands where CANTOR-as-a-whole did not

- **A single, controlled, falsifiable question** (remember vs. relearn) instead of a grand architecture.
- **Real models, real benchmarks, strong baselines, standard CL metrics** — the three things the ICML
  review flagged as missing.
- **A decisive early experiment** (Table 2) that de-risks the paper in ~3 weeks.
- **An honest scope:** it validates *one* CANTOR loop (§7.5 consolidation) rigorously, and cites the
  full report as the vision context — the report becomes related-work/framing, the measured locus
  result becomes the paper.

**Venue fit:** strongest at **CoLM** (LM-native, receptive to this exact systems-meets-learning
question) and competitive at **ICML/NeurIPS** if the locus result is clean; the harness + streams also
make a natural **NeurIPS Datasets & Benchmarks** companion release.

---

# 11. Real-model results (Kaggle T4, Qwen2.5-3B-Instruct)

First execution of the `--real` path on actual hardware. **These supersede the offline
simulation's magnitudes.** Recorded here because they change the plan.

## 11.1 Infrastructure (solved)

| issue | resolution |
|---|---|
| GPU/internet silently disabled | Kaggle gates both behind phone verification; used a verified account |
| `no kernel image is available` | Kaggle allotted a **P100 (sm_60)**; preinstalled **torch 2.10 supports sm_70+ only**. Fixed by pinning `--accelerator NvidiaTeslaT4` (sm_75). No code change. |
| no internet in kernel | attach Qwen as a **Kaggle model source** (mounted files) + use fp16 (drops the `bitsandbytes` dep) |
| `prepare_model_for_kbit_training` on fp16 | guarded behind `load_in_4bit` |

## 11.2 Backend bring-up — PASSED

`HFTrainableLM` validated on GPU: novel-instance accuracy **0.00 → 1.00** after 25 LoRA
steps (17.5 s) on ground-truth pairs, then **→ 0.00** after `restore()`. Generate, train,
snapshot, and rollback all work on real weights.

## 11.3 The locus experiment did NOT reproduce

| method | exact_recall | NIST (novel) | GCR |
|---|---|---|---|
| B2-memory | 0.556 | 0.167 | 0.000 |
| B7-hybrid | 0.667 | **0.167** | 0.000 |

B7's parametric arm contributed **nothing**. Three compounding causes, each measured:

1. **The eval-gate rejects the update that matters.** 25 LoRA steps on ~6 narrow examples
   genuinely degrade general capability (drops of 0.033–0.083 vs `eps=0.03`), so the gate
   correctly rolls them back. 6 generative-replay anchors against 6 task examples is too
   weak an anchor ratio.
2. **Surviving updates teach nothing.** Accepted nights still yield `schema probe acc = 0.00`.
3. **Replay starvation.** `counts={'S0': 6, 'S1': 1}` → only `S0` ever cleared the
   ≥2-self-success eligibility bar. Most schemas never train at all.

## 11.4 A real bug found: replay-target format mismatch

The replay buffer stored the model's **verbose deliberate** generation, while `train_lora`
renders the **terse** prompt and the fast path is evaluated terse — i.e. LoRA was taught
*"terse prompt → rambling answer."* Fixed: store the **parsed answer**
(`RolloutSolver(replay_raw_completion=False)`), self-distilling what deliberation concluded.

A/B under identical weights/stream/seed: NIST `0.000 → 0.083`, GCR `-0.188 → -0.125`.
**Directionally right but not a real effect** — 24 novel items give ~0.042 resolution, so
+0.083 is two items at n=1 seed. The fix is correct in principle; the bottleneck is upstream.

## 11.5 What was validated: the eval-gate works on real weights

With a properly-sized probe set (60 probes, noise floor 0.017 < `eps` 0.03), the gate
detected true regressions (0.033, 0.083) and rolled them back while accepting benign drift
(0.017). This is live H3 evidence, and it vindicates the probe-sizing rule from §Phase-3
bring-up: **`noise floor < eps < per-update drift`**.

## 11.6 Sim-to-real gap (plan-changing)

The simulation assumed `master_at=2` exposures and `drift=0.06` vs `eps=0.03`. Reality:
mastery needs **far more data** than assumed, and the drift-to-eps ratio is **worse**. Every
magnitude in Tables 1–4 is therefore untrustworthy as a prediction of real-model behaviour;
the offline harness validates *pipeline and control flow*, not effect sizes.

## 11.7 Methodological caution recorded

An early verdict ("gate rollback from probe noise") was **refuted by its own logs**: the
low-probe condition never rejected. The apparent effect came from **unseeded LoRA dropout**
across conditions. Lesson enforced in later runs: seed `torch`, vary exactly one factor, and
read the mechanism logs rather than the summary metric.

## 11.8 ANSWERED: skill acquisition is DATA-LIMITED

Learning-curve sweep (no gate, no evaluator, no solve loop; ground-truth pairs; seeded).

| config | n | steps | rank | skill | general | dGen |
|---|---|---|---|---|---|---|
| baseline | 0 | 0 | 16 | **0.500** | 0.950 | +0.000 |
| n=6 | 6 | 25 | 16 | **0.083** | 0.950 | +0.000 |
| n=6 | 6 | 100 | 16 | 0.250 | 1.000 | +0.050 |
| n=6 | 6 | 300 | 16 | 0.250 | 1.000 | +0.050 |
| n=24 | 24 | 100 | 16 | **1.000** | 1.000 | +0.050 |
| n=24 | 24 | 300 | 16 | **1.000** | 1.000 | +0.050 |
| n=24 +anchors(1:1) | 24 | 300 | 16 | 0.750 | 1.000 | +0.050 |
| n=24 rank64 | 24 | 300 | 64 | 0.917 | 1.000 | +0.050 |

* **NOT optimization-limited** — 100 vs 300 steps is identical at both data sizes.
* **NOT capacity-limited** — rank 64 (0.917) is no better than rank 16 (1.000).
* **DATA-LIMITED, decisively** — 6 examples → 0.250; 24 examples → 1.000.

### 11.8.1 The load-bearing finding
**Training on ~6 examples is WORSE than not training** (skill 0.500 → 0.083–0.250).
Small-data LoRA destroys the capability the model already had. This is precisely the regime
the consolidation loop ran in (`counts={'S0': 6}` per night) — **the loop was damaging itself
every night**, which fully explains §11.3's NIST of 0.000–0.083.

### 11.8.2 Two earlier claims RETRACTED
1. **"The eval-gate rejects real damage" — unsupported.** In the controlled sweep general
   capability never degrades (`dGen = +0.050` everywhere). The pipeline's 0.033–0.083 drops
   were borderline probes flipping under LoRA perturbation, not reproducible damage. `eps`
   must be re-derived against *measured* LoRA jitter, not assumed drift.
2. **Generative replay is counterproductive here.** At 1:1 it cost skill (1.000 → 0.750)
   while protecting general capability that needed no protection. The earlier proposal to
   *raise* the anchor ratio would have made things worse.

### 11.8.3 Statistical scope
12 skill probes (0.083 resolution), 40 general probes, n=1 seed. The n=6→n=24 gap is 9
probes (**decisive**); the anchor penalty is 3 probes (**suggestive**); the rank-64 delta is
1 probe (**noise**). Only the first is asserted.

## 11.9 Required design changes (before any further real-model run)

1. **Consolidate on a per-schema buffer of ≥ ~24 examples, never ~6.** Replace the
   `≥2 successes` eligibility bar with a minimum-examples-per-schema threshold, and/or raise
   `consolidate_every` so nights accumulate enough data. This is the A8 knob — but the real
   constraint is *examples per schema*, not nights.
2. **Add a no-regression floor:** never apply an update trained on fewer than N examples,
   since below the threshold it is strictly harmful.
3. **Drop or heavily downweight generative replay** at this scale; re-test only if a
   regime with genuine forgetting is demonstrated.
4. **Re-derive `eps`** from measured LoRA-perturbation jitter on the probe set.
5. Keep rank 16 and ~100 steps; both saturate.

The consolidation loop's premise — *"a handful of self-generated successes plus light LoRA
yields safe skill acquisition"* — is **false as specified**. Skill acquisition is reachable
(1.000 at n=24), but only above a data threshold the loop never reaches, because online
success rates cannot fill a per-schema buffer that fast. This is the central real-model
result and it supersedes the offline Phase 1–3 magnitudes.

> ⚠️ **RETRACTED BY §11.11.** The single-seed result below did **not** replicate across
> 3 seeds. Read §11.11 before citing anything in §11.10.

## 11.10 CORRECTED LOCUS RUN — H2 apparently supported (SINGLE SEED — later retracted)

All §11.9 changes applied (`min_examples=24`, `n_train=32`, `generative_replay=False`,
`eps=0.10`, `steps=100`, rank 16, parsed replay targets). Qwen2.5-3B, Kaggle T4, ~50 min.

| method | exact_recall | NIST (novel) | gap | GCR |
|---|---|---|---|---|
| B2-memory | 0.667 | 0.250 | −0.417 | 0.000 |
| B7-hybrid | 0.667 | **0.438** | −0.229 | 0.000 |

`NIST delta (B7 − B2) = +0.188`

### 11.10.1 Mechanism confirmed in the logs
```
buffer=29 counts={'S0': 29} need>=24 (floor=24) eligible=['S0'] train=29 target=['S0']
schema S0: probe acc=0.88 -> PROMOTED
GATE ACCEPT: general 0.975->1.000, promoted=1
buffer=33 counts={'S0': 29, 'S1': 4} eligible=['S0'] train=28 target=[]   <- S1 blocked by floor
```
* The **floor worked**: `S0` (29 ex.) trained; `S1` (4 ex.) was refused — the update that
  would have been harmful.
* The **weights learned**: schema probe acc **0.88** on 8 held-out novel probes, vs **0.00**
  in every prior run. Independent of the 16 items NIST is computed on.
* **`eps=0.10` removed the false rejections** — both nights ACCEPT; general rose to 1.000.
* **No forgetting** (`GCR = 0.000`) with generative replay off.

### 11.10.2 The locus signature, isolated
`exact_recall` is **identical** across arms (0.667 — memory's contribution, both arms have
memory), while `NIST` diverges (+0.188 — the weights' contribution). **Memory buys recall;
weights buy generalization.** This is H2's decisive prediction, and it is the first time the
real model has produced it.

### 11.10.3 Scope — what is NOT claimed
1. **n=1 seed, 16 novel items** → +0.188 is **3 items**. Mechanistically corroborated but
   **seed replication is required** before this is conclusive.
2. **Only 1 of 2 schemas trained.** `S1` starved at 4 examples. Online success cannot fill a
   per-schema buffer fast enough for later families — **the residual limitation**.
3. **H3 (forgetting) remains untested.** Nothing degraded and replay was off, so this run
   says nothing about whether replay/gate are *necessary*; it only removed the gate's false
   positives. The §11.8.2 retractions stand.

### 11.10.4 Next
* Replicate across ≥3 seeds × 3 stream orderings (the +0.188 must survive).
* Solve schema starvation: either many more episodes per schema, or a training recipe whose
  no-regression floor sits far below 24 examples.
* Construct a regime with **genuine** forgetting to test H3 at all.

## 11.11 SEED REPLICATION — **H2 NOT REPLICATED** (§11.10 retracted)

Identical corrected configuration, 1 family, 24 novel probes (resolution 0.042), 3 seeds,
paired per-seed deltas. Pre-registered bar: *positive on every seed AND mean > resolution*.

| seed | B2 NIST | B7 NIST | delta | B2 exact | B7 exact | B7 GCR | promoted |
|---|---|---|---|---|---|---|---|
| 0 | 0.500 | 0.417 | **−0.083** | 1.000 | 1.000 | 0.000 | yes |
| 1 | 0.333 | 0.375 | +0.042 | 0.875 | 0.875 | 0.000 | yes |
| 2 | 0.375 | 0.917 | **+0.542** | 1.000 | 1.000 | 0.000 | yes |

`mean = +0.167 ± 0.270 (pstdev)`, positive on **2/3** seeds. Sample SD 0.331 → SE 0.191 →
**t ≈ 0.87. No evidence of an effect.** The mean is carried entirely by seed 2; seed 0 is
negative. **The §11.10 result was seed variance.**

### 11.11.1 What survives
* **The no-regression floor works.** Every seed: `buffer=29–31`, eligible, trained, PROMOTED.
* **The weights do learn**: schema probe acc **0.62 / 0.62 / 0.75** vs **0.00** pre-fix.
* **No forgetting**: `GCR = 0.000` on all six runs, replay off.
* **The locus falsifier passes**: `exact_recall` identical across arms on every seed
  (0.958 both). Memory's contribution is exactly as theorised and stable. Only the
  **weights'** contribution fails to show.

### 11.11.2 Root cause: the task has NO HEADROOM
B2's baseline NIST is **0.333–0.500** — Qwen2.5-3B already solves a third to half of these
novel arithmetic word problems zero-shot. LoRA lifts schema accuracy to ~0.7. Maximum
detectable delta ≈ 0.2–0.35 against seed variance of **0.270**. **The effect is structurally
smaller than the noise.** No tuning of the loop can fix this; the *measurement* cannot
resolve it.

### 11.11.3 Second retraction
**§11.8's headline "n=24 → skill 1.000" was also n=1 seed.** The replication's three
independent probe readings (0.62, 0.62, 0.75) are the better estimate of post-training
skill. The 1.000 was optimistic. The *qualitative* sweep conclusions (data-limited; steps
and rank saturate; small-data updates are harmful) rest on larger gaps and stand — but their
magnitudes do not.

## 11.12 Required redesign: pick a task with headroom

The S-CTRL arithmetic family is **too easy for a 3B instruct model** to test the locus. The
fix is already in the repo: `make_tool_stream` prompts (`[F0|search+filter] tool-task 3,5,7`)
encode a hidden rule the base model **cannot guess** — measured base accuracy **0.00**
(fast *and* deliberate) in the very first bring-up, with LoRA reaching **1.00** on held-out
novel instances.

That is the regime the locus needs:
* `B2` (memory) can cache exact instances but **cannot generalise** → NIST ≈ 0.
* `B7` (weights) should transfer to novel instances → NIST ≫ 0.
* Contrast is maximal; no headroom ceiling; variance cannot swamp it.

**Caveat that must be handled:** with base accuracy 0.00 the agent solves nothing online, so
the replay buffer never fills (`REPLAY STARVATION`) and the floor blocks forever. The
consolidation loop **cannot bootstrap from zero self-successes** — the cold-start problem.
Any tool-stream locus run must therefore seed the buffer (demonstrations) or use a task with
base accuracy in a *middle band* (~0.15–0.30): low enough for headroom, high enough to
bootstrap. **Establishing that band is the next experiment**, not another loop tweak.

## 11.13 THE TRILEMMA (closability check) — no admissible self-generated task

Post-LoRA fast accuracy (24 train / 12 eval, disjoint; H0 = positive control).

| lvl | task | base_fast | base_delib | post_fast | dGen | verdict |
|---|---|---|---|---|---|---|
| H0 | cryptic 1-step rule **[control]** | 0.250 | 0.333 | **0.833** | −0.100 | ✓ closable |
| H2 | 2d×1d + 2d | 0.000 | 1.000 | 0.167 | 0.000 | ✗ |
| H4 | (2d+2d)×1d − 2d | 0.000 | 1.000 | 0.000 | **−0.400** | ✗ + real forgetting |
| H6 | 3-step chain (map-hard, arith-easy) | 0.000 | 0.833 | 0.000 | 0.000 | ✗ |
| H7 | distractor chain | 0.583 | 1.000 | 0.917 | 0.000 | ✗ no headroom |

* **Control lifts, nothing else in-band does** → the zeros are real, not instrument failure.
* **H6 refutes "arithmetic is the obstacle"**: trivial arithmetic, multi-step rule, still
  0/12. The boundary is **single-step vs. multi-step**, not computation vs. rule.
* **H4's dGen = −0.400** is the first genuine catastrophic forgetting → **§11.8.2's "no real
  damage" is UN-retracted**: forgetting is real and task-dependent; the gate has a job.
* **Trilemma:** {headroom, bootstrap, closable} — any two, never three. No self-generated task
  can demonstrate skill acquisition.

## 11.14 FINAL EXPERIMENT: demonstration-seeded locus (cryptic task, 3 seeds)

Seed the parametric buffer with 24 demonstrations (disjoint from all eval sets — verified);
the trilemma's only exit.

| seed | B2 NIST | B7 NIST | delta | B7 GCR | outcome |
|---|---|---|---|---|---|
| 0 | 0.000 | 0.958 | **+0.958** | 0.000 | landed |
| 1 | 0.042 | 0.042 | +0.000 | 0.000 | **gate rollback** (general 1.000→0.875 > eps 0.10) |
| 2 | 0.000 | 1.000 | **+1.000** | +0.125 | landed |

`mean = +0.653 ± 0.566`, positive **2/3**. **Pre-registered every-seed bar NOT met.**

* **Landed seeds: effect at ceiling.** B7 acquires a rule self-replay cannot bootstrap and
  memory cannot generalize (NIST 0 → ~1.0). This is the strongest signal in the project.
* **Not robust:** seed 1 rolled back on real capability damage.
* **exact_recall falsifier tripped** (B2 0.167 vs B7 0.708) — a **design asymmetry**: only the
  parametric buffer was seeded, so on this cold-start task B2's memory is starved and wins
  nothing. The **novel-instance** channel (B2≈0 vs B7≈1) is valid; the clean memory/weights
  *dissociation* is not isolated. Isolating it needs symmetric seeding — and per the trilemma,
  no *self-generated* task admits it.

**Bottom line.** Self-generated replay cannot bootstrap; demonstration-seeded replay can, for
**single-step** skills, **when the update survives the gate** — a narrow, unreliable
competence, not general self-improvement. This is the terminal real-model result; see
[`PAPER.md`](PAPER.md) for the synthesis.
