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
