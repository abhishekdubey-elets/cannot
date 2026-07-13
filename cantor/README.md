# CANTOR — Reference Implementation

A runnable Python prototype of the architecture designed in
[`../human-reasoning-to-llm-architecture.md`](../human-reasoning-to-llm-architecture.md)
(**C**ognitive **A**rchitecture for **N**eural **T**hought, **O**rchestration, and **R**easoning).

This is a *scaffold-grade* reference implementation: every module from the report
exists as real, documented, testable code with typed interfaces. The LLM itself is
pluggable (`llm.py`) — a deterministic `MockLLM` is included so the whole system
runs offline; swap in `AnthropicLLM`/`OpenAILLM`-style adapters for real inference.

## Docs ↔ Code map

| Report section | Concept | Code |
|---|---|---|
| Part 1 §1.1, Part 5 | System 1 / System 2, router, trigger conditions | [`dual_system.py`](cantor/dual_system.py) |
| Part 1 §1.14, Part 4/5 | Executive Controller (metacognition, compute allocation) | [`controller.py`](cantor/controller.py) |
| Part 1 §1.21, Part 5 §5.5 | Confidence Estimator (intrinsic + consistency + verifier, calibrated) | [`confidence.py`](cantor/confidence.py) |
| Part 1 §1.15, Part 4 | Goal Manager (persistent goal/subgoal stack) | [`goal_manager.py`](cantor/goal_manager.py) |
| Part 7 | Working / Short-term / Episodic / Semantic / Procedural memory, salience-gated writes, decay, retrieval scoring | [`memory.py`](cantor/memory.py) |
| Part 7 §7.5 | Sleep-inspired consolidation (replay → gist → skills) | [`consolidation.py`](cantor/consolidation.py) |
| Part 1 §1.12, §1.18 | Planner (hierarchical decomposition, plan caching → procedural skill) | [`planner.py`](cantor/planner.py) |
| Part 1 §1.13, Part 9 (RLVR) | Reflection / Critic with external-verifier grounding | [`critic.py`](cantor/critic.py) |
| Part 8 | World Model + Simulation module (pluggable; text-sim fallback) | [`world_model.py`](cantor/world_model.py) |
| Part 10 | The CANTOR agent: full execution pipeline + failure-recovery ladder | [`agent.py`](cantor/agent.py) |
| — | LLM backend interface + offline MockLLM | [`llm.py`](cantor/llm.py) |
| — | End-to-end demo | [`demo.py`](demo.py) |

## Run it

```bash
cd cantor
python demo.py          # the CANTOR agent: routing, memory, consolidation
python run_phase0.py    # continual-learning harness (EXPERIMENT_PLAN.md, Phase 0)
python run_phase1.py    # Table 2 locus experiment, offline smoke (Phase 1 wiring)
python run_phase1.py --real meta-llama/Llama-3.1-8B-Instruct   # real model + LoRA
python run_phase2.py    # realistic-domain replication + forward transfer (Phase 2)
python run_phase3.py    # ablation study: which mechanism carries the result (Phase 3)
```

No dependencies beyond the Python 3.10+ standard library for the offline paths;
the `--real` path needs `torch transformers peft` + a GPU.

## Phase 0 continual-learning harness

[`run_phase0.py`](run_phase0.py) + [`cantor/continual/`](cantor/continual/)
implement Phase 0 of [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md): the S-CTRL
controlled stream, retention sets, a general-capability probe, per-night
consolidation, and the CL metrics (ACC / BWT / FWT / NIST / GCR / cost). It
validates the evaluation pipeline and passes the stability gate (B0/B2
reproduce across seeds), and exhibits the qualitative H2/H3 patterns the real
study must show:

| module | role |
|---|---|
| [`continual/streams.py`](cantor/continual/streams.py) | S-CTRL generator: schemas, novel/repeat retention, `G` probe |
| [`continual/learner.py`](cantor/continual/learner.py) | parametric arm: `NoOp` / `SimulatedLoRA` / `RealLoRA` (Phase-1 stub) behind one interface |
| [`continual/solver.py`](cantor/continual/solver.py) | task → (correct, cost, route); drives REAL memory for the non-parametric arm |
| [`continual/metrics.py`](cantor/continual/metrics.py) | ACC/BWT/FWT/NIST/GCR/cost from the accuracy matrix |
| [`continual/evaluator.py`](cantor/continual/evaluator.py) | the stream → night → eval loop |
| [`continual/methods.py`](cantor/continual/methods.py) | B0/B2/B3/B7 (+ Phase-1 configs) as points in the design space |

**Offline scope (important):** the non-parametric arm runs the *real* CANTOR
memory code (episodic write → `sleep()` → semantic cache); the parametric arm
uses a *simulated* learner that models LoRA's two effects (schema skill transfer
and general-capability drift). This validates the harness and metrics — it is
**not** a scientific result. Phase 1 swaps in [`AnthropicLLM`](cantor/cantor/llm.py)
+ a real `RealLoRALearner` behind the same interfaces; no evaluator change.

**Known/expected harness artifact:** FWT ≈ 0 on S-CTRL because its schemas are
independent (no shared substructure to transfer). That is *correct* for this
controlled stream; the realistic benchmarks (AppWorld/τ-bench) are where FWT > 0
becomes a meaningful signal.

## Phase 1 — the locus experiment (real rollouts + real LoRA)

[`run_phase1.py`](run_phase1.py) runs the Table 2 go/no-go: model-in-the-loop
solving via a `TrainableLM`, correct trajectories consolidated into a LoRA adapter
with **eval-gating and rollback**, and the non-parametric memory recalling exact
instances. The parametric arm is behind these interfaces:

| module | role |
|---|---|
| [`continual/trainable.py`](cantor/continual/trainable.py) | `TrainableLM`: `HFTrainableLM` (real LoRA, import-guarded) + `FakeTrainableLM` (offline test double) |
| [`continual/learner.py`](cantor/continual/learner.py) | `RealLoRALearner`: prioritized replay batch → train → eval-gate → promote-or-rollback |
| [`continual/rollout_solver.py`](cantor/continual/rollout_solver.py) | `RolloutSolver`: solve by generation; effortful online solve vs. flaky fast-path eval; real memory cache |

The **same evaluator** runs both phases — Phase 1 just injects a `RolloutSolver`
+ `RealLoRALearner` into `evaluate(...)`. The offline smoke reproduces the locus
thesis (memory aces exact recall, lags novel transfer by ~0.5; weights close the
gap; eval-gate + replay hold GCR ≈ 0 while naive LoRA drops it ~0.24) — passing
`LOCUS GO/NO-GO: GO`. This validates the full control flow including LoRA
batch-building, eval-gating, and rollback; the numeric training is a test double
until `--real` swaps in `HFTrainableLM`.

## Phase 2 — realistic-domain replication + forward transfer

[`run_phase2.py`](run_phase2.py) checks the locus finding **replicates** on a
domain with shared substructure (S-TOOL: tool tasks whose families overlap by a
primitive skill), where **forward transfer becomes measurable** — the signal the
controlled independent-schema stream cannot produce (Phase 1 has FWT ≡ 0 by
construction).

| module | role |
|---|---|
| [`continual/streams.py`](cantor/continual/streams.py) `make_tool_stream` | S-TOOL: families sharing tool primitives → FWT > 0 |
| [`continual/trainable.py`](cantor/continual/trainable.py) `ToolFakeLM` | primitive-level mastery: partial mastery raises pre-training accuracy |
| [`continual/benchmarks.py`](cantor/continual/benchmarks.py) | `BenchmarkAdapter` + `SyntheticToolAdapter` (offline) + `AppWorld`/`tau-bench` stubs with the real `to_stream` contract |

Offline smoke result: locus replicates (B2 exact-recall 0.84 vs novel 0.26, gap
−0.58), **FWT emerges** for parametric methods (~0.40) while memory-only shows ~0,
and hybrid forgets far less than naive — `PHASE 2 GATE: PASS`. Real AppWorld /
τ-bench plug in behind `BenchmarkAdapter` with no evaluator change.

## Running `--real` (Lightning.ai Studio / Kaggle 2×T4 / Colab)

The `--real` path swaps `FakeTrainableLM` for `HFTrainableLM` (4-bit NF4 QLoRA,
fp16 on T4). **No platform API key is needed inside a Studio/notebook — you are
already on the compute.** Any *model* gating (e.g. Llama) uses a Hugging Face
token from the platform's Secrets manager, never hardcoded.

**Run this ON the GPU box.** If you provisioned a remote VM (e.g. Lightning
`vm ssh`), you must be inside that Linux session — a `C:\Users\...` traceback
means you're still on your laptop and the GPU is idle. `HFTrainableLM` raises a
clear error if `torch` sees no CUDA device.

```bash
pip install -r requirements-gpu.txt
# optional, only for gated models:  huggingface-cli login   (or set HF_TOKEN via Secrets)

python run_phase0.py                                   # 1. sanity: offline harness
python run_phase1.py                                   # 2. sanity: offline smoke (GO)
export CANTOR_4BIT=0                                    # big GPU (H100/A100)? skip 4-bit, run bf16
python run_phase1.py --real Qwen/Qwen2.5-3B-Instruct   # 3. real QLoRA locus run
```

On an 80 GB H100 the model loads onto one GPU with room to spare; leave
`CANTOR_4BIT` unset (keep 4-bit) only on small cards like a single T4.

Sizing: `Qwen2.5-3B-Instruct` is ungated and comfortable on one T4; 8B fits in
4-bit. **Skip the 32B scale check and real AppWorld/τ-bench on T4** — those need
A100-class GPUs (see the GPU discussion). Verify one short run end-to-end before
the full 3-seed sweep. `HFTrainableLM` is untested in this repo's authoring env
(no GPU there); the offline smokes validate everything except the numeric kernels.

## Phase 3 — ablation study (which mechanism carries the result)

[`run_phase3.py`](run_phase3.py) + [`continual/harness.py`](cantor/continual/harness.py)
(the `AblationConfig` knobs). Removes one mechanism at a time from the Full
hybrid and checks the gate: **each modeled mechanism moves its decisive metric.**

The headline is the **replay × eval-gate interaction** — a cleaner result than
isolated ablations, because the two are partially-redundant protections:

|            | gate ON (NIST / GCR) | gate OFF (NIST / GCR) |
|---|---|---|
| **replay ON**  | 0.96 / −0.04  (Full) | 1.00 / −0.05 (gate redundant) |
| **replay OFF** | 0.26 / **0.00** (gate refuses unsafe updates → no forgetting, no learning) | 1.00 / **−0.30** (catastrophic forgetting) |

Reading: generative replay is what lets you *learn **and** retain*; the eval-gate
alone can only protect general capability by *refusing to learn*. Plus A8
(consolidation-frequency, U-shaped) and A9 (capacity↔forgetting) curves. Removing
**parametric** collapses NIST (0.96→0.26). **Memory / prioritization / salience-gate**
are reported as honest offline no-ops (reasons in the runner header) rather than
faked. `PHASE 3 GATE: PASS`.

Bring-up here surfaced two real design bugs, fixed not tuned: the eval-gate
entangled rollback with per-schema promotion (now a clean "reject if general
regressed > eps"), and **`eps` was larger than the drift it had to detect** — a
gate blind to its own target. Fixing `eps < drift` plus sizing the gate's eval
set so measurement noise < `eps` made the interaction separate cleanly.

## Phase status

| phase | scope | status |
|---|---|---|
| 0 | harness + metrics + stability gate | ✅ gate PASS |
| 1 | locus experiment: real rollouts + eval-gated LoRA control flow | ✅ smoke GO (measured result needs `--real` GPU backend) |
| 2 | realistic-domain replication + forward transfer | ✅ gate PASS (offline; real adapters stubbed) |
| 3 | ablations + replay×gate interaction + A8/A9 curves | ✅ gate PASS (offline; 32B scale check needs A100-class GPU) |
| **real** | Qwen2.5-3B on Kaggle T4: bring-up → sweep → locus → replication → **trilemma → seeded locus** | ✅ backend · ⚠️ **narrow/unreliable result** — [`PAPER.md`](PAPER.md), [`EXPERIMENT_PLAN.md` §11](EXPERIMENT_PLAN.md) |

> **The real-model story is written up in [`PAPER.md`](PAPER.md).** Headline:
> **self-generated replay cannot bootstrap skill acquisition; demonstration-seeded replay can,
> for single-step skills, when the update survives the eval-gate** — narrow and unreliable,
> not general self-improvement.
>
> Chain of measured results: (1) below ~24 examples/schema, LoRA updates are *harmful* — the
> loop damaged itself nightly; a **no-regression floor** fixed it. (2) Fixed, the single-seed
> locus looked positive (+0.188) but **did not replicate** across 3 seeds (`+0.167 ± 0.270`,
> negative on one) — the task had no headroom. (3) A **trilemma**: headroom, bootstrap, and
> closability are mutually exclusive, so *no self-generated task* can show skill acquisition;
> and LoRA distils single-step rules but **not multi-step procedures** at 3B. (4) Seeding the
> buffer with demonstrations — the trilemma's only exit — gives near-perfect novel-instance
> transfer (0.00 → ~1.0) on **2/3 seeds**, the third rolled back on real forgetting.
>
> Four single-seed claims were retracted along the way; the offline harness passed every gate
> and predicted none of it. Simulation validates *pipelines*, not *effect sizes*.

The **measured** Phases 1–2 (real models/benchmarks) require the `--real` /
adapter backends (GPU + `torch transformers peft`); the offline runs validate the
full pipeline, control flow, and metric computations that those backends slot into.

## What the demo shows

1. An easy query routed to **System 1** (one pass, high confidence, cheap).
2. A hard/novel query **escalated to System 2**: retrieval → plan → per-subgoal
   reasoning → critic verification → revision loop → calibrated final confidence.
3. A query where confidence stays below the stakes threshold → the agent **defers**
   ("I don't know" as a first-class action — Part 10 §10.5).
4. Episodic writes during the session, then an offline **consolidation pass** that
   distills episodes into semantic facts and a cached procedural skill — so the
   same hard query the *second* time is answered from the skill cache at System-1
   cost (the "search → distill → faster policy" flywheel, Part 6 §6.3).

## Honest scope notes

- The `MockLLM` is deterministic pattern-matching — it exists to exercise the
  *architecture* (routing, memory, loops), not to be smart.
- The confidence estimator uses self-consistency spread + a calibration table;
  with a real backend you would add a trained intrinsic head (report §5.5).
- The world model is a text-simulation stub with the correct interface
  (`predict(state, action) -> next_state`); a JEPA/Dreamer-style latent model is
  the research upgrade (report Part 8).
- Consolidation updates *memory stores*, not weights. Eval-gated LoRA updates are
  the production analog (report §7.5, §10.7).
