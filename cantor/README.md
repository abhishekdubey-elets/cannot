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
```

No dependencies beyond the Python 3.10+ standard library for the offline paths.

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
