"""Shared run-builder for the ablation study (EXPERIMENT_PLAN.md §7, Phase 3).

An `AblationConfig` exposes every mechanism as a knob; `build_run` assembles the
matching (lm, memory, learner, solver) and drives the shared evaluator. Removing
one knob at a time gives Table 4; sweeping `consolidate_every` / `batch_budget`
gives the A8 / A9 curves.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..agent import CantorAgent
from ..llm import MockLLM
from .benchmarks import BenchmarkAdapter
from .evaluator import ContinualResult, evaluate
from .learner import NoOpLearner, RealLoRALearner
from .methods import MethodConfig
from .rollout_solver import RolloutSolver
from .streams import Stream


@dataclass
class AblationConfig:
    name: str
    use_memory: bool = True
    use_parametric: bool = True
    generative_replay: bool = True
    eval_gate: bool = True
    prioritized: bool = True
    salience_gate: bool = True
    consolidate_every: int = 1       # A8 curve
    batch_budget: int = 6            # A9 curve (replay budget ~ LoRA capacity)
    drift: float | None = None       # A9 override: capacity<->forgetting coupling


def oracle(*streams: Stream) -> dict[str, str]:
    o: dict[str, str] = {}
    for st in streams:
        for f in st.families:
            for t in f.train + f.retention_novel + f.retention_repeat:
                o[t.prompt] = t.answer
        for t in st.general_probe:
            o[t.prompt] = t.answer
    return o


def internal_probes(ps: Stream):
    # 60 general probes so the eval-gate's measurement noise (~1/60) is well
    # below eps (0.03) and below the per-update drift it must detect (0.06).
    sp = {f.schema: [t.prompt for t in f.retention_novel[:8]] for f in ps.families}
    gp = [t.prompt for t in ps.general_probe[:60]]
    anchors = [(t.prompt, t.answer) for t in ps.general_probe[:16]]
    return sp, gp, anchors


def build_run(adapter: BenchmarkAdapter, cfg: AblationConfig, seed: int) -> ContinualResult:
    stream = adapter.to_stream(seed)
    probe = adapter.to_stream(seed + 1000)          # disjoint internal probes
    orc = oracle(stream, probe)

    lm = adapter.make_lm(orc, stream.base_success)
    if cfg.drift is not None and hasattr(lm, "drift"):
        lm.drift = cfg.drift                        # A9: higher capacity -> more drift
        lm.anchor_drift = cfg.drift * (1.0 - getattr(lm, "recovery", 0.9))

    agent = CantorAgent(MockLLM())
    if cfg.use_parametric:
        sp, gp, anchors = internal_probes(probe)
        learner = RealLoRALearner(lm, cfg.generative_replay, cfg.eval_gate,
                                  internal_schema_probes=sp, internal_general_probes=gp,
                                  anchor_set=anchors, answer_of=orc,
                                  prioritized=cfg.prioritized,
                                  batch_budget=cfg.batch_budget, seed=seed)
    else:
        learner = NoOpLearner()

    solver = RolloutSolver(lm, agent, cfg.use_memory, cfg.use_parametric,
                           salience_gate=cfg.salience_gate)
    mc = MethodConfig(cfg.name, cfg.use_memory, cfg.use_parametric)
    return evaluate(mc, stream, seed, solver=solver, learner=learner,
                    consolidate_every=cfg.consolidate_every)
