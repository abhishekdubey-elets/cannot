"""Method registry (EXPERIMENT_PLAN.md §5).

Each method is a point in the design space (memory on/off x parametric on/off x
replay/gating). Phase 0 runs the four that define the harness gate and the
core H2/H3 story; the rest are registered with `phase=1` so the same evaluator
covers them once a real backend + LoRA learner are wired in.
"""

from __future__ import annotations

from dataclasses import dataclass

from .learner import NoOpLearner, ParametricLearner, SimulatedLoRALearner


@dataclass(frozen=True)
class MethodConfig:
    name: str
    use_memory: bool          # non-parametric episodic/semantic/procedural stores
    use_parametric: bool      # weight updates via the learner
    generative_replay: bool = False
    eval_gate: bool = False
    phase: int = 0            # 0 = runs in the offline harness; 1 = needs real LoRA
    note: str = ""

    def make_learner(self) -> ParametricLearner:
        if not self.use_parametric:
            return NoOpLearner()
        return SimulatedLoRALearner(self.generative_replay, self.eval_gate)


METHODS: dict[str, MethodConfig] = {
    "B0-frozen":    MethodConfig("B0-frozen", False, False, note="floor"),
    "B2-memory":    MethodConfig("B2-memory", True, False, note="non-parametric only (H2 arm)"),
    "B3-naive-lora": MethodConfig("B3-naive-lora", False, True,
                                  generative_replay=False, eval_gate=False,
                                  note="online LoRA, no replay -> forgetting (H3 arm)"),
    "B7-hybrid":    MethodConfig("B7-hybrid", True, True,
                                 generative_replay=True, eval_gate=True,
                                 note="ours: memory + eval-gated, replay-protected LoRA"),

    # Phase-1 configs (need real backend + LoRA); registered for completeness.
    "B1-raw-rag":   MethodConfig("B1-raw-rag", True, False, phase=1,
                                 note="raw-transcript retrieval baseline"),
    "B4-rehearsal": MethodConfig("B4-rehearsal", False, True, generative_replay=True,
                                 eval_gate=False, phase=1, note="rehearsal LoRA"),
}

PHASE0_METHODS = ["B0-frozen", "B2-memory", "B3-naive-lora", "B7-hybrid"]
