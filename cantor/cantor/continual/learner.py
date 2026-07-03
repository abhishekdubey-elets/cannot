"""Parametric learner interface + implementations (EXPERIMENT_PLAN.md §3, §5).

The parametric arm of the locus question. `consolidate()` is the offline "night"
weight update; `has_mastered()` reports whether a schema's skill has been learned
into the parameters; `general_health()` in [0,1] tracks erosion of base
capability (1.0 = untouched) — the mechanism GCR measures.

Three implementations:
  NoOpLearner          non-parametric methods (B0/B1/B2): never masters, never erodes.
  SimulatedLoRALearner offline stand-in modeling the two effects a real LoRA
                       update produces: (+) schema skill transfer to novel
                       instances, and (-) drift in general capability that
                       generative replay recovers and eval-gating throttles.
  RealLoRALearner      Phase-1 PEFT/LoRA implementation (import-guarded stub with
                       the concrete training flow documented).
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class ParametricLearner(ABC):
    @abstractmethod
    def consolidate(self, schema_success_counts: dict[str, int]) -> int:
        """Run one night's update. Returns number of schemas newly mastered."""

    @abstractmethod
    def has_mastered(self, schema: str) -> bool: ...

    @abstractmethod
    def general_health(self) -> float: ...


class NoOpLearner(ParametricLearner):
    def consolidate(self, schema_success_counts): return 0
    def has_mastered(self, schema): return False
    def general_health(self): return 1.0


class SimulatedLoRALearner(ParametricLearner):
    """Models the *effect* of eval-gated, replay-protected LoRA consolidation.

    Parameters mirror the ablations (Table 4):
      generative_replay  : recovers `recovery` fraction of each update's drift (A4)
      eval_gate          : require >= `gate_threshold` verified successes before a
                           schema is promoted (A6) — quality control
      drift              : general-capability erosion per newly-mastered schema (H3)
    """

    def __init__(self, generative_replay: bool, eval_gate: bool,
                 gate_threshold: int = 2, drift: float = 0.06, recovery: float = 0.9):
        self.generative_replay = generative_replay
        self.eval_gate = eval_gate
        self.gate_threshold = gate_threshold
        self.drift = drift
        self.recovery = recovery
        self._mastered: set[str] = set()
        self._health = 1.0

    def consolidate(self, schema_success_counts: dict[str, int]) -> int:
        need = self.gate_threshold if self.eval_gate else 1
        newly = [s for s, c in schema_success_counts.items()
                 if c >= need and s not in self._mastered]
        for s in newly:
            self._mastered.add(s)
        if newly:
            raw = self.drift * len(newly)
            effective = raw * (1.0 - (self.recovery if self.generative_replay else 0.0))
            self._health = max(0.0, self._health - effective)
        return len(newly)

    def has_mastered(self, schema): return schema in self._mastered
    def general_health(self): return self._health


class RealLoRALearner(ParametricLearner):  # pragma: no cover - Phase 1
    """Phase-1 implementation sketch. Requires torch + transformers + peft.

    consolidate():
      1. Build a training batch from prioritized replay of successful trajectories
         whose schema has >= gate_threshold verified successes (eval-gating).
      2. Interleave GENERATIVE REPLAY samples from the base distribution (prompts
         drawn from a frozen copy of the base model) to anchor general capability.
      3. Fit / merge a LoRA adapter (rank r) for a few steps.
      4. EVAL-GATE: promote the adapter only if it improves held-out self-generated
         schema accuracy AND does not regress an internal general probe by > eps;
         else roll back. `has_mastered` reflects promoted schemas.
    general_health() would be an actual measured probe score, not a scalar model.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "RealLoRALearner is the Phase-1 target; install torch/peft and "
            "implement the flow in this docstring. Offline harness uses "
            "SimulatedLoRALearner.")

    def consolidate(self, schema_success_counts): ...
    def has_mastered(self, schema): ...
    def general_health(self): ...
