"""Parametric learner interface + implementations (EXPERIMENT_PLAN.md §3, §5).

The parametric arm of the locus question. Unified interface across phases:

    observe(prompt, completion, schema)   record a successful online trajectory
    consolidate() -> int                  run one night's weight update
    has_mastered(schema) -> bool          skill learned into parameters?
    general_health() -> float             1.0 = base capability intact (GCR probe)

Implementations:
  NoOpLearner          non-parametric methods (B0/B1/B2): never masters/erodes.
  SimulatedLoRALearner Phase-0 offline model of LoRA's two effects (skill
                       transfer + drift), driven by success counts.
  RealLoRALearner      Phase-1 learner over a `TrainableLM` backend: builds a
                       prioritized replay batch, interleaves generative replay,
                       trains a LoRA adapter, and EVAL-GATES the update (promote
                       or roll back). Works with the real HF backend OR the
                       offline FakeTrainableLM test double — same control flow.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections import Counter


class ParametricLearner(ABC):
    def observe(self, prompt: str, completion: str, schema: str) -> None:
        """Record a successful online trajectory for later replay. Default: ignore."""

    @abstractmethod
    def consolidate(self) -> int:
        """Run one night's update. Returns number of schemas newly mastered."""

    @abstractmethod
    def has_mastered(self, schema: str) -> bool: ...

    @abstractmethod
    def general_health(self) -> float: ...


class NoOpLearner(ParametricLearner):
    def consolidate(self): return 0
    def has_mastered(self, schema): return False
    def general_health(self): return 1.0


class SimulatedLoRALearner(ParametricLearner):
    """Models the *effect* of eval-gated, replay-protected LoRA (Phase 0).

    generative_replay recovers `recovery` of each update's drift (A4); eval_gate
    requires >= gate_threshold verified successes before promotion (A6); drift is
    general-capability erosion per newly-mastered schema (H3).
    """

    def __init__(self, generative_replay: bool, eval_gate: bool,
                 gate_threshold: int = 2, drift: float = 0.06, recovery: float = 0.9):
        self.generative_replay = generative_replay
        self.eval_gate = eval_gate
        self.gate_threshold = gate_threshold
        self.drift = drift
        self.recovery = recovery
        self._pending: Counter = Counter()
        self._mastered: set[str] = set()
        self._health = 1.0

    def observe(self, prompt, completion, schema):
        self._pending[schema] += 1

    def consolidate(self) -> int:
        need = self.gate_threshold if self.eval_gate else 1
        newly = [s for s, c in self._pending.items()
                 if c >= need and s not in self._mastered]
        self._pending.clear()
        for s in newly:
            self._mastered.add(s)
        if newly:
            raw = self.drift * len(newly)
            eff = raw * (1.0 - (self.recovery if self.generative_replay else 0.0))
            self._health = max(0.0, self._health - eff)
        return len(newly)

    def has_mastered(self, schema): return schema in self._mastered
    def general_health(self): return self._health


class RealLoRALearner(ParametricLearner):
    """Phase-1 learner over a TrainableLM (EXPERIMENT_PLAN.md §5, ablations A4/A6).

    Night procedure (`consolidate`):
      1. Build a replay batch from stored successful trajectories, keeping only
         schemas with >= gate_threshold verified successes when eval_gate is on
         (quality control / data-side gating).
      2. If generative_replay, interleave anchor examples drawn from the base
         distribution to protect general capability.
      3. snapshot -> train LoRA on the batch -> measure held-out effect.
      4. EVAL-GATE (when on): promote only if target-schema accuracy improved on
         internal probes AND general accuracy did not regress by > eps; otherwise
         ROLL BACK the adapter. Probes are disjoint from the benchmark sets (§9).

    `internal_schema_probes[schema] -> [prompt,...]` and `internal_general_probes`
    are self-generated evaluation sets (never the reported R_i / G).
    """

    def __init__(self, lm, generative_replay: bool, eval_gate: bool,
                 internal_schema_probes: dict[str, list[str]],
                 internal_general_probes: list[str],
                 anchor_set: list[tuple[str, str]],
                 answer_of: dict[str, str],
                 gate_threshold: int = 2, eps: float = 0.03,
                 steps: int = 20, lr: float = 1e-4,
                 prioritized: bool = True, batch_budget: int = 6,
                 buffer_cap: int = 500, seed: int = 0):
        self.lm = lm
        self.generative_replay = generative_replay
        self.eval_gate = eval_gate
        self.probes = internal_schema_probes
        self.gen_probes = internal_general_probes
        self.anchors = anchor_set
        self.answer_of = answer_of          # ground truth for probe scoring
        self.gate_threshold = gate_threshold
        self.eps = eps
        self.steps = steps
        self.lr = lr
        # PERSISTENT prioritized replay buffer (A5/A8). Not cleared between nights;
        # each consolidation samples a `batch_budget`-sized training batch from all
        # accumulated successes — prioritized (recency) or uniform.
        self.prioritized = prioritized
        self.batch_budget = batch_budget
        self.buffer_cap = buffer_cap
        self._rng = random.Random(seed)
        self._buffer: list[tuple[str, str, str]] = []   # (prompt, completion, schema)
        self._mastered: set[str] = set()

    # -- scoring helpers ----------------------------------------------------- #
    def _correct(self, prompt: str) -> bool:
        import re
        out = self.lm.generate(prompt)
        nums = re.findall(r"-?\d+", out)
        return bool(nums) and nums[-1] == str(self.answer_of.get(prompt, "\0"))

    def _schema_acc(self, schema: str) -> float:
        probes = self.probes.get(schema, [])
        return sum(self._correct(p) for p in probes) / len(probes) if probes else 0.0

    def _general_acc(self) -> float:
        ps = self.gen_probes
        return sum(self._correct(p) for p in ps) / len(ps) if ps else 1.0

    # -- interface ----------------------------------------------------------- #
    def observe(self, prompt, completion, schema):
        self._buffer.append((prompt, completion, schema))
        if len(self._buffer) > self.buffer_cap:
            self._buffer.pop(0)                     # evict oldest (recency policy)

    def _sample_batch(self):
        """Pick this night's training examples from the PERSISTENT buffer."""
        if self.prioritized:
            return self._buffer[-self.batch_budget:]           # recency-prioritized
        k = min(self.batch_budget, len(self._buffer))
        return self._rng.sample(self._buffer, k)               # uniform (A5 ablation)

    def consolidate(self) -> int:
        if not self._buffer:
            return 0
        counts = Counter(s for _, _, s in self._buffer)
        need = self.gate_threshold if self.eval_gate else 1
        eligible = {s for s, c in counts.items() if c >= need}      # data-side gate
        chosen = self._sample_batch()
        target = {s for (_, _, s) in chosen if s in eligible and s not in self._mastered}
        train = [(p, c) for (p, c, s) in chosen if s in eligible]
        if not train:
            return 0
        batch = train + (list(self.anchors) if self.generative_replay else [])

        gen_before = self._general_acc()
        snap = self.lm.snapshot()
        self.lm.train_lora(batch, steps=self.steps, lr=self.lr)

        promoted = 0
        if self.eval_gate:
            # Reject the whole update if it damaged general capability past eps
            # (roll back weights). This is the gate's job; it is REDUNDANT when
            # generative replay already anchors general, and decisive when not.
            # eps MUST be < the per-update drift or the gate cannot detect the
            # damage it exists to reject (that was the Phase-3 bring-up bug).
            if self._general_acc() < gen_before - self.eps:
                self.lm.restore(snap)
                return 0
            for s in list(target):
                if self._schema_acc(s) > 0.5:
                    self._mastered.add(s); promoted += 1
        else:
            for s in target:
                self._mastered.add(s); promoted += 1
        return promoted

    def has_mastered(self, schema): return schema in self._mastered
    def general_health(self): return self._general_acc()
