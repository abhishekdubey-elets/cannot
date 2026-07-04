"""Benchmark adapters (EXPERIMENT_PLAN.md §4, Phase 2).

Maps an external agentic benchmark onto the harness's Stream/Task abstractions so
the SAME evaluator + metrics run on it. The contract an adapter must satisfy:

    to_stream(seed) -> Stream
        families        : task types / templates introduced sequentially
                          (a "schema"); realistic ones share substructure
        retention_novel : held-out NEW instances of the type -> NIST / transfer
        retention_repeat: EXACT repeats -> exact-instance recall (memory's job)
        general_probe   : a held-out capability the stream never trains on -> GCR

    make_lm(oracle, base_success) -> TrainableLM
        the fine-tunable backend for this domain (real HF model in production).

SyntheticToolAdapter is the offline Phase-2 domain (shared primitives -> FWT>0).
AppWorldAdapter / TauBenchAdapter are the real Phase-2 targets: stubs documenting
exactly how to project each benchmark's tasks into the four buckets above.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .streams import Stream, make_tool_stream
from .trainable import ToolFakeLM, TrainableLM


class BenchmarkAdapter(ABC):
    name: str

    @abstractmethod
    def to_stream(self, seed: int) -> Stream: ...

    @abstractmethod
    def make_lm(self, oracle: dict, base_success: dict) -> TrainableLM: ...


class SyntheticToolAdapter(BenchmarkAdapter):
    """Offline realistic-style domain: tool tasks with shared primitives."""
    name = "S-TOOL"

    def __init__(self, n_families: int = 6):
        self.n_families = n_families

    def to_stream(self, seed: int) -> Stream:
        return make_tool_stream(seed=seed, n_families=self.n_families)

    def make_lm(self, oracle, base_success) -> TrainableLM:
        return ToolFakeLM(oracle, base_success)


class AppWorldAdapter(BenchmarkAdapter):  # pragma: no cover - Phase 2 real
    """AppWorld (tool/agent tasks). To implement `to_stream`:
      * family  = an app/API scenario template (e.g. "send money", "file expense")
      * novel   = same template, unseen entities/values
      * repeat  = a previously-seen task instance verbatim
      * general = a held-out app family never trained on (GCR probe)
      * base_success = frozen-model pass@1 per family (calibrates difficulty)
    `make_lm` returns HFTrainableLM(model); correctness = task's own verifier.
    """
    name = "AppWorld"

    def to_stream(self, seed):
        raise NotImplementedError("Wire the AppWorld loader per this docstring.")

    def make_lm(self, oracle, base_success):
        from .trainable import HFTrainableLM
        return HFTrainableLM()


class TauBenchAdapter(BenchmarkAdapter):  # pragma: no cover - Phase 2 real
    """tau-bench (tool-agent, retail/airline). Same projection as AppWorld;
    family = a policy/task template, verifier = tau-bench's reward check."""
    name = "tau-bench"

    def to_stream(self, seed):
        raise NotImplementedError("Wire the tau-bench loader per this docstring.")

    def make_lm(self, oracle, base_success):
        from .trainable import HFTrainableLM
        return HFTrainableLM()


ADAPTERS = {a.name: a for a in [SyntheticToolAdapter()]}
