"""Continual-learning evaluation harness (Phase 0 of EXPERIMENT_PLAN.md).

Turns the CANTOR consolidation loop into a measurable continual-learning study:
streams, retention sets, a general-capability probe, per-night consolidation,
and the standard CL metrics (ACC / BWT / FWT / NIST / GCR / cost).

Offline validation (this package as shipped) drives the REAL CANTOR memory code
for the non-parametric arm and a documented SIMULATED learner for the parametric
arm. Phase 1 swaps in a real backend (`cantor.llm.AnthropicLLM`) and a real LoRA
learner (`RealLoRALearner`) behind the same interfaces — no evaluator changes.
"""

from .evaluator import ContinualResult, evaluate
from .methods import METHODS, MethodConfig
from .streams import Stream, make_stream

__all__ = ["evaluate", "ContinualResult", "METHODS", "MethodConfig",
           "make_stream", "Stream"]
