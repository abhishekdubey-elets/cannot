"""LLM backend interface.

The architecture treats the base LLM as System 1 (report §1.1, §2.5): a single
forward pass that returns a candidate answer plus raw signals the metacognitive
layer can consume. Everything above this interface (routing, memory, planning,
reflection) is backend-agnostic — swap `MockLLM` for a real API adapter without
touching the rest of the system.
"""

from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class LLMResponse:
    """One forward pass worth of output.

    `logprob` is the mean token log-probability of the completion — the raw
    (miscalibrated, §1.21) intrinsic signal the ConfidenceEstimator calibrates.
    """

    text: str
    logprob: float = -1.0
    meta: dict = field(default_factory=dict)


class LLM(ABC):
    """Minimal generation interface every backend must implement."""

    @abstractmethod
    def generate(self, prompt: str, temperature: float = 0.7) -> LLMResponse: ...

    def sample(self, prompt: str, k: int, temperature: float = 0.9) -> list[LLMResponse]:
        """k independent samples — used for self-consistency spread (§5.5)."""
        return [self.generate(f"{prompt}\n[sample:{i}]", temperature) for i in range(k)]


class MockLLM(LLM):
    """Deterministic offline backend for exercising the architecture.

    Behavior model:
      - Questions containing a fact from `knowledge` are answered confidently
        and consistently  -> System 1 handles them.
      - Arithmetic expressions are answered *unreliably* at high temperature
        (samples disagree) but reliably when a "step by step" cue is present
        -> forces escalation to System 2, where the critic/verifier grounds it.
      - Anything else gets a low-confidence guess -> exercises defer/ask.
    """

    knowledge = {
        "capital of france": "Paris",
        "boiling point of water": "100 degrees Celsius at sea level",
        "author of hamlet": "William Shakespeare",
    }

    def _seed(self, prompt: str) -> int:
        return int(hashlib.md5(prompt.encode()).hexdigest(), 16)

    def generate(self, prompt: str, temperature: float = 0.7) -> LLMResponse:
        p = prompt.lower()

        for cue, answer in self.knowledge.items():
            if cue in p:
                return LLMResponse(answer, logprob=-0.05)

        expr = re.search(r"(\d+)\s*([*+x])\s*(\d+)", p)
        if expr:
            a, op, b = int(expr.group(1)), expr.group(2), int(expr.group(3))
            true = a * b if op in "*x" else a + b
            deliberate = "step by step" in p or "subgoal" in p
            if deliberate or temperature < 0.3:
                return LLMResponse(str(true), logprob=-0.2, meta={"trace": f"{a}{op}{b} computed digit-wise"})
            # Fast path: plausible-but-unreliable pattern completion (§1.6).
            noise = self._seed(prompt) % 7
            guess = true if noise < 3 else true + (noise - 3) * max(1, true // 50)
            return LLMResponse(str(guess), logprob=-1.8)

        # Out-of-distribution: fluent confabulation with weak signal (§2.5).
        return LLMResponse(
            f"(uncertain guess about: {prompt[:48]}...)",
            logprob=-3.0 - (self._seed(prompt) % 5) / 2.0,
        )


def logprob_to_prob(logprob: float) -> float:
    """Squash mean logprob into (0,1) as a crude pre-calibration score."""
    return 1.0 / (1.0 + math.exp(-(logprob + 2.0) * 2.0))


class AnthropicLLM(LLM):
    """Real backend adapter (Phase 1). Requires `pip install anthropic` and an
    API key. Kept behind a lazy import so the offline harness never needs it.

    Note: chat APIs do not expose per-token logprobs, so the intrinsic
    confidence signal (§5.5) is weak here — the ConfidenceEstimator leans on
    self-consistency + verifier signals instead, exactly as the report argues
    for post-RLHF models whose logits are miscalibrated (§1.21).
    """

    def __init__(self, model: str = "claude-sonnet-5", api_key: str | None = None,
                 max_tokens: int = 1024):
        try:
            import anthropic  # noqa: F401
        except ImportError as e:  # pragma: no cover - env-dependent
            raise ImportError("pip install anthropic to use AnthropicLLM") from e
        import anthropic
        self._client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    def generate(self, prompt: str, temperature: float = 0.7) -> LLMResponse:  # pragma: no cover
        msg = self._client.messages.create(
            model=self.model, max_tokens=self.max_tokens, temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        # No logprobs from the chat API -> flag as unknown (-1.0); confidence
        # is then driven by consistency + verifier (§5.5).
        return LLMResponse(text.strip(), logprob=-1.0,
                           meta={"usage": getattr(msg, "usage", None)})
