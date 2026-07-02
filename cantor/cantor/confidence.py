"""Confidence Estimator — report §1.21, §5.5, §10.5.

Combines three complementary signals into one calibrated confidence that the
Executive Controller consumes for routing and the final answer/defer decision:

  1. intrinsic  — the backend's (miscalibrated) logprob, squashed.
  2. consistency — agreement across k independent samples (Wang et al., 2022).
  3. verifier   — pass/fail from an external checker when one exists (RLVR-grade
                  grounding; dominates the other signals when present).

Also produces a crude epistemic/aleatoric split: high sample *disagreement* with
low intrinsic confidence reads as epistemic ("I might not know this"), which the
controller treats as "gather more info / escalate", vs. aleatoric which reads as
"inherently uncertain — report it" (§10.5 failure-recovery ladder).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .llm import LLM, logprob_to_prob


@dataclass
class Confidence:
    score: float                 # calibrated P(answer is correct), in [0, 1]
    epistemic: float             # reducible-by-work share of the uncertainty
    consistency: float           # sample agreement in [0, 1]
    verified: bool | None = None # None = no verifier available

    def uncertain(self, threshold: float) -> bool:
        return self.score < threshold


class ConfidenceEstimator:
    """Stateless estimator; calibration table stands in for a trained head."""

    # Piecewise calibration map fit on "held-out data" (here: hand-set).
    # A real system fits temperature/Platt scaling here (§5.5).
    _CALIBRATION = [(0.9, 0.95), (0.7, 0.75), (0.5, 0.5), (0.3, 0.3), (0.0, 0.1)]

    def __init__(self, llm: LLM, k_samples: int = 3):
        self.llm = llm
        self.k = k_samples

    def _calibrate(self, raw: float) -> float:
        for lo, out in self._CALIBRATION:
            if raw >= lo:
                return out
        return 0.1

    def estimate(self, prompt: str, answer: str, logprob: float,
                 verified: bool | None = None) -> Confidence:
        intrinsic = logprob_to_prob(logprob)

        samples = self.llm.sample(prompt, self.k)
        votes = Counter(s.text.strip() for s in samples)
        votes[answer.strip()] += 1  # the candidate itself is a vote
        consistency = votes.most_common(1)[0][1] / (self.k + 1)

        raw = 0.4 * intrinsic + 0.6 * consistency
        if verified is True:
            raw = max(raw, 0.97)   # grounded check dominates (§5.5)
        elif verified is False:
            raw = min(raw, 0.05)

        # Disagreeing samples + weak intrinsic signal => mostly epistemic.
        epistemic = max(0.0, 1.0 - consistency) * (1.0 - intrinsic)
        return Confidence(self._calibrate(raw), round(epistemic, 3),
                          round(consistency, 3), verified)
