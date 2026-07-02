"""Executive Controller — report §1.14, Part 4, §5.3–5.4.

The metacognitive kernel: reads (confidence, stakes, novelty, conflict, budget)
and picks a route. This mirrors the default-interventionist policy (§1.1):
System 1 runs first and the controller decides whether its answer is good
enough or deliberation must be recruited.

Here the policy is an interpretable rule table; the production version is a
small RL-trained policy rewarded on task success minus compute cost (§10.7
step 4). The rule structure below is exactly the feature set that policy
would consume.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .confidence import Confidence


class Route(Enum):
    ANSWER = "answer"            # accept the fast answer
    VERIFY_LIGHT = "verify"      # fast answer + one critic pass
    SYSTEM2 = "system2"          # retrieve + plan + deliberate + critique
    DEFER = "defer"              # answer withheld; report uncertainty / ask


@dataclass
class ControlState:
    confidence: Confidence
    stakes: float        # 0..1  (irreversibility / importance)
    novelty: float       # 0..1  (distance from known task signatures)
    budget_left: float   # 0..1  (fraction of compute budget remaining)


class ExecutiveController:
    # Escalation thresholds — the knobs the RL policy would learn (§5.6:
    # calibration quality sets the cost/accuracy frontier).
    ACCEPT_CONF = 0.85
    VERIFY_CONF = 0.60

    def route(self, s: ControlState) -> Route:
        # Trigger conditions from §5.3, in priority order.
        if s.budget_left <= 0.0:
            # Out of budget: honest deferral beats a confident guess (§10.5).
            return Route.ANSWER if s.confidence.score >= self.VERIFY_CONF else Route.DEFER

        if s.confidence.score >= self.ACCEPT_CONF and s.stakes < 0.7 and s.novelty < 0.7:
            return Route.ANSWER

        if s.confidence.score >= self.VERIFY_CONF and s.stakes < 0.4:
            return Route.VERIFY_LIGHT

        # Low confidence, high stakes, or novelty -> recruit System 2...
        # ...but only if the uncertainty is epistemic (work can reduce it).
        if s.confidence.epistemic < 0.05 and s.confidence.consistency > 0.9:
            # Aleatoric: more thinking won't help; report the uncertainty.
            return Route.DEFER
        return Route.SYSTEM2

    def post_deliberation(self, conf: Confidence, stakes: float) -> Route:
        """Final gate after System 2 + critic (§10.2 step 7)."""
        threshold = 0.5 + 0.4 * stakes   # higher stakes demand more confidence
        return Route.ANSWER if conf.score >= threshold else Route.DEFER
