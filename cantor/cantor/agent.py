"""CantorAgent — the full execution pipeline of report §10.2.

    1. INGEST      goal onto GoalManager stack
    2. PERCEIVE    build working set (goal reinjection + memory retrieval)
    3. FAST PASS   System 1 candidate + confidence
    4. META-ROUTE  ExecutiveController picks a route
    5. DELIBERATE  plan -> per-subgoal slow reasoning + world model
    6. CRITIQUE    critic verifies; bounded revision loop
    7. DECIDE      calibrated confidence gates answer vs defer
    8. RECORD      salience-gated episodic write; skill caching on success
   10. (offline)   agent.sleep() runs consolidation

Failure-recovery ladder (§10.5): revise -> use verifier's correction ->
defer. "I don't know" is a terminal action, not an error state.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .confidence import Confidence, ConfidenceEstimator
from .consolidation import ConsolidationEngine, ConsolidationReport
from .controller import ControlState, ExecutiveController, Route
from .critic import Critic
from .goal_manager import GoalManager
from .llm import LLM
from .memory import (Episode, EpisodicMemory, ProceduralMemory, SemanticMemory,
                     WorkingMemory, task_signature)
from .planner import Planner
from .world_model import CompositeWorldModel


@dataclass
class AgentResult:
    answer: str | None          # None => deferred
    route: str                  # "system1" | "system1+verify" | "system2" | "skill-cache"
    confidence: Confidence
    cost_units: float           # relative compute spent (§5.6 accounting)
    trace: list[str] = field(default_factory=list)

    @property
    def deferred(self) -> bool:
        return self.answer is None


class CantorAgent:
    MAX_REVISIONS = 2

    def __init__(self, llm: LLM):
        self.llm = llm
        self.wm = WorkingMemory()
        self.episodic = EpisodicMemory()
        self.semantic = SemanticMemory()
        self.procedural = ProceduralMemory()
        self.goals = GoalManager()
        self.world_model = CompositeWorldModel(llm)
        self.planner = Planner(llm, self.procedural)
        self.critic = Critic(llm, self.world_model)
        self.confidence = ConfidenceEstimator(llm)
        self.controller = ExecutiveController()
        self.consolidator = ConsolidationEngine(self.episodic, self.semantic,
                                                self.procedural)

    # ------------------------------------------------------------------ #

    def respond(self, query: str, stakes: float = 0.3) -> AgentResult:
        trace: list[str] = []
        cost = 0.0

        # 1-2. INGEST + PERCEIVE
        goal = self.goals.push(query)
        self.wm.clear_role("goal")
        self.wm.write("goal", self.goals.reinject(), priority=1.0)
        for ep in self.episodic.retrieve(query, k=2):
            self.wm.write("retrieved", f"past: {ep.summary} [{ep.outcome}]", priority=0.6)
        for fact in self.semantic.query(query, k=2):
            self.wm.write("retrieved", f"fact: {fact.render()}", priority=0.7)

        # Semantic shortcut: a consolidated exact-answer fact for this precise
        # query answers below System-1 cost — yesterday's System-2 derivation
        # is today's recall (§6.3 flywheel). Exact-match only; the normalized
        # signature must never key a concrete answer.
        for fact in self.semantic.facts:
            if fact.relation == "answer_is" and fact.subject.lower() == query.lower():
                trace.append(f"semantic-cache hit (src={fact.source})")
                conf = Confidence(score=fact.confidence, epistemic=0.0, consistency=1.0)
                self.goals.complete(goal)
                return AgentResult(fact.obj, "semantic-cache", conf, 0.5, trace)

        # 3. FAST PASS (System 1)
        fast = self.llm.generate(query)
        cost += 1.0
        conf = self.confidence.estimate(query, fast.text, fast.logprob)
        cost += 0.5  # consistency samples are cheap partial passes
        novelty = 0.0 if self.procedural.lookup(task_signature(query)) else 0.5
        trace.append(f"S1: '{fast.text}' conf={conf.score} cons={conf.consistency}")

        # 4. META-ROUTE
        route = self.controller.route(ControlState(conf, stakes, novelty, budget_left=1.0))
        trace.append(f"controller -> {route.value}")

        if route == Route.ANSWER:
            self._record(query, f"{query} => {fast.text}", "success", salience=0.2)
            self.goals.complete(goal)
            return AgentResult(fast.text, "system1", conf, cost, trace)

        if route == Route.VERIFY_LIGHT:
            verdict = self.critic.verify(query, fast.text)
            cost += 1.0
            answer = verdict.corrected if (not verdict.ok and verdict.corrected) else fast.text
            conf = self.confidence.estimate(query, answer, fast.logprob,
                                            verified=verdict.ok or bool(verdict.corrected))
            trace.append(f"light verify: {verdict.feedback}")
            self._record(query, f"{query} => {answer}", "success", salience=0.4)
            self.goals.complete(goal)
            return AgentResult(answer, "system1+verify", conf, cost, trace)

        if route == Route.DEFER:
            self._record(query, f"deferred: {query}", "deferred", salience=0.5)
            self.goals.complete(goal)
            return AgentResult(None, "system1", conf, cost, trace)

        # 5. DELIBERATE (System 2)
        return self._deliberate(query, goal, stakes, cost, trace)

    # ------------------------------------------------------------------ #

    def _deliberate(self, query: str, goal, stakes: float, cost: float,
                    trace: list[str]) -> AgentResult:
        plan = self.planner.plan(query)
        trace.append(f"plan{' (cached)' if plan.from_cache else ''}: {plan.subgoals}")
        self.wm.write("plan", " | ".join(plan.subgoals), priority=0.9)

        answer = ""
        for sub in plan.subgoals:
            subgoal = self.goals.push(sub, parent=goal)
            out = self.llm.generate(f"{self.wm.render()}\nsubgoal: {sub}", temperature=0.2)
            cost += 3.0  # slow-path passes carry long context (§5.6)
            if out.text and not out.text.startswith("("):
                answer = out.text
            self.wm.write("intermediate", f"{sub} -> {out.text}", priority=0.5)
            self.goals.complete(subgoal)

        # 6. CRITIQUE + revision loop (reflection loop, §10.3)
        verified = None
        for attempt in range(self.MAX_REVISIONS + 1):
            verdict = self.critic.verify(query, answer)
            cost += 1.5
            trace.append(f"critic[{attempt}]: {'OK' if verdict.ok else verdict.feedback}"
                         f"{' (grounded)' if verdict.grounded else ''}")
            if verdict.ok:
                verified = verdict.grounded or None
                break
            if verdict.corrected:            # verifier supplied the fix
                answer = verdict.corrected
                verified = True
                trace.append(f"adopted verifier correction: {answer}")
                break
            out = self.llm.generate(f"revise step by step: {query}", temperature=0.1)
            cost += 3.0
            answer = out.text

        # 7. DECIDE
        conf = self.confidence.estimate(query, answer, logprob=-0.3, verified=verified)
        cost += 0.5
        final = self.controller.post_deliberation(conf, stakes)

        # 8. RECORD — System-2 work is salient by definition (§7.4)
        outcome = "success" if final == Route.ANSWER else "deferred"
        self._record(query, f"{query} => {answer}", outcome, salience=0.7)
        if final == Route.ANSWER and verified:
            self.planner.cache_success(plan, answer)
        self.goals.complete(goal)

        if final == Route.DEFER:
            trace.append("post-deliberation confidence below stakes threshold -> defer")
            return AgentResult(None, "system2", conf, cost, trace)
        return AgentResult(answer, "system2", conf, cost, trace)

    # ------------------------------------------------------------------ #

    def _record(self, query: str, summary: str, outcome: str, salience: float) -> None:
        self.episodic.write(Episode(summary=summary, outcome=outcome,
                                    salience=salience,
                                    task_signature=task_signature(query)))

    def sleep(self) -> ConsolidationReport:
        """Offline consolidation pass (§7.5, §10.2 step 10)."""
        return self.consolidator.sleep()
