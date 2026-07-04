"""Solver — maps a task to (correct, cost, route) (EXPERIMENT_PLAN.md §3).

Bridges the abstract competence model to the REAL CANTOR memory stores so the
non-parametric arm is exercised by actual repo code:

  * exact-instance recall is a genuine lookup in the agent's SemanticMemory,
    populated by the real `agent.sleep()` consolidation from real EpisodicMemory
    writes — i.e. B2's exact-recall number is produced by the code under test.
  * the parametric skill and base competence are supplied by the (simulated or,
    in Phase 1, real) learner + the task's fixed `solvability`.

Correctness precedence per task:
  1. exact cache hit (memory)            -> correct,  cheap  (semantic-cache)
  2. mastered schema skill (parametric)  -> solvability < p_skill*health, 1 unit
  3. base competence                     -> solvability < base_success,  6 units (S2 try)
General-probe tasks bypass memory/skill and are gated by learner health (GCR).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..agent import CantorAgent
from ..memory import Episode, task_signature
from .learner import ParametricLearner
from .streams import Task

P_SKILL = 0.92        # reliability once a schema is mastered parametrically
P_GENERAL = 0.85      # base reliability on the general probe (pre-erosion)
COST = {"semantic-cache": 0.5, "skill": 1.0, "system1": 1.0, "system2": 6.0}


@dataclass
class Outcome:
    correct: bool
    cost: float
    route: str
    completion: str = ""      # model output text (real rollouts); "" for simulated


class Solver:
    def __init__(self, agent: CantorAgent, learner: ParametricLearner,
                 base_success: dict[str, float], use_memory: bool,
                 use_parametric: bool):
        self.agent = agent
        self.learner = learner
        self.base_success = base_success
        self.use_memory = use_memory
        self.use_parametric = use_parametric

    # -- real memory lookup -------------------------------------------------- #
    def _cached(self, prompt: str) -> bool:
        for f in self.agent.semantic.facts:
            if f.relation == "answer_is" and f.subject.lower() == prompt.lower():
                return True
        return False

    def _remember(self, task: Task) -> None:
        """Write a solved online task to REAL episodic memory (salience high so
        `agent.sleep()` promotes an exact-answer fact — matches agent.py §8)."""
        self.agent.episodic.write(Episode(
            summary=f"{task.prompt} => {task.answer}", outcome="success",
            salience=0.7, task_signature=task_signature(task.prompt)))

    # -- solve --------------------------------------------------------------- #
    def solve(self, task: Task, train: bool) -> Outcome:
        if task.is_general:
            ok = task.solvability < P_GENERAL * self.learner.general_health()
            return Outcome(ok, COST["system1"], "system1")

        if self.use_memory and self._cached(task.prompt):
            return Outcome(True, COST["semantic-cache"], "semantic-cache")

        if self.use_parametric and self.learner.has_mastered(task.schema):
            ok = task.solvability < P_SKILL * self.learner.general_health()
            out = Outcome(ok, COST["skill"], "skill")
        else:
            ok = task.solvability < self.base_success[task.schema]
            out = Outcome(ok, COST["system2"], "system2")

        if train and out.correct and self.use_memory:
            self._remember(task)
        return out

    def night(self) -> None:
        """Non-parametric consolidation (REAL agent.sleep); parametric handled
        by the learner in the evaluator."""
        if self.use_memory:
            self.agent.sleep()
