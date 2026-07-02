"""Planner — report §1.12, §1.18, Part 4.

Hierarchical decomposition: goal -> subgoals, executed by the slow engine with
world-model support. Implements the model-based/model-free arbitration the
report takes from Daw (2005): before planning, check ProceduralMemory for a
cached skill (the 'habit'); plan from scratch only when no reliable habit
exists. Successful plans are handed to consolidation for caching (§6.3).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .llm import LLM
from .memory import ProceduralMemory, Skill, task_signature


@dataclass
class Plan:
    goal: str
    subgoals: list[str]
    from_cache: bool = False


class Planner:
    def __init__(self, llm: LLM, procedural: ProceduralMemory):
        self.llm = llm
        self.procedural = procedural

    def plan(self, goal: str) -> Plan:
        # Model-free path: a cached skill with enough successes is trusted.
        skill = self.procedural.lookup(task_signature(goal))
        if skill and skill.successes >= 1:
            return Plan(goal, list(skill.plan), from_cache=True)

        # Model-based path: decompose. The MockLLM can't decompose, so we use
        # a rule-based decomposer for the demo's task family; with a real
        # backend this is one structured-output call.
        expr = re.search(r"(\d+)\s*([*+x])\s*(\d+)", goal)
        if expr:
            a, op, b = expr.group(1), expr.group(2), expr.group(3)
            return Plan(goal, [
                f"compute {a} {op} {b} step by step",
                "verify the result with the world model",
            ])

        out = self.llm.generate(f"Decompose into 2-4 subgoals, one per line:\n{goal}", temperature=0.2)
        subgoals = [ln.strip("- ").strip() for ln in out.text.splitlines() if ln.strip()]
        return Plan(goal, subgoals or [goal])

    def cache_success(self, plan: Plan, answer: str) -> None:
        """Distill a verified success into a procedural skill (§7.5 step 3)."""
        self.procedural.store(Skill(task_signature(plan.goal), plan.subgoals, answer))
