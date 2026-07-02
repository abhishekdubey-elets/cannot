"""Goal Manager — report §1.15, Part 4, §10.4.

Persistent, protected goal/subgoal stack. The antidote to plan-drift: the top
goal is re-injected into working memory on every controller cycle, so it cannot
be diluted out of context the way a goal-in-prompt is (§2.3 gap #6).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Goal:
    description: str
    parent: "Goal | None" = None
    done: bool = False
    children: list["Goal"] = field(default_factory=list)


class GoalManager:
    def __init__(self):
        self._stack: list[Goal] = []

    def push(self, description: str, parent: Goal | None = None) -> Goal:
        goal = Goal(description, parent)
        if parent:
            parent.children.append(goal)
        self._stack.append(goal)
        return goal

    def complete(self, goal: Goal) -> None:
        goal.done = True
        self._stack = [g for g in self._stack if not g.done]

    @property
    def active(self) -> Goal | None:
        return self._stack[-1] if self._stack else None

    def reinject(self) -> str:
        """The line the controller writes into working memory each cycle."""
        chain: list[str] = []
        goal = self.active
        while goal:
            chain.append(goal.description)
            goal = goal.parent
        return " <= ".join(chain) if chain else "(no active goal)"
