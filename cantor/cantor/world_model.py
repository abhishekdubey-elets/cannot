"""World Model + Simulation module — report Part 8, §1.7–1.10.

Interface-first design: the contract is `predict(state, action) -> Prediction`,
which is what the Planner and Critic consume. Two implementations:

  ToolWorldModel  grounded route available *today* (report §8.3 option 2):
                  delegate to an external simulator/executor — here, a safe
                  arithmetic evaluator standing in for "call a real engine".
  TextWorldModel  LLM-imagined rollout — the weak fallback the report critiques
                  (entangled with language, unverified), kept for coverage.

The research upgrade is a JEPA/Dreamer-style latent model behind the same
interface (§8.3 option 1); nothing upstream would change.
"""

from __future__ import annotations

import ast
import operator
import re
from dataclasses import dataclass

from .llm import LLM


@dataclass
class Prediction:
    next_state: str
    grounded: bool       # True if produced by a real simulator/verifier
    confidence: float


class WorldModel:
    def predict(self, state: str, action: str) -> Prediction:
        raise NotImplementedError


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.FloorDiv: operator.floordiv}


def _safe_eval(expr: str) -> int | None:
    """Grounded arithmetic 'simulator' — whitelisted AST walk, no eval()."""
    try:
        def walk(node):
            if isinstance(node, ast.Expression):
                return walk(node.body)
            if isinstance(node, ast.Constant) and isinstance(node.value, int):
                return node.value
            if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
                return _OPS[type(node.op)](walk(node.left), walk(node.right))
            raise ValueError(node)
        return walk(ast.parse(expr, mode="eval"))
    except Exception:
        return None


class ToolWorldModel(WorldModel):
    """Ground predictions in an external engine wherever one exists."""

    def predict(self, state: str, action: str) -> Prediction:
        expr = re.search(r"[\d\s+\-*/()]+", action.replace("x", "*"))
        if expr:
            value = _safe_eval(expr.group().strip())
            if value is not None:
                return Prediction(str(value), grounded=True, confidence=0.99)
        return Prediction(f"(no grounded model for: {action})", grounded=False, confidence=0.2)


class TextWorldModel(WorldModel):
    """LLM 'imagination' fallback — fluent, unverified (§8.2)."""

    def __init__(self, llm: LLM):
        self.llm = llm

    def predict(self, state: str, action: str) -> Prediction:
        out = self.llm.generate(f"State: {state}\nAction: {action}\nPredicted outcome, step by step:")
        return Prediction(out.text, grounded=False, confidence=0.5)


class CompositeWorldModel(WorldModel):
    """Try grounded simulators first; fall back to imagination (§8.3)."""

    def __init__(self, llm: LLM):
        self.tool = ToolWorldModel()
        self.text = TextWorldModel(llm)

    def predict(self, state: str, action: str) -> Prediction:
        p = self.tool.predict(state, action)
        return p if p.grounded else self.text.predict(state, action)
