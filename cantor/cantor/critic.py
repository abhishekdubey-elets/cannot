"""Reflection / Critic — report §1.13, §5.5, Part 9 (RLVR).

The report's key claim about reflection: self-critique is only as good as the
model's ability to detect its own errors, so *external verifiers dominate*
whenever they exist. The critic therefore tries, in order:

  1. a grounded verifier (here: the ToolWorldModel's arithmetic engine),
  2. LLM self-critique in "deliberate" mode (the weak fallback).

Returns a Verdict the agent's revision loop consumes (§10.3 reflection loop).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .llm import LLM
from .world_model import WorldModel


@dataclass
class Verdict:
    ok: bool
    grounded: bool          # was an external verifier used?
    feedback: str
    corrected: str | None = None  # verifier may supply the right answer


class Critic:
    def __init__(self, llm: LLM, world_model: WorldModel):
        self.llm = llm
        self.world_model = world_model

    def verify(self, query: str, answer: str) -> Verdict:
        # Route 1: grounded verification via the world model / tool (§8.3).
        expr = re.search(r"(\d+\s*[*+x]\s*\d+)", query)
        if expr:
            pred = self.world_model.predict(state=query, action=expr.group(1))
            if pred.grounded:
                if answer.strip() == pred.next_state:
                    return Verdict(True, True, "verifier: match")
                return Verdict(False, True,
                               f"verifier: expected {pred.next_state}, got {answer}",
                               corrected=pred.next_state)

        # Route 2: ungrounded self-critique — flag it as such (§1.13 weakness).
        out = self.llm.generate(
            f"Question: {query}\nCandidate answer: {answer}\n"
            f"Check step by step for errors. Reply OK or the problem:",
            temperature=0.1,
        )
        looks_ok = "uncertain" not in out.text.lower()
        return Verdict(looks_ok, False, f"self-critique: {out.text[:80]}")
