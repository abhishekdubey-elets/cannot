"""RolloutSolver — real model-in-the-loop solving (Phase 1).

Solves each task by generating with a TrainableLM and parsing the answer, rather
than consulting a competence model. The non-parametric arm still runs the REAL
CANTOR memory: an exact cache hit short-circuits generation (the cheap route),
and correct online solutions are written to episodic memory for `agent.sleep()`
to consolidate. The parametric arm's effect enters through the shared `lm` that
the RealLoRALearner fine-tunes between nights.

This is the object Phase 1 injects into `evaluate(...)`; the loop is identical to
Phase 0, so the two phases share one evaluator.
"""

from __future__ import annotations

import re

from ..agent import CantorAgent
from ..memory import Episode, task_signature
from .solver import COST, Outcome
from .streams import Task
from .trainable import TrainableLM


def parse_answer(text: str) -> str | None:
    nums = re.findall(r"-?\d+", text)
    return nums[-1] if nums else None


class RolloutSolver:
    def __init__(self, lm: TrainableLM, agent: CantorAgent,
                 use_memory: bool, use_parametric: bool, salience_gate: bool = True,
                 replay_raw_completion: bool = False):
        self.lm = lm
        self.agent = agent
        self.use_memory = use_memory
        self.use_parametric = use_parametric   # LoRA effect enters via `lm`
        # Replay target format. Online solves run DELIBERATE (verbose reasoning),
        # but train_lora renders the TERSE prompt and the fast path is evaluated
        # terse. Storing the raw verbose generation therefore teaches
        # "terse prompt -> rambling answer" and degrades the model. Store the
        # PARSED answer instead: self-distil what deliberation concluded, not how
        # it rambled. `replay_raw_completion=True` reproduces the old (buggy)
        # behaviour for A/B comparison.
        self.replay_raw_completion = replay_raw_completion
        # A7 knob. NOTE (honest): offline this changes what the episodic store
        # keeps, but correctness here flows through the EXACT semantic cache, not
        # retrieval — so it does not move the offline metrics. Its effect appears
        # on the retrieval-augmented solve path (Phase-3 real). Wired for parity.
        self.salience_gate = salience_gate
        if not salience_gate:
            agent.episodic.SALIENCE_GATE = 0.0

    def _cached(self, prompt: str) -> bool:
        return any(f.relation == "answer_is" and f.subject.lower() == prompt.lower()
                   for f in self.agent.semantic.facts)

    def solve(self, task: Task, train: bool) -> Outcome:
        if self.use_memory and self._cached(task.prompt):
            return Outcome(True, COST["semantic-cache"], "semantic-cache", task.answer)

        # Online solving is effortful (verified System 2 -> high success, higher
        # cost); evaluation uses the fast path (cheap, flaky unless skill learned).
        text = self.lm.generate(task.prompt, deliberate=train)
        pred = parse_answer(text)
        correct = pred is not None and pred == str(task.answer)
        cost = COST["system2"] if train else COST["system1"]
        route = "deliberate" if train else "generate"
        replay_target = text if self.replay_raw_completion else (pred or "")
        out = Outcome(correct, cost, route, replay_target)

        if train and correct and self.use_memory:
            self.agent.episodic.write(Episode(
                summary=f"{task.prompt} => {task.answer}", outcome="success",
                salience=0.7, task_signature=task_signature(task.prompt)))
        return out

    def night(self) -> None:
        if self.use_memory:
            self.agent.sleep()
