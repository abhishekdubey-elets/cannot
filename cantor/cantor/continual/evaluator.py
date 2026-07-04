"""Continual-learning evaluator — the Phase 0 loop (EXPERIMENT_PLAN.md §3, §8).

For a given method and stream:

    baseline eval (night 0)  -> R[0, *], G[0]
    for each family i = 1..N:
        online-solve family i's train tasks     (plasticity; writes to memory)
        NIGHT: agent.sleep()  [non-parametric]  +  learner.consolidate() [parametric]
        eval ALL families' novel retention      -> R[i, *]
        eval general probe                       -> G[i]
    end: eval exact-repeat retention at night N; compute metrics

Evaluation solves are read-only (train=False) so measurement never mutates the
stores. The full R matrix (every family evaluated every night) is what enables
honest BWT/FWT, not just end-state accuracy.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from ..agent import CantorAgent
from ..llm import MockLLM
from .metrics import Metrics, compute
from .methods import MethodConfig
from .solver import Solver
from .streams import Stream


@dataclass
class ContinualResult:
    method: str
    seed: int
    metrics: Metrics
    R: dict
    g_curve: list[float]


def _acc(solver: Solver, tasks) -> float:
    if not tasks:
        return 0.0
    return sum(solver.solve(t, train=False).correct for t in tasks) / len(tasks)


def evaluate(config: MethodConfig, stream: Stream, seed: int,
             solver=None, learner=None, consolidate_every: int = 1) -> ContinualResult:
    """Run one method over the stream.

    Phase 0 (default): builds the simulated `Solver` + the config's learner.
    Phase 1: inject a `RolloutSolver` + `RealLoRALearner` and the same loop runs
    unchanged — this is the whole point of the shared interface.
    """
    if solver is None:
        agent = CantorAgent(MockLLM())          # REAL memory stores for the mem arm
        learner = learner or config.make_learner()
        solver = Solver(agent, learner, stream.base_success,
                        config.use_memory, config.use_parametric)
    elif learner is None:
        learner = config.make_learner()
    fams = stream.families
    N = len(fams)

    R: dict = {}
    g_curve: list[float] = []
    online_cost = 0.0
    routes: Counter = Counter()

    def eval_night(i: int) -> None:
        for j, fam in enumerate(fams, start=1):
            R[(i, j)] = _acc(solver, fam.retention_novel)
        g_curve.append(_acc(solver, stream.general_probe))

    # Night 0: pre-training baseline (no memory, no skills yet).
    eval_night(0)

    for i, fam in enumerate(fams, start=1):
        # --- online phase: solve this family's stream (plasticity + memory writes)
        for t in fam.train:
            out = solver.solve(t, train=True)
            online_cost += out.cost
            routes[out.route] += 1
            if out.correct:
                learner.observe(t.prompt, out.completion, t.schema)

        # --- NIGHT: consolidate both stores (every `consolidate_every` families; A8)
        if i % consolidate_every == 0 or i == N:
            solver.night()      # REAL non-parametric: episodic -> semantic/procedural
            learner.consolidate()   # parametric weight update (no-op for NoOpLearner)

        # --- eval all families + general probe after this night
        eval_night(i)

    exact_recall = sum(_acc(solver, f.retention_repeat) for f in fams) / N
    metrics = compute(R, N, g_curve, exact_recall, online_cost, dict(routes))
    return ContinualResult(config.name, seed, metrics, R, g_curve)
