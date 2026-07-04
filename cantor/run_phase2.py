"""Phase 2 runner — realistic-domain replication + forward transfer (§4, §8).

Phase 1 established the locus finding on the CONTROLLED independent-schema stream
(FWT is necessarily 0 there). Phase 2 checks the finding REPLICATES on a domain
with shared substructure (S-TOOL: tool tasks with overlapping primitives), where
forward transfer (FWT > 0) becomes a real, measurable signal — the thing a
realistic benchmark buys you over the controlled one.

Gate (EXPERIMENT_PLAN.md §8): locus replicates on >= 1 realistic domain, AND
parametric methods exhibit meaningful FWT while memory-only does not.

Modes: offline uses SyntheticToolAdapter (ToolFakeLM). Real AppWorld/tau-bench
adapters plug in behind the same BenchmarkAdapter interface (Phase 2 real).
"""

from __future__ import annotations

import statistics as stats

from cantor.agent import CantorAgent
from cantor.continual import evaluate
from cantor.continual.benchmarks import SyntheticToolAdapter
from cantor.continual.learner import NoOpLearner, RealLoRALearner
from cantor.continual.methods import MethodConfig
from cantor.continual.rollout_solver import RolloutSolver
from cantor.continual.streams import Stream
from cantor.llm import MockLLM

SEEDS = [0, 1, 2]
METHODS = ["B2-memory", "B3-naive-lora", "B7-hybrid"]
_CFG = {
    "B2-memory": MethodConfig("B2-memory", True, False),
    "B3-naive-lora": MethodConfig("B3-naive-lora", False, True,
                                  generative_replay=False, eval_gate=False),
    "B7-hybrid": MethodConfig("B7-hybrid", True, True,
                              generative_replay=True, eval_gate=True),
}


def _oracle(*streams: Stream) -> dict[str, str]:
    o: dict[str, str] = {}
    for st in streams:
        for f in st.families:
            for t in f.train + f.retention_novel + f.retention_repeat:
                o[t.prompt] = t.answer
        for t in st.general_probe:
            o[t.prompt] = t.answer
    return o


def _internal_probes(ps: Stream):
    sp = {f.schema: [t.prompt for t in f.retention_novel[:8]] for f in ps.families}
    gp = [t.prompt for t in ps.general_probe[:60]]
    anchors = [(t.prompt, t.answer) for t in ps.general_probe[:16]]
    return sp, gp, anchors


def run_method(adapter, method: str, seed: int):
    stream = adapter.to_stream(seed)
    probe = adapter.to_stream(seed + 1000)          # disjoint instances
    oracle = _oracle(stream, probe)
    cfg = _CFG[method]

    lm = adapter.make_lm(oracle, stream.base_success)
    agent = CantorAgent(MockLLM())

    if cfg.use_parametric:
        sp, gp, anchors = _internal_probes(probe)
        learner = RealLoRALearner(lm, cfg.generative_replay, cfg.eval_gate,
                                  internal_schema_probes=sp, internal_general_probes=gp,
                                  anchor_set=anchors, answer_of=oracle)
    else:
        learner = NoOpLearner()

    solver = RolloutSolver(lm, agent, cfg.use_memory, cfg.use_parametric)
    return evaluate(cfg, stream, seed, solver=solver, learner=learner)


def main() -> None:
    adapter = SyntheticToolAdapter()
    print(f"[mode] SMOKE (offline {adapter.name}) — validates replication + FWT, not a result")

    agg = {}
    for m in METHODS:
        runs = [run_method(adapter, m, s).metrics for s in SEEDS]
        agg[m] = {k: stats.mean(getattr(r, k) for r in runs)
                  for k in ["acc", "bwt", "fwt", "gcr", "exact_recall", "acc_per_cost"]}

    print(f"\nTABLE 3 ({adapter.name}) — locus + forward transfer on shared-substructure domain")
    hdr = (f"{'method':<15}{'ACC':>8}{'exact':>8}{'NIST':>8}{'gap':>8}"
           f"{'FWT':>8}{'BWT':>8}{'GCR':>8}{'acc/cost':>10}")
    print(hdr); print("-" * len(hdr))
    for m in METHODS:
        a = agg[m]
        print(f"{m:<15}{a['acc']:>8.3f}{a['exact_recall']:>8.3f}{a['acc']:>8.3f}"
              f"{a['acc'] - a['exact_recall']:>8.3f}{a['fwt']:>8.3f}{a['bwt']:>8.3f}"
              f"{a['gcr']:>8.3f}{a['acc_per_cost']:>10.4f}")

    print("\nGATE CHECKS (Phase 2 — EXPERIMENT_PLAN.md §8):")
    b2, b3, b7 = agg["B2-memory"], agg["B3-naive-lora"], agg["B7-hybrid"]
    checks = [
        ("locus replicates: memory aces exact recall, lags novel (B2 gap < -0.10)",
         (b2["exact_recall"] - b2["acc"]) > 0.10),
        ("weights close the gap here too (B7 NIST > B2 NIST + 0.10)",
         b7["acc"] > b2["acc"] + 0.10),
        ("FWT EMERGES for parametric methods (B7 FWT > 0.05)", b7["fwt"] > 0.05),
        ("memory-only shows ~no forward transfer (|B2 FWT| < 0.03)",
         abs(b2["fwt"]) < 0.03),
        ("hybrid forgets far less than naive (B7 GCR - B3 GCR > 0.10)",
         (b7["gcr"] - b3["gcr"]) > 0.10),
    ]
    ok = True
    for label, cond in checks:
        ok &= cond
        print(f"  [{'ok' if cond else 'XX'}] {label}")
    print(f"\nPHASE 2 GATE: {'PASS' if ok else 'FAIL'}  "
          f"({'locus replicates + FWT confirmed on realistic-style domain' if ok else 'investigate'})")


if __name__ == "__main__":
    main()
