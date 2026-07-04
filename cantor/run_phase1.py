"""Phase 1 runner — the Table 2 locus experiment (EXPERIMENT_PLAN.md §8).

This is the go/no-go: memory-only (B2) should ACE exact-instance recall but LAG
novel-instance skill transfer (NIST); the parametric methods (B3/B7) should close
the NIST gap; B7 should do so WITHOUT eroding general capability (GCR ~ 0).

Modes:
  --smoke (default): runs the FULL Phase-1 control flow OFFLINE using
                     FakeTrainableLM (real rollouts, real memory, real LoRA
                     batch-build / eval-gate / rollback — only the numeric
                     training is a test double). Validates wiring; NOT a result.
  --real  MODEL    : uses HFTrainableLM(MODEL) — needs torch/transformers/peft +
                     GPU. Same code path; fills the table with measured numbers.

Usage:
  python run_phase1.py                         # offline smoke
  python run_phase1.py --real meta-llama/Llama-3.1-8B-Instruct
"""

from __future__ import annotations

import statistics as stats
import sys

from cantor.agent import CantorAgent
from cantor.continual import evaluate, make_stream
from cantor.continual.learner import NoOpLearner, RealLoRALearner
from cantor.continual.methods import MethodConfig
from cantor.continual.rollout_solver import RolloutSolver
from cantor.continual.streams import Stream
from cantor.llm import MockLLM

SEEDS = [0, 1, 2]
LOCUS_METHODS = ["B2-memory", "B3-naive-lora", "B7-hybrid"]
_CFG = {
    "B2-memory": MethodConfig("B2-memory", use_memory=True, use_parametric=False),
    "B3-naive-lora": MethodConfig("B3-naive-lora", use_memory=False, use_parametric=True,
                                  generative_replay=False, eval_gate=False),
    "B7-hybrid": MethodConfig("B7-hybrid", use_memory=True, use_parametric=True,
                              generative_replay=True, eval_gate=True),
}


def _oracle(*streams: Stream) -> dict[str, str]:
    """prompt -> ground-truth answer, over all tasks in the given streams."""
    o: dict[str, str] = {}
    for st in streams:
        for f in st.families:
            for t in f.train + f.retention_novel + f.retention_repeat:
                o[t.prompt] = t.answer
        for t in st.general_probe:
            o[t.prompt] = t.answer
    return o


def _internal_probes(probe_stream: Stream):
    """Self-generated eval sets for the learner's eval-gate — DISJOINT from the
    benchmark stream (integrity constraint, EXPERIMENT_PLAN.md §3, §9)."""
    schema_probes = {f.schema: [t.prompt for t in f.retention_novel[:8]]
                     for f in probe_stream.families}
    general_probes = [t.prompt for t in probe_stream.general_probe[:60]]
    anchors = [(t.prompt, t.answer) for t in probe_stream.general_probe[:16]]
    return schema_probes, general_probes, anchors


def make_lm(real_model: str | None, oracle, base_success):
    if real_model:                                    # pragma: no cover - Phase 1
        from cantor.continual.trainable import HFTrainableLM
        return HFTrainableLM(real_model)
    from cantor.continual.trainable import FakeTrainableLM
    return FakeTrainableLM(oracle, base_success)


def run_method(method: str, seed: int, real_model: str | None):
    stream = make_stream(seed=seed, n_families=6)
    probe_stream = make_stream(seed=seed + 1000, n_families=6)   # disjoint instances
    oracle = _oracle(stream, probe_stream)
    cfg = _CFG[method]

    lm = make_lm(real_model, oracle, stream.base_success)
    agent = CantorAgent(MockLLM())                    # REAL memory stores

    if cfg.use_parametric:
        sp, gp, anchors = _internal_probes(probe_stream)
        learner = RealLoRALearner(lm, cfg.generative_replay, cfg.eval_gate,
                                  internal_schema_probes=sp, internal_general_probes=gp,
                                  anchor_set=anchors, answer_of=oracle)
    else:
        learner = NoOpLearner()

    solver = RolloutSolver(lm, agent, cfg.use_memory, cfg.use_parametric)
    return evaluate(cfg, stream, seed, solver=solver, learner=learner)


def main() -> None:
    real_model = None
    if "--real" in sys.argv:
        real_model = sys.argv[sys.argv.index("--real") + 1]
        print(f"[mode] REAL backend: {real_model}")
    else:
        print("[mode] SMOKE (offline FakeTrainableLM) — validates wiring, not a result")

    agg = {}
    for m in LOCUS_METHODS:
        runs = [run_method(m, s, real_model).metrics for s in SEEDS]
        agg[m] = {
            "exact": (stats.mean(r.exact_recall for r in runs)),
            "nist": (stats.mean(r.nist for r in runs)),
            "gcr": (stats.mean(r.gcr for r in runs)),
            "acc_cost": (stats.mean(r.acc_per_cost for r in runs)),
        }

    print("\nTABLE 2 (locus) — real rollouts + real LoRA control flow")
    hdr = f"{'method':<15}{'exact_recall':>14}{'NIST(novel)':>13}{'gap':>9}{'GCR':>9}{'acc/cost':>10}"
    print(hdr); print("-" * len(hdr))
    for m in LOCUS_METHODS:
        a = agg[m]
        print(f"{m:<15}{a['exact']:>14.3f}{a['nist']:>13.3f}"
              f"{a['nist'] - a['exact']:>9.3f}{a['gcr']:>9.3f}{a['acc_cost']:>10.4f}")

    print("\nGO/NO-GO CHECKS (Table 2 thesis — EXPERIMENT_PLAN.md §7):")
    b2, b3, b7 = agg["B2-memory"], agg["B3-naive-lora"], agg["B7-hybrid"]
    checks = [
        ("H2: memory-only aces exact recall but lags novel (gap < -0.10)",
         (b2["exact"] - b2["nist"]) > 0.10),
        ("H2: weights close the novel gap (B7 NIST > B2 NIST + 0.10)",
         b7["nist"] > b2["nist"] + 0.10),
        ("H3: naive LoRA erodes general capability (B3 GCR < -0.05)", b3["gcr"] < -0.05),
        # Relative check: H3 claims replay+gating SUBSTANTIALLY reduce forgetting
        # vs naive -> compare B7 to B3, not to a quantization-fragile absolute line.
        ("H3: eval-gate+replay cut forgetting >=2x vs naive "
         "(B7 GCR - B3 GCR > 0.10 and |B7| < 0.5|B3|)",
         (b7["gcr"] - b3["gcr"]) > 0.10 and abs(b7["gcr"]) < 0.5 * abs(b3["gcr"])),
    ]
    go = True
    for label, cond in checks:
        go &= cond
        print(f"  [{'ok' if cond else 'XX'}] {label}")
    print(f"\nLOCUS GO/NO-GO: {'GO' if go else 'NO-GO'}  "
          f"({'thesis pattern reproduced — proceed to Phase 2' if go else 'thesis unsupported — re-scope'})")


if __name__ == "__main__":
    main()
