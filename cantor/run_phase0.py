"""Phase 0 harness runner (EXPERIMENT_PLAN.md §8).

Runs the four core methods across seeds on the S-CTRL controlled stream, prints
the Table 1 (continual-learning) and Table 2 (locus) metric shapes, and checks
the harness GATE: B0 and B2 reproduce stably across seeds.

Offline, correctness is driven by the real CANTOR memory (non-parametric arm) +
a simulated learner (parametric arm) — see cantor/continual/README notes. This
validates the evaluation pipeline and metric computations end to end; it is NOT
a scientific result. Phase 1 swaps in a real backend + LoRA behind the same
interfaces and the tables fill with measured numbers.
"""

from __future__ import annotations

import statistics as stats

from cantor.continual import evaluate, make_stream
from cantor.continual.methods import METHODS, PHASE0_METHODS

SEEDS = [0, 1, 2, 3, 4]
GATE_STD = 0.05          # max tolerated across-seed std of ACC for B0/B2


def agg(method: str):
    runs = [evaluate(METHODS[method], make_stream(seed=s, n_families=6), s) for s in SEEDS]
    m = [r.metrics for r in runs]

    def ms(attr):
        vals = [getattr(x, attr) for x in m]
        return stats.mean(vals), (stats.pstdev(vals) if len(vals) > 1 else 0.0)
    return {a: ms(a) for a in
            ["acc", "bwt", "fwt", "gcr", "exact_recall", "online_cost", "acc_per_cost"]}


def fmt(mean_std):
    return f"{mean_std[0]:.3f}±{mean_std[1]:.3f}"


def main() -> None:
    results = {mth: agg(mth) for mth in PHASE0_METHODS}

    print("\nTABLE 1 - continual-learning metrics on S-CTRL (mean±std over "
          f"{len(SEEDS)} seeds)")
    print("  ACC/BWT/FWT/GCR: higher better; BWT,GCR near 0 = no forgetting\n")
    hdr = f"{'method':<15}{'ACC':>14}{'BWT':>14}{'FWT':>14}{'GCR':>14}{'acc/cost':>12}"
    print(hdr); print("-" * len(hdr))
    for mth in PHASE0_METHODS:
        r = results[mth]
        print(f"{mth:<15}{fmt(r['acc']):>14}{fmt(r['bwt']):>14}{fmt(r['fwt']):>14}"
              f"{fmt(r['gcr']):>14}{r['acc_per_cost'][0]:>12.4f}")

    print("\nTABLE 2 - locus: knowledge (exact recall) vs skill (novel transfer)")
    print("  memory-only should ACE exact recall but LAG novel transfer (H2)\n")
    hdr2 = f"{'method':<15}{'exact_recall':>16}{'NIST(novel)':>14}{'gap':>10}"
    print(hdr2); print("-" * len(hdr2))
    for mth in PHASE0_METHODS:
        r = results[mth]
        er, nist = r["exact_recall"][0], r["acc"][0]
        print(f"{mth:<15}{er:>16.3f}{nist:>14.3f}{nist - er:>10.3f}")

    print("\nGATE - stability of B0/B2 across seeds (std of ACC < "
          f"{GATE_STD}):")
    ok = True
    for mth in ["B0-frozen", "B2-memory"]:
        std = results[mth]["acc"][1]
        passed = std < GATE_STD
        ok &= passed
        print(f"  {mth:<12} ACC std = {std:.4f}  -> {'PASS' if passed else 'FAIL'}")
    print(f"\nPHASE 0 GATE: {'PASS' if ok else 'FAIL'}  "
          f"({'harness reproduces B0/B2 stably' if ok else 'investigate variance'})")

    # Sanity assertions on the qualitative pattern the harness must exhibit.
    print("\nHARNESS SANITY CHECKS (qualitative patterns the pipeline must show):")
    b2, b3, b7 = results["B2-memory"], results["B3-naive-lora"], results["B7-hybrid"]
    checks = [
        ("B2 exact_recall > B2 novel (memory helps recall, not transfer)",
         b2["exact_recall"][0] > b2["acc"][0] + 0.1),
        ("B7 novel-transfer > B2 novel-transfer (weights buy skill)",
         b7["acc"][0] > b2["acc"][0] + 0.1),
        ("B3 GCR clearly negative (naive LoRA forgets)", b3["gcr"][0] < -0.05),
        ("B7 GCR near zero (replay+gating protect capability)", b7["gcr"][0] > -0.05),
        ("B7 acc/cost > B3 acc/cost (memory routing is cheaper)",
         b7["acc_per_cost"][0] > b3["acc_per_cost"][0]),
    ]
    for label, cond in checks:
        print(f"  [{'ok' if cond else 'XX'}] {label}")


if __name__ == "__main__":
    main()
