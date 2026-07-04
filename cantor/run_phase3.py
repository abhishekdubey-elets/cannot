"""Phase 3 runner — ablation study + capacity/frequency curves (§7, §8).

Table 4 removes one mechanism at a time from the Full hybrid. The gate requires
each MODELED mechanism's ablation to move its decisive metric:

    remove parametric  -> NIST collapses (skill transfer needs weights)

Generative replay and eval-gating are PARTIALLY REDUNDANT protections of general
capability, so a single ablation from Full understates them. Their roles are
shown by the replay x gate 2x2 (decisive metric: GCR / NIST):

                 gate ON                         gate OFF
  replay ON   learn, no forget (Full)         same (gate redundant here)
  replay OFF  gate REFUSES unsafe updates:     no protection: general
              general safe, NIST collapses     CRATERS (forgetting)

Honest offline no-ops (reported, not gated):
  - memory:        no ACCURACY change when a perfect skill covers instances
                   (memory's locus role is the Phase 1/2 headline).
  - prioritization: uniform replay still masters everything under abundant data;
                   its value needs buffer pressure (real runs), absent here.
  - salience gate:  acts on RETRIEVAL, which the exact-cache path doesn't exercise.

Curves: A8 sweeps consolidation frequency; A9 sweeps replay/LoRA capacity coupled
to drift. Offline via SyntheticToolAdapter; real AppWorld/tau-bench + 32B scale
check plug in behind the same BenchmarkAdapter (stubbed).
"""

from __future__ import annotations

import statistics as stats

from cantor.continual.benchmarks import SyntheticToolAdapter
from cantor.continual.harness import AblationConfig, build_run

SEEDS = [0, 1, 2]
ADAPTER = SyntheticToolAdapter()


def agg(cfg: AblationConfig) -> dict:
    ms = [build_run(ADAPTER, cfg, s).metrics for s in SEEDS]
    keys = ["acc", "bwt", "fwt", "gcr", "exact_recall", "acc_per_cost"]
    return {k: stats.mean(getattr(m, k) for m in ms) for k in keys}


def main() -> None:
    print(f"[mode] SMOKE (offline {ADAPTER.name}) — validates the ablation "
          "pipeline + gate, not measured magnitudes\n")

    table4 = [
        AblationConfig("Full-hybrid"),
        AblationConfig("-parametric (A1)", use_parametric=False),
        AblationConfig("-memory (A2)*", use_memory=False),
        AblationConfig("-prioritization (A5)*", prioritized=False, batch_budget=4),
        AblationConfig("-salience-gate (A7)*", salience_gate=False),
    ]
    res = {c.name: agg(c) for c in table4}
    full = res["Full-hybrid"]

    print("TABLE 4 — single-mechanism ablations   (* = honest offline no-op, see header)")
    hdr = f"{'variant':<24}{'ACC/NIST':>10}{'exact':>8}{'GCR':>8}{'FWT':>8}{'acc/cost':>10}"
    print(hdr); print("-" * len(hdr))
    for c in table4:
        r = res[c.name]
        print(f"{c.name:<24}{r['acc']:>10.3f}{r['exact_recall']:>8.3f}{r['gcr']:>8.3f}"
              f"{r['fwt']:>8.3f}{r['acc_per_cost']:>10.4f}")

    # ---- replay x gate interaction (A4 x A6) -------------------------------
    print("\nREPLAY x EVAL-GATE interaction (decisive: NIST plasticity, GCR forgetting)")
    cell = {}
    for rep in (True, False):
        for gate in (True, False):
            r = agg(AblationConfig(f"r{rep}g{gate}", generative_replay=rep, eval_gate=gate))
            cell[(rep, gate)] = r
    print(f"{'':<14}{'gate ON':>22}{'gate OFF':>22}")
    print(f"{'':<14}{'NIST':>11}{'GCR':>11}{'NIST':>11}{'GCR':>11}")
    for rep in (True, False):
        on, off = cell[(rep, True)], cell[(rep, False)]
        print(f"{'replay ' + ('ON' if rep else 'OFF'):<14}"
              f"{on['acc']:>11.3f}{on['gcr']:>11.3f}{off['acc']:>11.3f}{off['gcr']:>11.3f}")

    # ---- A8: consolidation-frequency curve ---------------------------------
    print("\nA8 — consolidation frequency (families/night): plasticity vs cost")
    print(f"{'every K':>9}{'ACC':>9}{'GCR':>9}{'acc/cost':>11}")
    for K in [1, 2, 3, 6]:
        r = agg(AblationConfig(f"K={K}", consolidate_every=K))
        print(f"{K:>9}{r['acc']:>9.3f}{r['gcr']:>9.3f}{r['acc_per_cost']:>11.4f}")

    # ---- A9: replay/LoRA capacity curve (budget coupled to drift) ----------
    print("\nA9 — replay/LoRA capacity (budget<->drift): forgetting cost of capacity")
    print(f"{'budget':>8}{'drift':>8}{'ACC':>9}{'GCR':>9}")
    for budget, drift in [(2, 0.03), (4, 0.05), (6, 0.07), (10, 0.11)]:
        r = agg(AblationConfig(f"b={budget}", batch_budget=budget, drift=drift))
        print(f"{budget:>8}{drift:>8.2f}{r['acc']:>9.3f}{r['gcr']:>9.3f}")

    # ---- gate --------------------------------------------------------------
    print("\nGATE — each modeled mechanism moves its decisive metric:")
    ron_gon, roff_gon, roff_goff = cell[(True, True)], cell[(False, True)], cell[(False, False)]
    checks = [
        ("A1 parametric -> NIST collapses (Full - noParam > 0.30)",
         full["acc"] - res["-parametric (A1)"]["acc"] > 0.30),
        ("generative replay prevents forgetting "
         "(replay-off/gate-off GCR < Full GCR - 0.10)",
         roff_goff["gcr"] < full["gcr"] - 0.10),
        ("eval-gate protects general when replay absent "
         "(rGCR: gate-on - gate-off > 0.10)",
         roff_gon["gcr"] - roff_goff["gcr"] > 0.10),
        ("...at a plasticity cost (gate-on-no-replay NIST < Full - 0.10)",
         roff_gon["acc"] < full["acc"] - 0.10),
    ]
    ok = True
    for label, cond in checks:
        ok &= cond
        print(f"  [{'ok' if cond else 'XX'}] {label}")
    print("  [--] memory / prioritization / salience-gate: honest no-ops offline")
    print(f"\nPHASE 3 GATE: {'PASS' if ok else 'FAIL'}  "
          f"({'each modeled mechanism carries its decisive metric' if ok else 'investigate'})")

    print("\nSCALE CHECK (Phase-3 real): rerun with a 32B model via an HFTrainableLM "
          "adapter to confirm the ablation ordering is scale-robust (A100-class GPUs).")


if __name__ == "__main__":
    main()
