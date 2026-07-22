"""Detectability + power analysis for the DISPATCH hypothesis (CPU, no GPU).

Central claim: adaptive dispatch (right mechanism per query) beats the best single
fixed mechanism. The MAXIMUM possible advantage is (oracle - best_fixed), which is set
entirely by workload heterogeneity. This script computes that ceiling from an accuracy
profile, the accuracy a realistic controller achieves as a function of its routing
accuracy, and the sample size to detect controller > best_fixed.

The accuracy matrix below is ASSUMED (plausible for Qwen2.5-3B) and clearly marked;
run_calibrate.py measures the REAL matrix and this analysis is re-run on it before any
controller is trained. This is the 'is the effect even detectable' step we skipped
three times in the prior studies.
"""
from __future__ import annotations

import math
import statistics as stats

from dispatch.workload import TYPES

MECHS = ["fast", "cot", "tool", "retrieve"]

# ASSUMED accuracy A[qtype][mechanism] on Qwen2.5-3B. To be REPLACED by measurement.
A_ASSUMED = {
    "factual":    {"fast": 0.75, "cot": 0.75, "tool": 0.10, "retrieve": 0.30},
    "arithmetic": {"fast": 0.10, "cot": 0.45, "tool": 0.97, "retrieve": 0.05},
    "word":       {"fast": 0.15, "cot": 0.60, "tool": 0.30, "retrieve": 0.05},
    "recall":     {"fast": 0.20, "cot": 0.25, "tool": 0.05, "retrieve": 0.95},
}


def analyse(A, label):
    types = list(A)
    # oracle: best mechanism per type
    oracle = stats.mean(max(A[t].values()) for t in types)
    # best fixed: single mechanism with best mean across all types
    fixed = {m: stats.mean(A[t][m] for t in types) for m in MECHS}
    best_fixed_m = max(fixed, key=fixed.get)
    best_fixed = fixed[best_fixed_m]
    # random dispatch: uniform over mechanisms
    rand = stats.mean(stats.mean(A[t][m] for m in MECHS) for t in types)
    gain = oracle - best_fixed

    print(f"\n{'='*70}\n{label}\n{'='*70}")
    print(f"{'type':<12}" + "".join(f"{m:>10}" for m in MECHS) + f"{'best':>10}")
    for t in types:
        bm = max(A[t], key=A[t].get)
        print(f"{t:<12}" + "".join(f"{A[t][m]:>10.2f}" for m in MECHS) + f"{bm:>10}")
    print("-" * 62)
    print(f"{'fixed avg':<12}" + "".join(f"{fixed[m]:>10.2f}" for m in MECHS))
    print(f"\n  ORACLE (perfect dispatch)   = {oracle:.3f}")
    print(f"  BEST FIXED ({best_fixed_m:<8})       = {best_fixed:.3f}   <- the null to beat")
    print(f"  RANDOM DISPATCH             = {rand:.3f}")
    print(f"  DISPATCH CEILING (oracle-fixed) = {gain:.3f}   <- max advantage any controller can win")
    return dict(oracle=oracle, best_fixed=best_fixed, best_fixed_m=best_fixed_m,
                rand=rand, gain=gain, A=A)


def controller_acc(A, p_route):
    """Accuracy of a controller that routes to the correct mechanism with prob p_route,
    else to a uniformly-random other mechanism."""
    types = list(A)
    acc = []
    for t in types:
        best_m = max(A[t], key=A[t].get)
        others = [A[t][m] for m in MECHS if m != best_m]
        acc.append(p_route * A[t][best_m] + (1 - p_route) * stats.mean(others))
    return stats.mean(acc)


def n_required(p0, p1, alpha=0.05, power=0.80):
    z_a, z_b = 1.645, 0.842
    pbar = (p0 + p1) / 2
    num = (z_a * math.sqrt(2 * pbar * (1 - pbar))
           + z_b * math.sqrt(p0 * (1 - p0) + p1 * (1 - p1))) ** 2
    return math.ceil(num / (p1 - p0) ** 2)


def main():
    r = analyse(A_ASSUMED, "ASSUMED accuracy profile (to be replaced by measurement)")

    print(f"\n{'='*70}\nCONTROLLER accuracy vs routing accuracy (assumed profile)\n{'='*70}")
    print(f"{'route acc':>10}{'ctrl acc':>10}{'vs best_fixed':>15}{'captured gain':>15}")
    for pr in (0.5, 0.7, 0.85, 1.0):
        c = controller_acc(r["A"], pr)
        cap = (c - r["best_fixed"]) / r["gain"] if r["gain"] > 0 else 0
        print(f"{pr:>10.2f}{c:>10.3f}{c - r['best_fixed']:>+15.3f}{cap:>14.0%}")

    print(f"\n{'='*70}\nPOWER: n to detect controller > best_fixed (alpha=0.05, power=0.80)\n{'='*70}")
    print(f"{'route acc':>10}{'ctrl acc':>10}{'delta':>9}{'n/arm (total)':>16}{'n/type (x4)':>13}")
    for pr in (0.7, 0.85, 1.0):
        c = controller_acc(r["A"], pr)
        d = c - r["best_fixed"]
        if d <= 0.001:
            print(f"{pr:>10.2f}{c:>10.3f}{d:>+9.3f}   NOT DETECTABLE (controller <= best fixed)")
            continue
        n = n_required(r["best_fixed"], c)
        print(f"{pr:>10.2f}{c:>10.3f}{d:>+9.3f}{n:>16}{math.ceil(n/4):>13}")

    print(f"\n{'='*70}\nVERDICT (assumed profile)\n{'='*70}")
    if r["gain"] < 0.05:
        print("  DISPATCH CEILING < 0.05 -> NOT WORTH RUNNING. No controller can help;")
        print("  the workload lacks mechanism specialization. Redesign the workload.")
    else:
        n85 = n_required(r["best_fixed"], controller_acc(r["A"], 0.85))
        print(f"  Dispatch ceiling = {r['gain']:.2f} (LARGE). A controller routing at 85%")
        print(f"  accuracy would beat best-fixed by {controller_acc(r['A'],0.85)-r['best_fixed']:+.2f},")
        print(f"  detectable with ~{n85} total queries ({math.ceil(n85/4)}/type).")
        print("  -> WORTH RUNNING, *conditional on the MEASURED profile showing similar")
        print("     specialization*. Next step: run_calibrate.py measures the real matrix;")
        print("     re-run this analysis on it before building the controller.")


if __name__ == "__main__":
    main()
