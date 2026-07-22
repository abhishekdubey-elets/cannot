"""CALIBRATION (GPU) — measure the real accuracy matrix A[type][mechanism] on Qwen2.5-3B,
then print the powered go/no-go verdict for the dispatch experiment.

This is the decisive step: if the real model has mechanism specialization (oracle >>
best_fixed), a cognitive architecture has room and we proceed to the controller. If CoT
(or any single mechanism) already wins everywhere (oracle ~ best_fixed), the premise is
empirically empty at this scale — itself a finding worth reporting.

Results/traceback -> downloadable files (Kaggle does not capture stdout reliably).
"""
from __future__ import annotations

import os
import statistics as stats
import sys

sys.path.insert(0, ".")
from dispatch.mechanisms import MECHANISMS, QwenLM        # noqa: E402
from dispatch.workload import TYPES, check, make_workload  # noqa: E402

SEEDS = [0, 1, 2]
N_PER_TYPE = 16          # from power.py: 16/type detects a +0.22 dispatch effect at 80% power
MECHS = list(MECHANISMS)


def _dump(name, txt):
    try:
        open(f"/kaggle/working/{name}", "w", encoding="utf-8").write(txt)
    except Exception:
        pass


def find_model():
    for dp, dns, fns in os.walk("/kaggle/input"):
        if "config.json" in fns and any(f.endswith((".safetensors", ".bin")) for f in fns):
            if "3b-instruct" in dp.lower():
                return dp
    return None


def main():
    lines = []

    def P(*a):
        m = " ".join(str(x) for x in a)
        print(m, flush=True)
        lines.append(m)
        _dump("results.txt", "\n".join(lines))

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    P(f"device={device} torch={torch.__version__}")
    mp = find_model()
    P(f"model={mp}")
    lm = QwenLM(mp, device=device) if mp else None
    if lm is None:
        P("!! no model mounted"); return

    # A[type][mech] accuracy, averaged over seeds
    acc = {t: {m: [] for m in MECHS} for t in TYPES}
    for seed in SEEDS:
        qs = make_workload(seed, N_PER_TYPE)
        by_type = {t: [q for q in qs if q.qtype == t] for t in TYPES}
        for t in TYPES:
            for m in MECHS:
                hits = 0
                for q in by_type[t]:
                    out, _ = MECHANISMS[m](lm, q)
                    hits += int(check(q, out))
                acc[t][m].append(hits / len(by_type[t]))
        P(f"  seed {seed} done")

    A = {t: {m: stats.mean(acc[t][m]) for m in MECHS} for t in TYPES}

    P("\n" + "=" * 66)
    P("MEASURED accuracy matrix A[type][mechanism] (Qwen2.5-3B, 3 seeds)")
    P("=" * 66)
    P(f"{'type':<12}" + "".join(f"{m:>10}" for m in MECHS) + f"{'best':>10}")
    for t in TYPES:
        bm = max(A[t], key=A[t].get)
        P(f"{t:<12}" + "".join(f"{A[t][m]:>10.2f}" for m in MECHS) + f"{bm:>10}")

    fixed = {m: stats.mean(A[t][m] for t in TYPES) for m in MECHS}
    best_fixed_m = max(fixed, key=fixed.get)
    oracle = stats.mean(max(A[t].values()) for t in TYPES)
    best_fixed = fixed[best_fixed_m]
    rand = stats.mean(stats.mean(A[t][m] for m in MECHS) for t in TYPES)
    gain = oracle - best_fixed

    P("-" * 52)
    P(f"{'fixed avg':<12}" + "".join(f"{fixed[m]:>10.2f}" for m in MECHS))
    P(f"\n  ORACLE (perfect dispatch)       = {oracle:.3f}")
    P(f"  BEST FIXED ({best_fixed_m:<8})           = {best_fixed:.3f}")
    P(f"  RANDOM DISPATCH                 = {rand:.3f}")
    P(f"  DISPATCH CEILING (oracle-fixed) = {gain:.3f}")

    P("\n" + "=" * 66)
    P("GO / NO-GO (measured)")
    P("=" * 66)
    if gain >= 0.10:
        P(f"  GO: measured dispatch ceiling = {gain:.2f} >= 0.10.")
        P(f"  The 3B model HAS mechanism specialization; a controller has room to win.")
        P(f"  Proceed to run_dispatch.py (controller vs best-fixed vs random vs oracle).")
    else:
        P(f"  NO-GO: measured dispatch ceiling = {gain:.2f} < 0.10.")
        P(f"  '{best_fixed_m}' alone nearly matches the oracle -> the monolith is already")
        P(f"  near-optimal on this workload; a cognitive architecture would be overhead.")
        P(f"  This is a valid empirical result AGAINST the premise at 3B scale.")


if __name__ == "__main__":
    main()
