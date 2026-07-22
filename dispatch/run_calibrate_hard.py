"""CALIBRATION on the HARD workload (GPU) — measured go/no-go for dispatch.

Same protocol as run_calibrate.py but on tasks CoT structurally cannot solve
(out-of-context retrieval, large-integer arithmetic). Prints the measured accuracy
matrix, the dispatch ceiling, and the pre-registered verdict.
"""
from __future__ import annotations

import os
import statistics as stats
import sys

sys.path.insert(0, ".")
from dispatch.hard import MECH_FN, MECHS, TYPES, QwenLM, check, make_hard  # noqa: E402

SEEDS = [0, 1, 2]
N_PER_TYPE = 16


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
        msg = " ".join(str(x) for x in a)
        print(msg, flush=True); lines.append(msg); _dump("results.txt", "\n".join(lines))

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    P(f"device={device} torch={torch.__version__}")
    mp = find_model(); P(f"model={mp}")
    lm = QwenLM(mp, device=device) if mp else None
    if lm is None:
        P("!! no model"); return

    acc = {t: {m: [] for m in MECHS} for t in TYPES}
    for seed in SEEDS:
        qs, kb = make_hard(seed, N_PER_TYPE)
        for t in TYPES:
            tq = [q for q in qs if q.qtype == t]
            for m in MECHS:
                hits = sum(int(check(q, MECH_FN[m](lm, q, kb))) for q in tq)
                acc[t][m].append(hits / len(tq))
        P(f"  seed {seed} done")

    A = {t: {m: stats.mean(acc[t][m]) for m in MECHS} for t in TYPES}
    P("\n" + "=" * 66)
    P("MEASURED accuracy matrix — HARD workload (Qwen2.5-3B, 3 seeds)")
    P("=" * 66)
    P(f"{'type':<12}" + "".join(f"{m:>10}" for m in MECHS) + f"{'best':>10}")
    for t in TYPES:
        bm = max(A[t], key=A[t].get)
        P(f"{t:<12}" + "".join(f"{A[t][m]:>10.2f}" for m in MECHS) + f"{bm:>10}")

    fixed = {m: stats.mean(A[t][m] for t in TYPES) for m in MECHS}
    bfm = max(fixed, key=fixed.get)
    oracle = stats.mean(max(A[t].values()) for t in TYPES)
    best_fixed = fixed[bfm]
    rand = stats.mean(stats.mean(A[t][m] for m in MECHS) for t in TYPES)
    gain = oracle - best_fixed
    P("-" * 52)
    P(f"{'fixed avg':<12}" + "".join(f"{fixed[m]:>10.2f}" for m in MECHS))
    P(f"\n  ORACLE (perfect dispatch)       = {oracle:.3f}")
    P(f"  BEST FIXED ({bfm:<8})           = {best_fixed:.3f}")
    P(f"  RANDOM DISPATCH                 = {rand:.3f}")
    P(f"  DISPATCH CEILING (oracle-fixed) = {gain:.3f}")

    P("\n" + "=" * 66)
    P("GO / NO-GO (measured, hard workload)")
    P("=" * 66)
    if gain >= 0.10:
        P(f"  GO: dispatch ceiling = {gain:.2f} >= 0.10. CoT cannot span this task set;")
        P(f"  a controller has real headroom. Build run_dispatch.py (controller vs")
        P(f"  best-fixed vs random vs oracle) — n=16/type gives 80% power for +0.15.")
    else:
        P(f"  NO-GO: dispatch ceiling = {gain:.2f} < 0.10 even on capability-diverse tasks.")
        P(f"  Strong null: single mechanism suffices at this scale.")


if __name__ == "__main__":
    main()
