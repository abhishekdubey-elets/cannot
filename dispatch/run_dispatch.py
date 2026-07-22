"""THE DISPATCH EXPERIMENT (GPU) — does a real controller capture the 0.50 ceiling?

Hard-workload calibration established oracle=0.99, best-fixed(cot)=0.49, ceiling=0.50.
A perfect router would win 0.50; the question is whether a REAL controller can classify
each query to the right capability net of its own errors. We compare:

  best-fixed (cot)  : the monolith null to beat (0.49)
  random dispatch   : uniform mechanism choice (~0.32)
  LLM controller    : the model itself classifies the query type, then we route to the
                      capability pre-assigned to that type (the actual architecture)
  oracle dispatch   : perfect routing (0.99) — the ceiling

Pre-registered (from power.py): n=16/type x 3 seeds gives 80% power to detect the
controller beating best-fixed by +0.15. Kill condition: controller <= best-fixed.
Also reports the controller's routing accuracy (classification correctness) directly,
which localizes any failure to routing vs mechanism.
"""
from __future__ import annotations

import os
import statistics as stats
import sys

sys.path.insert(0, ".")
from dispatch.hard import MECH_FN, MECHS, TYPES, QwenLM, check, make_hard  # noqa: E402

SEEDS = [0, 1, 2]
N_PER_TYPE = 16
# each query type's suited mechanism (what the controller should route to)
TYPE_TO_MECH = {"factual": "fast", "word": "cot", "big_arith": "tool", "kb_lookup": "retrieve"}
_CLASSES = ", ".join(TYPES)


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


def classify(lm, q):
    """The controller: the LLM labels the query type. Returns (predicted_type, raw)."""
    sys_p = ("You are a router. Classify the task into exactly one category and reply with "
             "only the category word.\n"
             "- factual: a general-knowledge question (e.g. a capital city)\n"
             "- word: a multi-step arithmetic word problem with small numbers\n"
             "- big_arith: multiplying large numbers for an exact product\n"
             "- kb_lookup: asking for a value from a private registry/database")
    raw = lm.gen(sys_p, q.prompt, 8).strip().lower()
    for t in TYPES:
        if t in raw:
            return t, raw
    return "word", raw          # default to CoT if unclear


def main():
    lines = []

    def P(*a):
        m = " ".join(str(x) for x in a)
        print(m, flush=True); lines.append(m); _dump("results.txt", "\n".join(lines))

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    P(f"device={device}")
    mp = find_model(); P(f"model={mp}")
    lm = QwenLM(mp, device=device) if mp else None
    if lm is None:
        P("!! no model"); return

    methods = ["best_fixed_cot", "random", "controller", "oracle"]
    per_seed = {m: [] for m in methods}
    route_acc_seed = []

    import random as _r
    for seed in SEEDS:
        qs, kb = make_hard(seed, N_PER_TYPE)
        rng = _r.Random(seed + 999)
        hits = {m: 0 for m in methods}
        route_hits = 0
        for q in qs:
            # best-fixed: always CoT
            hits["best_fixed_cot"] += int(check(q, MECH_FN["cot"](lm, q, kb)))
            # random dispatch
            rm = rng.choice(MECHS)
            hits["random"] += int(check(q, MECH_FN[rm](lm, q, kb)))
            # oracle: route by true type
            om = TYPE_TO_MECH[q.qtype]
            hits["oracle"] += int(check(q, MECH_FN[om](lm, q, kb)))
            # controller: LLM classifies -> route
            pt, _ = classify(lm, q)
            route_hits += int(pt == q.qtype)
            cm = TYPE_TO_MECH[pt]
            hits["controller"] += int(check(q, MECH_FN[cm](lm, q, kb)))
        n = len(qs)
        for m in methods:
            per_seed[m].append(hits[m] / n)
        route_acc_seed.append(route_hits / n)
        P(f"  seed {seed}: " + "  ".join(f"{m}={hits[m]/n:.3f}" for m in methods)
          + f"  route_acc={route_hits/n:.3f}")

    def ci(v):
        mu = stats.mean(v); se = stats.stdev(v) / len(v) ** 0.5 if len(v) > 1 else 0
        return mu, 1.96 * se

    P("\n" + "=" * 66)
    P("DISPATCH EXPERIMENT — controller vs baselines (hard workload)")
    P("=" * 66)
    P(f"{'method':<18}{'accuracy':>12}{'95% CI':>12}")
    for m in methods:
        mu, h = ci(per_seed[m]); P(f"{m:<18}{mu:>12.3f}{'±%.3f' % h:>12}")
    rmu, rh = ci(route_acc_seed)
    P(f"\nController routing accuracy = {rmu:.3f} ± {rh:.3f}")

    bf, _ = ci(per_seed["best_fixed_cot"]); ctl, ctlh = ci(per_seed["controller"])
    orc, _ = ci(per_seed["oracle"])
    P("\n" + "=" * 66)
    P("VERDICT (pre-registered: controller > best-fixed by >= 0.15)")
    P("=" * 66)
    P(f"  controller {ctl:.3f} vs best-fixed(cot) {bf:.3f}  -> delta {ctl-bf:+.3f}")
    P(f"  captured {((ctl-bf)/(orc-bf)*100) if orc>bf else 0:.0f}% of the {orc-bf:.2f} oracle gain")
    if ctl - bf >= 0.15 and ctl - ctlh > bf:
        P("  -> SUPPORTED: a real controller beats the monolith when the workload spans")
        P("     capabilities no single mechanism has. Routing accuracy above localizes")
        P("     the residual gap to oracle (classification errors).")
    elif ctl - bf > 0.05:
        P("  -> PARTIAL: controller beats monolith but routing errors eat much of the gain.")
    else:
        P("  -> NOT SUPPORTED: routing errors erase the dispatch advantage; the monolith wins.")


if __name__ == "__main__":
    main()
