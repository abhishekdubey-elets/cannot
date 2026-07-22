"""ADVERSARIAL routing experiment (GPU) — does surface-ambiguity break dispatch?

Same four-method comparison as run_dispatch.py, on the adversarial workload where
known_fact and unknown_fact are surface-identical. Reports routing accuracy PER TYPE,
which localizes any failure: we expect ~1.0 on the calc pair (magnitude is visible) and
< 1.0 on the fact pair (requires the model to judge its own knowledge).
"""
from __future__ import annotations

import os
import random as _r
import statistics as stats
import sys

sys.path.insert(0, ".")
from dispatch.adversarial import (MECHS, ROUTER_SYS, TYPE_TO_MECH, TYPES,  # noqa: E402
                                  make_adversarial)
from dispatch.hard import MECH_FN, QwenLM, check                          # noqa: E402

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


def classify(lm, q):
    raw = lm.gen(ROUTER_SYS, q.prompt, 8).strip().lower()
    for t in TYPES:
        if t in raw:
            return t
    return "known_fact"          # default to answer-from-knowledge


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

    methods = ["best_fixed_cot", "controller", "oracle"]
    per_seed = {m: [] for m in methods}
    route_overall, route_by_type = [], {t: [] for t in TYPES}

    for seed in SEEDS:
        qs, kb = make_adversarial(seed, N_PER_TYPE)
        hits = {m: 0 for m in methods}
        rh = 0
        rht = {t: 0 for t in TYPES}
        cnt = {t: 0 for t in TYPES}
        for q in qs:
            hits["best_fixed_cot"] += int(check(q, MECH_FN["cot"](lm, q, kb)))
            hits["oracle"] += int(check(q, MECH_FN[TYPE_TO_MECH[q.qtype]](lm, q, kb)))
            pt = classify(lm, q)
            correct_route = int(pt == q.qtype)
            rh += correct_route
            rht[q.qtype] += correct_route
            cnt[q.qtype] += 1
            hits["controller"] += int(check(q, MECH_FN[TYPE_TO_MECH[pt]](lm, q, kb)))
        n = len(qs)
        for m in methods:
            per_seed[m].append(hits[m] / n)
        route_overall.append(rh / n)
        for t in TYPES:
            route_by_type[t].append(rht[t] / cnt[t])
        P(f"  seed {seed}: " + "  ".join(f"{m}={hits[m]/n:.3f}" for m in methods)
          + f"  route={rh/n:.3f}")

    def ci(v):
        mu = stats.mean(v); h = 1.96 * (stats.stdev(v) / len(v) ** 0.5 if len(v) > 1 else 0)
        return mu, h

    P("\n" + "=" * 66)
    P("ADVERSARIAL DISPATCH — controller vs baselines")
    P("=" * 66)
    P(f"{'method':<18}{'accuracy':>12}{'95% CI':>12}")
    for m in methods:
        mu, h = ci(per_seed[m]); P(f"{m:<18}{mu:>12.3f}{'±%.3f' % h:>12}")

    P(f"\nRouting accuracy OVERALL = {ci(route_overall)[0]:.3f} ± {ci(route_overall)[1]:.3f}")
    P("Routing accuracy BY TYPE (the diagnostic):")
    for t in TYPES:
        P(f"  {t:<14} {ci(route_by_type[t])[0]:.3f} ± {ci(route_by_type[t])[1]:.3f}")

    bf = ci(per_seed["best_fixed_cot"])[0]; ctl = ci(per_seed["controller"])[0]
    orc = ci(per_seed["oracle"])[0]
    P("\n" + "=" * 66)
    P("VERDICT")
    P("=" * 66)
    P(f"  controller {ctl:.3f} vs best-fixed(cot) {bf:.3f} vs oracle {orc:.3f}")
    cap = (ctl - bf) / (orc - bf) * 100 if orc > bf else 0
    P(f"  controller captured {cap:.0f}% of the oracle gain (clean workload: 100%)")
    if ctl - bf >= 0.15:
        P("  -> STILL SUPPORTED under surface-ambiguity, but note the routing-by-type")
        P("     diagnostic: any dropped gain localizes to the fact pair (metacognition).")
    else:
        P("  -> COMPLICATED: surface-ambiguous routing erodes the dispatch advantage.")
        P("     Dispatch is bounded by the router's metacognition, not just capability")
        P("     disjointness. This is the honest limit of the clean result.")


if __name__ == "__main__":
    main()
