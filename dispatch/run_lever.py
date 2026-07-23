"""THE CALIBRATION-SIGNAL LEVER (PAPER §8) — can metacognition be given to the router?

Gate 3b measured the bound: on surface-ambiguous queries the controller routes 3/4 types
perfectly but `unknown_fact` at 0.000 — it never recognizes that it does not know a fact,
so it never dispatches to retrieval, and captures only 50% of the +0.50 dispatch gain.

This experiment asks whether a cheap CALIBRATION SIGNAL recovers that routing. Architecture:
a two-stage controller. Stage 1 = coarse type classification (surface-separable, already
works). Stage 2 = for fact-like queries only, a metacognitive probe decides
known (-> cot) vs unknown (-> retrieve). Calc routing is left untouched.

  A baseline     : stage-1 only                        (reproduces Gate 3b)
  B self-ask     : "do you know this without lookup?"  (self-report)
  C consistency  : sample the answer k=3 at T=1.0; if the samples DISAGREE, the model
                   does not know it -> retrieve        (behavioural, not self-report)

PRE-REGISTERED SUCCESS (all three required, so a degenerate "send every fact to
retrieval" strategy cannot pass):
  1. unknown_fact routing  >= 0.50   (the bound actually moves)
  2. known_fact routing    >= 0.75   (it DISCRIMINATES; retrieve returns "" for real
                                      capitals, so flipping everything would self-harm)
  3. captured fraction     >  0.50   (beats the Gate-3b baseline end-to-end)

Mechanism outputs are cached per (query, mechanism): decoding is greedy and therefore
deterministic, so caching is exact and saves the repeated 400-token CoT calls.
"""
from __future__ import annotations

import os
import statistics as stats
import sys
from collections import Counter

sys.path.insert(0, ".")
from dispatch.adversarial import ROUTER_SYS, TYPE_TO_MECH, TYPES, make_adversarial  # noqa: E402
from dispatch.hard import MECH_FN, QwenLM, check                                    # noqa: E402

SEEDS = [0, 1, 2]
N_PER_TYPE = 12
MODES = ["baseline", "selfask", "consistency"]
K_CONSISTENCY = 3


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
    return "known_fact"


def probe_selfask(lm, q):
    """Self-report: does the model claim to know it? -> True = knows."""
    r = lm.gen("Reply with only YES or NO.",
               "Can you answer the following question correctly from your own knowledge, "
               "without looking anything up in a database?\n\n" + q.prompt, 6)
    return "yes" in r.strip().lower()[:6]


def probe_consistency(lm, q, k=K_CONSISTENCY):
    """Behavioural: sample k answers; agreement => it knows, disagreement => it does not."""
    outs = [lm.gen("Answer with only the final answer, nothing else.", q.prompt, 12,
                   temperature=1.0).strip().lower() for _ in range(k)]
    norm = [o.rstrip(".").strip() for o in outs]
    return len(set(norm)) == 1


def route(lm, q, mode, cache_cls):
    """Two-stage routing. Returns predicted type.
    (Query is a frozen dataclass with a dict field -> unhashable; key caches by id.)"""
    key = id(q)
    if key not in cache_cls:
        cache_cls[key] = classify(lm, q)
    base = cache_cls[key]
    if mode == "baseline" or base not in ("known_fact", "unknown_fact"):
        return base
    knows = probe_selfask(lm, q) if mode == "selfask" else probe_consistency(lm, q)
    return "known_fact" if knows else "unknown_fact"


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

    acc = {m: [] for m in MODES}
    acc["best_fixed_cot"] = []
    acc["oracle"] = []
    route_by_type = {m: {t: [] for t in TYPES} for m in MODES}

    for seed in SEEDS:
        qs, kb = make_adversarial(seed, N_PER_TYPE)
        mech_cache: dict = {}
        cls_cache: dict = {}

        def run_mech(q, mech):
            key = (id(q), mech)
            if key not in mech_cache:
                mech_cache[key] = MECH_FN[mech](lm, q, kb)
            return mech_cache[key]

        hits = {m: 0 for m in MODES}
        hits["best_fixed_cot"] = 0
        hits["oracle"] = 0
        rt = {m: {t: [0, 0] for t in TYPES} for m in MODES}   # [correct, total]

        for q in qs:
            hits["best_fixed_cot"] += int(check(q, run_mech(q, "cot")))
            hits["oracle"] += int(check(q, run_mech(q, TYPE_TO_MECH[q.qtype])))
            for mode in MODES:
                pt = route(lm, q, mode, cls_cache)
                rt[mode][q.qtype][1] += 1
                rt[mode][q.qtype][0] += int(pt == q.qtype)
                hits[mode] += int(check(q, run_mech(q, TYPE_TO_MECH[pt])))

        n = len(qs)
        for k in list(hits):
            acc[k].append(hits[k] / n)
        for mode in MODES:
            for t in TYPES:
                c, tot = rt[mode][t]
                route_by_type[mode][t].append(c / max(tot, 1))
        P(f"  seed {seed}: " + "  ".join(f"{m}={hits[m]/n:.3f}" for m in
                                         ["best_fixed_cot", "oracle"] + MODES))

    def mu(v):
        return stats.mean(v)

    def ci(v):
        return 1.96 * (stats.stdev(v) / len(v) ** 0.5) if len(v) > 1 else 0.0

    bf, orc = mu(acc["best_fixed_cot"]), mu(acc["oracle"])
    P("\n" + "=" * 72)
    P("CALIBRATION-SIGNAL LEVER — can a probe give the router metacognition?")
    P("=" * 72)
    P(f"{'method':<16}{'accuracy':>11}{'95%CI':>9}{'captured':>11}")
    P(f"{'best_fixed(cot)':<16}{bf:>11.3f}{ci(acc['best_fixed_cot']):>9.3f}{'—':>11}")
    for m in MODES:
        cap = (mu(acc[m]) - bf) / (orc - bf) if orc > bf else 0
        P(f"{m:<16}{mu(acc[m]):>11.3f}{ci(acc[m]):>9.3f}{cap:>10.0%}")
    P(f"{'oracle':<16}{orc:>11.3f}{ci(acc['oracle']):>9.3f}{'100%':>11}")

    P("\nROUTING ACCURACY BY TYPE (the diagnostic):")
    P(f"{'type':<14}" + "".join(f"{m:>14}" for m in MODES))
    for t in TYPES:
        P(f"{t:<14}" + "".join(f"{mu(route_by_type[m][t]):>14.3f}" for m in MODES))

    P("\n" + "=" * 72)
    P("VERDICT (pre-registered: unknown>=0.50 AND known>=0.75 AND captured>0.50)")
    P("=" * 72)
    base_cap = (mu(acc["baseline"]) - bf) / (orc - bf) if orc > bf else 0
    P(f"  baseline captured {base_cap:.0%} (Gate 3b reproduction)")
    any_win = False
    for m in ("selfask", "consistency"):
        unk, kn = mu(route_by_type[m]["unknown_fact"]), mu(route_by_type[m]["known_fact"])
        cap = (mu(acc[m]) - bf) / (orc - bf) if orc > bf else 0
        ok = unk >= 0.50 and kn >= 0.75 and cap > 0.50
        any_win |= ok
        P(f"  {m:<12} unknown={unk:.3f} known={kn:.3f} captured={cap:.0%} -> "
          f"{'LEVER WORKS' if ok else 'fails'}")
        if unk >= 0.50 and kn < 0.75:
            P(f"    (degenerate: routes facts to retrieval indiscriminately, "
              f"self-harming on known facts)")
    P("\n  -> " + ("METACOGNITIVE BOUND IS LIFTABLE: a cheap calibration signal recovers "
                   "the routing the classifier could not."
                   if any_win else
                   "BOUND HOLDS: neither self-report nor consistency gives the router "
                   "usable metacognition at this scale."))


if __name__ == "__main__":
    main()
