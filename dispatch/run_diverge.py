"""WHEN DO ESTIMATORS DISAGREE? — correctness prediction, not routing accuracy.

Metric shift: instead of "does the probe route correctly on a binary taxonomy", we ask
**which signal predicts CORRECTNESS**, via AUROC over all items, and we catalogue the two
failure modes an executive controller actually has to survive:

    high confidence -> WRONG      (the dangerous one: the controller proceeds and errs)
    low confidence  -> CORRECT    (false alarm: the controller wastes retrieval/compute)

Signals per query (all cheap, all read from the same greedy answer or k samples):
    logprob      mean token probability of the greedy answer      (1 call)
    consistency  fraction of k=5 samples matching the modal one   (5 calls)
    selfconf     self-reported 0-100 confidence / 100             (1 call)

Reported: AUROC per signal (overall), accuracy + mean signal per category, the divergence
catalogue, and pairwise signal disagreement. Saturation on the previous benchmark means any
AUROC < 1.0 here is informative — it shows the signals separating.
"""
from __future__ import annotations

import os
import statistics as stats
import sys
from collections import Counter

sys.path.insert(0, ".")
from dispatch.diverge import (CATEGORIES, auroc, gave_stale, graded,      # noqa: E402
                              make_divergence)
from dispatch.hard import QwenLM, gen_with_confidence                     # noqa: E402

SEEDS = [0, 1, 2]
N_PER_CAT = 6
K = 5
ANS_SYS = "Answer with only the final answer, nothing else."


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


def signals(lm, dq):
    answer, tokprob = gen_with_confidence(lm, ANS_SYS, dq.prompt, 12)
    samples = [lm.gen(ANS_SYS, dq.prompt, 12, temperature=1.0).strip().lower().rstrip(".")
               for _ in range(K)]
    modal = Counter(samples).most_common(1)[0][1] / K
    raw = lm.gen("Reply with only an integer from 0 to 100.",
                 "How confident are you (0-100) that you know the correct answer to this "
                 "question?\n\n" + dq.prompt, 6)
    import re
    m = re.search(r"\d{1,3}", raw)
    conf = min(int(m.group()), 100) / 100 if m else 0.0
    return dict(answer=answer, logprob=tokprob, consistency=modal, selfconf=conf)


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

    SIGS = ["logprob", "consistency", "selfconf"]
    rows = []   # (category, correct, stale, sig dict)
    for seed in SEEDS:
        P(f"collecting seed {seed} ...")
        for dq in make_divergence(seed, N_PER_CAT):
            s = signals(lm, dq)
            rows.append((dq.category, graded(dq, s["answer"]), gave_stale(dq, s["answer"]), s))

    P("\n" + "=" * 82)
    P(f"DIVERGENCE BENCHMARK — {len(rows)} items ({len(SEEDS)} seeds x {N_PER_CAT}/category)")
    P("=" * 82)

    P("\nPER-CATEGORY (accuracy and mean signal value):")
    P(f"{'category':<12}{'n':>4}{'accuracy':>10}" + "".join(f"{s:>13}" for s in SIGS))
    for c in CATEGORIES:
        sub = [r for r in rows if r[0] == c]
        if not sub:
            continue
        acc = stats.mean(float(r[1]) for r in sub)
        P(f"{c:<12}{len(sub):>4}{acc:>10.3f}"
          + "".join(f"{stats.mean(r[3][s] for r in sub):>13.3f}" for s in SIGS))
    stale_rows = [r for r in rows if r[0] == "outdated"]
    if stale_rows:
        P(f"\n  outdated: gave the STALE capital on "
          f"{stats.mean(float(r[2]) for r in stale_rows):.0%} of items")

    P("\nAUROC for predicting CORRECTNESS (0.5 = uninformative, <0.5 = anti-correlated):")
    labels = [r[1] for r in rows]
    aur = {}
    for s in SIGS:
        aur[s] = auroc([r[3][s] for r in rows], labels)
        P(f"  {s:<14}{aur[s]:.3f}")
    P(f"  -> best predictor of correctness: {max(aur, key=lambda k: aur[k])}")

    P("\nAUROC EXCLUDING the fictional control (the easy half):")
    hard = [r for r in rows if r[0] != "fictional"]
    if hard and 0 < sum(r[1] for r in hard) < len(hard):
        for s in SIGS:
            P(f"  {s:<14}{auroc([r[3][s] for r in hard], [r[1] for r in hard]):.3f}")
    else:
        P("  (degenerate: all-correct or all-incorrect on the non-fictional half)")

    P("\nDIVERGENCE CATALOGUE (threshold = per-signal median):")
    P(f"{'signal':<14}{'high-conf WRONG':>18}{'low-conf CORRECT':>19}")
    for s in SIGS:
        med = stats.median([r[3][s] for r in rows])
        hcw = sum(1 for r in rows if r[3][s] >= med and not r[1])
        lcc = sum(1 for r in rows if r[3][s] < med and r[1])
        P(f"{s:<14}{hcw:>18}{lcc:>19}")

    P("\nPAIRWISE DISAGREEMENT (fraction of items the two signals rank on opposite sides "
      "of their medians):")
    for i, a in enumerate(SIGS):
        for b in SIGS[i + 1:]:
            ma, mb = stats.median([r[3][a] for r in rows]), stats.median([r[3][b] for r in rows])
            d = stats.mean(float((r[3][a] >= ma) != (r[3][b] >= mb)) for r in rows)
            P(f"  {a:<12} vs {b:<12} {d:.3f}")

    P("\n" + "=" * 82)
    P("READING")
    P("=" * 82)
    def cat(c):
        return [r for r in rows if r[0] == c]

    # reference: mean signal on the CORRECT 'common' items = "what confident looks like"
    common = cat("common")
    base = {s: stats.mean(r[3][s] for r in common) for s in SIGS} if common else {}
    amb, out = cat("ambiguous"), cat("outdated")
    if amb:
        acc = stats.mean(float(r[1]) for r in amb)
        cons = stats.mean(r[3]["consistency"] for r in amb)
        # false alarm = correct answers whose consistency is DEPRESSED vs the common baseline
        fa = acc > 0.8 and cons < base.get("consistency", 1.0) - 0.05
        P(f"  ambiguous: accuracy {acc:.3f}, consistency {cons:.3f} "
          f"(vs common {base.get('consistency', float('nan')):.3f}) "
          f"-> {'consistency FALSE-ALARMS: right answers, depressed consistency' if fa else 'no false alarm'}")
    if out:
        acc = stats.mean(float(r[1]) for r in out)
        om = {s: stats.mean(r[3][s] for r in out) for s in SIGS}
        # confidently wrong = accuracy collapses while the INTROSPECTIVE/token signals stay
        # near baseline. Consistency is scored separately: it can partially detect stale
        # memory because a shaky recollection is emitted less consistently than a solid fact.
        blind = [s for s in ("selfconf", "logprob") if om[s] >= base.get(s, 1.0) - 0.10]
        cons_drop = base.get("consistency", 1.0) - om["consistency"]
        P(f"  outdated:  accuracy {acc:.3f}; signals "
          + ", ".join(f"{s}={om[s]:.2f}" for s in SIGS))
        if acc < 0.7 and blind:
            P(f"    -> CONFIDENTLY WRONG on {'+'.join(blind)} (near baseline while accuracy "
              f"halves); consistency partially detects it (dropped {cons_drop:.2f}) but weakly")
        else:
            P("    -> signals track the errors")
    P("\n  If any AUROC < 1.0, the benchmark is no longer saturated and the signals are "
      "separable -- which is the precondition for studying evidence FUSION.")


if __name__ == "__main__":
    main()
