"""INTERNAL vs EXTERNAL EVIDENCE (GPU) — does external evidence recover the stale row that
no internal signal can?

Collects, per query: three INTERNAL signals (log-prob, consistency, self-confidence) and one
EXTERNAL signal (knowledge-freshness flag, from an out-of-band change-log the model never
sees). Then compares, by AUROC-for-correctness overall AND per failure mode:

  best single internal | fused internal | external only | internal + external

Fusion weights fit on a CALIBRATION seed, applied unchanged to eval seeds. The decisive
per-category number is AUROC on `stale` (=outdated): the pre-registered claim is that fused
internal evidence stays near chance there while adding external evidence jumps it.
"""
from __future__ import annotations

import math
import os
import statistics as stats
import sys
from collections import Counter

sys.path.insert(0, ".")
from dispatch.diverge import CATEGORIES, gave_stale, graded, make_divergence   # noqa: E402
from dispatch.evidence import (auroc, country_of, fit_logistic, freshness_flag,  # noqa: E402
                               predict)
from dispatch.hard import QwenLM, gen_with_confidence                           # noqa: E402

EVAL_SEEDS = [0, 1, 2]
CALIB_SEED = 100
N_PER_CAT = 6
K = 5
INTERNAL = ["logprob", "consistency", "selfconf", "entropy"]
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
    ans, tokprob = gen_with_confidence(lm, ANS_SYS, dq.prompt, 12)
    samples = [lm.gen(ANS_SYS, dq.prompt, 12, temperature=1.0).strip().lower().rstrip(".")
               for _ in range(K)]
    counts = Counter(samples)
    modal = counts.most_common(1)[0][1] / K
    total = sum(counts.values())
    ent = -sum((c / total) * math.log(c / total) for c in counts.values())
    ent_norm = 1 - ent / math.log(K)                       # 1 = all-agree, 0 = all-different
    raw = lm.gen("Reply with only an integer from 0 to 100.",
                 "How confident are you (0-100) that you know the correct answer to this "
                 "question?\n\n" + dq.prompt, 6)
    import re
    m = re.search(r"\d{1,3}", raw)
    conf = min(int(m.group()), 100) / 100 if m else 0.0
    return dict(answer=ans, logprob=tokprob, consistency=modal, selfconf=conf,
                entropy=ent_norm, freshness=freshness_flag(country_of(dq.prompt)))


def collect(lm, seed, P):
    P(f"collecting seed {seed} ...")
    rows = []
    for dq in make_divergence(seed, N_PER_CAT):
        s = signals(lm, dq)
        rows.append((dq.category, int(graded(dq, s["answer"])), s))
    return rows


def feats(rows, keys):
    return [[r[2][k] for k in keys] for r in rows]


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

    calib = collect(lm, CALIB_SEED, P)
    ev = [r for s in EVAL_SEEDS for r in collect(lm, s, P)]
    y = [r[1] for r in ev]

    # fit the two fused models on calibration only
    wi, bi = fit_logistic(feats(calib, INTERNAL), [r[1] for r in calib])
    wa, ba = fit_logistic(feats(calib, INTERNAL + ["freshness"]),
                          [r[1] for r in calib])

    scores = {
        "logprob (best single internal)": [r[2]["logprob"] for r in ev],
        "consistency": [r[2]["consistency"] for r in ev],
        "selfconf": [r[2]["selfconf"] for r in ev],
        "FUSED internal (4 sig)": predict(feats(ev, INTERNAL), wi, bi),
        "external only (freshness)": [r[2]["freshness"] for r in ev],
        "internal + external": predict(feats(ev, INTERNAL + ["freshness"]), wa, ba),
    }

    P("\n" + "=" * 84)
    P(f"INTERNAL vs EXTERNAL EVIDENCE — AUROC for correctness ({len(ev)} items, "
      f"{len(EVAL_SEEDS)} seeds)")
    P("=" * 84)
    P(f"{'evidence source':<34}{'overall':>9}" +
      "".join(f"{c[:8]:>10}" for c in CATEGORIES))
    for name, sc in scores.items():
        per = []
        for c in CATEGORIES:
            idx = [i for i, r in enumerate(ev) if r[0] == c]
            # per-category AUROC needs both classes present in that category
            yc = [y[i] for i in idx]
            per.append(auroc([sc[i] for i in idx], yc) if 0 < sum(yc) < len(yc) else float("nan"))
        P(f"{name:<34}{auroc(sc, y):>9.3f}" +
          "".join((f"{v:>10.3f}" if v == v else f"{'--':>10}") for v in per))

    P("\n  (per-category AUROC is '--' when that category is all-correct or all-incorrect;")
    P("   only `outdated`/`stale` has a within-category correct/incorrect mix to separate.)")

    # decisive comparison on the stale row
    idx = [i for i, r in enumerate(ev) if r[0] == "outdated"]
    yc = [y[i] for i in idx]
    P("\n" + "=" * 84)
    P("DECISIVE ROW — AUROC on `outdated` (the confidently-wrong failure mode)")
    P("=" * 84)
    if 0 < sum(yc) < len(yc):
        fi = auroc([scores["FUSED internal (4 sig)"][i] for i in idx], yc)
        fx = auroc([scores["external only (freshness)"][i] for i in idx], yc)
        both = auroc([scores["internal + external"][i] for i in idx], yc)
        P(f"  fused internal (4 signals) : {fi:.3f}")
        P(f"  external only (freshness)  : {fx:.3f}")
        P(f"  internal + external        : {both:.3f}")
        P("\n  VERDICT (pre-registered):")
        P(f"    internal evidence blind to stale?   {'YES' if fi <= 0.65 else 'no'} "
          f"(fused internal AUROC {fi:.2f} on stale)")
        P(f"    external evidence recovers it?      {'YES' if fx - fi >= 0.20 else 'no'} "
          f"(+{fx - fi:.2f})")
        P(f"    combined dominates?                 {'YES' if both >= max(fi, fx) - 0.02 else 'no'}")
        if fi <= 0.65 and fx - fi >= 0.20:
            P("\n    -> CONFIRMED: no fusion of internal signals detects confident-wrong stale")
            P("       knowledge; a single external evidence source does. The controller needs")
            P("       EVIDENCE INTEGRATION (internal + external), not better calibration alone.")
    else:
        P(f"  degenerate (stale all one class: {sum(yc)}/{len(yc)} correct) — need more items")


if __name__ == "__main__":
    main()
