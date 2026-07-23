"""BEHAVIOURAL METACOGNITION — is the useful signal 'behaviour' as a class, or consistency?

Gate 4 showed a 3-sample consistency probe restores unknown-fact routing (0.000 -> 1.000)
where a YES/NO self-report probe fails degenerately. That comparison was ONE behavioural
estimator vs ONE introspective estimator, and the self-report prompt was never optimised —
so it cannot support a general claim. This experiment separates three questions:

  Q1 SCALING       how many consistency samples are actually needed? (k = 2, 3, 5)
  Q2 CLASS vs ONE  does a DIFFERENT behavioural estimator (token log-probability, one
                   forward pass, no resampling) also discriminate? If yes, the signal is
                   behavioural in general; if only consistency works, the finding is
                   about consistency specifically.
  Q3 CONFOUND      is self-report's failure prompt sensitivity? We add a fairly-elicited
                   introspective probe (numeric 0-100 confidence with a fitted threshold)
                   alongside the original YES/NO.

METHOD NOTE (thresholds). `logprob` and `selfask_conf` need a decision threshold. It is fitted
on a SEPARATE calibration seed (100) by maximising balanced accuracy, then applied unchanged to
the evaluation seeds (0,1,2). No threshold is selected on evaluation data.

METRIC. Routing accuracy on the fact pair, reported separately for known and unknown, plus
balanced accuracy. Discrimination (the Gate-4 criterion) requires known >= 0.75 AND
unknown >= 0.50 — a probe that simply declares ignorance everywhere fails it. Cost is model
calls per fact query.
"""
from __future__ import annotations

import os
import statistics as stats
import sys

sys.path.insert(0, ".")
from dispatch.adversarial import make_adversarial                    # noqa: E402
from dispatch.hard import QwenLM, gen_with_confidence                # noqa: E402

EVAL_SEEDS = [0, 1, 2]
CALIB_SEED = 100
N_PER_TYPE = 12
K_MAX = 5
FACT_TYPES = ("known_fact", "unknown_fact")


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


# ---------------------------------------------------------------- raw signals
def raw_signals(lm, q):
    """All per-query signals, computed once. Sample K_MAX times so k=2,3,5 reuse prefixes."""
    samples = [lm.gen("Answer with only the final answer, nothing else.", q.prompt, 12,
                      temperature=1.0).strip().lower().rstrip(".").strip()
               for _ in range(K_MAX)]
    yesno = lm.gen("Reply with only YES or NO.",
                   "Can you answer the following question correctly from your own knowledge, "
                   "without looking anything up in a database?\n\n" + q.prompt, 6)
    conf_raw = lm.gen("Reply with only an integer from 0 to 100.",
                      "How confident are you (0-100) that you know the correct answer to this "
                      "question from your own knowledge?\n\n" + q.prompt, 6)
    import re
    m = re.search(r"\d{1,3}", conf_raw)
    conf = min(int(m.group()), 100) if m else 0
    _, tokprob = gen_with_confidence(
        lm, "Answer with only the final answer, nothing else.", q.prompt, 12)
    return dict(samples=samples, yesno="yes" in yesno.strip().lower()[:6],
                conf=conf, tokprob=tokprob)


# ---------------------------------------------------------------- estimators
def knows_consistency(sig, k):
    return len(set(sig["samples"][:k])) == 1


ESTIMATORS = {
    "selfask_yesno":  (lambda sig, th: sig["yesno"],                 1, "introspective"),
    "selfask_conf":   (lambda sig, th: sig["conf"] >= th,            1, "introspective"),
    "consistency_k2": (lambda sig, th: knows_consistency(sig, 2),    2, "behavioural"),
    "consistency_k3": (lambda sig, th: knows_consistency(sig, 3),    3, "behavioural"),
    "consistency_k5": (lambda sig, th: knows_consistency(sig, 5),    5, "behavioural"),
    "logprob":        (lambda sig, th: sig["tokprob"] >= th,         1, "behavioural"),
}
NEEDS_THRESHOLD = {"selfask_conf": "conf", "logprob": "tokprob"}


def collect(lm, seed):
    qs, _ = make_adversarial(seed, N_PER_TYPE)
    facts = [q for q in qs if q.qtype in FACT_TYPES]
    return [(q, raw_signals(lm, q)) for q in facts]


def fit_threshold(data, field):
    """Pick the threshold maximising balanced accuracy on CALIBRATION data only."""
    vals = sorted({d[1][field] for d in data})
    cands = [(a + b) / 2 for a, b in zip(vals, vals[1:])] or [vals[0] if vals else 0]
    best, best_bal = cands[0], -1.0
    for th in cands:
        kn = [d for d in data if d[0].qtype == "known_fact"]
        un = [d for d in data if d[0].qtype == "unknown_fact"]
        ka = stats.mean(float(d[1][field] >= th) for d in kn) if kn else 0
        ua = stats.mean(float(d[1][field] < th) for d in un) if un else 0
        if (ka + ua) / 2 > best_bal:
            best_bal, best = (ka + ua) / 2, th
    return best, best_bal


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

    P(f"\ncollecting calibration signals (seed {CALIB_SEED}) ...")
    calib = collect(lm, CALIB_SEED)
    thresholds = {}
    for est, field in NEEDS_THRESHOLD.items():
        th, bal = fit_threshold(calib, field)
        thresholds[est] = th
        P(f"  {est:<16} threshold={th:.4f} (calibration balanced acc {bal:.3f})")

    per_seed = {e: {"known": [], "unknown": []} for e in ESTIMATORS}
    for seed in EVAL_SEEDS:
        P(f"collecting eval signals (seed {seed}) ...")
        data = collect(lm, seed)
        for est, (fn, cost, kind) in ESTIMATORS.items():
            th = thresholds.get(est, 0)
            kn = [d for d in data if d[0].qtype == "known_fact"]
            un = [d for d in data if d[0].qtype == "unknown_fact"]
            per_seed[est]["known"].append(stats.mean(float(fn(d[1], th)) for d in kn))
            per_seed[est]["unknown"].append(stats.mean(float(not fn(d[1], th)) for d in un))

    n_items = N_PER_TYPE * len(EVAL_SEEDS)
    P("\n" + "=" * 78)
    P(f"BEHAVIOURAL METACOGNITION — estimator comparison "
      f"({n_items} known + {n_items} unknown items, {len(EVAL_SEEDS)} seeds)")
    P("=" * 78)
    P(f"{'estimator':<17}{'kind':<15}{'known':>8}{'unknown':>9}{'balanced':>10}"
      f"{'calls':>7}  discriminates?")
    rows = {}
    for est, (fn, cost, kind) in ESTIMATORS.items():
        kn, un = stats.mean(per_seed[est]["known"]), stats.mean(per_seed[est]["unknown"])
        bal = (kn + un) / 2
        ok = kn >= 0.75 and un >= 0.50
        rows[est] = (kn, un, bal, cost, kind, ok)
        P(f"{est:<17}{kind:<15}{kn:>8.3f}{un:>9.3f}{bal:>10.3f}{cost:>7}  "
          f"{'YES' if ok else 'no'}")

    P("\n" + "=" * 78)
    P("ANSWERS")
    P("=" * 78)
    k2, k3, k5 = (rows[f"consistency_k{k}"][2] for k in (2, 3, 5))
    P(f"Q1 SCALING     balanced acc  k=2 {k2:.3f}  k=3 {k3:.3f}  k=5 {k5:.3f}")
    P(f"   -> {'k=2 already suffices (cheapest useful probe)' if k2 >= k3 - 0.02 else
             'more samples help; k=3+ needed'}")

    lp = rows["logprob"]
    P(f"\nQ2 CLASS vs ONE  logprob (behavioural, 1 call): known {lp[0]:.3f} "
      f"unknown {lp[1]:.3f} balanced {lp[2]:.3f} -> {'discriminates' if lp[5] else 'FAILS'}")
    P("   -> " + ("a SECOND, independent behavioural estimator also works: supports the "
                  "general 'behavioural signal' hypothesis"
                  if lp[5] else
                  "only consistency works among the behavioural estimators tested: the "
                  "finding is about CONSISTENCY, not behaviour as a class"))

    sy, sc = rows["selfask_yesno"], rows["selfask_conf"]
    P(f"\nQ3 CONFOUND    selfask_yesno balanced {sy[2]:.3f} ({'ok' if sy[5] else 'fails'});  "
      f"selfask_conf balanced {sc[2]:.3f} ({'ok' if sc[5] else 'fails'})")
    P("   -> " + ("a fairly-elicited introspective probe DOES work: the earlier failure was "
                  "prompt sensitivity, not a limit of introspection"
                  if sc[5] else
                  "introspection fails even when fairly elicited with a fitted threshold: "
                  "the earlier failure was not merely prompt sensitivity"))

    beh = [r for e, r in rows.items() if r[4] == "behavioural"]
    intro = [r for e, r in rows.items() if r[4] == "introspective"]
    P(f"\nSUMMARY  best behavioural balanced = {max(r[2] for r in beh):.3f}; "
      f"best introspective balanced = {max(r[2] for r in intro):.3f}")


if __name__ == "__main__":
    main()
