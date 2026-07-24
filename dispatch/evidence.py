"""INTERNAL vs EXTERNAL evidence — the epistemic-failure experiment.

Gate 6 established three failure modes that stress evidence differently:
  ignorance   (fictional)  internal signals LOW   -> caught by internal evidence
  ambiguity   (multi-cap)  internal signals MIXED -> a policy question, not a detection gap
  stale       (relocated)  internal signals HIGH but WRONG -> INVISIBLE to internal evidence

The hypothesis this file tests is information-theoretic, not empirical tuning:

  Internal evidence (log-prob, self-confidence, consistency, entropy) is a function of the
  model's own belief state. Stale knowledge is a *confident wrong belief*. No function of the
  belief can distinguish confident-right from confident-wrong -> internal evidence CANNOT
  catch the stale row, however it is fused. Only EXTERNAL evidence (a source independent of
  the belief) can. We model the cheapest possible external source: a knowledge-freshness flag
  = does an out-of-band table say this fact changed after the model's knowledge cutoff?

Prediction (pre-registered):
  1. The best FUSED INTERNAL signal still fails on `stale` (AUROC-on-stale <= ~0.6): fusing
     internal evidence cannot recover a failure mode all its inputs are blind to.
  2. A single EXTERNAL freshness flag recovers `stale` (large jump), at ~no cost on the others.
  3. Internal + external together dominate either alone across all four modes.

This reframes the controller from 'confidence estimation' to 'evidence integration', and
locates *why* external evidence is necessary rather than merely helpful.
"""
from __future__ import annotations

import math
import statistics as stats

# --- external evidence: an out-of-band freshness table (the model never sees this) ---
# country -> True if its capital changed AFTER a typical 3B knowledge cutoff, i.e. the model's
# belief is likely stale. This is EXTERNAL: independent of the model's own logits.
RELOCATED = {"Burundi", "Myanmar", "Nigeria", "Kazakhstan", "Palau", "Ivory Coast"}


def freshness_flag(country: str) -> float:
    """External evidence in [0,1]: 1.0 = fact is fresh/stable, 0.0 = known to have changed.
    A crude, cheap signal — a lookup in a change-log, not a judgement by the model."""
    return 0.0 if country in RELOCATED else 1.0


def country_of(prompt: str) -> str:
    # "What is the capital of X?" -> X
    return prompt.split("capital of", 1)[1].strip().rstrip("?").strip()


# --- fusion: logistic combiner with weights fit on a calibration split -----------
def _sigmoid(x):
    return 1 / (1 + math.exp(-max(-30, min(30, x))))


def fit_logistic(feats, labels, iters=400, lr=0.3):
    """Tiny logistic regression (features already ~[0,1]); weights fit on CALIBRATION only."""
    d = len(feats[0])
    w = [0.0] * d
    b = 0.0
    n = len(feats)
    for _ in range(iters):
        gw = [0.0] * d
        gb = 0.0
        for x, y in zip(feats, labels):
            p = _sigmoid(sum(wi * xi for wi, xi in zip(w, x)) + b)
            e = p - y
            for j in range(d):
                gw[j] += e * x[j]
            gb += e
        for j in range(d):
            w[j] -= lr * gw[j] / n
        b -= lr * gb / n
    return w, b


def predict(feats, w, b):
    return [_sigmoid(sum(wi * xi for wi, xi in zip(w, x)) + b) for x in feats]


def auroc(scores, labels):
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return float("nan")
    wins = sum((1.0 if p > n else 0.5 if p == n else 0.0) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))
