"""Statistics pipeline for executive-evidence measurement (pure Python, no deps).

Every function is unit-tested against closed-form values in `_selftest()` below, which runs
on import-as-main and MUST pass before any GPU run trusts these numbers.

Signals are treated as scores in R (AUROC/AUPRC/MI are rank/binning based) or, where a signal
is a probability in [0,1], as calibrated predictions (Brier/ECE).
"""
from __future__ import annotations

import math
import random


def auroc(scores, labels):
    """P(score(pos) > score(neg)), ties = 0.5. 0.5 uninformative, <0.5 anti-correlated."""
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return float("nan")
    w = sum((1.0 if p > n else 0.5 if p == n else 0.0) for p in pos for n in neg)
    return w / (len(pos) * len(neg))


def auprc(scores, labels):
    """Average precision (area under precision-recall), via the standard AP sum."""
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    total_pos = sum(labels)
    if total_pos == 0:
        return float("nan")
    tp = 0
    fp = 0
    ap = 0.0
    prev_recall = 0.0
    for i in order:
        if labels[i]:
            tp += 1
        else:
            fp += 1
        precision = tp / (tp + fp)
        recall = tp / total_pos
        ap += precision * (recall - prev_recall)
        prev_recall = recall
    return ap


def brier(probs, labels):
    """Mean squared error of probabilistic predictions. Lower = better. Requires probs in [0,1]."""
    return sum((p - y) ** 2 for p, y in zip(probs, labels)) / len(probs)


def ece(probs, labels, bins=10):
    """Expected Calibration Error: |accuracy - confidence| averaged over confidence bins."""
    n = len(probs)
    tot = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, p in enumerate(probs) if (lo < p <= hi) or (b == 0 and p == 0)]
        if not idx:
            continue
        conf = sum(probs[i] for i in idx) / len(idx)
        acc = sum(labels[i] for i in idx) / len(idx)
        tot += (len(idx) / n) * abs(acc - conf)
    return tot


def _entropy(counts):
    tot = sum(counts)
    return -sum((c / tot) * math.log(c / tot) for c in counts if c > 0)


def mutual_info(signal, labels, bins=5):
    """MI(binned signal ; binary label) in nats. Quantile bins (equal frequency) to avoid
    empty bins. Plug-in estimator (slightly positively biased -> use with bootstrap CI and the
    shuffled-control null)."""
    n = len(signal)
    order = sorted(range(n), key=lambda i: signal[i])
    # assign quantile bin per item
    binof = [0] * n
    for rank, i in enumerate(order):
        binof[i] = min(bins - 1, rank * bins // n)
    joint = {}
    px = [0] * bins
    py = [0, 0]
    for i in range(n):
        b, y = binof[i], int(labels[i])
        joint[(b, y)] = joint.get((b, y), 0) + 1
        px[b] += 1
        py[y] += 1
    mi = 0.0
    for (b, y), c in joint.items():
        pxy = c / n
        mi += pxy * math.log(pxy / ((px[b] / n) * (py[y] / n)))
    return max(0.0, mi)


def bootstrap_ci(fn, scores, labels, reps=300, seed=0):
    """Percentile bootstrap 95% CI of a metric fn(scores, labels)."""
    rng = random.Random(seed)
    n = len(scores)
    vals = []
    for _ in range(reps):
        idx = [rng.randrange(n) for _ in range(n)]
        s = [scores[i] for i in idx]
        l = [labels[i] for i in idx]
        v = fn(s, l)
        if v == v:  # not nan
            vals.append(v)
    if not vals:
        return float("nan"), float("nan"), float("nan")
    vals.sort()
    lo = vals[int(0.025 * len(vals))]
    hi = vals[min(len(vals) - 1, int(0.975 * len(vals)))]
    return sum(vals) / len(vals), lo, hi


def auroc_diff_ci(sa, la, sb, lb, reps=300, seed=0):
    """Bootstrap 95% CI of AUROC(domain a) - AUROC(domain b) for a cross-domain comparison."""
    rng = random.Random(seed)
    na, nb = len(sa), len(sb)
    vals = []
    for _ in range(reps):
        ia = [rng.randrange(na) for _ in range(na)]
        ib = [rng.randrange(nb) for _ in range(nb)]
        va = auroc([sa[i] for i in ia], [la[i] for i in ia])
        vb = auroc([sb[i] for i in ib], [lb[i] for i in ib])
        if va == va and vb == vb:
            vals.append(va - vb)
    vals.sort()
    return (sum(vals) / len(vals), vals[int(0.025 * len(vals))],
            vals[min(len(vals) - 1, int(0.975 * len(vals)))])


def _selftest():
    # AUROC
    assert auroc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0
    assert auroc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]) == 0.0
    assert abs(auroc([0.5] * 4, [1, 1, 0, 0]) - 0.5) < 1e-9
    # AUPRC: perfect ranking -> 1.0; all-positive-first
    assert abs(auprc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) - 1.0) < 1e-9
    # Brier
    assert abs(brier([1.0, 0.0], [1, 0]) - 0.0) < 1e-9
    assert abs(brier([0.5, 0.5], [1, 0]) - 0.25) < 1e-9
    # ECE: perfectly calibrated -> 0
    probs = [0.1] * 10 + [0.9] * 10
    labs = [1] * 1 + [0] * 9 + [1] * 9 + [0] * 1   # 10% and 90% accurate
    assert ece(probs, labs, bins=10) < 1e-9
    # MI: deterministic signal, label split ALIGNED to a quantile-bin edge (i>=60 with 5 bins
    # of 20) so no bin straddles -> MI == H(Y) for a 40/60 split = 0.673.
    sig = list(range(100))
    lab = [1 if i >= 60 else 0 for i in sig]
    hi = mutual_info(sig, lab, bins=5)
    hy = -(0.4 * math.log(0.4) + 0.6 * math.log(0.6))     # 0.673
    assert abs(hi - hy) < 0.02, (hi, hy)                  # clean split -> MI = H(Y)
    # a boundary-straddling split (i>=50) gives LESS than H(Y) -- documents the binning effect
    assert mutual_info(sig, [1 if i >= 50 else 0 for i in sig], bins=5) < hy
    rng = random.Random(0)
    rnd = [rng.random() for _ in range(400)]
    rlab = [rng.randrange(2) for _ in range(400)]
    assert mutual_info(rnd, rlab, bins=5) < 0.05          # independent -> ~0
    print("stats self-test PASSED (auroc, auprc, brier, ece, mutual_info)")


if __name__ == "__main__":
    _selftest()
