"""Statistics core for UGEN. Dependency-free (math/random only) so it runs identically in
Kaggle kernels and locally. Every estimator is unit-tested against closed-form values in
tests/test_stats.py, which must pass before any GPU run trusts these numbers.

Conventions: labels are 0/1 ints; "scores" are arbitrary reals (rank metrics); "probs" are
in [0,1] (calibration metrics). All MI-family estimators are plug-in on quantile bins and
therefore positively biased — always report them next to the same statistic computed on a
shuffled control (see shuffled_control()).
"""
from __future__ import annotations

import math
import random

EPS = 1e-12


# ---------------------------------------------------------------- rank metrics

def _rankdata(x):
    n = len(x)
    order = sorted(range(n), key=lambda i: x[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and x[order[j + 1]] == x[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def auroc(scores, labels):
    """P(score(pos) > score(neg)) with ties at 0.5, via the rank/Mann-Whitney identity
    (O(n log n), unlike the pairwise O(n^2) form)."""
    n_pos = sum(1 for l in labels if l)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rankdata(scores)
    rank_sum_pos = sum(r for r, l in zip(ranks, labels) if l)
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def auprc(scores, labels):
    """Average precision (step-wise AP sum)."""
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    total_pos = sum(labels)
    if total_pos == 0:
        return float("nan")
    tp = fp = 0
    ap = prev_recall = 0.0
    for i in order:
        if labels[i]:
            tp += 1
        else:
            fp += 1
        recall = tp / total_pos
        ap += (tp / (tp + fp)) * (recall - prev_recall)
        prev_recall = recall
    return ap


def hanley_mcneil_se(a, n_pos, n_neg):
    """Analytic SE of AUROC (Hanley & McNeil 1982) — used for power analysis only;
    inference uses the bootstrap."""
    q1 = a / (2 - a)
    q2 = 2 * a * a / (1 + a)
    var = (a * (1 - a) + (n_pos - 1) * (q1 - a * a) + (n_neg - 1) * (q2 - a * a)) / (n_pos * n_neg)
    return math.sqrt(max(0.0, var))


def cliffs_delta(scores, labels):
    """Cliff's delta between pos and neg score distributions == 2*AUROC - 1."""
    a = auroc(scores, labels)
    return float("nan") if a != a else 2 * a - 1


# ------------------------------------------------------- probability metrics

def brier(probs, labels):
    return sum((p - y) ** 2 for p, y in zip(probs, labels)) / len(probs)


def nll(probs, labels):
    """Mean negative log likelihood (nats), probabilities clipped to [EPS, 1-EPS]."""
    tot = 0.0
    for p, y in zip(probs, labels):
        p = min(1 - EPS, max(EPS, p))
        tot += -math.log(p) if y else -math.log(1 - p)
    return tot / len(probs)


def _bin_stats(probs, labels, bins, quantile):
    n = len(probs)
    if quantile:
        order = sorted(range(n), key=lambda i: probs[i])
        groups = [[] for _ in range(bins)]
        for rank, i in enumerate(order):
            groups[min(bins - 1, rank * bins // n)].append(i)
    else:
        groups = [[] for _ in range(bins)]
        for i, p in enumerate(probs):
            b = min(bins - 1, int(p * bins))
            groups[b].append(i)
    out = []
    for idx in groups:
        if not idx:
            continue
        conf = sum(probs[i] for i in idx) / len(idx)
        acc = sum(labels[i] for i in idx) / len(idx)
        out.append((len(idx), conf, acc))
    return out


def ece(probs, labels, bins=10, quantile=False):
    n = len(probs)
    return sum(w / n * abs(acc - conf) for w, conf, acc in _bin_stats(probs, labels, bins, quantile))


def mce(probs, labels, bins=10, quantile=False):
    st = _bin_stats(probs, labels, bins, quantile)
    return max(abs(acc - conf) for _, conf, acc in st) if st else float("nan")


def reliability_curve(probs, labels, bins=10, quantile=True):
    """[(weight, mean_conf, mean_acc)] for reliability diagrams. Quantile bins by default
    (verbalized confidence clusters at round numbers — equal-width bins go empty)."""
    return _bin_stats(probs, labels, bins, quantile)


# --------------------------------------------------------------- information

def _quantile_bins(signal, bins):
    """Equal-frequency bins with the constraint that tied values never straddle a bin
    boundary (each distinct value goes to the bin of its median rank). Signals like
    consistency have few distinct values; rank-splitting ties would corrupt the joint
    distribution and bias every MI/PID estimate built on it."""
    n = len(signal)
    order = sorted(range(n), key=lambda i: signal[i])
    # median rank per distinct value -> bin
    value_ranks = {}
    for rank, i in enumerate(order):
        value_ranks.setdefault(signal[i], []).append(rank)
    value_bin = {}
    for v, ranks in value_ranks.items():
        mid = ranks[len(ranks) // 2]
        value_bin[v] = min(bins - 1, mid * bins // n)
    return [value_bin[s] for s in signal]


def mutual_info(signal, labels, bins=5):
    """MI(quantile-binned signal ; binary label), nats. Plug-in (positively biased)."""
    n = len(signal)
    binof = _quantile_bins(signal, bins)
    joint, px, py = {}, [0] * bins, [0, 0]
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


def conditional_mutual_info(signal, labels, cond, bins=3):
    """I(signal ; label | cond), cond already discrete. Strata with one label contribute 0."""
    n = len(signal)
    total = 0.0
    for z in set(cond):
        idx = [i for i in range(n) if cond[i] == z]
        if len(idx) < 2:
            continue
        ys = [labels[i] for i in idx]
        if len(set(ys)) < 2:
            continue
        total += (len(idx) / n) * mutual_info(
            [signal[i] for i in idx], ys, bins=min(bins, max(2, len(idx) // 4)))
    return total


def pid_williams_beer(x1, x2, labels, bins=3):
    """Two-source partial information decomposition (Williams-Beer I_min) of
    I(Y ; X1, X2) into redundant / unique_1 / unique_2 / synergistic, in nats.

    Sources are quantile-binned. Plug-in estimator — positively biased on small samples, so
    callers MUST report the same decomposition on a shuffled control (PLAN.md section 4/5).
    Approximation justification: exact PID needs an optimisation over lattices for >2
    sources; for two sources Williams-Beer is closed-form and is the standard baseline
    decomposition (Williams & Beer 2010). Known caveat: I_min can over-count redundancy for
    mixed source-target dependencies; we use it comparatively (same estimator everywhere),
    never as an absolute quantity.
    """
    n = len(labels)
    b1, b2 = _quantile_bins(x1, bins), _quantile_bins(x2, bins)
    ys = [int(l) for l in labels]
    py = {}
    for y in ys:
        py[y] = py.get(y, 0) + 1
    # specific information I_spec(Y=y ; X) for each source
    def spec_info(bx):
        pxy, px = {}, {}
        for b, y in zip(bx, ys):
            pxy[(b, y)] = pxy.get((b, y), 0) + 1
            px[b] = px.get(b, 0) + 1
        out = {}
        for y, cy in py.items():
            s = 0.0
            for b, cb in px.items():
                c = pxy.get((b, y), 0)
                if c == 0:
                    continue
                p_x_given_y = c / cy
                p_y_given_x = c / cb
                s += p_x_given_y * (math.log(p_y_given_x) - math.log(cy / n))
            out[y] = max(0.0, s)
        return out
    s1, s2 = spec_info(b1), spec_info(b2)
    red = sum((cy / n) * min(s1[y], s2[y]) for y, cy in py.items())
    i1 = mutual_info(x1, ys, bins=bins)
    i2 = mutual_info(x2, ys, bins=bins)
    joint = [b1[i] * bins + b2[i] for i in range(n)]
    i12 = _discrete_mi(joint, ys)
    return dict(redundant=red, unique_1=max(0.0, i1 - red), unique_2=max(0.0, i2 - red),
                synergy=max(0.0, i12 - i1 - i2 + red), total=i12, mi_1=i1, mi_2=i2)


def _discrete_mi(xs, ys):
    n = len(xs)
    joint, px, py = {}, {}, {}
    for x, y in zip(xs, ys):
        joint[(x, y)] = joint.get((x, y), 0) + 1
        px[x] = px.get(x, 0) + 1
        py[y] = py.get(y, 0) + 1
    mi = 0.0
    for (x, y), c in joint.items():
        pxy = c / n
        mi += pxy * math.log(pxy / ((px[x] / n) * (py[y] / n)))
    return max(0.0, mi)


# ------------------------------------------------------------------ inference

def bootstrap_ci(fn, scores, labels, reps=1000, seed=0):
    """Percentile bootstrap 95% CI of fn(scores, labels). Returns (mean, lo, hi)."""
    rng = random.Random(seed)
    n = len(scores)
    vals = []
    for _ in range(reps):
        idx = [rng.randrange(n) for _ in range(n)]
        v = fn([scores[i] for i in idx], [labels[i] for i in idx])
        if v == v:
            vals.append(v)
    if not vals:
        return (float("nan"),) * 3
    vals.sort()
    return (sum(vals) / len(vals), vals[int(0.025 * len(vals))],
            vals[min(len(vals) - 1, int(0.975 * len(vals)))])


def diff_ci(fn, sa, la, sb, lb, reps=1000, seed=0):
    """Bootstrap 95% CI of fn(group a) - fn(group b), independent resampling per group."""
    rng = random.Random(seed)
    na, nb = len(sa), len(sb)
    vals = []
    for _ in range(reps):
        ia = [rng.randrange(na) for _ in range(na)]
        ib = [rng.randrange(nb) for _ in range(nb)]
        va = fn([sa[i] for i in ia], [la[i] for i in ia])
        vb = fn([sb[i] for i in ib], [lb[i] for i in ib])
        if va == va and vb == vb:
            vals.append(va - vb)
    if not vals:
        return (float("nan"),) * 3
    vals.sort()
    return (sum(vals) / len(vals), vals[int(0.025 * len(vals))],
            vals[min(len(vals) - 1, int(0.975 * len(vals)))])


def permutation_pvalue(fn, sa, la, sb, lb, perms=5000, seed=0):
    """Two-sided permutation test for fn(a) - fn(b): pool items, shuffle group membership.
    Tests the null that the two groups are exchangeable w.r.t. the statistic."""
    obs_a, obs_b = fn(sa, la), fn(sb, lb)
    if obs_a != obs_a or obs_b != obs_b:
        return float("nan"), float("nan")
    obs = obs_a - obs_b
    pool_s = list(sa) + list(sb)
    pool_l = list(la) + list(lb)
    na = len(sa)
    rng = random.Random(seed)
    idx = list(range(len(pool_s)))
    hits = 0
    valid = 0
    for _ in range(perms):
        rng.shuffle(idx)
        ia, ib = idx[:na], idx[na:]
        va = fn([pool_s[i] for i in ia], [pool_l[i] for i in ia])
        vb = fn([pool_s[i] for i in ib], [pool_l[i] for i in ib])
        if va != va or vb != vb:
            continue
        valid += 1
        if abs(va - vb) >= abs(obs) - 1e-12:
            hits += 1
    return obs, (hits + 1) / (valid + 1) if valid else float("nan")


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, order-preserving. NaNs pass through."""
    m = sum(1 for p in pvals if p == p)
    order = sorted((i for i, p in enumerate(pvals) if pvals[i] == pvals[i]),
                   key=lambda i: pvals[i])
    adj = [float("nan")] * len(pvals)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def spearman(x, y):
    rx, ry = _rankdata(x), _rankdata(y)
    n = len(rx)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return 0.0 if dx == 0 or dy == 0 else num / (dx * dy)


# --------------------------------------------------------------- calibration

def isotonic_fit(scores, labels):
    """Pool-adjacent-violators isotonic regression: returns (thresholds, values) mapping
    score -> calibrated P(correct). Monotone non-decreasing."""
    pairs = sorted(zip(scores, labels))
    blocks = [[s, float(l), 1.0] for s, l in pairs]  # [max_score, mean, weight]
    stack = []
    for b in blocks:
        stack.append(b)
        while len(stack) > 1 and stack[-2][1] >= stack[-1][1]:
            s2, m2, w2 = stack.pop()
            s1, m1, w1 = stack.pop()
            stack.append([max(s1, s2), (m1 * w1 + m2 * w2) / (w1 + w2), w1 + w2])
    return [b[0] for b in stack], [b[1] for b in stack]


def isotonic_apply(model, scores):
    """thresholds[i] is the max training score of block i -> a score belongs to the first
    block whose max covers it; scores above all blocks get the last block's value."""
    thresholds, values = model
    out = []
    for s in scores:
        v = values[-1]
        for t, val in zip(thresholds, values):
            if s <= t:
                v = val
                break
        out.append(v)
    return out


def selective_threshold(probs, labels, target_precision=0.8):
    """Smallest threshold t such that precision among {p >= t} >= target (on train data).
    Returns (t, coverage, precision). If unattainable, returns (inf, 0, nan)."""
    order = sorted(set(probs))
    best = (float("inf"), 0.0, float("nan"))
    for t in order:
        sel = [(p, y) for p, y in zip(probs, labels) if p >= t]
        if not sel:
            continue
        prec = sum(y for _, y in sel) / len(sel)
        if prec >= target_precision:
            best = (t, len(sel) / len(probs), prec)
            break
    return best


def shuffled_control(values, seed=0):
    """Permute a real signal across items — the empirical null carried through every table."""
    rng = random.Random(seed)
    out = list(values)
    rng.shuffle(out)
    return out


def dynamic_range(x):
    n = len(x)
    m = sum(x) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in x) / n)
    return dict(mean=m, sd=sd, lo=min(x), hi=max(x),
                n_unique=len(set(round(v, 4) for v in x)))
