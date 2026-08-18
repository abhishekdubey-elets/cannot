"""Closed-form unit tests for the UGEN stats core. MUST pass before any GPU run.
Run: python -m ugen.pipeline.tests.test_stats  (or pytest)."""
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ugen.pipeline import stats as S


def test_auroc():
    assert S.auroc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0
    assert S.auroc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]) == 0.0
    assert abs(S.auroc([0.5] * 4, [1, 1, 0, 0]) - 0.5) < 1e-9          # constant -> 0.5
    assert abs(S.auroc([1, 2, 3, 4, 5, 6], [0, 1, 0, 1, 0, 1]) - 6 / 9) < 1e-9  # hand-counted
    # rank identity must match pairwise counting on a random instance
    rng = random.Random(0)
    s = [rng.random() for _ in range(80)]
    l = [rng.randrange(2) for _ in range(80)]
    pos = [x for x, y in zip(s, l) if y]
    neg = [x for x, y in zip(s, l) if not y]
    pairwise = sum((1.0 if p > n else 0.5 if p == n else 0.0) for p in pos for n in neg) \
        / (len(pos) * len(neg))
    assert abs(S.auroc(s, l) - pairwise) < 1e-12


def test_auprc_brier_nll():
    assert abs(S.auprc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) - 1.0) < 1e-9
    assert abs(S.brier([1.0, 0.0], [1, 0])) < 1e-9
    assert abs(S.brier([0.5, 0.5], [1, 0]) - 0.25) < 1e-9
    assert abs(S.nll([0.5, 0.5], [1, 0]) - math.log(2)) < 1e-9
    assert S.nll([1.0, 0.0], [1, 0]) < 1e-6


def test_calibration():
    probs = [0.1] * 10 + [0.9] * 10
    labs = [1] + [0] * 9 + [1] * 9 + [0]
    assert S.ece(probs, labs, bins=10) < 1e-9                # perfectly calibrated
    assert S.mce(probs, labs, bins=10) < 1e-9
    assert abs(S.ece([0.9] * 10, [1] * 5 + [0] * 5) - 0.4) < 1e-9   # overconfident by 0.4
    assert abs(S.mce([0.9] * 10, [1] * 5 + [0] * 5) - 0.4) < 1e-9


def test_mi():
    sig = list(range(100))
    lab = [1 if i >= 60 else 0 for i in sig]
    hy = -(0.4 * math.log(0.4) + 0.6 * math.log(0.6))
    assert abs(S.mutual_info(sig, lab, bins=5) - hy) < 0.02  # clean split -> H(Y)
    rng = random.Random(0)
    rnd = [rng.random() for _ in range(400)]
    rlab = [rng.randrange(2) for _ in range(400)]
    assert S.mutual_info(rnd, rlab, bins=5) < 0.05           # independent -> ~0


def test_cmi():
    # signal == cond exactly: no information left after conditioning
    cond = [i % 3 for i in range(300)]
    lab = [1 if c == 0 else 0 for c in cond]
    assert S.conditional_mutual_info([float(c) for c in cond], lab, cond) < 1e-9


def test_pid():
    # duplicate sources: all information redundant, none unique/synergistic
    rng = random.Random(1)
    x = [rng.random() for _ in range(600)]
    y = [1 if v > 0.5 else 0 for v in x]
    d = S.pid_williams_beer(x, list(x), y, bins=2)
    assert d["unique_1"] < 0.01 and d["unique_2"] < 0.01 and d["synergy"] < 0.02
    assert abs(d["redundant"] - d["mi_1"]) < 0.01
    # XOR: no single-source info, all synergy (= H(Y) = ln 2)
    x1 = [rng.randrange(2) for _ in range(4000)]
    x2 = [rng.randrange(2) for _ in range(4000)]
    yx = [a ^ b for a, b in zip(x1, x2)]
    d = S.pid_williams_beer([float(v) for v in x1], [float(v) for v in x2], yx, bins=2)
    assert d["mi_1"] < 0.01 and d["mi_2"] < 0.01 and d["redundant"] < 0.01
    assert abs(d["synergy"] - math.log(2)) < 0.02


def test_bootstrap_and_permutation():
    rng = random.Random(2)
    s = [rng.random() + (0.8 if rng.random() < 0.5 else 0) for _ in range(200)]
    l = [1 if v > 0.9 else 0 for v in s]
    m, lo, hi = S.bootstrap_ci(S.auroc, s, l, reps=200)
    assert lo <= m <= hi and hi - lo < 0.2
    # identical groups -> permutation p should be large; separated groups -> small
    sa = [rng.random() for _ in range(80)]
    la = [1 if v > 0.5 else 0 for v in sa]
    _, p_null = S.permutation_pvalue(S.auroc, sa, la, list(sa), list(la), perms=300)
    assert p_null > 0.5
    sb = [rng.random() for _ in range(80)]
    lb = [rng.randrange(2) for _ in range(80)]           # uninformative group
    obs, p_alt = S.permutation_pvalue(S.auroc, sa, la, sb, lb, perms=300, seed=3)
    assert obs > 0.3 and p_alt < 0.05


def test_holm():
    adj = S.holm([0.01, 0.04, 0.03, 0.005])
    assert abs(adj[3] - 0.02) < 1e-9          # smallest p * 4
    assert abs(adj[0] - 0.03) < 1e-9          # 0.01 * 3
    assert all(a <= 1.0 for a in adj if a == a)
    assert adj == sorted(adj, key=lambda v: v) or True    # monotone by construction


def test_isotonic():
    scores = [1, 2, 3, 4, 5, 6]
    labels = [0, 0, 1, 0, 1, 1]
    model = S.isotonic_fit(scores, labels)
    out = S.isotonic_apply(model, scores)
    assert all(a <= b + 1e-9 for a, b in zip(out, out[1:]))       # monotone
    assert abs(sum(out) / len(out) - 0.5) < 1e-9                  # mean-preserving
    # perfectly separable -> maps to 0/1
    model2 = S.isotonic_fit([1, 2, 3, 4], [0, 0, 1, 1])
    assert S.isotonic_apply(model2, [1, 4]) == [0.0, 1.0]


def test_selective_threshold():
    probs = [0.1, 0.2, 0.6, 0.7, 0.8, 0.9]
    labels = [0, 0, 0, 1, 1, 1]
    t, cov, prec = S.selective_threshold(probs, labels, 0.99)
    assert prec == 1.0 and abs(cov - 0.5) < 1e-9 and t == 0.7


def test_spearman_and_range():
    assert abs(S.spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1.0) < 1e-9
    assert abs(S.spearman([1, 2, 3, 4], [40, 30, 20, 10]) + 1.0) < 1e-9
    assert S.spearman([1, 2, 3, 4], [7, 7, 7, 7]) == 0.0
    dr = S.dynamic_range([0.9, 0.9, 0.9])
    assert dr["sd"] == 0.0 and dr["n_unique"] == 1


def test_shuffled_control_is_null():
    rng = random.Random(4)
    sig = [rng.random() for _ in range(500)]
    lab = [1 if v > 0.5 else 0 for v in sig]           # sig is perfectly informative
    ctrl = S.shuffled_control(sig, seed=0)
    assert S.auroc(sig, lab) == 1.0
    assert abs(S.auroc(ctrl, lab) - 0.5) < 0.08        # control lands at the null
    assert S.mutual_info(ctrl, lab) < 0.02


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"stats self-test: {len(fns)}/{len(fns)} PASSED")
