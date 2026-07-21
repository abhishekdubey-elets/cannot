"""PAPER §10.3 — Formal power analysis, run BEFORE any method is re-run.

The §10.2 pilot failed because the null (random-policy accuracy) was 0.444, not the
0.25 we assumed, and n=24 gave SE~0.10 — larger than any between-method gap. This
script fixes that by DESIGN rather than by luck:

  (a) Empirically measure the null accuracy of a random policy on several candidate
      benchmark designs (grid size, wall density, and a UNIQUE-optimal-move filter).
  (b) Pick the design whose null is low and well-defined (target: chance ~ 0.25).
  (c) Compute the sample size n required to detect a target effect delta at that null,
      via the two-proportion normal approximation AND a Monte-Carlo check.

No world model, no LLM — pure combinatorics + BFS. Runs in seconds, CPU or GPU.
"""
from __future__ import annotations

import math
import random
import statistics as stats
import sys

sys.path.insert(0, ".")
from prospect.training.dataset import N_ACTIONS, GridWorld  # noqa: E402


def measure_null(grid, wall_p, unique_only, n=3000, seed=0):
    """Empirical random-policy accuracy + fraction of states usable under the filter.

    unique_only=True keeps only states with EXACTLY ONE optimal move -> chance = 1/4
    by construction, and verifies it empirically."""
    rng = random.Random(seed)
    hits, kept, seen = 0, 0, 0
    i = 0
    while kept < n and i < n * 40:
        env = GridWorld(n=grid, wall_p=wall_p, seed=seed * 100003 + i)
        i += 1
        opt = env.optimal_actions()
        if not opt:
            continue
        seen += 1
        if unique_only and len(opt) != 1:
            continue
        kept += 1
        hits += int(rng.randrange(N_ACTIONS) in opt)
    return dict(null=hits / max(kept, 1), usable_frac=kept / max(seen, 1),
                mean_opt=None, kept=kept)


def mean_optimal_count(grid, wall_p, n=2000, seed=1):
    rng = random.Random(seed)
    counts, i = [], 0
    while len(counts) < n and i < n * 40:
        env = GridWorld(n=grid, wall_p=wall_p, seed=seed * 7919 + i)
        i += 1
        opt = env.optimal_actions()
        if opt:
            counts.append(len(opt))
    return stats.mean(counts)


def n_required(p0, delta, alpha=0.05, power=0.80):
    """Two-proportion one-sided test: n PER GROUP to detect p1=p0+delta.
    Normal approximation; z_alpha one-sided, z_beta for the given power."""
    z_a = 1.645 if alpha == 0.05 else abs(_z(alpha))
    z_b = 0.842 if power == 0.80 else abs(_z(1 - power))
    p1 = min(p0 + delta, 0.999)
    pbar = (p0 + p1) / 2
    num = (z_a * math.sqrt(2 * pbar * (1 - pbar)) + z_b * math.sqrt(p0 * (1 - p0) + p1 * (1 - p1))) ** 2
    return math.ceil(num / delta ** 2)


def _z(p):  # inverse normal CDF (Acklam approx, good enough for alpha/beta)
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    pl = 0.02425
    if p < pl:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p <= 1 - pl:
        q = p - 0.5; r = q*q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)


def mc_power(p0, delta, n, trials=4000, alpha=0.05, seed=0):
    """Monte-Carlo check of the analytic n: simulate two Bernoulli samples of size n
    (null p0 vs alt p0+delta), one-sided two-proportion z-test, report empirical power."""
    rng = random.Random(seed)
    p1 = min(p0 + delta, 0.999)
    wins = 0
    for _ in range(trials):
        x0 = sum(rng.random() < p0 for _ in range(n))
        x1 = sum(rng.random() < p1 for _ in range(n))
        a, b = x1 / n, x0 / n
        pp = (x0 + x1) / (2 * n)
        se = math.sqrt(2 * pp * (1 - pp) / n) or 1e-9
        wins += int((a - b) / se > 1.645)
    return wins / trials


def main():
    print("=" * 76)
    print("POWER ANALYSIS (PAPER §10.3) — computed before any method is re-run")
    print("=" * 76)

    print("\n[A] Candidate benchmark designs: empirical random-policy null\n")
    print(f"{'design':<34}{'null':>8}{'usable%':>9}{'mean_opt':>10}")
    designs = [
        ("7x7 w=0.15  (pilot, ties kept)", 7, 0.15, False),
        ("7x7 w=0.15  UNIQUE-opt only",    7, 0.15, True),
        ("7x7 w=0.25  UNIQUE-opt only",    7, 0.25, True),
        ("9x9 w=0.20  UNIQUE-opt only",    9, 0.20, True),
        ("11x11 w=0.22 UNIQUE-opt only",  11, 0.22, True),
    ]
    results = {}
    for name, g, w, uniq in designs:
        r = measure_null(g, w, uniq, n=2500)
        mo = mean_optimal_count(g, w)
        results[name] = r
        print(f"{name:<34}{r['null']:>8.3f}{r['usable_frac']*100:>8.1f}%{mo:>10.2f}")

    # pick a UNIQUE-opt design (null ~ 0.25 by construction) with good yield
    chosen = max((d for d in designs if d[3]),
                 key=lambda d: results[d[0]]["usable_frac"])
    cname = chosen[0]
    p0 = results[cname]["null"]
    print(f"\n[B] Chosen design: {cname}")
    print(f"    empirical null p0 = {p0:.3f}  (unique-optimal-move => chance 0.25 by construction)")

    print("\n[C] Sample size to detect an effect (two-proportion, alpha=0.05, power=0.80)\n")
    print(f"{'target delta (p1-p0)':<24}{'n/group (analytic)':>20}{'MC power @n':>14}")
    plan = {}
    for delta in (0.10, 0.15, 0.20, 0.25):
        n = n_required(p0, delta)
        mc = mc_power(p0, delta, n)
        plan[delta] = (n, mc)
        print(f"{('+%.2f' % delta):<24}{n:>20}{mc:>14.2f}")

    # recommend a target and the resulting design
    d_star = 0.15
    n_star, _ = plan[d_star]
    seeds = 3
    per_seed = math.ceil(n_star / seeds)
    print(f"\n[D] RECOMMENDED POWERED DESIGN (target detectable delta = +{d_star:.2f}):")
    print(f"    null p0 = {p0:.3f};  n_total = {n_star} eval states;  "
          f"{seeds} seeds => {per_seed} states/seed")
    print(f"    vs the pilot's 24/seed at null 0.444 (SE~0.10, undetectable).")
    print(f"    Benchmark: {cname.split('  ')[0].strip()} with the UNIQUE-optimal-move filter.")
    print("\n[E] What this buys: chance is a well-defined 0.25, and a +0.15 improvement")
    print("    (0.25 -> 0.40) is detectable at 80% power. A null result then means")
    print("    'no effect >= 0.15', not 'benchmark could not tell'.")


if __name__ == "__main__":
    main()
