"""Continual-learning metrics (EXPERIMENT_PLAN.md §6).

Computed from the accuracy matrix R[(i, j)] = accuracy on family j's novel
retention set after night i (i = 0 is the pre-training baseline). Definitions
follow Lopez-Paz & Ranzato (2017, GEM):

  ACC  = mean_j R[N, j]                         end-of-stream competence
  BWT  = mean_{j<N} (R[N, j] - R[j, j])         negative => forgetting
  FWT  = mean_{j>=1} (R[j-1, j] - R[0, j])      does prior learning help new tasks
  NIST = mean_j R[N, j] on NOVEL instances      skill transfer (== ACC here)
  exact_recall = mean_j accuracy on REPEAT instances at night N
  GCR  = G[N] - G[0]                            general-capability retention
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Metrics:
    acc: float
    bwt: float
    fwt: float
    nist: float
    exact_recall: float
    gcr: float
    online_cost: float
    acc_per_cost: float
    route_mix: dict


def _mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def compute(R: dict, n_families: int, g_curve: list[float],
            exact_recall: float, online_cost: float,
            route_mix: dict) -> Metrics:
    N = n_families
    acc = _mean(R[(N, j)] for j in range(1, N + 1))
    bwt = _mean(R[(N, j)] - R[(j, j)] for j in range(1, N))       # exclude last
    fwt = _mean(R[(j - 1, j)] - R[(0, j)] for j in range(1, N + 1))
    gcr = g_curve[-1] - g_curve[0]
    # normalize acc_per_cost against a 1-unit/task reference so it reads like a
    # relative efficiency multiplier (>1 = cheaper-per-correct than baseline S2).
    acc_per_cost = (acc / online_cost) if online_cost else 0.0
    return Metrics(round(acc, 3), round(bwt, 3), round(fwt, 3), round(acc, 3),
                   round(exact_recall, 3), round(gcr, 3), round(online_cost, 1),
                   round(acc_per_cost, 4), route_mix)
