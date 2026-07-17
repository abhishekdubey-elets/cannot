"""Self-contained gridworld (PAPER §7). No gym, no internet, no external deps.

Chosen deliberately: deterministic + discrete + tiny means BFS gives the *exact* optimal
plan, so planning accuracy is measured against ground truth rather than a proxy.
"""
from __future__ import annotations

import random
from collections import deque

import numpy as np

ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]   # up, down, left, right
N_ACTIONS = len(ACTIONS)


class GridWorld:
    """N x N grid with walls, an agent and a goal. Obs = 3-channel one-hot grid."""

    def __init__(self, n: int = 7, wall_p: float = 0.15, seed: int = 0):
        self.n, self.wall_p, self.rng = n, wall_p, random.Random(seed)
        self.reset()

    # -- core ------------------------------------------------------------- #
    def reset(self):
        n = self.n
        while True:
            self.walls = {(r, c) for r in range(n) for c in range(n)
                          if self.rng.random() < self.wall_p}
            free = [(r, c) for r in range(n) for c in range(n) if (r, c) not in self.walls]
            if len(free) < 4:
                continue
            self.agent, self.goal = self.rng.sample(free, 2)
            if self.bfs(self.agent, self.goal) is not None:
                break
        return self.obs()

    def obs(self) -> np.ndarray:
        g = np.zeros((3, self.n, self.n), dtype=np.float32)
        g[0][self.agent] = 1.0
        g[1][self.goal] = 1.0
        for w in self.walls:
            g[2][w] = 1.0
        return g

    def dist(self, pos=None):
        """BFS distance to goal (None if unreachable)."""
        p = self.bfs(pos or self.agent, self.goal)
        return None if p is None else len(p)

    def step(self, a: int, shaped: bool = True):
        """Potential-based shaping (Ng et al. 1999): r += Φ(s') − Φ(s), Φ = −dist.

        §10.1 measured that the raw reward (−0.01/step, +1 only on goal-contact) leaves
        J(τ) FLAT: R_ψ → −0.01 everywhere, so every trajectory ties and the planner
        argmaxes noise. Shaping is potential-based, so it does not change the optimal
        policy — it only gives the value head a gradient to learn.
        """
        d0 = self.dist()
        dr, dc = ACTIONS[a]
        nr, nc = self.agent[0] + dr, self.agent[1] + dc
        if 0 <= nr < self.n and 0 <= nc < self.n and (nr, nc) not in self.walls:
            self.agent = (nr, nc)
        done = self.agent == self.goal
        r = 1.0 if done else -0.01
        if shaped and not done and d0 is not None:
            d1 = self.dist()
            if d1 is not None:
                r += 0.1 * (d0 - d1)               # +0.1 toward goal, −0.1 away
        return self.obs(), r, done

    def optimal_actions(self) -> set[int]:
        """ALL actions on a shortest path (ties count). Fairer than BFS's single pick,
        and applied identically to PROSPECT and every LLM baseline."""
        d0 = self.dist()
        if d0 is None or d0 == 0:
            return set()
        out = set()
        for ai, (dr, dc) in enumerate(ACTIONS):
            nr, nc = self.agent[0] + dr, self.agent[1] + dc
            if not (0 <= nr < self.n and 0 <= nc < self.n) or (nr, nc) in self.walls:
                continue
            d1 = self.dist((nr, nc))
            if d1 is not None and d1 == d0 - 1:
                out.add(ai)
        return out

    def render_text(self) -> str:
        rows = []
        for r in range(self.n):
            row = ""
            for c in range(self.n):
                row += ("A" if (r, c) == self.agent else "G" if (r, c) == self.goal
                        else "#" if (r, c) in self.walls else ".")
            rows.append(row)
        return "\n".join(rows)

    # -- ground truth ----------------------------------------------------- #
    def bfs(self, start, goal):
        """Shortest action sequence start->goal, or None. Gives the optimal plan."""
        if start == goal:
            return []
        seen, q = {start}, deque([(start, [])])
        while q:
            pos, path = q.popleft()
            for ai, (dr, dc) in enumerate(ACTIONS):
                nxt = (pos[0] + dr, pos[1] + dc)
                if (not (0 <= nxt[0] < self.n and 0 <= nxt[1] < self.n)
                        or nxt in self.walls or nxt in seen):
                    continue
                if nxt == goal:
                    return path + [ai]
                seen.add(nxt)
                q.append((nxt, path + [ai]))
        return None

    def optimal_first_action(self):
        p = self.bfs(self.agent, self.goal)
        return p[0] if p else None


def collect(n_episodes: int = 400, horizon: int = 12, grid: int = 7, seed: int = 0,
            expert_frac: float = 0.5):
    """Random + expert-mixture rollouts. Mixture matters: pure-random data rarely
    reaches the goal, so reward/value heads see almost no positive signal."""
    rng = random.Random(seed)
    O, A, R = [], [], []
    for ep in range(n_episodes):
        env = GridWorld(n=grid, seed=seed * 7919 + ep)
        obs = [env.obs()]
        acts, rews = [], []
        for _ in range(horizon):
            if rng.random() < expert_frac and (a_opt := env.optimal_first_action()) is not None:
                a = a_opt
            else:
                a = rng.randrange(N_ACTIONS)
            o, r, done = env.step(a)
            obs.append(o); acts.append(a); rews.append(r)
            if done:
                break
        # pad to fixed horizon so batching is trivial
        while len(acts) < horizon:
            obs.append(obs[-1]); acts.append(0); rews.append(0.0)
        O.append(np.stack(obs[:horizon + 1])); A.append(np.array(acts[:horizon]))
        R.append(np.array(rews[:horizon], dtype=np.float32))
    return (np.stack(O), np.stack(A).astype(np.int64), np.stack(R).astype(np.float32))
