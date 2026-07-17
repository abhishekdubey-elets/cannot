"""Latent Trajectory Tokens (PAPER §3.8) — the LLM <-> world-model bridge.

Upward: project a rollout summary into the LM embedding space so the model can
condition on a simulated future WITHOUT that future ever becoming text.
Downward: turn LLM action proposals into CEM/MCTS priors.

[IMPLEMENTED as the projection + interface; NOT yet validated against a real LM —
see PAPER §13.8. The pilot exercises the world model, not this bridge.]
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LatentTrajectoryTokens(nn.Module):
    """z(τ) = W([pool(ŝ_0..ŝ_H) ‖ J(τ) ‖ ū(τ)]) ∈ R^{m × d_llm}"""

    def __init__(self, d: int, d_llm: int = 2048, m: int = 4):
        super().__init__()
        self.m, self.d_llm = m, d_llm
        # pooled mean + last state + scalars (J, ū)
        self.proj = nn.Sequential(
            nn.Linear(2 * d + 2, 256), nn.ReLU(), nn.Linear(256, m * d_llm))

    def forward(self, traj: torch.Tensor, J: torch.Tensor, u: torch.Tensor):
        """traj:(B,H+1,d)  J:(B,)  u:(B,) -> (B,m,d_llm)"""
        feats = torch.cat([traj.mean(1), traj[:, -1], J.unsqueeze(-1), u.unsqueeze(-1)], -1)
        return self.proj(feats).view(-1, self.m, self.d_llm)


def proposals_to_priors(sequences: list[list[int]], horizon: int, n_actions: int,
                        strength: float = 2.0, device="cpu") -> torch.Tensor:
    """Downward path: LLM-proposed action sequences -> CEM prior logits (H,n_actions).

    The LLM supplies semantic plausibility; the world model supplies dynamical
    feasibility. Neither is trusted alone.
    """
    logits = torch.zeros(horizon, n_actions, device=device)
    for seq in sequences:
        for h, a in enumerate(seq[:horizon]):
            logits[h, a] += strength / max(len(sequences), 1)
    return logits
