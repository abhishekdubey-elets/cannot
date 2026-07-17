"""AC-JEPA world model: encoder, transition ensemble, losses, uncertainty, value.

Implements PAPER §3.1-3.6. Kept in one module for import simplicity; the paper's
file layout (encoder.py / transition.py / ...) maps 1:1 onto the classes here.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------- §3.1
class Encoder(nn.Module):
    """E_θ : o -> s. Small conv trunk; EMA target Ē is a detached copy."""

    def __init__(self, in_ch: int = 3, grid: int = 7, d: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.Flatten(), nn.Linear(32 * grid * grid, d),
        )

    def forward(self, o):                      # (B,3,H,W) -> (B,d)
        return self.net(o)


class EMATarget:
    """Ē_θ̄ : EMA of the online encoder. Never receives gradient (§3.1)."""

    def __init__(self, encoder: Encoder, tau: float = 0.996):
        self.target = copy.deepcopy(encoder)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.tau = tau

    @torch.no_grad()
    def update(self, online: Encoder):
        for pt, po in zip(self.target.parameters(), online.parameters()):
            pt.mul_(self.tau).add_(po.detach(), alpha=1 - self.tau)

    @torch.no_grad()
    def __call__(self, o):
        return self.target(o)


# --------------------------------------------------------------------- §3.3
class TransitionEnsemble(nn.Module):
    """{T_φk}: residual, action-conditioned. Residual form targets L≈1 (Prop. 1(ii))."""

    def __init__(self, d: int = 64, n_actions: int = 4, k: int = 5, hidden: int = 128,
                 residual: bool = True):
        super().__init__()
        self.k, self.residual = k, residual
        self.act_emb = nn.Embedding(n_actions, 16)
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(d + 16, hidden), nn.ReLU(),
                          nn.Linear(hidden, hidden), nn.ReLU(),
                          nn.Linear(hidden, d))
            for _ in range(k)
        ])

    def forward(self, s, a, member: int | None = None):
        """s:(B,d) a:(B,) -> (B,d) if member given, else (K,B,d)."""
        x = torch.cat([s, self.act_emb(a)], dim=-1)
        if member is not None:
            out = self.heads[member](x)
            return s + out if self.residual else out
        outs = torch.stack([h(x) for h in self.heads], 0)          # (K,B,d)
        return (s.unsqueeze(0) + outs) if self.residual else outs


# --------------------------------------------------------------------- §3.5
def ensemble_mean_and_uncertainty(preds: torch.Tensor):
    """preds:(K,B,d) -> mean:(B,d), u:(B,) epistemic disagreement."""
    mu = preds.mean(0)
    u = ((preds - mu.unsqueeze(0)) ** 2).sum(-1).mean(0)            # (B,)
    return mu, u


# --------------------------------------------------------------------- §3.6
class ValueNet(nn.Module):
    """R_ψ and V_ψ over latent states."""

    def __init__(self, d: int = 64, hidden: int = 128):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(d, hidden), nn.ReLU())
        self.r_head, self.v_head = nn.Linear(hidden, 1), nn.Linear(hidden, 1)

    def forward(self, s):
        h = self.body(s)
        return self.r_head(h).squeeze(-1), self.v_head(h).squeeze(-1)


# --------------------------------------------------------------------- §3.2.1
def vicreg_losses(s: torch.Tensor, gamma_v: float = 1.0, eps: float = 1e-4):
    """Anti-collapse. Without this the encoder may go constant, at which point
    L_pred = 0 and every trajectory scores identically -> planning is a no-op."""
    B, d = s.shape
    std = torch.sqrt(s.var(dim=0) + eps)
    l_var = F.relu(gamma_v - std).mean()
    sc = s - s.mean(dim=0, keepdim=True)
    cov = (sc.T @ sc) / max(B - 1, 1)
    off = cov - torch.diag_embed(torch.diagonal(cov))
    l_cov = (off ** 2).sum() / d
    return l_var, l_cov


def latent_variance(s: torch.Tensor) -> float:
    """Collapse diagnostic: mean per-dimension std. -> 0 means collapsed."""
    return s.std(dim=0).mean().item()
