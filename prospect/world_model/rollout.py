"""Algorithm 2 (uncertainty-gated latent rollout) + Algorithm 3 (CEM planner)
+ Algorithm 5 (latent counterfactuals). PAPER §3.5-3.7, §3.9.

Rollouts are batched across the whole CEM population: planning is a handful of
tensor ops, not a Python loop over candidates.
"""
from __future__ import annotations

import torch

from .modules import ensemble_mean_and_uncertainty


@torch.no_grad()
def rollout(trans, value, s0: torch.Tensor, actions: torch.Tensor,
            beta: float = float("inf"), gamma: float = 0.99, alpha: float = 1.0):
    """Algorithm 2, batched.

    s0:(B,d)  actions:(B,H)  beta: uncertainty budget (inf = fixed-horizon ablation A3)
    Returns J:(B,), H_eff:(B,), traj:(B,H+1,d), U:(B,)
    """
    B, H = actions.shape
    s = s0
    alive = torch.ones(B, dtype=torch.bool, device=s0.device)
    U = torch.zeros(B, device=s0.device)
    J = torch.zeros(B, device=s0.device)
    H_eff = torch.zeros(B, device=s0.device)
    traj = [s0]

    for h in range(H):
        preds = trans(s, actions[:, h])                    # (K,B,d)
        mu, u = ensemble_mean_and_uncertainty(preds)

        # ADAPTIVE HALT (§3.5): stop a lane once its cumulative uncertainty exceeds beta.
        would = U + u
        cont = alive & (would <= beta)
        if not cont.any():
            break

        r, _ = value(mu)
        J = J + torch.where(cont, (gamma ** h) * (r - alpha * u), torch.zeros_like(J))
        U = torch.where(cont, would, U)
        H_eff = H_eff + cont.float()
        s = torch.where(cont.unsqueeze(-1), mu, s)         # frozen lanes stop advancing
        alive = cont
        traj.append(s)

    _, v = value(s)                                        # terminal bootstrap (§3.6)
    J = J + (gamma ** H_eff) * v
    return J, H_eff, torch.stack(traj, 1), U


@torch.no_grad()
def cem_plan(trans, value, s0: torch.Tensor, n_actions: int, horizon: int = 8,
             pop: int = 256, iters: int = 4, elites: int = 32, beta: float = float("inf"),
             priors: torch.Tensor | None = None, device="cpu", alpha: float = 1.0):
    """Algorithm 3: CEM over discrete action sequences via per-step categorical logits.

    priors: optional (H,n_actions) logits from an LLM (§3.8 downward path).
    Returns best action sequence (H,) and its score.
    """
    logits = (priors.clone() if priors is not None
              else torch.zeros(horizon, n_actions, device=device))
    s0b = s0.expand(pop, -1)

    for _ in range(iters):
        probs = torch.softmax(logits, -1).unsqueeze(0).expand(pop, -1, -1)
        acts = torch.distributions.Categorical(probs=probs).sample()      # (pop,H)
        J, _, _, _ = rollout(trans, value, s0b, acts, beta=beta, alpha=alpha)
        top = J.topk(min(elites, pop)).indices
        elite = acts[top]                                                 # (M,H)
        counts = torch.zeros(horizon, n_actions, device=device)
        for h in range(horizon):
            counts[h] = torch.bincount(elite[:, h], minlength=n_actions).float()
        logits = torch.log(counts / counts.sum(-1, keepdim=True).clamp(min=1) + 1e-6)

    best = logits.argmax(-1)                                              # (H,)
    J, _, _, _ = rollout(trans, value, s0.unsqueeze(0), best.unsqueeze(0), beta=beta, alpha=alpha)
    return best, J.item()


@torch.no_grad()
def counterfactual(trans, value, traj_states: torch.Tensor, actions: torch.Tensor,
                   k: int, n_actions: int, beta: float = float("inf")):
    """Algorithm 5: do(a_k := a') for every a', reusing s_k from the factual rollout
    (minimal intervention: only descendants are re-simulated). Costs |A| MLP passes."""
    s_k = traj_states[:, k]                                   # (1,d) factual state at k
    J_fact, _, _, _ = rollout(trans, value, traj_states[:, 0], actions, beta=beta)
    out = {}
    for a_prime in range(n_actions):
        alt = actions.clone()
        alt[:, k] = a_prime
        # re-roll only from step k onward, reusing the factual prefix state
        J_cf, _, _, _ = rollout(trans, value, s_k, alt[:, k:], beta=beta)
        out[a_prime] = (J_cf - J_fact).item()
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))
