"""Algorithm 1: AC-JEPA training (PAPER §3.4, §5).

Multi-step rollout loss on the model's OWN predictions (exposure bias), scheduled
sampling from teacher forcing, VICReg anti-collapse, optional reward grounding.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .modules import (EMATarget, Encoder, TransitionEnsemble, ValueNet,
                      latent_variance, vicreg_losses)


def train_acjepa(data, cfg, device="cpu", log=print):
    O, A, R = data                                   # (N,H+1,3,g,g) (N,H) (N,H)
    N, Hp1 = O.shape[0], O.shape[1]
    H = min(cfg["horizon"], Hp1 - 1)
    grid = O.shape[-1]

    enc = Encoder(3, grid, cfg["d"]).to(device)
    ema = EMATarget(enc, cfg["tau"])
    ema.target.to(device)
    trans = TransitionEnsemble(cfg["d"], cfg["n_actions"], cfg["k"],
                               residual=cfg.get("residual", True)).to(device)
    val = ValueNet(cfg["d"]).to(device)
    opt = torch.optim.Adam(
        list(enc.parameters()) + list(trans.parameters()) + list(val.parameters()),
        lr=cfg["lr"])

    Ot = torch.tensor(O, device=device)
    At = torch.tensor(A, device=device)
    Rt = torch.tensor(R, device=device)

    for step in range(cfg["steps"]):
        idx = torch.randint(0, N, (cfg["batch"],), device=device)
        ob, ab, rb = Ot[idx], At[idx], Rt[idx]

        s = enc(ob[:, 0])                                    # (B,d) online encoder
        with torch.no_grad():                                # EMA targets, stop-grad
            tgt = torch.stack([ema(ob[:, h]) for h in range(1, H + 1)], 1)   # (B,H,d)

        # scheduled sampling: teacher forcing -> full rollout
        tf_p = max(0.0, 1.0 - step / (cfg["steps"] * cfg["tf_anneal"]))
        l_roll = torch.zeros((), device=device)
        s_hat = s
        for h in range(H):
            k = np.random.randint(cfg["k"])
            s_hat = trans(s_hat, ab[:, h], member=k)
            l_roll = l_roll + (cfg["lam"] ** h) * F.mse_loss(s_hat, tgt[:, h])
            if np.random.rand() < tf_p:                      # re-anchor on target
                s_hat = tgt[:, h]

        l_var, l_cov = vicreg_losses(s, cfg["gamma_v"])
        if not cfg.get("use_vicreg", True):                  # ablation A1
            l_var = l_cov = torch.zeros((), device=device)

        # --- reward + value grounding (§3.6) -----------------------------
        # BUGFIX (pilot v1): previously the reward head saw only step-0 rewards and
        # V_ψ had NO loss at all -- it stayed randomly initialised yet entered J(τ)
        # as the terminal bootstrap, so the planner optimised noise and scored at
        # chance. Ground R_ψ along the rollout and fit V_ψ to discounted returns.
        if cfg.get("use_reward", True):
            l_rew = torch.zeros((), device=device)
            s_r = s
            for h in range(H):                                # reward along the rollout
                k = np.random.randint(cfg["k"])
                s_r = trans(s_r, ab[:, h], member=k).detach()
                r_h, _ = val(s_r)
                l_rew = l_rew + F.mse_loss(r_h, rb[:, h])
            l_rew = l_rew / H

            g = cfg.get("gamma", 0.99)                        # Monte-Carlo returns
            disc = torch.tensor([g ** i for i in range(H)], device=device)
            returns = (rb[:, :H] * disc).flip(-1).cumsum(-1).flip(-1) / disc.clamp(min=1e-8)
            _, v_pred = val(s)
            l_val = F.mse_loss(v_pred, returns[:, 0])
        else:
            l_rew = l_val = torch.zeros((), device=device)

        loss = (l_roll + cfg["a_var"] * l_var + cfg["a_cov"] * l_cov
                + cfg["a_rew"] * l_rew + cfg.get("a_val", 1.0) * l_val)
        opt.zero_grad(); loss.backward(); opt.step()
        ema.update(enc)

        if step % max(1, cfg["steps"] // 8) == 0:
            log(f"  step {step:5d} | L={loss.item():.4f} roll={l_roll.item():.4f} "
                f"var={l_var.item():.4f} latent_std={latent_variance(s.detach()):.4f} tf_p={tf_p:.2f}")

    return enc, ema, trans, val
