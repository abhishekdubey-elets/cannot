"""Pilot experiment (PAPER §10.1) — MEASURED, multi-seed, with ablations.

Tests only the INTERNAL claims the compute permits:
  C1 anti-collapse    : does VICReg keep latent variance > 0? (A1)
  C2 drift/gate       : does h-step latent error grow, and does ensemble uncertainty
                        track it? (Prop. 1 / §3.5)
  C3 latent planning  : does CEM-over-AC-JEPA beat random action selection at choosing
                        the BFS-optimal first action? (A9)
  C4 adaptive horizon : does the uncertainty gate beat a fixed horizon? (A3)

It does NOT test DreamerV3/MuZero parity — see PAPER §10.2 (hypothetical).
"""
from __future__ import annotations

import random
import statistics as stats
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from prospect.training.dataset import N_ACTIONS, GridWorld, collect          # noqa: E402
from prospect.world_model.modules import latent_variance                      # noqa: E402
from prospect.world_model.rollout import cem_plan, counterfactual, rollout    # noqa: E402
from prospect.world_model.trainer import train_acjepa                         # noqa: E402

CFG = dict(d=64, k=5, n_actions=N_ACTIONS, lr=1e-3, steps=1500, batch=128,
           horizon=8, lam=0.9, tau=0.996, gamma_v=1.0, a_var=25.0, a_cov=1.0,
           a_rew=1.0, tf_anneal=0.5, use_vicreg=True, use_reward=True, residual=True)


def h_step_error(enc, ema, trans, device, grid=7, n=64, H=8, seed=0):
    """C2: latent rollout error and ensemble uncertainty, per horizon step."""
    errs = [[] for _ in range(H)]
    uncs = [[] for _ in range(H)]
    rng = random.Random(seed)
    for i in range(n):
        env = GridWorld(n=grid, seed=seed * 104729 + i)
        obs = [env.obs()]
        acts = []
        for _ in range(H):
            a = rng.randrange(N_ACTIONS)
            o, _, d = env.step(a)
            obs.append(o); acts.append(a)
            if d:
                break
        while len(acts) < H:
            obs.append(obs[-1]); acts.append(0)
        ob = torch.tensor(np.stack(obs[:H + 1]), device=device).unsqueeze(0)
        ab = torch.tensor(np.array(acts[:H]), device=device).unsqueeze(0)
        with torch.no_grad():
            s = enc(ob[:, 0])
            for h in range(H):
                preds = trans(s, ab[:, h])
                mu = preds.mean(0)
                u = ((preds - mu.unsqueeze(0)) ** 2).sum(-1).mean(0)
                tgt = ema(ob[:, h + 1])
                errs[h].append((mu - tgt).norm(dim=-1).item())
                uncs[h].append(u.item())
                s = mu
    return [stats.mean(e) for e in errs], [stats.mean(u) for u in uncs]


def planning_accuracy(enc, trans, val, device, grid=7, n=60, beta=float("inf"),
                      seed=0, mode="cem", alpha=1.0, horizon=6):
    """C3/C4: fraction of states where the chosen first action == BFS-optimal."""
    rng = random.Random(seed + 555)
    hit = 0
    total = 0
    for i in range(n):
        env = GridWorld(n=grid, seed=seed * 7907 + i + 10_000)
        opt = env.optimal_first_action()
        if opt is None:
            continue
        total += 1
        if mode == "random":
            a = rng.randrange(N_ACTIONS)
        else:
            with torch.no_grad():
                s0 = enc(torch.tensor(env.obs(), device=device).unsqueeze(0))
            seq, _ = cem_plan(trans, val, s0[0], N_ACTIONS, horizon=horizon, pop=192,
                              iters=4, elites=24, beta=beta, device=device, alpha=alpha)
            a = seq[0].item()
        hit += int(a == opt)
    return hit / max(total, 1)


def run_condition(name, cfg, seed, device, log):
    log(f"\n--- {name} | seed {seed} ---")
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    data = collect(n_episodes=400, horizon=cfg["horizon"], grid=7, seed=seed)
    enc, ema, trans, val = train_acjepa(data, cfg, device=device, log=log)

    with torch.no_grad():
        probe = torch.tensor(collect(64, cfg["horizon"], 7, seed + 99)[0][:, 0], device=device)
        lv = latent_variance(enc(probe))
    errs, uncs = h_step_error(enc, ema, trans, device, seed=seed)
    return dict(name=name, seed=seed, latent_std=lv, errs=errs, uncs=uncs,
                enc=enc, trans=trans, val=val)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log = print
    log(f"device={device}")
    SEEDS = [0, 1, 2]
    rows = []

    for seed in SEEDS:
        full = run_condition("full", dict(CFG), seed, device, log)
        # C3 / C4 / A9
        pa_gate = planning_accuracy(full["enc"], full["trans"], full["val"], device,
                                    beta=float(np.mean(full["uncs"][:4]) * 4), seed=seed)
        pa_fixed = planning_accuracy(full["enc"], full["trans"], full["val"], device,
                                     beta=float("inf"), seed=seed)
        pa_rand = planning_accuracy(full["enc"], full["trans"], full["val"], device,
                                    seed=seed, mode="random")
        pa_a0 = planning_accuracy(full["enc"], full["trans"], full["val"], device,
                                  beta=float("inf"), seed=seed, alpha=0.0, horizon=10)
        # A1: no VICReg
        cfg_a1 = dict(CFG); cfg_a1["use_vicreg"] = False
        a1 = run_condition("A1 (-VICReg)", cfg_a1, seed, device, log)
        pa_a1 = planning_accuracy(a1["enc"], a1["trans"], a1["val"], device, seed=seed)

        rows.append(dict(seed=seed, lv_full=full["latent_std"], lv_a1=a1["latent_std"],
                         pa_gate=pa_gate, pa_fixed=pa_fixed, pa_rand=pa_rand, pa_a1=pa_a1,
                         pa_a0=pa_a0,
                         errs=full["errs"], uncs=full["uncs"]))
        log(f"  seed {seed}: latent_std full={full['latent_std']:.3f} A1={a1['latent_std']:.4f} "
            f"| plan acc gate={pa_gate:.3f} fixed={pa_fixed:.3f} rand={pa_rand:.3f} A1={pa_a1:.3f}")

    # ---------------- report ----------------
    print("\n" + "=" * 78)
    print("PILOT RESULTS — MEASURED (PAPER §10.1)")
    print("=" * 78)
    print(f"{'seed':>5}{'latent_std':>12}{'A1 std':>9}{'plan:gate':>11}{'plan:fixed':>12}"
          f"{'plan:rand':>11}{'plan:A1':>9}")
    for r in rows:
        print(f"{r['seed']:>5}{r['lv_full']:>12.3f}{r['lv_a1']:>9.4f}{r['pa_gate']:>11.3f}"
              f"{r['pa_fixed']:>12.3f}{r['pa_rand']:>11.3f}{r['pa_a1']:>9.3f}")

    def m(key):
        return stats.mean(r[key] for r in rows), (stats.stdev([r[key] for r in rows])
                                                  if len(rows) > 1 else 0.0)
    print("\nC1 anti-collapse (A1):  latent_std full=%.3f±%.3f  vs  -VICReg=%.4f±%.4f"
          % (*m("lv_full"), *m("lv_a1")))
    print("C3 latent planning:     gate=%.3f±%.3f  random=%.3f±%.3f  (chance=0.25)"
          % (*m("pa_gate"), *m("pa_rand")))
    print("C4 adaptive horizon:    gate=%.3f±%.3f  fixed-H=%.3f±%.3f" % (*m("pa_gate"), *m("pa_fixed")))
    print("A1 planning w/o VICReg: %.3f±%.3f" % m("pa_a1"))
    print("DIAG alpha=0 (no pessimism), H=10: %.3f±%.3f  <- tests whether the uncertainty"
          % m("pa_a0"))
    print("     penalty was swamping the sparse reward in J (alpha=1, u~0.05 vs r~-0.01)")

    print("\nC2 drift & uncertainty (mean over seeds), Prop. 1 / §3.5:")
    H = len(rows[0]["errs"])
    print(f"{'h':>3}{'latent err':>12}{'ensemble u':>12}")
    for h in range(H):
        e = stats.mean(r["errs"][h] for r in rows)
        u = stats.mean(r["uncs"][h] for r in rows)
        print(f"{h+1:>3}{e:>12.4f}{u:>12.4f}")
    ecol = [stats.mean(r["errs"][h] for r in rows) for h in range(H)]
    ucol = [stats.mean(r["uncs"][h] for r in rows) for h in range(H)]
    if len(ecol) > 2:
        ex, ux = np.array(ecol), np.array(ucol)
        corr = float(np.corrcoef(ex, ux)[0, 1]) if ux.std() > 0 else float("nan")
        print(f"\ncorr(latent error, ensemble uncertainty) over horizon = {corr:+.3f}")
        print("  -> gate is informative iff this is strongly positive (§3.5)")

    g, gs = m("pa_gate"); r_, rs = m("pa_rand")
    print("\nVERDICT (pre-registered):")
    print(f"  C1 anti-collapse: {'PASS' if m('lv_full')[0] > 4 * max(m('lv_a1')[0], 1e-6) else 'FAIL'}")
    print(f"  C3 planning>random: {'PASS' if g > r_ + max(0.10, 2 * (gs + rs)) else 'FAIL'}"
          f"  ({g:.3f} vs {r_:.3f})")


if __name__ == "__main__":
    main()
