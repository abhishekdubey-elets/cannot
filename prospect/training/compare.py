"""PAPER §10.2 — MEASURED comparison: latent simulation vs linguistic simulation.

The paper's central hypothesis is that latent rollouts beat reasoning-in-language for
planning. This runs that comparison for real, on identical grids, with an identical
metric (is the chosen first move on a shortest path? — ties counted, applied to every
method equally), on one Tesla T4.

RUN     : random | Qwen2.5-3B direct | +CoT | +ToT(k=3) | PROSPECT (latent CEM)
NOT RUN : DreamerV3, MuZero (faithful reimplementation is weeks of work; a same-day
          version would be a strawman -> a rigged win). GPT-4 (no API access).
          These stay marked NOT RUN in the paper rather than fabricated.
"""
from __future__ import annotations

import os
import random
import statistics as stats
import sys
import time

import numpy as np
import torch

sys.path.insert(0, ".")
from prospect.llm.lm_baselines import QwenLM, cot, direct, tot            # noqa: E402
from prospect.training.dataset import N_ACTIONS, GridWorld, collect       # noqa: E402
from prospect.world_model.rollout import cem_plan                         # noqa: E402
from prospect.world_model.trainer import train_acjepa                     # noqa: E402

SEEDS = [0, 1, 2]
N_EVAL = 24
CFG = dict(d=64, k=5, n_actions=N_ACTIONS, lr=1e-3, steps=1500, batch=128,
           horizon=8, lam=0.9, tau=0.996, gamma_v=1.0, a_var=25.0, a_cov=1.0,
           a_rew=1.0, a_val=1.0, gamma=0.99, tf_anneal=0.5,
           use_vicreg=True, use_reward=True, residual=True)


def eval_grids(seed, n=N_EVAL, grid=7):
    out = []
    i = 0
    while len(out) < n:
        env = GridWorld(n=grid, seed=seed * 7907 + i + 10_000)
        i += 1
        if env.optimal_actions():
            out.append(env)
    return out


def find_model():
    for dp, dns, fns in os.walk("/kaggle/input"):
        if "config.json" in fns and any(f.startswith("model") and
                                        f.endswith((".safetensors", ".bin")) for f in fns):
            if "3b-instruct" in dp.lower():
                return dp
    return None


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}", flush=True)
    mp = find_model()
    print(f"qwen: {mp}", flush=True)
    lm = QwenLM(mp, device=device) if mp else None

    rows = {m: [] for m in ["random", "direct", "cot", "tot", "prospect"]}
    lat = {m: [] for m in rows}

    for seed in SEEDS:
        print(f"\n{'#'*70}\n# SEED {seed}\n{'#'*70}", flush=True)
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        envs = eval_grids(seed)
        rng = random.Random(seed + 555)

        # ---- PROSPECT: train AC-JEPA on SHAPED reward, then plan in latent -------
        t0 = time.time()
        data = collect(n_episodes=400, horizon=CFG["horizon"], grid=7, seed=seed)
        enc, ema, trans, val = train_acjepa(data, CFG, device=device,
                                            log=lambda *a: None)
        print(f"  AC-JEPA trained ({time.time()-t0:.0f}s)", flush=True)

        hits, lats = 0, []
        for env in envs:
            opt = env.optimal_actions()
            t1 = time.time()
            with torch.no_grad():
                s0 = enc(torch.tensor(env.obs(), device=device).unsqueeze(0))
            seq, _ = cem_plan(trans, val, s0[0], N_ACTIONS, horizon=8, pop=192,
                              iters=4, elites=24, device=device, alpha=0.0)
            lats.append(time.time() - t1)
            hits += int(seq[0].item() in opt)
        rows["prospect"].append(hits / len(envs)); lat["prospect"].append(stats.mean(lats))
        print(f"  prospect: {hits/len(envs):.3f}", flush=True)

        # ---- random ------------------------------------------------------------
        hits = sum(int(rng.randrange(N_ACTIONS) in env.optimal_actions()) for env in envs)
        rows["random"].append(hits / len(envs)); lat["random"].append(0.0)
        print(f"  random:   {hits/len(envs):.3f}", flush=True)

        # ---- LLM baselines -----------------------------------------------------
        if lm is None:
            continue
        for name, fn in [("direct", direct), ("cot", cot), ("tot", tot)]:
            hits, lats = 0, []
            for env in envs:
                mv, dt = fn(lm, env)
                lats.append(dt)
                hits += int(mv is not None and mv in env.optimal_actions())
            rows[name].append(hits / len(envs)); lat[name].append(stats.mean(lats))
            print(f"  {name}: {hits/len(envs):.3f}  ({stats.mean(lats):.2f}s/decision)",
                  flush=True)

    # ------------------------------ report ------------------------------------
    print("\n" + "=" * 78)
    print("TABLE §10.2 — MEASURED: latent vs linguistic simulation")
    print(f"(Qwen2.5-3B-Instruct, 7x7 gridworld, {N_EVAL} states x {len(SEEDS)} seeds,")
    print(" metric = chosen first move lies on a shortest path; ties counted)")
    print("=" * 78)
    print(f"{'method':<12}{'plan acc':>12}{'±sd':>8}{'latency/dec':>14}{'rel. lat':>10}")
    base_lat = stats.mean(lat["direct"]) if lat["direct"] else 1.0
    for m in ["random", "direct", "cot", "tot", "prospect"]:
        if not rows[m]:
            print(f"{m:<12}{'(no model)':>12}")
            continue
        mu = stats.mean(rows[m]); sd = stats.stdev(rows[m]) if len(rows[m]) > 1 else 0.0
        L = stats.mean(lat[m])
        rel = (L / base_lat) if base_lat > 0 else 0.0
        print(f"{m:<12}{mu:>12.3f}{sd:>8.3f}{L:>13.3f}s{rel:>9.1f}x")
    print(f"\n{'DreamerV3':<12}{'NOT RUN':>12}   faithful reimpl. = weeks; same-day = strawman")
    print(f"{'MuZero':<12}{'NOT RUN':>12}   same")
    print(f"{'GPT-4':<12}{'NOT RUN':>12}   no API access")

    if rows["cot"] and rows["prospect"]:
        p, c = stats.mean(rows["prospect"]), stats.mean(rows["cot"])
        r = stats.mean(rows["random"])
        print("\nVERDICT (pre-registered):")
        print(f"  PROSPECT > random?  {'YES' if p > r + 0.10 else 'NO'}  ({p:.3f} vs {r:.3f})")
        print(f"  PROSPECT > CoT?     {'YES' if p > c + 0.10 else 'NO'}  ({p:.3f} vs {c:.3f})")
        print(f"  CoT > random?       {'YES' if c > r + 0.10 else 'NO'}  ({c:.3f} vs {r:.3f})")
        if p <= r + 0.10:
            print("  -> central hypothesis NOT supported: latent planning fails to beat chance,")
            print("     so it cannot be claimed superior to linguistic reasoning.")


if __name__ == "__main__":
    main()
