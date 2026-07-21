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
CFG = dict(d=64, k=5, n_actions=N_ACTIONS, lr=1e-3, steps=1500, batch=128,
           horizon=8, lam=0.9, tau=0.996, gamma_v=1.0, a_var=25.0, a_cov=1.0,
           a_rew=1.0, a_val=1.0, gamma=0.99, tf_anneal=0.5,
           use_vicreg=True, use_reward=True, residual=True)

# --powered => §10.3 spec: unique-optimal-move filter (null=0.25) + n from power analysis.
POWERED = "--powered" in sys.argv
N_EVAL = 40 if POWERED else 24
WALL_P = 0.25 if POWERED else 0.15
UNIQUE_ONLY = POWERED             # exactly one optimal move -> chance = 0.25 by construction
NULL_P0 = 0.25 if POWERED else 0.44
DELTA = 0.15                      # pre-registered detectable effect


def eval_grids(seed, n=N_EVAL, grid=7):
    """Powered mode keeps only states with EXACTLY ONE optimal move (well-defined 0.25 null)."""
    out = []
    i = 0
    while len(out) < n and i < n * 200:
        env = GridWorld(n=grid, wall_p=WALL_P, seed=seed * 7907 + i + 10_000)
        i += 1
        opt = env.optimal_actions()
        if opt and (not UNIQUE_ONLY or len(opt) == 1):
            out.append(env)
    return out


class Tee:
    """Kaggle did not capture this script's stdout on a prior run (0-byte log) while
    output FILES downloaded fine. Write results to a file as the source of truth.

    Two bugs were fixed here: (1) it must accept print's kwargs (flush=, sep=, end=)
    since it replaces the global print; (2) it must call builtins.print, not the
    module-global print -- which is itself once patched (infinite recursion).
    """

    def __init__(self, path="/kaggle/working/results.txt"):
        import builtins
        self._print = builtins.print
        self.f = open(path, "w", encoding="utf-8") if os.path.isdir("/kaggle/working") else None

    def __call__(self, *a, **kw):
        msg = kw.get("sep", " ").join(str(x) for x in a)
        self._print(msg, flush=True)
        if self.f:
            self.f.write(msg + kw.get("end", "\n")); self.f.flush()


def find_model(log=print):
    """Locate a mounted causal-LM. Prior run silently found nothing and skipped every
    LLM baseline; now we log the mounts and fail loudly instead."""
    cands = []
    for dp, dns, fns in os.walk("/kaggle/input"):
        if "config.json" in fns and any(f.endswith((".safetensors", ".bin")) for f in fns):
            cands.append(dp)
    log(f"model candidates: {cands}")
    for c in cands:
        if "3b-instruct" in c.lower():
            return c
    return cands[0] if cands else None


def main():
    P = Tee()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    P(f"device={device}")
    P(f"mounts={os.listdir('/kaggle/input') if os.path.isdir('/kaggle/input') else 'none'}")
    mp = find_model(P)
    P(f"qwen: {mp}")
    if mp is None:
        P("!! NO MODEL MOUNTED -> LLM baselines will be SKIPPED (this is why the prior "
          "run finished in 8 min). Check model_sources in kernel-metadata.json.")
    lm = QwenLM(mp, device=device) if mp else None
    globals()["print"] = P          # route the rest of the report into the file too

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
    tag = "§10.4 POWERED" if POWERED else "§10.2 PILOT (underpowered)"
    print(f"TABLE {tag} — MEASURED: latent vs linguistic simulation")
    print(f"(Qwen2.5-3B, 7x7 w={WALL_P}{' UNIQUE-opt' if UNIQUE_ONLY else ' ties-kept'}, "
          f"{N_EVAL} states x {len(SEEDS)} seeds, null p0={NULL_P0:.2f})")
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

    def ci(vals):
        mu = stats.mean(vals)
        se = (stats.stdev(vals) / len(vals) ** 0.5) if len(vals) > 1 else 0.0
        return mu, 1.96 * se

    if rows["cot"] and rows["prospect"]:
        p, pe = ci(rows["prospect"]); c, ce = ci(rows["cot"]); r, re_ = ci(rows["random"])
        d, de = ci(rows["direct"]); t, te = ci(rows["tot"])
        print(f"\n95% CIs:  random {r:.3f}±{re_:.3f}  direct {d:.3f}±{de:.3f}  "
              f"cot {c:.3f}±{ce:.3f}  tot {t:.3f}±{te:.3f}  prospect {p:.3f}±{pe:.3f}")
        print(f"\nVERDICT (pre-registered, null={NULL_P0:.2f}, detectable delta={DELTA:.2f}):")
        best_llm = max(c, d, t)
        print(f"  PROSPECT beats null by >= {DELTA}?  "
              f"{'YES' if p - NULL_P0 >= DELTA else 'NO'}  (Δ={p-NULL_P0:+.3f})")
        print(f"  best-LLM beats null by >= {DELTA}?  "
              f"{'YES' if best_llm - NULL_P0 >= DELTA else 'NO'}  (Δ={best_llm-NULL_P0:+.3f})")
        print(f"  PROSPECT vs CoT CIs overlap?      "
              f"{'YES (indistinguishable)' if abs(p-c) < pe+ce else 'NO'}")
        if not POWERED:
            print("  NOTE: pilot is UNDERPOWERED (§10.2.1); accuracy verdicts are not valid here.")
        elif p - NULL_P0 < DELTA and best_llm - NULL_P0 < DELTA:
            print("  -> On a PROPERLY POWERED benchmark, NO method (latent or linguistic) shows a")
            print("     detectable planning effect. This is a valid null: the effect, if any, is")
            print("     < 0.15 — not a measurement artifact.")
        elif p - NULL_P0 >= DELTA and p - c > pe + ce:
            print("  -> central hypothesis SUPPORTED: latent planning beats null AND CoT, powered.")


if __name__ == "__main__":
    main()
