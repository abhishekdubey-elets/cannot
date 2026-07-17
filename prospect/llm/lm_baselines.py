"""Linguistic-simulation baselines: direct, Chain-of-Thought, Tree-of-Thoughts.

These are the baselines PROSPECT's central hypothesis is actually about ("latent
simulation beats reasoning in language"). All methods face the IDENTICAL task, the
IDENTICAL grids, and are scored on the IDENTICAL metric (is the chosen first move on
a shortest path?), so the comparison is apples-to-apples.

NOT included, deliberately: DreamerV3 / MuZero (a same-day reimplementation would be a
strawman baseline, i.e. a rigged win) and GPT-4 (no API access). See PAPER §10.2.
"""
from __future__ import annotations

import re
import time

import torch

WORDS = {"up": 0, "down": 1, "left": 2, "right": 3}
SYS = ("You are a precise pathfinding solver. Answer only what is asked.")


def render_prompt(env) -> str:
    return (
        f"Grid {env.n}x{env.n}. '#'=wall, 'A'=you, 'G'=goal, '.'=free.\n"
        f"{env.render_text()}\n"
        "Row 0 is the top; column 0 is the left.\n"
        "Moves: up=row-1, down=row+1, left=col-1, right=col+1.\n"
    )


def parse_move(text: str):
    """Last mentioned direction word wins (reasoning ends with the answer)."""
    hits = re.findall(r"\b(up|down|left|right)\b", text.lower())
    return WORDS[hits[-1]] if hits else None


class QwenLM:
    """Inference-only wrapper over the Kaggle-mounted Qwen2.5-3B-Instruct."""

    def __init__(self, path: str, device: str = "cuda"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(path)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        dt = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
        kw = dict(device_map={"": 0} if device == "cuda" else "cpu")
        try:
            self.m = AutoModelForCausalLM.from_pretrained(path, dtype=dt, **kw)
        except TypeError:
            self.m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dt, **kw)
        self.m.eval()

    @torch.no_grad()
    def gen(self, prompt: str, max_new_tokens: int = 8, temperature: float = 0.0) -> str:
        msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": prompt}]
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = self.tok(text, return_tensors="pt").to(self.m.device)
        out = self.m.generate(**ids, max_new_tokens=max_new_tokens,
                              do_sample=temperature > 0,
                              temperature=temperature if temperature > 0 else None,
                              pad_token_id=self.tok.pad_token_id)
        return self.tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)


# ------------------------------------------------------------------ baselines
def direct(lm: QwenLM, env):
    p = render_prompt(env) + "Which single move starts a shortest path from A to G? " \
                             "Reply with exactly one word: up, down, left, or right."
    t0 = time.time()
    txt = lm.gen(p, max_new_tokens=6)
    return parse_move(txt), time.time() - t0


def cot(lm: QwenLM, env):
    p = render_prompt(env) + ("Think step by step: locate A and G, trace a shortest path "
                              "around the walls, then state the FIRST move.\n"
                              "End your reply with exactly one word on the last line: "
                              "up, down, left, or right.")
    t0 = time.time()
    txt = lm.gen(p, max_new_tokens=320)
    return parse_move(txt), time.time() - t0


def tot(lm: QwenLM, env, k: int = 3):
    """Simplified Tree-of-Thoughts: sample k candidate branches, then self-evaluate."""
    t0 = time.time()
    base = render_prompt(env)
    cands = []
    for _ in range(k):
        txt = lm.gen(base + "Propose ONE candidate first move and justify it in one "
                            "sentence. End with the move as a single word.",
                     max_new_tokens=90, temperature=0.8)
        mv = parse_move(txt)
        if mv is not None:
            cands.append((mv, txt.strip().replace("\n", " ")[:160]))
    if not cands:
        return None, time.time() - t0
    opts = "\n".join(f"{i+1}. move={list(WORDS)[m]} | {j}" for i, (m, j) in enumerate(cands))
    pick = lm.gen(base + "Candidate first moves:\n" + opts +
                  "\n\nEvaluate which candidate is truly on a shortest path. "
                  "End with the chosen move as a single word.", max_new_tokens=140)
    mv = parse_move(pick)
    return (mv if mv is not None else cands[0][0]), time.time() - t0
