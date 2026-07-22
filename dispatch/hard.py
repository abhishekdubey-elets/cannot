"""HARD workload — tasks that require capabilities Chain-of-Thought STRUCTURALLY lacks.

The easy workload (workload.py) showed CoT is a universal solvent when every fact is
in-context and computation is small (measured ceiling 0.016 -> NO-GO). This workload
adds two tasks CoT cannot do by reasoning alone:

  factual   -> FAST     (in weights; fast is cheapest, CoT ties)
  word      -> COT      (multi-step mapping, small numbers)
  big_arith -> TOOL     (7-digit x 7-digit; CoT cannot compute this in-head)
  kb_lookup -> RETRIEVE (arbitrary fact ONLY in an out-of-context knowledge base;
                         not in the weights, not in the prompt)

Mechanisms differ in CAPABILITY, not just prompt style:
  fast/cot : model only.   tool : exact calculator.   retrieve : KB lookup (idealized).

If the MEASURED dispatch ceiling here is still < 0.10, the null against cognitive
architectures is strong. If it opens up, a controller has real headroom and we build it.
"""
from __future__ import annotations

import ast
import operator
import random
import re
from dataclasses import dataclass, field

TYPES = ["factual", "word", "big_arith", "kb_lookup"]
MECHS = ["fast", "cot", "tool", "retrieve"]

CAPITALS = {"France": "Paris", "Japan": "Tokyo", "Egypt": "Cairo", "Canada": "Ottawa",
            "Norway": "Oslo", "Kenya": "Nairobi", "Peru": "Lima", "Greece": "Athens",
            "Poland": "Warsaw", "Chile": "Santiago", "Sweden": "Stockholm", "Cuba": "Havana",
            "Ireland": "Dublin", "Hungary": "Budapest", "Nepal": "Kathmandu", "Ghana": "Accra"}
_ATTRS = ["activation_code", "shard_count", "resonance_level"]


@dataclass(frozen=True)
class Query:
    qtype: str
    prompt: str
    answer: str
    best_mechanism: str
    meta: dict = field(default_factory=dict)


def make_hard(seed: int, n_per_type: int):
    """Returns (queries, kb). The KB is the out-of-context knowledge base that only the
    retrieve mechanism can access; it is never shown in any prompt."""
    rng = random.Random(seed)
    kb: dict[str, dict[str, int]] = {}
    qs: list[Query] = []

    for _ in range(n_per_type):
        c = rng.choice(list(CAPITALS))
        qs.append(Query("factual", f"What is the capital of {c}?", CAPITALS[c], "fast"))

    for _ in range(n_per_type):
        a, b, c = rng.randint(2, 9), rng.randint(2, 9), rng.randint(1, 9)
        qs.append(Query("word",
                        f"A store has {a} shelves, each holding {b} boxes. {c} boxes are "
                        f"removed, then the number remaining is doubled. How many boxes now?",
                        str(((a * b) - c) * 2), "cot"))

    for _ in range(n_per_type):
        a, b = rng.randint(1_000_000, 9_999_999), rng.randint(1_000_000, 9_999_999)
        qs.append(Query("big_arith", f"Compute {a} * {b}. Give the exact product.",
                        str(a * b), "tool"))

    for _ in range(n_per_type):
        ent = f"unit-{rng.randint(1000, 9999)}"
        rec = {a: rng.randint(1000, 9999) for a in _ATTRS}
        kb[ent] = rec
        attr = rng.choice(_ATTRS)
        qs.append(Query("kb_lookup",
                        f"In the classified registry, what is the {attr} of {ent}? "
                        f"Answer with the number only.",
                        str(rec[attr]), "retrieve", {"entity": ent, "attr": attr}))

    rng.shuffle(qs)
    return qs, kb


def check(q: Query, out: str) -> bool:
    # answer-type-aware: string answers (capital names, KB values like 'Vantis37')
    # are graded by substring; integer answers by last-integer match. This is robust
    # to workloads whose non-'factual' types nonetheless have string answers.
    ans = q.answer.strip()
    if not re.fullmatch(r"-?\d+", ans):
        return ans.lower() in out.strip().lower()
    nums = re.findall(r"-?\d+", out)
    return bool(nums) and nums[-1] == ans


# --------------------------------------------------------------------- tool eval
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul}


def _safe_eval(expr):
    try:
        def w(n):
            if isinstance(n, ast.Expression): return w(n.body)
            if isinstance(n, ast.Constant): return n.value
            if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
                return _OPS[type(n.op)](w(n.left), w(n.right))
            raise ValueError
        return w(ast.parse(expr, mode="eval"))
    except Exception:
        return None


# --------------------------------------------------------------------- model
class QwenLM:
    def __init__(self, path, device="cuda"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self._t = torch
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

    def gen(self, system, user, max_new_tokens=16):
        torch = self._t
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = self.tok(text, return_tensors="pt").to(self.m.device)
        with torch.no_grad():
            out = self.m.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False,
                                  pad_token_id=self.tok.pad_token_id)
        return self.tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)


# --------------------------------------------------------------------- mechanisms
def m_fast(lm, q, kb):
    return lm.gen("Answer with only the final answer, nothing else.", q.prompt, 16)


def m_cot(lm, q, kb):
    return lm.gen("Reason step by step, then put the final answer on the last line.",
                  q.prompt, 400)


def m_tool(lm, q, kb):
    # normalize natural-language operators so the calculator handles "X times Y" too
    text = re.sub(r"\s+times\s+", " * ", q.prompt.lower())
    text = re.sub(r"\s+plus\s+", " + ", text)
    text = re.sub(r"\s+minus\s+", " - ", text)
    mt = re.search(r"(-?\d+(?:\s*[*+\-]\s*\d+)+)", text)
    if mt:
        v = _safe_eval(mt.group(1).replace(" ", ""))
        if v is not None:
            return str(v)
    return ""


def m_retrieve(lm, q, kb):
    # idealized retriever: exact KB lookup (models a working retrieval tool). The point
    # is CoT CANNOT access this at all; retriever quality is a separate axis.
    e, a = q.meta.get("entity"), q.meta.get("attr")
    if e in (kb or {}) and a in kb[e]:
        return str(kb[e][a])
    return ""


MECH_FN = {"fast": m_fast, "cot": m_cot, "tool": m_tool, "retrieve": m_retrieve}
