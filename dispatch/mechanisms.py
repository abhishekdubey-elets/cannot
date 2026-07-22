"""The four reasoning mechanisms the controller dispatches between.

Each is a distinct way of processing the SAME query. Specialization (or its absence)
is what run_calibrate.py measures.

  fast     : System-1 — terse, one short generation.
  cot      : System-2 — chain-of-thought, long generation.
  tool     : parse a computable arithmetic expression and evaluate it exactly (AST
             whitelist, no eval()); inapplicable queries get no answer.
  retrieve : grounding — instructed to copy the answer out of the provided context.
"""
from __future__ import annotations

import ast
import operator
import re

import torch

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.FloorDiv: operator.floordiv, ast.Div: operator.truediv}


def _safe_eval(expr: str):
    try:
        def walk(n):
            if isinstance(n, ast.Expression):
                return walk(n.body)
            if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
                return n.value
            if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
                return _OPS[type(n.op)](walk(n.left), walk(n.right))
            if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
                return -walk(n.operand)
            raise ValueError
        return walk(ast.parse(expr, mode="eval"))
    except Exception:
        return None


class QwenLM:
    def __init__(self, path, device="cuda"):
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
    def gen(self, system, user, max_new_tokens=16):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = self.tok(text, return_tensors="pt").to(self.m.device)
        out = self.m.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False,
                              pad_token_id=self.tok.pad_token_id)
        return self.tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)


# ---- mechanisms: (lm, query) -> (text_output, cost_tokens) -----------------
def fast(lm, q):
    return lm.gen("Answer with only the final answer, nothing else.", q.prompt, 12), 12


def cot(lm, q):
    return lm.gen("Reason step by step, then put the final answer on the last line.",
                  q.prompt, 300), 300


def tool(lm, q):
    # a calculator tool: only useful when the query contains a computable expression
    m = re.search(r"(-?\d+(?:\s*[*+\-/]\s*\d+)+)", q.prompt)
    if m:
        v = _safe_eval(m.group(1).replace(" ", ""))
        if v is not None:
            return str(int(v) if float(v).is_integer() else v), 0
    return "", 0                         # tool inapplicable -> no answer


def retrieve(lm, q):
    return lm.gen("The answer is stated in the text. Find and copy it. "
                  "Answer with only the answer.", q.prompt, 16), 16


MECHANISMS = {"fast": fast, "cot": cot, "tool": tool, "retrieve": retrieve}
