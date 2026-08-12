"""Free input features — computable from the item INPUT alone, zero model calls, no
ground-truth annotation (the E2pre leakage rule). These are the baselines every internal
signal must beat before being credited with information (PLAN.md P5).

Difficulty descriptors derived from gold annotations (e.g. GSM8K step counts) live in the
`difficulty` column and are NEVER used here.
"""
from __future__ import annotations

import re


def features(item):
    """item: frozen dataset dict (needs prompt / question_raw / options / answer_type)."""
    q = item.get("question_raw") or item["prompt"]
    prompt = item["prompt"]
    nums = [float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*\.?\d*", q)]
    f = dict(
        prompt_chars=float(len(prompt)),
        prompt_words=float(len(prompt.split())),
        question_words=float(len(q.split())),
        digit_count=float(sum(c.isdigit() for c in q)),
        n_numbers=float(len(nums)),
        max_num_log10=_log10_max(nums),
        n_options=float(len(item["options"])) if item.get("options") else 0.0,
    )
    if item["answer_type"] == "code":
        sig = re.search(r"def\s+\w+\(([^)]*)\)", q)
        f["n_args"] = float(len([a for a in sig.group(1).split(",") if a.strip()])) if sig else 0.0
        doc = re.search(r'"""(.*?)"""', q, re.DOTALL)
        f["docstring_words"] = float(len(doc.group(1).split())) if doc else 0.0
    else:
        f["n_args"] = 0.0
        f["docstring_words"] = 0.0
    return f


def _log10_max(nums):
    import math
    mx = max((abs(n) for n in nums), default=0.0)
    return math.log10(mx) if mx >= 1 else 0.0


FEATURE_NAMES = ["prompt_chars", "prompt_words", "question_words", "digit_count",
                 "n_numbers", "max_num_log10", "n_options", "n_args", "docstring_words"]
