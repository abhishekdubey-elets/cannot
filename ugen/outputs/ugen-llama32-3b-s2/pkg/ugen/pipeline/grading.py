"""Answer extraction and grading for the four answer types (mc_letter, numeric, yesno,
code). One shared module for all models — per-model extraction differences would masquerade
as signal differences (PLAN.md risk 2). Stdlib only; runs inside internet-off kernels.

Grading is deliberately conservative: an unparseable answer is wrong (depresses accuracy,
not signal ranking). Extraction failures are *flagged* (extraction_ok) so per-cell failure
rates can be reported and high-failure cells excluded from primary contrasts.
"""
from __future__ import annotations

import re
import subprocess
import sys

LETTERS = "ABCDEFGHIJ"


def final_segment(text):
    """The part of the response that should contain the answer: the last 'Final answer:'
    line if present, else the last non-empty line."""
    matches = list(re.finditer(r"final\s*answer\s*[:\-]?\s*", text, re.IGNORECASE))
    if matches:
        seg = text[matches[-1].end():].strip().split("\n")[0].strip()
        if seg:
            return seg, True
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    return (lines[-1] if lines else ""), False


def _norm(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def extract_mc(text, options):
    """-> (letter or None, extraction_ok). Letter parse first, then option-text match."""
    seg, had_marker = final_segment(text)
    valid = LETTERS[: len(options)]
    m = re.match(r"^\(?\**([A-J])\**[\).:\s]?", seg.strip())
    if m and m.group(1) in valid:
        return m.group(1), had_marker
    m = re.search(r"\b([A-J])\b", seg)
    if m and m.group(1) in valid:
        return m.group(1), had_marker
    seg_n = _norm(seg)
    if seg_n:
        for i, opt in enumerate(options):
            if _norm(str(opt)) == seg_n:
                return valid[i], had_marker
        hits = [i for i, opt in enumerate(options) if _norm(str(opt)) and _norm(str(opt)) in seg_n]
        if len(hits) == 1:
            return valid[hits[0]], had_marker
    return None, False


_NUM = re.compile(r"-?\$?\d[\d,]*\.?\d*")


def extract_number(text):
    """-> (float or None, extraction_ok). Last number in the final segment; falls back to
    the last number anywhere in the response."""
    seg, had_marker = final_segment(text)
    for source, ok in ((seg, had_marker), (text, False)):
        nums = _NUM.findall(source)
        if nums:
            raw = nums[-1].replace(",", "").replace("$", "")
            try:
                return float(raw), ok
            except ValueError:
                continue
    return None, False


def extract_yesno(text):
    seg, had_marker = final_segment(text)
    for source, ok in ((seg, had_marker), (text[-200:], False)):
        m = list(re.finditer(r"\b(yes|no|true|false)\b", source, re.IGNORECASE))
        if m:
            w = m[-1].group(1).lower()
            return ("yes" if w in ("yes", "true") else "no"), ok
    return None, False


def extract_code(text):
    """Last fenced code block, else the raw text. -> (code, extraction_ok)."""
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if blocks:
        return blocks[-1].strip(), True
    return text.strip(), False


def numbers_equal(a, b, rel=1e-4, abs_tol=1e-6):
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return False
    return abs(a - b) <= max(abs_tol, rel * max(abs(a), abs(b)))


def run_python(code, timeout=8):
    """Execute code in a fresh subprocess. -> (passed, detail). The subprocess exiting 0
    with the sentinel printed counts as a pass."""
    harness = code + "\nprint('__UGEN_PASS__')\n"
    try:
        r = subprocess.run([sys.executable, "-c", harness], capture_output=True,
                           text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as e:  # noqa: BLE001 — grading must never crash the run
        return False, f"launch_error:{e}"
    if r.returncode == 0 and "__UGEN_PASS__" in r.stdout:
        return True, "ok"
    err = (r.stderr or "").strip().split("\n")
    return False, (err[-1][:200] if err and err[-1] else f"exit_{r.returncode}")


def grade_code(generated_code, grade_spec, timeout=8):
    """grade_spec: {'kind': 'humaneval', 'test': ..., 'entry_point': ...} or
    {'kind': 'mbpp', 'tests': [...], 'imports': [...]}."""
    if grade_spec["kind"] == "humaneval":
        prog = (generated_code + "\n\n" + grade_spec["test"] +
                f"\ncheck({grade_spec['entry_point']})\n")
    else:
        prog = ("\n".join(grade_spec.get("imports") or []) + "\n" + generated_code +
                "\n\n" + "\n".join(grade_spec["tests"]) + "\n")
    return run_python(prog, timeout=timeout)


def grade(response_text, item, timeout=8):
    """Unified grading. item needs: answer_type, gold, options, grade_spec.
    -> dict(prediction, correct, extraction_ok, detail)."""
    at = item["answer_type"]
    if at == "mc_letter":
        pred, ok = extract_mc(response_text, item.get("options") or [])
        correct = pred is not None and pred == item["gold"]
        return dict(prediction=pred, correct=int(correct), extraction_ok=int(ok), detail="")
    if at == "numeric":
        pred, ok = extract_number(response_text)
        correct = pred is not None and numbers_equal(pred, item["gold"])
        return dict(prediction=pred, correct=int(correct), extraction_ok=int(ok), detail="")
    if at == "yesno":
        pred, ok = extract_yesno(response_text)
        correct = pred is not None and pred == str(item["gold"]).lower()
        return dict(prediction=pred, correct=int(correct), extraction_ok=int(ok), detail="")
    if at == "code":
        code, ok = extract_code(response_text)
        passed, detail = grade_code(code, item["grade_spec"], timeout=timeout)
        return dict(prediction=code, correct=int(passed), extraction_ok=int(ok), detail=detail)
    raise ValueError(f"unknown answer_type {at}")


def normalize_answer(response_text, item):
    """Canonical answer string for consistency voting across sampled generations.
    Code is handled separately (output-agreement, see runner)."""
    at = item["answer_type"]
    if at == "mc_letter":
        pred, _ = extract_mc(response_text, item.get("options") or [])
    elif at == "numeric":
        n, _ = extract_number(response_text)
        pred = None if n is None else f"{n:.6g}"
    elif at == "yesno":
        pred, _ = extract_yesno(response_text)
    else:
        pred = None
    return pred if pred is not None else "__NONE__"
