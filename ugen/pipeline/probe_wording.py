"""Wording-sensitivity probe (PLAN 9.9, preregistered supplementary analysis).

One greedy chain per item (frozen protocol), then the confidence question is asked four
ways on the SAME chain: the frozen wording plus three alternatives chosen to separate
number-anchoring from genuine absence of signal. w3 inverts the scale — a model that
answers "100" to "how likely are you wrong" is anchoring on the number, not reporting
uncertainty. No K-sampling, no signal extraction beyond correctness: the probe measures
elicitation fragility only, never feeds primary tables.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import grading
from .runner import (CONF_MAX_NEW, GREEDY_BATCH, LM, _greedy_with_oom_ladder,
                     find_model_dir, load_items, log, write_rows)
from .runner import ELICITATION as W0

# id -> (wording, inverted_scale)
WORDINGS = {
    "w0_original": (W0, False),
    "w1_probability": ("What is the probability that your final answer is correct? "
                       "Respond with only a number from 0 to 100.", False),
    "w2_frequency": ("If you answered 100 questions like this one, how many of your "
                     "answers would be correct? Respond with only a number from 0 to 100.",
                     False),
    "w3_inverted": ("How likely is it that your final answer is wrong? "
                    "Respond with only a number from 0 to 100.", True),
}

PROBE_BENCHMARKS = ("mmlu", "gsm8k")
N_ITEMS = 100


def elicit_with_wording(lm, prompt_answer_pairs, wording, seed):
    """Batched elicitation with an arbitrary follow-up wording. Same generation settings
    as runner.LM.elicit_confidence_multi; returns [(raw_reply, value_0_100)] per pair."""
    torch = lm.torch
    torch.manual_seed(seed)
    texts = [lm._encode_chat(p, assistant=a, follow_up=wording)
             for p, a in prompt_answer_pairs]
    enc = lm.tok(texts, return_tensors="pt", padding=True,
                 add_special_tokens=False).to(lm.m.device)
    with torch.no_grad():
        out = lm.m.generate(**enc, max_new_tokens=CONF_MAX_NEW, do_sample=False,
                            pad_token_id=lm.tok.pad_token_id)
    plen = enc["input_ids"].shape[1]
    results = []
    for i in range(len(texts)):
        reply = lm.tok.decode(out[i, plen:], skip_special_tokens=True)
        num, _ = grading.extract_number(reply)
        val = num if num is not None and 0 <= num <= 100 else float("nan")
        results.append((reply.strip()[:80], val))
    return results


def main(model_hint, model_name, data_dir, work_dir="/kaggle/working", seed=0,
         n=N_ITEMS):
    import math
    global_rows = []
    write_wordings_manifest(work_dir)
    lm = LM(find_model_dir(model_hint), model_name)
    log(f"probe_wording: model loaded ({model_name})")
    out_csv = Path(work_dir) / f"probe_wording_{model_name}.csv"
    for bench in PROBE_BENCHMARKS:
        items = load_items(data_dir, bench)[:n]
        log(f"[{bench}] probing {len(items)} items x {len(WORDINGS)} wordings")
        for start in range(0, len(items), GREEDY_BATCH):
            batch = items[start:start + GREEDY_BATCH]
            greedy = _greedy_with_oom_ladder(lm, batch, seed, bench)
            pairs = [(it["prompt"], res["text"]) for it, res in zip(batch, greedy)]
            per_wording = {}
            for wid, (wording, _inv) in WORDINGS.items():
                per_wording[wid] = elicit_with_wording(lm, pairs, wording, seed)
            for j, (it, res) in enumerate(zip(batch, greedy)):
                g = grading.grade(res["text"], it)
                row = dict(sample_id=it["sample_id"], benchmark=bench, model=model_name,
                           seed=seed, correct=g["correct"],
                           extraction_ok=g["extraction_ok"],
                           prediction=str(g["prediction"])[:200])
                for wid, (_w, inv) in WORDINGS.items():
                    reply, val = per_wording[wid][j]
                    conf = float("nan") if math.isnan(val) else \
                        (1 - val / 100.0) if inv else val / 100.0
                    row[f"conf_{wid}"] = conf          # always on the P(correct) scale
                    row[f"raw_{wid}"] = val            # as spoken, before inversion
                    row[f"reply_{wid}"] = reply
                global_rows.append(row)
            if (start // GREEDY_BATCH) % 3 == 0:
                write_rows(out_csv, global_rows)
                log(f"[{bench}] {len(global_rows)} probe rows (partial saved)")
    write_rows(out_csv, global_rows)
    log(f"probe_wording DONE: {len(global_rows)} rows -> {out_csv}")


def write_wordings_manifest(work_dir):
    (Path(work_dir) / "probe_wordings.json").write_text(
        json.dumps({k: dict(text=w, inverted=i) for k, (w, i) in WORDINGS.items()},
                   indent=1), encoding="utf-8")
