"""Kaggle-kernel evaluation engine (internet-off). Per item: greedy CoT pass with token
statistics, K sampled passes for consistency, frozen confidence elicitation. Writes one CSV
row per item; checkpoints after every benchmark and every PARTIAL_EVERY items; resumes from
any previously attached run's output.

Depends only on torch/transformers (present in the Kaggle image) + stdlib.
"""
from __future__ import annotations

import csv
import json
import math
import os
import time
from collections import Counter
from pathlib import Path

from . import grading

K_SAMPLES = 5
SAMPLE_TEMP = 0.7
SAMPLE_TOP_P = 0.9
MAX_NEW = 512
CONF_MAX_NEW = 12
GREEDY_BATCH = 8          # greedy score chunks are freed as computed -> can afford 8
SAMPLE_ITEMS = 2          # items per sampled-generate call (2 x K=5 -> 10 sequences)
PARTIAL_EVERY = 25

ELICITATION = ("How confident are you that your final answer is correct? "
               "Respond with only a number from 0 to 100.")

COLUMNS = ["sample_id", "benchmark", "domain", "subcategory", "difficulty", "model",
           "prompt", "prediction", "ground_truth", "correct", "self_confidence",
           "mean_logprob", "mean_logprob_answer", "entropy", "entropy_answer",
           "token_margin", "topk_mass", "consistency", "generation_length", "latency",
           "seed", "extraction_ok", "conf_parse_ok", "n_samples_parsed", "detail"]


# ------------------------------------------------------------------ model I/O

def find_model_dir(hint):
    """Locate an attached Kaggle model under /kaggle/input by substring hint."""
    cands = []
    for dp, _, fns in os.walk("/kaggle/input"):
        if "config.json" in fns and any(f.endswith((".safetensors", ".bin")) for f in fns):
            cands.append(dp)
    for dp in cands:
        if hint.lower() in dp.lower().replace("_", "-"):
            return dp
    raise FileNotFoundError(f"no attached model matches hint '{hint}'; found: {cands}")


class LM:
    def __init__(self, path, model_name):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.name = model_name
        self.tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        dt = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        try:
            self.m = AutoModelForCausalLM.from_pretrained(
                path, dtype=dt, device_map={"": 0}, trust_remote_code=True)
        except TypeError:
            self.m = AutoModelForCausalLM.from_pretrained(
                path, torch_dtype=dt, device_map={"": 0}, trust_remote_code=True)
        self.m.eval()
        self.tok.padding_side = "left"

    def _encode_chat(self, user, assistant=None, follow_up=None):
        msgs = [{"role": "user", "content": user}]
        if assistant is not None:
            msgs.append({"role": "assistant", "content": assistant})
        if follow_up is not None:
            msgs.append({"role": "user", "content": follow_up})
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def greedy_with_stats(self, prompts, seed):
        """Batched greedy generation returning per-sequence text + token statistics.
        Stats are computed with chunked tensor ops (a per-step .item() loop costs thousands
        of GPU syncs per batch and dominated wall time in S0)."""
        torch = self.torch
        torch.manual_seed(seed)
        texts = [self._encode_chat(p) for p in prompts]
        enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.m.device)
        t0 = time.time()
        with torch.no_grad():
            out = self.m.generate(
                **enc, max_new_tokens=MAX_NEW, do_sample=False,
                output_scores=True, return_dict_in_generate=True,
                pad_token_id=self.tok.pad_token_id)
        latency = (time.time() - t0) / len(prompts)
        prompt_len = enc["input_ids"].shape[1]
        gen_ids = out.sequences[:, prompt_len:]                    # [B, T]
        B, T = gen_ids.shape
        lp_tok = torch.empty(B, T)
        ent_tok = torch.empty(B, T)
        mar_tok = torch.empty(B, T)
        top5_tok = torch.empty(B, T)
        with torch.no_grad():
            for c0 in range(0, T, 64):                             # chunk over steps
                chunk = torch.stack(out.scores[c0:c0 + 64], dim=1).float()  # [B, t, V]
                lp = torch.log_softmax(chunk, dim=-1)
                p = lp.exp()
                ids_c = gen_ids[:, c0:c0 + chunk.shape[1]].unsqueeze(-1)
                lp_tok[:, c0:c0 + chunk.shape[1]] = lp.gather(-1, ids_c).squeeze(-1).cpu()
                ent_tok[:, c0:c0 + chunk.shape[1]] = (-(p * lp).sum(-1)).cpu()
                top5 = torch.topk(p, 5, dim=-1).values             # [B, t, 5]
                mar_tok[:, c0:c0 + chunk.shape[1]] = (top5[..., 0] - top5[..., 1]).cpu()
                top5_tok[:, c0:c0 + chunk.shape[1]] = top5.sum(-1).cpu()
                del chunk, lp, p, top5
        results = []
        for b in range(B):
            ids = []
            for tid in gen_ids[b].tolist():                        # trim at eos/pad
                ids.append(tid)
                if tid == self.tok.eos_token_id:
                    break
            n = len(ids)
            piece_lens = [len(self.tok.decode([tid])) for tid in ids]
            text = self.tok.decode(ids, skip_special_tokens=True)
            results.append(dict(
                text=text, ids=ids,
                logprobs=lp_tok[b, :n].tolist(), entropies=ent_tok[b, :n].tolist(),
                margins=mar_tok[b, :n].tolist(), topk_masses=top5_tok[b, :n].tolist(),
                piece_lens=piece_lens, latency=latency, n_tokens=n))
        del out
        torch.cuda.empty_cache()
        return results

    def sample_k_multi(self, prompts, k, seed):
        """K sampled generations for SEVERAL items in one generate() call.
        Returns [[k samples] per prompt]. Same sampling protocol (T, top-p, K) as the
        frozen plan — batching is an engineering change only."""
        torch = self.torch
        torch.manual_seed(seed)
        texts = [self._encode_chat(p) for p in prompts]
        enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.m.device)
        with torch.no_grad():
            out = self.m.generate(
                **enc, max_new_tokens=MAX_NEW, do_sample=True, temperature=SAMPLE_TEMP,
                top_p=SAMPLE_TOP_P, num_return_sequences=k,
                pad_token_id=self.tok.pad_token_id)
        plen = enc["input_ids"].shape[1]
        decoded = [self.tok.decode(out[i, plen:], skip_special_tokens=True)
                   for i in range(out.shape[0])]
        # generate() groups num_return_sequences per input: rows [i*k, (i+1)*k)
        return [decoded[i * k:(i + 1) * k] for i in range(len(prompts))]

    def sample_k(self, prompt, k, seed):
        return self.sample_k_multi([prompt], k, seed)[0]

    def elicit_confidence_multi(self, prompt_answer_pairs, seed):
        """Batched frozen elicitation. Returns [(conf, parse_ok)] per pair."""
        torch = self.torch
        torch.manual_seed(seed)
        texts = [self._encode_chat(p, assistant=a, follow_up=ELICITATION)
                 for p, a in prompt_answer_pairs]
        enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.m.device)
        with torch.no_grad():
            out = self.m.generate(**enc, max_new_tokens=CONF_MAX_NEW, do_sample=False,
                                  pad_token_id=self.tok.pad_token_id)
        plen = enc["input_ids"].shape[1]
        results = []
        for i in range(len(texts)):
            reply = self.tok.decode(out[i, plen:], skip_special_tokens=True)
            num, _ = grading.extract_number(reply)
            if num is None or num < 0 or num > 100:
                results.append((float("nan"), 0))
            else:
                results.append((num / 100.0, 1))
        return results

    def elicit_confidence(self, prompt, answer_text, seed):
        return self.elicit_confidence_multi([(prompt, answer_text)], seed)[0]


# ------------------------------------------------------------- signal helpers

def answer_span_stats(res, text):
    """Token statistics restricted to the final-answer line (P3). Locates the last
    'Final answer' occurrence by accumulating per-token decoded lengths — approximate for
    BPE boundary effects but consistent across models. NaN if no marker."""
    low = text.lower()
    pos = low.rfind("final answer")
    if pos < 0:
        return float("nan"), float("nan")
    cum, start_tok = 0, None
    for i, plen in enumerate(res["piece_lens"]):
        cum += plen
        if cum >= pos:
            start_tok = i
            break
    if start_tok is None or start_tok >= len(res["logprobs"]):
        return float("nan"), float("nan")
    lp = res["logprobs"][start_tok:]
    en = res["entropies"][start_tok:]
    return (sum(lp) / len(lp), sum(en) / len(en)) if lp else (float("nan"), float("nan"))


def consistency_of(samples, item):
    """Modal-answer fraction over K sampled generations (non-code)."""
    votes = [grading.normalize_answer(s, item) for s in samples]
    parsed = [v for v in votes if v != "__NONE__"]
    if not parsed:
        return 0.0, 0.0, 0
    counts = Counter(parsed)
    modal = counts.most_common(1)[0][1]
    consistency = modal / len(votes)
    probs = [c / len(parsed) for c in counts.values()]
    max_h = math.log(len(votes))
    ent = (-sum(p * math.log(p) for p in probs) / max_h) if max_h > 0 else 0.0
    return consistency, ent, len(parsed)


def code_consistency(samples, item, timeout=6):
    """Output-agreement on example inputs already present in the prompt (PLAN 3).
    Extracts `f(args)` call expressions from the grade spec's asserts / docstring tests,
    executes each sample on them, and votes on the joint output tuple."""
    calls = _example_calls(item)
    if not calls:
        return float("nan"), float("nan"), 0
    outputs = []
    for s in samples:
        code, _ = grading.extract_code(s)
        prog = (code + "\n\nimport json as _j\n_res=[]\n" +
                "\n".join(f"""
try:
    _res.append(repr({c}))
except Exception as _e:
    _res.append('ERR:'+type(_e).__name__)""" for c in calls) +
                "\nprint('__OUT__'+_j.dumps(_res))\n")
        outputs.append(_capture_out(prog, timeout))
    votes = [o for o in outputs if o is not None]
    if not votes:
        return 0.0, 0.0, 0
    counts = Counter(votes)
    modal = counts.most_common(1)[0][1]
    consistency = modal / len(outputs)
    probs = [c / len(votes) for c in counts.values()]
    max_h = math.log(len(outputs))
    ent = (-sum(p * math.log(p) for p in probs) / max_h) if max_h > 0 else 0.0
    return consistency, ent, len(votes)


def _capture_out(prog, timeout):
    import subprocess
    import sys as _sys
    try:
        r = subprocess.run([_sys.executable, "-c", prog], capture_output=True, text=True,
                           timeout=timeout)
    except Exception:  # noqa: BLE001
        return None
    for line in (r.stdout or "").splitlines():
        if line.startswith("__OUT__"):
            return line[7:]
    return None


def _example_calls(item):
    import re
    spec = item.get("grade_spec") or {}
    calls = []
    if spec.get("kind") == "mbpp":
        for t in spec.get("tests", []):
            m = re.match(r"\s*assert\s+(.+?)\s*==", t)
            if m:
                calls.append(m.group(1))
    elif spec.get("kind") == "humaneval":
        # docstring examples like >>> f(1, 2)
        for m in re.finditer(r">>>\s*([\w\.]+\(.*?\))\s*$", item.get("question_raw", ""),
                             re.MULTILINE):
            calls.append(m.group(1))
    return calls[:3]


# ------------------------------------------------------------------ main loop

def load_items(data_dir, benchmark):
    f = Path(data_dir) / f"{benchmark}.jsonl"
    return [json.loads(l) for l in open(f, encoding="utf-8")] if f.exists() else []


def done_ids_from_previous(work_dir, model_name):
    """Scan /kaggle/input (attached previous run outputs) + work_dir for completed rows."""
    done = {}
    roots = [Path("/kaggle/input"), Path(work_dir)]
    for root in roots:
        if not root.exists():
            continue
        for f in root.rglob(f"rows_{model_name}_*.csv"):
            try:
                with open(f, encoding="utf-8") as fh:
                    for row in csv.DictReader(fh):
                        done[(row["sample_id"], row["seed"])] = row
            except Exception:  # noqa: BLE001
                continue
    return done


def write_rows(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in COLUMNS})


def run_benchmark(lm, items, model_name, seed, work_dir, prev_done, limit=None):
    bench = items[0]["benchmark"]
    rows = []
    todo = []
    for it in items[: (limit or len(items))]:
        key = (it["sample_id"], str(seed))
        if key in prev_done:
            rows.append(dict(prev_done[key]))
        else:
            todo.append(it)
    log(f"[{bench}] {len(todo)} to run, {len(rows)} restored")
    partial = Path(work_dir) / f"rows_{model_name}_{bench}.partial.csv"

    for start in range(0, len(todo), GREEDY_BATCH):
        batch = todo[start:start + GREEDY_BATCH]
        try:
            greedy = lm.greedy_with_stats([it["prompt"] for it in batch], seed)
        except RuntimeError as e:            # GPU OOM -> retry item-by-item (PLAN 6)
            if "out of memory" not in str(e).lower():
                raise
            log(f"[{bench}] OOM at batch {start}; retrying singly")
            lm.torch.cuda.empty_cache()
            greedy = []
            for it in batch:
                greedy += lm.greedy_with_stats([it["prompt"]], seed)
        # sampled passes: SAMPLE_ITEMS items x K sequences per generate() call
        all_samples = []
        for s0 in range(0, len(batch), SAMPLE_ITEMS):
            chunk = batch[s0:s0 + SAMPLE_ITEMS]
            try:
                all_samples += lm.sample_k_multi([it["prompt"] for it in chunk],
                                                 K_SAMPLES, seed * 7919 + 13)
            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
                lm.torch.cuda.empty_cache()
                for it in chunk:
                    all_samples.append(lm.sample_k(it["prompt"], K_SAMPLES,
                                                   seed * 7919 + 13))
        # batched frozen elicitation
        try:
            confs = lm.elicit_confidence_multi(
                [(it["prompt"], res["text"]) for it, res in zip(batch, greedy)], seed)
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            lm.torch.cuda.empty_cache()
            confs = [lm.elicit_confidence(it["prompt"], res["text"], seed)
                     for it, res in zip(batch, greedy)]

        for it, res, samples, (conf, conf_ok) in zip(batch, greedy, all_samples, confs):
            text = res["text"]
            g = grading.grade(text, it)
            lp_ans, ent_ans = answer_span_stats(res, text)
            if it["answer_type"] == "code":
                cons, ent_answer_dist, n_parsed = code_consistency(samples, it)
            else:
                cons, ent_answer_dist, n_parsed = consistency_of(samples, it)
            n = max(1, res["n_tokens"])
            rows.append(dict(
                sample_id=it["sample_id"], benchmark=bench, domain=it["domain"],
                subcategory=it.get("subcategory", ""),
                difficulty="" if it.get("difficulty") is None else it["difficulty"],
                model=model_name, prompt=it["prompt"],
                prediction=str(g["prediction"])[:400], ground_truth=str(it["gold"])[:400],
                correct=g["correct"], self_confidence=conf,
                mean_logprob=sum(res["logprobs"]) / n,
                mean_logprob_answer=lp_ans,
                entropy=sum(res["entropies"]) / n,
                entropy_answer=ent_ans,
                token_margin=sum(res["margins"]) / n,
                topk_mass=sum(res["topk_masses"]) / n,
                consistency=cons, generation_length=res["n_tokens"],
                latency=res["latency"], seed=seed,
                extraction_ok=g["extraction_ok"], conf_parse_ok=conf_ok,
                n_samples_parsed=n_parsed, detail=g["detail"]))
            # answer-distribution entropy rides in a sidecar (schema stays as PLAN 3)
            _sidecar(work_dir, model_name, bench, it["sample_id"], seed, samples,
                     ent_answer_dist)
        if (start // GREEDY_BATCH) % (PARTIAL_EVERY // GREEDY_BATCH or 1) == 0:
            write_rows(partial, rows)
            log(f"[{bench}] {len(rows)} rows (partial saved)")

    final = Path(work_dir) / f"rows_{model_name}_{bench}.csv"
    write_rows(final, rows)
    if partial.exists():
        partial.unlink()
    acc = sum(int(r["correct"]) for r in rows) / max(1, len(rows))
    log(f"[{bench}] DONE n={len(rows)} acc={acc:.3f}")
    return rows


def _sidecar(work_dir, model_name, bench, sample_id, seed, samples, ent_answer_dist):
    f = Path(work_dir) / f"samples_{model_name}_{bench}.jsonl"
    with open(f, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(dict(sample_id=sample_id, seed=seed,
                                 entropy_answer_dist=ent_answer_dist,
                                 samples=[s[:2000] for s in samples]),
                            ensure_ascii=False) + "\n")


_LOGF = None


def log(msg):
    """Print AND append to /kaggle/working/run.log — kernel stdout logs are sometimes lost
    on hard container errors, but files in the working dir survive."""
    print(msg, flush=True)
    if _LOGF is not None:
        _LOGF.write(msg + "\n")
        _LOGF.flush()


def main(model_hint, model_name, benchmarks, data_dir, work_dir="/kaggle/working",
         seed=0, limit=None):
    global _LOGF
    _LOGF = open(Path(work_dir) / "run.log", "a", encoding="utf-8")
    try:
        _main(model_hint, model_name, benchmarks, data_dir, work_dir, seed, limit)
    except BaseException:
        import traceback
        tb = traceback.format_exc()
        log(tb)
        (Path(work_dir) / "traceback.txt").write_text(tb, encoding="utf-8")
        raise


def _main(model_hint, model_name, benchmarks, data_dir, work_dir, seed, limit):
    from ugen.pipeline.tests import test_stats  # stats must self-verify before any GPU row
    for k, fn in sorted(vars(test_stats).items()):
        if k.startswith("test_"):
            fn()
    log("stats self-test passed in-kernel")

    mdir = find_model_dir(model_hint)
    log(f"loading model from {mdir}")
    lm = LM(mdir, model_name)
    log("model loaded")
    prev = done_ids_from_previous(work_dir, model_name)
    log(f"restored {len(prev)} previously-completed rows")
    all_rows = []
    for bench in benchmarks:
        items = load_items(data_dir, bench)
        if not items:
            log(f"[{bench}] no frozen data — skipped")
            continue
        all_rows += run_benchmark(lm, items, model_name, seed, work_dir, prev, limit=limit)
    write_rows(Path(work_dir) / f"rows_{model_name}_ALL.csv", all_rows)
    log(f"TOTAL rows: {len(all_rows)}")
