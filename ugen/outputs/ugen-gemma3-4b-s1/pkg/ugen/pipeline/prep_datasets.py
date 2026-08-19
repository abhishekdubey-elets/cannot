"""Freeze evaluation subsets from official HuggingFace splits (PLAN.md section 1).

Runs LOCALLY (needs internet). Writes datasets/<benchmark>.jsonl + datasets/MANIFEST.json
(source, split, sampled indices, sha256, prompt-template hash). Kaggle kernels never touch
the network — they read these frozen files, so every model sees byte-identical prompts.

Usage:  python -m ugen.pipeline.prep_datasets [--out DIR] [--n 200]
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import re
from pathlib import Path

SEED = 0
N_DEFAULT = 200
LETTERS = "ABCDEFGHIJ"

COT_MC = ('Reason step by step, then give your final answer on the last line as '
          '"Final answer: <letter>".')
COT_NUM = ('Reason step by step, then give your final answer on the last line as '
           '"Final answer: <number>".')
COT_YN = ('Reason step by step, then give your final answer on the last line as '
          '"Final answer: Yes" or "Final answer: No".')
COT_CODE = ('Think about the problem if needed, then output the complete solution as a '
            'single ```python code block.')

# Frozen confidence-elicitation wording (PLAN.md section 3 — never edit between runs).
ELICITATION = ("How confident are you that your final answer is correct? "
               "Respond with only a number from 0 to 100.")


def mc_prompt(question, options):
    letters = LETTERS[: len(options)]
    opts = "\n".join(f"{l}. {o}" for l, o in zip(letters, options))
    return f"{question.strip()}\n\nOptions:\n{opts}\n\n{COT_MC}"


def _sample(n_total, n_want, seed):
    idx = list(range(n_total))
    random.Random(seed).shuffle(idx)
    return sorted(idx[: min(n_want, n_total)])


def _item(benchmark, domain, idx, prompt, answer_type, gold, options=None,
          grade_spec=None, subcategory="", difficulty=None, question_raw=""):
    return dict(sample_id=f"{benchmark}:{idx}", benchmark=benchmark, domain=domain,
                subcategory=subcategory, difficulty=difficulty, prompt=prompt,
                answer_type=answer_type, gold=gold, options=options,
                grade_spec=grade_spec, source_index=idx, question_raw=question_raw)


# ------------------------------------------------------------------ loaders

def load_mmlu(ds, n):
    d = ds("cais/mmlu", "all", split="test")
    keep = _sample(len(d), n, SEED)
    out = []
    for i in keep:
        r = d[i]
        out.append(_item("mmlu", "knowledge", i, mc_prompt(r["question"], r["choices"]),
                         "mc_letter", LETTERS[r["answer"]], r["choices"],
                         subcategory=r["subject"], question_raw=r["question"]))
    return out


def load_mmlu_pro(ds, n):
    d = ds("TIGER-Lab/MMLU-Pro", split="test")
    keep = _sample(len(d), n, SEED)
    out = []
    for i in keep:
        r = d[i]
        out.append(_item("mmlu_pro", "knowledge", i, mc_prompt(r["question"], r["options"]),
                         "mc_letter", LETTERS[r["answer_index"]], r["options"],
                         subcategory=r["category"], question_raw=r["question"]))
    return out


def load_truthfulqa(ds, n):
    d = ds("truthfulqa/truthful_qa", "multiple_choice", split="validation")
    keep = _sample(len(d), n, SEED)
    out = []
    for i in keep:
        r = d[i]
        choices = list(r["mc1_targets"]["choices"])
        labels = list(r["mc1_targets"]["labels"])
        # some mc1 items exceed the A-J alphabet: keep the correct option + the first 9
        # distractors (disclosed protocol adaptation; applied before shuffling)
        if len(choices) > 10:
            correct_i = labels.index(1)
            keep = [correct_i] + [j for j in range(len(choices)) if j != correct_i][:9]
            choices = [choices[j] for j in keep]
            labels = [labels[j] for j in keep]
        # mc1 lists the correct option FIRST -> shuffle per item (leakage guard, PLAN 1)
        order = list(range(len(choices)))
        random.Random(SEED * 100003 + i).shuffle(order)
        choices_s = [choices[j] for j in order]
        gold = LETTERS[order.index(labels.index(1))]
        out.append(_item("truthfulqa", "knowledge", i, mc_prompt(r["question"], choices_s),
                         "mc_letter", gold, choices_s, question_raw=r["question"]))
    return out


def load_gpqa(ds, n):
    """Gated — requires an HF token with accepted terms. Caller catches failure."""
    d = ds("Idavidrein/gpqa", "gpqa_main", split="train")   # gpqa ships a single split
    keep = _sample(len(d), n, SEED)
    out = []
    for i in keep:
        r = d[i]
        opts = [r["Correct Answer"], r["Incorrect Answer 1"], r["Incorrect Answer 2"],
                r["Incorrect Answer 3"]]
        order = list(range(4))
        random.Random(SEED * 100003 + i).shuffle(order)
        opts_s = [str(opts[j]).strip() for j in order]
        gold = LETTERS[order.index(0)]
        out.append(_item("gpqa", "knowledge", i, mc_prompt(r["Question"], opts_s),
                         "mc_letter", gold, opts_s,
                         subcategory=r.get("High-level domain", ""),
                         question_raw=r["Question"]))
    return out


def load_gsm8k(ds, n):
    d = ds("openai/gsm8k", "main", split="test")
    keep = _sample(len(d), n, SEED)
    out = []
    for i in keep:
        r = d[i]
        gold_raw = r["answer"].split("####")[-1].strip().replace(",", "")
        steps = len([l for l in r["answer"].split("####")[0].strip().split("\n") if l.strip()])
        out.append(_item("gsm8k", "arithmetic", i, f"{r['question'].strip()}\n\n{COT_NUM}",
                         "numeric", float(gold_raw), difficulty=steps,
                         question_raw=r["question"]))
    return out


def load_svamp(ds, n):
    d = ds("ChilleD/SVAMP", split="test")
    keep = _sample(len(d), n, SEED)
    out = []
    for i in keep:
        r = d[i]
        q = f"{r['Body'].strip()} {r['Question'].strip()}"
        out.append(_item("svamp", "arithmetic", i, f"{q}\n\n{COT_NUM}", "numeric",
                         float(r["Answer"]), subcategory=r.get("Type", ""), question_raw=q))
    return out


def load_multiarith(ds, n):
    d = ds("ChilleD/MultiArith", split="test")
    keep = _sample(len(d), n, SEED)
    out = []
    for i in keep:
        r = d[i]
        out.append(_item("multiarith", "arithmetic", i,
                         f"{r['question'].strip()}\n\n{COT_NUM}", "numeric",
                         float(r["final_ans"]), question_raw=r["question"]))
    return out


def load_commonsenseqa(ds, n):
    d = ds("tau/commonsense_qa", split="validation")   # test split is unlabelled (PLAN 1)
    keep = _sample(len(d), n, SEED)
    out = []
    for i in keep:
        r = d[i]
        texts = list(r["choices"]["text"])
        labels = list(r["choices"]["label"])
        gold = LETTERS[labels.index(r["answerKey"])]
        out.append(_item("commonsenseqa", "commonsense", i, mc_prompt(r["question"], texts),
                         "mc_letter", gold, texts, question_raw=r["question"]))
    return out


def load_arc_challenge(ds, n):
    d = ds("allenai/ai2_arc", "ARC-Challenge", split="test")
    keep = _sample(len(d), n, SEED)
    out = []
    for i in keep:
        r = d[i]
        texts = list(r["choices"]["text"])
        labels = list(r["choices"]["label"])     # sometimes "1".."4", sometimes "A".."D"
        if r["answerKey"] not in labels:
            continue
        gold = LETTERS[labels.index(r["answerKey"])]
        out.append(_item("arc_challenge", "commonsense", i, mc_prompt(r["question"], texts),
                         "mc_letter", gold, texts, question_raw=r["question"]))
    return out


def load_strategyqa(ds, n):
    d = ds("ChilleD/StrategyQA", split="test")
    keep = _sample(len(d), n, SEED)
    out = []
    for i in keep:
        r = d[i]
        gold = "yes" if bool(r["answer"]) else "no"
        out.append(_item("strategyqa", "commonsense", i,
                         f"{r['question'].strip()}\n\n{COT_YN}", "yesno", gold,
                         question_raw=r["question"]))
    return out


BBH_TASKS = ["date_understanding", "logical_deduction_five_objects",
             "tracking_shuffled_objects_three_objects", "causal_judgement",
             "web_of_lies", "temporal_sequences"]


def load_bbh(ds, n):
    per = max(1, n // len(BBH_TASKS))
    out = []
    for task in BBH_TASKS:
        d = ds("lukaemon/bbh", task, split="test")
        keep = _sample(len(d), per, SEED)
        for i in keep:
            r = d[i]
            target = r["target"].strip()
            m = re.fullmatch(r"\(([A-J])\)", target)
            inp = r["input"].strip()
            if m:
                # options are embedded in the input text; grade on the letter
                prompt = (f"{inp}\n\n" + COT_MC)
                out.append(_item("bbh", "reasoning", f"{task}:{i}", prompt, "mc_letter",
                                 m.group(1), _bbh_options(inp), subcategory=task,
                                 question_raw=inp))
            elif target.lower() in ("yes", "no", "true", "false"):
                gold = "yes" if target.lower() in ("yes", "true") else "no"
                out.append(_item("bbh", "reasoning", f"{task}:{i}",
                                 f"{inp}\n\n{COT_YN}", "yesno", gold,
                                 subcategory=task, question_raw=inp))
            else:
                continue   # free-form BBH targets are excluded (tasks above avoid them)
    return out


def _bbh_options(inp):
    opts = re.findall(r"\(([A-J])\)\s*([^\n(]+)", inp)
    return [t.strip() for _, t in opts] or None


def load_musr(ds, n):
    configs = ["murder_mysteries", "object_placements", "team_allocation"]
    per = max(1, n // len(configs))
    out = []
    for cfg in configs:
        d = ds("TAUR-Lab/MuSR", split=cfg)
        keep = _sample(len(d), per, SEED)
        for i in keep:
            r = d[i]
            choices = ast.literal_eval(r["choices"])
            q = f"{r['narrative'].strip()}\n\n{r['question'].strip()}"
            out.append(_item("musr", "reasoning", f"{cfg}:{i}", mc_prompt(q, choices),
                             "mc_letter", LETTERS[int(r["answer_index"])], choices,
                             subcategory=cfg, question_raw=q))
    return out


def load_humaneval(ds, n):
    d = ds("openai/openai_humaneval", split="test")
    keep = _sample(len(d), n, SEED)      # n>=164 keeps all
    out = []
    for i in keep:
        r = d[i]
        prompt = (f"Complete the following Python function.\n\n```python\n{r['prompt']}```"
                  f"\n\n{COT_CODE}")
        out.append(_item("humaneval", "coding", i, prompt, "code", r["canonical_solution"],
                         grade_spec=dict(kind="humaneval", test=r["test"],
                                         entry_point=r["entry_point"]),
                         subcategory=r["task_id"], question_raw=r["prompt"]))
    return out


def load_mbpp(ds, n):
    d = ds("google-research-datasets/mbpp", "sanitized", split="test")
    keep = _sample(len(d), n, SEED)
    out = []
    for i in keep:
        r = d[i]
        tests = "\n".join(r["test_list"])
        prompt = (f"{r['prompt'].strip()}\n\nYour code should pass these tests:\n"
                  f"```python\n{tests}\n```\n\n{COT_CODE}")
        out.append(_item("mbpp", "coding", i, prompt, "code", r["code"],
                         grade_spec=dict(kind="mbpp", tests=list(r["test_list"]),
                                         imports=list(r.get("test_imports") or [])),
                         subcategory=str(r["task_id"]), question_raw=r["prompt"]))
    return out


LOADERS = dict(mmlu=load_mmlu, mmlu_pro=load_mmlu_pro, truthfulqa=load_truthfulqa,
               gpqa=load_gpqa, gsm8k=load_gsm8k, svamp=load_svamp,
               multiarith=load_multiarith, commonsenseqa=load_commonsenseqa,
               arc_challenge=load_arc_challenge, strategyqa=load_strategyqa,
               bbh=load_bbh, musr=load_musr, humaneval=load_humaneval, mbpp=load_mbpp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "datasets"))
    ap.add_argument("--n", type=int, default=N_DEFAULT)
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    from datasets import load_dataset   # local dependency only
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = dict(seed=SEED, n_requested=args.n, elicitation=ELICITATION,
                    templates=dict(mc=COT_MC, numeric=COT_NUM, yesno=COT_YN, code=COT_CODE),
                    benchmarks={})
    mpath = out_dir / "MANIFEST.json"
    if mpath.exists():
        manifest = json.loads(mpath.read_text(encoding="utf-8"))

    for name, loader in LOADERS.items():
        if args.only and name not in args.only:
            continue
        try:
            items = loader(load_dataset, args.n)
        except Exception as e:  # noqa: BLE001 — gated/broken sources are skipped, recorded
            print(f"[SKIP] {name}: {type(e).__name__}: {str(e)[:200]}")
            manifest["benchmarks"][name] = dict(status=f"SKIPPED: {str(e)[:200]}")
            continue
        f = out_dir / f"{name}.jsonl"
        with open(f, "w", encoding="utf-8") as fh:
            for it in items:
                fh.write(json.dumps(it, ensure_ascii=False) + "\n")
        sha = hashlib.sha256(f.read_bytes()).hexdigest()
        manifest["benchmarks"][name] = dict(
            status="ok", n=len(items), sha256=sha,
            domain=items[0]["domain"], answer_type=items[0]["answer_type"],
            indices=[it["source_index"] for it in items])
        print(f"[OK]   {name}: {len(items)} items  sha256={sha[:12]}")

    mpath.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"manifest -> {mpath}")


if __name__ == "__main__":
    main()
