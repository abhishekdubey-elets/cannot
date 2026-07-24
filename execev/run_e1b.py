"""E1b — fixes E1's realised flaws and characterises WHY arithmetic confidence collapsed.

Six objectives:
  1. harden reasoning (1/2/3/5 steps) so it does not saturate
  2. dump RAW per-item rows for offline / adversarial re-analysis
  3. all THREE pairwise domain contrasts (E1 only coded the pre-registered pair)
  4. push the shuffled control through the contrast -> EMPIRICAL null for a difference
  5. K=5 so entropy and consistency are no longer rank-equivalent by construction
  6. ARITHMETIC CONFIDENCE DYNAMICS: does confidence track difficulty, track only
     correctness, or collapse to a constant?

Pre-registered predictions for objective 6 (stated before the run):
  P1  accuracy falls monotonically with tier in arithmetic  [sanity: difficulty is real]
  P2  spearman(selfconf, tier) in arithmetic is ~0 (|rho| < 0.2) while the same statistic
      in factoid is clearly negative -- confidence tracks knowledge difficulty but not
      computational difficulty
  P3  selfconf dynamic range within arithmetic is much narrower than within factoid
  P4  within a FIXED arithmetic tier, selfconf AUROC ~ 0.5 (no residual signal)
Falsification: if P2/P3 fail -- confidence DOES track computational difficulty -- then the
E1 arithmetic collapse is about something subtler than missing computational metacognition.
"""
from __future__ import annotations

import json
import math
import os
import random
import statistics as st
import sys
from collections import Counter

sys.path.insert(0, ".")
sys.path.insert(0, "/kaggle/working/pkg")
from execev.benchmark2 import DOMAINS, TIERS, graded, make_benchmark   # noqa: E402
from execev.stats import (auroc, auroc_diff_ci, bootstrap_ci, brier,   # noqa: E402
                          dynamic_range, ece, mutual_info, spearman)

EVAL_SEEDS = [0, 1, 2]
PER_TIER = 6                 # 6 per tier x 4 tiers x 3 domains x 3 seeds = 216 items
K = 5                        # breaks the entropy/consistency rank-equivalence of E1
MNT = 160
COT = "Reason briefly, then put the final answer on the last line."
INTERNAL = ["logprob", "consistency", "entropy", "selfconf"]
SIGNALS = INTERNAL + ["shuffled_ctrl"]


def _dump(name, txt):
    try:
        open(f"/kaggle/working/{name}", "w", encoding="utf-8").write(txt)
    except Exception:
        pass


def find_model():
    for dp, _, fns in os.walk("/kaggle/input"):
        if "config.json" in fns and any(f.endswith((".safetensors", ".bin")) for f in fns):
            if "3b-instruct" in dp.lower():
                return dp
    return None


class LM:
    def __init__(self, path):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.t = torch
        self.tok = AutoTokenizer.from_pretrained(path)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        dt = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        try:
            self.m = AutoModelForCausalLM.from_pretrained(path, dtype=dt, device_map={"": 0})
        except TypeError:
            self.m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dt, device_map={"": 0})
        self.m.eval()

    def _ids(self, system, user):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return self.tok(text, return_tensors="pt").to(self.m.device)

    def greedy_logprob(self, system, user, mnt=MNT):
        torch = self.t
        ids = self._ids(system, user)
        with torch.no_grad():
            o = self.m.generate(**ids, max_new_tokens=mnt, do_sample=False,
                                return_dict_in_generate=True, output_scores=True,
                                pad_token_id=self.tok.pad_token_id)
        seq = o.sequences[0][ids["input_ids"].shape[1]:]
        ps = []
        for i, sc in enumerate(o.scores):
            if i >= len(seq):
                break
            ps.append(float(torch.softmax(sc[0].float(), -1)[seq[i]]))
        return self.tok.decode(seq, skip_special_tokens=True), (sum(ps) / len(ps) if ps else 0.0)

    def sample(self, system, user, mnt=MNT):
        torch = self.t
        ids = self._ids(system, user)
        with torch.no_grad():
            o = self.m.generate(**ids, max_new_tokens=mnt, do_sample=True, temperature=1.0,
                                top_p=0.95, pad_token_id=self.tok.pad_token_id)
        return self.tok.decode(o[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)


def final_answer(text):
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    return lines[-1].lower() if lines else text.strip().lower()


def signals_for(lm, item):
    ans, lp = lm.greedy_logprob(COT, item.prompt)
    samp = [final_answer(lm.sample(COT, item.prompt)) for _ in range(K)]
    counts = Counter(samp)
    consistency = counts.most_common(1)[0][1] / K
    ent = -sum((c / K) * math.log(c / K) for c in counts.values())
    raw = lm.sample("Reply with only an integer 0 to 100.",
                    "How confident are you (0-100) that you can answer this correctly?\n\n"
                    + item.prompt, mnt=6)
    m = __import__("re").search(r"\d{1,3}", raw)
    return ans, dict(logprob=lp, consistency=consistency,
                     entropy=1 - ent / math.log(K),
                     selfconf=min(int(m.group()), 100) / 100 if m else 0.0)


def main():
    lines = []

    def P(*a):
        msg = " ".join(str(x) for x in a)
        print(msg, flush=True); lines.append(msg); _dump("results.txt", "\n".join(lines))

    import torch
    P(f"device={'cuda' if torch.cuda.is_available() else 'cpu'}")
    mp = find_model(); P(f"model={mp}")
    if not mp:
        P("!! no model"); return
    lm = LM(mp)

    rows = []
    rng = random.Random(12345)
    for seed in EVAL_SEEDS:
        P(f"collecting seed {seed} ...")
        for it in make_benchmark(seed, PER_TIER):
            ans, sig = signals_for(lm, it)
            sig["shuffled_ctrl"] = rng.random()
            rows.append(dict(domain=it.domain, tier=it.tier, complexity=it.complexity,
                             carries=it.carries, prompt=it.prompt,
                             accepted=list(it.accepted), answer=ans[-300:],
                             correct=int(graded(it, ans)), **sig))
    # OBJECTIVE 2 — raw dump
    _dump("rows.json", json.dumps(rows, indent=1))
    P(f"\ncollected {len(rows)} items -> rows.json dumped")

    def sub(**kw):
        return [r for r in rows if all(r[k] == v for k, v in kw.items())]

    def acc(rs):
        return st.mean(r["correct"] for r in rs) if rs else float("nan")

    P(f"overall accuracy = {acc(rows):.3f}")
    P("\nSATURATION CHECK (E1's realised flaw: reasoning hit 0.917)")
    for d in DOMAINS:
        a = acc(sub(domain=d))
        flag = "OK" if 0.2 <= a <= 0.85 else "!! SATURATED/FLOORED - domain unusable"
        P(f"  {d:<11} n={len(sub(domain=d)):>3}  accuracy={a:.3f}  {flag}")

    # ---------------- OBJECTIVE 6: confidence dynamics vs difficulty ----------------
    P("\n" + "=" * 96)
    P("OBJECTIVE 6 — DOES CONFIDENCE TRACK DIFFICULTY? (accuracy vs signals, by tier)")
    P("=" * 96)
    for d in DOMAINS:
        P(f"\n[{d}]  tier   n   accuracy | selfconf  logprob  consist  entropy")
        for t in TIERS:
            rs = sub(domain=d, tier=t)
            if not rs:
                continue
            P(f"          {t}  {len(rs):>3}     {acc(rs):.3f}  |"
              + "".join(f"   {st.mean(r[s] for r in rs):.3f} " for s in INTERNAL))
        rs = sub(domain=d)
        tiers = [r["tier"] for r in rs]
        P(f"   rho(accuracy, tier) = {spearman([r['correct'] for r in rs], tiers):+.3f}"
          "   <- P1 sanity: difficulty must be real (expect negative)")
        for s in INTERNAL:
            rho, lo, hi = bootstrap_ci(lambda a, b: spearman(a, b), [r[s] for r in rs], tiers)
            dr = dynamic_range([r[s] for r in rs])
            P(f"   rho({s:<11}, tier) = {rho:+.3f} [{lo:+.2f},{hi:+.2f}]"
              f"   range=[{dr['lo']:.2f},{dr['hi']:.2f}] sd={dr['sd']:.3f} uniq={dr['n_unique']}")

    # arithmetic-specific: carries as a tier-independent complexity measure
    ar = sub(domain="arithmetic")
    P("\nARITHMETIC ONLY — carries (exact, continuous, tier-independent complexity):")
    cs = [r["carries"] for r in ar]
    P(f"   carries range=[{min(cs)},{max(cs)}]   rho(accuracy, carries) = "
      f"{spearman([r['correct'] for r in ar], cs):+.3f}")
    for s in INTERNAL:
        rho, lo, hi = bootstrap_ci(lambda a, b: spearman(a, b), [r[s] for r in ar], cs)
        P(f"   rho({s:<11}, carries) = {rho:+.3f} [{lo:+.2f},{hi:+.2f}]")

    P("\nP4 — WITHIN a fixed arithmetic tier, does confidence still separate correct/incorrect?")
    for t in TIERS:
        rs = sub(domain="arithmetic", tier=t)
        y = [r["correct"] for r in rs]
        if 0 < sum(y) < len(y):
            P(f"   tier {t} (n={len(rs)}, acc={acc(rs):.2f})  selfconf AUROC="
              f"{auroc([r['selfconf'] for r in rs], y):.3f}   logprob AUROC="
              f"{auroc([r['logprob'] for r in rs], y):.3f}")
        else:
            P(f"   tier {t} (n={len(rs)}, acc={acc(rs):.2f})  -- all same label, AUROC undefined")

    # ---------------- Q1 overall + per-domain ----------------
    P("\n" + "=" * 96)
    P("Q1 — INFORMATION CONTENT (overall). K=5, so entropy != consistency now.")
    P("=" * 96)
    y_all = [r["correct"] for r in rows]
    P(f"{'signal':<16}{'AUROC':>17}{'Brier':>8}{'ECE':>7}{'MI (95% CI)':>22}")
    for s in SIGNALS:
        sc = [r[s] for r in rows]
        a, lo, hi = bootstrap_ci(auroc, sc, y_all)
        mi, ml, mh = bootstrap_ci(lambda x, l: mutual_info(x, l), sc, y_all)
        P(f"{s:<16}{a:>8.3f}[{lo:.2f},{hi:.2f}]{brier(sc, y_all):>8.3f}"
          f"{ece(sc, y_all):>7.3f}{mi:>10.3f}[{ml:.2f},{mh:.2f}]")

    P("\nAUROC per signal x domain:")
    P(f"{'signal':<16}" + "".join(f"{d:>14}" for d in DOMAINS))
    for s in SIGNALS:
        cells = []
        for d in DOMAINS:
            rs = sub(domain=d); y = [r["correct"] for r in rs]
            v = auroc([r[s] for r in rs], y) if 0 < sum(y) < len(y) else float("nan")
            cells.append(f"{v:>14.3f}" if v == v else f"{'--':>14}")
        P(f"{s:<16}" + "".join(cells))

    # ---------------- OBJECTIVES 3 & 4: all contrasts + empirical null ----------------
    P("\n" + "=" * 96)
    P("OBJECTIVES 3+4 — ALL PAIRWISE DOMAIN CONTRASTS, with the CONTROL as empirical null")
    P("=" * 96)
    pairs = [("factoid", "arithmetic"), ("factoid", "reasoning"), ("arithmetic", "reasoning")]
    for da, db in pairs:
        P(f"\n  {da} - {db}:")
        for s in SIGNALS:
            ra, rb = sub(domain=da), sub(domain=db)
            ya, yb = [r["correct"] for r in ra], [r["correct"] for r in rb]
            if not (0 < sum(ya) < len(ya) and 0 < sum(yb) < len(yb)):
                P(f"    {s:<14} -- undefined (a domain has one label only)"); continue
            diff, lo, hi = auroc_diff_ci([r[s] for r in ra], ya, [r[s] for r in rb], yb)
            tag = "DIFFERS" if (lo > 0 or hi < 0) else "inconclusive"
            note = "   <- EMPIRICAL NULL" if s == "shuffled_ctrl" else ""
            P(f"    {s:<14} {diff:+.3f} [{lo:+.2f},{hi:+.2f}]  {tag}{note}")

    P("\nAny real contrast must exceed the shuffled_ctrl row for the same pair.")
    P("\nDONE")


if __name__ == "__main__":
    main()
