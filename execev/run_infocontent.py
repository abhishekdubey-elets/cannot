"""E1 (GPU) — how much correctness-information does each internal signal carry, and does it
generalise across domains? Pre-registered in ROADMAP.md.

Per item (CoT answering, uniform across domains): log-prob, consistency, entropy, self-conf,
plus a SHUFFLED-CONTROL signal that MUST land at the null (MI~0, AUROC~0.5) or the pipeline is
untrusted. Metrics per signal, overall AND per domain: AUROC, AUPRC, Brier, ECE, MI (bootstrap
95% CI). Decisive output: per-domain AUROC and the cross-domain differences with CIs.
"""
from __future__ import annotations

import math
import os
import random
import statistics as stats
import sys
from collections import Counter

sys.path.insert(0, ".")
sys.path.insert(0, "/kaggle/working/pkg")
from execev.benchmark import DOMAINS, graded, make_benchmark          # noqa: E402
from execev.stats import (auroc, auroc_diff_ci, auprc, bootstrap_ci,  # noqa: E402
                          brier, ece, mutual_info)

EVAL_SEEDS = [0, 1, 2]
N_PER_DOMAIN = 20
K = 3
COT = "Reason briefly, then put the final answer on the last line."
SIGNALS = ["logprob", "consistency", "entropy", "selfconf", "shuffled_ctrl"]


def _dump(name, txt):
    try:
        open(f"/kaggle/working/{name}", "w", encoding="utf-8").write(txt)
    except Exception:
        pass


def find_model():
    for dp, dns, fns in os.walk("/kaggle/input"):
        if "config.json" in fns and any(f.endswith((".safetensors", ".bin")) for f in fns):
            if "3b-instruct" in dp.lower():
                return dp
    return None


class LM:
    def __init__(self, path, device):
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

    def greedy_logprob(self, system, user, mnt=200):
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

    def sample(self, system, user, mnt=200):
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
    entropy_norm = 1 - ent / math.log(K)
    raw = lm.sample("Reply with only an integer 0 to 100.",
                    "How confident are you (0-100) that you know the correct answer?\n\n"
                    + item.prompt, mnt=6)
    m = __import__("re").search(r"\d{1,3}", raw)
    conf = min(int(m.group()), 100) / 100 if m else 0.0
    return ans, dict(logprob=lp, consistency=consistency, entropy=entropy_norm, selfconf=conf)


def main():
    lines = []

    def P(*a):
        m = " ".join(str(x) for x in a)
        print(m, flush=True); lines.append(m); _dump("results.txt", "\n".join(lines))

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    P(f"device={device}")
    mp = find_model(); P(f"model={mp}")
    if not mp:
        P("!! no model"); return
    lm = LM(mp, device)

    rows = []   # (domain, correct, sigdict)
    rng = random.Random(12345)
    for seed in EVAL_SEEDS:
        P(f"collecting seed {seed} ...")
        for it in make_benchmark(seed, N_PER_DOMAIN):
            ans, sig = signals_for(lm, it)
            sig["shuffled_ctrl"] = rng.random()          # null control
            rows.append((it.domain, int(graded(it, ans)), sig))

    P(f"\ncollected {len(rows)} items")
    acc = stats.mean(r[1] for r in rows)
    P(f"overall accuracy = {acc:.3f} "
      f"({'saturated - benchmark FAILED' if acc > 0.9 or acc < 0.1 else 'mid-range - non-saturating OK'})")
    for d in DOMAINS:
        sub = [r for r in rows if r[0] == d]
        P(f"  {d:<11} n={len(sub)}  accuracy={stats.mean(r[1] for r in sub):.3f}")

    def col(sig, subset):
        return [r[2][sig] for r in subset], [r[1] for r in subset]

    P("\n" + "=" * 92)
    P("Q1 — INFORMATION CONTENT per signal (overall). MI in nats; [.. ] = bootstrap 95% CI")
    P("=" * 92)
    P(f"{'signal':<16}{'AUROC':>16}{'AUPRC':>9}{'Brier':>8}{'ECE':>7}{'MI (95% CI)':>22}")
    y_all = [r[1] for r in rows]
    for s in SIGNALS:
        sc = [r[2][s] for r in rows]
        a, alo, ahi = bootstrap_ci(auroc, sc, y_all)
        mi, mlo, mhi = bootstrap_ci(lambda x, l: mutual_info(x, l), sc, y_all)
        P(f"{s:<16}{a:>7.3f}[{alo:.2f},{ahi:.2f}]{auprc(sc, y_all):>9.3f}"
          f"{brier(sc, y_all):>8.3f}{ece(sc, y_all):>7.3f}{mi:>10.3f}[{mlo:.2f},{mhi:.2f}]")

    P("\n  CONTROL CHECK: shuffled_ctrl AUROC must be ~0.50 and MI ~0.0 (else pipeline untrusted).")

    P("\n" + "=" * 92)
    P("H2 — DOES PREDICTIVE POWER GENERALISE ACROSS DOMAINS? (AUROC per signal x domain)")
    P("=" * 92)
    P(f"{'signal':<16}" + "".join(f"{d:>14}" for d in DOMAINS))
    dom_auroc = {}
    for s in SIGNALS:
        cells = []
        for d in DOMAINS:
            sub = [r for r in rows if r[0] == d]
            yc = [r[1] for r in sub]
            v = auroc([r[2][s] for r in sub], yc) if 0 < sum(yc) < len(yc) else float("nan")
            dom_auroc[(s, d)] = (v, sub)
            cells.append(f"{v:>14.3f}" if v == v else f"{'--':>14}")
        P(f"{s:<16}" + "".join(cells))
    P("  ('--' = that domain is all-correct or all-incorrect; no within-domain separation)")

    P("\nCROSS-DOMAIN DIFFERENCE (factoid vs reasoning), the decisive test, with bootstrap CI:")
    for s in ("logprob", "consistency", "selfconf"):
        (va, fa), (vb, fb) = dom_auroc[(s, "factoid")], dom_auroc[(s, "reasoning")]
        if va == va and vb == vb:
            d, lo, hi = auroc_diff_ci([r[2][s] for r in fa], [r[1] for r in fa],
                                      [r[2][s] for r in fb], [r[1] for r in fb])
            sig = "DIFFERS" if (lo > 0 or hi < 0) else "inconclusive (CI spans 0)"
            P(f"  {s:<12} factoid {va:.3f} - reasoning {vb:.3f} = {d:+.3f} [{lo:+.2f},{hi:+.2f}]  -> {sig}")

    P("\n" + "=" * 92)
    P("READING (pre-registered)")
    P("=" * 92)
    best = max(("logprob", "consistency", "selfconf"),
               key=lambda s: bootstrap_ci(auroc, [r[2][s] for r in rows], y_all)[0])
    P(f"  H1 signals carry info: best single = {best} "
      f"(AUROC {bootstrap_ci(auroc, [r[2][best] for r in rows], y_all)[0]:.3f})")
    P("  H2 domain-dependence: read the cross-domain differences above. A CI spanning 0 means")
    P("     the factoid finding GENERALISES on that signal; a CI clear of 0 means it does NOT.")


if __name__ == "__main__":
    main()
