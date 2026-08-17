"""M1 scientific dashboard: one per-benchmark signal map + the P1-P5 verdict table.

Reads tables/rq1_signal_benchmark.csv + raw rows (for per-benchmark best free feature)
and writes tables/m1_dashboard.md. Presentation only — no new statistics beyond folded
free-feature AUROC per benchmark (same fold rule as analyze.py's P5 gate).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT.parent))

from ugen.pipeline import freefeat, stats as S  # noqa: E402
from ugen.analysis.analyze import load_rows, attach_free_features  # noqa: E402

SIG_COLS = ["selfconf", "consistency", "logprob", "logprob_answer", "entropy"]


def best_ff(bs):
    best, name = 0.5, ""
    labs = bs["correct"].tolist()
    for n in freefeat.FEATURE_NAMES:
        col = f"ff_{n}"
        if col not in bs or bs[col].nunique() < 2:
            continue
        a = S.auroc(bs[col].tolist(), labs)
        f = max(a, 1 - a)
        if f > best:
            best, name = f, n
    return best, name


def main():
    rq1 = pd.read_csv(PROJECT / "tables" / "rq1_signal_benchmark.csv")
    verd = pd.read_csv(PROJECT / "tables" / "prediction_verdicts.csv")
    df, _ = attach_free_features(load_rows(PROJECT / "outputs"), PROJECT / "datasets")

    lines = ["# Signal dashboard — per model (n per cell in rq1 table)", "",
             "AUROC vs correctness per benchmark. `sat` = saturation-excluded (acc outside "
             "[0.10, 0.90]); `QC` = >10% extraction failure, excluded from primary "
             "contrasts. Signals: higher = better discrimination; 0.5 = chance.", ""]
    order = ["mmlu", "mmlu_pro", "truthfulqa", "gsm8k", "svamp", "multiarith",
             "commonsenseqa", "arc_challenge", "strategyqa", "bbh", "musr",
             "humaneval", "mbpp"]
    for model, mt in rq1.groupby("model"):
        lines += [f"## {model}", "",
                  "| Benchmark | Domain | Acc | flags | Self-conf | Consistency | Logprob | "
                  "Logprob(ans) | Entropy | Best free feat |",
                  "|---|---|---:|---|---:|---:|---:|---:|---:|---:|"]
        for b in order:
            bt = mt[mt.benchmark == b]
            if not len(bt):
                continue
            r0 = bt.iloc[0]
            flags = "+".join([f for f, on in (("sat", r0.saturated), ("QC", r0.ext_flagged))
                              if on]) or ""
            cell = {s: bt[bt.signal == s]["auroc"] for s in SIG_COLS}
            fmt = {s: f"{v.iloc[0]:.3f}" if len(v) else "—" for s, v in cell.items()}
            fa, fn = best_ff(df[(df.model == model) & (df.benchmark == b)])
            lines.append(
                f"| {b} | {r0.domain} | {r0.accuracy:.3f} | {flags} | {fmt['selfconf']} | "
                f"{fmt['consistency']} | {fmt['logprob']} | {fmt['logprob_answer']} | "
                f"{fmt['entropy']} | {fa:.3f} ({fn}) |")
        lines.append("")
    lines += ["## Preregistered prediction verdicts", "",
              "| Model | Prediction | Statistic | 95% CI / falsifiers | Verdict |",
              "|---|---|---|---|---|"]
    for _, r in verd.iterrows():
        lines.append(f"| {r.model} | {r.prediction} | {r.statistic} | "
                     f"{r.ci if pd.notna(r.ci) else ''} | **{r.verdict}** |")
    out = PROJECT / "tables" / "m1_dashboard.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
