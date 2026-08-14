"""Wording-sensitivity probe analysis (PLAN 9.9, supplementary — never feeds primary tables).

Per wording x benchmark: response distribution (anchoring check), parse rate, AUROC vs
correctness with bootstrap CI. The decisive contrasts:
  - if every wording collapses to the same constant -> collapse is not wording-driven;
  - if w3 (inverted scale) also elicits "100" -> the model anchors on the number, and its
    raw replies do not track the question's semantics at all.

Usage: python analysis/analyze_probe.py [--csv PATH]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT.parent))

from ugen.pipeline import stats as S  # noqa: E402

WIDS = ["w0_original", "w1_probability", "w2_frequency", "w3_inverted"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()
    csv = a.csv or next(Path(PROJECT / "outputs").rglob("probe_wording_*.csv"), None)
    if csv is None:
        raise SystemExit("no probe_wording_*.csv under outputs/")
    df = pd.read_csv(csv)
    print(f"{csv}\n{len(df)} rows | acc mmlu="
          f"{df[df.benchmark == 'mmlu']['correct'].mean():.3f} gsm8k="
          f"{df[df.benchmark == 'gsm8k']['correct'].mean():.3f}\n")
    rows = []
    for wid in WIDS:
        c, r = f"conf_{wid}", f"raw_{wid}"
        raw = df[r].dropna()
        top = raw.value_counts(normalize=True).head(3)
        modal = ", ".join(f"{v:g}:{p * 100:.0f}%" for v, p in top.items())
        for bench, g in df.groupby("benchmark"):
            ok = g.dropna(subset=[c])
            if len(ok) < 20 or ok["correct"].nunique() < 2:
                auroc, lo, hi = np.nan, np.nan, np.nan
            else:
                auroc, lo, hi = S.bootstrap_ci(S.auroc, ok[c].tolist(),
                                               ok["correct"].tolist(), reps=1000, seed=0)
                auroc = S.auroc(ok[c].tolist(), ok["correct"].tolist())
            rows.append(dict(
                wording=wid, benchmark=bench, n=len(g),
                parse_rate=g[c].notna().mean(),
                mean_conf_correct=g[g.correct == 1][c].mean(),
                mean_conf_wrong=g[g.correct == 0][c].mean(),
                pct_raw_eq_100=(g[r] == 100).mean(),
                pct_raw_eq_0=(g[r] == 0).mean(),
                auroc=auroc, auroc_lo=lo, auroc_hi=hi,
                modal_raw_values=modal))
    t = pd.DataFrame(rows)
    out = PROJECT / "tables" / "probe_wording_summary"
    (PROJECT / "tables").mkdir(exist_ok=True)
    t.to_csv(f"{out}.csv", index=False)
    try:
        Path(f"{out}.md").write_text(t.to_markdown(index=False), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    pd.set_option("display.width", 200)
    print(t.to_string(index=False,
                      float_format=lambda x: f"{x:.3f}" if x == x else "nan"))
    print(f"\ntable -> {out}.csv")


if __name__ == "__main__":
    main()
