"""Scaling regression (PLAN.md section 4): which factors explain uncertainty quality?

Model A (item level): logistic  correct ~ z_signal * domain + log_prompt_len + model,
    cluster-robust SEs by benchmark. One fit per signal.
Model B (cell level): OLS  AUROC ~ domain + model, weighted by cell n.

Observational, exploratory — reported with CIs and effect sizes, never as causal claims.

Usage: python analysis/regression.py [--outputs DIR]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT.parent))
from ugen.analysis.analyze import SIGNALS, load_rows, write_table  # noqa: E402


def item_level(df, tables_dir):
    import statsmodels.formula.api as smf
    rows = []
    for sig, (col, orient) in SIGNALS.items():
        sub = df.dropna(subset=[col]).copy()
        if len(sub) < 500:
            continue
        sub["z_sig"] = orient * sub.groupby(["model", "benchmark"])[col].transform(
            lambda x: (x - x.mean()) / (x.std() or 1))
        sub["log_len"] = np.log1p(sub["generation_length"].astype(float))
        try:
            m = smf.logit("correct ~ z_sig * C(domain) + log_len + C(model)", sub).fit(
                disp=0, cov_type="cluster", cov_kwds=dict(groups=sub["benchmark"]))
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {sig}: {e}")
            continue
        for name, coef in m.params.items():
            if "z_sig" not in name:
                continue
            lo, hi = m.conf_int().loc[name]
            rows.append(dict(signal=sig, term=name, coef=coef, or_=np.exp(coef),
                             lo=lo, hi=hi, p=m.pvalues[name], n=int(m.nobs)))
    write_table(pd.DataFrame(rows), tables_dir, "regression_item_level")


def cell_level(tables_dir):
    import statsmodels.formula.api as smf
    f = Path(tables_dir) / "rq1_signal_benchmark.csv"
    if not f.exists():
        return
    t = pd.read_csv(f)
    t = t[t["saturated"] == 0]
    rows = []
    for sig, sub in t.groupby("signal"):
        if sub["model"].nunique() < 1 or len(sub) < 8:
            continue
        try:
            m = smf.wls("auroc ~ C(domain) + C(model)", sub, weights=sub["n"]).fit()
        except Exception as e:  # noqa: BLE001
            print(f"[skip cell] {sig}: {e}")
            continue
        for name, coef in m.params.items():
            lo, hi = m.conf_int().loc[name]
            rows.append(dict(signal=sig, term=name, coef=coef, lo=lo, hi=hi,
                             p=m.pvalues[name], r2=m.rsquared, n_cells=len(sub)))
    write_table(pd.DataFrame(rows), tables_dir, "regression_cell_level")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default=str(PROJECT / "outputs"))
    ap.add_argument("--tables", default=str(PROJECT / "tables"))
    a = ap.parse_args()
    df = load_rows(a.outputs)
    item_level(df, a.tables)
    cell_level(a.tables)


if __name__ == "__main__":
    main()
