"""Publication figures from raw rows + analysis tables. 300 dpi PNG + vector PDF.
Never computes new statistics beyond plotting transforms — numbers come from analyze.py
outputs or directly from raw rows via the same stats module.

Usage: python analysis/make_figures.py [--outputs DIR]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT.parent))
from ugen.analysis.analyze import SIGNALS, load_rows  # noqa: E402
from ugen.pipeline import stats as S  # noqa: E402

FIG = PROJECT / "figures"          # overridable via --figdir (dry-runs must not land here)
DOMAIN_ORDER = ["knowledge", "arithmetic", "commonsense", "reasoning", "coding"]

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 300, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"figure -> {name}")


def fig_domain_auroc(rq1):
    """Per-signal AUROC by domain with bootstrap CIs — the headline RQ2 figure."""
    for model in rq1["model"].unique():
        t = rq1[(rq1.model == model) & (rq1.saturated == 0)]
        agg = t.groupby(["domain", "signal"]).agg(
            auroc=("auroc", "mean"), lo=("auroc_lo", "mean"), hi=("auroc_hi", "mean"))
        sigs = [s for s in SIGNALS if s in t["signal"].unique()]
        doms = [d for d in DOMAIN_ORDER if d in t["domain"].unique()]
        fig, ax = plt.subplots(figsize=(7.0, 3.2))
        w = 0.8 / len(sigs)
        for k, sig in enumerate(sigs):
            xs, ys, es = [], [], []
            for i, d in enumerate(doms):
                if (d, sig) in agg.index:
                    r = agg.loc[(d, sig)]
                    xs.append(i + k * w)
                    ys.append(r.auroc)
                    es.append([[r.auroc - r.lo], [r.hi - r.auroc]])
            for x, y, e in zip(xs, ys, es):
                ax.errorbar(x, y, yerr=e, fmt="none", ecolor="gray", lw=0.8, capsize=1.5)
            ax.bar(xs, ys, width=w * 0.9, label=sig)
        ax.axhline(0.5, color="k", ls="--", lw=0.8)
        ax.set_xticks([i + 0.4 for i in range(len(doms))])
        ax.set_xticklabels(doms)
        ax.set_ylabel("AUROC (correct vs incorrect)")
        ax.set_ylim(0.3, 1.0)
        ax.set_title(f"Signal discrimination by domain — {model}")
        ax.legend(ncol=4, fontsize=7, frameon=False)
        save(fig, f"domain_auroc_{model}")


def fig_reliability(df):
    for model in df["model"].unique():
        sub = df[df["model"] == model]
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
        for ax, sig in zip(axes, ["selfconf", "consistency"]):
            col, _ = SIGNALS[sig]
            for domain, ds_ in sub.groupby("domain"):
                ok = ds_.dropna(subset=[col])
                if len(ok) < 50:
                    continue
                curve = S.reliability_curve(ok[col].clip(0, 1).tolist(),
                                            ok["correct"].tolist(), bins=8)
                ax.plot([c for _, c, _ in curve], [a for _, _, a in curve],
                        marker="o", ms=3, lw=1, label=domain)
            ax.plot([0, 1], [0, 1], "k--", lw=0.8)
            ax.set_xlabel(f"{sig} (predicted P(correct))")
            ax.set_ylabel("empirical accuracy")
            ax.set_title(sig)
        axes[0].legend(fontsize=7, frameon=False)
        fig.suptitle(f"Reliability by domain — {model}", y=1.03)
        save(fig, f"reliability_{model}")


def fig_roc_pr(df):
    for model in df["model"].unique():
        sub = df[df["model"] == model]
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))
        for sig, (col, orient) in SIGNALS.items():
            ok = sub.dropna(subset=[col])
            if len(ok) < 100:
                continue
            scores = (orient * ok[col]).to_numpy()
            labels = ok["correct"].to_numpy()
            order = np.argsort(-scores)
            l_sorted = labels[order]
            tpr = np.cumsum(l_sorted) / max(1, l_sorted.sum())
            fpr = np.cumsum(1 - l_sorted) / max(1, (1 - l_sorted).sum())
            prec = np.cumsum(l_sorted) / (np.arange(len(l_sorted)) + 1)
            axes[0].plot(fpr, tpr, lw=1, label=sig)
            axes[1].plot(tpr, prec, lw=1, label=sig)
        axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
        axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR"); axes[0].set_title("ROC")
        axes[1].set_xlabel("recall"); axes[1].set_ylabel("precision"); axes[1].set_title("PR")
        axes[0].legend(fontsize=6, frameon=False)
        fig.suptitle(f"Pooled ROC / PR — {model}", y=1.03)
        save(fig, f"roc_pr_{model}")


def fig_transfer_heatmap(tables_dir):
    f = Path(tables_dir) / "rq2_transfer_matrix.csv"
    if not f.exists():
        return
    t = pd.read_csv(f)
    for (model, sig), sub in t.groupby(["model", "signal"]):
        if sig not in ("selfconf", "consistency", "logprob"):
            continue
        mat = sub.pivot_table(index="train", columns="test", values="ece")
        fig, ax = plt.subplots(figsize=(5.6, 4.6))
        im = ax.imshow(mat.to_numpy(), cmap="viridis_r", aspect="auto")
        ax.set_xticks(range(len(mat.columns)), mat.columns, rotation=90, fontsize=6)
        ax.set_yticks(range(len(mat.index)), mat.index, fontsize=6)
        ax.set_xlabel("evaluated on"); ax.set_ylabel("calibrated on")
        fig.colorbar(im, label="ECE (lower=better)")
        ax.set_title(f"Calibration transfer — {sig}, {model}", fontsize=8)
        save(fig, f"transfer_ece_{sig}_{model}")


def fig_signal_corr(df):
    for (model, domain), sub in df.groupby(["model", "domain"]):
        pass  # per-domain matrices are in tables; figure shows pooled per model
    for model in df["model"].unique():
        sub = df[df["model"] == model]
        cols, names = [], []
        for sig, (col, orient) in SIGNALS.items():
            cols.append(orient * sub[col])
            names.append(sig)
        c = pd.concat(cols, axis=1).corr(method="spearman").to_numpy()
        fig, ax = plt.subplots(figsize=(4.6, 4.0))
        im = ax.imshow(c, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(names)), names, rotation=90, fontsize=7)
        ax.set_yticks(range(len(names)), names, fontsize=7)
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, f"{c[i, j]:.2f}", ha="center", va="center", fontsize=5)
        fig.colorbar(im, label="Spearman rho")
        ax.set_title(f"Signal correlations — {model}", fontsize=9)
        save(fig, f"signal_corr_{model}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default=str(PROJECT / "outputs"))
    ap.add_argument("--tables", default=str(PROJECT / "tables"))
    ap.add_argument("--figdir", default=None)
    a = ap.parse_args()
    if a.figdir:
        global FIG
        FIG = Path(a.figdir)
    df = load_rows(a.outputs)
    rq1_f = Path(a.tables) / "rq1_signal_benchmark.csv"
    if rq1_f.exists():
        fig_domain_auroc(pd.read_csv(rq1_f))
    fig_reliability(df)
    fig_roc_pr(df)
    fig_transfer_heatmap(a.tables)
    fig_signal_corr(df)
    print("figures complete")


if __name__ == "__main__":
    main()
