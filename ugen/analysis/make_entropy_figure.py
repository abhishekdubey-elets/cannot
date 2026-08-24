"""Main-text figure F6: 'Entropy is not one quantity — consistency and outcome-entropy
track correctness, token-entropy does not.'

Three panels, from frozen tables only (no new statistics beyond plotting transforms):
  A. Per-model AUROC of consistency vs vote-distribution (outcome) entropy vs token entropy.
  B. Information asymmetry I(C;Y|E) vs I(E;Y|C) per model x domain, with shuffled floors.
  C. The two entropies dissociate: Spearman rho(outcome-entropy, token-entropy) ~ 0.

Sources: tables/rq1_signal_benchmark.csv (consistency), tables/mechanism_entropy_variants.csv
(the two entropy AUROCs + their correlation), tables/rq_cmi_decomposition.csv (CMI, the
entropy_answer variant). 300 dpi PNG + vector PDF.

Run: /usr/local/bin/python3 analysis/make_entropy_figure.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
TABLES = PROJECT / "tables"
FIG = PROJECT / "figures"
MODELS = ["Qwen2.5-3B-Instruct", "Llama-3.2-3B-Instruct", "Gemma-3-4B-IT"]
SHORT = {"Qwen2.5-3B-Instruct": "Qwen2.5-3B",
         "Llama-3.2-3B-Instruct": "Llama-3.2-3B",
         "Gemma-3-4B-IT": "Gemma-3-4B"}
DOMS = ["knowledge", "arithmetic", "commonsense", "reasoning"]

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 300, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})

C_CONS = "#1b6ca8"    # consistency  (outcome / behavioural)
C_EDIST = "#4ba3c3"   # vote-distribution entropy (outcome)
C_TOK = "#d1495b"     # token entropy (token-level)


def consistency_auroc():
    """Mean clean-cell (unsaturated, extraction-clean) consistency AUROC per model."""
    r = pd.read_csv(TABLES / "rq1_signal_benchmark.csv")
    r = r[(r.signal == "consistency") & (r.saturated == 0) & (r.ext_flagged == 0)]
    return r.groupby("model")["auroc"].mean().to_dict()


def main():
    cons = consistency_auroc()
    ev = pd.read_csv(TABLES / "mechanism_entropy_variants.csv").set_index("model")
    cmi = pd.read_csv(TABLES / "rq_cmi_decomposition.csv")
    cmi = cmi[(cmi.entropy_variant == "entropy_answer") & (cmi.status == "ok")]

    fig = plt.figure(figsize=(13.0, 3.7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.5, 0.72], wspace=0.42)
    MI = {"Qwen2.5-3B-Instruct": "Q", "Llama-3.2-3B-Instruct": "L", "Gemma-3-4B-IT": "G"}

    # ---- Panel A: three signals' AUROC per model ----
    axA = fig.add_subplot(gs[0, 0])
    x = np.arange(len(MODELS)); w = 0.26
    consv = [cons[m] for m in MODELS]
    edist = [ev.loc[m, "auroc_ent_dist"] for m in MODELS]
    tok = [ev.loc[m, "auroc_token_ent"] for m in MODELS]
    axA.bar(x - w, consv, w, color=C_CONS, label="consistency")
    axA.bar(x, edist, w, color=C_EDIST, label="outcome entropy\n(K-vote distribution)")
    axA.bar(x + w, tok, w, color=C_TOK, label="token entropy\n(answer span)")
    axA.axhline(0.5, color="k", ls="--", lw=0.8)
    axA.set_ylim(0.45, 0.80)
    axA.set_xticks(x); axA.set_xticklabels([SHORT[m] for m in MODELS], rotation=20, ha="right")
    axA.set_ylabel("AUROC (correct vs incorrect)")
    axA.set_title("A. Outcome uncertainty tracks correctness;\ntoken uncertainty is weak and lineage-variable",
                  fontsize=8.5, loc="left")
    axA.legend(fontsize=6.5, loc="upper right", framealpha=0.9)

    # ---- Panel B: CMI asymmetry per model x domain ----
    axB = fig.add_subplot(gs[0, 1])
    labels, ic, ie, floor_c, floor_e = [], [], [], [], []
    for m in MODELS:
        for d in DOMS:
            row = cmi[(cmi.model == m) & (cmi.domain == d)]
            if row.empty:
                continue
            row = row.iloc[0]
            labels.append(f"{MI[m]}·{d[:4]}")
            ic.append(row["I(C;Y|E)"]); ie.append(row["I(E;Y|C)"])
            floor_c.append(row["cmi_C_ctrl"]); floor_e.append(row["cmi_E_ctrl"])
    xb = np.arange(len(labels)); wb = 0.38
    axB.bar(xb - wb / 2, ic, wb, color=C_CONS, label="I(consistency ; correct | entropy)")
    axB.bar(xb + wb / 2, ie, wb, color=C_TOK, label="I(entropy ; correct | consistency)")
    # shuffled-control floors as black ticks
    for xi, fc, fe in zip(xb, floor_c, floor_e):
        axB.hlines(fc, xi - wb, xi, color="k", lw=1.0)
        axB.hlines(fe, xi, xi + wb, color="k", lw=1.0)
    axB.set_xticks(xb); axB.set_xticklabels(labels, fontsize=6.8, rotation=45, ha="right")
    axB.set_ylabel("conditional MI (nats)")
    axB.set_title("B. Consistency carries unique correctness info beyond\ntoken entropy — not vice versa (Q/L/G = model; floors = ticks)",
                  fontsize=8.5, loc="left")
    axB.legend(fontsize=6.8, loc="upper right", framealpha=0.9)

    # ---- Panel C: the two entropies dissociate ----
    axC = fig.add_subplot(gs[0, 2])
    rho = [ev.loc[m, "rho_entdist_tokent"] for m in MODELS]
    colors = ["#6a8d92" for _ in MODELS]
    axC.barh(np.arange(len(MODELS)), rho, color=colors)
    axC.axvline(0, color="k", lw=0.8)
    axC.set_yticks(np.arange(len(MODELS)))
    axC.set_yticklabels([SHORT[m] for m in MODELS], fontsize=7.5)
    axC.set_xlim(-0.15, 0.4)
    axC.set_xlabel("Spearman ρ")
    axC.set_title("C. Outcome vs token\nentropy: ρ ≈ 0", fontsize=8.5, loc="left")
    for i, r in enumerate(rho):
        axC.text(r + (0.01 if r >= 0 else -0.01), i, f"{r:+.2f}",
                 va="center", ha="left" if r >= 0 else "right", fontsize=7)

    FIG.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"entropy_vs_consistency_main.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("figure -> entropy_vs_consistency_main.{png,pdf}")
    print(f"  consistency AUROC: {[round(v,3) for v in consv]}")
    print(f"  outcome  entropy : {[round(v,3) for v in edist]}")
    print(f"  token    entropy : {[round(v,3) for v in tok]}")
    print(f"  rho(out,tok)     : {[round(v,3) for v in rho]}")


if __name__ == "__main__":
    main()
