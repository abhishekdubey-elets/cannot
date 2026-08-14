"""UGEN offline analysis: RQ1-RQ5 over raw row CSVs produced by kernels.

Reads outputs/**/rows_*_.csv (never edits them), merges frozen items for free features,
writes tables/ (csv+md+tex) and analysis intermediates. Figures: make_figures.py.

Usage: python analysis/analyze.py [--outputs DIR]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT.parent))

from ugen.pipeline import freefeat, stats as S  # noqa: E402

# signal -> (column, orientation) ; orientation +1 means higher = more confident
SIGNALS = {
    "selfconf": ("self_confidence", +1),
    "consistency": ("consistency", +1),
    "logprob": ("mean_logprob", +1),
    "logprob_answer": ("mean_logprob_answer", +1),
    "entropy": ("entropy", -1),
    "entropy_answer": ("entropy_answer", -1),
    "token_margin": ("token_margin", +1),
    "topk_mass": ("topk_mass", +1),
}
PROB_SIGNALS = {"selfconf", "consistency"}          # already in [0,1]
PRIMARY_CONTRASTS = [("knowledge", "arithmetic"), ("knowledge", "reasoning"),
                     ("arithmetic", "commonsense")]
SATURATION = (0.10, 0.90)


def load_rows(outputs_dir):
    frames = []
    seen = set()
    for f in sorted(Path(outputs_dir).rglob("rows_*_ALL.csv")):
        frames.append(pd.read_csv(f))
        seen.add(f.parent)
    # fall back to per-benchmark files for runs that died before writing ALL
    for f in sorted(Path(outputs_dir).rglob("rows_*.csv")):
        if f.name.endswith("_ALL.csv") or ".partial." in f.name or f.parent in seen:
            continue
        frames.append(pd.read_csv(f))
    if not frames:
        raise SystemExit(f"no rows CSVs under {outputs_dir}")
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["sample_id", "model", "seed"], keep="first")
    df["correct"] = df["correct"].astype(int)
    return df


def attach_free_features(df, datasets_dir):
    items = {}
    for f in Path(datasets_dir).glob("*.jsonl"):
        for line in open(f, encoding="utf-8"):
            it = json.loads(line)
            items[it["sample_id"]] = it
    feats = {sid: freefeat.features(it) for sid, it in items.items()}
    for name in freefeat.FEATURE_NAMES:
        df[f"ff_{name}"] = df["sample_id"].map(
            lambda s, n=name: feats.get(s, {}).get(n, np.nan))
    return df, items


def cell_metrics(sub, sig_col, orient, prob_like, seed=0):
    ok = sub.dropna(subset=[sig_col])
    if len(ok) < 20 or ok["correct"].nunique() < 2:
        return None
    scores = (orient * ok[sig_col]).tolist()
    labels = ok["correct"].tolist()
    a, lo, hi = S.bootstrap_ci(S.auroc, scores, labels, reps=1000, seed=seed)
    out = dict(n=len(ok), n_missing=len(sub) - len(ok),
               auroc=S.auroc(scores, labels), auroc_lo=lo, auroc_hi=hi,
               auprc=S.auprc(scores, labels),
               mi=S.mutual_info(scores, labels))
    mi_m, mi_lo, mi_hi = S.bootstrap_ci(lambda s, l: S.mutual_info(s, l), scores, labels,
                                        reps=500, seed=seed)
    out.update(mi_lo=mi_lo, mi_hi=mi_hi)
    if prob_like:
        probs = ok[sig_col].clip(0, 1).tolist()
        out.update(brier=S.brier(probs, labels), nll=S.nll(probs, labels),
                   ece=S.ece(probs, labels, bins=10, quantile=True),
                   mce=S.mce(probs, labels, bins=10, quantile=True))
    else:
        iso = S.isotonic_fit(scores, labels)
        probs = S.isotonic_apply(iso, scores)
        out.update(brier=S.brier(probs, labels), nll=S.nll(probs, labels),
                   ece=S.ece(probs, labels, bins=10, quantile=True),
                   mce=S.mce(probs, labels, bins=10, quantile=True))
    ctrl = S.shuffled_control(scores, seed=seed)
    out["ctrl_auroc"] = S.auroc(ctrl, labels)
    out["ctrl_mi"] = S.mutual_info(ctrl, labels)
    return out


def rq1_tables(df, tables_dir):
    rows = []
    for (model, bench), sub in df.groupby(["model", "benchmark"]):
        acc = sub["correct"].mean()
        for sig, (col, orient) in SIGNALS.items():
            m = cell_metrics(sub, col, orient, sig in PROB_SIGNALS)
            if m is None:
                continue
            rows.append(dict(model=model, benchmark=bench, domain=sub["domain"].iloc[0],
                             signal=sig, accuracy=acc,
                             saturated=int(not SATURATION[0] <= acc <= SATURATION[1]), **m))
    t = pd.DataFrame(rows)
    write_table(t, tables_dir, "rq1_signal_benchmark")
    return t


def domain_pooled(df, model):
    """Per-signal per-domain pooled scores, excluding saturated benchmarks (PLAN 7)."""
    sub = df[df["model"] == model]
    acc = sub.groupby("benchmark")["correct"].mean()
    ok_bench = acc[(acc >= SATURATION[0]) & (acc <= SATURATION[1])].index
    return sub[sub["benchmark"].isin(ok_bench)]


def rq2_contrasts(df, tables_dir):
    rows, pvals = [], []
    for model in df["model"].unique():
        pooled = domain_pooled(df, model)
        for sig, (col, orient) in SIGNALS.items():
            for da, db in PRIMARY_CONTRASTS:
                A = pooled[(pooled["domain"] == da)].dropna(subset=[col])
                B = pooled[(pooled["domain"] == db)].dropna(subset=[col])
                if len(A) < 50 or len(B) < 50:
                    continue
                sa, la = (orient * A[col]).tolist(), A["correct"].tolist()
                sb, lb = (orient * B[col]).tolist(), B["correct"].tolist()
                d, lo, hi = S.diff_ci(S.auroc, sa, la, sb, lb, reps=1000)
                obs, p = S.permutation_pvalue(S.auroc, sa, la, sb, lb, perms=2000)
                primary = int(sig in ("selfconf", "consistency", "logprob") and
                              (da, db) == ("knowledge", "arithmetic"))
                rows.append(dict(model=model, signal=sig, contrast=f"{da}-{db}",
                                 auroc_a=S.auroc(sa, la), auroc_b=S.auroc(sb, lb),
                                 delta=obs, delta_lo=lo, delta_hi=hi, p_perm=p,
                                 primary=primary))
                pvals.append(p if primary else np.nan)
    t = pd.DataFrame(rows)
    if len(t):
        prim = t["primary"] == 1
        adj = S.holm(t.loc[prim, "p_perm"].tolist())
        t.loc[prim, "p_holm"] = adj
    write_table(t, tables_dir, "rq2_domain_contrasts")
    return t


def rq2_transfer_matrix(df, tables_dir):
    """Calibration transfer: isotonic fit on benchmark A -> ECE/Brier on B."""
    out = []
    for model in df["model"].unique():
        sub = df[df["model"] == model]
        benches = sorted(sub["benchmark"].unique())
        for sig, (col, orient) in SIGNALS.items():
            for ba in benches:
                A = sub[sub["benchmark"] == ba].dropna(subset=[col])
                if len(A) < 50 or A["correct"].nunique() < 2:
                    continue
                iso = S.isotonic_fit((orient * A[col]).tolist(), A["correct"].tolist())
                thr, _, _ = S.selective_threshold(
                    S.isotonic_apply(iso, (orient * A[col]).tolist()),
                    A["correct"].tolist(), 0.8)
                for bb in benches:
                    B = sub[sub["benchmark"] == bb].dropna(subset=[col])
                    if len(B) < 50 or B["correct"].nunique() < 2:
                        continue
                    probs = S.isotonic_apply(iso, (orient * B[col]).tolist())
                    labels = B["correct"].tolist()
                    sel = [(p, y) for p, y in zip(probs, labels) if p >= thr]
                    out.append(dict(
                        model=model, signal=sig, train=ba, test=bb,
                        ece=S.ece(probs, labels, quantile=True),
                        brier=S.brier(probs, labels),
                        auroc=S.auroc(probs, labels),
                        sel_coverage=len(sel) / len(labels),
                        sel_precision=(sum(y for _, y in sel) / len(sel)) if sel else np.nan))
    t = pd.DataFrame(out)
    write_table(t, tables_dir, "rq2_transfer_matrix")
    return t


def rq3_cross_model(rq1, tables_dir):
    """Signal quality profiles across models: benchmark-AUROC vector correlations."""
    rows = []
    models = rq1["model"].unique()
    for sig in rq1["signal"].unique():
        for i, ma in enumerate(models):
            for mb in models[i + 1:]:
                A = rq1[(rq1.model == ma) & (rq1.signal == sig)].set_index("benchmark")["auroc"]
                B = rq1[(rq1.model == mb) & (rq1.signal == sig)].set_index("benchmark")["auroc"]
                common = A.index.intersection(B.index)
                if len(common) < 4:
                    continue
                rows.append(dict(signal=sig, model_a=ma, model_b=mb, n_bench=len(common),
                                 spearman=S.spearman(A[common].tolist(), B[common].tolist())))
    t = pd.DataFrame(rows)
    write_table(t, tables_dir, "rq3_cross_model")
    return t


def rq4_disagreement(df, tables_dir, outputs_dir):
    """Pre-defined conflict patterns on within-benchmark z-scored signals."""
    z = df.copy()
    for sig, (col, orient) in SIGNALS.items():
        z[f"z_{sig}"] = orient * z.groupby(["model", "benchmark"])[col].transform(
            lambda x: (x - x.mean()) / (x.std() or 1))
    pats = {
        "selfconf_hi_logprob_lo": (z["z_selfconf"] > 1) & (z["z_logprob"] < -1),
        "consist_hi_entropy_hi": (z["z_consistency"] > 1) & (z["z_entropy"] > 1),
        "entropy_lo_wrong": (z["z_entropy"] < -1) & (z["correct"] == 0),
        "logprob_hi_wrong": (z["z_logprob"] > 1) & (z["correct"] == 0),
        "selfconf_lo_correct": (z["z_selfconf"] < -1) & (z["correct"] == 1),
    }
    rows = []
    ex_path = Path(outputs_dir) / "disagreement_examples.jsonl"
    with open(ex_path, "w", encoding="utf-8") as fh:
        for name, mask in pats.items():
            sel = z[mask]
            rows.append(dict(pattern=name, n=len(sel), frac=len(sel) / max(1, len(z)),
                             acc=sel["correct"].mean() if len(sel) else np.nan,
                             top_domains=";".join(sel["domain"].value_counts().head(3).index)))
            for _, r in sel.head(40).iterrows():
                fh.write(json.dumps(dict(pattern=name, sample_id=r["sample_id"],
                                         model=r["model"], benchmark=r["benchmark"],
                                         correct=int(r["correct"]),
                                         prediction=str(r["prediction"])[:200]),
                                    ensure_ascii=False) + "\n")
    t = pd.DataFrame(rows)
    write_table(t, tables_dir, "rq4_disagreement_patterns")
    # signal correlation matrix per domain
    corr_rows = []
    for (model, domain), sub in z.groupby(["model", "domain"]):
        cols = [f"z_{s}" for s in SIGNALS]
        c = sub[cols].corr(method="spearman")
        for i, si in enumerate(SIGNALS):
            for j, sj in enumerate(SIGNALS):
                if i < j:
                    corr_rows.append(dict(model=model, domain=domain, sig_a=si, sig_b=sj,
                                          rho=c.iloc[i, j]))
    write_table(pd.DataFrame(corr_rows), tables_dir, "rq4_signal_correlations")
    return t


def rq5_information(df, tables_dir):
    """Free-feature gate (P5) + CMI + PID per domain x model."""
    rows, pid_rows = [], []
    for (model, domain), sub in df.groupby(["model", "domain"]):
        labels = sub["correct"].tolist()
        if len(sub) < 150 or min(sum(labels), len(labels) - sum(labels)) < 30:
            rows.append(dict(model=model, domain=domain, status="underpowered — not run"))
            continue
        # best free feature by |AUROC-0.5|
        best_ff, best_a = None, 0.5
        for name in freefeat.FEATURE_NAMES:
            col = f"ff_{name}"
            if sub[col].nunique() < 2:
                continue
            a = S.auroc(sub[col].tolist(), labels)
            for cand in (a, 1 - a):
                if abs(cand - 0.5) > abs(best_a - 0.5):
                    best_a, best_ff = cand, (name if cand == a else f"-{name}")
        ff_col = f"ff_{best_ff.lstrip('-')}" if best_ff else None
        cond = None
        if ff_col:
            vals = sub[ff_col]
            cond = pd.qcut(vals.rank(method="first"), q=min(4, vals.nunique()),
                           labels=False).tolist()
        for sig, (col, orient) in SIGNALS.items():
            ok = sub.dropna(subset=[col])
            if len(ok) < 100:
                continue
            scores = (orient * ok[col]).tolist()
            labs = ok["correct"].tolist()
            r = dict(model=model, domain=domain, signal=sig, status="ok",
                     auroc=S.auroc(scores, labs), mi=S.mutual_info(scores, labs),
                     best_free_feature=best_ff, ff_auroc=best_a)
            if cond is not None:
                cond_ok = [cond[i] for i in range(len(sub)) if not pd.isna(sub[col].iloc[i])]
                r["cmi_given_ff"] = S.conditional_mutual_info(scores, labs, cond_ok)
                ctrl = S.shuffled_control(scores)
                r["ctrl_cmi_given_ff"] = S.conditional_mutual_info(ctrl, labs, cond_ok)
            rows.append(r)
        # PID: selfconf x consistency (primary), logprob x selfconf (secondary)
        for s1, s2 in (("selfconf", "consistency"), ("logprob", "selfconf")):
            c1, o1 = SIGNALS[s1]
            c2, o2 = SIGNALS[s2]
            ok = sub.dropna(subset=[c1, c2])
            if len(ok) < 150:
                continue
            x1 = (o1 * ok[c1]).tolist()
            x2 = (o2 * ok[c2]).tolist()
            labs = ok["correct"].tolist()
            d = S.pid_williams_beer(x1, x2, labs, bins=3)
            dc = S.pid_williams_beer(S.shuffled_control(x1), S.shuffled_control(x2, seed=1),
                                     labs, bins=3)
            pid_rows.append(dict(model=model, domain=domain, sources=f"{s1}x{s2}",
                                 **{k: v for k, v in d.items()},
                                 **{f"ctrl_{k}": v for k, v in dc.items()}))
    write_table(pd.DataFrame(rows), tables_dir, "rq5_free_feature_gate")
    write_table(pd.DataFrame(pid_rows), tables_dir, "rq5_pid")
    return rows, pid_rows


def prediction_verdicts(rq1, rq2, tables_dir, df):
    """P1-P5 verdict table — the pre-registered tests, evaluated mechanically."""
    v = []
    for model in rq2["model"].unique():
        t = rq2[(rq2.model == model)]
        p1 = t[(t.signal == "selfconf") & (t.contrast == "knowledge-arithmetic")]
        if len(p1):
            r = p1.iloc[0]
            verdict = ("SUPPORTED" if r.delta >= 0.15 and r.delta_lo > 0 else
                       "FALSIFIED" if r.delta < 0.10 or r.delta_lo <= 0 else "INCONCLUSIVE")
            v.append(dict(model=model, prediction="P1", statistic=f"dAUROC={r.delta:+.3f}",
                          ci=f"[{r.delta_lo:+.3f},{r.delta_hi:+.3f}]", verdict=verdict))
        pc = t[(t.signal == "consistency") & (t.contrast == "knowledge-arithmetic")]
        if len(p1) and len(pc):
            d_self, d_cons = p1.iloc[0].delta, pc.iloc[0].delta
            v.append(dict(model=model, prediction="P2",
                          statistic=f"d_cons={d_cons:+.3f} vs d_self={d_self:+.3f}",
                          ci="", verdict="SUPPORTED" if d_cons < d_self else "FALSIFIED"))
        arith = rq1[(rq1.model == model) & (rq1.domain == "arithmetic") &
                    (rq1.signal == "logprob") & (rq1.saturated == 0)]
        arith_ans = rq1[(rq1.model == model) & (rq1.domain == "arithmetic") &
                        (rq1.signal == "logprob_answer") & (rq1.saturated == 0)]
        if len(arith):
            full = arith["auroc"].mean()
            ans = arith_ans["auroc"].mean() if len(arith_ans) else np.nan
            verdict = "SUPPORTED" if full < 0.6 and (np.isnan(ans) or ans > full) else \
                      "FALSIFIED" if full >= 0.6 else "PARTIAL"
            v.append(dict(model=model, prediction="P3",
                          statistic=f"logprob_full={full:.3f}, answer={ans:.3f}",
                          ci="", verdict=verdict))
        # P4: consistency vs entropy_answer redundancy per benchmark
        sub = df[df["model"] == model]
        rhos = []
        for b, bs in sub.groupby("benchmark"):
            ok = bs.dropna(subset=["consistency", "entropy_answer"])
            if len(ok) > 30:
                rhos.append(S.spearman(ok["consistency"].tolist(),
                                       (-ok["entropy_answer"]).tolist()))
        if rhos:
            mn = min(rhos)
            v.append(dict(model=model, prediction="P4", statistic=f"min_rho={mn:.3f}",
                          ci="", verdict="SUPPORTED" if mn > 0.9 else "FALSIFIED"))
        # P5 (preregistered falsifiers, benchmark-level): internal beats best free feature
        # on GSM8K/SVAMP, or a free feature beats selfconf on MMLU/TruthfulQA. Saturated
        # benchmarks (acc outside SATURATION) are excluded, as in every primary test.
        def _best_ff_auroc(bs):
            best = 0.5
            labs = bs["correct"].tolist()
            for name in freefeat.FEATURE_NAMES:
                col = f"ff_{name}"
                if col not in bs or bs[col].nunique() < 2:
                    continue
                a = S.auroc(bs[col].tolist(), labs)
                best = max(best, a, 1 - a)   # folded: direction-free discrimination
            return best

        def _sig_auroc(bs, sig):
            col, orient = SIGNALS[sig]
            ok = bs.dropna(subset=[col])
            if len(ok) < 100:
                return np.nan
            return S.auroc((orient * ok[col]).tolist(), ok["correct"].tolist())

        stats_bits, falsified = [], []
        for b in ("gsm8k", "svamp"):
            bs = sub[sub["benchmark"] == b]
            if not len(bs) or not SATURATION[0] <= bs["correct"].mean() <= SATURATION[1]:
                continue
            ff = _best_ff_auroc(bs)
            internal = max((a for s in SIGNALS if not np.isnan(a := _sig_auroc(bs, s))),
                           default=np.nan)
            stats_bits.append(f"{b}: ff={ff:.3f} best_int={internal:.3f}")
            if not np.isnan(internal) and internal > ff:
                falsified.append(f"internal>ff on {b}")
        for b in ("mmlu", "truthfulqa"):
            bs = sub[sub["benchmark"] == b]
            if not len(bs) or not SATURATION[0] <= bs["correct"].mean() <= SATURATION[1]:
                continue
            ff = _best_ff_auroc(bs)
            sc = _sig_auroc(bs, "selfconf")
            stats_bits.append(f"{b}: ff={ff:.3f} selfconf={sc:.3f}")
            if not np.isnan(sc) and ff > sc:
                falsified.append(f"ff>selfconf on {b}")
        if stats_bits:
            v.append(dict(model=model, prediction="P5",
                          statistic="; ".join(stats_bits)[:200],
                          ci="; ".join(falsified) if falsified else "",
                          verdict="FALSIFIED" if falsified else "SUPPORTED"))
    t = pd.DataFrame(v)
    write_table(t, tables_dir, "prediction_verdicts")
    return t


def write_table(t, tables_dir, name):
    tables_dir = Path(tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    t.to_csv(tables_dir / f"{name}.csv", index=False)
    try:
        (tables_dir / f"{name}.md").write_text(t.to_markdown(index=False), encoding="utf-8")
        (tables_dir / f"{name}.tex").write_text(
            t.to_latex(index=False, float_format="%.3f"), encoding="utf-8")
    except Exception:  # noqa: BLE001 — markdown/latex are conveniences, CSV is the record
        pass
    print(f"table -> {name} ({len(t)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default=str(PROJECT / "outputs"))
    ap.add_argument("--datasets", default=str(PROJECT / "datasets"))
    ap.add_argument("--tables", default=str(PROJECT / "tables"))
    a = ap.parse_args()
    df = load_rows(a.outputs)
    df, _ = attach_free_features(df, a.datasets)
    print(f"{len(df)} rows, models={list(df['model'].unique())}, "
          f"benchmarks={df['benchmark'].nunique()}")
    rq1 = rq1_tables(df, a.tables)
    rq2 = rq2_contrasts(df, a.tables)
    rq2_transfer_matrix(df, a.tables)
    rq3_cross_model(rq1, a.tables)
    rq4_disagreement(df, a.tables, a.outputs)
    rq5_information(df, a.tables)
    prediction_verdicts(rq1, rq2, a.tables, df)
    print("analysis complete — tables/ updated. Figures: python analysis/make_figures.py")


if __name__ == "__main__":
    main()
