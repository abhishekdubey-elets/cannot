"""Phase C — mechanistic analysis of what consistency is capturing (and entropy is not).

All local, reads the frozen master rows + the samples_*.jsonl sidecars where available.
Outputs tables/mechanism_*.csv and a printed summary.

  1. Quadrants: P(Y=1 | C, E) with C banded (low <=0.4 / mid / high >=0.8) and E split at
     the per-benchmark median of entropy_answer — the direct look at C->Y vs E->Y.
  2. Wrong-but-consistent (C=1, Y=0): size, extraction share, benchmark profile — is
     convergent error a distinct population or a grading artifact?
  3. Confound structure: per-benchmark Spearman of C with generation_length, prompt size,
     and difficulty (where the benchmark provides it), summarized per model.
  4. Answer-distribution entropy vs token entropy (sidecars): are they even correlated,
     and which discriminates correctness?
  5. Sample-path divergence (sidecars): mean pairwise common-prefix fraction of the K
     samples. Among high-C items: is agreement reached via DIVERGENT paths (early split,
     same answer) more reliable than same-path agreement (possible mode collapse)?
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT.parent))

from ugen.pipeline import stats as S  # noqa: E402
from ugen.analysis.analyze import load_rows, attach_free_features  # noqa: E402

T = PROJECT / "tables"


def save(t, name):
    t.to_csv(T / f"{name}.csv", index=False)
    print(f"table -> {name} ({len(t)} rows)")


def c_band(c):
    return "C_high" if c >= 0.8 else ("C_low" if c <= 0.4 else "C_mid")


def quadrants(df):
    rows = []
    d = df.dropna(subset=["consistency", "entropy_answer"]).copy()
    d["e_med"] = d.groupby(["model", "benchmark"])["entropy_answer"].transform("median")
    d["E"] = np.where(d.entropy_answer > d.e_med, "E_high", "E_low")
    d["C"] = d.consistency.map(c_band)
    for (model, cb, eb), g in d.groupby(["model", "C", "E"]):
        top = g.benchmark.value_counts(normalize=True).head(3)
        rows.append(dict(model=model, C=cb, E=eb, n=len(g),
                         p_correct=g.correct.mean(),
                         top_benchmarks="; ".join(f"{b}:{p:.0%}" for b, p in top.items())))
    t = pd.DataFrame(rows).sort_values(["model", "C", "E"])
    save(t, "mechanism_quadrants")
    return t


def wrong_but_consistent(df):
    rows = []
    for model, g in df.groupby("model"):
        wc = g[(g.consistency == 1.0) & (g.correct == 0)]
        hi = g[g.consistency == 1.0]
        rows.append(dict(
            model=model, n_wrong_consistent=len(wc),
            pct_of_all=len(wc) / len(g),
            precision_at_C1=1 - len(wc) / max(1, len(hi)),
            extraction_ok_share=wc.extraction_ok.mean(),
            top_benchmarks="; ".join(f"{b}:{n}" for b, n in
                                     wc.benchmark.value_counts().head(4).items()),
            mean_prompt_words_vs_all=(wc.ff_prompt_words.mean() /
                                      g.ff_prompt_words.mean())))
    t = pd.DataFrame(rows)
    save(t, "mechanism_wrong_but_consistent")
    return t


def confounds(df):
    rows = []
    for model, g in df.groupby("model"):
        per_bench = {}
        for var in ("generation_length", "ff_prompt_words", "difficulty"):
            vals = []
            for b, bg in g.groupby("benchmark"):
                ok = bg.dropna(subset=["consistency", var]) if var in bg else bg.iloc[:0]
                if len(ok) > 50 and ok[var].nunique() > 3:
                    vals.append(S.spearman(ok.consistency.tolist(), ok[var].tolist()))
            per_bench[var] = vals
        rows.append(dict(model=model, **{
            f"rho_C_{v}": (f"{np.mean(vs):+.3f} [{min(vs):+.2f},{max(vs):+.2f}] "
                           f"({len(vs)} benches)") if vs else "n/a"
            for v, vs in per_bench.items()}))
    t = pd.DataFrame(rows)
    save(t, "mechanism_confounds")
    return t


def load_sidecars():
    recs = {}
    for f in PROJECT.glob("outputs/*/samples_*.jsonl"):
        if "ugen-s0-" in str(f):
            continue
        model = f.name.split("samples_")[1].rsplit("_", 1)[0]
        for line in open(f, encoding="utf-8"):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(d.get("seed", 0)) != "0":
                continue
            recs[(model, d["sample_id"])] = d
    for f in PROJECT.glob("outputs/*/gpu*/samples_*.jsonl"):
        model = f.name.split("samples_")[1].rsplit("_", 1)[0]
        for line in open(f, encoding="utf-8"):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            recs[(model, d["sample_id"])] = d
    return recs


def prefix_agreement(samples):
    """Mean pairwise common-prefix fraction over the K sampled texts."""
    ts = [s for s in samples if s]
    if len(ts) < 2:
        return np.nan
    fr = []
    for i in range(len(ts)):
        for j in range(i + 1, len(ts)):
            a, b = ts[i], ts[j]
            m = min(len(a), len(b))
            k = 0
            while k < m and a[k] == b[k]:
                k += 1
            fr.append(k / m if m else 0.0)
    return float(np.mean(fr))


def sidecar_analyses(df, recs):
    d = df.set_index(["model", "sample_id"])
    ent_rows, div_rows = [], []
    for model in df.model.unique():
        joined = []
        for (m, sid), rec in recs.items():
            if m != model or (m, sid) not in d.index:
                continue
            r = d.loc[(m, sid)]
            joined.append(dict(
                sample_id=sid, correct=int(r.correct), consistency=r.consistency,
                entropy_answer=r.entropy_answer,
                ent_dist=rec.get("entropy_answer_dist", np.nan),
                prefix_agree=prefix_agreement(rec.get("samples", []))))
        j = pd.DataFrame(joined).dropna(subset=["ent_dist"])
        if len(j) < 100:
            continue
        cov = len(j) / (df.model == model).sum()
        ok = j.dropna(subset=["entropy_answer"])
        labs = j.correct.tolist()
        ent_rows.append(dict(
            model=model, n=len(j), coverage=round(cov, 2),
            rho_entdist_tokent=S.spearman(ok.ent_dist.tolist(),
                                          ok.entropy_answer.tolist()),
            auroc_ent_dist=S.auroc((-j.ent_dist).tolist(), labs),
            auroc_token_ent=S.auroc((-ok.entropy_answer).tolist(), ok.correct.tolist())))
        p = j.dropna(subset=["prefix_agree"])
        hiC = p[p.consistency >= 0.8]
        med = hiC.prefix_agree.median()
        early = hiC[hiC.prefix_agree <= med]     # diverged early, still agreed
        late = hiC[hiC.prefix_agree > med]       # stayed on one path
        div_rows.append(dict(
            model=model, n_sidecar=len(p),
            rho_prefix_consistency=S.spearman(p.prefix_agree.tolist(),
                                              p.consistency.tolist()),
            p_correct_hiC_divergent_paths=early.correct.mean() if len(early) else np.nan,
            p_correct_hiC_same_path=late.correct.mean() if len(late) else np.nan,
            n_hiC=len(hiC)))
    save(pd.DataFrame(ent_rows), "mechanism_entropy_variants")
    save(pd.DataFrame(div_rows), "mechanism_path_divergence")


def main():
    df, _ = attach_free_features(load_rows(PROJECT / "outputs"), PROJECT / "datasets")
    df["difficulty"] = pd.to_numeric(df["difficulty"], errors="coerce")
    q = quadrants(df)
    pd.set_option("display.width", 220)
    print(q.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print()
    print(wrong_but_consistent(df).to_string(index=False,
                                             float_format=lambda x: f"{x:.3f}"))
    print()
    print(confounds(df).to_string(index=False))
    recs = load_sidecars()
    print(f"\nsidecar records: {len(recs)}")
    sidecar_analyses(df, recs)
    for name in ("mechanism_entropy_variants", "mechanism_path_divergence"):
        print(pd.read_csv(T / f"{name}.csv").to_string(index=False,
                                                       float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
