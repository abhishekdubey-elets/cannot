"""Information decomposition for the P4 follow-up: does consistency carry correctness
information that entropy does not (and vice versa)?

Per model x domain (never pooled across domains; strata gate as in rq5):
  I(C;Y), I(E;Y)          — marginal mutual information
  I(C;Y|E), I(E;Y|C)      — unique information after conditioning on the other signal
each against its shuffled-control floor (plug-in MI is positively biased) with joint
bootstrap CIs. E is the P4 entropy (entropy_answer, answer-span token entropy) with the
full-chain `entropy` as a secondary variant. rho(C,E) ~ 0 alone cannot distinguish
"independent useful signals" from "unrelated poor predictors" — this can.

Usage: python analysis/analyze_cmi.py
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT.parent))

from ugen.pipeline import stats as S  # noqa: E402
from ugen.analysis.analyze import load_rows  # noqa: E402

COND_BINS = 4
MIN_N, MIN_MINORITY = 150, 30


def qbins(vals, q=COND_BINS):
    r = pd.Series(vals).rank(method="first")
    return pd.qcut(r, q=min(q, r.nunique()), labels=False).tolist()


def cmi_boot(sig, labs, cond, reps=500, seed=0):
    """Joint bootstrap (resample rows, rebin cond) 95% CI for I(sig;Y|cond)."""
    rng = random.Random(seed)
    n = len(sig)
    vals = []
    for _ in range(reps):
        idx = [rng.randrange(n) for _ in range(n)]
        vals.append(S.conditional_mutual_info([sig[i] for i in idx],
                                              [labs[i] for i in idx],
                                              qbins([cond[i] for i in idx])))
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[min(len(vals) - 1, int(0.975 * len(vals)))]


def decompose(sub, e_col):
    ok = sub.dropna(subset=["consistency", e_col])
    labs = ok["correct"].tolist()
    if len(ok) < MIN_N or min(sum(labs), len(labs) - sum(labs)) < MIN_MINORITY:
        return None
    C = ok["consistency"].tolist()
    E = (-ok[e_col]).tolist()          # orient: higher = more confident, cosmetic for MI
    out = dict(n=len(ok), rho_CE=S.spearman(C, E))
    for name, sig, cond in (("C", C, E), ("E", E, C)):
        mi, mi_lo, mi_hi = S.bootstrap_ci(S.mutual_info, sig, labs, reps=500)
        ctrl = S.mutual_info(S.shuffled_control(sig), labs)
        cm = S.conditional_mutual_info(sig, labs, qbins(cond))
        cm_lo, cm_hi = cmi_boot(sig, labs, cond)
        cctrl = S.conditional_mutual_info(S.shuffled_control(sig), labs, qbins(cond))
        out.update({f"I({name};Y)": S.mutual_info(sig, labs),
                    f"I({name};Y)_lo": mi_lo, f"I({name};Y)_hi": mi_hi,
                    f"I({name};Y)_ctrl": ctrl,
                    f"I({name};Y|{'E' if name == 'C' else 'C'})": cm,
                    f"cmi_{name}_lo": cm_lo, f"cmi_{name}_hi": cm_hi,
                    f"cmi_{name}_ctrl": cctrl})
    # unique-information verdict: CMI must clear both its shuffled floor and its CI lower
    # bound must exceed the control (conservative, control-referenced)
    out["unique_C"] = int(out["cmi_C_lo"] > out["cmi_C_ctrl"])
    out["unique_E"] = int(out["cmi_E_lo"] > out["cmi_E_ctrl"])
    return out


def main():
    df = load_rows(PROJECT / "outputs")
    rows = []
    for e_col in ("entropy_answer", "entropy"):
        for (model, domain), sub in df.groupby(["model", "domain"]):
            d = decompose(sub, e_col)
            if d is None:
                rows.append(dict(model=model, domain=domain, entropy_variant=e_col,
                                 status="underpowered"))
            else:
                rows.append(dict(model=model, domain=domain, entropy_variant=e_col,
                                 status="ok", **d))
    t = pd.DataFrame(rows)
    out = PROJECT / "tables" / "rq_cmi_decomposition"
    t.to_csv(f"{out}.csv", index=False)
    try:
        Path(f"{out}.md").write_text(t.to_markdown(index=False), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    cols = ["model", "domain", "entropy_variant", "n", "rho_CE", "I(C;Y)", "I(E;Y)",
            "I(C;Y|E)", "cmi_C_ctrl", "I(E;Y|C)", "cmi_E_ctrl", "unique_C", "unique_E"]
    ok = t[t.status == "ok"]
    pd.set_option("display.width", 250)
    print(ok[cols].to_string(index=False,
                             float_format=lambda x: f"{x:.4f}" if x == x else "nan"))
    print(f"\ntable -> {out}.csv")


if __name__ == "__main__":
    main()
