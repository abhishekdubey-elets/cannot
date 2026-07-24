# Executive Evidence — Research Roadmap

A measurement-first program. **Measurement precedes mechanism; benchmarks precede
architectures; falsification precedes optimization.** No architecture is proposed until
repeated measurement establishes empirical regularities. Every experiment carries: null
hypothesis, expected effect size, power analysis, pre-registration, verification script,
confidence intervals, failure analysis, threats to validity.

## Closed hypotheses (do not rescue)

Refuted and retired: behavioural calibration always beats introspective; internal evidence is
information-theoretically blind to stale knowledge; more fused signals automatically help;
synthetic validation can validate a scientific hypothesis.

## Sequencing (each gate must pass before the next is built)

| Phase | Question | Precondition it establishes | Status |
|---|---|---|---|
| **E1** | Q1 — how much correctness-information does each internal signal carry, **and does it generalise across domains**? | that the signals are worth decomposing at all, and whether findings are domain-specific | **THIS PHASE** |
| E2 | Q2 — redundancy vs synergy among signals (PID / interaction information) | which signals are worth combining | gated on E1 |
| E3 | Q3 — disagreement-maximising multi-domain benchmark (expanded) | a non-saturating testbed | seeded by E1's benchmark |
| E4 | Q5 — marginal information gain of each external source, added one at a time | whether external evidence pays its cost | gated on E2 |
| E5 | Q4 — learned evidence weighting vs fixed rules (logistic / bandit / accumulator) | the controller, **only after** the evidence is characterised | gated on E4 |

## E1 — pre-registration (this phase)

**Design.** Three domains with *different uncertainty structure*, each with a continuous
difficulty gradient so the benchmark does not saturate:
- **factoid** (knowledge uncertainty): common / rare / relocated / fictional capitals.
- **arithmetic** (computation uncertainty): 2–5 digit products.
- **reasoning** (composition uncertainty): 1–3 step word problems, small numbers.

Uniform answering method (CoT) across domains so the domain comparison is fair. Per item, four
internal signals: **log-probability, response consistency, normalised entropy, elicited numeric
self-confidence**. Metrics per signal *and per domain*: **AUROC, AUPRC, Brier, ECE, mutual
information** (binned, bootstrap CI).

**Null hypotheses.** For each signal, MI(signal; correct) = 0 and AUROC = 0.5.

**H1 (expected, from factoid work).** Internal signals carry correctness-information overall
(AUROC > 0.6).

**H2 (the decisive test).** Predictive power **differs by domain**. Directional prior *to be
falsified*: confidence predicts correctness *better on factoid recall than on multi-step
reasoning*, because a wrong reasoning chain can be followed fluently and confidently. If
per-domain AUROC is flat across domains, H2 is rejected and the factoid results generalise.

**Effect size / power.** ~60 items/domain (20 × 3 seeds), ~50% base accuracy by construction →
SE(AUROC) ≈ 0.06. Powered (~80%) to detect a cross-domain AUROC difference ≥ 0.15. Smaller
differences are reported as inconclusive, not null.

**Threats to validity (pre-stated).** (i) One model (Qwen2.5-3B); (ii) CoT log-prob is over a
long generation, noisier than a terse answer's; (iii) grading is exact-match, so a correct
answer phrased unusually scores wrong (conservative — depresses accuracy, not signal); (iv) MI
is plug-in binned (bias toward positive) — mitigated by bootstrap CI and the MI=0 null check on
a shuffled control.

**Verification.** `stats.py` is unit-tested against closed-form values (AUROC of a perfect/
anti/random ranker; MI of a deterministic vs independent signal; ECE of a perfectly-calibrated
predictor) *before* any GPU run, and a **shuffled-label control** signal is included in every
run — its MI/AUROC must land at the null, or the pipeline is untrusted.

## Success criterion

Not higher accuracy. A **reproducible empirical law** about executive evidence — narrow,
falsifiable, powered, and surviving the shuffled-control and bootstrap-CI checks.
