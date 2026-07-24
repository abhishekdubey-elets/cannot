# E1 — Information content of internal evidence, and whether it generalises across domains

Kernel `execev-infocontent` (v2, Tesla T4), Qwen2.5-3B-Instruct, 180 items = 3 domains ×
20 items × 3 seeds. Pre-registered in [ROADMAP.md](ROADMAP.md) before the run.

**Benchmark did not saturate overall** (accuracy 0.622) — but see the per-domain flaw below.

| domain | n | accuracy |
|---|---|---|
| factoid (knowledge) | 60 | 0.700 |
| arithmetic (computation) | 60 | 0.250 |
| reasoning (composition) | 60 | **0.917 ← too easy, see Threat 1** |

## Q1 — information content per signal (overall)

| signal | AUROC [95% CI] | AUPRC | Brier | ECE | MI nats [95% CI] |
|---|---|---|---|---|---|
| **selfconf** (elicited graded) | **0.869 [0.82, 0.91]** | 0.943 | 0.269 | 0.303 | **0.259 [0.18, 0.34]** |
| logprob | 0.674 [0.59, 0.75] | 0.776 | 0.293 | 0.273 | 0.124 [0.06, 0.22] |
| consistency | 0.630 [0.57, 0.68] | 0.778 | 0.254 | 0.198 | 0.062 [0.02, 0.11] |
| entropy | 0.630 [0.57, 0.68] | 0.778 | 0.473 | 0.489 | 0.062 [0.02, 0.11] |
| *shuffled control* | *0.527 [0.42, 0.61]* | *0.638* | *0.336* | *0.298* | *0.020 [0.00, 0.05]* |

**Control lands at the null → pipeline trusted.**

### Three findings that hold

**(1) H1 confirmed, and elicited graded self-confidence is the best single signal by a wide
margin.** selfconf 0.869 vs logprob 0.674 — non-overlapping CIs. Note this is *introspective*
evidence beating *behavioural* evidence (consistency 0.630), further burying the retired
"behavioural calibration always beats introspective" claim: the ordering reverses once
confidence is elicited as a graded value instead of a binary self-report.

**(2) Discrimination and calibration come apart.** selfconf has the best ranking (AUROC 0.869)
but poor calibration (ECE 0.303); consistency ranks worst (0.630) yet is the best-calibrated
(ECE 0.198). **Consequence for the controller:** a fusion rule that consumes these as
probabilities (naive Bayes, log-odds accumulation) will be misled by the signal that ranks
best. Rank-based and probability-based fusion are not interchangeable here.

**(3) Entropy and consistency are the SAME SIGNAL at K=3 — a Q2 result arriving early.**
Identical AUROC, AUPRC and MI (0.630 / 0.778 / 0.062), differing only in Brier/ECE. This is
not coincidence: with 3 samples both are strictly monotone functions of the modal count
(consistency ∈ {1, .667, .333}; normalised entropy ∈ {1, .421, 0}), so they are
rank-equivalent **by construction**. Any PID/redundancy analysis treating them as two sources
would be measuring an artifact of the sample budget, not a property of evidence. **E2 must
either raise K or drop one of them.**

## H2 — does it generalise across domains? **NO.**

AUROC per signal × domain:

| signal | factoid | arithmetic | reasoning |
|---|---|---|---|
| logprob | 0.816 | **0.413** | 0.742 |
| consistency | 0.754 | 0.633 | 0.527 |
| entropy | 0.754 | 0.633 | 0.527 |
| selfconf | **0.984** | **0.500** | 0.724 |
| *shuffled control* | *0.648* | *0.603* | *0.560* |

Pre-registered decisive contrast (factoid − reasoning), bootstrap CI:

| signal | difference | verdict |
|---|---|---|
| logprob | +0.083 [−0.23, +0.47] | inconclusive |
| consistency | +0.228 [+0.11, +0.33] | **DIFFERS** |
| selfconf | +0.262 [+0.13, +0.41] | **DIFFERS** |

**H2 is supported and my directional prior survives its first test:** confidence predicts
correctness far better for knowledge retrieval than for computation or composition. The
factoid results — on which every previously "confirmed" finding in this project rests — **do
not generalise**.

**The sharpest cell is arithmetic.** selfconf falls to *exactly* 0.500 and logprob to 0.413
(at/below chance). An AUROC of exactly 0.500 is the signature of a **constant** signal (all
ties) — the same artifact that made last phase's freshness flag score 0.500 within a category.
The implied mechanism: **the model reports the same confidence for a 2-digit and a 5-digit
product — it cannot introspect on computational difficulty at all**, while its confidence
tracks knowledge availability well (0.984). *Hypothesis, not yet verified — needs the raw dump
(see Fix 2).*

## Threats to validity — two are realised flaws, stated plainly

**Threat 1 (realised): the reasoning domain saturated at 0.917.** Only 5 incorrect items in 60,
so every reasoning AUROC rests on 5 negatives. My own roadmap says "never design a benchmark
that saturates," and the reasoning domain did. **This weakens precisely the contrast I
pre-registered as decisive (factoid vs reasoning).** The factoid-vs-arithmetic contrast is far
better powered — and I did not compute its CI, because I only coded the pre-registered pair.

**Threat 2 (realised): the shuffled control is NOT at chance per-domain** (0.648 / 0.603 /
0.560) even though it is overall (0.527). With n=60 and unbalanced classes, the per-domain
AUROC noise floor is roughly ±0.15. Per-domain cells must be read with that slop, and the
control's own cross-domain difference (+0.088) shows the null for a *difference* is not
visibly 0 in this sample. The two "DIFFERS" verdicts have effect sizes (+0.23, +0.26) larger
than the control's, so they survive — but by less margin than the CIs alone suggest.

**Threat 3 (design): no raw per-item dump.** Only summary statistics were written, so the
arithmetic-constancy hypothesis and the missing contrasts cannot be checked without re-running.

Pre-stated and unchanged: single model (3B); CoT log-prob is over a long generation;
exact-match grading is conservative; MI is plug-in binned (mitigated by the control).

## What E1 licenses, and what it forbids

**Licensed:** internal evidence carries real correctness-information; graded elicited
confidence is the strongest single internal signal; discrimination ≠ calibration; entropy and
consistency are redundant by construction at K=3.

**Forbidden:** any claim that these results describe "LLM confidence" in general. They describe
**confidence about knowledge**. On computation, the best internal signal is worth nothing.

## Fixes carried into E1b (before any of E2–E5 proceeds)

1. Harden the reasoning domain (more steps / larger operands) until accuracy lands in 0.3–0.7.
2. Dump raw per-item rows for offline re-analysis and adversarial re-checking.
3. Compute **all three** pairwise domain contrasts, not just the pre-registered pair.
4. Run the shuffled control through the cross-domain contrast too, to get an empirical null for
   the difference rather than assuming 0.
5. Raise K (or drop entropy) so the two behavioural signals are not rank-equivalent.
