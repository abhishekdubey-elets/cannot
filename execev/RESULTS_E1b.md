# E1b — Confidence dynamics vs difficulty: knowledge vs computational metacognition

Kernel `execev-e1b` (T4), Qwen2.5-3B-Instruct, 216 items = 3 domains × 4 difficulty tiers ×
6 items × 3 seeds, K=5 samples. Pre-registered P1–P4 in `run_e1b.py` before the run.

**Two corrections applied to my own output before reading it** (both caught by the raw dump
that this phase existed to add):
1. The per-tier table's hand-written header disagreed with the print order, **mislabelling
   every column**. Values print in `INTERNAL` order (logprob, consistency, entropy, selfconf).
   All numbers below are the corrected assignment, verified against `rows.json` offline. The
   header is now generated from `INTERNAL` so it cannot drift again.
2. Reasoning saturated, so all reasoning-domain statistics are **void** (below).

## The headline: knowledge metacognition works, computational metacognition does not

Two cells with **identical accuracy (0.000)** and opposite confidence:

| item type | accuracy | mean self-confidence |
|---|---|---|
| factoid tier 3 — fictional countries | 0.000 | **0.000** |
| arithmetic tier 3 — 5-digit × 5-digit | 0.000 | **0.826** |

The model knows it cannot name the capital of *Qorvania*, and does not know it cannot multiply
89060 × 81726. Same failure, opposite introspective report.

### Confidence vs difficulty, by tier

**factoid** (knowledge uncertainty) — every signal tracks difficulty:

| tier | accuracy | logprob | consistency | entropy | selfconf |
|---|---|---|---|---|---|
| 0 common | 1.000 | 0.982 | 0.867 | 0.802 | 1.000 |
| 1 rare | 1.000 | 0.906 | 0.500 | 0.348 | 0.913 |
| 2 relocated | 0.667 | 0.895 | 0.389 | 0.206 | 0.934 |
| 3 fictional | 0.000 | 0.741 | 0.200 | −0.000 | **0.000** |

ρ(accuracy, tier) = −0.791. ρ(logprob, tier) = **−0.819** [−0.89, −0.73];
ρ(selfconf, tier) = **−0.743** [−0.84, −0.61].

**arithmetic** (computational uncertainty) — log-probability is *flat-high* and rises:

| tier | accuracy | logprob | consistency | entropy | selfconf |
|---|---|---|---|---|---|
| 0 (1d×1d) | 1.000 | 0.877 | 0.278 | 0.093 | 0.710 |
| 1 (2d×2d) | 0.833 | 0.886 | 0.333 | 0.132 | 0.952 |
| 2 (3d×2d) | 0.500 | 0.921 | 0.244 | 0.042 | 0.939 |
| 3 (5d×5d) | 0.000 | 0.890 | 0.200 | −0.000 | 0.826 |

ρ(accuracy, tier) = −0.756, so the difficulty gradient is real. But
ρ(logprob, tier) = **+0.328** [+0.08, +0.56] and ρ(selfconf, tier) = **+0.239** [−0.01, +0.48]
— both point the *wrong way*. Against the tier-independent complexity measure:
ρ(accuracy, carries) = −0.690 while ρ(selfconf, carries) = +0.169 [−0.09, +0.40] and
ρ(logprob, carries) = +0.262 [−0.02, +0.53]. Confirmed on two independent difficulty measures.

**Mechanistic reading for log-probability.** It tracks retrieval difficulty well (−0.82) and
computational difficulty *inversely* (+0.33). Long arithmetic chains are token-wise fluent —
digit sequences are locally predictable — so mean token probability rises with the length of a
wrong computation. **Log-probability measures fluency of the chain, not correctness of the
computation.**

## Verdicts on the pre-registered predictions

| | prediction | verdict |
|---|---|---|
| P1 | accuracy falls with tier | **passes** for factoid (−0.79) and arithmetic (−0.76); **fails** for reasoning (−0.16) |
| P2 | ρ(selfconf, tier) ≈ 0 on arithmetic, negative on factoid | **falsified in letter, confirmed in direction**: factoid −0.743 as predicted, but arithmetic is **+0.239**, not ~0 |
| P3 | selfconf dynamic range narrower on arithmetic | **partially**: sd 0.269 vs 0.438, but n_unique 18 vs 7 — *wider* variety, narrower spread |
| P4 | within a fixed arithmetic tier, selfconf AUROC ≈ 0.5 | **untestable**: tiers 0 and 3 have a single label. Tier 1 = 0.300, tier 2 = 0.543 |

**My E1 "constant signal" hypothesis is FALSIFIED.** I claimed E1's arithmetic AUROC of exactly
0.500 was the signature of a *constant* confidence. It is not: E1b gives 18 unique values,
sd 0.269, AUROC 0.416. Confidence on arithmetic **varies substantially — it is simply
uncorrelated to anti-correlated with correctness.** The E1 exact-0.500 was a coincidence of
that sample, and I over-read it. The real defect is anti-calibration, not constancy.

## Q1 re-run, excluding the void reasoning domain (n=144)

| signal | AUROC [95% CI] | MI [95% CI] |
|---|---|---|
| **consistency** | **0.738 [0.67, 0.80]** | 0.125 [0.06, 0.21] |
| entropy | 0.738 [0.67, 0.80] | 0.125 [0.06, 0.21] |
| selfconf | 0.688 [0.60, 0.77] | 0.095 [0.03, 0.18] |
| logprob | 0.687 [0.60, 0.77] | 0.120 [0.05, 0.21] |
| *shuffled control* | *0.549 [0.45, 0.64]* | *0.029 [0.01, 0.07]* |

## The transfer result — the most useful thing E1b produced

Domain contrasts with the **empirical null** (E1's Threat 2 is resolved: the control's own
contrast is now ≈0, +0.004 / +0.018 / +0.008, where in E1 it was as high as +0.088):

**factoid − arithmetic:**

| signal | Δ AUROC | verdict |
|---|---|---|
| selfconf | +0.490 [+0.35, +0.64] | **DIFFERS** |
| logprob | +0.421 [+0.25, +0.59] | **DIFFERS** |
| **consistency** | **+0.091 [−0.04, +0.22]** | **inconclusive — transfers** |
| *control* | *+0.004 [−0.17, +0.18]* | *empirical null* |

> **Behavioural evidence transfers across uncertainty types; introspective and token-level
> evidence do not.** Self-confidence swings 0.910 → 0.416 and log-probability 0.844 → 0.413
> between knowledge and computation, while consistency holds 0.782 → 0.688 with a CI spanning
> zero. Sampling the model repeatedly asks the same question of *any* competence; asking the
> model, or reading its token probabilities, asks a question whose meaning changes with the
> uncertainty type.

This is a **new claim, not a restoration of the retired one**. The retired claim was
"behavioural calibration is more *accurate* than introspective." This one is that behavioural
calibration is more *transferable*, which is a different axis and the one an executive
controller actually needs.

## Failures of this experiment, stated plainly

**Fix 1 FAILED — reasoning saturated again, worse (0.986, 71/72 correct, ONE negative).**
Hardening 1→5 chained operations did not help; 3B with CoT solves these. ρ(accuracy, tier) =
−0.16 means the tier gradient barely exists. **Every reasoning statistic in the kernel output
is void**, including the eye-catching selfconf AUROC of 0.070 and the +0.833 factoid−reasoning
contrast — both are one-item artifacts. Compositional uncertainty remains **unmeasured** after
two attempts. It needs a genuinely hard generator (longer chains, larger operands, distractors)
or a different task family.

**Fix 5 FAILED — entropy and consistency remain redundant at K=5**: ρ(consistency, entropy) =
**+0.9991**. Raising K from 3 to 5 did not separate them. This is now a *robust* finding rather
than a K=3 artifact: **they are one evidence source**, and any PID treating them as two will
report a spurious redundancy.

**A confound I introduced.** I changed the self-confidence elicitation between runs — E1 asked
"…that you **know the correct answer**", E1b asks "…that you **can answer this correctly**".
So E1's selfconf 0.869 and E1b's 0.688 are **not comparable**, and the apparent "collapse" of
selfconf across phases is partly my own uncontrolled edit. Within-E1b domain contrasts are
unaffected (same prompt throughout). Future runs must freeze elicitation wording.

**Fixes that worked:** raw dump (2) — which caught the column mislabelling and the void
reasoning cells; all-pairs contrasts (3); empirical null (4) — E1's Threat 2 resolved;
difficulty dynamics (6) — delivered, and falsified my own hypothesis.

## What E1b licenses for E2

* Run PID **per uncertainty type**, never pooled — a pooled decomposition would average a 0.91
  signal with a 0.42 one.
* Enter **one** behavioural signal, not two (ρ = 0.999).
* Only **two** domains are currently measurable (knowledge, computational). Compositional is
  unmeasured; ambiguity, stale knowledge and external uncertainty are untouched.
* The interesting decomposition question is now sharp: on arithmetic, consistency reaches 0.688
  while selfconf is 0.416 and logprob 0.413. **Does consistency carry unique information there,
  or is it redundant with a difficulty prior the controller could get for free?** A trivial
  feature (operand digit count) may match it — that baseline must be run *before* claiming
  consistency contributes unique information.
