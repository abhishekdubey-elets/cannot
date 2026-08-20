# DISCUSSION

Written from the measured map (RESULTS.md), not from the preregistered predictions —
which mostly failed, in informative ways. Per PLAN §0 this file is rewritten, not
patched, when evidence moves.

## 1. What the study set out to test, and what it found instead

The preregistration bet on a *domain* story: verbal confidence would work on knowledge
and fail on arithmetic (P1), and consistency and answer-entropy would be one signal
wearing two hats (P4). Neither survived contact with three model lineages. What emerged
is a *representation-level* story:

> **Uncertainty information in small instruct models lives in the distribution over
> outcomes, not in the distribution over tokens — and behavioral consistency is the only
> cheap signal that reads it out lineage-independently.**

## 2. One signal transfers; everything else is an instrument with a serial number

Consistency's AUROC is .740/.746/.756 across Qwen, Llama, and Gemma — three pretraining
corpora, three RLHF pipelines, one number. Every other signal varies qualitatively by
lineage: mean logprob goes from mildly useful (.61) to *anti-predictive* (.39, below
chance on Gemma); token entropy from .61 to chance (.54 mean, .49 on the sidecar subset);
verbal confidence from dead (Qwen) to moderately informative (Llama). A practitioner who
validates a token-level uncertainty proxy on one model family and deploys it on another
is running an uncalibrated instrument. The K-sample vote is the only signal in this study
that traveled.

## 3. Why P4 failed: "entropy" was two different quantities

The preregistration treated answer-distribution entropy and token-level entropy as
near-duplicates (ρ > .95 predicted). Measured: ρ between them is −.05 to +.31, and only
the outcome-level quantity discriminates correctness in all lineages (.70–.73 vs
.49–.71). The CMI decomposition sharpens this: I(C;Y|E) > 0 in 12/12 model×domain cells
while I(E;Y|C) ≈ 0 in 10/12. Token-level uncertainty is dominated by lineage-specific
surface variance (phrasing choices, formatting, chain length); the correctness-relevant
part of "uncertainty" is *which answer the process lands on when perturbed*, and only
resampling reads that.

## 4. What consistency actually measures: basin depth, not vote counting

Two mechanistic results triangulate an interpretation. First, among high-consistency
items, single-path agreement beats divergent-path agreement (.87–.88 vs .70–.78 correct):
when the model knows, temperature-0.7 perturbation barely moves the trajectory — the
generation sits in a deep basin. Second, the failure mode: wrong-but-consistent items
(6–10% of all, precision ceiling ≈ .85–.88 at C = 1) are extraction-clean convergent
errors concentrated on misconception-shaped benchmarks (TruthfulQA by design,
StrategyQA/CommonsenseQA/MuSR) with 1.5–1.9× longer prompts. Deep basins encode shared
misconceptions as readily as knowledge. Consistency measures *stability of the attractor*,
which is usually — but not definitionally — correctness.

## 5. Verbal confidence: a lineage phenotype, robust to elicitation

The Qwen collapse (95.6% "100"; and "0" when asked how likely it is *wrong* — semantics
understood, no graded signal) survived four elicitation wordings including probability
and frequency framings. Llama, same scale and protocol, produces graded confidence at
.615 AUROC. Whether a small model can verbalize uncertainty is a property of its training
lineage, not of scale, prompt, or task — a caution against both "LLMs can't express
uncertainty" and "just ask the model" as general claims.

## 6. Practical reading

For 3–4B instruct models: (i) spend the K-sample budget — consistency at K = 5 is the
only defensible cross-model uncertainty signal, at ~.75 AUROC and .84–.88 precision at
full agreement; (ii) treat token-level statistics as per-model calibrations requiring
revalidation on every lineage; (iii) expect the residual failure mode to be convergent
misconceptions on plausibility-shaped inputs, not random noise — ensembles of samples
cannot see a shared attractor.

## 7. Boundaries

Three lineages at 3–4.3B, one seed so far (preregistered seed run pending), frozen
512-token budget (truncation-deflated cells flagged and excluded from primary contrasts),
plug-in MI with shuffled-control floors, K = 5. The reasoning domain bounds every claim:
where accuracy nears chance, all signals — consistency included — carry little
information (I(C;Y) ≈ .006–.05 nats), and consistency's free-feature margin vanishes in
2/3 lineages there. The map is a map of what these models know; where they know nothing,
no cheap signal recovers it.
