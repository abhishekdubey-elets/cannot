# Consistency Is the Only Signal That Transfers: A Preregistered Three-Lineage Map of Cheap Uncertainty Signals in Small Instruct Models

*Draft manuscript, generated from frozen data (`outputs/master_v1`, SHA256-manifested) and
committed analysis artifacts only. Author list, affiliations, and venue formatting TBD.*

## Abstract

Practitioners routinely estimate whether a language model's answer is correct from cheap,
inference-time signals: verbalized confidence, token log-probabilities, token-level
entropy, or agreement across resampled generations. We preregistered five predictions
about what these signals measure and tested them on a frozen protocol — 13 standard
benchmarks across five domains, ~200 items each, identical prompts, one shared extraction
module — in three independent 3–4B instruct lineages (Qwen2.5-3B, Llama-3.2-3B,
Gemma-3-4B). Most predictions failed, and the failures organize into a sharper map.
(1) K-sample answer consistency (K=5) discriminates correctness at AUROC .740/.746/.756
across the three lineages — the only signal whose performance transfers. (2) Every
token-level signal is lineage-specific: mean log-probability ranges from mildly useful
(.61) to below chance (.39); token entropy from .61 to chance. (3) Verbalized confidence
has three phenotypes — collapsed (98% "100", robust to four elicitation wordings
including an inverted scale), graded-informative (.615), and compressed-high (.605).
(4) An information decomposition explains the structure: I(consistency; correctness |
entropy) exceeds its shuffled-control floor in 12/12 model×domain cells, while
I(entropy; correctness | consistency) does so in only 2/12 — uncertainty information
lives in the distribution over *outcomes*, not over *tokens*; the two entropies are
nearly uncorrelated (ρ = −.05 to .31). (5) Mechanistically, consistency behaves like a
measure of generation-basin depth: single-path agreement is more reliable than
divergent-path agreement (.87–.88 vs .70–.78), and its failure mode is a distinct
population of convergent misconceptions (6–10% of items, precision ceiling ≈ .85–.88 at
full agreement) concentrated on plausibility-shaped benchmarks.

## 1 Introduction

When a deployed language model answers a question, the deployer's first practical
question is: *should I trust this one?* Full solutions — training probes, ensembles of
models, retrieval verification — are often unavailable at inference time. What remain
are cheap signals computable from the model's own generation: ask the model how confident
it is; read the log-probability or entropy of the tokens it produced; or sample K
additional generations and count agreement.

These signals are used interchangeably as "model uncertainty," and the literature
evaluates them mostly one model family at a time. Whether a signal validated on one
lineage means the same thing on another is rarely tested, and the question of what each
signal *measures* — as opposed to how well it scores on one benchmark — is usually left
implicit.

We approached this as a measurement problem with a preregistered design. Before data
collection we froze: five falsifiable predictions (P1–P5) inherited from a pilot study on
a custom benchmark; a 13-benchmark, five-domain evaluation suite with per-item frozen
prompts; a single measurement protocol (greedy chain-of-thought pass, K=5 resampling at
temperature 0.7, one fixed confidence elicitation); exclusion rules (accuracy outside
[.10,.90]; extraction failure >10% per cell); and the statistical machinery (bootstrap
CIs on everything, permutation tests with Holm correction for the primary family,
shuffled-control floors under every plug-in information estimate). We then ran the frozen
protocol on three independent model lineages at matched scale.

**Contributions.**
1. A preregistered, QC-disciplined map of four cheap uncertainty-signal families × 13
   benchmarks × 3 model lineages (7,620 measured items), with all raw rows, code, and
   generated tables released.
2. The finding that **only behavioral consistency transfers across lineages**
   (AUROC .74–.76 in all three), while token-level signals vary from useful to
   anti-predictive and verbalized confidence splits into three lineage phenotypes —
   one of which (total collapse) we show is robust to four elicitation wordings.
3. An information-theoretic decomposition establishing that consistency contains
   correctness information that token entropy does not (12/12 cells), while the converse
   fails (10/12) — i.e., "uncertainty" factorizes into an outcome-level quantity that
   tracks correctness and a token-level quantity that mostly does not.
4. A mechanistic characterization: consistency reads the depth of the generation basin
   (single-path agreement > divergent-path agreement), and its residual errors are
   convergent misconceptions — a structured, benchmark-localized population, not noise.

## 2 Related work

**Verbalized confidence.** Language models can be prompted to state confidence in words
or numbers (Lin, Hilton, Evans 2022, "Teaching models to express their uncertainty";
Kadavath et al. 2022, "Language Models (Mostly) Know What They Know"; Tian et al. 2023,
"Just Ask for Calibration"). Findings there are dominated by larger models; our results
show that at 3–4B the very *existence* of a graded verbal signal is lineage-dependent.

**Self-consistency and resampling.** Wang et al. (2022, "Self-Consistency Improves Chain
of Thought Reasoning") introduced majority voting over sampled chains to improve
accuracy; subsequent work uses agreement as a confidence proxy. We measure what that
proxy contains, relative to and conditional on other signals.

**Semantic entropy and generation-level uncertainty.** Kuhn, Gal, Farquhar (2023,
"Semantic Uncertainty") and Farquhar et al. (2024, "Detecting hallucinations in large
language models using semantic entropy") argue for uncertainty measured over meanings
rather than tokens. Our answer-distribution/token-distribution dissociation (ρ down to
−.05; AUROC .70 vs .49 in the same model) is a within-protocol, cross-lineage
confirmation of that program's central premise, obtained with exact-match answer
equivalence rather than entailment clustering.

**Calibration of neural networks.** Guo et al. (2017, "On Calibration of Modern Neural
Networks") established the modern calibration toolkit (ECE, temperature scaling); we
report quantile-binned ECE/Brier alongside discrimination and keep the two questions
separate.

**Benchmarks.** MMLU (Hendrycks et al. 2021), MMLU-Pro (Wang et al. 2024), TruthfulQA
(Lin et al. 2022), GSM8K (Cobbe et al. 2021), SVAMP (Patel et al. 2021), MultiArith
(Roy & Roth 2015), CommonsenseQA (Talmor et al. 2019), ARC-Challenge (Clark et al. 2018),
StrategyQA (Geva et al. 2021), BIG-Bench Hard (Suzgun et al. 2022), MuSR (Sprague et al.
2024), HumanEval (Chen et al. 2021), MBPP (Austin et al. 2021).

**Information decomposition.** Conditional mutual information and the Williams–Beer
partial information decomposition (Williams & Beer 2010) provide the machinery for the
unique-information claims; we guard the plug-in estimators with shuffled-control floors
throughout.

## 3 Methods

**Models.** Qwen2.5-3B-Instruct, Llama-3.2-3B-Instruct, Gemma-3-4B-IT — three independent
pretraining/post-training lineages at matched scale, run fp16 on T4 GPUs with each
model's own chat template and no system prompt (byte-identical user prompts).

**Benchmarks and domains.** 13 benchmarks spanning knowledge (MMLU, MMLU-Pro,
TruthfulQA), arithmetic (GSM8K, SVAMP, MultiArith), commonsense (CommonsenseQA,
ARC-Challenge, StrategyQA), reasoning (BBH 6-task subset, MuSR), and coding (HumanEval,
MBPP). n = 200 per benchmark (or full split if smaller; 2,540 items/model), sampled once
with seed 0 and frozen to versioned JSONL with a manifest before any model ran.

**Protocol (frozen).** Per item: (1) greedy CoT pass (512 new tokens max) with
token-level statistics — mean log-probability and mean entropy over the full chain and
over the answer span, top-1/top-2 margin, top-5 mass; (2) K = 5 sampled passes
(T = 0.7, top-p 0.9) → consistency = modal-answer fraction (code: output-agreement on
prompt-visible examples) and the answer-distribution entropy; (3) a fixed follow-up turn
eliciting confidence 0–100. One shared extraction/grading module across models
(multiple-choice letter, numeric, yes/no, executed code); unparseable answers are graded
wrong and *flagged*, never dropped silently.

**Preregistered predictions.** P1: verbal confidence discriminates on knowledge but not
arithmetic (ΔAUROC ≥ .15). P2: consistency's discrimination transfers across domains
better than confidence's. P3: full-chain log-probability is weak on arithmetic (fluency,
not correctness). P4: consistency and answer-entropy are the same signal (ρ > .95
everywhere; falsified by any benchmark < .9). P5: a zero-cost input feature rivals
internal signals where difficulty is input-legible, and no input feature approaches
confidence on knowledge.

**QC and statistics.** Saturated cells (accuracy outside [.10,.90]) and cells with >10%
extraction failure are excluded from primary contrasts; both exclusions have
flagged-included sensitivity variants. 95% percentile bootstrap CIs on every statistic;
permutation tests (Holm-corrected within the primary family) for domain contrasts;
every MI/CMI/PID estimate is reported against the same estimator run on
label-preserving shuffled signals. Wording sensitivity of the confidence elicitation was
probed post hoc under preregistration cover (three alternative phrasings, M1, MMLU+GSM8K,
n=100): probability framing, frequency framing, and an inverted scale ("how likely is it
that your final answer is wrong?").

**Deviations.** All post-freeze changes are timestamped in DEVIATIONS.md; none alters the
measurement protocol (they are: analysis-side wiring of the preregistered QC rule, the
probe wording texts, infrastructure/engineering items, and the data freeze itself). Seed
sensitivity (seeds 1–2, five benchmarks, M1) is built but not yet run; all results below
are single-seed and say so.

## 4 Results

### 4.1 The signal map

Mean AUROC over unsaturated, extraction-clean cells (per-cell values, CIs, and
calibration metrics in `tables/rq1_signal_benchmark.csv`):

| signal | Qwen2.5-3B | Llama-3.2-3B | Gemma-3-4B |
|---|---:|---:|---:|
| consistency (K=5) | **.740** | **.746** | **.756** |
| verbal confidence | .519 | .615 | .605 |
| token entropy | .610 | .614 | .544 |
| mean log-probability | .606 | .606 | **.391** |

Consistency is the best signal in essentially every benchmark×model cell and its level is
lineage-stable. Token-level signals are not: Gemma's mean log-probability is
anti-predictive (below chance) on clean cells — fluent chains negatively associated with
correctness — making Gemma the only lineage where preregistered P3 was supported
(arithmetic logprob_full = .297). Verbal confidence spans three phenotypes: Qwen answers
"100" on 95.6% of items (mean confidence .992 when correct, .954 when wrong); Llama
produces a graded distribution (modal 80) at .615; Gemma a compressed-high distribution
(67% at "95") at .605.

### 4.2 Preregistered verdicts

Under strict QC, P1 is falsified in Llama (ΔAUROC = −.083, 95% CI [−.152, −.019] —
*sign-reversed*: arithmetic confidence discriminates better than knowledge) and Gemma
(−.124, [−.192, −.052]), and not evaluable in Qwen (the arithmetic pool empties after
exclusions; the labelled sensitivity variant gives +.008 [−.040, +.040], falsified). P2 is supported in
Qwen and Llama, falsified in Gemma. P4 is falsified in all three lineages
(min ρ = .001 / .009 / −.197 against a predicted > .95). P5's preregistered falsifiers
fire in all three. Full grid: `tables/prediction_verdicts.csv`.

### 4.3 Elicitation robustness: the collapse is the model, not the prompt

All four wordings on Qwen produce ≥94% modal replies with chance-level discrimination
(AUROC .47–.57, every CI spanning .5; parse rate 100%). The decisive cell is the inverted
scale: asked how likely its answer is *wrong*, the model answers "0" (94%) — it flips the
number with the semantics. The collapse is semantically coherent, genuinely held total
certainty, not number-anchoring or elicitation noncompliance
(`tables/probe_wording_summary.csv`).

### 4.4 Information structure: outcome uncertainty, not token uncertainty

Per model × domain (never pooled), with C = consistency, E = answer-span token entropy,
Y = correctness (`tables/rq_cmi_decomposition.csv`):

- I(C;Y|E) exceeds its shuffled-control floor in **12/12** cells (.009–.125 nats,
  typically 10–30× floor; verdicts require the bootstrap CI lower bound to clear the
  floor).
- I(E;Y|C) does so in **2/12** cells, both marginal (≤ .02 nats).
- ρ(C,E) ranges −.20 to +.27 across cells.

The asymmetry — consistency subsumes entropy's correctness information but not
conversely — replicates in all three lineages (Figure 6). A direct comparison of the two
"entropies" on the sidecar subset sharpens it: the entropy of the K-sample *answer
distribution* discriminates at .695–.731 in all three models, while *token-level* entropy
ranges .494–.706, and the two are nearly uncorrelated (ρ = −.05 / .24 / .31)
(`tables/mechanism_entropy_variants.csv`, Figure 6A,C). Conditioning on the best zero-cost
input feature (P5 machinery) removes consistency's information in only 2/15 cells, both in
the reasoning domain (`tables/rq5_consistency_after_ff.csv`).

An independent re-implementation of the estimators, reading the same frozen rows without
touching the analysis code (`analysis/audit.py`, `paper/AUDIT.md`), reproduces this
asymmetry: with a multi-permutation bias floor it finds I(C;Y|E) clearing in 10/12 cells
and I(E;Y|C) in 0/12. The two cells where consistency's margin does not clear the floor
under the stricter floor are both the *reasoning* domain — the domain where every signal
is weakest (§4.4, §4.6) and the free-feature gate also fails. The direction of the
asymmetry (C ≫ E) is estimator-robust; the exact "12/12" is not, and should be read as
"in every domain except the weakest."

### 4.5 Mechanism: basin depth and convergent misconceptions

P(correct) is a staircase in consistency (C ≤ .4: .28–.43; mid: .41–.52; C ≥ .8:
.78–.82) and is flat in entropy within each band (`tables/mechanism_quadrants.csv`).
Among high-consistency items, agreement reached along essentially a single reasoning path
(high mean pairwise common-prefix fraction across the K samples) is more reliable than
agreement reached along divergent paths — .87–.88 vs .70–.78 correct, in all three
models (`tables/mechanism_path_divergence.csv`). The residual failure mode is a distinct
population: wrong-but-fully-consistent items are 6.3–10.4% of all items (precision at
C = 1: .84–.88), are 91–97% extraction-clean, sit on prompts 1.5–1.9× longer than
average, and concentrate in TruthfulQA, MuSR, StrategyQA, and CommonsenseQA — benchmarks
shaped around plausible misconceptions (`tables/mechanism_wrong_but_consistent.csv`).
Consistency correlates only weakly with generation length (ρ = −.22 to −.34) and prompt
size (≈ −.1): it is not a repackaged difficulty feature.

### 4.6 Quality control

Cells exceeding the preregistered 10% extraction-failure threshold (all excluded from
primary contrasts, all with sensitivity variants): Qwen — BBH 13.1%, GSM8K 12.5%,
MMLU-Pro 46.5%; Llama — MMLU-Pro 32.0%, MBPP 68.5%; Gemma — MMLU-Pro 53.0%, MuSR 13.1%,
StrategyQA 14.5%. Diagnosis (no parser bugs found): truncation at the frozen 512-token
budget (e.g. 97–100% of flagged MMLU-Pro rows) and answer-format drift in which the
preregistered fallback still recovered a graded answer (GSM8K: 100% of flagged rows;
MBPP: Llama emits unfenced code, execution grading unaffected, accuracy .625).

## 5 Discussion

**One signal, three lineages.** The practical summary is uncomfortable for common
practice: of the four cheap signal families, only the K-sample vote means the same thing
across model families. Token-level statistics — the cheapest and most-used proxies —
change sign across lineages. A token-level uncertainty threshold tuned on one family and
shipped on another is an uncalibrated instrument; in the Gemma case it would be
systematically wrong.

**Why P4 failed.** "Entropy" conflated two quantities. Uncertainty over *which tokens to
emit next* is dominated by lineage-specific surface variance (phrasing, formatting, chain
length); uncertainty over *which answer the process lands on when perturbed* is the
correctness-relevant part — and only resampling reads it. This is a within-protocol,
exact-match analogue of the semantic-uncertainty argument, demonstrated as a double
dissociation inside a single model (Gemma: answer-distribution entropy .695, token
entropy .494, ρ = −.05).

**What consistency measures.** The path-divergence result argues that consistency is a
probe of *basin depth*: when the model knows, temperature perturbation barely moves the
trajectory. That framing predicts exactly the observed failure mode — deep basins encode
shared misconceptions as readily as knowledge, so the residual errors of a
consistency-gated system are convergent misconceptions on plausibility-shaped inputs, a
structured population invisible to any resampling scheme. It also explains why divergent
paths converging on one answer are *weaker* evidence than a single stable path, inverting
the ensemble intuition that independent agreement is stronger.

**Verbal confidence.** At this scale, whether graded verbal uncertainty exists at all is
a training-lineage property (three phenotypes under one protocol), and where it is absent
it is absent robustly (four wordings, semantics demonstrably understood). Claims of the
form "LLMs can/cannot verbalize uncertainty" need a lineage qualifier.

**Practical guidance for 3–4B deployments.** Spend the K-sample budget; gate on
agreement; expect ≈ .85 precision at full agreement; and treat the remaining errors as
misconception-shaped, not noise. Token statistics should be used only with per-lineage
validation, and verbal confidence only after checking the lineage produces one.

## 6 Limitations

Three lineages at 3–4.3B; findings may not extend to frontier scale. Single seed so far
(the preregistered seed run is built but unexecuted; all claims carry that caveat).
Frozen 512-token budget truncates verbose models on long-form benchmarks — flagged,
excluded from primary contrasts, and diagnosed, but a budget-sensitivity run does not
exist. K = 5 is small; consistency at .74–.76 is likely a floor, not a ceiling. Exact-
match/execution grading is conservative. MI/CMI/PID are plug-in estimates guarded by
shuffled controls, not debiased estimators. GPQA was inaccessible (gated); the knowledge
ceiling anchor is absent. Contamination of older benchmarks is plausible; we measure
uncertainty–correctness relationships, not capability. Full standing list:
LIMITATIONS.md.

## 7 Reproducibility

Everything derives from: frozen benchmark JSONLs with manifest; a frozen measurement
protocol (`pipeline/`), audited by a smoke test per model before any full run; raw rows
committed per session with an append-only kernel audit trail; a SHA256-manifested master
dataset (`outputs/master_v1`); analyses (`analysis/`) that read only frozen rows and emit
every table/figure cited here; and a timestamped deviations log. The preregistration
(PLAN.md) predates all data. As an internal check, every headline number was recomputed by
an independent second implementation that shares no code with the analysis pipeline
(`analysis/audit.py` → `paper/AUDIT.md`): data integrity (SHA256), the signal map
(reproduced to ±0.000 AUROC), the P4 falsification, and the C-over-E information asymmetry
all reproduce; the one caveat surfaced (reasoning-domain sensitivity of the exact CMI
cell count) is disclosed in §4.4.

## References

- Austin et al., 2021. *Program Synthesis with Large Language Models.*
- Chen et al., 2021. *Evaluating Large Language Models Trained on Code.*
- Clark et al., 2018. *Think you have Solved Question Answering? Try ARC.*
- Cobbe et al., 2021. *Training Verifiers to Solve Math Word Problems.*
- Farquhar, Kossen, Kuhn, Gal, 2024. *Detecting hallucinations in large language models
  using semantic entropy.*
- Geva et al., 2021. *Did Aristotle Use a Laptop? A Question Answering Benchmark with
  Implicit Reasoning Strategies.*
- Guo, Pleiss, Sun, Weinberger, 2017. *On Calibration of Modern Neural Networks.*
- Hendrycks et al., 2021. *Measuring Massive Multitask Language Understanding.*
- Kadavath et al., 2022. *Language Models (Mostly) Know What They Know.*
- Kuhn, Gal, Farquhar, 2023. *Semantic Uncertainty: Linguistic Invariances for
  Uncertainty Estimation in Natural Language Generation.*
- Lin, Hilton, Evans, 2022. *Teaching Models to Express Their Uncertainty in Words.*
- Lin, Hilton, Evans, 2022. *TruthfulQA: Measuring How Models Mimic Human Falsehoods.*
- Patel, Bhattamishra, Goyal, 2021. *Are NLP Models really able to Solve Simple Math
  Word Problems?*
- Roy & Roth, 2015. *Solving General Arithmetic Word Problems.*
- Sprague et al., 2024. *MuSR: Testing the Limits of Chain-of-thought with Multistep
  Soft Reasoning.*
- Suzgun et al., 2022. *Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can
  Solve Them.*
- Talmor et al., 2019. *CommonsenseQA: A Question Answering Challenge Targeting
  Commonsense Knowledge.*
- Tian et al., 2023. *Just Ask for Calibration: Strategies for Eliciting Calibrated
  Confidence Scores from Language Models Fine-Tuned with Human Feedback.*
- Wang et al., 2022. *Self-Consistency Improves Chain of Thought Reasoning in Language
  Models.*
- Wang et al., 2024. *MMLU-Pro: A More Robust and Challenging Multi-Task Language
  Understanding Benchmark.*
- Williams & Beer, 2010. *Nonnegative Decomposition of Multivariate Information.*
