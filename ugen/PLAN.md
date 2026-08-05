# UGEN — Do uncertainty signals generalize across reasoning domains?
## Research plan (pre-registered before any code)

**Status: FROZEN 2026-08-05, before implementation.** Changes after data collection begins
must be logged in `paper/DEVIATIONS.md` with a timestamp and reason.

## 0. Scientific question and lineage

**Question.** Do inexpensive uncertainty signals (elicited self-confidence, token
log-probability, entropy, response consistency, token margin, top-k mass) measure the same
thing across reasoning domains, models, and benchmark families — or are they domain-specific
instruments that happen to share a name?

**Lineage — what this study must confirm, refute, or bound.** The `execev` phase established,
on *custom* benchmarks with one model (Qwen2.5-3B), three regularities. This study's first job
is to test whether they survive contact with *standard academic* benchmarks and *multiple*
models. They are restated here as falsifiable predictions:

- **P1 (domain gap).** Elicited self-confidence discriminates correct from incorrect answers
  far better on knowledge tasks than on computation tasks
  (execev: AUROC 0.91 knowledge vs 0.42 arithmetic).
  *Prediction:* AUROC(selfconf, knowledge domain) − AUROC(selfconf, arithmetic domain) ≥ +0.15.
  *Falsified if* the pooled-domain contrast CI includes 0 or the gap is < +0.10.
- **P2 (behavioural transfer).** Response consistency is the most *transferable* signal: its
  cross-domain AUROC spread is smaller than self-confidence's and log-probability's
  (execev: consistency 0.78→0.69 across the same gap that took selfconf 0.91→0.42).
  *Falsified if* consistency's knowledge−arithmetic ΔAUROC is ≥ selfconf's, or its
  across-domain SD of AUROC is the largest of the signal set.
- **P3 (fluency, not correctness).** Mean log-probability over a chain-of-thought measures
  fluency of the chain: on arithmetic it is uncorrelated or anti-correlated with correctness
  (execev: AUROC 0.41), and log-prob restricted to the *answer tokens* beats full-chain
  log-prob on computation. *Falsified if* full-chain logprob AUROC ≥ 0.6 on arithmetic
  benchmarks, or answer-token logprob shows no advantage.
- **P4 (one behavioural source).** Answer-distribution entropy and consistency are the same
  evidence source (execev: ρ = 0.9991 at K=5). *Prediction:* Spearman ρ(consistency,
  answer-entropy) > 0.95 on every benchmark. *Falsified if* any benchmark shows ρ < 0.9 —
  which would itself be interesting (a domain where they dissociate).
- **P5 (free-feature gate).** On any task family where difficulty is legible from the input
  (arithmetic: digit counts; code: prompt complexity), a zero-cost input feature rivals or
  beats every internal signal, and internal signals carry no unique information after
  conditioning on it (execev: digit count AUROC 0.93 vs best internal 0.69). On knowledge
  tasks no input feature approaches selfconf. *Falsified if* internal signals beat the best
  free feature on GSM8K/SVAMP, or a free feature beats selfconf on MMLU/TruthfulQA.

Each prediction can fail. P1–P3 failing = the execev findings were artifacts of the custom
benchmark; that is a publishable finding and the story will be rewritten around it.

**Retired claims we will NOT rescue** (closed in execev/dispatch): "behavioural calibration
always beats introspective"; "internal evidence is information-theoretically blind";
"more fused signals automatically help"; "synthetic validation validates hypotheses."

## 1. Benchmarks — what and why

All from HuggingFace, official splits, no synthetic data. Per-benchmark evaluation subset:
n = 200 (or the whole split if smaller), sampled once locally with seed 0, frozen to JSONL,
versioned in `datasets/` with a manifest (source, revision, indices, SHA256). Kaggle kernels
run **internet-off** against this frozen copy — every model sees byte-identical prompts.

| # | benchmark | domain | HF source (split) | why included |
|---|---|---|---|---|
| 1 | MMLU | knowledge | `cais/mmlu` all (test) | canonical broad knowledge MC; anchors comparison to the calibration literature |
| 2 | MMLU-Pro | knowledge | `TIGER-Lab/MMLU-Pro` (test) | 10-option, harder — controls for MMLU saturation and for #options as a confound |
| 3 | TruthfulQA | knowledge | `truthful_qa` mc1 (validation) | knowledge where fluency misleads — adversarial for logprob (P3 stress test) |
| 4 | GPQA | knowledge | `Idavidrein/gpqa` main | graduate-level ceiling; **gated** — included only if HF token grants access, else skipped with justification |
| 5 | GSM8K | arithmetic | `openai/gsm8k` main (test) | canonical multi-step word arithmetic; the P1/P5 battleground |
| 6 | SVAMP | arithmetic | `ChilleD/SVAMP` (test) | perturbed one-step problems — separates pattern-matching from computation |
| 7 | MultiArith | arithmetic | `ChilleD/MultiArith` (test) | easy tier; likely near-saturated for 3–4B CoT models — kept as an explicit saturation probe (see §7 exclusion rule) |
| 8 | CommonsenseQA | commonsense | `tau/commonsense_qa` (validation) | canonical commonsense MC (test split is unlabelled → validation, disclosed) |
| 9 | ARC-Challenge | commonsense | `allenai/ai2_arc` ARC-Challenge (test) | science commonsense, different question style |
| 10 | StrategyQA | commonsense | `ChilleD/StrategyQA` (test) | implicit multi-hop yes/no — commonsense that *requires* composition; bridges domains |
| 11 | BBH (6 tasks) | reasoning | `lukaemon/bbh` (test) | logical/symbolic reasoning with per-task subcategories; 6 diverse tasks × 34 items: date_understanding, logical_deduction_five_objects, tracking_shuffled_objects_three_objects, causal_judgement, web_of_lies, temporal_sequences |
| 12 | MuSR | reasoning | `TAUR-Lab/MuSR` (3 configs) | long-context narrative reasoning — composition uncertainty that execev twice failed to make hard enough; these are known-hard for small models |
| 13 | HumanEval | coding | `openai/openai_humaneval` (test, all 164) | canonical code generation, execution-graded (ground truth is objective) |
| 14 | MBPP | coding | `google-research-datasets/mbpp` sanitized (test) | entry-level code — different difficulty stratum from HumanEval |
| — | ALFWorld / MiniWoB | planning | — | **SKIPPED, justified:** both are *interactive environments* (stateful simulators, multi-turn action loops). Our protocol measures uncertainty signals over a single verifiable generation; an environment rollout confounds per-step uncertainty with policy horizon effects, and neither runs offline inside an internet-disabled Kaggle kernel. Planning-flavoured *static* reasoning is partially covered by BBH tracking/temporal tasks and MuSR team_allocation. The limitation is recorded in LIMITATIONS.md. |

Domain assignment is fixed a priori (above) and never re-fitted to results.
StrategyQA is flagged a priori as a *bridge* benchmark (commonsense + composition); its
domain label is commonsense and will not be moved post hoc.

**TruthfulQA mc1 note:** the correct option is listed first in the raw data — options are
shuffled with the item seed at prep time (leakage guard, recorded in manifest).

## 2. Models — what and why

| priority | model | params | why | Kaggle source (internet-off) |
|---|---|---|---|---|
| M1 | Qwen2.5-3B-Instruct | 3.1B | continuity with execev (its predictions were made on this model — M1 is the direct replication axis) | `qwen-lm/qwen2.5/Transformers/3b-instruct` (attachment already proven) |
| M2 | Llama-3.2-3B-Instruct | 3.2B | different pretraining/RLHF lineage at matched scale — the cleanest cross-model test | `metaresearch/llama-3.2/transformers/3b-instruct` (consent required; verified at push time) |
| M3 | Gemma-3-4B-IT | 4.3B | third independent lineage; slightly larger | `google/gemma-3/transformers/gemma-3-4b-it` (consent required) |
| M4 | Phi-3.5-mini-instruct | 3.8B | distillation-heavy training recipe — a priori most likely to differ in calibration. **Deviation from the brief:** Phi-4-mini has no official Kaggle Models listing; Phi-3.5-mini is the closest official Phi. If Phi-4-mini becomes attachable, it replaces M4. | `Microsoft/phi-3/transformers/phi-3.5-mini-instruct` |

Fallbacks if consent/attachment fails for M2–M4: Qwen2.5-1.5B-Instruct and
Qwen2.5-7B-Instruct (4-bit NF4) — this degrades the lineage axis into a scale axis and the
paper's RQ3 claim would be weakened accordingly (stated honestly, not papered over).

All models fit a T4 (16 GB) in bf16/fp16 (3–4.3B ⇒ 6.5–9 GB weights + <2 GB KV cache at
our context lengths). 4-bit quantization is used only for fallback 7B runs and is recorded
in the `model` column (`-4bit` suffix) — quantization is a known confound for logprob-based
signals, so quantized and unquantized runs are never pooled.

## 3. Measurement protocol (frozen)

Per item, one model, one seed:

1. **Greedy CoT pass** (temperature 0, max 512 new tokens). Prompt = frozen template with
   "Reason step by step, then give your final answer on the last line as `Final answer: X`".
   Extract: prediction; `mean_logprob_full` and `mean_logprob_answer` (answer-line tokens
   only — P3 needs both); `entropy_token` (mean token-level predictive entropy, full and
   answer spans); `token_margin` (mean p₁−p₂ over answer tokens); `topk_mass` (mean top-5
   probability mass over answer tokens); `generation_length`; `latency`.
2. **K = 5 sampled passes** (temperature 0.7, top-p 0.9, batched in one forward loop).
   Extract: `consistency` = modal-answer fraction; `entropy_answer` = normalized entropy of
   the answer distribution (kept alongside consistency solely to test P4).
3. **Confidence elicitation** (frozen wording, identical across all models and benchmarks —
   the execev E1/E1b wording drift is a documented confound we will not repeat):
   follow-up user turn: *"How confident are you that your final answer is correct?
   Respond with only a number from 0 to 100."* → `self_confidence` ∈ [0,1].
4. **Shuffled control**: per benchmark, a pseudo-signal drawn by permuting a real signal's
   values across items (seeded). Its AUROC/MI must land at the empirical null in every cell
   or the pipeline is not trusted (execev rule, kept).

**Grading.** MC → letter extraction with fallback to option-text match; numeric → last-number
parse with normalization (commas, units) and relative tolerance 1e-4; yes/no → token match;
code → execute official tests in a subprocess (8 s timeout), pass@1 on the greedy sample.
Exact-match grading is conservative (depresses accuracy, not signal ranking) — disclosed.

**Code-domain consistency (protocol adaptation, disclosed):** textual agreement is
ill-defined for programs, so consistency = modal agreement of the K samples' *outputs on the
example inputs already present in the prompt* (HumanEval docstring examples; MBPP's standard
3-shot asserts). No held-out test inputs are used, so no leakage beyond the benchmark's own
standard prompt. If a prompt exposes no example inputs, consistency = NaN and the item is
excluded from consistency-based analyses (count reported).

**Seeds.** Main run: seed 0 end-to-end. Seed sensitivity: one benchmark per domain
(MMLU, GSM8K, CommonsenseQA, BBH, HumanEval) re-run with seeds 1 and 2 on M1 only; if
per-benchmark AUROCs move by more than their bootstrap CI width, all headline claims get a
seed-variance caveat and CIs are widened by the observed between-seed component.

**Output schema** (one row per item × model × seed, no aggregation-only storage):
`sample_id, benchmark, domain, subcategory, difficulty, model, prompt, prediction,
ground_truth, correct, self_confidence, mean_logprob, mean_logprob_answer, entropy,
entropy_answer, token_margin, topk_mass, consistency, generation_length, latency, seed`
(+ every sampled answer in a sidecar JSONL for re-analysis; `mean_logprob`/`entropy` are the
full-chain values, answer-span variants carry the suffix). `difficulty` is populated where
the benchmark provides or implies it (GSM8K: gold solution step count; MuSR/BBH: task
config; else NaN) — never invented.

## 4. Analysis plan

**RQ1 (what does each signal measure).** Per signal × benchmark × model: AUROC, AUPRC,
Brier, NLL, ECE, MCE (raw and after per-benchmark isotonic recalibration), MI with bootstrap
CI, all against the shuffled control. Reliability diagrams and ROC/PR curves per cell.

**RQ2 (cross-domain generalization).** (a) Per-signal domain-pooled AUROC with the three
pre-registered contrasts: knowledge−arithmetic (primary, P1), knowledge−reasoning,
arithmetic−commonsense. Permutation test + bootstrap CI on each ΔAUROC, per signal.
(b) **Calibration-transfer matrices**: fit isotonic calibration (and a single selective-
prediction threshold at 80% target precision) on benchmark A, evaluate ECE/Brier/coverage on
every benchmark B → 13×13 matrix per signal per model. AUROC is rank-invariant, so transfer
failure isolates *calibration* (mis-scaling) from *discrimination*.

**RQ3 (cross-model transfer).** Identical protocol on M1–M4; per-signal AUROC vectors over
benchmarks compared across models (Spearman of benchmark-profiles, sign agreement of domain
contrasts). Calibration transfer *across models* (fit on M1/benchmark A, test on M2/A).

**RQ4 (disagreement).** Signals z-scored within benchmark×model. Pre-defined conflict
patterns: selfconf-high∧logprob-low; consistency-high∧entropy-high (P4 violation);
entropy-low∧wrong (confidently-wrong); logprob-high∧wrong; selfconf-low∧correct
(underconfidence). k-means (k chosen by silhouette, k ∈ 2..8) over the signal-vector of
conflict items; clusters characterized by domain/benchmark/length composition and 5 verbatim
exemplars each in SUPPLEMENTARY.md.

**RQ5 (universal mechanisms / information structure).** Per domain (never pooled — execev
lesson): MI(signal; correct), CMI(signal; correct | best free feature) — the P5 gate — and a
two-source partial information decomposition (Williams–Beer I_min on quantile-discretized
signals; plug-in, positively biased, so every PID quantity is reported against the same
decomposition run on shuffled controls, and strata below 30 items are not decomposed).
Sources = selfconf × consistency (the two signal families E1b showed are distinct);
logprob × selfconf as secondary. If P4 holds, entropy enters no PID (it is consistency).

**Free features (P5, computed from the input alone, no ground-truth annotation):** prompt
token/char length, digit count, max-number magnitude (log10), option count, question word
count; code: argument count, docstring length. Per domain, the best single free feature by
AUROC is the conditioning variable for CMI. Features derived from ground-truth annotations
(e.g. gold step counts) are *difficulty descriptors*, never baselines (execev leakage rule).

**Scaling regression.** Logistic regression: correct ~ signal + domain + signal×domain +
prompt_length + model, with cluster-robust (by benchmark) errors; and OLS on per-cell AUROC ~
domain + model + benchmark family. Reported with effect sizes and CIs, labelled observational.

**Statistical rigor, applied uniformly:** 95% percentile bootstrap CIs (≥1000 reps) on every
reported statistic; permutation tests (≥5000 perms) for contrasts; Holm–Bonferroni within the
**primary family** = {P1, P2, P3, P5 headline tests} × per model (16 tests); everything else
is labelled exploratory with unadjusted p-values and CIs. Effect sizes: ΔAUROC (rank effect),
Cliff's delta for signal-vs-signal, odds ratios for the regression. No claim without a CI.

## 5. Power analysis and sample-size planning

Primary endpoint: per-domain per-signal AUROC and cross-domain ΔAUROC.

- Per benchmark, n = 200. Hanley–McNeil SE(AUROC) at A = 0.75 with a 60/40 class split:
  ≈ 0.037; at 80/20 (worse case): ≈ 0.047.
- Per domain (3 benchmarks pooled, ~600 items, assuming 25–75% accuracy):
  SE ≈ 0.021–0.027 ⇒ SE(ΔAUROC between two domains) ≈ 0.030–0.038.
- **Power for P1:** predicted Δ = +0.15 vs SE ≈ 0.035 ⇒ z ≈ 4.3, power > 0.99 at α = .05
  (Holm-corrected α = .003 ⇒ power ≈ 0.97). Minimum detectable Δ at 80% power ≈ 0.10.
- **Within-benchmark** cells (n = 200) are powered (80%) for ΔAUROC ≈ 0.15–0.18 — single-
  benchmark contrasts are secondary evidence only.
- Class-balance failure mode: any benchmark with accuracy > 0.90 or < 0.10 has < 20 minority
  items (SE > 0.08) → excluded from domain pooling by the pre-registered rule in §7; the
  cell is still reported (execev saturation lesson: reasoning domain died twice this way).
- MI estimates use 5 quantile bins ⇒ plug-in bias ≈ (bins−1)/(2n ln2) ≈ 0.01–0.02 nats at
  n = 200; the shuffled control carries the empirical bias floor in every table.
- PID strata: decomposition only where a domain×model cell has ≥ 150 items and ≥ 30 in the
  minority class; below that PID is reported as "underpowered — not run".

## 6. Kaggle runtime and memory budget

**Measured anchor:** execev-e1b (T4, Qwen2.5-3B bf16, unbatched) processed 216 items ×
(1 CoT + 5 samples + 1 elicitation) in a single session — ≈ 40–60 s/item unbatched. This
pipeline batches the K samples in one generate() call and greedy passes 8 items at a time.

Per model, per full sweep: ~2,560 items (13 benchmarks; GPQA excluded until access is
verified). Estimated wall time (T4, conservative 15–22 s/item effective):
**11–15 h ⇒ two kernel sessions per model** (12 h cap), resumed via the checkpoint
mechanism (§8). Per week Kaggle grants ~30 GPU-hours: **M1+M2 in week 1, M3+M4 in week 2.**

GPU memory (T4 16 GB): weights 6.5 GB (Qwen/Llama bf16) – 9 GB (Gemma-3-4B); KV cache < 1.5
GB at batch 8 × ~1.5k context; headroom > 5 GB. No quantization needed for M1–M4.
Batch sizes auto-scale down on OOM (caught, retried at half batch, logged).

**Smoke test S0 (runs before any full sweep):** all benchmarks × 8 items × M1, one session
(~45 min). Validates: dataset loads offline, prompt formatting, answer extraction per answer
type, code execution grading inside the kernel, signal extraction, schema, checkpoint/resume.
Full runs launch only after S0's output passes an offline audit (grading spot-check of 20
random items by hand).

## 7. Evaluation order (highest scientific value first) and decision gates

1. **S0** smoke test (M1, 8/benchmark) → audit gate.
2. **G1: M1 full sweep** — the replication axis: tests P1–P5 on standard benchmarks with the
   model the predictions were made on. If P1 fails here, the execev regularities are custom-
   benchmark artifacts; the paper pivots to that finding (decision gate, pre-stated).
3. **G2: M2 full sweep** (Llama-3.2-3B) — first cross-lineage test (RQ3 minimal viable).
4. **A1: full analysis pass on M1+M2** — transfer matrices, disagreement, PID, free-feature
   gate. RESULTS.md/DISCUSSION.md updated from measured numbers only.
5. **G3, G4: M3, M4 sweeps** — complete the model grid.
6. **A2: cross-model analysis, seed-sensitivity runs, final figures/tables/paper docs.**

Pre-registered exclusion rule (from §5): benchmarks outside 0.10–0.90 accuracy for a given
model are excluded from that model's *domain-pooled* statistics and marked "saturated/floor"
in all tables; per-benchmark cells are still reported. MultiArith is the expected casualty.

## 8. Engineering (reproducibility & resume)

- **Frozen data:** `pipeline/prep_datasets.py` runs locally (internet), samples with seed 0,
  writes `datasets/<benchmark>.jsonl` + `datasets/MANIFEST.json` (HF path, revision,
  indices, prompt template hash, SHA256 per file). Uploaded once as Kaggle dataset
  `abhishekakgec/ugen-data`; kernels attach it read-only.
- **Code as data:** `ugen` package uploaded as Kaggle dataset `abhishekakgec/ugen-src`
  (same mechanism as execev). Kernel scripts are thin entrypoints.
- **Checkpointing:** runner writes `rows_<model>_<benchmark>.csv` after *every benchmark*
  and a partial-rows file every 25 items; a resumed kernel attaches the previous run's
  output (Kaggle kernel-output as data source) and skips completed benchmarks/items.
- **No overwrites:** every kernel writes into a run-stamped directory; local `outputs/` is
  append-only; analysis reads raw CSVs and never edits them.
- **Verification before GPU:** `pipeline/stats.py` self-test (closed-form AUROC/MI/ECE/
  Brier/NLL/permutation cases) must pass locally and at kernel start; shuffled control in
  every run. Grading unit tests run against hand-written cases for all four answer types.
- Figures: 300 dpi PNG + vector PDF, generated by `analysis/make_figures.py` only from raw
  CSVs. Tables: CSV + Markdown + LaTeX from `analysis/make_tables.py`. Paper docs in
  `paper/` cite generated numbers by file reference.

## 9. Risks and confounds (pre-stated)

1. **Chat-template heterogeneity** — prompts are byte-identical but each model wraps them in
   its own chat template; this is inherent to instruct models and disclosed. No system prompt
   except each model's default-empty behaviour.
2. **Answer-extraction differences across models** could masquerade as signal differences —
   mitigated: one shared extraction module, per-model extraction-failure rates reported;
   cells with > 10% extraction failure flagged and excluded from primary contrasts.
3. **Elicitation compliance**: small models sometimes answer the confidence question with
   text. Non-parsable elicitation → NaN (rate reported), never imputed as 50.
4. **Verbalized-confidence granularity** (models emit round numbers: 80/90/95) — ECE binning
   uses quantile bins; discrimination metrics are rank-based and unaffected.
5. **Contamination**: GSM8K/MMLU are likely in pretraining data for all four models. We
   measure uncertainty-correctness relationships, not capability, and contamination affects
   all models/domains asymmetrically — flagged in LIMITATIONS.md; MMLU-Pro/MuSR (2024+) are
   the lower-contamination counterweights.
6. **CoT length couples to logprob/entropy** (longer chains → different token statistics) —
   generation_length is a covariate in every regression; answer-span variants control it.
7. **Saturation** (execev's recurring failure) — handled by the §7 exclusion rule, and MuSR/
   MMLU-Pro/GPQA-class benchmarks anchor the hard end.
8. **Plug-in MI/PID bias** — shuffled-control floor everywhere; strata minimums (§5).
9. **One elicitation wording** — a wording-sensitivity probe (3 alternative phrasings, MMLU +
   GSM8K, M1 only, n=100) quantifies elicitation fragility as a supplementary analysis.
10. **Kaggle preemption/12 h cap** — per-benchmark checkpoints + resume (§8); nothing is lost
    if a session dies mid-benchmark (25-item partial saves).

## 10. Fallback strategies (Kaggle limits)

- GPU quota exhausted → reduce to n = 120/benchmark for M3/M4 (SE(AUROC) grows ~0.05/cell;
  domain pooling still powered for Δ ≥ 0.13) — never drop benchmarks below 3 domains.
- Model attachment blocked (consent) → Qwen-family scale fallback (§2), RQ3 claim weakened
  and stated as such.
- GPQA inaccessible → knowledge domain rests on MMLU/MMLU-Pro/TruthfulQA (already 3
  benchmarks; GPQA was the ceiling anchor, its absence is a LIMITATIONS entry).
- Kernel internet unavailable *and* dataset upload fails → nothing runs; there is no
  synthetic fallback by design.
- T4 unavailable → P100 (same memory, ~1.3× slower; batch sizes unchanged).

## 11. Success criterion

Not confirming P1–P5. The deliverable is a **measured map of what each cheap uncertainty
signal is an instrument for**, across 13 standard benchmarks × 4 models, with calibrated
uncertainty about that map (CIs, controls, corrected tests), and a story rewritten to
whatever the map shows.
