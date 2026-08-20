# RESULTS

All numbers below are read from generated artifacts (`tables/*.csv`, `figures/`) produced
by `analysis/*.py` from the frozen master rows (`outputs/master_v1/`, SHA256 manifest).
Nothing is hand-computed. Data: 3 models × 13 benchmarks × ~200 items = 2,540 rows/model
(7,620 total), one seed (0), frozen protocol throughout.

## 1. Preregistered predictions (tables/prediction_verdicts.csv)

| | Qwen2.5-3B | Llama-3.2-3B | Gemma-3-4B |
|---|---|---|---|
| P1 domain gap in selfconf (≥ .15) | INCONCLUSIVE (QC) / FALSIFIED (sens.: Δ=+.008 [−.033,+.039]) | FALSIFIED (Δ=−.083 [−.143,−.022] — *reversed sign*) | FALSIFIED (Δ=−.124) |
| P2 consistency transfers better | SUPPORTED (sens.) | SUPPORTED | FALSIFIED (d_cons −.109 vs d_self −.124) |
| P3 full-chain logprob weak on arithmetic | FALSIFIED (.725) | FALSIFIED (.800) | **SUPPORTED** (.297) |
| P4 consistency ≈ answer-entropy (ρ > .95) | FALSIFIED (min ρ ≈ .001) | FALSIFIED (.009) | FALSIFIED (−.197) |
| P5 free-feature gate | FALSIFIED | FALSIFIED | FALSIFIED |

No prediction survived in all three lineages; P4 failed identically in all three, and its
failure mode is the study's central finding (§4–§6).

## 2. The signal map (tables/rq1_signal_benchmark.csv, tables/m1_dashboard.md)

Mean AUROC over unsaturated, extraction-clean cells:

| signal | Qwen | Llama | Gemma |
|---|---:|---:|---:|
| consistency | .740 | .746 | .756 |
| self-confidence | .519 | .615 | .605 |
| token entropy | .610 | .614 | .544 |
| mean logprob | .606 | .606 | **.391** |

- **Consistency is lineage-stable** (.74–.76) and is the best signal in essentially every
  benchmark × model cell.
- **Verbal self-confidence has three phenotypes**: collapsed (Qwen: 95.6% of replies are
  "100"; AUROC ≈ .52), graded-informative (Llama: modal 80 with spread; .615), and
  compressed-high (Gemma: 67% at "95"; .605). The wording probe (§7) shows the Qwen
  collapse is not an elicitation artifact.
- **Token-level signals are lineage-specific instruments.** Gemma's mean logprob is
  *below chance* on clean cells (arithmetic logprob_full = .297) — the only model where
  preregistered P3 held.

## 3. Domain contrasts (tables/rq2_domain_contrasts*.csv)

Under strict QC (saturation + extraction exclusions) the knowledge−arithmetic selfconf gap
predicted by P1 does not exist in any lineage; in Llama and Gemma the sign reverses
(arithmetic selfconf is *better*). Sensitivity variants (flagged cells included; saturated
cells included) do not change any qualitative conclusion.

## 4. Information structure (tables/rq_cmi_decomposition.csv)

Per model × domain, plug-in MI with shuffled-control floors and joint-bootstrap CIs;
"unique" requires the CMI CI lower bound to clear the control floor:

- **I(C;Y|E) clears its floor in 12/12 model×domain cells** (range .009–.125 nats;
  typically 10–30× the floor).
- **I(E;Y|C) clears in only 2/12** (both ≤ .02 nats, barely above floor).
- ρ(C,E) spans −.20 to +.27 by domain — the two signals are not even monotonically
  related in a stable way.

Consistency contains what answer-entropy knows about correctness, plus a large unique
component; the converse is false. Replicated in three independent lineages.

## 5. Free-feature gate (tables/rq5_consistency_after_ff.csv)

Conditioned on the best zero-cost input feature per domain, consistency retains unique
information in **13/15 cells** (CMI .035–.231 vs floors .002–.019). The two failures are
the reasoning domain in Qwen and Llama — the domain where all signals are weakest.

## 6. Mechanism (tables/mechanism_*.csv)

- **Quadrants** (`mechanism_quadrants`): P(correct) is a staircase in C — C_low .28–.43,
  C_mid .41–.52, C_high .78–.82 (all models); within each C band, splitting by entropy
  moves accuracy by only a few points with no consistent direction.
- **Answer-distribution entropy vs token entropy** (`mechanism_entropy_variants`):
  the vote-distribution entropy over K samples discriminates at .695–.731 in all three
  models; token-level entropy ranges .494–.706 and the two are nearly uncorrelated
  (ρ = −.05 to .31; Gemma fully dissociated). "Entropy" is not one quantity: outcome
  uncertainty tracks correctness, token uncertainty does not reliably.
- **Path divergence** (`mechanism_path_divergence`): among high-consistency items,
  agreement reached on essentially one reasoning path is more reliable (.87–.88 correct)
  than agreement via divergent paths (.70–.78) — in all three models. Consistency partly
  measures trajectory stability, not merely answer agreement.
- **Wrong-but-consistent** (`mechanism_wrong_but_consistent`): a real population, not a
  grading artifact — 6.3–10.4% of items, 91–97% extraction-clean, concentrated in
  TruthfulQA/MuSR/StrategyQA/CommonsenseQA, on prompts 1.5–1.9× longer than average.
  Precision at C = 1 is .84–.88.

## 7. Elicitation robustness (tables/probe_wording_summary.csv)

Four wordings (frozen original, probability, frequency, inverted), M1, MMLU+GSM8K,
n = 100 each, 100% parse rate. All four collapse (≥ 94% modal reply; AUROC .47–.57, every
CI spanning .5). The inverted wording flips the modal reply from 100 to 0 — the model
understands the question's semantics and reports near-total certainty regardless of
correctness. The Qwen confidence collapse is a property of the model, not the prompt.

## 8. Extraction QC (tables/rq1_signal_benchmark.csv: ext_fail, ext_flagged)

Cells over the preregistered 10% threshold, excluded from primary contrasts (sensitivity
variants reported): Qwen — BBH 13.1%, GSM8K 12.5%, MMLU-Pro 46.5%; Llama — MMLU-Pro 32%,
MBPP 68.5% (unfenced code; fallback still graded, acc .625); Gemma — MMLU-Pro 53%
(97% truncation), MuSR 13.1%, StrategyQA 14.5%. Failure mechanisms diagnosed in
LIMITATIONS.md §11 (truncation at the frozen 512-token budget and format drift; no parser
bugs found).

## 9. Pending

- Seed sensitivity (PLAN §3: M1, seeds 1+2, five benchmarks): kernel built and pushed
  (`ugen-qwen25-3b-seeds`), not yet run — awaiting GPU quota. Until it runs, headline
  claims carry an implicit single-seed caveat.
- M4 (Phi-3.5-mini): not run; the model grid stands at 3/4 lineages.
