# TABLES index

All generated under `tables/` (CSV is the record; md/tex where present are conveniences).

## Primary
- `rq1_signal_benchmark` — per model × benchmark × signal: AUROC/AUPRC/MI + bootstrap CIs,
  Brier/NLL/ECE/MCE, shuffled controls, saturation + extraction-QC flags. The map.
- `rq2_domain_contrasts` — preregistered domain contrasts (strict QC), permutation p,
  Holm-corrected within the primary family.
- `prediction_verdicts` — P1–P5 per model, mechanical, with QC-driven INCONCLUSIVE rows
  and labelled sensitivity rows.
- `rq_cmi_decomposition` — I(C;Y), I(E;Y), I(C;Y|E), I(E;Y|C) per model × domain with
  control floors + joint-bootstrap CIs. Central result.
- `m1_dashboard` (md) — human-readable two-model-era name, now covers all three models +
  verdicts.

## Sensitivity / robustness
- `rq2_domain_contrasts_sensitivity` — extraction-flagged cells included.
- `rq2_domain_contrasts_saturation_sensitivity` — saturated benchmarks included.
- `rq5_free_feature_gate`, `rq5_consistency_after_ff` — zero-cost input-feature
  conditioning; consistency survives 13/15 cells.
- `probe_wording_summary` — 4-wording elicitation probe (M1).

## Mechanism (Phase C)
- `mechanism_quadrants` — P(correct | C band × E half) per model.
- `mechanism_wrong_but_consistent` — the convergent-error population.
- `mechanism_confounds` — per-benchmark ρ(C, length/prompt-size/difficulty) summaries.
- `mechanism_entropy_variants` — answer-distribution vs token entropy (sidecar subset).
- `mechanism_path_divergence` — prefix-agreement analysis among high-C items.

## Exploratory
- `rq2_transfer_matrix` — cross-benchmark calibration transfer (ECE/Brier/selective).
- `rq3_cross_model` — cross-model AUROC profile correlations.
- `rq4_signal_correlations`, `rq4_disagreement_patterns` — signal geometry.
- `rq5_pid` — Williams–Beer PID (selfconf × consistency; logprob × selfconf) vs shuffled
  controls.
