# M1 dashboard — Qwen2.5-3B-Instruct (n per cell in rq1 table)

AUROC vs correctness per benchmark. `sat` = saturation-excluded (acc outside [0.10, 0.90]); `QC` = >10% extraction failure, excluded from primary contrasts. Signals: higher = better discrimination; 0.5 = chance.

| Benchmark | Domain | Acc | flags | Self-conf | Consistency | Logprob | Logprob(ans) | Entropy | Best free feat |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| mmlu | knowledge | 0.660 |  | 0.511 | 0.702 | 0.633 | 0.596 | 0.637 | 0.660 (prompt_words) |
| mmlu_pro | knowledge | 0.310 | QC | 0.509 | 0.758 | 0.495 | 0.634 | 0.498 | 0.599 (prompt_words) |
| truthfulqa | knowledge | 0.545 |  | 0.551 | 0.696 | 0.603 | 0.514 | 0.604 | 0.627 (n_options) |
| gsm8k | arithmetic | 0.845 | QC | 0.516 | 0.867 | 0.725 | 0.655 | 0.749 | 0.643 (prompt_words) |
| svamp | arithmetic | 0.955 | sat | 0.611 | 0.845 | 0.866 | 0.616 | 0.861 | 0.734 (digit_count) |
| multiarith | arithmetic | 0.989 | sat | 0.500 | 1.000 | 0.388 | 0.404 | 0.326 | 0.942 (prompt_chars) |
| commonsenseqa | commonsense | 0.700 |  | 0.546 | 0.740 | 0.611 | 0.616 | 0.616 | 0.533 (prompt_chars) |
| arc_challenge | commonsense | 0.870 |  | 0.494 | 0.834 | 0.584 | 0.538 | 0.592 | 0.559 (question_words) |
| strategyqa | commonsense | 0.640 |  | 0.523 | 0.610 | 0.547 | 0.589 | 0.546 | 0.520 (prompt_words) |
| bbh | reasoning | 0.525 | QC | 0.540 | 0.637 | 0.565 | 0.612 | 0.562 | 0.585 (n_numbers) |
| musr | reasoning | 0.475 |  | 0.511 | 0.598 | 0.423 | 0.448 | 0.426 | 0.606 (prompt_chars) |
| humaneval | coding | 0.738 |  | 0.500 | 0.793 | 0.787 | — | 0.789 | 0.695 (prompt_chars) |
| mbpp | coding | 0.680 |  | 0.516 | 0.949 | 0.661 | — | 0.669 | 0.601 (prompt_words) |

## Preregistered prediction verdicts

| Prediction | Statistic | 95% CI / falsifiers | Verdict |
|---|---|---|---|
| P1 | not evaluable: contrast pool empty after saturation + extraction-QC exclusions |  | **INCONCLUSIVE** |
| P1 (sensitivity: QC-flagged cells included; exploratory) | dAUROC=+0.008 | [-0.033,+0.039] | **FALSIFIED** |
| P2 | not evaluable (QC) |  | **INCONCLUSIVE** |
| P2 (sensitivity: QC-flagged cells included; exploratory) | d_cons=-0.116 vs d_self=+0.008 |  | **SUPPORTED** |
| P3 | not evaluable: no unsaturated unflagged arithmetic cell |  | **INCONCLUSIVE** |
| P3 (sensitivity: QC-flagged cells included; exploratory) | logprob_full=0.725, answer=0.655 |  | **FALSIFIED** |
| P4 | min_rho=0.001 |  | **FALSIFIED** |
| P5 | mmlu: ff=0.660 selfconf=0.511; truthfulqa: ff=0.627 selfconf=0.551 [QC-excluded: gsm8k] | ff>selfconf on mmlu; ff>selfconf on truthfulqa | **FALSIFIED** |
