# Signal dashboard — per model (n per cell in rq1 table)

AUROC vs correctness per benchmark. `sat` = saturation-excluded (acc outside [0.10, 0.90]); `QC` = >10% extraction failure, excluded from primary contrasts. Signals: higher = better discrimination; 0.5 = chance.

## Llama-3.2-3B-Instruct

| Benchmark | Domain | Acc | flags | Self-conf | Consistency | Logprob | Logprob(ans) | Entropy | Best free feat |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| mmlu | knowledge | 0.655 |  | 0.674 | 0.717 | 0.576 | 0.533 | 0.588 | 0.593 (prompt_words) |
| mmlu_pro | knowledge | 0.370 | QC | 0.742 | 0.781 | 0.584 | 0.547 | 0.594 | 0.605 (prompt_words) |
| truthfulqa | knowledge | 0.595 |  | 0.570 | 0.714 | 0.612 | 0.531 | 0.614 | 0.591 (n_options) |
| gsm8k | arithmetic | 0.770 |  | 0.600 | 0.889 | 0.781 | 0.547 | 0.803 | 0.631 (prompt_words) |
| svamp | arithmetic | 0.860 |  | 0.795 | 0.865 | 0.819 | 0.728 | 0.844 | 0.735 (digit_count) |
| multiarith | arithmetic | 0.983 | sat | 0.616 | 0.655 | 0.701 | 0.634 | 0.704 | 0.562 (prompt_words) |
| commonsenseqa | commonsense | 0.620 |  | 0.610 | 0.727 | 0.518 | 0.496 | 0.534 | 0.529 (question_words) |
| arc_challenge | commonsense | 0.845 |  | 0.594 | 0.798 | 0.580 | 0.564 | 0.580 | 0.536 (prompt_words) |
| strategyqa | commonsense | 0.675 |  | 0.589 | 0.635 | 0.529 | 0.585 | 0.525 | 0.558 (prompt_chars) |
| bbh | reasoning | 0.641 |  | 0.666 | 0.638 | 0.599 | 0.554 | 0.600 | 0.563 (n_options) |
| musr | reasoning | 0.434 |  | 0.497 | 0.589 | 0.492 | 0.620 | 0.483 | 0.566 (question_words) |
| humaneval | coding | 0.543 |  | 0.554 | 0.893 | 0.549 | — | 0.571 | 0.756 (prompt_words) |
| mbpp | coding | 0.625 | QC | 0.704 | 0.856 | 0.679 | — | 0.678 | 0.556 (prompt_words) |

## Qwen2.5-3B-Instruct

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

| Model | Prediction | Statistic | 95% CI / falsifiers | Verdict |
|---|---|---|---|---|
| Llama-3.2-3B-Instruct | P1 | dAUROC=-0.083 | [-0.143,-0.022] | **FALSIFIED** |
| Llama-3.2-3B-Instruct | P2 | d_cons=-0.170 vs d_self=-0.083 |  | **SUPPORTED** |
| Llama-3.2-3B-Instruct | P3 | logprob_full=0.800, answer=0.637 |  | **FALSIFIED** |
| Llama-3.2-3B-Instruct | P4 | min_rho=0.009 |  | **FALSIFIED** |
| Llama-3.2-3B-Instruct | P5 | gsm8k: ff=0.631 best_int=0.889; svamp: ff=0.735 best_int=0.865; mmlu: ff=0.593 selfconf=0.674; truthfulqa: ff=0.591 selfconf=0.570 | internal>ff on gsm8k; internal>ff on svamp; ff>selfconf on truthfulqa | **FALSIFIED** |
| Qwen2.5-3B-Instruct | P1 | not evaluable: contrast pool empty after saturation + extraction-QC exclusions |  | **INCONCLUSIVE** |
| Qwen2.5-3B-Instruct | P1 (sensitivity: QC-flagged cells included; exploratory) | dAUROC=+0.008 | [-0.033,+0.039] | **FALSIFIED** |
| Qwen2.5-3B-Instruct | P2 | not evaluable (QC) |  | **INCONCLUSIVE** |
| Qwen2.5-3B-Instruct | P2 (sensitivity: QC-flagged cells included; exploratory) | d_cons=-0.116 vs d_self=+0.008 |  | **SUPPORTED** |
| Qwen2.5-3B-Instruct | P3 | not evaluable: no unsaturated unflagged arithmetic cell |  | **INCONCLUSIVE** |
| Qwen2.5-3B-Instruct | P3 (sensitivity: QC-flagged cells included; exploratory) | logprob_full=0.725, answer=0.655 |  | **FALSIFIED** |
| Qwen2.5-3B-Instruct | P4 | min_rho=0.001 |  | **FALSIFIED** |
| Qwen2.5-3B-Instruct | P5 | mmlu: ff=0.660 selfconf=0.511; truthfulqa: ff=0.627 selfconf=0.551 [QC-excluded: gsm8k] | ff>selfconf on mmlu; ff>selfconf on truthfulqa | **FALSIFIED** |
