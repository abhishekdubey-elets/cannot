# LIMITATIONS

Standing entries (from PLAN.md, restated as they become operative):

1. **Planning domain absent.** ALFWorld/MiniWoB are interactive environments incompatible
   with the single-generation uncertainty protocol and with internet-off Kaggle kernels.
   Planning-flavoured static reasoning appears only via BBH tracking/temporal tasks and
   MuSR team_allocation. Claims are scoped to five domains, not six.
2. **GPQA excluded** — gated dataset, no authenticated HF token on the build machine. The
   knowledge domain rests on MMLU, MMLU-Pro, TruthfulQA.
3. **TruthfulQA option cap**: mc1 items with >10 options were reduced to the correct answer
   + first 9 distractors (A–J answer alphabet). Applied before shuffling; affects a minority
   of items; recorded in the loader.
4. **CommonsenseQA uses the validation split** (test is unlabelled).
5. **Small instruct models only (3–4.3B).** Findings may not extend to frontier scale.
6. **Likely pretraining contamination** of GSM8K/MMLU-class benchmarks for all models; we
   measure uncertainty-correctness relationships, not capability, but contamination could
   inflate confidence on memorized items. MMLU-Pro and MuSR are the lower-contamination
   anchors.
7. **Exact-match/execution grading is conservative** — unusual phrasings of correct answers
   are graded wrong. This depresses accuracy, not signal rankings; extraction-failure rates
   are reported per cell.
8. **Code consistency** uses output-agreement on prompt-visible example inputs only; items
   without such examples have consistency = NaN.
9. **Single elicitation wording** for self-confidence (frozen); a wording-sensitivity probe
   quantifies fragility on two benchmarks, one model.
10. **MI/PID are plug-in estimators** with positive bias; every table carries the shuffled-
    control floor, and strata below the pre-registered minimums are not decomposed.
