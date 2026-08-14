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
11. **Extraction-QC diagnosis (M1, measured 2026-08-14).** Three benchmarks exceeded the
    preregistered 10% extraction-failure threshold and are excluded from primary contrasts
    (sensitivity variants reported). The failures are **not parser bugs**:
    - *MMLU-Pro 46.5%*: 100% of flagged rows hit the 512-token generation cap — 10-option
      CoT does not fit the frozen MAX_NEW budget, so the chain never reaches "Final
      answer:". Accuracy .065 among flagged vs .523 among clean rows → the .310 headline
      accuracy is substantially truncation-deflated. Model×budget behavior, disclosed, not
      repaired (protocol frozen).
    - *BBH 13.1%*: same mechanism (73% of flagged rows truncated).
    - *GSM8K 12.5%*: format drift, not truncation (36% truncated) — every flagged row still
      yielded a numeric prediction via the preregistered fallback (last number in response),
      accuracy .600 among them. The flag marks missing "Final answer:" markers, not lost
      answers.
    `extraction_ok=0` therefore means "answer-format marker absent (fallback parse used)",
    which is the conservative, model-comparable definition from grading.py.
