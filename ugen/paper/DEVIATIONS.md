# DEVIATIONS from the frozen plan

Every post-freeze change to design or protocol is logged here with timestamp and reason.

- 2026-08-05 (pre-data): TruthfulQA repo id updated to `truthfulqa/truthful_qa` (legacy id
  removed on HF). No design change.
- 2026-08-05 (pre-data): TruthfulQA >10-option items capped to 10 (correct + first 9
  distractors) — the A–J answer alphabet cannot address 13 options. Applied uniformly
  before shuffling; also entered in LIMITATIONS.md.
- 2026-08-05 (pre-data): GPQA not frozen — gated without an HF token. This is the fallback
  pre-stated in PLAN.md section 10.
- 2026-08-14 (analysis-side): the preregistered extraction-QC rule (PLAN 9.2, >10% failure
  → excluded from primary contrasts) was implemented in analyze.py after G1 revealed it had
  never been wired in. Strict primary tables + clearly-labelled sensitivity variants
  (flagged cells included). Rule content unchanged from preregistration.
- 2026-08-14 (analysis-side): P5 verdict row added to prediction_verdicts — mechanical
  evaluation of the preregistered benchmark-level falsifiers; previously the gate was
  computed (rq5) but never distilled to a verdict.
- 2026-08-14: wording-probe phrasings fixed at push time (probe_wording.py WORDINGS):
  probability, frequency, and inverted-scale framings. PLAN 9.9 preregistered "3
  alternative phrasings" without specifying text; the inverted framing was chosen to
  separate number-anchoring from semantic overconfidence.
- 2026-08-14/17 (infrastructure): Kaggle account moved to abhishekrajdhardubey (G1/G2,
  probes), then M3 routed to abhishekakgec for separate GPU quota. Datasets re-uploaded
  per account; no protocol content changed. GPU type (T4 x2) must be set in the web UI —
  the API silently drops docker_image/machine_shape (CLI 1.7.4.5).
- 2026-08-19 (engineering, PLAN 6 scope): OOM-ladder fix — retries moved outside the
  except block (exception tracebacks pinned the failed generate's tensors; Gemma OOMed at
  batch 1 with ~13.7 GB of dead allocations) plus a remembered working-batch cap. Same
  computation, same protocol.
- 2026-08-19 (engineering): M3 run as dual-GPU data parallelism — two pipeline processes,
  one per T4, disjoint benchmark halves (rebalanced once between sessions). Per-item
  computation byte-identical to single-GPU runs.
- 2026-08-20: master dataset frozen (outputs/master_v1, SHA256 manifest). All analyses
  read it; raw rows immutable.
- 2026-08-24 (analysis-side): independent statistical audit added (`analysis/audit.py` →
  `paper/AUDIT.md`) — a from-scratch second implementation of AUROC/MI/CMI/Spearman that
  shares no code with the analysis pipeline, reading only the frozen master rows. Confirms
  SHA256 integrity, the §2 signal map (±0.000 AUROC), P4 falsification, and the C-over-E
  information asymmetry. Surfaced one caveat now disclosed in PAPER §4.4 / RESULTS §4: with
  a multi-permutation bias floor the CMI cell count is 10/12 (not 12/12) for I(C;Y|E), the
  two shortfalls confined to the reasoning domain. No data or protocol change.
- 2026-08-24 (figures): main-text entropy figure F6 built
  (`analysis/make_entropy_figure.py` → `figures/entropy_vs_consistency_main.{png,pdf}`)
  from frozen tables only; registered in FIGURES.md. Consolidates the "entropy is not one
  quantity" result (outcome vs token entropy AUROC; CMI asymmetry; ρ≈0 between the two).
- 2026-08-24 (references): all 21 PAPER.md references verified against primary sources
  (`paper/REFERENCES_VERIFIED.md`) — no errors. Recommendation logged there: add the three
  model technical reports (Qwen2.5 / Llama 3 / Gemma 3) for camera-ready.
- OPEN: seed-sensitivity run (PLAN 3) built and pushed but not yet executed (GPU quota,
  runs on Kaggle — not locally); M4/Phi-3.5 not run — model grid is 3/4. Both flagged in
  RESULTS.md 9.
