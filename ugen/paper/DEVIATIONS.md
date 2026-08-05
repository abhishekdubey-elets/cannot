# DEVIATIONS from the frozen plan

Every post-freeze change to design or protocol is logged here with timestamp and reason.

- 2026-08-05 (pre-data): TruthfulQA repo id updated to `truthfulqa/truthful_qa` (legacy id
  removed on HF). No design change.
- 2026-08-05 (pre-data): TruthfulQA >10-option items capped to 10 (correct + first 9
  distractors) — the A–J answer alphabet cannot address 13 options. Applied uniformly
  before shuffling; also entered in LIMITATIONS.md.
- 2026-08-05 (pre-data): GPQA not frozen — gated without an HF token. This is the fallback
  pre-stated in PLAN.md section 10.
