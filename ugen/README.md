# UGEN — do uncertainty signals generalize across reasoning domains?

Pre-registered study (frozen: [PLAN.md](PLAN.md)); deviations logged in
[paper/DEVIATIONS.md](paper/DEVIATIONS.md). Successor to `../execev`, on standard
benchmarks only.

## Layout

- `pipeline/` — shared package (stats core, grading, dataset freezing, kernel runner,
  free features). Dependency-free where it runs on Kaggle.
- `datasets/` — frozen benchmark subsets (JSONL + MANIFEST with indices and SHA256).
  13 benchmarks, 2,540 items, seed 0. Regenerate: `python -m ugen.pipeline.prep_datasets`.
- `experiments/` — Kaggle push tooling (`push_kernel.py`), audit trail in `pushed/`.
- `analysis/` — offline only, reads raw CSVs, never edits them:
  `analyze.py` (tables RQ1–RQ5 + P1–P5 verdicts) → `make_figures.py` → `regression.py`.
- `outputs/` — append-only raw kernel outputs (fetched via `push_kernel.py fetch <slug>`).
- `tables/`, `figures/` — generated artifacts (CSV is the record of authority).
- `paper/` — RESULTS / DISCUSSION / LIMITATIONS / DEVIATIONS / FIGURES / TABLES /
  SUPPLEMENTARY, updated only from measured numbers.

## Kaggle workflow (internet-off kernels)

```bash
python experiments/push_kernel.py src    # refresh code dataset  (abhishekakgec/ugen-src)
python experiments/push_kernel.py data   # refresh frozen data   (abhishekakgec/ugen-data)
python experiments/push_kernel.py s0     # smoke test (M1, 8 items/benchmark)
python experiments/push_kernel.py g1     # full M1 sweep (add --resume-from <slug> to continue)
python experiments/push_kernel.py probe m3   # CPU-only model-attachment check
python experiments/push_kernel.py fetch ugen-s0-qwen25-3b   # download outputs
```

Before trusting any GPU numbers: `python pipeline/tests/test_stats.py` and
`python pipeline/tests/test_grading.py` must pass (the runner also re-runs the stats
self-test in-kernel before generating a single row).
