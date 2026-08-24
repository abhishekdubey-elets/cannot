# Conference LaTeX build

Self-contained conversion of `paper/PAPER.md` (numbers reconciled to the frozen tables and
verified by `paper/AUDIT.md`; references verified in `paper/REFERENCES_VERIFIED.md`).

## Build

```bash
tectonic main.tex          # fetches packages on demand, runs bibtex, emits main.pdf
```

or with a standard TeX Live install:

```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Files

- `main.tex` — the manuscript (8 pp). Uses only widely available packages.
- `references.bib` — 21 verified references.
- `figures/entropy_vs_consistency_main.pdf` — main-text Figure 1 (F6).
- `figures/domain_auroc_Qwen2.5-3B-Instruct.pdf`, `figures/reliability_Qwen2.5-3B-Instruct.pdf`
  — copied in for optional appendix use (not yet `\includegraphics`d in the body).

## Switching to a specific venue

Replace the block marked `CONFERENCE STYLE` at the top of `main.tex` with the venue class and
delete the `geometry`/`times` lines, e.g.:

```latex
\documentclass{article}\usepackage[final]{neurips_2024}   % NeurIPS
% or
\documentclass{article}\usepackage{iclr2025_conference,times}  % ICLR
```

Drop the venue's `.sty` next to `main.tex`. The body, tables, figure, and bibliography are
venue-agnostic. Add the three model technical reports (Qwen2.5 / Llama 3 / Gemma 3) to
`references.bib` and cite them in §3 for camera-ready (see `paper/REFERENCES_VERIFIED.md`).
