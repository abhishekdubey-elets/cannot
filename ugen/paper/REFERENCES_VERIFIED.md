# Reference verification

Every reference in `PAPER.md` was checked against the primary source (authors, year, title,
venue) via web lookup on 2026-08-24. **All 21 entries are correct as cited.** No author-order,
year, title, or venue errors were found. Each is also cited in-text (§2), and every in-text
citation resolves to a list entry.

| # | cited as | verified | venue / id |
|---|---|:--:|---|
| 1 | Austin et al., 2021 — *Program Synthesis with Large Language Models* | ✓ | arXiv:2108.07732 (introduces MBPP) |
| 2 | Chen et al., 2021 — *Evaluating Large Language Models Trained on Code* | ✓ | arXiv:2107.03374 (Codex / HumanEval) |
| 3 | Clark et al., 2018 — *Think you have Solved Question Answering? Try ARC* | ✓ | arXiv:1803.05457 (ARC) |
| 4 | Cobbe et al., 2021 — *Training Verifiers to Solve Math Word Problems* | ✓ | arXiv:2110.14168 (GSM8K) |
| 5 | Farquhar, Kossen, Kuhn, Gal, 2024 — *Detecting hallucinations … semantic entropy* | ✓ | **Nature** 630(8017):625–630 |
| 6 | Geva et al., 2021 — *Did Aristotle Use a Laptop?* | ✓ | TACL 2021 (aclanthology 2021.tacl-1.21); StrategyQA |
| 7 | Guo, Pleiss, Sun, Weinberger, 2017 — *On Calibration of Modern Neural Networks* | ✓ | ICML 2017, PMLR v70:1321–1330 |
| 8 | Hendrycks et al., 2021 — *Measuring Massive Multitask Language Understanding* | ✓ | ICLR 2021 (MMLU) |
| 9 | Kadavath et al., 2022 — *Language Models (Mostly) Know What They Know* | ✓ | arXiv:2207.05221 |
| 10 | Kuhn, Gal, Farquhar, 2023 — *Semantic Uncertainty* | ✓ | ICLR 2023 (oral) |
| 11 | Lin, Hilton, Evans, 2022 — *Teaching Models to Express Their Uncertainty in Words* | ✓ | TMLR 2022, arXiv:2205.14334 |
| 12 | Lin, Hilton, Evans, 2022 — *TruthfulQA* | ✓ | ACL 2022 (2022.acl-long.229) |
| 13 | Patel, Bhattamishra, Goyal, 2021 — *Are NLP Models really able to Solve Simple Math Word Problems?* | ✓ | NAACL 2021 (2021.naacl-main.168); SVAMP |
| 14 | Roy & Roth, 2015 — *Solving General Arithmetic Word Problems* | ✓ | EMNLP 2015 (D15-1202); MultiArith |
| 15 | Sprague et al., 2024 — *MuSR* | ✓ | ICLR 2024, arXiv:2310.16049 |
| 16 | Suzgun et al., 2022 — *Challenging BIG-Bench Tasks…* | ✓ | arXiv:2210.09261 (BBH) |
| 17 | Talmor et al., 2019 — *CommonsenseQA* | ✓ | NAACL 2019 (N19-1421) |
| 18 | Tian et al., 2023 — *Just Ask for Calibration* | ✓ | EMNLP 2023 (2023.emnlp-main.330) |
| 19 | Wang et al., 2022 — *Self-Consistency Improves Chain of Thought Reasoning* | ✓ | ICLR 2023 (arXiv 2022) |
| 20 | Wang et al., 2024 — *MMLU-Pro* | ✓ | NeurIPS 2024 Datasets & Benchmarks |
| 21 | Williams & Beer, 2010 — *Nonnegative Decomposition of Multivariate Information* | ✓ | arXiv:1004.2515 |

## Note on count and completeness

- The reference list has **21** entries (the roadmap's "22 references" was approximate).
- **Recommended addition (not an error):** the three evaluated model lineages —
  Qwen2.5-3B-Instruct, Llama-3.2-3B-Instruct, Gemma-3-4B-IT — are named in §3 Methods but
  their technical reports are not cited. A camera-ready version should add the Qwen2.5
  Technical Report, the Llama 3 Herd of Models report, and the Gemma 3 Technical Report.
  These are left for the authors to insert with the exact citation each venue prefers
  (they were not auto-added here to avoid an unverified bibliographic entry).
- Optional venue upgrades for precision (all currently cited by short title, which is fine):
  Wang et al. 2022 and Kuhn et al. 2023 first appeared on arXiv in 2022 but were published
  at ICLR 2023; the paper cites both correctly by their commonly-used dates.
