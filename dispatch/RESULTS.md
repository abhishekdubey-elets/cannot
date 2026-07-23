# When Do Cognitive Architectures Help? — Measured Results

All numbers below are **measured** on Qwen2.5-3B-Instruct (Tesla T4, 3 seeds), via named
Kaggle kernels. Discipline: every experiment ran a **power/detectability analysis before the
GPU run** (`power.py`), and no accuracy claim is made where the benchmark lacked power to
detect it. Nothing here is a placeholder.

## The question

Instead of "can latent rollouts beat CoT," we ask when a *modular* architecture — a controller
dispatching queries to specialized mechanisms (fast, CoT, tool, retrieve) — beats a *monolith*
(the single best fixed mechanism). The whole value reduces to one measurable gap:
`oracle_dispatch − best_fixed_mechanism`, set entirely by workload heterogeneity.

## Gate 1 — easy workload: dispatch is overhead (`dispatch-calibrate`)

All facts in-context, small computation. Measured accuracy matrix:

| type | fast | cot | tool | retrieve |
|---|---|---|---|---|
| factual | 0.96 | 0.92 | 0.00 | 0.96 |
| arithmetic | 0.15 | 0.98 | 1.00 | 0.10 |
| word | 0.02 | 0.96 | 0.00 | 0.02 |
| recall | 1.00 | 1.00 | 0.00 | 1.00 |

`oracle=0.979, best_fixed(cot)=0.964` → **dispatch ceiling 0.016**. CoT is a universal solvent;
a controller cannot help. **NO-GO.**

## Gate 2 — hard workload: dispatch has real headroom (`dispatch-calibrate-hard`)

Two tasks require capabilities CoT structurally lacks: 7-digit×7-digit arithmetic, and an
out-of-context fact present only in an external registry.

| type | fast | cot | tool | retrieve | best |
|---|---|---|---|---|---|
| factual | 1.00 | 1.00 | 0.00 | 0.00 | fast |
| word | 0.12 | 0.96 | 0.00 | 0.00 | cot |
| big_arith | 0.00 | **0.00** | 1.00 | 0.00 | tool |
| kb_lookup | 0.00 | **0.00** | 0.00 | 1.00 | retrieve |

`oracle=0.990, best_fixed(cot)=0.490` → **dispatch ceiling 0.500**. CoT scores **0.00** on the
two tasks needing external computation/knowledge. **GO.**

## Gate 3a — controller, surface-separable queries (`dispatch-controller`)

A real LLM controller classifies each query, then routes to that type's mechanism.

| method | accuracy | 95% CI |
|---|---|---|
| best-fixed (cot) | 0.490 | ±0.010 |
| random dispatch | 0.365 | ±0.074 |
| **controller** | **0.990** | ±0.010 |
| oracle | 0.990 | ±0.010 |

**Routing accuracy 1.000.** The controller reaches the oracle: **captures 100% of the +0.50
gain.** When task types are distinguishable by surface form, a 3B model routes perfectly.

## Gate 3b — controller, surface-AMBIGUOUS queries (`dispatch-adversarial`)

Adversarial workload: known vs **unknown** facts look identical (a real capital vs a fictional
one), and easy vs hard arithmetic look identical. Routing now requires judgement, not pattern.

| method | accuracy | 95% CI |
|---|---|---|
| best-fixed (cot) | 0.490 | ±0.010 |
| **controller** | **0.740** | ±0.010 |
| oracle | 0.990 | ±0.010 |

Routing accuracy **overall 0.750**, **by type**:

| type | routing |
|---|---|
| known_fact | 1.000 |
| **unknown_fact** | **0.000** |
| easy_calc | 1.000 |
| hard_calc | 1.000 |

The controller still beats the monolith (+0.25) but now captures only **50%** of the oracle
gain. The entire shortfall localizes to one type: **unknown_fact routing is 0.000 across all
seeds** — the model never recognizes it does not know a fact, so it never dispatches to
retrieval. It routes by surface form flawlessly (3/4 types) but cannot introspect on its own
knowledge (0/4 on known-vs-unknown).

## Gate 4 — the calibration-signal lever: the bound is liftable (`dispatch-lever`)

Two-stage controller: stage 1 = coarse type classification (unchanged); stage 2 = for
fact-like queries only, a metacognitive probe decides known (→cot) vs unknown (→retrieve).

| method | accuracy | 95% CI | captured |
|---|---|---|---|
| best-fixed (cot) | 0.500 | 0.000 | — |
| baseline (Gate 3b) | 0.750 | 0.000 | 50% |
| **self-ask** ("do you know this?") | 0.847 | 0.036 | 69% |
| **consistency** (3 samples @ T=1.0; disagree ⇒ unknown) | **1.000** | 0.000 | **100%** |
| oracle | 1.000 | 0.000 | 100% |

Routing accuracy by type:

| type | baseline | self-ask | consistency |
|---|---|---|---|
| known_fact | 1.000 | **0.389** | 1.000 |
| unknown_fact | **0.000** | 1.000 | **1.000** |
| easy_calc | 1.000 | 1.000 | 1.000 |
| hard_calc | 1.000 | 1.000 | 1.000 |

**Consistency lifts the bound entirely** — unknown-fact routing 0.000 → 1.000 with known-fact
routing preserved at 1.000; the controller reaches the oracle (captured 100%).

**Self-report fails degenerately.** It also hits 1.000 on unknown_fact, but by claiming
ignorance almost everywhere: known_fact routing collapses to 0.389, sending real capitals to a
KB that lacks them. This is precisely why the pre-registration required *discrimination*
(`known ≥ 0.75`) and not just `unknown ≥ 0.50` — on the unknown-fact metric alone, self-ask
would have looked like a partial success.

> **Behavioural calibration works where introspective calibration fails.** A 3B model cannot
> *report* whether it knows a fact, but it reliably *exhibits* it: sampled three times it names
> the same real capital every time and a different invented capital each time. Three 12-token
> samples recover the full 50% of dispatch gain the classifier forfeited.

## The law (fully confirmed; no retraction)

> **Modular dispatch helps iff the task distribution demands disjoint capabilities** (Gate 1
> vs 2: ceiling 0.016 → 0.500), **and reaches the oracle iff the router can recognize which
> capability each query needs** (Gate 3a vs 3b: 100% → 50% captured). **The binding constraint
> is metacognitive:** a small model routes by surface form perfectly but cannot recognize its
> own ignorance, forfeiting exactly the gain the missing-knowledge capability offered.

## Scope / caveats (bounding the claim honestly)

* **3B, one model, hand-built 4-mechanism taxonomy.** The law is measured for this configuration.
* **Idealized tool/retriever** (exact calculator, exact KB lookup): this measures the *routing*
  question cleanly; a real RAG/tool pipeline stacks its own error on top.
* **Gate 3a's perfect routing is by construction** — the clean task types are surface-separable.
  Gate 3b is the honest stress test, and it is where the metacognitive bound appears.

## Provenance

`power.py` (CPU) ran before every GPU gate. Kernels: `dispatch-calibrate`,
`dispatch-calibrate-hard`, `dispatch-controller`, `dispatch-adversarial` (v3, re-run against
fully-fixed code after a data-grading bug was found and corrected — the v3 oracle 0.99 vs the
buggy 0.90 is the correction). All results/tracebacks captured to files, not stdout.
