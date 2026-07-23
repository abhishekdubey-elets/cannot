# When Do Cognitive Architectures Help? A Powered Study of Mechanism Dispatch and Its Metacognitive Bound

*Draft for a workshop / short-paper venue. Every number is measured on Qwen2.5-3B-Instruct
(Tesla T4, 3 seeds); no result is a placeholder, and each accuracy claim is backed by an
a-priori power analysis. Companion code and per-kernel logs: `dispatch/`.*

---

## Abstract

Cognitive-architecture proposals for LLM agents share a premise: distinct reasoning mechanisms
(fast recall, chain-of-thought, tools, retrieval) should **cooperate** under a controller,
rather than a single mechanism doing everything. We test this premise directly, and reduce it
to one measurable quantity — the gap between an oracle that dispatches each query to its best
mechanism and the best single fixed mechanism.

We find a sharp, three-part law. **(1)** When every task is solvable by in-context reasoning,
chain-of-thought is a *universal solvent*: the dispatch ceiling is **0.016**, so a controller
is pure overhead. **(2)** When the task distribution demands capabilities CoT structurally
lacks — out-of-context knowledge, exact large-number arithmetic — the ceiling opens to
**0.500**, and a real LLM controller that classifies-then-routes captures **100%** of it,
reaching the oracle, because the task types are separable by surface form. **(3)** When we make
the types *surface-ambiguous* (a real capital vs a fictional one; small vs large arithmetic),
the controller captures only **50%**, and the entire lost half localizes to a single failure:
it routes *unknown* facts to reasoning instead of retrieval **0% of the time** — the model
never recognizes that it does not know something.

The binding constraint on mechanism dispatch is therefore **metacognitive**, not
architectural. **(4)** But it is not intrinsic: replacing the router's *self-report* with a
*behavioural* calibration signal — sample the answer three times and check whether the samples
agree — restores unknown-fact routing to **1.000** and lifts the controller to the **oracle
(100% captured)**. Asked "do you know this?", the model is uninformative (it claims ignorance
everywhere, collapsing known-fact routing to 0.389); *watched* answering, it is perfectly
discriminative. Self-knowledge the model cannot report, it nonetheless exhibits.

---

## 1. Introduction

A recurring proposal for making LLMs "reason like humans" is a modular cognitive architecture:
an executive controller dispatching queries among specialized mechanisms — a fast pattern
matcher, a slow deliberator (CoT), a tool interface, a memory/retrieval system — each suited to
a different kind of query. The intuition is that different problems need different machinery,
so a committee of specialists should beat any single generalist.

This intuition is rarely tested against its null. A cognitive architecture is only worth its
latency and complexity if **dispatch beats the best single mechanism** on the workload. We make
that comparison the object of study.

**Reduction.** The maximum advantage any controller can win is
$$\Delta_{\max} = \mathrm{acc}(\text{oracle dispatch}) - \mathrm{acc}(\text{best fixed mechanism}),$$
which is set entirely by whether the workload has *mechanism specialization* — tasks on which
different mechanisms are best. If $\Delta_{\max}$ is small, no controller can help, regardless
of how clever it is. We therefore measure $\Delta_{\max}$ first, and only then ask whether a
real controller captures it.

**Contributions.**
1. A **three-gate empirical law** for when modular dispatch helps, each gate power-analysed
   before it was run (§4–§6).
2. The identification of the **metacognitive bound**: dispatch reaches the oracle iff the
   router can recognize which capability a query needs, and the hardest such recognition —
   *do I know this fact?* — fails completely at 3B (§6).
3. A methodological stance carried from a companion study: **compute the null and required
   sample size before running the method.** We apply it to every gate.

---

## 2. Setup

**Model.** Qwen2.5-3B-Instruct, fp16, single Tesla T4. Greedy decoding.

**Mechanisms** (four ways to process the same query, differing in *capability*, not just
prompt): `fast` (terse, one short generation), `cot` (chain-of-thought, long generation),
`tool` (a whitelisted arithmetic evaluator — exact computation), `retrieve` (a lookup in an
external knowledge base — out-of-context knowledge). The tool and retriever are *idealized*
(exact); this isolates the *routing* question from retriever/tool engineering (see §7).

**Metric.** Task accuracy (grading: string-substring for name answers, last-integer match for
numeric). All comparisons use identical queries and identical grading across mechanisms.

**Baselines.** `best_fixed` (the single mechanism with the highest mean accuracy — the
monolith null), `random` dispatch, `oracle` dispatch (route by true type — the ceiling), and
the `controller` (an LLM that classifies each query, then routes to that type's mechanism).

**Statistics.** 3 seeds; 95% CIs from the seed distribution. Power analysis (`power.py`)
computes $\Delta_{\max}$ and required $n$ from the measured accuracy matrix before the
controller is built.

---

## 3. Power analysis first

Before any controller run, `power.py` (CPU) takes an accuracy matrix $A[\text{type}][\text{mech}]$
and reports: the oracle, the best-fixed null, $\Delta_{\max}$, a controller's accuracy as a
function of its routing accuracy, and the sample size to detect the controller beating the null.
The calculation is self-validating — its analytic $n$ reproduces 0.80 Monte-Carlo power. Under
a plausible specialized profile it predicted $n\approx 16$ queries/type suffice to detect a
+0.22 effect; we used that sizing throughout. **No accuracy verdict is issued on a benchmark
whose $\Delta_{\max}$ or $n$ was not established in advance.**

---

## 4. Gate 1 — when one mechanism spans the tasks, dispatch is overhead

An "easy" workload (four types, all solvable in-context: capital-city facts, small word
problems, small arithmetic, in-prompt recall). Measured matrix:

| type | fast | cot | tool | retrieve |
|---|---|---|---|---|
| factual | 0.96 | 0.92 | 0.00 | 0.96 |
| arithmetic | 0.15 | 0.98 | 1.00 | 0.10 |
| word | 0.02 | 0.96 | 0.00 | 0.02 |
| recall | 1.00 | 1.00 | 0.00 | 1.00 |

$\mathrm{oracle}=0.979$, $\mathrm{best\_fixed(cot)}=0.964$, so $\Delta_{\max}=\mathbf{0.016}$.
**CoT is a universal solvent**: careful step-by-step reasoning handles factual lookup, small
arithmetic, word-problem mapping, and in-context reading. A perfect controller beats always-CoT
by under two points. **NO-GO** — building a controller here would be measured overhead. This
explains why frontier "reasoning" systems are one elaborated mechanism rather than a committee:
at this task difficulty, the specialist advantage does not exist.

---

## 5. Gate 2 — when capabilities are disjoint, dispatch has real headroom

A "hard" workload replaces two types with tasks CoT *structurally* cannot do: **big_arith**
(7-digit × 7-digit exact products) and **kb_lookup** (a fact present only in an out-of-context
registry, absent from both the weights and the prompt). Measured matrix:

| type | fast | cot | tool | retrieve | best |
|---|---|---|---|---|---|
| factual | 1.00 | 1.00 | 0.00 | 0.00 | fast |
| word | 0.12 | 0.96 | 0.00 | 0.00 | cot |
| big_arith | 0.00 | **0.00** | 1.00 | 0.00 | tool |
| kb_lookup | 0.00 | **0.00** | 0.00 | 1.00 | retrieve |

$\mathrm{oracle}=0.990$, $\mathrm{best\_fixed(cot)}=0.490$, so $\Delta_{\max}=\mathbf{0.500}$.
CoT scores **exactly 0.00** on both hard tasks — it cannot multiply seven-digit numbers in-head,
and it cannot know a fact that exists only in an external store. No single mechanism exceeds
0.49 average (each specialist is useless outside its lane). **GO.** Gates 1 and 2 together give
the first half of the law: *dispatch helps iff the workload spans capabilities no single
mechanism has.*

---

## 6. Gate 3 — a real controller, and the metacognitive bound

The oracle (0.99) assumes perfect routing. The real question is whether an LLM controller can
*recognize* which capability each query needs.

### 6.1 Surface-separable queries: the controller reaches the oracle

On the hard workload (types distinguishable by surface form — a giant multiplication looks
nothing like a registry lookup):

| method | accuracy | 95% CI |
|---|---|---|
| best-fixed (cot) | 0.490 | ±0.010 |
| random dispatch | 0.365 | ±0.074 |
| **controller** | **0.990** | ±0.010 |
| oracle | 0.990 | ±0.010 |

**Routing accuracy 1.000** on all seeds. The controller *is* the oracle: it captures **100%**
of the +0.50 gain. When query types differ in surface form, a 3B model routes flawlessly.

### 6.2 Surface-ambiguous queries: the bound appears

We make routing require judgement, not pattern. `known_fact` (a real capital) and
`unknown_fact` (a *fictional* country, answerable only by retrieval) are surface-identical;
`easy_calc` and `hard_calc` likewise. Result (confirmed on a clean re-run against corrected
grading code):

| method | accuracy | 95% CI |
|---|---|---|
| best-fixed (cot) | 0.490 | ±0.010 |
| **controller** | **0.740** | ±0.010 |
| oracle | 0.990 | ±0.010 |

The controller still beats the monolith (+0.25) but captures only **50%** of the gain. The
routing-by-type diagnostic localizes the loss precisely:

| type | routing accuracy |
|---|---|
| known_fact | 1.000 |
| **unknown_fact** | **0.000** |
| easy_calc | 1.000 |
| hard_calc | 1.000 |

The model routes three of four types **perfectly**, and the unknown-fact type at **0.000 across
all seeds**. It treats "capital of Tavisnik?" identically to "capital of Japan?", routes both to
reasoning, and confidently hallucinates on the fictional one rather than dispatching to
retrieval. **It can tell what kind of task it is looking at, but it cannot tell whether it knows
the answer.** The lost half of the dispatch gain is exactly the retrieval capability that
recognizing ignorance would have unlocked.

### 6.3 The law

> **Modular dispatch helps iff the task distribution demands disjoint capabilities**
> (Gate 1→2: $\Delta_{\max}$ 0.016 → 0.500), **and reaches the oracle iff the controller can
> recognize which capability each query needs** (Gate 3a→3b: 100% → 50% captured). **The
> binding constraint is metacognitive:** surface-form routing is easy; recognizing one's own
> ignorance is not, and it fails completely at 3B, forfeiting precisely the capability
> (retrieval) that missing knowledge requires.

---

## 7. Scope and threats to validity

We bound the claim deliberately.

* **One model, one scale (3B), one hand-built 4-mechanism taxonomy.** The law is measured for
  this configuration; the *magnitudes* are configuration-specific.
* **Idealized tool and retriever** (exact calculator, exact KB lookup). This is a deliberate
  choice — it isolates the *routing* question from tool/retriever error — but a deployed
  RAG/tool pipeline stacks its own error on top, so real end-to-end dispatch would fall below
  our oracle.
* **Gate 3a's perfect routing is partly by construction.** Clean types are surface-separable;
  the 1.000 is a property of the taxonomy, not evidence that routing is easy in general. Gate 3b
  is the honest stress test, and it is where the bound appears — so the *contrast* 3a→3b, not
  3a's absolute number, is the load-bearing result.
* **The unknown-fact routing failure could be prompt-sensitive.** A router explicitly asked to
  estimate its own confidence, or given a calibration signal, might recover some routing (see
  §8). We measured a natural classify-then-route controller; we do not claim 0.000 is
  irreducible, only that it is what a straightforward controller does.

---

## 8. Implications and next steps

**For cognitive-architecture design.** The result reframes the design target. The hard part is
not building specialist mechanisms or a router over surface features — a small model does that
perfectly. The hard part is **metacognition**: a controller that knows *when it doesn't know*,
so it can dispatch to retrieval. The measured 0.000 unknown-fact routing is a concrete target:
turning it positive is worth exactly the 50% of the dispatch gain it currently forfeits.

**The lever experiment — MEASURED (Gate 4).** We pre-registered and ran it. A two-stage
controller keeps stage-1 type classification and adds, for fact-like queries only, a
metacognitive probe deciding known (→cot) vs unknown (→retrieve). Two cheap signals:

| method | accuracy | captured | known_fact routing | unknown_fact routing |
|---|---|---|---|---|
| baseline (Gate 3b) | 0.750 | 50% | 1.000 | **0.000** |
| self-ask ("do you know this?") | 0.847 | 69% | **0.389** | 1.000 |
| **consistency** (3 samples @ T=1.0) | **1.000** | **100%** | **1.000** | **1.000** |
| oracle | 1.000 | 100% | — | — |

**The bound is liftable.** The consistency probe recovers unknown-fact routing entirely
(0.000 → 1.000) *without* damaging known-fact routing, and the controller reaches the oracle:
the full +0.50 dispatch gain, captured.

**But self-report fails degenerately**, and the manner of failure is the finding. It also
reaches 1.000 on unknown_fact — by claiming ignorance nearly everywhere, collapsing known_fact
routing to 0.389 and sending real capitals to a KB that does not contain them. Our
pre-registration required *discrimination* (`known ≥ 0.75`), not merely `unknown ≥ 0.50`,
precisely so this could not be scored as success; on the unknown-fact metric alone it would
have appeared to work.

> **Behavioural calibration succeeds where introspective calibration fails.** A 3B model cannot
> *report* whether it knows a fact — asked directly, it is uninformative — but it reliably
> *exhibits* the distinction: sampled three times it returns the same real capital every time,
> and a different invented capital each time. Self-knowledge the model cannot articulate is
> nonetheless recoverable from its behaviour, at a cost of three 12-token samples.

This upgrades the law's final clause: the metacognitive bound of §6.3 is **not** intrinsic to
the scale — it is a property of *how the controller is asked*. Routing on self-report leaves
half the dispatch gain on the table; routing on behavioural consistency captures all of it.

**Generalization.** More mechanisms, ambiguous *natural* queries (not a crisp 4-way taxonomy),
larger models, and a non-idealized retriever are all needed before the law generalizes beyond
the measured configuration. We report a clean, powered result on a controlled setting and mark
its edges rather than overclaiming past them.

---

## 9. Conclusion

Asking "how should an LLM reason like a human" as an architecture question invites unfalsifiable
diagrams. Asking it as a measurement — *does dispatch beat the monolith, and by how much* —
yields a sharp law: cooperation is overhead when one mechanism spans the tasks, worth up to the
full capability gap when they are disjoint, and capturable in full only when the router can
recognize which capability each query needs. At 3B, the last condition fails on the one
recognition that matters most — knowing what you don't know — which caps modular dispatch at
half its potential on ambiguous workloads.

That cap, however, is a property of *how the router is interrogated*, not of the scale. Asking
the model to introspect fails; observing whether it answers consistently succeeds completely,
restoring the full dispatch gain for the cost of three short samples. The frontier of cognitive
architectures is therefore metacognitive calibration — and the practical lesson is that such
calibration should be read from a model's **behaviour**, not solicited from its **self-report**.

---

## Appendix — provenance

Power analysis: `power.py` (CPU, before every gate). Kaggle kernels (Qwen2.5-3B, T4, 3 seeds):
`dispatch-calibrate` (Gate 1), `dispatch-calibrate-hard` (Gate 2), `dispatch-controller`
(Gate 3a), `dispatch-adversarial` (Gate 3b, v3 re-run against corrected grading code — the
correction moved the oracle from a buggy 0.90 to the true 0.99; the routing findings are
classification results, unaffected by the grading fix). All results and tracebacks were written
to downloadable files, not stdout. Measured numbers are recorded in `RESULTS.md`.
