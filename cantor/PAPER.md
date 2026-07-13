# Why Self-Improving LLM Agents Don't Bootstrap
### A measured account of the data threshold, the cold start, and the headroom problem

*Draft. Section 5 (headline locus experiment) awaits the closability decision; everything
else is settled by the runs in [`EXPERIMENT_PLAN.md` §11](EXPERIMENT_PLAN.md).*

---

## Abstract

Consolidation loops for self-improving agents share a premise: an agent's own successful
trajectories, replayed into a parameter-efficient update, yield safe skill acquisition. We
test this premise directly on Qwen2.5-3B and find it **false as specified**, for reasons that
are mechanistic and measurable rather than incidental.

Three obstacles compound. **(1) A data threshold.** Below ~24 examples per schema, a LoRA
update is not merely ineffective but *actively harmful*: six examples drive held-out skill
from 0.500 (untrained) down to 0.083–0.250. The canonical loop consolidates on ~6
self-generated successes per night, so it **damages itself on every cycle**. **(2) A cold
start.** The replay buffer is seeded only by the agent's own correct solves, so on tasks
where base accuracy is zero the buffer never fills and no learning is possible — while on
tasks where base accuracy is high there is no headroom to demonstrate learning.
**(3) A measurement floor.** In the regime where the loop *does* operate, the effect size is
smaller than seed variance, so single-seed results are unreliable; we retract three of our own.

We contribute a **no-regression floor** that repairs (1); an eval-gate calibration rule
(`noise floor < eps < per-update drift`); and a **task-admissibility trilemma** — headroom,
bootstrap, and closability are mutually exclusive across the task space, so no self-generated
task can exhibit skill acquisition. The one exit, **demonstration-seeded replay**, works: on
the single task that is closable and high-headroom, seeding the buffer lets the parametric arm
reach near-perfect novel-instance transfer (0.00 → ~1.0) that memory cannot match — but only
on 2/3 seeds, the third rolled back by the gate on genuine forgetting. We further show a
concrete capability boundary: LoRA distils a **single-step rule** into the fast path but
**cannot distil any multi-step procedure** at 3B, wherever the difficulty sits. Finally we
report that a simulated harness passing every control-flow gate predicts **none** of these
effects — simulation validates pipelines, not effect sizes — and we document **four single-seed
claims the model forced us to retract**, arguing this is part of the contribution, not an
appendix to it.

---

## 1. Introduction

*(setting: consolidation / self-improvement loops; CANTOR as instantiation; the premise
under test)*

**Contributions**

1. **The data-threshold effect.** Small-data LoRA consolidation is *harmful*, not neutral
   (§3). This inverts the usual "more data helps" framing: below threshold, **not training
   is strictly better than training**.
2. **The no-regression floor** — an eligibility criterion on examples-per-schema — which
   converts the loop from self-damaging to learning (§4).
3. **An eval-gate calibration rule.** A gate that rejects updates on a probe set must satisfy
   `1/|probes| < eps < per-update drift`. Violating the left inequality rejects good updates
   on measurement jitter; violating the right blinds the gate to real damage (§4.2).
4. **A task-admissibility criterion** for locus/consolidation experiments: `base_fast` low
   (headroom), `base_deliberate` high (bootstrap), and the gap **closable** by the update
   rule. Existing synthetic families satisfy at most two (§5).
5. **A negative methodological result:** an offline simulation passing every gate (stability,
   locus, forward transfer, ablations) mispredicts every real magnitude, because its learner
   *assumed* mastery after two exposures (§6).

---

## 2. Setup

Qwen2.5-3B-Instruct, LoRA rank 16 (fp16, Tesla T4). Task streams of *schema* families with
held-out **novel** instances (skill transfer) and **exact repeats** (recall). Two arms:

* **B2 (memory)** — non-parametric: episodic → semantic consolidation, exact-instance cache.
* **B7 (hybrid)** — memory **+** eval-gated LoRA consolidation of self-generated successes.

Metrics: **NIST** (accuracy on held-out novel instances, terse/fast path), **exact_recall**
(exact repeats), **GCR** (change in general-probe accuracy = forgetting).

The locus hypothesis (**H2**) predicts a dissociation: `exact_recall` equal across arms
(memory's contribution) while `NIST` favours B7 (the weights' contribution).

---

## 3. The data threshold (learning-curve sweep)

No gate, no evaluator, no solve loop; ground-truth pairs; seeded.

| config | n | steps | rank | skill | dGeneral |
|---|---|---|---|---|---|
| baseline | 0 | — | 16 | **0.500** | +0.000 |
| n=6 | 6 | 25 | 16 | **0.083** | +0.000 |
| n=6 | 6 | 100 | 16 | 0.250 | +0.050 |
| n=6 | 6 | 300 | 16 | 0.250 | +0.050 |
| n=24 | 24 | 100 | 16 | **1.000**\* | +0.050 |
| n=24 | 24 | 300 | 16 | **1.000**\* | +0.050 |
| n=24 + 1:1 anchors | 24 | 300 | 16 | 0.750 | +0.050 |
| n=24 | 24 | 300 | 64 | 0.917 | +0.050 |

* **Not optimization-limited** (100 ≡ 300 steps at both data sizes).
* **Not capacity-limited** (rank 64 no better than 16).
* **Data-limited, decisively** (6 → 0.25; 24 → ≈1.0).
* **Below threshold, training is worse than not training** (0.500 → 0.083–0.250).
* **Generative replay at 1:1 costs skill** (1.000 → 0.750) and protected general capability
  that never degraded (`dGeneral ≥ 0` throughout).

\* Single seed. Three later independent probe readings give 0.62/0.62/0.75; treat ≈0.7 as the
honest post-training estimate. The *ordering* is robust (9-probe gap); the magnitude is not.

**Consequence.** The loop consolidated on `counts={'S0': 6}` per night — squarely in the
harmful regime. Every downstream symptom (NIST ≈ 0, schema-probe accuracy 0.00, apparent
"gate damage") reduces to this single cause.

---

## 4. Repairs

### 4.1 The no-regression floor
A schema is eligible for consolidation only once its buffer holds ≥ `min_examples` examples;
otherwise the update is skipped as *strictly harmful*. With `min_examples = 24`, every seed
now trains and promotes (`buffer=29–31`, `PROMOTED`), schema-probe accuracy rises to
**0.62–0.75** (from **0.00**), and **GCR = 0.000** everywhere: skill without forgetting.

### 4.2 Eval-gate calibration
A gate rejecting updates when general accuracy drops by more than `eps`, measured on `P`
probes, requires

> `1/|P|  <  eps  <  per-update drift`

Violating the **left** inequality makes single-probe jitter reject good updates (observed:
12-probe set, `eps=0.03`, noise floor 0.083). Violating the **right** makes the gate blind to
the damage it exists to catch (observed: `eps=0.05` vs drift 0.06). We initially attributed
the pipeline's rollbacks to *real* damage; the controlled sweep (`dGeneral ≥ 0` always)
**retracts** that: they were jitter.

### 4.3 Replay-target format
The buffer stored the model's **verbose deliberate** generation while the update rendered the
**terse** prompt and evaluation used the terse fast path — i.e. LoRA was taught
*"terse prompt → rambling answer."* Storing the **parsed answer** self-distils what
deliberation concluded, not how it rambled. Necessary, but not sufficient: the effect was
+0.083 NIST (2 items, n=1) — below resolution, and upstream of the data threshold.

---

## 5. Task admissibility and the locus experiment

**The band is a gap.** Buffer-filling uses the *deliberate* path; NIST measures the *fast*
path. An admissible task therefore needs `base_deliberate` high (bootstrap) **and**
`base_fast` low (headroom) — which *is* the System-2→System-1 distillation the architecture
presumes. Measured:

| task | base_fast | base_deliberate | admissible? |
|---|---|---|---|
| cryptic (hidden rule) | 0.000 | 0.062 | ✗ cold start |
| S-CTRL arithmetic | 0.312 | 0.875 | ✗ no headroom |
| 2d×1d + 2d | 0.000 | 1.000 | candidate |
| (2d+2d)×1d − 2d | 0.000 | 1.000 | candidate |

A third criterion is required and was initially omitted: the gap must be **closable** by the
update rule (post-LoRA fast accuracy ≥ 0.60). A `base_fast = 0` arising from *computational*
inability (multi-digit arithmetic in an 8-token budget) is not closable by LoRA, whereas one
arising from an unlearned single-step *rule* is.

### 5.2 The trilemma (closability check)

Post-LoRA fast accuracy on ground-truth pairs (24 train, 12 eval, disjoint; positive control
H0 = cryptic single-step rule, which the first bring-up lifted 0.00→1.00):

| task | base_fast | base_delib | post_fast | closable? |
|---|---|---|---|---|
| H0 cryptic (1-step rule) **[control]** | 0.250 | 0.333 | **0.833** | ✓ |
| H2 2d×1d + 2d (arith) | 0.000 | 1.000 | 0.167 | ✗ |
| H4 (2d+2d)×1d − 2d (arith) | 0.000 | 1.000 | 0.000 | ✗ (dGen **−0.400**) |
| H6 3-step chain (**map-hard, arith-easy**) | 0.000 | 0.833 | 0.000 | ✗ |
| H7 distractor chain | 0.583 | 1.000 | 0.917 | ✗ no headroom |

The control lifts; nothing else in-band does. **H6 refutes the natural hypothesis** that the
obstacle is arithmetic: H6 has trivial arithmetic and a multi-step *rule*, and still lifts
0/12. **The boundary is single-step vs. multi-step, not computation vs. rule.** LoRA distils a
one-step mapping into the fast path and **cannot distil any sequential procedure** at this
scale, wherever the difficulty sits. H4 further produces genuine catastrophic forgetting
(`dGen = −0.400`), the first observed here.

**This yields a trilemma.** Every task satisfies exactly two of {headroom, bootstrap,
closable}, never all three:

| | headroom | bootstrap | closable | blocked by |
|---|---|---|---|---|
| H2/H4/H6 | ✓ | ✓ | ✗ | multi-step ⇒ not distillable |
| H0 cryptic | ✓ | ✗ | ✓ | rule unknown ⇒ cannot self-solve |
| H1/H7 | ✗ | ✓ | ✓ | known 1-step rule ⇒ no room |

To *have* headroom the fast path must fail; it fails either because the rule is unknown
(no bootstrap) or the task is multi-step (not closable). **Self-generated consolidation cannot
demonstrate skill acquisition anywhere in this space** — structurally, not for want of tuning.

### 5.3 The only exit: seed the buffer (final experiment)

The trilemma's sole opening is H0 — closable and high-headroom, failing *only* bootstrap. If
self-generation cannot fill the buffer, supply demonstrations. We seed the parametric buffer
with 24 ground-truth demonstrations (disjoint from all eval sets) on the cryptic task; 3
seeds, paired deltas.

| seed | B2 NIST | B7 NIST | delta | B7 GCR | outcome |
|---|---|---|---|---|---|
| 0 | 0.000 | 0.958 | **+0.958** | 0.000 | consolidation landed |
| 1 | 0.042 | 0.042 | +0.000 | 0.000 | **eval-gate rollback** (general 1.000→0.875) |
| 2 | 0.000 | 1.000 | **+1.000** | +0.125 | consolidation landed |

`mean = +0.653 ± 0.566`, positive on **2/3**. **When the update survives, the effect is at
ceiling**: B7 acquires a rule that self-replay cannot bootstrap and memory cannot generalize
(NIST 0 → ~1.0). **But consolidation is not reliable**: on seed 1 the gate rolled back an
update that dropped general accuracy 0.125 > eps — the same gate whose S-CTRL rejections we
had dismissed as jitter (§4.2), here catching real damage (cf. H4's −0.400).

**Two limitations bound this result.** (i) The pre-registered every-seed bar is **not met**;
the effect is real but variance-prone. (ii) We seeded only the *parametric* buffer, not the
memory store, so on this cold-start task B2's memory is starved and the clean memory/weights
*dissociation* is not isolated — only the **novel-instance** channel (B2≈0 vs B7≈1 on landed
seeds) is valid. A symmetric-seeding design is required to isolate the locus, and the
trilemma suggests no *self-generated* task can (memory needs solvable instances, which
implies either no headroom or no closability).

### 5.1 Why the first locus attempt was uninformative
On S-CTRL, three seeds gave `B7 − B2 NIST = +0.167 ± 0.270`, **negative on one seed**
(t ≈ 0.87). `exact_recall` was identical across arms on every seed (0.958) — the memory half
of H2 holds exactly. The weights half is unresolvable there: the base model already solves
33–50% of novel items, so the maximum detectable effect (~0.2–0.35) is **smaller than seed
variance**. This is a property of the *measurement*, not the loop.

---

## 6. Simulation validates pipelines, not effect sizes

An offline harness reproduced the intended dynamics and passed every gate: stability across
seeds, locus (`GO`), forward transfer, and a replay×eval-gate ablation interaction. It
predicted **none** of §3–§5, because its learner *assumed* mastery after two exposures and a
drift-to-eps ratio favourable to the gate. The simulation was never wrong about what it
modelled — control flow — and never able to test the premise. We recommend treating simulated
magnitudes as **untestable priors**, not predictions.

---

## 7. Threats to validity

* **Single model, single scale** (Qwen2.5-3B, LoRA r16). The data threshold is presumably
  scale- and task-dependent; we establish its *existence*, not its value.
* **Seed variance dominates** small effects here (SD 0.27 on NIST). We report paired per-seed
  deltas and refuse claims from single seeds — having made, and retracted, three.
* **Parsing/truncation.** A 192-token deliberate budget truncated long reasoning; the
  last-integer parser then read an intermediate value, biasing deliberate accuracy *downward*
  for long-reasoning tasks (observed: `fast > deliberate` on 2d×2d). Fixed at 384 tokens.
* **H3 (forgetting) untested.** No configuration produced genuine forgetting
  (`GCR = 0.000` throughout, replay off), so we cannot claim the gate or replay are
  *necessary* — only that miscalibrating the gate is harmful.

---

## 8. Conclusion

The failure of self-generated consolidation to bootstrap is not a tuning problem. It is the
conjunction of a **data threshold** below which updates corrode the model (§3), a **trilemma**
by which no self-generated task can simultaneously offer headroom, bootstrap, and a
distillable skill (§5.2), and a **measurement floor** beneath which effects vanish into seed
variance (§5.1). Repairing the loop — a no-regression floor, a calibrated gate, correctly
formatted replay targets — is necessary and, alone, insufficient: the experiment must also be
*admissible*, and in this task space none is.

**What the parametric arm can and cannot do is now sharp.** It distils a *single-step rule*
into the fast path (post-LoRA 0.00 → 0.83, and, seeded, 0.00 → ~1.0 novel-instance transfer
that memory cannot match). It **cannot** distil a *multi-step procedure* at 3B, wherever the
difficulty lies (H2/H4/H6 all lift 0/12) — a concrete boundary on the System-2→System-1
distillation such architectures presume. And self-generation **cannot start the loop at all**
on the tasks where learning is visible; an external demonstration seed is required, and even
then consolidation succeeds only when its update survives a gate that — contra our own earlier
retraction — sometimes catches real forgetting (dGen −0.125 to −0.400).

The honest headline is therefore not "self-improving agents work" nor "they don't," but:
**self-generated replay cannot bootstrap skill acquisition; demonstration-seeded replay can,
for single-step skills, when the update survives — a narrow and unreliable competence, not the
general self-improvement the loop's premise assumes.**

The broader lesson is methodological. A simulated harness passing every control-flow gate —
stability, locus, forward transfer, ablation interactions — licensed confidence that real
weights immediately withdrew, because the simulation *assumed* the very thing under test
(mastery after two exposures). We asserted four results from single seeds — an H2 confirmation,
a probe-noise diagnosis, a "no real damage" claim, and a "mapping-hard fixes it" hypothesis —
and the model refuted every one. What survived contact were only the effects large enough to
be seen through the noise. We submit that in this regime, only those deserve to be claimed —
and that reporting the four retractions is not incidental to the contribution but part of it.

---

## References
*(to add: LoRA/PEFT; experience replay & continual learning; catastrophic forgetting;
self-improvement / STaR / RFT; System-1/System-2 distillation; CLS theory.)*
