# E2 precondition — a FREE feature beats every internal confidence signal on arithmetic

Offline re-analysis of `e1b_rows.json` (no GPU; the raw dump paying for itself a second time).
The question: on arithmetic, consistency reaches AUROC 0.688 — but is that unique information,
or is it shadowing a difficulty prior the controller could compute for free?

**Answer: the free feature does not merely match consistency, it dominates every internal
signal by a wide margin.**

## Arithmetic (n=72, accuracy 0.583)

| feature | AUROC [95% CI] | cost |
|---|---|---|
| **−complexity (operand digit product)** | **0.928 [0.86, 0.98]** | **zero model calls** |
| −carries (exact) | 0.900 [0.82, 0.96] | zero model calls |
| consistency | 0.687 [0.60, 0.77] | 5 samples |
| selfconf | 0.417 [0.29, 0.53] | 1 call |
| logprob | 0.416 [0.28, 0.55] | ~free |
| *shuffled control* | *0.546 [0.42, 0.69]* | — |

Operand digit count is **readable directly from the prompt**. It predicts correctness at 0.928
while the best model-derived signal reaches 0.687 and two of the three are *below chance*.

### Does anything survive conditioning on difficulty?

I(signal; correct | difficulty), against the shuffled control as the empirical noise floor:

| signal | MI | CMI given tier | CMI given carries |
|---|---|---|---|
| consistency | 0.102 | 0.031 | 0.031 |
| logprob | 0.072 | 0.040 | 0.043 |
| selfconf | 0.034 | 0.052 | 0.087 |
| *shuffled control* | *0.011* | *0.022* | *0.040* |

**No signal is distinguishable from the control after conditioning.** Consistency retains
0.031 against a control floor of 0.022–0.040. The control *gaining* information under
conditioning (0.011 → 0.040) exposes the plug-in estimator's positive bias on 18-item strata —
which is precisely why the control was measured rather than assumed. Within-tier AUROCs agree
and are weak: consistency 0.656 / 0.667 on the two tiers that have both labels, against a
control at 0.600 / 0.506.

**Verdict: consistency carries no demonstrable unique information about arithmetic correctness
beyond operand digit count.**

## Factoid — the opposite, and a leak I must flag

| feature | AUROC | note |
|---|---|---|
| −tier | 0.968 | **circular — do not use** |
| selfconf | 0.905 [0.82, 0.98] | legitimate |
| logprob | 0.847 [0.74, 0.93] | legitimate |
| consistency | 0.785 [0.68, 0.88] | legitimate |

The factoid `tier` is **not a free feature**: it is my benchmark's ground-truth annotation
(common / rare / relocated / fictional), and knowing an item is "fictional" is close to knowing
the answer does not exist. Its 0.968 is **leakage, not a baseline**, and I am not claiming free
features beat confidence on factoids. The honest position is that **no legitimate free proxy
exists here** — nothing in the surface form of *"What is the capital of Qorvania?"* versus
*"…of France?"* reveals which is answerable without world knowledge. On factoids, self-report
at 0.905 is doing real work that no prompt feature replicates.

## The regularity this suggests

> **Internal confidence earns its cost only when task difficulty is not computable from the
> input.** Computational difficulty is surface-legible — digit count sits in the prompt — and a
> free reading of it (0.928) dominates every internal signal (≤0.687). Knowledge difficulty is
> *not* surface-legible, and there self-confidence reaches 0.905 with no free substitute.

This reframes the knowledge/computational split from "confidence works here, fails there" to
something with an architectural consequence: **the executive should estimate difficulty from
the input first, and spend model calls on confidence only where the input does not reveal
difficulty.** For arithmetic that means never sampling five times — just count the digits.

Stated narrowly, and falsifiable: it predicts that on any task family where a cheap input
feature correlates with difficulty (long code, deep recursion, large operands, long documents),
internal confidence signals will fail to add unique information over that feature.

## Consequences for the roadmap

**E2 as scoped is dead.** A PID over signals that are jointly dominated by a free feature would
decompose a benchmark artifact — exactly the failure this phase was created to avoid. Two of
the three domains cannot support it: arithmetic (free feature dominates), reasoning (saturated,
one negative). PID would run on factoid alone, where entropy and consistency are one source
(ρ = 0.999), leaving three signals — too thin for a decomposition to be worth the compute.

**New mandatory gate for every future domain:** a *legitimate* free-feature baseline (computable
from the input, no ground-truth annotation) must be measured **before** any internal signal is
credited with information. E1 and E1b both credited consistency without this check.

**What to run next, in order:**
1. Fix the compositional domain so it has a real difficulty gradient (two attempts failed).
2. Measure the free-feature baseline on it — is compositional difficulty surface-legible
   (step count is visible in the prompt) or not? This directly tests the regularity above.
3. Only then consider decomposition, and only in domains where internal signals beat their
   free baseline.

## Threats to validity

Single model (3B); 72 arithmetic items with 18 per stratum, so CMI is noisy and biased (the
control quantifies it); `complexity` and `tier` are near-collinear by construction in
arithmetic (both derive from digit count), so they are not independent evidence — `carries`
(0.900) is the closest to an independent confirmation; the regularity rests on **one** domain
where difficulty is legible and one where it is not, which is a two-point line.
