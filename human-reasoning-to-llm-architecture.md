# From Human Reasoning to Next-Generation Reasoning Engines
### An Engineering-Oriented Survey of Cognitive Mechanisms and Their Translation into LLM and Agentic Architectures

*A technical research report. Depth target: graduate survey. Orientation: computational and implementable, not descriptive psychology.*

---

## Executive Summary

The central engineering claim of this report is that today's frontier LLMs are, in cognitive terms, an extraordinarily powerful **System 1** — a parallel, amortized pattern-completion engine — bolted to a **weak, non-persistent, non-verifying System 2** that we simulate in-context with chain-of-thought and agent loops. Almost every open frontier problem (hallucination, long-horizon planning, continual learning, reliable tool use, calibration) is a symptom of the same root cause: **we have no true controller, no persistent typed memory, no world model separable from the token stream, and no learning signal that fires at inference time.**

The productive research direction is therefore *not* "make the transformer more brain-like at the neuron level." Backprop is not Hebbian plasticity and does not need to be. The productive direction is **architectural**: identify the *functional decomposition* the brain uses — fast recognition vs. slow deliberation, working vs. episodic vs. semantic vs. procedural memory, a generative world model used for simulation, a metacognitive controller that allocates effort and estimates confidence — and build these as **explicit, separable, individually-trainable modules** with typed interfaces, rather than hoping they emerge from scale inside one monolithic autoregressive network.

The report proceeds in ten parts: (1) a computational teardown of ~21 human reasoning mechanisms; (2) an analysis of what current models (GPT, Claude, Gemini, DeepSeek, Qwen, Llama, Mistral) actually approximate and where transformers fundamentally diverge from cognition; (3) a full cognition→component→gap→architecture mapping table; (4) a modular "GPT-10" design; (5) a concrete dual-system routing architecture with controller logic and cost math; (6) a critique of whether humans reason like ReAct/ToT/MCTS agents; (7) a multi-store memory architecture with consolidation; (8) world models, predictive processing and active inference for LLMs; (9) a research-frontier survey with per-paper problem/method/innovation/limitation notes; and (10) an original synthesized architecture — **CANTOR** (Cognitive Architecture for Neural Thought, Orchestration, and Reasoning) — with ASCII diagrams, execution pipeline, training/inference strategy, cost analysis, benchmarks, and open problems.

A recurring caution runs throughout: **biological plausibility is not an engineering objective.** It is a source of hypotheses. Airplanes do not flap. The value of neuroscience here is that evolution has already solved sample-efficient continual learning, effort allocation under metabolic budget, and robust generalization from tiny data — and studying *how it factored the problem* is cheaper than rediscovering the factoring by trial and error at $100M/training-run.

---

# PART 1 — HUMAN REASONING: A COMPUTATIONAL TEARDOWN

For each mechanism: **biological purpose → computational interpretation → strengths → weaknesses → neuroscience evidence**, ending with the **engineering takeaway** — the transferable design idea.

## 1.1 System 1 vs System 2 (Dual-Process Architecture)

**Biological purpose.** Metabolic economy. The brain runs on ~20 W. Deliberation is expensive (prefrontal, sustained, serial), so evolution defaults to cheap, fast, parallel pattern completion (System 1) and recruits expensive serial control (System 2) only when the fast path signals difficulty, novelty, conflict, or high stakes.

**Computational interpretation.** System 1 is **amortized inference**: a function approximator that has compiled the answer to frequently-encountered problems into a single feed-forward pass (O(1) recognition). System 2 is **iterative/algorithmic inference**: search, simulation, and working-memory manipulation with a variable number of serial steps (O(n) or worse). The crucial insight is that this is not two systems but a **compute-allocation spectrum**: the same problem can be solved by cached recognition (System 1) or by re-derivation (System 2), and a controller decides which based on expected error and cost.

**Strengths.** System 1: near-instantaneous, massively parallel, effortless, handles the 99% of life that is routine. System 2: compositional, can handle novel problems by re-combining primitives, can *verify* and *correct*.

**Weaknesses.** System 1: systematic biases (anchoring, availability, base-rate neglect), overconfident on out-of-distribution inputs, cannot represent explicit uncertainty well. System 2: slow, serially bottlenecked, capacity-limited (see working memory), effortful and thus lazily invoked — the famous result that people accept System 1's fluent-but-wrong answer rather than pay System 2's cost (Frederick's Cognitive Reflection Test).

**Neuroscience evidence.** Kahneman & Tversky's heuristics-and-biases program; Kahneman (2011) *Thinking, Fast and Slow*; Stanovich & West's naming of the systems; Evans & Stanovich (2013) on the "default-interventionist" architecture. Neurally, deliberation correlates with dorsolateral prefrontal cortex (dlPFC) and anterior cingulate (ACC) engagement; automatic responses run through basal ganglia and posterior cortical pattern stores.

**Engineering takeaway.** This is the single most important mapping in the report. **A frontier LLM forward pass *is* System 1.** Chain-of-thought is a *crude in-context simulation* of System 2 — but it runs in the same weights, has no separate verifier, and cannot be independently trained. The design lesson (Part 5) is to make the two systems *architecturally distinct*, with a *controller* that routes based on a *confidence estimate*, exactly as the brain gates PFC recruitment on ACC conflict signals.

## 1.2 Working Memory (WM)

**Biological purpose.** Hold and manipulate a small amount of task-relevant information *right now* — the mental scratchpad for reasoning, language comprehension, and action.

**Computational interpretation.** A small, high-bandwidth, rapidly-writable, actively-maintained buffer with a **central executive** (attention controller) plus modality-specific stores (Baddeley & Hitch's phonological loop, visuospatial sketchpad, and later the episodic buffer). Computationally: a set of addressable slots or **attractor states** in recurrent PFC circuits held active by persistent firing, gated by dopaminergic signals that decide what enters and what is protected from interference.

**Strengths.** Fast read/write; content is *bound to roles* (variable binding: "the *red* block is *on* the *blue* one"); supports manipulation, not just storage.

**Weaknesses.** Brutally capacity-limited: Miller's "7±2" (1956) chunks, revised down to **~4 independent items** (Cowan, 2001). Content decays without rehearsal and is vulnerable to interference. This limit is arguably the *defining constraint* of human cognition and the reason we invented external memory (writing, notebooks).

**Neuroscience evidence.** Persistent delay-period firing in dlPFC (Fuster, Goldman-Rakic); the capacity limit maps to a small number of coherent gamma cycles nested in a theta cycle (Lisman & Idiart's theta-gamma model → ~7 items). Load-dependent PFC BOLD signal saturating at ~4 items.

**Engineering takeaway.** The transformer's **context window + attention over the KV cache is a very large, non-decaying working memory** — arguably *superhuman* in raw capacity. But it lacks two WM properties: (a) *active maintenance/gating* (deciding what to keep vs. drop — everything in context is passively retained until evicted), and (b) *manipulable typed slots with variable binding* (attention approximates binding softly and unreliably). A designed system wants an explicit, writable, typed scratchpad *separate* from the raw token buffer (Part 7).

## 1.3 Long-Term Memory (LTM)

**Biological purpose.** Persist knowledge and experience across a lifetime, far beyond WM's seconds-scale horizon.

**Computational interpretation.** Massive, content-addressable, associative store with slow write (consolidation) and fast, cue-driven read (pattern completion). Divided into **declarative** (explicit: episodic + semantic) and **non-declarative** (implicit: procedural, priming, conditioning). The brain uses **Complementary Learning Systems** (McClelland, McNaughton & O'Reilly, 1995): a fast-learning hippocampus that stores specific episodes with minimal interference, and a slow-learning neocortex that extracts statistical structure over many replays.

**Strengths.** Enormous capacity; graceful, cue-based retrieval; generalization emerges from neocortical consolidation.

**Weaknesses.** Reconstructive and fallible (memories are re-synthesized, not replayed — hence false memories); retrieval is cue-dependent and biased; slow to consolidate durable knowledge.

**Neuroscience evidence.** Patient H.M. (bilateral hippocampectomy → dense anterograde amnesia with intact procedural learning) dissociated declarative from procedural memory and WM from LTM. Squire's taxonomy; place/replay cells in hippocampus.

**Engineering takeaway.** LLMs conflate everything into one store: **parametric memory (weights)**. This is *slow-write, consolidated, semantic* memory — the neocortex analog — with **no hippocampus**: no fast, one-shot, episodic write that can be laid down at inference time and later consolidated. This is the deepest architectural gap in current models and the reason they cannot learn from a conversation. CLS is the blueprint for the fix (Parts 7 and 10).

## 1.4 Episodic Memory

**Biological purpose.** Remember specific events — what happened, where, when — enabling one-shot learning, autobiographical continuity, and (crucially) *mental time travel* into the past and future.

**Computational interpretation.** Rapid, sparse, pattern-separated encoding of a bound conjunction (event + spatiotemporal + emotional context) into a single trace, retrievable by partial cue (pattern completion). The hippocampal dentate gyrus performs **pattern separation** (orthogonalizing similar experiences to avoid interference); CA3's recurrent net performs **pattern completion** (reconstructing the whole from a fragment).

**Strengths.** One-shot: you remember a single salient event forever. Indexed by rich context, enabling "where did I leave my keys" retrieval.

**Weaknesses.** Low capacity relative to experience stream (most episodes are forgotten); reconstructive distortion; retrieval-induced modification.

**Neuroscience evidence.** Tulving (1972, 1983) defined and dissociated episodic from semantic. Hippocampal "time cells" and replay during rest/sleep. Future imagining recruits the same hippocampal-default-mode network as past recall (Schacter & Addis, "constructive episodic simulation").

**Engineering takeaway.** This is *exactly* what RAG and vector-DB "agent memory" gesture at, but poorly: current episodic memory for agents is unstructured text chunks with cosine-similarity retrieval — no pattern separation (near-duplicates collide), no importance-weighted consolidation, no binding of event/time/outcome, no mechanism to *modify* traces. A proper episodic store (Part 7) needs typed event schemas, salience-gated writes, and separation of storage from a learned retrieval index.

## 1.5 Semantic Memory

**Biological purpose.** Store decontextualized general knowledge — facts, concepts, word meanings, category structure — abstracted away from the episodes that produced it.

**Computational interpretation.** A structured, compositional knowledge base: concepts as points/regions in a high-dimensional feature space, related by typed edges (is-a, part-of, causes). Emerges via slow neocortical consolidation that *averages over episodes* to extract regularities (the "gist").

**Strengths.** Compact, generalizable, compositional, supports inference over relations.

**Weaknesses.** Loses specifics; can be wrong if the statistical regularities are misleading; slow to update (which is *good* for stability — the stability/plasticity tradeoff).

**Neuroscience evidence.** Semantic dementia (anterior temporal lobe atrophy) degrades concepts while sparing episodic recency; the ATL as a "semantic hub" integrating modality-specific spokes (Patterson, Nestor & Rogers, 2007).

**Engineering takeaway.** LLM weights *are* a semantic memory — a superb one, storing far more relational knowledge than any human. The gaps: it is **not editable** (you cannot correct one fact without retraining or brittle model-editing like ROME/MEMIT), it **cannot cite provenance**, and it **cannot be told apart from confabulation** at read time. A designed system separates a *queryable, editable, provenance-tagged* semantic store from the parametric prior.

## 1.6 Pattern Recognition

**Biological purpose.** Instantly classify sensory input and situations into learned categories to enable fast, appropriate action.

**Computational interpretation.** Hierarchical feature extraction culminating in similarity-based matching against stored prototypes/exemplars — i.e., learned nonlinear embedding + nearest-prototype readout. This is the substrate of System 1.

**Strengths.** Fast, parallel, robust to noise and partial input, generalizes across surface variation.

**Weaknesses.** Overgeneralizes (sees faces in clouds, patterns in noise — apophenia); confidently wrong out-of-distribution; hard to introspect ("I just know").

**Neuroscience evidence.** Ventral visual stream hierarchy (V1→V2→V4→IT) with increasing receptive-field size and abstraction; face-selective regions (FFA); the entire deep-learning revolution is a validated computational homage to this hierarchy (Hubel & Wiesel → Fukushima's Neocognitron → CNNs).

**Engineering takeaway.** This is what deep nets do *best* and what transformers inherited. It is a solved-ish strength, not a gap. The lesson is the *reverse*: recognize that this strength is exactly what makes hallucination hard — a pattern-completer will always complete the pattern, plausibly, even with no grounding. The fix is not better recognition but an *added verification stage* (System 2).

## 1.7 Mental Simulation

**Biological purpose.** Predict the consequences of actions and events without executing them — "look before you leap" — enabling planning and risk-free evaluation.

**Computational interpretation.** Run the internal **generative world model** forward from a current state under a hypothetical action, sampling a rollout of predicted future states. Formally: sampling trajectories from a learned transition model `s_{t+1} ~ p(s|s_t, a_t)` and evaluating them.

**Strengths.** Enables planning, counterfactuals, and imagination without real-world cost; supports tool/skill transfer.

**Weaknesses.** Accumulates prediction error over long horizons (the world model drifts); biased by imaginability; expensive if run in detail.

**Neuroscience evidence.** Hippocampal forward replay/preplay of future trajectories before navigation decisions; motor imagery activates motor cortex; the default mode network as a simulation engine. Barsalou's grounded/embodied simulation.

**Engineering takeaway.** LLMs *can* simulate crudely by generating text descriptions of consequences, but they have **no separable, queryable, rollout-able world model** — the "simulation" is entangled with language generation and inherits its unreliability (Part 8). A dedicated simulation module with a latent world model and explicit rollout is a major design target.

## 1.8 Mental Models

**Biological purpose.** Maintain a coherent, manipulable internal representation of a system (physical, social, abstract) to support explanation and prediction.

**Computational interpretation.** A structured, relational representation (objects, properties, relations, dynamics) that supports **inference by inspection/manipulation** rather than by rule-application. Johnson-Laird's theory: humans reason by building and inspecting models of possibilities, not by applying formal logic — which explains our logic errors (we fail to consider all models) and our competence (we simulate the likely models).

**Strengths.** Flexible, supports "what if," transfers across domains, human-interpretable.

**Weaknesses.** We build *too few* models (typically one) and neglect alternatives, causing systematic reasoning errors (belief bias, failure on modus tollens).

**Neuroscience evidence.** Johnson-Laird's mental-models program; the parietal/prefrontal network for relational reasoning; representational similarity showing structured relational codes in PFC.

**Engineering takeaway.** LLMs have *implicit, distributed* mental models that are often incoherent across a long generation (they contradict themselves because there is no single maintained model, only locally-coherent token prediction). Designing an explicit, updatable structured state (a "belief state" or scene graph) that generation is *conditioned on and must remain consistent with* is a route to coherence.

## 1.9 Causal Reasoning

**Biological purpose.** Distinguish causation from correlation to enable effective action and control of the environment.

**Computational interpretation.** Build and query a **causal model** (Pearl's structural causal models / DAGs) supporting three rungs: association `P(y|x)`, intervention `P(y|do(x))`, and counterfactuals. Humans do rapid, approximate causal induction from sparse data using priors (agents cause, contact causes, temporal precedence).

**Strengths.** Enables intervention and generalization to novel situations; supports explanation and blame/credit assignment.

**Weaknesses.** Prone to illusory causation, over-attribution to agents, confounding by salient covariates.

**Neuroscience evidence.** Infants show causal expectations (Michotte launching effect); dlPFC and the "causal" parietal network; developmental work by Gopnik on children as intuitive causal-Bayes-net learners.

**Engineering takeaway.** Transformers learn **statistical/associational** structure (rung 1). They mimic causal language but do not reliably perform `do`-calculus or counterfactual reasoning — they will confidently give correlational answers to interventional questions. Pearl's "ladder of causation" is a precise diagnostic of the gap. Integrating explicit causal-graph induction, or training on interventional/counterfactual data, is an active frontier.

## 1.10 Counterfactual Reasoning

**Biological purpose.** Reason about "what would have happened if" — essential for credit assignment, learning from single outcomes, regret, and planning.

**Computational interpretation.** Rung 3 of Pearl's ladder: fix a factual world, perform a minimal intervention, and re-run the (causal, world) model forward. Requires holding two worlds — actual and hypothetical — simultaneously.

**Strengths.** Enables one-shot learning ("if I had braked earlier..."), moral reasoning, and off-policy evaluation.

**Weaknesses.** Combinatorially explosive; humans sample only a few "close" counterfactuals, biased toward controllable/recent factors (norm theory).

**Neuroscience evidence.** Regret activates orbitofrontal cortex; counterfactual credit assignment linked to ACC; patients with vmPFC damage show blunted counterfactual/regret processing.

**Engineering takeaway.** Directly enabled by a good world model + simulation module. In RL terms, counterfactual reasoning is *off-policy credit assignment / hindsight* (cf. Hindsight Experience Replay). For LLM agents, structured counterfactual rollout ("simulate the branch we didn't take, compare outcomes") is a high-value reflection primitive currently done only in ad-hoc text.

## 1.11 Analogical Reasoning

**Biological purpose.** Transfer knowledge from a familiar domain to a novel one by mapping relational structure — the engine of human flexibility and creativity.

**Computational interpretation.** **Structure mapping** (Gentner, 1983): align two representations by their *relational* (not surface) structure and project inferences across the alignment. Formally a graph-matching / structural-alignment problem prioritizing systematic higher-order relations.

**Strengths.** Enables rapid transfer, scientific insight, and learning novel domains from one example.

**Weaknesses.** Surface similarity often captures retrieval (we retrieve the wrong analog); mapping is error-prone; hard to find the *right* source analog.

**Neuroscience evidence.** Frontopolar cortex (BA 10) for relational integration; rostrolateral PFC scales with relational complexity (Christoff, Bunge).

**Engineering takeaway.** LLMs are *surprisingly good* at analogy (Mitchell's work shows partial ability on letter-string and Raven-style problems) because embeddings encode relational structure — this is an emergent strength. But they lack *deliberate, verifiable* structure mapping and fail on novel abstract analogies where memorized patterns don't help. Explicit relational representations + a mapping operator would strengthen this.

## 1.12 Planning

**Biological purpose.** Sequence actions to achieve goals over time, trading present effort for future reward.

**Computational interpretation.** Search over a tree/graph of action sequences using a world model, guided by value estimates and goals — **model-based control** (Daw et al., 2005, distinguishing model-based from model-free/habitual RL). Humans mix cheap model-free habits with expensive model-based planning, arbitrated by reliability.

**Strengths.** Flexible, handles novel goals, supports hierarchical decomposition and subgoaling.

**Weaknesses.** Combinatorial explosion (planning is PSPACE-hard in general); humans prune aggressively and satisfice; myopic discounting.

**Neuroscience evidence.** Model-based planning engages dlPFC and hippocampus (for simulation); model-free habits engage dorsolateral striatum. Prefrontal lesions impair goal maintenance and planning (Tower of London deficits).

**Engineering takeaway.** LLMs plan *implicitly and unreliably* in a single forward pass. Explicit planning — ToT, MCTS, planner-executor — is bolted on in-context. The brain's **model-based/model-free arbitration** is a direct design template: cache successful plans as fast "habits," fall back to expensive search only when the habit's predicted reliability is low (Part 5, Part 10).

## 1.13 Reflection

**Biological purpose.** Review one's own outputs, detect errors, and revise — self-correction without external feedback.

**Computational interpretation.** A second-pass evaluation of a first-pass output against goals/constraints, generating an error signal used to revise. Requires a *self-model* and a *critic*. Distinct from metacognition (below) in that reflection acts on *content*; metacognition acts on *the reasoning process*.

**Strengths.** Catches errors cheaply (cheaper than acting them out); improves quality iteratively.

**Weaknesses.** Limited by the quality of the internal critic (if you can't detect the error, you can't fix it); can rationalize rather than correct.

**Neuroscience evidence.** ACC/medial PFC error-related negativity (ERN) — a scalp EEG signal ~100ms after an erroneous response, *before* external feedback, showing an internal error monitor.

**Engineering takeaway.** Reflexion (Shinn et al., 2023) and self-refine are direct implementations. The key limitation is shared with humans: **an LLM's self-critique is only as good as its ability to detect its own errors**, and models are often *worse* at verifying than generating in domains lacking a ground-truth checker. Hence the value of *external* verifiers (tools, tests, other agents) — the RLVR insight (Part 2).

## 1.14 Metacognition ("thinking about thinking")

**Biological purpose.** Monitor and control one's own cognition — know what you know, allocate effort, decide when to stop.

**Computational interpretation.** A **meta-level** that observes the object-level (Nelson & Narens' framework): *monitoring* (confidence, feeling-of-knowing, judgment-of-learning) feeds *control* (allocate study time, choose strategy, decide to answer or defer). Computationally, a learned mapping from internal states (e.g., processing fluency, competing-hypothesis entropy) to control actions.

**Strengths.** Enables adaptive effort, strategy selection, and knowing-when-to-stop; foundation of calibration.

**Weaknesses.** Often miscalibrated (Dunning-Kruger); metacognitive access is partial and post-hoc.

**Neuroscience evidence.** Metacognitive accuracy correlates with anterior prefrontal cortex (aPFC/BA10) gray matter and connectivity (Fleming & Dolan); dissociable from object-level performance.

**Engineering takeaway.** This is arguably **the single largest missing module** in current LLMs. There is no meta-level: the model cannot reliably report its own confidence (logits are miscalibrated after RLHF), cannot decide *how hard to think* (fixed compute per token unless a scaffold forces more), and cannot decide when to stop or defer. The **Executive Controller / Confidence Estimator** (Parts 4, 5, 10) is the engineering embodiment of metacognition and the linchpin of the whole design.

## 1.15 Goal-Directed Behavior

**Biological purpose.** Pursue internally-represented objectives flexibly, subordinating immediate stimuli to longer-term aims.

**Computational interpretation.** Maintain a goal representation in WM that biases perception, action selection, and value computation (goal-conditioned policy/value). Requires goal maintenance against distraction and subgoal decomposition.

**Strengths.** Flexible, context-independent behavior; enables long-horizon achievement.

**Weaknesses.** Goal maintenance is fragile (distraction, goal neglect); competing goals cause interference; reward hacking / goal misgeneralization.

**Neuroscience evidence.** dlPFC goal maintenance; goal-neglect after frontal damage (Duncan); dopaminergic value signals in striatum.

**Engineering takeaway.** LLM agents receive a goal as a prompt but have **no persistent, protected goal representation** — the goal decays as context fills, causing the well-known "loses the plot on long tasks" failure. An explicit **Goal Manager** (a stack of active goals/subgoals with maintenance and re-injection) is needed for long-horizon reliability.

## 1.16 Attention

**Biological purpose.** Selectively route limited processing resources to task-relevant information, suppressing distractors.

**Computational interpretation.** A dynamic gating/weighting over inputs — **exactly the operation transformer attention names**, though the brain's version is both *bottom-up* (salience-driven, superior colliculus/pulvinar) and *top-down* (goal-driven, frontoparietal control network), and it *gates what enters WM*, not just what is read.

**Strengths.** Enables focus under overload; binds features; top-down control enables goal-relevant selection.

**Weaknesses.** Serial bottleneck (attentional blink, inattentional blindness — the invisible gorilla); limited number of simultaneous foci.

**Neuroscience evidence.** Posner cueing; Desimone & Duncan biased-competition model; the dorsal (goal) and ventral (salience) attention networks (Corbetta & Shulman).

**Engineering takeaway.** Transformers have superb *content-based* attention but it is *undirected by an explicit controller* — every head attends per learned weights, with no top-down "focus here because the goal demands it" signal, and no gating of what enters memory. An **Attention Controller** that biases retrieval and context construction toward goal-relevant content (and that *drops* irrelevant content to fight context dilution) is a design target (Parts 4, 10).

## 1.17 Chunking

**Biological purpose.** Overcome WM's tiny capacity by grouping elements into higher-order units, effectively expanding capacity through compression.

**Computational interpretation.** Learn recurring patterns and replace them with a single pointer/symbol — hierarchical compression that turns `4 items` of `7 elements each` into a manageable structure (the chess-master result: experts chunk board positions, novices see 25 pieces, experts see 5 formations).

**Strengths.** Massive effective capacity gain; enables expertise; foundation of skill.

**Weaknesses.** Domain-specific (a chess master's chunks don't help at go); requires extensive learning to form.

**Neuroscience evidence.** Chase & Simon's chess studies; Ericsson's expertise/skilled-memory theory; hippocampal-cortical binding of elements into unitized representations.

**Engineering takeaway.** Chunking maps to **learned tokenization/abstraction and hierarchical latent representations** — and to *macro-actions/skills* in agents (a learned subroutine invoked as one unit). It motivates hierarchical reasoning (below), learned skill libraries (Voyager-style), and *context compression* (summarizing spans into dense tokens — a live research area, e.g., Gist tokens).

## 1.18 Hierarchical Reasoning

**Biological purpose.** Manage complexity by decomposing problems and behavior into nested levels of abstraction (goals → subgoals → actions).

**Computational interpretation.** **Hierarchical RL / hierarchical planning**: a manager sets subgoals, a worker achieves them (options framework; feudal RL; Botvinick's hierarchical RL). Enables temporal abstraction — reasoning at the level of "cook dinner," not "contract this muscle fiber."

**Strengths.** Tames combinatorial explosion; enables transfer of subroutines; matches problem structure.

**Weaknesses.** Choosing the right decomposition is hard; wrong hierarchy is worse than none.

**Neuroscience evidence.** A rostro-caudal PFC gradient: anterior PFC handles abstract/temporally-extended goals, posterior handles concrete actions (Koechlin, Badre). Hierarchical structure in prefrontal representations.

**Engineering takeaway.** Directly motivates **planner-executor and hierarchical-planning agent designs** (Part 6, 10): a slow, abstract planner emits subgoals; a fast executor (possibly a smaller model) realizes them. The PFC gradient suggests using *different modules/model-sizes for different abstraction levels* — a concrete efficiency win.

## 1.19 Learning from Feedback

**Biological purpose.** Update behavior to increase future reward using outcome signals.

**Computational interpretation.** **Reinforcement learning via reward-prediction error (RPE)**: dopamine encodes `δ = r + γV(s') − V(s)`, the exact TD-error of temporal-difference learning (Schultz, Dayan & Montague, 1997 — one of the most beautiful theory/data matches in neuroscience). Learning is proportional to *surprise*, not reward magnitude.

**Strengths.** Learns from sparse, delayed, evaluative feedback; online and continual.

**Weaknesses.** Credit assignment over long delays; sample inefficiency for model-free; catastrophic if reward is misspecified.

**Neuroscience evidence.** Dopaminergic RPE (Schultz); ventral striatum value coding; the entire RL-neuroscience bridge.

**Engineering takeaway.** RLHF and RLVR *are* feedback learning — but applied **only at training time, in a batch, to the weights**. Humans learn from feedback *continually and at inference time*. The gap: **no online learning loop.** An agent that fails a task learns nothing durable (unless we log it to an episodic store and retrieve it — a workaround, not real weight updates). Closing this is the "learning loop" of Part 10 and the hardest open problem.

## 1.20 Curiosity

**Biological purpose.** Drive exploration and learning in the absence of external reward — seek information and novelty to build better world models.

**Computational interpretation.** **Intrinsic motivation**: reward proportional to *learning progress*, prediction error, or information gain (Schmidhuber's compression-progress theory; Oudeyer's learning-progress; count-based/curiosity-driven exploration in RL — Pathak et al., 2017). Formally, reward `∝ reduction in model uncertainty` or `∝ information gain about latent variables`.

**Strengths.** Solves sparse-reward exploration; builds broad competence; drives open-ended learning.

**Weaknesses.** "Noisy TV problem" — pure novelty-seeking gets trapped by irreducible randomness; can distract from goals.

**Neuroscience evidence.** Dopaminergic responses to novelty and information (not just reward); hippocampal novelty signals; ventral striatum activation for information-seeking even when non-instrumental.

**Engineering takeaway.** LLM agents have **no intrinsic drive** — they do exactly what they're told and never explore to improve themselves. A **Curiosity Module** that scores gaps in the model's knowledge/world-model and proposes information-gathering actions (queries, experiments, tool calls) enables self-improving, open-ended agents (Part 4, 10). This is what turns a passive tool into an active learner.

## 1.21 Uncertainty Estimation

**Biological purpose.** Represent confidence to guide decisions, information-gathering, and communication of reliability.

**Computational interpretation.** The **Bayesian brain**: represent beliefs as probability distributions and track their precision (inverse variance). Distinguish *aleatoric* (irreducible, environmental) from *epistemic* (reducible by learning) uncertainty. Precision-weighting modulates how much new evidence updates beliefs (see Part 8, active inference).

**Strengths.** Enables optimal evidence integration, risk-sensitive decisions, and knowing when to gather more data.

**Weaknesses.** Humans are systematically miscalibrated (overconfidence, conservatism, probability distortion — Prospect Theory's weighting function).

**Neuroscience evidence.** Neural populations encoding uncertainty (Ma et al., probabilistic population codes); noradrenaline (locus coeruleus) signaling unexpected uncertainty; pupil dilation tracking uncertainty.

**Engineering takeaway.** LLMs produce token probabilities that are **poorly calibrated after RLHF** and do not decompose into epistemic vs. aleatoric. There is no representation of "I don't know because I never learned this" vs. "the answer is inherently random." A **Confidence Estimator** that (a) is trained/calibrated separately, (b) distinguishes uncertainty types, and (c) feeds the controller's route/defer/gather decisions is essential (Parts 4, 5, 10).

---

### Part 1 Synthesis: The Functional Blueprint

Stepping back, the brain's factoring gives us a target module list:

```
 CONTROL LAYER    : Metacognition, Executive control, Attention control, Goal management
 REASONING LAYER  : System 2 (planning, simulation, causal/counterfactual/analogical), Reflection
 FAST LAYER       : System 1 (pattern recognition, amortized recall)
 MEMORY LAYER     : Working / Episodic / Semantic / Procedural, with consolidation
 MODEL LAYER      : Generative world model (for simulation & prediction)
 LEARNING LAYER   : Feedback (RPE), Curiosity (intrinsic), Uncertainty (precision)
```

The rest of the report shows current LLMs implement the FAST layer superbly, a fraction of the REASONING layer in-context, almost none of the CONTROL layer, a degenerate MEMORY layer, no separable MODEL layer, and no inference-time LEARNING layer — then designs the missing pieces.

---

# PART 2 — CURRENT LLMs: WHAT THEY ARE AND WHERE THEY DIVERGE

## 2.1 The Frontier Families (as of early 2026)

| Family | Vendor | Notable architectural / training traits | Cognitive character |
|---|---|---|---|
| **GPT (o-series, GPT-4.x/5)** | OpenAI | Pioneered RLHF at scale; **o1/o3** introduced large-scale RL on private chain-of-thought → "reasoning models" that spend inference-time compute; heavy tool integration | Strongest explicit System-2 emulation via long internal CoT |
| **Claude (3.x, 4.x, Fable/Opus 5)** | Anthropic | Constitutional AI (RLAIF), extended "thinking" mode with visible reasoning budgets, strong long-context and agentic/tool-use tuning | Balanced; emphasis on reflective, steerable reasoning and tool-use reliability |
| **Gemini (1.5/2.x/3)** | Google DeepMind | Natively multimodal, very long context (1M+ tokens), MoE; DeepMind heritage of search/RL (AlphaGo lineage) | Broadest context "working memory"; multimodal world grounding |
| **DeepSeek (V3, R1)** | DeepSeek | **R1** demonstrated strong reasoning via **RL with verifiable rewards (RLVR)** and GRPO, with emergent long CoT; open weights; efficient MoE | Proof that System-2 reasoning can be RL-elicited cheaply and openly |
| **Qwen (2.5, 3)** | Alibaba | Strong multilingual, MoE variants, competitive open reasoning models, long context | Open, strong general + reasoning; broad tool/agent support |
| **Llama (3.x, 4)** | Meta AI | Open-weights workhorse; dense and (Llama 4) MoE; long context; ecosystem standard | The open baseline; strong System 1, standard CoT System 2 |
| **Mistral (Large, Mixtral, Magistral)** | Mistral AI | Early open MoE (Mixtral sparse), efficient dense models, reasoning variants | Efficiency-focused; sparse MoE pioneer in open source |

**The convergence.** By 2026 the families have converged on a common recipe: a **decoder-only transformer**, increasingly **Mixture-of-Experts** for parameter efficiency, pretrained on trillions of tokens, post-trained with **SFT → preference RL (RLHF/RLAIF) → verifiable-reward RL (RLVR)**, augmented with **tool use** and **agent scaffolds**, and offered in a **"reasoning" mode** that spends inference-time compute on long chains of thought. The differences are of degree, data, and safety philosophy — not of fundamental cognitive architecture. **They are all the same cognitive creature: a giant System 1 with an RL-sharpened, in-context System-2 emulation.**

## 2.2 Cognitive Abilities Already Approximated

- **Pattern recognition / semantic memory (§1.6, §1.5):** Superhuman breadth of stored knowledge and fluent retrieval. This is the core competency.
- **Analogical reasoning (§1.11):** Strong, emergent, though not deliberate/verifiable.
- **Language & (partial) mental models (§1.8):** Fluent, locally coherent; can build implicit situation models.
- **In-context / few-shot learning:** A genuine emergent ability (Brown et al., 2020) — *transient* WM-based learning without weight change. Approximates rapid task adaptation.
- **System-2 emulation (§1.1, §1.12, §1.13):** Reasoning models (o1, R1) produce long, self-correcting chains that genuinely improve on math/code/logic — a real, if fragile, deliberation capability.
- **Tool use (§1.7 external offloading):** Can call calculators, code, search — offloading computation like humans use notebooks.

## 2.3 Cognitive Abilities Completely (or Largely) Missing

1. **Persistent, one-shot episodic memory (§1.3, §1.4).** No hippocampus. Nothing learned in a session survives it, except via external RAG hacks.
2. **Online learning from feedback (§1.19).** Weights are frozen at inference. The model cannot get better at a task by doing it.
3. **True metacognition / calibrated confidence (§1.14, §1.21).** No reliable self-monitoring, no self-directed effort allocation, no knowing-when-to-stop.
4. **A separable world model / mental simulation (§1.7, §1.8).** No latent model to roll forward independently of text.
5. **Causal & counterfactual reasoning at the interventional level (§1.9, §1.10).** Associational competence masquerading as causal.
6. **Persistent goal maintenance & executive control (§1.15, §1.12).** No protected goal stack; loses the plot on long horizons.
7. **Intrinsic motivation / curiosity (§1.20).** No self-driven exploration or self-improvement.
8. **Procedural memory / skill consolidation.** No mechanism to turn a repeatedly-derived solution into a fast, cached skill.

## 2.4 Where Transformers *Fundamentally* Differ from Cognition

- **Frozen vs. plastic.** Brains learn continuously and locally (Hebbian, neuromodulated); transformers separate a giant offline training phase from a frozen inference phase. *This is the deepest divergence.*
- **Stateless vs. stateful.** A transformer forward pass is a pure function of its context window; the brain is a persistent dynamical system with internal state that outlives any input.
- **Parallel-in-depth vs. recurrent-in-time.** A transformer has a *fixed* number of sequential computation steps (its depth) per token — it cannot "think longer" on a hard token without an external scaffold. The brain recurs arbitrarily. (This is why CoT works: it converts *depth* limits into *sequence length*, buying serial compute.)
- **No global workspace / broadcast.** The brain has a limited-capacity "global workspace" (Baars/Dehaene) that broadcasts selected content to all modules; the transformer has no analog of conscious-access gating.
- **Uniform vs. specialized compute.** Every token gets the same compute (modulo MoE routing); the brain spends wildly different resources on different problems.
- **Optimization target.** Next-token likelihood + RL on preferences/rewards ≠ evolution's multi-objective survival optimization; the inductive biases differ accordingly.

## 2.5 Mechanism-by-Mechanism: The Engineering Substrate

**Transformer attention.** Content-based soft retrieval: each token computes queries against keys and reads a weighted sum of values. It is a *learned, differentiable, associative memory read* over the context — a superb but *undirected* attention (§1.16). Cost is O(n²) in sequence length.

**Context windows.** The de facto **working memory** (§1.2), now 100K–1M+ tokens. Superhuman capacity, but: passively retained (no active gating), suffers "lost in the middle" positional degradation, and is *volatile* (gone after the request). Not a substitute for persistent memory.

**KV cache.** The stored keys/values of past tokens — the *physical substrate* of that working memory during generation. Grows linearly with context; a major memory-bandwidth cost. Analogous to sustained delay-period firing maintaining WM contents.

**Hidden states.** The per-layer residual-stream activations — the model's transient "mental state." Interpretability shows they encode features, world-state fragments (e.g., board state in Othello-GPT — Li et al., 2023), and task vectors. But they are *not persisted* and *not directly addressable* as memory.

**Token prediction.** The core objective and the core limitation. Autoregressive next-token prediction is a phenomenally effective self-supervised objective that forces the model to learn a compressed model of everything predictive of text. But it optimizes *local* coherence; global consistency, truth, and goal-achievement are only indirectly incentivized — the root of hallucination and plan-drift.

**Scaling laws.** Kaplan et al. (2020) and Hoffmann et al. (2022, "Chinchilla") established power-law relationships between loss and compute/data/parameters, and the compute-optimal data/param ratio. These *predict capability from compute* — an engineering planning tool with no cognitive analog, and increasingly bottlenecked by data availability (motivating synthetic data and RL).

**Emergent abilities.** Wei et al. (2022) documented capabilities appearing sharply at scale (multi-step arithmetic, instruction following). Debated (Schaeffer et al., 2023, argue some emergence is a metric artifact), but in-context learning and CoT are real qualitative shifts. Cognitively reminiscent of developmental phase transitions, though mechanistically unrelated.

**Reasoning traces (chain-of-thought).** Wei et al. (2022) CoT; Kojima et al. (2022) zero-shot "let's think step by step." The key computational fact: **CoT converts the transformer's fixed serial depth into unbounded serial length**, letting it externalize intermediate state into the token stream (a scratchpad — Nye et al., 2021). This is the mechanism by which LLMs *simulate* System 2. Reasoning models (o1, R1) train this behavior with RL rather than prompting.

**Inference-time compute.** The 2024–2025 paradigm shift (Snell et al., 2024; OpenAI o1): spend more FLOPs *at test time* (longer CoT, sampling + verification, search) to improve accuracy — trading latency/cost for capability, and often beating parameter-scaling per dollar on reasoning tasks. This is the closest thing to *deliberate effort allocation* (§1.14), but currently crude and uncontrolled.

**Mixture of Experts (MoE).** Shazeer et al. (2017); Switch Transformer (Fedus et al., 2021); Mixtral, DeepSeek-V3. A router activates a sparse subset of expert FFNs per token → more parameters (knowledge) at fixed FLOPs. Loosely analogous to *functional specialization*, but experts are token-level and learned, not cognitive modules — a **conditional-compute** trick, not a cognitive architecture.

**RLHF.** Ouyang et al. (2022, InstructGPT): fit a reward model to human preferences, optimize the policy against it (PPO/DPO). Aligns outputs to human intent — analogous to social/normative feedback shaping behavior. Known pathologies: reward hacking, sycophancy, calibration degradation.

**RLVR (RL with Verifiable Rewards).** DeepSeek-R1 (2025), OpenAI o-series: replace the learned reward model with an *automatic verifier* (unit tests, math checkers, proof assistants). Because the reward is grounded and un-gameable, RL can push genuine reasoning skill and elicit long CoT. This is the most important 2025 advance and the clearest instance of **learning from a ground-truth external critic** (§1.13) — the engineering version of "reflection with a real answer key."

**Tool use.** Toolformer (Schick et al., 2023); function calling; code interpreters. Offloads what the model is bad at (arithmetic, retrieval, execution) to reliable external systems — exactly cognitive offloading (§1.7). Now standard and central to agents.

**Memory (external).** RAG (Lewis et al., 2020), vector stores, and agent "memory" frameworks. A *degenerate episodic/semantic memory* (§1.4, §1.5): retrieval by embedding similarity over text chunks, no consolidation, no separation, no importance weighting. Functional but far from the CLS blueprint.

**Agent loops.** ReAct (Yao et al., 2022) interleaves reasoning and acting; the observe→think→act→observe cycle. This is the closest structural analog to the perception-cognition-action loop, but it is *stateless across episodes*, has *no persistent goal stack or metacontroller*, and *does not learn* from its trajectories except via logged text.

**Diagnosis.** Current LLMs are a *magnificent recognition-and-recall engine* (System 1 + semantic memory) wrapped in *scaffolds that emulate the missing modules in-context*. The scaffolds work because the base model is smart enough to role-play a controller, a planner, a critic. But role-playing a controller in the same frozen weights that generate the content is fragile: no independent optimization, no persistent state, no true error signal. **The path forward is to promote the scaffolds to real, separable, trainable modules.**

---

# PART 3 — MAPPING HUMAN COGNITION TO LLM COMPONENTS

The master table. Columns: **Human mechanism → today's nearest LLM component → what's missing → candidate architecture to close the gap.**

| # | Human Cognitive Mechanism | Nearest Current LLM Component | Missing Capability | Candidate Architecture |
|---|---|---|---|---|
| 1 | **System 1** (fast, amortized) | Single forward pass | Confidence-gated escalation | Base model as fast path + router |
| 2 | **System 2** (slow, serial) | Chain-of-thought / reasoning-model RL | Separable, verifiable, trainable deliberation with search | Dedicated slow engine: CoT + tree search + verifier |
| 3 | **Working memory** | Context window + KV cache | Active gating, typed slots, variable binding, persistence-across-turn | Explicit writable typed scratchpad + attention controller |
| 4 | **Long-term memory** | Model weights (parametric) | Fast one-shot write; editability; provenance | Complementary Learning Systems: fast store + slow consolidation |
| 5 | **Episodic memory** | RAG over text chunks | Pattern separation, salience gating, event schemas, trace update | Structured episodic store: typed events, learned index, decay |
| 6 | **Semantic memory** | Weights + RAG | Editable, queryable, cited facts | Editable knowledge graph / memory layer + retrieval |
| 7 | **Procedural memory** | (essentially none) | Cached, fast-executable learned skills | Skill library (code/macros) + retrieval + skill distillation |
| 8 | **Pattern recognition** | Transformer forward pass | — (a strength) | Keep as fast path; add OOD/uncertainty flag |
| 9 | **Mental simulation** | Text-generated "imagining" | Separable latent world model with rollout | Learned world model (JEPA-style latent) + rollout engine |
| 10 | **Mental models** | Implicit distributed state in activations | Explicit, maintained, consistent structured belief state | Structured world-state object generation is conditioned on |
| 11 | **Causal reasoning** | Statistical/associational patterns | Interventional (do-calculus), structural models | Causal-graph induction module / neuro-symbolic hybrid |
| 12 | **Counterfactual reasoning** | Text "what-if" prompting | Grounded minimal-intervention re-simulation | World model + counterfactual rollout operator |
| 13 | **Analogical reasoning** | Emergent from embeddings | Deliberate, verifiable structure mapping | Relational representation + structure-mapping operator |
| 14 | **Planning** | In-context CoT / ToT scaffold | Model-based search, hierarchical, cached | Planner module + world model + MB/MF arbitration |
| 15 | **Reflection** | Reflexion / self-refine prompting | Reliable self-error detection; external grounding | Trainable critic + external verifiers (RLVR-style) |
| 16 | **Metacognition** | (none reliable) | Effort allocation, self-monitoring, stopping | **Executive Controller + Confidence Estimator** |
| 17 | **Goal-directed behavior** | Goal-in-prompt | Persistent protected goal representation | **Goal Manager** (goal/subgoal stack, re-injection) |
| 18 | **Attention (top-down)** | Content-based self-attention | Goal-driven gating; drop-irrelevant | **Attention Controller** biasing retrieval & context |
| 19 | **Chunking** | Tokenization; latent abstraction | Learned hierarchical macro-abstractions | Context compression + macro-actions/skills |
| 20 | **Hierarchical reasoning** | Prompted decomposition | Native temporal abstraction, manager/worker split | Planner-executor with size-tiered modules |
| 21 | **Learning from feedback** | RLHF/RLVR (train-time only) | Online, inference-time weight/memory update | **Learning loop**: episodic write → periodic consolidation |
| 22 | **Curiosity** | (none) | Intrinsic drive; info-gain-seeking exploration | **Curiosity Module** scoring epistemic gaps |
| 23 | **Uncertainty estimation** | Token logits (miscalibrated) | Epistemic vs aleatoric; calibration | **Confidence Estimator** (separate head, calibrated) |

**How to read this table.** The right two columns are the report's engineering agenda. Rows 16, 17, 18, 22, 23 (the control + drive + calibration layer) and rows 4–7, 9, 21 (memory + world model + online learning) are where essentially all the value is — they are almost entirely absent today, and they are the modules Parts 4–10 design.

---

# PART 4 — DESIGNING A HUMAN-LIKE LLM ("GPT-10")

We now specify a modular cognitive architecture. Design principle: **separable modules with typed interfaces, individually trainable, orchestrated by a metacognitive controller.** Not one monolith — an *organization* of specialized networks sharing a common representational substrate (a shared embedding space so modules can exchange latent state, not just text).

## 4.1 Module Catalog (inputs → outputs → activation trigger)

**Executive Controller (metacognitive kernel).**
- *Inputs:* current goal state, working-memory summary, confidence estimate, budget (latency/cost), task features.
- *Outputs:* control actions — {answer now, invoke slow engine, retrieve memory, call tool, plan, simulate, reflect, defer/ask}.
- *Activation:* every cycle. It is the loop's scheduler.
- *Implementation:* a small policy network trained with RL on end-task reward + cost penalty (learns *when thinking pays*). The embodiment of §1.14.

**Fast Thinking Engine (System 1).**
- *Inputs:* prompt + working memory + retrieved context.
- *Outputs:* a candidate answer *and* a self-assessed confidence.
- *Activation:* always first (default-interventionist: try the cheap path).
- *Implementation:* the base LLM forward pass.

**Slow Thinking Engine (System 2).**
- *Inputs:* problem + working memory + world-model access + tools.
- *Outputs:* a reasoned solution with a trace and verification result.
- *Activation:* when the controller escalates (low confidence, high stakes, detected novelty/conflict).
- *Implementation:* reasoning-model CoT + optional tree/graph search + verifier calls.

**Working Memory.**
- *Inputs:* writes from any module (intermediate results, subgoals, retrieved facts).
- *Outputs:* current active state to all modules.
- *Activation:* continuous; gated by the Attention Controller.
- *Implementation:* a structured, typed, size-bounded buffer (not just raw tokens) — slots with roles, plus a compressed summary.

**Long-Term Memory (with Retrieval).** See Part 7. Episodic + semantic + procedural stores; write on salient events, read on cue.

**Planning Module.**
- *Inputs:* goal, world-model, current state, skill library.
- *Outputs:* a (hierarchical) plan — sequence/tree of subgoals and actions.
- *Activation:* multi-step goals where the Fast Engine's confidence in a direct answer is low.
- *Implementation:* model-based search over the world model; caches successful plans as procedural skills.

**Simulation Module + World Model.** See Part 8.
- *Inputs:* state + candidate action(s).
- *Outputs:* predicted next state(s) and outcomes; used by planner, critic, counterfactuals.
- *Activation:* planning, reflection, counterfactual queries.

**Reflection Module (Critic).**
- *Inputs:* a candidate output + goal/constraints + (optionally) tool/verifier results.
- *Outputs:* error assessment + revision suggestions or accept signal.
- *Activation:* after any candidate answer for medium/high-stakes tasks.
- *Implementation:* a critic model (possibly the same base model in a critic role, but ideally *separately trained* on error-detection) + external verifiers.

**Goal Manager.**
- *Inputs:* user request, controller updates, completion signals.
- *Outputs:* the active goal/subgoal stack; re-injects the top goal into WM each cycle.
- *Activation:* continuous; the antidote to plan-drift (§1.15).

**Tool Planner.**
- *Inputs:* current subgoal + tool registry + confidence.
- *Outputs:* tool selection + argument construction + result integration.
- *Activation:* when a subgoal is better served by an external system than by the model.

**Memory Retrieval Controller.** Learned query formulation over the memory stores (what to fetch, how much), gated by the Attention Controller — avoids context dilution.

**Curiosity Module.**
- *Inputs:* world-model uncertainty, knowledge gaps, unexplored branches.
- *Outputs:* intrinsic-reward signals / proposed information-gathering actions.
- *Activation:* during learning/exploration or when the task reveals a knowledge gap.

**Attention Controller.** Top-down biasing of what enters/stays in WM and what memory retrieval returns; implements §1.16.

**Self-Evaluation + Confidence Estimator.** A calibrated head (trained with proper scoring rules and post-hoc calibration) producing epistemic/aleatoric-decomposed confidence that drives the controller's routing and the decision to answer/defer/ask.

## 4.2 Internal Data Flow

```
                         ┌───────────────────────────────────────────────┐
        user input  ───▶ │              GOAL MANAGER (goal stack)         │
                         └───────────────┬───────────────────────────────┘
                                         │ active goal
                                         ▼
             ┌───────────────── EXECUTIVE CONTROLLER (metacognition) ─────────────────┐
             │  reads: goal, WM summary, confidence, budget                            │
             │  decides route each cycle                                               │
             └───┬───────────┬───────────┬───────────┬───────────┬──────────┬─────────┘
                 │           │           │           │           │          │
                 ▼           ▼           ▼           ▼           ▼          ▼
           FAST ENGINE   MEMORY      SLOW ENGINE   PLANNER    SIM + WORLD  TOOL
           (System 1)    RETRIEVAL   (System 2)    (HRL)      MODEL        PLANNER
                 │           │           │           │           │          │
                 └─────┬─────┴─────┬─────┴─────┬─────┴─────┬─────┴────┬─────┘
                       ▼           ▼           ▼           ▼          ▼
                 ┌──────────────── WORKING MEMORY (typed, gated) ──────────────┐
                 │        ▲ Attention Controller gates writes/reads            │
                 └──────────────────────────┬──────────────────────────────────┘
                                            ▼
                                REFLECTION / CRITIC + CONFIDENCE
                                            │ accept?  yes│no
                                            ▼             └──▶ back to Controller (revise/escalate)
                                    OUTPUT + write salient trace ──▶ EPISODIC MEMORY
                                            │
                                     (offline) CONSOLIDATION ──▶ SEMANTIC / PROCEDURAL / weights
```

## 4.3 When Each Module Activates (the default-interventionist policy)

1. Goal Manager frames the goal.
2. Controller tries the **Fast Engine** (cheap). Gets answer + confidence.
3. If confidence high and stakes low → **Reflection-lite** (quick sanity check) → output.
4. If confidence low / stakes high / novelty flagged → escalate:
   - Retrieve relevant **memory**.
   - If multi-step → **Planner** (using **World Model** to simulate options).
   - If a subgoal needs external truth/computation → **Tool Planner**.
   - Run **Slow Engine** on hard subproblems, with **search** where the branching factor warrants.
5. **Critic** verifies each candidate (ideally against an external verifier). If rejected → revise or escalate compute.
6. **Confidence Estimator** gates the final output: answer, or defer/ask if still uncertain.
7. Salient trajectories written to **Episodic Memory**; **Curiosity** logs knowledge gaps for later exploration.
8. Offline: **consolidation** distills repeated episodes into semantic facts and procedural skills (and, periodically, into weight updates).

The design's soul is step 2–4: **cheap by default, expensive only when metacognition says it will pay.** This is the brain's economy and the key to both quality and cost control.

---

# PART 5 — DUAL-SYSTEM LLM: A CONCRETE ROUTING ARCHITECTURE

Zooming into the System-1/System-2 core, because it is the most immediately buildable and highest-leverage piece.

## 5.1 Responsibilities

**System 1 (Fast).** Answer instantly; retrieve patterns; "autocomplete" reasoning; recognize common/known situations; produce a candidate + confidence. One forward pass. Handles the majority of queries (factual recall, simple transforms, familiar formats).

**System 2 (Slow).** Verify; plan; decompose; call tools; search memory; run tree/graph search; run world-model simulations; critique; iterate. Variable, potentially large inference-time compute. Handles novel, multi-step, high-stakes, or low-confidence problems.

## 5.2 The Router / Controller

```
             ┌──────────────────────────────────────────────┐
  query ───▶ │            FAST SYSTEM (1 forward pass)        │──▶ answer_1, conf_1, features
             └──────────────────────┬───────────────────────┘
                                    ▼
                    ┌──────────────────────────────┐
                    │        ROUTER / CONTROLLER     │
                    │  inputs: conf_1, task features,│
                    │  stakes, budget, novelty flag  │
                    │  policy: escalate?  how much?  │
                    └───────┬───────────────┬────────┘
              conf high     │               │  conf low / high stakes / novel
              stakes low     ▼               ▼
                       return answer_1   SYSTEM 2  (adaptive compute)
                                          ├─ decompose / plan
                                          ├─ retrieve memory
                                          ├─ tool calls
                                          ├─ tree/graph search + verifier
                                          ├─ world-model simulation
                                          └─ critic loop  ─┐
                                                           ▼
                                                  verified answer_2 + calibrated conf
```

## 5.3 Trigger Conditions (when to escalate to System 2)

Escalate when *any* of:
- **Low fast-path confidence** (calibrated, below a task-dependent threshold).
- **High stakes** (irreversible action, safety-relevant, user-flagged importance).
- **Detected novelty / distribution shift** (embedding far from training density; high hidden-state entropy).
- **Detected conflict** (multiple high-probability, mutually inconsistent completions — the ACC-conflict analog; measurable as multi-sample disagreement / self-consistency spread).
- **Verifier available** (if a cheap external check exists, cheap to verify → often worth escalating).
- **Explicit multi-step structure** (task requires a plan the fast path can't hold).

## 5.4 Controller Logic (pseudocode)

```python
def respond(query, budget):
    fast = system1(query)                     # answer, conf, hidden_features
    stakes = estimate_stakes(query)
    novelty = ood_score(fast.features)
    conflict = self_consistency_spread(query, k=3)   # cheap: 3 samples

    escalation = controller_policy(            # learned; RL-trained
        conf=fast.conf, stakes=stakes,
        novelty=novelty, conflict=conflict,
        budget=budget)

    if escalation.level == NONE:
        return fast.answer

    # Adaptive System-2 compute proportional to difficulty & budget
    ctx = memory.retrieve(query, k=escalation.retrieval_k)
    plan = planner.decompose(query, ctx) if escalation.plan else None
    sol  = system2(query, ctx, plan,
                   search=escalation.search_width,
                   tools=escalation.tools,
                   sim=escalation.simulate)
    verdict = critic.verify(sol, verifier=best_available_verifier(query))

    while not verdict.ok and budget.remaining():
        sol = system2.revise(sol, verdict.feedback)
        verdict = critic.verify(sol, ...)

    conf = confidence_estimator(sol, verdict)
    return sol if conf >= threshold(stakes) else defer_or_ask(sol, conf)
```

## 5.5 Confidence Estimation

Three complementary signals, combined and calibrated:
- **Intrinsic:** a dedicated confidence head trained with proper scoring rules (not raw softmax, which RLHF miscalibrates), ideally decomposing epistemic vs. aleatoric (e.g., via ensembling / MC-dropout-style perturbation / deep-ensemble surrogate).
- **Consistency:** self-consistency (Wang et al., 2022) — sample k reasoning paths; agreement ↑ confidence. Cheap and effective.
- **Verifier:** if an external checker passes, confidence ≈ 1 for that subclaim (RLVR's grounding).
Post-hoc **temperature/Platt calibration** on held-out data aligns stated confidence with empirical accuracy. The controller consumes calibrated confidence, not raw logits.

## 5.6 Latency & Cost Tradeoffs

| Route | Relative latency | Relative cost | Accuracy on hard tasks | Fraction of queries (typical) |
|---|---|---|---|---|
| System 1 only | 1× | 1× | baseline | ~70–85% |
| S1 + light verify | ~1.5× | ~1.5× | + | ~10–20% |
| System 2 (CoT + tools) | 5–20× | 5–20× | ++ | ~5–15% |
| System 2 + search/sim | 20–200× | 20–200× | +++ | ~1–5% |

The economic argument mirrors the metabolic one: **if you can route 80% of traffic to the 1× path with calibrated confidence, you pay the 20–200× cost only where it changes the answer.** Miscalibration is the enemy — over-escalation wastes money, under-escalation ships errors. Hence the confidence estimator is the highest-value component to get right. Empirically, adaptive routing (spend compute where it helps) dominates uniform "always think hard" on cost-adjusted accuracy (Snell et al., 2024 on compute-optimal test-time scaling).

---

# PART 6 — AGENTIC REASONING: DO HUMANS REASON LIKE ReAct AGENTS?

## 6.1 The honest answer

**Partially, and mostly no.** Humans *do* interleave thinking and acting (ReAct's core loop is genuinely cognitively plausible — the perception-action cycle). But most named agentic techniques are **engineering approximations that compensate for what a single frozen forward pass can't do** — they are *external* implementations of computations the brain does *internally and continuously*. The brain does not literally spawn five copies of itself to debate, nor expand an explicit search tree with backtracking in the way ToT does. It approximates these with a *single, recurrent, memory-equipped dynamical system* that has *internalized* good heuristics.

## 6.2 Technique-by-technique plausibility

| Technique | What it does | Biologically plausible? | Verdict |
|---|---|---|---|
| **ReAct** (Yao 2022) | Interleave reason + act + observe | **Yes** — mirrors perception-action cycle | Plausible core loop; humans do this |
| **Chain-of-Thought** | Serialize reasoning steps | **Partly** — matches inner speech / WM rehearsal, but humans don't verbalize *all* reasoning | Plausible as externalized WM; over-verbalized |
| **Self-consistency** (Wang 2022) | Sample many paths, majority-vote | **Weakly** — humans do consider a few alternatives, but don't run 40 independent solves | Engineering approximation of ensemble; humans use 1–3 |
| **Tree of Thoughts** (Yao 2023) | Explicit branching search + backtrack + value | **Partly** — humans do limited lookahead & backtrack, but *heavily pruned* (2–3 branches, not breadth-first) | Engineering; brain does aggressive model-based pruning |
| **Reflexion** (Shinn 2023) | Verbal self-feedback → retry | **Yes** — matches error-monitoring (ERN) + reattempt | Plausible; the critic is real, its reliability isn't |
| **Debate** (Du 2023; Irving 2018) | Multiple agents argue to truth | **Weakly** — social deliberation exists between people, but not inside one mind | Engineering; leverages diversity, not one brain |
| **Multi-agent systems** | Role-specialized agents cooperate | **At the group level, yes** (teams, division of labor); not within one cognition | Organizational analogy, not individual cognition |
| **Graph of Thoughts** (Besta 2024) | Reasoning as a DAG with merges | **Weakly** — associative memory is graph-like, but explicit GoT is engineering | Engineering approximation of associative reasoning |
| **MCTS** (Silver 2016/17) | Simulate rollouts, UCB, backup values | **Loosely** — hippocampal rollouts + value backup rhyme with it, but no explicit UCB bookkeeping | Engineering; brain's version is amortized/approximate |
| **Search-based reasoning** | Systematic state-space search | **No** — humans are terrible at exhaustive search; we satisfice | Pure engineering strength (a *complement* to human weakness) |
| **Program synthesis** | Emit executable programs | **Analogous** to procedural skill formation / tool-building | Engineering, but rhymes with skill compilation |
| **Planner-executor** | Abstract plan → concrete execution | **Yes** — PFC rostro-caudal hierarchy | Highly plausible; matches HRL in cortex |
| **Hierarchical planning** | Nested goals/subgoals | **Yes** — core to human goal pursuit | Highly plausible |

## 6.3 The key distinction: internalized vs. externalized computation

The brain **amortizes**: through experience it compiles good search/simulation/verification into fast intuitions (System 1), and runs explicit search only rarely and shallowly (System 2, heavily pruned). Agent frameworks **externalize** these into explicit, un-amortized loops because the frozen model *can't* amortize at inference time and *can't* prune well without a good learned value function.

**Engineering implication.** The most human-like — and likely most efficient — agent is not the one that runs the biggest explicit tree search. It is the one that (a) has a strong amortized policy/value (so search is shallow and well-pruned), (b) searches explicitly *only* when the value function is uncertain (MB/MF arbitration, §1.12), and (c) *learns from each episode* so today's expensive search becomes tomorrow's cheap intuition (the AlphaZero loop: search generates training targets that distill back into the fast policy). This "search → distill → faster policy → shallower search" flywheel is the deepest lesson agentic AI can take from cognition. Program synthesis and skill libraries (Voyager — Wang et al., 2023) are the LLM instantiation.

---

# PART 7 — MEMORY ARCHITECTURE

## 7.1 The five stores and their engineering specs

| Store | Timescale | Content | Write policy | Read policy | LLM design |
|---|---|---|---|---|---|
| **Working** | seconds / current task | active state, subgoals, intermediate results | every cycle, gated | direct, full | Typed bounded buffer + compressed summary; attention-gated |
| **Short-term** | minutes / session | recent turns, transient facts | on each turn | recency + relevance | Rolling context + running summary (chunk-compressed) |
| **Episodic** | events, persistent | (event, context, time, outcome, salience) | on salient events (salience-gated) | cue-based (learned index) | Structured event store + learned retriever; pattern-separated keys |
| **Semantic** | facts, durable | decontextualized facts, concepts, relations | via consolidation | query-based, provenance-tagged | Editable KB / memory layer + parametric prior |
| **Procedural** | skills, durable | executable skills / macros / policies | on repeated success | by task match | Skill/code library + retriever + distillation into policy |

## 7.2 Storage, retrieval, indexing

- **Storage.** Do *not* dump raw text. Store **typed records**: episodic events as `{summary, entities, time, location, action, outcome, salience, embedding}`; semantic facts as `{subject, relation, object, source, confidence, timestamp}`; skills as `{signature, code/plan, preconditions, success_stats}`.
- **Retrieval.** Learned, not just cosine similarity. Train a retriever end-to-end on downstream task success (as in RETRO / REALM / retrieval-augmented training) so it fetches *useful*, not merely *lexically similar*, memories. Combine dense + sparse + graph traversal (follow relation edges).
- **Indexing.** Hierarchical: coarse embedding ANN for candidate generation, then a re-ranker; plus a symbolic index over entities/relations for exact lookup and multi-hop.

## 7.3 Pattern separation (the anti-collision requirement)

Naive vector stores let near-duplicate memories collide and drown retrieval. Borrow the **dentate gyrus** trick: **orthogonalize keys** for similar-but-distinct episodes (sparse, high-dimensional, decorrelated keys) so pattern completion at read time reconstructs the *right* episode. Practically: learn separation-promoting key transforms; dedupe/merge true duplicates during consolidation; keep distinct-but-similar episodes distinguishable by context features (time, outcome).

## 7.4 Forgetting, decay, compression

- **Forgetting is a feature, not a bug.** Unbounded memory degrades retrieval (more collisions, more distractors). Implement **decay**: salience- and recency-weighted eviction (Generative Agents' retrieval score = α·recency + β·relevance + γ·importance — Park et al., 2023, is a workable prototype).
- **Compression.** Consolidate many episodes into semantic gist (summaries, statistics) — the neocortical averaging. Compress WM/short-term into dense "gist tokens" (learned soft-prompt summarization) to extend effective context cheaply.
- **Importance scoring.** A learned salience signal (surprise / reward-relevance / emotional-tag analog / user-flagged) gates what gets a durable episodic write — mirroring how the amygdala/neuromodulators tag emotional events for privileged consolidation.

## 7.5 Memory consolidation & sleep-inspired offline learning

The **Complementary Learning Systems** blueprint (McClelland et al., 1995), made into an engineering loop:

```
   ONLINE (inference)                      OFFLINE ("sleep")
   ────────────────                        ─────────────────
   fast episodic writes  ──────────▶  1. REPLAY salient episodes (prioritized, like PER)
   (hippocampus analog)               2. EXTRACT regularities → semantic facts (gist)
                                       3. DISTILL repeated successful plans → procedural skills
                                       4. (periodically) FINE-TUNE / LoRA-update weights
                                          on replayed + synthetic data → neocortex analog
                                       5. PRUNE / merge / decay episodic store
```

**Why offline.** Interleaved replay during "sleep" avoids catastrophic forgetting (the reason brains consolidate offline rather than overwriting online): mixing new episodes with replayed old ones keeps the slow store stable while integrating the new. This directly attacks the frozen-weights gap (§1.19, §2.3): the model *does* learn from experience — not instantly in-weight (dangerous, unstable), but through a **safe, replay-based consolidation cycle**. Prioritized replay (Schaul et al., 2016), generative replay (Shin et al., 2017), and dual-memory continual-learning methods are the concrete techniques. This is the single most important addition for turning a static model into a lifelong learner.

---

# PART 8 — WORLD MODELS

## 8.1 Do humans reason *because* they have a world model?

**Yes — this is arguably the deepest thesis in cognitive science.** The dominant contemporary framework holds that the brain is fundamentally a **prediction machine** that reasons by *simulating* a generative model of the world.

- **Predictive processing (Clark, 2013; Rao & Ballard, 1999; Friston).** The cortex is a hierarchical generative model that constantly predicts incoming signals; only **prediction errors** propagate upward, and perception is the brain's *best hypothesis* (controlled hallucination), not a readout of the senses. Reasoning = running this model forward/backward.
- **Bayesian brain hypothesis.** Perception and cognition as approximate Bayesian inference: priors + likelihood → posterior beliefs, with **precision-weighting** (confidence) controlling update size (§1.21). Explains illusions, cue integration, and top-down effects.
- **Active inference / Free Energy Principle (Friston, 2010).** A unifying claim: organisms act to minimize *variational free energy* (a bound on surprise). Action and perception both serve to make sensory input match predictions — you either update your model (perception) or change the world to fit it (action). Planning becomes inference over policies that minimize expected free energy, elegantly unifying *exploitation* (reach preferred states) and *exploration* (**information gain** — the formal basis of curiosity, §1.20).
- **Mental simulation / internal physics engine.** Humans predict physical outcomes (will the tower fall?) with an **intuitive physics** that behaves like a noisy Newtonian simulator (Battaglia, Hamrick & Tenenbaum, 2013 — "physics engine in the head"). We plan by running this engine.

So: humans reason by *querying and rolling out an internal generative world model*, weighting everything by precision/uncertainty, and acting to minimize predicted surprise. Reasoning is **model-based simulation**, not rule application.

## 8.2 Do LLMs have world models?

**Weakly and implicitly.** Evidence they build *some* internal world state: Othello-GPT learns a board representation (Li et al., 2023); models track entities and spatial relations in activations; probes recover latent structure. But this world model is (a) *entangled with language*, (b) *not separately queryable or rollable-forward*, (c) *inconsistent over long generations*, and (d) *not grounded/precise*. LLMs are trained to predict *tokens*, not *world states* — so their "world model" is whatever is incidentally useful for token prediction, which is far short of a controllable simulator.

## 8.3 How LLMs could integrate explicit world models

Three (complementary) routes:

1. **Latent predictive world models (LeCun's JEPA, 2022; Hafner's Dreamer/DreamerV3).** Learn a model that predicts *future latent representations* (not pixels/tokens) from state+action: `ẑ_{t+1} = f(z_t, a_t)`. Train by self-supervised prediction in representation space (JEPA avoids the wasteful pixel-reconstruction of generative world models). Give the agent a *rollable* simulator for planning (Dreamer plans by imagining latent rollouts). This is the cleanest path to a separable Simulation Module (Part 4/10).

2. **Neuro-symbolic / explicit state.** Maintain an explicit structured world state (scene graph, entity-relation store, physics/simulator, or code) that the LLM reads from and writes to, and *conditions generation on*. Grounds consistency and enables true `do`-interventions and counterfactuals (§1.9, §1.10). Program-of-Thought / tool-based simulators (call a real physics engine, a database, a code sandbox) are pragmatic versions available today.

3. **Predictive-processing training objectives.** Move beyond pure next-token: add objectives that predict *future latent state* and *consequences of actions*, and expose the model to *interventional* data (RL in environments, embodied/simulated experience) so it learns the `P(s'|s, do(a))` structure token prediction can't teach. Active-inference-style objectives (minimize expected free energy) could unify goal-seeking and curiosity in one principle.

**Payoff.** A separable world model gives, in one stroke: mental simulation (§1.7), coherent mental models (§1.8), causal/counterfactual reasoning (§1.9–1.10), model-based planning (§1.12), and a principled curiosity signal (§1.20, via information gain). It is the highest-leverage single addition after memory.

---

# PART 9 — RESEARCH FRONTIER SURVEY

Organized by theme. For each landmark: **problem / method / innovation / limitation / future direction.** (Citations in References.)

## 9.1 Foundations

**Transformers — Vaswani et al., 2017, "Attention Is All You Need" (NeurIPS).**
*Problem:* sequence modeling bottlenecked by recurrence. *Method:* pure self-attention encoder-decoder. *Innovation:* fully parallelizable, content-based long-range dependency modeling. *Limitation:* O(n²), fixed depth (no adaptive serial compute), no persistent memory. *Future:* efficient/linear attention, recurrence, memory augmentation.

**Scaling laws — Kaplan et al., 2020; Hoffmann et al., 2022 (Chinchilla).**
*Problem:* how to allocate compute? *Method:* empirical power-law fits. *Innovation:* predictive planning; compute-optimal token/param ratio (~20 tokens/param). *Limitation:* data-bounded; says nothing about *which* capabilities emerge. *Future:* data-constrained scaling, synthetic data, RL scaling laws.

**Emergent abilities — Wei et al., 2022 (TMLR); rebuttal Schaeffer et al., 2023 (NeurIPS).**
*Problem:* do new capabilities appear with scale? *Method:* benchmark sweeps. *Innovation:* documents phase-transition-like jumps. *Limitation:* partly metric artifact; unpredictable. *Future:* mechanistic accounts of emergence.

## 9.2 Reasoning & Test-Time Compute

**Chain-of-Thought — Wei et al., 2022 (NeurIPS); zero-shot Kojima et al., 2022.**
*Problem:* multi-step reasoning fails in one pass. *Method:* prompt intermediate steps. *Innovation:* converts fixed depth → variable serial length (externalized WM). *Limitation:* unreliable, unverified, can rationalize. *Future:* trained (not prompted) reasoning; faithful CoT.

**Self-Consistency — Wang et al., 2022 (ICLR).** Sample many CoTs, majority-vote → big accuracy gains. *Limitation:* k× cost, only for extractable answers. *Future:* adaptive sampling, verifier-weighted voting.

**Tree of Thoughts — Yao et al., 2023 (NeurIPS); Graph of Thoughts — Besta et al., 2024 (AAAI).** Explicit branching/DAG search with value heuristics + backtracking. *Innovation:* deliberate lookahead. *Limitation:* expensive, needs a good value function, brittle. *Future:* learned value functions, amortized search.

**o1 / reasoning models — OpenAI, 2024; DeepSeek-R1 — DeepSeek-AI, 2025.**
*Problem:* elicit robust reasoning, not prompt-hacked CoT. *Method:* large-scale **RL on reasoning traces with verifiable rewards (RLVR)**; R1 uses GRPO and shows *emergent* long CoT and self-verification from pure RL (R1-Zero). *Innovation:* reasoning as a *trained* behavior; open reproduction (R1). *Limitation:* long, sometimes redundant traces; reward hacking; domain-limited to verifiable tasks; overthinking. *Future:* efficient reasoning (adaptive length), generalizing beyond verifiable domains, process rewards.

**Test-time compute scaling — Snell et al., 2024; Wu et al., 2024.**
*Problem:* is inference compute worth more than train compute? *Method:* compare sampling/search/verification budgets. *Innovation:* compute-optimal *test-time* strategies can beat 14× larger models on hard tasks. *Limitation:* diminishing returns; needs verifiers/PRMs. *Future:* controllers that allocate test-time compute optimally (Part 5).

**Process reward models — Lightman et al., 2023 ("Let's Verify Step by Step", OpenAI).** Reward *each reasoning step*, not just the answer → better verification and search guidance. *Future:* automatic process supervision (avoid human step-labels).

## 9.3 Cognitive Architectures & Memory

**Neural Turing Machines / Differentiable Neural Computer — Graves et al., 2014/2016 (Nature).** Differentiable external read/write memory. *Innovation:* separates computation from addressable memory. *Limitation:* hard to train, didn't scale. *Future:* memory-augmented transformers done right.

**RAG — Lewis et al., 2020 (NeurIPS); REALM — Guu et al., 2020; RETRO — Borgeaud et al., 2022.** Retrieval-augmented generation; retrieval baked into pretraining (REALM/RETRO). *Innovation:* non-parametric, updatable knowledge. *Limitation:* retrieval quality bottleneck; no consolidation. *Future:* learned lifelong memory, consolidation.

**Generative Agents — Park et al., 2023 (UIST).** Believable agents with memory stream + retrieval (recency/importance/relevance) + reflection + planning. *Innovation:* a working prototype of episodic memory + reflection + planning in LLM agents. *Limitation:* text-only memory, hand-tuned scoring, no true learning. *Future:* structured/consolidated memory, learned salience.

**MemGPT — Packer et al., 2023.** OS-inspired virtual-memory paging between context and external store. *Innovation:* treats context as RAM, store as disk. *Future:* learned paging policies.

**Cognitive-architecture revivals — ACT-R / SOAR lineage; "LLM + cognitive architecture" (e.g., LIDA, CoALA — Sumers et al., 2023).** CoALA ("Cognitive Architectures for Language Agents") frames agents with memory + action space + decision procedure — a useful design taxonomy. *Future:* unifying classical cognitive architectures with LLM substrates.

## 9.4 Planning, RL, World Models

**AlphaGo / AlphaZero / MuZero — Silver et al., 2016/2017/2020 (Nature/Science).** MCTS + learned policy/value; MuZero learns a *latent model* and plans in it without knowing the rules. *Innovation:* search + learning flywheel; latent world-model planning. *Limitation:* narrow domains, huge compute. *Future:* apply MuZero-style latent planning to language/agents.

**Dreamer / DreamerV3 — Hafner et al., 2020/2023.** Learn a latent world model; plan by imagined rollouts; one config solves many domains (incl. Minecraft diamonds). *Innovation:* sample-efficient model-based RL. *Future:* language-grounded latent world models.

**JEPA / autonomous-machine-intelligence — LeCun, 2022 (position); I-JEPA/V-JEPA.** Predict in *representation* space; energy-based, hierarchical, model-based agents. *Innovation:* a principled non-generative world-model objective. *Limitation:* still maturing; not yet at LLM-scale reasoning. *Future:* JEPA-style world models feeding LLM planners.

**Voyager — Wang et al., 2023.** LLM agent in Minecraft that writes/stores/reuses **code skills** (procedural memory) with curriculum + curiosity. *Innovation:* lifelong skill library + self-driven curriculum in an LLM agent. *Future:* general skill libraries, cross-domain transfer.

**ReAct — Yao et al., 2022 (ICLR); Reflexion — Shinn et al., 2023 (NeurIPS); Toolformer — Schick et al., 2023.** The agentic-loop, self-reflection, and self-supervised tool-use primitives. *Limitation:* stateless across episodes, no persistent learning. *Future:* learning agents (Part 10).

## 9.5 Neuro-symbolic & Causal

**Neuro-symbolic — Mao et al. (Neuro-Symbolic Concept Learner, 2019); AlphaGeometry — Trinh et al., 2024 (Nature).** Combine neural intuition with symbolic solvers/verifiers; AlphaGeometry hits IMO-medalist geometry via neural proposer + symbolic deduction engine. *Innovation:* neural creativity + symbolic rigor/verification. *Limitation:* domain-specific glue. *Future:* general neuro-symbolic reasoning with LLMs as the neural proposer.

**Causality — Pearl (do-calculus; "The Book of Why", 2018); Schölkopf et al., 2021 ("Toward Causal Representation Learning").** *Problem:* ML learns correlation. *Method:* SCMs, causal representation learning. *Innovation:* the interventional/counterfactual ladder as a target for ML. *Limitation:* learning causal structure from observation is hard. *Future:* causal-representation objectives in foundation models.

## 9.6 Cognitive Science Bridges

**"Building Machines That Learn and Think Like People" — Lake, Ullman, Tenenbaum & Gershman, 2017 (BBS).** The manifesto: human-like AI needs intuitive physics, intuitive psychology, compositionality, causality, and learning-to-learn. Still the sharpest statement of what's missing. *Future:* the entire agenda of this report.

**Dual-process & metacognition for AI — Bengio's "System 2 deep learning" program; consciousness-priors; Global Workspace Theory for AI (Goyal & Bengio, 2022; VanRullen & Kanai, 2021).** Proposes attention-based bottlenecks / global workspaces as inductive biases for compositional, controllable reasoning. *Future:* workspace architectures as the controller substrate.

**MoE at scale — Shazeer et al., 2017; Switch — Fedus et al., 2021; open MoE (Mixtral 2024, DeepSeek-V3 2024).** Conditional compute for parameter efficiency. *Future:* *cognitive* modularity (not just token-level experts).

---

# PART 10 — CANTOR: A NEXT-GENERATION COGNITIVE ARCHITECTURE

**CANTOR** — *Cognitive Architecture for Neural Thought, Orchestration, and Reasoning.* A synthesis of Parts 1–9 into one implementable design.

**Three design commitments:**
1. **Modular, not monolithic.** Separable, individually-trainable modules over a *shared latent workspace* (a global-workspace bottleneck through which modules exchange latent state, not just text).
2. **Metacognition as the kernel.** A learned Executive Controller allocates compute (System 1 vs 2), routes to modules, and decides confidence/stopping — the brain's economy of effort.
3. **A closed learning loop.** Episodic writes + offline (CLS/sleep) consolidation turn experience into semantic knowledge, procedural skills, and periodic weight updates — lifelong learning without catastrophic forgetting.

## 10.1 Overall Architecture (module diagram)

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                              CANTOR COGNITIVE STACK                                 │
│                                                                                    │
│  ┌──────────────┐        ┌────────────────────────────────────────────┐           │
│  │ GOAL MANAGER │◀──────▶│        EXECUTIVE CONTROLLER (metacog)        │           │
│  │ goal/subgoal │        │  policy π(action | state,conf,budget)        │           │
│  │    stack     │        │  routes • allocates compute • stops          │           │
│  └──────┬───────┘        └───┬───────┬────────┬────────┬────────┬───────┘           │
│         │                    │       │        │        │        │                   │
│         ▼                    ▼       ▼        ▼        ▼        ▼                   │
│  ┌───────────────────────────────────────────────────────────────────────────┐   │
│  │              GLOBAL WORKSPACE  (shared latent WM; attention-gated)          │   │
│  │   typed slots · running gist summary · active beliefs · Attention Ctrl      │   │
│  └───┬─────────┬──────────┬──────────┬──────────┬───────────┬──────────┬──────┘   │
│      │         │          │          │          │           │          │           │
│  ┌───▼───┐ ┌───▼────┐ ┌───▼────┐ ┌───▼────┐ ┌───▼─────┐ ┌───▼────┐ ┌───▼─────┐    │
│  │ FAST  │ │  SLOW  │ │PLANNER │ │WORLD   │ │REFLECT/ │ │ TOOL   │ │CURIOSITY│    │
│  │ ENGINE│ │ ENGINE │ │ (HRL)  │ │MODEL + │ │ CRITIC  │ │PLANNER │ │ (intr.  │    │
│  │  (S1) │ │  (S2)  │ │        │ │ SIM    │ │         │ │        │ │ reward) │    │
│  └───────┘ └────────┘ └────────┘ └────────┘ └─────────┘ └────────┘ └─────────┘    │
│      │         │          │          │          │           │          │           │
│  ┌───▼─────────▼──────────▼──────────▼──────────▼───────────▼──────────▼──────┐   │
│  │                        CONFIDENCE ESTIMATOR (calibrated, epi/ale)           │   │
│  └────────────────────────────────┬───────────────────────────────────────────┘   │
│                                    ▼                                                │
│  ┌───────────────────────────── MEMORY SYSTEM ──────────────────────────────┐     │
│  │  WORKING │ SHORT-TERM │ EPISODIC(sep+salience) │ SEMANTIC(editable) │ PROCEDURAL│ │
│  │            ▲ retrieve (learned) ▲                    ▲ consolidate ▲          │ │
│  └───────────────────────────────────────────────────────────────────────────┘   │
│                                    │                                                │
│         OFFLINE "SLEEP" CONSOLIDATION LOOP: replay→gist→skill→(LoRA)weight-update    │
└──────────────────────────────────────────────────────────────────────────────────┘
```

## 10.2 Execution Pipeline

```
1. INGEST      Goal Manager parses request → pushes goal(s) onto stack; writes to workspace.
2. PERCEIVE    Attention Controller builds working set: recent context + retrieved memory (gated).
3. FAST PASS   System 1 proposes answer + confidence + features.
4. META-ROUTE  Controller reads (conf, stakes, novelty, conflict, budget) → decides route:
                  a) accept (high conf, low stakes)
                  b) verify-light
                  c) escalate to System 2 (+plan / +sim / +tools / +search)
5. DELIBERATE  (if escalated) Planner decomposes → per-subgoal:
                  World Model simulates candidate actions;
                  Tool Planner offloads to verifiers/execution;
                  System 2 reasons with shallow, value-guided search.
6. CRITIQUE    Reflection/Critic checks each candidate against goal + external verifier.
                  reject → revise (loop, budget-bounded) ; accept → continue.
7. DECIDE      Confidence Estimator gates final output → answer | defer | ask-user.
8. ACT/EMIT    Emit answer / take action.
9. RECORD      Write salient trajectory to Episodic memory (salience-gated).
               Curiosity logs knowledge gaps as future exploration targets.
10. (OFFLINE)  Consolidation: prioritized replay → extract semantic gist,
               distill repeated plans into Procedural skills, periodic LoRA weight update,
               prune/merge episodic store.
```

## 10.3 The Loops (each is a first-class, separately-tuned control loop)

- **Reasoning loop** (steps 3–7): fast→route→deliberate→critique→decide. Budget-bounded.
- **Planning loop** (within 5): decompose→simulate→select→execute→re-plan on failure (hierarchical; caches successful plans).
- **Reflection loop** (6): candidate→verify→revise, with an *external* verifier whenever available (RLVR-grade grounding), falling back to the trained critic otherwise.
- **Tool-use loop:** select tool→construct args→execute→integrate→verify result.
- **Learning loop** (9–10): episodic write → offline replay/consolidation → semantic/procedural/weight update. The flywheel: today's expensive search becomes tomorrow's fast intuition (AlphaZero-style distillation).

## 10.4 Memory Hierarchy & Attention Flow

- **Hierarchy:** Working (typed slots, seconds) → Short-term (session gist) → Episodic (salience-gated events, learned index, pattern-separated) → Semantic (editable, provenance-tagged) → Procedural (skill library). Consolidation moves information *down* the persistence gradient offline.
- **Attention flow:** top-down goal signal from Goal Manager + Controller biases (a) which memories the retriever fetches, (b) which workspace slots stay active, and (c) which are evicted — fighting context dilution and keeping the model *on-goal* over long horizons. Bottom-up salience (surprise, verifier failure) can interrupt and re-route (the ventral-attention analog).

## 10.5 Confidence Estimation & Failure Recovery

- **Confidence:** ensemble of (calibrated intrinsic head) + (self-consistency spread) + (verifier pass rate), decomposed into epistemic (→ gather info / retrieve / ask) vs aleatoric (→ report irreducible uncertainty).
- **Failure recovery ladder** (escalate only as needed): re-sample → reflect/revise → retrieve more → invoke tool/verifier → widen search → re-plan → **defer/ask user**. Each rung costs more; the controller climbs only while budget remains and expected value of extra compute is positive. Crucially, **"I don't know / here's my uncertainty"** is a first-class terminal action, not a failure — this alone would eliminate a large fraction of confident hallucinations.

## 10.6 Cost Analysis

- **Amortization is the whole game.** Route ~80% to System 1 (1×), ~15% to light verify (~2×), ~5% to full System 2 (10–100×). Expected cost ≈ `0.8·1 + 0.15·2 + 0.05·50 ≈ 3.6×` a pure-fast system, while capturing most of the accuracy of an "always-think-hard" system that would cost ~50×. **Calibration quality directly sets the cost/accuracy frontier** — a 10-point calibration improvement can halve deliberation spend at equal accuracy.
- **Memory cost:** retrieval + storage is cheap vs. deliberation; the win (avoiding re-derivation of known results, avoiding repeated mistakes) is large. Consolidation is offline/batched (cheap per query, amortized).
- **World-model simulation** is the priciest sub-module; gate it behind the planner and use latent (JEPA/Dreamer-style) rollouts, not token-level generation, to keep rollouts cheap.

## 10.7 Training Strategy

Staged, module-respecting:
1. **Pretrain** the shared backbone (next-token + auxiliary *latent-prediction* objectives to seed a world model).
2. **SFT** for instruction following, tool-call format, structured memory I/O, and reasoning traces.
3. **RLVR** on verifiable domains (math/code/logic/tool-tasks) to sharpen System 2 and the critic against *grounded* rewards; add **process rewards** for step-level supervision.
4. **Train the Controller** with RL on end-task reward *minus a compute/latency cost* — this is what teaches *when thinking pays* (the metacognitive policy). Train the **Confidence Estimator** with proper scoring rules + post-hoc calibration.
5. **Train the retriever** end-to-end on downstream success (not just similarity).
6. **World model:** self-supervised latent prediction + (where available) interventional/embodied/simulated experience to instill `P(s'|s,do(a))`.
7. **Consolidation / continual learning:** replay-based offline updates (LoRA/adapters to bound drift), generative replay to avoid catastrophic forgetting, with eval-gated promotion of updates.

## 10.8 Inference Strategy

Adaptive-compute, controller-driven (Part 5). Default cheap; escalate on calibrated need; always allow defer/ask. Cache: successful plans → procedural skills; frequent semantic queries → fast lookups; recent context → gist compression. Prefer *shallow, value-guided* search over broad brute-force (amortized policy makes search cheap).

## 10.9 Evaluation Benchmarks

- **Reasoning:** GPQA-Diamond, AIME/MATH, ARC-AGI (novel abstraction), FrontierMath, competition code (Codeforces/SWE-bench for agentic coding).
- **Long-horizon agency:** SWE-bench Verified, WebArena/OSWorld, GAIA, τ-bench (tool/agent), long-horizon task suites.
- **Memory & continual learning:** LoCoMo (long conversational memory), continual-learning suites measuring forward transfer *and* backward retention (catastrophic-forgetting resistance).
- **Calibration:** ECE / selective-prediction (accuracy vs. coverage), Brier score, and *knowing-when-to-defer* metrics.
- **Causal/counterfactual:** CRASS, causal-reasoning benchmarks, do-vs-see probes.
- **World model:** planning-accuracy in simulated environments; prediction error of latent rollouts.
- **Efficiency:** accuracy-per-FLOP and accuracy-per-dollar under adaptive compute (the metric that actually matters for the routing thesis).

## 10.10 Expected Advantages

- **Cost-adaptive accuracy:** most of "always-think-hard" quality at a fraction of the cost.
- **Reduced hallucination:** calibrated confidence + verifiers + defer-action.
- **Long-horizon reliability:** persistent goal stack + attention-gated memory beat plan-drift.
- **Lifelong learning:** consolidation loop lets it improve from experience without retraining from scratch.
- **Grounded reasoning:** separable world model enables real simulation, causal/counterfactual queries.
- **Interpretability & steerability:** typed modules and explicit memory are far more inspectable than a monolith.

## 10.11 Expected Bottlenecks

- **Controller training is hard:** the metacognitive policy needs good confidence signals *and* good cost accounting — a chicken/egg with the confidence estimator.
- **Consolidation stability:** avoiding catastrophic forgetting during online-ish weight updates is unsolved at scale; replay is expensive.
- **World-model fidelity:** long rollouts drift; latent models are hard to train to reasoning-grade fidelity for open domains.
- **Module interface overhead:** latent hand-offs risk information loss; text hand-offs are lossy and slow. Getting the shared workspace representation right is a research problem.
- **Verifier availability:** RLVR-grade grounding is scarce outside math/code — the critic is only as good as its verifier.
- **Latency variance:** adaptive compute means unpredictable latency (a product/ops problem).

## 10.12 Research Challenges (open problems)

1. **Inference-time weight learning that is safe and stable** — the frozen-weights gap. The single most important unsolved problem.
2. **Reliable, calibrated metacognition** — self-knowledge of what one does and doesn't know, decomposed epistemic/aleatoric.
3. **General verifiers** — grounding reflection beyond formally-checkable domains.
4. **Faithful reasoning** — ensuring the trace reflects the true computation (CoT unfaithfulness is documented).
5. **Compositional, editable world models** for open domains supporting true `do`-interventions.
6. **Learned salience & consolidation policies** — what to remember, what to forget, what to distill.
7. **A shared workspace representation** rich enough for latent inter-module communication without lossy text bottlenecks.
8. **Causal representation learning** from largely observational data.
9. **Evaluation of continual learning** — benchmarks that reward lifelong improvement and penalize forgetting.
10. **Alignment of a self-improving, curious, goal-persistent agent** — every added capability (persistent goals, intrinsic drives, self-modification) raises the alignment stakes proportionally.

---

## Engineering Recommendations (prioritized, buildable-first)

1. **Build the dual-system router now** (Part 5). Highest ROI, buildable on today's models: a calibrated confidence estimator + a controller that routes fast vs. slow. Immediate cost/accuracy wins.
2. **Add a structured, consolidating memory** (Part 7) — typed episodic store, learned retriever, offline gist/skill consolidation. Replace naive RAG. Turns sessions into learning.
3. **Promote the critic to a trained module grounded in external verifiers** (RLVR everywhere a verifier exists). Attack hallucination at the source.
4. **Make defer/ask/express-uncertainty a first-class output.** Cheapest single hallucination reduction.
5. **Introduce a persistent Goal Manager** for long-horizon agents. Fixes plan-drift.
6. **Invest in a separable latent world model** (JEPA/Dreamer-style) for planning and causal/counterfactual reasoning — the medium-term high-leverage bet.
7. **Close the learning loop** with replay-based offline consolidation and eval-gated LoRA updates — the long-term moonshot (continual learning).

## Open Research Questions (restated as testable programs)

- Can a learned controller allocate test-time compute to beat fixed-budget reasoning on accuracy-per-dollar across diverse tasks? (Measurable now.)
- Does replay-based consolidation enable measurable lifelong improvement without backward forgetting at frontier scale?
- Can a latent world model, trained with interventional data, reach interventional (rung-2/3) causal competence that pure token-prediction cannot?
- Is a global-workspace latent bottleneck a better inter-module channel than text, measured by end-task and information-retention?
- Can calibrated epistemic/aleatoric confidence be trained to drive both routing and honest deferral, and does it transfer OOD?

---

## References (representative; author–year)

**Cognitive science / neuroscience**
- Kahneman, D. (2011). *Thinking, Fast and Slow.*
- Evans, J. & Stanovich, K. (2013). Dual-process theories. *Perspectives on Psych. Science.*
- Baddeley, A. & Hitch, G. (1974). Working memory.
- Miller, G. (1956). The magical number seven. *Psych. Review.*
- Cowan, N. (2001). The magical number four. *BBS.*
- Tulving, E. (1972; 1983). Episodic and semantic memory.
- Squire, L. (2004). Memory systems taxonomy.
- McClelland, J., McNaughton, B. & O'Reilly, R. (1995). Complementary Learning Systems. *Psych. Review.*
- Schultz, W., Dayan, P. & Montague, P. (1997). A neural substrate of prediction and reward. *Science.*
- Daw, N. et al. (2005). Model-based vs model-free control. *Nature Neuroscience.*
- Botvinick, M. et al. (2009). Hierarchical RL and the PFC.
- Koechlin, E. & Badre, D. — rostro-caudal PFC hierarchy.
- Fleming, S. & Dolan, R. (2012). The neural basis of metacognition. *Phil. Trans. R. Soc.*
- Nelson, T. & Narens, L. (1990). Metamemory framework.
- Gentner, D. (1983). Structure-mapping. *Cognitive Science.*
- Johnson-Laird, P. (1983). *Mental Models.*
- Pearl, J. (2009; 2018). *Causality*; *The Book of Why.*
- Friston, K. (2010). The free-energy principle. *Nature Reviews Neuroscience.*
- Clark, A. (2013). Whatever next? Predictive processing. *BBS.*
- Rao, R. & Ballard, D. (1999). Predictive coding. *Nature Neuroscience.*
- Battaglia, P., Hamrick, J. & Tenenbaum, J. (2013). Intuitive physics engine. *PNAS.*
- Ma, W.J. et al. (2006). Probabilistic population codes. *Nature Neuroscience.*
- Schmidhuber, J. (1991–2010). Curiosity / compression progress.
- Oudeyer, P.-Y. & Kaplan, F. (2007). Intrinsic motivation.
- Lake, B., Ullman, T., Tenenbaum, J. & Gershman, S. (2017). Building machines that learn and think like people. *BBS.*
- Baars, B.; Dehaene, S. — Global Workspace Theory.

**Machine learning / LLMs**
- Vaswani, A. et al. (2017). Attention is all you need. *NeurIPS.*
- Brown, T. et al. (2020). GPT-3 / few-shot learners. *NeurIPS.*
- Kaplan, J. et al. (2020). Scaling laws.
- Hoffmann, J. et al. (2022). Chinchilla / compute-optimal.
- Wei, J. et al. (2022). Emergent abilities. *TMLR.*
- Schaeffer, R. et al. (2023). Are emergent abilities a mirage? *NeurIPS.*
- Wei, J. et al. (2022). Chain-of-thought prompting. *NeurIPS.*
- Kojima, T. et al. (2022). Zero-shot CoT. *NeurIPS.*
- Nye, M. et al. (2021). Show your work / scratchpads.
- Wang, X. et al. (2022). Self-consistency. *ICLR.*
- Yao, S. et al. (2022). ReAct. *ICLR.*
- Yao, S. et al. (2023). Tree of Thoughts. *NeurIPS.*
- Besta, M. et al. (2024). Graph of Thoughts. *AAAI.*
- Shinn, N. et al. (2023). Reflexion. *NeurIPS.*
- Du, Y. et al. (2023). Multi-agent debate.
- Lightman, H. et al. (2023). Let's verify step by step (process rewards). OpenAI.
- Snell, C. et al. (2024). Scaling test-time compute.
- OpenAI (2024). Learning to reason with LLMs (o1).
- DeepSeek-AI (2025). DeepSeek-R1: incentivizing reasoning via RL (RLVR/GRPO).
- Ouyang, L. et al. (2022). InstructGPT / RLHF. *NeurIPS.*
- Bai, Y. et al. (2022). Constitutional AI (RLAIF). Anthropic.
- Shazeer, N. et al. (2017). Sparsely-gated MoE. *ICLR.*
- Fedus, W. et al. (2021). Switch Transformer.
- Jiang, A. et al. (2024). Mixtral of Experts. Mistral AI.
- Lewis, P. et al. (2020). RAG. *NeurIPS.*
- Guu, K. et al. (2020). REALM. *ICML.*
- Borgeaud, S. et al. (2022). RETRO. *ICML.*
- Graves, A. et al. (2014; 2016). Neural Turing Machines; DNC. *Nature.*
- Park, J. et al. (2023). Generative Agents. *UIST.*
- Packer, C. et al. (2023). MemGPT.
- Sumers, T. et al. (2023). CoALA: Cognitive Architectures for Language Agents.
- Schick, T. et al. (2023). Toolformer.
- Wang, G. et al. (2023). Voyager.
- Silver, D. et al. (2016; 2017). AlphaGo; AlphaZero. *Nature/Science.*
- Schrittwieser, J. et al. (2020). MuZero. *Nature.*
- Hafner, D. et al. (2020; 2023). Dreamer / DreamerV3.
- LeCun, Y. (2022). A path towards autonomous machine intelligence (JEPA).
- Assran, M. et al. (2023). I-JEPA.
- Li, K. et al. (2023). Emergent world representations (Othello-GPT). *ICLR.*
- Schölkopf, B. et al. (2021). Toward causal representation learning. *Proc. IEEE.*
- Trinh, T. et al. (2024). AlphaGeometry. *Nature.*
- Mao, J. et al. (2019). Neuro-Symbolic Concept Learner. *ICLR.*
- Goyal, A. & Bengio, Y. (2022). Inductive biases for higher cognition / consciousness prior.
- VanRullen, R. & Kanai, R. (2021). Global workspace for AI.
- Schaul, T. et al. (2016). Prioritized experience replay. *ICLR.*
- Shin, H. et al. (2017). Continual learning with generative replay. *NeurIPS.*

*Note on citations: this is a synthesis report; author–year attributions reflect well-established works to guide reading. Verify exact venues/years against primary sources before formal use.*

---

*End of report.*
