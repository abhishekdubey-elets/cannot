"""Sleep-inspired consolidation — report §7.5, the CLS loop made concrete.

Offline pass over the episodic store (prioritized by salience, like PER):

  1. REPLAY   salient unconsolidated episodes,
  2. EXTRACT  regularities into SemanticMemory (gist, with provenance),
  3. DISTILL  repeated successful task signatures into ProceduralMemory skills,
  4. PRUNE    decayed episodic traces (forgetting as a feature, §7.4).

The production analog of step 2/3's endpoint is an eval-gated LoRA weight
update (§10.7 step 7); here consolidation targets the explicit stores, which
is the safe first rung of the same ladder.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .memory import EpisodicMemory, Fact, ProceduralMemory, SemanticMemory


@dataclass
class ConsolidationReport:
    replayed: int
    facts_extracted: int
    skills_distilled: int
    episodes_pruned: int


class ConsolidationEngine:
    SKILL_MIN_SUCCESSES = 1   # demo-friendly; production wants >= 3

    def __init__(self, episodic: EpisodicMemory, semantic: SemanticMemory,
                 procedural: ProceduralMemory):
        self.episodic = episodic
        self.semantic = semantic
        self.procedural = procedural

    def sleep(self) -> ConsolidationReport:
        pending = sorted((e for e in self.episodic.episodes if not e.consolidated),
                         key=lambda e: -e.salience)          # prioritized replay
        facts = 0
        by_signature: dict[str, list] = defaultdict(list)

        for ep in pending:
            by_signature[ep.task_signature].append(ep)
            if ep.outcome == "failure":
                # Failures consolidate into cautionary semantic facts —
                # the counterfactual lesson without weight updates (§1.10).
                self.semantic.assert_fact(Fact(
                    subject=ep.task_signature, relation="failed_when",
                    obj=ep.summary[:80], source="episodic-replay", confidence=0.8))
                facts += 1
            elif ep.outcome == "success" and "=>" in ep.summary and ep.salience >= 0.5:
                # Salient (System-2-derived) concrete results become exact
                # semantic facts keyed by the EXACT query — never by the
                # normalized signature, which would over-generalize the answer.
                query, answer = ep.summary.rsplit("=>", 1)
                self.semantic.assert_fact(Fact(
                    subject=query.strip(), relation="answer_is",
                    obj=answer.strip(), source="episodic-replay", confidence=0.95))
                facts += 1
            ep.consolidated = True

        skills = 0
        for sig, eps in by_signature.items():
            wins = [e for e in eps if e.outcome == "success"]
            if len(wins) < self.SKILL_MIN_SUCCESSES:
                continue
            existing = self.procedural.skills.get(sig)
            if existing:
                # Reinforce: repeated success raises the habit's reliability,
                # which the planner's MB/MF arbitration trusts more (§1.12).
                existing.successes += len(wins)
                skills += 1
            # (New procedure creation happens in Planner.cache_success at act
            # time; consolidation only reinforces and extracts facts. §7.5)

        pruned = self.episodic.decay()
        return ConsolidationReport(len(pending), facts, skills, pruned)
