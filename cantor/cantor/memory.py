"""Memory system — report Part 7.

Implements the five-store hierarchy with the properties the report argues naive
RAG lacks:

  WorkingMemory     typed, size-bounded slots with attention-gated eviction (§1.2)
  ShortTermMemory   rolling session gist (§7.1)
  EpisodicMemory    typed event records, salience-GATED writes (§7.4), decay,
                    and Generative-Agents-style retrieval scoring
                    score = a*recency + b*relevance + c*importance (Park 2023)
  SemanticMemory    editable, provenance-tagged facts (§1.5 gap: editability)
  ProceduralMemory  cached skills/plans keyed by task signature (§1.17, §6.3)

Embeddings are a stdlib-only bag-of-words cosine stand-in; swap `embed()` for a
real encoder. Pattern separation (§7.3) is approximated by dedup-on-write.
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter
from dataclasses import dataclass, field


# --------------------------------------------------------------------------- #
# Cheap embedding stand-in
# --------------------------------------------------------------------------- #

def embed(text: str) -> Counter:
    return Counter(re.findall(r"[a-z0-9]+", text.lower()))


def cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    dot = sum(a[t] * b[t] for t in a)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


# --------------------------------------------------------------------------- #
# Working memory (§1.2): typed slots, bounded, gated
# --------------------------------------------------------------------------- #

@dataclass
class Slot:
    role: str        # e.g. "goal", "retrieved", "intermediate", "plan"
    content: str
    priority: float  # attention-controller weight; low priority evicted first


class WorkingMemory:
    """Bounded typed buffer. Capacity limit is a *feature*: it forces the
    attention controller to keep only goal-relevant content (§1.16), instead of
    passively retaining everything like a raw context window."""

    def __init__(self, capacity: int = 12):
        self.capacity = capacity
        self.slots: list[Slot] = []

    def write(self, role: str, content: str, priority: float = 0.5) -> None:
        self.slots.append(Slot(role, content, priority))
        if len(self.slots) > self.capacity:
            # Attention-gated eviction: drop lowest-priority, oldest first.
            self.slots.sort(key=lambda s: s.priority)
            self.slots = self.slots[len(self.slots) - self.capacity:]

    def render(self) -> str:
        ordered = sorted(self.slots, key=lambda s: -s.priority)
        return "\n".join(f"[{s.role}] {s.content}" for s in ordered)

    def clear_role(self, role: str) -> None:
        self.slots = [s for s in self.slots if s.role != role]


# --------------------------------------------------------------------------- #
# Episodic memory (§1.4, §7): typed events, salience gate, decay, retrieval
# --------------------------------------------------------------------------- #

@dataclass
class Episode:
    summary: str
    outcome: str            # "success" | "failure" | "deferred"
    salience: float         # gates the write and weights retrieval
    task_signature: str     # normalized task key, used by consolidation
    timestamp: float = field(default_factory=time.time)
    embedding: Counter = field(default_factory=Counter)
    consolidated: bool = False

    def __post_init__(self):
        if not self.embedding:
            self.embedding = embed(self.summary)


class EpisodicMemory:
    SALIENCE_GATE = 0.3          # below this, the event is never written (§7.4)
    RECENCY_HALFLIFE_S = 3600.0  # decay constant for retrieval scoring

    def __init__(self):
        self.episodes: list[Episode] = []

    def write(self, episode: Episode) -> bool:
        if episode.salience < self.SALIENCE_GATE:
            return False
        # Pattern-separation approximation (§7.3): merge near-duplicates
        # instead of letting them collide at retrieval time.
        for old in self.episodes:
            if cosine(old.embedding, episode.embedding) > 0.95 and old.outcome == episode.outcome:
                old.salience = max(old.salience, episode.salience)
                old.timestamp = episode.timestamp
                return False
        self.episodes.append(episode)
        return True

    def retrieve(self, query: str, k: int = 3, now: float | None = None) -> list[Episode]:
        now = now or time.time()
        q = embed(query)

        def score(e: Episode) -> float:
            recency = math.exp(-(now - e.timestamp) / self.RECENCY_HALFLIFE_S)
            relevance = cosine(q, e.embedding)
            return 0.2 * recency + 0.5 * relevance + 0.3 * e.salience

        return sorted(self.episodes, key=score, reverse=True)[:k]

    def decay(self, min_salience: float = 0.35) -> int:
        """Forgetting as a feature (§7.4): evict low-salience consolidated traces."""
        before = len(self.episodes)
        self.episodes = [e for e in self.episodes
                         if e.salience >= min_salience or not e.consolidated]
        return before - len(self.episodes)


# --------------------------------------------------------------------------- #
# Semantic memory (§1.5): editable, provenance-tagged facts
# --------------------------------------------------------------------------- #

@dataclass
class Fact:
    subject: str
    relation: str
    obj: str
    source: str          # provenance — the thing weights can't give you
    confidence: float

    def render(self) -> str:
        return f"{self.subject} {self.relation} {self.obj} (src={self.source}, p={self.confidence})"


class SemanticMemory:
    def __init__(self):
        self.facts: list[Fact] = []

    def assert_fact(self, fact: Fact) -> None:
        # Editability: a new assertion about the same (subject, relation)
        # replaces the old one — the single-fact update weights can't do (§1.5).
        self.facts = [f for f in self.facts
                      if not (f.subject == fact.subject and f.relation == fact.relation)]
        self.facts.append(fact)

    def query(self, text: str, k: int = 3) -> list[Fact]:
        q = embed(text)
        scored = [(cosine(q, embed(f.render())), f) for f in self.facts]
        return [f for s, f in sorted(scored, key=lambda x: -x[0]) if s > 0.1][:k]


# --------------------------------------------------------------------------- #
# Procedural memory (§1.17, §6.3): cached skills — the distillation flywheel
# --------------------------------------------------------------------------- #

@dataclass
class Skill:
    task_signature: str
    plan: list[str]       # the subgoal decomposition that worked
    answer_template: str  # cached result or result-producing recipe
    successes: int = 1


class ProceduralMemory:
    def __init__(self):
        self.skills: dict[str, Skill] = {}

    def store(self, skill: Skill) -> None:
        existing = self.skills.get(skill.task_signature)
        if existing:
            existing.successes += 1
        else:
            self.skills[skill.task_signature] = skill

    def lookup(self, task_signature: str) -> Skill | None:
        return self.skills.get(task_signature)


def task_signature(query: str) -> str:
    """Normalize a query into a reusable task key (numbers -> N placeholder),
    so 'what is 37*24' and 'what is 41*19' share one skill."""
    return re.sub(r"\d+", "N", query.lower()).strip()
