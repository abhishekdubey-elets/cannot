"""S-CTRL — the controlled synthetic task stream (EXPERIMENT_PLAN.md §4).

A stream of task *families* (schemas) introduced sequentially. Each family has:
  - train tasks        : the online stream the agent solves as it arrives
  - retention_novel    : held-out NEW-number instances of the schema
                         -> measures skill transfer (NIST) and retention
  - retention_repeat   : EXACT duplicates of train prompts
                         -> measures exact-instance recall (what memory buys)

Plus a fixed general-capability probe `G`, unrelated to the stream, used to
detect erosion of base competence (GCR / catastrophic forgetting).

Design points that make the harness a valid test:
  * `solvability` in [0,1) is fixed per instance at generation time, so a
    solver's correctness is DETERMINISTIC given (method, learned-state). Seeds
    vary the *instances*, not hidden coin-flips -> across-seed variance reflects
    genuine sampling, which is what the stability gate checks.
  * prompts embed the varying numbers so distinct instances do not collide in
    the episodic store's near-duplicate merge (memory §7.3) — a harness-hygiene
    detail, verified by the exact-recall numbers in Table 2.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Task:
    id: str
    schema: str
    prompt: str
    answer: str
    solvability: float      # fixed intrinsic difficulty draw in [0,1)
    kind: str               # "train" | "novel" | "repeat" | "general"
    is_general: bool = False


@dataclass
class Family:
    schema: str
    base_success: float                       # P(correct) for an UNLEARNED agent
    train: list[Task]
    retention_novel: list[Task]
    retention_repeat: list[Task]


@dataclass
class Stream:
    families: list[Family]
    general_probe: list[Task]
    base_success: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        self.base_success = {f.schema: f.base_success for f in self.families}


# Templated schemas. `fn` computes the ground-truth answer; `text` renders a
# prompt whose number tokens dominate the bag-of-words (keeps episodes distinct).
_TEMPLATES = [
    ("crates", lambda a, b, c: a * b - c,
     "[{s}] store had {a} racks x {b} crates, shipped {c}. left?"),
    ("garden", lambda a, b, c: (a + b) * c,
     "[{s}] {a} beds plus {b} beds, each {c} plants. total?"),
    ("fleet", lambda a, b, c: a * c + b,
     "[{s}] {a} vans x {c} trips, plus {b} spare. runs?"),
    ("ledger", lambda a, b, c: (a - b) * c,
     "[{s}] balance {a} minus {b}, times {c} months. sum?"),
    ("orchard", lambda a, b, c: a * b + a * c,
     "[{s}] {a} trees give {b} apples and {c} pears. fruit?"),
]


def _mk(rng: random.Random, schema: str, tmpl, kind: str, idx: int) -> Task:
    name, fn, text = tmpl
    a, b, c = rng.randint(2, 12), rng.randint(1, 9), rng.randint(2, 9)
    prompt = text.format(s=schema, a=a, b=b, c=c)
    return Task(id=f"{schema}-{kind}-{idx}", schema=schema, prompt=prompt,
                answer=str(fn(a, b, c)), solvability=rng.random(), kind=kind)


def make_stream(seed: int, n_families: int = 6, n_train: int = 8,
                n_novel: int = 30, n_repeat: int = 12, n_general: int = 40) -> Stream:
    rng = random.Random(seed)
    families: list[Family] = []
    for k in range(n_families):
        tmpl = _TEMPLATES[k % len(_TEMPLATES)]
        schema = f"S{k}"
        base = round(rng.uniform(0.20, 0.40), 3)
        train = [_mk(rng, schema, tmpl, "train", i) for i in range(n_train)]
        novel = [_mk(rng, schema, tmpl, "novel", i) for i in range(n_novel)]
        # repeats reuse a trained prompt/answer exactly (fresh solvability so the
        # NO-memory path still faces the schema's base difficulty on them).
        repeat = []
        for i in range(n_repeat):
            src = rng.choice(train)
            repeat.append(Task(id=f"{schema}-repeat-{i}", schema=schema,
                               prompt=src.prompt, answer=src.answer,
                               solvability=rng.random(), kind="repeat"))
        families.append(Family(schema, base, train, novel, repeat))

    # General probe: a distinct family the stream never trains on.
    gtmpl = ("gen", lambda a, b, c: a + b + c,
             "[G] add {a} and {b} and {c}. total?")
    general = []
    for i in range(n_general):
        t = _mk(rng, "G", gtmpl, "general", i)
        general.append(Task(t.id, t.schema, t.prompt, t.answer, t.solvability,
                            "general", is_general=True))
    return Stream(families, general)


# --------------------------------------------------------------------------- #
# S-TOOL — realistic-style stream with SHARED PRIMITIVE SKILLS (Phase 2, §4).
#
# Each family is a short composition of tool primitives (search/filter/aggregate/
# format/auth). Consecutive families overlap by one primitive, so mastering an
# early family's primitives partially solves a later family BEFORE it is trained
# -> forward transfer (FWT > 0), the signal a controlled independent-schema
# stream cannot produce. The prompt encodes the primitives as `[Fk|pi+pj]` so the
# competence double (ToolFakeLM) can score partial mastery.
# --------------------------------------------------------------------------- #

_PRIMS = ["search", "filter", "aggregate", "format", "auth"]


def _mk_tool(rng: random.Random, schema: str, prims: list[str], kind: str,
             idx: int) -> Task:
    a, b, c = rng.randint(2, 12), rng.randint(1, 9), rng.randint(2, 9)
    tag = f"{schema}|{'+'.join(prims)}"
    prompt = f"[{tag}] tool-task {a},{b},{c}"
    return Task(id=f"{schema}-{kind}-{idx}", schema=schema, prompt=prompt,
                answer=str(a + b + c), solvability=rng.random(), kind=kind)


def make_tool_stream(seed: int, n_families: int = 6, n_train: int = 8,
                     n_novel: int = 30, n_repeat: int = 12,
                     n_general: int = 80) -> Stream:
    rng = random.Random(seed)
    families: list[Family] = []
    for k in range(n_families):
        schema = f"F{k}"
        prims = [_PRIMS[k % len(_PRIMS)], _PRIMS[(k + 1) % len(_PRIMS)]]  # overlap by 1
        base = round(rng.uniform(0.15, 0.30), 3)
        train = [_mk_tool(rng, schema, prims, "train", i) for i in range(n_train)]
        novel = [_mk_tool(rng, schema, prims, "novel", i) for i in range(n_novel)]
        repeat = []
        for i in range(n_repeat):
            src = rng.choice(train)
            repeat.append(Task(f"{schema}-repeat-{i}", schema, src.prompt, src.answer,
                               rng.random(), "repeat"))
        families.append(Family(schema, base, train, novel, repeat))

    general = []
    for i in range(n_general):
        a, b, c = rng.randint(2, 12), rng.randint(1, 9), rng.randint(2, 9)
        general.append(Task(f"G-general-{i}", "G",
                            f"[G] add {a} and {b} and {c}. total?", str(a + b + c),
                            rng.random(), "general", is_general=True))
    return Stream(families, general)
