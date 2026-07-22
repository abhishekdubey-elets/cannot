"""Heterogeneous workload for the dispatch hypothesis (self-contained, no internet).

The cognitive-architecture claim is only testable if NO single mechanism wins on all
query types. This workload is designed with four types, each with a DIFFERENT naturally
-suited mechanism, so that oracle dispatch >> best fixed mechanism:

  factual    -> FAST     (model knows it; CoT wastes tokens, tool/retrieve useless)
  arithmetic -> TOOL     (exact multi-digit compute; fast/CoT unreliable)
  word       -> COT      (multi-step MAPPING with small numbers; needs reasoning)
  recall     -> RETRIEVE (arbitrary fact stated only in a provided context passage)

If this design does not produce a large oracle-minus-best-fixed gap on the REAL model
(measured in run_calibrate.py), the dispatch experiment is not worth running.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

TYPES = ["factual", "arithmetic", "word", "recall"]

CAPITALS = {
    "France": "Paris", "Japan": "Tokyo", "Egypt": "Cairo", "Brazil": "Brasilia",
    "Canada": "Ottawa", "Norway": "Oslo", "Kenya": "Nairobi", "Peru": "Lima",
    "Greece": "Athens", "Thailand": "Bangkok", "Poland": "Warsaw", "Chile": "Santiago",
    "Portugal": "Lisbon", "Sweden": "Stockholm", "Vietnam": "Hanoi", "Cuba": "Havana",
    "Austria": "Vienna", "Ireland": "Dublin", "Morocco": "Rabat", "Hungary": "Budapest",
    "Finland": "Helsinki", "Iceland": "Reykjavik", "Ghana": "Accra", "Nepal": "Kathmandu",
}
_NOUNS = ["Zorblax", "Quenith", "Fendral", "Wextor", "Plimuth", "Garnix", "Vundor", "Krelby"]


@dataclass(frozen=True)
class Query:
    qtype: str
    prompt: str          # full prompt shown to every mechanism (incl. context if any)
    answer: str          # ground truth (string; ints as str)
    best_mechanism: str  # the naturally-suited mechanism (for the oracle / routing target)
    meta: dict = field(default_factory=dict)


def _factual(rng):
    c = rng.choice(list(CAPITALS))
    return Query("factual", f"What is the capital of {c}?", CAPITALS[c], "fast")


def _arithmetic(rng):
    a, b, c = rng.randint(21, 89), rng.randint(11, 39), rng.randint(10, 99)
    return Query("arithmetic", f"Compute {a} * {b} + {c}.", str(a * b + c), "tool")


def _word(rng):
    a, b, c = rng.randint(2, 9), rng.randint(2, 9), rng.randint(1, 9)
    ans = ((a * b) - c) * 2
    p = (f"A store has {a} shelves, each holding {b} boxes. {c} boxes are removed, "
         f"then the number remaining is doubled. How many boxes are there now?")
    return Query("word", p, str(ans), "cot")


def _recall(rng):
    name = rng.choice(_NOUNS)
    temp, mass, year = rng.randint(100, 999), rng.randint(10, 99), rng.randint(2100, 2999)
    ctx = (f"Context: The {name} device operates at {temp} kelvin, weighs {mass} kg, "
           f"and was certified in the year {year}.")
    field_q = rng.choice([("operating temperature in kelvin", str(temp)),
                          ("mass in kg", str(mass)),
                          ("certification year", str(year))])
    p = f"{ctx}\nQuestion: What is the {name} device's {field_q[0]}? Answer with the number only."
    return Query("recall", p, field_q[1], "retrieve", {"name": name})


_GEN = {"factual": _factual, "arithmetic": _arithmetic, "word": _word, "recall": _recall}


def make_workload(seed: int, n_per_type: int) -> list[Query]:
    rng = random.Random(seed)
    out = []
    for t in TYPES:
        for _ in range(n_per_type):
            out.append(_GEN[t](rng))
    rng.shuffle(out)
    return out


def check(query: Query, output: str) -> bool:
    """Grade a mechanism's raw text output against ground truth."""
    import re
    o = output.strip().lower()
    if query.qtype == "factual":
        return query.answer.lower() in o
    nums = re.findall(r"-?\d+", output)
    return bool(nums) and nums[-1] == query.answer
