"""ADVERSARIAL routing — where the right mechanism is NOT visible in the surface form.

The clean hard workload gave routing accuracy 1.000 because task types were surface-
distinct (a giant multiplication looks nothing like a registry lookup). Real workloads
are not. This workload makes routing a METACOGNITIVE problem: two fact types are
phrased IDENTICALLY, and the correct mechanism depends on whether the model KNOWS the
answer or must retrieve it — a judgment about its own knowledge.

  known_fact   -> COT       real-world fact the model knows (e.g. capital of France)
  unknown_fact -> RETRIEVE  a fact only in the private KB, phrased identically to known
  easy_calc    -> COT       small arithmetic CoT handles
  hard_calc    -> TOOL      large arithmetic CoT cannot

known_fact and unknown_fact are surface-indistinguishable ("What is the capital of X?"),
so a router must decide answerable-from-knowledge vs needs-lookup. If routing accuracy
drops on the fact pair, dispatch is bounded by the router's metacognition, not just by
capability-disjointness — the honest limit of the clean result.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from dispatch.hard import Query  # reuse dataclass  # noqa: F401

TYPES = ["known_fact", "unknown_fact", "easy_calc", "hard_calc"]
MECHS = ["fast", "cot", "tool", "retrieve"]
TYPE_TO_MECH = {"known_fact": "cot", "unknown_fact": "retrieve",
                "easy_calc": "cot", "hard_calc": "tool"}

# real facts a 3B model reliably knows
KNOWN = {"France": "Paris", "Japan": "Tokyo", "Italy": "Rome", "Spain": "Madrid",
         "Germany": "Berlin", "Egypt": "Cairo", "Canada": "Ottawa", "Russia": "Moscow",
         "China": "Beijing", "Greece": "Athens", "Portugal": "Lisbon", "Austria": "Vienna",
         "Cuba": "Havana", "Norway": "Oslo", "Poland": "Warsaw", "Ireland": "Dublin"}
# fictional "countries" whose capitals exist ONLY in the KB (identical surface form).
# Built combinatorially so we can draw UNIQUE entities without replacement (duplicate
# entities would overwrite the KB and desync the stored answer).
_FIC_PRE = ["Zeml", "Qorv", "Elbo", "Draz", "Vulp", "Noor", "Sylth", "Grenn",
            "Marov", "Tavis", "Bruv", "Kelm", "Ordr", "Pyre", "Xanth", "Wext"]
_FIC_SUF = ["ania", "oria", "mir", "arn", "ova", "nik", "tania", "enne"]
FICTIONAL = [p + s for p in _FIC_PRE for s in _FIC_SUF]         # 128 unique names
_CAPS = ["Qorvath", "Mirelle", "Vantis", "Solgrad", "Threnn", "Aloria", "Dunmar", "Kesh"]


def make_adversarial(seed: int, n_per_type: int):
    rng = random.Random(seed)
    kb: dict[str, dict[str, str]] = {}
    qs: list[Query] = []

    for _ in range(n_per_type):                                # known: cot answers
        c = rng.choice(list(KNOWN))
        qs.append(Query("known_fact", f"What is the capital of {c}?", KNOWN[c], "cot"))

    for c in rng.sample(FICTIONAL, n_per_type):                # unknown: unique entities
        cap = rng.choice(_CAPS) + str(rng.randint(1, 99))
        kb[c] = {"capital": cap}
        qs.append(Query("unknown_fact", f"What is the capital of {c}?", cap, "retrieve",
                        {"entity": c, "attr": "capital"}))

    for _ in range(n_per_type):                                # easy calc: cot
        a, b = rng.randint(11, 49), rng.randint(3, 9)
        qs.append(Query("easy_calc", f"How much is {a} times {b}?", str(a * b), "cot"))

    for _ in range(n_per_type):                                # hard calc: tool
        a, b = rng.randint(1_000_000, 9_999_999), rng.randint(1_000_000, 9_999_999)
        qs.append(Query("hard_calc", f"How much is {a} times {b}?", str(a * b), "tool"))

    rng.shuffle(qs)
    return qs, kb


ROUTER_SYS = (
    "You are a router. Reply with only one category word.\n"
    "- known_fact: a general-knowledge question you can answer from what you already know\n"
    "- unknown_fact: a question about an obscure/private entity you do NOT know and must look up\n"
    "- easy_calc: a small arithmetic problem you can compute directly\n"
    "- hard_calc: multiplying large numbers, needing an exact calculator")
