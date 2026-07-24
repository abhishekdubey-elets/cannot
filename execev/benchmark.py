"""Non-saturating, multi-domain, continuous-difficulty benchmark for E1.

Three domains with DIFFERENT uncertainty structure, each spanning a difficulty gradient so
overall accuracy sits mid-range (a benchmark that saturates at 1.0 is a failed benchmark):

  factoid    knowledge uncertainty : common / rare / relocated / fictional capitals
  arithmetic computation uncertainty: 2-5 digit products
  reasoning  composition uncertainty: 1-3 step small-number word problems

All items are programmatically gradeable (substring for names, integer for numbers).
`difficulty` is an ordinal 0..3 for within-domain analysis.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

DOMAINS = ["factoid", "arithmetic", "reasoning"]

COMMON = {"France": ["Paris"], "Japan": ["Tokyo"], "Italy": ["Rome"], "Spain": ["Madrid"],
          "Germany": ["Berlin"], "Egypt": ["Cairo"], "Russia": ["Moscow"], "China": ["Beijing"],
          "Greece": ["Athens"], "Portugal": ["Lisbon"], "Canada": ["Ottawa"], "Cuba": ["Havana"]}
RARE = {"Bhutan": ["Thimphu"], "Tuvalu": ["Funafuti"], "Comoros": ["Moroni"],
        "Kiribati": ["Tarawa", "South Tarawa"], "Vanuatu": ["Port Vila"], "Suriname": ["Paramaribo"],
        "Djibouti": ["Djibouti"], "Brunei": ["Bandar Seri Begawan"], "Lesotho": ["Maseru"],
        "Eritrea": ["Asmara"], "Bhutan ": ["Thimphu"], "Palau": ["Ngerulmud"]}
RELOCATED = {"Kazakhstan": ["Astana", "Nur-Sultan"], "Myanmar": ["Naypyidaw", "Nay Pyi Taw"],
             "Nigeria": ["Abuja"], "Burundi": ["Gitega"], "Ivory Coast": ["Yamoussoukro"]}
_FIC = [p + s for p in ["Zeml", "Qorv", "Draz", "Vulp", "Sylth", "Marov", "Kelm", "Xanth"]
        for s in ["ania", "oria", "mir", "arn", "ova", "nik"]]


@dataclass(frozen=True)
class Item:
    domain: str
    difficulty: int          # 0 easy .. 3 hardest/misleading
    prompt: str
    accepted: tuple          # any substring counts correct; () = impossible w/o retrieval
    meta: dict = field(default_factory=dict)


def _factoid(rng, n):
    out = []
    pools = [(0, COMMON), (1, RARE), (3, RELOCATED)]
    for diff, pool in pools:
        for c in rng.sample(list(pool), min(len(pool), max(1, n // 4))):
            out.append(Item("factoid", diff, f"What is the capital of {c}?", tuple(pool[c])))
    for c in rng.sample(_FIC, max(1, n // 4)):               # fictional: impossible
        out.append(Item("factoid", 2, f"What is the capital of {c}?", ()))
    return out[:n]


def _arithmetic(rng, n):
    out = []
    for _ in range(n):
        d = rng.choice([2, 3, 4, 5])                          # digit count -> difficulty
        a = rng.randint(10 ** (d - 1), 10 ** d - 1)
        b = rng.randint(10 ** (d - 1), 10 ** d - 1)
        out.append(Item("arithmetic", d - 2, f"What is {a} times {b}?", (str(a * b),)))
    return out


def _reasoning(rng, n):
    out = []
    for _ in range(n):
        steps = rng.choice([1, 2, 3])
        a, b, c, d = (rng.randint(2, 9) for _ in range(4))
        if steps == 1:
            p, v = f"A box holds {a} rows of {b} apples. How many apples?", a * b
        elif steps == 2:
            p, v = (f"A box holds {a} rows of {b} apples; {c} are removed. How many remain?",
                    a * b - c)
        else:
            p, v = (f"A box holds {a} rows of {b} apples; {c} are removed, then the rest is "
                    f"split into {d} equal groups. How many per group?",
                    (a * b - c) // d if d else 0)
        out.append(Item("reasoning", steps, p, (str(v),), {"exact": v}))
    return out


def make_benchmark(seed: int, n_per_domain: int):
    rng = random.Random(seed)
    items = _factoid(rng, n_per_domain) + _arithmetic(rng, n_per_domain) + _reasoning(rng, n_per_domain)
    rng.shuffle(items)
    return items


def graded(item: Item, output: str) -> bool:
    """Programmatic grader. Fictional (no accepted answer) is always incorrect."""
    import re
    if not item.accepted:
        return False
    if item.domain == "factoid":
        o = output.strip().lower()
        return any(a.lower() in o for a in item.accepted)
    nums = re.findall(r"-?\d+", output.replace(",", ""))
    return bool(nums) and nums[-1] == item.accepted[0]
