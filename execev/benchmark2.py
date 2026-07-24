"""E1b benchmark — difficulty-stratified, with complexity metadata for every item.

E1's benchmark is preserved verbatim in benchmark.py for reproducibility. This version fixes
E1's realised flaws and adds the machinery for the arithmetic-confidence-dynamics objective:

  * every domain has 4 EXPLICIT difficulty tiers, allocated evenly (not sampled), so each tier
    gets equal n -- E1 could not support per-tier analysis;
  * reasoning is hardened to 1/2/3/5 inference steps (E1 saturated at 0.917);
  * every item carries a `complexity` scalar and, for arithmetic, an exact `carries` count,
    so confidence can be regressed against difficulty rather than only against correctness.

Uncertainty types under test (the E1b framing):
  factoid    KNOWLEDGE uncertainty     "do I know this?"
  arithmetic COMPUTATIONAL uncertainty "can I execute this correctly?"
  reasoning  COMPOSITIONAL uncertainty "can I chain these steps correctly?"
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

DOMAINS = ["factoid", "arithmetic", "reasoning"]
TIERS = [0, 1, 2, 3]

COMMON = {"France": ["Paris"], "Japan": ["Tokyo"], "Italy": ["Rome"], "Spain": ["Madrid"],
          "Germany": ["Berlin"], "Egypt": ["Cairo"], "Russia": ["Moscow"], "China": ["Beijing"],
          "Greece": ["Athens"], "Portugal": ["Lisbon"], "Canada": ["Ottawa"], "Cuba": ["Havana"]}
RARE = {"Bhutan": ["Thimphu"], "Tuvalu": ["Funafuti"], "Comoros": ["Moroni"],
        "Kiribati": ["Tarawa"], "Vanuatu": ["Port Vila"], "Suriname": ["Paramaribo"],
        "Brunei": ["Bandar Seri Begawan"], "Lesotho": ["Maseru"], "Eritrea": ["Asmara"],
        "Palau": ["Ngerulmud"], "Nauru": ["Yaren"], "Guyana": ["Georgetown"]}
RELOCATED = {"Kazakhstan": ["Astana", "Nur-Sultan"], "Myanmar": ["Naypyidaw", "Nay Pyi Taw"],
             "Nigeria": ["Abuja"], "Burundi": ["Gitega"], "Ivory Coast": ["Yamoussoukro"],
             "Tanzania": ["Dodoma"]}
_FIC = [p + s for p in ["Zeml", "Qorv", "Draz", "Vulp", "Sylth", "Marov", "Kelm", "Xanth"]
        for s in ["ania", "oria", "mir", "arn", "ova", "nik"]]


@dataclass(frozen=True)
class Item:
    domain: str
    tier: int                # 0..3 ordinal difficulty WITHIN domain
    prompt: str
    accepted: tuple
    complexity: float        # continuous work proxy (domain-specific units)
    carries: int             # exact carry count (arithmetic); -1 where undefined
    meta: dict = field(default_factory=dict)


# ---------------- exact carry counting (complexity independent of tier) ----------------

def _carries_addlist(nums):
    """Carries generated when summing a list of ints column-wise."""
    c, carry, pos = 0, 0, 0
    while carry or any(n // (10 ** pos) for n in nums):
        s = sum((n // (10 ** pos)) % 10 for n in nums) + carry
        carry = s // 10
        if carry:
            c += 1
        pos += 1
        if pos > 40:
            break
    return c


def carries_mul(a: int, b: int) -> int:
    """Carries in standard long multiplication: within each partial product, plus the final sum."""
    total, partials = 0, []
    for i, db in enumerate(reversed(str(b))):
        d, carry, digits = int(db), 0, []
        for da in reversed(str(a)):
            p = int(da) * d + carry
            digits.append(p % 10)
            carry = p // 10
            if carry:
                total += 1
        if carry:
            digits.append(carry)
        partials.append(int("".join(str(x) for x in reversed(digits)) or "0") * (10 ** i))
    return total + _carries_addlist(partials)


# ---------------- domains ----------------

def _factoid(rng, per_tier):
    out = []
    for tier, pool in [(0, COMMON), (1, RARE), (2, RELOCATED)]:
        for c in rng.sample(list(pool), min(per_tier, len(pool))):
            out.append(Item("factoid", tier, f"What is the capital of {c}?",
                            tuple(pool[c]), float(tier), -1))
    for c in rng.sample(_FIC, per_tier):                       # tier 3: impossible w/o retrieval
        out.append(Item("factoid", 3, f"What is the capital of {c}?", (), 3.0, -1))
    return out


_ARITH_TIERS = [(1, 1), (2, 2), (3, 2), (5, 5)]                # (digits a, digits b)


def _arithmetic(rng, per_tier):
    out = []
    for tier, (da, db) in enumerate(_ARITH_TIERS):
        for _ in range(per_tier):
            a = rng.randint(10 ** (da - 1), 10 ** da - 1) if da > 1 else rng.randint(2, 9)
            b = rng.randint(10 ** (db - 1), 10 ** db - 1) if db > 1 else rng.randint(2, 9)
            out.append(Item("arithmetic", tier, f"What is {a} times {b}?", (str(a * b),),
                            float(da * db), carries_mul(a, b), {"a": a, "b": b, "da": da, "db": db}))
    return out


_REASON_STEPS = [1, 2, 3, 5]


def _reasoning(rng, per_tier):
    """Chained integer operations; difficulty = number of inference steps."""
    out = []
    for tier, k in enumerate(_REASON_STEPS):
        for _ in range(per_tier):
            v = rng.randint(3, 12)
            parts = [f"Start with {v}."]
            for _ in range(k):
                choice = rng.choice(["mul", "add", "sub", "div"])
                if choice == "div":
                    d = rng.choice([d for d in (2, 3, 4) if v % d == 0] or [1])
                    if d == 1:
                        choice = "add"
                    else:
                        v //= d; parts.append(f"Divide by {d}.")
                if choice == "mul":
                    n = rng.randint(2, 9); v *= n; parts.append(f"Multiply by {n}.")
                elif choice == "add":
                    n = rng.randint(2, 30); v += n; parts.append(f"Add {n}.")
                elif choice == "sub":
                    n = rng.randint(1, max(1, v - 1)); v -= n; parts.append(f"Subtract {n}.")
            out.append(Item("reasoning", tier, " ".join(parts) + " What is the result?",
                            (str(v),), float(k), -1, {"steps": k}))
    return out


def make_benchmark(seed: int, per_tier: int = 6):
    rng = random.Random(seed)
    items = _factoid(rng, per_tier) + _arithmetic(rng, per_tier) + _reasoning(rng, per_tier)
    rng.shuffle(items)
    return items


def graded(item: Item, output: str) -> bool:
    import re
    if not item.accepted:
        return False
    if item.domain == "factoid":
        o = output.strip().lower()
        return any(a.lower() in o for a in item.accepted)
    nums = re.findall(r"-?\d+", output.replace(",", ""))
    return bool(nums) and nums[-1] == item.accepted[0]
