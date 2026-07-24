"""DIVERGENCE benchmark — built so confidence estimators DISAGREE with correctness.

The estimator comparison saturated (numeric-confidence, log-prob and consistency all
reached balanced accuracy 1.000), so it can no longer separate them. Saturation happened
because the workload was binary: the model either knew a real capital or faced an invented
one. Real metacognition is not binary, and the interesting cases are the ones where a
confidence signal is CONFIDENTLY WRONG or UNCONFIDENTLY RIGHT.

Four categories, chosen because each is cleanly gradeable AND stresses estimators in a
different direction:

  common     unambiguous well-known capital        -> correct, all estimators high (control)
  ambiguous  country with SEVERAL valid capitals   -> CORRECT but samples disagree
                                                      => consistency should FALSE-ALARM
                                                         (low consistency, right answer)
  outdated   capital that MOVED recently           -> model gives the old one: WRONG while
                                                      confident and self-consistent
                                                      => the dangerous failure: every
                                                         estimator agrees and is wrong
  fictional  invented country                      -> wrong without retrieval, estimators low

`ambiguous` and `outdated` are the two divergence directions. If estimators still rank-order
correctness perfectly here, that is strong evidence; if they split, we learn WHICH signal
survives WHICH failure mode — which is what an executive controller actually needs.

SCOPE: still capitals-only, for gradeability. It does not cover multi-hop, false-premise,
or underspecified queries; those need free-form grading and are left to future work.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

CATEGORIES = ["common", "ambiguous", "outdated", "fictional"]

COMMON = {"France": ["Paris"], "Japan": ["Tokyo"], "Italy": ["Rome"], "Spain": ["Madrid"],
          "Germany": ["Berlin"], "Egypt": ["Cairo"], "Canada": ["Ottawa"],
          "Russia": ["Moscow"], "China": ["Beijing"], "Greece": ["Athens"],
          "Portugal": ["Lisbon"], "Austria": ["Vienna"]}

# several genuinely valid answers -> a sampler legitimately returns different ones
AMBIGUOUS = {
    "South Africa": ["Pretoria", "Cape Town", "Bloemfontein"],
    "Bolivia": ["Sucre", "La Paz"],
    "the Netherlands": ["Amsterdam", "The Hague"],
    "Sri Lanka": ["Colombo", "Sri Jayawardenepura Kotte", "Kotte"],
    "Malaysia": ["Kuala Lumpur", "Putrajaya"],
    "Eswatini": ["Mbabane", "Lobamba"],
    "Benin": ["Porto-Novo", "Cotonou"],
    "Tanzania": ["Dodoma", "Dar es Salaam"],
}

# capital relocated; a model trained on older text confidently emits the OLD one.
# `answer` = the CURRENT capital (what counts as correct); `stale` = the likely error.
OUTDATED = {
    "Burundi": (["Gitega"], "Bujumbura"),
    "Myanmar": (["Naypyidaw", "Nay Pyi Taw"], "Yangon"),
    "Nigeria": (["Abuja"], "Lagos"),
    "Kazakhstan": (["Astana", "Nur-Sultan"], "Almaty"),
    "Palau": (["Ngerulmud"], "Koror"),
    "Ivory Coast": (["Yamoussoukro"], "Abidjan"),
}

_PRE = ["Zeml", "Qorv", "Elbo", "Draz", "Vulp", "Noor", "Sylth", "Grenn",
        "Marov", "Tavis", "Bruv", "Kelm", "Ordr", "Pyre", "Xanth", "Wext"]
_SUF = ["ania", "oria", "mir", "arn", "ova", "nik", "tania", "enne"]
FICTIONAL = [p + s for p in _PRE for s in _SUF]


@dataclass(frozen=True)
class DQuery:
    category: str
    prompt: str
    accepted: tuple           # any of these counts as correct
    stale: str = ""           # the expected-but-wrong answer, for `outdated`
    meta: dict = field(default_factory=dict)


def make_divergence(seed: int, n_per_cat: int):
    rng = random.Random(seed)
    qs: list[DQuery] = []

    def q(country, accepted, cat, stale=""):
        return DQuery(cat, f"What is the capital of {country}?", tuple(accepted), stale)

    for c in rng.sample(list(COMMON), min(n_per_cat, len(COMMON))):
        qs.append(q(c, COMMON[c], "common"))
    for c in rng.sample(list(AMBIGUOUS), min(n_per_cat, len(AMBIGUOUS))):
        qs.append(q(c, AMBIGUOUS[c], "ambiguous"))
    for c in rng.sample(list(OUTDATED), min(n_per_cat, len(OUTDATED))):
        acc, stale = OUTDATED[c]
        qs.append(q(c, acc, "outdated", stale))
    for c in rng.sample(FICTIONAL, n_per_cat):
        qs.append(DQuery("fictional", f"What is the capital of {c}?", (), "", {"entity": c}))

    rng.shuffle(qs)
    return qs


def graded(dq: DQuery, out: str) -> bool:
    """Correct iff any accepted answer appears. Fictional has none -> always incorrect
    (the model cannot know it; only retrieval could supply it)."""
    o = out.strip().lower()
    return any(a.lower() in o for a in dq.accepted)


def gave_stale(dq: DQuery, out: str) -> bool:
    return bool(dq.stale) and dq.stale.lower() in out.strip().lower()


# ------------------------------------------------------------------ metrics
def auroc(scores, labels):
    """P(score(correct) > score(incorrect)); 0.5 = uninformative, <0.5 = anti-correlated."""
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return float("nan")
    wins = sum((1.0 if p > n else 0.5 if p == n else 0.0) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))
