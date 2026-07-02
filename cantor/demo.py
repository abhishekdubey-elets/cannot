"""End-to-end demo of the CANTOR reference implementation.

Walks the four scenarios in README.md:
  1. easy query      -> System 1 (cheap)
  2. hard arithmetic -> System 2: plan + verifier-grounded critic
  3. unknown query   -> honest deferral
  4. sleep, then the same hard query -> answered from the consolidated
     semantic cache below System-1 cost (the search->distill flywheel)
"""

from cantor import CantorAgent, MockLLM


def show(label: str, result) -> None:
    print(f"\n=== {label} ===")
    print(f"  route      : {result.route}")
    print(f"  answer     : {result.answer if not result.deferred else '(deferred - uncertainty reported)'}")
    print(f"  confidence : {result.confidence.score}  (consistency={result.confidence.consistency}, "
          f"epistemic={result.confidence.epistemic}, verified={result.confidence.verified})")
    print(f"  cost units : {result.cost_units}")
    for line in result.trace:
        print(f"    - {line}")


def main() -> None:
    agent = CantorAgent(MockLLM())

    # 1. Easy: known fact -> fast path accepts.
    show("1. easy (System 1)", agent.respond("What is the capital of France?"))

    # 2. Hard: fast path samples disagree -> controller escalates -> planner
    #    decomposes -> critic verifies against the grounded arithmetic engine.
    hard = "What is 37 * 24?"
    show("2. hard (System 2)", agent.respond(hard, stakes=0.5))

    # 3. Unknown: low confidence, high stakes, nothing to ground -> defer.
    show("3. unknown (defer)", agent.respond(
        "What was the exact population of Ur in 2000 BCE?", stakes=0.8))

    # 4. Sleep: consolidate episodes -> semantic facts + procedural skill.
    report = agent.sleep()
    print(f"\n=== consolidation ('sleep') ===")
    print(f"  replayed={report.replayed}  facts={report.facts_extracted}  "
          f"skills={report.skills_distilled}  pruned={report.episodes_pruned}")
    for fact in agent.semantic.facts:
        print(f"    semantic: {fact.render()}")

    # Same hard query again: now a cache hit at ~0.5 cost units instead of ~10.
    show("4. same hard query after sleep (semantic cache)", agent.respond(hard, stakes=0.5))


if __name__ == "__main__":
    main()
