"""Unit tests for extraction/grading — hand-written cases for all four answer types."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ugen.pipeline import grading as G


def test_mc():
    opts = ["Paris", "London", "Berlin", "Madrid"]
    assert G.extract_mc("blah blah\nFinal answer: A", opts) == ("A", True)
    assert G.extract_mc("Final answer: (C)", opts) == ("C", True)
    assert G.extract_mc("Final answer: B. London", opts) == ("B", True)
    assert G.extract_mc("I think the answer is Paris", opts)[0] == "A"      # text fallback
    assert G.extract_mc("The answer is definitely berlin.", opts)[0] == "C"
    assert G.extract_mc("Final answer: Z", opts) == (None, False)           # invalid letter
    assert G.extract_mc("**Final answer: D**", opts)[0] == "D"
    # option letter beyond the option count is rejected
    assert G.extract_mc("Final answer: E", opts) == (None, False)


def test_numeric():
    assert G.extract_number("So the total is 42.\nFinal answer: 42") == (42.0, True)
    assert G.extract_number("Final answer: 1,234.5") == (1234.5, True)
    assert G.extract_number("Final answer: $18") == (18.0, True)
    assert G.extract_number("the result is -7") == (-7.0, False)            # fallback path
    assert G.extract_number("no numbers here") == (None, False)
    assert G.numbers_equal(0.3333, 1 / 3, rel=1e-3)
    assert not G.numbers_equal(42, 43)


def test_yesno():
    assert G.extract_yesno("Final answer: Yes") == ("yes", True)
    assert G.extract_yesno("Final answer: FALSE") == ("no", True)
    assert G.extract_yesno("I believe the answer is no.") == ("no", False)


def test_code_humaneval():
    item = dict(answer_type="code", gold="", options=None, grade_spec=dict(
        kind="humaneval", entry_point="add",
        test="def check(candidate):\n    assert candidate(1, 2) == 3\n    assert candidate(-1, 1) == 0\n"))
    good = "Here is my solution:\n```python\ndef add(a, b):\n    return a + b\n```"
    bad = "```python\ndef add(a, b):\n    return a - b\n```"
    r = G.grade(good, item)
    assert r["correct"] == 1 and r["extraction_ok"] == 1
    assert G.grade(bad, item)["correct"] == 0
    assert G.grade("def add(a, b): return a + b", item)["correct"] == 1    # unfenced fallback


def test_code_mbpp():
    item = dict(answer_type="code", gold="", options=None, grade_spec=dict(
        kind="mbpp", tests=["assert double(3) == 6", "assert double(0) == 0"], imports=[]))
    assert G.grade("```python\ndef double(x):\n    return 2*x\n```", item)["correct"] == 1
    assert G.grade("```python\ndef double(x):\n    return x\n```", item)["correct"] == 0
    # infinite loop -> timeout -> wrong, not a crash
    r = G.grade("```python\ndef double(x):\n    while True: pass\n```", item, timeout=3)
    assert r["correct"] == 0 and r["detail"] == "timeout"


def test_grade_mc_numeric():
    item = dict(answer_type="mc_letter", gold="B", options=["x", "y", "z"], grade_spec=None)
    assert G.grade("Final answer: B", item)["correct"] == 1
    assert G.grade("Final answer: A", item)["correct"] == 0
    assert G.grade("gibberish with no answer", item)["correct"] == 0        # unparseable=wrong
    item2 = dict(answer_type="numeric", gold=72.0, options=None, grade_spec=None)
    assert G.grade("Reasoning...\nFinal answer: 72", item2)["correct"] == 1


def test_normalize():
    item = dict(answer_type="numeric", options=None)
    assert G.normalize_answer("Final answer: 42.0", item) == "42"
    assert G.normalize_answer("Final answer: 42", item) == "42"             # 42.0 == 42 vote
    assert G.normalize_answer("???", item) == "__NONE__"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"grading tests: {len(fns)}/{len(fns)} PASSED")
