"""task_score must work for gsm8k_eval, math500, and humaneval dataset names."""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ORCH = os.path.join(_HERE, "..", "..", "orchestration")
sys.path.insert(0, _ORCH)

import evaluate as _ev


def test_gsm8k_eval_is_scored_dataset():
    assert _ev._is_gsm8k_scored_dataset("gsm8k_eval")
    assert _ev._is_gsm8k_scored_dataset("gsm8k")
    assert not _ev._is_gsm8k_scored_dataset("humaneval")


def test_score_gsm8k_matches_final_number():
    gold = "Janet sells 16 - 4 = 12 eggs. #### 12"
    assert _ev.score_gsm8k("The answer is 12.", gold) == 1.0
    assert _ev.score_gsm8k("The answer is 13.", gold) == 0.0


def test_humaneval_not_gsm8k_scored():
    assert not _ev._is_gsm8k_scored_dataset("math500")


def test_math500_is_task_scored_dataset():
    assert _ev._is_math500_scored_dataset("math500")
    assert _ev._is_math500_scored_dataset("math500_30")
    assert _ev._is_task_scored_dataset("math500")
    assert not _ev._is_math500_scored_dataset("gsm8k_eval")


def test_extract_last_boxed_nested_braces():
    text = r"We get \boxed{\frac{1}{2}} and also \boxed{42}."
    assert _ev._extract_last_boxed(text) == "42"


def test_score_math500_boxed_match():
    gold = r"Therefore the answer is \boxed{\frac{3}{4}}."
    pred = r"Final answer: \boxed{\frac{3}{4}}"
    assert _ev.score_math500(pred, gold) == 1.0


def test_score_math500_boxed_mismatch():
    gold = r"\boxed{7}"
    pred = r"\boxed{8}"
    assert _ev.score_math500(pred, gold) == 0.0


def test_score_math500_numeric_fallback():
    gold = r"\boxed{12}"
    pred = "The result is 12."
    assert _ev.score_math500(pred, gold) == 1.0
