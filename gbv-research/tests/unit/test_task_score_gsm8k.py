"""task_score must work for gsm8k_eval (held-out pool), not only the legacy gsm8k name."""

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
