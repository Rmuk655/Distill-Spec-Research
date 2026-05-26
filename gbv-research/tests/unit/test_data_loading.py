"""
test_data_loading.py — unit tests for dataset loading utilities.

Functions under test:
  - GBV/util.py  : load_prompts_jsonl(path)
  - gbv-research/algorithms/train_qwen3.py : load_prompts(dataset_path)

Tests verify:
  1. Correct prompts are loaded from {"prompt": ...} records
  2. Empty lines are silently skipped
  3. Records missing the "prompt" key are silently skipped
  4. JSONL with alternate keys ("question", "instruction") handled by train_qwen3
  5. Empty dataset raises ValueError (train_qwen3.load_prompts)
  6. No prompt if the file has only whitespace / comment-like lines
  7. Unicode prompts handled correctly (math / code prompts)

All tests use tmp_path (pytest fixture) — no disk state persists between tests.
"""

import json
import os
import sys
import pytest

from util import load_prompts_jsonl
import train_qwen3 as _t

load_prompts = _t.load_prompts


# ===========================================================================
# load_prompts_jsonl  (GBV/util.py)
# ===========================================================================

class TestLoadPromptsJsonl:

    def test_basic_prompt_loading(self, tmp_path):
        p = tmp_path / "test.jsonl"
        records = [{"prompt": "Hello world"}, {"prompt": "Explain AI."}]
        p.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        prompts = load_prompts_jsonl(str(p))
        assert prompts == ["Hello world", "Explain AI."]

    def test_empty_lines_skipped(self, tmp_path):
        p = tmp_path / "test.jsonl"
        p.write_text('{"prompt": "A"}\n\n   \n{"prompt": "B"}\n')
        prompts = load_prompts_jsonl(str(p))
        assert prompts == ["A", "B"]

    def test_missing_key_skipped(self, tmp_path):
        p = tmp_path / "test.jsonl"
        records = [
            {"prompt": "Good prompt"},
            {"question": "No prompt key here"},   # missing "prompt" key
            {"other": "irrelevant"},
            {"prompt": "Also good"},
        ]
        p.write_text("\n".join(json.dumps(r) for r in records))
        prompts = load_prompts_jsonl(str(p))
        # Only records with "prompt" key should be returned
        assert "Good prompt" in prompts
        assert "Also good" in prompts
        # Records without "prompt" must be silently skipped (no exception)
        assert len(prompts) == 2

    def test_empty_file_returns_empty_list(self, tmp_path):
        p = tmp_path / "empty.jsonl"
        p.write_text("")
        prompts = load_prompts_jsonl(str(p))
        assert prompts == []

    def test_unicode_prompts(self, tmp_path):
        p = tmp_path / "unicode.jsonl"
        records = [
            {"prompt": "What is ∑_{i=1}^{n} i²?"},
            {"prompt": "def fib(n: int) → int: ..."},
            {"prompt": "解释量子纠缠。"},   # Chinese
        ]
        p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records),
                     encoding="utf-8")
        prompts = load_prompts_jsonl(str(p))
        assert len(prompts) == 3
        assert "∑" in prompts[0]

    def test_preserves_order(self, tmp_path):
        p = tmp_path / "ordered.jsonl"
        records = [{"prompt": str(i)} for i in range(10)]
        p.write_text("\n".join(json.dumps(r) for r in records))
        prompts = load_prompts_jsonl(str(p))
        assert prompts == [str(i) for i in range(10)]

    def test_returns_list_type(self, tmp_path):
        p = tmp_path / "single.jsonl"
        p.write_text('{"prompt": "only one"}')
        result = load_prompts_jsonl(str(p))
        assert isinstance(result, list)


# ===========================================================================
# load_prompts  (OSD/train_qwen3.py)
# ===========================================================================

class TestLoadPromptsTrainQwen3:

    def test_prompt_key(self, tmp_path):
        p = tmp_path / "data.jsonl"
        p.write_text('{"prompt": "What is Euler\'s identity?"}\n')
        prompts = load_prompts(str(p))
        assert len(prompts) == 1
        assert "Euler" in prompts[0]

    def test_question_key_fallback(self, tmp_path):
        """train_qwen3.load_prompts should also accept 'question' key."""
        p = tmp_path / "data.jsonl"
        p.write_text('{"question": "What is entropy?"}\n')
        prompts = load_prompts(str(p))
        assert len(prompts) == 1
        assert "entropy" in prompts[0]

    def test_instruction_key_fallback(self, tmp_path):
        """train_qwen3.load_prompts should also accept 'instruction' key."""
        p = tmp_path / "data.jsonl"
        p.write_text('{"instruction": "Summarise this paragraph."}\n')
        prompts = load_prompts(str(p))
        assert len(prompts) == 1

    def test_empty_prompt_string_skipped(self, tmp_path):
        """A record with prompt='' should not appear in output."""
        p = tmp_path / "data.jsonl"
        p.write_text('{"prompt": ""}\n{"prompt": "real prompt"}\n')
        prompts = load_prompts(str(p))
        assert "real prompt" in prompts
        assert "" not in prompts

    def test_empty_file_raises(self, tmp_path):
        """Empty dataset should raise ValueError (caught before training starts)."""
        p = tmp_path / "empty.jsonl"
        p.write_text("")
        with pytest.raises(ValueError, match="No prompts"):
            load_prompts(str(p))

    def test_none_path_returns_builtin_prompts(self):
        """load_prompts(None) should return the built-in PROMPTS list."""
        prompts = load_prompts(None)
        assert isinstance(prompts, list)
        assert len(prompts) > 0

    def test_returns_stripped_strings(self, tmp_path):
        """Prompts should be stripped of leading/trailing whitespace."""
        p = tmp_path / "data.jsonl"
        p.write_text('{"prompt": "  padded  "}\n')
        prompts = load_prompts(str(p))
        assert prompts[0] == "padded", f"Expected 'padded', got {prompts[0]!r}"
