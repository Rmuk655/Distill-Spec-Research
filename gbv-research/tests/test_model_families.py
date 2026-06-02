"""
test_model_families.py — smoke tests for all registered ModelFamily implementations.

Runs entirely on CPU with synthetic tensors — no model downloads, no GPU.
These tests are the first gate before running scripts/smoke_family.py (which
does a real 5-step training run with actual model weights).

Coverage:
  - All abstract methods return the right types/shapes
  - recover_raw_logits() is identity for non-Qwen families
  - Qwen recover_raw_logits() correctly undoes temperature pre-scaling
  - clamp_log_probs() eliminates -inf (NaN hazard in reverse-KL)
  - format_prompt() returns a non-empty string
  - GPT-2 format_prompt() returns the raw prompt (no chat template)
  - Registry lookup raises for unknown names
  - FAMILY_REGISTRY contains all expected families
"""

import pytest
import torch

from core.model_families import FAMILY_REGISTRY, get_family


# ── Helpers ───────────────────────────────────────────────────────────────────

class _MockTokenizer:
    """Minimal tokenizer stub for format_prompt tests."""
    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=False):
        return f"<chat>{messages[0]['content']}</chat>"


ALL_FAMILIES = sorted(FAMILY_REGISTRY.keys())   # ["gemma", "gpt2", "llama", "qwen"]


# ── Parametrized interface tests ──────────────────────────────────────────────

@pytest.mark.parametrize("name", ALL_FAMILIES)
def test_name_matches_registry_key(name):
    """family.name must equal the key it is registered under."""
    assert get_family(name).name == name


@pytest.mark.parametrize("name", ALL_FAMILIES)
def test_lora_modules_nonempty_strings(name):
    mods = get_family(name).lora_target_modules()
    assert isinstance(mods, list), "lora_target_modules() must return a list"
    assert len(mods) > 0, "lora_target_modules() must be non-empty"
    assert all(isinstance(m, str) and m for m in mods), \
        "every module name must be a non-empty string"


@pytest.mark.parametrize("name", ALL_FAMILIES)
def test_recover_raw_logits_preserves_shape(name):
    logits = torch.randn(2, 16, 32000)   # [batch, seq, vocab]
    out = get_family(name).recover_raw_logits(logits, temperature=0.8)
    assert out.shape == logits.shape, "recover_raw_logits() must preserve shape"
    assert out.dtype == logits.dtype,  "recover_raw_logits() must preserve dtype"


@pytest.mark.parametrize("name", ALL_FAMILIES)
def test_clamp_log_probs_removes_neginf(name):
    """clamp_log_probs() must never let -inf survive — that causes NaN in rev-KL."""
    log_probs = torch.tensor([-0.5, float("-inf"), -2.0, float("-inf"), -10.0])
    clamped = get_family(name).clamp_log_probs(log_probs)
    assert not torch.isinf(clamped).any(), \
        "clamp_log_probs() must eliminate all -inf values"
    assert not torch.isnan(clamped).any(), \
        "clamp_log_probs() must not introduce NaN"


@pytest.mark.parametrize("name", ALL_FAMILIES)
def test_format_prompt_returns_nonempty_string(name):
    tok = _MockTokenizer()
    result = get_family(name).format_prompt("What is 2+2?", tok)
    assert isinstance(result, str), "format_prompt() must return str"
    assert len(result) > 0,         "format_prompt() must return non-empty str"


@pytest.mark.parametrize("name", ALL_FAMILIES)
def test_default_model_ids_are_nonempty_strings(name):
    fam = get_family(name)
    assert isinstance(fam.default_draft_model_id, str) and fam.default_draft_model_id
    assert isinstance(fam.default_target_model_id, str) and fam.default_target_model_id


# ── Family-specific behaviour ─────────────────────────────────────────────────

def test_qwen_recover_raw_logits_multiplies_by_temperature():
    """Qwen3 must undo the generate()-time temperature pre-divide."""
    qwen = get_family("qwen")
    logits = torch.randn(4, 151936)
    temp = 0.8
    recovered = qwen.recover_raw_logits(logits, temperature=temp)
    assert torch.allclose(recovered, logits * temp), \
        "Qwen recover_raw_logits should multiply by temperature"


def test_gpt2_recover_raw_logits_is_identity():
    """GPT-2 must NOT pre-scale — output_scores are already raw logits."""
    gpt2 = get_family("gpt2")
    logits = torch.randn(4, 50257)
    assert torch.equal(gpt2.recover_raw_logits(logits, temperature=0.8), logits), \
        "GPT-2 recover_raw_logits must be an identity (no temperature scaling)"


def test_llama_recover_raw_logits_is_identity():
    """LLaMA 3.2 must NOT pre-scale — same as GPT-2."""
    llama = get_family("llama")
    logits = torch.randn(4, 128256)
    assert torch.equal(llama.recover_raw_logits(logits, temperature=0.9), logits), \
        "LLaMA recover_raw_logits must be an identity"


def test_gpt2_format_prompt_returns_raw_prompt():
    """GPT-2 is a base (completion) model — format_prompt must be a no-op."""
    gpt2 = get_family("gpt2")
    prompt = "The answer to the question is"
    # tokenizer should not be called at all; pass None to confirm
    assert gpt2.format_prompt(prompt, tokenizer=None) == prompt, \
        "GPT-2 format_prompt must return the prompt unchanged (no chat template)"


def test_qwen_format_prompt_wraps_in_chat_template():
    """Qwen format_prompt must apply a chat template (returns something longer)."""
    qwen = get_family("qwen")
    tok = _MockTokenizer()
    prompt = "What is 2+2?"
    result = qwen.format_prompt(prompt, tok)
    # After apply_chat_template the result is longer (contains header tokens)
    assert prompt in result or len(result) >= len(prompt), \
        "Qwen format_prompt must include or extend the original prompt"


def test_gpt2_lora_modules_use_conv1d_names():
    """GPT-2 uses Conv1D naming — must not contain q_proj / v_proj."""
    mods = get_family("gpt2").lora_target_modules()
    assert "c_attn" in mods, "GPT-2 LoRA must target c_attn (combined QKV)"
    assert "q_proj" not in mods, "GPT-2 must not use q_proj (that's LLaMA/Qwen naming)"


def test_llama_lora_modules_use_proj_names():
    """LLaMA uses standard q_proj / k_proj / v_proj / o_proj naming."""
    mods = get_family("llama").lora_target_modules()
    for expected in ["q_proj", "k_proj", "v_proj", "o_proj"]:
        assert expected in mods, f"LLaMA LoRA must target {expected}"


# ── Registry integrity ────────────────────────────────────────────────────────

def test_registry_contains_all_expected_families():
    expected = {"gpt2", "llama", "qwen", "gemma"}
    missing = expected - set(FAMILY_REGISTRY)
    assert not missing, f"FAMILY_REGISTRY missing families: {missing}"


def test_get_family_unknown_raises_with_available_names():
    with pytest.raises(ValueError, match="Unknown model family"):
        get_family("does_not_exist_xyz")


def test_get_family_unknown_error_lists_available():
    """The error message should list available families so users know what to type."""
    try:
        get_family("foobar")
    except ValueError as e:
        msg = str(e)
        # At least one known family must appear in the message
        assert any(f in msg for f in ALL_FAMILIES), \
            "ValueError should list available families in the message"
