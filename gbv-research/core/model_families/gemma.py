"""
Gemma model-family stub.

Covers (future):
  - google/gemma-2-2b, google/gemma-2-9b  (draft / target candidates)

Status: STUB — implement and test before use.
See docs/ADDING_A_MODEL_FAMILY.md for the checklist.

Key differences from Qwen to verify before enabling:
  1. Temperature: Gemma's generate() does NOT pre-divide logits by temperature
     in standard HuggingFace transformers (confirm with your transformers version).
  2. Forbidden tokens: Gemma does not use -inf logit bias for pad tokens in
     the same way as Qwen3.  clamp_log_probs() may be a no-op here.
  3. LoRA target modules: Gemma 2 uses q_proj, k_proj, v_proj, o_proj,
     gate_proj, up_proj, down_proj — but verify against model config.
  4. Chat template: use tokenizer.apply_chat_template (same API as Qwen).
  5. Vocabulary: Gemma 2 has vocab_size=256000, different from Qwen (151936).
     Speculative decoding requires draft and target to share the same vocab.
     Use Gemma 2 draft with Gemma 2 target, NOT cross-family.
"""

from __future__ import annotations

from typing import List

import torch

from .base import ModelFamily


class GemmaFamily(ModelFamily):
    """Google Gemma 2 model family (stub — not yet validated)."""

    @property
    def name(self) -> str:
        return "gemma"

    # ── LoRA ─────────────────────────────────────────────────────────────────

    def lora_target_modules(self) -> List[str]:
        # Gemma 2 architecture module names (verify against model.named_modules())
        return ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]

    # ── Temperature handling ─────────────────────────────────────────────────

    def recover_raw_logits(self, output_scores: torch.Tensor,
                           temperature: float) -> torch.Tensor:
        """
        TODO: Verify whether Gemma's generate() pre-divides by temperature.
        Current assumption: it does NOT (unlike Qwen3), so return as-is.
        To verify:  compare output_scores * temperature vs output_scores
        in a controlled run and check that the resulting distribution matches
        calling model.forward() directly.
        """
        # If Gemma does NOT pre-divide: return output_scores (identity)
        # If Gemma DOES pre-divide:     return output_scores * temperature
        return output_scores  # TODO: confirm

    # ── Forbidden token masking ───────────────────────────────────────────────

    def clamp_log_probs(self, log_probs: torch.Tensor) -> torch.Tensor:
        """
        TODO: Check if Gemma applies -inf bias to any tokens.
        Conservative: apply the same clamp as Qwen for safety.
        Remove if confirmed unnecessary (adds a small constant overhead).
        """
        return log_probs.clamp(min=-100.0)

    # ── Chat template ─────────────────────────────────────────────────────────

    def format_prompt(self, prompt: str, tokenizer) -> str:
        """Gemma 2 chat template (same apply_chat_template API as Qwen)."""
        try:
            messages = [{"role": "user", "content": prompt}]
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return prompt

    # ── Default model IDs ─────────────────────────────────────────────────────

    @property
    def default_draft_model_id(self) -> str:
        return "google/gemma-2-2b"

    @property
    def default_target_model_id(self) -> str:
        return "google/gemma-2-9b"


# ── Registry entry ────────────────────────────────────────────────────────────
FAMILY = GemmaFamily()
