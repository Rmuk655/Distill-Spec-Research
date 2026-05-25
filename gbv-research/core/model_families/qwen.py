"""
Qwen model-family implementation.

Covers:
  - Qwen2.5-0.5B / Qwen2.5-1.5B / Qwen2.5-7B  (draft candidates)
  - Qwen3-0.6B / Qwen3-8B                       (target candidates)

Known Qwen3 quirks handled here:
  1. generate(output_scores=True) returns logits / temperature, not raw logits.
     We multiply by temperature to recover raw logits before applying our own.
  2. Qwen3 applies -inf logit bias to forbidden / control tokens (thinking
     tokens, pad tokens) when thinking_mode is disabled.  After log_softmax
     those slots have log_prob = -inf, which causes NaN in reverse-KL training
     via  p_student * (log_s - (-inf)) = p_student * +inf.
     Fix: clamp log_probs to min=-100 before any subtraction.
"""

from __future__ import annotations

from typing import List

import torch

from .base import ModelFamily


class QwenFamily(ModelFamily):
    """Qwen2.5 / Qwen3 model family."""

    @property
    def name(self) -> str:
        return "qwen"

    # ── LoRA ─────────────────────────────────────────────────────────────────

    def lora_target_modules(self) -> List[str]:
        # Qwen2.5 / Qwen3 share the same module names.
        # Includes both attention projections and MLP projections for best
        # coverage at minimal trainable-parameter overhead.
        return ["q_proj", "k_proj", "v_proj", "o_proj",
                "up_proj", "gate_proj", "down_proj"]

    # ── Temperature handling ─────────────────────────────────────────────────

    def recover_raw_logits(self, output_scores: torch.Tensor,
                           temperature: float) -> torch.Tensor:
        """
        Qwen3's generate() pre-divides logits by sampling temperature before
        returning output_scores.  Multiply by temperature to undo this so we
        get the raw logits that can be re-scaled consistently during training.

        Reference: transformers/models/qwen3/modeling_qwen3.py – the logits
        processor applies temperature scaling before the scores are stored.
        """
        return output_scores * temperature

    # ── Forbidden token masking ───────────────────────────────────────────────

    def clamp_log_probs(self, log_probs: torch.Tensor) -> torch.Tensor:
        """
        Qwen3 assigns -inf logit bias to forbidden tokens (pad, thinking control
        tokens) when thinking is disabled.  After log_softmax these become -inf.
        In reverse-KL:  p_s * (log_s - log_t) = p_s * (log_s - (-inf)) = NaN.
        Clamp to -100 (prob ≈ exp(-100) ≈ 0) so the difference is bounded and
        gradients flow cleanly.
        """
        return log_probs.clamp(min=-100.0)

    # ── Chat template ─────────────────────────────────────────────────────────

    def format_prompt(self, prompt: str, tokenizer) -> str:
        """
        Apply Qwen3 chat template (user / assistant turns).
        Falls back to raw prompt if the tokenizer has no chat template.
        """
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
        # Laptop (4 GB VRAM): Qwen2.5-0.5B draft → Qwen3-0.6B target
        return "Qwen/Qwen2.5-0.5B"

    @property
    def default_target_model_id(self) -> str:
        return "Qwen/Qwen3-0.6B"


# ── Registry entry ────────────────────────────────────────────────────────────
# Imported by capsules/distillation/model_families/__init__.py
FAMILY = QwenFamily()
