"""
GPT-2 model-family implementation.

Covers:
  - distilgpt2  (82 M)   — draft model
  - gpt2-medium (355 M)  — target model

Both share the same BPE tokenizer (vocab_size = 50257), which is the
hard requirement for speculative decoding.  They run comfortably on CPU,
making this family ideal for proving convergence trends on a laptop before
scaling up to cloud GPUs with Qwen or LLaMA.

GPT-2 quirks vs Qwen:
  1. Temperature: generate(output_scores=True) returns raw logits — no
     temperature pre-scaling.  recover_raw_logits() is an identity.
  2. Forbidden tokens: no -inf logit bias applied.  clamp_log_probs() is
     a conservative no-op clamp (same guard as Gemma).
  3. Chat template: GPT-2 is a base (completion) model with no chat
     template.  format_prompt() returns the prompt unchanged.
  4. LoRA modules: GPT-2 uses Conv1D with different naming vs Llama/Qwen.
     c_attn = combined QKV projection; c_proj = output + MLP output.
"""

from __future__ import annotations

from typing import List

import torch

from .base import ModelFamily


class GPT2Family(ModelFamily):
    """OpenAI GPT-2 model family (distilgpt2 draft / gpt2-medium target)."""

    @property
    def name(self) -> str:
        return "gpt2"

    # ── LoRA ─────────────────────────────────────────────────────────────────

    def lora_target_modules(self) -> List[str]:
        # GPT-2 Conv1D naming (NOT the same as Llama q_proj / v_proj).
        # c_attn: combined QKV projection in each attention block.
        # c_proj: output projection in attention + MLP projection.
        # Targeting both gives good coverage without blowing up param count.
        return ["c_attn", "c_proj"]

    # ── Temperature handling ─────────────────────────────────────────────────

    def recover_raw_logits(self, output_scores: torch.Tensor,
                           temperature: float) -> torch.Tensor:
        """
        GPT-2's generate() does NOT pre-divide by temperature — output_scores
        are already raw logits.  Return as-is.
        """
        return output_scores

    # ── Forbidden token masking ───────────────────────────────────────────────

    def clamp_log_probs(self, log_probs: torch.Tensor) -> torch.Tensor:
        """
        GPT-2 does not apply -inf logit bias to any tokens, so no NaN
        hazard in reverse-KL.  Conservative clamp kept for safety.
        """
        return log_probs.clamp(min=-100.0)

    # ── Chat template ─────────────────────────────────────────────────────────

    def format_prompt(self, prompt: str, tokenizer) -> str:
        """
        GPT-2 is a plain completion model — no chat template.
        Return the prompt as-is so the model continues it directly.
        """
        return prompt

    # ── Default model IDs ─────────────────────────────────────────────────────

    @property
    def default_draft_model_id(self) -> str:
        # 82 M, downloads ~330 MB, runs on CPU
        return "distilgpt2"

    @property
    def default_target_model_id(self) -> str:
        # 355 M, downloads ~1.4 GB, runs on CPU (slow but feasible)
        return "gpt2-medium"


# ── Registry entry ────────────────────────────────────────────────────────────
FAMILY = GPT2Family()
