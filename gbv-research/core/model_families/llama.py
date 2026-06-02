"""
LLaMA model-family implementation.

Covers:
  - meta-llama/Llama-3.2-1B-Instruct  (1B)  — draft model
  - meta-llama/Llama-3.2-3B-Instruct  (3B)  — target model

Both use the same LLaMA 3 tokenizer (vocab_size = 128256), which satisfies
the shared-vocabulary requirement for speculative decoding.

Hardware notes for the laptop (RTX 500 Ada, 6 GB VRAM):
  1B draft  in BF16  ≈ 2 GB
  3B target in NF4   ≈ 1.5 GB   (set load_in_4bit: true in the YAML config)
  LoRA + activations ≈ 1-2 GB
  Total              ≈ 4.5-5.5 GB → comfortable 6 GB fit

LLaMA 3.2 quirks vs Qwen:
  1. Temperature: generate(output_scores=True) returns raw logits — no
     temperature pre-scaling.  recover_raw_logits() is an identity.
  2. Forbidden tokens: no -inf logit bias applied.  clamp_log_probs() is
     a conservative no-op clamp identical to GPT-2 / Gemma.
  3. Chat template: LLaMA 3 Instruct uses a header/eot format automatically
     handled by tokenizer.apply_chat_template().
  4. LoRA modules: standard Llama attention projections (q, k, v, o).
     Already present in training_scaffold._LORA_MODULES["llama"].

HuggingFace access: Llama-3.2 is gated.  Accept the licence at
  https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
then run:  huggingface-cli login
"""

from __future__ import annotations

from typing import List

import torch

from .base import ModelFamily


class LlamaFamily(ModelFamily):
    """Meta LLaMA 3.2 model family (1B draft / 3B target)."""

    @property
    def name(self) -> str:
        return "llama"

    # ── LoRA ─────────────────────────────────────────────────────────────────

    def lora_target_modules(self) -> List[str]:
        # Standard LLaMA attention projections.  MLP projections excluded to
        # keep the adapter small; add gate_proj/up_proj/down_proj if capacity
        # becomes the bottleneck on longer experiments.
        return ["q_proj", "k_proj", "v_proj", "o_proj"]

    # ── Temperature handling ─────────────────────────────────────────────────

    def recover_raw_logits(self, output_scores: torch.Tensor,
                           temperature: float) -> torch.Tensor:
        """
        LLaMA 3.2's generate() does NOT pre-divide by temperature —
        output_scores are raw logits.  Return as-is.
        """
        return output_scores

    # ── Forbidden token masking ───────────────────────────────────────────────

    def clamp_log_probs(self, log_probs: torch.Tensor) -> torch.Tensor:
        """
        LLaMA 3.2 does not apply -inf logit bias to any tokens, so no NaN
        hazard in reverse-KL.  Conservative clamp kept for safety.
        """
        return log_probs.clamp(min=-100.0)

    # ── Chat template ─────────────────────────────────────────────────────────

    def format_prompt(self, prompt: str, tokenizer) -> str:
        """
        Apply the LLaMA 3 Instruct chat template via HuggingFace's
        apply_chat_template.  Falls back to raw prompt if the tokenizer
        doesn't have a template (e.g. base/non-instruct variants).
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
        # 1B Instruct, ~2 GB BF16 — fits comfortably on 6 GB VRAM
        return "meta-llama/Llama-3.2-1B-Instruct"

    @property
    def default_target_model_id(self) -> str:
        # 3B Instruct, ~1.5 GB in NF4 4-bit (use load_in_4bit: true in config)
        return "meta-llama/Llama-3.2-3B-Instruct"


# ── Registry entry ────────────────────────────────────────────────────────────
FAMILY = LlamaFamily()
