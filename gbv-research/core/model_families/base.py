"""
Model-family abstraction layer.

Each model family (Qwen, Gemma, LLaMA, …) has subtly different behaviours
that affect distillation:

  1. LoRA target modules differ per architecture.
  2. HuggingFace generate() may pre-divide logits by temperature before
     returning output_scores (Qwen3 does this; Gemma does not).
  3. Some models apply -inf logit bias to forbidden / padding tokens
     (Qwen3 does this for control tokens).  In reverse-KL training this
     causes NaN — the family is responsible for masking them safely.
  4. Chat templates differ.

Adding a new family:
  1. Create capsules/distillation/model_families/<family_name>.py
  2. Subclass ModelFamily, implement all @abstractmethod methods.
  3. Register it: FAMILY_REGISTRY["my_family"] = MyFamily()
  4. Pass --model_family my_family to trainer.py.

See ADDING_A_MODEL_FAMILY.md for a step-by-step walkthrough.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import torch


class ModelFamily(ABC):
    """
    Abstract specification of model-family-specific behaviour.
    All methods are pure (no side effects) unless noted.
    """

    # ── Identity ─────────────────────────────────────────────────────────────

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in CLI args and registry keys, e.g. 'qwen'."""

    # ── LoRA configuration ───────────────────────────────────────────────────

    @abstractmethod
    def lora_target_modules(self) -> List[str]:
        """
        Return the list of Linear module names to attach LoRA adapters to.
        These are the attention projection + MLP weight names specific to
        this model architecture.

        Example (Qwen):  ["q_proj", "k_proj", "v_proj", "o_proj",
                          "up_proj", "gate_proj", "down_proj"]
        Example (Gemma): ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"]
        """

    # ── Temperature handling ─────────────────────────────────────────────────

    @abstractmethod
    def recover_raw_logits(self, output_scores: torch.Tensor,
                           temperature: float) -> torch.Tensor:
        """
        HuggingFace model.generate(output_scores=True) returns logits that
        may have already been divided by the sampling temperature (depends on
        the model family).

        This method recovers the *raw* (pre-temperature) logits so that
        training can apply its own temperature consistently.

        Args:
            output_scores: Tensor of shape [T, vocab_size] as returned by
                           model.generate(..., output_scores=True).
            temperature:   The temperature passed to model.generate().

        Returns:
            Raw logits, same shape.

        Implementation note:
          - If the family pre-divides by temperature (e.g. Qwen3):
                return output_scores * temperature
          - If the family does NOT pre-divide (most others):
                return output_scores
        """

    # ── Forbidden / padding token masking ────────────────────────────────────

    @abstractmethod
    def clamp_log_probs(self, log_probs: torch.Tensor) -> torch.Tensor:
        """
        Some model families apply -inf logit bias to forbidden tokens before
        sampling.  After log_softmax this gives log_prob = -inf for those
        positions.  In reverse-KL training, p_student * (log_s - (-inf))
        produces NaN / inf that corrupts gradients.

        This method should replace any -inf values with a large finite
        negative number so that the computation graph stays clean.

        Args:
            log_probs: Output of F.log_softmax(logits, dim=-1), shape [..., V].

        Returns:
            Clamped log_probs, same shape.

        Implementation note:
          - If the family has forbidden tokens (Qwen3): clamp(min=-100.0)
          - If the family does not:                     return log_probs
        """

    # ── Tokenizer / prompt formatting ────────────────────────────────────────

    @abstractmethod
    def format_prompt(self, prompt: str, tokenizer) -> str:
        """
        Apply the model family's chat template to a raw prompt string.

        Args:
            prompt:    Raw user-facing question / instruction.
            tokenizer: The HuggingFace tokenizer for this model.

        Returns:
            A formatted string ready to be tokenized and fed to the model.
            For base (non-chat) models simply return prompt unchanged.
        """

    # ── Verifier / tree-attention mask ───────────────────────────────────────

    def tree_attn_mask(self, mask_4d: torch.Tensor):
        """
        Return the attention mask in the format this model's forward() expects
        during the batched tree verification pass in target_tree_pass().

        The verifier builds a 4D additive bias tensor of shape
        (1, 1, n_tree_nodes, cached_len + n_tree_nodes) that encodes
        which tree nodes may attend to which ancestors.

        Default (most architectures): return the raw 4D tensor.
          Works for GPT-2, LLaMA, Gemma, Mistral, Phi, and any standard
          HuggingFace causal LM that accepts additive 4D attention bias.

        Override for Qwen3: return {"full_attention": mask_4d}
          Qwen3's custom attention kernel expects a named-key dict so it can
          bypass its own internal mask construction and use the supplied one.

        Args:
            mask_4d: Float tensor of shape (1, 1, n, seq_len) with 0.0 for
                     allowed positions and finfo.min for blocked positions.

        Returns:
            The mask in whatever format model.forward(attention_mask=...) needs.
        """
        return mask_4d  # standard: raw 4D additive bias tensor

    # ── Default model IDs (can be overridden by CLI) ─────────────────────────

    @property
    def default_draft_model_id(self) -> str:
        """HuggingFace model ID for the default draft model of this family."""
        raise NotImplementedError(
            f"Family '{self.name}' has no default_draft_model_id. "
            "Pass --draft explicitly."
        )

    @property
    def default_target_model_id(self) -> str:
        """HuggingFace model ID for the default target (teacher) model."""
        raise NotImplementedError(
            f"Family '{self.name}' has no default_target_model_id. "
            "Pass --target explicitly."
        )
