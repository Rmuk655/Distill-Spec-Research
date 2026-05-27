"""
Base protocol for distillation losses.

All loss functions follow the same signature so they can be swapped
via a registry without changing the training loop.

Adding a new loss:
  1. Create capsules/distillation/losses/<loss_name>.py
  2. Implement compute() following the LossOutput convention.
  3. Register it in capsules/distillation/losses/__init__.py.
  4. Pass --loss <loss_name> to trainer.py.

See docs/ADDING_A_LOSS.md for a step-by-step walkthrough.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class LossOutput:
    """
    Standardised return type from every loss function.

    Fields:
        loss:           Scalar tensor (the value to back-prop through).
        accept_weight:  Optional float — mean per-token acceptance probability
                        α = min(1, p_target/p_student).  Only EBE computes
                        this; other losses return None.
        diagnostics:    Optional dict with richer α statistics for logging.
                        Keys (when present): "mean", "std", "min",
                        "frac_lt_0.95", "frac_lt_0.80".
                        Only EBE-family losses populate this; others leave None.
    """
    loss: torch.Tensor
    accept_weight: Optional[float] = None
    diagnostics: Optional[dict] = None


def compute_loss(
    name: str,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    token_ids: Optional[torch.Tensor] = None,
    **kwargs,
) -> LossOutput:
    """
    Dispatch to the named loss function.

    Args:
        name:            Loss identifier (must be in LOSS_REGISTRY).
        student_logits:  Float32 tensor [T, vocab_size] from draft model.
        teacher_logits:  Float32 tensor [T, vocab_size] from frozen teacher.
        token_ids:       Long tensor [T] of generated token indices.
                         Required for EBE; ignored by KL / JSD / L1.
        **kwargs:        Extra keyword arguments forwarded to the loss
                         (e.g. kl_weight=0.1 for EBE).

    Returns:
        LossOutput with .loss scalar and optional .accept_weight.
    """
    from . import LOSS_REGISTRY  # avoid circular import at module level
    if name not in LOSS_REGISTRY:
        available = ", ".join(sorted(LOSS_REGISTRY))
        raise ValueError(
            f"Unknown loss '{name}'. Available: {available}. "
            "See docs/ADDING_A_LOSS.md to add a new one."
        )
    return LOSS_REGISTRY[name](student_logits, teacher_logits,
                               token_ids=token_ids, **kwargs)
