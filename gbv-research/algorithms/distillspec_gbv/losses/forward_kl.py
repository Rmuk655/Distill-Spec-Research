"""
Forward KL divergence loss: KL(target ∥ student).

This is the DistillSpec baseline (Section 3.1).  It is mode-covering:
the student is penalised for assigning low probability to tokens the
teacher assigns high probability, so it learns to spread probability
across all plausible completions.

Numerics:
  - Uses log_softmax + softmax (more stable than log(softmax) twice).
  - No special clamping needed: teacher logits enter only via softmax
    (bounded output).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from .base import LossOutput


def forward_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    token_ids: Optional[torch.Tensor] = None,  # unused, kept for uniform signature
    **_kwargs,
) -> LossOutput:
    """
    KL(p_teacher ∥ p_student) = H(p_teacher, p_student) − H(p_teacher).

    Because H(p_teacher) is constant w.r.t. student parameters, minimising
    this is equivalent to minimising the cross-entropy H(p_teacher, p_student).

    Args:
        student_logits: Float32 [T, V]
        teacher_logits: Float32 [T, V]

    Returns:
        LossOutput with scalar loss.
    """
    log_s = F.log_softmax(student_logits, dim=-1)   # [T, V]
    p_t   = F.softmax(teacher_logits,    dim=-1)    # [T, V]
    loss  = -(p_t * log_s).sum(dim=-1).mean()
    return LossOutput(loss=loss)
