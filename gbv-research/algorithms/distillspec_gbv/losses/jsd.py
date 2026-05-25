"""
Jensen–Shannon divergence loss.

JSD(p ∥ q) = α * KL(p ∥ m) + (1-α) * KL(q ∥ m),  where m = α*p + (1-α)*q.

With α=0.5 this is symmetric and bounded: JSD ∈ [0, log 2].
Used as an ablation against forward-KL (DistillSpec baseline).

JSD is mode-covering like forward-KL but slightly softer: it penalises
the student for diverging from the mixture m rather than from the teacher
directly, which can improve training stability.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from .base import LossOutput

_LOG_CLAMP_MIN: float = -1e9  # prevent log(0) in entropy terms


def jsd(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    token_ids: Optional[torch.Tensor] = None,  # unused
    alpha: float = 0.5,
    **_kwargs,
) -> LossOutput:
    """
    Jensen–Shannon divergence between student and teacher distributions.

    Args:
        student_logits: Float32 [T, V]
        teacher_logits: Float32 [T, V]
        alpha:          Mixture weight for the student (default 0.5 → symmetric).

    Returns:
        LossOutput with scalar loss.
    """
    p_s = F.softmax(student_logits, dim=-1)   # [T, V]
    p_t = F.softmax(teacher_logits, dim=-1)   # [T, V]
    m   = alpha * p_s + (1.0 - alpha) * p_t   # mixture distribution

    log_m = m.log().clamp(min=_LOG_CLAMP_MIN)
    kl_s  = (p_s * (p_s.log().clamp(min=_LOG_CLAMP_MIN) - log_m)).sum(-1).mean()
    kl_t  = (p_t * (p_t.log().clamp(min=_LOG_CLAMP_MIN) - log_m)).sum(-1).mean()

    loss  = alpha * kl_s + (1.0 - alpha) * kl_t
    return LossOutput(loss=loss)
