"""
L1 / Total-Variation distance loss.

TV(p, q) = 0.5 * Σ_v |p(v) - q(v)|

Symmetric, bounded in [0, 1], simple to implement.
Weaker distillation signal than KL / JSD in practice because it treats
all mismatches equally regardless of probability scale.
Use as ablation, not primary objective.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from .base import LossOutput


def l1(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    token_ids: Optional[torch.Tensor] = None,  # unused
    **_kwargs,
) -> LossOutput:
    """
    Total-variation distance between student and teacher distributions.

    Args:
        student_logits: Float32 [T, V]
        teacher_logits: Float32 [T, V]

    Returns:
        LossOutput with scalar loss in [0, 1].
    """
    p_s  = F.softmax(student_logits, dim=-1)
    p_t  = F.softmax(teacher_logits, dim=-1)
    loss = (p_s - p_t).abs().sum(dim=-1).mean() * 0.5
    return LossOutput(loss=loss)
