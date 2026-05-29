"""
Single-token EBE surrogate loss — ablation of the block-level EBE.

    L = -mean_{t=1}^{T}( alpha_t )     where  alpha_t = min(1, p_target(t) / p_student(t))

Directly maximises the mean per-token acceptance rate without any cumprod,
block structure, or KL regulariser.

Acceptance probability formula (matches spec-dec literature):
  alpha_t = min(1, p_target(t) / p_student(t))
  The clamp(max=0) kills the gradient when p_student <= p_target (draft
  underconfident).  Gradient only fires when draft is overconfident, pushing
  it DOWN toward the teacher.

Why this matters as an ablation
---------------------------------
Multi-token EBE (ebe.py) chains L=8 acceptance probabilities with cumprod.
If the cumprod structure is the source of gradient instability or vanishing,
single-token EBE will train stably and the comparison isolates the cause.

Design choices vs block EBE
-----------------------------
  1. No cumprod — gradient reaches every token independently, no block-
     boundary effects.
  2. No KL regulariser — single clean objective.  Add --kl_weight to the
     combined loss call if KL is needed alongside.
  3. Dense gradient — every token contributes equally regardless of position
     within a speculative block.

Offline-setting note
---------------------
On offline training data (human-generated or teacher-sampled), alpha_t ≈ 1
at most positions because the teacher typically assigns higher probability
than the (under-trained) draft.  Gradient is near-zero at those positions.
Use online_serve.py --kl_method ebe_single for the online version where
the draft generates the tokens, creating genuine alpha < 1 rejections.

Note: the online EBE is currently broken due to the buffer storing residual
tokens rather than the draft's actual proposals.  See online_serve.py
lines 197-228 for the documented bug and fix plan.

Reference: ablation of Thomas et al., arXiv:2602.16994v1 (2026), Section 4.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from .base import LossOutput


def ebe_single(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    token_ids: Optional[torch.Tensor] = None,
    **_kwargs,
) -> LossOutput:
    """
    Single-token EBE: L = −mean(α),  α_t = min(1, q_t / p_t).

    Args:
        student_logits: Float32 [T, V]  — draft model output.
        teacher_logits: Float32 [T, V]  — frozen teacher output.
        token_ids:      Long   [T]      — token indices of the generated
                                          sequence.  Required (raises if None).

    Returns:
        LossOutput with scalar loss, accept_weight (mean α), and diagnostics.

    Raises:
        ValueError if token_ids is None.
    """
    if token_ids is None:
        raise ValueError(
            "ebe_single loss requires token_ids (the generated token indices). "
            "Pass token_ids= from the training loop."
        )

    log_s = F.log_softmax(student_logits, dim=-1)           # [T, V]
    log_t = F.log_softmax(teacher_logits, dim=-1).detach()  # [T, V] teacher is fixed

    # Per-token log-probability of the actually-generated tokens
    log_p = log_s.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)  # [T]
    log_q = log_t.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)  # [T]

    # alpha_t = min(1, p_target/p_student) = exp(min(0, log_teacher - log_draft)).
    # log_q = log_teacher, log_p = log_student (draft) — naming follows the code
    # convention where log_p is the "proposal" and log_q is the "quality signal".
    # Gradient fires only when log_q < log_p (teacher < draft, i.e., overconfident).
    # Clamp >= 1e-6: prevents NaN in backward when student assigns near-zero prob.
    alpha = torch.exp(torch.clamp(log_q - log_p, max=0.0)).clamp(min=1e-6)  # [T]

    loss = -alpha.mean()

    # Rich α diagnostics — logged by the training loop every log_every steps.
    with torch.no_grad():
        _a = alpha.detach()
        _diag = {
            "mean":         _a.mean().item(),
            "std":          _a.std().item(),
            "min":          _a.min().item(),
            "frac_lt_0.95": (_a < 0.95).float().mean().item(),
            "frac_lt_0.80": (_a < 0.80).float().mean().item(),
        }
    return LossOutput(loss=loss, accept_weight=_diag["mean"], diagnostics=_diag)
