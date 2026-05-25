"""
Reverse KL divergence loss: KL(student ∥ target).

Mode-seeking behaviour: the student concentrates probability mass on
the teacher's most likely tokens and is allowed to assign zero mass
elsewhere.  Tends to produce sharper draft distributions which can
increase acceptance rate for the top-k tokens but hurt tail coverage.

Known numerical hazard (Qwen3-specific):
  Qwen3 applies -inf logit bias to forbidden / control tokens.
  After log_softmax those positions have log_t = -inf.
  Forward:  p_s * (log_s - (-inf)) = p_s * +inf → NaN when p_s ≈ 0.
  Backward: gradient ∝ 0 * inf = NaN → corrupts LoRA weights on step 1.

Fix (handled here via ModelFamily.clamp_log_probs):
  The caller should clamp log_t before passing to this function, OR
  this function clamps internally.  We do the internal clamp as the
  default path to be safe even if the caller forgets.

To use without internal clamping (if you are sure the teacher has no
-inf logits, e.g. Gemma with no forbidden-token bias), pass
`skip_clamp=True`.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from .base import LossOutput

# Minimum log-probability value to use when clamping.
# exp(-100) ≈ 3.7e-44 — negligible but finite.  Keeps log-ratio bounded
# so gradients don't explode even for tokens the teacher considers forbidden.
_LOG_CLAMP_MIN: float = -100.0


def reverse_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    token_ids: Optional[torch.Tensor] = None,  # unused, kept for uniform signature
    skip_clamp: bool = False,
    **_kwargs,
) -> LossOutput:
    """
    KL(p_student ∥ p_teacher) = Σ_v p_s[v] * (log_s[v] - log_t[v]).

    Args:
        student_logits: Float32 [T, V]
        teacher_logits: Float32 [T, V]
        skip_clamp:     If True, skip the -inf clamp on log_t.
                        Only safe if the teacher has no forbidden-token
                        logit biases (i.e. no -inf entries after softmax).

    Returns:
        LossOutput with scalar loss.
    """
    log_t = F.log_softmax(teacher_logits, dim=-1)
    if not skip_clamp:
        log_t = log_t.clamp(min=_LOG_CLAMP_MIN)

    log_s = F.log_softmax(student_logits, dim=-1)
    p_s   = F.softmax(student_logits,    dim=-1)
    loss  = (p_s * (log_s - log_t)).sum(dim=-1).mean()
    return LossOutput(loss=loss)
