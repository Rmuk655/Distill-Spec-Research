"""
Expected Block Efficiency (EBE) loss — novel contribution.

Directly optimises the speculative-decoding inference metric:
    E[tau+1] = 1 + sum_{k=1}^{L} prod_{i=1}^{k} alpha_i

where alpha_i = min(1, p_target(t_i) / p_student(t_i)) is the per-token
acceptance probability and L is the block (draft window) length.

The ratio min(1, p_target/p_student) follows the standard speculative
decoding acceptance formula (Leviathan et al. 2023, Chen et al. 2023):
when p_student > p_target (draft overconfident), alpha < 1 and the EBE
gradient pushes p_student DOWN toward p_target.  When p_student <= p_target
(draft underconfident), the clamp(max=0) kills the gradient for EBE — only
the KL regulariser provides signal at those positions.

Offline gradient behaviour:
  On offline (human-generated) training data, EBE provides gradient only
  at positions where the draft is MORE confident than the teacher.  In
  early training, the draft is typically underconfident on precise tokens
  (teacher >> draft at most positions), so EBE gradient is near-zero and
  the KL regulariser dominates.  As training progresses and the draft
  approaches the teacher, EBE begins contributing.

  kl_weight=1.0 (default) ensures that even when EBE gradient is near-zero
  (early training), the KL term provides a signal equivalent to standalone
  KL distillation.  Using kl_weight=0.1 was found to make EBE approximately
  10x weaker than standalone KL, producing near-baseline or slightly worse
  results on offline data.

Key design choices:
  1. Per-block computation (not whole-sequence cumprod).
     Inference calls the draft model in windows of L tokens.  Using L=8
     blocks in training matches the inference scale exactly and avoids
     the "giant product" problem where early tokens accumulate an
     unrealistically large gradient multiplier.

  2. KL regulariser (kl_weight=1.0 default).
     EBE gradient vanishes for tokens where draft <= teacher (alpha = 1).
     The full-weight KL term provides gradient at those positions and
     ensures training never falls below standalone KL quality.

  3. Numerical clamping (alpha >= 1e-6).
     torch.cumprod backward divides by each element; if any alpha = 0 the
     backward produces NaN.  Clamping to 1e-6 prevents this.

Reference: Thomas et al., arXiv:2602.16994v1 (2026), Section 4.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from .base import LossOutput

# Must match GBV/runner.py default --L (inference block length).
# Changing this requires re-running the full eval suite for comparability.
DEFAULT_BLOCK_LEN: int = 8


def ebe(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    token_ids: Optional[torch.Tensor] = None,
    kl_weight: float = 1.0,
    block_len: int = DEFAULT_BLOCK_LEN,
    **_kwargs,
) -> LossOutput:
    """
    Block-level EBE surrogate + forward-KL regulariser.

    Args:
        student_logits: Float32 [T, V]  — draft model output.
        teacher_logits: Float32 [T, V]  — frozen teacher output.
        token_ids:      Long   [T]      — token indices of the generated
                                          sequence.  Required (raises if None).
        kl_weight:      Weight for the KL regulariser (default 1.0).
                        Set to 1.0 so the KL term matches standalone KL
                        distillation strength when EBE gradient is near-zero
                        (common in early offline training).  The previous
                        default of 0.1 made EBE ~10x weaker than standalone KL.
        block_len:      Draft window length L (default 8, matches inference).

    Returns:
        LossOutput with scalar loss and accept_weight (mean alpha).

    Raises:
        ValueError if token_ids is None.
    """
    if token_ids is None:
        raise ValueError(
            "EBE loss requires token_ids (the generated token indices). "
            "Pass token_ids= from the training loop."
        )

    log_s = F.log_softmax(student_logits, dim=-1)           # [T, V]
    log_t = F.log_softmax(teacher_logits, dim=-1).detach()  # [T, V] teacher is fixed

    # Per-token log-probability of the actually-generated tokens
    log_p = log_s.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)  # [T]
    log_q = log_t.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)  # [T]

    # alpha_i = min(1, p_target/p_student) = exp(min(0, log_teacher - log_draft)).
    # log_q = log_teacher, log_p = log_student (draft).  Variable naming follows
    # the code convention; the formula matches spec-dec literature (Leviathan 2023).
    # Gradient fires only when log_q < log_p (draft overconfident); zero when
    # draft is underconfident (common in early offline training).
    # Clamp >= 1e-6 prevents NaN in cumprod backward when alpha approaches 0.
    alpha = torch.exp(torch.clamp(log_q - log_p, max=0.0)).clamp(min=1e-6)  # [T]

    # EBE over non-overlapping blocks of length block_len
    T = alpha.shape[0]
    n_blocks = max(1, T // block_len)
    ebe_val = torch.zeros(1, device=alpha.device, dtype=alpha.dtype)
    for b in range(n_blocks):
        blk = alpha[b * block_len : (b + 1) * block_len]      # [≤ block_len]
        ebe_val = ebe_val - (1.0 + torch.cumprod(blk, dim=0).sum())
    ebe_val = ebe_val / n_blocks   # average → scale-independent of sequence length

    # KL regulariser: forward KL provides gradient for tokens already accepted
    kl_reg = F.kl_div(log_s, log_t.exp(), reduction="batchmean")

    loss = ebe_val + kl_weight * kl_reg

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
