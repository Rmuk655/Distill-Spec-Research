"""
Flat (per-token, off-policy) distillation losses.

These compute a divergence between the student and teacher distributions at
every token of a fixed teacher-generated sequence.  They DO NOT use the
speculative-decoding tree structure — for that, see losses/tree.py.

All flat losses share the same signature:

    loss_fn(student_logits, teacher_logits, **_kwargs) -> torch.Tensor

so they can be swapped through the registry in losses/__init__.py without
changing the training loop.

Shapes
------
student_logits : float32  [T, V]   draft model output (WITH grad)
teacher_logits : float32  [T, V]   teacher model output (frozen / detached)

Where T is the number of tokens in the rollout and V is the vocab size.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def forward_kl(student_logits, teacher_logits, **_kw):
    """
    KL(teacher ∥ student) — mode-covering.

    Standard DistillSpec / Hinton-style distillation: the student is pushed
    to spread mass over wherever the teacher has mass.  Tends to produce a
    student that is conservative (matches the teacher's full support).
    """
    log_s = F.log_softmax(student_logits, dim=-1)                # [T, V]
    p_t   = F.softmax(teacher_logits, dim=-1).detach()           # [T, V]
    # KL(p_t || p_s) = sum_v p_t * (log p_t - log p_s).  F.kl_div expects
    # log_target=False (default) and reduction="batchmean" averages over T.
    return F.kl_div(log_s, p_t, reduction="batchmean")


def reverse_kl(student_logits, teacher_logits, **_kw):
    """
    KL(student ∥ teacher) — mode-seeking.

    The student concentrates mass on the teacher's peak.  For speculative
    decoding this directly maximises min(1, p_t/p_s) for the most likely
    tokens, which raises naive acceptance.  The downside is the student
    can ignore the tail of the teacher distribution.
    """
    log_s = F.log_softmax(student_logits, dim=-1)
    log_t = F.log_softmax(teacher_logits, dim=-1).detach().clamp(min=-100.0)
    p_s   = log_s.exp()
    return (p_s * (log_s - log_t)).sum(dim=-1).mean()


def jsd(student_logits, teacher_logits, alpha=0.5, **_kw):
    """
    Jensen–Shannon divergence — symmetric, bounded in [0, log 2].

    JSD(p ∥ q ; α) = α · KL(q ∥ m) + (1−α) · KL(p ∥ m)
    where m = α·q + (1−α)·p is the mixture distribution.

    With α=0.5 this is the standard symmetric JSD.  Softer than forward KL —
    useful when teacher and student disagree sharply, because the gradient is
    bounded.
    """
    log_s = F.log_softmax(student_logits, dim=-1)
    log_t = F.log_softmax(teacher_logits, dim=-1).detach()
    p_s   = log_s.exp()
    p_t   = log_t.exp()
    m     = alpha * p_s + (1.0 - alpha) * p_t                    # mixture
    log_m = m.clamp(min=1e-12).log()
    kl_s  = (p_s * (log_s - log_m)).sum(dim=-1).mean()
    kl_t  = (p_t * (log_t - log_m)).sum(dim=-1).mean()
    return alpha * kl_s + (1.0 - alpha) * kl_t


def l1(student_logits, teacher_logits, **_kw):
    """
    L1 distance between student and teacher distributions, averaged per token.

    Equivalent (up to a constant) to 2 · TV(p_s, p_t).  Linear-in-error rather
    than log-error like KL, which gives a more stable gradient when the two
    distributions are far apart.
    """
    p_s = F.softmax(student_logits, dim=-1)
    p_t = F.softmax(teacher_logits, dim=-1).detach()
    return (p_s - p_t).abs().sum(dim=-1).mean()


def tvplus(student_logits, teacher_logits, **_kw):
    """
    One-sided TV: TV⁺(q, p) = Σₓ max(0, q(x) − p(x)).

    Directly minimises the per-token rejection probability under NSS (K=1) and
    Global Resolution (K>1).  For K i.i.d. drafts, expected acceptance =
    1 − TV⁺(q,p)^K, so this loss is the theoretically-grounded training
    objective for both verifiers.

    Unlike JSD/forward-KL, which push q toward p everywhere, TV⁺ only
    penalises tokens where the draft *overshoots* the teacher — the tokens a
    rejection sampler would reject.  Tokens where q(x) < p(x) are already
    accepted at full rate and contribute zero loss, so gradient is not wasted
    on them.  This makes TV⁺ the tightest possible surrogate for NSS/GR
    acceptance rate.
    """
    p_s = F.softmax(student_logits, dim=-1)
    p_t = F.softmax(teacher_logits, dim=-1).detach()
    # max(0, q - p) per token, summed over vocab, averaged over positions
    return F.relu(p_s - p_t).sum(dim=-1).mean()


# Registry: name → callable.  train.py looks the loss up here by --loss flag.
FLAT_LOSSES = {
    "forward_kl":      forward_kl,
    "reverse_kl":      reverse_kl,
    "jsd":             jsd,
    "jsd_flat_enrich": jsd,   # same loss fn — routing in train.py samples K stochastic teacher paths
    "l1":              l1,
    "tvplus":          tvplus,  # one-sided TV: direct surrogate for NSS/Global Resolution acceptance
}
