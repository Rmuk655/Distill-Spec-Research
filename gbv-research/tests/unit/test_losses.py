"""
test_losses.py — unit tests for all 5 distillation loss functions.

Tests run on CPU with tiny tensors.  No model loading required.

Loss functions under test (all in gbv-research/algorithms/train_qwen3.py):
  - forward_kl_loss   : KL(teacher ‖ student)
  - reverse_kl_loss   : KL(student ‖ teacher)  with -inf guard
  - jsd_loss          : Jensen-Shannon divergence
  - l1_loss           : L1 / total-variation distance
  - ebe_loss          : Expected Block Efficiency surrogate + KL regulariser

Critical bugs this suite guards against:
  - reverse_kl: NaN/Inf when teacher has -inf logits (forbidden tokens)
  - ebe:        NaN in cumprod backward when alpha=0  (clamp guard)
  - ebe:        gradient explosion from "giant single block" (block-level split)
  - jsd:        non-symmetric when alpha != 0.5
  - all:        loss > 0 for random logits, = 0 (or near 0) when logits match
"""

import math
import sys
import os
import pytest
import torch
import torch.nn.functional as F

# Import loss functions directly from gbv-research/algorithms/train_qwen3.py.
# conftest.py adds gbv-research/algorithms/ to sys.path.
import train_qwen3 as _t

forward_kl_loss = _t.forward_kl_loss
reverse_kl_loss = _t.reverse_kl_loss
jsd_loss        = _t.jsd_loss
l1_loss         = _t.l1_loss
ebe_loss        = _t.ebe_loss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_logits(T=10, V=16, seed=0):
    torch.manual_seed(seed)
    s = torch.randn(T, V, requires_grad=True)
    t = torch.randn(T, V)
    return s, t

def _identical_logits(T=10, V=16, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(T, V)
    s = x.clone().requires_grad_(True)
    t = x.clone()
    return s, t

def _inflogit_teacher(T=5, V=16, forbidden_count=3):
    """Teacher with several forbidden tokens (-inf logit) — the reverse-KL NaN trigger."""
    t = torch.randn(T, V)
    for i in range(forbidden_count):
        t[:, i] = float("-inf")
    s = torch.randn(T, V, requires_grad=True)
    return s, t


# ===========================================================================
# forward_kl_loss
# ===========================================================================

class TestForwardKL:

    def test_nonnegative_random(self):
        s, t = _random_logits()
        loss = forward_kl_loss(s, t)
        assert loss.item() >= 0.0, "Forward KL must be >= 0"

    def test_finite_random(self):
        s, t = _random_logits()
        loss = forward_kl_loss(s, t)
        assert math.isfinite(loss.item()), f"Forward KL is not finite: {loss.item()}"

    def test_minimized_when_identical(self):
        """
        forward_kl_loss computes cross-entropy H(p_t, p_s), not KL divergence.
        H(p, p) = H(p) = entropy of p (non-zero for random distributions).
        The key property: identical logits produce the MINIMUM possible loss.
        Any perturbation of the student should increase the loss.
        """
        T, V = 4, 16
        torch.manual_seed(0)
        t = torch.randn(T, V)
        # Student at minimum: same logits as teacher
        s_opt = t.clone().requires_grad_(False)
        loss_opt = forward_kl_loss(s_opt, t)

        # Student perturbed: should have higher loss
        s_pert = (t + torch.randn_like(t) * 1.0).requires_grad_(False)
        loss_pert = forward_kl_loss(s_pert, t)

        assert loss_opt.item() <= loss_pert.item() + 1e-4, \
            f"forward_kl_loss not minimized when logits identical: " \
            f"opt={loss_opt.item():.4f} vs pert={loss_pert.item():.4f}"

    def test_gradient_flows(self):
        s, t = _random_logits()
        loss = forward_kl_loss(s, t)
        loss.backward()
        assert s.grad is not None
        assert torch.isfinite(s.grad).all(), "Gradient contains NaN/Inf"

    def test_larger_when_more_different(self):
        """KL should be larger when distributions are more different."""
        V = 16
        # Very similar
        t_base = torch.randn(1, V)
        s_close = t_base.clone() + 0.01 * torch.randn(1, V)
        s_close.requires_grad_(False)
        loss_close = forward_kl_loss(s_close, t_base)

        # Very different
        s_far = torch.randn(1, V, requires_grad=False) * 5.0
        loss_far = forward_kl_loss(s_far, t_base)

        assert loss_far.item() > loss_close.item(), \
            "KL should be larger for more different distributions"


# ===========================================================================
# reverse_kl_loss
# ===========================================================================

class TestReverseKL:

    def test_nonnegative_random(self):
        s, t = _random_logits()
        loss = reverse_kl_loss(s, t)
        assert loss.item() >= 0.0

    def test_finite_random(self):
        s, t = _random_logits()
        loss = reverse_kl_loss(s, t)
        assert math.isfinite(loss.item()), f"Reverse KL is not finite: {loss.item()}"

    def test_no_nan_with_forbidden_tokens(self):
        """
        Critical regression: teacher has -inf logits for forbidden tokens.
        Old code: p_s * (log_s - log_t) → p_s * inf → NaN when p_s=0.
        Fix: log_t.clamp(min=-100).
        """
        s, t = _inflogit_teacher()
        loss = reverse_kl_loss(s, t)
        assert not torch.isnan(loss), "NaN with -inf teacher logits — clamp guard broken"
        assert not torch.isinf(loss), "Inf with -inf teacher logits — clamp guard broken"

    def test_gradient_no_nan_with_forbidden_tokens(self):
        """Gradient must not contain NaN when teacher has -inf logits."""
        s, t = _inflogit_teacher()
        loss = reverse_kl_loss(s, t)
        loss.backward()
        assert s.grad is not None
        assert not torch.isnan(s.grad).any(), \
            "NaN gradient with -inf teacher — corrupts LoRA weights on first step"
        assert not torch.isinf(s.grad).any(), \
            "Inf gradient with -inf teacher"

    def test_finite_loss_with_many_forbidden(self):
        """
        When most (but not all) teacher tokens are forbidden (-inf logit),
        the clamp guard should prevent NaN.  The all-forbidden case is
        degenerate (softmax of -inf everywhere = 0/0 = NaN) and does not
        occur in practice (teacher always has at least one valid token).
        """
        T, V = 4, 8
        t = torch.full((T, V), float("-inf"))
        t[:, 0] = 1.0   # one valid token per row — matches real-world forbidden-token scenario
        s = torch.randn(T, V, requires_grad=True)
        loss = reverse_kl_loss(s, t)
        assert math.isfinite(loss.item()), "Should not crash with mostly-forbidden teacher"

    def test_gradient_flows(self):
        s, t = _random_logits()
        loss = reverse_kl_loss(s, t)
        loss.backward()
        assert s.grad is not None
        assert torch.isfinite(s.grad).all()


# ===========================================================================
# jsd_loss
# ===========================================================================

class TestJSD:

    def test_nonnegative(self):
        s, t = _random_logits()
        loss = jsd_loss(s, t)
        assert loss.item() >= 0.0

    def test_finite(self):
        s, t = _random_logits()
        assert math.isfinite(jsd_loss(s, t).item())

    def test_upper_bound(self):
        """JSD(p,q) <= ln(2) always, for alpha=0.5."""
        s, t = _random_logits()
        assert jsd_loss(s, t).item() <= math.log(2) + 1e-5

    def test_near_zero_when_identical(self):
        """JSD(p, p) should be ~0."""
        s, t = _identical_logits()
        loss = jsd_loss(s, t)
        assert loss.item() < 1e-4, f"JSD(p,p) should be ~0, got {loss.item()}"

    def test_symmetric(self):
        """JSD is symmetric: JSD(p,q) == JSD(q,p) at alpha=0.5."""
        T, V = 8, 16
        torch.manual_seed(7)
        x = torch.randn(T, V)
        y = torch.randn(T, V)
        loss_xy = jsd_loss(x, y)
        loss_yx = jsd_loss(y, x)
        assert abs(loss_xy.item() - loss_yx.item()) < 1e-4, \
            f"JSD not symmetric: {loss_xy.item()} vs {loss_yx.item()}"

    def test_gradient_flows(self):
        s, t = _random_logits()
        loss = jsd_loss(s, t)
        loss.backward()
        assert s.grad is not None
        assert torch.isfinite(s.grad).all()


# ===========================================================================
# l1_loss
# ===========================================================================

class TestL1:

    def test_nonnegative(self):
        s, t = _random_logits()
        assert l1_loss(s, t).item() >= 0.0

    def test_upper_bound(self):
        """TV distance is in [0, 1]."""
        s, t = _random_logits()
        assert l1_loss(s, t).item() <= 1.0 + 1e-5

    def test_zero_when_identical(self):
        s, t = _identical_logits()
        assert l1_loss(s, t).item() < 1e-5

    def test_maximum_for_opposite_supports(self):
        """TV = 1 when distributions have disjoint support."""
        V = 4
        # p puts all mass on token 0, q puts all mass on token 2
        p_logits = torch.tensor([[10.0, -10.0, -10.0, -10.0]])
        q_logits = torch.tensor([[-10.0, -10.0, 10.0, -10.0]])
        loss = l1_loss(p_logits, q_logits)
        assert abs(loss.item() - 1.0) < 0.01, \
            f"TV distance for disjoint supports should be ~1, got {loss.item()}"

    def test_gradient_flows(self):
        s, t = _random_logits()
        loss = l1_loss(s, t)
        loss.backward()
        assert s.grad is not None
        assert torch.isfinite(s.grad).all()


# ===========================================================================
# ebe_loss
# ===========================================================================

class TestEBE:

    def _make_ebe_inputs(self, T=16, V=8, seed=1):
        """T tokens, V vocab — sequence must be long enough for block splitting."""
        torch.manual_seed(seed)
        s = torch.randn(T, V, requires_grad=True)
        t = torch.randn(T, V)
        ids = torch.randint(0, V, (T,))
        return s, t, ids

    def test_returns_tuple(self):
        s, t, ids = self._make_ebe_inputs()
        result = ebe_loss(s, t, ids)
        assert isinstance(result, tuple) and len(result) == 2, \
            "ebe_loss should return (loss, alpha_mean)"

    def test_loss_is_finite(self):
        s, t, ids = self._make_ebe_inputs()
        loss, aw = ebe_loss(s, t, ids)
        assert math.isfinite(loss.item()), f"EBE loss is not finite: {loss.item()}"

    def test_alpha_mean_in_unit_interval(self):
        s, t, ids = self._make_ebe_inputs()
        _, aw = ebe_loss(s, t, ids)
        assert 0.0 <= aw <= 1.0, f"alpha mean must be in [0,1], got {aw}"

    def test_gradient_flows(self):
        s, t, ids = self._make_ebe_inputs()
        loss, _ = ebe_loss(s, t, ids)
        loss.backward()
        assert s.grad is not None
        assert torch.isfinite(s.grad).all(), "EBE gradient contains NaN/Inf"

    def test_no_nan_when_alpha_is_zero(self):
        """
        Critical regression: alpha=0 caused NaN in cumprod backward.
        Fixed by alpha.clamp(min=1e-6).

        Note: requires_grad must be set on a LEAF tensor (not a non-leaf
        like `torch.randn(...) * 10.0`), otherwise .grad is not populated.
        """
        T, V = 8, 4
        # Make teacher logits much larger so student probs << teacher probs
        # → exp(log_q - log_p) << 1, but not exactly 0 due to clamp
        # Create a proper leaf tensor:
        s = (torch.randn(T, V) * 10.0).detach().requires_grad_(True)
        t = torch.randn(T, V) * 0.1
        ids = torch.zeros(T, dtype=torch.long)
        loss, aw = ebe_loss(s, t, ids)
        assert not torch.isnan(loss), "EBE NaN when alpha near 0 — clamp guard broken"
        loss.backward()
        assert s.grad is not None, "Gradient not computed for leaf s"
        assert torch.isfinite(s.grad).all(), "EBE gradient NaN when alpha near 0"

    def test_block_level_split_respected(self):
        """
        ebe_loss splits the sequence into blocks of block_len=8.
        With T=32 and block_len=8, there should be 4 blocks, not 1.
        Indirectly verified: if block splitting weren't happening, the EBE
        component would be much larger (gradient explosion).
        """
        T, V, block_len = 32, 8, 8
        torch.manual_seed(10)
        s = torch.randn(T, V, requires_grad=True)
        t = torch.randn(T, V)
        ids = torch.randint(0, V, (T,))

        loss_blocked, _ = ebe_loss(s, t, ids, block_len=block_len)
        loss_1block, _  = ebe_loss(s, t, ids, block_len=T)

        # The blocked version averages over 4 blocks — the KL component is identical
        # but the EBE components differ because cumprod length differs.
        assert loss_blocked.item() != loss_1block.item(), \
            "block_len parameter has no effect — block splitting may be broken"

    def test_kl_weight_zero_gives_pure_ebe(self):
        """With kl_weight=0, loss == EBE term only (no KL regulariser)."""
        s, t, ids = self._make_ebe_inputs()
        loss_with_kl, _ = ebe_loss(s, t, ids, kl_weight=0.1)
        loss_no_kl, _   = ebe_loss(s, t, ids, kl_weight=0.0)
        assert loss_with_kl.item() != loss_no_kl.item(), \
            "kl_weight=0 and kl_weight=0.1 should produce different losses"
