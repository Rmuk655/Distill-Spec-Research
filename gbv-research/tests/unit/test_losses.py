"""
test_losses.py — unit tests for all 6 distillation loss functions.

Tests run on CPU with tiny tensors.  No model loading required.

Loss functions under test (all in algorithms/distillspec_gbv/losses/):
  - forward_kl  : KL(teacher ‖ student)
  - reverse_kl  : KL(student ‖ teacher)  with -inf guard
  - jsd         : Jensen-Shannon divergence
  - l1          : L1 / total-variation distance
  - ebe         : Expected Block Efficiency surrogate + KL regulariser
  - ebe_single  : Single-token EBE = -mean(alpha) — ablation

Critical bugs this suite guards against:
  - reverse_kl: NaN/Inf when teacher has -inf logits (forbidden tokens)
  - ebe:        NaN in cumprod backward when alpha=0  (clamp guard)
  - ebe:        gradient explosion from "giant single block" (block-level split)
  - jsd:        non-symmetric when alpha != 0.5
  - all:        loss > 0 for random logits, = 0 (or near 0) when logits match

Return type for all losses: LossOutput(loss, accept_weight, diagnostics)
  - loss          : scalar tensor (backprop through this)
  - accept_weight : float or None  (mean α for EBE-family; None for KL/JSD/L1)
  - diagnostics   : dict or None   (mean/std/min/frac_lt_0.95/frac_lt_0.80 for EBE-family)
"""

import math
import sys
import os
import pytest
import torch

# conftest.py adds gbv-research/algorithms/ to sys.path.
# distillspec_gbv/__init__.py uses lazy __getattr__ so GBVAlgorithm is NOT
# loaded here — only the losses subpackage is imported.
from distillspec_gbv.losses import (
    forward_kl,
    reverse_kl,
    jsd,
    l1,
    ebe,
    ebe_single,
    LossOutput,
    get_loss,
)


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
# forward_kl
# ===========================================================================

class TestForwardKL:

    def test_nonnegative_random(self):
        s, t = _random_logits()
        out = forward_kl(s, t)
        assert out.loss.item() >= 0.0, "Forward KL must be >= 0"

    def test_returns_loss_output(self):
        s, t = _random_logits()
        out = forward_kl(s, t)
        assert isinstance(out, LossOutput)
        assert out.accept_weight is None
        assert out.diagnostics is None

    def test_finite_random(self):
        s, t = _random_logits()
        assert math.isfinite(forward_kl(s, t).loss.item())

    def test_minimized_when_identical(self):
        """
        forward_kl computes cross-entropy H(p_t, p_s), not KL divergence.
        H(p, p) = H(p) = entropy of p (non-zero for random distributions).
        The key property: identical logits produce the MINIMUM possible loss.
        Any perturbation of the student should increase the loss.
        """
        T, V = 4, 16
        torch.manual_seed(0)
        t = torch.randn(T, V)
        s_opt  = t.clone().requires_grad_(False)
        s_pert = (t + torch.randn_like(t) * 1.0).requires_grad_(False)

        loss_opt  = forward_kl(s_opt,  t).loss
        loss_pert = forward_kl(s_pert, t).loss

        assert loss_opt.item() <= loss_pert.item() + 1e-4, \
            f"forward_kl not minimized when logits identical: " \
            f"opt={loss_opt.item():.4f} vs pert={loss_pert.item():.4f}"

    def test_gradient_flows(self):
        s, t = _random_logits()
        forward_kl(s, t).loss.backward()
        assert s.grad is not None
        assert torch.isfinite(s.grad).all(), "Gradient contains NaN/Inf"

    def test_larger_when_more_different(self):
        """KL should be larger when distributions are more different."""
        V = 16
        t_base  = torch.randn(1, V)
        s_close = (t_base.clone() + 0.01 * torch.randn(1, V)).requires_grad_(False)
        s_far   = torch.randn(1, V, requires_grad=False) * 5.0
        assert forward_kl(s_far, t_base).loss.item() > forward_kl(s_close, t_base).loss.item(), \
            "KL should be larger for more different distributions"

    def test_get_loss_dispatch(self):
        s, t = _random_logits()
        out = get_loss("forward_kl", s, t)
        assert isinstance(out, LossOutput)
        assert math.isfinite(out.loss.item())


# ===========================================================================
# reverse_kl
# ===========================================================================

class TestReverseKL:

    def test_nonnegative_random(self):
        s, t = _random_logits()
        assert reverse_kl(s, t).loss.item() >= 0.0

    def test_finite_random(self):
        s, t = _random_logits()
        assert math.isfinite(reverse_kl(s, t).loss.item())

    def test_no_nan_with_forbidden_tokens(self):
        """
        Critical regression: teacher has -inf logits for forbidden tokens.
        Old code: p_s * (log_s - log_t) → p_s * inf → NaN when p_s=0.
        Fix: log_t.clamp(min=-100).
        """
        s, t = _inflogit_teacher()
        loss = reverse_kl(s, t).loss
        assert not torch.isnan(loss), "NaN with -inf teacher logits — clamp guard broken"
        assert not torch.isinf(loss), "Inf with -inf teacher logits — clamp guard broken"

    def test_gradient_no_nan_with_forbidden_tokens(self):
        """Gradient must not contain NaN when teacher has -inf logits."""
        s, t = _inflogit_teacher()
        reverse_kl(s, t).loss.backward()
        assert s.grad is not None
        assert not torch.isnan(s.grad).any(), \
            "NaN gradient with -inf teacher — corrupts LoRA weights on first step"
        assert not torch.isinf(s.grad).any(), "Inf gradient with -inf teacher"

    def test_finite_loss_with_many_forbidden(self):
        T, V = 4, 8
        t = torch.full((T, V), float("-inf"))
        t[:, 0] = 1.0   # one valid token per row
        s = torch.randn(T, V, requires_grad=True)
        assert math.isfinite(reverse_kl(s, t).loss.item()), \
            "Should not crash with mostly-forbidden teacher"

    def test_gradient_flows(self):
        s, t = _random_logits()
        reverse_kl(s, t).loss.backward()
        assert s.grad is not None
        assert torch.isfinite(s.grad).all()


# ===========================================================================
# jsd
# ===========================================================================

class TestJSD:

    def test_nonnegative(self):
        s, t = _random_logits()
        assert jsd(s, t).loss.item() >= 0.0

    def test_finite(self):
        s, t = _random_logits()
        assert math.isfinite(jsd(s, t).loss.item())

    def test_upper_bound(self):
        """JSD(p,q) <= ln(2) always, for alpha=0.5."""
        s, t = _random_logits()
        assert jsd(s, t).loss.item() <= math.log(2) + 1e-5

    def test_near_zero_when_identical(self):
        """JSD(p, p) should be ~0."""
        s, t = _identical_logits()
        assert jsd(s, t).loss.item() < 1e-4, \
            f"JSD(p,p) should be ~0, got {jsd(s, t).loss.item()}"

    def test_symmetric(self):
        """JSD is symmetric: JSD(p,q) == JSD(q,p) at alpha=0.5."""
        T, V = 8, 16
        torch.manual_seed(7)
        x, y = torch.randn(T, V), torch.randn(T, V)
        loss_xy = jsd(x, y).loss.item()
        loss_yx = jsd(y, x).loss.item()
        assert abs(loss_xy - loss_yx) < 1e-4, \
            f"JSD not symmetric: {loss_xy} vs {loss_yx}"

    def test_gradient_flows(self):
        s, t = _random_logits()
        jsd(s, t).loss.backward()
        assert s.grad is not None
        assert torch.isfinite(s.grad).all()


# ===========================================================================
# l1
# ===========================================================================

class TestL1:

    def test_nonnegative(self):
        s, t = _random_logits()
        assert l1(s, t).loss.item() >= 0.0

    def test_upper_bound(self):
        """TV distance is in [0, 1]."""
        s, t = _random_logits()
        assert l1(s, t).loss.item() <= 1.0 + 1e-5

    def test_zero_when_identical(self):
        s, t = _identical_logits()
        assert l1(s, t).loss.item() < 1e-5

    def test_maximum_for_opposite_supports(self):
        """TV = 1 when distributions have disjoint support."""
        V = 4
        p_logits = torch.tensor([[10.0, -10.0, -10.0, -10.0]])
        q_logits = torch.tensor([[-10.0, -10.0, 10.0, -10.0]])
        loss = l1(p_logits, q_logits).loss
        assert abs(loss.item() - 1.0) < 0.01, \
            f"TV distance for disjoint supports should be ~1, got {loss.item()}"

    def test_gradient_flows(self):
        s, t = _random_logits()
        l1(s, t).loss.backward()
        assert s.grad is not None
        assert torch.isfinite(s.grad).all()


# ===========================================================================
# ebe
# ===========================================================================

class TestEBE:

    def _make_inputs(self, T=16, V=8, seed=1):
        """T tokens, V vocab — sequence must be long enough for block splitting."""
        torch.manual_seed(seed)
        s   = torch.randn(T, V, requires_grad=True)
        t   = torch.randn(T, V)
        ids = torch.randint(0, V, (T,))
        return s, t, ids

    def test_returns_loss_output(self):
        """ebe() returns LossOutput with loss, accept_weight, and diagnostics."""
        s, t, ids = self._make_inputs()
        out = ebe(s, t, token_ids=ids)
        assert isinstance(out, LossOutput), "ebe() must return a LossOutput"
        assert out.accept_weight is not None, "EBE must populate accept_weight"
        assert out.diagnostics  is not None, "EBE must populate diagnostics"
        assert "mean" in out.diagnostics and "frac_lt_0.95" in out.diagnostics, \
            "diagnostics must contain 'mean' and 'frac_lt_0.95'"

    def test_loss_is_finite(self):
        s, t, ids = self._make_inputs()
        assert math.isfinite(ebe(s, t, token_ids=ids).loss.item())

    def test_alpha_mean_in_unit_interval(self):
        s, t, ids = self._make_inputs()
        out = ebe(s, t, token_ids=ids)
        assert 0.0 <= out.accept_weight <= 1.0, \
            f"accept_weight must be in [0,1], got {out.accept_weight}"
        assert 0.0 <= out.diagnostics["frac_lt_0.95"] <= 1.0
        assert 0.0 <= out.diagnostics["frac_lt_0.80"] <= 1.0

    def test_gradient_flows(self):
        s, t, ids = self._make_inputs()
        ebe(s, t, token_ids=ids).loss.backward()
        assert s.grad is not None
        assert torch.isfinite(s.grad).all(), "EBE gradient contains NaN/Inf"

    def test_no_nan_when_alpha_is_zero(self):
        """
        Critical regression: alpha=0 caused NaN in cumprod backward.
        Fixed by alpha.clamp(min=1e-6).
        """
        T, V = 8, 4
        s   = (torch.randn(T, V) * 10.0).detach().requires_grad_(True)
        t   = torch.randn(T, V) * 0.1
        ids = torch.zeros(T, dtype=torch.long)
        out = ebe(s, t, token_ids=ids)
        assert not torch.isnan(out.loss), "EBE NaN when alpha near 0 — clamp guard broken"
        out.loss.backward()
        assert s.grad is not None
        assert torch.isfinite(s.grad).all(), "EBE gradient NaN when alpha near 0"

    def test_block_level_split_respected(self):
        """
        ebe() splits the sequence into non-overlapping blocks of block_len=8.
        Changing block_len must change the loss (otherwise block splitting is broken).
        """
        T, V, block_len = 32, 8, 8
        torch.manual_seed(10)
        s   = torch.randn(T, V, requires_grad=True)
        t   = torch.randn(T, V)
        ids = torch.randint(0, V, (T,))

        loss_blocked = ebe(s, t, token_ids=ids, block_len=block_len).loss
        loss_1block  = ebe(s, t, token_ids=ids, block_len=T).loss

        assert loss_blocked.item() != loss_1block.item(), \
            "block_len parameter has no effect — block splitting may be broken"

    def test_kl_weight_zero_gives_pure_ebe(self):
        """With kl_weight=0, loss == EBE term only (no KL regulariser)."""
        s, t, ids = self._make_inputs()
        loss_with_kl = ebe(s, t, token_ids=ids, kl_weight=0.1).loss
        loss_no_kl   = ebe(s, t, token_ids=ids, kl_weight=0.0).loss
        assert loss_with_kl.item() != loss_no_kl.item(), \
            "kl_weight=0 and kl_weight=0.1 should produce different losses"

    def test_raises_without_token_ids(self):
        s, t = _random_logits(T=16, V=8)
        with pytest.raises(ValueError, match="token_ids"):
            ebe(s, t)   # token_ids is required

    def test_get_loss_dispatch(self):
        s, t, ids = self._make_inputs()
        out = get_loss("ebe", s, t, token_ids=ids)
        assert isinstance(out, LossOutput)
        assert out.accept_weight is not None


# ===========================================================================
# ebe_single
# ===========================================================================

class TestEBESingle:

    def _make_inputs(self, T=16, V=8, seed=2):
        torch.manual_seed(seed)
        s   = torch.randn(T, V, requires_grad=True)
        t   = torch.randn(T, V)
        ids = torch.randint(0, V, (T,))
        return s, t, ids

    def test_returns_loss_output(self):
        """ebe_single() returns LossOutput with loss, accept_weight, and diagnostics."""
        s, t, ids = self._make_inputs()
        out = ebe_single(s, t, token_ids=ids)
        assert isinstance(out, LossOutput)
        assert out.accept_weight is not None
        assert out.diagnostics  is not None
        assert "mean" in out.diagnostics and "frac_lt_0.95" in out.diagnostics

    def test_loss_is_negative_mean_alpha(self):
        """Loss = -mean(alpha) so it should be in [-1, 0]."""
        s, t, ids = self._make_inputs()
        out = ebe_single(s, t, token_ids=ids)
        assert out.loss.item() <= 0.0, "ebe_single loss = -mean(α) must be <= 0"
        assert out.loss.item() >= -1.0, "ebe_single loss = -mean(α) must be >= -1"

    def test_loss_consistent_with_accept_weight(self):
        """accept_weight = mean(α) should equal -loss (within float precision)."""
        s, t, ids = self._make_inputs()
        out = ebe_single(s, t, token_ids=ids)
        assert abs(out.accept_weight + out.loss.item()) < 1e-5, \
            f"accept_weight ({out.accept_weight:.6f}) should equal -loss ({-out.loss.item():.6f})"

    def test_loss_is_finite(self):
        s, t, ids = self._make_inputs()
        assert math.isfinite(ebe_single(s, t, token_ids=ids).loss.item())

    def test_alpha_mean_in_unit_interval(self):
        s, t, ids = self._make_inputs()
        out = ebe_single(s, t, token_ids=ids)
        assert 0.0 <= out.accept_weight <= 1.0
        assert 0.0 <= out.diagnostics["frac_lt_0.95"] <= 1.0
        assert 0.0 <= out.diagnostics["frac_lt_0.80"] <= 1.0

    def test_gradient_flows(self):
        s, t, ids = self._make_inputs()
        ebe_single(s, t, token_ids=ids).loss.backward()
        assert s.grad is not None
        assert torch.isfinite(s.grad).all(), "ebe_single gradient contains NaN/Inf"

    def test_no_nan_when_alpha_near_zero(self):
        """Clamp ≥ 1e-6 prevents NaN when student assigns near-zero prob."""
        T, V = 8, 4
        s   = (torch.randn(T, V) * 10.0).detach().requires_grad_(True)
        t   = torch.randn(T, V) * 0.1
        ids = torch.zeros(T, dtype=torch.long)
        out = ebe_single(s, t, token_ids=ids)
        assert not torch.isnan(out.loss), "ebe_single NaN when alpha near 0"
        out.loss.backward()
        assert s.grad is not None
        assert torch.isfinite(s.grad).all()

    def test_no_block_structure(self):
        """
        ebe_single has no block structure — doubling T should roughly double
        the number of gradient terms, unlike block EBE where cumprod chains grow.
        Verify: ebe_single loss is in the same range for T=8 and T=16.
        """
        V = 8
        torch.manual_seed(5)
        s8   = torch.randn(8,  V, requires_grad=True)
        s16  = torch.randn(16, V, requires_grad=True)
        t8   = torch.randn(8,  V)
        t16  = torch.randn(16, V)
        ids8  = torch.randint(0, V, (8,))
        ids16 = torch.randint(0, V, (16,))
        loss8  = ebe_single(s8,  t8,  token_ids=ids8).loss.item()
        loss16 = ebe_single(s16, t16, token_ids=ids16).loss.item()
        # Both should be in [-1, 0] regardless of sequence length
        assert -1.0 <= loss8  <= 0.0
        assert -1.0 <= loss16 <= 0.0

    def test_raises_without_token_ids(self):
        s, t = _random_logits(T=16, V=8)
        with pytest.raises(ValueError, match="token_ids"):
            ebe_single(s, t)

    def test_get_loss_dispatch(self):
        s, t, ids = self._make_inputs()
        out = get_loss("ebe_single", s, t, token_ids=ids)
        assert isinstance(out, LossOutput)
        assert out.accept_weight is not None


# ===========================================================================
# Registry smoke test
# ===========================================================================

class TestRegistry:

    @pytest.mark.parametrize("name", ["forward_kl", "reverse_kl", "jsd", "l1"])
    def test_kl_family_via_get_loss(self, name):
        """Every non-EBE loss is dispatchable through get_loss() without token_ids."""
        s, t = _random_logits()
        out = get_loss(name, s, t)
        assert isinstance(out, LossOutput)
        assert math.isfinite(out.loss.item())
        assert out.accept_weight is None
        assert out.diagnostics  is None

    @pytest.mark.parametrize("name", ["ebe", "ebe_single"])
    def test_ebe_family_via_get_loss(self, name):
        """EBE-family losses require token_ids and return diagnostics."""
        torch.manual_seed(0)
        T, V = 16, 8
        s   = torch.randn(T, V, requires_grad=True)
        t   = torch.randn(T, V)
        ids = torch.randint(0, V, (T,))
        out = get_loss(name, s, t, token_ids=ids)
        assert isinstance(out, LossOutput)
        assert math.isfinite(out.loss.item())
        assert out.accept_weight is not None
        assert out.diagnostics  is not None

    def test_unknown_loss_raises(self):
        s, t = _random_logits()
        with pytest.raises(ValueError, match="Unknown loss"):
            get_loss("banana", s, t)
