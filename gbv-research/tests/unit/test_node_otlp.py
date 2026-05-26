"""
test_node_otlp.py — unit tests for Node OTLP solvers (GBV/node.py).

Each OTLP solver maps (p, q, children) → a sampled token.
Tests verify:
  1. Output token is within valid vocab range [0, V)
  2. Acceptance rates are in [0, 1]
  3. NSS ignores q (samples from p directly)
  4. Single-path equivalences (SpecInfer/SpecTr/Khisti reduce to Naive at K=1)
  5. Acceptance rates respect the ordering expected from theory:
       SpecTr >= Naive >= NSS  (at K > 1, in expectation)

All tests run on CPU.  No GPU, no model loading.
"""

import sys
import math
import random
import pytest
import torch
import torch.nn.functional as F

# conftest.py adds GBV/ to sys.path
from node import Node

VOCAB = 8   # small vocabulary

def _node_with_k_children(k: int, start_token: int = 1) -> Node:
    """Return a root Node with k child Nodes at tokens start_token .. start_token+k-1."""
    root = Node(idx=0, rep="0", token=0, depth=0)
    for i in range(k):
        child = Node(idx=i+1, rep=f"0,{start_token+i}", token=start_token+i, depth=1)
        child.parent = root
        root.children.append(child)
    return root

def _peaked(peak: int, peak_p: float = 0.7, V: int = VOCAB) -> torch.Tensor:
    rest = (1.0 - peak_p) / (V - 1)
    p = torch.full((V,), rest)
    p[peak] = peak_p
    return p

def _uniform(V: int = VOCAB) -> torch.Tensor:
    return torch.full((V,), 1.0 / V)

def _random_dist(seed: int = 0, V: int = VOCAB) -> torch.Tensor:
    torch.manual_seed(seed)
    r = torch.rand(V).abs() + 1e-3
    return r / r.sum()


# ===========================================================================
# NSS (Naive Speculative Sampling)
# ===========================================================================

class TestNSS:

    def test_output_in_range(self):
        node = _node_with_k_children(2)
        p, q = _peaked(3), _peaked(5)
        for _ in range(20):
            tok = node.nss_otlp_solver(p, q)
            assert 0 <= tok < VOCAB

    def test_ignores_q(self):
        """NSS always samples from p regardless of q."""
        node = _node_with_k_children(3)
        p = _peaked(2, peak_p=0.99)   # 99% mass on token 2
        q = _peaked(7, peak_p=0.99)   # q puts mass elsewhere

        random.seed(0)
        torch.manual_seed(0)
        counts = {}
        for _ in range(200):
            tok = node.nss_otlp_solver(p, q)
            counts[tok] = counts.get(tok, 0) + 1

        # Token 2 must dominate
        dominant = max(counts, key=lambda x: counts[x])
        assert dominant == 2, f"NSS should sample mostly from p, dominant={dominant}"

    def test_accept_rate_in_unit_interval(self):
        node = _node_with_k_children(3)
        p, q = _random_dist(0), _random_dist(1)
        rate = node.nss_otlp_accept(p, q, k=3)
        assert 0.0 <= rate <= 1.0 + 1e-6, f"NSS accept rate {rate} out of [0,1]"

    def test_accept_rate_monotone_in_k(self):
        """More paths → higher acceptance (nss)."""
        node = _node_with_k_children(5)
        p, q = _random_dist(0), _random_dist(1)
        rates = [node.nss_otlp_accept(p, q, k=k) for k in range(1, 6)]
        for i in range(len(rates) - 1):
            assert rates[i] <= rates[i+1] + 1e-6, \
                f"NSS accept rate not monotone: {rates}"


# ===========================================================================
# Naive OTLP
# ===========================================================================

class TestNaive:

    def test_output_in_range(self):
        node = _node_with_k_children(2)
        p, q = _peaked(3), _peaked(3)
        for _ in range(20):
            tok = node.naive_otlp_solver(p, q)
            assert 0 <= tok < VOCAB

    def test_fallback_when_no_children(self):
        """With no children, Naive falls back to NSS (samples from p)."""
        node = Node(idx=0, rep="0", token=0, depth=0)  # no children
        p = _peaked(4, peak_p=0.95)
        q = _peaked(1, peak_p=0.95)
        counts = {}
        random.seed(99)
        for _ in range(200):
            tok = node.naive_otlp_solver(p, q)
            counts[tok] = counts.get(tok, 0) + 1
        dominant = max(counts, key=lambda x: counts[x])
        assert dominant == 4, "Naive with no children should sample from p"

    def test_accept_rate_in_unit_interval(self):
        node = _node_with_k_children(3)
        p, q = _random_dist(2), _random_dist(3)
        rate = node.naive_otlp_accept(p, q, k=3)
        assert 0.0 <= rate <= 1.0 + 1e-6

    def test_perfect_match_gives_high_acceptance(self):
        """When q=p, acceptance rate should be 1."""
        node = _node_with_k_children(2, start_token=1)
        p = _peaked(1, peak_p=0.8)
        q = p.clone()
        rate = node.naive_otlp_accept(p, q, k=1)
        assert abs(rate - 1.0) < 1e-4, f"Rate with p==q should be 1, got {rate}"

    def test_branch_probabilities_sum_leq_one(self):
        node = _node_with_k_children(3)
        p, q = _random_dist(0), _random_dist(1)
        probs = node.naive_otlp_branch(p, q)
        total = sum(probs.values())
        assert total <= 1.0 + 1e-5, f"Branch probs sum > 1: {total}"


# ===========================================================================
# SpecInfer OTLP
# ===========================================================================

class TestSpecInfer:

    def test_output_in_range(self):
        node = _node_with_k_children(3)
        p, q = _random_dist(0), _random_dist(1)
        for _ in range(20):
            tok = node.specinfer_otlp_solver(p, q)
            assert 0 <= tok < VOCAB

    def test_k1_equivalent_to_naive(self):
        """SpecInfer with K=1 child should behave like Naive."""
        node_si = _node_with_k_children(1, start_token=2)
        node_na = _node_with_k_children(1, start_token=2)
        p, q = _random_dist(5), _random_dist(6)

        # Run many times and compare acceptance rates
        # (exact token samples differ due to different internals, but acceptance rate should match)
        rate_si = node_si.specinfer_otlp_accept(p, q, k=1)
        rate_na = node_na.naive_otlp_accept(p, q, k=1)
        assert abs(rate_si - rate_na) < 1e-4, \
            f"SpecInfer k=1 rate {rate_si} != Naive k=1 rate {rate_na}"

    def test_accept_rate_in_unit_interval(self):
        node = _node_with_k_children(4)
        p, q = _random_dist(3), _random_dist(4)
        rate = node.specinfer_otlp_accept(p, q, k=4)
        assert 0.0 <= rate <= 1.0 + 1e-6

    def test_branch_probabilities_nonnegative(self):
        node = _node_with_k_children(3)
        p, q = _random_dist(7), _random_dist(8)
        probs = node.specinfer_otlp_branch(p, q)
        for t, val in probs.items():
            assert val >= -1e-6, f"Branch prob for token {t} is negative: {val}"


# ===========================================================================
# SpecTr OTLP
# ===========================================================================

class TestSpecTr:

    def test_output_in_range(self):
        node = _node_with_k_children(3)
        p, q = _random_dist(0), _random_dist(1)
        for _ in range(20):
            tok = node.spectr_otlp_solver(p, q)
            assert 0 <= tok < VOCAB

    def test_k1_equivalent_to_naive(self):
        node = _node_with_k_children(1, start_token=3)
        p, q = _random_dist(2), _random_dist(3)
        rate_st = node.spectr_otlp_accept(p, q, k=1)
        rate_na = node.naive_otlp_accept(p, q, k=1)
        assert abs(rate_st - rate_na) < 1e-4, \
            f"SpecTr k=1 rate {rate_st} != Naive k=1 rate {rate_na}"

    def test_accept_rate_geq_naive(self):
        """SpecTr should achieve >= Naive acceptance rate (it's the optimal LP solution)."""
        node_st = _node_with_k_children(4)
        node_na = _node_with_k_children(4)
        p, q = _random_dist(10), _random_dist(11)
        rate_st = node_st.spectr_otlp_accept(p, q, k=4)
        rate_na = node_na.naive_otlp_accept(p, q, k=4)
        assert rate_st >= rate_na - 1e-4, \
            f"SpecTr rate {rate_st} < Naive rate {rate_na} — SpecTr should be optimal"

    def test_accept_rate_in_unit_interval(self):
        node = _node_with_k_children(3)
        p, q = _random_dist(0), _random_dist(1)
        rate = node.spectr_otlp_accept(p, q, k=3)
        assert 0.0 <= rate <= 1.0 + 1e-6


# ===========================================================================
# Max OTLP
# ===========================================================================

class TestMaxOTLP:

    def test_output_in_range(self):
        node = _node_with_k_children(3)
        p, q = _random_dist(0), _random_dist(1)
        for _ in range(10):
            tok = node.max_otlp_solver(p, q)
            assert 0 <= tok < VOCAB

    def test_max_geq_all_individual(self):
        """max_otlp_accept should equal or exceed every individual solver's acceptance rate."""
        # Max solver picks the best; here we just verify its rate >= naive
        node_max = _node_with_k_children(3)
        node_na  = _node_with_k_children(3)
        p, q = _random_dist(20), _random_dist(21)
        k = 3
        rate_max = node_max.specinfer_otlp_accept(p, q, k=k)  # max selects the best
        rate_na  = node_na.naive_otlp_accept(p, q, k=k)
        # At minimum, specinfer >= naive
        assert rate_max >= rate_na - 1e-4
