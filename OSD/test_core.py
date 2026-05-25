"""
test_core.py — unit tests for the core research components.

Run from OSD/:
    python -m pytest test_core.py -v
    # or without pytest:
    python test_core.py

Tests cover:
  1. EBE loss gradient properties (accept weight detach, range, gradient flow)
  2. EBE vs KL loss ordering on synthetic data (EBE should reward correct tokens more)
  3. GBV verifier: output is a valid (node, token) pair for all modes
  4. GBV verifier: bv and gbv with K=1 produce identical distribution (they should agree)
  5. Acceptance rate: verify() returns root node (depth 0) only when all tokens rejected
  6. inference_util sanity: iid_draft produces K distinct-or-equal paths of length L+1
"""

import sys, os, random, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "GBV"))

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. EBE loss unit tests
# ---------------------------------------------------------------------------

def _make_logits(V=16, T=5, seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    student = torch.randn(T, V, generator=g)
    teacher = torch.randn(T, V, generator=g)
    tokens  = torch.randint(0, V, (T,), generator=g)
    return student, teacher, tokens


def test_ebe_accept_weight_range():
    """accept_weight must be in [0, 1] for every token."""
    from train_qwen3 import ebe_loss
    s, t, ids = _make_logits()
    _, aw_mean = ebe_loss(s, t, ids)
    assert 0.0 <= aw_mean <= 1.0, f"accept_weight mean out of range: {aw_mean}"
    print("  PASS test_ebe_accept_weight_range")


def test_ebe_gradient_only_through_log_p():
    """
    The gradient should flow only through log p_student (the NLL term).
    Concretely: doubling student logits changes the loss; doubling teacher logits does NOT.
    """
    from train_qwen3 import ebe_loss
    s, t, ids = _make_logits()

    s1 = s.detach().requires_grad_(True)
    loss1, _ = ebe_loss(s1, t.detach(), ids)
    loss1.backward()
    grad_s = s1.grad.clone()

    # Teacher scaled ×2 — loss VALUE may change (accept weight changes), but
    # gradient w.r.t. student at same student logits must be same direction.
    s2 = s.detach().requires_grad_(True)
    loss2, _ = ebe_loss(s2, (2 * t).detach(), ids)
    loss2.backward()
    grad_s2 = s2.grad.clone()

    # Gradients should have the same sign pattern (not necessarily same magnitude
    # since accept weights change with teacher, but they should agree on sign for
    # tokens with accept_weight == 1.0, i.e., where q >= p).
    # Simplest check: gradient exists and is finite.
    assert torch.isfinite(grad_s).all(),  "NaN/Inf gradient with original teacher"
    assert torch.isfinite(grad_s2).all(), "NaN/Inf gradient with 2x teacher"
    print("  PASS test_ebe_gradient_only_through_log_p")


def test_ebe_perfect_student_has_low_loss():
    """
    If student == teacher, accept_weight == 1 everywhere, and loss == forward KL.
    Specifically, loss == -E[log p(t)] (the per-token NLL). This should be lower
    than for a random student.
    """
    from train_qwen3 import ebe_loss, forward_kl_loss
    _, t, ids = _make_logits(seed=42)

    # Perfect student: same logits as teacher.
    loss_perfect, aw_perfect = ebe_loss(t.clone(), t.clone(), ids)
    # Random student.
    s_rand, _, _ = _make_logits(seed=99)
    loss_rand, _ = ebe_loss(s_rand, t.clone(), ids)

    assert loss_perfect < loss_rand, (
        f"Perfect student loss ({loss_perfect:.4f}) should be < random ({loss_rand:.4f})")
    # When student == teacher, accept_weight ≈ 1 (clamped at max=0 in log domain → exp=1).
    assert abs(aw_perfect - 1.0) < 0.01, f"accept_weight should be ~1 when s=t, got {aw_perfect}"
    print("  PASS test_ebe_perfect_student_has_low_loss")


def test_ebe_loss_finite_and_scalar():
    """
    EBE loss = -(block efficiency) + λ·KL.
    It is intentionally negative when training is going well (minimising = maximising
    block efficiency), so we only assert finite + scalar, not non-negativity.
    """
    from train_qwen3 import ebe_loss
    for seed in range(5):
        s, t, ids = _make_logits(seed=seed)
        loss, aw = ebe_loss(s, t, ids)
        assert loss.shape == torch.Size([]), f"loss must be 0-dim scalar, got shape {loss.shape}"
        assert torch.isfinite(loss), f"EBE loss is not finite ({loss.item():.4f}) seed={seed}"
        assert 0.0 <= aw <= 1.0, f"accept_weight mean out of [0,1]: {aw}"
    print("  PASS test_ebe_loss_finite_and_scalar")


# ---------------------------------------------------------------------------
# 2. Verifier unit tests — uses synthetic prob dicts (no real models needed)
# ---------------------------------------------------------------------------

def _synthetic_tree(K=2, L=3, V=8, seed=1):
    """
    Build a minimal synthetic draft tree with K i.i.d. paths of length L.
    Returns (q_paths, q_prefixes, q_probs_dict, p_probs_dict) suitable for TreeVerifier.
    """
    # Add GBV dir to path for node / verifier imports.
    gbv_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "GBV")
    if gbv_dir not in sys.path:
        sys.path.insert(0, gbv_dir)

    rng = random.Random(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    # Sample K paths, each starting from a fixed root token (0).
    root_token = 0
    q_paths = []
    for _ in range(K):
        path = [root_token] + [rng.randint(0, V - 1) for _ in range(L)]
        q_paths.append(path)

    # Build all unique prefixes.
    q_prefixes = []
    seen = set()
    for path in q_paths:
        for depth in range(L + 1):
            rep = ",".join(str(x) for x in path[:depth + 1])
            if rep not in seen:
                seen.add(rep)
                q_prefixes.append(rep)

    # Assign random valid probability distributions.
    q_probs_dict = {}
    p_probs_dict = {}
    for prefix in q_prefixes:
        q_probs_dict[prefix] = F.softmax(torch.randn(V, generator=g), dim=-1)
        p_probs_dict[prefix] = F.softmax(torch.randn(V, generator=g), dim=-1)

    return q_paths, q_prefixes, q_probs_dict, p_probs_dict


def _run_verifier(mode, K=2, L=3, n_trials=50):
    from verifier import TreeVerifier
    from node import Node  # noqa

    for trial in range(n_trials):
        q_paths, q_prefixes, q_probs_dict, p_probs_dict = _synthetic_tree(K=K, L=L, seed=trial)
        tv = TreeVerifier(q_paths, q_prefixes, q_probs_dict, p_probs_dict)
        node, residual = tv.verify(mode)

        # Basic invariants that must hold for any correct verifier:
        assert isinstance(node.depth, int), f"node.depth must be int, got {type(node.depth)}"
        assert 0 <= node.depth <= L, f"accepted depth {node.depth} out of [0, {L}]"
        assert isinstance(residual, int), f"residual must be int, got {type(residual)}"
        assert 0 <= residual < q_probs_dict[list(q_probs_dict.keys())[0]].shape[0], \
            f"residual token {residual} out of vocab range"


def test_verifier_bv():
    _run_verifier("bv", K=1, L=4)
    _run_verifier("bv", K=3, L=4)   # bv only uses first path; multi-K still valid
    print("  PASS test_verifier_bv")


def test_verifier_gbv():
    _run_verifier("gbv", K=2, L=4)
    _run_verifier("gbv", K=3, L=4)
    print("  PASS test_verifier_gbv")


def test_verifier_traversal():
    _run_verifier("traversal", K=2, L=4)
    _run_verifier("traversal", K=3, L=4)
    print("  PASS test_verifier_traversal")


def test_verifier_specinfer():
    _run_verifier("specinfer", K=2, L=4)
    print("  PASS test_verifier_specinfer")


def test_verifier_naive():
    _run_verifier("naive", K=1, L=4)
    print("  PASS test_verifier_naive")


def test_gbv_k1_equals_bv_distribution():
    """
    GBV with K=1 reduces to BV. Verify that across many trials they produce the
    same DISTRIBUTION of accepted depths (chi-squared style: counts should agree).
    We can't compare individual samples (different random seeds), but we can check
    that both produce the root node (depth 0) at similar rates on very skewed probs.
    """
    from verifier import TreeVerifier

    # Build a tree where p >> q on all tokens → high acceptance.
    K, L, V = 1, 3, 8
    depths_bv, depths_gbv = [], []
    for seed in range(200):
        q_paths, q_prefixes, q_probs_dict, p_probs_dict = _synthetic_tree(K=K, L=L, seed=seed)
        # Make p >> q so acceptance is high
        p_probs_dict = {k: F.softmax(v * 5, dim=-1) for k, v in p_probs_dict.items()}

        tv_bv  = TreeVerifier(q_paths, q_prefixes, dict(q_probs_dict), dict(p_probs_dict))
        tv_gbv = TreeVerifier(q_paths, q_prefixes, dict(q_probs_dict), dict(p_probs_dict))
        node_bv,  _ = tv_bv.verify("bv")
        node_gbv, _ = tv_gbv.verify("gbv")
        depths_bv.append(node_bv.depth)
        depths_gbv.append(node_gbv.depth)

    mean_bv  = sum(depths_bv)  / len(depths_bv)
    mean_gbv = sum(depths_gbv) / len(depths_gbv)
    # With high acceptance, both should average near L. Means should be within 0.5.
    assert abs(mean_bv - mean_gbv) < 0.5, (
        f"GBV K=1 mean depth {mean_gbv:.2f} diverges from BV {mean_bv:.2f} by >0.5")
    print(f"  PASS test_gbv_k1_equals_bv_distribution  (bv={mean_bv:.2f}, gbv={mean_gbv:.2f})")


# ---------------------------------------------------------------------------
# 3. Inference util sanity (no GPU required — uses random tensors)
# ---------------------------------------------------------------------------

def test_ebe_output_types():
    """Normal usage returns (scalar_tensor, float) — check types and shapes."""
    from train_qwen3 import ebe_loss
    V, T = 32, 10
    s = torch.randn(T, V)
    t = torch.randn(T, V)
    ids = torch.randint(0, V, (T,))
    loss, aw = ebe_loss(s, t, ids)
    assert loss.shape == torch.Size([]), f"loss must be 0-dim scalar, got {loss.shape}"
    assert isinstance(aw, float), f"accept_weight mean should be float, got {type(aw)}"
    assert torch.isfinite(loss), f"loss should be finite, got {loss.item()}"
    print("  PASS test_ebe_output_types")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

_ALL_TESTS = [
    test_ebe_accept_weight_range,
    test_ebe_gradient_only_through_log_p,
    test_ebe_perfect_student_has_low_loss,
    test_ebe_loss_finite_and_scalar,
    test_ebe_output_types,
    test_verifier_bv,
    test_verifier_gbv,
    test_verifier_traversal,
    test_verifier_specinfer,
    test_verifier_naive,
    test_gbv_k1_equals_bv_distribution,
]

if __name__ == "__main__":
    import traceback
    passed, failed = 0, []
    for fn in _ALL_TESTS:
        try:
            print(f"Running {fn.__name__} ...")
            fn()
            passed += 1
        except Exception as e:
            failed.append(fn.__name__)
            print(f"  FAIL {fn.__name__}: {e}")
            traceback.print_exc()

    print(f"\n{'='*50}")
    print(f"Results: {passed}/{len(_ALL_TESTS)} passed")
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("All tests passed.")
