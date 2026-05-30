"""
tree_harness.py — glue between the toy models and the REAL code.

Three jobs:

1. Make `distillspec_gbv.losses.tree_losses` and
   `distillspec_gbv.verifiers.otlp_registry` importable from anywhere
   (adds gbv-research/algorithms to sys.path).

2. Monte-Carlo block-efficiency estimator that drives the *real* TreeVerifier.
   Block efficiency (BE) is the headline metric in runner.py:

       BE = (new tokens committed) / (target calls)

   For a single verify() call the verifier returns (ver_node, residual_token).
   The accepted draft prefix has length `ver_node.depth` (depths 1..d), and one
   bonus/residual token is always emitted, so the block contributes

       block_len = ver_node.depth + 1

   new tokens. Averaging block_len over many trees is our BE estimate — the
   exact quantity each tree loss is trying to maximise.

   IMPORTANT verifier hygiene (learned from tests/unit/test_verifiers.py):
     * traversal_verify MUTATES its p/q dicts and node objects → every trial
       must use a FRESH TreeVerifier built from FRESH tensors.
     * Node carries CLASS-LEVEL caches keyed by id(p); ids get recycled across
       trials, so we clear them before every trial.
"""

from __future__ import annotations

import os
import sys

# ── make the real package importable ───────────────────────────────────────
_HERE        = os.path.dirname(os.path.abspath(__file__))           # debug/tree_losses
_GBV_RESEARCH = os.path.dirname(os.path.dirname(_HERE))             # gbv-research/
_ALGORITHMS  = os.path.join(_GBV_RESEARCH, "algorithms")            # gbv-research/algorithms
for _p in (_ALGORITHMS, _GBV_RESEARCH):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from distillspec_gbv.losses.tree_losses import (  # noqa: E402
    compute_tree_loss, TREE_LOSS_NAMES,
)
from distillspec_gbv.losses import get_loss as _get_loss, LOSS_REGISTRY  # noqa: E402
from distillspec_gbv.verifiers.otlp_registry import TreeVerifier  # noqa: E402
from distillspec_gbv.verifiers.tree import Node                   # noqa: E402

from tiny_models import TinyMarkovModel, sample_tree, build_dicts, sample_paths  # noqa: E402


# ---------------------------------------------------------------------------
# Shared flat-loss helper  (used by web_debugger and visual_debugger)
# ---------------------------------------------------------------------------

FLAT_LOSS_NAMES = frozenset(LOSS_REGISTRY.keys())
# Losses that select a single token (on-policy sample) rather than using the
# full distribution; get_loss raises if token_ids is not supplied.
_NEEDS_TOKEN_IDS = frozenset({"ebe", "ebe_single"})


def flat_loss(loss_name: str, s_logit: torch.Tensor, t_logit: torch.Tensor,
              generator=None):
    """Call the real get_loss, auto-sampling token_ids for ebe/ebe_single.

    s_logit / t_logit: [1, V] tensors (student with grad, teacher detached).
    For ebe/ebe_single the next token is sampled on-policy from the student.
    """
    if loss_name in _NEEDS_TOKEN_IDS:
        with torch.no_grad():
            probs = F.softmax(s_logit.detach(), dim=-1)
            tok = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)
        return _get_loss(loss_name, s_logit, t_logit, token_ids=tok)
    return _get_loss(loss_name, s_logit, t_logit)


# Loss name  ↔  verifier name  (the cross-pair hypothesis: train L_V, eval V)
LOSS_TO_VERIFIER = {
    "bv_tree":        "bv",
    "gbv_tree":       "gbv",
    "traversal_tree": "traversal",
    "naive_tree":     "naive",
    "nss_tree":       "nss",
    "specinfer_tree": "specinfer",
    "spectr_tree":    "spectr",
    "khisti_tree":    "khisti",
    # generic divergences have no single matching verifier; default to gbv (this work)
    "kl_tree":        "gbv",
    "rev_kl_tree":    "gbv",
    "jsd_tree":       "gbv",
    "ebe_tree":       "naive",   # exact for K=1; approx for K>1 (ignores multi-branch correction)
}

ALL_VERIFIERS = ["naive", "nss", "bv", "specinfer", "spectr", "traversal", "gbv"]
# khisti excluded from sweeps by default — its scipy LP solver is ~0.5 s/call.


def _clear_node_caches() -> None:
    """Wipe the class-level OTLP caches between trials (ids get recycled)."""
    Node.naive_cache.clear()
    Node.spectr_cache.clear()
    Node.specinfer_cache.clear()
    Node.khisti_cache.clear()


def block_length_once(q_paths, q_prefixes, q_probs_dict, p_probs_dict, mode: str) -> int:
    """Run ONE verify() on a fresh tree and return block_len = ver_node.depth + 1."""
    _clear_node_caches()
    tv = TreeVerifier(q_paths, q_prefixes, q_probs_dict, p_probs_dict)
    ver_node, _res = tv.verify(mode)
    return ver_node.depth + 1


def estimate_block_efficiency(
    student: TinyMarkovModel,
    teacher: TinyMarkovModel,
    mode: str,
    tokens,
    K: int,
    L: int,
    temp: float = 1.0,
    n_trials: int = 300,
    seed: int = 0,
    student_temp: float = None,
    teacher_temp: float = None,
) -> float:
    """Monte-Carlo BE for `mode` using the real verifier on the toy models.

    Each trial: pick a pending token, sample a fresh detached tree, run the
    verifier once, record block_len. Return the mean over n_trials.

    student_temp / teacher_temp override `temp` for the respective model;
    use to simulate the sharpness gap between a large teacher and small student.
    """
    g = torch.Generator().manual_seed(seed)
    total = 0
    for t in range(n_trials):
        pending = int(tokens[t % len(tokens)])
        q_paths, q_prefixes, q_probs_dict, p_probs_dict = sample_tree(
            student, teacher, pending, K, L, temp, with_grad=False, generator=g,
            student_temp=student_temp, teacher_temp=teacher_temp,
        )
        total += block_length_once(q_paths, q_prefixes, q_probs_dict, p_probs_dict, mode)
    return total / n_trials


def block_efficiency_table(student, teacher, tokens, K, L, temp,
                           verifiers=None, n_trials=300, seed=0,
                           student_temp=None, teacher_temp=None) -> dict:
    """BE for every verifier in `verifiers` (default ALL_VERIFIERS)."""
    verifiers = verifiers or ALL_VERIFIERS
    return {
        v: estimate_block_efficiency(student, teacher, v, tokens, K, L, temp,
                                     n_trials=n_trials, seed=seed,
                                     student_temp=student_temp, teacher_temp=teacher_temp)
        for v in verifiers
    }


def tree_loss_value(student, teacher, pending, loss_name, K, L, temp,
                    generator=None, student_temp=None, teacher_temp=None) -> torch.Tensor:
    """Build a tree WITH grad and call the REAL compute_tree_loss.

    Returns the scalar loss tensor (grad_fn attached) — backprop reaches the
    student's W parameter.
    """
    q_paths, q_prefixes, q_probs_dict, p_probs_dict = sample_tree(
        student, teacher, pending, K, L, temp, with_grad=True, generator=generator,
        student_temp=student_temp, teacher_temp=teacher_temp,
    )
    return compute_tree_loss(loss_name, q_probs_dict, p_probs_dict, q_paths, L=L, K=K)
