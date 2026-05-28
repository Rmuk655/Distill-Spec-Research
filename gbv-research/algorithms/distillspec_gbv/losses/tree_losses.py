"""
Tree-structured distillation losses for BV, GBV, and Traversal verifiers.

Background — why flat losses don't help tree verifiers
-------------------------------------------------------
The standard trainer generates a linear teacher sequence and trains the
draft model to match the teacher's token-level distribution.  The tree
verifiers (bv_verify, gbv_verify, traversal_verify in otlp_registry.py)
do NOT use the per-token acceptance rate α_i = min(1, p[t]/q[t]) on
teacher-generated tokens.  Instead they compute:

  BV block acceptance:    h_i = relu(w_i * p - q).sum()
                                / (relu(w_i * p - q).sum() + 1 - w_i)

  where p, q are full-vocabulary distributions at tree node i, and
  w_i = Π_{j≤i} min(1, p[t_j]/q[t_j]) is the cumulative weight.

This integral depends on how q distributes mass across ALL vocab tokens,
not just the teacher-generated one.  Flat EBE trains the right statistic
(α_i) but on the wrong distribution (teacher's linear rollout), while
the verifier evaluates on the student's own draft tree.

These losses fix both problems:
  1. Training data source: student's own draft tree → target scoring
     (on-policy — see tree_training.py).
  2. Loss objective: verifier-specific E[τ] as a differentiable function
     of q_probs_dict (the student's per-node distributions WITH grad).

Loss functions
--------------
  kl_tree         Forward KL at every tree node (on-policy baseline).
                  Easiest to implement; already much better than flat KL
                  because it trains on the student's own tree nodes.

  bv_tree         Differentiable surrogate for E[τ_BV], averaged over K paths.
                  Gradient flows through q[token] (chain weight) and q[:]
                  (block acceptance integral) at each node.

  gbv_tree        GBV path selection (no-grad argmax, straight-through),
                  then BV loss on the selected path with q_skew substituted.

  traversal_tree  Surrogate for E[τ_traversal]: maximise mean leaf weight
                  w_leaf = Π_ancestors min(1, p[t]/q[t]).
                  Differentiable; avoids the non-differentiable sequential
                  rejection process of the exact traversal verifier.

Usage
-----
These functions are called from trainer.py when args.loss is one of
{"kl_tree", "bv_tree", "gbv_tree", "traversal_tree"}.  They are NOT
registered in LOSS_REGISTRY (which has the flat (s_log, t_log) interface).
Instead, trainer.py dispatches directly by name.

All functions return a scalar tensor with grad_fn attached.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# kl_tree — on-policy KL at every tree node
# ---------------------------------------------------------------------------

def kl_tree_loss(
    q_probs_dict: Dict[str, torch.Tensor],   # {prefix → [V]}, WITH grad
    p_probs_dict: Dict[str, torch.Tensor],   # {prefix → [V]}, detached
) -> torch.Tensor:
    """
    Forward KL at every non-leaf node in the draft tree.

    KL(p || q) = Σ_v p[v] * log(p[v] / q[v])

    Averaged over all tree nodes.  The on-policy construction (training on
    the student's own draft tree nodes) is already a significant improvement
    over flat KL on the teacher's linear rollout.

    Args:
        q_probs_dict:  Student distributions at each non-leaf tree node,
                       WITH gradient (from draft_tree_forward_with_grad).
        p_probs_dict:  Teacher distributions at the same nodes, detached.

    Returns:
        Scalar tensor — mean KL over tree nodes.
    """
    device = next(iter(q_probs_dict.values())).device
    total  = torch.zeros(1, device=device)
    n      = 0

    for pfx, q in q_probs_dict.items():
        if pfx not in p_probs_dict:
            continue
        p = p_probs_dict[pfx].detach().to(q.dtype)
        # F.kl_div expects log-probs as first arg; reduction="sum" then we normalise
        total = total + F.kl_div(q.log().clamp(min=-100.0), p, reduction="sum")
        n += 1

    if n == 0:
        return torch.zeros(1, device=device, requires_grad=True)
    return total / n


# ---------------------------------------------------------------------------
# bv_tree — differentiable surrogate for E[τ_BV]
# ---------------------------------------------------------------------------

def bv_tree_loss(
    q_probs_dict: Dict[str, torch.Tensor],   # {prefix → [V]}, WITH grad
    p_probs_dict: Dict[str, torch.Tensor],   # {prefix → [V]}, detached
    path: List[int],                          # one full path, length L+1
    L: int,
) -> torch.Tensor:
    """
    Differentiable surrogate for E[τ_BV] along one path.

    Replicates the bv_verify block-acceptance formula in otlp_registry.py:
      w_i  = w_{i-1} * min(1, p[t_i] / q[t_i])      chain weight
      h_i  = relu(w_i * p - q).sum()                  block acceptance (non-leaf)
               / (relu(w_i * p - q).sum() + 1 - w_i)
      h_L  = w_L                                       leaf acceptance

    E[τ_BV] = Σ_{i=1}^{L}  i * h_i * Π_{j<i}(1 - h_j)

    Gradient flows through:
      • q[token]  — denominator of the chain weight α_i = min(1, p/q)
      • q[:]      — the whole-vocab integral relu(w*p - q).sum()

    Args:
        q_probs_dict:  Student distributions (WITH grad) at non-leaf nodes.
        p_probs_dict:  Teacher distributions (detached) at the same nodes.
        path:          One draft path; path[0] = pending token, path[i] = token at depth i.
        L:             Draft block length (path has L+1 elements).

    Returns:
        Scalar tensor = -E[τ_BV]   (negative because we minimise).
    """
    device = next(iter(q_probs_dict.values())).device
    w      = torch.ones(1, device=device)       # cumulative weight w_0 = 1
    h_list: List[torch.Tensor] = []

    for i in range(1, L + 1):
        prefix = ",".join(str(x) for x in path[:i])
        token  = path[i]

        if prefix not in q_probs_dict or prefix not in p_probs_dict:
            # Path walked off the tracked tree — stop here
            break

        p = p_probs_dict[prefix].detach().to(w.dtype)          # [V], frozen
        q = q_probs_dict[prefix].to(w.dtype)                   # [V], WITH grad

        # ── Chain weight: w_i = w_{i-1} * min(1, p[t] / q[t]) ───────────────
        # Gradient flows through q[token] (the denominator).
        # We clamp to [1e-9, 1] for numerical safety.
        alpha = torch.clamp(p[token] / q[token].clamp(min=1e-9), max=1.0)
        w = w * alpha

        # ── Block acceptance h_i ──────────────────────────────────────────────
        if i < L:
            # h_i = relu(w * p - q).sum() / (relu(w * p - q).sum() + 1 - w)
            # Detach w to avoid a second-order path through the chain weight;
            # the primary gradient already entered via q[token] above.
            residual = F.relu(w.detach() * p - q)              # [V]
            num = residual.sum()
            h   = num / (num + 1.0 - w.detach() + 1e-10)
        else:
            # Leaf: block acceptance = weight itself
            h = w

        h_list.append(h)

    if not h_list:
        return torch.zeros(1, device=device, requires_grad=True)

    # ── E[τ_BV] = Σ_i i * h_i * Π_{j<i}(1 - h_j) ───────────────────────────
    # Survival is detached so gradient doesn't loop through all earlier h terms.
    e_tau    = torch.zeros(1, device=device)
    survival = torch.ones(1,  device=device)
    for i, h in enumerate(h_list):
        e_tau    = e_tau + (i + 1) * survival * h
        survival = survival * (1.0 - h.detach())

    return -e_tau   # minimise negative E[τ]


def bv_tree_loss_all_paths(
    q_probs_dict: Dict[str, torch.Tensor],
    p_probs_dict: Dict[str, torch.Tensor],
    q_paths: List[List[int]],
    L: int,
) -> torch.Tensor:
    """BV loss averaged over all K draft paths."""
    total = torch.zeros(1, device=next(iter(q_probs_dict.values())).device)
    for path in q_paths:
        total = total + bv_tree_loss(q_probs_dict, p_probs_dict, path, L)
    return total / len(q_paths)


# ---------------------------------------------------------------------------
# gbv_tree — GBV path selection + BV loss
# ---------------------------------------------------------------------------

def _gbv_select_path(
    q_probs_dict: Dict[str, torch.Tensor],
    p_probs_dict: Dict[str, torch.Tensor],
    q_paths: List[List[int]],
    L: int,
) -> List[int]:
    """
    Replicate GBV's greedy path selection: walk the tree from root to leaf,
    at each step choosing the child with the highest p[t]/q[t] ratio.
    Runs under torch.no_grad — the selection itself carries no gradient.
    Returns the selected path (list of token IDs, length L+1).
    """
    # Build node → children map from paths
    children: Dict[str, List[Tuple[str, int]]] = {}   # prefix → [(child_prefix, token)]
    for path in q_paths:
        for i in range(len(path) - 1):
            pfx  = ",".join(str(x) for x in path[:i + 1])
            cpfx = ",".join(str(x) for x in path[:i + 2])
            tok  = path[i + 1]
            entry = (cpfx, tok)
            if entry not in children.get(pfx, []):
                children.setdefault(pfx, []).append(entry)

    # Greedy walk: at each node pick the child with the highest min(1, p/q) ratio
    root_pfx = str(q_paths[0][0])
    selected: List[int] = [q_paths[0][0]]   # start with the pending token
    cur_pfx  = root_pfx

    for _ in range(L):
        if cur_pfx not in children or cur_pfx not in q_probs_dict:
            break
        p = p_probs_dict.get(cur_pfx)
        q = q_probs_dict.get(cur_pfx)
        if p is None or q is None:
            break

        p = p.detach()
        q = q.detach()

        best_cpfx, best_tok, best_ratio = None, None, -1.0
        for cpfx, tok in children[cur_pfx]:
            qt = q[tok].item()
            pt = p[tok].item()
            ratio = min(1.0, pt / qt) if qt > 1e-9 else 0.0
            if ratio > best_ratio:
                best_cpfx, best_tok, best_ratio = cpfx, tok, ratio

        if best_tok is None:
            break
        selected.append(best_tok)
        cur_pfx = best_cpfx

    return selected


def gbv_tree_loss(
    q_probs_dict: Dict[str, torch.Tensor],
    p_probs_dict: Dict[str, torch.Tensor],
    q_paths: List[List[int]],
    L: int,
    K: int,
) -> torch.Tensor:
    """
    GBV tree loss: select the best path (no-grad straight-through), then
    compute the BV loss on that path.

    Straight-through means the path selection is treated as a fixed sample
    from the student's tree — gradients flow through the BV loss computation
    on the selected path, not through the selection criterion itself.

    Args:
        q_probs_dict:  Student distributions WITH grad.
        p_probs_dict:  Teacher distributions, detached.
        q_paths:       All K draft paths.
        L:             Draft block length.
        K:             Number of paths (unused directly, kept for API symmetry).

    Returns:
        Scalar tensor = -E[τ_BV on GBV-selected path].
    """
    with torch.no_grad():
        best_path = _gbv_select_path(q_probs_dict, p_probs_dict, q_paths, L)

    # Gradient flows through q_probs_dict values for nodes on the selected path
    return bv_tree_loss(q_probs_dict, p_probs_dict, best_path, L)


# ---------------------------------------------------------------------------
# traversal_tree — leaf-weight surrogate for E[τ_traversal]
# ---------------------------------------------------------------------------

def traversal_tree_loss(
    q_probs_dict: Dict[str, torch.Tensor],
    p_probs_dict: Dict[str, torch.Tensor],
    q_paths: List[List[int]],
    L: int,
    K: int,
) -> torch.Tensor:
    """
    Differentiable surrogate for E[τ_traversal].

    traversal_verify() accepts the first leaf (by DFS order) whose
    weight w_leaf clears a uniform random threshold.  The weight of a
    leaf is w_leaf = Π_{node on path} min(1, p[t]/q[t]), which is the
    same chain product that node.weight is initialised to in traversal_verify.

    Maximising mean w_leaf across all K leaves provides a well-defined
    differentiable objective that directly incentivises the draft model to
    keep w_leaf high — equivalent to making every draft token more likely
    under the target than under the student.

    The sequential rejection process and DFS ordering are NOT modelled
    here; they are stochastic and make the exact objective non-differentiable.
    This surrogate is a lower bound on what traversal achieves and converges
    in the same direction.

    Args:
        q_probs_dict:  Student distributions WITH grad at non-leaf nodes.
        p_probs_dict:  Teacher distributions, detached.
        q_paths:       All K draft paths.
        L:             Draft block length.
        K:             Number of draft paths.

    Returns:
        Scalar tensor = -mean_w_leaf  (negative because we minimise).
    """
    device = next(iter(q_probs_dict.values())).device
    total  = torch.zeros(1, device=device)

    for path in q_paths:
        w = torch.ones(1, device=device)
        for i in range(1, L + 1):
            prefix = ",".join(str(x) for x in path[:i])
            token  = path[i]
            if prefix not in q_probs_dict or prefix not in p_probs_dict:
                break
            p = p_probs_dict[prefix].detach().to(w.dtype)
            q = q_probs_dict[prefix].to(w.dtype)
            alpha = torch.clamp(p[token] / q[token].clamp(min=1e-9), max=1.0)
            w = w * alpha
        total = total + w

    return -(total / K)   # minimise negative mean leaf weight


# ---------------------------------------------------------------------------
# Registry and dispatch
# ---------------------------------------------------------------------------

TREE_LOSS_NAMES = frozenset({"kl_tree", "bv_tree", "gbv_tree", "traversal_tree"})


def compute_tree_loss(
    name: str,
    q_probs_dict: Dict[str, torch.Tensor],
    p_probs_dict: Dict[str, torch.Tensor],
    q_paths: List[List[int]],
    L: int,
    K: int,
) -> torch.Tensor:
    """
    Dispatch to the named tree loss.

    Args:
        name:          One of {"kl_tree", "bv_tree", "gbv_tree", "traversal_tree"}.
        q_probs_dict:  Student distributions WITH grad (from draft_tree_forward_with_grad).
        p_probs_dict:  Teacher distributions, detached (from target_tree_pass).
        q_paths:       K draft paths, each of length L+1.
        L:             Draft block length.
        K:             Number of paths.

    Returns:
        Scalar loss tensor with grad_fn attached.
    """
    if name == "kl_tree":
        return kl_tree_loss(q_probs_dict, p_probs_dict)
    elif name == "bv_tree":
        return bv_tree_loss_all_paths(q_probs_dict, p_probs_dict, q_paths, L)
    elif name == "gbv_tree":
        return gbv_tree_loss(q_probs_dict, p_probs_dict, q_paths, L, K)
    elif name == "traversal_tree":
        return traversal_tree_loss(q_probs_dict, p_probs_dict, q_paths, L, K)
    else:
        raise ValueError(
            f"Unknown tree loss '{name}'. "
            f"Available: {sorted(TREE_LOSS_NAMES)}."
        )
