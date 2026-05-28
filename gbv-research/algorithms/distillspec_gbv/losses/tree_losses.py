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


def _compute_skew_formula(
    q: torch.Tensor,
    q_joint: torch.Tensor,
    q_joint_cdf: torch.Tensor,
    q_cdf: torch.Tensor,
    K: int,
) -> torch.Tensor:
    """
    Differentiable GBV skew formula for a single tree node.

    Replicates TreeVerifier.compute_skew() from otlp_registry.py WITH gradient
    flow through q so that loss.backward() reaches the LoRA parameters.

    The formula (see the docstring of compute_skew in otlp_registry.py for the
    full derivation) is:

        q_skew = q * Σ_{i=0}^{K-1}  ((A+y+z)/A_denom)^i  *  ((A+y)/A_denom)^(K-1-i)

    then relu-clamp and renormalise, where
        A = q_joint_cdf - q_joint          scalar
        y = q_joint * (q_cdf - q)          [V]
        z = q_joint * q                    [V]
        A_denom = clamp(A, min=1e-8)       (stability; detached so grad goes through A only)

    Args:
        q:            Draft distribution [V], WITH grad.
        q_joint:      Product of q[token] along path to this node (scalar).
        q_joint_cdf:  Joint CDF value for this node (scalar).
        q_cdf:        CDF of q under p/q-ratio ordering [V], WITH grad.
        K:            Number of i.i.d. draft paths.

    Returns:
        q_skew [V], WITH grad.  Falls back to raw q on numerical failure.
    """
    if K == 1:
        return q   # skew formula degenerates to identity when K=1

    A = q_joint_cdf - q_joint               # scalar — has grad for depth ≥ 1
    y = q_joint * (q_cdf - q)               # [V], WITH grad
    z = q_joint * q                          # [V], WITH grad

    # Stability divisor is detached so the gradient runs through A, not A_denom.
    A_denom  = A.detach().clamp(min=1e-8)
    A_stable = A / A_denom                  # scalar, WITH grad
    y_stable = y / A_denom                  # [V], WITH grad
    z_stable = z / A_denom                  # [V], WITH grad

    q_skew = torch.zeros_like(q)            # zero baseline (no grad)
    for i in range(K):
        q_skew = q_skew + (
            (A_stable + y_stable + z_stable) ** i
            * (A_stable + y_stable) ** (K - 1 - i)
        )
    q_skew = q_skew * q                     # [V], WITH grad through both q and the sum
    q_skew = F.relu(q_skew)                 # [V], WITH grad

    # Validity check — only the boolean decision is no-grad; the tensor itself keeps grad.
    with torch.no_grad():
        s_check = q_skew.sum()
        valid   = (s_check > 0).item() and torch.isfinite(s_check).item()

    if not valid:
        return q   # fallback — grad still flows through q

    s = q_skew.sum()                        # scalar, WITH grad
    return q_skew / s


def _compute_q_skew_along_path(
    q_probs_dict: Dict[str, torch.Tensor],   # {prefix → [V]}, WITH grad
    p_probs_dict: Dict[str, torch.Tensor],   # {prefix → [V]}, detached
    path: List[int],                          # GBV-selected path, length L+1
    L: int,
    K: int,
) -> Dict[str, torch.Tensor]:
    """
    Compute the GBV skewed distribution at every non-leaf node along the
    selected path, WITH gradient flow through q.

    Replicates the q_joint / q_joint_cdf / q_cdf bookkeeping from
    gbv_verify() in otlp_registry.py, but operates on tensors that carry
    grad so that loss.backward() propagates through q_skew into q.

    Gradient paths:
      * q_cdf  = q[order].cumsum() reordered   — grad through q[order]
      * q_joint  = Π q[path_token]             — grad through each q[token]
      * q_joint_cdf recursion                  — grad through q_cdf, q_joint
      * _compute_skew_formula(...)             — grad through all the above

    The order / inv_order permutation and the argsort are computed without
    grad (discrete ops); they act as fixed indices into q.

    Args:
        q_probs_dict:  Student distributions WITH grad at non-leaf nodes.
        p_probs_dict:  Teacher distributions, detached.
        path:          GBV-selected path, length L+1.  path[0] = pending token.
        L:             Draft block length.
        K:             Number of i.i.d. draft paths.

    Returns:
        {prefix → q_skew [V]} WITH grad for depths 0 .. L-1 on the path.
        Keys are the same comma-separated prefix strings as q_probs_dict.
    """
    if K == 1:
        # q_skew == q for K=1; just pass through the relevant slices.
        return {
            ",".join(str(x) for x in path[:i + 1]): q_probs_dict[",".join(str(x) for x in path[:i + 1])]
            for i in range(L)
            if ",".join(str(x) for x in path[:i + 1]) in q_probs_dict
        }

    device = next(iter(q_probs_dict.values())).device
    q_skew_dict: Dict[str, torch.Tensor] = {}

    # Running state — updated depth-by-depth.
    prev_q_joint:     Optional[torch.Tensor] = None   # scalar, WITH grad after depth 0
    prev_q_joint_cdf: Optional[torch.Tensor] = None   # scalar, WITH grad after depth 0
    prev_q_cdf:       Optional[torch.Tensor] = None   # [V], WITH grad after depth 0

    for i in range(L):   # non-leaf depths 0 … L-1
        prefix = ",".join(str(x) for x in path[:i + 1])

        if prefix not in q_probs_dict or prefix not in p_probs_dict:
            break   # walked off the tracked tree

        q = q_probs_dict[prefix]                        # [V], WITH grad
        p = p_probs_dict[prefix].detach().to(q.dtype)  # [V], frozen

        # ── p/q ratio ordering (no grad — argsort is discrete) ──────────────
        with torch.no_grad():
            eps   = torch.finfo(q.dtype).eps
            ratio = torch.minimum(
                torch.ones_like(p),
                p / q.detach().clamp(min=eps),
            )
            order     = torch.argsort(ratio, stable=True)   # [V]
            inv_order = torch.empty_like(order)
            inv_order[order] = torch.arange(len(order), device=device)

        # ── q_cdf WITH grad ──────────────────────────────────────────────────
        # q_cdf[v] = Σ_{u : ratio[u] ≤ ratio[v]} q[u]
        # Use fixed permutation `order` to sort, cumsum, then invert.
        q_cdf_sorted = q[order].cumsum(dim=-1)   # [V], WITH grad
        q_cdf        = q_cdf_sorted[inv_order]   # [V], WITH grad

        # ── q_joint and q_joint_cdf ──────────────────────────────────────────
        if i == 0:
            # Root: joint prob of "sampling the current node" = 1 (no tokens sampled yet).
            cur_q_joint     = q.new_tensor(1.0)   # scalar, no grad (definitional constant)
            cur_q_joint_cdf = q.new_tensor(1.0)   # scalar, no grad
        else:
            # Token that moved us FROM parent (depth i-1) TO this node (depth i).
            cur_token     = path[i]
            parent_prefix = ",".join(str(x) for x in path[:i])
            parent_q      = q_probs_dict[parent_prefix]   # [V], WITH grad

            # q_joint = parent.q_joint * q_parent[cur_token]
            cur_q_joint = prev_q_joint * parent_q[cur_token]   # scalar, WITH grad

            # q_joint_cdf = parent.q_joint_cdf - parent.q_joint
            #               + parent.q_joint * parent.q_cdf[cur_token]
            cur_q_joint_cdf = (
                prev_q_joint_cdf
                - prev_q_joint
                + prev_q_joint * prev_q_cdf[cur_token]   # WITH grad
            )

        # ── skewed distribution at this node ─────────────────────────────────
        q_skew_dict[prefix] = _compute_skew_formula(
            q, cur_q_joint, cur_q_joint_cdf, q_cdf, K
        )

        # Advance running state for the next depth.
        prev_q_joint     = cur_q_joint
        prev_q_joint_cdf = cur_q_joint_cdf
        prev_q_cdf       = q_cdf

    return q_skew_dict


def gbv_tree_loss(
    q_probs_dict: Dict[str, torch.Tensor],
    p_probs_dict: Dict[str, torch.Tensor],
    q_paths: List[List[int]],
    L: int,
    K: int,
) -> torch.Tensor:
    """
    GBV tree loss: select the best path (no-grad), compute the skewed draft
    distribution q_skew at each node WITH gradients, then run BV on that path
    using q_skew in place of q — matching exactly what gbv_verify() does.

    Background
    ----------
    gbv_verify() walks the tree greedily (highest p/q ratio at each step),
    builds q_skew at every visited node via compute_skew() — which adjusts
    the effective draft distribution to account for the K i.i.d. path
    structure — and then runs block verification with q_skew substituted
    for q in both the chain weight and the block acceptance integral.

    A naive implementation that passes raw q_probs_dict to bv_tree_loss
    produces a gradient signal that does NOT match what the verifier
    actually optimises, because the verifier never sees raw q during BV
    on a multi-path tree.

    Gradient flow
    -------------
    Path selection  →  no-grad (discrete argmax, straight-through)
    q_skew computation:
      • q_cdf = q[order].cumsum()      — grad through q values (fixed permutation)
      • q_joint = Π q[token]           — grad through each path token's probability
      • q_joint_cdf recursion          — grad through q_cdf and q_joint
      • _compute_skew_formula(...)     — grad through the whole polynomial in q
    BV loss on q_skew → grad through q_skew[token] (chain weight) and q_skew[:]
    (block acceptance integral).

    Args:
        q_probs_dict:  Student distributions WITH grad (from draft_tree_forward_with_grad).
        p_probs_dict:  Teacher distributions, detached.
        q_paths:       All K draft paths.
        L:             Draft block length.
        K:             Number of paths.

    Returns:
        Scalar tensor = -E[τ_BV on GBV-selected path, using q_skew].
    """
    device = next(iter(q_probs_dict.values())).device

    # ── Step 1: greedy path selection (no grad) ───────────────────────────────
    with torch.no_grad():
        best_path = _gbv_select_path(q_probs_dict, p_probs_dict, q_paths, L)

    # ── Step 2: differentiable q_skew along the selected path ────────────────
    q_skew_dict = _compute_q_skew_along_path(
        q_probs_dict, p_probs_dict, best_path, L, K
    )

    if not q_skew_dict:
        return torch.zeros(1, device=device, requires_grad=True)

    # ── Step 3: BV loss with q_skew substituted for q everywhere ─────────────
    # bv_tree_loss uses its first argument (q_probs_dict) for both the chain
    # weight α = p[t]/q[t] and the block acceptance integral relu(w*p - q).sum().
    # Passing q_skew_dict here replicates what gbv_verify does.
    return bv_tree_loss(q_skew_dict, p_probs_dict, best_path, L)


# ---------------------------------------------------------------------------
# ebe_tree — on-policy EBE on the student's own draft tree
# ---------------------------------------------------------------------------

def ebe_tree_loss(
    q_probs_dict: Dict[str, torch.Tensor],   # {prefix → [V]}, WITH grad
    p_probs_dict: Dict[str, torch.Tensor],   # {prefix → [V]}, detached
    q_paths: List[List[int]],                 # K draft paths, each length L+1
    L: int,
    K: int,
    kl_weight: float = 0.1,
) -> torch.Tensor:
    """
    On-policy Expected Block Efficiency — on-policy ablation of flat EBE.

    Flat EBE (ebe.py) computes α = min(1, q_teacher[t]/p_draft[t]) for the
    TEACHER's own generated tokens.  At inference the verifier evaluates the
    STUDENT's actual draft tokens, creating an off-policy mismatch that kills
    the gradient signal.

    ebe_tree fixes this by computing the EBE formula on the student's own
    K draft paths drawn from the student's own tree:

        α_i = min(1, p_target[t_i] / q_draft[t_i])   where t_i ∈ student path
        EBE  = −mean_k (1 + Σ_{k=1}^{L} Π_{i=1}^{k} α_i)

    Optional KL regulariser (same as flat EBE): prevents gradient collapse
    when α ≈ 1 everywhere, by providing a smooth gradient from the full
    distribution at every tree node.

    Ablation role in the paper table
    ---------------------------------
      flat EBE      → off-policy, token-level acceptance
      ebe_tree      → on-policy, token-level acceptance    ← this function
      bv_tree       → on-policy, full-vocab block integral (theoretically optimal)

    If ebe_tree >> flat EBE  → the off-policy mismatch was the main culprit.
    If bv_tree   >> ebe_tree → full-vocab integral also matters (use bv_tree).

    Args:
        q_probs_dict:  Student distributions WITH grad at non-leaf tree nodes.
        p_probs_dict:  Teacher distributions, detached.
        q_paths:       All K draft paths sampled from the student's tree.
        L:             Draft block length.
        K:             Number of paths.
        kl_weight:     Weight λ for the KL regulariser (default 0.1).

    Returns:
        Scalar tensor = −E[τ_EBE] (minimise to maximise block efficiency).
    """
    device   = next(iter(q_probs_dict.values())).device
    total_ebe = torch.zeros(1, device=device)
    total_kl  = torch.zeros(1, device=device)
    n_paths   = 0
    n_nodes   = 0

    for path in q_paths:
        alpha_list: List[torch.Tensor] = []

        for i in range(1, L + 1):
            prefix = ",".join(str(x) for x in path[:i])
            if i >= len(path):
                break
            token = path[i]

            if prefix not in q_probs_dict or prefix not in p_probs_dict:
                break

            p = p_probs_dict[prefix].detach().to(q_probs_dict[prefix].dtype)   # [V] frozen
            q = q_probs_dict[prefix]                                            # [V] WITH grad

            # α = min(1, p[t] / q[t]) — clamp ≥ 1e-6 prevents NaN in cumprod backward
            alpha = torch.clamp(
                p[token] / q[token].clamp(min=1e-9), max=1.0
            ).clamp(min=1e-6)
            alpha_list.append(alpha)

            if kl_weight > 0.0:
                total_kl = total_kl + F.kl_div(
                    q.log().clamp(min=-100.0), p, reduction="sum"
                )
                n_nodes += 1

        if not alpha_list:
            continue

        # E[τ_EBE] = 1 + Σ_{k=1}^L Π_{i=1}^k α_i  (torch.cumprod gives the prefix products)
        blk       = torch.stack(alpha_list)                     # [≤L], WITH grad
        ebe_path  = -(1.0 + torch.cumprod(blk, dim=0).sum())   # scalar, minimise negative
        total_ebe = total_ebe + ebe_path
        n_paths  += 1

    if n_paths == 0:
        return torch.zeros(1, device=device, requires_grad=True)

    loss = total_ebe / n_paths
    if kl_weight > 0.0 and n_nodes > 0:
        loss = loss + kl_weight * (total_kl / n_nodes)

    return loss


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

TREE_LOSS_NAMES = frozenset({
    "kl_tree", "bv_tree", "gbv_tree", "traversal_tree", "ebe_tree",
})


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
        name:          One of TREE_LOSS_NAMES.
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
    elif name == "ebe_tree":
        return ebe_tree_loss(q_probs_dict, p_probs_dict, q_paths, L, K)
    else:
        raise ValueError(
            f"Unknown tree loss '{name}'. "
            f"Available: {sorted(TREE_LOSS_NAMES)}."
        )
