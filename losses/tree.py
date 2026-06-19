"""
Tree-structured (on-policy) distillation losses.

These losses are computed on the STUDENT's own draft tree — K sampled paths
of length L from the student, drawn fresh at every training step — rather
than on a teacher rollout.  Each loss optimises an objective tied to a
specific verifier's acceptance behaviour:

    Generic divergences (loss does not depend on the verifier):
        kl_tree, rev_kl_tree, jsd_tree

    Verifier-aligned (loss = −E[τ_V] for verifier V):
        bv_tree, gbv_tree, traversal_tree         (non-OT verifiers)
        naive_tree, nss_tree, specinfer_tree,     (OT-based verifiers)
        spectr_tree, khisti_tree

The math at a glance
--------------------
Block efficiency under verifier V along an L-step path:

    E[τ_V] = Σ_{i=1..L}  Π_{j=1..i} α_V(p_j, q_j, K)

where α_V is the per-node acceptance probability for verifier V, p is the
teacher distribution, q is the student (draft) distribution, K is the number
of paths.  We train the student to maximise this telescoping product by
minimising  L_V = −E[τ_V].  The per-verifier α_V formulas below are exact
PyTorch translations of the formulas in verifiers/node.py.

All functions share the signature

    loss_fn(q_probs_dict, p_probs_dict, q_paths, L, K) -> torch.Tensor

so they can be dispatched from train.py through the registry at the bottom
of this file without any branching in the training loop.

Inputs
------
q_probs_dict : { "<prefix>" : tensor[V] WITH grad }  — student distributions
                                                       at every non-leaf node.
p_probs_dict : { "<prefix>" : tensor[V] detached  }  — teacher at same nodes.
q_paths      : list of K paths, each a list of L+1 token IDs.
L            : draft block length.
K            : number of draft paths per step.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1.  Generic divergences on tree nodes (no verifier in the formula)
# ---------------------------------------------------------------------------

def kl_tree(q_probs_dict, p_probs_dict, q_paths=None, L=None, K=None, **_kw):
    """Forward KL(p_teacher ∥ q_student), averaged over non-leaf tree nodes."""
    device = next(iter(q_probs_dict.values())).device
    total  = torch.zeros(1, device=device)
    n      = 0
    for prefix, q in q_probs_dict.items():
        if prefix not in p_probs_dict:
            continue
        p = p_probs_dict[prefix].detach().to(q.dtype)
        total = total + F.kl_div(q.log().clamp(min=-100.0), p, reduction="sum")
        n += 1
    if n == 0:
        return torch.zeros(1, device=device, requires_grad=True)
    return total / n


def rev_kl_tree(q_probs_dict, p_probs_dict, q_paths=None, L=None, K=None, **_kw):
    """Reverse KL(q_student ∥ p_teacher), averaged over non-leaf tree nodes."""
    device = next(iter(q_probs_dict.values())).device
    total  = torch.zeros(1, device=device)
    n      = 0
    for prefix, q in q_probs_dict.items():
        if prefix not in p_probs_dict:
            continue
        p     = p_probs_dict[prefix].detach().to(q.dtype)
        log_q = q.log().clamp(min=-100.0)
        log_p = p.log().clamp(min=-100.0)
        total = total + (q * (log_q - log_p)).sum()
        n    += 1
    if n == 0:
        return torch.zeros(1, device=device, requires_grad=True)
    return total / n


def jsd_tree(q_probs_dict, p_probs_dict, q_paths=None, L=None, K=None, alpha=0.5, **_kw):
    """Symmetric JSD at every non-leaf node (mixture weight α=0.5 by default)."""
    device = next(iter(q_probs_dict.values())).device
    total  = torch.zeros(1, device=device)
    n      = 0
    for prefix, q in q_probs_dict.items():
        if prefix not in p_probs_dict:
            continue
        p     = p_probs_dict[prefix].detach().to(q.dtype)
        m     = alpha * q + (1.0 - alpha) * p
        log_m = m.log().clamp(min=-100.0)
        log_q = q.log().clamp(min=-100.0)
        log_p = p.log().clamp(min=-100.0)
        total = total + alpha * (q * (log_q - log_m)).sum() \
                      + (1.0 - alpha) * (p * (log_p - log_m)).sum()
        n    += 1
    if n == 0:
        return torch.zeros(1, device=device, requires_grad=True)
    return total / n


# ---------------------------------------------------------------------------
# 2.  Non-OT verifier-aligned losses (BV, GBV, Traversal)
# ---------------------------------------------------------------------------

def _bv_path_loss(q_probs_dict, p_probs_dict, path, L):
    """Differentiable -E[τ_BV] along ONE path.  See verifiers/node.py bv_verify."""
    device = next(iter(q_probs_dict.values())).device
    w      = torch.ones(1, device=device)                       # cumulative weight
    h_list = []

    for i in range(1, L + 1):
        prefix = ",".join(str(x) for x in path[:i])
        token  = path[i]
        if prefix not in q_probs_dict or prefix not in p_probs_dict:
            break
        p = p_probs_dict[prefix].detach().to(w.dtype)
        q = q_probs_dict[prefix].to(w.dtype)

        # Chain weight w_i = w_{i-1} * min(1, p[t]/q[t]).  Grad flows via q[token].
        alpha = torch.clamp(p[token] / q[token].clamp(min=1e-9),
                            min=1e-6, max=1.0 - 1e-6)
        w = w * alpha

        # Block acceptance h_i: full-vocab integral, except at the leaf it = w.
        if i < L:
            # Detach w in the integral to keep gradient single-pass through q.
            residual = F.relu(w.detach() * p - q)
            num = residual.sum()
            c   = (1.0 - w.detach()).clamp(min=0.05)            # stability floor
            h   = (num / (num + c)).clamp(max=1.0 - 1e-6)
        else:
            h = w.clamp(max=1.0 - 1e-6)
        h_list.append(h)

    if not h_list:
        return torch.zeros(1, device=device, requires_grad=True)

    # E[τ_BV] = Σ_i i · h_i · Π_{j<i}(1 - h_j); detach survival to avoid
    # second-order paths through earlier h's.
    e_tau    = torch.zeros(1, device=device)
    survival = torch.ones(1,  device=device)
    for i, h in enumerate(h_list):
        e_tau    = e_tau + (i + 1) * survival * h
        survival = survival * (1.0 - h.detach())
    return -e_tau


def bv_tree(q_probs_dict, p_probs_dict, q_paths, L, K=None, **_kw):
    """BV-aligned loss averaged over the K draft paths."""
    total = torch.zeros(1, device=next(iter(q_probs_dict.values())).device)
    for path in q_paths:
        total = total + _bv_path_loss(q_probs_dict, p_probs_dict, path, L)
    return total / len(q_paths)


def traversal_tree(q_probs_dict, p_probs_dict, q_paths, L, K, **_kw):
    """
    Differentiable surrogate for E[τ_traversal] via _bv_path_loss per path.

    The traversal verifier applies BV-style leaf-rejection on a K-path tree
    (identical to BV at K=1).  The previous running-weight surrogate used
    p[token]/q[token] (single sampled token) for its gradient, which is sparse:
    it can only push q DOWN at sampled tokens and has no signal to push q UP
    toward p.  Over training the model finds a degenerate minimum where q
    concentrates on a teacher-rejected token (p[token]≈0 → alpha→0 → e_tau→0
    → loss→0 → gradient vanishes → training freezes).

    Fix: use _telescoping_loss(_alpha_naive).  _alpha_naive = Σ_v min(p[v], q[v])
    is strictly positive for any full-softmax distribution — the zero minimum is
    structurally unreachable.  _bv_path_loss (previous attempt) still collapses
    because every gradient term is multiplied by w, and w gates on p[token]/q[token]
    which can reach zero.  At K=1 traversal and naive are identical verifiers.
    For K>1 this loses verifier-specificity but retains a stable training signal.
    Val is still measured with traversal verifier (train.py _LOSS_TO_VERIFIER),
    so checkpoint selection remains traversal-aligned.
    """
    return _telescoping_loss(_alpha_naive, q_probs_dict, p_probs_dict, q_paths, L, K)


# --- GBV (greedy path + BV on skewed distribution) ----------------------------

def _gbv_select_path(q_probs_dict, p_probs_dict, q_paths, L):
    """No-grad greedy walk: at each node pick the child with highest p[t]/q[t]."""
    children: Dict[str, List[Tuple[str, int]]] = {}
    for path in q_paths:
        for i in range(len(path) - 1):
            pfx  = ",".join(str(x) for x in path[:i + 1])
            cpfx = ",".join(str(x) for x in path[:i + 2])
            tok  = path[i + 1]
            entry = (cpfx, tok)
            if entry not in children.get(pfx, []):
                children.setdefault(pfx, []).append(entry)

    selected: List[int] = [q_paths[0][0]]
    cur_pfx = str(q_paths[0][0])
    for _ in range(L):
        if cur_pfx not in children or cur_pfx not in q_probs_dict:
            break
        p = p_probs_dict.get(cur_pfx)
        q = q_probs_dict.get(cur_pfx)
        if p is None or q is None:
            break
        p, q = p.detach(), q.detach()
        best = (None, None, -1.0)
        for cpfx, tok in children[cur_pfx]:
            qt = q[tok].item()
            pt = p[tok].item()
            r  = pt / (qt + torch.finfo(p.dtype).eps)
            if r > best[2]:
                best = (cpfx, tok, r)
        if best[1] is None:
            break
        selected.append(best[1])
        cur_pfx = best[0]
    return selected


def _gbv_skew_at_node(q, q_joint, q_joint_cdf, q_cdf, K):
    """GBV's per-node skew formula — see verifiers/node.py compute_skew."""
    if K == 1:
        return q
    A         = q_joint_cdf - q_joint
    y         = q_joint * (q_cdf - q)
    z         = q_joint * q
    A_denom   = A.detach().clamp(min=1e-8)
    A_stable  = A / A_denom
    y_stable  = y / A_denom
    z_stable  = z / A_denom

    q_skew = torch.zeros_like(q)
    for i in range(K):
        q_skew = q_skew + (A_stable + y_stable + z_stable) ** i \
                        * (A_stable + y_stable) ** (K - 1 - i)
    q_skew = F.relu(q_skew * q)

    with torch.no_grad():
        s = q_skew.sum()
        if not (s.item() > 0 and torch.isfinite(s).item()):
            return q                                            # fallback
    return q_skew / q_skew.sum()


def _gbv_skew_dict(q_probs_dict, p_probs_dict, path, L, K):
    """Build {prefix → q_skew} along the GBV-selected path, WITH grad through q."""
    if K == 1:
        return {",".join(str(x) for x in path[:i + 1]):
                q_probs_dict[",".join(str(x) for x in path[:i + 1])]
                for i in range(L)
                if ",".join(str(x) for x in path[:i + 1]) in q_probs_dict}

    device = next(iter(q_probs_dict.values())).device
    skew   = {}
    prev_q_joint = prev_q_joint_cdf = prev_q_cdf = None

    for i in range(L):
        prefix = ",".join(str(x) for x in path[:i + 1])
        if prefix not in q_probs_dict or prefix not in p_probs_dict:
            break
        q = q_probs_dict[prefix]
        p = p_probs_dict[prefix].detach().to(q.dtype)

        # Argsort on p/q ratio (discrete — no grad), then cumsum q under that order.
        with torch.no_grad():
            eps   = torch.finfo(q.dtype).eps
            ratio = p / q.detach().clamp(min=eps)
            order = torch.argsort(ratio, stable=True)
            inv   = torch.empty_like(order)
            inv[order] = torch.arange(len(order), device=device)
        q_cdf = q[order].cumsum(dim=-1)[inv]                    # [V], WITH grad

        if i == 0:
            cur_q_joint     = q.new_tensor(1.0)
            cur_q_joint_cdf = q.new_tensor(1.0)
        else:
            cur_tok       = path[i]
            parent_prefix = ",".join(str(x) for x in path[:i])
            parent_q      = q_probs_dict[parent_prefix]
            cur_q_joint     = prev_q_joint * parent_q[cur_tok]
            cur_q_joint_cdf = (prev_q_joint_cdf - prev_q_joint
                               + prev_q_joint * prev_q_cdf[cur_tok])

        skew[prefix] = _gbv_skew_at_node(q, cur_q_joint, cur_q_joint_cdf, q_cdf, K)
        prev_q_joint, prev_q_joint_cdf, prev_q_cdf = cur_q_joint, cur_q_joint_cdf, q_cdf

    return skew


def gbv_tree(q_probs_dict, p_probs_dict, q_paths, L, K, **_kw):
    """
    GBV-aligned loss: pick the best path (no grad), substitute q_skew for q,
    then run the BV path loss.  Matches gbv_verify in verifiers/node.py.
    """
    device   = next(iter(q_probs_dict.values())).device
    with torch.no_grad():
        path = _gbv_select_path(q_probs_dict, p_probs_dict, q_paths, L)
    skew = _gbv_skew_dict(q_probs_dict, p_probs_dict, path, L, K)
    if not skew:
        return torch.zeros(1, device=device, requires_grad=True)
    return _bv_path_loss(skew, p_probs_dict, path, L)


# ---------------------------------------------------------------------------
# 3.  OT-based verifier-aligned losses
#     Per-node α formulas — exact PyTorch ports of verifiers/node.py.
#     Each is a closed-form function of (p, q, K) with grad flowing through q.
# ---------------------------------------------------------------------------

def _alpha_naive(p, q, K):
    """α for naive (Chen/Leviathan).  K=1 → α = Σ min(p,q).  K>1 adds residual."""
    accept = torch.minimum(p, q).sum()
    if K > 1:
        accept = accept + (F.relu(p - q) * (1.0 - (1.0 - q) ** (K - 1))).sum()
    return accept


def _alpha_nss(p, q, K):
    """α for NSS = Σ p · (1 - (1 - q)^K).  Smooth in q for any K."""
    return (p * (1.0 - (1.0 - q) ** K)).sum()


def _alpha_specinfer(p, q, K):
    """α for SpecInfer — iterative closed form, K steps of residual update."""
    eps    = 1e-6
    reject = torch.ones(1, device=p.device, dtype=p.dtype)
    miss   = torch.ones_like(p)
    p_cur  = p
    for _ in range(K):
        r = (1.0 - torch.minimum(p_cur, q).sum()).clamp(min=eps)
        reject = reject * r
        miss   = miss * (1.0 - F.relu(q - p_cur) / r)
        p_next = F.relu(p_cur - q)
        s      = p_next.sum()
        if s.item() <= 0:
            p_next = p_next + 1e-4
            s      = p_next.sum()
        p_cur = p_next / s.clamp(min=eps)
    accept = (1.0 - reject) + reject * (p_cur * (1.0 - miss)).sum()
    return accept.squeeze() if accept.dim() > 0 else accept


def _alpha_spectr(p, q, K):
    """
    α for SpecTr.  ρ is found by binary search and DETACHED (first-order
    surrogate — the implicit-function gradient through ρ adds ~30% wallclock
    for a tight first-order approximation).
    """
    if K <= 1:
        return _alpha_naive(p, q, K)
    rho_low, rho_high = 1.0, float(K)
    with torch.no_grad():
        q_det = q.detach()
        for _ in range(20):                                     # 20 iters → 1e-5 tol
            rho     = 0.5 * (rho_low + rho_high)
            beta_v  = float(torch.minimum(p / rho, q_det).sum())
            p_acc_v = 1.0 - (1.0 - beta_v) ** K
            if p_acc_v >= rho * beta_v:
                rho_low  = rho
            else:
                rho_high = rho
        rho_det = rho_high

    eps  = 1e-6
    beta = torch.minimum(p / rho_det, q).sum()
    if beta.item() >= 1.0:
        return torch.ones(1, device=p.device, dtype=p.dtype)

    p_acc           = 1.0 - (1.0 - beta) ** K
    one_minus_beta  = (1.0 - beta).clamp(min=eps)
    p_acc_over_beta = p_acc / beta.clamp(min=eps)
    p_res_raw       = F.relu(p - torch.minimum(p / rho_det, q) * p_acc_over_beta)
    p_res           = p_res_raw / p_res_raw.sum().clamp(min=eps)
    r               = F.relu(q - p / rho_det) / one_minus_beta
    return p_acc + (1.0 - p_acc) * (p_res * (1.0 - (1.0 - r) ** K)).sum()


def _alpha_khisti(p, q, K):
    """
    α for Khisti.  We approximate the exact rank-LP solver with a softmax
    importance reweighting:  q_imp ∝ q · softmax((K-1)·log(p/q)).  Reduces
    to naive at K=1 and concentrates draft mass on high p/q tokens as K grows.
    """
    if K <= 1:
        return _alpha_naive(p, q, K)
    eps         = 1e-9
    log_ratio   = torch.log((p + eps) / q.clamp(min=eps))
    weights     = F.softmax((K - 1) * log_ratio, dim=-1)
    q_imp_raw   = q * weights * float(p.numel())
    q_imp       = q_imp_raw / q_imp_raw.sum().clamp(min=eps)
    return torch.minimum(p, q_imp).sum()


def _telescoping_loss(alpha_fn, q_probs_dict, p_probs_dict, q_paths, L, K,
                      detach_survival=True):
    """
    L = -E[τ] = - Σ_path Σ_{i=1..L} Π_{j=1..i} α(p_j, q_j, K),  averaged over paths.

    detach_survival=True (default):
        Survival Π_{j<i} αⱼ is detached at each step so the gradient enters only
        through the most recent α — single-pass / first-order surrogate, same
        trick the BV path loss uses.  Coefficient on ∇αᵢ is just Wᵢ=Π_{j<i}αⱼ.

    detach_survival=False (the `naive_tree_full` variant):
        Survival keeps grad → autograd computes the EXACT ∇E[τ] via the product
        rule.  The coefficient on ∇αₛ becomes Wₛ·(1 + α_{s+1} + α_{s+1}α_{s+2} +
        …) — i.e. early tokens are credited for gating all downstream depth.
        This is the controlled test of whether the detach was costing the
        credit-assignment signal.  Gradients through early α's are amplified by
        the downstream product, so GRAD_CLIP (already applied in train.py /
        algo_sanity.py) is load-bearing here.
    """
    device  = next(iter(q_probs_dict.values())).device
    total   = torch.zeros(1, device=device)
    n_paths = 0

    for path in q_paths:
        e_tau    = torch.zeros(1, device=device)
        survival = torch.ones(1, device=device)
        used     = False
        for i in range(1, L + 1):
            prefix = ",".join(str(x) for x in path[:i])
            if prefix not in q_probs_dict or prefix not in p_probs_dict:
                break
            q     = q_probs_dict[prefix]
            p     = p_probs_dict[prefix].detach().to(q.dtype)
            alpha = alpha_fn(p, q, K)
            e_tau    = e_tau + survival * alpha
            survival = survival * (alpha.detach() if detach_survival else alpha)
            used     = True
        if used:
            total += e_tau
            n_paths += 1

    if n_paths == 0:
        return torch.zeros(1, device=device, requires_grad=True)
    return -(total / n_paths)


def naive_tree     (q, p, paths, L, K, **_kw): return _telescoping_loss(_alpha_naive, q, p, paths, L, K)
def naive_tree_full(q, p, paths, L, K, **_kw): return _telescoping_loss(_alpha_naive, q, p, paths, L, K, detach_survival=False)

# Off-policy variants — same loss formula, but train.py routes these through
# compute_offpolicy_tree_loss() so the training path comes from the teacher's
# greedy rollout rather than draft samples.  This prevents survival collapse on
# hard prompts (teacher tokens have high acceptance → products stay non-negligible).
# op_naive_tree_full additionally un-detaches survival → exact ∇E[τ] with full
# depth credit (the "researcher's depth" wired into the gradient, not a scalar).
def op_naive_tree     (q, p, paths, L, K, **_kw): return _telescoping_loss(_alpha_naive, q, p, paths, L, K)
def op_naive_tree_full(q, p, paths, L, K, **_kw): return _telescoping_loss(_alpha_naive, q, p, paths, L, K, detach_survival=False)

def nss_tree      (q, p, paths, L, K, **_kw): return _telescoping_loss(_alpha_nss,       q, p, paths, L, K)
def specinfer_tree(q, p, paths, L, K, **_kw): return _telescoping_loss(_alpha_specinfer, q, p, paths, L, K)
def spectr_tree   (q, p, paths, L, K, **_kw): return _telescoping_loss(_alpha_spectr,    q, p, paths, L, K)
def khisti_tree   (q, p, paths, L, K, **_kw): return _telescoping_loss(_alpha_khisti,    q, p, paths, L, K)


# ---------------------------------------------------------------------------
# Registry — train.py looks up the loss here by --loss flag.
# Every value has the signature (q_probs_dict, p_probs_dict, q_paths, L, K, **_kw).
# ---------------------------------------------------------------------------
TREE_LOSSES = {
    "kl_tree":         kl_tree,
    "rev_kl_tree":     rev_kl_tree,
    "jsd_tree":        jsd_tree,
    "bv_tree":         bv_tree,
    "gbv_tree":        gbv_tree,
    "traversal_tree":   traversal_tree,
    "naive_tree":       naive_tree,
    "naive_tree_full":  naive_tree_full,
    "op_naive_tree":    op_naive_tree,
    "op_naive_tree_full": op_naive_tree_full,
    "nss_tree":         nss_tree,
    "specinfer_tree":  specinfer_tree,
    "spectr_tree":     spectr_tree,
    "khisti_tree":     khisti_tree,
}
