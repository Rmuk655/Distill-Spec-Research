"""
loss_variants.py — experimental tree-loss variants for the web debugger.

WHY THIS FILE EXISTS
--------------------
Reading the REAL loss code (distillspec_gbv/losses/tree_losses.py) confirms three
gradient weaknesses that are documented right in the source:

  1. bv_tree_loss / gbv_tree_loss DETACH the chain weight inside the block
     integral (`F.relu(w.detach()*p - q)`, `1 - w.detach()`) AND detach the
     survival product (`survival * (1 - h.detach())`).  The unified OT scaffold
     `_verifier_aligned_tree_loss` ALSO detaches survival (`survival * alpha.detach()`).
     → only the most-recent factor carries gradient; the leaf term is a product of
       L small α's → vanishing for large L.  With NO KL anchor the signal is weak.
  2. The loss clamps each FACTOR `Π min(1, p/q)` (monotone, deficit never recovered)
     while bv_verify clamps the PRODUCT `min(1, w·p/q)` (a token with p/q>1 can
     restore weight).  → the loss is a valid surrogate but not the verifier weight.
  3. spectr_tree DETACHES ρ; khisti_tree uses an LP-free softmax surrogate.

These are *deliberate* choices in the production loss (speed / stability) and we do
NOT touch them.  Instead this module offers variants that "mathematically stick" —
they restore a dense, non-vanishing, correctly-directed gradient — so you can A/B
them live in web_debugger.py against the production losses on the toy models.

Every variant reuses the REAL helpers (`_gbv_select_path`,
`_compute_q_skew_along_path`, `_alpha_*`, `kl_tree_loss`, …) imported from the
production module, so the source of truth stays the source of truth.

THE FIXES
---------
* `+kl` anchor   add  λ · mean KL(p‖q)  (the SAME term ebe_tree already carries).
                 KL(p‖q) is dense (every node, full vocab) and its gradient pushes
                 q→p, which monotonically raises every verifier's acceptance α.
                 This is the single most robust fix.
* `fullgrad`     stop detaching w and the survival product → the exact gradient of
                 the −E[τ] surrogate (every h_i and the survival telescoping now
                 back-propagate).  Higher variance, but nothing is truncated.
* `faithful`     use the verifier's PRODUCT clamp  w_i = min(1, w_{i-1}·p/q)
                 (recovery allowed, matches bv_verify) + fullgrad + KL anchor.
                 The closest-to-correct BV/GBV training objective.
* `ift` (spectr) give ρ a real gradient via the implicit-function theorem
                 (differentiable fixed point) instead of detaching it.

Each variant returns a scalar tensor with grad_fn, signature
    fn(q_probs_dict, p_probs_dict, q_paths, L, K) -> Tensor
matching compute_tree_loss, so the harness/debugger can call them interchangeably.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List

_HERE = os.path.dirname(os.path.abspath(__file__))
_ALGO = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "algorithms")
if _ALGO not in sys.path:
    sys.path.insert(0, _ALGO)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from distillspec_gbv.losses.tree_losses import (  # noqa: E402
    compute_tree_loss, TREE_LOSS_NAMES,
    kl_tree_loss,
    bv_tree_loss_all_paths, gbv_tree_loss,
    spectr_tree_loss, khisti_tree_loss, traversal_tree_loss,
    _gbv_select_path, _compute_q_skew_along_path,
    _alpha_naive, _verifier_aligned_tree_loss,
)

DEFAULT_LAMBDA = 0.1   # same weight ebe_tree uses for its KL regulariser

# Per-variant recommended λ defaults. Reflects how much gradient detachment the
# primary loss has: fully-detached losses need λ=1.0; partially-detached use 0.3–0.5;
# full-gradient variants use 0.1 (KL is extra regularisation, not the sole signal).
VARIANT_LAM_DEFAULTS: Dict[str, float] = {
    "bv_tree_kl":       0.5,   # BV detaches chain weights upstream
    "bv_tree_fullgrad": 0.1,   # full gradient, no detachment — KL is extra regularisation
    "bv_tree_faithful": 0.3,   # product-clamp version; partial detach
    "gbv_tree_kl":      0.5,   # GBV selects one path, detaches others — KL covers 2/3 of nodes
    "gbv_tree_faithful":0.3,   # similar to gbv_kl but product clamp
    "spectr_tree_ift":  0.3,   # IFT grad through ρ; partial grad
    "spectr_tree_kl":   1.0,   # ρ fully detached — KL is sole signal for ρ terms
    "khisti_tree_kl":   1.0,   # LP fully detached — KL is literally the only gradient
    "traversal_tree_kl":0.5,   # leaf-weight; terminal nodes have no grad
}


# ---------------------------------------------------------------------------
# Shared pieces
# ---------------------------------------------------------------------------

def _kl_anchor(q_dict, p_dict) -> torch.Tensor:
    """Mean KL(p‖q) over tree nodes — identical to kl_tree_loss (reused)."""
    return kl_tree_loss(q_dict, p_dict)


def _bv_surrogate_path(q_dict, p_dict, path, L,
                       product_clamp: bool, detach: bool) -> torch.Tensor:
    """−E[τ_BV] along one path, with two switches.

    product_clamp=False, detach=True  ->  EXACTLY the production bv_tree_loss.
    product_clamp=True                 ->  verifier-faithful weight (recovery allowed).
    detach=False                       ->  full pathwise gradient (nothing truncated).
    """
    device = next(iter(q_dict.values())).device
    w = torch.ones(1, device=device)
    h_list: List[torch.Tensor] = []

    for i in range(1, L + 1):
        prefix = ",".join(str(x) for x in path[:i])
        if prefix not in q_dict or prefix not in p_dict:
            break
        token = path[i]
        p = p_dict[prefix].detach().to(w.dtype)
        q = q_dict[prefix].to(w.dtype)

        ratio = p[token] / q[token].clamp(min=1e-9)
        if product_clamp:
            w = torch.clamp(w * ratio, max=1.0)        # clamp the PRODUCT (verifier)
        else:
            w = w * torch.clamp(ratio, max=1.0)        # clamp each FACTOR (loss)

        if i < L:
            wp = w if not detach else w.detach()
            residual = F.relu(wp * p - q)
            num = residual.sum()
            h = num / (num + 1.0 - wp + 1e-10)
        else:
            h = w
        h_list.append(h)

    if not h_list:
        return torch.zeros(1, device=device, requires_grad=True)

    e_tau = torch.zeros(1, device=device)
    survival = torch.ones(1, device=device)
    for i, h in enumerate(h_list):
        e_tau = e_tau + (i + 1) * survival * h
        survival = survival * (1.0 - (h if not detach else h.detach()))
    return -e_tau


def _bv_surrogate_all(q_dict, p_dict, q_paths, L,
                      product_clamp: bool, detach: bool) -> torch.Tensor:
    device = next(iter(q_dict.values())).device
    total = torch.zeros(1, device=device)
    for path in q_paths:
        total = total + _bv_surrogate_path(q_dict, p_dict, path, L, product_clamp, detach)
    return total / max(len(q_paths), 1)


def _alpha_spectr_ift(p: torch.Tensor, q: torch.Tensor, K: int) -> torch.Tensor:
    """SpecTr acceptance α with a REAL gradient through ρ via the implicit
    function theorem (differentiable fixed point), instead of detaching ρ.

    ρ solves  F(ρ,q) = 1 − (1−β)^K − ρ·β = 0,  β = Σ min(p/ρ, q).
    IFT:  dρ/dq = −(∂F/∂q)/(∂F/∂ρ).  We realise this with the standard
    differentiable-fixed-point trick

        ρ̃ = ρ* + (F(ρ*,q).detach() − F(ρ*,q)) / (∂F/∂ρ)|_{ρ*}

    whose value is ρ* (the bracket is 0) but whose gradient w.r.t. q equals the
    IFT gradient.  ρ* and ∂F/∂ρ are computed detached.
    """
    if K <= 1:
        return _alpha_naive(p, q, K)

    eps = 1e-6
    # ── ρ* by binary search (no grad) ─────────────────────────────────────────
    with torch.no_grad():
        q_det = q.detach()
        lo, hi = 1.0, float(K)
        for _ in range(25):
            rho = 0.5 * (lo + hi)
            beta_v = float(torch.minimum(p / rho, q_det).sum())
            if (1.0 - (1.0 - beta_v) ** K) >= rho * beta_v:
                lo = rho
            else:
                hi = rho
        rho_star = hi
        beta_star = float(torch.minimum(p / rho_star, q_det).sum())
        # ∂β/∂ρ = Σ_{v: p/ρ < q} (−p/ρ²)
        mask = (p / rho_star < q_det).to(p.dtype)
        dbeta_drho = float((-(p) / (rho_star ** 2) * mask).sum())
        # ∂F/∂ρ = K(1−β)^{K−1}·∂β/∂ρ − β − ρ·∂β/∂ρ
        dF_drho = (K * (1.0 - beta_star) ** (K - 1) * dbeta_drho
                   - beta_star - rho_star * dbeta_drho)
        if abs(dF_drho) < 1e-6:
            dF_drho = 1e-6 if dF_drho >= 0 else -1e-6

    # ── differentiable ρ̃ (value = ρ*, grad = IFT) ────────────────────────────
    beta_q = torch.minimum(p / rho_star, q).sum()             # grad through q
    F_q = 1.0 - (1.0 - beta_q) ** K - rho_star * beta_q
    rho = rho_star + (F_q.detach() - F_q) / dF_drho

    # ── α with grad flowing through q AND ρ(q) ────────────────────────────────
    beta = torch.minimum(p / rho, q).sum()
    if float(beta) >= 1.0:
        return torch.ones(1, device=p.device, dtype=p.dtype)
    p_acc = 1.0 - (1.0 - beta) ** K
    one_minus_beta = (1.0 - beta).clamp(min=eps)
    p_acc_over_beta = p_acc / beta.clamp(min=eps)
    p_res_raw = F.relu(p - torch.minimum(p / rho, q) * p_acc_over_beta)
    p_res = p_res_raw / p_res_raw.sum().clamp(min=eps)
    r = F.relu(q - p / rho) / one_minus_beta
    return p_acc + (1.0 - p_acc) * (p_res * (1.0 - (1.0 - r) ** K)).sum()


# ---------------------------------------------------------------------------
# The variants
# ---------------------------------------------------------------------------

# Every variant takes the SAME (q, p, paths, L, K, lam) signature.  `lam` is the
# weight of the KL(p‖q) anchor; set lam=0 to see the raw variant, raise it (via the
# web slider) to watch a degrading variant become a learner.

def bv_tree_kl(q, p, paths, L, K, lam=0.5):
    """REAL bv_tree (factor clamp, detached grad) + λ·KL anchor."""
    return bv_tree_loss_all_paths(q, p, paths, L) + lam * _kl_anchor(q, p)


def bv_tree_fullgrad(q, p, paths, L, K, lam=0.1):
    """bv surrogate with NO detach (full pathwise gradient) + optional λ·KL.
    Full gradient means KL is extra regularisation, not the sole signal."""
    base = _bv_surrogate_all(q, p, paths, L, product_clamp=False, detach=False)
    return base + lam * _kl_anchor(q, p) if lam > 0 else base


def bv_tree_faithful(q, p, paths, L, K, lam=0.3):
    """Verifier-correct PRODUCT clamp  w_i=min(1, w_{i-1}·p/q)  (recovery allowed)
    on the proven detached gradient + λ·KL.  The product clamp keeps w larger, so
    this variant needs a LARGER λ than bv_tree_kl to let the anchor lead."""
    return (_bv_surrogate_all(q, p, paths, L, product_clamp=True, detach=True)
            + lam * _kl_anchor(q, p))


def gbv_tree_kl(q, p, paths, L, K, lam=0.5):
    """REAL gbv_tree + λ·KL anchor."""
    return gbv_tree_loss(q, p, paths, L, K) + lam * _kl_anchor(q, p)


def gbv_tree_faithful(q, p, paths, L, K, lam=0.3):
    """GBV select (no grad) → q_skew (with grad) → product-clamp BV + λ·KL."""
    with torch.no_grad():
        best = _gbv_select_path(q, p, paths, L)
    q_skew = _compute_q_skew_along_path(q, p, best, L, K)
    if not q_skew:
        return torch.zeros(1, device=next(iter(q.values())).device, requires_grad=True)
    return (_bv_surrogate_path(q_skew, p, best, L, product_clamp=True, detach=True)
            + lam * _kl_anchor(q, p))


def spectr_tree_ift(q, p, paths, L, K, lam=0.3):
    """SpecTr with a REAL gradient through ρ (implicit function theorem) + λ·KL.
    IFT grad through ρ gives partial gradient; anchor complements it."""
    base = _verifier_aligned_tree_loss(_alpha_spectr_ift, q, p, paths, L, K)
    return base + lam * _kl_anchor(q, p) if lam > 0 else base


def spectr_tree_kl(q, p, paths, L, K, lam=1.0):
    """REAL spectr_tree (ρ detached) + λ·KL anchor.
    ρ fully detached → KL is sole training signal for ρ terms; needs λ=1.0."""
    return spectr_tree_loss(q, p, paths, L, K) + lam * _kl_anchor(q, p)


def khisti_tree_kl(q, p, paths, L, K, lam=1.0):
    """REAL khisti_tree (LP-free surrogate) + λ·KL anchor.
    LP fully detached → KL is literally the only gradient; needs λ=1.0."""
    return khisti_tree_loss(q, p, paths, L, K) + lam * _kl_anchor(q, p)


def traversal_tree_kl(q, p, paths, L, K, lam=0.5):
    """REAL traversal_tree (leaf-weight surrogate) + λ·KL anchor."""
    return traversal_tree_loss(q, p, paths, L, K) + lam * _kl_anchor(q, p)


# name -> (fn, base_loss_for_trace, matched_verifier, short description)
VARIANTS = {
    "bv_tree_kl":       (bv_tree_kl,       "bv_tree",        "bv",     "REAL bv_tree + λ·KL(p‖q) anchor — fixes the weak signal"),
    "bv_tree_fullgrad": (bv_tree_fullgrad, "bv_tree",        "bv",     "bv surrogate, NO detach (full pathwise grad); experiment"),
    "bv_tree_faithful": (bv_tree_faithful, "bv_tree",        "bv",     "verifier-correct product clamp + detached grad + λ·KL (needs larger λ)"),
    "gbv_tree_kl":      (gbv_tree_kl,      "gbv_tree",       "gbv",    "REAL gbv_tree + λ·KL anchor"),
    "gbv_tree_faithful":(gbv_tree_faithful,"gbv_tree",       "gbv",    "GBV skew + product-clamp BV + λ·KL"),
    "spectr_tree_ift":  (spectr_tree_ift,  "spectr_tree",    "spectr", "SpecTr with IFT gradient through ρ (not detached)"),
    "spectr_tree_kl":   (spectr_tree_kl,   "spectr_tree",    "spectr", "REAL spectr_tree + λ·KL anchor"),
    "khisti_tree_kl":   (khisti_tree_kl,   "khisti_tree",    "khisti", "REAL khisti_tree + λ·KL anchor"),
    "traversal_tree_kl":(traversal_tree_kl,"traversal_tree", "traversal","REAL traversal_tree + λ·KL anchor"),
}


def is_variant(name: str) -> bool:
    return name in VARIANTS


def base_loss(name: str) -> str:
    return VARIANTS[name][1] if name in VARIANTS else name


def matched_verifier(name: str) -> str:
    if name in VARIANTS:
        return VARIANTS[name][2]
    from tree_harness import LOSS_TO_VERIFIER
    return LOSS_TO_VERIFIER.get(name, "gbv")


def variant_descriptions() -> Dict[str, str]:
    return {k: v[3] for k, v in VARIANTS.items()}


def compute_any(name: str, q, p, paths, L, K, lam: float = DEFAULT_LAMBDA) -> torch.Tensor:
    """Dispatch to a production loss or a variant — same signature for both.
    `lam` (KL anchor weight) is forwarded only to variants; production losses
    ignore it.  Callers that want the per-variant recommended default should
    pass `lam=VARIANT_LAM_DEFAULTS.get(name, DEFAULT_LAMBDA)`."""
    if name in VARIANTS:
        return VARIANTS[name][0](q, p, paths, L, K, lam)
    return compute_tree_loss(name, q, p, paths, L, K)
