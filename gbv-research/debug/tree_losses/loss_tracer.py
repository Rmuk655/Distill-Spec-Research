"""
loss_tracer.py — open up the black box of each tree loss.

For a given tree it re-derives, node-by-node, the SAME quantities the real loss
computes (chain weight w_i, block acceptance h_i, per-node acceptance α_V,
the telescoping E[τ] = Σ_i Π_{j≤i} α_j, …) and packages them into a plain dict
for printing / plotting.

To guarantee the tracer never drifts from the implementation it explains, every
tracer ALSO calls the real compute_tree_loss on the identical inputs and reports
both numbers; visual_debugger asserts they agree. The real loss functions and
their private α-helpers are imported directly — the tracer is a *mirror*, not a
reimplementation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Dict, List

from tree_harness import compute_tree_loss  # noqa: F401  (re-exported convenience)
from distillspec_gbv.losses.tree_losses import (
    _alpha_naive, _alpha_nss, _alpha_specinfer, _alpha_spectr, _alpha_khisti,
    _gbv_select_path, _compute_q_skew_along_path,
)

_ALPHA_FN = {
    "naive_tree":     _alpha_naive,
    "nss_tree":       _alpha_nss,
    "specinfer_tree": _alpha_specinfer,
    "spectr_tree":    _alpha_spectr,
    "khisti_tree":    _alpha_khisti,
}

NODE_DIVERGENCES = {"kl_tree", "rev_kl_tree", "jsd_tree"}


def _f(x) -> float:
    return float(x.detach().item()) if torch.is_tensor(x) else float(x)


# ---------------------------------------------------------------------------
# Node-level divergences (kl / rev_kl / jsd)
# ---------------------------------------------------------------------------

def _trace_node_divergence(name, q_dict, p_dict):
    rows = []
    for pfx, q in q_dict.items():
        if pfx not in p_dict:
            continue
        p = p_dict[pfx].to(q.dtype)
        log_q = q.log().clamp(min=-100.0)
        log_p = p.log().clamp(min=-100.0)
        if name == "kl_tree":
            term = (p * (log_p - log_q)).sum()        # KL(p‖q)
        elif name == "rev_kl_tree":
            term = (q * (log_q - log_p)).sum()        # KL(q‖p)
        else:  # jsd_tree, α=0.5
            m = 0.5 * q + 0.5 * p
            log_m = m.log().clamp(min=-100.0)
            term = 0.5 * (q * (log_q - log_m)).sum() + 0.5 * (p * (log_p - log_m)).sum()
        rows.append({"prefix": pfx, "term": _f(term)})
    mean = sum(r["term"] for r in rows) / max(len(rows), 1)
    return {"kind": "node_divergence", "rows": rows, "traced_value": mean}


# ---------------------------------------------------------------------------
# BV chain-weight / block-acceptance trace (also the core of GBV)
# ---------------------------------------------------------------------------

def _trace_bv_path(q_dict, p_dict, path, L):
    """Mirror bv_tree_loss along one path. Returns per-depth rows + E[τ]."""
    rows = []
    w = 1.0
    h_list = []
    for i in range(1, L + 1):
        prefix = ",".join(str(x) for x in path[:i])
        if prefix not in q_dict or prefix not in p_dict:
            break
        token = path[i]
        p = p_dict[prefix]
        q = q_dict[prefix]
        alpha = min(1.0, _f(p[token]) / max(_f(q[token]), 1e-9))
        w = w * alpha
        if i < L:
            residual = F.relu(w * p - q)
            num = _f(residual.sum())
            h = num / (num + 1.0 - w + 1e-10)
        else:
            h = w
        h_list.append(h)
        rows.append({"depth": i, "token": token, "alpha": alpha, "w": w, "h": h})
    # E[τ_BV] = Σ_i (i) · h_i · Π_{j<i}(1 - h_j)
    e_tau, survival = 0.0, 1.0
    for i, h in enumerate(h_list):
        e_tau += (i + 1) * survival * h
        survival *= (1.0 - h)
    return rows, e_tau


def _trace_bv(q_dict, p_dict, q_paths, L):
    # bv_tree_loss_all_paths averages -E[τ] over all K paths
    per_path = []
    e_taus = []
    for path in q_paths:
        rows, e_tau = _trace_bv_path(q_dict, p_dict, path, L)
        per_path.append({"path": path, "rows": rows, "E_tau": e_tau})
        e_taus.append(e_tau)
    mean_e_tau = sum(e_taus) / max(len(e_taus), 1)
    return {"kind": "bv", "per_path": per_path, "traced_value": -mean_e_tau,
            "E_tau": mean_e_tau}


def _trace_gbv(q_dict, p_dict, q_paths, L, K):
    with torch.no_grad():
        best_path = _gbv_select_path(q_dict, p_dict, q_paths, L)
    q_skew = _compute_q_skew_along_path(q_dict, p_dict, best_path, L, K)
    rows, e_tau = _trace_bv_path(q_skew, p_dict, best_path, L)
    return {"kind": "gbv", "selected_path": best_path, "rows": rows,
            "traced_value": -e_tau, "E_tau": e_tau,
            "note": "BV run on the GBV-selected path with q replaced by q_skew"}


# ---------------------------------------------------------------------------
# Verifier-aligned OT path-product trace (naive/nss/specinfer/spectr/khisti)
# ---------------------------------------------------------------------------

def _trace_alpha_product(name, q_dict, p_dict, q_paths, L, K):
    alpha_fn = _ALPHA_FN[name]
    per_path, e_taus = [], []
    for path in q_paths:
        rows = []
        e_tau, survival = 0.0, 1.0
        for i in range(1, L + 1):
            prefix = ",".join(str(x) for x in path[:i])
            if prefix not in q_dict or prefix not in p_dict:
                break
            q = q_dict[prefix]
            p = p_dict[prefix].to(q.dtype)
            alpha = _f(alpha_fn(p, q, K))
            e_tau += survival * alpha          # contribution Π_{j≤i} α_j
            survival *= alpha
            rows.append({"depth": i, "token": path[i], "alpha": alpha,
                         "cum_product": e_tau})
        if rows:
            per_path.append({"path": path, "rows": rows, "E_tau": e_tau})
            e_taus.append(e_tau)
    mean_e_tau = sum(e_taus) / max(len(e_taus), 1)
    return {"kind": "alpha_product", "per_path": per_path,
            "traced_value": -mean_e_tau, "E_tau": mean_e_tau}


# ---------------------------------------------------------------------------
# EBE (single-path naive E[τ], on-policy) and traversal (leaf-weight surrogate)
# ---------------------------------------------------------------------------

def _trace_ebe(q_dict, p_dict, q_paths, L, K, kl_weight=0.1):
    per_path, ebe_vals = [], []
    total_kl, n_nodes = 0.0, 0
    for path in q_paths:
        rows, alphas = [], []
        for i in range(1, L + 1):
            prefix = ",".join(str(x) for x in path[:i])
            if i >= len(path) or prefix not in q_dict or prefix not in p_dict:
                break
            token = path[i]
            p, q = p_dict[prefix], q_dict[prefix]
            alpha = min(1.0, _f(p[token]) / max(_f(q[token]), 1e-9))
            alphas.append(alpha)
            # same KL regulariser the real ebe_tree adds (KL(p‖q) at each visited node)
            if kl_weight > 0.0:
                total_kl += _f((p * (p.log().clamp(min=-100.0) - q.log().clamp(min=-100.0))).sum())
                n_nodes += 1
            cum = 1.0
            for a in alphas:
                cum *= a
            rows.append({"depth": i, "token": token, "alpha": alpha, "prefix_prod": cum})
        if alphas:
            cumprod, run = [], 1.0
            for a in alphas:
                run *= a
                cumprod.append(run)
            ebe = 1.0 + sum(cumprod)         # E[τ_EBE] = 1 + Σ_k Π_{i≤k} α_i
            per_path.append({"path": path, "rows": rows, "E_tau": ebe})
            ebe_vals.append(ebe)
    mean = sum(ebe_vals) / max(len(ebe_vals), 1)
    kl_term = kl_weight * (total_kl / n_nodes) if (kl_weight > 0.0 and n_nodes) else 0.0
    # real ebe_tree returns  -mean(E[τ])  +  kl_weight · mean KL(p‖q)
    return {"kind": "ebe", "per_path": per_path, "traced_value": -mean + kl_term,
            "E_tau": mean, "kl_term": kl_term,
            "note": f"includes KL regulariser λ={kl_weight}"}


def _trace_traversal(q_dict, p_dict, q_paths, L, K):
    per_path, weights = [], []
    for path in q_paths:
        rows, w = [], 1.0
        for i in range(1, L + 1):
            prefix = ",".join(str(x) for x in path[:i])
            if prefix not in q_dict or prefix not in p_dict:
                break
            token = path[i]
            p, q = p_dict[prefix], q_dict[prefix]
            alpha = min(1.0, _f(p[token]) / max(_f(q[token]), 1e-9))
            w *= alpha
            rows.append({"depth": i, "token": token, "alpha": alpha, "w_leaf": w})
        per_path.append({"path": path, "rows": rows, "w_leaf": w})
        weights.append(w)
    mean_w = sum(weights) / max(len(weights), 1)
    return {"kind": "traversal", "per_path": per_path, "traced_value": -mean_w,
            "mean_w_leaf": mean_w,
            "note": "surrogate maximises mean leaf weight; DFS rejection not modelled"}


# ---------------------------------------------------------------------------
# Public dispatch
# ---------------------------------------------------------------------------

def trace(loss_name, q_dict, p_dict, q_paths, L, K):
    """Return a structured trace dict for `loss_name`, plus `real_value` from
    the actual compute_tree_loss for cross-checking."""
    if loss_name in NODE_DIVERGENCES:
        tr = _trace_node_divergence(loss_name, q_dict, p_dict)
    elif loss_name == "bv_tree":
        tr = _trace_bv(q_dict, p_dict, q_paths, L)
    elif loss_name == "gbv_tree":
        tr = _trace_gbv(q_dict, p_dict, q_paths, L, K)
    elif loss_name == "traversal_tree":
        tr = _trace_traversal(q_dict, p_dict, q_paths, L, K)
    elif loss_name == "ebe_tree":
        tr = _trace_ebe(q_dict, p_dict, q_paths, L, K)
    elif loss_name in _ALPHA_FN:
        tr = _trace_alpha_product(loss_name, q_dict, p_dict, q_paths, L, K)
    else:
        # Give a helpful message if the user accidentally passed a flat loss name
        _FLAT = {"forward_kl", "reverse_kl", "jsd", "l1", "ebe", "ebe_single"}
        if loss_name in _FLAT:
            tree_equiv = loss_name + "_tree" if loss_name not in ("forward_kl", "reverse_kl", "jsd", "l1") else None
            hint = f"  Did you mean '{tree_equiv}'?" if tree_equiv else ""
            raise ValueError(
                f"'{loss_name}' is a flat (sequence-level) loss — it has no tree tracer. "
                f"This debugger only supports tree-structured losses.{hint}"
            )
        raise ValueError(f"no tracer for '{loss_name}' — supported: "
                         f"{sorted(NODE_DIVERGENCES | set(_ALPHA_FN) | {'bv_tree','gbv_tree','traversal_tree','ebe_tree'})}")

    real = compute_tree_loss(loss_name, q_dict, p_dict, q_paths, L=L, K=K)
    tr["loss_name"] = loss_name
    tr["real_value"] = _f(real)
    return tr
