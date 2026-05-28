"""
Tree-structured draft forward pass for training.

During inference the verifiers (BV, GBV, Traversal) consume per-node
full-vocabulary distributions q_probs_dict / p_probs_dict built by
iid_draft + target_tree_pass.  This module provides the training-side
analogue: re-run the draft model over a *fixed* sampled tree WITH
gradients, so that tree-aware losses can back-propagate through the
student's per-node distributions.

Key design:
  - The sampled paths (q_paths) are treated as fixed inputs (no grad
    through the sampling step itself).
  - A single tree-attention forward pass — same mask as target_tree_pass —
    gives every tree node's distribution in one shot.
  - Gradient flows through softmax(logits/temp) at each node, all the
    way back to the LoRA parameters.

Compatibility note:
  Requires the same {"full_attention": mask} dict-style attention that
  target_tree_pass uses.  Verified against Qwen3 family.  Other families
  need the same attention_type == "full_attention" attribute on their
  attention layers (checked in load_models in verifiers/utils.py).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Dict, List


def draft_tree_forward_with_grad(
    draft_model,
    prompt_ids: torch.Tensor,        # [1, plen]  — tokenised prompt
    q_paths: List[List[int]],        # K paths of length L+1 (first token = pending)
    L: int,
    K: int,
    q_temp: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """
    Re-run the draft model over the fixed sampled tree WITH gradients.

    Analogous to target_tree_pass() in verifiers/draft_generator.py, but:
      - Uses the draft model instead of the target model.
      - Keeps gradient-enabled logits (no torch.no_grad wrapper).
      - Returns only non-leaf nodes (depth 0 … L-1) because those are the
        nodes the verifiers query q_probs_dict at.

    Args:
        draft_model:  The trainable student model (with LoRA adapters).
        prompt_ids:   Tokenised prompt, shape [1, plen].
        q_paths:      K paths, each of length L+1.  Produced by iid_draft().
                      The first element of every path is the shared pending token.
        L:            Draft block length (depth of tree).
        K:            Number of draft paths.
        q_temp:       Draft sampling temperature (same as used in iid_draft).

    Returns:
        q_probs_dict_grad:  Dict mapping prefix strings (e.g. "42,7,19") to
                            [V] float32 tensors WITH requires_grad=True.
                            Only contains non-leaf prefixes (depth < L).
    """
    device = prompt_ids.device
    dtype  = next(draft_model.parameters()).dtype

    # ── Build unique prefix list (same ordering as target_tree_pass) ──────────
    q_prefixes: List[str] = []
    q_token_ids: List[int] = []
    for path in q_paths:
        for i, tok in enumerate(path):
            pfx = ",".join(str(x) for x in path[:i + 1])
            if pfx not in q_prefixes:
                q_prefixes.append(pfx)
                q_token_ids.append(tok)

    n_nodes   = len(q_prefixes)
    q_tokens  = torch.tensor(q_token_ids, device=device, dtype=torch.long).unsqueeze(0)
    # q_tokens: [1, n_nodes]

    # ── Prompt KV cache (frozen context — no grad needed) ────────────────────
    with torch.no_grad():
        prompt_out  = draft_model(prompt_ids, use_cache=True, return_dict=True)
        p_cache     = prompt_out.past_key_values
        cached_len  = prompt_ids.shape[1]

    # ── Tree attention mask ───────────────────────────────────────────────────
    # Each tree node attends to all prompt tokens and all its ancestors
    # (including itself).  All other cross-node entries are -inf.
    # Shape: [1, 1, n_nodes, cached_len + n_nodes] — same as target_tree_pass.
    mask = torch.zeros(
        (n_nodes, cached_len + n_nodes), device=device, dtype=dtype
    )
    mask[:, cached_len:] = torch.finfo(dtype).min          # no cross-node attn by default

    for i, pi in enumerate(q_prefixes):
        for j, pj in enumerate(q_prefixes):
            # Node i attends to node j iff j is an ancestor of i (or j == i).
            if pi.startswith(pj + ",") or pi == pj:
                mask[i, cached_len + j] = 0.0

    mask = mask.unsqueeze(0).unsqueeze(0)                   # [1, 1, n_nodes, total]

    # ── Tree forward pass WITH grad ───────────────────────────────────────────
    draft_model.train()
    out   = draft_model(
        q_tokens,
        past_key_values=p_cache,
        attention_mask={"full_attention": mask},
        use_cache=True,          # must be True for past_key_values to be honoured
        return_dict=True,
    )
    # out.logits: [1, n_nodes, vocab_size]
    logits = out.logits[0].float()                          # [n_nodes, V], float32
    probs  = F.softmax(logits / q_temp, dim=-1)             # [n_nodes, V], WITH grad

    # ── Build q_probs_dict — non-leaf nodes only ──────────────────────────────
    # Depth of prefix "t0,t1,...,td" is d (0-indexed from root).
    # Leaf depth = L.  We only need depths 0 … L-1.
    q_probs_dict_grad: Dict[str, torch.Tensor] = {}
    for i, pfx in enumerate(q_prefixes):
        depth = len(pfx.split(",")) - 1
        if depth < L:
            q_probs_dict_grad[pfx] = probs[i]              # [V] with grad

    return q_probs_dict_grad


def verify_tree_forward_grad(
    draft_model,
    prompt_ids: torch.Tensor,
    q_paths: List[List[int]],
    L: int,
    K: int,
    q_temp: float = 1.0,
) -> None:
    """
    Smoke test: run draft_tree_forward_with_grad and assert that every
    returned distribution has requires_grad=True.

    Call this once after model load in the tree-loss training path.
    Raises AssertionError with a helpful message if grad is not flowing.
    """
    q_probs_dict_grad = draft_tree_forward_with_grad(
        draft_model, prompt_ids, q_paths, L=L, K=K, q_temp=q_temp
    )

    if not q_probs_dict_grad:
        raise AssertionError(
            "draft_tree_forward_with_grad returned an empty dict — "
            "check that L > 0 and q_paths are non-empty."
        )

    failed = [pfx for pfx, dist in q_probs_dict_grad.items() if not dist.requires_grad]
    if failed:
        raise AssertionError(
            f"requires_grad=False on {len(failed)} node(s): {failed[:3]}{'...' if len(failed) > 3 else ''}. "
            "Likely cause: draft_model is in eval() with no LoRA parameters requiring grad, "
            "or torch.no_grad() context is active around this call."
        )

    sample_pfx  = next(iter(q_probs_dict_grad))
    sample_dist = q_probs_dict_grad[sample_pfx]
    print(
        f"[tree-grad] ✓  requires_grad=True on all {len(q_probs_dict_grad)} non-leaf node(s). "
        f"Sample: '{sample_pfx}'  shape={tuple(sample_dist.shape)}"
    )
    return q_probs_dict_grad
