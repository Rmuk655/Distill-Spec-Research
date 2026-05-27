import random
import torch
import json
import torch.nn.functional as F
from typing import List, Tuple, Dict, Mapping
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
try:
    from .utils import *
except ImportError:
    import os as _os, sys as _sys
    _here = _os.path.dirname(_os.path.abspath(__file__))
    if _here not in _sys.path:
        _sys.path.insert(0, _here)
    from utils import *  # type: ignore[no-redef]

"""
Constructs a draft tree via i.i.d. path sampling.
Takes in:
    - q_model: draft model, from which we form a draft tree
    - q_cache: draft KV-cache, each KV tensor has shape (1, num_heads, context_len - 1, head_dim)
    - context_pending: shape (1, 1), last token ID, which currently has not been committed to p_cache/q_cache
    - K: number of i.i.d. draft paths
    - L: draft block length
    - q_temp: draft sampling temperature
Returns:
    - q_paths: 2D list of shape (K, L + 1) containing indices for tokens on K drafted length-L paths (with pending token at start)
    - q_cache: updated draft KV-cache, each KV tensor now has shape (K, num_heads, context_len + L, head_dim)
    - q_probs_dict: dict mapping draft tree prefixes (e.g. "12,24,1780" or "12") to draft probability dist (torch.Tensor shape (V))
"""
def iid_draft(
    q_model: AutoModelForCausalLM,
    q_cache: DynamicCache,
    context_pending: torch.Tensor,
    K: int = 4, L: int = 8, q_temp: float = 1.0
) -> Tuple[List[List[int]], DynamicCache, Dict[str, torch.Tensor]]:
    q_probs_dict = {}

    # Initialize paths to start at pending token.
    q_paths = [[context_pending[0, 0].item()] for _ in range(K)]

    # Expand cache and tokens to batch size K for batched decoding.
    q_cache = expand_cache(q_cache, K)
    next_tokens = context_pending.expand(K, 1)

    # Perform L-step autoregressive batched draft decoding, updating the batch-K cache.
    for _ in range(L):
        q_out = q_model(
            next_tokens,
            use_cache=True,
            return_dict=True,
            past_key_values=q_cache,
        )
        q_cache = q_out.past_key_values
        q_probs = F.softmax(q_out.logits[:, -1, :] / q_temp, dim=-1)
        next_tokens = torch.multinomial(q_probs, num_samples=1)

        # Update path and probability info.
        for i in range(K):
            prefix_rep = ",".join(str(x) for x in q_paths[i])
            q_probs_dict[prefix_rep] = q_probs[i]
            q_paths[i].append(next_tokens[i, 0].item())

    # One more batched draft forward pass to fill in the draft cache rows for the paths' last tokens.
    q_out = q_model(
        next_tokens,
        use_cache=True,
        return_dict=True,
        past_key_values=q_cache,
    )
    q_cache = q_out.past_key_values
    return q_paths, q_cache, q_probs_dict



"""
Performs the batched forward target pass over the draft tree.
Takes in:
    - p_model: target model
    - p_cache: target KV-cache, each KV tensor has shape (1, num_heads, context_len - 1, head_dim)
    - q_paths: 2D list of shape (K, L + 1) containing indices for tokens on K drafted length-L paths (with pending token at start)
    - K: number of i.i.d. draft paths
    - L: draft block length
    - p_temp: target sampling temperature
Returns:
    - q_prefixes: list of prefixes (e.g. "12,24,1780" or "12") in the draft tree, in the same order that the target forward pass is performed
    - q_tokens: tensor of last tokens (ints) for draft tree prefixes, in the same order as q_prefixes
    - p_cache: updated draft KV-cache, each KV tensor now has shape (K, num_heads, context_len + L, head_dim)
    - p_probs_dict: dict mapping draft tree prefixes (e.g. "12,24,1780" or "12") to target probability dist (torch.Tensor shape (V))
"""
def target_tree_pass(
    p_model: AutoModelForCausalLM,
    p_cache: DynamicCache,
    q_paths: List[List[int]],
    K: int = 4, L: int = 8, p_temp: float = 1.0
) -> Tuple[List[str], torch.LongTensor, DynamicCache, Dict[str, torch.Tensor]]:
    device = p_model.device
    dtype = p_model.dtype

    # Build fixed ordering of draft tree prefixes, and extract corresponding last tokens of prefixes.
    q_prefixes = []
    q_tokens = []
    for q_path in q_paths:
        for i, q_token in enumerate(q_path):
            q_prefix = ",".join(str(x) for x in q_path[:i + 1])
            if q_prefix not in q_prefixes:
                q_prefixes.append(q_prefix)
                q_tokens.append(q_token)
    q_tokens = torch.LongTensor(q_tokens).to(device).unsqueeze(0)

    # Form attn mask to perform the forward pass on draft tree prefixes. Initially, draft tree nodes attend to only the cached tokens.
    cached_len = p_cache.layers[0].keys.shape[-2]
    added_len = q_tokens.shape[-1]
    mask = torch.zeros((added_len, cached_len + added_len), device=device, dtype=dtype)
    mask[:, cached_len:] = torch.finfo(dtype).min       # Negative infinity entries means no draft node attends to another.

    # Build tree-based portion of attn mask, such that draft tree nodes now also attend to ancestor nodes (subprefixes).
    for i, q_prefix_i in enumerate(q_prefixes):
        for j, q_prefix_j in enumerate(q_prefixes):
            if q_prefix_i.startswith(q_prefix_j + ",") or q_prefix_i == q_prefix_j:
                mask[i, cached_len + j] = 0.0
    mask = mask.unsqueeze(0).unsqueeze(0)

    # Perform the batched target model forward pass with the tree attn mask, updating the target KV-cache.
    p_out = p_model(
        q_tokens,
        use_cache=True,
        return_dict=True,
        past_key_values=p_cache,
        attention_mask={"full_attention": mask},
    )
    p_cache = p_out.past_key_values

    # Update the target probability info.
    p_probs = F.softmax(p_out.logits / p_temp, dim=-1)
    p_probs_dict = {}
    for i, q_prefix in enumerate(q_prefixes):
        p_probs_dict[q_prefix] = p_probs[0, i]

    return q_prefixes, q_tokens, p_cache, p_probs_dict
