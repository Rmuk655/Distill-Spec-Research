"""
delayed_draft.py — delayed-expansion draft and eval loop.

Eval-time composition experiment: run existing checkpoints under DDTE-style
delayed branching (single path for L1 steps, then K branches for L-L1 steps)
without retraining.  Tests whether tree structure alone can stack with enrich
training gains.

Uses Rahul's iid_draft / target_tree_pass / verifiers unchanged via read-only
imports.  Mirrors the pattern in verifier_safe.py: researcher code is never
modified, only called.

Usage via eval.py:
    python eval.py --checkpoint <ckpt> --modes traversal,specinfer --L1 3

When L1=0 (default), eval.py uses the original speculative_decoding_loop
from main.py — no behaviour change.

Option A (lagged teacher entropy, zero extra target pass cost):
    After each iteration, teacher entropy at depth 1 is read from the
    current iteration's target pass and stored on p_model._delayed_entropy.
    The NEXT iteration uses it to pick L1: low entropy → branch late (high L1);
    high entropy → branch early (low L1 or no delay).
    Set --L1_adaptive to enable; --L1 becomes L1_max in that mode.
"""
import time
import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional

from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from verifiers.inference_util import iid_draft, target_tree_pass
from verifiers.util import slice_cache
from verifier_safe import safe_verify, VerifierError


def delayed_iid_draft(
    q_model: AutoModelForCausalLM,
    q_cache: DynamicCache,
    context_pending: torch.Tensor,
    K: int = 4,
    L: int = 8,
    q_temp: float = 1.0,
    L1: int = 0,
) -> Tuple[List[List[int]], DynamicCache, Dict[str, torch.Tensor]]:
    """
    Delayed-expansion draft: single path for L1 steps, then K i.i.d. branches
    for the remaining L-L1 steps.

    Falls back to iid_draft (normal root branching) when L1 <= 0 or L1 >= L.

    Returns (q_paths, q_cache, q_probs_dict) — same format as iid_draft so it
    is a drop-in replacement in any speculative decoding iter.

    Cache note: iid_draft's extra final forward pass adds the last token's KV to
    the cache.  We trim that entry via slice_cache before phase 2 so phase 2's
    iid_draft processes it fresh across K expanded copies (batch 1 → K).
    """
    if L1 <= 0 or L1 >= L:
        return iid_draft(q_model, q_cache, context_pending, K=K, L=L, q_temp=q_temp)

    # Phase 1: single path for L1 steps.
    single_paths, q_cache, single_probs = iid_draft(
        q_model, q_cache, context_pending, K=1, L=L1, q_temp=q_temp
    )
    # single_paths[0] = [pending, t1, ..., tL1]  length = L1+1
    # q_cache batch_dim=1, seq includes KV for tL1 (from iid_draft's extra fwd pass)

    # Trim tL1's KV so phase-2 iid_draft can re-process it from batch 1 → K correctly.
    seq_len = q_cache.layers[0].keys.shape[-2]
    q_cache = slice_cache(q_cache, batch_idx=[0], token_idx=list(range(seq_len - 1)))

    # Phase 2: K branches from tL1 for L-L1 more steps.
    branch_token = torch.tensor([[single_paths[0][-1]]], device=q_model.device)
    branch_paths, q_cache, branch_probs = iid_draft(
        q_model, q_cache, branch_token, K=K, L=L - L1, q_temp=q_temp
    )
    # branch_paths[k] = [tL1, b1_k, ..., b(L-L1)_k]  length = L-L1+1

    # Stitch: full path = stem (length L1+1) + branch tail (skip duplicate tL1).
    stem = single_paths[0]
    q_paths = [stem + branch_paths[k][1:] for k in range(K)]
    # q_paths[k] length = (L1+1) + (L-L1) = L+1  ✓

    # Remap branch_probs keys from local (branch_token-rooted) to global (context_pending-rooted).
    # branch_probs keys look like "3406", "3406,b1", ... but traversal_verify needs
    # "760,31925,3406", "760,31925,3406,b1", ... matching the full q_paths prefix format.
    stem_prefix = ",".join(str(x) for x in stem[:-1])  # stem without tL1, e.g. "760,31925"
    remapped_branch_probs = {stem_prefix + "," + k: v for k, v in branch_probs.items()}

    return q_paths, q_cache, {**single_probs, **remapped_branch_probs}


def _delayed_iter(
    p_model: AutoModelForCausalLM,
    q_model: AutoModelForCausalLM,
    p_cache: DynamicCache,
    q_cache: DynamicCache,
    context_cached: torch.Tensor,
    context_pending: torch.Tensor,
    verification_algo: str,
    K: int = 4,
    L: int = 8,
    p_temp: float = 1.0,
    q_temp: float = 1.0,
    L1: int = 0,
):
    """Single speculative decoding iteration with delayed-expansion draft.

    Identical to speculative_decoding_iter in main.py except step (1) calls
    delayed_iid_draft instead of iid_draft.  Profiling hooks are preserved so
    eval.py's stats collection works unchanged.
    """
    _run_stats = getattr(p_model, "_spec_run_stats", None)

    # (1) Delayed draft tree construction.
    if _run_stats is not None:
        torch.cuda.synchronize()
        _t_draft = time.perf_counter()
    q_paths, q_cache, q_probs_dict = delayed_iid_draft(
        q_model, q_cache, context_pending, K=K, L=L, q_temp=q_temp, L1=L1
    )
    if _run_stats is not None:
        torch.cuda.synchronize()
        _run_stats["time_draft"] += time.perf_counter() - _t_draft

    # (2) Batched target forward pass — unchanged.
    if _run_stats is not None:
        torch.cuda.synchronize()
        _t_target = time.perf_counter()
    q_prefixes, q_tokens, p_cache, p_probs_dict = target_tree_pass(
        p_model, p_cache, q_paths, K=K, L=L, p_temp=p_temp
    )
    if _run_stats is not None:
        torch.cuda.synchronize()
        _run_stats["time_target"] += time.perf_counter() - _t_target
        _run_stats["total_tree_nodes"] += len(q_prefixes)

    # Option A: store teacher entropy at depth 1 for next iteration's adaptive L1.
    # Reads from the target pass we just ran — zero extra compute.
    if q_prefixes:
        depth1_prefix = q_prefixes[0]  # first prefix = pending token (depth 0); depth 1 = q_prefixes[1] if exists
        # Use first depth-1 prefix available (any path, they share the stem under delayed branching).
        depth1_probs = p_probs_dict.get(depth1_prefix)
        if depth1_probs is not None:
            ent = -(depth1_probs * depth1_probs.clamp(min=1e-9).log()).sum().item()
            p_model._delayed_entropy = ent  # read by loop for next iter's adaptive L1

    # (3) Verification — routed through safe_verify (our wrapper, not Rahul's verifier directly).
    _dbg = getattr(p_model, "_spec_debug_ctx", {})
    if _run_stats is not None:
        torch.cuda.synchronize()
        _t_verify = time.perf_counter()
    ver_node, res_token = safe_verify(
        q_paths, q_prefixes, q_probs_dict, p_probs_dict, verification_algo,
        prompt=_dbg.get("prompt", "<unknown>"),
        K=_dbg.get("K"), L=_dbg.get("L"), p_temp=_dbg.get("p_temp"),
        prompt_idx=_dbg.get("prompt_idx"),
    )
    if _run_stats is not None:
        torch.cuda.synchronize()
        _run_stats["time_verify"] += time.perf_counter() - _t_verify

    tau = ver_node.depth
    ver_prefix = ver_node.rep
    path_idx = min(
        i for i, path in enumerate(q_paths)
        if ver_prefix == ",".join(str(x) for x in path[:tau + 1])
    )

    # (4) Cache and context update — identical to main.py.
    if _run_stats is not None:
        torch.cuda.synchronize()
        _t_cache = time.perf_counter()
    cached_len = context_cached.shape[-1]
    prefix_slice = [
        i for i, q_prefix in enumerate(q_prefixes)
        if ver_prefix.startswith(q_prefix + ",") or ver_prefix == q_prefix
    ]
    p_cache = slice_cache(p_cache, [0], list(range(cached_len)) + [(cached_len + x) for x in prefix_slice])
    q_cache = slice_cache(q_cache, [path_idx], list(range(cached_len + tau + 1)))
    context_cached = torch.cat([context_cached, q_tokens[:, prefix_slice]], dim=-1)
    context_pending[0, 0] = res_token
    if _run_stats is not None:
        torch.cuda.synchronize()
        _run_stats["time_cache"] += time.perf_counter() - _t_cache

    # KV peak tracking.
    if _run_stats is not None:
        _kv_target = sum(
            (lyr[0].numel() + lyr[1].numel()) * lyr[0].element_size()
            for lyr in p_cache
        )
        if _kv_target > _run_stats["kv_target_peak_bytes"]:
            _run_stats["kv_target_peak_bytes"] = _kv_target
        _kv_draft = sum(
            (lyr[0].numel() + lyr[1].numel()) * lyr[0].element_size()
            for lyr in q_cache
        )
        if _kv_draft > _run_stats["kv_draft_peak_bytes"]:
            _run_stats["kv_draft_peak_bytes"] = _kv_draft

    return p_cache, q_cache, context_cached, context_pending, tau


def delayed_speculative_decoding_loop(
    p_model: AutoModelForCausalLM,
    q_model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prompt: str,
    verification_algo: str,
    eos_token_id: Optional[int] = None,
    max_new_tokens: int = 128,
    K: int = 4,
    L: int = 8,
    p_temp: float = 1.0,
    q_temp: float = 1.0,
    L1: int = 0,
    L1_adaptive: bool = False,
    entropy_threshold: float = 1.5,
    L1_tau: bool = False,
):
    # In entropy-adaptive mode with no explicit L1, default branch depth to L//2.
    if L1_adaptive and L1 == 0:
        L1 = max(1, L // 2)
    # tau-lagged: initial L1 defaults to L//2 until first acceptance depth is observed.
    if L1_tau and L1 == 0:
        L1 = max(1, L // 2)
    """Full speculative decoding loop with delayed-expansion draft.

    Drop-in replacement for speculative_decoding_loop in main.py.
    Populates p_model._spec_profile["runs"] with the same structure so
    eval.py's stats collection works unchanged.

    Args:
        L1: fixed branch depth (0 = root branching, same as main.py).
        L1_adaptive: if True, use lagged teacher entropy to adapt L1 each
            iteration (Option A — zero extra target pass cost).  L1 becomes
            L1_max in this mode; iterations where teacher is uncertain use L1=0.
        entropy_threshold: nat threshold above which teacher is considered
            uncertain → fall back to L1=0 (root branching) for that iteration.
    """
    p_model._spec_debug_ctx = {
        "prompt": prompt,
        "K": K, "L": L, "p_temp": p_temp, "q_temp": q_temp,
        "prompt_idx": getattr(p_model, "_spec_prompt_idx", None),
    }

    if not hasattr(p_model, "_spec_profile"):
        p_model._spec_profile = {"runs": []}
    _run_stats = {
        "time_draft": 0.0, "time_target": 0.0,
        "time_verify": 0.0, "time_cache": 0.0,
        "total_tree_nodes": 0,
        "kv_target_peak_bytes": 0, "kv_draft_peak_bytes": 0,
    }
    p_model._spec_run_stats = _run_stats
    p_model._delayed_entropy = None  # reset entropy carry-over
    _prev_tau: Optional[int] = None  # for tau-lagged adaptive

    torch.cuda.synchronize()
    _loop_start = time.perf_counter()

    if eos_token_id is None:
        eos_token_id = p_model.config.eos_token_id

    context_cached = torch.tensor(tok.encode(prompt), device=p_model.device).unsqueeze(0)
    context_init_len = context_cached.shape[-1]

    p_out = p_model(context_cached, use_cache=True, return_dict=True)
    q_out = q_model(context_cached, use_cache=True, return_dict=True)
    p_cache = p_out.past_key_values
    q_cache = q_out.past_key_values

    p_probs_last = F.softmax(p_out.logits[:, -1, :] / p_temp, dim=-1)
    context_pending = torch.multinomial(p_probs_last, num_samples=1)

    target_calls = 0
    while context_cached.shape[-1] + 1 < context_init_len + max_new_tokens:
        # L1 selection for this iteration:
        #   Fixed:        always iter_L1 = L1 (may be 0 = root branch = same as main.py)
        #   tau-lagged:   iter_L1 = max(1, prev_tau - 1); prior accepted depth guides branch point
        #   entropy-adapt:iter_L1 = L1 when teacher confident; 0 (root) when uncertain
        if L1_tau:
            # Use previous iteration's accepted depth as branch point.
            # First iteration: use L//2 (already set in L1 above).
            iter_L1 = max(1, min(L - 1, _prev_tau - 1)) if _prev_tau is not None else L1
        elif L1_adaptive:
            iter_L1 = (0 if (p_model._delayed_entropy is None or
                             p_model._delayed_entropy > entropy_threshold) else L1)
        else:
            iter_L1 = L1

        p_cache, q_cache, context_cached, context_pending, _iter_tau = _delayed_iter(
            p_model, q_model, p_cache, q_cache,
            context_cached, context_pending,
            verification_algo,
            K=K, L=L, p_temp=p_temp, q_temp=q_temp, L1=iter_L1,
        )
        _prev_tau = _iter_tau  # update lagged acceptance depth for next iteration
        full_seq = torch.cat([context_cached, context_pending], dim=-1)
        target_calls += 1
        if (full_seq == eos_token_id).any():
            break

    torch.cuda.synchronize()
    _run_stats["target_calls"] = target_calls
    _run_stats["total_time"] = time.perf_counter() - _loop_start
    _run_stats["gen_tokens"] = max(full_seq.shape[-1] - context_init_len, 0)
    p_model._spec_profile["runs"].append(_run_stats)

    return full_seq
