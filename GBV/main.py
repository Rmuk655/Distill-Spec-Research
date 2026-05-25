import random
import json
import os
import sys
import argparse

# Force UTF-8 stdout/stderr so Unicode characters in print() never crash on
# Windows cp1252 terminals (or any other non-UTF-8 default encoding).
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass  # Python < 3.7 or non-text stream

# Reduce CUDA allocator fragmentation — important on small GPUs (T4, etc.).
# Set automatically; override with PYTORCH_CUDA_ALLOC_CONF=<custom> in environment.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

import torch
import time
from tqdm import tqdm
import torch.nn.functional as F
from typing import List, Tuple, Dict, Mapping, Callable, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

from util import *
from inference_util import *
from node import *
from verifier import *


"""
Runs the full speculative decoding algorithm on a prompt.
Takes in:
    - p_model: target model, from which we aim to sample
    - q_model: draft model, from which we form a draft tree
    - tok: tokenizer, must be shared by target and draft model
    - prompt: input prompt as a string
    - verification_algo: which verification algorithm to run. Can choose from the following:
        (a) OTLP-based methods -> 'naive', 'nss', 'specinfer', 'spectr'
        (b) Other methods -> 'bv', 'gbv', 'traversal'
    - eos_token_id: when to stop generation
    - max_new_tokens: how many tokens to generate if no EOS is reached
    - K: number of i.i.d. draft paths
    - L: draft block length
    - p_temp: target sampling temperature
    - q_temp: draft sampling temperature
"""
def speculative_decoding_loop(
    p_model: AutoModelForCausalLM,
    q_model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prompt: str,
    verification_algo: str,
    eos_token_id : int = None, max_new_tokens: int = 128, K: int = 4, L: int = 8, p_temp: float = 1.0, q_temp: float = 1.0,
):
    # Profiling init of per-run stats and timer
    if not hasattr(p_model, "_spec_profile"):
        p_model._spec_profile = {"runs": []}
    _run_stats = {
        "time_draft": 0.0,
        "time_target": 0.0,
        "total_tree_nodes": 0,
        "kv_target_peak_bytes": 0,
        "kv_draft_peak_bytes": 0,
    }
    p_model._spec_run_stats = _run_stats
    torch.cuda.synchronize()
    _loop_start = time.perf_counter()

    if eos_token_id is None:
        eos_token_id = p_model.config.eos_token_id

    # Build cached context from prompt.
    context_cached = torch.tensor(tok.encode(prompt), device=p_model.device)
    context_cached = context_cached.unsqueeze(0)
    context_init_len = context_cached.shape[-1]

    # During the prefill phase, intialize target and draft KV caches.
    p_out = p_model(context_cached, use_cache=True, return_dict=True)
    q_out = q_model(context_cached, use_cache=True, return_dict=True)
    p_cache = p_out.past_key_values
    q_cache = q_out.past_key_values

    # Sample first pending token from the last target distribution.
    p_probs_last = F.softmax(p_out.logits[:, -1, :] / p_temp, dim=-1)
    context_pending = torch.multinomial(p_probs_last, num_samples=1)

    # Main speculative loop until EOS or max generated length is reached.
    target_calls = 0
    while context_cached.shape[-1] + 1 < context_init_len + max_new_tokens:
        p_cache, q_cache, context_cached, context_pending = speculative_decoding_iter(
            p_model,
            q_model,
            p_cache,
            q_cache,
            context_cached,
            context_pending,
            verification_algo,
            K=K, L=L, p_temp=p_temp, q_temp=q_temp,
        )
        full_seq = torch.cat([context_cached, context_pending], dim=-1)
        target_calls += 1
        if (full_seq == eos_token_id).any():
            break

    # Profiling finalization of stats
    torch.cuda.synchronize()
    _run_stats["target_calls"] = target_calls
    _run_stats["total_time"] = time.perf_counter() - _loop_start
    _run_stats["gen_tokens"] = max(full_seq.shape[-1] - context_init_len, 0)
    p_model._spec_profile["runs"].append(_run_stats)
            
    return full_seq



"""
Performs a single speculative decoding iteration, with time profiling of three steps.
(1) Construct the draft tree (nodes are contexts, root node is "context_pending") from i.i.d. draft model sampling
(2) Perform target model forward pass on the draft tree
(3) Using data from (1) + (2), run a verification algo to select a length (tau + 1) draft tree prefix and a residual token
(4) The model caches and contexts are updated accordingly, to append the selected prefix and residual token
Takes in:
    - p_model: target model, from which we aim to sample
    - q_model: draft model, from which we form a draft tree
    - p_cache: target KV-cache, each KV tensor has shape (1, num_heads, context_len - 1, head_dim)
    - q_cache: draft KV-cache, each KV tensor has shape (1, num_heads, context_len - 1, head_dim)
    - context_cached: shape (1, context_len - 1), token IDs corresponding to p_cache/q_cache rows
    - context_pending: shape (1, 1), last token ID, which currently has not been committed to p_cache/q_cache
    - verification_algo: which verifier to run for (3)
    - K: number of i.i.d. draft paths
    - L: draft block length
    - p_temp: target sampling temperature
    - q_temp: draft sampling temperature
Returns:
    - p_cache: updated target KV-cache, each KV tensor now has shape (1, num_heads, context_len + tau, head_dim)
    - q_cache: updated draft KV-cache, each KV tensor now has shape (1, num_heads, context_len + tau, head_dim)
    - context_cached: updated to shape (1, context_len + tau)
    - context_pending: still shape (1, 1), now the residual token
"""
def speculative_decoding_iter(
    p_model: AutoModelForCausalLM,
    q_model: AutoModelForCausalLM,
    p_cache: DynamicCache,
    q_cache: DynamicCache,
    context_cached: torch.Tensor,
    context_pending: torch.Tensor,
    verification_algo: str,
    K: int = 4, L: int = 8, p_temp: float = 1.0, q_temp: float = 1.0, 
):
    # Start profiling
    _run_stats = getattr(p_model, "_spec_run_stats", None)

    # (1) Draft tree construction with profiling
    if _run_stats is not None:
        torch.cuda.synchronize()
        _t_draft = time.perf_counter()
    q_paths, q_cache, q_probs_dict = iid_draft(q_model, q_cache, context_pending, K=K, L=L, q_temp=q_temp)
    if _run_stats is not None:
        torch.cuda.synchronize()
        _run_stats["time_draft"] += time.perf_counter() - _t_draft

    # (2) Batched target forward pass with profiling
    if _run_stats is not None:
        torch.cuda.synchronize()
        _t_target = time.perf_counter()
    q_prefixes, q_tokens, p_cache, p_probs_dict = target_tree_pass(p_model, p_cache, q_paths, K=K, L=L, p_temp=p_temp)
    if _run_stats is not None:
        torch.cuda.synchronize()
        _run_stats["time_target"] += time.perf_counter() - _t_target
        _tree_size = len(q_prefixes)
        _run_stats["total_tree_nodes"] += _tree_size

    # (3) Verification to select one node on one draft path, plus a residual token, and then update caches and contexts
    tree_verifier = TreeVerifier(q_paths, q_prefixes, q_probs_dict, p_probs_dict)
    ver_node, res_token = tree_verifier.verify(verification_algo)
    tau = ver_node.depth
    ver_prefix = ver_node.rep
    path_idx = min(i for i, path in enumerate(q_paths) if ver_prefix == ",".join(str(x) for x in path[:tau + 1]))

    # (4) Update caches and contexts accordingly for the next iteration of decoding.
    cached_len = context_cached.shape[-1]
    prefix_slice = [i for i, q_prefix in enumerate(q_prefixes) if ver_prefix.startswith(q_prefix + ",") or ver_prefix == q_prefix]
    p_cache = slice_cache(p_cache, [0], list(range(cached_len)) + [(cached_len + x) for x in prefix_slice])
    q_cache = slice_cache(q_cache, [path_idx], list(range(cached_len + tau + 1)))
    context_cached = torch.cat([context_cached, q_tokens[:, prefix_slice]], dim=-1)
    context_pending[0, 0] = res_token

    # Profile KV cache sizes
    if _run_stats is not None:
        _kv_target = 0
        for _layer in p_cache:
            _k = _layer[0]
            _v = _layer[1]
            _bytes = _k.element_size()
            _kv_target += (_k.numel() + _v.numel()) * _bytes
        if _kv_target > _run_stats["kv_target_peak_bytes"]:
            _run_stats["kv_target_peak_bytes"] = _kv_target

        _kv_draft = 0
        for _layer in q_cache:
            _k = _layer[0]
            _v = _layer[1]
            _bytes = _k.element_size()
            _kv_draft += (_k.numel() + _v.numel()) * _bytes
        if _kv_draft > _run_stats["kv_draft_peak_bytes"]:
            _run_stats["kv_draft_peak_bytes"] = _kv_draft

    return p_cache, q_cache, context_cached, context_pending


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p_model", type=str, default="Qwen/Qwen3-32B", help="Target model (must share tokenizer with draft)")
    ap.add_argument("--q_model", type=str, default="Qwen/Qwen3-0.6B", help="Draft model (must share tokenizer with target)")
    ap.add_argument("--data", type=str, default="data/humaneval.jsonl", help="Path to JSONL with {'prompt': ...} per line")
    ap.add_argument( "--L", type=int, default=8, help="Speculative window length")
    ap.add_argument("--K", type=int, default=3, help="Number of draft paths")
    ap.add_argument("--max_new_tokens", type=int, default=128, help="Maximum number of new tokens to generate")
    ap.add_argument("--mode", type=str, default="gbv", choices=["naive", "nss", "specinfer", "spectr", "bv", "gbv", "traversal"], help="Verification algorithm to use")
    ap.add_argument("--p_temp", type=float, default=1.0, help="Target model sampling temperature")
    ap.add_argument("--q_temp", type=float, default=1.0, help="Draft model sampling temperature")
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--dtype", type=str, default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    ap.add_argument("--seed", type=int, default=123)
    # Batch args — comma-separated lists that override the single-value args above.
    # When provided, models are loaded once and all combinations are run in sequence.
    ap.add_argument("--modes",   type=str, default=None,
                    help="Comma-separated verification algorithms, e.g. 'gbv,specinfer,traversal'. "
                         "Overrides --mode when provided.")
    ap.add_argument("--Ks",     type=str, default=None,
                    help="Comma-separated K values, e.g. '1,3,5'. Overrides --K when provided.")
    ap.add_argument("--p_temps", type=str, default=None,
                    help="Comma-separated target temperatures, e.g. '0.6,1.0'. "
                         "Overrides --p_temp when provided.")
    ap.add_argument("--compile", action="store_true",
                    help="Apply torch.compile(mode='reduce-overhead', dynamic=True) to the "
                         "draft model to cut Python→CUDA dispatch overhead.  Draft-only: "
                         "target uses a custom tree-attention mask incompatible with compile. "
                         "First 1-2 prompts are slower (warm-up); subsequent ones are faster.")
    ap.add_argument("--load_in_4bit", action="store_true",
                    help="Load the TARGET (p_model) in 4-bit NF4 via bitsandbytes. "
                         "Use on Colab free T4 (15 GB VRAM) with Qwen3-8B: "
                         "bfloat16 is ~16 GB (OOM), 4-bit NF4 is ~5 GB (fits). "
                         "Requires: pip install bitsandbytes. "
                         "The draft (q_model) is always loaded in the requested --dtype.")
    args = ap.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)

    # Resolve batch vs single-value args.
    # --modes / --Ks / --p_temps (comma-separated) override --mode / --K / --p_temp.
    modes_list  = [m.strip() for m in args.modes.split(",")]    if args.modes   else [args.mode]
    Ks_list     = [int(k)    for k in args.Ks.split(",")]       if args.Ks      else [args.K]
    temps_list  = [float(t)  for t in args.p_temps.split(",")]  if args.p_temps else [args.p_temp]
    combos      = [(mode, K, T) for mode in modes_list for K in Ks_list for T in temps_list]

    # Load models once — shared across all combos.
    tok, p_model, q_model = load_models(
        args.p_model, args.q_model, device=args.device, dtype=args.dtype,
        compile_draft=args.compile,
        load_in_4bit=getattr(args, "load_in_4bit", False),
    )

    # Load prompts.
    prompts = load_prompts_jsonl(args.data)
    print(f"Loaded {len(prompts)} prompt(s) | {len(combos)} combo(s) | "
          f"p={args.p_model}, q={args.q_model}, L={args.L}, q_temp={args.q_temp}")

    # Run every (mode, K, T) combo — models stay loaded throughout.
    for (mode, K, p_temp) in combos:
        p_model._spec_profile = {"runs": []}   # reset per-combo profiler

        for idx, prompt in tqdm(enumerate(prompts), total=len(prompts),
                                desc=f"mode={mode} K={K} T={p_temp}"):
            _ = speculative_decoding_loop(
                p_model=p_model, q_model=q_model, tok=tok,
                prompt=prompt, verification_algo=mode,
                eos_token_id=None, max_new_tokens=args.max_new_tokens,
                K=K, L=args.L, p_temp=p_temp, q_temp=args.q_temp,
            )

        # Collect aggregate profiling metrics for this combo.
        runs             = p_model._spec_profile["runs"]
        total_calls      = sum(r["target_calls"]      for r in runs)
        total_gen_tokens = sum(r["gen_tokens"]         for r in runs)
        total_time       = sum(r["total_time"]         for r in runs)
        time_draft       = sum(r["time_draft"]         for r in runs)
        time_target      = sum(r["time_target"]        for r in runs)
        total_tree_nodes = sum(r["total_tree_nodes"]   for r in runs)
        kv_target_peak   = max((r["kv_target_peak_bytes"] for r in runs), default=0)
        kv_draft_peak    = max((r["kv_draft_peak_bytes"]  for r in runs), default=0)

        block_eff      = total_gen_tokens / total_calls if total_calls > 0 else float("nan")
        tokens_per_sec = total_gen_tokens / total_time  if total_time  > 0 else float("nan")
        ms_per_token   = 1000.0 / tokens_per_sec        if tokens_per_sec > 0 else float("nan")
        draft_pct      = 100.0 * time_draft  / total_time if total_time > 0 else float("nan")
        target_pct     = 100.0 * time_target / total_time if total_time > 0 else float("nan")
        avg_tree_nodes = total_tree_nodes / total_calls    if total_calls > 0 else float("nan")

        # Machine-parseable tagged line (run_all.py uses this exact format).
        print(f"Block efficiency (mode={mode}, K={K}, T={p_temp}): {block_eff:.6f}")
        print(f"Throughput (tokens / second): {tokens_per_sec:.6f}")
        print(f"Walltime (ms / token): {ms_per_token:.6f}")
        print(f"Time breakdown draft/target (%): {draft_pct:.1f} / {target_pct:.1f}")
        print(f"Avg tree nodes per target call: {avg_tree_nodes:.2f}")
        print(f"Peak KV target/draft (MB): "
              f"{kv_target_peak / 1024**2:.2f} / {kv_draft_peak / 1024**2:.2f}")
