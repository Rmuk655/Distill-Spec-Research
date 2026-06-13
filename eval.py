"""
eval.py — block-efficiency + alpha evaluation for a trained draft model.

Loads the draft checkpoint + the Qwen3-8B teacher, runs speculative decoding
on a held-out prompt set, and reports:

    block_efficiency  = total_generated_tokens / total_target_calls
    throughput        = total_generated_tokens / wall_time   (tok/s)
    avg_tree_nodes    = average tree size per target call

Usage:
    python eval.py --checkpoint checkpoints/kl_tree/ckpt_best  --mode gbv
    python eval.py --checkpoint checkpoints/gbv_tree/ckpt_best --mode gbv --K 3 --L 8
    python eval.py --checkpoint Qwen/Qwen3-0.6B --mode naive --dataset alpaca   # baseline
    python eval.py --checkpoint ckpts/best --mode gbv --modes naive,gbv,traversal  # sweep

Results print to stdout AND append one row to results.csv for later analysis.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime

import torch
from tqdm import tqdm

import verifiers  # noqa: F401 — sys.path injection
from util             import load_prompts_jsonl, set_seed
from inference_util   import iid_draft, target_tree_pass
from verifier         import TreeVerifier
from transformers     import AutoTokenizer, AutoModelForCausalLM

from data_io import get_path as dataset_path


# ═══════════════════════════════════════════════════════════════════════════
#  HARDCODED CONSTANTS — edit these to change defaults
# ═══════════════════════════════════════════════════════════════════════════
TEACHER_MODEL  = "Qwen/Qwen3-8B"
DEFAULT_K      = 3
DEFAULT_L      = 8
DEFAULT_MAX_NEW_TOKENS = 128
DEFAULT_TEMP   = 1.0
RESULTS_CSV    = os.path.join(os.path.dirname(__file__), "results.csv")
# ═══════════════════════════════════════════════════════════════════════════


VERIFIER_MODES = ["naive", "nss", "specinfer", "spectr", "khisti",
                  "bv", "gbv", "traversal"]


# ---------------------------------------------------------------------------
# Speculative decoding loop — same as /GBV/main.py, inlined here so eval.py
# is self-contained and a new-grad reader can follow the whole pipeline.
# ---------------------------------------------------------------------------

@torch.no_grad()
def speculative_decode_one(p_model, q_model, tokenizer, prompt: str, mode: str,
                            K: int, L: int, max_new_tokens: int, temp: float):
    """
    Run one prompt through speculative decoding under verifier `mode`.
    Returns a dict of per-prompt stats (target_calls, gen_tokens, total_time,
    tree_nodes).

    @torch.no_grad(): eval never backpropagates.  Without this, draft model
    softmax outputs carry requires_grad=True into TreeVerifier / node.py, which
    triggers a UserWarning when node.py converts q[token] to a Python float
    (float() on a grad-tracked tensor).  no_grad() also removes unnecessary
    autograd overhead during inference.
    """
    import torch.nn.functional as F

    device = p_model.device
    # Prefill both models' KV caches on the prompt.
    ctx = torch.tensor(tokenizer.encode(prompt), device=device, dtype=torch.long).unsqueeze(0)
    init_len = ctx.shape[-1]
    p_out = p_model(ctx, use_cache=True, return_dict=True)
    q_out = q_model(ctx, use_cache=True, return_dict=True)
    p_cache, q_cache = p_out.past_key_values, q_out.past_key_values

    # First pending token from teacher's last position.
    pending = torch.multinomial(F.softmax(p_out.logits[:, -1, :] / temp, dim=-1),
                                num_samples=1)

    target_calls   = 0
    total_tree_nodes = 0
    t0 = time.perf_counter()
    while ctx.shape[-1] + 1 < init_len + max_new_tokens:
        # 1. Build student's K×L draft tree (no grad — eval is frozen).
        q_paths, q_cache, q_probs_dict = iid_draft(
            q_model, q_cache, pending, K=K, L=L, q_temp=temp,
        )
        # 2. Score every tree node under the teacher.
        q_prefixes, q_tokens, p_cache, p_probs_dict = target_tree_pass(
            p_model, p_cache, q_paths, K=K, L=L, p_temp=temp,
        )
        total_tree_nodes += len(q_prefixes)

        # 3. Verifier decides how many tokens to accept (τ) and which residual to emit.
        tv = TreeVerifier(q_paths, q_prefixes, q_probs_dict, p_probs_dict)
        ver_node, res_token = tv.verify(mode)
        tau        = ver_node.depth
        ver_prefix = ver_node.rep
        path_idx   = min(i for i, path in enumerate(q_paths)
                         if ver_prefix == ",".join(str(x) for x in path[:tau + 1]))

        # 4. Splice both KV caches to keep only accepted-prefix rows.
        from util import slice_cache
        cached_len = ctx.shape[-1]
        prefix_slice = [i for i, pfx in enumerate(q_prefixes)
                        if ver_prefix.startswith(pfx + ",") or ver_prefix == pfx]
        p_cache = slice_cache(p_cache, [0],
                              list(range(cached_len)) + [cached_len + x for x in prefix_slice])
        q_cache = slice_cache(q_cache, [path_idx], list(range(cached_len + tau + 1)))
        ctx     = torch.cat([ctx, q_tokens[:, prefix_slice]], dim=-1)
        pending[0, 0] = res_token

        full = torch.cat([ctx, pending], dim=-1)
        target_calls += 1
        if (full == p_model.config.eos_token_id).any():
            break

    total_time = time.perf_counter() - t0
    gen_tokens = max(full.shape[-1] - init_len, 0)
    # Decode the generated portion (everything after the prompt) so callers
    # like inference.py can print it.  eval.py just ignores this field.
    gen_text = tokenizer.decode(full[0, init_len:].tolist(), skip_special_tokens=True)
    return {
        "target_calls":     target_calls,
        "gen_tokens":       gen_tokens,
        "total_time":       total_time,
        "total_tree_nodes": total_tree_nodes,
        "generated_text":   gen_text,
    }


# ---------------------------------------------------------------------------
# Per-mode aggregate over a prompt set
# ---------------------------------------------------------------------------

def evaluate_one_mode(p_model, q_model, tokenizer, prompts, mode,
                     K, L, max_new_tokens, temp):
    """Run the prompt set under one verifier mode and aggregate stats."""
    runs = []
    for prompt in tqdm(prompts, desc=f"mode={mode} K={K} L={L}", ncols=80):
        runs.append(speculative_decode_one(
            p_model, q_model, tokenizer, prompt, mode,
            K=K, L=L, max_new_tokens=max_new_tokens, temp=temp,
        ))

    total_calls = sum(r["target_calls"]    for r in runs)
    total_gen   = sum(r["gen_tokens"]       for r in runs)
    total_time  = sum(r["total_time"]       for r in runs)
    total_nodes = sum(r["total_tree_nodes"] for r in runs)

    block_eff      = total_gen / total_calls   if total_calls   > 0 else float("nan")
    throughput     = total_gen / total_time    if total_time    > 0 else float("nan")
    avg_tree_nodes = total_nodes / total_calls if total_calls   > 0 else float("nan")

    return {
        "block_eff":      block_eff,
        "throughput":     throughput,
        "avg_tree_nodes": avg_tree_nodes,
        "n_prompts":      len(prompts),
        "total_gen":      total_gen,
        "total_time":     total_time,
    }


# ---------------------------------------------------------------------------
# CSV append (one row per eval cell — easy to grep / pandas later)
# ---------------------------------------------------------------------------

CSV_COLUMNS = ["timestamp", "checkpoint", "dataset", "mode", "K", "L",
               "n_prompts", "block_eff", "throughput_tok_s", "avg_tree_nodes",
               "total_gen_tokens", "total_time_s"]


def append_csv_row(row: dict):
    """Append one row to results.csv, creating the file (with header) if needed."""
    new_file = not os.path.isfile(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if new_file:
            w.writeheader()
        w.writerow(row)


# ---------------------------------------------------------------------------
# Argparse + main
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Block-efficiency eval (A100, Qwen3).")
    ap.add_argument("--checkpoint", required=True,
                    help="Path to draft checkpoint dir (or HF model id for baseline).")
    ap.add_argument("--mode",      default="gbv",   choices=VERIFIER_MODES,
                    help="Verifier mode (default gbv).  Override via --modes for a sweep.")
    ap.add_argument("--modes",     default=None,
                    help="Comma-separated verifier modes for a sweep, e.g. 'naive,gbv,traversal'.")
    ap.add_argument("--dataset",   default="gsm8k_eval",
                    help="Dataset name (gsm8k_eval, alpaca, math500, humaneval, mtbench).")
    ap.add_argument("--K",         type=int, default=DEFAULT_K)
    ap.add_argument("--L",         type=int, default=DEFAULT_L)
    ap.add_argument("--n",         type=int, default=100,
                    help="How many prompts from the dataset to evaluate.")
    ap.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    ap.add_argument("--temp",      type=float, default=DEFAULT_TEMP)
    ap.add_argument("--seed",      type=int, default=123)
    ap.add_argument("--output",    default=None,
                    help="Override output CSV path (default: results.csv next to eval.py). "
                         "Set to a unique path when running multiple parallel processes.")
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    if args.output:
        global RESULTS_CSV
        RESULTS_CSV = args.output

    # Load models — teacher always Qwen3-8B; draft is the checkpoint.
    print(f"[load] draft={args.checkpoint}")
    print(f"[load] teacher={TEACHER_MODEL}")
    # Tokenizer: always load from TEACHER_MODEL, not the checkpoint dir.
    # save_checkpoint() only saves model weights (model.save_pretrained), NOT tokenizer
    # files (tokenizer.json etc.).  Loading AutoTokenizer from a path without tokenizer
    # files can silently fall back to a broken tokenizer whose .encode() returns floats,
    # causing a RuntimeError in the embedding layer.  Draft + teacher share a vocab
    # (required for speculative decoding), so TEACHER_MODEL is always the right source.
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    q_model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to("cuda").eval()
    p_model = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to("cuda").eval()

    # Prompts
    data_path = dataset_path(args.dataset)
    prompts = load_prompts_jsonl(data_path)[:args.n]
    print(f"[data] {data_path} — {len(prompts)} prompts")

    # Modes to sweep
    modes = [m.strip() for m in args.modes.split(",")] if args.modes else [args.mode]

    print()
    print("=" * 78)
    print(f"  Eval  draft={args.checkpoint}  dataset={args.dataset}  "
          f"K={args.K} L={args.L} n={len(prompts)}")
    print("=" * 78)

    for mode in modes:
        set_seed(args.seed)   # reset before every mode so RNG state is identical
        stats = evaluate_one_mode(p_model, q_model, tokenizer, prompts, mode,
                                  K=args.K, L=args.L,
                                  max_new_tokens=args.max_new_tokens, temp=args.temp)
        print(f"\n  mode={mode:11s}  BE={stats['block_eff']:.4f}  "
              f"throughput={stats['throughput']:.1f} tok/s  "
              f"avg_tree_nodes={stats['avg_tree_nodes']:.1f}")

        append_csv_row({
            "timestamp":        datetime.utcnow().isoformat(timespec="seconds"),
            "checkpoint":       args.checkpoint,
            "dataset":          args.dataset,
            "mode":             mode,
            "K":                args.K,
            "L":                args.L,
            "n_prompts":        stats["n_prompts"],
            "block_eff":        f"{stats['block_eff']:.6f}",
            "throughput_tok_s": f"{stats['throughput']:.4f}",
            "avg_tree_nodes":   f"{stats['avg_tree_nodes']:.4f}",
            "total_gen_tokens": stats["total_gen"],
            "total_time_s":     f"{stats['total_time']:.2f}",
        })

    print(f"\n[done] results appended to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
