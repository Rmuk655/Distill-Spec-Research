"""
inference.py — run speculative decoding on ONE prompt, print the generated text.

Use this to manually sanity-check a checkpoint or to demo the pipeline.
eval.py does the same thing across a whole prompt set and computes BE — this
script is the single-prompt cousin meant for human inspection.

Usage:
    python inference.py --checkpoint checkpoints/kl_tree/ckpt_best \
                        --mode gbv \
                        --prompt "What is 17 * 23?"

    # With explicit K, L and a longer generation
    python inference.py --checkpoint Qwen/Qwen3-0.6B --mode naive --K 3 --L 8 \
                        --max_new_tokens 256 \
                        --prompt "Write a haiku about block verification."

Prints the decoded text plus per-call stats (target calls, tree nodes, BE).
"""
from __future__ import annotations

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import verifiers  # noqa: F401 — sys.path injection
from util             import set_seed
from eval             import speculative_decode_one    # we reuse the loop from eval.py


TEACHER_MODEL = "Qwen/Qwen3-8B"


def parse_args():
    ap = argparse.ArgumentParser(description="Single-prompt speculative decoding.")
    ap.add_argument("--checkpoint", required=True,
                    help="Draft model checkpoint dir (or HF id for baseline).")
    ap.add_argument("--mode",       default="gbv",
                    choices=["naive", "nss", "specinfer", "spectr", "khisti",
                             "bv", "gbv", "traversal"])
    ap.add_argument("--prompt",     required=True, help="The user prompt as a string.")
    ap.add_argument("--K",          type=int,   default=3)
    ap.add_argument("--L",          type=int,   default=8)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temp",       type=float, default=1.0)
    ap.add_argument("--seed",       type=int,   default=123)
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    print(f"[load] draft={args.checkpoint}")
    print(f"[load] teacher={TEACHER_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    q_model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to("cuda").eval()
    p_model = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to("cuda").eval()

    print(f"\n[inference] prompt = {args.prompt!r}")
    print(f"[inference] mode={args.mode}  K={args.K}  L={args.L}  "
          f"max_new_tokens={args.max_new_tokens}\n")

    # speculative_decode_one (imported from eval.py) does the full speculative loop
    # but only returns aggregate stats — we re-tokenise the prompt here so we can
    # also print the generated text.  In a future cleanup, refactor to return both.
    t0 = time.time()
    stats = speculative_decode_one(
        p_model, q_model, tokenizer, args.prompt, args.mode,
        K=args.K, L=args.L,
        max_new_tokens=args.max_new_tokens, temp=args.temp,
    )
    elapsed = time.time() - t0

    block_eff = stats["gen_tokens"] / stats["target_calls"] \
                if stats["target_calls"] > 0 else float("nan")

    print()
    print("─" * 70)
    print("GENERATED TEXT:")
    print(stats.get("generated_text", "<no text>"))
    print("─" * 70)
    print()
    print("=" * 70)
    print(f"  generated tokens   : {stats['gen_tokens']}")
    print(f"  target model calls : {stats['target_calls']}")
    print(f"  total tree nodes   : {stats['total_tree_nodes']}")
    print(f"  block efficiency   : {block_eff:.3f}")
    print(f"  wall time          : {elapsed:.2f} s  "
          f"({stats['gen_tokens']/elapsed:.1f} tok/s)")
    print("=" * 70)


if __name__ == "__main__":
    main()
