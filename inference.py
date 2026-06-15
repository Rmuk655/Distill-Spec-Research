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

import verifiers  # noqa: F401 — sys.path injection
from util   import set_seed, load_models
from main   import speculative_decoding_loop
from config import TEACHER_MODEL, VERIFIER_MODES, DEFAULT_K, DEFAULT_L, DEFAULT_MAX_NEW_TOKENS, DEFAULT_TEMP, DEFAULT_DTYPE


def parse_args():
    ap = argparse.ArgumentParser(description="Single-prompt speculative decoding.")
    ap.add_argument("--checkpoint", required=True,
                    help="Draft model checkpoint dir (or HF id for baseline).")
    ap.add_argument("--mode",       default="gbv", choices=VERIFIER_MODES)
    ap.add_argument("--prompt",     required=True, help="The user prompt as a string.")
    ap.add_argument("--K",          type=int,   default=DEFAULT_K)
    ap.add_argument("--L",          type=int,   default=DEFAULT_L)
    ap.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    ap.add_argument("--temp",       type=float, default=DEFAULT_TEMP)
    ap.add_argument("--seed",       type=int,   default=123)
    ap.add_argument("--device",     default="cuda",
                    help="CUDA device to use, e.g. cuda:1 (default: auto-select freest GPU)")
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    print(f"[load] draft={args.checkpoint}")
    print(f"[load] teacher={TEACHER_MODEL}")
    tok, p_model, q_model = load_models(TEACHER_MODEL, args.checkpoint,
                                        device=args.device, dtype=DEFAULT_DTYPE)

    print(f"\n[inference] prompt = {args.prompt!r}")
    print(f"[inference] mode={args.mode}  K={args.K}  L={args.L}  "
          f"max_new_tokens={args.max_new_tokens}\n")

    p_model._spec_profile = {"runs": []}
    t0 = time.time()
    full_seq = speculative_decoding_loop(
        p_model=p_model, q_model=q_model, tok=tok,
        prompt=args.prompt, verification_algo=args.mode,
        max_new_tokens=args.max_new_tokens, K=args.K, L=args.L,
        p_temp=args.temp, q_temp=args.temp,
    )
    elapsed = time.time() - t0

    stats     = p_model._spec_run_stats
    init_len  = len(tok.encode(args.prompt))
    gen_text  = tok.decode(full_seq[0, init_len:].tolist(), skip_special_tokens=True)
    block_eff = stats["gen_tokens"] / stats["target_calls"] \
                if stats["target_calls"] > 0 else float("nan")

    print()
    print("─" * 70)
    print("GENERATED TEXT:")
    print(gen_text)
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
