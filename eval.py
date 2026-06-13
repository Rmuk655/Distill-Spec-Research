"""
eval.py — block-efficiency + alpha evaluation for a trained draft model.

Thin wrapper around speculative_decoding_loop() from main.py (GBV source of truth).
This file owns: dataset selection, multi-mode sweep, per-mode seed reset, CSV logging.
All speculative decoding logic lives in main.py — do not duplicate it here.

    block_efficiency  = total_generated_tokens / total_target_calls
    throughput        = total_generated_tokens / wall_time   (tok/s)
    avg_tree_nodes    = average tree size per target call

Dataset setup (run once before eval; default gsm8k_eval has 100 prompts):
    python -m data_io.download --datasets gsm8k --n 1000 --force

Usage:
    python eval.py --checkpoint checkpoints/kl_tree/ckpt_best  --mode gbv
    python eval.py --checkpoint checkpoints/gbv_tree/ckpt_best --mode gbv --K 3 --L 8
    python eval.py --checkpoint Qwen/Qwen3-0.6B --mode naive --dataset alpaca   # baseline
    python eval.py --checkpoint ckpts/best --mode gbv --modes naive,gbv,traversal  # sweep

    # parallel eval across modes (parallel-safe --output flag):
    python eval.py --checkpoint Qwen/Qwen3-0.6B --K 1 --n 1000 --modes naive     --output out_naive.csv &
    python eval.py --checkpoint Qwen/Qwen3-0.6B --K 1 --n 1000 --modes specinfer --output out_specinfer.csv &

Results print to stdout AND append one row to results.csv for later analysis.
"""
from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime

from tqdm import tqdm

import verifiers  # noqa: F401 — sys.path injection
from util    import load_prompts_jsonl, set_seed, load_models
from main    import speculative_decoding_loop

from data_io import get_path as dataset_path


# ═══════════════════════════════════════════════════════════════════════════
#  HARDCODED CONSTANTS — edit these to change defaults
# ═══════════════════════════════════════════════════════════════════════════
TEACHER_MODEL        = "Qwen/Qwen3-8B"
DEFAULT_K            = 3
DEFAULT_L            = 8
DEFAULT_MAX_NEW_TOKENS = 128
DEFAULT_TEMP         = 1.0
RESULTS_CSV          = os.path.join(os.path.dirname(__file__), "results.csv")
# ═══════════════════════════════════════════════════════════════════════════


VERIFIER_MODES = ["naive", "nss", "specinfer", "spectr", "khisti",
                  "bv", "gbv", "traversal"]


# ---------------------------------------------------------------------------
# Per-mode aggregate — calls speculative_decoding_loop (source of truth)
# ---------------------------------------------------------------------------

def evaluate_one_mode(p_model, q_model, tok, prompts, mode, K, L, max_new_tokens, temp):
    """Run the prompt set under one verifier mode and aggregate stats.

    Per-run stats (target_calls, gen_tokens, total_time, total_tree_nodes) are
    written by speculative_decoding_loop() into p_model._spec_profile["runs"].
    """
    p_model._spec_profile = {"runs": []}
    for prompt in tqdm(prompts, desc=f"mode={mode} K={K} L={L}", ncols=80):
        speculative_decoding_loop(
            p_model=p_model, q_model=q_model, tok=tok,
            prompt=prompt, verification_algo=mode,
            max_new_tokens=max_new_tokens, K=K, L=L,
            p_temp=temp, q_temp=temp,
        )

    runs        = p_model._spec_profile["runs"]
    total_calls = sum(r["target_calls"]     for r in runs)
    total_gen   = sum(r["gen_tokens"]       for r in runs)
    total_time  = sum(r["total_time"]       for r in runs)
    total_nodes = sum(r["total_tree_nodes"] for r in runs)

    return {
        "block_eff":      total_gen / total_calls   if total_calls > 0 else float("nan"),
        "throughput":     total_gen / total_time    if total_time  > 0 else float("nan"),
        "avg_tree_nodes": total_nodes / total_calls if total_calls > 0 else float("nan"),
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
    ap.add_argument("--device",    default="cuda",
                    help="CUDA device to use, e.g. cuda:1 (default: auto-select freest GPU)")
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

    print(f"[load] draft={args.checkpoint}")
    print(f"[load] teacher={TEACHER_MODEL}")
    tok, p_model, q_model = load_models(TEACHER_MODEL, args.checkpoint,
                                        device=args.device, dtype="bf16")

    data_path = dataset_path(args.dataset)
    prompts   = load_prompts_jsonl(data_path)[:args.n]
    print(f"[data] {data_path} — {len(prompts)} prompts")

    modes = [m.strip() for m in args.modes.split(",")] if args.modes else [args.mode]

    print()
    print("=" * 78)
    print(f"  Eval  draft={args.checkpoint}  dataset={args.dataset}  "
          f"K={args.K} L={args.L} n={len(prompts)}")
    print("=" * 78)

    for mode in modes:
        set_seed(args.seed)   # reset before every mode so RNG state is identical
        stats = evaluate_one_mode(p_model, q_model, tok, prompts, mode,
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
