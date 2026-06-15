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
import json
import os
from datetime import datetime

from tqdm import tqdm

import verifiers  # noqa: F401 — sys.path injection
from util    import load_prompts_jsonl, set_seed, load_models
from main    import speculative_decoding_loop

from data_io import get_path as dataset_path
from config  import (TEACHER_MODEL, DEFAULT_K, DEFAULT_L, DEFAULT_MAX_NEW_TOKENS,
                     DEFAULT_TEMP, DEFAULT_DTYPE, DEFAULT_SEED, VERIFIER_MODES, block_eff)

RESULTS_CSV = os.path.join(os.path.dirname(__file__), "results.csv")


# ---------------------------------------------------------------------------
# Resume state helpers — one JSONL per (mode, K, L), one line per prompt
# ---------------------------------------------------------------------------

def _state_path(csv_path: str, mode: str, K: int, L: int, checkpoint: str = "") -> str:
    logs_dir = os.path.join(os.path.dirname(os.path.abspath(csv_path)), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    if checkpoint:
        parts = checkpoint.replace("\\", "/").rstrip("/").split("/")
        ckpt_tag = "_".join(parts[-2:]) if len(parts) >= 2 else parts[-1]
        filename = f"{ckpt_tag}.{mode}_K{K}_L{L}.state.jsonl"
    else:
        filename = f"{mode}_K{K}_L{L}.state.jsonl"
    return os.path.join(logs_dir, filename)


def _load_state(path: str) -> dict[int, dict]:
    """Return {prompt_idx: run_dict} from an existing state file, or {}."""
    done: dict[int, dict] = {}
    if not os.path.isfile(path):
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                done[r["prompt_idx"]] = r
    return done


# ---------------------------------------------------------------------------
# Per-mode aggregate — calls speculative_decoding_loop (source of truth)
# ---------------------------------------------------------------------------

def evaluate_one_mode(p_model, q_model, tok, prompts, mode, K, L,
                      max_new_tokens, temp, state_path: str | None = None):
    """Run the prompt set under one verifier mode and aggregate stats.

    Resumes from a prior partial run if state_path exists — already-completed
    prompts are skipped and their stats are loaded from disk so block_eff is
    computed correctly over the full n_prompts denominator.
    """
    done = _load_state(state_path) if state_path else {}
    if done:
        print(f"  [resume] {len(done)}/{len(prompts)} prompts already done "
              f"— skipping, loading from {state_path}")

    state_f = open(state_path, "a", encoding="utf-8") if state_path else None
    all_runs: list[dict] = []

    try:
        for i, prompt in enumerate(tqdm(prompts, desc=f"mode={mode} K={K} L={L}", ncols=80)):
            if i in done:
                all_runs.append(done[i])
                continue

            p_model._spec_profile = {"runs": []}
            speculative_decoding_loop(
                p_model=p_model, q_model=q_model, tok=tok,
                prompt=prompt, verification_algo=mode,
                max_new_tokens=max_new_tokens, K=K, L=L,
                p_temp=temp, q_temp=temp,
            )
            run = p_model._spec_profile["runs"][0]
            run["prompt_idx"] = i
            all_runs.append(run)

            if state_f:
                state_f.write(json.dumps(run) + "\n")
                state_f.flush()
    finally:
        if state_f:
            state_f.close()

    total_calls  = sum(r["target_calls"]     for r in all_runs)
    total_gen    = sum(r["gen_tokens"]       for r in all_runs)
    total_time   = sum(r["total_time"]       for r in all_runs)
    total_nodes  = sum(r["total_tree_nodes"] for r in all_runs)
    time_draft   = sum(r.get("time_draft",   0.0) for r in all_runs)
    time_target  = sum(r.get("time_target",  0.0) for r in all_runs)
    time_verify  = sum(r.get("time_verify",  0.0) for r in all_runs)
    time_cache   = sum(r.get("time_cache",   0.0) for r in all_runs)

    # Keys match CSV_COLUMNS (minus the metadata fields added by log_result).
    return {
        "n_prompts":         len(all_runs),
        "target_calls":      total_calls,
        "block_eff":         block_eff(total_gen, total_calls),
        "throughput_tok_s":  total_gen / total_time    if total_time  > 0 else float("nan"),
        "time_per_call_ms":  1000.0 * total_time / total_calls  if total_calls > 0 else float("nan"),
        "time_per_token_ms": 1000.0 * total_time / total_gen    if total_gen   > 0 else float("nan"),
        "avg_tree_nodes":    total_nodes / total_calls if total_calls > 0 else float("nan"),
        "total_gen_tokens":  total_gen,
        "total_time_s":      total_time,
        "time_draft_s":      time_draft,
        "time_target_s":     time_target,
        "time_verify_s":     time_verify,
        "time_cache_s":      time_cache,
    }


# ---------------------------------------------------------------------------
# CSV append (one row per eval cell — easy to grep / pandas later)
# ---------------------------------------------------------------------------

CSV_COLUMNS = ["timestamp", "checkpoint", "dataset", "mode", "K", "L",
               "n_prompts", "target_calls", "block_eff", "throughput_tok_s",
               "time_per_call_ms", "time_per_token_ms",
               "avg_tree_nodes", "total_gen_tokens", "total_time_s",
               "time_draft_s", "time_target_s", "time_verify_s", "time_cache_s"]


def append_csv_row(row: dict):
    """Append one row to results.csv, creating the file (with header) if needed."""
    new_file = not os.path.isfile(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if new_file:
            w.writeheader()
        w.writerow(row)


# Format specs for float stats columns; integer columns pass through as-is.
_FLOAT_FMT = {
    "block_eff": ".6f", "throughput_tok_s": ".4f",
    "time_per_call_ms": ".3f", "time_per_token_ms": ".3f",
    "avg_tree_nodes": ".4f", "total_time_s": ".2f",
    "time_draft_s": ".3f", "time_target_s": ".3f",
    "time_verify_s": ".3f", "time_cache_s": ".3f",
}


def log_result(stats: dict, args, mode: str):
    """Print a one-line summary and append a CSV row — called once per mode."""
    print(f"\n  mode={mode:11s}  BE={stats['block_eff']:.4f}  "
          f"throughput={stats['throughput_tok_s']:.1f} tok/s  "
          f"avg_tree_nodes={stats['avg_tree_nodes']:.1f}")
    row = {
        "timestamp":  datetime.utcnow().isoformat(timespec="seconds"),
        "checkpoint": args.checkpoint, "dataset": args.dataset,
        "mode": mode, "K": args.K, "L": args.L,
    }
    for k, v in stats.items():
        row[k] = format(v, _FLOAT_FMT[k]) if k in _FLOAT_FMT else v
    append_csv_row(row)


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
    ap.add_argument("--seed",      type=int, default=DEFAULT_SEED)
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
                                        device=args.device, dtype=DEFAULT_DTYPE)

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
        sp = _state_path(RESULTS_CSV, mode, args.K, args.L, args.checkpoint)
        stats = evaluate_one_mode(p_model, q_model, tok, prompts, mode,
                                  K=args.K, L=args.L,
                                  max_new_tokens=args.max_new_tokens, temp=args.temp,
                                  state_path=sp)
        log_result(stats, args, mode)

    print(f"\n[done] results appended to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
