#!/usr/bin/env python3
"""
04_baseline_eval.py — One-time baseline evaluation of the untrained GPT-2 draft.
Measures block efficiency of distilgpt2 (no training) against gpt2-medium.
This is the reference point every trained model is compared against.

Runtime: ~2-3 hr on CPU (5 prompts × 5 verifier modes + alpha + perplexity).
Logs to W&B if WANDB_API_KEY is set.
Results written to db/results.db (dashboard reads from here).

Usage:
    source deploy/ats/00_env.sh
    python deploy/ats/04_baseline_eval.py
    python deploy/ats/04_baseline_eval.py --n 2   # faster, fewer prompts
"""
import argparse
import os
import subprocess
import sys
import time

ROOT    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EVAL_PY = os.path.join(ROOT, "orchestration", "evaluate.py")
DATA_5  = os.path.join(ROOT, "core", "datasets", "raw", "gsm8k_5.jsonl")
LOG_DIR = os.path.join(ROOT, "db", "logs", "ats")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5,
                    help="Prompts per eval mode (default: 5, ~2-3 hr).")
    ap.add_argument("--modes", default="alpha",
                    help="Verifier modes (default: alpha only — BE needs Qwen3 KV cache format, see docs/ISSUES.md #3).")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)

    wandb_key = os.environ.get("WANDB_API_KEY", "")
    use_wandb = bool(wandb_key and wandb_key != "PASTE_YOUR_WANDB_KEY_HERE")

    print()
    print("=" * 60)
    print("  ATS Cloud — Baseline Evaluation")
    print("  Model:  distilgpt2 (untrained) → gpt2-medium")
    print(f"  Modes:  {args.modes}")
    print(f"  n:      {args.n} prompts per mode")
    print(f"  W&B:    {'enabled' if use_wandb else 'offline/disabled'}")
    n_modes = len(args.modes.split(","))
    # CPU eval: ~380 s/prompt per mode
    eta_hr = args.n * n_modes * 380 / 3600
    print(f"  Est.:   ~{eta_hr:.0f} hr on CPU ({args.n} × {n_modes} modes × ~6 min/prompt)")
    print()
    print("  Tip: run in a screen session:")
    print("    screen -S baseline")
    print("    python deploy/ats/04_baseline_eval.py")
    print("    Ctrl+A D  to detach; screen -r baseline to reattach")
    print("=" * 60)
    print()

    cmd = [
        sys.executable, EVAL_PY,
        "--student",       "distilgpt2",
        "--teacher",       "gpt2-medium",
        "--student_label", "baseline",
        "--datasets",      "gsm8k",
        "--modes",         args.modes,
        "--K",             "3",
        "--temperature",   "1.0",
        "--n",             str(args.n),
        "--max_tokens",    "50",
        "--hw_tier",       "laptop",   # GPT-2 = small-model tier
        "--skip_fetch",
        "--skip_existing",             # safe to re-run; skips already-done modes
        "--model_family",  "gpt2",
        "--wandb_project", os.environ.get("WANDB_PROJECT", "distillspec"),
        "--wandb_group",   "ats-gpt2-baseline",
    ]
    if not use_wandb:
        cmd.append("--no_wandb")

    if args.dry_run:
        print("DRY RUN — command that would run:")
        print(" ", " ".join(cmd))
        return

    log_path = os.path.join(LOG_DIR, "04_baseline_eval.log")
    print(f"Logging to: {log_path}")
    print(f"Streaming live output below:\n")

    t0 = time.time()
    # Stream output to both terminal and log file
    with open(log_path, "w", buffering=1) as lf:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            cwd=ROOT, text=True, bufsize=1,
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            lf.write(line)
        proc.wait()

    elapsed = (time.time() - t0) / 3600
    print()
    if proc.returncode == 0:
        print(f"✓  Baseline eval complete ({elapsed:.1f} hr).")
        print("   Results in db/results.db — start dashboard to view:")
        print("   bash deploy/ats/07_dashboard.sh")
        print()
        print("   Next: python deploy/ats/05_train_parallel.py")
    else:
        print(f"✗  Baseline eval FAILED (rc={proc.returncode}) after {elapsed:.1f} hr.")
        print(f"   Log: {log_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
