#!/usr/bin/env python3
"""
06_eval_trained.py — Merge LoRA adapters and evaluate all trained models.
Runs after 05_train_parallel.py completes.

Steps per loss:
  1. Merge LoRA adapter into full model (trainer.py --merge_only)
  2. Evaluate merged model (evaluate.py, 5 prompts × 6 modes)
  3. Log results to W&B + results.db

Evaluations run sequentially (each uses all CPU resources for best speed).
Total runtime: ~2-3 hr per model → run overnight or in a screen session.

Usage:
    source deploy/ats/00_env.sh
    screen -S eval
    python deploy/ats/06_eval_trained.py
    Ctrl+A D   (detach; screen -r eval to reattach)

Flags:
    --losses kl,jsd    Evaluate only specific losses
    --n 2              Fewer prompts (faster, less precise)
    --skip_merge       Skip merge if already done
    --dry_run          Print plan without running
"""
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime

ROOT      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAINER   = os.path.join(ROOT, "algorithms", "distillspec_gbv", "trainer.py")
EVAL_PY   = os.path.join(ROOT, "orchestration", "evaluate.py")
CKPT_ROOT = os.path.join(ROOT, "db", "checkpoints")
LOG_DIR   = os.path.join(ROOT, "db", "logs", "ats")

EXCLUDED = {"ebe","ebe_single","ebe_tree",
            "online","online_ebe","online_ebe_single","online_kl_tree","online_ebe_tree"}

ALL_LOSSES = [
    "kl","rev_kl","jsd","l1",
    "kl_tree","rev_kl_tree","jsd_tree",
    "bv_tree","gbv_tree","traversal_tree",
    "naive_tree","nss_tree","specinfer_tree","spectr_tree","khisti_tree",
]


def _merge(loss: str, dry_run: bool) -> bool:
    adapter_dir = os.path.join(CKPT_ROOT, f"{loss}-gpt2")
    merged_dir  = os.path.join(CKPT_ROOT, f"{loss}-gpt2-merged")
    log_path    = os.path.join(LOG_DIR, f"merge_{loss}.log")

    # Check adapter exists
    if not os.path.isdir(adapter_dir):
        print(f"  [{loss:20s}] SKIP merge — no adapter at {adapter_dir}")
        return False

    # Check already merged
    if os.path.isdir(merged_dir):
        print(f"  [{loss:20s}] merge already done → {merged_dir}")
        return True

    cmd = [
        sys.executable, TRAINER,
        "--merge_only",
        "--model_family", "gpt2",
        "--draft",        "distilgpt2",
        "--adapter",      adapter_dir,
        "--output",       merged_dir,
    ]
    if dry_run:
        print(f"  [{loss:20s}] [DRY] merge → {merged_dir}")
        return True

    t0 = time.time()
    with open(log_path, "w") as lf:
        r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=ROOT)
    elapsed = time.time() - t0
    if r.returncode == 0:
        print(f"  [{loss:20s}] merged ({elapsed:.0f}s) → {merged_dir}")
        return True
    else:
        print(f"  [{loss:20s}] MERGE FAILED — see {log_path}")
        return False


def _evaluate(loss: str, merged_dir: str, n: int,
              use_wandb: bool, wandb_project: str, dry_run: bool) -> bool:
    log_path = os.path.join(LOG_DIR, f"eval_{loss}.log")
    cmd = [
        sys.executable, EVAL_PY,
        "--student",       merged_dir,
        "--teacher",       "gpt2-medium",
        "--student_label", loss,
        "--datasets",      "gsm8k",
        "--modes",         "alpha,bv,gbv,traversal,specinfer,naive",
        "--K",             "3",
        "--teacher_temps", "1.0",
        "--draft_temp",    "1.0",
        "--n",             str(n),
        "--max_tokens",    "50",
        "--hw_tier",       "laptop",
        "--skip_fetch",
        "--skip_existing",
        "--model_family",  "gpt2",
        "--loss_name",     loss,
        "--train_steps",   "1000",
        "--wandb_project", wandb_project,
        "--wandb_group",   "ats-gpt2-eval",
        "--run_label",     f"gpt2-cpu-{loss}",
    ]
    if not use_wandb:
        cmd.append("--no_wandb")

    if dry_run:
        print(f"  [{loss:20s}] [DRY] eval n={n}")
        return True

    t0 = time.time()
    print(f"  [{loss:20s}] evaluating (n={n}, 6 modes × ~6 min/prompt)...")
    with open(log_path, "w") as lf:
        r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=ROOT)
    elapsed = (time.time() - t0) / 60
    if r.returncode == 0:
        print(f"  [{loss:20s}] eval done ({elapsed:.0f} min)")
        return True
    else:
        print(f"  [{loss:20s}] EVAL FAILED — see {log_path}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--losses",      default=None)
    ap.add_argument("--n",           type=int, default=5)
    ap.add_argument("--skip_merge",  action="store_true")
    ap.add_argument("--dry_run",     action="store_true")
    args = ap.parse_args()

    losses = ALL_LOSSES if not args.losses else \
             [l.strip() for l in args.losses.split(",")]
    losses = [l for l in losses if l not in EXCLUDED]

    wandb_key     = os.environ.get("WANDB_API_KEY", "")
    use_wandb     = bool(wandb_key and wandb_key != "PASTE_YOUR_WANDB_KEY_HERE")
    wandb_project = os.environ.get("WANDB_PROJECT", "distillspec")

    os.makedirs(LOG_DIR, exist_ok=True)

    # Estimated time: n × 6 modes × 380 s/prompt on CPU
    eta_per = args.n * 6 * 380 / 3600
    eta_total = eta_per * len(losses)

    print()
    print("=" * 62)
    print("  ATS Cloud — Eval Trained Models")
    print(f"  Losses : {len(losses)}")
    print(f"  n      : {args.n} prompts per mode")
    print(f"  Est.   : ~{eta_per:.1f} hr/model × {len(losses)} = ~{eta_total:.0f} hr total")
    print(f"  W&B    : {'enabled → ' + wandb_project if use_wandb else 'offline'}")
    print()
    print("  Run in a screen session (see header).")
    print("=" * 62)
    print()

    failed_merge = []
    failed_eval  = []
    t_start      = time.time()

    for i, loss in enumerate(losses, 1):
        merged_dir = os.path.join(CKPT_ROOT, f"{loss}-gpt2-merged")
        ts = datetime.now().strftime("%H:%M")
        print(f"\n[{ts}] {i}/{len(losses)} — {loss}")
        print("-" * 40)

        # Merge
        if not args.skip_merge:
            ok = _merge(loss, args.dry_run)
            if not ok:
                failed_merge.append(loss)
                continue
        else:
            if not os.path.isdir(merged_dir):
                print(f"  [{loss:20s}] no merged model, skipping eval")
                failed_merge.append(loss)
                continue

        # Eval
        ok = _evaluate(loss, merged_dir, args.n, use_wandb, wandb_project, args.dry_run)
        if not ok:
            failed_eval.append(loss)

    elapsed_hr = (time.time() - t_start) / 3600
    print()
    print("=" * 62)
    print(f"  Completed in {elapsed_hr:.1f} hr")
    if not failed_merge and not failed_eval:
        print(f"  ✓  All {len(losses)} models evaluated.")
        print()
        print("  View results:")
        print("    bash deploy/ats/07_dashboard.sh")
        print("    python db/analyze_results.py")
    else:
        if failed_merge: print(f"  ✗  Merge failed: {failed_merge}")
        if failed_eval:  print(f"  ✗  Eval failed:  {failed_eval}")
        print(f"  Logs: {LOG_DIR}/merge_*.log  {LOG_DIR}/eval_*.log")
    print("=" * 62)


if __name__ == "__main__":
    main()
