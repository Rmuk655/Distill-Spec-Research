#!/usr/bin/env python3
"""
05_train_parallel.py — Train all 15 GPT-2 losses in parallel on CPU.
128 GB RAM · 2-socket CPU. Each job uses ~2-3 GB RAM; 15 jobs = ~40 GB.

W&B logging: each loss creates its own W&B run under group 'ats-gpt2-train'.
Logs: db/logs/ats/train_<loss>.log  (tail any with monitor.py --tail <loss>)

Usage:
    source deploy/ats/00_env.sh
    screen -S training
    python deploy/ats/05_train_parallel.py
    Ctrl+A D   (detach screen — jobs keep running)

    # From another SSH session:
    python deploy/ats/monitor.py --watch

Flags:
    --steps N         Steps per loss (default: 1000)
    --losses kl,jsd   Train only specific losses
    --dry_run         Show plan without launching
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

ROOT     = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
TRAINER  = os.path.join(ROOT, "algorithms", "distillspec_gbv", "trainer.py")
DATASET  = os.path.join(ROOT, "core", "datasets", "raw", "gsm8k_train.jsonl")
LOG_DIR  = os.path.join(ROOT, "db", "logs", "ats")
CKPT_ROOT= os.path.join(ROOT, "db", "checkpoints")
PID_FILE = os.path.join(LOG_DIR, "ats_pids.json")

EXCLUDED = {"ebe","ebe_single","ebe_tree",
            "online","online_ebe","online_ebe_single","online_kl_tree","online_ebe_tree"}

ALL_LOSSES = [
    "kl","rev_kl","jsd","l1",
    "kl_tree","rev_kl_tree","jsd_tree",
    "bv_tree","gbv_tree","traversal_tree",
    "naive_tree","nss_tree","specinfer_tree","spectr_tree","khisti_tree",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps",   type=int, default=1000)
    ap.add_argument("--losses",  default=None)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    losses = ALL_LOSSES if not args.losses else \
             [l.strip() for l in args.losses.split(",")]
    losses = [l for l in losses if l not in EXCLUDED]

    wandb_key = os.environ.get("WANDB_API_KEY", "")
    use_wandb = bool(wandb_key and wandb_key != "PASTE_YOUR_WANDB_KEY_HERE")
    wandb_project = os.environ.get("WANDB_PROJECT", "distillspec")

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CKPT_ROOT, exist_ok=True)

    # Estimated time
    step_s  = 14   # ~14 s/step on CPU for GPT-2
    eta_hr  = args.steps * step_s / 3600
    ram_gb  = len(losses) * 2.5

    print()
    print("=" * 62)
    print("  ATS Cloud — Parallel Training")
    print(f"  Family : gpt2  (distilgpt2 → gpt2-medium)")
    print(f"  Losses : {len(losses)}")
    print(f"  Steps  : {args.steps} per loss")
    print(f"  Est.   : ~{eta_hr:.0f} hr (losses run in parallel)")
    print(f"  RAM    : ~{ram_gb:.0f} GB of 128 GB")
    print(f"  W&B    : {'enabled → ' + wandb_project if use_wandb else 'offline'}")
    print(f"  Logs   : {LOG_DIR}/train_<loss>.log")
    print()
    for i, l in enumerate(losses, 1):
        print(f"    {i:2d}. {l}")
    print("=" * 62)
    print()

    if args.dry_run:
        print("DRY RUN — not launching.")
        return

    procs = {}
    started_at = datetime.now().isoformat()

    for loss in losses:
        log_path = os.path.join(LOG_DIR, f"train_{loss}.log")
        log_f    = open(log_path, "w", buffering=1)
        cmd = [
            sys.executable, TRAINER,
            "--model_family", "gpt2",
            "--draft",        "distilgpt2",
            "--target",       "gpt2-medium",
            "--loss",         loss,
            "--steps",        str(args.steps),
            "--device",       "cpu",
            "--teacher_temp", "1.0",
            "--dataset",      DATASET,
            "--output",       os.path.join(CKPT_ROOT, f"{loss}-gpt2"),
            "--log_every",    "50",
            "--save_every",   "250",
            "--lora_r",       "8",
            "--lora_alpha",   "16",
            "--wandb_project", wandb_project,
            "--wandb_group",  "ats-gpt2-train",
            "--run_label",    f"gpt2-cpu-{loss}",
        ]
        if not use_wandb:
            cmd.append("--no_wandb")

        proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, cwd=ROOT)
        procs[loss] = {"pid": proc.pid, "proc": proc, "log": log_path}
        print(f"  [{loss:20s}] PID {proc.pid:6d}  →  train_{loss}.log")

    # Save PID file for monitor.py
    pid_data = {
        "started_at":  started_at,
        "family":      "gpt2",
        "draft":       "distilgpt2",
        "target":      "gpt2-medium",
        "steps":       args.steps,
        "device":      "cpu",
        "losses":      {l: procs[l]["pid"] for l in procs},
        "log_dir":     LOG_DIR,
        "log_prefix":  "train_",
    }
    with open(PID_FILE, "w") as f:
        json.dump(pid_data, f, indent=2)

    print()
    print(f"All {len(procs)} jobs launched.")
    print()
    print("Monitor from another SSH session:")
    print("  python deploy/ats/monitor.py --watch")
    print("  python deploy/ats/monitor.py --tail kl")
    print()
    print("This terminal will block until all jobs finish.")
    print("Safe to Ctrl+C — jobs continue running; use monitor.py to track.")
    print()

    # Wait and report
    try:
        failed = []
        for loss, info in procs.items():
            rc = info["proc"].wait()
            ts = datetime.now().strftime("%H:%M")
            if rc == 0:
                print(f"  [{ts}] [{loss:20s}] DONE")
            else:
                print(f"  [{ts}] [{loss:20s}] FAILED rc={rc}")
                failed.append(loss)
    except KeyboardInterrupt:
        print("\nInterrupted. Jobs still running — monitor with:")
        print("  python deploy/ats/monitor.py --watch")
        return

    print()
    print("=" * 62)
    if not failed:
        print(f"  ✓  All {len(procs)} losses trained successfully.")
        print()
        print("  Next: merge adapters + evaluate")
        print("  python deploy/ats/06_eval_trained.py")
    else:
        print(f"  ✗  {len(failed)} failed: {failed}")
        print(f"  Logs: {LOG_DIR}/train_<loss>.log")
    print("=" * 62)


if __name__ == "__main__":
    main()
