#!/usr/bin/env python3
"""
ATS Cloud Parallel Trainer
==========================
Launches all GPT-2 distillation losses in parallel on the Ubuntu bare-metal
server (128 GB RAM, 2-socket CPU, no GPU).

With 128 GB RAM you can run all 15 losses simultaneously.  Each job uses
~2-3 GB RAM (models + optimizer), well within the 128 GB budget.

Usage
-----
  python deploy/ats_train.py                      # all 15 losses, 1000 steps
  python deploy/ats_train.py --steps 500          # faster, fewer steps
  python deploy/ats_train.py --losses kl,rev_kl   # specific losses only
  python deploy/ats_train.py --family llama        # LLaMA (needs GPU attached)
  python deploy/ats_train.py --dry_run            # print plan without running

After training finishes, run evaluation:
  python deploy/ats_train.py --eval_only

Monitor progress in another SSH session:
  python deploy/ats_status.py
  python deploy/ats_status.py --tail kl           # live tail of one loss log
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

HERE     = os.path.dirname(os.path.abspath(__file__))
ROOT     = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

TRAINER   = os.path.join(ROOT, "algorithms", "distillspec_gbv", "trainer.py")
EVAL_MAIN = os.path.join(ROOT, "orchestration", "experiment.py")
DATASET   = os.path.join(ROOT, "core", "datasets", "raw", "gsm8k_train.jsonl")
LOG_DIR   = os.path.join(ROOT, "db", "logs", "ats")
CKPT_ROOT = os.path.join(ROOT, "db", "checkpoints")
PID_FILE  = os.path.join(ROOT, "db", "logs", "ats_pids.json")

# Losses that are known-broken — always excluded (see docs/ISSUES.md)
EXCLUDED = {
    "ebe", "ebe_single", "ebe_tree",
    "online", "online_ebe", "online_ebe_single", "online_kl_tree", "online_ebe_tree",
}

ALL_LOSSES = [
    "kl", "rev_kl", "jsd", "l1",
    "kl_tree", "rev_kl_tree", "jsd_tree",
    "bv_tree", "gbv_tree", "traversal_tree",
    "naive_tree", "nss_tree", "specinfer_tree", "spectr_tree", "khisti_tree",
]


def _ensure_online():
    """Clear offline env vars so model downloads work if needed."""
    for k in ("TRANSFORMERS_OFFLINE", "HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ.pop(k, None)


def _launch_loss(family: str, draft: str, target: str,
                 loss: str, steps: int, device: str,
                 teacher_temp: float) -> tuple:
    """Start trainer.py for one loss. Returns (process, log_path)."""
    log_path = os.path.join(LOG_DIR, f"{family}_{loss}.log")
    ckpt_out = os.path.join(CKPT_ROOT, f"{loss}-{family}")
    log_f = open(log_path, "w", buffering=1)

    cmd = [
        sys.executable, TRAINER,
        "--model_family", family,
        "--draft",        draft,
        "--target",       target,
        "--loss",         loss,
        "--steps",        str(steps),
        "--device",       device,
        "--no_wandb",
        "--teacher_temp", str(teacher_temp),
        "--dataset",      DATASET,
        "--output",       ckpt_out,
        "--log_every",    "50",   # print step every 50 steps
    ]
    proc = subprocess.Popen(
        cmd, stdout=log_f, stderr=subprocess.STDOUT,
        cwd=ROOT, text=True,
    )
    return proc, log_path


def main():
    ap = argparse.ArgumentParser(
        description="ATS Cloud parallel GPT-2 training — all losses at once.")
    ap.add_argument("--family",  default="gpt2",
                    help="Model family: gpt2 (default) | llama | qwen")
    ap.add_argument("--steps",   type=int, default=1000,
                    help="Training steps per loss (default: 1000)")
    ap.add_argument("--losses",  default=None,
                    help="Comma-separated loss subset. Default: all 15.")
    ap.add_argument("--device",  default="cpu",
                    help="Device: cpu (default) | cuda")
    ap.add_argument("--dry_run", action="store_true",
                    help="Print the plan without running.")
    ap.add_argument("--eval_only", action="store_true",
                    help="Skip training; just run evaluation.")
    ap.add_argument("--eval_config", default="server_gpt2",
                    help="Config name for eval-only mode (default: server_gpt2).")
    args = ap.parse_args()

    from core.model_families import get_family
    fam    = get_family(args.family)
    draft  = fam.default_draft_model_id
    target = fam.default_target_model_id
    # GPT-2 uses temperature=1.0 (no pre-scaling)
    teacher_temp = 1.0 if args.family == "gpt2" else 0.8

    losses = ALL_LOSSES
    if args.losses:
        losses = [l.strip() for l in args.losses.split(",") if l.strip()]
    losses = [l for l in losses if l not in EXCLUDED]

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CKPT_ROOT, exist_ok=True)

    # ── Eval-only mode ────────────────────────────────────────────────────────
    if args.eval_only:
        print("Eval-only mode: skipping training, running evaluation...")
        cmd = [
            sys.executable, EVAL_MAIN,
            "--config", args.eval_config,
            "--eval_only", "--yes",
        ]
        print("CMD:", " ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, cwd=ROOT)
        return

    # ── Print plan ────────────────────────────────────────────────────────────
    print()
    print("=" * 62)
    print("  ATS Cloud Parallel Training")
    print(f"  Family : {args.family}")
    print(f"  Draft  : {draft}")
    print(f"  Target : {target}")
    print(f"  Steps  : {args.steps} per loss")
    print(f"  Device : {args.device}")
    print(f"  Losses : {len(losses)}")
    for l in losses:
        print(f"           {l}")
    print(f"  Logs   : {LOG_DIR}/")
    ram_gb = len(losses) * 2.5
    print(f"  Est. RAM: ~{ram_gb:.0f} GB of 128 GB")
    step_s = 14 if args.device == "cpu" else 2
    eta_hr = (args.steps * step_s) / 3600
    print(f"  Est. time (sequential equiv): ~{eta_hr:.0f} hr; parallel: ~{eta_hr:.0f} hr")
    print("=" * 62)
    print()

    if args.dry_run:
        print("DRY RUN — no jobs launched.")
        return

    _ensure_online()

    # ── Launch all losses ─────────────────────────────────────────────────────
    started_at = datetime.now().isoformat()
    procs = {}   # loss → (proc, log_path)

    print(f"Launching {len(losses)} jobs at {started_at}...")
    for loss in losses:
        proc, log_path = _launch_loss(
            args.family, draft, target, loss,
            args.steps, args.device, teacher_temp,
        )
        procs[loss] = (proc, log_path)
        print(f"  [{loss:20s}] PID {proc.pid:6d}  →  logs/ats/{args.family}_{loss}.log")

    # Save PIDs so ats_status.py can find them
    pid_data = {
        "started_at":  started_at,
        "family":      args.family,
        "draft":       draft,
        "target":      target,
        "steps":       args.steps,
        "device":      args.device,
        "losses":      {loss: procs[loss][0].pid for loss in procs},
        "log_dir":     LOG_DIR,
    }
    with open(PID_FILE, "w") as f:
        json.dump(pid_data, f, indent=2)

    print()
    print(f"All {len(procs)} jobs running.")
    print(f"Monitor in another SSH session:")
    print(f"  python deploy/ats_status.py")
    print(f"  python deploy/ats_status.py --tail kl   # live tail")
    print()
    print("Waiting for all jobs to complete (this terminal will block)...")
    print("Safe to Ctrl+C — jobs continue as background processes.")
    print("Use 'python deploy/ats_status.py' to check progress.")
    print()

    # ── Wait and collect results ──────────────────────────────────────────────
    results = {}
    try:
        for loss, (proc, log_path) in procs.items():
            rc = proc.wait()
            status = "PASS" if rc == 0 else f"FAIL (rc={rc})"
            results[loss] = rc
            ts = datetime.now().strftime("%H:%M:%S")
            print(f"  [{ts}] [{loss:20s}] {status}")
    except KeyboardInterrupt:
        print("\nInterrupted — jobs still running in background.")
        print("Check with: python deploy/ats_status.py")
        return

    # ── Summary ───────────────────────────────────────────────────────────────
    passed = [l for l, rc in results.items() if rc == 0]
    failed = [l for l, rc in results.items() if rc != 0]
    print()
    print("=" * 62)
    print(f"  Training complete: {len(passed)} passed, {len(failed)} failed")
    if failed:
        print(f"  Failed: {failed}")
        print(f"  Logs:   {LOG_DIR}/")
    else:
        print(f"  All losses converged successfully.")
        print()
        print("  Next: run evaluation")
        print(f"  python orchestration/experiment.py --config {args.eval_config} "
              f"--eval_only --yes")
    print("=" * 62)


if __name__ == "__main__":
    main()
