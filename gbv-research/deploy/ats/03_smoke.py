#!/usr/bin/env python3
"""
03_smoke.py — Smoke test on ATS hardware (~10-15 min).
Verifies every code path works on this specific machine before
committing to a full training run.

Tests:
  1. trainer.py: 10 steps, kl loss, gpt2 family, CPU
  2. runner.py:  2 prompts, gbv mode, gpt2 pair, CPU
  3. evaluate.py: perplexity check only (fast, no BE eval)

Usage:
    source deploy/ats/00_env.sh
    python deploy/ats/03_smoke.py
    python deploy/ats/03_smoke.py --losses kl,rev_kl,jsd  # test specific losses

Pass criteria: no crashes, finite loss values, acceptance rate > 0.
"""
import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

TRAINER  = os.path.join(ROOT, "algorithms", "distillspec_gbv", "trainer.py")
RUNNER   = os.path.join(ROOT, "algorithms", "distillspec_gbv", "verifiers", "runner.py")
EVAL_PY  = os.path.join(ROOT, "orchestration", "evaluate.py")
DATA     = os.path.join(ROOT, "core", "datasets", "raw", "gsm8k_30.jsonl")
LOG_DIR  = os.path.join(ROOT, "db", "logs", "ats")
CKPT_DIR = os.path.join(ROOT, "db", "checkpoints", "smoke")

SMOKE_LOSSES = [
    "kl", "rev_kl", "jsd", "l1",
    "kl_tree", "bv_tree", "gbv_tree",
]


def _run(label: str, cmd: list, timeout: int = 600) -> bool:
    """Run a command, print PASS/FAIL. Returns True on success."""
    print(f"  [{label:35s}] ", end="", flush=True)
    log_path = os.path.join(LOG_DIR, f"smoke_{label.replace(' ', '_')}.log")
    t0 = time.time()
    with open(log_path, "w") as lf:
        r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                           cwd=ROOT, timeout=timeout)
    elapsed = time.time() - t0
    if r.returncode == 0:
        print(f"PASS  ({elapsed:.0f}s)")
        return True
    else:
        print(f"FAIL  ({elapsed:.0f}s) → {log_path}")
        # Show last few lines
        with open(log_path, errors="replace") as f:
            tail = f.readlines()[-8:]
        for line in tail:
            print(f"         {line.rstrip()}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--losses", default=None,
                    help="Comma-separated losses to test (default: 7 representative)")
    args = ap.parse_args()

    losses = SMOKE_LOSSES
    if args.losses:
        losses = [l.strip() for l in args.losses.split(",")]

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)

    print()
    print("=" * 60)
    print("  ATS Cloud Smoke Test")
    print(f"  Testing {len(losses)} losses × 10 steps + verifier check")
    print("=" * 60)
    print()

    failures = []
    total = 0

    # ── 1. trainer.py: 10 steps per loss ──────────────────────────────────────
    print("Phase 1 — Training (10 steps each):")
    for loss in losses:
        total += 1
        ok = _run(
            f"train/{loss}",
            [
                sys.executable, TRAINER,
                "--model_family", "gpt2",
                "--draft",        "distilgpt2",
                "--target",       "gpt2-medium",
                "--loss",         loss,
                "--steps",        "10",
                "--device",       "cpu",
                "--no_wandb",
                "--dataset",      DATA,
                "--output",       os.path.join(CKPT_DIR, loss),
                "--log_every",    "5",
                "--max_new_tokens", "16",
                "--lora_r",       "4",
            ],
            timeout=300,
        )
        if not ok:
            failures.append(f"train/{loss}")
    print()

    # ── 2. runner.py: speculative decoding with GPT-2 ─────────────────────────
    print("Phase 2 — Verifier (2 prompts, gbv mode):")
    total += 1
    ok = _run(
        "runner/gbv",
        [
            sys.executable, RUNNER,
            "--p_model", "gpt2-medium",
            "--q_model", "distilgpt2",
            "--modes",   "gbv",
            "--Ks",      "3",
            "--p_temps", "1.0",
            "--data",    DATA,
            "--device",  "cpu",
            "--L",       "5",
            "--max_new_tokens", "16",
        ],
        timeout=600,
    )
    if not ok:
        failures.append("runner/gbv")
    print()

    # ── 3. evaluate.py: perplexity only (fast sanity check) ──────────────────
    print("Phase 3 — Perplexity check (baseline model):")
    total += 1
    ok = _run(
        "eval/perplexity",
        [
            sys.executable, EVAL_PY,
            "--student",       "distilgpt2",
            "--teacher",       "gpt2-medium",
            "--student_label", "smoke_baseline",
            "--datasets",      "gsm8k",
            "--modes",         "alpha",  # alpha only = fast (no BE subprocess)
            "--K",             "3",
            "--teacher_temps", "1.0",
            "--draft_temp",    "1.0",
            "--n",             "2",
            "--max_tokens",    "16",
            "--hw_tier",       "laptop",
            "--skip_fetch",
            "--no_wandb",
        ],
        timeout=600,
    )
    if not ok:
        failures.append("eval/perplexity")
    print()

    # ── Summary ───────────────────────────────────────────────────────────────
    passed = total - len(failures)
    print("=" * 60)
    if not failures:
        print(f"  ✓  All {total} smoke checks PASSED.")
        print()
        print("  Next: python deploy/ats/04_baseline_eval.py")
    else:
        print(f"  ✗  {len(failures)}/{total} checks FAILED: {failures}")
        print(f"  Logs: {LOG_DIR}/smoke_*.log")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
