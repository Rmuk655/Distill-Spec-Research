#!/usr/bin/env python
"""
smoke_family.py — quick end-to-end smoke test for a model family.

Downloads (or uses cached) draft + target weights for the given family,
then runs N steps of forward-KL distillation training directly via
trainer.py.  No experiment.py orchestration overhead.

This is the second gate after test_model_families.py (unit tests with
synthetic tensors).  Run this when you want to confirm that a family's
real weights load correctly and that LoRA wraps without error.

Usage:
    python scripts/smoke_family.py --family gpt2
    python scripts/smoke_family.py --family llama --steps 3
    python scripts/smoke_family.py --family qwen --draft Qwen/Qwen2.5-0.5B --target Qwen/Qwen3-0.6B
    python scripts/smoke_family.py --family gpt2 --all_losses   # smoke every loss variant

Exit codes:
    0 — PASS
    1 — FAIL (trainer returned non-zero or crashed)
"""

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Add repo root so we can import core.model_families without installing
sys.path.insert(0, ROOT)

from core.model_families import FAMILY_REGISTRY, get_family  # noqa: E402

TRAINER   = os.path.join(ROOT, "algorithms", "distillspec_gbv", "trainer.py")
SMOKE_DS  = os.path.join(ROOT, "core", "datasets", "raw", "gsm8k_30.jsonl")

# Losses to run in --all_losses mode (excludes known-broken ones; see docs/ISSUES.md)
_SMOKE_LOSSES = [
    "forward_kl", "rev_kl", "jsd", "l1",
    "kl_tree", "rev_kl_tree", "jsd_tree",
    "bv_tree", "gbv_tree", "traversal_tree",
]


def _run_loss(family_name: str, draft: str, target: str,
              loss: str, steps: int, device: str) -> bool:
    """Run trainer.py for a single loss.  Returns True on success."""
    cmd = [
        sys.executable, TRAINER,
        "--model_family", family_name,
        "--draft",        draft,
        "--target",       target,
        "--loss",         loss,
        "--steps",        str(steps),
        "--dataset",      SMOKE_DS,
        "--no_wandb",
        "--max_new_tokens", "16",
        "--lora_r",       "4",
        "--lora_alpha",   "8",
        "--device",       device,
    ]
    print(f"  [{loss:20s}] ", end="", flush=True)
    ret = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if ret.returncode == 0:
        print("PASS")
        return True
    else:
        print("FAIL")
        # Show last few lines of stderr to help diagnose
        tail = "\n".join(ret.stderr.strip().splitlines()[-8:])
        print(f"           stderr tail:\n{tail}")
        return False


def main():
    ap = argparse.ArgumentParser(
        description="Quick end-to-end smoke test for a model family.")
    ap.add_argument("--family",  required=True,
                    choices=sorted(FAMILY_REGISTRY),
                    help="Model family to test (must be registered in FAMILY_REGISTRY).")
    ap.add_argument("--draft",   default=None,
                    help="Override draft model ID (default: family.default_draft_model_id).")
    ap.add_argument("--target",  default=None,
                    help="Override target model ID (default: family.default_target_model_id).")
    ap.add_argument("--steps",   type=int, default=5,
                    help="Training steps per loss (default: 5).")
    ap.add_argument("--device",  default="cuda",
                    choices=["cuda", "cpu"],
                    help="Device to use (default: cuda; use cpu if no GPU).")
    ap.add_argument("--all_losses", action="store_true",
                    help="Smoke-test every non-broken loss variant, not just forward_kl.")
    args = ap.parse_args()

    family = get_family(args.family)
    draft  = args.draft  or family.default_draft_model_id
    target = args.target or family.default_target_model_id
    losses = _SMOKE_LOSSES if args.all_losses else ["forward_kl"]

    print(f"\n{'='*60}")
    print(f"  smoke_family.py")
    print(f"  family  : {args.family}")
    print(f"  draft   : {draft}")
    print(f"  target  : {target}")
    print(f"  steps   : {args.steps}")
    print(f"  device  : {args.device}")
    print(f"  losses  : {', '.join(losses)}")
    print(f"  lora    : {family.lora_target_modules()}")
    print(f"{'='*60}\n")

    results = {}
    for loss in losses:
        results[loss] = _run_loss(
            args.family, draft, target, loss, args.steps, args.device)

    passed = sum(results.values())
    failed = len(results) - passed

    print(f"\n{'='*60}")
    print(f"  Results: {passed}/{len(results)} passed")
    if failed:
        print(f"  FAILED : {', '.join(k for k, v in results.items() if not v)}")
    print(f"{'='*60}\n")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
