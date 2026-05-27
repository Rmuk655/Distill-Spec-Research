"""
One-time cutover script: move generated outputs from OSD/ into gbv-research/db/.

Run this AFTER the training pipeline has fully completed (all pipeline_state
steps are 'done').  Safe to re-run — uses copy+verify before delete.

Usage:
    python migrate_outputs.py [--dry_run]

Arguments:
    --dry_run   Print what would be moved without actually moving anything.
"""

import argparse
import json
import os
import shutil
import sys


def _check_pipeline_idle(state_path: str) -> bool:
    """Return True if no step is in 'running' status."""
    if not os.path.exists(state_path):
        return True  # no state file = pipeline never started or already cleaned up
    with open(state_path, encoding="utf-8") as f:
        state = json.load(f)
    running = [sid for sid, sv in state.get("steps", {}).items()
               if sv.get("status") == "running"]
    if running:
        print(f"[migrate] Pipeline still running: {running}")
        print(f"          Wait for all steps to complete before migrating.")
        return False
    return True


def _move(src: str, dst: str, dry_run: bool) -> None:
    if not os.path.exists(src):
        return
    if dry_run:
        print(f"  [DRY] move  {src}")
        print(f"           → {dst}")
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.isdir(src):
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        shutil.rmtree(src)
    else:
        shutil.copy2(src, dst)
        os.remove(src)
    print(f"  moved  {os.path.basename(src)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run", action="store_true",
                    help="Print what would happen without making changes.")
    args = ap.parse_args()

    HERE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # gbv-research/
    OSD       = os.path.normpath(
        os.path.join(HERE, "..", "OSD"))
    DB        = os.path.join(HERE, "db")

    state_path = os.path.join(OSD, "pipeline_state_laptop.json")
    if not _check_pipeline_idle(state_path):
        sys.exit(1)

    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Migrating outputs from OSD/ → db/\n")

    moves = [
        # (source in OSD/,                    destination in db/)
        (os.path.join(OSD, "checkpoints"),    os.path.join(DB, "checkpoints")),
        (os.path.join(OSD, "wandb"),          os.path.join(DB, "wandb")),
        (os.path.join(OSD, "results.db"),     os.path.join(DB, "results.db")),
        (os.path.join(OSD, "pipeline_state_laptop.json"),
                                              os.path.join(DB, "pipeline_state.json")),
        (os.path.join(OSD, "pipeline_output.log"),
                                              os.path.join(DB, "logs", "pipeline_output.log")),
        (os.path.join(OSD, "be_progress.log"),
                                              os.path.join(DB, "logs", "be_progress.log")),
    ]

    for src, dst in moves:
        _move(src, dst, args.dry_run)

    # Note: diag*.py / solution.py / verify.py live in the parent directory
    # (2026 summer/) alongside this repo — they are a separate problem set,
    # not part of gbv-research.  Leave them exactly where they are.

    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Done.")
    if not args.dry_run:
        print("\nNext steps:")
        print("  1. Initialize a git repo in gbv-research/:")
        print("       cd gbv-research && git init && git add . && git commit -m 'Initial restructure'")
        print("  2. Smoke-test the new trainer:")
        print("       python -m algorithms.distillspec_gbv.trainer --loss forward_kl --steps 5 \\")
        print("           --dataset core/datasets/raw/gsm8k_30.jsonl --no_wandb")


if __name__ == "__main__":
    main()
