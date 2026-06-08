#!/usr/bin/env python3
"""
Reclaim disk under specdist storage after merge.

Scans checkpoints/ for training dirs whose *_merged/ sibling is eval-ready,
then drops resume-only artifacts (ckpt_step_*, ckpt_latest/, LoRA adapters, etc.).

Typical use on A100 when local storage quota is full:

    python deploy/reclaim_checkpoint_disk.py \\
        --storage-root $HOME/specdist

    python deploy/reclaim_checkpoint_disk.py --report \\
        --storage-root $HOME/specdist

    python deploy/reclaim_checkpoint_disk.py --dry-run --force
"""

from __future__ import annotations

import argparse
import os
import sys

_GBV = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _GBV not in sys.path:
    sys.path.insert(0, _GBV)

from algorithms.distillspec_gbv.trainer import cleanup_training_artifacts  # noqa: E402


def _dir_size(path: str) -> int:
    total = 0
    if os.path.isfile(path):
        return os.path.getsize(path)
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def _fmt_gb(nbytes: int) -> str:
    return f"{nbytes / 1024 ** 3:.2f} GB"


def _report_storage(storage_root: str) -> None:
    print(f"Storage root: {storage_root}")
    if not os.path.isdir(storage_root):
        print("  (not found)")
        return

    buckets = (
        "checkpoints",
        "hf_cache",
        "logs",
        "results.db",
    )
    for name in buckets:
        path = os.path.join(storage_root, name)
        if os.path.exists(path):
            print(f"  {name:14s} {_fmt_gb(_dir_size(path))}")
        else:
            print(f"  {name:14s} (missing)")

    ckpt_root = os.path.join(storage_root, "checkpoints")
    if not os.path.isdir(ckpt_root):
        return

    print("\nCheckpoint dirs (top 20 by size):")
    entries = []
    for name in os.listdir(ckpt_root):
        path = os.path.join(ckpt_root, name)
        if os.path.isdir(path) and not name.endswith("_merged"):
            entries.append((name, _dir_size(path)))
    entries.sort(key=lambda x: x[1], reverse=True)
    for name, size in entries[:20]:
        merged = os.path.join(ckpt_root, name + "_merged")
        tag = " [merged ready]" if os.path.isfile(os.path.join(merged, "config.json")) else ""
        print(f"  {name:40s} {_fmt_gb(size)}{tag}")


def _find_cleanup_targets(ckpt_root: str) -> list[str]:
    if not os.path.isdir(ckpt_root):
        return []
    targets: list[str] = []
    for name in sorted(os.listdir(ckpt_root)):
        if name.endswith("_merged"):
            continue
        path = os.path.join(ckpt_root, name)
        if not os.path.isdir(path):
            continue
        merged_cfg = os.path.join(ckpt_root, name + "_merged", "config.json")
        if os.path.isfile(merged_cfg):
            targets.append(path)
    return targets


def main() -> int:
    ap = argparse.ArgumentParser(description="Reclaim disk under specdist checkpoints/")
    ap.add_argument(
        "--storage-root",
        default=os.environ.get("SPECDIST_STORAGE_ROOT", ""),
        help="specdist root (default: $SPECDIST_STORAGE_ROOT)",
    )
    ap.add_argument("--report", action="store_true",
                    help="Print disk usage only; do not delete anything")
    ap.add_argument("--dry-run", action="store_true",
                    help="List targets without deleting")
    ap.add_argument("--force", action="store_true",
                    help="Re-run cleanup even if .post_merge_cleanup_done exists")
    args = ap.parse_args()

    storage_root = args.storage_root or os.path.expanduser("~/specdist")
    ckpt_root = os.path.join(storage_root, "checkpoints")

    _report_storage(storage_root)

    if args.report:
        return 0

    targets = _find_cleanup_targets(ckpt_root)
    if not targets:
        print("\nNo merged-ready checkpoint dirs found — nothing to reclaim.")
        print("Tip: delete abandoned training dirs manually, or prune old *_merged/ dirs")
        print("     for losses you no longer need.")
        return 0

    print(f"\nReclaim targets ({len(targets)}):")
    for path in targets:
        print(f"  {path}")

    if args.dry_run:
        print("\n[dry-run] no files deleted")
        return 0

    total_removed = 0
    for path in targets:
        before = _dir_size(path)
        cleanup_training_artifacts(path, force=args.force)
        after = _dir_size(path)
        total_removed += max(0, before - after)

    print(f"\nReclaimed approximately {_fmt_gb(total_removed)} across {len(targets)} dir(s).")
    _report_storage(storage_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
