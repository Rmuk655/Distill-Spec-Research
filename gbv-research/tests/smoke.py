"""
smoke.py — pre-commit smoke test runner
========================================

Run this before every commit to catch regressions before they hit the repo.

What it does
------------
1. Runs the full unit test suite (~60 s, CPU-only, no model downloads)
2. Runs the pipeline smoke test (~35 min, requires GPU + models downloaded)

The pipeline smoke test exercises every loss function and every verifier at
reduced scale (50 steps instead of 1000, 5 eval prompts instead of 10) so
the full round-trip from training → merge → GBV eval is verified end-to-end.

Usage
-----
    # From gbv-research/
    python tests/smoke.py                    # unit tests + pipeline smoke
    python tests/smoke.py --unit-only        # unit tests only (<60 s, no GPU)
    python tests/smoke.py --pipeline-only    # pipeline smoke only (needs GPU)
    python tests/smoke.py --config server    # use server config instead of laptop

Pass criteria
-------------
- Exit code 0: all checks passed
- Exit code 1: at least one check failed (see output for details)

Equivalent manual commands
--------------------------
    python -m pytest tests/unit/ -q
    python orchestration/experiment.py --config laptop --smoke --yes

Pre-commit hook
---------------
The hook runs the unit suite automatically on every `git commit`.
The hook script is committed at hooks/pre-commit.
To install on a fresh clone (run once from the repo root):
    # Linux / macOS / Git Bash:
    cp gbv-research/hooks/pre-commit .git/hooks/pre-commit
    chmod +x .git/hooks/pre-commit
    # Windows (Git for Windows runs Python hooks natively — no chmod needed):
    copy gbv-research\\hooks\\pre-commit .git\\hooks\\pre-commit

Adding a new unit test
----------------------
Drop a test_*.py file in tests/unit/ — it is picked up automatically by both
the pre-commit hook and `python tests/run_unit_tests.py`. No registration needed.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).parent
_ROOT = _HERE.parent  # gbv-research/


def run_unit_tests() -> bool:
    """Run pytest unit suite. Returns True on pass."""
    print("\n" + "=" * 60)
    print("  STEP 1/2 — Unit tests (tests/unit/)")
    print("=" * 60)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/unit/", "-q"],
        cwd=str(_ROOT),
    )
    return result.returncode == 0


def run_pipeline_smoke(config: str) -> bool:
    """Run pipeline smoke test. Returns True on pass."""
    print("\n" + "=" * 60)
    print(f"  STEP 2/2 — Pipeline smoke (--config {config})")
    print("  Exercises: all losses (50 steps each) + all verifiers (5 prompts)")
    print("  Expected time: ~35 min on laptop GPU")
    print("=" * 60)
    result = subprocess.run(
        [sys.executable, "orchestration/experiment.py",
         "--config", config, "--smoke", "--yes"],
        cwd=str(_ROOT),
    )
    return result.returncode == 0


def main() -> None:
    p = argparse.ArgumentParser(
        description="Pre-commit smoke test: unit tests + pipeline smoke",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--unit-only",     action="store_true",
                   help="Run unit tests only (fast, no GPU needed)")
    p.add_argument("--pipeline-only", action="store_true",
                   help="Run pipeline smoke only (skips unit tests)")
    p.add_argument("--config",        default="laptop",
                   help="Pipeline config to use (default: laptop)")
    args = p.parse_args()

    results: dict[str, bool] = {}

    if not args.pipeline_only:
        results["unit_tests"] = run_unit_tests()

    if not args.unit_only:
        results["pipeline_smoke"] = run_pipeline_smoke(args.config)

    # Summary
    print("\n" + "=" * 60)
    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {status}  {name}")
        if not passed:
            all_passed = False
    print("=" * 60)

    if all_passed:
        print("\n  All smoke checks passed. Safe to commit.\n")
        sys.exit(0)
    else:
        print("\n  Smoke check(s) FAILED. Fix before committing.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
