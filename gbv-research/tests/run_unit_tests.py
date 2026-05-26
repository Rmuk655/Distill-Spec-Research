"""
run_unit_tests.py — entry-point script for running the unit test suite.

Usage (from gbv-research/ or from 2026 summer/):
    python gbv-research/tests/run_unit_tests.py          # full suite
    python gbv-research/tests/run_unit_tests.py --fast   # skip slow tests
    python gbv-research/tests/run_unit_tests.py -k losses  # only loss tests

Exit code:
    0  all tests passed
    1  one or more tests failed (blocks pre-commit)

The script forwards all extra arguments to pytest, so any pytest flag works:
    python gbv-research/tests/run_unit_tests.py -v --tb=long
"""

import sys
import os

# Ensure gbv-research/ is on sys.path so conftest.py is found
_HERE      = os.path.dirname(os.path.abspath(__file__))
_GBV_ROOT  = os.path.dirname(_HERE)

try:
    import pytest
except ImportError:
    print("[ERROR] pytest not found.  Install with: pip install pytest")
    sys.exit(1)

if __name__ == "__main__":
    test_dir = os.path.join(_GBV_ROOT, "tests", "unit")

    # Build pytest args
    args = [test_dir]
    for arg in sys.argv[1:]:
        if arg == "--fast":
            args += ["-m", "not slow"]
        else:
            args.append(arg)

    # Add config file location
    args += ["--rootdir", _GBV_ROOT, "--override-ini=testpaths="]

    print(f"[tests] Running unit tests in {test_dir}")
    print(f"[tests] pytest args: {args}")
    rc = pytest.main(args)
    sys.exit(rc)
