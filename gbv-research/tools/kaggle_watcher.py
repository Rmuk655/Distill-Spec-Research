"""
kaggle_watcher.py — Pull live Kaggle kernel output to a local folder and
                    keep the SpecDist dashboard refreshed automatically.

Usage
-----
    python gbv-research/tools/kaggle_watcher.py \\
        --kernel <username>/<kernel-slug> \\
        --dest   ~/Downloads/kaggle-run  \\
        --interval 30

Authentication
--------------
One-time setup (run once, credentials cached):

    kaggle auth login          # OAuth browser flow (recommended)
  OR
    # Save token from https://kaggle.com/settings/api → ~/.kaggle/kaggle.json
    #   { "username": "...", "key": "..." }

What it does
------------
Every --interval seconds:
  1. Calls the Kaggle Kernels Output API to download the latest files from
     /kaggle/working/specdist/ on the running (or completed) kernel.
  2. Maps them to the paths the local dashboard expects:
       specdist/results.db                  → <dest>/results.db
       specdist/logs/pipeline_output.log    → <dest>/pipeline_output.log
       specdist/logs/be_progress.log        → <dest>/be_progress.log
       specdist/pipeline_state_kaggle.json  → <dest>/pipeline_state_kaggle.json
  3. Prints a one-line status with file sizes and timestamps.

Run the dashboard in a second terminal:
    python gbv-research/dashboard/training_dashboard.py --root ~/Downloads/kaggle-run
"""

import argparse
import os
import shutil
import sys
import time
import zipfile
import io
from datetime import datetime


# ---------------------------------------------------------------------------
# File mapping: path inside the Kaggle kernel output zip → local filename
# ---------------------------------------------------------------------------
_FILE_MAP = {
    os.path.join("specdist", "results.db"):                        "results.db",
    os.path.join("specdist", "logs", "pipeline_output.log"):       "pipeline_output.log",
    os.path.join("specdist", "logs", "be_progress.log"):           "be_progress.log",
    os.path.join("specdist", "pipeline_state_kaggle.json"):        "pipeline_state_kaggle.json",
}


def _kaggle_api():
    """Return an authenticated KaggleApi instance."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApiExtended
    except ImportError:
        print("kaggle not installed. Run: pip install kaggle")
        sys.exit(1)
    api = KaggleApiExtended()
    api.authenticate()
    return api


def _fetch_once(api, kernel: str, dest: str) -> dict[str, int]:
    """
    Download kernel output files and copy them to dest.
    Returns dict of {local_filename: bytes_written} for changed files.
    """
    # kernels_output_download returns a zip-like response
    try:
        response = api.kernels_output_download(kernel, _preload_content=False)
        raw = response.read()
    except Exception as exc:
        print(f"  [warn] API call failed: {exc}")
        return {}

    # The response may be a zip archive or a single file
    updated = {}
    os.makedirs(dest, exist_ok=True)

    if zipfile.is_zipfile(io.BytesIO(raw)):
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            for archive_path, local_name in _FILE_MAP.items():
                # Kaggle normalises to forward slashes inside zip
                zip_key = archive_path.replace(os.sep, "/")
                if zip_key not in zf.namelist():
                    continue
                data = zf.read(zip_key)
                out_path = os.path.join(dest, local_name)
                # Only write if content changed (avoid needless dashboard reload flicker)
                if os.path.exists(out_path):
                    with open(out_path, "rb") as f:
                        if f.read() == data:
                            continue
                with open(out_path, "wb") as f:
                    f.write(data)
                updated[local_name] = len(data)
    else:
        # Single-file response — assume it's results.db (rare but possible)
        out_path = os.path.join(dest, "results.db")
        with open(out_path, "wb") as f:
            f.write(raw)
        updated["results.db"] = len(raw)

    return updated


def _fmt_size(n: int) -> str:
    if n >= 1_048_576:
        return f"{n/1_048_576:.1f} MB"
    if n >= 1024:
        return f"{n/1024:.1f} KB"
    return f"{n} B"


def _print_status(kernel: str, dest: str, updated: dict, elapsed: float):
    ts = datetime.now().strftime("%H:%M:%S")
    if updated:
        files = "  ".join(f"{name} ({_fmt_size(sz)})" for name, sz in updated.items())
        print(f"[{ts}] updated: {files}  ({elapsed:.1f}s)")
    else:
        print(f"[{ts}] no changes  ({elapsed:.1f}s)")


def _check_kernel_status(api, kernel: str) -> str:
    """Return kernel status string, or '?' if unavailable."""
    try:
        status = api.kernels_status(kernel)
        return status.get("status", "?")
    except Exception:
        return "?"


def main():
    p = argparse.ArgumentParser(
        description="Pull live Kaggle kernel output to a local folder for dashboard viewing."
    )
    p.add_argument("--kernel", required=True,
                   metavar="USER/SLUG",
                   help="Kaggle kernel identifier, e.g. mukund655/specdist-run")
    p.add_argument("--dest", default=os.path.expanduser("~/Downloads/kaggle-run"),
                   metavar="DIR",
                   help="Local destination folder (default: ~/Downloads/kaggle-run)")
    p.add_argument("--interval", type=int, default=30,
                   metavar="SEC",
                   help="Poll interval in seconds (default: 30)")
    p.add_argument("--once", action="store_true",
                   help="Pull once and exit (no polling loop)")
    args = p.parse_args()

    dest = os.path.abspath(os.path.expanduser(args.dest))
    os.makedirs(dest, exist_ok=True)

    print(f"\nKaggle Watcher")
    print(f"  kernel   : {args.kernel}")
    print(f"  dest     : {dest}")
    print(f"  interval : {args.interval}s")
    print(f"\nDashboard command (run in a separate terminal):")
    print(f"  python gbv-research/dashboard/training_dashboard.py --root \"{dest}\"\n")

    api = _kaggle_api()

    # Check kernel exists / is accessible
    kstatus = _check_kernel_status(api, args.kernel)
    print(f"Kernel status: {kstatus}\n")

    iteration = 0
    while True:
        iteration += 1
        t0 = time.time()
        updated = _fetch_once(api, args.kernel, dest)
        elapsed = time.time() - t0
        _print_status(args.kernel, dest, updated, elapsed)

        if args.once:
            break

        # Check if kernel has finished — slow down polling if so
        if iteration % 10 == 0:
            kstatus = _check_kernel_status(api, args.kernel)
            if kstatus in ("complete", "error", "cancelled"):
                print(f"Kernel status: {kstatus} — doing one final pull then stopping.")
                _fetch_once(api, args.kernel, dest)
                break
            print(f"  (kernel status: {kstatus})")

        time.sleep(args.interval)

    print("\nDone. Files are in:", dest)


if __name__ == "__main__":
    main()
