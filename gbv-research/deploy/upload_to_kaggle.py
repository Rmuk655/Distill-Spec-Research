#!/usr/bin/env python3
"""
upload_to_kaggle.py — upload model weights + eval datasets to Kaggle.

Run from the gbv-research/ directory (one-time setup, re-run to update versions):

    python deploy/upload_to_kaggle.py                  # both
    python deploy/upload_to_kaggle.py --only weights   # model weights only
    python deploy/upload_to_kaggle.py --only data      # eval datasets only

Requires:
    pip install kagglehub
    export KAGGLE_USERNAME=your-kaggle-username
    # Place ~/.kaggle/kaggle.json (API key) — https://www.kaggle.com/settings/account

After uploading:
    Attach the datasets to your Kaggle notebook:
        Add Data → Your Datasets → qwen3-hf-cache   → set KAGGLE_HF_DATASET in Cell 0
        Add Data → Your Datasets → specdist-datasets → set KAGGLE_DATA_DATASET in Cell 0
"""

from __future__ import annotations
import argparse
import os
import sys

def _check_kagglehub() -> None:
    try:
        import kagglehub  # noqa: F401
    except ImportError:
        print("kagglehub not installed. Run: pip install kagglehub")
        sys.exit(1)


def upload_model_weights(username: str) -> None:
    """Upload ~/.cache/huggingface/hub/ as username/qwen3-hf-cache."""
    import kagglehub

    hf_cache = os.path.expanduser("~/.cache/huggingface/hub")
    if not os.path.isdir(hf_cache):
        print(f"HF cache not found: {hf_cache}")
        print("Download first:")
        print("  huggingface-cli download Qwen/Qwen3-0.6B --ignore-patterns '*.gguf' '*.bin'")
        print("  huggingface-cli download Qwen/Qwen3-8B   --ignore-patterns '*.gguf' '*.bin'")
        return

    handle = f"{username}/qwen3-hf-cache"
    print(f"Uploading {hf_cache} → kaggle.com/datasets/{handle}")
    print("(skips *.gguf, *.bin, *.h5, *.msgpack — safetensors + configs only)")
    kagglehub.dataset_upload(
        handle,
        hf_cache,
        version_notes="Qwen3-0.6B + Qwen3-8B safetensors",
        ignore_patterns=["*.gguf", "*.bin", "*.h5", "*.msgpack", "*.pt"],
    )
    print(f"Done. Attach as KAGGLE_HF_DATASET = '/kaggle/input/qwen3-hf-cache'")


def upload_eval_datasets(username: str) -> None:
    """Upload core/datasets/raw/ as username/specdist-datasets."""
    import kagglehub

    raw_dir = os.path.join(os.path.dirname(__file__), "..", "core", "datasets", "raw")
    raw_dir = os.path.normpath(raw_dir)
    if not os.path.isdir(raw_dir):
        print(f"Datasets directory not found: {raw_dir}")
        print("Run the downloader first:")
        print("  python core/datasets/downloader.py")
        return

    files = [f for f in os.listdir(raw_dir) if f.endswith(".jsonl")]
    if not files:
        print(f"No .jsonl files in {raw_dir} — nothing to upload.")
        return

    handle = f"{username}/specdist-datasets"
    print(f"Uploading {len(files)} JSONL files from {raw_dir} → kaggle.com/datasets/{handle}")
    for f in sorted(files):
        size_kb = os.path.getsize(os.path.join(raw_dir, f)) // 1024
        print(f"  {f}  ({size_kb} KB)")
    kagglehub.dataset_upload(
        handle,
        raw_dir,
        version_notes="GSM8K, HumanEval, math500, alpaca, mtbench eval sets",
    )
    print(f"Done. Attach as KAGGLE_DATA_DATASET = '/kaggle/input/specdist-datasets'")


def main() -> None:
    _check_kagglehub()

    parser = argparse.ArgumentParser(description="Upload SpecDist datasets to Kaggle.")
    parser.add_argument("--only", choices=["weights", "data"],
                        help="Upload only one type (default: both)")
    parser.add_argument("--username", default=os.environ.get("KAGGLE_USERNAME", ""),
                        help="Kaggle username (or set KAGGLE_USERNAME env var)")
    args = parser.parse_args()

    if not args.username:
        print("Error: set KAGGLE_USERNAME env var or pass --username")
        print("  export KAGGLE_USERNAME=your-kaggle-username")
        sys.exit(1)

    if args.only != "data":
        upload_model_weights(args.username)
    if args.only != "weights":
        upload_eval_datasets(args.username)


if __name__ == "__main__":
    main()
