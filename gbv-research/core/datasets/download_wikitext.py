"""
Download WikiText-2 for GPT-2 family distillation.

WikiText-2 is the correct dataset for distilgpt2 → gpt2-medium distillation:
  - Both models were trained on WebText (curated Reddit links = general English web text)
  - WikiText-2 is similar: clean Wikipedia prose, fully in-distribution
  - GSM8K math is completely OOD for GPT-2 → high KL, noisy gradients, poor convergence
  - WikiText → all losses converge cleanly, trends visible in 500 steps

Usage:
    python core/datasets/download_wikitext.py
    python core/datasets/download_wikitext.py --force   # re-download even if exists
"""

import argparse
import json
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_RAW  = os.path.join(_HERE, "raw")

TRAIN_PATH = os.path.join(_RAW, "wikitext_train.jsonl")
EVAL_10_PATH = os.path.join(_RAW, "wikitext_10.jsonl")
EVAL_5_PATH  = os.path.join(_RAW, "wikitext_5.jsonl")


def download(force=False):
    if os.path.exists(TRAIN_PATH) and not force:
        n = sum(1 for _ in open(TRAIN_PATH))
        print(f"wikitext_train.jsonl already exists ({n} prompts). Use --force to re-download.")
        return

    print("Downloading WikiText-2 (wikitext-2-raw-v1, train split)...")
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' library not installed. Run: pip install datasets")
        sys.exit(1)

    # Allow downloading even in offline mode
    for v in ("HF_DATASETS_OFFLINE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ.pop(v, None)

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    # Keep paragraphs of 20-300 words; skip section headers (=...=)
    prompts = []
    for item in ds:
        text = item["text"].strip()
        if not text or text.startswith("="):
            continue
        words = text.split()
        if 20 <= len(words) <= 300:
            prompts.append({"prompt": text})

    print(f"  {len(prompts)} usable paragraphs (filtered from {len(ds)} raw entries)")

    os.makedirs(_RAW, exist_ok=True)

    # Training set (all paragraphs)
    with open(TRAIN_PATH, "w", encoding="utf-8") as f:
        for p in prompts:
            f.write(json.dumps(p) + "\n")
    print(f"  Saved: wikitext_train.jsonl  ({len(prompts)} prompts)")

    # Fixed eval sets (seeded random)
    rng = random.Random(42)
    val_prompts = rng.sample(prompts, min(10, len(prompts)))

    with open(EVAL_10_PATH, "w", encoding="utf-8") as f:
        for p in val_prompts:
            f.write(json.dumps(p) + "\n")
    print(f"  Saved: wikitext_10.jsonl  (10 val prompts)")

    with open(EVAL_5_PATH, "w", encoding="utf-8") as f:
        for p in val_prompts[:5]:
            f.write(json.dumps(p) + "\n")
    print(f"  Saved: wikitext_5.jsonl   (5 val prompts)")

    print("\nDone. Update laptop_gpt2.yaml:")
    print("  dataset:")
    print("    train: core/datasets/raw/wikitext_train.jsonl")
    print("    eval:  core/datasets/raw/wikitext_5.jsonl")
    print("    val_dataset: wikitext_5.jsonl")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Download WikiText-2 for GPT-2 distillation")
    p.add_argument("--force", action="store_true", help="Re-download even if already exists")
    args = p.parse_args()
    download(force=args.force)
