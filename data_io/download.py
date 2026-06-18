"""
download.py — fetch the 5 evaluation datasets + GSM8K training split to JSONL.

Each output is one JSON object per line with at least {"prompt": "..."}.
Files land in data_io/raw/.

Usage:
    python -m data_io.download                 # fetch everything (default n=100 per eval set)
    python -m data_io.download --datasets gsm8k,math500
    python -m data_io.download --n 1000        # bigger eval sets
    python -m data_io.download --train         # also fetch gsm8k train split

Files produced:
    raw/gsm8k_train.jsonl     — 7,473 GSM8K train prompts (training data)
    raw/gsm8k_val.jsonl       — first 100 of GSM8K test (validation during training)
    raw/gsm8k_eval.jsonl      — next 200..200+n of GSM8K test (final eval)
    raw/alpaca.jsonl, math500.jsonl, humaneval.jsonl, mtbench.jsonl  — eval sets

The val / eval split of GSM8K test is deterministic (random.Random(42) shuffle)
and the two pools never overlap, so val-best checkpoint selection does NOT bias
the reported eval numbers.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import urllib.request

# All files live under data_io/raw/ — the train and eval scripts know to look here.
HERE     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "raw")
os.makedirs(DATA_DIR, exist_ok=True)

# Val pool size = how many of the 1319 GSM8K test prompts we hold out for
# validation during training.  The eval set starts at index 200, so we leave
# a 100-prompt buffer between val and eval to be safe.
VAL_POOL_SIZE = 200


def save_jsonl(path: str, items: list) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            if isinstance(item, str):
                item = {"prompt": item}
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  -> {len(items):4d} prompts saved to {os.path.relpath(path, HERE)}")


# ---------------------------------------------------------------------------
# Individual dataset fetchers
# ---------------------------------------------------------------------------

def fetch_gsm8k_train(force: bool = False) -> str:
    """Download the full 7473-prompt GSM8K train split (training data)."""
    path = os.path.join(DATA_DIR, "gsm8k_train.jsonl")
    if os.path.isfile(path) and not force:
        print(f"  gsm8k_train.jsonl already present — skipping")
        return path

    # Direct GitHub source — avoids HF Hub auth / offline issues.
    url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl"
    print(f"  fetching gsm8k_train from {url} ...")
    items = []
    with urllib.request.urlopen(url, timeout=60) as r:
        for line in r:
            obj = json.loads(line)
            items.append({"prompt": obj["question"], "answer": obj["answer"]})
    save_jsonl(path, items)
    return path


def fetch_gsm8k_val_and_eval(n_eval: int = 100, force: bool = False):
    """
    Download GSM8K test (1319 prompts), then split into:
        items[0:100]    → gsm8k_val.jsonl    (used during training for val_loss)
        items[200:200+n_eval] → gsm8k_eval.jsonl   (used by eval.py)
    The 100-prompt buffer at [100..200] is never used.

    Both splits use a deterministic shuffle (Random(42)) so the same prompts
    land in the same buckets across machines and across reruns.
    """
    val_path  = os.path.join(DATA_DIR, "gsm8k_val.jsonl")
    eval_path = os.path.join(DATA_DIR, f"gsm8k_eval.jsonl")

    if os.path.isfile(val_path) and os.path.isfile(eval_path) and not force:
        print(f"  gsm8k_val.jsonl + gsm8k_eval.jsonl already present — skipping")
        return val_path, eval_path

    url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
    print(f"  fetching gsm8k_test from {url} ...")
    items = []
    with urllib.request.urlopen(url, timeout=60) as r:
        for line in r:
            obj = json.loads(line)
            items.append({"prompt": obj["question"], "answer": obj["answer"]})

    # Deterministic shuffle — same seed everywhere, no global state.
    random.Random(42).shuffle(items)

    save_jsonl(val_path,  items[0:100])
    save_jsonl(eval_path, items[VAL_POOL_SIZE : VAL_POOL_SIZE + n_eval])
    return val_path, eval_path


def fetch_hf(dataset_id: str, split: str, field: str, name: str,
             n: int = 100, config: str = None, extra_field: str = None,
             force: bool = False) -> str:
    """
    Generic HuggingFace dataset fetch — used for alpaca/math500/humaneval/mtbench.

    Falls back to a small list of canned prompts if the HF Hub is unreachable.
    """
    path = os.path.join(DATA_DIR, f"{name}.jsonl")
    if os.path.isfile(path) and not force:
        print(f"  {name}.jsonl already present — skipping")
        return path

    print(f"  fetching {name} from {dataset_id} ...")
    try:
        from datasets import load_dataset
        ds = load_dataset(dataset_id, config, split=split) if config \
             else load_dataset(dataset_id, split=split)
        items = []
        for row in ds:
            prompt = row.get(field, "")
            if isinstance(prompt, list):
                prompt = prompt[0] if prompt else ""
            if not (prompt and isinstance(prompt, str)):
                continue
            item = {"prompt": prompt}
            if extra_field and extra_field in row:
                item["answer"] = row[extra_field]
            items.append(item)
            if len(items) >= n:
                break
        save_jsonl(path, items)
    except Exception as e:
        print(f"  WARNING: HF download failed for {name}: {e}")
        # Minimal canned fallback so the pipeline at least runs.
        items = [{"prompt": f"Test prompt {i} for {name}."} for i in range(min(n, 10))]
        save_jsonl(path, items)
    return path


# ---------------------------------------------------------------------------
# Dataset registry — eval scripts look up file paths through this.
# ---------------------------------------------------------------------------

EVAL_DATASETS = {
    # name        → (hf_id,                       split,  prompt_field,   config, answer_field)
    "alpaca":     ("tatsu-lab/alpaca",            "train",   "instruction", None,  "output"),
    "math500":    ("HuggingFaceH4/MATH-500",      "test",    "problem",     None,  "answer"),
    "humaneval":  ("openai/openai_humaneval",       "test",    "prompt",      None,  "canonical_solution"),
    "mtbench":    ("philschmid/mt-bench",         "train",   "turns",       None,  None),
}

# Training datasets backed by hendrycks/competition_math.
# level_filter: set of "Level N" strings to keep, or None for all levels.
MATH_TRAIN_DATASETS = {
    "math_train": ("train", None),
    "math_hard":  ("train", {"Level 4", "Level 5"}),
    "math_val":   ("test",  None),
}


def fetch_math(name: str, split: str, level_filter: set | None = None,
               n: int = 10_000, force: bool = False) -> str:
    """Fetch hendrycks/competition_math, optionally filtered by level."""
    path = os.path.join(DATA_DIR, f"{name}.jsonl")
    if os.path.isfile(path) and not force:
        print(f"  {name}.jsonl already present — skipping")
        return path

    label = f"levels {sorted(level_filter)}" if level_filter else "all levels"
    print(f"  fetching {name} (competition_math {split}, {label}) ...")
    try:
        from datasets import load_dataset
        ds = load_dataset("hendrycks/competition_math", split=split)
        items = []
        for row in ds:
            if level_filter and row.get("level") not in level_filter:
                continue
            prompt = row.get("problem", "")
            if not (prompt and isinstance(prompt, str)):
                continue
            items.append({"prompt": prompt, "answer": row.get("solution", ""),
                          "level": row.get("level", ""), "type": row.get("type", "")})
            if len(items) >= n:
                break
        save_jsonl(path, items)
    except Exception as e:
        print(f"  WARNING: HF download failed for {name}: {e}")
        items = [{"prompt": f"Test prompt {i} for {name}."} for i in range(10)]
        save_jsonl(path, items)
    return path


def fetch_all(n_eval: int = 100, train: bool = False, datasets: list = None,
              force: bool = False) -> None:
    """Top-level entry point — fetch every dataset we evaluate on."""
    requested = set(datasets) if datasets else set(EVAL_DATASETS) | {"gsm8k"}

    if train:
        fetch_gsm8k_train(force=force)

    if "gsm8k" in requested:
        fetch_gsm8k_val_and_eval(n_eval=n_eval, force=force)

    for name, (split, level_filter) in MATH_TRAIN_DATASETS.items():
        if name in requested:
            fetch_math(name, split=split, level_filter=level_filter, force=force)

    for name, (hf_id, split, field, config, ans) in EVAL_DATASETS.items():
        if name not in requested:
            continue
        fetch_hf(hf_id, split, field, name, n=n_eval, config=config,
                 extra_field=ans, force=force)


def get_path(name: str) -> str:
    """Resolve a dataset name to its on-disk JSONL path (must be downloaded)."""
    known = ({"gsm8k_train", "gsm8k_val", "gsm8k_eval"}
             | set(EVAL_DATASETS) | set(MATH_TRAIN_DATASETS))
    if name in known:
        return os.path.join(DATA_DIR, f"{name}.jsonl")
    raise KeyError(f"Unknown dataset '{name}'. Known: {sorted(known)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", type=str, default=None,
                    help="Comma-separated subset (e.g. 'gsm8k,math500').  Default: all.")
    ap.add_argument("--n",        type=int, default=1000,
                    help="Prompts per eval set (default 1000; GSM8K test has 1319 total).")
    ap.add_argument("--train",    action="store_true",
                    help="Also fetch gsm8k_train.jsonl (the 7473-prompt training split).")
    ap.add_argument("--force",    action="store_true",
                    help="Re-download even if files exist.")
    args = ap.parse_args()
    ds = [d.strip() for d in args.datasets.split(",")] if args.datasets else None
    fetch_all(n_eval=args.n, train=args.train, datasets=ds, force=args.force)
    print("\nDone.")
