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
    raw/spec_bench.jsonl      — Spec-Bench (480 prompts, 6 categories) — paper eval
    raw/math_hard.jsonl, math_val.jsonl, math_eval.jsonl        — MATH level 4+5
    raw/olympiad_eval.jsonl   — OlympiadBench, EVAL-ONLY (no val/train split);
        1177 rows total (open-ended + theorem-proving English text-only math
        combined); NOT fetched by --train default — opt in with
        --datasets olympiad_eval

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


def fetch_spec_bench(force: bool = False) -> str:
    """Download Spec-Bench — the standard speculative-decoding eval benchmark.

    Spec-Bench (Xia et al. 2024, https://github.com/hemingkx/Spec-Bench) is the
    field-standard multi-domain benchmark for measuring speculative-decoding
    speedup: 480 prompts across 6 categories (80 each) — multi-turn conversation,
    translation, summarization, question answering, mathematical reasoning, and
    retrieval-augmented generation.

    USE FOR THE PAPER: report block-eff per category to show the draft transfers
    beyond the training domain.  For day-to-day iteration, keep using math_hard.

    The per-row "category" field is preserved so eval can break results down by
    subdomain.  Each row's first turn is taken as the prompt.
    """
    path = os.path.join(DATA_DIR, "spec_bench.jsonl")
    if os.path.isfile(path) and not force:
        print("  spec_bench.jsonl already present — skipping")
        return path

    url = ("https://raw.githubusercontent.com/hemingkx/Spec-Bench/"
           "main/data/spec_bench/question.jsonl")
    print(f"  fetching spec_bench from {url} ...")
    items = []
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            for line in r:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                turns = obj.get("turns") or []
                prompt = turns[0] if turns else obj.get("prompt", "")
                if not (prompt and isinstance(prompt, str)):
                    continue
                items.append({"prompt": prompt,
                              "category": obj.get("category", ""),
                              "question_id": obj.get("question_id", "")})
        save_jsonl(path, items)
    except Exception as e:
        print(f"  WARNING: spec_bench download failed: {e}")
        items = [{"prompt": f"Test prompt {i} for spec_bench.", "category": "fallback"}
                 for i in range(10)]
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

_HENDRYCKS_SUBJECTS = [
    "algebra", "counting_and_probability", "geometry",
    "intermediate_algebra", "number_theory", "prealgebra", "precalculus",
]
_HARD_LEVELS = {"Level 4", "Level 5", "4", "5", 4, 5}


def _is_hard(level) -> bool:
    return str(level).strip() in _HARD_LEVELS


def fetch_math_hard_and_val(force: bool = False,
                            n_val: int = 200, n_eval: int = 1000):
    """
    Build math_hard / math_val / math_eval from level-4+5 MATH problems.

    Pools all level-4+5 problems from both HF splits, shuffles with seed 42,
    then carves out fixed-size val and eval pools (same pattern as gsm8k):
        math_val   — n_val  problems (default 200) for during-training checkpoint selection
        math_eval  — n_eval problems (default 1000) held out for final eval
        math_hard  — remainder for training (~5332 from 6532 total)

    Strategy A — EleutherAI/hendrycks_math (7 subject configs, ~6532 level-4+5 problems).
    Strategy B — fallback to HuggingFaceH4/MATH-500 (~262 level-4+5 problems).
    """
    hard_path = os.path.join(DATA_DIR, "math_hard.jsonl")
    val_path  = os.path.join(DATA_DIR, "math_val.jsonl")
    eval_path = os.path.join(DATA_DIR, "math_eval.jsonl")

    if (os.path.isfile(hard_path) and os.path.isfile(val_path)
            and os.path.isfile(eval_path) and not force):
        print("  math_hard.jsonl + math_val.jsonl + math_eval.jsonl already present — skipping")
        return hard_path, val_path, eval_path

    def _row_to_item(row):
        prompt = row.get("problem", "")
        if not (prompt and isinstance(prompt, str)):
            return None
        return {"prompt": prompt,
                "answer": row.get("solution", row.get("answer", "")),
                "level":  row.get("level", ""),
                "type":   row.get("type", row.get("subject", ""))}

    def _split_and_save(all_items):
        random.Random(42).shuffle(all_items)
        # val and eval are carved from the front so they never overlap train
        save_jsonl(val_path,  all_items[:n_val])
        save_jsonl(eval_path, all_items[n_val:n_val + n_eval])
        save_jsonl(hard_path, all_items[n_val + n_eval:])

    # ── Strategy A: pool both HF splits, filter levels 4+5 ───────────────────
    print("  fetching EleutherAI/hendrycks_math (7 subjects × train+test) ...")
    try:
        from datasets import load_dataset
        all_hard = []
        for subj in _HENDRYCKS_SUBJECTS:
            for split in ("train", "test"):
                ds = load_dataset("EleutherAI/hendrycks_math", subj, split=split)
                for row in ds:
                    item = _row_to_item(row)
                    if item and _is_hard(item["level"]):
                        all_hard.append(item)
        if all_hard:
            _split_and_save(all_hard)
            return hard_path, val_path, eval_path
        print("  EleutherAI/hendrycks_math returned empty — falling back to MATH-500 split")
    except Exception as e:
        print(f"  EleutherAI/hendrycks_math failed ({e}) — falling back to MATH-500 split")

    # ── Strategy B: MATH-500, filter levels 4+5 ──────────────────────────────
    print("  fetching HuggingFaceH4/MATH-500 as fallback ...")
    try:
        from datasets import load_dataset
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        all_hard = [item for row in ds
                    if (item := _row_to_item(row)) and _is_hard(item["level"])]
        _split_and_save(all_hard)
    except Exception as e:
        print(f"  WARNING: all downloads failed: {e}")
        for p in (hard_path, val_path, eval_path):
            save_jsonl(p, [{"prompt": f"Test prompt {i}."} for i in range(10)])
    return hard_path, val_path, eval_path


def fetch_olympiad_eval(force: bool = False):
    """
    Build olympiad_eval (EVAL-ONLY — no val, no train split) from OlympiadBench —
    a harder-than-MATH competition benchmark, used as a held-out cross-domain
    axis to check whether a draft trained on math_hard transfers to harder
    problems. There is no olympiad_hard / olympiad_val: this dataset is never
    trained or checkpoint-selected on, only evaluated with eval.py.

    NOT auto-fetched by --train (opt-in only, via --datasets olympiad_eval)
    since it's a large extra download most setups don't need.

    Schema CONFIRMED (2026-07-01, on-cluster checks with a real HF token, plus
    a survey of every other "olympiadbench"-named HF dataset by download count
    — most turned out to be repackagings of this same 674-row source, not
    independently larger data). Combines TWO English text-only configs from
    `Hothan/OlympiadBench` (the canonical mirror, confirmed identical to
    lscpku/OlympiadBench-official):
        OE_TO_maths_en_COMP — 674 rows, open-ended
        TP_TO_maths_en_COMP — 503 rows, theorem-proving
    giving 1177 total rows, ALL written to olympiad_eval.jsonl (use eval.py's
    --n flag to take a subset). Theorem-proving rows have `final_answer=None`
    (proofs have no short answer) — harmless here since the pipeline never
    grades `answer`/`solution`, only uses `question` as the eval prompt
    (block efficiency, not answer correctness). Both fields `question` (str)
    and `final_answer` (list or None) confirmed exact for both configs.
    """
    eval_path = os.path.join(DATA_DIR, "olympiad_eval.jsonl")

    if os.path.isfile(eval_path) and not force:
        print("  olympiad_eval.jsonl already present — skipping")
        return eval_path

    _PROMPT_FIELDS = ("question", "problem", "prompt")
    _ANSWER_FIELDS = ("final_answer", "answer", "solution")

    def _row_to_item(row):
        prompt = None
        for f in _PROMPT_FIELDS:
            v = row.get(f)
            if isinstance(v, str) and v:
                prompt = v
                break
        if prompt is None:
            return None
        answer = ""
        for f in _ANSWER_FIELDS:
            v = row.get(f)
            if isinstance(v, list) and v:
                answer = v[0]
                break
            if isinstance(v, str) and v:
                answer = v
                break
        return {"prompt": prompt, "answer": answer,
                "subject": row.get("subject", ""), "source": "olympiadbench"}

    print("  fetching Hothan/OlympiadBench (OE_TO_maths_en_COMP + TP_TO_maths_en_COMP) ...")
    try:
        from datasets import load_dataset
        all_items = []
        for cfg in ("OE_TO_maths_en_COMP", "TP_TO_maths_en_COMP"):
            ds = load_dataset("Hothan/OlympiadBench", cfg, split="train")
            for row in ds:
                item = _row_to_item(row)
                if item:
                    all_items.append(item)
        if all_items:
            random.Random(42).shuffle(all_items)
            save_jsonl(eval_path, all_items)
            return eval_path
        print("  Hothan/OlympiadBench returned 0 usable rows — falling back to canned placeholders")
    except Exception as e:
        print(f"  Hothan/OlympiadBench failed ({e}) — falling back to canned placeholders")

    # Canned fallback — same degrade-soft pattern as fetch_hf(); lets the
    # pipeline run end-to-end even if the HF schema drifted or is offline.
    save_jsonl(eval_path, [{"prompt": f"[olympiadbench fallback] placeholder problem {i}.",
                            "answer": "", "subject": "", "source": "canned_fallback"}
                           for i in range(300)])
    return eval_path


def fetch_dapo_math(n: int = 17000, force: bool = False):
    """
    Build dapo_math_train.jsonl (TRAINING-ONLY, no val/eval split) from
    BytedTsinghua-SIA/DAPO-Math-17k — a larger alternative to math_hard.jsonl
    (~5,332 rows), raised by Rahul (2026-07-16 WhatsApp thread) as a way to
    reduce same-prompt repetition over long training runs.

    SCHEMA CAVEAT (checked live 2026-07-17, same discipline as
    fetch_olympiad_eval's schema note — don't trust a dataset's field names
    from memory): the HF `train` split reports num_examples=1,791,700 rows,
    NOT ~17,000 as the "17k" name implies. First 5,000 streamed rows all had
    unique `extra_info.index` UUIDs; full-dataset uniqueness was NOT verified
    (would require streaming and hashing all 1.8M rows). Treat "17k" as a
    legacy/marketing name, not a verified row count — this fetcher takes only
    the first `n` (default 17000, matching the namesake) rather than assuming
    the whole file is small or fully de-duplicated.

    Row schema (confirmed live):
        data_source   : str, e.g. "math_dapo"
        prompt        : list[{"role": "user", "content": <full instruction-
                         formatted problem text, already asks the model to
                         answer in "Answer: $X" form>}] — single-turn, use
                         prompt[0]["content"] directly as our "prompt" field
        ability       : str, e.g. "MATH"
        reward_model  : {"ground_truth": <short answer str>, "style": ...}
        extra_info    : {"index": <uuid>}

    This is a TRAINING pool only — no official val/test split shipped with
    the dataset. Deliberately NOT wired into math_val/math_eval: keep those
    untouched so existing results stay comparable. Use via
    --train_dataset dapo_math_train once opted in (not fetched by default;
    pass --datasets dapo_math_train explicitly).
    """
    path = os.path.join(DATA_DIR, "dapo_math_train.jsonl")
    if os.path.isfile(path) and not force:
        print("  dapo_math_train.jsonl already present — skipping")
        return path

    print(f"  fetching BytedTsinghua-SIA/DAPO-Math-17k (streaming, first {n} rows) ...")
    try:
        from datasets import load_dataset
        ds = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", split="train", streaming=True)
        items = []
        for row in ds:
            msgs = row.get("prompt") or []
            prompt = msgs[0]["content"] if msgs and "content" in msgs[0] else None
            if not prompt:
                continue
            answer = (row.get("reward_model") or {}).get("ground_truth", "")
            items.append({"prompt": prompt, "answer": answer,
                          "source": row.get("data_source", "")})
            if len(items) >= n:
                break
        save_jsonl(path, items)
    except Exception as e:
        print(f"  WARNING: DAPO-Math-17k download failed: {e}")
        save_jsonl(path, [{"prompt": f"[dapo fallback] placeholder problem {i}.",
                            "answer": "", "source": "canned_fallback"}
                           for i in range(300)])
    return path


def fetch_all(n_eval: int = 100, train: bool = False, datasets: list = None,
              force: bool = False) -> None:
    """Top-level entry point — fetch every dataset we evaluate on."""
    requested = set(datasets) if datasets else set(EVAL_DATASETS) | {"gsm8k", "spec_bench"}

    if train:
        fetch_gsm8k_train(force=force)
        if not datasets:
            requested |= {"math_hard", "math_val"}

    if "gsm8k" in requested:
        fetch_gsm8k_val_and_eval(n_eval=n_eval, force=force)

    if "math_hard" in requested or "math_val" in requested or "math_eval" in requested:
        fetch_math_hard_and_val(force=force)

    if "olympiad_eval" in requested:
        fetch_olympiad_eval(force=force)

    if "dapo_math_train" in requested:
        fetch_dapo_math(n=n_eval if n_eval != 100 else 17000, force=force)

    if "spec_bench" in requested:
        fetch_spec_bench(force=force)

    for name, (hf_id, split, field, config, ans) in EVAL_DATASETS.items():
        if name not in requested:
            continue
        fetch_hf(hf_id, split, field, name, n=n_eval, config=config,
                 extra_field=ans, force=force)


def get_path(name: str) -> str:
    """Resolve a dataset name to its on-disk JSONL path (must be downloaded)."""
    known = ({"gsm8k_train", "gsm8k_val", "gsm8k_eval",
              "math_hard", "math_val", "math_eval",
              "olympiad_eval", "dapo_math_train",
              "spec_bench"} | set(EVAL_DATASETS))
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
