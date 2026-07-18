#!/usr/bin/env python
"""
top_checkpoints_by_passk.py — direct answer to "which checkpoint has the best
pass@k for each k value": for every k in the sweep CSV, ranks all swept
checkpoints by RAW pass@k (not delta vs student) and prints the top N.

Separate table per dataset (math_eval/gsm8k_eval/olympiad_eval) -- never
mixed, same reasoning as analyze_passk_movement.py: each dataset has a very
different ceiling (gsm8k saturates, olympiad is low-ceiling), so a combined
ranking across datasets would just reflect dataset difficulty, not which
checkpoint is actually best.

USAGE:
    python scripts/top_checkpoints_by_passk.py
    python scripts/top_checkpoints_by_passk.py --top 5 --datasets math_eval.jsonl
"""
import argparse
import csv
from collections import defaultdict


def load_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run_name(model_path):
    # .../<sweep_root>/<run_name>/ckpt_best -> <run_name>
    parts = model_path.rstrip("/").split("/")
    return parts[-2] if len(parts) >= 2 else model_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/passK/passk_hparam_sweep.csv")
    ap.add_argument("--datasets", default="math_eval.jsonl,gsm8k_eval.jsonl,olympiad_eval.jsonl")
    ap.add_argument("--ks", default="1,2,4,8,16,32,64")
    ap.add_argument("--top", type=int, default=3)
    args = ap.parse_args()

    rows = load_csv(args.csv)
    ks = [int(k) for k in args.ks.split(",")]
    datasets = [d.strip() for d in args.datasets.split(",")]

    # (dataset, k) -> list of (pass_at_k, run_name)
    by_dk = defaultdict(list)
    for r in rows:
        ds = r["dataset"]
        k = int(r["k"])
        if ds not in datasets or k not in ks:
            continue
        by_dk[(ds, k)].append((float(r["pass_at_k"]), run_name(r["model"])))

    for ds in datasets:
        print(f"\n=== {ds} ===")
        for k in ks:
            entries = sorted(by_dk.get((ds, k), []), reverse=True)[: args.top]
            if not entries:
                print(f"  k={k}: no data")
                continue
            ranked = ", ".join(f"{name} ({val:.4f})" for val, name in entries)
            print(f"  k={k:>2}: {ranked}")


if __name__ == "__main__":
    main()
