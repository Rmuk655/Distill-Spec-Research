#!/usr/bin/env python
"""
top_checkpoints_by_passk.py — "does ANY training config give a real pass@k
improvement over the untrained student, and which loss/param wins?"

Pulls from BOTH results/passK/passk_hparam_sweep.csv (this LR/warmup/CE-anneal
sweep, 0.6B/8B) AND results/passK/passk_sweep_checkpoints.csv (older
enrich/tree-loss/DDTE-era runs, some 0.6B/8B and some 1.7B/32B) -- these are
just different trained models evaluated the same way (same pass_at_k formula,
same n/n_prompts), so there's no principled reason to hide one set. Each row
is tagged with its model pair (0.6B/8B vs 1.7B/32B) so the SIZE CONFOUND is
visible instead of silently mixed in -- a bigger draft (1.7B) has more raw
capacity than 0.6B independent of training, so a 1.7B checkpoint beating a
0.6B one isn't evidence about loss/param choice by itself.

NOISE FLOOR: derived (2 x std(pass@k)) from the 0.6B/8B prefix_overlap/prob
flat low-LR cluster (see analyze_passk_movement.py for the same method +
rationale). This floor is ONLY valid for 0.6B/8B rows -- there is no repeated-
config cluster for the 1.7B/32B pair in the data, so 1.7B/32B deltas are
reported but flagged as "floor unknown", not silently passed/failed against
a floor computed from a different-sized model.

Reports, per dataset per k:
  1. Top-N checkpoints by raw pass@k, tagged with model pair and loss family,
     with the gap to the next-ranked checkpoint marked ABOVE/WITHIN noise.
  2. Delta vs the untrained 0.6B student baseline for the same top-N, marked
     ABOVE/WITHIN noise (0.6B/8B rows only) or UNKNOWN (1.7B/32B rows) --
     this directly answers "does training even help, verified".

USAGE:
    python scripts/top_checkpoints_by_passk.py
    python scripts/top_checkpoints_by_passk.py --top 5 --datasets math_eval.jsonl
"""
import argparse
import csv
import statistics
from collections import defaultdict

NOISE_CLUSTER_PROB_06B_8B = [
    "po_prob_lr1e-6_wu20", "po_prob_lr3e-6_wu20", "po_prob_lr7e-6_wu20",
    "po_prob_lr1e-5_wu20_lrmin0.1_wd0.01",
]


def load_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run_name(model_path):
    parts = model_path.rstrip("/").split("/")
    return parts[-2] if len(parts) >= 2 else model_path


def model_pair(model_path):
    return "1.7B/32B" if "Qwen32B-Qwen1.7B" in model_path else "0.6B/8B"


def loss_family(name):
    if name.startswith("jsd"):
        return "jsd"
    if name.startswith("po_") or "prefix_overlap" in name or name.startswith("enrich"):
        if "enrich" in name:
            return "enrich"
        return "prefix_overlap"
    return "other"


def passk_at(rows, frag, ds, k):
    """Match by substring in the model PATH -- for checkpoint rows (noise cluster)."""
    for r in rows:
        if frag in r["model"] and r["dataset"] == ds and int(r["k"]) == k:
            return float(r["pass_at_k"])
    return None


def passk_at_exact(rows, model_name, ds, k):
    """Match by exact model field -- for baseline rows ('Qwen/Qwen3-0.6B'), which
    aren't checkpoint paths and would be mangled by run_name()'s path-parsing."""
    for r in rows:
        if r["model"] == model_name and r["dataset"] == ds and int(r["k"]) == k:
            return float(r["pass_at_k"])
    return None


def derive_noise_floor(rows, ds, ks):
    floor = {}
    for k in ks:
        vals = [v for v in (passk_at(rows, f, ds, k) for f in NOISE_CLUSTER_PROB_06B_8B) if v is not None]
        floor[k] = round(2 * statistics.stdev(vals), 4) if len(vals) >= 3 else None
    return floor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/passK/passk_hparam_sweep.csv")
    ap.add_argument("--legacy_csv", default="results/passK/passk_sweep_checkpoints.csv")
    ap.add_argument("--baselines_csv", default="results/passK/passk_baselines.csv")
    ap.add_argument("--datasets", default="math_eval.jsonl,gsm8k_eval.jsonl,olympiad_eval.jsonl")
    ap.add_argument("--ks", default="1,2,4,8,16,32,64")
    ap.add_argument("--top", type=int, default=3)
    args = ap.parse_args()

    rows = load_csv(args.csv)
    try:
        rows += load_csv(args.legacy_csv)
    except FileNotFoundError:
        print(f"[warn] {args.legacy_csv} not found -- reporting sweep CSV only")
    try:
        base_rows = load_csv(args.baselines_csv)
    except FileNotFoundError:
        base_rows = []
        print(f"[warn] {args.baselines_csv} not found -- no student/teacher baseline comparison")

    ks = [int(k) for k in args.ks.split(",")]
    datasets = [d.strip() for d in args.datasets.split(",")]

    by_dk = defaultdict(list)
    for r in rows:
        ds, k = r["dataset"], int(r["k"])
        if ds not in datasets or k not in ks:
            continue
        name = run_name(r["model"])
        by_dk[(ds, k)].append((float(r["pass_at_k"]), name, model_pair(r["model"]), loss_family(name)))

    for ds in datasets:
        print(f"\n=== {ds} ===")
        floor = derive_noise_floor(rows, ds, ks)
        student_06b = {k: passk_at_exact(base_rows, "Qwen/Qwen3-0.6B", ds, k) for k in ks} if base_rows else {}
        student_17b = {k: passk_at_exact(base_rows, "Qwen/Qwen3-1.7B", ds, k) for k in ks} if base_rows else {}
        for k in ks:
            entries = sorted(by_dk.get((ds, k), []), reverse=True)
            if not entries:
                print(f"  k={k}: no data")
                continue
            top = entries[: args.top]
            f = floor.get(k)
            s06, s17 = student_06b.get(k), student_17b.get(k)
            print(f"  k={k:>2}  (noise floor for 0.6B/8B prob: {f if f is not None else 'n/a'}"
                  f"{f', 0.6B student baseline: {s06:.4f}' if s06 is not None else ''}"
                  f"{f', 1.7B student baseline: {s17:.4f}' if s17 is not None else ''})")
            for i, (val, name, pair, fam) in enumerate(top):
                gap_next = val - top[i + 1][0] if i + 1 < len(top) else None
                gap_tag = ""
                if gap_next is not None:
                    if pair == "0.6B/8B" and f is not None:
                        gap_tag = f" | gap to next: {gap_next:.4f} ({'ABOVE' if gap_next > f else 'within'} floor)"
                    else:
                        gap_tag = f" | gap to next: {gap_next:.4f} (floor unknown for {pair})"
                own_student = s06 if pair == "0.6B/8B" else s17
                delta_tag = ""
                if own_student is not None:
                    delta = val - own_student
                    if pair == "0.6B/8B" and f is not None:
                        delta_tag = f" | delta vs own-size student: {delta:+.4f} ({'ABOVE' if abs(delta) > f else 'within'} floor)"
                    else:
                        delta_tag = f" | delta vs own-size student: {delta:+.4f} (floor unknown for {pair})"
                print(f"    #{i+1} {name} [{pair}, {fam}] = {val:.4f}{gap_tag}{delta_tag}")


if __name__ == "__main__":
    main()
