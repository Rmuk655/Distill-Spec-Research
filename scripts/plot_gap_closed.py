#!/usr/bin/env python
"""
plot_gap_closed.py — "who is lifting pass@k closest to the teacher, more than
others?" Two panels per dataset:

  1. Raw delta vs student (the numbers from analyze_passk_movement.py's §1
     significant-checkpoints table), grouped bars per checkpoint.
  2. The SAME deltas expressed as % of the student->teacher gap closed --
     this is the actual "closer to teacher" ranking, since a raw delta of
     0.06 means very different things depending on how big the gap was to
     start with (see the earlier gap-closed analysis: gsm8k k=1 closed 153%
     of a 0.14 gap, math_eval k=64 closed 0% of a 0.06 gap).

Checkpoints are sorted by mean %-gap-closed across k2/k4/k8/k16, descending,
so the best "lifter" is always leftmost.

USAGE:
    python scripts/plot_gap_closed.py --dataset olympiad_eval.jsonl
    python scripts/plot_gap_closed.py --dataset math_eval.jsonl
    # override with an explicit list instead of auto-deriving:
    python scripts/plot_gap_closed.py --dataset math_eval.jsonl --checkpoints po_prob_lr3e-6_wu20,po_prob_lr7e-6_wu20

CHECKPOINT SELECTION: by default, auto-derives the "significant" set the same
way analyze_passk_movement.py's §1 does -- every 0.6B/8B checkpoint in the CSV
whose delta-vs-student clears the derived per-k noise floor (2 x std over the
flat low-LR prefix_overlap/prob cluster) at ANY of k2/k4/k8/k16. This means the
significant set is genuinely different per dataset (math_eval/gsm8k_eval/
olympiad_eval each have their own floor and their own noise), not a hardcoded
list copied from one dataset's report. Pass --checkpoints to override.
"""
import argparse
import csv
import os
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

K_OF_INTEREST = [2, 4, 8, 16]

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


def passk_at_frag(rows, frag, ds, k):
    for r in rows:
        if frag in r["model"] and r["dataset"] == ds and int(r["k"]) == k:
            return float(r["pass_at_k"])
    return None


def passk_at_exact(rows, name, ds, k):
    for r in rows:
        if r["model"] == name and r["dataset"] == ds and int(r["k"]) == k:
            return float(r["pass_at_k"])
    return None


def derive_noise_floor(rows, ds):
    floor = {}
    for k in K_OF_INTEREST:
        vals = [v for v in (passk_at_frag(rows, f, ds, k) for f in NOISE_CLUSTER_PROB_06B_8B) if v is not None]
        floor[k] = 2 * statistics.stdev(vals) if len(vals) >= 3 else None
    return floor


def auto_significant_checkpoints(rows, ds, student, floor):
    """Same rule as analyze_passk_movement.py's §1: every checkpoint with at
    least one |delta vs student| clearing that k's derived noise floor."""
    all_names = sorted(set(run_name(r["model"]) for r in rows))
    significant = []
    for name in all_names:
        for k in K_OF_INTEREST:
            v = passk_at_frag(rows, name, ds, k)
            if v is None or student.get(k) is None or floor.get(k) is None:
                continue
            if abs(v - student[k]) >= floor[k]:
                significant.append(name)
                break
    return significant


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/passK/passk_hparam_sweep.csv")
    ap.add_argument("--baselines_csv", default="results/passK/passk_baselines.csv")
    ap.add_argument("--dataset", default="olympiad_eval.jsonl")
    ap.add_argument("--checkpoints", default=None,
                    help="comma list to override; default auto-derives the "
                         "noise-floor-significant set for this dataset")
    ap.add_argument("--outdir", default="results/passK")
    args = ap.parse_args()

    rows = load_csv(args.csv)
    base = load_csv(args.baselines_csv)
    ds = args.dataset
    tag = ds.replace(".jsonl", "")

    student = {k: passk_at_exact(base, "Qwen/Qwen3-0.6B", ds, k) for k in K_OF_INTEREST}
    teacher = {k: passk_at_exact(base, "Qwen/Qwen3-8B", ds, k) for k in K_OF_INTEREST}
    gap = {k: (teacher[k] - student[k]) if teacher[k] is not None and student[k] is not None else None
           for k in K_OF_INTEREST}

    if args.checkpoints:
        checkpoints = [c.strip() for c in args.checkpoints.split(",")]
    else:
        floor = derive_noise_floor(rows, ds)
        checkpoints = auto_significant_checkpoints(rows, ds, student, floor)
        print(f"[auto] {len(checkpoints)} checkpoint(s) clear the {tag} noise floor: {', '.join(checkpoints)}")

    data = []  # (name, {k: delta}, {k: pct_closed}, mean_pct)
    for ck in checkpoints:
        deltas, pcts = {}, {}
        for k in K_OF_INTEREST:
            v = passk_at_frag(rows, ck, ds, k)
            if v is None or student[k] is None:
                continue
            d = v - student[k]
            deltas[k] = d
            pcts[k] = 100 * d / gap[k] if gap[k] else None
        if not deltas:
            continue
        mean_pct = np.mean([p for p in pcts.values() if p is not None])
        data.append((ck, deltas, pcts, mean_pct))
    data.sort(key=lambda t: -t[3])

    names = [d[0] for d in data]
    colors = ["#2a78d6", "#008300", "#e34948", "#eda100"]
    x = np.arange(len(names))
    width = 0.2

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(11, len(names) * 0.7), 11))

    for i, k in enumerate(K_OF_INTEREST):
        vals = [d[1].get(k, 0) for d in data]
        ax1.bar(x + (i - 1.5) * width, vals, width, label=f"Δk{k}", color=colors[i])
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("delta pass@k vs student (raw)")
    ax1.set_title(f"Raw lift vs untrained student ({tag})")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)
    ax1.axhline(0, color="black", linewidth=0.8)

    for i, k in enumerate(K_OF_INTEREST):
        vals = [d[2].get(k, 0) for d in data]
        ax2.bar(x + (i - 1.5) * width, vals, width, label=f"k{k}", color=colors[i])
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("% of student->teacher gap closed")
    ax2.set_title(f"Same lift, normalized by gap to teacher ({tag})\n"
                  "'closer to teacher' ranking, sorted descending by mean %", fontsize=10)
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.axhline(100, color="gray", linestyle="--", linewidth=1, label="teacher level")

    fig.tight_layout()
    os.makedirs(args.outdir, exist_ok=True)
    path = os.path.join(args.outdir, f"gap_closed_{tag}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")
    print("\nranked by mean % of gap closed (k2/k4/k8/k16 avg):")
    for name, deltas, pcts, mean_pct in data:
        print(f"  {name}: mean={mean_pct:.1f}%  " +
              ", ".join(f"k{k}={pcts.get(k, 0):.0f}%" for k in K_OF_INTEREST))


if __name__ == "__main__":
    main()
