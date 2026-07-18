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
    python scripts/plot_gap_closed.py --dataset math_eval.jsonl --checkpoints po_prob_lr3e-6_wu20,po_prob_lr7e-6_wu20
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

K_OF_INTEREST = [2, 4, 8, 16]

# significant (above-noise-floor) checkpoints from analyze_passk_movement.py's
# olympiad_eval report -- override with --checkpoints for another dataset/list.
DEFAULT_CHECKPOINTS = [
    "po_prob_lr1e-5_wu5_lrmin0.1_wd0.01", "po_prob_lr1e-5_wu20_wd0.0001",
    "po_prob_topk50_lr1e-5_wu20", "po_prob_lr1e-5_wu20_wd0.001",
    "po_prob_lr1e-5_wu10_lrmin0.1_wd0.01", "po_prob_topk20_lr1e-5_wu20",
    "po_prob_lr7e-6_wu20", "po_prob_ceanneal3000_auxw2.0_lr1e-5_wu20",
    "po_prob_lr3e-6_wu20", "po_prob_ceanneal1500_lr1e-5_wu20",
    "po_prob_lr1e-5_wu20_lrmin0.1_wd0.01", "jsd_lr3e-6_wu10_lrmin0.1_wd0.01",
    "po_prob_ceanneal_lr1e-5_wu20",
]


def load_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/passK/passk_hparam_sweep.csv")
    ap.add_argument("--baselines_csv", default="results/passK/passk_baselines.csv")
    ap.add_argument("--dataset", default="olympiad_eval.jsonl")
    ap.add_argument("--checkpoints", default=",".join(DEFAULT_CHECKPOINTS))
    ap.add_argument("--outdir", default="results/passK")
    args = ap.parse_args()

    rows = load_csv(args.csv)
    base = load_csv(args.baselines_csv)
    checkpoints = [c.strip() for c in args.checkpoints.split(",")]
    ds = args.dataset
    tag = ds.replace(".jsonl", "")

    student = {k: passk_at_exact(base, "Qwen/Qwen3-0.6B", ds, k) for k in K_OF_INTEREST}
    teacher = {k: passk_at_exact(base, "Qwen/Qwen3-8B", ds, k) for k in K_OF_INTEREST}
    gap = {k: (teacher[k] - student[k]) if teacher[k] is not None and student[k] is not None else None
           for k in K_OF_INTEREST}

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
