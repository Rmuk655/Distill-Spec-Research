#!/usr/bin/env python
"""
plot_prob_vs_jsd_advantage.py — chart of "prob's advantage over jsd" from the
gap-closed metric: for each dataset's BEST prob checkpoint and BEST jsd
checkpoint (by mean %-gap-closed, same selection as plot_gap_closed.py),
plots % of student->teacher gap closed at k=2/4/8/16 side by side, plus the
prob-minus-jsd advantage in percentage points.

Two panels: (1) grouped bars, prob vs jsd, one group per (dataset, k) --
the direct visual comparison; (2) prob's advantage in pp, grouped by k,
one bar group per dataset -- the "how much better, at each k" table as a chart.

USAGE:
    python scripts/plot_prob_vs_jsd_advantage.py
"""
import csv
import os
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

K_OF_INTEREST = [2, 4, 8, 16]
DATASETS = ["math_eval.jsonl", "gsm8k_eval.jsonl", "olympiad_eval.jsonl"]
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


def best_checkpoint(rows, ds, student, gap, family_prefix):
    """Best (highest mean %-gap-closed) checkpoint whose run_name starts with
    family_prefix ('jsd' or 'po_prob'/'po_')."""
    names = sorted(set(run_name(r["model"]) for r in rows if r["dataset"] == ds))
    best_name, best_mean, best_pcts = None, -1e9, None
    for name in names:
        if family_prefix == "jsd" and not name.startswith("jsd"):
            continue
        if family_prefix == "prob" and not name.startswith("po_"):
            continue
        pcts = {}
        for k in K_OF_INTEREST:
            v = passk_at_frag(rows, name, ds, k)
            if v is None or student.get(k) is None or gap.get(k) in (None, 0):
                continue
            pcts[k] = 100 * (v - student[k]) / gap[k]
        if len(pcts) < len(K_OF_INTEREST):
            continue
        mean_pct = np.mean(list(pcts.values()))
        if mean_pct > best_mean:
            best_name, best_mean, best_pcts = name, mean_pct, pcts
    return best_name, best_pcts


def main():
    rows = load_csv("results/passK/passk_hparam_sweep.csv")
    base = load_csv("results/passK/passk_baselines.csv")
    outdir = "results/passK"
    os.makedirs(outdir, exist_ok=True)

    per_ds = {}
    for ds in DATASETS:
        student = {k: passk_at_exact(base, "Qwen/Qwen3-0.6B", ds, k) for k in K_OF_INTEREST}
        teacher = {k: passk_at_exact(base, "Qwen/Qwen3-8B", ds, k) for k in K_OF_INTEREST}
        gap = {k: (teacher[k] - student[k]) if teacher[k] is not None and student[k] is not None else None
               for k in K_OF_INTEREST}
        jsd_name, jsd_pcts = best_checkpoint(rows, ds, student, gap, "jsd")
        prob_name, prob_pcts = best_checkpoint(rows, ds, student, gap, "prob")
        per_ds[ds] = (jsd_name, jsd_pcts, prob_name, prob_pcts)
        print(f"{ds}: best jsd={jsd_name} {jsd_pcts}, best prob={prob_name} {prob_pcts}")

    tags = [d.replace(".jsonl", "") for d in DATASETS]

    # -------- panel 1: prob vs jsd side by side, per dataset per k --------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=True)
    for ax, ds, tag in zip(axes, DATASETS, tags):
        jsd_name, jsd_pcts, prob_name, prob_pcts = per_ds[ds]
        x = np.arange(len(K_OF_INTEREST))
        width = 0.35
        ax.bar(x - width / 2, [jsd_pcts[k] for k in K_OF_INTEREST], width, label=f"jsd ({jsd_name})", color="#2a78d6")
        ax.bar(x + width / 2, [prob_pcts[k] for k in K_OF_INTEREST], width, label=f"prob ({prob_name})", color="#e34948")
        ax.set_xticks(x)
        ax.set_xticklabels([f"k{k}" for k in K_OF_INTEREST])
        ax.set_title(tag, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=7, loc="upper left")
    axes[0].set_ylabel("% of student->teacher gap closed")
    fig.suptitle("Best jsd vs best prob checkpoint, by k (% of gap closed)")
    fig.tight_layout()
    path1 = os.path.join(outdir, "prob_vs_jsd_gap_closed.png")
    fig.savefig(path1, dpi=150)
    plt.close(fig)
    print(f"wrote {path1}")

    # -------- panel 2: prob's advantage in pp, grouped by k, per dataset --------
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(tags))
    width = 0.2
    colors = ["#2a78d6", "#008300", "#e34948", "#eda100"]
    for i, k in enumerate(K_OF_INTEREST):
        advs = [per_ds[ds][3][k] - per_ds[ds][1][k] for ds in DATASETS]
        bars = ax.bar(x + (i - 1.5) * width, advs, width, label=f"k{k}", color=colors[i])
        for b, v in zip(bars, advs):
            ax.annotate(f"+{v:.1f}", (b.get_x() + b.get_width() / 2, v), textcoords="offset points",
                        xytext=(0, 3), ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(tags)
    ax.set_ylabel("prob advantage over jsd (percentage points of gap closed)")
    ax.set_title("prob's advantage over jsd, by k and dataset\n(best-of-family checkpoints, mean-%-gap-closed selection)", fontsize=10)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path2 = os.path.join(outdir, "prob_advantage_over_jsd.png")
    fig.savefig(path2, dpi=150)
    plt.close(fig)
    print(f"wrote {path2}")


if __name__ == "__main__":
    main()
