#!/usr/bin/env python
"""
plot_ablation_families.py -- one grouped-bar chart per ablation family in
notes/passk_prob_vs_jsd_research_note.md's "Ablations tested" table. Each
chart: x = variant, grouped bars = Delta(variant - jsd) at k16/k32/k64 on
gsm8k_eval (the dataset where any of these ever beats jsd beyond noise),
with the derived noise-floor band shaded so bars inside it read as
not-significant.

Noise floor = 2 x std(pass@k) across the flat low-LR NOISE_CLUSTER_PROB
checkpoints, same method and same cluster as analyze_passk_movement.py,
just extended to k=16/32/64 here (that script only covers k up to 16).

USAGE:
    python scripts/plot_ablation_families.py
"""
import csv
import os
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATASET = "gsm8k_eval.jsonl"
KS = [16, 32, 64]

NOISE_CLUSTER_PROB = [
    "po_prob_lr1e-6_wu20", "po_prob_lr3e-6_wu20", "po_prob_lr7e-6_wu20",
    "po_prob_lr1e-5_wu20_lrmin0.1_wd0.01",
]
JSD_LR1E5 = "jsd_lr1e-5_wu10_lrmin0.1_wd0.01"
JSD_LR3E6 = "jsd_lr3e-6_wu10_lrmin0.1_wd0.01"

# (family title, jsd baseline for this family, [(variant label, checkpoint frag), ...])
FAMILIES = [
    ("grad_clip", JSD_LR1E5, [
        ("clip=10", "po_prob_gradclip10_lr1e-5_wu20"),
        ("clip=100", "po_prob_gradclip100_lr1e-5_wu20"),
        ("clip=1000", "po_prob_gradclip1000_lr1e-5_wu20"),
    ]),
    ("M_samples_per_root", JSD_LR3E6, [
        ("M=4 (default)", "po_prob_lr3e-6_wu20"),
        ("M=8", "po_prob_M8_lr3e-6_wu20"),
        ("M=16 (warm)", "po_prob_M16_lr3e-6_wu20"),
        ("M=16 (cold)", "po_prob_M16_lr3e-6_wu20_cold"),
    ]),
    ("teacher_temp", JSD_LR1E5, [
        ("ttemp=1.0 (default)", "po_prob_lr1e-5_wu20_lrmin0.1_wd0.01"),
        ("ttemp=0.7", "po_prob_ttemp0.7_lr1e-5_wu20"),
        ("ttemp=0.5", "po_prob_ttemp0.5_lr1e-5_wu20"),
    ]),
    ("CE_anneal_timing_lr3e-6", JSD_LR3E6, [
        ("no anneal", "po_prob_lr3e-6_wu20"),
        ("ceanneal3000_auxw0.5", "po_prob_ceanneal3000_auxw0.5_lr3e-6_wu20"),
    ]),
    ("root_spacing_offset", JSD_LR1E5, [
        ("N=16 fixed", "po_prob_multiroot_N16_lr1e-5_wu20"),
        ("N=16 random offset", "po_prob_multiroot_N16_randoff_lr1e-5_wu20"),
    ]),
    ("N_root_spacing", JSD_LR1E5, [
        ("N=8", "po_prob_multiroot_N8_lr1e-5_wu20"),
        ("N=16", "po_prob_multiroot_N16_lr1e-5_wu20"),
        ("N=16 random offset", "po_prob_multiroot_N16_randoff_lr1e-5_wu20"),
        ("N=32", "po_prob_multiroot_N32_lr1e-5_wu20"),
    ]),
]


def load_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def passk_at(rows, frag, dataset, k):
    for r in rows:
        if frag in r["model"] and r["dataset"] == dataset and int(r["k"]) == k:
            return float(r["pass_at_k"])
    return None


def derive_noise_floor(rows, dataset, ks):
    floor = {}
    for k in ks:
        vals = [v for v in (passk_at(rows, f, dataset, k) for f in NOISE_CLUSTER_PROB) if v is not None]
        floor[k] = round(2 * statistics.stdev(vals), 4) if len(vals) >= 3 else 0.05
    return floor


def main():
    rows = load_csv("results/passK/passk_hparam_sweep.csv")
    outdir = "results/passK"
    os.makedirs(outdir, exist_ok=True)
    floor = derive_noise_floor(rows, DATASET, KS)
    print(f"noise floor on {DATASET} at k={KS}: {floor}")

    for title, jsd_frag, variants in FAMILIES:
        labels, deltas = [], {k: [] for k in KS}
        for label, frag in variants:
            for k in KS:
                v = passk_at(rows, frag, DATASET, k)
                j = passk_at(rows, jsd_frag, DATASET, k)
                deltas[k].append(None if (v is None or j is None) else v - j)
            labels.append(label)

        fig, ax = plt.subplots(figsize=(7, 4.5))
        x = range(len(labels))
        width = 0.25
        colors = {16: "#a3c2f4", 32: "#5b8fd6", 64: "#0b3d91"}
        for i, k in enumerate(KS):
            xs = [xi + (i - 1) * width for xi in x]
            ys = [d if d is not None else 0 for d in deltas[k]]
            bars = ax.bar(xs, ys, width=width, color=colors[k], label=f"k={k}")
            for b, d in zip(bars, deltas[k]):
                if d is not None and abs(d) <= floor[k]:
                    b.set_hatch("//")
                    b.set_alpha(0.5)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
        ax.set_ylabel(f"Delta pass@k vs jsd ({jsd_frag})", fontsize=8)
        ax.set_title(f"{title} -- gsm8k_eval, vs jsd\n(hatched/faded bars are inside the noise floor)", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        path = os.path.join(outdir, f"ablation_{title}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
