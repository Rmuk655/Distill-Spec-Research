#!/usr/bin/env python
"""
plot_ablation_families.py -- one pass@k-vs-k line chart per ablation family in
notes/passk_prob_vs_jsd_research_note.md's "Ablations tested" table. Same
style as plot_passk_curves_all.py (x=k log scale, student/teacher as bold
dashed reference lines) but zoomed in to only that family's variants plus its
jsd baseline, instead of all 38 checkpoints.

USAGE:
    python scripts/plot_ablation_families.py
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATASET = "gsm8k_eval.jsonl"
ALL_K = [1, 2, 4, 8, 16, 32, 64]

JSD_LR1E5 = "jsd_lr1e-5_wu10_lrmin0.1_wd0.01"
JSD_LR3E6 = "jsd_lr3e-6_wu10_lrmin0.1_wd0.01"

COLORS = ["#b30000", "#e08214", "#1a9850", "#762a83"]

# (family title, jsd baseline label+frag, [(variant label, checkpoint frag), ...])
FAMILIES = [
    ("grad_clip", ("jsd", JSD_LR1E5), [
        ("clip=10", "po_prob_gradclip10_lr1e-5_wu20"),
        ("clip=100", "po_prob_gradclip100_lr1e-5_wu20"),
        ("clip=1000", "po_prob_gradclip1000_lr1e-5_wu20"),
    ]),
    ("M_samples_per_root", ("jsd", JSD_LR3E6), [
        ("M=4 (default)", "po_prob_lr3e-6_wu20"),
        ("M=8", "po_prob_M8_lr3e-6_wu20"),
        ("M=16 (warm)", "po_prob_M16_lr3e-6_wu20"),
        ("M=16 (cold)", "po_prob_M16_lr3e-6_wu20_cold"),
    ]),
    ("teacher_temp", ("jsd", JSD_LR1E5), [
        ("ttemp=1.0 (default)", "po_prob_lr1e-5_wu20_lrmin0.1_wd0.01"),
        ("ttemp=0.7", "po_prob_ttemp0.7_lr1e-5_wu20"),
        ("ttemp=0.5", "po_prob_ttemp0.5_lr1e-5_wu20"),
    ]),
    ("CE_anneal_timing_lr3e-6", ("jsd", JSD_LR3E6), [
        ("no anneal", "po_prob_lr3e-6_wu20"),
        ("ceanneal3000_auxw0.5", "po_prob_ceanneal3000_auxw0.5_lr3e-6_wu20"),
    ]),
    ("root_spacing_offset", ("jsd", JSD_LR1E5), [
        ("N=16 fixed", "po_prob_multiroot_N16_lr1e-5_wu20"),
        ("N=16 random offset", "po_prob_multiroot_N16_randoff_lr1e-5_wu20"),
    ]),
    ("N_root_spacing", ("jsd", JSD_LR1E5), [
        ("N=8", "po_prob_multiroot_N8_lr1e-5_wu20"),
        ("N=16", "po_prob_multiroot_N16_lr1e-5_wu20"),
        ("N=16 random offset", "po_prob_multiroot_N16_randoff_lr1e-5_wu20"),
        ("N=32", "po_prob_multiroot_N32_lr1e-5_wu20"),
    ]),
]


def load_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def passk_at_frag(rows, frag, dataset, k):
    for r in rows:
        if frag in r["model"] and r["dataset"] == dataset and int(r["k"]) == k:
            return float(r["pass_at_k"])
    return None


def passk_at_exact(rows, name, dataset, k):
    for r in rows:
        if r["model"] == name and r["dataset"] == dataset and int(r["k"]) == k:
            return float(r["pass_at_k"])
    return None


def main():
    rows = load_csv("results/passK/passk_hparam_sweep.csv")
    base = load_csv("results/passK/passk_baselines.csv")
    outdir = "results/passK"
    os.makedirs(outdir, exist_ok=True)

    student = [passk_at_exact(base, "Qwen/Qwen3-0.6B", DATASET, k) for k in ALL_K]
    teacher = [passk_at_exact(base, "Qwen/Qwen3-8B", DATASET, k) for k in ALL_K]

    for title, (jsd_label, jsd_frag), variants in FAMILIES:
        jsd_ys = [passk_at_frag(rows, jsd_frag, DATASET, k) for k in ALL_K]

        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        ax.plot(ALL_K, student, "--", color="black", linewidth=2, zorder=3,
                label="untrained student (0.6B)")
        ax.plot(ALL_K, teacher, "--", color="gray", linewidth=2, zorder=3,
                label="teacher (8B)")
        ax.plot(ALL_K, jsd_ys, "--", color="#0b3d91", linewidth=2, zorder=4,
                label=f"{jsd_label} baseline ({jsd_frag})")

        for i, (label, frag) in enumerate(variants):
            ys = [passk_at_frag(rows, frag, DATASET, k) for k in ALL_K]
            if any(y is None for y in ys):
                print(f"  [skip] {title}: {frag} missing data on {DATASET}")
                continue
            ax.plot(ALL_K, ys, "-o", color=COLORS[i % len(COLORS)], linewidth=2,
                     markersize=5, zorder=5, label=label)

        ax.set_xscale("log", base=2)
        ax.set_xticks(ALL_K)
        ax.set_xticklabels([str(k) for k in ALL_K])
        ax.set_xlabel("k")
        ax.set_ylabel("pass@k")
        ax.set_title(f"{title} -- {DATASET.replace('.jsonl', '')}", fontsize=10)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7, loc="lower right")
        fig.tight_layout()
        path = os.path.join(outdir, f"ablation_{title}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
