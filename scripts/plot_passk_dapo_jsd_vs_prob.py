#!/usr/bin/env python
"""
plot_passk_dapo_jsd_vs_prob.py -- pass@k vs k, one chart per dataset, comparing
jsd and prefix_overlap/prob when both are trained on dapo_math_train (DAPO-Math-17k)
instead of the usual math_hard, against the same untrained-student/teacher
reference lines used in plot_passk_curves_all.py.

prob's checkpoint is po_prob_multiroot_N16_freshM1_ceanneal9000_auxw0.5_dapo17k --
prob's established best-BE deployment pick on dapo (see hyperparam_sweep_research_report.md),
not a best-of-sweep-after-seeing-pass@k cherry pick.

USAGE:
    python scripts/plot_passk_dapo_jsd_vs_prob.py
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ALL_K = [1, 2, 4, 8, 16, 32, 64]
DATASETS = ["math_eval.jsonl", "gsm8k_eval.jsonl", "olympiad_eval.jsonl"]
JSD_FRAG = "jsd_lr1e-5_wu10_lrmin0.1_wd0.01_dapo17k"
PROB_FRAG = "po_prob_multiroot_N16_freshM1_ceanneal9000_auxw0.5_dapo17k_lr1e-5_wu20"


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
    rows = load_csv("results/passK/passk_hparam_sweep.csv")
    base = load_csv("results/passK/passk_baselines.csv")
    outdir = "results/passK"
    os.makedirs(outdir, exist_ok=True)

    for ds in DATASETS:
        tag = ds.replace(".jsonl", "")
        student = [passk_at_exact(base, "Qwen/Qwen3-0.6B", ds, k) for k in ALL_K]
        teacher = [passk_at_exact(base, "Qwen/Qwen3-8B", ds, k) for k in ALL_K]
        jsd_ys = [passk_at_frag(rows, JSD_FRAG, ds, k) for k in ALL_K]
        prob_ys = [passk_at_frag(rows, PROB_FRAG, ds, k) for k in ALL_K]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(ALL_K, student, "--", color="black", linewidth=2, zorder=3,
                label="untrained student (0.6B)")
        ax.plot(ALL_K, teacher, "--", color="gray", linewidth=2, zorder=3,
                label="teacher (8B)")
        ax.plot(ALL_K, jsd_ys, "-s", color="#0b3d91", linewidth=2.5, markersize=6, zorder=5,
                label="jsd + dapo17k")
        ax.plot(ALL_K, prob_ys, "-o", color="#b30000", linewidth=2.5, markersize=6, zorder=5,
                label="prob + dapo17k (ceanneal9000_auxw0.5)")

        ax.set_xscale("log", base=2)
        ax.set_xticks(ALL_K)
        ax.set_xticklabels([str(k) for k in ALL_K])
        ax.set_xlabel("k")
        ax.set_ylabel("pass@k")
        ax.set_title(f"pass@k vs k, jsd vs prob trained on dapo_math_train ({tag})", fontsize=10)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8, loc="lower right")
        fig.tight_layout()
        path = os.path.join(outdir, f"passk_dapo_jsd_vs_prob_{tag}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
