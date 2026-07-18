#!/usr/bin/env python
"""
plot_passk_curves_all.py — one line per checkpoint, x=k, y=pass@k, student and
teacher baselines drawn as bold dashed reference lines -- the lift is directly
the vertical gap between a checkpoint's line and the student line, bounded
above by the teacher line.

Every 0.6B/8B checkpoint in the sweep CSV gets a line (thin, colored by loss
family: prefix_overlap/prob = red shades, jsd = blue shades) so nothing is
hidden -- but the best-of-family prob and best-of-family jsd checkpoints
(same selection as plot_prob_vs_jsd_advantage.py) are drawn bold, others
drawn thin/faded, so the winners are easy to spot without removing the rest.

One chart per dataset (math_eval/gsm8k_eval/olympiad_eval), x-axis log scale.

USAGE:
    python scripts/plot_passk_curves_all.py
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ALL_K = [1, 2, 4, 8, 16, 32, 64]
K_OF_INTEREST = [2, 4, 8, 16]
DATASETS = ["math_eval.jsonl", "gsm8k_eval.jsonl", "olympiad_eval.jsonl"]


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


def best_checkpoint(rows, ds, student, teacher, family_prefix):
    names = sorted(set(run_name(r["model"]) for r in rows if r["dataset"] == ds))
    best_name, best_mean = None, -1e9
    for name in names:
        if family_prefix == "jsd" and not name.startswith("jsd"):
            continue
        if family_prefix == "prob" and not name.startswith("po_"):
            continue
        pcts = []
        for k in K_OF_INTEREST:
            v = passk_at_frag(rows, name, ds, k)
            s, t = student.get(k), teacher.get(k)
            if v is None or s is None or t is None or t == s:
                continue
            pcts.append(100 * (v - s) / (t - s))
        if len(pcts) < len(K_OF_INTEREST):
            continue
        mean_pct = np.mean(pcts)
        if mean_pct > best_mean:
            best_name, best_mean = name, mean_pct
    return best_name


def main():
    rows = load_csv("results/passK/passk_hparam_sweep.csv")
    base = load_csv("results/passK/passk_baselines.csv")
    outdir = "results/passK"
    os.makedirs(outdir, exist_ok=True)

    for ds in DATASETS:
        tag = ds.replace(".jsonl", "")
        student = {k: passk_at_exact(base, "Qwen/Qwen3-0.6B", ds, k) for k in ALL_K}
        teacher = {k: passk_at_exact(base, "Qwen/Qwen3-8B", ds, k) for k in ALL_K}
        best_prob = best_checkpoint(rows, ds, student, teacher, "prob")
        best_jsd = best_checkpoint(rows, ds, student, teacher, "jsd")

        names = sorted(set(run_name(r["model"]) for r in rows if r["dataset"] == ds))

        fig, ax = plt.subplots(figsize=(9, 6.5))
        for name in names:
            ys = [passk_at_frag(rows, name, ds, k) for k in ALL_K]
            if any(y is None for y in ys):
                continue
            is_jsd = name.startswith("jsd")
            if name == best_prob:
                ax.plot(ALL_K, ys, "-o", color="#b30000", linewidth=2.5, markersize=5, zorder=5,
                        label=f"BEST prob: {name}")
            elif name == best_jsd:
                ax.plot(ALL_K, ys, "-s", color="#0b3d91", linewidth=2.5, markersize=5, zorder=5,
                        label=f"BEST jsd: {name}")
            else:
                color = "#f4a3a3" if not is_jsd else "#a3c2f4"
                ax.plot(ALL_K, ys, "-", color=color, linewidth=0.9, alpha=0.7, zorder=2)

        s_ys = [student[k] for k in ALL_K]
        t_ys = [teacher[k] for k in ALL_K]
        ax.plot(ALL_K, s_ys, "--", color="black", linewidth=2, zorder=6, label="untrained student (0.6B)")
        ax.plot(ALL_K, t_ys, "--", color="gray", linewidth=2, zorder=6, label="teacher (8B)")

        ax.set_xscale("log", base=2)
        ax.set_xticks(ALL_K)
        ax.set_xticklabels([str(k) for k in ALL_K])
        ax.set_xlabel("k")
        ax.set_ylabel("pass@k")
        ax.set_title(f"pass@k vs k, every 0.6B/8B checkpoint ({tag})\n"
                     "thin red = other prob runs, thin blue = other jsd runs, bold = best of each family", fontsize=10)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8, loc="lower right")
        fig.tight_layout()
        path = os.path.join(outdir, f"passk_curves_all_{tag}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
