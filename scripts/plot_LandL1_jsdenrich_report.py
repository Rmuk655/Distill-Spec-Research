#!/usr/bin/env python
"""
plot_LandL1_jsdenrich_report.py -- jsd_flat_enrich_K3-only L/L1 sweep, final
rerun data (results/passK/LandL1sweepJsdEnrich.csv =
l_sweep_jsd_enrich_final.csv [L1=0, all 9 verifiers] +
l1_sweep_jsd_enrich_final.csv [L1=6/8/9/10/12, traversal+specinfer],
checkpoint jsd_flat_enrich_K3_mathhard_s456, L=16/32).

One figure per eval dataset. Each figure plots BE (left axis, solid, circle
markers) and throughput (right axis, dashed, triangle markers) together,
for traversal and specinfer, at L=16 and L=32, across L1.

USAGE:
    python scripts/plot_LandL1_jsdenrich_report.py
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATASETS = ["math_eval", "gsm8k_eval", "olympiad_eval"]
VERIFIERS = ["traversal", "specinfer"]
Ls = [16, 32]
VCOLORS = {"traversal": "#0b3d91", "specinfer": "#b35806"}
LSTYLES = {16: "-", 32: "--"}


def load(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def series(rows, v, L, dataset):
    xs, be, tp = [], [], []
    for L1 in [0, 6, 8, 9, 10, 12]:
        mode = v if L1 == 0 else f"{v}_dL{L1}"
        for r in rows:
            if (r["mode"] == mode and r["L"] == str(L) and r["dataset"] == dataset
                    and r.get("L1", "0") == str(L1)):
                xs.append(L1)
                be.append(float(r["block_eff"]))
                tp.append(float(r["throughput_tok_s"]))
                break
    return xs, be, tp


def main():
    rows = load("results/passK/LandL1sweepJsdEnrich.csv")
    outdir = "results/passK"
    os.makedirs(outdir, exist_ok=True)

    for dataset in DATASETS:
        fig, ax1 = plt.subplots(figsize=(8.5, 6))
        ax2 = ax1.twinx()
        handles = []
        for v in VERIFIERS:
            for L in Ls:
                xs, be, tp = series(rows, v, L, dataset)
                if not xs:
                    continue
                color = VCOLORS[v]
                ls = LSTYLES[L]
                h1, = ax1.plot(xs, be, ls, color=color, marker="o", linewidth=2,
                                markersize=6, label=f"{v}, L={L} -- BE")
                h2, = ax2.plot(xs, tp, ls, color=color, marker="^", linewidth=1.3,
                                markersize=6, alpha=0.55, label=f"{v}, L={L} -- throughput")
                handles.extend([h1, h2])
        ax1.set_xlabel("L1 (0 = full branching, no delay)")
        ax1.set_ylabel("block_eff (solid, circle)")
        ax2.set_ylabel("throughput, tok/s (dashed hue, triangle)")
        ax1.set_title(f"jsd_flat_enrich_K3 checkpoint: BE and throughput vs L1, at L=16/32 -- {dataset}", fontsize=10)
        ax1.grid(alpha=0.3)
        ax1.legend(handles=handles, fontsize=7, loc="center left", bbox_to_anchor=(1.08, 0.5))
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"LandL1_jsdenrich_report_{dataset}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote LandL1_jsdenrich_report_{dataset}.png")


if __name__ == "__main__":
    main()
