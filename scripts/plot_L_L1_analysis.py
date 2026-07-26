#!/usr/bin/env python
"""
plot_L_L1_analysis.py -- combines every L/L1 eval-grid CSV in the repo into
one crisp set of charts: how does tree depth L affect each verifier's BE and
throughput, and where does the L1 (delayed-branching) sweet spot sit, at
L=8/16/32, for both BE and throughput. Every figure is generated once per
dataset (math_eval, gsm8k_eval, olympiad_eval), not just math_eval.

Sources (K=3):
  - results/jsd_enrich_ddte_results.csv          -- L=8, L1=0 (9 verifiers) + L1=2..6 (traversal/specinfer); math_eval + gsm8k_eval (~1 row) only, no olympiad_eval
  - results/per_checkpoint_sweeps_2026-07/jsd_mathhard_s123.csv
                                                 -- SAME jsd_mathhard_s123 checkpoint, L=8, math_eval + olympiad_eval (no gsm8k_eval); used here ONLY to fill in olympiad_eval L=8 for jsd, averaged over its 4 repeat-run timestamps per (mode, L1)
  - results/passK/l_sweep_jsd_rerun.csv         -- L=16/32, L1=0, 9 verifiers, jsd_lr1e-5_wu10_lrmin0.1_wd0.01 (confirmatory rerun, all 3 datasets complete)
  - results/passK/l_sweep_jsd_enrich_final.csv  -- L=16/32, L1=0, 9 verifiers, jsd_flat_enrich_K3_mathhard_s456 (final rerun, all 3 datasets complete)
  - results/passK/l1_sweep_jsd_rerun.csv        -- L=16/32, L1=6/8/9/10/12, traversal+specinfer, same jsd ckpt (rerun, all 3 datasets complete)
  - results/passK/l1_sweep_jsd_enrich_final.csv -- L=16/32, L1=6/8/9/10/12, traversal+specinfer, same enrich ckpt (final rerun, all 3 datasets complete)

CAVEAT baked into every chart title: the L=8 checkpoint (jsd_mathhard_s123 /
jsd_flat_enrich_K3_mathhard_s123) is NOT the same training run as the L=16/32
checkpoint (jsd_lr1e-5_wu10_lrmin0.1_wd0.01 / jsd_flat_enrich_K3_mathhard_s456)
-- same loss family and dataset, different seed/hyperparams. Trend direction
is still meaningful; exact magnitude comparisons across L=8 vs L=16/32 should
be read with that in mind. L=8 olympiad_eval exists only for the jsd
checkpoint (via the per_checkpoint_sweeps file above), not for enrich --
enrich's L=8 stays math_eval/gsm8k_eval only, and its olympiad_eval charts
start at L=16.

USAGE:
    python scripts/plot_L_L1_analysis.py
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATASETS = ["math_eval", "gsm8k_eval", "olympiad_eval"]
VERIFIERS_9 = ["naive", "nss", "specinfer", "spectr", "khisti", "max", "bv", "gbv", "traversal"]
COLORS_9 = {
    "naive": "#2a78d6", "nss": "#eb6834", "specinfer": "#1baf7a", "spectr": "#eda100",
    "khisti": "#e87ba4", "max": "#008300", "bv": "#4a3aa7", "gbv": "#e34948", "traversal": "#000000",
}
Ls = [8, 16, 32]
L1_COLORS = {8: "#a3c2f4", 16: "#5b8fd6", 32: "#0b3d91"}


def load(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def avg_supp_row(rows, mode, L1, dataset, K="3"):
    matches = [r for r in rows if r["mode"] == mode and r["K"] == K
               and r["dataset"] == dataset and r.get("L1", "0") == str(L1) and r["L"] == "8"]
    if not matches:
        return None
    be = sum(float(r["block_eff"]) for r in matches) / len(matches)
    tp = sum(float(r["throughput_tok_s"]) for r in matches) / len(matches)
    return {"block_eff": str(be), "throughput_tok_s": str(tp)}


def ddte_row(rows, ckpt_frag, mode, L1, dataset, K="3", jsd_l8_supp=None):
    for r in rows:
        if (ckpt_frag in r["checkpoint"] and r["mode"] == mode and r["K"] == K
                and r["dataset"] == dataset and r.get("L1", "0") == str(L1) and r["L"] == "8"):
            return r
    if dataset == "olympiad_eval" and ckpt_frag == "jsd_mathhard_s123" and jsd_l8_supp is not None:
        return avg_supp_row(jsd_l8_supp, mode, L1, dataset, K)
    return None


def sweep_row(rows, mode, L, dataset):
    for r in rows:
        if r["mode"] == mode and r["L"] == str(L) and r["dataset"] == dataset:
            return r
    return None


def l1_row(rows, mode_base, L, L1, dataset):
    mode = f"{mode_base}_dL{L1}"
    for r in rows:
        if r["mode"] == mode and r["L"] == str(L) and r["dataset"] == dataset and r["L1"] == str(L1):
            return r
    return None


def fig1(outdir, dataset, tag, ddte, l_jsd, jsd_l8_supp):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for v in VERIFIERS_9:
        be, tp = [], []
        for L in Ls:
            if L == 8:
                r = ddte_row(ddte, "jsd_mathhard_s123", v, 0, dataset, jsd_l8_supp=jsd_l8_supp)
            else:
                r = sweep_row(l_jsd, v, L, dataset)
            be.append(float(r["block_eff"]) if r else None)
            tp.append(float(r["throughput_tok_s"]) if r else None)
        axes[0].plot(Ls, be, "-o", color=COLORS_9[v], label=v, linewidth=2, markersize=5)
        axes[1].plot(Ls, tp, "-o", color=COLORS_9[v], label=v, linewidth=2, markersize=5)
    for ax, ylab, title in zip(axes, ["block_eff", "throughput (tok/s)"],
                               ["BE vs L, by verifier", "Throughput vs L, by verifier"]):
        ax.set_xticks(Ls)
        ax.set_xlabel("L")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    axes[1].legend(fontsize=7, loc="upper right")
    fig.suptitle(f"K=3, L1=0, {dataset} -- L=8 ckpt differs from L=16/32 ckpt (see script docstring)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"L_L1_fig1_be_throughput_vs_L_all_verifiers_{tag}.png"), dpi=150)
    plt.close(fig)


def fig23(outdir, dataset, tag, ddte, l_jsd, l1_jsd, jsd_l8_supp):
    for v, fig_n in [("traversal", 2), ("specinfer", 3)]:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        for L in Ls:
            xs, be, tp = [], [], []
            if L == 8:
                for L1 in [0, 2, 3, 4, 5, 6]:
                    r = ddte_row(ddte, "jsd_mathhard_s123", v if L1 == 0 else f"{v}_dL{L1}", L1, dataset, jsd_l8_supp=jsd_l8_supp)
                    if r:
                        xs.append(L1); be.append(float(r["block_eff"])); tp.append(float(r["throughput_tok_s"]))
            else:
                base = sweep_row(l_jsd, v, L, dataset)
                if base:
                    xs.append(0); be.append(float(base["block_eff"])); tp.append(float(base["throughput_tok_s"]))
                for L1 in [6, 8, 9, 10, 12]:
                    r = l1_row(l1_jsd, v, L, L1, dataset)
                    if r:
                        xs.append(L1); be.append(float(r["block_eff"])); tp.append(float(r["throughput_tok_s"]))
            if xs:
                axes[0].plot(xs, be, "-o", color=L1_COLORS[L], label=f"L={L}", linewidth=2, markersize=5)
                axes[1].plot(xs, tp, "-o", color=L1_COLORS[L], label=f"L={L}", linewidth=2, markersize=5)
        for ax, ylab, title in zip(axes, ["block_eff", "throughput (tok/s)"],
                                   [f"{v}: BE vs L1", f"{v}: throughput vs L1"]):
            ax.set_xlabel("L1 (0 = full branching, no delay)")
            ax.set_ylabel(ylab)
            ax.set_title(title, fontsize=10)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        fig.suptitle(f"K=3, {dataset}, jsd checkpoint -- {v}", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"L_L1_fig{fig_n}_be_throughput_vs_L1_{v}_{tag}.png"), dpi=150)
        plt.close(fig)


def fig4(outdir, dataset, tag, ddte, l_jsd, l1_jsd, jsd_l8_supp):
    fig, ax = plt.subplots(figsize=(7.5, 6))
    markers = {"traversal": "o", "specinfer": "s"}
    for v in ["traversal", "specinfer"]:
        for L in Ls:
            pts = []
            if L == 8:
                for L1 in [0, 2, 3, 4, 5, 6]:
                    r = ddte_row(ddte, "jsd_mathhard_s123", v if L1 == 0 else f"{v}_dL{L1}", L1, dataset, jsd_l8_supp=jsd_l8_supp)
                    if r:
                        pts.append((float(r["block_eff"]), float(r["throughput_tok_s"])))
            else:
                base = sweep_row(l_jsd, v, L, dataset)
                if base:
                    pts.append((float(base["block_eff"]), float(base["throughput_tok_s"])))
                for L1 in [6, 8, 9, 10, 12]:
                    r = l1_row(l1_jsd, v, L, L1, dataset)
                    if r:
                        pts.append((float(r["block_eff"]), float(r["throughput_tok_s"])))
            if pts:
                xs, ys = zip(*pts)
                ax.scatter(xs, ys, color=L1_COLORS[L], marker=markers[v], s=60,
                           label=f"{v}, L={L}", edgecolors="black", linewidths=0.5)
    ax.set_xlabel("block_eff (higher = more tokens accepted per call)")
    ax.set_ylabel("throughput (tok/s, higher = faster wall-clock)")
    ax.set_title(f"Efficiency frontier: throughput vs BE, {dataset}\n(top-right is better; circle=traversal, square=specinfer)", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"L_L1_fig4_efficiency_frontier_{tag}.png"), dpi=150)
    plt.close(fig)


def fig5(outdir, dataset, tag, ddte, l_enrich, l1_enrich):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    v = "traversal"
    for L in Ls:
        xs, be, tp = [], [], []
        if L == 8:
            for L1 in [0, 2, 3, 4, 5, 6]:
                r = ddte_row(ddte, "jsd_flat_enrich_K3_mathhard_s123", v if L1 == 0 else f"{v}_dL{L1}", L1, dataset)
                if r:
                    xs.append(L1); be.append(float(r["block_eff"])); tp.append(float(r["throughput_tok_s"]))
        else:
            base = sweep_row(l_enrich, v, L, dataset)
            if base:
                xs.append(0); be.append(float(base["block_eff"])); tp.append(float(base["throughput_tok_s"]))
            for L1 in [6, 8, 9, 10, 12]:
                r = l1_row(l1_enrich, v, L, L1, dataset)
                if r:
                    xs.append(L1); be.append(float(r["block_eff"])); tp.append(float(r["throughput_tok_s"]))
        if xs:
            axes[0].plot(xs, be, "-o", color=L1_COLORS[L], label=f"L={L}", linewidth=2, markersize=5)
            axes[1].plot(xs, tp, "-o", color=L1_COLORS[L], label=f"L={L}", linewidth=2, markersize=5)
    for ax, ylab, title in zip(axes, ["block_eff", "throughput (tok/s)"],
                               ["enrich: BE vs L1", "enrich: throughput vs L1"]):
        ax.set_xlabel("L1")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"K=3, {dataset}, jsd_flat_enrich_K3 checkpoint -- traversal (confirmatory check)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"L_L1_fig5_be_throughput_vs_L1_enrich_traversal_{tag}.png"), dpi=150)
    plt.close(fig)


def fig6(outdir, dataset, tag, l_jsd, l_enrich, l1_jsd, l1_enrich):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    ckpt_colors = {"jsd": "#0b3d91", "enrich": "#b30000"}
    for ax, v in zip(axes, ["traversal", "specinfer"]):
        for tag_c, base, l1set in [("jsd", l_jsd, l1_jsd), ("enrich", l_enrich, l1_enrich)]:
            xs, be = [], []
            base_row = sweep_row(base, v, 32, dataset)
            if base_row:
                xs.append(0); be.append(float(base_row["block_eff"]))
            for L1 in [6, 8, 9, 10, 12]:
                r = l1_row(l1set, v, 32, L1, dataset)
                if r:
                    xs.append(L1); be.append(float(r["block_eff"]))
            if xs:
                ax.plot(xs, be, "-o", color=ckpt_colors[tag_c], label=tag_c, linewidth=2, markersize=6)
        ax.set_xlabel("L1")
        ax.set_ylabel("block_eff")
        ax.set_title(f"{v}, L=32: jsd vs enrich", fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    fig.suptitle(f"K=3, {dataset}, L=32 -- does enrich benefit more from L1 than plain jsd?", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"L_L1_fig6_jsd_vs_enrich_L32_{tag}.png"), dpi=150)
    plt.close(fig)


def main():
    ddte = load("results/jsd_enrich_ddte_results.csv")
    l_jsd = load("results/passK/l_sweep_jsd_rerun.csv")
    l_enrich = load("results/passK/l_sweep_jsd_enrich_final.csv")
    l1_jsd = load("results/passK/l1_sweep_jsd_rerun.csv")
    l1_enrich = load("results/passK/l1_sweep_jsd_enrich_final.csv")
    jsd_l8_supp = load("results/per_checkpoint_sweeps_2026-07/jsd_mathhard_s123.csv")
    outdir = "results/passK"
    os.makedirs(outdir, exist_ok=True)

    for dataset in DATASETS:
        tag = dataset
        fig1(outdir, dataset, tag, ddte, l_jsd, jsd_l8_supp)
        fig23(outdir, dataset, tag, ddte, l_jsd, l1_jsd, jsd_l8_supp)
        fig4(outdir, dataset, tag, ddte, l_jsd, l1_jsd, jsd_l8_supp)
        fig5(outdir, dataset, tag, ddte, l_enrich, l1_enrich)
        fig6(outdir, dataset, tag, l_jsd, l_enrich, l1_jsd, l1_enrich)
        print(f"wrote figs 1-6 for {dataset}")


if __name__ == "__main__":
    main()
