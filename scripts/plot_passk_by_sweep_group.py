#!/usr/bin/env python
"""
plot_passk_by_sweep_group.py — one pass@k-vs-k chart PER sweep dimension,
instead of cramming all 18 checkpoints (winner/near/below x 6 groups) onto one
axis. Reads results/passk_hparam_sweep.csv (written by
scripts/passk_eval_sweep_checkpoints.sh in RUN_NAMES mode) and emits:

    Results/passK/passk_group_lr.png            (prob LR: 3e-6 / 7e-6 / 1e-6)
    Results/passK/passk_group_lr_jsd.png         (jsd LR:  1e-5 / 3e-6 / 1e-6)
    Results/passK/passk_group_warmup.png         (5% / 10% / 20%)
    Results/passK/passk_group_lrmin.png          (0.1 / 0.01)
    Results/passK/passk_group_topk.png           (0 / 20 / 50)
    Results/passK/passk_group_weight_decay.png   (0.01 / 0.001 / 0.0001)
    Results/passK/passk_group_ceanneal.png       (anneal_steps + aux_weight winners)

Every chart also overlays the untrained Qwen3-0.6B (student floor) and
Qwen3-8B (teacher ceiling) baselines from Results/passK/passk_baselines.csv,
as dashed gray reference lines -- so each sweep point is read against "how far
did this move from the student, how much ceiling is left before the teacher,"
not just against its sibling checkpoints.

USAGE:
    python scripts/plot_passk_by_sweep_group.py
    python scripts/plot_passk_by_sweep_group.py --dataset math_eval.jsonl --csv results/passk_hparam_sweep.csv
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTDIR_DEFAULT = "Results/passK"
FOREST = ["#2a78d6", "#008300", "#e34948", "#eda100", "#4a3aa7"]

# checkpoint dir name (as it appears in the CSV's `model` path) -> readable label.
# GROUPS: {chart_key: {label: run_name_fragment}}
GROUPS = {
    "lr (prob)": {
        "3e-6 (winner)": "po_prob_lr3e-6_wu20",
        "7e-6 (near)": "po_prob_lr7e-6_wu20",
        "1e-6 (below)": "po_prob_lr1e-6_wu20",
    },
    "lr (jsd)": {
        "1e-5 (winner)": "jsd_lr1e-5_wu10_lrmin0.1_wd0.01",
        "3e-6 (near)": "jsd_lr3e-6_wu10_lrmin0.1_wd0.01",
        "1e-6 (below)": "jsd_lr1e-6_wu10_lrmin0.1_wd0.01",
    },
    "warmup": {
        "20% (winner)": "po_prob_lr1e-5_wu20_lrmin0.1_wd0.01",
        "10% (near)": "po_prob_lr1e-5_wu10_lrmin0.1_wd0.01",
        "5% (below)": "po_prob_lr1e-5_wu5_lrmin0.1_wd0.01",
    },
    "lr_min_ratio": {
        "0.1 (winner)": "po_prob_lr1e-5_wu20_lrmin0.1_wd0.01",
        "0.01 (below)": "po_prob_lr1e-5_wu20_lrmin0.01_wd0.01",
    },
    "teacher top-k": {
        "topk=0 (winner)": "po_prob_lr1e-5_wu20_lrmin0.1_wd0.01",
        "topk=20 (near)": "po_prob_topk20_lr1e-5_wu20",
        "topk=50 (below)": "po_prob_topk50_lr1e-5_wu20",
    },
    "weight decay": {
        "wd=0.0001 (winner)": "po_prob_lr1e-5_wu20_wd0.0001",
        "wd=0.01 (near)": "po_prob_lr1e-5_wu20_lrmin0.1_wd0.01",
        "wd=0.001 (below)": "po_prob_lr1e-5_wu20_wd0.001",
    },
    "CE-anneal": {
        "anneal=3000,aux_w=0.5 (winner)": "po_prob_ceanneal3000_auxw0.5_lr1e-5_wu20",
        "anneal=3000,aux_w=1.0 (near)": "po_prob_ceanneal_lr1e-5_wu20",
        "anneal=4000,aux_w=1.0": "po_prob_ceanneal4000_lr1e-5_wu20",
        "anneal=1500,aux_w=1.0 (below)": "po_prob_ceanneal1500_lr1e-5_wu20",
        "anneal=3000,aux_w=2.0 (below)": "po_prob_ceanneal3000_auxw2.0_lr1e-5_wu20",
    },
}


def load_csv(path):
    import csv
    rows = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def series_for(rows, run_fragment, dataset):
    """model column is the full ckpt_best path -- match by substring."""
    pts = {}
    for r in rows:
        if run_fragment in r["model"] and r["dataset"] == dataset:
            pts[int(r["k"])] = float(r["pass_at_k"])
    return pts


def plot_group(rows, group_name, members, dataset, outdir, baseline_rows):
    fig, ax = plt.subplots(figsize=(7, 5))
    any_data = False

    # student floor / teacher ceiling, same on every chart for comparability
    for label, model_id, style in (
        ("Qwen3-0.6B (untrained student)", "Qwen/Qwen3-0.6B", dict(color="#888", linestyle="--")),
        ("Qwen3-8B (teacher)", "Qwen/Qwen3-8B", dict(color="#888", linestyle=":")),
    ):
        pts = series_for(baseline_rows, model_id, dataset)
        if pts:
            ks = sorted(pts)
            ax.plot(ks, [pts[k] for k in ks], marker="x", markersize=5, linewidth=1.5,
                    label=label, zorder=1, **style)

    for (label, frag), color in zip(members.items(), FOREST):
        pts = series_for(rows, frag, dataset)
        if not pts:
            continue
        any_data = True
        ks = sorted(pts)
        ax.plot(ks, [pts[k] for k in ks], "-o", color=color, linewidth=2, markersize=6,
                label=label, zorder=2)
    if not any_data:
        plt.close(fig)
        return False
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32, 64])
    ax.set_xticklabels(["1", "2", "4", "8", "16", "32", "64"])
    ax.set_xlabel("k")
    ax.set_ylabel("pass@k")
    ax.set_title(f"{group_name} — pass@k ({dataset.replace('.jsonl', '')})")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fname = "passk_group_" + group_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_") + ".png"
    fig.savefig(os.path.join(outdir, fname), dpi=150)
    plt.close(fig)
    print(f"wrote {fname}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/passk_hparam_sweep.csv")
    ap.add_argument("--baselines_csv", default="Results/passK/passk_baselines.csv")
    ap.add_argument("--dataset", default="math_eval.jsonl")
    ap.add_argument("--outdir", default=OUTDIR_DEFAULT)
    args = ap.parse_args()

    if not os.path.isfile(args.csv):
        raise SystemExit(f"{args.csv} not found -- run scripts/passk_eval_sweep_checkpoints.sh first")
    rows = load_csv(args.csv)
    baseline_rows = load_csv(args.baselines_csv) if os.path.isfile(args.baselines_csv) else []
    if not baseline_rows:
        print(f"WARNING: {args.baselines_csv} not found -- charts will have no student/teacher overlay")
    os.makedirs(args.outdir, exist_ok=True)

    any_written = False
    for group_name, members in GROUPS.items():
        any_written |= plot_group(rows, group_name, members, args.dataset, args.outdir, baseline_rows)
    if not any_written:
        print("No matching rows found for any group -- check --dataset and that the CSV "
              "actually contains these run names (grep the model column).")


if __name__ == "__main__":
    main()
