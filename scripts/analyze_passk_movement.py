#!/usr/bin/env python
"""
analyze_passk_movement.py — two questions the charts alone don't answer directly:

  1. Which checkpoint moved pass@k (k=2,4,8,16 especially) the MOST relative to
     the untrained Qwen3-0.6B student baseline? Ranks every swept checkpoint by
     delta-vs-student at each k -- but only REPORTS a delta as real movement if
     it clears a noise-floor threshold (see NOISE_FLOOR_PASSK below).
  2. Did BE and pass@k8/k16 move in a correlated fashion across the sweep, or
     did some group move BE without moving pass@k (or vice versa)? Per sweep
     group: BE range vs pass@k8/k16 range.

NOISE FLOOR CAVEAT (read before trusting any number here): this sweep has NO
repeated seeds and NO repeated pass@k evals -- there is no MEASURED noise floor
for pass@k anywhere in this project (same gap flagged for block_eff in
mine_wandb_runs.py). NOISE_FLOOR_PASSK below is a HEURISTIC, chosen by analogy
to the ~0.05 BE gap this whole sweep has treated as noise-level (e.g. the
warmup/lr_min/wd stages), not a validated statistical bound. Treat any
"significant" call below as a candidate worth a repeat-seed check, not a
settled fact.

Reads results/passk_hparam_sweep.csv (checkpoints) + Results/passK/passk_baselines.csv
(student/teacher) + the known best_be per checkpoint (hardcoded from
summarize_sweep.py's table -- update this dict if the sweep changes).

USAGE:
    python scripts/analyze_passk_movement.py
    python scripts/analyze_passk_movement.py --dataset math_eval.jsonl --out results/passk_movement_report.md
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# HEURISTIC, not measured -- see caveat above. Deltas smaller than this are NOT
# reported as real movement, only listed for completeness.
NOISE_FLOOR_PASSK = 0.05

# checkpoint dir fragment -> best_val_block_eff, from the summarize_sweep table
# (hardcode -> update this if the sweep changes; keeps this script standalone).
BEST_BE = {
    "po_prob_lr3e-6_wu20": 5.701,
    "po_prob_lr7e-6_wu20": 5.673,
    "po_prob_lr1e-6_wu20": 5.591,
    "jsd_lr1e-5_wu10_lrmin0.1_wd0.01": 5.994,
    "jsd_lr3e-6_wu10_lrmin0.1_wd0.01": 5.971,
    "jsd_lr1e-6_wu10_lrmin0.1_wd0.01": 5.752,
    "po_prob_lr1e-5_wu20_lrmin0.1_wd0.01": 5.608,
    "po_prob_lr1e-5_wu10_lrmin0.1_wd0.01": 5.560,
    "po_prob_lr1e-5_wu5_lrmin0.1_wd0.01": 5.501,
    "po_prob_lr1e-5_wu20_lrmin0.01_wd0.01": 5.589,
    "po_prob_topk20_lr1e-5_wu20": 5.591,
    "po_prob_topk50_lr1e-5_wu20": 5.506,
    "po_prob_lr1e-5_wu20_wd0.0001": 5.617,
    "po_prob_lr1e-5_wu20_wd0.001": 5.532,
    "po_prob_ceanneal1500_lr1e-5_wu20": 5.737,
    "po_prob_ceanneal_lr1e-5_wu20": 5.796,
    "po_prob_ceanneal4000_lr1e-5_wu20": 5.764,
    "po_prob_ceanneal3000_auxw0.5_lr1e-5_wu20": 5.848,
    "po_prob_ceanneal3000_auxw2.0_lr1e-5_wu20": 5.794,
}

GROUPS = {
    "lr (prob)": ["po_prob_lr3e-6_wu20", "po_prob_lr7e-6_wu20", "po_prob_lr1e-6_wu20",
                  "po_prob_lr1e-5_wu20_lrmin0.1_wd0.01"],
    "lr (jsd)": ["jsd_lr1e-5_wu10_lrmin0.1_wd0.01", "jsd_lr3e-6_wu10_lrmin0.1_wd0.01",
                 "jsd_lr1e-6_wu10_lrmin0.1_wd0.01"],
    "warmup": ["po_prob_lr1e-5_wu20_lrmin0.1_wd0.01", "po_prob_lr1e-5_wu10_lrmin0.1_wd0.01",
               "po_prob_lr1e-5_wu5_lrmin0.1_wd0.01"],
    "lr_min_ratio": ["po_prob_lr1e-5_wu20_lrmin0.1_wd0.01", "po_prob_lr1e-5_wu20_lrmin0.01_wd0.01"],
    "teacher top-k": ["po_prob_lr1e-5_wu20_lrmin0.1_wd0.01", "po_prob_topk20_lr1e-5_wu20",
                      "po_prob_topk50_lr1e-5_wu20"],
    "weight decay": ["po_prob_lr1e-5_wu20_wd0.0001", "po_prob_lr1e-5_wu20_lrmin0.1_wd0.01",
                      "po_prob_lr1e-5_wu20_wd0.001"],
    "CE-anneal": ["po_prob_ceanneal3000_auxw0.5_lr1e-5_wu20", "po_prob_ceanneal_lr1e-5_wu20",
                  "po_prob_ceanneal4000_lr1e-5_wu20", "po_prob_ceanneal1500_lr1e-5_wu20",
                  "po_prob_ceanneal3000_auxw2.0_lr1e-5_wu20"],
}

K_OF_INTEREST = [2, 4, 8, 16]


def load_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def passk_at(rows, frag_or_id, dataset, k):
    for r in rows:
        if frag_or_id in r["model"] and r["dataset"] == dataset and int(r["k"]) == k:
            return float(r["pass_at_k"])
    return None


def plot_movement(move_rows, outdir, dataset):
    """Grouped bar chart: Δk2/Δk4/Δk8/Δk16 vs student per checkpoint, with the
    heuristic noise-floor band shaded -- only bars clearing it are solid;
    below-floor bars are drawn hatched/faded so they read as 'not significant'."""
    import numpy as np
    names = [frag for frag, _ in move_rows]
    ks = K_OF_INTEREST
    colors = ["#2a78d6", "#008300", "#e34948", "#eda100"]
    fig, ax = plt.subplots(figsize=(max(10, len(names) * 0.6), 6))
    x = np.arange(len(names))
    width = 0.2
    for i, k in enumerate(ks):
        vals = [d.get(k) if d.get(k) is not None else 0 for _, d in move_rows]
        bars = ax.bar(x + (i - 1.5) * width, vals, width, label=f"Δk{k}", color=colors[i])
        for b, v in zip(bars, vals):
            if abs(v) < NOISE_FLOOR_PASSK:
                b.set_alpha(0.3)
                b.set_hatch("//")
    ax.axhline(NOISE_FLOOR_PASSK, color="#888", linestyle="--", linewidth=1)
    ax.axhline(-NOISE_FLOOR_PASSK, color="#888", linestyle="--", linewidth=1)
    ax.text(len(names) - 0.5, NOISE_FLOOR_PASSK, f" heuristic noise floor (±{NOISE_FLOOR_PASSK})",
            fontsize=8, color="#888", va="bottom", ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Δ pass@k vs untrained student")
    ax.set_title(f"Pass@k movement vs untrained student ({dataset.replace('.jsonl','')})\n"
                 "hatched/faded bars are BELOW the heuristic noise floor -- not reported as real")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(outdir, "passk_movement_vs_student.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/passk_hparam_sweep.csv")
    ap.add_argument("--baselines_csv", default="Results/passK/passk_baselines.csv")
    ap.add_argument("--dataset", default="math_eval.jsonl")
    ap.add_argument("--out", default="results/passk_movement_report.md")
    ap.add_argument("--outdir", default="Results/passK")
    args = ap.parse_args()

    if not os.path.isfile(args.csv):
        raise SystemExit(f"{args.csv} not found -- run scripts/passk_eval_sweep_checkpoints.sh first")
    rows = load_csv(args.csv)
    base_rows = load_csv(args.baselines_csv)

    student_k = {k: passk_at(base_rows, "Qwen/Qwen3-0.6B", args.dataset, k) for k in K_OF_INTEREST}
    teacher_k = {k: passk_at(base_rows, "Qwen/Qwen3-8B", args.dataset, k) for k in K_OF_INTEREST}

    L = []
    L.append(f"# Pass@k movement analysis ({args.dataset.replace('.jsonl','')})\n")
    L.append(f"> **Noise floor caveat:** no repeated seeds/evals anywhere in this sweep -- "
             f"there is no MEASURED noise floor for pass@k. `NOISE_FLOOR_PASSK="
             f"{NOISE_FLOOR_PASSK}` below is a HEURISTIC by analogy to the ~0.05 BE gap "
             f"already treated as noise-level elsewhere in this sweep, not a validated bound. "
             f"Deltas below it are shown but NOT reported as real movement.\n")
    L.append(f"Student (untrained 0.6B) pass@k: " +
             ", ".join(f"k{k}={student_k[k]}" for k in K_OF_INTEREST if student_k[k] is not None))
    L.append(f"\n\nTeacher (8B) pass@k: " +
             ", ".join(f"k{k}={teacher_k[k]}" for k in K_OF_INTEREST if teacher_k[k] is not None) + "\n")

    # -------- 1. ranking by delta vs student, gated by noise floor --------
    L.append("\n## 1. Ranked by movement vs untrained student (delta = checkpoint − student)\n")
    L.append(f"![pass@k movement vs student](../Results/passK/passk_movement_vs_student.png)\n")
    move_rows = []
    for frag in BEST_BE:
        vals = {k: passk_at(rows, frag, args.dataset, k) for k in K_OF_INTEREST}
        if all(v is None for v in vals.values()):
            continue
        deltas = {k: (round(vals[k] - student_k[k], 4) if vals[k] is not None and student_k[k] is not None else None)
                  for k in K_OF_INTEREST}
        move_rows.append((frag, deltas))
    move_rows.sort(key=lambda t: -(t[1].get(8) or -9))

    significant = [(f, d) for f, d in move_rows
                   if any(v is not None and abs(v) >= NOISE_FLOOR_PASSK for v in d.values())]
    below_floor = [f for f, d in move_rows if (f, d) not in significant]

    L.append(f"\n**Checkpoints with ≥1 delta clearing the heuristic noise floor "
             f"(±{NOISE_FLOOR_PASSK}):**\n")
    L.append("| checkpoint | Δk2 | Δk4 | Δk8 | Δk16 |")
    L.append("|---|---|---|---|---|")
    for frag, d in significant:
        L.append(f"| {frag} | {d.get(2)} | {d.get(4)} | {d.get(8)} | {d.get(16)} |")
    if below_floor:
        L.append(f"\n_{len(below_floor)} checkpoint(s) had no delta clearing the floor at any "
                 f"k — not listed as significant: {', '.join(below_floor)}_\n")

    # -------- 2. per-group: did BE move with pass@k8/16? --------
    L.append("\n\n## 2. Per-group: did BE and pass@k8/k16 move together?\n")
    L.append("range = max−min across the group's checkpoints. `pass@k range` compared "
             f"against the same heuristic floor ({NOISE_FLOOR_PASSK}); BE range compared "
             "against the ~0.05 BE floor used elsewhere in this sweep.\n")
    L.append("| group | BE range | passk_k8 range | passk_k16 range | correlated? |")
    L.append("|---|---|---|---|---|")
    for gname, frags in GROUPS.items():
        bes = [BEST_BE[f] for f in frags if f in BEST_BE]
        k8s = [v for v in (passk_at(rows, f, args.dataset, 8) for f in frags) if v is not None]
        k16s = [v for v in (passk_at(rows, f, args.dataset, 16) for f in frags) if v is not None]
        be_range = round(max(bes) - min(bes), 3) if len(bes) > 1 else None
        k8_range = round(max(k8s) - min(k8s), 4) if len(k8s) > 1 else None
        k16_range = round(max(k16s) - min(k16s), 4) if len(k16s) > 1 else None
        verdict = "?"
        if be_range is not None and k8_range is not None:
            be_big = be_range > 0.05
            pk_big = (k8_range or 0) >= NOISE_FLOOR_PASSK or (k16_range or 0) >= NOISE_FLOOR_PASSK
            verdict = "YES (moved together)" if be_big == pk_big else \
                      "NO -- BE moved, pass@k didn't" if be_big else "NO -- pass@k moved, BE didn't"
        L.append(f"| {gname} | {be_range} | {k8_range} | {k16_range} | {verdict} |")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"wrote {args.out}")

    os.makedirs(args.outdir, exist_ok=True)
    plot_movement(move_rows, args.outdir, args.dataset)


if __name__ == "__main__":
    main()
