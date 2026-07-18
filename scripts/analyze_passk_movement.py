#!/usr/bin/env python
"""
analyze_passk_movement.py — two questions the charts alone don't answer directly:

  1. Which checkpoint moved pass@k (k=2,4,8,16 especially) the MOST relative to
     the untrained Qwen3-0.6B student baseline? Ranks every swept checkpoint by
     delta-vs-student at each k -- but only REPORTS a delta as real movement if
     it clears a per-k noise-floor threshold (see derive_noise_floor() below).
  2. Did BE and pass@k8/k16 move in a correlated fashion across the sweep, or
     did some group move BE without moving pass@k (or vice versa)? Per sweep
     group: BE range vs pass@k8/k16 range.

NOISE FLOOR -- DERIVED FROM DATA, NOT ASSUMED: there are no repeated seeds and
no per-prompt data saved (compute_passk_for_dataset aggregates to a single mean
pass@k per (model,dataset,k) before it's ever written out), so a textbook
repeated-measurement or bootstrap noise floor isn't available. But we DON'T
need to borrow the BE heuristic either: the low-LR `prob` cluster
(NOISE_CLUSTER_PROB below, lr=1e-6..1e-5) was ALREADY established as
statistically indistinguishable in block_eff (~0.11 spread) earlier in this
sweep. If BE is flat there, pass@k should be too, for the same underlying
reason -- so the SPREAD of pass@k across that cluster, at each k, is a real
number computed from data already in hand, not an assumption. This script
derives floor[k] = 2 x std(pass@k across the cluster) per k, and
uses that (falling back to a small default only if the cluster data isn't in
the CSV yet). Caveat this DOES still conflate two things -- true measurement
noise vs. a possible tiny genuine LR effect within that "flat" region -- so
treat it as an upper-bound-ish proxy, not a textbook confidence interval.

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

# The already-established "flat" low-LR prob cluster (~0.11 BE spread, called
# noise-level earlier in this sweep) -- used to DERIVE a per-k pass@k noise
# floor from data, instead of assuming one. See module docstring.
NOISE_CLUSTER_PROB = [
    "po_prob_lr1e-6_wu20", "po_prob_lr3e-6_wu20", "po_prob_lr7e-6_wu20",
    "po_prob_lr1e-5_wu20_lrmin0.1_wd0.01",
]
DEFAULT_NOISE_FLOOR = 0.05  # fallback only, if the cluster isn't in the CSV yet

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


def derive_noise_floor(rows, dataset):
    """Per-k noise floor = 2 x std(pass@k) across NOISE_CLUSTER_PROB -- an
    empirical proxy from data already in hand, not a borrowed constant. Falls
    back to DEFAULT_NOISE_FLOOR per-k if fewer than 3 cluster members have data
    (too few points to trust a std estimate)."""
    import statistics
    floor = {}
    for k in K_OF_INTEREST:
        vals = [v for v in (passk_at(rows, f, dataset, k) for f in NOISE_CLUSTER_PROB) if v is not None]
        if len(vals) >= 3:
            floor[k] = round(2 * statistics.stdev(vals), 4)
        else:
            floor[k] = DEFAULT_NOISE_FLOOR
    return floor


def plot_movement(move_rows, outdir, dataset, floor, chart_name):
    """Grouped bar chart: Δk2/Δk4/Δk8/Δk16 vs student per checkpoint, with the
    DERIVED per-k noise-floor band shaded -- only bars clearing it are solid;
    below-floor bars are drawn hatched/faded so they read as 'not significant'."""
    import numpy as np
    names = [frag for frag, _ in move_rows]
    ks = K_OF_INTEREST
    colors = ["#2a78d6", "#008300", "#e34948", "#eda100"]
    fig, ax = plt.subplots(figsize=(max(10, len(names) * 0.6), 6.5))
    x = np.arange(len(names))
    width = 0.2
    for i, k in enumerate(ks):
        vals = [d.get(k) if d.get(k) is not None else 0 for _, d in move_rows]
        bars = ax.bar(x + (i - 1.5) * width, vals, width, label=f"Δk{k}", color=colors[i])
        for b, v in zip(bars, vals):
            if abs(v) < floor[k]:
                b.set_alpha(0.3)
                b.set_hatch("//")
    # per-k floor lines (different per k, so no single flat band)
    for i, k in enumerate(ks):
        ax.axhline(floor[k], color=colors[i], linestyle=":", linewidth=1, alpha=0.6)
        ax.axhline(-floor[k], color=colors[i], linestyle=":", linewidth=1, alpha=0.6)
    floor_str = ", ".join(f"k{k}=±{floor[k]}" for k in ks)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Δ pass@k vs untrained student")
    ax.set_title(f"Pass@k movement vs untrained student ({dataset.replace('.jsonl','')})\n"
                 f"noise floor per k (2×std across the flat low-LR cluster): {floor_str}\n"
                 "hatched/faded bars are below that checkpoint-and-k's floor -- not reported as real",
                 fontsize=9)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(outdir, chart_name)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def run_for_dataset(rows, base_rows, dataset, out_path, outdir):
    """One dataset's full report + chart -- datasets are NEVER mixed (math_hard's
    checkpoints behave very differently on a saturated set like gsm8k vs. the
    sensitive math_eval vs. the low-ceiling olympiad, so each needs its own
    story, not an averaged-together one)."""
    student_k = {k: passk_at(base_rows, "Qwen/Qwen3-0.6B", dataset, k) for k in K_OF_INTEREST}
    teacher_k = {k: passk_at(base_rows, "Qwen/Qwen3-8B", dataset, k) for k in K_OF_INTEREST}
    if all(v is None for v in student_k.values()):
        print(f"  [skip] no baseline data for {dataset} in {out_path!r} run -- nothing to report")
        return
    floor = derive_noise_floor(rows, dataset)
    chart_name = f"passk_movement_vs_student_{dataset.replace('.jsonl', '')}.png"

    L = []
    L.append(f"# Pass@k movement analysis ({dataset.replace('.jsonl','')})\n")
    L.append(f"> **Noise floor — derived from data, not assumed:** no repeated seeds and no "
             f"per-prompt data survive into the CSV, so a textbook repeated-measurement/"
             f"bootstrap floor isn't directly available. Instead: the low-LR `prob` cluster "
             f"(`{', '.join(NOISE_CLUSTER_PROB)}`) was already established as statistically "
             f"indistinguishable in block_eff (~0.11 spread) earlier in this sweep — if BE is "
             f"flat there, pass@k should be too. The per-k floor below is `2×std(pass@k)` "
             f"across that cluster, computed from data already in hand: "
             f"{', '.join(f'k{k}=±{floor[k]}' for k in K_OF_INTEREST)}. This still conflates "
             f"true measurement noise with a possible tiny genuine LR effect in that region — "
             f"an upper-bound-ish proxy, not a textbook CI. Deltas below it are shown but NOT "
             f"reported as real movement.\n")
    L.append(f"Student (untrained 0.6B) pass@k: " +
             ", ".join(f"k{k}={student_k[k]}" for k in K_OF_INTEREST if student_k[k] is not None))
    L.append(f"\n\nTeacher (8B) pass@k: " +
             ", ".join(f"k{k}={teacher_k[k]}" for k in K_OF_INTEREST if teacher_k[k] is not None) + "\n")

    # -------- 1. ranking by delta vs student, gated by noise floor --------
    L.append("\n## 1. Ranked by movement vs untrained student (delta = checkpoint − student)\n")
    L.append(f"![pass@k movement vs student](../Results/passK/{chart_name})\n")
    move_rows = []
    for frag in BEST_BE:
        vals = {k: passk_at(rows, frag, dataset, k) for k in K_OF_INTEREST}
        if all(v is None for v in vals.values()):
            continue
        deltas = {k: (round(vals[k] - student_k[k], 4) if vals[k] is not None and student_k[k] is not None else None)
                  for k in K_OF_INTEREST}
        move_rows.append((frag, deltas))
    move_rows.sort(key=lambda t: -(t[1].get(8) or -9))

    significant = [(f, d) for f, d in move_rows
                   if any(v is not None and abs(v) >= floor[k] for k, v in d.items())]
    below_floor = [f for f, d in move_rows if (f, d) not in significant]

    L.append(f"\n**Checkpoints with ≥1 delta clearing its k's derived noise floor:**\n")
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
             f"against the same derived per-k floor (k8=±{floor[8]}, k16=±{floor[16]}); "
             "BE range compared against the ~0.05 BE floor used elsewhere in this sweep.\n")
    L.append("| group | BE range | passk_k8 range | passk_k16 range | correlated? |")
    L.append("|---|---|---|---|---|")
    for gname, frags in GROUPS.items():
        bes = [BEST_BE[f] for f in frags if f in BEST_BE]
        k8s = [v for v in (passk_at(rows, f, dataset, 8) for f in frags) if v is not None]
        k16s = [v for v in (passk_at(rows, f, dataset, 16) for f in frags) if v is not None]
        be_range = round(max(bes) - min(bes), 3) if len(bes) > 1 else None
        k8_range = round(max(k8s) - min(k8s), 4) if len(k8s) > 1 else None
        k16_range = round(max(k16s) - min(k16s), 4) if len(k16s) > 1 else None
        verdict = "?"
        if be_range is not None and k8_range is not None:
            be_big = be_range > 0.05
            pk_big = (k8_range or 0) >= floor[8] or (k16_range or 0) >= floor[16]
            verdict = "YES (moved together)" if be_big == pk_big else \
                      "NO -- BE moved, pass@k didn't" if be_big else "NO -- pass@k moved, BE didn't"
        L.append(f"| {gname} | {be_range} | {k8_range} | {k16_range} | {verdict} |")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"wrote {out_path}")

    os.makedirs(outdir, exist_ok=True)
    plot_movement(move_rows, outdir, dataset, floor, chart_name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/passk_hparam_sweep.csv")
    ap.add_argument("--baselines_csv", default="Results/passK/passk_baselines.csv")
    ap.add_argument("--datasets", default="math_eval.jsonl,gsm8k_eval.jsonl,olympiad_eval.jsonl",
                    help="comma list -- run separately, never mixed, one report+chart each")
    ap.add_argument("--out_prefix", default="results/passk_movement_report")
    ap.add_argument("--outdir", default="Results/passK")
    args = ap.parse_args()

    if not os.path.isfile(args.csv):
        raise SystemExit(f"{args.csv} not found -- run scripts/passk_eval_sweep_checkpoints.sh first")
    rows = load_csv(args.csv)
    base_rows = load_csv(args.baselines_csv)

    for dataset in [d.strip() for d in args.datasets.split(",")]:
        tag = dataset.replace(".jsonl", "")
        out_path = f"{args.out_prefix}_{tag}.md"
        print(f"=== {dataset} ===")
        run_for_dataset(rows, base_rows, dataset, out_path, args.outdir)


if __name__ == "__main__":
    main()
