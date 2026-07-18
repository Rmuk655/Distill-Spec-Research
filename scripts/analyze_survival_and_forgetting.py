#!/usr/bin/env python
"""
analyze_survival_and_forgetting.py — two follow-ups from the mining report:

  1. SURVIVAL CURVES: bucket runs by LR, plot "fraction still healthy by step
     N" per bucket (Kaplan-Meier-style, using each run's time_to_collapse as
     the event time, censored at its final step if it never collapsed).
     Reframes the LR-vs-collapse story as a proper survival-analysis chart
     instead of a table of single time_to_collapse numbers.

  2. FORGETTING vs aux_weight: does the CE-anchor work by reducing forgetting
     (Lopez-Paz & Ranzato backward-transfer), not just by regularizing
     gradients generally? Pulls val/forgetting trajectories for the CE-anneal
     sweep (aux_weight=0/0.5/1.0/2.0) and plots mean/final forgetting against
     aux_weight -- a MECHANISM check, not just another "did BE move" table.

CAVEATS (read before citing either chart):
  - Survival curves here are an EMPIRICAL step-function, not a proper
    Kaplan-Meier estimator with confidence bands -- small per-bucket N (often
    1-8 runs), so read shapes, not precise probabilities.
  - Forgetting-vs-aux_weight is correlational over ~4-6 runs, one seed each --
    a real hypothesis worth a controlled follow-up, not a proven mechanism.

USAGE (needs wandb + pandas + numpy; matplotlib for the charts):
    python scripts/analyze_survival_and_forgetting.py
    python scripts/analyze_survival_and_forgetting.py --entity ... --project ...
"""
import argparse
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTDIR_DEFAULT = "Results/passK"

# LR buckets for the survival chart -- prob 0.6B/8B family, run-name fragment match.
LR_BUCKETS = {
    "healthy (1e-6..1e-5)": ["lr1e-06", "lr2e-06", "lr3e-06", "lr5e-06", "lr7e-06", "lr1e-05"],
    "mild (1e-4)": ["lr0.0001", "lr1e-04"],
    "collapse (>=3e-4)": ["lr0.0003", "lr0.003", "lr0.01", "lr3e-04", "lr3e-03", "lr1e-02"],
}

# CE-anneal sweep -- matched by CONFIG VALUES, not run name. Verified live
# (2026-07-17): the W&B name slug does NOT encode weight_decay, lr_min_ratio,
# warmup_steps, or top-k -- 8 different sweep variants (wd/lrmin/warmup/topk)
# all share the IDENTICAL display name
# "Qwen3-0.6B__Qwen3-8B__prefix_overlap_prob_singleM4_L8_lr1e-05_math_hard_s123".
# Name-based matching is fundamentally ambiguous for the no-anchor baseline;
# config values are exact ground truth regardless of naming.
CEANNEAL_CONFIGS = [
    # (aux_weight, dict of exact config values that must all match)
    (0.0, {"lr": 1e-5, "warmup_steps": 125, "lr_min_ratio": 0.1, "weight_decay": 0.01,
           "prefix_teacher_topk": 0, "prefix_aux_weight": 0.0}),
    (0.5, {"lr": 1e-5, "prefix_aux_weight": 0.5, "prefix_anneal_steps": 3000}),
    (1.0, {"lr": 1e-5, "prefix_aux_weight": 1.0, "prefix_anneal_steps": 3000}),
    (2.0, {"lr": 1e-5, "prefix_aux_weight": 2.0, "prefix_anneal_steps": 3000}),
]
# every candidate must ALSO match this, regardless of which aux_weight bucket
CEANNEAL_BASE_FILTER = {"loss": "prefix_overlap", "prefix_objective": "prob"}


def config_matches(config, required):
    for k, v in required.items():
        cv = config.get(k)
        if cv is None:
            return False
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            try:
                if abs(float(cv) - float(v)) > 1e-9:
                    return False
            except (TypeError, ValueError):
                return False
        elif cv != v:
            return False
    return True


def get_runs(entity, project):
    import wandb
    return list(wandb.Api().runs(f"{entity}/{project}"))


def run_time_to_collapse_and_final_step(run):
    """(time_to_collapse or None, final_step) from this run's val/block_eff history."""
    import numpy as np
    try:
        h = run.history(keys=["val/block_eff", "train/step"], samples=6000)
    except Exception:
        return None, None
    if h is None or h.empty or "val/block_eff" not in h:
        return None, None
    vb = h["val/block_eff"].dropna()
    if vb.empty:
        return None, None
    step = h.get("train/step")
    vs = step.to_numpy()[vb.index] if step is not None else np.arange(len(vb))
    va = vb.to_numpy()
    run_max = np.maximum.accumulate(va)
    crashed = np.where(va < 0.7 * run_max)[0]
    ttc = int(vs[crashed[0]]) if crashed.size else None
    final_step = int(vs[-1])
    return ttc, final_step


def run_forgetting_series(run):
    try:
        h = run.history(keys=["val/forgetting", "train/step"], samples=6000)
    except Exception:
        return None
    if h is None or h.empty or "val/forgetting" not in h:
        return None
    s = h["val/forgetting"].dropna()
    return s if not s.empty else None


def plot_survival(runs, outdir):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = {"healthy (1e-6..1e-5)": "#008300", "mild (1e-4)": "#eda100", "collapse (>=3e-4)": "#e34948"}
    any_data = False
    for bucket, frags in LR_BUCKETS.items():
        events = []  # (time_to_collapse_or_final_step, collapsed_bool)
        for r in runs:
            if not any(f in r.name for f in frags):
                continue
            if "prefix_overlap" not in str(r.config.get("loss", "")):
                continue
            if str(r.config.get("prefix_objective", "")) != "prob":
                continue
            ttc, final_step = run_time_to_collapse_and_final_step(r)
            if final_step is None:
                continue
            events.append((ttc if ttc is not None else final_step, ttc is not None))
        if not events:
            continue
        any_data = True
        max_step = max(t for t, _ in events)
        xs = list(range(0, max_step + 100, 100))
        ys = []
        n = len(events)
        for x in xs:
            still_healthy = sum(1 for t, collapsed in events if not (collapsed and t <= x))
            ys.append(still_healthy / n)
        ax.step(xs, ys, where="post", label=f"{bucket} (n={n})", color=colors.get(bucket, "#888"), linewidth=2)
    if not any_data:
        print("[survival] no matching runs found -- check LR_BUCKETS fragments against actual run names")
        plt.close(fig)
        return
    ax.set_xlabel("training step")
    ax.set_ylabel("fraction of runs still healthy")
    ax.set_title("Empirical survival curves: fraction healthy vs step, by LR bucket\n"
                 "(prefix_overlap/prob, 0.6B/8B, math_hard) -- step-function, small N per bucket", fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = os.path.join(outdir, "survival_curves_by_lr.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def plot_forgetting_vs_auxweight(runs, outdir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    points = []  # (aux_weight, mean_forgetting, final_forgetting, run_name)
    for r in runs:
        if not config_matches(r.config, CEANNEAL_BASE_FILTER):
            continue
        for aw, required in CEANNEAL_CONFIGS:
            if config_matches(r.config, required):
                s = run_forgetting_series(r)
                if s is None:
                    continue
                points.append((aw, float(s.mean()), float(s.iloc[-1]), r.name))
                break
    if not points:
        print("[forgetting] no matching runs found -- check CEANNEAL_CONFIGS values against actual run.config")
        plt.close(fig)
        return
    points.sort()
    aws = [p[0] for p in points]
    means = [p[1] for p in points]
    finals = [p[2] for p in points]

    ax1.plot(aws, means, "-o", color="#2a78d6", linewidth=2, markersize=8)
    for aw, m in zip(aws, means):
        ax1.annotate(f"{m:.3f}", (aw, m), textcoords="offset points", xytext=(0, 6), fontsize=8, ha="center")
    ax1.set_xlabel("aux_weight (CE-anchor strength)")
    ax1.set_ylabel("mean val/forgetting over training")
    ax1.set_title("Mean forgetting vs anchor strength")
    ax1.grid(alpha=0.3)

    ax2.plot(aws, finals, "-s", color="#e34948", linewidth=2, markersize=8)
    for aw, f in zip(aws, finals):
        ax2.annotate(f"{f:.3f}", (aw, f), textcoords="offset points", xytext=(0, 6), fontsize=8, ha="center")
    ax2.set_xlabel("aux_weight (CE-anchor strength)")
    ax2.set_ylabel("final val/forgetting")
    ax2.set_title("Final-step forgetting vs anchor strength")
    ax2.grid(alpha=0.3)

    fig.suptitle("Does the CE-anchor work by reducing forgetting? (prefix_overlap/prob, lr=1e-5/wu20)")
    fig.tight_layout()
    path = os.path.join(outdir, "forgetting_vs_aux_weight.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")
    print("aux_weight -> (mean_forgetting, final_forgetting):",
          {aw: (round(m, 4), round(f, 4)) for aw, m, f, _ in points})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", default="rmukund16-indian-institute-of-technology-hyderabad")
    ap.add_argument("--project", default="distillspec-pipeline")
    ap.add_argument("--outdir", default=OUTDIR_DEFAULT)
    args = ap.parse_args()

    try:
        import wandb, numpy  # noqa
    except ImportError as e:
        raise SystemExit(f"needs wandb + numpy: {e}")

    os.makedirs(args.outdir, exist_ok=True)
    runs = get_runs(args.entity, args.project)
    plot_survival(runs, args.outdir)
    plot_forgetting_vs_auxweight(runs, args.outdir)


if __name__ == "__main__":
    main()
