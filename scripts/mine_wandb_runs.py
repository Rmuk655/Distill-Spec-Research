#!/usr/bin/env python
"""
mine_wandb_runs.py — offline insight-mining across EVERY logged W&B run.

WHY: instead of tuning one hyperparameter at a time by hand ("graduate student
descent"), pull the entire run history through the W&B API once and let the
data say -- objectively, across all runs at once -- which knobs actually
matter, what predicts collapse, how big the noise floor is, and whether any of
the rejected/collapsed objectives (enrich, traversal, nss, depth_weight, ...)
ever had salvageable value.

IMPORTANT CAVEAT (this is the real intellectual content, printed in the report
too): these runs are an OBSERVATIONAL dataset, not a designed experiment. If
high LR was always paired with short warmup, no method here can separate their
effects -- the confound is baked in. So the mining gives ASSOCIATION (where to
look, what correlates with good/bad), not clean CAUSAL attribution. Use it to
decide which few CONTROLLED sweeps are worth running, not as a substitute for
them.

USAGE (run on the cluster, where W&B auth already lives):
    python scripts/mine_wandb_runs.py                       # writes Results/wandb_mining_report.md
    python scripts/mine_wandb_runs.py --collapse_history    # also pull per-step history (slower)
    python scripts/mine_wandb_runs.py --project distillspec-pipeline --out Results/mining.md

Needs: wandb, pandas. sklearn optional (falls back to correlation-only importance).
"""
import argparse
import math
import os
import sys
from collections import defaultdict

# Hyperparameter config keys we care about (superset -- missing ones become NaN,
# handling the schema drift across weeks of runs where new flags appeared).
NUMERIC_HPARAMS = [
    "lr", "warmup_steps", "lr_min_ratio", "weight_decay",
    "prefix_teacher_topk", "prefix_M", "prefix_aux_weight",
    "prefix_anneal_steps", "prefix_root_spacing", "K", "L", "steps",
]
CATEGORICAL_HPARAMS = ["loss", "prefix_objective", "prefix_aux", "draft", "teacher"]

# Candidate summary keys for "best block efficiency" -- naming drifted, try all.
BEST_BE_KEYS = ["val/best_block_eff", "best_val_block_eff", "val/block_eff",
                "best_block_eff"]


def get_best_be(summary):
    for k in BEST_BE_KEYS:
        v = summary.get(k)
        if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
            return float(v)
    return None


def flatten_runs(runs):
    import pandas as pd
    rows = []
    for r in runs:
        cfg = {k: r.config.get(k) for k in NUMERIC_HPARAMS + CATEGORICAL_HPARAMS}
        summ = dict(r.summary)
        best_be = get_best_be(summ)
        final_be = summ.get("val/block_eff")
        rows.append({
            "name": r.name, "id": r.id, "state": r.state,
            "best_be": best_be,
            "final_be": float(final_be) if isinstance(final_be, (int, float)) else None,
            **cfg,
        })
    df = pd.DataFrame(rows)
    # log10(lr) is the meaningful scale for a tree/correlation model, not raw lr.
    if "lr" in df:
        df["log10_lr"] = df["lr"].apply(
            lambda x: math.log10(x) if isinstance(x, (int, float)) and x and x > 0 else None)
    return df


def loss_family_table(df):
    """Per (loss, objective, pair) group: n, best/median BE, collapse rate.
    Answers 'did the rejected objectives ever have value.'"""
    import pandas as pd
    def pair(row):
        d, t = str(row.get("draft")), str(row.get("teacher"))
        d = d.split("/")[-1] if d and d != "None" else "?"
        t = t.split("/")[-1] if t and t != "None" else "?"
        return f"{d}/{t}"
    df = df.copy()
    df["pair"] = df.apply(pair, axis=1)
    df["group"] = (df["loss"].astype(str) + " / " +
                   df["prefix_objective"].astype(str).replace("None", "-") + " / " +
                   df["pair"])
    out = []
    for g, sub in df.groupby("group"):
        be = sub["best_be"].dropna()
        if be.empty:
            continue
        collapsed = ((sub["best_be"] < 2.0) |
                     ((sub["final_be"].notna()) & (sub["final_be"] < 0.7 * sub["best_be"]))).sum()
        out.append({
            "group": g, "n": len(sub), "best_BE": round(be.max(), 3),
            "median_BE": round(be.median(), 3),
            "collapse_rate": f"{collapsed}/{len(sub)}",
        })
    return pd.DataFrame(out).sort_values("best_BE", ascending=False)


def param_importance(df, subset_mask, target="best_be"):
    """RandomForest feature importance on a clean numeric subset (default: the
    prob-objective 0.6B/8B sweep). Falls back to |correlation| if no sklearn."""
    import pandas as pd
    sub = df[subset_mask].copy()
    feats = [c for c in ["log10_lr", "warmup_steps", "lr_min_ratio", "weight_decay",
                         "prefix_teacher_topk", "prefix_M", "prefix_aux_weight",
                         "prefix_anneal_steps"] if c in sub.columns]
    sub = sub.dropna(subset=[target])
    X = sub[feats].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = sub[target]
    if len(sub) < 8:
        return None, feats, len(sub), "too few runs (<8) for a meaningful fit"
    # Drop zero-variance features (a param that never varied can't be 'important').
    varying = [f for f in feats if X[f].nunique() > 1]
    X = X[varying]
    try:
        from sklearn.ensemble import RandomForestRegressor
        rf = RandomForestRegressor(n_estimators=300, random_state=0)
        rf.fit(X, y)
        imp = sorted(zip(varying, rf.feature_importances_), key=lambda t: -t[1])
        return imp, varying, len(sub), "random_forest"
    except ImportError:
        corr = sorted(((f, abs(X[f].corr(y))) for f in varying),
                      key=lambda t: -(t[1] if not math.isnan(t[1]) else 0))
        return corr, varying, len(sub), "abs_correlation (sklearn not installed)"


def marginal_view(df, subset_mask):
    """Coordinate-descent view computed across ALL runs at once: for each
    hyperparam, mean best_be grouped by its value."""
    sub = df[subset_mask]
    out = {}
    for f in ["log10_lr", "warmup_steps", "lr_min_ratio", "weight_decay",
              "prefix_teacher_topk", "prefix_M", "prefix_aux_weight", "prefix_anneal_steps"]:
        if f not in sub.columns:
            continue
        g = sub.dropna(subset=["best_be", f]).groupby(f)["best_be"]
        if g.ngroups > 1:
            out[f] = [(round(k, 4) if isinstance(k, float) else k, round(v, 3), n)
                      for (k, v), n in zip(g.mean().items(), g.size())]
    return out


def noise_floor(df):
    """Near-duplicate configs -> spread in best_be = a lower bound on run-to-run
    noise. If this is comparable to your 'wins', those wins aren't real."""
    key_cols = [c for c in ["loss", "prefix_objective", "log10_lr", "warmup_steps",
                            "lr_min_ratio", "weight_decay", "prefix_teacher_topk",
                            "prefix_M", "prefix_aux_weight", "prefix_anneal_steps",
                            "draft", "teacher"] if c in df.columns]
    dup_groups = []
    for _, sub in df.dropna(subset=["best_be"]).groupby(
            [df[c].astype(str) for c in key_cols]):
        if len(sub) > 1:
            be = sub["best_be"]
            dup_groups.append((sub["name"].iloc[0], len(sub),
                               round(be.max() - be.min(), 3), round(be.std(), 4)))
    return dup_groups


def collapse_timing(runs, df, collapse_be):
    """For runs that collapsed, when did grad_norm first flatline / block_eff
    first crash? Pulls per-step history (slow) -- only with --collapse_history."""
    import pandas as pd
    collapsed_names = set(df[(df["best_be"].notna()) &
                             (df["best_be"] < collapse_be)]["name"])
    rows = []
    for r in runs:
        if r.name not in collapsed_names:
            continue
        try:
            h = r.history(keys=["train/grad_norm", "val/block_eff", "train/step"],
                          samples=2000)
        except Exception:
            continue
        if h is None or h.empty:
            continue
        gn = h.get("train/grad_norm")
        first_dead = None
        if gn is not None:
            # first step where grad_norm stays ~0 for a stretch (dead-gradient onset)
            dead = (gn.fillna(0).abs() < 1e-6)
            run_len = 0
            for i, d in enumerate(dead):
                run_len = run_len + 1 if d else 0
                if run_len >= 20:
                    first_dead = int(h["train/step"].iloc[i]) if "train/step" in h else i
                    break
        rows.append((r.name, first_dead))
    return rows


def write_report(path, df, runs, args):
    import pandas as pd
    lines = []
    W = lines.append
    W("# W&B run-mining report\n")
    W(f"Mined **{len(df)} runs** from `{args.entity}/{args.project}`.\n")
    W("> **Caveat (read first):** these are OBSERVATIONAL runs, not a designed "
      "experiment. Confounded hyperparameters can't be separated here — this "
      "gives *association* (where to look), not *causation*. Use it to pick "
      "which controlled sweeps are worth running, not to replace them.\n")

    W("## 1. Loss-family comparison — did the rejected objectives have value?\n")
    lf = loss_family_table(df)
    W("| loss / objective / pair | n | best BE | median BE | collapse rate |")
    W("|---|---|---|---|---|")
    for _, r in lf.iterrows():
        W(f"| {r['group']} | {r['n']} | {r['best_BE']} | {r['median_BE']} | {r['collapse_rate']} |")
    W("")

    W("## 2. Hyperparameter importance (prob-objective 0.6B/8B subset)\n")
    mask = ((df["loss"] == "prefix_overlap") & (df["prefix_objective"] == "prob"))
    imp, feats, n, method = param_importance(df, mask)
    if imp is None:
        W(f"_{method}_\n")
    else:
        W(f"Fit on {n} runs via {method}. Ranked most→least important:\n")
        W("| feature | importance |")
        W("|---|---|")
        for f, v in imp:
            W(f"| {f} | {round(v, 4)} |")
    W("")

    W("## 3. Marginal effect per hyperparam (across all prob runs)\n")
    mv = marginal_view(df, mask)
    for f, rows in mv.items():
        W(f"**{f}** — (value → mean best_BE, n):")
        W("  " + "  ".join(f"{k}:{v}(n={n})" for k, v, n in rows))
        W("")

    W("## 4. Noise floor (near-duplicate configs)\n")
    dups = noise_floor(df)
    if not dups:
        W("_No repeated configs found — every run is a single seed, so the "
          "run-to-run noise floor is UNKNOWN. Any 'win' smaller than a real "
          "seed-to-seed spread is unverified. Strongly consider 2–3 seeds on "
          "the current best config before trusting its margin._\n")
    else:
        W("| example run | n dups | BE range | BE std |")
        W("|---|---|---|---|")
        for name, k, rng, std in dups:
            W(f"| {name} | {k} | {rng} | {std} |")
    W("")

    if args.collapse_history:
        W("## 5. Collapse timing (dead-gradient onset)\n")
        ct = collapse_timing(runs, df, args.collapse_be)
        W("| collapsed run | first dead-gradient step |")
        W("|---|---|")
        for name, step in ct:
            W(f"| {name} | {step if step is not None else 'n/a'} |")
        W("\n_If dead-gradient onset consistently precedes the final crash, an "
          "ASHA-style auto-kill at that step would save the wasted compute._\n")

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[mine] wrote {path} ({len(df)} runs)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", default="rmukund16-indian-institute-of-technology-hyderabad")
    ap.add_argument("--project", default="distillspec-pipeline")
    ap.add_argument("--out", default="Results/wandb_mining_report.md")
    ap.add_argument("--collapse_history", action="store_true",
                    help="also pull per-step history for dead-gradient timing (slower)")
    ap.add_argument("--collapse_be", type=float, default=2.0,
                    help="best_be below this = 'collapsed' for the timing analysis")
    args = ap.parse_args()

    try:
        import wandb  # noqa
        import pandas  # noqa
    except ImportError as e:
        sys.exit(f"needs wandb + pandas: {e}")

    import wandb
    api = wandb.Api()
    runs = list(api.runs(f"{args.entity}/{args.project}"))
    if not runs:
        sys.exit("no runs found — check --entity/--project")
    df = flatten_runs(runs)
    write_report(args.out, df, runs, args)


if __name__ == "__main__":
    main()
