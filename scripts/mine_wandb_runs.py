#!/usr/bin/env python
"""
mine_wandb_runs.py — offline insight-mining across EVERY logged W&B run.

WHY: instead of tuning one hyperparameter at a time by hand ("graduate student
descent"), pull the entire run history through the W&B API once and let the
data say -- objectively, across all runs at once -- which knobs actually
matter, what predicts collapse, how big the noise floor is, whether any of the
rejected objectives (enrich, traversal, nss, depth_weight, ...) had value, and
which training runs actually moved pass@k (even where block-eff regressed).

IMPORTANT CAVEAT (printed in the report too): these runs are an OBSERVATIONAL
dataset, not a designed experiment. Confounded hyperparameters can't be
separated here -- this gives ASSOCIATION (where to look), not CAUSATION. Use it
to pick which controlled sweeps are worth running, not to replace them.

GROUPING: every comparison is split by (loss, objective, draft/teacher pair,
train_dataset) so a math_hard 0.6B/8B run is never averaged against a gsm8k or
1.7B/32B run.

USAGE (run on the cluster where W&B auth lives):
    python scripts/mine_wandb_runs.py                    # cheap: config+summary only
    python scripts/mine_wandb_runs.py --history          # + loss/grad/pass@k trends (slower)
    python scripts/mine_wandb_runs.py --history --out Results/mining.md

Needs: wandb, pandas. sklearn optional (falls back to correlation-only importance).
"""
import argparse
import math
import os
import sys

NUMERIC_HPARAMS = [
    "lr", "warmup_steps", "lr_min_ratio", "weight_decay",
    "prefix_teacher_topk", "prefix_M", "prefix_aux_weight",
    "prefix_anneal_steps", "prefix_root_spacing", "K", "L", "steps",
]
CATEGORICAL_HPARAMS = ["loss", "prefix_objective", "prefix_aux", "draft",
                       "teacher", "train_dataset", "val_dataset"]

BEST_BE_KEYS = ["val/best_block_eff", "best_val_block_eff", "val/block_eff",
                "best_block_eff"]
PASSK_KS = [1, 2, 4, 8, 16, 32, 64]

# grad_norm is logged every 10 raw steps but is only nonzero on optimizer steps
# (grad_accum=8), i.e. every lcm(8,10)=40 raw steps -> ~3/4 of logged values are
# EXPECTED zeros, not collapse. So "dead gradient" must mean: no nonzero grad in
# a late window, not merely "many zeros". See grad-norm-zero-pattern memory.
GRAD_ACCUM = 8


def get_best_be(summary):
    for k in BEST_BE_KEYS:
        v = summary.get(k)
        if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
            return float(v)
    return None


def _short(x):
    x = str(x)
    return x.split("/")[-1] if x and x != "None" else "?"


def flatten_runs(runs):
    import pandas as pd
    rows = []
    for r in runs:
        cfg = {k: r.config.get(k) for k in NUMERIC_HPARAMS + CATEGORICAL_HPARAMS}
        summ = dict(r.summary)
        best_be = get_best_be(summ)
        final_be = summ.get("val/block_eff")
        row = {
            "name": r.name, "id": r.id, "state": r.state,
            "best_be": best_be,
            "final_be": float(final_be) if isinstance(final_be, (int, float)) else None,
            **cfg,
        }
        # final (last-logged) pass@k straight from summary -- cheap, no history call
        for k in PASSK_KS:
            for ds in ("math_val", "math_eval"):
                v = summ.get(f"passk_{ds}/k{k}")
                if isinstance(v, (int, float)):
                    row[f"passk_{ds}_k{k}"] = float(v)
        rows.append(row)
    df = pd.DataFrame(rows)
    if "lr" in df:
        df["log10_lr"] = df["lr"].apply(
            lambda x: math.log10(x) if isinstance(x, (int, float)) and x and x > 0 else None)
    return df


def group_key(df):
    """Split by loss / objective / pair / train_dataset so nothing gets crossed."""
    return (df["loss"].astype(str) + " / " +
            df["prefix_objective"].astype(str).replace("None", "-") + " / " +
            df["draft"].map(_short) + "→" + df["teacher"].map(_short) + " / " +
            df["train_dataset"].astype(str))


def loss_family_table(df):
    import pandas as pd
    df = df.copy()
    df["group"] = group_key(df)
    out = []
    for g, sub in df.groupby("group"):
        be = sub["best_be"].dropna()
        if be.empty:
            continue
        collapsed = ((sub["best_be"] < 2.0) |
                     ((sub["final_be"].notna()) & (sub["final_be"] < 0.7 * sub["best_be"]))).sum()
        out.append({"group": g, "n": len(sub), "best_BE": round(be.max(), 3),
                    "median_BE": round(be.median(), 3),
                    "collapse_rate": f"{collapsed}/{len(sub)}"})
    return pd.DataFrame(out).sort_values("best_BE", ascending=False)


def param_importance(df, mask, target="best_be"):
    import pandas as pd
    sub = df[mask].dropna(subset=[target]).copy()
    feats = [c for c in ["log10_lr", "warmup_steps", "lr_min_ratio", "weight_decay",
                         "prefix_teacher_topk", "prefix_M", "prefix_aux_weight",
                         "prefix_anneal_steps"] if c in sub.columns]
    if len(sub) < 8:
        return None, len(sub), "too few runs (<8) for a meaningful fit"
    X = sub[feats].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    varying = [f for f in feats if X[f].nunique() > 1]
    X, y = X[varying], sub[target]
    try:
        from sklearn.ensemble import RandomForestRegressor
        rf = RandomForestRegressor(n_estimators=300, random_state=0).fit(X, y)
        return sorted(zip(varying, rf.feature_importances_), key=lambda t: -t[1]), len(sub), "random_forest"
    except ImportError:
        corr = sorted(((f, abs(X[f].corr(y))) for f in varying),
                      key=lambda t: -(t[1] if not math.isnan(t[1]) else 0))
        return corr, len(sub), "abs_correlation (no sklearn)"


def marginal_view(df, mask):
    sub = df[mask]
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
    key_cols = [c for c in ["loss", "prefix_objective", "log10_lr", "warmup_steps",
                            "lr_min_ratio", "weight_decay", "prefix_teacher_topk",
                            "prefix_M", "prefix_aux_weight", "prefix_anneal_steps",
                            "draft", "teacher", "train_dataset"] if c in df.columns]
    groups = []
    for _, sub in df.dropna(subset=["best_be"]).groupby([df[c].astype(str) for c in key_cols]):
        if len(sub) > 1:
            be = sub["best_be"]
            groups.append((sub["name"].iloc[0], len(sub),
                           round(be.max() - be.min(), 3), round(be.std(), 4)))
    return groups


def history_analysis(runs, df):
    """One history pass per run -> training-health + pass@k movement.

    Training health (Rahul: loss should be smooth, grads alive-not-spiking):
      loss_cv       : std/|mean| of train/loss in the last 50% (lower = smoother)
      grad_late_max : max NONZERO grad_norm in the last 25% (~0 => dead/collapsed)
      grad_max      : max grad_norm over the run (spike magnitude)
    Pass@k movement (recent runs only): first->last delta of passk_math_val/k{8,16}
      and of val/block_eff, so we can surface runs where pass@k rose while BE fell.
    """
    import pandas as pd
    keys = (["train/loss", "train/grad_norm", "val/block_eff", "train/step"] +
            [f"passk_math_val/k{k}" for k in PASSK_KS])
    health, movement = [], []
    for r in runs:
        try:
            h = r.history(keys=keys, samples=5000)
        except Exception:
            continue
        if h is None or h.empty:
            continue

        loss = h.get("train/loss")
        gn = h.get("train/grad_norm")
        loss_cv = grad_late_max = grad_max = None
        if loss is not None and loss.notna().sum() > 10:
            lv = loss.dropna().to_numpy()
            tail = lv[len(lv) // 2:]
            m = abs(tail.mean())
            loss_cv = round(float(tail.std() / m), 4) if m > 1e-9 else None
        if gn is not None and gn.notna().sum() > 10:
            gv = gn.fillna(0).to_numpy()
            grad_max = round(float(gv.max()), 1)
            tail = gv[int(len(gv) * 0.75):]
            nz = tail[tail > 1e-6]
            grad_late_max = round(float(nz.max()), 2) if nz.size else 0.0
        if any(v is not None for v in (loss_cv, grad_late_max, grad_max)):
            health.append((r.name, loss_cv, grad_late_max, grad_max))

        def delta(col):
            s = h.get(col)
            if s is None:
                return None
            s = s.dropna()
            return round(float(s.iloc[-1] - s.iloc[0]), 4) if len(s) >= 2 else None
        d8, d16 = delta("passk_math_val/k8"), delta("passk_math_val/k16")
        dbe = delta("val/block_eff")
        if d8 is not None or d16 is not None:
            movement.append((r.name, d8, d16, dbe))
    return health, movement


def write_report(path, df, runs, args):
    lines = []
    W = lines.append
    W("# W&B run-mining report\n")
    W(f"Mined **{len(df)} runs** from `{args.entity}/{args.project}`.\n")
    W("> **Caveat:** OBSERVATIONAL runs, not a designed experiment — confounded "
      "hyperparameters can't be separated. This is *association* (where to look), "
      "not *causation*. Every table below is split by loss/objective/pair/dataset "
      "so nothing is crossed.\n")

    W("## 1. Loss-family comparison — did the rejected objectives have value?\n")
    lf = loss_family_table(df)
    W("| loss / objective / pair / dataset | n | best BE | median BE | collapse rate |")
    W("|---|---|---|---|---|")
    for _, r in lf.iterrows():
        W(f"| {r['group']} | {r['n']} | {r['best_BE']} | {r['median_BE']} | {r['collapse_rate']} |")
    W("")

    W("## 2. Hyperparameter importance (prefix_overlap/prob, 0.6B→8B, math_hard)\n")
    mask = ((df["loss"] == "prefix_overlap") & (df["prefix_objective"] == "prob") &
            (df["train_dataset"].astype(str).str.contains("math_hard", na=False)))
    imp, n, method = param_importance(df, mask)
    if imp is None:
        W(f"_{method}_\n")
    else:
        W(f"Fit on {n} runs via {method}, most→least important:\n")
        W("| feature | importance |\n|---|---|")
        for f, v in imp:
            W(f"| {f} | {round(v, 4)} |")
    W("")

    W("## 3. Marginal effect per hyperparam (value → mean best_BE, n)\n")
    for f, rows in marginal_view(df, mask).items():
        W(f"**{f}** — " + "  ".join(f"{k}:{v}(n={n})" for k, v, n in rows) + "\n")

    W("## 4. Noise floor (repeated configs)\n")
    dups = noise_floor(df)
    if not dups:
        W("_No repeated configs — every run is a single seed, so the run-to-run "
          "noise floor is UNKNOWN. Any 'win' smaller than a real seed-to-seed "
          "spread is unverified. Run 2–3 seeds on the current best config before "
          "trusting its margin._\n")
    else:
        W("| example run | n | BE range | BE std |\n|---|---|---|---|")
        for name, k, rng, std in dups:
            W(f"| {name} | {k} | {rng} | {std} |")
    W("")

    if args.history:
        health, movement = history_analysis(runs, df)
        W("## 5. Training health (loss smoothness + gradient sanity)\n")
        W("loss_cv = std/|mean| of loss over the last half (lower = smoother, "
          "Rahul wants low). grad_late_max = biggest NONZERO grad in the last "
          "quarter (~0 ⇒ dead-gradient collapse). grad_max = spike magnitude.\n")
        W("| run | loss_cv | grad_late_max | grad_max |\n|---|---|---|---|")
        for name, cv, glm, gm in sorted(health, key=lambda t: (t[1] is None, t[1] or 0)):
            W(f"| {name} | {cv} | {glm} | {gm} |")
        W("")
        W("## 6. Pass@k movement — which run moved k8/k16 most (recent runs only)\n")
        W("first→last Δ of passk_math_val within each run. **Watch the rows where "
          "Δk8/Δk16 > 0 but Δblock_eff < 0** — pass@k improving while BE regresses "
          "means the run may be learning something BE alone misses.\n")
        W("| run | Δpassk_k8 | Δpassk_k16 | Δblock_eff | passk↑ & BE↓? |\n|---|---|---|---|---|")
        for name, d8, d16, dbe in sorted(movement, key=lambda t: -(t[2] or t[1] or -9)):
            flag = "YES" if ((d16 or d8 or 0) > 0 and (dbe or 0) < 0) else ""
            W(f"| {name} | {d8} | {d16} | {dbe} | {flag} |")
        W("")

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[mine] wrote {path} ({len(df)} runs)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", default="rmukund16-indian-institute-of-technology-hyderabad")
    ap.add_argument("--project", default="distillspec-pipeline")
    ap.add_argument("--out", default="Results/wandb_mining_report.md")
    ap.add_argument("--history", action="store_true",
                    help="pull per-step history for loss/grad/pass@k trends (slower)")
    args = ap.parse_args()

    try:
        import wandb  # noqa
        import pandas  # noqa
    except ImportError as e:
        sys.exit(f"needs wandb + pandas: {e}")
    import wandb
    runs = list(wandb.Api().runs(f"{args.entity}/{args.project}"))
    if not runs:
        sys.exit("no runs found — check --entity/--project")
    df = flatten_runs(runs)
    write_report(args.out, df, runs, args)


if __name__ == "__main__":
    main()
