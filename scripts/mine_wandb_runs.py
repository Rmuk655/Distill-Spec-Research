#!/usr/bin/env python
"""
mine_wandb_runs.py — a run-MINING SYSTEM over every logged W&B run, not a
feature-importance notebook. One export step, one dataframe, four output layers:

  1. WHAT MATTERS GLOBALLY   — importance three ways (RF impurity + permutation +
     standardized-linear; SHAP if installed), because RF impurity is distorted
     when sweep features move together (lr/warmup/wd/clip correlate).
  2. WHAT MATTERS CONDITIONALLY — the hidden-gem layer: pairwise interactions,
     subgroup/rule slices, and a Pareto front over (BE, pass@k, stability).
  3. WHAT PREDICTS COLLAPSE EARLY — an operational pruning rule from first-window
     signals (grad slope, grad flatline, val trajectory, loss instability).
  4. WHAT IS SIGNAL VS NOISE — repeated-config variance = the belief threshold.

Plus trajectory clustering (group runs by their curves — fast starters / late
bloomers / quiet collapsers — not just final numbers).

OBSERVATIONAL, NOT EXPERIMENTAL: output is association / ranking / candidate
hypotheses, not causal proof. Workflow: mine associations -> form hypotheses ->
validate with a small controlled sweep. Every table is split by
(loss, objective, draft->teacher, train_dataset) so nothing gets crossed.

USAGE (on the cluster, W&B auth present):
    python scripts/mine_wandb_runs.py --history         # full system (slower)
    python scripts/mine_wandb_runs.py                   # fast: config+summary layers only

Needs: wandb, pandas, numpy. Optional: scikit-learn (importance/collapse/cluster),
shap (extra attribution). Missing optionals degrade gracefully.
"""
import argparse
import math
import os
import sys

NUMERIC_HPARAMS = ["lr", "warmup_steps", "lr_min_ratio", "weight_decay",
                   "prefix_teacher_topk", "prefix_M", "prefix_aux_weight",
                   "prefix_anneal_steps", "prefix_root_spacing", "K", "L", "steps"]
CATEGORICAL_HPARAMS = ["loss", "prefix_objective", "prefix_aux", "draft",
                       "teacher", "train_dataset", "val_dataset"]
BEST_BE_KEYS = ["val/best_block_eff", "best_val_block_eff", "val/block_eff", "best_block_eff"]
PASSK_KS = [1, 2, 4, 8, 16, 32, 64]
# grad_norm nonzero only on optimizer steps (grad_accum=8, logged/10) => ~3/4 of
# logged values are EXPECTED zeros. "Dead gradient" = no nonzero grad in a window.
EARLY_CUTOFF_FRAC = 0.35   # "early window" = first 35% of training for collapse prediction


def get_best_be(s):
    for k in BEST_BE_KEYS:
        v = s.get(k)
        if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
            return float(v)
    return None


def _short(x):
    x = str(x)
    return x.split("/")[-1] if x and x != "None" else "?"


# ----------------------------------------------------------------------------- ingest
def flatten_runs(runs):
    import pandas as pd
    rows = []
    for r in runs:
        cfg = {k: r.config.get(k) for k in NUMERIC_HPARAMS + CATEGORICAL_HPARAMS}
        s = dict(r.summary)
        best_be, final_be = get_best_be(s), s.get("val/block_eff")
        row = {"name": r.name, "id": r.id, "state": r.state, "best_be": best_be,
               "final_be": float(final_be) if isinstance(final_be, (int, float)) else None, **cfg}
        for k in PASSK_KS:
            for ds in ("math_val", "math_eval"):
                v = s.get(f"passk_{ds}/k{k}")
                if isinstance(v, (int, float)):
                    row[f"passk_{ds}_k{k}"] = float(v)
        rows.append(row)
    df = pd.DataFrame(rows)
    if "lr" in df:
        df["log10_lr"] = df["lr"].apply(lambda x: math.log10(x) if isinstance(x, (int, float)) and x and x > 0 else None)
    df["collapse_flag"] = ((df["best_be"] < 2.0) |
                           ((df["final_be"].notna()) & (df["final_be"] < 0.7 * df["best_be"]))).astype("Int64")
    return df


def group_key(df):
    return (df["loss"].astype(str) + " / " + df["prefix_objective"].astype(str).replace("None", "-") +
            " / " + df["draft"].map(_short) + "→" + df["teacher"].map(_short) + " / " + df["train_dataset"].astype(str))


def prob_mask(df):
    return ((df["loss"] == "prefix_overlap") & (df["prefix_objective"] == "prob") &
            (df["train_dataset"].astype(str).str.contains("math_hard", na=False)))


# ----------------------------------------------------------------------------- history
def pull_history(runs, df):
    """One history call per run -> per-run derived features + raw val curve.
    Fills training-health, pass@k movement, early-window collapse signals,
    time_to_collapse, best_step, and the resampled val trajectory for clustering."""
    import numpy as np
    keys = (["train/loss", "train/grad_norm", "val/block_eff", "train/step"] +
            [f"passk_math_val/k{k}" for k in PASSK_KS])
    per = {}
    curves = {}
    for r in runs:
        try:
            h = r.history(keys=keys, samples=6000)
        except Exception:
            continue
        if h is None or h.empty:
            continue
        step = h.get("train/step")
        step = step.to_numpy() if step is not None else np.arange(len(h))
        rec = {}
        # --- training health ---
        loss = h.get("train/loss")
        if loss is not None and loss.notna().sum() > 10:
            lv = loss.dropna().to_numpy(); tail = lv[len(lv)//2:]
            m = abs(tail.mean()); rec["loss_cv"] = round(float(tail.std()/m), 4) if m > 1e-9 else None
        gn = h.get("train/grad_norm")
        if gn is not None and gn.notna().sum() > 10:
            gv = gn.fillna(0).to_numpy()
            rec["grad_max"] = round(float(gv.max()), 1)
            late = gv[int(len(gv)*0.75):]; nz = late[late > 1e-6]
            rec["grad_late_max"] = round(float(nz.max()), 2) if nz.size else 0.0
            # early-window signals for the collapse predictor
            n_early = max(5, int(len(gv) * EARLY_CUTOFF_FRAC))
            eg = gv[:n_early]; enz = eg[eg > 1e-6]
            rec["grad_early_flatfrac"] = round(float((eg <= 1e-6).mean()), 3)
            if enz.size >= 3:  # slope of nonzero early grads (rising vs dying)
                x = np.arange(enz.size); rec["grad_early_slope"] = round(float(np.polyfit(x, enz, 1)[0]), 4)
        # --- val trajectory: time_to_collapse, best_step, early slope, curve ---
        vb = h.get("val/block_eff")
        if vb is not None and vb.notna().sum() >= 3:
            v = vb.dropna(); vs = step[vb.notna().to_numpy()] if len(step) == len(vb) else np.arange(len(v))
            va = v.to_numpy()
            rec["best_step"] = int(vs[int(va.argmax())])
            run_max = np.maximum.accumulate(va)
            crashed = np.where(va < 0.7 * run_max)[0]
            rec["time_to_collapse"] = int(vs[crashed[0]]) if crashed.size else None
            ne = max(2, int(len(va) * EARLY_CUTOFF_FRAC))
            if ne >= 2:
                rec["val_early_slope"] = round(float(np.polyfit(np.arange(ne), va[:ne], 1)[0]), 5)
            # resample to 20 pts over [0,1] for clustering
            xp = (vs - vs.min()) / (vs.max() - vs.min() + 1e-9)
            curves[r.name] = np.interp(np.linspace(0, 1, 20), xp, va)
        # --- pass@k movement ---
        for k in (8, 16):
            s = h.get(f"passk_math_val/k{k}")
            if s is not None and s.dropna().shape[0] >= 2:
                sd = s.dropna(); rec[f"dpassk_k{k}"] = round(float(sd.iloc[-1] - sd.iloc[0]), 4)
        if vb is not None and vb.dropna().shape[0] >= 2:
            vd = vb.dropna(); rec["dblock_eff"] = round(float(vd.iloc[-1] - vd.iloc[0]), 4)
        per[r.name] = rec
    return per, curves


# ----------------------------------------------------------------------------- LAYER 1
def importance_three_ways(df, mask):
    import numpy as np, pandas as pd
    sub = df[mask].dropna(subset=["best_be"]).copy()
    feats = [c for c in ["log10_lr", "warmup_steps", "lr_min_ratio", "weight_decay",
                         "prefix_teacher_topk", "prefix_M", "prefix_aux_weight", "prefix_anneal_steps"]
             if c in sub.columns]
    if len(sub) < 8:
        return None, f"too few runs ({len(sub)}<8)"
    X = sub[feats].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    varying = [f for f in feats if X[f].nunique() > 1]
    X, y = X[varying], sub["best_be"].to_numpy()
    cols = {"feature": varying}
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.inspection import permutation_importance
        from sklearn.linear_model import LinearRegression
        rf = RandomForestRegressor(n_estimators=400, random_state=0).fit(X, y)
        cols["rf_impurity"] = [round(v, 4) for v in rf.feature_importances_]
        pi = permutation_importance(rf, X, y, n_repeats=30, random_state=0)
        cols["permutation"] = [round(v, 4) for v in pi.importances_mean]
        Xs = (X - X.mean()) / (X.std().replace(0, 1))
        lin = LinearRegression().fit(Xs, y)
        cols["std_linear_coef"] = [round(v, 4) for v in lin.coef_]
        try:
            import shap
            expl = shap.TreeExplainer(rf); sv = np.abs(expl.shap_values(X)).mean(0)
            cols["shap_mean_abs"] = [round(float(v), 4) for v in sv]
        except Exception:
            pass
        return pd.DataFrame(cols).sort_values("permutation", ascending=False), f"n={len(sub)}"
    except ImportError:
        corr = [round(abs(np.corrcoef(X[f], y)[0, 1]), 4) for f in varying]
        cols["abs_corr"] = corr
        return pd.DataFrame(cols).sort_values("abs_corr", ascending=False), f"n={len(sub)} (no sklearn, corr only)"


# ----------------------------------------------------------------------------- LAYER 2
def subgroup_rules(df, mask, min_support=4, target="best_be"):
    """1- and 2-condition slices with the biggest lift over the subgroup mean."""
    import pandas as pd
    sub = df[mask].dropna(subset=[target]).copy()
    if len(sub) < min_support * 2:
        return None
    base = sub[target].mean()
    feats = [c for c in ["log10_lr", "warmup_steps", "lr_min_ratio", "weight_decay",
                         "prefix_teacher_topk", "prefix_M", "prefix_aux_weight", "prefix_anneal_steps"]
             if c in sub.columns and pd.to_numeric(sub[c], errors="coerce").nunique() > 1]
    conds = []  # (label, boolean mask)
    for f in feats:
        col = pd.to_numeric(sub[f], errors="coerce")
        for q in (col.quantile(0.33), col.quantile(0.66)):
            conds.append((f"{f}<={round(q,4)}", col <= q))
            conds.append((f"{f}>{round(q,4)}", col > q))
    rules = []
    for i, (la, ma) in enumerate(conds):
        for lb, mb in ([(None, None)] + conds[i+1:]):
            m = ma if mb is None else (ma & mb)
            if m.sum() < min_support:
                continue
            lift = sub.loc[m, target].mean() - base
            label = la if lb is None else f"{la} AND {lb}"
            rules.append((label, int(m.sum()), round(sub.loc[m, target].mean(), 3), round(lift, 3)))
    rules.sort(key=lambda t: -t[3])
    top = rules[:8] + [("— worst —", 0, 0, 0)] + sorted(rules, key=lambda t: t[3])[:4]
    return base, top


def pareto_front(df, per):
    """Non-dominated over (best_be↑, final pass@k_k16↑, stability=-loss_cv↑)."""
    import numpy as np
    rows = []
    for _, r in df.iterrows():
        h = per.get(r["name"], {})
        pk = r.get("passk_math_val_k16")
        cv = h.get("loss_cv")
        if r["best_be"] is None or pk is None or cv is None:
            continue
        rows.append((r["name"], float(r["best_be"]), float(pk), -float(cv)))
    front = []
    for a in rows:
        if not any(b[1] >= a[1] and b[2] >= a[2] and b[3] >= a[3] and b[1:] != a[1:] for b in rows):
            front.append(a)
    return sorted(front, key=lambda t: -t[1])


# ----------------------------------------------------------------------------- LAYER 3
def collapse_predictor(df, per):
    """Predict collapse_flag from EARLY-window history signals -> a pruning rule."""
    import numpy as np, pandas as pd
    feats = ["grad_early_flatfrac", "grad_early_slope", "val_early_slope"]
    rows = []
    for _, r in df.iterrows():
        h = per.get(r["name"], {})
        if pd.isna(r["collapse_flag"]) or any(f not in h for f in feats):
            continue
        rows.append([h[f] for f in feats] + [int(r["collapse_flag"])])
    if len(rows) < 10:
        return None, f"only {len(rows)} runs with early signals + label"
    arr = np.array(rows, float); X, y = arr[:, :-1], arr[:, -1]
    out = {"n": len(rows), "n_collapsed": int(y.sum()), "feats": feats}
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        if len(set(y)) == 2:
            clf = LogisticRegression(max_iter=1000).fit((X - X.mean(0)) / (X.std(0) + 1e-9), y)
            out["coef"] = [round(c, 3) for c in clf.coef_[0]]
            out["auc_insample"] = round(roc_auc_score(y, clf.predict_proba((X - X.mean(0)) / (X.std(0) + 1e-9))[:, 1]), 3)
    except ImportError:
        pass
    # single best-threshold rule on grad_early_flatfrac (most interpretable pruning rule)
    ff = X[:, 0]
    best = None
    for thr in np.unique(ff):
        pred = (ff >= thr).astype(int)
        acc = (pred == y).mean()
        if best is None or acc > best[1]:
            best = (round(float(thr), 3), round(float(acc), 3))
    out["rule"] = f"collapse if grad_early_flatfrac >= {best[0]}  (in-sample acc {best[1]})"
    return out, "ok"


# ----------------------------------------------------------------------------- trajectory clustering
def cluster_trajectories(curves):
    import numpy as np
    if len(curves) < 8:
        return None
    names = list(curves); M = np.vstack([curves[n] for n in names])
    Mz = (M - M.mean(1, keepdims=True)) / (M.std(1, keepdims=True) + 1e-9)
    try:
        from sklearn.cluster import KMeans
        k = min(4, len(names) // 3)
        lab = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(Mz)
    except ImportError:
        return None
    out = {}
    for c in range(lab.max() + 1):
        members = [names[i] for i in range(len(names)) if lab[i] == c]
        mean_curve = M[lab == c].mean(0)
        # crude shape label
        slope = mean_curve[-1] - mean_curve[0]
        peak = mean_curve.argmax() / len(mean_curve)
        shape = ("collapser" if mean_curve[-1] < 0.7 * mean_curve.max()
                 else "riser" if slope > 0.1 else "early-peak/decline" if peak < 0.4 else "flat/stable")
        out[c] = (shape, len(members), members[:4])
    return out


# ----------------------------------------------------------------------------- noise floor
def noise_floor(df):
    key_cols = [c for c in ["loss", "prefix_objective", "log10_lr", "warmup_steps", "lr_min_ratio",
                            "weight_decay", "prefix_teacher_topk", "prefix_M", "prefix_aux_weight",
                            "prefix_anneal_steps", "draft", "teacher", "train_dataset"] if c in df.columns]
    groups = []
    for _, sub in df.dropna(subset=["best_be"]).groupby([df[c].astype(str) for c in key_cols]):
        if len(sub) > 1:
            be = sub["best_be"]
            groups.append((sub["name"].iloc[0], len(sub), round(be.max() - be.min(), 3), round(be.std(), 4)))
    return groups


# ----------------------------------------------------------------------------- report
def write_report(path, df, runs, args):
    import pandas as pd
    L = []; W = L.append
    W("# W&B run-mining report\n")
    W(f"Mined **{len(df)} runs** from `{args.entity}/{args.project}`.\n")
    W("> OBSERVATIONAL, not experimental — association / ranking / hypotheses, "
      "not causal proof. Mine → hypothesize → validate with a controlled sweep. "
      "Tables split by loss/objective/pair/dataset so nothing is crossed.\n")

    W("## 1. Loss-family comparison (rejected objectives — any value?)\n")
    d = df.copy(); d["group"] = group_key(d); rows = []
    for g, sub in d.groupby("group"):
        be = sub["best_be"].dropna()
        if be.empty:
            continue
        rows.append((g, len(sub), round(be.max(), 3), round(be.median(), 3),
                     f"{int(sub['collapse_flag'].fillna(0).sum())}/{len(sub)}"))
    W("| loss / objective / pair / dataset | n | best BE | median BE | collapse |\n|---|---|---|---|---|")
    for r in sorted(rows, key=lambda t: -t[2]):
        W(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} |")
    W("")

    mask = prob_mask(df)
    W("## 2. Global importance — three ways (prob 0.6B→8B math_hard)\n")
    W("_RF impurity is distorted by correlated sweep features; trust permutation "
      "+ standardized-linear agreement more._\n")
    imp, note = importance_three_ways(df, mask)
    if imp is None:
        W(f"_{note}_\n")
    else:
        W(f"_{note}_\n")
        W("| " + " | ".join(imp.columns) + " |")
        W("|" + "---|" * len(imp.columns))
        for _, r in imp.iterrows():
            W("| " + " | ".join(str(r[c]) for c in imp.columns) + " |")
    W("")

    W("## 3. Conditional layer — subgroup rules (lift over subgroup mean)\n")
    sg = subgroup_rules(df, mask)
    if sg is None:
        W("_too few runs for rule mining_\n")
    else:
        base, top = sg
        W(f"subgroup mean best_BE = {round(base,3)}. Slices by lift:\n")
        W("| rule | n | mean best_BE | lift |\n|---|---|---|---|")
        for la, n, mv, lift in top:
            W(f"| {la} | {n} | {mv} | {lift} |")
    W("")

    if args.history:
        per, curves = pull_history(runs, df)

        W("## 4. Pareto front — best_BE vs pass@k(k16) vs stability(−loss_cv)\n")
        pf = pareto_front(df, per)
        if not pf:
            W("_no runs with all three of best_be + passk_k16 + loss_cv_\n")
        else:
            W("| run | best_BE | passk_k16 | −loss_cv |\n|---|---|---|---|")
            for n, be, pk, s in pf:
                W(f"| {n} | {round(be,3)} | {round(pk,3)} | {round(s,4)} |")
        W("")

        W("## 5. Training health (loss smoothness + gradient sanity)\n")
        W("loss_cv = std/|mean| last-half (low=smooth). grad_late_max = biggest "
          "nonzero grad in last quarter (~0 ⇒ dead-gradient collapse). grad_max = spike.\n")
        W("| run | loss_cv | grad_late_max | grad_max | best_step | time_to_collapse |")
        W("|---|---|---|---|---|---|")
        for name in sorted(per, key=lambda n: (per[n].get("loss_cv") is None, per[n].get("loss_cv") or 0)):
            h = per[name]
            W(f"| {name} | {h.get('loss_cv')} | {h.get('grad_late_max')} | {h.get('grad_max')} "
              f"| {h.get('best_step')} | {h.get('time_to_collapse')} |")
        W("")

        W("## 6. Pass@k movement — k8/k16 vs block_eff (flag: pass@k↑ while BE↓)\n")
        mv = [(n, per[n].get("dpassk_k8"), per[n].get("dpassk_k16"), per[n].get("dblock_eff"))
              for n in per if "dpassk_k8" in per[n] or "dpassk_k16" in per[n]]
        W("| run | Δk8 | Δk16 | Δblock_eff | passk↑&BE↓ |\n|---|---|---|---|---|")
        for n, d8, d16, dbe in sorted(mv, key=lambda t: -(t[2] or t[1] or -9)):
            flag = "YES" if ((d16 or d8 or 0) > 0 and (dbe or 0) < 0) else ""
            W(f"| {n} | {d8} | {d16} | {dbe} | {flag} |")
        W("")

        W("## 7. Early-signal collapse predictor (pruning rule)\n")
        cp, cnote = collapse_predictor(df, per)
        if cp is None:
            W(f"_{cnote}_\n")
        else:
            W(f"n={cp['n']}, collapsed={cp['n_collapsed']}. Features: {cp['feats']}\n")
            if "coef" in cp:
                W(f"logistic std-coef = {cp['coef']}, in-sample AUC = {cp.get('auc_insample')}\n")
            W(f"**Pruning rule:** {cp['rule']}  _(in-sample, small N — descriptive)_\n")
        W("")

        W("## 8. Trajectory clusters (curve shape, not final value)\n")
        cl = cluster_trajectories(curves)
        if cl is None:
            W("_needs sklearn + ≥8 runs with val curves_\n")
        else:
            W("| cluster | shape | n | examples |\n|---|---|---|---|")
            for c, (shape, n, ex) in cl.items():
                W(f"| {c} | {shape} | {n} | {', '.join(ex)} |")
        W("")

    W("## Noise floor (repeated configs = your belief threshold)\n")
    dups = noise_floor(df)
    if not dups:
        W("_No repeated configs — single seed everywhere, noise floor UNKNOWN. "
          "Any 'win' below a real seed-to-seed spread is unverified. Run 2–3 "
          "seeds on the current best config before trusting its margin._\n")
    else:
        W("| example run | n | BE range | BE std |\n|---|---|---|---|")
        for name, k, rng, std in dups:
            W(f"| {name} | {k} | {rng} | {std} |")

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"[mine] wrote {path} ({len(df)} runs)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", default="rmukund16-indian-institute-of-technology-hyderabad")
    ap.add_argument("--project", default="distillspec-pipeline")
    ap.add_argument("--out", default="results/wandb_mining_report.md")
    ap.add_argument("--history", action="store_true", help="pull per-step history (layers 4-8; slower)")
    args = ap.parse_args()
    try:
        import wandb, pandas, numpy  # noqa
    except ImportError as e:
        sys.exit(f"needs wandb + pandas + numpy: {e}")
    import wandb
    runs = list(wandb.Api().runs(f"{args.entity}/{args.project}"))
    if not runs:
        sys.exit("no runs found — check --entity/--project")
    write_report(args.out, flatten_runs(runs), runs, args)


if __name__ == "__main__":
    main()
