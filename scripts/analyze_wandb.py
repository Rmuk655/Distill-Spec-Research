"""
analyze_wandb.py — post-hoc convergence/insight analysis of W&B TRAINING runs.

Companion to scripts/analyze_evals.py: that one reads the final eval CSVs; this one
reads the *training* runs (job_type=train) straight from the W&B Public API and turns
their metric histories into a ranked table + grouped insights + a self-contained HTML
report. Answers "which loss family / K / L / teacher_temp actually trains best, how
fast, how stably, and how seed-robustly" without clicking through dozens of runs.

Design:
  * Pulls runs across one or more projects, keeps job_type=train (override with --job_type).
  * CACHES everything to --cache_dir (parquet, CSV fallback) so re-runs are instant and
    offline. Re-pull from W&B live with --refresh.
  * Two layers: a per-run table (config + summary + derived convergence features) and the
    cached per-run metric history (for the sparklines / re-analysis).
  * Pure-pandas feature/grouping functions are isolated from the W&B calls so they can be
    unit-tested without network.

Auth: uses WANDB_API_KEY from the environment (same as the wandb CLI). No key in code.

Usage:
    export WANDB_API_KEY=...            # you already have this
    python scripts/analyze_wandb.py \
        --entity <your-wandb-entity> \
        --projects distillspec-pipeline \
        --out wandb_analysis --refresh          # first run: pull live + cache

    # subsequent runs read the cache (fast, offline); drop --refresh
    python scripts/analyze_wandb.py --entity <e> --projects distillspec-pipeline --out wandb_analysis

    # multiple projects, different primary metric, only finished runs
    python scripts/analyze_wandb.py --entity <e> \
        --projects distillspec-pipeline other-project \
        --primary_metric val/smoothed_block_eff --refresh

Outputs (under --out):
    runs_summary.csv        one row per run: config + summary + convergence features, ranked
    seed_groups.csv         config-minus-seed groups: mean/std/n of best metric (seed robustness)
    hyperparam_effects.csv  marginal mean of best metric per hyperparameter value
    flags.csv               overfit / stuck / diverged / crashed / single-seed runs
    report.html             self-contained sortable report (open in a browser, no server)

Requires: wandb, pandas. Optional: pyarrow (parquet cache; falls back to CSV),
matplotlib (sparklines in the HTML; skipped if absent).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:                       # noqa: BLE001
    _HAS_MPL = False


# Metrics pulled into each run's cached history (others are ignored to keep it light).
DEFAULT_METRICS = [
    "val/block_eff", "val/smoothed_block_eff", "val/forgetting",
    "val/n_easy", "val/n_medium", "val/n_hard",
    "train/loss", "train/lr", "train/grad_norm", "train/path_diversity",
]
# Config keys promoted to first-class columns (everything else is still kept, prefixed cfg.).
CANON_CONFIG = ["loss", "K", "L", "seed", "teacher_temp", "draft_temp", "draft", "teacher",
                "train_dataset", "val_dataset", "aux_loss", "aux_weight", "aux_mode",
                "prefix_objective", "prefix_M", "steps", "lr"]
# Hyperparameters whose marginal effect on the primary metric we tabulate.
DEFAULT_EFFECT_PARAMS = ["loss", "K", "L", "teacher_temp", "aux_loss", "aux_weight", "prefix_objective"]


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def _trapz(y, x):
    fn = getattr(np, "trapezoid", None) or np.trapz   # np>=2 renamed trapz→trapezoid
    return float(fn(y, x))


def _scalarize(d: dict) -> dict:
    """Keep only JSON-scalar values from a config/summary dict (drop media, nested, None)."""
    out = {}
    for k, v in (d or {}).items():
        if k.startswith("_"):
            continue
        if isinstance(v, bool) or isinstance(v, (int, float, str)):
            out[k] = v
        elif isinstance(v, (np.integer, np.floating)):
            out[k] = v.item()
    return out


def _seed_of(name: str, cfg: dict):
    if "seed" in cfg and cfg["seed"] not in (None, ""):
        return cfg["seed"]
    m = re.search(r"_s(\d+)", str(name))
    return int(m.group(1)) if m else None


def _group_label(name: str) -> str:
    """Run identity with the seed token stripped, for cross-seed grouping."""
    return re.sub(r"_s\d+", "", str(name)).rstrip("_")


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", str(s)).strip("-")


# --------------------------------------------------------------------------- #
# Convergence features (pure pandas — unit-testable without W&B)
# --------------------------------------------------------------------------- #

def convergence_features(hist: pd.DataFrame, primary: str,
                         last_window: int = 5, plateau_tol: float = 0.02) -> dict:
    """Derive per-run convergence/stability features from one run's metric history.

    hist must have an x column '_step' (or 'step') plus metric columns. Returns a flat
    dict of features; anything not computable is NaN. Feature glossary:
      <p>_best / _best_step   peak value and where it occurred
      <p>_final               last logged value
      <p>_drop_from_best      best - final  (>0 = declined after peak = overfit signature)
      <p>_steps_to_90pct_gain first step reaching init + 0.9*(best-init) (convergence speed)
      <p>_plateau_step        first step within plateau_tol of best (learning effectively done)
      <p>_auc_norm            mean height of the curve (fast+high rewarded)
      <p>_last_mean/_last_std stability over the final `last_window` points
      <p>_slope_early         linear slope over first third (early learning rate)
      <p>_monotonic_frac      fraction of step-to-step improvements
    Plus loss_final/min, grad_norm nonzero frac, forgetting trend, and loss↔metric coupling.
    """
    f: dict = {}
    if hist is None or hist.empty:
        return f
    xcol = "_step" if "_step" in hist.columns else ("step" if "step" in hist.columns else None)
    if xcol is None:
        hist = hist.reset_index().rename(columns={"index": "_step"})
        xcol = "_step"

    def _series(col):
        if col not in hist.columns:
            return None
        s = hist[[xcol, col]].dropna().sort_values(xcol)
        return s if len(s) else None

    # --- primary metric block -------------------------------------------------
    s = _series(primary)
    if s is not None:
        x = s[xcol].to_numpy(dtype=float)
        y = s[primary].to_numpy(dtype=float)
        best_i = int(np.argmax(y))
        init, best, final = float(y[0]), float(y[best_i]), float(y[-1])
        f[f"{primary}_best"] = best
        f[f"{primary}_best_step"] = float(x[best_i])
        f[f"{primary}_final"] = final
        f[f"{primary}_init"] = init
        f[f"{primary}_drop_from_best"] = best - final
        gain = best - init
        if gain > 1e-9:
            thr = init + 0.9 * gain
            reached = x[y >= thr]
            f[f"{primary}_steps_to_90pct_gain"] = float(reached[0]) if len(reached) else float("nan")
        else:
            f[f"{primary}_steps_to_90pct_gain"] = float(x[0])
        # plateau: first step within tol of the eventual best
        near = x[y >= best - plateau_tol]
        f[f"{primary}_plateau_step"] = float(near[0]) if len(near) else float("nan")
        if x[-1] > x[0]:
            f[f"{primary}_auc_norm"] = _trapz(y, x) / (x[-1] - x[0])
        tail = y[-last_window:]
        f[f"{primary}_last_mean"] = float(np.mean(tail))
        f[f"{primary}_last_std"] = float(np.std(tail))
        third = max(2, len(y) // 3)
        if third >= 2 and np.ptp(x[:third]) > 0:
            f[f"{primary}_slope_early"] = float(np.polyfit(x[:third], y[:third], 1)[0])
        diffs = np.diff(y)
        f[f"{primary}_monotonic_frac"] = float(np.mean(diffs > 0)) if len(diffs) else float("nan")
        f[f"{primary}_n_points"] = int(len(y))

    # --- loss -----------------------------------------------------------------
    sl = _series("train/loss")
    if sl is not None:
        y = sl["train/loss"].to_numpy(dtype=float)
        f["loss_final"] = float(y[-1])
        f["loss_min"] = float(np.min(y))
        f["loss_last_std"] = float(np.std(y[-max(10, last_window):]))

    # --- grad norm (the grad_accum-zero pattern) ------------------------------
    sg = _series("train/grad_norm")
    if sg is not None:
        g = sg["train/grad_norm"].to_numpy(dtype=float)
        f["grad_nonzero_frac"] = float(np.mean(g > 0)) if len(g) else float("nan")
        f["grad_max"] = float(np.max(g)) if len(g) else float("nan")

    # --- forgetting trend (rising = bad) --------------------------------------
    sf = _series("val/forgetting")
    if sf is not None and len(sf) >= 2 and np.ptp(sf[xcol]) > 0:
        f["forgetting_final"] = float(sf["val/forgetting"].to_numpy()[-1])
        f["forgetting_slope"] = float(np.polyfit(sf[xcol].to_numpy(float),
                                                 sf["val/forgetting"].to_numpy(float), 1)[0])

    # --- loss <-> primary coupling within the run (H0 diagnostic) -------------
    if s is not None and sl is not None:
        try:
            merged = pd.merge_asof(s.rename(columns={primary: "m"}),
                                   sl.rename(columns={"train/loss": "loss"}),
                                   on=xcol, direction="nearest")
            if merged["m"].notna().sum() >= 5:
                f["loss_metric_spearman"] = float(merged["m"].corr(merged["loss"], method="spearman"))
        except Exception:
            pass
    return f


def flag_run(row: pd.Series, primary: str, noise: float = 0.15) -> list[str]:
    """Return a list of quality flags for one run row (from runs_summary)."""
    flags = []
    drop = row.get(f"{primary}_drop_from_best")
    best_step = row.get(f"{primary}_best_step")
    # overfit: fell from peak by > noise AND peaked in the first 60% of training
    last_step = row.get(f"{primary}_n_points")
    if pd.notna(drop) and drop > noise:
        flags.append("declined_after_peak")
    # stuck: loss essentially flat AND primary barely moved from init
    gain = (row.get(f"{primary}_best") or np.nan) - (row.get(f"{primary}_init") or np.nan)
    if pd.notna(row.get("loss_last_std")) and row["loss_last_std"] < 1e-4:
        flags.append("loss_flatlined")
    if pd.notna(gain) and gain < noise:
        flags.append("no_metric_gain")
    # diverged: final loss well above its min
    if pd.notna(row.get("loss_final")) and pd.notna(row.get("loss_min")) and \
       row["loss_min"] > 0 and row["loss_final"] > 2.0 * row["loss_min"]:
        flags.append("loss_diverged")
    if str(row.get("state", "")).lower() not in ("finished", ""):
        flags.append(f"state_{row.get('state')}")
    return flags


# --------------------------------------------------------------------------- #
# Cross-run aggregations (pure pandas)
# --------------------------------------------------------------------------- #

def seed_groups(df: pd.DataFrame, primary: str) -> pd.DataFrame:
    """Group by run identity minus seed → mean/std/n of best primary (seed robustness)."""
    bestcol = f"{primary}_best"
    if bestcol not in df.columns:
        return pd.DataFrame()
    g = (df.groupby("group_label")[bestcol]
           .agg(n_seeds="count", mean_best="mean", std_best="std",
                min_best="min", max_best="max")
           .reset_index()
           .sort_values("mean_best", ascending=False))
    g["seed_spread"] = g["max_best"] - g["min_best"]
    return g


def hyperparam_effects(df: pd.DataFrame, primary: str, params: list[str]) -> pd.DataFrame:
    """Marginal mean of best primary per value of each hyperparameter (crude ablation)."""
    bestcol = f"{primary}_best"
    if bestcol not in df.columns:
        return pd.DataFrame()
    rows = []
    for p in params:
        if p not in df.columns:
            continue
        sub = df[df[p].notna()]
        for val, grp in sub.groupby(p):
            rows.append({"param": p, "value": val, "n_runs": len(grp),
                         "mean_best": grp[bestcol].mean(), "std_best": grp[bestcol].std(),
                         "max_best": grp[bestcol].max()})
    return pd.DataFrame(rows).sort_values(["param", "mean_best"], ascending=[True, False])


# --------------------------------------------------------------------------- #
# W&B pull + cache (the only network-touching part)
# --------------------------------------------------------------------------- #

def _cache_write(df: pd.DataFrame, path_noext: str):
    try:
        df.to_parquet(path_noext + ".parquet")
    except Exception:
        df.to_csv(path_noext + ".csv", index=False)


def _cache_read(path_noext: str):
    if os.path.exists(path_noext + ".parquet"):
        try:
            return pd.read_parquet(path_noext + ".parquet")
        except Exception:
            pass
    if os.path.exists(path_noext + ".csv"):
        return pd.read_csv(path_noext + ".csv")
    return None


def pull_runs(entity, projects, job_type, cache_dir, samples, refresh, extra_filter):
    """Return (runs_df, {run_key: history_df}). Uses cache unless --refresh."""
    os.makedirs(cache_dir, exist_ok=True)
    hist_dir = os.path.join(cache_dir, "history")
    os.makedirs(hist_dir, exist_ok=True)
    runs_cache = os.path.join(cache_dir, "runs_raw")

    cached = None if refresh else _cache_read(runs_cache)
    histories: dict[str, pd.DataFrame] = {}
    if cached is not None:
        print(f"[cache] loaded {len(cached)} runs from {runs_cache}.* (use --refresh to re-pull)")
        for key in cached["run_key"]:
            h = _cache_read(os.path.join(hist_dir, _slug(key)))
            if h is not None:
                histories[key] = h
        return cached, histories

    import wandb
    api = wandb.Api(timeout=60)
    filt = {"jobType": job_type} if job_type else {}
    if extra_filter:
        filt.update(json.loads(extra_filter))

    rows = []
    for proj in projects:
        path = proj if "/" in proj else (f"{entity}/{proj}" if entity else proj)
        print(f"[pull] {path}  filter={filt}")
        runs = api.runs(path, filters=filt)
        for run in runs:
            key = f"{run.project}/{run.id}"
            cfg = _scalarize(dict(run.config))
            summ = _scalarize(dict(run.summary))
            row = {"run_key": key, "run_id": run.id, "name": run.name,
                   "project": run.project, "job_type": run.job_type, "state": run.state,
                   "runtime_s": run.summary.get("_runtime"), "created_at": str(run.created_at),
                   "url": run.url}
            row["seed"] = _seed_of(run.name, cfg)
            row["group_label"] = _group_label(run.name)
            for k in CANON_CONFIG:
                if k in cfg:
                    row[k] = cfg[k]
            for k, v in cfg.items():
                row.setdefault(f"cfg.{k}", v)
            for k, v in summ.items():
                row[f"sum.{k}"] = v
            rows.append(row)

            try:
                h = run.history(samples=samples, pandas=True)
            except Exception as e:
                print(f"  [warn] history failed for {key}: {e}")
                h = pd.DataFrame()
            if h is not None and not h.empty:
                histories[key] = h
                _cache_write(h, os.path.join(hist_dir, _slug(key)))
        print(f"  → {len(rows)} runs so far")

    runs_df = pd.DataFrame(rows)
    _cache_write(runs_df, runs_cache)
    print(f"[cache] wrote {len(runs_df)} runs + {len(histories)} histories to {cache_dir}/")
    return runs_df, histories


# --------------------------------------------------------------------------- #
# HTML report
# --------------------------------------------------------------------------- #

def _sparkline_b64(hist: pd.DataFrame, primary: str) -> str:
    if not _HAS_MPL or hist is None or primary not in hist.columns:
        return ""
    s = hist[[c for c in ("_step", primary) if c in hist.columns]].dropna()
    if s.empty:
        return ""
    import io, base64
    fig, ax = plt.subplots(figsize=(1.6, 0.4))
    ax.plot(s.iloc[:, 0] if "_step" in s.columns else range(len(s)), s[primary], lw=0.8)
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=80, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _html_table(df: pd.DataFrame, table_id: str, float_fmt="{:.3f}") -> str:
    if df.empty:
        return "<p><em>(empty)</em></p>"
    def fmt(v):
        if isinstance(v, float):
            return "" if pd.isna(v) else float_fmt.format(v)
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return ""
        return str(v)
    head = "".join(f"<th onclick=\"sortTable('{table_id}',{i})\">{c}</th>"
                   for i, c in enumerate(df.columns))
    body = "".join("<tr>" + "".join(f"<td>{fmt(v)}</td>" for v in row) + "</tr>"
                   for row in df.itertuples(index=False))
    return f"<table id='{table_id}'><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


_HTML_JS = """
<style>
 body{font-family:system-ui,Arial,sans-serif;margin:24px;color:#1a1a1a}
 h1{font-size:20px} h2{font-size:15px;margin-top:28px;border-bottom:1px solid #ddd}
 table{border-collapse:collapse;font-size:11px;margin:8px 0;width:100%}
 th,td{border:1px solid #ddd;padding:3px 6px;text-align:right}
 th{background:#f4f4f4;cursor:pointer;position:sticky;top:0}
 td:first-child,th:first-child{text-align:left}
 tr:nth-child(even){background:#fafafa}
 .note{color:#666;font-size:12px}
</style>
<script>
function sortTable(id,col){
 var t=document.getElementById(id),rows=Array.from(t.tBodies[0].rows);
 var asc=t.getAttribute('data-sort')!=col+'-asc';
 rows.sort(function(a,b){
   var x=a.cells[col].innerText,y=b.cells[col].innerText;
   var nx=parseFloat(x),ny=parseFloat(y);
   if(!isNaN(nx)&&!isNaN(ny)){return asc?nx-ny:ny-nx;}
   return asc?x.localeCompare(y):y.localeCompare(x);});
 rows.forEach(function(r){t.tBodies[0].appendChild(r);});
 t.setAttribute('data-sort',col+(asc?'-asc':'-desc'));
}
</script>
"""


def write_report(out, primary, runs_df, groups_df, effects_df, flags_df, histories, top_n=25):
    bestcol = f"{primary}_best"
    ranked = runs_df.sort_values(bestcol, ascending=False) if bestcol in runs_df else runs_df
    show_cols = [c for c in ["name", "project", "seed", "loss", "K", "L", "teacher_temp",
                             bestcol, f"{primary}_final", f"{primary}_drop_from_best",
                             f"{primary}_plateau_step", f"{primary}_last_std",
                             "loss_final", "loss_metric_spearman", "runtime_s"]
                 if c in ranked.columns]
    top = ranked[show_cols].head(top_n).copy()

    spark_html = ""
    if _HAS_MPL and histories:
        cells = []
        for _, r in ranked.head(top_n).iterrows():
            b64 = _sparkline_b64(histories.get(r["run_key"]), primary)
            img = f"<img src='{b64}'>" if b64 else ""
            bv = r.get(bestcol)
            bstr = f"{bv:.3f}" if isinstance(bv, (int, float)) and pd.notna(bv) else ""
            cells.append(f"<tr><td>{r['name']}</td><td>{bstr}</td><td>{img}</td></tr>")
        spark_html = ("<h2>Primary-metric sparklines (top runs)</h2>"
                      "<table><thead><tr><th>run</th><th>best</th><th>curve</th></tr></thead>"
                      f"<tbody>{''.join(cells)}</tbody></table>")

    n = len(runs_df)
    n_flagged = len(flags_df)
    overview = (f"<p class='note'>{n} training runs across "
                f"{runs_df['project'].nunique()} project(s). Primary metric: <b>{primary}</b>. "
                f"{n_flagged} run(s) flagged. Click any column header to sort.</p>")

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>W&B training analysis</title>{_HTML_JS}</head><body>
<h1>W&B training-run analysis — {primary}</h1>
{overview}
<h2>Top runs by {primary}_best</h2>{_html_table(top,'top')}
<h2>Seed robustness (config groups, minus seed)</h2>
<p class='note'>High <code>seed_spread</code> = result depends on the seed; low spread with high
<code>mean_best</code> = robustly good. This is the variance-collapse lens.</p>
{_html_table(groups_df.head(60),'groups')}
<h2>Hyperparameter marginal effects</h2>
<p class='note'>Mean best-metric per value, marginalizing over everything else — a crude
from-logs ablation, confounded by co-varying settings; use as a hint, not proof.</p>
{_html_table(effects_df,'effects')}
<h2>Flagged runs</h2>{_html_table(flags_df,'flags')}
{spark_html}
</body></html>"""
    path = os.path.join(out, "report.html")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"[out ] {path}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def build_summary(runs_df, histories, primary, last_window):
    feats = []
    for _, r in runs_df.iterrows():
        h = histories.get(r["run_key"])
        feats.append(convergence_features(h, primary, last_window=last_window))
    fdf = pd.DataFrame(feats, index=runs_df.index)
    return pd.concat([runs_df, fdf], axis=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"),
                    help="W&B entity/team (default $WANDB_ENTITY). Or put entity/ in --projects.")
    ap.add_argument("--projects", nargs="+", required=True,
                    help="One or more W&B project names (or entity/project).")
    ap.add_argument("--job_type", default="train",
                    help="Keep only this job_type (default 'train'; '' = all).")
    ap.add_argument("--primary_metric", default="val/block_eff",
                    help="Metric convergence features + ranking are built on.")
    ap.add_argument("--effect_params", nargs="+", default=DEFAULT_EFFECT_PARAMS,
                    help="Hyperparameters to tabulate marginal effects for.")
    ap.add_argument("--out", default="wandb_analysis", help="Output directory.")
    ap.add_argument("--cache_dir", default=None,
                    help="Cache dir for pulled runs/histories (default <out>/cache).")
    ap.add_argument("--refresh", action="store_true",
                    help="Re-pull from W&B live (default: use cache if present).")
    ap.add_argument("--samples", type=int, default=2000,
                    help="History points per run to fetch (sampled). 2000 is plenty for features.")
    ap.add_argument("--last_window", type=int, default=5,
                    help="Val points at the end used for stability (last_mean/last_std).")
    ap.add_argument("--noise", type=float, default=0.15,
                    help="Effect-size floor for flags (n=100 SE ~0.1-0.15).")
    ap.add_argument("--filters", default=None,
                    help="Extra MongoDB-style run filter as JSON, merged with job_type.")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cache_dir = args.cache_dir or os.path.join(args.out, "cache")
    primary = args.primary_metric

    runs_df, histories = pull_runs(args.entity, args.projects, args.job_type,
                                   cache_dir, args.samples, args.refresh, args.filters)
    if runs_df.empty:
        raise SystemExit("No runs returned. Check --entity/--projects/--job_type and WANDB_API_KEY.")

    df = build_summary(runs_df, histories, primary, args.last_window)

    # Flags
    flag_lists = df.apply(lambda r: flag_run(r, primary, args.noise), axis=1)
    df["flags"] = flag_lists.map(lambda xs: ",".join(xs))
    # single-seed groups → reproducibility gap
    grp = seed_groups(df, primary)
    singletons = set(grp[grp["n_seeds"] == 1]["group_label"]) if not grp.empty else set()
    df.loc[df["group_label"].isin(singletons), "flags"] = (
        df.loc[df["group_label"].isin(singletons), "flags"]
          .map(lambda s: (s + ",single_seed").lstrip(",")))

    bestcol = f"{primary}_best"
    ranked = df.sort_values(bestcol, ascending=False) if bestcol in df else df
    ranked.to_csv(os.path.join(args.out, "runs_summary.csv"), index=False)
    print(f"[out ] runs_summary.csv ({len(ranked)} runs)")

    grp.to_csv(os.path.join(args.out, "seed_groups.csv"), index=False)
    print(f"[out ] seed_groups.csv ({len(grp)} config groups)")

    eff = hyperparam_effects(df, primary, args.effect_params)
    eff.to_csv(os.path.join(args.out, "hyperparam_effects.csv"), index=False)
    print(f"[out ] hyperparam_effects.csv")

    flags_df = (df[df["flags"] != ""]
                [[c for c in ["name", "project", "seed", "loss", bestcol,
                              f"{primary}_drop_from_best", "flags"] if c in df.columns]]
                .sort_values("flags"))
    flags_df.to_csv(os.path.join(args.out, "flags.csv"), index=False)
    print(f"[out ] flags.csv ({len(flags_df)} flagged)")

    write_report(args.out, primary, df, grp, eff, flags_df, histories)

    print(f"\n[done] {args.out}/  — open report.html in a browser.")
    if bestcol in ranked.columns and not ranked.empty:
        print(f"\nTop 5 by {bestcol}:")
        cols = [c for c in ["name", "seed", bestcol, f"{primary}_drop_from_best", "flags"] if c in ranked]
        print(ranked[cols].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
