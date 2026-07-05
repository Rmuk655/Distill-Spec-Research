"""
analyze_evals.py — cross-checkpoint comparison of eval CSVs.

Each checkpoint's eval writes one CSV (append-mode) with a fixed schema whose key
columns are (checkpoint, dataset, mode, K, L, block_eff, ...).  This script
consolidates ALL of them into one tidy long table, computes Δ block_eff vs a
JSD baseline, and emits the tables + heatmaps needed to answer:

  - does a loss beat JSD?                → sign of delta_vs_base
  - across verifiers, or one mode?       → delta heatmap row (checkpoint × mode×K)
  - specific to a K×verifier combo?      → isolated positive cell
  - best loss×verifier combo?            → best_combos.csv (argmax block_eff)
  - do losses suit specific verifiers?   → loss×verifier heatmap (Δ mean over K)
  - trends across K?                     → k_trends_*.png line facets

Design choices baked in:
  * CSVs are append-mode → we DEDUPE to the latest timestamp per
    (checkpoint, dataset, mode, K, L).
  * The JSD baseline is AVERAGED over every checkpoint whose name matches
    --baseline_regex (so s123/s456 seeds average out), per (pair, dataset, mode, K).
  * Δ smaller than --noise (default 0.15, the n=100 SE floor) is treated as "no
    signal": grayed out in heatmaps and excluded from the beats-JSD list.

Usage (run on the box where the CSVs live):
    python analyze_evals.py \
        --glob "/sensei-fs-3/users/rkrishna/**/logs/*.csv" \
        --out  /sensei-fs-3/users/rkrishna/analysis \
        --metric block_eff

    # only the 1.7B/32B pair, tighter noise floor:
    python analyze_evals.py --glob "...**/logs/*.csv" --pair "1.7B/32B" --noise 0.10

Outputs (under --out):
    master_long.csv                      deduped tidy table + derived cols + delta
    pivot_<pair>_<dataset>.csv           block_eff, rows=checkpoint, cols=mode_K
    delta_<pair>_<dataset>.csv           same but Δ vs JSD baseline
    best_combos.csv                      ranked (checkpoint,mode,K) per pair/dataset
    beats_jsd.csv                        only rows with Δ > noise (with which modes/Ks)
    delta_heat_<pair>_<dataset>.png      checkpoint × mode×K Δ heatmap
    lossverifier_<pair>_<dataset>.png    checkpoint × mode Δ heatmap (mean over K)
    k_trends_<pair>_<dataset>.png        block_eff vs K, one panel per verifier

Requires: pandas, matplotlib (both already in the training venv).
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

# Force UTF-8 stdout so status prints never crash on a cp1252 (Windows) console.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass

import numpy as np
import pandas as pd

# Plotting is OPTIONAL — the CSV/table outputs work without it. If matplotlib is
# missing (e.g. a bare analysis env), we still emit every table and just skip PNGs.
try:
    import matplotlib
    matplotlib.use("Agg")                   # headless — write PNGs, never open a window
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    _HAS_MPL = True
except Exception:                           # noqa: BLE001 — any import failure → tables-only
    _HAS_MPL = False


# --------------------------------------------------------------------------- #
# Load + normalize
# --------------------------------------------------------------------------- #

def _checkpoint_name(path: str) -> str:
    """/.../checkpoints/<name>/ckpt_best  →  <name>  (the run identity)."""
    p = str(path).replace("\\", "/").rstrip("/")
    parts = p.split("/")
    # name is the dir just above ckpt_best/ckpt_latest; fall back to basename.
    if parts and parts[-1].startswith("ckpt_"):
        return parts[-2] if len(parts) >= 2 else parts[-1]
    return parts[-1]


def _pair_from_teacher(teacher: str) -> str:
    """Map teacher model id → the draft/teacher pair label."""
    t = str(teacher)
    if "32B" in t:
        return "1.7B/32B"
    if "8B" in t:
        return "0.6B/8B"
    return "unknown"


_SEED_RE = re.compile(r"_s(\d+)")
_K_RE    = re.compile(r"_K\d+")


def _family(name: str) -> str:
    """Collapse a checkpoint name to its loss 'family' (strip seed + _K# tag)."""
    n = _SEED_RE.sub("", name)
    n = _K_RE.sub("", n)
    return n.rstrip("_")


def load_all(glob_pat: str) -> pd.DataFrame:
    """Read every matching CSV (per-file, so headers never leak into data), concat."""
    files = sorted(glob.glob(glob_pat, recursive=True))
    if not files:
        raise SystemExit(f"No CSVs matched: {glob_pat}")
    frames = []
    for f in files:
        try:
            # on_bad_lines="skip": recover the good rows from ragged old CSVs
            # instead of dropping the whole file (new files are well-formed).
            df = pd.read_csv(f, on_bad_lines="skip")
        except Exception as e:                              # skip empty/corrupt files
            print(f"[warn] skipping {f}: {e}")
            continue
        if "block_eff" not in df.columns:
            continue
        df["_src_file"] = os.path.basename(f)
        frames.append(df)
    if not frames:
        raise SystemExit("No usable CSVs (none had a block_eff column).")
    df = pd.concat(frames, ignore_index=True)
    print(f"[load] {len(files)} files -> {len(df)} raw rows")
    return df


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce types, derive identity columns, drop failed rows, dedupe to latest."""
    for c in ("block_eff", "throughput_tok_s", "K", "L", "n_prompts",
              "target_calls", "avg_tree_nodes", "L1"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop failed/empty eval cells (block_eff nan, no target calls).
    df = df[df["block_eff"].notna() & (df.get("target_calls", 1) > 0)].copy()

    df["checkpoint_name"] = df["checkpoint"].map(_checkpoint_name)
    df["family"]          = df["checkpoint_name"].map(_family)
    if "teacher_model" in df.columns:
        df["pair"] = df["teacher_model"].map(_pair_from_teacher)
    else:
        df["pair"] = "unknown"
    df["mode_K"] = df["mode"].astype(str) + "_K" + df["K"].astype("Int64").astype(str)

    # Append-mode CSVs → keep the LATEST row per identity tuple.
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    key = ["checkpoint_name", "pair", "dataset", "mode", "K", "L"]
    df = (df.sort_values("timestamp")
            .drop_duplicates(subset=key, keep="last")
            .reset_index(drop=True))
    print(f"[norm] {len(df)} rows after dedupe  |  "
          f"{df['checkpoint_name'].nunique()} checkpoints  |  "
          f"pairs={sorted(df['pair'].unique())}  |  "
          f"datasets={sorted(df['dataset'].unique())}")
    return df


# --------------------------------------------------------------------------- #
# Baseline + delta
# --------------------------------------------------------------------------- #

def add_delta(df: pd.DataFrame, baseline_regex: str, metric: str) -> pd.DataFrame:
    """Δ = metric − baseline, where baseline = mean over checkpoints matching
    baseline_regex, per (pair, dataset, mode, K).  Averaging over seeds gives a
    less-noisy bar."""
    brx = re.compile(baseline_regex)
    is_base = df["checkpoint_name"].map(lambda n: bool(brx.search(n)))
    base_rows = df[is_base]
    if base_rows.empty:
        print(f"[warn] baseline regex '{baseline_regex}' matched NO checkpoint — "
              f"delta columns will be NaN. Checkpoints present: "
              f"{sorted(df['checkpoint_name'].unique())[:8]}...")
    else:
        print(f"[base] baseline = mean of {sorted(base_rows['checkpoint_name'].unique())}")

    base = (base_rows.groupby(["pair", "dataset", "mode", "K"])[metric]
                     .mean().rename("baseline").reset_index())
    df = df.merge(base, on=["pair", "dataset", "mode", "K"], how="left")
    df["delta_vs_base"] = df[metric] - df["baseline"]
    df["is_baseline"] = is_base.values
    return df


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #

def _annotated_heatmap(mat: pd.DataFrame, title: str, path: str,
                       noise: float, vlim: float | None = None):
    """Diverging heatmap of Δ (rows × cols) with per-cell numbers; |Δ|<noise gray."""
    if not _HAS_MPL or mat.empty:
        return
    data = mat.values.astype(float)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return
    vlim = vlim or max(0.05, float(np.nanmax(np.abs(finite))))
    norm = TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim)

    fig_h = max(2.5, 0.42 * len(mat.index) + 1.5)
    fig_w = max(4.0, 0.60 * len(mat.columns) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(data, aspect="auto", cmap="RdBu_r", norm=norm)
    ax.set_xticks(range(len(mat.columns)))
    ax.set_xticklabels(mat.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels(mat.index, fontsize=7)
    ax.set_title(title, fontsize=9)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if not np.isfinite(v):
                continue
            sig = abs(v) >= noise
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                    fontsize=6,
                    color=("black" if sig else "0.6"),
                    fontweight=("bold" if sig else "normal"))
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label=f"Δ (|Δ|<{noise} = gray)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"[plot] {path}")


def k_trend_facets(sub: pd.DataFrame, metric: str, title: str, path: str):
    """block_eff vs K, one panel per verifier, one line per checkpoint."""
    modes = sorted(sub["mode"].unique())
    if not _HAS_MPL or not modes:
        return
    ncol = min(3, len(modes))
    nrow = int(np.ceil(len(modes) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.0 * nrow),
                             squeeze=False, sharex=True)
    ckpts = sorted(sub["checkpoint_name"].unique())
    cmap = plt.get_cmap("tab20")
    color = {c: cmap(i % 20) for i, c in enumerate(ckpts)}
    for idx, mode in enumerate(modes):
        ax = axes[idx // ncol][idx % ncol]
        m = sub[sub["mode"] == mode]
        for c in ckpts:
            cc = m[m["checkpoint_name"] == c].sort_values("K")
            if cc.empty:
                continue
            ax.plot(cc["K"], cc[metric], marker="o", ms=3, lw=1,
                    color=color[c], label=c)
        ax.set_title(mode, fontsize=8)
        ax.set_xlabel("K"); ax.set_ylabel(metric, fontsize=7)
        ax.grid(alpha=0.3)
    for j in range(len(modes), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)),
               fontsize=6, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=[0, 0.04, 1, 0.98])
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {path}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", str(s)).strip("-")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glob", required=True,
                    help="Glob for eval CSVs, e.g. '/sensei-fs-3/users/rkrishna/**/logs/*.csv' "
                         "(quote it; recursive ** supported).")
    ap.add_argument("--out", default="analysis_out", help="Output directory.")
    ap.add_argument("--metric", default="block_eff",
                    help="Metric to compare (block_eff | throughput_tok_s).")
    ap.add_argument("--baseline_regex", default=r"^jsd_math_?hard_s\d+$",
                    help="Regex over checkpoint_name selecting the JSD baseline(s) "
                         "to average. Default matches plain flat-JSD (jsd_mathhard_s123 / "
                         "jsd_math_hard_s456). Use e.g. '^jsd_flat_enrich_K3_' to compare "
                         "against enrich instead.")
    ap.add_argument("--noise", type=float, default=0.15,
                    help="Δ below this magnitude is treated as no-signal (n=100 SE floor).")
    ap.add_argument("--pair", default=None, help="Restrict to one pair, e.g. '1.7B/32B'.")
    ap.add_argument("--dataset", default=None, help="Restrict to one dataset, e.g. math_eval.")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = normalize(load_all(args.glob))
    if args.pair:
        df = df[df["pair"] == args.pair]
    if args.dataset:
        df = df[df["dataset"] == args.dataset]
    if df.empty:
        raise SystemExit("No rows after pair/dataset filter.")
    df = add_delta(df, args.baseline_regex, args.metric)

    df.to_csv(os.path.join(args.out, "master_long.csv"), index=False)
    print(f"[out ] master_long.csv ({len(df)} rows)")

    # beats-JSD list (signal only) — the headline "does it help" answer.
    beats = (df[df["delta_vs_base"] > args.noise]
             .sort_values("delta_vs_base", ascending=False)
             [["pair", "dataset", "checkpoint_name", "mode", "K",
               args.metric, "baseline", "delta_vs_base"]])
    beats.to_csv(os.path.join(args.out, "beats_jsd.csv"), index=False)
    print(f"[out ] beats_jsd.csv ({len(beats)} cells beat baseline by >{args.noise})")

    # best (checkpoint,mode,K) per pair/dataset, ranked by metric.
    best = (df.sort_values(args.metric, ascending=False)
              .groupby(["pair", "dataset"], group_keys=False)
              .head(15)
              [["pair", "dataset", "checkpoint_name", "mode", "K",
                args.metric, "delta_vs_base"]])
    best.to_csv(os.path.join(args.out, "best_combos.csv"), index=False)
    print(f"[out ] best_combos.csv")

    # Per (pair, dataset): pivots + heatmaps + K-trends.
    for (pair, ds), sub in df.groupby(["pair", "dataset"]):
        tag = f"{_slug(pair)}_{_slug(ds)}"

        piv = sub.pivot_table(index="checkpoint_name", columns="mode_K",
                              values=args.metric, aggfunc="mean")
        piv = piv.reindex(sorted(piv.columns, key=lambda c: (c.rsplit("_K", 1)[0],
                                  int(c.rsplit("_K", 1)[1]))), axis=1)
        piv.to_csv(os.path.join(args.out, f"pivot_{tag}.csv"))

        dpiv = sub.pivot_table(index="checkpoint_name", columns="mode_K",
                               values="delta_vs_base", aggfunc="mean")
        dpiv = dpiv.reindex(piv.columns, axis=1)
        dpiv.to_csv(os.path.join(args.out, f"delta_{tag}.csv"))

        _annotated_heatmap(dpiv, f"Δ{args.metric} vs JSD — {pair} / {ds}",
                           os.path.join(args.out, f"delta_heat_{tag}.png"), args.noise)

        # loss × verifier interaction: Δ averaged over K.
        lv = sub.pivot_table(index="checkpoint_name", columns="mode",
                             values="delta_vs_base", aggfunc="mean")
        _annotated_heatmap(lv, f"Δ{args.metric} vs JSD (mean over K) — {pair} / {ds}",
                           os.path.join(args.out, f"lossverifier_{tag}.png"), args.noise)

        k_trend_facets(sub, args.metric, f"{args.metric} vs K — {pair} / {ds}",
                       os.path.join(args.out, f"k_trends_{tag}.png"))

    print(f"\n[done] all outputs in {args.out}/")
    if not beats.empty:
        print("\nTop cells beating baseline:")
        print(beats.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
