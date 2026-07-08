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

Glob patterns are expanded with $VARS and ~ before matching, so paths stay
machine-agnostic — set your output root once (e.g. `export OUT=/your/output/root`)
and reference `${OUT}` in patterns / the --config file instead of hardcoding it.

Usage (run on the box where the CSVs live; assumes `export OUT=<your root>`):
    python scripts/analyze_evals.py \
        --glob "${OUT}/**/logs/*.csv" \
        --out  "${OUT}/analysis" \
        --metric block_eff

    # only the 1.7B/32B pair, tighter noise floor:
    python scripts/analyze_evals.py --glob "${OUT}/**/logs/*.csv" --pair "1.7B/32B" --noise 0.10

Outputs (under --out):
    master_long.csv                      deduped tidy table + derived cols + delta
    pivot_<pair>_<dataset>.csv           block_eff, rows=checkpoint, cols=mode_K
    delta_<pair>_<dataset>.csv           same but Δ vs baseline
    best_combos.csv                      ranked (checkpoint,mode,K) per pair/dataset
    beats_baseline.csv                   only rows with Δ > noise (with which modes/Ks)
    delta_heat_<pair>_<dataset>.png      checkpoint × mode×K Δ heatmap
    lossverifier_<pair>_<dataset>.png    checkpoint × mode Δ heatmap (mean over K)
    k_trends_<pair>_<dataset>.png        block_eff vs K, one panel per verifier

Requires: pandas, matplotlib (both already in the training venv).
"""
from __future__ import annotations

import argparse
import glob
import json
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


def load_all(glob_pats) -> pd.DataFrame:
    """Read every matching CSV (per-file, so headers never leak into data), concat.

    glob_pats: one pattern (str) or a list of patterns — pass a list to select a SET
    of checkpoints that no single glob can express, e.g. a plain-JSD baseline (no
    numeric suffix) alongside an enrich family at specific K values:
        --glob "logs/jsd_mathhard_s123.csv" "logs/jsd_flat_enrich_K[3-6]_*.csv"
    (character classes like [3-6] work in plain glob; brace lists like {3,4,5,6} do
    NOT — glob.glob has no OR operator, which is why multiple patterns are accepted
    here instead of trying to force one pattern to do both jobs.)
    """
    if isinstance(glob_pats, str):
        glob_pats = [glob_pats]
    # Expand $VARS / ~ so patterns can be written machine-agnostically (e.g.
    # "${OUT}/logs/*.csv") and resolved from the environment at run time.
    files = sorted({f for pat in glob_pats
                    for f in glob.glob(os.path.expanduser(os.path.expandvars(pat)),
                                       recursive=True)})
    if not files:
        # ValueError (not SystemExit) so a --config battery can skip an
        # empty/pending bucket and continue with the rest.
        raise ValueError(f"No CSVs matched: {glob_pats}")
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


def _prefers_attn(attn_backend, prefer_attn: str) -> int:
    """1 if this row's logged attn_backend matches the preferred DRAFT backend, else 0.

    attn_backend format (telemetry._model_attn_backend): distinct impls across
    target+draft joined with '+', e.g. 'sdpa' (both sdpa) or 'sdpa+flash_attention_2
    (flash_attn-x)' (target sdpa, draft FA2 — target is ALWAYS sdpa, its tree-attention
    mask is incompatible with FA2, see eval-attn-backend-cross-machine-divergence memory).
    So presence/absence of 'flash_attention_2' in the string tells you the draft's backend.
    """
    has_fa2 = "flash_attention_2" in str(attn_backend)
    return int(has_fa2 if prefer_attn == "flash_attention_2" else not has_fa2)


def normalize(df: pd.DataFrame, prefer_attn: str = "sdpa"):
    """Coerce types, derive identity columns, drop failed rows, dedupe to latest.

    Cross-machine eval sweeps can silently mix attn_backend for the same
    (checkpoint,dataset,mode,K,L) cell (Transformers auto-selects the draft's backend
    per-machine based on local flash-attn availability) — this changes block_eff by
    up to ~0.2 (bf16 rounding flips a sampled token / accept-reject decision in
    stochastic decoding), comparable to the n=100 seed-noise floor. Rather than
    silently keeping "whichever ran last" for a contaminated cell, dedup here breaks
    ties toward `prefer_attn` (default "sdpa", matching eval.py's default since
    2026-07-06) and flags every contaminated cell.

    Returns (deduped_df, conflicts_df). conflicts_df is None if attn_backend isn't in
    the data or no cell had more than one distinct backend; otherwise it holds the RAW
    (pre-dedup) rows for every contaminated cell, side by side, for audit — write it to
    a CSV rather than trusting the dedupe silently.
    """
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
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    key = ["checkpoint_name", "pair", "dataset", "mode", "K", "L"]

    conflicts = None
    if "attn_backend" in df.columns:
        n_backends = df.groupby(key)["attn_backend"].nunique(dropna=True)
        mixed_keys = n_backends[n_backends > 1].index
        if len(mixed_keys):
            conflicts = (df.set_index(key).loc[mixed_keys]
                          .reset_index().sort_values(key + ["timestamp"]))
            print(f"[warn] {len(mixed_keys)} (checkpoint,dataset,mode,K,L) cells have "
                  f"MIXED attn_backend across duplicate rows (cross-machine contamination) "
                  f"— keeping the '{prefer_attn}'-preferring row per cell (or latest if "
                  f"neither matches). Full conflicting rows written to "
                  f"attn_backend_conflicts.csv for audit — spot-check before trusting "
                  f"deltas on those specific cells.")
        df["_prefers_attn"] = df["attn_backend"].map(lambda s: _prefers_attn(s, prefer_attn))
    else:
        df["_prefers_attn"] = 0

    # Append-mode CSVs → keep the latest row per identity tuple, but break ties toward
    # the preferred backend first (stable sort: within each key's rows, non-preferred
    # sort before preferred, earlier timestamp before later — keep="last" then picks
    # the preferred backend's latest row whenever one exists for that key).
    df = (df.sort_values(["_prefers_attn", "timestamp"])
            .drop_duplicates(subset=key, keep="last")
            .drop(columns=["_prefers_attn"])
            .reset_index(drop=True))
    print(f"[norm] {len(df)} rows after dedupe  |  "
          f"{df['checkpoint_name'].nunique()} checkpoints  |  "
          f"pairs={sorted(df['pair'].unique())}  |  "
          f"datasets={sorted(df['dataset'].unique())}")
    return df, conflicts


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

def consistent_wins(df: pd.DataFrame, noise: float, min_win_frac: float = 0.75) -> pd.DataFrame:
    """Checkpoint x verifier combos where the win holds across MOST of that
    verifier's K values, not just one noisy cell.

    beats_baseline.csv filters individual (checkpoint,mode,K) cells with zero
    correction for the fact that one checkpoint x verifier pair spans up to 4
    such cells (K=1..4) — a single lucky cell out of dozens tested is expected
    by chance even under pure noise (see eval-attn-backend-cross-machine-
    divergence memory: single-cell noise alone spans up to ~0.22, bigger than
    the default 0.15 threshold). Requiring the sign to hold across most K
    values for the SAME (checkpoint, verifier) pair is a much stronger signal
    that the effect is real rather than a threshold-crossing fluke.
    """
    g = df.groupby(["pair", "dataset", "checkpoint_name", "mode"])
    agg = g.agg(n_k=("K", "nunique"),
                n_k_win=("delta_vs_base", lambda s: int((s > noise).sum())),
                mean_delta=("delta_vs_base", "mean"),
                min_delta=("delta_vs_base", "min")).reset_index()
    agg["win_frac"] = agg["n_k_win"] / agg["n_k"]
    out = agg[(agg["win_frac"] >= min_win_frac) & (agg["mean_delta"] > noise)]
    return out.sort_values("mean_delta", ascending=False)


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


def run_analysis(glob_pats, out, metric="block_eff",
                 baseline_regex=r"^jsd_math_?hard_s\d+$", noise=0.15,
                 pair=None, dataset=None, prefer_attn="sdpa",
                 checkpoint_regex=None, label=""):
    """One comparison: load a bucket of CSVs, compute Δ vs baseline, write tables+plots.

    Everything the single-run CLI does, factored out so --config can call it once
    per named bucket into its own subfolder. `label` is only for console prefixing.
    Returns the beats-baseline DataFrame (may be empty)."""
    tag_pfx = f"[{label}] " if label else ""
    os.makedirs(out, exist_ok=True)
    try:
        raw = load_all(glob_pats)
    except ValueError as e:
        print(f"{tag_pfx}[warn] {e} — skipping (files not present yet?).")
        return pd.DataFrame()
    df, attn_conflicts = normalize(raw, prefer_attn=prefer_attn)
    if attn_conflicts is not None:
        cpath = os.path.join(out, "attn_backend_conflicts.csv")
        attn_conflicts.to_csv(cpath, index=False)
        _n = attn_conflicts.drop_duplicates(
            ["checkpoint_name", "pair", "dataset", "mode", "K", "L"]).shape[0]
        print(f"{tag_pfx}[out ] {cpath} ({len(attn_conflicts)} raw rows across {_n} contaminated cells)")
    if checkpoint_regex:
        _crx = re.compile(checkpoint_regex)
        df = df[df["checkpoint_name"].map(lambda n: bool(_crx.search(n)))]
        print(f"{tag_pfx}[filter] --checkpoint_regex kept {df['checkpoint_name'].nunique()} "
              f"checkpoint(s): {sorted(df['checkpoint_name'].unique())}")
    if pair:
        df = df[df["pair"] == pair]
    if dataset:
        df = df[df["dataset"] == dataset]
    if df.empty:
        print(f"{tag_pfx}[warn] no rows after checkpoint_regex/pair/dataset filter — skipping bucket.")
        return pd.DataFrame()
    df = add_delta(df, baseline_regex, metric)

    df.to_csv(os.path.join(out, "master_long.csv"), index=False)
    print(f"{tag_pfx}[out ] master_long.csv ({len(df)} rows)")

    beats = (df[df["delta_vs_base"] > noise]
             .sort_values("delta_vs_base", ascending=False)
             [["pair", "dataset", "checkpoint_name", "mode", "K",
               metric, "baseline", "delta_vs_base"]])
    beats.to_csv(os.path.join(out, "beats_baseline.csv"), index=False)
    print(f"{tag_pfx}[out ] beats_baseline.csv ({len(beats)} cells beat baseline by >{noise})")

    wins = consistent_wins(df, noise)
    wins.to_csv(os.path.join(out, "consistent_wins.csv"), index=False)
    print(f"{tag_pfx}[out ] consistent_wins.csv ({len(wins)} checkpoint x verifier pairs "
          f"win on >=75% of their K values — the real 'does X do better under verifier Y' answer)")

    best = (df.sort_values(metric, ascending=False)
              .groupby(["pair", "dataset"], group_keys=False)
              .head(15)
              [["pair", "dataset", "checkpoint_name", "mode", "K", metric, "delta_vs_base"]])
    best.to_csv(os.path.join(out, "best_combos.csv"), index=False)

    for (pr, ds), sub in df.groupby(["pair", "dataset"]):
        tag = f"{_slug(pr)}_{_slug(ds)}"
        piv = sub.pivot_table(index="checkpoint_name", columns="mode_K",
                              values=metric, aggfunc="mean")
        piv = piv.reindex(sorted(piv.columns, key=lambda c: (c.rsplit("_K", 1)[0],
                                  int(c.rsplit("_K", 1)[1]))), axis=1)
        piv.to_csv(os.path.join(out, f"pivot_{tag}.csv"))

        dpiv = sub.pivot_table(index="checkpoint_name", columns="mode_K",
                               values="delta_vs_base", aggfunc="mean").reindex(piv.columns, axis=1)
        dpiv.to_csv(os.path.join(out, f"delta_{tag}.csv"))
        _annotated_heatmap(dpiv, f"Δ{metric} vs baseline — {pr} / {ds}",
                           os.path.join(out, f"delta_heat_{tag}.png"), noise)

        lv = sub.pivot_table(index="checkpoint_name", columns="mode",
                             values="delta_vs_base", aggfunc="mean")
        _annotated_heatmap(lv, f"Δ{metric} vs baseline (mean over K) — {pr} / {ds}",
                           os.path.join(out, f"lossverifier_{tag}.png"), noise)
        k_trend_facets(sub, metric, f"{metric} vs K — {pr} / {ds}",
                       os.path.join(out, f"k_trends_{tag}.png"))

    print(f"{tag_pfx}[done] {out}/")
    return beats


def _run_config(config_path: str, out_root: str):
    """Run every named bucket in a JSON/YAML config into out_root/<name>/.

    Config schema (JSON shown; .yaml/.yml also accepted if PyYAML is installed):
        {
          "defaults": { "metric": "block_eff", "noise": 0.15,
                        "baseline_regex": "^jsd_math_?hard_s\\\\d+$" },
          "buckets": [
            { "name": "enrich_vs_jsd",
              "glob": ["/…/logs/jsd_mathhard_s123.csv",
                       "/…/logs/jsd_flat_enrich_*.csv"] },
            { "name": "po_warm_which_objective",
              "glob": ["/…/logs/po_*warm*.csv"],
              "baseline_regex": "logprob.*warm" }
          ]
        }
    Per-bucket keys override defaults: glob (required), baseline_regex, metric,
    noise, pair, dataset, prefer_attn, checkpoint_regex. `name` → subfolder."""
    # utf-8-sig: transparently strips a UTF-8 BOM if the config was saved by an
    # editor/PowerShell that adds one; identical to utf-8 when no BOM is present.
    if config_path.endswith((".yaml", ".yml")):
        try:
            import yaml
            cfg = yaml.safe_load(open(config_path, encoding="utf-8-sig"))
        except ImportError:
            raise SystemExit("PyYAML not installed — convert the config to .json or `pip install pyyaml`.")
    else:
        cfg = json.load(open(config_path, encoding="utf-8-sig"))

    defaults = cfg.get("defaults", {})
    buckets = cfg.get("buckets", [])
    if not buckets:
        raise SystemExit(f"No 'buckets' in {config_path}.")

    summary = []
    for b in buckets:
        name = b.get("name") or _slug(str(b.get("glob")))
        params = {**defaults, **b}
        if "glob" not in params:
            print(f"[{name}] [warn] no 'glob' — skipping.")
            continue
        print(f"\n{'='*70}\n=== bucket: {name}\n{'='*70}")
        beats = run_analysis(
            glob_pats=params["glob"],
            out=os.path.join(out_root, name),
            metric=params.get("metric", "block_eff"),
            baseline_regex=params.get("baseline_regex", r"^jsd_math_?hard_s\d+$"),
            noise=params.get("noise", 0.15),
            pair=params.get("pair"),
            dataset=params.get("dataset"),
            prefer_attn=params.get("prefer_attn", "sdpa"),
            checkpoint_regex=params.get("checkpoint_regex"),
            label=name,
        )
        summary.append((name, len(beats)))

    print(f"\n{'='*70}\n[config] {len(summary)} buckets done → {out_root}/")
    for name, nbeats in summary:
        print(f"    {name:40s}  {nbeats} cell(s) beat baseline")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None,
                    help="JSON/YAML file defining named comparison buckets (each with its own "
                         "globs + baseline), run into --out/<name>/. Lets you define the whole "
                         "'enrich vs jsd / tree-log vs jsd / PO cold vs jsd / PO warm vs jsd / "
                         "which-PO-objective' battery once and re-run it as new checkpoints land. "
                         "See _run_config docstring for the schema. When set, --glob is ignored.")
    ap.add_argument("--glob", nargs="+",
                    help="One or more globs for eval CSVs (quote each; recursive ** supported). "
                         "Pass multiple patterns to select a specific SET of checkpoints that no "
                         "single glob can express — e.g. a plain-JSD baseline plus an enrich "
                         "family: --glob \"logs/jsd_mathhard_s123.csv\" \"logs/jsd_flat_enrich_*.csv\" "
                         "(character classes like [3-6] work; brace lists like {3,4,5,6} do NOT "
                         "— glob has no OR operator). Required unless --config is given.")
    ap.add_argument("--checkpoint_regex", default=None,
                    help="After loading, keep only rows whose checkpoint_name matches this "
                         "regex — an independent filter from --glob, for when CSVs aren't neatly "
                         "one-per-checkpoint. E.g. '^jsd_mathhard_s123$|^po_(nss|logprob)_.*'.")
    ap.add_argument("--out", default="analysis_out", help="Output directory (root for --config).")
    ap.add_argument("--metric", default="block_eff",
                    help="Metric to compare (block_eff | throughput_tok_s).")
    ap.add_argument("--baseline_regex", default=r"^jsd_math_?hard_s\d+$",
                    help="Regex over checkpoint_name selecting the baseline(s) to average. Default "
                         "matches plain flat-JSD (jsd_mathhard_s123 / jsd_math_hard_s456) on both "
                         "model-pair boxes. Use e.g. 'logprob.*warm' for a within-family baseline.")
    ap.add_argument("--noise", type=float, default=0.15,
                    help="Δ below this magnitude is treated as no-signal (n=100 SE floor).")
    ap.add_argument("--pair", default=None, help="Restrict to one pair, e.g. '1.7B/32B'.")
    ap.add_argument("--dataset", default=None, help="Restrict to one dataset, e.g. math_eval.")
    ap.add_argument("--prefer_attn", default="sdpa", choices=["sdpa", "flash_attention_2"],
                    help="Tie-break for cross-machine attn_backend contamination per cell. "
                         "Default 'sdpa' matches eval.py's default since 2026-07-06.")
    args = ap.parse_args()

    if args.config:
        _run_config(args.config, args.out)
    elif args.glob:
        run_analysis(glob_pats=args.glob, out=args.out, metric=args.metric,
                     baseline_regex=args.baseline_regex, noise=args.noise,
                     pair=args.pair, dataset=args.dataset, prefer_attn=args.prefer_attn,
                     checkpoint_regex=args.checkpoint_regex)
    else:
        ap.error("one of --config or --glob is required")


if __name__ == "__main__":
    main()
