#!/usr/bin/env python
"""
passk_vs_jsd_above_floor.py — direct answer to "does CE-anneal push pass@k up
(beating jsd), and is it above the noise floor, or is that at the cost of BE?"

Pulls LIVE from W&B (same runs() API as scripts/mine_wandb_runs.py). For every
prefix_overlap/prob run and the jsd anchor, it reads each run's OWN best_val_
block_eff checkpoint (the one actually deployed as ckpt_best) and the pass@k
values logged nearest that same step -- NOT the run's final-logged pass@k,
which could be a later, already-declined checkpoint. This matches the fairness
discipline already used in notes/hyperparam_sweep_research_report.md ("the ONE
config per family you'd actually deploy, chosen by block_eff before looking at
pass@k") -- comparing every run's own peak-BE checkpoint's pass@k, not a
cherry-picked later point.

Noise floor for pass@k is DERIVED from data (2 x std across the already-
established "flat" low-LR prob cluster), same method and same cluster as
scripts/analyze_passk_movement.py -- not assumed or invented fresh here.

USAGE (remote box, W&B auth already present):
    python scripts/passk_vs_jsd_above_floor.py
    python scripts/passk_vs_jsd_above_floor.py --jsd_anchor jsd_lr1e-5_wu10_lrmin0.1_wd0.01
    python scripts/passk_vs_jsd_above_floor.py --name_filter ceanneal   # only CE-anneal runs
"""
import argparse
import math
import statistics
import sys

PASSK_KS = [2, 4, 8, 16]
BEST_BE_KEYS = ["val/best_block_eff", "best_val_block_eff", "val/block_eff", "best_block_eff"]

# Same "flat" low-LR prob cluster used in scripts/analyze_passk_movement.py to
# derive a per-k pass@k noise floor from data already established as
# statistically indistinguishable in block_eff (~0.11 BE spread).
NOISE_CLUSTER_PROB = [
    "po_prob_lr1e-6_wu20", "po_prob_lr3e-6_wu20", "po_prob_lr7e-6_wu20",
    "po_prob_lr1e-5_wu20_lrmin0.1_wd0.01",
]
DEFAULT_FLOOR = 0.05


def get_best_be(summary):
    for k in BEST_BE_KEYS:
        v = summary.get(k)
        if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
            return float(v)
    return None


def passk_at_deployment(run):
    """Pull val/block_eff + passk_math_val/k{...} history; return (best_be, best_be_step,
    {k: passk value at the nearest passk-logged step <= best_be_step})."""
    keys = ["train/step", "val/block_eff"] + [f"passk_math_val/k{k}" for k in PASSK_KS]
    try:
        h = run.history(keys=keys, samples=6000)
    except Exception:
        return None, None, {}
    if h is None or h.empty or "val/block_eff" not in h or h["val/block_eff"].dropna().empty:
        return None, None, {}
    vb = h["val/block_eff"].dropna()
    best_idx = vb.idxmax()
    best_be = float(vb.loc[best_idx])
    best_step = h["train/step"].loc[best_idx] if "train/step" in h else best_idx

    passk = {}
    for k in PASSK_KS:
        col = f"passk_math_val/k{k}"
        if col not in h:
            continue
        sub = h[[col]].dropna()
        if sub.empty:
            continue
        # nearest logged passk row at or before best_idx (fall back to nearest overall)
        prior = sub.loc[:best_idx]
        row = prior.iloc[-1] if not prior.empty else sub.iloc[0]
        passk[k] = float(row[col])
    return best_be, best_step, passk


def derive_floor(cache):
    """floor[k] = 2 x std(passk-at-deployment) across NOISE_CLUSTER_PROB, from data
    already pulled in this run (not a second wandb round-trip)."""
    floor = {}
    for k in PASSK_KS:
        vals = [cache[n]["passk"][k] for n in NOISE_CLUSTER_PROB
                if n in cache and k in cache[n]["passk"]]
        floor[k] = round(2 * statistics.stdev(vals), 4) if len(vals) >= 3 else DEFAULT_FLOOR
    return floor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", default="rmukund16-indian-institute-of-technology-hyderabad")
    ap.add_argument("--project", default="distillspec-pipeline")
    ap.add_argument("--jsd_anchor", default="jsd_lr1e-5_wu10_lrmin0.1_wd0.01",
                    help="jsd run whose deployed (best-BE) pass@k is the anchor to beat")
    ap.add_argument("--name_filter", default=None,
                    help="only include prob runs whose name contains this substring "
                         "(e.g. 'ceanneal' to isolate the CE-anneal group)")
    args = ap.parse_args()

    try:
        import wandb
    except ImportError:
        sys.exit("needs `wandb` (pip install wandb) and W&B auth on this machine")

    print(f"[wandb] querying {args.entity}/{args.project} ...")
    all_runs = list(wandb.Api().runs(f"{args.entity}/{args.project}"))
    print(f"[wandb] {len(all_runs)} runs found; pulling history per run (slow-ish)...")

    cache = {}   # name -> {"run":, "url":, "best_be":, "best_step":, "passk": {k: v}}
    for r in all_runs:
        name = r.name
        # only bother pulling history for runs we might actually use: jsd anchor,
        # the noise-floor cluster, and prob/prefix_overlap runs (optionally filtered)
        cfg = r.config
        is_prob = cfg.get("loss") == "prefix_overlap" and cfg.get("prefix_objective") == "prob"
        is_relevant = (name == args.jsd_anchor) or (name in NOISE_CLUSTER_PROB) or is_prob
        if not is_relevant:
            continue
        if args.name_filter and is_prob and args.name_filter not in name:
            continue
        best_be, best_step, passk = passk_at_deployment(r)
        if best_be is None:
            continue
        cache[name] = {"run": r, "url": r.url, "best_be": best_be,
                       "best_step": best_step, "passk": passk}

    if args.jsd_anchor not in cache:
        sys.exit(f"jsd anchor run '{args.jsd_anchor}' not found or has no usable history — "
                 f"check the exact run name in W&B and pass --jsd_anchor")

    floor = derive_floor(cache)
    jsd = cache[args.jsd_anchor]
    print(f"\n[noise floor, derived from {NOISE_CLUSTER_PROB}]  "
          + ", ".join(f"k{k}=±{floor[k]}" for k in PASSK_KS))
    print(f"[jsd anchor] {args.jsd_anchor}  best_be={jsd['best_be']:.3f} @ step {jsd['best_step']}  "
          f"passk={ {k: round(v,4) for k,v in jsd['passk'].items()} }")
    print(f"             {jsd['url']}")

    rows = []
    for name, d in cache.items():
        if name == args.jsd_anchor:
            continue
        d_be = d["best_be"] - jsd["best_be"]
        d_pk = {k: (d["passk"][k] - jsd["passk"][k]) if k in d["passk"] and k in jsd["passk"] else None
                for k in PASSK_KS}
        clears = [k for k in PASSK_KS if d_pk[k] is not None and d_pk[k] >= floor[k]]
        rows.append((name, d, d_be, d_pk, clears))

    rows.sort(key=lambda t: -(t[2] if t[2] is not None else -99))  # by BE delta vs jsd

    print(f"\n{'run':<45} {'best_be':>8} {'Δbe_vs_jsd':>11}   "
          + "  ".join(f"{'Δk'+str(k)+'_vs_jsd':>14}" for k in PASSK_KS) + "   clears_floor")
    for name, d, d_be, d_pk, clears in rows:
        pk_str = "  ".join(f"{(round(d_pk[k],4) if d_pk[k] is not None else '-'):>14}" for k in PASSK_KS)
        flag = ",".join(f"k{k}" for k in clears) if clears else "-"
        print(f"{name:<45} {d.get('best_be'):>8.3f} {d_be:>11.3f}   {pk_str}   {flag}")
        print(f"{'':<45} {d['url']}")

    winners = [(n, cl) for n, _, _, _, cl in rows if cl]
    print("\n=== SUMMARY ===")
    if winners:
        print(f"{len(winners)} run(s) beat jsd's deployed pass@k above the derived noise floor "
              f"at >=1 k:")
        for n, cl in winners:
            d = cache[n]
            tradeoff = "BE still below jsd" if d["best_be"] < jsd["best_be"] else "BE also >= jsd"
            print(f"  {n}: clears at k={cl}  (best_be={d['best_be']:.3f} vs jsd={jsd['best_be']:.3f}, {tradeoff})")
            print(f"    {d['url']}")
    else:
        print("No run's pass@k-at-deployment clears the derived noise floor above jsd's own "
              "deployed pass@k. (Matches the note's existing 'jsd vs prob fair comparison' "
              "finding: prob's pass@k 'win' only appears best-of-sweep-after-seeing-pass@k, "
              "not at each family's real deployed pick.)")


if __name__ == "__main__":
    main()
