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

IMPORTANT — runs are identified by CONFIG, not by name. wandb_utils.py's
run_slug() builds the W&B run *name* differently per loss: for prefix_overlap
it embeds lr/root-mode/aux/anneal (e.g. "..._prob_singleM4_L8_lr1e-05_..."),
but for jsd (and any other non-tree/non-enrichment loss) it is JUST
"jsd_K{K}_{dataset}_s{seed}" -- lr/warmup/lr_min/wd are never in the name at
all. The checkpoint --output directory fragments used elsewhere in this repo
(e.g. "jsd_lr1e-5_wu10_lrmin0.1_wd0.01", "po_prob_lr1e-6_wu20") are a
DIFFERENT string chosen at launch time, NOT the W&B run name -- matching one
against the other silently finds nothing. `config=vars(args)` is always fully
logged regardless of naming, so every identification here is done through
r.config fields (loss, lr, lr_min_ratio, weight_decay, prefix_anneal_steps,
prefix_root_spacing, ...), which is robust to the naming scheme entirely.

USAGE (remote box, W&B auth already present):
    python scripts/passk_vs_jsd_above_floor.py
    python scripts/passk_vs_jsd_above_floor.py --jsd_lr 1e-5 --jsd_lr_min_ratio 0.1 --jsd_weight_decay 0.01
    python scripts/passk_vs_jsd_above_floor.py --ceanneal_only   # only CE-anneal runs (prefix_anneal_steps>0)
"""
import argparse
import math
import statistics
import sys

PASSK_KS = [2, 4, 8, 16]
BEST_BE_KEYS = ["val/best_block_eff", "best_val_block_eff", "val/block_eff", "best_block_eff"]

# The already-established "flat" low-LR prob cluster (~0.11 BE spread, same
# cluster scripts/analyze_passk_movement.py uses) -- identified by CONFIG
# (single-root prob, no CE-anchor, matching lr_min/wd) not by name.
NOISE_CLUSTER_LRS = [1e-6, 3e-6, 7e-6, 1e-5]
DEFAULT_FLOOR = 0.05


def is_noise_cluster_member(cfg):
    if cfg.get("loss") != "prefix_overlap" or cfg.get("prefix_objective") != "prob":
        return False
    if (cfg.get("prefix_root_spacing") or 0) > 0:          # single-root only
        return False
    if (cfg.get("prefix_anneal_steps") or 0) > 0:           # no CE-anchor
        return False
    lr = cfg.get("lr")
    return any(lr is not None and abs(lr - target) / target < 1e-6 for target in NOISE_CLUSTER_LRS)


def is_jsd_anchor(cfg, jsd_lr, jsd_lr_min_ratio, jsd_weight_decay, tol=1e-9):
    if cfg.get("loss") != "jsd":
        return False
    lr = cfg.get("lr")
    if lr is None or abs(lr - jsd_lr) / jsd_lr > 1e-6:
        return False
    lrm = cfg.get("lr_min_ratio")
    if lrm is not None and abs(lrm - jsd_lr_min_ratio) > 1e-6:
        return False
    wd = cfg.get("weight_decay")
    if wd is not None and abs(wd - jsd_weight_decay) > 1e-6:
        return False
    return True


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


def derive_floor(cluster_passk):
    """floor[k] = 2 x std(passk-at-deployment) across the noise cluster members
    actually found (from data already pulled in this run)."""
    floor = {}
    for k in PASSK_KS:
        vals = [d["passk"][k] for d in cluster_passk if k in d["passk"]]
        floor[k] = round(2 * statistics.stdev(vals), 4) if len(vals) >= 3 else DEFAULT_FLOOR
    return floor


def label(cfg):
    """Human-readable identifier built from CONFIG (not the ambiguous wandb name)."""
    if cfg.get("loss") == "jsd":
        return f"jsd_lr{cfg.get('lr'):g}_lrmin{cfg.get('lr_min_ratio')}_wd{cfg.get('weight_decay')}"
    root = f"multiN{cfg.get('prefix_root_spacing')}" if (cfg.get("prefix_root_spacing") or 0) > 0 \
        else f"singleM{cfg.get('prefix_M')}"
    tag = f"prob_{root}_lr{cfg.get('lr'):g}"
    if (cfg.get("prefix_anneal_steps") or 0) > 0:
        tag += f"_ce{cfg.get('prefix_aux_weight')}anneal{cfg.get('prefix_anneal_steps')}"
    return tag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", default="rmukund16-indian-institute-of-technology-hyderabad")
    ap.add_argument("--project", default="distillspec-pipeline")
    ap.add_argument("--jsd_lr", type=float, default=1e-5)
    ap.add_argument("--jsd_lr_min_ratio", type=float, default=0.1)
    ap.add_argument("--jsd_weight_decay", type=float, default=0.01)
    ap.add_argument("--ceanneal_only", action="store_true",
                    help="only include prob runs with prefix_anneal_steps>0 (CE-anneal group)")
    args = ap.parse_args()

    try:
        import wandb
    except ImportError:
        sys.exit("needs `wandb` (pip install wandb) and W&B auth on this machine")

    print(f"[wandb] querying {args.entity}/{args.project} ...")
    all_runs = list(wandb.Api().runs(f"{args.entity}/{args.project}"))
    print(f"[wandb] {len(all_runs)} runs found; pulling history per relevant run (slow-ish)...")

    jsd_run, cluster_hits, prob_hits = None, [], []
    for r in all_runs:
        cfg = r.config
        if is_jsd_anchor(cfg, args.jsd_lr, args.jsd_lr_min_ratio, args.jsd_weight_decay):
            best_be, best_step, passk = passk_at_deployment(r)
            if best_be is not None and (jsd_run is None or best_be > jsd_run["best_be"]):
                jsd_run = {"run": r, "cfg": cfg, "url": r.url, "best_be": best_be,
                          "best_step": best_step, "passk": passk}
            continue
        is_prob = cfg.get("loss") == "prefix_overlap" and cfg.get("prefix_objective") == "prob"
        if not is_prob:
            continue
        if args.ceanneal_only and (cfg.get("prefix_anneal_steps") or 0) <= 0:
            continue
        best_be, best_step, passk = passk_at_deployment(r)
        if best_be is None:
            continue
        entry = {"run": r, "cfg": cfg, "url": r.url, "best_be": best_be,
                 "best_step": best_step, "passk": passk}
        prob_hits.append(entry)
        if is_noise_cluster_member(cfg):
            cluster_hits.append(entry)

    if jsd_run is None:
        sys.exit(f"no jsd run found matching lr={args.jsd_lr}, lr_min_ratio={args.jsd_lr_min_ratio}, "
                 f"weight_decay={args.jsd_weight_decay} -- adjust --jsd_lr/--jsd_lr_min_ratio/"
                 f"--jsd_weight_decay, or check r.config['loss']=='jsd' runs exist in this project")

    floor = derive_floor(cluster_hits)
    print(f"\n[noise floor] derived from {len(cluster_hits)} noise-cluster runs (lrs {NOISE_CLUSTER_LRS}): "
          + ", ".join(f"k{k}=±{floor[k]}" for k in PASSK_KS))
    print(f"[jsd anchor] {label(jsd_run['cfg'])}  best_be={jsd_run['best_be']:.3f} @ step {jsd_run['best_step']}  "
          f"passk={ {k: round(v,4) for k,v in jsd_run['passk'].items()} }")
    print(f"             {jsd_run['url']}")

    rows = []
    for d in prob_hits:
        if d["run"].id == jsd_run["run"].id:
            continue
        d_be = d["best_be"] - jsd_run["best_be"]
        d_pk = {k: (d["passk"][k] - jsd_run["passk"][k]) if k in d["passk"] and k in jsd_run["passk"] else None
                for k in PASSK_KS}
        clears = [k for k in PASSK_KS if d_pk[k] is not None and d_pk[k] >= floor[k]]
        rows.append((label(d["cfg"]), d, d_be, d_pk, clears))

    rows.sort(key=lambda t: -(t[2] if t[2] is not None else -99))

    print(f"\n{'run (from config)':<45} {'best_be':>8} {'Δbe_vs_jsd':>11}   "
          + "  ".join(f"{'Δk'+str(k)+'_vs_jsd':>14}" for k in PASSK_KS) + "   clears_floor")
    for name, d, d_be, d_pk, clears in rows:
        pk_str = "  ".join(f"{(round(d_pk[k],4) if d_pk[k] is not None else '-'):>14}" for k in PASSK_KS)
        flag = ",".join(f"k{k}" for k in clears) if clears else "-"
        print(f"{name:<45} {d['best_be']:>8.3f} {d_be:>11.3f}   {pk_str}   {flag}")
        print(f"{'':<45} {d['url']}")

    winners = [(name, d, cl) for name, d, _, _, cl in rows if cl]
    print("\n=== SUMMARY ===")
    if winners:
        print(f"{len(winners)} run(s) beat jsd's deployed pass@k above the derived noise floor at >=1 k:")
        for name, d, cl in winners:
            tradeoff = "BE still below jsd" if d["best_be"] < jsd_run["best_be"] else "BE also >= jsd"
            print(f"  {name}: clears at k={cl}  (best_be={d['best_be']:.3f} vs jsd={jsd_run['best_be']:.3f}, {tradeoff})")
            print(f"    {d['url']}")
    else:
        print("No run's pass@k-at-deployment clears the derived noise floor above jsd's own "
              "deployed pass@k. (Matches the note's existing 'jsd vs prob fair comparison' "
              "finding: prob's pass@k 'win' only appears best-of-sweep-after-seeing-pass@k, "
              "not at each family's real deployed pick.)")


if __name__ == "__main__":
    main()
