"""
wandb_utils.py — W&B run naming and initialization for the training pipeline.

Handles: run slug (checkpoint dir name + W&B run name), run creation/resume,
and metadata persistence so a killed-and-resumed run continues the same
dashboard URL instead of splitting into two runs.
"""
from __future__ import annotations

import json
import os

from losses import (is_tree_loss, is_enrichment_loss, is_flat_enrich_loss,
                    is_prefix_overlap_loss)


def run_slug(args) -> str:
    """Loss + dataset component of the run identifier, shared by the checkpoint dir
    and the W&B run name. L is omitted for flat/flat-enrich losses where tree depth
    is not a parameter; K is always included since it may distinguish enrich width."""
    if is_prefix_overlap_loss(args.loss):
        # K is meaningless for prefix_overlap; encode the flags that actually
        # distinguish runs in a sweep (objective, root mode, L, LR, aux, warm-start).
        root = (f"multiN{args.prefix_root_spacing}"
                if args.prefix_root_spacing > 0 else f"singleM{args.prefix_M}")
        slug = (f"{args.loss}_{args.prefix_objective}_{root}"
                f"_L{args.prefix_L}_lr{args.lr:g}_{args.train_dataset}_s{args.seed}")
        if args.prefix_aux_weight > 0:
            slug += f"_{args.prefix_aux}{args.prefix_aux_weight:g}"
            if args.prefix_anneal_steps > 0:
                slug += f"anneal{args.prefix_anneal_steps}"
        if args.draft:
            slug += "_warm"
        return slug
    uses_L = is_tree_loss(args.loss) or is_enrichment_loss(args.loss)
    if uses_L:
        slug = f"{args.loss}_K{args.K}_L{args.L}_{args.train_dataset}_s{args.seed}"
    else:
        slug = f"{args.loss}_K{args.K}_{args.train_dataset}_s{args.seed}"
    if args.aux_mode == "depth_weight":
        tag = "lin" if args.depth_linear else f"lam{args.depth_lambda}"
        slug += f"+dw_{args.aux_loss or 'naive_tree'}_{tag}"
    elif args.aux_loss:
        slug += f"+{args.aux_loss}x{args.aux_weight}"
    return slug


def _define_step_metric(run):
    """Drive every chart's x-axis off an explicit `train/step` field instead of
    W&B's internal auto-increment step counter.

    Why this matters for resume: we log every LOG_EVERY steps but checkpoint only
    every SAVE_EVERY steps, so on a kill+resume the W&B history is *ahead* of
    ckpt_latest. Replaying those steps with an explicit `step=` arg makes W&B
    drop/overwrite them ("step must be monotonically increasing"), which shows up
    as a gap + overwritten train/loss, train/lr, train/grad_norm. Logging against
    a defined `train/step` metric (and NOT passing step=) lets W&B append cleanly
    and plot by true training step regardless of the internal counter."""
    try:
        run.define_metric("train/step")
        run.define_metric("*", step_metric="train/step")
    except Exception as e:
        print(f"[wandb] define_metric skipped ({e})")


def setup_wandb(args, output_dir, resumed: bool, wandb_project: str):
    """Initialise W&B, resuming the saved run id if <output>/wandb_run.json exists."""
    if args.no_wandb:
        print("[wandb] disabled (--no_wandb)")
        return None
    try:
        import wandb
    except ImportError:
        print("[wandb] not installed — skipping (pip install wandb to enable)")
        return None

    meta_path = os.path.join(output_dir, "wandb_run.json")
    saved = None
    if resumed and os.path.isfile(meta_path) and not args.fresh_wandb:
        try:
            saved = json.load(open(meta_path, encoding="utf-8"))
        except Exception as e:
            print(f"[wandb] WARNING: could not read {meta_path} ({e}) — starting fresh W&B run")
            saved = None
    elif resumed and not os.path.isfile(meta_path):
        print(f"[wandb] no saved run ID at {meta_path} — starting fresh W&B run")

    tags = [args.loss, args.train_dataset, f"K{args.K}", f"L{args.L}"]
    if args.aux_mode == "depth_weight":
        tags.append(f"depthw:{args.aux_loss or 'naive_tree'}")
        tags.append("lin" if args.depth_linear else f"lam{args.depth_lambda}")
    elif args.aux_loss:
        tags.append(f"aux:{args.aux_loss}")
    run_name = run_slug(args)
    init_kw = dict(project=wandb_project, name=run_name,
                   tags=tags, config=vars(args))
    if saved:
        init_kw["id"]     = saved["run_id"]
        init_kw["resume"] = "must"
        try:
            run = wandb.init(**init_kw)
            _define_step_metric(run)
            print(f"[wandb] resumed run {saved['run_id']}: {run.url}")
            return run
        except Exception as e:
            print(f"[wandb] resume failed ({e}); starting fresh run")
            init_kw.pop("id", None)

    init_kw["resume"] = "allow"
    run = wandb.init(**init_kw)
    _define_step_metric(run)
    json.dump({"run_id": run.id, "name": run.name, "project": wandb_project, "url": run.url},
              open(meta_path, "w", encoding="utf-8"))
    print(f"[wandb] {run.url}")
    return run