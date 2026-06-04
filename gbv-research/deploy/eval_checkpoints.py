"""
eval_checkpoints.py — sweep milestone checkpoints and build convergence curves.

For each milestone checkpoint saved during training (at milestone_every=500
grad-steps), this script:
  1. Merges the LoRA adapter into a temporary full model
  2. Runs evaluate.py at n=20 prompts (alpha + gbv only — fast, ~5 min/ckpt)
  3. Stores alpha / block_eff / task_score vs train_step in:
     - results.db  →  checkpoint_evals table
     - W&B         →  checkpoint_eval/alpha, checkpoint_eval/be_gbv, etc.

This produces the convergence curves needed for the paper (Figure X: alpha/BE
vs training steps for each loss). Without this, you only have the final eval
value — no evidence that the metric improved monotonically.

Usage (run AFTER training finishes, BEFORE the final merged eval):
    python deploy/eval_checkpoints.py \\
        --config a100_qwen \\
        --loss kl \\
        --storage_root /home/colligo/specdist

    # Or for all losses after a full training run:
    for loss in kl rev_kl jsd l1 kl_tree gbv_tree traversal_tree; do
        python deploy/eval_checkpoints.py \\
            --config a100_qwen --loss $loss \\
            --storage_root /home/colligo/specdist
    done

Output in W&B run:
    checkpoint_eval/alpha_mean     vs train_step
    checkpoint_eval/be_gbv         vs train_step
    checkpoint_eval/be_traversal   vs train_step
    checkpoint_eval/task_score     vs train_step  (if --task_score flag)

Output in results.db:
    checkpoint_evals table — queryable by label / loss_name / train_step
"""

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_GBV  = os.path.dirname(_HERE)

sys.path.insert(0, os.path.join(_GBV, "db"))
sys.path.insert(0, os.path.join(_GBV, "orchestration"))


def _parse_args():
    p = argparse.ArgumentParser(description="Sweep milestone checkpoints for convergence curves")
    p.add_argument("--config", default="a100_qwen", help="YAML config name (e.g. a100_qwen)")
    p.add_argument("--loss",   required=True,        help="Loss name, e.g. kl, gbv_tree")
    p.add_argument("--storage_root", default=None,
                   help="Root of persistent storage (default: uses db/ inside repo)")
    p.add_argument("--n",      type=int, default=20,
                   help="Prompts per checkpoint eval (default 20 — fast but representative)")
    p.add_argument("--modes",  default="alpha,gbv",
                   help="Verifier modes to eval at each checkpoint (default: alpha,gbv)")
    p.add_argument("--task_score", action="store_true",
                   help="Also measure GSM8K accuracy at each checkpoint (~2 min extra/ckpt)")
    p.add_argument("--wandb_project", default="distillspec")
    p.add_argument("--experiment_tag", default="checkpoint_sweep")
    p.add_argument("--dry_run", action="store_true", help="Print plan but don't run evals")
    return p.parse_args()


def _load_config(config_name):
    """Load YAML config (with base resolution) via experiment._load_config_yaml."""
    import experiment as _exp
    return _exp._load_config_yaml(config_name)


def _run_tag_for_loss(loss: str, cfg: dict) -> str:
    """Reproduce the checkpoint directory name: {loss}-gsm8k-{pair_tag}."""
    draft  = cfg.get("draft", "Qwen/Qwen3-0.6B")
    target = cfg.get("target", "Qwen/Qwen3-8B")
    # _run_tag logic from experiment.py: abbreviate model IDs
    def _abbrev(m):
        m = m.split("/")[-1].lower()
        m = re.sub(r"qwen3?[-_]?", "q", m)
        m = re.sub(r"llama[-_]?3\.?2?[-_]?", "l3", m)
        m = re.sub(r"instruct", "i", m)
        m = re.sub(r"[^a-z0-9]", "", m)
        return m[:8]
    load4 = cfg.get("load_in_4bit", False)
    pair  = f"{_abbrev(draft)}-{_abbrev(target)}" + ("-4b" if load4 else "")
    # Loss-to-shortname map (from experiment.py _ckpt helper)
    _SHORT = {
        "kl": "kl", "rev_kl": "rev_kl", "jsd": "jsd", "l1": "l1",
        "ebe": "ebe", "ebe_single": "ebe_single",
        "kl_tree": "kl_tree", "rev_kl_tree": "rev_kl_tree", "jsd_tree": "jsd_tree",
        "bv_tree": "bv_tree", "gbv_tree": "gbv_tree", "traversal_tree": "trav_tree",
        "naive_tree": "naive_tree", "nss_tree": "nss_tree",
        "specinfer_tree": "si_tree", "spectr_tree": "st_tree", "khisti_tree": "kh_tree",
    }
    short = _SHORT.get(loss, loss.replace("_", ""))
    return f"{short}-gsm8k-{pair}"


def find_milestone_checkpoints(ckpt_dir: str):
    """Return sorted list of (train_step, checkpoint_path) for all milestones."""
    pattern = os.path.join(ckpt_dir, "milestone_*")
    milestones = []
    for path in glob.glob(pattern):
        basename = os.path.basename(path)
        m = re.match(r"milestone_(\d+)", basename)
        if m and os.path.exists(os.path.join(path, "adapter_config.json")):
            milestones.append((int(m.group(1)), path))
    return sorted(milestones)


def merge_lora_temp(adapter_path: str, draft_model: str) -> str:
    """Merge LoRA adapter into a temp directory. Returns temp dir path."""
    tmp = tempfile.mkdtemp(prefix="ckpt_merge_")
    _TRAIN = os.path.join(_GBV, "algorithms", "distillspec_gbv", "trainer.py")
    cmd = [
        sys.executable, _TRAIN,
        "--merge_only",
        "--adapter", adapter_path,
        "--draft",   draft_model,
        "--output",  tmp,
    ]
    print(f"    Merging LoRA → {tmp} ...")
    result = subprocess.run(cmd, cwd=_GBV, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    [WARN] merge failed: {result.stderr[-500:]}")
        shutil.rmtree(tmp, ignore_errors=True)
        return None
    return tmp


def run_eval(merged_path: str, target: str, n: int, modes: str,
             task_score: bool, cfg: dict, experiment_tag: str,
             db_path: str = None) -> dict:
    """Run evaluate.py on the merged checkpoint. Returns {mode: {alpha, be, ...}}."""
    _EVAL = os.path.join(_GBV, "orchestration", "evaluate.py")
    env = os.environ.copy()
    if db_path:
        env["SPECDIST_DB_PATH"] = db_path

    cmd = [
        sys.executable, _EVAL,
        "--student",  merged_path,
        "--teacher",  target,
        "--student_label", "checkpoint_sweep",
        "--datasets", "gsm8k",
        "--modes",    modes,
        "--K",        str(cfg.get("eval_K_values", [3])[0] if isinstance(cfg.get("eval_K_values"), list) else 3),
        "--temperature", "1.0",
        "--n",        str(n),
        "--skip_fetch",
        "--hw_tier",  cfg.get("hw_tier", "a100"),
        "--model_family", cfg.get("model_family", "qwen"),
        "--no_wandb",   # convergence W&B logging handled by this script
        "--experiment_tag", experiment_tag,
    ]
    if task_score:
        cmd.append("--task_score")
        cmd += ["--task_batch", str(cfg.get("eval_task_batch", 8))]

    result = subprocess.run(cmd, cwd=_GBV, capture_output=True, text=True, env=env)
    output = result.stdout + result.stderr

    # Parse results from output
    metrics = {}
    for line in output.splitlines():
        # "alpha=0.5123 ±0.0234"
        m = re.search(r"alpha=([0-9.]+)\s*±([0-9.]+)", line)
        if m:
            metrics.setdefault("alpha", {})
            metrics["alpha"]["alpha_mean"] = float(m.group(1))
            metrics["alpha"]["alpha_ci95"] = float(m.group(2))
        # "block_eff=3.7714"
        m = re.search(r"block_eff=([0-9.]+)", line)
        if m:
            mode_m = re.search(r"\[(\w+)\]", line)
            mode = mode_m.group(1) if mode_m else "be"
            metrics.setdefault(mode, {})
            metrics[mode]["block_eff"] = float(m.group(1))
        # "task_score=0.290"
        m = re.search(r"task_score=([0-9.]+)", line)
        if m:
            metrics.setdefault("alpha", {})
            metrics["alpha"]["task_score"] = float(m.group(1))

    return metrics


def main():
    args = _parse_args()
    cfg  = _load_config(args.config)

    draft  = cfg.get("draft",  "Qwen/Qwen3-0.6B")
    target = cfg.get("target", "Qwen/Qwen3-8B")

    # Resolve checkpoint root
    storage = args.storage_root or os.environ.get("STORAGE_ROOT") or os.path.join(_GBV, "db")
    ckpt_root   = os.path.join(storage, "checkpoints")
    run_label   = _run_tag_for_loss(args.loss, cfg)
    ckpt_dir    = os.path.join(ckpt_root, run_label)
    db_path     = os.path.join(storage, "results.db")
    os.environ["SPECDIST_DB_PATH"] = db_path

    print(f"\n{'='*60}")
    print(f"  Checkpoint sweep: {args.loss}")
    print(f"  Checkpoint dir  : {ckpt_dir}")
    print(f"  DB              : {db_path}")
    print(f"  n per eval      : {args.n}")
    print(f"  modes           : {args.modes}")
    print(f"{'='*60}")

    milestones = find_milestone_checkpoints(ckpt_dir)
    if not milestones:
        print(f"\n  [WARN] No milestone checkpoints found in {ckpt_dir}")
        print(f"  Are you running this after training? (milestone_every=500 by default)")
        return

    print(f"\n  Found {len(milestones)} milestone(s):")
    for step, path in milestones:
        print(f"    step {step:5d}: {os.path.basename(path)}")

    if args.dry_run:
        print("\n  [dry-run] Would evaluate each milestone at n={args.n}.")
        return

    # Import DB functions
    import results_db
    results_db.DB_PATH = db_path

    # Optionally init W&B for convergence curve logging
    _wandb = None
    try:
        import wandb
        _wandb = wandb.init(
            project=args.wandb_project,
            name=f"ckpt_sweep_{args.loss}_{run_label}",
            job_type="checkpoint_sweep",
            tags=[args.loss, "checkpoint_sweep", args.experiment_tag],
            config={"loss": args.loss, "n_prompts": args.n, "modes": args.modes},
            reinit="allow",
        )
        print(f"\n  W&B: {_wandb.url}")
    except Exception as e:
        print(f"  [W&B] Skipped: {e}")

    for train_step, adapter_path in milestones:
        print(f"\n  ── Step {train_step} ──────────────────────────────────────────")

        merged = merge_lora_temp(adapter_path, draft)
        if merged is None:
            print(f"    [SKIP] Merge failed for step {train_step}")
            continue

        try:
            metrics = run_eval(
                merged_path=merged, target=target,
                n=args.n, modes=args.modes, task_score=args.task_score,
                cfg=cfg, experiment_tag=args.experiment_tag, db_path=db_path,
            )

            wandb_row = {"train_step": train_step}

            for mode, vals in metrics.items():
                alpha   = vals.get("alpha_mean")
                ci95    = vals.get("alpha_ci95")
                be      = vals.get("block_eff")
                tscore  = vals.get("task_score")

                # Store in DB
                results_db.insert_checkpoint_eval(
                    label=run_label, loss_name=args.loss, train_step=train_step,
                    mode=mode, n_prompts=args.n,
                    alpha_mean=alpha, alpha_ci95=ci95,
                    block_eff=be, task_score=tscore,
                    experiment_tag=args.experiment_tag,
                )

                print(f"    [{mode}] alpha={alpha:.4f}±{ci95:.4f}" if alpha else
                      f"    [{mode}] BE={be:.4f}" if be else f"    [{mode}] (no result)")

                # Build W&B log row
                if alpha is not None:
                    wandb_row[f"checkpoint_eval/{mode}_alpha"] = alpha
                if be is not None:
                    wandb_row[f"checkpoint_eval/{mode}_be"] = be
                if tscore is not None:
                    wandb_row[f"checkpoint_eval/task_score"] = tscore

            if _wandb:
                _wandb.log(wandb_row)

        finally:
            shutil.rmtree(merged, ignore_errors=True)

    if _wandb:
        _wandb.finish()

    print(f"\n  Done. Convergence data in DB (checkpoint_evals table) and W&B.")
    print(f"  To query: python -c \"import results_db; print(results_db.query_checkpoint_evals(loss_name='{args.loss}'))\"")


if __name__ == "__main__":
    main()
