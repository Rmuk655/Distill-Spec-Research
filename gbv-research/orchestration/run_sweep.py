"""
run_sweep.py — Launch a W&B hyperparameter sweep for SpecDist distillation.

This script:
  1. Reads orchestration/configs/sweep.yaml (or a custom --sweep_config file)
  2. Registers the sweep with W&B (wandb.sweep) → prints a sweep ID
  3. Starts a local sweep agent (wandb.agent) that calls train_qwen3.py
     once per trial with the hyperparameters sampled by the sweep controller

Usage
-----
# One-time: register sweep + run 20 trials on this machine
    python orchestration/run_sweep.py --count 20

# Run more agents in parallel (each on a separate GPU / machine):
    wandb agent <entity>/<project>/<sweep_id>

# Use a custom sweep config:
    python orchestration/run_sweep.py --sweep_config my_sweep.yaml --count 10

# Restrict to a single loss (overrides the 'loss' parameter in the YAML):
    python orchestration/run_sweep.py --loss kl --count 12

# Dry-run: print the sweep config without registering anything:
    python orchestration/run_sweep.py --dry_run

Notes
-----
- Each trial runs train_qwen3.py as a subprocess so GPU memory is fully released
  between trials.  wandb.agent() is used in function-call mode (not command mode)
  so we can inject extra fixed args (draft path, dataset path) cleanly.
- The sweep agent reads W&B API key / entity / project from the same
  orchestration/wandb_config.json used by pipeline.py — no extra auth needed.
- Sweep results appear in W&B under the group name set in sweep.yaml
  (default: hparam_sweep_v1).  Use the W&B Parallel Coordinates chart to
  visualise which hyperparameters matter most.
"""

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(HERE)
_OSD_DIR   = os.path.join(os.path.dirname(_REPO_ROOT), "OSD")
_TRAIN_SCRIPT = os.path.join(_OSD_DIR, "train_qwen3.py")
_DEFAULT_SWEEP_CFG = os.path.join(HERE, "configs", "sweep.yaml")
_WANDB_CFG = os.path.join(HERE, "wandb_config.json")


# ---------------------------------------------------------------------------
# W&B config loader (mirrors pipeline.py _load_wandb_config)
# ---------------------------------------------------------------------------

def _load_wandb_config() -> dict:
    """Return {api_key, entity, project} from wandb_config.json if present."""
    if not os.path.exists(_WANDB_CFG):
        return {}
    try:
        with open(_WANDB_CFG, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"  [wandb] Warning: could not read wandb_config.json: {e}")
        return {}


def _apply_wandb_env(wc: dict):
    """Set WANDB_ env vars so subprocess inherits auth."""
    if wc.get("api_key"):
        os.environ["WANDB_API_KEY"] = wc["api_key"]
    if wc.get("entity"):
        os.environ["WANDB_ENTITY"] = wc["entity"]
    if wc.get("project"):
        os.environ["WANDB_PROJECT"] = wc["project"]


# ---------------------------------------------------------------------------
# Sweep launch
# ---------------------------------------------------------------------------

def _load_sweep_yaml(path: str) -> dict:
    """Load and return the sweep config YAML as a dict."""
    try:
        import yaml
    except ImportError:
        sys.exit("pyyaml not found. Run: pip install pyyaml")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not cfg:
        sys.exit(f"Empty or invalid sweep config: {path}")
    return cfg


def _override_loss(sweep_cfg: dict, loss: str) -> dict:
    """Replace the 'loss' parameter in sweep_cfg with a fixed single value."""
    import copy
    cfg = copy.deepcopy(sweep_cfg)
    cfg.setdefault("parameters", {})["loss"] = {"value": loss}
    return cfg


def launch_sweep(sweep_cfg_path: str, count: int, loss_filter: str = None,
                 dry_run: bool = False) -> None:
    """Register a W&B sweep and run `count` trials on this machine.

    Parameters
    ----------
    sweep_cfg_path : str
        Path to sweep.yaml (or custom YAML).
    count : int
        Number of trials to run locally.  Each trial calls train_qwen3.py once.
    loss_filter : str or None
        If set, override the 'loss' parameter to a fixed single value.
    dry_run : bool
        If True, print the config and exit without registering the sweep.
    """
    try:
        import wandb
    except ImportError:
        sys.exit("wandb not found. Run: pip install wandb")

    wc = _load_wandb_config()
    _apply_wandb_env(wc)

    sweep_cfg = _load_sweep_yaml(sweep_cfg_path)
    if loss_filter:
        sweep_cfg = _override_loss(sweep_cfg, loss_filter)
        print(f"  [sweep] Loss locked to: {loss_filter}")

    if dry_run:
        import yaml
        print("\n─── Sweep config (dry-run, not registered) ───────────────────")
        print(yaml.dump(sweep_cfg, default_flow_style=False))
        return

    # Remove the 'program' key — we drive execution ourselves via subprocess
    # so W&B doesn't try to call the script directly (avoids path issues on Windows).
    sweep_cfg.pop("program", None)

    entity  = wc.get("entity")  or os.environ.get("WANDB_ENTITY")
    project = wc.get("project") or os.environ.get("WANDB_PROJECT") or "distillspec"

    print(f"\n  [sweep] Registering sweep → entity={entity}  project={project}")
    sweep_id = wandb.sweep(sweep_cfg, entity=entity, project=project)
    print(f"  [sweep] Sweep ID: {sweep_id}")
    print(f"  [sweep] Dashboard: https://wandb.ai/{entity}/{project}/sweeps/{sweep_id}")
    print(f"  [sweep] Starting {count} local agent trial(s)...\n")

    # Save sweep_id for reference (so you can resume / add more agents later)
    _id_file = os.path.join(HERE, ".last_sweep_id")
    with open(_id_file, "w") as f:
        f.write(f"{entity}/{project}/{sweep_id}\n")
    print(f"  [sweep] Sweep ID saved to {_id_file}")
    print(f"  [sweep] To add more agents:  wandb agent {entity}/{project}/{sweep_id}\n")

    def _trial(config=None):
        """Called by wandb.agent() for each trial.  Runs train_qwen3.py as a subprocess."""
        with wandb.init(config=config) as run:
            cfg = dict(wandb.config)

            # Build CLI args from the sampled hyperparams
            cmd = [sys.executable, _TRAIN_SCRIPT]
            _arg_keys = [
                "loss", "lr", "lora_r", "lora_alpha", "lora_dropout",
                "teacher_temp", "ebe_kl_weight", "jsd_alpha",
                "grad_clip", "steps", "draft", "target", "dataset",
                "seed", "wandb_group",
            ]
            for key in _arg_keys:
                if key in cfg:
                    val = cfg[key]
                    if isinstance(val, bool):
                        if val:
                            cmd.append(f"--{key}")
                    else:
                        cmd += [f"--{key}", str(val)]

            # Pass W&B run ID so train_qwen3.py resumes the correct run
            env = {**os.environ, "WANDB_RUN_ID": run.id, "WANDB_RESUME": "allow"}

            print(f"\n  [trial {run.id}] CMD: {' '.join(cmd)}\n")
            result = subprocess.run(cmd, env=env)
            if result.returncode != 0:
                print(f"  [trial {run.id}] ⚠ Non-zero exit code: {result.returncode}")

    wandb.agent(sweep_id, function=_trial, count=count,
                entity=entity, project=project)
    print(f"\n  [sweep] Done — {count} trial(s) completed.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Launch a W&B hyperparameter sweep for SpecDist.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--sweep_config", default=_DEFAULT_SWEEP_CFG,
                   help=f"Path to sweep YAML config (default: {_DEFAULT_SWEEP_CFG})")
    p.add_argument("--count", type=int, default=20,
                   help="Number of trials to run on this machine (default: 20)")
    p.add_argument("--loss", default=None,
                   choices=["forward_kl", "ebe", "reverse_kl", "jsd", "l1"],
                   help="Lock the loss to a single value (overrides sweep.yaml). "
                        "Useful for a focused LR / lora_r sweep on one loss.")
    p.add_argument("--dry_run", action="store_true",
                   help="Print the resolved sweep config without registering.")
    args = p.parse_args()

    if not os.path.exists(args.sweep_config):
        p.error(f"Sweep config not found: {args.sweep_config}")

    launch_sweep(
        sweep_cfg_path=args.sweep_config,
        count=args.count,
        loss_filter=args.loss,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
