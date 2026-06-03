"""
aip_run.py — SpecDist pipeline launcher for GPU compute environments.

Usage (run from the repo root or any directory):
    python deploy/aip_run.py                          # smoke test first
    python deploy/aip_run.py --config a10_qwen        # A10G 24 GB
    python deploy/aip_run.py --config a100_qwen       # A100 40 GB
    python deploy/aip_run.py --config a10_qwen --losses kl
    python deploy/aip_run.py --config a10_qwen --losses kl_tree --no_smoke
    python deploy/aip_run.py --config a100_qwen --resume   # skip smoke, resume

Environment variables (set before running):
    WANDB_API_KEY   — W&B API key from https://wandb.ai/authorize
    HF_TOKEN        — HuggingFace token (optional; Qwen3 is public)
    STORAGE_ROOT    — where to write checkpoints/logs/results.db
                      defaults to  <repo>/db/  (local)
                      set to a persistent network path for cloud environments

GPU → config mapping:
    nvidia-smi shows 24 GB  (A10G)  →  a10_qwen    8B BF16, 2000 steps
    nvidia-smi shows 40 GB  (A100)  →  a100_qwen   8B BF16, 2000 steps  (paper)
    nvidia-smi shows 80 GB  (A100)  →  a100_qwen   same config
    No GPU                          →  server_gpt2 CPU-only, GPT-2
"""

import argparse
import os
import subprocess
import sys

# ── Locate repo root ──────────────────────────────────────────────────────────
_THIS_FILE  = os.path.abspath(__file__)
_DEPLOY_DIR = os.path.dirname(_THIS_FILE)
_GBV_DIR    = os.path.dirname(_DEPLOY_DIR)          # gbv-research/
_REPO_DIR   = os.path.dirname(_GBV_DIR)             # Distill-Spec-Research/


def _parse_args():
    p = argparse.ArgumentParser(
        description="SpecDist pipeline launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python deploy/aip_run.py                            # smoke then full (a10_qwen)
  python deploy/aip_run.py --config a100_qwen         # A100 paper run
  python deploy/aip_run.py --config a10_qwen --losses kl
  python deploy/aip_run.py --config a10_qwen --losses kl_tree --no_smoke
  python deploy/aip_run.py --resume                   # skip smoke, resume pipeline
        """,
    )
    p.add_argument("--config",      default="a10_qwen",
                   help="Config name (default: a10_qwen). "
                        "Options: a10_qwen, a100_qwen, kaggle, colab, colab_lite, "
                        "server_gpt2, laptop_qwen, laptop_gpt2")
    p.add_argument("--losses",      default=None,
                   help="Comma-separated loss names to run "
                        "(default: all). E.g. 'kl' or 'kl,kl_tree'")
    p.add_argument("--storage_root", default=None,
                   help="Root directory for checkpoints, results.db, logs. "
                        "Defaults to STORAGE_ROOT env var or <repo>/db/")
    p.add_argument("--no_smoke",       action="store_true",
                   help="Skip preflight smoke test")
    p.add_argument("--resume",         action="store_true",
                   help="Resume without smoke test (implies --no_smoke)")
    p.add_argument("--dry_run",        action="store_true",
                   help="Print the experiment.py command without running it")
    p.add_argument("--experiment_tag", default=None,
                   help="Human-readable tag for this session's eval results "
                        "(default: auto-generated from hostname+timestamp). "
                        "E.g. --experiment_tag rmukundServer")
    return p.parse_args()


def _gpu_info():
    """Print GPU summary and return (n_gpus, total_vram_gb)."""
    try:
        import torch
        n = torch.cuda.device_count()
        if n == 0:
            print("GPU    : none detected")
            return 0, 0
        total = 0
        for i in range(n):
            prop = torch.cuda.get_device_properties(i)
            gb   = prop.total_memory / 1024 ** 3
            total += gb
            print(f"GPU {i}  : {prop.name}  {gb:.0f} GB VRAM")
        return n, total
    except ImportError:
        print("GPU    : torch not installed (install deps first)")
        return 0, 0


def _install_deps():
    """Install PyTorch + ML stack if not already present."""
    try:
        import torch
        import transformers
        print(f"deps   : torch {torch.__version__}  transformers {transformers.__version__}")
        return
    except ImportError:
        pass

    print("deps   : installing (torch + ML stack)...")
    cmds = [
        [sys.executable, "-m", "pip", "install", "-q",
         "torch", "torchvision", "torchaudio",
         "--index-url", "https://download.pytorch.org/whl/cu128"],
        [sys.executable, "-m", "pip", "install", "-q",
         "transformers>=4.51", "peft>=0.10", "accelerate",
         "bitsandbytes>=0.46.1", "datasets", "sentencepiece",
         "wandb", "flask", "pyyaml", "scipy"],
    ]
    for cmd in cmds:
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print("ERROR: dependency install failed")
            sys.exit(1)
    print("deps   : installed")


def _auth():
    """Authenticate W&B and HuggingFace from env vars."""
    wandb_key = os.environ.get("WANDB_API_KEY", "")
    if wandb_key:
        try:
            import wandb
            wandb.login(key=wandb_key, relogin=True)
            print("W&B    : authenticated")
        except Exception as e:
            print(f"W&B    : login failed ({e})")
    else:
        print("W&B    : WANDB_API_KEY not set — W&B logging disabled")

    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        try:
            subprocess.run(
                ["huggingface-cli", "login", "--token", hf_token,
                 "--add-to-git-credential"],
                capture_output=True
            )
            print("HF     : authenticated")
        except Exception:
            print("HF     : huggingface-cli not available (optional)")
    else:
        print("HF     : HF_TOKEN not set (not needed for Qwen3)")


def _resolve_storage_root(args):
    """Determine storage root: CLI arg > env var > repo/db/."""
    if args.storage_root:
        return args.storage_root
    env = os.environ.get("STORAGE_ROOT", "")
    if env:
        return env
    # Default: local db/ directory inside the repo
    return os.path.join(_GBV_DIR, "db")


def _warn_stale_trainers():
    """Warn if leftover trainer/evaluate subprocesses are found — but do NOT kill them.

    Auto-killing would terminate intentionally parallel jobs the user kicked off.
    Instead, print a warning with GPU VRAM usage so the user can decide whether
    to kill manually before starting a new run.
    """
    try:
        result = subprocess.run(
            ["ps", "aux"], capture_output=True, text=True, timeout=5
        )
        keywords = ["trainer.py", "evaluate.py", "runner.py"]
        stale = []
        for line in result.stdout.splitlines():
            if any(kw in line for kw in keywords):
                parts = line.split()
                if len(parts) > 1:
                    try:
                        stale.append((int(parts[1]), line.split()[-1]))
                    except ValueError:
                        pass
        if stale:
            print(f"\n  [WARNING] {len(stale)} pipeline subprocess(es) already running:")
            for pid, cmd in stale[:5]:
                print(f"    PID {pid}: {cmd}")
            print("  If these are stale (from a crashed run), kill them first:")
            print("    pkill -f trainer.py; pkill -f evaluate.py; pkill -f runner.py")
            print("  If these are intentional parallel jobs, ignore this warning.\n")
    except Exception:
        pass


def main():
    args = parse_args = _parse_args()

    print("=" * 68)
    print(f"  SpecDist Pipeline")
    print(f"  config  : {args.config}")
    print(f"  losses  : {args.losses or 'all'}")
    print("=" * 68)

    _warn_stale_trainers()   # warn (not kill) if stale subprocesses are found
    _install_deps()
    _auth()

    n_gpus, total_vram = _gpu_info()
    storage_root = _resolve_storage_root(args)
    os.makedirs(storage_root, exist_ok=True)
    print(f"storage : {storage_root}")

    # HF cache inside storage root so models persist across restarts
    hf_cache = os.path.join(storage_root, "hf_cache")
    os.makedirs(hf_cache, exist_ok=True)
    os.environ["HF_HOME"]            = hf_cache
    os.environ["TRANSFORMERS_CACHE"] = hf_cache
    for flag in ("TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "HF_HUB_OFFLINE"):
        os.environ.pop(flag, None)

    # Build experiment.py command
    experiment = os.path.join(_GBV_DIR, "orchestration", "experiment.py")
    cmd = [
        sys.executable, experiment,
        "--config",       args.config,
        "--storage_root", storage_root,
        "--yes",
    ]
    if args.losses:
        cmd += ["--losses", args.losses]
    if args.experiment_tag:
        cmd += ["--experiment_tag", args.experiment_tag]

    no_smoke = args.no_smoke or args.resume
    if not no_smoke:
        cmd.append("--smoke")
        print("\nRunning smoke test first...")
        print("(pass --no_smoke to skip)\n")

    if args.dry_run:
        print("\nDry run — command that would execute:")
        print(" ".join(cmd))
        return

    print(f"\nLaunching: {' '.join(cmd[-6:])}")
    result = subprocess.run(cmd, cwd=_GBV_DIR)
    if result.returncode != 0:
        print(f"\nFailed (exit {result.returncode})")
        sys.exit(result.returncode)

    if not no_smoke:
        # Smoke passed — now run full pipeline
        full_cmd = [c for c in cmd if c != "--smoke"]
        print("\nSmoke passed. Starting full pipeline...")
        result = subprocess.run(full_cmd, cwd=_GBV_DIR)
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
