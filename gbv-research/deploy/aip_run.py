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

# ── Auto-relaunch under venv if running with system Python ────────────────────
# Problem: running `python deploy/aip_run.py` without activating the venv uses
# the system Python which lacks torch/transformers.  On Python 3.12, pip also
# throws "_distutils_hack has no attribute 'add_shim'" and the install fails.
#
# Fix: if a specdist venv exists and we're NOT already inside it, re-exec this
# script under the venv Python automatically.  The user still gets the same
# command they typed; no output is lost.
_VENV_PYTHON = os.path.join(
    os.path.expanduser("~"), "specdist", "venv", "bin", "python"
)
_IN_VENV = (
    sys.prefix != sys.base_prefix                         # inside ANY venv
    or os.environ.get("VIRTUAL_ENV")                      # venv is activated
    or "specdist" in sys.executable                       # running from specdist venv path
)
if not _IN_VENV and os.path.isfile(_VENV_PYTHON):
    print(f"[aip_run] Not in venv — re-launching under {_VENV_PYTHON}")
    os.execv(_VENV_PYTHON, [_VENV_PYTHON] + sys.argv)
    # os.execv replaces this process; code below only runs if exec fails

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
  python deploy/aip_run.py --config a100_qwen --losses traversal_tree --train_only --no_smoke
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
    p.add_argument("--train_only",     action="store_true",
                   help="Train + merge only (skip baseline and post-train eval). "
                        "Use with --losses traversal_tree to train one loss.")
    p.add_argument("--from_step",      default=None,
                   help="Start at this pipeline step id, e.g. train_trav_tree_gsm8k")
    p.add_argument("--dry_run",        action="store_true",
                   help="Print the experiment.py command without running it")
    p.add_argument("--experiment_tag", default=None,
                   help="Human-readable tag for this session's eval results "
                        "(default: auto-generated from hostname+timestamp). "
                        "E.g. --experiment_tag rmukundServer")
    p.add_argument("--force_eval", action="store_true",
                   help="Force re-evaluation even when results exist in DB. "
                        "Use when re-training a loss to add NEW comparison rows "
                        "without deleting old ones. Old rows are preserved; new "
                        "rows get a different run_tag. Always set --experiment_tag "
                        "alongside this so you can distinguish runs in W&B/dashboard.")
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
    if args.force_eval:
        cmd.append("--force_eval")
    if args.train_only:
        cmd.append("--train_only")
    if args.from_step:
        cmd += ["--from", args.from_step]

    no_smoke = args.no_smoke or args.resume
    if not no_smoke:
        cmd.append("--smoke")
        print("\nRunning smoke test first...")
        print("(pass --no_smoke to skip)\n")

    if args.dry_run:
        print("\nDry run — command that would execute:")
        print(" ".join(cmd))
        return

    # ── Pre-launch: clear stale Python bytecode cache ─────────────────────
    # Python caches compiled .pyc files in __pycache__/ dirs.  On network /
    # container filesystems, mtime-based invalidation can fail after a git pull,
    # causing old bytecode (e.g. un-fixed specInfer) to run despite updated .py.
    # Clearing takes <1s and guarantees fresh imports on every pipeline launch.
    import glob as _glob, shutil as _shutil
    _pyc_count = 0
    for _pyc in _glob.glob(os.path.join(_GBV_DIR, "**", "*.pyc"), recursive=True):
        try: os.remove(_pyc); _pyc_count += 1
        except OSError: pass
    for _pycache in _glob.glob(os.path.join(_GBV_DIR, "**", "__pycache__"), recursive=True):
        try: _shutil.rmtree(_pycache); _pyc_count += 1
        except OSError: pass
    if _pyc_count:
        print(f"  [startup] Cleared {_pyc_count} stale .pyc / __pycache__ entries.")

    # ── Pre-launch: disable specInfer if transformers ≥5.x ────────────────
    # DynamicCache internal state (_seen_tokens etc.) changed in transformers 5.x
    # making our specInfer KV-cache surgery invalid.  The inline alpha fallback
    # gives IDENTICAL values at ~20% lower speed.  Disable automatically so the
    # eval never wastes time attempting a doomed specInfer call.
    import importlib.util as _ilu
    try:
        import transformers as _tf
        _tf_ver = tuple(int(x) for x in _tf.__version__.split(".")[:2])
        if _tf_ver >= (5, 0):
            os.environ.setdefault("SPECDIST_DISABLE_SPECINFER", "1")
            print(f"  [startup] transformers {_tf.__version__} ≥ 5.x detected — "
                  f"specInfer disabled (SPECDIST_DISABLE_SPECINFER=1); using inline alpha fallback.")
    except Exception:
        pass

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
