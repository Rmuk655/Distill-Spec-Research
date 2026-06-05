"""
aip_run.py — SpecDist pipeline launcher for GPU compute environments.

Simple per-loss workflow (recommended):
    python deploy/aip_run.py --loss kl --train       # train kl only
    python deploy/aip_run.py --loss kl --eval        # eval kl only (skips baseline)
    python deploy/aip_run.py --loss kl               # full pipeline: train + eval
    python deploy/aip_run.py --loss bv_tree --train  # auto-applies --lr 1e-5

Full pipeline / legacy usage:
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


# Tree losses that use a full-vocab BV integral — gradients are amplified
# so the recommended learning rate is 1e-5 (vs the default 3e-5).
_HIGH_GRAD_LOSSES = {"bv_tree", "gbv_tree"}

# Losses whose eval GSM8K step ID differs from the default eval_{loss}_gsm8k pattern.
_LOSS_TO_EVAL_STEP = {
    "traversal_tree": "eval_trav_tree_gsm8k",
    "specinfer_tree": "eval_si_tree_gsm8k",
    "spectr_tree":    "eval_st_tree_gsm8k",
}


def _eval_step_id(loss: str) -> str:
    """Return the GSM8K eval step ID for a given loss name."""
    return _LOSS_TO_EVAL_STEP.get(loss, f"eval_{loss}_gsm8k")


def _parse_args():
    p = argparse.ArgumentParser(
        description="SpecDist pipeline launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Simple per-loss workflow:
  python deploy/aip_run.py --loss kl --train          # train kl only
  python deploy/aip_run.py --loss kl --eval           # eval kl (skips baseline)
  python deploy/aip_run.py --loss kl                  # full: train + eval
  python deploy/aip_run.py --loss bv_tree --train     # auto-applies --lr 1e-5

Legacy / full-pipeline:
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
    # ── Simplified single-loss interface ─────────────────────────────────────
    p.add_argument("--loss",        default=None,
                   help="Single loss to run (alias for --losses with one value). "
                        "E.g. --loss kl  or  --loss bv_tree")
    p.add_argument("--train",       action="store_true",
                   help="Train only (alias for --train_only). "
                        "Use with --loss: python deploy/aip_run.py --loss kl --train")
    p.add_argument("--eval",        action="store_true",
                   help="Eval only (alias for --eval_only). "
                        "Automatically skips baseline when a single --loss is given. "
                        "Use with --loss: python deploy/aip_run.py --loss kl --eval")
    # ── Legacy / full-pipeline flags ─────────────────────────────────────────
    p.add_argument("--losses",      default=None,
                   help="Comma-separated loss names to run "
                        "(default: all). E.g. 'kl' or 'kl,kl_tree'")
    p.add_argument("--storage_root", default=None,
                   help="Root for checkpoints/, results.db, logs. "
                        "On Pluto defaults to gbv-research/db/ (local RAM). "
                        "Sensei FS paths are ignored unless SPECDIST_USE_SENSEI=1.")
    p.add_argument("--no_smoke",       action="store_true",
                   help="Skip preflight smoke test")
    p.add_argument("--resume",         action="store_true",
                   help="Resume without smoke test (implies --no_smoke)")
    p.add_argument("--train_only",     action="store_true",
                   help="Train + merge only (skip baseline and post-train eval). "
                        "Use with --losses traversal_tree to train one loss.")
    p.add_argument("--eval_only",      action="store_true",
                   help="Eval only — skip all training and merge steps. "
                        "Requires pre-built merged models. "
                        "Use --eval instead for the simpler single-loss interface.")
    p.add_argument("--from_step", "--from", dest="from_step", default=None,
                   help="Start at this pipeline step id, e.g. merge_rev_kl_gsm8k "
                        "(alias: --from; passed to experiment.py as --from)")
    p.add_argument("--yes", action="store_true",
                   help="No-op for compatibility — aip_run always passes --yes to experiment.py")
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
    p.add_argument("--lr",           type=float, default=None,
                   help="Learning rate override (passed to experiment.py --lr). "
                        "bv_tree / gbv_tree auto-set to 1e-5 if omitted.")
    p.add_argument("--train_steps",  type=int,   default=None,
                   help="Override training steps from config (passed to experiment.py --train_steps). "
                        "Use for Phase-2 continuation, e.g. --train_steps 5000.")
    p.add_argument("--warmup_steps", type=int,   default=None,
                   help="Override LR warmup gradient-steps from config "
                        "(passed to experiment.py --warmup_steps).")
    p.add_argument("--lora_r",       type=int,   default=None,
                   help="LoRA rank override (passed to experiment.py --lora_r).")
    return p, p.parse_args()


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


def _is_sensei_path(path: str) -> bool:
    return bool(path) and path.replace("\\", "/").startswith("/sensei-fs")


def _pluto_ram_storage() -> str | None:
    """Default storage on Pluto: local RAM disk under /home/colligo/ram.

    Uses gbv-research/db/ so checkpoints land in db/checkpoints/ (same path as
    training when STORAGE_ROOT was unset).  Sensei FS is NOT used — quota is tight
    and SQLite checkpoints do not belong on Lustre.
    """
    _ram = "/home/colligo/ram"
    if os.path.isdir(_ram):
        return os.path.join(_GBV_DIR, "db")
    return None


def _resolve_storage_root(args):
    """Determine storage root: CLI > STORAGE_ROOT env > Pluto RAM > repo/db/.

    On Pluto (/home/colligo/ram present), Sensei FS paths are ignored even when
    STORAGE_ROOT=/sensei-fs/... is set in the shell — checkpoints, logs, and DB
    stay on local disk.  Set SPECDIST_USE_SENSEI=1 to force Sensei (not recommended).
    """
    _ram_default = _pluto_ram_storage() or os.path.join(_GBV_DIR, "db")
    _force_sensei = os.environ.get("SPECDIST_USE_SENSEI", "").strip() in ("1", "true", "yes")

    def _maybe_redirect(path: str, source: str) -> str:
        if _force_sensei or not _is_sensei_path(path):
            return path
        print(f"[storage] {source} points at Sensei FS ({path}) — "
              f"using local RAM storage instead: {_ram_default}")
        print(f"[storage] (set SPECDIST_USE_SENSEI=1 to override)")
        return _ram_default

    if args.storage_root:
        return _maybe_redirect(args.storage_root, "--storage_root")
    env = os.environ.get("STORAGE_ROOT", "")
    if env:
        return _maybe_redirect(env, "STORAGE_ROOT")
    if _pluto_ram_storage():
        return _ram_default
    _home_specdist = os.path.join(os.path.expanduser("~"), "specdist")
    if os.access(os.path.expanduser("~"), os.W_OK):
        return _home_specdist
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
    p, args = _parse_args()

    # ── Resolve convenience aliases ───────────────────────────────────────────
    # --loss X   → --losses X  (singular convenience)
    if args.loss and args.losses:
        p.error("Pass --loss or --losses, not both.")
    if args.loss:
        args.losses = args.loss

    # --train / --eval mutual exclusivity (mirrors experiment.py --train_only / --eval_only)
    if args.train and args.eval:
        p.error("Pass at most one of --train / --eval.")
    if args.train and args.train_only:
        p.error("Pass at most one of --train / --train_only (they are the same thing).")
    if args.eval and args.eval_only:
        p.error("Pass at most one of --eval / --eval_only (they are the same thing).")

    # Normalize to the legacy flags that experiment.py understands
    if args.train:
        args.train_only = True
    if args.eval:
        args.eval_only = True

    # Auto-apply --lr 1e-5 for BV-integral losses that amplify gradients
    if args.losses and args.lr is None:
        _single = args.losses.strip().split(",")[0].strip()  # check first loss
        if _single in _HIGH_GRAD_LOSSES and "," not in args.losses:
            args.lr = 1e-5
            print(f"[aip_run] Auto-setting --lr 1e-5 for {_single} "
                  f"(BV integral amplifies gradients; override with --lr)")

    # Auto-skip baseline when eval_only + single loss:
    # --eval --loss kl  →  --from eval_kl_gsm8k  (skip eval_baseline_gsm8k)
    if args.eval_only and args.losses and "," not in args.losses and not args.from_step:
        _single_loss = args.losses.strip()
        args.from_step = _eval_step_id(_single_loss)
        print(f"[aip_run] --eval with single loss '{_single_loss}': "
              f"skipping baseline, --from {args.from_step}")

    print("=" * 68)
    print(f"  SpecDist Pipeline")
    print(f"  config  : {args.config}")
    print(f"  losses  : {args.losses or 'all'}")
    if args.train_only:
        print(f"  mode    : train only")
    elif args.eval_only:
        print(f"  mode    : eval only (from: {args.from_step or 'start'})")
    print("=" * 68)

    _warn_stale_trainers()   # warn (not kill) if stale subprocesses are found
    _install_deps()
    _auth()

    n_gpus, total_vram = _gpu_info()
    storage_root = _resolve_storage_root(args)
    os.makedirs(storage_root, exist_ok=True)
    print(f"storage : {storage_root}")
    # results.db lives under storage_root (local on Pluto); override only if set.
    if not os.environ.get("SPECDIST_DB_PATH", "").strip():
        _db = os.path.join(storage_root, "results.db")
        os.makedirs(os.path.dirname(_db) or ".", exist_ok=True)
        os.environ["SPECDIST_DB_PATH"] = _db
        print(f"results.db: {_db}")

    # HF model cache: honour a pre-set HF_HOME (e.g. local RAM disk on Pluto)
    # so we do not re-download ~18 GB into Sensei FS when models already live
    # under ~/ram/specdist/hf_cache.  Checkpoints/DB/logs still use storage_root.
    _hf_from_env = os.environ.get("HF_HOME", "").strip()
    if _hf_from_env:
        hf_cache = _hf_from_env
        os.makedirs(hf_cache, exist_ok=True)
        print(f"HF cache : {hf_cache}  (HF_HOME env — not using storage_root/hf_cache)")
    else:
        hf_cache = os.path.join(storage_root, "hf_cache")
        os.makedirs(hf_cache, exist_ok=True)
        print(f"HF cache : {hf_cache}  (under storage_root)")
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
    if args.eval_only:
        cmd.append("--eval_only")
    if args.from_step:
        cmd += ["--from", args.from_step]
    if args.lr is not None:
        cmd += ["--lr", str(args.lr)]
    if args.train_steps is not None:
        cmd += ["--train_steps", str(args.train_steps)]
    if args.warmup_steps is not None:
        cmd += ["--warmup_steps", str(args.warmup_steps)]
    if args.lora_r is not None:
        cmd += ["--lora_r", str(args.lora_r)]

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
