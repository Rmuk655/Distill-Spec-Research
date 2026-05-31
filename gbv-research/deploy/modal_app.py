"""
modal_app.py — on-demand A100 run path for the SpecDist / Distill-Spec-Research
pipeline on Modal (https://modal.com).

This is a *pure launcher*: it builds a GPU image, mounts a persistent Volume,
clones the repo at runtime, and invokes the SAME CLI entrypoint that the Kaggle
bootstrap uses:

    python orchestration/experiment.py --config a100 --storage_root /vol/specdist --yes

It does NOT touch any core pipeline code (experiment.py / trainer.py /
evaluate.py / runner.py / hw_scheduler.py).  Same philosophy as the Kaggle
bootstrap in deploy/kaggle.ipynb + deploy/deploy_utils.py.

--------------------------------------------------------------------------------
Quickstart (full setup in docs/MODAL.md)
--------------------------------------------------------------------------------
    pip install modal
    modal setup

    # One secret holding whatever tokens you have (all keys optional):
    #   HF_TOKEN        — HuggingFace (Qwen3 is public, so optional)
    #   WANDB_API_KEY   — Weights & Biases (omit -> WANDB_MODE=offline)
    #   GITHUB_TOKEN    — only needed if the GitHub repo is private
    modal secret create specdist-secrets \
        HF_TOKEN=hf_xxx WANDB_API_KEY=xxx GITHUB_TOKEN=ghp_xxx

    # 1. Smoke test FIRST (cheap — a few minutes; catches setup errors):
    modal run deploy/modal_app.py --smoke

    # 2. Phase-1 tiered iteration — train + val_loss + light BE sanity:
    #    forward_kl flat baseline (loss name "forward_kl"; CLI alias "--losses kl"):
    modal run deploy/modal_app.py --config profiles/train_one_loss --losses kl
    #    tree-loss variant (Phase-2):
    modal run deploy/modal_app.py --config profiles/tree_variant_week --losses kl_tree

    # 3. A100 confirmation ONLY — full GSM8K n=1319, all verifiers, ≥3 seeds.
    #    Reserve for ≤2 confirmed publication candidates; these are the paper numbers:
    modal run deploy/modal_app.py --config a100 --losses kl
--------------------------------------------------------------------------------
"""

import os
import shlex
import subprocess
import sys

import modal

# ──────────────────────────────────────────────────────────────────────────────
# Module-level knobs — edit these to change GPU / budget / repo / persistence.
# ──────────────────────────────────────────────────────────────────────────────

APP_NAME = "specdist"

# GPU type.  Easy to change:  "A100" (40 GB) | "A100-80GB" | "L4" | "T4" | ...
# A100 ≈ $5.30 / hour on Modal (per-second billing).  A T4 (~$0.59/h) is enough
# for a --smoke run if you want to conserve credit; the 8B teacher needs the A100.
GPU_TYPE = "A100"

# Wall-clock ceiling for one invocation (training + eval is long).  8 hours.
TIMEOUT_SECONDS = 8 * 60 * 60

# Repo to clone at runtime (matches the Kaggle bootstrap).  Always pulls the
# latest pushed code so the run never goes stale against your local tree.
REPO_URL = "https://github.com/Rmuk655/Distill-Spec-Research.git"
DEFAULT_GIT_REF = "main"

# Where the repo is cloned inside the (ephemeral) container.
REPO_DIR = "/root/Distill-Spec-Research"
GBV_DIR = f"{REPO_DIR}/gbv-research"

# Persistent Volume — survives across runs so re-runs resume (the pipeline
# skips completed steps via its state machine).  Holds: results.db, checkpoints,
# logs, pipeline state, and the HF model cache.
VOLUME_NAME = "specdist-vol"
VOLUME_MOUNT = "/vol"
STORAGE_ROOT = f"{VOLUME_MOUNT}/specdist"          # --storage_root for experiment.py
HF_CACHE_DIR = f"{STORAGE_ROOT}/hf_cache"          # HF_HOME inside the Volume

# Single consolidated secret.  Individual keys are optional — missing keys are
# handled gracefully (public model download, offline W&B, anonymous git clone).
# Using ONE secret (instead of three) means a missing optional token never fails
# the run with "Secret not found"; you just omit the key you don't have.
SECRET_NAME = "specdist-secrets"

# Default pipeline config.  a100.yaml already uses HF repo ids
# (Qwen/Qwen3-0.6B draft, Qwen/Qwen3-8B teacher), so no Modal-specific YAML is
# needed — the same config the Colab/Lightning A100 runs use.
DEFAULT_CONFIG = "a100"

# ──────────────────────────────────────────────────────────────────────────────
# Image — install the SAME deps the pipeline needs (derived from
# gbv-research/requirements.txt + the extras deploy_utils.install_deps adds:
# bitsandbytes>=0.46.1, accelerate, torchao>=0.16.0).  torch is installed
# separately (as setup_env.py does); on Modal GPU containers the default PyPI
# torch wheel ships bundled CUDA.  Pins mirror the repo exactly — nothing extra
# is pinned.
# ──────────────────────────────────────────────────────────────────────────────

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    # torch first (matches setup_env.py ordering; CUDA wheel from default index).
    .pip_install("torch")
    # bitsandbytes force-upgraded first with the repo's pin (transformers needs
    # >=0.46.1 for 4-bit NF4) — same reason deploy_utils installs it up front.
    .pip_install("bitsandbytes>=0.46.1")
    # Remaining deps = requirements.txt verbatim + deploy_utils extras.
    .pip_install(
        # Core ML
        "transformers>=4.40.0",
        "peft>=0.10.0",
        "accelerate>=0.29.0",
        "datasets>=2.18.0",
        "tokenizers>=0.19.0",
        "sentencepiece>=0.1.99",
        "protobuf>=3.20.0",
        # Training utilities
        "scipy>=1.12.0",
        "torchao>=0.16.0",          # deploy_utils extra (PEFT LoRA init compat)
        # Numerics
        "numpy>=1.26.0",
        "pandas>=2.2.0",
        # Web dashboard
        "flask>=3.0.0",
        "flask-cors>=4.0.0",
        # Utilities
        "tqdm>=4.66.0",
        "requests>=2.31.0",
        "huggingface-hub>=0.22.0",
        "pyyaml>=6.0",
        "wandb>=0.16.0",
        # Original OSD deps (kept for compatibility — match requirements.txt)
        "fschat==0.2.23",
        "packaging",
    )
    # Default to offline W&B inside the image; flipped to online at runtime when
    # WANDB_API_KEY is present in the secret.
    .env({"WANDB_MODE": "offline"})
)

app = modal.App(APP_NAME)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# Attach the consolidated secret.  If it does not exist yet, `modal run` will
# tell you to create it (see docs/MODAL.md).
secrets = [modal.Secret.from_name(SECRET_NAME)]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers (run inside the container)
# ──────────────────────────────────────────────────────────────────────────────

def _run(cmd, **kwargs):
    """Run a shell command, streaming stdout/stderr to the Modal logs."""
    if isinstance(cmd, str):
        printable = cmd
    else:
        printable = " ".join(shlex.quote(c) for c in cmd)
    print(f"  $ {printable}", flush=True)
    return subprocess.run(cmd, **kwargs)


def _clone_repo(git_ref: str) -> None:
    """Clone (or refresh) the repo at git_ref, mirroring the Kaggle bootstrap.

    Injects GITHUB_TOKEN into the clone URL for private repos (optional).
    """
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    clone_url = REPO_URL.replace("https://", f"https://{token}@") if token else REPO_URL

    if not os.path.isdir(REPO_DIR):
        _run(["git", "clone", REPO_URL if not token else clone_url, REPO_DIR], check=True)
    else:
        if token:
            _run(["git", "-C", REPO_DIR, "remote", "set-url", "origin", clone_url], check=False)
        _run(["git", "-C", REPO_DIR, "fetch", "origin", git_ref], check=True)

    # Check out the requested ref (branch, tag, or commit) and hard-reset to it.
    _run(["git", "-C", REPO_DIR, "checkout", git_ref], check=True)
    _run(["git", "-C", REPO_DIR, "pull", "--ff-only", "origin", git_ref], check=False)
    head = subprocess.run(
        ["git", "-C", REPO_DIR, "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True,
    )
    print(f"  Repo @ {git_ref} -> {head.stdout.strip()}", flush=True)


def _setup_env() -> None:
    """Point all SpecDist storage + HF cache env vars inside the Volume.

    experiment.py --storage_root also sets the SPECDIST_* vars itself, but we set
    them here too (matches deploy_utils.run_pipeline) and additionally set the HF
    cache so downloaded weights persist in the Volume across runs.
    """
    os.makedirs(os.path.join(STORAGE_ROOT, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(STORAGE_ROOT, "logs"), exist_ok=True)
    os.makedirs(HF_CACHE_DIR, exist_ok=True)

    os.environ["SPECDIST_STORAGE_ROOT"] = STORAGE_ROOT
    os.environ["SPECDIST_DB_PATH"] = os.path.join(STORAGE_ROOT, "results.db")
    os.environ["SPECDIST_LOGS_ROOT"] = os.path.join(STORAGE_ROOT, "logs")

    os.environ["HF_HOME"] = HF_CACHE_DIR
    os.environ["TRANSFORMERS_CACHE"] = HF_CACHE_DIR
    os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_CACHE_DIR, "datasets")
    # Never inherit a stale offline flag — the child must reach HF on first run.
    for flag in ("TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "HF_HUB_OFFLINE"):
        os.environ.pop(flag, None)

    # Mirror HF token aliases for the hub client.
    hf = os.environ.get("HF_TOKEN", "").strip()
    if hf:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf

    # W&B: online only if a key is present; otherwise stay offline (image default).
    if os.environ.get("WANDB_API_KEY", "").strip():
        os.environ.pop("WANDB_MODE", None)
        print("  W&B: online (WANDB_API_KEY present)", flush=True)
    else:
        os.environ["WANDB_MODE"] = "offline"
        print("  W&B: offline (no WANDB_API_KEY in secret)", flush=True)


def _prefetch_models(config: str) -> None:
    """Download draft + teacher weights into the Volume-backed HF cache.

    Unlike deploy_utils.prefetch_models (which skips >5 GB models to protect a
    tiny Google Drive), the Modal Volume has ample room, so we fetch BOTH models
    here.  snapshot_download is idempotent — subsequent runs skip re-download.
    """
    import yaml
    from huggingface_hub import snapshot_download

    yaml_path = os.path.join(
        GBV_DIR, "orchestration", "configs", config.replace("/", os.sep) + ".yaml"
    )
    if not os.path.exists(yaml_path):
        print(f"  [prefetch] config not found ({yaml_path}); trainer will download at run time.", flush=True)
        return

    with open(yaml_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    models = cfg.get("models", {})
    draft = models.get("draft", "Qwen/Qwen3-0.6B")
    target = models.get("target", "Qwen/Qwen3-8B")

    for model_id in dict.fromkeys([draft, target]):
        if not isinstance(model_id, str) or "/" not in model_id:
            # Not an HF repo id (e.g. a local path override) — skip prefetch.
            continue
        print(f"  [prefetch] {model_id} -> {HF_CACHE_DIR} …", flush=True)
        try:
            snapshot_download(model_id, ignore_patterns=["*.gguf", "*.bin"])
            print(f"  [prefetch] ✓ {model_id}", flush=True)
        except Exception as exc:  # noqa: BLE001 — best-effort; trainer retries
            print(f"  [prefetch] ✗ {model_id}: {exc} (trainer will retry at run time)", flush=True)


def _fetch_training_data() -> None:
    """Download gsm8k_train.jsonl if missing (mirrors deploy_utils.fetch_training_data)."""
    raw_dir = os.path.join(GBV_DIR, "core", "datasets", "raw")
    train_path = os.path.join(raw_dir, "gsm8k_train.jsonl")
    if os.path.exists(train_path):
        print("  [data] gsm8k_train.jsonl already present — skipping download.", flush=True)
        return
    os.makedirs(raw_dir, exist_ok=True)
    downloader = os.path.join(GBV_DIR, "core", "datasets", "downloader.py")
    print("  [data] downloading gsm8k_train.jsonl …", flush=True)
    _run([sys.executable, downloader, "--datasets", "gsm8k_train"], cwd=GBV_DIR, check=False)


# ──────────────────────────────────────────────────────────────────────────────
# Main GPU function
# ──────────────────────────────────────────────────────────────────────────────

@app.function(
    image=image,
    gpu=GPU_TYPE,
    timeout=TIMEOUT_SECONDS,
    volumes={VOLUME_MOUNT: volume},
    secrets=secrets,
)
def run_pipeline(
    config: str = DEFAULT_CONFIG,
    smoke: bool = False,
    losses: str = None,
    git_ref: str = DEFAULT_GIT_REF,
    extra_args: list = None,
) -> int:
    """Clone the repo and run the SAME experiment.py CLI the Kaggle path uses.

    Parameters
    ----------
    config      YAML profile name under orchestration/configs/ (default "a100").
    smoke       True = quick crash-check (--smoke): a few prompts, tiny tokens.
    losses      Comma-separated subset, e.g. "kl" (forward-KL flat baseline).
                None = run all losses defined in the config.
    git_ref     Branch / tag / commit to check out (default "main").
    extra_args  Extra args appended to the experiment.py command line.
    """
    _clone_repo(git_ref)
    _setup_env()

    if not smoke:
        # Smoke runs don't need the big teacher prefetched — keep them cheap/fast.
        _prefetch_models(config)
    _fetch_training_data()

    cmd = [
        sys.executable, "orchestration/experiment.py",
        "--config", config,
        "--storage_root", STORAGE_ROOT,
        "--yes",
    ]
    if smoke:
        cmd.append("--smoke")
    if losses:
        cmd += ["--losses", losses]
    if extra_args:
        cmd += list(extra_args)

    mode = "SMOKE TEST" if smoke else f"FULL pipeline ({config})"
    loss_note = f"  losses={losses}" if losses else "  losses=ALL"
    print(f"\n=== {mode}{loss_note} ===", flush=True)
    print(f"  GPU={GPU_TYPE}  storage_root={STORAGE_ROOT}", flush=True)

    # Stream stdout/stderr straight to the Modal logs (no capture).
    result = _run(cmd, cwd=GBV_DIR, check=False)

    # Persist everything written to the Volume (db, checkpoints, logs, hf cache).
    volume.commit()

    if result.returncode == 0:
        print("\n✓ Pipeline finished — artifacts committed to the Volume.", flush=True)
    else:
        print(
            f"\n✗ Exit {result.returncode} — re-run the same command to resume "
            "(completed steps are skipped via the state machine).",
            flush=True,
        )
    return result.returncode


# ──────────────────────────────────────────────────────────────────────────────
# Local entrypoint — what `modal run deploy/modal_app.py [...]` invokes.
# ──────────────────────────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(
    config: str = DEFAULT_CONFIG,
    smoke: bool = False,
    losses: str = None,
    git_ref: str = DEFAULT_GIT_REF,
):
    """Launch the pipeline on a Modal GPU worker.

    Examples
    --------
        modal run deploy/modal_app.py --smoke
        modal run deploy/modal_app.py --losses kl
        modal run deploy/modal_app.py
        modal run deploy/modal_app.py --config a100 --git-ref main
    """
    rc = run_pipeline.remote(
        config=config,
        smoke=smoke,
        losses=losses,
        git_ref=git_ref,
    )
    if rc != 0:
        raise SystemExit(rc)
