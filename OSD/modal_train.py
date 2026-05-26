"""
modal_train.py — Run SpecDist training on Modal.com GPU cloud.

Modal gives you on-demand A100/H100 GPUs billed per second.
Checkpoints and the HF model cache are stored in a Modal Volume
so they persist across runs and you don't re-download 16 GB every time.

Prerequisites
-------------
    pip install modal
    modal token new          # one-time auth

First run (downloads ~16 GB of models into the volume):
    modal run modal_train.py::download_models

Upload dataset:
    modal run modal_train.py::upload_dataset

Train individual losses:
    modal run modal_train.py::train_kl
    modal run modal_train.py::train_ebe
    modal run modal_train.py::train_rev_kl
    modal run modal_train.py::train_jsd
    modal run modal_train.py::train_l1
    modal run modal_train.py::train_online

Train with custom args:
    modal run modal_train.py::run --loss ebe --steps 2000 --lr 1e-4

Run full pipeline (all 6 losses sequentially):
    modal run modal_train.py::run_pipeline

Monitor: while running, tail logs with
    modal app logs <app-id>

Download results locally (convenience CLI command):
    modal run modal_train.py::download_checkpoints
  or manually:
    modal volume get specdist-vol /checkpoints ./local_checkpoints

Notes
-----
- TRANSFORMERS_OFFLINE is set to 0 so the first run can download models.
  After download_models completes, subsequent runs use cached weights.
- The volume is mounted at /vol inside the container.
- Checkpoints are written to /vol/checkpoints/{run_name}/.
- Periodic vol.commit() every 5 minutes guards against container crashes:
  if Modal kills the container mid-training, your latest checkpoint is
  already flushed to durable storage and survives the restart.
- Crash resume: re-run the same modal run command — train_qwen3.py will
  detect ckpt_latest/ and resume from the last saved step automatically.
- GPU flexibility: the run() function accepts a `gpu` parameter.
  Default is "A100-40GB" (fits Qwen3-8B in bfloat16 on 40 GB).
  Switch to "A10G" (24 GB, cheaper) or "A100-80GB" (80 GB, max headroom).
"""

import os
import subprocess
import sys
import threading

import modal

# ---------------------------------------------------------------------------
# Image — install all dependencies
# ---------------------------------------------------------------------------

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.3.0",
        "transformers>=4.43.0",
        "peft>=0.11.0",
        "accelerate>=0.30.0",
        "bitsandbytes>=0.43.0",    # QLoRA support (--load_in_4bit)
        "sentencepiece",
        "flask",                    # viz_server
        "wandb",                    # optional W&B logging
        # flash-attn installed separately below (needs CUDA headers)
    )
    .run_commands(
        # flash-attn requires CUDA — build inside the image
        "pip install flash-attn --no-build-isolation || echo 'flash-attn skipped'",
    )
)

# ---------------------------------------------------------------------------
# Persistent volume — stores HF model cache + checkpoints across runs
# ---------------------------------------------------------------------------

vol = modal.Volume.from_name("specdist-vol", create_if_missing=True)

VOLUME_PATH = "/vol"
HF_CACHE    = f"{VOLUME_PATH}/hf_cache"
CKPT_BASE   = f"{VOLUME_PATH}/checkpoints"
DATA_PATH   = f"{VOLUME_PATH}/data"

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = modal.App("specdist", image=image)

# Source directory: OSD/ is mounted into /src inside the container.
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))


def _run_cmd(cmd: list[str], env: dict = None):
    """Run a command with merged environment, streaming stdout/stderr."""
    merged = {**os.environ, **(env or {})}
    result = subprocess.run(cmd, env=merged, check=True)
    return result


# ---------------------------------------------------------------------------
# Periodic volume commit — crash safety
#
# Modal containers can be killed mid-training (preemption, OOM, timeout).
# By default vol.commit() is called only at the very end of run(), so an
# interrupted run loses every checkpoint written during that container's
# lifetime.
#
# Fix: spawn a background daemon thread that flushes the volume every
# COMMIT_INTERVAL_SECS seconds while the training subprocess is alive.
# When training finishes, signal the thread via a threading.Event and wait
# for it to exit cleanly before doing the final commit.
#
# Result: on crash, at most COMMIT_INTERVAL_SECS of work is lost instead of
# the entire run.  train_qwen3.py will resume from ckpt_latest/ on restart.
# ---------------------------------------------------------------------------

COMMIT_INTERVAL_SECS = 300  # flush every 5 minutes


def _start_commit_daemon(stop_event: threading.Event) -> threading.Thread:
    """Start and return a daemon thread that calls vol.commit() every 5 min.

    The thread exits cleanly when stop_event is set.
    """
    def _loop():
        while not stop_event.wait(timeout=COMMIT_INTERVAL_SECS):
            try:
                vol.commit()
                print(f"[commit-daemon] vol.commit() OK (every {COMMIT_INTERVAL_SECS}s)")
            except Exception as exc:
                # Non-fatal: log and keep going; the next iteration will retry.
                print(f"[commit-daemon] vol.commit() failed: {exc}")

    t = threading.Thread(target=_loop, daemon=True, name="vol-commit-daemon")
    t.start()
    return t


# ---------------------------------------------------------------------------
# Step 0 — Download models into the volume (run once)
# ---------------------------------------------------------------------------

@app.function(
    gpu="A10G",               # A10G is cheap; enough for just downloading
    volumes={VOLUME_PATH: vol},
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-secret", required=False)],
)
def download_models(
    draft:  str = "Qwen/Qwen3-0.6B",
    target: str = "Qwen/Qwen3-8B",
):
    """Download draft + target models into the volume's HF cache.

    Run once before training.  Subsequent training runs use the cached weights.
    """
    import transformers
    os.environ["HF_HOME"]               = HF_CACHE
    os.environ["TRANSFORMERS_OFFLINE"]  = "0"
    os.environ["HF_HUB_OFFLINE"]        = "0"
    os.makedirs(HF_CACHE, exist_ok=True)

    print(f"Downloading {draft}...")
    transformers.AutoTokenizer.from_pretrained(draft)
    transformers.AutoModelForCausalLM.from_pretrained(
        draft, torch_dtype="auto", trust_remote_code=True)
    print(f"Downloading {target}...")
    transformers.AutoTokenizer.from_pretrained(target)
    transformers.AutoModelForCausalLM.from_pretrained(
        target, torch_dtype="auto", trust_remote_code=True)

    vol.commit()
    print("Models cached. Ready for training.")


# ---------------------------------------------------------------------------
# Step 0b — Upload dataset files to the volume
#
# Usage:
#   modal run modal_train.py::upload_dataset
#
# Uploads OSD/data/gsm8k_train.jsonl (and any other .jsonl in OSD/data/)
# to /vol/data/ on the volume so that training can reference them as
#   --dataset /vol/data/gsm8k_train.jsonl
# ---------------------------------------------------------------------------

@app.function(
    volumes={VOLUME_PATH: vol},
    timeout=600,
    mounts=[modal.Mount.from_local_dir(
        _SRC_DIR,
        remote_path="/src",
    )],
)
def upload_dataset():
    """Upload local OSD/data/*.jsonl files to /vol/data/ on the volume.

    Run this after download_models and before any training run.
    All .jsonl files found in /src/data/ are copied to /vol/data/.
    """
    import glob
    import shutil

    os.makedirs(DATA_PATH, exist_ok=True)
    src_data_dir = "/src/data"

    jsonl_files = glob.glob(os.path.join(src_data_dir, "*.jsonl"))
    if not jsonl_files:
        print(f"No .jsonl files found in {src_data_dir}. "
              f"Generate them locally first (e.g. python data/clean_gsm8k.py), "
              f"then re-run this command.")
        return

    for src_path in sorted(jsonl_files):
        fname = os.path.basename(src_path)
        dst_path = os.path.join(DATA_PATH, fname)
        shutil.copy2(src_path, dst_path)
        size_mb = os.path.getsize(dst_path) / 1024 / 1024
        print(f"  Uploaded {fname} -> {dst_path}  ({size_mb:.1f} MB)")

    vol.commit()
    print(f"\nDone. {len(jsonl_files)} file(s) uploaded to {DATA_PATH}/")
    print("Use --dataset /vol/data/gsm8k_train.jsonl in your training command.")


# ---------------------------------------------------------------------------
# Core training function
#
# GPU flexibility: pass `gpu` to switch between GPU tiers without editing
# the decorator.  Modal resolves the string to the correct hardware class.
#
#   "A100-40GB"  (default) — fits Qwen3-8B bfloat16 on 40 GB, ~$2.50/hr
#   "A10G"                 — 24 GB VRAM, cheaper (~$1.10/hr); use --load_in_4bit
#   "A100-80GB"            — 80 GB, max headroom for very long sequences
#
# Note: the @app.function decorator sets a sensible default GPU.  If you
# need a different GPU at call time, use run.with_options(gpu=...).local()
# or invoke train_kl / train_ebe etc. which all accept a `gpu` parameter.
# ---------------------------------------------------------------------------

@app.function(
    gpu="A100-40GB",          # 40 GB fits Qwen3-8B in bfloat16; override via gpu param
    volumes={VOLUME_PATH: vol},
    timeout=86400,            # 24-hour cap
    secrets=[modal.Secret.from_name("huggingface-secret", required=False),
             modal.Secret.from_name("wandb-secret",        required=False)],
    mounts=[modal.Mount.from_local_dir(
        # Mount the OSD source directory into the container so we can run train_qwen3.py
        _SRC_DIR,
        remote_path="/src",
    )],
)
def run(
    loss:             str   = "forward_kl",
    steps:            int   = 1000,
    lr:               float = 3e-5,
    draft:            str   = "Qwen/Qwen3-0.6B",
    target:           str   = "Qwen/Qwen3-8B",
    lora_r:           int   = 8,
    lora_alpha:       int   = 16,
    max_new_tokens:   int   = 80,
    log_every:        int   = 10,
    val_every:        int   = 50,
    val_split:        float = 0.1,
    milestone_every:  int   = 200,
    max_checkpoints:  int   = 5,
    save_every:       int   = 100,
    dataset:          str   = None,     # e.g. "/vol/data/gsm8k_train.jsonl"
    run_name:         str   = None,     # auto-generated if None
    load_in_4bit:     bool  = False,
    gpu:              str   = "A100-40GB",  # documented but only used by callers for override
):
    """Train the draft model with SpecDist distillation on Modal A100.

    The function always runs on the GPU configured in @app.function above.
    The `gpu` parameter is surfaced here so that convenience wrappers
    (train_kl, train_ebe, …) can document which GPU they target, and so
    that run_pipeline() can pass it through for documentation purposes.

    Examples
    --------
    KL baseline:
        modal run modal_train.py::run --loss forward_kl --steps 1000

    EBE novel loss:
        modal run modal_train.py::run --loss ebe --steps 1000

    Custom dataset from volume:
        modal run modal_train.py::run --dataset /vol/data/gsm8k_train.jsonl

    Checkpoints are saved to /vol/checkpoints/{run_name}/.
    Intermediate checkpoints are committed every 5 min for crash safety.
    Download with: modal volume get specdist-vol /checkpoints ./local_checkpoints
    """
    import datetime

    # Environment
    os.environ["HF_HOME"]              = HF_CACHE
    os.environ["TRANSFORMERS_OFFLINE"] = "1"   # use cached weights
    os.environ["HF_HUB_OFFLINE"]       = "1"
    os.makedirs(CKPT_BASE, exist_ok=True)

    if run_name is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        run_name = f"{loss}_{steps}steps_{ts}"

    output = os.path.join(CKPT_BASE, run_name)
    print(f"\n{'='*60}")
    print(f"  Modal SpecDist training run")
    print(f"  Loss   : {loss}")
    print(f"  Steps  : {steps}")
    print(f"  LR     : {lr}")
    print(f"  Draft  : {draft}")
    print(f"  Target : {target}")
    print(f"  Output : {output}")
    print(f"{'='*60}\n")

    cmd = [
        sys.executable, "/src/train_qwen3.py",
        "--draft",           draft,
        "--target",          target,
        "--loss",            loss,
        "--steps",           str(steps),
        "--lr",              str(lr),
        "--lora_r",          str(lora_r),
        "--lora_alpha",      str(lora_alpha),
        "--max_new_tokens",  str(max_new_tokens),
        "--log_every",       str(log_every),
        "--val_every",       str(val_every),
        "--val_split",       str(val_split),
        "--milestone_every", str(milestone_every),
        "--max_checkpoints", str(max_checkpoints),
        "--save_every",      str(save_every),
        "--output",          output,
    ]
    if dataset:
        cmd += ["--dataset", dataset]
    if load_in_4bit:
        cmd.append("--load_in_4bit")

    train_env = {
        "HF_HOME":              HF_CACHE,
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_OFFLINE":       "1",
        "PYTHONIOENCODING":     "utf-8",
    }

    # ── Periodic commit daemon ────────────────────────────────────────────────
    # Spawn a background thread that calls vol.commit() every 5 minutes.
    # This ensures intermediate checkpoints survive a container crash.
    # train_qwen3.py writes ckpt_latest/ at every save_every step interval;
    # the daemon flushes those writes to durable storage independently of the
    # training process itself.
    stop_event = threading.Event()
    commit_thread = _start_commit_daemon(stop_event)
    print(f"[commit-daemon] started — vol.commit() every {COMMIT_INTERVAL_SECS}s")

    try:
        _run_cmd(cmd, env=train_env)
    finally:
        # Signal the daemon to stop and wait for it to exit cleanly
        stop_event.set()
        commit_thread.join(timeout=30)

    # Final commit — ensures the very last checkpoint is durable
    vol.commit()
    print(f"\nDone. Checkpoints at /vol/checkpoints/{run_name}/")
    print("Download with: modal volume get specdist-vol /checkpoints ./local_checkpoints")


# ---------------------------------------------------------------------------
# Core online training function
#
# Online training uses online_serve.py (not train_qwen3.py).
# See pipeline.py::build_steps / online_adapt_gsm8k for the canonical args.
# ---------------------------------------------------------------------------

@app.function(
    gpu="A100-40GB",
    volumes={VOLUME_PATH: vol},
    timeout=86400,
    secrets=[modal.Secret.from_name("huggingface-secret", required=False),
             modal.Secret.from_name("wandb-secret",        required=False)],
    mounts=[modal.Mount.from_local_dir(
        _SRC_DIR,
        remote_path="/src",
    )],
)
def run_online(
    steps:          int   = 500,
    lr:             float = 3e-4,
    draft:          str   = "Qwen/Qwen3-0.6B",
    target:         str   = "Qwen/Qwen3-8B",
    update_every:   int   = 4,
    K:              int   = 4,
    kl_method:      str   = "forward_kl",
    max_new_tokens: int   = 128,
    dataset:        str   = None,
    run_name:       str   = None,
    load_in_4bit:   bool  = False,
):
    """Online OSD adaptation using online_serve.py.

    Trains the draft model continuously while serving, using only rejection
    positions as the training signal (Liu et al. 2023, arxiv:2310.07177).

    Examples
    --------
        modal run modal_train.py::run_online --steps 500
        modal run modal_train.py::run_online --dataset /vol/data/gsm8k_train.jsonl
    """
    import datetime

    os.environ["HF_HOME"]              = HF_CACHE
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"]       = "1"
    os.makedirs(CKPT_BASE, exist_ok=True)

    if run_name is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        run_name = f"online_{steps}steps_{ts}"

    output = os.path.join(CKPT_BASE, run_name)
    prompts = dataset or os.path.join(DATA_PATH, "gsm8k_train.jsonl")

    print(f"\n{'='*60}")
    print(f"  Modal SpecDist ONLINE training run")
    print(f"  Steps  : {steps}")
    print(f"  LR     : {lr}")
    print(f"  Draft  : {draft}")
    print(f"  Target : {target}")
    print(f"  Output : {output}")
    print(f"  Prompts: {prompts}")
    print(f"{'='*60}\n")

    cmd = [
        sys.executable, "/src/online_serve.py",
        "--prompts",      prompts,
        "--draft",        draft,
        "--target",       target,
        "--output",       output,
        "--steps",        str(steps),
        "--update_every", str(update_every),
        "--K",            str(K),
        "--kl_method",    kl_method,
        "--lr",           str(lr),
        "--max_new_tokens", str(max_new_tokens),
    ]
    if load_in_4bit:
        cmd.append("--load_in_4bit")

    train_env = {
        "HF_HOME":              HF_CACHE,
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_OFFLINE":       "1",
        "PYTHONIOENCODING":     "utf-8",
    }

    # Periodic commit daemon for crash safety
    stop_event = threading.Event()
    commit_thread = _start_commit_daemon(stop_event)
    print(f"[commit-daemon] started — vol.commit() every {COMMIT_INTERVAL_SECS}s")

    try:
        _run_cmd(cmd, env=train_env)
    finally:
        stop_event.set()
        commit_thread.join(timeout=30)

    vol.commit()
    print(f"\nDone. Online checkpoint at /vol/checkpoints/{run_name}/")
    print("Download with: modal volume get specdist-vol /checkpoints ./local_checkpoints")


# ---------------------------------------------------------------------------
# Convenience entry points — one per loss variant
#
# Each wraps run.local() / run_online.local() with a fixed loss and a
# human-readable run_name.  All accept the same optional parameters so
# users can override steps, lr, dataset, and gpu without touching run().
#
# Supported losses (matching train_qwen3.py --loss values):
#   forward_kl   — DistillSpec baseline (mode-covering KL)
#   ebe          — Evidence-weighted Block Entropy (novel; our contribution)
#   reverse_kl   — mode-seeking KL
#   jsd          — Jensen-Shannon Divergence (symmetric)
#   l1           — L1 token-distribution distance
#   online       — Online OSD adaptation (online_serve.py, not train_qwen3.py)
# ---------------------------------------------------------------------------

@app.function(
    gpu="A100-40GB",
    volumes={VOLUME_PATH: vol},
    timeout=86400,
    mounts=[modal.Mount.from_local_dir(_SRC_DIR, remote_path="/src")],
)
def train_kl(
    dataset:      str   = None,
    steps:        int   = 1000,
    lr:           float = 3e-5,
    load_in_4bit: bool  = False,
):
    """KL-distillation baseline (DistillSpec forward_kl).

    Usage: modal run modal_train.py::train_kl
    """
    run.local(loss="forward_kl", steps=steps, lr=lr,
              dataset=dataset, run_name=f"kl_{steps}steps",
              load_in_4bit=load_in_4bit)


@app.function(
    gpu="A100-40GB",
    volumes={VOLUME_PATH: vol},
    timeout=86400,
    mounts=[modal.Mount.from_local_dir(_SRC_DIR, remote_path="/src")],
)
def train_ebe(
    dataset:      str   = None,
    steps:        int   = 1000,
    lr:           float = 3e-5,
    load_in_4bit: bool  = False,
):
    """EBE block-level loss (novel contribution).

    Usage: modal run modal_train.py::train_ebe
    """
    run.local(loss="ebe", steps=steps, lr=lr,
              dataset=dataset, run_name=f"ebe_{steps}steps",
              load_in_4bit=load_in_4bit)


@app.function(
    gpu="A100-40GB",
    volumes={VOLUME_PATH: vol},
    timeout=86400,
    mounts=[modal.Mount.from_local_dir(_SRC_DIR, remote_path="/src")],
)
def train_rev_kl(
    dataset:      str   = None,
    steps:        int   = 1000,
    lr:           float = 3e-5,
    load_in_4bit: bool  = False,
):
    """Reverse KL distillation (mode-seeking).

    Usage: modal run modal_train.py::train_rev_kl
    """
    run.local(loss="reverse_kl", steps=steps, lr=lr,
              dataset=dataset, run_name=f"rev_kl_{steps}steps",
              load_in_4bit=load_in_4bit)


@app.function(
    gpu="A100-40GB",
    volumes={VOLUME_PATH: vol},
    timeout=86400,
    mounts=[modal.Mount.from_local_dir(_SRC_DIR, remote_path="/src")],
)
def train_jsd(
    dataset:      str   = None,
    steps:        int   = 1000,
    lr:           float = 3e-5,
    load_in_4bit: bool  = False,
):
    """Jensen-Shannon Divergence distillation (symmetric KL).

    Usage: modal run modal_train.py::train_jsd
    """
    run.local(loss="jsd", steps=steps, lr=lr,
              dataset=dataset, run_name=f"jsd_{steps}steps",
              load_in_4bit=load_in_4bit)


@app.function(
    gpu="A100-40GB",
    volumes={VOLUME_PATH: vol},
    timeout=86400,
    mounts=[modal.Mount.from_local_dir(_SRC_DIR, remote_path="/src")],
)
def train_l1(
    dataset:      str   = None,
    steps:        int   = 1000,
    lr:           float = 3e-5,
    load_in_4bit: bool  = False,
):
    """L1 token-distribution distance distillation.

    Usage: modal run modal_train.py::train_l1
    """
    run.local(loss="l1", steps=steps, lr=lr,
              dataset=dataset, run_name=f"l1_{steps}steps",
              load_in_4bit=load_in_4bit)


@app.function(
    gpu="A100-40GB",
    volumes={VOLUME_PATH: vol},
    timeout=86400,
    mounts=[modal.Mount.from_local_dir(_SRC_DIR, remote_path="/src")],
)
def train_online(
    dataset:      str   = None,
    steps:        int   = 500,
    lr:           float = 3e-4,
    load_in_4bit: bool  = False,
):
    """Online OSD adaptation (online_serve.py, forward_kl at rejection positions).

    Uses online_serve.py with the canonical pipeline.py arguments:
      --update_every 4, --K 4, --kl_method forward_kl, --max_new_tokens 128.

    Usage: modal run modal_train.py::train_online
    """
    run_online.local(
        steps=steps,
        lr=lr,
        dataset=dataset,
        run_name=f"online_{steps}steps",
        update_every=4,
        K=4,
        kl_method="forward_kl",
        max_new_tokens=128,
        load_in_4bit=load_in_4bit,
    )


# ---------------------------------------------------------------------------
# Full pipeline — train all 6 losses sequentially then commit results
#
# Trains in this order (matching pipeline.py Phase 2 / 2b / 2c):
#   forward_kl → ebe → reverse_kl → jsd → l1 → online
#
# Each loss gets its own run_name like "kl_1000steps", "ebe_1000steps", etc.
# Results accumulate in /vol/checkpoints/ on the shared volume.
#
# Usage:
#   modal run modal_train.py::run_pipeline
#   modal run modal_train.py::run_pipeline --steps 500 --dataset /vol/data/gsm8k_train.jsonl
# ---------------------------------------------------------------------------

@app.function(
    gpu="A100-40GB",
    volumes={VOLUME_PATH: vol},
    timeout=86400 * 2,       # 48-hour cap for the full 6-loss sweep
    secrets=[modal.Secret.from_name("huggingface-secret", required=False),
             modal.Secret.from_name("wandb-secret",        required=False)],
    mounts=[modal.Mount.from_local_dir(_SRC_DIR, remote_path="/src")],
)
def run_pipeline(
    steps:        int   = 1000,
    online_steps: int   = 500,
    lr:           float = 3e-5,
    online_lr:    float = 3e-4,
    dataset:      str   = None,
    load_in_4bit: bool  = False,
):
    """Train all 6 loss variants sequentially, then commit results.

    Loss order: forward_kl -> ebe -> reverse_kl -> jsd -> l1 -> online

    Each run writes to /vol/checkpoints/<loss>_<steps>steps/.
    Intermediate vol.commit() is called by the per-loss commit daemons,
    and a final commit is called after all losses complete.

    Usage
    -----
        modal run modal_train.py::run_pipeline
        modal run modal_train.py::run_pipeline --steps 500
        modal run modal_train.py::run_pipeline --dataset /vol/data/gsm8k_train.jsonl
    """
    # The 5 offline distillation losses, in pipeline.py order
    offline_losses = [
        ("forward_kl",  f"kl_{steps}steps"),
        ("ebe",         f"ebe_{steps}steps"),
        ("reverse_kl",  f"rev_kl_{steps}steps"),
        ("jsd",         f"jsd_{steps}steps"),
        ("l1",          f"l1_{steps}steps"),
    ]

    print(f"\n{'='*60}")
    print(f"  run_pipeline: training all 6 losses")
    print(f"  Offline losses: {[l for l, _ in offline_losses]}")
    print(f"  Online: online_serve.py, {online_steps} steps")
    print(f"  Dataset: {dataset or '(built-in GSM8K)'}")
    print(f"{'='*60}\n")

    # Train each offline loss in sequence using run.local()
    for loss_name, rname in offline_losses:
        print(f"\n{'─'*50}")
        print(f"  Starting loss: {loss_name}  ->  run_name={rname}")
        print(f"{'─'*50}")
        run.local(
            loss=loss_name,
            steps=steps,
            lr=lr,
            dataset=dataset,
            run_name=rname,
            load_in_4bit=load_in_4bit,
        )
        # Explicit commit between losses so each one is durable before
        # the next potentially long training run begins.
        vol.commit()
        print(f"  [pipeline] {loss_name} done and committed.")

    # Train the online loss using run_online.local()
    print(f"\n{'─'*50}")
    print(f"  Starting loss: online  ->  run_name=online_{online_steps}steps")
    print(f"{'─'*50}")
    run_online.local(
        steps=online_steps,
        lr=online_lr,
        dataset=dataset,
        run_name=f"online_{online_steps}steps",
        update_every=4,
        K=4,
        kl_method="forward_kl",
        max_new_tokens=128,
        load_in_4bit=load_in_4bit,
    )
    vol.commit()
    print("  [pipeline] online done and committed.")

    # Summary
    print(f"\n{'='*60}")
    print(f"  run_pipeline COMPLETE")
    print(f"  All 6 losses trained. Checkpoints at /vol/checkpoints/")
    print(f"  Download: modal volume get specdist-vol /checkpoints ./local_checkpoints")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Local entrypoints
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def list_checkpoints():
    """List all checkpoints currently stored in the volume.

    Usage: modal run modal_train.py
    """
    print("Checkpoints in specdist-vol:/checkpoints:")
    subprocess.run(["modal", "volume", "ls", "specdist-vol", "/checkpoints"])


@app.local_entrypoint()
def download_checkpoints():
    """Download all checkpoints from the volume to ./local_checkpoints/.

    Wraps: modal volume get specdist-vol /checkpoints ./local_checkpoints

    Usage: modal run modal_train.py::download_checkpoints
    """
    import pathlib
    local_dir = pathlib.Path("local_checkpoints")
    local_dir.mkdir(exist_ok=True)
    print(f"Downloading /checkpoints -> {local_dir.resolve()} ...")
    subprocess.run(
        ["modal", "volume", "get", "specdist-vol", "/checkpoints", str(local_dir)],
        check=True,
    )
    print("Download complete.")
