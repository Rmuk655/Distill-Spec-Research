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

Train:
    modal run modal_train.py::train_kl
    modal run modal_train.py::train_ebe

Train with custom args:
    modal run modal_train.py::run --loss ebe --steps 2000 --lr 1e-4

Monitor: while running, tail logs with
    modal app logs <app-id>

Download results locally:
    modal volume get specdist-vol /checkpoints ./local_checkpoints

Notes
-----
- TRANSFORMERS_OFFLINE is set to 0 so the first run can download models.
  After download_models completes, subsequent runs can set it back to 1.
- The volume is mounted at /vol inside the container.
- Checkpoints are written to /vol/checkpoints/{run_name}/.
- Copy your data/ directory to the volume with:
    modal volume put specdist-vol data/ /data/
  Then use --dataset /data/gsm8k_train.jsonl
"""

import os
import subprocess
import sys

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


def _run_cmd(cmd: list[str], env: dict = None):
    """Run a command with merged environment, streaming output."""
    merged = {**os.environ, **(env or {})}
    result = subprocess.run(cmd, env=merged, check=True)
    return result


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
# Core training function
# ---------------------------------------------------------------------------

@app.function(
    gpu=modal.gpu.A100(size="80GB"),  # Qwen3-8B target needs ~16 GB fp16 → 80 GB headroom
    volumes={VOLUME_PATH: vol},
    timeout=86400,                    # 24-hour cap
    secrets=[modal.Secret.from_name("huggingface-secret", required=False),
             modal.Secret.from_name("wandb-secret",        required=False)],
    mounts=[modal.Mount.from_local_dir(
        # Mount the OSD source directory into the container so we can run train_qwen3.py
        os.path.dirname(os.path.abspath(__file__)),
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
):
    """Train the draft model with SpecDist distillation on Modal A100.

    Examples
    --------
    KL baseline:
        modal run modal_train.py::run --loss forward_kl --steps 1000

    EBE novel loss:
        modal run modal_train.py::run --loss ebe --steps 1000

    Custom dataset from volume:
        modal run modal_train.py::run --dataset /vol/data/gsm8k_train.jsonl

    Checkpoints are saved to /vol/checkpoints/{run_name}/.
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

    _run_cmd(cmd, env={
        "HF_HOME":              HF_CACHE,
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_OFFLINE":       "1",
        "PYTHONIOENCODING":     "utf-8",
    })

    # Persist checkpoint to the volume
    vol.commit()
    print(f"\nDone. Checkpoints at /vol/checkpoints/{run_name}/")
    print("Download with: modal volume get specdist-vol /checkpoints ./local_checkpoints")


# ---------------------------------------------------------------------------
# Convenience entry points
# ---------------------------------------------------------------------------

@app.function(
    gpu=modal.gpu.A100(size="80GB"),
    volumes={VOLUME_PATH: vol},
    timeout=86400,
    mounts=[modal.Mount.from_local_dir(
        os.path.dirname(os.path.abspath(__file__)), remote_path="/src")],
)
def train_kl(dataset: str = None, steps: int = 1000, lr: float = 3e-5):
    """KL-distillation baseline (DistillSpec).  Usage: modal run modal_train.py::train_kl"""
    run.local(loss="forward_kl", steps=steps, lr=lr,
              dataset=dataset, run_name=f"kl_{steps}steps")


@app.function(
    gpu=modal.gpu.A100(size="80GB"),
    volumes={VOLUME_PATH: vol},
    timeout=86400,
    mounts=[modal.Mount.from_local_dir(
        os.path.dirname(os.path.abspath(__file__)), remote_path="/src")],
)
def train_ebe(dataset: str = None, steps: int = 1000, lr: float = 3e-5):
    """EBE block-level loss (novel).  Usage: modal run modal_train.py::train_ebe"""
    run.local(loss="ebe", steps=steps, lr=lr,
              dataset=dataset, run_name=f"ebe_{steps}steps")


# ---------------------------------------------------------------------------
# Local entrypoint — list checkpoints in the volume
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def list_checkpoints():
    """List all checkpoints currently stored in the volume.
    Usage: modal run modal_train.py
    """
    import subprocess
    print("Checkpoints in specdist-vol:/checkpoints:")
    subprocess.run(["modal", "volume", "ls", "specdist-vol", "/checkpoints"])
