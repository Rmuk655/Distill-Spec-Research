"""
modal_app.py — Run the full SpecDist pipeline on Modal.com A100.

Modal gives on-demand A100/H100 GPUs billed per second.  Models and
checkpoints live in a persistent Volume so you never re-download 16 GB
on every run.

Prerequisites
-------------
    pip install modal
    modal token new          # one-time browser auth

One-time model download (run once, ~15 min):
    modal run launchers/modal_app.py::download_models

Full pipeline (all 6 losses, ~4–6 h on A100):
    modal run launchers/modal_app.py::run_pipeline

Smoke test (code-path check, ~45 min):
    modal run launchers/modal_app.py::run_pipeline --smoke

Single loss:
    modal run launchers/modal_app.py::run_pipeline --losses kl

Custom GPU (default A100-40GB; A10G is cheaper):
    modal run launchers/modal_app.py::run_pipeline --gpu A10G

Download results locally:
    modal volume get specdist-vol /checkpoints ./local_checkpoints
    modal volume get specdist-vol /results.db  ./results.db

Monitor W&B:
    https://wandb.ai/{your-entity}/distillspec

Crash recovery:
    Modal containers can be killed mid-run (preemption, OOM, timeout).
    A background thread flushes vol.commit() every 5 min so at most
    5 min of work is lost.  train_qwen3.py auto-resumes from ckpt_latest/.
    Just re-run the same command to continue.
"""

import os
import subprocess
import sys
import threading

import modal

# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "curl")
    .pip_install(
        # Core deps — pin torch for reproducibility
        "torch==2.3.0",
        "transformers>=4.55.0,<6.0",
        "peft>=0.11.0",
        "accelerate>=0.30.0",
        "bitsandbytes>=0.43.0",   # QLoRA (--load_in_4bit for T4-class GPUs)
        "sentencepiece",
        "datasets",
        "scipy",
        "wandb",
        "pyyaml>=6.0",
        "flask",
        "flask-cors",
    )
    .run_commands(
        # flash-attn needs CUDA headers — build inside the image
        "pip install flash-attn --no-build-isolation || echo '[modal] flash-attn skipped — using sdpa'"
    )
    .run_commands(
        # Clone the research repo into the image so all code is available
        "git clone --depth 1 https://github.com/Rmuk655/Distill-Spec-Research.git /repo"
    )
)

# ---------------------------------------------------------------------------
# Persistent volume — model cache + checkpoints
# ---------------------------------------------------------------------------

vol = modal.Volume.from_name("specdist-vol", create_if_missing=True)

VOL_PATH      = "/vol"
STORAGE_ROOT  = VOL_PATH          # all specdist artifacts live at volume root
HF_CACHE      = f"{VOL_PATH}/hf_cache"
CKPT_ROOT     = f"{STORAGE_ROOT}/checkpoints"   # derived; used in download hints
DATA_PATH     = f"{VOL_PATH}/data"
RESULTS_DB    = f"{STORAGE_ROOT}/results.db"

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = modal.App("specdist", image=image)

# ---------------------------------------------------------------------------
# Volume commit daemon — crash safety
#
# Flushes vol.commit() every COMMIT_INTERVAL seconds while training.
# On container crash: at most COMMIT_INTERVAL seconds of work is lost.
# ---------------------------------------------------------------------------

COMMIT_INTERVAL = 300   # 5 minutes


def _start_commit_daemon(stop_event: threading.Event) -> threading.Thread:
    def _loop():
        while not stop_event.wait(timeout=COMMIT_INTERVAL):
            try:
                vol.commit()
                print(f"[vol-daemon] commit OK ({COMMIT_INTERVAL}s interval)")
            except Exception as exc:
                print(f"[vol-daemon] commit failed (non-fatal): {exc}")
    t = threading.Thread(target=_loop, daemon=True, name="vol-commit-daemon")
    t.start()
    return t


# ---------------------------------------------------------------------------
# Shared env setup
# ---------------------------------------------------------------------------

def _setup_env(wandb_key: str | None = None, hf_token: str | None = None):
    """Set environment variables used by experiment.py and train_qwen3.py."""
    os.environ["HF_HOME"]              = HF_CACHE
    os.environ["TRANSFORMERS_OFFLINE"] = "0"   # always online on Modal
    os.environ["HF_HUB_OFFLINE"]       = "0"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

    # ── Specdist storage env vars ───────────────────────────────────────────
    # experiment.py reads these and propagates them to all subprocesses, so every
    # tool (train, eval, online) writes to the same persistent volume paths.
    os.environ["SPECDIST_STORAGE_ROOT"] = STORAGE_ROOT
    os.environ["SPECDIST_DB_PATH"]       = RESULTS_DB
    os.environ["SPECDIST_LOGS_ROOT"]     = os.path.join(STORAGE_ROOT, "logs")

    if wandb_key:
        os.environ["WANDB_API_KEY"] = wandb_key
    if hf_token:
        os.environ["HF_TOKEN"]                  = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"]    = hf_token

    # Create dirs
    os.makedirs(HF_CACHE,  exist_ok=True)
    os.makedirs(CKPT_ROOT, exist_ok=True)
    os.makedirs(DATA_PATH, exist_ok=True)
    os.makedirs(os.environ["SPECDIST_LOGS_ROOT"], exist_ok=True)


# ---------------------------------------------------------------------------
# Step 0 — Download models into the volume (run ONCE)
# ---------------------------------------------------------------------------

@app.function(
    gpu="A10G",                        # A10G is cheapest; enough for downloads
    volumes={VOL_PATH: vol},
    timeout=3600,
    secrets=[
        modal.Secret.from_name("huggingface-secret", required=False),
        modal.Secret.from_name("wandb-secret",       required=False),
    ],
)
def download_models(
    draft:  str = "Qwen/Qwen3-0.6B",
    target: str = "Qwen/Qwen3-8B",
):
    """Download draft + target models into the volume's HF cache.

    Run this ONCE before the first training run so subsequent runs start
    immediately without downloading 16 GB every time.

        modal run launchers/modal_app.py::download_models
    """
    import transformers

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    _setup_env(hf_token=hf_token)

    print(f"\n=== Downloading models into {HF_CACHE} ===")
    for model_id in [draft, target]:
        print(f"\n--- {model_id} ---")
        tok = transformers.AutoTokenizer.from_pretrained(model_id)
        tok.save_pretrained(os.path.join(HF_CACHE, model_id.replace("/", "--")))
        mdl = transformers.AutoModelForCausalLM.from_pretrained(
            model_id, low_cpu_mem_usage=True
        )
        del mdl   # just prime the cache
        print(f"  ✓ {model_id} cached")

    vol.commit()
    print("\n=== Download complete — volume committed ===")


# ---------------------------------------------------------------------------
# Step 1 — Upload training dataset
# ---------------------------------------------------------------------------

@app.function(
    volumes={VOL_PATH: vol},
    timeout=300,
)
def upload_dataset(local_path: str = "capsules/datasets/raw/gsm8k_train.jsonl"):
    """Upload a local dataset JSONL into the volume.

    Usage:
        modal run launchers/modal_app.py::upload_dataset \\
            --local-path capsules/datasets/raw/gsm8k_train.jsonl
    """
    import shutil
    dest = os.path.join(DATA_PATH, os.path.basename(local_path))
    shutil.copy(local_path, dest)
    vol.commit()
    print(f"Uploaded {local_path} → {dest}")


# ---------------------------------------------------------------------------
# Step 2 — Full pipeline (train → merge → eval, all losses)
# ---------------------------------------------------------------------------

@app.function(
    gpu="A100-40GB",                   # switch to A10G (24 GB) to save cost
    volumes={VOL_PATH: vol},
    timeout=36000,                     # 10 h hard limit
    secrets=[
        modal.Secret.from_name("wandb-secret",       required=False),
        modal.Secret.from_name("huggingface-secret", required=False),
    ],
)
def run_pipeline(
    gpu:    str          = "A100-40GB",
    smoke:  bool         = False,
    losses: str | None   = None,        # e.g. "kl,ebe" or None for all
    config: str          = "server",    # pipeline config name (server / colab)
    extra:  list[str]    = [],
):
    """Run the full SpecDist pipeline on Modal A100.

    Examples
    --------
    Full run:
        modal run launchers/modal_app.py::run_pipeline

    Smoke test:
        modal run launchers/modal_app.py::run_pipeline --smoke

    Single loss:
        modal run launchers/modal_app.py::run_pipeline --losses kl

    Cheap A10G GPU:
        modal run launchers/modal_app.py::run_pipeline --gpu A10G --config colab
    """
    wandb_key = os.environ.get("WANDB_API_KEY")
    hf_token  = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    _setup_env(wandb_key=wandb_key, hf_token=hf_token)

    repo_dir = "/repo/gbv-research"
    sys.path.insert(0, repo_dir)

    pipeline_cmd = [
        sys.executable,
        os.path.join(repo_dir, "orchestration", "experiment.py"),
        "--config",       config,
        "--storage_root", STORAGE_ROOT,   # pins ALL artifacts to the persistent volume
        "--yes",
    ]
    if smoke:
        pipeline_cmd.append("--smoke")
    if losses:
        pipeline_cmd += ["--losses", losses]
    pipeline_cmd += extra

    print(f"\n=== SpecDist pipeline on Modal ===")
    print(f"  GPU          : {gpu}")
    print(f"  Config       : {config}")
    print(f"  Storage root : {STORAGE_ROOT}")
    print(f"  Results DB   : {RESULTS_DB}")
    print(f"  Checkpoints  : {CKPT_ROOT}")
    print(f"  Cmd          : {' '.join(pipeline_cmd)}")
    print()

    # Start volume commit daemon (crash safety)
    stop_event = threading.Event()
    daemon = _start_commit_daemon(stop_event)

    try:
        result = subprocess.run(pipeline_cmd, cwd=repo_dir)
    finally:
        stop_event.set()
        daemon.join(timeout=10)
        vol.commit()
        print("[modal] Final vol.commit() done.")

    if result.returncode != 0:
        raise SystemExit(f"Pipeline failed (exit code {result.returncode})")

    print("\n=== Pipeline complete ===")
    print(f"  Storage root : {STORAGE_ROOT}")
    print(f"  Results DB   : {RESULTS_DB}")
    print(f"  Checkpoints  : {CKPT_ROOT}")
    print(f"  To download:")
    print(f"    modal volume get specdist-vol results.db ./results.db")
    print(f"    modal volume get specdist-vol checkpoints ./local_checkpoints")
    print(f"    modal volume get specdist-vol logs ./logs")


# ---------------------------------------------------------------------------
# Utility — download results back to local disk
# ---------------------------------------------------------------------------

@app.function(volumes={VOL_PATH: vol})
def download_checkpoints(dest: str = "./modal_checkpoints"):
    """Print the volume get command to download checkpoints locally."""
    print(f"\nTo download checkpoints, run:")
    print(f"  modal volume get specdist-vol /checkpoints {dest}")
    print(f"\nTo download results.db:")
    print(f"  modal volume get specdist-vol /results.db ./results.db")


# ---------------------------------------------------------------------------
# Entry point: `python launchers/modal_app.py` prints a help summary
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(__doc__)
