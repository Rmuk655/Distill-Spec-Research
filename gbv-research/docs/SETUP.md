# Setup Guide

Step-by-step instructions for getting a new machine running the full pipeline,
including WandB credentials, first smoke test, and dashboard.

---

## 1. Prerequisites

```bash
# Python 3.10+, CUDA 11.8+
pip install -r requirements.txt
# transformers >= 4.51 required for Qwen3 support
```

GPU requirements:

| Config | Models | VRAM needed | Notes |
|--------|--------|-------------|-------|
| `laptop` | Qwen2.5-0.5B → Qwen3-0.6B | 4 GB | RTX 500 / any modern laptop GPU |
| `colab`  | Qwen3-0.6B → Qwen3-8B (4-bit) | 9 GB | Free Colab T4 (15 GB) — teacher in 4-bit NF4 |
| `server` | Qwen3-0.6B → Qwen3-8B (bf16) | 24 GB | A10G / A100 / 3090 — full precision |

**Colab free T4 note**: Qwen3-8B in bfloat16 = ~16 GB → OOM on T4 (15 GB).
The `colab` config automatically loads the frozen teacher in **4-bit NF4** via
`bitsandbytes` (~5 GB), bringing total VRAM to ~9 GB. Install the extra dep:

```bash
pip install bitsandbytes
```

---

## 2. Download datasets

```bash
python OSD/fetch_datasets.py --datasets gsm8k
```

This writes `gsm8k_train.jsonl` (7,473 prompts) and `gsm8k_30.jsonl` (30-prompt
eval set) to `gbv-research/core/datasets/raw/`.

---

## 3. WandB credentials (per-machine, never committed)

Each researcher creates their own `orchestration/wandb_config.json`.
This file is **gitignored** so API keys never touch the repository.

```bash
cp orchestration/wandb_config.json.example orchestration/wandb_config.json
```

Edit `orchestration/wandb_config.json` with your own credentials:

```json
{
  "api_key": "wandb_v1_PASTE_YOUR_KEY_HERE",
  "entity":  "your-wandb-username",
  "project": "specdist-gbv"
}
```

| Field     | Where to find it                                              |
|-----------|---------------------------------------------------------------|
| `api_key` | https://wandb.ai/settings → "API Keys" → copy the key       |
| `entity`  | Your W&B username (or team name for shared workspace)        |
| `project` | Project name in W&B — use `specdist-gbv` for this research   |

**How it works**: at pipeline startup, `pipeline.py` reads this file and sets
`WANDB_API_KEY`, `WANDB_ENTITY`, and `WANDB_PROJECT` as environment variables.
Every subprocess (training runs, eval runs) inherits them automatically.
If the file is absent the pipeline falls back to whatever `wandb login`
last stored in `~/.netrc` (or `~/_netrc` on Windows) — so CI/Colab runs
work without any file.

### Team credentials

| Researcher | W&B username       | Notes                               |
|------------|--------------------|-------------------------------------|
| Ram        | `rkrishnaiyer`     | Uses `~/_netrc` cached login        |
| Mukund     | `rmukund16`        | Needs `wandb_config.json` on laptop |

---

## 4. Smoke test (verify everything works, ~30-40 min)

Run this on any new machine before the overnight full run.
It exercises every loss function and every verifier at reduced scale
(50 steps instead of 1000, 5 eval prompts instead of 10).

```bash
# From gbv-research/
python orchestration/pipeline.py --config laptop --smoke --yes
```

What it runs:
- **Phase 1**: baseline eval (untrained draft, all 6 verifier modes)
- **Phase 2**: 50-step training for each of 6 losses (kl, ebe, rev_kl, jsd, l1, online)
- **Phase 3**: eval every trained model on gsm8k (all 6 verifiers)
- **Phase 4**: skipped in smoke (multi-dataset eval — too slow)

Expected time breakdown:
- Phase 1 eval: ~5 min
- Phase 2 per-loss: ~3-5 min each × 6 = ~25 min
- Phase 3 per-eval: ~3-4 min each × 6 = ~20 min

Pass criteria: no NaN/OOM errors, all 19 steps (Phase 1 + Phase 2 + Phase 3) green.

### Watch progress

```bash
# Live log
Get-Content db/logs/pipeline_output.log -Wait -Tail 40   # PowerShell
# or: tail -f db/logs/pipeline_output.log                # bash

# Status snapshot (safe to run while pipeline is running — does NOT kill it)
python orchestration/pipeline.py --config laptop --smoke --status

# Dashboard
python OSD/viz_server.py   # http://127.0.0.1:5000
```

---

## 5. Unit tests (run before smoke — < 30 s)

The unit test suite exercises every loss function, every verifier, all cache
operations, pipeline step generation, and data loading — all on CPU with no
model downloads.  Run it once on a new machine to verify the environment:

```bash
# From gbv-research/
python tests/run_unit_tests.py          # full suite  (~15 s)
python tests/run_unit_tests.py --fast   # skip slow tests  (~10 s)
```

Or directly with pytest:
```bash
python -m pytest tests/unit/ -v
```

**Pre-commit hook** — automatically installed at `.git/hooks/pre-commit`.
Every `git commit` runs the unit tests first; the commit is blocked if any test
fails.  Bypass in an emergency only with `git commit --no-verify`.

Test coverage:

| Module | File | What's tested |
|---|---|---|
| Loss functions | `test_losses.py` | forward_kl, reverse_kl (NaN guard), jsd, l1, ebe (block split, gradient) |
| Node OTLP solvers | `test_node_otlp.py` | naive, nss, specinfer, spectr, max — output range, acceptance rates |
| Verifiers | `test_verifiers.py` | bv, gbv, traversal, specinfer, naive — depth/token range, tree structure |
| KV-cache ops | `test_cache_ops.py` | slice_cache, expand_cache — shapes, content, round-trips |
| Pipeline steps | `test_pipeline_steps.py` | step IDs, --steps counts, --load_in_4bit propagation, verifier modes |
| Data loading | `test_data_loading.py` | JSONL parsing, missing keys, fallback keys, empty files |

---

## 6. Full pipeline run (after smoke passes)

```bash
python orchestration/pipeline.py --config laptop --yes
```

What changes vs smoke:
- 1000 steps/loss (instead of 50)
- n=10 eval prompts, max_tokens=50, K=3+5, temps=0.6+1.0 (instead of n=5, K=3, temp=0.6)
- Phase 4 (multi-dataset) runs: humaneval, math500, mtbench, alpaca

Expected time: 6-8 hours on laptop (RTX 500 4GB), ~2 hours on A100.

---

## 6. Dashboard

The dashboard reads results directly from `db/results.db` — no W&B dependency.

```bash
# Start from OSD/ directory
python OSD/viz_server.py          # http://127.0.0.1:5000
python OSD/viz_server.py --port 8080   # custom port
```

What it shows:
- Block efficiency table (rows = trained models, cols = verifier modes)
- Training loss curves
- Per-prompt acceptance rate breakdown
- Live pipeline log tail (polls `db/logs/pipeline_output.log`)
- Pipeline step progress bar (reads `orchestration/pipeline_state_*.json`)

---

## 7. Clean restart

If the pipeline gets into a bad state (stale checkpoints, corrupted state file):

```bash
# Wipe everything + reset state (then relaunch manually)
python orchestration/clean_restart.py --config laptop --no_restart

# OR: wipe + auto-relaunch
python orchestration/clean_restart.py --config laptop
```

What gets wiped: `db/checkpoints/`, `db/logs/`, `db/wandb/`, `db/results.db`.
What is NOT wiped: source code, datasets, config files.
Both `pipeline_state_laptop.json` and `pipeline_state_laptop_smoke.json` are reset to all-pending.

---

## 8. Ephemeral compute — Colab, Kaggle, Modal

Colab, Kaggle, and Modal all wipe local `/content` or `/tmp` disk when the
session ends.  **Without persistent storage you lose every checkpoint.**
This section shows the exact commands for each platform.

---

### 8a. Free Colab T4 — persistent checkpoints via Google Drive

**Why Drive?**  Drive is the only storage that survives a Colab session restart.
We pass `--ckpt_root` to pipeline.py so every checkpoint lands on Drive instead
of the ephemeral local disk.

```python
# ── Cell 1: mount Drive (do this FIRST, before anything else) ──────────────
from google.colab import drive
drive.mount('/content/drive')

# Create your persistent checkpoint directory once:
import os
DRIVE_CKPT = "/content/drive/MyDrive/specdist/checkpoints"
os.makedirs(DRIVE_CKPT, exist_ok=True)
print(f"Checkpoints will survive session restarts at: {DRIVE_CKPT}")

# ── Cell 2: clone repo + install deps ──────────────────────────────────────
!git clone https://github.com/Rmuk655/Distill-Spec-Research.git
%cd Distill-Spec-Research/gbv-research
!pip install -r requirements.txt
!pip install bitsandbytes    # required for 4-bit teacher on T4

# ── Cell 3: W&B auth (use Colab Secrets tab or paste key) ──────────────────
import os
os.environ["WANDB_API_KEY"] = "wandb_v1_..."   # your key
os.environ["WANDB_ENTITY"]  = "your-username"
os.environ["WANDB_PROJECT"] = "specdist-gbv"

# ── Cell 4: fetch training data (eval sets already in repo) ────────────────
!python OSD/fetch_datasets.py --datasets gsm8k

# ── Cell 5: run pipeline with Drive checkpoints ─────────────────────────────
# --ckpt_root → all checkpoints go to Drive, not the ephemeral /content disk
# --config colab → loads 8B teacher in 4-bit NF4 (fits on T4's 15 GB)
# --yes → skip interactive prompts (required in notebooks)
DRIVE_CKPT = "/content/drive/MyDrive/specdist/checkpoints"
!python orchestration/pipeline.py \
    --config colab \
    --ckpt_root {DRIVE_CKPT} \
    --yes
```

**VRAM breakdown on T4 (15 GB):**
- Qwen3-8B teacher in 4-bit NF4: ~5 GB
- Qwen3-0.6B draft in bfloat16: ~1.2 GB
- LoRA + optimizer + activations: ~2.5 GB
- **Total: ~8–9 GB** → 6 GB headroom on T4

**Resuming after session death:**

Colab sessions die after ~90 min idle (or sooner with free tier).  When you
restart:

```python
# Cell 1: remount Drive (always first)
from google.colab import drive
drive.mount('/content/drive')

# Cell 2: re-run pipeline — it auto-skips steps whose done_check files exist on Drive
DRIVE_CKPT = "/content/drive/MyDrive/specdist/checkpoints"
!python orchestration/pipeline.py \
    --config colab \
    --ckpt_root {DRIVE_CKPT} \
    --yes
```

The pipeline reads `adapter_model.safetensors` and `config.json` from Drive to
detect completed steps — no manual bookkeeping needed.

**Tips:**
- Run `--smoke` first (~45 min) to verify no crashes before the overnight run.
- Colab disconnects after ~90 min idle — keep the browser tab active or use
  Colab Pro (persistent sessions up to 12 hours).
- Drive writes are slow (~50 MB/s).  Milestone checkpoints (`--milestone_every`)
  are what matter for resume — `ckpt_latest/` is overwritten each time.

---

### 8b. Colab Pro / Kaggle / any A100 (24 GB+)

No 4-bit needed — the 8B teacher fits in bfloat16 on 24+ GB:

```python
# Colab Pro / A100: use server config (bf16 teacher, no quantisation)
DRIVE_CKPT = "/content/drive/MyDrive/specdist/checkpoints"
!python orchestration/pipeline.py \
    --config server \
    --ckpt_root {DRIVE_CKPT} \
    --yes
```

```bash
# Kaggle: attach this repo as a dataset, point ckpt_root to /kaggle/working
!python orchestration/pipeline.py \
    --config server \
    --ckpt_root /kaggle/working/specdist/checkpoints \
    --yes
```

Kaggle sessions persist for the duration of the run (up to 12 hours) but
checkpoints are NOT saved between sessions — download the output manually
from the Kaggle session output panel when done.

---

### 8c. Modal (recommended for overnight / paper-quality runs)

Modal gives on-demand A100 GPUs billed per second.  All checkpoints live in
a **persistent Modal Volume** (`specdist-vol`) that survives container
restarts — you never lose a checkpoint even if Modal preempts your run.

The training script commits the volume every 5 minutes automatically.
On restart, `train_qwen3.py` auto-resumes from `ckpt_latest/` — no flags
needed.

#### One-time setup

```bash
pip install modal
modal token new          # opens browser for auth (one-time per machine)
```

#### First run — download models (~10 min, ~16 GB, free)

```bash
# Downloads Qwen3-0.6B and Qwen3-8B into the persistent volume's HF cache.
# Run once; all subsequent training runs use the cached weights offline.
modal run OSD/modal_train.py::download_models
```

#### Upload training data

```bash
# Copies OSD/data/gsm8k_train.jsonl (and any other .jsonl) to /vol/data/
# Run after download_models and before any training.
modal run OSD/modal_train.py::upload_dataset
```

#### Train individual losses

```bash
modal run OSD/modal_train.py::train_kl       # forward KL (DistillSpec baseline)
modal run OSD/modal_train.py::train_ebe      # EBE block-level loss (novel)
modal run OSD/modal_train.py::train_rev_kl   # reverse KL (ablation)
modal run OSD/modal_train.py::train_jsd      # Jensen-Shannon (ablation)
modal run OSD/modal_train.py::train_l1       # L1 total-variation (ablation)
modal run OSD/modal_train.py::train_online   # Online OSD adaptation

# Custom args — override steps, lr, dataset:
modal run OSD/modal_train.py::run --loss ebe --steps 2000 --lr 1e-4
modal run OSD/modal_train.py::run --loss forward_kl \
    --dataset /vol/data/gsm8k_train.jsonl --steps 1000
```

#### Run the full pipeline (all 6 losses, ~2 hours on A100-40GB)

```bash
modal run OSD/modal_train.py::run_pipeline
modal run OSD/modal_train.py::run_pipeline --steps 500   # shorter sweep
```

#### Monitor progress

```bash
# In a second terminal while training is running:
modal app logs <app-id>     # tail live logs
```

#### Download checkpoints when done

```bash
# Convenience command (wraps modal volume get):
modal run OSD/modal_train.py::download_checkpoints

# Or manually:
modal volume get specdist-vol /checkpoints ./local_checkpoints
```

#### Crash resume

Modal containers can be preempted or OOM-killed.  The training script saves
`ckpt_latest/` every 100 steps (configurable via `--save_every`), and the
Modal volume is committed every 5 minutes.  To resume:

```bash
# Just re-run the same command — train_qwen3.py detects ckpt_latest/ and
# resumes from the last saved step automatically.
modal run OSD/modal_train.py::train_kl
```

#### GPU options

| GPU | VRAM | Cost | Use case |
|-----|------|------|----------|
| `A10G` | 24 GB | ~$1.10/hr | Use with `--load_in_4bit` for 8B teacher |
| `A100-40GB` | 40 GB | ~$2.50/hr | Default — 8B teacher in bfloat16 |
| `A100-80GB` | 80 GB | ~$3.70/hr | Extra headroom, long sequences |

To use a different GPU, edit `@app.function(gpu=...)` in `modal_train.py` or
run with `run.with_options(gpu="A10G").local(...)`.

---

### 8d. Manual standalone run (single training step, no pipeline orchestrator)

```bash
# Colab free T4 — must pass --load_in_4bit manually:
python OSD/train_qwen3.py \
    --draft  Qwen/Qwen3-0.6B \
    --target Qwen/Qwen3-8B \
    --loss   forward_kl \
    --load_in_4bit \
    --steps  1000 \
    --output /content/drive/MyDrive/specdist/checkpoints/kl-8b

# Server / A100 — no 4-bit flag needed:
python OSD/train_qwen3.py \
    --draft  Qwen/Qwen3-0.6B \
    --target Qwen/Qwen3-8B \
    --loss   forward_kl \
    --steps  1000 \
    --output ./db/checkpoints/kl-8b
```

**Notes for all ephemeral sessions:**
- Always pass `--yes` on Colab/Kaggle/Modal to skip interactive prompts.
- The pipeline state file (`pipeline_state_colab.json`) is on local disk and
  lost on session death — but `--ckpt_root` on Drive/volume means `done_check`
  files persist, so re-running `pipeline.py --yes` auto-skips completed steps.
- W&B logs sync to the cloud in real time — they are always preserved even if
  the session dies mid-run.

---

## 9. MLOps — hyperparameter overrides, loss filtering, and W&B sweeps

### 9a. No-code experiment variation

All training hyperparameters are exposed as CLI flags so researchers can run
experiments without editing any source file.

**Run only specific losses** (skip the rest):
```bash
# Train and eval only KL and EBE — skip rev_kl, jsd, l1, online
python orchestration/pipeline.py --config laptop --smoke --losses kl,ebe
```

**Override steps per loss** (e.g., quick 200-step ablation):
```bash
python orchestration/pipeline.py --config laptop --train_steps 200 --losses kl,ebe
```

**Override learning rate** (single run or sweep baseline):
```bash
python orchestration/pipeline.py --config laptop --lr 1e-4 --losses kl
```

**Override LoRA rank** (compare r=8 vs r=16 on laptop):
```bash
python orchestration/pipeline.py --config laptop --lora_r 16 --losses kl,ebe
```

**Standalone `train_qwen3.py` with all sweep-friendly args**:
```bash
python OSD/train_qwen3.py \
    --loss ebe \
    --steps 500 \
    --lr 5e-5 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.0 \
    --teacher_temp 1.0 \
    --ebe_kl_weight 0.05 \
    --grad_clip 0.5 \
    --draft Qwen/Qwen3-0.6B \
    --target Qwen/Qwen3-8B \
    --dataset OSD/data/gsm8k_train.jsonl \
    --output db/checkpoints/ebe-ablation-v1
```

### 9b. YAML config files — per-environment defaults

Each compute environment has a YAML in `orchestration/configs/`:

| File | Environment | Key settings |
|------|-------------|-------------|
| `laptop.yaml` | RTX 500, 6 GB | lr=3e-5, lora_r=8, steps=1000 |
| `server.yaml` | A100 40/80 GB | lr=3e-5, lora_r=16, steps=5000 |
| `colab.yaml`  | T4 15 GB | lr=3e-5, lora_r=8, steps=500, 4-bit teacher |

CLI args (`--lr`, `--lora_r`, `--teacher_temp`) override YAML values, which
override the hardcoded defaults. Priority: `CLI > YAML > code default`.

To add a new environment (e.g., Kaggle P100):
1. Copy `orchestration/configs/laptop.yaml` → `orchestration/configs/kaggle.yaml`
2. Adjust `hardware`, `training`, and `checkpointing` sections
3. Add `"kaggle": {"draft": ..., "target": ...}` to `CONFIGS` dict in `pipeline.py`
4. Run: `python orchestration/pipeline.py --config kaggle --ckpt_root /kaggle/working/ckpts`

### 9c. W&B hyperparameter sweeps

A W&B sweep runs many trials automatically, each with different hyperparameters.
The Bayesian optimizer finds the best combination faster than a manual grid.

**Step 1 — Register the sweep and run 20 trials locally**:
```bash
python orchestration/run_sweep.py --count 20
```

This outputs a sweep ID and a dashboard URL:
```
  [sweep] Sweep ID: abc123xyz
  [sweep] Dashboard: https://wandb.ai/<entity>/distillspec/sweeps/abc123xyz
  [sweep] Starting 20 local agent trial(s)...
```

**Step 2 — Add more parallel agents** (each runs on a separate GPU / machine):
```bash
# On another machine, pick up where the first agent left off:
wandb agent <entity>/distillspec/<sweep_id>
```

**Focused sweep — single loss, vary LR + lora_r only**:
```bash
python orchestration/run_sweep.py --loss ebe --count 12
```

**Dry-run — inspect the search space without registering**:
```bash
python orchestration/run_sweep.py --dry_run
```

**Custom sweep config** (fork `sweep.yaml` and pass it):
```bash
python orchestration/run_sweep.py \
    --sweep_config orchestration/configs/my_sweep.yaml \
    --count 30
```

**Default search space** (defined in `orchestration/configs/sweep.yaml`):

| Parameter | Distribution | Range |
|-----------|-------------|-------|
| `lr` | log-uniform | 1e-5 – 1e-4 |
| `loss` | categorical | forward_kl, ebe, reverse_kl, jsd, l1 |
| `lora_r` | categorical | 4, 8, 16 |
| `ebe_kl_weight` | log-uniform | 0.01 – 0.5 |
| `jsd_alpha` | uniform | 0.1 – 0.9 |
| `teacher_temp` | categorical | 0.6, 0.8, 1.0 |
| `grad_clip` | categorical | 0.5, 1.0, 2.0 |
| `lora_dropout` | categorical | 0.0, 0.05, 0.1 |

All sweep runs appear in W&B under group `hparam_sweep_v1` — use the
**Parallel Coordinates** chart to identify which hyperparameters drive val/loss.

### 9d. Reading sweep results

After a sweep, pull the best config:
```python
import wandb
api = wandb.Api()
sweep = api.sweep("<entity>/distillspec/<sweep_id>")
best_run = sweep.best_run()
print(best_run.config)   # → the winning hyperparameter set
```

Then lock those values in the appropriate YAML and re-run the full pipeline:
```bash
python orchestration/pipeline.py --config server \
    --lr 4.2e-5 --lora_r 16 \
    --experiment_tag "best_sweep_v1"
```
