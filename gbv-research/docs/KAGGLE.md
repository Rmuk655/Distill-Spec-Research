# Kaggle Setup Guide — SpecDist

Kaggle is a **100% free** platform that gives every registered user:

| Resource | Kaggle free tier | Colab free tier |
|---|---|---|
| GPU | NVIDIA T4 (16 GB VRAM) | NVIDIA T4 (16 GB VRAM) |
| System RAM | **29 GB** | ~12 GB |
| Session length | **9 hours** | ~12 h (but 60-90 min idle timeout) |
| GPU quota | **30 h / week** (resets dynamically) | ~4-6 h / week |
| Persistent storage | 20 GB /kaggle/working/ + 100 GB datasets | Google Drive |
| Accelerator options | T4 ×1 or **T4 ×2** | T4 ×1 |

**For SpecDist, Kaggle is the best free option:** longer sessions, more GPU hours,
and the 29 GB RAM unlocks the 8B teacher on a single T4.

---

## Why Kaggle can run the 8B teacher (Colab cannot)

4-bit NF4 quantization (bitsandbytes) loads each weight tensor as BF16 into **CPU RAM**
first, then converts to NF4. For Qwen3-8B: ~16 GB CPU RAM intermediates.

- **Colab** has ~12 GB system RAM → Linux OOM killer fires (SIGKILL, no Python exception) → pipeline dies silently at model load
- **Kaggle** has 29 GB system RAM → 16 GB NF4 intermediates fit → 8B teacher loads successfully

Use `CONFIG = "kaggle"` in the notebook. This runs the same 8B teacher as the A100 config, but in 4-bit NF4 so it fits on the T4's 16 GB VRAM.

**VRAM budget for `kaggle` config (single T4, 16 GB):**

| Component | VRAM |
|---|---|
| Qwen3-8B teacher (4-bit NF4) | ~4.5 GB |
| Qwen3-0.6B draft (BF16) | ~1.2 GB |
| LoRA adapters + optimizer | ~0.1 GB |
| Activations + KV cache | ~2.0 GB |
| **Total** | **~7.8 GB** — 8 GB headroom |

---

## Quick start

### 1. Create a free Kaggle account

https://www.kaggle.com → Sign up → verify email.

### 2. Create a new notebook

My Work → Code → + New Notebook

### 3. Enable GPU and Internet

Right sidebar (or Settings icon):
- **Accelerator**: GPU T4 × 1 (or T4 × 2 for dual GPU — see below)
- **Internet**: On

### 4. Add Kaggle Secrets

Add-ons → Secrets → + Add Secret (in the left sidebar of the notebook editor):

| Secret name | Value | Required? |
|---|---|---|
| `WANDB_API_KEY` | From https://wandb.ai/authorize | Optional (offline mode if missing) |
| `HF_TOKEN` | From https://huggingface.co/settings/tokens | No — Qwen3 models are public |
| `GITHUB_TOKEN` | Personal Access Token with `repo` scope | Yes, if the repo is private |

### 5. Upload the notebook

Upload `deploy/kaggle.ipynb` → set `CONFIG = "kaggle"` → Shift+F5 (Run All).

---

## Pre-uploading datasets (zero-download sessions)

Upload once locally — every future Kaggle session starts with **zero downloads**.

### How it works

| Without pre-upload | With pre-upload |
|---|---|
| Session start: ~10 min downloading 8B model from HuggingFace | Instant — weights already in attached dataset |
| Session start: ~5 s downloading GSM8K/eval JSONL from HF Datasets | Instant — JSONL files already in `core/datasets/raw/` |

`deploy_utils.py` handles both automatically when the variables are set in Cell 0:

- **`KAGGLE_HF_DATASET`** → `HF_HOME` is pointed at the dataset path. `snapshot_download()` (called by the trainer at load time) finds all weight files already cached and skips the download entirely.
- **`KAGGLE_DATA_DATASET`** → `bootstrap()` copies every `.jsonl` file from the dataset into `core/datasets/raw/` **before** `fetch_training_data()` runs. `fetch_training_data()` already skips files that exist — so the HuggingFace download never fires.

Code is always fetched via `git clone` / `git pull` (never pre-uploaded — must be latest).

---

### Step 1 — Set up Kaggle API credentials (one-time)

1. Go to https://www.kaggle.com/settings/account → **API** → **Create New Token**
2. This downloads `kaggle.json`. Place it at `~/.kaggle/kaggle.json`:
   ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
   chmod 600 ~/.kaggle/kaggle.json
   ```
3. Set your username:
   ```bash
   export KAGGLE_USERNAME=your-kaggle-username   # add to ~/.bashrc to persist
   ```

### Step 2 — Download assets locally (one-time, ~17 GB total)

```bash
pip install kagglehub huggingface_hub

# Model weights — safetensors only (no .gguf/.bin quantised files)
huggingface-cli download Qwen/Qwen3-0.6B --ignore-patterns '*.gguf' '*.bin'
huggingface-cli download Qwen/Qwen3-8B   --ignore-patterns '*.gguf' '*.bin'
# Both land in ~/.cache/huggingface/hub/

# Eval + training datasets (~10 MB total)
# Run from gbv-research/ directory:
python core/datasets/downloader.py
# Files land in core/datasets/raw/
```

### Step 3 — Upload to Kaggle

```bash
# Run from gbv-research/ directory:
python deploy/upload_to_kaggle.py
```

This calls `kagglehub.dataset_upload()` for each dataset:
- **First run**: creates the private dataset on Kaggle
- **Subsequent runs**: creates a new version (previous versions remain accessible)

Two private datasets are created:

| Dataset slug | Contents | Size | Cell 0 variable |
|---|---|---|---|
| `qwen3-hf-cache` | Qwen3-0.6B + 8B safetensors + configs | ~16 GB | `KAGGLE_HF_DATASET` |
| `specdist-datasets` | GSM8K, HumanEval, math500, alpaca, mtbench JSONL | ~10 MB | `KAGGLE_DATA_DATASET` |

Upload only one type if needed:
```bash
python deploy/upload_to_kaggle.py --only weights   # model weights only
python deploy/upload_to_kaggle.py --only data      # eval datasets only
```

### Step 4 — Attach datasets to the notebook

In the Kaggle notebook editor:
**Add Data** (right sidebar) → **Your Datasets** → select `qwen3-hf-cache` → **Add**
Repeat for `specdist-datasets`.

Both datasets mount read-only under `/kaggle/input/`.

### Step 5 — Set variables in `kaggle.ipynb` Cell 0

```python
KAGGLE_HF_DATASET   = "/kaggle/input/qwen3-hf-cache"
KAGGLE_DATA_DATASET = "/kaggle/input/specdist-datasets"
```

Run Cell 0. Models and data are available **instantly** — `bootstrap()` handles the rest.

### Updating datasets

When you add new eval datasets or download a new model variant, re-run:
```bash
python deploy/upload_to_kaggle.py
```
`kagglehub.dataset_upload()` creates a new dataset version automatically. You don't need to detach and re-attach in the notebook — Kaggle always mounts the latest version.

---

## Session timeout and keep-alive

| Timeout type | Duration | Can be prevented? |
|---|---|---|
| Inactivity (browser open) | 60 min | Yes — keep-alive JS (already in notebook) |
| Inactivity (browser closed) | 60 min | No — leave browser tab open |
| Absolute session limit | 9 hours | No — use BACKGROUND=True + resume from checkpoint |

**The keep-alive JS in Cells 0 and 1** dispatches a synthetic `mousemove` event every
45 seconds to the browser document. This keeps the Kaggle session alive for the full 9 hours
as long as the browser tab is open.

**For long runs (>9 h total work):**
1. Run `BACKGROUND=True` so the pipeline writes checkpoints and pipeline state to disk
2. When the 9-hour session expires, start a new session and run Cell 1 (Resume)
3. The pipeline reads the state file and skips all completed steps automatically
4. Kaggle's `save_every: 25` in `kaggle.yaml` means at worst 25 steps (~1-2 min) of work is lost on a crash

---

## Dual GPU: T4 × 2

Kaggle lets you enable **two T4 GPUs** for free — the only platform that does this.

### How to enable

Settings → Accelerator → **GPU T4 × 2**

### What it gives you

With `load_in_4bit=True` (the `kaggle` config), the trainer uses `device_map="auto"`,
which automatically distributes the teacher model across all available GPUs.

| Scenario | GPU 0 | GPU 1 | Notes |
|---|---|---|---|
| T4 × 1 (default) | 8B NF4 (4.5 GB) + draft (1.2 GB) | — | Already fits with 8 GB headroom |
| T4 × 2 | 8B NF4 sharded (2-3 GB each) + draft | overflow layers | Even more headroom per GPU |
| T4 × 2 (no quantization) | Requires trainer change — see below | | |

With dual T4 + NF4, the 8B teacher shards automatically across both GPUs.
The draft model stays on `cuda:0`. No code changes needed.

### Dual T4 without quantization (future, requires trainer change)

To run 8B in BF16 across two T4s (removes quantization noise):
- Teacher needs `device_map={"": 0}` on GPU 0 (16 GB exactly fits one T4)
- Draft needs `.to("cuda:1")` on GPU 1
- Loss computation needs one cross-device tensor move per step

This requires a small change to `algorithms/distillspec_gbv/trainer.py` (add a
`--teacher_device` argument and pass it to `from_pretrained`). Not yet implemented.
Use 4-bit NF4 + T4 × 2 in the meantime — quantization noise averages out over 1000 training steps.

## Persisting checkpoints across sessions

Kaggle's `/kaggle/working/` is **wiped on session restart**. To persist checkpoints:

### After each session
1. Go to **Notebook → Data → Output** tab
2. Click **+ New Dataset**
3. Name it `specdist-checkpoints`
4. This saves your entire `/kaggle/working/` directory as a Kaggle dataset

### For the next session
1. **Add Data** → Your Datasets → select `specdist-checkpoints`
2. Run Cell 1 (Resume) — it automatically restores checkpoints from `/kaggle/input/specdist-checkpoints/`

The pipeline's crash-safe state machine then skips all completed training and evaluation steps.

---

## CONFIG reference

| CONFIG | Teacher | Steps | VRAM | Time on T4 | Use when |
|---|---|---|---|---|---|
| `colab_lite` | Qwen3-1.7B (BF16) | 300 | ~5.6 GB | ~25 min | Quick trend check, first run |
| `colab` | Qwen3-4B (BF16) | 500 | ~10.7 GB | ~4 h | Safe choice, no quantization |
| `kaggle` | Qwen3-8B (4-bit NF4) | 1000 | ~7.8 GB | ~5-8 h | **Best for Kaggle** — 8B signal, fits T4 |

`kaggle` is the recommended config for Kaggle. It uses the same teacher as the A100 paper
runs (Qwen3-8B) but quantized to 4-bit NF4 to fit on T4 VRAM. The extra system RAM on
Kaggle (29 GB) makes this possible.

---

## Troubleshooting

### Model download fails / HF Hub timeout

Kaggle's outbound bandwidth is rate-limited on the free tier. If `prefetch_models()` times
out mid-download:
1. Re-run Cell 1 (Resume) — `snapshot_download` is idempotent and continues partial downloads
2. Or use the pre-uploaded dataset method above

### `UserSecretsClient` raises `BackendError`

The secret doesn't exist yet. Go to Add-ons → Secrets and add it. The `_kread()` helper
in the notebook will fall back to no-op if the secret is missing (W&B runs offline, HF
downloads anonymously).

### GPU not available

Settings → Accelerator must be set before starting the session. Changing it requires
a session restart (all in-progress work is lost). Always set it before running any cell.

### Session expired mid-run (9-hour limit)

Run Cell 1 (Resume). The pipeline state machine automatically skips all completed steps.
Worst case: 25 steps (~1-2 min on T4) of work is redone (controlled by `save_every: 25`
in `kaggle.yaml`).

### OOM during training

The `kaggle` config is sized conservatively for single T4. If you see OOM:
1. Check that you have GPU T4 ×1 or ×2 selected (not CPU)
2. Try `CONFIG = "colab"` (4B teacher, BF16, more headroom)
3. Make sure no other notebook sessions are using the same GPU quota
