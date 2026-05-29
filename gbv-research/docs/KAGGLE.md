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

## Using Kaggle Models (zero-download for model weights)

Kaggle hosts the Qwen3 model family at **[kaggle.com/models/qwen-lm/qwen-3](https://www.kaggle.com/models/qwen-lm/qwen-3)**.
No upload needed — just attach the models to your notebook and they appear instantly.

| Without Kaggle Models | With Kaggle Models |
|---|---|
| ~5–10 min downloading 8B model at every session start | Instant — weights already attached |
| Counts against Kaggle internet quota | No download, no quota |

### Step 1 — Attach models in the notebook editor

In the Kaggle notebook editor:
1. Click **Add-ons** (top menu) → **Add Model**
2. Search for `qwen-3` → select **qwen-lm/qwen-3**
3. Choose framework: **Transformers** → variation: **0.6b** → **Add**
4. Repeat for the **8b** variation

Both mount read-only at:
- `/kaggle/input/models/qwen-lm/qwen-3/transformers/0.6b/1` — Qwen3-0.6B draft
- `/kaggle/input/models/qwen-lm/qwen-3/transformers/8b/1` — Qwen3-8B teacher

### Step 2 — Cell 0 is already configured

`kaggle.ipynb` Cell 0 already has:
```python
KAGGLE_DRAFT_MODEL  = "/kaggle/input/models/qwen-lm/qwen-3/transformers/0.6b/1"
KAGGLE_TARGET_MODEL = "/kaggle/input/models/qwen-lm/qwen-3/transformers/8b/1"
```

If those paths exist at runtime, the pipeline passes them directly to `--draft` and `--target`, bypassing HuggingFace Hub entirely. If you skip this step, it falls back to downloading automatically.

---

## Attaching the GSM8K training dataset (zero-download for train data)

The GSM8K training set (7,473 problems) is normally downloaded from HuggingFace during bootstrap (~3 MB, ~30 s). Kaggle already hosts it — attach it once and every session starts with it instantly.

### Step 1 — Attach the dataset in the notebook editor

**Add Data** (right sidebar) → search `grade-school-math-8k` → select  
**thedevastator / grade-school-math-8k-q-a** → **Add**

Mounts at `/kaggle/input/datasets/thedevastator/grade-school-math-8k-q-a/`

### Step 2 — Cell 0 is already configured

`kaggle.ipynb` Cell 0 already has:
```python
KAGGLE_GSM8K_DATASET = "/kaggle/input/datasets/thedevastator/grade-school-math-8k-q-a"
```

At startup, `bootstrap()` converts `main_train.csv` → `gsm8k_train.jsonl` (7,473 rows, matching our standard format) and skips the HuggingFace download entirely.

**Which CSV file is used:** only `main_train.csv`. The `socratic_train.csv` (Socratic prompting format) and `main_test.csv` are ignored — we use the small committed JSONL files for eval, not this CSV.

---

## Pre-uploading eval datasets (optional)

The eval JSONL files (~10 MB total) are committed to the repo and auto-downloaded on first run — no pre-upload needed for most users. If you want truly zero-network sessions:

### Step 1 — Generate datasets locally

```bash
# Run from gbv-research/
python core/datasets/downloader.py
# Files land in core/datasets/raw/
```

### Step 2 — Upload to Kaggle (one-time)

Go to https://www.kaggle.com/datasets → **+ New Dataset**:

1. Name: `specdist-datasets` → upload the `.jsonl` files from `core/datasets/raw/` → **Create**

Dataset is private by default.

### Step 3 — Attach to the notebook

**Add Data** (right sidebar) → **Your Datasets** → select `specdist-datasets` → **Add**

Mounts read-only under `/kaggle/input/specdist-datasets/`.

### Step 4 — Set variable in `kaggle.ipynb` Cell 0

```python
KAGGLE_DATA_DATASET = "/kaggle/input/specdist-datasets"
```

Run Cell 0 — eval data is available **instantly**.

### Updating datasets

To add new eval datasets: open the dataset on kaggle.com → **+ New Version** → upload the new files.

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
