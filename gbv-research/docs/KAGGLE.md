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

Use `CONFIG = "kaggle"` in the notebook. This runs the same 8B teacher as the A100 config,
but in 4-bit NF4 so it fits on the T4's 16 GB VRAM.

**VRAM budget for `kaggle` config (single T4, 16 GB):**

| Component | VRAM |
|---|---|
| Qwen3-8B teacher (4-bit NF4) | ~4.5 GB |
| Qwen3-0.6B draft (BF16) | ~1.2 GB |
| LoRA adapters + optimizer | ~0.1 GB |
| Activations + KV cache | ~2.0 GB |
| **Total** | **~7.8 GB** — 8 GB headroom |

---

## Complete setup checklist (first run)

Everything the pipeline needs is hosted on Kaggle — no external downloads required.

### Step 1 — Create a Kaggle account

https://www.kaggle.com → Sign up → verify email.

### Step 2 — Create a new notebook and upload kaggle.ipynb

My Work → Code → + New Notebook → File → Import Notebook → upload `deploy/kaggle.ipynb`.

### Step 3 — Enable GPU and Internet

Right sidebar → Settings:
- **Accelerator**: GPU T4 × 1 (or T4 × 2 for dual GPU — see below)
- **Internet**: On  _(needed for git clone + pip install; not needed for model weights)_

### Step 4 — Add Kaggle secrets

Add-ons → Secrets → + Add Secret:

| Secret name | Value | Required? |
|---|---|---|
| `WANDB_API_KEY` | From https://wandb.ai/authorize | Optional — offline mode if missing |
| `GITHUB_TOKEN` | PAT with `repo` scope | Only if the repo is private |
| `HF_TOKEN` | From https://huggingface.co/settings/tokens | **Not needed** — all data comes from Kaggle |
| `KAGGLE_USERNAME` | Your Kaggle username | Only for **auto-backup** (Cell 2b) |
| `KAGGLE_KEY` | Token from kaggle.com → Settings → API → Create New Token | Only for **auto-backup** (Cell 2b) |

> **Important — enable each secret for the notebook.**  
> Secrets exist in your Kaggle account but must be explicitly granted per notebook.  
> In the Secrets panel, **tick the checkbox** next to each secret you want accessible.  
> If the checkbox is unchecked, `UserSecretsClient().get_secret()` returns empty and  
> git clone will fail with _"could not read Username"_.

### Step 5 — Attach 2 models and 1 dataset

All three are on Kaggle. Attach them once; they persist across every session automatically.

#### Model weights (Add-ons → Add Model → search `qwen-3`)

| Model | Framework | Variation | Mount path |
|---|---|---|---|
| Qwen3-0.6B draft | Transformers | **0.6b** | `/kaggle/input/models/qwen-lm/qwen-3/transformers/0.6b/1` |
| Qwen3-8B teacher | Transformers | **8b** | `/kaggle/input/models/qwen-lm/qwen-3/transformers/8b/1` |

> **Which 8b variant to pick:** plain `8b` (not `8b-base`, `8b-fp8`, `8b-awq`).  
> Our code applies 4-bit NF4 quantization itself via bitsandbytes — pre-quantized variants conflict.

#### Training data (Add Data → search `grade-school-math-8k`)

| Dataset | Owner | Files used |
|---|---|---|
| grade-school-math-8k-q-a | thedevastator | `main_train.csv` → converted to `gsm8k_train.jsonl` |

> `socratic_train.csv` and `main_test.csv` are ignored. Eval JSONL files come from the git clone — nothing to attach.

### Step 6 — Cell 0 is already configured

All paths are pre-set in `kaggle.ipynb` Cell 0. No editing needed unless you want to change CONFIG.

**Run all: Shift+F5.**

---

## What bootstrap does (and what it skips)

With all three attachments above, bootstrap startup time drops to ~2 min (pip install only):

| Step | Without Kaggle attachments | With Kaggle attachments |
|---|---|---|
| Install pip deps | ~2 min | ~2 min (always required) |
| Download Qwen3-0.6B (1.3 GB) | ~1-2 min | **Skipped** — mounted from Kaggle Models |
| Download Qwen3-8B (16 GB) | ~5-10 min | **Skipped** — mounted from Kaggle Models |
| Download gsm8k_train.jsonl (3 MB) | ~30 s | **Skipped** — converted from CSV in <1 s |
| HF authentication | ~1 s | ~1 s (still runs, harmless) |
| W&B authentication | ~1 s | ~1 s |
| **Total startup** | **~10-15 min** | **~2 min** |

The pipeline passes model paths directly to `--draft` and `--target` in `experiment.py`,
so the HF Hub is never contacted for model weights. Tokenizer files load from the same
mounted paths (they're included in the Kaggle Model variant).

---

## Monitoring a running pipeline (Cell 2)

Cell 2 is a **print-to-cell log viewer** — it does not start a web server or open a URL.
Output appears directly below the cell in the notebook.

```
Run Cell 2 → output appears below ↓
```

| Setting | Effect |
|---|---|
| `AUTO_REFRESH = False` | Print once and stop |
| `AUTO_REFRESH = True` | Clear and reprint every `REFRESH_SECS` seconds — interrupt the cell (■) to stop |

**What it shows:**
- Every pipeline step with status: `OK` done · `>>` running · `..` pending · `XX` failed
- Last 60 lines of `pipeline_output.log`
- Any per-step error logs

> The Flask web dashboard (`start_dashboard()`) works on Colab only — it uses  
> `google.colab.output.eval_js` which does not exist on Kaggle.  
> Cell 2 with `AUTO_REFRESH = True` is your monitoring view on Kaggle.

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
1. Run `BACKGROUND=True` so the pipeline writes checkpoints and state to disk
2. When the session expires, start a new session and run Cell 1 (Resume)
3. The pipeline state machine skips all completed steps automatically
4. `kaggle.yaml` sets `save_every: 25` — at worst 25 steps (~1-2 min) of work is redone

---

## Dual GPU: T4 × 2

Kaggle lets you enable **two T4 GPUs** for free — the only free platform that does this.

Settings → Accelerator → **GPU T4 × 2**

With `load_in_4bit=True` (the `kaggle` config), `device_map="auto"` shards the teacher
model across both GPUs automatically. No code changes needed.

| Scenario | GPU 0 | GPU 1 |
|---|---|---|
| T4 × 1 (default) | 8B NF4 (4.5 GB) + draft (1.2 GB) | — |
| T4 × 2 | 8B NF4 sharded (2-3 GB each) + draft | overflow layers |

---

## Persisting checkpoints across sessions

Kaggle's `/kaggle/working/` is **wiped on session restart** (idle timeout, the
9-hour limit, or a browser disconnect). Anything still in
`/kaggle/working/specdist/checkpoints/` at that moment is lost. Pick **one** of
the options below — auto-backup (recommended) is the only one that survives an
unattended overnight reset without manual steps.

### Option A — Auto-backup to a Kaggle Dataset (recommended)

Cell 2b snapshots `checkpoints/` (+ `results.db`) to a Kaggle Dataset every
30 min and once more when the run finishes, so a session reset only ever costs
the **in-progress** loss — all finished losses are restored automatically next
session.

1. One-time: add `KAGGLE_USERNAME` and `KAGGLE_KEY` as Secrets (see Step 4) and
   tick their checkboxes. Get the key from kaggle.com → Settings → API →
   **Create New Token** (the `key` field of the downloaded `kaggle.json`).
2. Each session: run **Cell 1** (Resume), then run **Cell 2b** (Auto-backup) in
   its own cell and leave it running. The first run **creates** the
   `specdist-checkpoints` dataset; later runs push new **versions**. It also
   doubles as a keep-alive; interrupt the cell to stop (it does a final backup).
3. Next session: **Add Data** → Your Datasets → attach `specdist-checkpoints` →
   run **Cell 1**, which restores everything before resuming.

> Tune the interval via `INTERVAL_MIN` in Cell 2b. To point at an existing
> dataset with a different name, set `DATASET_SLUG = "<user>/<name>"` there.

### Option B — Enable Persistence (no extra dataset)

Notebook → right sidebar → **Settings → Persistence → "Files only"** (or
"Variables and files"). `/kaggle/working/` then survives kernel restarts within
the saved notebook. Simplest, but does not protect a brand-new session.

### Option C — Manual snapshot after a session

1. **Notebook → Data → Output** tab → **+ New Dataset** → name it `specdist-checkpoints`.
2. Next session: **Add Data** → Your Datasets → attach it → run **Cell 1**, which
   restores checkpoints from `/kaggle/input/specdist-checkpoints/`.

In all cases the pipeline state machine skips completed training/eval steps on
resume, so restored work is never repeated.

> **Overnight runs:** prefer **Save Version → "Save & Run All (Commit)"**, which
> runs the notebook headless to completion regardless of your browser, with
> Option A as the safety net.

---

## CONFIG reference

| CONFIG | Teacher | Steps | VRAM | Time on T4 | Use when |
|---|---|---|---|---|---|
| `colab_lite` | Qwen3-1.7B (BF16) | 300 | ~5.6 GB | ~25 min | Quick trend check, first run |
| `colab` | Qwen3-4B (BF16) | 500 | ~10.7 GB | ~4 h | Safe choice, no quantization |
| `kaggle` | Qwen3-8B (4-bit NF4) | 1000 | ~7.8 GB | ~5-8 h | **Best for Kaggle** |

`kaggle` uses the same 8B teacher as the A100 paper runs but quantized to 4-bit NF4 to
fit T4 VRAM. Quantization noise averages out over 1000 training steps.

---

## Troubleshooting

### Model path not found (`os.path.isdir` returns False)

The Kaggle Model was not attached, or the variation was wrong.
- Confirm: Add-ons → the model should appear in the input panel
- Check the path: it must be **Transformers framework**, not GGUF or AWQ
- Variation must be exactly `0.6b` and `8b` (plain, not `0.6b-base`, `8b-fp8`, etc.)
- If paths don't exist, the notebook falls back to HF Hub download automatically (~10 min)

### `git clone` fails — _"could not read Username"_ or exit code 128

Two separate causes:

**Cause 1: secret not enabled for this notebook.**  
The secret exists in your account but the checkbox next to it in the Secrets panel is unchecked.  
Fix: Add-ons → Secrets → tick the checkbox next to `GITHUB_TOKEN` → re-run Cell 0.

**Cause 2: secret doesn't exist yet.**  
Fix: Add-ons → Secrets → + Add Secret → Name: `GITHUB_TOKEN` → Value: your PAT → Save → tick checkbox → re-run Cell 0.  
Create a classic PAT at github.com → Settings → Developer settings → Personal access tokens → tick `repo` scope.

### `UserSecretsClient` raises `BackendError`

The secret doesn't exist yet. Add-ons → Secrets → add it. The `_kread()` helper falls back
to no-op if missing (W&B runs offline, git clone uses HTTPS without auth).

### GPU not available

Settings → Accelerator must be set **before** starting the session. Changing it mid-session
requires a restart (all in-progress work is lost). Always set GPU before running any cell.

### Session expired mid-run (9-hour limit)

Run **Cell 1** (Resume). Worst case: 25 steps (~1-2 min) of work is redone.

### OOM during training

1. Check GPU T4 ×1 or ×2 is selected (not CPU)
2. Try `CONFIG = "colab"` (4B teacher, BF16, more VRAM headroom)
3. Confirm no other Kaggle sessions are using your GPU quota simultaneously
