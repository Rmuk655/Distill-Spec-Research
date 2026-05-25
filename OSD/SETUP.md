# SpecDist — Setup & Run Guide

This is a **fully offline** package once models are downloaded. You download
everything once on a machine with internet, then transfer and run with no
network required.

---

## Hardware Requirements

| Config | Draft | Target | Min VRAM |
|--------|-------|--------|----------|
| `laptop` | Qwen2.5-0.5B | Qwen3-0.6B | 2 GB (4 GB recommended) |
| `server` / `colab` | Qwen3-0.6B | Qwen3-8B | 40 GB (A100) |

**Less than 2 GB free VRAM?** The pipeline handles this automatically:
`run_all.py` detects OOM and retries on CPU. Training (`train_qwen3.py`)
does not auto-fall-back — see Troubleshooting below.

---

## Pipeline Phases

`pipeline.py` runs 18 steps across 5 phases in order:

| Phase | Steps | What it does | Est. time (laptop) |
|---|---|---|---|
| **Phase 0** | 3 | Merge pre-existing LoRA checkpoints (kl200, ebe200, lr-sweep) | ~5 min each; **auto-skipped** if `*_merged/` dirs already exist |
| **Phase 1** | 6 | Quick eval — baseline + LR-sweep models on gsm8k (alpha + BE) | ~1–2 h |
| **Phase 2** | 4 | Train KL-1000 + EBE-1000 on gsm8k_train, then merge each | ~5 h each |
| **Phase 3** | 2 | Eval the newly trained kl1000 + ebe1000 on gsm8k | ~1 h |
| **Phase 4** | 3 | Eval baseline + kl1000 + ebe1000 on humaneval, math500, mtbench, alpaca | ~2 h |

> **Why was Phase 0 skipped?** The `done_check` file for each merge step is
> `checkpoints/<name>_merged/config.json`. If that file exists on disk, the
> pipeline treats the step as already done — no re-run needed.

---

## Step 1 — Download (needs internet, run once)

On any machine with internet access:

```bash
cd OSD/
pip install -r requirements.txt

# Server config (Qwen3-0.6B draft + Qwen3-8B target + all datasets):
python setup_download.py --config server

# Laptop config (smaller models, ~2 GB total):
python setup_download.py --config laptop
```

Models cache to `~/.cache/huggingface/hub/`. Datasets save to `OSD/data/`.

```bash
# Download only models or only datasets:
python setup_download.py --config laptop --models_only
python setup_download.py --config laptop --datasets_only
```

---

## Step 2 — Transfer to Remote Machine

### Option A — rsync (SSH server)

```bash
rsync -avz --progress \
    ~/.cache/huggingface/hub/ \
    user@server:/home/user/.cache/huggingface/hub/

rsync -avz --progress \
    /path/to/OSD/ \
    user@server:/home/user/OSD/
```

### Option B — Google Colab via Google Drive

1. Zip `OSD/` and `~/.cache/huggingface/hub/`, upload to Google Drive.
2. In Colab:

```python
from google.colab import drive
drive.mount('/content/drive')

import subprocess
subprocess.run('cp -r /content/drive/MyDrive/huggingface_hub /root/.cache/huggingface/hub',
               shell=True, check=True)
subprocess.run('cp -r /content/drive/MyDrive/OSD /content/OSD',
               shell=True, check=True)
```

### Option C — Colab with internet (download directly)

If the Colab runtime has internet, skip the transfer entirely and run
`setup_download.py` directly in the notebook (see Colab section below).

---

## Step 3 — Install Dependencies

```bash
cd OSD/
pip install -r requirements.txt
```

Minimum versions known to work:

```
torch>=2.2.0
transformers>=4.40.0
peft>=0.10.0
accelerate>=0.28.0
flask>=3.0.0
```

---

## Step 4 — Run the Pipeline

```bash
cd OSD/

# Laptop config (Qwen2.5-0.5B draft -> Qwen3-0.6B target):
python pipeline.py --config laptop --yes

# Server config (Qwen3-0.6B draft -> Qwen3-8B target):
python pipeline.py --config server --yes
```

The pipeline:
- Prints GPU name and free VRAM at startup.
- Persists state to `pipeline_state_laptop.json` (or `_server.json`).
- **Crash-safe**: if it stops for any reason — power loss, OOM, keyboard
  interrupt — just re-run the same command. It reads the state file and
  skips every step already marked done. Eval steps also pass
  `--skip_existing` to `run_all.py`, so partially-completed eval cells
  are not re-run.

> **The pipeline is a foreground process — it exits when a step fails.**
> The dashboard will go idle (no running badge). Just re-run to continue:
> ```bash
> python pipeline.py --config laptop --yes
> ```
> If the failed step was a **training step**, reset its status in
> `pipeline_state_laptop.json` from `"failed"` → `"pending"` first
> (training always resumes from `ckpt_latest` — no work is lost).

```bash
# Check status without running:
python pipeline.py --config laptop --status

# Resume from a specific step:
python pipeline.py --config laptop --from eval_kl1000_gsm8k --yes

# Dry-run (print plan, run nothing):
python pipeline.py --config laptop --dry_run

# Force restart from step 1 (clears state, keeps checkpoints):
python pipeline.py --config laptop --restart --yes
```

### Experiment tags — labelling runs for historical comparison

Every run gets a unique `run_tag` (timestamp-based, e.g.
`20260522_143022_qwen2.5-0.5b_alpha_gsm8k_K1`).  On top of that you can attach
a free-text **experiment_tag** that groups all cells from one experimental
variant together — useful when you want to filter the dashboard to "v2 EBE loss"
vs "baseline".

```bash
# Label all cells produced by this pipeline run:
python pipeline.py --config laptop --yes \
    --experiment_tag "v2 EBE loss clipped accept weight"

# Or when calling run_all.py directly:
python run_all.py \
    --student Qwen/Qwen2.5-0.5B \
    --teacher Qwen/Qwen3-0.6B \
    --datasets gsm8k --modes alpha \
    --experiment_tag "baseline laptop May-22"
```

**Default tag** — if you omit `--experiment_tag`, the pipeline automatically
sets it to `{hostname}-{YYYYMMDD_HHMM}-v{n}` where `n` auto-increments per
machine per calendar day (e.g. `desktop-abc123-20260522_1430-v1` for the first
run of the day, `-v2` for the second, and so on).
This means every run is always labelled, and successive runs on the same
machine are immediately distinguishable in the dashboard.

In the viz dashboard (`python viz_server.py`) use the **Experiment Tag** chips
in the left sidebar to show/hide entire experimental variants at once, or type
in the search box to filter by substring.

---

## Step 5 — View Results Dashboard

Start the viz server once; it stays open while the pipeline runs.
The dashboard **auto-refreshes every 30 s** — no manual page refresh needed.

```bash
cd OSD/
python viz_server.py
```

Open **http://localhost:5000** in your browser.

### What the dashboard shows while the pipeline is running

**Black status bar (top of every page, always visible):**
```
SpecDist  [eval_baseline_gsm8k ●]  Phase 1 — Quick Eval  ████░░░░  3/18 (16.7%)  GPU 2.8/4.0 GB  ↻ 28s
```

| Element | What it means |
|---|---|
| **Step badge (yellow/pulsing)** | Currently executing pipeline step |
| **Phase label** | Which phase that step belongs to |
| **Progress bar `n/18`** | Done / total pipeline steps |
| **GPU bar** | Live VRAM usage (orange → red when >85% used) |
| **Last: …** | Most recently completed DB entry (model / dataset / metric) |
| **↻ Ns** | Countdown to next auto-refresh |
| **Steps button** | Expands all 18 steps with colour-coded status inline |

The bar is green when the pipeline is idle, yellow-pulsing while a step runs,
and red if a step failed.  Charts in all tabs re-draw automatically on every
30 s poll as new DB rows arrive.

### SSH port-forwarding (remote server)

On your laptop:
```bash
ssh -L 5000:localhost:5000 user@server
```

Then on the server:
```bash
cd OSD/
python viz_server.py --host 0.0.0.0 --port 5000
```

Open `http://localhost:5000` on your laptop.

### Google Colab

```python
import subprocess
from google.colab.output import eval_js

proc = subprocess.Popen(
    ['python', 'viz_server.py', '--host', '0.0.0.0', '--port', '5000'],
    cwd='/content/OSD'
)
print("Dashboard:", eval_js("google.colab.kernel.proxyPort(5000)"))
```

---

## Crash-safe training on free Colab (session restarts)

Free Colab kills your session after ~1 hour of inactivity or 12 hours total.
`train_qwen3.py` now saves a mid-training checkpoint every 100 steps
(configurable with `--save_every`).  **The key is to write the checkpoint to
Google Drive, not `/content/` which is wiped on session death.**

### How it works

Every `--save_every` steps the script writes:
```
/content/drive/MyDrive/OSD/checkpoints/kl-run/
  ckpt_latest/
    adapter_config.json   ← LoRA config
    adapter_model.safetensors  ← LoRA weights
    optimizer.pt          ← Adam state (momentum, variance)
  training_state.json     ← { "step": 400, "loss_history": [...] }
```

On restart, **re-run the exact same command**.  The script detects the
checkpoint and continues from step 401.  No flag needed.

When training finishes the state file is renamed `training_state.done.json`
so subsequent re-runs don't re-resume.

### Colab cells for crash-safe training

**Cell 1 — Mount Drive (run every session)**
```python
from google.colab import drive
drive.mount('/content/drive')
```

**Cell 2 — Install deps (run every session)**
```bash
%%bash
pip install -q torch transformers peft accelerate bitsandbytes
```

**Cell 3 — Copy code (first session only, or when you update code)**
```bash
%%bash
cp -r /content/drive/MyDrive/OSD /content/OSD
# Copy model cache too (or download with setup_download.py)
cp -r /content/drive/MyDrive/huggingface_hub /root/.cache/huggingface/hub
```

**Cell 4 — Train (re-run this cell on every restart — it auto-resumes)**
```bash
%%bash
cd /content/OSD
python train_qwen3.py \
    --draft  Qwen/Qwen3-0.6B \
    --target Qwen/Qwen3-8B \
    --loss   forward_kl \
    --steps  1000 \
    --save_every 100 \
    --output /content/drive/MyDrive/OSD/checkpoints/kl-run \
    --dataset data/gsm8k_train.jsonl \
    2>&1 | tee /content/drive/MyDrive/OSD/train_run.log
```

On first run: starts from step 0 and prints `Step 100/1000 | loss: ...`
On re-run after crash: prints `[RESUME] Checkpoint found — 400/1000 steps done`
and continues from step 401.

---

## Fitting an 8B model on free Colab (T4, 15 GB VRAM)

An 8B model in fp16 needs ~16 GB — just over the T4 limit.  Three options:

### Option A — QLoRA (recommended, no extra cost)

Load the **frozen 8B teacher in 4-bit** (~5 GB) while keeping the 0.6B draft
in bfloat16 for training.  Total VRAM: ~12 GB → fits comfortably.

```bash
python train_qwen3.py \
    --draft  Qwen/Qwen3-0.6B \
    --target Qwen/Qwen3-8B \
    --loss   forward_kl \
    --load_in_4bit \          # ← enables 4-bit NF4 teacher loading
    --steps  1000 \
    --save_every 100 \
    --output /content/drive/MyDrive/OSD/checkpoints/kl-run
```

Needs `bitsandbytes` (`pip install bitsandbytes`).  The 4-bit teacher is
frozen so its logits are very close to fp16 (NF4 is designed for this).
Your distilled draft's quality loss vs a fp16 teacher is negligible.

For **eval** (GBV block efficiency) with an 8B target, also load in 4-bit:
add `--dtype int8` to GBV calls, or use the `--load_in_4bit` flag if you
wire it through (see `GBV/util.py`).

### Option B — Kaggle free GPUs (no code changes, more hours)

Kaggle gives **30 h/week** of free GPU time with a T4 (or P100).
Same T4 limitation applies, so combine with QLoRA above.

1. Upload OSD folder to a Kaggle dataset.
2. Create a notebook, attach your dataset.
3. Run the same commands as above.

Kaggle sessions last up to 12 hours (vs ~1 hour idle on free Colab),
so fewer restarts.

### Option C — Use Qwen3-1.7B as target (no code changes, no extra deps)

Qwen3-1.7B in bfloat16 = ~3.4 GB.  With the 0.6B draft total VRAM = ~5 GB.
Loads fine on any T4 with plenty of room for activations.

```bash
python train_qwen3.py \
    --draft  Qwen/Qwen3-0.6B \
    --target Qwen/Qwen3-1.7B \   # ← instead of 8B
    --loss   forward_kl \
    --steps  1000 --save_every 100 \
    --output /content/drive/MyDrive/OSD/checkpoints/kl-1.7b-run
```

The 1.7B target is weaker than 8B, so block efficiency gains may be smaller,
but it's a fast and clean ablation point.

### VRAM summary

| Setup | Teacher VRAM | Draft VRAM | Total | Fits T4? |
|---|---|---|---|---|
| 8B fp16 | 16 GB | 2 GB | >18 GB | ✗ |
| **8B 4-bit + 0.6B fp16** | **5 GB** | **2 GB** | **~12 GB** | **✓** |
| 1.7B fp16 + 0.6B fp16 | 3.4 GB | 2 GB | ~7 GB | ✓ |
| 0.6B fp16 + 0.5B fp16 (laptop) | 1.5 GB | 1.5 GB | ~5 GB | ✓ |

---

## Google Colab — Cell-by-Cell

Runtime: **A100 GPU** (server config requires ~40 GB VRAM).

**Cell 1 — Mount Drive & copy files**
```python
from google.colab import drive
drive.mount('/content/drive')

import subprocess
subprocess.run('cp -r /content/drive/MyDrive/huggingface_hub /root/.cache/huggingface/hub',
               shell=True, check=True)
subprocess.run('cp -r /content/drive/MyDrive/OSD /content/OSD',
               shell=True, check=True)
print("Files ready.")
```

**Cell 2 — Install dependencies**
```bash
%%bash
cd /content/OSD
pip install -q -r requirements.txt

# Flash Attention 2 — A100/H100 only (free Colab T4: skip this, SDPA is used automatically)
# Gives 2-4× faster attention and ~30% less VRAM. One-time install per runtime.
pip install flash-attn --no-build-isolation
```

After installing, `train_qwen3.py` and `run_all.py` will print `Attn: flash_attention_2`
at startup. On T4 runtimes without `flash-attn`, they print `Attn: sdpa` — no action needed.

**Cell 3 — (Optional) Download models if not pre-copied**
```bash
%%bash
cd /content/OSD
python setup_download.py --config server
```

**Cell 4 — Run pipeline in background + open live dashboard**

Run the pipeline and the viz server together so you can monitor progress in
real time.  The pipeline runs in a background thread; the dashboard runs in
the cell and prints a clickable URL.  The auto-refresh (every 30 s) keeps
charts and the status bar current without any further interaction.

```python
# Cell 4 — Launch pipeline (background) + dashboard (foreground)
import subprocess, threading, time
from google.colab.output import eval_js

OSD = '/content/OSD'

# ── Start pipeline in background ───────────────────────────────────────────
# stdout/stderr written to pipeline_run.log — you can `!tail -f pipeline_run.log`
# in a separate cell at any time to see the raw log.
_pipe_log = open(f'{OSD}/pipeline_run.log', 'w')
_pipe = subprocess.Popen(
    ['python', '-u', 'pipeline.py', '--config', 'server', '--yes'],
    cwd=OSD,
    stdout=_pipe_log, stderr=subprocess.STDOUT,
)
print(f"Pipeline started  (PID {_pipe.pid}) → log: {OSD}/pipeline_run.log")

# ── Wait a moment for the pipeline to write its first state update ──────────
time.sleep(3)

# ── Start viz server (foreground — Colab needs a live cell for proxyPort) ──
_viz = subprocess.Popen(
    ['python', '-u', 'viz_server.py', '--host', '0.0.0.0', '--port', '5000'],
    cwd=OSD,
)
time.sleep(2)

url = eval_js("google.colab.kernel.proxyPort(5000)")
print(f"\nDashboard (live, auto-refreshes every 30s):\n  {url}\n")
print("Status bar at the top shows: current step  |  phase  |  n/18 progress  |  GPU memory")
print("Click 'Steps' in the bar to see all 18 steps colour-coded inline.")
print("\nTo tail raw pipeline output in another cell:  !tail -f /content/OSD/pipeline_run.log")

# Block this cell so the viz server stays alive (Ctrl+C or interrupt the cell to stop)
try:
    _pipe.wait()
    print("\nPipeline finished.")
except KeyboardInterrupt:
    print("Cell interrupted — viz server keeps running until the runtime ends.")
```

> **Tip:** If you want to run the pipeline AND keep this cell free for other
> work, interrupt it after the dashboard URL is printed — `_viz` stays alive
> as a background process for the rest of the Colab session.

**Cell 5 — (Optional) Tail the raw log while dashboard is open**
```bash
%%bash
tail -f /content/OSD/pipeline_run.log
```

**Cell 6 — Check pipeline status at any point (no dashboard needed)**
```bash
%%bash
cd /content/OSD
python pipeline.py --config server --status
```

---

## Kaggle — Cell-by-Cell

**What Kaggle gives you:** 30 h/week free GPU, T4 (15 GB VRAM) or T4 ×2 (30 GB
combined), 12-hour sessions, 20 GB `/kaggle/working/` disk, internet **ON by default**.
No Pro account needed.  Same T4 hardware as free Colab — combine with QLoRA (see
"Fitting 8B model" section) for the 8B target.

---

### Path conventions

| Path | Purpose | Survives session end? |
|------|---------|-----------------------|
| `/kaggle/working/` | Your output dir (20 GB) | Yes — appears in Output tab |
| `/kaggle/input/<dataset>/` | Read-only, attached datasets | n/a (re-attach each session) |
| `~/.cache/huggingface/` | Default HF model cache | **No** — wiped on restart |

**Key rule:** put everything you want to keep into `/kaggle/working/`.  After a
session ends, Kaggle lets you save `/kaggle/working/` as a new dataset version —
re-attach it next session to restore models and checkpoints instantly.

---

### One-time: upload your code as a Kaggle dataset

From your laptop:
```bash
# Install Kaggle CLI once
pip install kaggle
# Put kaggle.json from https://www.kaggle.com/settings → API → "Create New Token"
# into ~/.kaggle/kaggle.json

cd /path/to   # parent of OSD/
zip -r osd-code.zip OSD/
kaggle datasets create -p . --dir-mode zip   # follow prompts to name it e.g. "osd-code"
```

Whenever you update the code locally, push a new version:
```bash
kaggle datasets version -p . -m "update message"
```

---

### First session (download models + train)

**Notebook settings** *(sidebar, before running any cells)*
- Accelerator → **GPU T4 x1** (or T4 ×2 for 8B eval with 30 GB headroom)
- Internet → **ON** (needed for pip install and first-time model download)

Attach your code dataset: *Add data → Your datasets → osd-code*.

**Cell 1 — Install dependencies**
```python
import subprocess
subprocess.run(
    'pip install -q torch transformers peft accelerate bitsandbytes flask',
    shell=True, check=True
)
```

**Cell 2 — Redirect HF cache & allow downloads**

> **Important:** `%%bash` cells do **not** inherit Python `os.environ` changes.
> Set env vars inside each `%%bash` cell, or run this Python cell and then use
> `%env` / inline `VAR=val python …` syntax.

```python
import os
# Point HF cache into /kaggle/working/ so it survives as notebook output.
os.environ["HF_HOME"] = "/kaggle/working/hf_cache"
# Allow downloads this session (overrides the OFFLINE=1 default in the scripts).
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["HF_HUB_OFFLINE"]       = "0"
print("HF_HOME:", os.environ["HF_HOME"])
```

**Cell 3 — Copy OSD code to working dir**
```python
import subprocess, os
subprocess.run(
    'cp -r /kaggle/input/osd-code/OSD /kaggle/working/OSD',
    shell=True, check=True
)
os.chdir('/kaggle/working/OSD')
print("CWD:", os.getcwd())
```
Replace `osd-code` with the slug of your uploaded dataset (shown in its URL:
`kaggle.com/datasets/<you>/osd-code`).

**Cell 4 — Download models (first session only)**
```bash
%%bash
export HF_HOME=/kaggle/working/hf_cache
export TRANSFORMERS_OFFLINE=0
export HF_HUB_OFFLINE=0
cd /kaggle/working/OSD
python setup_download.py --config server
echo "Cache size: $(du -sh /kaggle/working/hf_cache)"
```

**Cell 5 — Train**
```bash
%%bash
export HF_HOME=/kaggle/working/hf_cache
export TRANSFORMERS_OFFLINE=0   # models now cached; 0 or 1 both work
cd /kaggle/working/OSD

python train_qwen3.py \
    --draft  Qwen/Qwen3-0.6B \
    --target Qwen/Qwen3-8B \
    --load_in_4bit \
    --loss   forward_kl \
    --steps  1000 \
    --save_every 100 \
    --output /kaggle/working/checkpoints/kl-run \
    --dataset data/gsm8k_train.jsonl \
    2>&1 | tee /kaggle/working/train_run.log
```

**Cell 6 — Save outputs** *(run before the session ends)*

Click **"Save Version"** (top-right corner of the notebook editor) →
*"Save & Run All (Commit)"*.  Kaggle saves `/kaggle/working/` as a versioned output.

To make the model cache and checkpoints available as a re-attachable dataset:
- Go to your notebook → **Output** tab
- Click *"New Dataset"* to create a persistent Kaggle dataset from this output

---

### Subsequent sessions (auto-resume training)

1. Open your notebook.
2. *Add data* → *Your datasets* → attach your saved **hf_cache dataset** and
   **checkpoints dataset** (created in Cell 6 above).
3. Copy them back into `/kaggle/working/`:

**Cell 2b — Restore from saved datasets**
```python
import subprocess, os

# Restore model cache
subprocess.run(
    'cp -r /kaggle/input/osd-hf-cache/hf_cache /kaggle/working/hf_cache',
    shell=True, check=True
)
# Restore training checkpoint
subprocess.run(
    'cp -r /kaggle/input/osd-checkpoints/kl-run /kaggle/working/checkpoints/kl-run',
    shell=True, check=True
)

os.environ["HF_HOME"] = "/kaggle/working/hf_cache"
os.environ["TRANSFORMERS_OFFLINE"] = "1"   # models cached — stay offline
print("Restored. hf_cache:", subprocess.check_output('du -sh /kaggle/working/hf_cache', shell=True).decode().strip())
```

4. Re-run Cell 5 — the script detects `training_state.json` and prints:
   ```
   [RESUME] Checkpoint found — 400/1000 steps done. Continuing from step 401.
   ```

---

### Quick eval run (no training)

```bash
%%bash
export HF_HOME=/kaggle/working/hf_cache
export TRANSFORMERS_OFFLINE=1   # models already cached
cd /kaggle/working/OSD

python run_all.py \
    --student Qwen/Qwen2.5-0.5B \
    --teacher Qwen/Qwen3-8B \
    --datasets gsm8k --modes gbv,specinfer \
    --K 3 --n 30 \
    2>&1 | tee /kaggle/working/eval_run.log
```

---

### Live dashboard via ngrok (run alongside the pipeline)

Kaggle has no built-in port proxy (unlike Colab's `eval_js`), so we use
**ngrok** to get a public HTTPS URL.  Run the pipeline in the background and
the dashboard in the same cell — the status bar auto-refreshes every 30 s.

**Cell — Install ngrok (once per session)**
```python
import subprocess
subprocess.run('pip install pyngrok -q', shell=True, check=True)
```

**Cell — Start pipeline in background + open live dashboard**
```python
import subprocess, time
from pyngrok import ngrok

OSD = '/kaggle/working/OSD'

# ── (Optional) set a free authtoken for longer-lived tunnels ───────────────
# Get yours from https://dashboard.ngrok.com/get-started/your-authtoken
# ngrok.set_auth_token("YOUR_TOKEN_HERE")

# ── Start pipeline in background ───────────────────────────────────────────
_pipe_log = open(f'{OSD}/pipeline_run.log', 'w')
_pipe = subprocess.Popen(
    ['python', '-u', 'pipeline.py', '--config', 'server', '--yes'],
    cwd=OSD,
    stdout=_pipe_log, stderr=subprocess.STDOUT,
    env={**__import__('os').environ,
         'HF_HOME': '/kaggle/working/hf_cache',
         'TRANSFORMERS_OFFLINE': '1',
         'PYTHONIOENCODING': 'utf-8'},
)
print(f"Pipeline started  (PID {_pipe.pid})")

# ── Start viz server ────────────────────────────────────────────────────────
time.sleep(3)   # let pipeline write first state update
_viz = subprocess.Popen(
    ['python', '-u', 'viz_server.py', '--host', '0.0.0.0', '--port', '5000'],
    cwd=OSD,
)
time.sleep(2)

# ── Open ngrok tunnel ───────────────────────────────────────────────────────
tunnel = ngrok.connect(5000)
print(f"\nDashboard (live, auto-refreshes every 30s):\n  {tunnel.public_url}\n")
print("Status bar shows: current step | phase | n/18 progress | GPU memory")
print("Click 'Steps' in the status bar to see all 18 steps colour-coded.")
print(f"\nTail raw log:  !tail -f {OSD}/pipeline_run.log")
```

Free ngrok gives a random `https://…ngrok-free.app` URL that lasts for the
session.  With a free account token (3 lines, 30 s to set up) the URL
persists across tunnel reconnects within the same session.

**Cell — Tail the raw pipeline log**
```python
# In a separate cell — shows every line printed by run_all.py / train_qwen3.py
import subprocess
subprocess.run(f'tail -f /kaggle/working/OSD/pipeline_run.log', shell=True)
```

**Cell — Check status without the dashboard**
```python
import subprocess
subprocess.run(
    ['python', 'pipeline.py', '--config', 'server', '--status'],
    cwd='/kaggle/working/OSD'
)
```

#### What the dashboard shows on Kaggle

The experience is identical to a local browser because the dashboard is
pure HTML/JS that calls the Flask JSON API — there is nothing Kaggle-specific
in the UI.  The black status bar at the top shows:

```
SpecDist  [train_ebe_gsm8k ●]  Phase 2 — Training  ██████████  10/18 (55.6%)  GPU 11.2/15.0 GB  ↻ 17s
```

Click **Steps** to expand all 18 pipeline steps with their current status
(`done` green / `running` yellow-pulsing / `failed` red / `pending` grey).

> **ngrok free-tier limits:** 1 tunnel active at a time, 20,000 connections/month.
> For a 12-hour Kaggle session with 30 s auto-refresh that's ~1,440 requests —
> well within the free quota.  If you hit limits, increase the refresh interval
> by editing `startAutoRefresh(30)` → `startAutoRefresh(120)` in viz_server.py.

---

### T4 ×2 (30 GB combined VRAM)

Select **GPU T4 ×2** in notebook accelerator settings to get two T4s with
30 GB total.  The OSD scripts automatically use both when a model doesn't fit
on a single GPU — `device_map="auto"` in `load_models()` handles placement.

For training, draft and teacher both default to `cuda:0`.  If VRAM is tight
even with 4-bit loading, move the frozen teacher to the second GPU:
```python
# In train_qwen3.py (or before calling it), change:
#   target_model = target_model.to("cuda")
# to:
#   target_model = target_model.to("cuda:1")
```
This frees 5 GB on `cuda:0` for draft forward/backward passes.

---

## Speed & Memory — what is automatic vs. opt-in

### What the scripts do automatically (no flags needed)

| Optimisation | Where it's applied | Expected gain |
|---|---|---|
| **SDPA (fused attention)** | `train_qwen3.py`, `run_all.py` — every `from_pretrained()` | 10-20 % faster attention; free if PyTorch ≥ 2.0 |
| **Flash Attention 2** | Same scripts — auto-selected if `flash-attn` is installed | 2-4× faster attention, ~30 % less VRAM; Ampere+ GPU only |
| **CUDA allocator tuning** (`max_split_size_mb:128`) | `train_qwen3.py`, `run_all.py`, `GBV/main.py` — set via `os.environ.setdefault` | Reduces OOM frequency from memory fragmentation on T4/P100 |
| **Pre-tokenisation** | `train_qwen3.py` training loop | Eliminates Jinja2 + tokenizer overhead per step (~5-15 ms/step) |
| **Single teacher forward pass** | `train_qwen3.py` — `output_scores=True` during `generate()` | ~40-50 % less teacher compute per training step |
| **`torch.inference_mode()`** | All eval paths | Faster than `no_grad()`; disables version tracking |
| **Model pre-loading for alpha eval** | `run_all.py` main loop | Student + teacher loaded once, shared across all datasets |
| **GBV subprocess batching** | `run_all.py` `run_be_batch()` | 72 model cold-starts → 6 (one per dataset) |

### What you opt in to with flags

| Flag | Where | When to use | Expected gain |
|---|---|---|---|
| `--compile` | `train_qwen3.py` | Linux/Colab, PyTorch ≥ 2.0 | 10-30 % per step after ~60 s one-time compile; skipped automatically on Windows |
| `--load_in_4bit` | `train_qwen3.py` only | 8B teacher on T4 (15 GB) | Reduces teacher VRAM 16 GB → 5 GB via NF4 quantisation |

> **`--compile` on Colab:** just add `--compile` to your training command — the
> Linux Colab environment fully supports `torch.compile`.  On Windows (laptop
> development), it's skipped automatically with a warning.

### Flash Attention 2 — one-time install

`flash-attn` is not in `requirements.txt` because it needs a CUDA-compiled wheel
that takes several minutes to build.  Install it once on your Colab/Kaggle instance:

```bash
# A100 / H100 (Colab Pro, server):
pip install flash-attn --no-build-isolation

# T4 (free Colab / Kaggle) — FA2 does NOT support T4 (Turing arch).
# SDPA is used automatically instead — no install needed.
```

After installation, `train_qwen3.py` and `run_all.py` detect it at startup and
print `Attn: flash_attention_2`.

### 4-bit teacher — training vs. final eval

`--load_in_4bit` is for **training only** to save VRAM on a T4.  The frozen
teacher's logits are very close to fp16 (NF4 is designed for this), so distilled
draft quality is essentially unaffected.

**For published comparison numbers**, run your final block-efficiency eval without
`--load_in_4bit`:
- `run_all.py` never uses `--load_in_4bit` — it always loads models in fp16/bf16.
- Training: use `--load_in_4bit` to fit on T4.
- Eval: run `run_all.py` normally (no flag needed).

This ensures your block efficiency numbers are apples-to-apples with literature
baselines that use full-precision models.

### Override CUDA allocator settings

The default `max_split_size_mb:128` is a conservative setting that avoids large
contiguous allocation failures.  On machines with plenty of VRAM (A100, 40 GB)
you can disable it and let PyTorch use its default strategy:

```bash
export PYTORCH_CUDA_ALLOC_CONF=""          # blank = PyTorch default
# or use the newer expandable_segments mode (PyTorch ≥ 2.1):
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
```

---

## Troubleshooting

### Why does the pipeline print "Pre-batching N BE cells → M subprocesses"?

The block-efficiency (BE) evaluation uses `GBV/main.py` as a subprocess.
Previously each (mode, K, temperature) combination launched a separate Python
process that reloaded both models — 72 cold-starts for the default grid.

The pipeline now groups all (mode, K, T) combos that share the same dataset
into **one** subprocess call.  Models load once, all combos run back-to-back,
then the process exits.  For the default grid this reduces 72 cold-starts to
6 (one per dataset).  On a 40 GB A100 with 8B models, this alone saves
~90 minutes.

You can still run `GBV/main.py` directly for a single combo:
```bash
python GBV/main.py --p_model Qwen/Qwen3-0.6B --q_model Qwen/Qwen2.5-0.5B \
    --mode gbv --K 3 --p_temp 1.0 --data OSD/data/gsm8k_30.jsonl
```

Or multiple combos in one call (same as what the pipeline does internally):
```bash
python GBV/main.py --p_model Qwen/Qwen3-0.6B --q_model Qwen/Qwen2.5-0.5B \
    --modes gbv,specinfer,traversal --Ks 1,3,5 --p_temps 0.6,1.0 \
    --data OSD/data/gsm8k_30.jsonl
```

---

### "CUDA out of memory" during eval

`run_all.py` catches `OutOfMemoryError` automatically and retries the same
step on CPU. You will see:

```
[OOM] CUDA out of memory loading models (NNN MB free). Retrying on CPU — will be ~20x slower.
```

No action needed — just slower.

### "CUDA out of memory" during training

`train_qwen3.py` does not auto-fall-back. Try:

```bash
# Reduce generated sequence length (biggest memory saver):
python train_qwen3.py --loss ebe --max_new_tokens 40 ...

# Use gradient checkpointing (saves ~30% activation memory):
# Already enabled for full SFT. For LoRA, add to train_qwen3.py:
#   draft_model.enable_input_require_grads()
#   draft_model.gradient_checkpointing_enable()
```

Or switch to CPU training (very slow, but works anywhere):
```bash
# Unset CUDA before running:
CUDA_VISIBLE_DEVICES="" python train_qwen3.py --loss forward_kl --steps 200 ...
```

### "Connection error" / "requests.exceptions.ConnectionError"

`run_all.py` and `train_qwen3.py` both default to `TRANSFORMERS_OFFLINE=1` at
startup and print:

```
[OSD] HF offline mode ON (default). Models must already be cached locally.
Set TRANSFORMERS_OFFLINE=0 before running to allow first-time downloads
(needed on a fresh Kaggle/Colab session).
```

**Models already cached — keep offline (default):**
```bash
# Nothing to do — the default is correct.
export TRANSFORMERS_OFFLINE=1   # explicit, same as default
```

**Models not cached yet (first Kaggle / Colab session):**
```bash
# Override before running any script:
export TRANSFORMERS_OFFLINE=0
export HF_HUB_OFFLINE=0
python setup_download.py --config server
```

In a notebook `%%bash` cell, prefix each command:
```bash
%%bash
TRANSFORMERS_OFFLINE=0 HF_HUB_OFFLINE=0 python setup_download.py --config server
```

**`pipeline.py` still injects TRANSFORMERS_OFFLINE=1 automatically** — if you
run scripts directly (not via the pipeline), set the env vars yourself as shown.

### Pipeline step shows "running" after a crash

**This is handled automatically.** On startup `pipeline.py` scans the state
file and resets any `"running"` step to `"pending"`, prints a `[RECOVER]`
message, then continues normally.  Eval steps always pass `--skip_existing`
so no duplicate DB rows are inserted on the re-run.

You will see at startup:
```
[RECOVER] 1 step(s) were left in 'running' state (pipeline was interrupted).
  Reset to 'pending': eval_baseline_gsm8k
  Eval steps pass --skip_existing, so no duplicate DB entries will be created on re-run.
```

**You do not need to do anything** — just re-run the pipeline command.

If you want to force a specific step to re-run from scratch (discarding its
partial DB results), reset it manually:

```python
import json
with open("pipeline_state_laptop.json") as f: state = json.load(f)
state["steps"]["eval_baseline_gsm8k"]["status"] = "pending"
with open("pipeline_state_laptop.json", "w") as f: json.dump(state, f, indent=2)
```

Or reset everything and restart from scratch (keeps checkpoints on disk):

```bash
python pipeline.py --config laptop --restart --yes
```

### Check what's done so far

```bash
python pipeline.py --config laptop --status
```

---

## File Structure (what gets transferred)

```
OSD/                        ← this directory
  pipeline.py               ← crash-safe master runner (phases 0–4)
  run_all.py                ← eval orchestrator (alpha + BE × datasets × modes)
  train_qwen3.py            ← KL / EBE LoRA training
  viz_server.py             ← Flask results dashboard  →  http://localhost:5000
  results_db.py             ← SQLite schema + insert/query helpers
  setup_download.py         ← one-time model + dataset download
  fetch_datasets.py         ← dataset fetchers
  results.db                ← auto-created; stores all eval results
  requirements.txt
  data/
    gsm8k_30.jsonl
    gsm8k_train.jsonl
    humaneval.jsonl
    math500_30.jsonl
    mtbench_80.jsonl
    alpaca_30.jsonl
    alpaca_train.jsonl
  checkpoints/              ← LoRA adapters (created by training steps)
    kl200/                  ← pre-trained (from earlier experiments)
    kl200_merged/
    ebe200/
    ebe200_merged/
    ebe_lr1e-5/             ← LR-sweep variants (Phase 0 merges these)
    ebe_lr1e-5_merged/
    ...
    kl1000-gsm8k/           ← created by Phase 2 training
    kl1000-gsm8k_merged/
    ebe1000-gsm8k/
    ebe1000-gsm8k_merged/

~/.cache/huggingface/hub/   ← model weights (separate from OSD/)
  models--Qwen--Qwen3-0.6B/
  models--Qwen--Qwen3-8B/
  models--Qwen--Qwen2.5-0.5B/
```
