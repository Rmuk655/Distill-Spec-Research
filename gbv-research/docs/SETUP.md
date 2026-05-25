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
- **Laptop config** (Qwen2.5-0.5B → Qwen3-0.6B): 4 GB VRAM minimum
- **Server config** (Qwen3-0.6B → Qwen3-8B): 24 GB VRAM (A10G / A100 / 3090)

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

## 5. Full pipeline run (after smoke passes)

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

## 8. Colab / Kaggle setup

On ephemeral compute (no persistent disk):

1. Clone the repo at session start
2. `pip install -r requirements.txt`
3. Set `WANDB_API_KEY` as a Colab secret or `os.environ` line (no file needed)
4. Datasets are in the repo (`core/datasets/raw/*.jsonl`) — no re-download needed for eval sets
5. For training: download `gsm8k_train.jsonl` via `fetch_datasets.py`

Checkpoints and WandB logs are lost when the session dies — that's expected.
The pipeline state file (`pipeline_state_server.json`) is also lost.
Always run `--yes` on Colab to skip interactive prompts.

```bash
WANDB_API_KEY="wandb_v1_..." python orchestration/pipeline.py --config server --yes
```
