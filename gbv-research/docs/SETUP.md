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

| Config | Models | VRAM / Device | Notes |
|--------|--------|--------------|-------|
| `laptop_gpt2` | distilgpt2 → gpt2-medium | **CPU only** | No GPU needed. Proves convergence cheaply. |
| `laptop` | Qwen2.5-0.5B → Qwen3-0.6B | 4 GB VRAM | RTX 500 / any modern laptop GPU |
| `laptop_llama` | Llama-3.2-1B → 3B (4-bit target) | 4 GB VRAM | Research-grade pair (3× size gap vs 1.2× for `laptop`) |
| `colab`  | Qwen3-0.6B → Qwen3-8B (4-bit) | 9 GB | Free Colab T4 (15 GB) — teacher in 4-bit NF4 |
| `server` | Qwen3-0.6B → Qwen3-8B (bf16) | 24 GB | A10G / A100 / 3090 — full precision |

**Which laptop config to use?**
- `laptop_gpt2` — out of GPU credits, just need convergence evidence, or testing on CPU server
- `laptop` — Qwen pair, standard code exerciser; teacher barely bigger than student → weak signal
- `laptop_llama` — best laptop option for research claims: 1B → 3B-NF4 (3× gap), real distillation signal; needs HF gated access (`huggingface-cli login`)

**Colab / Kaggle T4 note**: Qwen3-8B in bfloat16 = ~16 GB → OOM on T4 (15 GB).
The `colab` and `kaggle` configs automatically load the frozen teacher in
**4-bit NF4** via `bitsandbytes` (~4.5 GB), bringing total VRAM to ~7–9 GB.
`transformers` hard-requires `bitsandbytes>=0.46.1` for 4-bit loading:

```bash
pip install "bitsandbytes>=0.46.1"
```

`deploy_utils.install_deps()` (Colab/Kaggle notebooks) handles this automatically
with `pip install -U bitsandbytes>=0.46.1` so the base-image version is always
upgraded regardless of what the platform pre-installs.

---

## 2. Download datasets

```bash
# Run from gbv-research/
python core/datasets/downloader.py
```

This downloads `gsm8k_train.jsonl` (7,473 training prompts) to `core/datasets/raw/`.

**Small eval sets are already committed to the repo** — no download needed:

| File | Size | Used for |
|---|---|---|
| `gsm8k_5/10/30.jsonl` | 3–8 KB | T4 Phase 3 GSM8K eval |
| `alpaca_30.jsonl` | 2 KB | T4/A100 Phase 4 generalization |
| `math500_30.jsonl` | 3 KB | T4/A100 Phase 4 generalization |
| `humaneval.jsonl` | 3 KB | T4/A100 Phase 4 generalization |
| `mtbench_80.jsonl` | 7 KB | T4/A100 Phase 4 generalization |
| `diverse50.jsonl` | 4 KB | Laptop smoke tests |

**Large A100 eval sets** (`gsm8k_1319.jsonl`, `alpaca_100.jsonl`, `math500_100.jsonl`) are not committed. They are downloaded automatically by `evaluate.py` on the first A100 eval run — no manual step needed.

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

**How it works**: at pipeline startup, `experiment.py` reads this file and sets
`WANDB_API_KEY`, `WANDB_ENTITY`, and `WANDB_PROJECT` as environment variables.
Every subprocess (training runs, eval runs) inherits them automatically.
If the file is absent the pipeline falls back to whatever `wandb login`
last stored in `~/.netrc` (or `~/_netrc` on Windows) — so CI/Colab runs
work without any file.

### Team credentials

| Researcher   | W&B username    | Notes                               |
|--------------|-----------------|-------------------------------------|
| Krishnan R   | `rkrishnaiyer`  | Uses `~/_netrc` cached login        |
| Mukund R     | `rmukund16`     | Needs `wandb_config.json` on laptop |
| Rahul Thomas | TBD             | Account not yet configured          |

---

## 4. Model weights — first-time download

Model weights download automatically the first time the pipeline runs.
**No manual step required.** `experiment.py` detects uncached models and
downloads them before starting subprocesses, then switches to offline mode
for all subsequent runs.

For gated models (LLaMA 3.2), accept the licence on HuggingFace first:
```bash
huggingface-cli login   # one-time, stores token in ~/.cache/huggingface/
```
Then run the pipeline normally — the download happens automatically.

---

## 5. Smoke test (verify everything works)

Run this on any new machine before the overnight full run.

### Option A — Full Qwen smoke (~30-40 min, needs GPU)
```bash
python orchestration/experiment.py --config laptop --smoke --yes
```
Exercises every loss function and every verifier at reduced scale.

### Option B — GPT-2 GPU smoke (~15-20 min, laptop GPU)
```bash
python orchestration/experiment.py --config laptop_gpt2 --smoke --yes
```
Runs distilgpt2 → gpt2-medium on the laptop GPU. Fast, validates the GPT-2
code path. ebe/online losses are excluded (see `docs/ISSUES.md`).

### Option B2 — GPT-2 CPU-path smoke (~30-45 min, forces CPU on any machine)
```bash
python orchestration/experiment.py --config server_gpt2 --smoke --yes --device cpu
```
**The right smoke before running the ATS/AIP CPU server.** Forces CPU even
when a GPU is present (`--device cpu`), so you exercise the **exact same code
path** that runs on the CPU-only ATS server. Run this on your laptop first.

Config names map to hardware+device, not platforms:
| Config | device | hw_tier | Intended for |
|--------|--------|---------|-------------|
| `laptop_gpt2` | cuda | laptop | Laptop GPU — fast GPT-2 development |
| `server_gpt2` | cpu | cpu | ATS/AIP CPU server (and local CPU smoke) |

`--device` can override any config: `--device cpu` forces CPU, `--device cuda`
forces GPU, `--device auto` (default) detects automatically.

### Option C — LLaMA smoke (~20-30 min, GPU, research-grade pair)
```bash
python orchestration/experiment.py --config laptop_llama --smoke --yes
```
Requires `huggingface-cli login` first. Gives research-meaningful signal
(1B → 3B-NF4 = 3× size gap).

### Option D — Single-family quick check (fastest, any family)
```bash
# No experiment.py overhead — just trainer.py, 5 steps
python scripts/smoke_family.py --family gpt2            # CPU, ~2 min
python scripts/smoke_family.py --family llama           # GPU, ~5 min
python scripts/smoke_family.py --family gpt2 --all_losses   # all losses, ~15 min
```

What it runs:
- **Phase 1**: baseline eval (untrained draft, all 6 verifier modes)
- **Phase 2**: 10-step training for every loss in `laptop.yaml` (flat losses kl, rev_kl,
  jsd, l1, ebe, ebe_single + the `*_tree` losses; online losses are excluded on laptop)
- **Phase 3**: eval every trained model on gsm8k (all 6 verifiers; tree losses use the
  paired tree verifier only)
- **Phase 4**: skipped on laptop (multi-dataset eval runs on T4/A100 only)

For the exact loss list, step counts, and eval params see the canonical
[`orchestration/configs/laptop.yaml`](../orchestration/configs/laptop.yaml).

Expected time breakdown:
- Phase 1 eval: ~5 min
- Phase 2 per-loss: ~3-5 min each × 6 = ~25 min
- Phase 3 per-eval: ~3-4 min each × 6 = ~20 min

Pass criteria: no NaN/OOM errors, all 22 steps (Phase 1 + Phase 2 + Phase 3) green.

### Watch progress

```bash
# Live log
Get-Content db/logs/pipeline_output.log -Wait -Tail 40   # PowerShell
# or: tail -f db/logs/pipeline_output.log                # bash

# Status snapshot (safe to run while pipeline is running — does NOT kill it)
python orchestration/experiment.py --config laptop --smoke --status

# Dashboard
python dashboard/training_dashboard.py   # http://127.0.0.1:5000
```

---

## 6. Unit tests (run before smoke — < 60 s)

**Required on every new machine before first commit.**  
`pytest` must be installed (it is in `requirements.txt`) or the pre-commit hook
silently skips and commits go through untested.

```bash
# Verify pytest is installed
pip install pytest            # or: pip install -r requirements.txt

# From gbv-research/
python -m pytest tests/unit/ -q     # full suite (~60 s on CPU, no GPU needed)
```

The suite exercises every loss function, every verifier, all cache operations,
pipeline step generation, data loading, trainer correctness, and all four model
families — all on CPU with no model downloads.  **216 tests** (182 core +
34 model-family), expected output: `216 passed`.

The model-family tests (`tests/test_model_families.py`) cover all four registered
families (gpt2, llama, qwen, gemma): temperature recovery, LoRA module names,
log-prob clamping, chat template, and registry integrity.

**Pre-commit hook** — the hook script is committed at `hooks/pre-commit` and
must be installed once per machine:

```bash
# Linux / macOS / Git Bash (from repo root):
cp gbv-research/hooks/pre-commit .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit

# Windows (Git for Windows runs Python hooks natively — no chmod):
copy gbv-research\hooks\pre-commit .git\hooks\pre-commit
```

Every `git commit` then automatically runs `pytest tests/unit/` first.
The commit is **blocked** if any test fails.

⚠️ **The hook silently skips if pytest is not installed** — always verify
`python -m pytest --version` works before making commits on a new machine.
Bypass only in genuine emergencies: `git commit --no-verify`.

**Adding a new test**: drop `test_*.py` in `tests/unit/` — picked up automatically
by the hook, `run_unit_tests.py`, and `smoke.py --unit-only`. No registration needed.

Test coverage:

| Module | File | What's tested |
|---|---|---|
| Loss functions | `test_losses.py` | forward_kl, reverse_kl (NaN guard), jsd, l1, ebe (block split, gradient) |
| Node OTLP solvers | `test_node_otlp.py` | naive, nss, specinfer, spectr, max — output range, acceptance rates |
| Verifiers | `test_verifiers.py` | bv, gbv, traversal, specinfer, naive — depth/token range, tree structure |
| KV-cache ops | `test_cache_ops.py` | slice_cache, expand_cache — shapes, content, round-trips |
| Pipeline steps | `test_pipeline_steps.py` | step IDs, --steps counts, --load_in_4bit propagation, verifier modes |
| Data loading | `test_data_loading.py` | JSONL parsing, missing keys, fallback keys, empty files |
| Trainer logit path | `test_trainer_logit_equivalence.py` | causal forward-pass == autoregressive scores, temperature recovery, slice indexing, zero-gen guard, gradient identity |

---

## 7. Full pipeline run (after smoke passes)

```bash
python orchestration/experiment.py --config laptop --yes
```

What changes vs smoke (see [`orchestration/configs/laptop.yaml`](../orchestration/configs/laptop.yaml)
for the canonical values):
- 100 steps/loss (instead of 10) — exercises all checkpoint/val/PPL paths; **not** paper-scale
- n=5 eval prompts (instead of 3); K and temperature stay single-valued (K=3, temp=0.6 —
  laptop is a code exerciser, not a sweep)
- Phase 4 multi-dataset eval is **not** run on laptop (T4/A100 only)

Expected time: ~2-3 hours on laptop (RTX 500). The laptop run is a code-path exerciser
with a 0.6B teacher — its numbers are meaningless; real trends come from T4/A100.

---

## 8. Dashboard

The dashboard reads results directly from `db/results.db` — no W&B dependency.

```bash
# Run from gbv-research/
python dashboard/training_dashboard.py          # http://127.0.0.1:5000
python dashboard/training_dashboard.py --port 8080   # custom port
```

The UI is organized into tabs: **Training**, **Results**, **Robustness**, **Analysis**, **Data**.
What it shows:
- Block efficiency table / heatmap (rows = trained models, cols = verifier modes) — **Results** tab
- Training loss curves (**Training** tab) and acceptance-rate / verifier robustness charts (**Robustness** tab)
- **Per-prompt acceptance-rate breakdown** — the **α by Category (diverse50)** heatmap on the
  **Analysis** tab, built from the `/api/per_prompt/<run_id>` endpoint (needs the diverse50
  Phase-4 alpha runs, so it's empty on laptop where Phase 4 is skipped)
- Live pipeline log tail (polls `db/logs/pipeline_output.log`)
- Pipeline step progress bar (reads `orchestration/pipeline_state_*.json`)

---

## 9. Clean restart

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

## 10. Ephemeral compute — Colab, Kaggle, Modal (pointers)

Colab, Kaggle, and Modal all wipe local `/content` or `/tmp` disk when the
session ends, so each platform persists checkpoints to Drive or a Modal Volume.
**This guide stays focused on local setup** — the full, authoritative workflow
for each platform lives in its own doc. Use this routing table and follow the link:

| Platform | Use it for | Persistence | Canonical guide |
|---|---|---|---|
| **Colab (free T4)** | one-click smoke / single-loss runs | Google Drive | [`deploy/colab_quickstart.ipynb`](../deploy/colab_quickstart.ipynb) — open and **Runtime → Run all** |
| **Kaggle (free T4)** | recommended free T4 (29 GB RAM → 8B NF4 teacher) | Kaggle datasets / auto-backup | [`docs/KAGGLE.md`](./KAGGLE.md) (uses `deploy/kaggle.ipynb`) |
| **Colab Pro / A100 (24 GB+)** | bf16 8B teacher, 3-tier profile sweep | Drive | [`deploy/PROGRESSION.md`](../deploy/PROGRESSION.md) + [`orchestration/configs/profiles/README.md`](../orchestration/configs/profiles/README.md) |
| **Modal (on-demand T4)** | bursty per-second-billed Phase-1 exploration | persistent Modal Volume | [`docs/MODAL.md`](./MODAL.md) — see 9c below |

Common to every ephemeral session:
- Pass `--yes` to skip interactive prompts.
- Point `--storage_root` (or `--ckpt_root`) at Drive / the Modal Volume so
  `done_check` files survive a session death and re-running auto-skips completed steps.
- W&B logs sync to the cloud in real time, so they survive even if the session dies mid-run.

---

### 9c. Modal — on-demand T4, per-second billing

Modal runs the **same** `experiment.py` CLI that Kaggle uses via the pure launcher
`deploy/modal_app.py`. Checkpoints, `results.db`, logs, and the HF model cache all
live in a persistent Modal Volume (`specdist-vol`), so re-running the same command
resumes automatically. The launcher exposes a single `modal run` entrypoint with a
few flags — there are **no** `::train_kl` / `::download_models`-style subcommands.

```bash
pip install modal
modal setup                                  # one-time browser auth

# Always smoke first (a few minutes of T4 time):
modal run deploy/modal_app.py --smoke

# Phase-1 flat baseline (forward KL):
modal run deploy/modal_app.py --losses kl

# Tree-loss variant:
modal run deploy/modal_app.py --config profiles/tree_variant_week --losses kl_tree
```

| Flag | Default | Meaning |
|---|---|---|
| `--config` | `profiles/train_one_loss` | YAML profile under `orchestration/configs/` |
| `--smoke` | off | quick crash-check (a few prompts, tiny tokens) |
| `--losses` | (all) | comma-separated loss subset, e.g. `kl,rev_kl,jsd` |
| `--git-ref` | `main` | branch / tag / commit to clone at runtime |

> **Full Modal guide** (secret setup, cost/budget, GPU switching, resume details):
> [`docs/MODAL.md`](./MODAL.md). A100 confirmation runs go to the IITH cluster, not
> Modal — see `docs/COMPUTE.md`.

---

## 11. MLOps — hyperparameter overrides, loss filtering, and W&B sweeps

### 10a. No-code experiment variation

All training hyperparameters are exposed as CLI flags so researchers can run
experiments without editing any source file.

**Run only specific losses** (skip the rest):
```bash
# Train and eval only KL and EBE — skip rev_kl, jsd, l1, online
python orchestration/experiment.py --config laptop --smoke --losses kl,ebe
```

**Override steps per loss** (e.g., quick 200-step ablation):
```bash
python orchestration/experiment.py --config laptop --train_steps 200 --losses kl,ebe
```

**Override learning rate** (single run or sweep baseline):
```bash
python orchestration/experiment.py --config laptop --lr 1e-4 --losses kl
```

**Override LoRA rank** (compare r=8 vs r=16 on laptop):
```bash
python orchestration/experiment.py --config laptop --lora_r 16 --losses kl,ebe
```

**Standalone `trainer.py` with all sweep-friendly args**:
```bash
python algorithms/distillspec_gbv/trainer.py \
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
    --dataset core/datasets/raw/gsm8k_train.jsonl \
    --output db/checkpoints/ebe-ablation-v1
```

### 10b. YAML config files — per-environment defaults

Each compute environment has a YAML in `orchestration/configs/`:

| File | Environment | Key settings |
|------|-------------|-------------|
| `laptop.yaml` | RTX 500, 6 GB | lr=3e-5, lora_r=4, steps=100 |
| `server.yaml` | A100 40/80 GB | lr=3e-5, lora_r=16, steps=5000 |
| `colab.yaml`  | T4 15 GB | lr=3e-5, lora_r=8, steps=500, 4-bit teacher |

CLI args (`--lr`, `--lora_r`, `--teacher_temp`) override YAML values, which
override the hardcoded defaults. Priority: `CLI > YAML > code default`.

To add a new environment (e.g., Kaggle P100):
1. Copy `orchestration/configs/laptop.yaml` → `orchestration/configs/kaggle.yaml`
2. Adjust `hardware`, `training`, and `checkpointing` sections
3. Add `"kaggle": {"draft": ..., "target": ...}` to `CONFIGS` dict in `experiment.py`
4. Run: `python orchestration/experiment.py --config kaggle --ckpt_root /kaggle/working/ckpts`

### 10c. W&B hyperparameter sweeps

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

### 10d. Reading sweep results

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
python orchestration/experiment.py --config server \
    --lr 4.2e-5 --lora_r 16 \
    --experiment_tag "best_sweep_v1"
```
