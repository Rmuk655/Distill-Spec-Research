# GBV Research — Speculative Decoding via Distillation

Train a small draft model to better match a large target model's token distribution,
improving acceptance rates in speculative decoding.
Novel contribution: **EBE loss** — directly optimises block efficiency instead of KL divergence.

---

## Docs index

| Document | Contents |
|----------|----------|
| **[docs/SETUP.md](docs/SETUP.md)** | First-time setup, WandB config, smoke test, environments (Colab/Modal/server) |
| **[docs/GUIDE.md](docs/GUIDE.md)** | Full experiment guide: pipeline phases, loss descriptions, verifier modes, W&B sweeps, health checks |
| **[docs/DESIGN.md](docs/DESIGN.md)** | Architecture decisions: loss math, training loop, verifier design, future directions |
| **[docs/PROJECT_CONTEXT.md](docs/PROJECT_CONTEXT.md)** | Codebase map, research decisions log, baselines, what's borrowed vs. novel |
| **[docs/ADDING_A_LOSS.md](docs/ADDING_A_LOSS.md)** | How to add a new distillation loss |
| **[docs/ADDING_AN_ALGORITHM.md](docs/ADDING_AN_ALGORITHM.md)** | How to add a new verifier algorithm |
| **[docs/ADDING_A_MODEL_FAMILY.md](docs/ADDING_A_MODEL_FAMILY.md)** | How to add Gemma/LLaMA/Mistral support |
| **[docs/DELIVERABLES.md](docs/DELIVERABLES.md)** | Research progress record |

---

## Quick start

**Full walkthrough → [docs/SETUP.md](docs/SETUP.md)**

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download datasets
python core/datasets/downloader.py --datasets gsm8k

# 3. WandB credentials (one-time)
cp orchestration/wandb_config.json.example orchestration/wandb_config.json
# Edit with your api_key, entity, project

# 4. Smoke first (~5 min, any GPU — run before every real session):
python orchestration/experiment.py --config profiles/smoke --smoke --yes --losses forward_kl

# 5. Phase-1 tiered run (train + val_loss + light BE sanity, ~1.5–2 h on P100):
python orchestration/experiment.py --config profiles/train_one_loss --yes \
  --losses forward_kl --skip_existing --experiment_tag p1_forward_kl
```

**Tree losses `bv_tree` / `gbv_tree`:** default YAML `lr` (3e-5) is tuned for KL-style losses; the BV block-acceptance integral amplifies gradients — use `--lr 1e-5` when training these two (see `deploy/A100_SETUP.md` or `docs/GUIDE.md` § `bv_tree`).

---

## Repository layout

```
2026 summer/                    <- git repo root
|
+-- OSD/                        Original OSD codebase (LiuXiaoxuanPKU/OSD, unmodified)
|   +-- distill/specInfer/      Alpha-eval Generator used by online_serve.py
|
+-- GBV/                        Reference copy of Rahul Thomas's GBV codebase
|                               (anonymous.4open.science/r/GBV-BED8, unmodified)
|
+-- gbv-research/               Research project (this directory)
    |
    +-- algorithms/
    |   +-- training_scaffold.py    Shared HW setup, model load, LoRA, checkpoint, WandB utilities
    |   +-- online_serve.py         Online speculative distillation trainer
    |   +-- eagle_bench.py          EAGLE baseline runner
    |   +-- distillspec_gbv/
    |       +-- losses/             forward_kl, ebe, reverse_kl, jsd, l1 (registry-based)
    |       +-- trainer.py          Offline distillation trainer (all 5 offline losses)
    |       +-- verifiers/          Production verifiers (runner.py = eval entry point)
    |
    +-- core/
    |   +-- datasets/raw/           JSONL eval + training sets
    |   +-- model_families/         Qwen/Gemma tokenizer helpers
    |
    +-- orchestration/
    |   +-- experiment.py           Crash-safe orchestrator: Phases 1–4
    |   +-- evaluate.py             Stand-alone eval: single model × all verifier modes
    |   +-- run_sweep.py            W&B hyperparameter sweep launcher
    |   +-- configs/sweep.yaml      Sweep parameter grid
    |   +-- clean_restart.py        Wipe outputs + reset state
    |
    +-- db/                         All generated outputs (gitignored)
    |   +-- checkpoints/            LoRA adapters + merged models
    |   +-- results.db              SQLite experiment database
    |   +-- logs/                   pipeline_output.log, be_progress.log
    |
    +-- dashboard/
    |   +-- training_dashboard.py   Live training dashboard (Flask, port 5000)
    |
    +-- paper/
    |   +-- analyze_results.py      Generate paper statistics + significance tests from results.db
    |
    +-- deploy/
    |   +-- modal_app.py            Modal cloud launcher
    |   +-- colab_quickstart.ipynb  Colab / Kaggle quickstart notebook
    |   +-- eval_ebe_sweep.sh       EBE sweep eval script
    |
    +-- scripts/
    |   +-- setup_download.py       One-time model + dataset download
    |   +-- setup_env.py            One-time venv + deps installation
    |   +-- migrate_outputs.py      Post-pipeline OSD → db/ output migration
    |
    +-- docs/                       See docs index above
    +-- tests/
    |   +-- unit/                   CPU-only unit tests (~60 s)
    |   +-- smoke.py                Pre-commit: unit tests + pipeline smoke
    +-- references/                 External baselines (read-only, never imported)
        +-- adaspec/
        +-- legacy-osd-paper/
```

---

## Current results

Results are in `db/results.db`. Visualise live at `http://127.0.0.1:5000` after running:

```bash
python dashboard/training_dashboard.py
```

For paper statistics and significance tests:

```bash
python paper/analyze_results.py --out report.md
```

---

## Team

- **Krishnan R** (IIT Hyderabad) — Research Engineer: implementation, experiments, benchmarking
- **Rahul Thomas** (PhD Student, Columbia University) — Research Lead: direction, novelty, EBE math, publication
- **Target**: ICLR mid-September 2026
