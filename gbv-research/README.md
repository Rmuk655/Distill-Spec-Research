# GBV Research — Speculative Decoding via Distillation

Train a small draft model to better match a large target model's token distribution,
improving acceptance rates in speculative decoding.
Novel contribution: **EBE loss** — directly optimises block efficiency instead of KL divergence.

## Directory layout

**Repository root** (`2026 summer/`) contains three top-level directories that
`orchestration/pipeline.py` stitches together at runtime:

```
2026 summer/                    <- git repo root
|
+-- OSD/                        git submodule → LiuXiaoxuanPKU/OSD (unmodified upstream)
|   +-- distill/specInfer/      Alpha-eval Generator used by online_serve.py
|
+-- GBV/                        Reference copy of Rahul Thomas's GBV codebase
|                               (anonymous.4open.science/r/GBV-BED8 — original, unmodified)
|                               Production verifier is now distillspec_gbv/verifiers/runner.py;
|                               GBV/ will be restored to exact reference once Phase 3 confirms.
|
+-- gbv-research/               Research project (this directory)
    |
    +-- algorithms/             Training scripts and algorithm implementations
    |   +-- training_scaffold.py    Shared training utilities (HW setup, model load, LoRA,
    |   |                           checkpoint, WandB, results_db) — import instead of copy
    |   +-- train_qwen3.py          Offline distillation training script (all 5 losses)
    |   +-- online_serve.py         Online speculative distillation training (OSD)
    |   +-- distillspec_gbv/        Novel EBE training code + production verifiers
    |       +-- losses/             forward_kl, ebe, reverse_kl, jsd, l1 (class-based)
    |       +-- trainer.py          Training loop (model-family-agnostic replacement for train_qwen3)
    |       +-- verifiers/          Production verifiers — runner.py is the Phase 2 eval entry point
    |
    +-- core/
    |   +-- datasets/raw/       JSONL eval + training sets
    |   |   +-- gsm8k_train.jsonl   7,473 training prompts (gitignored -- large)
    |   |   +-- gsm8k_30.jsonl      30-prompt fixed eval set (tracked)
    |   |   +-- gsm8k_5.jsonl       5-prompt smoke eval set (tracked)
    |   +-- model_families/     Qwen/Gemma model-specific tokenizer helpers
    |
    +-- orchestration/          Pipeline coordination
    |   +-- pipeline.py         Crash-safe orchestrator (Phase 1->4 + optional EAGLE)
    |   +-- run_all.py          Eval subprocess -- calls distillspec_gbv/verifiers/runner.py
    |   |                       (Phase 2 complete; GBV/main.py kept as fallback until Phase 3)
    |   +-- clean_restart.py    Wipe outputs + reset state + relaunch
    |   +-- pipeline_state_laptop.json        Full-run state (tracked in git)
    |   +-- pipeline_state_laptop_smoke.json  Smoke state (separate, never blocks full run)
    |   +-- wandb_config.json.example         Copy -> wandb_config.json (gitignored)
    |
    +-- db/                     All generated outputs (entirely gitignored)
    |   +-- checkpoints/        LoRA adapters + merged models
    |   +-- results.db          SQLite experiment database
    |   +-- wandb/              W&B local run logs
    |   +-- logs/               pipeline_output.log, be_progress.log
    |
    +-- dashboard/
    |   +-- training_dashboard.py   Live training dashboard (Flask, port 5000)
    |
    +-- docs/
    |   +-- SETUP.md                First-time setup guide (WandB, smoke test, dashboard)
    |   +-- PROJECT_CONTEXT.md      Research context, decisions log, baselines
    |   +-- ADDING_A_LOSS.md        How to add a new loss objective
    |   +-- ADDING_AN_ALGORITHM.md  How to add a new verifier algorithm
    |   +-- ADDING_A_MODEL_FAMILY.md
    |
    +-- references/             External code -- read-only, never imported by pipeline
    |   +-- adaspec/            AdaSpec (Hu et al., 2024) — ablation comparison
    |   +-- legacy-osd-paper/   OSD paper plot data — read-only reference
    |
    +-- tests/
        +-- test_core.py        Unit tests: EBE loss properties, verifier invariants
```

## Quick start

See **[docs/SETUP.md](docs/SETUP.md)** for the full walkthrough including WandB setup.

### 1. Install + download data

```bash
pip install -r requirements.txt
python ../OSD/fetch_datasets.py --datasets gsm8k   # OSD/ is a sibling of gbv-research/
```

### 2. Set up WandB credentials (one-time per machine)

```bash
cp orchestration/wandb_config.json.example orchestration/wandb_config.json
# Edit orchestration/wandb_config.json with your api_key, entity, project
```

### 3. Smoke test (~35 min — verify all losses + verifiers before overnight run)

```bash
python orchestration/pipeline.py --config laptop --smoke --yes
```

### 4. Full pipeline (~6-8 hrs on laptop)

```bash
python orchestration/pipeline.py --config laptop --yes
```

### 5. Dashboard

```bash
python dashboard/training_dashboard.py    # http://127.0.0.1:5000
```

## Pipeline structure

The pipeline runs 4 phases in sequence:

| Phase | Steps | Smoke | Full |
|-------|-------|-------|------|
| **1 — Baseline** | `eval_baseline_gsm8k`: unmodified draft, all 6 verifiers | 5 prompts | 10 prompts |
| **2 — Training** | train + merge × 7 losses: kl, ebe, rev_kl, jsd, l1, online, online_ebe | 50 steps/loss | 1000 steps/loss |
| **3 — GSM8K Eval** | eval every trained model, all 6 verifier modes | 5 prompts, K=3, temp=0.6 | 10 prompts, K=3+5, temps=0.6+1.0 |
| **4 — Multi-Dataset** | eval on humaneval, math500, mtbench, alpaca | skipped | runs |

Smoke uses a **separate state file** (`pipeline_state_laptop_smoke.json`) so smoke
"done" marks never prevent the real pipeline from re-running Phase 2 training.

### Useful commands while running

```bash
# Status check (safe — does NOT kill the running pipeline)
python orchestration/pipeline.py --config laptop --smoke --status

# Live log
Get-Content db/logs/pipeline_output.log -Wait -Tail 40   # PowerShell
tail -f db/logs/pipeline_output.log                      # bash/WSL
```

## Loss functions

| Loss | Description | Status |
|------|-------------|--------|
| `forward_kl` | KL(target ∥ student). DistillSpec baseline. Mode-covering. | Baseline |
| `ebe` | Expected Block Efficiency — directly optimises acceptance product. Novel. | Novel |
| `reverse_kl` | KL(student ∥ target). Mode-seeking ablation. | Ablation |
| `jsd` | Jensen-Shannon divergence. Symmetric ablation. | Ablation |
| `l1` | L1 on probability distributions. Total-variation ablation. | Ablation |
| `online` | Online speculative distillation (forward_kl on live SD outputs). | Ablation |
| `online_ebe` | Online distillation with EBE loss — combines online adaptation with block-efficiency objective. | Novel ablation |

## Verifier modes

All 6 modes run in both smoke and full pipeline:

| Mode | Description |
|------|-------------|
| `alpha` | Token-level acceptance rate (chain SD floor) |
| `bv` | Block Verification — accept/reject whole blocks |
| `gbv` | Generalised BV — optimal transport over block prefixes (novel, > BV) |
| `traversal` | Longest surviving path — empirically best single-path verifier |
| `specinfer` | SpecInfer published baseline — multi-path joint probability |
| `naive` | Naive chain SD — same as alpha but explicit implementation |

## Baseline numbers (Qwen2.5-0.5B → Qwen3-0.6B, gsm8k, untrained)

| Verifier | K=3 BE | K=5 BE |
|----------|--------|--------|
| specinfer | 2.520 | 2.512 |
| gbv | 2.797 | 2.786 |
| traversal | 2.954 | 3.107 |

Any trained model must beat `specinfer K=3 = 2.520`. Regression → investigate immediately.

## Current results

| Model | Loss | GSM8K BE (specinfer K=3) | Notes |
|-------|------|--------------------------|-------|
| Baseline | — | 2.520 | Pre-training reference |
| forward_kl 1000 steps | forward_kl | TBD | DistillSpec baseline |
| ebe 1000 steps | ebe | TBD | Novel EBE objective |

Full results in `db/results.db`; visualise at http://127.0.0.1:5000.

## Team

- **Krishnan R** (IIT Hyderabad) — Research Engineer: implementation, experiments, benchmarking
- **Rahul Thomas** (PhD Student, Columbia University) — Research Lead: direction, novelty, EBE math, publication
- **Target**: ICLR mid-September 2026
