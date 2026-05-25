# SpecDist — DistillSpec + Tree-based Verification

**Direction #1:** Knowledge distillation (EBE loss) combined with GBV tree-based speculative decoding verification.

Built on top of [OSD (Online Speculative Decoding)](https://arxiv.org/pdf/2310.07177.pdf) and [GBV](https://github.com/...).

---

## What This Does

Trains a small draft model (0.5B) to match the output distribution of a larger target model (0.6B or 8B) using:
- **Forward KL loss** (DistillSpec baseline)
- **Expected Block Efficiency (EBE) loss** (novel contribution) — directly optimises token acceptance probability

Evaluates block efficiency and acceptance rate with tree-based verification modes (specinfer, GBV, traversal) across multiple datasets.

All results saved to SQLite with timestamped run IDs. Interactive web dashboard for analysis.

---

## Quick Start

### 1. Setup (one command)

```bash
# Clone repos side-by-side
git clone <OSD-repo> OSD
git clone <GBV-repo> GBV

# Run setup (creates venv, installs everything, downloads datasets, inits DB)
python OSD/setup_env.py

# On server — specify CUDA version
python OSD/setup_env.py --cuda 12.1
```

### 2. Run Everything

```bash
# Activate venv
# Windows:
venv\Scripts\activate
# Linux:
source venv/bin/activate

# Laptop (Qwen 0.6B models, ~4-6 hours full run)
python OSD/run_all.py \
    --student Qwen/Qwen2.5-0.5B \
    --teacher Qwen/Qwen3-0.6B

# Server (8B target, much faster per prompt)
python OSD/run_all.py \
    --student Qwen/Qwen2.5-0.5B \
    --teacher Qwen/Qwen3-8B \
    --n 50
```

That's it. Two arguments. Runs all datasets, all verifier modes, all K values, saves to DB.

### 3. Dashboard

```bash
python OSD/viz_server.py
# Open http://localhost:5000/
```

---

## Detailed Usage

### run_all.py — master evaluation runner

```
python OSD/run_all.py --student <model> --teacher <model> [options]

Required:
  --student   HuggingFace ID or local path to student (draft) model
  --teacher   HuggingFace ID or local path to teacher (target) model

Optional:
  --student_label  Short name for student (auto-inferred from path)
  --datasets       Comma-separated: diverse50,gsm8k,humaneval,math500,mtbench,alpaca
                   (default: all six)
  --modes          Comma-separated: alpha,specinfer,gbv,traversal
                   (default: all four)
  --K              Comma-separated K values: 1,3,5 (default: 1,3,5)
  --temperature    Comma-separated: 0.6,1.0 (default: 1.0)
  --n              Prompts per dataset (default: 30; humaneval=164, mtbench=80 always)
  --max_tokens     Max generated tokens per prompt (default: 60)
  --full           Run full default matrix
  --skip_fetch     Skip dataset download step
  --dry_run        Print plan without running

Examples:
  # Quick smoke test (2 min)
  python OSD/run_all.py \
      --student Qwen/Qwen2.5-0.5B --teacher Qwen/Qwen3-0.6B \
      --datasets diverse50 --modes alpha --K 1 --n 10

  # Alpha only, all datasets
  python OSD/run_all.py \
      --student Qwen/Qwen2.5-0.5B --teacher Qwen/Qwen3-0.6B \
      --modes alpha

  # Block efficiency sweep, gbv + traversal, K=3,5
  python OSD/run_all.py \
      --student OSD/checkpoints/ebe200_merged --teacher Qwen/Qwen3-0.6B \
      --student_label ebe200 \
      --modes specinfer,gbv,traversal --K 3,5
```

### Training (separate from eval)

```bash
# Train with EBE loss, 200 steps
python OSD/train_qwen3.py \
    --loss ebe --steps 200 --lr 3e-5 \
    --output OSD/checkpoints/ebe200

# Merge LoRA adapter into base weights for GBV evaluation
python OSD/train_qwen3.py --merge_only \
    --base Qwen/Qwen2.5-0.5B \
    --adapter OSD/checkpoints/ebe200 \
    --output OSD/checkpoints/ebe200_merged
```

### Seeding existing results into DB

```bash
# Import hand-collected results (alpha-50, K-sweep from prior runs)
python OSD/seed_db.py

# Add a single result from the REPL
python -c "
from seed_db import add_run
add_run('ebe200', 'gbv', K=3, dataset='gsm8k', block_eff=2.81)
"
```

---

## Datasets

| Dataset | Prompts used | Source | Focus |
|---|---|---|---|
| diverse50 | 50 | Local | Factual, coding, CS, ML, math |
| gsm8k | 30 (default) | HuggingFace: `gsm8k` | Math word problems |
| humaneval | 164 (all) | HuggingFace: `openai_humaneval` | Python code completion |
| math500 | 30 (default) | HuggingFace: `lighteval/MATH-Hard` | Competition math |
| mtbench | 80 (all) | HuggingFace: `HuggingFaceH4/mt_bench_prompts` | Multi-turn, diverse |
| alpaca | 30 (default) | HuggingFace: `tatsu-lab/alpaca` | Instruction following |

Downloaded automatically by `run_all.py` (skips if already present). All stored as JSONL in `OSD/data/`.

---

## Results Database

All results saved to `OSD/results.db` (SQLite).

```
runs table:
  run_tag         TEXT  UNIQUE  — e.g. 20260521_143022_ebe200_specinfer_diverse50_K3
  id              INT   autoincrement
  ts              TEXT  — ISO timestamp
  draft_label     TEXT  — baseline | kl200 | ebe200 | custom
  dataset         TEXT
  mode            TEXT  — alpha | specinfer | gbv | traversal | bv
  K               INT
  temperature     REAL
  alpha_mean      REAL
  alpha_ci95      REAL
  block_eff       REAL
  throughput      REAL  — tok/sec
  ms_per_tok      REAL

per_prompt table:
  run_id, prompt_idx, category, alpha, block_eff, gen_tokens, wall_ms

train_curves table:
  label, loss_name, step, loss, accept_weight
```

Query from Python:
```python
import sys; sys.path.insert(0, 'OSD')
import results_db

# All EBE runs
rows = results_db.query_runs({"draft_label": "ebe200"})

# Per-prompt breakdown for run 7
pp = results_db.query_per_prompt(7)
```

---

## Dashboard Features

Start: `python OSD/viz_server.py` → `http://localhost:5000/`

| Tab | What it shows |
|---|---|
| Alpha | Grouped bar: alpha by condition × dataset, with CI error bars |
| BE vs K | Line chart: block efficiency vs K, grouped by draft model |
| Mode Comparison | Grouped bar: specinfer/gbv/traversal × draft model |
| Sensitivity | Configurable: any X axis (K/temp/steps/LR) vs any Y (alpha/BE/throughput) |
| Per-Category | Heatmap: loss × category → alpha for diverse50 |
| Throughput | Bar chart + alpha vs throughput scatter |
| Training Curves | Loss vs step for all training runs |
| All Runs | Sortable table, CSV export |

---

## Hardware Requirements

| Setup | GPU | VRAM | Notes |
|---|---|---|---|
| Laptop eval (0.6B) | RTX 500 Ada | 4 GB | Both models: ~2.1 GB inference |
| Laptop training (LoRA) | RTX 500 Ada | 4 GB | ~2.6 GB peak with LoRA |
| Server eval (8B) | A100 40GB or 3090 24GB | 24+ GB | 8B target ~16 GB bfloat16 |
| Server training (8B) | A100 40GB | 40 GB | Draft + frozen 8B target |

**Critical:** never run multiple GPU processes simultaneously on the laptop — VRAM deadlock.  
`run_all.py` is always sequential by design.

---

## File Structure

```
OSD/
  run_all.py          Master eval runner (student + teacher -> all results)
  train_qwen3.py      LoRA training with KL / EBE loss
  results_db.py       SQLite persistence layer
  viz_server.py       Flask + Plotly web dashboard
  fetch_datasets.py   Dataset downloader (GSM8K, HumanEval, MATH500, MTBench, Alpaca)
  setup_env.py        One-file environment setup script
  seed_db.py          Import existing/manual results into DB
  eval_all.py         Lower-level eval runner (called by run_all.py)
  requirements.txt    pip dependencies (PyTorch installed separately)
  DELIVERABLES.md     Research deliverables and benchmark results
  data/               Evaluation datasets (JSONL)
  checkpoints/        LoRA adapters and merged models
  results.db          SQLite results database (auto-created)

GBV/                  Tree-based verification (separate repo)
  main.py
  data/eval30.jsonl
```

---

## Reproducing Results

```bash
# 1. Setup
python OSD/setup_env.py

# 2. Train draft models
python OSD/train_qwen3.py --loss kl  --steps 200 --lr 3e-5 --output OSD/checkpoints/kl200
python OSD/train_qwen3.py --loss ebe --steps 200 --lr 3e-5 --output OSD/checkpoints/ebe200
python OSD/train_qwen3.py --merge_only --base Qwen/Qwen2.5-0.5B \
    --adapter OSD/checkpoints/kl200  --output OSD/checkpoints/kl200_merged
python OSD/train_qwen3.py --merge_only --base Qwen/Qwen2.5-0.5B \
    --adapter OSD/checkpoints/ebe200 --output OSD/checkpoints/ebe200_merged

# 3. Run all eval for baseline
python OSD/run_all.py --student Qwen/Qwen2.5-0.5B    --teacher Qwen/Qwen3-0.6B
# 4. Run all eval for KL-200
python OSD/run_all.py --student OSD/checkpoints/kl200_merged  --teacher Qwen/Qwen3-0.6B \
    --student_label kl200
# 5. Run all eval for EBE-200
python OSD/run_all.py --student OSD/checkpoints/ebe200_merged --teacher Qwen/Qwen3-0.6B \
    --student_label ebe200

# 6. View results
python OSD/viz_server.py
```

---

*Original OSD paper: [Online Speculative Decoding (ICML 2024)](https://arxiv.org/pdf/2310.07177.pdf)*
