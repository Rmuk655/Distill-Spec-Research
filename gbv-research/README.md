# GBV Research — Speculative Decoding via Distillation

Improving speculative decoding acceptance rates by training a small draft
model to better match a large target model's token distribution.

## Directory layout

```
gbv-research/
├── capsules/                   Core research code (modular, by concern)
│   ├── distillation/           Knowledge-distillation training
│   │   ├── losses/             Pluggable loss objectives
│   │   │   ├── forward_kl.py  DistillSpec baseline (recommended)
│   │   │   ├── reverse_kl.py  Mode-seeking ablation
│   │   │   ├── jsd.py         Jensen-Shannon ablation
│   │   │   ├── l1.py          Total-variation ablation
│   │   │   └── ebe.py         Expected Block Efficiency (novel)
│   │   ├── model_families/     LLM-family-specific behaviour
│   │   │   ├── qwen.py        Qwen2.5 / Qwen3 (active)
│   │   │   └── gemma.py       Google Gemma (stub)
│   │   └── trainer.py         Training loop (model-family-agnostic)
│   │
│   ├── verification/           Speculative decoding at inference time
│   │   ├── algorithms/         All 8 verifier implementations
│   │   ├── runner.py           Main speculative decoding loop
│   │   ├── tree.py             Draft tree node abstractions
│   │   └── draft_generator.py i.i.d. draft path construction
│   │
│   ├── datasets/               Dataset management
│   │   ├── sources/            Per-dataset cleaning scripts
│   │   ├── downloader.py       Fetch from HuggingFace Hub
│   │   └── loader.py           Unified JSONL loader
│   │
│   └── dashboard/
│       └── server.py           Flask web results dashboard
│
├── orchestration/              Pipeline coordination
│   ├── pipeline.py             Crash-safe multi-step orchestrator
│   ├── eval_pipeline.py        Evaluation runner
│   ├── run_all.py              Full train→merge→eval pipeline
│   └── configs/
│       ├── laptop.yaml         RTX 500, Qwen2.5-0.5B → Qwen3-0.6B
│       └── server.yaml         A100, Qwen3-0.6B → Qwen3-8B
│
├── db/                         All generated run outputs (gitignored)
│   ├── checkpoints/            LoRA adapters and merged models
│   ├── wandb/                  W&B local logs
│   ├── results.db              SQLite experiment database
│   └── logs/                   Pipeline and training logs
│
├── paper/                      Publication artifacts
│   ├── figures/                Generated plots
│   ├── tables/                 Result tables
│   └── plots/                  Plot generation scripts
│
├── references/                 Original borrowed codebases (read-only)
│   ├── osd-original/           OSD (borrowed training framework)
│   ├── gbv-original/           GBV algorithms (Thomas et al. 2026)
│   └── adaspec/                AdaSpec reference implementation
│
├── docs/
│   ├── ADDING_A_MODEL_FAMILY.md
│   └── ADDING_A_LOSS.md
│
└── tests/                      Test suite
```

## Quick start

### Laptop (4 GB VRAM)

```bash
cd gbv-research

# 1. Install dependencies
pip install -r requirements.txt

# 2. Download datasets
python -m capsules.datasets.downloader --datasets gsm8k

# 3. Train (forward KL, 1000 steps, GSM8K)
python -m capsules.distillation.trainer \
    --loss forward_kl --steps 1000 \
    --dataset capsules/datasets/raw/gsm8k_train.jsonl \
    --val_dataset capsules/datasets/raw/gsm8k_30.jsonl

# 4. Run full evaluation
python orchestration/eval_pipeline.py \
    --drafts db/checkpoints/forward_kl-qwen-1000steps/ckpt_best_merged \
    --datasets gsm8k --modes traversal,gbv --K 1,3,5

# 5. Open dashboard
python -m capsules.dashboard.server   # → http://localhost:5000
```

### Server (A100 / Colab)

```bash
python -m capsules.distillation.trainer \
    --model_family qwen \
    --draft Qwen/Qwen3-0.6B --target Qwen/Qwen3-8B \
    --loss ebe --steps 5000 \
    --dataset capsules/datasets/raw/gsm8k_train.jsonl \
    --load_in_4bit     # only if VRAM < 24 GB
```

## Extending the framework

| Task | Guide |
|---|---|
| Add a new model family (Gemma, LLaMA, …) | [docs/ADDING_A_MODEL_FAMILY.md](docs/ADDING_A_MODEL_FAMILY.md) |
| Add a new loss objective | [docs/ADDING_A_LOSS.md](docs/ADDING_A_LOSS.md) |
| Add a new dataset | Drop JSONL in `capsules/datasets/raw/`, add an entry to `capsules/datasets/sources/` |
| Add a new verifier algorithm | Implement in `capsules/verification/algorithms/registry.py` |

## Current results (Laptop — Qwen2.5-0.5B → Qwen3-0.6B, GSM8K)

| Model | Val KL loss | PPL | Notes |
|---|---|---|---|
| Baseline (no fine-tuning) | — | 19.76 | Pre-training PPL |
| forward_kl 1000 steps | — | ~9.47 | DistillSpec baseline |
| ebe 1000 steps | ~−8.44 | — | Novel EBE objective |
| reverse_kl 1000 steps | ~9.21 | 26.05 | Mode-seeking; PPL inflated (cross-family artifact) |
| jsd 1000 steps | ~0.12 | — | In progress |

Full eval results with block efficiency and throughput: `db/results.db`

## Team

- **Mukund** (IIT Hyderabad) — student researcher
- **Ram** (Adobe) — advisor
- Target: ICLR mid-September 2026
