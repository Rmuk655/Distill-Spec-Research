# SpecDist ML Research Progression

This document describes the standard 4-level workflow for running SpecDist
experiments, from local sanity checks through to paper-quality results.

This mirrors standard MLOps practice for ML research with large language
models: **debug on the smallest hardware, look for trends on mid-tier, establish
results on production hardware, finalize on the largest scale.**

---

## The 4-Level Ladder

| Level | Config | Teacher | Hardware | Steps | Wall time | W&B group | Purpose |
|-------|--------|---------|----------|-------|-----------|-----------|---------|
| 1 | `laptop` | Qwen3-0.6B (0.6 GB) | Your laptop CPU/MPS/CUDA | 50-100 | ~5-15 min | `laptop-gsm8k` | Debug: verify code runs, losses go down |
| 2 | `colab_lite` | Qwen3-1.7B BF16 (3.4 GB) | Free T4 (15 GB VRAM) | 300 | ~25 min | `colab-lite` | Trend: does the loss function help? |
| 3 | `colab` | Qwen3-4B BF16 (8 GB) | Free T4 (15 GB VRAM) | 500 | ~2-4 h | `colab-t4` | Results: publishable block-efficiency numbers |
| 4 | `colab_a100` | Qwen3-8B BF16 (16 GB) | Colab Pro/Pro+ A100 (40 GB) | 2000 | ~2-3 h | `colab-a100` | Paper: best quality, full eval, ablations |

---

## VRAM Budget by Level

| Level | Draft | Teacher | Activations | Total | Headroom |
|-------|-------|---------|------------|-------|----------|
| 1 (laptop) | 1.2 GB | 1.2 GB | ~0.5 GB | ~3 GB | CPU/MPS headroom varies |
| 2 (colab_lite) | 1.2 GB | 3.4 GB | ~1.0 GB | **~5.6 GB** | 9.4 GB free on T4 |
| 3 (colab) | 1.2 GB | 8.0 GB | ~1.5 GB | **~10.7 GB** | 4.3 GB free on T4 |
| 4 (colab_a100) | 1.2 GB | 16.0 GB | ~2.5 GB | **~19.7 GB** | 20 GB free on A100 |

**Why no 4-bit quantization?**
4-bit NF4 loading (bitsandbytes) reads each weight tensor as BF16 into CPU RAM
first, then converts to NF4. For Qwen3-8B: ~399 tensors × ~40 MB = ~16 GB of
CPU RAM intermediates. Colab has ~12 GB system RAM. The Linux OOM killer fires
with SIGKILL (exit code -9) — Python never runs, no retry possible.

All three T4/A100 configs use plain BF16 loading: direct mmap from disk to
VRAM, zero CPU RAM spike, zero OOM risk.

---

## W&B Run Naming

Every training run produces a W&B run named:

```
{run_label}-{loss}_{family}_{steps}steps
```

Examples:
- `0.6B-laptop-kl_qwen_100steps`      — Level 1 laptop debug
- `1.7B-T4-lite-kl_qwen_300steps`     — Level 2 colab_lite trend run
- `1.7B-T4-lite-ebe_qwen_300steps`    — Level 2 with EBE loss
- `4B-T4-kl_qwen_500steps`            — Level 3 production T4 run
- `4B-T4-gbv_qwen_500steps`           — Level 3 with GBV loss
- `8B-A100-kl_qwen_2000steps`         — Level 4 paper run

This makes it possible to compare across levels in a single W&B dashboard:
filter by `run_label` prefix to see all EBE runs regardless of teacher size,
or filter by `wandb_group` to see only a specific tier.

The `run_label` is set in each YAML's `logging.run_label` field. It is
automatically forwarded from the YAML through `experiment.py` to `trainer.py`.

---

## Notebooks

| Notebook | Config | Teacher | When to use |
|----------|--------|---------|-------------|
| `colab_lite_quickstart.ipynb` | `colab_lite` | 1.7B BF16 | Level 2: first T4 run, ~25 min, trend check |
| `colab_quickstart.ipynb` | `colab` | 4B BF16 | Level 3: production T4 run, ~2-4 h |
| `colab_a100_quickstart.ipynb` | `colab_a100` | 8B BF16 | Level 4: paper runs on A100 (Pro/Pro+ only) |

All three notebooks share the same cell structure (Cell 0 one-shot bootstrap,
Cell 5 monitor, etc.). The only differences are the `CONFIG` default and the
VRAM budget notes in the title cell.

---

## Standard Experiment Workflow

### Step 1 — Laptop sanity check (Level 1)

```bash
# In gbv-research/
python orchestration/experiment.py --config laptop --smoke --yes
```

Smoke test runs 10 training steps and a tiny eval. Should complete in < 5 min.

**Gate**: losses go down, no exceptions, eval modes produce numbers.
**If this fails**: fix the code. Nothing else should run until this passes.

---

### Step 2 — Colab Lite trend run (Level 2)

Open `deploy/colab_lite_quickstart.ipynb` on a free T4 runtime.

Run Cell 0 with `SMOKE = False` (or `SMOKE = True` first to verify end-to-end
in ~10 min before committing to the full 25 min run).

**What to look for:**
- Loss curve: should decrease steadily (not flatline, not spike)
- Block efficiency: any improvement over the naive baseline?
- Alpha eval (token acceptance): direction consistent with loss improvement?

**Gate**: at least one loss function shows a consistent positive trend.
**If no trend**: revisit the loss implementation before moving to Level 3.
**Cost if wrong**: 25 min, not 4 hours.

---

### Step 3 — Colab T4 production run (Level 3)

Open `deploy/colab_quickstart.ipynb` on a free T4 runtime.

Run with `CONFIG = "colab"` (default). 500 steps, ~2-4 h.

**What to look for:**
- Block efficiency delta vs naive baseline > noise (> 5% consistently)
- Results reproducible across 2 seeds
- Alpha eval consistent with block efficiency improvement

**Gate**: effect is statistically meaningful, directionally consistent, and
larger than run-to-run variance (compare two seeds).
**These are publishable numbers.** Record the W&B run IDs.

---

### Step 4 — A100 paper runs (Level 4)

Open `deploy/colab_a100_quickstart.ipynb` on a **Colab Pro/Pro+ A100 runtime**.

Run with `CONFIG = "colab_a100"` (default). 2000 steps, ~2-3 h.

**Requirements before running Level 4:**
- Level 3 results are clean and reproducible
- The hypothesis being tested is finalized (no more code changes)
- At least 2 Level 3 seeds confirm the direction

**What's different from Level 3:**
- Larger teacher (8B vs 4B) → stronger distillation signal
- More steps (2000 vs 500) → converged model
- Larger LoRA rank (r=16 vs r=8) → better adaptation
- `torch.compile` enabled → ~10-30% faster training
- More eval prompts (50 vs 20) → tighter confidence intervals

---

## What Changes Between Levels (What Stays the Same)

### Same across all levels:
- Codebase (orchestration/, algorithms/, core/)
- Pipeline steps (train → merge → eval)
- Loss functions (KL, EBE, GBV, etc.)
- Verifier implementations (SpecInfer, GBV, traversal, etc.)
- Dataset (GSM8K train/eval split)
- W&B project (`distillspec`)
- Crash-safe checkpoint resume

### Varies by level:
| Parameter | laptop | colab_lite | colab | colab_a100 |
|-----------|--------|------------|-------|------------|
| `target` (teacher) | Qwen3-0.6B | Qwen3-1.7B | Qwen3-4B | Qwen3-8B |
| `steps` | 50-100 | 300 | 500 | 2000 |
| `lora_r` | 4 | 4 | 8 | 16 |
| `lora_alpha` | 8 | 8 | 16 | 32 |
| `save_every` | 10 | 25 | 25 | 100 |
| `max_new_tokens` | 32 | 128 | 96 | 128 |
| `n_prompts` (eval) | 10 | 20 | 20 | 50 |
| `K_values` (eval) | [3] | [3,5] | [3,5] | [1,3,5,8] |
| `compile` | false | false | false | true |
| `load_in_4bit` | false | false | false | false |
| `wandb_group` | laptop-gsm8k | colab-lite | colab-t4 | colab-a100 |
| `run_label` | 0.6B-laptop | 1.7B-T4-lite | 4B-T4 | 8B-A100 |

---

## Comparing Levels in W&B

In the W&B project `distillspec`, use these filters to compare:

**Compare a single loss across teacher sizes:**
```
# Filter runs by name prefix
run_label:1.7B-T4-lite AND loss:ebe   ← Level 2 EBE
run_label:4B-T4 AND loss:ebe           ← Level 3 EBE
run_label:8B-A100 AND loss:ebe         ← Level 4 EBE
```

**See all runs at a given level:**
```
group:colab-lite        ← all Level 2 runs
group:colab-t4          ← all Level 3 runs
group:colab-a100        ← all Level 4 runs
```

**Compare losses at the same level:**
```
group:colab-t4          → overlay kl, ebe, gbv, jsd, l1 curves
```

---

## Adding a New Config Tier

If you need a new hardware tier (e.g. Kaggle T4x2, Lambda Labs A10G):

1. Copy the nearest YAML (e.g. `cp colab.yaml kaggle.yaml`)
2. Update `hardware:`, `models:`, `training:`, `checkpointing:`, `logging:`
3. Set `logging.run_label` to something like `"4B-kaggle"` or `"8B-A10G"`
4. Set `logging.wandb_group` to something like `"kaggle-t4"` or `"lambda-a10g"`
5. The pipeline, trainer, and W&B wiring pick up the new config automatically

No code changes needed — the config is self-contained.
