# Hyperparameters and Data Tuning

Researcher-facing reference for **why** each data split and hyperparameter was chosen,
how training validation differs from final evaluation, and how to tune without
re-running everything.

See also: [`GUIDE.md`](GUIDE.md) (pipeline), [`ISSUES.md`](ISSUES.md) (broken losses),
[`ADDING_A_MODEL_FAMILY.md`](ADDING_A_MODEL_FAMILY.md) (Qwen → GPT-2 / LLaMA).

---

## 1. Data splits (GSM8K)

All splits are **deterministic** (`random.Random(42)` in `core/datasets/downloader.py`).
The same prompts appear in the same order on every machine after `fetch_all()`.

### 1.1 Fold layout (1319-item GSM8K **test** split)

Training uses the **train** split (`gsm8k_train.jsonl`, 7473 items) — completely separate.

```
Shuffled GSM8K test set [0 ─────────────────────────────── 1318]
│
│  [0 ── 29]       gsm8k_30.jsonl        ← val1 (fast)
│  [0 ── 99]       gsm8k_100.jsonl       ← val2 (slow) — subset of val1 range
│  [100 ─ 199]     buffer (unused — extra separation)
│  [200 ─ 299]     gsm8k_eval_100.jsonl   ← eval_final (pipeline default)
│  [200 ─ 1318]    gsm8k_eval_1119.jsonl ← paper eval (all non-val items)
│
└── Training: gsm8k_train.jsonl (train split — no overlap with any test item)
```

| Pool | File | Items | Used for |
|------|------|-------|----------|
| **Train** | `gsm8k_train.jsonl` | 7473 | Gradient updates only |
| **val1 (fast)** | `gsm8k_30.jsonl` | 30 | Early stopping, cheap trend check |
| **val2 (slow)** | `gsm8k_100.jsonl` | 100 | `ckpt_best_slow` selection (when enabled) |
| **eval_final** | `gsm8k_eval_100.jsonl` | 100 | Post-training BE/PPL — **all models, same file** |
| **paper eval** | `gsm8k_eval_1119.jsonl` | 1119 | Full held-out test for publication |

**Non-overlap guarantee:** `fetch_gsm8k_eval()` skips the first 200 shuffled items
(`_GSM8K_VAL_POOL_SIZE = 200`), so eval_final never shares prompts with val1 or val2.

### 1.2 What happens during training

| Stage | Dataset | Purpose | Checkpoint saved |
|-------|---------|---------|------------------|
| Forward/backward | `gsm8k_train` | Learn LoRA weights | — |
| Fast val (every 100 steps) | `gsm8k_30` | Plateau detection, early stop | `ckpt_best` |
| Slow val (every 500 steps) | `gsm8k_100` | Reliable ranking | `ckpt_best_slow` |
| PPL health check | 5 prompts from val set | Detect divergence | — |

**Important:** val1 and val2 are for **training decisions only**. They must not be used
for reported BE/alpha numbers — that would bias results toward checkpoints selected on
the same prompts.

### 1.3 What happens at eval_final

Pipeline eval steps read **`evaluation.eval_datasets`** from YAML (e.g. `bases/a100.yaml`).
The gsm8k* entry (default `gsm8k_eval`) → `gsm8k_eval_100.jsonl`. Change the YAML only —
no per-loss edits in `experiment.py`.

- **Baseline** (unmodified draft) and **every trained loss** evaluate on the **identical**
  100 prompts → loss-to-loss comparisons are valid.
- Reproducible: same file, same order, same seed — reruns produce the same prompt list.
- Override only for paper: `n_prompts_gsm8k: 1119` + `gsm8k_eval_1119.jsonl`.

```bash
# Ensure eval pool exists (auto-fetched on first pipeline run)
python core/datasets/downloader.py --datasets gsm8k_eval --n 100
```

### 1.4 GPT-2 / WikiText (laptop convergence)

`laptop_gpt2.yaml` uses **WikiText-2**, not GSM8K — distilgpt2 and gpt2-medium were
trained on WebText; GSM8K math is OOD for them. WikiText gives in-distribution KL signal
for proving convergence before spending A100 budget on Qwen/Llama.

---

## 2. Hyperparameter defaults and rationale

Primary config: `orchestration/configs/a100_qwen.yaml` (inherits `bases/a100.yaml`).

### 2.1 Training core

| Parameter | Default (A100 Qwen) | Rationale |
|-----------|---------------------|-----------|
| `steps` | 4000 | 500 optimizer updates at `grad_accum=8`; minimum for convergence signal |
| `grad_accum` | 8 | 2× more optimizer steps vs 16; tree losses benefit from fresher updates |
| `lr` | 3e-5 | DistillSpec/EAGLE convention; works for KL/JSD/tree (except BV) |
| `warmup_steps` | 400 (10%) | Linear warmup → linear decay to 0; 50 optimizer warmup steps |
| `lora_r` / `lora_alpha` | 16 / 32 | Wider rank for 0.6B→8B gap (~13× parameter ratio) |
| `teacher_temperature` | 0.8 | Softer teacher distribution; standard in distillation literature |
| `grad_clip` | 5.0 | Tree losses have larger norms (path weights Π αᵢ); 1.0 over-clips |
| `seed` | 42 | Reproducible data order and init |
| `max_new_tokens` | 128 | On-policy rollout length for tree training |

**Optimizer math:** `optimizer_steps = steps / grad_accum`. At 4000/8 = **500 updates**.
Warmup ends at step 400 → 360 decay steps. This is the real training budget, not 4000.

### 2.2 Per-loss overrides

| Loss | Override | Why |
|------|----------|-----|
| `bv_tree`, `gbv_tree` | `--lr 1e-5` | BV block-acceptance integral amplifies gradients (`∂h/∂q ∝ 1/(1−w)`); clamps in `tree_losses.py` bound but lower lr still helps |
| Flat KL/JSD/L1 | default `3e-5` | Per-token KL gradients are O(1) |
| `online_*` | `online_lr: 3e-4` (10×) | On-policy adaptation; **currently disabled** — see ISSUES.md |
| `online_ebe_*` | `online_ebe_lr: 1e-4` | Lower than online_kl; EBE gradient scale differs |

```bash
# BV tree (auto lr in aip_run)
python deploy/aip_run.py --loss bv_tree --train

# Phase-2 continuation from ckpt_best
python deploy/aip_run.py --loss kl_tree --train --train_steps 5000 --lr 1e-5 --warmup_steps 100
```

### 2.3 Health / validation during training

| Parameter | Default (a100_qwen) | Rationale |
|-----------|-------------------|-----------|
| `val_every` | 100 | Fast val every 100 grad steps (~45 s/check on A100) |
| `slow_val_every` | 500 | Slow val every 500 steps (~2.5 min/check) |
| `slow_val_n` | 100 | SE ≈ 0.017 — reliable for checkpoint ranking |
| `early_stop_patience` | 5 | Stop after 5 consecutive no-improve checks |
| `ppl_check_every` | 500 | Pre-training baseline PPL; warn if > 1.25× baseline |

**Two-tier val design:**

- **Fast (n=30):** cheap gating — "better / not better". Noise floor ≈ 0.003 val loss.
  Do not over-interpret small fast-val swings.
- **Slow (n=100):** ranking metric — drives `ckpt_best_slow` used for merge/eval.

Total val overhead ≈ 50 min over 4000 steps (acceptable vs ~27 min flat training).

### 2.4 Tree training

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| `tree_K` | 4 | Draft tree width (paths per step) |
| `tree_L` | 8 | Draft tree depth (tokens per path) |
| Eval `K` | 3 | Verifier block size at eval (matches paper convention) |
| Eval `L` | 8 | Max draft length per verification round |

### 2.5 Checkpointing

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| `save_every` | 100 | `ckpt_latest` for resume |
| `milestone_every` | 200 | 10 points over 2000 steps for convergence curves |
| `max_checkpoints` | 8 | Disk budget on shared storage |

Merge step bakes LoRA into base weights → `*_merged/` for eval.

---

## 3. Evaluation scope (tiered)

Do **not** run 4 datasets × 9 modes on every experiment. Use tiers:

| Stage | Datasets | Modes | When |
|-------|----------|-------|------|
| **Development** | `gsm8k_eval` only | All 9 (or 4: alpha,bv,gbv,traversal) | Every converged loss |
| **Candidate selection** | gsm8k_eval + humaneval | All 9 | Top ~5 losses after dev ranking |
| **Paper confirmation** | gsm8k_eval + humaneval + math500 + mtbench + alpaca | All 9 | Top 3 losses only |

**YAML control** (`orchestration/configs/bases/a100.yaml`):

```yaml
evaluation:
  eval_datasets: [gsm8k_eval]   # Phase 3 only; Phase 4 auto-skipped
```

| `eval_datasets` | Phase 3 file | Phase 4 |
|-----------------|--------------|---------|
| `[gsm8k_eval]` | `gsm8k_eval_100.jsonl` | skipped |
| `[gsm8k]` | `gsm8k_100.jsonl` (val pool — dev only) | skipped |
| `[gsm8k_eval, humaneval, …]` | held-out gsm8k | humaneval, math500, … |

Paper mode — override in leaf config (`a100_qwen.yaml`):

```yaml
evaluation:
  eval_datasets: [gsm8k_eval, humaneval, math500, mtbench, alpaca]
  n_prompts_gsm8k: 1119   # uses gsm8k_eval_1119.jsonl
  max_tokens: 100         # optional — longer generations for hard problems
```

### 3.1 Eval knobs (all YAML-tunable)

These live under `evaluation:` in `bases/a100.yaml` / `bases/laptop.yaml` unless a
leaf config overrides them. Wired in `experiment.py` → `evaluate.py` CLI.

| YAML key | CLI flag | Default (A100 base) | What it controls |
|----------|----------|---------------------|------------------|
| `eval_datasets` | `--datasets` | `[gsm8k_eval]` | Which JSONL pool(s) to run |
| `n_prompts_gsm8k` | `--n` (Phase 3) | 100 | Prompt count for gsm8k* eval |
| `n_prompts` | `--n` (Phase 4) | 100 | Prompt count for humaneval/math500/… |
| `max_tokens` | `--max_tokens` | **50** (full) | Tokens generated **per prompt** during **alpha + BE** eval (spec-decode measurement length) |
| `max_tokens_smoke` | `--max_tokens` (smoke) | **30** | Same, for `--smoke` runs only |
| `modes` | `--modes` | 9 verifiers | Which verifiers to sweep |
| `K_values` | `--K` | `[3]` | Block size at eval |
| `L` | `--L` | 8 | Draft depth per verification round |
| `temperatures` | `--temperature` | `[1.0]` | Sampling temperature |
| `task_batch` | `--task_batch` | 32 (A100) / 4 (laptop) | Parallel prompts for **task_score** only |
| `tree_eval_modes` | tree-loss `--modes` | 8-verifier matrix | A100 tree-loss alignment sweep |
| `exclude_modes_when_4bit` | filters `--modes` | `[alpha]` | Drop modes that OOM with NF4 teacher |

**`max_tokens` is NOT training `max_new_tokens`.** Training uses `training.max_new_tokens`
(128) for teacher rollouts during distillation. Eval `max_tokens` caps how long each
**eval prompt** runs under speculative decoding when measuring α or block efficiency.

**Why 50 (full) / 30 (smoke)?** GSM8K answers are usually short; 50 tokens is enough
to measure acceptance/BE without 4× eval cost. Smoke uses 30 to keep the fast path under
~15 minutes. Paper runs often raise **both** `n_prompts_gsm8k` and `max_tokens`:

| Tier | `n_prompts_gsm8k` | `max_tokens` | Approx cost driver |
|------|-------------------|--------------|-------------------|
| Smoke | 3 (code default) | 30 (YAML) | Code-path check |
| Finalization | 100 | 50 | ~1–1.5 h/loss (4–9 modes) |
| Paper | 1119 | 50–100 | ~8–14 h/loss (45 cells) |

### 3.2 Quality metrics at eval (not the same thing)

Three separate checks run inside `evaluate.py`:

| Metric | How enabled | What it measures | Matches against |
|--------|-------------|------------------|-----------------|
| **Block efficiency / α** | Always (per mode) | Spec-decode speedup | — (throughput metric) |
| **Perplexity** | **Default ON** (`--no_perplexity` to skip) | LM quality on prompt text | Baseline PPL on same dataset |
| **task_score** | `--task_score` (pipeline passes this on GSM8K evals) | **Downstream accuracy** | **Gold answer in JSONL** |

**task_score on GSM8K:** The student generates a full answer (greedy, 512 tokens).
`score_gsm8k()` extracts the **final number** from the generated text and compares it
to the **`answer` field** in each JSONL row (the official GSM8K solution text, which
contains `#### 42` style markers). Exact numeric match → 1.0, else 0.0. Mean over
prompts = accuracy (0–1).

This works for **`gsm8k_eval`** the same as `gsm8k` — both files use the same
`{prompt, answer}` schema from HuggingFace GSM8K. The name `gsm8k_eval` only identifies
**which split** (held-out items[200:299]), not a different scoring rule.

**task_score runs on `alpha` cells only** (needs draft+teacher loaded). BE modes still
record `block_eff`; they do not run task_score.

### 3.3 Experiment matrix (what to build)

One row per (model, dataset, mode). Example for GSM8K finalization:

| Model | Dataset | Mode | BE | α |
|-------|---------|------|----|---|
| baseline | gsm8k_eval | bv | 3.79 | 0.58 |
| baseline | gsm8k_eval | gbv | 5.11 | 0.62 |
| kl_tree | gsm8k_eval | bv | 4.82 | 0.65 |
| kl_tree | gsm8k_eval | gbv | 6.27 | 0.71 |
| traversal_tree | gsm8k_eval | bv | 3.55 | 0.55 |

Query from `results.db` or W&B dashboard. All rows must use the same `gsm8k_eval_*` file.

---

## 4. Phase-2 continuation (converged but plateauing)

When val loss flattens before `steps` completes (e.g. kl_tree best=0.4389 at step 1200):

1. Training finished or early-stopped — `ckpt_best` is valid.
2. For more improvement, continue from best weights with fresh optimizer + lower lr:

```bash
# On Pluto — seed ckpt_latest from best, drop stale Adam state
cp -a checkpoints/kl_tree-.../ckpt_best/. checkpoints/kl_tree-.../ckpt_latest/
rm checkpoints/kl_tree-.../ckpt_latest/optimizer.pt

# Phase 2
python deploy/aip_run.py --loss kl_tree --train --train_steps 5000 --lr 1e-5 --warmup_steps 100
```

**Resume behaviour:** `--resume` reloads weights + optimizer state but **restarts the LR
schedule from warmup** (scheduler state is not saved). For clean Phase 2, delete
`optimizer.pt` and optionally remove root `adapter_config.json` if pipeline skips train.

---

## 5. Multi-model portability

Current production pair: **Qwen3-0.6B → Qwen3-8B** (`a100_qwen.yaml`).

| Family | Config | Draft → Target | Dataset | Status |
|--------|--------|----------------|---------|--------|
| Qwen3 | `a100_qwen.yaml` | 0.6B → 8B | GSM8K | Production |
| GPT-2 | `laptop_gpt2.yaml` | distilgpt2 → gpt2-medium | WikiText-2 | Convergence test (CPU) |
| LLaMA | `a100_llama.yaml` (planned) | 1B → 3B | GSM8K | Stub |

**Effort to add a family:** ~1 file (`core/model_families/<name>.py`) + 1 YAML leaf.
Four behaviours must be verified: LoRA modules, temperature recovery, forbidden-token
masking, chat template. See [`ADDING_A_MODEL_FAMILY.md`](ADDING_A_MODEL_FAMILY.md).

Hard dependencies on Qwen today:
- Default `model_family: qwen` in trainer/evaluate
- Chat template for instruction datasets (MTBench, Alpaca)
- `hw_tier` VRAM estimates tuned for 0.6B/8B pair

GPT-2 and LLaMA paths are already wired in `FAMILY_REGISTRY`; switching is a config
change, not a rewrite.

---

## 6. Known broken / disabled losses

Do not tune hyperparameters for these until fixed — see [`ISSUES.md`](ISSUES.md):

| Loss | Symptom | Disabled in |
|------|---------|-------------|
| `ebe`, `ebe_single`, `ebe_tree` | BE −0.2 vs baseline; PPL fine | All configs via `exclude_losses` |
| `online_*` | PPL 2× baseline; BE −0.68 | All configs via `exclude_losses` |

---

## 7. Quick reference commands

```bash
# Train one loss
python deploy/aip_run.py --loss kl_tree --train --no_smoke

# Eval one loss (baseline auto-skipped; uses gsm8k_eval)
python deploy/aip_run.py --loss kl_tree --eval --no_smoke

# Force re-eval after gsm8k_eval fix (old rows used gsm8k val pool)
python deploy/aip_run.py --loss kl --eval --force_eval --experiment_tag reeval_v2 --no_smoke

# Dry-run pipeline step list
python orchestration/experiment.py --config a100_qwen --dry_run --losses kl,rev_kl
```

---

## 8. Decision checklist for researchers

Before changing a hyperparameter, ask:

1. **Data:** Am I evaluating on `gsm8k_eval` (held-out) or `gsm8k` (val pool)?
2. **Budget:** Is 500 optimizer steps enough, or do I need Phase 2?
3. **Loss type:** BV integral → `lr 1e-5`; KL/tree → default `3e-5`.
4. **Val noise:** Fast val swing < 0.003 → ignore; trust slow val trend.
5. **Eval cost:** Dev = 1 dataset; paper = 5 datasets × top-3 only.
6. **Checkpoint:** Merge uses `ckpt_best` or `ckpt_best_slow` — confirm which your run saved.
