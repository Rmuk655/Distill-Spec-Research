# SpecDist ML Research Progression

This document describes the standard 4-level workflow for running SpecDist
experiments, from local sanity checks through to paper-quality results.

This mirrors standard MLOps practice for ML research with large language
models: **debug on the smallest hardware, look for trends on mid-tier, establish
results on production hardware, finalize on the largest scale.**

---

## The 4-Level Ladder

| Level | Mode | Config | Teacher | Steps | Wall time | W&B group | Purpose |
|-------|------|--------|---------|-------|-----------|-----------|---------|
| 1a | `--smoke` | `laptop` | 0.6B | 10 | ~3-5 min | `laptop-gsm8k` | Crash test: does the pipeline start? |
| 1b | full | `laptop` | 0.6B | 100 | ~15-20 min | `laptop-gsm8k` | Code exercise: does every path work? |
| 2 | full | `colab_lite` | 1.7B BF16 | 300 | ~25 min | `colab-lite` | Trend: does the loss function help? |
| 3 | full | `colab` | 4B BF16 | 500 | ~2-4 h | `colab-t4` | Results: publishable block-efficiency numbers |
| 4 | full | `colab_a100` | 8B BF16 | 2000 | ~2-3 h | `colab-a100` | Paper: best quality, full eval, ablations |

> **Why can't Level 1 produce trends?**
> The laptop teacher (Qwen3-0.6B) is the *same size class* as the draft (Qwen2.5-0.5B).
> Distillation from an equal-capacity model produces near-zero signal.
> Level 1 is purely a code correctness gate. Start looking for trends at Level 2.

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

### Step 1a — Crash test (Level 1a)

```bash
# In gbv-research/
python orchestration/experiment.py --config laptop --smoke --yes
```

Runs **10 training steps** and a minimal eval. Should complete in **< 5 min**.

What `--smoke` exercises:
- Python imports, CUDA device detection
- One forward + backward pass per loss function
- Dataset loading
- One checkpoint save
- A tiny eval (1-2 prompts per mode)

**Gate**: no exceptions, loss values are finite numbers.
**If this fails**: fix the code. Do not proceed to 1b until 1a passes.

---

### Step 1b — Full code-path exercise (Level 1b)

```bash
# In gbv-research/
python orchestration/experiment.py --config laptop --yes
```

Runs **100 training steps**, full eval suite. Should complete in **< 20 min**.

What the full laptop run exercises that smoke does NOT:
- Rolling checkpoint save/load cycle (save_every=10 → 10 saves)
- Milestone checkpoints (milestone_every=50 → 2 permanent saves)
- Validation health check (val_every=25 → 4 validation passes)
- PPL threshold check (ppl_check_every=50 → 2 checks)
- Full merge step (LoRA → base model)
- All 6 eval modes: alpha, bv, gbv, traversal, specinfer, naive
- W&B logging (run appears in wandb.ai under group `laptop-gsm8k`)
- Database writes (results.db)
- Pipeline state machine (done_check files)

**Gate**: all 6 eval modes produce output, W&B run visible, no exceptions.

> **Important**: do NOT interpret block-efficiency numbers from this run.
> The 0.6B teacher is the same size as the draft — distillation signal is
> near-zero. The numbers are structurally meaningless at this level.
> Only check that the eval modes *ran* and produced valid-looking numbers.

**If this fails**: fix the code. Do not run colab_lite until 1b passes.

**After 1b passes — summarize and log:**
```bash
python tools/run_summary.py --hw_tier laptop --markdown --log_id 20260528-001
# paste output into deploy/RUN_LOG.md Level 1 section
```

**🔐 Code Review Gate → Level 2** (must complete before first colab_lite run):
- [ ] Level 1b passes (all 6 modes, no errors)
- [ ] PR opened with all changes since last review
- [ ] Loss function implementation reviewed against paper/spec
- [ ] Verifier implementation reviewed against GBV spec
- [ ] Reviewer sign-off recorded in `deploy/RUN_LOG.md` Level 1 section

---

### Step 2 — Colab Lite trend run (Level 2)

Open `deploy/colab_lite_quickstart.ipynb` on a free T4 runtime.

Run Cell 0 with `SMOKE = False` (or `SMOKE = True` first to verify end-to-end
in ~10 min before committing to the full 25 min run).

**What to look for:**
- Loss curve: should decrease steadily (not flatline, not spike)
- Block efficiency: any improvement over the naive baseline?
- Alpha eval (token acceptance): direction consistent with loss improvement?

**After the run — summarize and log:**
```bash
python tools/run_summary.py --hw_tier colab_lite --markdown --log_id 20260528-002
# paste output into deploy/RUN_LOG.md Level 2 section
# record W&B run URL immediately
```

**Gate**: at least one loss function shows a consistent positive trend.
**If no trend**: revisit the loss implementation before moving to Level 3.
**Cost if wrong**: 25 min × 2 seeds, not 4 hours × 2 seeds.

**🔐 Code Review Gate → Level 3** (must complete before first colab run):
- [ ] Level 2 shows positive trend on ≥2 verifiers
- [ ] Second Level 2 seed (different `seed:` in YAML) confirms direction
- [ ] PR reviewed: hypothesis matches what the code actually tests
- [ ] PR reviewed: no data leakage (eval prompts not in train set)
- [ ] W&B run URLs recorded in `deploy/RUN_LOG.md`
- [ ] Reviewer sign-off recorded

---

### Step 3 — Colab T4 production run (Level 3)

Open `deploy/colab_quickstart.ipynb` on a free T4 runtime.

Run with `CONFIG = "colab"` (default). 500 steps, ~2-4 h.

**What to look for:**
- Block efficiency delta vs naive baseline > noise (> 5% consistently)
- Results reproducible across 2 seeds
- Alpha eval consistent with block efficiency improvement

**After the run — summarize and log:**
```bash
python tools/run_summary.py --hw_tier colab --markdown --log_id 20260528-003
# paste into deploy/RUN_LOG.md Level 3 section
```

**Gate**: effect is statistically meaningful, directionally consistent, and
larger than run-to-run variance (compare two seeds).
**These are publishable numbers.** Record the W&B run IDs immediately.

**🔐 Code Review Gate → Level 4** (must complete before any A100 run):
- [ ] Level 3 confirmed across ≥2 seeds
- [ ] Effect > 5% on ≥1 verifier × K combination
- [ ] PR reviewed: no bugs introduced since Level 2 review
- [ ] Hypothesis text finalized and written into `GUIDE.md` — no more code changes
- [ ] Ablation plan agreed (what to ablate on A100, which seeds, which K values)
- [ ] Both researchers sign off — this is the commitment point before expensive compute
- [ ] Sign-off recorded in `deploy/RUN_LOG.md`

---

### Step 4 — A100 paper runs (Level 4)

Open `deploy/colab_a100_quickstart.ipynb` on a **Colab Pro/Pro+ A100 runtime**.

A100 runs use the **3-tier profile system** — do not call `--config colab_a100` directly.
Full workflow: `orchestration/configs/profiles/README.md`.

**Tier 1 — train baseline losses** (~2-3 h, run first):
```bash
# Smoke first (catches crashes in ~5 min):
python orchestration/experiment.py --config profiles/a100_baseline_losses --smoke --yes \
  --storage_root /content/drive/MyDrive/specdist
# Full run:
python orchestration/experiment.py --config profiles/a100_baseline_losses --yes \
  --storage_root /content/drive/MyDrive/specdist
```
Trains KL, JSD, L1, online KL. Produces 4 merged model checkpoints.

**Tier 2 — verifier sweep** (~1-2 h, after Tier 1 checkpoints exist):
```bash
python orchestration/experiment.py --config profiles/a100_verifier_sweep --yes \
  --storage_root /content/drive/MyDrive/specdist
```
`eval_only: true` — no training. Re-evaluates all baselines at n=200 across all 6 verifiers.

**Tier 3 — new loss** (~2-3 h, when a new algorithm is ready):
```bash
cp orchestration/configs/profiles/a100_new_loss_template.yaml \
   orchestration/configs/profiles/a100_ebe_v2.yaml
# edit: set losses: [ebe, kl, jsd, l1, online], update wandb_group and run_label
python orchestration/experiment.py --config profiles/a100_ebe_v2 --yes \
  --storage_root /content/drive/MyDrive/specdist
```

**Requirements before running Level 4** (all gates above must be closed):
- Level 3 results are clean and reproducible
- The hypothesis being tested is finalized (no more code changes)
- At least 2 Level 3 seeds confirm the direction
- Ablation plan is written down

**After each tier — summarize and log:**
```bash
python tools/run_summary.py --hw_tier a100 --markdown --log_id 20260528-004
# paste into deploy/RUN_LOG.md Level 4 section
```

**What's different from Level 3:**
- Larger teacher (8B vs 4B) → stronger distillation signal
- More steps (2000 vs 500) → converged model
- Larger LoRA rank (r=16 vs r=8) → better adaptation
- `torch.compile` enabled → ~10-30% faster training
- Tier 2 verifier sweep uses n=200 → paper-quality confidence intervals

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
| Parameter | laptop (1a/1b) | colab_lite (2) | colab (3) | colab_a100 (4) |
|-----------|---------------|----------------|-----------|----------------|
| `target` (teacher) | Qwen3-0.6B | Qwen3-1.7B | Qwen3-4B | Qwen3-8B |
| `steps` | 10 (smoke) / 100 | 300 | 500 | 2000 |
| `lora_r` | 4 | 4 | 8 | 16 |
| `lora_alpha` | 8 | 8 | 16 | 32 |
| `save_every` | 10 | 25 | 25 | 100 |
| `max_new_tokens` | 32 | 128 | 96 | 128 |
| `n_prompts` (eval) | 5 | 20 | 20 | 50 |
| `K_values` (eval) | [3] | [3,5] | [3,5] | [1,3,5,8] |
| `compile` | false | false | false | true |
| `load_in_4bit` | false | false | false | false |
| `wandb_group` | laptop-gsm8k | colab-lite | colab-t4 | colab-a100 |
| `run_label` | 0.6B-laptop | 1.7B-T4-lite | 4B-T4 | 8B-A100 |
| Results meaningful? | **No** (code check only) | Yes (trends) | Yes (publishable) | Yes (paper) |

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
