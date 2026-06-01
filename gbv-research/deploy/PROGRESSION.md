# SpecDist ML Research Progression

This document describes the standard **3-tier research hierarchy** for running SpecDist
experiments, from laptop crash checks through to paper-quality results.

This mirrors standard MLOps practice for ML research with large language
models: **debug on the smallest hardware, find trends on mid-tier free compute,
confirm on production hardware.**

---

## The 3-Tier Research Hierarchy

| Tier | Hardware | Config | Teacher | Steps | Wall time | W&B group | Purpose |
|------|----------|--------|---------|-------|-----------|-----------|---------|
| 1 | Laptop (RTX 500 Ada) | `laptop` | Qwen3-0.6B | 100 (smoke: 10) | ~25 min (smoke, all losses) / ~2-3 hr (full, all losses) | `laptop-gsm8k` | Code exerciser: does every loss path run? |
| 2 | Kaggle T4x2 / Modal T4 | `kaggle` | Qwen3-8B NF4 | 1000 | ~4-8 h | `kaggle-t4x2` | Exploration: which losses rank best? |
| 3 | A100 | `a100` | Qwen3-8B BF16 | 2000 | ~15-20 min train | `a100` | Confirmation: paper-quality numbers |

> **Why can't Tier 1 produce trends?**
> The laptop teacher (Qwen3-0.6B) is the *same size class* as the draft (Qwen2.5-0.5B).
> Distillation from a near-equal-capacity model produces near-zero signal.
> Tier 1 is purely a code correctness gate. Start looking for trends at Tier 2.

> **Why Kaggle T4x2 for exploration, not Colab?**
> Kaggle provides ~29 GB system RAM vs Colab's ~12 GB. 4-bit NF4 loading of the
> Qwen3-8B teacher requires ~16 GB CPU RAM intermediates — feasible on Kaggle, OOM
> on Colab. Kaggle gives A100-equivalent teacher quality on free-tier hardware.

---

## VRAM Budget by Tier

| Tier | Draft | Teacher | Activations | Total | Headroom |
|------|-------|---------|-------------|-------|----------|
| 1 (laptop) | 1.2 GB | 1.2 GB | ~0.5 GB | ~3 GB | CPU/MPS headroom varies |
| 2 (Kaggle T4x2) | 1.2 GB | 4.5 GB NF4 | ~2.0 GB | **~7.8 GB** on GPU 0 | 8+ GB free; teacher split across both T4s |
| 3 (A100) | 1.2 GB | 16.0 GB BF16 | ~2.5 GB | **~19.7 GB** | ~20 GB free on A100 40 GB |

**Why 4-bit NF4 on Kaggle T4x2?**
Each T4 has 16 GB VRAM. Qwen3-8B in BF16 (~16 GB) won't fit on a single card.
4-bit NF4 reduces teacher VRAM to ~4.5 GB. `device_map="auto"` automatically
splits the teacher across both T4 GPUs — no code changes needed, just select T4 x2
in Kaggle notebook settings. The BF16→NF4 conversion uses ~16 GB of CPU RAM
intermediates; Kaggle's ~29 GB system RAM handles this; Colab's ~12 GB does not.

**Why plain BF16 on A100?**
The A100 (40 GB) fits Qwen3-8B in BF16 directly: mmap from disk to VRAM, zero
CPU RAM spike, zero OOM risk. BF16 also gives a cleaner distillation signal than NF4.

---

## W&B Run Naming

Every training run produces a W&B run named:

```
{run_label}-{loss}_{family}_{teacher}_{steps}steps
```

Examples:
- `0.6B-laptop-kl_qwen_qwen3-0.6b_200steps`        — Tier 1 laptop code check
- `0.6B-laptop-gbv_tree_qwen_qwen3-0.6b_200steps`  — Tier 1 with GBV tree loss
- `8B-kaggle-kl_qwen_qwen3-8b_1000steps`           — Tier 2 Kaggle exploration
- `8B-kaggle-ebe_qwen_qwen3-8b_1000steps`          — Tier 2 with EBE loss
- `8B-modal-ebe_qwen_qwen3-8b_1000steps`           — Tier 2 Modal T4 EBE run
- `8B-a100-kl_qwen_qwen3-8b_2000steps`            — Tier 3 paper confirmation
- `8B-a100-gbv_tree_qwen_qwen3-8b_2000steps`      — Tier 3 with GBV tree loss

This makes it possible to compare across tiers in a single W&B dashboard:
filter by `run_label` prefix to see all EBE runs regardless of teacher size,
or filter by `wandb_group` to see only a specific tier.

The `run_label` is set in each YAML's `logging.run_label` field. It is
automatically forwarded from the YAML through `experiment.py` to `trainer.py`.

---

## Notebooks

Kaggle T4x2 runs use **`deploy/kaggle.ipynb`**:

- **Cell 0** — one-shot bootstrap: installs dependencies, mounts storage, launches training
- **Cell 1** — resume: re-runs from the latest checkpoint after a session restart

Run Cell 0 first. If the session crashes, reopen the notebook and run Cell 1.

> There are no separate quickstart notebooks for Colab or A100. Kaggle T4x2 is
> the primary free-tier compute. A100 runs use `experiment.py` directly.

---

## Standard Experiment Workflow

### Step 1a — Crash test (Tier 1)

```bash
# In gbv-research/
python orchestration/experiment.py --config laptop --smoke --yes
```

Runs **10 training steps, 20 train prompts (capped), K=3, 3 eval prompts, 4 verifier modes**. Total **~8-15 min** on laptop. `--smoke` always auto-resets state AND always re-runs every step even if state says "done" — immune to OneDrive sync races.

What `--smoke` exercises:
- Python imports, CUDA/MPS device detection
- 10 forward + backward passes per loss function (confirms grad flow, no NaN)
- Dataset loading + tokenization (20 prompts — ~0.3s vs ~2min for full 6726)
- One checkpoint save + one validation pass (3 fixed prompts from `gsm8k_5.jsonl`)
- Eval: all 6 verifier modes (alpha, bv, gbv, traversal, specinfer, naive) × 3 prompts each
  (traversal ~4s/prompt, specinfer ~8s/prompt on laptop — kept because smoke must exercise every code path)

**Gate**: no exceptions, loss values are finite numbers.
**If this fails**: fix the code. Do not proceed to 1b until 1a passes.

---

### Step 1b — Full code-path exercise (Tier 1)

```bash
# In gbv-research/
python orchestration/experiment.py --config laptop --yes

# Force clean re-run (wipe all done marks, restart from step 1):
python orchestration/experiment.py --config laptop --yes --restart
```

> **Note**: `--smoke` always auto-resets (no `--restart` needed with smoke). Use `--restart` to force-clean a non-smoke run.

Runs **100 training steps on every non-online loss** with **5 eval prompts per mode**. Should complete in **~2-3 hr**.

> Each loss function (`ebe`, `rev_kl`, `jsd`, `bv_tree`, `gbv_tree`, ...) has unique code. All must be exercised here before T4 promotion — bugs found on laptop are cheap; bugs found on T4 cost GPU-hours.
>
> For a faster (~60-75 min) pipeline-only check: `python orchestration/experiment.py --config laptop --losses kl,kl_tree --yes`

What the full laptop run exercises that smoke does NOT:
- Rolling checkpoint save/load cycle (save_every=20 → 5 saves per loss at 100 steps)
- Milestone checkpoints (milestone_every=100 → fires at step 100 per loss)
- Validation health check (val_every=50 → 2 val passes per loss)
- PPL threshold check (ppl_check_every=100 → 1 check per loss)
- Full merge step (LoRA → base model, per loss)
- All eval modes: alpha, bv, gbv, traversal, specinfer, naive (5 prompts each)
- W&B logging (run appears in wandb.ai under group `laptop-gsm8k`)
- Database writes (results.db)
- Pipeline state machine (done_check files)

**Gate**: all eval modes produce output, W&B run visible, no exceptions.

> **Important**: do NOT interpret block-efficiency numbers from this run.
> The 0.6B teacher is the same size as the draft — distillation signal is
> near-zero. The numbers are structurally meaningless at this level.
> Only check that the eval modes *ran* and produced valid-looking numbers.

**If this fails**: fix the code. Do not run Kaggle until Tier 1 passes.

**🔐 Code Review Gate → Tier 2** (must complete before first Tier 2 run):

> Tier 2 can be Kaggle T4x2 OR Modal T4 — both are valid Tier 2 platforms.

- [ ] Tier 1b passes (all eval modes, no errors)
- [ ] PR opened with all changes since last review
- [ ] Loss function implementation reviewed against paper/spec
- [ ] Verifier implementation reviewed against GBV spec

---

### Step 2 — Tier 2 exploration runs (Kaggle T4x2 or Modal T4)

Kaggle T4x2 and Modal T4 are equivalent Tier 2 platforms — use whichever has quota remaining. For Kaggle, open **`deploy/kaggle.ipynb`** on a **Kaggle T4 x2** session (always T4 x2 — P100 is incompatible with PyTorch 2.10+cu128); for Modal T4, run `experiment.py` directly on the Modal instance.

Run **Cell 0** with `SMOKE = False`. For a quick end-to-end check (~10 min),
run with `SMOKE = True` first before committing to the full 4-8 h run.

If a session crashes, reopen the notebook and run **Cell 1** (resume from
latest checkpoint — worst-case loss is 25 steps).

> **Compute budget**: Kaggle gives 30 GPU-h/week. T4 x2 burns 2× the weekly
> quota, leaving ~15 effective session-hours per week. Plan runs accordingly.
> Tier 2 trains one loss at a time; budget **3-4 h per loss** on Kaggle T4x2.

**What to look for:**
- Loss curve: should decrease steadily (not flatline, not spike)
- Block efficiency: any improvement over the naive baseline?
- W&B: no grad_norm, overfit, or plateau alerts firing

**Gate**: at least one loss function shows a consistent positive trend.
**If no trend**: revisit the loss implementation before moving to Tier 3.
**Cost if wrong**: 4-8 h × 2 seeds here, not 15-20 min × 2 seeds on A100.

**🔐 Code Review Gate → Tier 3** (must complete before any A100 run):
- [ ] Tier 2 shows positive trend on ≥ 2 verifiers
- [ ] Second Tier 2 seed (different `seed:` in YAML) confirms direction
- [ ] PR reviewed: hypothesis matches what the code actually tests
- [ ] PR reviewed: no data leakage (eval prompts not in train set)
- [ ] W&B run URLs recorded in `deploy/RUN_LOG.md`

---

### Step 3 — A100 confirmation runs (Tier 3)

Run on any A100 — IITH, Modal A100, Colab Pro/Pro+, or any cloud A100.

```bash
# In gbv-research/ (on any A100):
python orchestration/experiment.py --config a100 --storage_root /path/to/specdist --yes
```

2000 steps, ~15-20 min training (A100 is ~4× faster than T4).

**What to look for:**
- Block efficiency delta vs naive baseline > noise (> 5% consistently)
- Results reproducible across 2 seeds
- Alpha eval consistent with block efficiency improvement

**Gate**: effect is statistically meaningful, directionally consistent, and
larger than run-to-run variance (compare two seeds).
**These are publishable numbers.** Record the W&B run IDs immediately.

**🔐 Final Review Gate**:
- [ ] Tier 3 confirmed across ≥ 2 seeds
- [ ] Effect > 5% on ≥ 1 verifier × K combination
- [ ] PR reviewed: no bugs introduced since Tier 2 review
- [ ] Hypothesis text finalized — no more code changes before paper submission
- [ ] Ablation plan agreed (what to ablate, which seeds, which K values)

**What's different from Tier 2:**
- Teacher in plain BF16 (vs NF4) → stronger, cleaner distillation signal
- More steps (2000 vs 1000) → more converged model
- Larger LoRA rank (r=16 vs r=8) → better adaptation
- `torch.compile` enabled → ~10-30% faster training
- Full GSM8K test set (n=1319) → paper-quality confidence intervals
- 9-verifier eval matrix → complete loss × verifier alignment table

---

## Automated W&B Alerts

Four health checks fire W&B alerts automatically during training:

| Alert | Condition | What it means |
|-------|-----------|---------------|
| **Grad norm spike** | `grad_norm > 10` | Training instability; check LR or data batch |
| **Overfitting** | `overfit_ratio > 1.3` | Train loss diverging from val loss; possible memorization |
| **Val plateau** | val loss flat for N consecutive checks | Model stopped learning; consider LR or loss change |
| **Slow convergence** | loss barely moved at 10% of total steps | Run may be wasted; verify hyperparameters before continuing |

These alerts fire at all tiers. They are especially useful during long Kaggle
sessions (4-8 h) where continuous terminal monitoring is not practical.

---

## What Changes Between Tiers (What Stays the Same)

### Same across all tiers:
- Codebase (orchestration/, algorithms/, core/)
- Pipeline steps (train → merge → eval)
- Loss functions (KL, EBE, GBV, tree variants, etc.)
- Verifier implementations (SpecInfer, GBV, traversal, etc.)
- Dataset (GSM8K train/eval split)
- W&B project (`distillspec`)
- Crash-safe checkpoint resume

### Varies by tier:

| Parameter | Tier 1 (laptop) | Tier 2 (Kaggle T4x2 / Modal T4) | Tier 3 (A100) |
|-----------|-----------------|----------------------|--------------------|
| `target` (teacher) | Qwen3-0.6B | Qwen3-8B NF4 | Qwen3-8B BF16 |
| `steps` | 200 (smoke: 10) | 1000 | 2000 |
| `lora_r` | 4 | 8 | 16 |
| `lora_alpha` | 8 | 16 | 32 |
| `save_every` | 20 | 25 | 100 |
| `max_new_tokens` | 32 | 96 | 128 |
| `n_prompts` (eval) | 10 | 10 | 100 (GSM8K: 1319) |
| `K_values` (eval) | [3] | [3] | [3] |
| `compile` | false | false | true |
| `load_in_4bit` | false | true | false |
| `wandb_group` | laptop-gsm8k | kaggle-t4x2 | a100 |
| `run_label` | 0.6B-laptop | 8B-kaggle | 8B-a100 |
| Results meaningful? | **No** (code check only) | Yes (trends) | Yes (paper) |

---

## Comparing Tiers in W&B

In the W&B project `distillspec`, use these filters:

**See all runs at a given tier:**
```
group:laptop-gsm8k    ← all Tier 1 runs
group:kaggle-t4x2     ← all Tier 2 Kaggle runs
group:modal-t4        ← all Tier 2 Modal runs (once modal.yaml is configured)
group:a100            ← all Tier 3 A100 runs
```

**Compare a single loss across teacher sizes:**
```
run_label:8B-kaggle AND loss:ebe    ← Tier 2 EBE
run_label:8B-a100 AND loss:ebe      ← Tier 3 EBE
```

**Compare losses at the same tier:**
```
group:kaggle-t4x2    → overlay kl, ebe, gbv, jsd, l1 curves
```

**Filter to a specific pipeline session (experiment tag):**
```
tags contains kri655-20260601_1041    ← one specific pipeline session
```

---

## Adding a New Config Tier

If you need a new hardware tier (e.g. Modal T4, Lambda Labs A10G):

1. Copy the nearest YAML (e.g. `cp kaggle.yaml modal.yaml`)
2. Update `hardware:`, `models:`, `training:`, `checkpointing:`, `logging:`
3. Set `logging.run_label` to something like `"8B-modal"` or `"8B-A10G"`
4. Set `logging.wandb_group` to something like `"modal-t4"` or `"lambda-a10g"`
5. The pipeline, trainer, and W&B wiring pick up the new config automatically

No code changes needed — the config is self-contained.
