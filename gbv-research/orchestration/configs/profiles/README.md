# Experiment Profiles — A100 3-Tier Workflow

Profiles are self-contained YAML configs that control what the pipeline runs.
Set `experiment: losses:` to pick which losses train/eval, and `experiment: eval_only: true`
to skip training entirely and only run evaluation.

## The 3-Tier A100 Research Workflow

Each tier builds on the previous. Run them in order on an A100 Colab session.

```
┌─────────────────────────────────────────────────────────────────┐
│ Tier 1 — a100_baseline_losses                                   │
│   Train KL, JSD, L1, online KL (well-known distillation algos) │
│   → Produces 4 merged model checkpoints                         │
│   → Establishes the comparison floor for all novel losses       │
│   Time: ~2-3 h on A100                                          │
└───────────────────────────────┬─────────────────────────────────┘
                                │ checkpoints exist
                    ┌───────────┴──────────────┐
                    ▼                          ▼
┌───────────────────────────┐  ┌──────────────────────────────────┐
│ Tier 2 — verifier_sweep   │  │ Tier 3 — new_loss_template       │
│   eval_only: true          │  │   Train one new loss             │
│   No training. Re-evals    │  │   (copy template, fill in name)  │
│   all 4 baselines at       │  │   Eval new loss + baselines      │
│   n=200 across all         │  │   across all 6 verifiers         │
│   6 verifier algorithms    │  │   → Did the new idea win?        │
│   → Which verifier works   │  │   Time: ~2-3 h on A100           │
│     best with which loss?  │  └──────────────────────────────────┘
│   Time: ~1-2 h on A100    │
└───────────────────────────┘
```

## Available Profiles

| Profile | Tier | Purpose | Training? |
|---------|------|---------|-----------|
| `profiles/a100_baseline_losses` | 1 | Train KL/JSD/L1/online; establish baselines | Yes |
| `profiles/a100_verifier_sweep` | 2 | Eval-only; deep verifier comparison at n=200 | No |
| `profiles/a100_new_loss_template` | 3 | Template: train a new loss, compare vs baselines | Yes (new loss only) |
| `profiles/online_only_laptop` | dev | Mukund: laptop online-KL debug without offline losses | Yes (online only) |
| `profiles/promote_kl_jsd_l1` | legacy | Superseded by a100_baseline_losses | Yes |

## Usage

```bash
# ── Tier 1: train baseline losses ────────────────────────────────────────────
# Smoke test first (catches crashes in ~5 min before committing to 2-3 h run):
python orchestration/experiment.py --config profiles/a100_baseline_losses --smoke --yes \
  --storage_root /content/drive/MyDrive/specdist

# Full run:
python orchestration/experiment.py --config profiles/a100_baseline_losses --yes \
  --storage_root /content/drive/MyDrive/specdist

# ── Tier 2: deep verifier sweep (run after Tier 1 completes) ─────────────────
python orchestration/experiment.py --config profiles/a100_verifier_sweep --yes \
  --storage_root /content/drive/MyDrive/specdist

# ── Tier 3: new loss (copy template first, fill in loss name) ────────────────
cp orchestration/configs/profiles/a100_new_loss_template.yaml \
   orchestration/configs/profiles/a100_ebe_v2.yaml
# edit a100_ebe_v2.yaml: set losses: [ebe, kl, jsd, l1, online]
python orchestration/experiment.py --config profiles/a100_ebe_v2 --yes \
  --storage_root /content/drive/MyDrive/specdist

# ── Override losses on the fly (no config file needed) ───────────────────────
python orchestration/experiment.py --config colab_a100 --losses kl,jsd --yes
python orchestration/experiment.py --config colab_a100 --eval_only --yes   # skip all training
```

## `experiment:` Section Reference

```yaml
experiment:
  losses: [kl, jsd, l1, online]   # Subset of ALL_LOSSES to train + eval.
                                    # ALL_LOSSES: kl ebe ebe_single rev_kl jsd l1
                                    #             online online_ebe online_ebe_single
                                    # Omit or set to null to run everything.
                                    # CLI --losses always overrides this.

  eval_only: true                   # Skip Phase 2 (train + merge) entirely.
                                    # Only Phase 1 baseline + Phase 3/4 evals run.
                                    # Requires merged model checkpoints to exist.
                                    # CLI --eval_only always overrides this.

  seed_override: 123                # Override training.seed without editing the base config.
                                    # Useful for second-seed reproducibility runs.
```

## Adding a New Profile

```bash
# Option A — new loss profile (most common):
cp profiles/a100_new_loss_template.yaml profiles/a100_my_loss.yaml
# edit: set losses, wandb_group, run_label

# Option B — from scratch:
cp profiles/a100_baseline_losses.yaml profiles/my_ablation.yaml
# edit the experiment: section
```

## What the Pipeline Runs Per Phase

| Phase | Content | Smoke? | eval_only? |
|-------|---------|--------|------------|
| Phase 1 | Baseline eval (untrained draft) on gsm8k | ✅ runs | ✅ runs |
| Phase 2 | train + merge for each selected loss | ✅ (10 steps) | ❌ skipped |
| Phase 3 | eval each loss on gsm8k | ✅ (n=5) | ✅ runs |
| Phase 4 | eval each loss on humaneval/math500/mtbench/alpaca | ❌ skipped | ✅ runs |
