# Experiment Profiles — A100 Research Workflow

Profiles are self-contained YAML configs that control what the pipeline runs.
Set `experiment: losses:` to pick which losses train/eval, and `experiment: eval_only: true`
to skip training entirely and only run evaluation.

## Two Parallel Tracks

The A100 work splits into two independent tracks that share the same baseline
(Track A Tier 1) for the paper comparison table.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│ TRACK A — Flat distillation losses (KL, JSD, L1, online; standard objectives)  │
│ TRACK B — Tree-structured loss family (on-policy; novel contribution)           │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Track A — Flat Losses (3-Tier)

```
┌─────────────────────────────────────────────────────────────────┐
│ Tier A-1 — a100_baseline_losses                                 │
│   Train KL, JSD, L1, online KL (well-known distillation algos) │
│   → Produces 4 merged model checkpoints                         │
│   → Establishes comparison floor for ALL novel losses           │
│   Time: ~2-3 h on A100                                          │
└───────────────────────────────┬─────────────────────────────────┘
                                │ checkpoints exist
                    ┌───────────┴──────────────┐
                    ▼                          ▼
┌───────────────────────────┐  ┌──────────────────────────────────┐
│ Tier A-2 — verifier_sweep │  │ Tier A-3 — new_loss_template     │
│   eval_only: true          │  │   Train one new flat loss        │
│   No training. Re-evals    │  │   (copy template, fill in name)  │
│   all 4 baselines at       │  │   Eval new loss + baselines      │
│   n=200 across all         │  │   across all 6 verifiers         │
│   6 verifier algorithms    │  │   → Did the new idea win?        │
│   → Which verifier works   │  │   Time: ~2-3 h on A100           │
│     best with which loss?  │  └──────────────────────────────────┘
│   Time: ~1-2 h on A100    │
└───────────────────────────┘
```

### Track B — Tree-Structured Loss Family (1-Tier)

```
┌─────────────────────────────────────────────────────────────────────┐
│ Tier B-1 — a100_tree_losses                                         │
│                                                                     │
│   7 offline tree losses:                                            │
│     kl_tree, rev_kl_tree, jsd_tree (universal divergence)          │
│     bv_tree, gbv_tree, traversal_tree (verifier-specific)          │
│     ebe_tree (off-policy mismatch ablation)                         │
│                                                                     │
│   2 online tree variants:                                           │
│     online_kl_tree, online_ebe_tree                                 │
│                                                                     │
│   Evaluates ONLY with non-OT verifiers: bv, gbv, traversal         │
│   (tree losses directly optimise these criteria; OT verifiers       │
│   are out-of-distribution for tree-trained models)                  │
│                                                                     │
│   Prerequisite: Tier A-1 (for comparison baseline numbers)         │
│   Time: ~4-6 h on A100 (9 losses × 2000 steps)                    │
└─────────────────────────────────────────────────────────────────────┘
```

**Key hypotheses answered by Track B** (see GUIDE.md §1):
- **H5**: Tree losses > flat counterparts on matching non-OT verifier?  
  (bv_tree vs kl on bv_verify, gbv_tree vs kl on gbv_verify, etc.)
- **H6**: Online tree > flat online on non-OT verifiers?

## Available Profiles

| Profile | Track | Tier | Purpose | Training? |
|---------|-------|------|---------|-----------|
| `profiles/a100_baseline_losses` | A | A-1 | Train KL/JSD/L1/online; establish baselines | Yes |
| `profiles/a100_verifier_sweep` | A | A-2 | Eval-only; deep verifier comparison at n=200 | No |
| `profiles/a100_new_loss_template` | A | A-3 | Template: train a new flat loss vs baselines | Yes (new loss only) |
| `profiles/a100_tree_losses` | B | B-1 | Full tree loss family; non-OT verifiers only | Yes |
| `profiles/online_only_laptop` | dev | — | Laptop online-KL debug without offline losses | Yes (online only) |
| `profiles/promote_kl_jsd_l1` | legacy | — | Superseded by a100_baseline_losses | Yes |

## Usage

```bash
# ── Track A, Tier 1: train flat baseline losses ───────────────────────────────
python orchestration/experiment.py --config profiles/a100_baseline_losses --smoke --yes \
  --storage_root /content/drive/MyDrive/specdist    # smoke check first
python orchestration/experiment.py --config profiles/a100_baseline_losses --yes \
  --storage_root /content/drive/MyDrive/specdist

# ── Track A, Tier 2: deep verifier sweep (run after Tier A-1) ────────────────
python orchestration/experiment.py --config profiles/a100_verifier_sweep --yes \
  --storage_root /content/drive/MyDrive/specdist

# ── Track A, Tier 3: new flat loss (copy template first) ─────────────────────
cp orchestration/configs/profiles/a100_new_loss_template.yaml \
   orchestration/configs/profiles/a100_ebe_v2.yaml
# edit losses: [ebe, kl, jsd, l1, online]
python orchestration/experiment.py --config profiles/a100_ebe_v2 --yes \
  --storage_root /content/drive/MyDrive/specdist

# ── Track B, Tier 1: full tree loss family ────────────────────────────────────
python orchestration/experiment.py --config profiles/a100_tree_losses --smoke --yes \
  --storage_root /content/drive/MyDrive/specdist    # smoke check first
python orchestration/experiment.py --config profiles/a100_tree_losses --yes \
  --storage_root /content/drive/MyDrive/specdist

# ── Track B: offline tree losses only (skip online variants) ─────────────────
python orchestration/experiment.py --config profiles/a100_tree_losses --yes \
  --losses kl_tree,rev_kl_tree,jsd_tree,bv_tree,gbv_tree,traversal_tree,ebe_tree \
  --storage_root /content/drive/MyDrive/specdist

# ── Override losses on the fly (no config file needed) ───────────────────────
python orchestration/experiment.py --config colab_a100 --losses kl,jsd --yes
python orchestration/experiment.py --config colab_a100 --eval_only --yes
```

## `experiment:` Section Reference

```yaml
experiment:
  losses: [kl, jsd, l1, online]   # Subset of ALL_LOSSES to train + eval.
                                    # Flat losses:  kl ebe ebe_single rev_kl jsd l1
                                    #               online online_ebe online_ebe_single
                                    # Tree losses:  kl_tree rev_kl_tree jsd_tree
                                    #               bv_tree gbv_tree traversal_tree ebe_tree
                                    #               online_kl_tree online_ebe_tree
                                    # Omit or set to null to run all losses.
                                    # CLI --losses always overrides this.

  eval_only: true                   # Skip Phase 2 (train + merge) entirely.
                                    # Only Phase 1 baseline + Phase 3/4 evals run.
                                    # Requires merged model checkpoints to exist.
                                    # CLI --eval_only always overrides this.

  seed_override: 123                # Override training.seed without editing the base config.
                                    # Useful for second-seed reproducibility runs.
```

```yaml
tree_training:                      # Only needed for *_tree losses
  tree_K: 4     # i.i.d. draft paths per step; keep ≤ 4 for gbv_tree stability
  tree_L: 8     # draft block depth; must match evaluation.L
```

## Adding a New Profile

```bash
# Option A — new flat loss profile (most common):
cp profiles/a100_new_loss_template.yaml profiles/a100_my_loss.yaml
# edit: set losses, wandb_group, run_label

# Option B — new tree loss profile:
cp profiles/a100_tree_losses.yaml profiles/a100_my_tree_loss.yaml
# edit: add new loss to experiment.losses list

# Option C — from scratch:
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

**Tree losses** use `modes=bv,gbv,traversal` for all eval steps (hardcoded in experiment.py
as `_TREE_NON_OT`), regardless of the `evaluation.modes` config value.
