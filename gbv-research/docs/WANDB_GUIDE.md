# W&B Guide for SpecDist Researchers

## Key config fields (Group By / Filter by these)

| Field | Values | Use for |
|-------|--------|---------|
| `config.loss` | kl, ebe, l1, jsd, reverse_kl, bv_tree, kl_tree, ebe_tree | Compare loss functions |
| `config.run_mode` | smoke, full | Filter out smoke runs (always filter to `full` for real results) |
| `config.hw_tier` | laptop, colab, kaggle, a100 | Filter by compute tier |
| `config.steps` | 10, 100, 1000 | Compare training length |
| `config.seed` | 42, … | Check reproducibility |
| `config.lr` | 3e-4, … | LR ablation |
| `config.lora_r` | 4, 8, 16 | LoRA rank ablation |
| `config.teacher_temp` | 0.6, 0.8, 1.0 | Temperature sensitivity |
| `job_type` | train, eval | Separate training from evaluation runs |
| tags | smoke, full, kl, eval, laptop, … | Fast search/filter |

## Recommended default filter
Always add: `config.run_mode = full` to exclude smoke runs from charts.

## Top 20 researcher questions → W&B recipe

| # | Question | Filter | Group by | Metric |
|---|----------|--------|----------|--------|
| 1 | Which loss has lowest final val loss? | run_mode=full, job_type=train | loss | val/loss (last) |
| 2 | Does each loss train without overfitting? | run_mode=full, job_type=train | loss | train/loss + val/loss overlay |
| 3 | Which loss gives best block efficiency? | run_mode=full, job_type=eval | loss | summary/best_BE |
| 4 | Does LR decay correctly? | run_mode=full, job_type=train | loss | train/lr |
| 5 | Are gradients healthy (no spikes)? | run_mode=full, job_type=train | loss | gradients/lora_grad_norm |
| 6 | Which loss converges fastest? | run_mode=full, job_type=train | loss | val/loss (x=train_step) |
| 7 | Does tree loss outperform flat KL? | run_mode=full | loss (filter: kl vs kl_tree) | val/loss + best_BE |
| 8 | Does warmup ratio matter? | run_mode=full, job_type=train | loss | train/lr first 10% of steps |
| 9 | Which hw_tier reproduces best? | run_mode=full | hw_tier | val/loss (compare same loss) |
| 10 | Does LoRA rank matter? | run_mode=full | lora_r | val/loss |
| 11 | Is alpha distribution healthy? | run_mode=full, job_type=eval | loss | artifacts: per_prompt_alpha |
| 12 | Does teacher temp affect quality? | run_mode=full | teacher_temp | val/loss + best_BE |
| 13 | Which verifier mode gives best BE? | run_mode=full, job_type=eval | — | eval_summary_table |
| 14 | Does seed affect results significantly? | run_mode=full | seed | val/loss (same loss, different seeds) |
| 15 | How does BE scale with training steps? | run_mode=full, job_type=eval | steps | best_BE |
| 16 | Is train loss decreasing monotonically? | run_mode=full, job_type=train | loss | train/loss (smoothed) |
| 17 | Does EMA loss track raw loss? | run_mode=full, job_type=train | loss | train/loss + train/loss_ema |
| 18 | Is VRAM usage within budget? | any | hw_tier | train/peak_vram_mb |
| 19 | Did any run trigger a W&B alert? | filter by alerts tag | — | alerts panel |
| 20 | Smoke vs full: are code paths consistent? | group run_mode | run_mode | val/loss at step 10 |

## Group by recommendation
Use **`config.loss`** as your primary group-by for training runs.
Use **`config.run_mode`** as a secondary filter (exclude smoke for real results).
Avoid grouping by `config.draft` or `config.target` — these are fixed per experiment.
