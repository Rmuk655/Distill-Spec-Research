# Config Propagation Audit

How YAML settings reach `trainer.py` and `evaluate.py`, what is still
hardcoded in `experiment.py`, and how to change behavior **without editing Python**.

Propagation chain (guarded by `tests/unit/test_yaml_propagation.py`):

```
orchestration/configs/*.yaml
  → _load_config_yaml() / _yaml_cfg_to_hparams()     [Layer 1]
  → train_hparams dict in main()                     [Layer 2]
  → build_steps() → _train_hargs / _ec / _eval_cmd   [Layer 3]
  → trainer.py / evaluate.py subprocess argv
```

There is no `pipeline.py` — the orchestrator is `orchestration/experiment.py`.

---

## YAML-controlled (change in `bases/*.yaml` or leaf config)

### Training → `trainer.py`

| YAML path | CLI flag | Notes |
|-----------|----------|-------|
| `training.lr` | `--lr` | Override CLI: `--lr` |
| `training.steps` | `--steps` | Via `train_steps` in hparams |
| `training.grad_accum` | `--grad_accum` | |
| `training.warmup_steps` | `--warmup_steps` | |
| `training.lora_r` / `lora_alpha` | `--lora_r` / `--lora_alpha` | |
| `training.grad_clip` | `--grad_clip` | |
| `training.teacher_temperature` | `--teacher_temp` | |
| `training.max_new_tokens` | `--max_new_tokens` | |
| `training.seed` | `--seed` | |
| `dataset.train` | `--dataset` | Last-value-wins over per-step gsm8k default |
| `dataset.val_dataset` | `--val_dataset` | val1 pool (`gsm8k_30`, etc.) |
| `health.val_every` | `--val_every` | Fast val |
| `health.slow_val_every` / `slow_val_n` | `--slow_val_every` / `--slow_val_n` | Slow val |
| `health.early_stop_patience` | `--early_stop_patience` | All losses when > 0 |
| `health.unstable_nan_action` | `--nan_action` | Unstable losses only |
| `health.unstable_early_stop_patience` | `--early_stop_patience` | When global patience = 0 |
| `tree_training.tree_K` / `tree_L` | `--tree_K` / `--tree_L` | Tree train steps only |
| `checkpointing.save_every` / `milestone_every` | same | |
| `logging.wandb_*` / `run_label` | same | |
| `hardware.device` / `compile` | `--device` / `--compile` | |

### Evaluation → `evaluate.py`

| YAML path | CLI flag | Notes |
|-----------|----------|-------|
| `evaluation.eval_datasets` | `--datasets` | gsm8k* → Phase 3; rest → Phase 4 |
| `evaluation.modes` | `--modes` | |
| `evaluation.K_values` | `--K` | |
| `evaluation.temperatures` | `--temperature` | |
| `evaluation.n_prompts` | `--n` | Secondary datasets |
| `evaluation.n_prompts_gsm8k` | `--n` | Phase 3 gsm8k count |
| `evaluation.max_tokens` | `--max_tokens` | Full eval (default 50 in bases) |
| `evaluation.max_tokens_smoke` | `--max_tokens` | Smoke only (default 30) |
| `evaluation.L` | `--L` | Draft depth at eval (match `tree_L`; evaluate.py default 8) |
| `evaluation.q_temp` | `--q_temp` | Draft sampling temp in BE verifier (default 1.0) |
| `evaluation.tree_eval_modes` | tree-loss `--modes` (A100) | Full 8-verifier alignment matrix |
| `evaluation.tree_eval_modes_subset` | tree-loss `--modes` (T4/colab) | Non-OT subset when `hw_tier != a100` |
| `evaluation.exclude_modes_when_4bit` | filters `--modes` | Default `[alpha]` when `load_in_4bit: true` |
| `evaluation.task_batch` | `--task_batch` | |
| `online_adapt.K` / `update_every` | `--K` / `--update_every` | All `online_*_adapt` steps |
| `online_adapt.milestone_every` | `--milestone_every` | `online_ebe*` steps |
| `online_adapt.early_stop_patience` | `--early_stop_patience` | `online_ebe*` steps |
| `online_adapt.ebe_block_len` | `--ebe_block_len` | `online_ebe` only |
| `models.family` | `--model_family` | Train + eval |
| `hardware.hw_tier` | `--hw_tier` | |

### Experiment scope

| YAML path | Effect |
|-----------|--------|
| `experiment.losses` | Which losses run |
| `experiment.exclude_losses` | Blacklist |
| `experiment.eval_only` / `train_only` / `light_eval` | Pipeline mode |

---

## Intentionally hardcoded in `experiment.py`

These are **not** bugs — they encode pipeline structure:

| Item | Why hardcoded |
|------|----------------|
| Step IDs (`train_kl_gsm8k`, …) | Pipeline state machine |
| Loss → script mapping (`forward_kl`, `kl_tree`, …) | Registry |
| Phase 1 / 2 / 3 / 4 ordering | Research workflow |
| Smoke defaults (`_steps=10`, `n=3`) | Fast path unless YAML `train_steps` set |
| `_LOSS_STEP_PREFIXES`, `_TREE_PAIRED` | Loss-filter + tree smoke pairing |
| `gsm8k_train.jsonl` per-step default | Overridden by `dataset.train` via `_train_hargs` |
| Per-step `--dataset gsm8k_train.jsonl` | Overridden last by `dataset.train` via `_train_hargs` | See below |

---

## Per-step `gsm8k_train.jsonl` default

Each training step in `build_steps()` still embeds `--dataset …/gsm8k_train.jsonl` in its
command list. That is the **fallback** when no YAML override exists.

When `dataset.train` is set (e.g. `laptop_gpt2.yaml` → wikitext), `_train_hargs` appends
`--dataset <train path>` **last**. `trainer.py` uses argparse last-value-wins, so the YAML
path wins without editing 20+ step definitions.

Same pattern for online adapt: `--prompts` defaults to `gsm8k_train.jsonl` but uses
`dataset.train` when set (`online_adapt` reads `_online_prompts`).

---

## 4-bit alpha exclusion

`alpha` verification runs draft + full teacher in **one** `evaluate.py` process. With
`hardware.load_in_4bit: true`, NF4 dequantization can spike **system RAM** (not just VRAM)
and the kernel OOM-kills the job (exit -9) before Python can catch it.

**Fix (YAML):** `evaluation.exclude_modes_when_4bit: [alpha]` (default). Applied whenever
`load_in_4bit=True`, even if `evaluation.modes` lists `alpha`. Kaggle sets
`modes: [bv, gbv, …]` explicitly; the exclude list documents why.

BE verifiers (`bv`, `gbv`, `traversal`, …) run one mode per subprocess — safe on 4-bit tiers.

---

## How to change eval dataset (correct way)

Edit **one line** in `bases/a100.yaml` (or leaf override):

```yaml
evaluation:
  eval_datasets: [gsm8k_eval]   # held-out items[200:299]
```

Do **not** edit 26 call sites in `experiment.py`. `_ec()` reads `_gsm8k_dataset` from this list.

Paper mode:

```yaml
evaluation:
  eval_datasets: [gsm8k_eval, humaneval, math500, mtbench, alpaca]
```

---

## Tests

```bash
pytest tests/unit/test_yaml_propagation.py -v
pytest tests/unit/test_pipeline_steps.py -v
```

`test_yaml_propagation.py` covers Layers 1–3 for training flags, eval_datasets / eval_L,
online_adapt hargs, tree_eval_modes, and 4-bit mode exclusion.
