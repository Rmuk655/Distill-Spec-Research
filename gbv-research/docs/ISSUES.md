# Known Issues

> **Disabled losses** are excluded from all free-tier configs (`kaggle.yaml`, `colab.yaml`, `a100.yaml`,
> `laptop.yaml`, `laptop_gpt2.yaml`, `laptop_llama.yaml`) via `exclude_losses:` in the YAML.
> Re-enable once the root cause is fixed.
>
> **W&B note:** laptop and smoke runs have `logging.no_wandb: true` — no W&B runs are
> created. This is intentional (code exerciser only). Kaggle/Colab/A100 configs still log to W&B.

## 1. EBE Loss Broken

**Symptom**: BE delta -0.202 vs baseline on laptop (bv/K=3: 2.846 vs baseline 3.048).

**Evidence**: Perplexity looks fine (7.96 vs 7.47 baseline) — model isn't diverging, it's simply not learning the right objective. The language modeling signal is intact but the EBE signal isn't propagating usefully.

**Disabled in**: `ebe`, `ebe_single`, `ebe_tree` (all free-tier configs: `kaggle.yaml`, `colab.yaml`, `a100.yaml`).

**Investigation pointer**: Check loss gradient computation in [`algorithms/distillspec_gbv/trainer.py`](../algorithms/distillspec_gbv/trainer.py) — verify that EBE gradients are non-zero and correctly scaled relative to the LM loss.

---

## 2. Online Loss Diverged

**Symptom**: Perplexity 15.46 vs baseline 7.47 (~2×); BE delta -0.679 (worst of all losses tested).

**Evidence**: Both metrics degrade together, indicating true training divergence rather than a metric artifact.

**Hypothesis**: LR too high (online training uses 10× the offline LR), reward-scale miscalibration, or a bug in KL computation at reject positions.

**Disabled in**: `online`, `online_ebe`, `online_ebe_single`, `online_kl_tree`, `online_ebe_tree` (all free-tier configs).

**Investigation pointer**: Check [`orchestration/experiment.py`](../orchestration/experiment.py) `online_adapt` section and [`algorithms/online_serve.py`](../algorithms/online_serve.py) — confirm LR schedule, reward normalization, and KL mask at reject positions.

---

## 3. GPT-2 BE Eval Not Supported (alpha only)

**Symptom**: All BE modes (bv, gbv, traversal, specinfer, naive) crash with `AttributeError: 'GPT2LMHeadModel' object has no attribute 'model'` followed by cache format errors.

**Root cause**: The verifier code (`verifiers/draft_generator.py`, `verifiers/utils.py`) is built on two Qwen3-specific interfaces:
1. **Custom KV cache** — `.layers[i].keys/.values` accessed by `slice_cache()`, `expand_cache()`, and `target_tree_pass()`. GPT-2 returns a legacy tuple `((k1,v1), (k2,v2), ...)` which has no `.layers` attribute.
2. **Custom tree attention mask** — `attention_mask={"full_attention": tensor}` passed in `target_tree_pass()`. GPT-2 expects a standard 4D tensor, not a dict.

**Impact**: GPT-2 configs (`laptop_gpt2.yaml`, `server_gpt2.yaml`) evaluate with `modes: [alpha]` only. Alpha (token acceptance rate) is sufficient for convergence trend detection.

**Alpha already works** — `alpha=0.1658` for untrained baseline, and improves with training. For proving "our loss > forward_kl", alpha curves across training steps are sufficient.

**To fix later** (for publishable BE results on GPT-2):
- Refactor `slice_cache()` and `expand_cache()` in `verifiers/utils.py` to handle both tuple and DynamicCache formats
- Update `target_tree_pass()` to use architecture-specific attention mask format
- Alternatively: require GPT-2 to use `config.use_cache=True` with explicit DynamicCache conversion
