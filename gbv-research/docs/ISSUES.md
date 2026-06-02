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

## 3. GPT-2 BE Eval — FIXED (CompatCache adapter)

---

## 4. specinfer + naive OOM on 4 GB Laptop GPU

**Symptom**: `specinfer` and `naive` verifier modes crash after 60-90 seconds with a native Windows process abort (exit code 0xC000013A — GPU driver kills the process). `bv`, `gbv`, `traversal`, and `alpha` all work correctly on the same hardware.

**Root cause**: Both are OTLP-based verifiers (`otlp_registry.py`). Their verification step (after `target_tree_pass`) uses scipy/numpy optimal-transport solvers. The solver allocates additional tensors beyond model weights + KV cache, pushing past the available VRAM headroom on a 4 GB GPU.

**Impact**: specinfer and naive are excluded from all laptop configs (`laptop.yaml`, `laptop_gpt2.yaml`, `laptop_llama.yaml`) and the smoke eval. They run correctly on T4 (15 GB) and A100 (40 GB).

**What you still get on laptop** — sufficient for convergence proof and GBV research:
- `alpha` (token acceptance rate) ✓
- `bv` (block verification baseline) ✓
- `gbv` (our method — **key research metric**) ✓
- `traversal` (tree traversal baseline) ✓

**specinfer + naive will be verified on**: `colab.yaml`, `kaggle.yaml`, `a100.yaml` (all run on T4/A100).

## 3. GPT-2 BE Eval — FIXED (CompatCache adapter)

**Was**: All BE modes crashed because the verifier assumed Qwen3-specific interfaces:
1. KV cache: `.layers[i].keys/.values` — GPT-2 returns a legacy tuple
2. Attention mask: `{"full_attention": tensor}` dict — GPT-2 needs raw 4D tensor

**Fix applied** (see commit history): Added `CompatCache` adapter class in `verifiers/utils.py`
that normalises any model's `past_key_values` to `.layers[i].keys/.values` and reconstructs
the model-native format for each forward call. Added `_attn_mask_for_model()` to detect
whether the model expects a dict or tensor. GPT-2, LLaMA, Gemma and Qwen3 all supported.

**All 6 verifier modes now work for all model families.**
