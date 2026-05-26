# SpecDist Launchers

One-command scripts for running the SpecDist pipeline on any compute provider.
Every launcher calls the same underlying command:

```bash
python orchestration/pipeline.py \
    --config  {provider_config} \
    --ckpt_root {persistent_storage} \
    --yes
```

The **config** abstracts the hardware; the **launcher** abstracts the cloud.

---

## Which provider should I use?

| Provider | GPU | VRAM | Session | Cost | Best for |
|---|---|---|---|---|---|
| **Free Colab** | T4 | 15 GB | ~90 min | free | Quick smoke tests, first runs |
| **Free Colab Pro** | A100 | 40 GB | 12 h | ~$10/mo | Overnight runs |
| **Kaggle** | T4/P100 | 15 GB | **12 h** | free (30 h/wk) | **Longer free sessions** |
| **Modal A100** | A100-40GB | 40 GB | unlimited | ~$1.10/h | Paper-quality overnight runs |
| **Modal A10G** | A10G | 24 GB | unlimited | ~$0.76/h | Budget paid runs |
| **RunPod** | RTX 3090+ | 24 GB | unlimited | ~$0.44/h | Cheapest paid option |
| **HF Spaces** | T4/A10G | 15–24 GB | unlimited | ~$0.60/h | HF-integrated experiments |

**Rule of thumb:**
- **Free T4 (Colab / Kaggle)** → use `--config colab` (8B teacher in 4-bit NF4)
- **Paid A100 (Modal / RunPod)** → use `--config server` (8B teacher in bfloat16, faster)
- **Local laptop** → use `--config laptop` (smoke tests only)

---

## Launcher files

| File | Provider | Usage |
|---|---|---|
| `../notebooks/colab_quickstart.ipynb` | Google Colab | Open in Colab, run top to bottom |
| `kaggle.ipynb` | Kaggle Kernels | Upload to Kaggle, run top to bottom |
| `modal_app.py` | Modal.com | `modal run launchers/modal_app.py::run_pipeline` |
| `runpod.sh` | RunPod | `bash runpod.sh` in pod terminal |
| `_provider.py` | All (shared) | Config registry + helper functions |

---

## Switching providers

The only thing that changes is `--config` and `--ckpt_root`.

```python
# _provider.py gives you the right values per environment:
from launchers._provider import PROVIDERS, detect_provider, build_pipeline_cmd

p = detect_provider()                   # auto-detects Colab / Kaggle / Modal / etc.
cmd = build_pipeline_cmd(p, smoke=True) # returns the correct subprocess argv
subprocess.run(cmd)
```

Or manually:

```bash
# Free Colab / Kaggle T4
python orchestration/pipeline.py --config colab   --ckpt_root /content/drive/MyDrive/specdist/checkpoints --yes

# Modal / RunPod A100
python orchestration/pipeline.py --config server  --ckpt_root /vol/checkpoints --yes

# Local smoke test
python orchestration/pipeline.py --config laptop  --smoke --yes
```

---

## Which teacher size for T4? (Option A vs B vs C)

**TL;DR: Always use Option A (QLoRA 8B). The `colab` config does this automatically.**

| Option | Teacher | VRAM | Quality | Verdict |
|---|---|---|---|---|
| **A: QLoRA 8B** | Qwen3-8B (4-bit NF4) | ~9 GB | ⭐⭐⭐ Best | **Use this** |
| B: Kaggle extra time | same as A | same | same | Same option A, just more hours |
| C: 1.7B teacher | Qwen3-1.7B (bf16) | ~4 GB | ⭐ Weak | Ablation only |

**Why A wins on T4:**

- `colab.yaml` already loads the 8B teacher in **4-bit NF4** via `bitsandbytes`.
  Total VRAM: ~5 GB teacher + ~1.2 GB draft + ~2.5 GB LoRA/activations = **~9 GB**.
  T4 has 15 GB → **6 GB headroom**.

- The 4-bit teacher's logits are nearly identical to bfloat16 (NF4 is designed for
  frozen inference — quantisation error averages out over 1000s of training steps).

- A 1.7B teacher (Option C) produces weaker distillation targets. Your GBV vs
  baselines comparison needs a strong teacher to show meaningful block-efficiency
  differences. Use 1.7B only as a fast ablation: "does teacher size matter?"

**To run with the recommended setup:**

```bash
# Colab — just open notebooks/colab_quickstart.ipynb and run all cells.
# The colab config already uses 8B 4-bit NF4 teacher — no extra flags needed.

# Kaggle — open launchers/kaggle.ipynb and set CONFIG = "colab"

# Local smoke test to verify the code path:
python orchestration/pipeline.py --config laptop --smoke --yes
```

---

## Adding a new provider

1. Add a `ProviderConfig` entry to `launchers/_provider.py`
2. Create `launchers/my_provider.{ipynb,py,sh}` using `build_pipeline_cmd(p)` for the command
3. That's it — the pipeline, training, and eval code is the same on all providers

Typical new provider file is < 80 lines.
