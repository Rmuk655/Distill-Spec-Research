# SpecDist Deployment

One-command scripts for running the SpecDist pipeline on any compute provider.
Every launcher calls the same underlying command:

```bash
python orchestration/experiment.py \
    --config  {provider_config} \
    --ckpt_root {persistent_storage} \
    --yes
```

The **config** abstracts the hardware; the **launcher** abstracts the cloud.

---

## Which provider should I use?

| Provider | GPU | VRAM | System RAM | Session | Cost | Best for |
|---|---|---|---|---|---|---|
| **Free Colab** | T4 | 16 GB | ~12 GB | ~90 min idle | free | Quick smoke tests, first runs |
| **Kaggle** | T4 (×1 or ×2) | 16 GB | **29 GB** | **9 h (60 min idle)** | free (30 h/wk) | **Best free option (no persistent storage)** |
| **Lightning AI** | T4-equiv | 16 GB | ~32 GB | unlimited | free (15 credits/mo ≈ 22-30 h) | **Best free option with persistent storage** |
| **Adobe AIP** | A100 / varies | ≥ 16 GB | ample | **4 h guaranteed** | free (internal) | **Best option if you have AIP access** |
| **Colab Pro** | A100 | 40 GB | ~50 GB | 12 h | ~$10/mo | Overnight paper runs |
| **Modal A100** | A100-40GB | 40 GB | ample | unlimited | ~$1.10/h | Paper-quality overnight runs |
| **Modal A10G** | A10G | 24 GB | ample | unlimited | ~$0.76/h | Budget paid runs |
| **RunPod** | RTX 3090+ | 24 GB | ample | unlimited | ~$0.44/h | Cheapest paid option |
| **Lightning AI** | L40S | 48 GB | ample | unlimited | ~$1.10-1.50/h | Paid: VS Code + A100-class GPU |
| **HF Spaces** | T4/A10G | 15–24 GB | ample | unlimited | ~$0.60/h | HF-integrated experiments |

**Rule of thumb:**
- **Kaggle T4** → use `--config kaggle` (8B teacher in 4-bit NF4; 29 GB RAM makes NF4 loading work)
- **Lightning AI free T4** → use `--config kaggle` (same RAM headroom as Kaggle; storage persists — models download once)
- **Free Colab T4** → use `--config colab` (4B teacher in plain BF16; Colab's 12 GB RAM can't load 8B NF4)
- **Adobe AIP A100** → use `--config a100` (8B BF16, full run in ~2-3 h — fits one 4-hour session; see `deploy/aip.ipynb`)
- **Adobe AIP T4/V100 16 GB** → use `--config kaggle` (8B NF4; 4-hour sessions mean ~1 loss/session)
- **Paid A100 / L40S (Modal / RunPod / Lightning AI)** → use `--config a100` or `server` (8B teacher in bfloat16, no quantization)
- **Local laptop** → use `--config laptop` (smoke tests only)

---

## Deployment files

| File | Provider | Usage |
|---|---|---|
| `colab_quickstart.ipynb` | Google Colab | Open in Colab, run top to bottom |
| `kaggle.ipynb` | Kaggle Kernels | Upload to Kaggle, run top to bottom |
| `lightning.ipynb` | Lightning AI Studios | Open in Studio Jupyter, run top to bottom |
| `aip.ipynb` | Adobe AI Platform | Open in AIP VS Code Jupyter, run top to bottom |
| `modal_app.py` | Modal.com | `modal run deploy/modal_app.py::run_pipeline` |
| `runpod.sh` | RunPod | `bash runpod.sh` in pod terminal |
| `_provider.py` | All (shared) | Config registry + helper functions |

---

## Switching providers

The only thing that changes is `--config` and `--ckpt_root`.

```python
# _provider.py gives you the right values per environment:
from deploy._provider import PROVIDERS, detect_provider, build_pipeline_cmd

p = detect_provider()                   # auto-detects Colab / Kaggle / Modal / etc.
cmd = build_pipeline_cmd(p, smoke=True) # returns the correct subprocess argv
subprocess.run(cmd)
```

Or manually:

```bash
# Free Colab / Kaggle T4
python orchestration/experiment.py --config colab   --ckpt_root /content/drive/MyDrive/specdist/checkpoints --yes

# Modal / RunPod A100
python orchestration/experiment.py --config server  --ckpt_root /vol/checkpoints --yes

# Local smoke test
python orchestration/experiment.py --config laptop  --smoke --yes
```

---

## Which teacher size for T4?

| Option | Config | Teacher | VRAM | Quality | Platform |
|---|---|---|---|---|---|
| **8B NF4 (best)** | `kaggle` | Qwen3-8B (4-bit NF4) | ~7.8 GB | ⭐⭐⭐ Best | **Kaggle only** (29 GB RAM) |
| **4B BF16 (safe)** | `colab` | Qwen3-4B (BF16) | ~10.7 GB | ⭐⭐ Good | Colab or Kaggle |
| **1.7B BF16 (fast)** | `colab_lite` | Qwen3-1.7B (BF16) | ~5.6 GB | ⭐ Weak | Colab or Kaggle |

**Why `colab` uses 4B instead of 8B:**

4-bit NF4 quantization loads each weight tensor as BF16 into CPU RAM first (~16 GB
intermediates for 8B model). Colab has ~12 GB system RAM → OOM kill (SIGKILL, no
Python exception). `colab.yaml` uses the 4B teacher in plain BF16 to avoid this.

Kaggle has **29 GB system RAM** → 16 GB intermediates fit → 8B NF4 works.

**Recommendation:**
- **Kaggle:** use `CONFIG = "kaggle"` (8B NF4, same teacher quality as A100)
- **Free Colab:** use `CONFIG = "colab"` (4B BF16, no quantization risk)
- Use 1.7B (`colab_lite`) only for quick ~25-min trend checks

```bash
# Kaggle — open deploy/kaggle.ipynb and set CONFIG = "kaggle" (default)

# Colab — open deploy/colab_quickstart.ipynb and set CONFIG = "colab" (default)

# Local smoke test to verify the code path:
python orchestration/experiment.py --config laptop --smoke --yes
```

---

## Adding a new provider

1. Add a `ProviderConfig` entry to `deploy/_provider.py`
2. Create `deploy/my_provider.{ipynb,py,sh}` using `build_pipeline_cmd(p)` for the command
3. That's it — the pipeline, training, and eval code is the same on all providers

Typical new provider file is < 80 lines.
