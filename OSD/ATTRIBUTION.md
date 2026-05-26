# Attribution

The code in this directory is the **OSD (Online Speculative Decoding)** codebase.

| Field | Value |
|---|---|
| **Source** | https://github.com/LiuXiaoxuanPKU/OSD |
| **Paper** | "Online Speculative Decoding", Xia et al., 2024 — https://arxiv.org/abs/2310.07177 |
| **License** | Apache-2.0 (see original repo) |

## Files modified for this research project

| File | Change |
|---|---|
| `train_qwen3.py` | Rewritten for Qwen3 models; added EBE, reverse-KL, JSD, L1 losses; added `--merge_only` flag for LoRA merging; added crash-safe checkpoint resume |
| `online_serve.py` | Added OOM-safe logit masking before softmax to reduce peak VRAM on 4 GB GPUs |

All other files in this directory are unmodified from the original repository.

## Planned migration

Once the active pipeline run completes (Phase 2 of the research cleanup):
1. `train_qwen3.py` and `online_serve.py` will be moved to `gbv-research/algorithms/`
2. This directory will be converted to a **git submodule** pointing to the original repo:
   ```
   git submodule add https://github.com/LiuXiaoxuanPKU/OSD OSD
   ```
   This makes attribution machine-readable: `.gitmodules` will link directly to the
   upstream repository and commit hash.
