# Attribution

The code in this directory is the **OSD (Online Speculative Decoding)** codebase.

| Field | Value |
|---|---|
| **Source** | https://github.com/LiuXiaoxuanPKU/OSD |
| **Paper** | "Online Speculative Decoding", Xia et al., 2024 — https://arxiv.org/abs/2310.07177 |
| **License** | Apache-2.0 (see original repo) |

## Files modified for this research project

These files were modified and have since been moved to `gbv-research/algorithms/`
(Phase 2 migration — commit `16f0560`). This directory now contains only the
original, unmodified OSD codebase.

| File | Change | Current location |
|---|---|---|
| `train_qwen3.py` | Rewritten for Qwen3 models; added EBE, reverse-KL, JSD, L1 losses; added `--merge_only` flag; added crash-safe checkpoint resume | `gbv-research/algorithms/train_qwen3.py` |
| `online_serve.py` | Added OOM-safe logit masking before softmax to reduce peak VRAM on 4 GB GPUs | `gbv-research/algorithms/online_serve.py` |

All files currently in this directory are unmodified from the original repository.

## Future migration

This directory will be converted to a **git submodule** pointing to the original repo
after the current pipeline run completes:
```
git submodule add https://github.com/LiuXiaoxuanPKU/OSD OSD
```
This makes attribution machine-readable: `.gitmodules` will link directly to the
upstream repository and commit hash.
