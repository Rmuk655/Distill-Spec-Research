# Attribution

The code in the **`OSD/`** directory is the **OSD (Online Speculative Decoding)** codebase.
This file lives at the repo root because `OSD/` is a git submodule (the submodule
directory is owned by the upstream repo and cannot hold our custom files).

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
| `train_qwen3.py` | **Written from scratch for this project** — not derived from any OSD file. OSD's training code (`distill/train.py`) uses Hugging Face Trainer + fastchat dataclasses; our script uses a raw PyTorch loop and was designed from the ground up for DistillSpec + EBE. | `gbv-research/algorithms/train_qwen3.py` |
| `online_serve.py` | Added OOM-safe logit masking before softmax to reduce peak VRAM on 4 GB GPUs | `gbv-research/algorithms/online_serve.py` |

All files currently in this directory are unmodified from the original repository.

## Git submodule

`OSD/` is tracked as a **git submodule** pointing to the original upstream repository:
```
git submodule add https://github.com/LiuXiaoxuanPKU/OSD OSD
```
The `.gitmodules` file links directly to the upstream repository and pinned commit hash,
making attribution machine-readable. To initialise after cloning this repo:
```
git submodule update --init OSD
```
