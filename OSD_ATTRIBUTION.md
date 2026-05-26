# Attribution

The code in the **`OSD/`** directory is the **OSD (Online Speculative Decoding)** codebase.
This file lives at the repo root because `OSD/` is a git submodule (the submodule
directory is owned by the upstream repo and cannot hold our custom files).

| Field | Value |
|---|---|
| **Source** | https://github.com/LiuXiaoxuanPKU/OSD |
| **Paper** | "Online Speculative Decoding", Xia et al., 2024 — https://arxiv.org/abs/2310.07177 |
| **License** | Apache-2.0 (see original repo) |

## Files we modified from the OSD codebase

One file from the original OSD codebase was modified and moved to `gbv-research/algorithms/`
(Phase 2 migration — commit `16f0560`). The `OSD/` submodule itself is unmodified upstream.

| File | Change | Current location |
|---|---|---|
| `online_serve.py` | Added OOM-safe logit masking before softmax to reduce peak VRAM on 4 GB GPUs | `gbv-research/algorithms/online_serve.py` |

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
