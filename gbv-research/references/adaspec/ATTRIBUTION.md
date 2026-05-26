# Attribution

The code in this directory is the **AdaSPEC** codebase.

| Field | Value |
|---|---|
| **Source** | https://github.com/yuezhouhu/adaspec |
| **Paper** | "AdaSPEC: Adaptive Speculative Decoding for Efficient LLM Serving", Hu et al., 2024 — https://arxiv.org/abs/2405.13357 |
| **License** | See LICENSE file in this directory |

## Role in this project

Reference codebase only — **never imported or executed**. Kept for:
- Ablation comparison: AdaSPEC uses a learned acceptance threshold; our EBE loss
  directly optimises the acceptance product without a threshold hyper-parameter.
- Implementation reference: draft-target probability ratio computation.

No files in this directory have been modified from the original repository.
