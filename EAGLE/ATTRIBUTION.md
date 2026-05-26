# Attribution

The code in this directory is the **EAGLE** codebase (EAGLE / EAGLE-2 / EAGLE-3).

| Field | Value |
|---|---|
| **Source** | https://github.com/SafeAILab/EAGLE |
| **Papers** | EAGLE: Li et al., 2024 — https://arxiv.org/abs/2401.15077 |
|  | EAGLE-2: Li et al., 2024 — https://arxiv.org/abs/2406.16858 |
|  | EAGLE-3: Li et al., 2025 — https://arxiv.org/abs/2503.01840 |
| **License** | Apache-2.0 (see LICENSE in this directory) |

## Role in this project

Reference codebase — **not imported or executed** by our pipeline. Kept for:

- Architecture reference: EAGLE's draft head design (1-layer FC over hidden states)
  informed our `algorithms/eagle_bench.py` reimplementation
- Comparison baseline: EAGLE-3 is our primary comparison target (see PROJECT_CONTEXT.md
  "Minimum bar: beat EAGLE-3 throughput in ≥1 setting")
- Training recipe reference: `EAGLE/eagle/traineagle3/` for understanding their
  hidden-state collection + head training approach

Our `gbv-research/algorithms/eagle_bench.py` is a **self-contained reimplementation**
for Qwen3 models — it does not import from this directory.

## No modifications

No files in this directory have been modified from the upstream repository.
