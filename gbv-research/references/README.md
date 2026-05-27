# References — Original Borrowed Code

This directory contains the **original, unmodified** versions of the external
codebases this project builds on.  They are here for provenance tracking and
diff comparison — **do not import from here in production code**.

| Directory | Source | Purpose |
|---|---|---|
| `adaspec/` | AdaSpec reference implementation | Adaptive speculative decoding baseline for comparison |

The original OSD codebase (LiuXiaoxuanPKU/OSD) lives at the repo root as `OSD/`
(a git submodule / reference copy). The original GBV codebase lives at the repo
root as `GBV/` (reference copy from anonymous.4open.science/r/GBV-BED8/).

## What we changed vs. the originals

### vs. OSD (upstream: LiuXiaoxuanPKU/OSD)
Our research code does **not** use OSD's training classes. We wrote a new
training loop in `algorithms/train_qwen3.py` with the following differences:
- Plain PyTorch loop instead of `transformers.Trainer` subclass
- LoRA adapters (PEFT) instead of full fine-tuning
- Teacher-sample-only mode (no wrong_token_ids masking)
- All 5 offline losses + 2 online losses implemented
- Crash-safe checkpoint/resume, W&B logging, health checks
- Fixed `torch_dtype=` → `dtype=` for transformers ≥ 4.51

We do borrow `OSD/distill/specInfer/generator.py` for the alpha-eval Generator
(evaluate.py imports it via sys.path). This is a legitimate borrow — the
Generator class is unchanged.

### vs. GBV (upstream: anonymous.4open.science/r/GBV-BED8/)
Our evolved version in `algorithms/distillspec_gbv/verifiers/` makes these changes:
- Reorganised flat 6-file structure into a proper Python package
- Renamed files: `main.py` → `runner.py`, `node.py` → `tree.py`, etc.
- Updated imports to use relative package imports (no sys.path hacks)
- Added `__init__.py` with algorithm documentation
- No algorithmic changes — all verification logic is preserved exactly

After Phase 3 eval confirms `algorithms/distillspec_gbv/verifiers/runner.py`
produces identical results to `GBV/main.py`, `GBV/` will be reverted to the
exact reference repo (see `GBV/ATTRIBUTION.md` for the revert plan).

## Attribution

When publishing results from this project, cite:
- OSD: Liu et al., "Online Speculative Decoding," arXiv:2310.07177, 2023
  (https://github.com/LiuXiaoxuanPKU/OSD)
- GBV: Thomas et al., "Generalized Block Verification for Speculative Decoding,"
  arXiv:2602.16994v1, 2026.
- AdaSpec: [citation to be filled in]
