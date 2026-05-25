# References — Original Borrowed Code

This directory contains the **original, unmodified** versions of the external
codebases this project builds on.  They are here for provenance tracking and
diff comparison — **do not import from here in production code**.

| Directory | Source | Purpose |
|---|---|---|
| `osd-original/` | [Distillation Spec (OSD)](https://github.com/example/osd) — borrowed from another researcher | Original knowledge-distillation training framework (HuggingFace Trainer-based) |
| `gbv-original/` | Thomas et al. arXiv:2602.16994v1 (2026) | Original GBV verification algorithms (6 files) |
| `adaspec/` | AdaSpec reference implementation | Adaptive speculative decoding baseline for comparison |

## What we changed vs. the originals

### vs. OSD (osd-original/)
- Renamed `train.py` → `capsules/distillation/trainer.py` (model-agnostic)
- Extracted loss functions into `capsules/distillation/losses/` (one file per loss)
- Added `ModelFamily` abstraction for Qwen/Gemma/etc. differences
- Fixed temperature-recovery bug (`/ temp` → `* temp` in output_scores handling)
- Fixed reverse-KL NaN on Qwen3 forbidden tokens (clamp -inf → -100)
- Replaced HuggingFace `Trainer` subclass with direct training loop (more control)
- Added crash-safe checkpoint/resume, W&B logging, health checks

### vs. GBV (gbv-original/)
- Reorganised flat 6-file structure into `capsules/verification/` capsule
- Renamed files: `main.py` → `runner.py`, `node.py` → `tree.py`, etc.
- Updated imports to use relative package imports (no sys.path hacks)
- Added `__init__.py` with algorithm documentation
- No algorithmic changes — all verification logic is preserved exactly

## Attribution

When publishing results from this project, cite:
- OSD: [original paper / repo to be filled in]
- GBV: Thomas et al., "Generalized Block Verification for Speculative Decoding,"
  arXiv:2602.16994v1, 2026.
- AdaSpec: [citation to be filled in]
