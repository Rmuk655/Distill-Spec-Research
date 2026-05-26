# Attribution

The code in this directory is the **GBV (Generalised Block Verification)** codebase.

| Field | Value |
|---|---|
| **Source** | https://anonymous.4open.science/r/GBV-BED8/ |
| **Paper** | "Generalised Block Verification for Speculative Decoding" (anonymous submission) |
| **Note** | Anonymous submission — git submodule not possible until camera-ready release |

A verbatim copy of the original files (as downloaded from the anonymous link) is
preserved at `gbv-research/references/gbv-original/` for reference.

## Files modified for this research project

| File | Change |
|---|---|
| `main.py` | Added UTF-8 stdout/stderr reconfiguration for Windows; added `PYTORCH_CUDA_ALLOC_CONF` env-var to reduce CUDA allocator fragmentation on small GPUs |
| `util.py` | Updated `from_pretrained()` dtype kwarg for transformers ≥ 4.51 compatibility |

All other files are unmodified from the original anonymous submission.

## Evolved production version

A restructured, evolved version of this codebase lives at
`gbv-research/algorithms/distillspec_gbv/verifiers/`. That version is the
target for Phase 2 migration (the pipeline will switch to calling
`verifiers/runner.py` instead of `GBV/main.py`).
