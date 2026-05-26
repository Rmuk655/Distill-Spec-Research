# Attribution

The code in this directory is the **GBV (Generalised Block Verification)** codebase,
written by **Rahul Thomas** (Columbia University PhD student, research lead on this project).

This directory is kept separate in the monorepo for organisational clarity — Rahul
maintains GBV as its own repository, and the copy here is the authoritative version
used as the verification backend during the current pipeline phase.

## Team context

| Person | Affiliation | Role |
|---|---|---|
| Rahul Thomas | Columbia University | Research Lead — owns GBV algorithms, EBE math, publication |
| Krishnan R | IIT Hyderabad | Research Engineer — owns implementation, experiments, pipeline |

GBV is **not** third-party or external borrowed code. It is original work by a
co-author on this project.

## Files modified from original version

| File | Change |
|---|---|
| `main.py` | Added UTF-8 stdout/stderr reconfiguration for Windows; added `PYTORCH_CUDA_ALLOC_CONF` env-var to reduce CUDA allocator fragmentation on small GPUs |
| `util.py` | Updated `from_pretrained()` dtype kwarg for transformers ≥ 4.51 compatibility |
| `verifier.py` | Expanded module docstring with research findings (Thomas et al., 2026 empirical ordering; traversal vs OT-based analysis); no logic changes |

`node.py` is unmodified.

## Evolved production version

A restructured, production-ready version of this codebase lives at
`gbv-research/algorithms/distillspec_gbv/verifiers/`. That version adds:
- Package structure (`__init__.py` with clean exports)
- Corrected import paths
- Integration with the `gbv-research` training pipeline

The Phase 2 migration target is for `orchestration/run_all.py` to call
`verifiers/runner.py` directly instead of shelling out to `GBV/main.py`.
