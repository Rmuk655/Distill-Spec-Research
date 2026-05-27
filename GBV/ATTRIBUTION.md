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

## Production version (Phase 2 — migration complete)

`orchestration/run_all.py` now calls
`gbv-research/algorithms/distillspec_gbv/verifiers/runner.py` as the primary
evaluation entry point. `GBV/main.py` is retained as a fallback only until
Phase 3 evals confirm runner.py is correct (expected: next pipeline run).

After Phase 3 confirmation, this directory will be restored to an **exact copy**
of the anonymous reference repo (anonymous.4open.science/r/GBV-BED8):
- Revert `main.py`, `util.py`, `verifier.py` to reference versions
- Remove the `GBV_DIR` fallback from `run_all.py`
- `ATTRIBUTION.md` and `data/` stay (they were never in the reference)

The files in `distillspec_gbv/verifiers/` incorporate all infrastructure additions
(Windows compat, load_in_4bit, torch.compile, corrected dtype kwarg) that were
temporarily patched into the files here. Those patches are no longer needed in
GBV/ once the production verifier is runner.py.
