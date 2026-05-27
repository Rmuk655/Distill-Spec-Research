# Attribution

The code in this directory is the **GBV (Generalised Block Verification)** codebase,
written by **Rahul Thomas** (Columbia University PhD student, research lead on this project).

This directory is an **exact copy** of the anonymous reference repo
([anonymous.4open.science/r/GBV-BED8](https://anonymous.4open.science/r/GBV-BED8), commit `b3dc0a7`).
No files have been modified. `ATTRIBUTION.md` is the only addition — it was not in the reference repo.

## Team context

| Person | Affiliation | Role |
|---|---|---|
| Rahul Thomas | Columbia University | Research Lead — owns GBV algorithms, EBE math, publication |
| Krishnan R | IIT Hyderabad | Research Engineer — owns implementation, experiments, pipeline |

GBV is **not** third-party or external borrowed code. It is original work by a
co-author on this project.

## How GBV is used in this pipeline

The production evaluation entry point is
`gbv-research/algorithms/distillspec_gbv/verifiers/runner.py`, which re-implements
GBV's algorithm with additional infrastructure (Windows compat, `load_in_4bit`,
`torch.compile`, corrected dtype kwarg for transformers ≥ 4.51).
`GBV/` is kept here as the canonical algorithm reference — read-only.

Do **not** edit any `.py` files in this directory. Infrastructure changes belong in
`distillspec_gbv/verifiers/runner.py`.
