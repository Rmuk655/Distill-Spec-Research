# specInfer (bundled copy)

This directory is the **canonical** `specInfer` package for the gbv-research pipeline.

`evaluate.py` prepends `gbv-research/algorithms/` to `sys.path` **before** the sibling
`OSD/distill/` fallback, so eval and alpha measurement always import from here — not from
the git submodule.

## Why not edit the OSD submodule?

The repo root has an `OSD/` git submodule (LiuXiaoxuanPKU/OSD). Editing files there:

- marks the parent repo as `m OSD` dirty even when eval uses the bundled copy
- requires a submodule commit **and** a parent pointer bump to ship changes
- is easy to desync from what actually runs on the A100 server

**Rule:** specInfer / alpha / BE fixes → edit **this folder** and commit in
`gbv-research/`. Leave `OSD/` at the pinned submodule commit unless you are
explicitly working on online OSD training (Phase 2c) inside that repo.

## Sync policy

| File | Bundled vs OSD submodule |
|------|--------------------------|
| `generator.py`, `logger.py`, `__init__.py` | kept in sync at import time |
| `common.py`, `proposer.py`, `verifier.py` | may diverge — bundled wins at runtime |

When pulling fixes from upstream OSD, copy into this directory manually and commit here.
Do not rely on submodule dirty state for pipeline behavior.
