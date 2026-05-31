# Modal Setup Guide — SpecDist on an on-demand A100

[Modal](https://modal.com) gives every account **$30 / month of free compute credit**.
This guide runs the full SpecDist pipeline on an **on-demand A100** that you pay for
only while it runs (per-second billing) — ideal for short, bursty research runs.

It uses `deploy/modal_app.py`, a **pure launcher** that builds a GPU image, mounts a
persistent Volume, clones the repo, and invokes the **same CLI** the Kaggle bootstrap
uses:

```
python orchestration/experiment.py --config a100 --storage_root /vol/specdist --yes
```

No core pipeline code is modified — `modal_app.py` lives entirely outside
`experiment.py` / `trainer.py` / `evaluate.py` / `runner.py` / `hw_scheduler.py`,
exactly like the Kaggle bootstrap.

---

## Cost — budget against your $30 credit

| GPU (Modal) | ~Price / hour | What it's for |
|---|---|---|
| **A100 (40 GB)** | **≈ $5.30 / h** | Full pipeline (8B teacher, bf16) — the default |
| A100-80GB | ≈ $7+ / h | Only if you hit VRAM limits |
| T4 (16 GB) | ≈ $0.59 / h | Cheap `--smoke` sanity checks |

Billing is **per second** and you are charged **only while the function runs** (image
build is free, idle time is free). A100 at $5.30/h means the **$30 credit ≈ 5.6 A100-hours**.

> **Always run `--smoke` first.** It costs a few minutes of A100 time and catches
> setup/config errors before you commit hours of credit to a full run.

---

## One-time setup

### 1. Install the Modal client and authenticate

```bash
pip install modal
modal setup          # opens a browser to link your Modal account
```

### 2. Create the secret

The app reads **one** Modal secret named `specdist-secrets`. Every key inside is
**optional** — include only the tokens you have. Using a single secret means a
missing optional token never fails the run with "Secret not found"; you just omit
that key.

```bash
modal secret create specdist-secrets \
    HF_TOKEN=hf_xxxxxxxxxxxxxxxxx \
    WANDB_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx \
    GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxx
```

| Key | Where to get it | Required? |
|---|---|---|
| `HF_TOKEN` | https://huggingface.co/settings/tokens | **Optional** — Qwen3-0.6B/8B are public |
| `WANDB_API_KEY` | https://wandb.ai/authorize | **Optional** — omit → `WANDB_MODE=offline` |
| `GITHUB_TOKEN` | GitHub → Settings → Developer settings → PAT (`repo` scope) | Only if the repo is **private** |

Minimal version (public repo, no W&B, public models) — create an empty-ish secret so
the name exists:

```bash
modal secret create specdist-secrets HF_TOKEN=hf_xxxxxxxx
```

> The Volume `specdist-vol` is created automatically on first run
> (`create_if_missing=True`) — nothing to set up manually.

---

## Running

> Run all commands from inside `gbv-research/` (so `deploy/modal_app.py` resolves).

### a) Smoke test (do this first)

```bash
modal run deploy/modal_app.py --smoke
```

Quick crash-check: a few prompts, tiny token budget, exercises every loss + verifier.
Cheap on the A100; if you want it even cheaper, set `GPU_TYPE = "T4"` at the top of
`modal_app.py` for smoke runs.

### b) Forward-KL flat baseline first (research iteration order #1)

The researcher's priority is **(1) flat baselines like forward KL → (2) tree-losses →
(3) GBV verifier → (4) online variants last**. The flat forward-KL loss is named **`kl`**
in this repo (`kl_tree` is its tree variant). Run just that baseline:

```bash
modal run deploy/modal_app.py --losses kl
```

You can pass any comma-separated subset, e.g. `--losses kl,rev_kl,jsd`. Valid names are
the keys in `experiment.py`'s `_LOSS_STEP_PREFIXES` (`kl, rev_kl, jsd, l1, ebe,
ebe_single, kl_tree, …, gbv_tree, …`).

### c) Full A100 pipeline

```bash
modal run deploy/modal_app.py
```

Runs **all** losses defined in `a100.yaml` (the full 8×8 loss-verifier matrix). This is
long (hours) — budget against your credit and prefer running loss subsets across several
short invocations (each resumes via the Volume; see below).

### Other options

```bash
modal run deploy/modal_app.py --config a100 --git-ref main   # pin a branch/tag/commit
modal run deploy/modal_app.py --losses gbv_tree              # GBV verifier-aligned loss
```

| Flag | Default | Meaning |
|---|---|---|
| `--config` | `a100` | YAML profile under `orchestration/configs/` |
| `--smoke` | off | Quick crash-check mode |
| `--losses` | (all) | Comma-separated subset of losses |
| `--git-ref` | `main` | Branch / tag / commit to clone |

---

## Persistence & resume (the Volume)

A Modal **Volume** named `specdist-vol` is mounted at `/vol`, and everything persistent
lives under `/vol/specdist`:

```
/vol/specdist/
  results.db            # SPECDIST_DB_PATH
  checkpoints/          # LoRA adapters / merged models
  logs/                 # SPECDIST_LOGS_ROOT (pipeline_output.log, step errors)
  hf_cache/             # HF_HOME — Qwen3 weights cached here, persists across runs
  pipeline_state_*.json # state machine — drives resume
```

The launcher sets `--storage_root /vol/specdist`, which makes `experiment.py` write
`SPECDIST_STORAGE_ROOT` / `SPECDIST_DB_PATH` / `SPECDIST_LOGS_ROOT` there, and it points
`HF_HOME` into the Volume so model weights download **once**.

**Resume is automatic.** Each container is fresh, but the Volume is not: re-running the
**same command** picks up where the last run stopped — the pipeline's state machine skips
already-completed train/eval steps, and the HF cache means Qwen3-8B is **not** re-downloaded.
The Volume is committed at the end of every run (and the function always returns its exit
code, so a crash still leaves committed checkpoints to resume from).

> First full run downloads Qwen3-8B (~16 GB) into `hf_cache/` — a few minutes, once.
> Subsequent runs skip it.

---

## Switching GPU type

Edit the constant near the top of `deploy/modal_app.py`:

```python
GPU_TYPE = "A100"        # -> "A100-80GB" | "L4" | "T4" | "H100" | ...
```

- `"A100"` = 40 GB (default, fits Qwen3-8B in bf16 with headroom).
- `"A100-80GB"` if you ever bump the teacher size or batch.
- `"T4"` (16 GB) is fine for `--smoke` to save credit, but the 8B bf16 teacher will **not**
  fit a T4 for a real run — use the `kaggle` config (4-bit NF4) if you must use a T4.

`TIMEOUT_SECONDS` (default 8 h) is right below it if a long full run needs more headroom.

---

## Notes / assumptions

- **No `modal.yaml` needed.** `a100.yaml` already references HF repo ids
  (`Qwen/Qwen3-0.6B`, `Qwen/Qwen3-8B`), so `--config a100` works directly on Modal.
- **Deps** mirror `requirements.txt` + the extras `deploy_utils.install_deps` adds
  (`bitsandbytes>=0.46.1`, `accelerate`, `torchao>=0.16.0`); `torch` is installed
  separately. Pins match the repo exactly.
- **W&B** runs offline unless `WANDB_API_KEY` is in the secret.
- The repo is **cloned at runtime** (default branch `main`) so a run always uses the
  latest pushed code — push your changes before launching.
