# Modal Setup Guide — SpecDist on T4 (Phase-1 exploration)

> **⚠️ CREDIT WARNING (June 2026): Modal now offers only $1 free credit**, not the $30
> previously advertised. The $1 credit covers approximately **1–2 minutes of T4 time**.
> Modal is **not viable for full research runs**. Use it only for a single targeted
> smoke test or short ablation. For primary compute:
> - **Tier 2 exploration (T4)**: use Kaggle (free quota) or Lightning AI (free credits)
> - **A100 confirmation**: use **IITH cluster at ₹80/GPU-hr** (~$0.94/hr) — best value

[Modal](https://modal.com) gives every new account a small free compute credit.
This guide documents the SpecDist pipeline on an **on-demand T4** that you pay for
only while it runs (per-second billing).

It uses `deploy/modal_app.py`, a **pure launcher** that builds a GPU image, mounts a
persistent Volume, clones the repo, and invokes the **same CLI** the Kaggle bootstrap
uses:

```
python orchestration/experiment.py --config profiles/train_one_loss --storage_root /vol/specdist --yes
```

No core pipeline code is modified — `modal_app.py` lives entirely outside
`experiment.py` / `trainer.py` / `evaluate.py` / `runner.py` / `hw_scheduler.py`,
exactly like the Kaggle bootstrap.

---

## Cost — real credit situation (June 2026)

| GPU (Modal) | ~Price / hour | Effective free budget | Use it for |
|---|---|---|---|
| **T4 (16 GB)** | **≈ $0.59 / h** | **~1–2 min on $1 credit** | One smoke test only |
| A100 (40 GB) | ≈ $5.30 / h | **~10 min on $1 credit** | Not viable — use IITH |
| A100-80GB | ≈ $7+ / h | ~8 min | Not viable — use IITH |

Billing is **per second** and you are charged **only while the function runs** (image
build is free, idle time is free). **With $1 free credit, any non-trivial run requires
a payment method on file.**

**Better alternatives for actual research:**
- **Kaggle T4 x2**: free, ~30 GPU-hr/week (burns 2× quota, so ~15 effective hr)
- **Lightning AI**: free monthly credits, T4 available
- **IITH A100**: ₹80/GPU-hr (~$0.94/hr) — use for all A100 confirmation runs
- **GCP $300 trial**: T4 available after GPU quota request

> **Always run `--smoke` first.** Catches setup errors before you commit paid time.

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
Cheap on the T4 (the default GPU).

### b) Phase-1 tiered iteration (train + val_loss + light BE sanity)

Phase-1 runs are **tiered**: train + val_loss curves + a light BE sanity check (n=100,
K=3, matched verifier, GSM8K only). Heavy full-GSM8K eval is deferred to A100
confirmation. Use `profiles/train_one_loss` (flat losses) or `profiles/tree_variant_week`
(tree losses), which both set `experiment.light_eval: true`.

```bash
# Forward-KL flat baseline — tiered (train + val_loss + light BE, 1 verifier):
modal run deploy/modal_app.py --config profiles/train_one_loss --losses kl

# Tree-loss variant — tiered (train + light BE, tree-matched verifiers):
modal run deploy/modal_app.py --config profiles/tree_variant_week --losses kl_tree
```

`forward_kl` is accepted as an alias for `kl`. You can pass any comma-separated subset,
e.g. `--losses kl,rev_kl,jsd`. Valid names are the keys in `experiment.py`'s
`_LOSS_STEP_PREFIXES` (`kl, rev_kl, jsd, l1, ebe, ebe_single, kl_tree, …, gbv_tree, …`).

### c) Full A100 confirmation (IITH cluster — not Modal)

A100 confirmation runs use the IITH cluster, not Modal. See `docs/COMPUTE.md` for
IITH setup. The commands below are provided for reference only if you ever use
Modal A100.

```bash
modal run deploy/modal_app.py --config a100 --losses kl
```

`a100.yaml` references HF repo IDs (`Qwen/Qwen3-0.6B`, `Qwen/Qwen3-8B`), so `--config
a100` works directly on Modal. Runs the **full** verifier matrix and heavier eval (n=1319,
K∈{3,5}, multi-seed) — these are the paper numbers. Reserve for ≤2 confirmed candidates.

```bash
modal run deploy/modal_app.py   # all losses in a100.yaml (the full 8×8 matrix)
```

Long (hours) — budget against your credit and prefer running loss subsets across several
short invocations (each resumes via the Volume; see below).

### Other options

```bash
modal run deploy/modal_app.py --config a100 --git-ref main   # pin a branch/tag/commit (A100, IITH preferred)
modal run deploy/modal_app.py --losses gbv_tree              # GBV verifier-aligned loss
```

| Flag | Default | Meaning |
|---|---|---|
| `--config` | `profiles/train_one_loss` | YAML profile under `orchestration/configs/` |
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
GPU_TYPE = "T4"          # default — fits kaggle/profiles configs (4-bit NF4 teacher)
# GPU_TYPE = "A100"      # only if doing a one-off Modal A100 run; prefer IITH cluster
```

- `"T4"` (16 GB) is the default for Phase-1 exploration — requires the `kaggle` /
  `profiles/` configs with 4-bit NF4 teacher. **The 8B bf16 teacher does NOT fit a T4.**
- `"A100"` (40 GB) if doing a one-off Modal A100 run; A100 confirmation runs should
  normally go to the IITH cluster — see `docs/COMPUTE.md`.
- `"A100-80GB"` if you ever bump the teacher size or batch.

`TIMEOUT_SECONDS` (default 4 h) is right below it — T4 Phase-1 runs fit comfortably
in 4 hours.

---

## Notes / assumptions

- **No `modal.yaml` needed.** `profiles/train_one_loss.yaml` and `profiles/tree_variant_week.yaml`
  already reference HF repo ids (`Qwen/Qwen3-0.6B`, `Qwen/Qwen3-8B`), so
  `--config profiles/train_one_loss` works directly on Modal. Use `--config profiles/train_one_loss`
  (not `a100`) for T4 runs.
- **Deps** mirror `requirements.txt` + the extras `deploy_utils.install_deps` adds
  (`bitsandbytes>=0.46.1`, `accelerate`, `torchao>=0.16.0`); `torch` is installed
  separately. Pins match the repo exactly.
- **W&B** runs offline unless `WANDB_API_KEY` is in the secret.
- The repo is **cloned at runtime** (default branch `main`) so a run always uses the
  latest pushed code — push your changes before launching.
