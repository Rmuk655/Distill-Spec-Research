# A100 setup — IIT Hyderabad GPU server

**Hardware:** NVIDIA A100 (40 GB typical)  
**Model pair:** Qwen3-0.6B draft → Qwen3-8B teacher (BF16, no 4-bit)  
**Cost:** about **₹80/hour** (IIT Hyderabad allocation)  
**Config:** `--config a100` or `--config a100_qwen` (equivalent)

Flat `ebe` / `ebe_single` are excluded by default in `orchestration/configs/bases/a100.yaml` (off-policy on teacher rollouts). Prefer verifier-aligned tree losses: `naive_tree`, `bv_tree`, `gbv_tree`, `traversal_tree`, `specinfer_tree`.

---

## One-time setup

```bash
# 1. Secrets (replace with real values)
export WANDB_API_KEY="your-key-from-wandb.ai/authorize"
export GITHUB_TOKEN="ghp_..."          # only if repo is private
export STORAGE_ROOT=/sensei-fs/users/rkrishna/specdist     # checkpoints, logs, results.db persist here

# 2. Clone repo
cd /sensei-fs/users/rkrishna
git clone https://github.com/Rmuk655/Distill-Spec-Research.git

# 3. Install deps, auth, GPU check (creates $STORAGE_ROOT/venv if missing)
bash /sensei-fs/users/rkrishna/Distill-Spec-Research/gbv-research/deploy/aip_gpu_setup.sh a100_qwen
```

### After pulling doc/code fixes from GitHub

```bash
git -C /sensei-fs/users/rkrishna/Distill-Spec-Research pull

# Re-run setup (reuses existing venv at /sensei-fs/users/rkrishna/specdist/venv, refreshes deps + ~/.specdist_env)
bash /sensei-fs/users/rkrishna/Distill-Spec-Research/gbv-research/deploy/aip_gpu_setup.sh a100_qwen

source ~/.specdist_env
cd "$GBV_DIR"
```

`aip_gpu_setup.sh` writes `~/.specdist_env` with `STORAGE_ROOT` and `GBV_DIR`. Source it in every new SSH session:

```bash
source ~/.specdist_env
cd "$GBV_DIR"
```

---

## Smoke test (run first)

```bash
cd /sensei-fs/users/rkrishna/Distill-Spec-Research/gbv-research
python deploy/aip_run.py --config a100_qwen
```

Smoke runs a short pipeline check (~10–20 min). If it passes, run full training.

---

## Training (one loss at a time)

`--losses` alone still runs **baseline eval** first (`eval_baseline_gsm8k`), then train/merge/eval for that loss. If a checkpoint already exists, **training is skipped** and you only see eval.

**Train only (no baseline, no eval):**

```bash
python deploy/aip_run.py --config a100_qwen --losses traversal_tree --train_only --no_smoke
```

**Train + merge + eval for one loss:**

```bash
python deploy/aip_run.py --config a100_qwen --losses traversal_tree --no_smoke
```

**Start at the training step (skip baseline):**

```bash
python deploy/aip_run.py --config a100_qwen --losses traversal_tree --from_step train_trav_tree_gsm8k --no_smoke
```

**Force re-train** (delete checkpoint + reset state first):

```bash
rm -rf /sensei-fs/users/rkrishna/specdist/checkpoints/trav_tree-gsm8k-q0.6b-q8b
rm -rf /sensei-fs/users/rkrishna/specdist/checkpoints/trav_tree-gsm8k-q0.6b-q8b_merged
# Reset train/merge/eval steps in pipeline state (or use deploy/rerun_loss.sh)
python deploy/aip_run.py --config a100_qwen --losses traversal_tree --train_only --no_smoke
```

Check what the pipeline thinks is done:

```bash
python orchestration/experiment.py --config a100_qwen --status --storage_root /sensei-fs/users/rkrishna/specdist
```

Step IDs for `traversal_tree` (not `traversal`):

| Step | ID |
|------|-----|
| Train | `train_trav_tree_gsm8k` |
| Merge | `merge_trav_tree_gsm8k` |
| Eval | `eval_trav_tree_gsm8k` |

Checkpoint dir: `/sensei-fs/users/rkrishna/specdist/checkpoints/trav_tree-gsm8k-q0.6b-q8b`

Other losses:

```bash
python deploy/aip_run.py --config a100_qwen --losses kl --train_only --no_smoke
python deploy/aip_run.py --config a100_qwen --losses naive_tree --train_only --no_smoke
python deploy/aip_run.py --config a100_qwen --losses traversal_tree --train_only --no_smoke
```

### `bv_tree` / `gbv_tree` — use a lower learning rate

The BV block-acceptance integral amplifies gradients compared with `kl_tree` or flat KL (see `algorithms/distillspec_gbv/losses/tree_losses.py`). YAML still uses `lr: 3e-5` for the full sweep; for these two losses override on the CLI:

```bash
python orchestration/experiment.py --config a100_qwen \
  --losses bv_tree --lr 1e-5 --train_only --yes --storage_root /sensei-fs/users/rkrishna/specdist
```

Same pattern for `gbv_tree`. `deploy/aip_run.py` does not pass `--lr` through — call `experiment.py` as above (or add `--lr` to your own wrapper).

**Timing (approx.):** ~15–20 min per loss for 2000 steps on A100; full 17-loss sweep is multi-hour — run one loss per session if needed.

**Monitor:**

```bash
tail -f /sensei-fs/users/rkrishna/specdist/logs/a100_qwen-q0.6b-q8b/pipeline_output.log
python orchestration/experiment.py --config a100_qwen --status
```

---

## Resume after disconnect or kill

Do **not** use `clean_restart` if you want to keep checkpoints.

```bash
source ~/.specdist_env
cd "$GBV_DIR"

# Stop old job (if still running)
pkill -TERM -f "experiment.py --config a100_qwen"
sleep 5
pkill -9 -f "experiment.py --config a100_qwen"   # only if still alive

python deploy/aip_run.py --config a100_qwen --resume
```

Training resumes from `ckpt_latest` + `training_state.json` inside each loss checkpoint dir. Pipeline skips completed steps and retries failed ones.

---

## Kill and restart (same loss, keep progress)

```bash
kill -TERM <pid>    # from top / ps — usually the python trainer or experiment.py parent
python deploy/aip_run.py --config a100_qwen --resume
```

---

## Full clean restart (wipes checkpoints)

Only when you intentionally want step 0:

```bash
python orchestration/clean_restart.py --config a100_qwen
```

---

## Evaluation

Eval runs automatically after train+merge in the pipeline. To re-eval only:

```bash
python deploy/aip_run.py --config a100_qwen --resume
# with experiment.eval_only in YAML, or use orchestration/evaluate.py directly
```

Default A100 eval modes (in `bases/a100.yaml`): `alpha`, `naive`, `nss`, `specinfer`, `spectr`, `khisti`, `bv`, `gbv`, `traversal`.

---

## Where logs live

| What | Path (typical on Pluto) |
|------|-------------------------|
| Pipeline stdout | `/sensei-fs/users/rkrishna/specdist/logs/a100_qwen-q0.6b-q8b/pipeline_output.log` |
| Per-step errors | `/sensei-fs/users/rkrishna/specdist/logs/a100_qwen-q0.6b-q8b/step_<id>_error.log` |
| Training (trainer) | `/sensei-fs/users/rkrishna/specdist/logs/<loss>-gsm8k-q0.6b-q8b/train.log` (if experiment routes there) |
| W&B local run files | `/sensei-fs/users/rkrishna/Distill-Spec-Research/gbv-research/db/wandb/wandb/run-<date>-<id>/logs/` |
| W&B dashboard | URL printed as `[wandb] https://wandb.ai/...` in pipeline or train log |

W&B stores under the **repo** (`gbv-research/db/wandb/`), not under `STORAGE_ROOT`, unless you set `WANDB_DIR` yourself. Checkpoints and `results.db` use `STORAGE_ROOT` (`/sensei-fs/users/rkrishna/specdist`).

```bash
# Tail pipeline
tail -f /sensei-fs/users/rkrishna/specdist/logs/a100_qwen-q0.6b-q8b/pipeline_output.log

# Tail one W&B run (replace run id)
tail -f /sensei-fs/users/rkrishna/Distill-Spec-Research/gbv-research/db/wandb/wandb/run-20260604_062658-1zvnc3xf/logs/debug.log
```

---

## Git pull: fix "URL rejected: Bad hostname"

This happens when `origin` was set twice with `GITHUB_TOKEN` embedded:

`https://TOKEN@TOKEN@github.com/...`

**Fix once (from repo root):**

```bash
cd /sensei-fs/users/rkrishna/Distill-Spec-Research

# Remove embedded credentials — use plain HTTPS URL
git remote set-url origin https://github.com/Rmuk655/Distill-Spec-Research.git

# Public repo: pull without token in URL
git pull

# Private repo: use env var, do NOT paste token into the URL manually
export GITHUB_TOKEN="ghp_..."   # GitHub PAT with repo scope
git pull
```

Re-run setup only after fixing the remote (setup script now strips old credentials before re-injecting):

```bash
bash /sensei-fs/users/rkrishna/Distill-Spec-Research/gbv-research/deploy/aip_gpu_setup.sh a100_qwen
```

**Security:** If a token appeared in a terminal log or chat, revoke it at GitHub → Settings → Developer settings → Personal access tokens and create a new one.

---

## Related docs

| Doc | Use when |
|-----|----------|
| [deploy/README.md](README.md) | All providers (Colab, Kaggle, Modal, …) |
| [deploy/ats/README.md](ats/README.md) | **CPU-only** server (no GPU) — GPT-2 pair, parallel training |
