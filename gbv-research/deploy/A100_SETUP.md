# A100 setup — IIT Hyderabad GPU server (Pluto)

**Hardware:** NVIDIA A100 (40 GB typical)  
**Model pair:** Qwen3-0.6B draft → Qwen3-8B teacher (BF16, no 4-bit)  
**Config:** `--config a100_qwen`

Flat `ebe` / `ebe_single` are excluded by default (off-policy on teacher rollouts). Prefer verifier-aligned tree losses: `naive_tree`, `bv_tree`, `gbv_tree`, `traversal_tree`, `specinfer_tree`.

---

## Storage layout

Two tiers — keep them separate:

| What | Where | Why |
|------|-------|-----|
| git repo, venv, HF model weights | `/home/colligo/ram/` | Large local quota; reproducible from scratch |
| W&B local files, pipeline logs | `/home/colligo/ram/specdist/` | Ephemeral — not worth Sensei quota |
| **Checkpoints, results.db, merged models** | **`/sensei-fs/users/rkrishna/specdist/`** | Irreplaceable mid-run — must survive session timeouts |

Rule of thumb: **if you can re-create it in < 30 min, it goes in `/home/colligo/ram`**.

---

## Sensei FS quota

If you hit "Disk quota exceeded" on Sensei, check actual usage:

```bash
# Lustre filesystems (most likely on Pluto):
lfs quota -u rkrishna /sensei-fs

# Generic NFS quota:
quota -s

# Fallback — byte count in your directory (may undercount if quota tracks inodes):
du -sh /sensei-fs/users/rkrishna/
```

> **Note:** `du` showing only ~14 MB while quota is exceeded means the quota is
> tracked at the filesystem level (inode count or project quota), not raw bytes.
> `lfs quota` is the authoritative command on Lustre. Ask the Sensei platform team
> for the exact quota command if none of the above work.

---

## One-time setup

```bash
# 1. Secrets (replace with real values)
export WANDB_API_KEY="your-key-from-wandb.ai/authorize"
export GITHUB_TOKEN="ghp_..."   # only if repo is private

# 2. Bootstrap clone (only if repo not yet on local storage)
git clone https://github.com/Rmuk655/Distill-Spec-Research.git \
    /home/colligo/ram/Distill-Spec-Research

# 3. Run setup — clones repo to /home/colligo/ram, puts checkpoints on Sensei FS
bash /home/colligo/ram/Distill-Spec-Research/gbv-research/deploy/aip_gpu_setup.sh a100_qwen
```

`aip_gpu_setup.sh` writes `~/.specdist_env` with all paths. Source it in every new SSH session:

```bash
source ~/.specdist_env
cd "$GBV_DIR"
```

### After pulling updates from GitHub

```bash
git -C /home/colligo/ram/Distill-Spec-Research pull

# Re-run setup (reuses existing venv, refreshes deps + ~/.specdist_env)
bash /home/colligo/ram/Distill-Spec-Research/gbv-research/deploy/aip_gpu_setup.sh a100_qwen

source ~/.specdist_env
cd "$GBV_DIR"
```

---

## Smoke test (run first)

```bash
source ~/.specdist_env && cd "$GBV_DIR"
python deploy/aip_run.py --config a100_qwen
```

Smoke runs a short pipeline check (~10–20 min). If it passes, run full training.

---

## Training (one loss at a time)

`--losses` alone runs **baseline eval** first, then train/merge/eval for that loss. If a checkpoint already exists, training is skipped.

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
python deploy/aip_run.py --config a100_qwen --losses traversal_tree --train_only --no_smoke
```

Check what the pipeline thinks is done:

```bash
python orchestration/experiment.py --config a100_qwen --status
```

Step IDs for `traversal_tree`:

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

The BV block-acceptance integral amplifies gradients compared with `kl_tree` or flat KL. YAML still uses `lr: 3e-5` for the full sweep; override on the CLI for these two:

```bash
python orchestration/experiment.py --config a100_qwen \
  --losses bv_tree --lr 1e-5 --train_only --yes
```

Same pattern for `gbv_tree`. `deploy/aip_run.py` does not pass `--lr` through — call `experiment.py` directly as above.

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

Training resumes from `ckpt_latest` + `training_state.json` on Sensei FS. W&B run also resumes (same dashboard URL) because `wandb_run.json` is saved alongside the checkpoint.

---

## Kill and restart (same loss, keep progress)

```bash
kill -TERM <pid>
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

Eval runs automatically after train+merge. To re-eval only:

```bash
python deploy/aip_run.py --config a100_qwen --resume
```

Default modes (in `bases/a100.yaml`): `alpha`, `naive`, `nss`, `specinfer`, `spectr`, `khisti`, `bv`, `gbv`, `traversal`.

Eval uses up to **3 GPUs in parallel** by default (shared machine — polite limit). Override:

```bash
SPECDIST_MAX_EVAL_GPUS=1 python deploy/aip_run.py ...   # force serial
SPECDIST_MAX_EVAL_GPUS=2 python deploy/aip_run.py ...   # lighter footprint
```

---

## Where logs live

| What | Path |
|------|------|
| Pipeline stdout | `/sensei-fs/users/rkrishna/specdist/logs/a100_qwen-q0.6b-q8b/pipeline_output.log` |
| Per-step errors | `/sensei-fs/users/rkrishna/specdist/logs/a100_qwen-q0.6b-q8b/step_<id>_error.log` |
| W&B local run files | `/home/colligo/ram/specdist/wandb/wandb/run-<date>-<id>/logs/` |
| W&B dashboard | URL printed as `[wandb] https://wandb.ai/...` in pipeline log |

```bash
# Tail pipeline
tail -f /sensei-fs/users/rkrishna/specdist/logs/a100_qwen-q0.6b-q8b/pipeline_output.log

# Tail W&B local log (replace run id)
tail -f /home/colligo/ram/specdist/wandb/wandb/run-<id>/logs/debug.log
```

---

## Git pull: fix "URL rejected: Bad hostname"

```bash
cd /home/colligo/ram/Distill-Spec-Research

# Remove embedded credentials — use plain HTTPS URL
git remote set-url origin https://github.com/Rmuk655/Distill-Spec-Research.git

# Public repo: pull without token in URL
git pull
```

Re-run setup after fixing the remote:

```bash
bash /home/colligo/ram/Distill-Spec-Research/gbv-research/deploy/aip_gpu_setup.sh a100_qwen
```

---

## Related docs

| Doc | Use when |
|-----|----------|
| [deploy/README.md](README.md) | All providers (Colab, Kaggle, Modal, …) |
| [deploy/ats/README.md](ats/README.md) | **CPU-only** server (no GPU) — GPT-2 pair, parallel training |
