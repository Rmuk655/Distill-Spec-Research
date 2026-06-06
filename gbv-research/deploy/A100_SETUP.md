# A100 setup — IIT Hyderabad GPU server (Pluto)

**Hardware:** NVIDIA A100 (40 GB typical)  
**Model pair:** Qwen3-0.6B draft → Qwen3-8B teacher (BF16, no 4-bit)  
**Config:** `--config a100_qwen`

Flat `ebe` / `ebe_single` are excluded by default (off-policy on teacher rollouts). Prefer verifier-aligned tree losses: `naive_tree`, `bv_tree`, `gbv_tree`, `traversal_tree`, `specinfer_tree`.

---

## Storage layout (Pluto — default)

On Pluto, checkpoints, `results.db`, and pipeline logs use **local RAM disk** under the repo (Sensei FS is ignored unless `SPECDIST_USE_SENSEI=1` — quota is tight and SQLite on Lustre is unreliable).


| What                                             | Path                                       | Why                                   |
| ------------------------------------------------ | ------------------------------------------ | ------------------------------------- |
| git repo, venv                                   | `/home/colligo/ram/Distill-Spec-Research/` | Reproducible from GitHub              |
| HF model cache                                   | `/home/colligo/ram/specdist/hf_cache`      | Set `HF_HOME` here; large downloads   |
| W&B local files                                  | `/home/colligo/ram/specdist/wandb`         | Set `WANDB_DIR` here                  |
| **Checkpoints, merged models, results.db, logs** | `**.../gbv-research/db/`**                 | Single tree; survives normal sessions |


Concrete paths (after `source ~/.specdist_env`):


| Artifact      | Path                                                       |
| ------------- | ---------------------------------------------------------- |
| Checkpoints   | `$GBV_DIR/db/checkpoints/`                                 |
| Merged models | `$GBV_DIR/db/checkpoints/<loss>-gsm8k-q0.6b-q8b_merged/`   |
| Results DB    | `$GBV_DIR/db/results.db`                                   |
| Pipeline log  | `$GBV_DIR/db/logs/a100_qwen-q0.6b-q8b/pipeline_output.log` |
| BE progress   | `$GBV_DIR/db/logs/.../be_progress_*.log`                   |


Rule of thumb: **if you can re-create it in < 30 min, it goes in `/home/colligo/ram/specdist` or HF cache; irreplaceable training artifacts go in `gbv-research/db/`.**

### Sensei FS (optional — not recommended on Pluto)

Only if you explicitly set `SPECDIST_USE_SENSEI=1`. Check quota with `lfs quota -u rkrishna /sensei-fs`.

---

## One-time setup

```bash
# 1. Secrets (replace with real values)
export WANDB_API_KEY="your-key-from-wandb.ai/authorize"
export GITHUB_TOKEN="ghp_..."   # only if repo is private

# 2. Bootstrap clone (only if repo not yet on local storage)
git clone https://github.com/Rmuk655/Distill-Spec-Research.git \
    /home/colligo/ram/Distill-Spec-Research

# 3. Run setup — venv + ~/.specdist_env with paths
bash /home/colligo/ram/Distill-Spec-Research/gbv-research/deploy/aip_gpu_setup.sh a100_qwen
```

`aip_gpu_setup.sh` writes `~/.specdist_env`. Source it in every new SSH session:

```bash
source ~/.specdist_env
cd "$GBV_DIR"
export HF_HOME=/home/colligo/ram/specdist/hf_cache
export WANDB_DIR=/home/colligo/ram/specdist/wandb
```

**Eval datasets** (gsm8k_eval, humaneval, math500, …) are fetched automatically by
`aip_gpu_setup.sh`. **MATH-500** is re-fetched with `--force` when the on-disk JSONL
has no `"answer"` field (older prompts-only files cannot be scored). Manual one-liner
after a git pull that adds MATH-500 task_score:

```bash
cd "$GBV_DIR"
python core/datasets/downloader.py --datasets math500 --n 100 --force
```

### After pulling updates from GitHub

```bash
git -C /home/colligo/ram/Distill-Spec-Research pull
bash /home/colligo/ram/Distill-Spec-Research/gbv-research/deploy/aip_gpu_setup.sh a100_qwen
source ~/.specdist_env && cd "$GBV_DIR"
```

---

## Smoke test (run first)

```bash
source ~/.specdist_env && cd "$GBV_DIR"
python deploy/aip_run.py --config a100_qwen
```

Smoke runs a short pipeline check (~10–20 min). If it passes, run full training.

---

## Research workflow (one loss → variations → next loss)

This is the default workflow for research / ML engineering on this project:

1. **Pick one loss** (e.g. `rev_kl`, `gbv_tree`).
2. **Train + merge once** for that loss (checkpoint is reused across eval variations).
3. **Run eval one or more times** with different `**--experiment_tag`** values — each tag is one experimental condition (verifier set, `L`, `n`, code version, LR ablation after re-train, etc.).
4. **Compare** within the loss (filter by `experiment_tag`) or across losses (filter by `draft_label` / `loss_name`).
5. **Move to the next loss** and repeat.

`experiment_tag` is stored on **every eval cell** in `results.db` (column `experiment_tag`, indexed) and in **W&B** (run `config.experiment_tag` + tag). Use it as the primary handle for “which variation was this?”

### Naming convention (recommended)

Use predictable tags so filtering is easy:

```text
{loss}_{what_changed}

Examples:
  rev_kl_baseline          # default YAML eval (9 verifiers, L=8, n=100)
  rev_kl_L5                # evaluation.L=5 in YAML, then re-eval
  rev_kl_verifiers_bv_gbv  # evaluation.modes: [bv, gbv] only
  rev_kl_code_ef4a865      # after a git pull / bugfix, --force re-eval
  gbv_tree_lr1e5           # trained with lr override
```

YAML default tag (`logging.experiment_tag` in `a100_qwen.yaml`) applies when you omit `--experiment_tag`. **Always pass `--experiment_tag` explicitly when comparing variations** so auto-generated hostname tags do not collide confusingly.

### Phase 1 — train + merge one loss

```bash
python deploy/aip_run.py --config a100_qwen --loss rev_kl --train --no_smoke
```

Or train + eval in one go with a baseline tag:

```bash
python deploy/aip_run.py --config a100_qwen --loss rev_kl --no_smoke \
  --experiment_tag rev_kl_baseline
```

`bv_tree` / `gbv_tree` — use lower LR (`1e-5`); `aip_run.py` auto-applies for single-loss runs, or call `experiment.py --lr 1e-5` directly.

### Phase 2 — eval variations (same checkpoint, different tags)

**Add verifiers without redoing finished modes** (default: skip cells already in DB):

```bash
# First run: modes [bv, gbv] in YAML (or default 9 modes)
python deploy/aip_run.py --config a100_qwen --loss rev_kl --eval --no_smoke \
  --experiment_tag rev_kl_bv_gbv

# Second run: add traversal — only missing modes run; bv/gbv skipped if already saved
# (use a NEW tag if this is a distinct experiment condition you want to compare)
python deploy/aip_run.py --config a100_qwen --loss rev_kl --eval --no_smoke \
  --experiment_tag rev_kl_bv_gbv_traversal
```

**Re-run the same cells** (new DB rows, old rows kept) — e.g. after a code fix or hyperparameter change:

```bash
python deploy/aip_run.py --config a100_qwen --loss rev_kl --eval --force --no_smoke \
  --experiment_tag rev_kl_L8_rerun_v2
```

**Surgical eval** (one loss, custom modes — bypasses pipeline YAML):

```bash
python orchestration/evaluate.py \
  --student db/checkpoints/rev_kl-gsm8k-q0.6b-q8b_merged \
  --teacher Qwen/Qwen3-8B \
  --student_label rev_kl \
  --datasets gsm8k_eval \
  --modes bv,gbv,traversal \
  --K 3 --n 100 --L 8 \
  --hw_tier a100 \
  --experiment_tag rev_kl_verifier_slice \
  --skip_existing
```

### Phase 3 — next loss

```bash
python deploy/aip_run.py --config a100_qwen --loss gbv_tree --train --no_smoke
python deploy/aip_run.py --config a100_qwen --loss gbv_tree --eval --no_smoke \
  --experiment_tag gbv_tree_baseline
```

### Flags cheat sheet


| Goal                      | Command flags                                                 |
| ------------------------- | ------------------------------------------------------------- |
| One loss only             | `--loss rev_kl` or `--losses rev_kl`                          |
| Train only                | `--train` / `--train_only`                                    |
| Eval only (skip baseline) | `--eval` / `--eval_only`                                      |
| Resume after crash        | `--resume` (skips done pipeline steps + DB cells)             |
| Re-run eval cells         | `--force` (alias `--force_eval`) + **new** `--experiment_tag` |
| Label this variation      | `--experiment_tag <name>` (required for comparisons)          |
| Start mid-pipeline        | `--from merge_rev_kl_gsm8k`                                   |


`**--resume` + `--force`:** skips smoke and done train/merge steps, but **re-runs eval steps** and **all eval cells** (new rows). Use with a fresh `--experiment_tag`.

---

## Training (one loss at a time)

`--losses` alone runs **baseline eval** first, then train/merge/eval for that loss. If a checkpoint already exists, training is skipped.

**Train only (no baseline, no eval):**

```bash
python deploy/aip_run.py --config a100_qwen --loss traversal_tree --train --no_smoke
```

**Train + merge + eval for one loss:**

```bash
python deploy/aip_run.py --config a100_qwen --loss traversal_tree --no_smoke \
  --experiment_tag trav_tree_baseline
```

**Start at the training step (skip baseline):**

```bash
python deploy/aip_run.py --config a100_qwen --loss traversal_tree \
  --from train_trav_tree_gsm8k --no_smoke
```

**Force re-train** (delete checkpoint + reset state first):

```bash
rm -rf "$GBV_DIR/db/checkpoints/trav_tree-gsm8k-q0.6b-q8b"
rm -rf "$GBV_DIR/db/checkpoints/trav_tree-gsm8k-q0.6b-q8b_merged"
python deploy/aip_run.py --config a100_qwen --loss traversal_tree --train --no_smoke
```

Check pipeline state:

```bash
python orchestration/experiment.py --config a100_qwen --status
```

Step IDs for `traversal_tree`:


| Step  | ID                      |
| ----- | ----------------------- |
| Train | `train_trav_tree_gsm8k` |
| Merge | `merge_trav_tree_gsm8k` |
| Eval  | `eval_trav_tree_gsm8k`  |


Checkpoint dir: `$GBV_DIR/db/checkpoints/trav_tree-gsm8k-q0.6b-q8b`

**Monitor:**

```bash
tail -f "$GBV_DIR/db/logs/a100_qwen-q0.6b-q8b/pipeline_output.log"
python orchestration/experiment.py --config a100_qwen --status
```

---

## Resume after disconnect or kill

Do **not** use `clean_restart` if you want to keep checkpoints.

```bash
source ~/.specdist_env && cd "$GBV_DIR"

pkill -TERM -f "experiment.py --config a100_qwen"
sleep 5
pkill -9 -f "experiment.py --config a100_qwen"   # only if still alive

python deploy/aip_run.py --config a100_qwen --resume
```

- **Pipeline step** (`eval_rev_kl_gsm8k` in `pipeline_state_*.json`): if marked **done**, that whole eval subprocess is skipped.
- **Eval cell** (one row in `results.db`): if present and `--skip_existing` (default), that `(loss, dataset, mode, K, T)` combo is skipped inside a running eval.

If eval was **interrupted** mid-run, the step stays **failed/pending**, `--resume` re-launches eval, and only **missing cells** run.

Training resumes from `ckpt_latest` + `training_state.json`. W&B training run resumes via `wandb_run.json` beside the checkpoint.

---

## Evaluation

Eval runs automatically after train+merge. Default modes (in `bases/a100.yaml`): `alpha`, `naive`, `nss`, `specinfer`, `spectr`, `khisti`, `bv`, `gbv`, `traversal`.

Eval uses up to **3 GPUs in parallel** by default. Override:

```bash
SPECDIST_MAX_EVAL_GPUS=1 python deploy/aip_run.py ...   # force serial
```

Results are written to `**results.db` and W&B as each verifier mode finishes** (not only at end of run).

---

## Comparing results (filter by `experiment_tag`)

Every eval cell stores:


| Field                                      | Use for                                          |
| ------------------------------------------ | ------------------------------------------------ |
| `experiment_tag`                           | **Which variation / run condition**              |
| `draft_label`                              | **Which loss** (`rev_kl`, `gbv_tree`, …)         |
| `mode`                                     | Verifier algorithm (`bv`, `gbv`, `traversal`, …) |
| `dataset`, `K`, `temperature`, `n_prompts` | Eval slice                                       |
| `block_eff`, `alpha_mean`, `task_score`    | Metrics                                          |
| `run_tag`                                  | Unique row id (timestamp); auto-generated        |


### List tags and row counts

```bash
cd "$GBV_DIR"
python -c "
import sys; sys.path.insert(0, 'db'); import results_db
from collections import Counter
rows = results_db.query_runs()
for tag in results_db.distinct_values('experiment_tag'):
    n = sum(1 for r in rows if r.get('experiment_tag') == tag)
    losses = sorted({r['draft_label'] for r in rows if r.get('experiment_tag') == tag})
    print(f'{tag}: {n} rows  losses={losses}')
"
```

### Compare two tags for one loss (block efficiency)

```bash
python -c "
import sys; sys.path.insert(0, 'db'); import results_db
TAGS = ['rev_kl_baseline', 'rev_kl_L5']
LOSS = 'rev_kl'
rows = [r for r in results_db.query_runs()
        if r.get('draft_label') == LOSS
        and r.get('experiment_tag') in TAGS
        and r.get('block_eff') is not None]
rows.sort(key=lambda r: (r['experiment_tag'], r['mode'], r['dataset']))
print(f\"{'tag':<28} {'mode':<12} {'dataset':<10} {'K':>2} {'BE':>8}\")
for r in rows:
    print(f\"{r.get('experiment_tag',''):<28} {r['mode']:<12} {r['dataset']:<10} {r['K']:>2} {r['block_eff']:>8.4f}\")
"
```

### Paper metrics table (α, BE, GSM8K EM) — one `experiment_tag`

Filter by **`experiment_tag` only** (do not filter `dataset` until you know what is
in your DB). Rows from smoke runs or older configs may use `gsm8k`; current A100
finalization uses `gsm8k_eval` — both are valid GSM8K eval pools.

```bash
cd "$GBV_DIR"
python -c "
import sys; sys.path.insert(0, 'db'); import results_db
TAG = 'KrishnanRIITHServer'   # logging.experiment_tag in a100_qwen.yaml, or your --experiment_tag
rows = [r for r in results_db.query_runs()
        if r.get('experiment_tag') == TAG]
labels = ['baseline', 'l1', 'kl', 'jsd', 'rev_kl']
print(f\"{'Model':<12} {'α (alpha)':>20} {'BE (gbv)':>10} {'GSM8K EM':>10}\")
for lab in labels:
    alpha = next((r for r in rows if r['draft_label']==lab and r['mode']=='alpha'), {})
    gbv   = next((r for r in rows if r['draft_label']==lab and r['mode']=='gbv'), {})
    em    = alpha.get('task_score')
    print(f\"{lab:<12} {str(alpha.get('alpha_mean') or '-'):>20} {str(gbv.get('block_eff') or '-'):>10} {str(em if em is not None else '-'):>10}\")
"
```

Optional: restrict to one dataset once you know the name:

```bash
python -c "
import sys; sys.path.insert(0, 'db'); import results_db
rows = results_db.query_runs()
tags = sorted({r.get('experiment_tag') for r in rows if r.get('experiment_tag')})
dss  = sorted({r.get('dataset') for r in rows if r.get('dataset') and 'gsm8k' in r.get('dataset')})
print('experiment_tags:', tags[:10], '...')
print('gsm8k* datasets :', dss)
"
```

### Troubleshooting results.db (table shows `-` for most losses)

| Symptom | Cause | Fix |
| -------- | ----- | --- |
| `baseline` / `l1` / `kl` all `-` | Those eval steps never ran for this tag | Run baseline: `python deploy/aip_run.py --config a100_qwen --from eval_baseline_gsm8k --no_smoke`. Then per loss: `--loss kl --eval --force --no_smoke` |
| Only `jsd` has `BE (gbv)`, no α or EM | Eval interrupted after some verifier modes, or α skipped by `--skip_existing` | `python deploy/aip_run.py --config a100_qwen --loss jsd --eval --force --no_smoke` |
| Query with `dataset='gsm8k_eval'` returns nothing | Rows stored as `gsm8k` (smoke/older run) | Drop the dataset filter, or list datasets (snippet above) |
| Rows exist but wrong tag | `--eval --loss X` skips baseline; auto-generated tags differ from YAML | Pass `--experiment_tag KrishnanRIITHServer` on every eval; or query without tag filter |
| Everything empty | Wrong `results.db` | `python -c "import sys; sys.path.insert(0,'db'); import results_db; print(results_db.DB_PATH)"` — must match `aip_run` storage line |

**Why `--skip_existing` hides rows:** `evaluate.py` skips a cell when *any* prior row
exists for `(draft_label, dataset, mode, K)` — **ignoring `experiment_tag`**. A smoke
run can block a full run from inserting α/EM under your tag while still writing a new
`gbv` row. Use `--force` (alias `--force_eval`) to re-insert all cells.

**`--eval --loss jsd` skips baseline** by design (`--from eval_jsd_gsm8k`). For a full
comparison table you need `eval_baseline_gsm8k` plus each loss's `eval_*_gsm8k` step.

### SQLite (direct)

```bash
sqlite3 "$GBV_DIR/db/results.db" \
  "SELECT experiment_tag, draft_label, mode, dataset, K, block_eff
   FROM runs WHERE experiment_tag LIKE 'rev_kl%' ORDER BY experiment_tag, mode;"
```

### W&B

- Each eval invocation creates a W&B run with `config.experiment_tag` and tag `experiment_tag`.
- **Compare variations:** W&B → Runs → filter/group by `experiment_tag` or `tags`.
- **Compare losses:** filter by `config.student_label` / `config.loss_name`.
- Per-cell metrics also land in `wandb.summary` as eval completes (e.g. `BE/gsm8k_eval/gbv/K3`).

### Markdown report (all data in DB)

```bash
python db/analyze_results.py --out report.md
python db/analyze_results.py --section be --compare rev_kl,gbv_tree
```

### Dashboard (optional)

```bash
python OSD/viz_server.py   # sibling OSD repo — reads results.db, supports experiment_tag filter
```

---

## Where logs live


| What                | Path                                                           |
| ------------------- | -------------------------------------------------------------- |
| Pipeline stdout     | `$GBV_DIR/db/logs/a100_qwen-q0.6b-q8b/pipeline_output.log`     |
| Per-step errors     | `$GBV_DIR/db/logs/a100_qwen-q0.6b-q8b/step_<id>_error.log`     |
| W&B local run files | `/home/colligo/ram/specdist/wandb/wandb/run-<date>-<id>/logs/` |
| W&B dashboard       | URL printed as `[wandb] https://wandb.ai/...` in pipeline log  |


```bash
tail -f "$GBV_DIR/db/logs/a100_qwen-q0.6b-q8b/pipeline_output.log"
```

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

## Git pull: fix "URL rejected: Bad hostname"

```bash
cd /home/colligo/ram/Distill-Spec-Research
git remote set-url origin https://github.com/Rmuk655/Distill-Spec-Research.git
git pull
bash gbv-research/deploy/aip_gpu_setup.sh a100_qwen
```

---

## FAQ

### Training crashed — how do I restart from the checkpoint?

**Do not** run `clean_restart` — that wipes pipeline state. Training resume is automatic as long as the checkpoint directory is intact.

**1. Kill any orphaned job** (SSH disconnect, OOM kill, Ctrl-C):

```bash
source ~/.specdist_env && cd "$GBV_DIR"

pkill -TERM -f "experiment.py --config a100_qwen"
sleep 5
pkill -9 -f "experiment.py --config a100_qwen"   # only if still alive
```

**2. Confirm checkpoint files exist** (example: `kl` on `a100_qwen`):

```bash
CKPT="$GBV_DIR/db/checkpoints/kl-gsm8k-q0.6b-q8b"
ls -la "$CKPT/ckpt_latest/adapter_config.json" \
       "$CKPT/training_state.json" \
       "$CKPT/wandb_run.json"
```


| File                  | Purpose                                                        |
| --------------------- | -------------------------------------------------------------- |
| `ckpt_latest/`        | Latest LoRA weights (rolling save)                             |
| `training_state.json` | Optimizer + global step (e.g. step 2801/4000)                  |
| `wandb_run.json`      | W&B run id so training continues on the **same dashboard run** |


**3. Restart training** — pick the loss and training step id:

```bash
python deploy/aip_run.py --config a100_qwen --loss kl \
  --from train_kl_gsm8k --no_smoke
```

Equivalent if you were already mid-pipeline:

```bash
python deploy/aip_run.py --config a100_qwen --loss kl --resume --no_smoke
```

**4. Verify resume in the log** — you should see:

```text
[RESUME] Continuing from step 2801/4000
[wandb] Resuming run eyp9xg22 (from wandb_run.json)
```

W&B local files live under `$WANDB_DIR` (e.g. `/home/colligo/ram/specdist/wandb/wandb/run-.../logs/`). That path is just the **local cache**. Resume uses `wandb_run.json` **inside the checkpoint dir**, not the log folder path.

**Step IDs** (use with `--from`):


| Loss             | Train step id           | Checkpoint dir               |
| ---------------- | ----------------------- | ---------------------------- |
| `kl`             | `train_kl_gsm8k`        | `kl-gsm8k-q0.6b-q8b/`        |
| `rev_kl`         | `train_rev_kl_gsm8k`    | `rev_kl-gsm8k-q0.6b-q8b/`    |
| `gbv_tree`       | `train_gbv_tree_gsm8k`  | `gbv_tree-gsm8k-q0.6b-q8b/`  |
| `traversal_tree` | `train_trav_tree_gsm8k` | `trav_tree-gsm8k-q0.6b-q8b/` |


List all step ids: `python orchestration/experiment.py --config a100_qwen --dry_run --losses kl`

`**[RECOVER] … Reset to 'pending'`** after a crash is normal — the pipeline re-runs interrupted steps; training still resumes from `ckpt_latest`.

**Start over from step 0** (only if you intend to discard progress):

```bash
rm -rf "$GBV_DIR/db/checkpoints/kl-gsm8k-q0.6b-q8b"
python deploy/aip_run.py --config a100_qwen --loss kl --train --no_smoke
```

---

## Related docs


| Doc                                                                                                                   | Use when                                   |
| --------------------------------------------------------------------------------------------------------------------- | ------------------------------------------ |
| [deploy/README.md](README.md)                                                                                         | All providers (Colab, Kaggle, Modal, …)    |
| [docs/HYPERPARAMETERS_AND_DATA.md](../docs/HYPERPARAMETERS_AND_DATA.md)                                               | Eval temps, datasets, `--force` examples   |
| [orchestration/configs/profiles/a100_verifier_sweep.yaml](../orchestration/configs/profiles/a100_verifier_sweep.yaml) | Eval-only all-verifier sweep across losses |
| [deploy/ats/README.md](ats/README.md)                                                                                 | **CPU-only** server (no GPU) — GPT-2 pair  |


