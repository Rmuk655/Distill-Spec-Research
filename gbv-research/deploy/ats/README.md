# ATS Cloud — Bare-Metal Server Scripts

Adobe ATS Cloud · Ubuntu 24.04 · 128 GB RAM · 2-socket CPU · **no GPU**

Model pair: **distilgpt2 (82M) → gpt2-medium (355M)**  
Purpose: prove convergence trends when GPU credits (Kaggle/Colab) are exhausted.

---

## Complete Workflow

Run scripts one at a time in order. Each is standalone and idempotent.

```
Step 0   Configure environment (every SSH session)
Step 1   One-time setup — install deps, download models
Step 2   Verify setup — check every prerequisite
Step 3   Smoke test — 10-step crash test on this hardware (~15 min)
Step 4   Baseline eval — untrained model reference (~2-3 hr)
Step 5   Parallel training — all 15 losses simultaneously (~4-6 hr)
Step 6   Eval trained models — merge + evaluate each (~2-3 hr/model)
Step 7   Dashboard — visualise results locally
```

Monitor is a sidecar: run it from a second SSH session any time.

---

## Step-by-step

### Step 0 — Configure environment (every new SSH session)
```bash
# Edit WANDB_API_KEY in the file first!
nano deploy/ats/00_env.sh

source deploy/ats/00_env.sh
```

### Step 1 — One-time setup
```bash
bash deploy/ats/01_setup.sh
```
Installs pip deps · downloads GSM8K dataset · downloads GPT-2 weights (~1.7 GB) · logs into W&B.

### Step 2 — Verify setup
```bash
python deploy/ats/02_verify.py
```
Checks Python version · imports · RAM · model family registry · cached weights · datasets · W&B.  
Exit 0 = ready. Exit 1 = fix and re-run 01_setup.sh.

### Step 3 — Smoke test (~15 min)
```bash
python deploy/ats/03_smoke.py
```
10-step training for 7 representative losses · verifier run · perplexity check.  
PASS = hardware works, proceed. FAIL = see logs in `db/logs/ats/smoke_*.log`.

### Step 4 — Baseline evaluation (~2-3 hr)
Run in a screen session so it survives SSH disconnect:
```bash
screen -S baseline
source deploy/ats/00_env.sh
python deploy/ats/04_baseline_eval.py
# Ctrl+A D to detach; screen -r baseline to reattach
```
Evaluates **untrained** distilgpt2 on 5 prompts × 6 verifier modes.  
Results → `db/results.db`. W&B group: `ats-gpt2-baseline`.

Faster option (2 prompts, ~1 hr):
```bash
python deploy/ats/04_baseline_eval.py --n 2
```

### Step 5 — Parallel training (~4-6 hr)
SSH session 1 — training:
```bash
screen -S training
source deploy/ats/00_env.sh
python deploy/ats/05_train_parallel.py
# Ctrl+A D to detach
```

SSH session 2 — monitor while training:
```bash
python deploy/ats/monitor.py --watch           # live table, refresh 30s
python deploy/ats/monitor.py --tail kl         # tail kl loss log
python deploy/ats/monitor.py --summary         # one-line status
```

Specific losses only:
```bash
python deploy/ats/05_train_parallel.py --losses kl,rev_kl,jsd --steps 500
```

W&B: each loss creates its own run under group `ats-gpt2-train`.  
Logs: `db/logs/ats/train_<loss>.log`.

### Step 6 — Eval trained models (~2-3 hr per model, 30-45 hr total)
```bash
screen -S eval
source deploy/ats/00_env.sh
python deploy/ats/06_eval_trained.py
# Ctrl+A D to detach
```

This merges each LoRA adapter into a full model, then evaluates it.  
W&B group: `ats-gpt2-eval`. Logs: `db/logs/ats/eval_<loss>.log`.

Specific losses or fewer prompts:
```bash
python deploy/ats/06_eval_trained.py --losses kl,rev_kl --n 2
```

### Step 7 — Dashboard (run any time, from any step)
```bash
bash deploy/ats/07_dashboard.sh
```

**Access from your laptop via SSH tunnel:**
```bash
# In a local terminal (not on the server):
ssh -L 5000:localhost:5000 user@<ats-server-ip>

# Then open in browser:
http://localhost:5000
```

The dashboard shows:
- Training loss curves (after step 5)
- Block efficiency table vs baseline (after step 6)
- Model family filter: **DistilGPT-2** chip appears when GPT-2 results are in the DB

---

## Timing estimates (CPU, all 15 losses)

| Step | Wall time | Notes |
|------|-----------|-------|
| Smoke (step 3) | ~15 min | 7 losses × 10 steps |
| Baseline eval (step 4) | ~2-3 hr | 5 prompts × 6 modes on CPU |
| Training (step 5) | ~4-6 hr | all 15 losses in parallel |
| Eval trained (step 6) | ~30-45 hr | 15 models × 2-3 hr each, sequential |
| **Total** | **~37-55 hr** | run over 2-3 days unattended |

---

## W&B setup

Edit `deploy/ats/00_env.sh`:
```bash
export WANDB_API_KEY="your_key_from_wandb.ai/authorize"
export WANDB_ENTITY="your_username"
export WANDB_PROJECT="distillspec"
```

If `WANDB_API_KEY` is not set, training runs log offline (results still go to `results.db`).  
You can sync offline runs later: `wandb sync db/wandb/offline-run-*`

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Model not cached | `bash deploy/ats/01_setup.sh` |
| `TRANSFORMERS_OFFLINE=1` error | `source deploy/ats/00_env.sh` |
| Monitor shows FAIL | `python deploy/ats/monitor.py --tail <loss>` to see error |
| Eval too slow | `--n 2` for 2 prompts instead of 5 |
| W&B offline | Set `WANDB_API_KEY` in `00_env.sh` |
| Too many parallel jobs | `--losses kl,rev_kl,jsd` to subset |
