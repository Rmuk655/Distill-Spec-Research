"""One-time script: patch deploy/aip.ipynb with Cell 4 (parallel training).
Run from gbv-research/:  python scripts/add_aip_cell4.py
"""
import json, os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path = os.path.join(HERE, "deploy", "aip.ipynb")

with open(path, encoding="utf-8") as f:
    nb = json.load(f)

# ── 1. Update title cell ──────────────────────────────────────────────────────

title_src = "".join(nb["cells"][0]["source"])

# Add Cell 4 to the table
title_src = title_src.replace(
    "| 3. Verify | Check artifacts before session ends | instant |",
    "| 3. Verify | Check artifacts before session ends | instant |\n"
    "| 4. Parallel | All losses simultaneously (multi-GPU or CPU) | ~2-3 h |",
)

# Extend the Multi-GPU section
title_src = title_src.replace(
    "The draft model always trains on `cuda:0`.",
    "The draft model always trains on `cuda:0`.\n\n"
    "**Cell 4 — Parallel training across all losses:**  \n"
    "Assigns each loss group to a separate GPU (or CPU for GPT-2).  \n"
    "3 GPUs x 5 losses each = all 15 losses done in ~2-3 h instead of 30-40 h sequential.  \n"
    "Set `FAMILY = \"gpt2\"` to run CPU convergence runs alongside Qwen GPU runs.",
)

nb["cells"][0]["source"] = [title_src]

# ── 2. New Cell 4 ─────────────────────────────────────────────────────────────

cell4_code = '''\
# =============================================================================
# Cell 4 — PARALLEL TRAINING  (all losses simultaneously)
#
# On AIP with multiple GPUs (up to 3): assigns each loss group to a separate
# GPU so all 15 losses train simultaneously.
#   3 GPUs x 5 losses = ~2-3 h total  (vs 30-40 h sequential on one GPU)
#
# For GPT-2 (FAMILY="gpt2"): launches all losses as CPU background processes.
# No VRAM needed — useful alongside a Qwen GPU run, or on the ATS Cloud server.
#
# Logs go to STORAGE_ROOT/logs/parallel/<family>_<loss>.log
# Results land in the same results.db — dashboard model-family filter
# (DistilGPT-2, LLaMA, Qwen chips) distinguishes pairs automatically.
# =============================================================================

import os, subprocess, sys

# -- Edit these (must match Cell 0) -------------------------------------------
_LDAP        = os.environ.get("USER", "YOUR_LDAP_HERE")
STORAGE_HOME = os.path.expanduser("~")
REPO_DIR     = f"{STORAGE_HOME}/Distill-Spec-Research"
GBV_DIR      = f"{REPO_DIR}/gbv-research"
STORAGE_ROOT = f"{GBV_DIR}/db"

# FAMILY: which model pair to train
#   "gpt2"  — distilgpt2 (82M) → gpt2-medium (355M)  — CPU, no VRAM
#   "llama" — Llama-3.2-1B → 3B-Instruct              — GPU (needs HF login)
#   "qwen"  — Qwen2.5-0.5B → Qwen3-0.6B/8B            — GPU, use a100/kaggle cfg
FAMILY = "gpt2"     # ← "gpt2" for CPU run; "qwen" for multi-GPU Qwen run

STEPS  = 1000       # steps per loss (gpt2: ~2-4 hr/loss CPU; qwen: ~10-17 min/loss A100)

# GPU assignment for FAMILY="qwen" (ignored for CPU families)
# Adjust group sizes to match your GPU count (AIP gives up to 3)
GPU_GROUPS = {
    0: ["kl", "rev_kl", "jsd", "l1", "kl_tree"],
    1: ["rev_kl_tree", "jsd_tree", "bv_tree", "gbv_tree", "traversal_tree"],
    2: ["naive_tree", "nss_tree", "specinfer_tree", "spectr_tree", "khisti_tree"],
}
# Losses for CPU families (ebe/online excluded per docs/ISSUES.md)
CPU_LOSSES = [
    "kl", "rev_kl", "jsd", "l1",
    "kl_tree", "rev_kl_tree", "jsd_tree",
    "bv_tree", "gbv_tree", "traversal_tree",
    "naive_tree", "nss_tree", "specinfer_tree", "spectr_tree", "khisti_tree",
]
# ---------------------------------------------------------------------------

sys.path.insert(0, GBV_DIR)
from core.model_families import get_family

family_obj = get_family(FAMILY)
draft_id   = family_obj.default_draft_model_id
target_id  = family_obj.default_target_model_id
use_cpu    = FAMILY in ("gpt2",)   # add "llama" here if no GPU

trainer    = os.path.join(GBV_DIR, "algorithms", "distillspec_gbv", "trainer.py")
ckpt_root  = os.path.join(STORAGE_ROOT, "checkpoints")
dataset    = os.path.join(GBV_DIR, "core", "datasets", "raw", "gsm8k_train.jsonl")
log_dir    = os.path.join(STORAGE_ROOT, "logs", "parallel")
os.makedirs(log_dir, exist_ok=True)
os.makedirs(ckpt_root, exist_ok=True)

# Make HF online so models can download if not cached
for _flag in ("TRANSFORMERS_OFFLINE", "HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE"):
    os.environ.pop(_flag, None)

procs = []

if use_cpu:
    # ── CPU path: all losses in parallel ─────────────────────────────────────
    print(f"Launching {len(CPU_LOSSES)} {FAMILY} losses on CPU in parallel...")
    print(f"  draft={draft_id}  target={target_id}  steps={STEPS}")
    for loss in CPU_LOSSES:
        log_path = os.path.join(log_dir, f"{FAMILY}_{loss}.log")
        log_f = open(log_path, "w")
        p = subprocess.Popen(
            [
                sys.executable, trainer,
                "--model_family", FAMILY,
                "--draft",        draft_id,
                "--target",       target_id,
                "--loss",         loss,
                "--steps",        str(STEPS),
                "--device",       "cpu",
                "--no_wandb",
                "--teacher_temp", "1.0",
                "--dataset",      dataset,
                "--output",       os.path.join(ckpt_root, f"{loss}-{FAMILY}"),
            ],
            stdout=log_f, stderr=subprocess.STDOUT, cwd=GBV_DIR,
        )
        procs.append((loss, p, log_f))
        print(f"  [{loss:20s}] PID {p.pid:6d}  log: parallel/{FAMILY}_{loss}.log")

else:
    # ── GPU path: one group per GPU (Qwen / LLaMA with GPU) ──────────────────
    import torch
    n_gpus = torch.cuda.device_count()
    print(f"Detected {n_gpus} GPU(s). Launching loss groups across GPUs...")
    print(f"  draft={draft_id}  target={target_id}  steps={STEPS}")
    for gpu_idx, losses in GPU_GROUPS.items():
        if gpu_idx >= n_gpus:
            print(f"  GPU {gpu_idx}: not available ({n_gpus} GPUs total) — skipping")
            continue
        for loss in losses:
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_idx)}
            log_path = os.path.join(log_dir, f"{FAMILY}_{loss}_gpu{gpu_idx}.log")
            log_f = open(log_path, "w")
            p = subprocess.Popen(
                [
                    sys.executable, trainer,
                    "--model_family", FAMILY,
                    "--draft",        draft_id,
                    "--target",       target_id,
                    "--loss",         loss,
                    "--steps",        str(STEPS),
                    "--device",       "cuda",
                    "--no_wandb",
                    "--dataset",      dataset,
                    "--output",       os.path.join(ckpt_root, f"{loss}-{FAMILY}"),
                ],
                stdout=log_f, stderr=subprocess.STDOUT,
                env=env, cwd=GBV_DIR,
            )
            procs.append((loss, p, log_f))
            print(f"  GPU{gpu_idx} [{loss:20s}] PID {p.pid:6d}")

print(f"\\n{len(procs)} jobs running. Waiting (this cell blocks until all finish)...")
print(f"Tail a log: tail -f {os.path.join(log_dir, FAMILY + '_kl.log')}")

failed = []
for loss, p, log_f in procs:
    rc = p.wait()
    log_f.close()
    status = "OK" if rc == 0 else f"FAILED rc={rc}"
    print(f"  [{loss:20s}] {status}")
    if rc != 0:
        failed.append(loss)

if failed:
    print(f"\\n{len(failed)} losses failed: {failed}")
    print(f"Tail logs in: {log_dir}")
else:
    print(f"\\nAll {len(procs)} losses trained successfully!")
    print("Next: run experiment.py --eval_only or use the eval step in Cell 0/1.")
'''

new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "id": "aip-parallel-train",
    "metadata": {},
    "outputs": [],
    "source": cell4_code.splitlines(keepends=True),
}

# Check it's not already there
ids = [c.get("id") for c in nb["cells"]]
if "aip-parallel-train" not in ids:
    nb["cells"].append(new_cell)
    print(f"Added Cell 4. Total cells: {len(nb['cells'])}")
else:
    # Replace existing
    for i, c in enumerate(nb["cells"]):
        if c.get("id") == "aip-parallel-train":
            nb["cells"][i] = new_cell
    print("Replaced existing Cell 4.")

with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("Saved:", path)
