#!/usr/bin/env bash
# runpod.sh — One-command SpecDist setup on RunPod.
#
# RunPod gives on-demand community GPUs (RTX 3090/4090, ~$0.20–0.50/hr) or
# secure A100/H100 cloud GPUs.  /workspace is a persistent network volume.
#
# Prerequisites:
#   1. Create a pod at https://runpod.io — choose "PyTorch 2.1" template
#      so Python, CUDA, and torch are pre-installed.
#   2. Set RunPod secrets (pod settings → Environment Variables):
#        WANDB_API_KEY  — https://wandb.ai/authorize
#        HF_TOKEN       — https://huggingface.co/settings/tokens
#   3. Open the pod terminal (Connect → Start Web Terminal) and run:
#        bash /workspace/runpod.sh
#      OR paste this URL into the terminal:
#        curl -fsSL https://raw.githubusercontent.com/Rmuk655/Distill-Spec-Research/main/gbv-research/deploy/runpod.sh | bash
#
# What this script does:
#   1. Clones / updates the repo into /workspace/Distill-Spec-Research
#   2. Installs Python dependencies
#   3. Runs the full pipeline with server config and checkpoints on /workspace
#
# To resume after pod restart:
#   bash /workspace/runpod.sh          # idempotent — skips completed steps

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
REPO_URL     ="https://github.com/Rmuk655/Distill-Spec-Research.git"
REPO_DIR     ="/workspace/Distill-Spec-Research"
GBV_DIR      ="${REPO_DIR}/gbv-research"
STORAGE_ROOT ="/workspace/specdist"          # ALL artifacts: DB + checkpoints + logs
HF_HOME      ="/workspace/hf_cache"
CONFIG       ="${SPECDIST_CONFIG:-server}"   # override: SPECDIST_CONFIG=colab bash runpod.sh
SMOKE        ="${SPECDIST_SMOKE:-}"         # set to "--smoke" for a quick test
LOSSES       ="${SPECDIST_LOSSES:-}"        # e.g. "kl,ebe" — leave empty for all

# ── Auth ──────────────────────────────────────────────────────────────────────
echo "=== SpecDist RunPod Setup ==="
echo "  Repo        : ${REPO_DIR}"
echo "  Storage root: ${STORAGE_ROOT}"
echo "  Config      : ${CONFIG}"
echo ""

# W&B (read from env var set in RunPod secrets)
if [ -n "${WANDB_API_KEY:-}" ]; then
    echo "[auth] W&B key found — logging enabled"
else
    echo "[auth] WANDB_API_KEY not set — W&B runs will be offline"
    export WANDB_MODE=offline
fi

# HuggingFace
if [ -n "${HF_TOKEN:-}" ]; then
    export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
    echo "[auth] HF token found"
fi

export HF_HOME="${HF_HOME}"
export TRANSFORMERS_OFFLINE=0      # always online on RunPod (we're on fast cloud)
export HF_HUB_OFFLINE=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Specdist storage env vars — experiment.py propagates these to all subprocesses
export SPECDIST_STORAGE_ROOT="${STORAGE_ROOT}"
export SPECDIST_DB_PATH="${STORAGE_ROOT}/results.db"
export SPECDIST_LOGS_ROOT="${STORAGE_ROOT}/logs"

# ── 1. Clone or update repo ───────────────────────────────────────────────────
mkdir -p /workspace
if [ ! -d "${REPO_DIR}/.git" ]; then
    echo "[setup] Cloning repo..."
    git clone --depth 1 "${REPO_URL}" "${REPO_DIR}"
else
    echo "[setup] Repo already cloned — pulling latest..."
    git -C "${REPO_DIR}" pull --ff-only 2>/dev/null || echo "[setup] Pull failed (local changes?) — continuing"
fi

cd "${GBV_DIR}"

# ── 2. Install dependencies ───────────────────────────────────────────────────
echo "[setup] Installing dependencies..."
pip install -q -r requirements.txt bitsandbytes accelerate
# flash-attn gives 2-4x speedup on A100/H100 — skip silently if build fails
pip install -q flash-attn --no-build-isolation 2>/dev/null || echo "[setup] flash-attn skipped (using sdpa)"

# ── 3. Create storage directory ───────────────────────────────────────────────
mkdir -p "${STORAGE_ROOT}"
mkdir -p "${STORAGE_ROOT}/checkpoints"
mkdir -p "${STORAGE_ROOT}/logs"

# ── 4. Run pipeline ───────────────────────────────────────────────────────────
echo ""
echo "[pipeline] Starting pipeline..."
CMD="python orchestration/experiment.py --config ${CONFIG} --storage_root ${STORAGE_ROOT} --yes"
[ -n "${SMOKE}" ]  && CMD="${CMD} ${SMOKE}"
[ -n "${LOSSES}" ] && CMD="${CMD} --losses ${LOSSES}"
echo "[pipeline] Command: ${CMD}"
echo ""

${CMD}

echo ""
echo "=== Pipeline complete ==="
echo "  Storage root: ${STORAGE_ROOT}"
echo "  Results DB  : ${STORAGE_ROOT}/results.db"
echo "  Checkpoints : ${STORAGE_ROOT}/checkpoints/"
echo ""
echo "  To download results:"
echo "    rsync -avz root@\$(runpodctl get pod \$RUNPOD_POD_ID | grep ip | awk '{print \$NF}'):${STORAGE_ROOT}/results.db ."
echo "    rsync -avz root@\$(runpodctl get pod \$RUNPOD_POD_ID | grep ip | awk '{print \$NF}'):${STORAGE_ROOT}/checkpoints/ ./local_checkpoints/"
