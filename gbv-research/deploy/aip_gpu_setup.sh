#!/usr/bin/env bash
# =============================================================================
# aip_gpu_setup.sh — One-shot setup for Adobe AI Platform GPU sessions
#
# Run in the AIP VS Code terminal AFTER creating the session:
#   bash Distill-Spec-Research/gbv-research/deploy/aip_gpu_setup.sh
#   bash Distill-Spec-Research/gbv-research/deploy/aip_gpu_setup.sh a10_qwen
#   bash Distill-Spec-Research/gbv-research/deploy/aip_gpu_setup.sh a100_qwen
#
# GPU instance → CONFIG mapping:
#   g5.12xlarge  (NVIDIA A10G,  4×24 GB)  → a10_qwen   (Qwen3-8B BF16)
#   p4d.24xlarge (NVIDIA A100, 40 GB)      → a100_qwen  (Qwen3-8B BF16, paper)
#   p4de.24xlarge(NVIDIA A100, 80 GB)      → a100_qwen  (same, more headroom)
#
# Docker image to select in AIP UI:
#   Training Base (CUDA 12.8.1, Ubuntu 24.04, CUDNN 9.18.1.3, NCCL 2.29.7, Python 3.12)
#
# GPUs per pod:
#   1 GPU — single run at a time (safe default)
#   4 GPUs — pipeline runs 4 losses in parallel (faster; for overnight runs)
#
# Persistent storage: Sensei FS at /sensei-fs-3/users/$USER
# Secrets: export WANDB_API_KEY / HF_TOKEN / GITHUB_TOKEN in terminal first.
# =============================================================================

set -euo pipefail

# ── CONFIG (override via arg: bash aip_gpu_setup.sh a10_qwen) ────────────────
CONFIG="${1:-a10_qwen}"    # a10_qwen | a100_qwen | colab_lite | server_gpt2

# ── Paths ─────────────────────────────────────────────────────────────────────
LDAP="${USER:-$(whoami)}"
SENSEI_ROOT="/sensei-fs-3/users/${LDAP}"
REPO_URL="https://github.com/Rmuk655/Distill-Spec-Research.git"
REPO_DIR="${SENSEI_ROOT}/Distill-Spec-Research"
GBV_DIR="${REPO_DIR}/gbv-research"
STORAGE_ROOT="${SENSEI_ROOT}/specdist"
HF_CACHE="${STORAGE_ROOT}/hf_cache"

echo "========================================================================"
echo "  AIP GPU Setup — config=${CONFIG}"
echo "  Sensei FS : ${SENSEI_ROOT}"
echo "  Storage   : ${STORAGE_ROOT}"
echo "========================================================================"
echo ""

# ── 1. Verify Sensei FS ───────────────────────────────────────────────────────
if [ ! -d "${SENSEI_ROOT}" ]; then
    echo "ERROR: Sensei FS not found at ${SENSEI_ROOT}"
    echo "  Check available paths: ls /sensei-fs-3/users/"
    echo "  Set SENSEI_ROOT manually if your path differs."
    exit 1
fi
echo "[1/6] Sensei FS accessible: ${SENSEI_ROOT}"

# ── 2. Clone or update repo ───────────────────────────────────────────────────
GH_TOKEN="${GITHUB_TOKEN:-}"
if [ -n "${GH_TOKEN}" ]; then
    CLONE_URL=$(echo "${REPO_URL}" | sed "s|https://|https://${GH_TOKEN}@|")
else
    CLONE_URL="${REPO_URL}"
    echo "      WARNING: GITHUB_TOKEN not set — will fail for private repos"
fi

if [ ! -d "${REPO_DIR}/.git" ]; then
    echo "[2/6] Cloning repo to Sensei FS..."
    git clone --depth 1 "${CLONE_URL}" "${REPO_DIR}"
else
    echo "[2/6] Updating repo..."
    if [ -n "${GH_TOKEN}" ]; then
        git -C "${REPO_DIR}" remote set-url origin "${CLONE_URL}" 2>/dev/null || true
    fi
    git -C "${REPO_DIR}" pull --ff-only || echo "      (pull skipped — not fast-forward)"
fi

# ── 3. Install Python dependencies ───────────────────────────────────────────
echo "[3/6] Installing dependencies (torch + ML stack)..."

# PyTorch: CUDA 12.8 wheel matches the Training Base docker image
pip install --quiet --upgrade pip
pip install --quiet \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

pip install --quiet \
    transformers>=4.51 \
    peft>=0.10 \
    accelerate \
    bitsandbytes>=0.46.1 \
    datasets \
    sentencepiece \
    wandb \
    flask \
    pyyaml \
    scipy

echo "      Dependencies installed."

# ── 4. Set environment variables ──────────────────────────────────────────────
echo "[4/6] Configuring environment..."

# HF cache on Sensei FS so models persist across sessions
mkdir -p "${HF_CACHE}"
export HF_HOME="${HF_CACHE}"
export TRANSFORMERS_CACHE="${HF_CACHE}"
# Allow downloading on first run; experiment.py sets OFFLINE after models are cached
unset TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE HF_HUB_OFFLINE 2>/dev/null || true

# Authenticate HuggingFace (for gated models like LLaMA)
if [ -n "${HF_TOKEN:-}" ]; then
    huggingface-cli login --token "${HF_TOKEN}" --add-to-git-credential 2>/dev/null || true
    echo "      HuggingFace: authenticated"
else
    echo "      HuggingFace: HF_TOKEN not set (not needed for Qwen3)"
fi

# Authenticate W&B
if [ -n "${WANDB_API_KEY:-}" ]; then
    python -c "import wandb; wandb.login(key='${WANDB_API_KEY}', relogin=True)" 2>/dev/null && \
        echo "      W&B: authenticated" || echo "      W&B: login failed (check WANDB_API_KEY)"
else
    echo "      W&B: WANDB_API_KEY not set — W&B logging disabled"
fi

# Write persistent env file so new terminals pick up the vars
ENVFILE="${SENSEI_ROOT}/.specdist_env"
cat > "${ENVFILE}" <<EOF
export HF_HOME="${HF_CACHE}"
export TRANSFORMERS_CACHE="${HF_CACHE}"
export REPO_DIR="${REPO_DIR}"
export GBV_DIR="${GBV_DIR}"
export STORAGE_ROOT="${STORAGE_ROOT}"
EOF
echo "      Env saved to ${ENVFILE} (source it in new terminals)"

# ── 5. Download training data ─────────────────────────────────────────────────
echo "[5/6] Checking training data..."
DATA_DIR="${GBV_DIR}/core/datasets/raw"
if [ ! -f "${DATA_DIR}/gsm8k_train.jsonl" ]; then
    echo "      Downloading gsm8k_train.jsonl (~3 MB)..."
    python "${GBV_DIR}/core/datasets/downloader.py" --quiet 2>/dev/null || \
        echo "      (downloader not available — experiment.py will fetch on first run)"
else
    echo "      gsm8k_train.jsonl already present."
fi

# ── 6. Verify GPU + print launch command ─────────────────────────────────────
echo "[6/6] GPU check..."
python - <<'PYEOF'
import subprocess, sys
try:
    import torch
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            gb = p.total_memory / 1024**3
            print(f"      GPU {i}: {p.name}  {gb:.0f} GB VRAM")
    else:
        print("      No CUDA GPUs detected — check instance type")
except ImportError:
    print("      torch not importable — dependency install may have failed")
PYEOF

echo ""
echo "========================================================================"
echo "  Setup complete! Launch with:"
echo ""
echo "  # Smoke test first (always):"
echo "  cd ${GBV_DIR}"
echo "  python orchestration/experiment.py --config ${CONFIG} --smoke --yes \\"
echo "    --storage_root ${STORAGE_ROOT}"
echo ""
echo "  # Full run (one loss at a time):"
echo "  python orchestration/experiment.py --config ${CONFIG} --losses kl --yes \\"
echo "    --storage_root ${STORAGE_ROOT}"
echo ""
echo "  # Resume after session restart:"
echo "  source ${ENVFILE}"
echo "  python orchestration/experiment.py --config ${CONFIG} --yes \\"
echo "    --storage_root ${STORAGE_ROOT}"
echo ""
echo "  # Monitor logs:"
echo "  tail -f ${STORAGE_ROOT}/logs/${CONFIG}-*/pipeline_output.log"
echo "========================================================================"
