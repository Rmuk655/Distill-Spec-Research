#!/usr/bin/env bash
# =============================================================================
# aip_gpu_setup.sh — One-shot environment setup for GPU compute sessions.
#
# Usage:
#   bash deploy/aip_gpu_setup.sh                    # default: a10_qwen
#   bash deploy/aip_gpu_setup.sh a10_qwen           # A10G 24 GB
#   bash deploy/aip_gpu_setup.sh a100_qwen          # A100 40/80 GB
#   bash deploy/aip_gpu_setup.sh server_gpt2        # CPU-only
#
# Environment variables (set before running):
#   WANDB_API_KEY   — W&B key from https://wandb.ai/authorize
#   HF_TOKEN        — HuggingFace token (optional; Qwen3 is public)
#   GITHUB_TOKEN    — PAT if repo is private
#   STORAGE_ROOT    — where to write checkpoints/logs/results.db
#                     defaults to <repo>/db/ (or $HOME/specdist if writable)
#
# GPU → config:
#   24 GB  (A10G)  →  a10_qwen    Qwen3-8B BF16, 2000 steps, ~80 min/loss
#   40 GB  (A100)  →  a100_qwen   Qwen3-8B BF16, 2000 steps, ~17 min/loss
#   No GPU         →  server_gpt2 GPT-2 CPU-only
# =============================================================================

set -euo pipefail

CONFIG="${1:-a10_qwen}"

# ── Locate the repo ───────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GBV_DIR="$(dirname "$SCRIPT_DIR")"          # gbv-research/
REPO_DIR="$(dirname "$GBV_DIR")"            # Distill-Spec-Research/

# ── Storage root ──────────────────────────────────────────────────────────────
# Priority: env var > $HOME/specdist > local db/
if [ -n "${STORAGE_ROOT:-}" ]; then
    STORAGE="${STORAGE_ROOT}"
elif [ -w "$HOME" ]; then
    STORAGE="$HOME/specdist"
else
    STORAGE="${GBV_DIR}/db"
fi

HF_CACHE="${STORAGE}/hf_cache"

echo "========================================================================"
echo "  SpecDist GPU Setup"
echo "  Config  : ${CONFIG}"
echo "  Repo    : ${GBV_DIR}"
echo "  Storage : ${STORAGE}"
echo "========================================================================"
echo ""

mkdir -p "${STORAGE}" "${HF_CACHE}"

# ── Pull latest code ──────────────────────────────────────────────────────────
GH_TOKEN="${GITHUB_TOKEN:-}"
if [ -n "${GH_TOKEN}" ]; then
    REPO_URL=$(git -C "${REPO_DIR}" remote get-url origin 2>/dev/null || echo "")
    if [ -n "${REPO_URL}" ]; then
        AUTH_URL=$(echo "${REPO_URL}" | sed "s|https://|https://${GH_TOKEN}@|")
        git -C "${REPO_DIR}" remote set-url origin "${AUTH_URL}" 2>/dev/null || true
    fi
fi

echo "[1/5] Pulling latest code..."
git -C "${REPO_DIR}" pull --ff-only 2>/dev/null || echo "      (already up to date or skip)"

# ── Install Python dependencies ───────────────────────────────────────────────
echo "[2/5] Installing dependencies..."

pip install --quiet --upgrade pip

pip install --quiet \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

pip install --quiet \
    "transformers>=4.51" \
    "peft>=0.10" \
    accelerate \
    "bitsandbytes>=0.46.1" \
    datasets \
    sentencepiece \
    wandb \
    flask \
    pyyaml \
    scipy

echo "      Done."

# ── Environment variables ─────────────────────────────────────────────────────
echo "[3/5] Setting environment..."

export HF_HOME="${HF_CACHE}"
export TRANSFORMERS_CACHE="${HF_CACHE}"
unset TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE HF_HUB_OFFLINE 2>/dev/null || true

# Persist env across new terminals
ENVFILE="${HOME}/.specdist_env"
cat > "${ENVFILE}" <<EOF
export HF_HOME="${HF_CACHE}"
export TRANSFORMERS_CACHE="${HF_CACHE}"
export STORAGE_ROOT="${STORAGE}"
export GBV_DIR="${GBV_DIR}"
EOF
echo "      Env saved to ${ENVFILE}  (source it in new terminals)"

# W&B auth
WANDB_KEY="${WANDB_API_KEY:-}"
if [ -n "${WANDB_KEY}" ]; then
    python -c "import wandb; wandb.login(key='${WANDB_KEY}', relogin=True)" 2>/dev/null && \
        echo "      W&B: authenticated" || echo "      W&B: login failed"
else
    echo "      W&B: WANDB_API_KEY not set — set it to enable W&B logging"
fi

# HF auth (optional — Qwen3 is public)
HF_TOKEN="${HF_TOKEN:-}"
if [ -n "${HF_TOKEN}" ]; then
    huggingface-cli login --token "${HF_TOKEN}" --add-to-git-credential 2>/dev/null && \
        echo "      HF:  authenticated" || echo "      HF:  login skipped"
fi

# ── Training data ─────────────────────────────────────────────────────────────
echo "[4/5] Checking training data..."
DATA_FILE="${GBV_DIR}/core/datasets/raw/gsm8k_train.jsonl"
if [ ! -f "${DATA_FILE}" ]; then
    python "${GBV_DIR}/core/datasets/downloader.py" 2>/dev/null || \
        echo "      (experiment.py will download on first run)"
else
    LINES=$(wc -l < "${DATA_FILE}")
    echo "      gsm8k_train.jsonl: ${LINES} prompts"
fi

# ── GPU check ─────────────────────────────────────────────────────────────────
echo "[5/5] GPU check..."
python - <<'PYEOF'
try:
    import torch
    if not torch.cuda.is_available():
        print("      No CUDA GPUs — check instance type")
    else:
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"      GPU {i}: {p.name}  {p.total_memory/1024**3:.0f} GB VRAM")
except ImportError:
    print("      torch install may have failed — check pip output above")
PYEOF

echo ""
echo "========================================================================"
echo "  Setup complete!"
echo ""
echo "  Smoke test (run first):"
echo "    cd ${GBV_DIR}"
echo "    python deploy/aip_run.py --config ${CONFIG}"
echo ""
echo "  One loss (after smoke passes):"
echo "    python deploy/aip_run.py --config ${CONFIG} --losses kl --no_smoke"
echo ""
echo "  Resume after restart:"
echo "    source ${ENVFILE}"
echo "    python deploy/aip_run.py --config ${CONFIG} --resume"
echo ""
echo "  Monitor logs:"
echo "    tail -f ${STORAGE}/logs/${CONFIG}-*/pipeline_output.log"
echo "========================================================================"
