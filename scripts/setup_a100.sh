#!/usr/bin/env bash
# setup_a100.sh — one-shot environment setup for an A100 box.
#
# Run from anywhere; the script changes to the repo root and creates venv +
# installs deps + downloads datasets.  Idempotent — re-running just re-checks.
#
# Usage:
#   bash scripts/setup_a100.sh
#   bash scripts/setup_a100.sh --no-data        # skip dataset download
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

echo "[setup] repo: ${REPO_DIR}"

# 1. venv
if [ ! -d "venv" ]; then
    echo "[setup] creating venv ..."
    python3 -m venv venv
fi
# shellcheck disable=SC1091
source venv/bin/activate
echo "[setup] python: $(which python)"

# 2. CUDA stability vars — injected into venv/bin/activate so they are set on
#    every subsequent `source venv/bin/activate` without a separate step.
ACTIVATE="venv/bin/activate"
CUDA_BLOCK="# DistillSpec CUDA env vars (injected by setup_a100.sh)"
if ! grep -qF "${CUDA_BLOCK}" "${ACTIVATE}"; then
    cat >> "${ACTIVATE}" <<'EOF'

# DistillSpec CUDA env vars (injected by setup_a100.sh)
export TORCH_CUDNN_SDPA_ENABLED=0          # avoid cuDNN MHA graph failures on long seqs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # reduce fragmentation OOM
EOF
    echo "[setup] CUDA env vars written to ${ACTIVATE}"
else
    echo "[setup] CUDA env vars already present in ${ACTIVATE} — skipping"
fi

# 3. dependencies
echo "[setup] pip install -r requirements.txt ..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

# 4. datasets (skip with --no-data)
if [ "${1:-}" != "--no-data" ]; then
    echo "[setup] downloading datasets (gsm8k, math_hard, math_val, alpaca, math500, humaneval, mtbench) ..."
    python -m data_io.download --train
fi

# 5. W&B login reminder
if ! python -c "import wandb; assert wandb.api.api_key" 2>/dev/null; then
    echo "[setup] reminder: run 'wandb login' before training to enable W&B logging"
    echo "         (or pass --no_wandb to train.py to disable)"
fi

echo
echo "[setup] done.  Try:"
echo "  python train.py --loss kl_tree"
echo "  python eval.py  --checkpoint checkpoints/kl_tree/ckpt_best --mode gbv"
