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

# 2. dependencies
echo "[setup] pip install -r requirements.txt ..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

# 3. datasets (skip with --no-data)
if [ "${1:-}" != "--no-data" ]; then
    echo "[setup] downloading datasets (gsm8k train+val+eval, alpaca, math500, humaneval, mtbench) ..."
    python -m data_io.download --train
fi

# 4. W&B login reminder
if ! python -c "import wandb; assert wandb.api.api_key" 2>/dev/null; then
    echo "[setup] reminder: run 'wandb login' before training to enable W&B logging"
    echo "         (or pass --no_wandb to train.py to disable)"
fi

echo
echo "[setup] done.  Try:"
echo "  python train.py --loss kl_tree"
echo "  python eval.py  --checkpoint checkpoints/kl_tree/ckpt_best --mode gbv"
