#!/usr/bin/env bash
# =============================================================================
# 01_setup.sh — One-time setup for the ATS Cloud bare-metal server.
# Ubuntu 24.04 · 128 GB RAM · 2-socket CPU · no GPU
#
# Run once after provisioning:
#   source deploy/ats/00_env.sh
#   bash deploy/ats/01_setup.sh
#
# After this completes, run 02_verify.py to confirm everything is ready.
# =============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$ROOT"

echo "====================================================="
echo "  ATS Cloud — One-Time Setup"
echo "  Root: $ROOT"
echo "====================================================="
echo

# ── 1. Python deps ────────────────────────────────────────────────────────────
echo "[1/5] Installing Python dependencies..."
pip install -r requirements.txt -q
echo "      OK"

# ── 2. PyTorch threads — use all cores for CPU matrix ops ────────────────────
NCORES=$(nproc)
echo "[2/5] CPU cores detected: $NCORES"
echo "      Setting OMP_NUM_THREADS=$NCORES for maximum throughput."
export OMP_NUM_THREADS=$NCORES
# Make persistent
grep -qxF "export OMP_NUM_THREADS=$NCORES" ~/.bashrc 2>/dev/null || \
    echo "export OMP_NUM_THREADS=$NCORES" >> ~/.bashrc
echo "      OK"

# ── 3. Datasets ───────────────────────────────────────────────────────────────
echo "[3/5] Downloading GSM8K training dataset..."
python core/datasets/downloader.py
echo "      OK"

# ── 4. GPT-2 model weights ────────────────────────────────────────────────────
echo "[4/5] Downloading GPT-2 weights (distilgpt2 + gpt2-medium, ~1.7 GB)..."
python - <<'PY'
import os
for k in ("TRANSFORMERS_OFFLINE","HF_HUB_OFFLINE","HF_DATASETS_OFFLINE"):
    os.environ.pop(k, None)
from transformers import AutoModelForCausalLM, AutoTokenizer
for mid in ["distilgpt2", "gpt2-medium"]:
    print(f"  {mid}...", flush=True)
    AutoTokenizer.from_pretrained(mid)
    AutoModelForCausalLM.from_pretrained(mid)
print("  Done.")
PY
echo "      OK"

# ── 5. W&B login ─────────────────────────────────────────────────────────────
echo "[5/5] W&B authentication..."
if [ -n "$WANDB_API_KEY" ] && [ "$WANDB_API_KEY" != "PASTE_YOUR_WANDB_KEY_HERE" ]; then
    python -c "import wandb; wandb.login(key='$WANDB_API_KEY', relogin=True)" \
        && echo "      Logged in to W&B." \
        || echo "      WARNING: W&B login failed — check WANDB_API_KEY."
else
    echo "      SKIPPED (WANDB_API_KEY not set — runs will log offline)."
    echo "      Set WANDB_API_KEY in deploy/ats/00_env.sh and re-source."
fi

echo
echo "====================================================="
echo "  Setup complete."
echo "  Next: python deploy/ats/02_verify.py"
echo "====================================================="
