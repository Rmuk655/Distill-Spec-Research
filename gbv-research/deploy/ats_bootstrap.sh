#!/usr/bin/env bash
# =============================================================================
# ATS Cloud Bootstrap — Ubuntu 24.04 bare-metal (128 GB RAM, 2-socket CPU)
# Run ONCE on a fresh ATS server to set up the environment.
#
# Usage:
#   bash deploy/ats_bootstrap.sh
#
# What it does:
#   1. Install Python deps (pip install -r requirements.txt)
#   2. Download GSM8K training dataset
#   3. Download GPT-2 model weights (distilgpt2 + gpt2-medium, ~1.7 GB total)
#   4. Verify everything is ready for training
#
# After this runs once, use:
#   python deploy/ats_train.py          — train all losses in parallel
#   python deploy/ats_status.py         — check progress any time
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=============================="
echo "  ATS Cloud Bootstrap"
echo "  Repo: $REPO_ROOT"
echo "=============================="
echo

# ── 1. Python deps ─────────────────────────────────────────────────────────
echo "[1/4] Installing Python dependencies..."
cd "$REPO_ROOT"
pip install -r requirements.txt --quiet
echo "      Done."
echo

# ── 2. Download datasets ───────────────────────────────────────────────────
echo "[2/4] Downloading GSM8K training dataset..."
python core/datasets/downloader.py
echo "      Done."
echo

# ── 3. Download GPT-2 model weights ───────────────────────────────────────
echo "[3/4] Downloading GPT-2 model weights (distilgpt2 + gpt2-medium, ~1.7 GB)..."
python - <<'PYEOF'
import os
# Ensure HF is online for download
for k in ("TRANSFORMERS_OFFLINE", "HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE"):
    os.environ.pop(k, None)

from transformers import AutoModelForCausalLM, AutoTokenizer

for model_id in ["distilgpt2", "gpt2-medium"]:
    print(f"  Downloading {model_id}...")
    AutoTokenizer.from_pretrained(model_id)
    AutoModelForCausalLM.from_pretrained(model_id)
    print(f"  {model_id} cached.")
print("  All models ready.")
PYEOF
echo

# ── 4. Verify ──────────────────────────────────────────────────────────────
echo "[4/4] Verifying setup..."
python - <<'PYEOF'
import sys, os
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__ if "__file__" in dir() else ".")))
sys.path.insert(0, root)

from core.model_families import get_family
gpt2 = get_family("gpt2")
print(f"  Family: {gpt2.name}")
print(f"  Draft:  {gpt2.default_draft_model_id}")
print(f"  Target: {gpt2.default_target_model_id}")
print(f"  LoRA:   {gpt2.lora_target_modules()}")

import torch
cpus = torch.get_num_threads()
print(f"  CPU threads available: {cpus}")
print("  Setup OK — ready to train.")
PYEOF

echo
echo "=============================="
echo "  Bootstrap complete!"
echo
echo "  Next step — start parallel training:"
echo "    python deploy/ats_train.py"
echo
echo "  Monitor progress (from another SSH session):"
echo "    python deploy/ats_status.py"
echo "=============================="
