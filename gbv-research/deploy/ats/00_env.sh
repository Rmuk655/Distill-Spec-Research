#!/usr/bin/env bash
# =============================================================================
# 00_env.sh — Configure environment variables for the ATS server.
# Source this file (don't run it) in every SSH session:
#   source deploy/ats/00_env.sh
#
# Or add to ~/.bashrc for persistence:
#   echo "source ~/Distill-Spec-Research/gbv-research/deploy/ats/00_env.sh" >> ~/.bashrc
# =============================================================================

# ── Edit these ────────────────────────────────────────────────────────────────
export WANDB_API_KEY="PASTE_YOUR_WANDB_KEY_HERE"   # https://wandb.ai/authorize
export WANDB_ENTITY="rkrishnaiyer"                  # your W&B username
export WANDB_PROJECT="distillspec"
# export WANDB_BASE_URL="https://adobesensei.wandb.io"  # uncomment for Adobe W&B

# ── No GPU — tell PyTorch explicitly so all fallbacks use CPU cleanly ─────────
export CUDA_VISIBLE_DEVICES=""

# ── HF offline disabled — allow downloads (models cached after first run) ─────
unset TRANSFORMERS_OFFLINE
unset HF_HUB_OFFLINE
unset HF_DATASETS_OFFLINE

# ── W&B mode: online when key is set, offline otherwise ──────────────────────
if [ -z "$WANDB_API_KEY" ] || [ "$WANDB_API_KEY" = "PASTE_YOUR_WANDB_KEY_HERE" ]; then
    export WANDB_MODE="offline"
    echo "[env] W&B: OFFLINE mode (set WANDB_API_KEY to enable sync)"
else
    unset WANDB_MODE
    echo "[env] W&B: ONLINE — project=${WANDB_PROJECT} entity=${WANDB_ENTITY}"
fi

echo "[env] CUDA_VISIBLE_DEVICES='' (CPU-only server)"
echo "[env] ATS environment ready."
