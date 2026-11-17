#!/usr/bin/env bash
# Setup for Ubuntu/WSL (CUDA GPU needed to train/eval): venv, pinned deps, datasets.
# Run:  bash setup.sh [--no-data]
# For an A100/H100 box with flash-attn built from source, models and the vLLM venv use the repo scripts instead:
#   bash scripts/setup_a100.sh        bash scripts/setup_vllm_env.sh
# Uses ./venv (not .venv) to match README.md and the repo's own scripts.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

sudo apt-get update
sudo apt-get install -y build-essential git python3 python3-venv python3-pip python3-dev

[ -d venv ] || python3 -m venv venv
venv/bin/pip install --upgrade pip
venv/bin/pip install -r requirements.txt   # torch/transformers are pinned on purpose, see header of requirements.txt

if [ "${1:-}" != "--no-data" ]; then
  # Documented in README.md: GSM8K train split + the 5 evaluation datasets (needs network).
  venv/bin/python -m data_io.download --train --n 1000
  # Optional datasets from README.md:
  #   venv/bin/python -m data_io.download --datasets math_hard,math_val,math_eval
  #   venv/bin/python -m data_io.download --datasets olympiad_eval
fi

echo "Done. Next: source venv/bin/activate; wandb login (or export WANDB_API_KEY, or pass --no-wandb)."
echo "Model weights (Qwen/Qwen3-0.6B, Qwen/Qwen3-8B) are fetched from Hugging Face on first use."
