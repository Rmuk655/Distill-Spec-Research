#!/usr/bin/env bash
# setup_vllm_env.sh — separate, isolated venv for vLLM-based pass@k eval.
#
# WHY A SEPARATE VENV: vllm==0.25.1 requires torch==2.11.0, but the main
# training venv (requirements.txt) pins torch==2.12.0 for kernel/dispatch
# perf comparability across runs. Installing vllm into that SAME venv
# silently downgrades torch and breaks the pin — this is exactly what
# happened once already (Python.h missing -> Triton JIT compile failed ->
# a parallel pip step then bumped torch 2.11.0->2.12.0 mid-session, leaving
# the repo's main venv in a broken, inconsistent state).
#
# Usage:
#   bash scripts/setup_vllm_env.sh
#   source venv-vllm/bin/activate
#   python scripts/passk_eval.py --model ... --dataset ...
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

if [ ! -d "venv-vllm" ]; then
    echo "[setup-vllm] creating venv-vllm ..."
    python3 -m venv venv-vllm
fi
# shellcheck disable=SC1091
source venv-vllm/bin/activate
echo "[setup-vllm] python: $(which python)"

# Python.h required for Triton's JIT-compiled CUDA utils (vLLM's default
# torch.compile path) -- same check/install pattern as setup_a100.sh's
# flash-attn step. Non-fatal: a failed apt install must not abort setup.
if ! python -c "import os,sysconfig; raise SystemExit(0 if os.path.exists(os.path.join(sysconfig.get_path('include'),'Python.h')) else 1)" 2>/dev/null; then
    echo "[setup-vllm] Python.h missing — installing python3.12-dev ..."
    sudo apt-get update -qq || true
    sudo apt-get install -y python3.12-dev || \
        echo "[setup-vllm] WARNING: could not install python3.12-dev — vLLM's torch.compile path will fail at runtime (Triton JIT compile error). Workaround: pass enforce_eager=True to LLM(...) in scripts/passk_eval.py to skip compilation entirely (slower, but no system package needed)."
fi

echo "[setup-vllm] pip install -r requirements-vllm.txt ..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements-vllm.txt

echo
echo "[setup-vllm] done. Activate with: source venv-vllm/bin/activate"
echo "  python scripts/passk_eval.py --model Qwen/Qwen3-0.6B --dataset data_io/raw/math_val.jsonl --n 4 --n_prompts 5"
