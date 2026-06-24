#!/usr/bin/env bash
# setup_a100.sh — one-shot environment setup for A100 or H100 boxes.
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

# 4. flash-attn
#    Try precompiled wheel first (works when nvcc == torch's CUDA build).
#    If that fails (e.g. H100 with system CUDA 12.8 vs torch cu130), patch
#    torch's overly-strict version check in the venv, build from source, restore.
if python -c "import flash_attn" 2>/dev/null; then
    echo "[setup] flash-attn already installed — skipping"
else
    echo "[setup] installing flash-attn (precompiled wheel) ..."
    if pip install --quiet flash-attn --no-build-isolation 2>/dev/null; then
        python -c "import flash_attn; print('[setup] flash-attn', flash_attn.__version__, 'installed OK')"
    else
        echo "[setup] precompiled wheel unavailable — building from source (~15 min) ..."
        SYSTEM_CUDA=$(find /usr/local -maxdepth 2 -name "nvcc" 2>/dev/null | head -1 | xargs dirname | xargs dirname)
        if [ -z "${SYSTEM_CUDA}" ]; then
            echo "[setup] WARNING: no system nvcc found; skipping flash-attn (eval will use SDPA fallback)"
        else
            # Python dev headers required for C++ extension builds
            if ! dpkg -s python3.12-dev &>/dev/null 2>&1; then
                echo "[setup] installing python3.12-dev ..."
                sudo apt-get install -y python3.12-dev
            fi
            # Use Python for the patch+build+restore so the restore always runs
            # even if the build fails (set -e would skip it in bash).
            SYSTEM_CUDA_ARG="${SYSTEM_CUDA}" python - << 'PATCHEOF'
import os, re, shutil, subprocess, sys
import torch.utils.cpp_extension as m

path = m.__file__
bak  = path + '.bak'
cuda = os.environ['SYSTEM_CUDA_ARG']

shutil.copy(path, bak)
code = open(path).read()
patched = re.sub(
    r'raise RuntimeError\(CUDA_MISMATCH_MESSAGE,',
    r'return  # bypassed by setup_a100.sh: raise RuntimeError(CUDA_MISMATCH_MESSAGE,',
    code,
)
open(path, 'w').write(patched)
print("[setup] cpp_extension.py patched")

env = os.environ.copy()
env['CUDA_HOME']             = cuda
env['TORCH_CUDA_ARCH_LIST']  = '8.0;9.0'
env['MAX_JOBS']              = '4'
ret = subprocess.run(
    [sys.executable, '-m', 'pip', 'install', 'flash-attn', '--no-build-isolation'],
    env=env,
)

shutil.copy(bak, path)
os.remove(bak)
print("[setup] cpp_extension.py restored")

if ret.returncode != 0:
    print("[setup] ERROR: flash-attn build failed — eval will use SDPA fallback")
    sys.exit(0)  # non-fatal; don't abort the whole setup
PATCHEOF
            python -c "import flash_attn; print('[setup] flash-attn', flash_attn.__version__, 'installed OK')" 2>/dev/null || \
                echo "[setup] flash-attn not available — eval will use SDPA fallback"
        fi
    fi
fi

# 5. sitecustomize — auto-enable flash_attention_2 for all from_pretrained calls.
#    Newer transformers requires explicit attn_implementation="flash_attention_2";
#    this injects it transparently without touching researcher source files.
SITE_PKG=$(python -c "import site; print(site.getsitepackages()[0])")
SITECUST="${SITE_PKG}/sitecustomize.py"
if ! grep -qF "fa2_from_pretrained" "${SITECUST}" 2>/dev/null; then
    cat > "${SITECUST}" << 'SITEOF'
# Inject flash_attention_2 as default when flash_attn is installed.
# In transformers 4.51, get_correct_attn_implementation() hardcodes
# "sdpa" when requested_attention is None. Patch that one line's effect.
try:
    import flash_attn  # only activate when flash_attn is actually installed
    from transformers.modeling_utils import PreTrainedModel
    _orig = PreTrainedModel.get_correct_attn_implementation

    def _fa2_get_correct(self, requested_attention=None, is_init_check=False):
        if requested_attention is None:
            requested_attention = "flash_attention_2"
        return _orig(self, requested_attention, is_init_check=is_init_check)

    PreTrainedModel.get_correct_attn_implementation = _fa2_get_correct
except Exception:
    pass  # silent — never break Python startup
SITEOF
    echo "[setup] sitecustomize.py written — flash_attention_2 will be used automatically"
else
    echo "[setup] sitecustomize.py already present — skipping"
fi

# 6. datasets (skip with --no-data)
if [ "${1:-}" != "--no-data" ]; then
    echo "[setup] downloading datasets (gsm8k, math_hard, math_val, alpaca, math500, humaneval, mtbench) ..."
    python -m data_io.download --train
fi

# 7. W&B login reminder
if ! python -c "import wandb; assert wandb.api.api_key" 2>/dev/null; then
    echo "[setup] reminder: run 'wandb login' before training to enable W&B logging"
    echo "         (or pass --no_wandb to train.py to disable)"
fi

echo
echo "[setup] done.  Try:"
echo "  python train.py --loss kl_tree"
echo "  python eval.py  --checkpoint checkpoints/kl_tree/ckpt_best --mode gbv"
