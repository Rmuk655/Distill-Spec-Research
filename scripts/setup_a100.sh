#!/usr/bin/env bash
# setup_a100.sh — one-shot environment setup for A100 or H100 boxes.
#
# Run from anywhere; the script changes to the repo root and creates venv +
# installs deps + downloads datasets.  Idempotent — re-running just re-checks.
#
# Usage:
#   bash scripts/setup_a100.sh
#   bash scripts/setup_a100.sh --no-data           # skip dataset download
#   bash scripts/setup_a100.sh --no-models         # skip model weight download
#   bash scripts/setup_a100.sh --no-flash-attn     # skip flash-attn install entirely
#                                                     (SDPA fallback — correctness
#                                                     unaffected, just slower per step)
#   # flags combine, e.g.:
#   bash scripts/setup_a100.sh --no-flash-attn --no-data
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

SKIP_DATA=0
SKIP_MODELS=0
SKIP_FLASH=0
for _arg in "$@"; do
    case "${_arg}" in
        --no-data)        SKIP_DATA=1 ;;
        --no-models)       SKIP_MODELS=1 ;;
        --no-flash-attn)  SKIP_FLASH=1 ;;
        *) echo "[setup] WARNING: unrecognized flag '${_arg}' — ignoring" ;;
    esac
done

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

# 4. flash-attn (skip entirely with --no-flash-attn)
#    Try precompiled wheel first (works when nvcc == torch's CUDA build).
#    If that fails (e.g. H100 with system CUDA 12.8 vs torch cu130), patch
#    torch's overly-strict version check in the venv, build from source, restore.
if [ "${SKIP_FLASH}" = "1" ]; then
    echo "[setup] --no-flash-attn passed — skipping flash-attn (SDPA fallback, correctness unaffected)"
elif python -c "import flash_attn" 2>/dev/null; then
    echo "[setup] flash-attn already installed — skipping"
else
    echo "[setup] installing flash-attn (precompiled wheel) ..."
    if pip install --quiet flash-attn --no-build-isolation 2>/dev/null; then
        python -c "import flash_attn; print('[setup] flash-attn', flash_attn.__version__, 'installed OK')"
    else
        echo "[setup] precompiled wheel unavailable — building from source (~15 min) ..."
        # Locate nvcc: search /usr/local up to depth 4 (covers both
        # /usr/local/cuda-12.x/bin/nvcc and /usr/local/cuda/bin/nvcc symlink
        # layouts), then fall back to PATH.  Do NOT glob paths that may not
        # exist (/usr/cuda*) — find exits nonzero on missing paths and
        # set -euo pipefail would kill the script silently.
        _NVCC=$(find /usr/local -maxdepth 4 -name "nvcc" 2>/dev/null | head -1)
        [ -z "${_NVCC}" ] && _NVCC=$(command -v nvcc 2>/dev/null || true)
        if [ -n "${_NVCC}" ]; then
            SYSTEM_CUDA=$(dirname "$(dirname "${_NVCC}")")
        else
            SYSTEM_CUDA=""
        fi
        if [ -z "${SYSTEM_CUDA}" ]; then
            echo "[setup] WARNING: no system nvcc found; skipping flash-attn (eval will use SDPA fallback)"
        else
            # Python dev headers (Python.h) required for C++ extension builds.
            # Check for the header directly (more reliable than dpkg — the venv's
            # python may not come from the apt python3.12 package).  Make the apt
            # install NON-FATAL: a failed/absent package must not kill the whole
            # setup via set -euo pipefail.  apt-get update first fixes the common
            # "Unable to locate package" on containers with a stale package list.
            if ! python -c "import os,sysconfig; raise SystemExit(0 if os.path.exists(os.path.join(sysconfig.get_path('include'),'Python.h')) else 1)" 2>/dev/null; then
                echo "[setup] Python.h missing — installing python3.12-dev ..."
                sudo apt-get update -qq || true
                sudo apt-get install -y python3.12-dev || \
                    echo "[setup] WARNING: could not install python3.12-dev — flash-attn build may fail (eval will use SDPA fallback)"
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

# 5. force flash_attention_2 as the default — without touching researcher source.
#    Newer transformers (4.51) hardcodes "sdpa" in
#    PreTrainedModel.get_correct_attn_implementation() when none is requested.
#    We patch that via a .pth file (NOT sitecustomize.py — the system
#    /usr/lib/python3.12/sitecustomize.py shadows the venv one). Python's site
#    module executes `import` lines in every .pth at startup regardless of
#    sys.path ordering, so this is never shadowed.
SITE_PKG=$(python -c "import site; print(site.getsitepackages()[0])")
PATCH_MOD="${SITE_PKG}/fa2_default_patch.py"
PATCH_PTH="${SITE_PKG}/fa2_default_patch.pth"
# Always overwrite the .py so content changes in this script are picked up on
# subsequent setup runs.  The .pth only needs to be created once — its presence
# is what tells Python's site module to `import fa2_default_patch` at startup.
cat > "${PATCH_MOD}" << 'PATCHMOD'
# DistillSpec: default to flash_attention_2 for small (draft) models only.
# transformers 4.51 get_correct_attn_implementation() hardcodes "sdpa" when
# requested_attention is None; this patches that one line's effect.
#
# We cannot apply flash_attention_2 globally: the target (large) model is called
# with attention_mask={"full_attention": mask} — a custom dict for tree speculative
# decoding that flash_attention_2 cannot handle (triggers CUDA index OOB).  The
# draft (small) model uses standard causal attention and is safe for flash_attention_2.
#
# Heuristic: hidden_size < 3000 → draft (Qwen3-0.6B = 1024) → FA2
#            hidden_size ≥ 3000 → target (Qwen3-8B = 4096) → keep SDPA default
try:
    import flash_attn  # only activate when flash_attn is actually installed
    from transformers.modeling_utils import PreTrainedModel
    _orig = PreTrainedModel.get_correct_attn_implementation

    def _fa2_get_correct(self, requested_attention=None, *args, **kwargs):
        if requested_attention is None:
            cfg = getattr(self, 'config', None)
            hidden = getattr(cfg, 'hidden_size', 0)
            if 0 < hidden < 3000:
                requested_attention = "flash_attention_2"
        return _orig(self, requested_attention, *args, **kwargs)

    PreTrainedModel.get_correct_attn_implementation = _fa2_get_correct
except Exception:
    pass  # silent — never break Python startup
PATCHMOD
if [ ! -f "${PATCH_PTH}" ]; then
    echo "import fa2_default_patch" > "${PATCH_PTH}"
    echo "[setup] flash_attention_2 default patch installed (.pth) — verifying ..."
    python -c "
from transformers import AutoModelForCausalLM
import torch, sys
# light check: confirm the patch is live without loading weights
import fa2_default_patch  # noqa
from transformers.modeling_utils import PreTrainedModel
ok = PreTrainedModel.get_correct_attn_implementation.__name__ == '_fa2_get_correct'
print('[setup] flash_attention_2 patch active:', ok)
" 2>/dev/null || echo "[setup] (patch verification skipped)"
else
    echo "[setup] flash_attention_2 default patch already present (content refreshed)"
fi

# 6. datasets (skip with --no-data)
if [ "${SKIP_DATA}" != "1" ]; then
    echo "[setup] downloading datasets (gsm8k, math_hard, math_val, alpaca, math500, humaneval, mtbench) ..."
    python -m data_io.download --train
    # OlympiadBench — harder than math_hard, EVAL-ONLY (no val/train split —
    # never checkpoint-selected or trained on, only used with eval.py). Schema
    # confirmed on-cluster with a real HF token (see data_io/download.py).
    echo "[setup] downloading OlympiadBench (eval-only) ..."
    python -m data_io.download --datasets olympiad_eval
fi

# 6b. model weights — cache to the default HF cache (~/.cache/huggingface, local
#     box disk; NOT a shared network filesystem). Idempotent: hf skips files already present.
#     Both pairs so either the 0.6B/8B default or the 1.7B/32B capacity study
#     runs without a cold-start download. Skip with --no-models.
if [ "${SKIP_MODELS}" != "1" ]; then
    echo "[setup] caching Qwen3 model weights (default HF cache) ..."
    for _m in Qwen/Qwen3-0.6B Qwen/Qwen3-8B Qwen/Qwen3-1.7B Qwen/Qwen3-32B; do
        echo "[setup]   ${_m}"
        huggingface-cli download "${_m}" --exclude "*.pth" "*.gguf" "original/*" || \
            echo "[setup]   WARNING: download of ${_m} failed — retry manually"
    done
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
