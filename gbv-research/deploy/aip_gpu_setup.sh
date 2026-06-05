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

# =============================================================================
# EVERYTHING on Sensei FS — truly one-time-ever setup.
#
# After this script runs once, all state (repo, venv, models, checkpoints, DB)
# lives in /sensei-fs/users/rkrishna/. On a new session you only need:
#   source ~/.specdist_env
#
# Sensei FS (/sensei-fs/users/rkrishna) — 500 GB quota:
#   Distill-Spec-Research/   → git repo  (~200 MB + history)
#   specdist/venv/           → Python venv (~8 GB, torch + transformers)
#   specdist/hf_cache/       → Qwen3-8B + 0.6B (~18 GB, cached forever)
#   specdist/checkpoints/    → trained LoRA adapters (~130 GB over full run)
#   specdist/results.db      → eval database (~50 MB)
#   specdist/logs/           → pipeline logs (~5 GB)
#   ─────────────────────────────────────────────────────────────────────
#   Total:  ~161 GB / 500 GB  ✅
#
# NFS venv: Python import on first subprocess call is ~5-10s from NFS vs ~1s
# local. Over 17 training runs × 20-30 min each, this is <1% overhead — fine.
# =============================================================================

SENSEI_BASE="/sensei-fs/users/rkrishna"
SENSEI_REPO="${SENSEI_BASE}/Distill-Spec-Research"
SENSEI_SPECDIST="${SENSEI_BASE}/specdist"
REPO_URL="https://github.com/Rmuk655/Distill-Spec-Research.git"

# ── Determine which repo directory we're running from ─────────────────────────
# If already running from Sensei repo, use it. Otherwise clone to Sensei first.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GBV_DIR="$(dirname "$SCRIPT_DIR")"
REPO_DIR="$(dirname "$GBV_DIR")"

# ── Use Sensei FS if available, fall back to $HOME ────────────────────────────
if [ -d "${SENSEI_BASE}" ] && [ -w "${SENSEI_BASE}" ]; then
    STORAGE="${SENSEI_SPECDIST}"
    USE_SENSEI=1
elif [ -n "${STORAGE_ROOT:-}" ]; then
    STORAGE="${STORAGE_ROOT}"
    USE_SENSEI=0
elif [ -w "$HOME" ]; then
    STORAGE="$HOME/specdist"
    USE_SENSEI=0
    echo "  [storage] Sensei FS not available — using \$HOME/specdist (not persistent across machines)"
else
    STORAGE="${GBV_DIR}/db"
    USE_SENSEI=0
fi

HF_CACHE="${STORAGE}/hf_cache"
VENV_DIR="${STORAGE}/venv"

echo "========================================================================"
echo "  SpecDist GPU Setup  (truly one-time-ever on Sensei FS)"
echo "  Config  : ${CONFIG}"
echo "  Storage : ${STORAGE}"
echo "  Venv    : ${VENV_DIR}"
echo "  HF cache: ${HF_CACHE}"
echo "========================================================================"
echo ""

mkdir -p "${STORAGE}" "${HF_CACHE}"

# ── [0] Clone repo to Sensei if not already there ─────────────────────────────
if [ "${USE_SENSEI:-0}" = "1" ] && [ ! -d "${SENSEI_REPO}/.git" ]; then
    echo "[0/5] Cloning repo to Sensei FS (one-time-ever)..."
    git clone "${REPO_URL}" "${SENSEI_REPO}"
    GBV_DIR="${SENSEI_REPO}/gbv-research"
    REPO_DIR="${SENSEI_REPO}"
    echo "      Repo now at: ${SENSEI_REPO}"
    echo "      Future sessions: cd ${SENSEI_REPO}/gbv-research"
elif [ "${USE_SENSEI:-0}" = "1" ]; then
    echo "[0/5] Repo already at Sensei: ${SENSEI_REPO}"
    # Update REPO_DIR to Sensei path so remaining steps use Sensei repo
    GBV_DIR="${SENSEI_REPO}/gbv-research"
    REPO_DIR="${SENSEI_REPO}"
else
    echo "[0/5] Not on Sensei — using repo at: ${REPO_DIR}"
fi

# ── Pull latest code ──────────────────────────────────────────────────────────
# Inject GITHUB_TOKEN only into a clean HTTPS URL.  If origin already contains
# credentials (e.g. from a prior setup run), blind sed 's|https://|https://TOKEN@|'
# produces https://TOKEN@TOKEN@github.com/... → "URL rejected: Bad hostname".
GH_TOKEN="${GITHUB_TOKEN:-}"
if [ -n "${GH_TOKEN}" ]; then
    REPO_URL=$(git -C "${REPO_DIR}" remote get-url origin 2>/dev/null || echo "")
    case "${REPO_URL}" in
        git@*|ssh://*)
            # SSH remotes: do not rewrite with HTTPS token
            ;;
        https://*|http://*)
            CLEAN_URL=$(echo "${REPO_URL}" | sed -E 's#https?://[^/@]+@#https://#')
            AUTH_URL=$(echo "${CLEAN_URL}" | sed "s|https://|https://${GH_TOKEN}@|")
            git -C "${REPO_DIR}" remote set-url origin "${AUTH_URL}" 2>/dev/null || true
            ;;
    esac
fi

echo "[1/5] Pulling latest code..."
if ! git -C "${REPO_DIR}" pull --ff-only 2>/dev/null; then
    echo "      git pull failed — if you see 'Bad hostname', reset origin:"
    echo "        git -C ${REPO_DIR} remote set-url origin https://github.com/Rmuk655/Distill-Spec-Research.git"
    echo "      Then pull again (public repo) or set GITHUB_TOKEN once and re-run this script."
fi

# Clone OSD (public repo) alongside Distill-Spec-Research if not already present.
# evaluate.py looks for specInfer at:  <parent_of_repo>/OSD/distill/specInfer/
# i.e. one level up from the cloned repo, at ~/ram/OSD/ when repo is at ~/ram/Distill-Spec-Research/
OSD_DIR="$(dirname "${REPO_DIR}")/OSD"
if [ ! -d "${OSD_DIR}/.git" ]; then
    echo "      Cloning OSD (specInfer Generator, public repo)..."
    git clone https://github.com/LiuXiaoxuanPKU/OSD "${OSD_DIR}" 2>/dev/null && \
        echo "      OSD cloned to ${OSD_DIR}" || \
        echo "      OSD clone failed — alpha eval will use inline fallback (correct values)"
else
    echo "      OSD already present at ${OSD_DIR}"
fi

# ── Python environment ────────────────────────────────────────────────────────
VENV_DIR="${VENV_HOME}"
if [ ! -d "${VENV_DIR}" ]; then
    echo "[2/5] Creating virtual environment at ${VENV_DIR}..."
    python3 -m venv "${VENV_DIR}"
else
    echo "[2/5] Using existing venv at ${VENV_DIR}"
fi
source "${VENV_DIR}/bin/activate"
echo "      Python: $(python --version)  ($(which python))"

# Install Python dependencies ─────────────────────────────────────────────────
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

# ── Training data (download HERE — before offline flags are set in step 3) ────
# This must come before sourcing ~/.specdist_env which sets HF_DATASETS_OFFLINE=1.
# If training data is downloaded after offline flags are set, the HF datasets
# library refuses to make network calls and the download fails.
DATA_FILE="${GBV_DIR}/core/datasets/raw/gsm8k_train.jsonl"
if [ ! -f "${DATA_FILE}" ]; then
    echo "      Downloading gsm8k_train.jsonl (7473 training prompts, ~3 MB)..."
    python - <<PYEOF
import urllib.request, json, os, sys
GBV = os.environ.get("GBV_DIR", ".")
path = os.path.join(GBV, "core/datasets/raw/gsm8k_train.jsonl")
os.makedirs(os.path.dirname(path), exist_ok=True)

# Download directly from GitHub — no HF library, no auth, no offline-mode conflicts.
# load_dataset('gsm8k') is broken in newer huggingface-hub (needs namespace 'openai/gsm8k').
# GitHub raw source is stable and always accessible.
url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl"
try:
    prompts = []
    with urllib.request.urlopen(url, timeout=60) as r:
        for line in r:
            item = json.loads(line)
            prompts.append(json.dumps({"prompt": item["question"]}) + "\n")
    with open(path, "w") as f:
        f.writelines(prompts)
    print(f"      gsm8k_train.jsonl: {len(prompts)} prompts saved (from GitHub)")
except Exception as e:
    # Fallback: try HF datasets library with explicit namespace
    try:
        from datasets import load_dataset
        for v in ("HF_DATASETS_OFFLINE","HF_HUB_OFFLINE","TRANSFORMERS_OFFLINE"):
            os.environ.pop(v, None)
        ds = load_dataset("openai/gsm8k", "main", split="train")
        with open(path, "w") as f:
            for item in ds:
                f.write(json.dumps({"prompt": item["question"]}) + "\n")
        print(f"      gsm8k_train.jsonl: {len(ds)} prompts saved (from HF)")
    except Exception as e2:
        print(f"      WARNING: download failed: {e} / {e2}")
        print("      Fix: python -c \"import urllib.request,json,os; [open('core/datasets/raw/gsm8k_train.jsonl','a').write(json.dumps({'prompt':json.loads(l)['question']})+chr(10)) for l in urllib.request.urlopen('https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl')]\"")
PYEOF
else
    echo "      gsm8k_train.jsonl: $(wc -l < "${DATA_FILE}") prompts (already present)"
fi

# ── Environment variables ─────────────────────────────────────────────────────
echo "[3/5] Setting environment..."

export HF_HOME="${HF_CACHE}"
export TRANSFORMERS_CACHE="${HF_CACHE}"
# Allow downloads on this first run (models may not be cached yet).
# After setup completes, offline mode is set in the persistent env file so
# subsequent sessions skip the HF Hub etag check and the warning:
#   "Warning: You are sending unauthenticated requests to the HF Hub"
unset TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE HF_HUB_OFFLINE 2>/dev/null || true

# Persist env across new terminals — includes venv activation and offline mode.
# Offline mode: models are cached in HF_HOME after first run; all future loads
# read from local cache with no network calls → no HF Hub warnings.
ENVFILE="${HOME}/.specdist_env"
cat > "${ENVFILE}" <<EOF
source "${VENV_DIR}/bin/activate"
export HF_HOME="${HF_CACHE}"
export TRANSFORMERS_CACHE="${HF_CACHE}"
export STORAGE_ROOT="${STORAGE}"
export GBV_DIR="${GBV_DIR}"
# Offline mode: models already cached — skip HF Hub network checks.
# Unset these if you need to download updated models.
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
# Avoid CUDA OOM from transformers caching_allocator_warmup (>= 4.46).
# That function pre-allocates ~half the model size in FP16 during load.
# expandable_segments allows the allocator to use multiple non-contiguous
# blocks instead of one giant contiguous allocation — avoids OOM spikes.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
EOF

# Persist WANDB_API_KEY so trainer.py / evaluate.py subprocesses can auth.
# wandb.login() writes to ~/.netrc but subprocess environments don't reliably
# read from there.  The env var is the only reliable cross-subprocess auth.
WANDB_KEY="${WANDB_API_KEY:-}"
if [ -n "${WANDB_KEY}" ]; then
    echo "export WANDB_API_KEY='${WANDB_KEY}'" >> "${ENVFILE}"
    echo "      WANDB_API_KEY added to ${ENVFILE} (persists across sessions and subprocesses)"
else
    echo "      WANDB_API_KEY not set — W&B training runs will show [wandb] None"
    echo "      Fix: export WANDB_API_KEY=your-key && bash aip_gpu_setup.sh"
fi
# Auto-source in every new shell by adding to ~/.bashrc (idempotent — only adds once).
BASHRC="${HOME}/.bashrc"
MARKER="# specdist env"
if ! grep -q "${MARKER}" "${BASHRC}" 2>/dev/null; then
    echo "" >> "${BASHRC}"
    echo "${MARKER}" >> "${BASHRC}"
    echo "[ -f '${ENVFILE}' ] && source '${ENVFILE}'" >> "${BASHRC}"
    echo "      Added to ${BASHRC} — env loads automatically in new shells"
else
    echo "      Already in ${BASHRC} — no change needed"
fi
# Source now so current session picks it up without needing a new terminal.
source "${ENVFILE}"
echo "      Env sourced in current shell (TRANSFORMERS_OFFLINE=1, venv active)"

# W&B auth — login so wandb CLI works interactively (env var handles subprocess auth above)
WANDB_KEY="${WANDB_API_KEY:-}"
if [ -n "${WANDB_KEY}" ]; then
    python -c "import wandb; wandb.login(key='${WANDB_KEY}', relogin=True)" 2>/dev/null && \
        echo "      W&B: authenticated (interactive + subprocess auth via env var)"  || \
        echo "      W&B: login failed"
else
    echo "      W&B: WANDB_API_KEY not set — training will log [wandb] None"
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
    echo "      Downloading gsm8k_train.jsonl (7473 prompts, ~3 MB)..."
    # Temporarily disable offline mode for this download — the env file already
    # set HF_DATASETS_OFFLINE=1 (from step 3 source), but we need online access
    # to download the training data for the first time.
    env -u HF_DATASETS_OFFLINE -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE \
    python - <<'PYEOF'
import sys, os, json
sys.path.insert(0, os.environ.get("GBV_DIR", "."))
try:
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="train")
    path = os.path.join(os.environ.get("GBV_DIR", "."),
                        "core/datasets/raw/gsm8k_train.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for item in ds:
            f.write(json.dumps({"prompt": item["question"]}) + "\n")
    print(f"      gsm8k_train.jsonl: {len(ds)} prompts saved")
except Exception as e:
    print(f"      WARNING: could not download gsm8k_train.jsonl: {e}")
    print("      Training will fail without this file.")
    print("      Run manually with offline mode disabled:")
    print("        unset HF_DATASETS_OFFLINE HF_HUB_OFFLINE")
    print("        python -c \"from datasets import load_dataset; import json")
    print("        ds=load_dataset('gsm8k','main',split='train')")
    print("        [open('core/datasets/raw/gsm8k_train.jsonl','w').write(json.dumps({'prompt':i['question']})+chr(10)) for i in ds]\"")
PYEOF
    # Also fetch other eval sets (already offline-safe since they use custom downloader)
    env -u HF_DATASETS_OFFLINE -u HF_HUB_OFFLINE \
    python "${GBV_DIR}/core/datasets/downloader.py" 2>/dev/null || true
else
    LINES=$(wc -l < "${DATA_FILE}")
    echo "      gsm8k_train.jsonl: ${LINES} prompts (already present)"
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
