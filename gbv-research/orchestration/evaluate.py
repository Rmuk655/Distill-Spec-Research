"""
evaluate.py — single entry-point for the full SpecDist evaluation pipeline.

Verification algorithm reference (Thomas et al., 2026 — arXiv:2602.16994v1):
    "Dynamic Delayed Tree Expansion for Improved Multi-Path Speculative Decoding"

    Empirical BE ranking (averaged across Qwen/Gemma/Llama, 5 datasets, 8 sampling configs):
        traversal (5.31) > spectr (4.61) ≈ specinfer (4.58) > bv (4.30) > nss (4.05)

    Key insight: Traversal's bottom-up acceptance outperforms all OT-based methods (+15% BE)
    because OT-based methods waste tree budget at shallow nodes (where p≈q and gains are small).
    GBV improves on single-path BV by greedy multi-path selection, achieving BE similar to SpecInfer
    with lower compute overhead.

    Our training objective (block-level EBE) directly targets E[τ+1] = 1 + Σ_{k} Π_{i≤k} α_i,
    the same quantity measured in Table 2 of Thomas et al. (2026).

Takes two required arguments:
    --student   path or HuggingFace ID of the draft (student) model
    --teacher   path or HuggingFace ID of the target (teacher) model

Everything else uses sensible defaults and can be overridden.

Usage:
    # Laptop (0.6B models)
    python evaluate.py \\
        --student Qwen/Qwen2.5-0.5B \\
        --teacher Qwen/Qwen3-0.6B

    # Server (8B target)
    python evaluate.py \\
        --student Qwen/Qwen2.5-0.5B \\
        --teacher Qwen/Qwen3-8B \\
        --n 50

    # Quick smoke test
    python evaluate.py \\
        --student Qwen/Qwen2.5-0.5B \\
        --teacher Qwen/Qwen3-0.6B \\
        --datasets diverse50 --modes alpha --K 1 --n 10

    # Laptop full run (all datasets, all modes, ~4-6 hours)
    python evaluate.py \\
        --student Qwen/Qwen2.5-0.5B \\
        --teacher Qwen/Qwen3-0.6B \\
        --full

Outputs:
    results.db          SQLite with all results (run_tag, metrics, per-prompt)
    DELIVERABLES.md     Updated with new results table
"""

import sys, os, json, time, subprocess, re, argparse, threading
from datetime import datetime

# Windows terminals default to cp1252 which can't encode Unicode box-drawing or
# arrow characters.  Force UTF-8 output with replacement so any stray non-ASCII
# char in a print() never crashes the whole evaluation run.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass  # Python < 3.7 or non-text stream (e.g. redirected to a file)

# Offline mode — prevents HF Hub network calls on cached / air-gapped setups.
# Default: ON on local machines (models already downloaded),
#          OFF automatically in cloud envs (Colab/Modal/Spaces) where the HF
#          cache is empty on every fresh session.
# Manual override: set TRANSFORMERS_OFFLINE=0 or =1 before running.

def _is_cloud_env() -> bool:
    """Return True when running inside Colab / Modal / HF Spaces / Kaggle."""
    _cloud_keys = (
        "COLAB_BACKEND_VERSION",   # Google Colab
        "COLAB_RELEASE_TAG",       # Google Colab (alt key)
        "MODAL_TASK_ID",           # Modal
        "SPACE_ID",                # HuggingFace Spaces
        "KAGGLE_KERNEL_RUN_TYPE",  # Kaggle
    )
    return any(k in os.environ for k in _cloud_keys)

if "TRANSFORMERS_OFFLINE" not in os.environ:
    if _is_cloud_env():
        os.environ["TRANSFORMERS_OFFLINE"] = "0"
        os.environ["HF_HUB_OFFLINE"] = "0"
        print(
            "[OSD] Cloud env detected — HF online mode ON. "
            "Models will be downloaded from HuggingFace Hub on first run."
        )
    else:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        print(
            "[OSD] HF offline mode ON (default). "
            "Models must already be cached locally. "
            "Set TRANSFORMERS_OFFLINE=0 before running to allow first-time downloads."
        )

# Reduce CUDA allocator fragmentation on small GPUs (T4, P100).
# Set automatically; override with PYTORCH_CUDA_ALLOC_CONF=<custom> in environment.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)  # gbv-research/
sys.path.insert(0, _HERE)
# specInfer package resolution — two locations checked in priority order:
#
# 1. algorithms/specinfer/ (bundled copy inside gbv-research) — PRIMARY.
#    These 6 files (generator.py, common.py, proposer.py, verifier.py, logger.py,
#    __init__.py) are committed to the repo so specInfer works regardless of whether
#    the OSD sibling repo is cloned.  No external dependency, no submodule needed.
#    Copied from LiuXiaoxuanPKU/OSD at a known-good commit; update manually if needed.
#
# 2. OSD/distill/ (sibling repo fallback) — kept for backward compat.
#    OSD is the original Online Speculative Decoding codebase (Liu et al. 2023).
#    Populated by: git clone https://github.com/LiuXiaoxuanPKU/OSD ~/ram/OSD
_BUNDLED_SPECINFER = os.path.join(_PARENT, "algorithms")  # contains specinfer/ package
_OSD_DIR = os.path.join(os.path.dirname(_PARENT), "OSD")
sys.path.insert(0, os.path.join(_OSD_DIR, "distill"))     # fallback: sibling OSD repo
sys.path.insert(0, _BUNDLED_SPECINFER)                    # primary: bundled copy wins
# gbv-research/db/ MUST come LAST (position 0 wins) so `import results_db` resolves
# to gbv-research/db/results_db.py — not OSD/results_db.py — and writes to db/results.db
# which is where the dashboard reads from.
sys.path.insert(0, os.path.join(_PARENT, "db"))

# Module-level flag: specInfer fallback warning is shown at most once per process.
_SPECINFER_FALLBACK_WARNED: bool = False


def _now_tag() -> str:
    """Return a compact UTC timestamp string, e.g. '20260601_1041'."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")

# ---------------------------------------------------------------------------
# Transformers 4.55.x dtype-serialization bug-fix (applied once at import time)
#
# Root cause: Qwen configs store "dtype": "bfloat16" in config.json.
# transformers 4.55.x converts that string to torch.bfloat16 (a torch.dtype
# object) in from_dict(), then calls logger.info(f"Model config {config}").
# The f-string evaluates config.__repr__() → to_json_string() → json.dumps(),
# which crashes: "Object of type dtype is not JSON serializable".
#
# Fix: monkey-patch PretrainedConfig.to_json_string to convert torch.dtype
# objects to strings before serialization.  The patch is idempotent and
# applies globally — covers compute_perplexity(), run_alpha(), run_specinfer(),
# and every other from_pretrained() call in this file.
# ---------------------------------------------------------------------------
def _patch_config_json_serialization():
    try:
        import json, torch
        from transformers import PretrainedConfig
        _orig = PretrainedConfig.to_json_string
        if getattr(_orig, "_dtype_patch_applied", False):
            return  # already patched (e.g. if run_all is imported twice)
        def _safe(self, use_diff=True):
            try:
                return _orig(self, use_diff=use_diff)
            except TypeError:
                # Fallback: stringify any torch.dtype values before json.dumps
                d = self.to_diff_dict() if use_diff else self.to_dict()
                def _fix(o):
                    if isinstance(o, dict):
                        return {k: _fix(v) for k, v in o.items()}
                    if isinstance(o, torch.dtype):
                        return str(o).replace("torch.", "")
                    return o
                return json.dumps(_fix(d), indent=2, sort_keys=True) + "\n"
        _safe._dtype_patch_applied = True
        PretrainedConfig.to_json_string = _safe
    except Exception:
        pass   # transformers not installed yet — patch will be applied lazily

_patch_config_json_serialization()


# ── Suppress noisy HuggingFace logging ───────────────────────────────────────
# "Loading weights: X%" progress bars and advisory warnings clutter log files.
# These appear because evaluate.py logs to a file (non-TTY) and HF doesn't
# check isatty() before showing them.
#
# We suppress here rather than in each from_pretrained() call because the
# transformers logging module is global — one call covers all subsequent loads.
try:
    import transformers as _hf
    _hf.logging.set_verbosity_error()        # suppress INFO/WARNING messages
    _hf.logging.disable_progress_bar()       # suppress "Loading weights: X%"
except Exception:
    pass   # transformers not yet installed (e.g. during import-time dry-run)

# Suppress "unauthenticated requests to HF Hub" advisory — we use local cache.
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")


def _dtype_kwargs(dtype) -> dict:
    """Return the correct dtype kwarg for AutoModelForCausalLM.from_pretrained().

    transformers >= 4.51 (required for Qwen3) supports dtype= and deprecates
    torch_dtype=.  Always use dtype= — no version check needed.
    """
    return {"dtype": dtype}


import results_db
# Dataset utilities live in gbv-research/core/datasets/downloader.py — same API as
# the original OSD/fetch_datasets.py but saves to core/datasets/raw/ by default.
sys.path.insert(0, os.path.join(_PARENT, "core", "datasets"))
import downloader as _fd
_fd.DATA_DIR = os.path.join(_PARENT, "core", "datasets", "raw")
os.makedirs(_fd.DATA_DIR, exist_ok=True)
from downloader import fetch_all, get_dataset_path, ALL_DATASETS

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_DATASETS = ["diverse50", "gsm8k", "humaneval", "math500", "mtbench", "alpaca"]
DEFAULT_MODES    = ["alpha", "specinfer", "gbv", "traversal"]
DEFAULT_K        = [1, 3, 5]
DEFAULT_TEMPS    = [0.6, 1.0]   # robustness sweep: two temperatures per run
DEFAULT_N        = 50

# Verifier script: runner.py inside gbv-research/ (Phase 3 confirmed correct).
_VERIFIER_SCRIPT = os.path.join(
    _PARENT, "algorithms", "distillspec_gbv", "verifiers", "runner.py"
)


# ---------------------------------------------------------------------------
# GPU diagnostics helper
# ---------------------------------------------------------------------------

def _gpu_info() -> str:
    """One-line GPU status string, or 'CPU-only'."""
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(0)
            return (f"{torch.cuda.get_device_name(0)}  "
                    f"{free/1024**3:.1f} GB free / {total/1024**3:.1f} GB total")
    except Exception:
        pass
    return "CPU-only"


# Forced device override, set from --device in main().  None = auto-detect.
# 'cpu' forces CPU even on a GPU machine so a laptop can smoke-test the exact
# CPU code path that runs on the CPU-only ATS/AIP server.
_FORCED_DEVICE: str | None = None


def _pick_device():
    """Return (device, dtype), honouring the --device override, warn if VRAM tight."""
    import torch
    if _FORCED_DEVICE == "cpu":
        return "cpu", torch.float32
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(0)
        free_gb = free / 1024**3
        if free_gb < 1.5:
            print(f"  [WARN] Only {free_gb:.1f} GB VRAM free — OOM likely. "
                  f"Will retry on CPU automatically if needed.")
        return "cuda", torch.float16
    print("  [INFO] No CUDA GPU detected — running on CPU (slow but correct)")
    return "cpu", torch.float32


# Free-VRAM floor (MB) for keeping BOTH student + teacher resident on GPU for
# alpha eval.  Two separate floors:
#
# Laptop (4 GB card, 0.5B+0.6B fp16 pair):
#   Weights: ~2.2 GB.  KV cache + activations + fragmentation: ~0.5 GB.
#   Total: ~2.7 GB.  With 3.2 GB typically free after BE subprocess exits,
#   that leaves ~0.5 GB headroom — tight but safe.  Use 2700 MB floor.
#
# T4 / A100 (larger models, 8B teacher in NF4 + 0.6B student):
#   Weights alone ~6 GB.  Use 6000 MB floor.
_ALPHA_VRAM_FLOOR_MB        = 6000   # T4 / A100 / server
_ALPHA_VRAM_FLOOR_MB_LAPTOP = 2700   # 4 GB laptop (0.5B+0.6B fp16)


def _alpha_device_guard(default_device, default_dtype):
    """Proactively pick the device for loading BOTH alpha-eval models.

    On Windows under severe VRAM pressure the CUDA driver HARD-ABORTS the whole
    process (native access violation → Windows exit code 3221225786 / 0xC000013A)
    BEFORE PyTorch can raise the catchable torch.cuda.OutOfMemoryError.  When that
    happens the reactive `except torch.cuda.OutOfMemoryError` CPU fallback never
    runs and the pipeline step dies hard.  This guard avoids ever *attempting*
    the GPU load that triggers the native crash.

    Rules:
      * hw_tier == "laptop": load on CPU unconditionally.  The 0.6B + 0.5B pair
        plus KV cache + activations needs more headroom than a ~4 GB laptop card
        has alongside the driver/desktop/other allocations.  Alpha eval on laptop
        is a correctness / code-path check (not a perf measurement), so CPU is
        acceptable — ~20x slower but it won't crash.
      * any other tier on cuda: require a conservative free-VRAM floor
        (_ALPHA_VRAM_FLOOR_MB) before the load; fall back to CPU otherwise.

    Returns (device, dtype).  Pass-through unchanged when default_device != cuda.
    """
    import torch
    tier = getattr(args, "hw_tier", None) if args is not None else None
    if default_device != "cuda" or not torch.cuda.is_available():
        return default_device, default_dtype
    # hw_tier=cpu means the machine IS a CPU-only server (ATS/AIP).  Return CPU
    # directly — no VRAM floor logic, no device mismatch risk.  (On a true CPU
    # server CUDA is not available, so we would have returned above; this branch
    # guards the edge case where hw_tier=cpu but CUDA is accidentally present.)
    if tier == "cpu":
        return "cpu", torch.float32
    free_mb = torch.cuda.mem_get_info(0)[0] // 1024**2
    # Use a tier-specific VRAM floor instead of unconditionally forcing CPU on laptop.
    # The CPU path causes a device mismatch (cpu/cuda:0) when specInfer or the
    # KV-cache machinery creates CUDA tensors independently of the model device.
    # Using GPU avoids the mismatch and is ~20x faster.  The floor is calibrated
    # to the actual model sizes: 0.5B+0.6B fp16 (~2.7 GB) on laptop vs large
    # NF4 teacher on T4/A100 (~6 GB).
    floor_mb = _ALPHA_VRAM_FLOOR_MB_LAPTOP if tier == "laptop" else _ALPHA_VRAM_FLOOR_MB
    if free_mb < floor_mb:
        print(f"  [alpha preload] hw_tier={tier} / free VRAM={free_mb} MB < "
              f"{floor_mb} MB floor -> loading alpha-eval models on CPU "
              f"(device-mismatch risk accepted; GPU would OOM / hard-crash)")
        return "cpu", torch.float32
    print(f"  [alpha preload] hw_tier={tier} / free VRAM={free_mb} MB >= "
          f"{floor_mb} MB floor -> loading alpha-eval models on GPU (cuda)")
    return "cuda", default_dtype


def _pick_attn_impl() -> str:
    """
    Returns the fastest attention backend available on this machine:
      flash_attention_2  — requires pip install flash-attn (Ampere+ GPU: A100, H100).
      sdpa               — built into PyTorch ≥ 2.0, no extra deps, fast on any CUDA GPU.
      eager              — pure-PyTorch fallback, always works.
    Passed to every AutoModelForCausalLM.from_pretrained() call automatically.
    """
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        pass
    try:
        import torch
        if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            return "sdpa"
    except ImportError:
        pass
    return "eager"

_ATTN_IMPL = _pick_attn_impl()


# ---------------------------------------------------------------------------
# EAGLE published reference numbers (from EAGLE / EAGLE-2 papers)
# Source: arXiv 2401.15077 (EAGLE), 2406.16858 (EAGLE-2)
# NOTE: measured on LLaMA/Vicuna family, NOT Qwen3. Use as order-of-magnitude
# reference only until we measure EAGLE on Qwen3 directly.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Quality-warning collector
# Inline [WARN] prints are useful during the run; this list lets us print a
# single prominent block at the very end so nothing is missed in the scroll.
# ---------------------------------------------------------------------------

_run_warnings: list = []

# Module-level reference to parsed CLI args, set by main() before run_cell() is called.
# run_cell() is a module-level function and cannot see main()'s local scope, so args
# must live here for run_cell() to access train_steps, hw_tier, and load_in_4bit.
args = None


def _warn(msg: str):
    """Print a warning immediately AND add it to the end-of-run summary block."""
    print(f"  [WARN] {msg}")
    _run_warnings.append(msg)


EAGLE_REFERENCE = {
    # (model_family, benchmark): (speedup_x, block_eff)
    ("vicuna-7b",   "mt_bench"):   (2.95, 3.85),
    ("vicuna-13b",  "mt_bench"):   (3.05, 3.96),
    ("llama2-13b",  "mt_bench"):   (2.82, 3.71),
    ("llama3-8b",   "mt_bench"):   (3.52, 4.10),   # EAGLE-2
    ("llama3-70b",  "mt_bench"):   (3.72, 4.26),   # EAGLE-2
    # Qwen3: no published EAGLE numbers yet — placeholder
    ("qwen3-0.6b",  "gsm8k"):      (None, None),
    ("qwen3-8b",    "gsm8k"):      (None, None),
}


# ---------------------------------------------------------------------------
# Label / loss-name inference
# ---------------------------------------------------------------------------

# Known loss names (from pipeline.py training steps).
# Used to infer loss_name from student_label when evaluate.py is called by the pipeline.
_KNOWN_LOSSES = frozenset(["kl", "ebe", "rev_kl", "jsd", "l1", "online", "baseline"])


def infer_loss_name(student_label: str) -> str:
    """Return the loss name from a student label like 'kl', 'ebe', 'rev_kl'.

    The pipeline passes --student_label 'kl' / 'ebe' / ... which directly encodes
    the training loss.  For unknown labels we fall back to the label itself so the
    DB always has a meaningful value rather than the opaque sentinel 'custom'.
    """
    if student_label in _KNOWN_LOSSES:
        return student_label
    # Try stripping dataset suffix e.g. 'kl-gsm8k' → 'kl'
    base = student_label.split("-")[0].split("_")[0]
    if base in _KNOWN_LOSSES:
        return base
    return student_label   # fallback: use label as-is (never 'custom')


def infer_label(path: str) -> str:
    """Derive a short human-readable label from a model path or HF id."""
    name = path.rstrip("/\\")
    name = os.path.basename(name) or os.path.basename(os.path.dirname(name))
    # Shorten common names
    replacements = [
        ("Qwen2.5-0.5B", "qwen2.5-0.5b"),
        ("Qwen3-0.6B",   "qwen3-0.6b"),
        ("Qwen3-8B",     "qwen3-8b"),
        ("Qwen3-1.7B",   "qwen3-1.7b"),
        ("_merged",      ""),
    ]
    for old, new in replacements:
        name = name.replace(old, new)
    return name.lower()


def make_run_tag(student_label: str, mode: str, dataset: str, K: int,
                 temperature: float = None) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S%f")[:19]  # YYYYmmdd_HHMMSS + 2 ms digits
    tag = f"{ts}_{student_label}_{mode}_{dataset}_K{K}"
    if temperature is not None:
        t_str = f"{temperature:.2f}".replace(".", "p")
        tag += f"_T{t_str}"
    return tag


# ---------------------------------------------------------------------------
# Task accuracy scoring
# ---------------------------------------------------------------------------

def _extract_gsm8k_number(text: str):
    """Extract the final numeric answer from GSM8K-style text."""
    import re
    # Gold format: "...#### 18" — extract after ####
    m = re.search(r"####\s*([\d,\.\-]+)", text)
    if m:
        return m.group(1).replace(",", "").strip()
    # Fallback: last standalone number in text
    nums = re.findall(r"-?[\d,]+\.?\d*", text)
    return nums[-1].replace(",", "") if nums else None


def score_gsm8k(generated: str, gold_answer_text: str) -> float:
    """Return 1.0 if the final number in `generated` matches the GSM8K gold answer."""
    gold = _extract_gsm8k_number(gold_answer_text or "")
    pred = _extract_gsm8k_number(generated or "")
    if gold is None or pred is None:
        return 0.0
    try:
        return 1.0 if abs(float(gold) - float(pred)) < 1e-6 else 0.0
    except ValueError:
        return 1.0 if gold.strip() == pred.strip() else 0.0


def score_humaneval(prompt: str, generated: str, test_code: str,
                    entry_point: str = "", timeout: int = 10) -> float:
    """Return 1.0 if generated code passes HumanEval test cases (pass@1)."""
    import signal, traceback

    full_code = prompt + generated + "\n" + (test_code or "")
    namespace = {}

    def _handler(signum, frame):
        raise TimeoutError("execution timeout")

    try:
        # Windows doesn't support SIGALRM — use a thread-based timeout
        import threading
        result = [0.0]
        error = [None]

        def _run():
            try:
                exec(compile(full_code, "<humaneval>", "exec"), namespace)
                # Run check function if present
                check_fn = namespace.get(f"check_{entry_point}") or namespace.get("check")
                if check_fn:
                    check_fn(namespace.get(entry_point))
                result[0] = 1.0
            except Exception as e:
                error[0] = str(e)

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout=timeout)
        if t.is_alive():
            return 0.0  # timeout
        return result[0]
    except Exception:
        return 0.0


def run_task_score(student_path: str, dataset: str, prompts: list,
                   max_tokens: int = 512,
                   preloaded_student=None,
                   preloaded_tokenizer=None) -> dict:
    """
    Generate full responses with the student model (greedy, no SD) and
    compute task accuracy. Used as a quality-preservation sanity check.

    GSM8K  → exact match on final numerical answer
    HumanEval → pass@1 via code execution
    Other  → None (not scored)

    preloaded_student / preloaded_tokenizer: pass the already-resident model
    from the alpha-eval preload to avoid loading a SECOND copy while the
    teacher is still occupying VRAM (was causing OOM → silent CPU fallback
    → task_score taking 7+ hours instead of 2-3 minutes).
    """
    if dataset not in ("gsm8k", "humaneval"):
        return {"task_score": None, "scored": 0}

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    _own_model = preloaded_student is None   # True = we loaded it, we must free it

    if preloaded_student is not None:
        # Reuse the already-loaded draft model — no extra VRAM needed
        model    = preloaded_student
        tokenizer = preloaded_tokenizer or AutoTokenizer.from_pretrained(student_path, use_fast=False)
        device   = next(model.parameters()).device
        print(f"  [task_score] reusing preloaded student model (device={device})")
    else:
        device, dtype = _pick_device()
        tokenizer = AutoTokenizer.from_pretrained(student_path, use_fast=False)

        model = None
        for attempt in range(2):
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    student_path, **_dtype_kwargs(dtype), low_cpu_mem_usage=True,
                    attn_implementation=_ATTN_IMPL,
                ).to(device).eval()
                break
            except torch.cuda.OutOfMemoryError:
                if device == "cpu":
                    raise
                free_mb = torch.cuda.mem_get_info(0)[0] // 1024**2
                _n_p = len(prompts)
                _est_h = _n_p * 512 * 0.5 / 3600
                print(f"\n{'!' * 70}")
                print(f"  [WARNING] task_score FALLING BACK TO CPU — THIS WILL TAKE HOURS")
                print(f"  GPU OOM: only {free_mb} MB free (needed ~1200 MB for 0.6B model).")
                print(f"  CPU estimate: {_n_p} prompts × 512 tokens × ~500 ms/tok ≈ {_est_h:.1f} hrs")
                print(f"  ROOT CAUSE: alpha preloaded models still in VRAM when task_score runs.")
                print(f"  FIX applied in ef4a865: pass preloaded_student= to reuse existing model.")
                print(f"  IMMEDIATE: kill this process and git pull then restart.")
                print(f"{'!' * 70}\n")
                torch.cuda.empty_cache()
                device, dtype = "cpu", torch.float32

    # ── Batched generation: process N prompts simultaneously ──────────────
    # Sequential (old): 100 prompts × 1 at a time → GPU sits ~30% utilised
    # Batched (new): N prompts in parallel → GPU ~80-90% utilised
    #
    # Left-padding is REQUIRED for decoder-only models so all prompts start
    # generating at the same position in the padded batch.  With left-padding,
    # output[:, padded_len:] is the generated text for every prompt, no slicing
    # arithmetic needed per item.
    #
    # Batch size: 8 fits comfortably for a 0.6B model on any GPU (A100/T4).
    # Increase to 16 on A100-40GB or 32 on A100-80GB for more throughput.
    # Batch size from YAML evaluation.task_batch (machine-specific).
    # Set in hardware base YAMLs — NOT hardcoded here:
    #   bases/a100.yaml   → task_batch: 32  (A100-40GB + 0.6B: roofline ≈ 127)
    #   bases/laptop.yaml → task_batch:  4  (RTX 6GB + 82M)
    # Passed via --task_batch arg from experiment.py.
    # SPECDIST_TASK_BATCH env var overrides for ad-hoc tuning.
    # Default=1 (safe baseline — set your YAML to get the right value).
    _task_batch = int(os.environ.get("SPECDIST_TASK_BATCH",
                      str(getattr(args, "task_batch", 1))))
    tokenizer.padding_side = "left"   # required for decoder-only batched generation
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    scores = []
    _texts  = [item.get("prompt", item) if isinstance(item, dict) else item for item in prompts]
    _golds  = [item.get("answer", "")   if isinstance(item, dict) else "" for item in prompts]
    _tests  = [item.get("test", "")     if isinstance(item, dict) else "" for item in prompts]
    _eps    = [item.get("entry_point","") if isinstance(item, dict) else "" for item in prompts]

    n_batches = (len(prompts) + _task_batch - 1) // _task_batch
    print(f"  [task_score] {len(prompts)} prompts in {n_batches} batch(es) of {_task_batch}")

    for b_start in range(0, len(prompts), _task_batch):
        batch_texts = _texts[b_start : b_start + _task_batch]
        batch_enc   = tokenizer(batch_texts, return_tensors="pt",
                                padding=True, truncation=True,
                                max_length=1024).to(device)
        padded_len  = batch_enc["input_ids"].shape[-1]

        with torch.inference_mode():
            out = model.generate(
                **batch_enc,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        # Generated tokens start at padded_len for every item (left-padded inputs)
        gen_ids = out[:, padded_len:]

        for j, gen in enumerate(gen_ids):
            idx = b_start + j
            generated = tokenizer.decode(gen, skip_special_tokens=True)
            if dataset == "gsm8k":
                scores.append(score_gsm8k(generated, _golds[idx]))
            elif dataset == "humaneval":
                scores.append(score_humaneval(batch_texts[j], generated,
                                              _tests[idx], _eps[idx]))

    if _own_model:
        del model   # only free what we allocated — don't delete the preloaded alpha model
        if str(device) == "cuda":
            torch.cuda.empty_cache()

    mean_score = sum(scores) / len(scores) if scores else 0.0
    return {"task_score": mean_score, "scored": len(scores),
            "per_prompt_scores": scores}


# ---------------------------------------------------------------------------
# Alpha evaluation (inline)
# ---------------------------------------------------------------------------

def run_alpha(student_path: str, teacher_path: str, student_label: str,
              prompts: list, temperature: float,
              max_propose: int = 5, max_tokens: int = 30,
              preloaded: tuple = None,
              load_in_4bit: bool = False) -> dict:
    """
    preloaded: optional (device, dtype, tokenizer, student_model, teacher_model, same)
    tuple supplied by main() when models are kept resident across multiple alpha cells
    (avoids reloading both models for every dataset).  When None, loads fresh as before.
    """
    import torch
    import numpy as np
    import torch.nn.functional as _F
    from transformers import AutoTokenizer, AutoModelForCausalLM

    # specInfer lives in OSD/distill/ which may not be present on every machine.
    # Fall back to an inline draft-propose / target-verify loop when it is absent.
    try:
        from specInfer.generator import Generator as _SpecInferGenerator
        _SPECINFER_AVAILABLE = True
    except ImportError:
        _SpecInferGenerator = None
        _SPECINFER_AVAILABLE = False
        # Likely cause: OSD submodule not initialised on this machine.
        # Fix: run  git submodule update --init --recursive  from the repo root.
        # Alpha measurements will use the inline fallback (greedy draft) which
        # gives slightly different alpha values than the real specInfer generator.

    _owns_models = (preloaded is None)   # True → we loaded, we must free

    if preloaded is not None:
        device, dtype, tokenizer, student_model, teacher_model, same = preloaded
    else:
        device, dtype = _pick_device()
        # Same PROACTIVE VRAM guard as the alpha pre-load path.  In practice the
        # pre-load path (main()) handles laptop alpha eval, so this branch with
        # preloaded=None is rarely hit on laptop — but guard it too so a direct /
        # standalone run_alpha() call can't trigger the 0xC000013A native crash.
        device, dtype = _alpha_device_guard(device, dtype)

        tokenizer = AutoTokenizer.from_pretrained(teacher_path, use_fast=False)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        student_model = teacher_model = None
        _load_4bit = load_in_4bit
        for attempt in range(2):
            try:
                student_model = AutoModelForCausalLM.from_pretrained(
                    student_path, **_dtype_kwargs(dtype), low_cpu_mem_usage=True,
                    attn_implementation=_ATTN_IMPL,
                ).to(device).eval()
                same = (student_path == teacher_path)
                if same:
                    teacher_model = student_model
                elif _load_4bit:
                    # QLoRA mode: frozen teacher in 4-bit NF4 — fits 8B on T4 (15 GB)
                    try:
                        from transformers import BitsAndBytesConfig as _BnB
                    except ImportError:
                        raise SystemExit("bitsandbytes required for --load_in_4bit. "
                                         "Run: pip install bitsandbytes")
                    _bnb = _BnB(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                bnb_4bit_compute_dtype=torch.bfloat16,
                                bnb_4bit_use_double_quant=True)
                    teacher_model = AutoModelForCausalLM.from_pretrained(
                        teacher_path, quantization_config=_bnb,
                        device_map={"": "cuda:0"}, low_cpu_mem_usage=True,
                        attn_implementation=_ATTN_IMPL,
                    ).eval()
                    print(f"  [alpha] Teacher loaded in 4-bit NF4 on cuda:0")
                else:
                    teacher_model = AutoModelForCausalLM.from_pretrained(
                        teacher_path, **_dtype_kwargs(dtype), low_cpu_mem_usage=True,
                        attn_implementation=_ATTN_IMPL,
                    ).to(device).eval()
                break
            except torch.cuda.OutOfMemoryError:
                if device == "cpu":
                    raise
                free_mb = torch.cuda.mem_get_info(0)[0] // 1024**2
                print(f"\n  [OOM] CUDA out of memory loading models ({free_mb} MB free). "
                      f"Retrying alpha eval on CPU — will be ~20x slower.")
                torch.cuda.empty_cache()
                if student_model is not None:
                    del student_model
                    student_model = None
                device, dtype = "cpu", torch.float32

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    alphas, per_prompt_rows = [], []
    total_tokens, total_time = 0, 0.0
    draft_times, verify_times = [], []

    # specInfer internally creates CUDA buffers at Generator.__init__ regardless
    # of where the models live — it has no device parameter.  On a CPU run this
    # contaminates the process with cuda:0 tensors, causing wrapper_CUDA_cat to
    # crash when the inline fallback later tries to cat CPU hidden states with
    # those leftover CUDA buffers.  Skip specInfer entirely when device is CPU.
    #
    # SPECDIST_DISABLE_SPECINFER=1 forces the inline fallback even on CUDA.
    # Use when specInfer's DynamicCache API is incompatible with the installed
    # transformers version and the .pyc cache fix hasn't propagated yet.
    # Alpha values are IDENTICAL — inline fallback is ~20% slower only.
    _specinfer_env_disabled = os.environ.get("SPECDIST_DISABLE_SPECINFER", "0") == "1"
    _use_specinfer = _SPECINFER_AVAILABLE and device == "cuda" and not _specinfer_env_disabled
    if _specinfer_env_disabled and device == "cuda":
        print("  [alpha] specInfer disabled via SPECDIST_DISABLE_SPECINFER=1 — using inline fallback.")
    if _use_specinfer:
        generator = _SpecInferGenerator(
            small_model=student_model, large_model=teacher_model,
            tokenizer=tokenizer, max_propose_num=max_propose,
            is_encoder_decoder=False, use_cache=True,
        )
    else:
        generator = None
        global _SPECINFER_FALLBACK_WARNED
        if not _SPECINFER_FALLBACK_WARNED:
            reason = ("running on CPU — specInfer requires CUDA"
                      if device != "cuda" else
                      "not found (git submodule update --init --recursive to fix)")
            print(f"  [alpha] specInfer skipped — {reason}; using inline fallback "
                  "(warned once per session)")
            _SPECINFER_FALLBACK_WARNED = True

    for i, item in enumerate(prompts):
        prompt = item["prompt"] if isinstance(item, dict) else item
        category = item.get("category") if isinstance(item, dict) else None
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        if _use_specinfer:
            try:
                with torch.inference_mode():
                    output = generator.generate(
                        input_ids=input_ids,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        attention_mask=torch.ones_like(input_ids),
                    )
                if device == "cuda":
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0

                gen_tokens = (output.output[0].shape[-1] - input_ids.shape[-1]
                              if hasattr(output.output[0], "shape") else max_tokens)
                total_tokens += gen_tokens
                total_time += elapsed
                alpha = float(output.alpha_sum) / output.sample_steps if output.sample_steps > 0 else 0.0
            except (AttributeError, TypeError) as _spec_err:
                # specInfer KV-cache API mismatch (e.g. DynamicCache object is not
                # subscriptable in newer transformers).  Switch all remaining prompts
                # to the inline fallback — same alpha values, no crash.
                if i == 0:
                    print(f"  [alpha] specInfer incompatible with this transformers version "
                          f"({type(_spec_err).__name__}: {_spec_err}); using inline fallback.")
                _SPECINFER_AVAILABLE = False
                generator = None
                _use_specinfer = False   # ← CRITICAL: also update local var so fallback runs
                # Fall through to inline block below for this prompt.
        if not _use_specinfer:
            # Inline fallback: draft-propose / target-verify without specInfer.
            # Uses KV cache for draft proposals (1 prefill + K single-token passes)
            # and a single teacher forward pass per round — avoids redundant full-
            # sequence re-run of the student that the old code did.
            accepted_total, proposed_total = 0, 0
            cur_ids = input_ids
            d_pkv = None   # draft KV cache — grown with accepted tokens each round
            with torch.inference_mode():
                for _ in range(max(1, max_tokens // max_propose)):
                    # ── Draft: prefill (or single-token if we have KV cache) ──
                    if d_pkv is None:
                        d_out = student_model(cur_ids, use_cache=True)
                    else:
                        # Only feed the last accepted token; reuse cached context
                        d_out = student_model(cur_ids[:, -1:], past_key_values=d_pkv,
                                              use_cache=True)
                    d_pkv_propose = d_out.past_key_values

                    # ── Propose K draft tokens ──
                    # logits[0, -1] → shape (vocab,)   ← correct: scalar-indexed last pos
                    # logits[0, -1:] → shape (1, vocab) ← WRONG: [tok] would index dim-0
                    #                                      causing "index N out of bounds
                    #                                      for dimension 0 with size 1"
                    draft_tokens = []
                    draft_logits = []           # per-token logits, each shape (vocab,)
                    d_logits = d_out.logits[0, -1]   # (vocab,)
                    pkv_k = d_pkv_propose
                    tmp_ids = cur_ids
                    for _k in range(max_propose):
                        tok = int(d_logits.argmax())
                        draft_tokens.append(tok)
                        draft_logits.append(d_logits)        # (vocab,) — correct shape
                        next_tok = torch.tensor([[tok]], device=device)
                        d_kout = student_model(next_tok, past_key_values=pkv_k,
                                               use_cache=True)
                        d_logits = d_kout.logits[0, -1]      # (vocab,)
                        pkv_k = d_kout.past_key_values
                        tmp_ids = torch.cat([tmp_ids, next_tok], dim=1)

                    # ── Target: score full candidate in ONE forward pass ──
                    cand_ids = tmp_ids            # cur_ids + draft_tokens
                    t_all_logits = teacher_model(cand_ids).logits[0]  # (seq, V)
                    # Draft logits come from the proposal loop above — no extra pass needed

                    # ── Acceptance under temperature (sequential rejection sampling) ──
                    bonus_start = cur_ids.shape[-1] - 1
                    n_accepted = 0
                    for _k in range(max_propose):
                        pos = bonus_start + _k
                        tok = draft_tokens[_k]
                        if temperature > 0:
                            t_p = float(_F.softmax(t_all_logits[pos] / temperature, dim=-1)[tok])
                            d_p = float(_F.softmax(draft_logits[_k] / temperature, dim=-1)[tok])
                            ratio = t_p / max(d_p, 1e-9)
                        else:
                            ratio = 1.0 if int(t_all_logits[pos].argmax()) == tok else 0.0
                        proposed_total += 1
                        if torch.rand(1).item() < min(1.0, ratio):
                            n_accepted += 1
                            accepted_total += 1
                        else:
                            break

                    # ── Advance: accepted draft tokens + 1 bonus from target ──
                    bonus = int(t_all_logits[bonus_start + n_accepted].argmax())
                    new_toks = draft_tokens[:n_accepted] + [bonus]
                    new_tensor = torch.tensor([new_toks], device=device)
                    cur_ids = torch.cat([cur_ids, new_tensor], dim=1)
                    # Update draft KV cache to cover the accepted tokens
                    # (next iteration prefill will extend from here)
                    d_pkv = d_pkv_propose   # cached up to cur_ids before proposal
                    if cur_ids.shape[-1] >= input_ids.shape[-1] + max_tokens:
                        break

            if device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            gen_tokens = cur_ids.shape[-1] - input_ids.shape[-1]
            total_tokens += gen_tokens
            total_time += elapsed
            alpha = accepted_total / proposed_total if proposed_total > 0 else 0.0

        alphas.append(alpha)
        per_prompt_rows.append(dict(prompt_idx=i, category=category, alpha=alpha,
                                    block_eff=None, gen_tokens=gen_tokens,
                                    wall_ms=elapsed * 1000))

    import numpy as np
    alphas = np.array(alphas)
    n = len(alphas)
    peak_vram = torch.cuda.max_memory_allocated() / 1024**2 if device == "cuda" else 0.0

    # Only free models if we loaded them (not when caller holds them resident)
    if _owns_models:
        del student_model
        if not same:
            del teacher_model
        if device == "cuda":
            torch.cuda.empty_cache()

    return dict(
        alpha_mean=float(alphas.mean()), alpha_std=float(alphas.std()),
        alpha_ci95=float(1.96 * alphas.std() / (n ** 0.5)),
        throughput=total_tokens / total_time if total_time > 0 else 0,
        ms_per_tok=total_time / total_tokens * 1000 if total_tokens > 0 else 0,
        peak_vram_mb=peak_vram,
        # Latency breakdown (ms per target call, approximate via wall time / calls)
        draft_latency_ms=(sum(draft_times)/len(draft_times)*1000) if draft_times else None,
        verify_latency_ms=(sum(verify_times)/len(verify_times)*1000) if verify_times else None,
        per_prompt=per_prompt_rows,
        # Which acceptance-measurement path produced these alphas.  When specInfer
        # is incompatible with the installed transformers version, alpha is computed
        # by the inline draft-propose/target-verify fallback — a DIFFERENT algorithm.
        # Recorded so alpha rows are never silently misattributed to "true specInfer".
        alpha_method=("specinfer" if _SPECINFER_AVAILABLE else "inline_fallback"),
    )


# ---------------------------------------------------------------------------
# GBV block efficiency (subprocess)
# ---------------------------------------------------------------------------

def run_be(student_path: str, teacher_path: str,
           mode: str, K: int, L: int, data_path: str, max_new_tokens: int,
           _device: str = "cuda") -> dict | None:
    """Run GBV block-efficiency eval as a subprocess. Retries on CPU if GPU OOMs."""
    import torch
    cmd = [
        sys.executable,
        _VERIFIER_SCRIPT,
        "--p_model", teacher_path,
        "--q_model", student_path,
        "--mode", mode,
        "--K", str(K),
        "--L", str(L),
        "--max_new_tokens", str(max_new_tokens),
        "--data", data_path,
        "--device", _device,
    ]
    _sub_env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                                errors="replace", timeout=1800, cwd=_PARENT, env=_sub_env)
        out = result.stdout + result.stderr
        m = re.search(r"Block efficiency[^:]*:\s*([\d.]+)", out)
        if not m:
            oom = "out of memory" in out.lower() or "outofmemory" in out.lower()
            if oom and _device == "cuda":
                print(f"    [OOM] GBV subprocess ran out of GPU memory — retrying on CPU (slower)")
                torch.cuda.empty_cache()
                return run_be(student_path, teacher_path, mode, K, L,
                              data_path, max_new_tokens, _device="cpu")
            err_preview = (result.stderr or result.stdout or "(no output)")[:500]
            print(f"    [WARN] No block efficiency in GBV output "
                  f"(rc={result.returncode}, mode={mode}, K={K}, device={_device}):\n"
                  f"    {err_preview}")
            return None
        return {"block_eff": float(m.group(1))}
    except subprocess.TimeoutExpired:
        print(f"    [TIMEOUT] GBV subprocess timed out after 30 min "
              f"(mode={mode}, K={K}). Consider reducing --n or --max_tokens.")
        return None
    except Exception as e:
        print(f"    [ERROR] GBV subprocess error: {e}")
        return None


def run_be_batch(student_path: str, teacher_path: str, data_path: str,
                 modes: list, Ks: list, temps: list,
                 L: int, max_new_tokens: int,
                 _device: str = "cuda",
                 load_in_4bit: bool = False) -> dict:
    """
    Run all (mode, K, temp) combos in ONE GBV subprocess — models load once.

    Instead of spawning one subprocess per (mode, K, T) cell (the old behaviour
    that reloaded both models 72 times), this passes comma-separated --modes,
    --Ks, --p_temps to the verifier script (_VERIFIER_SCRIPT, Phase 2: runner.py)
    which iterates over all combos in one process and prints a tagged result line:
        Block efficiency (mode=gbv, K=3, T=1.0): 2.345678

    Output is streamed to be_progress.log in real time (readable via the
    viz_server "Logs" panel or `tail -f be_progress.log` in a terminal).

    Returns {(mode, K, temp): block_eff} dict.
    Falls back to CPU automatically on OOM.
    """
    import torch
    n_combos = len(modes) * len(Ks) * len(temps)
    print(f"    loading models once for {n_combos} combo(s) "
          f"({len(modes)} mode(s) x {len(Ks)} K x {len(temps)} temp(s)) ...")
    cmd = [
        sys.executable,
        "-u",                              # -u = unbuffered stdout/stderr in subprocess
        _VERIFIER_SCRIPT,
        "--p_model",  teacher_path,
        "--q_model",  student_path,
        "--modes",    ",".join(modes),
        "--Ks",       ",".join(str(k) for k in Ks),
        "--p_temps",  ",".join(str(t) for t in temps),
        "--L",        str(L),
        "--max_new_tokens", str(max_new_tokens),
        "--data",     data_path,
        "--device",   _device,
        "--dtype",    "bf16",      # explicit bf16 — avoids silent fp32 fallback on CUDA GPUs
        # model_family → runner.py uses family.tree_attn_mask() so the verifier
        # never inspects model architecture directly (separation of concerns).
        "--model_family", getattr(args, "model_family", "qwen") or "qwen",
    ]
    if load_in_4bit:
        cmd.append("--load_in_4bit")   # GBV/main.py loads teacher in 4-bit NF4 on Colab T4
    # PYTHONUNBUFFERED=1 forces line-by-line flushing inside the subprocess so
    # be_progress.log updates in real time rather than in large chunks.
    #
    # TRANSFORMERS_OFFLINE / HF_HUB_OFFLINE: explicitly pass experiment.py's
    # decision through to the subprocess.  The OSD module sets
    # TRANSFORMERS_OFFLINE=1 at import time (its "offline by default" policy),
    # which overrides experiment.py's choice to stay online for first-time
    # model downloads.  By pinning these to whatever experiment.py decided
    # (or "0" if it left them unset = online), we prevent OSD from blocking
    # downloads of models that aren't cached yet (e.g. distilgpt2, gpt2-medium).
    _sub_env = {
        **os.environ,
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUNBUFFERED": "1",
        # Propagate the HF offline decision from experiment.py explicitly.
        # evaluate.py's module-level code sets TRANSFORMERS_OFFLINE=1 for
        # non-cloud runs; but experiment.py may have set it to "0" for first-
        # time downloads. os.environ at this point reflects experiment.py's
        # decision, so {**os.environ} already carries the right value.
        # Explicit keys here guard against any future module-level override
        # between subprocess creation and this dict being built.
        "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE", "1"),
        "HF_HUB_OFFLINE":       os.environ.get("HF_HUB_OFFLINE",       "1"),
    }
    if sys.platform != "win32":   # expandable_segments is Linux-only
        _sub_env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    # SPECDIST_LOGS_ROOT is set by experiment.py to the run-specific log subdir
    # (e.g. db/logs/laptop_gpt2-dg2-g2m/) so be_progress.log lands alongside
    # pipeline_output.log for the same run.  Falls back to the flat db/logs/ dir
    # when evaluate.py is called directly without the pipeline.
    _db_logs = (os.environ.get("SPECDIST_LOGS_ROOT")
                or os.path.join(_PARENT, "db", "logs"))
    os.makedirs(_db_logs, exist_ok=True)
    _be_log = os.path.join(_db_logs, "be_progress.log")

    try:
        # Open log for streaming — stderr merged into stdout so tqdm bars appear too.
        # Output is teed to BOTH be_progress.log (for Dashboard / tail) AND our own
        # stdout (which pipeline.py redirects to pipeline_output.log), so eval progress
        # is visible in the main pipeline log without opening a second file.
        # Kill truly orphaned runner.py processes from a previous interrupted run.
        # Without this, a restarted evaluate.py (e.g. after Kaggle cell re-run)
        # launches a second BE subprocess while the first one is still loading
        # models, exhausting GPU VRAM.
        #
        # SAFE on multi-GPU servers: we only kill runner.py processes whose
        # PARENT process is dead (i.e. genuinely orphaned — their evaluate.py
        # was killed but they kept running).  We never kill runner.py processes
        # that are alive children of another active evaluate.py (a different GPU
        # slot on the same machine).  Killing all runner.py system-wide would
        # terminate another GPU's eval mid-run.
        try:
            import psutil
            for p in psutil.process_iter(["pid", "ppid", "cmdline"]):
                try:
                    cmdline = " ".join(p.info["cmdline"] or [])
                    if "runner.py" not in cmdline or p.pid == os.getpid():
                        continue
                    # Check if this runner.py is truly orphaned:
                    # its parent process no longer exists or is not running.
                    ppid = p.info.get("ppid", 0)
                    try:
                        parent = psutil.Process(ppid)
                        parent_alive = parent.is_running()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        parent_alive = False
                    if not parent_alive:
                        p.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except ImportError:
            pass  # psutil not installed — best-effort only
        import torch as _torch
        _torch.cuda.empty_cache()

        with open(_be_log, "w", encoding="utf-8", errors="replace") as _log_f:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                cwd=_PARENT, env=_sub_env,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
            )
            print(f"    GBV subprocess PID {proc.pid} — streaming to be_progress.log", flush=True)

            def _tee(pipe):
                """Write each line to be_progress.log AND pipeline_output.log."""
                for line in pipe:
                    _log_f.write(line)
                    _log_f.flush()
                    sys.stdout.write(line)
                    sys.stdout.flush()

            _tee_thread = threading.Thread(target=_tee, args=(proc.stdout,), daemon=True)
            _tee_thread.start()
            # Timeout: scale with n_prompts × n_combos so large A100 evals don't
            # time out mid-run.  At ~5s/prompt on A100:
            #   n=5,   8 combos → 200s   → floor at 1800s (30 min)
            #   n=100, 8 combos → 4000s  → 4000s
            #   n=1319,8 combos → 52760s → 52760s (~14.7h)
            # Add 30% safety margin.  The timeout is only a kill-switch for truly
            # hung processes; a healthy run finishes well within this bound.
            _n_prompts_est = sum(1 for _ in open(data_path)) if os.path.exists(data_path) else 100
            _n_combos = len(modes) * len(Ks) * len(temps)
            _be_timeout = max(1800, int(_n_prompts_est * _n_combos * 5 * 1.3))
            _timeout_h = _be_timeout / 3600
            print(f"    BE timeout: {_be_timeout}s ({_timeout_h:.1f}h) "
                  f"for {_n_prompts_est} prompts × {_n_combos} combos")
            _timed_out = False
            try:
                rc = proc.wait(timeout=_be_timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                _timed_out = True
                print(f"    [TIMEOUT] GBV batch subprocess timed out after {_timeout_h:.1f}h. "
                      f"Consider reducing --n or --max_tokens.")
                # Do NOT return here — fall through to parse whatever completed combos
                # are already in the log.  Combos that finished before the timeout will
                # be saved to the DB; on restart --skip_existing will skip them so only
                # the unfinished combos re-run.
            _tee_thread.join(timeout=10)  # drain any last lines before closing the log

        # Read back the log for parsing (now the process has exited or been killed)
        with open(_be_log, encoding="utf-8", errors="replace") as f:
            out = f.read()

        # OOM retry only applies to clean (non-timeout) exits
        if not _timed_out:
            oom = "out of memory" in out.lower() or "outofmemory" in out.lower()
            if oom and _device == "cuda":
                print(f"    [OOM] GBV batch subprocess OOM — retrying on CPU (slower)")
                torch.cuda.empty_cache()
                return run_be_batch(student_path, teacher_path, data_path,
                                    modes, Ks, temps, L, max_new_tokens, _device="cpu",
                                    load_in_4bit=load_in_4bit)

        # Parse tagged output: "Block efficiency (mode=gbv, K=3, T=1.0): 2.345678"
        # Works on partial output — only fully-printed lines are matched.
        results: dict = {}
        for m in re.finditer(
                r"Block efficiency \(mode=(\w+), K=(\d+), T=([\d.]+)\):\s*([\d.]+)", out):
            results[(m.group(1), int(m.group(2)), float(m.group(3)))] = float(m.group(4))

        if _timed_out:
            if results:
                remaining = n_combos - len(results)
                print(f"    [PARTIAL] Recovered {len(results)}/{n_combos} combo(s) from "
                      f"timed-out run. Re-run to complete {remaining} remaining combo(s) "
                      f"(--skip_existing will skip the {len(results)} already saved).")
            else:
                print(f"    [PARTIAL] No completed combos in timed-out run — "
                      f"all {n_combos} combo(s) will re-run on restart.")
        elif not results:
            err_preview = out[:600] or "(no output)"
            print(f"    [WARN] No BE values in GBV batch output "
                  f"(rc={rc}):\n    {err_preview}")
        else:
            print(f"    {len(results)}/{n_combos} BE results received.")
        return results

    except Exception as e:
        print(f"    [ERROR] GBV batch subprocess error: {e}")
        return {}


# ---------------------------------------------------------------------------
# Perplexity measurement
# ---------------------------------------------------------------------------

def measure_perplexity(model_path: str, prompts: list, max_tokens: int = 200) -> dict:
    """
    Measure perplexity of model_path on the given prompts.
    Uses the model to score its own continuations (auto-regressive NLL).
    Lower perplexity = better language model quality.
    """
    import torch
    import math
    from transformers import AutoTokenizer, AutoModelForCausalLM

    device, dtype = _pick_device()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, **_dtype_kwargs(dtype), low_cpu_mem_usage=True,
        attn_implementation=_ATTN_IMPL,
    ).to(device).eval()
    # Suppress "loss_type=None is unrecognized" warning from transformers ≥4.46.
    # The default fallback IS ForCausalLMLoss — we're just making it explicit
    # so transformers doesn't log a warning on every forward pass with labels=.
    if not getattr(model.config, "loss_type", None):
        model.config.loss_type = "ForCausalLMLoss"

    total_nll, total_tokens = 0.0, 0
    per_prompt = []
    for i, item in enumerate(prompts):
        prompt = item["prompt"] if isinstance(item, dict) else item
        ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                        max_length=max_tokens).input_ids.to(device)
        with torch.inference_mode():
            out = model(ids, labels=ids)
        nll = out.loss.item()
        n_tok = ids.shape[-1] - 1
        total_nll += nll * n_tok
        total_tokens += n_tok
        per_prompt.append({"prompt_idx": i, "ppl": math.exp(nll)})

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    avg_nll = total_nll / total_tokens if total_tokens > 0 else float("nan")
    return {"perplexity": math.exp(avg_nll), "per_prompt": per_prompt}


# ---------------------------------------------------------------------------
# Skip-existing check
# ---------------------------------------------------------------------------

def _already_run(student_label: str, dataset: str, mode: str, K: int, temperature: float,
                  student_path: str | None = None, n_prompts: int | None = None) -> bool:
    """Return True if a matching run already exists in the DB with the same prompt count.

    n_prompts is checked when provided to prevent smoke results (n=5) from
    blocking full-run eval (n=1319 for A100, n=100 for T4).  Without this,
    a smoke baseline with n=5 would make _already_run return True and the
    full-run baseline eval would be silently skipped.

    student_path is included in the match so results for Qwen/Qwen2.5-0.5B
    and distilgpt2 both labelled 'baseline' are never confused.
    """
    runs = results_db.query_runs({
        "draft_label": student_label,
        "dataset": dataset,
        "mode": mode,
        "K": K,
    })
    matching = [r for r in runs if abs(r.get("temperature", 0) - temperature) < 0.01]
    if not matching:
        return False
    # Filter by n_prompts: a smoke result (n=5) must NOT block a full run (n=1319).
    # Allow ±10% tolerance for minor prompt-count differences between runs.
    if n_prompts is not None:
        matching = [r for r in matching
                    if r.get("n_prompts") is None                    # old row, no n_prompts stored
                    or abs(r.get("n_prompts", 0) - n_prompts) <= max(1, n_prompts * 0.1)]
        if not matching:
            return False
    # If we know the student path, require it to match so different model
    # families sharing the same label don't cross-skip each other.
    if student_path:
        return any(r.get("draft_path", "") == student_path for r in matching)
    return True


# ---------------------------------------------------------------------------
# Load prompts
# ---------------------------------------------------------------------------

def load_prompts(dataset: str, n: int) -> list:
    path = get_dataset_path(dataset, n)
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Dataset {dataset} not found. Run fetch_datasets.py first.")
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items[:n] if n else items


# ---------------------------------------------------------------------------
# Single experiment cell
# ---------------------------------------------------------------------------

def run_cell(student_path: str, teacher_path: str, student_label: str,
             dataset: str, mode: str, K: int, L: int,
             temperature: float, n: int, max_tokens: int,
             skip_existing: bool = False, run_perplexity: bool = False,
             task_score: bool = False, experiment_tag: str = None,
             preloaded: tuple = None) -> dict:
    run_tag = make_run_tag(student_label, mode, dataset, K)

    if skip_existing and _already_run(student_label, dataset, mode, K, temperature,
                                       student_path=student_path, n_prompts=n):
        print(f"\n[{run_tag}] SKIP (already in DB)")
        return {"skipped": True}

    print(f"\n[{run_tag}]")

    prompts = load_prompts(dataset, n)
    data_path = get_dataset_path(dataset, n)

    base_row = dict(
        run_tag=run_tag,
        draft_label=student_label,
        draft_path=student_path,
        target_path=teacher_path,
        loss_name=infer_loss_name(student_label),
        train_steps=args.train_steps, learning_rate=0.0, lora_rank=0,
        dataset=dataset, n_prompts=len(prompts),
        mode=mode, K=K, L=L, temperature=temperature,
        experiment_tag=experiment_tag,
    )

    if mode == "alpha":
        res = run_alpha(student_path, teacher_path, student_label,
                        prompts, temperature, max_tokens=max_tokens,
                        preloaded=preloaded,
                        load_in_4bit=getattr(args, "load_in_4bit", False))
        row = {**base_row,
               "alpha_mean": res["alpha_mean"], "alpha_std": res["alpha_std"],
               "alpha_ci95": res["alpha_ci95"], "throughput": res["throughput"],
               "ms_per_tok": res["ms_per_tok"], "peak_vram_mb": res["peak_vram_mb"],
               "draft_latency_ms": res.get("draft_latency_ms"),
               "verify_latency_ms": res.get("verify_latency_ms")}
        # Record the alpha measurement path in notes so inline-fallback alpha is
        # never mistaken for true specInfer alpha (see run_alpha alpha_method).
        _amethod = res.get("alpha_method", "specinfer")
        row["notes"] = (f"alpha_method={_amethod}"
                        + (f"; {row['notes']}" if row.get("notes") else ""))
        if _amethod != "specinfer":
            print(f"  [alpha] measured via {_amethod} (NOT true specInfer) — "
                  f"tagged in DB notes")

        # Task accuracy (GSM8K / HumanEval only)
        if task_score and dataset in ("gsm8k", "humaneval"):
            print(f"  [task_score] running {dataset} accuracy...")
            # Pass preloaded student model to avoid loading a second copy while
            # the teacher is still resident in GPU memory (caused OOM → CPU fallback).
            _pre_sm = preloaded[3] if preloaded else None
            _pre_tok = preloaded[2] if preloaded else None
            ts_res = run_task_score(student_path, dataset, prompts, max_tokens=512,
                                    preloaded_student=_pre_sm,
                                    preloaded_tokenizer=_pre_tok)
            row["task_score"] = ts_res.get("task_score")
            print(f"  task_score={row['task_score']:.3f} ({ts_res['scored']} samples)")
            # Quality guard: warn if student is meaningfully worse than the best baseline on record
            _bl_scores = [
                r["task_score"] for r in results_db.query_runs(
                    {"draft_label": "baseline", "dataset": dataset, "mode": "alpha"})
                if r.get("task_score") is not None
            ]
            if _bl_scores and row["task_score"] is not None:
                _best_bl = max(_bl_scores)
                if row["task_score"] < _best_bl - 0.05:   # >5 pp drop
                    _warn(f"task_score={row['task_score']:.3f} is >5pp below "
                          f"baseline={_best_bl:.3f} on {dataset} — "
                          f"check for training collapse or wrong model path.")

        print(f"  alpha={res['alpha_mean']:.4f} ±{res['alpha_ci95']:.4f}  "
              f"{res['throughput']:.2f} tok/s"
              + (f"  task={row.get('task_score'):.3f}" if row.get("task_score") is not None else ""))
        run_id = results_db.insert_run(row, hw_tier=args.hw_tier)
        pp = res["per_prompt"]
        for r in pp:
            r["run_id"] = run_id
        results_db.insert_per_prompt_batch(pp)

        # ── W&B: eval runs have no time-series; write everything to summary ──
        # Using wandb.summary (not wandb.log) avoids meaningless "step" charts.
        # Summary keys show up as columns in the W&B runs comparison table.
        try:
            import wandb as _wmod
            if _wmod.run is not None:
                # Per-dataset alpha detail omitted from summary to keep panel count ≤12.
                # Full per-prompt detail is in the per_prompt_alpha artifact below.
                # Aggregate mean_alpha / max_alpha are written at end-of-run in main().
                # Per-prompt table stored as a W&B Artifact so it appears in the
                # Artifacts tab (not as a chart panel — there is no meaningful step axis).
                if pp:
                    _art = _wmod.Artifact(
                        name=f"per_prompt_alpha_{dataset}_{run_tag}",
                        type="eval_detail",
                        description=f"Per-prompt alpha values for {dataset}",
                    )
                    _tbl = _wmod.Table(
                        columns=["prompt_idx", "category", "alpha", "wall_ms"],
                        data=[[r2.get("prompt_idx", i), r2.get("category"),
                               r2.get("alpha", 0), r2.get("wall_ms")]
                              for i, r2 in enumerate(pp)]
                    )
                    _art.add(_tbl, f"alpha_{dataset}")
                    _wmod.log_artifact(_art)
        except Exception:
            pass  # W&B logging is always best-effort

    else:
        res = run_be(student_path, teacher_path, mode, K, L, data_path, max_tokens)
        if res is None:
            return {}
        row = {**base_row, "block_eff": res["block_eff"]}
        print(f"  block_eff={res['block_eff']:.4f}")
        run_id = results_db.insert_run(row, hw_tier=args.hw_tier)

        # No per-(mode,dataset) W&B summary key here: writing BE/{mode}/{dataset}
        # produces up to 25 panels (5 modes × 5 datasets) that clutter the run.
        # The per-mode average (BE/{mode}) is written once at end-of-run in main(),
        # and full per-(mode,dataset) detail lives in the eval_summary_table.

    row["id"] = run_id
    return row


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(results: list):
    print("\n" + "=" * 90)
    print(f"{'RUN TAG':<45} {'ALPHA':>8} {'BE':>8} {'TASK':>7} {'PPL':>7}")
    print("-" * 90)
    for r in results:
        tag   = r.get("run_tag", "?")[:44]
        alpha = f"{r['alpha_mean']:.4f}"  if r.get("alpha_mean")  is not None else "      -"
        be    = f"{r['block_eff']:.4f}"   if r.get("block_eff")   is not None else "      -"
        task  = f"{r['task_score']:.3f}"  if r.get("task_score")  is not None else "     -"
        ppl   = f"{r['perplexity']:.2f}"  if r.get("perplexity")  is not None else "     -"
        print(f"{tag:<45} {alpha:>8} {be:>8} {task:>7} {ppl:>7}")
    print("=" * 90)
    # Quality-preservation check: if any trained model has task_score, compare to baseline
    trained = [r for r in results if r.get("task_score") is not None and r.get("draft_label") != "baseline"]
    baseline = [r for r in results if r.get("task_score") is not None and r.get("draft_label") == "baseline"]
    if trained and baseline:
        best_bl = max(r["task_score"] for r in baseline)
        for r in trained:
            delta = r["task_score"] - best_bl
            flag = "  *** [WARN] >5pp drop — check for collapse" if delta < -0.05 else ""
            print(f"  quality check: {r['draft_label']} task={r['task_score']:.3f} "
                  f"vs baseline={best_bl:.3f}  delta={delta:+.3f}{flag}")
    print(f"\nTotal: {len(results)} runs saved to results.db")


# ---------------------------------------------------------------------------
# DELIVERABLES update
# ---------------------------------------------------------------------------

def update_deliverables(results: list, student_label: str, teacher_path: str):
    md_path = os.path.join(_HERE, "DELIVERABLES.md")
    if not os.path.exists(md_path):
        return

    # Build a compact results table
    alpha_rows = [r for r in results if r.get("alpha_mean") is not None]
    be_rows    = [r for r in results if r.get("block_eff") is not None]

    section = f"\n\n---\n## Auto-Generated Results: {student_label} vs {os.path.basename(teacher_path)}\n"
    section += f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n\n"

    if alpha_rows:
        section += "### Alpha (token acceptance rate)\n"
        section += "| run_tag | dataset | T | alpha | CI95 | tok/s | task_score |\n"
        section += "|---|---|---|---|---|---|---|\n"
        for r in alpha_rows:
            task_str = f"{r['task_score']:.3f}" if r.get("task_score") is not None else "-"
            section += (f"| {r['run_tag']} | {r['dataset']} | {r['temperature']} "
                        f"| {r['alpha_mean']:.4f} | ±{r['alpha_ci95']:.4f} "
                        f"| {r.get('throughput', 0):.2f} | {task_str} |\n")

    if be_rows:
        section += "\n### Block Efficiency\n"
        section += "| run_tag | dataset | mode | K | block_eff |\n"
        section += "|---|---|---|---|---|\n"
        for r in be_rows:
            section += (f"| {r['run_tag']} | {r['dataset']} | {r['mode']} "
                        f"| {r['K']} | {r['block_eff']:.4f} |\n")

    ppl_rows = [r for r in results if r.get("perplexity") is not None]
    if ppl_rows:
        section += "\n### Perplexity (quality preservation check)\n"
        section += "| run_tag | dataset | perplexity | vs baseline |\n"
        section += "|---|---|---|---|\n"
        # Find baseline PPL per dataset for delta column
        bl_ppl = {r["dataset"]: r["perplexity"] for r in ppl_rows if r.get("draft_label") == "baseline"}
        for r in ppl_rows:
            ref = bl_ppl.get(r["dataset"])
            delta_str = (f"+{r['perplexity']-ref:.2f} ({'WARN: +>10%' if r['perplexity'] > ref*1.1 else 'ok'})"
                         if ref and r.get("draft_label") != "baseline" else "-")
            section += (f"| {r['run_tag']} | {r['dataset']} "
                        f"| {r['perplexity']:.2f} | {delta_str} |\n")

    with open(md_path, "a", encoding="utf-8") as f:
        f.write(section)
    print(f"\nAppended {len(results)} result rows to DELIVERABLES.md")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="SpecDist full evaluation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--student", required=True,
                   help="Student (draft) model — HF ID or local path")
    p.add_argument("--teacher", required=True,
                   help="Teacher (target) model — HF ID or local path")
    p.add_argument("--student_label", default=None,
                   help="Short label for student (auto-inferred from path if omitted)")
    p.add_argument("--datasets", default=",".join(DEFAULT_DATASETS),
                   help=f"Comma-separated datasets (default: {','.join(DEFAULT_DATASETS)})")
    p.add_argument("--modes", default=",".join(DEFAULT_MODES),
                   help=f"Comma-separated modes (default: {','.join(DEFAULT_MODES)})")
    p.add_argument("--K", default=",".join(map(str, DEFAULT_K)),
                   help="Comma-separated K values (default: 1,3,5)")
    p.add_argument("--temperature", default=",".join(map(str, DEFAULT_TEMPS)),
                   help="Comma-separated temperatures (default: 1.0)")
    p.add_argument("--n", type=int, default=DEFAULT_N,
                   help="Prompts per dataset (default: 30; humaneval/mtbench use all if larger)")
    p.add_argument("--max_tokens", type=int, default=60)
    p.add_argument("--L", type=int, default=5, help="Max draft length for GBV (default: 5)")
    p.add_argument("--full", action="store_true",
                   help="Run full matrix: all datasets × all modes × K=1,3,5")
    p.add_argument("--skip_fetch", action="store_true",
                   help="Skip dataset download step")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip cells that already have a result in results.db (resume)")
    p.add_argument("--no_perplexity", dest="perplexity", action="store_false",
                   help="Skip perplexity measurement (perplexity runs by default as "
                        "a quality-preservation check; use --no_perplexity to skip)")
    p.set_defaults(perplexity=True)
    p.add_argument("--task_score", action="store_true",
                   help="Compute task accuracy: GSM8K exact match, HumanEval pass@1")
    p.add_argument("--task_batch", type=int, default=1,
                   help="Batch size for task_score generation (prompts processed in parallel). "
                        "Set per machine in YAML evaluation.task_batch (a100=32, laptop=4). "
                        "Default=1 is safe everywhere. Override: SPECDIST_TASK_BATCH env var.")
    p.add_argument("--train_steps", type=int, default=0,
                   help="Number of training steps used to produce this checkpoint "
                        "(stored in DB; 0 = baseline / not trained). "
                        "Passed automatically by experiment.py.")
    p.add_argument("--loss_name", default=None,
                   help="Loss function name of the trained model (e.g. kl_tree). "
                        "Logged to W&B config for cross-run comparison. "
                        "Passed automatically by experiment.py.")
    p.add_argument("--lora_rank", type=int, default=None,
                   help="LoRA rank used for this checkpoint (stored in DB for analysis)")
    p.add_argument("--experiment_tag", default=None,
                   help="Free-text label for this experimental run stored in the DB, "
                        "e.g. 'v2 EBE loss clipped accept weight' or 'laptop baseline May-22'. "
                        "All cells in this invocation get the same tag. Used to filter "
                        "and group historical runs in the viz dashboard. "
                        "Each run still has its own unique run_tag timestamp.")
    p.add_argument("--dry_run", action="store_true",
                   help="Print experiment plan without running anything")
    # ── W&B ──────────────────────────────────────────────────────────────────
    p.add_argument("--load_in_4bit", action="store_true",
                   help="Load the teacher (target) model in 4-bit NF4 using bitsandbytes. "
                        "Required for --config colab (free T4, 15 GB VRAM): Qwen3-8B in "
                        "bfloat16 is ~16 GB and OOMs; 4-bit reduces it to ~5 GB. "
                        "Requires: pip install bitsandbytes. "
                        "The student (draft) model is always loaded in bfloat16.")
    p.add_argument("--hw_tier", default="laptop",
                   choices=["laptop", "cpu", "colab_lite", "colab", "a100"],
                   help="Hardware tier tag stored in DB (default: laptop). "
                        "Set by experiment.py from YAML hardware.hw_tier. "
                        "laptop=4GB GPU; cpu=CPU-only server (ATS/AIP, GPT-2 convergence); "
                        "colab=T4 (Kaggle or Colab); a100=A100 (paper results).")
    p.add_argument("--model_family", default="qwen",
                   help="Model family key (qwen/gpt2/llama/gemma).  Forwarded to the "
                        "BE verifier subprocess (runner.py) so it uses the correct "
                        "tree_attn_mask format: Qwen3 expects a {'full_attention': tensor} "
                        "dict; GPT-2/LLaMA/Gemma expect a raw 4D tensor.  Without this, "
                        "the verifier defaults to qwen and GPT-2 crashes with "
                        "\"'dict' object has no attribute 'ndim'\".")
    p.add_argument("--no_wandb", action="store_true",
                   help="Disable W&B logging for this eval run.")
    p.add_argument("--wandb_project", default="distillspec",
                   help="W&B project name for eval logging (default: distillspec). "
                        "Shares the same project as training so you can overlay "
                        "train curves and eval results in one W&B workspace.")
    p.add_argument("--wandb_entity", default=None,
                   help="W&B entity (team or username). Defaults to your logged-in account.")
    p.add_argument("--wandb_group", default=None,
                   help="W&B group for this eval run. Should match the trainer's wandb_group "
                        "(e.g. 'laptop-gsm8k', 'colab-lite') so training and eval runs from "
                        "the same tier appear together in W&B. Forwarded from YAML by "
                        "experiment.py. Falls back to experiment_tag when not set.")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                   help="Compute device (default: auto). '--device cpu' forces CPU "
                        "for alpha + BE eval even on a GPU machine — lets a GPU laptop "
                        "smoke-test the exact CPU path that runs on the ATS/AIP server.")
    global args
    args = p.parse_args()

    # Honour the forced device for all eval paths (alpha preload, perplexity, BE).
    global _FORCED_DEVICE
    if getattr(args, "device", "auto") == "cpu":
        _FORCED_DEVICE = "cpu"

        # Two distinct CPU eval scenarios:
        #
        #   hw_tier=cpu  (server_gpt2.yaml — ATS Cloud, 128 GB RAM, NO GPU)
        #     CPU IS the intended device for the whole research run.
        #     Run full n/max_tokens/L — these produce the actual convergence results.
        #     The server has many CPU cores and fast memory; 128 GB RAM means
        #     gpt2-medium fits in L3 cache after first load (~20-50 s/prompt vs
        #     400-700 s/prompt on a laptop with 16 GB RAM and no ML-tuned BLAS).
        #
        #   hw_tier=laptop/colab/a100  (GPU machine + --device cpu forced)
        #     User is spot-checking the CPU code path on their GPU laptop.
        #     The laptop has 16 GB RAM; gpt2-medium tree attention saturates it
        #     (~400-700 s/prompt).  Cap n/max_tokens/L so the check finishes
        #     in minutes, not hours.
        #
        _hw = getattr(args, "hw_tier", "laptop")
        _is_cpu_server = (_hw == "cpu")

        if _is_cpu_server:
            print("  [device] --device cpu — CPU server (hw_tier=cpu); full eval params kept")
        else:
            print("  [device] --device cpu — forcing CPU for all eval (CPU-path smoke test)")
            # Cap params so the GPU-laptop CPU code-path check finishes in ~15-20 min.
            # Tree attention on a laptop CPU: ~400-700 s/prompt.
            # With cap (n=2, max_tokens=20, L=3): ~2-3 min per mode, ~15 min for 5 modes.
            _CPU_CAP_N          = 2   # prompts (enough to exercise tree variation)
            _CPU_CAP_MAX_TOKENS = 20  # tokens  (each token ~5 s on laptop CPU)
            _CPU_CAP_L          = 3   # draft block depth (shorter tree = fewer nodes)
            _cpu_reduced = []
            if args.n > _CPU_CAP_N:
                _cpu_reduced.append(f"n {args.n}→{_CPU_CAP_N}")
                args.n = _CPU_CAP_N
            if args.max_tokens > _CPU_CAP_MAX_TOKENS:
                _cpu_reduced.append(f"max_tokens {args.max_tokens}→{_CPU_CAP_MAX_TOKENS}")
                args.max_tokens = _CPU_CAP_MAX_TOKENS
            if args.L > _CPU_CAP_L:
                _cpu_reduced.append(f"L {args.L}→{_CPU_CAP_L}")
                args.L = _CPU_CAP_L
            if _cpu_reduced:
                print(f"  [cpu-cap] Reduced for laptop-CPU speed: {', '.join(_cpu_reduced)}")
                print(f"            (code-path check only; research numbers come from GPU)")

    # Default experiment_tag: {hostname}-{YYYYMMDD_HHMM}-v{n}
    # The version number auto-increments per machine per calendar day so
    # successive runs on the same machine are distinguishable in the dashboard.
    # Example: DESKTOP-ABC123-20260522_1302-v1, -v2, -v3 ...
    if args.experiment_tag is None:
        import socket
        _hostname = (socket.gethostname()
                     .lower()
                     .replace(" ", "-")
                     .replace("_", "-")[:24])   # cap length for readability
        _ts       = datetime.now().strftime("%Y%m%d_%H%M")
        _date_pfx = f"{_hostname}-{datetime.now().strftime('%Y%m%d')}"
        _existing = results_db.distinct_values("experiment_tag")
        _n        = sum(1 for t in _existing if t and t.startswith(_date_pfx)) + 1
        args.experiment_tag = f"{_hostname}-{_ts}-v{_n}"

    # --- OSD legacy checkpoint fallback ---
    # If the requested student path is under gbv-research/db/checkpoints/ but doesn't
    # exist there, check OSD/checkpoints/ (where older runs stored their artifacts).
    _ckpt_marker = os.path.join("db", "checkpoints" + os.sep).replace("/", os.sep)
    if (os.sep in args.student and not os.path.exists(args.student)):
        _ckpt_idx = args.student.replace("/", os.sep).find(_ckpt_marker)
        if _ckpt_idx >= 0:
            _rel = args.student.replace("/", os.sep)[_ckpt_idx + len(_ckpt_marker):]
            _osd_path = os.path.join(_OSD_DIR, "checkpoints", _rel)
            if os.path.exists(_osd_path):
                print(f"  [INFO] student path not found at {args.student}")
                print(f"         Falling back to OSD checkpoint: {_osd_path}")
                args.student = _osd_path
    # ----------------------------------------

    # ── Merged-model existence guard ─────────────────────────────────────────
    # Validate the student (draft) path BEFORE spending time on W&B / model
    # loading.  A missing merged model means either:
    #   a) Training hasn't been merged yet → run the merge step first.
    #   b) Training ran on a different model family and left a stale checkpoint
    #      from a different family under the same name (e.g. Qwen kl-gsm8k
    #      blocking GPT-2 kl-gsm8k — now fixed by family-scoped naming).
    #   c) Training hasn't run at all yet.
    # Without this guard, from_pretrained() raises confusing deep errors
    # (IndexError in embedding, shape mismatch, FileNotFoundError) that look
    # like code bugs rather than missing prerequisites.
    # Detect whether the student arg is a LOCAL path (checkpoint dir) vs an
    # HuggingFace model ID (e.g. "Qwen/Qwen3-0.6B", "distilgpt2").
    #
    # HF model IDs also contain "/" (namespace/name), so we cannot use
    # "/"  in args.student — that would flag Qwen/Qwen3-0.6B as local.
    #
    # A local path is identified by:
    #   - absolute path  (/home/... or C:\...)
    #   - explicitly relative path (./... or ../...)
    # HF model IDs never start with / or .
    _student_is_local = (
        args.student.startswith(os.sep) or       # /absolute/path
        args.student.startswith("/") or           # Unix absolute (os.sep=\\ on Windows)
        args.student.startswith(".") or           # ./relative or ../relative
        (len(args.student) > 1 and args.student[1] == ":")  # Windows C:\...
    )
    if _student_is_local and not os.path.isdir(args.student):
        # Try to give a helpful hint about WHICH training step to run
        _student_base = os.path.basename(args.student.rstrip("/\\"))
        # kl-gsm8k_merged-gpt2 → kl-gsm8k-gpt2 (the adapter dir before merge)
        _adapter_hint = _student_base.replace("_merged", "")
        print(
            f"\n  [FATAL] Student model not found: {args.student}\n"
            f"\n  The merged checkpoint '{_student_base}' does not exist.\n"
            f"  This means one of:\n"
            f"    (a) Training + merge haven't run yet for this loss / model family.\n"
            f"        Run:  python orchestration/experiment.py --config <config> --losses "
            f"{_adapter_hint.split('-')[0]} --yes\n"
            f"    (b) The checkpoint was built for a different model family and the old\n"
            f"        name clashed (fixed by the family-suffix naming scheme).\n"
            f"        Run:  python orchestration/clean_restart.py --config <config> --yes\n"
            f"    (c) The checkpoint was wiped by a previous clean_restart.\n"
            f"        Re-run training.\n",
            file=sys.stderr,
        )
        sys.exit(2)

    student_label = args.student_label or infer_label(args.student)
    print(f"\nSpecDist Evaluation Pipeline")
    print(f"  Student : {args.student}  [{student_label}]")
    print(f"  Teacher : {args.teacher}")
    print(f"  GPU     : {_gpu_info()}")
    if args.experiment_tag:
        print(f"  Tag     : {args.experiment_tag}")

    # ── W&B eval run (optional — silently skips if not installed / --no_wandb)
    # What gets logged per eval cell:
    #   eval/alpha_mean, eval/alpha_ci95, eval/throughput (alpha mode)
    #   eval/block_eff                                    (BE modes: specinfer, gbv, traversal)
    #   eval/task_score, eval/perplexity                  (when measured)
    #   per-prompt alpha table (wandb.Table)              (alpha mode)
    # Each cell is one wandb.log() call; dataset/mode/K are stored as dimensions.
    # ─────────────────────────────────────────────────────────────────────────
    if not getattr(args, "no_wandb", False):
        try:
            import wandb as _wandb_mod
            _wandb_dir = os.path.join(_PARENT, "db", "wandb")
            os.makedirs(_wandb_dir, exist_ok=True)
            _run_mode = "smoke" if getattr(args, "n", 10) <= 3 else "full"
            _wandb_mod.init(
                project=args.wandb_project,
                entity=getattr(args, "wandb_entity", None),
                name=f"eval_{student_label}_{args.hw_tier}_{_now_tag()}",
                job_type="eval",
                # Group matches trainer's wandb_group (e.g. "laptop-gsm8k", "colab-lite")
                # so training + eval runs from the same tier are in the same W&B group.
                # Falls back to experiment_tag for backward compat when group not set.
                group=getattr(args, "wandb_group", None) or args.experiment_tag,
                tags=[student_label, "eval", "phase1", args.hw_tier,
                      args.experiment_tag,
                      _run_mode],   # "smoke" or "full" for easy filtering
                dir=_wandb_dir,   # store run files under db/wandb/, not orchestration/wandb/
                config={
                    "student":        args.student,
                    "student_label":  student_label,
                    "teacher":        args.teacher,
                    "experiment_tag": args.experiment_tag,
                    "gpu":            _gpu_info(),
                    "n_prompts":      args.n,
                    "max_tokens":     args.max_tokens,
                    "hw_tier":        args.hw_tier,
                    "run_mode":       _run_mode,
                    "train_steps":    getattr(args, "train_steps", None),
                    "loss_name":      getattr(args, "loss_name", None),
                },
                reinit="return_previous",
            )
            print(f"  [wandb] Eval run: {_wandb_mod.run.url}")
        except Exception as _we:
            print(f"  [wandb] Skipped ({type(_we).__name__}: {_we})")

    if args.full:
        datasets = DEFAULT_DATASETS
        modes    = DEFAULT_MODES
        Ks       = DEFAULT_K
        temps    = DEFAULT_TEMPS
    else:
        datasets = args.datasets.split(",")
        modes    = args.modes.split(",")
        Ks       = [int(k) for k in args.K.split(",")]
        temps    = [float(t) for t in args.temperature.split(",")]

    # Fetch datasets (skip if already present)
    if not args.skip_fetch:
        print("\nFetching datasets (skip if present)...")
        fetch_all(n=args.n, datasets=datasets)

    # Plan
    cells = [(ds, mode, K, T)
             for ds in datasets
             for mode in modes
             for K in Ks
             for T in temps]

    # For alpha mode K doesn't vary the experiment meaningfully — deduplicate
    seen, deduped = set(), []
    for ds, mode, K, T in cells:
        key = (ds, mode, K, T) if mode != "alpha" else (ds, mode, 1, T)
        if key not in seen:
            seen.add(key)
            deduped.append((ds, mode, K if mode != "alpha" else 1, T))
    cells = deduped

    print(f"\nPlan: {len(cells)} experiment cells  "
          f"({len(datasets)} datasets × {len(modes)} modes × {len(Ks)} K-values)")
    if args.dry_run:
        for ds, mode, K, T in cells:
            print(f"  {make_run_tag(student_label, mode, ds, K)}")
        return

    # Ensure DB schema exists
    results_db._connect().close()

    # Perplexity pass (once per dataset, model-pair independent of modes)
    ppl_results = {}
    if args.perplexity:
        print("\n--- Measuring perplexity ---")
        for ds in datasets:
            n = args.n
            if ds == "humaneval": n = 164
            elif ds == "mtbench": n = 80
            # Skip if already measured and --skip_existing is set
            if args.skip_existing and _already_run(student_label, ds, "perplexity", 0, 0.0,
                                                     student_path=args.student, n_prompts=n):
                print(f"  {student_label} | {ds} | PPL=SKIP (already in DB)")
                continue
            try:
                prompts = load_prompts(ds, n)
                ppl_res = measure_perplexity(args.student, prompts)
                ppl_results[ds] = ppl_res["perplexity"]
                print(f"  {student_label} | {ds} | PPL={ppl_res['perplexity']:.2f}")
                # Save perplexity in its own column (+ human-readable notes)
                ppl_row = dict(
                    run_tag=make_run_tag(student_label, "perplexity", ds, 0),
                    draft_label=student_label, draft_path=args.student,
                    target_path=args.teacher, loss_name=infer_loss_name(student_label),
                    train_steps=args.train_steps, learning_rate=0.0,
                    lora_rank=args.lora_rank or 0,
                    dataset=ds, n_prompts=len(prompts), mode="perplexity",
                    K=0, L=0, temperature=0.0,
                    perplexity=ppl_res["perplexity"],
                    notes=f"perplexity={ppl_res['perplexity']:.4f}",
                )
                results_db.insert_run(ppl_row, hw_tier=args.hw_tier)
                # Quality guard: warn if this model's PPL is much worse than the
                # baseline FOR THE SAME MODEL PAIR (same draft_path + target_path).
                # Filter by draft_path to avoid cross-family false alarms, e.g.
                # GPT-2 PPL=51 flagged against Qwen baseline PPL=15.
                import math as _math
                _baseline_ppls = [
                    r["perplexity"] for r in results_db.query_runs(
                        {"draft_label": "baseline", "dataset": ds, "mode": "perplexity"})
                    if r.get("perplexity") is not None
                    and r.get("draft_path") == args.student   # same model family only
                ]
                if _baseline_ppls:
                    _ref_ppl = min(_baseline_ppls)
                    if ppl_res["perplexity"] > _ref_ppl * 1.10:
                        _warn(f"PPL={ppl_res['perplexity']:.2f} is >10% worse than "
                              f"baseline PPL={_ref_ppl:.2f} on {ds} — "
                              f"possible training collapse or tokenizer mismatch.")
            except Exception as e:
                print(f"  [WARN] perplexity failed for {ds}: {e}")

    # ── Note on evaluation ordering ──────────────────────────────────────────
    # BE batch runs BEFORE alpha pre-load so both never compete for VRAM
    # simultaneously.  The BE subprocess exits (freeing GPU memory) before alpha
    # models are loaded into the main process.  On a 6 GB laptop GPU this saves
    # ~2.4 GB peak concurrent usage (student+teacher loaded twice = 4.8 GB) and
    # avoids the BE subprocess OOM-retrying on CPU due to the alpha pre-load
    # already holding half the VRAM.  The sequential cell loop uses _be_cache and
    # _alpha_preloaded independently, so the swap is safe.
    # ── Pre-load models for alpha evaluation (shared across all datasets) ───
    # Without this, run_alpha() reloads both models from disk for every
    # (dataset × temperature) cell — N_alpha_cells cold-starts.
    # On a T4 with 8B models each cold-start costs ~60 s; keeping them resident
    # cuts that to one load total (saves N_cells-1 loads).
    # NOTE: pre-load happens AFTER BE batch (see block below) so alpha and BE
    # never compete for VRAM at the same time.
    _alpha_preloaded = None
    _has_alpha = any(mode == "alpha" for _, mode, _, _ in cells)

    # ── Pre-batch all GBV / BE evaluations (one subprocess per dataset) ────
    # Runs BEFORE alpha pre-load so BE subprocess and alpha models never compete
    # for VRAM simultaneously.  On a 6 GB laptop, having both loaded at once
    # costs ~4.8 GB (student+teacher×2) and can force the BE subprocess to retry
    # on CPU.  By running BE first (VRAM clean), both can use the full GPU budget.
    # Avoids N×72 separate subprocess launches each reloading both models.
    # Instead: one GBV process per dataset loads models once, runs all
    # (mode, K, T) combos for that dataset, then exits cleanly before alpha loads.
    be_cells_to_run: set = set()
    for ds, mode, K, T in cells:
        if mode != "alpha":
            if not (args.skip_existing and _already_run(student_label, ds, mode, K, T,
                                                         student_path=args.student, n_prompts=args.n)):
                be_cells_to_run.add((ds, mode, K, T))

    _be_cache: dict = {}   # (ds, mode, K, T) -> block_eff
    if be_cells_to_run:
        be_by_ds: dict = {}
        for (ds, mode, K, T) in be_cells_to_run:
            entry = be_by_ds.setdefault(ds, {"modes": set(), "Ks": set(), "Ts": set()})
            entry["modes"].add(mode)
            entry["Ks"].add(K)
            entry["Ts"].add(T)

        n_old = len(be_cells_to_run)
        n_new = len(be_by_ds)
        print(f"\n-- Pre-batching {n_old} BE cell(s) -> {n_new} subprocess(es) "
              f"(was {n_old} before batching) --")
        for ds, items in be_by_ds.items():
            modes_list = sorted(items["modes"])
            Ks_list    = sorted(items["Ks"])
            Ts_list    = sorted(items["Ts"])
            # Cap at the benchmark's full size but always respect --n.
            # humaneval=164, mtbench=80 are the full benchmark sizes.
            # With --n 5 (laptop), use 5 prompts — enough to exercise all code paths.
            _ds_max = {"humaneval": 164, "mtbench": 80}
            n_ds = min(args.n, _ds_max.get(ds, args.n)) if args.n else _ds_max.get(ds, 10)
            data_path = get_dataset_path(ds, n_ds)
            n_c = len(modes_list) * len(Ks_list) * len(Ts_list)
            print(f"  {ds}: {len(modes_list)} mode(s) × {len(Ks_list)} K × "
                  f"{len(Ts_list)} temp(s) = {n_c} combo(s)")
            # VRAM strategy for the BE subprocess on laptop (4 GB GPU):
            # Running all modes in ONE subprocess causes allocator fragmentation /
            # KV-cache residue that hard-aborts the GPU driver (0xC000013A native
            # crash — not catchable by PyTorch OOM handler).
            # Previous fix: whole batch on CPU → correct but 29× slower.
            # Better fix: one GPU subprocess PER MODE.  Each subprocess exits
            # cleanly, releasing all VRAM and cache before the next mode starts.
            # This matches pre-batching speed (~13 s/prompt on GPU) with no crash.
            _hw = getattr(args, "hw_tier", "laptop")
            if _FORCED_DEVICE == "cpu" or _hw == "cpu":
                # CPU batch path: either forced via --device cpu (laptop smoke test)
                # or hw_tier=cpu (ATS/AIP CPU-only server where no GPU exists).
                # Runs all modes in ONE CPU subprocess — no VRAM isolation needed.
                _reason = "--device cpu (CPU-path smoke test)" if _FORCED_DEVICE == "cpu" \
                          else "hw_tier=cpu (CPU-only server)"
                print(f"  [BE batch] {_reason} -> running BE on CPU")
                batch_res = run_be_batch(
                    args.student, args.teacher, data_path,
                    modes_list, Ks_list, Ts_list,
                    args.L, args.max_tokens,
                    _device="cpu",
                    load_in_4bit=False,   # 4-bit requires CUDA; CPU uses full precision
                )
                for (m, k, t), be in batch_res.items():
                    _be_cache[(ds, m, k, t)] = be
            elif _hw == "laptop":
                # Run one GPU subprocess per mode so each exits cleanly (releasing
                # VRAM) before the next starts — prevents allocator fragmentation
                # that hard-crashes the GPU driver (0xC000013A) mid-batch.
                # If a mode's GPU subprocess produces no results (OOM or crash),
                # retry that specific mode on CPU so the pipeline still completes.
                print(f"  [BE batch] hw_tier=laptop -> one GPU subprocess per mode "
                      f"(serial isolation; VRAM-wait between modes; CPU fallback on VRAM exhaustion)")
                # Local helper: poll until enough VRAM is free (or give up after
                # `timeout_s`).  On Windows the GPU driver takes 1-3 s to reclaim
                # VRAM after a subprocess exits; launching the next one too early
                # causes a native 0xC000013A abort.  2700 MB headroom matches the
                # validated alpha floor for the 0.5B+0.6B pair (2000 was too tight —
                # naive still crashed deep in the pipeline once fragmentation built up).
                def _wait_for_vram(mode_name, need_mb=2700, timeout_s=30):
                    try:
                        import torch as _t, time as _time
                        if not _t.cuda.is_available():
                            return
                        for _ in range(timeout_s):
                            _t.cuda.empty_cache()
                            if _t.cuda.mem_get_info(0)[0] // 1024**2 >= need_mb:
                                return
                            _time.sleep(1)
                        _free = _t.cuda.mem_get_info(0)[0] // 1024**2
                        print(f"  [BE batch] mode={mode_name}: VRAM still low "
                              f"({_free} MB < {need_mb}) after {timeout_s}s — trying anyway")
                    except Exception:
                        pass

                for _mode in modes_list:
                    _wait_for_vram(_mode)
                    _mode_res = run_be_batch(
                        args.student, args.teacher, data_path,
                        [_mode], Ks_list, Ts_list,
                        args.L, args.max_tokens,
                        _device="cuda",
                        load_in_4bit=getattr(args, "load_in_4bit", False),
                    )
                    if not _mode_res:
                        # GPU subprocess produced no results — almost always a
                        # transient native VRAM abort (0xC000013A) from fragmentation
                        # accumulated deep in the pipeline.  Before the 30x-slower CPU
                        # path, give the GPU ONE more chance with fully-cleared VRAM
                        # and a longer settle — this recovers most transient crashes
                        # at ~6 s/prompt instead of ~340 s/prompt.
                        print(f"  [BE batch] mode={_mode}: GPU subprocess failed "
                              f"-> clearing VRAM and retrying ONCE on GPU")
                        _wait_for_vram(_mode, need_mb=2700, timeout_s=20)
                        _mode_res = run_be_batch(
                            args.student, args.teacher, data_path,
                            [_mode], Ks_list, Ts_list,
                            args.L, args.max_tokens,
                            _device="cuda",
                            load_in_4bit=getattr(args, "load_in_4bit", False),
                        )
                    if not _mode_res:
                        # GPU retry also failed — fall back to CPU (slow but correct).
                        print(f"  [BE batch] mode={_mode}: GPU retry failed "
                              f"-> retrying on CPU (only this mode, ~30x slower)")
                        _mode_res = run_be_batch(
                            args.student, args.teacher, data_path,
                            [_mode], Ks_list, Ts_list,
                            args.L, args.max_tokens,
                            _device="cpu",
                            load_in_4bit=getattr(args, "load_in_4bit", False),
                        )
                    for (m, k, t), be in _mode_res.items():
                        _be_cache[(ds, m, k, t)] = be
            else:
                batch_res = run_be_batch(
                    args.student, args.teacher, data_path,
                    modes_list, Ks_list, Ts_list,
                    args.L, args.max_tokens,
                    _device="cuda",
                    load_in_4bit=getattr(args, "load_in_4bit", False),
                )
                for (m, k, t), be in batch_res.items():
                    _be_cache[(ds, m, k, t)] = be
        print(f"  BE pre-batch done: {len(_be_cache)} result(s) cached.\n")

    # ── Pre-load models for alpha evaluation (shared across all datasets) ───
    # Runs AFTER BE batch so both never compete for VRAM simultaneously.
    # Without pre-loading, run_alpha() reloads both models from disk for every
    # (dataset × temperature) cell — N_alpha_cells cold-starts.
    # On a T4 with 8B models each cold-start costs ~60 s; keeping them resident
    # cuts that to one load total (saves N_cells-1 loads).
    if _has_alpha:
        import torch as _torch
        from transformers import AutoTokenizer as _ATok, AutoModelForCausalLM as _AMLM
        # dtype-serialization patch already applied at module load time
        # (see _patch_config_json_serialization() at top of this file)
        _patch_config_json_serialization()   # no-op if already applied

        # Wait for VRAM to be released by the last BE subprocess before checking
        # the free VRAM for the alpha guard.  The BE per-mode subprocesses exit
        # cleanly, but on Windows the GPU driver can take 1-3 s to reclaim their
        # VRAM.  Without this wait the alpha guard sees stale low-VRAM and falls
        # back to CPU, then hits a device mismatch because specInfer creates
        # CUDA tensors regardless of the model's device.  The same settle logic
        # is used between BE per-mode subprocesses via _wait_for_vram().
        if _torch.cuda.is_available():
            _alpha_floor_mb = (_ALPHA_VRAM_FLOOR_MB_LAPTOP
                               if getattr(args, "hw_tier", "laptop") == "laptop"
                               else _ALPHA_VRAM_FLOOR_MB)
            import time as _time
            for _settle_attempt in range(20):   # wait up to 20 s
                _torch.cuda.empty_cache()
                if _torch.cuda.mem_get_info(0)[0] // 1024**2 >= _alpha_floor_mb:
                    break
                _time.sleep(1)
            # No warning here — _alpha_device_guard will print the decision.

        _device, _dtype = _pick_device()
        # PROACTIVE VRAM guard: never attempt the dual-model GPU load when it
        # would hard-crash the process (Windows 0xC000013A) before OOM can raise.
        # On laptop this forces CPU; on other tiers it enforces a free-VRAM floor.
        _device, _dtype = _alpha_device_guard(_device, _dtype)
        _load_4bit_pre = getattr(args, "load_in_4bit", False)
        print(f"\n-- Pre-loading models for alpha eval (1 load shared across all datasets) --")
        _tok = _ATok.from_pretrained(args.teacher, use_fast=False)
        if _tok.pad_token_id is None:
            _tok.pad_token = _tok.eos_token
        _same_models = (args.student == args.teacher)
        _s_model = None
        for _attempt in range(2):
            try:
                _s_model = _AMLM.from_pretrained(
                    args.student, **_dtype_kwargs(_dtype), low_cpu_mem_usage=True,
                    attn_implementation=_ATTN_IMPL,
                ).to(_device).eval()
                if _same_models:
                    _t_model = _s_model
                elif _load_4bit_pre and _device == "cuda":
                    # load_in_4bit=True: teacher in 4-bit NF4 so it fits T4 (15 GB).
                    # device_map={"": "cuda:0"} avoids accelerate's two-pass device-map
                    # computation, putting all shards directly on the GPU — prevents
                    # the ~16 GB CPU-RAM spike that caused OOM at 56% tensor load.
                    from transformers import BitsAndBytesConfig as _BnBPre
                    _bnb_pre = _BnBPre(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                       bnb_4bit_compute_dtype=_torch.bfloat16,
                                       bnb_4bit_use_double_quant=True)
                    _t_model = _AMLM.from_pretrained(
                        args.teacher, quantization_config=_bnb_pre,
                        device_map={"": "cuda:0"}, low_cpu_mem_usage=True,
                        attn_implementation=_ATTN_IMPL,
                    ).eval()
                    print(f"  [alpha preload] Teacher loaded in 4-bit NF4 on cuda:0")
                else:
                    _t_model = _AMLM.from_pretrained(
                        args.teacher, **_dtype_kwargs(_dtype), low_cpu_mem_usage=True,
                        attn_implementation=_ATTN_IMPL,
                    ).to(_device).eval()
                break
            except _torch.cuda.OutOfMemoryError:
                if _device == "cpu":
                    raise
                _free = _torch.cuda.mem_get_info(0)[0] // 1024**2
                print(f"  [OOM] {_free} MB free — retrying pre-load on CPU")
                _torch.cuda.empty_cache()
                if _s_model is not None:
                    del _s_model
                    _s_model = None
                _device, _dtype = "cpu", _torch.float32
        _vram = (_torch.cuda.memory_allocated() / 1024**2
                 if _device == "cuda" else 0.0)
        print(f"  Models resident. VRAM: {_vram:.0f} MB  "
              f"(kept until all alpha cells finish)")
        _alpha_preloaded = (_device, _dtype, _tok, _s_model, _t_model, _same_models)

    # ── Run sequentially ─────────────────────────────────────────────────────
    all_results = []
    failed_cells = []
    t_start = time.perf_counter()
    for i, (ds, mode, K, T) in enumerate(cells, 1):
        print(f"\n[{i}/{len(cells)}]", end="")
        # Humaneval and mtbench: use all prompts (164 / 80)
        n = args.n
        if ds == "humaneval": n = 164
        elif ds == "mtbench": n = 80

        if mode == "alpha":
            try:
                result = run_cell(
                    student_path=args.student,
                    teacher_path=args.teacher,
                    student_label=student_label,
                    dataset=ds, mode=mode, K=K, L=args.L,
                    temperature=T, n=n, max_tokens=args.max_tokens,
                    skip_existing=args.skip_existing,
                    task_score=args.task_score,
                    experiment_tag=args.experiment_tag,
                    preloaded=_alpha_preloaded,   # models stay resident
                )
                if result and not result.get("skipped"):
                    all_results.append(result)
            except Exception as e:
                print(f"  [ERROR] {e}")
                failed_cells.append((ds, mode, K, T, str(e)))
        else:
            # BE result already computed in pre-batch step above
            run_tag = make_run_tag(student_label, mode, ds, K, T)
            if args.skip_existing and _already_run(student_label, ds, mode, K, T,
                                                     student_path=args.student, n_prompts=args.n):
                print(f" [{run_tag}] SKIP (already in DB)")
                continue
            be_val = _be_cache.get((ds, mode, K, T))
            if be_val is None:
                print(f" [{run_tag}] WARN: no cached BE result (batch may have failed)")
                failed_cells.append((ds, mode, K, T, "no cached BE result"))
                continue
            row = dict(
                run_tag=run_tag,
                draft_label=student_label,
                draft_path=args.student,
                target_path=args.teacher,
                loss_name=infer_loss_name(student_label),
                train_steps=args.train_steps, learning_rate=0.0, lora_rank=0,
                dataset=ds, n_prompts=n,
                mode=mode, K=K, L=args.L, temperature=T,
                block_eff=be_val,
                experiment_tag=args.experiment_tag,
            )
            run_id = results_db.insert_run(row, hw_tier=args.hw_tier)
            row["id"] = run_id
            print(f" [{run_tag}] block_eff={be_val:.4f}")
            all_results.append(row)

            # BE results are aggregated into a single eval_summary_table at end-of-run
            # (see main() W&B finish block below).  Logging per-cell via wandb.log()
            # would create separate tiny panels per verifier mode — not useful.

    # Release pre-loaded alpha models now that all cells are done
    if _alpha_preloaded is not None:
        import torch as _torch
        _dev, _, _, _sm, _tm, _same_flag = _alpha_preloaded
        del _sm
        if not _same_flag:
            del _tm
        if _dev == "cuda":
            _torch.cuda.empty_cache()
        print("  Alpha models released.")

    elapsed = time.perf_counter() - t_start
    n_done = len(all_results)
    print(f"\nCompleted {n_done}/{len(cells)} cells in {elapsed/60:.1f} min"
          + (f" ({elapsed/n_done:.0f}s avg)" if n_done else ""))

    print_summary(all_results)
    update_deliverables(all_results, student_label, args.teacher)

    # ── End-of-run quality warning summary ───────────────────────────────────
    # Collected by _warn() throughout the run; printed prominently so nothing
    # is missed in the terminal scroll-back.
    if _run_warnings:
        print("\n" + "!" * 70)
        print("  QUALITY WARNINGS — review before treating results as final:")
        for i, w in enumerate(_run_warnings, 1):
            print(f"  [{i}] {w}")
        print("!" * 70)
    else:
        print("\n  All quality checks passed (no PPL / task_score divergence warnings).")

    print(f"\nDashboard: python viz_server.py  ->  http://localhost:5000/\n")

    # W&B: close eval run cleanly with full summary
    try:
        import wandb as _wmod_final
        if _wmod_final.run is not None:
            _all_alphas = [r.get("alpha_mean") for r in all_results if r.get("alpha_mean") is not None]
            _all_bes    = [r.get("block_eff")  for r in all_results if r.get("block_eff")  is not None]
            if _all_alphas:
                _wmod_final.summary["mean_alpha"] = sum(_all_alphas) / len(_all_alphas)
                _wmod_final.summary["max_alpha"]  = max(_all_alphas)
            if _all_bes:
                _wmod_final.summary["mean_block_eff"] = sum(_all_bes) / len(_all_bes)
                _wmod_final.summary["max_block_eff"]  = max(_all_bes)

            # Per-verifier averages (e.g. "BE/bv", "BE/gbv", "alpha/gsm8k") ──────
            # These appear as columns in the W&B runs-comparison table — useful for
            # ranking losses by best verifier BE or by GSM8K alpha across many runs.
            # Per-dataset-mode detail lives in eval_summary_table (keep panel count ≤15).
            from collections import defaultdict as _dd
            _be_by_mode: dict = _dd(list)
            _alpha_by_ds: dict = _dd(list)
            for r in all_results:
                if r.get("block_eff") is not None:
                    _be_by_mode[r["mode"]].append(r["block_eff"])
                if r.get("alpha_mean") is not None and r.get("dataset"):
                    _alpha_by_ds[r["dataset"]].append(r["alpha_mean"])
            for _mode, _vals in _be_by_mode.items():
                _wmod_final.summary[f"BE/{_mode}"] = round(sum(_vals) / len(_vals), 4)
            for _ds, _vals in _alpha_by_ds.items():
                _wmod_final.summary[f"alpha/{_ds}"] = round(sum(_vals) / len(_vals), 4)

            # ── Pivot table: dataset × mode → all metrics side-by-side ──────────
            # This is the PRIMARY eval view in W&B.  Open the run → Artifacts tab
            # → eval_summary_table to get a sortable/filterable table.
            # Dimensions: dataset, mode (verifier), K, temperature
            # Values: block_eff, alpha_mean, alpha_ci95, throughput (tok/s),
            #         ms_per_tok, task_score, perplexity
            # Use W&B table "Group by" → dataset or mode for pivot-style views.
            # Temperature and K are fixed per run — use them as filters, not axes.
            def _round(v, d=4):
                try: return round(float(v), d)
                except: return None
            # loss_name: the training objective this model was distilled with.
            # "baseline" when no training was done (reference run).
            # This is the primary key for cross-run comparison — without it the
            # table just shows verifier metrics with no way to tell which trained
            # model produced them (the run name is not visible inside the table).
            _loss_col = getattr(args, "loss_name", None) or "baseline"
            _tbl_rows = [
                [
                    _loss_col,
                    r.get("dataset", ""),
                    r.get("mode", ""),
                    r.get("K", 1),
                    r.get("temperature", 1.0),
                    _round(r.get("block_eff")),
                    _round(r.get("alpha_mean")),
                    _round(r.get("alpha_ci95")),
                    _round(r.get("throughput")),
                    _round(r.get("ms_per_tok")),
                    _round(r.get("task_score")),
                    _round(r.get("perplexity")),
                ]
                for r in all_results
            ]
            if _tbl_rows:
                _eval_tbl = _wmod_final.Table(
                    columns=["loss_name",
                             "dataset", "mode", "K", "temperature",
                             "block_eff", "alpha_mean", "alpha_ci95",
                             "throughput_tok_s", "ms_per_tok",
                             "task_score", "perplexity"],
                    data=_tbl_rows,
                )
                _wmod_final.log({"eval_summary_table": _eval_tbl})
            if _all_bes:
                _wmod_final.summary["summary/best_BE"] = max(_all_bes)

            _wmod_final.finish()
    except Exception:
        pass

    if failed_cells:
        print("\n" + "!" * 70)
        print(f"  EVAL FAILED — {len(failed_cells)} requested cell(s) produced no result.")
        print("  This usually means the pre-batched BE subprocess failed or OOMed.")
        print("  The pipeline step will remain retryable instead of being marked done.")
        for ds, mode, K, T, reason in failed_cells[:20]:
            print(f"  - {student_label} | {ds} | {mode} | K={K} | T={T}: {reason}")
        if len(failed_cells) > 20:
            print(f"  ... {len(failed_cells) - 20} more")
        print("!" * 70)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
