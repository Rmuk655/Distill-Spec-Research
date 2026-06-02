#!/usr/bin/env python3
"""
02_verify.py — Verify that ATS setup completed correctly.
Run after 01_setup.sh and after any environment change.

Usage:
    python deploy/ats/02_verify.py

Exit 0 = all checks passed. Exit 1 = one or more checks failed.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
WARN = "\033[93m!\033[0m"

checks_failed = 0

def ok(msg):      print(f"  {PASS}  {msg}")
def fail(msg):    global checks_failed; checks_failed += 1; print(f"  {FAIL}  {msg}")
def warn(msg):    print(f"  {WARN}  {msg}")
def section(msg): print(f"\n{msg}")


# ── Python version ─────────────────────────────────────────────────────────────
section("Python")
import platform
v = sys.version_info
if v >= (3, 10):
    ok(f"Python {v.major}.{v.minor}.{v.micro}")
else:
    fail(f"Python {v.major}.{v.minor} — need ≥ 3.10")


# ── Core imports ───────────────────────────────────────────────────────────────
section("Core imports")
try:
    import torch
    ok(f"torch {torch.__version__}")
except ImportError:
    fail("torch not installed — run: pip install -r requirements.txt")

try:
    import transformers
    ok(f"transformers {transformers.__version__}")
except ImportError:
    fail("transformers not installed")

try:
    import peft
    ok(f"peft {peft.__version__}")
except ImportError:
    fail("peft not installed")

try:
    import wandb
    ok(f"wandb {wandb.__version__}")
except ImportError:
    warn("wandb not installed — W&B logging will be disabled")

try:
    import flask
    ok(f"flask {flask.__version__} (dashboard)")
except ImportError:
    warn("flask not installed — viz server won't start")


# ── CPU / RAM ──────────────────────────────────────────────────────────────────
section("Hardware")
import torch
n_threads = torch.get_num_threads()
if n_threads >= 8:
    ok(f"CPU threads: {n_threads}")
else:
    warn(f"CPU threads: {n_threads} (low — set OMP_NUM_THREADS=<ncores>)")

try:
    import psutil
    ram_gb = psutil.virtual_memory().total / 1024**3
    if ram_gb >= 64:
        ok(f"RAM: {ram_gb:.0f} GB")
    else:
        warn(f"RAM: {ram_gb:.0f} GB (low for full parallel runs)")
except ImportError:
    warn("psutil not installed — cannot check RAM")

cuda = torch.cuda.is_available()
if not cuda:
    ok("No GPU (expected for ATS CPU-only server)")
else:
    warn(f"GPU detected: {torch.cuda.get_device_name(0)} — set CUDA_VISIBLE_DEVICES='' for CPU-only")


# ── Model family registry ──────────────────────────────────────────────────────
section("Model families")
try:
    from core.model_families import FAMILY_REGISTRY, get_family
    expected = {"gpt2", "llama", "qwen", "gemma"}
    registered = set(FAMILY_REGISTRY.keys())
    if expected <= registered:
        ok(f"Registry: {sorted(registered)}")
    else:
        fail(f"Missing families: {expected - registered}")

    gpt2 = get_family("gpt2")
    ok(f"gpt2: draft={gpt2.default_draft_model_id}  target={gpt2.default_target_model_id}")
    ok(f"gpt2: LoRA targets={gpt2.lora_target_modules()}")
except Exception as e:
    fail(f"model_families error: {e}")


# ── Model weights cached ───────────────────────────────────────────────────────
section("Model weights (HF cache)")
try:
    from huggingface_hub import try_to_load_from_cache
    for mid in ["distilgpt2", "gpt2-medium"]:
        cached = try_to_load_from_cache(repo_id=mid, filename="config.json")
        if cached:
            ok(f"{mid}: cached")
        else:
            fail(f"{mid}: NOT cached — run 01_setup.sh")
except ImportError:
    warn("huggingface_hub not available — cannot verify cache")


# ── Datasets ───────────────────────────────────────────────────────────────────
section("Datasets")
datasets = {
    "gsm8k_train.jsonl":  "Training data (7473 prompts)",
    "gsm8k_5.jsonl":      "Eval data (5 prompts)",
    "gsm8k_30.jsonl":     "Smoke test data",
}
ds_root = os.path.join(ROOT, "core", "datasets", "raw")
for fname, desc in datasets.items():
    path = os.path.join(ds_root, fname)
    if os.path.exists(path):
        size = os.path.getsize(path)
        ok(f"{fname}: {size:,} bytes  ({desc})")
    else:
        fail(f"{fname} NOT FOUND — run: python core/datasets/downloader.py")


# ── W&B ────────────────────────────────────────────────────────────────────────
section("W&B")
api_key = os.environ.get("WANDB_API_KEY", "")
if not api_key or api_key == "PASTE_YOUR_WANDB_KEY_HERE":
    warn("WANDB_API_KEY not set — runs will log offline. Edit deploy/ats/00_env.sh")
else:
    ok(f"WANDB_API_KEY set (entity={os.environ.get('WANDB_ENTITY','?')}  "
       f"project={os.environ.get('WANDB_PROJECT','?')})")
    try:
        import wandb
        wandb.login(key=api_key, relogin=False, verify=True)
        ok("W&B login verified")
    except Exception as e:
        warn(f"W&B login check: {e}")


# ── Unit tests (quick) ─────────────────────────────────────────────────────────
section("Unit tests (model families)")
import subprocess
r = subprocess.run(
    [sys.executable, "-m", "pytest",
     os.path.join(ROOT, "tests", "test_model_families.py"),
     "-q", "--tb=no"],
    capture_output=True, text=True, cwd=ROOT,
)
if r.returncode == 0:
    lines = [l for l in r.stdout.splitlines() if "passed" in l or "failed" in l]
    ok(f"test_model_families: {lines[-1] if lines else 'passed'}")
else:
    fail(f"test_model_families FAILED:\n{r.stdout[-500:]}")


# ── Summary ────────────────────────────────────────────────────────────────────
print()
print("=" * 50)
if checks_failed == 0:
    print(f"  {PASS}  All checks passed. Ready for training.")
    print()
    print("  Next: python deploy/ats/03_smoke.py")
else:
    print(f"  {FAIL}  {checks_failed} check(s) failed. Fix above, re-run 01_setup.sh.")
print("=" * 50)
sys.exit(0 if checks_failed == 0 else 1)
