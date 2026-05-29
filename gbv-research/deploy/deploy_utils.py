"""
deploy_utils.py — shared helpers for all SpecDist notebook environments.

Works on Google Colab (T4 / A100), Kaggle Kernels, and any Jupyter session.
Loaded after git clone, so it is NOT available during the initial clone step.
The bootstrap cell inlines a minimal _prereq_token() + clone block, then does:
    sys.path.insert(0, f"{GBV_DIR}/deploy")
    from deploy_utils import bootstrap, run_pipeline
and delegates everything else here.
"""

import datetime
import glob
import json
import os
import subprocess
import sys
import threading
import time

# ---------------------------------------------------------------------------
# Default paths (ephemeral container)
# ---------------------------------------------------------------------------
REPO_URL   = "https://github.com/Rmuk655/Distill-Spec-Research.git"
REPO_DIR   = "/content/Distill-Spec-Research"
GBV_DIR    = f"{REPO_DIR}/gbv-research"
DRIVE_ROOT = "/content/drive/MyDrive/specdist"


# ---------------------------------------------------------------------------
# Secrets
# ---------------------------------------------------------------------------

def get_secret(name: str) -> str:
    """Read from Colab Secrets, then Kaggle Secrets, then environment variables."""
    try:
        from google.colab import userdata  # type: ignore
        v = userdata.get(name)
        if v:
            return v
    except Exception:
        pass
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore
        v = UserSecretsClient().get_secret(name)
        if v:
            return v
    except Exception:
        pass
    return os.environ.get(name, "")


def _auth_repo_url(base_url: str) -> str:
    """Inject GITHUB_TOKEN into a clone URL for private repos."""
    tok = get_secret("GITHUB_TOKEN")
    if not tok:
        print("⚠ GITHUB_TOKEN not set — clone will fail for a private repo.")
        print("  Add via: left sidebar → 🔑 Secrets → GITHUB_TOKEN")
        print("  Create a classic PAT (repo scope) at https://github.com/settings/tokens")
        return base_url
    # Both classic PATs (ghp_*) and fine-grained PATs (github_pat_*) work as
    # HTTP basic-auth passwords.  The x-access-token prefix is for GitHub App
    # installation tokens only — do NOT use it for personal access tokens.
    return base_url.replace("https://", f"https://{tok}@")


# ---------------------------------------------------------------------------
# Setup: install deps
# ---------------------------------------------------------------------------

def install_deps(gbv_dir: str = GBV_DIR) -> None:
    """pip-install requirements.txt + bitsandbytes + accelerate."""
    req = os.path.join(gbv_dir, "requirements.txt")
    # flash-attn is optional — may fail on older drivers
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q",
         "-r", req, "bitsandbytes", "accelerate", "flash-attn"],
        check=False,
    )
    # Core deps must succeed.
    # torchao>=0.16.0: Colab ships 0.10.0 but PEFT 0.14+ raises ImportError when
    # torchao is present at an incompatible version (instead of silently skipping it).
    # Upgrading torchao here prevents the LoRA init crash in trainer.py.
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q",
         "-r", req, "bitsandbytes", "accelerate", "torchao>=0.16.0"],
        check=True,
    )
    print("[3/5] Dependencies installed")


def _free_gb(path: str) -> float:
    """Return free space in GB at the filesystem containing path."""
    import shutil
    try:
        return shutil.disk_usage(path).free / 1024 ** 3
    except Exception:
        return 0.0


# Minimum Drive free space (GB) below which we fall back to the runtime disk.
# Free Drive accounts ship with 15 GB total; after repo + DB + other files,
# <2 GB is "essentially full" — even a Qwen3-0.6B download (1.3 GB) could fail.
_DRIVE_MIN_FREE_GB  = 2.0

# Models larger than this threshold (GB) won't be pre-cached on Drive even when
# Drive has plenty of space — they go to the runtime disk (~100 GB free).
_DRIVE_MAX_MODEL_GB = 5.0

# Known approximate sizes in GB (safetensors, no quantisation).
# Used to decide where to cache before downloading.
_MODEL_SIZE_GB = {
    "Qwen/Qwen3-0.6B": 1.3,
    "Qwen/Qwen3-1.7B": 3.5,
    "Qwen/Qwen3-4B":   8.0,
    "Qwen/Qwen3-8B":  16.0,
    "Qwen/Qwen3-14B": 28.0,
    "Qwen/Qwen3-32B": 64.0,
}


def setup_hf_cache(drive_root: str = DRIVE_ROOT) -> str:
    """Point HuggingFace cache at Drive (if space allows) or runtime disk.

    Decision logic
    --------------
    Drive free ≥ _DRIVE_MIN_FREE_GB (2 GB):
        HF_HOME → Drive/hf_cache  — small models survive session restarts.
        Models > _DRIVE_MAX_MODEL_GB (5 GB) are still skipped by prefetch_models
        and will be downloaded to runtime disk at train/eval time.
    Drive free < _DRIVE_MIN_FREE_GB:
        HF_HOME → ~/.cache/huggingface  — runtime disk (~100 GB free, ephemeral).
        All models re-download on each session, but nothing crashes.
        A warning is printed so the user knows to clear Drive.

    Always clears the three OFFLINE env flags so subprocesses can reach HF.

    Returns the cache path that was actually set (may be Drive or local).
    """
    hf_cache = os.path.join(drive_root, "hf_cache")
    os.makedirs(hf_cache, exist_ok=True)
    drive_free = _free_gb(hf_cache)

    if drive_free < _DRIVE_MIN_FREE_GB:
        # Drive too full — fall back to ephemeral runtime disk
        local_cache = os.path.expanduser("~/.cache/huggingface")
        os.makedirs(local_cache, exist_ok=True)
        cache = local_cache
        print(f"[cache] ⚠  Drive low: {drive_free:.1f} GB free "
              f"(need ≥ {_DRIVE_MIN_FREE_GB:.1f} GB)")
        print(f"[cache] HF model cache → {cache}  "
              f"(runtime disk, ~100 GB free, ephemeral)")
        print(f"[cache]    Models re-download each session until you free up Drive.")
    else:
        cache = hf_cache
        print(f"[cache] HF model cache → {cache}  ({drive_free:.1f} GB free on Drive)")

    os.environ["HF_HOME"]            = cache
    os.environ["TRANSFORMERS_CACHE"] = cache
    os.environ["HF_DATASETS_CACHE"]  = os.path.join(cache, "datasets")
    # Always allow online lookups — never inherit a stale OFFLINE flag
    os.environ.pop("TRANSFORMERS_OFFLINE",  None)
    os.environ.pop("HF_DATASETS_OFFLINE",   None)
    os.environ.pop("HF_HUB_OFFLINE",        None)
    return cache


def prefetch_models(config: str, gbv_dir: str = GBV_DIR,
                    drive_root: str = DRIVE_ROOT) -> None:
    """Pre-download draft + teacher model weights before the pipeline starts.

    Must be called AFTER setup_hf_cache() — it reads HF_HOME from the environment
    to decide where to write, so it automatically works whether setup_hf_cache()
    chose Drive or the runtime disk.

    Strategy when HF_HOME → Drive (Drive has ≥ 2 GB free):
    - Models ≤ 5 GB (Qwen3-0.6B, Qwen3-1.7B): cached on Drive — download once,
      reuse across sessions.
    - Models > 5 GB (Qwen3-4B, Qwen3-8B, …): Drive free space on a free Google
      account is 15 GB total; these won't fit.  Skipped here; the trainer
      downloads them to the runtime disk at run time (~5-10 min first session).

    Strategy when HF_HOME → runtime disk (Drive had < 2 GB free):
    - All models download to ~/.cache/huggingface/ (~100 GB free, ephemeral).
      No "too large for Drive" skip — everything fits locally.
      Models re-download on every fresh session, but the pipeline never crashes.

    Safe to re-run — snapshot_download() skips files already cached.
    """
    import yaml  # type: ignore
    from huggingface_hub import snapshot_download  # type: ignore

    yaml_path = os.path.join(
        gbv_dir, "orchestration", "configs",
        config.replace("/", os.sep) + ".yaml",
    )
    if not os.path.exists(yaml_path):
        print(f"[prefetch] YAML not found: {yaml_path} — skipping")
        return

    with open(yaml_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    models_cfg = cfg.get("models", {})
    draft_id   = models_cfg.get("draft",  "Qwen/Qwen3-0.6B")
    target_id  = models_cfg.get("target", "Qwen/Qwen3-1.7B")

    # Use whatever cache location setup_hf_cache() chose — may be Drive or
    # the local runtime disk if Drive was too full.
    cache_dir  = os.environ.get("HF_HOME",
                                os.path.join(drive_root, "hf_cache"))
    cache_free = _free_gb(cache_dir)
    on_drive   = os.path.abspath(drive_root) in os.path.abspath(cache_dir)

    for model_id in dict.fromkeys([draft_id, target_id]):   # dedupe, keep order
        est_gb = _MODEL_SIZE_GB.get(model_id, 99.0)         # unknown = assume large

        # Large models stay off Drive even when Drive has free space — they won't
        # fit on a free Google account (15 GB total).  If we've already fallen
        # back to runtime disk (on_drive=False), this check is skipped and the
        # model downloads locally instead (~100 GB free).
        if on_drive and est_gb > _DRIVE_MAX_MODEL_GB:
            print(f"[prefetch] {model_id} (~{est_gb:.0f} GB) — too large for Drive "
                  f"(limit {_DRIVE_MAX_MODEL_GB:.0f} GB). "
                  f"Trainer downloads to runtime disk at run time "
                  f"(~5-10 min first session).")
            continue

        if est_gb > cache_free - 0.5:   # keep 0.5 GB headroom
            dest = "Drive" if on_drive else "runtime disk"
            print(f"[prefetch] {model_id} (~{est_gb:.0f} GB) — not enough space on "
                  f"{dest} ({cache_free:.1f} GB free). Trainer downloads at run time.")
            continue

        dest_label = "Drive cache" if on_drive else "runtime disk"
        print(f"[prefetch] {model_id} (~{est_gb:.1f} GB) → {dest_label} …", flush=True)
        try:
            path = snapshot_download(model_id, ignore_patterns=["*.gguf", "*.bin"])
            cache_free -= est_gb        # update estimate for next iteration
            print(f"[prefetch] ✓  {model_id}  → {path}")
        except Exception as exc:
            print(f"[prefetch] ✗  {model_id}: {exc}")
            print("  Trainer will attempt download at run time.")


def _from_kaggle_gsm8k_csv(csv_path: str, out_jsonl: str) -> bool:
    """Convert main_train.csv from thedevastator/grade-school-math-8k-q-a to gsm8k_train.jsonl.

    The CSV has 'question' and 'answer' columns matching our standard JSONL format.
    Returns True on success (file written), False if conversion failed.
    """
    import csv
    try:
        q_col = a_col = None
        with open(csv_path, encoding="utf-8", newline="") as f:
            cols = next(csv.reader(f))
        for c in cols:
            cl = c.lower().strip()
            if cl == "question":
                q_col = c
            elif cl == "answer":
                a_col = c
        if not q_col or not a_col:
            print(f"[data] ⚠ CSV columns {cols} — expected 'question' and 'answer'. Skipping.")
            return False
        count = 0
        with open(csv_path, encoding="utf-8", newline="") as f, \
             open(out_jsonl, "w", encoding="utf-8") as out:
            for row in csv.DictReader(f):
                q = row.get(q_col, "").strip()
                a = row.get(a_col, "").strip()
                if q and a:
                    out.write(json.dumps({"question": q, "answer": a}) + "\n")
                    count += 1
        print(f"[data] gsm8k_train.jsonl ← {os.path.basename(csv_path)} ({count} rows)")
        return count > 0
    except Exception as exc:
        print(f"[data] CSV conversion failed: {exc}")
        return False


def fetch_training_data(gbv_dir: str = GBV_DIR) -> None:
    """Download gsm8k_train.jsonl to core/datasets/raw/ if not already present.

    This is a one-time setup step — safe to re-run (skips if file exists).
    Call this from the Colab setup cell after install_deps().

    The file is gitignored (7,473 prompts, ~3 MB) so it must be fetched on
    every fresh Colab session.  The downloader falls back to 30 hardcoded
    GSM8K prompts if HuggingFace is unreachable.

    Raises RuntimeError if the file is still missing after the download attempt
    so the pipeline never starts without training data.
    """
    raw_dir    = os.path.join(gbv_dir, "core", "datasets", "raw")
    train_path = os.path.join(raw_dir, "gsm8k_train.jsonl")
    if os.path.exists(train_path):
        print(f"[data] gsm8k_train.jsonl already present — skipping download.")
        return
    os.makedirs(raw_dir, exist_ok=True)
    print("[data] Downloading gsm8k_train.jsonl from HuggingFace…")
    downloader = os.path.join(gbv_dir, "core", "datasets", "downloader.py")
    result = subprocess.run(
        [sys.executable, downloader, "--datasets", "gsm8k_train"],
        cwd=gbv_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    # Always show downloader output — makes failures immediately visible
    if result.stdout:
        print(result.stdout.rstrip())
    if result.stderr:
        print("[data stderr]", result.stderr[:800].rstrip())
    if os.path.exists(train_path):
        import pathlib
        size_kb = pathlib.Path(train_path).stat().st_size // 1024
        print(f"[data] ✓ gsm8k_train.jsonl ready ({size_kb} KB)")
    else:
        raise RuntimeError(
            f"[data] gsm8k_train.jsonl still missing after download attempt "
            f"(downloader exit {result.returncode}).\n"
            f"  Check the stderr output above for the root cause.\n"
            f"  Manual retry:\n"
            f"    import subprocess, sys\n"
            f"    subprocess.run([sys.executable, '{downloader}', '--datasets', 'gsm8k_train'], check=True)"
        )


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def auth_wandb() -> None:
    """Login to W&B from WANDB_API_KEY secret, or fall back to offline mode."""
    key = get_secret("WANDB_API_KEY")
    if key:
        os.environ["WANDB_API_KEY"] = key
        import wandb  # type: ignore
        wandb.login(key=key, relogin=True)
        print("[4/5] W&B authenticated")
    else:
        os.environ["WANDB_MODE"] = "offline"
        print("[4/5] W&B offline  (add WANDB_API_KEY via left sidebar → Secrets)")


def auth_hf() -> None:
    """Login to HuggingFace Hub from HF_TOKEN secret."""
    tok = get_secret("HF_TOKEN")
    if tok:
        os.environ["HF_TOKEN"] = tok
        os.environ["HUGGING_FACE_HUB_TOKEN"] = tok
        try:
            from huggingface_hub import login  # type: ignore
            login(token=tok, add_to_git_credential=False)
            print("     HuggingFace authenticated")
        except Exception:
            pass
    else:
        print("     HF_TOKEN not set (OK for public Qwen3 models)")


def keep_alive() -> None:
    """Inject a 45-second JS heartbeat to prevent Colab idle-timeout.

    Call this at the top of any long-running cell (Cell 7, Cell 8, etc.).
    The heartbeat fires a synthetic mousemove event and clicks the reconnect
    button if it appears, keeping the browser from triggering idle logout.

    Safe to call multiple times — the JS guard (window.__sd_ka) ensures only
    one interval is registered per page.
    """
    try:
        from IPython.display import display, Javascript  # type: ignore
        display(Javascript("""
(function(){
  if(window.__sd_ka)return;
  window.__sd_ka=setInterval(function(){
    document.dispatchEvent(new MouseEvent('mousemove',{bubbles:true}));
    var b=document.querySelector('[data-tooltip="Reconnect to runtime"]');
    if(b)b.click();
  },45000);
  console.log('[specdist] keep-alive on (45 s heartbeat)');
})();
"""))
    except Exception:
        pass  # not in a browser / no IPython — silently skip


def check_gpu(warn_below_gb: float = 12.0) -> None:
    """Print GPU name + free VRAM; warn if below warn_below_gb."""
    import torch  # type: ignore
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(0)
        gb = total / 1024 ** 3
        print(f"     GPU: {torch.cuda.get_device_name(0)}  "
              f"{free / 1024 ** 3:.1f} GB free / {gb:.1f} GB total")
        if gb < warn_below_gb:
            print(f"⚠ Only {gb:.1f} GB VRAM — colab config needs T4 (15 GB)")
    else:
        print("⚠ No GPU — Runtime → Change runtime type → T4 / A100 GPU")


# ---------------------------------------------------------------------------
# One-call bootstrap (replaces 60-80 lines of duplicated cell logic)
# ---------------------------------------------------------------------------

def bootstrap(
    config: str,
    storage_root: str,
    repo_dir: str,
    *,
    gbv_dir: str = None,
    mount_drive: bool = False,
    drive_mount_path: str = "/content/drive",
    kaggle_hf_dataset: str = None,
    kaggle_data_dataset: str = None,
    kaggle_gsm8k_dataset: str = None,
    skip_model_prefetch: bool = False,
    restore_checkpoints: bool = False,
    checkpoint_dataset_names: tuple = (
        "specdist-checkpoints", "specdist_checkpoints",
        "specdist-output", "specdist",
    ),
    warn_vram_below_gb: float = 12.0,
) -> None:
    """One-call environment bootstrap for all SpecDist notebook platforms.

    Call this after cloning the repo and inserting deploy/ into sys.path.
    All subsequent setup (Drive mount, secrets, dirs, deps, HF cache, prefetch,
    W&B auth, HF auth, training data, GPU check, optional checkpoint restore)
    happens inside this function.

    Parameters
    ----------
    config            YAML profile name (e.g. "kaggle", "colab", "a100").
    storage_root      Where checkpoints/logs/results.db live.
                      Colab: DRIVE_ROOT.  Kaggle: /kaggle/working/specdist.
    repo_dir          Root of the cloned repo (parent of gbv-research/).
    gbv_dir           gbv-research/ subdir. Defaults to repo_dir/gbv-research.
    mount_drive       Mount Google Drive at drive_mount_path before doing anything
                      else (Colab only; no-op on Kaggle / local).
    drive_mount_path  Mount point for Drive (default /content/drive).
    kaggle_hf_dataset Path to a pre-attached Kaggle dataset containing HF weights.
                      When set and the path exists, sets HF_HOME there and skips
                      model download entirely.
    kaggle_data_dataset Path to a pre-attached Kaggle dataset containing eval/training
                      JSONL files. When set, .jsonl files are copied into
                      core/datasets/raw/ before fetch_training_data() runs so that
                      step skips the HuggingFace download entirely.
    kaggle_gsm8k_dataset Path to thedevastator/grade-school-math-8k-q-a Kaggle dataset.
                      When set, main_train.csv is converted to gsm8k_train.jsonl and
                      placed in core/datasets/raw/, skipping the HF download.
    skip_model_prefetch When True, skips prefetch_models() (the HF model weight download).
                      Set this whenever Kaggle Models are attached via Add Model — the
                      pipeline receives model paths via --draft/--target instead.
                      Without this flag, prefetch would waste 5-10 min downloading
                      17+ GB of weights that are already mounted read-only.
    restore_checkpoints  Search /kaggle/input/<name>/ for a saved checkpoint
                      dataset and restore it into storage_root/checkpoints/.
                      Use this in Resume cells on Kaggle.
    checkpoint_dataset_names  Ordered list of Kaggle input dataset name variants
                      to try when restore_checkpoints=True.
    warn_vram_below_gb  Warn if total GPU VRAM is below this threshold.
                      Use 12.0 for T4, 30.0 for A100.
    """
    import shutil

    if gbv_dir is None:
        gbv_dir = os.path.join(repo_dir, "gbv-research")

    # 1. Keep-alive JS heartbeat (prevents idle timeout on Colab and Kaggle).
    #    The Reconnect-button querySelector is a no-op on Kaggle — mousemove still fires.
    keep_alive()

    # 2. Mount Google Drive (Colab only — skipped on Kaggle / local).
    if mount_drive:
        try:
            from google.colab import drive  # type: ignore
            drive.mount(drive_mount_path)
            print(f"[boot] Drive → {drive_mount_path}")
        except Exception as exc:
            print(f"⚠ Drive mount failed: {exc}")

    # 3. Populate all platform secrets → environment.
    #    get_secret() tries Colab userdata → Kaggle UserSecretsClient → env var.
    for _k in ("WANDB_API_KEY", "HF_TOKEN", "GITHUB_TOKEN"):
        _v = get_secret(_k)
        if _v:
            os.environ[_k] = _v

    # 4. Storage directories.
    os.makedirs(os.path.join(storage_root, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(storage_root, "logs"), exist_ok=True)
    os.chdir(gbv_dir)

    # 5. Install pip dependencies (wiped on every session restart).
    install_deps(gbv_dir)

    # 6. HF model cache: use pre-attached Kaggle dataset (instant) or download.
    if kaggle_hf_dataset and os.path.isdir(kaggle_hf_dataset):
        # Legacy: pre-uploaded HF cache as a Kaggle dataset.
        os.environ["HF_HOME"]            = kaggle_hf_dataset
        os.environ["TRANSFORMERS_CACHE"] = kaggle_hf_dataset
        for _flag in ("TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "HF_HUB_OFFLINE"):
            os.environ.pop(_flag, None)
        print(f"[cache] Using attached HF cache dataset: {kaggle_hf_dataset}")
    else:
        # Set up the cache directory (fast, always needed for tokenizer artefacts).
        setup_hf_cache(storage_root)
        if skip_model_prefetch:
            # Kaggle Models are attached — model weights arrive via --draft/--target.
            # Skipping prefetch_models() saves 5-10 min and ~17 GB of downloads.
            print("[prefetch] Kaggle Models attached — skipping HF weight download.")
        else:
            prefetch_models(config, gbv_dir, storage_root)

    # 7. Authenticate W&B and HuggingFace.
    auth_wandb()
    auth_hf()

    # 8a. Pre-populate training + eval data from Kaggle dataset (if attached).
    #     fetch_training_data() (step 8) checks for file existence and skips if present.
    #     Copy .jsonl files from the dataset into core/datasets/raw/ before it runs.
    if kaggle_data_dataset and os.path.isdir(kaggle_data_dataset):
        _raw_dir = os.path.join(gbv_dir, "core", "datasets", "raw")
        os.makedirs(_raw_dir, exist_ok=True)
        _copied = 0
        for _fn in os.listdir(kaggle_data_dataset):
            _src = os.path.join(kaggle_data_dataset, _fn)
            _dst = os.path.join(_raw_dir, _fn)
            if os.path.isfile(_src) and _fn.endswith(".jsonl") and not os.path.exists(_dst):
                shutil.copy2(_src, _dst)
                _copied += 1
        if _copied:
            print(f"[data] {_copied} dataset file(s) <- {kaggle_data_dataset}")
        else:
            print(f"[data] Datasets already present — skipping copy from {kaggle_data_dataset}")

    # 8b. Convert GSM8K CSV from Kaggle dataset → gsm8k_train.jsonl (if provided).
    #     Runs before fetch_training_data() so the HF download is skipped when the
    #     file is already present (fetch_training_data checks existence first).
    if kaggle_gsm8k_dataset and os.path.isdir(kaggle_gsm8k_dataset):
        _raw_dir    = os.path.join(gbv_dir, "core", "datasets", "raw")
        _train_path = os.path.join(_raw_dir, "gsm8k_train.jsonl")
        if os.path.exists(_train_path):
            print("[data] gsm8k_train.jsonl already present — skipping CSV conversion")
        else:
            os.makedirs(_raw_dir, exist_ok=True)
            _csv = os.path.join(kaggle_gsm8k_dataset, "main_train.csv")
            if os.path.isfile(_csv):
                _from_kaggle_gsm8k_csv(_csv, _train_path)
            else:
                print(f"[data] main_train.csv not found in {kaggle_gsm8k_dataset}")

    # 8. Training data (downloads gsm8k_train.jsonl if missing; ~3 MB, idempotent).
    fetch_training_data(gbv_dir)

    # 9. GPU check — warn if below the expected VRAM for this platform.
    check_gpu(warn_below_gb=warn_vram_below_gb)

    # 10. Restore checkpoints from a Kaggle input dataset (Resume cell only).
    if restore_checkpoints:
        _ckpt_dst = os.path.join(storage_root, "checkpoints")
        _restored = False
        for _ds_name in checkpoint_dataset_names:
            _ds_path = f"/kaggle/input/{_ds_name}"
            if os.path.isdir(_ds_path):
                # Restore results.db if present in dataset and not yet on disk.
                _db_src = os.path.join(_ds_path, "specdist", "results.db")
                _db_dst = os.path.join(storage_root, "results.db")
                if os.path.exists(_db_src) and not os.path.exists(_db_dst):
                    shutil.copy2(_db_src, _db_dst)
                    print(f"[restore] results.db ← {_db_src}")
                # Restore any checkpoint dirs / files not already present.
                for _item in os.listdir(_ds_path):
                    _src = os.path.join(_ds_path, _item)
                    _dst = os.path.join(_ckpt_dst, _item)
                    if not os.path.exists(_dst):
                        if os.path.isdir(_src):
                            shutil.copytree(_src, _dst)
                        else:
                            shutil.copy2(_src, _dst)
                print(f"[restore] Checkpoints ← {_ds_path}: {os.listdir(_ckpt_dst)}")
                _restored = True
                break
        if not _restored:
            print("[restore] No checkpoint dataset found — starting from scratch.")
            print("  Attach via: Add Data → Your Datasets → 'specdist-checkpoints'")

    print()


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run_pipeline(
    config: str,
    drive_root: str = DRIVE_ROOT,
    gbv_dir: str = GBV_DIR,
    *,
    smoke: bool = False,
    losses: str = None,
    background: bool = False,
    extra_args: list = None,
):
    """
    Launch orchestration/experiment.py with the given config.

    Returns:
        subprocess.Popen if background=True, else subprocess.CompletedProcess.
    """
    os.chdir(gbv_dir)
    os.environ.update({
        "SPECDIST_STORAGE_ROOT": drive_root,
        "SPECDIST_DB_PATH":      os.path.join(drive_root, "results.db"),
        "SPECDIST_LOGS_ROOT":    os.path.join(drive_root, "logs"),
    })
    # Ensure child process can reach HuggingFace — never inherit a stale offline flag
    for _flag in ("TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "HF_HUB_OFFLINE"):
        os.environ.pop(_flag, None)
    log_file = os.path.join(drive_root, "logs", "pipeline_output.log")

    cmd = [sys.executable, "orchestration/experiment.py",
           "--config", config, "--storage_root", drive_root, "--yes"]
    if smoke:
        cmd.append("--smoke")
    if losses:
        cmd += ["--losses", losses]
    if extra_args:
        cmd += extra_args

    mode = "SMOKE TEST" if smoke else f"FULL pipeline ({config})"
    print(f"{'[BG] ' if background else ''}{mode}")
    print(f"  Log → {log_file}")
    print(f"  DB  → {drive_root}/results.db")

    if background:
        # experiment.py manages its own log file (_PIPELINE_LOG → pipeline_output.log).
        # Discard subprocess stdout/stderr — experiment.py writes everything it needs to
        # the log file itself.  Do NOT tail the log to sys.stdout here: a background
        # thread writing to sys.stdout bleeds into whatever Colab cell runs next.
        # Use monitor() or start_dashboard() to track progress.
        proc = subprocess.Popen(cmd, cwd=gbv_dir,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
        print(f"\n  PID {proc.pid} — run Cell 6 (monitor) to track progress")
        print(f"  Log → {log_file}")
        print(f"  Stop: import os, signal; os.kill({proc.pid}, signal.SIGTERM)")
        return proc
    else:
        result = subprocess.run(cmd, cwd=gbv_dir)
        if result.returncode == 0:
            print("\n✓ Pipeline complete — call start_dashboard() to view results.")
        else:
            print(f"\n✗ Exit {result.returncode} — re-run to resume from last checkpoint.")
            print(f"  Full log: {log_file}")
        return result


# ---------------------------------------------------------------------------
# Profile inspector (reads YAML, prints key params before a run)
# ---------------------------------------------------------------------------

def show_profile(profile: str, gbv_dir: str = GBV_DIR) -> None:
    """Read a YAML profile and print key training params."""
    import yaml  # type: ignore
    yaml_path = os.path.join(
        gbv_dir, "orchestration", "configs",
        profile.replace("/", os.sep) + ".yaml",
    )
    try:
        with open(yaml_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        tr   = cfg.get("training", {})
        hw   = cfg.get("hardware", {})
        mdl  = cfg.get("models", {})
        tree = cfg.get("tree_training", {})
        ev   = cfg.get("evaluation", {})
        quant = "4-bit NF4" if hw.get("load_in_4bit") else "BF16"
        print(f"  Profile  : {profile}")
        print(f"  Teacher  : {mdl.get('target', '?')}  ({quant})")
        print(f"  Draft    : {mdl.get('draft', '?')}")
        print(f"  Steps    : {tr.get('steps', '?')}  "
              f"lr={tr.get('lr', '?')}  lora_r={tr.get('lora_r', '?')}")
        if tree:
            print(f"  Tree     : K={tree.get('tree_K', '?')}  "
                  f"L={tree.get('tree_L', '?')}")
        print(f"  Eval     : modes={ev.get('modes', '?')}  "
              f"n_prompts={ev.get('n_prompts', '?')}")
    except Exception as e:
        print(f"  Profile  : {profile}  (could not read YAML: {e})")


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

def start_dashboard(
    drive_root: str = DRIVE_ROOT,
    gbv_dir: str = GBV_DIR,
    port: int = 5000,
):
    """Start the Flask training dashboard and print its Colab proxy URL."""
    from google.colab.output import eval_js  # type: ignore
    db_path = os.path.join(drive_root, "results.db")
    if not os.path.exists(db_path):
        print(f"⚠ No results DB at {db_path}")
        print("  Run the pipeline first, then call start_dashboard().")
        return None
    os.environ["SPECDIST_DB_PATH"]   = db_path
    os.environ["SPECDIST_LOGS_ROOT"] = os.path.join(drive_root, "logs")

    proc_box = [None]

    def _run():
        proc_box[0] = subprocess.Popen(
            [sys.executable,
             os.path.join(gbv_dir, "dashboard", "training_dashboard.py"),
             "--host", "0.0.0.0", "--port", str(port)],
            env=os.environ.copy(),
        )
        proc_box[0].wait()

    threading.Thread(target=_run, daemon=True).start()
    time.sleep(3)
    url = eval_js(f"google.colab.kernel.proxyPort({port})")
    print(f"✓ Dashboard → {url}")
    print("  Auto-refreshes every 15 s. Stop: Runtime → Interrupt execution.")
    return proc_box[0]


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------

def monitor(
    drive_root: str = DRIVE_ROOT,
    config: str = "colab",
    log_tail: int = 60,
    auto_refresh: bool = False,
    refresh_secs: int = 20,
) -> None:
    """
    Print pipeline state + log tail.
    Set auto_refresh=True for a live updating view (interrupt cell to stop).
    """
    from IPython.display import clear_output  # type: ignore

    # Normalise config → slug (matches experiment.py _config_slug logic).
    # Accepts both slash form ("profiles/colab_lite_tree_losses") and
    # already-normalised form ("profiles_colab_lite_tree_losses").
    _config_slug = config.replace("/", "_").replace(os.sep, "_")
    log_file   = os.path.join(drive_root, "logs", "pipeline_output.log")
    state_file = os.path.join(drive_root, f"pipeline_state_{_config_slug}.json")

    def _show():
        sep = "=" * 62
        print(sep); print("PIPELINE STATE"); print(sep)
        if os.path.exists(state_file):
            state  = json.load(open(state_file, encoding="utf-8"))
            steps  = state.get("steps", {})
            counts: dict = {}
            for sid, info in steps.items():
                s = info.get("status", "pending")
                counts[s] = counts.get(s, 0) + 1
                icon = {"done": "OK", "running": ">>",
                        "failed": "XX", "error": "!!", "pending": ".."}.get(s, "..")
                ts = (info.get("finished_at") or
                      info.get("started_at") or "")[:16]
                print(f"  [{icon}] {sid:45s}  {s:8s}  {ts}")
            print(f"\n  Done:{counts.get('done', 0)}  "
                  f"Running:{counts.get('running', 0)}  "
                  f"Pending:{counts.get('pending', 0)}  "
                  f"Failed:{counts.get('failed', 0)}  "
                  f"Error:{counts.get('error', 0)}")
        else:
            print(f"  State file not found: {state_file}")
            print("  Pipeline not started yet (or Drive not mounted).")

        print()
        print(sep); print(f"LOG (last {log_tail} lines)"); print(sep)
        if os.path.exists(log_file):
            lines = open(log_file, encoding="utf-8",
                         errors="replace").readlines()
            print("".join(lines[-log_tail:]))
            age = (datetime.datetime.now().timestamp() -
                   os.path.getmtime(log_file))
            status = ("ACTIVE" if age < 120
                      else f"STALE ({int(age // 60)} min ago)")
            print(f"[{len(lines)} lines | {status}]")
        else:
            print(f"  Not found: {log_file}")

        err_logs = sorted(
            glob.glob(os.path.join(drive_root, "logs", "step_*_error.log")))
        if err_logs:
            print()
            print(sep)
            print(f"ERRORS ({len(err_logs)} file(s))")
            print(sep)
            for ef in err_logs:
                txt = open(ef, encoding="utf-8",
                           errors="replace").read()
                print(f"\n--- {os.path.basename(ef)} ---")
                print(txt[-2000:] if len(txt) > 2000 else txt)

        if auto_refresh:
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            print(f"\n[{ts} | next in {refresh_secs}s | interrupt to stop]")

    if auto_refresh:
        print(f"Auto-refresh every {refresh_secs}s — interrupt cell to stop.")
        while True:
            clear_output(wait=True)
            _show()
            time.sleep(refresh_secs)
    else:
        _show()
