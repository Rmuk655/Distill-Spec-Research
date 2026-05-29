"""
_provider.py — SpecDist provider configuration registry.

All launchers (Colab, Kaggle, Modal, RunPod, local) share the same
underlying command:

    python orchestration/experiment.py \
        --config  {provider.pipeline_config} \
        --storage_root {provider.storage_root} \
        --yes

``storage_root`` is the single persistent directory that contains ALL
experiment artifacts:
    {storage_root}/
        results.db           ← SQLite experiment DB
        checkpoints/         ← LoRA adapters
        logs/                ← pipeline + training logs
        pipeline_state_*.json

This makes every artifact relocatable: change ``storage_root`` and
nothing else needs updating.

This file defines what differs per provider: hardware config, storage
path, session limits, and cost.  To add a new provider:

    1. Add a ProviderConfig entry to PROVIDERS below.
    2. Create deploy/my_provider.{ipynb,py,sh} that reads from PROVIDERS.

Usage from any launcher script
-------------------------------
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from deploy._provider import PROVIDERS, detect_provider, build_pipeline_cmd

    p = detect_provider()   # auto-detects current environment
    cmd = build_pipeline_cmd(p, smoke=True)
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ProviderConfig:
    name: str                    # short identifier
    pipeline_config: str         # maps to orchestration/configs/{X}.yaml
    storage_root: str            # persistent root: DB + checkpoints + logs live here
    hf_home: Optional[str]       # HF_HOME (None = platform default)
    gpu_vram_gb: float           # expected available VRAM in GB
    session_max_h: float         # hard session wall-clock limit in hours
    cost_per_hour: float         # approx USD/hr (0 = free tier)
    notes: str = ""
    # --- derived helpers ---
    env_signal: tuple[str, ...] = field(default_factory=tuple)
    # env vars that uniquely identify this provider (used by detect_provider)

    @property
    def ckpt_root(self) -> str:
        """Backward-compat shim — checkpoints live under storage_root."""
        return os.path.join(self.storage_root, "checkpoints") if self.storage_root else ""


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------
# Add a new entry here when you add a new provider launcher.
# Column meanings:
#   pipeline_config → which YAML under orchestration/configs/ to use
#   storage_root    → PERSISTENT root dir: results.db + checkpoints/ + logs/
#                     Must survive session restart (Drive / volume / /workspace)
#   gpu_vram_gb     → usable VRAM in GB
#   session_max_h   → hard wall-clock limit (0 = no limit)
#   cost_per_hour   → USD (0 = free / included)

PROVIDERS: dict[str, ProviderConfig] = {

    # ── Free cloud ──────────────────────────────────────────────────────────
    "colab": ProviderConfig(
        name             = "colab",
        pipeline_config  = "colab",
        storage_root     = "/content/drive/MyDrive/specdist",
        hf_home          = None,            # Colab has no persistent HF cache; models dl each session
        gpu_vram_gb      = 15.0,            # free T4 = 15 GB
        session_max_h    = 1.5,             # free tier: ~90 min idle disconnect
        cost_per_hour    = 0.0,
        notes            = (
            "Free T4 (15 GB). Teacher loaded in 4-bit NF4 (colab.yaml). "
            "All artifacts (DB, checkpoints, logs) on Google Drive — survive restart. "
            "Mount Drive BEFORE running the launcher."
        ),
        env_signal       = ("COLAB_BACKEND_VERSION", "COLAB_RELEASE_TAG"),
    ),

    "colab_pro": ProviderConfig(
        name             = "colab_pro",
        pipeline_config  = "server",        # A100 → full bf16, no 4-bit needed
        storage_root     = "/content/drive/MyDrive/specdist",
        hf_home          = None,
        gpu_vram_gb      = 40.0,            # A100 40 GB
        session_max_h    = 12.0,            # Pro: up to 12 h
        cost_per_hour    = 0.0,             # included in Pro subscription
        notes            = (
            "Colab Pro A100 (40 GB). Server config: 8B teacher in bfloat16. "
            "Runs all 6 losses overnight in one session."
        ),
        env_signal       = ("COLAB_BACKEND_VERSION",),
    ),

    "kaggle": ProviderConfig(
        name             = "kaggle",
        pipeline_config  = "colab",         # same T4 hardware as free Colab
        storage_root     = "/kaggle/working/specdist",
        hf_home          = "/kaggle/working/hf_cache",
        gpu_vram_gb      = 15.0,            # T4 or P100 depending on availability
        session_max_h    = 12.0,            # 12 h sessions, 30 h/week free
        cost_per_hour    = 0.0,
        notes            = (
            "Free Kaggle GPU (T4/P100, 30 h/week). Longer sessions than free Colab. "
            "No Drive — artifacts to /kaggle/working/specdist/. "
            "Download results.db + checkpoints as Kaggle output after each session."
        ),
        env_signal       = ("KAGGLE_KERNEL_RUN_TYPE",),
    ),

    # ── Paid cloud ──────────────────────────────────────────────────────────
    "modal": ProviderConfig(
        name             = "modal",
        pipeline_config  = "server",        # A100 → full bf16 teacher
        storage_root     = "/vol",          # Modal persistent volume mounted at /vol
        hf_home          = "/vol/hf_cache",
        gpu_vram_gb      = 40.0,            # A100-40GB default; switchable
        session_max_h    = 0.0,             # no hard limit; billed per second
        cost_per_hour    = 1.10,            # A100-40GB ~$1.10/h; A10G ~$0.76/h
        notes            = (
            "Modal.com on-demand A100. Persistent Volume at /vol stores all artifacts "
            "(results.db, checkpoints, logs) across container restarts."
        ),
        env_signal       = ("MODAL_TASK_ID",),
    ),

    "runpod": ProviderConfig(
        name             = "runpod",
        pipeline_config  = "server",
        storage_root     = "/workspace/specdist",
        hf_home          = "/workspace/hf_cache",
        gpu_vram_gb      = 24.0,            # RTX 3090 community (~$0.44/h)
        session_max_h    = 0.0,
        cost_per_hour    = 0.44,            # varies; community GPU
        notes            = (
            "RunPod community / secure cloud. /workspace is a persistent network volume. "
            "Use the runpod.sh launcher for one-command setup."
        ),
        env_signal       = ("RUNPOD_POD_ID",),
    ),

    "hf_spaces": ProviderConfig(
        name             = "hf_spaces",
        pipeline_config  = "colab",
        storage_root     = "/data/specdist",  # HF persistent Space storage
        hf_home          = "/data/hf_cache",
        gpu_vram_gb      = 15.0,
        session_max_h    = 0.0,
        cost_per_hour    = 0.60,            # Zero GPU ~$0.60/h; A10G ~$1.05/h
        notes            = (
            "HuggingFace Spaces ZeroGPU / persistent GPU. "
            "/data is the persistent Space storage. "
            "Set WANDB_API_KEY and HF_TOKEN in Space secrets."
        ),
        env_signal       = ("SPACE_ID",),
    ),

    # ── Adobe internal ──────────────────────────────────────────────────────
    "aip": ProviderConfig(
        name             = "aip",
        pipeline_config  = "a100",  # default: A100 BF16; switch to "kaggle" for T4/V100 16 GB
        storage_root     = "",            # set dynamically in aip.ipynb: /sensei-fs-3/users/$USER/specdist
        hf_home          = "",            # set to /sensei-fs-3/users/$USER/specdist/hf_cache in notebook
        gpu_vram_gb      = 40.0,          # A100 40 GB; override if allocated a smaller GPU
        session_max_h    = 4.0,           # AIP interactive: 4-hour guaranteed runtime
        cost_per_hour    = 0.0,           # internal Adobe resource
        notes            = (
            "Adobe AI Platform interactive session. "
            "Sensei FS at /sensei-fs-3/users/$USER/ persists across sessions (like Lightning AI). "
            "Models download once, checkpoints never lost. "
            "4-hour session limit — always use BACKGROUND=True and Cell 1 to resume. "
            "Secrets: export VAR=value in VS Code terminal (no platform secret API). "
            "Up to 3 GPUs/user; device_map=auto shards teacher across all GPUs."
        ),
        env_signal       = ("AIP_POD_NAME", "ADOBE_AIP_ENV"),  # Adobe AIP K8s env vars (if set)
    ),

    # ── Local ───────────────────────────────────────────────────────────────
    "local_laptop": ProviderConfig(
        name             = "local_laptop",
        pipeline_config  = "laptop",
        storage_root     = "",              # empty → experiment.py defaults (gbv-research/db/)
        hf_home          = None,
        gpu_vram_gb      = 4.0,
        session_max_h    = 0.0,
        cost_per_hour    = 0.0,
        notes            = "Local laptop GPU. Smoke tests only (too slow for full runs).",
        env_signal       = (),
    ),

    "local_server": ProviderConfig(
        name             = "local_server",
        pipeline_config  = "server",
        storage_root     = "",              # empty → experiment.py defaults (gbv-research/db/)
        hf_home          = None,
        gpu_vram_gb      = 24.0,
        session_max_h    = 0.0,
        cost_per_hour    = 0.0,
        notes            = "Local server / desktop GPU (RTX 3090 / 4090 or better).",
        env_signal       = (),
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def detect_provider() -> ProviderConfig:
    """Auto-detect the current provider from environment variables.

    Returns the matching ProviderConfig, or ``PROVIDERS["local_laptop"]``
    if no cloud provider is detected.
    """
    for cfg in PROVIDERS.values():
        if any(k in os.environ for k in cfg.env_signal):
            return cfg
    return PROVIDERS["local_laptop"]


def build_pipeline_cmd(
    provider: ProviderConfig,
    smoke: bool = False,
    losses: str | None = None,
    extra: list[str] | None = None,
) -> list[str]:
    """Return the subprocess argv list for running the pipeline.

    Parameters
    ----------
    provider : ProviderConfig
        Target provider (from PROVIDERS registry).
    smoke : bool
        If True, add ``--smoke`` (quick code-path verification).
    losses : str | None
        Comma-separated loss names, e.g. ``"kl,ebe"``. None = all.
    extra : list[str] | None
        Any additional experiment.py flags.
    """
    repo_root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
    cmd = [
        sys.executable,
        os.path.join(repo_root, "orchestration", "experiment.py"),
        "--config", provider.pipeline_config,
        "--yes",
    ]
    # storage_root pins all artifacts (DB, checkpoints, logs) to persistent storage.
    # On cloud providers this must point to mounted persistent storage so artifacts
    # survive session termination.  For local runs leave storage_root empty and
    # experiment.py will use its project-relative defaults.
    if provider.storage_root:
        cmd += ["--storage_root", provider.storage_root]
    if smoke:
        cmd.append("--smoke")
    if losses:
        cmd += ["--losses", losses]
    if extra:
        cmd += extra
    return cmd


def print_provider_table() -> None:
    """Print a comparison table of all registered providers."""
    header = f"{'Provider':<16} {'Config':<12} {'VRAM':>6} {'Session':>8} {'$/hr':>6}  {'storage_root':<35}  Notes"
    print(header)
    print("-" * len(header))
    for p in PROVIDERS.values():
        session = f"{p.session_max_h:.0f}h" if p.session_max_h else "∞"
        cost    = f"${p.cost_per_hour:.2f}" if p.cost_per_hour else "free"
        vram    = f"{p.gpu_vram_gb:.0f}GB"
        sr      = p.storage_root or "(project default)"
        print(f"{p.name:<16} {p.pipeline_config:<12} {vram:>6} {session:>8} {cost:>6}  {sr:<35}  {p.notes[:50]}")


if __name__ == "__main__":
    print_provider_table()
    print()
    detected = detect_provider()
    print(f"Detected provider: {detected.name}")
    print(f"Pipeline command : {' '.join(build_pipeline_cmd(detected))}")
