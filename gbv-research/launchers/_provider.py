"""
_provider.py — SpecDist provider configuration registry.

All launchers (Colab, Kaggle, Modal, RunPod, local) share the same
underlying command:

    python orchestration/pipeline.py \
        --config  {provider.pipeline_config} \
        --ckpt_root {provider.ckpt_root} \
        --yes

This file defines what differs per provider: hardware config, checkpoint
storage path, session limits, and cost.  To add a new provider:

    1. Add a ProviderConfig entry to PROVIDERS below.
    2. Create launchers/my_provider.{ipynb,py,sh} that reads from PROVIDERS.

Usage from any launcher script
-------------------------------
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from launchers._provider import PROVIDERS, detect_provider, build_pipeline_cmd

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
    ckpt_root: str               # persistent checkpoint storage path
    hf_home: Optional[str]       # HF_HOME (None = platform default)
    gpu_vram_gb: float           # expected available VRAM in GB
    session_max_h: float         # hard session wall-clock limit in hours
    cost_per_hour: float         # approx USD/hr (0 = free tier)
    notes: str = ""
    # --- derived helpers ---
    env_signal: tuple[str, ...] = field(default_factory=tuple)
    # env vars that uniquely identify this provider (used by detect_provider)


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------
# Add a new entry here when you add a new provider launcher.
# Column meanings:
#   pipeline_config → which YAML under orchestration/configs/ to use
#   ckpt_root       → where checkpoints are written (must survive session restart)
#   gpu_vram_gb     → usable VRAM in GB
#   session_max_h   → hard wall-clock limit (0 = no limit)
#   cost_per_hour   → USD (0 = free / included)

PROVIDERS: dict[str, ProviderConfig] = {

    # ── Free cloud ──────────────────────────────────────────────────────────
    "colab": ProviderConfig(
        name             = "colab",
        pipeline_config  = "colab",
        ckpt_root        = "/content/drive/MyDrive/specdist/checkpoints",
        hf_home          = None,            # Colab has no persistent HF cache; models dl each session
        gpu_vram_gb      = 15.0,            # free T4 = 15 GB
        session_max_h    = 1.5,             # free tier: ~90 min idle disconnect
        cost_per_hour    = 0.0,
        notes            = (
            "Free T4 (15 GB). Teacher loaded in 4-bit NF4 (colab.yaml). "
            "Checkpoints go to Google Drive — survives session restart. "
            "Mount Drive BEFORE cloning repo."
        ),
        env_signal       = ("COLAB_BACKEND_VERSION", "COLAB_RELEASE_TAG"),
    ),

    "colab_pro": ProviderConfig(
        name             = "colab_pro",
        pipeline_config  = "server",        # A100 → full bf16, no 4-bit needed
        ckpt_root        = "/content/drive/MyDrive/specdist/checkpoints",
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
        ckpt_root        = "/kaggle/working/specdist/checkpoints",
        hf_home          = "/kaggle/working/hf_cache",
        gpu_vram_gb      = 15.0,            # T4 or P100 depending on availability
        session_max_h    = 12.0,            # 12 h sessions, 30 h/week free
        cost_per_hour    = 0.0,
        notes            = (
            "Free Kaggle GPU (T4/P100, 30 h/week). Longer sessions than free Colab. "
            "No Drive — checkpoints to /kaggle/working/. "
            "Download checkpoint as Kaggle output after each session."
        ),
        env_signal       = ("KAGGLE_KERNEL_RUN_TYPE",),
    ),

    # ── Paid cloud ──────────────────────────────────────────────────────────
    "modal": ProviderConfig(
        name             = "modal",
        pipeline_config  = "server",        # A100 → full bf16 teacher
        ckpt_root        = "/vol/checkpoints",
        hf_home          = "/vol/hf_cache",
        gpu_vram_gb      = 40.0,            # A100-40GB default; switchable
        session_max_h    = 0.0,             # no hard limit; billed per second
        cost_per_hour    = 1.10,            # A100-40GB ~$1.10/h; A10G ~$0.76/h
        notes            = (
            "Modal.com on-demand A100. Persistent Volume stores models + checkpoints "
            "across container restarts. Best for overnight paper-quality runs."
        ),
        env_signal       = ("MODAL_TASK_ID",),
    ),

    "runpod": ProviderConfig(
        name             = "runpod",
        pipeline_config  = "server",
        ckpt_root        = "/workspace/specdist/checkpoints",
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
        ckpt_root        = "/data/checkpoints",  # HF persistent storage
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

    # ── Local ───────────────────────────────────────────────────────────────
    "local_laptop": ProviderConfig(
        name             = "local_laptop",
        pipeline_config  = "laptop",
        ckpt_root        = "",              # uses pipeline.py default (gbv-research/db/checkpoints)
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
        ckpt_root        = "",
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
        Any additional pipeline.py flags.
    """
    repo_root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
    cmd = [
        sys.executable,
        os.path.join(repo_root, "orchestration", "pipeline.py"),
        "--config", provider.pipeline_config,
        "--yes",
    ]
    if provider.ckpt_root:
        cmd += ["--ckpt_root", provider.ckpt_root]
    if smoke:
        cmd.append("--smoke")
    if losses:
        cmd += ["--losses", losses]
    if extra:
        cmd += extra
    return cmd


def print_provider_table() -> None:
    """Print a comparison table of all registered providers."""
    header = f"{'Provider':<16} {'Config':<12} {'VRAM':>6} {'Session':>8} {'$/hr':>6}  Notes"
    print(header)
    print("-" * len(header))
    for p in PROVIDERS.values():
        session = f"{p.session_max_h:.0f}h" if p.session_max_h else "∞"
        cost    = f"${p.cost_per_hour:.2f}" if p.cost_per_hour else "free"
        vram    = f"{p.gpu_vram_gb:.0f}GB"
        print(f"{p.name:<16} {p.pipeline_config:<12} {vram:>6} {session:>8} {cost:>6}  {p.notes[:60]}")


if __name__ == "__main__":
    print_provider_table()
    print()
    detected = detect_provider()
    print(f"Detected provider: {detected.name}")
    print(f"Pipeline command : {' '.join(build_pipeline_cmd(detected))}")
