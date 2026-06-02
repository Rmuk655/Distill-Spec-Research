"""
training_scaffold.py — shared training utilities for all gbv-research training scripts.

Every training script (distillspec_gbv/trainer.py, online_serve.py, eagle_train.py, …) imports
from here instead of copy-pasting the same WandB, checkpoint, DB, hardware, and model
loading boilerplate.

Design principles
-----------------
• All functions are standalone (no base class, no mandatory inheritance).
  Use the ones you need; ignore the rest.
• Model-family-agnostic: pass ``model_family`` to get the right LoRA target modules
  for Qwen, Gemma, LLaMA, Mistral, etc.  Adding a new family = one dict entry.
• Graceful degradation: WandB and results_db are optional — training always works
  without them (no internet, CI, unit tests).
• No hidden state: every function is pure or explicitly takes all its inputs as args.
  Nothing is stored in module-level globals except the DB import shim.

Usage example
-------------
    from training_scaffold import (
        setup_hw_opts, load_target_model, wrap_lora_draft, load_lora_draft,
        setup_wandb, save_checkpoint, try_resume, write_train_step, make_optimizer,
        lora_target_modules,
    )

    device = torch.device("cuda")
    setup_hw_opts(device)
    target = load_target_model(args.target, dtype=torch.bfloat16)
    draft_base = load_target_model(args.draft, dtype=torch.bfloat16, frozen=False)
    draft = wrap_lora_draft(draft_base, r=8, alpha=16,
                            target_modules=lora_target_modules("qwen"))
    optimizer = make_optimizer(draft, lr=3e-5)
    wandb_run = setup_wandb(project="distillspec", run_name="kl-gsm8k", config=vars(args))

    resume = try_resume(args.output)
    start_step = resume["step"] if resume else 0
    if resume:
        draft = load_lora_draft(draft_base, resume["adapter_path"])

    for step, batch in enumerate(data[start_step:], start=start_step):
        loss = my_loss_fn(draft, target, batch)
        ...
        write_train_step(label="kl-gsm8k", loss_name="forward_kl",
                         step=step+1, loss=loss.item(), split="train")
        if (step + 1) % args.save_every == 0:
            save_checkpoint(draft, tokenizer, args.output, step+1,
                            extra={"baseline_alpha": 0.73})
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Model family registry
# ---------------------------------------------------------------------------

# LoRA target modules differ per model architecture.
# Add a new family here — no other file needs changing.
_LORA_MODULES: dict[str, list[str]] = {
    "gpt2":    ["c_attn", "c_proj"],   # Conv1D naming (combined QKV + output)
    "qwen":    ["q_proj", "v_proj"],
    "gemma":   ["q_proj", "k_proj", "v_proj", "o_proj"],
    "llama":   ["q_proj", "v_proj"],
    "mistral": ["q_proj", "v_proj"],
    "phi":     ["q_proj", "v_proj", "k_proj"],
    "falcon":  ["query_key_value"],
}


def lora_target_modules(model_family: str) -> list[str]:
    """Return the LoRA target module names for a given model family.

    Falls back to the Qwen defaults with a warning if the family is unknown.
    """
    family = model_family.lower()
    if family not in _LORA_MODULES:
        logging.getLogger(__name__).warning(
            "Unknown model_family=%r — falling back to qwen LoRA targets %s. "
            "Add an entry to _LORA_MODULES in training_scaffold.py.",
            family, _LORA_MODULES["qwen"],
        )
        return _LORA_MODULES["qwen"]
    return _LORA_MODULES[family]


# ---------------------------------------------------------------------------
# Hardware setup
# ---------------------------------------------------------------------------

def setup_hw_opts(device: torch.device) -> None:
    """Enable TF32, cuDNN auto-tune, and (optionally) torch.compile.

    Call once at the start of any training script, before loading models.
    Safe to call on CPU — all CUDA-specific opts are guarded.

    Why TF32 + cuDNN benchmark are always on (Ampere+ GPUs):
      • TF32 matmuls: ~3× faster than FP32 on Ampere, no accuracy impact for training
      • cuDNN benchmark: picks the fastest kernel for each input shape at first use;
        pays ~5 s once, saves 10-30% every subsequent forward pass
    """
    log = logging.getLogger(__name__)
    if device.type != "cuda":
        return
    try:
        gpu_name = torch.cuda.get_device_name(device)
        # TF32: Ampere (A100, RTX 30xx, RTX 40xx) and later
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        log.info(
            "HW opts: TF32=on (Ampere+)  cuDNN-bench=on  GPU=%s", gpu_name
        )
    except Exception as exc:
        log.debug("setup_hw_opts: %s", exc)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _resolve_dtype(dtype) -> torch.dtype:
    """Accept a torch.dtype, string ('bf16', 'fp16', 'fp32'), or None → torch.dtype."""
    if isinstance(dtype, torch.dtype):
        return dtype
    mapping = {
        "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
        "fp16": torch.float16,  "float16":  torch.float16,
        "fp32": torch.float32,  "float32":  torch.float32,
    }
    if isinstance(dtype, str) and dtype.lower() in mapping:
        return mapping[dtype.lower()]
    return torch.bfloat16  # safe default


def load_target_model(
    model_path: str,
    dtype=torch.bfloat16,
    load_in_4bit: bool = False,
    frozen: bool = True,
    device: Optional[torch.device] = None,
) -> AutoModelForCausalLM:
    """Load a target (or base) model.

    Parameters
    ----------
    model_path   : HuggingFace model ID or local path
    dtype        : torch.dtype or string ('bf16', 'fp16', 'fp32')
    load_in_4bit : Load in 4-bit NF4 via bitsandbytes (for Colab T4 + 8B models)
    frozen       : If True, disable gradients on all parameters (target model)
    device       : Target device; auto-detected if None
    """
    log = logging.getLogger(__name__)
    torch_dtype = _resolve_dtype(dtype)
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError:
            raise SystemExit(
                "bitsandbytes is required for load_in_4bit=True.\n"
                "Install with:  pip install bitsandbytes"
            )
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            quantization_config=bnb_cfg,
            device_map="auto",
            use_safetensors=True,
        )
        log.info("Loaded %s in 4-bit NF4 (QLoRA mode)", model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            dtype=torch_dtype,          # transformers >= 4.51 (required for Qwen3)
            low_cpu_mem_usage=True,
            device_map=None,
            use_safetensors=True,
        ).to(dev)
        log.info("Loaded %s  dtype=%s  device=%s", model_path, torch_dtype, dev)

    if frozen:
        for p in model.parameters():
            p.requires_grad_(False)
        model.eval()

    return model


def load_tokenizer(model_path: str) -> AutoTokenizer:
    """Load tokenizer; sets pad_token = eos_token if absent."""
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


# ---------------------------------------------------------------------------
# LoRA helpers
# ---------------------------------------------------------------------------

def wrap_lora_draft(
    base_model: AutoModelForCausalLM,
    r: int = 8,
    alpha: int = 16,
    target_modules: Optional[list[str]] = None,
    model_family: str = "qwen",
) -> "PeftModel":
    """Wrap a base model with a fresh LoRA adapter for draft training.

    Picks target_modules from the family registry if not supplied explicitly.
    """
    from peft import LoraConfig, TaskType, get_peft_model  # type: ignore
    modules = target_modules or lora_target_modules(model_family)
    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(base_model, cfg)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.getLogger(__name__).info(
        "LoRA wrapped: r=%d  alpha=%d  modules=%s  trainable_params=%s",
        r, alpha, modules, f"{n:,}",
    )
    return model


def load_lora_draft(
    base_model: AutoModelForCausalLM,
    adapter_path: str,
) -> "PeftModel":
    """Load an existing LoRA adapter onto a base model (trainable)."""
    from peft import PeftModel  # type: ignore
    logging.getLogger(__name__).info("Loading LoRA adapter from: %s", adapter_path)
    return PeftModel.from_pretrained(base_model, adapter_path, is_trainable=True)


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

def make_optimizer(model: torch.nn.Module, lr: float = 3e-5) -> torch.optim.AdamW:
    """Standard AdamW over trainable parameters only."""
    return torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
    )


# ---------------------------------------------------------------------------
# Checkpoint save / resume
# ---------------------------------------------------------------------------

def save_checkpoint(
    model,
    tokenizer: AutoTokenizer,
    output_dir: str,
    step: int,
    extra: Optional[dict[str, Any]] = None,
    subdir: str = "ckpt_latest",
) -> None:
    """Save LoRA adapter + resume_state.json to <output_dir>/<subdir>/.

    resume_state.json stores step + any extra key/value pairs (e.g.
    baseline_alpha, best_val_loss) needed to restore state on restart.

    Safe to call from any training script — passes silently on failure
    so a disk-full or permission error never kills training.
    """
    log = logging.getLogger(__name__)
    dest = Path(output_dir) / subdir
    try:
        dest.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(dest))
        tokenizer.save_pretrained(str(dest))
        state = {"step": step, **(extra or {})}
        with open(dest / "resume_state.json", "w", encoding="utf-8") as f:
            json.dump(state, f)
        log.info("[ckpt] Saved checkpoint at step %d → %s", step, dest)
    except Exception as exc:
        log.warning("[ckpt] Checkpoint save failed at step %d: %s", step, exc)


def try_resume(
    output_dir: str,
    subdir: str = "ckpt_latest",
) -> Optional[dict[str, Any]]:
    """Check for a crash-resume checkpoint.

    Returns a dict with at least ``{"step": N, "adapter_path": str}`` and any
    extra keys stored by ``save_checkpoint``, or None if no checkpoint exists.

    Usage::

        resume = try_resume(args.output)
        start_step = resume["step"] if resume else 0
        if resume:
            draft = load_lora_draft(base, resume["adapter_path"])
    """
    log = logging.getLogger(__name__)
    ckpt_dir = Path(output_dir) / subdir
    adapter_file = ckpt_dir / "adapter_model.safetensors"
    state_file = ckpt_dir / "resume_state.json"

    if not adapter_file.exists():
        return None

    state: dict[str, Any] = {}
    if state_file.exists():
        try:
            with open(state_file, encoding="utf-8") as f:
                state = json.load(f)
        except Exception as exc:
            log.warning("[resume] Could not read resume_state.json: %s", exc)

    state["adapter_path"] = str(ckpt_dir)
    step = int(state.get("step", 0))
    log.info("[resume] Found checkpoint at step %d in %s", step, ckpt_dir)
    return state


# ---------------------------------------------------------------------------
# Results DB integration
# ---------------------------------------------------------------------------

# Lazy import: resolved once, then reused.  None if db/ is unreachable.
_results_db = None
_results_db_resolved = False


def _get_db():
    """Import results_db lazily; return None if unavailable."""
    global _results_db, _results_db_resolved
    if _results_db_resolved:
        return _results_db
    _results_db_resolved = True
    _db_dir = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "db")
    )
    if os.path.isdir(_db_dir) and _db_dir not in sys.path:
        sys.path.insert(0, _db_dir)
    try:
        import results_db as _rdb  # type: ignore
        _results_db = _rdb
    except ImportError:
        pass
    return _results_db


def write_train_step(
    label: str,
    loss_name: str,
    step: int,
    loss: float,
    split: str = "train",
    learning_rate: Optional[float] = None,
    lora_rank: Optional[int] = None,
    accept_weight: Optional[float] = None,
) -> None:
    """Write one training or validation step to results.db.

    Silently no-ops if the DB is unavailable (no internet, unit tests, etc.).
    ``split`` is "train" (solid line) or "val" (dashed line) in the dashboard.

    For online adapt runs: pass loss=KL_loss for split="train" and
    loss=(1 - eval_alpha) for split="val" so the val line trends downward
    like the offline runs.
    """
    db = _get_db()
    if db is None:
        return
    try:
        db.insert_train_step(
            label=label,
            loss_name=loss_name,
            step=step,
            loss=loss,
            split=split,
            learning_rate=learning_rate,
            lora_rank=lora_rank,
            accept_weight=accept_weight,
        )
    except Exception:
        pass  # never crash training on a DB write failure


# ---------------------------------------------------------------------------
# WandB integration
# ---------------------------------------------------------------------------

def setup_wandb(
    project: str,
    run_name: str,
    config: Optional[dict] = None,
    wandb_dir: Optional[str] = None,
    tags: Optional[list[str]] = None,
):
    """Initialise a WandB run.  Returns the wandb module, or None if unavailable.

    Usage::

        wandb = setup_wandb("distillspec", "kl-gsm8k", config=vars(args))
        if wandb:
            wandb.log({"train/loss": loss.item()}, step=step)
            # at end:
            wandb.finish()
    """
    log = logging.getLogger(__name__)
    try:
        import wandb as _wandb  # type: ignore
    except ImportError:
        log.warning("wandb not installed — logging to stdout only")
        return None

    # Default wandb dir: gbv-research/db/wandb/
    if wandb_dir is None:
        wandb_dir = os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "db", "wandb")
        )
    os.makedirs(wandb_dir, exist_ok=True)

    try:
        _wandb.init(
            project=project,
            name=run_name,
            config=config or {},
            dir=wandb_dir,
            tags=tags or [],
        )
        log.info("WandB run: %s/%s", project, run_name)
        return _wandb
    except Exception as exc:
        log.warning("WandB init failed (%s) — continuing without WandB", exc)
        return None
