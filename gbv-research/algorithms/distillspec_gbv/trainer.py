"""
DistillSpec training loop — model-family-agnostic.

Trains a small draft model via knowledge distillation from a frozen large
target (teacher) model.  Supports multiple loss objectives and any model
family registered in algorithms/distillspec_gbv/model_families/.

Quick start (laptop, Qwen family):
    python -m algorithms.distillspec_gbv.trainer \\
        --loss forward_kl --steps 1000 \\
        --dataset ../../core/datasets/raw/gsm8k_train.jsonl

Colab / server (Qwen3 8B teacher):
    python -m algorithms.distillspec_gbv.trainer \\
        --model_family qwen \\
        --draft Qwen/Qwen3-0.6B --target Qwen/Qwen3-8B \\
        --loss ebe --steps 5000 \\
        --dataset ../../core/datasets/raw/gsm8k_train.jsonl \\
        --output ../../db/checkpoints/ebe-qwen3-8b

To add a new model family (e.g. Gemma):
    See docs/ADDING_A_MODEL_FAMILY.md

To add a new loss (e.g. alpha-divergence):
    See docs/ADDING_A_LOSS.md
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import random
import shutil

# Apply OMP thread count BEFORE torch is imported so the setting takes effect.
# Set via --omp_threads CLI arg (experiment.py passes it for CPU server configs).
# Defaults to leaving PyTorch's auto-detection in place (no override).
def _apply_omp_threads():
    import argparse as _ap
    _p = _ap.ArgumentParser(add_help=False)
    _p.add_argument("--omp_threads", type=int, default=0)
    _a, _ = _p.parse_known_args()
    if _a.omp_threads > 0:
        os.environ.setdefault("OMP_NUM_THREADS", str(_a.omp_threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(_a.omp_threads))
_apply_omp_threads()

# ── Offline mode ─────────────────────────────────────────────────────────────
# Prevents HuggingFace Hub network calls on already-cached / air-gapped setups.
# Override with TRANSFORMERS_OFFLINE=0 before running for first-time downloads.
_hf_offline_was_set = "TRANSFORMERS_OFFLINE" in os.environ
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
if not _hf_offline_was_set and os.environ.get("TRANSFORMERS_OFFLINE") == "1":
    print(
        "[trainer] HF offline mode ON. "
        "Set TRANSFORMERS_OFFLINE=0 for first-time model downloads."
    )

# Reduce CUDA allocator fragmentation on small GPUs.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

# Silence one specific, benign PEFT warning that fires on every adapter save:
#   "Could not find a config file in Qwen/Qwen2.5-0.5B - will assume that the
#    vocabulary was not modified."
# PEFT looks up the base-model config by its HF id to check vocab alignment; in
# offline mode it can't resolve the id to a local dir, so it warns and (correctly)
# assumes the vocab is unchanged — which is true for LoRA (we never touch the
# embedding/vocab). The warning is harmless but fires 17x/run and clutters logs.
# Filter ONLY this exact message — all other PEFT/transformers warnings still show.
import warnings as _warnings
_warnings.filterwarnings(
    "ignore",
    message=r".*Could not find a config file.*will assume that the vocabulary was not modified.*",
)

import torch
import torch.nn.functional as F
import transformers
from torch.optim import AdamW
from peft import get_peft_model, LoraConfig, TaskType, PeftModel

from core.model_families import get_family, ModelFamily
from algorithms.distillspec_gbv.losses import get_loss, LossOutput
from algorithms.distillspec_gbv.losses import compute_tree_loss, TREE_LOSS_NAMES
from algorithms.distillspec_gbv.tree_training import (
    draft_tree_forward_with_grad, verify_tree_forward_grad,
)

# Optional: log training curves to results.db so the dashboard shows loss plots.
# Silently disabled if results_db can't be imported (e.g. wrong cwd or missing dep).
try:
    from db import results_db as _results_db
except Exception:
    _results_db = None


# ---------------------------------------------------------------------------
# Attention backend selection
# ---------------------------------------------------------------------------

def _select_attention_backend() -> str:
    """Pick the fastest attention backend available: FA2 > SDPA > eager."""
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        pass
    if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        return "sdpa"
    return "eager"


_ATTN_IMPL = _select_attention_backend()


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DistillSpec training — model-family-agnostic distillation trainer."
    )

    # ── Models ───────────────────────────────────────────────────────────────
    p.add_argument("--model_family", default="qwen",
                   help="Model family key (registered in model_families/__init__.py). "
                        "Determines LoRA targets, temperature handling, and prompt format. "
                        "Default: qwen.  Available: qwen, gemma (stub).")
    p.add_argument("--draft",  default=None,
                   help="Draft model HuggingFace ID or local path. "
                        "Defaults to family.default_draft_model_id.")
    p.add_argument("--target", default=None,
                   help="Frozen teacher HuggingFace ID or local path. "
                        "Defaults to family.default_target_model_id.")
    p.add_argument("--teacher_temp", type=float, default=0.8,
                   help="Sampling temperature for the teacher's generate() call. "
                        "The trainer undoes any family-specific temperature pre-scaling "
                        "via ModelFamily.recover_raw_logits().")

    # ── Training ─────────────────────────────────────────────────────────────
    p.add_argument("--loss", default="forward_kl",
                   choices=[
                       "forward_kl", "reverse_kl", "jsd", "l1", "ebe", "ebe_single",
                       # Tree-structured losses — on-policy, verifier-specific objectives
                       # Node-level divergence variants (same tree data, different divergence):
                       "kl_tree",      # forward KL(p ∥ q) at each node — mode-covering
                       "rev_kl_tree",  # reverse KL(q ∥ p) at each node — mode-seeking
                       "jsd_tree",     # symmetric JSD at each node — bounded, stable
                       # Verifier-aligned tree losses (non-OT):
                       "bv_tree", "gbv_tree", "traversal_tree",
                       # Verifier-aligned tree losses (OT-based): use each verifier's
                       # own closed-form per-node acceptance probability α_V from
                       # node.py to optimise E[τ_V] = Σ_i Π_{j≤i} α_V(p_j,q_j,K).
                       "naive_tree",     # Chen/Leviathan single-path verifier
                       "nss_tree",       # Naive Speculative Sampling
                       "specinfer_tree", # SpecInfer K-iter rejection
                       "spectr_tree",    # SpecTr K-SEQ (ρ detached — see _alpha_spectr)
                       "khisti_tree",    # Khisti canonical decomp (LP-free surrogate)
                       # On-policy EBE (token-level, ablation of flat EBE off-policy issue):
                       "ebe_tree",
                   ],
                   help="Distillation loss objective.  Tree losses (*_tree) use the "
                        "student's own draft tree as training data (on-policy).  "
                        "Requires --tree_K and --tree_L.")
    p.add_argument("--tree_K", type=int, default=4,
                   help="Number of i.i.d. draft paths for tree losses (default 4, "
                        "should match inference K).")
    p.add_argument("--tree_L", type=int, default=8,
                   help="Draft block length for tree losses (default 8, "
                        "should match inference L and EBE DEFAULT_BLOCK_LEN).")
    p.add_argument("--steps",  type=int, default=1000,
                   help="Total training steps.")
    p.add_argument("--lr",     type=float, default=3e-5)
    p.add_argument("--warmup_steps", type=int, default=None,
                   help="Linear warmup steps (DistillSpec/EAGLE convention). "
                        "Default: 10%% of --steps.  Set 0 to disable.")
    p.add_argument("--max_new_tokens", type=int, default=80,
                   help="Max tokens the teacher generates per step.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_shuffle", action="store_true",
                   help="Disable dataset shuffling.")
    p.add_argument("--grad_clip", type=float, default=1.0,
                   help="Gradient norm clipping threshold (0 = disabled).")
    p.add_argument("--grad_accum", type=int, default=4,
                   help="Gradient accumulation steps (effective batch = grad_accum × 1 prompt). "
                        "Default 4 smooths the noisy per-step KL loss without extra VRAM.")

    # ── Sweep-friendly hyperparameters ───────────────────────────────────────
    p.add_argument("--ebe_kl_weight", type=float, default=0.1,
                   help="KL regulariser weight in EBE loss.  "
                        "L_total = L_EBE + ebe_kl_weight * L_KL.  Sweep: 0.01–0.5.")
    p.add_argument("--jsd_alpha", type=float, default=0.5,
                   help="JSD interpolation weight (0=reverse KL, 1=forward KL).  Sweep: 0.1–0.9.")

    # ── LoRA ─────────────────────────────────────────────────────────────────
    p.add_argument("--lora_r",       type=int,   default=8)
    p.add_argument("--lora_alpha",   type=int,   default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05,
                   help="LoRA adapter dropout probability.  Sweep: 0.0–0.2.")
    p.add_argument("--no_lora",      action="store_true",
                   help="Full SFT instead of LoRA (needs more VRAM).")

    # ── Data ─────────────────────────────────────────────────────────────────
    p.add_argument("--dataset", default=None,
                   help="Path to JSONL training file (field: 'prompt').")
    p.add_argument("--val_dataset", default=None,
                   help="Separate JSONL validation file.  Overrides --val_split.")
    p.add_argument("--val_split", type=float, default=0.1,
                   help="Fraction of training data held out for validation.")
    p.add_argument("--val_every", type=int, default=50,
                   help="Compute validation loss every N steps (0 to disable).")
    p.add_argument("--max_train_prompts", type=int, default=0,
                   help="Cap training dataset to this many prompts (0 = no cap). "
                        "Smoke mode passes 20 — enough for 10 steps with shuffle variety, "
                        "tokenises in ~0.3s vs ~2min for the full 6726-prompt set.")

    # ── Output / checkpointing ────────────────────────────────────────────────
    p.add_argument("--output", default=None,
                   help="Checkpoint directory.  Defaults to "
                        "../../db/checkpoints/<loss>-<family>-<steps>steps.")
    p.add_argument("--save_every", type=int, default=100,
                   help="Overwrite crash-safe checkpoint every N steps.")
    p.add_argument("--milestone_every", type=int, default=200,
                   help="Save a permanent numbered checkpoint every N steps.")
    p.add_argument("--max_checkpoints", type=int, default=5,
                   help="Max milestone checkpoints to keep on disk.")
    p.add_argument("--merge_only", action="store_true",
                   help="Merge LoRA adapter and exit (no training).")
    p.add_argument("--adapter", default=None,
                   help="Path to LoRA adapter for --merge_only.")

    # ── Logging / W&B ────────────────────────────────────────────────────────
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--wandb_project", default="distillspec")
    p.add_argument("--wandb_entity",  default=None)
    p.add_argument("--wandb_group",   default=None)
    p.add_argument("--run_label",     default="",
                   help="short tag prepended to the W&B run name so runs from "
                        "different configs/teacher sizes are distinguishable, e.g. "
                        "'4B-T4' -> '4B-T4-kl_qwen_500steps'. "
                        "Set via logging.run_label in the YAML config.")
    p.add_argument("--no_wandb",      action="store_true")

    # ── Health checks ────────────────────────────────────────────────────────
    p.add_argument("--nan_action", default="stop",
                   choices=["stop", "skip", "warn"])
    p.add_argument("--health_every", type=int, default=100)
    p.add_argument("--ppl_check_every", type=int, default=200)
    p.add_argument("--ppl_threshold",   type=float, default=1.25)
    p.add_argument("--early_stop_patience", type=int, default=0)

    # ── Hardware ─────────────────────────────────────────────────────────────
    p.add_argument("--load_in_4bit", action="store_true",
                   help="Load teacher in 4-bit NF4 (QLoRA). Saves VRAM for 8B teachers.")
    p.add_argument("--compile", action="store_true",
                   help="torch.compile() the draft model (PyTorch 2.0+).")
    p.add_argument("--hw_tier", default="laptop",
                   choices=["laptop", "cpu", "colab_lite", "colab", "a100"],
                   help="Hardware tier (default: laptop). Controls PPL sample size. "
                        "Set by experiment.py from YAML hardware.hw_tier. "
                        "laptop=4GB GPU; cpu=CPU-only server (ATS/AIP, GPT-2 convergence); "
                        "colab=T4 GPU (Kaggle or Colab); a100=A100 (paper results).")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                   help="Compute device (default: auto = cuda if available else cpu). "
                        "Set '--device cpu' to force CPU even on a GPU machine — lets a "
                        "GPU laptop smoke-test the exact CPU code path before running on "
                        "the CPU-only ATS/AIP server. Set by experiment.py from YAML "
                        "hardware.device, or override on the CLI.")
    p.add_argument("--omp_threads", type=int, default=0,
                   help="Override OMP_NUM_THREADS/MKL_NUM_THREADS for CPU inference. "
                        "0 = let PyTorch auto-detect (default). "
                        "Set via hardware.omp_threads in the YAML config (e.g. 16 for "
                        "a 32-core Xeon server). Has no effect on GPU runs.")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_prompts(path: str | None) -> list[str]:
    """Load prompts from a JSONL file.  Supports 'prompt', 'question', 'instruction' fields."""
    if path is None:
        raise ValueError("--dataset is required.  "
                         "Point it to a JSONL file with a 'prompt' field.")
    prompts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = (obj.get("prompt") or obj.get("question")
                    or obj.get("instruction") or "")
            if text:
                prompts.append(text.strip())
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    print(f"[data] Loaded {len(prompts)} training prompts from {path}")
    return prompts


# ---------------------------------------------------------------------------
# Merge utility
# ---------------------------------------------------------------------------

def _resolve_adapter_dir(output_dir: str) -> str:
    """Return a subdirectory of *output_dir* that holds a complete LoRA adapter.

    Training saves rolling checkpoints to ckpt_latest/ and (optionally) the
    best val-loss snapshot to ckpt_best/.  The final root-level save can be
    skipped when a resumed run detects training is already complete, leaving
    adapter weights only under ckpt_latest/.  Merge steps pass the output root,
    so we probe root → ckpt_best → ckpt_latest before failing.
    """
    for sub in ("", "ckpt_best", "ckpt_latest"):
        d = output_dir if not sub else os.path.join(output_dir, sub)
        if os.path.isfile(os.path.join(d, "adapter_config.json")):
            if sub:
                print(f"[merge] adapter not at output root; using {d}")
            return d
    raise FileNotFoundError(
        f"No adapter_config.json under {output_dir} "
        f"(checked root, ckpt_best/, ckpt_latest/). "
        f"Re-run the training step for this loss."
    )


def merge_lora_and_save(draft_model_id: str, adapter_path: str) -> None:
    """Merge a LoRA adapter into the base model and save to <adapter_path>_merged/."""
    adapter_dir = _resolve_adapter_dir(adapter_path)

    print(f"Loading base: {draft_model_id}")
    base = transformers.AutoModelForCausalLM.from_pretrained(
        draft_model_id, dtype=torch.bfloat16, low_cpu_mem_usage=True)

    print(f"Loading LoRA: {adapter_dir}")
    # On Windows + OneDrive, PEFT's _get_peft_type calls os.path.isfile on
    # the adapter_config.json path.  OneDrive virtualises files so
    # os.path.isfile returns False even when the file is present, causing
    # PEFT to fall through to hf_hub_download → validate_repo_id which
    # rejects Windows paths (spaces, backslashes, length > 96 chars) with
    # HFValidationError.  Workaround: read adapter_config.json ourselves and
    # pass the constructed LoraConfig via `config=` so _get_peft_type is
    # never called.
    config_path = os.path.join(adapter_dir, "adapter_config.json")
    with open(config_path, encoding="utf-8") as _f:
        _cfg = json.load(_f)
    # In PEFT 0.19.1 every key in adapter_config.json is a valid LoraConfig
    # constructor param — no stripping needed; just forward the whole dict.
    lora_config = LoraConfig(**_cfg)

    model  = PeftModel.from_pretrained(base, adapter_dir, config=lora_config)
    merged = model.merge_and_unload()
    out    = adapter_path + "_merged"
    merged.save_pretrained(out)
    transformers.AutoTokenizer.from_pretrained(draft_model_id).save_pretrained(out)
    print(f"Merged → {out}")


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _save_checkpoint(
    draft_model, optimizer, step: int, output_dir: str,
    losses: list, accept_weights: list,
    milestone: bool = False, max_checkpoints: int = 5,
    best_val_loss: float = None, val_no_improve_count: int = None,
    no_lora: bool = False,
) -> None:
    """
    Save crash-safe checkpoint to <output_dir>/ckpt_latest/.
    If milestone=True also saves to <output_dir>/ckpt_step_{step:05d}/.
    """
    ckpt_dir = os.path.join(output_dir, "ckpt_latest")
    os.makedirs(ckpt_dir, exist_ok=True)

    if no_lora:
        torch.save(draft_model.state_dict(), os.path.join(ckpt_dir, "model.pt"))
    else:
        draft_model.save_pretrained(ckpt_dir)

    torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.pt"))

    state = {
        "step": step,
        "loss_history": losses[-500:],
        "accept_weight_history": accept_weights[-500:],
    }
    if best_val_loss is not None:
        state["best_val_loss"]      = best_val_loss
        state["val_no_improve_count"] = val_no_improve_count
    with open(os.path.join(output_dir, "training_state.json"), "w") as f:
        json.dump(state, f)

    if milestone:
        dest = os.path.join(output_dir, f"ckpt_step_{step:05d}")
        shutil.copytree(ckpt_dir, dest, dirs_exist_ok=True)
        # Prune oldest milestones if over limit
        if max_checkpoints > 0:
            milestones = sorted(
                d for d in os.listdir(output_dir)
                if d.startswith("ckpt_step_")
                and os.path.isdir(os.path.join(output_dir, d))
            )
            while len(milestones) > max_checkpoints:
                shutil.rmtree(os.path.join(output_dir, milestones.pop(0)),
                              ignore_errors=True)


# ---------------------------------------------------------------------------
# Validation / PPL helpers
# ---------------------------------------------------------------------------

def _compute_val_loss(
    draft_model, target_model, val_prompts: list, tok_cache: dict,
    tokenizer, args, device: str, family: ModelFamily,
    max_prompts: int = 10,
) -> tuple[float, float | None]:
    """
    Compute validation loss on up to max_prompts held-out prompts.
    Returns (mean_loss, mean_accept_weight_or_None).
    """
    sample = val_prompts[:max_prompts]
    val_losses, val_aws = [], []

    draft_model.eval()
    with torch.no_grad():
        for prompt in sample:
            if prompt not in tok_cache:
                formatted = family.format_prompt(prompt, tokenizer)
                tok_cache[prompt] = tokenizer(formatted, return_tensors="pt").input_ids

            prompt_ids = tok_cache[prompt].to(device)
            plen = prompt_ids.shape[1]

            gen_out = target_model.generate(
                prompt_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.teacher_temp,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
            )
            full_ids = gen_out.sequences
            gen_len  = full_ids.shape[1] - plen
            if gen_len == 0:
                continue

            # Single parallel forward pass — raw logits, no recover_raw_logits needed.
            t_log = target_model(full_ids).logits[0, plen - 1:-1, :].float()

            student_logits = draft_model(full_ids).logits[:, :-1, :].float()
            s_log   = student_logits[:, plen - 1:, :].squeeze(0)
            gen_ids = full_ids[0, plen:]

            # Tree losses cannot be evaluated with get_loss() — they require a
            # structured K-path tree built from the student's own samples, which
            # would be expensive to build per val prompt and unnecessary for a
            # health metric.  Use forward KL as a proxy: it is a tight upper
            # bound on the tree loss's language-quality component and will
            # correctly signal PPL collapse / distribution drift.
            val_loss_name = "forward_kl" if args.loss in TREE_LOSS_NAMES else args.loss
            out = get_loss(val_loss_name, s_log, t_log, token_ids=gen_ids,
                          kl_weight=args.ebe_kl_weight, alpha=args.jsd_alpha)
            val_losses.append(out.loss.item())
            if out.accept_weight is not None:
                val_aws.append(out.accept_weight)

    draft_model.train()
    mean_loss = sum(val_losses) / len(val_losses) if val_losses else float("nan")
    mean_aw   = sum(val_aws)   / len(val_aws)    if val_aws    else None
    return mean_loss, mean_aw


def _compute_ppl(
    draft_model, prompts: list, tok_cache: dict,
    tokenizer, device: str, family: ModelFamily,
    max_prompts: int = 5,
) -> float:
    """
    Compute perplexity of the draft model on a small sample of prompts.
    Used as a secondary health metric — primary metric is val_loss.
    """
    sample = prompts[:max_prompts]
    nlls = []
    draft_model.eval()
    with torch.no_grad():
        for prompt in sample:
            if prompt not in tok_cache:
                formatted = family.format_prompt(prompt, tokenizer)
                tok_cache[prompt] = tokenizer(formatted, return_tensors="pt").input_ids
            ids = tok_cache[prompt].to(device)
            if ids.shape[1] < 2:
                continue
            logits = draft_model(ids).logits.float()
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = ids[:, 1:].contiguous()
            nll = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1), reduction="mean")
            nlls.append(nll.item())
    draft_model.train()
    return math.exp(sum(nlls) / len(nlls)) if nlls else float("nan")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _tree_training_step(
    draft_model,
    target_model,
    prompt_ids: torch.Tensor,
    loss_name: str,
    K: int,
    L: int,
    temperature: float,
) -> torch.Tensor:
    """
    One training step for tree-structured losses (kl_tree / bv_tree / gbv_tree / traversal_tree).

    Steps:
      1. Student builds a draft tree via iid_draft (no grad — sampling is discrete).
      2. Target scores the same tree via target_tree_pass (no grad — teacher is frozen).
      3. Student re-runs over the fixed tree tokens WITH grad (draft_tree_forward_with_grad).
      4. Tree loss is computed using q_probs_dict_grad + p_probs_dict.

    Returns scalar loss tensor with grad_fn attached.
    """
    from algorithms.distillspec_gbv.verifiers.draft_generator import iid_draft, target_tree_pass

    device = prompt_ids.device

    with torch.no_grad():
        # ── 1. Student builds draft tree ───────────────────────────────────────
        q_out_init   = draft_model(prompt_ids, use_cache=True, return_dict=True)
        q_cache_init = q_out_init.past_key_values
        pending      = torch.multinomial(
            F.softmax(q_out_init.logits[0, -1] / temperature, dim=-1), 1
        ).unsqueeze(0)                                      # [1, 1]
        q_paths, _, _ = iid_draft(
            draft_model, q_cache_init, pending,
            K=K, L=L, q_temp=temperature,
        )

        # ── 2. Target scores the draft tree ───────────────────────────────────
        p_out_init   = target_model(prompt_ids, use_cache=True, return_dict=True)
        p_cache_init = p_out_init.past_key_values
        _, _, _, p_probs_dict = target_tree_pass(
            target_model, p_cache_init, q_paths,
            K=K, L=L, p_temp=temperature,
        )
        # p_probs_dict: {prefix → [V]}, all detached (target is frozen)

    # ── 3. Student re-runs on fixed tree WITH grad ─────────────────────────────
    q_probs_dict_grad = draft_tree_forward_with_grad(
        draft_model, prompt_ids, q_paths, L=L, K=K, q_temp=temperature,
    )

    # ── 4. Tree loss ───────────────────────────────────────────────────────────
    return compute_tree_loss(
        loss_name, q_probs_dict_grad, p_probs_dict, q_paths, L=L, K=K,
    )


def main() -> None:
    args = parse_args()

    family: ModelFamily = get_family(args.model_family)

    # Fill model defaults from family if not provided
    if args.draft  is None:
        args.draft  = family.default_draft_model_id
    if args.target is None:
        args.target = family.default_target_model_id
    if args.output is None:
        args.output = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "db", "checkpoints",
            f"{args.loss}-{family.name}-{args.steps}steps"
        )

    if args.merge_only:
        merge_lora_and_save(args.draft, args.adapter)
        return

    # Device selection honours --device so a GPU laptop can still smoke-test the
    # CPU code path before shipping long runs to the CPU-only ATS/AIP server.
    #   auto (default): cuda if available else cpu  (original behaviour)
    #   cpu           : force CPU even when CUDA is present (CPU-path smoke test)
    #   cuda          : require CUDA (errors clearly if unavailable)
    _dev_arg = getattr(args, "device", "auto")
    if _dev_arg == "cpu":
        device = "cpu"
    elif _dev_arg == "cuda":
        if not torch.cuda.is_available():
            print("[FATAL] --device cuda requested but no CUDA GPU is available.",
                  file=sys.stderr)
            sys.exit(1)
        device = "cuda"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device        : {device}" + ("  (--device cpu: forced CPU path)"
          if _dev_arg == "cpu" and torch.cuda.is_available() else ""))
    print(f"Model family  : {family.name}")
    print(f"Draft         : {args.draft}  ({'full SFT' if args.no_lora else f'LoRA r={args.lora_r}'})")
    print(f"Target        : {args.target}  (frozen bfloat16)")
    print(f"Loss          : {args.loss}")
    print(f"Attn backend  : {_ATTN_IMPL}")
    print(f"Steps         : {args.steps}")
    print(f"Teacher temp  : {args.teacher_temp}")
    if args.dataset:
        print(f"Dataset       : {args.dataset}")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.target)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── Load target (frozen teacher) ──────────────────────────────────────────
    print("\nLoading target (frozen)...")
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        # Build a max_memory map so device_map="auto" uses ALL visible GPUs,
        # and reserves 1 GiB on each for activations.  This prevents the new
        # transformers (4.51+) lazy-load path from packing everything onto
        # GPU 0 and OOM-ing at 95% during the BF16→NF4 materialization step.
        import torch as _t
        _max_mem = {i: f"{int(_t.cuda.get_device_properties(i).total_memory / 1024**3) - 1}GiB"
                    for i in range(_t.cuda.device_count())}
        _max_mem["cpu"] = "24GiB"   # allow CPU offload if model > all GPUs combined
        target_model = transformers.AutoModelForCausalLM.from_pretrained(
            args.target,
            quantization_config=bnb_cfg,
            device_map="auto",
            max_memory=_max_mem,
            low_cpu_mem_usage=True,   # load to CPU first → quantize → move to GPU
            attn_implementation=_ATTN_IMPL)
    elif device == "cpu":
        # Forced-CPU path (--device cpu): load the teacher on CPU even when a GPU
        # is present, so a GPU laptop can smoke-test the exact CPU code path that
        # runs on the ATS/AIP server.  No device_map="auto" (that would grab the GPU).
        target_model = transformers.AutoModelForCausalLM.from_pretrained(
            args.target, dtype=torch.float32,   # fp32 on CPU (bf16 CPU matmul is slow/unsupported)
            low_cpu_mem_usage=True,
            attn_implementation=_ATTN_IMPL).to("cpu")
    else:
        import torch as _t
        _max_mem_bf16 = (
            {i: f"{int(_t.cuda.get_device_properties(i).total_memory / 1024**3) - 1}GiB"
             for i in range(_t.cuda.device_count())}
            if _t.cuda.is_available() else None
        )
        target_model = transformers.AutoModelForCausalLM.from_pretrained(
            args.target, dtype=torch.bfloat16,
            device_map="auto",
            max_memory=_max_mem_bf16,
            low_cpu_mem_usage=True,
            attn_implementation=_ATTN_IMPL)
    target_model.eval()
    for p in target_model.parameters():
        p.requires_grad_(False)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if device == "cuda":
        print(f"  VRAM after teacher: {torch.cuda.memory_allocated() / 1024**2:.0f} MB")
    else:
        print(f"  Teacher loaded on CPU")

    # ── Load draft ────────────────────────────────────────────────────────────
    print("Loading draft...")
    # fp32 on CPU (bf16 CPU matmul is slow/partially-unsupported), bf16 on GPU.
    _draft_dtype = torch.float32 if device == "cpu" else torch.bfloat16
    draft_base = transformers.AutoModelForCausalLM.from_pretrained(
        args.draft, dtype=_draft_dtype, low_cpu_mem_usage=True,
        attn_implementation=_ATTN_IMPL).to(device)
    if args.no_lora:
        draft_model = draft_base
        draft_model.gradient_checkpointing_enable()
        trainable = sum(p.numel() for p in draft_model.parameters())
        print(f"  Full SFT: {trainable / 1e6:.1f}M trainable params")
    else:
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=family.lora_target_modules(),  # ← family-specific
            bias="none",
        )
        draft_model = get_peft_model(draft_base, lora_cfg)
        draft_model.print_trainable_parameters()
    if device == "cuda":
        print(f"  VRAM (both): {torch.cuda.memory_allocated() / 1024**2:.0f} MB\n")
    else:
        print(f"  Draft + teacher loaded on CPU\n")

    optimizer = AdamW(
        [p for p in draft_model.parameters() if p.requires_grad], lr=args.lr)

    # Linear warmup → linear decay schedule (DistillSpec / EAGLE convention).
    # Zero memory and compute overhead; prevents the constant-LR overshoot seen
    # when val_loss stops improving after the first few hundred steps.
    #
    # IMPORTANT: the scheduler must be expressed in OPTIMIZER steps, not gradient
    # accumulation steps.  scheduler.step() is called only once every grad_accum
    # gradient steps, so num_training_steps must be steps // grad_accum.  If we
    # pass raw `args.steps` (gradient steps) the scheduler expects 1000 calls but
    # only receives 62 (for grad_accum=16) — training ends during warmup and the
    # LR never decays.
    from transformers import get_linear_schedule_with_warmup
    _ga = max(1, args.grad_accum)
    _n_opt_steps  = max(1, (args.steps + _ga - 1) // _ga)  # total optimizer updates
    _warmup_grad  = args.warmup_steps if args.warmup_steps is not None else max(1, args.steps // 10)
    _warmup_opt   = max(1, _warmup_grad // _ga)             # warmup in optimizer steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps  = _warmup_opt,
        num_training_steps= _n_opt_steps,
    )
    print(f"LR schedule   : linear warmup {_warmup_opt} opt-steps ({_warmup_grad} grad-steps)"
          f" → linear decay to 0 by opt-step {_n_opt_steps} (grad-step {args.steps})")

    torch.manual_seed(args.seed)

    # ── Crash-safe resume ─────────────────────────────────────────────────────
    ckpt_latest = os.path.join(args.output, "ckpt_latest")
    state_path  = os.path.join(args.output, "training_state.json")
    _ckpt_marker = "adapter_config.json" if not args.no_lora else "model.pt"
    _resuming = (os.path.exists(state_path)
                 and os.path.isdir(ckpt_latest)
                 and os.path.exists(os.path.join(ckpt_latest, _ckpt_marker)))

    start_step, losses, accept_weights = 0, [], []
    _resumed_best_val, _resumed_no_improve = None, 0

    if _resuming:
        with open(state_path, encoding="utf-8-sig") as f:
            saved = json.load(f)
        start_step     = saved.get("step", 0)
        losses         = saved.get("loss_history", [])
        accept_weights = saved.get("accept_weight_history", [])
        _raw_bvl       = saved.get("best_val_loss")
        _resumed_best_val   = float(_raw_bvl) if _raw_bvl is not None else None
        _resumed_no_improve = int(saved.get("val_no_improve_count", 0))
        if not args.no_lora:
            _tmp = PeftModel.from_pretrained(draft_base, ckpt_latest, is_trainable=True)
            # strict=False: _tmp wraps draft_base in a new adapter layer while
            # draft_model is already a PeftModel — keys differ by adapter name.
            # We are copying weights, not stacking a second adapter, so missing
            # or unexpected keys are expected and should not raise an error.
            draft_model.load_state_dict(_tmp.state_dict(), strict=False)
            del _tmp
        else:
            draft_model.load_state_dict(
                torch.load(os.path.join(ckpt_latest, "model.pt"), map_location=device))
        opt_path = os.path.join(ckpt_latest, "optimizer.pt")
        if os.path.exists(opt_path):
            optimizer.load_state_dict(torch.load(opt_path, map_location=device))
        print(f"\n[RESUME] Continuing from step {start_step + 1}/{args.steps}")
    else:
        os.makedirs(args.output, exist_ok=True)

    if start_step >= args.steps:
        print(f"[RESUME] Training already complete ({start_step}/{args.steps}).")
        # Final root-level adapter save may have been skipped on a prior crash or
        # an older trainer version — ensure merge can find adapter_config.json.
        if not args.no_lora and not os.path.isfile(
                os.path.join(args.output, "adapter_config.json")):
            print(f"[RESUME] Finalizing adapter at {args.output}")
            draft_model.save_pretrained(args.output)
        return

    # ── W&B (optional) ────────────────────────────────────────────────────────
    # Label used for results.db train_curves rows — basename of the output dir,
    # e.g. "kl_tree-gsm8k".  Matches the checkpoint directory name the dashboard
    # uses when linking curves to pipeline steps.
    _db_label = os.path.basename(os.path.normpath(args.output))

    _wandb = None
    if not args.no_wandb:
        try:
            import wandb as _w

            def _teacher_short(target: str) -> str:
                """Extract a short version tag from the teacher model path or HF ID.
                'Qwen/Qwen3-8B' → 'qwen3-8b', '/kaggle/input/.../qwen3-8b/1' → 'qwen3-8b'
                Strips numeric-only last segments (e.g. /1, /v2) from paths.
                """
                import re
                parts = re.split(r"[/\\]", target.strip())
                # Drop empty parts and numeric-only segments at the end
                parts = [p for p in parts if p and not re.fullmatch(r"\d+", p)]
                tag = parts[-1] if parts else "unknown"
                # Strip org prefix from HF IDs like "Qwen/Qwen3-8B"
                tag = tag.split("/")[-1]
                return tag.lower().replace("_", "-")

            # Run name encodes: teacher-size + config-tier + loss + family + teacher + steps
            # e.g. "8B-kaggle-kl_qwen_qwen3-8b_1000steps"
            # run_label comes from logging.run_label in the YAML (forwarded by
            # experiment.py); if absent falls back to plain loss+family+steps.
            _label_prefix = f"{args.run_label}-" if args.run_label else ""
            # Format: {run_label}-{loss}_{family}_{teacher}_{steps}steps
            # e.g. "0.6B-laptop-kl_qwen_qwen3-0.6b_200steps"
            _run_name = f"{_label_prefix}{args.loss}_{family.name}_{_teacher_short(args.target)}_{args.steps}steps"
            _is_smoke = (args.steps <= 10)
            _run_mode = "smoke" if _is_smoke else "full"
            _wandb = _w.init(
                project=args.wandb_project, entity=args.wandb_entity,
                group=args.wandb_group or None,
                job_type="train",
                name=_run_name,
                tags=[args.loss, "train", "phase1",
                      getattr(args, "hw_tier", "unknown"),
                      f"seed{args.seed}",
                      _run_mode],   # "smoke" or "full" for easy filtering
                config={
                    "loss":              args.loss,
                    "hw_tier":           getattr(args, "hw_tier", "unknown"),
                    "run_mode":          _run_mode,
                    "steps":             args.steps,
                    "lr":                args.lr,
                    "grad_accum":        args.grad_accum,
                    "batch_size":        getattr(args, "batch_size", 1),
                    "lora_r":            args.lora_r,
                    "teacher_temp":      args.teacher_temp,
                    "seed":              args.seed,
                    "family":            family.name,
                    "draft":             args.draft,
                    "target":            args.target,
                    "attn_impl":         _ATTN_IMPL,
                    "max_train_prompts": getattr(args, "max_train_prompts", 0),
                },
                resume="allow",
            )
            # Use training step as x-axis for all train metrics.
            # This ensures charts show step 0-1000, not W&B's internal 0-N counter.
            _w.define_metric("train_step")
            _w.define_metric("train/*",   step_metric="train_step")
            _w.define_metric("val/*",     step_metric="train_step")
            _w.define_metric("gradients/*", step_metric="train_step")
            # Do NOT watch gradients (log="gradients" uploads 392 per-layer histograms
            # that are unreadable). We log aggregated LoRA grad norms manually instead.
            print(f"  [wandb] {_wandb.url}")
        except Exception as e:
            print(f"  [wandb] Skipped ({type(e).__name__}: {e})")

    # ── Dataset split ─────────────────────────────────────────────────────────
    all_prompts = load_prompts(args.dataset)
    # Smoke mode: cap training dataset so pre-tokenization is fast.
    # --max_train_prompts 50 → tokenize 50 prompts instead of 6726 (saves ~2 min on laptop).
    if getattr(args, "max_train_prompts", 0) > 0:
        all_prompts = all_prompts[:args.max_train_prompts]
    if args.val_dataset:
        val_prompts   = load_prompts(args.val_dataset)
        train_prompts = all_prompts
    elif args.val_every > 0 and args.val_split > 0:
        n_val         = max(1, int(len(all_prompts) * args.val_split))
        train_prompts = all_prompts[:-n_val]
        val_prompts   = all_prompts[-n_val:]
    else:
        train_prompts = all_prompts
        val_prompts   = []

    prompts = train_prompts
    n = len(prompts)
    if n == 0:
        raise ValueError("No training prompts after split.")

    # ── Pre-tokenise ─────────────────────────────────────────────────────────
    print("Pre-tokenising prompts...")
    _tok_cache: dict = {}
    for _p in prompts:
        _tok_cache[_p] = tokenizer(
            family.format_prompt(_p, tokenizer),
            return_tensors="pt"
        ).input_ids
    print(f"  {len(_tok_cache)} prompts cached.")

    def _epoch_prompts(epoch: int) -> list:
        if args.no_shuffle:
            return list(prompts)
        rng = random.Random(args.seed + epoch)
        order = list(range(n))
        rng.shuffle(order)
        return [prompts[i] for i in order]

    shuffled = _epoch_prompts(start_step // n)

    # ── Health state ──────────────────────────────────────────────────────────
    _nan_count            = 0
    _val_loss_history     = []
    _best_val_loss        = (_resumed_best_val if _resuming and _resumed_best_val is not None
                             else float("inf"))
    _val_no_improve_count = (_resumed_no_improve if _resuming else 0)
    _current_ppl          = None
    _baseline_ppl         = None
    _stop_training        = False
    # EMA state dict: stash loss_ema here instead of on the wandb Run object.
    # Newer W&B versions raise AttributeError when setting arbitrary attributes on Run.
    _ema_state: dict = {}

    if args.ppl_check_every > 0:
        # Use 1 prompt on laptop (small teacher, value is meaningless — just exercises the code path).
        # Use 5 prompts on T4/A100 where the PPL signal is actionable.
        _ppl_n       = 1 if getattr(args, "hw_tier", "laptop") == "laptop" else 5
        _ppl_sample  = (val_prompts[:_ppl_n] if val_prompts else prompts[:_ppl_n])
        print("Computing pre-training baseline PPL...")
        _baseline_ppl = _compute_ppl(
            draft_model, _ppl_sample, _tok_cache, tokenizer, device, family)
        _current_ppl  = _baseline_ppl
        print(f"  [health] Baseline PPL = {_baseline_ppl:.2f}  "
              f"(warn if > {_baseline_ppl * args.ppl_threshold:.2f})\n")

    # ── Tree-loss smoke test (grad check) ────────────────────────────────────
    # Runs once before the main loop to confirm that grad flows through the
    # tree forward pass before we spend any steps on it.
    if args.loss in TREE_LOSS_NAMES:
        from algorithms.distillspec_gbv.verifiers.draft_generator import iid_draft
        from algorithms.distillspec_gbv.verifiers.draft_generator import target_tree_pass
        print("[tree] Running draft_tree_forward_with_grad smoke test (grad check)…")
        _smoke_prompt = prompts[0]
        _smoke_ids    = _tok_cache[_smoke_prompt].to(device)
        with torch.no_grad():
            _smoke_out   = draft_model(_smoke_ids, use_cache=True, return_dict=True)
            _smoke_qcache = _smoke_out.past_key_values
            _smoke_pending = torch.multinomial(
                F.softmax(_smoke_out.logits[0, -1] / args.teacher_temp, dim=-1), 1
            ).unsqueeze(0)
            _smoke_paths, _, _ = iid_draft(
                draft_model, _smoke_qcache, _smoke_pending,
                K=args.tree_K, L=args.tree_L, q_temp=args.teacher_temp,
            )
        # verify_tree_forward_grad runs the forward WITH grad and asserts requires_grad=True
        verify_tree_forward_grad(
            draft_model, _smoke_ids, _smoke_paths,
            L=args.tree_L, K=args.tree_K, q_temp=args.teacher_temp,
        )
        del _smoke_out, _smoke_qcache, _smoke_pending, _smoke_paths
        torch.cuda.empty_cache()
        print()

    # ── Training loop ─────────────────────────────────────────────────────────
    optimizer.zero_grad()   # start clean; re-zeroed inside loop after each accum window
    _grad_norm: float | None = None   # updated on each optimizer step; logged to W&B
    for step in range(start_step, args.steps):
        if _stop_training:
            break

        idx = step % n
        if idx == 0 and step > 0:
            shuffled = _epoch_prompts(step // n)
        prompt = shuffled[idx]

        prompt_ids = _tok_cache[prompt].to(device)

        # ── Flat-sequence teacher rollout (skipped for tree losses) ─────────────
        # Tree losses build their own on-policy rollout inside _tree_training_step.
        s_log = t_log = gen_ids = None
        if args.loss not in TREE_LOSS_NAMES:
            with torch.no_grad():
                gen_out = target_model.generate(
                    prompt_ids,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.teacher_temp,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    # output_scores=True is intentionally omitted: storing 80
                    # per-token score tensors in a Python list and then
                    # torch.stack()-ing them on every step is the primary CPU
                    # bottleneck (80 small sequential GPU dispatches + Python
                    # list append × 80 + stack).  Instead we do one parallel
                    # teacher forward pass on the full sequence below, which is
                    # a single large GPU matmul — same FLOPs, far less Python
                    # dispatch overhead → significantly better GPU utilisation.
                )

            full_ids = gen_out.sequences
            plen     = prompt_ids.shape[1]
            gen_len  = full_ids.shape[1] - plen
            if gen_len == 0:
                losses.append(float("nan"))
                continue

            # Single parallel teacher forward pass for all generated positions.
            # Mathematically identical to output_scores: causal masking ensures
            # logits[0, plen-1+i] == the distribution the teacher used when
            # sampling token i during generation.  Raw logits (no temperature
            # applied), so recover_raw_logits() is not needed.
            with torch.no_grad():
                t_log = target_model(full_ids).logits[0, plen - 1:-1, :].float()

            # Draft: forward pass with gradient
            draft_model.train()
            student_logits = draft_model(full_ids).logits[:, :-1, :].float()
            s_log   = student_logits[:, plen - 1:, :].squeeze(0)  # [gen_len, V]
            gen_ids = full_ids[0, plen:]                            # [gen_len]

        try:
            if args.loss in TREE_LOSS_NAMES:
                out  = None
                aw   = None
                loss = _tree_training_step(
                    draft_model, target_model, prompt_ids,
                    args.loss, args.tree_K, args.tree_L, args.teacher_temp,
                )
            else:
                out = get_loss(args.loss, s_log, t_log, token_ids=gen_ids,
                              kl_weight=args.ebe_kl_weight, alpha=args.jsd_alpha)
                loss = out.loss
                aw   = out.accept_weight
        except Exception as exc:
            _nan_count += 1
            print(f"\n  [health] ⚠ Loss exception at step {step+1}: {exc}")
            import traceback; traceback.print_exc()
            if args.nan_action == "stop":
                _save_checkpoint(draft_model, optimizer, step + 1, args.output,
                                 losses, accept_weights, no_lora=args.no_lora)
                _stop_training = True
                break
            losses.append(float("nan"))
            continue

        # NaN / Inf guard
        loss_val = loss.item()
        if not math.isfinite(loss_val):
            _nan_count += 1
            if args.nan_action == "stop":
                _save_checkpoint(draft_model, optimizer, step + 1, args.output,
                                 losses, accept_weights, no_lora=args.no_lora)
                _stop_training = True
                break
            elif args.nan_action == "skip":
                losses.append(float("nan"))
                continue
            # warn: fall through

        # Gradient accumulation: accumulate for grad_accum steps, then update.
        # Dividing by grad_accum keeps loss magnitude consistent regardless of accum size.
        (loss / args.grad_accum).backward()
        if (step + 1) % args.grad_accum == 0 or (step + 1) == args.steps:
            _trainable = [p for p in draft_model.parameters() if p.requires_grad]
            # Always compute the total grad norm (pre-clip); clip_grad_norm_ returns it.
            # When grad_clip==0 (disabled) use inf so clipping is a no-op but norm is still
            # computed — spikes > 5.0 signal instability, persistent high = clip too loose.
            _grad_norm = torch.nn.utils.clip_grad_norm_(
                _trainable,
                args.grad_clip if args.grad_clip > 0 else float("inf"),
            ).item()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        losses.append(loss_val)
        if aw is not None:
            accept_weights.append(aw)

        # ── Progress logging ─────────────────────────────────────────────────
        if (step + 1) % args.log_every == 0:
            peak_vram = torch.cuda.max_memory_allocated() / 1024**2
            print(f"Step {step+1:4d}/{args.steps} | "
                  f"train_loss: {loss_val:.4f} | peak VRAM: {peak_vram:.0f} MB")
            if _wandb:
                # EMA-smoothed loss (α=0.1): cleaner trend line than raw per-step loss.
                # Use a local dict instead of setting attributes on the wandb Run object
                # (newer wandb versions raise AttributeError on arbitrary attribute sets).
                _loss_ema = _ema_state.get("loss_ema", loss_val)
                _loss_ema = 0.9 * _loss_ema + 0.1 * loss_val
                _ema_state["loss_ema"] = _loss_ema

                _wlog = {
                    "train_step":          step + 1,   # x-axis for all train/* metrics
                    "train/loss":          loss_val,
                    "train/loss_ema":      _loss_ema,  # smooth trend line
                    "train/lr":            scheduler.get_last_lr()[0],
                    "train/peak_vram_mb":  peak_vram,
                }
                if _grad_norm is not None:
                    _gn = float(_grad_norm)
                    _wlog["train/grad_norm"] = _gn
                    # Per-component LoRA gradient health (replaces 392 histogram panels).
                    # Mean + std + max across all LoRA-A and LoRA-B matrices.
                    # Near-zero mean = dead adapter.  High std = uneven layer updates.
                    try:
                        import statistics as _stats
                        _a_norms, _b_norms = [], []
                        for _pname, _pp in draft_model.named_parameters():
                            if _pp.grad is not None:
                                _pn = _pp.grad.norm().item()
                                if "lora_A" in _pname:
                                    _a_norms.append(_pn)
                                elif "lora_B" in _pname:
                                    _b_norms.append(_pn)
                        if _a_norms:
                            _wlog["gradients/lora_A_norm_mean"] = sum(_a_norms) / len(_a_norms)
                            _wlog["gradients/lora_A_norm_max"]  = max(_a_norms)
                            if len(_a_norms) > 1:
                                _wlog["gradients/lora_A_norm_std"] = _stats.stdev(_a_norms)
                        if _b_norms:
                            _wlog["gradients/lora_B_norm_mean"] = sum(_b_norms) / len(_b_norms)
                            _wlog["gradients/lora_B_norm_max"]  = max(_b_norms)
                            if len(_b_norms) > 1:
                                _wlog["gradients/lora_B_norm_std"] = _stats.stdev(_b_norms)
                    except Exception:
                        pass  # gradient logging is best-effort; never crash training

                    # ── Automated alerts (appear in W&B notifications + run log) ──
                    try:
                        _cur_step = step + 1
                        if _gn > 10.0:
                            # _gn is the PRE-clip norm.  When --grad_clip > 0 the actual
                            # step is already clipped to that value, so a high pre-clip
                            # norm is informational (the optimizer step is bounded), not
                            # an explosion.  Make the message reflect whether clipping is
                            # active so "lower --grad_clip" isn't suggested when it's
                            # already doing its job.
                            _clip_active = args.grad_clip > 0
                            _clip_note = (f"(pre-clip; step clipped to {args.grad_clip})"
                                          if _clip_active else "(no clipping active)")
                            _action = ("reduce --lr — pre-clip grads are persistently large"
                                       if _clip_active
                                       else "reduce --lr or set --grad_clip")
                            _wandb.alert(
                                title="Large pre-clip gradient norm",
                                text=f"grad_norm={_gn:.2f} {_clip_note} at step {_cur_step}. "
                                     f"ACTION: {_action}.",
                                level="WARN",
                            )
                            print(f"  [ALERT] grad_norm={_gn:.2f} > 10 {_clip_note} — {_action}")
                    except Exception:
                        pass
                _wandb.log(_wlog)
            if _results_db is not None:
                try:
                    _results_db.insert_train_step(
                        label=_db_label,
                        loss_name=args.loss,
                        step=step + 1,
                        loss=loss_val,
                        learning_rate=args.lr,
                        lora_rank=args.lora_r,
                        split="train",
                    )
                except Exception:
                    pass  # never crash training due to DB write failure

        # ── Crash-safe checkpoint ─────────────────────────────────────────────
        if args.save_every > 0 and (step + 1) % args.save_every == 0:
            is_milestone = (args.milestone_every > 0
                            and (step + 1) % args.milestone_every == 0)
            _save_checkpoint(
                draft_model, optimizer, step + 1, args.output,
                losses, accept_weights,
                milestone=is_milestone,
                max_checkpoints=args.max_checkpoints,
                best_val_loss=_best_val_loss,
                val_no_improve_count=_val_no_improve_count,
                no_lora=args.no_lora,
            )
            print(f"  [ckpt] step {step+1}/{args.steps} → {args.output}/ckpt_latest")

        # ── Validation ───────────────────────────────────────────────────────
        if val_prompts and args.val_every > 0 and (step + 1) % args.val_every == 0:
            v_loss, v_aw = _compute_val_loss(
                draft_model, target_model, val_prompts, _tok_cache,
                tokenizer, args, device, family, max_prompts=100,
            )
            _val_loss_history.append(v_loss)
            # Tree losses validate with a forward_kl PROXY (see _compute_val_loss):
            # the tree objective needs an expensive K-path tree per prompt, so val
            # uses forward_kl as a language-quality guard instead.  Label it clearly
            # so the printed value is never mistaken for the tree objective itself.
            _val_is_proxy = args.loss in TREE_LOSS_NAMES
            _val_tag = "val_loss(fkl-proxy)" if _val_is_proxy else "val_loss"
            print(f"Step {step+1:4d}/{args.steps} | {_val_tag}: {v_loss:.4f}")

            if v_loss < _best_val_loss:
                _best_val_loss = v_loss
                _val_no_improve_count = 0
                best_dir = os.path.join(args.output, "ckpt_best")
                _save_checkpoint(
                    draft_model, optimizer, step + 1, args.output,
                    losses, accept_weights, no_lora=args.no_lora)
                shutil.copytree(
                    os.path.join(args.output, "ckpt_latest"),
                    best_dir, dirs_exist_ok=True)
                print(f"  [best] New best val {v_loss:.4f} at step {step+1} "
                      f"→ {best_dir}")
            else:
                _val_no_improve_count += 1
                print(f"  [val]  No improvement "
                      f"(best={_best_val_loss:.4f}, streak={_val_no_improve_count})")

            if _wandb:
                _cur_ema = _ema_state.get("loss_ema", v_loss)
                # val_train_gap = val - train_ema: positive means val is worse than train
                # (overfitting). Works correctly for both positive-range losses (KL, L1: 0→∞)
                # and negative-range losses (EBE, tree: -∞→0). The old overfit_ratio
                # (v_loss / train_ema) blew up to ±1e8 in early warmup when train_ema≈1e-9,
                # and had inverted sign semantics for negative losses.
                # Renamed from train/overfit_ratio as of this commit.
                _val_gap = v_loss - _cur_ema
                log = {
                    "train_step":           step + 1,
                    "train/val_loss":       v_loss,
                    "train/loss_ema":       _cur_ema,
                    "train/val_train_gap":  _val_gap,
                }
                if v_aw is not None:
                    log["val/accept_weight"] = v_aw
                _wandb.log(log)

                # ── Automated alerts for convergence decisions ─────────────────
                try:
                    _cur_step = step + 1
                    # val_train_gap is only meaningful when train and val measure the
                    # SAME quantity.  For tree losses, val_loss is a forward_kl proxy
                    # (range ~2) while train_ema is the tree objective (range ~0 or
                    # negative) — their difference is structural, not overfitting, and
                    # would fire a guaranteed false "overfitting" alert every check.
                    # Skip the gap alert for tree losses (metrics are incommensurable).
                    if _val_gap > 0.5 and not _val_is_proxy:
                        _wandb.alert(
                            title="Overfitting detected",
                            text=f"val_train_gap={_val_gap:.4f} at step {_cur_step} "
                                 f"(val={v_loss:.4f}, train_ema={_cur_ema:.4f}). "
                                 f"ACTION: lower --lr, reduce steps, or add more data.",
                            level="WARN",
                        )
                        print(f"  [ALERT] val_train_gap={_val_gap:.4f} > 0.5 — check LR/data")
                    if _val_no_improve_count >= args.early_stop_patience > 0:
                        _wandb.alert(
                            title="Val loss plateau",
                            text=f"No val improvement for {_val_no_improve_count} checks "
                                 f"(best={_best_val_loss:.4f}). "
                                 f"ACTION: check train/lr curve — warmup may need tuning.",
                            level="WARN",
                        )
                    # Early convergence check: after 10% of steps, loss should have dropped
                    if _cur_step == max(10, args.steps // 10):
                        _initial_loss = losses[0] if losses else v_loss
                        _drop_pct = (_initial_loss - _cur_ema) / max(_initial_loss, 1e-8) * 100
                        if _drop_pct < 10:
                            _wandb.alert(
                                title="Slow convergence",
                                text=f"Loss dropped only {_drop_pct:.1f}% after "
                                     f"{_cur_step} steps (from {_initial_loss:.4f} to "
                                     f"{_cur_ema:.4f}). "
                                     f"ACTION: increase --lr or check --warmup_steps.",
                                level="WARN",
                            )
                            print(f"  [ALERT] slow convergence: {_drop_pct:.1f}% drop after "
                                  f"{_cur_step} steps — consider increasing LR")
                except Exception:
                    pass  # alerts are best-effort
            if _results_db is not None:
                try:
                    _results_db.insert_train_step(
                        label=_db_label,
                        loss_name=args.loss,
                        step=step + 1,
                        loss=v_loss,
                        learning_rate=args.lr,
                        lora_rank=args.lora_r,
                        split="val",
                    )
                except Exception:
                    pass

            if (args.early_stop_patience > 0
                    and _val_no_improve_count >= args.early_stop_patience):
                print(f"\n[EARLY STOP] val loss has not improved for "
                      f"{_val_no_improve_count} consecutive checks.  Stopping.")
                _stop_training = True
                break

    # ── Final checkpoint ──────────────────────────────────────────────────────
    final_step = min(start_step + len(losses), args.steps)
    _save_checkpoint(
        draft_model, optimizer, final_step, args.output,
        losses, accept_weights,
        best_val_loss=_best_val_loss,
        val_no_improve_count=_val_no_improve_count,
        no_lora=args.no_lora,
    )
    # Save final adapter to the output root so that:
    #   1. experiment.py done_check (adapter_config.json at root) passes
    #   2. --merge_only --adapter <output_dir> finds adapter_config.json at root
    # This mirrors the behaviour of earlier trainer versions that flat-loss
    # checkpoints were originally created with.
    if not args.no_lora:
        draft_model.save_pretrained(args.output)

    print(f"\n[done] Training complete.  Checkpoint: {args.output}/ckpt_latest")
    print(f"       Best val loss: {_best_val_loss:.4f}  "
          f"→  {args.output}/ckpt_best")
    print(f"\nTo merge LoRA for inference:")
    print(f"  python -m algorithms.distillspec_gbv.trainer "
          f"--merge_only --adapter {args.output}/ckpt_best "
          f"--draft {args.draft}")

    if _wandb:
        _wandb.finish()


if __name__ == "__main__":
    main()
