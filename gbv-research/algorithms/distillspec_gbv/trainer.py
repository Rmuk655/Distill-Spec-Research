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

import torch
import torch.nn.functional as F
import transformers
from torch.optim import AdamW
from peft import get_peft_model, LoraConfig, TaskType, PeftModel

from core.model_families import get_family, ModelFamily
from algorithms.distillspec_gbv.losses import get_loss, LossOutput


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
    p.add_argument("--teacher_temperature", type=float, default=0.8,
                   help="Sampling temperature for the teacher's generate() call. "
                        "The trainer undoes any family-specific temperature pre-scaling "
                        "via ModelFamily.recover_raw_logits().")

    # ── Training ─────────────────────────────────────────────────────────────
    p.add_argument("--loss", default="forward_kl",
                   choices=["forward_kl", "reverse_kl", "jsd", "l1", "ebe", "ebe_single"],
                   help="Distillation loss objective.")
    p.add_argument("--steps",  type=int, default=1000,
                   help="Total training steps.")
    p.add_argument("--lr",     type=float, default=3e-5)
    p.add_argument("--max_new_tokens", type=int, default=80,
                   help="Max tokens the teacher generates per step.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_shuffle", action="store_true",
                   help="Disable dataset shuffling.")

    # ── LoRA ─────────────────────────────────────────────────────────────────
    p.add_argument("--lora_r",     type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--no_lora",    action="store_true",
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

def merge_lora_and_save(draft_model_id: str, adapter_path: str) -> None:
    """Merge a LoRA adapter into the base model and save to <adapter_path>_merged/."""
    print(f"Loading base: {draft_model_id}")
    base   = transformers.AutoModelForCausalLM.from_pretrained(
        draft_model_id, torch_dtype=torch.bfloat16)
    print(f"Loading LoRA: {adapter_path}")
    model  = PeftModel.from_pretrained(base, adapter_path)
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
                temperature=args.teacher_temperature,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )
            if not gen_out.scores:
                continue

            full_ids = gen_out.sequences.clone()
            # Undo family-specific temperature pre-scaling
            t_log = (torch.stack(gen_out.scores, dim=0)
                     .squeeze(1).float())
            t_log = family.recover_raw_logits(t_log, args.teacher_temperature)

            student_logits = draft_model(full_ids).logits[:, :-1, :].float()
            s_log   = student_logits[:, plen - 1:, :].squeeze(0)
            gen_ids = full_ids[0, plen:]

            out = get_loss(args.loss, s_log, t_log, token_ids=gen_ids)
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device        : {device}")
    print(f"Model family  : {family.name}")
    print(f"Draft         : {args.draft}  ({'full SFT' if args.no_lora else f'LoRA r={args.lora_r}'})")
    print(f"Target        : {args.target}  (frozen bfloat16)")
    print(f"Loss          : {args.loss}")
    print(f"Attn backend  : {_ATTN_IMPL}")
    print(f"Steps         : {args.steps}")
    print(f"Teacher temp  : {args.teacher_temperature}")
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
        target_model = transformers.AutoModelForCausalLM.from_pretrained(
            args.target, quantization_config=bnb_cfg, device_map="auto",
            attn_implementation=_ATTN_IMPL)
    else:
        target_model = transformers.AutoModelForCausalLM.from_pretrained(
            args.target, torch_dtype=torch.bfloat16,
            attn_implementation=_ATTN_IMPL).to(device)
    target_model.eval()
    for p in target_model.parameters():
        p.requires_grad_(False)
    print(f"  VRAM: {torch.cuda.memory_allocated() / 1024**2:.0f} MB")

    # ── Load draft ────────────────────────────────────────────────────────────
    print("Loading draft...")
    draft_base = transformers.AutoModelForCausalLM.from_pretrained(
        args.draft, torch_dtype=torch.bfloat16,
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
            lora_dropout=0.05,
            target_modules=family.lora_target_modules(),  # ← family-specific
            bias="none",
        )
        draft_model = get_peft_model(draft_base, lora_cfg)
        draft_model.print_trainable_parameters()
    print(f"  VRAM (both): {torch.cuda.memory_allocated() / 1024**2:.0f} MB\n")

    optimizer = AdamW(
        [p for p in draft_model.parameters() if p.requires_grad], lr=args.lr)

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
            draft_model.load_state_dict(_tmp.state_dict())
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
        return

    # ── W&B (optional) ────────────────────────────────────────────────────────
    _wandb = None
    if not args.no_wandb:
        try:
            import wandb as _w
            _wandb = _w.init(
                project=args.wandb_project, entity=args.wandb_entity,
                group=args.wandb_group,
                name=f"{args.loss}_{family.name}_{args.steps}steps",
                config={**vars(args), "family": family.name,
                        "attn_impl": _ATTN_IMPL},
                resume="allow",
            )
            _w.watch(draft_model, log="gradients", log_freq=100)
            print(f"  [wandb] {_wandb.url}")
        except Exception as e:
            print(f"  [wandb] Skipped ({type(e).__name__}: {e})")

    # ── Dataset split ─────────────────────────────────────────────────────────
    all_prompts = load_prompts(args.dataset)
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

    if args.ppl_check_every > 0:
        _ppl_sample  = val_prompts[:5] if val_prompts else prompts[:5]
        print("Computing pre-training baseline PPL...")
        _baseline_ppl = _compute_ppl(
            draft_model, _ppl_sample, _tok_cache, tokenizer, device, family)
        _current_ppl  = _baseline_ppl
        print(f"  [health] Baseline PPL = {_baseline_ppl:.2f}  "
              f"(warn if > {_baseline_ppl * args.ppl_threshold:.2f})\n")

    # ── Training loop ─────────────────────────────────────────────────────────
    for step in range(start_step, args.steps):
        if _stop_training:
            break

        idx = step % n
        if idx == 0 and step > 0:
            shuffled = _epoch_prompts(step // n)
        prompt = shuffled[idx]

        prompt_ids = _tok_cache[prompt].to(device)

        # Teacher: generate + capture logits
        with torch.no_grad():
            gen_out = target_model.generate(
                prompt_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.teacher_temperature,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )
        if not gen_out.scores:
            losses.append(float("nan"))
            continue

        full_ids = gen_out.sequences.clone()
        plen     = prompt_ids.shape[1]

        # Recover raw logits (undo family-specific temp pre-scaling)
        raw_scores = torch.stack(gen_out.scores, dim=0).squeeze(1).float()
        t_log = family.recover_raw_logits(raw_scores, args.teacher_temperature)

        # Draft: forward pass with gradient
        draft_model.train()
        student_logits = draft_model(full_ids).logits[:, :-1, :].float()
        s_log   = student_logits[:, plen - 1:, :].squeeze(0)  # [gen_len, V]
        gen_ids = full_ids[0, plen:]                            # [gen_len]

        try:
            out = get_loss(args.loss, s_log, t_log, token_ids=gen_ids)
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

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in draft_model.parameters() if p.requires_grad], 1.0)
        optimizer.step()

        losses.append(loss_val)
        if aw is not None:
            accept_weights.append(aw)

        # ── Progress logging ─────────────────────────────────────────────────
        if (step + 1) % args.log_every == 0:
            peak_vram = torch.cuda.max_memory_allocated() / 1024**2
            print(f"Step {step+1:4d}/{args.steps} | "
                  f"train_loss: {loss_val:.4f} | peak VRAM: {peak_vram:.0f} MB")
            if _wandb:
                _wandb.log({
                    "train/loss": loss_val,
                    "train/lr": args.lr,
                    "train/peak_vram_mb": peak_vram,
                    "step": step + 1,
                })

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
                tokenizer, args, device, family, max_prompts=30,
            )
            _val_loss_history.append(v_loss)
            print(f"Step {step+1:4d}/{args.steps} | val_loss: {v_loss:.4f}")

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
                log = {"val/loss": v_loss, "step": step + 1}
                if v_aw is not None:
                    log["val/accept_weight"] = v_aw
                _wandb.log(log)

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
