"""
checkpointing.py — save / resume / model-loading for the training pipeline.

Public API:
  load_models(draft_id, teacher_id, device, load_in_4bit, lora_config)
      -> (tokenizer, draft, teacher)
  save_checkpoint(model, optimizer, scheduler, output_dir, name, state, use_lora)
  try_resume(model, optimizer, scheduler, output_dir)
      -> (start_step, state_dict)
"""
from __future__ import annotations

import json
import os

import torch
from transformers import AutoModelForCausalLM

import verifiers  # noqa: F401
from util import load_models as _sot_load


def save_checkpoint(model, optimizer, scheduler, output_dir, name, state, use_lora: bool = False,
                     hidden_proj=None):
    """Write model + optimizer + scheduler + training state to <output>/<name>/.

    hidden_proj (--hidden_weight > 0, see notes/hidden_state_overlap_objective.md)
    is a separate nn.Module, not part of `model`, so model.save_pretrained()
    never sees it — its weights need their own file or a fresh --resume would
    silently reinitialize it (only its optimizer momentum would carry over,
    since it shares the optimizer instance with the draft's own params).
    """
    target = os.path.join(output_dir, name)
    os.makedirs(target, exist_ok=True)
    if use_lora:
        model.save_pretrained(target)                       # PEFT-aware save
    else:
        model.save_pretrained(target, safe_serialization=True)
    torch.save({
        "optimizer": optimizer.state_dict(),
        "optimizer_class": type(optimizer).__name__,
        "scheduler": scheduler.state_dict() if scheduler else None,
    }, os.path.join(target, "optim.pt"))
    if hidden_proj is not None:
        torch.save(hidden_proj.state_dict(), os.path.join(target, "hidden_proj.pt"))
    json.dump(state, open(os.path.join(target, "state.json"), "w"))

def try_resume(model, optimizer, scheduler, output_dir, hidden_proj=None):
    """Load ckpt_latest if it exists.  Returns (start_step, training_state)."""
    latest = os.path.join(output_dir, "ckpt_latest")
    state_path = os.path.join(latest, "state.json")
    print(f"[resume] looking for checkpoint at {latest}")
    if not os.path.isfile(state_path):
        if os.path.isdir(latest):
            print(f"[resume] WARNING: {latest} exists but state.json is missing — starting from scratch")
        else:
            print(f"[resume] no checkpoint found — starting from scratch")
        return 0, {}
    state = json.load(open(state_path, encoding="utf-8"))
    step = state.get("step", 0)
    best_be = state.get("best_val_block_eff", 0.0)
    print(f"[resume] found state: step={step}  best_be={best_be:.3f} — loading weights...")
    if "cmd" in state:
        resume_cmd = " ".join(state["cmd"]) + " --resume"
        print(f"[resume] to resume again after next kill:\n  {resume_cmd}")
    # Load model weights
    model_state = AutoModelForCausalLM.from_pretrained(
        latest, torch_dtype=torch.bfloat16,
    ).state_dict()
    model.load_state_dict(model_state, strict=False)
    # Load optimizer + scheduler. Optimizer state format is NOT compatible across
    # optimizer classes (e.g. torch.optim.AdamW's fp32 exp_avg/exp_avg_sq vs
    # bitsandbytes AdamW8bit's int8-quantized state keyed as state["state1"]).
    # IMPORTANT: torch's generic Optimizer.load_state_dict() does NOT validate
    # that the incoming state's internal keys match what this optimizer class
    # expects — it silently copies whatever keys were saved into self.state[p],
    # so loading an AdamW checkpoint into AdamW8bit raises NO error here. The
    # crash instead happens later, deep inside the first optimizer.step() call
    # (KeyError: 'state1'), by which point resume already looks like it
    # succeeded. So we check the saved optimizer class explicitly BEFORE
    # attempting the load, rather than relying on an exception that doesn't
    # reliably occur at this point.
    optim_blob = torch.load(os.path.join(latest, "optim.pt"), map_location="cpu")
    # Checkpoints saved before this field existed used plain torch.optim.AdamW
    # (the only optimizer this codebase had before --optim_8bit) — assume that,
    # not "unknown/compatible", so old checkpoints get the same safety check.
    saved_optim_class = optim_blob.get("optimizer_class", "AdamW")
    current_optim_class = type(optimizer).__name__
    if saved_optim_class != current_optim_class:
        print(f"[resume] WARNING: checkpoint optimizer ({saved_optim_class}) does not match "
              f"the current optimizer ({current_optim_class}) — skipping optimizer/scheduler "
              f"state, continuing with fresh optimizer state. Model weights (the actual "
              f"training progress) resumed successfully.")
    else:
        try:
            optimizer.load_state_dict(optim_blob["optimizer"])
            if scheduler and optim_blob.get("scheduler"):
                scheduler.load_state_dict(optim_blob["scheduler"])
        except Exception as e:
            print(f"[resume] WARNING: optimizer/scheduler state failed to load ({e}) — "
                  f"continuing with fresh optimizer state. Model weights (the actual "
                  f"training progress) resumed successfully.")
    if hidden_proj is not None:
        proj_path = os.path.join(latest, "hidden_proj.pt")
        if os.path.isfile(proj_path):
            hidden_proj.load_state_dict(torch.load(proj_path, map_location="cpu"))
            print(f"[resume] restored hidden_proj weights from {proj_path}")
        else:
            print(f"[resume] WARNING: --hidden_weight > 0 but no hidden_proj.pt found at "
                  f"{latest} (checkpoint predates this flag, or was saved without it) — "
                  f"hidden_proj starts from a fresh random init while the draft's own "
                  f"weights resume normally.")
    print(f"[resume] restored step={step} from {latest}")
    return step, state



def load_models(draft_id: str, teacher_id: str, device: str = "cuda",
                load_in_4bit: bool = False, lora_config: dict | None = None):
    """Load draft + teacher, configured for training.
    
    lora_config: if None -> full fine-tuning.  If dict with keys r, alpha, dropout
                 -> wrap draft in LoRA with those settings.
    """
    tok, teacher, draft = _sot_load(teacher_id, draft_id, device=device, load_in_4bit=load_in_4bit)
    # util.load_models disables grad globally (inference default); restore for training.
    torch.set_grad_enabled(True)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    if lora_config is not None:
        from peft import LoraConfig, get_peft_model
        print(f"[load] wrapping draft in LoRA r={lora_config['r']} alpha={lora_config['alpha']}")
        lora_cfg = LoraConfig(
            r=lora_config['r'], lora_alpha=lora_config['alpha'],
            lora_dropout=lora_config['dropout'],
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        )
        draft = get_peft_model(draft, lora_cfg)
        draft.print_trainable_parameters()
    else:
        for p in draft.parameters():
            p.requires_grad_(True)

    draft.train()
    return tok, draft, teacher