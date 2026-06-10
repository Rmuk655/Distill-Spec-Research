"""
train.py — DistillSpec / verifier-aligned distillation trainer.

One file, single GPU (A100), Qwen3 only.  Edit the HARDCODED CONSTANTS block
near the top before running, or pass the most common knobs as CLI flags:

    python train.py --loss kl                    # flat forward-KL baseline
    python train.py --loss kl_tree               # on-policy tree forward-KL
    python train.py --loss gbv_tree --steps 4000 --lr 1e-5
    python train.py --loss kl_tree --resume      # resume from <OUTPUT>/ckpt_latest

Pipeline:
    1. load draft (Qwen3-0.6B) and teacher (Qwen3-8B) — both BF16 on the A100.
    2. for each step:
         flat loss → teacher rollout + student forward → divergence at every token.
         tree loss → sample K student draft paths,
                     run teacher tree forward (no grad)            → p_probs_dict,
                     run student tree forward (WITH grad)          → q_probs_dict,
                     loss = -E[tau_V](q_probs_dict, p_probs_dict, K, L).
    3. every VAL_EVERY steps → val_loss on held-out gsm8k_val.jsonl,
       update best_val_loss, save ckpt_best, log to W&B.
    4. every SAVE_EVERY  steps → write ckpt_latest + training_state.json.
    5. resume picks up from ckpt_latest if --resume is passed.

Loss families (see losses/__init__.py for full list):
    Flat   : forward_kl, reverse_kl, jsd, l1
    Tree   : kl_tree, rev_kl_tree, jsd_tree,
             bv_tree, gbv_tree, traversal_tree,
             naive_tree, nss_tree, specinfer_tree, spectr_tree, khisti_tree
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Dict, List

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Local modules
from losses import ALL_LOSSES, FLAT_LOSSES, TREE_LOSSES, get_loss, is_tree_loss
from data_io import get_path as dataset_path

# Import the inference-time draft tree builder from /GBV verifier code.
# verifiers/__init__.py adds the verifiers folder to sys.path so this works.
import verifiers  # noqa: F401  — side effect: sys.path injection
from inference_util import iid_draft, target_tree_pass


# ═══════════════════════════════════════════════════════════════════════════
#  HARDCODED CONSTANTS — edit these before running, or override with CLI flags.
# ═══════════════════════════════════════════════════════════════════════════
DRAFT_MODEL     = "Qwen/Qwen3-0.6B"
TEACHER_MODEL   = "Qwen/Qwen3-8B"

# Training hyper-parameters
STEPS           = 4000                       # number of gradient-accum steps
GRAD_ACCUM      = 8                          # opt-steps = STEPS / GRAD_ACCUM = 500
LR              = 3e-5                       # bv_tree / gbv_tree may need 1e-5
WARMUP_STEPS    = 50                         # 10 % of opt-steps (STEPS/GRAD_ACCUM=500)
GRAD_CLIP       = 1.0                        # DistillSpec Table S1 (arXiv:2310.08461) — 1.0 is the LLM fine-tuning standard (LLaMA, GPT-3, Qwen3)
SEED            = 42

# Speculative-decoding shape (used by all *_tree losses)
K               = 4                          # number of draft paths
L               = 8                          # draft block length
DRAFT_TEMP      = 1.0                        # q_temp for the draft model
TEACHER_TEMP    = 1.0                        # for flat-loss teacher rollout — DistillSpec (arXiv:2310.08461) uses T=1.0
MAX_NEW_TOKENS  = 128                        # generated sequence length for flat losses

# LoRA toggle (full fine-tune is default).  Set USE_LORA = True for adapter
# training — saves disk and lets you keep many checkpoints.  Full FT of
# Qwen3-0.6B in BF16 with AdamW state ≈ 12 GB so the A100 40 GB fits fine.
USE_LORA        = False
LORA_R          = 16
LORA_ALPHA      = 32
LORA_DROPOUT    = 0.05

# Validation & checkpointing cadence
VAL_EVERY       = 100                        # gradient-accum steps between val checks
VAL_PROMPTS     = 25                         # val loss averaged over this many prompts
SAVE_EVERY      = 200                        # ckpt_latest write cadence
LOG_EVERY       = 10                         # console + W&B step-log cadence

# Storage layout — change OUTPUT_ROOT to your preferred checkpoint dir
OUTPUT_ROOT     = os.path.join(os.path.dirname(__file__), "checkpoints")
WANDB_PROJECT   = "distillspec-pipeline"
# ═══════════════════════════════════════════════════════════════════════════


# ---------------------------------------------------------------------------
# Utility: build a tokenised prompt with grad-free KV cache (used by tree losses)
# ---------------------------------------------------------------------------

def _prompt_to_ids(prompt: str, tokenizer, device) -> torch.Tensor:
    """Tokenise a string prompt to [1, plen] LongTensor on the model's device."""
    return torch.tensor(tokenizer.encode(prompt), device=device, dtype=torch.long).unsqueeze(0)


def _prepare_prompt_cache(model, prompt_ids):
    """Prefill the KV cache for the prompt under no_grad — same as iid_draft does."""
    with torch.no_grad():
        out = model(prompt_ids, use_cache=True, return_dict=True)
    return out.past_key_values


# ---------------------------------------------------------------------------
# Student draft-tree forward WITH grad — needed for tree losses.
# (For inference / no-grad draft tree, see verifiers/inference_util.iid_draft.)
# ---------------------------------------------------------------------------

def draft_tree_forward_with_grad(
    draft_model, prompt_ids: torch.Tensor,
    q_paths: List[List[int]], L: int, q_temp: float,
) -> Dict[str, torch.Tensor]:
    """
    Re-run the student over the K sampled paths in ONE tree-attention forward,
    keeping requires_grad=True on the resulting distributions.

    The paths themselves were sampled without grad (iid_draft).  Here we just
    score every node of that fixed tree under the current student weights so
    the loss can back-propagate through q_probs_dict.

    Returns {prefix: q_probs[V]} for every NON-LEAF node (depth < L).
    """
    device = prompt_ids.device
    dtype  = next(draft_model.parameters()).dtype

    # 1. Build unique prefix list (same node ordering as target_tree_pass).
    q_prefixes, q_token_ids = [], []
    for path in q_paths:
        for i, tok in enumerate(path):
            pfx = ",".join(str(x) for x in path[:i + 1])
            if pfx not in q_prefixes:
                q_prefixes.append(pfx)
                q_token_ids.append(tok)
    n_nodes = len(q_prefixes)
    q_tokens = torch.tensor(q_token_ids, device=device, dtype=torch.long).unsqueeze(0)

    # 2. Prompt prefill (no grad — prompt is fixed).
    with torch.no_grad():
        out        = draft_model(prompt_ids, use_cache=True, return_dict=True)
        p_cache    = out.past_key_values
        cached_len = prompt_ids.shape[1]

    # 3. Tree attention mask: node i attends to ancestors j (and itself).
    mask = torch.zeros((n_nodes, cached_len + n_nodes), device=device, dtype=dtype)
    mask[:, cached_len:] = torch.finfo(dtype).min          # default: no cross-node attn
    for i, pi in enumerate(q_prefixes):
        for j, pj in enumerate(q_prefixes):
            if pi.startswith(pj + ",") or pi == pj:
                mask[i, cached_len + j] = 0.0
    mask = mask.unsqueeze(0).unsqueeze(0)                  # [1, 1, n_nodes, total]

    # 4. Tree forward WITH grad.  Qwen3 expects mask in dict form (same as
    #    verifiers/inference_util.py:target_tree_pass).
    draft_model.train()
    out = draft_model(
        q_tokens, past_key_values=p_cache,
        attention_mask={"full_attention": mask},
        use_cache=True, return_dict=True,
    )
    logits = out.logits[0].float()                         # [n_nodes, V]
    probs  = F.softmax(logits / q_temp, dim=-1)            # [n_nodes, V] WITH grad

    # 5. Keep only non-leaf nodes (depth 0 .. L-1) — the verifier losses only
    #    query q_probs_dict at internal nodes.
    return {
        pfx: probs[i]
        for i, pfx in enumerate(q_prefixes)
        if len(pfx.split(",")) - 1 < L
    }


# ---------------------------------------------------------------------------
# Flat-loss path: teacher generates a sequence, student forward, divergence.
# ---------------------------------------------------------------------------

def compute_flat_loss(loss_fn, draft, teacher, prompt_ids,
                       max_new_tokens=128, teacher_temp=0.8):
    """
    Teacher greedily generates max_new_tokens.  Student is then forwarded on
    [prompt + generated_tokens] WITH grad.  Loss = divergence(student_logits,
    teacher_logits) on the generated portion only.
    """
    with torch.no_grad():
        # Teacher rollout — argmax (do_sample=False) for stable training data.
        # Drop `temperature` because it's silently ignored when do_sample=False
        # (HF warns about it).  Pass an explicit attention_mask of all-ones
        # because we set pad_token=eos_token, and HF cannot infer the mask
        # in that case for a single un-padded prompt.
        attn_mask = torch.ones_like(prompt_ids)
        gen = teacher.generate(
            prompt_ids, attention_mask=attn_mask,
            max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=teacher.config.eos_token_id,
            use_cache=True,
        )
        # Forward teacher once on the full sequence to grab logits for the loss.
        t_out = teacher(gen, return_dict=True)
        t_logits = t_out.logits[0, prompt_ids.shape[1]-1:-1].float()   # [T, V]

    s_out = draft(gen, return_dict=True)
    s_logits = s_out.logits[0, prompt_ids.shape[1]-1:-1].float()        # [T, V]
    return loss_fn(s_logits, t_logits)


# ---------------------------------------------------------------------------
# Tree-loss path: sample K student paths, target tree pass, student tree pass.
# ---------------------------------------------------------------------------

def compute_tree_loss(loss_fn, draft, teacher, prompt_ids,
                      K, L, draft_temp, teacher_temp):
    """
    Build a fresh K×L draft tree from the student, score it under both models,
    then call the tree loss on (q_probs_dict_grad, p_probs_dict, q_paths, L, K).
    """
    # 1. Pending token from teacher's last position (matches inference).
    with torch.no_grad():
        t_out = teacher(prompt_ids, use_cache=True, return_dict=True)
        p_cache = t_out.past_key_values
        p_probs_last = F.softmax(t_out.logits[:, -1, :] / teacher_temp, dim=-1)
        context_pending = torch.multinomial(p_probs_last, num_samples=1)

    # 2. Sample K student draft paths (no grad — paths are fixed inputs).
    with torch.no_grad():
        q_out_prefill = draft(prompt_ids, use_cache=True, return_dict=True)
        q_cache = q_out_prefill.past_key_values
        q_paths, _, _ = iid_draft(
            draft, q_cache, context_pending, K=K, L=L, q_temp=draft_temp,
        )

    # 3. Target tree forward (no grad — teacher is frozen).
    with torch.no_grad():
        _, _, _, p_probs_dict = target_tree_pass(
            teacher, p_cache, q_paths, K=K, L=L, p_temp=teacher_temp,
        )

    # 4. Student tree forward WITH grad → q_probs_dict_grad.
    q_probs_dict_grad = draft_tree_forward_with_grad(
        draft, prompt_ids, q_paths, L=L, q_temp=draft_temp,
    )

    # 5. Call the loss.
    return loss_fn(q_probs_dict_grad, p_probs_dict, q_paths, L, K)


# ---------------------------------------------------------------------------
# Validation: average per-prompt loss over up to VAL_PROMPTS prompts.
# Uses forward_kl as the comparable val metric for tree losses (the tree
# objectives themselves are not directly comparable across loss families).
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_val_loss(draft, teacher, tokenizer, val_prompts, args):
    draft.eval()
    losses = []
    for prompt in val_prompts[:VAL_PROMPTS]:
        ids = _prompt_to_ids(prompt, tokenizer, draft.device)
        # Use forward_kl on a short teacher rollout as a stable cross-loss proxy.
        loss = compute_flat_loss(
            FLAT_LOSSES["forward_kl"], draft, teacher, ids,
            max_new_tokens=32, teacher_temp=args.teacher_temp,
        )
        losses.append(loss.item())
    draft.train()
    return sum(losses) / max(1, len(losses))


# ---------------------------------------------------------------------------
# W&B setup with run-id resume (so a killed-and-resumed training continues
# the same dashboard URL instead of splitting across two runs).
# ---------------------------------------------------------------------------

def setup_wandb(args, output_dir, resumed: bool):
    """Initialise W&B, resuming the saved run id if <output>/wandb_run.json exists."""
    if args.no_wandb:
        print("[wandb] disabled (--no_wandb)")
        return None
    try:
        import wandb
    except ImportError:
        print("[wandb] not installed — skipping (pip install wandb to enable)")
        return None

    meta_path = os.path.join(output_dir, "wandb_run.json")
    saved = None
    if resumed and os.path.isfile(meta_path) and not args.fresh_wandb:
        try:
            saved = json.load(open(meta_path, encoding="utf-8"))
        except Exception:
            saved = None

    run_name = f"{args.loss}_K{K}_L{L}_seed{args.seed}"
    init_kw = dict(project=WANDB_PROJECT, name=run_name,
                   tags=[args.loss, f"K{K}", f"L{L}"], config=vars(args))
    if saved:
        init_kw["id"]     = saved["run_id"]
        init_kw["resume"] = "must"
        try:
            run = wandb.init(**init_kw)
            print(f"[wandb] resumed run {saved['run_id']}: {run.url}")
            return run
        except Exception as e:
            print(f"[wandb] resume failed ({e}); starting fresh run")
            init_kw.pop("id", None)

    init_kw["resume"] = "allow"
    run = wandb.init(**init_kw)
    json.dump({"run_id": run.id, "name": run.name, "project": WANDB_PROJECT},
              open(meta_path, "w", encoding="utf-8"))
    print(f"[wandb] {run.url}")
    return run


# ---------------------------------------------------------------------------
# Checkpoint save / resume
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, scheduler, output_dir, name, state):
    """Write model + optimizer + scheduler + training state to <output>/<name>/."""
    target = os.path.join(output_dir, name)
    os.makedirs(target, exist_ok=True)
    if USE_LORA:
        model.save_pretrained(target)                       # PEFT-aware save
    else:
        model.save_pretrained(target, safe_serialization=True)
    torch.save({
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
    }, os.path.join(target, "optim.pt"))
    json.dump(state, open(os.path.join(target, "state.json"), "w"))


def try_resume(model, optimizer, scheduler, output_dir):
    """Load ckpt_latest if it exists.  Returns (start_step, training_state)."""
    latest = os.path.join(output_dir, "ckpt_latest")
    state_path = os.path.join(latest, "state.json")
    if not os.path.isfile(state_path):
        return 0, {}
    state = json.load(open(state_path, encoding="utf-8"))
    # Load model weights
    model_state = AutoModelForCausalLM.from_pretrained(
        latest, torch_dtype=torch.bfloat16,
    ).state_dict()
    model.load_state_dict(model_state, strict=False)
    # Load optimizer + scheduler
    optim_blob = torch.load(os.path.join(latest, "optim.pt"), map_location="cpu")
    optimizer.load_state_dict(optim_blob["optimizer"])
    if scheduler and optim_blob.get("scheduler"):
        scheduler.load_state_dict(optim_blob["scheduler"])
    print(f"[resume] restored step={state.get('step', 0)} from {latest}")
    return state.get("step", 0), state


# ---------------------------------------------------------------------------
# Model loading (Qwen3 only; A100; BF16; optional LoRA)
# ---------------------------------------------------------------------------

def load_models(draft_id: str, teacher_id: str):
    """Load draft + teacher on cuda:0 in bfloat16.  Teacher is frozen."""
    print(f"[load] tokenizer={draft_id}")
    tok = AutoTokenizer.from_pretrained(draft_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"[load] draft={draft_id}  (BF16, will be trained)")
    draft = AutoModelForCausalLM.from_pretrained(
        draft_id, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to("cuda")

    print(f"[load] teacher={teacher_id}  (BF16, frozen)")
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_id, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to("cuda").eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    if USE_LORA:
        from peft import LoraConfig, get_peft_model
        print(f"[load] wrapping draft in LoRA r={LORA_R} alpha={LORA_ALPHA}")
        lora_cfg = LoraConfig(
            r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        )
        draft = get_peft_model(draft, lora_cfg)
        draft.print_trainable_parameters()
    else:
        # Full fine-tuning — every parameter trainable.
        for p in draft.parameters():
            p.requires_grad_(True)

    return tok, draft, teacher


# ---------------------------------------------------------------------------
# Argparse + main
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="DistillSpec pipeline trainer (A100, Qwen3).")
    ap.add_argument("--loss",   required=True, choices=sorted(ALL_LOSSES.keys()),
                    help="Loss function to train with.  See losses/__init__.py.")
    ap.add_argument("--steps",  type=int, default=STEPS,
                    help=f"Total gradient-accumulation steps (default {STEPS}).")
    ap.add_argument("--lr",     type=float, default=LR,
                    help=f"Peak learning rate (default {LR}; use 1e-5 for bv/gbv_tree).")
    ap.add_argument("--seed",   type=int, default=SEED)
    ap.add_argument("--output", type=str, default=None,
                    help=f"Output dir for checkpoints (default {OUTPUT_ROOT}/<loss>).")
    ap.add_argument("--resume", action="store_true",
                    help="Resume from <output>/ckpt_latest if it exists.")
    ap.add_argument("--no_wandb",     action="store_true", help="Disable W&B logging.")
    ap.add_argument("--fresh_wandb",  action="store_true",
                    help="Start a new W&B run even when resuming (default: reuse run id).")
    ap.add_argument("--teacher_temp", type=float, default=TEACHER_TEMP)
    ap.add_argument("--draft_temp",   type=float, default=DRAFT_TEMP)
    return ap.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = args.output or os.path.join(OUTPUT_ROOT, args.loss)
    os.makedirs(output_dir, exist_ok=True)
    print(f"[output] {output_dir}")

    # Models
    tokenizer, draft, teacher = load_models(DRAFT_MODEL, TEACHER_MODEL)

    # Optimiser + linear warmup → constant LR
    trainable = [p for p in draft.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.999),
                                  weight_decay=0.0)   # DistillSpec uses no regularisation

    # LR schedule: linear warmup for WARMUP_STEPS optimizer steps, then cosine
    # decay to LR_MIN_RATIO × peak.  Matches DistillSpec (arXiv:2310.08461) which
    # uses linear warmup + cosine cooldown.  WARMUP_STEPS counts optimizer steps
    # (scheduler.step() fires once per GRAD_ACCUM training steps), so
    # WARMUP_STEPS=50 → 50 × GRAD_ACCUM = 400 training steps = 10 % of 4000.
    # Previously WARMUP_STEPS was set to 400 (training-step count, not opt-step
    # count), causing 80 % of the run to be in warmup with LR never reaching peak.
    LR_MIN_RATIO = 0.1                       # cosine decays to 10 % of peak LR
    total_opt_steps = args.steps // GRAD_ACCUM
    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return step / max(1, WARMUP_STEPS)
        # Cosine decay from 1.0 → LR_MIN_RATIO over remaining opt-steps
        progress = (step - WARMUP_STEPS) / max(1, total_opt_steps - WARMUP_STEPS)
        progress = min(progress, 1.0)
        import math
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return LR_MIN_RATIO + (1.0 - LR_MIN_RATIO) * cosine
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Resume?
    start_step, train_state = (0, {})
    if args.resume:
        start_step, train_state = try_resume(draft, optimizer, scheduler, output_dir)
    best_val_loss = train_state.get("best_val_loss", float("inf"))

    # W&B
    wandb_run = setup_wandb(args, output_dir, resumed=(start_step > 0))

    # Data
    print(f"[data] train = {dataset_path('gsm8k_train')}")
    print(f"[data] val   = {dataset_path('gsm8k_val')}")
    train_prompts = [json.loads(l)["prompt"]
                     for l in open(dataset_path("gsm8k_train"), encoding="utf-8")]
    val_prompts   = [json.loads(l)["prompt"]
                     for l in open(dataset_path("gsm8k_val"),   encoding="utf-8")]
    random.Random(args.seed).shuffle(train_prompts)
    print(f"[data] {len(train_prompts)} train / {len(val_prompts)} val prompts")

    # Dispatch
    loss_fn  = get_loss(args.loss)
    tree     = is_tree_loss(args.loss)
    print(f"[loss] {args.loss}  ({'tree' if tree else 'flat'})")

    # ── Training loop ─────────────────────────────────────────────────────────
    draft.train()
    optimizer.zero_grad(set_to_none=True)
    losses_log: List[float] = []
    t0 = time.time()

    for step in range(start_step, args.steps):
        prompt = train_prompts[step % len(train_prompts)]
        ids    = _prompt_to_ids(prompt, tokenizer, draft.device)

        if tree:
            loss = compute_tree_loss(loss_fn, draft, teacher, ids,
                                     K=K, L=L,
                                     draft_temp=args.draft_temp,
                                     teacher_temp=args.teacher_temp)
        else:
            loss = compute_flat_loss(loss_fn, draft, teacher, ids,
                                     max_new_tokens=MAX_NEW_TOKENS,
                                     teacher_temp=args.teacher_temp)

        # Gradient accumulation: scale by 1/GRAD_ACCUM, only step every GRAD_ACCUM micro-steps.
        (loss / GRAD_ACCUM).backward()
        if (step + 1) % GRAD_ACCUM == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        else:
            grad_norm = torch.tensor(0.0)

        losses_log.append(loss.item())

        # Console log
        if (step + 1) % LOG_EVERY == 0:
            avg = sum(losses_log[-LOG_EVERY:]) / LOG_EVERY
            elapsed = time.time() - t0
            print(f"step={step+1:5d}/{args.steps}  loss={avg:.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}  "
                  f"grad={grad_norm.item():.2f}  elapsed={elapsed/60:.1f}m")
            if wandb_run:
                wandb_run.log({
                    "train/loss":      avg,
                    "train/lr":        scheduler.get_last_lr()[0],
                    "train/grad_norm": grad_norm.item(),
                }, step=step + 1)

        # Validation + checkpoint best
        if (step + 1) % VAL_EVERY == 0:
            val_loss = compute_val_loss(draft, teacher, tokenizer, val_prompts, args)
            print(f"  [val] step={step+1}  val_loss={val_loss:.4f}  "
                  f"best={best_val_loss:.4f}")
            if wandb_run:
                wandb_run.log({"val/loss": val_loss}, step=step + 1)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(draft, optimizer, scheduler, output_dir, "ckpt_best",
                                state={"step": step + 1, "val_loss": val_loss,
                                       "best_val_loss": best_val_loss})
                print(f"  [val] saved ckpt_best (val_loss={val_loss:.4f})")

        # Rolling latest checkpoint
        if (step + 1) % SAVE_EVERY == 0:
            save_checkpoint(draft, optimizer, scheduler, output_dir, "ckpt_latest",
                            state={"step": step + 1,
                                   "best_val_loss": best_val_loss})

    # Final save
    save_checkpoint(draft, optimizer, scheduler, output_dir, "ckpt_final",
                    state={"step": args.steps, "best_val_loss": best_val_loss})
    print(f"\n[done] {args.loss}: total time = {(time.time()-t0)/60:.1f} min  "
          f"best_val_loss = {best_val_loss:.4f}")
    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
