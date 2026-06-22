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
    3. every VAL_EVERY steps → val/block_eff on gsm8k_val.jsonl,
       update best_val_block_eff, save ckpt_best, log to W&B.
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
import math
import os
import random
import sys
import time
from typing import Dict, List

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

# Local modules
from losses import (ALL_LOSSES, FLAT_LOSSES, TREE_LOSSES, get_loss, is_tree_loss,
                    is_offpolicy_tree_loss, is_enrichment_loss, is_flat_enrich_loss)
from data_io import get_path as dataset_path
from config  import DRAFT_MODEL, TEACHER_MODEL, DEFAULT_K, DEFAULT_L, DEFAULT_MAX_NEW_TOKENS, DEFAULT_TEMP, DEFAULT_SEED, block_eff

# verifiers/__init__.py adds the verifiers folder to sys.path so this works.
import verifiers  # noqa: F401  — side effect: sys.path injection
from util           import set_seed, load_prompts_jsonl, load_models as _sot_load
from inference_util import iid_draft, target_tree_pass
from main           import speculative_decoding_loop
from node           import Node  # class-level caches cleared each step to avoid id() reuse bugs


def _clear_node_caches():
    Node.naive_cache.clear()
    Node.spectr_cache.clear()
    Node.specinfer_cache.clear()
    Node.khisti_cache.clear()
from verifier        import TreeVerifier  # for --aux_mode depth_weight (expected_*_depths)
from verifier_safe  import VerifierError


# ═══════════════════════════════════════════════════════════════════════════
#  HARDCODED CONSTANTS — edit config.py for shared defaults; override here for
#  training-specific values, or pass CLI flags.
# ═══════════════════════════════════════════════════════════════════════════
# Training hyper-parameters
STEPS           = 4000                       # number of gradient-accum steps
GRAD_ACCUM      = 8                          # opt-steps = STEPS / GRAD_ACCUM = 500
LR              = 3e-5                       # bv_tree / gbv_tree may need 1e-5
WARMUP_STEPS    = 50                         # fallback only — overridden in main() to 10% of total_opt_steps
GRAD_CLIP       = 1.0                        # DistillSpec Table S1 (arXiv:2310.08461) — 1.0 is the LLM fine-tuning standard (LLaMA, GPT-3, Qwen3)
LR_MIN_RATIO    = 0.1                        # cosine decays to 10 % of peak LR
SEED            = DEFAULT_SEED

# Speculative-decoding shape (used by all *_tree losses) — defaults from config.py
K               = DEFAULT_K                  # number of draft paths — K=3 matches offline eval default (eval.py, results CSV)
L               = DEFAULT_L                  # draft block length
DRAFT_TEMP      = DEFAULT_TEMP               # q_temp for the draft model
VAL_K           = K                          # val tree paths — tied to K so training and validation always use the same tree shape
VAL_L           = L                          # val tree depth  — change to test different L without touching training
TEACHER_TEMP    = DEFAULT_TEMP               # for flat-loss teacher rollout — DistillSpec (arXiv:2310.08461) uses T=1.0
MAX_NEW_TOKENS  = DEFAULT_MAX_NEW_TOKENS     # generated sequence length for flat losses

# LoRA toggle (full fine-tune is default).  Set USE_LORA = True for adapter
# training — saves disk and lets you keep many checkpoints.  Full FT of
# Qwen3-0.6B in BF16 with AdamW state ≈ 12 GB so the A100 40 GB fits fine.
USE_LORA        = False
LORA_R          = 16
LORA_ALPHA      = 32
LORA_DROPOUT    = 0.05

# Validation & checkpointing cadence
VAL_EVERY       = 100                        # gradient-accum steps between val checks
VAL_PROMPTS     = 25                         # val loss averaged over this many prompts (kept small to limit val overhead)
SAVE_EVERY      = 200                        # ckpt_latest write cadence
LOG_EVERY       = 10                         # console + W&B step-log cadence

# Dataset names (resolved via data_io.get_path)
TRAIN_DATASET   = "gsm8k_train"
VAL_DATASET     = "gsm8k_val"

# Storage layout — change OUTPUT_ROOT to your preferred checkpoint dir
OUTPUT_ROOT     = os.path.join(os.path.dirname(__file__), "checkpoints")
WANDB_PROJECT   = "distillspec-pipeline"

# Maps each tree loss to the verifier mode used for val block_eff measurement.
# Losses not listed (kl_tree, rev_kl_tree, jsd_tree, flat losses) fall through to "traversal".
LOSS_TO_VERIFIER = {
    "naive_tree":         "naive",
    "naive_tree_full":    "naive",   # full-gradient (un-detached survival) variant
    "op_naive_tree":      "naive",   # off-policy: teacher greedy path, detached survival
    "op_naive_tree_full": "naive",   # off-policy: teacher greedy path, exact ∇E[τ]
    "nss_tree":      "nss",
    "specinfer_tree":"specinfer",
    "spectr_tree":   "spectr",
    "khisti_tree":   "khisti",
    "bv_tree":       "bv",
    "gbv_tree":      "gbv",
    "traversal_tree":"traversal",
}
# ═══════════════════════════════════════════════════════════════════════════


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
                       max_new_tokens=128):
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
# Flat enrichment loss: K stochastic teacher rollouts, student scored on each.
# Apples-to-apples with compute_flat_loss — same max_new_tokens, same JSD at
# every token.  Only variable vs jsd flat: do_sample=True (stochastic teacher)
# and K paths averaged.  K=1 isolates greedy-vs-stochastic; K>1 adds diversity.
# Cost: K × compute_flat_loss per step.
# ---------------------------------------------------------------------------

def compute_flat_enrich_loss(loss_fn, draft, teacher, prompt_ids,
                             K, max_new_tokens=128, teacher_temp=1.0):
    """Sample K stochastic teacher rollouts; score student on each; average loss.

    Returns (loss, path_diversity) where path_diversity is the fraction of token
    positions where at least one path disagrees with path 0 (0 = all paths identical,
    1 = all positions differ).  Used to diagnose whether K paths add novel contexts.
    """
    attn_mask = torch.ones_like(prompt_ids)
    losses = []
    gen_tokens = []   # collect generated token ids for diversity measurement
    for _ in range(K):
        with torch.no_grad():
            gen = teacher.generate(
                prompt_ids, attention_mask=attn_mask,
                max_new_tokens=max_new_tokens, do_sample=True,
                temperature=teacher_temp,
                pad_token_id=teacher.config.eos_token_id,
                use_cache=True,
            )
            gen_tokens.append(gen[0, prompt_ids.shape[1]:])   # [T_i]
            t_out    = teacher(gen, return_dict=True)
            t_logits = t_out.logits[0, prompt_ids.shape[1]-1:-1].float()   # [T, V]

        s_out    = draft(gen, return_dict=True)
        s_logits = s_out.logits[0, prompt_ids.shape[1]-1:-1].float()        # [T, V]
        losses.append(loss_fn(s_logits, t_logits))

    # Path diversity: fraction of positions where paths disagree (only when K > 1).
    if K > 1:
        min_len = min(t.shape[0] for t in gen_tokens)
        stacked = torch.stack([t[:min_len] for t in gen_tokens], dim=0)  # [K, T]
        path_diversity = (stacked != stacked[0:1]).any(dim=0).float().mean().item()
    else:
        path_diversity = 0.0

    return sum(losses) / K, path_diversity


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
# Off-policy tree loss: teacher's greedy sequence is the training path.
# This avoids on-policy survival collapse — teacher tokens have near-unit
# self-acceptance so survival products stay non-negligible at depth L.
# q_probs are scored with grad; p_probs are frozen teacher logits on same path.
# ---------------------------------------------------------------------------

def compute_offpolicy_tree_loss(loss_fn, draft, teacher, prompt_ids,
                                K, L, draft_temp, teacher_temp):
    """
    Use teacher's greedy rollout (L+1 tokens) as the single training path.
    Teacher scores its own tokens → α stays high → survival products non-negligible.
    Student is scored on the same path with grad → acceptance gradient flows cleanly.
    """
    with torch.no_grad():
        attn_mask = torch.ones_like(prompt_ids)
        gen = teacher.generate(
            prompt_ids, attention_mask=attn_mask,
            max_new_tokens=L + 1, do_sample=False,
            pad_token_id=teacher.config.eos_token_id,
            use_cache=True,
        )
        teacher_tokens = gen[0, prompt_ids.shape[1]:].tolist()  # L+1 tokens
        # Build single-path list: [context_pending, tok1, …, tokL].
        # target_tree_pass expects paths of length L+1.
        q_paths = [teacher_tokens[: L + 1]]

        t_cache = teacher(prompt_ids, use_cache=True, return_dict=True).past_key_values
        _, _, _, p_probs_dict = target_tree_pass(
            teacher, t_cache, q_paths, K=1, L=L, p_temp=teacher_temp,
        )

    q_probs_dict_grad = draft_tree_forward_with_grad(
        draft, prompt_ids, q_paths, L=L, q_temp=draft_temp,
    )

    return loss_fn(q_probs_dict_grad, p_probs_dict, q_paths, L, K)


# ---------------------------------------------------------------------------
# Enrichment tree loss: the K branches are sampled from the TEACHER (its own
# plausible continuations), not the draft.  Identical to compute_tree_loss
# except iid_draft runs on the teacher.  The draft is pulled toward the teacher
# by JSD at every node of that teacher-generated tree — covering the off-greedy
# branch states the verifier visits at inference, where flat JSD never trains.
# K=1 reduces to a single teacher path (the matched control); K>1 is enrichment.
# No acceptance/telescoping term: the loss is whatever loss_fn is (jsd_enrich).
# ---------------------------------------------------------------------------

def compute_enrichment_loss(loss_fn, draft, teacher, prompt_ids,
                            K, L, draft_temp, teacher_temp):
    """Teacher-enriched tree: sample K teacher branches, score under both models,
    then call the loss on (q_probs_dict_grad, p_probs_dict, q_paths, L, K)."""
    with torch.no_grad():
        # 1. Pending token from the teacher's last position.
        t_out = teacher(prompt_ids, use_cache=True, return_dict=True)
        p_cache_score = t_out.past_key_values
        p_probs_last = F.softmax(t_out.logits[:, -1, :] / teacher_temp, dim=-1)
        context_pending = torch.multinomial(p_probs_last, num_samples=1)

        # 2. Sample K TEACHER paths.  iid_draft expands/consumes its cache, so
        #    use a fresh teacher prefill here, separate from the scoring cache.
        p_cache_sample = teacher(prompt_ids, use_cache=True,
                                 return_dict=True).past_key_values
        q_paths, _, _ = iid_draft(
            teacher, p_cache_sample, context_pending, K=K, L=L, q_temp=teacher_temp,
        )

        # 3. Target tree pass — teacher distributions at every node (JSD targets).
        _, _, _, p_probs_dict = target_tree_pass(
            teacher, p_cache_score, q_paths, K=K, L=L, p_temp=teacher_temp,
        )

    # 4. Draft tree forward WITH grad over the teacher-generated tree.
    q_probs_dict_grad = draft_tree_forward_with_grad(
        draft, prompt_ids, q_paths, L=L, q_temp=draft_temp,
    )

    # 5. Per-node loss (jsd_enrich → JSD at every node).
    return loss_fn(q_probs_dict_grad, p_probs_dict, q_paths, L, K)


# ---------------------------------------------------------------------------
# (3) Depth-as-weight: scalar E[τ_V] over the draft tree, used to MULTIPLY a
# flat loss.  The depth has NO gradient (pure-Python DP) — this is per-prompt
# loss reweighting, not an acceptance gradient.  Cost: one extra target tree
# pass per step.  See --aux_mode depth_weight.
# ---------------------------------------------------------------------------

@torch.no_grad()
def expected_depth_scalar(draft, teacher, prompt_ids, K, L, verifier,
                          draft_temp, teacher_temp) -> float:
    """E[accepted depth] for `verifier` on a fresh student draft tree (no grad)."""
    t_out = teacher(prompt_ids, use_cache=True, return_dict=True)
    p_cache = t_out.past_key_values
    p_probs_last = F.softmax(t_out.logits[:, -1, :] / teacher_temp, dim=-1)
    context_pending = torch.multinomial(p_probs_last, num_samples=1)

    q_out = draft(prompt_ids, use_cache=True, return_dict=True)
    q_paths, _, _ = iid_draft(draft, q_out.past_key_values, context_pending,
                              K=K, L=L, q_temp=draft_temp)
    q_prefixes, _, _, p_probs_dict = target_tree_pass(
        teacher, p_cache, q_paths, K=K, L=L, p_temp=teacher_temp)
    q_probs_dict = draft_tree_forward_with_grad(draft, prompt_ids, q_paths,
                                                L=L, q_temp=draft_temp)

    tv = TreeVerifier(q_paths, q_prefixes, q_probs_dict, p_probs_dict)
    depths = getattr(tv, f"expected_{verifier}_depths")(L)   # list over cutoffs
    return float(depths[-1])                                 # full-depth E[τ_V]


# ---------------------------------------------------------------------------
# Validation: block efficiency via eval.speculative_decode_one (same code path
# as offline eval.py).  Verifier mode matched to training loss; traversal for
# losses with no direct pairing (best general BE, Thomas et al. 2026 Table 2).
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_val_metrics(draft, teacher, tokenizer, val_prompts, args):
    """Returns (aggregate_block_eff, per_prompt_block_eff) over VAL_PROMPTS prompts.

    per_prompt_block_eff is a {prompt_index: block_eff} dict — used by the caller
    to compute a backward-transfer (forgetting) metric without any extra forward
    passes: it is the SAME val pass, just keeping per-prompt numbers instead of
    only their aggregate.

    VerifierError (from verifier_safe.py) is caught per-prompt so a bug in
    verifier.py does not abort training.  Skipped prompts are excluded from the
    block_eff denominator; if all prompts fail, aggregate returns nan.
    """
    mode = LOSS_TO_VERIFIER.get(args.loss, "traversal")
    draft.eval()
    total_gen, total_calls, skipped = 0, 0, 0
    per_prompt: dict[int, float] = {}
    for i, prompt in enumerate(val_prompts[:VAL_PROMPTS]):
        teacher._spec_prompt_idx = i
        try:
            speculative_decoding_loop(
                p_model=teacher, q_model=draft, tok=tokenizer,
                prompt=prompt, verification_algo=mode,
                max_new_tokens=MAX_NEW_TOKENS, K=VAL_K, L=VAL_L,
                p_temp=args.val_temp, q_temp=args.val_temp,
            )
        except VerifierError as ve:
            skipped += 1
            print(f"  [val-skip] prompt {i}: {ve}")
            continue
        s = teacher._spec_run_stats
        total_gen   += s["gen_tokens"]
        total_calls += s["target_calls"]
        per_prompt[i] = block_eff(s["gen_tokens"], s["target_calls"])
    if skipped:
        print(f"  [val] {skipped}/{VAL_PROMPTS} prompts skipped (verifier errors)")
    draft.train()
    return block_eff(total_gen, total_calls), per_prompt


def _update_forgetting(best_per_prompt: dict, val_pp: dict) -> float:
    """Backward-transfer (forgetting) metric (Lopez-Paz & Ranzato 2017).

    For each val prompt, track its best-ever block_eff.  Forgetting at this step
    is the mean drop from each prompt's personal best to its current value:
        forgetting = mean_i max(0, best_i - current_i)
    0 = no prompt has regressed below its peak; large = the student learned
    some prompts then lost them (signature of H3: teacher confusing the student).
    Updates best_per_prompt in place.  Uses the SAME val prompts as block_eff —
    no extra forward passes.
    """
    if not val_pp:
        return 0.0
    drops = []
    for idx, be in val_pp.items():
        prev_best = best_per_prompt.get(idx, be)
        drops.append(max(0.0, prev_best - be))
        best_per_prompt[idx] = max(prev_best, be)
    return sum(drops) / len(drops)


# ---------------------------------------------------------------------------
# W&B setup with run-id resume (so a killed-and-resumed training continues
# the same dashboard URL instead of splitting across two runs).
# ---------------------------------------------------------------------------

def run_slug(args) -> str:
    """Loss + dataset component of the run identifier, shared by the checkpoint dir
    and the W&B run name. L is omitted for flat/flat-enrich losses where tree depth
    is not a parameter; K is always included since it may distinguish enrich width."""
    uses_L = is_tree_loss(args.loss) or is_enrichment_loss(args.loss)
    if uses_L:
        slug = f"{args.loss}_K{args.K}_L{args.L}_{args.train_dataset}_s{args.seed}"
    else:
        slug = f"{args.loss}_K{args.K}_{args.train_dataset}_s{args.seed}"
    if args.aux_mode == "depth_weight":
        tag = "lin" if args.depth_linear else f"lam{args.depth_lambda}"
        slug += f"+dw_{args.aux_loss or 'naive_tree'}_{tag}"
    elif args.aux_loss:
        slug += f"+{args.aux_loss}x{args.aux_weight}"
    return slug


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
        except Exception as e:
            print(f"[wandb] WARNING: could not read {meta_path} ({e}) — starting fresh W&B run")
            saved = None
    elif resumed and not os.path.isfile(meta_path):
        print(f"[wandb] no saved run ID at {meta_path} — starting fresh W&B run")

    tags = [args.loss, args.train_dataset, f"K{K}", f"L{L}"]
    if args.aux_mode == "depth_weight":
        tags.append(f"depthw:{args.aux_loss or 'naive_tree'}")
        tags.append("lin" if args.depth_linear else f"lam{args.depth_lambda}")
    elif args.aux_loss:
        tags.append(f"aux:{args.aux_loss}")
    run_name = run_slug(args)
    init_kw = dict(project=WANDB_PROJECT, name=run_name,
                   tags=tags, config=vars(args))
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
    json.dump({"run_id": run.id, "name": run.name, "project": WANDB_PROJECT, "url": run.url},
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
    # Load optimizer + scheduler
    optim_blob = torch.load(os.path.join(latest, "optim.pt"), map_location="cpu")
    optimizer.load_state_dict(optim_blob["optimizer"])
    if scheduler and optim_blob.get("scheduler"):
        scheduler.load_state_dict(optim_blob["scheduler"])
    print(f"[resume] restored step={step} from {latest}")
    return step, state


# ---------------------------------------------------------------------------
# Model loading (Qwen3 only; A100; BF16; optional LoRA)
# ---------------------------------------------------------------------------

def load_models(draft_id: str, teacher_id: str, device: str = "cuda", load_in_4bit: bool = False):
    """Load draft + teacher via verifiers/util.load_models, then configure for training.
    util.load_models handles device selection and BF16 loading; we add the training-specific
    setup: re-enable grad, freeze teacher, optionally wrap draft in LoRA."""
    tok, teacher, draft = _sot_load(teacher_id, draft_id, device=device, load_in_4bit=load_in_4bit)
    # util.load_models disables grad globally (inference default); restore for training.
    torch.set_grad_enabled(True)
    teacher.eval()
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

    draft.train()
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
    ap.add_argument("--K", type=int, default=DEFAULT_K,
                    help=f"Tree width — number of paths per step (default {DEFAULT_K}). "
                         f"For jsd_enrich: K=1 is the matched single-path control, "
                         f"K>1 is the enrichment (teacher teaches additional paths). "
                         f"Validation tree width follows --K.")
    ap.add_argument("--L", type=int, default=DEFAULT_L,
                    help=f"Tree depth / draft block length (default {DEFAULT_L}).")
    ap.add_argument("--lr",     type=float, default=LR,
                    help=f"Peak learning rate (default {LR}; use 1e-5 for bv/gbv_tree).")
    ap.add_argument("--train_dataset", default=TRAIN_DATASET,
                    help=f"Training dataset name passed to data_io.get_path (default {TRAIN_DATASET}).")
    ap.add_argument("--val_dataset",   default=VAL_DATASET,
                    help=f"Validation dataset name passed to data_io.get_path (default {VAL_DATASET}).")
    ap.add_argument("--seed",   type=int, default=SEED)
    ap.add_argument("--output", type=str, default=None,
                    help=f"Output dir for checkpoints (default {OUTPUT_ROOT}/<loss>[+<aux>x<weight>]).")
    ap.add_argument("--resume", action="store_true",
                    help="Resume from <output>/ckpt_latest if it exists.")
    ap.add_argument("--no_wandb",     action="store_true", help="Disable W&B logging.")
    ap.add_argument("--fresh_wandb",  action="store_true",
                    help="Start a new W&B run even when resuming (default: reuse run id).")
    ap.add_argument("--teacher_temp", type=float, default=TEACHER_TEMP)
    ap.add_argument("--draft_temp",   type=float, default=DRAFT_TEMP)
    ap.add_argument("--device",       default="cuda",
                    help="CUDA device, e.g. cuda:1 (default: auto-select freest GPU)")
    ap.add_argument("--teacher", type=str, default=TEACHER_MODEL,
                    help=f"Teacher model name or local path (default: {TEACHER_MODEL}).")
    ap.add_argument("--load_in_4bit", action="store_true",
                    help="Load teacher in 4-bit NF4 via bitsandbytes. Needed for large "
                         "teachers (e.g. 32B) on GPUs where bf16 does not fit.")
    ap.add_argument("--aux_loss",   type=str, default=None,
                    choices=sorted(ALL_LOSSES.keys()),
                    help="Optional auxiliary loss: total = primary + aux_weight * aux. "
                         "Typical use: --loss forward_kl --aux_loss naive_tree --aux_weight 0.1")
    ap.add_argument("--aux_weight", type=float, default=0.1,
                    help="Scalar weight applied to the auxiliary loss (default 0.1).")
    ap.add_argument("--aux_mode", choices=["add", "depth_weight"], default="add",
                    help="'add' (default): total = primary + aux_weight*aux.  "
                         "'depth_weight': multiply the (flat) primary loss by "
                         "exp(depth_lambda*(d - EMA(d))), where d = E[tau_V] of the "
                         "verifier named by --aux_loss.  Depth has NO gradient — this "
                         "is per-prompt loss reweighting (researcher's scalar scheme).")
    ap.add_argument("--depth_lambda", type=float, default=0.0,
                    help="Signed exponent for --aux_mode depth_weight.  >0 amplify "
                         "loss on deep-tree prompts, <0 amplify shallow, 0 = plain "
                         "flat (control).  EMA-centred so E[w]~=1 (no LR confound).")
    ap.add_argument("--depth_linear", action="store_true",
                    help="--aux_mode depth_weight: weight = d / EMA(d) — the "
                         "researcher's 'tree_depth * loss', mean-normalised so E[w]~=1 "
                         "(linear in depth, but no LR confound; self-adapts as d drifts). "
                         "Ignores --depth_lambda.")
    ap.add_argument("--warmup_steps", type=int, default=None,
                    help="LR warmup in optimizer steps (scheduler.step() calls). "
                         "Default: auto = 10%% of total_opt_steps (steps // GRAD_ACCUM). "
                         "Saved in state.json and restored on resume so the curve is continuous.")
    ap.add_argument("--val_temp", type=float, default=0.2,
                    help="Sampling temperature for val block_eff decoding.  Low (0.2) "
                         "is near-deterministic → far lower run-to-run variance than "
                         "the 0.8 training temp.  Cannot be 0 (softmax/temp divide).")
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    # Tree shape is module-global (used by the loss helpers, setup_wandb run name,
    # and validation).  Override from --K/--L so train and val always share shape.
    global K, L, VAL_K, VAL_L
    K, L = args.K, args.L
    VAL_K, VAL_L = K, L

    output_dir = args.output or os.path.join(OUTPUT_ROOT, run_slug(args))
    os.makedirs(output_dir, exist_ok=True)
    print(f"[output] {output_dir}")

    # Warn if starting fresh over an existing checkpoint
    if not args.resume:
        _state_path = os.path.join(output_dir, "ckpt_latest", "state.json")
        if os.path.isfile(_state_path):
            _s = json.load(open(_state_path))
            _step = _s.get("step", 0)
            _be   = _s.get("best_val_block_eff", 0.0)
            _cmd  = " ".join(_s["cmd"]) + " --resume" if "cmd" in _s else "(add --resume to this command)"
            print(f"\n*** WARNING: existing checkpoint at step={_step} "
                  f"best_be={_be:.3f} will be OVERWRITTEN ***")
            print(f"*** To continue from it run: {_cmd} ***\n")

    # Models
    tokenizer, draft, teacher = load_models(DRAFT_MODEL, args.teacher, device=args.device,
                                             load_in_4bit=args.load_in_4bit)

    # Optimiser + linear warmup → constant LR
    trainable = [p for p in draft.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.999),
                                  weight_decay=0.0)   # DistillSpec uses no regularisation

    # LR schedule: linear warmup then cosine decay to LR_MIN_RATIO × peak.
    # Both sched_steps and warmup_opt_steps are anchored to the values from the
    # run's first launch (saved in state.json) so the LR curve is continuous on
    # resume — restarting with a different --steps or --warmup_steps doesn't
    # reshape the already-completed portion of the schedule.
    sched_steps = args.steps
    warmup_opt_steps_override = None   # restored from state.json on resume if present
    if args.resume:
        _sp = os.path.join(output_dir, "ckpt_latest", "state.json")
        if os.path.isfile(_sp):
            _saved = json.load(open(_sp, encoding="utf-8"))
            sched_steps = _saved.get("sched_steps", args.steps)
            warmup_opt_steps_override = _saved.get("warmup_opt_steps", None)
            if sched_steps != args.steps:
                print(f"[lr] schedule horizon anchored to original {sched_steps} steps "
                      f"(--steps={args.steps}); steps beyond {sched_steps} run at "
                      f"LR_MIN_RATIO={LR_MIN_RATIO}*peak (no LR jump on resume)")
    total_opt_steps = sched_steps // GRAD_ACCUM
    # Warmup: explicit --warmup_steps overrides auto; on resume, the saved value
    # takes precedence over both so the curve stays continuous across restarts.
    if warmup_opt_steps_override is not None:
        warmup_opt_steps = warmup_opt_steps_override
    elif args.warmup_steps is not None:
        warmup_opt_steps = args.warmup_steps
    else:
        warmup_opt_steps = max(1, total_opt_steps // 10)   # 10% of opt-steps
    print(f"[lr] total_opt_steps={total_opt_steps}  warmup_opt_steps={warmup_opt_steps} "
          f"({100*warmup_opt_steps/max(1,total_opt_steps):.1f}%)")

    def lr_lambda(step):
        if step < warmup_opt_steps:
            return step / max(1, warmup_opt_steps)
        # Cosine decay from 1.0 → LR_MIN_RATIO over remaining opt-steps
        progress = (step - warmup_opt_steps) / max(1, total_opt_steps - warmup_opt_steps)
        progress = min(progress, 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return LR_MIN_RATIO + (1.0 - LR_MIN_RATIO) * cosine
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Resume?
    start_step, train_state = (0, {})
    if args.resume:
        start_step, train_state = try_resume(draft, optimizer, scheduler, output_dir)
    best_val_block_eff = train_state.get("best_val_block_eff", 0.0)

    # W&B
    wandb_run = setup_wandb(args, output_dir, resumed=(start_step > 0))

    # Data
    print(f"[data] train = {dataset_path(args.train_dataset)}")
    print(f"[data] val   = {dataset_path(args.val_dataset)}")
    train_prompts = load_prompts_jsonl(dataset_path(args.train_dataset))
    val_prompts   = load_prompts_jsonl(dataset_path(args.val_dataset))
    random.Random(args.seed).shuffle(train_prompts)
    print(f"[data] {len(train_prompts)} train / {len(val_prompts)} val prompts")

    # Dispatch
    loss_fn       = get_loss(args.loss)
    tree          = is_tree_loss(args.loss)
    offpolicy     = is_offpolicy_tree_loss(args.loss)
    enrichment    = is_enrichment_loss(args.loss)
    flat_enrich   = is_flat_enrich_loss(args.loss)
    _mode = ("flat enrichment" if flat_enrich else
             "enrichment tree" if enrichment else
             "off-policy tree" if offpolicy else
             "tree" if tree else "flat")
    print(f"[loss] {args.loss}  ({_mode})")

    aux_loss_fn = get_loss(args.aux_loss) if args.aux_loss else None
    aux_is_tree = is_tree_loss(args.aux_loss) if args.aux_loss else False
    if aux_loss_fn is not None:
        if args.aux_mode == "depth_weight":
            _dw_tag = "lin" if args.depth_linear else f"lam={args.depth_lambda}"
            print(f"[aux]  {args.aux_loss}  ({'tree' if aux_is_tree else 'flat'})  "
                  f"depth_weight={_dw_tag}")
        else:
            print(f"[aux]  {args.aux_loss}  ({'tree' if aux_is_tree else 'flat'})  "
                  f"weight={args.aux_weight}")

    # ── Training loop ─────────────────────────────────────────────────────────
    draft.train()
    optimizer.zero_grad(set_to_none=True)
    losses_log: List[float] = []
    depth_ema: float | None = None   # running mean of E[tau_V] for --aux_mode depth_weight
    depth_w = 1.0
    depth_d = 0.0                     # last raw E[tau_V] (logged so the sweep is observable)
    path_div = 0.0                    # last flat_enrich teacher-path diversity (K>1 only)
    best_per_prompt: dict[int, float] = {}   # each val prompt's best-ever block_eff (forgetting)
    t0 = time.time()

    for step in range(start_step, args.steps):
        prompt = train_prompts[step % len(train_prompts)]
        ids    = torch.tensor(tokenizer.encode(prompt), device=draft.device, dtype=torch.long).unsqueeze(0)

        if flat_enrich:
            loss, path_div = compute_flat_enrich_loss(loss_fn, draft, teacher, ids,
                                                      K=K, max_new_tokens=MAX_NEW_TOKENS,
                                                      teacher_temp=args.teacher_temp)
        elif enrichment:
            loss = compute_enrichment_loss(loss_fn, draft, teacher, ids,
                                           K=K, L=L,
                                           draft_temp=args.draft_temp,
                                           teacher_temp=args.teacher_temp)
        elif offpolicy:
            loss = compute_offpolicy_tree_loss(loss_fn, draft, teacher, ids,
                                              K=K, L=L,
                                              draft_temp=args.draft_temp,
                                              teacher_temp=args.teacher_temp)
        elif tree:
            loss = compute_tree_loss(loss_fn, draft, teacher, ids,
                                     K=K, L=L,
                                     draft_temp=args.draft_temp,
                                     teacher_temp=args.teacher_temp)
        else:
            loss = compute_flat_loss(loss_fn, draft, teacher, ids,
                                     max_new_tokens=MAX_NEW_TOKENS)

        if args.aux_mode == "depth_weight":
            # (3) Multiply the (flat) primary loss by a detached depth weight.
            verifier = LOSS_TO_VERIFIER.get(args.aux_loss, "naive")
            _clear_node_caches()
            d = expected_depth_scalar(draft, teacher, ids, K, L, verifier,
                                      args.draft_temp, args.teacher_temp)
            depth_ema = d if depth_ema is None else 0.9 * depth_ema + 0.1 * d
            if args.depth_linear:
                # Researcher's literal "tree_depth * loss", mean-normalised so
                # E[w]~=1 (w proportional to depth, but no LR confound).
                depth_w = d / max(depth_ema, 1e-6)
            else:
                depth_w = math.exp(args.depth_lambda * (d - depth_ema))   # E[w]~=1
            depth_d = d
            loss = depth_w * loss
        elif aux_loss_fn is not None:
            if aux_is_tree:
                aux = compute_tree_loss(aux_loss_fn, draft, teacher, ids,
                                        K=K, L=L,
                                        draft_temp=args.draft_temp,
                                        teacher_temp=args.teacher_temp)
            else:
                aux = compute_flat_loss(aux_loss_fn, draft, teacher, ids,
                                        max_new_tokens=MAX_NEW_TOKENS)
            loss = loss + args.aux_weight * aux

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

        # Console log + W&B train metrics
        if (step + 1) % LOG_EVERY == 0:
            avg = sum(losses_log[-LOG_EVERY:]) / LOG_EVERY
            elapsed = time.time() - t0
            print(f"step={step+1:5d}/{args.steps}  loss={avg:.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}  "
                  f"grad={grad_norm.item():.2f}  "
                  f"{f'pathdiv={path_div:.3f}  ' if (flat_enrich and K > 1) else ''}"
                  f"elapsed={elapsed/60:.1f}m")

            # Compute val metrics at val steps BEFORE logging so train + val go
            # into a single wandb.log() call — two separate calls at the same
            # step cause the second to be silently dropped in wandb ≥0.15.
            val_be = None
            val_forget = None
            if (step + 1) % VAL_EVERY == 0:
                _clear_node_caches()
                val_be, _val_pp = compute_val_metrics(draft, teacher, tokenizer, val_prompts, args)
                val_forget = _update_forgetting(best_per_prompt, _val_pp)
                print(f"  [val] step={step+1}  block_eff={val_be:.3f}  "
                      f"best={best_val_block_eff:.3f}  forget={val_forget:.3f}")

            if wandb_run:
                wandb_run.log({
                    "train/loss":      avg,
                    "train/lr":        scheduler.get_last_lr()[0],
                    "train/grad_norm": grad_norm.item(),
                    **({"train/path_diversity": path_div}
                       if (flat_enrich and K > 1) else {}),
                    **({"train/depth_w": depth_w, "train/depth_d": depth_d}
                       if args.aux_mode == "depth_weight" else {}),
                    **({"val/block_eff": val_be} if val_be is not None else {}),
                    **({"val/forgetting": val_forget} if val_forget is not None else {}),
                }, step=step + 1)

        # Validation + checkpoint best (val_be already computed above if LOG step)
        if (step + 1) % VAL_EVERY == 0:
            if (step + 1) % LOG_EVERY != 0:
                # VAL_EVERY not a multiple of LOG_EVERY — compute val now
                _clear_node_caches()
                val_be, _val_pp = compute_val_metrics(draft, teacher, tokenizer, val_prompts, args)
                val_forget = _update_forgetting(best_per_prompt, _val_pp)
                print(f"  [val] step={step+1}  block_eff={val_be:.3f}  "
                      f"best={best_val_block_eff:.3f}  forget={val_forget:.3f}")
                if wandb_run:
                    wandb_run.log({"val/block_eff": val_be,
                                   "val/forgetting": val_forget}, step=step + 1)
            if val_be > best_val_block_eff:
                best_val_block_eff = val_be
                save_checkpoint(draft, optimizer, scheduler, output_dir, "ckpt_best",
                                state={"step": step + 1, "val_block_eff": val_be,
                                       "best_val_block_eff": best_val_block_eff})
                print(f"  [val] saved ckpt_best (block_eff={val_be:.3f})")
                if wandb_run:
                    wandb_run.summary["val/best_block_eff"] = best_val_block_eff

        # Rolling latest checkpoint
        if (step + 1) % SAVE_EVERY == 0:
            save_checkpoint(draft, optimizer, scheduler, output_dir, "ckpt_latest",
                            state={"step": step + 1,
                                   "best_val_block_eff": best_val_block_eff,
                                   "sched_steps": sched_steps,
                                   "warmup_opt_steps": warmup_opt_steps,
                                   "cmd": sys.argv})

    # Final save — refresh the rolling ckpt_latest (no separate ckpt_final dir,
    # so a multi-combo sweep keeps only ckpt_best + ckpt_latest per run and does
    # not blow the disk quota).  ckpt_best holds the val-best model.
    save_checkpoint(draft, optimizer, scheduler, output_dir, "ckpt_latest",
                    state={"step": args.steps, "best_val_block_eff": best_val_block_eff,
                           "sched_steps": sched_steps, "warmup_opt_steps": warmup_opt_steps,
                           "cmd": sys.argv})
    print(f"\n[done] {args.loss}: total time = {(time.time()-t0)/60:.1f} min  "
          f"best_val_block_eff = {best_val_block_eff:.3f}")
    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
