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
import shutil
import subprocess
import sys
import time
from typing import Dict, List

# CUDA env vars MUST be set before `import torch` (torch reads them at CUDA init).
# setdefault so an explicit prefix / activate export still wins. This makes the
# settings independent of how the venv was created — setup_a100.sh injects them
# into venv/bin/activate, but a manual `python -m venv && pip install` flow skips
# that script and would otherwise run unprotected (the fragmentation OOMs on the
# 1.7B/32B pair, 2026-07-01, were exactly this: a manually-set-up box with
# PYTORCH_CUDA_ALLOC_CONF unset).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TORCH_CUDNN_SDPA_ENABLED", "0")

import torch
import torch.nn.functional as F

# Local modules
from losses import (ALL_LOSSES, FLAT_LOSSES, TREE_LOSSES, get_loss, is_tree_loss,
                    is_offpolicy_tree_loss, is_enrichment_loss, is_flat_enrich_loss,
                    is_prefix_overlap_loss)
from data_io import get_path as dataset_path
from config  import DRAFT_MODEL, TEACHER_MODEL, DEFAULT_K, DEFAULT_L, DEFAULT_MAX_NEW_TOKENS, DEFAULT_TEMP, DEFAULT_SEED, block_eff, BE_EASY_FRAC, BE_HARD_FRAC

# verifiers/__init__.py adds the verifiers folder to sys.path so this works.
import verifiers  # noqa: F401  — side effect: sys.path injection
from util           import set_seed, load_prompts_jsonl
from passk_utils     import pass_at_k, extract_answer, extract_gold, is_correct
from main           import speculative_decoding_loop
from node           import Node  # class-level caches cleared each step to avoid id() reuse bugs


def _clear_node_caches():
    Node.naive_cache.clear()
    Node.spectr_cache.clear()
    Node.specinfer_cache.clear()
    Node.khisti_cache.clear()
from verifier_safe  import VerifierError

from diagnostics import compute_train_diag_scalars
from losses.compute import (draft_tree_forward_with_grad, compute_flat_loss,
                             compute_flat_enrich_loss, compute_tree_loss,
                             compute_offpolicy_tree_loss, compute_enrichment_loss,
                             compute_prefix_overlap_loss,
                             compute_prefix_overlap_multiroot_loss, expected_depth_scalar)
from losses import LOSS_TO_VERIFIER
from validation import compute_val_metrics, _update_forgetting
from checkpointing import load_models, save_checkpoint, try_resume
from wandb_utils import run_slug, setup_wandb

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
VAL_EVERY       = 400                        # gradient-accum steps between val checks
VAL_PROMPTS     = 100                        # larger set → lower SE, less winner's-curse bias
SAVE_EVERY      = 400                        # ckpt_latest write cadence (matches VAL_EVERY)
LOG_EVERY       = 10                         # console + W&B step-log cadence
VAL_EMA_ALPHA   = 0.3                        # smoothed-val EMA weight for checkpoint selection (α=0.3 → ~3-4 check window)
EARLY_STOP_PAT  = 15                         # default patience: 15 × 400 = 6000 steps without smoothed improvement
EARLY_STOP_DELTA = 0.0                       # min improvement threshold (0 = strict; >0 ignores noise-level fluctuations)

# Dataset names (resolved via data_io.get_path)
TRAIN_DATASET   = "gsm8k_train"
VAL_DATASET     = "gsm8k_val"

# Storage layout — change OUTPUT_ROOT to your preferred checkpoint dir
OUTPUT_ROOT     = os.path.join(os.path.dirname(__file__), "checkpoints")
WANDB_PROJECT   = "distillspec-pipeline"

# ═══════════════════════════════════════════════════════════════════════════


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
                    help=f"Tree depth / draft block length (default {DEFAULT_L}). "
                         f"For prefix_overlap this is the teacher-continuation / window "
                         f"length over which prefix overlap is summed.")
    ap.add_argument("--prefix_teacher_topk", type=int, default=0,
                    help="prefix_overlap only: restrict the TEACHER's sampling to its top-k "
                         "tokens per step (top_k passed to teacher.generate). 0 = disabled "
                         "(current behaviour). Interpretation (b) of Rahul's 'top-k': narrows "
                         "the sampled continuation to high-prob teacher tokens the draft can "
                         "actually match, mitigating the prob-objective's tail-token gradient "
                         "collapse — WITHOUT changing the objective math. Distinct from K (tree "
                         "branches) and M (--prefix_M rollouts). Note: narrowing sampling reduces "
                         "context diversity, so it partially trades against the --prefix_M lever.")
    ap.add_argument("--prefix_M", type=int, default=4,
                    help="prefix_overlap single-root only: M = number of teacher "
                         "continuations per prompt for the Monte-Carlo estimator "
                         "(doc gives no default; tunable). Distinct from --K (tree width). "
                         "Ignored when --prefix_root_spacing > 0 (multi-root is M=1).")
    ap.add_argument("--prefix_root_spacing", type=int, default=0,
                    help="prefix_overlap only: 0 = single root (the prompt). >0 = "
                         "multi-root (doc §5): one teacher rollout of length "
                         "--prefix_rollout_len, roots every N tokens, M=1 per root.")
    ap.add_argument("--prefix_rollout_len", type=int, default=MAX_NEW_TOKENS,
                    help=f"prefix_overlap multi-root only: teacher rollout length to "
                         f"slide root windows over (default {MAX_NEW_TOKENS}).")
    ap.add_argument("--prefix_aux", choices=["ce", "jsd"], default="ce",
                    help="prefix_overlap only: secondary term on the SAME continuations. "
                         "'ce' = doc §7 cross-entropy (cheap, reuses token log-probs); "
                         "'jsd' = symmetric JSD (beyond the doc; one extra teacher forward).")
    ap.add_argument("--prefix_aux_weight", type=float, default=0.0,
                    help="prefix_overlap only: λ for the secondary (--prefix_aux) term "
                         "(0 = off; doc recommends a small positive value). Do NOT also "
                         "pass --aux_loss — the secondary term is computed internally.")
    ap.add_argument("--prefix_anneal_steps", type=int, default=0,
                    help="prefix_overlap only: if >0, linearly anneal --prefix_aux_weight "
                         "from its value down to 0 over this many steps. CE carries the "
                         "cold start; prefix term takes over as a_i grow. 0 = no anneal.")
    ap.add_argument("--prefix_objective", choices=["prob", "logprob", "traversal", "nss"],
                    default="prob",
                    help="prefix_overlap only: 'prob' = doc's Σ_t qθ(P_{1:t}) (exact "
                         "E[LCP]; gradient collapses from cold start). 'logprob' = "
                         "Σ_t log qθ(P_{1:t}) = position-weighted CE (trainability "
                         "variant, NOT the doc's objective; gradient never collapses). "
                         "'traversal' = analytical traversal-BE surrogate with weights "
                         "Σ_{d≥i} K(1-α_d)^{K-1}α_d (adaptive in α AND --K; independent-"
                         "branches approximation to the traversal verifier, not exact). "
                         "'nss' = NSS-aligned one-sided CE: -Σ_i (L-i+1)·𝟙[p_i>q_i]·log q_i; "
                         "gradient only where draft lags teacher; one extra teacher forward/step.")
    ap.add_argument("--prefix_min_root", type=int, default=0,
                    help="prefix_overlap multi-root only: skip roots before this rollout "
                         "position (0 = all roots, default). Set to rollout_len//2 for "
                         "deep-bias training: only supervise the second half of each "
                         "teacher rollout, teaching the draft to stay on track deep in "
                         "context. Intended for warm-start runs (jsd_flat init + deep-only "
                         "prefix supervision). Ignored when prefix_root_spacing=0.")
    ap.add_argument("--prefix_random_offset", action="store_true",
                    help="prefix_overlap multi-root only: start roots at a random offset "
                         "o~Unif{0..N-1} each step (doc §5 uniform-over-positions) "
                         "instead of a fixed 0 (every-Nth objective).")
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
    # --- State-distribution interventions for jsd_flat_enrich (all off by default) ---
    ap.add_argument("--draft_prefix_max", type=int, default=0,
                    help="jsd_flat_enrich only. >0 enables draft-conditioned rollout: the "
                         "draft generates a prefix of random length in "
                         "[draft_prefix_min, draft_prefix_max] (no grad) and the teacher's "
                         "K rollouts continue FROM that draft-generated context instead of "
                         "the bare prompt. Attacks exposure/covariate-shift directly. "
                         "Default 0 = disabled (original jsd_flat_enrich behavior).")
    ap.add_argument("--draft_prefix_min", type=int, default=0,
                    help="Lower bound for --draft_prefix_max's random prefix length.")
    ap.add_argument("--teacher_temp_spread", type=str, default=None,
                    help="jsd_flat_enrich only. Comma-separated temperatures, e.g. "
                         "'0.5,1.0,1.5', cycled across the K rollouts instead of one fixed "
                         "--teacher_temp. Cheapest state-distribution lever (no extra forward "
                         "passes). Default None = use scalar --teacher_temp for all K.")
    ap.add_argument("--perturb_context_frac", type=float, default=0.0,
                    help="jsd_flat_enrich only. Fraction (0-1) of the last "
                         "--perturb_context_window prompt tokens to replace with random vocab "
                         "ids before the teacher rolls out, synthesizing 'draft made an error "
                         "upstream' states without an actual draft forward pass. Default 0.0 "
                         "= disabled.")
    ap.add_argument("--perturb_context_window", type=int, default=16,
                    help="Number of trailing prompt tokens eligible for --perturb_context_frac.")
    ap.add_argument("--curriculum", action="store_true",
                    help="Any loss. After one full epoch (so every prompt has an initial "
                         "score), switch from cyclic prompt order to weighted sampling biased "
                         "toward prompts with high recent training loss (EMA, alpha=0.3), "
                         "EXCLUDING prompts where the teacher itself is uncertain (entropy > "
                         "--curriculum_entropy_threshold at the prompt's last token) — a hard "
                         "prompt with an uncertain teacher likely means an ambiguous label, "
                         "not a learnable draft gap, and boosting it adds noise not signal.")
    ap.add_argument("--curriculum_gamma", type=float, default=1.0,
                    help="Sampling weight = loss_ema ** gamma. Higher = more aggressively "
                         "concentrated on the hardest prompts.")
    ap.add_argument("--curriculum_entropy_threshold", type=float, default=2.0,
                    help="Teacher next-token entropy (nats) above which a prompt is excluded "
                         "from curriculum up-weighting regardless of its loss.")
    ap.add_argument("--device",       default="cuda",
                    help="CUDA device, e.g. cuda:1 (default: auto-select freest GPU)")
    ap.add_argument("--teacher", type=str, default=TEACHER_MODEL,
                    help=f"Teacher model name or local path (default: {TEACHER_MODEL}).")
    ap.add_argument("--draft", type=str, default=None,
                    help="Override the draft model path/name (default: uses DRAFT_MODEL "
                         "from config). Use to warm-start from a pre-trained checkpoint, "
                         "e.g. --draft checkpoints/jsd_s123/ckpt_best.")
    ap.add_argument("--load_in_4bit", action="store_true",
                    help="Load teacher in 4-bit NF4 via bitsandbytes. Needed for large "
                         "teachers (e.g. 32B) on GPUs where bf16 does not fit. "
                         "WARNING: flat losses (jsd, jsd_flat_enrich) call teacher.generate() "
                         "for max_new_tokens sequential steps — bitsandbytes NF4 pays a "
                         "dequantization cost on every single-token forward, so this is "
                         "~15-20x slower per step for flat losses than for tree losses "
                         "(which only need ~L sequential steps). Prefer --optim_8bit + full "
                         "bf16 teacher for flat losses if memory allows.")
    ap.add_argument("--optim_8bit", action="store_true",
                    help="Use bitsandbytes 8-bit AdamW instead of torch.optim.AdamW for the "
                         "draft's optimizer state (roughly halves optimizer memory: int8 "
                         "m/v instead of fp32). Orthogonal to --load_in_4bit (which "
                         "quantizes the TEACHER) — this quantizes the DRAFT's optimizer "
                         "state, freeing headroom that can let a flat-loss run afford the "
                         "full bf16 teacher instead of 4-bit, avoiding the generate() "
                         "slowdown described under --load_in_4bit.")
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
    ap.add_argument("--lr_min_ratio", type=float, default=None,
                    help=f"Cosine-decay floor: LR ends at this fraction of peak. "
                         f"Default: module constant LR_MIN_RATIO ({LR_MIN_RATIO}). "
                         f"Rahul's 'final lr' sweep: try {{0.1, 0.01}}.")
    ap.add_argument("--weight_decay", type=float, default=0.0,
                    help="AdamW L2 weight-decay coefficient (regularisation). "
                         "Default 0.0 = DistillSpec's no-regularisation setting. "
                         "Sweep needs a NONZERO base (e.g. 0.01) for 1x/10x/0.1x to mean anything.")
    ap.add_argument("--val_temp", type=float, default=0.2,
                    help="Sampling temperature for val block_eff decoding.  Low (0.2) "
                         "is near-deterministic → far lower run-to-run variance than "
                         "the 0.8 training temp.  Cannot be 0 (softmax/temp divide).")
    ap.add_argument("--val_every", type=int, default=VAL_EVERY,
                    help=f"Steps between val block_eff checks (default {VAL_EVERY}). "
                         "Override to a small value (e.g. 8) for a quick smoke test that "
                         "actually exercises the validation code path (different memory "
                         "profile than plain training — K-branch tree decoding) before "
                         "committing a checkpoint to a long run.")
    ap.add_argument("--train_diagnose", action="store_true",
                    help="Run lightweight JSD-vs-BE diagnostic at every val check. "
                         "Logs diag/rho, diag/sigma_jsd, diag/mean_jsd, diag/sigma_be "
                         "to the training W&B run as a time series. "
                         "Overhead: ~1 teacher greedy generate + 2 forward passes per val "
                         "prompt — roughly same wall-clock as one val pass (≈2x val time).")
    ap.add_argument("--passk_datasets", type=str, default="",
                    help="Comma-list of eval datasets to run pass@k on, logging passk_<ds>/k<K> to "
                         "W&B for K in {1,2,4,8,16,32,64}. Empty = off. Cadence is --passk_every_n_vals "
                         "(NOT the same as the block-eff val check, which always runs every --val_every). "
                         "E.g. 'math_eval,gsm8k_eval,olympiad_eval'. Training-stability "
                         "diagnostic (pass@64 flat across training => coverage not collapsing).")
    ap.add_argument("--passk_every_n_vals", type=int, default=1,
                    help="Run pass@k (either tier) only every this-many VAL CHECKS, not every one -- "
                         "block-eff itself is unaffected and still computed every --val_every regardless. "
                         "pass@k's unbatched HF generate() loop is far pricier per-check than block-eff "
                         "(measured: ~1-2h vs ~15min), so this decouples the two. Always an exact "
                         "multiple of --val_every by construction (same reasoning as "
                         "--passk_full_every_n_vals). Default 1 = every val check (old behaviour).")
    ap.add_argument("--passk_samples", type=int, default=64,
                    help="Samples/prompt for the ROUTINE pass@k tier (every val check). Must be "
                         ">= max k you want from this tier (e.g. 16 => pass@1/2/4/8/16 only). "
                         "If --passk_full_every_n_vals=0 (default) this is the ONLY tier -- "
                         "unchanged single-tier behaviour. Cost per val = passk_samples x "
                         "passk_n_prompts x (#datasets) draft generations.")
    ap.add_argument("--passk_n_prompts", type=int, default=100,
                    help="Fixed #prompts per dataset for the ROUTINE tier (first N, deterministic).")
    ap.add_argument("--passk_full_every_n_vals", type=int, default=0,
                    help="If >0, every this-many VAL CHECKS (not steps) also run a bigger 'full' "
                         "pass@k tier (passk_full_samples/passk_full_n_prompts) instead of the "
                         "routine tier -- e.g. 4 => full tier every 4th val check, whatever "
                         "--val_every is set to (so this stays correct if val_every ever changes; "
                         "it's always an exact multiple by construction, no alignment to get wrong). "
                         "0 (default) = disabled, single-tier only.")
    ap.add_argument("--passk_full_samples", type=int, default=64,
                    help="Samples/prompt for the FULL tier (only used when --passk_full_every_n_vals fires).")
    ap.add_argument("--passk_full_n_prompts", type=int, default=None,
                    help="#prompts for the FULL tier. Default: same as --passk_n_prompts (samples is "
                         "the only thing that changes tier-to-tier unless you set this explicitly).")
    ap.add_argument("--passk_temp", type=float, default=0.8,
                    help="Sampling temperature for pass@k. Default 0.8 (standard). NOTE: pass@k "
                         "needs sample DIVERSITY — at val_temp=0.2 the k samples are near-identical "
                         "so pass@64 collapses to pass@1 and the curve is uninformative. Keep ~0.8.")
    ap.add_argument("--passk_max_new_tokens", type=int, default=512,
                    help="Generation length for pass@k. MUST be long enough to reach the answer "
                         "(the block-eff val's MAX_NEW_TOKENS=128 truncates most MATH solutions "
                         "before \\boxed{} => pass@k reads ~0). 512 is a safe default; raise for "
                         "very long solutions (cost scales linearly).")
    ap.add_argument("--passk_backend", choices=["hf", "vllm"], default="hf",
                    help="'hf' (default): unchanged in-process, unbatched HF generate() loop -- "
                         "blocks training, ~1-2h per check. 'vllm': snapshot the current draft "
                         "weights to an immutable dir and spawn scripts/passk_eval_and_log.py as a "
                         "NON-BLOCKING background subprocess (from a separate venv-vllm, see "
                         "scripts/setup_vllm_env.sh) that logs results into THIS SAME W&B run when "
                         "it finishes -- training continues immediately, no ~1-2h stall.")
    ap.add_argument("--vllm_python", type=str, default=None,
                    help="Path to the venv-vllm python interpreter for --passk_backend vllm. "
                         "Default: <repo_root>/venv-vllm/bin/python (scripts/setup_vllm_env.sh's "
                         "convention).")
    ap.add_argument("--vllm_gpu_mem_frac", type=float, default=0.2,
                    help="--gpu_memory_utilization for the spawned vLLM subprocess. Kept low by "
                         "default (unlike passk_eval.py's standalone default 0.9) because this "
                         "process shares the GPU with the actively-training model/optimizer/"
                         "activations, not a free GPU.")
    ap.add_argument("--vllm_keep_snapshots", type=int, default=2,
                    help="How many recent passk_snapshots/ dirs to retain if the spawned subprocess "
                         "hasn't cleaned up its own yet (safety net against unbounded disk growth "
                         "if a child crashes before self-deleting; each is just draft-model-sized).")
    ap.add_argument("--early_stop_patience", type=int, default=EARLY_STOP_PAT,
                    help=f"Stop training if smoothed val BE has not improved for this many "
                         f"consecutive val checks (default {EARLY_STOP_PAT}; 0 = disabled). "
                         f"Each check is VAL_EVERY={VAL_EVERY} steps, so default = "
                         f"{EARLY_STOP_PAT * VAL_EVERY} steps without improvement.")
    ap.add_argument("--early_stop_min_delta", type=float, default=EARLY_STOP_DELTA,
                    help="Minimum improvement in smoothed val BE to reset the patience counter "
                         "(default 0.0 = strict; e.g. 0.005 ignores sub-0.5%% fluctuations). "
                         "A val check counts as 'no improve' only if val_be_ema < best_smoothed_be + delta.")
    ap.add_argument("--min_steps", type=int, default=0,
                    help="Floor before patience early-stop may fire — training always runs at "
                         "least this many steps (default 0 = no floor). The divergence abort "
                         "(--divergence_abort_frac) still applies below this floor.")
    ap.add_argument("--divergence_abort_frac", type=float, default=0.0,
                    help="Abort immediately (even below --min_steps) if smoothed val BE drops "
                         "more than this fraction below the best seen (e.g. 0.20 = stop if "
                         "val_be_ema < 0.8 × best_smoothed_be). Default 0.0 = disabled.")
    return ap.parse_args()


def _curriculum_sample_idx(prompt_loss_ema, prompt_entropy_ema, gamma, entropy_threshold, rng):
    """Pick a training prompt index biased toward high recent loss, excluding prompts
    where the teacher itself is uncertain (see --curriculum help text for rationale).
    Weight = loss_ema ** gamma; entropy_ema > threshold -> weight forced to 0.
    Unseen prompts (loss_ema is None) get the max weight seen so far, so they're
    prioritized for their first visit rather than starved. Falls back to uniform if
    every prompt is excluded (shouldn't happen once entropy_threshold is sane)."""
    default_w = max((w for w in prompt_loss_ema if w is not None), default=1.0)
    weights = []
    for loss_w, ent_w in zip(prompt_loss_ema, prompt_entropy_ema):
        if ent_w is not None and ent_w > entropy_threshold:
            weights.append(0.0)
        elif loss_w is None:
            weights.append(default_w)
        else:
            weights.append(max(loss_w, 1e-6) ** gamma)
    total = sum(weights)
    if total <= 0:
        return rng.randrange(len(prompt_loss_ema))
    r = rng.random() * total
    upto = 0.0
    for i, w in enumerate(weights):
        upto += w
        if upto >= r:
            return i
    return len(weights) - 1


def _serializable_args(args) -> dict:
    """Return vars(args) minus hardware-specific fields that have no analysis value."""
    skip = {"device", "cpu_threads", "resume"}
    return {k: v for k, v in vars(args).items() if k not in skip}


@torch.no_grad()
def _train_passk(draft, tokenizer, prompts, golds, n, temp, k_values, max_new_tokens):
    """In-training pass@k on a FIXED prompt set (draft model, HF generate).

    Generates n samples/prompt, grades against gold (passk_utils), returns
    {k: pass@k rate} averaged over gradeable prompts. Pure diagnostic —
    uses the already-loaded draft; NOT the spec-decoding path. Caller wraps this
    in try/except so a failure never kills training. Returns {} if nothing gradeable.
    NOTE: cost is n x len(prompts) generations — gate behind a coarse cadence.
    """
    was_training = draft.training
    draft.eval()
    per_prompt_c = []
    device = draft.device
    for prompt, gold in zip(prompts, golds):
        if gold is None:
            continue
        enc = tokenizer(prompt, return_tensors="pt").to(device)
        gen = draft.generate(
            **enc, max_new_tokens=max_new_tokens, do_sample=True, temperature=temp,
            num_return_sequences=n, pad_token_id=tokenizer.eos_token_id, use_cache=True,
        )
        plen = enc["input_ids"].shape[1]
        texts = tokenizer.batch_decode(gen[:, plen:], skip_special_tokens=True)
        c = sum(1 for t in texts if is_correct(extract_answer(t), gold))
        per_prompt_c.append(c)
    if was_training:
        draft.train()
    if not per_prompt_c:
        return {}
    out = {}
    for k in k_values:
        if k > n:
            continue
        vals = [pass_at_k(n, c, k) for c in per_prompt_c]
        out[k] = sum(vals) / len(vals)   # {k: pass@k}
    return out


def _spawn_vllm_passk(draft, tokenizer, output_dir, step, tier, datasets_csv,
                       n, n_prompts, args, wandb_run):
    """Snapshot the current draft weights to an IMMUTABLE per-step directory and
    spawn scripts/passk_eval_and_log.py as a non-blocking background process
    that logs pass@k into wandb_run once it finishes. Training continues
    immediately -- never blocks like the --passk_backend hf path.

    The snapshot (not ckpt_latest/ckpt_best) is what the child reads, because
    ckpt_latest gets overwritten by future val checks while the child may
    still be mid-read -- see scripts/passk_eval_and_log.py's docstring.
    """
    if wandb_run is None:
        print("  [passk-vllm] skipped: no active W&B run to attach the async job to "
              "(pass --no_wandb off, or use --passk_backend hf)")
        return
    if USE_LORA:
        print("  [passk-vllm] skipped: USE_LORA=True -- save_pretrained here would write "
              "adapter-only weights vLLM can't load as a full causal LM. Use --passk_backend hf.")
        return
    repo_root = os.path.dirname(os.path.abspath(__file__))
    vllm_python = args.vllm_python or os.path.join(repo_root, "venv-vllm", "bin", "python")
    if not os.path.isfile(vllm_python):
        print(f"  [passk-vllm] skipped: vllm_python not found at {vllm_python} "
              f"(run scripts/setup_vllm_env.sh, or pass --vllm_python)")
        return

    snap_root = os.path.join(output_dir, "passk_snapshots")
    snap_name = f"step{step}_{tier}"
    snap_dir = os.path.join(snap_root, snap_name)
    draft.save_pretrained(snap_dir, safe_serialization=True)
    tokenizer.save_pretrained(snap_dir)

    # Safety net: prune old snapshot dirs beyond the retention window, in case a
    # prior child crashed before self-deleting its own (--cleanup_model_dir).
    # Each is just draft-model-sized (small: 0.6B/1.7B), but unbounded over a
    # long run is still wasteful.
    if os.path.isdir(snap_root):
        others = sorted(
            (d for d in os.listdir(snap_root) if d != snap_name
             and os.path.isdir(os.path.join(snap_root, d))),
            key=lambda d: os.path.getmtime(os.path.join(snap_root, d)))
        for _old in others[:max(0, len(others) - (args.vllm_keep_snapshots - 1))]:
            shutil.rmtree(os.path.join(snap_root, _old), ignore_errors=True)

    script = os.path.join(repo_root, "scripts", "passk_eval_and_log.py")
    cmd = [vllm_python, "-u", script,
           "--model", snap_dir, "--datasets", datasets_csv,
           "--n", str(n), "--n_prompts", str(n_prompts),
           "--temp", str(args.passk_temp), "--max_tokens", str(args.passk_max_new_tokens),
           "--gpu_memory_utilization", str(args.vllm_gpu_mem_frac),
           "--wandb_run_id", wandb_run.id, "--wandb_project", WANDB_PROJECT,
           "--train_step", str(step), "--tier", tier, "--cleanup_model_dir"]
    log_path = os.path.join(snap_root, f"{snap_name}.log")
    _logf = open(log_path, "w")
    # Strip WANDB_SERVICE: Popen inherits the full parent env by default, and this
    # points at the wandb-core daemon THIS process is already using to stream the
    # same run ID. The child's wandb.init(id=..., resume="must") reusing that same
    # service collides with the parent's still-open handle on that run
    # ("ServerResponseError: run ID <id> is in use") -- the child must spin up its
    # own private wandb-core service instead of sharing the parent's.
    _env = os.environ.copy()
    _env.pop("WANDB_SERVICE", None)
    proc = subprocess.Popen(cmd, stdout=_logf, stderr=subprocess.STDOUT, cwd=repo_root, env=_env)
    _logf.close()   # child holds its own dup'd fd; safe to close the parent's handle
    print(f"  [passk-vllm] spawned pid={proc.pid} tier={tier} step={step} "
          f"snapshot={snap_dir} log={log_path}")


def main():
    import datetime
    print(f"\n{'='*72}\n[session] {datetime.datetime.now().isoformat(timespec='seconds')}  pid={os.getpid()}\n{'='*72}")
    args = parse_args()
    set_seed(args.seed)

    # Tree shape is module-global (used by the loss helpers, setup_wandb run name,
    # and validation).  Override from --K/--L so train and val always share shape.
    global K, L, VAL_K, VAL_L
    K, L = args.K, args.L
    VAL_K, VAL_L = K, L
    _VAL_BE_EASY = BE_EASY_FRAC * VAL_L   # e.g. 0.75 × 8 = 6.0
    _VAL_BE_HARD = BE_HARD_FRAC * VAL_L   # e.g. 0.375 × 8 = 3.0

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
    draft_model_path = args.draft if args.draft else DRAFT_MODEL
    tokenizer, draft, teacher = load_models(draft_model_path, args.teacher, device=args.device,
                                             load_in_4bit=args.load_in_4bit,
                                             lora_config={"r": LORA_R, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT} if USE_LORA else None)

    # Optimiser + linear warmup → constant LR
    trainable = [p for p in draft.parameters() if p.requires_grad]
    if args.optim_8bit:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise SystemExit("bitsandbytes required for --optim_8bit. "
                             "Run: pip install bitsandbytes")
        optimizer = bnb.optim.AdamW8bit(trainable, lr=args.lr, betas=(0.9, 0.999),
                                        weight_decay=args.weight_decay)
        print("[optim] using bitsandbytes AdamW8bit (int8 optimizer state)")
    else:
        optimizer = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.999),
                                      weight_decay=args.weight_decay)   # 0.0 = DistillSpec no-reg default
    # Cosine-decay floor: CLI --lr_min_ratio overrides the module constant.
    lr_min_ratio = args.lr_min_ratio if args.lr_min_ratio is not None else LR_MIN_RATIO
    if args.weight_decay != 0.0:
        print(f"[optim] weight_decay={args.weight_decay}")
    if args.lr_min_ratio is not None:
        print(f"[lr] cosine floor overridden: lr_min_ratio={lr_min_ratio} (default {LR_MIN_RATIO})")

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
                      f"LR_MIN_RATIO={lr_min_ratio}*peak (no LR jump on resume)")
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
        # Cosine decay from 1.0 → lr_min_ratio over remaining opt-steps
        progress = (step - warmup_opt_steps) / max(1, total_opt_steps - warmup_opt_steps)
        progress = min(progress, 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return lr_min_ratio + (1.0 - lr_min_ratio) * cosine
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Resume?
    start_step, train_state = (0, {})
    if args.resume:
        start_step, train_state = try_resume(draft, optimizer, scheduler, output_dir)
    best_val_block_eff = train_state.get("best_val_block_eff", 0.0)
    best_smoothed_be   = train_state.get("best_smoothed_be", 0.0)
    val_be_ema         = train_state.get("val_be_ema", None)
    no_improve_count   = train_state.get("no_improve_count", 0)

    # W&B
    wandb_run = setup_wandb(args, output_dir, resumed=(start_step > 0), wandb_project=WANDB_PROJECT)

    # Data
    print(f"[data] train = {dataset_path(args.train_dataset)}")
    print(f"[data] val   = {dataset_path(args.val_dataset)}")
    train_prompts = load_prompts_jsonl(dataset_path(args.train_dataset))
    val_prompts   = load_prompts_jsonl(dataset_path(args.val_dataset))
    random.Random(args.seed).shuffle(train_prompts)
    print(f"[data] {len(train_prompts)} train / {len(val_prompts)} val prompts")

    # pass@k eval sets: prompts + parallel gold answers, first N of each listed dataset.
    # load_prompts_jsonl() drops the gold, so re-read each file to pick up
    # `answer`/`solution` in file order (deterministic first-N => same set every check).
    # Loaded once at the larger of the two tiers' N so both tiers slice from the SAME
    # prefix (routine tier's prompts are always a prefix of the full tier's).
    passk_full_n_prompts = args.passk_full_n_prompts if args.passk_full_n_prompts is not None else args.passk_n_prompts
    passk_sets = {}   # {dataset_name: (prompts[:maxN], golds[:maxN])}
    _passk_ds = [d.strip() for d in args.passk_datasets.split(",") if d.strip()]
    _maxN = max(args.passk_n_prompts, passk_full_n_prompts)
    for _ds in _passk_ds:
        _pr, _go = [], []
        with open(dataset_path(_ds), encoding="utf-8") as _vf:
            for _line in _vf:
                _line = _line.strip()
                if not _line:
                    continue
                _obj = json.loads(_line)
                if "prompt" not in _obj:
                    continue
                _pr.append(_obj["prompt"])
                _graw = _obj.get("answer") or _obj.get("solution") or ""
                _go.append(extract_gold(_graw) if _graw else None)
        passk_sets[_ds] = (_pr[:_maxN], _go[:_maxN])
        _n_gold = sum(1 for g in _go[:args.passk_n_prompts] if g is not None)
        print(f"[passk] {_ds}: {_n_gold}/{min(args.passk_n_prompts, len(_pr))} prompts gradeable "
              f"(routine: samples={args.passk_samples} n_prompts={args.passk_n_prompts} temp={args.passk_temp})")
    if _passk_ds:
        _passk_every_steps = args.val_every * args.passk_every_n_vals   # always exact by construction
        print(f"[passk] cadence: every {args.passk_every_n_vals} val checks (= every {_passk_every_steps} steps); "
              f"block-eff val itself still runs every {args.val_every} steps regardless")
    if _passk_ds and args.passk_full_every_n_vals > 0:
        _full_every_steps = args.val_every * args.passk_full_every_n_vals   # always exact by construction
        print(f"[passk] full tier: samples={args.passk_full_samples} n_prompts={passk_full_n_prompts} "
              f"every {args.passk_full_every_n_vals} val checks (= every {_full_every_steps} steps)")

    # Dispatch
    loss_fn       = get_loss(args.loss)
    tree          = is_tree_loss(args.loss)
    offpolicy     = is_offpolicy_tree_loss(args.loss)
    enrichment    = is_enrichment_loss(args.loss)
    flat_enrich   = is_flat_enrich_loss(args.loss)
    prefix_ov     = is_prefix_overlap_loss(args.loss)
    _mode = ("prefix overlap" if prefix_ov else
             "flat enrichment" if flat_enrich else
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
    grad_clip_events = 0              # optimizer steps where grad_norm > GRAD_CLIP (clip was active)
    grad_opt_steps   = 0              # total optimizer steps (denominator for the clip fraction)
    depth_ema: float | None = None   # running mean of E[tau_V] for --aux_mode depth_weight
    depth_w = 1.0
    depth_d = 0.0                     # last raw E[tau_V] (logged so the sweep is observable)
    path_div = 0.0                    # last flat_enrich teacher-path diversity (K>1 only)
    best_per_prompt: dict[int, float] = {}   # each val prompt's best-ever block_eff (forgetting)
    # smoothed-val state (restored from train_state on resume above)
    # val_be_ema / best_smoothed_be / no_improve_count already loaded from train_state
    t0 = time.time()

    # --optim_8bit and curriculum are independent; curriculum needs one full pass
    # through train_prompts (cyclic) before every prompt has an initial loss/entropy
    # score, then switches to weighted sampling. See --curriculum help text.
    curriculum_rng = random.Random(args.seed + 1)   # separate stream from data shuffle
    prompt_loss_ema: List[float | None] = [None] * len(train_prompts)
    prompt_entropy_ema: List[float | None] = [None] * len(train_prompts)

    for step in range(start_step, args.steps):
        if args.curriculum and step >= len(train_prompts):
            prompt_idx = _curriculum_sample_idx(prompt_loss_ema, prompt_entropy_ema,
                                                args.curriculum_gamma,
                                                args.curriculum_entropy_threshold,
                                                curriculum_rng)
        else:
            prompt_idx = step % len(train_prompts)
        prompt = train_prompts[prompt_idx]
        ids    = torch.tensor(tokenizer.encode(prompt), device=draft.device, dtype=torch.long).unsqueeze(0)

        if prefix_ov:
            if args.prefix_anneal_steps > 0:
                frac = max(0.0, 1.0 - step / args.prefix_anneal_steps)
                aux_w = args.prefix_aux_weight * frac
            else:
                aux_w = args.prefix_aux_weight
            if args.prefix_root_spacing > 0:
                loss = compute_prefix_overlap_multiroot_loss(
                    draft, teacher, ids, L=args.L,
                    N=args.prefix_root_spacing, rollout_len=args.prefix_rollout_len,
                    teacher_temp=args.teacher_temp,
                    aux=args.prefix_aux, aux_weight=aux_w,
                    objective=args.prefix_objective,
                    random_offset=args.prefix_random_offset, K=args.K,
                    min_root=args.prefix_min_root,
                    teacher_topk=args.prefix_teacher_topk)
            else:
                loss = compute_prefix_overlap_loss(
                    draft, teacher, ids, M=args.prefix_M, L=args.L,
                    teacher_temp=args.teacher_temp,
                    aux=args.prefix_aux, aux_weight=aux_w,
                    objective=args.prefix_objective, K=args.K,
                    teacher_topk=args.prefix_teacher_topk)
        elif flat_enrich:
            # grad_accum=GRAD_ACCUM: backward each of the K rollouts immediately
            # (memory-safe against a large teacher) rather than holding all K
            # draft-forward graphs simultaneously for one combined backward() —
            # see compute_flat_enrich_loss docstring. This means gradients are
            # ALREADY applied by the time this call returns; the returned loss
            # is detached (logging only) and must not be combined with
            # depth_weight/aux_loss below (enforced by the assert).
            assert args.aux_mode != "depth_weight" and aux_loss_fn is None, (
                "jsd_flat_enrich's memory-safe per-rollout backward is incompatible "
                "with --aux_mode depth_weight / --aux_loss: those would silently not "
                "contribute to the gradient, since backward already happened inside "
                "compute_flat_enrich_loss.")
            _teacher_temps = ([float(t) for t in args.teacher_temp_spread.split(",")]
                              if args.teacher_temp_spread else None)
            loss, path_div = compute_flat_enrich_loss(loss_fn, draft, teacher, ids,
                                                      K=K, max_new_tokens=MAX_NEW_TOKENS,
                                                      teacher_temp=args.teacher_temp,
                                                      grad_accum=GRAD_ACCUM,
                                                      draft_prefix_min=args.draft_prefix_min,
                                                      draft_prefix_max=args.draft_prefix_max,
                                                      teacher_temps=_teacher_temps,
                                                      perturb_frac=args.perturb_context_frac,
                                                      perturb_window=args.perturb_context_window)
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
        # flat_enrich already backpropagated internally, per-rollout, inside
        # compute_flat_enrich_loss (memory-safe against a large teacher) — its
        # returned loss is detached and calling .backward() on it again would
        # either error (no grad_fn) or be a silent no-op.
        if not flat_enrich:
            (loss / GRAD_ACCUM).backward()
        if (step + 1) % GRAD_ACCUM == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
            grad_opt_steps += 1
            if grad_norm.item() > GRAD_CLIP:
                grad_clip_events += 1
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        else:
            grad_norm = torch.tensor(0.0)
        grad_clip_frac = grad_clip_events / max(1, grad_opt_steps)

        losses_log.append(loss.item())

        if args.curriculum:
            # Cheap teacher-only forward (prompt tokens only, no continuation) to get
            # this prompt's next-token entropy for the exclusion gate. Independent of
            # whichever loss branch ran above, so curriculum works with any --loss.
            with torch.no_grad():
                _ent_logits = teacher(ids, return_dict=True).logits[0, -1].float()
                _ent_probs  = F.softmax(_ent_logits, dim=-1)
                _entropy    = -(_ent_probs * _ent_probs.clamp(min=1e-9).log()).sum().item()
            _alpha = 0.3
            _cur_loss = loss.item()
            prompt_loss_ema[prompt_idx] = (_cur_loss if prompt_loss_ema[prompt_idx] is None
                else (1 - _alpha) * prompt_loss_ema[prompt_idx] + _alpha * _cur_loss)
            prompt_entropy_ema[prompt_idx] = (_entropy if prompt_entropy_ema[prompt_idx] is None
                else (1 - _alpha) * prompt_entropy_ema[prompt_idx] + _alpha * _entropy)

        # Console log + W&B train metrics
        if (step + 1) % LOG_EVERY == 0:
            avg = sum(losses_log[-LOG_EVERY:]) / LOG_EVERY
            elapsed = time.time() - t0
            print(f"step={step+1:5d}/{args.steps}  loss={avg:.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}  "
                  f"grad={grad_norm.item():.2f}  clip%={100*grad_clip_frac:.0f}  "
                  f"{f'pathdiv={path_div:.3f}  ' if (flat_enrich and K > 1) else ''}"
                  f"elapsed={elapsed/60:.1f}m")

            # Compute val metrics at val steps BEFORE logging so train + val go
            # into a single wandb.log() call — two separate calls at the same
            # step cause the second to be silently dropped in wandb ≥0.15.
            val_be = None
            val_forget = None
            _diag = {}
            _passk = {}
            if (step + 1) % args.val_every == 0:
                _clear_node_caches()
                val_be, _val_pp = compute_val_metrics(draft, teacher, tokenizer, val_prompts, mode=LOSS_TO_VERIFIER.get(args.loss, "traversal"), val_temp=args.val_temp, max_new_tokens=MAX_NEW_TOKENS, val_k=VAL_K, val_l=VAL_L, n_prompts=VAL_PROMPTS)
                # pass@k is decoupled from block-eff's cadence: block-eff above always runs
                # every --val_every (cheap, ~15min), but pass@k's unbatched HF generate() loop
                # is far pricier per-check (measured ~1-2h), so it only runs every
                # --passk_every_n_vals-th val check. Wrapped so a pass@k failure can never
                # kill training. Logged as passk_<dataset>/k<K> so W&B plots one curve per
                # dataset per K, regardless of which tier produced a given point.
                # (step+1) is already a val_every multiple here (we're inside the val-check
                # block), so checking it's ALSO a multiple of val_every*n_vals is exactly
                # "is this the n_vals-th val check" -- always exact, nothing to misalign.
                _is_full_passk = (args.passk_full_every_n_vals > 0
                                   and (step + 1) % (args.val_every * args.passk_full_every_n_vals) == 0)
                _is_passk_check = _is_full_passk or (
                    (step + 1) % (args.val_every * args.passk_every_n_vals) == 0)
                if _is_passk_check:
                    if _is_full_passk:
                        _pk_n, _pk_nprompts, _pk_tier = args.passk_full_samples, passk_full_n_prompts, "full"
                    else:
                        _pk_n, _pk_nprompts, _pk_tier = args.passk_samples, args.passk_n_prompts, "routine"
                    if args.passk_backend == "vllm":
                        # Non-blocking: the spawned subprocess logs its own passk_* entry
                        # directly into wandb_run at train/step=step+1 once it finishes --
                        # _passk stays empty here on purpose, nothing to merge into THIS
                        # step's wandb_run.log() call below.
                        _spawn_vllm_passk(draft, tokenizer, output_dir, step + 1, _pk_tier,
                                          args.passk_datasets, _pk_n, _pk_nprompts, args, wandb_run)
                    else:
                        for _ds, (_pr_all, _go_all) in passk_sets.items():
                            _pr, _go = _pr_all[:_pk_nprompts], _go_all[:_pk_nprompts]
                            try:
                                _pk_ks = [k for k in (1, 2, 4, 8, 16, 32, 64) if k <= _pk_n]
                                _pk = _train_passk(draft, tokenizer, _pr, _go,
                                                   n=_pk_n, temp=args.passk_temp,
                                                   k_values=_pk_ks, max_new_tokens=args.passk_max_new_tokens)
                                _passk.update({f"passk_{_ds}/k{k}": v for k, v in _pk.items()})
                                if _pk:
                                    print(f"  [passk:{_ds}] ({_pk_tier}, n={_pk_n}) " + "  ".join(
                                        f"k{k}={v:.3f}" for k, v in _pk.items()))
                            except Exception as _e:
                                print(f"  [passk:{_ds}] skipped (error: {_e})")
                val_be_ema = val_be if val_be_ema is None else (1 - VAL_EMA_ALPHA) * val_be_ema + VAL_EMA_ALPHA * val_be
                val_forget = _update_forgetting(best_per_prompt, _val_pp)
                _n_easy   = sum(1 for b in _val_pp.values() if b >= _VAL_BE_EASY)
                _n_medium = sum(1 for b in _val_pp.values() if _VAL_BE_HARD <= b < _VAL_BE_EASY)
                _n_hard   = sum(1 for b in _val_pp.values() if b < _VAL_BE_HARD)
                print(f"  [val] step={step+1}  block_eff={val_be:.3f}  smoothed={val_be_ema:.3f}  "
                      f"best={best_val_block_eff:.3f}  forget={val_forget:.3f}  "
                      f"easy={_n_easy} med={_n_medium} hard={_n_hard}")
                if args.train_diagnose:
                    _diag = compute_train_diag_scalars(
                        teacher, draft, tokenizer, val_prompts, _val_pp,
                        MAX_NEW_TOKENS, draft.device,
                        trained_loss=args.loss,
                    )
                    print(f"  [diag]  rho={_diag['diag/rho']:+.3f}  "
                          f"σ(JSD)={_diag['diag/sigma_jsd']:.4f}  "
                          f"mean(JSD)={_diag['diag/mean_jsd']:.4f}")
                else:
                    _diag = {}

            if wandb_run:
                # No explicit step= — x-axis is the defined train/step metric, so
                # resume appends cleanly even when ckpt_latest lags W&B history.
                wandb_run.log({
                    "train/step":      step + 1,
                    "train/loss":      avg,
                    "train/lr":        scheduler.get_last_lr()[0],
                    "train/grad_norm": grad_norm.item(),
                    "train/grad_clip_frac": grad_clip_frac,
                    **({"train/path_diversity": path_div}
                       if (flat_enrich and K > 1) else {}),
                    **({"train/depth_w": depth_w, "train/depth_d": depth_d}
                       if args.aux_mode == "depth_weight" else {}),
                    **({"val/block_eff": val_be,
                        "val/smoothed_block_eff": val_be_ema} if val_be is not None else {}),
                    **({"val/forgetting": val_forget} if val_forget is not None else {}),
                    **({"val/n_easy": _n_easy, "val/n_medium": _n_medium,
                        "val/n_hard": _n_hard} if val_be is not None else {}),
                    **_diag,
                    **_passk,
                })

        # Validation + checkpoint best (val_be already computed above if LOG step)
        if (step + 1) % args.val_every == 0:
            if (step + 1) % LOG_EVERY != 0:
                # VAL_EVERY not a multiple of LOG_EVERY — compute val now
                _clear_node_caches()
                val_be, _val_pp = compute_val_metrics(draft, teacher, tokenizer, val_prompts, mode=LOSS_TO_VERIFIER.get(args.loss, "traversal"), val_temp=args.val_temp, max_new_tokens=MAX_NEW_TOKENS, val_k=VAL_K, val_l=VAL_L, n_prompts=VAL_PROMPTS)
                val_be_ema = val_be if val_be_ema is None else (1 - VAL_EMA_ALPHA) * val_be_ema + VAL_EMA_ALPHA * val_be
                val_forget = _update_forgetting(best_per_prompt, _val_pp)
                _n_easy   = sum(1 for b in _val_pp.values() if b >= _VAL_BE_EASY)
                _n_medium = sum(1 for b in _val_pp.values() if _VAL_BE_HARD <= b < _VAL_BE_EASY)
                _n_hard   = sum(1 for b in _val_pp.values() if b < _VAL_BE_HARD)
                print(f"  [val] step={step+1}  block_eff={val_be:.3f}  smoothed={val_be_ema:.3f}  "
                      f"best={best_val_block_eff:.3f}  forget={val_forget:.3f}  "
                      f"easy={_n_easy} med={_n_medium} hard={_n_hard}")
                if wandb_run:
                    wandb_run.log({"train/step": step + 1,
                                   "val/block_eff": val_be,
                                   "val/smoothed_block_eff": val_be_ema,
                                   "val/forgetting": val_forget,
                                   "val/n_easy": _n_easy,
                                   "val/n_medium": _n_medium,
                                   "val/n_hard": _n_hard})
            if val_be > best_val_block_eff:
                best_val_block_eff = val_be
                if wandb_run:
                    wandb_run.summary["val/best_block_eff"] = best_val_block_eff
            if val_be_ema > best_smoothed_be + args.early_stop_min_delta:
                best_smoothed_be = val_be_ema
                no_improve_count = 0
                save_checkpoint(draft, optimizer, scheduler, output_dir, "ckpt_best",
                                state={"step": step + 1, "val_block_eff": val_be,
                                        "best_val_block_eff": best_val_block_eff,
                                        "best_smoothed_be": best_smoothed_be,
                                        "val_be_ema": val_be_ema,
                                        "no_improve_count": no_improve_count,
                                        "train_args": _serializable_args(args)},
                                use_lora=USE_LORA)
                print(f"  [val] saved ckpt_best (smoothed={val_be_ema:.3f}  raw={val_be:.3f})")
                if wandb_run:
                    wandb_run.summary["val/best_smoothed_be"] = best_smoothed_be
            else:
                _opt_step_now = (step + 1) // GRAD_ACCUM
                _in_warmup = _opt_step_now < warmup_opt_steps
                if _in_warmup:
                    print(f"  [val] no improve (warmup, patience frozen at "
                          f"{no_improve_count}/{args.early_stop_patience})  "
                          f"(smoothed={val_be_ema:.3f}  best_smoothed={best_smoothed_be:.3f})")
                else:
                    no_improve_count += 1
                    print(f"  [val] no improve {no_improve_count}/{args.early_stop_patience}  "
                          f"(smoothed={val_be_ema:.3f}  best_smoothed={best_smoothed_be:.3f})")
                _diverged = (args.divergence_abort_frac > 0.0 and best_smoothed_be > 0.0
                             and val_be_ema < (1.0 - args.divergence_abort_frac) * best_smoothed_be)
                _patience_hit = (not _in_warmup and args.early_stop_patience > 0
                                 and no_improve_count >= args.early_stop_patience
                                 and (step + 1) >= args.min_steps)
                if _diverged or _patience_hit:
                    _why = (f"diverged: smoothed {val_be_ema:.3f} < "
                            f"{(1.0 - args.divergence_abort_frac):.2f}×{best_smoothed_be:.3f}"
                            if _diverged else "patience exhausted")
                    print(f"  [early stop] {_why} at step {step+1}; saving and stopping.")
                    save_checkpoint(draft, optimizer, scheduler, output_dir, "ckpt_latest",
                                    state={"step": step + 1,
                                           "best_val_block_eff": best_val_block_eff,
                                           "best_smoothed_be": best_smoothed_be,
                                           "val_be_ema": val_be_ema,
                                           "no_improve_count": no_improve_count,
                                           "sched_steps": sched_steps,
                                           "warmup_opt_steps": warmup_opt_steps,
                                           "cmd": sys.argv,
                                           "train_args": _serializable_args(args)},
                                    use_lora=USE_LORA)
                    break

        # Rolling latest checkpoint
        if (step + 1) % SAVE_EVERY == 0:
            save_checkpoint(draft, optimizer, scheduler, output_dir, "ckpt_latest",
                            state={"step": step + 1,
                                   "best_val_block_eff": best_val_block_eff,
                                   "best_smoothed_be": best_smoothed_be,
                                   "val_be_ema": val_be_ema,
                                   "no_improve_count": no_improve_count,
                                   "sched_steps": sched_steps,
                                   "warmup_opt_steps": warmup_opt_steps,
                                   "cmd": sys.argv,
                                   "train_args": _serializable_args(args)},
                            use_lora=USE_LORA)

    # Final save — refresh the rolling ckpt_latest (no separate ckpt_final dir,
    # so a multi-combo sweep keeps only ckpt_best + ckpt_latest per run and does
    # not blow the disk quota).  ckpt_best holds the val-best model.
    save_checkpoint(draft, optimizer, scheduler, output_dir, "ckpt_latest",
                    state={"step": args.steps, "best_val_block_eff": best_val_block_eff,
                           "best_smoothed_be": best_smoothed_be,
                           "val_be_ema": val_be_ema,
                           "no_improve_count": no_improve_count,
                           "sched_steps": sched_steps, "warmup_opt_steps": warmup_opt_steps,
                           "cmd": sys.argv,
                           "train_args": _serializable_args(args)},
                    use_lora=USE_LORA)
    print(f"\n[done] {args.loss}: total time = {(time.time()-t0)/60:.1f} min  "
          f"best_val_block_eff = {best_val_block_eff:.3f}  best_smoothed = {best_smoothed_be:.3f}")
    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
