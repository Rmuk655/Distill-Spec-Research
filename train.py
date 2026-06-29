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

# Local modules
from losses import (ALL_LOSSES, FLAT_LOSSES, TREE_LOSSES, get_loss, is_tree_loss,
                    is_offpolicy_tree_loss, is_enrichment_loss, is_flat_enrich_loss,
                    is_prefix_overlap_loss)
from data_io import get_path as dataset_path
from config  import DRAFT_MODEL, TEACHER_MODEL, DEFAULT_K, DEFAULT_L, DEFAULT_MAX_NEW_TOKENS, DEFAULT_TEMP, DEFAULT_SEED, block_eff, BE_EASY_FRAC, BE_HARD_FRAC

# verifiers/__init__.py adds the verifiers folder to sys.path so this works.
import verifiers  # noqa: F401  — side effect: sys.path injection
from util           import set_seed, load_prompts_jsonl
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
EARLY_STOP_PAT  = 5                          # default patience: 5 × 400 = 2000 steps without smoothed improvement
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
    ap.add_argument("--train_diagnose", action="store_true",
                    help="Run lightweight JSD-vs-BE diagnostic at every val check. "
                         "Logs diag/rho, diag/sigma_jsd, diag/mean_jsd, diag/sigma_be "
                         "to the training W&B run as a time series. "
                         "Overhead: ~1 teacher greedy generate + 2 forward passes per val "
                         "prompt — roughly same wall-clock as one val pass (≈2x val time).")
    ap.add_argument("--early_stop_patience", type=int, default=EARLY_STOP_PAT,
                    help=f"Stop training if smoothed val BE has not improved for this many "
                         f"consecutive val checks (default {EARLY_STOP_PAT}; 0 = disabled). "
                         f"Each check is VAL_EVERY={VAL_EVERY} steps, so default = "
                         f"{EARLY_STOP_PAT * VAL_EVERY} steps without improvement.")
    ap.add_argument("--early_stop_min_delta", type=float, default=EARLY_STOP_DELTA,
                    help="Minimum improvement in smoothed val BE to reset the patience counter "
                         "(default 0.0 = strict; e.g. 0.005 ignores sub-0.5%% fluctuations). "
                         "A val check counts as 'no improve' only if val_be_ema < best_smoothed_be + delta.")
    return ap.parse_args()


def _serializable_args(args) -> dict:
    """Return vars(args) minus hardware-specific fields that have no analysis value."""
    skip = {"device", "cpu_threads", "resume"}
    return {k: v for k, v in vars(args).items() if k not in skip}


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
    depth_ema: float | None = None   # running mean of E[tau_V] for --aux_mode depth_weight
    depth_w = 1.0
    depth_d = 0.0                     # last raw E[tau_V] (logged so the sweep is observable)
    path_div = 0.0                    # last flat_enrich teacher-path diversity (K>1 only)
    best_per_prompt: dict[int, float] = {}   # each val prompt's best-ever block_eff (forgetting)
    # smoothed-val state (restored from train_state on resume above)
    # val_be_ema / best_smoothed_be / no_improve_count already loaded from train_state
    t0 = time.time()

    for step in range(start_step, args.steps):
        prompt = train_prompts[step % len(train_prompts)]
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
                    min_root=args.prefix_min_root)
            else:
                loss = compute_prefix_overlap_loss(
                    draft, teacher, ids, M=args.prefix_M, L=args.L,
                    teacher_temp=args.teacher_temp,
                    aux=args.prefix_aux, aux_weight=aux_w,
                    objective=args.prefix_objective, K=args.K)
        elif flat_enrich:
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
            _diag = {}
            if (step + 1) % VAL_EVERY == 0:
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
                })

        # Validation + checkpoint best (val_be already computed above if LOG step)
        if (step + 1) % VAL_EVERY == 0:
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
                if not _in_warmup and args.early_stop_patience > 0 and no_improve_count >= args.early_stop_patience:
                    print(f"  [early stop] patience exhausted at step {step+1}; saving and stopping.")
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
