"""
Online Speculative Decoding (OSD) — online draft model fine-tuning
==================================================================
Implements the algorithm from:
    Liu et al. 2023, "Online Speculative Decoding"
    https://arxiv.org/abs/2310.07177

Adapted for: Qwen2.5-0.5B (draft) → Qwen3-0.6B (target)

What this does
--------------
Standard speculative decoding uses a *fixed* draft model trained offline.
At inference time the draft proposes K tokens and the target verifier either
accepts or rejects each one.  If the draft's distribution diverges from the
target's, most proposals are rejected, giving little speedup.

OSD closes that gap *online*: every `update_every` prompts we collect the
sequences that were actually generated (including which positions were
rejected) and fine-tune the draft with a KL loss computed **only at the
rejected positions**.  Accepted positions are already well-aligned and do not
need a gradient signal.  Because we re-run both models at update time (rather
than caching stale logits), the training signal stays fresh even as the draft
weights change.

How to verify improvement
--------------------------
1. Baseline alpha  — run `--steps 0` (no updates) and record acceptance rate.
2. Online alpha    — run with the default settings; watch `online/eval_alpha`
                     in wandb.  It should rise over the first ~100 prompts and
                     then plateau.
3. Summary line    — at the end the script prints:
       baseline_alpha=X.XX  final_alpha=Y.YY  delta=+Z.ZZ

Usage
-----
    python online_serve.py \\
        --prompts  data/gsm8k_train.jsonl \\
        --draft    Qwen/Qwen2.5-0.5B \\
        --target   Qwen/Qwen3-0.6B \\
        --output   checkpoints/online-gsm8k \\
        --steps    500 \\
        --update_every 4 \\
        --K 4 \\
        --max_new_tokens 128 \\
        --temperature 0.6 \\
        --lr 3e-4 \\
        --lora_r 8 \\
        --kl_method forward_kl \\
        --log_every 10 \\
        --eval_alpha_every 50 \\
        --n_eval_prompts 20 \\
        --wandb_project distillspec \\
        --dtype bfloat16

Reference
---------
Liu, X., Cai, Z., Wu, Y., Weng, L., & Liang, P. (2023).
Online Speculative Decoding. arXiv:2310.07177.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
from collections import deque
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Shared training utilities — DB wiring and HW setup from training_scaffold.
# training_scaffold.py lives in the same directory (gbv-research/algorithms/).
# write_train_step handles DB path resolution and is silently no-op if the DB
# is unavailable. train_curves rows: split="train" (KL loss) and
# split="val" (1 - eval_alpha = rejection rate, lower = better).
# ---------------------------------------------------------------------------
from training_scaffold import (
    write_train_step as _write_train_step,
    setup_hw_opts    as _setup_hw_opts_scaffold,
)

# peft is optional at import time so we give a clear error if missing
try:
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
except ImportError as _peft_err:
    raise ImportError(
        "peft is required: pip install peft"
    ) from _peft_err

# wandb is optional — gracefully disabled if not installed
try:
    import wandb as _wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

_OSD_DIR = os.path.dirname(os.path.abspath(__file__))
_SUMMER_DIR = os.path.dirname(_OSD_DIR)
_GBV_DIR = os.path.join(_SUMMER_DIR, "gbv-research")
_WANDB_DIR = os.path.join(_GBV_DIR, "db", "wandb")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("osd")


# ---------------------------------------------------------------------------
# Hardware optimisations (same set as train_qwen3.py)
# _setup_hw_opts imported from training_scaffold above as _setup_hw_opts_scaffold.

# ---------------------------------------------------------------------------
# KL loss
# ---------------------------------------------------------------------------

def _kl_at_positions(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    wrong_mask: torch.Tensor,
    kl_method: str,
    temperature: float,
) -> torch.Tensor:
    """Compute KL divergence at rejected (wrong) positions only.

    Args:
        student_logits: float tensor of shape (B, T, V) — draft model outputs.
        teacher_logits: float tensor of shape (B, T, V) — target model outputs.
        wrong_mask:     bool tensor of shape (B, T)    — True at positions
                        where the draft token was rejected by the verifier.
        kl_method:      one of "forward_kl", "reverse_kl", "jsd".
        temperature:    softmax temperature used for both models.

    Returns:
        Scalar loss averaged over the valid (rejected) positions.

    Memory note
    -----------
    With V=151 936 (Qwen vocabulary) the naive approach of computing distributions
    over the full (B, T, V) tensor before masking materialises ~1 GB per intermediate
    tensor — easily 5–6 GB of temporaries before model weights.

    Instead we **select rejected positions first** (flat index into B*T positions),
    then compute distributions only on the much smaller (N_rejected, V) slice.
    At alpha≈0.87 only ~13 % of tokens are rejected, so memory drops ~8×.
    PyTorch fancy-index supports autograd, so gradients flow correctly.
    """
    B, T, V = student_logits.shape
    mask_flat = wrong_mask.reshape(-1)           # (B*T,)

    if mask_flat.sum() == 0:
        # No rejected positions — return a differentiable zero so optimiser
        # still has a valid grad (all zeros, no update).
        return (student_logits * 0).sum()

    # ── Select only rejected positions ──────────────────────────────────────
    # student slice keeps the computation graph; teacher slice is detached
    # (target_model already ran under torch.no_grad(), but .detach() is cheap
    #  insurance against accidental grad leakage on teacher weights).
    s_sel = student_logits.reshape(-1, V)[mask_flat].float() / temperature   # (N_rej, V) grad
    t_sel = teacher_logits.reshape(-1, V)[mask_flat].detach().float() / temperature  # (N_rej, V) no grad

    # Free the full (B, T, V) tensors before computing softmax distributions.
    # student_logits retains its grad tape via s_sel; tgt can be freed completely.
    del student_logits, teacher_logits
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Compute distributions on the small (N_rejected, V) slices ────────────
    log_s = F.log_softmax(s_sel, dim=-1)         # (N_rej, V) — needs grad

    if kl_method == "forward_kl":
        # KL(target ∥ student) — mode-covering, standard distillation
        # Only need p_t and log_s → skip materialising p_s entirely
        with torch.no_grad():
            log_t = F.log_softmax(t_sel, dim=-1)
            p_t   = log_t.exp()
        kl = (p_t * (log_t - log_s)).sum(dim=-1)              # (N_rej,)

    elif kl_method == "reverse_kl":
        # KL(student ∥ target) — mode-seeking
        with torch.no_grad():
            log_t = F.log_softmax(t_sel, dim=-1)
        p_s = log_s.exp()
        kl = (p_s * (log_s - log_t)).sum(dim=-1)              # (N_rej,)

    elif kl_method == "jsd":
        # Jensen-Shannon divergence — symmetric, bounded in [0, ln2]
        with torch.no_grad():
            log_t = F.log_softmax(t_sel, dim=-1)
            p_t   = log_t.exp()
        p_s   = log_s.exp()
        m     = 0.5 * (p_s + p_t)
        log_m = (m + 1e-10).log()
        kl = (
            0.5 * (p_s * (log_s - log_m)).sum(dim=-1)
            + 0.5 * (p_t * (log_t - log_m)).sum(dim=-1)
        )                                                       # (N_rej,)
    else:
        raise ValueError(f"Unknown kl_method: {kl_method!r}")

    return kl.mean()


# ---------------------------------------------------------------------------
# EBE loss for online speculative decoding
# ---------------------------------------------------------------------------

def _ebe_online(
    student_logits: torch.Tensor,   # (B, T, V) — grad required
    teacher_logits: torch.Tensor,   # (B, T, V) — detached (no grad)
    token_ids: torch.Tensor,        # (B, T)    — actual generated tokens
    wrong_mask: torch.Tensor,       # (B, T)    — True at rejected positions (KL reg only)
    block_len: int = 8,
    kl_weight: float = 0.1,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Block-level Expected Block Efficiency loss for online speculative decoding.

    Unlike _kl_at_positions, this function does NOT pre-select rejected positions:
    EBE is computed over full sequences and the gradient is naturally zero at
    accepted positions — ∂α/∂θ = 0 when α = min(1, q/p) = 1 (i.e. p ≤ q, draft
    already under-estimates or matches the target).  No wrong_mask surgery needed.

    The KL regulariser is applied only at rejected positions (wrong_mask=True),
    consistent with OSD's principle of not updating the draft at positions it
    already gets right.

    Why this works where offline EBE doesn't
    ----------------------------------------
    Offline EBE trains on teacher-sampled tokens: q(t) is always high (the teacher
    chose t), so min(1, q/p) ≈ 1 for almost every token → EBE gradient ≈ 0.

    Here tokens come from an actual speculative decoding run.  At rejected positions
    the draft proposed token t_draft with high p_draft but the target gave it low
    q_target → α = q/p < 1 → real EBE gradient.  The block-level cumprod then
    amplifies that gradient for positions deep in an accepted run (high-value fixes).

    Args:
        student_logits  (B, T, V)  Draft logits; gradient required.
        teacher_logits  (B, T, V)  Target logits; must be detached before call.
        token_ids       (B, T)     Token ids of the generated sequence.
        wrong_mask      (B, T)     True where the draft was rejected (for KL reg).
        block_len               Speculative block length — match inference --K.
        kl_weight               Weight of KL regulariser (0 = pure EBE, no reg).
        temperature             Logit temperature; applied to both models.
    """
    B, T, V = student_logits.shape

    # Compute log-softmax once for both EBE (gather) and KL (full-distribution).
    # log_s keeps grad; log_t is a no-grad view of the already-detached teacher.
    log_s = F.log_softmax(student_logits.float() / temperature, dim=-1)  # (B, T, V)
    with torch.no_grad():
        log_t = F.log_softmax(teacher_logits.float() / temperature, dim=-1)  # (B, T, V)

    # ── EBE term ──────────────────────────────────────────────────────────────
    # Gather per-token log-probs for the actual tokens → (B, T) tiny scalars.
    # Gradient flows through log_p (draft); log_q is from the frozen teacher.
    ids = token_ids.unsqueeze(-1)                                           # (B, T, 1)
    log_p = log_s.gather(-1, ids).squeeze(-1)                              # (B, T)
    log_q = log_t.gather(-1, ids).squeeze(-1)                              # (B, T)

    # α = min(1, q/p) = exp(min(0, log_q − log_p))
    # Clamp ≥ 1e-6: prevents NaN in cumprod backward when a token is completely
    # off-distribution (q ≈ 0 → log_q → −∞ → alpha → 0 → cumprod backward divides by 0)
    alpha = torch.exp(torch.clamp(log_q - log_p, max=0.0)).clamp(min=1e-6)  # (B, T)

    # Sum block-level (1 + cumprod) over all blocks and sequences.
    n_blocks = max(1, T // block_len)
    ebe = alpha.new_zeros(())
    for b in range(B):
        for blk in range(n_blocks):
            start = blk * block_len
            chunk = alpha[b, start : start + block_len]            # (≤ block_len,)
            ebe = ebe - (1.0 + torch.cumprod(chunk, dim=0).sum())
    ebe = ebe / (B * n_blocks)   # normalise to be scale-independent of batch/seq len

    if kl_weight == 0.0:
        return ebe

    # ── KL regulariser — rejected positions only ──────────────────────────────
    # Rationale: EBE gradient is 0 for under-estimated tokens (α = 1); KL covers
    # those positions.  We restrict it to rejected positions (wrong_mask) rather
    # than all tokens to stay consistent with OSD's philosophy of not penalising
    # the draft where it already aligns with the target.
    mask_flat = wrong_mask.reshape(-1)                                      # (B*T,)
    if mask_flat.sum() == 0:
        return ebe

    s_rej = log_s.reshape(-1, V)[mask_flat]                                # (N_rej, V)
    with torch.no_grad():
        t_rej = log_t.reshape(-1, V)[mask_flat]                            # (N_rej, V)
        p_t   = t_rej.exp()                                                # (N_rej, V)

    # Forward KL(target ∥ draft) at rejected positions
    kl = (p_t * (t_rej - s_rej)).sum(dim=-1).mean()                       # scalar

    return ebe + kl_weight * kl


# ---------------------------------------------------------------------------
# Speculative decoding (single sequence, K tokens per step)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _draft_propose(
    draft_model,
    input_ids: torch.Tensor,
    K: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Autoregressively generate K draft tokens.

    Args:
        draft_model: the (LoRA-wrapped) draft model.
        input_ids:   (1, L) prompt token ids.
        K:           number of tokens to propose.

    Returns:
        draft_tokens:  (K,) proposed token ids (int64).
        draft_logprob: (K,) log-prob of each proposed token under the draft.
                       Computed via a single extra forward pass over the
                       extended sequence for efficiency.
    """
    device = input_ids.device
    extended = input_ids.clone()  # (1, L)

    for _ in range(K):
        out = draft_model(extended)
        next_logits = out.logits[:, -1, :]  # (1, V)
        next_tok = next_logits.argmax(dim=-1, keepdim=True)  # greedy
        extended = torch.cat([extended, next_tok], dim=1)

    # extended is now (1, L+K); draft tokens are the last K positions
    draft_tokens = extended[0, -K:]  # (K,)

    # Re-run the full candidate sequence in ONE forward pass to get
    # per-position log-probs for the K draft positions.
    # Position i in extended predicts token i+1, so draft position j
    # (0-indexed from L) is predicted at logit index L-1+j.
    with torch.no_grad():
        logits_full = draft_model(extended).logits  # (1, L+K, V)

    L = input_ids.shape[1]
    # logits at positions L-1 .. L+K-2  predict tokens at positions L .. L+K-1
    draft_logits = logits_full[0, L - 1 : L + K - 1, :]  # (K, V)
    draft_logprob = F.log_softmax(draft_logits.float(), dim=-1)
    # gather the log-prob of each actually-chosen draft token
    chosen_logprob = draft_logprob[
        torch.arange(K, device=device), draft_tokens
    ]  # (K,)

    return draft_tokens, chosen_logprob


@torch.no_grad()
def speculative_step(
    draft_model,
    target_model,
    input_ids: torch.Tensor,
    K: int,
    max_new_tokens: int,
    temperature: float,
    eos_token_id: int,
) -> Tuple[torch.Tensor, List[int], float, int]:
    """Run linear speculative decoding until EOS or max_new_tokens.

    Uses the standard rejection-sampling algorithm (Leviathan et al. 2023).

    Args:
        draft_model:    draft model (LoRA-wrapped, frozen for serving).
        target_model:   target model (fully frozen).
        input_ids:      (1, L) prompt.
        K:              draft tokens per speculative step.
        max_new_tokens: hard cap on tokens generated.
        temperature:    sampling temperature; 1.0 = no scaling.
        eos_token_id:   stop token.

    Returns:
        full_ids:        (1, L + n_generated) complete sequence including prompt.
        wrong_positions: list of absolute token positions (0-indexed within the
                         *full_ids* tensor) where a draft token was rejected.
        alpha:           per-token acceptance rate for this call (in [0, 1]).
        target_calls:    number of target model forward passes (= speculative
                         loop iterations).  block_efficiency = n_generated /
                         target_calls — the primary metric for this project.
    """
    device = input_ids.device
    seq = input_ids.clone()  # (1, current_len)
    wrong_positions: List[int] = []
    n_accepted_total = 0
    n_proposed_total = 0
    target_calls = 0

    while seq.shape[1] - input_ids.shape[1] < max_new_tokens:
        prompt_len = seq.shape[1]
        remaining = max_new_tokens - (prompt_len - input_ids.shape[1])
        k = min(K, remaining)
        if k == 0:
            break

        # --- Draft proposes k tokens ---
        draft_tokens, draft_lp = _draft_propose(draft_model, seq, k)
        # draft_tokens: (k,), draft_lp: (k,) log-probs

        # --- Target verifies all k tokens in ONE forward pass ---
        candidate = torch.cat(
            [seq, draft_tokens.unsqueeze(0)], dim=1
        )  # (1, prompt_len + k)

        target_logits = target_model(candidate).logits  # (1, prompt_len+k, V)
        target_calls += 1
        # Target's prediction for position j (0-indexed from seq start) is at
        # logit index j-1.  Draft token at absolute position `prompt_len + j`
        # (j in 0..k-1) is predicted by target logit at `prompt_len - 1 + j`.
        tgt_logits_k = target_logits[
            0, prompt_len - 1 : prompt_len + k - 1, :
        ]  # (k, V)
        tgt_lp_k = F.log_softmax(tgt_logits_k.float() / temperature, dim=-1)
        tgt_p_k = tgt_lp_k.exp()

        draft_lp_temp = F.log_softmax(
            # re-scale draft logits by temperature — draft_lp was already
            # computed at temperature=1; recompute from logits is cleaner
            # but we approximate here by dividing stored lp by temperature
            # (equivalent when we also rescale the acceptance ratio)
            # For correctness we recompute fresh below.
            torch.zeros(1, device=device),  # placeholder
            dim=-1,
        )
        # --- Recompute draft probs at temperature for acceptance ratio ---
        with torch.no_grad():
            full_logits = draft_model(candidate).logits  # (1, pl+k, V)
        draft_logits_k = full_logits[
            0, prompt_len - 1 : prompt_len + k - 1, :
        ]  # (k, V)
        draft_lp_temp_k = F.log_softmax(
            draft_logits_k.float() / temperature, dim=-1
        )  # (k, V)
        draft_p_k = draft_lp_temp_k.exp()

        # --- Sequential acceptance loop ---
        accepted_up_to = -1  # last accepted index (exclusive: -1 means none)
        for i in range(k):
            tok = draft_tokens[i].item()
            p_tgt = tgt_p_k[i, tok].item()
            p_dft = draft_p_k[i, tok].item()
            accept_prob = min(1.0, p_tgt / (p_dft + 1e-9))
            n_proposed_total += 1
            if random.random() < accept_prob:
                accepted_up_to = i
                n_accepted_total += 1
            else:
                # Rejection: sample from adjusted distribution and stop
                abs_pos = prompt_len + i  # position in candidate sequence
                wrong_positions.append(abs_pos)
                # adjusted dist: (p_target - p_draft)_+  (clipped & renormed)
                adjusted = (tgt_p_k[i] - draft_p_k[i]).clamp(min=0.0)
                adjusted_sum = adjusted.sum()
                if adjusted_sum > 1e-9:
                    adjusted = adjusted / adjusted_sum
                    new_tok = torch.multinomial(adjusted, num_samples=1)
                else:
                    # fallback: sample from target
                    new_tok = torch.multinomial(tgt_p_k[i], num_samples=1)
                seq = torch.cat(
                    [
                        seq,
                        draft_tokens[:i].unsqueeze(0),  # accepted so far
                        new_tok.unsqueeze(0),
                    ],
                    dim=1,
                )
                break
        else:
            # All k draft tokens accepted — append them and sample bonus token
            seq = torch.cat([seq, draft_tokens.unsqueeze(0)], dim=1)
            # Bonus token from target distribution at the last position
            bonus_logits = target_logits[0, prompt_len + k - 1, :]
            bonus_p = F.softmax(bonus_logits.float() / temperature, dim=-1)
            bonus_tok = torch.multinomial(bonus_p, num_samples=1)
            seq = torch.cat([seq, bonus_tok.unsqueeze(0)], dim=1)

        # Check for EOS in newly appended tokens
        last_generated = seq[0, prompt_len:].tolist()
        if eos_token_id in last_generated:
            # Truncate at first EOS (inclusive)
            eos_idx = last_generated.index(eos_token_id)
            seq = seq[:, : prompt_len + eos_idx + 1]
            break

    alpha = n_accepted_total / max(n_proposed_total, 1)
    return seq, wrong_positions, alpha, target_calls


# ---------------------------------------------------------------------------
# Buffer for online updates
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Simple list-based buffer that stores (input_ids, wrong_positions) pairs."""

    def __init__(self):
        self._data: List[Tuple[torch.Tensor, List[int]]] = []

    def add(self, input_ids: torch.Tensor, wrong_positions: List[int]):
        # Store on CPU to save GPU memory between update steps
        self._data.append((input_ids.cpu(), wrong_positions))

    def __len__(self):
        return len(self._data)

    def clear(self):
        self._data.clear()

    def as_batch(
        self, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pad sequences and build wrong_mask.

        Returns:
            input_ids:  (B, T_max) padded token ids.
            wrong_mask: (B, T_max) bool, True at rejected positions.
        """
        max_len = max(ids.shape[1] for ids, _ in self._data)
        B = len(self._data)
        padded = torch.zeros(B, max_len, dtype=torch.long)
        mask = torch.zeros(B, max_len, dtype=torch.bool)

        for b, (ids, wrong_pos) in enumerate(self._data):
            L = ids.shape[1]
            padded[b, :L] = ids[0]
            for p in wrong_pos:
                if p < max_len:
                    mask[b, p] = True

        return padded.to(device), mask.to(device)


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_metrics(
    draft_model,
    target_model,
    prompts: List[str],
    tokenizer,
    K: int,
    max_new_tokens: int,
    temperature: float,
    device: torch.device,
    n_eval: int = 20,
) -> Tuple[float, float]:
    """Measure acceptance rate AND block efficiency on a random subset of prompts.

    Returns:
        alpha: mean per-token acceptance rate across eval prompts.
        be:    block efficiency = total_generated_tokens / total_target_calls.
               Computed as a ratio of totals (not mean of per-prompt values) so
               short sequences don't dilute the estimate.  This is the primary
               throughput metric: for K=4 and untrained draft, ~3.1 tokens/call;
               a well-trained draft should push this toward 4+.
    """
    sample = random.sample(prompts, min(n_eval, len(prompts)))
    alphas: List[float] = []
    total_generated = 0
    total_target_calls = 0
    draft_model.eval()
    for text in sample:
        ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
        seq, _, alpha, tc = speculative_step(
            draft_model,
            target_model,
            ids,
            K=K,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            eos_token_id=tokenizer.eos_token_id,
        )
        alphas.append(alpha)
        total_generated += seq.shape[1] - ids.shape[1]
        total_target_calls += tc
    mean_alpha = float(sum(alphas) / len(alphas)) if alphas else 0.0
    be = total_generated / max(total_target_calls, 1)
    return mean_alpha, be


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_prompts(path: str) -> List[str]:
    """Load prompts from a .jsonl file.

    Looks for a 'question' key first (GSM8K style), then 'prompt', then
    falls back to the full JSON string representation.
    """
    prompts = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "question" in obj:
                prompts.append(obj["question"])
            elif "prompt" in obj:
                prompts.append(obj["prompt"])
            else:
                prompts.append(json.dumps(obj))
    return prompts


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_models(args: argparse.Namespace, device: torch.device):
    """Load target (frozen) and draft (LoRA-wrapped, trainable) models.

    When args.load_in_4bit is True the FROZEN TARGET is loaded in 4-bit NF4
    (bitsandbytes QLoRA).  Use on Colab free T4 (15 GB) with Qwen3-8B:
      bfloat16 ~16 GB → OOM;  4-bit NF4 ~5 GB → comfortable fit.
    The draft is always loaded in full dtype (bfloat16) — only the frozen
    teacher is quantised so training quality is unaffected.
    """
    dtype = getattr(torch, args.dtype)

    log.info("Loading target model: %s", args.target)
    if getattr(args, "load_in_4bit", False):
        try:
            from transformers import BitsAndBytesConfig as _BnB
        except ImportError:
            raise SystemExit("bitsandbytes required for --load_in_4bit. "
                             "Run: pip install bitsandbytes")
        _bnb = _BnB(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True)
        target_model = AutoModelForCausalLM.from_pretrained(
            args.target, quantization_config=_bnb, device_map="auto",
        )
        log.info("Target loaded in 4-bit NF4 (QLoRA mode)")
    else:
        target_model = AutoModelForCausalLM.from_pretrained(
            args.target,
            torch_dtype=dtype,
            device_map=device,
        )
    target_model.eval()
    for p in target_model.parameters():
        p.requires_grad_(False)

    log.info("Loading draft base model: %s", args.draft)
    draft_base = AutoModelForCausalLM.from_pretrained(
        args.draft,
        torch_dtype=dtype,
    ).to(device)

    # Auto-resume: if a ckpt_latest/ exists in the output dir use it as the
    # starting adapter (written by the periodic save_every checkpoint logic).
    # Explicit --adapter still takes priority so manual overrides work.
    _ckpt_latest_dir = Path(args.output) / "ckpt_latest"
    _auto_adapter = (str(_ckpt_latest_dir)
                     if (_ckpt_latest_dir / "adapter_model.safetensors").exists()
                     else None)
    _adapter_src = (args.adapter if (args.adapter and args.adapter.lower() != "none")
                    else _auto_adapter)

    if _adapter_src:
        log.info("Loading existing LoRA adapter from: %s", _adapter_src)
        draft_model = PeftModel.from_pretrained(
            draft_base, _adapter_src, is_trainable=True
        )
    else:
        log.info("Wrapping draft with fresh LoRA (r=%d)", args.lora_r)
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_r * 2,
            target_modules=["q_proj", "v_proj"],
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )
        draft_model = get_peft_model(draft_base, lora_cfg)

    draft_model.to(device)
    n_trainable = sum(p.numel() for p in draft_model.parameters() if p.requires_grad)
    log.info("Trainable draft params: %s", f"{n_trainable:,}")

    # torch.compile — Linux/Colab/server only (Triton not available on Windows)
    if getattr(args, "compile", False):
        import platform
        if platform.system() == "Windows":
            log.warning("--compile skipped: torch.compile has limited support on Windows")
        elif not hasattr(torch, "compile"):
            log.warning("--compile skipped: PyTorch < 2.0")
        else:
            log.info("Compiling draft model (one-time ~60 s)...")
            draft_model = torch.compile(draft_model, mode="reduce-overhead")
            log.info("  Done.")

    return target_model, draft_model


# ---------------------------------------------------------------------------
# Single update step
# ---------------------------------------------------------------------------

def update_step(
    draft_model,
    target_model,
    buffer: ReplayBuffer,
    optimizer: torch.optim.Optimizer,
    kl_method: str,
    temperature: float,
    device: torch.device,
    ebe_kl_weight: float = 0.1,
    ebe_block_len: int = 8,
) -> float:
    """Run one gradient update using all buffered sequences.

    We re-run BOTH models on the buffered sequences to get fresh logits —
    we never cache logits across update steps because draft weights change.

    kl_method controls the training objective:
        "forward_kl"  — KL(target ∥ draft) at rejected positions (OSD original)
        "reverse_kl"  — KL(draft ∥ target) at rejected positions
        "jsd"         — Jensen-Shannon at rejected positions
        "ebe"         — Block-level EBE over full sequences + KL at rejected positions

    Returns:
        Scalar loss value (float).
    """
    input_ids, wrong_mask = buffer.as_batch(device)
    # wrong_mask: (B, T) — True at rejected positions
    # We compute loss at positions 1..T (predicting next token) so we shift:
    #   model input: input_ids[:, :-1]
    #   labels / mask: input_ids[:, 1:] / wrong_mask[:, 1:]
    # (wrong_positions were stored as absolute positions in the *full* sequence
    #  which includes the prompt; the shift is consistent because position p
    #  in the full sequence corresponds to logit at index p-1 for next-token.)
    x            = input_ids[:, :-1]    # (B, T-1) — model input
    token_ids    = input_ids[:, 1:]     # (B, T-1) — actual generated tokens (for EBE)
    shifted_mask = wrong_mask[:, 1:]    # (B, T-1) — rejection mask aligned with logits

    if shifted_mask.sum() == 0:
        log.debug("No rejected positions in buffer — skipping update")
        return 0.0

    # Target forward (no grad) — free activations immediately after to reclaim VRAM
    with torch.no_grad():
        tgt_logits = target_model(x).logits  # (B, T-1, V)
    # Empty cache after target forward: KV-cache activations and intermediate
    # activations from the target model are no longer needed.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Draft forward (with grad)
    draft_model.train()
    stu_logits = draft_model(x).logits  # (B, T-1, V)

    if kl_method == "ebe":
        # _ebe_online computes block-level EBE over full sequences.
        # Gradient is naturally zero at accepted positions (∂α/∂θ = 0 when α=1);
        # KL regulariser is applied only at rejected positions (wrong_mask).
        loss = _ebe_online(
            stu_logits, tgt_logits, token_ids, shifted_mask,
            block_len=ebe_block_len,
            kl_weight=ebe_kl_weight,
            temperature=temperature,
        )
    else:
        # _kl_at_positions selects rejected positions FIRST (N_rej ≪ B*T) and
        # frees the full (B, T-1, V) logit tensors inside before softmax.
        loss = _kl_at_positions(stu_logits, tgt_logits, shifted_mask, kl_method, temperature)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(draft_model.parameters(), max_norm=1.0)
    optimizer.step()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # reclaim grad buffers before next speculative step

    return loss.item()


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Online Speculative Decoding — online draft fine-tuning"
    )
    parser.add_argument("--prompts", default="data/gsm8k_train.jsonl")
    parser.add_argument("--draft", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--target", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--adapter", default=None,
                        help="Path to existing LoRA checkpoint, or None")
    parser.add_argument("--output", default="checkpoints/online-gsm8k")
    parser.add_argument("--steps", type=int, default=500,
                        help="Number of prompts to process")
    parser.add_argument("--update_every", type=int, default=4,
                        help="Update draft every N prompts")
    parser.add_argument("--K", type=int, default=4,
                        help="Draft tokens per speculative step")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument(
        "--kl_method",
        default="forward_kl",
        choices=["forward_kl", "reverse_kl", "jsd", "ebe"],
        help=(
            "Training objective. "
            "'forward_kl'/'reverse_kl'/'jsd': KL variant at rejected positions (OSD original). "
            "'ebe': block-level EBE over full sequences + KL reg at rejected positions."
        ),
    )
    parser.add_argument(
        "--ebe_kl_weight", type=float, default=0.1,
        help="Weight of the KL regulariser when --kl_method ebe is used. "
             "Set to 0 for pure EBE (no regulariser). Default: 0.1.",
    )
    parser.add_argument(
        "--ebe_block_len", type=int, default=8,
        help="Speculative block length for EBE cumprod. "
             "Should match --K (draft tokens per step). Default: 8.",
    )
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--eval_alpha_every", type=int, default=50)
    parser.add_argument("--n_eval_prompts", type=int, default=20)
    parser.add_argument("--save_every", type=int, default=50,
                        help="Save a LoRA checkpoint to <output>/ckpt_latest/ every N steps. "
                             "On restart the pipeline loads from there and resumes without "
                             "re-running already-processed prompts or re-measuring baseline alpha.")
    parser.add_argument("--wandb_project", default="distillspec")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None,
                        help="cuda / cpu (auto-detected if not set)")
    parser.add_argument("--compile", action="store_true",
                        help="torch.compile() the draft model (10-30%% speedup on Linux/Colab). "
                             "Skipped automatically on Windows.")
    parser.add_argument("--load_in_4bit", action="store_true",
                        help="Load the target model in 4-bit NF4 (bitsandbytes). "
                             "Required for --config colab with Qwen3-8B on a free T4 (15 GB). "
                             "Requires: pip install bitsandbytes.")
    args = parser.parse_args()

    # --- Reproducibility ---
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # --- Device ---
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Using device: %s", device)
    _setup_hw_opts_scaffold(device)

    # --- wandb ---
    use_wandb = WANDB_AVAILABLE and args.wandb_project
    if use_wandb:
        os.makedirs(_WANDB_DIR, exist_ok=True)
        _wandb.init(
            project=args.wandb_project,
            config=vars(args),
            name=f"osd-K{args.K}-r{args.lora_r}-{args.kl_method}",
            dir=_WANDB_DIR,
        )
    else:
        if not WANDB_AVAILABLE:
            log.warning("wandb not installed — logging to stdout only")

    # --- Load data ---
    log.info("Loading prompts from: %s", args.prompts)
    all_prompts = load_prompts(args.prompts)
    log.info("Total prompts available: %d", len(all_prompts))
    if len(all_prompts) == 0:
        sys.exit("ERROR: no prompts loaded — check --prompts path")

    # Separate a held-out eval split (first n_eval_prompts)
    eval_prompts = all_prompts[: args.n_eval_prompts]
    train_prompts = all_prompts[args.n_eval_prompts :]

    # --- Load tokenizer (use target tokenizer — they must be compatible) ---
    log.info("Loading tokenizer from: %s", args.target)
    tokenizer = AutoTokenizer.from_pretrained(args.target)
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # --- Load models ---
    target_model, draft_model = load_models(args, device)

    # --- Optimizer ---
    optimizer = AdamW(
        [p for p in draft_model.parameters() if p.requires_grad],
        lr=args.lr,
    )

    # --- Resume state: load step offset + cached baseline_alpha from ckpt_latest ---
    # Written by the periodic checkpoint saves below; avoids re-measuring baseline
    # alpha (~5 min) and re-processing already-seen prompts on restart.
    # _ckpt_latest_dir is also used inside load_models() for auto-adapter detection;
    # we define it here so main() owns it in its own scope (avoids NameError).
    _ckpt_latest_dir = Path(args.output) / "ckpt_latest"
    _resume_file = _ckpt_latest_dir / "resume_state.json"
    start_step = 0
    if _resume_file.exists():
        try:
            with open(_resume_file, encoding="utf-8") as _rf:
                _rs = json.load(_rf)
            start_step = int(_rs.get("step", 0))
            log.info("[RESUME] Resuming from step %d / %d", start_step, args.steps)
        except Exception as _re:
            log.warning("[RESUME] Could not read resume_state.json (%s) — starting fresh", _re)
            start_step = 0

    # --- Baseline alpha + BE (before any updates) ---
    # Skip re-measurement when resuming — use the values cached in resume_state.json.
    if start_step > 0 and _resume_file.exists():
        try:
            with open(_resume_file, encoding="utf-8") as _rf:
                _rs2 = json.load(_rf)
                baseline_alpha = float(_rs2.get("baseline_alpha", -1))
                baseline_be    = float(_rs2.get("baseline_be",    0.0))
            if baseline_alpha < 0:
                raise ValueError("baseline_alpha missing")
            log.info("[RESUME] Restored baseline_alpha=%.4f  baseline_be=%.3f",
                     baseline_alpha, baseline_be)
        except Exception:
            baseline_alpha = None  # will measure below
            baseline_be    = 0.0
    else:
        baseline_alpha = None
        baseline_be    = 0.0

    if baseline_alpha is None:
        log.info("Measuring baseline acceptance rate and block efficiency...")
        draft_model.eval()
        baseline_alpha, baseline_be = evaluate_metrics(
            draft_model,
            target_model,
            eval_prompts,
            tokenizer,
            K=args.K,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            device=device,
            n_eval=args.n_eval_prompts,
        )
        log.info("Baseline  alpha=%.4f  be=%.3f", baseline_alpha, baseline_be)
    if use_wandb:
        _wandb.summary["baseline_alpha"] = baseline_alpha
        _wandb.summary["baseline_be"]    = baseline_be

    # --- Main online loop ---
    buffer = ReplayBuffer()
    alpha_window: deque = deque(maxlen=50)  # rolling window for online/alpha
    update_count = 0
    last_loss = 0.0

    # Build the full prompt sequence then slice off already-processed steps so
    # we resume exactly where we left off without reprocessing any prompts.
    _all_steps_prompts = (
        train_prompts * math.ceil(args.steps / max(len(train_prompts), 1))
    )[:args.steps]

    for _loop_idx, prompt in enumerate(_all_steps_prompts[start_step:]):
        step = start_step + _loop_idx
        # 1. Tokenize prompt
        input_ids = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).input_ids.to(device)

        # 2. Speculative decoding (draft eval mode, no grad needed)
        draft_model.eval()
        with torch.no_grad():
            full_ids, wrong_positions, alpha, _ = speculative_step(
                draft_model,
                target_model,
                input_ids,
                K=args.K,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                eos_token_id=tokenizer.eos_token_id,
            )

        # 3. Record into buffer
        buffer.add(full_ids, wrong_positions)
        alpha_window.append(alpha)

        # 4. Periodic update
        if (step + 1) % args.update_every == 0 and len(buffer) > 0:
            last_loss = update_step(
                draft_model,
                target_model,
                buffer,
                optimizer,
                kl_method=args.kl_method,
                temperature=args.temperature,
                device=device,
                ebe_kl_weight=args.ebe_kl_weight,
                ebe_block_len=args.ebe_block_len,
            )
            buffer.clear()
            update_count += 1

        # 5. Logging
        if (step + 1) % args.log_every == 0:
            rolling_alpha = sum(alpha_window) / len(alpha_window)
            log.info(
                "step=%d  rolling_alpha=%.4f  loss=%.6f  updates=%d",
                step + 1,
                rolling_alpha,
                last_loss,
                update_count,
            )
            if use_wandb:
                _wandb.log(
                    {
                        "online/alpha": rolling_alpha,
                        "online/loss": last_loss,
                        "online/step": step + 1,
                    },
                    step=step + 1,
                )
            # Periodic checkpoint — crash-safe resume without re-running prompts.
            # Saves to <output>/ckpt_latest/ every save_every steps.
            if (step + 1) % args.save_every == 0:
                try:
                    _ckpt_latest_dir.mkdir(parents=True, exist_ok=True)
                    draft_model.save_pretrained(str(_ckpt_latest_dir))
                    tokenizer.save_pretrained(str(_ckpt_latest_dir))
                    with open(_ckpt_latest_dir / "resume_state.json", "w", encoding="utf-8") as _rsf:
                        json.dump({"step": step + 1,
                                   "baseline_alpha": baseline_alpha,
                                   "baseline_be": baseline_be}, _rsf)
                    log.info("[ckpt] Saved checkpoint at step %d → %s", step + 1, _ckpt_latest_dir)
                except Exception as _ce:
                    log.warning("[ckpt] Checkpoint save failed at step %d: %s", step + 1, _ce)

            # Write KL loss to results.db so the dashboard Training Loss Curves
            # panel shows the online run alongside kl/ebe/rev_kl/jsd/l1 curves.
            _write_train_step(
                label=os.path.basename(args.output),
                loss_name=args.kl_method,
                step=step + 1,
                loss=last_loss,
                learning_rate=args.lr,
                lora_rank=args.lora_r,
                split="train",
            )

        # 6. Periodic held-out eval
        if (step + 1) % args.eval_alpha_every == 0:
            draft_model.eval()
            eval_alpha, eval_be = evaluate_metrics(
                draft_model,
                target_model,
                eval_prompts,
                tokenizer,
                K=args.K,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                device=device,
                n_eval=args.n_eval_prompts,
            )
            log.info("step=%d  eval_alpha=%.4f  eval_be=%.3f", step + 1, eval_alpha, eval_be)
            if use_wandb:
                _wandb.log({
                    "online/eval_alpha": eval_alpha,
                    "online/eval_be":    eval_be,
                }, step=step + 1)
            # Write rejection rate (1 - eval_alpha) as val "loss" so the curve
            # goes DOWN like other val curves (lower rejection = better draft).
            _write_train_step(
                label=os.path.basename(args.output),
                loss_name=args.kl_method,
                step=step + 1,
                loss=1.0 - eval_alpha,   # rejection rate: lower = better
                learning_rate=args.lr,
                lora_rank=args.lora_r,
                split="val",
            )

    # --- Final evaluation ---
    draft_model.eval()
    final_alpha, final_be = evaluate_metrics(
        draft_model,
        target_model,
        eval_prompts,
        tokenizer,
        K=args.K,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        device=device,
        n_eval=args.n_eval_prompts,
    )
    delta_alpha = final_alpha - baseline_alpha
    delta_be    = final_be    - baseline_be
    log.info(
        "DONE  baseline_alpha=%.4f  final_alpha=%.4f  Δalpha=%+.4f  "
        "baseline_be=%.3f  final_be=%.3f  Δbe=%+.3f",
        baseline_alpha, final_alpha, delta_alpha,
        baseline_be,    final_be,    delta_be,
    )
    print(
        f"\nbaseline_alpha={baseline_alpha:.4f}  "
        f"final_alpha={final_alpha:.4f}  "
        f"alpha_improvement={delta_alpha:+.4f}\n"
        f"baseline_be={baseline_be:.3f}  "
        f"final_be={final_be:.3f}  "
        f"be_improvement={delta_be:+.3f}"
    )
    if use_wandb:
        _wandb.summary["final_alpha"]      = final_alpha
        _wandb.summary["alpha_improvement"] = delta_alpha
        _wandb.summary["final_be"]         = final_be
        _wandb.summary["be_improvement"]   = delta_be
        _wandb.finish()

    # --- Save adapter ---
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    draft_model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    log.info("LoRA adapter saved to: %s", output_path)

    merge_cmd = (
        f"python -c \""
        f"from peft import PeftModel; "
        f"from transformers import AutoModelForCausalLM; "
        f"import torch; "
        f"base = AutoModelForCausalLM.from_pretrained('{args.draft}', torch_dtype=torch.bfloat16); "
        f"m = PeftModel.from_pretrained(base, '{output_path}'); "
        f"m = m.merge_and_unload(); "
        f"m.save_pretrained('{output_path}-merged')\""
    )
    print(f"\nTo merge LoRA weights into a standalone model:\n{merge_cmd}")


if __name__ == "__main__":
    main()
