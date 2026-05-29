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

# Loss registry — canonical implementations live in distillspec_gbv/losses/.
# online_serve.py is in algorithms/ which is sys.path[0] at runtime, so the
# subpackage import resolves without any extra sys.path surgery.
from distillspec_gbv.losses import get_loss as _get_loss

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

    # ── Route through canonical loss registry ─────────────────────────────────
    # s_sel and t_sel are (N_rej, V) temperature-scaled logits (not log-probs);
    # the package functions (forward_kl / reverse_kl / jsd) apply log_softmax
    # internally, so we hand them the raw scaled logits directly.
    # Memory efficiency: only the small rejected-position slice is ever
    # materialised in the loss computation — same guarantee as before.
    return _get_loss(kl_method, s_sel, t_sel).loss


# ---------------------------------------------------------------------------
# EBE loss for online speculative decoding
#
# !! BUG — EBE GRADIENT IS EFFECTIVELY ZERO (2026-05-28, diagnosed by Rahul)
# !! DO NOT USE this function until the buffer is redesigned.
# !! See DECISIONS.md §2 for the full root-cause analysis and fix plan.
#
# Root cause:
#   speculative_step() stores the *accepted + residual* sequence in the buffer.
#   At each rejected position, the stored token is the *teacher-resampled residual*
#   token, NOT the draft's actual proposal.
#
#   For a residual token t_res:
#     q_teacher(t_res) is HIGH  — teacher assigned it reasonable prob (it was sampled
#                                  from the teacher's adjusted distribution (p_t - p_d)+)
#     p_draft(t_res) is LOW     — draft didn't like it (that's why it was in the residual)
#     → log_q - log_p >> 0  → clamped to 0  → α(t_res) = 1.0
#
#   For an accepted token t_acc:
#     α(t_acc) = 1.0 by the definition of speculative decoding acceptance.
#
#   RESULT: α = 1.0 everywhere → EBE gradient = 0 everywhere.
#   Only the KL regulariser runs, and it's operating on the wrong tokens.
#   The incoherent KL update corrupts the draft → PPL rises → BE → 1.0.
#   This explains the "far left" scatter in the dashboard.
#
# Fix (requires buffer redesign):
#   The replay buffer must separately store "what the draft actually proposed
#   at each position" (draft_proposed_ids: Tensor[T]), alongside the
#   accepted+residual sequence. _ebe_online must then:
#     - For accepted positions:    use stored token (same as now, α = 1, ∂/∂θ = 0)
#     - For rejected positions:    use draft_proposed_ids[pos] (not the residual)
#                                  so p_draft is HIGH and q_teacher can be LOW → α < 1 → real gradient
#
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

    ** BROKEN ** — see the BUG comment above. The EBE gradient is zero everywhere
    because token_ids contains residual-resampled tokens at rejected positions,
    not the draft's actual proposals. Do not use until buffer is redesigned.

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

    # Temperature-scaled logits (kept as named tensors so the KL regulariser can
    # pass raw logits to the loss registry — package functions apply log_softmax
    # internally and therefore require logits, not log-probs).
    s_scaled = student_logits.float() / temperature                           # (B, T, V) grad
    with torch.no_grad():
        t_scaled = teacher_logits.float() / temperature                       # (B, T, V) no grad
    log_s = F.log_softmax(s_scaled, dim=-1)                                   # (B, T, V)
    with torch.no_grad():
        log_t = F.log_softmax(t_scaled, dim=-1)                               # (B, T, V)

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

    # ── KL regulariser — rejected positions only (via loss registry) ──────────
    # Rationale: EBE gradient is 0 for under-estimated tokens (α = 1); KL covers
    # those positions.  We restrict it to rejected positions (wrong_mask) rather
    # than all tokens to stay consistent with OSD's philosophy of not penalising
    # the draft where it already aligns with the target.
    # We pass s_scaled / t_scaled (raw temperature-scaled logits) to the registry
    # because package functions apply log_softmax internally.
    mask_flat = wrong_mask.reshape(-1)                                       # (B*T,)
    if mask_flat.sum() == 0:
        return ebe

    s_rej = s_scaled.reshape(-1, V)[mask_flat]                              # (N_rej, V) grad
    t_rej = t_scaled.reshape(-1, V)[mask_flat].detach()                     # (N_rej, V) no grad
    kl = _get_loss("forward_kl", s_rej, t_rej).loss                         # scalar

    return ebe + kl_weight * kl


# ---------------------------------------------------------------------------
# EBE-single loss for online speculative decoding
# ---------------------------------------------------------------------------

def _ebe_single_online(
    student_logits: torch.Tensor,   # (B, T, V) — grad required
    teacher_logits: torch.Tensor,   # (B, T, V) — detached (no grad)
    token_ids: torch.Tensor,        # (B, T)    — actual generated tokens
    wrong_mask: torch.Tensor,       # (B, T)    — True at rejected positions
    temperature: float = 1.0,
) -> torch.Tensor:
    """Single-token EBE at rejected positions: Loss = −mean(α) where α = min(1, q/p).

    Ablation of _ebe_online: removes the block cumprod structure entirely.
    Only computes gradients at rejected positions (wrong_mask=True) to stay
    consistent with OSD's principle of not updating at well-aligned positions.

    This tests whether the product structure in _ebe_online is the bottleneck
    (gradient vanishing through 4-8 chained multiplications) vs. the EBE
    concept itself.
    """
    B, T, V = student_logits.shape
    mask_flat = wrong_mask.reshape(-1)                              # (B*T,)

    if mask_flat.sum() == 0:
        return (student_logits * 0).sum()

    # Select only rejected positions (N_rej ≪ B*T — memory-efficient)
    s_sel    = student_logits.reshape(-1, V)[mask_flat].float() / temperature  # (N_rej, V) grad
    t_sel    = teacher_logits.reshape(-1, V)[mask_flat].detach().float() / temperature  # no grad
    ids_flat = token_ids.reshape(-1)[mask_flat]                                # (N_rej,)
    del student_logits, teacher_logits
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Route through canonical loss registry — same α = min(1, q/p) math as
    # offline ebe_single, but applied only at rejected positions.
    # s_sel / t_sel are temperature-scaled logits; ebe_single applies log_softmax
    # internally and accepts token_ids for per-position gathering.
    return _get_loss("ebe_single", s_sel, t_sel, token_ids=ids_flat).loss


# ---------------------------------------------------------------------------
# Speculative decoding (single sequence, K tokens per step)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _draft_propose(
    draft_model,
    input_ids: torch.Tensor,
    K: int,
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Autoregressively generate K draft tokens by sampling at temperature T.

    BUG FIX (2026-05-29): The previous implementation used greedy argmax
    (temperature = 0) while speculative_step computed the acceptance ratio
    using temperature-scaled softmax probabilities.  This violated the
    fundamental spec-dec assumption: the acceptance ratio min(1, p_tgt/p_dft)
    is only valid when the token was actually sampled from q_draft(·|T).

    With argmax sampling:
      - The chosen token has the highest temperature-scaled softmax probability
      - p_dft = q_draft_T(argmax) is very large → p_tgt/p_dft << 1 → most
        tokens rejected even when draft and teacher agree on the top choice
      - Inflated rejection count → KL updates push model toward teacher at
        spurious positions → divergence (ppl 15.46 vs baseline 7.47 observed)

    Fix: sample from the temperature-scaled draft distribution so the token's
    proposal probability exactly matches the p_dft used in the acceptance ratio.
    This aligns with OSD (Liu et al. 2023) which requires t ~ q_draft(·|T).

    Args:
        draft_model: the (LoRA-wrapped) draft model.
        input_ids:   (1, L) prompt token ids.
        K:           number of tokens to propose.
        temperature: sampling temperature; must match the temperature passed to
                     speculative_step so the acceptance ratio is consistent.

    Returns:
        draft_tokens:  (K,) proposed token ids (int64).
        draft_logprob: (K,) temperature-scaled log-prob of each proposed token
                       under the draft.  Used by callers that need the log-prob
                       for importance-weighting or logging.
    """
    device = input_ids.device
    extended = input_ids.clone()  # (1, L)

    for _ in range(K):
        out = draft_model(extended)
        next_logits = out.logits[:, -1, :]  # (1, V)
        if temperature <= 0.0:
            # temperature = 0 is deterministic argmax (used for smoke tests / debugging)
            next_tok = next_logits.argmax(dim=-1, keepdim=True)
        else:
            probs = F.softmax(next_logits.float() / temperature, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)  # (1, 1)
        extended = torch.cat([extended, next_tok], dim=1)

    # extended is now (1, L+K); draft tokens are the last K positions
    draft_tokens = extended[0, -K:]  # (K,)

    # Re-run the full candidate sequence in ONE forward pass to get
    # per-position temperature-scaled log-probs for the K draft positions.
    # Position i in extended predicts token i+1, so draft position j
    # (0-indexed from L) is predicted at logit index L-1+j.
    logits_full = draft_model(extended).logits  # (1, L+K, V)

    L = input_ids.shape[1]
    # logits at positions L-1 .. L+K-2  predict tokens at positions L .. L+K-1
    draft_logits = logits_full[0, L - 1 : L + K - 1, :]  # (K, V)
    draft_logprob = F.log_softmax(draft_logits.float() / temperature, dim=-1)
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

        # --- Draft proposes k tokens (sampled at temperature T, not greedy) ---
        # temperature must be passed here so the proposal distribution matches
        # the distribution used in the acceptance ratio (p_tgt/p_dft) below.
        draft_tokens, draft_lp = _draft_propose(draft_model, seq, k,
                                                 temperature=temperature)
        # draft_tokens: (k,), draft_lp: (k,) temperature-scaled log-probs

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
            dtype=dtype,          # transformers ≥ 4.51 (torch_dtype= deprecated)
            device_map=device,
        )
    target_model.eval()
    for p in target_model.parameters():
        p.requires_grad_(False)

    log.info("Loading draft base model: %s", args.draft)
    draft_base = AutoModelForCausalLM.from_pretrained(
        args.draft,
        dtype=dtype,              # transformers ≥ 4.51 (torch_dtype= deprecated)
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
        _lora_alpha = args.lora_alpha if args.lora_alpha is not None else args.lora_r * 2
        log.info("Wrapping draft with fresh LoRA (r=%d, alpha=%d)", args.lora_r, _lora_alpha)
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=_lora_alpha,
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
    elif kl_method == "ebe_single":
        # _ebe_single_online: −mean(α) at rejected positions only.
        # Ablation: tests whether the cumprod chain is the source of instability.
        loss = _ebe_single_online(
            stu_logits, tgt_logits, token_ids, shifted_mask,
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
# On-policy tree distillation update (online-tree mode)
# ---------------------------------------------------------------------------
#
# WHY THIS IS BETTER THAN FLAT ONLINE TRAINING
# ---------------------------------------------
# Flat online training (update_step above) trains on the replay buffer of
# accepted+residual sequences.  For KL: only rejected positions get gradient.
# For EBE: zero gradient everywhere (see the BUG comment above).
#
# Tree training builds the student's own K-path draft tree from the CURRENT
# prompt, scores it with the target, and back-propagates through the student's
# distributions at every tree node — no dead gradient positions.
#
# This gives:
#   1. On-policy data: student trains on its own actual proposals, not the
#      teacher's residual-resampled tokens.
#   2. Prompt-distribution-matched: each update uses the real served prompt,
#      not a fixed offline training set.
#   3. Richer gradient: K paths × L nodes per update vs N_rejected ≪ K*L
#      positions from the flat buffer.
#
# VRAM: building K=2 paths costs ~2× a single forward pass for the draft,
# ~2× for the target (tree attention pass), then one more WITH-grad draft pass.
# Total: ~3 draft forwards + 2 target forwards per update (vs 1+1 flat).
# Use tree_K=2 on T4/Colab, tree_K=4 on A100.
#
# PAIRING WITH VERIFIERS
# ----------------------
# kl_tree and ebe_tree pair most naturally with online serving because the
# standard speculative decoding verifier uses per-token acceptance
# α = min(1, p[t]/q[t]) — the same quantity these losses optimize.
# bv_tree/gbv_tree also work (they also use on-policy tree data) but are
# designed for their specific non-OT verifiers.
# ---------------------------------------------------------------------------

def _tree_online_update(
    draft_model,
    target_model,
    prompt_ids: torch.Tensor,     # [1, T] tokenised prompt
    optimizer,
    loss_name: str,               # any name in TREE_LOSS_NAMES
    tree_K: int,
    tree_L: int,
    temperature: float,
) -> float:
    """
    One on-policy tree distillation step for the online training loop.

    Mirrors _tree_training_step from trainer.py but adapted for the online
    context: takes a single prompt tensor rather than a DataLoader batch,
    and does NOT use the replay buffer (builds a fresh tree from scratch).

    Steps:
      1. Draft builds a K-path tree from the current prompt (no grad).
      2. Target scores every tree node (no grad).
      3. Draft re-runs over the fixed tree WITH grad.
      4. Compute tree loss → backward → optimizer step.

    Returns the float loss value for logging.
    """
    # Lazy imports: only paid when tree mode is actually active.
    from distillspec_gbv.losses.tree_losses import compute_tree_loss
    from distillspec_gbv.tree_training import draft_tree_forward_with_grad
    from distillspec_gbv.verifiers.draft_generator import iid_draft, target_tree_pass

    device = prompt_ids.device

    draft_model.eval()
    with torch.no_grad():
        # ── 1. Student builds draft tree ─────────────────────────────────────
        q_init  = draft_model(prompt_ids, use_cache=True, return_dict=True)
        q_cache = q_init.past_key_values
        pending = torch.multinomial(
            F.softmax(q_init.logits[0, -1] / temperature, dim=-1), 1
        ).unsqueeze(0)                              # [1, 1]
        q_paths, _, _ = iid_draft(
            draft_model, q_cache, pending,
            K=tree_K, L=tree_L, q_temp=temperature,
        )

        # ── 2. Target scores the draft tree ──────────────────────────────────
        p_init  = target_model(prompt_ids, use_cache=True, return_dict=True)
        p_cache = p_init.past_key_values
        _, _, _, p_probs_dict = target_tree_pass(
            target_model, p_cache, q_paths,
            K=tree_K, L=tree_L, p_temp=temperature,
        )
        # p_probs_dict: {prefix → [V]}, all detached

    # ── 3. Student re-runs on fixed tree WITH grad ────────────────────────────
    q_probs_grad = draft_tree_forward_with_grad(
        draft_model, prompt_ids, q_paths,
        L=tree_L, K=tree_K, q_temp=temperature,
    )

    # ── 4. Tree loss + backward ───────────────────────────────────────────────
    loss = compute_tree_loss(
        loss_name, q_probs_grad, p_probs_dict, q_paths,
        L=tree_L, K=tree_K,
    )
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(draft_model.parameters(), max_norm=1.0)
    optimizer.step()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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
    parser.add_argument("--lora_alpha", type=int, default=None,
                        help="LoRA alpha. Defaults to 2 * lora_r if not set.")
    parser.add_argument("--milestone_every", type=int, default=0,
                        help="Save a permanent milestone checkpoint to ckpt_step_NNNNN/ "
                             "every N steps in addition to the rolling ckpt_latest/. "
                             "0 = disabled. Useful for EBE/online runs that can peak early "
                             "and then collapse — milestone preserves the best model.")
    parser.add_argument("--early_stop_patience", type=int, default=0,
                        help="Stop training if eval_be worsens for this many consecutive "
                             "eval windows. 0 = disabled (run all --steps).")
    parser.add_argument(
        "--kl_method",
        default="forward_kl",
        choices=["forward_kl", "reverse_kl", "jsd", "ebe", "ebe_single"],
        help=(
            "Flat training objective (replay-buffer mode).  Ignored when --tree_loss is set. "
            "'forward_kl'/'reverse_kl'/'jsd': KL variant at rejected positions (OSD original). "
            "'ebe': block-level EBE over full sequences + KL reg at rejected positions. "
            "'ebe_single': single-token EBE = −mean(α) at rejected positions (ablation vs ebe)."
        ),
    )
    parser.add_argument(
        "--tree_loss",
        default=None,
        choices=[
            "kl_tree",      # forward KL at each tree node (on-policy baseline)
            "rev_kl_tree",  # reverse KL at each tree node (mode-seeking)
            "jsd_tree",     # JSD at each tree node (symmetric, bounded)
            "ebe_tree",     # on-policy EBE — natural pair for online serving
            "bv_tree",      # BV block acceptance integral (on-policy)
            "gbv_tree",     # GBV with q_skew (on-policy)
            "traversal_tree",
        ],
        help=(
            "On-policy tree distillation loss.  When set, replaces the flat replay-buffer "
            "update_step with _tree_online_update: builds a K-path draft tree from the current "
            "prompt, scores with target, and back-propagates through all tree nodes.  "
            "Most natural pairings for online (standard SD) verifier: kl_tree, ebe_tree.  "
            "Use --tree_K to control the path count (default 2 for VRAM budget on T4/Colab; "
            "use 4 on A100)."
        ),
    )
    parser.add_argument(
        "--tree_K", type=int, default=2,
        help="Number of i.i.d. draft paths per tree training step.  "
             "Default 2 (safe on T4/Colab 15 GB).  Use 4 on A100.",
    )
    parser.add_argument(
        "--tree_L", type=int, default=8,
        help="Draft block depth for tree training.  Should match --K (tokens per step).  "
             "Default 8.",
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
    # Early-stop tracking: count consecutive eval windows where eval_be worsens.
    _best_eval_be = baseline_be
    _early_stop_bad_windows = 0

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

        # 3. Record into buffer (alpha tracking; tree mode doesn't use the buffer
        #    for training but still tracks alpha for logging/eval).
        buffer.add(full_ids, wrong_positions)
        alpha_window.append(alpha)

        # 4. Periodic update
        if (step + 1) % args.update_every == 0:
            if args.tree_loss:
                # On-policy tree distillation: build a fresh K-path tree from the
                # current prompt and back-propagate through all tree nodes.
                # Does NOT use the replay buffer — on-policy data is built here.
                if len(input_ids[0]) > 1:   # skip if prompt is too short for a tree
                    last_loss = _tree_online_update(
                        draft_model,
                        target_model,
                        input_ids,
                        optimizer,
                        loss_name=args.tree_loss,
                        tree_K=args.tree_K,
                        tree_L=args.tree_L,
                        temperature=args.temperature,
                    )
                    update_count += 1
            elif len(buffer) > 0:
                # Flat replay-buffer update (original OSD behaviour).
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
            # Milestone checkpoint — permanent named snapshot, never overwritten.
            # Fires at multiples of --milestone_every (independent of save_every).
            if args.milestone_every and (step + 1) % args.milestone_every == 0:
                _ms_dir = Path(args.output) / f"ckpt_step_{step+1:05d}"
                try:
                    _ms_dir.mkdir(parents=True, exist_ok=True)
                    draft_model.save_pretrained(str(_ms_dir))
                    tokenizer.save_pretrained(str(_ms_dir))
                    log.info("[ckpt] Milestone at step %d → %s", step + 1, _ms_dir)
                except Exception as _me:
                    log.warning("[ckpt] Milestone save failed at step %d: %s", step + 1, _me)

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
            # Early stopping: stop if eval_be worsens for patience consecutive windows.
            if eval_be > _best_eval_be:
                _best_eval_be = eval_be
                _early_stop_bad_windows = 0
            else:
                _early_stop_bad_windows += 1
                if args.early_stop_patience and _early_stop_bad_windows >= args.early_stop_patience:
                    log.info(
                        "[early_stop] eval_be did not improve for %d consecutive windows "
                        "(best=%.3f, current=%.3f) — stopping at step %d",
                        _early_stop_bad_windows, _best_eval_be, eval_be, step + 1,
                    )
                    break

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
