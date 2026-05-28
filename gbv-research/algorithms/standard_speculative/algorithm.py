"""
Standard speculative decoding — Leviathan et al. 2022 (arXiv:2211.17192).

No training required.  Serves as the comparison baseline and as the
simplest possible SpecDecAlgorithm implementation to read when adding
a new algorithm.
"""

from __future__ import annotations

import time
from typing import Optional

import torch
import torch.nn.functional as F

from algorithms.base import (
    SpecDecAlgorithm, AlgorithmMeta, GenerationResult,
    DraftSource, VerificationType, TrainingType,
)


class StandardSpecAlgorithm(SpecDecAlgorithm):
    """
    Token-level accept/reject speculative decoding.
    No tree, no training — just draft K tokens, verify each one, accept/reject.
    """

    meta = AlgorithmMeta(
        name              = "standard",
        paper_title       = "Fast Inference from Transformers via Speculative Decoding",
        arxiv_id          = "2211.17192",
        draft_source      = DraftSource.INDEPENDENT,
        verification      = VerificationType.TOKEN_LEVEL,
        training          = TrainingType.NONE,
        needs_draft_model = True,
        notes             = "Leviathan et al. 2022.  Linear draft, token-level "
                            "accept/reject.  Use K=1 (single path, no tree).",
    )

    def setup(self, target_model, tokenizer, device, draft_model=None,
              model_family=None, **kwargs) -> None:
        if draft_model is None:
            raise ValueError("standard speculative decoding requires a draft_model.")
        self.target  = target_model
        self.draft   = draft_model
        self.tok     = tokenizer
        self.device  = device

    def generate(
        self,
        prompt_ids,
        max_new_tokens: int = 128,
        K: int = 4,           # draft block length (called γ in the paper)
        L: int = 8,           # ignored — standard SD uses K as the draft length
        temperature: float = 1.0,
        **kwargs,
    ) -> GenerationResult:
        """
        Standard speculative decoding:
          1. Draft model generates K tokens autoregressively.
          2. Target model scores all K+1 positions in one forward pass.
          3. Accept / reject each draft token via rejection sampling.
          4. Always append at least one target-sampled token.
        """
        # prompt_ids contract: shape [1, prompt_len] (single prompt, batch dim present).
        # Reject larger batches explicitly — silently using ids[0] would drop all rows
        # after the first with no error.
        if prompt_ids.dim() == 2:
            if prompt_ids.shape[0] != 1:
                raise ValueError(
                    f"StandardSpecAlgorithm.generate() processes one prompt at a time; "
                    f"got batch_size={prompt_ids.shape[0]}. "
                    "Loop over prompts in the caller."
                )
            ids = prompt_ids.clone().to(self.device)   # [1, prompt_len]
        else:
            ids = prompt_ids.unsqueeze(0).to(self.device)  # [prompt_len] → [1, prompt_len]

        n_draft = 0
        n_accept = 0
        n_target_calls = 0
        t_draft = 0.0
        t_target = 0.0

        generated = []
        t0_total = time.perf_counter()

        while len(generated) < max_new_tokens:
            # ── Step 1: draft K tokens ──────────────────────────────────────
            t0 = time.perf_counter()
            draft_ids   = []
            draft_probs = []
            ctx = ids
            with torch.no_grad():
                for _ in range(K):
                    out   = self.draft(ctx)
                    logit = out.logits[0, -1, :] / temperature
                    prob  = F.softmax(logit, dim=-1)
                    token = torch.multinomial(prob, 1)
                    draft_ids.append(token.item())
                    draft_probs.append(prob[token].item())
                    ctx = torch.cat([ctx, token.unsqueeze(0)], dim=-1)
            t_draft += time.perf_counter() - t0
            n_draft += K

            # ── Step 2: target scores the draft + one extra position ────────
            t0 = time.perf_counter()
            full_seq = torch.cat([
                ids,
                torch.tensor(draft_ids, device=self.device).unsqueeze(0)
            ], dim=-1)
            with torch.no_grad():
                target_logits = self.target(full_seq).logits[0]   # [L+K, V]
            n_target_calls += 1
            t_target += time.perf_counter() - t0

            # ── Step 3: token-level accept / reject ─────────────────────────
            accepted = 0
            for i, (did, dp) in enumerate(zip(draft_ids, draft_probs)):
                tgt_logit = target_logits[ids.shape[1] - 1 + i] / temperature
                tgt_prob  = F.softmax(tgt_logit, dim=-1)
                r = torch.rand(1).item()
                ratio = (tgt_prob[did] / (dp + 1e-9)).item()
                if r < ratio:
                    # accept
                    generated.append(did)
                    accepted += 1
                    n_accept += 1
                    if did == self.tok.eos_token_id:
                        break
                else:
                    # reject: sample residual distribution
                    residual = (tgt_prob - dp * torch.ones_like(tgt_prob)).clamp(min=0)
                    if residual.sum() > 0:
                        residual = residual / residual.sum()
                        token = torch.multinomial(residual, 1).item()
                    else:
                        token = torch.multinomial(tgt_prob, 1).item()
                    generated.append(token)
                    n_target_calls += 1   # implicit call for residual
                    break

            if accepted == K:
                # All accepted — also sample one more from target
                tgt_logit = target_logits[ids.shape[1] - 1 + K] / temperature
                tgt_prob  = F.softmax(tgt_logit, dim=-1)
                bonus = torch.multinomial(tgt_prob, 1).item()
                generated.append(bonus)

            # Update context with accepted tokens
            new_ids = torch.tensor(generated[-accepted - 1:], device=self.device)
            ids = torch.cat([ids, new_ids.unsqueeze(0)], dim=-1)

            if self.tok.eos_token_id in generated[-K - 1:]:
                break

        elapsed = time.perf_counter() - t0_total

        return GenerationResult(
            token_ids         = generated,
            n_draft_tokens    = n_draft,
            n_accepted_tokens = n_accept,
            n_target_calls    = n_target_calls,
            time_draft_s      = t_draft,
            time_target_s     = t_target,
            time_total_s      = elapsed,
        )

    def train(self, *args, **kwargs) -> None:
        print("[standard] No training needed — standard speculative decoding "
              "works with any pre-trained draft model.")
