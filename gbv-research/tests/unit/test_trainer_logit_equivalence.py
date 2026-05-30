"""
test_trainer_logit_equivalence.py
==================================
Verifies that the optimised teacher-logit path introduced in trainer.py is
mathematically equivalent to the original output_scores approach.

Change under test (trainer.py — training AND validation loops):

  BEFORE:
    gen_out = model.generate(..., output_scores=True, temperature=T)
    scores  = torch.stack(gen_out.scores).squeeze(1).float()
    t_log   = scores * T             # recover_raw_logits undoes /T scaling

  AFTER:
    gen_out = model.generate(..., temperature=T)   # no output_scores
    t_log   = model(gen_out.sequences).logits[0, plen-1:-1, :].float()

Correctness argument
--------------------
A standard transformer with causal masking guarantees that the logit vector at
position `plen-1+i` in a full forward pass on the complete sequence equals the
distribution the model computed when sampling token i autoregressively
(because causal masking ensures position j only attends to j′ ≤ j, so the
KV state at step i of generation is identical to the KV state at row plen-1+i
of a full forward pass).

The fresh forward pass also returns *raw* logits (no temperature applied),
matching what recover_raw_logits undoes.

These tests confirm this numerically using a tiny hand-rolled causal
transformer — **no HuggingFace model download required**, runs on CPU in <1 s.
"""

import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Tiny causal transformer (stands in for any HF CausalLM)
# ---------------------------------------------------------------------------

class _TinyCausalLM(nn.Module):
    """
    Minimal single-layer causal transformer language model.
    vocab_size=32, embed_dim=16, n_heads=2, max_seq=32
    Produces logits[batch, seq, vocab] with standard causal masking.
    """
    VOCAB = 32
    DIM   = 16
    HEADS = 2
    MAXL  = 32

    def __init__(self, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.embed  = nn.Embedding(self.VOCAB, self.DIM)
        self.pos    = nn.Embedding(self.MAXL,  self.DIM)
        self.attn   = nn.MultiheadAttention(self.DIM, self.HEADS, batch_first=True)
        self.fc_out = nn.Linear(self.DIM, self.VOCAB, bias=False)
        self.eval()

    def forward(self, input_ids):
        B, L = input_ids.shape
        pos_ids = torch.arange(L, device=input_ids.device).unsqueeze(0)
        x = self.embed(input_ids) + self.pos(pos_ids)   # [B, L, D]
        mask = torch.triu(torch.full((L, L), float("-inf")), diagonal=1)
        x, _ = self.attn(x, x, x, attn_mask=mask, need_weights=False)
        return self.fc_out(x)   # [B, L, VOCAB]

    def generate(self, prompt_ids, max_new_tokens, temperature=1.0,
                 collect_scores=False):
        """
        Simple greedy / temperature-sampled autoregressive generation.
        Returns (full_ids, scores_list) where scores_list is empty when
        collect_scores=False (mirrors transformers' output_scores flag).
        """
        ids    = prompt_ids.clone()
        scores = []
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = self.forward(ids)[0, -1, :]   # [VOCAB]
                if collect_scores:
                    # Mimic transformers: store logits /temperature
                    scores.append((logits / temperature).unsqueeze(0))
                probs  = F.softmax(logits / temperature, dim=-1)
                next_t = torch.argmax(probs, keepdim=True).unsqueeze(0)  # greedy
                ids    = torch.cat([ids, next_t], dim=1)
        return ids, scores


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_model():
    return _TinyCausalLM(seed=7)


class TestLogitEquivalenceCore:
    """
    The three invariants that the trainer change relies on.
    """

    def test_causal_forward_equals_autoregressive_step(self, tiny_model):
        """
        At position plen-1+i, a full forward pass on the complete sequence
        gives the same logit vector as the model computed when autoregressively
        generating token i.

        This is the core mathematical property the trainer change relies on.
        """
        prompt = torch.tensor([[1, 5, 10]])
        plen   = prompt.shape[1]
        N      = 6

        # Old: collect per-token logits during generation (scores = logits/T, T=1 → identical)
        full_ids, scores_list = tiny_model.generate(
            prompt, max_new_tokens=N, temperature=1.0, collect_scores=True)
        t_log_old = torch.cat(scores_list, dim=0).float()   # [N, VOCAB]

        # New: single forward pass
        with torch.no_grad():
            t_log_new = tiny_model(full_ids)[0, plen - 1:-1, :].float()  # [N, VOCAB]

        assert t_log_old.shape == t_log_new.shape, (
            f"Shape mismatch: old={t_log_old.shape} new={t_log_new.shape}"
        )
        max_diff = (t_log_old - t_log_new).abs().max().item()
        assert max_diff < 1e-5, (
            f"Logits differ by {max_diff:.2e} — causal equivalence broken.\n"
            f"old[:2]: {t_log_old[:2].tolist()}\n"
            f"new[:2]: {t_log_new[:2].tolist()}"
        )

    def test_temperature_recovery(self, tiny_model):
        """
        With temperature T != 1:
          - output_scores stores logits / T
          - recover_raw_logits multiplies by T → raw logits
          - fresh forward pass gives raw logits directly
        Both must match.
        """
        prompt = torch.tensor([[2, 4, 6, 8]])
        plen   = prompt.shape[1]
        T      = 0.7
        N      = 5

        full_ids, scores_list = tiny_model.generate(
            prompt, max_new_tokens=N, temperature=T, collect_scores=True)
        # Old: scores = logits/T → multiply back
        t_log_old = torch.cat(scores_list, dim=0).float() * T

        # New: raw logits from fresh forward pass
        with torch.no_grad():
            t_log_new = tiny_model(full_ids)[0, plen - 1:-1, :].float()

        max_diff = (t_log_old - t_log_new).abs().max().item()
        assert max_diff < 1e-5, (
            f"T={T}: logits differ by {max_diff:.2e} after temperature recovery"
        )

    def test_shape_slicing(self, tiny_model):
        """
        full_ids.shape = [1, plen + gen_len]
        We want gen_len rows: positions plen-1 through plen+gen_len-2 inclusive.
        Slice [plen-1 : -1] must equal [plen-1 : plen-1+gen_len].
        Verify for several plen / gen_len combinations.
        """
        for plen in (2, 4, 7):
            for gen_len in (1, 3, 8):
                ids = torch.randint(0, _TinyCausalLM.VOCAB, (1, plen + gen_len))
                with torch.no_grad():
                    logits = tiny_model(ids)[0]  # [plen+gen_len, VOCAB]
                slice_neg   = logits[plen - 1:-1]
                slice_exact = logits[plen - 1: plen - 1 + gen_len]
                assert torch.equal(slice_neg, slice_exact), (
                    f"Slice mismatch for plen={plen}, gen_len={gen_len}"
                )
                assert slice_neg.shape[0] == gen_len, (
                    f"Wrong number of rows: got {slice_neg.shape[0]}, expected {gen_len}"
                )

    def test_gen_len_zero_guard(self, tiny_model):
        """
        If no tokens are generated (edge-case), gen_len == 0.
        The guard `if gen_len == 0: continue` must be detectable,
        and the slice [plen-1:-1] must be empty (not crash, not silent wrong data).
        """
        prompt  = torch.tensor([[1, 2, 3]])
        plen    = prompt.shape[1]
        # Simulate: full_ids == prompt (0 tokens generated)
        full_ids = prompt.clone()
        gen_len  = full_ids.shape[1] - plen   # == 0
        assert gen_len == 0, "Test setup error"

        # Verify the slice is empty when gen_len == 0
        with torch.no_grad():
            logits = tiny_model(full_ids)[0, plen - 1:-1, :]
        assert logits.shape[0] == 0, (
            f"Expected empty tensor for gen_len=0, got shape {logits.shape}"
        )


class TestLogitEquivalenceDistributions:
    """
    Verify that training with old vs new t_log produces the same gradient
    directions (loss is identical → backward pass is identical).
    """

    def test_forward_kl_loss_identical(self, tiny_model):
        """
        KL(t_log_old ‖ s_log) == KL(t_log_new ‖ s_log)
        when t_log_old and t_log_new are numerically equal.
        """
        torch.manual_seed(99)
        prompt   = torch.tensor([[3, 7, 12]])
        plen     = prompt.shape[1]
        N        = 4

        full_ids, scores_list = tiny_model.generate(
            prompt, max_new_tokens=N, temperature=1.0, collect_scores=True)
        t_log_old = torch.cat(scores_list, dim=0).float()

        with torch.no_grad():
            t_log_new = tiny_model(full_ids)[0, plen - 1:-1, :].float()

        # Use a random student logit tensor for both loss computations
        torch.manual_seed(42)
        s_log = torch.randn(N, _TinyCausalLM.VOCAB)

        def fwd_kl(s, t):
            p_t = F.softmax(t, dim=-1)
            return F.cross_entropy(s, p_t)

        loss_old = fwd_kl(s_log, t_log_old).item()
        loss_new = fwd_kl(s_log, t_log_new).item()

        assert abs(loss_old - loss_new) < 1e-5, (
            f"Forward KL loss differs: old={loss_old:.6f} new={loss_new:.6f}"
        )

    def test_gradient_direction_identical(self, tiny_model):
        """
        Student gradient must be identical whether we use t_log_old or t_log_new.
        """
        torch.manual_seed(10)
        prompt = torch.tensor([[1, 3]])
        plen   = prompt.shape[1]
        N      = 5

        full_ids, scores_list = tiny_model.generate(
            prompt, max_new_tokens=N, temperature=1.0, collect_scores=True)
        t_log_old = torch.cat(scores_list, dim=0).float()

        with torch.no_grad():
            t_log_new = tiny_model(full_ids)[0, plen - 1:-1, :].float()

        torch.manual_seed(77)
        s_old = torch.randn(N, _TinyCausalLM.VOCAB, requires_grad=True)
        s_new = s_old.clone().detach().requires_grad_(True)

        p_t_old = F.softmax(t_log_old, dim=-1)
        p_t_new = F.softmax(t_log_new, dim=-1)

        F.cross_entropy(s_old, p_t_old).backward()
        F.cross_entropy(s_new, p_t_new).backward()

        max_grad_diff = (s_old.grad - s_new.grad).abs().max().item()
        assert max_grad_diff < 1e-5, (
            f"Gradient differs by {max_grad_diff:.2e} between old and new t_log"
        )
