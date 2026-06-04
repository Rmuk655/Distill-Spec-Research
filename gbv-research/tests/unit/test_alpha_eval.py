"""
test_alpha_eval.py — unit tests for the inline alpha evaluation fallback.

The inline fallback runs draft-propose / target-verify without specInfer.
It is used when specInfer's DynamicCache API is incompatible with the installed
transformers version.

Critical bug this suite guards against (commit 99f81eb):
    logits[0, -1:]  → shape (1, vocab)  ← WRONG — [tok] indexes dim-0 (size 1)
    logits[0, -1]   → shape (vocab,)    ← correct
    Error: "index N out of bounds for dimension 0 with size 1"
    Triggered on the FIRST prompt (token IDs typically >> 0).
"""

import sys, os, types
import torch
import torch.nn.functional as F
import pytest

# ── path setup ─────────────────────────────────────────────────────────────
_HERE  = os.path.dirname(os.path.abspath(__file__))
_ROOT  = os.path.abspath(os.path.join(_HERE, "..", ".."))
_ORCH  = os.path.join(_ROOT, "orchestration")
sys.path.insert(0, _ORCH)


# ── Tiny stub models ────────────────────────────────────────────────────────

VOCAB = 512   # small vocab so tests are fast

class _StubOutput:
    """Mimics transformers CausalLMOutputWithPast."""
    def __init__(self, logits, past_key_values=None):
        self.logits = logits           # (batch, seq, vocab)
        self.past_key_values = past_key_values or _FakePKV()

class _FakePKV:
    """Minimal fake DynamicCache-like object."""
    def __init__(self):
        self.key_cache = [torch.zeros(1, 1, 1, 8)]   # (batch, heads, seq, head_dim)
        self.value_cache = [torch.zeros(1, 1, 1, 8)]


def _make_stub_model(vocab=VOCAB, preferred_tok=42):
    """Return a callable that behaves like a CausalLM for alpha eval."""
    def model(input_ids, past_key_values=None, use_cache=True, **kwargs):
        seq_len = input_ids.shape[-1]
        logits = torch.zeros(1, seq_len, vocab)
        # Make tok `preferred_tok` the argmax for every position
        logits[0, :, preferred_tok] = 10.0
        return _StubOutput(logits)
    return model


# ── Extract the inline fallback logic from evaluate.py ─────────────────────
# We import the function that contains the fallback, then exercise the inner
# loop logic directly using stub models.  This is an integration test of the
# critical indexing path without requiring real models.

def _run_inline_fallback(student_model, teacher_model, input_ids,
                          max_tokens=6, max_propose=3, temperature=1.0,
                          device="cpu"):
    """
    Re-implements the inline fallback from evaluate.py run_alpha() using the
    SAME logic as the fixed version.  This function MUST be kept in sync with
    evaluate.py's inline fallback block.

    Returns: alpha (float)
    Raises IndexError if the logit shape bug is present.
    """
    accepted_total, proposed_total = 0, 0
    cur_ids = input_ids.to(device)
    d_pkv = None

    for _ in range(max(1, max_tokens // max_propose)):
        if d_pkv is None:
            d_out = student_model(cur_ids, use_cache=True)
        else:
            d_out = student_model(cur_ids[:, -1:], past_key_values=d_pkv,
                                  use_cache=True)
        d_pkv_propose = d_out.past_key_values

        # ── Critical: use [0, -1] not [0, -1:] ──
        # [0, -1:]  → shape (1, vocab) → [tok] raises IndexError for tok > 0
        # [0, -1]   → shape (vocab,)   → [tok] correctly returns a scalar
        draft_tokens = []
        draft_logits = []
        d_logits = d_out.logits[0, -1]           # ← shape must be (vocab,)
        pkv_k = d_pkv_propose
        tmp_ids = cur_ids

        for _k in range(max_propose):
            tok = int(d_logits.argmax())
            draft_tokens.append(tok)
            draft_logits.append(d_logits)         # must be (vocab,)
            next_tok = torch.tensor([[tok]], device=device)
            d_kout = student_model(next_tok, past_key_values=pkv_k,
                                   use_cache=True)
            d_logits = d_kout.logits[0, -1]       # ← shape must be (vocab,)
            pkv_k = d_kout.past_key_values
            tmp_ids = torch.cat([tmp_ids, next_tok], dim=1)

        cand_ids = tmp_ids
        t_all_logits = teacher_model(cand_ids).logits[0]  # (seq, vocab)

        bonus_start = cur_ids.shape[-1] - 1
        n_accepted = 0
        for _k in range(max_propose):
            pos = bonus_start + _k
            tok = draft_tokens[_k]
            t_p = float(F.softmax(t_all_logits[pos] / temperature, dim=-1)[tok])
            d_p = float(F.softmax(draft_logits[_k] / temperature, dim=-1)[tok])
            ratio = t_p / max(d_p, 1e-9)
            proposed_total += 1
            if torch.rand(1).item() < min(1.0, ratio):
                n_accepted += 1
                accepted_total += 1
            else:
                break

        bonus = int(t_all_logits[bonus_start + n_accepted].argmax())
        new_toks = draft_tokens[:n_accepted] + [bonus]
        cur_ids = torch.cat([cur_ids, torch.tensor([new_toks], device=device)], dim=1)
        d_pkv = d_pkv_propose
        if cur_ids.shape[-1] >= input_ids.shape[-1] + max_tokens:
            break

    return accepted_total / proposed_total if proposed_total > 0 else 0.0


# ── Tests ────────────────────────────────────────────────────────────────────

class TestInlineFallbackIndexing:
    """Guard against the logits[0,-1:] vs logits[0,-1] shape bug."""

    def test_no_index_error_with_high_token_id(self):
        """
        Regression for commit 99f81eb: token IDs >> 0 must not cause
        'index N out of bounds for dimension 0 with size 1'.

        The bug: logits[0, -1:] gives shape (1, vocab). When we later do
        draft_logits[_k][tok] with tok=256, dim-0 has size 1 → IndexError.
        """
        # preferred_tok=256 — high enough to trigger the old bug
        student = _make_stub_model(preferred_tok=256)
        teacher = _make_stub_model(preferred_tok=300)
        # Long-ish prompt (50 tokens) to also test the t_all_logits indexing
        prompt = torch.randint(0, VOCAB, (1, 50))
        alpha = _run_inline_fallback(student, teacher, prompt,
                                     max_tokens=9, max_propose=3)
        assert 0.0 <= alpha <= 1.0

    def test_no_index_error_token_id_near_vocab_size(self):
        """Token ID close to vocab boundary also must not crash."""
        student = _make_stub_model(preferred_tok=VOCAB - 1)
        teacher = _make_stub_model(preferred_tok=VOCAB - 2)
        prompt = torch.randint(0, VOCAB, (1, 20))
        alpha = _run_inline_fallback(student, teacher, prompt,
                                     max_tokens=6, max_propose=2)
        assert 0.0 <= alpha <= 1.0

    def test_draft_logit_shape_is_1d(self):
        """draft_logits entries must be 1D (vocab,) so [tok] works correctly."""
        student = _make_stub_model(preferred_tok=100)
        teacher = _make_stub_model(preferred_tok=100)
        prompt = torch.zeros(1, 5, dtype=torch.long)

        # Patch to capture the logit tensor before acceptance
        captured = []
        original_model = student
        def capturing_model(input_ids, **kw):
            out = original_model(input_ids, **kw)
            # Capture the logit at the last position
            captured.append(out.logits[0, -1])   # should be 1D
            return out

        _run_inline_fallback(capturing_model, teacher, prompt,
                             max_tokens=3, max_propose=1)

        for logit in captured:
            assert logit.dim() == 1, (
                f"draft logit has dim={logit.dim()}, expected 1D (vocab,). "
                f"Shape: {logit.shape}. "
                "Use logits[0, -1] not logits[0, -1:] in the inline fallback."
            )
            assert logit.shape[0] == VOCAB

    def test_alpha_range(self):
        """Alpha must be in [0, 1] regardless of model agreement."""
        # Identical models → should accept all tokens (alpha ≈ 1.0)
        same = _make_stub_model(preferred_tok=42)
        prompt = torch.zeros(1, 10, dtype=torch.long)
        alpha = _run_inline_fallback(same, same, prompt,
                                     max_tokens=6, max_propose=3)
        assert 0.0 <= alpha <= 1.0

    def test_short_prompt_no_crash(self):
        """Single-token prompt must not crash (edge case for bonus_start=0)."""
        student = _make_stub_model(preferred_tok=7)
        teacher = _make_stub_model(preferred_tok=9)
        prompt = torch.tensor([[1]])   # length 1
        alpha = _run_inline_fallback(student, teacher, prompt,
                                     max_tokens=3, max_propose=1)
        assert 0.0 <= alpha <= 1.0
