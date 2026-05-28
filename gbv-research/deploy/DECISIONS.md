# SpecDist Research Decision Log

Every research direction decision is recorded here.
Rule: **if it wasn't decided here, it wasn't decided — it's still open.**

Format: one section per research review. Most recent first.

---

## Research Review — 2026-05-28

**Attendees:** Rahul (research lead), Ram (PM/engineer)

### Decisions

| # | Decision | Owner | Rationale |
|---|----------|-------|-----------|
| 1 | **KL, JSD, L1 → promote to Level 4 (A100)** | Rahul | Training curves look good; block efficiency is reasonable; ready for paper-quality runs |
| 2 | **EBE offline → suspend, needs full tree eval loop** | Rahul | Current implementation uses a single intermediate check, not the full tree acceptance loop. Claude-generated logic is incorrect. Do not run on A100 until reworked. |
| 3 | **GBV/BV → needs tuning, do not promote yet** | Rahul | Block efficiency currently poor; root cause unknown. Needs hyperparameter tuning before Level 4. |
| 4 | **online_ebe → suspend, root cause bug confirmed** | Ram/Engineer | EBE gradient is zero everywhere due to buffer design flaw (see §2 below). Causing PPL rise and BE collapse. Suspend until buffer redesign. |
| 5 | **online_ebe_single → suspend (same root cause)** | Ram/Engineer | Same buffer flaw as online_ebe; single-token variant has the same zero-gradient problem at rejected positions. |

---

## §2 — online_ebe Bug Root Cause (2026-05-28)

**Symptom:** online_ebe runs show PPL and block efficiency both moving to extreme
low values (far left on scatter plots). BE approaches 1.0 (no speculative benefit).

**Root cause (confirmed):**

`speculative_step()` stores the *accepted + teacher-resampled residual* sequence
in the replay buffer. At each rejected position, the stored token is the
**teacher-resampled residual token** (sampled from `(p_teacher - p_draft)+`),
NOT the draft's actual proposal.

When `_ebe_online` computes `α = min(1, q_teacher(token) / p_draft(token))`:

- For the **residual token** at a rejected position:
  - `q_teacher(t_res)` is HIGH — teacher chose it from its adjusted distribution
  - `p_draft(t_res)` is LOW — draft didn't like it (that's why it's in the residual)
  - → `log_q − log_p >> 0` → clamped to 0 → **α = 1.0** → zero EBE gradient

- For **accepted tokens**: α = 1.0 by definition → zero EBE gradient

**Result:** EBE gradient is zero everywhere. Only the KL regulariser fires, but
it's operating on the wrong tokens (residual-resampled, not draft-proposed).
The incoherent KL update corrupts the draft model → PPL rises → BE collapses.

**Fix required:**
The `ReplayBuffer` must separately store `draft_proposed_ids: Tensor[T]` —
what the draft actually proposed at each position before accept/reject.
`_ebe_online` then uses `draft_proposed_ids[pos]` (not the stored sequence)
at rejected positions, so `p_draft` is high and `q_teacher` can be genuinely
lower → real EBE gradient.

**Effort:** ~2 days. Not on the critical path for A100 runs (KL/JSD/L1 are ready).

**Code pointers:**
- `algorithms/online_serve.py:_ebe_online` — BUG comment at top of function
- `algorithms/online_serve.py:speculative_step` — where draft tokens are discarded
- `algorithms/online_serve.py:ReplayBuffer.as_batch` — where buffer is assembled

---

## §3 — EBE Offline: Full Tree Eval Loop Required (2026-05-28)

**Decision:** The current offline EBE implementation computes loss over a single
speculative block (fixed-length chunk of tokens). It does not implement the full
GBV tree acceptance loop.

**Rahul's assessment:** "Claude-generated EBE logic is not good. There is no
intermediary like single EBE — it needs to be full tree eval loop."

**Meaning:** The EBE loss should model the tree-structured speculative decoding
process (K parallel paths of depth L), not a linear sequence. The acceptance
probability for a block should account for:
1. The tree structure (multiple paths, GBV/BV verifier selects the best path)
2. The correct stopping condition (first rejection along the selected path)
3. The block efficiency formula E[τ+1] over the tree

**Current state:** `algorithms/distillspec_gbv/losses/ebe.py` computes cumprod
over linear blocks. This is a simplification that loses the tree structure.

**Action:** EBE is suspended from A100 runs. Rework requires implementing the
full GBV tree acceptance loop inside the EBE loss computation. Separate task.

---

## §4 — GBV/BV Tuning (2026-05-28)

**Decision:** GBV and BV block efficiency are currently poor. Rahul says this needs
tuning before A100 promotion.

**Suspected causes (not yet confirmed):**
- Learning rate may be too high/low for the GBV-specific loss signal
- KL regularisation weight (`kl_weight`) may be dominating the GBV objective
- LoRA rank may be insufficient to express the required distribution shift

**Action:** Do NOT include gbv/bv in A100 runs until at least one Level 2 or
Level 3 run shows positive delta vs naive baseline on the same verifier.

---

## §5 — Level 4 (A100) Promotion Decision (2026-05-28)

**Ready for A100:**
- [ ] kl (forward KL distillation)
- [ ] jsd (Jensen-Shannon divergence)
- [ ] l1 (L1 token distribution matching)

**Not ready (suspended):**
- [ ] ebe — needs full tree eval loop rework
- [ ] gbv — needs tuning
- [ ] bv — needs tuning
- [ ] online_ebe — buffer redesign required
- [ ] online_ebe_single — same as online_ebe
- [ ] rev_kl — not reviewed yet; pending

**Gate before A100 run:** Level 3 gate checklist in `deploy/PROGRESSION.md` must
be complete for all three promoted losses.

---

## Parking Lot (ideas deferred, not discarded)

| Idea | Deferred by | Why deferred | When to revisit |
|------|------------|--------------|-----------------|
| EBE with full tree loop | Rahul, 2026-05-28 | Requires significant reimplementation | After KL/JSD/L1 A100 runs produce baseline numbers |
| online_ebe buffer redesign | Ram, 2026-05-28 | 2-day effort, not on critical path | After A100 baseline runs |
| GBV/BV hyperparameter sweep | Rahul, 2026-05-28 | Need baseline first | After KL A100 establishes floor |
| EAGLE baseline comparison | — | Not started | After we have publishable numbers |
| Rev KL assessment | — | Not reviewed | Next research review |
