# SpecDist Experiment Run Log

Every experiment run is recorded here. Rule: **if it wasn't logged, it didn't happen.**

To generate a summary row after a run completes:
```bash
python tools/run_summary.py --markdown --log_id 20260528-001
```

---

## Exp ID Format: `YYYYMMDD-NNN`

Increment NNN within a day (001, 002, …). Never reuse an ID.

## States

| Symbol | Meaning |
|--------|---------|
| ✅ | PASS / COMPLETE |
| ❌ | FAIL / regression |
| 🔄 | RUNNING |
| ⏳ | PLANNED |
| ~~strikethrough~~ | SUPERSEDED |

---

## Level 1 — Laptop Code-Check Runs

> Numbers from these runs are **not research results**. Only check: did it run?
> Teacher = 0.6B = same capacity as draft → distillation signal is near-zero.

To generate a summary:
```bash
python tools/run_summary.py --hw_tier laptop --markdown
```

| exp_id | date | smoke/full | steps | modes_ran | final_loss | errors | status | notes |
|--------|------|-----------|-------|-----------|-----------|--------|--------|-------|
| *(template)* | YYYYMMDD | smoke \| full | 10 \| 100 | alpha\|bv\|gbv\|… | 0.XXXX | 0 | ✅ \| ❌ | |

### Code Review Gate (before Level 2)

Before the first Level 2 run or after any code change:

- [ ] Level 1b (full laptop) run passes (all 6 eval modes, no errors)
- [ ] PR created and reviewed by: *(reviewer name)*
- [ ] Reviewer confirmed: loss function implementation correct vs paper
- [ ] Reviewer confirmed: verifier implementation correct vs GBV spec
- [ ] No open TODO/FIXME in modified files
- [ ] Checklist signed off: *(your name, date)*

---

## Level 2 — Colab Lite Trend Runs (1.7B teacher, ~25 min)

> These runs assess **direction** only. Effect sizes at this scale are not publishable.
> Gate: does at least one loss show consistent positive trend vs naive baseline?

To generate a summary:
```bash
python tools/run_summary.py --hw_tier colab_lite --markdown
```

| exp_id | date | loss | steps | seed | BE_specinfer_K3 | BE_gbv_K3 | Δ_naive | alpha | status | wandb | notes |
|--------|------|------|-------|------|----------------|----------|---------|-------|--------|-------|-------|
| BASELINE | — | naive | 0 | — | — | — | 0.000 | — | ✅ | — | canonical baseline |
| *(template)* | YYYYMMDD | kl\|ebe\|gbv | 300 | 42 | X.XXX | X.XXX | ±X.XXX | X.XX | ✅\|❌ | wandb.ai/… | |

### Code Review Gate (before Level 3)

- [ ] Level 2 run shows positive trend (Δ_naive > 0 on ≥2 verifiers)
- [ ] Second Level 2 seed confirms direction
- [ ] PR reviewed by: *(reviewer name)*
- [ ] Reviewer confirmed: hypothesis matches what the code actually tests
- [ ] Reviewer confirmed: no data leakage in eval (eval prompts not in train)
- [ ] Checklist signed off: *(your name, date)*

---

## Level 3 — Colab T4 Results Runs (4B teacher, ~2-4 h)

> These are **publishable numbers**. Record W&B run IDs immediately.
> Gate: effect > noise across 2 seeds, delta clearly stated.

To generate a summary:
```bash
python tools/run_summary.py --hw_tier colab --markdown
```

| exp_id | date | loss | steps | seed | BE_specinfer_K3 | BE_specinfer_K5 | BE_gbv_K3 | BE_gbv_K5 | Δ_best | alpha | status | wandb | notes |
|--------|------|------|-------|------|----------------|----------------|----------|----------|--------|-------|--------|-------|-------|
| BASELINE | — | naive | 0 | — | — | — | — | — | 0.000 | — | ✅ | — | canonical |
| *(template)* | YYYYMMDD | kl | 500 | 42 | X.XXX | X.XXX | X.XXX | X.XXX | +X.XXX | X.XX | ✅\|❌ | wandb.ai/… | |

### Code Review Gate (before Level 4)

- [ ] Level 3 results confirmed across ≥2 seeds
- [ ] Effect > 5% on at least one verifier × K combination
- [ ] PR reviewed by: *(reviewer name)*
- [ ] Reviewer confirmed: no bugs introduced since Level 2 review
- [ ] Hypothesis text finalized — no more code changes before A100 runs
- [ ] Ablation plan agreed (what to ablate on A100)
- [ ] Checklist signed off: *(your name, date)*

---

## Level 4 — A100 Paper Runs (8B teacher, ~2-3 h, Colab Pro/Pro+)

> Paper-quality results. Run only after Level 3 gate passes.
> Requires Colab Pro/Pro+ with A100 runtime.

To generate a summary:
```bash
python tools/run_summary.py --hw_tier a100 --markdown
```

| exp_id | date | loss | steps | seed | BE_specinfer_K3 | BE_gbv_K3 | BE_traversal_K3 | alpha | task_score | status | wandb | notes |
|--------|------|------|-------|------|----------------|----------|----------------|-------|-----------|--------|-------|-------|
| BASELINE | — | naive | 0 | — | — | — | — | — | — | ✅ | — | canonical |
| *(template)* | YYYYMMDD | kl | 2000 | 42 | X.XXX | X.XXX | X.XXX | X.XX | X.XX | ✅\|❌ | wandb.ai/… | |

---

## Baselines (pin these — never delete)

Record the untrained baseline once per hardware tier. All deltas are relative to these.

| hw_tier | date | verifier | K | BE_baseline | alpha_baseline | n_prompts | notes |
|---------|------|----------|---|-------------|---------------|-----------|-------|
| laptop | — | specinfer | 3 | — | — | 5 | fill after first Level 1b run |
| colab_lite | — | specinfer | 3 | — | — | 20 | fill after first Level 2 run |
| colab | — | specinfer | 3 | — | — | 20 | fill after first Level 3 run |
| a100 | — | specinfer | 3 | — | — | 50 | fill after first Level 4 run |

---

## W&B Run Index

Quick-reference index of all W&B runs. Keep this in sync with the tables above.

| exp_id | wandb_group | run_name | wandb_url | key_result |
|--------|------------|----------|-----------|-----------|
| *(template)* | colab-lite | 1.7B-T4-lite-kl_qwen_300steps | https://wandb.ai/… | BE=X.XXX |

---

## Failed / Superseded Runs

Log failed runs here — they are evidence. Do not delete.

| exp_id | date | hw_tier | what_failed | root_cause | fix_applied |
|--------|------|---------|------------|-----------|------------|
| *(template)* | YYYYMMDD | laptop | OOM at step 7 | 4-bit NF4 CPU RAM spike | switched to BF16 |

---

## Research Review Log

Research direction decisions. One entry per review session. Most recent first.
Rule: if it is not here, it was not decided — it is still open.

---

### Review — 2026-05-28 · Rahul (research lead), Krishnan R. (engineer)

**Status going in:** Phase 3 Level 2 colab_lite runs complete for KL, JSD, L1, EBE, GBV, BV, online.

#### Decisions

| Loss / Component | Decision | Rationale |
|---|---|---|
| **KL, JSD, L1** | ✅ Promote to A100 (Level 4) | Training curves look good; block efficiency reasonable; ready for paper runs |
| **EBE (offline)** | ⏸ Suspend — needs rework | Implementation uses single linear block, not the full tree acceptance loop. Needs to model GBV tree structure (K parallel paths × L depth). Do not run on A100 until rewritten. |
| **GBV, BV** | ⏸ Suspend — needs tuning | Block efficiency poor. Root cause unknown. Needs hyperparameter sweep (LR, kl_weight, LoRA rank) before A100 promotion. |
| **online_ebe** | ⏸ Suspend — bug confirmed | Replay buffer stores accepted+residual sequence. At rejected positions the stored residual token has q_teacher high and p_draft low → α=1 → EBE gradient zero everywhere. Incoherent KL corrupts model. Fix requires storing `draft_proposed_ids` separately in buffer. |
| **online_ebe_single** | ⏸ Suspend | Same root cause as online_ebe. |
| **online (KL)** | ✅ Can continue — unaffected | `forward_kl` online path is correct. `_kl_at_positions` operates on the right tokens (rejected positions, KL between student and teacher at those positions). |

#### Parking Lot (deferred, not abandoned)

| Idea | Owner | Reopen when |
|---|---|---|
| EBE full tree eval loop | Rahul | After KL/JSD/L1 A100 baseline established |
| online_ebe buffer redesign | Krishnan R. | After A100 baseline runs; ~2 days effort |
| GBV/BV hyperparameter sweep | Rahul | After KL A100 establishes comparison floor |
| EAGLE baseline | — | After publishable numbers exist |
| rev_kl assessment | — | Next research review |
