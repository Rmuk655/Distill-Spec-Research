# DistillSpec Research Project — Execution Plan (Revised)

**Revision date**: 2026-05-22  
**Submission target**: ICLR 2027 (deadline ~mid-September 2026, ~17 weeks away)  
**Status**: Weeks 1–4 infrastructure complete. Experiments running. Entering Week 5.

---

## Project Hypothesis

> A framework that jointly optimizes draft-model distillation (KL vs. EBE loss) and tree-based verification (GBV/traversal) improves speculative decoding block efficiency and robustness more than either approach alone — and more robustly than EAGLE-3 under distribution and sampling shifts.

The core novelty is replacing the forward KL training objective with a loss that directly targets the quantity the verifier actually measures — tokens accepted per target call.

---

## Current State (End of Week 4)

### Completed

| Component | Status | Location |
|---|---|---|
| DistillSpec trainer (teacher-sample, LoRA) | Done | `train_qwen3.py` |
| Forward KL loss | Done | `train_qwen3.py:233` |
| EBE token-level surrogate loss | Done | `train_qwen3.py:268` |
| GBV subprocess integration | Done | `run_all.py:run_be()` |
| Full eval pipeline (alpha + BE × 5 modes × K × T) | Done | `run_all.py` |
| Crash-safe master runner | Done | `pipeline.py` |
| SQLite results DB | Done | `results.db` |
| Flask viz dashboard (Key Results, BE vs K, Temperature, etc.) | Done | `viz_server.py` |
| Dataset fetch (gsm8k, humaneval, math500, mtbench, alpaca) | Done | `fetch_datasets.py` |
| Offline packaging for server / Colab | Done | `setup_download.py`, `SETUP.md` |
| Baseline BE numbers (untrained Qwen2.5-0.5B → Qwen3-0.6B) | Done | `results.db` |

**Baseline numbers (untrained draft, gsm8k_30)**:

| Mode | K=3 | K=5 |
|---|---|---|
| specinfer | 2.520 | 2.512 |
| gbv | 2.797 | 2.786 |
| traversal | 2.954 | 3.107 |

### In Progress / Immediately Next

| Task | Status |
|---|---|
| KL 1000 steps on gsm8k_train | Running (pipeline step `train_kl_gsm8k`) |
| EBE 1000 steps on gsm8k_train | Queued after KL |
| Multi-dataset eval (humaneval, math500, mtbench, alpaca) | Queued |
| Block-level analytic EBE loss | Not started — see Week 5 |

---

## EBE Loss — Two Versions (Critical for the Paper)

The current implementation is the **token-level surrogate** (implemented, running):

```
L_token = -Σ_t  min(1, q(t)/p(t)).detach()  ×  log p_draft(t)
```

This treats each token's gradient as independent. It is biased because it ignores the product structure of block-level acceptance.

The **block-level analytic EBE** is the real contribution. For a block of K tokens:

```
E[BE] = 1 + Σ_{k=1}^{K}  ∏_{i=1}^{k} α_i      where α_i = min(1, q(t_i)/p(t_i))
```

The gradient of the k-th product term for token j:

```
∂/∂θ [∏_{i=1}^{k} α_i] = Σ_{j=1}^{k}  (∂α_j/∂θ)  ×  ∏_{i≠j} α_i
```

Token j's gradient is amplified by the product of all other accepted tokens' α values in that block. Tokens deep in a successfully-accepted run receive stronger gradient — because they are the positions where extending the run matters most. This is categorically different from the independent-weight surrogate.

**Third option — REINFORCE / policy gradient** (tree-aware, requires verifier at training time):

```
∇_θ E[BE] = E_{block ~ p_draft} [ BE(block) × ∇_θ log p_draft(block) ]
```

Advantages: works for any verifier (gbv, traversal, spectr) without differentiating through the tree. Disadvantages: high variance, requires running the verifier at every training step (expensive — doubles VRAM).

**Recommendation for now**: implement block-level analytic EBE (Week 5). Save REINFORCE for server runs when VRAM budget allows.

| Loss | Gradient coupling | Verifier call at train | Accounts for tree | Bias |
|---|---|---|---|---|
| Token-level surrogate (current) | None | No | No | High |
| Block-level analytic EBE | Yes, via ∏α_i | No | No (linear K only) | Low |
| REINFORCE | Yes, via reward | Yes | Yes | None |

---

## Revised Weekly Plan

### Weeks 1–4 — COMPLETE

Deliverables achieved:
- PyTorch / transformer competence
- Speculative decoding baseline understood and running
- DistillSpec trainer implemented (train_qwen3.py)
- OSD codebase read and simplified (DistillTrainer → plain PyTorch loop, removed online mode, wrong_token_ids, buffer, HF Trainer dependency)
- GBV codebase integrated
- Full evaluation pipeline operational
- Baseline BE numbers established
- First KL/EBE training runs (200 steps, LR sweep) executed

---

### Week 5 — Block-Level EBE + First Real Results

**Goal**: determine whether distillation moves the needle; implement the correct EBE gradient.

**Krishnan R**:
- Wait for KL-1000 and EBE-1000 pipeline runs to complete
- Plot the 3×3 core matrix (Loss × Mode × K) from `viz_server.py`
- Implement block-level analytic EBE in `train_qwen3.py` as `--loss block_ebe`
  - Compute cumulative products of `α_i = min(1, q/p)` over K-token windows
  - Gradient flows through each `α_j` via `∂α_j/∂θ = (1/p(t_j)) × ∂p_θ/∂θ` when `q < p`
- Run block_ebe-1000 on gsm8k, compare against token_ebe-1000
- Deliverables: core result plots, block_ebe checkpoint, comparison table

**Rahul Thomas**:
- Review core 3×3 matrix results — is the EBE advantage on gbv/traversal visible?
- Confirm whether block-level gradient is the intended Phase 2 loss, or REINFORCE
- Decide: stay on token-level surrogate if results are already significant, or implement block-level
- Deliverables: direction confirmation, interpretation of early results

**PM**:
- Verify all runs are logged in results.db
- Flag any failed pipeline steps
- Maintain comparison matrix (baseline vs kl200 vs ebe200 vs kl1000 vs ebe1000)

---

### Week 6 — Server Runs + Qwen3-8B Target

**Goal**: reproduce Weeks 5 results at scale with the real target model.

**Krishnan R**:
- Transfer `OSD/` + HF cache to server (use `SETUP.md` + `setup_download.py --config server`)
- Re-run pipeline with `--config server` (Qwen3-0.6B draft → Qwen3-8B target)
- Key rows to add to the experiment matrix:
  - KL-1000-gsm8k + specinfer/gbv/traversal (Qwen3-8B target)
  - block_ebe-1000-gsm8k + specinfer/gbv/traversal (Qwen3-8B target)
- Laptop qualifier for moving to server: kl1000 shows BE improvement ≥ 0.1 over baseline on specinfer
- Deliverables: server BE numbers, comparison against laptop results

**Rahul Thomas**:
- Evaluate whether server results change the story
- Decide on temperature and K sweep scope for the paper
- Deliverables: go/no-go on current direction, paper scope confirmation

**Risk**: Qwen3-8B + Qwen3-0.6B at training time needs ~20GB VRAM. Training may require `--batch_size 1 --grad_accum 8`. Eval-only needs ~18GB (target only).

---

### Week 7 — EAGLE Baseline + Robustness Sweep

**Goal**: establish the paper's comparison baseline and show distribution robustness.

**Krishnan R**:
- Run EAGLE comparison using `eagle_bench.py` (already written):
  - Phase gen: extract hidden states from Qwen3-0.6B
  - Phase train: train EagleHead (1-layer transformer)
  - Phase eval: measure BE with EAGLE draft vs. our LoRA draft
- Temperature sweep: T = 0.4, 0.6, 0.8, 1.0, 1.2 on gsm8k
- Dataset sweep: gsm8k → humaneval → math500 → mtbench → alpaca (robustness)
- Deliverables: EAGLE BE numbers, temperature line plots, dataset robustness bar chart

**Rahul Thomas**:
- Confirm which EAGLE variant is the right comparison (EAGLE-1 or EAGLE-2, not EAGLE-3 since it uses Qwen3-specific hidden states)
- Identify the key robustness claim: does our method degrade less than EAGLE under T shifts?
- Deliverables: competition analysis, paper positioning

**Key claim to test**: EAGLE specializes on one model family. Our method is distribution-agnostic. Under dataset shift (gsm8k → alpaca) and temperature shift (0.6 → 1.0), our BE loss degrades less.

---

### Week 8 — Ablations + Variance Analysis

**Goal**: make the evidence defensible at review.

**Krishnan R**:
- Run 3 seeds for each key configuration (baseline, kl1000, block_ebe1000)
- Report mean ± std on all BE numbers
- Ablation table:
  - block_ebe-1000 vs. token_ebe-1000 (does the gradient coupling matter?)
  - KL + block_ebe hybrid (α-weighted sum) — 3 α values
  - LoRA rank sweep (r=4, 8, 16)
- AdaSPEC ablation if time permits: train reference model, implement mask, compare
- Deliverables: ablation table, variance analysis, confidence intervals

**Rahul Thomas**:
- Review claim strength: is block_ebe strictly better than KL on tree verifiers?
- Identify any confounders (tokenization, prompt distribution, temperature)
- Deliverables: claim validation, evidence review

---

### Week 9 — Paper Writing Begins

**Goal**: transform results into a paper narrative.

**Krishnan R**:
- Write implementation + experiment sections
- Produce all final figures (use `viz_server.py` charts as drafts, clean for paper):
  - Figure 1: Loss × Mode heatmap (3×3 core matrix)
  - Figure 2: BE Gain % over baseline
  - Figure 3: BE vs Temperature (line, not bar)
  - Figure 4: Dataset robustness
  - Figure 5: Ablations table
- Write appendix: hyperparameter details, reproducibility

**Rahul Thomas**:
- Introduction, method framing, related work
- Novelty positioning: "we show that matching the training objective to the verifier's acceptance structure — via block-level EBE — provides gains that KL cannot, especially for tree-based verification"
- Venue recommendation: ICLR Workshop on Efficient LLM Systems, or main track if results strong
- Deliverables: paper draft sections 1–3

---

### Weeks 10–11 — Paper Polish + Submission Preparation

**Goal**: submission-ready paper.

**Krishnan R**:
- Rerun any critical experiments flagged in Week 9 review
- Clean the codebase (`OSD/` directory, remove dead code from `distill/` experiments)
- Verify all results in DB match paper tables
- Reproducibility checklist: `requirements.txt`, `SETUP.md`, `pipeline.py --dry_run`

**Rahul Thomas**:
- Final claim validation
- Related work completeness check (Medusa, EAGLE, Sequoia, SpecTr, Traversal Verification)
- Reviewer-proofing the novelty argument

**PM**:
- Submission deadline tracking
- Author list, acknowledgements, supplementary material checklist
- ArXiv preprint coordination

---

## Experiment Matrix (Paper Target)

### Core 3×3 (Laptop — Qwen2.5-0.5B → Qwen3-0.6B)

| Draft training | specinfer K=3 | gbv K=3 | traversal K=3 |
|---|---|---|---|
| Untrained baseline | 2.520 | 2.797 | 2.954 |
| KL-1000-gsm8k | TBD | TBD | TBD |
| block_ebe-1000-gsm8k | TBD | TBD | TBD |

**Expected story**: block_ebe > KL on gbv/traversal. KL ≈ block_ebe on specinfer (token independence is fine for linear chain). The gap widens with K.

### Server Matrix (Qwen3-0.6B → Qwen3-8B)

Same 3×3 plus EAGLE comparison row. This is the paper's headline result.

### Robustness (secondary contribution)

| Model | gsm8k | humaneval | math500 | mtbench | alpaca |
|---|---|---|---|---|---|
| baseline | | | | | |
| KL-1000 | | | | | |
| block_ebe-1000 | | | | | |
| EAGLE | | | | | |

Temperature sensitivity: run the above at T=0.6 and T=1.0. Show BE degrades less for block_ebe than EAGLE under distribution shift.

---

## Decision Log

| Decision | Resolution | Owner | Date |
|---|---|---|---|
| KL direction | Forward KL (mode-covering, directly min. acceptance gap) | Rahul Thomas | Week 2 |
| Draft model | Qwen2.5-0.5B (pretrained, shared vocab with Qwen3) | Rahul Thomas | Week 2 |
| wrong_token_ids | Not used (DistillSpec trains on all positions) | Rahul Thomas | Week 2 |
| Sample source | Teacher-only (matches target's inference distribution) | Rahul Thomas | Week 2 |
| HF Trainer | Removed (plain PyTorch loop — simpler, sufficient) | Krishnan R | Week 3 |
| AdaSPEC | Not in baseline; potential ablation in Week 8 | Rahul Thomas | Week 3 |
| Online mode | Excluded from Phase 1; revisit at Week 5 start | Rahul Thomas | Week 3 |
| EBE loss type | Token-level surrogate now; block-level analytic in Week 5 | Rahul Thomas | Pending |
| REINFORCE | Reserve for server; too expensive on laptop | Rahul Thomas | Pending |

---

## Scope Boundaries (Enforced)

**IN scope**:
- Qwen2.5-0.5B draft → Qwen3-0.6B (laptop) / Qwen3-8B (server) target
- LoRA-only fine-tuning
- KL and EBE (token-level, block-level analytic) losses
- Verifiers: specinfer, gbv, traversal (from GBV codebase — do not reimplement)
- Datasets: gsm8k (primary), humaneval, math500, mtbench, alpaca (robustness)
- EAGLE as comparison baseline (from `eagle_bench.py`)
- AdaSPEC as optional Week 8 ablation

**OUT of scope** (do not pursue):
- Online distillation (during serving) — revisit at Week 5 if results are weak
- REINFORCE / policy gradient — requires verifier at every training step
- Custom CUDA kernels, hardware optimization
- Multi-device, MoE, multimodal
- Medusa / DFlash / FastEagle replication
- Training from scratch

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Token-level EBE shows no advantage over KL | Medium | High | Implement block-level analytic EBE; if still weak, pivot to robustness claim alone |
| Qwen3-8B OOM during training on server | Medium | Medium | batch=1, grad_accum=8, bfloat16; check `SETUP.md` |
| EAGLE beats our method on all metrics | Low | High | Robustness claim is sufficient for workshop; block_ebe may still win on tree verifiers |
| KL-1000 pipeline run stalls again | Medium | Low | `pipeline.py` offline fix applied; monitor `pipeline_state_laptop.json` |
| ICLR deadline moves earlier | Low | High | Target submission-ready draft by Week 10 with 1-week buffer |

---

## Key Files

```
OSD/
  train_qwen3.py      ← Training (forward_kl, token_ebe; add block_ebe in Week 5)
  run_all.py          ← Evaluation (alpha + BE via GBV subprocess)
  pipeline.py         ← Crash-safe runner (--config laptop/server)
  viz_server.py       ← Results dashboard
  results_db.py       ← SQLite schema
  eagle_bench.py      ← EAGLE comparison pipeline (gen→train→eval)
  DESIGN.md           ← Technical reference (OSD vs DistillSpec vs GBV)
  SETUP.md            ← Server/Colab setup guide

GBV/                  ← Multi-path verifier codebase (used as subprocess, do not modify)
  main.py             ← Entry: --mode specinfer|bv|gbv|traversal|spectr
```

---

## Success Criteria

**Minimum (workshop-level)**:
- Reproducible BE improvement ≥ 0.1 on any verifier mode after distillation
- block_ebe ≥ KL on gbv or traversal at K=3 or K=5
- Results hold across at least 2 datasets and 2 temperatures

**Strong (conference-level)**:
- block_ebe + gbv/traversal beats EAGLE on robustness (dataset/temperature shifts)
- Results hold on Qwen3-8B target
- Full 3×3 matrix with confidence intervals across 3 seeds
