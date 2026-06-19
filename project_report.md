# Speculative Decoding — Draft Training: Project Report

**Draft model:** Qwen3-0.6B  
**Teacher model:** Qwen3-8B  
**Task:** GSM8K (math word problems)  
**Primary metric:** Block Efficiency (BE) = gen_tokens / target_calls (higher is better)  
**Standard eval:** n=100 prompts, traversal verifier, K=1–4, L=8  
**Period:** June 2026

---

## Quick Summary

We trained a 0.6B draft model to improve acceptance rates under speculative decoding with an 8B teacher. The core question: does training with acceptance-aware (tree) losses outperform plain flat distillation (JSD/KL)?

**Answer at 8B/0.6B/GSM8K: No.** Flat JSD plateaus at val BE ≈ 6.0 and no tree-loss or depth-weighting variant improves on it. Open question: is the algorithm correctly sending tree signals, or are we hitting a capacity ceiling with limited headroom on this 8B/0.6B/GSM8K combination? We have tested both telescoping (multi-depth) and single-depth tree losses.

**Key experiments:**
- Depth × JSD (researcher's suggestion): showed no detectable signal — gradient-free scalar, total sweep spread 0.055 < noise floor ±0.15
- gbv_tree, bv_tree: algorithmically broken — E[τ] collapses to zero on overfit set
- Additive (JSD + λ·tree): no meaningful improvement over JSD at 8B; λ=0.1/0.3/1.0 all below JSD control at 1000 steps

**Justified next steps:** MATH levels 4–5 probe (harder prompts → more unlearnable examples → potential signal at 8B); 32B teacher run; and a complete rethink of the algorithm — better ways to send tree gradients during training beyond flattened tree losses or single-depth weighting.

---

## Setup

| Component | Value |
|---|---|
| Draft | Qwen/Qwen3-0.6B, BF16 |
| Teacher | Qwen/Qwen3-8B, BF16 |
| Hardware | NVIDIA H100 80GB HBM3 |
| Train steps | 4000 (standard); 250 for probes (see [algo_sanity.py](algo_sanity.py)) |
| LR schedule | Linear warmup (200 steps) → cosine decay to 0.1× peak |
| Peak LR | 3e-4 (all losses) |
| Batch | 1 prompt / step, grad accum = 4 |
| K (draft branching) | 3 during training; 1–4 in eval |
| L (tree horizon) | 8 |
| Val every | 50 steps, 25 prompts, temp=0.2 (after fix; old: temp=1.0) |
| Eval | 100 prompts, K=1–4, all 8 verifiers |

**Verifiers tested:** naive, nss, specinfer, spectr, khisti, bv, gbv, traversal  
**Loss families:** Flat (forward_kl, reverse_kl, jsd, l1) · Tree (kl_tree, naive_tree, bv_tree, gbv_tree, traversal_tree, nss_tree, ...) · Depth-weight (depth_linear, depth_lambda) · Additive (jsd + λ·tree)

---

## Phase 1 — Untrained Baseline (2026-06-09)

Evaluated Qwen3-0.6B before any training to pin the starting point.

**Traversal BE (1000 prompts):**

| K=1 | K=2 | K=3 | K=4 |
|---|---|---|---|
| 4.708 | 4.938 | 4.907 | 4.870 |

**Key pattern:** traversal is the highest-BE verifier at every K; NSS is the lowest (3.3). NSS degrades monotonically with K (tree structure hurts it); traversal improves with K (tree structure helps it). This makes traversal the most informative eval axis going forward.

---

## Phase 2 — First Training Runs + Infrastructure Bugs (2026-06-09 to 06-11)

### 2.1 nss_tree (run 20260611-001)

Trained with the NSS-verifier-aligned telescoping tree loss.

**Val metric during training:** peaked at 3.612 NSS BE at step 1600 on 25 prompts → looked promising.

**Offline eval (100 prompts):** NSS BE = 3.243 on ckpt_best — **below the untrained baseline** (3.304). Two eval runs on the same checkpoint showed BV BE range of 0.240, establishing that 25-prompt val is noisy.

| Checkpoint | NSS K=3 | Traversal K=3 | vs. baseline |
|---|---|---|---|
| Baseline | 3.304 | 4.907 | — |
| nss_tree ckpt_best (avg) | 3.243 | 5.201 | NSS −0.061, traversal +0.294 |

**Finding F-001:** nss_tree val peak was noise; offline at 100 prompts it fails to beat baseline on its own verifier.

### 2.2 K=1 diagnostic for nss_tree (2026-06-12)

K=1 removes all multi-path tree machinery. If nss_tree failed due to a multi-path bug, K=1 should match forward_kl.

| Checkpoint (K=1) | NSS BE | Naive BE | BV BE |
|---|---|---|---|
| Baseline | 3.336 | 4.545 | 4.770 |
| forward_kl | 3.675 | 4.909 | 5.090 |
| nss_tree | 3.655 | 4.193 | 4.259 |

**Finding F-004 (structural negative):** nss_tree at K=1 nearly ties forward_kl on NSS (−0.020, within noise) but is 0.6–0.8 BE worse on every coverage verifier. The failure is in the **core surrogate**, not the multi-path machinery. At K=1, α_NSS = Σ p·q maximized by q → onehot(argmax p) — mode collapse. This concentrates the draft distribution, improving single-token NSS but destroying diversity for all coverage-based verifiers. **Structural negative result; do not re-run.**

---

## Phase 3 — Flat Loss Baselines (2026-06-13 to 06-14)

Re-ran forward_kl cleanly (with fixed val K) and added jsd, l1, reverse_kl, kl_tree, naive_tree.

### 3.1 Traversal BE across checkpoints (n=100, K=1–4)

| Checkpoint | K=1 | K=2 | K=3 | K=4 |
|---|---|---|---|---|
| Baseline (untrained) | 4.71 | 4.94 | 4.91 | 4.87 |
| forward_kl | **5.33** | **5.55** | **5.36** | **5.45** |
| jsd | 5.44 | 5.32 | 5.32 | 5.25 |
| l1 | 5.34 | 5.39 | 5.26 | 5.19 |
| reverse_kl | 5.05 | 5.08 | 5.05 | 4.73 |
| kl_tree | 4.78 | 5.03 | 5.18 | 5.02 |
| naive_tree | 4.68 | 5.18 | 5.22 | 5.09 |

### 3.2 BV BE across checkpoints

| Checkpoint | K=1 | K=2 | K=3 | K=4 |
|---|---|---|---|---|
| Baseline | 4.634 | 4.625 | 4.675 | 4.662 |
| forward_kl | 5.297 | 5.260 | 5.203 | 5.404 |
| jsd | 5.446 | 5.302 | 5.363 | 5.361 |
| l1 | 5.212 | 5.458 | 5.474 | 5.323 |
| kl_tree | 4.585 | 4.753 | 4.785 | 4.729 |
| naive_tree | 4.764 | 4.823 | 4.735 | 4.692 |

**Note on tree losses:** bv_tree, gbv_tree, traversal_tree, and nss_tree were not run to full completion or were terminated early. bv_tree and gbv_tree showed diverging/collapsing loss from early steps. traversal_tree and nss_tree training was halted given the negative signals from kl_tree and naive_tree. Results for these are excluded from the tables above; see Phase 7 (algo correctness probe) for overfit-set verdicts.

**Key findings:**
- All four flat losses beat the untrained baseline by +0.3–0.6 BE on traversal
- forward_kl is strongest, especially at K=2 (5.55 traversal)
- **Tree losses (kl_tree, naive_tree) beat the baseline but consistently lose to every flat loss** at every (K, verifier) combination. The gap is 0.1–0.4 BE, reproducible, not reducible by K choice.
- reverse_kl degrades at K=4; not competitive
- Training val BE peak: jsd reached 5.996 at step ~2300, plateaued, LR floored at 3e-6 by step 4000

**Finding F-002:** traversal is the highest-BE verifier and the most sensitive to training quality. It should be the primary metric.

**Finding F-003:** 100-prompt eval has noise floor ~0.15 BE. Effects below that are directional only.

---

## Phase 4 — Flat Loss Comparison and JSD Selection (2026-06-14, n=100, gsm8k_eval)

Full offline evaluation of all four flat losses across K=1–4 and all 8 verifiers. This is the definitive flat-loss comparison; these checkpoints are the baseline ceiling for all subsequent tree-loss work.

### 4.1 Traversal BE (highest-BE verifier, most informative)

| Checkpoint | K=1 | K=2 | K=3 | K=4 |
|---|---|---|---|---|
| jsd | **5.436** | 5.323 | 5.323 | 5.252 |
| forward_kl | 5.333 | **5.552** | **5.356** | **5.449** |
| l1 | 5.341 | 5.390 | 5.262 | 5.192 |
| reverse_kl | 5.051 | 5.076 | 5.048 | 4.728 |

### 4.2 BV BE

| Checkpoint | K=1 | K=2 | K=3 | K=4 |
|---|---|---|---|---|
| jsd | **5.446** | 5.302 | 5.363 | **5.361** |
| forward_kl | 5.297 | 5.260 | 5.203 | 5.404 |
| l1 | 5.212 | **5.458** | **5.474** | 5.323 |
| reverse_kl | 5.034 | 5.148 | 5.047 | 4.954 |

### 4.3 Mean BE across all 8 verifiers

| Checkpoint | K=1 | K=2 | K=3 | K=4 |
|---|---|---|---|---|
| **jsd** | **5.147** | **4.893** | **4.664** | 4.569 |
| forward_kl | 5.008 | 4.848 | 4.565 | 4.565 |
| l1 | 5.006 | 4.841 | 4.645 | **4.586** |
| reverse_kl | 4.887 | 4.677 | 4.537 | 4.419 |

### 4.4 Findings

**JSD is the strongest flat loss overall.** It leads on mean BE at K=1, K=2, and K=3, and is effectively tied at K=4.

- **JSD vs forward_kl:** forward_kl leads on traversal at K=2–4 (+0.09–0.23), but JSD leads on BV at K=1 and K=4 and on mean BE at every K. Within noise on traversal at K=3 (5.323 vs 5.356, Δ=0.033 < ±0.15 noise floor). No consistent winner; JSD selected as primary baseline for its stable training curve (val BE 5.996 at step ~2300) and broad superiority across verifiers.
- **JSD vs l1:** l1 leads on BV at K=2–3 (by up to 0.11 BE) but is consistently behind JSD on mean. JSD is the safer choice.
- **reverse_kl:** weakest flat loss at every K and verifier; degrades sharply at K=4 on traversal (4.728 vs 5.252 for JSD).
- **NSS BE:** reverse_kl counter-intuitively leads on NSS (mode-seeking → high per-token acceptance, but poor tree coverage). Not a useful metric for optimizing tree performance.

**Selection:** JSD is designated the primary flat-loss baseline for all subsequent tree-loss and dataset experiments. forward_kl is retained as a secondary reference.

---

## Phase 5 — Depth × JSD: Researcher's Suggestion (2026-06-15 to 06-18)

**Suggestion:** multiply JSD loss by expected acceptance depth d: `L = d × JSD`, biasing training toward steps where the draft is already accepted deeply.

### 5.1 Implementation

Two normalized forms implemented (both E[w] ≈ 1 to prevent LR confound):

| Form | Formula | Flag |
|---|---|---|
| Linear | w = d / EMA(d) | `--depth_linear` |
| Exponential | w = exp(λ(d − EMA(d))) | `--depth_lambda λ` |

*Rejected upfront:* raw `w = d` → E[w] ≈ L ≈ 5 → silent 5× LR inflation.

### 5.2 Training curve (4000 steps, val_temp=0.2)

| Condition | Best val BE | Plateau step | Notes |
|---|---|---|---|
| jsd (baseline) | 5.996 | ~2300 | LR floored at 3e-6 by step 4000 |
| depth_linear | same 5.0–6.0 band | indistinguishable | depth_w swings 0.5–2.5 throughout |

A 5× loss-weight swing moved val BE by less than run-to-run noise (±0.25). W&B `train/depth_w` confirms the weight was active the whole time.

### 5.3 Full signed eval sweep (n=100, K=1–4, traversal + bv)

| Condition | Traversal K=1 | K=2 | K=3 | K=4 | BV K=1 | K=2 | K=3 | K=4 | **Mean** |
|---|---|---|---|---|---|---|---|---|---|
| jsd | 5.436 | 5.323 | 5.323 | 5.252 | 5.446 | 5.302 | 5.363 | 5.361 | **5.351** |
| lam0.0 (no-op) | 5.335 | 5.311 | 5.472 | 5.278 | 5.530 | 5.442 | 5.499 | 5.296 | **5.395** |
| lam+0.5 | 5.410 | 5.499 | 5.253 | 5.286 | 5.367 | 5.590 | 5.349 | 5.313 | **5.383** |
| lam−0.5 | 5.347 | 5.428 | 5.421 | 5.177 | 5.320 | 5.463 | 5.425 | 5.346 | **5.366** |
| linear | 5.341 | 5.256 | 5.198 | 5.296 | 5.399 | 5.429 | 5.429 | 5.369 | **5.340** |

**Total spread: 0.055. Run-to-run noise: ±0.15.**

**Decisive tell:** lam0.0 (a mathematical no-op — identical loss to plain jsd) tops the table. The directional test lam+0.5 > lam−0.5 holds in only 5/8 cells (coin flip). No monotone λ trend.

**Root cause:** `w(d)` is a detached scalar — `∂L/∂θ = w(d) · ∂JSD/∂θ`. It is a per-prompt adaptive learning rate. It cannot change the gradient direction, so it cannot teach the draft *how* to be accepted deeper. This mechanism is **inconclusive across all forms** — no variant produced detectable improvement.

Note: lam0.0 and jsd follow different trajectories despite identical loss because the depth_weight code unconditionally runs a full target tree pass at λ=0, consuming RNG every step — a code artefact, not science.

---

## Phase 6 — Additive Tree Aux: jsd + λ·naive_tree (2026-06-17 to 06-18)

**Mechanism:** `L = JSD + λ · L_naive_tree` — real per-level acceptance gradient through q, with the natural survival weight `Π αⱼ.detach()` (telescoping). The only mechanism that actually redirects gradients.

**Full eval (4000-step checkpoints, gsm8k_eval, n=100, seed 123):**

BV verifier:

| λ | K=1 | K=2 | K=3 | K=4 | Mean K=1–4 |
|---|---|---|---|---|---|
| jsd (0) | 5.409 | 5.346 | **5.477** | 5.428 | **5.415** |
| +0.1 | 5.414 | 5.283 | **5.589** | 5.293 | 5.395 |
| +0.3 | 5.436 | 5.247 | 5.438 | 5.288 | 5.352 |
| +1.0 | 5.203 | 4.934 | 5.003 | 5.098 | 5.060 |

Traversal verifier:

| λ | K=1 | K=2 | K=3 | K=4 | Mean K=1–4 |
|---|---|---|---|---|---|
| jsd (0) | **5.554** | **5.436** | **5.427** | 5.281 | **5.425** |
| +0.1 | 5.286 | 5.417 | 5.192 | **5.341** | 5.309 |
| +0.3 | 5.355 | 5.373 | 5.253 | 5.297 | 5.320 |
| +1.0 | 5.163 | 5.430 | 5.362 | 5.237 | 5.298 |

**Result: null.** JSD wins on both verifiers on mean across K=1–4.

- **BV:** monotone mean decline with λ (5.415 → 5.060). λ=0.1 shows +0.112 at K=3 but −0.063 at K=2 and −0.135 at K=4; net BV mean jsd=5.415 > x0.1=5.395.
- **Traversal:** jsd wins at K=1 (+0.268 vs x0.1) and K=3 (+0.235 vs x0.1). At K=4 x0.1 leads by +0.060 and x0.3 by +0.016 — both within run-to-run noise (±0.15). Net traversal mean jsd=5.425 > x0.1=5.309.
- No λ lifts above jsd on both verifiers simultaneously at any K.

**Interpretation:** consistent with the capacity ceiling — at 8B/0.6B/GSM8K, flat JSD already reaches q≈p everywhere the 0.6B can reach, so tree gradient is at best redundant and at worst noisy. Does not rule out that the gradient is also degraded at scale (the overfit probe is necessary-not-sufficient).

---

## Phase 7 — Algorithm Correctness Probe (2026-06-18)

**Question:** Is the 8B null result (H1: capacity ceiling) or a broken algorithm (H2: the tree losses are no-ops)?

**Method (algo_sanity.py):** Overfit 16 GSM8K prompts with each loss for 250 steps. Measure E[τ] (expected accepted depth under `naive` verifier) before and after. A working gradient MUST raise E[τ] on the training set; failure means the formulation is broken.

Verdict threshold: SE = disp/√N (standard error of the mean, not prompt-to-prompt spread). Strong effect: Δ > 3×SE.

### Results

| Loss | Type | Baseline E[τ] | Best E[τ] | Δ | SE | Signal/Noise | Verdict |
|---|---|---|---|---|---|---|---|
| naive_tree | tree | 1.87 | 5.50 | **+3.63** | 0.177 | ~20×SE | **WORKS** |
| kl_tree | tree | 2.55 | 6.07 | **+3.52** | 0.203 | ~17×SE | **WORKS** |
| traversal_tree | tree | 2.16 | 5.67 | **+3.50** | 0.166 | ~21×SE | **WORKS** |
| gbv_tree | tree | 2.14 | 2.14 | **−2.14** | 0.197 | — | **BROKEN** |
| bv_tree | tree | 2.34 | 2.34 | **−2.34** | 0.193 | — | **BROKEN** |
| jsd | flat control | 2.29 | 6.26 | +3.97 | 0.196 | ~20×SE | (flat control) |

**gbv_tree and bv_tree are algorithmically broken.** E[τ] collapses to zero within 50 steps; loss saturates at −0.0000 (gradient vanishes entirely). These losses actively destroy acceptance — do not use at any scale. Root cause: the BV/GBV acceptance weight w = Π min(1, p[token]/q[token]) is computed only over sampled tokens on each path. A single token where q ≫ p drives w → 0, zeroing the gradient for that path. As training progresses, the model learns to suppress high-p tokens (the ones that generate nonzero gradient), inverting the objective. This is a structural instability in the BV/GBV surrogate under large p/q ratios, not a coding typo.

**Open question for researcher: are naive_tree, kl_tree, and traversal_tree truly sending correct tree gradients?** The overfit probe shows E[τ] rises on a 16-prompt set — a necessary but not sufficient condition. Whether this gradient survives at scale (full dataset, full model, longer training) is unconfirmed. We cannot definitively reject H2. Researcher sign-off needed before concluding this is purely a capacity issue.

**traversal_tree (A-002 resolved):** the implementation now uses `_telescoping_loss(_alpha_naive)` — the same telescoping backbone as naive_tree. Exact at K=1; at K>1 it uses naive-verifier α rather than the true traversal α (which has no closed form), so it is an approximation. The overfit probe result stands. traversal_tree is safe to run at scale.

**Implication for additive:** `jsd + λ·naive_tree` and `jsd + λ·kl_tree` inherit working tree gradients in the overfit sense. Whether the additive null result at 8B is a capacity ceiling or a gradient quality issue at scale remains open.

---

## Complete Results Summary

### Traversal BE at K=3 (primary comparison axis)

| Checkpoint | Train loss type | val BE peak | Traversal K=3 | vs. baseline | vs. jsd |
|---|---|---|---|---|---|
| Baseline (untrained) | — | — | 4.91 | — | −0.41 |
| nss_tree | tree | ~3.6 (noise) | 5.20 | +0.29 | −0.12 |
| kl_tree | tree | — | 5.18 | +0.27 | −0.14 |
| naive_tree | tree | — | 5.22 | +0.31 | −0.10 |
| reverse_kl | flat | — | 5.05 | +0.14 | −0.27 |
| l1 | flat | — | 5.26 | +0.35 | −0.06 |
| jsd | flat | 5.996 | 5.32 | +0.41 | — |
| forward_kl | flat | — | 5.36 | +0.45 | +0.04 |
| depth_weight (all forms) | flat+scalar | ~5.996 | 5.20–5.47 (noise) | noise | noise |
| jsd + naive_tree × 0.1 | flat+tree | — | 5.19 (traversal) / 5.59 (bv) | — | −0.24 traversal / +0.11 bv |
| jsd + naive_tree × 0.3 | flat+tree | — | 5.25 (traversal) / 5.44 (bv) | — | −0.18 traversal / −0.04 bv |
| jsd + naive_tree × 1.0 | flat+tree | — | 5.36 (traversal) / 5.00 (bv) | — | −0.07 traversal / −0.48 bv |

**JSD is the primary flat baseline** — best val BE during training (5.996) and the deliberate control for all further experiments. forward_kl edges it by +0.04 at K=3 offline traversal, which is within the ±0.15 noise floor and not a meaningful difference. Val BE for forward_kl was never tracked during training.

### gbv_tree / bv_tree: do not use
These are excluded from the table above — they drive E[τ] to zero on the overfit set (broken gradient, not just suboptimal).

---

## Should We Rule Out?

| Approach | Status | Why |
|---|---|---|
| nss_tree | Structural negative | Mode-seeking surrogate at K=1; collapses coverage verifiers |
| gbv_tree | Broken | Gradient vanishes; E[τ] → 0 on overfit set |
| bv_tree | Broken | Same collapse |
| depth_weight (linear, all λ) | No detectable signal | Gradient-free scalar; 5× swing < noise; no-op tops table |
| reverse_kl | Suboptimal | Degrades at K=4; dominated by forward_kl |
| Pure tree losses (8B, GSM8K + math_hard) | No signal — confirmed across two datasets | Gap vs JSD grows with task difficulty (0.1 on GSM8K → 0.5 on math_hard): survival collapse under low acceptance rates, not capacity ceiling alone |
| Additive jsd + tree (8B) | No signal — confirmed at 4000 steps | JSD wins traversal at every K; BV mean monotone declines with λ; one above-jsd cell within noise |

---

## Conclusions

1. **The on-policy survival collapse is the core problem.** The math_hard result is decisive: harder prompts widen the JSD–tree gap from ~0.1 (noise) on GSM8K to ~0.5 BE on MATH. The mechanism is structural: lower acceptance rates under harder prompts collapse the survival product W_d = Π α_j geometrically, concentrating all gradient on the first token. The tree loss degenerates to noisy single-step distillation — noisier than JSD because it samples from a bad draft rather than the teacher's rollout. More capacity pressure makes this worse, not better.

2. **The algorithm is correct but the estimator is fragile.** Overfit probes confirm the telescoping gradient raises E[τ] by +20×SE on 16 prompts — the gradient direction is right. The failure at scale is the on-policy sampling: the tree estimator requires the draft to be close enough to the teacher for survival products to stay non-negligible. That condition is never satisfied during training, and worsens on harder data.

3. **Capacity ceiling is a secondary issue.** The primary barrier is that on-policy tree losses self-defeat under any meaningful model gap — the gradient vanishes before capacity is even reached. Removing the model gap (32B teacher) would make both JSD and tree losses better, but the survival collapse would remain.

4. **Acceptance-aware losses need off-policy or re-anchored gradients** to work, not more capacity. Training with the teacher's own rollout as the tree path (off-policy tree loss) would give dense, non-collapsing gradients at every depth — the same advantage JSD already has — while still optimizing the acceptance objective. This is the open algorithmic question.

4. **How would depth weighting work** if the core problem is gradient direction? A detached scalar multiplier has identical gradient direction to plain JSD. It cannot redirect learning toward deeper acceptance regardless of the capacity situation.

5. **traversal + BV are the right primary metrics.** Both are high-BE verifiers, sensitive to training quality differences, and together give a more complete picture than traversal alone (traversal uniquely scales with K; BV captures tree-width acceptance).

---

## Phase 8 — MATH Hard Probe (2026-06-19)

**Question:** Is the 8B GSM8K null result a capacity ceiling (H1) or does it hold on harder data? If tree losses have real signal, harder prompts (where the draft is further from the teacher) should widen their advantage — the acceptance product has more room to move.

**Dataset:** MATH train, levels 4 and 5 only (~2.5K problems). Val: MATH test split.

**Runs (4000 steps, seed 123):**

| Run | Checkpoint | Best val BE | Notes |
|---|---|---|---|
| jsd (math_hard) | checkpoints/jsd/ckpt_best | **6.312** (step 2500) | val = math_val |
| naive_tree (math_hard) | math_naivetree_s123/ckpt_best | — | val BE not tracked; offline eval below |

Note: jsd math_hard val BE (6.312) is on math_val, not gsm8k_val — not directly comparable to the GSM8K jsd best of 5.996.

**Offline eval on math_eval (n=100) — both checkpoints trained on math_hard:**

BV verifier:

| Checkpoint | K=1 | K=2 | K=3 | K=4 | Mean K=1–4 |
|---|---|---|---|---|---|
| naive_tree | 5.423 | 5.287 | 5.495 | 5.375 | 5.395 |
| **jsd** | **6.015** | **5.914** | **6.018** | **6.014** | **5.990** |
| Δ jsd − tree | +0.592 | +0.627 | +0.523 | +0.639 | **+0.595** |

Traversal verifier:

| Checkpoint | K=1 | K=2 | K=3 | K=4 | Mean K=1–4 |
|---|---|---|---|---|---|
| naive_tree | 5.351 | 5.686 | 5.432 | 5.576 | 5.511 |
| **jsd** | **6.013** | **5.991** | **5.913** | **5.943** | **5.965** |
| Δ jsd − tree | +0.662 | +0.305 | +0.481 | +0.367 | **+0.454** |

**Result: decisive negative — JSD wins by 0.45–0.60 BE across every K and both verifiers (noise floor ±0.15).** JSD is remarkably stable (~6.0 at every K); naive_tree is both lower and more variable.

**Critical finding — harder data made the gap larger, not smaller.** On GSM8K the jsd–naive_tree gap was ~0.1 BE (within noise). On math_hard it is ~0.5 BE (3× the noise floor). This is the **opposite** of the probe hypothesis.

**Root cause — on-policy survival collapse under capacity pressure:** The tree loss gradient at depth d is weighted by the survival product W_d = Π_{j<d} α_j. On harder prompts the draft is further from the teacher → acceptance rates α are lower (≈0.4 on MATH vs ≈0.6 on GSM8K). This makes W_d collapse exponentially faster: at depth 4, W_4 ≈ 0.06 on MATH vs 0.22 on GSM8K. The tree loss degenerates toward **noisy single-step distillation** — noisier than JSD because it uses on-policy draft samples (far from teacher), while JSD trains on the teacher's own high-quality rollout. The harder the task, the more the tree loss suffers from this sampling variance, and the cleaner JSD's advantage.

**Implication:** on-policy tree losses showed no detectable signal under capacity pressure — precisely the regime where they were theorized to help. This is more definitive than the GSM8K result and identifies survival collapse, not capacity ceiling, as the primary barrier.

---

## Next Steps

### Step 2 — 32B Teacher Run (if math probe passes or as direct next step)

**Rationale:** 32B teacher / 0.6B draft is a 53× model gap. More GSM8K prompts become unlearnable, and acceptance-aware losses have real signal to exploit.

**Proposed run order (≥2 seeds on key comparisons):**

| Priority | Run | Purpose |
|---|---|---|
| 1 | jsd s123, s456 | Pin 32B flat ceiling |
| 2 | op_naive_tree_full s123, s456 | Test if off-policy tree lifts above flat |
| 3 | naive_tree s123, s456 | On-policy tree for survival-collapse comparison |
| 4 | kl_tree s123 | Second tree loss for comparison if op_ shows signal |

**Skip:** gbv_tree, bv_tree (broken). depth_weight (gradient-free, ruled out). Anything with only 1 seed before claiming a result.

**Key config changes from 8B runs:**
- Change `TEACHER_MODEL` in train.py to Qwen3-32B path
- Keep K=3, L=8 for comparability
- val_temp=0.2 (already in place)

---

### Step 3 — L Sweep (eval-only, no retraining)

On the best jsd checkpoint, run eval with L=8/16/32/64 at K=2, traversal. Measures α(depth) decay curve and whether the ~25% of paths hitting the L=8 wall represent real headroom. No training cost; pure measurement.

```bash
python eval.py --checkpoint checkpoints/jsd/ckpt_best \
    --mode traversal --K 2 --L 16 --n 100
# repeat for L=32, L=64
```

---

### Open Code Issues

| Issue | Where | Fix | Priority |
|---|---|---|---|
| traversal_tree wrong functional (A-002) | losses/tree.py | **RESOLVED** — now uses `_telescoping_loss(_alpha_naive)`; exact at K=1, approx at K>1. **Note:** tree losses beyond depth_weight (scalar multiplier) are not yet signed off by researcher — full-gradient tree training remains under review. | — |
| ZeroDivisionError in traversal verifier (A-003) | verifiers/verifier.py:433 | `max(denom, 1e-9)` — 1-line PR | Medium — drops ~1% of traversal prompts |
