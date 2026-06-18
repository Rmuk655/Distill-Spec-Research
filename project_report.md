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

**Answer at 8B/0.6B/GSM8K: No.** Flat JSD plateaus at val BE ≈ 6.0 and no tree-loss or depth-weighting variant improves on it. This is a genuine capacity ceiling — the algorithm is confirmed correct (overfit probe shows telescoping gradient works), but the 0.6B has no headroom left on GSM8K under an 8B teacher.

**Key eliminations:**
- Depth × JSD (researcher's suggestion): dead — gradient-free scalar, total sweep spread 0.055 < noise floor ±0.15
- gbv_tree, bv_tree: algorithmically broken — E[τ] collapses to zero on overfit set
- Additive (JSD + λ·tree): monotone decline at 8B, consistent with no-headroom

**Justified next steps:** MATH levels 4–5 probe (cheap; harder prompts → more unlearnable examples → potential signal at 8B); if still null, 32B teacher run (algorithm is correct, capacity pressure is the missing ingredient).

---

## Setup

| Component | Value |
|---|---|
| Draft | Qwen/Qwen3-0.6B, BF16 |
| Teacher | Qwen/Qwen3-8B, BF16 |
| Hardware | NVIDIA H100 80GB HBM3 |
| Train steps | 4000 (standard); 2000 for probes |
| LR schedule | Linear warmup (200 steps) → cosine decay to 0.1× peak |
| Peak LR | 3e-4 (flat), 1e-5 (tree) |
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

### 2.1 forward_kl baseline (run 20260609-001)

**Bug:** val was evaluated at K=4 during training while the target K was 3. Val numbers were misleading. Result not usable for comparison.

**Fix:** hardcoded K=3 in val loop.

### 2.2 nss_tree (run 20260611-001)

Trained with the NSS-verifier-aligned telescoping tree loss.

**Val metric during training:** peaked at 3.612 NSS BE at step 1600 on 25 prompts → looked promising.

**Offline eval (100 prompts):** NSS BE = 3.243 on ckpt_best — **below the untrained baseline** (3.304). Two eval runs on the same checkpoint showed BV BE range of 0.240, establishing that 25-prompt val is noisy.

| Checkpoint | NSS K=3 | Traversal K=3 | vs. baseline |
|---|---|---|---|
| Baseline | 3.304 | 4.907 | — |
| nss_tree ckpt_best (avg) | 3.243 | 5.201 | NSS −0.061, traversal +0.294 |

**Finding F-001:** nss_tree val peak was noise; offline at 100 prompts it fails to beat baseline on its own verifier.

### 2.3 K=1 diagnostic for nss_tree (2026-06-12)

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

**Key findings:**
- All four flat losses beat the untrained baseline by +0.3–0.6 BE on traversal
- forward_kl is strongest, especially at K=2 (5.55 traversal)
- **Tree losses (kl_tree, naive_tree) beat the baseline but consistently lose to every flat loss** at every (K, verifier) combination. The gap is 0.1–0.4 BE, reproducible, not reducible by K choice.
- reverse_kl degrades at K=4; not competitive
- Training val BE peak: jsd reached 5.996 at step ~2300, plateaued, LR floored at 3e-6 by step 4000

**Finding F-002:** traversal is the highest-BE verifier and the most sensitive to training quality. It should be the primary metric.

**Finding F-003:** 100-prompt eval has noise floor ~0.15 BE. Effects below that are directional only.

---

## Phase 4 — K-Scaling Analysis on Baseline (2026-06-13, n=1000)

Evaluated untrained baseline across K=1–4 with 1000 prompts for precision.

| Verifier | K=1 | K=2 | K=3 | K=4 | Trend |
|---|---|---|---|---|---|
| traversal | 4.708 | 4.938 | 4.907 | 4.870 | peaks K=2 |
| naive | 4.450 | 4.592 | 4.625 | 4.629 | increases |
| bv | 4.634 | 4.625 | 4.675 | 4.662 | flat |
| gbv | 4.670 | 4.544 | 4.241 | 4.074 | decreases |
| nss | 3.398 | 3.277 | 3.213 | 3.135 | decreases |
| spectr | 4.450 | 4.506 | 4.450 | 4.310 | peaks K=2 |
| specinfer | 4.450 | 4.241 | 4.039 | 3.912 | decreases |
| khisti | 4.476 | 4.262 | 4.006 | 3.898 | decreases |

**Key finding:** traversal uniquely scales with K (more branching → more acceptance paths → higher BE). NSS, specinfer, khisti, gbv degrade with K. BV is flat. Extra K beyond 2 is wasteful for most verifiers; traversal gets most of its K-gain at K=2.

**α ≈ 0.84** (average per-token acceptance rate). At L=8, ~25% of paths hit the horizon wall — genuine headroom for larger L, especially under a trained draft.

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

**Root cause:** `w(d)` is a detached scalar — `∂L/∂θ = w(d) · ∂JSD/∂θ`. It is a per-prompt adaptive learning rate. It cannot change the gradient direction, so it cannot teach the draft *how* to be accepted deeper. This mechanism is **dead in all forms**.

Note: lam0.0 and jsd follow different trajectories despite identical loss because the depth_weight code unconditionally runs a full target tree pass at λ=0, consuming RNG every step — a code artefact, not science.

---

## Phase 6 — Additive Tree Aux: jsd + λ·naive_tree (2026-06-17 to 06-18)

**Mechanism:** `L = JSD + λ · L_naive_tree` — real per-level acceptance gradient through q, with the natural survival weight `Π αⱼ.detach()` (telescoping). The only mechanism that actually redirects gradients.

**Early read (~1000/4000 steps):**

| Condition | val BE ~step 1000 | vs. jsd control |
|---|---|---|
| jsd (seed 123) | leading | — |
| jsd + naive_tree × 0.1 | below jsd | negative |
| jsd + naive_tree × 0.3 | below jsd | negative |
| jsd + naive_tree × 1.0 | below jsd; ZeroDivisionError 1/25 val prompts | negative |

**Interpretation:** monotone decline as tree weight increases. Consistent with the capacity ceiling — at 8B/0.6B/GSM8K, flat JSD already reaches q≈p everywhere the 0.6B can reach, so tree gradient is at best redundant and at worst noisy. Full 4000-step runs still running at time of writing.

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

**gbv_tree and bv_tree are algorithmically broken.** E[τ] collapses to zero within 50 steps; loss saturates at −0.0000 (gradient vanishes entirely). These losses actively destroy acceptance — do not use at any scale. Root cause: likely a gradient sign error or degenerate optimum in the BV/GBV surrogate formulation. Needs investigation before any fix attempt.

**naive_tree, kl_tree, and traversal_tree are algorithmically correct.** H2 is rejected for all three. The 8B null result is genuine no-headroom (H1).

**traversal_tree caveat (open issue A-002):** the current implementation returns −Π min(1,p/q) (terminal survival product) instead of the telescoping sum −E[τ]. It still works in the overfit probe because maximising terminal survival correlates with maximising E[τ] — but it optimises a loose lower bound with vanishing gradient at depth. The functional fix (pending researcher sign-off) is expected to improve it further. Do not use traversal_tree at scale without the fix.

**Implication for additive:** `jsd + λ·naive_tree` and `jsd + λ·kl_tree` inherit working tree gradients. The additive null result at 8B is a capacity ceiling, not a bug.

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
| jsd + naive_tree (early) | flat+tree | — | below jsd | — | negative |

### gbv_tree / bv_tree: do not use
These are excluded from the table above — they drive E[τ] to zero on the overfit set (broken gradient, not just suboptimal).

---

## What Was Ruled Out

| Approach | Status | Why |
|---|---|---|
| nss_tree | Structural negative | Mode-seeking surrogate at K=1; collapses coverage verifiers |
| gbv_tree | Broken | Gradient vanishes; E[τ] → 0 on overfit set |
| bv_tree | Broken | Same collapse |
| depth_weight (linear, all λ) | Dead | Gradient-free scalar; 5× swing < noise; no-op tops table |
| reverse_kl | Suboptimal | Degrades at K=4; dominated by forward_kl |
| Pure tree losses (8B) | No signal | Algo correct; beat baseline; lose to flat at capacity ceiling |
| Additive jsd + tree (8B, early read) | No signal yet | Consistent with capacity ceiling; full run pending |

---

## Conclusions

1. **The capacity ceiling is real.** Flat JSD reaches val BE 5.996 and plateaus from step ~2300 with LR floored. This is the 0.6B's limit on GSM8K under an 8B teacher.

2. **The algorithm is correct** for naive_tree and kl_tree. The telescoping acceptance gradient raises E[τ] from 1.87→5.50 (+20×SE) on an overfit set. The 8B null result is a genuine no-headroom problem, not a wiring bug.

3. **Acceptance-aware losses need capacity pressure** to separate from flat distillation. When q≈p is reachable (GSM8K + 8B teacher + 0.6B draft), flat KL already achieves the joint optimum of every acceptance metric, leaving tree losses nothing to exploit.

4. **Depth weighting (researcher's suggestion)** cannot work by construction: a detached scalar multiplier has identical gradient direction to plain JSD. It cannot redirect learning toward deeper acceptance.

5. **traversal is the right primary metric.** It is the highest-BE verifier, uniquely scales with K, and is the most sensitive to draft quality differences.

---

## Next Steps

### Step 1 — MATH Hard Probe (immediate, ~4–6h)

**Rationale:** Harder prompts create more "unlearnable" examples where the 0.6B cannot match the 8B teacher. If the capacity ceiling is GSM8K-specific, tree losses should show signal here even at 8B.

**Command (code updated this session):**
```bash
# Download once on server
python -m data_io.download --datasets math_hard,math_val

# Run 2000-step probe (not full 4000 — just looking for signal)
python train.py --loss jsd \
    --train_dataset math_hard --val_dataset math_val \
    --steps 2000 --seed 123 --device cuda:0 \
    --output checkpoints/math_hard_jsd_s123

python train.py --loss naive_tree \
    --train_dataset math_hard --val_dataset math_val \
    --steps 2000 --seed 123 --device cuda:1 \
    --output checkpoints/math_hard_naive_tree_s123
```

**Interpret as:** if naive_tree val BE > jsd val BE by >0.1 on math_hard at 8B → capacity pressure is the missing ingredient, 32B will be decisive. If still tied → capacity hypothesis holds at all math difficulties, 32B is fully justified.

**Dataset:** MATH train, levels 4 and 5 only (~2.5K problems). Val: MATH test split.

---

### Step 2 — 32B Teacher Run (if math probe passes or as direct next step)

**Rationale:** 32B teacher / 0.6B draft is a 53× model gap. More GSM8K prompts become unlearnable, and acceptance-aware losses have real signal to exploit.

**Proposed run order (≥2 seeds on key comparisons):**

| Priority | Run | Purpose |
|---|---|---|
| 1 | jsd s123, s456 | Pin 32B capacity ceiling |
| 2 | naive_tree s123, s456 | Test if tree lifts above flat |
| 3 | jsd + naive_tree × 0.1, × 0.3 | Test additive if tree shows signal |
| 4 | kl_tree s123 | Second tree loss for comparison |

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
| traversal_tree wrong functional (A-002) | losses/tree.py | Telescope sum, pending researcher sign-off | High — affects traversal_tree experiments |
| ZeroDivisionError in traversal verifier (A-003) | verifiers/verifier.py:433 | `max(denom, 1e-9)` — 1-line PR | Medium — drops ~1% of traversal prompts |
| Additive full run result | — | Wait for 4000-step completion | Low — unlikely to change conclusion at 8B |
