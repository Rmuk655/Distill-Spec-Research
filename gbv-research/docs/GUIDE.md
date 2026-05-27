# SpecDist Experiment — Research Guide

**System**: Qwen2.5-0.5B draft → Qwen3-0.6B target (laptop smoke) | Qwen3-0.6B → Qwen3-8B (colab/a100)  
**Goal**: Train the draft model with a novel block-level EBE loss so it gets accepted more often by the target, speeding up generation without changing what the target produces.

---

## Table of Contents

0. [The Three-Stage Experiment Sequence](#0-the-three-stage-experiment-sequence)
1. [Research Hypotheses](#1-research-hypotheses)
2. [Verifier Hierarchy and What We Need to Prove](#2-verifier-hierarchy-and-what-we-need-to-prove)
3. [Minimal Experiment Matrix for Paper](#3-minimal-experiment-matrix-for-paper)
4. [Key Metrics](#4-key-metrics)
5. [Pipeline Phases (by tier)](#5-pipeline-phases-by-tier)
6. [Training Losses](#6-training-losses)
7. [Dashboard](#7-dashboard)
8. [How to Run](#8-how-to-run)
   - 8a. [Checkpoint Progression Analysis](#8a-analysing-training-progress-with-checkpoints)
   - 8b. [Server / Colab / Modal Training](#8b-running-training-on-a-server-colab--modal)
   - 8c. [New Environment Setup (setup_download.py)](#8c-setting-up-a-new-environment-setup_downloadpy)
   - 8d. [Results Report (analyze_results.py)](#8d-generating-a-results-report-analyze_resultspy)
   - 8e. [EAGLE Benchmark](#8e-eagle-benchmark)
9. [Automatic Training Health Checks](#9-automatic-training-health-checks)
10. [Weights & Biases Integration](#10-weights--biases-integration)
11. [Quick Interpretation Cheatsheet](#11-quick-interpretation-cheatsheet)
12. [KL Divergence Comparison](#12-kl-divergence-comparison)
13. [Online Speculative Decoding (OSD)](#13-online-speculative-decoding-osd)

---

## 0. The Three-Stage Experiment Sequence

The experiment runs in three hardware tiers, each with a specific purpose. **Never skip a tier**; each stage gates the next.

### Stage 0 — Laptop Smoke (code correctness)

**Hardware**: Any laptop with ≥ 4 GB VRAM (or CPU).  
**Models**: Qwen2.5-0.5B draft → Qwen3-0.6B target.  
**Dataset**: diverse50.  
**hw_tier tag in results.db**: `laptop`

**Purpose**: Verify every code path runs without crash or OOM. Results are **not paper-quality** and should not appear in the paper.

| Parameter | Value |
|---|---|
| Train steps | 100–200 per loss |
| Eval prompts | 10 (n=10) |
| Losses | ALL: forward_kl, reverse_kl, jsd, ebe, l1 |
| Verifiers | ALL: traversal, specinfer, gbv, naive, bv |
| LRs | 1e-5, 3e-5, 1e-4 |
| K | 3, 5 |
| Temperature | 0.6, 1.0 |

```bash
python experiment.py --config laptop --yes --smoke
```

**Decision gate**: if any step crashes or OOMs, fix it before proceeding. If all steps pass, proceed to Stage 1.

---

### Stage 1 — Colab L4/T4 (trend formation, quantized)

**Hardware**: Google Colab T4 (15 GB) or L4.  
**Models**: Qwen3-0.6B draft → Qwen3-8B target **with `--load_in_4bit`** (4-bit NF4 on target).  
**Dataset**: gsm8k_30 (30 prompts) for eval.  
**hw_tier tag in results.db**: `colab`

**Purpose**: Directional signal — does EBE beat forward KL? Do the trends hold? Alpha and block_eff are **directionally meaningful** but timing/throughput are NOT reliable (quantized target changes latency).

| Parameter | Value |
|---|---|
| Train steps | 500 |
| Eval prompts | n=10 |
| Losses | Main: forward_kl, ebe |
| Verifiers | traversal, specinfer, gbv |
| K | 3, 5 |
| Temperature | 0.6, 1.0 |

```bash
python experiment.py --config colab --yes
```

**NOTE**: `--load_in_4bit` is colab-only. The pipeline adds it automatically for `--config colab` and never adds it for `--config server` or `--config a100`.

**Decision gate**: if EBE shows higher alpha/BE than forward_kl at n=10, proceed to Stage 2. If not, investigate before committing to A100 GPU hours.

---

### Stage 2 — A100 (paper quality, bf16)

**Hardware**: A100 (40/80 GB) or H100. Full bf16 — **no quantization**.  
**Models**: Qwen3-0.6B draft → Qwen3-8B target, full bf16.  
**Datasets**: gsm8k_train for training, gsm8k_30 + math500_30 for eval.  
**hw_tier tag in results.db**: `a100`

**Purpose**: Paper-quality numbers. These are the numbers that go in the paper.

| Parameter | Value |
|---|---|
| Train steps | 1000 |
| Eval prompts | n=30 |
| Losses | forward_kl, ebe (+ rev_kl, jsd for §12) |
| Verifiers | ALL 4: traversal, specinfer, gbv, bv |
| K | 3, 5 |
| Temperature | 0.6, 1.0 |

```bash
python experiment.py --config server --yes
```

**IMPORTANT**: `evaluate.py --hw_tier a100` includes a guard that errors if the target model appears quantized. This prevents accidentally tagging quantized (Colab) results as a100 tier.

**Decision gate**: after A100 runs, use `analyze_results.py` to generate the paper table.

---

## 1. Research Hypotheses

The paper tests four hypotheses. Every experiment should be designed to confirm or refute one or more of them.

### H1 — Traversal verifier > SpecInfer (primary claim)

**Prediction**: traversal block efficiency > specinfer block efficiency, consistently across K=3 and K=5, across gsm8k and math500, at both T=0.6 and T=1.0.

**What "consistently" means**: traversal wins in ≥ 6/8 cells of the {K=3,5} × {gsm8k, math500} × {T=0.6,1.0} grid. A single win at K=5/T=0.6 is not sufficient.

**Rationale**: Traversal uses bottom-up acceptance which avoids wasting tree budget at shallow nodes (where p≈q and marginal gains are small). SpecInfer uses top-down prefix acceptance which discards valid non-prefix subtrees.

---

### H2 — EBE loss > forward KL (training claim)

**Prediction**: A draft trained with EBE loss achieves higher block efficiency under traversal verifier than a draft trained with forward KL, given the same number of training steps and learning rate.

**What to measure**: BE_traversal(EBE) − BE_traversal(KL) at K=3, K=5, both temperatures, both datasets.

**Rationale**: EBE directly optimises E[τ+1] = 1 + Σ_k Π_{i≤k} α_i, which is exactly block efficiency. Forward KL minimises KL at every token position independently without caring about the sequential acceptance product.

---

### H3 — Online OSD beats offline best

**Prediction**: Starting from the best offline-trained checkpoint (EBE, lr=1e-4), online OSD adaptation on incoming GSM8K queries raises traversal block efficiency by ≥0.15 tokens/call vs the offline checkpoint alone.

**Rationale**: offline training minimises KL at every position; online training targets only rejection positions as they appear in live inference — a signal directly tied to what the verifier rejects, not a proxy loss.

**Dataset dependence**: online results are distribution-matched to the adaptation dataset. Run separate online experiments for each eval dataset if you want cross-dataset claims.

**What would refute H3**: online eval_alpha at step 500 ≤ offline eval_alpha (indicates online training is not providing useful signal over the offline baseline).

---

### H4 — Improvements generalise across datasets

**Prediction**: EBE > KL ordering holds on both gsm8k and math500, at both temperatures. The gain on math500 is within 50% of the gain on gsm8k (shows the improvement is not GSM8K-specific).

**What to measure**: Phase 4 multi-dataset eval. Compare BE gain (%) over baseline for EBE vs KL, across all datasets.

---

## 2. Verifier Hierarchy and What We Need to Prove

The verifier is the algorithm that decides, given a draft tree of K paths each of length L, which prefix to accept.

### Expected ranking (Thomas et al. 2026, arXiv:2602.16994v1)

```
traversal (5.31) > spectr (4.61) ≈ specinfer (4.58) > bv (4.30) > nss (4.05)
```

For our paper, the relevant ordering is:

```
traversal ≥ gbv > specinfer > bv > naive
```

| Verifier | Role in paper | Run tier |
|---|---|---|
| `traversal` | **Primary claim** — must beat specinfer | colab + a100 |
| `gbv` | Secondary claim — shows GBV also benefits | colab + a100 |
| `specinfer` | **Key comparison** for H1 | colab + a100 |
| `bv` | Lower bound (single-path) | a100 only |
| `naive` | Lower bound (token-level) | laptop smoke only |
| `nss`, `spectr`, `khisti` | Research curiosity, exercise only | laptop smoke only |

**Why naive and bv must be included**: They serve as lower bounds. A paper that only shows EBE beats KL under traversal but doesn't also show the ordering traversal > bv > naive is incomplete — reviewers will ask.

### If traversal doesn't consistently beat specinfer

Do NOT quietly drop the comparison. Report it in a conditional framing:

> "Traversal outperforms specinfer when alpha ≥ X (well-trained draft); below that threshold both methods are equivalent."

This is itself a useful finding: it tells practitioners when traversal is worth the extra verification compute.

The threshold X is estimated from the data: plot BE_traversal − BE_specinfer vs alpha_mean, fit a piecewise linear model. If the crossover point is at alpha ≈ 0.6, report that.

---

## 3. Minimal Experiment Matrix for Paper

36 eval runs are required for the paper. Each row is one (model × dataset × verifier × K × temperature) combination.

| # | Model | Dataset | Verifier | K | T | Hypothesis |
|---|---|---|---|---|---|---|
| 1–4 | baseline | gsm8k | traversal, specinfer | 3, 5 | 0.6, 1.0 | H1 reference |
| 5–8 | kl1000 | gsm8k | traversal, specinfer | 3, 5 | 0.6, 1.0 | H1, H2 |
| 9–12 | ebe1000 | gsm8k | traversal, specinfer | 3, 5 | 0.6, 1.0 | **H1, H2** |
| 13–16 | baseline | math500 | traversal, specinfer | 3, 5 | 0.6, 1.0 | H4 reference |
| 17–20 | kl1000 | math500 | traversal, specinfer | 3, 5 | 0.6, 1.0 | H4 |
| 21–24 | ebe1000 | math500 | traversal, specinfer | 3, 5 | 0.6, 1.0 | **H4** |
| 25–28 | baseline | gsm8k | gbv, bv | 3, 5 | 0.6, 1.0 | H1 bounds |
| 29–32 | ebe1000 | gsm8k | gbv, bv | 3, 5 | 0.6, 1.0 | H1, H2 bounds |
| 33–34 | online | gsm8k | traversal | 3, 5 | 0.6 | **H3** |
| 35–36 | online | math500 | traversal | 3, 5 | 0.6 | H3, H4 |

Rows 9–12 and 21–24 are the core paper results. All 36 must be in the a100 tier.

---

## 4. Key Metrics

### Alpha (α) — Token Acceptance Rate
**Range**: 0 to 1. **Higher is better.**

Alpha is the per-token acceptance probability: given that the target has seen position *t*, how likely is it to accept the draft's proposal at position *t+1*?

```
α_t = min(1,  q(token_t) / p_draft(token_t))
```

where `q` is the target's probability and `p_draft` is the draft's probability for the actual token generated. When `p_draft ≥ q` the proposal is always accepted (α=1). When `p_draft < q` the proposal is accepted with probability `q/p_draft < 1`.

**Reported as**: mean ± 95% CI across all prompts and token positions.

**What to look for**: a trained model should have higher alpha than the baseline. An increase from 0.7 → 0.8 means the draft is much more frequently in agreement with the target.

---

### Block Efficiency (BE) — Tokens per Target Call
**Range**: 1 to K·L+1. **Higher is better.**

Block efficiency is the average number of tokens added to the output per target model forward pass. A value of 1.0 means every target call only adds one token (pure autoregressive, no speedup). A value of 3.0 means each target call adds three tokens on average.

```
BE = (total tokens generated) / (number of target calls)
```

With a draft tree of width K and depth L, the theoretical maximum is K·L+1 (all tokens accepted). In practice, BE depends on how aligned the draft is with the target.

**Why this matters more than alpha**: alpha is token-level; BE captures the sequential structure of a full speculative block. A draft that gets all early tokens right provides a higher BE than one that scatters its accuracy across positions.

---

### Perplexity (PPL)
**Range**: 1 to ∞. **Lower is better. Do not let it drift far from baseline.**

Perplexity measures how surprised the draft model is by the evaluation dataset. It is used as a sanity check: if training pushes PPL much higher than the baseline (e.g., baseline=9.1, trained=13.9), the model has degraded as a language model even if its acceptance rate went up.

**Healthy outcome**: PPL stays within ~20% of baseline after training.

---

### Throughput (tokens/sec) and Latency (ms/token)
Measured end-to-end for the speculative decoding loop. Throughput increases with BE because more tokens are produced per expensive target call.

**NOTE**: throughput is hardware-dependent. Only compare throughput numbers from the same tier (a100 vs a100, never colab vs a100). The dashboard HW TIER filter prevents mixing tiers in charts.

---

## 5. Pipeline Phases (by tier)

### Laptop smoke tier (--config laptop --smoke)

```bash
python experiment.py --config laptop --yes --smoke
```

Phase 0: Merge pre-existing LoRA adapters.  
Phase 1: Quick eval on gsm8k — baseline + LR sweep (3 LRs × 6 eval steps).  
Phase 5: Laptop smoke — ALL verifiers (traversal, specinfer, gbv, naive, bv) + ALL losses (200 steps each on diverse50). Exercises every code path.

**State file**: `pipeline_state_laptop_smoke.json` (separate from the full pipeline so smoke "done" marks never block real training).

---

### Colab tier (--config colab)

```bash
python experiment.py --config colab --yes
```

Phase 0–1: same structure as laptop.  
Phase 2: Training KL + EBE on gsm8k, 1000 steps, **with `--load_in_4bit`** (target loaded in 4-bit NF4 for T4 VRAM).  
Phase 2b: Reverse KL + JSD variants.  
Phase 2c: Online OSD adaptation.  
Phase 3: Full eval on gsm8k (traversal, specinfer, gbv).  
Phase 4: Multi-dataset eval.

All results tagged `hw_tier=colab` in results.db. Alpha and BE are directionally meaningful; throughput is not reliable due to quantization.

---

### A100 tier (--config server)

```bash
python experiment.py --config server --yes
```

Same phases as colab, but:
- Target loads in **full bf16** — NO `--load_in_4bit`.
- All results tagged `hw_tier=a100`.
- n=30 prompts for paper-quality statistics.
- Both datasets: gsm8k_30 and math500_30 for eval.
- ALL 4 verifiers: traversal, specinfer, gbv, bv.

The `evaluate.py --hw_tier a100` guard will error if it detects a quantized target, preventing accidental contamination of paper-quality data.

---

### Per-phase details

#### Phase 0 — Merge (3 steps, ~5 min each)

Pre-trained LoRA adapters (from an earlier LR sweep) are merged into a copy of the base draft model and saved to disk as a standalone model.

**What "merge" means**: LoRA training adds two small matrices A and B alongside every weight W. The merged model computes `W' = W + A·B` and saves W' — the adapters disappear, the model file is a normal HuggingFace model.

**Inputs**: `checkpoints/ebe_lr{X}/` (LoRA adapter)  
**Outputs**: `checkpoints/ebe_lr{X}_merged/` (full model)

---

#### Phase 1 — Quick Eval on GSM8K (6 steps, ~10–15 min each)

**Purpose**: Rapidly see whether the LR-sweep checkpoints improved over baseline on math word problems, before committing to a 5-hour training run.

| Step | Model evaluated | What it tests |
|---|---|---|
| `eval_baseline_gsm8k` | Raw draft (no training) | Reference point for all comparisons |
| `eval_kl200_gsm8k` | KL-distilled, 200 steps on diverse data | Sanity check: does KL distillation help? |
| `eval_ebe200_gsm8k` | EBE-distilled, 200 steps on diverse data | **Key early signal**: does the novel loss help? |
| `eval_ebe_lr1e5_gsm8k` | EBE, lr=1e-5 | Learning rate sensitivity |
| `eval_ebe_lr3e5_gsm8k` | EBE, lr=3e-5 | Learning rate sensitivity |
| `eval_ebe_lr1e4_gsm8k` | EBE, lr=1e-4 | Learning rate sensitivity |

---

#### Phase 2 — Full Training on GSM8K

**Purpose**: Train the draft model properly on real GSM8K data, for 1000 steps.

| Step | Action |
|---|---|
| `train_kl_gsm8k` | Train with KL distillation loss for 1000 steps |
| `merge_kl_gsm8k` | Merge the KL-1000 LoRA into a standalone model file |
| `train_ebe_gsm8k` | Train with EBE loss for 1000 steps |
| `merge_ebe_gsm8k` | Merge the EBE-1000 LoRA into a standalone model file |

**Colab vs A100**: The pipeline automatically adds `--load_in_4bit` for colab config and omits it for server config. Never manually add `--load_in_4bit` to server runs.

---

#### Phase 3 — Full Eval on GSM8K

| Step | Model | What it proves |
|---|---|---|
| `eval_kl1000_gsm8k` | KL-1000 trained draft | Baseline trained model performance |
| `eval_ebe1000_gsm8k` | EBE-1000 trained draft | **Core result**: does EBE beat KL after full training? |

---

#### Phase 4 — Multi-Dataset Eval

| Step | Datasets covered |
|---|---|
| `eval_baseline_all` | HumanEval, MATH-500, MTBench, Alpaca (diverse50) |
| `eval_kl1000_all` | Same 4 datasets |
| `eval_ebe1000_all` | Same 4 datasets |

**Purpose**: Show that the improvement is not specific to GSM8K.

---

## 6. Training Losses

### KL Distillation (baseline training method)

```
L_KL = KL( q_target || p_draft ) = Σ_v  q(v) · log( q(v) / p_draft(v) )
```

Standard knowledge distillation. Minimizes the KL divergence from target to draft over the full vocabulary at every token position.

**Limitation**: treats every token position independently. Doesn't directly optimize the acceptance probability product that determines block efficiency.

---

### EBE Loss — Block-Level Expected Block Efficiency (novel contribution)

```
L_EBE = -(1 + Σ_{k=1}^{T} Π_{i=1}^{k} α_i)  +  λ · KL(q || p_draft)

where  α_i = min(1, q(token_i) / p_draft(token_i))
```

**What it optimises**: the expected number of tokens accepted in a speculative block, which is exactly `1 + Σ_k Π_{i≤k} α_i`. Minimizing the negative of this maximizes block efficiency directly.

**Gradient intuition**: for token *i* where `p_draft > q` (draft over-estimates), the gradient pushes `p_draft` *down* toward `q`. The cumulative product (`torch.cumprod`) amplifies this signal for early tokens — if token 1 is the bottleneck, fixing it benefits the entire block.

**KL regularizer** (λ=0.1): the EBE gradient vanishes when `p_draft ≤ q` (the token is already under-estimated, α=1). Without KL, under-estimated token probabilities drift unconstrained, causing perplexity collapse. The KL term provides gradient for those positions.

**Key difference from KL**: KL only pushes draft probabilities toward the target; it cannot push them away when the draft is over-confident. EBE explicitly corrects over-estimates, which is the root cause of low acceptance rates.

---

## 7. Dashboard

Start with:
```
python viz_server.py       # opens http://localhost:5000
```

The dashboard auto-refreshes every 30 seconds. The status bar at the top shows live pipeline progress.

---

### HW Tier Filter (NEW)

At the top of the sidebar is a **HW TIER** filter with three chips:

| Chip | Color | Meaning |
|---|---|---|
| `laptop` | Blue (#4299e1) | Smoke/correctness runs only — exclude from paper charts |
| `colab` | Orange (#ed8936) | Trend-formation, quantized — directional only |
| `a100` | Green (#48bb78) | Paper-quality, bf16 — include in paper tables |

**Default**: all three selected.

**Typical workflow for paper tables**: deselect `laptop` and `colab`, keep only `a100`. All charts will then show only paper-quality numbers.

The filter applies to all charts simultaneously (alpha, BE vs K, mode comparison, etc.). Legacy runs inserted before the `hw_tier` column was added are treated as `laptop`.

---

### Sidebar Filters

Click chips to filter which runs are shown in all charts. Multiple chips within one category are OR'd; across categories they are AND'd.

Example: select `ebe1000` in Draft/Loss AND `gsm8k` in Dataset AND `a100` in HW TIER → shows only the EBE-1000 model on GSM8K on A100 hardware.

Click **Apply Filters** to update charts, **Clear All** to reset all filters including HW TIER back to all-selected.

---

### Status Bar (top of page)

| Element | Meaning |
|---|---|
| **Pulsing yellow badge** | Step currently running |
| **Green badge** | Step completed |
| **Red badge** | Step failed — check logs |
| **Progress bar** | N of total steps done |
| **GPU bar** | Live VRAM usage (orange → red as it fills) |
| **DB:** label | Most recently completed evaluation cell |
| **Steps button** | Expand/collapse per-step status panel with tooltips |
| **Logs button** | Show/hide live GBV subprocess output (`be_progress.log`) |

---

### Tab: Key Results

The three hypothesis-proving charts. Start here.

**Chart A — Loss × Verifier Heatmap**  
Rows = verifier modes, Columns = distillation models (baseline, kl200, ebe200, …). Color intensity = block efficiency. Darker red = higher BE.

*What to look for*: the EBE column should be darker than the KL column, especially for traversal and GBV modes.

**Chart B — BE Gain over Baseline (%)**  
Shows `(BE_model − BE_baseline) / BE_baseline × 100` for each trained model and verifier mode.

*What to look for*: EBE bars should be taller than KL bars. Negative bars mean the model is *worse* than the untrained baseline.

**Chart C — Dataset Robustness**  
Block efficiency across all four datasets (GSM8K, HumanEval, MATH-500, diverse50).

*What to look for*: consistent ordering across datasets (EBE > KL > baseline). If EBE wins only on GSM8K but loses on HumanEval, the model has overfit to the training distribution.

---

### Tab: Alpha

**Alpha by Condition (bar chart)**  
Mean token acceptance rate (α) per model, grouped by dataset. Error bars = 95% CI.

**Alpha Distribution (horizontal bars)**  
Same data, horizontal orientation, easier to compare models across many datasets.

---

### Tab: BE vs K

**Block Efficiency vs K (line chart)**  
X axis = K (3 or 5), Y axis = mean BE. One line per model.

*What to look for*: all lines should increase with K. The gap between EBE and baseline should widen as K increases.

**BE vs K — All Modes (subplots)**  
Same chart repeated for each verifier mode side by side, so you can see which mode benefits most from K.

---

### Tab: Mode Comparison

Grouped bar chart: X axis = verifier mode, Y axis = BE. One bar group per model.

*What to look for*: the ordering traversal ≥ gbv > specinfer > bv should hold (H1). EBE should gain more than KL (H2).

---

### Tab: Temperature

**Temperature Sensitivity (line chart)**  
X axis = sampling temperature (0.6, 1.0), Y axis = BE. One line per model.

*What to look for*: lines should decrease left-to-right (lower T = higher BE). The *relative ordering* of models should be preserved at both temperatures.

**Temperature Gain vs Baseline (line chart)**  
`BE_model − BE_baseline` at each temperature.

---

### Tab: Sensitivity

Fully configurable scatter/line chart. Choose any X axis (K, temperature, train steps, learning rate) and any Y axis (alpha, BE, throughput, latency), grouped by any categorical column.

---

### Tab: Per-Category

Heatmap of token acceptance rate broken down by prompt category within the `diverse50` dataset.

---

### Tab: Throughput

**Throughput by Condition (bar chart)**  
Tokens per second, grouped by dataset and verifier mode.

**IMPORTANT**: Only compare throughput within the same `hw_tier`. Use the HW TIER filter to select `a100` only before reading throughput charts.

---

### Tab: Training Curves

Loss plotted against training step. One line per model. Updated live during training.

*What to look for*:
- Both curves should decrease smoothly
- EBE loss starts large (the cumprod term can be large) but should converge
- If a curve spikes upward or oscillates, the learning rate is too high
- **Red banner** appears if val loss rises >10% above its minimum → early stopping recommended

---

### Tab: All Runs

Full table of every evaluation row in the database. Supports sorting and can be exported to CSV.

**Key columns**:
- `hw_tier`: which hardware tier produced this result (laptop / colab / a100)
- `draft_label`: which model was evaluated
- `mode`: verifier algorithm
- `K`: tree width
- `temperature`: sampling temperature
- `alpha_mean` / `alpha_ci95`: acceptance rate with confidence interval
- `block_eff`: block efficiency
- `throughput`: tokens per second
- `perplexity`: language model perplexity (sanity check)
- `experiment_tag`: tags multiple runs from the same experiment session

---

## 8. How to Run

### Laptop smoke test (verify setup, ~10 min)
```bash
python experiment.py --config laptop --yes --smoke
```
Runs 5 prompts, max 30 tokens, K=3, modes=gbv+specinfer, temp=0.6.  
Also runs Phase 5 laptop_smoke: ALL verifiers including naive and bv, ALL losses (200 steps) on diverse50 — exercises every code path.

### Full colab run (trend formation, ~8-12 hr on T4)
```bash
python experiment.py --config colab --yes
```
`--load_in_4bit` is added automatically. Do NOT add it manually.

### Full A100 run (paper-quality, ~24-48 hr on A100)
```bash
python experiment.py --config server --yes
```
No quantization. Full bf16. Results tagged `hw_tier=a100`.

### Check status without running
```bash
python experiment.py --status
```

### Resume after a crash
The pipeline saves state after every step and resumes from the first `pending` step:
```bash
python experiment.py --config server --yes
```

### View dashboard
```bash
python viz_server.py
```
Open http://localhost:5000 in a browser. Use the HW TIER filter in the sidebar to select which tier's results to display.

### Force restart from a specific step
```bash
python experiment.py --config server --from eval_baseline_gsm8k --yes
```

---

## 8a. Analysing Training Progress with Checkpoints

Training saves permanent numbered checkpoints every 200 steps (controlled by `--milestone_every`):
```
checkpoints/kl-run/
  ckpt_latest/          ← crash-safe, always overwritten
  ckpt_step_00200/      ← permanent snapshot at step 200
  ckpt_step_00400/      ← permanent snapshot at step 400
  ...
```

To answer "is training actually helping?", evaluate block efficiency at each checkpoint:

```bash
# 1. Merge the LoRA adapter into a standalone model
python train_qwen3.py --merge_only \
    --adapter checkpoints/kl-run/ckpt_step_00200 \
    --draft Qwen/Qwen3-0.6B

# 2. Run speculative decoding eval on the merged model
python evaluate.py \
    --student checkpoints/kl-run/ckpt_step_00200_merged \
    --teacher Qwen/Qwen3-8B \
    --student_label kl_step200 \
    --datasets gsm8k --modes gbv,specinfer --K 3 --n 10 \
    --hw_tier a100

# 3. Repeat for each checkpoint, then view in dashboard
python viz_server.py
```

The Mode Comparison tab will show a bar for each checkpoint. If BE increases step-by-step, training is working. If it flatlines or drops after a certain step, that is where to stop training.

**What rising validation loss means**: A red banner appears if val loss rises >10% above its minimum. This is the clearest early indicator that training is starting to memorise the training set rather than learning generalizable alignment.

---

## 8b. Running Training on a Server (Colab / Modal)

The existing `train_qwen3.py` runs anywhere — no code changes needed. These are the setup steps for each environment.

---

### Google Colab (free T4, 15 GB VRAM)

```python
# Cell 1: Mount Drive so checkpoints survive session restart
from google.colab import drive
drive.mount('/content/drive')

# Cell 2: Install dependencies
!pip install -q peft transformers accelerate bitsandbytes

# Cell 3: Clone / upload the OSD directory, then train
import os
os.environ["TRANSFORMERS_OFFLINE"] = "0"   # allow first-time download

# COLAB ONLY: --load_in_4bit loads the 8B target in 4-bit NF4 for T4 15GB
# DO NOT use --load_in_4bit on A100/server runs (bf16 full precision required)
!python train_qwen3.py \
    --draft  Qwen/Qwen3-0.6B \
    --target Qwen/Qwen3-8B \
    --load_in_4bit \
    --loss forward_kl \
    --steps 1000 \
    --output /content/drive/MyDrive/OSD/checkpoints/kl-run \
    --dataset data/gsm8k_train.jsonl \
    --milestone_every 200 \
    --val_every 50
```

**Notes:**
- `--load_in_4bit` is safe for colab only — the teacher is frozen during training.
- Set `--output` to a Google Drive path so checkpoints survive session death.
- After the first run, set `TRANSFORMERS_OFFLINE=1` to avoid re-downloading.
- When running `evaluate.py` on colab after training, pass `--hw_tier colab` to tag results correctly.

#### Validation frequency and compute cost

**Why `--val_every` defaults to 50 and not 10 (like `--log_every`):**

Training loss is free — it is already computed during the forward pass of every training step. Validation loss is expensive — it runs a completely separate inference cycle for each val prompt.

**Recommended settings per environment:**

| Environment | `--val_every` | `--val_split` | Rationale |
|---|---|---|---|
| Laptop (≤ 4 GB) | 50 (default) | 0.10 | Low overhead; 10 prompts cap keeps val ≤ 70 s |
| Colab T4 | 100 | 0.05 | Avoids 20% overhead eating into session time |
| A100 / H100 server | 50 | 0.10 | Cheap enough at this GPU speed |
| Quick debug run | 0 | — | Disable val entirely with `--val_every 0` |

---

### Modal.com (A100 on-demand, billed per second)

Modal requires a Python wrapper around the training call. `modal_train.py` in this directory is that wrapper.

```bash
# One-time setup
pip install modal
modal token new

# Download models into Modal's persistent volume (~16 GB, runs once)
modal run modal_train.py::download_models

# Train — KL baseline
modal run modal_train.py::train_kl

# Train — EBE novel loss
modal run modal_train.py::train_ebe

# Custom args (e.g. 2000 steps, lr=1e-4)
modal run modal_train.py::run --loss ebe --steps 2000 --lr 1e-4

# Download checkpoints from the volume to your laptop
modal volume get specdist-vol /checkpoints ./local_checkpoints
```

**Cost reference** (A100-80GB): 1000 steps × ~30 s/step ≈ 8–9 hours ≈ $8–12 USD on Modal.

---

## 8c. Setting Up a New Environment (setup_download.py)

`setup_download.py` is a **one-time-per-machine** (or **one-time-per-session** on ephemeral compute) script that downloads all required model weights and datasets.

### When to run it

| Environment | When to run |
|---|---|
| **Your laptop** | Once. After that, `TRANSFORMERS_OFFLINE=1` is set automatically. |
| **Kaggle** | Every new session (kernel restarts clear the disk cache). |
| **Google Colab** | Every new runtime unless you save to Google Drive. |
| **Modal.com** | Once into the persistent volume via `modal run modal_train.py::download_models`. |

### Usage

```bash
# Laptop (0.5B draft + 0.6B target — default)
python setup_download.py --config laptop

# Server / Colab / Kaggle (0.6B draft + 8B target)
python setup_download.py --config server

# Check what will be downloaded without downloading
python setup_download.py --config server --dry_run
```

---

## 8d. Generating a Results Report (analyze_results.py)

`analyze_results.py` reads from `results.db` and generates a statistical comparison of all evaluation runs.

### Usage

```bash
# Print Markdown report to terminal
python analyze_results.py

# Save to file (for the paper or a PR)
python analyze_results.py --out report.md

# Paper runs only (a100 tier)
python analyze_results.py --draft_labels baseline,kl1000-gsm8k,ebe1000-gsm8k

# Filter by dataset
python analyze_results.py --dataset gsm8k --out gsm8k_report.md
```

### What it reports

- Mean alpha ± 95% CI for each model × dataset × mode combination
- Block efficiency mean and standard deviation
- **Statistical tests**: Welch t-test (unequal-variance), Cohen's d effect size, 95% CI on the difference
- A "winner" row for each pair: does EBE significantly beat KL? (p < 0.05 threshold)

### When to run it

- After Phase 3 completes (first complete results for kl1000 and ebe1000 on GSM8K)
- After Phase 4 completes (cross-dataset results)
- When preparing paper tables

---

## 8e. EAGLE Benchmark

EAGLE is a separate speculative-decoding method that trains a lightweight 1-layer head directly on the **target** model's hidden states. It provides a strong external baseline.

### Run on Colab / Modal / server only — not on laptop

The EAGLE head trains on the **target** model's hidden states. The paper's target is **Qwen3-8B** which requires 16+ GB VRAM.

| Environment | EAGLE? | Why |
|---|---|---|
| **Laptop** (0.6B target) | Do not run | 0.6B head is not a useful baseline |
| **Colab A100** (8B target) | Run here | Fits in VRAM |
| **Modal.com** (8B target) | Run here | Persistent volume survives runs |
| **In-house server** | Run here | Best option if available |

### Running Eagle as part of the pipeline

```bash
# Run full pipeline + EAGLE baseline — use --config server or colab (never laptop)
python experiment.py --config server --yes --eagle

# Run EAGLE phases only (Phases 0–4 already done)
python experiment.py --config server --yes --eagle --from eagle_gen
```

---

## 9. Automatic Training Health Checks

`train_qwen3.py` runs five automated health checks as training proceeds.

### 9a. How Each Check Works

| Check | Trigger | Pass condition | What you see in logs |
|---|---|---|---|
| **1. Loss decreasing** | Every `--health_every` steps (default 100) | Last-10 avg < First-10 avg | `↓ IMPROVING` in health report |
| **2. No NaN in loss** | Every step, before backward | No NaN/Inf | `[HEALTH] ✓ none` in report |
| **3. PPL not collapsed** | Every `--ppl_check_every` steps (default 200) | PPL ≤ baseline × `--ppl_threshold` (default 1.25×) | `[HEALTH] PPL=X.XX  ratio=Y.YY× ✓ within threshold` |
| **4. Checkpoint files exist** | After each `--save_every` save | `adapter_config.json` present in `ckpt_latest/` | `[HEALTH] ✓ Checkpoint verified` |
| **5. Val loss not rising** | Every `--val_every` steps (default 50) | Val loss ≤ best × 1.10 | Health report row 3; red banner in dashboard |

### 9b. New Command-Line Flags

```bash
# NaN handling (default: stop and save checkpoint)
python train_qwen3.py ... --nan_action stop   # abort on NaN (saves checkpoint first)
python train_qwen3.py ... --nan_action skip   # discard step, continue
python train_qwen3.py ... --nan_action warn   # log only, continue

# Health report frequency
python train_qwen3.py ... --health_every 100  # print report every 100 steps

# Val loss frequency
python train_qwen3.py ... --val_every 50      # check val loss every 50 train steps
python train_qwen3.py ... --val_split 0.10    # hold out 10% of prompts for val

# PPL vs baseline
python train_qwen3.py ... --ppl_check_every 200 --ppl_threshold 1.25

# Early stopping (recommended for unattended server runs)
python train_qwen3.py ... --early_stop_patience 5
```

---

## 10. Weights & Biases Integration

### 10a. Does W&B help for training only, or also for inference / eval?

**Both**:

| Phase | What W&B tracks | Script |
|---|---|---|
| **Training** | Loss curves (train + val), PPL health, LR, GPU util, gradient histograms | `train_qwen3.py` |
| **Eval / Inference** | Alpha, block efficiency, task score, throughput per eval cell | `evaluate.py` |
| **Hyperparameter sweep** | LR × LoRA-rank × loss-type sweep | `sweep_config.yaml` |

### 10b. Setup

```bash
pip install wandb
wandb login   # enter your API key from https://wandb.ai/authorize
```

W&B is **optional** — if not installed or not logged in, training and eval continue without it.

### 10c. Training Metrics

Every training run logs to the `distillspec` W&B project:

| W&B metric | Logged every | Description |
|---|---|---|
| `train/loss` | `--log_every` steps | Rolling average training loss |
| `train/lr` | `--log_every` steps | Learning rate |
| `train/peak_vram_mb` | `--log_every` steps | Peak CUDA memory |
| `train/steps_per_sec` | `--log_every` steps | Training throughput |
| `val/loss` | `--val_every` steps | Held-out validation loss |
| `train/ppl` | `--ppl_check_every` steps | Draft model perplexity |
| `train/ppl_ratio` | `--ppl_check_every` steps | PPL / baseline |

### 10d. Eval Metrics

Each eval cell logs one point to W&B:

| W&B metric | Available in mode | Description |
|---|---|---|
| `eval/alpha_mean` | alpha | Token acceptance rate |
| `eval/block_eff` | specinfer, gbv, traversal | Average tokens per verification call |
| `eval/task_score` | alpha + `--task_score` | GSM8K exact match / HumanEval pass@1 |

### 10e. Hyperparameter Sweeps

The sweep config at `OSD/sweep_config.yaml` runs a **Bayesian sweep** over lr, lora_r, and loss type.

```bash
# Step 1 — create the sweep
wandb sweep sweep_config.yaml

# Step 2 — start an agent
wandb agent <entity>/distillspec/abc123xy
```

---

## 11. Quick Interpretation Cheatsheet

| Observation | Likely cause | Action |
|---|---|---|
| EBE alpha < KL alpha | EBE loss not converging, or LR too high | Check Training Curves tab; try lower LR |
| EBE PPL >> baseline PPL (e.g. 13 vs 9) | EBE loss pushing over-estimated tokens without KL regularizer | Ensure `kl_weight > 0` in training; re-run |
| BE gain is negative | Training made the draft *worse* | Check PPL; model may have collapsed |
| **Red banner in Training Curves tab** | **Val loss rose >10% above its minimum** | **Early stopping — the best checkpoint is the one just before the uptick** |
| Traversal doesn't beat specinfer | Draft alignment below threshold | Check alpha; plot traversal-specinfer vs alpha to find threshold |
| `NaN` in training loss | `alpha=0` in cumprod (target prob ≈ 0 for that token) | Should not happen after `clamp(min=1e-6)` fix; if recurs, lower LR |
| Charts show "No data yet" | Eval step not finished yet | Wait; check status bar for running step |
| Alpha high but throughput low | Verification overhead (traversal) dominates | Expected for large K on slow hardware; compare at K=3 |
| BE flat across K values | Draft too misaligned | EBE may not have converged; check LR and training steps |
| Colab BE > A100 BE | Quantization artifact or different n prompts | Use HW TIER filter; only compare within same tier |
| BE varies wildly across runs | Too few prompts (n=10 on colab) | Use n=30 on A100 for stable paper numbers |

---

## 12. KL Divergence Comparison

DistillSpec (Zhou et al. 2023) Section 3.1 explicitly uses **forward KL** — KL(target ∥ student).

| Direction | Formula | Property | Expected Acceptance |
|---|---|---|---|
| Forward KL | KL(target ∥ student) | Mode-covering — student must cover all target modes | Highest (baseline) |
| Reverse KL | KL(student ∥ target) | Mode-seeking — student collapses to high-confidence tokens | Lower for tail tokens |
| JSD | ½KL(s∥m)+½KL(t∥m) | Symmetric blend, bounded [0, log 2] | Between fwd and rev |
| EBE | Expected block efficiency | Directly optimizes acceptance probability | Highest overall |

**Why forward KL is right for speculative decoding:** Acceptance rate = probability that draft token matches target's sample. To maximise this, the draft must assign nonzero probability to *every* token the target could produce — exactly what forward KL enforces.

**How to run all variants:**
The pipeline runs all three KL variants as Phase 2b steps:
```bash
python experiment.py --config server --yes --from train_rev_kl_gsm8k
```

Results appear in the dashboard's Training Curves and Key Results tabs, color-coded:
- Forward KL: orange
- Reverse KL: red
- JSD: green
- EBE: blue
- Online OSD: purple

---

## 13. Online Speculative Decoding (OSD)

**Paper:** Liu et al. 2023 — https://arxiv.org/abs/2310.07177

### How it improves over offline distillation

| Mode | Training data | When it adapts |
|---|---|---|
| Offline (DistillSpec) | Fixed dataset (GSM8K train) | Once, before deployment |
| Online (OSD) | Live inference requests | Continuously, while serving |

Offline distillation trains on a fixed dataset sampled before deployment. If the actual queries differ from the training distribution, the draft model's acceptance rate will be lower. **Online OSD eliminates this gap** by updating the draft model in real time as queries arrive.

### Algorithm

```
For each incoming prompt:
  1. Run speculative decode: draft proposes K tokens, target verifies
  2. Record full sequence + positions where draft was rejected
  3. Buffer (sequence, rejection_positions)

Every update_every prompts:
  4. Re-run BOTH models on buffer (target frozen, draft trainable)
  5. Compute forward KL loss ONLY at rejection positions
  6. Update draft weights
```

**Why only rejection positions?** Accepted positions already agree with the target — their gradient is near zero. Training on rejection positions focuses the update where the draft is wrong.

### Running online adaptation

```bash
# Start from base draft (no prior training)
python online_serve.py --prompts data/gsm8k_train.jsonl --steps 500 --output checkpoints/online-gsm8k

# Start from a pre-trained offline checkpoint (recommended — faster convergence)
python online_serve.py --prompts data/gsm8k_train.jsonl --steps 500 \
    --adapter checkpoints/ebe1000-gsm8k \
    --output checkpoints/online-gsm8k
```

### Tests that prove improvement

#### 1. Wandb training curves (live, during the run)

| Metric | What it means | Expected trend |
|---|---|---|
| `online/alpha` | Rolling acceptance rate | Rises from ~0.45 → 0.55+ over 500 steps |
| `online/eval_alpha` | Acceptance rate on 30 held-out prompts | Rises monotonically; end value > baseline |
| `online/loss` | Forward KL loss at rejection positions | Decreases (model correcting its errors) |

**Pass threshold (H3):** `eval_alpha` at step 500 > `eval_alpha` at step 0.  
A successful online run should reach ≥ 0.52 on GSM8K after 500 steps starting from baseline (~0.446).

#### 2. Ranked comparison test (run after pipeline finishes Phase 2c)

```bash
python test_kl_comparison.py --n 20 --dataset data/gsm8k_train.jsonl
```

This runs GBV evaluation on all merged checkpoints and prints a ranked table.

#### 3. Dashboard comparison

After Phase 3/4 evals complete, open `http://127.0.0.1:5000/`:
- **Key Results tab** → `online` bar should be taller than `baseline` bar in the alpha chart
- **Mode Comparison tab** → select `online-gsm8k` vs `ebe1000-gsm8k` to see the gap

### KL method for online mode

Online OSD defaults to `--kl_method forward_kl`, matching the paper.

```bash
python online_serve.py --kl_method reverse_kl ...  # ablation
python online_serve.py --kl_method jsd ...          # symmetric blend
```

### In-pipeline integration

```
Phase 2  — train_kl_gsm8k, train_ebe_gsm8k    (offline, 1000 steps each)
Phase 2b — train_rev_kl_gsm8k, train_jsd_gsm8k (KL variant comparison)
Phase 2c — online_adapt_gsm8k                  (online OSD, 500 prompts, forward_kl)
Phase 3  — full eval on gsm8k (all 5 variants: baseline, kl, ebe, rev_kl, jsd, online)
Phase 4  — full eval on all datasets
```
