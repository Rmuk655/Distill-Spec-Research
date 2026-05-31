# SpecDist / Distill-Spec-Research — Publication-Grade Research & Ablation Plan

*Speculative-decoding draft-model distillation with tree-aligned loss functions.*

This plan is grounded in an audit of the actual codebase (May 2026). Every loss,
verifier, metric, dataset, and config name below is taken from source — file
paths are cited inline so claims are checkable. Hypothetical names are avoided.

---

## Part 0 — Codebase audit (ground truth)

### 0.1 Loss functions

**Flat (token-level, off-policy) — `LOSS_REGISTRY` in
`algorithms/distillspec_gbv/losses/__init__.py`:**

| name | file | status |
|---|---|---|
| `forward_kl` | `losses/forward_kl.py` | published / standard (DistillSpec) — **control baseline** |
| `reverse_kl` | `losses/reverse_kl.py` | published / standard |
| `jsd` | `losses/jsd.py` | published / standard |
| `l1` | `losses/l1.py` | standard (total-variation surrogate) |
| `ebe` | `losses/ebe.py` | experimental (expected block efficiency, off-policy) |
| `ebe_single` | `losses/ebe_single.py` | experimental |

**Tree-structured (on-policy) — `TREE_LOSS_NAMES` in
`algorithms/distillspec_gbv/losses/tree_losses.py`:**

| name | category | status | notes (from source) |
|---|---|---|---|
| `kl_tree` | generic divergence | novel (on-policy) | forward KL(p∥q) at every tree node; "on-policy baseline" |
| `rev_kl_tree` | generic divergence | novel | reverse KL(q∥p), mode-seeking |
| `jsd_tree` | generic divergence | novel | symmetric JSD, bounded gradient |
| `bv_tree` | verifier-aligned (non-OT) | novel | differentiable surrogate for E[τ_BV] |
| `gbv_tree` | verifier-aligned (non-OT) | novel — **"this work"** | GBV path-select (straight-through) + BV on q_skew |
| `traversal_tree` | verifier-aligned (non-OT) | novel | leaf-weight surrogate for E[τ_traversal] |
| `naive_tree` | verifier-aligned (OT) | novel | exact α from `naive_otlp_accept` |
| `nss_tree` | verifier-aligned (OT) | novel | exact α from `nss_otlp_accept` |
| `specinfer_tree` | verifier-aligned (OT) | novel | exact K-iteration α from `specinfer_otlp_accept` |
| `spectr_tree` | verifier-aligned (OT) | novel — **approximate** | ρ is **detached** (first-order surrogate, see `_alpha_spectr` docstring) |
| `khisti_tree` | verifier-aligned (OT) | novel — **approximate** | **LP-free softmax surrogate**, not the true LB (see `_alpha_khisti`) |
| `ebe_tree` | on-policy ablation | novel | on-policy EBE; excluded from `a100.yaml` (superseded) |

Tree losses are dispatched by `compute_tree_loss(name, …)` and are *not* in
`LOSS_REGISTRY`; the trainer (`trainer.py`) branches on `args.loss in
TREE_LOSS_NAMES`. Two OT losses (`spectr_tree`, `khisti_tree`) are explicit
mathematical approximations — flag them as such in any results table.

### 0.2 Verifiers — `TreeVerifier.verify()` in `verifiers/otlp_registry.py`

| mode | one-line behaviour | type |
|---|---|---|
| `naive` | Chen/Leviathan single-token rejection sampling (`naive_otlp_solver`) | OT, single-path |
| `bv` | Block Verification (Sun 2024c): recursive block acceptance on first path | non-OT, single-path |
| `nss` | Naive Speculative Sampling (Miao 2024): per-token OT solver | OT, multi-path |
| `spectr` | SpecTr K-SEQ (Sun 2023): ρ-scaled OT solver | OT, multi-path |
| `specinfer` | SpecInfer (Miao 2024): K-iteration residual rejection | OT, multi-path |
| `khisti` | Canonical decomposition (Khisti 2025): LP rank-tournament | OT, multi-path |
| `max` | max-ratio OT solver (utility mode) | OT |
| `traversal` | Traversal Verification (Weng 2025): bottom-up DFS leaf acceptance | non-OT, multi-path |
| `gbv` | Greedy Block Verification (**this work**): greedy path-select + skewed-q BV | non-OT, multi-path |

Reference BE ordering encoded in source (Thomas et al. 2026, arXiv:2602.16994v1):
`traversal (5.31) > spectr (4.61) ≈ specinfer (4.58) > bv (4.30) > nss (4.05)`.
`spectr/specinfer/khisti/max ≡ naive` at K=1; `traversal/gbv ≡ bv` at K=1.

### 0.3 Metrics logged — `db/results_db.py` schema

`runs` table columns: `run_tag, ts, draft_label, draft_path, target_path,
loss_name, train_steps, learning_rate, lora_rank, dataset, n_prompts, mode, K,
L, temperature, alpha_mean, alpha_std, alpha_ci95, block_eff, block_eff_std,
throughput, ms_per_tok, peak_vram_mb, task_score, perplexity, draft_latency_ms,
verify_latency_ms, notes, experiment_tag, hw_tier, seed, git_sha, wandb_url`.

`per_prompt`: `prompt_idx, category, alpha, block_eff, gen_tokens, wall_ms`.
`train_curves`: `step, loss, accept_weight, split ∈ {train,val}`.

**Variance/CI capture — critical finding:**
- **Alpha**: `alpha_std` and `alpha_ci95` *are* computed in `run_alpha()`
  (`evaluate.py`) as `std(alphas)` and `1.96·std/√n` over prompts — a normal
  approximation across prompts, **not** bootstrap, **not** across seeds.
- **Block efficiency**: `block_eff_std` column **exists but is never written**.
  `run_be()` parses a single scalar via regex (`Block efficiency…: X`) and
  `per_prompt.block_eff` is left `None` on the BE path. **BE has zero variance,
  CI, or per-prompt breakdown captured today.**
- `seed` column exists but is populated from a single fixed value.

### 0.4 Datasets — `core/datasets/downloader.py`

| dataset | eval size | role |
|---|---|---|
| `gsm8k` | configurable `n`; **full test = 1319** | primary benchmark |
| `humaneval` | 164 (all) | code generation, generalization |
| `mtbench` | 80 (all) | multi-turn chat, generalization |
| `math500` | configurable `n` | competition math, generalization |
| `alpaca` | configurable `n` | instruction following, generalization |
| `diverse50` | 50 | mixed smoke set |
| `gsm8k_train` | 7,473 (6,726 train / 747 val @ 10% split) | training |
| `alpaca_train` | 52,002 | training |

### 0.5 Configs / hardware tiers — `orchestration/configs/`

| config | draft | teacher | quant | steps | lr | warmup | grad_accum | lora r/α | max_new_tok | eval K / T / n | eval modes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `laptop` | Qwen2.5-0.5B | Qwen3-0.6B | none | 200 | 3e-5 | — | 4 (default) | 4/8 | 32 | K3 / T0.6 / 10 | alpha,bv,gbv,traversal,specinfer,naive |
| `colab_lite` | Qwen3-0.6B | Qwen3-1.7B | none | 300 | 3e-5 | — | 4 | 4/8 | 128 | K3 / T1.0 / 20 | bv,gbv |
| `colab` | Qwen3-0.6B | Qwen3-4B | none | 500 | 3e-5 | 50 | 16 | 8/16 | 96 | K3 / T1.0 / 50 | alpha,bv,gbv,traversal,specinfer,naive |
| `kaggle` | Qwen3-0.6B | Qwen3-8B | 4-bit NF4 | 1000 | 3e-5 | 100 | 16 | 8/16 | 96 | K3 / T1.0 / 100 | bv,gbv,traversal,specinfer,naive |
| `a100` | Qwen3-0.6B | Qwen3-8B | bf16 | 2000 | 3e-5 | 200 | 16 | 16/32 | 128 | K3 / T1.0 / 100 (gsm8k **1319**) | alpha,naive,nss,specinfer,spectr,khisti,bv,gbv,traversal |

Train `teacher_temperature = 0.8` across configs; eval `T=1.0` is the paper
standard. T4 tiers train all 17 losses; `a100` runs the 9-verifier matrix.
The user-facing draft↔teacher gap is **0.6B → 8B** at the top tier.

### 0.6 Existing statistical rigor — what EXISTS vs MISSING

**Exists** (`paper/analyze_results.py`):
- Welch's two-sample t-test (`welch_t_pvalue`), Cohen's d, `ci95 = 1.96·σ/√n`,
  significance stars, % change, K-sensitivity tables, cross-dataset CV.
- **But** the p-value uses a hand-rolled incomplete-beta approximation
  (`_regularized_incomplete_beta`, commented "very rough… good enough for
  p<0.05"), and the tests group **across runs by label** — with one run per cell
  the group size is 1 and the test returns `n/a`.

**Missing (must be added for publication):**
1. **Multi-seed orchestration.** `experiment.py` runs a single `seed` (default
   42, optional `seed_override`); no loop over seeds, no seed dimension in run
   identity beyond the (unused-for-aggregation) `seed` column.
2. **BE variance/CI.** `block_eff_std` and `per_prompt.block_eff` are never
   populated; the verifier prints a single mean. No way to put error bars on BE.
3. **Bootstrap.** No bootstrap anywhere in the repo.
4. **Paired tests over prompts.** No paired bootstrap / paired t-test; current
   comparison is unpaired Welch across runs.
5. **Multiple-comparison correction.** No Bonferroni/Holm despite many
   loss×verifier comparisons.
6. **Exact p-values.** No `scipy.stats` dependency; p-values are approximate.

### 0.7 W&B logging

- **Training** (`trainer.py`): `train/loss`, `train/lr`, `train/peak_vram_mb`,
  `step`; `val/loss`, `val/accept_weight`; `wandb.watch(log="gradients",
  log_freq=100)`. Run name = `{run_label}-{loss}_{family}_{steps}steps`.
- **Eval** (`evaluate.py`): `eval/{dataset}/{mode}/alpha_mean|alpha_ci95|
  throughput|ms_per_tok`, flat `eval/alpha_mean`, `eval/throughput`,
  `eval/{dataset}/task_score`, per-prompt `alpha` `wandb.Table`; BE mode:
  `eval/{dataset}/{mode}/K{K}/block_eff`, `eval/block_eff`; run summary
  `mean/max_alpha`, `mean/max_block_eff`. Training and eval share the
  `distillspec` project and group by `experiment_tag`.

---

## Section A — Research questions & hypotheses

All metrics below are columns that already exist in `runs` (Section 0.3) unless
noted. Primary metric is **block efficiency `block_eff` (BE)**; secondary are
`throughput`/`ms_per_tok`, `alpha_mean` (α), `task_score`, `perplexity`.

**RQ1 (primary) — Does an on-policy tree-aligned loss improve BE over the
flat-KL baseline?**
- **H1.** A draft trained with `kl_tree` (and/or `bv_tree`/`gbv_tree`/
  `traversal_tree`) achieves higher `block_eff` than a draft trained with flat
  `forward_kl`, evaluated under the matched verifier at K=3, T=1.0 on GSM8K.
- *Confirm if:* mean BE(tree) − BE(`forward_kl`) > 0 with non-overlapping 95%
  CIs and paired-bootstrap p < 0.05. *Refute if:* CI of the difference contains 0.

**RQ2 — Loss↔verifier alignment: is the best loss for verifier V the loss
aligned to V?** (the alignment hypothesis stated in `tree_losses.py`.)
- **H2.** For each verifier V ∈ {naive, nss, specinfer, spectr, khisti, bv, gbv,
  traversal}, the draft trained with the matching `*_tree` loss is at or near the
  top of the BE column for V (i.e. the argmax over losses of BE under V lands on
  the V-aligned loss, or within its CI).
- *Confirm if:* diagonal of the loss×verifier BE matrix dominates its column
  (within CI) for ≥ 5/8 verifiers. *Refute if:* a single loss (e.g. `kl_tree`)
  wins every column → alignment provides no specific advantage.

**RQ3 — On-policy vs off-policy: how much of the gain is the tree (on-policy)
data source vs the verifier-specific objective?**
- **H3.** BE(`ebe_tree`) > BE(flat `ebe`) (on-policy fixes the off-policy
  mismatch documented in `ebe_tree_loss`), and BE(`bv_tree`) ≥ BE(`ebe_tree`)
  (full-vocab block integral adds further gain). This is the ablation ladder
  already written into the `ebe_tree` docstring.

**RQ4 — K-sensitivity.** **H4.** BE is monotonically non-decreasing from K=3 to
K=5 for multi-path verifiers (nss, spectr, specinfer, khisti, traversal, gbv);
gbv is expected to degrade earliest because `compute_skew()` is documented
numerically unstable for K>4.

**RQ5 — Temperature-sensitivity.** **H5.** Relative loss ranking by BE is stable
between T=0.6 and T=1.0 (T affects absolute acceptance but not which loss wins).

**RQ6 — Generalization.** **H6.** A BE improvement found on GSM8K holds on ≥1 of
{humaneval, mtbench, math500, alpaca} (sign of Δ preserved, p < 0.05 on ≥1).

**RQ7 — Quality preservation.** **H7.** Tree-loss drafts do not regress
`task_score` (GSM8K exact-match / HumanEval pass@1) or `perplexity` by more than
the existing guard thresholds (>5pp task drop / >10% PPL rise in `evaluate.py`).

---

## Section B — Ablation matrix

### B.1 Factors × levels

| factor | levels | source of names |
|---|---|---|
| **loss** | control `forward_kl`; off-policy `ebe`; on-policy generic `kl_tree`, `rev_kl_tree`, `jsd_tree`; verifier-aligned non-OT `bv_tree`, `gbv_tree`, `traversal_tree`; verifier-aligned OT `naive_tree`, `nss_tree`, `specinfer_tree`, `spectr_tree`, `khisti_tree`; on-policy ablation `ebe_tree` | `TREE_LOSS_NAMES`, `LOSS_REGISTRY` |
| **verifier (eval mode)** | `naive`, `bv`, `nss`, `spectr`, `specinfer`, `khisti`, `traversal`, `gbv` (+ `alpha`) | `TreeVerifier.verify()` |
| **K** | {3, 5} | configs use 3; user adds 5 |
| **temperature** | T=1.0 (standard); T=0.6 (robustness, Week 7) | eval defaults |
| **dataset** | `gsm8k` (primary, 1319); `humaneval` (164), `mtbench` (80), `math500`, `alpaca` (generalization) | `downloader.py` |

### B.2 Control / baseline

- **Control loss:** flat `forward_kl` — the standard DistillSpec objective and
  the only loss every prior speculative-distillation paper uses.
- **Control verifier per comparison:** the verifier *matched* to the loss under
  test (diagonal cell), plus `naive`/`bv` as universal references. At K=1 all OT
  verifiers collapse to `naive` and `gbv/traversal` collapse to `bv`, so K=1 is
  used only as an internal sanity equivalence check, not a research cell.

### B.3 One-factor-at-a-time vs full grid

- **Weeks 2–4 (T4): one-factor-at-a-time (OFAT).** Vary loss with verifier fixed
  to the matched mode, K=3, T=1.0, GSM8K only. Cheap, isolates RQ1/RQ3.
- **Week 5 (A100): the loss×verifier diagonal + full column for RQ2.** This is a
  selective grid, not the full 13×8 = 104 cell cross-product — only the
  promoted losses (≈5–6) × all 8 verifiers.
- **Full 13×8 cross-pair grid is "extended"** (nice-to-have for the appendix),
  run only if A100 budget allows after the core result lands.

### B.4 Core (must-run) vs extended (nice-to-have)

**Core losses (8):** `forward_kl` (control), `ebe`, `kl_tree`, `bv_tree`,
`gbv_tree`, `traversal_tree`, `specinfer_tree`, `nss_tree`.
**Core verifiers (5):** `naive`, `bv`, `specinfer`, `traversal`, `gbv`.
**Core dataset:** GSM8K. **Core K/T:** K∈{3,5}, T=1.0. **Core seeds:** 3.

**Extended:** `rev_kl_tree`, `jsd_tree`, `naive_tree`, `spectr_tree`,
`khisti_tree`, `ebe_tree`, flat `reverse_kl`/`jsd`/`l1`/`ebe_single`; verifiers
`nss`, `spectr`, `khisti`; datasets humaneval/mtbench/math500/alpaca; T=0.6;
K=8; full 13×8 grid.

### B.5 Run-count & GPU-hour estimates

Assumptions (from configs): T4 training ≈ 1–2 s/step flat, ~2–4× for tree losses
(builds a K-path tree + 3 model passes per step, see `_tree_training_step`);
A100 ≈ 0.3–0.5 s/step flat. BE eval cost scales with `n_prompts × max_new_tokens`
and verifier complexity (traversal/gbv slowest). These are planning estimates,
not measurements — calibrate after Week 1.

**Training runs**

| phase | hw | losses × seeds | steps | ≈ h/run | subtotal |
|---|---|---|---|---|---|
| Wk2 convergence | T4 | 2 × 2 | 1000 | ~1.0 | ~4 h |
| Wk3 loss ablation | T4 | 13 × 1 | 1000 | ~1.0 | ~13 h |
| Wk5 scale top set | A100 | 6 × 3 = 18 | 2000 | ~0.6 | ~11 h |
| **training total** | | **~40 checkpoints** | | | **~28 GPU-h** |

**Eval cells** (1 cell = checkpoint × dataset × verifier × K × T)

| phase | hw | cells | ≈ subtotal |
|---|---|---|---|
| Wk1 baseline smoke | T4 | ~10 | ~2 h |
| Wk3 loss ablation (GSM8K n≈100, 5 verifiers, K3) | T4 | 13×5 ≈ 65 | ~10 h |
| Wk4 verifier×loss (top 6 × 8 verifiers, K3) | T4 | ~48 | ~10 h |
| Wk5 core grid (18 ckpt × 8 verifiers × K∈{3,5}, GSM8K n=1319) | A100 | ~288 | ~55 h |
| Wk6 generalization (top 12 ckpt × 4 datasets × 8 verifiers, K3) | A100 | ~384 | ~30 h |
| Wk7 K/T sensitivity (top 6 ckpt × {K3,5(,8)} × {T0.6,1.0}, GSM8K n≈500) | A100 | ~72 | ~20 h |
| **eval total** | | **~850 cells** | |

**Headline budget (estimates, ±40%):**
- **T4 (Kaggle/Colab, exploration): ~50 GPU-hours** total across Weeks 1–4
  (≈12 h/week — comfortably inside Kaggle's ~30 h/week quota and 9 h sessions).
- **A100 (final + generalization + sensitivity): ~115 GPU-hours** across Weeks
  5–7. The single biggest line is Week-5 GSM8K@1319 BE eval (~55 h) because BE
  has no batching across checkpoints and traversal/gbv are slow at n=1319.
- **Total program: ~165 GPU-hours, ~40 training checkpoints, ~850 eval cells.**

Cost lever: the full 13×8 extended grid roughly *doubles* A100 eval. Keep it for
the appendix and gate it on the core result being significant.

---

## Section C — Statistical rigor protocol

### C.1 Seeds per cell

- **≥ 3 seeds** for every cell that enters a published number (e.g. 42, 123, 7).
  Rationale: 3 is the minimum that yields a non-degenerate t-distribution
  (df=2) and a meaningful across-seed σ; LoRA distillation has real seed
  variance (init + data-shuffle order via `_epoch_prompts(seed+epoch)`).
  T4 exploration (Weeks 3–4) may use 1 seed to rank; promotion to A100 requires 3.

### C.2 How to report

- **Per cell:** mean ± 95% CI. Two CI sources, reported separately:
  1. **Across-seed CI** (n=3): t-distribution, `mean ± t_{0.975,2}·s/√3`
     (t≈4.30). This is the headline error bar for "is the loss better".
  2. **Within-run across-prompt CI** (n=prompts): **bootstrap** the per-prompt
     metric (10,000 resamples, percentile 2.5/97.5). Requires populating
     `per_prompt.block_eff` first (see C.6).
- Tables show `BE = mean ± CI (n_seeds=3)`; figures show across-seed error bars.

### C.3 Significance tests — which test for which comparison

| comparison | unit | test |
|---|---|---|
| tree loss vs `forward_kl`, same verifier/K/T/dataset | **per-prompt BE**, same prompts | **paired bootstrap** over prompts on the BE difference (10k resamples); report Δ and its 95% CI / p. This is the primary RQ1 test. |
| tree loss vs `forward_kl`, aggregate across seeds | seed means (n=3) | **Welch's t-test** across seeds (small-n; report exact p via `scipy.stats.ttest_ind(equal_var=False)`, not the repo's beta approximation). |
| loss A vs loss B (verifier ranking, RQ2) | per-prompt BE | paired bootstrap per verifier column |
| K=3 vs K=5 (RQ4) | per-prompt BE, same prompts | paired bootstrap (paired by prompt) |
| effect size | — | report **Cohen's d** (`analyze_results.py` already computes it) alongside every p. |

Paired-over-prompts is strongly preferred over unpaired Welch for the primary
claim: the same prompts are evaluated under both losses, and pairing removes
prompt-difficulty variance (the dominant noise source at small n).

### C.4 Multiple-comparison correction

The RQ2 matrix is ≈ 6 losses × 8 verifiers = 48 simultaneous comparisons. Apply
**Holm–Bonferroni** (less conservative than plain Bonferroni, no independence
assumption) to the family of p-values within each results table. Report both raw
and Holm-adjusted p; call a result "significant" only on the adjusted value.
For the single pre-registered primary comparison (RQ1: best tree loss vs
`forward_kl` on GSM8K), no correction is needed — it is one planned test.

### C.5 Minimum sample size

- Current configs: T4 `n_prompts` = 10 (laptop) / 20 (colab_lite) / 50 (colab) /
  100 (kaggle); A100 = 100 secondary, **1319 (full GSM8K) primary**.
- **Recommendation:** T4 exploration uses n=50–100 (enough to *rank*, not to
  publish). **Final A100 GSM8K numbers must use the full 1319-prompt test set**
  (`eval_n_prompts_gsm8k: 1319` already in `a100.yaml`). Generalization datasets
  use their full size (humaneval 164, mtbench 80) and n≈300–500 for math500/alpaca.
- A paired-bootstrap on 1319 prompts resolves BE differences of ~0.1 block;
  n=50 resolves only ~0.4–0.5 — adequate for ranking, not for the final delta.

### C.6 MISSING-in-code checklist (to add — **not implemented in this plan**)

- [ ] **Multi-seed runner**: loop `experiment.py` over a `seeds: [42,123,7]`
      list; tag each run with its `seed` in `runs.seed` (column already exists).
- [ ] **Populate `block_eff_std`** and **`per_prompt.block_eff`**: make
      `runner.py` emit per-prompt BE (not just the mean), parse it in
      `run_be()`/`run_be_batch()`, and write both columns.
- [ ] **Bootstrap module**: `paper/stats.py` with `paired_bootstrap(a, b,
      n=10000)` returning Δ, CI, p — operating on per-prompt arrays.
- [ ] **Replace approximate p-value**: add `scipy` and swap
      `_regularized_incomplete_beta` / `welch_t_pvalue` for `scipy.stats`.
- [ ] **Holm–Bonferroni** helper applied to each comparison family.
- [ ] **Seed-aware aggregation** in `analyze_results.py`: group by
      (loss, verifier, K, T, dataset) across seeds before testing, instead of
      grouping by `draft_label` only.
- [ ] **Pre-registration**: freeze RQ1's test (loss, verifier, dataset, n, seeds)
      before looking at A100 numbers.

---

## Section D — Week-by-week execution plan (8 weeks)

Hardware key: **T4** = Kaggle/Colab free (quota-limited, 9 h sessions), **A100** =
occasional Pro/AIP access. Each week has a quantitative **EXIT** gate.

### Week 1 — Pipeline validation + baseline sanity (T4)
- **Objective:** clean end-to-end run; confirm the untrained-baseline BE is in a
  sane range before training anything.
- **Runs:** `experiment.py --config laptop --smoke` then `--config kaggle`
  baseline-only (no training) on GSM8K n=50, modes `bv,gbv,traversal,specinfer,
  naive`, K=3, T=1.0.
- **Inspect:** the `[run_tag] block_eff=…` lines; `peak_vram_mb`; PPL/task_score
  guard warnings; W&B `eval/block_eff`.
- **Sanity targets:** for K=3 the **theoretical max BE is L+1 = 9** (`L=8`);
  state-of-art Qwen-class drafts land ≈ **3** at K=3. Untrained 0.6B→8B baseline
  is expected ~1.5–2.5. traversal ≥ bv ≈ gbv ordering should roughly hold.
- **EXIT:** end-to-end run completes with no exceptions; baseline BE finite and
  within ~[1.3, 3.0]; no OOM; results land in `results.db`.

### Week 2 — Single-loss convergence study (T4)
- **Objective:** establish that training works and pick LR/warmup/grad_accum from
  curves; compare `kl_tree` vs flat `forward_kl`.
- **Runs:** train `forward_kl` and `kl_tree`, **2 seeds each**, 1000 steps,
  kaggle config (grad_accum 16, warmup 100, lr 3e-5). Watch `train/loss`,
  `val/loss` (`val_every=50`), `accept_weight`.
- **Inspect:** train vs val loss curves in the dashboard (`train_curves`,
  `split` train/val); look for the **plateau seen earlier at best≈0.3766** and
  any val-loss bounce after ~step 150 (the reason warmup was added).
- **Tune:** if val loss bounces, lower LR or lengthen warmup; if train loss is
  flat, raise LR or reduce grad_accum; if val ≫ train, add LoRA dropout.
- **EXIT:** val-loss plateau detected (no improvement for `early_stop_patience`
  checks) **and** the two seeds agree (final val-loss within ~10%).

### Week 3 — Loss-function ablation (T4, cheap subset)
- **Objective:** rank all 13 losses by BE improvement over `forward_kl`.
- **Runs:** train each loss × 1 seed (1000 steps); eval GSM8K n=50–100, matched
  verifier + `bv`/`naive`, K=3, T=1.0.
- **Inspect:** BE-improvement bar chart (Δ vs `forward_kl`); training warnings;
  drop any loss that NaNs or collapses task_score.
- **EXIT:** **top-3 losses identified with 95% CIs (across the n prompts) that do
  not overlap the `forward_kl` baseline.** If none separate at n=100, raise n
  before promoting.

### Week 4 — Verifier × loss interaction (T4)
- **Objective:** find the best loss×verifier pairing (RQ2 dry run).
- **Runs:** reuse Week-3 checkpoints for the top ~6 losses; eval all 8 verifiers,
  K=3, T=1.0, GSM8K n=100. (No new training.)
- **Inspect:** the loss×verifier BE heatmap; check whether the diagonal
  (V-aligned loss best under V) shows up; note `spectr_tree`/`khisti_tree`
  (approximate) behaviour.
- **EXIT:** a best pairing per verifier identified and a short-list of ≤6 losses
  promoted to A100.

### Week 5 — Scale to A100, full GSM8K (A100) — **primary result**
- **Objective:** statistically rigorous BE comparison.
- **Runs:** train promoted losses + `forward_kl` control, **3 seeds**, 2000 steps,
  a100 config; eval **GSM8K n=1319**, 8 verifiers, **K∈{3,5}**, T=1.0.
- **Inspect:** mean±CI BE table (across-seed) + paired-bootstrap Δ vs control;
  throughput-vs-BE frontier; task_score/PPL guards.
- **EXIT (the gate that defines success):** **paired-bootstrap p < 0.05 (and
  non-overlapping across-seed CIs) for ≥1 tree loss beating `forward_kl` on BE**
  under its matched verifier on full GSM8K.

### Week 6 — Generalization (A100)
- **Objective:** test H6 on other datasets.
- **Runs:** reuse Week-5 checkpoints (top 3 losses + control, 3 seeds); eval
  humaneval (164), mtbench (80), math500 (n≈300), alpaca (n≈300), 8 verifiers,
  K=3, T=1.0.
- **Inspect:** per-dataset Δ-BE with error bars; cross-dataset CV
  (`analyze_datasets`); confirm sign of improvement is preserved.
- **EXIT:** improvement holds (sign preserved, p<0.05) on **≥1** additional
  dataset.

### Week 7 — Tuning / robustness + paper tables (A100)
- **Objective:** K-sensitivity (RQ4), temperature-sensitivity (RQ5), final tables.
- **Runs:** top 2 losses + control, 3 seeds; K∈{3,5(,8)} × T∈{0.6,1.0}, GSM8K
  n≈500.
- **Inspect:** BE-vs-K and BE-vs-T line plots; check gbv degradation at K>4
  (`compute_skew` instability); confirm ranking stable across T.
- **EXIT:** complete, CI-annotated results tables for every RQ; ranking shown
  stable (or its instability characterized).

### Week 8 — Write-up, figures, reproducibility
- **Objective:** generate all figures from `results.db`; reproducibility check.
- **Runs:** re-run `analyze_results.py` (after the C.6 stats upgrades) →
  Markdown tables; regenerate plots; re-train **one** cell from scratch and
  confirm BE within CI of the recorded value; verify `git_sha`/`seed`/`wandb_url`
  are logged for every published run.
- **EXIT:** every number in the paper traces to a `run_tag` + `git_sha`; a
  re-run reproduces within CI.

---

## Section E — Graphs to inspect and how to read them

1. **Train & val loss curves** (`train_curves`, dashboard).
   - *Good:* both fall then plateau; val tracks train within a small gap.
   - *Wrong:* val rises after falling (overfit / LR too high → shorten run, lower
     LR, add dropout); train flat from the start (LR too low / grad_accum too
     high → raise LR); flat plateau stuck near the **earlier best≈0.3766**
     (signal saturated → try a different loss or larger teacher gap).

2. **BE-improvement bar chart per loss** — Δ`block_eff` vs `forward_kl`, with
   **across-seed error bars**.
   - *Good:* bar clears 0 and its CI excludes 0.
   - *Wrong:* CI straddles 0 → not significant (increase n or seeds, do not
     report as a win).

3. **BE vs K line plot** (K∈{3,5,8}).
   - *Good:* non-decreasing for multi-path verifiers.
   - *Wrong:* gbv BE drops at K>4 → expected (`compute_skew` instability); cap
     gbv at K≤4 and footnote it.

4. **BE vs temperature line plot** (T∈{0.6,1.0}).
   - *Good:* absolute BE shifts but loss ranking is preserved.
   - *Wrong:* ranking flips with T → temperature confounds the claim; report per-T.

5. **Throughput vs BE scatter** (`throughput`/`ms_per_tok` vs `block_eff`) — the
   speed/quality frontier.
   - *Good:* tree-loss drafts sit up-and-right (higher BE at equal/again better
     throughput).
   - *Wrong:* high BE but low throughput → verifier overhead eats the gain
     (especially traversal/gbv); report wall-clock speedup, not BE alone.

6. **Alpha (acceptance-rate) distribution** (`per_prompt.alpha`, `alpha_mean ±
   alpha_ci95`).
   - *Good:* distribution shifts right vs baseline; α rises with BE.
   - *Wrong:* α up but BE flat → verifier/temperature mismatch between α-eval and
     BE-eval (the `analyze_quality_flags` "BE/ALPHA MISMATCH" check); align T.

---

## Section F — Risks & mitigations

| risk | evidence in code | mitigation |
|---|---|---|
| **OOM on T4** loading 8B teacher | kaggle config uses 4-bit NF4; `colab.yaml` drops to 4B because NF4's CPU-RAM spike SIGKILLs Colab's 12 GB | use `kaggle` (29 GB RAM) for 8B-NF4; `colab` (4B bf16) otherwise; the OOM→CPU retry in `run_alpha`/`run_be` is a slow fallback, not a fix — prefer the right config |
| **Orphaned eval subprocesses** exhausting VRAM on re-run | `run_be_batch` kills stray `runner.py` via psutil; `expandable_segments:True` | keep psutil installed; one BE subprocess per dataset (already batched) |
| **T4 timeouts**, esp. traversal K=5 | `run_be`/`run_be_batch` have 30 min / 2 h timeouts; partial results are recovered and `--skip_existing` resumes | on T4 cap K=3 for traversal/gbv; push K=5 to A100; rely on partial-recovery + resume |
| **Kaggle checkpoint loss** (ephemeral `/kaggle/working`) | configs set `storage_root`; trainer writes `ckpt_latest`/`ckpt_best` + `training_state.json` for crash-safe resume | always pass `--storage_root`; mirror to Drive/persistent volume; `save_every=25` on T4 |
| **Small-sample noise** | T4 n=10–100; BE has no CI today | rank on T4, *decide* only on A100 n=1319 with ≥3 seeds + paired bootstrap (Section C) |
| **Teacher-model appropriateness** (0.6B draft vs 8B teacher; laptop's 0.5B↔0.6B is pure code-check) | laptop.yaml warns its numbers are "meaningless" | never interpret laptop/colab_lite numbers as findings; publish only 8B-teacher (kaggle/a100) results |
| **Approximate losses misread as exact** | `spectr_tree` (ρ detached), `khisti_tree` (LP-free) are documented surrogates | label both "approximate" in every table; if either underperforms its aligned verifier, attribute to the surrogate, not the alignment hypothesis (per their docstrings) |
| **Stats invalid by construction** | single seed, no BE variance, approximate p-values (Section 0.6) | implement the Section C.6 checklist *before* Week 5; do not publish from single-seed BE means |
| **gbv numerical instability at K>4** | `compute_skew` docstring: "not recommended … K>4" | restrict gbv/gbv_tree to K≤4; footnote |
