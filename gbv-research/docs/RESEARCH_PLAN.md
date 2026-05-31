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

**The RQ order matches the locked iteration priority (Section D): flat baseline →
tree loss → GBV verifier → online variants.** Nothing downstream is interpretable
until RQ1 (the trusted flat baseline) is established.

**RQ1 (foundation, PRIORITY 1) — Is the published flat-loss baseline
(`forward_kl`) reproduced at a trusted, expected BE on the real A100 setup?**
- **H1.** A draft distilled with flat `forward_kl` (the standard DistillSpec
  objective) on the 0.6B→8B pair reaches a stable BE in the expected range under
  the standard verifiers (`naive`/`bv`) at K=3, T=1.0 on GSM8K — and matches the
  published-method order of magnitude (SOTA Qwen-class ≈ 3 at K=3; L+1=9 max).
- *Confirm if:* `forward_kl` BE is finite, reproducible across seeds (across-seed
  CI tight), and within the expected band; this becomes the **reference number**
  every later claim is measured against. *Refute if:* the baseline is unstable or
  implausibly low/high → fix the pipeline before any novel work (the foundation
  is invalid otherwise).

**RQ2 (PRIORITY 2 — core novel contribution) — Does on-policy tree-loss training
improve BE over the flat `forward_kl` baseline from RQ1?**
- **H2.** A draft trained with a tree loss (`kl_tree` first; then `bv_tree`/
  `gbv_tree`/`traversal_tree`) achieves higher `block_eff` than the RQ1
  `forward_kl` baseline, evaluated under the matched verifier at K=3, T=1.0 on
  GSM8K.
- *Confirm if:* mean BE(tree) − BE(`forward_kl`) > 0 with non-overlapping 95%
  CIs and paired-bootstrap p < 0.05. *Refute if:* CI of the difference contains 0.

**RQ3 (PRIORITY 2, supporting) — On-policy vs off-policy: how much of the
tree-loss gain is the on-policy data source vs the verifier-specific objective?**
- **H3.** BE(`ebe_tree`) > BE(flat `ebe`) (on-policy fixes the off-policy
  mismatch documented in `ebe_tree_loss`), and BE(`bv_tree`) ≥ BE(`ebe_tree`)
  (full-vocab block integral adds further gain). This is the ablation ladder
  already written into the `ebe_tree` docstring.

**RQ4 (PRIORITY 3 — verifier-side contribution) — Loss↔verifier alignment, and
does the GBV verifier pay off?** (the alignment hypothesis in `tree_losses.py`.)
- **H4.** For each verifier V ∈ {naive, nss, specinfer, spectr, khisti, bv, gbv,
  traversal}, the draft trained with the matching `*_tree` loss is at or near the
  top of the BE column for V; and **`gbv` verification yields higher BE than
  single-path `bv`** for a tree-loss draft at K≤4.
- *Confirm if:* the loss×verifier BE diagonal dominates its column (within CI) for
  ≥ 5/8 verifiers, **and** BE(`gbv`) > BE(`bv`) for the best tree-loss draft.
  *Refute if:* a single loss wins every column (alignment adds nothing) or `gbv`
  does not beat `bv`.
- *Gating:* pursued **only after RQ2 shows tree-loss promise** (priority order).

**RQ5 (PRIORITY 4 — lowest, deferred) — Do online / online-tree variants
(`online_kl_tree`, `online_ebe_tree`, …) add BE over the best offline tree loss?**
- **H5.** An online-adapted draft beats its best offline tree-loss counterpart on
  BE under the matched verifier. *Confirm if:* Δ>0 with non-overlapping CIs.
  *Refute if:* CI contains 0. **Deferred to the very end** — only attempted if
  RQ2/RQ4 land and budget remains (online code is currently disabled pending a
  smoke-verification, per `kaggle.yaml` notes).

**RQ6 — K-sensitivity.** **H6.** BE is monotonically non-decreasing from K=3 to
K=5 for multi-path verifiers (nss, spectr, specinfer, khisti, traversal, gbv);
gbv is expected to degrade earliest because `compute_skew()` is documented
numerically unstable for K>4.

**RQ7 — Temperature-sensitivity.** **H7.** Relative loss ranking by BE is stable
between T=0.6 and T=1.0 (T affects absolute acceptance but not which loss wins).

**RQ8 — Generalization.** **H8.** A BE improvement found on GSM8K holds on ≥1 of
{humaneval, mtbench, math500, alpaca} (sign of Δ preserved, p < 0.05 on ≥1).

**RQ9 — Quality preservation.** **H9.** Tree-loss drafts do not regress
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
| **temperature** | T=1.0 (standard); T=0.6 (robustness direction-check, Week 6) | eval defaults |
| **dataset** | `gsm8k` (primary, 1319); `humaneval` (164), `mtbench` (80), `math500`, `alpaca` (generalization) | `downloader.py` |

### B.2 Control / baseline

- **Control loss:** flat `forward_kl` — the standard DistillSpec objective and
  the only loss every prior speculative-distillation paper uses.
- **Control verifier per comparison:** the verifier *matched* to the loss under
  test (diagonal cell), plus `naive`/`bv` as universal references. At K=1 all OT
  verifiers collapse to `naive` and `gbv/traversal` collapse to `bv`, so K=1 is
  used only as an internal sanity equivalence check, not a research cell.

### B.3 One-factor-at-a-time vs full grid

- **Weeks 2–5 (Kaggle P100): one-factor-at-a-time (OFAT), 1 seed, n=50.** Vary
  loss with verifier fixed to the matched mode, K=3, T=1.0, GSM8K only. Cheap,
  isolates RQ1–RQ3 for *direction*; Week 5 widens to the verifier columns for RQ4.
- **A100 confirmation: only the 1–2 promoted loss×verifier pairings + control.**
  Not a grid — the funnel has already collapsed the matrix to its finalists.
- **Full 13×8 cross-pair grid is "extended"** (appendix-only), deferred to a
  later Lightning-credit month if the core result lands — never on the $30 Modal
  critical path.

### B.4 Core (must-run) vs extended (nice-to-have)

**Core losses (8):** `forward_kl` (control), `ebe`, `kl_tree`, `bv_tree`,
`gbv_tree`, `traversal_tree`, `specinfer_tree`, `nss_tree`.
**Core verifiers (5):** `naive`, `bv`, `specinfer`, `traversal`, `gbv`.
**Core dataset:** GSM8K. **Core K/T:** K∈{3,5}, T=1.0. **Core seeds:** 3.

**Extended:** `rev_kl_tree`, `jsd_tree`, `naive_tree`, `spectr_tree`,
`khisti_tree`, `ebe_tree`, flat `reverse_kl`/`jsd`/`l1`/`ebe_single`; verifiers
`nss`, `spectr`, `khisti`; datasets humaneval/mtbench/math500/alpaca; T=0.6;
K=8; full 13×8 grid.

### B.5 The free-tier funnel (hard budget reframe)

This is a **strict free-tier project**. There is no paid A100 pool. The earlier
~165 GPU-hour plan (incl. ~115 A100-h) is **not affordable** and is replaced by
the funnel below.

**Compute reality (hard limits):**

| resource | budget | role | reliability |
|---|---|---|---|
| **Kaggle free** | **~30 GPU-h / week, resets weekly** | **ALL exploration** (rank losses, kill losers, tune LR) | the only reliable compute |
| Modal | **$30 one-time signup credit** (not yet signed up) | **ONE A100 confirmation run, spent last** | one-shot, irreplaceable |
| Lightning | ~15–22 free credits/mo | backup A100 if Modal runs short | secondary |
| Colab free | **already exhausted** | throwaway / backup only — **never on the critical path** | unreliable |

> ### ⚠️ MANDATE: on Kaggle use **P100 (single 16 GB GPU)** or **T4×1** — **NEVER T4×2.**
> Quota burns **per GPU-wall-hour**: P100 and T4×1 burn **1 quota-h per wall-h**;
> **T4×2 burns 2× the quota for the same work.** Every model in this project
> (Qwen3-0.6B draft + Qwen3-8B teacher in 4-bit NF4 ≈ 7.8 GB, see `kaggle.yaml`)
> fits in a single 16 GB GPU, so the second GPU buys nothing and doubles the
> burn rate. **Select "P100" (preferred) or "GPU T4 x1" in Kaggle notebook
> settings. If you see T4×2, change it before running anything.**

**The funnel:**

```
Kaggle P100 (exploration, ~30 GPU-h/wk × ~6 wk)        Modal A100 (confirm, ONCE)
┌────────────────────────────────────────────┐        ┌──────────────────────────┐
│ 13 losses → rank → kill losers → tune LR     │  ───►  │ top 1–2 loss×verifier      │
│ n=50, K=3, T=1.0, 1 seed (DIRECTION only)    │        │ full GSM8K n=1319          │
│ matched verifier + bv/naive refs             │        │ 3 seeds, K∈{3,5}           │
│ exit: ≤2 candidates that beat forward_kl     │        │ → the ONLY paper numbers   │
└────────────────────────────────────────────┘        └──────────────────────────┘
```

- **Kaggle P100 = exploration.** Everything that *ranks* losses and tunes
  hyperparameters happens here, cheaply: n=50 prompts, K=3 only, T=1.0, **1 seed**.
  These numbers establish **direction only** — they decide what to promote, not
  what goes in the paper.
- **Modal A100 = confirmation, once.** After T4 narrows to ≤2 loss×verifier
  pairings, spend the $30 credit on a **single** rigorous run: full GSM8K
  (n=1319), 3 seeds, K∈{3,5}. These are the only publication numbers.

Assumptions (from configs): P100 training ≈ 1.5–3 s/step (slower than T4 for
matmul but fits 16 GB); tree losses ~2–4× flat (K-path tree + 3 model passes per
step, see `_tree_training_step`); A100 ≈ 0.3–0.6 s/step. BE eval scales with
`n_prompts × max_new_tokens` × verifier complexity (traversal/gbv slowest).
Calibrate after Week 1 and adjust the per-week ceiling.

### B.6 Frugality rules (non-negotiable on Kaggle)

1. **P100 or T4×1, never T4×2** (see mandate above) — 1 quota-h/wall-h, not 2×.
2. **n=50 prompts** for all Kaggle eval (not 100). Enough to rank, half the cost.
3. **K=3 only** on Kaggle. K=5 is deferred entirely to the A100 confirmation.
4. **1 seed** for Kaggle ranking. Multi-seed is an A100-only luxury.
5. **`--skip_existing` always.** Every restart resumes; never recompute a cell
   already in `results.db`.
6. **Religious Cell-2b checkpoint backup.** `/kaggle/working` is ephemeral; a
   lost session = wasted *irreplaceable* weekly quota. Always pass
   `--storage_root`, mirror `ckpt_latest`/`ckpt_best` + `results.db` to a Kaggle
   Dataset (or Drive) at session end, `save_every=25`.
7. **Kill losers immediately.** A loss that doesn't beat `forward_kl` direction
   at n=50 after a full 1000-step train is dropped — do not spend a second week
   on it. One hypothesis-batch per week; protect the ceiling.
8. **One 9-hour session ≈ 9 quota-h.** Plan each week as ≤3 sessions so a single
   crash costs ≤⅓ of the week's budget.

---

## Section C — Statistical rigor protocol

### C.1 Seeds per cell

- **≥ 3 seeds** for every cell that enters a published number (e.g. 42, 123, 7).
  Rationale: 3 is the minimum that yields a non-degenerate t-distribution
  (df=2) and a meaningful across-seed σ; LoRA distillation has real seed
  variance (init + data-shuffle order via `_epoch_prompts(seed+epoch)`).
- **Budget split (important):** under the free-tier funnel (B.5) all
  multi-seed work happens **only in the single A100 confirmation run**. **Kaggle
  exploration uses 1 seed** — those numbers establish **direction only** and are
  never reported as results. The CI / paired-bootstrap / Holm–Bonferroni
  machinery below applies exclusively to the A100 confirmation numbers, which are
  the only ones that go in the paper.

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
| tree loss vs `forward_kl`, same verifier/K/T/dataset | **per-prompt BE**, same prompts | **paired bootstrap** over prompts on the BE difference (10k resamples); report Δ and its 95% CI / p. This is the primary RQ2 test (vs the RQ1 baseline). |
| tree loss vs `forward_kl`, aggregate across seeds | seed means (n=3) | **Welch's t-test** across seeds (small-n; report exact p via `scipy.stats.ttest_ind(equal_var=False)`, not the repo's beta approximation). |
| loss A vs loss B (verifier ranking, RQ4) | per-prompt BE | paired bootstrap per verifier column |
| K=3 vs K=5 (RQ6) | per-prompt BE, same prompts | paired bootstrap (paired by prompt) |
| effect size | — | report **Cohen's d** (`analyze_results.py` already computes it) alongside every p. |

Paired-over-prompts is strongly preferred over unpaired Welch for the primary
claim: the same prompts are evaluated under both losses, and pairing removes
prompt-difficulty variance (the dominant noise source at small n).

### C.4 Multiple-comparison correction

The RQ4 matrix is ≈ 6 losses × 8 verifiers = 48 simultaneous comparisons. Apply
**Holm–Bonferroni** (less conservative than plain Bonferroni, no independence
assumption) to the family of p-values within each results table. Report both raw
and Holm-adjusted p; call a result "significant" only on the adjusted value.
For the single pre-registered primary comparison (RQ2: best tree loss vs the RQ1
`forward_kl` baseline on GSM8K), no correction is needed — it is one planned test.

### C.5 Minimum sample size

- Current configs: T4 `n_prompts` = 10 (laptop) / 20 (colab_lite) / 50 (colab) /
  100 (kaggle); A100 = 100 secondary, **1319 (full GSM8K) primary**.
- **Free-tier recommendation:** **Kaggle exploration uses n=50** (frugality rule
  B.6.2 — enough to *rank*, half the cost of 100, never published). **The single
  A100 confirmation uses the full 1319-prompt GSM8K test set**
  (`eval_n_prompts_gsm8k: 1319` already in `a100.yaml`).
- A paired-bootstrap on 1319 prompts resolves BE differences of ~0.1 block;
  n=50 resolves only ~0.4–0.5 — adequate for ranking direction, not for the
  final delta. This is exactly why n=50 stays on Kaggle and 1319 is reserved for
  the one A100 run.

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
- [ ] **Pre-registration**: freeze RQ2's primary test (loss, verifier, dataset, n, seeds)
      before looking at A100 numbers.

---

## Section D — Quota-bounded execution plan (≈6 Kaggle weeks + staged A100 confirmation)

> ### 🔒 Iteration priority (LOCKED)
> The algorithm is **not final** and will need **multiple iterations** — this plan
> is a **loop, not a one-shot**. Every iteration proceeds in this strict order,
> and **each step gates the next**:
>
> 1. **FIRST — Flat-baseline benchmark (`forward_kl`).** Establish a solid,
>    trusted published-loss baseline. **The first real A100 (Modal) spend produces
>    the `forward_kl` baseline numbers** (full GSM8K where budget allows). Nothing
>    else is meaningful until this is reproduced in the expected BE range.
> 2. **SECOND — Tree-loss training** (`kl_tree`, `bv_tree`, `gbv_tree`, …), the
>    core novel contribution, compared against the step-1 baseline.
> 3. **THIRD — GBV verifier** (verifier-side contribution) — only after tree-loss
>    training shows promise.
> 4. **LAST — online / online-tree variants** (`online_kl_tree`,
>    `online_ebe_tree`, …) — lowest priority, deferred to the very end.
>
> Do not advance a step until the previous step's EXIT gate passes. On each loop
> iteration (algo revision) re-enter at the earliest step the change affects.

**Cadence rule:** Kaggle resets ~30 GPU-h/week. Every week is **one
hypothesis-batch with a hard ≤30 GPU-h ceiling** (target ≤27 h to keep a safety
buffer for a crashed session). **P100 / T4×1 only** (B.5 mandate). All Kaggle
runs: n=50, K=3, T=1.0, **1 seed**, `--skip_existing`, Cell-2b backup at session
end. Numbers from the Kaggle weeks are **direction only** — they decide promotion,
not the paper. Each week states a GPU-h budget and a quantitative EXIT gate.

> **Hour-accounting note:** "train h" assumes P100 ≈ 1.5–3 s/step; flat losses
> ~0.5–0.8 h/1000 steps, tree losses ~1.0–1.8 h/1000 steps. "eval h" assumes
> GSM8K n=50, K=3, batched per checkpoint across the listed verifiers, ~0.2–0.4 h
> per checkpoint-batch. Recalibrate the ceiling after Week 1's measured rates.

**Weekly Kaggle budget & running total (each week ≤ 30 GPU-h ceiling):**

| week | activity | GPU-h | ≤30? | cumulative |
|---|---|---:|:---:|---:|
| 1 | pipeline validation + baseline + timing calibration | ~5 | ✅ | ~5 |
| 2 | convergence + LR tuning (`kl_tree` vs `forward_kl`) | ~12 | ✅ | ~17 |
| 3 | loss ablation batch 1 (core 7 losses) | ~27 | ✅ | ~44 |
| 4 | loss ablation batch 2 (extended ~6 losses) | ~27 | ✅ | ~71 |
| 5 | verifier × loss interaction (eval-only) | ~10 | ✅ | ~81 |
| 6 | K/T direction pre-check + crash buffer + A100 prep | ≤20 | ✅ | ~101 |

**≈ 101 Kaggle GPU-h across 6 weeks, no single week > 30** → fits free Kaggle
quota with per-week headroom for one crashed session. **A100 spend is separate:
~$22 of the $30 Modal credit, once, after Week 6** (D.cost). Total paid GPU =
**one** confirmation run.

### Week 1 — Pipeline validation + baseline sanity (Kaggle P100) · budget ≈ 5 GPU-h
- **Objective:** clean end-to-end run; confirm the untrained-baseline BE is sane;
  **measure real P100 s/step and eval-per-cell time** to calibrate all later weeks.
- **Runs:** `experiment.py --config laptop --smoke` (local/throwaway) then on
  P100 `--config kaggle` baseline-only (no training) on GSM8K **n=50**, modes
  `bv,gbv,traversal,specinfer,naive`, **K=3**, T=1.0, `--skip_existing`.
- **Inspect:** `[run_tag] block_eff=…`, `peak_vram_mb` (must stay < 16 GB),
  PPL/task_score guards, wall-time per cell; W&B `eval/block_eff`.
- **Sanity targets:** K=3 **theoretical max BE = L+1 = 9** (`L=8`); SOTA
  Qwen-class drafts ≈ **3** at K=3; untrained 0.6B→8B baseline ~1.5–2.5;
  traversal ≥ bv ≈ gbv should roughly hold.
- **EXIT:** end-to-end completes, no OOM on a single 16 GB GPU, baseline BE finite
  in ~[1.3, 3.0], results in `results.db`, **per-step and per-cell timings
  recorded** for budgeting.

### Week 2 — PRIORITY 1: flat `forward_kl` baseline + LR tuning (Kaggle P100) · budget ≈ 12 GPU-h
- **Objective:** establish the **trusted flat baseline first** (priority step 1,
  Kaggle-direction version); prove training works; pick LR/warmup/grad_accum.
- **Runs:** train **`forward_kl` (the baseline)**, 1 seed, 1000 steps, kaggle
  config (grad_accum 16, warmup 100, lr 3e-5) ≈ ~1.3 h; then `kl_tree` (first tree
  loss) ≈ ~1.3 h; if curves look off, **one** LR-variant retrain (~1.3 h). Eval
  both on GSM8K n=50, modes `bv,naive` (+`gbv`) K=3 (~1 h). Padding covers a crash.
- **Inspect:** `forward_kl` train/val curves (`train_curves`, `split`); the
  **plateau seen earlier at best≈0.3766**; any val bounce after ~step 150 (why
  warmup was added); is the `forward_kl` baseline BE in the expected band?
- **Tune:** val bounce → lower LR / lengthen warmup; train flat → raise LR /
  reduce grad_accum; val ≫ train → add LoRA dropout.
- **EXIT (gates priority step 2):** `forward_kl` baseline trains cleanly, val-loss
  plateaus, and its BE is sane/plausible. **Do not interpret any tree-loss number
  until this flat baseline is established.** Frozen LR/warmup carried forward.

### Weeks 3–4 — Loss-function ablation, batched (Kaggle P100) · budget ≈ 27 + 27 GPU-h
- **Objective:** rank all 13 losses by BE direction vs `forward_kl`; kill losers.
- **Why two weeks:** training 13 losses × ~1.3 h ≈ 17 h train + ~5 h eval ≈ 22 h
  would *just* fit one week, but with no crash margin. **Split the loss list
  across two weekly quotas** (~7 losses/week) so one lost session never costs the
  whole ablation.
  - **Week 3 batch:** `forward_kl` (control), `kl_tree`, `bv_tree`, `gbv_tree`,
    `traversal_tree`, `specinfer_tree`, `nss_tree` (the core set, B.4).
  - **Week 4 batch:** `rev_kl_tree`, `jsd_tree`, `naive_tree`, `spectr_tree`
    (approx), `khisti_tree` (approx), `ebe`, `ebe_tree` (extended set).
- **Runs/week:** ~7 losses × 1 seed × 1000 steps (~17 h) + eval each GSM8K n=50,
  matched verifier + `bv`/`naive`, K=3 (~5 h). `--skip_existing` throughout.
- **Inspect:** BE-improvement bar chart (Δ vs `forward_kl`); drop on sight any
  loss that NaNs, collapses `task_score`, or fails to beat control direction.
- **EXIT (end of Week 4):** a ranked list; **top ≤4 losses** that beat
  `forward_kl` direction at n=50 carried forward. Kill the rest (B.6.7).

### Week 5 — PRIORITY 3: verifier × loss interaction incl. GBV (Kaggle P100, eval-only) · budget ≈ 10 GPU-h
- **Objective:** find the best loss×verifier pairing (RQ4 direction), incl.
  whether `gbv` beats `bv` — pick the **1–2 pairings** to confirm on A100.
- **Runs:** **no new training** — reuse the top ≤4 Week-3/4 checkpoints; eval all
  available verifiers (`naive,bv,nss,specinfer,traversal,gbv`; add
  `spectr,khisti` if time), K=3, T=1.0, GSM8K n=50. ~4 ckpt × batched verifiers ≈
  8–10 h.
- **Inspect:** loss×verifier BE heatmap; does the diagonal (V-aligned loss best
  under V, RQ4) appear? Does `gbv` > `bv` for the best tree-loss draft? Note
  `spectr_tree`/`khisti_tree` (approximate) behaviour.
- **EXIT:** **the 1–2 best loss×verifier pairings selected** (plus `forward_kl`
  as control) — this is the entire input to the A100 confirmation.

### Week 6 — Buffer / robustness pre-check + A100 prep (Kaggle P100) · budget ≤ 20 GPU-h
- **Objective:** absorb slippage; sanity-check K/T direction cheaply before
  spending the irreplaceable A100 credit; freeze the confirmation spec.
- **Runs:** for the 1–2 finalists, a cheap K-direction peek on P100 (K=3 vs a
  single K=5 cell, n=50) and T=0.6 vs T=1.0 ranking check (~10 h). Reserve the
  rest as crash buffer for Weeks 2–5 overrun.
- **Inspect:** does ranking survive a K and T change (RQ6/RQ7 direction)? If a
  finalist only wins at one T, note it.
- **EXIT:** confirmation spec frozen (which 1–2 pairings, which verifier each);
  Modal account created and the run scripted with `--skip_existing` + backup.

### A100 confirmation — staged Modal $30 spend, **in locked priority order**
The single $30 Modal credit is spent **once, last, and strictly in priority
order**. The credit is consumed stage-by-stage; **each stage gates the next**, so
if the baseline is wrong no novel-method credit is wasted. These are the **only
paper numbers** (3 seeds, full GSM8K, the Section C stats apply).

- **Stage A — PRIORITY 1: flat `forward_kl` baseline (spend this FIRST).**
  Train `forward_kl`, **3 seeds** (42/123/7), 2000 steps, a100 config; eval **full
  GSM8K n=1319**, verifiers `naive` + `bv`, K=3, T=1.0. This is the trusted
  foundation — the reference BE every later claim is measured against.
  - **GATE:** the `forward_kl` baseline must reproduce within the expected BE
    range (Week-1 sanity band, ≈ SOTA Qwen-class order of magnitude) with a tight
    across-seed CI. **Do NOT proceed to Stage B (tree losses) until this passes.**
    If it fails, stop and fix the pipeline — spend no further credit.
- **Stage B — PRIORITY 2: best tree loss vs the Stage-A baseline.**
  Train the top tree-loss finalist (e.g. `kl_tree`/`bv_tree`/`gbv_tree`), 3 seeds,
  same config; eval full GSM8K n=1319 under its **matched verifier + `naive`/`bv`
  reference**, K=3.
  - **GATE (defines project success):** paired-bootstrap p < 0.05 (Holm-adjusted)
    **and** non-overlapping across-seed CIs for the tree loss beating the Stage-A
    `forward_kl` baseline. Proceed to Stage C only if a tree loss wins.
- **Stage C — PRIORITY 3: GBV verifier.** Re-evaluate the Stage-B winning
  tree-loss checkpoints under **`gbv` vs `bv`** (and the OT verifiers if credit
  remains), K≤4, full GSM8K. Tests whether multi-path GBV verification adds BE
  over single-path BV (H4).
  - **GATE:** BE(`gbv`) > BE(`bv`) for the tree-loss draft (within CI).
- **Stage D — PRIORITY 4 (LAST, defer): online / online-tree variants.**
  `online_kl_tree`/`online_ebe_tree` etc. — **only if credit remains** (it likely
  will not; online code is disabled pending smoke-verification). In practice push
  Stage D to a follow-up **Lightning-credit** month so the Modal credit is never
  blocked on the lowest-priority work.

**$30 fit (see D.cost):** Stages A+B at **3 seeds, K=3, matched + `naive`/`bv`
ref** are the **Trimmed-B** spec (~$22, ~10.5 A100-h) and **fit**. Stage C is a
cheap eval-only add-on on existing checkpoints (~$3–5). **K=5, a second pairing,
and Stage D do NOT fit the $30** — defer them to Lightning credits. Spend order is
A → B → C, stopping the moment the credit is exhausted (the baseline + the single
headline tree-loss result are the non-negotiable minimum).

### D.cost — does the A100 confirmation fit in $30?

**Assumptions:** Modal **A100-40 GB ≈ $2.10/hr** (Qwen3-8B bf16 fits 40 GB; an
80 GB card ≈ $2.50/hr is unnecessary). $30 ⟹ **~12–14 A100-hours**. A100 train
≈ 0.3–0.6 s/step → ~0.4–0.7 h per 2000-step checkpoint (tree losses higher end).
GSM8K **n=1319** BE eval ≈ ~0.4–0.8 h per (checkpoint × verifier × K) batch
(≈26× the n=50 Kaggle cell; traversal/gbv slowest).

| spec | checkpoints | train h | eval cells | eval h | total h | ≈ cost | fits $30? |
|---|---|---|---|---|---|---|---|
| **Full** (2 pairings + control, 3 seeds, K∈{3,5}, matched + bv + naive) | 9 | ~5.5 | 9 × 3 verif × 2 K = 54 | ~25 | **~30 h** | **~$63** | **No** |
| **Trimmed-A** (2 pairings + control, **2 seeds**, K=3, matched + naive) | 6 | ~3.5 | 6 × 2 × 1 = 12 | ~7 | ~10.5 | ~$22 | **Yes** |
| **Trimmed-B (recommended)** (1 pairing + control, **3 seeds**, K=3, matched + naive) | 6 | ~3.5 | 6 × 2 × 1 = 12 | ~7 | ~10.5 | ~$22 | **Yes** |

**Verdict:** the full multi-seed × K∈{3,5} × 2-pairing spec **does NOT fit $30**
(~$63, ~30 A100-h). **It fits only after cutting one dimension.** Recommended cut:
**Trimmed-B** — keep **3 seeds** (preserves the across-seed CI, the core
rigor claim) and the strongest **single** pairing, drop **K=5** (K=3 is the
config default and the headline number) and drop the second pairing. That lands
≈ **$22**, leaving ~$8 (~4 A100-h) of credit for **one** retry or a single K=5
add-on cell on the winner. If two pairings must both be confirmed, use
**Trimmed-A** (2 seeds) instead — but 3 seeds is the better rigor trade.
Lightning's ~15–22 monthly credits are the fallback to add K=5 or the second
pairing in a follow-up month without touching the spent Modal credit.

### After the confirmation — write-up & reproducibility (no GPU)
- Run `analyze_results.py` (after the C.6 stats upgrades) → Markdown tables;
  regenerate plots; verify `git_sha`/`seed`/`wandb_url` logged for every confirmed
  run; re-derive every paper number from a `run_tag`. A from-scratch re-run is a
  Lightning-credit task, not a Modal one (protect the spent $30).

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
| **Stats invalid by construction** | single seed, no BE variance, approximate p-values (Section 0.6) | implement the Section C.6 checklist *before* the A100 confirmation; Kaggle 1-seed numbers are direction-only, never published |
| **Wasted weekly quota** (lost session, T4×2 mis-select) | `/kaggle/working` ephemeral; T4×2 doubles burn (B.5) | P100/T4×1 only; Cell-2b backup every session; `--skip_existing`; ≤3 sessions/week so one crash ≤ ⅓ of quota |
| **Burning the $30 Modal credit early or twice** | one-time, irreplaceable | spend ONCE, last, on the Trimmed-B spec only after Kaggle narrows to ≤2 pairings; keep ~$8 buffer for one retry; Lightning credits for any extension |
| **gbv numerical instability at K>4** | `compute_skew` docstring: "not recommended … K>4" | restrict gbv/gbv_tree to K≤4; footnote |
