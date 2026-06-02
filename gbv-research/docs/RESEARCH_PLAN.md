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

### 0.5 Hardware tiers and run commands

> For hardware tier configs, step-by-step run commands, and W&B group structure, see `deploy/PROGRESSION.md`. For compute costs and tier selection, see `docs/COMPUTE.md`.

**Key principle: always use 8B teacher for any research decision.**

| hw_tier | Config | Hardware | Teacher | Research-valid? | Purpose |
|---|---|---|---|---|---|
| `laptop` | `laptop.yaml` | RTX 500 Ada 4GB | Qwen3-0.6B | **No** — teacher ≈ draft | Crash-check: every code path runs |
| `cpu` | `server_gpt2.yaml` | 128GB RAM, CPU-only (ATS Cloud) | GPT-2-M (355M) | **No** — too small | Convergence proof on GPT-2; shows method works in principle |
| `colab` | `kaggle.yaml` | T4 x2 (Kaggle/Modal/Lightning) | Qwen3-8B NF4 | **✓ Yes** | Exploration: rank losses, tune LR, kill losers |
| `a100` | `a100.yaml` | A100 40/80 GB | Qwen3-8B BF16 | **✓ Yes** | Publication confirmation only |

**Per-loss iteration** (works on all persistent platforms):
```bash
# Run one loss at a time — baseline runs once, --skip_existing resumes:
python orchestration/experiment.py --config a100 --losses kl --yes
python orchestration/experiment.py --config a100 --losses bv_tree --yes
# Each invocation: train → merge → eval for that loss only
```

**Kaggle constraint:** T4 x2 burns 2× quota (~15 effective h/wk) → 1 loss per session practical.
Use `LOSSES = "kl"` in Cell 0, change each session. See `docs/KAGGLE.md`.

### 0.6 Existing statistical rigor — what EXISTS vs MISSING

**Exists:**
- **Statistical analysis** (`paper/analyze_results.py`): Welch's two-sample t-test, Cohen's d, `ci95 = 1.96·σ/√n`, significance stars, % change, K-sensitivity tables, cross-dataset CV. Note: p-values use a hand-rolled incomplete-beta approximation — approximate, not scipy-backed; with a single run per cell the test returns `n/a`.
- **W&B logging**: per-run summary metrics (eval/BE/alpha by verifier/dataset), `job_type=train/eval`, `wandb_group` per tier, run name encodes teacher+loss+family+steps. See `deploy/PROGRESSION.md` for the full W&B metric taxonomy and run naming convention.
- **Automated training alerts** (`trainer.py`): grad_norm > 10, overfit_ratio > 1.3, val plateau, slow convergence at 10% of steps — all fire W&B alerts automatically.
- **LoRA gradient health**: lora_A/B mean/max/std logged per step to W&B.

**Missing (must be added for publication):**
1. **Multi-seed orchestration.** `experiment.py` runs a single `seed` (default 42); no loop over seeds, no seed dimension in run identity beyond the (unused-for-aggregation) `seed` column.
2. **BE variance/CI.** `block_eff_std` and `per_prompt.block_eff` are never populated; the verifier prints a single mean. No way to put error bars on BE.
3. **Bootstrap CI for BE comparisons.** No bootstrap anywhere in the repo.
4. **Paired tests over prompts.** No paired bootstrap / paired t-test; current comparison is unpaired Welch across runs.
5. **Multiple-comparison correction.** No Bonferroni/Holm despite many loss×verifier comparisons.
6. **Exact p-values.** No `scipy.stats` dependency; p-values are approximate (hand-rolled incomplete-beta).

### 0.7 Current W&B observable metrics

For the current W&B metric taxonomy, run naming convention, and group structure, see `deploy/PROGRESSION.md` — W&B details are not duplicated here.

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

- **Tier 2 exploration (Kaggle T4x2 or Modal T4): one-factor-at-a-time (OFAT), 1 seed, n=50.** Vary
  loss with verifier fixed to the matched mode, K=3, T=1.0, GSM8K only. Cheap,
  isolates RQ1–RQ3 for *direction*; later widens to the verifier columns for RQ4.
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

### B.5 Compute platform selection and budget

> For compute platform selection, quota constraints, the exploration/confirmation funnel, and cost breakdown, see `docs/COMPUTE.md`.

### B.6 Per-session protocol and frugality rules

> For per-session frugality rules, checkpoint backup procedures, Kaggle/Modal session protocol, and the full run checklist, see `deploy/PROGRESSION.md`.

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

- **Tier 2 exploration uses n=50** — enough to rank losses, half the eval cost of n=100, and never published as final results.
- **A100 confirmation uses the full 1319-prompt GSM8K test set** — this is the only sample size that goes in the paper.
- A paired-bootstrap on 1319 prompts resolves BE differences of ~0.1 block; n=50 resolves only ~0.4–0.5 — adequate for ranking direction, not for the final delta. This is exactly why n=50 stays on Tier 2 and 1319 is reserved for the one A100 run.

> For per-tier n_prompts configuration values, see `deploy/PROGRESSION.md`.

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

## Section D — Iteration priority and A100 staged confirmation

> ### 🔒 Iteration priority (LOCKED)
> The algorithm is **not final** and will need **multiple iterations** — this plan
> is a **loop, not a one-shot**. Every iteration proceeds in this strict order,
> and **each step gates the next**:
>
> 1. **FIRST — Flat-baseline benchmark (`forward_kl`).** Establish a solid,
>    trusted published-loss baseline. The first A100 spend produces the
>    `forward_kl` baseline numbers (full GSM8K). Nothing else is meaningful until
>    this is reproduced in the expected BE range.
> 2. **SECOND — Tree-loss training** (`kl_tree`, `bv_tree`, `gbv_tree`, …), the
>    core novel contribution, compared against the step-1 baseline.
> 3. **THIRD — GBV verifier** (verifier-side contribution) — only after tree-loss
>    training shows promise.
> 4. **LAST — online / online-tree variants** (`online_kl_tree`,
>    `online_ebe_tree`, …) — lowest priority, deferred to the very end.
>
> Do not advance a step until the previous step's EXIT gate passes. On each loop
> iteration (algo revision) re-enter at the earliest step the change affects.

> For the week-by-week Tier 2 execution plan, session protocol, and per-tier EXIT gates, see `deploy/PROGRESSION.md`. For compute budget, quota constraints, and cost breakdown, see `docs/COMPUTE.md`.

### A100 confirmation — staged spend, **in locked priority order**
The A100 credit is spent **once, last, and strictly in priority order**. The
credit is consumed stage-by-stage; **each stage gates the next**, so if the
baseline is wrong no novel-method credit is wasted. These are the **only paper
numbers** (3 seeds, full GSM8K, the Section C stats apply).

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

**Spend order is A → B → C**, stopping the moment the credit is exhausted. The `forward_kl` baseline + the single headline tree-loss result are the non-negotiable minimum. Stages A+B at 3 seeds, K=3, matched + naive/bv reference fit within a typical $30 free credit. For the full cost breakdown (Trimmed-A / Trimmed-B spec analysis), see `docs/COMPUTE.md`.

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
| **OOM on T4** loading 8B teacher | `kaggle.yaml`/Modal T4 use 4-bit NF4 (Kaggle 29 GB system RAM handles the CPU intermediates); `colab.yaml` drops to Qwen3-4B BF16 (Colab 12 GB RAM cannot absorb the 8B NF4 CPU spike) | use `kaggle` or Modal T4 for 8B-NF4; `colab.yaml` for Colab (4B bf16); the OOM→CPU retry in `run_alpha`/`run_be` is a slow fallback, not a fix — prefer the right config |
| **Orphaned eval subprocesses** exhausting VRAM on re-run | `run_be_batch` kills stray `runner.py` via psutil; `expandable_segments:True` | keep psutil installed; one BE subprocess per dataset (already batched) |
| **T4 timeouts**, esp. traversal K=5 | `run_be`/`run_be_batch` have 30 min / 2 h timeouts; partial results are recovered and `--skip_existing` resumes | on T4 cap K=3 for traversal/gbv; push K=5 to A100; rely on partial-recovery + resume |
| **Kaggle checkpoint loss** (ephemeral `/kaggle/working`) | configs set `storage_root`; trainer writes `ckpt_latest`/`ckpt_best` + `training_state.json` for crash-safe resume | always pass `--storage_root`; mirror to Drive/persistent volume; `save_every=25` on T4 |
| **Small-sample noise** | T4 n=10–100; BE has no CI today | rank on T4, *decide* only on A100 n=1319 with ≥3 seeds + paired bootstrap (Section C) |
| **Teacher-model appropriateness** (0.6B draft vs 8B teacher; laptop's 0.5B↔0.6B is pure code-check) | laptop config warns its numbers are "meaningless" | never interpret laptop/small-teacher numbers as findings; publish only 8B-teacher (Tier 2+) results |
| **Approximate losses misread as exact** | `spectr_tree` (ρ detached), `khisti_tree` (LP-free) are documented surrogates | label both "approximate" in every table; if either underperforms its aligned verifier, attribute to the surrogate, not the alignment hypothesis (per their docstrings) |
| **Stats invalid by construction** | single seed, no BE variance, approximate p-values (Section 0.6) | implement the Section C.6 checklist *before* the A100 confirmation; Kaggle 1-seed numbers are direction-only, never published |
| **Wasted weekly quota** (lost session) | `/kaggle/working` ephemeral; T4 x2 burns 2× → ~15 effective h/wk | Always T4 x2 (only working Kaggle GPU); checkpoint backup every session; `--skip_existing`; ≤1 session/week so one crash ≤ quota — see `deploy/PROGRESSION.md` |
| **Burning the $30 Modal credit early or twice** | one-time, irreplaceable | spend ONCE, last, on the Trimmed-B spec only after Kaggle narrows to ≤2 pairings; keep ~$8 buffer for one retry; Lightning credits for any extension |
| **gbv numerical instability at K>4** | `compute_skew` docstring: "not recommended … K>4" | restrict gbv/gbv_tree to K≤4; footnote |
