# Speculative Decoding — Draft Distillation: Project Report

**Task:** distil a small draft model to maximise block efficiency (BE) under speculative decoding against a larger teacher.
**Primary metric:** BE = generated tokens / target model calls (higher is better).
**This document is a map, not a results dump.** Every experiment family below has a dedicated research note with full tables, ablations, and mechanism analysis. Raw per-run numbers live in the committed `results/*.csv` (note: a root-level `results.csv` is the **gitignored** local runtime dump `eval.py` appends to — it is *not* in the repo, so `results/` is the source of truth); wandb run URLs are in each note's ablation table. Read this document to find which note to open, not to read the numbers themselves. The one exception is the 0.6B/8B hyperparameter sweep, which is included in full in the section "Hyperparameter sweep — research report" below.

---

## Status at a glance

| Model pair | Status | Best result vs. flat JSD | Note |
|---|---|---|---|
| Qwen3-0.6B draft / Qwen3-8B teacher | **Exploration closed** | `jsd_flat_enrich` (K=4 sweet spot) — only family to consistently beat JSD on traversal, both datasets | see family table below |
| Qwen3-1.7B draft / Qwen3-32B teacher | **First seed swept** | none confirmed; enrich does **not** cleanly replicate, depth_weight's matched-K diagonal does **not** replicate, log-tree & PO-cold actively harmful (all single seed) | see family table + `per_checkpoint_sweeps_2026-07/` |

**Headline finding (0.6B/8B):** every loss family that reweights or restructures training on **teacher-context tokens** (flat divergences, tree losses, prefix-overlap, depth-weighting) plateaus at the same ≈5.9–6.1 traversal-K3 BE ceiling as plain flat JSD. The one family that broke through is **enrichment** (`jsd_flat_enrich`), which trains on *fresh stochastic teacher rollouts* instead of reweighting a fixed teacher-context sequence, and the win is real on the deployment verifier: at 0.6B/8B, mean traversal Δ over K=2–4 is **+0.19 (math) / +0.12 (olympiad)** at the K=4 sweet spot (K=6 degrades — more rollouts stop helping). The **OOD (+0.12) win is matched-backend-verified** (olympiad K2–4 are sdpa-to-sdpa, +0.05/+0.29/+0.01); the math +0.19 is cross-backend (enrich sdpa vs baseline FA2) but robust as a multi-cell mean since the sdpa−FA2 gap nets ~0.

**Headline finding (1.7B/32B, first seed):** the enrich win does **not** cleanly replicate at the bigger capacity gap — `enrich_K3` traversal Δ is −0.05/−0.13/**+0.27**/+0.09 across K on math_eval (sign-flips; single seed). Two robust *negative* results emerged instead: (a) **log-space tree losses (`traversal_log`, `nss_log`) are catastrophic at this pair** — traversal Δ ≈ **−0.4 to −0.6 at K1–K3** (K4 −0.22 to −0.39), far past the ~0.2 noise floor, and matched-backend (FA2) at K1–K3; (b) a **capacity-vs-depth tradeoff** — with *both* JSD baselines fully converged, the 1.7B/32B draft beats 0.6B/8B at shallow speculation (traversal K1/K2 math) but loses at deep speculation (K3/K4) and across olympiad_eval. Not a training-budget artifact (both baselines early-stopped on a genuine val plateau).

---

## Experiment families

Each row is a family of losses tested under the same mechanism; open the note for the full ablation table, gradient analysis, and verdict.

| Family | Losses / flags | Question tested | Verdict | Note |
|---|---|---|---|---|
| Flat baselines | `forward_kl`, `reverse_kl`, `jsd`, `l1` | Which token-matching divergence is strongest? | JSD selected as primary baseline (broadest win across verifiers) | — (see Phase 1–4 history in git log if needed; not carried forward here) |
| Tree losses (on-policy) | `naive_tree`, `traversal_tree`, `kl_tree`, `bv_tree`, `gbv_tree`, `nss_tree`, `specinfer_tree`, `spectr_tree`, `khisti_tree`; log-space variants `traversal_log`, `naive_log`, `nss_log`; off-policy `op_naive_tree(_full)`; REINFORCE `*_tree_pg` (removed) | Does training on the draft's own sampled tree (rather than teacher context) redirect the gradient toward acceptance? | No gain over JSD, and the log-space variants make it **worse**. `bv_tree`/`gbv_tree` collapse to zero gradient (single-token gate). Log-space fixes product-underflow but at 1.7B/32B `traversal_log`/`nss_log` are **catastrophic** — traversal Δ −0.4 to −0.6 at K1–K3, K4 −0.22 to −0.39 (`per_checkpoint_sweeps_2026-07/{traversal,nss}_log_K3_*_Qwen32B-Qwen1.7B.csv`). The log wrapper amplifies gradient on low-α (deep, unlikely) nodes, over-fitting the draft to tree tails it can't actually reach. | [`tree_losses_research_note.md`](notes/tree_losses_research_note.md) |
| Depth-weighting | `--aux_mode depth_weight --aux_loss <tree_loss> [--depth_linear \| --depth_lambda λ]` | Does curriculum-reweighting JSD by expected acceptance depth help? | No detectable signal alone (detached scalar, can't redirect gradient direction). One combined run (`jsd + depth_weight_lin + naive_tree`, K3, 40K steps) shows **no** verified traversal win at matched K — the +0.29 previously cited was a K_eval=1 BE (6.163) mislabeled as K=3; the true K_eval=3 Δ is +0.03, within noise. The one 0.6B/8B pattern worth confirming (matched train-K=eval-K diagonal, +0.067/8-of-9 at K3) **does NOT replicate at 1.7B/32B** (`jsd_dw_lin_naive_K3`: K_eval=3 mean −0.008, 4/9) — family closed. | [`depth_weight_research_note.md`](notes/depth_weight_research_note.md) |
| Enrichment | `jsd_flat_enrich --K <M>` (flat, stochastic teacher rollouts); `enrich_K3` (1.7B/32B); `enrich_draftcond_K3` (draft-conditioned rollouts); `jsd_enrich` (tree variant) | Does training on fresh stochastic teacher paths (not a fixed teacher-context sequence) beat flat JSD? | **Yes at 0.6B/8B — the only positive result.** Flat enrich beats JSD on traversal both datasets; **K=4 is the sweet spot** (mean K2–4 Δ +0.19 math / +0.12 oly), K5 ties, K6 degrades. **Does NOT cleanly replicate at 1.7B/32B** (`enrich_K3` sign-flips across K, single seed). `enrich_draftcond` (draft-conditioned rollouts) now swept K1–4 on math_eval **and olympiad_eval**: **underperforms both JSD and standard enrich on both datasets** — draft-conditioning the rollouts reverses the enrich win (traversal Δ vs flat-enrich −0.08 to −0.19 math, −0.15 to −0.37 olympiad), disconfirming the "on-policy beats off-policy" hypothesis for acceptance-matching in- and out-of-distribution. | [`enrich_research_note.md`](notes/enrich_research_note.md) |
| Prefix-overlap (PO) | `prefix_overlap --prefix_objective {prob,logprob,traversal,nss}`; cold-start (`po_*_multiN4`, 1.7B/32B) and JSD-warm-start (`po_*_warm*`, 0.6B/8B) | Does explicitly rewarding teacher-prefix log-probability (Rahul's PO estimator, 4 approximations) beat JSD? | No, confirmed at both pairs. Warm-start 0.6B/8B: traversal Δ hovers ±0.1 (redundant with JSD). Cold-start 1.7B/32B: **worse** — and the `traversal` objective (which directly targets the deployment verifier) is the **single worst** variant (Δ −0.45/−0.38/−0.27/−0.19 across K1–4, matched-backend FA2 at K1–3). Small positive blips on olympiad for `po_nss`/`po_logprob` (≈+0.05 to +0.17 across K) but sub-noise and reverse on math_eval. The 0.6B/8B `nss_warmlogprob` "borderline" has one traversal-K3 win (+0.156) surrounded by losses at K1/K2/K4 — not a consistent effect. | [`prefix_overlap_research_note.md`](notes/prefix_overlap_research_note.md) |
| Verifier study (DDTE) | eval-only; `traversal_dAdapt`, `traversal_dL{3,4,5}` across all trained checkpoints | Does delayed tree branching help, and does it hold on *trained* drafts (paper only tested untrained)? | **Yes, and it transfers to trained drafts** — best `dL` variant beats base traversal at K3/K4 for **nearly every** checkpoint on both pairs (uplift grows with K, ≈0 at K1): 0.6/8 math K3/K4 up to +0.38, 1.7/32 math K3/K4 **+0.20 to +0.79** (largest single uplift +0.94, `enrich_draftcond` specinfer_dL5 K4). Three 0.6/8 cells are marginally negative (PO-family checkpoints). Uplift is *inverse* to base draft quality but does NOT reorder absolute BE — collapsed drafts + DDTE stay ~0.5 BE below trained + DDTE (two-tier). Uplift is intra-file so **backend-clean**. |  [`ddte_research_note.md`](notes/ddte_research_note.md) |
| LK-α (acceptance-rate distillation) | `lk_alpha`, `lk_alpha_enrich --K <M>` | Does directly optimizing `-log Σ_v min(p,q)` (Samarin et al., arXiv 2602.23881) beat JSD? | **No — ties/loses JSD; the mechanistic prediction held.** From-scratch `lk_alpha` (0.6B/8B, greedy rollout, sdpa) now run: no consistent win on any deployment verifier (traversal mean K2–4 ≈ −0.05 math / +0.03 oly, sign-flips within noise), and a clear **loss** on traversal K3 (−0.27; absolute 5.685 vs the 5.870/5.957 bars). Confirms that the `1/α` amplification `lk_alpha` shares with `naive_log`/`traversal_log` (which tie/lose) is not sufficient on independent (non-EAGLE) drafts — the differentiator in the paper is EAGLE-conditioning + scale, not the loss. `lk_alpha_enrich` (stochastic-rollout variant) unrun. | [`lk_alpha_research_note.md`](notes/lk_alpha_research_note.md) |

---

## Open question driving current work

Every family above trains on **teacher-context tokens** except enrichment. The uniform ceiling across families is the signature of either (a) JSD already saturating the information content of teacher-context training data, or (b) the 0.6B draft lacking capacity to use a stronger signal even if given one. These are indistinguishable at a single model pair.

**Status:** Qwen3-1.7B / Qwen3-32B first-seed sweep complete (one representative loss per family, cold-start, single seed; per-checkpoint CSVs in `results/per_checkpoint_sweeps_2026-07/`). Result: no family confirmed to beat JSD on traversal, and enrich's 0.6B/8B win did not replicate — but the two JSD baselines *did* converge fairly (both early-stopped on a val plateau, verified from training logs), so this is a real capacity-signal result, not undertraining. The remaining blocker to any positive claim here is a **second seed** — every 1.7B/32B number is n=1, and the measured noise floor at this pair is still uncharacterized.

---

## Hyperparameter sweep — research report

0.6B/8B, `math_hard` training data, 5k steps, single seed unless noted. Raw
CSVs + all charts: [results/passK](https://github.com/Rmuk655/Distill-Spec-Research/tree/Pipeline/results/passK).
Pass@k findings (jsd vs prob, dapo dataset, multi-root fresh-vs-tail-reuse,
per-hyperparameter pass@k ablations) live in
[passk_prob_vs_jsd_research_note.md](notes/passk_prob_vs_jsd_research_note.md) —
not repeated here. This report covers training-time metrics only: does
the axis change convergence behavior or `best_val_block_eff`.

### Summary — one row per axis

Training convergence/collapse behavior is covered in the Detail section below,
not repeated per-row here — only two axes (LR, grad_clip) actually showed
anything convergence-related; a "convergence" column that's empty for 13 of
15 rows wasn't earning its place.

"Value kept" is what the actual deployment picks used going forward — jsd's
(`jsd_lr1e-5_wu10_lrmin0.1_wd0.01`, BE=5.994) and prob's
(`po_prob_ceanneal3000_auxw0.5_lr1e-5_wu20`, BE=5.848), the two real
best-BE-selected configs referenced throughout this sweep, not a
best-of-sweep-after-seeing-the-result pick — read off directly from each
checkpoint's own name/config, not re-derived here.

| hyperparameter | values tried | best_val_block_eff | pass@k | value kept |
|---|---|---|---|---|
| LR, sub-1e-5 boundary | 2e-6, 3e-6 vs 1e-5 | 2e-6 (5.710) ties 3e-6 (5.701); the gap to 1e-5 (5.608, ~0.10) is *also* smaller than this neighborhood's own ~0.20 noise floor — not a confirmed win over 1e-5 either, all three are indistinguishable here | not tested at this sub-axis | `1e-5` (neither 2e-6 nor 3e-6 confirmed better) |
| warmup % | 5, 10, 20 | 20% edges out (5.608 > 5.560 > 5.501), within noise | **[passk note](notes/passk_prob_vs_jsd_research_note.md#ablations-tested--none-changed-the-qualitative-picture-except-n) — 20% clears jsd at all 3 k's, 10% at 2, 5% at 1: directionally consistent, unlike BE** | prob kept `20%`; jsd kept `10%` — differs by loss family |
| lr_min_ratio | 0.1, 0.01 | no effect (5.608 vs 5.589) | **[passk note](notes/passk_prob_vs_jsd_research_note.md#ablations-tested--none-changed-the-qualitative-picture-except-n) — 0.1 clears jsd at all 3 k's, 0.01 clears none; diverges from the BE result** | `0.1`, both families |
| teacher top-k | 0, 20, 50 | no benefit, mild harm at higher k | [passk note](notes/passk_prob_vs_jsd_research_note.md#ablations-tested--none-changed-the-qualitative-picture-except-n) — all three clear jsd at all 3 k's, no harm despite the BE result | `0` (default) |
| weight decay | 0.01, 0.001, 0.0001 | no trend, all within noise | **[passk note](notes/passk_prob_vs_jsd_research_note.md#ablations-tested--none-changed-the-qualitative-picture-except-n) — 0.0001/0.01 clear jsd, 0.001 clears nothing: non-monotonic, worse than both neighbors** | `0.01` |
| grad_clip | 1.0 (default), 10, 100, 1000 | BE rises loosely as the threshold loosens (1.0→10→100→1000 = 5.517→5.647→5.535→5.692), but not cleanly — 100 breaks the trend. The only gap big enough to maybe be real (not noise) is 1000 vs the default (see Detail for the convergence shape) | [passk note](notes/passk_prob_vs_jsd_research_note.md#ablations-tested--none-changed-the-qualitative-picture-except-n) — only `gradclip100` clears jsd beyond noise | `1.0` (default) — the 1000 hint was never adopted, single-seed only |
| teacher_temp | 0.5, 0.7 vs default 1.0 | **BE ranks 0.7 > 0.5 > default** (5.643/5.602/5.517, within noise) | **[passk note](notes/passk_prob_vs_jsd_research_note.md#ablations-tested--none-changed-the-qualitative-picture-except-n) ranks it the opposite way: default > 0.5 > 0.7** — the default clears jsd by the widest margin, `0.7` (the best-BE point) clears it nowhere | `1.0` (default) — turned out to be the right call on pass@k despite ranking worst on BE |
| grad_accum | 8 (default), 16, 32 | tightest spread of any single-axis probe, no effect | [passk note](notes/passk_prob_vs_jsd_research_note.md) — no effect there either | `8` (default) |
| multi-root N (tail-reuse) | 8, 16, 32, +random-offset | all four inside single-root baseline's noise floor, no trend | [passk note](notes/passk_prob_vs_jsd_research_note.md#multi-root-two-different-ways-to-build-the-continuation-at-each-root) — N=16/16+offset/32 clear jsd beyond noise, N=8 doesn't | not adopted into the main deployment pick (single-root); `N=16` used only for the separate dapo follow-up |
| fresh vs tail-reuse estimator | N=16/32, M=1/2, ± CE-anneal | all points inside the same noise band, no BE surprise | [passk note](notes/passk_prob_vs_jsd_research_note.md#multi-root-two-different-ways-to-build-the-continuation-at-each-root) — tail-reuse wins at N=16, fresh only matches it at N=32 | tail-reuse, whenever multi-root is used at all |
| training dataset | `math_hard` vs `dapo_math_train` | within noise | [passk note](notes/passk_prob_vs_jsd_research_note.md) — dapo trails `math_eval`, leads `gsm8k_eval`/`olympiad_eval`; real, not uniform | `math_hard` remains primary; dapo used only for its own comparison |
| M (samples/root) | 4, 8, 16 (warm), 16 (cold) | M4=5.701, M8=5.810, M16=5.619 (warm) / 5.694 (cold redo) — no trend, all within noise (see Detail for the M16 caveat) | [passk note](notes/passk_prob_vs_jsd_research_note.md#ablations-tested--none-changed-the-qualitative-picture-except-n) — tracks the individual checkpoint, not M itself | `4` (default) |
| CE-anneal `anneal_steps` | 1500, 3000, 4000 | 3000 wins (5.796) | **[passk note](notes/passk_prob_vs_jsd_research_note.md#ablations-tested--none-changed-the-qualitative-picture-except-n) — the BE-best pick (3000) clears jsd nowhere on pass@k; 4000 (worse on BE) clears it at 2 of 3 k's** | `3000` — kept on BE despite the pass@k result above |
| CE-anneal `aux_weight` | 0.5, 1.0, 2.0 | 0.5 is prob's overall best (5.848) | [passk note](notes/passk_prob_vs_jsd_research_note.md#ablations-tested--none-changed-the-qualitative-picture-except-n), CE-anneal timing row | `0.5` |
| forgetting vs `aux_weight` | 0.5, 1.0, 2.0 | anchor reduces forgetting up to 1.0, rises again at 2.0, doesn't track the BE-optimal 0.5 | n/a | `0.5` (BE-optimal chosen over `1.0`, the forgetting-optimal point) |

### Detail — axes with content beyond what the Summary table cells already say

Charts: [LR probe jsd vs prob](results/passK/lr_probe_jsd_vs_prob.png) ·
[Collapse timing by LR bucket](results/passK/survival_curves_by_lr.png) ·
[Vanilla sweep dashboard](results/passK/sweep_dashboard_vanilla.png) ·
[CE-anneal sweep dashboard](results/passK/sweep_dashboard_ceanneal.png) ·
[Forgetting vs aux_weight](results/passK/forgetting_vs_aux_weight.png).

- **LR boundary probe, jsd's own behavior (not in the table above, which is prob-only):** `jsd` holds a flat plateau across `3e-6`–`1e-5` (~5.97–5.99), dipping below outside that range.
- **Collapse timing by LR bucket:** healthy range (1e-6–1e-5) never collapses; `1e-4` collapses gradually/staggered by step ~2000; `≥3e-4` collapses almost instantly.

**Grad clip {10,100,1000} vs default (1.0):** `prob`, lr=1e-5, single seed each. The real lever is the effective per-step update magnitude the threshold *allows*, not the clip frequency: raw grad norms are ~50–100 (spiking to 200–700), the same across all four runs, so a threshold of 1.0 or 10 clips ~100% of steps but rescales to a step 10× larger at 10, 100 clips ~half the steps, and 1000 essentially never clips. `best_val_block_eff` rises roughly with allowed magnitude — 5.517 (1.0) → 5.647 (10) → 5.692 (1000) — with 100 (5.535) the odd point out, so the trend is directional, not clean. 1000 vs default is the largest gap (Δ=0.175), just above the ~0.13–0.15 noise floor and the only comparison here that plausibly clears it, still single-seed.

| grad_clip | best_val_block_eff | clip% at convergence | end-of-run (~step 4800) | run |
|---|---|---|---|---|
| 1.0 (default) | 5.517 | ~100% | — | [link](https://wandb.ai/rmukund16-indian-institute-of-technology-hyderabad/distillspec-pipeline/runs/pe2pb5la) |
| 10 | 5.647 | ~100% | ~5.40 | [link](https://wandb.ai/rmukund16-indian-institute-of-technology-hyderabad/distillspec-pipeline/runs/9w41gr72) |
| 100 | 5.535 | ~48% | ~5.40 | [link](https://wandb.ai/rmukund16-indian-institute-of-technology-hyderabad/distillspec-pipeline/runs/xr0bchto) |
| 1000 | 5.692 | ~0% | ~5.48 | [link](https://wandb.ai/rmukund16-indian-institute-of-technology-hyderabad/distillspec-pipeline/runs/1obcwkp6) |

All four grad_clip variants show the same early-peak-then-decline shape in `smoothed_block_eff`: e.g. 1000 peaks ~5.68–5.69 then falls to ~5.48 by step 4800, 100 peaks ~5.52 then falls to ~5.40. The higher peak at 1000 does NOT persist — by end-of-run all thresholds converge to ~5.40–5.48, within noise of each other. Since `ckpt_best` deploys the peak (not the final step), the peak is the number that counts, which is why the table's `best_val_block_eff` differs from the raw chart's right edge. In every case `val/forgetting` climbs in lockstep (0 → ~1.0–1.2) as block_eff falls — the forgetting metric doing its job, not a gradient-sign bug (`train/loss` decreases monotonically throughout). Grad_clip does not prevent this decline at any threshold tested. None of the four points close the gap to jsd's real best-BE deployment pick (5.994) — grad_clip tuning moves prob's BE by ~0.1–0.2 at most, not enough to touch that ~0.3–0.5 gap.

- **M16 caveat (not in the table above):** the M16 cell's two values (5.619 / 5.694) come from a crash-recovery warm-start resume vs. a clean cold-start redo of the same config. An earlier claim that M16 was genuinely worse than M4/M8 was based on the warm-start number and did not hold up once redone cleanly as a cold start — use 5.694, not 5.619, if citing M16 elsewhere.
- **Forgetting vs `aux_weight`, one addition:** since the forgetting-minimizing point (`aux_weight=1.0`) isn't the BE-optimal point (`0.5`), the anchor isn't working *purely* through reduced forgetting — something else about `aux_weight=0.5` is driving its BE advantage.

### Caveat

Single seed throughout — gaps under ~0.1 BE are within likely noise, not resolved without repeat seeds. Noise floor itself is non-uniform by LR (e.g. ~0.004 near `3e-6` vs ~0.20 near `2e-6`) — check any claimed gap against the local noise level, not a single sweep-wide number.

---

## Where results live

- **Per-checkpoint eval rows:** committed under `results/*.csv` (named per experiment batch). A root-level `results.csv` is the gitignored local runtime dump (`eval.py` append target), not tracked in the repo. The **2026-07 full-sweep** (both pairs, every loss family in the tables above, all verifiers × K=1–4 × math_eval+olympiad_eval) lives in [`results/per_checkpoint_sweeps_2026-07/`](results/per_checkpoint_sweeps_2026-07/) — one CSV per checkpoint; baselines are `jsd_mathhard_s123.csv` (0.6B/8B) and `jsd_math_hard_s123_Qwen32B-Qwen1.7B.csv` (1.7B/32B) — the two filenames differ only by an underscore, so the 1.7B/32B baseline is now suffixed with the explicit pair to avoid mixing them up. Reproduce the deltas cited above with `scripts/analyze_evals.py --glob "results/per_checkpoint_sweeps_2026-07/*.csv"`.
- **Attn-backend hygiene (audited 2026-07):** compare only matched `sdpa`-vs-`sdpa` or `FA2`-vs-`FA2` cells — the measured sdpa−FA2 gap is −0.37…+0.23 per cell (mean ≈ −0.02), so multi-cell means are robust but single cells are not. Current file backends: enrich / draftcond / lk_alpha / dw_K3(1.7-32) = sdpa; log-tree / PO-cold(1.7-32) = FA2; the 0.6/8 JSD baseline math base-verifiers = FA2 (its olympiad K2–4 = sdpa); the 1.7/32 baseline = FA2 at K1–3, sdpa at K4. The 1.7/32 log-tree/PO negatives are matched (FA2) at K1–3; the enrich OOD win is matched (sdpa) at K2–4. Each note carries a per-claim backend caveat.
- **Training curves:** wandb project `distillspec-pipeline`; every run tagged with loss, dataset, K, L, and (as of the 1.7B/32B study) the model pair.
- **Raw logs / stdout:** for the 1.7B/32B study, under `$OUT/Qwen32B-Qwen1.7B/{checkpoints,logs,output}` (with `OUT` your output root).
- **CLI reference:** [`README.md`](README.md) — every flag used above, grouped by experiment family.
