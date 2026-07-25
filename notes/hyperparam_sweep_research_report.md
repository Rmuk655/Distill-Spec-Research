# Hyperparameter sweep — research report

0.6B/8B, `math_hard` training data, 5k steps, single seed unless noted. Raw
CSVs + all charts: [results/passK](https://github.com/Rmuk655/Distill-Spec-Research/tree/Pipeline/results/passK).
Pass@k findings (jsd vs prob, dapo dataset, multi-root fresh-vs-tail-reuse,
per-hyperparameter pass@k ablations) live in
[passk_prob_vs_jsd_research_note.md](passk_prob_vs_jsd_research_note.md) —
not repeated here. This report covers training-time metrics only: does
the axis change convergence behavior or `best_val_block_eff`.

## Summary — one row per axis

Training convergence/collapse behavior is covered in the Detail section below,
not repeated per-row here — only two axes (LR, grad_clip) actually showed
anything convergence-related; a "convergence" column that's empty for 13 of
15 rows wasn't earning its place.

| hyperparameter | values tried | best_val_block_eff | pass@k |
|---|---|---|---|
| LR, sub-1e-5 boundary | 2e-6, 3e-6 vs 1e-5 | 2e-6 (5.710) ties 3e-6 (5.701); the gap to 1e-5 (5.608, ~0.10) is *also* smaller than this neighborhood's own ~0.20 noise floor — not a confirmed win over 1e-5 either, all three are indistinguishable here | not tested at this sub-axis |
| warmup % | 5, 10, 20 | 20% edges out (5.608 > 5.560 > 5.501), within noise | **[passk note](passk_prob_vs_jsd_research_note.md#ablations-tested--none-changed-the-qualitative-picture-except-n) — 20% clears jsd at all 3 k's, 10% at 2, 5% at 1: directionally consistent, unlike BE** |
| lr_min_ratio | 0.1, 0.01 | no effect (5.608 vs 5.589) | **[passk note](passk_prob_vs_jsd_research_note.md#ablations-tested--none-changed-the-qualitative-picture-except-n) — 0.1 clears jsd at all 3 k's, 0.01 clears none; diverges from the BE result** |
| teacher top-k | 0, 20, 50 | no benefit, mild harm at higher k | [passk note](passk_prob_vs_jsd_research_note.md#ablations-tested--none-changed-the-qualitative-picture-except-n) — all three clear jsd at all 3 k's, no harm despite the BE result |
| weight decay | 0.01, 0.001, 0.0001 | no trend, all within noise | **[passk note](passk_prob_vs_jsd_research_note.md#ablations-tested--none-changed-the-qualitative-picture-except-n) — 0.0001/0.01 clear jsd, 0.001 clears nothing: non-monotonic, worse than both neighbors** |
| grad_clip | 1.0 (default), 10, 100, 1000 | directional rise with allowed magnitude, not clean; only 1000-vs-default gap plausibly clears noise (see Detail for the convergence shape) | [passk note](passk_prob_vs_jsd_research_note.md#ablations-tested--none-changed-the-qualitative-picture-except-n) — only `gradclip100` clears jsd beyond noise |
| teacher_temp | 0.5, 0.7 vs default 1.0 | 0.7 edges out 0.5 and default, within noise | **[passk note](passk_prob_vs_jsd_research_note.md#ablations-tested--none-changed-the-qualitative-picture-except-n) reverses this** — with the default actually included, it clears jsd by the widest margin of the three, not the narrowest |
| grad_accum | 8 (default), 16, 32 | tightest spread of any single-axis probe, no effect | [passk note](passk_prob_vs_jsd_research_note.md) — no effect there either |
| multi-root N (tail-reuse) | 8, 16, 32, +random-offset | all four inside single-root baseline's noise floor, no trend | [passk note](passk_prob_vs_jsd_research_note.md#multi-root-two-different-ways-to-build-the-continuation-at-each-root) — N=16/16+offset/32 clear jsd beyond noise, N=8 doesn't |
| fresh vs tail-reuse estimator | N=16/32, M=1/2, ± CE-anneal | all points inside the same noise band, no BE surprise | [passk note](passk_prob_vs_jsd_research_note.md#multi-root-two-different-ways-to-build-the-continuation-at-each-root) — tail-reuse wins at N=16, fresh only matches it at N=32 |
| training dataset | `math_hard` vs `dapo_math_train` | within noise | [passk note](passk_prob_vs_jsd_research_note.md) — dapo trails `math_eval`, leads `gsm8k_eval`/`olympiad_eval`; real, not uniform |
| M (samples/root) | 4, 8, 16 (warm), 16 (cold) | no trend, all within noise | [passk note](passk_prob_vs_jsd_research_note.md#ablations-tested--none-changed-the-qualitative-picture-except-n) — tracks the individual checkpoint, not M itself |
| CE-anneal `anneal_steps` | 1500, 3000, 4000 | 3000 wins (5.796) | **[passk note](passk_prob_vs_jsd_research_note.md#ablations-tested--none-changed-the-qualitative-picture-except-n) — the BE-best pick (3000) clears jsd nowhere on pass@k; 4000 (worse on BE) clears it at 2 of 3 k's** |
| CE-anneal `aux_weight` | 0.5, 1.0, 2.0 | 0.5 is prob's overall best (5.848) | [passk note](passk_prob_vs_jsd_research_note.md#ablations-tested--none-changed-the-qualitative-picture-except-n), CE-anneal timing row |
| forgetting vs `aux_weight` | 0.5, 1.0, 2.0 | anchor reduces forgetting up to 1.0, rises again at 2.0, doesn't track the BE-optimal 0.5 | n/a |

## Detail — axes with training-dynamics behavior worth more than one line

[LR probe jsd vs prob](../results/passK/lr_probe_jsd_vs_prob.png)

- **LR boundary probe (below 1e-5):** `prob` peaks at `2e-6` (5.710), edging out `3e-6` (5.701) by less than that neighborhood's own noise floor (~0.20) — effectively a tie. The gap to `1e-5` (5.608, ~0.09–0.10) is also smaller than that same ~0.20 floor, so this isn't a confirmed win over `1e-5` either — all three sit within noise of each other, the vanilla sweep's "locked" LR just wasn't demonstrably suboptimal or optimal here. `jsd` holds a flat plateau `3e-6`–`1e-5` (~5.97–5.99), dips below.
- **LR, wide range (up to 1e-4+):** both losses collapse hard above `1e-4`; jsd's cliff is later/gentler than prob's — see [Collapse timing by LR bucket](../results/passK/survival_curves_by_lr.png) below for the actual step-by-step shape.

[Survival curves by LR bucket](../results/passK/survival_curves_by_lr.png)

- **Collapse timing by LR bucket:** healthy range (1e-6–1e-5) never collapses; `1e-4` collapses gradually/staggered by step ~2000; `≥3e-4` collapses almost instantly.

[Vanilla sweep dashboard](../results/passK/sweep_dashboard_vanilla.png)

- **LR (jsd + prob):** both collapse hard above `lr=1e-4`; jsd's cliff is later/gentler than prob's.
- **Warmup {5,10,20%}:** 20% edges out 5%/10% (5.608 > 5.560 > 5.501) — within single-seed noise.
- **lr_min {0.1,0.01}:** no effect (5.608 vs 5.589).
- **Top-k {0,20,50}:** no benefit; mild harm at higher k (5.608 > 5.591 > 5.506).
- **Weight decay {0.01,0.001,0.0001}:** no clear trend, all within noise (5.617/5.608/5.532).

**Grad clip {10,100,1000} vs default (1.0):** `prob`, lr=1e-5, single seed each. The real lever is the effective per-step update magnitude the threshold *allows*, not the clip frequency: raw grad norms are ~50–100 (spiking to 200–700), the same across all four runs, so a threshold of 1.0 or 10 clips ~100% of steps but rescales to a step 10× larger at 10, 100 clips ~half the steps, and 1000 essentially never clips. `best_val_block_eff` rises roughly with allowed magnitude — 5.517 (1.0) → 5.647 (10) → 5.692 (1000) — with 100 (5.535) the odd point out, so the trend is directional, not clean. 1000 vs default is the largest gap (Δ=0.175), just above the ~0.13–0.15 noise floor and the only comparison here that plausibly clears it, still single-seed.

| grad_clip | best_val_block_eff | clip% at convergence | end-of-run (~step 4800) | run |
|---|---|---|---|---|
| 1.0 (default) | 5.517 | ~100% | — | [link](https://wandb.ai/rmukund16-indian-institute-of-technology-hyderabad/distillspec-pipeline/runs/pe2pb5la) |
| 10 | 5.647 | ~100% | ~5.40 | [link](https://wandb.ai/rmukund16-indian-institute-of-technology-hyderabad/distillspec-pipeline/runs/9w41gr72) |
| 100 | 5.535 | ~48% | ~5.40 | [link](https://wandb.ai/rmukund16-indian-institute-of-technology-hyderabad/distillspec-pipeline/runs/xr0bchto) |
| 1000 | 5.692 | ~0% | ~5.48 | [link](https://wandb.ai/rmukund16-indian-institute-of-technology-hyderabad/distillspec-pipeline/runs/1obcwkp6) |

All four grad_clip variants show the same early-peak-then-decline shape in `smoothed_block_eff`: e.g. 1000 peaks ~5.68–5.69 then falls to ~5.48 by step 4800, 100 peaks ~5.52 then falls to ~5.40. The higher peak at 1000 does NOT persist — by end-of-run all thresholds converge to ~5.40–5.48, within noise of each other. Since `ckpt_best` deploys the peak (not the final step), the peak is the number that counts, which is why the table's `best_val_block_eff` differs from the raw chart's right edge. In every case `val/forgetting` climbs in lockstep (0 → ~1.0–1.2) as block_eff falls — the forgetting metric doing its job, not a gradient-sign bug (`train/loss` decreases monotonically throughout). Grad_clip does not prevent this decline at any threshold tested. None of the four points close the gap to jsd's real best-BE deployment pick (5.994) — grad_clip tuning moves prob's BE by ~0.1–0.2 at most, not enough to touch that ~0.3–0.5 gap.

**Teacher temp {0.5,0.7} vs default (1.0):** `prob`, lr=1e-5, single seed each — 0.7 edges out 0.5 and the default (5.643 > 5.602 > 5.517). Spread is 0.126 BE, the same order as other single-axis probes at this exact lr=1e-5 point (grad_clip's spread was 0.130, lr_min's 0.072) — within noise, no confirmed effect on BE.

**Grad_accum {8 (default),16,32}, `prob` lr=1e-6 (CLOSED):** best_smoothed — 8=5.547, 16=5.552, 32=5.510. Spread is 0.042 BE — tighter than every other single-axis probe in this sweep — no detectable effect at any setting. [8](https://wandb.ai/rmukund16-indian-institute-of-technology-hyderabad/distillspec-pipeline/runs/wafdxkbi) · [16](https://wandb.ai/rmukund16-indian-institute-of-technology-hyderabad/distillspec-pipeline/runs/5n0jzhfb) · [32](https://wandb.ai/rmukund16-indian-institute-of-technology-hyderabad/distillspec-pipeline/runs/kciurhqk).

**Multi-root, tail-reuse estimator — N {8,16,32} and random-offset, `prob` lr=1e-5:** N=8=5.708, N=16=5.699, N=32=5.647, N=16+random-offset=5.744 — all four fit inside the 0.10–0.15 noise floor of the single-root baseline (5.608), no monotonic trend with N.

**Fresh vs tail-reuse multi-root estimator, `prob` lr=1e-5:** every fresh-family BE value tested (`freshM1`=5.653/5.609 final, `freshM1_ceanneal3000`=5.779, `freshM2`=5.720/5.671, `N32_freshM1`=5.677/5.607) sits inside the same 5.6–5.8 noise band as tail-reuse itself (N16=5.699, N32=5.647) — no BE surprise anywhere on this axis; every difference across N, M, and CE-anneal-on-fresh comes back within noise on BE. The pass@k story is where this axis actually resolves — see the linked passk note section.

**Training dataset: `dapo_math_train` (DAPO-Math-17k) vs `math_hard`, otherwise identical config (`multiroot N16 freshM1`, `prob` lr=1e-5, wu20):** BE stays within noise (best_smoothed 5.527 dapo vs 5.561 math_hard, Δ=-0.034). [dapo run](https://wandb.ai/rmukund16-indian-institute-of-technology-hyderabad/distillspec-pipeline/runs/omyt093i).

**Single-root M {4,8,16}, `prob` lr=3e-6 (CLOSED):** best_val_block_eff — M4=5.701 (cold-start), M8=5.810 (cold-start), M16=5.619 (warm-started resume after a crash) / 5.694 (clean cold-start redo). No trend with `M`: all values (≈5.62–5.81) fit inside the 0.10–0.15 noise floor. An earlier claim that M16 was genuinely worse was based on the crash-recovery warm-start run and did not hold up once redone cleanly as a cold start.

[CE-anneal sweep dashboard](../results/passK/sweep_dashboard_ceanneal.png)

- **CE-anneal `anneal_steps` {1500,3000,4000}:** 3000 wins (5.796), 4000 close (5.764), 1500 lowest (5.737).
- **CE-anneal `aux_weight` {0.5,1.0,2.0}:** 0.5 is the new overall best for `prob` (5.848), beating 1.0/2.0 (~5.79 both) and the 25k-step extension (5.833).

[Forgetting vs aux_weight](../results/passK/forgetting_vs_aux_weight.png)

- **Forgetting vs `aux_weight`:** anchor reduces forgetting up to `aux_weight=1.0`, then rises again at `2.0` — doesn't track the BE-optimal `0.5`, so the anchor isn't working *purely* through reduced forgetting.

## Caveat

Single seed throughout — gaps under ~0.1 BE are within likely noise, not resolved without repeat seeds. Noise floor itself is non-uniform by LR (e.g. ~0.004 near `3e-6` vs ~0.20 near `2e-6`) — check any claimed gap against the local noise level, not a single sweep-wide number.
