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
| M (samples/root) | 4, 8, 16 (warm), 16 (cold) | M4=5.701, M8=5.810, M16=5.619 (warm) / 5.694 (cold redo) — no trend, all within noise (see Detail for the M16 caveat) | [passk note](passk_prob_vs_jsd_research_note.md#ablations-tested--none-changed-the-qualitative-picture-except-n) — tracks the individual checkpoint, not M itself |
| CE-anneal `anneal_steps` | 1500, 3000, 4000 | 3000 wins (5.796) | **[passk note](passk_prob_vs_jsd_research_note.md#ablations-tested--none-changed-the-qualitative-picture-except-n) — the BE-best pick (3000) clears jsd nowhere on pass@k; 4000 (worse on BE) clears it at 2 of 3 k's** |
| CE-anneal `aux_weight` | 0.5, 1.0, 2.0 | 0.5 is prob's overall best (5.848) | [passk note](passk_prob_vs_jsd_research_note.md#ablations-tested--none-changed-the-qualitative-picture-except-n), CE-anneal timing row |
| forgetting vs `aux_weight` | 0.5, 1.0, 2.0 | anchor reduces forgetting up to 1.0, rises again at 2.0, doesn't track the BE-optimal 0.5 | n/a |

## Detail — axes with content beyond what the Summary table cells already say

Charts: [LR probe jsd vs prob](../results/passK/lr_probe_jsd_vs_prob.png) ·
[Collapse timing by LR bucket](../results/passK/survival_curves_by_lr.png) ·
[Vanilla sweep dashboard](../results/passK/sweep_dashboard_vanilla.png) ·
[CE-anneal sweep dashboard](../results/passK/sweep_dashboard_ceanneal.png) ·
[Forgetting vs aux_weight](../results/passK/forgetting_vs_aux_weight.png).

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

## Caveat

Single seed throughout — gaps under ~0.1 BE are within likely noise, not resolved without repeat seeds. Noise floor itself is non-uniform by LR (e.g. ~0.004 near `3e-6` vs ~0.20 near `2e-6`) — check any claimed gap against the local noise level, not a single sweep-wide number.
