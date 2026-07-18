# Hyperparameter sweep — research report

0.6B/8B, math_hard, 5k steps, single seed unless noted. Raw CSVs + all charts: [Results/passK](https://github.com/Rmuk655/Distill-Spec-Research/tree/Pipeline/Results/passK)

![vanilla sweep dashboard](../Results/passK/sweep_dashboard_vanilla.png)

- **LR (jsd + prob):** both collapse hard above `lr=1e-4`; jsd's cliff is later/gentler than prob's.
- **Warmup {5,10,20%}:** 20% edges out 5%/10% (5.608 > 5.560 > 5.501) — within single-seed noise.
- **lr_min {0.1,0.01}:** no effect (5.608 vs 5.589).
- **Top-k {0,20,50}:** no benefit; mild harm at higher k (5.608 > 5.591 > 5.506).
- **Weight decay {0.01,0.001,0.0001}:** no clear trend, all within noise (5.617/5.608/5.532).

![LR probe jsd vs prob](../Results/passK/lr_probe_jsd_vs_prob.png)

- **LR boundary probe (below 1e-5):** `prob` peaks at `3e-6` (5.701), not `1e-5` — the vanilla sweep's "locked" LR wasn't optimal. `jsd` holds a flat plateau `3e-6`–`1e-5` (~5.97–5.99), dips below.

![CE-anneal sweep dashboard](../Results/passK/sweep_dashboard_ceanneal.png)

- **CE-anneal `anneal_steps` {1500,3000,4000}:** 3000 wins (5.796), 4000 close (5.764), 1500 lowest (5.737).
- **CE-anneal `aux_weight` {0.5,1.0,2.0}:** 0.5 is the new overall best for `prob` (5.848), beating 1.0/2.0 (~5.79 both) and the 25k-step extension (5.833).

![survival curves by LR bucket](../Results/passK/survival_curves_by_lr.png)

- **Collapse timing by LR bucket:** healthy range (1e-6–1e-5) never collapses; `1e-4` collapses gradually/staggered by step ~2000; `≥3e-4` collapses almost instantly.

![forgetting vs aux_weight](../Results/passK/forgetting_vs_aux_weight.png)

- **Forgetting vs `aux_weight`:** anchor reduces forgetting up to `aux_weight=1.0`, then rises again at `2.0` — doesn't track the BE-optimal `0.5`, so the anchor isn't working *purely* through reduced forgetting.

![pass@k by checkpoint](../Results/passK/passk_by_checkpoint_math_eval_06b_8b.png)
![pass@k by model size](../Results/passK/passk_by_model_size_all_datasets.png)

- **Pass@k:** computation confirmed correct (monotonic in k everywhere); 2 of 5 checkpoints trail the untrained draft at high k despite beating it at k=1 — training narrows response diversity even as it raises single-shot accuracy (see [passk_diversity_collapse_research_note.md](passk_diversity_collapse_research_note.md)).

## Caveat

Single seed throughout — gaps under ~0.1 BE are within likely noise, not resolved without repeat seeds. Noise floor itself is non-uniform by LR (e.g. ~0.004 near `3e-6` vs ~0.20 near `2e-6`) — check any claimed gap against the local noise level, not a single sweep-wide number.
