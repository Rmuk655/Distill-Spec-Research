# Hyperparameter sweep — research report

0.6B/8B, math_hard, 5k steps, single seed unless noted. Raw CSVs + all charts: [results/passK](https://github.com/Rmuk655/Distill-Spec-Research/tree/Pipeline/results/passK)

![vanilla sweep dashboard](../results/passK/sweep_dashboard_vanilla.png)

- **LR (jsd + prob):** both collapse hard above `lr=1e-4`; jsd's cliff is later/gentler than prob's.
- **Warmup {5,10,20%}:** 20% edges out 5%/10% (5.608 > 5.560 > 5.501) — within single-seed noise.
- **lr_min {0.1,0.01}:** no effect (5.608 vs 5.589).
- **Top-k {0,20,50}:** no benefit; mild harm at higher k (5.608 > 5.591 > 5.506).
- **Weight decay {0.01,0.001,0.0001}:** no clear trend, all within noise (5.617/5.608/5.532).

![LR probe jsd vs prob](../results/passK/lr_probe_jsd_vs_prob.png)

- **LR boundary probe (below 1e-5):** `prob` peaks at `2e-6` (5.710), edging out `3e-6` (5.701) by less than that neighborhood's own noise floor (~0.20) — effectively a tie, both well clear of `1e-5` (5.608). The vanilla sweep's "locked" LR wasn't optimal either way. `jsd` holds a flat plateau `3e-6`–`1e-5` (~5.97–5.99), dips below.

![CE-anneal sweep dashboard](../results/passK/sweep_dashboard_ceanneal.png)

- **CE-anneal `anneal_steps` {1500,3000,4000}:** 3000 wins (5.796), 4000 close (5.764), 1500 lowest (5.737).
- **CE-anneal `aux_weight` {0.5,1.0,2.0}:** 0.5 is the new overall best for `prob` (5.848), beating 1.0/2.0 (~5.79 both) and the 25k-step extension (5.833).

![survival curves by LR bucket](../results/passK/survival_curves_by_lr.png)

- **Collapse timing by LR bucket:** healthy range (1e-6–1e-5) never collapses; `1e-4` collapses gradually/staggered by step ~2000; `≥3e-4` collapses almost instantly.

![forgetting vs aux_weight](../results/passK/forgetting_vs_aux_weight.png)

- **Forgetting vs `aux_weight`:** anchor reduces forgetting up to `aux_weight=1.0`, then rises again at `2.0` — doesn't track the BE-optimal `0.5`, so the anchor isn't working *purely* through reduced forgetting.

![pass@k by checkpoint](../results/passK/passk_by_checkpoint_math_eval_06b_8b.png)
![pass@k by model size](../results/passK/passk_by_model_size_all_datasets.png)

- **Pass@k:** computation confirmed correct (monotonic in k everywhere); 2 of 5 checkpoints trail the untrained draft at high k despite beating it at k=1 — training narrows response diversity even as it raises single-shot accuracy (see [passk_diversity_collapse_research_note.md](passk_diversity_collapse_research_note.md)).

![gap closed math_eval](../results/passK/gap_closed_math_eval.png)
![gap closed gsm8k_eval](../results/passK/gap_closed_gsm8k_eval.png)
![gap closed olympiad_eval](../results/passK/gap_closed_olympiad_eval.png)

- **Who lifts pass@k closest to teacher, best-of-sweep (% of student→teacher gap closed, mean over k2/k4/k8/k16):** best is dataset-dependent — `po_prob_lr7e-6_wu20` on gsm8k (51.7%), `po_prob_lr1e-5_wu5_lrmin0.1_wd0.01` on olympiad (65.1%), `po_prob_ceanneal1500_lr1e-5_wu20` on math_eval (40.4%, much lower ceiling). **Caveat: this is the best of ~14-20 variants per family, picked after seeing pass@k — a hindsight/multiple-comparisons selection, not a config you'd choose in advance.**
- **jsd vs prob, fair comparison (the ONE config per family you'd actually deploy, chosen by block_eff *before* looking at pass@k):** `po_prob_ceanneal3000_auxw0.5_lr1e-5_wu20` (prob's real best-BE pick, 5.848) vs `jsd_lr1e-5_wu10_lrmin0.1_wd0.01` (jsd's real best-BE pick, 5.994) — **every difference at every k, every dataset is within the noise floor.** The "prob beats jsd" story above only shows up when cherry-picking prob's best-of-many against jsd's best-of-few; at the actual deployment configs, loss choice makes no verified practical dent in pass@k here, despite jsd's clearly better block_eff. BE and pass@k are different axes, but this data does NOT support "prob wins on pass@k" as a real effect.

![prob vs jsd gap closed](../results/passK/prob_vs_jsd_gap_closed.png)
![prob advantage over jsd](../results/passK/prob_advantage_over_jsd.png)
![pass@k curves math_eval](../results/passK/passk_curves_all_math_eval.png)
![pass@k curves gsm8k_eval](../results/passK/passk_curves_all_gsm8k_eval.png)
![pass@k curves olympiad_eval](../results/passK/passk_curves_all_olympiad_eval.png)

- **Same check with a plain LR-only prob pick (no CE-anchor at all, `lr=2e-6`/`3e-6`, prob's actual best-BE points) vs jsd's best-BE pick:** still 0 of 12 (3 datasets × k2/4/8/16) comparisons clear the noise floor individually. But prob's raw pass@k is numerically higher in all 12/12 — a fully consistent one-sided direction that would be a coincidence under a true null. Suggestive of a small real effect this single-seed data can't confirm point-by-point, not proof of one. At matched LR (`prob lr=1e-5` vs `jsd lr=1e-5`), one comparison (gsm8k k16) does clear the floor (+0.0385 vs floor 0.0313) — the one individually-significant hit found across all these checks.

## Caveat

Single seed throughout — gaps under ~0.1 BE are within likely noise, not resolved without repeat seeds. Noise floor itself is non-uniform by LR (e.g. ~0.004 near `3e-6` vs ~0.20 near `2e-6`) — check any claimed gap against the local noise level, not a single sweep-wide number.
