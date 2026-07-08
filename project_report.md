# Speculative Decoding — Draft Distillation: Project Report

**Task:** distil a small draft model to maximise block efficiency (BE) under speculative decoding against a larger teacher.
**Primary metric:** BE = generated tokens / target model calls (higher is better).
**This document is a map, not a results dump.** Every experiment family below has a dedicated research note with full tables, ablations, and mechanism analysis. Raw per-run numbers live in `Results/*.csv` and `results.csv`; wandb run URLs are in each note's ablation table. Read this document to find which note to open, not to read the numbers themselves.

---

## Status at a glance

| Model pair | Status | Best result vs. flat JSD | Note |
|---|---|---|---|
| Qwen3-0.6B draft / Qwen3-8B teacher | **Exploration closed** | `jsd_flat_enrich` (K=4 sweet spot) — only family to consistently beat JSD on traversal, both datasets | see family table below |
| Qwen3-1.7B draft / Qwen3-32B teacher | **First seed swept** | none confirmed; enrich does **not** cleanly replicate (single seed); log-tree actively harmful | see family table + `per_checkpoint_sweeps_2026-07/` |

**Headline finding (0.6B/8B):** every loss family that reweights or restructures training on **teacher-context tokens** (flat divergences, tree losses, prefix-overlap, depth-weighting) plateaus at the same ≈5.9–6.1 traversal-K3 BE ceiling as plain flat JSD. The one family that broke through is **enrichment** (`jsd_flat_enrich`), which trains on *fresh stochastic teacher rollouts* instead of reweighting a fixed teacher-context sequence, and the win is real on the deployment verifier: at 0.6B/8B, mean traversal Δ over K=2–4 is **+0.19 (math) / +0.12 (olympiad)** at the K=4 sweet spot (K=6 degrades — more rollouts stop helping).

**Headline finding (1.7B/32B, first seed):** the enrich win does **not** cleanly replicate at the bigger capacity gap — `enrich_K3` traversal Δ is −0.05/−0.13/**+0.27**/+0.09 across K on math_eval (sign-flips; single seed). Two robust *negative* results emerged instead: (a) **log-space tree losses (`traversal_log`, `nss_log`) are catastrophic at this pair** — traversal Δ ≈ **−0.4 to −0.6 at every K**, far past the ~0.2 noise floor; (b) a **capacity-vs-depth tradeoff** — with *both* JSD baselines fully converged, the 1.7B/32B draft beats 0.6B/8B at shallow speculation (traversal K1/K2 math) but loses at deep speculation (K3/K4) and across olympiad_eval. Not a training-budget artifact (both baselines early-stopped on a genuine val plateau).

---

## Experiment families

Each row is a family of losses tested under the same mechanism; open the note for the full ablation table, gradient analysis, and verdict.

| Family | Losses / flags | Question tested | Verdict | Note |
|---|---|---|---|---|
| Flat baselines | `forward_kl`, `reverse_kl`, `jsd`, `l1` | Which token-matching divergence is strongest? | JSD selected as primary baseline (broadest win across verifiers) | — (see Phase 1–4 history in git log if needed; not carried forward here) |
| Tree losses (on-policy) | `naive_tree`, `traversal_tree`, `kl_tree`, `bv_tree`, `gbv_tree`, `nss_tree`, `specinfer_tree`, `spectr_tree`, `khisti_tree`; log-space variants `traversal_log`, `naive_log`, `nss_log`; off-policy `op_naive_tree(_full)`; REINFORCE `*_tree_pg` (removed) | Does training on the draft's own sampled tree (rather than teacher context) redirect the gradient toward acceptance? | No gain over JSD, and the log-space variants make it **worse**. `bv_tree`/`gbv_tree` collapse to zero gradient (single-token gate). Log-space fixes product-underflow but at 1.7B/32B `traversal_log`/`nss_log` are **catastrophic** — traversal Δ −0.4 to −0.6 at every K (`per_checkpoint_sweeps_2026-07/{traversal,nss}_log_K3_*.csv`). The log wrapper amplifies gradient on low-α (deep, unlikely) nodes, over-fitting the draft to tree tails it can't actually reach. | [`tree_losses_research_note.md`](notes/tree_losses_research_note.md) |
| Depth-weighting | `--aux_mode depth_weight --aux_loss <tree_loss> [--depth_linear \| --depth_lambda λ]` | Does curriculum-reweighting JSD by expected acceptance depth help? | No detectable signal alone (detached scalar, can't redirect gradient direction). One combined run (`jsd + depth_weight_lin + naive_tree`, K3, 40K steps) showed +0.29 traversal — single seed, not confirmed, likely driven by depth_weight not naive_tree. | [`depth_weight_research_note.md`](notes/depth_weight_research_note.md) |
| Enrichment | `jsd_flat_enrich --K <M>` (flat, stochastic teacher rollouts); `enrich_K3` (1.7B/32B); `enrich_draftcond_K3` (draft-conditioned rollouts); `jsd_enrich` (tree variant) | Does training on fresh stochastic teacher paths (not a fixed teacher-context sequence) beat flat JSD? | **Yes at 0.6B/8B — the only positive result.** Flat enrich beats JSD on traversal both datasets; **K=4 is the sweet spot** (mean K2–4 Δ +0.19 math / +0.12 oly), K5 ties, K6 degrades. **Does NOT cleanly replicate at 1.7B/32B** (`enrich_K3` sign-flips across K, single seed). `enrich_draftcond` (draft-conditioned rollouts) now swept K1–4: **underperforms both JSD and standard enrich** — draft-conditioning the rollouts reverses the enrich win (traversal Δ vs flat-enrich −0.08 to −0.19; bv −0.16 to −0.38 at every K), disconfirming the "on-policy beats off-policy" hypothesis for acceptance-matching. | [`enrich_research_note.md`](notes/enrich_research_note.md) |
| Prefix-overlap (PO) | `prefix_overlap --prefix_objective {prob,logprob,traversal,nss}`; cold-start (`po_*_multiN4`, 1.7B/32B) and JSD-warm-start (`po_*_warm*`, 0.6B/8B) | Does explicitly rewarding teacher-prefix log-probability (Rahul's PO estimator, 4 approximations) beat JSD? | No, confirmed at both pairs. Warm-start 0.6B/8B: traversal Δ hovers ±0.1 (redundant with JSD). Cold-start 1.7B/32B: **worse** — and the `traversal` objective (which directly targets the deployment verifier) is the **single worst** variant (Δ −0.27 to −0.45 across K). Small positive blips only on olympiad K3+ for `po_nss`/`po_logprob`, sign-inconsistent. | [`prefix_overlap_research_note.md`](notes/prefix_overlap_research_note.md) |
| Verifier study (DDTE) | eval-only; `traversal_dAdapt`, `traversal_dL{3,4,5}` across all trained checkpoints | Does delayed tree branching help, and does it hold on *trained* drafts (paper only tested untrained)? | **Yes, and it transfers to trained drafts** — best `dL` variant beats base traversal by **+0.15 to +0.78 at K3/K4 for every checkpoint** (incl. JSD baseline, enrich, PO), uplift grows with K. Uplift is slightly *smaller* on enrich models (less headroom — enrich already recovers part of what DDTE adds). | [`ddte_research_note.md`](notes/ddte_research_note.md) |
| LK-α (acceptance-rate distillation) | `lk_alpha`, `lk_alpha_enrich --K <M>` | Does directly optimizing `-log Σ_v min(p,q)` (Samarin et al., arXiv 2602.23881) beat JSD? | Pending run/eval. Mechanistically, `lk_alpha`'s `-log(·)` wrapper gives `∂L/∂θ = (1/α)·∂α/∂θ` — automatic gradient amplification when acceptance is low. Our `naive_log`/`traversal_log` tree losses already carry the identical `1/α` factor (via the `(L-i+1)/α_i` telescoping-log coefficient, see tree-losses note) and still tie/lose to JSD on our independent (non-EAGLE-conditioned) drafts, so this amplification mechanism alone is not expected to be sufficient here — the decisive test is a from-scratch run (no JSD warm-start), not a warm-started one. | no dedicated note yet |

---

## Open question driving current work

Every family above trains on **teacher-context tokens** except enrichment. The uniform ceiling across families is the signature of either (a) JSD already saturating the information content of teacher-context training data, or (b) the 0.6B draft lacking capacity to use a stronger signal even if given one. These are indistinguishable at a single model pair.

**Status:** Qwen3-1.7B / Qwen3-32B first-seed sweep complete (one representative loss per family, cold-start, single seed; per-checkpoint CSVs in `Results/per_checkpoint_sweeps_2026-07/`). Result: no family confirmed to beat JSD on traversal, and enrich's 0.6B/8B win did not replicate — but the two JSD baselines *did* converge fairly (both early-stopped on a val plateau, verified from training logs), so this is a real capacity-signal result, not undertraining. The remaining blocker to any positive claim here is a **second seed** — every 1.7B/32B number is n=1, and the measured noise floor at this pair is still uncharacterized.

---

## Where results live

- **Per-checkpoint eval rows:** `results.csv` (0.6B/8B pair, root of repo) and `Results/*.csv` (named per experiment batch). The **2026-07 full-sweep** (both pairs, every loss family in the tables above, all verifiers × K=1–4 × math_eval+olympiad_eval) lives in [`Results/per_checkpoint_sweeps_2026-07/`](Results/per_checkpoint_sweeps_2026-07/) — one CSV per checkpoint; baselines are `jsd_mathhard_s123.csv` (0.6B/8B) and `jsd_math_hard_s123.csv` (1.7B/32B). Reproduce the deltas cited above with `scripts/analyze_evals.py --glob "Results/per_checkpoint_sweeps_2026-07/*.csv"`.
- **Training curves:** wandb project `distillspec-pipeline`; every run tagged with loss, dataset, K, L, and (as of the 1.7B/32B study) the model pair.
- **Raw logs / stdout:** for the 1.7B/32B study, under `$OUT/Qwen32B-Qwen1.7B/{checkpoints,logs,output}` (with `OUT` your output root).
- **CLI reference:** [`README.md`](README.md) — every flag used above, grouped by experiment family.
