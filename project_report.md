# Speculative Decoding — Draft Distillation: Project Report

**Task:** distil a small draft model to maximise block efficiency (BE) under speculative decoding against a larger teacher.
**Primary metric:** BE = generated tokens / target model calls (higher is better).
**This document is a map, not a results dump.** Every experiment family below has a dedicated research note with full tables, ablations, and mechanism analysis. Raw per-run numbers live in `Results/*.csv` and `results.csv`; wandb run URLs are in each note's ablation table. Read this document to find which note to open, not to read the numbers themselves.

---

## Status at a glance

| Model pair | Status | Best result vs. flat JSD | Note |
|---|---|---|---|
| Qwen3-0.6B draft / Qwen3-8B teacher | **Exploration closed** | `jsd_flat_enrich` K=3 — only variant to consistently beat JSD | see family table below |
| Qwen3-1.7B draft / Qwen3-32B teacher | **In progress** | — | no note yet; see `memory/qwen17b-qwen32b-capacity-study` (not in-repo) |

**Headline finding (0.6B/8B):** every loss family that reweights or restructures training on **teacher-context tokens** (flat divergences, tree losses, prefix-overlap, depth-weighting) plateaus at the same ≈5.9–6.1 traversal-K3 BE ceiling as plain flat JSD. The one family that broke through is **enrichment** (`jsd_flat_enrich`), which trains on *fresh stochastic teacher rollouts* instead of reweighting a fixed teacher-context sequence. This is the basis for testing whether the ceiling is a **training-signal ceiling** (JSD already extracts everything teacher-context data offers) or a **draft-capacity ceiling** — the question the 1.7B/32B pair is designed to answer.

---

## Experiment families

Each row is a family of losses tested under the same mechanism; open the note for the full ablation table, gradient analysis, and verdict.

| Family | Losses / flags | Question tested | Verdict | Note |
|---|---|---|---|---|
| Flat baselines | `forward_kl`, `reverse_kl`, `jsd`, `l1` | Which token-matching divergence is strongest? | JSD selected as primary baseline (broadest win across verifiers) | — (see Phase 1–4 history in git log if needed; not carried forward here) |
| Tree losses (on-policy) | `naive_tree`, `traversal_tree`, `kl_tree`, `bv_tree`, `gbv_tree`, `nss_tree`, `specinfer_tree`, `spectr_tree`, `khisti_tree`; log-space variants `traversal_log`, `naive_log`, `nss_log`; off-policy `op_naive_tree(_full)`; REINFORCE `*_tree_pg` (removed) | Does training on the draft's own sampled tree (rather than teacher context) redirect the gradient toward acceptance? | No consistent gain over JSD. `bv_tree`/`gbv_tree` are algorithmically broken (gradient collapses to 0). `tree_pg` REINFORCE was structurally wrong (sharpens q instead of spreading it) — removed from the codebase. Log-space fix solves gradient vanishing but not the ceiling. | [`tree_losses_research_note.md`](notes/tree_losses_research_note.md) |
| Depth-weighting | `--aux_mode depth_weight --aux_loss <tree_loss> [--depth_linear \| --depth_lambda λ]` | Does curriculum-reweighting JSD by expected acceptance depth help? | No detectable signal alone (detached scalar, can't redirect gradient direction). One combined run (`jsd + depth_weight_lin + naive_tree`, K3, 40K steps) showed +0.29 traversal — single seed, not confirmed, likely driven by depth_weight not naive_tree. | [`depth_weight_research_note.md`](notes/depth_weight_research_note.md) |
| Enrichment | `jsd_flat_enrich --K <M>` (flat, stochastic teacher rollouts), `jsd_enrich` (tree variant, same idea) | Does training on fresh stochastic teacher paths (not a fixed teacher-context sequence) beat flat JSD? | **Yes — the only positive result.** M=3 beats seed-averaged flat JSD across most verifiers/K_eval; gain is seed-magnitude-dependent (rescues weak seeds more than strong ones). | [`enrich_research_note.md`](notes/enrich_research_note.md) |
| Prefix-overlap (PO) | `prefix_overlap --prefix_objective {prob,logprob,traversal,nss} --prefix_root_spacing N` | Does explicitly rewarding teacher-prefix log-probability (Rahul's PO estimator, 4 approximations) beat JSD? | No — all four objectives are redundant with what JSD already maximises; `traversal` additionally suffers gradient attenuation at product-form depth (fixed but still no gain). | [`prefix_overlap_research_note.md`](notes/prefix_overlap_research_note.md) |
| Verifier study (DDTE) | eval-only, no training | Which verifier best predicts real speedup / is hardest to beat? | Traversal dominates OT-based verifiers; divergence from OT baselines grows with tree depth. | [`ddte_research_note.md`](notes/ddte_research_note.md) |

---

## Open question driving current work

Every family above trains on **teacher-context tokens** except enrichment. The uniform ceiling across families is the signature of either (a) JSD already saturating the information content of teacher-context training data, or (b) the 0.6B draft lacking capacity to use a stronger signal even if given one. These are indistinguishable at a single model pair.

**Current experiment:** Qwen3-1.7B draft / Qwen3-32B teacher, one representative loss per family above, all cold-start, single seed, run to a fixed 15K-step floor (see `train.py --min_steps` / `--divergence_abort_frac`, added for this study — early-stop no longer fires before the floor, only a genuine >20% divergence aborts early). If the ceiling persists at 1.7B, it's a training-signal ceiling; if any family breaks through, it's capacity.

---

## Where results live

- **Per-checkpoint eval rows:** `results.csv` (0.6B/8B pair, root of repo) and `Results/*.csv` (named per experiment batch — see filenames for which sweep).
- **Training curves:** wandb project `distillspec-pipeline`; every run tagged with loss, dataset, K, L, and (as of the 1.7B/32B study) the model pair.
- **Raw logs / stdout:** for the 1.7B/32B study, `/sensei-fs-3/users/rkrishna/Qwen32B-Qwen1.7B/{checkpoints,logs,output}`.
- **CLI reference:** [`README.md`](README.md) — every flag used above, grouped by experiment family.
