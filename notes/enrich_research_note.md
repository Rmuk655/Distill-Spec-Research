# Enrichment Distillation (`jsd_flat_enrich`)
## Research Note

**Status:** The **one training-side family that beats flat JSD** — but only cleanly at 0.6B/8B. `jsd_flat_enrich` trains flat JSD on **M stochastic teacher rollouts** per prompt (`teacher.generate(do_sample=True)`) instead of one greedy rollout — broader teacher-context coverage, no draft proposals/verifier in the loop. Wins on the traversal (deployment) verifier at 0.6B/8B (sweet spot **M=4**), generalises OOD, but **does not cleanly replicate at 1.7B/32B** (single seed), and the on-policy variant (draft-sampled rollouts) is **negative**. Best current read: enrich is mostly a **variance reducer / initialisation-rescuer**, with a real but small ceiling lift on prefix/budget verifiers.

**Draft–Teacher:** Qwen3-0.6B / Qwen3-8B (primary) and Qwen3-1.7B / Qwen3-32B (cross-pair) | **Train:** math_hard | **Eval:** math_eval + olympiad_eval, temp=0.2, n=100 (n=1000 for the two-seed study) | **Notation:** `M` = training rollouts (`--K` train); `K_eval` = inference branches (`--K` eval).
**JSD baselines (0.6B/8B):** per-checkpoint-sweep `jsd_mathhard_s123` (trav K3 = **5.957**); 40K-converged `jsd_mathhard_s123` (trav K3 = **5.870**). **1.7B/32B:** `jsd_math_hard_s123` (trav K3 = 5.783). *Don't cross the two 0.6/8 baselines.* Noise floor SE ≈ 0.10–0.15 (n=100), ≈0.016 (n=1000).

---

## Experiments & verdicts

| Experiment | Config | What it tested | Beat flat JSD on traversal? | Beat beyond noise (>0.15)? |
|---|---|---|---|---|
| flat enrich M=3 (temp07) | `jsd_flat_enrich --K 3` | stochastic teacher rollouts | Partial — mean K2–4 **+0.13** math / +0.02 oly | temp07 confound; see M=4 |
| **flat enrich M=4** | `--K 4` | rollout-count sweep | **Yes — mean K2–4 +0.185 math / +0.116 oly** | **Yes (bv/traversal)** — sweet spot |
| flat enrich M=5 | `--K 5` | more rollouts | Yes — +0.141 / +0.060 | ties M=4, slight fade |
| flat enrich M=6 | `--K 6` | even more rollouts | Weak — +0.071 / +0.050 | **degrades** (more rollouts stop helping) |
| enrich M=3 @ **1.7B/32B** | `enrich_K3` | cross-pair replication | **No (clean)** — trav Δ −0.05/−0.13/**+0.27**/+0.09 (K1–4); oly +0.16/+0.11/−0.06 | sign-flips; single seed |
| **draftcond M=3** (on-policy) | `enrich_draftcond_K3` | draft-sampled rollouts | **No** — trav Δ −0.11/+0.07/+0.03/+0.06 (within noise) | **worse than teacher-enrich** (bv −0.16 to −0.38) |
| teacher_temp=1.5 | `..._ttemp1.5` | negative control | — (teacher incoherent) | ρ −0.645→−0.543; sanity check only |
| M=1 vs flat-40K | `jsd_flat_enrich_M1` | single stochastic rollout | No — −0.022 | one rollout ≈ flat; M>1 is what moves it |

*(0.6/8 M-sweep Δ vs per-checkpoint baseline 5.957; verified from `per_checkpoint_sweeps_2026-07/`.)*

---

## Findings (what / result / data / conjecture)

1. **M=4 is the sweet spot; M=6 degrades.** Traversal mean K2–4 rises M3→M4 (+0.13→+0.185 math) then falls by M6 (+0.07). Data: `jsd_flat_enrich_K{3_temp07,4,5,6}_*`. Read: a few stochastic paths add coverage; too many just re-sample the same teacher basin (`train/path_diversity` ∈ [0.8,1.0], so not collapse — diminishing returns, not degeneracy).

2. **OOD holds (0.6B/8B).** M=4 transfers to olympiad_eval (mean K2–4 +0.116; +0.286 at K_eval=3). The one positive family generalises out of distribution.

3. **Cross-pair replication fails at 1.7B/32B (single seed).** `enrich_K3` traversal Δ sign-flips across K (−0.05/−0.13/+0.27/+0.09); does not reproduce the clean 0.6/8 win. Only one M (=3), one seed, uncharacterised noise → "did not replicate," not "refuted." Matched-M=4 + second seed is the decisive follow-up. Ties to the **capacity-vs-depth** tradeoff (wider gap wins shallow-K, loses deep-K/OOD).

4. **On-policy / draft-conditioned enrich is NEGATIVE — disconfirms the OPD premise.** `enrich_draftcond` (rollouts sampled from the *draft*) does not beat JSD on traversal (within noise) and is clearly worse than teacher-rollout enrich at every verifier×K (bv/naive −0.16 to −0.38). Mechanism (§bv, below): enrich helps by exposing the draft to the *teacher's* full support (de-sharpening q→p); sampling from the *draft's own* narrow distribution reinforces its modes — the wrong direction, same reason reverse-KL mode-seeking hurts BE. So "student-sampled beats off-policy" (OPD/DOPD) does **not** hold for acceptance-matching here.

5. **Two-seed consistency (0.6B/8B): `bv` is the most cross-seed-robust verifier.** From `jsd_enrich_ddte_results.csv` (train seeds s123/s456, eval seed 123); M3 Δ vs same-seed JSD, both-seed winners ranked by worst case:

   | verifier / K | s123 Δ | s456 Δ | worst-case |
   |---|---|---|---|
   | naive K=3 | +0.263 | +0.314* | +0.263 |
   | bv K=1 | +0.264 | +0.195 | +0.195 |
   | bv K=3 | +0.431 | +0.185* | +0.185 |
   | specinfer K=2 | +0.351 | +0.111 | +0.111 |
   | bv K=2 | +0.232 | +0.101 | +0.101 |

   `bv` wins on both seeds at every K; `traversal` is only marginal (both-seed win at K1 +0.10, but s456 K2 −0.05). *s456 `naive/bv K=3` mix the enrich n=1000 eval vs the n=100 baseline (only s456 rows available at K3) — extra uncertainty; s123 and other cells are matched n=100.

   **bv mechanism conjecture (code-grounded, `verifiers/verifier.py:234`).** bv walks a single path gating each step by `w_i = min(1, w_{i-1}·p/q)` (clamped ≤1) → punished specifically when **q > p**. Greedy-JSD lets q sharpen onto the teacher mode; enrich de-sharpens q toward the teacher's full distribution, relieving exactly that penalty. bv depends on the *marginal* q–p match (enrich stabilises it); traversal depends on *tree branch diversity* (higher seed variance) — why bv's gain is more seed-consistent. `gbv` (same gate + K-skew) is *not* seed-consistent → corroborates. **Status: conjecture.**

6. **bv-consistency does NOT cleanly carry to 1.7B/32B.** `enrich_K3` bv Δ (math) −0.15/+0.43/+0.18/+0.08; at K3 `max` (+0.31)/`specinfer` (+0.28)/`traversal` (+0.27) all exceed bv (+0.18). Which verifier lights up is pair-dependent at single seed.

7. **Variance reduction is the dominant, best-supported effect (n=1000, both seeds).** Flat JSD has a large systematic seed spread (s456−s123 ≈ +0.16–0.19 at every K_eval); enrich collapses it to ≈ −0.03 to −0.15 — rescues the weak seed *up*, trims the strong seed *down* to a common ~6.0–6.05 basin. Best one-line read: **enrich is an initialisation-rescuer / variance reducer, not primarily a ceiling-raiser.** (Needs ≥5 seeds to quantify.)

8. **vs the converged flat-40K baseline: M=3 wins traversal K3, M=1 does not.** M=3-8K beats flat-40K at traversal K3 by **+5.9% (s123) / +2.9% (s456)** — and flat-40K used *more* teacher rollouts (40K vs M3's ~24K), so the lift is not just rollout volume. M=1-8K is −0.022 (no edge). bv advantage over flat-40K is convergence-*speed*, not a raised ceiling (holds s123 +1.9%, reverses s456 −1.4%). Suggestive, not a proof flat can't reach it (single trajectory, 2 seeds, no CIs).

9. **Cross-verifier response is heterogeneous (the strongest finding).** M=3−flat-8K (seed-avg, n=100) positive on 7–9 of 9 verifiers at every K_eval, but K-behaviour splits three ways: **improve with K** (traversal +0.17→+0.25, bv +0.23→+0.31 — prefix/budget), **worsen with K** (gbv collapses to ~0; specinfer/max go negative by K≥3 — residual/competition), **K-agnostic** (naive/spectr/nss/khisti). Gain is *not* on a matched-K=M diagonal (largest at K_eval=1,2) — enrich is trajectory-level coverage, **not** tree-structure training (we claim coverage, not verifier-tree alignment).

10. **ρ diagnostic (supporting only, not load-bearing).** Per-verifier |ρ(JSD,BE)| ordering traversal −0.645 > bv −0.518 > naive −0.413 > specinfer −0.402 tracks how each verifier aggregates acceptance. The flat→M1→M3 ρ strengthening (−0.396→−0.645) is **not** used as evidence (higher M = lower-variance JSD estimate mechanically tightens any correlation). ρ ≠ causation; findings rest on eval BE.

---

## Data provenance & caveats (verified 2026-07)

- **In `Results/`, source of truth:** M-sweep / cross-pair / draftcond in [`per_checkpoint_sweeps_2026-07/`](../Results/per_checkpoint_sweeps_2026-07/); two-seed / cross-verifier / n=1000 in [`jsd_enrich_ddte_results.csv`](../Results/jsd_enrich_ddte_results.csv); converged baseline in the [40K CSV](https://github.com/Rmuk655/Distill-Spec-Research/blob/Pipeline/Results/jsd_jsd_dw_naive_tree_results_math_hard_40000_steps.csv).
- **NOT in committed Results (report from logs, flag before citing):** n=1000 **s123** rows don't exist (only s456 has n=1000, at a subset of K); the ρ / mean_BE / forward-KL / path-diversity **diagnostics are `--diagnose` outputs** with no columns in any committed CSV. Re-export before any paper table depends on them.

---

## Positioning & open directions (theory tail)

- **Design space is 2-D, not a ladder.** *Axis A (states):* greedy teacher → **stochastic teacher (enrich)** → verifier-accepted states → error-anchored replay (Draft-OPD). *Axis B (objective):* uniform JSD (enrich) → γ^{k-1} asymmetric KL (Draft-OPD) → predicted depth/survival weight (`depth_weight`) → exact survival-weighted ∂BE/∂θ (Rahul). Enrich moves **Axis A only, uniform JSD**; the axes compose.
- **Prior work / novelty risk.** The mismatch motivation is not new (DistillSpec, GKD, VSD, SKD, OSD, Draft-OPD). Enrich is the *simplest teacher-sampling point*; defensible claims are narrow: (a) empirical **cross-verifier map** (9 verifiers × K_eval) + per-verifier acceptance-functional explanation; (b) stochastic-teacher JSD as a low-complexity alternative measured against Draft-OPD-style replay under matched compute.
- **Key limitation (now empirically confirmed):** enrich never trains on the draft's *own* rejected states — and the on-policy variant that does (`draftcond`, Finding 4) is negative. So the exposure gap is not closed by moving to student states here.
- **DDTE / GTO relationship.** Composes with DDTE (verification/tree axis; Thomas et al. 2026, arXiv:2602.16994) — DDTE picks the tree, enrich trains the draft that fills it; DDTE's "traversal dominates, divergence grows with depth" corroborates our verifier ordering. **GTO (Hu 2025a)** ≈ `depth_weight` + `tree_pg`, already published → our results on that corner are a replication/negative, not novel.
- **Top future direction — verifier-accepted state distribution.** Train on draft proposals the *verifier accepts* (accepted prefix + teacher-corrected first reject). Blocker is credit through the discrete accept/reject: does Rahul's survival-weighted gradient (from NSS) extend to traversal/BV? If yes → exact; if no → stop-grad surrogate. Distribution moves as the draft learns (replay buffer / trust region).
- **Must-add before a paper:** ≥3 seeds (flat + enrich, both pairs); paired bootstrap CIs on headline cells; **compute-matched** flat baseline; Draft-OPD replay ablation; ≥1 more dataset + model pair; mild teacher-temp sweep (0.4/0.7/1.0); wall-clock tok/s + output quality.
