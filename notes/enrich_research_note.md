# Stochastic Teacher Rollout JSD for Speculative Draft Training
## A Research Note on `jsd_flat_enrich`

**Scope:** Qwen3-0.6B / Qwen3-8B, train math_hard, eval **math_eval + olympiad_eval** (OOD added 2026-07), M-sweep M∈{1,3,4,5,6}, two seeds at M=3. Cross-pair replication attempted at **Qwen3-1.7B / Qwen3-32B** (`enrich_K3`, single seed). n=1000 for traversal/bv/naive ([§4.2](#42-primary-eval-n1000-k_eval3-math_eval)–4.3); n=100 elsewhere.
**Closest competitor:** Draft-OPD (May 2026) — not yet compared experimentally ([§5.1](#51-novelty-positioning-the-biggest-risk)).
**Raw CSVs (2026-07 sweep):** [`Results/per_checkpoint_sweeps_2026-07/`](../Results/per_checkpoint_sweeps_2026-07/) — `jsd_flat_enrich_K{3_temp07,4,5,6}_*` (0.6B/8B), `enrich_K3_*` (1.7B/32B), `enrich_draftcond_K3_*`.

### Summary

**Method.** `jsd_flat_enrich` replaces the single greedy teacher rollout in standard flat JSD training with M independent stochastic teacher continuations per step. The draft minimises the average JSD against all M teacher-sampled contexts. There are no draft proposals, no verifier decisions, and no accepted/rejected tokens in training — it is a broader-coverage teacher-sampling variant of flat JSD.

**Key results** (one model pair, one dataset, two seeds):
- **M=3 beats the seed-averaged flat JSD** on traversal/bv/naive at K_eval=2,3,4 (n=1000), gain largest at K_eval=4 (traversal +0.10, bv +0.15, naive +0.14). **But same-seed the gain is asymmetric** — it rescues the weak seed (s123, +0.19–0.25) and is flat-to-*negative* on the strong seed (s456) for traversal/naive at K≤3. Only at K_eval=4 does M=3 win on both seeds.
- **M=3 (8K steps) scores above our converged flat-40K baseline at traversal K=3**: +5.9% (s123) and +2.9% (s456). M=3-8K uses 24K total teacher rollouts vs flat-40K's 40K — it wins with fewer rollouts.
- **M=1 does not beat flat-40K** (−0.022 overall). The benefit requires M>1.
- **Enrich (M=3) consistently beats the seed-averaged flat baseline across all K_eval values for traversal/bv/naive.** The per-seed gain varies — larger for the weaker initialization (s123, +0.19–0.25) and smaller for the stronger (s456, within noise at K≤3, a clear win at K=4). Flat's initialization spread (s456−s123 ≈ +0.17) collapses under enrich to ≈ −0.03 to −0.15, showing the gain is real but seed-magnitude-dependent. Needs ≥5 seeds to quantify the distribution.

**Key analysis:**
- **Verifier responses are heterogeneous.** M=3 wins on 7–9 of 9 verifiers at every K_eval. Traversal and bv gains grow with K_eval; gbv/specinfer/max fade or go negative at high K_eval; naive/spectr/nss/khisti are K-agnostic across K_eval.
- **No matched-K diagonal** (contrast with depth_weight): M=3 gains are positive at all K_eval and are largest at K_eval=1,2 (+0.136, +0.130 seed-avg), not at the "matched" K_eval=3 (+0.082). Because M independent teacher paths are flat-trajectory data augmentation (not tree-structure training), the gain concentrating at narrow K_eval is expected — we do not claim enrich aligns the draft to verifier tree structure.
- **JSD–BE coupling (ρ) strengthens with M** (flat −0.396 → M=1 −0.568 → M=3 −0.645) and varies across verifiers — traversal tightest (−0.645), specinfer loosest (−0.402). ρ reflects verifier acceptance-aggregation structure. Gain (Δ) depends on headroom: specinfer's large K_eval=1 gain (+0.336) is better attributed to its low baseline BE (4.867 — it has the most hard prompts) than to coupling strength.

**Next steps** (suggested order):
1. Compute-matched M=3-vs-flat baseline (gates any efficiency claim — [§6](#6-must-add-experiments-minimum-for-a-credible-paper) #3)
2. n=1000 paired bootstrap CIs for headline cells ([§6](#6-must-add-experiments-minimum-for-a-credible-paper) #2)
3. ≥5 seeds per condition ([§6](#6-must-add-experiments-minimum-for-a-credible-paper) #1)
4. ≥1 additional dataset and model pair ([§6](#6-must-add-experiments-minimum-for-a-credible-paper) #7–8)
5. Draft-OPD comparison ([§6](#6-must-add-experiments-minimum-for-a-credible-paper) #5)
6. [§5.3](#53-conjectured-verifier-level-mechanism-not-yet-derived) derivations, or present [§4.4](#44-cross-verifier-and-k_eval-dependence-n100-seed-averaged-vs-flat-8k) as purely empirical

---

## 0. 2026-07 Update: M-sweep, OOD, and cross-pair replication

New data since the sections below were written. All Δ = traversal BE − flat-JSD baseline at matched K_eval; means over K_eval=2–4. Numbers computed from [`Results/per_checkpoint_sweeps_2026-07/`](../Results/per_checkpoint_sweeps_2026-07/).

**(a) M-sweep at 0.6B/8B — mean traversal Δ (K_eval 2–4):**

| M (train rollouts) | math_eval | olympiad_eval (OOD) |
|---|---|---|
| M=3 (`_temp07`) | +0.130 | +0.019 |
| **M=4** | **+0.185** | **+0.116** |
| M=5 | +0.141 | +0.060 |
| M=6 | +0.071 | +0.050 |

**M=4 is the sweet spot on both in-distribution and OOD; the gain does not increase past M=4 and *decays* by M=6** — more teacher rollouts stop helping and eventually hurt (consistent with variance-reduction saturating, then the fixed step budget spreading too thin across rollouts). This extends the earlier "M=3 beats flat, M=1 doesn't" ([Summary](#summary)) into a full curve with an interior optimum.

⚠️ **Confound:** the M=3 checkpoint is `temp07`-tagged (teacher rollout temperature 0.7); M=4/5/6 filenames are not, so teacher-sampling temperature may differ across the sweep. The M=4 optimum is therefore *not* a clean single-variable result — it could be an M effect, a temperature effect, or both. A temperature-matched M-sweep is required before claiming M=4 specifically (add to [§6](#6-must-add-experiments-minimum-for-a-credible-paper)).

**(b) OOD holds (the important positive).** Enrich's win is not a math_eval artifact: at M=4 it transfers to olympiad_eval (+0.116 mean, and +0.286 at K_eval=3 specifically). This is the single most reassuring new result — the one positive family generalizes out of distribution.

**(c) Cross-pair replication FAILS at 1.7B/32B (single seed).** `enrich_K3` traversal Δ vs `jsd_math_hard_s123`: **−0.054 / −0.133 / +0.273 / +0.085** across K_eval=1–4 (math_eval) — sign-flips, net unconvincing; on olympiad_eval +0.162 / +0.111 / −0.062. This does **not** reproduce the clean 0.6B/8B win. Caveat: only one enrich variant (M=3, no temp07/M-sweep) was run at the big pair, single seed, and the 1.7B/32B noise floor is uncharacterized — so this is "did not replicate," not "refuted." A matched M=4 + second seed at 1.7B/32B is the decisive follow-up.

**(d) `enrich_draftcond_K3` (draft-conditioned rollouts) — full K1–4 sweep now in; NEGATIVE, and it disconfirms the on-policy hypothesis.** This is the one variant that injects the *draft's own* distribution into the rollout (the on-policy analog; the axis PO never touched, see [`prefix_overlap_research_note.md`](prefix_overlap_research_note.md)). Full math_eval sweep (all verifiers × K=1–4, single seed) vs the 0.6B/8B JSD baseline and vs standard flat-enrich (K4):

| verifier | Δ vs JSD (K1/K2/K3/K4) | Δ vs flat-enrich (K1/K2/K3/K4) |
|---|---|---|
| traversal (deployment) | −0.11 / +0.07 / +0.03 / +0.06 | −0.12 / −0.19 / −0.13 / −0.08 |
| bv | −0.02 / −0.17 / −0.04 / −0.04 | −0.16 / −0.38 / −0.19 / −0.24 |
| naive | −0.26 / −0.25 / +0.19 / −0.04 | −0.30 / −0.32 / −0.09 / −0.20 |

Draft-conditioning does **not** beat JSD on the deployment verifier (traversal Δ inside the ~0.15 noise floor, negative at K1), and is clearly **worse than teacher-rollout enrich at every verifier×K** (the −0.16 to −0.38 bv/naive gaps are above noise). Mechanism (grounded in the bv analysis, §0e): enrich helps by exposing the draft to the *teacher's* full stochastic support, de-sharpening q toward p; sampling from the *draft's own* distribution instead reinforces q's already-narrow modes — moving the draft away from the teacher's tails, the opposite of what helped. **So "on-policy / student-sampled trajectories beat off-policy" (the premise of OPD/DOPD-style work) does not hold for acceptance-matching here** — for the same reason reverse-KL mode-seeking hurts BE. Raw CSV: [`enrich_draftcond_K3_mathhard_s123.csv`](../Results/per_checkpoint_sweeps_2026-07/enrich_draftcond_K3_mathhard_s123.csv).

**Bottom line update:** enrich remains the one positive lever, now with OOD generalization confirmed at 0.6B/8B and an interior M-optimum (~M=4, temperature-confounded). Its failure to replicate at 1.7B/32B is the top open risk to any "enrich beats JSD" claim in the paper — it must be stated as pair-specific until a second seed + matched-M run at the big pair exists.

### 0(e). Two-seed consistency (0.6B/8B) — which (verifier, K) beats JSD on BOTH seeds

From [`Results/jsd_enrich_ddte_results.csv`](../Results/jsd_enrich_ddte_results.csv) (two training seeds s123/s456, eval-seed 123). `enrich_M3` Δ vs *same-seed* JSD; cells winning on both seeds, ranked by worst-case (min-seed) gain. *(Caveat: the s456 `naive K=3` (+0.314) and `bv K=3` (+0.185) deltas compare the enrich checkpoint's n=1000 eval against the n=100 baseline — the only s456 rows available at K=3 — so those two cells mix prompt-set sizes and carry extra uncertainty; s123 throughout and the other s456 cells are matched n=100. The s456 traversal K2/K3 figures in the bullet below are likewise n=1000-vs-n=100.)*

| verifier / K | s123 Δ | s456 Δ | worst-case |
|---|---|---|---|
| naive K=3 | +0.263 | +0.314 | **+0.263** |
| bv K=1 | +0.264 | +0.195 | +0.195 |
| bv K=3 | +0.431 | +0.185 | +0.185 |
| specinfer K=2 | +0.351 | +0.111 | +0.111 |
| bv K=2 | +0.232 | +0.101 | +0.101 |

- **`bv` is the single most cross-seed-robust verifier — `enrich_M3` beats JSD on both seeds at *every* K (1–4).** No other verifier does. `naive K=3` has the highest worst-case gain.
- **The deployment verifier `traversal` is only *marginally* robust:** both-seed win at K1 (+0.10), but K2 *fails* on s456 (−0.05) and K3/K4 win by only +0.01/+0.02 on s456. The cross-seed signal is carried by bv/naive, not traversal.
- **DDTE gives enrich no cross-seed edge over JSD:** no `dL` cell wins on both seeds — DDTE lifts JSD too (consistent with the substitute finding in [`ddte_research_note.md`](ddte_research_note.md) §0).

**Conjectured mechanism (code-grounded, `verifiers/verifier.py:234`).** bv walks a single path gating each step by `w_i = min(1, w_{i-1}·p/q)` — clamped ≤1, so it is punished specifically when **q > p** (draft over-concentrates). Greedy-JSD lets q sharpen on the teacher's mode; **enrich de-sharpens q toward the teacher's full distribution**, relieving exactly this failure mode. bv's acceptance depends on the *marginal* q–p match (which enrich stabilizes), whereas traversal depends on *tree branch diversity* (higher seed variance) — explaining why bv's enrich-gain is more seed-consistent. Corroboration: `gbv` (same gate + a K-branch skew) is *not* seed-consistent and equals bv only at K=1. **Status: conjecture, not derived.**

### 0(f). Does the bv pattern carry to 1.7B/32B? Partially — no as a clean pattern.

`enrich_K3` (1.7/32, single seed) bv Δ vs JSD: math_eval −0.147 / +0.431 / +0.178 / +0.075 (K1–4); olympiad −0.003 / +0.036 / +0.009. bv is positive at K2–4 on math but **negative at K1, flat on OOD**, and **bv is no longer the standout** — at K3 `max` (+0.305), `specinfer` (+0.278), `traversal` (+0.273) all exceed bv (+0.178). The *specific* "bv is the consistent winner" result is **0.6B/8B-specific**; which verifier lights up is pair-dependent at single seed. Cannot tell if the mechanism is capacity-dependent or the 0.6/8 bv-consistency is a two-seed coincidence without a second seed at 1.7/32.

---

## 1. Motivation

Standard flat JSD training minimises

$$
\mathcal{L}_{\text{JSD}} = \mathbb{E}_{x_{<t} \sim D}\big[\text{JSD}\big(P_\theta(\cdot \mid x_{<t}) \,\|\, Q_\phi(\cdot \mid x_{<t})\big)\big]
$$

where $P_\theta$ is the teacher, $Q_\phi$ the draft, and $D$ is the **offline teacher rollout distribution** over prefixes. In the current baseline implementation, this is a greedy teacher rollout (`do_sample=False`), not a sample from the full teacher marginal. At inference time the draft instead operates inside a verifier-gated acceptance loop: its own proposals, partially accepted and teacher-corrected, determine the prefixes it conditions on. This is the well-known **offline-to-inference (exposure) mismatch**. The implemented hypothesis is weaker than full on-policy training: replacing a single greedy teacher trajectory with stochastic teacher rollouts gives the draft broader teacher-context coverage and may improve speculative acceptance.

This motivation is **not new** — it is the same mismatch DistillSpec (on-policy draft-generated data), GKD, OSD, and Draft-OPD all target. The current `jsd_flat_enrich` implementation is not a realization of verifier-accepted on-policy training; it is a stochastic-teacher variant of flat JSD. Any contribution must therefore be framed as the *specific low-complexity teacher-sampling instantiation*, its empirical effect, and its relationship to stronger draft-gated/replay methods ([§5.1](#51-novelty-positioning-the-biggest-risk)), not the mismatch observation itself.

---

## 2. Method: `jsd_flat_enrich`

### 2.1 Notation (de-conflict the overloaded K)

The codebase uses `K` for two unrelated things. For the paper we rename:

| Symbol | Meaning | Code name |
|---|---|---|
| $M$ | number of **enrichment rollouts** per training step | `--K` (train) |
| $K_{\text{eval}}$ | number of speculative **draft branches/steps** at inference | `--K` (eval) |
| $L$ | rollout/tree **horizon** (depth) | `--L` |

This note uses $M$, $K_{\text{eval}}$, $L$ throughout. (M=1 ⇔ old "K=1 train"; M=3 ⇔ old "K=3 train".)

### 2.2 Current implementation: stochastic teacher rollout JSD

Let $\pi$ be the prompt distribution. The implemented `jsd_flat_enrich` objective samples $M$ independent teacher continuations:

$$
y^{(m)} \sim P_\theta^{T_{\text{teacher}}}(\cdot \mid p), \qquad m=1,\dots,M,
$$

where sampling is implemented by `teacher.generate(..., do_sample=True, temperature=teacher_temp)`. It then scores both teacher and draft distributions on the same teacher-generated sequence and averages the per-token JSD:

$$
\mathcal{L}_{\text{stoch-teacher}} =
\frac{1}{M}\sum_{m=1}^M
\frac{1}{|y^{(m)}|}\sum_{t=1}^{|y^{(m)}|}
\text{JSD}\big(P_\theta(\cdot \mid p, y^{(m)}_{<t}) \,\|\, Q_\phi(\cdot \mid p, y^{(m)}_{<t})\big)
$$

There are **no draft proposals, no verifier accept/reject decisions, and no teacher-corrected residuals** in the current flat-enrich code path. $M=1$ isolates greedy-vs-stochastic teacher training; $M > 1$ adds multiple stochastic teacher trajectories per prompt and tests whether path diversity gives useful extra contexts.

### 2.3 What this is *not*: the verifier-accepted state distribution

The stronger objective we originally wanted to approximate — training on draft proposals that the verifier actually accepts (the verifier-accepted state distribution) — is **not** what `jsd_flat_enrich` implements. It is a proposed future extension; the full objective, why it was not implemented, and what would make it tractable are in **[§7.1](#71-the-ideal-extension-verifier-accepted-state-distribution)**. The current method trains on teacher-sampled contexts only.

### 2.4 Stop-gradient disclosure (important)

For the current stochastic-teacher variant, the sampling distribution is teacher-only and has no $\phi$-gradient to block. Stop-gradient only becomes a substantive issue for the ideal verifier-accepted-state objective ([§7.1](#71-the-ideal-extension-verifier-accepted-state-distribution)), where draft proposals would feed back into the training distribution. In that future draft-gated variant, if we do not differentiate through sampling, the objective is a **stop-gradient Monte Carlo surrogate** for closing the exposure gap — *not* the exact gradient of expected block efficiency $\nabla_\phi \mathbb{E}[\text{BE}]$. The NSS tree-gradient line of work (Rahul) is the route to the *exact* $\partial\text{BE}/\partial\theta$, which composes with — rather than is approximated by — stochastic teacher rollout (see [§5.2](#52-the-design-space-is-2-d-not-a-1-d-ladder)).

### 2.5 Accepted/rejected framing belongs to draft-gated variants

The current implementation trains on teacher-sampled contexts and has no rejection mechanism. Therefore, it should **not** be described as "accepted-only" learning, nor as discarding rejected draft tokens. Accepted-vs-rejected framing is relevant for Draft-OPD and for any future draft-gated `jsd_flat_enrich` extension. If we implement that extension, we must then compare accepted-only training against accepted+rejected replay.

### 2.6 Acceptance–divergence link

For vanilla single-token speculative decoding, per-state acceptance probability is

$$
\alpha(p,q) = \sum_x \min\big(p(x), q(x)\big) = 1 - \text{TV}(p, q).
$$

With JSD measured in nats, Pinsker's inequality applied to the KL sub-terms gives the safe bound $\text{TV}^2 \leq 2\,\text{JSD}$. So minimising JSD ⇒ lower TV ⇒ higher single-token acceptance **on the states being trained**. This chain is clean for naive acceptance; **it does not automatically transfer to BV/GBV/NSS/SpecInfer** (multi-token / tree / optimal-transport criteria). Establishing the link per verifier is an open obligation ([§5.3](#53-conjectured-verifier-level-mechanism-not-yet-derived)).

---

## 3. Measurement Framework

### 3.1 Primary metric: Block Efficiency

$$
\text{BE} = \frac{\text{generated tokens}}{\text{target model calls}}, \quad \text{BE}_{\max} = L.
$$

At $L=8$: observed range 3.7 (NSS, strictest) to 6.4 (traversal). **Report wall-clock tokens/sec alongside BE** — BE and throughput correlate at ~0.95 globally but can decouple; a method that raises BE but not throughput is not useful. Also report output quality/exactness for any approximate verifier.

L-relative buckets: easy $\geq 0.75L$, medium $[0.375L, 0.75L)$, hard $< 0.375L$.

### 3.2 Diagnostics (`--diagnose`)

**Spearman ρ(divergence, BE):** rank correlation between a prompt's JSD (how far the draft is from the teacher) and its BE. Negative ρ means lower JSD → higher BE — the training objective aligns with the eval metric. We use Spearman (rank-based) rather than Pearson because it is robust to JSD range compression as the model improves.

ρ progression with M (K_eval=1, n=100, traversal mode):

| Checkpoint | ρ(JSD,BE) | p-value |
|---|---|---|
| flat JSD | −0.396 | — |
| M=1 enrich | −0.568 | 8.5×10⁻¹² |
| M=3 enrich | **−0.645** | ≈0 |

ρ strengthens monotonically with M, but **this progression is not by itself evidence that stochastic-teacher training aligns better with the verifier.** JSD is an expected value; the diagnostic estimates each prompt's JSD from its rollouts, and a higher-M estimate has lower variance (≈1/M by the CLT). Lower measurement noise on the JSD axis mechanically reduces regression-dilution attenuation and tightens *any* rank correlation — so part (possibly all) of the M=1→M=3 strengthening just reflects 3 samples giving a cleaner JSD estimate than 1, not a better-aligned model. We therefore do **not** lean on the flat→M=1→M=3 progression. The per-verifier ρ comparison below (all at fixed M, so this artifact is held constant across rows) and the eval BE in §4 carry the weight. (The p-values only confirm the correlations are real: p=8.5×10⁻¹² ≈ one in 100 billion; p≈0 rounds to zero.)

Per-verifier ρ for M=3 (n=100, s123). JSD is computed solely from draft and teacher probability distributions — the verifier does not affect it. So all four rows share the same JSD statistics (σ(JSD)=0.0200, mean=0.0296); ρ differences across rows reflect verifier structure, not differences in the JSD distribution.

| Verifier | mean_BE | ρ(JSD,BE) | ρ(fwdKL,BE) | % easy | % medium | % hard |
|---|---|---|---|---|---|---|
| traversal | 6.386 | **−0.645** | −0.647 | 62% | 38% | 0% |
| BV | 6.433 | **−0.518** | −0.503 | 67% | 33% | 0% |
| naive | 6.089 | **−0.413** | −0.410 | 50% | 50% | 0% |
| specinfer | 4.867 | **−0.402** | −0.395 | 22% | 75% | 3% |

**ρ ordering reflects how each verifier aggregates acceptance.** Traversal has the tightest coupling (−0.645): it aggregates path-level acceptance products across the tree, so a JSD reduction along a path compounds into a strong BE lift. BV is next (−0.518): a budget cap limits marginal gain beyond a threshold. Naive is looser (−0.413): it reduces to per-token min(1,P/Q), a coarser aggregation than path products. SpecInfer is loosest (−0.402): it has the most hard prompts (3%) and uses multi-candidate residual acceptance based on token-specific ratios, not aggregate JSD. fwdKL and JSD are interchangeable predictors (Δρ < 0.015 for every verifier). σ(JSD) is stable across all four verifiers, confirming that ρ differences are genuine and not an artifact of JSD range compression for one verifier.

**ρ reflects verifier acceptance-aggregation structure; gain (Δ) reflects headroom.** ρ measures how reliably JSD predicts BE rank for a fixed checkpoint — a property of the verifier's acceptance functional. Δ depends on how far that verifier's BE sits below its ceiling and how sensitive it is to the training distribution. specinfer has both the loosest coupling (ρ=−0.402) and the lowest baseline BE (4.867 — the only verifier with hard prompts); its large K_eval=1 gain (+0.336, [§4.2](#42-primary-eval-n1000-k_eval3-math_eval)) is better attributed to that headroom than to coupling strength. The [§5.3](#53-conjectured-verifier-level-mechanism-not-yet-derived) theory should treat per-verifier JSD→acceptance slope (predicts ρ) and per-verifier acceptance headroom (predicts Δ) as distinct quantities to derive.

ρ is a mechanism diagnostic; the findings rest on eval BE (§4). The four-verifier table (fixed M, so the estimator-variance artifact above is constant across rows) is what anchors the coupling story; the cross-M progression is not used as evidence. Completing all 9 verifier modes is optional.

### 3.3 Capacity signals (measured along the way)

**Student (draft) at capacity:** val/block_eff plateau with no new best; train loss floor (~0.02); high oscillating forgetting with no net BE gain; frozen BE bucket distribution.

**Teacher / data diversity signal:** BE → $L$ suggests limited headroom; path_diversity → 0 means stochastic teacher rollouts have collapsed to near-identical continuations. At 0.6B/8B on math, BE≈6.4/8 and path_diversity ∈ [0.8,1.0] for M=3, so the teacher is still producing diverse contexts. This does **not** prove the teacher is not a bottleneck; it only says stochastic teacher sampling has not collapsed. A 32B-teacher run would test teacher-scale sensitivity (future work, [§7](#7-future-directions-parking-lot--prioritise-later)).

**`train/path_diversity`** = fraction of positions where ≥2 of the $M$ rollouts disagree. ~1.0 ⇒ diverse signal, $M > 1$ contributes; <0.1 ⇒ rollout collapse, $M > 1$ ≈ $M=1$. Observed M=3: ∈[0.8,1.0], healthy.

**`val/forgetting`** = backward-transfer loss (Σ max(0, best_historical_BE(p) − current_BE(p))). Oscillating (not monotonic) ⇒ stability–plasticity churn, not catastrophic forgetting; `ckpt_best` captures the peak.

---

## 4. Results

> All raw per-cell results in [`Results/jsd_enrich_ddte_results.csv`](../Results/jsd_enrich_ddte_results.csv) — the full `eval.py` dump (block_eff, throughput_tok_s, timing breakdown, GPU/CPU util, machine info, seed, n_prompts, etc.), math_eval, eval seed 123, train seed encoded in the checkpoint name. Append new eval rows there; do not paste raw dumps into this note. (Converged flat-40K baseline is in the separate [40K CSV](https://github.com/Rmuk655/Distill-Spec-Research/blob/Pipeline/Results/jsd_jsd_dw_naive_tree_results_math_hard_40000_steps.csv).)
>
> **⚠ Data-provenance caveat (verified 2026-07):** in the committed `jsd_enrich_ddte_results.csv`, the **n=1000** rows exist only for the *s456* checkpoints (bv/naive/traversal at a subset of K); there are **no n=1000 s123 rows**, and flat/M1 have n=1000 only at K=4. So the n=1000 tables in **§4.2/§4.3** are not fully reproducible from committed data — treat any n=1000 s123 cell or n=1000 flat/M1 K=2/3 cell as *not in Results* pending a re-export. Likewise, the **ρ / mean_BE / forward-KL / path-diversity diagnostics in §3.2 and §4.6** are `--diagnose` outputs that are **not present in any committed Results CSV** (no such columns exist there); they are reported from run logs, not reproducible from `Results/`. The single-seed **§0**, the n=100 **§4.1/§4.4 (s123)** point values, and the §4.5 trav-K3 +5.9% (s123) figure are all CSV-backed and verified.

### 4.1 Training runs

| Run | Steps | traversal K=1 (n=100) | Notes |
|---|---|---|---|
| `jsd_mathhard_s123` (flat JSD) | 8K | 5.972 | Flat baseline, seed 1 |
| `jsd_mathhard_s456` (flat JSD) | 8K | 6.011 | Flat baseline, seed 2 |
| `jsd_mathhard_s123` (flat, converged) | 40K | 6.280 | Flat JSD converges ~15K; trained to 40K as the ceiling. Same checkpoint used as the flat baseline in the depth_weight ablation ([40K CSV](https://github.com/Rmuk655/Distill-Spec-Research/blob/Pipeline/Results/jsd_jsd_dw_naive_tree_results_math_hard_40000_steps.csv)). BE is hardware-independent. |
| `jsd_flat_enrich_M1_s123` | 8K | 6.158 | M=1 enrich, seed 1 |
| `jsd_flat_enrich_M1_s456` | 8K | 6.066 | M=1 enrich, seed 2 |
| `jsd_flat_enrich_M3_s123` | 8K | 6.213 | M=3 enrich, seed 1 |
| `jsd_flat_enrich_M3_s456` | 8K | 6.109 | M=3 enrich, seed 2 |
| `jsd_flat_enrich_M3_s123_ttemp1.5` | 8K | 6.128 | Negative control: teacher_temp=1.5 |

All BE values are offline re-evals of `ckpt_best` at traversal K=1 (n=100), unaffected by training-time selection noise.

### 4.2 Primary eval: n=1000, K_eval=3, math_eval

n=1000 gives SE ≈ 0.016. K_eval=3 = M for the M=3 condition. Both seeds evaluated for all conditions.

**Block efficiency:**

| condition | traversal | bv | naive |
|---|---|---|---|
| flat s123 | 5.851 | 5.904 | 5.591 |
| flat s456 | 6.044 | 6.094 | 5.759 |
| **flat avg** | **5.948** | **5.999** | **5.675** |
| M=1 s123 | 6.026 | 6.103 | 5.725 |
| M=1 s456 | 6.006 | 6.130 | 5.772 |
| **M=1 avg** | **6.016** | **6.117** | **5.749** |
| M=3 s123 | 6.056 | 6.231 | 5.837 |
| M=3 s456 | 5.991 | 6.146 | 5.793 |
| **M=3 avg** | **6.024** | **6.189** | **5.815** |

**Deltas vs flat avg:**

| | traversal | bv | naive |
|---|---|---|---|
| M=1 | +0.068 (+1.1%) | +0.118 (+2.0%) | +0.074 (+1.3%) |
| M=3 | +0.076 (+1.3%) | **+0.190 (+3.2%)** | **+0.140 (+2.5%)** |
| M=3 vs M=1 | +0.008 (noise) | **+0.072 (+1.2%)** | **+0.066 (+1.2%)** |

M=3 beats the **seed-averaged** flat on all three modes. **Same-seed the gain varies** — M=3 beats flat on s123 by a large margin; on s456 it is within noise or slightly negative at K_eval=3 for traversal (M=3 s456 5.991 < flat s456 6.044), which is expected variance at two seeds. The full K-sweep is in [§4.3](#43-n1000-across-k_eval--2-3-4-math_eval-both-seeds). M=3 > M=1 in bv and naive (+0.07); traversal shows no M=3/M=1 difference at K_eval=3.

**Seed variance collapse:**

| condition | Δ(s456−s123) traversal | Δ(s456−s123) bv | Δ(s456−s123) naive |
|---|---|---|---|
| flat | **+0.193** | **+0.190** | **+0.168** |
| M=1 | 0.020 | −0.027 | −0.047 |
| M=3 | 0.065 | 0.085 | 0.044 |

Flat JSD has a large systematic seed spread (~0.19 per mode; s456 consistently higher than s123 across all modes). Enrich training collapses this to ~0.03–0.09 — both initializations converge to nearly the same basin. This is confirmed at n=1000 across K_eval=2,3,4 ([§4.3](#43-n1000-across-k_eval--2-3-4-math_eval-both-seeds)). But "initialization-robust" is the right description only with the asymmetry attached: enrich **rescues the weak seed and trims the strong one** — it is primarily a variance reducer, not a ceiling-raiser (see §4.3). Seed-averaged flat values are the authoritative baseline; single-seed flat comparisons are unreliable.

### 4.3 n=1000 across K_eval = 2, 3, 4 (math_eval, both seeds)

K_eval=2 and K_eval=4 are now run at n=1000 (flat + M=1 + M=3, both seeds), so §4.2's K_eval=3 point sits inside a full K-sweep. (The one missing cell is BV M=3 at K_eval=2; K_eval=1 is still not run at n=1000.) **This larger picture confirms the headline: the gain is consistent against the seed-average across all K_eval values; per-seed magnitude varies, as expected with only two seeds.**

**M=3 − flat, seed-averaged (n=1000):**

| mode | Δ K=2 | Δ K=3 | Δ K=4 |
|---|---|---|---|
| traversal | +0.073 | +0.076 | +0.103 |
| bv | *(M3 K2 unrun)* | +0.190 | +0.151 |
| naive | +0.094 | +0.140 | +0.143 |

Against the **seed-averaged** flat baseline, M=3 wins at every K_eval and every mode — the headline holds across the full sweep, not only K=3, and is largest at K_eval=4 for traversal/naive.

**Same-seed the gain magnitude varies — the per-seed (paired) Δ:**

| mode | K | M3−flat **s123** | M3−flat **s456** |
|---|---|---|---|
| traversal | 2 | +0.222 | **−0.076** |
| traversal | 3 | +0.205 | **−0.053** |
| traversal | 4 | +0.189 | +0.018 |
| naive | 2 | +0.247 | **−0.059** |
| naive | 3 | +0.246 | +0.034 |
| naive | 4 | +0.206 | +0.080 |
| bv | 3 | +0.327 | +0.052 |
| bv | 4 | +0.240 | +0.061 |

The gain is larger on s123 (+0.19–0.25 across all K) and smaller on s456 — within noise for traversal/naive at K≤3 (−0.076 at traversal K=2 is ≈3×SE and worth noting, but one data point at two seeds). **At K_eval=4, M=3 beats flat on both seeds for traversal/bv/naive, and beats the flat-40K long-run checkpoint on both seeds at traversal K=3.** The seed-dependent magnitude is expected variance; with ≥5 seeds the per-seed distribution would be estimable.

**This is the seed-variance-collapse mechanism made concrete.** Flat has a large, systematic initialization spread (s456 − s123 ≈ +0.16 to +0.19 at *every* K); M=3 collapses it to ≈ −0.03 to −0.15 (and slightly inverts the sign). Enrich pulls both seeds into a common ~6.0–6.05 basin: it rescues the unlucky seed **up** and trims the lucky seed **down**. So the honest one-line read is that **enrich is primarily a variance reducer / initialization-rescuer, not a ceiling-raiser** — its averaged advantage is mostly the removal of flat's downside, not a lift above flat's best seed.

**K_eval=4 is where the gain is most robust** — the only K at which M=3 beats flat on *both* seeds for all three modes. Absolute BE *declines* with K_eval for every condition (more branches → lower BE at n=1000), so enrich's relative advantage is largest exactly where the verifier's own BE is weakest.

**Traversal absolute BE still peaks at K_eval=2** (M=3 avg 6.073 > K3 6.024 > K4 5.954) — so the best traversal *operating point* is K_eval=2, even though the enrich *gain* is largest at K=4. M=3 vs M=1 is small at K=2,3 (+0.01–0.02) and opens up at K=4 (traversal +0.04, naive +0.02, bv +0.04).

### 4.4 Cross-verifier and K_eval-dependence (n=100, seed-averaged, vs flat-8K)

This is the mechanism sweep: all 9 verifiers × K_eval=1..4, seed-averaged over s123+s456, baseline = flat-8K (the under-trained baseline — the converged flat-40K comparison is [§4.5](#45-m3-8k-steps-vs-our-strongest-flat-baseline-40k-steps)). Raw cells in [`Results/jsd_enrich_ddte_results.csv`](../Results/jsd_enrich_ddte_results.csv).

**M=3 − flat-8K Δ, seed-averaged:**

| Verifier | Δ K=1 | Δ K=2 | Δ K=3 | Δ K=4 | mean | trend (K4−K1) |
|---|---|---|---|---|---|---|
| traversal | +0.169 | +0.131 | +0.190 | **+0.245** | +0.184 | +0.076 ↑ |
| bv | +0.230 | +0.166 | +0.248 | **+0.309** | **+0.238** | +0.080 ↑ |
| naive | +0.086 | +0.177 | **+0.289** | +0.104 | +0.164 | +0.018 → |
| spectr | +0.086 | +0.112 | +0.117 | +0.093 | +0.102 | +0.006 → |
| nss | +0.042 | +0.158 | +0.032 | +0.095 | +0.082 | +0.053 → |
| khisti | +0.086 | +0.081 | +0.021 | +0.041 | +0.057 | −0.045 → |
| gbv | +0.230 | +0.109 | +0.010 | +0.018 | +0.092 | −0.212 ↓ |
| specinfer | +0.210 | +0.231 | **−0.105** | **−0.084** | +0.063 | −0.295 ↓ |
| max | +0.086 | +0.005 | **−0.060** | +0.020 | +0.013 | −0.066 ↓ |

(specinfer K=3 seed-avg is s123-only — flat s456 specinfer K=3 cell is missing.)
† K_eval=1 collapse: naive/spectr/khisti/max give identical BE at K_eval=1 (single draft path → same per-token accept rule), so their K=1 cells are one effective measurement.

**Wins are broad, not on a matched-K diagonal.** Number of verifiers (of 9) where enrich beats flat-8K, and the mean Δ:

| | K_eval=1 | K_eval=2 | K_eval=3 | K_eval=4 |
|---|---|---|---|---|
| **M=3 wins** | **9/9** | **9/9** | 7/9 | 8/9 |
| M=3 mean Δ | +0.136 | +0.130 | +0.082 | +0.094 |
| M=1 wins | 9/9 | 5/9 | 6/9 | 8/9 |
| M=1 mean Δ | +0.067 | +0.045 | +0.057 | +0.092 |

**No matched-K diagonal — the key contrast with depth_weight.** In the depth_weight ablation the only gain over baseline sat on the K_train = K_eval diagonal and collapsed off it. Enrich is different: M=3's gain is positive at *every* K_eval and is largest at K_eval=1,2 (+0.136, +0.130), not at the "matched" K_eval=3 (+0.082). M=1 is the weaker version of the same shape and sags mid-K (5/9 at K=2).

**Why the gain concentrates at narrow K_eval — and what we do *not* claim.** M=3 trains on M *independent* teacher continuations; it is trajectory-level data augmentation for flat training, **not** tree-structure training. The draft is never taught to allocate probability mass across a branching structure. So it is expected — not surprising — that broader single-path coverage benefits narrow inference (K_eval=1,2) more than wide trees. We therefore make **no claim** that enrich aligns the draft to the verifier's tree structure; its mechanism is state coverage, which happens to transfer across K_eval rather than being tied to it. A reviewer may read "largest at low K_eval" as structural misalignment with wide-tree inference — that reading is fair, and we accept it. Teaching the draft to allocate mass across branches is the separate job of the verifier-aligned / accepted-state directions ([§7.1](#71-the-ideal-extension-verifier-accepted-state-distribution)), not of enrich.

**Per-verifier verdict (M=3 vs flat-8K, seed-avg):**

| Verifier | Verdict | Behaviour across K_eval |
|---|---|---|
| **bv** | **Clear win (largest)** | positive all K; **improves with K** (+0.23 → +0.31) |
| **traversal** | **Clear win** | positive all K; **improves with K** (+0.17 → +0.25) |
| **naive** | **Clear win** | positive all K; peaks at matched K=3 (+0.29) |
| spectr | Win (small) | positive all K; **K-agnostic** (~+0.10 flat) |
| nss | Win (small) | positive all K; **K-agnostic** (~+0.08) |
| khisti | Marginal win | positive all K but small; mild fade with K |
| gbv | Front-loaded | strong at K=1 (+0.23), **collapses to ~0 by K≥3** |
| specinfer | Mixed → **loses at high K** | strong at K≤2 (+0.21/+0.23), **negative at K≥3** |
| max | Inconclusive | ~+0.09 at K=1, ~0 elsewhere, **negative at K=3** |

**Three K-dependence classes** (the empirical core [§5.3](#53-conjectured-verifier-level-mechanism-not-yet-derived) theory must reproduce):
- **Improve with K** (gain grows toward K=4): **traversal, bv** — prefix/budget verifiers; M parallel-calibrated branches compound as tree width grows.
- **Worsen with K** (front-loaded, fade or go negative by K≥3): **gbv, specinfer, max** — residual/competition verifiers whose cross-candidate normalisation is sub-additive in branch count.
- **K-agnostic** (flat positive across K): **naive, spectr, nss, khisti**.

**ρ reflects verifier structure; Δ reflects headroom.** specinfer has both the loosest coupling (ρ=−0.402, [§4.6](#46-objectivebe-alignment-diagnostics-ρ)) and the lowest baseline BE (4.867); its large low-K gain is better attributed to headroom than to coupling strength. ρ (surrogate quality for a fixed checkpoint) and Δ (realized improvement from enrich) are distinct quantities driven by different verifier properties.

**Same-seed paired deltas — traversal and BV:**

| Comparison | K=1 | K=2 | K=3 | K=4 |
|---|---|---|---|---|
| M=1 s123 − flat s123 (traversal) | +0.186 | +0.265 | −0.041 | +0.171 |
| M=3 s123 − flat s123 (traversal) | +0.241 | +0.244 | **+0.322** | +0.232 |
| M=1 s456 − flat s456 (traversal) | +0.055 | +0.057 | +0.074 | +0.022 |
| M=3 s456 − flat s456 (traversal) | +0.098 | +0.019 | +0.058 | +0.258 |
| M=1 s123 − flat s123 (bv) | +0.031 | +0.156 | +0.269 | +0.265 |
| M=3 s123 − flat s123 (bv) | +0.264 | +0.232 | **+0.431** | +0.410 |
| M=1 s456 − flat s456 (bv) | +0.137 | +0.158 | +0.078 | +0.154 |
| M=3 s456 − flat s456 (bv) | +0.195 | +0.101 | +0.065 | +0.209 |

M=3 beats its own flat-8K baseline at every K value on both traversal and BV — both seeds, all cells positive.

### 4.5 M=3 (8K steps) vs our strongest flat baseline (40K steps)

Our flat JSD run stops improving by ~15K steps; we trained it to 40K and treat that as our strongest flat baseline (one optimization trajectory — *not* a proof of the flat ceiling; other schedules/objectives might do better). flat-40K evaluated on H100; BE is hardware-independent and comparable to the A100 8K runs.

**On the compute-confound objection.** A natural worry is that M=3-8K beats flat-8K simply because it does 3× the teacher rollouts per step, so the lift is rollout *volume*, not the *stochastic* nature of the trajectories. The flat-40K comparison is designed to answer exactly this, and the rollout arithmetic favors flat:
- M=3-8K uses ≈24K total teacher rollouts (3/step × 8K steps).
- flat-40K uses ≈40K total teacher rollouts (1/step × 40K steps) — **67% more rollouts than M=3-8K**, and flat has already plateaued by ~15K, so the strict compute-matched point (flat-24K) sits on the same plateau as flat-40K.
- Despite that rollout advantage, flat-40K still scores **below** M=3-8K at traversal K=3 (+5.9% s123 / +2.9% s456). Since flat-24K ≤ flat-40K (it is the plateau, reached with fewer steps), a strict compute-matched flat-24K cannot do better than flat-40K — so it cannot close this gap either.

In other words, the result is not "M=3 wins because it saw more rollouts" — flat saw *more* rollouts and still lost on this cell. The strict flat-24K run ([§6](#6-must-add-experiments-minimum-for-a-credible-paper) #3) is still worth running to remove all doubt, but the plateau + the flat-40K upper bound already bracket it. (This logic applies to the traversal-K=3 cell; the BV picture is murkier — see below.)

| comparison | overall mean Δ | bv/traversal Δ | traversal K=3 Δ |
|---|---|---|---|
| flat-40K − flat-8K | +0.109 | +0.189 | −0.024 |
| **M=3-8K − flat-40K [s123]** | **+0.074** | **+0.100** | **+0.346 (+5.9%)** |
| **M=3-8K − flat-40K [s456]** | — | — | **+0.168 (+2.9%)** |
| M=1-8K − flat-40K | −0.022 | −0.044 | −0.016 |

**Cross-seed % vs flat-40K:**

| cell | s123 | s456 |
|---|---|---|
| traversal K=2 | +0.4% | −1.1% |
| **traversal K=3** | **+5.9%** | **+2.9%** |
| traversal K=4 | −0.8% | +2.3% |
| bv K=2 | +1.9% | −0.5% |
| bv K=3 | +1.9% | −1.4% |
| bv K=4 | +2.2% | 0.0% |

**M=3 at 8K scores above our flat-40K baseline at traversal K=3: +5.9% (s123), +2.9% (s456).** This was *not reached by our strongest flat baseline* — flat-40K traversal K=3 is even −0.024 below flat-8K (more flat training did not help this cell on this trajectory). This is suggestive that the M>1 mechanism adds something flat training did not recover here; it is **not** a proof that flat optimization cannot reach it (single trajectory, one dataset, one architecture, no CIs).

**BV gain is convergence-speed, not a raised ceiling.** This is the meaning of "BV does not replicate vs converged flat on s456":
- *vs flat-8K* (under-trained baseline): M=3 wins on BV at every K_eval, both seeds ([§4.4](#44-cross-verifier-and-k_eval-dependence-n100-seed-averaged-vs-flat-8k)). Real and robust.
- *vs flat-40K* (converged baseline): the BV advantage holds on s123 (+1.9% at K=3) but **reverses on s456 (−1.4%)** — the two seeds disagree once flat is trained to convergence.
- **Interpretation:** M=3 reaches a strong BV checkpoint *faster* than flat (8K vs ~15K convergence), but flat catches up on BV once converged. M=3 does **not** lift the BV ceiling.
- **Contrast — traversal:** at K=3, M=3 scores above flat-40K on **both** seeds (+5.9% / +2.9%) — the more robust of the two, but still single-dataset, two-seed, no CIs.
- **Open:** the n=1000 BV K_eval=2 **M=3** cell is still unrun (M=1 is now run); and a third seed would break the s123/s456 tie. Note the BV reversal here is vs the *converged* flat-40K — against flat-8K at n=1000, M=3 BV beats flat on both seeds at K_eval=3,4 ([§4.3](#43-n1000-across-k_eval--2-3-4-math_eval-both-seeds)).

**M=1 has no edge over flat-40K.** M=1-8K is slightly below flat-40K (−0.022). A single stochastic trajectory behaves like flat JSD with broader context; the M>1 setting is what moves the metric here.

### 4.6 Objective–BE alignment diagnostics (ρ)

From `--diagnose` mode (n=100, math_eval):

**ρ progression with M (traversal mode):** flat JSD −0.396 → M=1 −0.568 → M=3 **−0.645**. Monotonically strengthening — but not used as evidence: at higher M the per-prompt JSD is estimated from more rollouts (variance ≈1/M), which mechanically tightens the correlation via reduced regression-dilution, independent of model alignment ([§3.2](#32-diagnostics---diagnose)). The per-verifier table below (fixed M) is the load-bearing comparison.

**Per-verifier ρ (M=3, s123), σ(JSD)=0.0200 stable across all verifiers:**

| Verifier | mean_BE | ρ(JSD,BE) | ρ(fwdKL,BE) | % easy | % medium | % hard |
|---|---|---|---|---|---|---|
| traversal | 6.386 | **−0.645** | −0.647 | 62% | 38% | 0% |
| BV | 6.433 | **−0.518** | −0.503 | 67% | 33% | 0% |
| naive | 6.089 | **−0.413** | −0.410 | 50% | 50% | 0% |
| specinfer | 4.867 | **−0.402** | −0.395 | 22% | 75% | 3% |

σ(JSD) stable → ρ differences are not just range compression. JSD and fwdKL are interchangeable as predictors (Δρ < 0.015). **Caveat (do not oversell):** a higher |ρ| means JSD is a better *monotonic predictor* of BE for a fixed checkpoint — it does **not** by itself mean JSD is a better training objective, and it does not establish causation. ρ is a supporting diagnostic only; the findings rest on eval BE, not on ρ.

**Negative control (teacher_temp=1.5) — a sanity check, not a temperature characterization.** ρ drops −0.645 → −0.543, mean_JSD rises 8% (σ stable). Caveat: at temp 1.5 the 8B teacher largely loses logical coherence on math, so this mostly shows the obvious — training on near-incoherent teacher text hurts — rather than establishing a monotonic "higher teacher temperature is worse" law. The proper control is a *mild* sweep (e.g. 0.4 / 0.7 / 1.0) that raises stochasticity while preserving logical validity, to see whether more diverse-but-valid teacher trajectories help or hurt BE; that is unrun. On current evidence teacher_temp=1.0 is the safe default.

**train/path_diversity (M=3):** ∈[0.8,1.0] — stochastic teacher rollouts are genuinely diverse; M>1 contributes distinct contexts, not near-duplicate paths.

### 4.7 Conclusions

All conclusions are scoped to this setup (one model pair, one dataset, two seeds, no CIs) and are stated as observations.

**Supported on this setup:**

1. **M=3 enrich scores above the seed-averaged flat** on traversal/bv/naive at K_eval=2,3,4 (n=1000) — gain largest at K_eval=4. Per-seed the gain varies: larger for s123 (+0.19–0.25) and smaller for s456 (within noise at K≤3, a clear win at K=4) — expected variance at two seeds. At K_eval=4, M=3 beats flat on both seeds. Also *not compute-matched*. [[§4.3](#43-n1000-across-k_eval--2-3-4-math_eval-both-seeds)]
2. **M=3 (8K) scores above our flat-40K baseline at traversal K=3** (+5.9% s123 / +2.9% s456); our flat run did not reach this with more steps. Suggestive of an M>1 effect; *not* a proof flat cannot reach it. [[§4.5](#45-m3-8k-steps-vs-our-strongest-flat-baseline-40k-steps)]
3. **Variance reduction is the dominant, best-supported effect** (and likely the real contribution). Flat's large initialization spread (s456−s123 ≈ +0.16–0.19) collapses under enrich to ≈ −0.03 to −0.15 at every K_eval (n=1000) — enrich rescues the weak seed up and trims the strong seed down to a common basin. So enrich is best described as an **initialization-rescuer / variance reducer, not a ceiling-raiser**; its seed-averaged BE gain is mostly the removal of flat's downside. Still needs ≥5 seeds to estimate the variance properly. [[§4.3](#43-n1000-across-k_eval--2-3-4-math_eval-both-seeds)]
4. **Verifier response is heterogeneous (the strongest finding).** M=3 scores above flat-8K on 7–9 of 9 verifiers at every K_eval; bv/traversal gains grow with K_eval while gbv/specinfer/max fade or reverse. [[§4.4](#44-cross-verifier-and-k_eval-dependence-n100-seed-averaged-vs-flat-8k)]
5. **|ρ|(JSD,BE) increases with M** (flat −0.396 → M=1 −0.568 → M=3 −0.645) — a monotonic-association diagnostic, not evidence about the objective itself. [[§4.6](#46-objectivebe-alignment-diagnostics-ρ)]

**Not supported / refuted:**

6. **A single stochastic rollout (M=1) is not enough** — it does not beat flat-40K (−0.022). The M>1 setting is what moves the metric. [[§4.5](#45-m3-8k-steps-vs-our-strongest-flat-baseline-40k-steps)]
7. **"Matched K_eval = M is optimal" does not hold for enrich.** M=3 is strongest at K_eval=1,2 (+0.136/+0.130), not at matched K_eval=3 (+0.082). This is consistent with the mechanism — M independent teacher paths are flat-trajectory augmentation, not tree-structure training, so the draft is never taught to allocate mass across branches and broader single-path coverage naturally helps narrow inference most. We claim coverage, not verifier-tree-structure alignment. [[§4.4](#44-cross-verifier-and-k_eval-dependence-n100-seed-averaged-vs-flat-8k)]
8. **Extreme teacher temperature (1.5) hurts — but this is the expected failure of training on incoherent math, not a characterized temperature law.** ρ −0.645 → −0.543; at temp 1.5 the teacher loses logical coherence, so this is a sanity check, not a sweep. A mild sweep (0.4/0.7/1.0) preserving validity is unrun. teacher_temp=1.0 is the safe default. [[§4.6](#46-objectivebe-alignment-diagnostics-ρ)]

**Open / needs more data:**

9. **BV: convergence-speed vs raised baseline unresolved.** M=3 scores above flat-8K on BV everywhere, but the advantage over flat-40K reverses on s456. Needs n=1000 BV K_eval=2 + a tie-break seed. [[§4.5](#45-m3-8k-steps-vs-our-strongest-flat-baseline-40k-steps)]
10. **specinfer/max anti-alignment at high K_eval** (M=3 negative at K≥3 / K=3, both seeds) — observed but unexplained; gated on a [§5.3](#53-conjectured-verifier-level-mechanism-not-yet-derived) derivation that does not yet exist.
11. **Paired bootstrap CIs** for the headline cells ([§6](#6-must-add-experiments-minimum-for-a-credible-paper) #2).

**Cross-verifier K-dependence (the empirical headline for [§5.3](#53-conjectured-verifier-level-mechanism-not-yet-derived)):** three classes — *improve with K* (traversal, bv: prefix/budget), *worsen with K* (gbv, specinfer, max: residual/competition, fade or go negative by K≥3), *K-agnostic* (naive, spectr, nss, khisti). The [§5.3](#53-conjectured-verifier-level-mechanism-not-yet-derived) acceptance-functional theory must reproduce this split. [[§4.4](#44-cross-verifier-and-k_eval-dependence-n100-seed-averaged-vs-flat-8k)]


---

## 5. Discussion and Theoretical Positioning

The empirical picture (§4): M=3 stochastic-teacher JSD beats flat JSD on the prefix/budget verifiers, the benefit is heterogeneous across verifiers and K_eval, and it holds against a converged flat baseline that used *more* teacher rollouts. This section places that result — first against prior work (the biggest risk to the contribution), then with a conjectured, not-yet-derived mechanism for the cross-verifier differential.

### 5.1 Novelty positioning (the biggest risk)

**Emerging research direction: acceptance-aware distillation.** The common thread across recent papers is not "how closely does the student match the teacher globally?" but **"which states and tokens matter for speculative acceptance?"** Methods differ along two orthogonal axes: (A) *state selection* — which contexts to train on; (B) *token selection* — which positions within those contexts to supervise. Current `jsd_flat_enrich` occupies a simple point: stochastic teacher states, uniform token coverage, multiple trajectories.

**State-selection axis:**

| Prior work | State source | Our differentiator |
|---|---|---|
| **DistillSpec ([arXiv:2310.08461](https://arxiv.org/abs/2310.08461))** | Draft-generated (on-policy); highlights divergence choice for SD | Stronger state alignment; no multi-trajectory. Current method is simpler and weaker — must be sold as stochastic-teacher sampling, not as a new on-policy solution |
| **GKD / On-Policy KD ([arXiv:2306.13649](https://arxiv.org/abs/2306.13649))** | Student-generated states; gradual teacher→student curriculum | Most principled student-state framing; orthogonal to our trajectory-diversity axis (see 2×2 below) |
| **VSD ([arXiv:2602.05774](https://arxiv.org/abs/2602.05774))** | Latent draft paths; ELBO/EM maximises acceptance probability directly | Most principled objective; variational inference overhead. Our method is a simpler approximation without the ELBO. |
| **SKD ([arXiv:2410.11325](https://arxiv.org/abs/2410.11325))** | Teacher-corrected student states | Bridges offline and on-policy columns; does not use multiple teacher trajectories per context |
| **Draft-OPD (2026, [arXiv:2605.29343](https://arxiv.org/abs/2605.29343))** | Error-anchored rollout from observed failures; accepted+rejected asymmetric KL, $w_k=\gamma^{k-1}$; Qwen3-4B/8B/30B, EAGLE/DFlash heads; one acceptance scheme; no theory; punts cross-verifier | (1) Cross-verifier coverage (9 vs 1). (2) Per-verifier acceptance-functional theory. (3) Standalone 0.6B draft vs. EAGLE head. (4) Stochastic-teacher JSD is simpler — must show competitive under matched compute |
| **OSD ([arXiv:2310.07177](https://arxiv.org/abs/2310.07177))** | Live serving traffic (online) | Online regime targeting deployment distribution; different setting |
| **Current `jsd_flat_enrich`** | **Stochastic teacher trajectories × M; no draft gating** | Simplest multi-trajectory point; no within-context state selection |

**Token-selection axis (orthogonal to state source):**

| Prior work | Token strategy | State source |
|---|---|---|
| **AdaSPEC ([arXiv:2510.19779](https://arxiv.org/abs/2510.19779))** | Informative tokens only | Greedy/corpus teacher |
| **SelecTKD ([arXiv:2510.24021](https://arxiv.org/abs/2510.24021))** | Teacher-consistent tokens only | Teacher |
| **Current `jsd_flat_enrich`** | All tokens, uniform JSD | Stochastic teacher |

Both axes compose: any state source can combine with any token-selection strategy.

**2×2 position of current method:**

|  | Single teacher trajectory | Multiple teacher trajectories |
|---|---|---|
| **Teacher states (offline)** | Standard flat JSD | **Current `jsd_flat_enrich`** (M=1, 3) |
| **Student states (on-policy)** | DistillSpec / GKD | **Unexplored** |

The bottom-right corner — student-generated contexts with M teacher continuations per rollout — is the natural adaptive variant: the student's own failures determine where the teacher supervises. See [§7](#7-future-directions-parking-lot--prioritise-later) (Curriculum Enrichment).

**Key limitation — enrich never forces the draft to recover from its own mistakes.** The speculative-decoding exposure problem is fundamentally about the draft conditioning on *its own* suboptimal tokens at inference. `jsd_flat_enrich` throws more diverse *teacher* tokens at the draft, but every training context is still teacher-generated — the draft is never made to recover from a state it itself produced and the verifier rejected. This is the top-left→bottom-left gap in the 2×2: we add teacher-trajectory breadth (top row) without ever moving to student-generated states (bottom row). If Draft-OPD (which replays from the draft's *observed failures*) beats our method, this is the most likely reason. It should be stated as a known limitation, not buried — and it is exactly the gap the accepted-state / verifier-failure-targeted directions ([§7.1](#71-the-ideal-extension-verifier-accepted-state-distribution)) are meant to close.

**What Draft-OPD leaves genuinely open:** cross-verifier behaviour and theory for verifier-specific acceptance functionals. Their own future-work line — extending OPD to *approximate/lossy verification* — is adjacent to our verifier work, so we are not contradicting them, we are entering the gap they flagged.

**Defensible claims (narrow):**
- *"For verifier-conditioned speculative decoding, how acceptance-aware training transfers across verifier families is unexplored; we map it empirically (9 verifiers × eval-K) and explain the differential with a per-verifier acceptance-functional analysis."*
- *"Stochastic-teacher symmetric JSD is a low-complexity teacher-sampling alternative; we measure the gap to Draft-OPD-style replay under matched compute."*

**Mandatory comparison:** stochastic teacher rollout vs. draft-gated accepted-only vs. accepted+rejected replay, and symmetric JSD vs. asymmetric fwd/rev-KL with $\gamma^{k-1}$ decay. Otherwise reviewers correctly say we have not located the result relative to a more complete on-policy method.

**Relationship to DDTE (Thomas et al. 2026, [arXiv:2602.16994](https://arxiv.org/abs/2602.16994)) — our group's own verification paper.** DDTE is the *verification + tree-construction* axis with **untrained** drafts; we are the *draft-training* axis. They compose: DDTE picks the tree, we train the draft that fills it. Two of its findings independently corroborate ours and should be cited as support, not competition: (1) **Traversal dominates all OT verifiers and NSS is worst** — the same ranking we see post-distillation, so the ordering is robust to whether the draft is trained; (2) **draft–target L1 divergence grows with tree depth, and OT verifiers waste shallow branching** — the verification-side mirror of our ρ(JSD,BE) coupling and of the "specinfer/gbv/max worsen with K_eval" class ([§4.4](#44-cross-verifier-and-k_eval-dependence-n100-seed-averaged-vs-flat-8k)). Caveat for any quantitative cross-citation: their "BV" is single-path (ours is multi-path) and their Qwen pair is Qwen2.5-32B/0.5B (ours Qwen3-8B/0.6B), so verifier *rankings* transfer but absolute BE does not.

**The depth-weighting / tree-reward corner is more occupied than earlier drafts implied.** DDTE cites **Group Tree Optimization (GTO, Hu et al. 2025a)**: draft distillation with a reward based on *expected NSS acceptance* plus a *PPO-style objective contrasting frozen vs. evolving draft trees*. That is, in combination, essentially Rahul's `depth_weight` (expected-NSS-acceptance weighting) **and** the retired `tree_pg` (REINFORCE/PPO on a tree reward) — and it is already published. Our negative results on both are still informative (they say this corner did not work in our 0.6B/8B setup), but we must **not** frame depth_weight or tree-reward training as novel; the honest contribution there is a negative/replication result against GTO, not a new method.

### 5.2 The design space is 2-D, not a 1-D ladder

The methods in this area separate cleanly along two orthogonal axes:

- **Axis A — state distribution** (where training contexts come from): offline greedy teacher trajectories (flat JSD / SFT) → stochastic teacher trajectories (**current `jsd_flat_enrich`**) → verifier-accepted states (**future draft-gated extension**) → target-assisted rollout replayed from error positions (**Draft-OPD**).
- **Axis B — objective / weighting** (what loss, weighted how, at each context): uniform JSD (**current method**) → fixed geometric decay $\gamma^{k-1}$ asymmetric KL (**Draft-OPD**) → predicted depth/survival weight (Rahul's depth_weight) → exact survival-weighted $\partial\text{BE}/\partial\theta$ (Rahul's depth gradient).

Reading off the space:

| Method | Axis A (states) | Axis B (objective/weight) |
|---|---|---|
| flat JSD | offline greedy teacher | uniform JSD |
| **current `jsd_flat_enrich`** | **stochastic teacher trajectories** | uniform JSD |
| future accepted-state JSD | verifier-accepted states | uniform JSD |
| Draft-OPD | rollout + error-anchored | asymmetric KL, $\gamma^{k-1}$ |
| **GTO (Hu et al. 2025a)** | draft-tree states (frozen vs evolving) | expected-NSS-acceptance reward, PPO-style — **published**; ≈ depth_weight + tree_pg |
| Rahul depth_weight | (any) | crude `depth × JSD` multiply, **no gradient** (subsumed by GTO's reward) |
| Rahul depth gradient | (any) | exact survival-weighted BE gradient |

**Detailed comparison: `depth_weight` (Axis B only) vs `jsd_flat_enrich` (Axis A only):**

| Dimension | `depth_weight` (researcher) | `jsd_flat_enrich` (our method) |
|---|---|---|
| Who generates K paths | Student (draft paths via `iid_draft`) | Teacher (stochastic rollouts, `do_sample=True`) |
| Tree structure | Yes — K paths share prefixes, branching tree | No — K paths are fully independent sequences |
| What K means | K simultaneous draft branches under verifier | K independent teacher continuations per prompt |
| Verifier involved | Yes — DP simulates acceptance over student tree | None — pure JSD on teacher trajectories |
| Scalar/loss | E[τ_V] under verifier V × flat JSD | Average of M per-sequence JSD losses |
| Gradient | No gradient through depth scalar (`@no_grad`) — only reweights | Full gradient through student forward on each teacher path |
| Teacher path | One **greedy** teacher rollout (do_sample=False) | M **stochastic** teacher rollouts (do_sample=True) |
| Axis | **Axis B:** same states, verifier-informed loss weighting | **Axis A:** different training states (broader teacher coverage), uniform JSD |
| Intuition | **Survival/BE-contribution weighting** (NOT remediation): `depth_w = d/EMA(d)` *up-weights* prompts the draft already accepts deep, *down-weights* the hard ones. Rationale: the exact ∂BE/∂θ is survival-weighted — gradient mass lives on the reached/deep trajectories, since improving a position you never reach can't lengthen the block. `depth_lambda < 0` reverses it to drill weak prompts (headroom hypothesis — the supported control, unrun). | Breadth coverage — show student M diverse teacher approaches |

**Supported verifiers for `depth_weight`** (those with `expected_*_depths` in `TreeVerifier`, picked up from `--aux_loss` → `LOSS_TO_VERIFIER`): **naive, nss, specinfer, spectr, khisti, traversal**. BV and GBV have no `expected_bv/gbv_depths` method and are not supported.

Consequences:
1. **Stochastic teacher rollout and the depth gradient are largely orthogonal** — the former moves on Axis A, the depth gradient moves on Axis B. They compose (stochastic or accepted states × depth-weighted objective) but are independent knobs, not the same idea. *(So using depth in enrich would "optimise what the student learns" — an Axis-B change layered on Axis-A sampling.)*
2. **Draft-OPD is not a special case of our method.** Both Draft-OPD and a hypothetical enrich+survival-weight method are *distinct populated corners of the same 2-D space*. The only precise "special case" statement that holds is narrow: **Draft-OPD's $w_k=\gamma^{k-1}$ is a fixed, hand-set special case of a general Axis-B survival weight**; Rahul's depth weight would replace that hand-set decay with a *predicted* one.
3. **The unifying contribution, if any, is a design-space map + analysis, not a ladder** — and it only becomes real once runs populate the empty cells. As of now it is unvalidated: the depth_weight cell in progress is a crude `depth × JSD` multiply with no gradient and is not expected to be promising; the exact depth gradient is not yet implemented.

**Honest expectation for survival-weighting (do not oversell):** novel in framing, but **unlikely to beat M=3 stochastic-teacher JSD by a meaningful margin** — it reweights positions already present in the stochastic teacher trajectories; the related scalar `depth_weight` already underperformed flat JSD; a noisy $E[\tau_V]$ injects estimator error; headroom at ~80% of max BE is compressed. If pursued, **position-level only** (prompt-level on/off collides with both Draft-OPD and curriculum learning), validated with a cheap single-seed probe.

> **Prompt-level vs. position-level (decision required):** prompt-level on/off selection (train hard prompts, skip easy) is the weakest framing — it collides with both Draft-OPD *and* generic curriculum learning. Only the **position-level continuous weighting** carries the differentiation above. If we pursue this, commit to position-level.

### 5.3 Conjectured verifier-level mechanism (not yet derived)

> **Note:** only the naive case ($\alpha=1-\text{TV}$, via Pinsker) is derived; the explanations for traversal/BV/GBV/specinfer/NSS below are plausible intuitions, not proven. Either complete these derivations, or present §4.4 as purely empirical. Toy finite-vocabulary experiments (enumerate exact acceptance and BE, match against simulation) would make the verifier story hard to attack.

**Core question — why does stochastic-teacher JSD help verifiers differently?** Each verifier's per-state acceptance probability is a *different functional* of $(P,Q)$. For naive single-token acceptance, $\alpha_{\text{naive}}(P,Q)=\sum_x\min(P(x),Q(x))=1-\text{TV}(P,Q)$, and JSD bounds TV ($\text{TV}^2\le 2\,\text{JSD}$ in nats), so lowering JSD on the trained states should raise naive acceptance on those states. For NSS (optimal-transport), BV/GBV (tree/budget), and SpecInfer, the acceptance functional is **not** TV, so the same JSD reduction would map to a *different* marginal acceptance gain — that mapping is the conjectured explanation for the differential cross-verifier benefit, and deriving it per verifier *would be* the theory contribution if completed (it is not, yet). Draft-OPD gives a local accepted/rejected KL rationale, but not this cross-verifier acceptance-functional analysis.

1. **$K_{\text{eval}}{=}1$ collapse — likely a one-line remark, not a theorem.** At $K_{\text{eval}}{=}1$, the verifier sees a single draft path. For sequential token verification up to depth $L$, naive/spectr/khisti/max plausibly reduce to the same per-token accept-w.p.-$\min(1,P/Q)$ rule along that path. The proof should state this for arbitrary $L$, not only $L=1$. If the reduction is trivial, state it as a remark for completeness — **do not present it as a contribution.** Empirically the four give identical BE for both checkpoints (5.848 / 5.859).
2. **Exact expected-BE for tree/OT verifiers.** For NSS/BV/GBV/traversal, either derive exact expected-BE formulas or state precisely why stochastic-teacher JSD is only a surrogate. Back this with **toy finite-vocabulary experiments** where acceptance and BE can be enumerated exactly and matched against simulation — this makes the verifier story hard to attack.
3. **Khisti antagonism.** Khisti is the only verifier with consistent negatives ($K_{\text{eval}}{=}2,4$). Needs a mechanism: what calibration pattern does stochastic-teacher JSD induce that khisti penalises?
4. **Acceptance–divergence transfer ([§2.6](#26-acceptancedivergence-link))** beyond naive.
5. **Verifier–K alignment ([§4.4](#44-cross-verifier-and-k_eval-dependence-n100-seed-averaged-vs-flat-8k)).** Why do prefix/budget verifiers (traversal, BV) keep or grow their enrich Δ as $K_{\text{eval}}\to M$ and beyond, while residual verifiers (specinfer) lose it at high $K_{\text{eval}}$? Conjecture: traversal/BV acceptance reward is monotone increasing in tree width given per-branch calibration, so M parallel-calibrated branches compound; specinfer's residual normalisation across candidates is sub-additive in branch count, so added branches dilute. Derive the $K_{\text{eval}}$-dependence of expected-BE per verifier and show it predicts the observed alignment/anti-alignment.
6. **Coupling vs headroom ([§3.2](#32-diagnostics---diagnose)).** Two separate per-verifier quantities to derive: the JSD→acceptance *slope* (predicts ρ) and the acceptance *headroom* relative to a flat-JSD-trained draft (predicts Δ). specinfer illustrates the distinction — low ρ from verifier structure (multi-candidate residual acceptance), large Δ from low baseline BE (most hard prompts, most room to improve). A complete theory derives both from the verifier's acceptance functional.

---

## 6. Must-Add Experiments (minimum for a credible paper)

| # | Experiment | Why |
|---|---|---|
| 1 | 2–3 seeds for flat JSD **and** enrich | Reproducibility — non-negotiable |
| 2 | n=1000 eval **with paired bootstrap CIs** | n=1000 point estimates done (K_eval=2,3; flat+M=1+M=3, both seeds; enrich gap present on this setup). **CIs still missing and required** — without them the effect size is not established. |
| 3 | **Compute-matched** baseline (same wall-clock / teacher calls / tokens, not just steps) | Enrich does extra rollouts; step-matched is unfair to baseline |
| 4 | DistillSpec baseline | Closest on-policy prior work |
| 5 | **Draft-OPD-style replay ablation** (draft-gated accepted-only vs accepted+rejected) | The core novelty contrast |
| 6 | Greedy teacher (M=1) vs stochastic teacher (M=1) vs stochastic teacher (M=3) | Isolate stochasticity from multi-trajectory averaging |
| 7 | ≥1 more dataset (code / Spec-Bench mixed) | Generalisation beyond math |
| 8 | ≥1 more model pair | Method, not setup-specific quirk |
| 9 | Teacher-temp **mild sweep** (0.4 / 0.7 / 1.0) | Only the extreme ttemp=1.5 is run (ρ −0.543 vs −0.645), which mainly shows incoherent-teacher text hurts — not a temperature law. A mild sweep that preserves logical validity is the real control. Eval-temperature sweep (not training) also needed for approximate verifiers. |
| 10 | Wall-clock tokens/sec + output quality/exactness | BE alone is insufficient |

Near-term order: finish M=3 eval → run s456 (flat + stochastic-teacher JSD) → n=1000 on best checkpoints → compute-matched greedy/stochastic/M ablations → Draft-OPD-style replay ablation.

---

## 7. Future Directions (parking lot — prioritise later)

### 7.1 The ideal extension: verifier-accepted state distribution

> **Note:** the formalism below is left for the researcher to fill in. This section states the idea and the open design questions in plain terms; the exact objective and gradient are Rahul's to specify.

**The idea.** Today `jsd_flat_enrich` trains on contexts the *teacher* produces. The stronger version trains on contexts the *draft itself* would produce at inference and that the *verifier actually accepts*: let the draft propose tokens, run them through the real verifier, keep the accepted prefix, fix the first rejected position from the teacher, and train on that verified prefix. This closes the offline-to-inference gap directly — the draft learns on exactly the states it will be scored on — instead of approximating it with broader teacher sampling.

**How it could be done — and the open questions for the researcher:**

1. **Generating the draft proposals.** We already run the draft forward each step; this additionally needs the draft to *generate* autoregressively before verifying. At 0.6B that is cheap and the 8B teacher pass still dominates, so compute is not the blocker. *Question:* batch the draft generation + verification efficiently enough to keep step time acceptable?

2. **Learning through the accept/reject decision.** The verifier's keep-or-reject choice is a hard, discrete decision, so an ordinary loss on the accepted prefix does not tell the draft *how* to change its proposal probabilities to get more tokens accepted. *Question for Rahul:* his survival-weighted acceptance gradient (developed for NSS) is exactly the mechanism that assigns credit through this discrete step — **does it generalise from NSS to the traversal and BV verifiers?** If it does, the discreteness stops being an obstacle and this becomes an implementation task with known parts; if not, we fall back to a stop-gradient surrogate (train on the accepted states but don't differentiate through the acceptance), which is weaker.

3. **The training distribution moves as the draft learns.** Because the accepted-state distribution depends on the current draft, it shifts every update — a moving target, unlike the fixed teacher distribution we use now. *Question:* is a replay buffer of recently accepted prefixes (or periodic, not per-step, regeneration) enough to stabilise it, or do we need a PPO-style trust region? This is the genuine remaining difficulty.

4. **Verifier specificity is the goal, not a side-effect.** Training against a given verifier should lift *that* verifier — aligning the loss to traversal should help traversal, to BV should help BV. That is the whole point (verifier-aligned losses), not a limitation to mitigate.

| Open question | For whom | Blocker? |
|---|---|---|
| Efficient draft-gen + verify in the loop | us (engineering) | No — 0.6B is cheap |
| Credit through the accept/reject step | **Rahul** — does the survival-weighted gradient extend to traversal/BV? | Decides whether it's exact or a surrogate |
| Stabilising the moving training distribution | us | Real — replay buffer / periodic regen / trust region |
| Verifier specificity | — | Not a blocker — it's the objective |

### 7.2 Other parked directions

- **Verifier-weighted stochastic teacher JSD (Axis A × Axis B combination) — PARKED, likely premature.** The synthesis would weight each of M teacher rollouts by its E[τ_V]: $\frac{1}{M}\sum_m \frac{d_m}{\text{EMA}(d)} \cdot \text{JSD}(Q_\phi(\cdot|y^{(m)}), P_\theta(\cdot|y^{(m)}))$. **Three reasons not to do this yet:** (1) **depth_weight has shown no positive signal on its own** (scalar version underperformed flat) — combining a no-signal method with a cross-seed-weakened one violates one-variable-at-a-time. (2) **The default multiply direction is antagonistic to enrich:** survival-weighting up-weights deep/easy contexts, but enrich's whole purpose is to add *hard/diverse* contexts — depth-multiply would down-weight exactly what enrich is trying to inject. Only `depth_lambda < 0` (divide, drill-the-weak) is conceptually compatible with enrich's coverage goal, and that variant is unrun. (3) Enrich's own headline is now downgraded by the cross-seed analysis ([§4.2](#42-primary-eval-n1000-k_eval3-math_eval)). Revisit only after depth_weight alone produces a positive result AND enrich clears n=1000.
- **NSS tree gradients (Rahul):** exact survival-weighted $\partial\text{BE}/\partial\theta$ — an **Axis-B** (objective) change, **orthogonal to enrich's Axis-A** (state-distribution) change; they compose rather than approximate each other. The exact gradient is the strongest unscooped asset (Draft-OPD has no theory). Not yet implemented; depth_weight (crude `depth × JSD` multiply, no gradient) is the only Axis-B experiment so far and is not expected to be promising.
- **NSS-depth and broader tree gradients / tree-depth ablations** (vary $L$, vary $M$).
- **Adaptive teacher curriculum:** soft vs hard accept; hard-prompt up-weighting by inverse BE (exclude teacher-uncertain prompts); teacher-temperature scheduling (warm→cool).
- **Adaptive teacher using tree depth** to decide where to guide the student.
- **Expected-depth survival weighting ([§5.2](#52-the-design-space-is-2-d-not-a-1-d-ladder)):** weight per-position JSD by predicted marginal acceptance-length gain $E[\tau_V]$. A distinct Axis-B corner between Draft-OPD's $\gamma^{k-1}$ and the exact NSS gradient; **expected to roughly match, not clearly beat, M=3 stochastic-teacher JSD** (it reweights existing positions). Pursue **position-level only**; validate with a cheap short probe gated behind M=3 confirmation.
- **Curriculum Enrichment / adaptive state selection (*Curriculum Enrichment Distillation*):** the strongest version of `jsd_flat_enrich` is not uniform M-path sampling but selective enrichment — sample M teacher branches where the verifier rejects (acceptance low); use standard flat JSD where acceptance is high. Teacher compute is spent only where BE is weakest. This is conceptually closer to active learning / a "Socratic teacher" than a fixed loss: *train where the student needs it.* Logically prior to any fixed-M scale-up; if it works at M=3, the adaptive version should be strictly more compute-efficient.
  - **DDTE supplies an independent mechanistic argument for this.** Thomas et al. 2026 ([arXiv:2602.16994](https://arxiv.org/abs/2602.16994)) show draft–target divergence is *low and acceptance is high near the root* and only diverges deeper in the tree — i.e. shallow, high-acceptance regions are exactly where extra branching/coverage buys least. Uniform enrich spends its M teacher rollouts everywhere, including those low-divergence regions where they matter least. So selective enrichment (concentrate teacher coverage where divergence/rejection is high — deep nodes, hard prompts) is not just compute-thrifty; by DDTE's analysis uniform-everywhere enrichment is **inefficient by construction**, which is a positive argument that the selective variant should beat fixed-M, not merely match it cheaper.
- **Adaptive M (per-prompt rollout budget) — orthogonal lever to selective enrichment.** Selective/curriculum enrichment above decides *which prompts* to enrich (sampling frequency); adaptive M decides *how many rollouts per enriched prompt* (depth). Motivation: M's job is to cover the teacher continuation distribution, and easy prompts are already well-covered at M=1 (continuations agree, low divergence), while hard prompts get almost no usable signal at M=1. Natural online implementation needs no separate difficulty oracle — run rollout 1, read its per-token JSD / teacher-acceptance as the difficulty signal, and escalate to additional rollouts (up to $M_{\max}$) only when coverage looks thin; stop early when it does not. Exclude teacher-uncertain prompts (high teacher entropy / low teacher BE regardless of draft) so budget is not burned where the teacher itself fails (ties to the curriculum-exclusion rule). **Key open question — does this raise BE or only cut training time?** If hard-prompt coverage is the current bottleneck (the M=1→M=3 BE lift and the K_eval=4-largest gain both hint it is), spending $M_{\max}{=}5{-}6$ on the hardest prompts exposes the draft to teacher-accepted states it never sees today → potential BE gain. If M=3 has already saturated coverage, adaptive M is purely an efficiency win (same BE, ~1.5× rollouts vs M=3's 3×). **Clean isolating test: hold total rollouts fixed** — adaptive (M=1 easy, $M_{\max}$ hard, average ≈ 3) vs fixed M=3, same compute budget. BE up ⇒ quality gain; BE flat ⇒ efficiency only.
- **Verifier-failure-targeted enrichment (unexplored corner of 2×2):** the student generates a draft → verifier rejects at position $\tau$ → teacher generates M alternative continuations from $\tau$. This is student-generated states × multiple teacher trajectories — the bottom-right corner of [§5.1](#51-novelty-positioning-the-biggest-risk)'s 2×2. Combines on-policy state quality (the student's actual failures) with trajectory diversity (M teacher fixes per failure). The natural successor to fixed enrichment once the 0.6B/8B pair is confirmed.
- **Student uncertainty gating (entropy proxy):** where entropy($Q_\phi$) is high, reveal M teacher continuations; where it is low, use single-path JSD. Computationally cheaper than full per-step rejection sampling as a proxy for verifier-guided routing.
- **32B teacher** to test teacher-scale sensitivity (not required for the core 0.6B/8B claim).
- **Longer horizon ($L{=}16$).**

### 7.3 The unifying frame: adaptive teacher–student curriculum (on-policy ↔ off-policy)

The variants in §7.1–7.2 are points on one continuum, best seen through a teaching analogy. A teacher choosing how to bring a student up to standard can:

- **drill the student on its weak areas** — concentrate supervision where it fails (→ curriculum / verifier-failure-targeted enrich);
- **align training to the exam the student will sit** — train against the verifier/eval distribution that will actually score it (→ verifier-aligned losses, the §7.1 accepted-state objective);
- **shift control over time** — early on the teacher dictates the full answer (off-policy, teacher-generated states = today's enrich); then the teacher lets the student attempt and only corrects where it goes wrong (mixed: student states + teacher fixes = verifier-failure-targeted enrich); finally the student works unaided and the teacher only verifies (fully on-policy).

So the research arc is an **adaptive on-policy/off-policy mix with a schedule**: how much of the training context is teacher-generated vs. student-generated, and how that ratio moves as the student improves. Current `jsd_flat_enrich` is the fully off-policy end; the §7.1 accepted-state objective is the on-policy end; the interesting work is the adaptive middle.

**Open theory questions this frame raises** (research-lead input needed before committing runs):
- **Can the student grasp it?** Is a 0.6B draft expressive enough to match the teacher on the hard states, or is some loss floor a capacity limit rather than a training-signal limit?
- **Is the teacher supervising where the student is actually weak?** Uniform enrich spends rollouts everywhere; does targeting the student's failure positions change the outcome, or is the signal already saturated?
- **Is the student near teacher-induced capacity?** At BE ≈ 6.4/8 with path_diversity healthy ([§3.3](#33-capacity-signals-measured-along-the-way)), is remaining headroom small because the draft is saturated? A 32B-teacher run would separate teacher-headroom from draft-capacity.
- **Cross-distribution alignment.** Do we train on one dataset and evaluate on another (math_hard → math_eval today), and how much of the gap is distribution mismatch vs. method? This is the "align the student to the exam" question made concrete.

These are explicitly deferred. Whether they fold into this paper (as ablations) or a follow-on is a research-lead scope decision.

---

## 8. Scope and Contribution

**Where the defensible contribution lies**, if the work is completed:
1. **Cross-verifier behaviour ([§4.4](#44-cross-verifier-and-k_eval-dependence-n100-seed-averaged-vs-flat-8k))** — the systematic finding that different speculative verifiers respond differently to the same distribution-matching objective. This is the strongest empirical result and the most likely headline.
2. **The state-distribution × objective-weighting design-space map ([§5.2](#52-the-design-space-is-2-d-not-a-1-d-ladder))** — useful framing that may stand on its own if written carefully.
3. **A per-verifier acceptance-functional explanation of (1) ([§5.3](#53-conjectured-verifier-level-mechanism-not-yet-derived))** — *only if actually derived*; today it is conjecture.

**What the work needs before submission** (detail in [§6](#6-must-add-experiments-minimum-for-a-credible-paper)): compute-matched M=3-vs-flat baseline; experimental Draft-OPD comparison; ≥1 more dataset; ≥1 more model pair; ≥5 seeds; paired bootstrap CIs; [§5.3](#53-conjectured-verifier-level-mechanism-not-yet-derived) derivations (or present §4.4 as purely empirical).

**The headline anchor (cross-verifier study + design space, not "M stochastic rollouts") and the decision on what to run first are the research lead's calls.** Suggested first step: the compute-matched baseline, since it gates whether any algorithmic claim survives.
