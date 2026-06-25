# Stochastic Teacher Rollout JSD for Speculative Draft Training
## A Research Note on `jsd_flat_enrich`

**Scope:** one draft (Qwen3-0.6B) / one teacher (Qwen3-8B), one training dataset (math_hard), one eval domain (math_eval), two seeds. n=1000 for traversal/bv/naive ([§4.2](#42-primary-eval-n1000-k_eval3-math_eval)–4.3); n=100 for the 9-verifier × K_eval sweep ([§4.4](#44-cross-verifier-and-k_eval-dependence-n100-seed-averaged-vs-flat-8k)).
**Closest competitor:** Draft-OPD (May 2026) — not yet compared experimentally ([§6](#6-novelty-positioning-the-biggest-risk)).

### Summary

**Method.** `jsd_flat_enrich` replaces the single greedy teacher rollout in standard flat JSD training with M independent stochastic teacher continuations per step. The draft minimises the average JSD against all M teacher-sampled contexts. There are no draft proposals, no verifier decisions, and no accepted/rejected tokens in training — it is a broader-coverage teacher-sampling variant of flat JSD.

**Key results** (one model pair, one dataset, two seeds):
- **M=3 beats flat JSD** on traversal, bv, and naive at K_eval=3, n=1000, both seeds: +1.3% (traversal), +3.2% (bv), +2.5% (naive). M=3 does ~3× the teacher generation per step — not compute-matched.
- **M=3 (8K steps) scores above our converged flat-40K baseline at traversal K=3**: +5.9% (s123) and +2.9% (s456). Our flat run did not reach this with additional training steps.
- **M=1 does not beat flat-40K** (−0.022 overall). The benefit requires M>1.
- **Cross-seed spread collapses under enrich** (flat ~0.19 → enrich ~0.03–0.06). Suggests initialization robustness — tentative, N=2 cannot estimate variance.

**Key analysis:**
- **Verifier responses are heterogeneous.** M=3 wins on 7–9 of 9 verifiers at every K_eval. Traversal and bv gains grow with K_eval; gbv/specinfer/max fade or go negative at high K_eval; naive/spectr/nss/khisti are K-agnostic across K_eval.
- **No matched-K diagonal** (contrast with depth_weight): M=3 gains are positive at all K_eval and are largest at K_eval=1,2 (+0.136, +0.130 seed-avg), not at the "matched" K_eval=3 (+0.082). Broader training coverage transfers regardless of inference tree width.
- **JSD–BE coupling (ρ) strengthens with M** (flat −0.396 → M=1 −0.568 → M=3 −0.645) and varies across verifiers. Coupling and enrich gain are orthogonal: specinfer has the weakest coupling (ρ=−0.402) but the largest K_eval=1 gain (+0.336).

**Next steps** (suggested order):
1. Compute-matched M=3-vs-flat baseline (gates any efficiency claim — [§7](#7-must-add-experiments-minimum-for-a-credible-paper) #3)
2. n=1000 paired bootstrap CIs for headline cells ([§7](#7-must-add-experiments-minimum-for-a-credible-paper) #2)
3. ≥5 seeds per condition ([§7](#7-must-add-experiments-minimum-for-a-credible-paper) #1)
4. ≥1 additional dataset and model pair ([§7](#7-must-add-experiments-minimum-for-a-credible-paper) #7–8)
5. Draft-OPD comparison ([§7](#7-must-add-experiments-minimum-for-a-credible-paper) #5)
6. [§5](#5-verifier-level-math-open-obligations--currently-conjecture-not-derivation) derivations, or present [§4.4](#44-cross-verifier-and-k_eval-dependence-n100-seed-averaged-vs-flat-8k) as purely empirical

---

## 1. Motivation

Standard flat JSD training minimises

$$
\mathcal{L}_{\text{JSD}} = \mathbb{E}_{x_{<t} \sim D}\big[\text{JSD}\big(P_\theta(\cdot \mid x_{<t}) \,\|\, Q_\phi(\cdot \mid x_{<t})\big)\big]
$$

where $P_\theta$ is the teacher, $Q_\phi$ the draft, and $D$ is the **offline teacher rollout distribution** over prefixes. In the current baseline implementation, this is a greedy teacher rollout (`do_sample=False`), not a sample from the full teacher marginal. At inference time the draft instead operates inside a verifier-gated acceptance loop: its own proposals, partially accepted and teacher-corrected, determine the prefixes it conditions on. This is the well-known **offline-to-inference (exposure) mismatch**. The implemented hypothesis is weaker than full on-policy training: replacing a single greedy teacher trajectory with stochastic teacher rollouts gives the draft broader teacher-context coverage and may improve speculative acceptance.

This motivation is **not new** — it is the same mismatch DistillSpec (on-policy draft-generated data), GKD, OSD, and Draft-OPD all target. The current `jsd_flat_enrich` implementation is not a realization of verifier-accepted on-policy training; it is a stochastic-teacher variant of flat JSD. Any contribution must therefore be framed as the *specific low-complexity teacher-sampling instantiation*, its empirical effect, and its relationship to stronger draft-gated/replay methods ([§6](#6-novelty-positioning-the-biggest-risk)), not the mismatch observation itself.

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

The stronger objective we originally wanted to approximate — training on draft proposals that the verifier actually accepts (the verifier-accepted state distribution) — is **not** what `jsd_flat_enrich` implements. It is a proposed future extension; the full objective, why it was not implemented, and what would make it tractable are in **[§8.1](#81-the-ideal-extension-verifier-accepted-state-distribution)**. The current method trains on teacher-sampled contexts only.

### 2.4 Stop-gradient disclosure (important)

For the current stochastic-teacher variant, the sampling distribution is teacher-only and has no $\phi$-gradient to block. Stop-gradient only becomes a substantive issue for the ideal verifier-accepted-state objective ([§8.1](#81-the-ideal-extension-verifier-accepted-state-distribution)), where draft proposals would feed back into the training distribution. In that future draft-gated variant, if we do not differentiate through sampling, the objective is a **stop-gradient Monte Carlo surrogate** for closing the exposure gap — *not* the exact gradient of expected block efficiency $\nabla_\phi \mathbb{E}[\text{BE}]$. The NSS tree-gradient line of work (Rahul) is the route to the *exact* $\partial\text{BE}/\partial\theta$, which composes with — rather than is approximated by — stochastic teacher rollout (see [§6.1](#61-the-design-space-is-2-d-not-a-1-d-ladder)).

### 2.5 Accepted/rejected framing belongs to draft-gated variants

The current implementation trains on teacher-sampled contexts and has no rejection mechanism. Therefore, it should **not** be described as "accepted-only" learning, nor as discarding rejected draft tokens. Accepted-vs-rejected framing is relevant for Draft-OPD and for any future draft-gated `jsd_flat_enrich` extension. If we implement that extension, we must then compare accepted-only training against accepted+rejected replay.

### 2.6 Acceptance–divergence link

For vanilla single-token speculative decoding, per-state acceptance probability is

$$
\alpha(p,q) = \sum_x \min\big(p(x), q(x)\big) = 1 - \text{TV}(p, q).
$$

With JSD measured in nats, Pinsker's inequality applied to the KL sub-terms gives the safe bound $\text{TV}^2 \leq 2\,\text{JSD}$. So minimising JSD ⇒ lower TV ⇒ higher single-token acceptance **on the states being trained**. This chain is clean for naive acceptance; **it does not automatically transfer to BV/GBV/NSS/SpecInfer** (multi-token / tree / optimal-transport criteria). Establishing the link per verifier is an open obligation ([§5](#5-verifier-level-math-open-obligations--currently-conjecture-not-derivation)).

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

ρ strengthens monotonically with M. The p-values confirm statistical significance: p=8.5×10⁻¹² means the probability this correlation arose by chance from 100 prompts is one in 100 billion; p≈0 means the same probability is negligible to the point of rounding to zero.

Per-verifier ρ for M=3 (n=100, s123). JSD is computed solely from draft and teacher probability distributions — the verifier does not affect it. So all four rows share the same JSD statistics (σ(JSD)=0.0200, mean=0.0296); ρ differences across rows reflect verifier structure, not differences in the JSD distribution.

| Verifier | mean_BE | ρ(JSD,BE) | ρ(fwdKL,BE) | % easy | % medium | % hard |
|---|---|---|---|---|---|---|
| traversal | 6.386 | **−0.645** | −0.647 | 62% | 38% | 0% |
| BV | 6.433 | **−0.518** | −0.503 | 67% | 33% | 0% |
| naive | 6.089 | **−0.413** | −0.410 | 50% | 50% | 0% |
| specinfer | 4.867 | **−0.402** | −0.395 | 22% | 75% | 3% |

**ρ ordering reflects how each verifier aggregates acceptance.** Traversal has the tightest coupling (−0.645): it aggregates path-level acceptance products across the tree, so a JSD reduction along a path compounds into a strong BE lift. BV is next (−0.518): a budget cap limits marginal gain beyond a threshold. Naive is looser (−0.413): it reduces to per-token min(1,P/Q), a coarser aggregation than path products. SpecInfer is loosest (−0.402): it has the most hard prompts (3%) and uses multi-candidate residual acceptance based on token-specific ratios, not aggregate JSD. fwdKL and JSD are interchangeable predictors (Δρ < 0.015 for every verifier). σ(JSD) is stable across all four verifiers, confirming that ρ differences are genuine and not an artifact of JSD range compression for one verifier.

**Coupling (ρ) and enrich gain (Δ) are orthogonal.** specinfer has the weakest coupling (ρ=−0.402) yet the largest K_eval=1 enrich gain (+0.336, [§4.2](#42-primary-eval-n1000-k_eval3-math_eval)). ρ measures how reliably JSD predicts BE rank for a fixed checkpoint; Δ measures how much enrich's broader training distribution moves BE. A verifier can have loose JSD coupling but large gain headroom (specinfer), or tight coupling with moderate headroom (traversal). The [§5](#5-verifier-level-math-open-obligations--currently-conjecture-not-derivation) theory must account for both: per-verifier JSD→acceptance slope (predicts ρ) and per-verifier acceptance headroom (predicts Δ).

ρ is a mechanism diagnostic; the findings rest on eval BE (§4). The traversal progression (−0.396 → −0.568 → −0.645) and the four-verifier table suffice to anchor the §5 coupling story; completing all 9 verifier modes is optional.

### 3.3 Capacity signals (measured along the way)

**Student (draft) at capacity:** val/block_eff plateau with no new best; train loss floor (~0.02); high oscillating forgetting with no net BE gain; frozen BE bucket distribution.

**Teacher / data diversity signal:** BE → $L$ suggests limited headroom; path_diversity → 0 means stochastic teacher rollouts have collapsed to near-identical continuations. At 0.6B/8B on math, BE≈6.4/8 and path_diversity ∈ [0.8,1.0] for M=3, so the teacher is still producing diverse contexts. This does **not** prove the teacher is not a bottleneck; it only says stochastic teacher sampling has not collapsed. A 32B-teacher run would test teacher-scale sensitivity (future work, [§8](#8-future-directions-parking-lot--prioritise-later)).

**`train/path_diversity`** = fraction of positions where ≥2 of the $M$ rollouts disagree. ~1.0 ⇒ diverse signal, $M > 1$ contributes; <0.1 ⇒ rollout collapse, $M > 1$ ≈ $M=1$. Observed M=3: ∈[0.8,1.0], healthy.

**`val/forgetting`** = backward-transfer loss (Σ max(0, best_historical_BE(p) − current_BE(p))). Oscillating (not monotonic) ⇒ stability–plasticity churn, not catastrophic forgetting; `ckpt_best` captures the peak.

---

## 4. Results

> All raw per-cell BE in [`Results/jsd_enrich_results.csv`](../Results/jsd_enrich_results.csv). Append new rows to that file; do not paste raw eval dumps into this note.

### 4.1 Training runs

| Run | Steps | Best val BE | Notes |
|---|---|---|---|
| `jsd_mathhard_s123` (flat JSD) | 8K | 6.01 | Flat baseline, seed 1 |
| `jsd_mathhard_s456` (flat JSD) | 8K | 6.291 | Flat baseline, seed 2 |
| `jsd_mathhard_s123` (flat, converged) | 40K | trav K=1 = 6.280 (n=100) | Flat JSD converges ~15K; trained to 40K as the ceiling. Same checkpoint used as the flat baseline in the depth_weight ablation ([40K CSV](https://github.com/Rmuk655/Distill-Spec-Research/blob/Pipeline/Results/jsd_jsd_dw_naive_tree_results_math_hard_40000_steps.csv)). BE is hardware-independent. |
| `jsd_flat_enrich_M1_s123` | 8K | 6.409 | M=1 enrich, seed 1 |
| `jsd_flat_enrich_M1_s456` | 8K | — | M=1 enrich, seed 2 |
| `jsd_flat_enrich_M3_s123` | 8K | 6.420 | M=3 enrich, seed 1 |
| `jsd_flat_enrich_M3_s456` | 8K | — | M=3 enrich, seed 2 |
| `jsd_flat_enrich_M3_s123_ttemp1.5` | 8K | — | Negative control: teacher_temp=1.5 |

`ckpt_best` is winner's-curse biased (max over ~hundreds of 25-prompt val evals, SE≈0.15–0.20). All reported BE uses n=100 or n=1000 offline re-evals of `ckpt_best`, unaffected by selection noise.

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

M=3 beats flat on all three modes, both seeds. M=3 > M=1 in bv and naive (+0.07); traversal shows no M=3/M=1 difference at K_eval=3 (see [§4.3](#43-k_eval2-results-n1000-enrich-only--flat-k_eval2-not-yet-run) — traversal peaks at K_eval=2).

**Seed variance collapse:**

| condition | Δ(s456−s123) traversal | Δ(s456−s123) bv | Δ(s456−s123) naive |
|---|---|---|---|
| flat | **+0.193** | **+0.190** | **+0.168** |
| M=1 | 0.020 | −0.027 | −0.047 |
| M=3 | 0.065 | 0.085 | 0.044 |

Flat JSD has a large systematic seed spread (~0.19 per mode; s456 consistently higher than s123 across all modes). Enrich training collapses this to ~0.03–0.09 — both initializations converge to nearly the same basin. Enrich is initialization-robust. Seed-averaged flat values are the authoritative baseline; single-seed flat comparisons are unreliable.

### 4.3 K_eval=2 results (n=1000, enrich only — flat K_eval=2 not yet run)

| condition | traversal | naive |
|---|---|---|
| M=1 s123 | 6.073 | 5.808 |
| M=1 s456 | 6.047 | 5.847 |
| **M=1 avg** | **6.060** | **5.828** |
| M=3 s123 | 6.141 | 5.926 |
| M=3 s456 | 6.005 | 5.778 |
| **M=3 avg** | **6.073** | **5.852** |
| M=3 vs M=1 | +0.013 | +0.024 |

**Traversal peaks at K_eval=2.** M=3 traversal at K_eval=2 (avg 6.073) > K_eval=3 (avg 6.024) on both seeds. The K_eval=M alignment is not confirmed for traversal at n=1000; for traversal, M=3 is most effective at K_eval=2. BV at K_eval=2 is not yet run at n=1000.

### 4.4 Cross-verifier and K_eval-dependence (n=100, seed-averaged, vs flat-8K)

This is the mechanism sweep: all 9 verifiers × K_eval=1..4, seed-averaged over s123+s456, baseline = flat-8K (the under-trained baseline — the converged flat-40K comparison is [§4.5](#45-m3-8k-steps-vs-our-strongest-flat-baseline-40k-steps)). Raw cells in [`Results/jsd_enrich_results.csv`](../Results/jsd_enrich_results.csv).

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

**No matched-K diagonal — the key contrast with depth_weight.** In the depth_weight ablation the only gain over baseline sat on the K_train = K_eval diagonal and collapsed off it. Enrich is the opposite: M=3's gain is positive at *every* K_eval and is actually **largest at K_eval=1,2** (+0.136, +0.130), not at the "matched" K_eval=3 (+0.082). The mechanism (broader teacher context) is not tied to inference tree width — enrich wins regardless of K_eval. M=1 is the weaker version of the same shape and sags mid-K (5/9 at K=2).

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

**Three K-dependence classes** (the empirical core [§5](#5-verifier-level-math-open-obligations--currently-conjecture-not-derivation) theory must reproduce):
- **Improve with K** (gain grows toward K=4): **traversal, bv** — prefix/budget verifiers; M parallel-calibrated branches compound as tree width grows.
- **Worsen with K** (front-loaded, fade or go negative by K≥3): **gbv, specinfer, max** — residual/competition verifiers whose cross-candidate normalisation is sub-additive in branch count.
- **K-agnostic** (flat positive across K): **naive, spectr, nss, khisti**.

**ρ and Δ are orthogonal.** specinfer has a large low-K gain despite the weakest ρ(JSD,BE) (−0.402, [§4.6](#46-objectivebe-alignment-diagnostics-ρ)). ρ measures surrogate alignment for a fixed checkpoint; Δ measures realized headroom — a verifier can have loose coupling but high headroom.

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

Our flat JSD run stops improving by ~15K steps; we trained it to 40K and treat that as our strongest flat baseline (one optimization trajectory — *not* a proof of the flat ceiling; other schedules/objectives might do better). Step-count caveat: M=3-8K does ≈24K teacher rollouts (3/step) vs flat-40K's ≈40K (1/step), so the per-step compute differs in both directions (M=3 does more per step, fewer steps). **This is not a compute-matched comparison; the compute-matched baseline ([§7](#7-must-add-experiments-minimum-for-a-credible-paper)) is still required before any efficiency claim.** flat-40K evaluated on H100; BE is hardware-independent and comparable to the A100 8K runs.

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
- **Open:** the n=1000 BV K_eval=2 cell is unrun, and a third seed would break the s123/s456 tie.

**M=1 has no edge over flat-40K.** M=1-8K is slightly below flat-40K (−0.022). A single stochastic trajectory behaves like flat JSD with broader context; the M>1 setting is what moves the metric here.

### 4.6 Objective–BE alignment diagnostics (ρ)

From `--diagnose` mode (n=100, math_eval):

**ρ progression with M (traversal mode):** flat JSD −0.396 → M=1 −0.568 → M=3 **−0.645**. Monotonically strengthening.

**Per-verifier ρ (M=3, s123), σ(JSD)=0.0200 stable across all verifiers:**

| Verifier | mean_BE | ρ(JSD,BE) | ρ(fwdKL,BE) | % easy | % medium | % hard |
|---|---|---|---|---|---|---|
| traversal | 6.386 | **−0.645** | −0.647 | 62% | 38% | 0% |
| BV | 6.433 | **−0.518** | −0.503 | 67% | 33% | 0% |
| naive | 6.089 | **−0.413** | −0.410 | 50% | 50% | 0% |
| specinfer | 4.867 | **−0.402** | −0.395 | 22% | 75% | 3% |

σ(JSD) stable → ρ differences are not just range compression. JSD and fwdKL are interchangeable as predictors (Δρ < 0.015). **Caveat (do not oversell):** a higher |ρ| means JSD is a better *monotonic predictor* of BE for a fixed checkpoint — it does **not** by itself mean JSD is a better training objective, and it does not establish causation. ρ is a supporting diagnostic only; the findings rest on eval BE, not on ρ.

**Negative control (teacher_temp=1.5):** ρ drops from −0.645 to −0.543, mean_JSD rises 8%, σ stable — consistent with genuine signal loss rather than range compression. Plausible mechanism: the student learns a noisier teacher distribution while the verifier runs at temp=1.0 at inference (training-inference mismatch). On this evidence teacher_temp=1.0 is the better setting; we did not sweep further.

**train/path_diversity (M=3):** ∈[0.8,1.0] — stochastic teacher rollouts are genuinely diverse; M>1 contributes distinct contexts, not near-duplicate paths.

### 4.7 Conclusions

All conclusions are scoped to this setup (one model pair, one dataset, two seeds, no CIs) and are stated as observations.

**Supported on this setup:**

1. **M=3 enrich scores above flat JSD** on traversal (+1.3%), bv (+3.2%), naive (+2.5%), both seeds, n=1000, K_eval=3, gaps >4×SE. *Not compute-matched* — could be partly the extra teacher generation. [[§4.2](#42-primary-eval-n1000-k_eval3-math_eval)]
2. **M=3 (8K) scores above our flat-40K baseline at traversal K=3** (+5.9% s123 / +2.9% s456); our flat run did not reach this with more steps. Suggestive of an M>1 effect; *not* a proof flat cannot reach it. [[§4.5](#45-m3-8k-steps-vs-our-strongest-flat-baseline-40k-steps)]
3. **Cross-seed spread is smaller under enrich** (~0.19 → ~0.03–0.06), *suggesting* initialization robustness. Tentative only — N=2 cannot estimate variance; needs ≥5 seeds. Seed-averaged flat baselines are in any case mandatory. [[§4.2](#42-primary-eval-n1000-k_eval3-math_eval)]
4. **Verifier response is heterogeneous (the strongest finding).** M=3 scores above flat-8K on 7–9 of 9 verifiers at every K_eval; bv/traversal gains grow with K_eval while gbv/specinfer/max fade or reverse. [[§4.4](#44-cross-verifier-and-k_eval-dependence-n100-seed-averaged-vs-flat-8k)]
5. **|ρ|(JSD,BE) increases with M** (flat −0.396 → M=1 −0.568 → M=3 −0.645) — a monotonic-association diagnostic, not evidence about the objective itself. [[§4.6](#46-objectivebe-alignment-diagnostics-ρ)]

**Not supported / refuted:**

6. **A single stochastic rollout (M=1) is not enough** — it does not beat flat-40K (−0.022). The M>1 setting is what moves the metric. [[§4.5](#45-m3-8k-steps-vs-our-strongest-flat-baseline-40k-steps)]
7. **"Matched K_eval = M is optimal" does not hold for enrich.** M=3 is strongest at K_eval=1,2 (+0.136/+0.130), not at matched K_eval=3 (+0.082) — K-agnostic, opposite to the depth_weight ablation's diagonal pattern. [[§4.4](#44-cross-verifier-and-k_eval-dependence-n100-seed-averaged-vs-flat-8k)]
8. **Higher teacher temperature does not help** — teacher_temp=1.5 degrades alignment (ρ −0.645 → −0.543); 1.0 is the better setting on this evidence. [[§4.6](#46-objectivebe-alignment-diagnostics-ρ)]

**Open / needs more data:**

9. **BV: convergence-speed vs raised baseline unresolved.** M=3 scores above flat-8K on BV everywhere, but the advantage over flat-40K reverses on s456. Needs n=1000 BV K_eval=2 + a tie-break seed. [[§4.5](#45-m3-8k-steps-vs-our-strongest-flat-baseline-40k-steps)]
10. **specinfer/max anti-alignment at high K_eval** (M=3 negative at K≥3 / K=3, both seeds) — observed but unexplained; gated on a [§5](#5-verifier-level-math-open-obligations--currently-conjecture-not-derivation) derivation that does not yet exist.
11. **Paired bootstrap CIs** for the headline cells ([§7](#7-must-add-experiments-minimum-for-a-credible-paper) #2).

**Cross-verifier K-dependence (the empirical headline for [§5](#5-verifier-level-math-open-obligations--currently-conjecture-not-derivation)):** three classes — *improve with K* (traversal, bv: prefix/budget), *worsen with K* (gbv, specinfer, max: residual/competition, fade or go negative by K≥3), *K-agnostic* (naive, spectr, nss, khisti). The [§5](#5-verifier-level-math-open-obligations--currently-conjecture-not-derivation) acceptance-functional theory must reproduce this split. [[§4.4](#44-cross-verifier-and-k_eval-dependence-n100-seed-averaged-vs-flat-8k)]


---

## 5. Verifier-Level Math (open obligations — currently conjecture, not derivation)

> **Note:** only the naive case ($\alpha=1-\text{TV}$, via Pinsker) is derived; the explanations for traversal/BV/GBV/specinfer/NSS below are plausible intuitions, not proven. Either complete these derivations, or present §4.4 as purely empirical. Toy finite-vocabulary experiments (enumerate exact acceptance and BE, match against simulation) would make the verifier story hard to attack.

**Core question — why does stochastic-teacher JSD help verifiers differently?** Each verifier's per-state acceptance probability is a *different functional* of $(P,Q)$. For naive single-token acceptance, $\alpha_{\text{naive}}(P,Q)=\sum_x\min(P(x),Q(x))=1-\text{TV}(P,Q)$, and JSD bounds TV ($\text{TV}^2\le 2\,\text{JSD}$ in nats), so lowering JSD on the trained states should raise naive acceptance on those states. For NSS (optimal-transport), BV/GBV (tree/budget), and SpecInfer, the acceptance functional is **not** TV, so the same JSD reduction would map to a *different* marginal acceptance gain — that mapping is the conjectured explanation for the differential cross-verifier benefit, and deriving it per verifier *would be* the theory contribution if completed (it is not, yet). Draft-OPD gives a local accepted/rejected KL rationale, but not this cross-verifier acceptance-functional analysis.

1. **$K_{\text{eval}}{=}1$ collapse — likely a one-line remark, not a theorem.** At $K_{\text{eval}}{=}1$, the verifier sees a single draft path. For sequential token verification up to depth $L$, naive/spectr/khisti/max plausibly reduce to the same per-token accept-w.p.-$\min(1,P/Q)$ rule along that path. The proof should state this for arbitrary $L$, not only $L=1$. If the reduction is trivial, state it as a remark for completeness — **do not present it as a contribution.** Empirically the four give identical BE for both checkpoints (5.848 / 5.859).
2. **Exact expected-BE for tree/OT verifiers.** For NSS/BV/GBV/traversal, either derive exact expected-BE formulas or state precisely why stochastic-teacher JSD is only a surrogate. Back this with **toy finite-vocabulary experiments** where acceptance and BE can be enumerated exactly and matched against simulation — this makes the verifier story hard to attack.
3. **Khisti antagonism.** Khisti is the only verifier with consistent negatives ($K_{\text{eval}}{=}2,4$). Needs a mechanism: what calibration pattern does stochastic-teacher JSD induce that khisti penalises?
4. **Acceptance–divergence transfer ([§2.6](#26-acceptancedivergence-link))** beyond naive.
5. **Verifier–K alignment ([§4.4](#44-cross-verifier-and-k_eval-dependence-n100-seed-averaged-vs-flat-8k)).** Why do prefix/budget verifiers (traversal, BV) keep or grow their enrich Δ as $K_{\text{eval}}\to M$ and beyond, while residual verifiers (specinfer) lose it at high $K_{\text{eval}}$? Conjecture: traversal/BV acceptance reward is monotone increasing in tree width given per-branch calibration, so M parallel-calibrated branches compound; specinfer's residual normalisation across candidates is sub-additive in branch count, so added branches dilute. Derive the $K_{\text{eval}}$-dependence of expected-BE per verifier and show it predicts the observed alignment/anti-alignment.
6. **Coupling vs headroom ([§3.2](#32-diagnostics---diagnose)).** Two separate per-verifier functionals: the JSD→acceptance *slope* (predicts ρ) and the acceptance *headroom* relative to a flat-JSD-trained draft (predicts Δ). These are empirically orthogonal (specinfer: low ρ, high Δ). A complete theory derives both from the verifier's acceptance functional.

---

## 6. Novelty Positioning (the biggest risk)

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

The bottom-right corner — student-generated contexts with M teacher continuations per rollout — is the natural adaptive variant: the student's own failures determine where the teacher supervises. See [§8](#8-future-directions-parking-lot--prioritise-later) (Curriculum Enrichment).

**What Draft-OPD leaves genuinely open:** cross-verifier behaviour and theory for verifier-specific acceptance functionals. Their own future-work line — extending OPD to *approximate/lossy verification* — is adjacent to our verifier work, so we are not contradicting them, we are entering the gap they flagged.

**Defensible claims (narrow):**
- *"For verifier-conditioned speculative decoding, how acceptance-aware training transfers across verifier families is unexplored; we map it empirically (9 verifiers × eval-K) and explain the differential with a per-verifier acceptance-functional analysis."*
- *"Stochastic-teacher symmetric JSD is a low-complexity teacher-sampling alternative; we measure the gap to Draft-OPD-style replay under matched compute."*

**Mandatory comparison:** stochastic teacher rollout vs. draft-gated accepted-only vs. accepted+rejected replay, and symmetric JSD vs. asymmetric fwd/rev-KL with $\gamma^{k-1}$ decay. Otherwise reviewers correctly say we have not located the result relative to a more complete on-policy method.

### 6.1 The design space is 2-D, not a 1-D ladder

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
| Rahul depth_weight | (any) | crude `depth × JSD` multiply, **no gradient** |
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

---

## 7. Must-Add Experiments (minimum for a credible paper)

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
| 9 | ~~Temperature robustness — teacher_temp~~ | **Answered**: ttemp=1.5 run shows ρ=−0.543 vs default ρ=−0.645; higher teacher_temp causes training-inference mismatch. teacher_temp=1.0 preferred on this evidence. Eval temperature sweep (not training) still needed if running approximate verifiers. |
| 10 | Wall-clock tokens/sec + output quality/exactness | BE alone is insufficient |

Near-term order: finish M=3 eval → run s456 (flat + stochastic-teacher JSD) → n=1000 on best checkpoints → compute-matched greedy/stochastic/M ablations → Draft-OPD-style replay ablation.

---

## 8. Future Directions (parking lot — prioritise later)

### 8.1 The ideal extension: verifier-accepted state distribution

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

### 8.2 Other parked directions

- **Verifier-weighted stochastic teacher JSD (Axis A × Axis B combination) — PARKED, likely premature.** The synthesis would weight each of M teacher rollouts by its E[τ_V]: $\frac{1}{M}\sum_m \frac{d_m}{\text{EMA}(d)} \cdot \text{JSD}(Q_\phi(\cdot|y^{(m)}), P_\theta(\cdot|y^{(m)}))$. **Three reasons not to do this yet:** (1) **depth_weight has shown no positive signal on its own** (scalar version underperformed flat) — combining a no-signal method with a cross-seed-weakened one violates one-variable-at-a-time. (2) **The default multiply direction is antagonistic to enrich:** survival-weighting up-weights deep/easy contexts, but enrich's whole purpose is to add *hard/diverse* contexts — depth-multiply would down-weight exactly what enrich is trying to inject. Only `depth_lambda < 0` (divide, drill-the-weak) is conceptually compatible with enrich's coverage goal, and that variant is unrun. (3) Enrich's own headline is now downgraded by the cross-seed analysis ([§4.2](#42-primary-eval-n1000-k_eval3-math_eval)). Revisit only after depth_weight alone produces a positive result AND enrich clears n=1000.
- **NSS tree gradients (Rahul):** exact survival-weighted $\partial\text{BE}/\partial\theta$ — an **Axis-B** (objective) change, **orthogonal to enrich's Axis-A** (state-distribution) change; they compose rather than approximate each other. The exact gradient is the strongest unscooped asset (Draft-OPD has no theory). Not yet implemented; depth_weight (crude `depth × JSD` multiply, no gradient) is the only Axis-B experiment so far and is not expected to be promising.
- **NSS-depth and broader tree gradients / tree-depth ablations** (vary $L$, vary $M$).
- **Adaptive teacher curriculum:** soft vs hard accept; hard-prompt up-weighting by inverse BE (exclude teacher-uncertain prompts); teacher-temperature scheduling (warm→cool).
- **Adaptive teacher using tree depth** to decide where to guide the student.
- **Expected-depth survival weighting ([§6.1](#61-the-design-space-is-2-d-not-a-1-d-ladder)):** weight per-position JSD by predicted marginal acceptance-length gain $E[\tau_V]$. A distinct Axis-B corner between Draft-OPD's $\gamma^{k-1}$ and the exact NSS gradient; **expected to roughly match, not clearly beat, M=3 stochastic-teacher JSD** (it reweights existing positions). Pursue **position-level only**; validate with a cheap short probe gated behind M=3 confirmation.
- **Curriculum Enrichment / adaptive state selection (*Curriculum Enrichment Distillation*):** the strongest version of `jsd_flat_enrich` is not uniform M-path sampling but selective enrichment — sample M teacher branches where the verifier rejects (acceptance low); use standard flat JSD where acceptance is high. Teacher compute is spent only where BE is weakest. This is conceptually closer to active learning / a "Socratic teacher" than a fixed loss: *train where the student needs it.* Logically prior to any fixed-M scale-up; if it works at M=3, the adaptive version should be strictly more compute-efficient.
- **Verifier-failure-targeted enrichment (unexplored corner of 2×2):** the student generates a draft → verifier rejects at position $\tau$ → teacher generates M alternative continuations from $\tau$. This is student-generated states × multiple teacher trajectories — the bottom-right corner of [§6](#6-novelty-positioning-the-biggest-risk)'s 2×2. Combines on-policy state quality (the student's actual failures) with trajectory diversity (M teacher fixes per failure). The natural successor to fixed enrichment once the 0.6B/8B pair is confirmed.
- **Student uncertainty gating (entropy proxy):** where entropy($Q_\phi$) is high, reveal M teacher continuations; where it is low, use single-path JSD. Computationally cheaper than full per-step rejection sampling as a proxy for verifier-guided routing.
- **32B teacher** to test teacher-scale sensitivity (not required for the core 0.6B/8B claim).
- **Longer horizon ($L{=}16$).**

These are explicitly deferred. Whether they fold into this paper (as ablations) or a follow-on is a research-lead scope decision.

---

## 9. Scope and Contribution

**Where the defensible contribution lies**, if the work is completed:
1. **Cross-verifier behaviour ([§4.4](#44-cross-verifier-and-k_eval-dependence-n100-seed-averaged-vs-flat-8k))** — the systematic finding that different speculative verifiers respond differently to the same distribution-matching objective. This is the strongest empirical result and the most likely headline.
2. **The state-distribution × objective-weighting design-space map ([§6.1](#61-the-design-space-is-2-d-not-a-1-d-ladder))** — useful framing that may stand on its own if written carefully.
3. **A per-verifier acceptance-functional explanation of (1) ([§5](#5-verifier-level-math-open-obligations--currently-conjecture-not-derivation))** — *only if actually derived*; today it is conjecture.

**What the work needs before submission** (detail in [§7](#7-must-add-experiments-minimum-for-a-credible-paper)): compute-matched M=3-vs-flat baseline; experimental Draft-OPD comparison; ≥1 more dataset; ≥1 more model pair; ≥5 seeds; paired bootstrap CIs; [§5](#5-verifier-level-math-open-obligations--currently-conjecture-not-derivation) derivations (or present §4.4 as purely empirical).

**The headline anchor (cross-verifier study + design space, not "M stochastic rollouts") and the decision on what to run first are the research lead's calls.** Suggested first step: the compute-matched baseline, since it gates whether any algorithmic claim survives.
