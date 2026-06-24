# Stochastic Teacher Rollout JSD for Speculative Draft Training
## A Research Note on `jsd_flat_enrich`

**Status:** Early internal result — promising signal, **not yet paper-ready.** M=1 complete on both seeds (s123, s456); M=3 complete on s123. The one remaining replication is **M=3 on the second seed (s456)**.
**Draft–Teacher pair:** Qwen3-0.6B draft / Qwen3-8B teacher
**Training data:** math_hard. **Eval data:** math_eval (n=100; paper target n=1000).
**Positioning:** Builds on [2602.16994]; closely related to DistillSpec, on-policy GKD, Online Speculative Decoding, and **Draft-OPD (May 2026)** — see §6 for the differentiator we must defend.

> **Honest framing.** The current evidence supports: *"stochastic teacher rollouts may improve flat JSD on this Qwen3 math setup."* It does **not** yet prove a general method. This note is scoped to make the claim defensible, identify what is novel vs. incremental, and list the experiments and proofs required before submission.

---

## 1. Motivation

Standard flat JSD training minimises

$$\mathcal{L}_{\text{JSD}} = \mathbb{E}_{x_{\lt t} \sim D}\big[\text{JSD}\big(P_\theta(\cdot \mid x_{\lt t}) \,\|\, Q_\phi(\cdot \mid x_{\lt t})\big)\big]$$

where $P_\theta$ is the teacher, $Q_\phi$ the draft, and $D$ is the **offline teacher rollout distribution** over prefixes. In the current baseline implementation, this is a greedy teacher rollout (`do_sample=False`), not a sample from the full teacher marginal. At inference time the draft instead operates inside a verifier-gated acceptance loop: its own proposals, partially accepted and teacher-corrected, determine the prefixes it conditions on. This is the well-known **offline-to-inference (exposure) mismatch**. The implemented hypothesis is weaker than full on-policy training: replacing a single greedy teacher trajectory with stochastic teacher rollouts gives the draft broader teacher-context coverage and may improve speculative acceptance.

This motivation is **not new** — it is the same mismatch DistillSpec (on-policy draft-generated data), GKD, OSD, and Draft-OPD all target. The current `jsd_flat_enrich` implementation is not a realization of verifier-accepted on-policy training; it is a stochastic-teacher variant of flat JSD. Any contribution must therefore be framed as the *specific low-complexity teacher-sampling instantiation*, its empirical effect, and its relationship to stronger draft-gated/replay methods (§6), not the mismatch observation itself.

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

$$y^{(m)} \sim P_\theta^{T_{\text{teacher}}}(\cdot \mid p), \qquad m=1,\dots,M,$$

where sampling is implemented by `teacher.generate(..., do_sample=True, temperature=teacher_temp)`. It then scores both teacher and draft distributions on the same teacher-generated sequence and averages the per-token JSD:

$$\mathcal{L}_{\text{stoch-teacher}} =
\frac{1}{M}\sum_{m=1}^M
\frac{1}{|y^{(m)}|}\sum_{t=1}^{|y^{(m)}|}
\text{JSD}\big(P_\theta(\cdot \mid p, y^{(m)}_{\lt t}) \,\|\, Q_\phi(\cdot \mid p, y^{(m)}_{\lt t})\big).$$

There are **no draft proposals, no verifier accept/reject decisions, and no teacher-corrected residuals** in the current flat-enrich code path. $M=1$ isolates greedy-vs-stochastic teacher training; $M \gt 1$ adds multiple stochastic teacher trajectories per prompt and tests whether path diversity gives useful extra contexts.

### 2.3 Ideal extension: verifier-accepted state distribution

The stronger objective we originally wanted to approximate is the **verifier-induced prefix kernel** $\mathcal{J}_V$: sample draft proposals, verify them with $V$, keep the accepted prefix, resample the first rejected position from the teacher residual, and train on the resulting verified prefix. Formally, $x^{\star} \sim \mathcal{J}_V(Q_\phi, P_\theta; \pi)$ and

$$\mathcal{L}_{\text{accepted-state}} =
\mathbb{E}_{x^{\star} \sim \mathcal{J}_V}\!\left[\frac{1}{|x^{\star}|}\sum_{t=1}^{|x^{\star}|}
\text{JSD}\big(P_\theta(\cdot \mid p, x^{\star}_{\lt t}) \,\|\, Q_\phi(\cdot \mid p, x^{\star}_{\lt t})\big)\right].$$

This is a **proposed extension / idealized objective**, not the current `jsd_flat_enrich` implementation.

### 2.4 Stop-gradient disclosure (important)

For the current stochastic-teacher variant, the sampling distribution is teacher-only and has no $\phi$-gradient to block. Stop-gradient only becomes a substantive issue for the ideal $\mathcal{J}_V$ objective, where draft proposals would feed back into the training distribution. In that future draft-gated variant, if we do not differentiate through sampling, the objective is a **stop-gradient Monte Carlo surrogate** for closing the exposure gap — *not* the exact gradient of expected block efficiency $\nabla_\phi \mathbb{E}[\text{BE}]$. The NSS tree-gradient line of work (Rahul) is the route to the *exact* $\partial\text{BE}/\partial\theta$, which composes with — rather than is approximated by — stochastic teacher rollout (see §6.1).

### 2.5 Accepted/rejected framing belongs to draft-gated variants

The current implementation trains on teacher-sampled contexts and has no rejection mechanism. Therefore, it should **not** be described as "accepted-only" learning, nor as discarding rejected draft tokens. Accepted-vs-rejected framing is relevant for Draft-OPD and for any future draft-gated `jsd_flat_enrich` extension. If we implement that extension, we must then compare accepted-only training against accepted+rejected replay.

### 2.6 Acceptance–divergence link

For vanilla single-token speculative decoding, per-state acceptance probability is

$$\alpha(p,q) = \sum_x \min\big(p(x), q(x)\big) = 1 - \text{TV}(p, q).$$

With JSD measured in nats, Pinsker's inequality applied to the KL sub-terms gives the safe bound $\text{TV}^2 \leq 2\,\text{JSD}$. So minimising JSD ⇒ lower TV ⇒ higher single-token acceptance **on the states being trained**. This chain is clean for naive acceptance; **it does not automatically transfer to BV/GBV/NSS/SpecInfer** (multi-token / tree / optimal-transport criteria). Establishing the link per verifier is an open obligation (§5).

---

## 3. Measurement Framework

### 3.1 Primary metric: Block Efficiency

$$\text{BE} = \frac{\text{generated tokens}}{\text{target model calls}}, \quad \text{BE}_{\max} = L.$$

At $L=8$: observed range 3.7 (NSS, strictest) to 6.4 (traversal). **Report wall-clock tokens/sec alongside BE** — BE and throughput correlate at ~0.95 globally but can decouple; a method that raises BE but not throughput is not useful. Also report output quality/exactness for any approximate verifier.

L-relative buckets: easy $\geq 0.75L$, medium $[0.375L, 0.75L)$, hard $\lt 0.375L$.

### 3.2 Diagnostics (`--diagnose`)

**Spearman $\rho$(divergence, BE):** rank correlation, robust to JSD range compression as the model improves (unlike Pearson r).
- Observed (K_eval=1, n=100): flat JSD $\rho=-0.396$; M=1 $\rho=-0.568$ ($p=8.5\times10^{-12}$); M=3 $\rho=-0.645$ ($p\approx0$). Monotonically strengthening objective–BE alignment with M (all traversal mode).
- Per-verifier ρ for M=3 (all n=100, same σ(JSD)=0.0200, mean_JSD=0.0296 — JSD is model-only, independent of verifier):

| Verifier | mean_BE | ρ(JSD,BE) | ρ(fwdKL,BE) | R²(JSD) | % easy | % medium | % hard | σ(BE) |
|---|---|---|---|---|---|---|---|---|
| traversal | 6.386 | **−0.645** | −0.647 | 0.37 | 62% | 38% | 0% | 1.064 |
| BV | 6.433 | **−0.518** | −0.503 | 0.24 | 67% | 33% | 0% | 1.110 |
| naive | 6.089 | **−0.413** | −0.410 | 0.19 | 50% | 50% | 0% | 1.307 |
| specinfer | 4.867 | **−0.402** | −0.395 | 0.20 | 22% | 75% | 3% | 1.136 |

  σ(JSD) stable across all four → ρ differences are Case A (true signal, not range restriction). ρ(fwdKL,BE) ≈ ρ(JSD,BE) within 0.015 for every verifier — fwdKL and JSD are interchangeable as BE predictors. ρ ordering (traversal > BV > naive ≈ specinfer): traversal aggregates path-level acceptance products (strong JSD→BE slope); BV has a budget cap that truncates marginal gains; naive reduces to per-token min(1,P/Q), a coarser aggregation; specinfer is the strictest verifier here (mean_BE=4.867, the only one with hard prompts), and its multi-candidate residual acceptance depends on token-specific ratios rather than aggregate JSD → loosest coupling.

  **Coupling (ρ) and improvement (Δ from enrich) are orthogonal axes — do not conflate them.** specinfer has the *lowest* ρ (−0.402) yet the *largest* K_eval=1 enrich gain (+0.336, §4.2). ρ measures how reliably JSD predicts BE rank *for a fixed checkpoint* (surrogate quality); Δ measures how much enrich's broadened distribution *moves* BE (realized headroom). A verifier can have loose JSD coupling but large headroom (specinfer) or tight coupling and moderate headroom (traversal). The §5 theory must account for both: per-verifier JSD→acceptance *slope* (predicts ρ) and per-verifier acceptance *headroom* (predicts Δ).

  **ρ is a supplementary mechanism diagnostic, NOT a paper deliverable.** The four verifiers above (M=3 s123) + the traversal progression (flat −0.396 → M=1 −0.568 → M=3 −0.645) already suffice to anchor the §5 coupling story; completing all 9 modes is optional polish. A full diagnose grid (every checkpoint × 9 verifiers) is being recorded to W&B (`diag/*` keys) but is not required and need not be scraped from stdout. If ever assembled, keep two axes separate: (1) per-checkpoint hierarchy (fix checkpoint, sweep modes); (2) cross-checkpoint progression (fix mode, sweep flat→M=1→M=3). Mixed-checkpoint cells (e.g. NSS ρ=−0.335 on M=1 s456; GBV ρ=−0.308 on M=3 ttemp1.5) belong to their own checkpoint's row, not this table. Directionally the loose tail already matches theory (NSS = survival-weighted/OT = weakest JSD surrogate). **The paper rests on eval BE, not ρ.**

**$\sigma$(JSD) stability (Case A vs B):** Case A (σ stable, ρ↑) = true signal; Case B (σ collapses, ρ stable) = range restriction. M=3 σ(JSD)=0.0200, mean_JSD=0.0296 across both diagnose runs — σ not collapsed ⇒ **Case A** confirmed.

### 3.3 Capacity signals (measured along the way)

**Student (draft) at capacity:** val/block_eff plateau with no new best; train loss floor (~0.02); high oscillating forgetting with no net BE gain; frozen BE bucket distribution.

**Teacher / data diversity signal:** BE → $L$ suggests limited headroom; path_diversity → 0 means stochastic teacher rollouts have collapsed to near-identical continuations. At 0.6B/8B on math, BE≈6.4/8 and path_diversity ∈ [0.8,1.0] for M=3, so the teacher is still producing diverse contexts. This does **not** prove the teacher is not a bottleneck; it only says stochastic teacher sampling has not collapsed. A 32B-teacher run would test teacher-scale sensitivity (future work, §8).

**`train/path_diversity`** = fraction of positions where ≥2 of the $M$ rollouts disagree. ~1.0 ⇒ diverse signal, $M \gt 1$ contributes; <0.1 ⇒ rollout collapse, $M \gt 1$ ≈ $M=1$. Observed M=3: ∈[0.8,1.0], healthy.

**`val/forgetting`** = backward-transfer loss (Σ max(0, best_historical_BE(p) − current_BE(p))). Oscillating (not monotonic) ⇒ stability–plasticity churn, not catastrophic forgetting; `ckpt_best` captures the peak.

---

## 4. Results (encouraging but inconclusive)

### 4.1 Training runs

| Run | Steps | Best val BE | Status |
|---|---|---|---|
| `jsd_mathhard_s123` (flat JSD) | 8K | 6.01 | Complete |
| `jsd_flat_enrich_M1_s123` | 8K | 6.409 | Complete |
| `jsd_flat_enrich_M3_s123` | 8K | 6.420 | Complete; eval done |
| `jsd_flat_enrich_K3_mathhard_s123_ttemp1.5` | 8K | — | Complete; **teacher_temp=1.5 hurts**: diagnose traversal ρ=−0.543 vs default ρ=−0.645; mean_JSD=0.0321 vs 0.0296 (8% worse); mean_BE=6.340 vs 6.386. σ(JSD) stable → true signal loss, not range restriction. Mechanism: student learns to match noisier teacher; inference verifier uses ttemp=1.0, creating training-inference temperature mismatch. **teacher_temp=1.0 confirmed; no further temp variants needed.** |
| `jsd_flat_enrich_M1_s456` | 8K | — | Complete; **full K_eval=1..4 eval done** (best n=100 traversal BE 6.111 at K=2). vs flat-s456 (paired): overall +0.037, bv/trav +0.092 — within the seed-variance band (§4.4); M=1 enrich ≈ plain jsd on both seeds. |
| `jsd_mathhard_s456` (flat JSD) | 8K | 6.291 | Complete (total 1052 min); second-seed flat baseline |

### 4.2 Three-checkpoint comparison at K_eval=1 (n=100, math_eval)

> **Raw per-cell BE for every (run × seed × verifier × K_eval) lives in [`Results/jsd_enrich_results.csv`](../Results/jsd_enrich_results.csv).** This note holds only the derived deltas and insights — do not paste new raw eval dumps here; append them to that file and update the insight prose.

Block efficiency for all three checkpoints at K_eval=1, L=8. **M=3 beats flat JSD on all 9 verifiers — no exceptions.** Monotonic improvement on 8/9 verifiers (NSS: M=3 ≈ M=1).
**Caveat: single-seed, n=100 point estimates, not significance-tested.** See §4.4 — and note the measured seed floor (~+0.09 mean / 0.12 std) below before reading any single cell as a result.

| Verifier | flat JSD | M=1 | M=3 | Δ(M=3−flat) | Δ(M=3−M=1) |
|---|---|---|---|---|---|
| traversal | 5.972 | 6.158 | 6.213 | +0.241 | +0.054 |
| BV | 5.892 | 5.923 | 6.157 | +0.264 | **+0.234** |
| naive | 5.848 | 5.859 | 5.952 | +0.104 | +0.093† |
| specinfer | 5.668 | 5.765 | 6.004 | **+0.336** | **+0.239** |
| spectr | 5.848 | 5.859 | 5.952 | +0.104 | +0.093† |
| khisti | 5.848 | 5.859 | 5.952 | +0.104 | +0.093† |
| GBV | 5.892 | 5.923 | 6.157 | +0.264 | **+0.234** |
| NSS | 4.224 | 4.373 | 4.355 | +0.131 | −0.018 |
| max | 5.848 | 5.859 | 5.952 | +0.104 | +0.093† |

† K_eval=1 collapse (§5) — naive/spectr/khisti/max give identical BE; four cells are one effective measurement, not independent.

### 4.2b M=3 full eval across K_eval=1..4 (n=100, math_eval) — complete

M=3 vs flat JSD, Δ = M=3 − flat, per verifier per K_eval. **M=3 beats flat JSD on every verifier at K_eval=1 and K_eval=2; at K_eval=3,4 it still wins on all but GBV-K3 (−0.040).**

| Verifier | Δ K=1 | Δ K=2 | Δ K=3 | Δ K=4 | M=3 raw across K (1→4) |
|---|---|---|---|---|---|
| traversal | +0.241 | +0.244 | **+0.322** | +0.232 | 6.213 / 6.163 / 6.216 / 5.894 |
| BV | +0.264 | +0.232 | **+0.431** | **+0.411** | 6.157 / 6.224 / 6.233 / **6.283** |
| GBV | +0.264 | +0.376 | −0.040 | +0.056 | 6.157 / 5.586 / 4.995 / 4.690 |
| naive | +0.104 | **+0.393** | +0.263 | +0.202 | 5.952 / 6.012 / 5.805 / 5.686 |
| specinfer | **+0.336** | **+0.351** | +0.016 | +0.021 | 6.004 / 5.330 / 4.795 / 4.600 |
| spectr | +0.104 | +0.261 | +0.157 | +0.148 | 5.952 / 5.620 / 5.377 / 5.241 |
| khisti | +0.104 | +0.079 | +0.187 | +0.068 | 5.952 / 5.242 / 4.607 / 4.452 |
| NSS | +0.131 | +0.087 | +0.010 | +0.095 | 4.355 / 4.106 / 3.855 / 3.744 |
| max | +0.104 | +0.054 | +0.025 | +0.205 | 5.952 / 5.260 / 4.996 / 4.873 |

### 4.2c Verifier–K alignment: does train-M match eval-K? (CONFIRMED for traversal/BV)

The sharpest single finding in the multi-K data. Compare each enrich checkpoint's Δ-vs-flat across K_eval:

| Verifier | M=1 Δ vs flat (K=1/2/3/4) | M=3 Δ vs flat (K=1/2/3/4) |
|---|---|---|
| traversal | +0.186 / +0.265 / **−0.041** / +0.171 | +0.241 / +0.244 / **+0.322** / +0.232 |
| BV | +0.031 / +0.156 / +0.268 / +0.265 | +0.264 / +0.232 / **+0.431** / **+0.411** |

- **M=1 traversal *loses to flat* at K_eval=3 (−0.041)** — the lone negative in the M=1 traversal row. **M=3 erases that dip** (+0.322 at K_eval=3) and posts its single best traversal Δ there. Directly: M=3 traversal at K_eval=3 = 6.216 vs M=1 = 5.853, a **+0.363** within-K gap. *Caveat (§4.4): the −0.041 itself is within the ±0.13 per-cell noise floor — do not lean on the dip being real; the load-bearing fact is the **+0.322 / +0.363**, which is ~2.5× the floor.*
- **BV: M=3 advantage grows with K_eval and peaks at K=3/4** (+0.431, +0.411); M=3 BV raw BE *increases monotonically* with K (6.157→6.283), while flat BV is flat-to-declining. M=3 > M=1 at every K for BV.

**Mechanism (the deployable claim):** training with M diverse teacher rollouts calibrates the draft across M parallel contexts. At inference, K_eval branches demand the draft be well-aligned on K simultaneous paths. M=1 calibrates one path → excellent at K_eval≤2, degrades when K_eval exceeds training diversity (traversal K=3 dip). M=3 calibrates three → sustains acceptance through K_eval=3,4 for the prefix/budget verifiers (traversal, BV) whose acceptance reward grows with tree width. **Prediction: set train-M ≥ target inference K_eval.** This matters for production tree-SD (K=3–8 branches): matched M gives higher acceptance at high branch counts without extra inference cost.

**Caveat — not universal.** The alignment is verifier-specific. specinfer is *anti-aligned*: M=3 gain concentrates at K=1,2 (+0.336, +0.351) and collapses at K=3,4 (+0.016, +0.021). specinfer's multi-candidate residual acceptance dilutes per-candidate gain as branch count rises, so coverage from enrich pays off most with few candidates. naive peaks at K=2; NSS is flat across K. The §5 functional analysis must explain why prefix/budget verifiers align with train-M while residual/OT verifiers do not.

### 4.3 Two empirical patterns (conjectures, not results)

**Pattern 1 — K_eval scaling of Δ(M=3−flat) is verifier-split:** prefix/budget verifiers (traversal, BV) keep or grow their Δ through K_eval=3,4; residual/competition verifiers (specinfer) lose it at high K; OT (NSS) stays flat-small. The growth subset (BV +0.264→+0.431, naive +0.104→+0.393 peaking at K=2) is the cleaner K_eval-scaling signal. Single-seed, n=100. **Do not state as established.**

**Pattern 2 — Coupling vs headroom are distinct (revised, see §3.2):** ρ(JSD,BE) and Δ(enrich−flat) do *not* track together. specinfer has the loosest coupling (ρ=−0.402) yet the largest K_eval=1 gain (+0.336); traversal has the tightest coupling (ρ=−0.645) and a moderate gain (+0.241). The M=3−M=1 column at K_eval=1 splits along verifier selectivity (BV/GBV/specinfer gain +0.23–0.24 from the extra rollouts; traversal +0.054; NSS flat) — selective verifiers extract more from added teacher diversity because they probe positions a single rollout never trains. **This cross-verifier story is the empirical headline — if it survives second seed and n=1000.** Do not claim it before then.

### 4.4 Statistics

**Noise / seed floor (this gates every cell-level claim).** One clean null comparison in [`Results/jsd_enrich_results.csv`](../Results/jsd_enrich_results.csv): **same method, different seed** (`jsd_mathhard_s456 − s123`, both plain flat JSD, both 8K) → mean Δ **+0.093**, std **0.123** across 35 cells. So **a single-cell BE difference below ≈0.12 is within seed noise, and a whole-grid +0.09 mean shift is what a seed swap alone produces.** Any individual cell in §4.2b/§4.2c — including the "M=1 traversal loses at K=3 (−0.041)" dip — must not be read alone; only grid-aggregate means and patterns that repeat across the bv/traversal block survive. *(Note: `block_eff` is an accept/reject ratio and is hardware-independent, so the A100↔H100 BE values are directly comparable; the +0.11 A100→H100 difference for `jsd_mathhard_s123` is **not** a hardware offset — it is the flat 8K→25K training gain, see below.)*

**Two enrich effects, both paired at matched 8K steps:**

| Comparison | overall mean Δ (n≈36) | bv/traversal mean Δ | verdict |
|---|---|---|---|
| M=1 enrich − flat jsd-8K [s123] | +0.097 | +0.163 | ≈ seed floor |
| M=1 enrich − flat jsd-8K [s456] | +0.037 | +0.092 | **below** floor |
| **M=3 enrich − flat jsd-8K [s123]** | **+0.182** | **+0.297** | ~2× floor |
| M=3 − M=1 enrich [s123] | +0.084 | +0.134 | ≈ floor |
| ttemp1.5 − ttemp1.0 (M=3 neg-control) | −0.079 | −0.068 | correctly negative |

**The decisive comparison — enrich-8K vs converged flat-25K [s123]** (`block_eff` is an accept/reject ratio, so it is hardware-independent and these values compare directly):

| Comparison | overall mean Δ | bv/trav Δ | traversal K=3 | reading |
|---|---|---|---|---|
| flat-25K − flat-8K | +0.109 | +0.189 | −0.024 | 3× longer flat training buys ~+0.11 |
| **M=3 enrich-8K − flat-25K** | **+0.074** | **+0.100** | **+0.346** | **M=3-8K beats *converged* flat** |
| M=1 enrich-8K − flat-25K | −0.022 | −0.044 | −0.016 | single-path enrich < flat-25K |

**Reading:**
1. **Multi-path is the mechanism, and it is not reproducible by training flat longer.** M=3 enrich at **8K** beats flat trained all the way to **25K** (its ceiling — flat converges ~15K, §4.5) by +0.074 overall / +0.10 on bv/traversal, peaking at **traversal K=3 = +0.346** (~3× the seed floor). This is the multi-path/verifier–K alignment signal (§4.2c) surviving against converged flat.
2. **Compute-matched, M=3 wins.** M=3-8K ≈ 24K teacher-rollouts vs flat-25K ≈ 25K rollouts — roughly equal compute — and M=3 still wins. The earlier worry (flat-25K traversal K=1=6.28 > M=3-8K K=1=6.213) is real *only at K=1* (saturated); the advantage lives at K≥2 where tree width matters.
3. **M=1 enrich = flat with fewer steps.** M=1-8K (8K rollouts) loses to flat-25K (−0.022); single-path enrich has no structural edge — more compute beats it. So the benefit is specifically **M>1**, not "enrich."
4. Negative control behaves (higher teacher temp hurts).

**Cross-seed status:** M=1 replicates directionally on both seeds (magnitude seed-sensitive, +0.097 vs +0.037). **M=3 has only s123 — the s456 M=3 run is now the single load-bearing experiment:** it must reproduce both the +0.30-vs-flat-8K *and* the +0.10/traversal-K=3=+0.35-vs-flat-25K multi-path effect.

A naive "29/36 wins, $p\approx10^{-6}$" binomial would be **invalid** here: the 36 cells are highly correlated (same prompts, same checkpoint, related verifiers, shared $K_{\text{eval}}$ grid), and the four $K_{\text{eval}}{=}1$ collapsed cells are duplicate counts. The independence assumption is false.

**Correct approach:**
- Pre-declare a small set of aggregate metrics (e.g. mean BE on traversal; mean BE on naive; one strict-verifier metric).
- Paired **prompt-level bootstrap / permutation** CIs (resample prompts, both checkpoints evaluated on the same resample).
- n=1000 to bring SE from ~0.05 to ~0.016 so the ±0.14 traversal effect and the borderline cells resolve.

**M=3 full eval result:** best val 6.420 (25-prompt val, SE≈0.15 — indistinguishable from M=1's 6.409). Full n=100 eval shows M=3 traversal BE=6.213 vs M=1=6.158 — a gap of +0.054, just above SE≈0.10 but not significant at n=100 alone. No M-scaling-in-training claim until second seed confirms.

### 4.5 Convergence / matched-compute: enrich vs converged flat

**The converged-flat eval now exists** and is in [`Results/jsd_enrich_results.csv`](../Results/jsd_enrich_results.csv) as `jsd_mathhard_s123_25k` (the 25K flat run, `train_steps=25000`; flat JSD converges ~15K so 25K is at/past ceiling). `block_eff` is hardware-independent, so its BE is directly comparable to the 8K A100 runs. The decisive deltas (full table in §4.4):

- **flat-25K − flat-8K = +0.109 overall** — 3× longer flat training is a real ~+0.11 gain (so flat-8K was genuinely under-trained; the comparison had to be made against converged flat).
- **M=3 enrich-8K − flat-25K = +0.074 overall, +0.100 bv/traversal, traversal K=3 = +0.346.** **M=3 enrich at 8K beats flat trained to its 25K ceiling**, concentrated on the prefix/budget verifiers and high K — exactly the §4.2c multi-path/verifier–K alignment signature.

  **As percentages, with the load-bearing nuance** (vs matched-step flat-8K | vs converged flat-25K):

  | cell | vs flat-8K | vs flat-25K |
  |---|---|---|
  | traversal K=2 | +4.1% | +0.4% |
  | **traversal K=3** | **+5.5%** | **+5.9%** |
  | traversal K=4 | +4.1% | **−0.8%** |
  | bv K=2 | +3.9% | +1.9% |
  | bv K=3 | +7.4% | +1.9% |
  | bv K=4 | +7.0% | +2.2% |

  **The bankable claim against converged flat is `traversal K=3 ≈ +6%`.** BV looks strong vs flat-8K (+7%) but **most of that is a convergence-speed effect** — flat's own 8K→25K gain is largest exactly on bv K=3 (+0.31), so vs converged flat the bv margin shrinks to ~+2%. So: **traversal K=3 is structural; BV is mostly "enrich converges faster."** Both are real wins, but only the traversal one is a ceiling improvement.

  **Inverted-V at K_eval = M (the sharpest alignment evidence, now that flat-25K K=4 exists).** Traversal Δ vs *converged* flat: **K=2 +0.4% → K=3 +5.9% → K=4 −0.8%.** The advantage over converged flat peaks exactly at K_eval = train-M = 3 and is gone by K=4 > M. naive shows the same shape (K=4 −1.1%). This is far stronger than "K=3 is good": it is a *falsifiable* prediction — an M=4 enrich run should move the peak to K=4. It also explains why earlier K=4-vs-flat-8K looked like a win (+4.1%): that was just flat being under-trained, not a real K=4 edge. **Headline claim should be stated as "enrich-M maximises BE at K_eval≈M," with M=3/traversal-K=3 as the worked instance.**
- **M=1 enrich-8K − flat-25K = −0.022.** Single-path enrich loses to converged flat → M=1 enrich is just flat with fewer steps, **no structural edge.**

**Compute accounting.** M=3-8K ≈ 24K teacher-rollouts ≈ flat-25K's ~25K rollouts → **roughly matched compute, and M=3 still wins.** The only place converged flat catches M=3 is **K_eval=1** (saturated single path: flat-25K traversal K=1=6.28 vs M=3-8K=6.213, −0.068); the M=3 advantage is entirely a **K≥2 tree-width effect**, which is the honest and defensible framing.

**What is now established vs still open:**
- *Established (s123):* the high-K verifier–K alignment (§4.2c) is **structural** — flat cannot replicate it by training 3× longer. Flat trains on one greedy path; more steps sharpen single-path matching but cannot manufacture multi-path calibration. Bet the paper on this.
- *Open (the one load-bearing run):* **M=3 s456.** Must reproduce M=3 enrich-8K > flat (both the 8K and 25K comparisons) on bv/traversal at K≥2. Until then the headline rests on a single seed.

**Convergence point (40K flat run):** flat JSD **converges ~15K** — val plateaus (5.74–6.42), `best`=6.608 frozen since ~15K, forgetting climbs 0→1.28 (post-convergence churn), loss at floor. Consequence: converged enrich runs need only ~15–20K with early stopping, not 40K.

**Checkpoint-selection caveat (methodology):** `best`-on-25-prompt-val is winner's-curse biased — max over hundreds of noisy evals (SE≈0.15–0.20), bias grows with run length. Reported BE tables are unaffected (n=100 offline re-evals of `ckpt_best`), but selection is noisy. Going forward: select by smoothed/EMA val, ≥100-prompt val less frequently, early-stop with patience (now implemented in `train.py`).

**Remaining decisive experiment:** **M=3 s456** (train 8K + n=100 eval) — confirm the multi-path effect replicates across seeds. Then, optionally, M=3-to-convergence to show whether the +0.10 bv/traversal margin over flat-25K holds or grows. Both rank **above** n=1000 / 2nd dataset.

---

## 5. Verifier-Level Math (open obligations)

**Core question — why does stochastic-teacher JSD help verifiers differently?** Each verifier's per-state acceptance probability is a *different functional* of $(P,Q)$. For naive single-token acceptance, $\alpha_{\text{naive}}(P,Q)=\sum_x\min(P(x),Q(x))=1-\text{TV}(P,Q)$, and JSD bounds TV ($\text{TV}^2\le 2\,\text{JSD}$ in nats), so lowering JSD on the trained states should raise naive acceptance on those states. For NSS (optimal-transport), BV/GBV (tree/budget), and SpecInfer, the acceptance functional is **not** TV, so the same JSD reduction maps to a *different* marginal acceptance gain — that mapping is the explanation for the differential cross-verifier benefit, and deriving it per verifier is the central theory contribution. Draft-OPD gives a local accepted/rejected KL rationale, but not this cross-verifier acceptance-functional analysis.

1. **$K_{\text{eval}}{=}1$ collapse — likely a one-line remark, not a theorem.** At $K_{\text{eval}}{=}1$, the verifier sees a single draft path. For sequential token verification up to depth $L$, naive/spectr/khisti/max plausibly reduce to the same per-token accept-w.p.-$\min(1,P/Q)$ rule along that path. The proof should state this for arbitrary $L$, not only $L=1$. If the reduction is trivial, state it as a remark for completeness — **do not present it as a contribution.** Empirically the four give identical BE for both checkpoints (5.848 / 5.859).
2. **Exact expected-BE for tree/OT verifiers.** For NSS/BV/GBV/traversal, either derive exact expected-BE formulas or state precisely why stochastic-teacher JSD is only a surrogate. Back this with **toy finite-vocabulary experiments** where acceptance and BE can be enumerated exactly and matched against simulation — this makes the verifier story hard to attack.
3. **Khisti antagonism.** Khisti is the only verifier with consistent negatives ($K_{\text{eval}}{=}2,4$). Needs a mechanism: what calibration pattern does stochastic-teacher JSD induce that khisti penalises?
4. **Acceptance–divergence transfer (§2.6)** beyond naive.
5. **Verifier–K alignment (§4.2c).** Why do prefix/budget verifiers (traversal, BV) keep or grow their enrich Δ as $K_{\text{eval}}\to M$ and beyond, while residual verifiers (specinfer) lose it at high $K_{\text{eval}}$? Conjecture: traversal/BV acceptance reward is monotone increasing in tree width given per-branch calibration, so M parallel-calibrated branches compound; specinfer's residual normalisation across candidates is sub-additive in branch count, so added branches dilute. Derive the $K_{\text{eval}}$-dependence of expected-BE per verifier and show it predicts the observed alignment/anti-alignment.
6. **Coupling vs headroom (§3.2).** Two separate per-verifier functionals: the JSD→acceptance *slope* (predicts ρ) and the acceptance *headroom* relative to a flat-JSD-trained draft (predicts Δ). These are empirically orthogonal (specinfer: low ρ, high Δ). A complete theory derives both from the verifier's acceptance functional.

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

The bottom-right corner — student-generated contexts with M teacher continuations per rollout — is the natural adaptive variant: the student's own failures determine where the teacher supervises. See §8 (Curriculum Enrichment).

**What Draft-OPD leaves genuinely open:** cross-verifier behaviour and theory for verifier-specific acceptance functionals. Their own future-work line — extending OPD to *approximate/lossy verification* — is adjacent to our verifier work, so we are not contradicting them, we are entering the gap they flagged.

**Defensible claims (narrow):**
- *"For verifier-conditioned speculative decoding, how acceptance-aware training transfers across verifier families is unexplored; we map it empirically (9 verifiers × eval-K) and explain the differential with a per-verifier acceptance-functional analysis."*
- *"Stochastic-teacher symmetric JSD is a low-complexity teacher-sampling alternative; we measure the gap to Draft-OPD-style replay under matched compute."*

**Non-defensible claims (do not pitch):** *"State distribution matters / train on inference-time states"* (DistillSpec + Draft-OPD own this) and *"a new training method for speculative decoding"* (too broad).

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
| 2 | n=1000 eval with paired bootstrap CIs | Resolve effect size |
| 3 | **Compute-matched** baseline (same wall-clock / teacher calls / tokens, not just steps) | Enrich does extra rollouts; step-matched is unfair to baseline |
| 4 | DistillSpec baseline | Closest on-policy prior work |
| 5 | **Draft-OPD-style replay ablation** (draft-gated accepted-only vs accepted+rejected) | The core novelty contrast |
| 6 | Greedy teacher (M=1) vs stochastic teacher (M=1) vs stochastic teacher (M=3) | Isolate stochasticity from multi-trajectory averaging |
| 7 | ≥1 more dataset (code / Spec-Bench mixed) | Generalisation beyond math |
| 8 | ≥1 more model pair | Method, not setup-specific quirk |
| 9 | ~~Temperature robustness — teacher_temp~~ | **Answered**: ttemp=1.5 run shows ρ=−0.543 vs default ρ=−0.645; higher teacher_temp causes training-inference mismatch. teacher_temp=1.0 confirmed. Eval temperature sweep (not training) still needed if running approximate verifiers. |
| 10 | Wall-clock tokens/sec + output quality/exactness | BE alone is insufficient |

Near-term order: finish M=3 eval → run s456 (flat + stochastic-teacher JSD) → n=1000 on best checkpoints → compute-matched greedy/stochastic/M ablations → Draft-OPD-style replay ablation.

---

## 8. Future Directions (parking lot — prioritise later)

- **NSS tree gradients (Rahul):** exact survival-weighted $\partial\text{BE}/\partial\theta$ — an **Axis-B** (objective) change, **orthogonal to enrich's Axis-A** (state-distribution) change; they compose rather than approximate each other. The exact gradient is the strongest unscooped asset (Draft-OPD has no theory). Not yet implemented; depth_weight (crude `depth × JSD` multiply, no gradient) is the only Axis-B experiment so far and is not expected to be promising.
- **NSS-depth and broader tree gradients / tree-depth ablations** (vary $L$, vary $M$).
- **Adaptive teacher curriculum:** soft vs hard accept; hard-prompt up-weighting by inverse BE (exclude teacher-uncertain prompts); teacher-temperature scheduling (warm→cool).
- **Adaptive teacher using tree depth** to decide where to guide the student.
- **Expected-depth survival weighting (§6.1):** weight per-position JSD by predicted marginal acceptance-length gain $E[\tau_V]$. A distinct Axis-B corner between Draft-OPD's $\gamma^{k-1}$ and the exact NSS gradient; **expected to roughly match, not clearly beat, M=3 stochastic-teacher JSD** (it reweights existing positions). Pursue **position-level only**; validate with a cheap short probe gated behind M=3 confirmation.
- **Curriculum Enrichment / adaptive state selection (*Curriculum Enrichment Distillation*):** the strongest version of `jsd_flat_enrich` is not uniform M-path sampling but selective enrichment — sample M teacher branches where the verifier rejects (acceptance low); use standard flat JSD where acceptance is high. Teacher compute is spent only where BE is weakest. This is conceptually closer to active learning / a "Socratic teacher" than a fixed loss: *train where the student needs it.* Logically prior to any fixed-M scale-up; if it works at M=3, the adaptive version should be strictly more compute-efficient.
- **Verifier-failure-targeted enrichment (unexplored corner of 2×2):** the student generates a draft → verifier rejects at position $\tau$ → teacher generates M alternative continuations from $\tau$. This is student-generated states × multiple teacher trajectories — the bottom-right corner of §6's 2×2. Combines on-policy state quality (the student's actual failures) with trajectory diversity (M teacher fixes per failure). The natural successor to fixed enrichment once the 0.6B/8B pair is confirmed.
- **Student uncertainty gating (entropy proxy):** where entropy($Q_\phi$) is high, reveal M teacher continuations; where it is low, use single-path JSD. Computationally cheaper than full per-step rejection sampling as a proxy for verifier-guided routing.
- **32B teacher** to test teacher-scale sensitivity (not required for the core 0.6B/8B claim).
- **Longer horizon ($L{=}16$).**

These are explicitly deferred. Whether they fold into this paper (as ablations) or a follow-on is a research-lead scope decision.

---

## 9. Scope Verdict

**Promising early signal; not yet conclusive — and given Draft-OPD, the defensible contribution sits in two places.** Draft-OPD tested one acceptance scheme with no theory, so two things remain genuinely open and are where the contribution lives:
1. **Cross-verifier behaviour + the per-verifier acceptance-functional math (§5)** — they punted this; it is the clearest open ground.
2. **The exact survival-weighted gradient (Rahul, Axis B)** — no theory in their paper; the strongest unscooped asset.

Current `jsd_flat_enrich` is best understood as **one simple stochastic-teacher corner of the 2-D design space (§6.1)**, not the headline. Publishable **only if** (a) framed as the cross-verifier analysis + design-space map rather than "accepted-state distillation works," (b) backed by §5 math, (c) supported by §7 matched-compute / multi-seed / multi-dataset evidence, and ideally (d) anchored on the exact gradient once implemented. As-is it is a strong internal / workshop-direction result. **The pivot decision (re-anchor headline on cross-verifier + exact gradient) is the research lead's to make** — and should not be taken until the M=3 eval lands and the empty design-space cells start to fill.
