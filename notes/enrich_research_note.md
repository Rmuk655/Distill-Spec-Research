# Stochastic Teacher Rollout JSD for Speculative Draft Training
## A Research Note on `jsd_flat_enrich`

**Status:** Early internal result — promising signal, **not yet paper-ready.** M=1 and M=3 training and evaluation complete (s123); second seed (s456) pending.
**Draft–Teacher pair:** Qwen3-0.6B draft / Qwen3-8B teacher
**Training data:** math_hard. **Eval data:** math_eval (n=100; paper target n=1000).
**Positioning:** Builds on [2602.16994]; closely related to DistillSpec, on-policy GKD, Online Speculative Decoding, and **Draft-OPD (May 2026)** — see §6 for the differentiator we must defend.

> **Honest framing.** The current evidence supports: *"stochastic teacher rollouts may improve flat JSD on this Qwen3 math setup."* It does **not** yet prove a general method. This note is scoped to make the claim defensible, identify what is novel vs. incremental, and list the experiments and proofs required before submission.

---

## 1. Motivation

Standard flat JSD training minimises

$$\mathcal{L}_{\text{JSD}} = \mathbb{E}_{x_{<t} \sim D}\big[\text{JSD}\big(P_\theta(\cdot \mid x_{<t}) \,\|\, Q_\phi(\cdot \mid x_{<t})\big)\big]$$

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
\text{JSD}\big(P_\theta(\cdot \mid p, y^{(m)}_{<t}) \,\|\, Q_\phi(\cdot \mid p, y^{(m)}_{<t})\big).$$

There are **no draft proposals, no verifier accept/reject decisions, and no teacher-corrected residuals** in the current flat-enrich code path. $M=1$ isolates greedy-vs-stochastic teacher training; $M>1$ adds multiple stochastic teacher trajectories per prompt and tests whether path diversity gives useful extra contexts.

### 2.3 Ideal extension: verifier-accepted state distribution

The stronger objective we originally wanted to approximate is the **verifier-induced prefix kernel** $\mathcal{J}_V$: sample draft proposals, verify them with $V$, keep the accepted prefix, resample the first rejected position from the teacher residual, and train on the resulting verified prefix. Formally, $x^* \sim \mathcal{J}_V(Q_\phi, P_\theta; \pi)$ and

$$\mathcal{L}_{\text{accepted-state}} =
\mathbb{E}_{x^* \sim \mathcal{J}_V}\!\left[\frac{1}{|x^*|}\sum_{t=1}^{|x^*|}
\text{JSD}\big(P_\theta(\cdot \mid p, x^*_{<t}) \,\|\, Q_\phi(\cdot \mid p, x^*_{<t})\big)\right].$$

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

L-relative buckets: easy $\geq 0.75L$, medium $[0.375L, 0.75L)$, hard $<0.375L$.

### 3.2 Diagnostics (`--diagnose`)

**Spearman $\rho$(divergence, BE):** rank correlation, robust to JSD range compression as the model improves (unlike Pearson r).
- Observed (K_eval=1, n=100): flat JSD $\rho=-0.396$; M=1 $\rho=-0.568$ ($p=8.5\times10^{-12}$); M=3 $\rho=-0.645$ ($p\approx0$). Monotonically strengthening objective–BE alignment with M (all traversal mode).
- Per-verifier ρ for M=3: traversal $\rho=-0.645$ ($p\approx0$); BV $\rho=-0.518$ ($p\approx2\times10^{-9}$). σ(JSD)=0.0200 identical for both — σ is stable, so the ρ difference is real, not range restriction. Traversal (permissive, longest-prefix acceptance) has tighter JSD→BE coupling than BV (budget-limited tree). **This per-verifier ρ hierarchy is the empirical anchor for the §5 acceptance-functional theory.**

**$\sigma$(JSD) stability (Case A vs B):** Case A (σ stable, ρ↑) = true signal; Case B (σ collapses, ρ stable) = range restriction. M=3 σ(JSD)=0.0200, mean_JSD=0.0296 across both diagnose runs — σ not collapsed ⇒ **Case A** confirmed.

### 3.3 Capacity signals (measured along the way)

**Student (draft) at capacity:** val/block_eff plateau with no new best; train loss floor (~0.02); high oscillating forgetting with no net BE gain; frozen BE bucket distribution.

**Teacher / data diversity signal:** BE → $L$ suggests limited headroom; path_diversity → 0 means stochastic teacher rollouts have collapsed to near-identical continuations. At 0.6B/8B on math, BE≈6.4/8 and path_diversity ∈ [0.8,1.0] for M=3, so the teacher is still producing diverse contexts. This does **not** prove the teacher is not a bottleneck; it only says stochastic teacher sampling has not collapsed. A 32B-teacher run would test teacher-scale sensitivity (future work, §8).

**`train/path_diversity`** = fraction of positions where ≥2 of the $M$ rollouts disagree. ~1.0 ⇒ diverse signal, $M>1$ contributes; <0.1 ⇒ rollout collapse, $M>1$ ≈ $M=1$. Observed M=3: ∈[0.8,1.0], healthy.

**`val/forgetting`** = backward-transfer loss (Σ max(0, best_historical_BE(p) − current_BE(p))). Oscillating (not monotonic) ⇒ stability–plasticity churn, not catastrophic forgetting; `ckpt_best` captures the peak.

---

## 4. Results (encouraging but inconclusive)

### 4.1 Training runs

| Run | Steps | Best val BE | Status |
|---|---|---|---|
| `jsd_mathhard_s123` (flat JSD) | 8K | 6.01 | Complete |
| `jsd_flat_enrich_M1_s123` | 8K | 6.409 | Complete |
| `jsd_flat_enrich_M3_s123` | 8K | 6.420 | Complete; eval done |
| `jsd_flat_enrich_K3_mathhard_s123_ttemp1.5` | 8K | — | Complete (teacher_temp=1.5 variant); K=1 traversal/BV: 6.128/6.145 vs default 6.213/6.157 — slight decline, within noise; preliminary no-benefit signal |
| `…_M1_s456`, `jsd_mathhard_s456` | 8K | — | Pending (second seed) |

### 4.2 Three-checkpoint comparison at K_eval=1 (n=100, math_eval)

Block efficiency for all three checkpoints at K_eval=1, L=8. **M=3 beats flat JSD on all 9 verifiers — no exceptions.** Monotonic improvement on 8/9 verifiers (NSS: M=3 ≈ M=1).
**Caveat: single-seed, n=100 point estimates, not significance-tested.** See §4.4.

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

### 4.2b M=3 K_eval=2 results (n=100, math_eval)

M=3 beats flat JSD on all 9 verifiers at K_eval=2 — the all-positive pattern from K_eval=1 holds.

| Verifier | flat K=2 | M=1 K=2 | M=3 K=2 | Δ(M=3−flat) | Δ(M=3−M=1) |
|---|---|---|---|---|---|
| traversal | 5.920 | 6.185 | 6.163 | +0.243 | −0.022 |
| BV | 5.992 | 6.148 | 6.224 | +0.232 | +0.076 |
| naive | 5.619 | 5.684 | 6.012 | **+0.393** | +0.328 |
| specinfer | 4.979 | 4.994 | 5.330 | **+0.351** | +0.336 |
| spectr | 5.359 | 5.729 | 5.620 | +0.261 | −0.109 |
| khisti | 5.163 | 5.028 | 5.242 | +0.079 | +0.214 |
| GBV | 5.210 | 5.537 | 5.586 | **+0.376** | +0.049 |
| NSS | 4.019 | 3.988 | 4.106 | +0.087 | +0.118 |
| max | 5.205 | 5.136 | 5.260 | +0.055 | +0.124 |

Δ(M=3−flat) growth from K=1 to K=2: naive +0.104→+0.393, GBV +0.264→+0.376, specinfer +0.336→+0.351 (stable). Traversal/BV stable. NSS/max/khisti slight decline. K_eval=3,4 pending.

### 4.2c M=1 vs flat JSD across K_eval=1..4 (n=100, math_eval; M=3 K_eval=3,4 pending)

Δ = enrich(M=1) − flat JSD per K_eval. Single seed; M=3 multi-K eval not yet run beyond K=2.

| Verifier | $K_{\text{eval}}{=}1$ | $2$ | $3$ | $4$ | Pattern |
|---|---|---|---|---|---|
| traversal | +0.186 | +0.265 | **−0.041** | +0.171 | Mostly positive, one negative |
| BV | +0.031 | +0.156 | +0.268 | +0.265 | Grows with $K_{\text{eval}}$ |
| naive | +0.011† | +0.065 | +0.135 | +0.324 | Grows with $K_{\text{eval}}$ |
| specinfer | +0.097 | +0.015 | +0.136 | +0.158 | Weakly up |
| spectr | +0.011† | +0.370 | +0.052 | +0.224 | Noisy positive |
| NSS | +0.149 | −0.030 | +0.103 | +0.095 | Flat positive |
| GBV | +0.031 | +0.327 | **−0.113** | −0.015 | Peaks then vanishes |
| khisti | +0.011† | **−0.135** | +0.189 | **−0.161** | Alternating / negative |
| max | +0.011† | −0.069 | +0.056 | +0.151 | Weakly positive |

† K_eval=1 collapse — not independent evidence.

### 4.3 Two empirical patterns (conjectures, not results)

**Pattern 1 — K_eval scaling of Δ(M−flat):** at M=1, naive and BV gaps grow with K_eval (exceptions: traversal K=3 negative; GBV reverses at 3,4; khisti alternating). The signal is cleaner at M=3 (K=2 data, §4.2b): naive +0.104→+0.393 (+0.289 growth); GBV +0.264→+0.376; specinfer +0.336→+0.351 (stable). NSS/max/khisti decline slightly at M=3 K=2 — the declining subset needs a mechanism (NSS OT-global, max envelope, khisti pairwise). **Growing-Δ subset (naive, GBV, specinfer) is the cleaner K_eval-scaling signal.** Single-seed, n=100; M=3 K=3,4 pending. **Do not state as established.**

**Pattern 2 — Selective verifiers gain more from M (new, K_eval=1):** The M=3 vs M=1 column in §4.2 splits clearly along verifier selectivity: BV/GBV/SpecInfer gain +0.23–0.24, while traversal gains only +0.054 and NSS is flat (−0.018, within SE≈0.10). One interpretation: verifiers with tighter intermediate-position acceptance criteria (BV/GBV require budget-limited token trees; SpecInfer uses multi-token speculation) expose misalignment at positions a single stochastic teacher trajectory never trains; M=3 adds diverse teacher contexts that cover those positions. Traversal (most permissive, accepts any length path) and NSS (global OT criterion, already partially addressed at M=1) are less sensitive to this. **This is the cross-verifier story — if it survives second seed and n=1000, it is the empirical headline.** Do not claim it before then.

### 4.4 Statistics

A naive "29/36 wins, $p\approx10^{-6}$" binomial would be **invalid** here: the 36 cells are highly correlated (same prompts, same checkpoint, related verifiers, shared $K_{\text{eval}}$ grid), and the four $K_{\text{eval}}{=}1$ collapsed cells are duplicate counts. The independence assumption is false.

**Correct approach:**
- Pre-declare a small set of aggregate metrics (e.g. mean BE on traversal; mean BE on naive; one strict-verifier metric).
- Paired **prompt-level bootstrap / permutation** CIs (resample prompts, both checkpoints evaluated on the same resample).
- n=1000 to bring SE from ~0.05 to ~0.016 so the ±0.14 traversal effect and the borderline cells resolve.

**M=3 full eval result:** best val 6.420 (25-prompt val, SE≈0.15 — indistinguishable from M=1's 6.409). Full n=100 eval shows M=3 traversal BE=6.213 vs M=1=6.158 — a gap of +0.054, just above SE≈0.10 but not significant at n=100 alone. No M-scaling-in-training claim until second seed confirms.

---

## 5. Verifier-Level Math (open obligations)

**Core question — why does stochastic-teacher JSD help verifiers differently?** Each verifier's per-state acceptance probability is a *different functional* of $(P,Q)$. For naive single-token acceptance, $\alpha_{\text{naive}}(P,Q)=\sum_x\min(P(x),Q(x))=1-\text{TV}(P,Q)$, and JSD bounds TV ($\text{TV}^2\le 2\,\text{JSD}$ in nats), so lowering JSD on the trained states should raise naive acceptance on those states. For NSS (optimal-transport), BV/GBV (tree/budget), and SpecInfer, the acceptance functional is **not** TV, so the same JSD reduction maps to a *different* marginal acceptance gain — that mapping is the explanation for the differential cross-verifier benefit, and deriving it per verifier is the central theory contribution. Draft-OPD gives a local accepted/rejected KL rationale, but not this cross-verifier acceptance-functional analysis.

1. **$K_{\text{eval}}{=}1$ collapse — likely a one-line remark, not a theorem.** At $K_{\text{eval}}{=}1$, the verifier sees a single draft path. For sequential token verification up to depth $L$, naive/spectr/khisti/max plausibly reduce to the same per-token accept-w.p.-$\min(1,P/Q)$ rule along that path. The proof should state this for arbitrary $L$, not only $L=1$. If the reduction is trivial, state it as a remark for completeness — **do not present it as a contribution.** Empirically the four give identical BE for both checkpoints (5.848 / 5.859).
2. **Exact expected-BE for tree/OT verifiers.** For NSS/BV/GBV/traversal, either derive exact expected-BE formulas or state precisely why stochastic-teacher JSD is only a surrogate. Back this with **toy finite-vocabulary experiments** where acceptance and BE can be enumerated exactly and matched against simulation — this makes the verifier story hard to attack.
3. **Khisti antagonism.** Khisti is the only verifier with consistent negatives ($K_{\text{eval}}{=}2,4$). Needs a mechanism: what calibration pattern does stochastic-teacher JSD induce that khisti penalises?
4. **Acceptance–divergence transfer (§2.6)** beyond naive.

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
| 9 | Temperature robustness (esp. eval temp=1.0 and teacher_temp sweep) | Temperature sensitivity is known in SD/KD; current default teacher_temp is 1.0 |
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
