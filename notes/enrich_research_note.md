# Enriched Speculative Decoding Training
## A Research Note on `jsd_flat_enrich`

**Status:** K=1 complete and evaluated; K=3 training complete, evaluation in progress (2026-06-23)  
**Draft–Teacher pair:** Qwen3-0.6B draft / Qwen3-8B teacher  
**Training data:** math_hard (hard math prompts)  
**Eval data:** math_eval (100 prompts; final paper target: 1000)  
**Builds on:** [2602.16994] — extends the training objective to close the distribution gap between draft training and inference-time acceptance.

---

## 1. Motivation

Standard flat JSD training minimises:

$$\mathcal{L}_{\text{JSD}} = \text{JSD}(P_\theta(\cdot \mid x_{<t}) \;\|\; Q_\phi(\cdot \mid x_{<t}))$$

where $P_\theta$ is the teacher and $Q_\phi$ is the draft, evaluated at positions $x_{<t}$ drawn from the **teacher's marginal distribution**. This creates a distribution mismatch: at inference time, the draft operates in a stochastic acceptance loop where its mistakes change the context it conditions on. The training distribution (teacher greedy/sampled) is systematically cleaner than the inference distribution.

**Hypothesis:** Training on sequences drawn from the actual acceptance distribution $\mathcal{J}_V(Q_\phi, P_\theta)$ — the joint distribution the draft sees during speculative decoding — will reduce this gap and improve block efficiency.

---

## 2. Method: `jsd_flat_enrich`

At each training step, instead of drawing $x_{<t}$ from $P_\theta$, we run $K$ steps of stochastic speculative decoding to obtain an **enriched context** $x^*$:

**Algorithm (K=1 enrichment, traversal verifier):**
1. Sample a prompt $p$ from training set.
2. Draft proposes $L$ tokens: $x^{(0)} \sim Q_\phi(\cdot \mid p)$ (tree of depth $L$).
3. Teacher evaluates each token $x_t^{(0)}$: accept with probability $\min\!\left(1,\, \frac{P_\theta(x_t \mid p, x^*_{<t})}{Q_\phi(x_t \mid p, x^*_{<t})}\right)$ (naive rule); other verifiers (BV, NSS, SpecInfer) apply their respective criteria.
4. At the first rejection, resample from the teacher-corrected distribution. Concatenate accepted prefix to get $x^*$.
5. Compute training loss at accepted positions:

$$\mathcal{L}_{\text{enrich}} = \frac{1}{|x^*|}\sum_{t=1}^{|x^*|} \text{JSD}\!\left(P_\theta(\cdot \mid p, x^*_{<t}) \;\Big\|\; Q_\phi(\cdot \mid p, x^*_{<t})\right)$$

For $K > 1$: repeat steps 2–4 for $K$ independent rollouts, then average the loss across rollouts. Higher $K$ provides a richer Monte Carlo estimate of the acceptance distribution and greater diversity of training contexts.

**Key distinction from flat JSD:** $x^*$ is a sample from $\mathcal{J}_V(Q_\phi, P_\theta)$, the distribution the draft actually encounters at inference time. Flat JSD trains on teacher-sampled sequences; enrich trains on verifier-accepted sequences.

### 2.1 Training health signals

**`train/path_diversity`** = fraction of token positions at which at least two of the $K$ rollouts differ:

$$\text{pathdiv} = \frac{1}{L} \sum_{t=1}^{L} \mathbf{1}\!\left[\exists\, i \neq j : x_t^{(i)} \neq x_t^{(j)}\right]$$

- pathdiv $\approx 1.0$: teacher's stochastic sampling produces genuinely diverse accepted sequences; $K > 1$ contributes real additional signal.
- pathdiv $< 0.1$: rollout collapse — draft and teacher agree almost everywhere; $K > 1$ reduces to $K = 1$ effectively.
- Observed in K=3 run: pathdiv $\in [0.8, 1.0]$ throughout, with occasional batch dips to $0.5$–$0.6$. **Healthy.**

**`val/forgetting`** = backward transfer loss: how much block efficiency the model has lost on prompts it previously mastered:

$$\text{forget}_t = \sum_{p \in \mathcal{P}} \max\!\left(0,\; \hat{b}^*(p) - b_t(p)\right)$$

where $\hat{b}^*(p)$ is the best historical block efficiency for prompt $p$, and $b_t(p)$ is the current block efficiency. Oscillating forgetting (not monotonically increasing) indicates stability–plasticity tradeoff rather than catastrophic forgetting. The `ckpt_best` mechanism captures the peak before any forgetting-driven regression.

---

## 3. Measurement Framework

### 3.1 Primary metric: Block Efficiency

$$\text{BE} = \frac{\text{generated tokens}}{\text{target model calls}}, \quad \text{max BE} = L$$

At $L = 8$: BE = 8 is the theoretical maximum (draft accepts all tokens with no teacher calls beyond the initial ones). Observed range: 3.7 (NSS, strictest) to 6.4 (traversal, current best).

**L-relative bucketing (verifier-agnostic):**
- Easy: $\text{BE} \geq 0.75L$
- Medium: $0.375L \leq \text{BE} < 0.75L$
- Hard: $\text{BE} < 0.375L$

### 3.2 Diagnostic signals (`--diagnose` mode)

**Spearman $\rho$(divergence, BE):** rank correlation between per-prompt JSD/fwd-KL and per-prompt block efficiency.
- $\rho < -0.3$, $p < 0.01$: H0 ruled out — the training objective is tracking BE.
- $\rho > -0.1$: objective mismatch — loss minimisation is not moving BE.
- **Observed:** flat JSD $\rho = -0.396$; enrich K=1 $\rho = -0.568$ ($p = 8.5 \times 10^{-12}$). Enrich exhibits stronger alignment between JSD and BE — a mechanistic signal that it is training on a more informative distribution.

**$\sigma$(JSD) stability (Case A vs. B):**
- Case A: $\sigma$ stable + $\rho$ increases → true signal improvement (more calibrated on hard prompts).
- Case B: $\sigma$ collapses + $\rho$ stable → range restriction (model uniformly better, Pearson r fooled).
- **Observed:** $\sigma$(JSD) identical for flat and enrich checkpoints → **Case A confirmed.** The improvement is real, not an artefact of range compression.

### 3.3 Verifier taxonomy and K=1 collapse

| Family | Members | K=1 behaviour |
|---|---|---|
| Single-token threshold | naive, spectr, khisti, max | **Collapse** — all identical at K=1 |
| Tree acceptance | traversal, BV, GBV | K-sensitive; tree structure used |
| Optimal transport | NSS | Strictest; independent of collapse |
| SpecInfer-style | specinfer | Intermediate |

**K=1 verifier collapse (empirically confirmed):** At $K=1$ (single-token acceptance), naive / spectr / khisti / max all produce identical BE for both checkpoints (flat JSD: 5.848; enrich: 5.859, $\Delta = +0.011$, within noise). This confirms the theoretical prediction that all threshold-based single-token verifiers reduce to the same rule at $K=1$: accept token $x$ with probability $\min(1, P(x)/Q(x))$. At $K \geq 2$, the verifiers diverge and the enrich advantage emerges.

*Requires formal mathematical proof — open item.*

### 3.4 Capacity signals

**Student (draft) at capacity:**
- `val/block_eff` plateau over many checkpoints with no new best.
- Training loss floor reached ($\sim 0.02$ for this setup).
- High oscillating forgetting with no net BE gain — model is redistributing rather than growing.
- BE bucket distribution frozen (no shift from hard → medium → easy).

**Teacher at capacity (not bottlenecking):**
- BE approaches $L$ (current best: $6.42/8 = 80\%$; teacher has room to offer).
- `train/path_diversity` collapses to near 0 (draft matches teacher everywhere) — *not observed*.
- At 0.6B draft / 8B teacher on math: teacher is **not** at capacity. Draft architecture is the binding constraint.
- To test teacher scale: 32B teacher experiment (see §7 Future Directions).

---

## 4. Results

### 4.1 Training runs

| Run | Steps | Best val BE | Status |
|---|---|---|---|
| `jsd_mathhard_s123` (flat JSD) | 8 K | 6.01 | Complete |
| `jsd_flat_enrich_K1_mathhard_s123` | 8 K | 6.409 | Complete |
| `jsd_flat_enrich_K3_mathhard_s123` | 8 K | 6.420 | Complete; eval in progress |
| `jsd_flat_enrich_K1_mathhard_s456` | 8 K | — | Pending (second seed) |
| `jsd_mathhard_s456` | 8 K | — | Pending (second seed baseline) |

### 4.2 K=1 enrich vs. flat JSD (n=100, math_eval)

Delta = enrich K=1 $-$ flat JSD. Bold = $|\Delta| > 0.10$ (estimated significance at paired $n=100$).

| Verifier | K=1 | K=2 | K=3 | K=4 | Pattern |
|---|---|---|---|---|---|
| traversal | **+0.186** | **+0.265** | −0.041 | **+0.171** | Flat positive |
| BV | +0.031 | **+0.156** | **+0.268** | **+0.265** | **Grows with K** |
| naive | +0.011† | +0.065 | **+0.135** | **+0.324** | **Grows with K** |
| specinfer | +0.097 | +0.015 | **+0.136** | **+0.158** | Weakly up |
| spectr | +0.011† | **+0.370** | +0.052 | **+0.224** | Noisy positive |
| NSS | **+0.149** | −0.030 | +0.103 | +0.095 | Flat positive |
| GBV | +0.031 | **+0.327** | −0.113 | −0.015 | Peaks K=2 |
| khisti | +0.011† | **−0.135** | **+0.189** | **−0.161** | Alternating |
| max | +0.011† | −0.069 | +0.056 | **+0.151** | Weakly positive |

† K=1 collapse — these four verifiers are theoretically equivalent at $K=1$.

**Win rate:** 29/36 positive deltas (80.6%). Under the null, $P(\geq 29\,|\,n=36) \approx 10^{-6}$ (binomial). Direction is established.

**Stat note:** Runs are paired (same 100 prompts, both checkpoints). Estimated $\sigma_\Delta \approx 0.5$ per prompt $\Rightarrow$ SE $\approx 0.05$ for $n=100$; $2\sigma$ threshold $\approx 0.10$. Results with $|\Delta| < 0.05$ are noise-level.

### 4.3 K=3 enrich (preliminary)

Training best val BE: **6.420** (step 7500/8000), marginally above K=1 (6.409) — within 25-prompt val noise. Full eval across all verifiers $\times$ K=1..4 in progress. *This table will be filled once eval completes.*

| Verifier | K=1 | K=2 | K=3 | K=4 |
|---|---|---|---|---|
| traversal | 6.213 | — | — | — |
| BV | 6.157 | — | — | — |
| naive | (running) | — | — | — |
| … | … | … | … | … |

---

## 5. Open Questions for Mathematical Verification

1. **K=1 collapse proof:** Formally show that naive, spectr, khisti, and max acceptance criteria all reduce to $\min(1, P/Q)$ comparison per token at $K=1$, $L=1$. The empirical confirmation (four verifiers give identical BE for both checkpoints) is strong, but the proof would clarify boundary conditions.

2. **Gap-grows-with-K conjecture:** Explain why a model trained on $K=1$ enrich shows monotonically growing advantage at higher eval-$K$ for naive and BV. Likely connected to: the $K=1$ acceptance distribution having heavier tails than teacher-greedy sampling, so the draft learns to handle suboptimal token choices more gracefully — generalising to the multi-step correction setting even when only trained on single-step rollouts.

3. **NSS strict advantage:** At $K=1$, enrich shows larger absolute gain under NSS (+0.149) than under naive (+0.011). Why? NSS uses optimal-transport acceptance, meaning only high-quality drafts pass. Enrich may produce a higher fraction of "clean" accepted sequences that NSS approves of, compared to teacher-sampled training contexts.

4. **Khisti antagonism:** Khisti shows clearly negative deltas at $K=2$ ($-0.135$) and $K=4$ ($-0.161$). This is the only verifier with consistent negatives. Hypothesis: Khisti's acceptance criterion penalises a specific calibration pattern that enrich training inadvertently introduces. Needs investigation of how khisti's function differs from naive at $K=2$.

5. **GBV peak at K=2:** GBV shows the largest single-cell gain (+0.327 at K=2) but no benefit at K=3,4. Why is GBV uniquely sensitive at exactly K=2?

---

## 6. Next Steps (Near Term)

| Action | Rationale |
|---|---|
| Full eval of K=3 enrich (in progress) | Determine if enrichment K scales: K=3 > K=1 |
| Second seed (`s456`) for K=1 and K=3 | Reproducibility; required for any paper claim |
| $n=1000$ eval on best checkpoint | Reduce SE from 0.05 to 0.016; borderline results will resolve |
| Extend K=3 to 12–15K steps | Best checkpoint appeared at 93% of training; model may still be improving |
| `--diagnose` on K=3 checkpoint | Check if $\rho$ further strengthens vs K=1 ($-0.568$) |
| NSS tree gradient integration | Rahul's exact $\partial\text{BE}/\partial\theta$ via survival-weighted NSS; ablation vs enrich |
| Spec-Bench evaluation (480 prompts, 6 domains) | Cross-domain generalisation test |

---

## 7. Future Directions

**Scaling:**
- **32B teacher:** Current 8B teacher is not at capacity on math (BE = 6.42/8 = 80%). Upgrading to 32B would test whether teacher scale adds information for a fixed 0.6B draft. Likely a future-work item — not needed to establish the enrich claim for the 0.6B/8B pair.

**Training distribution:**
- **Adaptive curriculum (hard-prompt up-weighting):** Weight training prompts by inverse BE during rollout — easy prompts (high BE) already have low loss, hard prompts (low BE) have the most signal. Deferred pending K=3 confirmation.
- **Teacher temperature scheduling:** `teacher_temp=1.5` creates more diverse rollouts (higher pathdiv) at the cost of lower-quality accepted sequences. `teacher_temp=1.0` is the current default; adaptive scheduling (warm → cool) may combine both benefits.
- **Soft teacher curriculum:** Instead of binary accept/reject, weight gradient by teacher confidence; allows gradient through "near-accepts." Requires changes to the acceptance sampling logic.

**Gradient signal:**
- **NSS tree gradients:** Exact survival-weighted $\partial\text{BE}/\partial\theta$ from Rahul. Theoretically the gold standard; enrich is a Monte Carlo approximation of this. Direct comparison will quantify how much the distribution sampling approach recovers.
- **Tree depth experiments:** K and L are both training hyperparameters (K controls enrichment depth, L controls tree width). Current: K=1/3, L=8. Ablation: fix K, vary L; fix L, vary K.
- **Adaptive tree depth:** Use per-prompt BE to dynamically choose L during training. Easy prompts use smaller L (less compute); hard prompts use larger L (more signal).

**Evaluation:**
- **Multi-domain (Spec-Bench):** 480 prompts across 6 categories (math, code, QA, translation, summarisation, chat). Needed to establish that enrich generalises beyond math.
- **Longer drafts (L=16):** Does enrich advantage scale with L?

---

## 8. Scope Note

`jsd_flat_enrich` (this note) is scoped as a potential standalone paper: the distribution-mismatch motivation, the enrichment mechanism, and the empirical win across 9 verifiers are a self-contained contribution. The NSS tree gradients, depth experiments, and curriculum variants are candidates for either inclusion (as ablations showing enrich is one rung of a hierarchy) or a follow-on paper.

---

*For questions or to request raw eval CSVs, contact: rkrishna@adobe.com*
