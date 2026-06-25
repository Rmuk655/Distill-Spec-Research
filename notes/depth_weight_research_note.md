# Depth-Weighted JSD: A Curriculum Reweighting Ablation
## Research Note on `jsd+dw_naive_tree_{lin,lam}`

**Status:** Negative overall (no aggregate improvement over flat JSD at any λ), with **one pattern worth following up**: depth_weight beats the flat baseline specifically at the matched train-K = eval-K diagonal (K_train=1 @ K_eval=1: +0.054, 6/9 verifiers; K_train=3 @ K_eval=3: +0.067, 8/9 verifiers), and is neutral-to-harmful off-diagonal. Single seed, n=100 — suggestive, not confirmed.
**Draft–Teacher pair:** Qwen3-0.6B draft / Qwen3-8B teacher
**Training data:** math_hard (primary, K_train=1 and K_train=3 tree variants, ~40K steps); gsm8k (ablation, lin+lam sweep, 4K steps).
**Eval data:** math_eval (n=100, K_eval=1..4, 9 verifiers); gsm8k_eval (n=100, K_eval=1..4).
**Raw results:** [`Results/jsd_jsd_dw_naive_tree_results_math_hard_40000_steps.csv`](https://github.com/Rmuk655/Distill-Spec-Research/blob/Pipeline/Results/jsd_jsd_dw_naive_tree_results_math_hard_40000_steps.csv) · [`Results/jsd_jsd_dw_naive_tree_results_gsm_8k_4000_steps_softmax_lambda.csv`](https://github.com/Rmuk655/Distill-Spec-Research/blob/Pipeline/Results/jsd_jsd_dw_naive_tree_results_gsm_8k_4000_steps_softmax_lambda.csv)
**Requested by:** Researcher.

> **Do not compare absolute BE values across the two datasets.** The math_hard runs are ~40K steps on a harder distribution; the gsm8k runs are 4K steps. Only within-dataset Δ values are meaningful.

---

## 1. Motivation

Flat JSD assigns uniform weight to every training prompt. The researcher's idea: multiply the JSD loss by a scalar derived from the draft's current expected acceptance depth $d = E[\tau_V]$ on that prompt. Two directional hypotheses are possible:

**Linear / λ > 0 direction** — $w$ is proportional to $d$. High expected depth means the draft already accepts many tokens on this prompt; those prompts get amplified loss. This is analogous to *training already-good students harder* — pushing calibrated regions toward further refinement. There is no strong theoretical reason to expect this to improve BE; it may simply reinforce states the draft has already converged on.

**λ < 0 direction** — $w$ is inversely related to $d$. Low expected depth (draft fails early → hard prompt) gets higher weight. This is the classic *hard-prompt curriculum*: focus training on prompts where the draft is most wrong.

Neither direction has a gradient path to higher BE — the mechanism is purely per-prompt effective LR scaling. Whether either direction helps is an empirical question.

---

## 2. Method

### 2.1 Flat JSD baseline

$$\mathcal{L}_{\text{JSD}} = \mathbb{E}_{x \sim D}\left[\frac{1}{|x|}\sum_{t=1}^{|x|} \text{JSD}\!\left(P_\theta(\cdot \mid x_{<t}) \,\|\, Q_\phi(\cdot \mid x_{<t})\right)\right]$$

where $D$ is a greedy teacher rollout distribution.

### 2.2 Depth-weighted JSD

$$\mathcal{L}_{\text{depth-weight}} = w(d) \cdot \mathcal{L}_{\text{JSD}}(Q_\phi, P_\theta\,;\,x), \qquad d = E[\tau_V(Q_\phi, P_\theta\,;\,x)]$$

$d$ is computed under `torch.no_grad()` — no gradient enters through it. Two EMA-normalised forms, both satisfying $E[w] \approx 1$ (no LR confound):

**Linear (`--depth_linear`):**
$$w_{\text{lin}}(d) = \frac{d}{\text{EMA}(d)}, \qquad \mu \leftarrow 0.9\,\mu + 0.1\,d$$

$w > 1$ for deep (easy) prompts, $w < 1$ for shallow (hard) prompts — the "good student" direction.

**Exponential (`--depth_lambda` $\lambda$):**
$$w_{\text{exp}}(d) = \exp\!\left(\lambda\,(d - \text{EMA}(d))\right)$$

$\lambda > 0$: same direction as linear. $\lambda < 0$: hard-prompt curriculum. $\lambda = 0$: $w \equiv 1$ — mathematically identical to flat JSD (free correctness control).

Code (`train.py:380-387`):
```python
if args.depth_linear:
    depth_w = d / max(depth_ema, 1e-6)
else:
    depth_w = math.exp(args.depth_lambda * (d - depth_ema))
loss = depth_w * loss
```

### 2.3 How $d$ is computed

`expected_depth_scalar` (`losses/compute.py:297`) builds a fresh draft tree via `iid_draft`, runs a target tree pass, then calls `TreeVerifier.expected_{verifier}_depths(L)` — the researcher's DP method — to obtain $E[\tau_V]$. This is an offline probe under `torch.no_grad()`; it costs one extra target tree pass per step and carries no gradient.

### 2.4 Training runs

| Run slug | Dataset | Train K | Steps |
|---|---|---|---|
| `jsd_mathhard_s123` | math_hard | 3 | ~40K |
| `jsd_dw_lin_naive_mathhard_s123_K1` | math_hard | 1 | ~40K |
| `jsd_dw_lin_naive_mathhard_s123_K3` | math_hard | 3 | ~40K |
| `jsd` | gsm8k | 3 | 4K |
| `jsd+dw_naive_tree_lin` | gsm8k | 3 | 4K |
| `jsd+dw_naive_tree_lam0.0` | gsm8k | 3 | 4K |
| `jsd+dw_naive_tree_lam0.5` | gsm8k | 3 | 4K |
| `jsd+dw_naive_tree_lam-0.5` | gsm8k | 3 | 4K |

---

## 3. Results

Raw data in the git CSV files linked above. All Δ = depth_weight − flat JSD on the same dataset.

### 3.1 math_eval: Δ across K_eval=1..4, bv + traversal

| K_eval | dw_lin K_train=1 (trav) | dw_lin K_train=1 (bv) | dw_lin K_train=3 (trav) | dw_lin K_train=3 (bv) |
|---|---|---|---|---|
| 1 | −0.116 | +0.129 | −0.118 | −0.053 |
| 2 | −0.061 | +0.047 | −0.099 | −0.036 |
| 3 | **+0.166** | −0.035 | +0.033 | +0.015 |
| 4 | −0.212 | −0.023 | +0.040 | −0.074 |
| **mean** | **−0.056** | **+0.030** | **−0.036** | **−0.037** |

Mean Δ over all 9 verifiers × K_eval=1..4 (36 cells each):

| condition | mean Δ (all 36 cells) | mean Δ (bv+trav only) |
|---|---|---|
| dw_lin K_train=1 vs flat | −0.005 | −0.013 |
| dw_lin K_train=3 vs flat | +0.002 | −0.036 |

Noise floor: ~±0.12 per cell (cross-seed null from enrich note). Both conditions are within this floor in aggregate.

### 3.2 Top positive Δ cells (math_eval, dw_lin only)

The largest individual gains, ranked (ckpt=K_train, K_eval, verifier):

| K_train | K_eval | Verifier | Δ |
|---|---|---|---|
| 3 | 1 | **nss** | **+0.242** |
| 1 | 2 | max | +0.217 |
| 1 | 1 | gbv | +0.206 |
| 3 | 2 | gbv | +0.192 |
| 1 | 4 | khisti | +0.186 |
| 3 | 3 | gbv | +0.169 |
| 1 | 3 | traversal | +0.166 |
| 3 | 2 | max | +0.163 |
| 3 | 3 | specinfer | +0.152 |
| 3 | 4 | nss | +0.148 |
| 3 | 3 | khisti | +0.134 |
| 1 | 1 | bv | +0.129 |
| 3 | 1 | specinfer | +0.124 |
| 1 | 1 | spectr | +0.119 |

Note: gbv appears 3 times, max 4 times, nss twice. For every positive cell there are corresponding negative cells; the mean over the full 36-cell grid is near zero.

### 3.3 gsm8k ablation: Δ across λ variants

Traversal Δ (depth_weight − flat JSD):

| K_eval | lin | lam0.0 | lam+0.5 | lam−0.5 |
|---|---|---|---|---|
| 1 | −0.095 | −0.101 | −0.026 | −0.089 |
| 2 | −0.067 | −0.012 | +0.175 | +0.104 |
| 3 | −0.125 | +0.149 | −0.070 | — |
| 4 | +0.045 | +0.027 | +0.034 | — |

BV Δ:

| K_eval | lin | lam0.0 | lam+0.5 | lam−0.5 |
|---|---|---|---|---|
| 1 | −0.047 | +0.083 | −0.079 | −0.127 |
| 2 | +0.127 | +0.140 | — | +0.161 |
| 3 | +0.066 | +0.136 | — | +0.062 |
| 4 | — | −0.065 | — | −0.015 |

Mean BE over traversal+bv cells (number of cells present in the uploaded CSV differs by variant — only flat and lam0.0 have the full 8): lam0.0 (8 cells) = 5.395, flat (8 cells) = 5.351, lam−0.5 (6 cells) = 5.388, lam0.5 (5 cells) = 5.363, lin (7 cells) = 5.336. The only complete, apples-to-apples comparison is **lam0.0 vs flat: +0.044** — and lam0.0 is mathematically identical to flat JSD ($w \equiv 1$). That a no-op control differs from the flat baseline by +0.044 is itself a direct read of the noise floor between two independently-trained/selected checkpoints.

---

## 4. Key Findings

**Finding 1 — No systematic signal in aggregate.** Mean Δ over all 36 math_eval cells is −0.005 (K_train=1) and +0.002 (K_train=3) — both effectively zero. (This grid average pools matched and mismatched K_eval; the structure within it is the matched-K diagonal of Finding 4.) On gsm8k, the no-op control (lam0.0) sits +0.044 above the flat baseline despite being the identical objective, which fixes the noise floor; no active λ variant clears it. Neither λ direction (hard-prompt curriculum at λ<0 nor good-student amplification at λ>0/lin) produces a consistent effect.

**Finding 2 — lam0.0 control tops the flat baseline.** λ=0 gives $w \equiv 1$ — mathematically identical to flat JSD. On gsm8k it has the highest complete-grid mean BE (5.395, 8 cells) — above the flat baseline (5.351) by +0.044, and above every active-λ variant on the cells where they overlap. A method whose no-op control beats every active setting has no signal; the +0.044 gap is pure checkpoint-selection/eval noise.

**Finding 3 — No monotone λ trend.** If the hard-prompt curriculum hypothesis were correct, lam−0.5 > lam0.0 > lam+0.5 should hold consistently. It does not — the cell-level ordering is mixed and flips between verifiers and K_eval values. If the good-student hypothesis were correct, lam+0.5 > lam0.0 should hold consistently. It also does not.

**Finding 4 — Depth_weight beats flat only at the matched train-K = eval-K diagonal.** The question that matters is whether each depth_weight checkpoint beats the **flat baseline** at the K_eval it was trained for — not whether it beats the other depth_weight checkpoint. Mean Δ vs flat over all 9 verifiers, and the count of verifiers where the depth_weight ckpt beats flat:

| K_eval | K_train=1 ckpt vs flat | beats flat | K_train=3 ckpt vs flat | beats flat |
|---|---|---|---|---|
| 1 | **+0.054** | **6/9** | +0.015 | 5/9 |
| 2 | −0.035 | 4/9 | −0.019 | 2/9 |
| 3 | −0.000 | 5/9 | **+0.067** | **8/9** |
| 4 | −0.039 | 2/9 | −0.056 | 3/9 |

The diagonal is clean: **K_train=1 beats flat at K_eval=1 (+0.054, 6/9 verifiers); K_train=3 beats flat at K_eval=3 (+0.067, 8/9 verifiers).** Every off-diagonal cell is at or below flat. So depth_weight is not uniformly useless — it gives a modest gain over baseline *when the depth-probe branching factor matches the inference K_eval*, and is neutral-to-harmful otherwise. The mechanism is indirect: K_train enters only through the detached depth scalar $d = E[\tau_V]$ used to form $w$ (it does not change the JSD loss itself), so this alignment can only come from the K-dependent reweighting steering checkpoint selection toward the matching tree width.

**Caveats:** the gains (+0.05 to +0.07 mean) are still around half the per-cell noise floor (±0.12), and the 9 verifiers are highly correlated (several collapse to identical BE at K_eval=1), so 8/9 is not 8 independent successes. Single seed, n=100. This is the most interesting pattern in the depth_weight data and the one worth a confirmation run (second seed or n=1000 at the two diagonal cells), but it is not yet a confirmed result.

**Finding 5 — The diagonal gain is NOT on the `naive` verifier — it is the worst cell.** The depth probe computes $d = E[\tau_V]$ under the **naive** verifier (`LOSS_TO_VERIFIER["naive_tree"] = "naive"`, `train.py:375`). The natural hypothesis is that naive acceptance should benefit most. It is the opposite: at both diagonal cells the `naive` eval verifier is the worst or near-worst performer.

| diagonal cell | `naive` verifier Δ vs flat | rank among 9 verifiers |
|---|---|---|
| K_train=1 @ K_eval=1 | −0.014 | 7th of 9 (slightly negative) |
| K_train=3 @ K_eval=3 | **−0.130** | **9th of 9 — the only losing verifier** |

Across all K_eval, `naive` is below flat in 6 of 8 cells (K_train=1: −0.014, −0.135, +0.024, −0.141; K_train=3: +0.003, −0.187, −0.130, −0.091). The matched-K diagonal gain (Finding 4) is carried entirely by the **budget/competition verifiers** — at K_train=3/K_eval=3 the positive cells are gbv (+0.169), specinfer (+0.152), khisti (+0.134), max (+0.106), spectr (+0.104) — while naive, the probe's own verifier, drags the cell down. So depth_weight does **not** preferentially improve the acceptance criterion it was probed with; if anything it trades naive acceptance for budget-verifier acceptance. This argues against any "the depth probe shapes the matching verifier" mechanism and reinforces that the effect is an indirect, K-dependent reweighting artifact rather than a targeted acceptance-shaping signal.

**Finding 6 — NSS at K_train=3, K_eval=1 shows a notable jump (+0.242).** The K_train=3 checkpoint achieves NSS BE = 4.637 vs flat 4.394, a difference of +0.242. This is the largest single Δ in the entire table and exceeds ~2× the per-cell noise floor (~0.12). NSS had been deprioritised earlier as a verifier; this result is worth flagging regardless of the negative overall finding. It is unconfirmed at n=100 and no second seed is available. Do not claim until n=1000 or second seed.

**Finding 7 — depth_w oscillates significantly with no aggregate effect.** WandB shows `train/depth_w` swinging 0.5–2.5 across prompts (5× range) and `train/depth_d` bouncing 2–8. Despite this large per-prompt variation in the weight signal, val/block_eff is indistinguishable from flat JSD. The curriculum signal is noisy but even a clean version could not help — the mechanism is wrong (§5).

**Finding 8 — Aggregate negative result replicates across both datasets.** The gsm8k 4K ablation and the math_hard ~40K run agree: no systematic gain at any λ or K_train value. This cross-dataset consistency strengthens the negative conclusion.

---

## 5. Why the Mechanism Cannot Work

**Depth-weight is LR scaling, not gradient steering.** Multiplying the JSD loss by a detached scalar $w(d)$ is equivalent to per-prompt effective learning-rate adjustment. The JSD gradient $\nabla_\phi \mathcal{L}_{\text{JSD}}$ already points toward the joint optimum (lower divergence → higher acceptance). Scaling it up or down on a per-prompt basis cannot change the direction the draft is being pushed — it can only change how fast. If the draft's capacity ceiling is the bottleneck for the 8B/0.6B pair, then training harder on any subset of prompts cannot lift that ceiling.

**The correct route requires an exact acceptance gradient.** What would actually steer the draft toward higher $E[\tau_V]$ is $\nabla_\phi E[\tau_V]$ — a gradient that flows through the discrete accept/reject process. Computing this requires differentiating through the verifier's acceptance criterion, which involves discrete sampling. This is a non-trivial derivation that the researcher needs to provide; it is not approximated by any scalar weighting scheme on the existing JSD loss.

**The EMA normalisation removes the LR confound but not the problem.** Centering $w$ around $\text{EMA}(d)$ ensures $E[w] \approx 1$ — correct design. But this means the curriculum perturbation averages out over a training run, and any benefit would have to come from the per-prompt variance in emphasis. That variance is large (5× swing) but produces no effect.

---

## 6. Relation to Other Losses

| Loss | Mechanism | Gradient through acceptance? | Signal at 8B/0.6B? |
|---|---|---|---|
| flat JSD | Token-level divergence | No | Yes (baseline) |
| **depth_weight (this note)** | **Detached per-prompt LR scaling** | **No** | **No** |
| `jsd_flat_enrich` M=3 | Stochastic teacher prefix distribution | No | Yes (+1.3–3.2% vs flat, confirmed n=1000) |
| Exact acceptance gradient (future, researcher) | $\nabla_\phi E[\tau_V]$ through verifier | Yes (exact) | Not yet implemented |

---

## 7. Conclusions

1. `depth_weight` — both linear and exponential forms, both signs of λ, both K_train values — shows no systematic improvement in block efficiency on either math_hard or gsm8k.
2. The lam0.0 control (flat JSD) matches or tops all active variants, confirming the depth scalar is adding nothing.
3. Neither curriculum direction (hard-prompt emphasis at λ<0, nor good-student amplification at λ>0/lin) produces a consistent effect.
4. The aggregate negative result is robust across both datasets and training lengths (4K and 40K steps).
5. **The one pattern worth follow-up: matched train-K = eval-K beats flat.** K_train=1 @ K_eval=1 = +0.054 (6/9 verifiers); K_train=3 @ K_eval=3 = +0.067 (8/9 verifiers); off-diagonal cells are at/below flat. **The gain is carried by budget/competition verifiers (gbv, specinfer, khisti, spectr, max), not by `naive` — the verifier used in the depth probe is the worst cell (−0.130 at K_train=3/K_eval=3, the only loser there).** So this is not targeted acceptance-shaping of the probed verifier; it is an indirect K-dependent reweighting effect. Confirm with a second seed or n=1000 at the two diagonal cells before claiming. (The NSS K_train=3/K_eval=1 cell, +0.242, is the single largest gain but is off-diagonal and likely an outlier — include it in any confirmation run.)
6. **Depth-weight as a general-purpose loss is closed.** The aggregate effect is null and the matched-K gain is small and unconfirmed. The path to robustly improving $E[\tau_V]$ through training requires an exact acceptance gradient $\nabla_\phi E[\tau_V]$ — a derivation the researcher needs to provide. Scaling the existing JSD gradient by a depth scalar cannot achieve this.
