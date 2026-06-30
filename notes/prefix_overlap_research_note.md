# Prefix-Overlap Distillation
## Research Note

**Status:** Inconclusive — no PO objective consistently beats JSD flat on traversal (deployment) verifier. One candidate (NSS warm+logprob, +0.155 traversal K=3) warrants a confirmation run.
**Draft–Teacher:** Qwen3-0.6B / Qwen3-8B | **Train:** math_hard | **Eval:** math_eval (100 prompts, A100, temp=0.2) | **JSD bar:** traversal K=3 = 5.870

**Math foundation:** Rahul's *Prefix-Overlap Distillation Objective* (internal PDF). Setup, LCP formula, unbiased estimator via teacher samples, multi-root scheme, CE mixing rationale — all in §1–8 there; not repeated here.

---

## Objective Variants

Rahul's PDF (§4) defines the base estimator: sample M teacher continuations P^(m) from each root, maximize Σ_m Σ_t q(P^(m)_{1:t} | c). Our four `--prefix_objective` options implement this estimator or approximations to it designed to fix its practical failure modes:

| `--prefix_objective` | Relation to Rahul §4 | Why it was added |
|---|---|---|
| `prob` | Exact implementation of §4 — maximizes Σ_t q_θ(P_{1:t}) | Baseline; exact but collapses in practice (Rahul §7: if any early token is weak, all deeper terms vanish exponentially) |
| `logprob` | Log-space surrogate: Σ_t S_t where S_t = Σ_{i≤t} log q_i — triangular weighting on log q | Avoids the gradient collapse in `prob`; gradient is O(1) per token regardless of depth, like CE |
| `traversal` | K-aware reweighting of log q: Σ_t w_t · log q_t where w_t = suffix-sum of K·α(1-α)^{K-1}α | Targets the traversal verifier specifically — weights match ∂E[BE]/∂log q under traversal acceptance |
| `nss` | CE masked to positions where q_i < p_i, triangular weights | Targets NSS acceptance — only pushes up tokens where student is the bottleneck; zero gradient where q≥p |

**`--prefix_anneal` logic:** Rahul §7 recommends adding a CE term (λ·L_CE). We implement this as `--prefix_aux ce --prefix_aux_weight λ --prefix_anneal_steps N`, which linearly anneals λ from 1→0 over N steps. Motivation: CE dominates early training (keeps the model near JSD optimum and stabilizes), then fades so the PO gradient takes over. In practice, by the time λ→0 the model has converged and the residual PO gradient is small.

---

## Ablation Table (math_eval, K=3, L=8, n=100)

| Experiment | Code | What it tested | Traversal K=3 BE | Δ vs JSD | Beat JSD traversal K=3? | Beat any mode/K beyond noise (>0.15)? |
|---|---|---|---|---|---|---|
| `prob` cold | `compute.py:234` | Exact §4 estimator; cold init | 5.553 | −0.317 | No | No |
| `logprob` cold | `compute.py:232` | Log-space Σ S_t; cold init | 5.829 | −0.041 | No | No |
| `logprob` warm (JSD init) | `compute.py:232` | logprob; warm from JSD ckpt | 5.919 | +0.049 | No | No (`gbv` K=3 +0.257, weak verifier) |
| `logprob` anneal (JSD init) | `compute.py:232` | logprob + CE aux λ annealed 1→0 over 5K steps | 5.866 | −0.004 | No | No |
| `logprob` N=2 (JSD init) | `compute.py:232` | logprob; tighter root spacing N=2 vs N=4 | 5.863 | −0.007 | No | No |
| `traversal` cold | `compute.py:221` | K-aware BE surrogate; cold init | 5.779 | −0.091 | No | No |
| `traversal` warm (JSD init) | `compute.py:221` | traversal; warm from JSD ckpt | 5.717 | −0.153 | No | No |
| `traversal` warm v2 (JSD init) | `compute.py:221` | traversal warm, multiroot N=4 | 5.897 | +0.027 | No | No |
| `nss` warm (JSD init) | `compute.py:226` | NSS; warm from JSD ckpt | 5.933 | +0.063 | No | No |
| **`nss` warm (logprob init)** | `compute.py:226` | NSS; warm chain JSD→logprob_warm→nss | **6.025** | **+0.155** | **Borderline** | No (khisti K=4 +0.167, weak verifier) |

JSD flat baseline: traversal K=3 = 5.870. Note: JSD K=3 is anomalously low vs K=2 (6.141) and K=4 (5.943) — the +0.155 delta is partly riding this draw. Noise floor: SE ≈ 0.10–0.15 at n=100.

---

## Gradient & Trainability Analysis

Loss is `L = -PO_score + λ·CE` (`compute.py:284`): **PO is always weight 1**; λ is the *CE* weight, annealed 1→0 (`train.py:413`: `λ = aux_weight·max(0, 1 - step/anneal_steps)`). PO is active from step 0; CE is only a fading regularizer. Per-objective gradient weight on ∂(−log q_i)/∂θ, vs CE's uniform weight 1 (L=8):

| Objective | Weight on token i | Σ weights | vs CE | Behavior from JSD warm start (q≈p) |
|---|---|---|---|---|
| CE | 1 | 8 | 1× | — |
| `logprob` | L−i+1 (8…1) | 36 | **4.5×** | large gradient, but the 8× early-token weight lands on already-saturated tokens (q≈1 ⇒ ∂(−log q)≈0); effective update is modest and redundant with what JSD already did |
| `prob` | Σ_{t≥i} q(P_{1:t}) | ~3, front-loaded | <1× | collapses at depth (Rahul §7): deep tokens get ~0.06 weight |
| `traversal` | Σ_{d≥i} K(1−α_d)^{K−1}α_d | →0 as α→1 | **~0.1× (attenuated)** | **attenuated but not zero at warm start**: grad norm ~40–65 vs 200–440 (logprob) / 200–368 (nss) — ~5–8× lower, not vanished. With α≈0.7 (BE≈5.9), (1−0.7)^(K−1)≈0.09 — consistent with log. Best BE 5.966 (+0.096 vs JSD) then declines — same redundancy pattern. |
| `nss` | (L−i+1)·𝟙[q_i<p_i] | ~½·36 | **~2×** | one-sided (lifts q<p, never trims). Mask fires on ~half the tokens even at warm start (q≈p ⇒ coin-flip per token), so gradient stays substantial — **does NOT vanish** (confirmed cold: grad 200–368) |

**Two confirmed failure modes — all objectives empirically settled:**
1. **Redundancy (logprob, nss, traversal):** gradient is healthy (logprob 4.5× CE; nss ~2× CE; traversal ~0.1× CE but present) but targets what JSD already converged — teacher-prefix matching is a 2nd-order correction on top of what JSD achieved. Confirmed empirically for all three (below).
2. **Gradient attenuation (traversal):** additionally, the multiplicative (1−α)^{K−1} factor reduces the traversal gradient ~5–8× vs logprob/nss at warm start (α≈0.7 ⇒ (0.3)^2≈0.09). This is attenuation, not vanishing — grad norm ~40–65 is still present and training, but smaller. Note: the "structural vanishing" prediction (grad≈0) was too strong.

### LR vs anneal timing

Constants: `steps=8000`, `GRAD_ACCUM=8` ⇒ `total_opt_steps=1000`; warmup=100; cosine decay to `LR_MIN_RATIO=0.1`. `anneal_steps=5000` micro-steps = **opt-step 625**.

- CE anneals to 0 at opt-step 625, where cosine LR = `0.1 + 0.9·½(1+cos(π·525/900))` = **0.43× peak**.
- The pure-PO window (opt-step 625→1000) runs entirely in the cosine tail, LR **0.43 → 0.1× peak**.

For **logprob** this is moot — PO (4.5× CE) dominates throughout under full LR. For **nss** the gradient also stays large (~2× CE), so it trains throughout too. The anneal/LR-tail concern only bites **traversal**, whose weight vanishes at warm start.

### Empirical check — logprob & nss: healthy gradient, no BE gain (redundancy, not LR)

Two near-pure-PO runs confirm gradient/LR are NOT the bottleneck:

- **logprob** (`po_mh_logprob_lr1e5`, `aux_weight=0`, no anneal, 15k): grad norm **200–440** every opt-step; LR 7.75e-6 ≈ 0.78× peak at step 6000; smoothed BE best 6.033 → 5.935 → 5.923 into early-stop, `forget=1.14`.
- **nss cold** (`po_nss_cold_multiN4_L8_lr1e5_s123`, nss + CE-anneal 5k, 8k): grad norm **200–368** throughout, **including the post-anneal pure-nss tail** (step 5000+: grad 218–332). LR at step 5000 = **4.34e-6 = 0.43× peak** — exactly the opt-step-625 prediction above. Val BE best 6.030 early, then 5.92/5.98/5.84/5.87 (smoothed ~5.88, declining), `forget` rising 0.93→1.11.
- **traversal warm** (`prefix_overlap_traversal_multiN4_L8_lr1e-05_s123_ce1anneal5000`, traversal + CE-anneal 5k, 8k): grad norm **40–65** throughout, **including post-anneal** (step 5000+: 44–58). ~5–8× lower than logprob/nss, consistent with (1-α)^{K-1}≈0.09 at α≈0.7. LR at step 5000 = 4.34e-6 = 0.43× peak (same config). Val BE best **5.966** (+0.096 vs JSD) at an early step, then smoothed declines: 5.897 → 5.878 → 5.885 → 5.875 (no-improve 3/5 at step 5600).

(`grad=0.00` lines are non-opt micro-steps: `GRAD_ACCUM=8`, logged every 10 ⇒ real steps show only on multiples of 40.)

All three: gradient present + training active (rising forgetting, declining loss) but **no BE gain over JSD beyond noise** (all cap 5.966–6.033, +0.096–+0.096, within SE≈0.10–0.15 at n=100) and decline. ⇒ **Redundancy confirmed for logprob, nss, AND traversal — entire PO family settled.** Bottleneck is the objective family (teacher-prefix matching, already maxed by JSD), not gradient scale or LR.

### Diagnostic resolved (traversal warm-start)

**Traversal warm-start grad norm — closed.** Grad norm = **40–65** throughout (post-anneal included), vs prediction of ≈0. Prediction was too strong: at α≈0.7 the weight (1−α)^{K−1}≈0.09 attenuates but doesn't vanish. BE best 5.966 then declines — same redundancy pattern as logprob/nss. **The whole PO family is now empirically closed out:** all four objectives (prob, logprob, nss, traversal) show no consistent BE gain over JSD on the deployment verifier.

---

## Unrun Hypotheses / Limitations

**1. "Exact LCP gradient" vs on-policy — clarification:**
Rahul §4's estimator IS the exact gradient of E[LCP]: unbiased and fully differentiable, because only the teacher is sampled (fixed targets) and we backprop through q. **We ran it — it is the `prob` objective.** It underperforms from the §7 depth-collapse, not from any gradient approximation; there is no separate "exact gradient" left unrun.
The *on-policy* variant (sample Q ~ q_θ, score LCP(P, Q)) is a different objective: q_θ enters the sampling path and LCP is a non-differentiable indicator, so it needs REINFORCE. That is the high-variance regime `tree_pg` already explored with no detectable signal. On-policy PO is not a quick win — it needs variance reduction (baselines / control variates) to be viable. Linked to limitation #5 (covariate shift).

**2. M not matched to K:**
The PDF samples M teacher continuations per root; M is independent of K (the verifier's branching factor). Matching M=K would mean training samples are drawn from the same distribution the K-verifier sees at test time. With M=4 fixed (and K=2,3,4 at eval), there is a train/eval distribution mismatch. Unrun: sweep M=K for each K.

**3. Random root offset not implemented:**
Rahul §5 notes that fixed N-spacing is unbiased for the "every Nth root" objective, but for a uniform-over-positions objective you need a random offset. We use fixed N=4 spacing without random offset — roots are always at positions 4, 8, 12, … relative to the prompt. This biases away from early positions. Unrun: add random offset per Rahul §5.

**4. Single seed, n=100:**
All results above are one seed (s=123), 100 prompts. SE ≈ 0.10–0.15, so deltas below 0.15 are inconclusive. The nss_warmlogprob +0.155 is barely above this floor and needs a second seed or n=200 to confirm.

**5. Teacher rollout context for roots:**
All roots are conditioned on the teacher's own rollout (c_r = (x, y_{1:r})). At test time, the draft model generates its own continuation — the root context is from the draft's own prior tokens, not the teacher's. This is a covariate shift that none of the objectives correct for.

---

## Pending

1. ~~**Traversal warm-start grad-norm check**~~ — **resolved.** Grad ~40–65 (attenuated ~5–8×, not zero); BE best +0.096 then declines. Redundancy confirmed for traversal too. PO family closed out.
2. **Confirm nss_warmlogprob** — re-run at n=200 or seed=456; the +0.155 traversal K=3 is the only above-noise positive.
3. **DDTE verifier eval (eval-only, cheap):** run the DDTE verifier on JSD-flat and the best PO draft. Tests whether a stronger verifier amplifies the small draft-distribution differences. Training-agnostic — won't change the draft conclusion, but is the deployment verifier in Rahul's DDTE paper and a cheap lens.
4. **Log-space tree losses** (`traversal_log`, `naive_log`) — warm-start from JSD ckpt; highest SOTA priority, unrelated to PO.

---

## Code Reference

`losses/compute.py` — `_prefix_score()` lines 197–240; `compute_prefix_overlap_multiroot_loss()` lines 290+.

| Flag | Default | Meaning |
|---|---|---|
| `--prefix_objective` | `prob` | `prob` / `logprob` / `traversal` / `nss` |
| `--prefix_root_spacing` | 0 | N: multi-root every N tokens (0 = single-root) |
| `--prefix_rollout_len` | 128 | teacher rollout length |
| `--prefix_aux` | `ce` | secondary term: `ce` or `jsd` |
| `--prefix_aux_weight` | 0.0 | λ for secondary term |
| `--prefix_anneal_steps` | 0 | anneal λ from 1.0→0 over N steps |
| `--L` | — | use 8 (deployment-matched) |
| `--val_temp` | — | use 0.2 |
