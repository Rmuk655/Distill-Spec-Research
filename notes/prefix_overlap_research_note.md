# Prefix-Overlap Distillation
## Research Note

**Status:** Inconclusive — no PO objective consistently beats JSD flat on the traversal (deployment) verifier. One candidate (NSS warm+logprob, +0.155 traversal K=3) is at the noise floor and needs a confirmation run.
**Draft–Teacher:** Qwen3-0.6B / Qwen3-8B | **Train:** math_hard | **Eval:** math_eval (100 prompts, A100, temp=0.2) | **JSD bar:** traversal K=3 = 5.870

**Math foundation:** Rahul's *Prefix-Overlap Distillation Objective* (internal PDF). Setup, LCP formula, unbiased estimator via teacher samples, multi-root scheme, CE mixing rationale — all in §1–8 there; not repeated here.

---

## Objective Variants

Rahul's PDF (§4) defines the base estimator: sample M teacher continuations P^(m) from each root, maximize Σ_m Σ_t q(P^(m)_{1:t} | c). The four `--prefix_objective` options implement this or approximations designed to fix its practical failure modes:

| `--prefix_objective` | Relation to Rahul §4 | Why it was added |
|---|---|---|
| `prob` | Exact implementation of §4 — maximizes Σ_t q_θ(P_{1:t}) | Baseline; exact but collapses in practice (Rahul §7: if any early token is weak, all deeper terms vanish exponentially) |
| `logprob` | Log-space surrogate: Σ_t S_t where S_t = Σ_{i≤t} log q_i — triangular weighting on log q | Avoids gradient collapse in `prob`; gradient is O(1) per token regardless of depth, like CE |
| `traversal` | K-aware reweighting of log q: Σ_t w_t · log q_t where w_t = suffix-sum of K·α(1-α)^{K-1}α | Targets the traversal verifier specifically — weights match ∂E[BE]/∂log q under traversal acceptance |
| `nss` | CE masked to positions where q_i < p_i, triangular weights | Targets NSS acceptance — only pushes up tokens where the student is the bottleneck |

**`--prefix_anneal` logic:** Implemented as `--prefix_aux ce --prefix_aux_weight λ --prefix_anneal_steps N`, linearly annealing λ from 1→0 over N steps. CE provides a stable anchor early in training; by the time λ→0 the PO gradient takes over.

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

JSD flat baseline: traversal K=3 = 5.870. JSD K=3 is anomalously low vs K=2 (6.141) and K=4 (5.943) — the +0.155 delta is partly riding this draw. Noise floor: SE ≈ 0.10–0.15 at n=100.

---

## Gradient & Trainability Analysis

Loss is `L = -PO_score + λ·CE` (`compute.py:284`): PO weight is always 1; λ is the CE weight, annealed 1→0 (`train.py:413`). PO is active from step 0; CE is a fading regularizer. Per-objective gradient weight on ∂(−log q_i)/∂θ, vs CE's uniform weight 1 (L=8):

| Objective | Weight on token i | Σ weights | vs CE | Behavior at warm start (q≈p) |
|---|---|---|---|---|
| CE | 1 | 8 | 1× | — |
| `logprob` | L−i+1 (8…1) | 36 | **4.5×** | Large gradient, but points where JSD already converged |
| `prob` | Σ_{t≥i} q(P_{1:t}) | ~3, front-loaded | <1× | Collapses at depth (Rahul §7): deep tokens get ~0.06 weight |
| `traversal` | Σ_{d≥i} K(1−α_d)^{K−1}α_d | →0 as α→1 | **~0.1×** | Attenuated at warm start: with α≈0.7 (BE≈5.9), (1−α)^{K−1}≈0.09 → grad norm ~40–65, vs 200–440 for logprob |
| `nss` | (L−i+1)·𝟙[q_i<p_i] | ~½·36 | **~2×** | Mask fires on ~half the tokens even at q≈p (coin-flip per token); gradient does not vanish |

**Two failure modes:**
1. **Redundancy (logprob, nss, traversal):** all three show active gradients and weight movement (rising forgetting) but no BE gain over JSD. Teacher-prefix matching is a correction on top of what JSD already achieved.
2. **Gradient attenuation (traversal):** the (1−α)^{K−1} factor additionally reduces the traversal gradient ~5–8× vs logprob/nss at warm start. The gradient is present but small.

### LR vs anneal timing

For the 8k-step configs (`steps=8000, GRAD_ACCUM=8`): total_opt_steps=1000, warmup=100, cosine decay to LR_MIN_RATIO=0.1. With `anneal_steps=5000`: CE→0 at opt-step 625, where LR = **0.43× peak**. The pure-PO window (opt-steps 625–1000) runs in the cosine tail at 0.43→0.1× peak.

For `logprob` (4.5× CE gradient) and `nss` (~2× CE), LR timing is not the bottleneck — the PO gradient is large throughout. For `traversal` (attenuated ~0.1× CE), the 8k-step config puts the pure-PO phase at low LR. A 15k-step run with the same `anneal_steps=5000` shifts CE→0 to 33% through training at LR=**0.86× peak**, giving traversal a much longer window at near-full LR.

### Empirical checks

Three runs with healthy gradient and LR, all showing no BE gain:

- **logprob** (`po_mh_logprob_lr1e5`, no anneal, 15k steps): grad 200–440; LR 7.75e-6 at step 6000; val BE peaks 6.033, declines to 5.923 at early-stop, forget=1.14.
- **nss cold** (`po_nss_cold_multiN4_L8_lr1e5_s123`, CE-anneal 5k, 8k steps): grad 200–368 including post-anneal; LR 4.34e-6 at step 5000; val BE peaks 6.030, smoothed ~5.88 declining, forget rising 0.93→1.11.
- **traversal warm** (`prefix_overlap_traversal_multiN4_L8_lr1e-05_s123_ce1anneal5000`, CE-anneal 5k, 8k steps): grad 40–65 throughout; val BE peaks 5.966 (+0.096), then declines.

(`grad=0.00` lines are non-optimizer micro-steps: `GRAD_ACCUM=8`, logged every 10 → optimizer steps appear only on multiples of 40.)

---

## Limitations

**1. On-policy sampling not implemented:**
Rahul §4's estimator IS the exact gradient of E[LCP] — unbiased, with teacher as the fixed sampler. The on-policy variant (sample Q ~ q_θ, score LCP(P, Q)) requires REINFORCE because q_θ enters the sampling path and LCP is a non-differentiable indicator. This is the high-variance regime previously explored with no detectable signal. On-policy PO needs variance reduction (baselines / control variates) to be viable.

**2. M not matched to K:**
M teacher continuations per root is fixed at 4 independent of K. Matching M=K would align training samples with the verifier's distribution at test time. Unrun: sweep M=K for each K.

**3. Random root offset not implemented:**
Rahul §5 requires a random offset for unbiased coverage of all positions. Fixed N=4 spacing (roots always at 4, 8, 12, …) biases away from early positions. Unrun: add random offset.

**4. Single seed, n=100:**
SE ≈ 0.10–0.15. Deltas below 0.15 are inconclusive. The nss_warmlogprob +0.155 is at this floor.

**5. Teacher rollout context for roots:**
All roots use the teacher's own prior tokens as context. At test time the draft generates its own prior tokens — a covariate shift that none of the objectives correct for.

---

## Pending

1. **Traversal 15k warm (running):** `steps=15000, anneal_steps=5000` — CE→0 at LR=0.86× peak. Early best raw=6.042 / smoothed=5.951 at step 2000, step 2400/15000. Watch through the anneal boundary (~step 5000).
2. **Confirm nss_warmlogprob** — re-run at n=200 or seed=456; +0.155 traversal K=3 is the only above-noise positive.
3. **DDTE verifier eval (eval-only):** run DDTE verifier on JSD-flat and best PO draft. Tests whether a stronger verifier amplifies small draft-distribution differences.
4. **Log-space tree losses** (`traversal_log`, `naive_log`) — warm-start from JSD ckpt; highest priority, unrelated to PO.

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
