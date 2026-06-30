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
| `traversal` | Σ_{d≥i} K(1−α_d)^{K−1}α_d | →0 as α→1 | **→0×** | **structurally vanishes at a warm start**: draft matches teacher ⇒ α_d≈1 ⇒ (1−α_d)^{K−1}≈0 ⇒ weights≈0. Explains traversal-warm = −0.153 (LR drift + noise, no signal) |
| `nss` | (L−i+1)·𝟙[q_i<p_i] | masked | one-sided | only lifts deficient tokens, never trims; at q≈p the mask fires on few tokens with little room |

**Two distinct failure modes, not one:**
1. **Redundancy (logprob):** gradient is large but points where JSD already converged — depth-reweighting is a 2nd-order correction on top of the token-matching JSD already achieved.
2. **Structural vanishing (traversal, nss):** the very property that makes them BE-aligned (α-adaptive / catch-up-masked weights) drives the gradient to ~0 exactly at a good warm start.

### LR vs anneal timing

Constants: `steps=8000`, `GRAD_ACCUM=8` ⇒ `total_opt_steps=1000`; warmup=100; cosine decay to `LR_MIN_RATIO=0.1`. `anneal_steps=5000` micro-steps = **opt-step 625**.

- CE anneals to 0 at opt-step 625, where cosine LR = `0.1 + 0.9·½(1+cos(π·525/900))` = **0.43× peak**.
- The pure-PO window (opt-step 625→1000) runs entirely in the cosine tail, LR **0.43 → 0.1× peak**.

For **logprob** this is moot — PO (4.5× CE) dominates throughout under full LR; the issue is redundancy, not budget. For **traversal/nss warm**, PO weight ≈0 early, so the first 625 opt-steps are effectively CE-only (≈ more JSD) at LR 1.0→0.43, and the only PO-dominant window is the low-LR tail. The objective these runs were named for barely trains with meaningful LR.

### Diagnostics to disambiguate (cheap, one run each)

1. **Pure PO, full LR budget (clean control — run first):** warm-start from JSD, `--prefix_aux_weight 0` (no CE), fresh warmup→cosine. If it still doesn't move ⇒ structural (redundancy/vanishing), not LR. If it moves ⇒ anneal/LR timing was starving it.
2. **Constant λ=1, no anneal** (`--prefix_anneal_steps 0 --prefix_aux_weight 1`): isolates whether the anneal-to-zero transition (vs keeping the regularizer on) was the problem.
3. **Short anneal** (`--prefix_anneal_steps 500`): CE stabilizes only the first ~60 opt-steps; PO gets opt-steps 60→1000 at high LR.

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

1. **Pure-PO full-LR diagnostic** (see Diagnostics #1) — the single experiment that decides whether the flat results are an LR/anneal artifact or structural. Run before any other PO follow-up.
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
