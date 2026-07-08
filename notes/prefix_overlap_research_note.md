# Prefix-Overlap Distillation
## Research Note

**Status:** Closed — negative. No PO objective beats JSD flat on the traversal (deployment) verifier at **either** model pair. The one 0.6B/8B borderline (NSS warm+logprob, +0.155 traversal K=3) sat on the noise floor and did not survive the 1.7B/32B replication; cold-start PO at 1.7B/32B is clearly *worse* than JSD, and the `traversal` objective — the one that directly targets the deployment verifier — is the single worst variant of all.
**Draft–Teacher:** Qwen3-0.6B / Qwen3-8B (warm-start) **and** Qwen3-1.7B / Qwen3-32B (cold-start) | **Train:** math_hard | **Eval:** math_eval + olympiad_eval (100 prompts, A100, temp=0.2) | **JSD bars:** traversal K=3 = 5.870 (0.6B/8B), 5.783 (1.7B/32B). Raw CSVs: [`Results/per_checkpoint_sweeps_2026-07/`](../Results/per_checkpoint_sweeps_2026-07/) (`po_*`).

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

## 1.7B/32B Cold-Start Replication (math_eval + olympiad_eval, L=8, n=100)

Cold-start (no JSD warm), single seed, `--prefix_root_spacing 4`. Δ = traversal BE − `jsd_math_hard_s123` at matched K. This is the decisive test the 0.6B/8B borderline needed — and it fails.

| Objective | Code | trav Δ K1 | K2 | K3 | K4 | Verdict |
|---|---|---|---|---|---|---|
| `traversal` cold (`po_traversal_multiN4`) | `compute.py:221` | −0.450 | −0.379 | −0.268 | −0.193 | **Worst variant.** The objective that directly targets the traversal verifier is the *most* harmful — consistently, at every K, far past noise. |
| `nss` cold (`po_nss_multiN4`) | `compute.py:226` | −0.147 | −0.248 | +0.116 | +0.166 | Sign-flips; net negative at low K. |
| `logprob` cold (`po_logprob_multiN4`) | `compute.py:232` | −0.053 | −0.229 | +0.068 | +0.099 | Sign-flips; net negative at low K. |

On **olympiad_eval** `po_nss`/`po_logprob` are mildly *positive* across K (≈+0.05 to +0.17) while `po_traversal` stays strongly negative (−0.20 to −0.35) — but the positives are sub-noise and reverse on math_eval, so no consistent win. Full per-verifier grids in the CSVs.

**Why `po_traversal` is the worst, not the best:** the K-aware weight `w_t ∝ K(1−α)^{K−1}α` up-weights *deep, low-α* prefix positions — exactly the tokens a cold draft can't yet match. Cold-start, this pours gradient into unreachable tree tails (same failure mode as the log-tree losses; see [`tree_losses_research_note.md`](tree_losses_research_note.md)). Warm-start on 0.6B/8B masked this because the draft was already near the JSD optimum; cold-start exposes it. **Lesson: an objective analytically aligned with the deployment verifier is not automatically a good *training* signal** — it can be actively anti-curricular.

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

## Resolved (was Pending)

1. **1.7B/32B cold-start replication — DONE, negative.** See the cold-start table above. The nss_warmlogprob borderline did not generalize; PO is closed as a negative family.
2. **DDTE verifier eval — DONE.** DDTE (`traversal_dL{3,4,5}`) was run across every PO draft and the JSD baseline; it lifts BE by +0.15–0.5 at K3/K4 *uniformly* (PO and JSD alike), so it does **not** differentially favor PO drafts — a stronger verifier does not rescue PO. Full analysis in [`ddte_research_note.md`](ddte_research_note.md).
3. **Relation to enrichment.** PO reweights training on a *fixed teacher-context* prefix; enrichment instead trains on *fresh stochastic teacher rollouts* and is the one family that beats JSD (0.6B/8B). The unrun on-policy PO variant (Limitation 1) and the `enrich_draftcond` experiment (draft-conditioned rollouts, [`enrich_research_note.md`](enrich_research_note.md)) are the two ways to inject the draft's *own* distribution into the prefix signal — the axis PO never touched. If any PO idea is revived, it should be the on-policy/draft-conditioned one, not more teacher-context objectives.

Log-space tree losses (`traversal_log`, `nss_log`) are complete and share PO-traversal's failure mode — results in [`tree_losses_research_note.md`](tree_losses_research_note.md).

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
