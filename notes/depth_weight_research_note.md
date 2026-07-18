# Depth-Weighted JSD (curriculum reweighting ablation)
## Research Note on `jsd+dw_naive_tree_{lin,lam}`

**Status:** **Negative overall** — no aggregate BE gain over flat JSD at any λ or K_train, on either dataset or **either model pair**. At 0.6B/8B there was one pattern worth confirming — depth_weight beat flat *only* on the matched **train-K = eval-K diagonal** (K_train=1 @ K_eval=1: **+0.054, 6/9** verifiers; K_train=3 @ K_eval=3: **+0.067, 8/9**), neutral-to-harmful off-diagonal — but the **1.7B/32B replication (2026-07) FAILS: the K_train=3 @ K_eval=3 diagonal is mean −0.008, 4/9** (§cross-pair). So the diagonal does not reproduce at the wider capacity gap; the family is closed. Mechanism (§why): a *detached* scalar weight is per-prompt LR scaling — it cannot steer the gradient toward higher acceptance, so it cannot work in principle.

**Draft–Teacher:** Qwen3-0.6B / Qwen3-8B (primary) + **Qwen3-1.7B / Qwen3-32B (cross-pair, `jsd_dw_lin_naive_K3`)** | **Train:** math_hard (K_train=1 and 3, ~40K steps) + gsm8k (λ-sweep, 4K) | **Eval:** math_eval + olympiad_eval + gsm8k_eval, n=100, K_eval=1–4, 9 verifiers.
**Data (source of truth):** [`results/jsd_jsd_dw_naive_tree_results_math_hard_40000_steps.csv`](../results/jsd_jsd_dw_naive_tree_results_math_hard_40000_steps.csv) · [`results/jsd_jsd_dw_naive_tree_results_gsm_8k_4000_steps_softmax_lambda.csv`](../results/jsd_jsd_dw_naive_tree_results_gsm_8k_4000_steps_softmax_lambda.csv). **⚠ Parse caveat:** the 40K CSV has 54 ragged rows (extra 45th field on all K_eval=3/4 and the entire `..._K3` checkpoint) — `pandas` silently drops them; use the `csv` module (index cols by header position). **Do not compare absolute BE across datasets** (40K math_hard vs 4K gsm8k); only within-dataset Δ is meaningful.

**What it is** (`train.py:380`): `loss = w(d) * L_JSD`, with `d = E[τ_V]` computed under `no_grad` (`compute.py:297`, DP over an `iid_draft` tree — one extra target pass/step, no gradient). Linear `w = d/EMA(d)` (up-weights easy/deep prompts — "train good students harder"); exp `w = exp(λ(d−EMA))` (λ<0 = hard-prompt curriculum, λ>0 = same as linear, **λ=0 ≡ flat JSD**, a free no-op control). EMA-centred so E[w]≈1 (no LR confound).

---

## Experiments & verdicts

| Experiment | Config | What it tested | Beat flat JSD? | Beyond noise (±0.12)? |
|---|---|---|---|---|
| dw_lin K_train=1 | math_hard 40K | linear up-weight | Only at K_eval=1 (+0.054, 6/9) | No — mean −0.005/36 cells |
| dw_lin K_train=3 | math_hard 40K | linear, K=3 probe | Only at K_eval=3 (+0.067, 8/9) | No — mean +0.002/36 cells |
| lam0.0 (no-op) | gsm8k 4K | λ=0 ≡ flat control | "+0.044" over flat | **No — this is the noise floor** |
| lam+0.5 | gsm8k 4K | good-student | mixed, no trend | No |
| lam−0.5 | gsm8k 4K | hard-prompt curriculum | mixed, no trend | No |
| lin | gsm8k 4K | linear | mixed, no trend | No |

---

## Findings

1. **No aggregate signal.** Math_eval mean Δ = −0.005 (K_train=1) / +0.002 (K_train=3) over all 36 cells; the per-cell Δ distribution is symmetric about zero (biggest positive +0.242, biggest **negative −0.266** — larger). The sorted positive-tail looks impressive only because it's truncated. Bv/traversal Δ, K_eval=1–4:

   | K_eval | dw_lin K1 trav | dw_lin K1 bv | dw_lin K3 trav | dw_lin K3 bv |
   |---|---|---|---|---|
   | 1 | −0.116 | +0.129 | −0.118 | −0.053 |
   | 2 | −0.061 | +0.047 | −0.099 | −0.036 |
   | 3 | **+0.166** | −0.035 | +0.033 | +0.015 |
   | 4 | −0.212 | −0.023 | +0.040 | −0.074 |

2. **The no-op control fixes the noise floor.** On gsm8k, `lam0.0` (mathematically identical to flat JSD, w≡1) scores **+0.044 above** the flat baseline over the complete 8-cell grid — and above every active-λ variant where they overlap. A no-op beating every active setting ⇒ no signal; +0.044 is pure checkpoint-selection/eval noise between two independently trained/selected runs.

3. **No monotone λ trend.** Neither hard-prompt (λ<0 > λ=0 > λ>0) nor good-student (λ>0 > λ=0) ordering holds; cell-level ordering flips across verifiers and K_eval.

4. **The one pattern — matched train-K = eval-K beats flat.** Mean Δ vs flat over 9 verifiers, with beat-count:

   | K_eval | K_train=1 vs flat | beats | K_train=3 vs flat | beats |
   |---|---|---|---|---|
   | 1 | **+0.054** | **6/9** | +0.015 | 5/9 |
   | 2 | −0.035 | 4/9 | −0.019 | 2/9 |
   | 3 | −0.000 | 5/9 | **+0.067** | **8/9** |
   | 4 | −0.039 | 2/9 | −0.056 | 3/9 |

   Diagonal clean, off-diagonal at/below flat. K_train enters only via the detached scalar `d = E[τ_V]` (it doesn't change the JSD loss), so the alignment can only be K-dependent reweighting steering checkpoint selection toward the matching tree width.

5. **The diagonal gain is NOT on `naive` — the probe's own verifier is the worst cell.** The probe computes `d` under **naive** (`LOSS_TO_VERIFIER["naive_tree"]="naive"`), but at K_train=3/K_eval=3 the `naive` eval verifier is **−0.130, 9th of 9 (only loser)**; the gain is carried by gbv (+0.169), specinfer (+0.152), khisti (+0.134), max (+0.106), spectr (+0.104). So depth_weight does *not* shape the verifier it probes — the effect is indirect K-dependent reweighting, not targeted acceptance-shaping.

6. **NSS K_train=3 @ K_eval=1 = +0.242** — the single largest cell (~2× noise floor), off-diagonal, likely an outlier; flag but don't claim (n=100, no 2nd seed).

7. **`train/depth_w` swings 0.5–2.5 (5× range), `depth_d` 2–8 — yet val/BE is indistinguishable from flat.** Large curriculum perturbation, zero net effect (WandB training metric, not in eval CSVs). Consistent across gsm8k (4K) and math_hard (40K).

8. **Cross-pair (1.7B/32B) — the matched-K diagonal does NOT replicate (2026-07).** `jsd_dw_lin_naive_K3` (Qwen3-1.7B/32B, K_train=3, sdpa, full K1–4 + olympiad) vs the 1.7/32 JSD baseline. The matched diagonal cell (K_eval=3) is **mean −0.008, 4/9** on math_eval — nothing like the 0.6/8 **+0.067, 8/9**; and `naive` @K_eval=3 is −0.047 (probe verifier still not favoured). All K_eval means are near-zero and sign-mixed (math K1/2/3/4 = −0.016 / −0.009 / **−0.008** / +0.072; olympiad = +0.048 / −0.013 / −0.027 / −0.081), the off-diagonal K_eval=4 math bump (+0.072, 7/9) not matching any train-K. So the one 0.6/8 pattern worth confirming **fails at the wider capacity gap** — consistent with §why (a detached scalar cannot steer the gradient; any apparent 0.6/8 diagonal was an indirect selection artifact that doesn't survive a different pair). *(Backend caveat: the 1.7/32 math base-verifier baseline cells are FA2 while the dw run is sdpa — cross-backend ~0.15 noise — but the deltas sit near zero regardless, so the non-replication holds.)* Data: [`results/per_checkpoint_sweeps_2026-07/jsd_dw_lin_naive_K3_math_hard_s123_Qwen32B-Qwen1.7B.csv`](../results/per_checkpoint_sweeps_2026-07/jsd_dw_lin_naive_K3_math_hard_s123_Qwen32B-Qwen1.7B.csv).

---

## Why it can't work + positioning (theory tail)

- **Detached scalar = per-prompt LR scaling, not gradient steering.** `w(d)·L_JSD` with `d` under `no_grad` only rescales the *magnitude* of `∇_φ L_JSD` per prompt — it cannot change the *direction* the draft is pushed. If the 0.6B draft's capacity is the bottleneck, training harder on any subset can't lift the ceiling. EMA-centring makes the perturbation average out; the 5× per-prompt variance produces nothing.
- **Correct route = exact acceptance gradient** `∇_φ E[τ_V]` flowing through the discrete accept/reject — a derivation (Rahul's) that no scalar weighting approximates.
- **Novelty correction — GTO (Hu et al. 2025a, cited in DDTE arXiv:2602.16994) is prior work.** GTO combines an *expected-NSS-acceptance* reward with a *PPO-style frozen-vs-evolving-tree* objective ≈ this note's `depth_weight` + the retired `tree_pg`. So this corner is **published**; our contribution is a *negative/replication* result (detached-scalar depth_weight adds no signal at 0.6B/8B, and §why explains why a scalar can't work in principle) — not a new method.
- **Verdict:** depth_weight as a general-purpose loss is **closed** — and now doubly so: the one 0.6/8 pattern worth confirming (the matched-K diagonal) did not replicate at 1.7B/32B (Finding 8). No further runs warranted. Supported verifiers for the probe (have `expected_*_depths`): naive, nss, specinfer, spectr, khisti, traversal — **not** bv/gbv.
