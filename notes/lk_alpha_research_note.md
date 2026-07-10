# LK-α Acceptance-Rate Distillation (`lk_alpha`)
## Research Note

**Status:** **Negative — ties/loses to flat JSD, no consistent win on any deployment verifier.** LK-α (Samarin et al., arXiv:2602.23881) trains the draft by directly maximising single-draft acceptance: `L = −log α`, `α = Σ_v min(p_s, p_t) = 1 − TV(p_s, p_t)`. The `−log` wrapper gives `∂L/∂θ = (1/α)·∂α/∂θ` — gradient amplified exactly when acceptance is low — and (unlike JSD) only pushes q **up** on the deficit set `{v : q(v) < p(v)}`, never spending capacity where `q ≥ p` (`losses/flat.py:93`). This is the **from-scratch decisive test** the [project report](../project_report.md) called for; the mechanistic prediction (the same `1/α` factor is already carried by `naive_log`/`traversal_log`, which tie/lose to JSD) **held**.

**Draft–Teacher:** Qwen3-0.6B / Qwen3-8B | **Train:** math_hard, from scratch (not JSD-warm) | **Eval:** math_eval + olympiad_eval, n=100, seed 123, L=8, K=1–4, **sdpa** | **Variant:** greedy `lk_alpha` (single greedy teacher rollout); the deployment-matched `lk_alpha_enrich` (K stochastic rollouts) is **unrun**.
**Data:** [`Results/per_checkpoint_sweeps_2026-07/lk_alpha_mathhard_s123.csv`](../Results/per_checkpoint_sweeps_2026-07/lk_alpha_mathhard_s123.csv).

> **⚠ Backend caveat (matters here).** `lk_alpha` is **sdpa**; the per-checkpoint 0.6/8 JSD baseline (`jsd_mathhard_s123.csv`) has **FA2** for all math_eval base-verifier cells (sdpa only for olympiad K2–4 and DDTE modes). So math_eval deltas below are **cross-backend** (~0.15 BE noise, see [[eval-attn-backend-cross-machine-divergence]]). The **olympiad traversal K2–4** comparison *is* matched sdpa-to-sdpa — and shows no win either.

---

## Experiment & verdict

| Metric | lk_alpha | vs JSD | Verdict |
|---|---|---|---|
| traversal BE, math K1–4 | 6.083 / 6.052 / **5.685** / 5.771 | Δ +0.06 / +0.18 / **−0.27** / −0.06 (mean K2–4 ≈ −0.05)† | **Loses at K3**; ties elsewhere |
| traversal BE, oly K1–4 | 5.984 / 5.995 / 5.710 / 5.627 | Δ +0.14 / **+0.17 / +0.05 / −0.13** (K2–4 matched sdpa; mean ≈ +0.03) | No win — sign-flips within noise |
| best deployment verifier (bv) | — | math Δ +0.16/+0.08/−0.06/+0.13; oly −0.31/+0.08/+0.24/+0.09 | Sign-flips both datasets |
| any consistently-positive verifier? | — | only **gbv** (weak verifier): math +0.16/+0.06/+0.13/+0.07 | Not deployment-relevant |

†math baseline is FA2 (cross-backend); the K3 loss (−0.273, absolute 5.685 vs canonical 0.6/8 bars 5.870 [40K] / 5.957 [per-ckpt]) is larger than the ~0.15 backend noise, so it survives the caveat.

---

## Findings

1. **No consistent win over JSD on any deployment verifier.** Across all 9 verifiers × K=1–4 × 2 datasets, every deployment-relevant verifier (traversal, bv, specinfer) **sign-flips across K within the noise band**; the mean traversal Δ is ≈ −0.05 (math) / +0.03 (olympiad) — indistinguishable from zero. Only `gbv` (a weak verifier) is uniformly slightly positive.

2. **Clear loss on the traversal deployment verifier at K3.** math K3 = 5.685, ~0.27 below JSD and below both canonical 0.6/8 bars (5.870 / 5.957) — the one above-noise, backend-robust signal, and it's *negative*. The deficit-set-only gradient de-emphasises the high-acceptance tail that traversal's bottom-up path products reward, so it under-serves exactly the verifier that matters most.

3. **The `1/α` amplification mechanism alone is not sufficient — as predicted.** LK-α is the flat, per-token, full-vocabulary form of the same `−log(Σ min)` acceptance functional our `naive_log` / `traversal_log` tree losses carry (identical `1/α` factor). Those tie/lose to JSD at 0.6/8 and collapse at 1.7/32 ([`tree_losses_research_note.md`](tree_losses_research_note.md)); LK-α as a standalone flat objective reproduces the "tie/lose" outcome. On independent (non-EAGLE-conditioned) drafts, gradient amplification when acceptance is low does not, by itself, beat JSD's balanced two-sided divergence.

4. **Why LK-α helps in the original paper but not here — the differentiator is conditioning + scale, not the loss math.** Samarin et al. apply LK-α to EAGLE-style feature-conditioned draft heads at larger scale; our result isolates the *loss* on a standalone 0.6B draft and finds the loss alone carries no advantage over JSD (consistent with the [[lk-loss-decomposition]] analysis).

---

## Open / caveats

- **`lk_alpha_enrich` (deployment-matched) is unrun.** The greedy variant tested here trains on one greedy teacher rollout; the stochastic-rollout version (K teacher paths, the enrich analog) is the fairer replication of LK's chain setup. Given the greedy version shows no signal and enrich's own gains are modest/variance-driven ([`enrich_research_note.md`](enrich_research_note.md)), a large enrich-LK win is unlikely, but it's the one untested cell.
- **Backend:** re-run the 0.6/8 JSD baseline base-verifiers under sdpa (or lk_alpha under FA2) to remove the cross-backend caveat on math_eval; the conclusion (no win) already holds on the matched-sdpa olympiad subset and on the backend-robust K3 loss.
- Single seed, n=100.
