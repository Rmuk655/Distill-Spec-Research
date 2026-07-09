# Eval-Time Delayed Tree Expansion (DDTE) + Distillation
## Research Note

**Status:** DDTE = **eval-time-only** delayed branching (K=1 stem for L1 steps, then K branches for L−L1). It is the **cleanest win in the program** and composes with training: it helps *every* trained draft on *both* pairs, uplift grows with K, and **traversal + enrich + DDTE = 6.418 is the global best BE at 0.6B/8B** (beats bv 6.283). But it **compensates for, does not substitute, training** (two-tier: collapsed drafts + DDTE stay ~0.5 BE below trained + DDTE). Extends Thomas et al. 2026 (arXiv:2602.16994), who tested untrained drafts only.

**Draft–Teacher:** Qwen3-0.6B/8B (primary, 3 ckpts) + 1.7B/32B (cross-pair) | **Eval:** math_eval + olympiad_eval, n=100, seed 123, L=8, K=1–4 | **L1 variants:** `dL{2..6}` (fixed), `dAdapt` (entropy-lagged, deprecated), `dTau` (τ-lagged, no signal) | Impl: `delayed_draft.py`, `--L1`.
**Data source:** §0 cross-checkpoint/both-pair sweep from [`per_checkpoint_sweeps_2026-07/`](../Results/per_checkpoint_sweeps_2026-07/); §1+ (0.6/8, 3 ckpts, single seed) from [`jsd_enrich_ddte_results.csv`](../Results/jsd_enrich_ddte_results.csv) (filter to s123 — file also has s456/ttemp1.5). Metric: **DDTE uplift = best-dL traversal BE − same ckpt's base (L1=0) traversal BE**, matched K.

---

## 0. Headline: DDTE across all trained drafts, both pairs (2026-07)

**Finding 1 — DDTE helps every trained draft; uplift grows with K, ≈0 at K=1.**

| pair / dataset | K=2 uplift | K=3 uplift | K=4 uplift |
|---|---|---|---|
| 0.6B/8B math | −0.03 to +0.28 | +0.04 to +0.38 | +0.11 to +0.38 |
| 1.7B/32B math | +0.0 to +0.36 | +0.20 to +0.71 | +0.45 to +0.79 |
| 1.7B/32B olympiad | +0.03 to +0.24 | +0.29 to +0.55 | +0.27 to +0.53 |

Uplift ≈0 at K=1 (no branching to delay) and rises monotonically with K — deeper trees make branch placement matter more. This is the key extension of Thomas et al.: DDTE transfers cleanly to distilled, enriched, PO- and even collapsed drafts, and to a second (wider) capacity gap.

**Finding 2 — uplift is *inversely* related to base draft quality (a delta, NOT an absolute reordering).** Weaker base → bigger uplift (1.7/32 math K4: jsd +0.785 vs stronger enrich_K3 +0.594; collapsed log-tree up to +0.71). **But absolute best-DDTE BE stays two-tier:**

| 1.7B/32B math, best-DDTE BE | K3 | K4 |
|---|---|---|
| jsd / enrich / po_nss / po_logprob + DDTE (well-trained) | 6.19–6.33 | 6.20–**6.37** |
| traversal_log / nss_log + DDTE (collapsed) | 5.70–5.90 | 5.77–5.83 |

A collapsed draft + DDTE lands ~0.5 BE **below** any well-trained draft + DDTE → **DDTE does not replace training; the top BE needs both.** (0.6/8: jsd base 5.83 → +enrich-train 5.99 → +DDTE 6.37 [enrich_K6]; enrich+DDTE 6.23–6.37 beats jsd+DDTE 6.12. At 1.7/32 the enrich edge vanishes: jsd+DDTE 6.37 ties enrich+DDTE 6.27 — same capacity-vs-depth effect as without DDTE.)

![DDTE lift per checkpoint: base traversal vs best delayed-branching, per pair (K=4, math_eval, single seed)](ddte_block_efficiency_3d.png)

*Base traversal (grey) vs best-DDTE (blue) per checkpoint, K=4 math. Left 0.6/8, right 1.7/32; annotations = uplift. Two tiers visible on the right. Single seed (s123).*

**Caveat:** all single-seed. Finding 1 is robust (~15 checkpoints, 2 pairs). Finding 2's *within-tier* orderings (jsd vs enrich at 1.7/32; which of jsd/po tops K3) are inside one-seed noise — don't rank within the well-trained tier without a second seed.

---

## 1. Detailed study (0.6B/8B, 3 checkpoints, single seed)

Two training tiers × eval-time DDTE: `jsd_flat`, `enrich_M1`, `enrich_M3`. Best rows (bold = ckpt/verifier peak):

**Specinfer — DDTE reshapes the landscape (steep L1 climb):**

| K | base (L1=0) | best M3 | M3−flat @ best | notes |
|---|---|---|---|---|
| 3 | flat 5.35 / M3 4.79 | **M3 dL5 = 6.015** | +0.27 | +1.22 over M3 base 4.795 |
| 4 | flat 5.04 / M3 5.07 | **M3 dL5 = 5.866** | +0.44 | largest M3−flat gap |

M3−flat grows monotonically with L1 (K=4: +0.03 @dL2 → +0.27 @dL4 → +0.44 @dL5). flat/M1 peak at dL6, M3 at dL5.

**Traversal — DDTE barely reshapes it, but the ceiling is highest:**

| K | base | best | value |
|---|---|---|---|
| 3 | flat 5.89 / M3 6.22 | **M3 dL4** | 6.361 (dL4 > dL5 all ckpts) |
| 4 | flat 5.66 / M3 5.89 | **M3 dL5** | **6.418** (global best; dL5 > dL4 for M3) |

**Cross-verifier ranking @ M3:** traversal+DDTE **6.418** > bv (std) 6.283 > traversal (std) 6.216 > specinfer+DDTE 6.015 > naive (std) 6.012. DDTE closes ~75% of the specinfer↔traversal gap (K=3: 1.42 @L1=0 → 0.64 @matched dL4 → 0.35 remaining) but does **not** make specinfer beat traversal.

**dTau (τ-lagged adaptive) — no detectable signal.** Specinfer: −0.28 to −0.58 vs best fixed L1 across all ckpts/K (ruled out). Traversal: within 0.04–0.17 at K2/3 but −0.22 at K4 (inconclusive). `dAdapt` (entropy-lagged, binary L1=0 fallback) deprecated — fallback triggered ~always on math.

**Throughput recovers at high L1.** Draft units = L1 + (L−L1)·K. dL4 K4 = 20 (~7–9 tok/s, Phase-1 overhead dominates); dL5 K4 = 17 (~14–17 tok/s, matches std specinfer); root L1=0 = 33 (~12–13). So dL5 is both best-BE and throughput-neutral.

---

## Mechanism, limitations & must-add (theory tail)

- **Why specinfer gains 5–7× more than traversal.** OT verifiers accept **top-down**: K root branches all sample the same high-prob root token → waste K passes in the shallow high-acceptance zone. Delaying to L1≈4–5 puts branches only where divergence grows (deep). Traversal accepts **bottom-up** — indifferent to where branches originate → DDTE only trims prunable branches.
- **Why specinfer+DDTE still loses to traversal.** (1) **Off-policy mismatch** — the draft was trained on root-branching trees, never on stem+delayed-branch prefixes. (2) A residual **verifier-algorithm gap** (top-down OT acceptance < bottom-up traversal for the same tree). (1) is the fixable one.
- **Optimal L1 ≈ mean(τ) − 1 (checkpoint-dependent).** M3 (K1 BE 6.00 → mean τ ≈5) peaks at L1=5; flat (K1 BE 5.67) indifferent L1=5–6. Stronger draft → divergence zone starts deeper → branch later. Deploy as a fixed per-checkpoint value; no adaptive rule needed (dTau adds variance without directional signal).
- **Relationship to enrich.** DDTE (verification/tree axis) composes with enrich (draft-training axis): DDTE picks the tree, enrich trains the draft that fills it. Uplift is slightly *smaller* on enrich drafts (less headroom — enrich already recovers part of what DDTE adds); reconciled with §0 Finding 2 (uplift is a delta, absolute BE stays tiered).
- **Top must-add:** (1) **second seed** — traversal effects are small enough that K=2 orderings may flip; (2) **training-time delayed expansion** (~50 lines in `build_training_data.py`) — removes off-policy mismatch, the central test of whether specinfer can catch traversal; (3) **MLP-selector single-pass draft** (from Rahul) for a valid throughput comparison vs the paper; (4) ≥1 more dataset.
