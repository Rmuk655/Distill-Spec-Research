# Pass@k: trained drafts vs teacher, and prob vs jsd

39 checkpoints, real `ckpt_best` offline eval, `n=64` samples/prompt, 100 held-out
prompts per dataset: [results/passK/passk_hparam_sweep.csv](../results/passK/passk_hparam_sweep.csv).
`pass@k = 1 - C(n-c,k)/C(n,k)` (Chen et al. 2021). Noise floor = 2×std(pass@k)
across the flat low-LR `prob` cluster, per k per dataset — a delta only counts
as real if it clears this.

## Summary

- **Training helps over the untrained student almost everywhere.** A small
  number of checkpoints cross back below the student at high k (see
  Anomalies) — checkpoint-specific, not a general consequence of training.
- **Prob beats jsd in direction on every dataset, but only `gsm8k_eval`
  clears the noise floor.** `math_eval` and `olympiad_eval` are directionally
  prob-favoring but not usable as a go/no-go signal on their own.
- **Trained on `dapo_math_train` instead of `math_hard`, the gap opens up on
  `olympiad_eval` too** — the first fair jsd-vs-prob pair in this sweep with
  a floor-clearing gap outside `gsm8k`.
- **Of every hyperparameter ablated (grad_clip, M, teacher_temp, CE-anneal
  timing, root spacing offset), none changed the qualitative picture.** The
  one exception is root spacing `N`, and even there the effect is
  inconsistent across estimator variants (see below).
- **`best_val_block_eff` is a training-time-only metric** (computed on
  `math_val` during training) — never computed on the three held-out sets.
  Don't compare it to held-out pass@k as if it were "BE on that dataset."

## Prob vs jsd, by dataset

| dataset | direction (of 38) | beats jsd beyond noise? | read |
|---|---|---|---|
| `math_eval` | 30–36/38 above jsd (k1–k32), 23/33 at k64 | never | directional only |
| `olympiad_eval` | 32–37/38 above jsd (k1–k32), 23/34 at k64 | 3/38 configs | directional only |
| `gsm8k_eval` | consistently above | 14–22/38 configs, concentrated at k16/k32/k64 | the one real pattern |

Charts (best-of-family, hindsight-selected by pass@k gap-closed — a
different, complementary selection to the BE-deployed-pick comparisons
below): [math_eval](../results/passK/passk_curves_all_math_eval.png) ·
[olympiad_eval](../results/passK/passk_curves_all_olympiad_eval.png) ·
[gsm8k_eval](../results/passK/passk_curves_all_gsm8k_eval.png).

**Trained on `dapo_math_train` instead of `math_hard`** (real deployment
picks per family, not cherry-picked after seeing pass@k):

| dataset | Δ (prob − jsd) | beats jsd beyond noise? |
|---|---|---|
| `olympiad_eval` | k16 +0.036, k32 +0.048, k64 +0.070 | yes, widening with k |
| `math_eval` | k16 +0.027, k64 +0.030 | right at the floor |
| `gsm8k_eval` | k64 +0.010 | no |

Charts: [math_eval](../results/passK/passk_dapo_jsd_vs_prob_math_eval.png) ·
[gsm8k_eval](../results/passK/passk_dapo_jsd_vs_prob_gsm8k_eval.png) ·
[olympiad_eval](../results/passK/passk_dapo_jsd_vs_prob_olympiad_eval.png).
Single seed, one checkpoint per family, same caveat as everywhere else here —
but the first dataset in the whole sweep where prob clears the floor
outside `gsm8k`.

## Ablations tested — none changed the qualitative picture, except N

Two separate knobs get tested below, not one: **M** = number of teacher
continuations sampled per root (design doc §4, CLI flag `--prefix_M`).
**N** = spacing between roots along one teacher rollout (design doc §5,
CLI flag `--prefix_root_spacing`) — a *larger* N means *fewer, more
widely-spaced* roots, not more roots. "Beats jsd beyond noise on `gsm8k`"
below is shorthand for: that config's pass@k gap vs jsd on `gsm8k_eval` is
bigger than the noise floor at the k's where it's reported.

| variable | values tested | effect on pass@k |
|---|---|---|
| grad_clip | 10, 100, 1000 | no consistent winner — only `gradclip100` beats jsd beyond noise on `gsm8k`, the other two don't |
| M (samples/root) | 4, 8, 16 (warm), 16 (cold) | no trend with M; `M16_cold` fails to beat jsd where its warm-started twin did |
| teacher_temp | 0.5, 0.7 | `ttemp=0.5` beats jsd beyond noise on `gsm8k`, `0.7` doesn't. **Gap: the default (`ttemp=1.0`) was never run through pass@k, and only 2 non-default points were tested — not enough points to see a real trend or rule one out. Adding the default plus a third temp value is the natural next step before drawing any conclusion here.** |
| CE-anneal timing (lr=3e-6) | anneal vs no-anneal | both beat jsd beyond noise on `gsm8k` k32 similarly; anneal doesn't move the needle at this LR |
| root spacing offset | fixed vs random | negligible vs plain N=16 |
| dataset | `math_hard` vs `dapo_math_train` | changes *which* dataset the prob-jsd gap beats noise on — real, not noise |
| N (root spacing) | 8, 16, 16+offset, 32 | N=16/16+offset/32 beat jsd beyond noise on `gsm8k` k32/k64; **N=8 doesn't beat it anywhere** |

## Multi-root: two different ways to build the continuation at each root

Both variants below use the same N (root spacing, §5). They differ in
*where the teacher continuation at each root comes from*:

- **Tail-reuse** (the cheap approximation): the teacher generates ONE long
  rollout up front; every root along it just reuses the remaining tail of
  that same rollout as its "continuation." One teacher `generate()` call
  total, but continuations at different roots are correlated (they're all
  slices of the same sampled path) — not the independent-per-root sampling
  the design doc's unbiasedness proof (§5) actually assumes.
- **Fresh** (`freshM{1,2}`, doc-faithful): the teacher draws a genuinely new,
  independent continuation sample at *each* root separately, matching the
  doc's literal spec — at the cost of extra teacher calls.

Tail-reuse, `gsm8k` k32/k64 Δ vs jsd: N=16 **+0.03–0.04** (beats jsd beyond
noise), N=32 **+0.022–0.030** (beats jsd beyond noise), N=8 **~+0.02** (does
not).

Fresh at the **same N=16 does not reproduce tail-reuse's win** — it reverses
(Δ −0.009 to −0.040 at k8–k64):

| variant | N | gsm8k vs jsd | verdict |
|---|---|---|---|
| tail-reuse (baseline) | 16 | beats jsd beyond noise at k16/k32/k64 | wins |
| `freshM1` bare | 16 | −0.009 to −0.040 | reverses, worse than jsd |
| `freshM1_ceanneal3000` | 16 | −0.009 to +0.007 | ~ties jsd, doesn't beat it |
| `freshM2` (M=2) | 16 | ~0.000 by k64 | ties jsd, doesn't beat it |
| `freshM1` | 32 | +0.030 at k32/k64 | **beats jsd beyond noise — reproduces the win** |

**Conclusion: the cheap approximation (tail-reuse) is the one that currently
works. The theoretically-correct estimator (fresh) only matches it once N is
increased to 32 — it does not reproduce the win at the same N=16 tail-reuse
uses.** Two readings are both consistent with this, and it isn't settled
which is right: either fresh genuinely needs more roots to bring its own
variance down to a competitive level, or the correlation tail-reuse
introduces (repeatedly scoring against the same sampled rollout) is doing
something actively useful rather than just being a passable approximation.
Single seed — worth confirming before leaning on either explanation.

## Anomalies

**A few trained checkpoints cross back below the untrained student at high
k**, despite leading at k=1. On `math_eval`: `warm_anneal_lr1e5` and
`jsd_mathhard_s123` lead at k=1 (0.178/0.153 vs 0.119) but the untrained
curve is steeper and passes them by k≈16–32, finishing higher at k=64 (0.570
vs 0.530/0.510). Other checkpoints on the same dataset (`lr5e6`, `ce_lr1e5`)
don't show this — so it's checkpoint-specific, not caused by training itself.
[Chart](../results/passK/passk_by_checkpoint_math_eval_06b_8b.png). Control
confirming the pass@k math itself is fine: across model
*size* (no training) every curve is strictly monotonic —
[chart](../results/passK/passk_by_model_size_all_datasets.png).

**Same crossing shows up on the dapo-trained checkpoints, `math_eval`
only.** Both jsd and prob trained on dapo lead the student at k=1 (+0.006,
+0.028) but finish k=64 below it (−0.060, −0.030) —
[chart](../results/passK/passk_dapo_jsd_vs_prob_math_eval.png). The same two
checkpoints do *not* cross on `gsm8k_eval` or `olympiad_eval`. That rules out
a generic "training narrows diversity" explanation, since that would show up
on every dataset, not just one. It matches the dataset mismatch already
established for dapo (trails on `math_eval`, leads on `gsm8k_eval`): dapo's
training distribution preserves sample diversity on
`gsm8k_eval`/`olympiad_eval`-style problems but narrows it specifically on
`math_eval`-style ones. **This mechanism is a hypothesis inferred from the
curve shape, not a measured result** — the direct test would be comparing
per-prompt output diversity between the dapo- and math_hard-trained
checkpoints on the `math_eval` prompts where dapo fails at high k.

## Dataset headroom (k=64, untrained 0.6B → teacher 8B)

| dataset | 0.6B → 8B | note |
|---|---|---|
| `olympiad_eval` | 0.240 → 0.330 | lowest absolute scores everywhere (even the 32B teacher only hits 0.36) |
| `math_eval` | 0.570 → 0.630 | mid-range, not saturated — but nothing clears noise here regardless |
| `gsm8k_eval` | 0.920 → 0.990 | near-ceiling, yet the one dataset with a real prob-vs-jsd pattern |

## Practical read

- Judge a checkpoint by pass@k gap-closed to the teacher, per dataset — not
  by pass@1 or training-time block_eff alone. Charts:
  [math_eval](../results/passK/gap_closed_math_eval.png) ·
  [gsm8k_eval](../results/passK/gap_closed_gsm8k_eval.png) ·
  [olympiad_eval](../results/passK/gap_closed_olympiad_eval.png).
- `gsm8k_eval` beats jsd beyond the noise floor for `math_hard`-trained
  checkpoints. `olympiad_eval` beats jsd beyond the noise floor for
  `dapo`-trained checkpoints. `math_eval` beats it for neither.
