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
- **Trained on `dapo_math_train` instead of `math_hard`, prob genuinely beats
  the untrained student on `olympiad_eval`, jsd only ties it** — a real
  win, not just a jsd-vs-prob artifact. The matching gap on `math_eval` is
  hollow by comparison: both checkpoints lose to the untrained student there
  (see Anomalies), so "prob beats jsd" on `math_eval` is a race between two
  checkpoints that both failed to learn on that dataset.
- **Several hyperparameters that showed no effect on BE do show a real
  pass@k effect, and it isn't always the BE-preferred value that wins:**
  warmup 20% clearly beats jsd on `gsm8k` where 5%/10% don't as cleanly;
  `lr_min_ratio=0.1` beats jsd where `0.01` doesn't at all; weight decay
  `0.001` beats jsd nowhere despite sitting between two values that both
  do; and CE-anneal `anneal_steps=3000` — the actual BE-best pick — beats
  jsd nowhere on pass@k, while `4000` (worse on BE) beats it at two of
  three k's. `grad_clip`, `M`, `teacher_temp`, and root spacing offset show
  no consistent pass@k lever. Root spacing `N` is the other confirmed
  effect, though inconsistent across estimator variants (see below).
- **`best_val_block_eff` is a training-time-only metric** (computed on
  `math_val` during training) — never computed on the three held-out sets.
  Don't compare it to held-out pass@k as if it were "BE on that dataset."

## Prob vs jsd, by dataset

| dataset | direction (of 38) | beats jsd beyond noise? | beats untrained student? | read |
|---|---|---|---|---|
| `math_eval` | 30–36/38 above jsd (k1–k32), 23/33 at k64 | never | yes, except 2 checkpoints that cross back below at high k (see Anomalies) | learning happens, jsd/prob tie |
| `olympiad_eval` | 32–37/38 above jsd (k1–k32), 23/34 at k64 | 3/38 configs | yes, no crossing cases documented here | learning happens, jsd/prob mostly tie |
| `gsm8k_eval` | consistently above | 14–22/38 configs, concentrated at k16/k32/k64 | yes, no crossing cases documented here | learning happens, and prob pulls ahead of jsd |

Charts (best-of-family, hindsight-selected by pass@k gap-closed — a
different, complementary selection to the BE-deployed-pick comparisons
below): [math_eval](../results/passK/passk_curves_all_math_eval.png) ·
[olympiad_eval](../results/passK/passk_curves_all_olympiad_eval.png) ·
[gsm8k_eval](../results/passK/passk_curves_all_gsm8k_eval.png).

**Trained on `dapo_math_train` instead of `math_hard`** (real deployment
picks per family, not cherry-picked after seeing pass@k):

| dataset | jsd vs student (k1→k64) | prob vs student (k1→k64) | prob vs jsd | beats jsd beyond noise? |
|---|---|---|---|---|
| `olympiad_eval` | +0.004 → −0.010, ties (not the Anomalies crossing) | +0.013 → +0.060, **wins outright, growing with k** | k16 +0.036, k32 +0.048, k64 +0.070 | yes, widening with k |
| `math_eval` | +0.006 → **−0.060, loses to student** (Anomalies crossing) | +0.028 → **−0.030, loses to student** (Anomalies crossing) | k16 +0.027, k64 +0.030 | right at the floor |
| `gsm8k_eval` | +0.047 → +0.000, ties by k64 | +0.076 → +0.010, **wins**, margin shrinks with k | k64 +0.010 | no |

This is the difference your question was pointing at: on `olympiad_eval`,
prob isn't just edging out jsd, it's genuinely learning something jsd isn't
— it beats the untrained student outright and the margin *grows* with k. On
`math_eval`, prob "beating jsd beyond noise" would mean nothing on its own,
since **both are losing to the untrained student there** — that's a race
between two checkpoints that both failed to learn on this dataset, not a
real win. `gsm8k_eval` sits in between: both beat the student, but the
margin over the *student* shrinks toward zero by k64 for both, even though
prob still edges jsd there by k64.

Charts: [math_eval](../results/passK/passk_dapo_jsd_vs_prob_math_eval.png) ·
[gsm8k_eval](../results/passK/passk_dapo_jsd_vs_prob_gsm8k_eval.png) ·
[olympiad_eval](../results/passK/passk_dapo_jsd_vs_prob_olympiad_eval.png).
Single seed, one checkpoint per family, same caveat as everywhere else here.

## Ablations tested — none changed the qualitative picture, except N

Two separate knobs get tested below, not one: **M** = number of teacher
continuations sampled per root (design doc §4, CLI flag `--prefix_M`).
**N** = spacing between roots along one teacher rollout (design doc §5,
CLI flag `--prefix_root_spacing`) — a *larger* N means *fewer, more
widely-spaced* roots, not more roots. "Beats jsd beyond noise on `gsm8k`"
below is shorthand for: that config's pass@k gap vs jsd on `gsm8k_eval` is
bigger than the noise floor at the k's where it's reported.

Deltas below are vs jsd on `gsm8k_eval` at k16/k32/k64, noise floor derived
per k from the flat low-LR `prob` cluster (2×std): 0.031/0.027/0.034. Every
family gets a chart per dataset (math_eval/gsm8k_eval/olympiad_eval), same
style as the dataset charts above, zoomed in to each family's own variants
plus its jsd baseline (not all 38 checkpoints):
[scripts/plot_ablation_families.py](../scripts/plot_ablation_families.py).

| variable | values tested | effect on pass@k | charts |
|---|---|---|---|
| grad_clip | 10, 100, 1000 | no consistent winner — only `gradclip100` beats jsd beyond noise on `gsm8k`, the other two don't | [math_eval](../results/passK/ablation_grad_clip_math_eval.png) · [gsm8k_eval](../results/passK/ablation_grad_clip_gsm8k_eval.png) · [olympiad_eval](../results/passK/ablation_grad_clip_olympiad_eval.png) |
| M (samples/root) | 4 (default), 8, 16 (warm), 16 (cold) | no trend with M; `M16_cold` fails to beat jsd where its warm-started twin did | [math_eval](../results/passK/ablation_M_samples_per_root_math_eval.png) · [gsm8k_eval](../results/passK/ablation_M_samples_per_root_gsm8k_eval.png) · [olympiad_eval](../results/passK/ablation_M_samples_per_root_olympiad_eval.png) |
| teacher_temp | 1.0 (default), 0.7, 0.5 | **the default beats jsd by the widest margin of the three (+0.038/+0.035/+0.040 at k16/32/64), `ttemp=0.5` also clears (+0.033/+0.027/+0.030), `ttemp=0.7` doesn't (+0.013/+0.018/+0.020, inside the floor).** With the default actually included, there's no support for "lowering teacher_temp helps" — the unmodified default is the strongest point, not the weakest. One confound: the default checkpoint used here also differs in `lr_min_ratio`/`weight_decay` from the two ttemp points (0.1/0.01 vs their bare CLI defaults), so this isn't a perfectly clean single-axis read, just the closest match available in the sweep. | [math_eval](../results/passK/ablation_teacher_temp_math_eval.png) · [gsm8k_eval](../results/passK/ablation_teacher_temp_gsm8k_eval.png) · [olympiad_eval](../results/passK/ablation_teacher_temp_olympiad_eval.png) |
| CE-anneal timing (lr=3e-6) | anneal vs no-anneal | both beat jsd beyond noise on `gsm8k` k32 similarly; anneal doesn't move the needle at this LR | [math_eval](../results/passK/ablation_CE_anneal_timing_lr3e-6_math_eval.png) · [gsm8k_eval](../results/passK/ablation_CE_anneal_timing_lr3e-6_gsm8k_eval.png) · [olympiad_eval](../results/passK/ablation_CE_anneal_timing_lr3e-6_olympiad_eval.png) |
| root spacing offset | fixed vs random | negligible vs plain N=16 | [math_eval](../results/passK/ablation_root_spacing_offset_math_eval.png) · [gsm8k_eval](../results/passK/ablation_root_spacing_offset_gsm8k_eval.png) · [olympiad_eval](../results/passK/ablation_root_spacing_offset_olympiad_eval.png) |
| dataset | `math_hard` vs `dapo_math_train` | changes *which* dataset the prob-jsd gap beats noise on — real, not noise | [math_eval](../results/passK/passk_dapo_jsd_vs_prob_math_eval.png) · [gsm8k_eval](../results/passK/passk_dapo_jsd_vs_prob_gsm8k_eval.png) · [olympiad_eval](../results/passK/passk_dapo_jsd_vs_prob_olympiad_eval.png) |
| N (root spacing) | 8, 16, 16+offset, 32 | N=16/16+offset/32 beat jsd beyond noise on `gsm8k` k32/k64; **N=8 doesn't beat it anywhere** | [math_eval](../results/passK/ablation_N_root_spacing_math_eval.png) · [gsm8k_eval](../results/passK/ablation_N_root_spacing_gsm8k_eval.png) · [olympiad_eval](../results/passK/ablation_N_root_spacing_olympiad_eval.png) |
| warmup % | 5, 10, 20 (lr=1e-5) | 20% beats jsd beyond noise on `gsm8k` at all three k's (k16/32/64); 10% beats it at two (k16/k32); 5% at one (k32 only) — directionally consistent with more warmup helping, unlike BE which showed no clear trend on this axis | [math_eval](../results/passK/ablation_warmup_pct_math_eval.png) · [gsm8k_eval](../results/passK/ablation_warmup_pct_gsm8k_eval.png) · [olympiad_eval](../results/passK/ablation_warmup_pct_olympiad_eval.png) |
| lr_min_ratio | 0.1 (default), 0.01 | 0.1 beats jsd beyond noise at all three k's; 0.01 beats it at none — a real divergence from the BE result, which showed no effect | [math_eval](../results/passK/ablation_lr_min_ratio_math_eval.png) · [gsm8k_eval](../results/passK/ablation_lr_min_ratio_gsm8k_eval.png) · [olympiad_eval](../results/passK/ablation_lr_min_ratio_olympiad_eval.png) |
| teacher top-k | 0 (default), 20, 50 | all three beat jsd beyond noise at all three k's, similar magnitude — no harm in pass@k despite BE showing "mild harm at higher k" | [math_eval](../results/passK/ablation_teacher_topk_math_eval.png) · [gsm8k_eval](../results/passK/ablation_teacher_topk_gsm8k_eval.png) · [olympiad_eval](../results/passK/ablation_teacher_topk_olympiad_eval.png) |
| weight decay | 0.0001, 0.001, 0.01 (default) | 0.0001 beats jsd beyond noise most strongly (all three k's, largest gaps), 0.01 also beats it at all three, **0.001 beats it at none** — non-monotonic, worse than both its neighbors | [math_eval](../results/passK/ablation_weight_decay_math_eval.png) · [gsm8k_eval](../results/passK/ablation_weight_decay_gsm8k_eval.png) · [olympiad_eval](../results/passK/ablation_weight_decay_olympiad_eval.png) |
| CE-anneal `anneal_steps` | 1500, 3000 (BE-best pick), 4000 | 1500 and 3000 beat jsd beyond noise at none of the three k's; 4000 beats it at two (k32/k64) — **the BE-optimal `anneal_steps` (3000) doesn't carry over to pass@k** | [math_eval](../results/passK/ablation_CE_anneal_steps_math_eval.png) · [gsm8k_eval](../results/passK/ablation_CE_anneal_steps_gsm8k_eval.png) · [olympiad_eval](../results/passK/ablation_CE_anneal_steps_olympiad_eval.png) |

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

Four separate variables get tangled together in the checkpoints tested here
— sampling method (tail-reuse vs fresh), M (continuations per root), whether
a CE-anneal is mixed in, and N (root spacing). Only five points were run, not
a full grid, so laid out with one column per variable, each row below
differs from *one specific other row* in exactly one variable — not from
every other row at once:

| method | N | M | CE-anneal | `gsm8k` Δ vs jsd (k16/k32/k64) | verdict |
|---|---|---|---|---|---|
| tail-reuse | 16 | 1 | no | beats jsd beyond noise at all three | wins |
| fresh | 16 | 1 | no | −0.009 to −0.040 | reverses, worse than jsd |
| fresh | 16 | 1 | yes (ceanneal3000) | −0.009 to +0.007 | ~ties jsd, doesn't beat it |
| fresh | 16 | 2 | no | ~0.000 by k64 | ties jsd, doesn't beat it |
| fresh | 32 | 1 | no | +0.030 at k32/k64 | **beats jsd beyond noise** |

Charts (all five lines together): [math_eval](../results/passK/ablation_tail_reuse_vs_fresh_math_eval.png) · [gsm8k_eval](../results/passK/ablation_tail_reuse_vs_fresh_gsm8k_eval.png) · [olympiad_eval](../results/passK/ablation_tail_reuse_vs_fresh_olympiad_eval.png).
Tail-reuse across N specifically (N=8/16/16-offset/32, method held fixed) —
same data as the N row in the ablations table above:
[math_eval](../results/passK/ablation_N_root_spacing_math_eval.png) · [gsm8k_eval](../results/passK/ablation_N_root_spacing_gsm8k_eval.png) · [olympiad_eval](../results/passK/ablation_N_root_spacing_olympiad_eval.png).

The four *valid* single-axis reads, each holding the other three columns fixed:

1. **Method** (tail-reuse vs fresh, N=16/M=1/no-anneal fixed): tail-reuse
   wins, fresh reverses. This is the headline result.
2. **M** (1 vs 2, fresh/N=16/no-anneal fixed): M=1 reverses, M=2 roughly
   ties jsd — doesn't reverse as badly, still doesn't beat it.
3. **CE-anneal** (on vs off, fresh/N=16/M=1 fixed): off reverses, on roughly
   ties jsd — closes most of the reversal, doesn't flip it into a win.
4. **N** (16 vs 32, fresh/M=1/no-anneal fixed): N=16 reverses, N=32 beats
   jsd beyond noise — the same N that makes tail-reuse work also makes
   fresh work.

Tail-reuse itself, across N=8/16/32 (a separate comparison, not part of the
table above): N=16 **+0.03–0.04** (beats jsd beyond noise), N=32
**+0.022–0.030** (beats jsd beyond noise), N=8 **~+0.02** (does not).

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
checkpoints do *not* cross on
[`gsm8k_eval`](../results/passK/passk_dapo_jsd_vs_prob_gsm8k_eval.png) or
[`olympiad_eval`](../results/passK/passk_dapo_jsd_vs_prob_olympiad_eval.png).
That rules out
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
