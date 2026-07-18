# Trained drafts trail the untrained baseline at high k

## Formula

`pass@k = 1 - C(n-c, k) / C(n, k)` (Chen et al. 2021), `n=64` samples/prompt,
`c` correct among them, averaged over 100 held-out prompts. At `k=n=64` this
is just "did ≥1 of 64 tries succeed" per prompt.

## Finding

[passk_by_checkpoint_math_eval_06b_8b.png](../results/passK/passk_by_checkpoint_math_eval_06b_8b.png):
`warm_anneal_lr1e5` and `jsd_mathhard_s123` beat the untrained `Qwen3-0.6B`
at `k=1` (0.178/0.153 vs 0.119) but the untrained curve is steeper and
crosses back above both by `k≈16-32`, finishing higher at `k=64`
(0.570 vs 0.530/0.510). `lr5e6` and `ce_lr1e5` stay above baseline the whole
curve — checkpoint-specific, not universal.

[passk_by_model_size_all_datasets.png](../results/passK/passk_by_model_size_all_datasets.png)
is the control: across model *size* (no training), every curve is strictly
monotonic at every k — confirms the pass@k math is correct, this is a real
training effect.

**Why:** `pass@1` tracks mean per-sample accuracy; `pass@64` tracks fraction
of prompts solved by *any* of 64 tries. Training toward the teacher's
preferred completion style raises accuracy where that style already works,
at the cost of the diverse alternate phrasings that let brute-force sampling
stumble onto a fix elsewhere — some prompts go from "occasionally solved" to
"never solved in 64 tries." Same signature as "RL sharpens, doesn't expand,
reasoning capacity" (pass@1↑, pass@256 flat/down vs. base) — here for
KL-based distillation, not RL.

## Dataset roles

| dataset | k=64, 0.6B→8B | reading |
|---|---|---|
| `olympiad_eval` | 0.240 → 0.330 | lowest absolute scores everywhere (32B teacher itself only hits 0.36) — most headroom, but not the most *sensitive* differentiator |
| `math_eval` | 0.570 → 0.630 | not saturated, not floor-bound — every training run's k=64 number sits somewhere between clearly-better and clearly-worse than baseline, so it's the dataset that actually separates good configs from bad ones |
| `gsm8k_eval` | 0.920 → 0.990 | near-ceiling for every model size — too easy to carry training signal |

## Did training move these? (checkpoint k=64 vs untrained 0.6B baseline)

| checkpoint | math_eval | gsm8k_eval | olympiad_eval |
|---|---|---|---|
| baseline (untrained 0.6B) | 0.570 | 0.920 | 0.240 |
| `lr5e6` | 0.610 (+0.040) | 0.920 (±0) | 0.330 (+0.090) |
| `ce_lr1e5` | 0.590 (+0.020) | 0.950 (+0.030) | 0.310 (+0.070) |
| `lr1e5` | 0.560 (−0.010) | 0.880 (−0.040) | 0.310 (+0.070) |
| `warm_anneal_lr1e5` | 0.530 (−0.040) | 0.930 (+0.010) | 0.230 (−0.010) |
| `jsd_mathhard_s123` | 0.510 (−0.060) | 0.920 (±0) | 0.270 (+0.030) |

Pattern: **every** checkpoint improves or holds on `olympiad_eval` — matches
it having the most headroom, easiest dataset to show *some* gain on.
`math_eval` is where the checkpoints actually split — `lr5e6`/`ce_lr1e5`
genuinely improve on the untrained draft, `warm_anneal`/`jsd_mathhard`
regress below it. `gsm8k_eval` barely moves either direction — consistent
with it already being saturated regardless of training.

## Practical read

Track `math_eval` pass@64 as the primary go/no-go regression check —
it's the only one of the three that actually distinguishes a good
training run from a bad one.
