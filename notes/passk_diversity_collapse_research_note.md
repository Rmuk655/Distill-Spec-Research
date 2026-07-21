# Pass@k diversity collapse — and why math_eval isn't the differentiator once you add a noise floor

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

## Correction — the ranking above had no noise floor (2026-07-21)

The "did training move these" table used raw deltas against one baseline,
eyeballed, no noise floor. Re-checked against a derived per-k/per-dataset
floor (2×std across the flat low-LR `prob` cluster, same method as
`scripts/analyze_passk_movement.py`), comparing every offline-evaluated
checkpoint's real `ckpt_best` pass@k (`results/passK/passk_hparam_sweep.csv`,
27 checkpoints) against jsd's actual deployed pick:

- `math_eval`: **0/27** configs clear the floor at any k — the floor itself
  is wide (0.04–0.09), wide enough to swamp every raw delta in the table
  above. The "math_eval is where checkpoints split" read below was never
  verified against noise.
- `gsm8k_eval`: **~15/27** configs clear the floor at k8–k64, several by
  2–3x the floor size (deltas 0.04–0.08 vs floor 0.02–0.04) — the opposite
  of "barely moves."
- `olympiad_eval`: only 2/27 clear, and marginally.

Caveat: even jsd's own `lr1e-6` sibling clears jsd's `lr1e-5` pick's k64 by
0.04, so part of the `gsm8k_eval` signal may be ceiling-effect noise on an
already-saturated benchmark rather than a clean training effect — read the
15/27 count directionally, not as 15 confirmed wins.

## Practical read (revised)

`math_eval` pass@64 movement in the table above was never noise-gated —
do not use it as a go/no-go check without deriving a floor first.
Floor-gated, `gsm8k_eval` at k32/k64 is currently the most robust
differentiator found so far (caveat above still applies). Re-derive the
floor whenever the checkpoint set changes — it is not a fixed constant.
