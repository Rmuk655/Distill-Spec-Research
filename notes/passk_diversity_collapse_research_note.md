# Pass@k at high k: distillation can leave the draft worse than doing nothing — and math_eval isn't the noise-floor-gated differentiator

## Formula

`pass@k = 1 - C(n-c, k) / C(n, k)` (Chen et al. 2021), `n=64` samples/prompt,
`c` correct among them, averaged over 100 held-out prompts. At `k=n=64` this
is just "did ≥1 of 64 tries succeed" per prompt.

## Finding 1 — training can make a checkpoint trail its own untrained baseline at high k

[passk_by_checkpoint_math_eval_06b_8b.png](../results/passK/passk_by_checkpoint_math_eval_06b_8b.png):
`warm_anneal_lr1e5` and `jsd_mathhard_s123` beat the untrained `Qwen3-0.6B`
at `k=1` (0.178/0.153 vs 0.119) but the untrained curve is steeper and
crosses back above both by `k≈16-32`, finishing higher at `k=64`
(0.570 vs 0.530/0.510). `lr5e6` and `ce_lr1e5` stay above baseline the whole
curve — checkpoint-specific, not universal.

[passk_by_model_size_all_datasets.png](../results/passK/passk_by_model_size_all_datasets.png)
is the control: across model *size* (no training), every curve is strictly
monotonic at every k — confirms the pass@k math is correct, this is a real
training effect, not a computation bug.

**Why:** `pass@1` tracks mean per-sample accuracy; `pass@64` tracks fraction
of prompts solved by *any* of 64 tries. Training toward the teacher's
preferred completion style raises accuracy where that style already works,
at the cost of the diverse alternate phrasings that let brute-force sampling
stumble onto a fix elsewhere — some prompts go from "occasionally solved" to
"never solved in 64 tries." Same signature as "RL sharpens, doesn't expand,
reasoning capacity" (pass@1↑, pass@256 flat/down vs. base) — here for
KL-based distillation, not RL.

## Main lesson — trained vs untrained draft vs teacher

Distillation training exists to close the pass@64 gap between the
untrained draft and the teacher. That's not what happens reliably. Using
the untrained-0.6B → teacher-8B endpoints already in the table above as
the gap, and % gap closed = (checkpoint − untrained) / (teacher − untrained):

| checkpoint | math_eval | gsm8k_eval | olympiad_eval |
|---|---|---|---|
| `lr5e6` | 67% | 0% | **100%** |
| `ce_lr1e5` | 33% | 43% | 78% |
| `lr1e5` | **−17%** | **−57%** | 78% |
| `warm_anneal_lr1e5` | **−67%** | 14% | **−11%** |
| `jsd_mathhard_s123` | **−100%** | 0% | 33% |

Negative = the checkpoint ends up *further from the teacher than the
untrained draft already was* — not a small effect. `jsd_mathhard_s123` on
`math_eval` gives back a full gap-width; `lr1e5` on `gsm8k_eval` gives back
more than half. No checkpoint here closes gap on all three datasets except
`ce_lr1e5`, and even it never gets above 78%.

**So: pass@1 improving is not evidence the draft moved toward the
teacher's actual behavior at high k.** Training can leave the draft
*worse* than doing nothing, depending on dataset — check pass@64 gap-closed
per dataset before calling a checkpoint an improvement, don't infer it
from pass@1 or from block_eff alone.

## Finding 2 — which dataset actually differentiates configs needs a noise floor, not eyeballing

Absolute k=64 level by dataset, 0.6B→8B:

| dataset | k=64, 0.6B→8B | headroom |
|---|---|---|
| `olympiad_eval` | 0.240 → 0.330 | lowest absolute scores everywhere (even the 32B teacher only hits 0.36) — most headroom |
| `math_eval` | 0.570 → 0.630 | mid-range, not saturated, not floor-bound |
| `gsm8k_eval` | 0.920 → 0.990 | near-ceiling for every model size |

A first pass ranked these by eyeballing raw deltas against one baseline
checkpoint, no noise floor — that ranking (below) turned out to be wrong
once actually checked against a derived floor.

**What a real per-k/per-dataset noise floor shows instead:** derive
floor[k] = 2×std(pass@k) across the flat low-LR `prob` cluster (same method
as `scripts/analyze_passk_movement.py`), then compare every offline-evaluated
checkpoint's real `ckpt_best` pass@k (`results/passK/passk_hparam_sweep.csv`,
27 checkpoints) against jsd's actual deployed pick:

- **`math_eval`: 0/27** configs clear the floor at any k. The floor itself
  is wide (0.04–0.09) — wide enough to swamp raw deltas that look like a
  real split by eye. This dataset does **not** currently distinguish good
  configs from bad ones once noise is accounted for.
- **`gsm8k_eval`: ~15/27** configs clear the floor at k8–k64, several by
  2–3x the floor size (deltas 0.04–0.08 vs floor 0.02–0.04) — despite
  `gsm8k_eval` looking "saturated and uninformative" by raw absolute level,
  it is currently the most robust floor-clearing differentiator found.
- **`olympiad_eval`: 2/27** clear, and only marginally.

Caveat: even jsd's own `lr1e-6` sibling clears jsd's `lr1e-5` deployed
pick's k64 by 0.04 — some of the `gsm8k_eval` signal may be ceiling-effect
noise on an already-saturated benchmark rather than a clean training
effect. Read "15/27 clear the floor" as directional evidence, not as 15
individually-confirmed wins.

For reference, the original raw-delta table this correction replaces
(checkpoint k=64 vs untrained 0.6B baseline, no floor applied):

| checkpoint | math_eval | gsm8k_eval | olympiad_eval |
|---|---|---|---|
| baseline (untrained 0.6B) | 0.570 | 0.920 | 0.240 |
| `lr5e6` | 0.610 (+0.040) | 0.920 (±0) | 0.330 (+0.090) |
| `ce_lr1e5` | 0.590 (+0.020) | 0.950 (+0.030) | 0.310 (+0.070) |
| `lr1e5` | 0.560 (−0.010) | 0.880 (−0.040) | 0.310 (+0.070) |
| `warm_anneal_lr1e5` | 0.530 (−0.040) | 0.930 (+0.010) | 0.230 (−0.010) |
| `jsd_mathhard_s123` | 0.510 (−0.060) | 0.920 (±0) | 0.270 (+0.030) |

This table is kept only as a historical record of the eyeballed read that
did not survive a proper floor check — do not use it as a go/no-go signal.

## Practical read

- Don't use `math_eval` pass@64 movement as a go/no-go check without a
  derived noise floor — the raw deltas above look meaningful but none
  clear a real floor.
- `gsm8k_eval` at k32/k64, floor-gated, is currently the most robust
  differentiator available — with the ceiling-effect caveat above still
  in force.
- Re-derive the floor whenever the checkpoint set changes; it is not a
  fixed constant, and it can shift meaningfully as new runs join the
  flat low-LR reference cluster.
