# Trained drafts trail the untrained baseline at high k

## The metric

`pass@k`: probability that at least one of `k` sampled completions is correct,
estimated unbiasedly (Chen et al. 2021, Codex) from `n=64` samples per prompt
with `c` correct among them:

```
pass@k = 1 - C(n-c, k) / C(n, k)
```

averaged over the 100-prompt held-out set. At `k=n=64` this collapses to a
binary per-prompt indicator: 1 if `c ≥ 1` (at least one of the 64 draws
succeeded), 0 if `c = 0`. So `pass@64` = fraction of prompts the model can
solve *at all* given generous sampling budget, while `pass@1` = mean
per-sample accuracy across all `n×prompts` draws.

## The finding

See [passk_by_checkpoint_math_eval_06b_8b.png](passk_by_checkpoint_math_eval_06b_8b.png):
`po_mh_prob_warm_anneal_lr1e5` and `jsd_mathhard_s123` both start **above**
the untrained `Qwen3-0.6B` draft at `k=1` (0.178/0.153 vs 0.119) but the
untrained draft's curve is steeper and crosses back above both by `k≈16-32`,
finishing higher at `k=64` (0.570 vs 0.530/0.510). `po_mh_prob_lr5e6` and
`po_mh_prob_ce_lr1e5`, by contrast, stay above the untrained baseline across
the whole curve — this isn't universal to "any trained checkpoint," it's
specific to which training run.

[passk_by_model_size_all_datasets.png](passk_by_model_size_all_datasets.png)
is the control: across model *sizes* (no training involved), every curve is
strictly monotonic in size at every k — confirming the pass@k computation
itself is correct and this crossover is a real training effect, not a
measurement artifact.

## Why: distillation narrows the response distribution

`pass@1` is dragged up by the *average* per-sample success rate; `pass@64`
is dragged down whenever a chunk of prompts goes from "occasionally solved
across many diverse attempts" to "never solved in any of 64 attempts."
Training toward matching the teacher's preferred completions concentrates
probability mass onto the teacher's dominant solution style — raising
per-sample accuracy on prompts already in that style's reach, while some
other prompts lose the diverse alternate phrasings/approaches that let
brute-force sampling stumble onto a correct path. Net effect: better
single-shot output, worse ceiling under heavy sampling. This is the same
"RL/distillation sharpens rather than expands capability" pattern reported
elsewhere for RL-tuned reasoning models (pass@1 up, pass@256 flat or down
vs. the base model) — this project's data shows the identical signature for
KL-based distillation, not just RL.

## Practical read

Not every training run does this (`lr5e6`/`ce_lr1e5` avoid it) — worth
treating "does pass@64 stay above the untrained baseline" as a standing
regression check on future checkpoints, not assuming it's automatic.
