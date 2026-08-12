# LinkedIn series: building a speculative decoding research platform

This folder holds the draft posts for a 12 part series. Post 1 is the platform overview (the hub). Posts 2 to 11 each expand one paragraph of that overview into its own post. The series is organized along two axes at once: breadth, the range of CS and ML disciplines a real research platform actually touches, and depth, how far into systems thinking each one goes, since that is the thread tying every post back together regardless of which discipline it started from.

## Reading order

1. Platform overview (the hub)
2. Isolating the metric: why block efficiency, not tokens per second. Ends on the partial derivative framing, hold every other variable still, let only the one you care about move.
3. **[Needs a full rewrite, see below]** How do you know an improvement is real: sample sizes, seeds, and what beating a baseline actually requires.
4. The checkpoint that scored differently every run
5. The 0.2 gain that was a rounding error
6. Fitting a 32B model on one GPU
7. What a KV cache is and why vLLM exists
7a. Most people test one loss against one verifier, we tested 34 against 9
8. The loss that matched my metric and still lost
9. Two training ideas that failed
10. The one change that beat the baseline
11. When one metric was not enough

## Post 3, reworked: status and what it needs to become

`03_noise_floor.md` as it stands only covers the single seed noise floor idea, run near identical configs, measure the spread, do not trust a delta smaller than that. The reworked version needs to go further, and it picks up directly where post 2 left off:

Post 2's ending is that isolating one variable is a partial derivative, freeze everything else, let one thing move. Post 3's opening move is the complication: even after you freeze everything else, the one thing you let move is still noisy on its own, because speculative decoding's acceptance test is a probabilistic draw, not a deterministic function of the checkpoint. So holding every other variable constant does not hand you a clean answer, it hands you a distribution, and the real statistics lesson is what it actually takes to tell a genuine improvement apart from that distribution's own width.

Ground this in what the project actually had to do, escalating in that order:
- A single comparison at n=100 prompts is not enough, the noise floor itself is roughly 0.10 to 0.15 in block efficiency at that sample size.
- One seed is not enough either. The depth weighted curriculum's own headline result, a claimed +0.29 win, turned out to be a mislabeled K compared against the wrong baseline cell, the real delta was +0.03, inside noise.
- One dataset is not enough. Results that held on math_hard did not always hold on olympiad_eval.
- One model pair is not enough. Enrichment's clean win at the 0.6B/8B pair did not cleanly replicate at 1.7B/32B, sign flipping across K on a single seed, and that pair's own noise floor was never even characterized because everything there was n=1.
- Eventually, one loss family is not enough either, since a pattern that looks like a real effect for one objective can be a property of that specific objective's variance, not evidence about training methods in general.

The honest through line: every one of these escalations exists because of the same root cause, the randomness inherent in autoregressive sampling and in the accept or reject draw itself, which survives no matter how carefully everything else is held fixed. That is the real statistics lesson, and it should be grounded entirely in the project's own record of getting this wrong before getting it right, not presented as something obvious from the start.

Grounding sources for the rewrite: project report noise floor SE figures; `depth_weight_research_note.md` for the +0.29 to +0.03 correction; `enrich_research_note.md` for the two seed n=1000 study (s123 vs s456) and the 0.6/8 vs 1.7/32 replication failure; the explicit n=1 caveat on every 1.7B/32B number; `passk_prob_vs_jsd_research_note.md` for the "no significant signal across most settings" finding, which is the same lesson applied to an entire study, not just one metric.

## Breadth and depth: what discipline each post actually comes from

The point of laying this out is that the series is not one long systems diary, it is a spread across real CS and ML fundamentals, with systems as the depth axis running underneath nearly all of them.

| Post | Breadth: primary discipline | Depth: the systems thread underneath it |
|---|---|---|
| 2 | Computer architecture (memory bandwidth, HBM) and experimental design | isolating a variable inside a live running system |
| 3 | Statistics (sample size, variance, seeds, replication) | applying that rigor to a nondeterministic live system, not a textbook dataset |
| 4 | Operating systems (process and resource contention, scheduling) | GPU sharing across concurrent processes |
| 5 | Scientific computing and numerical methods (floating point, kernel determinism) | numerics inside a live training and eval pipeline |
| 6 | Computer architecture and capacity planning | scaling one pipeline across four hardware tiers |
| 7 | Operating systems and systems software (caching, memory management, serving) | building a real serving path, not just a training script |
| 7a | ML fundamentals (taxonomy of objectives and verifiers) | a breadth first survey before the deep dives that follow |
| 8 | ML fundamentals (loss functions, gradient behavior) | why a loss trains or does not, independent of its formula |
| 9 | ML fundamentals (reinforcement learning, curriculum learning) | same, from the failure side |
| 10 | ML fundamentals plus systems (verifier restructuring) | combining a training lever and a verification lever |
| 11 | Statistics (a second metric, sampling based evaluation) | pass@k as its own measurement discipline, not just a footnote to block efficiency |

## Closing paragraph to append to the hub (Post 1)

Add this at the end of the platform overview so it points forward to the rest of the series:

> I am writing up the pieces of this platform one at a time, roughly one post a week. Each one takes a single line from above and tells the full story behind it: why block efficiency and not tokens per second, how I measured whether a result was real, the two measurement bugs that fooled me, the training objectives that failed and the one that worked, and what a second metric revealed that the first one hid. Follow along if you want the details.

## Voice rules (keep every post consistent)

1. Write like a student explaining what he learned, not like a textbook. Short sentences. First person.
2. No hyphens, no contractions, no em dashes. These are the fastest way to sound generated.
3. Define any term the first time it appears, in plain words.
4. Start in the middle of the problem, not with a grand intro.
5. Show what failed. Every post should admit at least one thing I got wrong.
6. Every number is real and traceable to the notes, the results CSVs, or the git log. No invented figures.
7. Drop hype words: crucial, testament, landscape, robust, seamless, delve.
8. Only reference already-published posts, never a later one, with no exceptions. Posts go out sequentially, so a forward reference like "post 10 covers this," or even a "Next: ..." teaser in the footer, is a dead pointer to a reader who has no way to see it yet. The series footer should only say which post this is, nothing about what comes after.

## Grounding sources per post

- Post 2: profiling row in results/per_checkpoint_sweeps_2026-07/jsd_mathhard_s123.csv; the 8B vs 32B teacher timing and GPU utilization comparison; interview prep section on tokens/sec.
- Post 3: see the rework section above for the full grounding list.
- Post 4: telemetry.py cpu_util_avg_pct; the one eval per GPU fix.
- Post 5: git commit "fix(eval): default --force_attn to sdpa"; eval attn backend notes.
- Post 6: interview prep memory section; optim_8bit and setup commits.
- Post 7: passk pipeline commits (async vLLM, isolated venv); passk_utils.py.
- Post 7a: verifiers/verifier.py docstring (the 9 verifiers, OTLP vs non-OTLP); losses/tree.py and losses/flat.py registries for the exact loss count and categories.
- Post 8: tree_losses_research_note.md, prefix_overlap_research_note.md.
- Post 9: tree_losses_research_note.md (tree_pg), depth_weight_research_note.md.
- Post 10: enrich_research_note.md, ddte_research_note.md.
- Post 11: passk_prob_vs_jsd_research_note.md; the project report closing.
