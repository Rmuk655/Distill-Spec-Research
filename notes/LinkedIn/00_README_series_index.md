# LinkedIn series: building a speculative decoding research platform

This folder holds the draft posts for a 11 part series. Post 1 is the platform overview you already wrote (the hub). Posts 2 to 11 each expand one paragraph of that overview into its own short post.

## Reading order

1. Platform overview (the hub, your existing article)
2. Why I stopped trusting tokens per second
3. Measuring the noise floor
4. The checkpoint that scored differently every run
5. The 0.2 gain that was a rounding error
6. Fitting a 32B model on one GPU
7. What a KV cache is and why vLLM exists
8. The loss that matched my metric and still lost
9. Two training ideas that failed
10. The one change that beat the baseline
11. When one metric was not enough

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

- Post 2: profiling row in results/per_checkpoint_sweeps_2026-07/jsd_mathhard_s123.csv; interview prep section on tokens/sec.
- Post 3: noise floor SE from the project report; the +0.29 to +0.03 correction in depth_weight_research_note.md.
- Post 4: telemetry.py cpu_util_avg_pct; the one eval per GPU fix.
- Post 5: git commit "fix(eval): default --force_attn to sdpa"; eval attn backend notes.
- Post 6: interview prep memory section; optim_8bit and setup commits.
- Post 7: passk pipeline commits (async vLLM, isolated venv); passk_utils.py.
- Post 8: tree_losses_research_note.md, prefix_overlap_research_note.md.
- Post 9: tree_losses_research_note.md (tree_pg), depth_weight_research_note.md.
- Post 10: enrich_research_note.md, ddte_research_note.md.
- Post 11: passk_prob_vs_jsd_research_note.md; the project report closing.
