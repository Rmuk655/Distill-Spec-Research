# ML fundamentals: 34 objectives against 9 verifiers, and why the aligned ones lost

This is post 8 in my series. It covers the breadth of what I tested, then the depth of why the most obvious choice among all of it turned out to be the worst one.

## Most people test one loss against one verifier, I tested 34 against 9

A lot of work in this space picks one training objective, checks it against one verifier, and reports a result. That comparison hides a real question: does an objective that helps on the verifier it was designed for still help on a different one. Before believing any single result, I wanted the full grid, not one cell of it.

Speculative decoding needs a verifier: the rule that decides which of the draft's guesses the target actually accepts, while still guaranteeing the output matches the target exactly. Six of the nine have that lossless guarantee proven for them already in the literature, called optimal transport based, or OTLP:

- naive: the rule from the original speculative decoding paper, one path at a time
- nss: naive speculative sampling, checks multiple paths
- spectr, specinfer, khisti: three different multi path constructions, each from a separate paper
- max: combines the above into one rule

Three are newer and not OT based, block verification style instead:

- bv: block verification, one path at a time
- traversal: multi path, from my mentor Rahul's own published work
- gbv: greedy block verification, multi path, also his

The loss functions, by category:

- **Flat baselines**: forward KL, reverse KL, JSD, L1. Match the teacher's whole output distribution, no tree structure involved.
- **Tree losses**: one shaped around nearly every verifier above, naive, traversal, bv, gbv, nss, specinfer, spectr, khisti, plus a fully differentiable variant. These train on the draft's own sampled guess tree instead of a fixed teacher sequence.
- **Log space tree losses**: the same family, reformulated to fix a vanishing gradient at depth. Six variants.
- **Off policy tree losses**: two variants trained on trajectories from a fixed reference policy rather than the live one.
- **Enrichment**: flat and tree versions, trained on several stochastic teacher rollouts instead of one greedy one.
- **Prefix overlap**: four objective variants targeting the teacher's prefix probability directly, each with a cold start and a warm start version.
- **Depth weighted curriculum**: not a new loss on its own, a wrapper that reweights any tree loss by expected acceptance depth.
- **LK alpha**: acceptance rate distillation from a recent paper, plus an enrichment variant of it.
- **REINFORCE**: one variant, rewarding block efficiency directly, later removed.

That is more than thirty named objectives, checked against nine verifiers, across two model pairs. Most of that grid is not glamorous. Most cells confirm the same plateau. But you cannot tell a real signal from a lucky cell without filling in the rest of the grid first, and as far as I can tell, nobody had actually done that before for this specific problem.

## The result that changed how I think about training objectives

I was optimizing block efficiency, which depends on an acceptance test between the draft and the target. So the obvious idea, sitting right there in the grid above, is LK alpha: train the draft on an objective that targets that exact acceptance test directly. Match what you are graded on.

I tried it. It was the single worst objective I tested. Not by a little, and consistently. At the wider model gap it dropped block efficiency from the start.

![Objectives that target acceptance directly, at the wider model pair](../../results/passK/linkedin_post8_negative_results_1p7_32.png)
*At the wider model pair, the objectives that target acceptance most directly all sit below the plain baseline, most of them at every tree width K.*

Same lesson from a second idea in the grid, the log space tree losses. Part of the acceptance signal fades deeper into the guess sequence, so its gradient is weak at depth. The standard fix is to move the objective into log space. When I applied it, results got worse, and at the wider gap training collapsed.

Both failed for the same reason. They poured gradient into deep, unlikely parts of the guess tree, positions the draft almost never reaches. The draft spent its limited capacity matching branches it would rarely visit, instead of getting the early, common tokens right, which are the ones that decide acceptance.

The objective looked aligned in its formula but was misaligned in where it sent the signal. The plain baseline put its effort in more useful places without being told to.

This is the lesson I repeat most from the whole grid. Matching the form of your metric does not mean the loss will train well. What matters is where the gradient lands, token by token. A loss can be correct on paper and still be a bad teacher.

I expected the opposite: closer to the metric, better to train. The data disagreed, twice, in a grid built specifically to catch exactly this kind of illusion. Now I stop asking does this match the metric, and ask where does this send the gradient.

## Further reading

Samarin et al., [LK alpha: Beyond KL for Speculative Decoding Distillation](https://arxiv.org/abs/2602.23881), the acceptance rate objective this post's negative result is built on. Miao et al., [SpecInfer: Accelerating Generative LLM Serving with Speculative Inference and Token Tree Verification](https://arxiv.org/abs/2305.09781), one of the OTLP verifiers in the grid above.

---

Part of a series on building a speculative decoding research platform. Post 8.
