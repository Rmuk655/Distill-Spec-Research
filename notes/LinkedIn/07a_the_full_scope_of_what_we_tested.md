# Most people test one loss against one verifier. We tested 34 against 9.

A lot of work in this space picks one training objective, checks it against one verifier, and reports a result. That comparison hides a real question: does an objective that helps on the verifier it was designed for still help on a different one. Before believing any single result, I wanted the full grid, not one cell of it.

## The verifiers, all nine

Speculative decoding needs a verifier: the rule that decides which of the draft's guesses the target actually accepts, while still guaranteeing the output matches the target exactly. Six of the nine have that lossless guarantee proven for them already in the literature, called optimal transport based, or OTLP:

- naive: the rule from the original speculative decoding paper, one path at a time
- nss: naive speculative sampling, checks multiple paths
- spectr, specinfer, khisti: three different multi path constructions, each from a separate paper
- max: combines the above into one rule

Three are newer and not OT based, block verification style instead:

- bv: block verification, one path at a time
- traversal: multi path, from my mentor Rahul's own published work
- gbv: greedy block verification, multi path, also his, tied to a delayed branching technique that changes where the guess tree splits

## The loss functions, by category

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

---

Part of a series on building a speculative decoding research platform.
