# The one change that beat the baseline

This is post 10 in my series. After a lot of things that did not work, here is what did, and the honest limits on it.

Most of my training ideas tried to reweight or reshape the loss on the same fixed set of teacher tokens. They all landed at about the same ceiling as the plain baseline. One change broke through, and it was not a cleverer loss. It was a change to the data the draft trained on.

The idea is called enrichment. Normally you take one greedy continuation from the teacher and train the draft to match it. Instead, I sampled several different continuations from the teacher, letting it vary, and trained on all of them. The draft sees a broader picture of what the teacher might say, not just the single most likely path.

This worked. On the smaller model pair, it beat the baseline on the deployment verifier, and the sweet spot was four sampled continuations. Fewer than that helped less. More than that, around six, started to fade, because the extra samples just repeated the same region and stopped adding new information. The gain also held on a harder dataset the model never saw in training, which is the stronger kind of evidence, since it is easy to win on data you trained on and harder to win on data you did not.

Now the honest part, because this is where it would be easy to overclaim. The win was clean on the smaller pair. On the wider pair, with the bigger models, it did not cleanly repeat. The numbers there flipped sign across settings, and I only had one random seed, so I cannot call it a real win at that scale. The most honest summary is that enrichment helped at one scale, in a measurable and repeatable way, and the jury is still out at the other.

The strongest single result in the project came from a different direction entirely, the verification side. My mentor had published a technique that delays where the draft's guess tree splits. Instead of branching into many candidates at the very first token, it commits to one stem and only splits later, at the point where the draft and target actually start to disagree. His paper only ever tested it on untrained draft models. I tested it on my trained ones.

It helped almost everywhere. The chart below shows the lift across every trained checkpoint on both model pairs. Verifiers that were already near their ceiling gained a little. Verifiers that were further behind gained a lot, in some cases over twenty percent. And stacking this on top of the enrichment training above gave the best single result I got.

![Delayed tree branching lift across every trained checkpoint, both model pairs](../medium_chart_ddte_anon.png)
*Delayed tree branching, every trained checkpoint, both model pairs. Almost all improved, and the weaker the starting point, the larger the lift.*

The lesson: when reshaping the loss kept hitting the same wall, the things that actually moved the needle came from elsewhere, the data the draft saw and the structure of how it was verified. Sometimes the ceiling is not in the part you keep polishing.

---

Part of the series. This is post 10.
