# What a KV cache is and why vLLM exists

This is post 7 in my series. It is the one where I admit I did not really understand serving until I had to generate a lot of samples fast.

At one point I needed to check something different from block efficiency. I wanted to know if a trained draft model could actually solve problems, so I needed to sample many full answers per problem, tens of thousands of samples in total, and grade them. My first instinct was the simple one. Load the model, call it in a loop, collect the outputs.

That was far too slow, and understanding why taught me two ideas I had heard of but never really felt.

The first is the KV cache. When a model generates text one token at a time, each new token attends to all the tokens before it. If you recomputed everything from scratch at every step, you would redo the same work over and over. Instead the model saves the attention keys and values for the tokens it has already seen, and reuses them. That saved memory is the KV cache. It is what makes generation not scale terribly with length. It is also why long sequences eat so much memory, because the cache grows with every token.

The second idea is what a real serving engine does with that cache. I used vLLM for this part. Two things it does mattered to me.

One is paged attention. A naive setup gives each sequence one big block of memory for its cache. When sequences have different lengths and finish at different times, that memory fragments and gets wasted. Paged attention instead breaks the cache into small fixed pages, like how an operating system manages memory. Nothing gets stranded, and a finished sequence frees its pages immediately for the next one.

The other is continuous batching. In a naive batch you wait for every sequence in the batch to finish before starting new ones, so the whole batch moves at the speed of its slowest member. Continuous batching admits a new sequence the moment a slot frees up. The GPU stays busy instead of idling while it waits for one long straggler.

Put together, these are why the sampling job that would have taken a very long time in a plain loop finished in a reasonable one.

![The evaluation pipeline, with the pass at k path running on vLLM with a batched KV cache](../medium_chart_architecture.png)
*The evaluation half of the platform. The pass at k path runs on vLLM with a batched KV cache, in its own isolated environment, kept separate from the training stack.*

There was one more thing I had to respect. vLLM pins specific versions of its dependencies, and those fight with the versions my training code needed. So I did not install it into my training environment. I gave it a completely separate environment and ran it as its own pipeline, with its own resume logic so that if it got killed it did not redo finished work.

The lesson is that generating text at scale is a systems problem, not just a modeling one. A KV cache is the reason generation is fast enough to be practical, and a serving engine like vLLM exists to manage that cache well across many requests at once. I knew those words before. I did not understand them until my simple loop was too slow.

---

Part of the series. This is post 7. Next: the training objective that matched my metric perfectly and still lost.
