# Isolating variables: my piece of the puzzle, not the whole one

In [Post 1](LINK_TO_POST_1), I said I optimized block efficiency, not tokens per second. Here is why.

Block efficiency = accepted tokens / target model calls. It isolates the algorithm from the hardware and serving stack underneath it.

## Why inference speed matters

Training is a one time cost. Inference is paid on every request, forever, at huge volume. A small saving per token, times a billion requests a day, is real money and lower latency.

## The problem with tokens per second

Tokens per second depends on three things: [the hardware](https://www.intoai.pub/p/what-every-ai-engineer-must-know-about-nvidia-gpus), [the serving stack](https://www.intoai.pub/p/10-llm-inference-optimization-techniques) (KV caching, quantization, batching, speculative decoding), and the algorithm. Change any one and the number moves. Report it alone and no one can tell which one changed.

We worked on two things inside speculative decoding: training the draft, and choosing the verifier. We needed a metric that held hardware and serving stack fixed, and moved only when the algorithm did.

## What block efficiency is

Normal decoding predicts one token, then runs the whole model again for the next one. Speculative decoding lets a small draft guess several tokens ahead; the target checks them all in one pass and keeps the longest prefix it agrees with. Block efficiency counts how many guesses got confirmed per expensive target call.

α = acceptance probability at one node. Every verifier defines its own α, so it is not comparable across verifiers and says nothing about tree depth. Block efficiency is comparable across all nine and accounts for depth, which is why it, not α, is the headline metric.

Prompt: "The capital of France is", target says "Paris, which sits on the Seine." A good draft guessing "Paris, which sits on" gets all five tokens accepted in one pass. A weak draft guessing "Paris, a lovely old" gets only "Paris," accepted, two tokens.

Both drafts above produce the same final text; the draft only changes how many target calls that costs. Block efficiency measures that cost, not quality, and not measured speed, though the two usually move together. Later in this post, I describe a case where they do not.

Block efficiency barely depends on hardware, since accepting a token is just a probability check. One exception: different attention kernels round bf16 slightly differently, which can flip an accept or reject decision and move block efficiency by a few tenths.

## Why tokens per second is hardware dependent

Prefill, the whole prompt in one pass, is compute bound. Decode, one token at a time, is usually memory bandwidth bound at low batch size: little math, lots of bytes to stream per token. Larger batches push decode back toward compute bound, which is why serving stacks batch aggressively. My numbers below are read at the low batch regime I actually ran.

The 32B teacher used about 66 GB of an 80 GB card, yet GPU compute utilization sat at 33.5 percent, barely above the 8B teacher's 31.6. If tensor cores were the bottleneck, a model that size should have kept them far busier. It did not. The bigger model also took longer per call, 49ms against 86, and that extra time is HBM traffic, not extra math.

So decode speed tracks memory bandwidth more than compute, interconnect too for multi GPU models, everything here ran on one. HBM bandwidth alone rises substantially from an A100 to an H100 to a B200. Same model, same code, newer hardware, more tokens per second, zero change to the algorithm. That alone makes tokens per second the wrong metric for isolating what I actually changed.

For a deeper tour of this ground, [The Engineering Behind LLM Inference](https://www.youtube.com/playlist?list=PLqO45Dg1pMhlDBZTMqVL2GU-14xYip2y2).

## The serving stack moves it too

KV caching, continuous batching, FlashAttention, quantization, paged attention, prefill and decode disaggregation: real production techniques, none touching model quality, none used here, since the goal was to isolate speculative decoding on its own.

![Where my work sits in the inference stack](../../results/passK/linkedin_post2_where_our_work_sits.png)
*Hardware and serving software stayed fixed. Speculative decoding is an existing technique. Inside it, two levers, both tested: the loss that trains the draft, and the verifier, specifically delayed tree branching on already trained drafts, which [the original paper](https://arxiv.org/abs/2602.16994) only tested on untrained ones.*

A real before and after: same draft, teacher, verifier, K, hardware. I compared two training objectives. Plain JSD trains on one greedy teacher rollout, so the training contexts come from the teacher's single most likely continuation, and JSD matches full distributions at each context. Enrichment trains on several sampled continuations instead, so the draft learns more than the teacher's top pick.

![Training the draft differently moved block efficiency, and throughput moved with it](../../results/passK/linkedin_post2_training_moved_be_and_throughput.png)
*Left: block efficiency. Right: throughput. At K=2, 3, 4, enrichment raises both together, throughput by about 2 to 2.5 tokens per second each time. At K=1 both are flat: enrichment's benefit is a second guess to fall back on, and K=1 has none.*

## Where block efficiency and throughput split

The enrichment result above: block efficiency up, throughput up together. Not guaranteed. Here is where it breaks.

Increasing tree depth from L=16 to L=32, same checkpoint, verifier, K, dataset: block efficiency rises from 7.56 to 8.93. Throughput drops from 11.7 to 7.4 tokens per second. Same direction on a different verifier, dataset, and checkpoint. A deeper tree accepts more guesses, but costs more draft and verification work than those guesses are worth.

Block efficiency went up. The system got slower.

That is the honest limit of the metric: it measures whether the algorithm converts a proposal into accepted tokens, not what that proposal cost to produce or check.

## The takeaway

Tokens per second = f(hardware, serving stack, algorithm). Move one, the number moves, so alone it never says which. Block efficiency holds hardware and serving stack fixed and measures the algorithm's acceptance effectiveness alone.

Block efficiency asks whether speculative decoding is proposing and confirming tokens well. But throughput is what actually matters in real world systems.

---

Part of a series on building a speculative decoding research platform. Post 2.
