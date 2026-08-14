# Isolating variables: my piece of the puzzle, not the whole one

In [Engineering an LLM inference research platform for speculative decoding](LINK_TO_POST_1), I wrote one line almost in passing: I optimized block efficiency, not tokens per second. If you wondered why, read on.

Block efficiency is accepted tokens per target model call. It was the primary metric for the algorithmic side of this project, because it isolates speculative effectiveness from the hardware and the serving stack sitting underneath it.

## Why inference speed is worth obsessing over

Training a large model is a one time bill. Inference is paid on every request, forever, and the volume is enormous. Products like ChatGPT are reported to handle on the order of a billion messages a day, and every message is many forward passes through a large model. For a deployed model, inference, not training, dominates lifetime cost.

At that scale small percentages turn into big money. Cut the cost per token by a few percent, times a billion requests a day, and you save millions of dollars a year. The same speedup also cuts latency, so the product feels fast instead of slow. This is where systems engineering and machine learning intersect in a genuinely interesting way, and neither field explains inference speed on its own.

## The problem: tokens per second depends on hardware and software optimizations, not just the algorithm

The obvious speed metric is tokens processed per second. The trouble is that it is a function of two independent things, and our work was a narrow slice of the second one:

1. the underlying compute and memory hardware, [what it looks like](https://www.intoai.pub/p/what-every-ai-engineer-must-know-about-nvidia-gpus) and [how it actually behaves during inference](https://www.intoai.pub/p/a-hardware-level-tour-of-llm-inference),
2. [how optimized the inference serving stack is](https://www.intoai.pub/p/10-llm-inference-optimization-techniques): KV caching, quantization, continuous batching, and speculative decoding, which pairs a small draft model from the same family as the target, sharing its tokenizer and vocabulary, with the large target model, all sitting in that same list of techniques.

Speculative decoding was the one technique we worked inside of, and inside that, two narrow questions. First, training: can a small draft model be trained to predict tokens as close as possible to what the big teacher would have predicted, including losses shaped directly around a specific verifier's own acceptance rule, not just generic distribution matching. Second, verification: independent of how the draft is trained, does changing the verifier itself, for instance restructuring how its guess tree branches, get more out of a draft that is already trained. I worked on both.

Change any one of these and tokens per second moves. If we reported it, no one could tell whether a gain came from a better trained draft, from a smarter serving stack, or from the hardware we happened to grab that day. We needed a metric that holds the hardware and the rest of the serving stack fixed and isolates just the draft's training.

## What block efficiency is, by example

Normal decoding is autoregressive: each token is predicted conditioned on every token generated before it. In practice that means you generate one token, append it to the sequence, and run the model again to predict the next one. Cached attention keys and values mean earlier tokens are not recomputed from scratch, but a fresh forward pass still happens for every single token, and that per token pass is the expensive part.

Speculative decoding gets around paying that cost one token at a time. A small fast draft model guesses several tokens ahead. The large target model checks all of them in a single forward pass and keeps the longest prefix it agrees with. Block efficiency is accepted tokens per target forward pass, how many guesses you got confirmed for the price of one expensive pass.

Take the prompt "The capital of France is", and suppose the target would generate "Paris, which sits on the Seine". A good draft guesses "Paris, which sits on" and the target accepts all five tokens in one pass. A weaker draft guesses "Paris, a lovely old" and the target only accepts "Paris," before the guess diverges, confirming two tokens and generating the rest itself.

Here is the part that matters: correct speculative decoding preserves the target model's own output distribution, the draft changes how efficiently a sample gets produced, not the distribution being sampled from. In this example that means the final text comes out the same either way, "Paris, which sits on the Seine". The draft never changes the answer, only how many tokens get confirmed per target call. So block efficiency is not a quality number, the weaker draft is not giving a worse answer, it is just forcing more target calls to produce the same one. It is an algorithmic efficiency number instead, a count of expensive calls avoided, not a measurement of how fast a request actually finished. Those two things usually move together. Later in this post I found a case where they do not, and keeping them separate is the whole reason this post exists.

Unlike wall clock throughput, block efficiency is largely invariant to hardware speed and serving stack performance, as long as the model, the decoding configuration, and the numerical behavior are held fixed. Run this on a slow naive loop on an A100, or on a fast batched stack like vLLM on a B200, and tokens per second will be very different, while the count of accepted tokens per target call stays close to the same, because accepting a token is a probability check between the draft and the target, not a property of which box or framework ran it. One real caveat worth naming: different attention kernels round bf16 math slightly differently, and that rounding can shift which side of an accept or reject threshold a token lands on, moving block efficiency by a few tenths on individual cells even with everything else held fixed. Numerical behavior is part of what held fixed has to mean, not something block efficiency is automatically immune to.

One objection worth answering directly. When I profiled a real run, the draft spent about 766 seconds generating across one hundred prompts, the target spent about 109 seconds verifying, and GPU utilization sat around 31 percent, because the draft launches one tiny operation per token and pays that launch cost every time. So why not just make the draft cheaper to run instead of retraining it. That is a real, separate lever, and people do optimize it at deploy time: batching draft steps together, using CUDA graphs to cut repeated kernel launch overhead, keeping a warm draft process alive instead of reloading it per request. We did not build or test any of those three, naming them is pointing at real levers other people use at deploy time, not describing work done here. Our question was narrower: holding the draft's own execution cost fixed, does a better trained draft get more of its guesses accepted. That is the algorithm question, separate from the engineering question of how cheaply you can run whatever draft you have.

## The hardware axis: why tokens per second tracks memory bandwidth, not compute

Here is the systems fact that made it click, and I checked it against our own runs rather than just citing it. LLM inference has two phases. Prefill processes the whole prompt in one forward pass and is compute bound, the tensor cores stay busy. Decode generates one token at a time, one forward pass each, and at the batch size this project ran at, one request at a time, it tends to be memory bandwidth bound instead, because each step does relatively little math per byte of model weight it has to stream off the GPU. That balance is not fixed. Larger batches raise how much compute happens per byte moved and can push decode back toward compute bound, which is exactly why serving systems batch aggressively. My own numbers below are read against the low batch regime I actually measured, not as a claim about every batch size.

Our own profiling makes the case simply. The 32B teacher alone used about 66 GB of the 80 GB card, yet GPU compute utilization sat at only 33.5 percent while it ran, barely above the 31.6 percent for the much smaller 8B teacher. If the tensor cores were the bottleneck, a model that size occupying most of the card should have kept them far busier than a third of the time. It did not. The bigger model also took longer per call, 49 milliseconds for the 8B teacher against 86 for the 32B one, and that extra time is the cost of streaming far more weight bytes through HBM, not extra math.

So at the batch size I ran at, decode speed tracks memory bandwidth more than raw compute, and would track interconnect bandwidth too if a model were split across GPUs, though everything in this project ran on a single GPU, so interconnect was never actually my bottleneck. HBM bandwidth alone climbs substantially from an A100 to an H100 to a B200. Same model, same code, newer hardware, more tokens per second, with zero change to the algorithm. That alone makes tokens per second a bad metric for my question.

For a deeper, first principles tour of this same ground, the memory wall, kernels, quantization, and serving, [The Engineering Behind LLM Inference](https://www.youtube.com/playlist?list=PLqO45Dg1pMhlDBZTMqVL2GU-14xYip2y2) is a good watch.

## The software axis: the serving stack moves it too

The rest of the number comes from the serving system, none of which touches model quality. The real ones people run in production: KV caching, more efficient attention variants that shrink the cache, continuous batching, fused kernels like FlashAttention, quantization, prefix caching, pruning, paged attention, prefill and decode disaggregation, and speculative decoding itself. We did not use any of these other techniques, because the goal was to isolate the impact of speculative decoding on its own.

![Where my work sits in the inference stack](../../results/passK/linkedin_post2_where_our_work_sits.png)
*Hardware and serving software stayed fixed. Speculative decoding itself is an existing technique. Inside it, two separate levers, both tested: the distillation loss that trains the draft to mimic the teacher, and the verifier, specifically delayed tree branching applied to already trained drafts, which the original paper had only tested on untrained ones. Block efficiency is what tells me whether either piece moved.*

Does changing only the training method actually move block efficiency? Here is a real before and after that shows it does. Same draft and teacher, same verifier, same K, the tree width at evaluation, same hardware, same harness. The only thing that changed between the two bars in each group is how the draft was trained. Plain JSD trains on one teacher rollout, generated greedily, so the training contexts come from the teacher's single most likely continuation, and JSD is the per token divergence between the two full distributions measured at each of those contexts. Enrichment instead trains on several different continuations sampled from the teacher, so the draft learns to cover more of what the teacher might actually say, not just its top pick.

![Training the draft differently moved block efficiency, and throughput moved with it](../../results/passK/linkedin_post2_training_moved_be_and_throughput.png)
*Left: block efficiency. Right: measured throughput, tokens per second. At K=2, 3, and 4, enrichment training raised block efficiency over the plain JSD baseline, and throughput rose right along with it, by about 2 to 2.5 tokens per second each time. At K=1 both numbers barely move. Enrichment's whole benefit is giving the draft several plausible guesses instead of one, and at K=1 there is no second guess to fall back on, just a single pick checked on its own, where plain JSD already does fine. The gain needs more than one branch to have somewhere to land. This is block efficiency doing its job: it predicted the throughput gain before I ever needed to measure wall clock time to know the training method was working.*

## Block efficiency and throughput are not the same thing, and here is where they split

The enrichment result above is the case where a block efficiency gain showed up as a throughput gain too. That pairing is not guaranteed, and I have a real case where it breaks.

Increasing the draft tree's depth from L=16 to L=32, same checkpoint, same traversal verifier, same K=3, same dataset, raises block efficiency from about 7.56 to 8.93. By the algorithm's own metric, that is a real gain, more guesses confirmed per target call. Measured throughput moves the opposite way, from about 11.7 to 7.4 tokens per second, and the same direction shows up on a different verifier, a different dataset, and a different checkpoint too, not a one off. A deeper tree gets more of its guesses accepted, but building and checking a tree that much larger costs more draft time and more verification work than the extra accepted tokens are worth.

Block efficiency went up. The system got slower.

That is the honest limit of the metric, not a flaw in it. Block efficiency measures whether the algorithm is doing its one job well, turning a proposal into accepted tokens. It has nothing to say about what that proposal cost to produce or to check. The enrichment result is the case where the gain was cheap enough to reach the stopwatch. The tree depth result is the case where it was not. Reading block efficiency correctly means remembering it can move for reasons that never reach measured speed, in either direction.

## The takeaway

Tokens per second is f(hardware, serving stack, algorithm). Move any one of them and the number moves, so on its own it never tells you which one moved it. If I want to know what my training change actually did, everything else has to stay fixed, otherwise a change in the metric could just as easily be the hardware or the serving stack, not my work. Block efficiency is what let me hold hardware and the serving stack still and measure the speculative algorithm's acceptance effectiveness on its own.

But holding the outside world still does not make the algorithm's own cost disappear, the tree depth result above is proof of that. Block efficiency tells me whether speculative decoding is proposing and confirming tokens well. Throughput tells me whether that gain actually survived the cost of getting it. I need both, read as two different questions, not one metric standing in for the other.

---

Part of a series on building a speculative decoding research platform. Post 2.
