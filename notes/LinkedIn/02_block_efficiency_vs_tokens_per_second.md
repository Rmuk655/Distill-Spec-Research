# Tokens per second measures the system. Block efficiency measures the algorithm.

In [Engineering an LLM inference research platform for speculative decoding](LINK_TO_POST_1), I wrote one line almost in passing: I optimized block efficiency, not tokens per second. If you wondered why, read on.

Block efficiency is accepted tokens per target model call. For this problem it is the better metric, because it isolates the algorithm from the hardware and the serving stack sitting underneath it.

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

Normal decoding is autoregressive: each token is predicted conditioned on every token generated before it. In practice that means you generate one token, append it to the sequence, and run the whole model again to predict the next one, a full forward pass per token, every time. That per token pass is the expensive part.

Speculative decoding gets around paying that cost one token at a time. A small fast draft model guesses several tokens ahead. The large target model checks all of them in a single forward pass and keeps the longest prefix it agrees with. Block efficiency is accepted tokens per target forward pass, how many guesses you got confirmed for the price of one expensive pass.

Take the prompt "The capital of France is", and suppose the target would generate "Paris, which sits on the Seine". A good draft guesses "Paris, which sits on" and the target accepts all five tokens in one pass. A weaker draft guesses "Paris, a lovely old" and the target only accepts "Paris," before the guess diverges, confirming two tokens and generating the rest itself.

Here is the part that matters: the final text is identical either way, "Paris, which sits on the Seine", because the target's own distribution is what gets output regardless of the draft. The draft never changes the answer, only how many tokens get confirmed per target call. So block efficiency is purely a speed number. The weaker draft is not giving a worse answer, it is just forcing more target calls to produce the same one.

None of that depends on hardware or software. Run this on a slow naive loop on an A100, or on a fast batched stack like vLLM on a B200, and tokens per second will be very different. But the count of accepted tokens per target call stays the same, because accepting a token only depends on a probability check between the draft and the target, not on which box or framework ran it.

One objection worth answering directly. When I profiled a real run, the draft spent about 766 seconds generating across one hundred prompts, the target spent about 109 seconds verifying, and GPU utilization sat around 31 percent, because the draft launches one tiny operation per token and pays that launch cost every time. So why not just make the draft cheaper to run instead of retraining it. That is a real, separate lever, and people do optimize it at deploy time: batching draft steps together, using CUDA graphs to cut repeated kernel launch overhead, keeping a warm draft process alive instead of reloading it per request. We did not simulate. Our question was narrower: holding the draft's own execution cost fixed, does a better trained draft get more of its guesses accepted. That is the algorithm question, separate from the engineering question of how cheaply you can run whatever draft you have.

## The hardware axis: why tokens per second tracks memory bandwidth, not compute

Here is the systems fact that made it click, and I checked it against our own runs rather than just citing it. LLM inference has two phases. Prefill processes the whole prompt in one forward pass and is compute bound, the tensor cores stay busy. Decode generates one token at a time, one forward pass each, and it is memory bandwidth bound, not compute bound.

Our own profiling shows this directly. The 32B teacher has four times the parameters of the 8B teacher, so a compute bound system should take roughly four times as long per forward call. Measured across the same 100 prompts, traversal verifier, K=3: the 8B teacher took about 49 milliseconds per target call, the 32B teacher took about 86, less than double, not four times. GPU compute utilization barely moved either, 31.6 percent for the 8B teacher against 33.5 percent for the 32B one, despite the model quadrupling in size. If compute were the bottleneck, that jump in parameters should have pushed utilization up sharply. It barely moved, because in both cases the GPU was waiting on memory bandwidth to stream weights, not on the math itself.

So decode speed tracks memory bandwidth and interconnect, not raw compute. Move the same model from an A100 to an H100 to a B200, and NVLink bandwidth between GPUs alone jumps from about 900 GB/s to 1.8 TB/s, on top of a faster HBM generation. Same model, same code, newer hardware, more tokens per second, with zero change to the algorithm. That alone makes tokens per second a bad metric for my question.

## The software axis: the serving stack moves it too

The rest of the number comes from the serving system, none of which touches model quality. The real ones people run in production: KV caching, more efficient attention variants that shrink the cache, continuous batching, fused kernels like FlashAttention, quantization, prefix caching, pruning, paged attention, prefill and decode disaggregation, and speculative decoding itself. We did not use any of these other techniques, because the goal was to isolate the impact of speculative decoding on its own.

![Where my work sits in the inference stack](../../results/passK/linkedin_post2_where_our_work_sits.png)
*Hardware and serving software were fixed, given. Speculative decoding itself is an existing technique. Inside it, two separate levers, both tested: the distillation loss that trains the draft to mimic the teacher, and the verifier, specifically delayed tree branching applied to already trained drafts, which the original paper had only tested on untrained ones. Block efficiency is what tells me whether either piece moved.*

Here is that link made concrete, with a real before and after. Same draft and teacher, same verifier, same K, the tree width at evaluation, same hardware, same harness. The only thing that changed between the two bars in each group is how the draft was trained. Plain JSD trains the draft to match the teacher's single most likely continuation. Enrichment instead trains it on several different continuations sampled from the teacher, so the draft learns to cover more of what the teacher might actually say, not just its top pick. Post 10 covers this method in full, here it is just the two training methods being compared.

![Training the draft differently moved block efficiency, and throughput moved with it](../../results/passK/linkedin_post2_training_moved_be_and_throughput.png)
*Left: block efficiency. Right: measured throughput, tokens per second. At K=2, 3, and 4, enrichment training raised block efficiency over the plain JSD baseline, and throughput rose right along with it, by about 2 to 2.5 tokens per second each time. At K=1 both numbers barely move, which is honest to show rather than hide, and it has a real reason: enrichment's whole benefit is giving the draft several plausible guesses instead of one, and at K=1 there is no second guess to fall back on, just a single pick checked on its own, where plain JSD already does fine. The gain needs more than one branch to have somewhere to land. This is block efficiency doing its job: it predicted the throughput gain before I ever needed to measure wall clock time to know the training method was working.*

## The takeaway

Once I saw that tokens per second is really f(hardware, serving stack), and our work was one narrow term buried inside that second factor, the choice made sense. Block efficiency was Rahul's idea, my research mentor's: hold the hardware and the rest of the serving stack fixed and measure only the draft's training. Testing that properly took hundreds of runs, not one lucky comparison. And because the ratio itself carries no seconds and no hardware in it, once it moved, I knew the algorithm had actually changed, not the machine underneath it. Once I could tell memory bandwidth, cache design, and batching apart from the algorithm itself, systems work stopped feeling intimidating and started feeling like a puzzle I actually wanted to solve. That is what got me hooked on ML systems.

---

Part of a series on building a speculative decoding research platform. Post 2. Next: how I checked whether a result was real before believing it.
