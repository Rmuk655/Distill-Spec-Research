# Capacity planning: fitting a 32B model, then serving it

This is post 7 in my series. It is where I actually learned how GPU memory works, and where I learned that generating text at scale is a systems problem, not just a modeling one, because I had no choice on either front.

## The memory budget is not one number

I set the project up to climb a ladder of hardware before spending money on it. First a laptop, just to check the code ran and the loss went down, with a 0.6B teacher whose capacity matches the draft, so the numbers meant nothing but the pipeline worked. Then free Colab and Kaggle T4 cards, about 15GB each, with a 1.7B and then a 4B teacher, to see whether a loss function moved block efficiency at all before committing to a long run. None of this was real training. It was scaffolding, cheap places to catch a bug or a dead idea early.

Even at that scale the memory line items showed up, and one lesson came early. The obvious way to fit a bigger teacher on a small card is 4 bit quantization, storing weights in fewer bits. It does not work on Colab, for a non obvious reason. Loading a model in 4 bit reads each weight tensor into CPU RAM as bf16 first and then converts, and for an 8B teacher that intermediate is about 16GB against Colab's 12GB of system RAM, so the Linux out of memory killer fires before Python even starts. On the free tiers I fit bigger teachers the plain way instead, a smaller teacher in bf16, and left quantization for a machine that could load it.

That machine was the A100. Once I had it, the real training happened there, and the teacher climbed from 8B to 32B, which is where memory got genuinely hard. A parameter in bf16 is two bytes, so 32 billion of them is about 64GB of weights sitting there doing nothing. That barely fits an 80GB card, and only if nothing else competes for room. And the weights are never the whole story. The teacher is frozen, but the draft still trains, so you store optimizer state, two extra numbers per parameter for the common optimizer. Then activations, the intermediate values kept to compute gradients. Then the KV cache, the attention keys and values for the tokens generated so far, which grows with the sequence. The real question was never do the weights fit, it was do the weights plus the optimizer plus the activations plus the cache fit, all at once.

Three techniques bought that room on the A100, each with its own catch.

LoRA trains only a small set of extra adapter weights and freezes the rest of the draft, so the optimizer state covers just that small set instead of the whole model.

Quantizing the teacher was the heaviest tool, and the one with the sharpest catch. Loading the 32B teacher in 4 bit instead of bf16 drops it from about 64GB to about 18GB. But quantization is not free. Flat losses call the teacher's generate() once per token, many sequential steps, and 4 bit weights get dequantized on every one of those forward passes. That made flat losses about eighteen times slower per step than tree losses on the 32B pair, 18.3 seconds a step against 1.0. The memory I saved on paper, I paid back in wall clock, and only for one class of loss. A 4 bit teacher also gives slightly noisier training targets, one more reason to avoid it when the memory math allows.

That slowdown is why I reached for the third tool, quantizing the optimizer state. Storing Adam's two running statistics in 8 bit instead of 32 bit frees enough room that a run could sometimes afford the full bf16 teacher instead of the 4 bit one, skipping the dequantization tax. It quantizes the optimizer, not the teacher, so the two are independent. My first attempt was worse than doing nothing: I pre-allocated the 8 bit optimizer state at startup, which forced a full gradient sized allocation before the model loading transients had freed, a higher peak than the lazy path it was meant to help, and it hit an out of memory error at the very first step. I reverted it and let the state allocate lazily.

The smaller teachers, 8B and down, ran clean in full bf16 on the A100 with none of these. The tools were not a ladder I climbed down as hardware improved. They were what made the biggest teacher fit on the one machine that could hold it at all.

The lesson is that memory is not one number. It is a budget with several line items, and the line item that kills you changes as you scale. Weights, optimizer state, activations, cache. Knowing which one is about to overflow is most of the battle, and you only really learn it when a card says no.

One distinction worth being precise about, since it is easy to blur: everything above is GPU orchestration, scheduling many independent single GPU jobs, packing several small runs onto one A100 when they fit, one eval per GPU when they did not. It is not distributed training. Every model in this project, including the 32B teacher, fit entirely on one GPU. There was never a model split across GPUs with NCCL collectives, no tensor or pipeline parallelism, no cross GPU communication to reason about. Orchestration and distributed computation solve different problems and get confused often. This project only needed the first.

## Future work: what changes past one GPU

Everything above assumed a model that fits on a single card, true for every model in this project, even the 32B teacher. That assumption breaks past a certain size, and the next problem is worth naming rather than skipping. A 70 billion parameter model in full precision is roughly 140GB of weights alone, before optimizer state, activations, or cache, and that does not fit on one 80GB H100. It has to be split across several GPUs, and how well that works depends on how those GPUs actually talk to each other, not just how many of them there are.

An 8 GPU H100 server has 640GB of HBM in total, but only if the GPUs can share work efficiently. With NVSwitch, every GPU gets a full 900GB/s to every other GPU at once. Without it, that same 900GB/s splits into separate point to point links, about 128GB/s per pair in an eight GPU box. Same hardware otherwise, a very different ceiling on how fast the GPUs can cooperate.

From there the real question is how you split the work, model parallel across GPUs versus workload parallel with a full copy on each. I have not built either. It is the direct next problem past everything else in this post.

## Fitting the model was half the problem, serving it fast was the other half

At one point I needed to check something different from block efficiency. I wanted to know if a trained draft model could actually solve problems, so I needed to sample many full answers per problem, tens of thousands of samples in total, and grade them. My first instinct was the simple one. Load the model, call it in a loop, collect the outputs.

That was far too slow, and understanding why taught me two ideas I had heard of but never really felt.

The first is the KV cache, the same memory structure from the budget above, now seen from the serving side: it is why long sequences eat so much memory, and it is also the reason a naive generation loop wastes so much of it, one sequence at a time, nothing shared.

The second is what a real serving engine does about that. I used vLLM for this part, mainly for two things: paged attention, which breaks the cache into small fixed pages instead of one big block per sequence, so nothing gets stranded when sequences finish at different times, and continuous batching, which admits a new sequence the moment a slot frees up instead of waiting for the whole batch's slowest member. Put together, these are why the sampling job that would have taken a very long time in a plain loop finished in a reasonable one.

![The evaluation pipeline, with the pass at k path running on vLLM with a batched KV cache](../medium_chart_architecture.png)
*The evaluation half of the platform. The pass at k path runs on vLLM with a batched KV cache, in its own isolated environment, kept separate from the training stack.*

There was one more thing I had to respect. vLLM pins specific versions of its dependencies, and those fight with the versions my training code needed. So I did not install it into my training environment. I gave it a completely separate environment and ran it as its own pipeline, with its own resume logic so that if it got killed it did not redo finished work.

## The lesson, on both halves

Capacity planning and serving are the same discipline applied at two different times. Knowing which resource is about to overflow, weights, optimizer state, activations, cache, memory bandwidth, decides whether training even starts. Knowing what a serving engine does with that same cache, paged attention, continuous batching, decides whether generation finishes in a reasonable time once training is done. I knew the words KV cache and vLLM before this project. I did not understand either one until a memory budget said no and a simple loop was too slow.

## Further reading

Kwon et al., [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) (2023), the vLLM paper, for paged attention and continuous batching in full. NVIDIA, [NVLink and NVSwitch Supercharge Large Language Model Inference](https://developer.nvidia.com/blog/nvidia-nvlink-and-nvidia-nvswitch-supercharge-large-language-model-inference/), for the bandwidth figures used above.

---

Part of a series on building a speculative decoding for LLM inference research platform. Post 7.
