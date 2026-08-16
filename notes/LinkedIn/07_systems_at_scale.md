# Capacity planning: the model got smaller and still would not load

In my [first post](https://medium.com/@rmukund16/engineering-an-llm-inference-research-platform-for-speculative-decoding-405d2f0bf53f), I described the hardware climb behind this project: a 16GB Colab GPU, then Kaggle, then a 40GB A100, then an 80GB one, dropping LoRA, the 8 bit optimizer, and teacher quantization along the way. This is the story behind that line. The short version: I badly underestimated the hardware, and fitting a model turned out to be different from fitting its weights.

## The cheap start

I did not begin on an A100. I began on my laptop GPU, a few GB of memory, running a 0.6B draft against a 0.6B teacher. The numbers there meant nothing, since a teacher the same size as the draft teaches it almost nothing, but it was enough to build the pipeline and write dummy tests that exercised every path. It was immediately clear that anything real, even a 0.6B draft against an 8B teacher, would need far more than a free tier.

## The model got smaller and still would not load

The 8B teacher in bf16 is about 16GB. A free Colab T4 has 15GB. So I reached for [4 bit quantization](https://huggingface.co/blog/4bit-transformers-bitsandbytes), storing the weights in 4 bits instead of 16 to shrink the teacher to about 4GB. On paper it fit easily.

It still died, and the reason surprised me. Loading in 4 bit does not go straight to 4 bit. The library reads each weight tensor into CPU RAM as bf16 first, then converts, so for an 8B model the loading path briefly needs about 16GB of system RAM. Colab has about 12GB. The process was killed before the model ever reached the GPU.

The final size was not the only number that mattered. Peak memory during loading mattered too, and here it was on the CPU, not the GPU.

## Kaggle had the RAM Colab did not

Kaggle gave me two T4s and, more importantly, about 29GB of system RAM. That was enough headroom for the 4 bit loading spike, so the 8B teacher finally loaded, split across the two cards. I also had [LoRA](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora) for the draft, which trains only a small set of extra weights and freezes the rest, so the optimizer state stays small. But free tiers cap your session and kick you off, so even here I could not finish a full run. Kaggle proved the setup worked. It could not carry the real experiments.

## The real training, and 64GB of weights

That happened on A100s. The largest pair was a 1.7B draft and a 32B teacher. A bf16 parameter is two bytes, so 32 billion of them is about 64GB of weights, most of an 80GB card before anything else exists. And the draft was training, so it also needed gradients, activations, and optimizer state, and generation needed a KV cache. I stopped asking whether the model fit and started asking what was using memory at the exact point it stopped fitting.

Each memory tool had a catch. The 4 bit teacher shrank 64GB to about 18GB, but one class of losses generates from the teacher token by token, and 4 bit weights are dequantized on every one of those steps, so those runs took about 18.3 seconds per step against 1.0 for the others. I had traded a memory problem for a speed problem. The 8 bit optimizer cut Adam's state, but my first version pre-allocated it at startup, which collided with the temporary allocations during model loading and hit an out of memory error on the very first step. I let it allocate lazily instead. The lesson stuck: peak memory depends on when things exist, not just how big they are.

The smaller teachers, 8B and down, ran in full bf16 with none of this. I only needed the tricks when the 32B pair pushed the card to its edge.

## One thing this was not

I ran many independent jobs across several GPUs and packed small ones onto a single card. I never split one model across GPUs. No tensor parallelism, no pipeline parallelism, no cross card communication. Every model, including the 32B teacher, fit on one GPU. That is orchestration, not distributed training. The moment a model no longer fits on a single card, the bandwidth between cards becomes the whole problem, and I did not build that part.

## The lesson

I started by asking whether a model fit on a GPU. I ended by asking which resource ran out first, and when. Weights were rarely the answer. It was the optimizer state, the activations, the loading spike, or the KV cache, and often it came down to timing, two large things existing at the same moment. Fitting a model is not the same as fitting its weights.

---

Part of a series on building a speculative decoding for LLM inference research platform. Post 7.
