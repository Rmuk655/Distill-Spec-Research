# Capacity planning: the model got smaller and still would not load

In my [first post](https://medium.com/@rmukund16/engineering-an-llm-inference-research-platform-for-speculative-decoding-405d2f0bf53f), I described the hardware climb behind this project: my laptop, free GPUs, then A100s. I made that progression sound cleaner than it was. Most of it was me finding the next thing that did not fit.

## Why I started with models too small to matter

I started on my laptop GPU with a 0.6B draft and a 0.6B teacher.

That pair was not useful for the actual research. A teacher the same size as the draft teaches it almost nothing. It was useful only for checking that the training loop ran, gradients existed, checkpoints saved, and every loss path could finish.

Then I moved to free Colab and Kaggle GPUs with smaller teachers. The idea was simple: spend cheap hardware on bugs before spending A100 time on them.

That worked until memory became the bug.

## Why a 4 bit model still ran out of memory

An 8B teacher in bf16 is about 16GB of raw weights. A free Colab T4 has about 15GB of GPU memory, so I reached for [4 bit quantization](https://huggingface.co/blog/4bit-transformers-bitsandbytes).

On raw weight arithmetic that cuts roughly 16GB to 4GB, plus quantization metadata and the parts that stay at higher precision. On paper it fit easily.

It still died.

The problem was not GPU memory. In my Transformers and bitsandbytes loading path, each weight tensor was read into CPU memory at full width and then converted, so loading briefly needed far more system RAM than the final 4 bit model. Colab has about 12GB of system RAM, and it ran out before loading finished.

That was my first memory mistake: **I calculated what had to fit at the end, not what had to fit along the way.**

Kaggle had about 29GB of system RAM, enough headroom for that loading spike, so the 8B teacher in 4 bit actually loaded there. But free tiers cap your session and kick you off, so the free machines stayed useful mainly as scaffolding. The full experiments moved to A100s.

## Why 8B was easy and 32B was not

The 8B teachers ran in bf16 on the A100 without any of the tricks I had needed on smaller machines.

Then I moved to the 1.7B draft and 32B teacher pair.

A 32B teacher in bf16 is about 64GB of weights alone, since a bf16 parameter is two bytes and 32 billion of them is roughly 64GB. On an 80GB A100 that left little room for anything else. The teacher was frozen, but the draft was training, so there were gradients, activations, and optimizer state. Generation also needed a KV cache. Temporary allocations could push the peak higher again.

The question changed from "does the model fit?" to **"what exists at the moment memory peaks?"**

## Every way I made it fit had a catch

I used three tools.

**LoRA** reduced how much of the draft I trained, which also reduced the optimizer state I had to keep.

**4 bit teacher quantization** was the bigger memory saving. In my setup the loaded 32B teacher fell from roughly 64GB in bf16 to about 18GB.

But the flat losses used the teacher through a sequential generation path, one forward pass per token, and the quantized path made an already expensive loop worse. Those runs reached about 18.3 seconds per training step, against about 1.0 second for the tree loss path on the same model pair. Those are different execution paths, so that is not an 18 times quantization slowdown. What mattered to me was simpler: I had solved the memory problem and ended up with a wall clock problem instead.

The third tool was an **8 bit optimizer**, which shrank the memory used by Adam's state and sometimes gave me enough room to keep the teacher in bf16.

My first implementation made things worse. I allocated the optimizer state early, which overlapped with temporary startup allocations and hit an out of memory error on the very first step. I removed the warmup and let it allocate lazily. Same eventual state, different peak. One fit and one did not.

## Multiple GPUs did not make this distributed training

By this point I was using several GPUs, but each training run still lived on one GPU.

I used multiple cards to run independent experiments in parallel and packed smaller jobs together when memory allowed. I did not use tensor parallelism, pipeline parallelism, or NCCL collectives to split a single training step across GPUs.

That is orchestration, not distributed training.

The 32B teacher still fit on one 80GB A100. Once a model no longer fits on one card, the communication between GPUs becomes part of the problem. I did not build that system here.

## The lesson

I started by asking whether the weights fit in GPU memory. By the 32B runs I was thinking about weights, gradients, optimizer state, activations, KV cache, temporary allocations, and when each one existed.

The free tiers taught me that the final model can fit while the loading path does not. The A100 taught me that shrinking the weights can just move the problem somewhere else. The optimizer experiment taught me that allocation timing alone can decide whether a run fits.

I started the project asking how many parameters fit on a GPU. The better question turned out to be: **what is alive when memory peaks?**

---

Part of a series on building a speculative decoding for LLM inference research platform. Post 7.
