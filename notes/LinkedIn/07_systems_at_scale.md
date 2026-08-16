# Capacity planning: the model got smaller and still would not load

In my [first post](https://medium.com/@rmukund16/engineering-an-llm-inference-research-platform-for-speculative-decoding-405d2f0bf53f), I described the hardware climb behind this project: my laptop, free GPUs, then A100s. I made that progression sound cleaner than it was. Most of it was me finding the next thing that did not fit.

## Why I started with models too small to matter

I started on my laptop GPU with a 0.6B draft and a 0.6B teacher.

That pair was useless for the research, since a teacher the same size as the draft teaches it almost nothing. It only checked that the loop ran, gradients existed, checkpoints saved, and every loss path finished.

Then I moved to free Colab and Kaggle GPUs with smaller teachers. The idea was simple: spend cheap hardware on bugs before spending A100 time on them.

It worked until the bug was memory itself. You cannot debug not having enough memory on a machine that does not have enough memory.

## Why a 4 bit model still ran out of memory

An 8B teacher in bf16 is about 16GB of raw weights. A free Colab T4 has about 15GB of GPU memory, so I reached for [4 bit quantization](https://huggingface.co/blog/4bit-transformers-bitsandbytes).

On raw weight arithmetic that cuts roughly 16GB to 4GB. On paper it fit easily.

It still died.

The problem was not GPU memory. In my loading path, each weight tensor was read into CPU memory at full width and then converted, so loading briefly needed far more system RAM than the final 4 bit model. Colab has about 12GB, and it ran out before loading finished.

That was my first memory mistake: **I calculated what had to fit at the end, not what had to fit along the way.**

Kaggle had about 29GB of system RAM, enough headroom for that loading spike, so the 8B teacher in 4 bit actually loaded there. But free tiers cap your session and kick you off, so the free machines stayed useful mainly as scaffolding. The full experiments moved to A100s.

## Why 8B was easy and 32B was not

The 8B teachers ran in bf16 on the A100 with none of the tricks I had collected on the way up. The 1.7B draft with the 32B teacher was where I needed all of them at once.

A 32B teacher in bf16 is about 64GB of weights alone, two bytes a parameter times 32 billion. On an 80GB A100 that leaves little room for anything else, and the teacher is only part of the bill. It is frozen, but the draft is training, so there are gradients, activations, and optimizer state, and generation needs a KV cache.

What decides whether the run fits is not the sum of those sizes. It is which of them are alive at the same instant, the single worst moment, the peak. Two large allocations existing at the same time can overflow a card that would have held either one alone. The question changed from "does the model fit?" to **"what is alive when memory peaks?"**

## Every way I made it fit had a catch

I used three tools.

**LoRA** reduced how much of the draft I trained, which also reduced the optimizer state I had to keep.

**4 bit teacher quantization** was the bigger memory saving. In my setup the loaded 32B teacher fell from roughly 64GB in bf16 to about 18GB.

But the flat losses used the teacher through a sequential generation path, one forward pass per token, and the quantized path made an already expensive loop worse. Those runs reached about 18.3 seconds per training step, against about 1.0 second for the tree loss path on the same model pair. Those are different execution paths, so that is not an 18 times quantization slowdown. But the point was simpler: I had solved the memory problem and created a wall clock problem instead.

The third tool was an **8 bit optimizer**, which shrank the memory used by Adam's state and sometimes gave me enough room to keep the teacher in bf16.

My first implementation made things worse. I allocated the optimizer state early, which overlapped with temporary startup allocations and hit an out of memory error on the very first step. I removed the warmup and let it allocate lazily. Same eventual state, different peak. One fit and one did not.

None of these were invented for the 32B run. They were tools I had picked up earlier, each on a machine that had already told me no, and together they finally got the 32B pair onto a single 80GB A100.

## Multiple GPUs did not make this distributed training

By this point I was using several GPUs, but each run still lived on one card. I used them to run independent experiments in parallel and packed smaller jobs together when memory allowed. I never split a single training step across GPUs with tensor parallelism, pipeline parallelism, or NCCL collectives.

That is orchestration, not distributed training.

The 32B teacher still fit on one 80GB A100. The next rung up would not. A 70B teacher, paired with an 8B draft to keep roughly the same tenfold gap, is about 140GB in bf16, past what any single 80GB card holds. No memory trick closes that, the model itself has to be split across GPUs, and then the [bandwidth between them](https://developer.nvidia.com/blog/nvidia-nvlink-and-nvidia-nvswitch-supercharge-large-language-model-inference/) becomes the constraint. That is real distributed training, and I did not build it here.

## The lesson

The free tiers taught me the final model can fit while the loading path does not. The A100 taught me that shrinking the weights just moves the problem somewhere else. The optimizer experiment taught me that allocation timing alone can decide whether a run fits.

I started the project asking how many parameters fit on a GPU. The better question was always about the single worst instant, not the final total: same parts, different peak, different answer.

---

Part of a series on building a speculative decoding for LLM inference research platform. Post 7.
