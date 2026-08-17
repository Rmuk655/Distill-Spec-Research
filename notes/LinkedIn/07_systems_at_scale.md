# Capacity planning: from a laptop GPU to an 80GB A100

In my [first post](https://medium.com/@rmukund16/engineering-an-llm-inference-research-platform-for-speculative-decoding-405d2f0bf53f), I described the hardware climb behind this project: my laptop, free GPUs, then A100s. I made that progression sound cleaner than it was. Most of it was me finding the next thing that did not fit.

## Why I started with models too small to matter

I started on my laptop GPU with a 0.6B draft and a 0.6B teacher.

That pair is not very useful for the research, since a teacher the same size as the draft teaches it almost nothing. But it was great for debugging locally: to make sure the training ran, gradients existed, checkpoints saved, and every loss path was exercised.

The next question was which model pair to choose given the hardware I could get. I found free [Colab](https://colab.research.google.com/) and [Kaggle](https://www.kaggle.com/) GPUs to spend cheap hardware on bugs before spending A100 time on them. Colab gives one T4 with about 15GB of GPU memory; Kaggle gives two T4s. But which pair actually fits, especially with a non trivial capacity gap between teacher and student? My mentor wanted us to aim for roughly a tenfold gap, which made a 0.6B draft against an 8B teacher the smallest serious pair in that family. Then the real question: could it even fit?

## How I tried to fit an 8B model on free hardware

An 8B teacher in bf16 is about 16GB of raw weights. A free T4 has about 15GB. So I reached for [4 bit quantization](https://huggingface.co/blog/4bit-transformers-bitsandbytes), which on raw weight arithmetic cuts roughly 16GB to 4GB. On paper it fit easily.

It still died, and not on GPU memory. In my loading path, each weight tensor was read into CPU memory at full width and then converted, so loading briefly needed far more system RAM than the final 4 bit model. Colab has about 12GB of system RAM, and it ran out before loading finished.

That was my first memory mistake: **I calculated what had to fit at the end, not what had to fit along the way.**

Kaggle had about 29GB of system RAM, enough headroom for that loading spike, so the 8B teacher in 4 bit actually loaded there. I also leaned on [LoRA](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora) and an 8 bit optimizer to shrink what the training step itself had to hold. But on free tiers it was hard to complete a meaningful training run at all. The full experiments moved to A100s.

## Why 8B was easy and 32B was not

The 8B teacher ran in plain bf16 on a 40GB A100 with none of the tricks I had needed on a T4; a 0.6B draft against a 16GB teacher fits that card easily.

The 1.7B draft with the 32B teacher did not. A 32B teacher in bf16 is about 64GB of weights alone, two bytes a parameter times 32 billion, which does not fit a 40GB card at all. On the 40GB card I had to load it in 4 bit just to get started. The 80GB card was the first with real headroom, and by that point I could run the teacher in bf16 and the draft in a plain full fine tune, no quantization at all.

Getting there taught me the accounting. What decides whether a run fits is not the sum of the final sizes but which allocations are alive at the same instant, the peak. Two large things coexisting can overflow a card that would have held either alone, and shrinking one weight can just move the peak somewhere else. The question I ended up asking was not "does the model fit?" but **"what is alive when memory peaks?"**

## Every way I made it fit had a catch

Three tools carried the fit on the tight machines, the T4 for the 8B teacher and the 40GB A100 for the 32B one.

**LoRA** reduced how much of the draft I trained, which also reduced the optimizer state I had to keep.

**4 bit teacher quantization** was the bigger memory saving. In my setup the loaded 32B teacher fell from roughly 64GB in bf16 to about 18GB. But the flat losses ran the teacher through a sequential generation path, and quantized weights get dequantized on every step, so those runs reached about 18.3 seconds per training step against about 1.0 for the tree loss path on the same pair. Those are different execution paths, so it is not a clean 18 times quantization cost. Either way, I had traded a memory problem for a wall clock one.

The third tool was an **8 bit optimizer**, which shrank Adam's state and sometimes gave me enough room to keep the teacher in bf16. My first implementation made things worse: I allocated the optimizer state early, which overlapped with temporary startup allocations and hit an out of memory error on the very first step. I removed the warmup and let it allocate lazily. Same eventual state, different peak. One fit and one did not.

These were rungs on the way up, not the final state. On the tight cards, the T4 and the 40GB A100, a teacher too big for the card had to be loaded in 4 bit, and LoRA and the 8 bit optimizer shaved the training that sat on top. As the cards grew I shed them one at a time. The 80GB A100 finally had the headroom to drop all three, a full fine tune with a full precision teacher and no quantization, which is the cleanest setup and the one I trusted most.

## Multiple GPUs did not make this distributed training

By this point I was using several GPUs, but each run still lived on one card. I used them to run independent experiments in parallel and packed smaller jobs together when memory allowed. I never split a single training step across GPUs with tensor parallelism, pipeline parallelism, or NCCL collectives.

That is orchestration, not distributed training.

The 32B teacher still fit on one 80GB A100. The next rung up would not. A 70B teacher, paired with an 8B draft to keep roughly the same tenfold gap, is about 140GB in bf16, past what any single 80GB card holds. No memory trick closes that, the model itself has to be [split across GPUs](https://huggingface.co/docs/transformers/en/perf_train_gpu_many) with tensor or pipeline parallelism, and the bandwidth between them becomes the constraint. That is real distributed training, and I did not build it here.

## The lesson

The free tiers taught me the final model can fit while the loading path does not. The A100 taught me that shrinking the weights just moves the problem somewhere else, and that allocation timing alone can decide whether a run fits.

I started the project asking how many parameters fit on a GPU. The better question was about the single worst instant, not the final total.

---

Part of a series on building a speculative decoding for LLM inference research platform. Post 7.
