# Capacity planning: from a laptop GPU to an 80GB A100

In my [first post](https://medium.com/@rmukund16/engineering-an-llm-inference-research-platform-for-speculative-decoding-405d2f0bf53f), I described the hardware climb behind this project: my laptop, free GPUs, then A100s. I made that progression sound cleaner than it was. Most of it was me finding the next thing that did not fit.

## Why I started with models too small to matter

I started on my laptop GPU with a 0.6B draft and a 0.6B teacher.

That pair is not very useful for the research, since a teacher the same size as the draft teaches it almost nothing. But it was great for debugging locally: to check the training ran, gradients existed, checkpoints saved, and every loss path was exercised.

The next question was which pair to run given the hardware I could get. I found free [Colab](https://colab.research.google.com/) and [Kaggle](https://www.kaggle.com/) GPUs to catch bugs cheaply before spending A100 time. Colab gives one T4 with about 15GB; Kaggle gives two. But which pair fits, especially with a real capacity gap between teacher and student? My mentor wanted roughly a tenfold gap, which made a 0.6B draft against an 8B teacher the smallest serious pair in that family. Could it even fit?

## How I tried to fit an 8B model on free hardware

An 8B teacher in bf16 is about 16GB of raw weights. A free T4 has about 15GB. So I reached for [4 bit quantization](https://huggingface.co/blog/4bit-transformers-bitsandbytes), which on raw weight arithmetic cuts roughly 16GB to 4GB. On paper it fit easily.

It still died, and not on GPU memory. In my Transformers and bitsandbytes loading path, quantizing the pretrained checkpoint needed far more temporary CPU memory than the final 4 bit model, and Colab's roughly 12GB of system RAM ran out before loading finished.

**The model fit after quantization. The process of getting it there did not.**

Kaggle had about 29GB of system RAM, enough headroom for that loading spike, so the 8B teacher in 4 bit actually loaded there. On top of it I trained the draft with [LoRA](https://huggingface.co/docs/peft/main/conceptual_guides/lora), which freezes the draft's base weights and trains only a small set of adapter parameters. The 4 bit quantization reduced the teacher's weight memory; LoRA reduced the gradients and optimizer state needed to train the draft. But free tiers still capped how far I could go, so the real experiments moved to [A100s](https://www.nvidia.com/en-us/data-center/a100/).

## Why 8B was easy and 32B was not

The 8B teacher ran in plain bf16 on a 40GB A100 with none of the tricks I had needed on a T4.

The 1.7B draft with the 32B teacher did not. A 32B teacher in bf16 is about 64GB of weights alone, two bytes a parameter times 32 billion, which does not fit a 40GB card. On the 40GB card I loaded it in 4 bit, cutting the weights to roughly 16GB plus metadata, enough to fit. The catch: flat losses run the teacher token by token, and 4 bit weights are dequantized on every step, so those runs hit about 18.3 seconds a step against 1.0 for the tree loss path on the same pair, trading a memory problem for a wall clock one. I tried a [bitsandbytes 8 bit optimizer](https://huggingface.co/docs/bitsandbytes/explanations/optimizers) to dodge it, storing the optimizer state in 8 bit to keep the teacher in bf16, though my first version allocated that state too early and hit an out of memory error on the first step. The 80GB A100 ended all of it: enough headroom to drop every trick and run the pair as a plain full fine tune with a full precision teacher.

Getting there taught me that memory is about the peak, not just the model size. Two large allocations existing at the same time can be enough to OOM. The question changed from "does the model fit?" to **"what is using memory when it fails?"**

## Multiple GPUs did not make this distributed training

By this point I was using several GPUs, but each run lived on one card, running independent experiments in parallel with smaller jobs packed together when memory allowed. I never split a single training step across GPUs with tensor parallelism, pipeline parallelism, or NCCL collectives.

That is orchestration, not distributed training.

The 32B teacher still fit on one 80GB A100. The next rung up would not. A 70B teacher, paired with an 8B draft to keep roughly the same tenfold gap, is about 140GB in bf16, past what any single 80GB card holds. No memory trick closes that, the model itself has to be [split across GPUs](https://huggingface.co/docs/transformers/en/perf_train_gpu_many) with tensor or pipeline parallelism, and the bandwidth between them becomes the constraint. That is real distributed training, and I did not build it here.

## The lesson

The free tiers taught me a model can fit at the end while the loading path does not. The A100s taught me that shrinking a weight just moves the problem, and that timing alone can decide a run.

I started by asking how many parameters fit on a GPU. The better question was the single worst instant, not the final total.

---

Part of a series on building a speculative decoding for LLM inference research platform. Post 7.
