# Capacity planning: from a laptop GPU to an 80GB A100

In my [first post](https://medium.com/@rmukund16/engineering-an-llm-inference-research-platform-for-speculative-decoding-405d2f0bf53f), I summarized the hardware climb behind this project: free GPUs, then a 40GB A100, then an 80GB one. What I skipped was why I kept moving. Each machine hit a different memory limit, and along the way I picked up three ways to make the runs fit: LoRA to train fewer draft parameters, 4 bit teacher quantization to shrink the teacher weights, and an 8 bit Adam optimizer to shrink the draft's optimizer state. As the hardware got larger, I could remove those compromises one by one. The model size and hardware progression taught me what was actually consuming the memory.

## Why I started with models too small to matter

I started on my laptop GPU with a 0.6B draft and a 0.6B teacher.

That pair is not very useful for the research, since a teacher the same size as the draft teaches it almost nothing. But it was great for debugging locally: to check the training ran, gradients existed, checkpoints saved, and every loss path was exercised.

The next question was which pair to run given the hardware I could get. I found free [Colab](https://colab.research.google.com/) and [Kaggle](https://www.kaggle.com/) GPUs to catch bugs cheaply before spending A100 time. Colab gives one T4 with about 15GB; Kaggle gives two. But which pair fits, especially with a real capacity gap between teacher and student? My mentor wanted roughly a tenfold gap, which made a Qwen3 0.6B draft against a Qwen3 8B teacher the smallest serious pair in that family. Could it even fit?

## How I tried to fit an 8B model on free hardware

An 8B teacher in bf16 is about 16GB of raw weights. The T4 I got on free Colab had about 15GB of GPU memory, so it could not fit there in bf16. I reached for [4 bit NF4 quantization](https://huggingface.co/docs/transformers/quantization/bitsandbytes), which on raw weight arithmetic cuts roughly 16GB to 4GB. On paper it fit easily.

It still died, and this time GPU memory was not the problem. My Colab VM had only about 12.7GB of CPU RAM. In my Transformers and bitsandbytes loading path, quantizing the pretrained checkpoint needed substantially more temporary CPU memory than the final 4 bit model, and the VM ran out of system RAM before loading finished.

I thought I needed a bigger GPU. The first thing I actually ran out of was CPU RAM. My mistake was budgeting for the model after loading, not the memory needed while loading it.

Kaggle had about 29GB of system RAM, enough headroom for that loading spike, so the 8B teacher in 4 bit actually loaded there. On top of it I trained the draft with [LoRA](https://huggingface.co/docs/peft/main/conceptual_guides/lora), which freezes the draft's base weights and trains only a small set of adapter parameters. The 4 bit quantization reduced the teacher's weight memory; LoRA reduced the gradients and optimizer state needed to train the draft. But free tiers still capped how far I could go, so the real experiments moved to [A100s](https://www.nvidia.com/en-us/data-center/a100/).

## Why 8B was easy and 32B was not

The 8B teacher ran in plain bf16 on a 40GB A100 with none of the tricks I had needed on a T4.

The 1.7B draft with the 32B teacher did not. A 32B teacher in bf16 is about 64GB of weights alone, two bytes a parameter times 32 billion, which does not fit a 40GB card, so there I loaded it in 4 bit, cutting the weights to roughly 16GB plus metadata. The catch: flat losses run the teacher token by token, and 4 bit weights are dequantized on every forward pass, so those runs hit about 18.3 seconds a step against 1.0 for the tree loss path on the same pair, trading a memory problem for a wall clock one.

The 80GB A100 was the real fix: the bf16 teacher fit there without quantizing. Fitting the draft's training alongside it was tight, and one thing I tried, a [bitsandbytes 8 bit optimizer](https://huggingface.co/docs/bitsandbytes/explanations/optimizers) to shrink the optimizer state, backfired at first: it allocated that state at startup, before the model loading spike had cleared, and hit a GPU out of memory error on the first step. Once I let it allocate lazily, the 80GB had the headroom to run the pair as a plain full fine tune, no quantization at all.

Getting there taught me that memory is about the peak, not just the model size. Two large allocations at the same time can be enough to OOM. The question changed from "does the model fit?" to **"what is using memory when it fails?"**

## What happens when one GPU is no longer enough?

The 32B teacher still fit on one 80GB A100. The next rung up would not. A 70B teacher is about 140GB of weights in bf16, already past what a single 80GB card can hold before the draft or any training state exists.

At that point the problem changes. The model itself has to be split across GPUs. [Tensor parallelism](https://huggingface.co/docs/transformers/en/perf_train_gpu_many#tensor-parallelism) splits work within layers across GPUs, while [pipeline parallelism](https://huggingface.co/docs/transformers/en/perf_train_gpu_many#pipeline-parallelism) puts different groups of layers on different GPUs. Once GPUs have to cooperate on the same training step, communication matters too, using libraries such as NCCL for operations like all-reduce.

I never had to do that here. I used several GPUs, but each run still lived on one card. The GPUs ran independent experiments in parallel, with smaller jobs packed together when memory allowed. That was multi-GPU orchestration, not distributed training.

## The lesson

The free tiers taught me a model can fit at the end while the loading path does not. The A100s taught me that shrinking a weight just moves the problem, and timing alone can decide a run.

I started by asking how many parameters fit on a GPU. The better question was the worst instant, not the final total.

---

Part of a series on building a speculative decoding for LLM inference research platform. Post 7.
