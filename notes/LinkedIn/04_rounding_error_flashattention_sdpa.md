# Scientific computing: how a faster attention backend faked a 0.2 gain

In [my platform overview](https://medium.com/@rmukund16/engineering-an-llm-inference-research-platform-for-speculative-decoding-405d2f0bf53f), I flagged this one in a sentence: FlashAttention versus SDPA looked like just a speed choice, until two runs that differed only in the attention backend diverged by about 0.2 block efficiency, enough to look like a real result when it was only bf16 rounding. Here is the full story.

My evaluation was slow. One verifier on one hundred prompts took fifteen to twenty minutes, nine verifiers per checkpoint was close to three hours, and across three datasets it was most of a day. I had dozens of checkpoints. So I went looking for something faster, and found FlashAttention. I set it up on one of my machines and started moving runs onto it, expecting a free speedup and nothing else. I was wrong.

## What SDPA and FlashAttention are

Attention is one of the biggest compute and memory costs in these models, and there is more than one way to implement it. I was using PyTorch's SDPA path, its built in default. [FlashAttention](https://huggingface.co/docs/text-generation-inference/en/conceptual/flash_attention) is an alternative that computes attention in tiles, avoiding writing the full attention matrix to the GPU's high bandwidth memory. Same math, different order of arithmetic. The FlashAttention build I was using needed an Ampere generation GPU or newer, which is why it was an option on my A100 and H100 but not the older free tier cards I started on.

## The accident: same everything, two different scores

I was running the same evals across two machines to clear the backlog. Then I saw something that should have been impossible: the same checkpoint, same seed, same verifier, same prompts, scored two different block efficiencies depending on which machine ran it. I was not testing backends. I stumbled into this.

The cause was silent. Both machines could run FlashAttention, but I had only installed it on one of them, and the library auto picks the backend based on whether it is installed. So the same code quietly ran two different implementations. These models run in bf16, which has far less precision than fp32, and doing the same math in a different order can change the last bits. Normally that does not matter.

But speculative decoding eventually turns those probabilities into a discrete decision: accept this token or reject it. Near one of those decision boundaries, a last bit difference can flip the outcome. Once one token flips, the rest of the generation follows a different path. The paired differences reached about 0.2 block efficiency, comparable to my run to run noise floor, and they went in both directions. FlashAttention was not making the model better; changing the backend was perturbing the result enough to masquerade as a model level effect.

The cross machine version of this is confounded, of course. Two machines differ in more than the library. So I isolated it: I ran both backends on the same A100, same checkpoint, same seed, same prompts, and toggled only whether FlashAttention was in use. The divergence stayed. With the hardware and the rest of the setup held fixed, changing the attention backend was enough to reproduce it.

![Same A100, same checkpoint, same seed, same prompts, only the attention backend changed](../../results/passK/linkedin_post5_backend_divergence.png)
*Each point is one evaluation cell, both backends run on the same A100. Block efficiency under SDPA on the x axis, under FlashAttention on the y axis. If the backend did not matter, every point would sit on the dashed line. The red ones land outside the run to run noise band, from nothing but the backend.*

## The catch: it was never pure FlashAttention

I said I moved runs onto FlashAttention, but it was never pure FlashAttention. It was always a mix: the draft on FlashAttention, the target on SDPA. Here is why the target could not switch.

To verify a whole tree of proposed tokens in one pass, the target uses a custom mask saying which tokens are ancestors of which. This comes from the verification step itself, not the training loss, so it holds for every checkpoint. The FlashAttention path in my stack did not support that custom tree mask, only ordinary causal or padding masks, so it crashed on the tree. That is not a fundamental limit, specialized serving kernels do support these speculative decoding patterns. I had no such kernel, only the default library, so the target stayed on SDPA and only the small draft could switch.

## The lesson

I standardized on SDPA everywhere. It runs on every GPU I had, and it makes every run comparable to every other run. I gave up FlashAttention's speed to get that. When I audited old results, some had mixed backends, so I flagged those rather than trust the difference as a model effect.

This is really a reproducibility problem, and it reaches past the backend. [My earlier post](02_block_efficiency_vs_tokens_per_second.md) said to hold the hardware and serving stack fixed and move only the algorithm. But on a rented cloud machine you never get the same physical box twice, so you hold the software fixed instead: a setup script and a pinned requirements file that install the exact same torch and transformers every run, FlashAttention present or absent on purpose, not by accident. I pin those versions because an unpinned torch drifted between installs and quietly made throughput non comparable. Pinning does not make two rented A100s identical, drivers, firmware, and clocks still vary, but it removes the one source of variation that was actually mine to control.

Low precision differences are usually harmless. But put a hard decision boundary downstream, and speculative decoding is exactly that, a tiny numerical difference becomes a measurable experimental signal.

---

Part of a series on building a speculative decoding for LLM inference research platform. Post 4.
