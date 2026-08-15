# GPU profiling: the drafter dominated runtime, not the teacher

Post 2 showed GPU compute utilization staying flat around 32 percent, whether the teacher was 8B or 32B. I assumed the big teacher was the expensive part of every run. Profiling across the whole experiment grid, not just that one comparison, said otherwise.

## What I expected

Going from an 8B teacher to a 32B one is four times the parameters. I expected the bigger model to take over as the dominant cost: more math, more weight bytes to move, more GPU time spent per run.

## What the whole grid showed

Over 3,200 eval runs, spanning different losses, verifiers, K, L, and checkpoints, the small draft model consumed most of the measured draft-plus-target inference time, not the large teacher. Draft time was 78 to 94 percent of combined draft-plus-target time, averaging 85 percent, and the draft never stopped being the majority of the time in any single sweep. Even after making the teacher four times larger, the draft was still where most of the measured time went. That is the actual surprise here, not a specific percentage.

Target cost nearly doubled as expected going from the 8B teacher to the 32B one, 48ms to 84ms per block. Draft cost did not move with it, 344ms with the 0.6B draft against 333ms with the 1.7B one, roughly flat. Making the teacher four times larger increased target latency substantially, but not enough to make it the dominant cost: draft share fell only from 88 to 80 percent.

With an increase in tree depth, draft time increased far more than target time: going from L=16 to L=32, draft time per block nearly doubled, 575ms to 1122ms, while target time rose only 56ms to 81ms. Of the additional wall time added by going deeper, 95 percent came from the draft, 3 percent from the target, 2 percent from verification. The reason is execution structure, not model size: the draft constructs the tree through sequential depth steps, with the K branches at each step batched together, while the target evaluates the finished tree in one batched pass, once. Model size was not predicting the bottleneck. Execution structure was.

## Why 12 percent does not mean "barely using memory"

The 32B teacher held about 66 GB on an 80 GB card, yet its memory-bus reading was 13.2 percent, barely above the 8B teacher's 12.2, and compute utilization sat at 31 to 34 percent for both. These readings come from NVML, the same driver library behind `nvidia-smi`. They are activity counters, the percent of time each resource was doing anything at all, not measurements of achieved FLOPs or memory bandwidth, so they cannot by themselves identify a bottleneck. NVML utilization could not identify the bottleneck. Decomposing wall-clock time did.

Low activity on both counters is consistent with the GPU spending most of its time between small draft calls rather than continuously busy on either kind of work. PCIe traffic rules out one culprit in the meantime: 77 MB/s back from the GPU against 17 MB/s out, both orders of magnitude below link capacity, so bandwidth was not saturated either.

## When does block efficiency actually predict throughput?

Hold tree depth fixed, only the checkpoint varies within one verifier at a time: block efficiency and throughput move together tightly, mean within-group correlation 0.96 across 33 groups (three checkpoints per group). Different verifiers do move block efficiency by different amounts and cost somewhat differently to run, but that cost difference is small next to the draft's own, which is most of a block's total time. Tree depth is what actually breaks the proxy, because it restructures the draft's cost directly: the L=16 to L=32 result above is the clearest case, block efficiency rose while throughput fell.

This was not specific to traversal, the verifier used above. Across eleven verifiers at the same tree shape, throughput fell from L=8 to L=32 in every single case, and block efficiency rose in ten of eleven, specinfer was essentially flat past L=16.

![Block efficiency and throughput move in opposite directions as tree depth grows, for every verifier](../../results/passK/linkedin_post3_be_tps_by_verifier_across_L.png)
*Blue: block efficiency. Red: throughput. Same shape for every verifier tested.*

![Block efficiency predicts throughput only when cost is held fixed](../../results/passK/linkedin_post3_be_tps_correlation_collapse.png)
*Left: one configuration, three checkpoints, tree depth and verifier fixed. Right: the same verifier at two tree depths pooled together.*

Block efficiency is a reliable proxy for throughput within one fixed tree depth and verifier, the two things that set how much draft and target work a block actually costs. It stops being one across different tree depths, the clearest case here: going from L=16 to L=32, it points the opposite way from throughput.

## What I would test next

No CUDA timeline trace, so I know where the time went, not yet the exact low-level cause, a tool like Nsight Systems is the obvious next measurement. Two optimizations used by production serving engines are absent here too: [continuous batching](https://www.usenix.org/conference/osdi22/presentation/yu), the technique behind vLLM, could amortize small draft steps across multiple prompts instead of evaluating one at a time, and [CUDA graphs](https://pytorch.org/docs/stable/notes/cuda.html#cuda-graphs) could reduce repeated launch overhead by replaying captured GPU work instead of dispatching it fresh every step. I have not measured how much either would recover, that needs the timeline trace first. I also only ever trained 0.6B and 1.7B drafts, two points, not a scaling curve on how draft size affects draft latency.

## The lesson

A higher block efficiency does not mean higher throughput once the cost of producing the tree changes, tree depth was the clearest example, deeper trees accepted more tokens per target call while making the system slower. Do not infer the bottleneck from model size, and do not stop at the metric you optimized. Profile the path that actually determines wall-clock time.

---

Part of a series on building a speculative decoding research platform. Post 3.
