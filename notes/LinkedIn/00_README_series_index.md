# LinkedIn series: building a speculative decoding research platform

A 10 part series. Post 1 is the platform overview (the hub). Posts 2 to 10 each go deep on one capability a FAANG or frontier lab hiring loop actually screens for in an ML systems or ML research engineer, but only capabilities this project genuinely exercised. Two axes at once: breadth, the range of CS and ML disciplines a real research platform touches, and depth, how far below each abstraction the work actually went, since "can you go one layer below the tool you are using" is the exact question these interviews probe.

Hard rule for every post: point to real code, a real results CSV, or a real git commit. Never manufacture a capability the project did not exercise. Where a capability was not exercised, it goes in the "what this series does not cover" section below, stated plainly, not implied into existence.

## Reading order (the 10)

1. **Platform overview** (the hub). The whole system end to end.
2. **Isolating variables: block efficiency, not tokens per second.** Experimental design, choosing a metric that isolates the algorithm from hardware, serving stack, and harness. Ends on the partial derivative framing. Grounded: profiling row in `jsd_mathhard_s123.csv` (766s draft, 109s target, 31.6% SM util).
3. **GPU profiling and the roofline: why decode was not even saturating the GPU.** GPU and hardware depth, the single biggest under-shown area. The telemetry harness plus a roofline read. Grounded: `telemetry.py` (NVML SM util, memory util, PCIe TX/RX, NVLink TX/RX across up to 12 links, SM/memory clocks, temperature); CSV columns `gpu_sm_util_avg_pct`, `gpu_mem_util_avg_pct`, `gpu_pcie_tx/rx_avg_kbs`; the 8B vs 32B teacher comparison (49ms vs 86ms per target call, 31.6% vs 33.5% SM util at 4x the parameters). Honest note the post must make: SM util 31.6% with memory util only 12.2% means at batch 1 with a tiny draft the run was dispatch bound, not even memory bandwidth saturated, and the NVLink columns are empty because this was single GPU, no interconnect traffic to measure.
4. **Numerics and kernels: the 0.2 gain that was a rounding error.** Scientific computing plus kernel level systems. Grounded: git `fix(eval): default --force_attn to sdpa`; the backend divergence figure (46 matched cells, −0.23 to +0.37 from the kernel alone); and the deeper fact, FlashAttention could not be enabled globally because the draft's custom tree mask forward pass uses a dict based attention mask FA2 does not support, so SDPA had to be forced there specifically (git `fix(train): force SDPA for the draft's custom tree-mask`).
5. **Operating systems: the checkpoint that scored differently every run.** Process and resource contention, GPU sharing, scheduling. Grounded: `telemetry.py` `cpu_util_avg_pct`; the one eval per GPU fix.
6. **Statistics: what it takes to call an improvement real.** Sample size, seeds, datasets, model pairs, and the honest model family limitation. Grounded: noise floor SE; +0.293 to +0.03 mislabeled K correction (`tree_losses_research_note.md`, `interview_prep.md`); cross pair replication failure (`enrich_research_note.md`); the Qwen only limitation stated as unmet.
7. **Systems at scale: fitting a 32B model and serving it.** Capacity planning plus serving systems, and the honest orchestration versus distributed distinction. Merges the old memory post and the KV cache / vLLM post. Grounded: the 16GB Colab to 80GB A100 progression, shedding LoRA then optimizer quantization then teacher quantization; `passk` pipeline commits (async vLLM, isolated venv, resumable); the plain statement that this was single GPU job orchestration and crash recovery, not NCCL distributed training.
8. **ML breadth and depth: 34 objectives against 9 verifiers, and why the aligned ones failed.** Merges the taxonomy post and the metric aligned loss post. Grounded: `verifiers/verifier.py` docstring (9 verifiers, OTLP vs non OTLP); `losses/tree.py` and `losses/flat.py` registries (the exact objective count and categories); the log space collapse (−0.4 to −0.6 at 1.7B/32B) and the gradient allocation explanation.
9. **Research judgment from failure: two ideas that failed, and why.** Reinforcement learning and curriculum weighting, why the data falsified them before any clean argument could. Grounded: `tree_losses_research_note.md` (tree_pg REINFORCE monotonic decline), `depth_weight_research_note.md` (detached scalar cannot steer gradient direction).
10. **What worked, and why one metric was not enough to trust it.** Merges the enrichment/DDTE result post and the pass@k post. The result plus the coupled system thesis that ties the whole series together. Grounded: `enrich_research_note.md` (M=4 sweet spot, OOD transfer), `ddte_research_note.md` (delayed branching on trained drafts), `passk_prob_vs_jsd_research_note.md` (pass@1 up while pass@64 down, no significant signal across most settings).

## FAANG capability map: what each post is evidence for

The point of the reorder is that the series now covers a spread of what a hiring loop screens for, with a concrete artifact behind each, rather than eleven variations on one theme.

| Capability a hiring loop looks for | Post | The real artifact behind it |
|---|---|---|
| Experimental design, metric awareness | 2, 6 | block efficiency choice; noise floor discipline |
| GPU hardware, profiling, roofline | 3 | telemetry harness, SM/mem util, PCIe traffic, 8B vs 32B roofline read |
| Numerical behavior, kernel level systems | 4 | bf16 backend divergence, FA2 vs custom tree mask incompatibility |
| Operating systems, scheduling, contention | 5 | eval contention bug, one eval per GPU |
| Statistical rigor, research maturity | 6, 10 | seeds, replication, model family limit, pass@k disagreement |
| Systems at scale, memory, serving | 7 | memory budget across four GPU tiers, KV cache, vLLM, orchestration |
| ML breadth (objective and verifier taxonomy) | 8 | 34 objectives x 9 verifiers grid |
| ML depth, gradient behavior | 8, 9 | why aligned losses and RL failed, gradient allocation |
| Research judgment, failure analysis | 9, 10 | falsified hypotheses, coupled system thesis |

## What this series does not cover (stated plainly, not implied away)

These are real capabilities a top loop may probe that this project did not exercise. The series must not imply them. They are the honest next steps, not current evidence:

- CUDA or Triton kernel authoring. Every kernel used was someone else's, called and profiled, never written.
- NCCL, collectives, tensor or pipeline parallelism, distributed training. Every model fit on one GPU, so this was single GPU orchestration, not distributed computation. The telemetry harness can measure NVLink but the eval data has those columns empty for exactly this reason.
- Production serving. Reliability, p99 latency, failover, autoscaling, multi tenancy. This was a research platform with crash recovery, not a production fleet.
- C++, and general DS&A / SWE interview performance. Not established by the writeups either way.
- A novel, peer validated research result. The honest contribution is a rigorous negative-heavy study plus two modest positive signals, not a new state of the art.

## Migration status (files vs this plan)

The post files were originally drafted as 12 (hub plus 11, including 7a). This plan collapses them to 10. Still to do:
- NEW: write post 3, the GPU profiling and roofline post (the highest leverage gap filler, all data already in the CSVs).
- MERGE: old post 6 (32B memory) + old post 7 (KV cache/vLLM) into new post 7.
- MERGE: old post 7a (taxonomy) + old post 8 (aligned loss failed) into new post 8.
- MERGE: old post 10 (what worked) + old post 11 (pass@k) into new post 10.
- Only post 2 and post 3-old (noise floor, now new post 6) have had a full content pass. The rest have had structural fixes only.

## Voice rules (keep every post consistent)

1. Write like a student explaining what he learned, not like a textbook. Short sentences. First person.
2. No hyphens, no contractions, no em dashes. These are the fastest way to sound generated.
3. Define any term the first time it appears, in plain words.
4. Start in the middle of the problem, not with a grand intro.
5. Show what failed. Every post should admit at least one thing I got wrong.
6. Every number is real and traceable to the notes, the results CSVs, or the git log. No invented figures.
7. Drop hype words: crucial, testament, landscape, robust, seamless, delve.
8. Only reference already published posts, never a later one, with no exceptions. The series footer should only say which post this is, nothing about what comes after.
9. Only claim capabilities the project actually exercised. If it was not done, it belongs in the "does not cover" section, not implied into a post.
