# LinkedIn series: building a speculative decoding research platform

A 10 part series. Post 1 is the platform overview (the hub). Posts 2 to 10 each go deep on one capability a FAANG or frontier lab hiring loop actually screens for in an ML systems or ML research engineer, but only capabilities this project genuinely exercised. Two axes at once: breadth, the range of CS and ML disciplines a real research platform touches, and depth, how far below each abstraction the work actually went, since "can you go one layer below the tool you are using" is the exact question these interviews probe.

Hard rules for every post:
1. Point to real code, a real results CSV, or a real git commit for every number.
2. Never manufacture a capability the project did not exercise. Where a capability was not exercised, it goes in the "what this series does not cover" section, stated plainly, never implied into existence.
3. Include a further reading link to the real paper or resource behind the underlying theory, the way the flagship post already does. A post that teaches a concept (KV cache, pass@k, an OTLP verifier, an acceptance objective) should point at where that concept actually comes from, not just describe it in isolation.

## Reading order (final, files match this exactly)

1. **Platform overview** (the hub). The whole system end to end.
2. **`02_block_efficiency_vs_tokens_per_second.md`** — Isolating variables: my piece of the puzzle, not the whole one. Experimental design, choosing a metric that isolates the algorithm from hardware, serving stack, and harness. Ends on the partial derivative framing. Grounded: profiling row in `jsd_mathhard_s123.csv` (766s draft, 109s target, 31.6% SM util); the 8B vs 32B teacher timing/utilization comparison.
3. **`03_gpu_profiling_and_roofline.md`** — GPU profiling and the roofline: why decode was not even saturating the GPU. Grounded: `telemetry.py` (NVML SM util, memory util, PCIe TX/RX, NVLink, clocks, temperature); the 8B vs 32B roofline read (49ms vs 86ms per call, 31.6% vs 33.5% SM util at 4x the parameters, neither compute nor memory saturated, dispatch bound instead). Names the CUDA graphs / fused kernel lever this points at but does not attempt.
4. **`04_rounding_error_flashattention_sdpa.md`** — Numerics and kernels: the 0.2 gain that was a rounding error. Two separate stories: flat losses, where FA2 and SDPA both run but round differently in bf16 (46 matched cells, −0.23 to +0.37 from the kernel alone); and tree losses, where FA2 was never usable at all, since the draft's custom tree mask forward pass needs a dict based attention mask FA2 does not support, forcing SDPA specifically there.
5. **`05_eval_contention_bug.md`** — Operating systems: the checkpoint that scored differently every run. Process and resource contention, GPU sharing, scheduling. Grounded: `telemetry.py` `cpu_util_avg_pct`; the one eval per GPU fix.
6. **`06_experimental_methodology.md`** — The sweep, the seeds, and the eval set you did not train on. Hyperparameter sweep as the exploration tool, then the statistics of reading it: sample size, seeds, datasets (an eval time axis, not a training time one, since every checkpoint trained on math_hard regardless), model pairs, and the honest model family limit. Grounded: `hyperparam_sweep_research_report.md`, `results/passK/passk_hparam_sweep.csv` (LR swinging gsm8k pass@1 by 0.225 while pass@64 converges; the deployed pick sitting inside its own noise floor while a hindsight pick clears it); noise floor SE; +0.293 to +0.03 mislabeled K correction (`tree_losses_research_note.md`, `interview_prep.md`); cross pair replication failure (`enrich_research_note.md`); gsm8k vs olympiad dataset headroom (`passk_prob_vs_jsd_research_note.md`).
7. **`07_systems_at_scale.md`** — Capacity planning: fitting a 32B model, then serving it. Memory budget across four GPU tiers, KV cache, vLLM, and the honest orchestration-versus-distributed-training distinction (every model fit on one GPU, no NCCL, no tensor/pipeline parallelism). Grounded: the 16GB Colab to 80GB A100 progression; `passk` pipeline commits (async vLLM, isolated venv, resumable).
8. **`08_ml_breadth_and_depth.md`** — 34 objectives against 9 verifiers, and why the aligned ones lost. The full taxonomy, then the metric-aligned-loss failure. Grounded: `verifiers/verifier.py` docstring (9 verifiers, OTLP vs non-OTLP); `losses/tree.py` / `losses/flat.py` registries; the log space collapse (−0.4 to −0.6 at 1.7B/32B) and the gradient allocation explanation.
9. **`09_two_training_ideas_that_failed.md`** — Research judgment from failure: reinforcement learning and curriculum weighting, why the data falsified them before any clean argument could. Grounded: `tree_losses_research_note.md` (tree_pg REINFORCE monotonic decline), `depth_weight_research_note.md` (detached scalar cannot steer gradient direction).
10. **`10_what_worked_and_one_metric.md`** — What worked, and why one metric was not enough to trust it. The finale. Enrichment, delayed branching (including the new both-axes-at-once evidence, not just a BE lift), pass@k disagreeing with block efficiency, and the coupled system thesis stated explicitly, tying the whole series together. Grounded: `enrich_research_note.md` (M=4 sweet spot, OOD transfer), `ddte_research_note.md` (delayed branching on trained drafts, corrected uplift figures), `results/passK/LandL1sweepJsd.csv` (delayed branching raising block efficiency and throughput together), `passk_prob_vs_jsd_research_note.md` (pass@1 up while pass@64 down on specific checkpoints, not universally; no significant signal across most settings).

## FAANG capability map: what each post is evidence for

| Capability a hiring loop looks for | Post | The real artifact behind it |
|---|---|---|
| Experimental design, metric awareness | 2, 6 | block efficiency choice; hyperparameter sweep + noise floor discipline |
| GPU hardware, profiling, roofline | 3 | telemetry harness, SM/mem util, PCIe traffic, 8B vs 32B roofline read |
| Numerical behavior, kernel level systems | 4 | bf16 backend divergence, FA2 vs custom tree mask incompatibility |
| Operating systems, scheduling, contention | 5 | eval contention bug, one eval per GPU |
| Statistical rigor, research maturity | 6, 10 | seeds, replication, model family limit, pass@k disagreement |
| Systems at scale, memory, serving | 7 | memory budget across four GPU tiers, KV cache, vLLM, orchestration |
| ML breadth (objective and verifier taxonomy) | 8 | 34 objectives x 9 verifiers grid |
| ML depth, gradient behavior | 8, 9 | why aligned losses and RL failed, gradient allocation |
| Research judgment, failure analysis | 9, 10 | falsified hypotheses, coupled system thesis |

## What this series does not cover (stated plainly, not implied away)

These are real capabilities a top loop may probe that this project did not exercise. The series must not imply them. They are honest next steps, not current evidence:

- CUDA or Triton kernel authoring. Every kernel used was someone else's, called and profiled, never written. Post 3's own profiling names the specific opening this leaves: the workload is dispatch bound, which is exactly what CUDA graphs or a fused kernel are for, a concrete identified lever to raise real tokens per second, not attempted. State it as follow up work, not as a capability the series demonstrates.
- NCCL, collectives, tensor or pipeline parallelism, distributed training. Every model fit on one GPU, so this was single GPU orchestration, not distributed computation. The telemetry harness can measure NVLink but the eval data has those columns empty for exactly this reason.
- Production serving. Reliability, p99 latency, failover, autoscaling, multi tenancy. This was a research platform with crash recovery, not a production fleet.
- C++, and general DS&A / SWE interview performance. Not established by the writeups either way.
- A novel, peer validated research result. The honest contribution is a rigorous negative-heavy study plus two modest positive signals, not a new state of the art.

If the six month target is the OpenAI or Anthropic tier bar specifically, rather than the FAANG new grad tier, closing the first two of these is where actual new work belongs, not more writing about the existing project.

## Closing paragraph to append to the hub (Post 1)

Add this at the end of the platform overview so it points forward to the rest of the series:

> I am writing up the pieces of this platform one at a time, roughly one post a week. Each one takes a single line from above and tells the full story behind it. Follow along if you want the details.

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
10. Every post that teaches a concept links to the real paper or resource it comes from, not just a description in isolation.

## Data mining log (for traceability, not for publishing)

Three research agents mined `notes/` and `results/` in full for this restructure. Their verified findings, and two real corrections they surfaced in the source notes (not just the LinkedIn posts), are folded into the posts above. Worth recording here in case of a future audit:
- `enrich_research_note.md` line 36 overclaimed "+0.94 is the largest DDTE uplift observed anywhere in the program." The committed CSV (`jsd_enrich_ddte_results.csv`) contains larger uplifts (+1.220, +1.266) from standard enrich, already documented elsewhere in `ddte_research_note.md`. Fixed in the note directly, not just worked around in the LinkedIn text.
- The "block efficiency rises with L while throughput falls" framing, used in an earlier draft of post 2's chart before it was replaced, was directionally true only for strong verifiers (bv, traversal); weak verifiers (nss, khisti) flatline or slightly decline on block efficiency as L rises (`results/passK/LandL1sweepJsd.csv`). Not currently live in any post, flagged here so it is not reintroduced without the caveat.
