# Lessons from an LLM Inference Research Internship: Speculative Decoding, Distillation, and Evaluation

During my second-year summer, I worked on LLM inference optimization through speculative decoding, under the guidance of Rahul [@Rahul], a researcher at Columbia University.

Over eight weeks I:

- Ran 500+ tracked experiments (about 250 training runs), logged in Weights & Biases
- Evaluated 71 trained checkpoints across 4,000+ measurements
- Built a multi-GPU experiment and evaluation pipeline in PyTorch and HuggingFace, with generation served through vLLM, growing to roughly 10,000 lines of training, evaluation, and experiment-management code
- Developed and experimentally validated hypotheses for why several distillation objectives, some adapted from published work and some designed from scratch, underperformed in this setting
- Developed a multi-sample distillation approach that beat the standard flat-divergence baseline (JSD)
- Extended a verification-time speculative decoding technique to trained draft models. The original paper only evaluated it on untrained ones

The project combined research, systems engineering, and experimental design: reproducing published methods, building scalable infrastructure, and figuring out why objectives succeeded or failed.

## The problem

LLMs decode one token at a time, and each token costs a full forward pass. Speculative decoding adds a small, fast draft model that proposes several tokens ahead. A large target model then verifies them in one parallel pass and accepts the longest matching prefix. The core metric is block efficiency: accepted tokens per target call.

Raising block efficiency means raising the draft's acceptance rate, and the standard lever for that is knowledge distillation: training the draft to match the target's output distribution. The project's central question was which distillation objective actually moves block efficiency, tested on a Qwen 0.6B/8B draft-target pair and a wider Qwen 1.7B/32B pair.

![Project architecture](medium_chart_architecture.png)
*Three layers: the inference-time mechanism being optimized, the training loop that shapes the draft model, and the evaluation infrastructure that measured whether it worked.*

## Building the research infrastructure

I built a multi-GPU experiment pipeline for parallel training, checkpoint recovery after crashes, automated evaluation across every verifier by tree-width by dataset combination, and cross-checkpoint analysis at scale. It started on a memory-constrained Colab T4, needing 4-bit quantization just to fit the 8B target, then Kaggle's dual-T4 setup, then A100s for the full experiment sweeps. A separate vLLM-based pipeline generated tens of thousands of samples for pass@k benchmarking, using a paged KV-cache (each token's attention keys and values, cached so the model never recomputes them) plus continuous batching to stay fast at that scale, with resumable, checkpoint-skipping logic so a killed run never lost completed work.

A typical research iteration involved implementing a new objective in PyTorch, validating its gradients on a handful of toy steps, launching multi-GPU sweeps across all 500+ runs tracked in Weights & Biases, evaluating every checkpoint across the full verifier grid, and extending custom analysis scripts to check whether observed gains exceeded the estimated noise floor rather than being single-seed noise.

One infrastructure bug is worth calling out on its own. FlashAttention and PyTorch's native SDPA kernel compute mathematically identical attention but round differently in bf16, almost always invisibly. Here, acceptance is a hard threshold, so a last-decimal rounding difference can flip one accepted token and cascade through the rest of a generation. Two runs on two machines, identical except for attention backend, diverged by roughly 0.2 block efficiency, comparable to the effects I was trying to measure. Fixing it meant pinning one backend everywhere and logging which ran, which required reasoning about kernel-level numerics, not model architecture.

## Investigating training objectives

I designed ablation studies to isolate which distillation objectives transfer to higher block efficiency, and which only look correct on paper.

One family I tested was a verifier-aligned tree loss: a training objective built directly around the same accept/reject rule used at inference time, training the draft to match the target deep into a sequence of guesses, not just the first token. Written naturally, its gradient signal vanishes with depth. The standard fix, a log-space reformulation, solves that but shifts weight toward deep, low-probability continuations the draft rarely reaches in practice. I applied it, and results got worse, including an outright collapse in a from-scratch, wider-capacity-gap setting where block efficiency dropped by 0.4 to 0.6. A second objective, built to target the exact deployment verifier directly, the most "aligned" choice on paper, was the single worst performer of all, for the same reason: matching the target's exact acceptance criterion pushed weight onto positions the draft could not yet reach. The lesson: an objective aligned with the eval metric on paper is not automatically trainable. Gradient allocation across the sequence mattered more than how closely its form matched the final metric.

Two further hypotheses looked unlikely to work before I ran them, and I confirmed both with several variants each. A depth-weighted curriculum used a detached scalar multiplying the loss, which rescales gradient magnitude, not direction, so it could not fix a capacity bottleneck no matter how it was tuned, and it did not. A REINFORCE-based formulation rewarded block efficiency directly, but that objective pulls a model toward its own high-reward trajectories rather than the target's broader output distribution, and the metric declined in a clean, monotonic pattern consistent with that mechanism.

## Extending existing methods

Multi-rollout distillation, sampling several stochastic continuations from the target instead of a single greedy one, was the one training-side change that beat the baseline. Broader coverage of the target's actual output distribution outperformed standard JSD distillation, with a sweet spot at four rollouts, and the gain held on an out-of-distribution eval set.

I found the strongest improvement on the verification side. A recent technique delays where the draft's speculation tree branches, committing to a single stem before splitting into candidates, targeting the point where draft and target distributions actually diverge instead of branching at the first token. The original paper only evaluated this on untrained draft models. I swept it across every trained checkpoint in the project, on both the 0.6B/8B and 1.7B/32B pairs, and it improved block efficiency on nearly all of them, with larger gains at greater speculation depth and on weaker starting checkpoints. To the best of my knowledge, no prior work had evaluated this technique on trained, distilled drafts, and stacking it on the multi-rollout distillation above produced the best single block-efficiency result in the project.

I then derived the branch-point setting instead of grid-searching it. It tracks each checkpoint's own mean acceptance depth, so a stronger draft branches later on its own, a value read from the model's measured behavior rather than a search.

![Delayed-branching lift, both model pairs](medium_chart_ddte_anon.png)
*Delayed tree-branching applied to every checkpoint I trained, across two draft/target size pairs. Nearly every checkpoint improved, with larger gains on weaker starting points.*

## Building better evaluation metrics

Block efficiency measures whether the draft's output distribution matches the target's, not whether distillation makes the draft a better reasoner. To check that directly, I designed and built a pass@k evaluation pipeline: sample k completions per problem, check whether any is correct.

Across checkpoints, this revealed effects invisible to block efficiency alone. Some checkpoints improved pass@1 while regressing at pass@64: distillation sharpened single-sample accuracy but narrowed output diversity, the opposite of what many-sample decoding needs. Others improved reasoning despite lower acceptance rates than a different checkpoint, so inference efficiency and reasoning quality respond to training differently. Training closed a real but bounded fraction of the gap to the target, roughly a third to two-thirds depending on dataset and k, never all of it.

![pass@k across every trained checkpoint](medium_chart_passk_anon.png)
*Training closes part of the gap to the target model, never all of it, and by how much depends heavily on k.*

I added two further axes: forgetting (regression on previously-solved problems mid-training, a backward-transfer check) and difficulty-bucketed accuracy (easy, medium, hard, split by baseline performance). Both surfaced runs that looked healthy in aggregate but were quietly regressing on already-solved problems, or only helping the easy bucket.

Which dataset I evaluated on mattered almost as much as which checkpoint. I ran everything across three benchmarks of increasing difficulty: grade-school word problems, a harder competition-math set, and an olympiad-level set. Both block efficiency and pass@k, and how much training improved either, shifted noticeably by dataset. The easiest set was close to saturated for every model, so it barely separated good runs from bad; the harder sets showed far more spread in both baseline scores and improvement size.

None of this is meaningful without knowing how much run-to-run noise to expect. I estimated a per-k noise floor from near-duplicate configurations and compared it against the full spread observed across every setting swept.

![Noise floor vs. observed spread, per k](medium_chart_noise_floor_anon.png)
*At most k, the floor and the full sweep's spread are nearly identical, meaning most apparent differences between configurations are noise, not signal. k=64 is the exception, where the spread clearly exceeds the floor.*

## Closing

This project reinforced something I'll carry into future ML research: reliable systems come from optimization, evaluation, and infrastructure together, not any one in isolation. The most valuable output was not a single winning experiment. It was the discipline to test hypotheses rigorously, infrastructure solid enough to trust the results it produced, and evaluation broad enough to catch what a single metric would miss.
