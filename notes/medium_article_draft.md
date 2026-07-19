# Can you distill a faster draft model without breaking it?

During my second-year summer, I worked on LLM inference optimization through speculative decoding, under the guidance of Rahul [@Rahul], an ML researcher at Columbia University.

Over eight weeks I:

- Ran 500+ tracked experiments (about 250 training runs), logged in Weights & Biases
- Evaluated 71 trained checkpoints across 4,000+ measurements
- Built and operated a multi-GPU speculative-decoding research platform, training, evaluation, and analysis, in PyTorch, HuggingFace, and vLLM, growing to roughly 10,000 lines of code
- Set out to beat a strong, well-proven baseline with several new distillation objectives, some from published work and some designed from scratch; most struggled to beat it, but a few did, and validated why the rest fell short
- Developed a multi-sample distillation approach that beat the standard flat-divergence baseline (JSD)
- Extended a verification-time decoding technique to trained draft models, a case the original paper never evaluated

## The problem

LLMs decode one token at a time, and each token costs a full forward pass. Speculative decoding adds a small, fast draft model that proposes several tokens ahead; a large target model verifies them in one parallel pass and accepts the longest matching prefix. The core metric is block efficiency: accepted tokens per target call. Since the target dominates inference cost, block efficiency is a direct, hardware-agnostic multiplier on decode throughput, the number worth optimizing rather than raw tokens-per-second from any one harness.

Raising it means raising the draft's acceptance rate, and the standard lever is knowledge distillation: training the draft to match the target's output distribution. The central question: which distillation objective actually moves block efficiency, tested on a student-teacher pair (Qwen 0.6B/8B) and a wider one (Qwen 1.7B/32B), though the setup is easily extensible to other model families and pairings.

<img src="medium_chart_architecture.png" width="650" alt="Project architecture" />
*The inference mechanism, the training loop, and the evaluation infrastructure built around both.*

## Building the research infrastructure

I built the infrastructure needed to run, recover, evaluate, and analyze hundreds of training experiments in parallel across heterogeneous GPU hardware: a scheduler and checkpoint manager with crash recovery, an evaluation harness covering every verifier by tree-width by dataset combination, and cross-checkpoint analysis tooling on top, moving from a memory-constrained Colab T4 (4-bit quantization just to fit the 8B target) to Kaggle's dual-T4s to A100s for the full sweeps. A separate vLLM pipeline, with its own async backend and isolated environment, generated tens of thousands of samples for pass@k benchmarking, using a paged KV-cache (cached attention keys/values so tokens are never recomputed), continuous batching, and tensor-parallel sharding available for the larger 32B teacher, with resumable, checkpoint-skipping logic so a killed run never lost completed work.

The 32B teacher was the harder memory constraint, not the 80GB A100s themselves. In bf16 it needs roughly 64GB just for weights, so once the student, its optimizer state, and activations for multiple stochastic rollouts are added, there is little headroom left, and it does not fit at all on a 40GB card. Quantizing it to 4-bit (NF4, via bitsandbytes) made it fit, but that had a real cost: NF4 pays a dequantization penalty on every forward pass, so objectives that call the teacher's own token-by-token generation, rather than scoring one pre-built sequence, got roughly 15-20x slower per step. Where memory allowed, the better fix was quantizing the draft's own optimizer state instead (8-bit AdamW), which freed enough headroom to keep the teacher in full bf16 and avoid that slowdown entirely.

A typical iteration: implement a new objective in PyTorch, validate its gradients on a handful of toy steps, launch multi-GPU sweeps tracked in Weights & Biases, evaluate every checkpoint across the full verifier grid, and extend analysis scripts to check whether observed gains exceeded the estimated noise floor.

A few things kept that loop from wasting hardware. Evaluation is CPU-bound (building and walking verifier trees), not GPU-bound, so I ran it one process per GPU as a proxy for one process per CPU core group, with threads per process capped to physical cores divided by GPU count; left uncapped, parallel evals thrash the same CPU cache and skew the timing measurements, not just correctness. Training was the opposite: for the smaller pair (0.6B student, 8B teacher) two runs comfortably shared one 80GB A100, since neither came close to using the full 80GB alone, so I packed multiple concurrent runs per GPU there while giving the 1.7B/32B pair, with its much larger memory footprint, one run per GPU. Every sweep also had early stopping built in: a smoothed validation block-efficiency EMA tracked against a patience counter, so a run plateaued or diverging stopped consuming a GPU instead of running to a fixed step count regardless of whether it was still improving. And both training and evaluation cached progress to disk, checkpoint-resumability for training, completed (verifier, tree-width, checkpoint, dataset) cells for eval, so a killed sweep resumed only the unfinished work instead of redoing it.

One systems lesson surprised me: profiling the harness showed GPU utilization well under 100% during generation, meaning time was going into CPU-GPU handoff and per-step Python overhead, not into the model's actual matrix multiplications. A production setup would engineer that overhead away; I did not attempt that here, since it wasn't the point of the project. So instead of tokens/sec, which would have measured my harness's inefficiency as much as the algorithm, I reported block efficiency: a hardware-agnostic ratio that isolates how good the algorithm itself is, independent of how well any particular implementation is engineered.

FlashAttention and PyTorch's native SDPA kernel compute mathematically identical attention but round differently in bf16, almost always invisibly. Here, acceptance is a hard threshold, so a last-decimal rounding difference can flip one accepted token and cascade through the rest of a generation. Two runs on two machines, identical except for attention backend, diverged by roughly 0.2 block efficiency, comparable to the effects I was measuring. The fix was pinning one backend everywhere and logging which ran, which required reasoning about kernel-level numerics, not model architecture.

## Investigating training objectives

We designed ablation studies to isolate which distillation objectives transfer to higher block efficiency versus which only look correct on paper.

One family, an idea of mine built on Rahul's existing verifier implementation as the starting point, trains the draft to match the target deep into a sequence of guesses, not just the first token. Its gradient naturally vanishes with depth, so the standard fix moves it into log-space. I applied that fix and results got worse, including an outright collapse at a wider capacity gap, from scratch, where block efficiency dropped by 0.4 to 0.6: the reformulation solves vanishing gradients but overcorrects, shifting weight toward deep, low-probability continuations the draft rarely reaches. A second objective, also my own idea, targeted the exact deployment verifier directly, the most "aligned" choice on paper; it turned out to be a dud, the single worst performer of anything tested, for the same reason. That is the lesson I took from it: an objective aligned with the eval metric on paper is not automatically trainable. Gradient allocation mattered more than how closely its form matched the final metric.

We also tested a depth-weighted curriculum Rahul thought could work, and I separately explored a REINFORCE-based formulation on my own; why neither held up only became clear empirically, not from a rigorous argument in advance. The curriculum multiplied the loss by a detached scalar, which rescales gradient magnitude, not direction, so it could not fix a capacity bottleneck, and it did not. REINFORCE rewarded block efficiency directly, but that objective pulls a model toward its own high-reward trajectories rather than the target's broader distribution, and the metric declined in a clean, monotonic pattern consistent with that mechanism.

## Extending existing methods

Multi-rollout distillation, sampling several stochastic continuations from the target instead of one greedy one, was the one training-side change that beat the baseline, with a sweet spot at four rollouts. Trained only on MATH, the gain held on OlympiadBench, never seen in training.

The strongest improvement came from the verification side. A recent technique delays where the draft's speculation tree branches, committing to a single stem before splitting into candidates at the point where draft and target distributions actually diverge, instead of branching at the first token. The original paper only evaluated this on untrained draft models. Swept across every trained checkpoint, both model pairs, it improved block efficiency on nearly all of them, by up to roughly 0.8 additional accepted tokens per call (about 14% on the larger pair). We could not find a prior evaluation of this on trained, distilled drafts, and stacking it on the multi-rollout distillation above gave the best single result in the project.

I then derived the branch-point setting instead of grid-searching it, tracking each checkpoint's own mean acceptance depth so stronger drafts branch later.

<img src="medium_chart_ddte_anon.png" width="650" alt="Delayed-branching lift, both model pairs" />
*Delayed tree-branching, every checkpoint, both model pairs: nearly all improved, more so from weaker starting points.*

## Building better evaluation metrics

Block efficiency measures whether the draft matches the target's output distribution, not whether distillation makes it a better reasoner. To check that directly, I built a pass@k pipeline per Rahul's direction: sample k completions per problem, check whether any is correct.

This revealed effects invisible to block efficiency alone. Some checkpoints improved pass@1 while regressing at pass@64: distillation sharpened single-sample accuracy but narrowed output diversity, the opposite of what many-sample decoding needs. Others improved reasoning despite lower acceptance rates than a different checkpoint, so the two metrics respond to training differently. Training closed a real but bounded fraction of the gap to the target, roughly a third to two-thirds, never all of it.

<img src="medium_chart_passk_anon.png" width="650" alt="pass@k across every trained checkpoint" />
*Training closes part of the gap to the target, never all of it.*

I added two further axes, forgetting (regression on previously-solved problems mid-training) and difficulty-bucketed accuracy, which surfaced runs that looked healthy in aggregate but were quietly regressing on already-solved problems, or only helping the easy bucket.

Which dataset I evaluated on mattered almost as much as which checkpoint. Across GSM8K, Hendrycks MATH, and OlympiadBench, in increasing difficulty, GSM8K was close to saturated for every model, so it barely separated good runs from bad; the harder sets showed far more spread in baseline scores and improvement.

None of this is meaningful without knowing how much run-to-run noise to expect. I estimated a per-k noise floor from near-duplicate configurations and compared it against the full spread observed across every setting swept: at most k, the floor matched the full sweep's spread, meaning most differences between configurations were noise, not signal.

## Closing

This project reinforced something I will carry forward: reliable systems come from optimization, evaluation, and infrastructure together, not any one in isolation. The most valuable output was not a single winning experiment, but the discipline to test hypotheses rigorously, build infrastructure solid enough to trust, and evaluate broadly enough to catch what one metric would miss.
