# Building a multi GPU research platform for speeding up LLM inference

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

Raising it means raising the draft's acceptance rate, and the standard lever is knowledge distillation ([DistillSpec](https://arxiv.org/abs/2310.08461), Rahul's earlier paper on the idea): training the draft to match the target's output distribution. The central question: which distillation objective actually moves block efficiency, tested on a student-teacher pair (Qwen 0.6B/8B) and a wider one (Qwen 1.7B/32B), though the setup is easily extensible to other model families and pairings.

<img src="medium_chart_architecture.png" width="650" alt="Project architecture" />
*The inference mechanism, the training loop, and the evaluation infrastructure built around both.*

## Building the research infrastructure

I built the infrastructure needed to run, recover, evaluate, and analyze hundreds of training experiments in parallel across heterogeneous GPU hardware: an experiment orchestration layer with crash recovery, an evaluation harness covering every verifier ([Traversal Verification](https://arxiv.org/abs/2505.12398), [SpecTr](https://arxiv.org/abs/2310.15141), [SpecInfer](https://arxiv.org/abs/2305.09781), and others, training-free tree-based algorithms for a fixed draft model) by tree-width by dataset combination, and analysis tooling that pulled every run's metric history straight from the Weights & Biases API to rank configs, group by seed for robustness, and flag runs that had overfit, stalled, or diverged, moving from a memory-constrained Colab T4 (4-bit quantization just to fit the 8B target) to Kaggle's dual-T4s to A100s for the full sweeps. A separate vLLM pipeline, with its own async backend and isolated environment, generated tens of thousands of samples for pass@k benchmarking, using a paged KV-cache (cached attention keys/values so tokens are never recomputed) and continuous batching, with resumable, checkpoint-skipping logic so a killed run never lost completed work.

The 32B teacher was the harder memory constraint. In bf16 it needs roughly 64GB just for weights, so once the student, its optimizer state, and activations for multiple stochastic rollouts are added, there is little headroom left even on an 80GB card, and it does not fit at all on a 40GB one. Early on, before we had 80GB hardware, I worked around that two ways: LoRA to shrink the trainable parameter count, and quantizing the draft's own optimizer state (8-bit AdamW); both became unnecessary once 80GB A100s were available and training moved to a full fine-tune. The teacher itself still trained against a 4-bit (NF4) quantized copy throughout, since even at 80GB the full bf16 teacher alongside everything else didn't reliably fit; evaluation reloaded the full-precision bf16 teacher separately, so the reported numbers reflect the deployment-accurate target, not the quantized approximation used during training.

A typical iteration: implement a new objective in PyTorch, validate its gradients on a handful of toy steps, launch multi-GPU sweeps tracked in Weights & Biases, evaluate every checkpoint across the full verifier grid, and extend analysis scripts to check whether observed gains exceeded the estimated noise floor.

A few things kept that loop from wasting hardware. Evaluation is CPU-bound rather than GPU-bound, so I ran it one process per GPU with threads capped per process to avoid CPU contention across parallel evals, keeping each eval a clean room: without that isolation, the same checkpoint could measure differently run to run, contention alone would move the numbers, not just wall time. Training was the opposite: the smaller pair (0.6B student, 8B teacher) fit comfortably enough to pack multiple concurrent runs on one 80GB A100, while the much larger 1.7B/32B pair got one run per GPU. Every sweep also had early stopping built in, a smoothed validation block-efficiency metric tracked against a patience counter, so a plateaued or diverging run freed its GPU instead of running to a fixed step count. And both training and evaluation cached progress to disk, so a killed sweep resumed only the unfinished work instead of redoing it.

One systems lesson surprised me: profiling the harness showed GPU utilization well under 100% during generation, meaning time was going into CPU-GPU handoff and per-step Python overhead, not into the model's actual matrix multiplications. A production setup would engineer that overhead away; I did not attempt that here, since it wasn't the point of the project. So instead of tokens/sec, which would have measured my harness's inefficiency as much as the algorithm, we reported block efficiency, Rahul's choice of metric: a hardware-agnostic ratio that isolates how good the algorithm itself is, independent of how well any particular implementation is engineered.

FlashAttention and PyTorch's native SDPA kernel compute mathematically identical attention but round differently in bf16, almost always invisibly. Here, acceptance is a hard threshold, so a last-decimal rounding difference can flip one accepted token and cascade through the rest of a generation. Two runs on two machines, identical except for attention backend, diverged by roughly 0.2 block efficiency, comparable to the effects I was measuring. The fix was pinning one backend everywhere and logging which ran, which required reasoning about kernel-level numerics, not model architecture.

## Investigating training objectives

We designed ablation studies to isolate which distillation objectives transfer to higher block efficiency versus which only look correct on paper.

Rahul provided the verifier implementations and the papers behind them; on top of that, as part of an initial learning phase to explore by building, I designed two new objectives myself. One trains the draft to match the target deep into a sequence of guesses, not just the first token. Its gradient naturally vanishes with depth, so the standard fix moves it into log-space. I applied that fix and results got worse, including an outright collapse at a wider capacity gap, from scratch, where block efficiency dropped by 0.4 to 0.6: the reformulation solves vanishing gradients but overcorrects, shifting weight toward deep, low-probability continuations the draft rarely reaches. The second targeted the exact deployment verifier directly, the most "aligned" choice on paper; it turned out to be a dud, the single worst performer of anything tested, for the same reason. That is the lesson I took from it: an objective aligned with the eval metric on paper is not automatically trainable. Gradient allocation mattered more than how closely its form matched the final metric.

We also tested a depth-weighted curriculum Rahul thought could work, and I separately explored a REINFORCE-based formulation on my own; why neither held up only became clear empirically, not from a rigorous argument in advance. The curriculum multiplied the loss by a detached scalar, which rescales gradient magnitude, not direction, so it could not fix a capacity bottleneck, and it did not. REINFORCE rewarded block efficiency directly, but that objective pulls a model toward its own high-reward trajectories rather than the target's broader distribution, and the metric declined in a clean, monotonic pattern consistent with that mechanism.

## Extending existing methods

Multi-rollout distillation, sampling several stochastic continuations from the target instead of one greedy one, was the one training-side change that beat the baseline, with a sweet spot at four rollouts. Trained only on MATH, the gain held on OlympiadBench, never seen in training.

The strongest improvement came from the verification side. [Rahul's own published research](https://arxiv.org/abs/2602.16994) introduced a technique that delays where the draft's speculation tree branches, committing to a single stem before splitting into candidates at the point where draft and target distributions actually diverge, instead of branching at the first token. That paper only evaluated it on untrained draft models. Swept across every trained checkpoint, both model pairs, it improved block efficiency on nearly all of them, by up to roughly 0.8 additional accepted tokens per call (about 14% on the larger pair). We could not find a prior evaluation of this on trained, distilled drafts, and stacking it on the multi-rollout distillation above gave the best single result in the project.

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
