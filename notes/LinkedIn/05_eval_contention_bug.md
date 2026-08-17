# Operating systems: getting 500 experiments through a handful of GPUs

In [my first post](https://medium.com/@rmukund16/engineering-an-llm-inference-research-platform-for-speculative-decoding-405d2f0bf53f), I mentioned that I ran more than 500 experiments over six weeks. I had a much smaller number of GPUs than that number suggests, and a short window to get through all of it.

The time consuming part was getting through dozens of configurations and finding out which ideas actually held up. With limited GPU time and six weeks to run them, I needed a way to keep experiments moving without managing every job by hand.

## Where it started: watching the run

At first I ran experiments the plain way: start a training job in a terminal, watch the validation number climb, and when it looked done or stuck, kill it and start the next one by hand. One depth weighted loss ablation needed a genuinely converged reference number, so I ran it this way for the full 40,000 steps, ten times the project's default of 4,000. That obviously does not scale to hundreds of runs.

## The first problem: jobs die

The obvious fix was to stop watching: kick a run off from the terminal and walk away. Closing an SSH session could send SIGHUP to foreground jobs and kill a multi-hour run with the terminal. A shared disk filling up mid run kills the process just as easily.

`nohup` solved the disconnect problem. The harder part was recovering when a run actually crashed. My `--resume` path loads the training and optimizer state from `ckpt_latest` and continues from there, but it took a few failures before I trusted it.

One problem was that failures were too quiet. `try_resume` could look for a checkpoint that did not exist, and [Weights and Biases](https://docs.wandb.ai/models) could start a new run instead of resuming the old one.

The bigger test came when a shared disk filled up during a sweep and killed several runs at once. I needed recovery to depend on saved state, not on me reconstructing the run. This was the fix: store each run's exact launch command, hyperparameters, and Weights and Biases run name in a `state.json` file. Relaunching a killed run just meant pointing `--resume` at its checkpoint directory.

## How I packed more work onto each GPU

Keeping runs alive was only half the problem. A 0.6B draft with an 8B teacher did not use all 80GB of an A100, so a lot of GPU memory was sitting unused.

I built a memory-aware scheduler, checking `torch.cuda.mem_get_info` and picking the emptiest card, for jobs that could actually share one. For the largest runs there was nothing to schedule: the 1.7B draft with the 32B teacher needed its own 80GB A100, so I pinned one run per card by hand with `CUDA_VISIBLE_DEVICES` instead. A hyperparameter sweep of four variants became four launches, one per index, all backgrounded with `nohup` at the same time. Four sweep configurations meant four GPUs running simultaneously instead of four runs waiting in sequence.

## Evaluation was the one place packing did not belong

I tried using every idle GPU for evaluation too. Rahul had already warned me against it in one of our conversations: sharing a GPU makes a throughput measurement meaningless, since another process changes the resources actually available to the one you are timing.

Speculative decoding also does real CPU work, building and checking the tree of proposed tokens, and nothing stopped two evaluation processes from competing for the same cores. They were also sharing the GPU itself. My scheduler only knew that both jobs fit in memory; it did not know whether running them together would interfere with the measurement. One evaluation per GPU, with its own CPU thread budget, removed that interference.

Evaluation used the same pinning, but wrapped in a loop instead of separate commands, since one checkpoint measured across every combination of tree width, dataset, and branch point is a much bigger grid than four configs. A single `nohup` block would loop through all of it, tens of combinations deep, on one pinned GPU, while the other GPUs ran their own training or evaluation sweeps in parallel.

## When a run was no longer worth the GPU time

I had found a way to fit more work onto each GPU. Now I needed to stop wasting time on runs that had already plateaued.

I added `--early_stop_patience`, which tracks smoothed validation block efficiency and stops a run after a fixed number of checks without improvement. Five checks turned out to be too aggressive for slower, noisier objectives, so I raised it to fifteen and added a warmup guard. I also added a divergence check so a run that was clearly collapsing could stop immediately.

The GPU time started going to runs that were still worth continuing, not simply every run that had been launched.

## The lesson

Using the GPUs efficiently meant more than keeping them busy. Packing increased aggregate experiment throughput by using memory that would otherwise sit idle, not the speed of any single job. Checkpointing and resume recovered work after failures. Early stopping reclaimed compute from runs that were going nowhere. Evaluation showed the limit: once sharing interfered with the measurement, isolation mattered more than utilization.

The goal was never maximum GPU occupancy. It was maximum useful experiment throughput without sacrificing reproducibility.

---

Part of a series on building a speculative decoding for LLM inference research platform. Post 5.
