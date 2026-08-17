# Operating systems: two jobs fit on the GPU. They still could not share it.

This is post 5. I had a handful of GPUs and a long queue of runs, so I optimized for speed: pack a job onto any card with free memory, and launch everything unattended and in parallel. Both habits came from the same instinct, use every idle cycle, and both taught me something about running work on shared hardware.

## If it fits, schedule it

I had written the scheduler to place jobs by free memory. It asked each GPU how much memory was free, through `torch.cuda.mem_get_info`, and sent the next job to the emptiest card. If two jobs fit, it ran both. For training that was obviously efficient, and it worked.

Then I pointed the same policy at evaluation.

## The same checkpoint stopped giving the same answer

I evaluated one saved checkpoint, the exact same file, more than once, and got different block efficiency scores. Not wildly different, but past my noise floor, and I could not explain them.

I ruled out the obvious causes one at a time. Sampling randomness? The seed was fixed. Different GPUs with different setups? The device and specs matched. The attention backend, which I already knew could shift results? The logs showed the same backend on every run. Same model, same seed, same inputs, and the score still moved. That is the scary case, because it means you cannot trust any number the harness produces.

## Memory capacity was not resource isolation

The clue was a column in my telemetry: CPU utilization. While I was debugging, it read high on the shared runs, which pointed me at the cores rather than the GPU. Speculative decoding does real CPU work, building and checking the tree of proposed tokens, and each process spawns its own pool of PyTorch threads. Nothing in my scheduler stopped two of them from oversubscribing the same cores.

Two honest boundaries here. First, that CPU reading was a live debugging observation, not a clean controlled measurement: I never logged a matched shared versus isolated pair for the same checkpoint, so I cannot point to a CSV that proves the spike. Second, CPU contention became my leading hypothesis for the instability, and it is easy to see how it changes wall clock time, but exactly how it changed the output of a fixed seed run, rather than just slowing it down, I did not trace to the bottom. The most likely path is that thread count changes the order of floating point reductions in the CPU math, and in bf16 a last bit difference can flip a speculative accept or reject and cascade, the same edge I hit with attention kernels earlier in this series. I could not prove that chain, so I will not claim it.

What I could reproduce was simpler. One evaluation per GPU, with each eval given its own CPU thread budget (physical cores divided by GPU count, so processes stop fighting for cores), made the scores stable and repeatable again. Shared evaluation was unstable. Isolated evaluation was not.

## The scheduler learned that training and evaluation are different workloads

So the placement policy became workload dependent. Training still packs: it wants throughput, and a little interference between runs does not change what a checkpoint learns. Evaluation runs exclusive, one process per GPU with a fixed thread budget, because its whole job is to produce a number I can trust twice.

Fitting by memory was the right rule for one of these and the wrong rule for the other.

## Fitting the work on was not the only way to lose it

Packing was one habit from moving fast. Running jobs unattended was the other, and it had its own failure mode: they die. A dropped SSH connection sends SIGHUP and kills every foreground child, so a multi hour training run vanishes with the terminal. A shared disk fills up mid run and the process crashes. Babysitting was not the fix; surviving and resuming was. I launched runs under nohup so a disconnect could not kill them, had each run write its own exact command to disk so a crashed job could relaunch from its own record rather than flags I rebuilt by hand, and made the long sampling jobs skip work they had already finished so a restart cost minutes, not hours.

## The lesson

My scheduler originally knew one thing about a job: how much GPU memory it needed. That was enough to pack training runs, not enough to protect evaluation. Two jobs fitting on one GPU did not mean they could share it without moving the measurement.

After this, "the evaluation finished" was no longer enough. A result was trustworthy only if it reproduced when I reran the same checkpoint, same seed, same configuration. Free memory told me the jobs fit. It said nothing about whether they could run independently.

Moving fast on shared hardware was never just about fitting work onto it. It was about isolating what had to be reproducible, and making everything else survive being interrupted.

---

Part of a series on building a speculative decoding for LLM inference research platform. Post 5.
