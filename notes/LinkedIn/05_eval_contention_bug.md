# Operating systems: the checkpoint that scored differently every run

This is post 5 in my series. It is about a bug that taught me not to trust my own measurements until I understood them.

I was trying to move fast, so I packed work onto every GPU I had. If a GPU had room, I put another job on it. This worked fine for training. Then I did the same thing for evaluation, and something strange happened.

I evaluated the same saved model, the exact same file, more than once. I got different scores. Not wildly different, but different enough to matter, and different in a way I could not explain.

My first guess was randomness in sampling. But I had fixed the random seed, so that was not it. My second guess was that the different runs were on different GPUs with different setups. But no, the setups matched. My third guess was the attention backend, which I already knew could cause differences. But the logs showed the same backend on every run.

So the model was identical, the seed was identical, the inputs were identical, and the score still moved. That is a scary place to be, because it means you cannot trust any number you produce.

The thing that finally pointed me at the answer was a column in my logging. I record CPU usage during each run. When two evaluations shared the same GPU, that CPU usage spiked. The two processes were fighting over the same underlying CPU cores. Speculative decoding does a fair amount of work on the CPU, building and checking the tree of proposed tokens. When two of those ran on one GPU, they interfered with each other in a way that leaked into the results, not just the speed.

I want to be honest about the depth of what I know here. What I proved is that isolating evaluation to one process per GPU made the numbers stable and repeatable again. Exactly why the contention changed the output of a fixed seed run, rather than just slowing it down, I have a reasonable theory about but did not trace all the way to the bottom. I fixed it and moved on, because a stable measurement was what I needed.

The fix was simple once I understood it. One evaluation per GPU. No sharing. Training could still pack, but evaluation could not.

The lesson stuck with me more than most. A measurement bug looks exactly like a discovery. Both show up as a number that changed. If I had not gone looking, I could easily have reported one of those wobbles as a real effect of some training method, when it was really just two processes stepping on each other. Before I trust a result, I now trust the thing that produced it.

---

Part of the series. This is post 5.
