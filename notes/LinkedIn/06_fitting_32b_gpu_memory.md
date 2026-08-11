# Fitting a 32B model on one GPU

This is post 6 in my series. It is where I actually learned how GPU memory works, because I had no choice.

The project started on a 16GB Colab GPU. It ended on an 80GB A100. In between it ran on a 30GB Kaggle GPU and a 40GB A100. Every step up taught me something, because at each size a different thing was about to run out of memory.

Start with the big teacher model, the 32B one. A parameter in full precision is two bytes. 32 billion parameters times two bytes is about 64GB. That is just the weights sitting there doing nothing. It does not fit on a 40GB card at all. It barely fits on an 80GB card, and only if almost nothing else is competing for room.

But the weights are not the whole story, and this is the part I did not appreciate before. When you train, you also store optimizer state. The common optimizer keeps two extra numbers per parameter, which roughly triples the memory the trainable model needs compared to the weights alone. Then there are activations, the intermediate values you keep around to compute gradients. And in speculative decoding there is also the KV cache, which is the memory that holds the attention keys and values for the tokens generated so far. That cache grows as the sequence grows.

So the real question was never just, do the weights fit. It was, do the weights plus the optimizer state plus the activations plus the cache fit, all at once.

Because I started small, I could not just throw memory at it. I had to give things up and add them back as I got bigger cards. Early on I used LoRA, which trains only a small set of extra weights instead of the whole model, so the heavy optimizer state only covers the small part. I also quantized things, which means storing numbers in fewer bits to save room. I quantized the optimizer state. At the tightest points I even quantized the teacher model itself.

As the GPUs got bigger, I peeled these off one at a time, in order. First I could afford a full optimizer again. Then I could keep the teacher in full precision instead of a quantized version, which mattered because a quantized teacher gives slightly noisier targets. By the 80GB card I was doing full precision training with no tricks, which is the cleanest setup and the one I trusted most.

The lesson is that memory is not one number. It is a budget with several line items, and the line item that kills you changes as you scale. Weights, optimizer state, activations, cache. Knowing which one is about to overflow is most of the battle, and you only really learn it when a card says no.

---

Part of the series. This is post 6.
