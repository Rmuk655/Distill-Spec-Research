# Scientific computing: the 0.2 gain that was a rounding error

Post 4. The second time a measurement fooled me, and a subtler one.

There are two ways to compute attention in these models: FlashAttention and the default one, SDPA. They compute the same thing. I assumed the only difference was speed.

Then two runs of the same trained model landed about 0.2 apart in block efficiency. That is bigger than my noise floor, so it looked real. But nothing about the model had changed. The only difference was the attention backend.

Here is why that matters. These models run in bf16, which keeps only a few digits. The two backends do the same math in a different order, and in low precision, order changes the last digit. Normally that digit does not matter.

But speculative decoding has a sharp edge. Whether a token is accepted is a comparison. If two numbers are close, a last digit difference flips it. One backend accepts, the other rejects, and once one token flips the rest of the generation follows a different path. A rounding difference at one token becomes a visibly different result.

![Same checkpoint, only the attention backend changed](../../results/passK/linkedin_post5_backend_divergence.png)
*Each bar is one evaluation cell for the same checkpoint. The only thing that changed between the two runs of a cell was the attention kernel. That alone moved block efficiency from about minus 0.2 to plus 0.4.*

So the gap was not my model improving. It was the same model rounding differently, snowballing through the acceptance test.

The fix: pin one backend everywhere, log which one ran, and only compare like with like. When I audited old results, some had mixed backends, and I had to treat those with suspicion.

There is a second, separate issue underneath this one. For flat losses, both backends run fine, they just round differently, the 0.2 swing above. For tree losses it was not a rounding choice at all: the draft's custom tree mask needs a dict-based attention mask, which FlashAttention does not support, so SDPA had to be forced on that code path regardless of speed or numerics. One is two correct implementations disagreeing in the last decimal. The other is one implementation simply not being usable there. Worth keeping straight, mixing them up is its own way to compare the wrong things.

The lesson: know your numerics before you call something a result. Low precision rounding is usually invisible. But if your system has a hard threshold, and speculative decoding does, it can turn into a fake signal.

---

Post 4.
