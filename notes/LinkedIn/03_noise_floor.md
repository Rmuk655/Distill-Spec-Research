# Freezing every other variable was not enough

Post 2 ended on a partial derivative: hold hardware and the serving stack still, and the only thing left free to move is the one variable you actually care about. That sounds like it should settle things. It does not, and the reason is worth sitting with.

Speculative decoding's accept or reject step is a probability draw, not a lookup. Even with the exact same checkpoint, the exact same prompt, and every other variable frozen solid, the number you measure still has width to it, because the sampling itself is random. Freezing everything else does not remove the noise. It just makes sure the noise left over actually belongs to the one thing you meant to test.

So the real question was never just "did the number move." It was "did it move by more than the width of the noise it was always going to have anyway." That took longer to earn than I expected, and it escalated through stages I did not see coming at the start.

## One hundred prompts was not enough

The first stage was running a config a few times and watching how much the number wobbled on its own, with nothing else changed. At one hundred evaluation prompts, that wobble in block efficiency sits around 0.10 to 0.15, purely from run to run variation. Any claimed improvement smaller than that is not a result yet. I started graying those out in my own analysis instead of reporting them.

## One seed was not enough

I learned this one the hard way. One training run, a combination of plain distribution matching, a depth weighted curriculum, and a tree shaped loss, had a headline number, a block efficiency gain of about 0.293 over the baseline, well above that 0.10 to 0.15 floor, which made it look real. When I went back to write it up properly and pulled the raw rows, the comparison was wrong. The 0.293 came from reading a result at a tree width of one and comparing it against the baseline's number at a tree width of three, two different settings, not the same one. Lined up correctly, at the same tree width on both sides, the real difference was 0.03. Inside the noise. Not a result at all.

That mistake is why I stopped trusting any number from a single run, however good it looked. A single seed can hand you something that looks like signal purely because it is one draw from a noisy process, and you will not know it until you check.

## One dataset was not enough

A method that looked solid on one set of math problems did not always carry over to a harder set. Training and evaluating on the same distribution can flatter a result in a way that quietly stops holding once the questions get harder, or just different. A real claim needed to survive more than the one dataset it was tuned against.

## One model pair was not enough

This is the one that stretched the point furthest. A training method that beat the baseline cleanly on a smaller draft and teacher pair did not cleanly repeat on a bigger one. The gains flipped sign depending on how far ahead the draft was allowed to guess, instead of the same clear pattern the smaller pair had shown. The part that made me careful rather than just disappointed: I never even measured the noise floor for that bigger pair. Every number I had for it came from a single run. So I could not call it a real failure to replicate, only that I did not yet have enough runs to say anything with confidence either way.

## Eventually, even one loss family was not enough

The last stage was realizing that a pattern holding up within one family of objectives is not the same as it holding up in general. A result can look consistent purely because you are still inside the variance of the one thing you tested, not because you found something true more broadly. After enough of this, one honest study across a whole set of comparisons came back with no result clearing the floor almost anywhere. That is not a result to be embarrassed about. It is what happens when you actually go looking with the right level of rigor instead of stopping at the first number that looks good.

## What this actually teaches

Every one of these escalations traces back to the same cause. Speculative decoding's randomness, in generation and in the accept or reject draw itself, does not go away just because every other variable was held perfectly still. It survives the partial derivative. Statistics, in this context, is not a formality bolted on afterward. It is the discipline of knowing how much of what you are looking at is signal, and how much is the noise a perfectly controlled experiment still has left over.

---

Part of a series on building a speculative decoding research platform.
