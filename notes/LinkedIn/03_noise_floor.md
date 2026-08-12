# Statistics: freezing every other variable was not enough

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

## Even a full sweep does not always clear the floor

Zooming out further, one complete study, comparing many training configurations against each other at once, after accounting for all of this run to run noise, came back with no result clearing the floor almost anywhere. That is not a result to be embarrassed about. It is what a real search looks like once you stop stopping at the first number that looks good.

## And every bit of this was still one model family

Here is the limitation I have not closed. Every escalation above, more prompts, more seeds, more datasets, more model pairs, still only ever touched one family of models, Qwen. A claim solid enough for a paper needs the same effect checked against other families entirely, Gemma, GPT, Llama, not just more seeds and datasets inside the one family already tested. That escalation is the one I have not done, and it is worth saying plainly rather than pretending the story is more finished than it is.

## What this actually teaches

Every escalation up through the model pairs traces back to the same cause. Speculative decoding's randomness, in generation and in the accept or reject draw itself, does not go away just because every other variable was held perfectly still. It survives the partial derivative. The model family gap is a different kind of limitation, not noise but reach, how far a result is actually allowed to generalize, and it is the honest reason this is a well tested finding on one family, not yet a settled one. Statistics, in this context, is not a formality bolted on afterward. It is the discipline of knowing how much of what you see is signal, how much is leftover noise, and how far what you found is actually allowed to travel.

---

Part of a series on building a speculative decoding research platform.
