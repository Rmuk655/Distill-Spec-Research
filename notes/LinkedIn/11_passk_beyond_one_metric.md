# When one metric was not enough

This is post 11, the finale of my series. It is about the moment I realized my main metric was hiding things, and what I did about it.

Block efficiency, my main metric, measures one thing: how well the draft matches the target's output. It does not measure whether the draft actually got better at solving problems. Those sound like the same thing. They are not.

To check the second thing directly, I built a second pipeline. For each problem, sample many answers, and check whether any of them is correct. This is called pass at k. If any of the k tries is right, the problem counts as solved. It measures whether the ability is in there at all, across many attempts.

This revealed things block efficiency could not see. Some checkpoints got better at getting the answer right on a single try, but worse when given many tries. Training had sharpened them onto one confident answer and narrowed their variety, which is exactly the wrong thing if you are going to sample many times. My main metric was happy. Pass at k showed the hidden cost. Other checkpoints improved their reasoning even though they had lower acceptance rates than a different checkpoint, which means the two metrics do not even agree on which model is better. They respond to training differently.

I added two more views on top. One tracked forgetting, whether a model got worse on problems it used to solve as training went on. The other broke accuracy down by difficulty. Together they caught runs that looked fine on average but were quietly regressing underneath, or only improving on the easy problems while doing nothing for the hard ones.

Which dataset I measured on mattered almost as much as which model I measured. On the easiest set, nearly every model was close to maxed out, so it barely separated good runs from bad. The harder sets spread the models out and actually showed differences. If I had only looked at the easy set, I would have concluded that nothing mattered, when really the easy set just could not see the differences.

The chart below shows the pass at k picture across every trained checkpoint. Training closed part of the gap to the big teacher model, somewhere between a third and two thirds of it, but never all of it. There is a real ceiling to how much a small draft can absorb from a much larger teacher.

![pass at k across every trained checkpoint, gap to the teacher](../medium_chart_passk_anon.png)
*Training closes part of the gap to the teacher, never all of it.*

And the honest closing note for the whole series. After accounting for run to run noise, most of the settings I swept did not produce a signal I could trust. That is not a failure. It is the result. A lot of careful measurement, a few real effects, and a clear map of what does not work and why. I would rather report that than a confident number that falls apart on a second look.

Thank you for reading this far. If you went through the whole series, you saw the real shape of a research project: mostly plumbing, careful measurement, a lot of dead ends, and a few things that held up. That is what doing this work actually looks like.

---

This is the final post in the series on building a speculative decoding research platform. Start from post 1 if you want the full picture.
