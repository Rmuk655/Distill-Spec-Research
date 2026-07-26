# Interview Prep — Speculative Decoding / Draft Distillation Project

Reorganized around ML Systems / Research Engineering interview themes: architecture first, then
GPU/memory, inference serving, profiling, numerics, PyTorch/HF engineering, and systems-design
hypotheticals. Pure research-methodology content (loss-function math, tree-loss mechanics, verifier
theory, negative-result reflection) is kept in its own section afterward, clearly labeled, since a
research-engineering interviewer may still go there — but it's no longer interleaved with the
systems content.

Every technical answer is grounded in actual code/results (file:line cited), not reconstructed from
general knowledge. Questions marked **[PERSONALIZE]** are things only you can answer honestly — I've
given you the real candidate events from the project, but pick the one that's actually true for you
and speak from your own memory, not a script.

Two corrections to watch for, since interviewers may assume otherwise from the article:
- **Not "430 experiments."** Real numbers: 500+ tracked runs (~250 training runs), 71 checkpoints
  evaluated across 4,000+ measurements.
- **No "Total Variation" loss by that name.** `l1` is the implemented equivalent — its own docstring
  says "Equivalent (up to a constant) to 2·TV(p_s, p_t)" (`losses/flat.py:80-90`).

---

## 0. Calibration: What Depth Is Actually Expected

A useful outside read on this (paraphrased from a BizChat/Copilot conversation reviewing the
article as a 3rd-year-undergrad, 6-week project): the failure mode on both sides of an interview is
evaluating a student project against a staff-engineer bar. The right question is not "can you derive
vLLM from first principles," it is "do you understand these topics deeply enough relative to what
you actually claimed you did." Rough expected-depth calibration for this project specifically:

| Topic | Expected depth |
|---|---|
| Speculative decoding (acceptance, block efficiency, why tokens/sec misleads) | Deep |
| Experimental methodology (noise floor, ablations, seeds) | Deep |
| PyTorch training stack (autograd, loss implementation, gradient flow, mixed precision) | Moderate-Deep |
| Statistical rigor (what "significant" means here, why single-seed is a real gap) | Moderate-Deep |
| GPU memory accounting (weights/optimizer-state/activations/KV-cache, at a high level) | Moderate |
| vLLM (continuous batching, KV-cache reuse, why it improves throughput) | Moderate |
| Distributed training concepts (DDP vs. tensor/model parallel, even though none was used here) | Basic-Moderate |
| CUDA internals, ATen internals, scheduler internals, kernel fusion details | Not required |

Concretely for PyTorch: since new distillation objectives were implemented, expect to be asked about
autograd (why min(1,q/p) is differentiable except at a measure-zero kink, §2/§R2), how a loss is
actually implemented (the JSD/l1/tree-loss registries, §R1), gradient flow through a block (why
REINFORCE's gradient direction was wrong, §R2), and mixed precision basics (bf16 throughout, §4) —
not scheduler/ATen/kernel-fusion internals, and not DDP mechanics beyond "why wasn't it needed"
(§3), since none was used.

**The actual discriminator interviewers use, more than any of the above**: asking about mistakes, not
successes. A candidate with genuine ownership gets *more* detailed and specific when describing what
went wrong; someone with shallow or AI-assisted understanding tends to get vaguer. Prepare this one
for real, with specifics only you can supply:

**Q: Tell me about the three biggest mistakes you made during the project.**

Three real, well-documented candidates from this project's actual history — pick the ones that are
genuinely yours to tell, and add the specific detail (how you found it, how long it took, what you
were thinking at the time) that only you know:

1. **A mislabeled result that overstated a win.** The combined depth-weight + naive-tree loss was
   originally reported as +0.293 over JSD at K=3 — but that number was actually a K_eval=1 result
   misread against the K=3 baseline. The true matched-K delta was +0.03, within noise. Caught during
   a later audit of the research notes against raw results, not at the time it was first reported.
2. **Letting the attention backend silently vary across machines before pinning it.** Transformers
   auto-selects `sdpa` vs. `flash_attention_2` depending on what happens to be installed on a given
   box. Since speculative decoding's acceptance test is a hard threshold, this caused up to ~0.2
   block-efficiency divergence between otherwise-identical runs — a real risk of misreading noise as
   signal in any comparison that mixed backends before `--force_attn` was pinned to default to
   `sdpa` everywhere.
3. **Running parallel evals on shared GPUs before realizing it corrupted measurements, not just
   slowed them down.** CPU contention between eval processes on the same GPU made the same
   checkpoint measure differently run to run — found only after noticing inconsistent numbers, not
   anticipated in advance.

**[PERSONALIZE: if you have a fourth or better example that's more personally yours — e.g. a
specific config bug, a sweep you had to redo, a moment you initially misread a chart — use that
instead of or alongside these. The goal is genuine specificity, not reciting this list.]**

---

## 0.6 Thinking Questions (Problem Formulation, Rigor, Debugging, Judgment, AI Usage)

A second outside review (also paraphrased from a BizChat/Copilot conversation) argued the sharpest
interview questions test *thinking*, not knowledge, and deliberately avoid anything with a cached
LLM answer. These require genuine synthesis from the actual project — drafted below from real
project history and, for the AI-usage questions, from real moments in *this session* where I made a
mistake you caught. Read these as scaffolding, then answer in your own words with your own memory of
the moment.

### Problem Formulation

**Q: You had six weeks. Why spend it on distillation objectives instead of verifier algorithms, architecture changes, quantization, or serving optimizations?**

Leverage, not preference. The verifiers (traversal, SpecTr, SpecInfer, GBV) were already built and
tested by Rahul — reimplementing them has near-zero information value. Model-architecture changes and
serving optimizations are engineering work with a lower novelty ceiling for a research internship,
and don't touch the actual open question. The scope document's central hypothesis was specifically
"which distillation objective raises block efficiency" — that's the one axis where six weeks of
exploration could produce genuinely new evidence, because it hadn't been systematically tested
against this verifier suite before. It's also the axis with the fastest, cheapest experiment loop
(one GPU, one training run, one eval sweep) — highest expected information gain per unit time, which
is the real answer, not "my mentor suggested it."

**Q: Given another six weeks and 8 A100s, what's the first experiment you'd run?**

A second (and third) seed on every 1.7B/32B result — not a new hypothesis. Every conclusion at that
pair is currently single-seed, and several deltas "sign-flip across K" in a way consistent with pure
noise against the measured floor (SE ≈ 0.10-0.15). Nothing else generated at that pair is trustworthy
until this is resolved, so it dominates every other candidate experiment on expected value: it either
confirms or invalidates the entire pair's conclusions at once, rather than adding one more untrusted
data point next to the others.

**Q: What result changed your mind most?**

The prefix-overlap `traversal` objective. Expected it to be a strong performer — it directly targets
the deployment verifier's exact quantity, the most "aligned on paper" choice available. Data showed
it was the single worst performer of everything tested, at both model pairs. That revised my model of
what makes a loss trainable: gradient allocation (where the signal actually lands, token-by-token and
depth-by-depth) matters more than how closely a loss's mathematical form resembles the eval metric —
I expected metric-alignment to predict trainability, and it did not.

### Experimental Rigor

**Q: What's the strongest result, and why do you believe it isn't noise?**

Multi-rollout enrichment at 0.6B/8B: mean traversal Δ +0.19 (math) / +0.12 (olympiad) across K=2-4,
against a measured noise floor of SE ≈ 0.10-0.15 from near-duplicate configurations. The olympiad
(out-of-distribution) result is matched-backend verified (sdpa-to-sdpa throughout); the math result
is cross-backend but robust as a multi-cell mean since the sdpa-FA2 gap nets close to zero across many
cells. Honest caveat: this is still single-pair-confirmed — it does not cleanly replicate at
1.7B/32B, so "isn't noise" is qualified to the 0.6B/8B pair specifically, not a universal claim.

**Q: The FlashAttention/SDPA gap was ~0.2 block efficiency. How did you rule out random initialization?**

There's a false premise worth naming directly: no re-initialization or retraining happens between the
two runs being compared. It's the exact same trained checkpoint, same weights, same seed, same
prompts, evaluated twice — the only thing that changes is which attention kernel runs underneath.
Since nothing about the model or data differs, the divergence is attributable to the backend by
construction, not to training-time variance; this is an eval-time numerics question, not a training
reproducibility question.

**Q: If I deleted all your graphs and gave you only the raw checkpoints, how would you prove multi-rollout distillation works?**

Re-run `eval.py` fresh on both checkpoints (enrich vs. flat JSD baseline) with the exact `--seed`,
`--n`, and a pinned `--force_attn` so backend can't confound the comparison; compare the resulting
delta against the independently-measured noise floor from near-duplicate configs; check the sign
holds across the K sweep (K=2,3,4), not one cherry-picked K; and check it holds on a dataset never
seen in training (OlympiadBench), since generalization to an unseen distribution is stronger evidence
than an in-distribution number alone.

### Systems Debugging

**Q: Tell me about the hardest bug in the project. Walk me through the investigation.**

The CPU-contention eval bug is the strongest real candidate: the same checkpoint measured differently
across parallel evals, run to run. Be ready to narrate the actual sequence for yourself — what the
first symptom looked like, what you ruled out first (fixed seed, so not sampling nondeterminism;
distinct `--device` flags, so not a GPU-assignment collision; `attn_backend` column matched across
the runs, so not the FlashAttention/SDPA issue), and what pointed you at CPU contention specifically
(the `telemetry.py` `cpu_util_avg_pct` column spiking when evals shared a GPU's underlying CPU cores).
**Be honest about the depth of the root cause you actually have**: if what you have is "isolating eval
to one process per GPU resolved it" rather than a fully traced mechanism for *why* CPU contention
changed a fixed-seed run's actual output rather than just its wall-clock time, say that plainly rather
than inventing precision you don't have — a plausible hypothesis (CPU-bound verifier-tree
construction racing with generation in a way that leaked into batching/timing-sensitive logic) is a
fine thing to offer as a hypothesis, clearly labeled as one, not as a proven mechanism.

**Q: If I gave you the same issue tomorrow, how would you diagnose it faster?**

Observability, not investigation: the fix already logs `cpu_util_avg_pct` per run via the pynvml/
psutil telemetry thread, so a contention-caused discrepancy would now show a suspicious correlation
immediately instead of requiring after-the-fact detective work. And one-eval-per-GPU is now the
default rule from having hit this once — the second occurrence would be prevented, not just diagnosed
faster.

### Judgment Under Uncertainty

**Q: An AI tool says your REINFORCE implementation is wrong. Your experiments say it's correct but performs poorly. Which do you trust?**

Neither, outright — design a discriminating experiment instead. The actual resolution in this project
was exactly that: the REINFORCE loss's failure matched a specific, theoretically-predicted signature
(monotonic block-efficiency decline, 4.81 → 4.0, consistent with mode-seeking score-function ascent
when the objective needed mode-covering) rather than a random or unstable failure pattern. A
mechanically-correct-but-structurally-misaligned estimator produces a *predictable* failure shape; a
buggy implementation usually does not. That's the discriminating evidence, not a coin flip between
"trust the tool" and "trust the experiment."

**Q: You find a 5% gain from a new objective. Do you publish it?**

Depends on variance, reproducibility, and cost of being wrong — not on the number alone. This
project's own history is the cautionary tale: a previously-cited +0.293 win turned out to be a
mismatched-K comparison, true delta +0.03, inside the noise floor. Before trusting a 5% number: check
it against the measured noise floor at that scale, check a second seed, check it holds across K and
across an out-of-distribution dataset. A single-seed, single-K, single-dataset 5% is not evidence yet.

**Q: The best-performing method violates your intuition. Do you deploy it?**

Separate "does it hold up under measurement" from "do I understand why." Enrichment (multi-rollout)
is a good anchor: it beat flat JSD somewhat unexpectedly, but the mechanism (fresh stochastic teacher
rollouts vs. a frozen teacher-context sequence) is understood well enough to reason about its failure
modes, so deploying it with monitoring is reasonable. The DDTE branch-point setting is a better
example of the opposite case: the "stronger drafts can delay branching to lower depths" pattern is
observed, not derived from a mechanism — I'd want more understanding of *why* before hard-committing
a specific setting into production without knowing where it breaks.

### AI Usage

**Q: What was the most useful thing AI helped with?**

Fast, accurate cross-referencing against a large codebase and many research notes — checking exact
numbers, catching inconsistencies between a note's claim and the raw CSV, drafting boilerplate sweep
scripts — and iterating on written analysis (like this article) quickly once the underlying facts
were established.

**Q: What's the biggest mistake AI made during the project?**

Real, specific, and checkable from this very session, not a hypothetical:
1. An early draft of this article implied tensor-parallel sharding was actively used for the 32B
   teacher. Checked against the actual launcher scripts (`scripts/passk_eval_sweep_checkpoints.sh`,
   `scripts/run_passk_baselines.sh`) and found neither ever set `--tensor_parallel_size` above its
   default of 1 — the capability existed and was documented, but was never actually exercised. The
   article had to be corrected to remove the implication.
2. The article at one point attributed DistillSpec to Rahul as "his earlier paper on the idea" —
   flatly wrong; it's an existing published paper he did not write. Had to be corrected twice: first
   softened to "a paper he co-authored" (still wrong), then removed entirely once confirmed he isn't
   an author at all.
3. A generated chart initially implied a "family winner" between two loss families (bolding one
   config as "best") — which was actually a multiple-comparisons artifact this project had
   independently established doesn't hold under a fair, pre-registered comparison. Caught and
   corrected before it shipped, but it's a good example of AI defaulting to a cleaner, more
   definitive-looking narrative unless explicitly checked against the more rigorous finding already
   on record.

Each of these is the "Claude suggested/wrote X, we checked it against the actual code or data, it was
wrong because Y" pattern — genuine, verifiable, and yours to narrate with the specifics of how you
caught each one.

**Q: If AI were taken away tomorrow, what becomes impossible, and what stays unchanged?**

A reasonable, honest split: coding and cross-referencing speed drops substantially — a lot of the
verification-heavy work (grepping a large codebase, tallying exact counts, catching a mismatched
number across a dozen notes) would take much longer by hand. Drafting and iterating written analysis
would also slow down a lot. But the actual research judgment calls — deciding block efficiency was the
right metric, recognizing the REINFORCE mode-seeking mechanism, noticing the CPU-contention bug,
choosing to test the "most metric-aligned" objective as a hypothesis worth falsifying — none of that
depended on AI; those were the human decisions AI helped implement and verify faster, not decisions AI
made.

### The Single Best Question

**Q: Tell me about one conclusion you were confident in, and later discovered was wrong. What convinced you?**

The depth-weight + naive-tree combined result. Originally reported as a real win, +0.293 over JSD at
K=3 — confident in it because it matched a plausible story (depth-weighting should help by
emphasizing tokens near the acceptance boundary). Later, during a systematic audit cross-checking
every research note against the actual committed CSVs, found the +0.293 figure was a K_eval=1 result
being compared against a K=3 baseline — a mismatched, apples-to-oranges comparison. At the correct
matched K=3, the real delta was +0.03, well inside the noise floor. What actually convinced me: not a
subtle statistical argument, but a mechanical one — re-pulling the raw CSV and matching the K column
explicitly, row by row, instead of trusting the number as cited in the note. The mismatch was a plain
labeling error, and finding it required distrusting a written conclusion enough to re-derive it from
raw data directly.

---

## Quick-Hit Summary (have these cold, 2-3 sentences each)

1. **Speculative decoding, first principles.** Draft proposes K tokens cheaply; target verifies all
   K in one parallel forward pass; accept the longest prefix passing a rejection-sampling test,
   which keeps the output distribution *exactly* the target's. Metric: block efficiency (accepted
   tokens per target call). → §2
2. **Describe your architecture.** Bash scripts launching one process per GPU
   (`CUDA_VISIBLE_DEVICES`/`--device cuda:N`), JSON/JSONL state files on disk for resumability, W&B
   for tracking, a pynvml telemetry thread per run, custom training loop. Not a distributed
   scheduler — say so directly if asked. → §3
3. **Why block efficiency, not tokens/sec?** Profiling showed the harness was CPU-dispatch-bound
   (~31.6% GPU util), so tokens/sec would measure harness engineering as much as the algorithm.
   Block efficiency is a pure ratio, invariant to implementation quality. → §5
4. **What bottleneck remains?** CPU-GPU dispatch / per-step Python overhead in the draft's
   sequential `generate()` loop — draft wall-time (766s) dwarfed target verify-time (109s) at only
   ~31.6% SM utilization. Fix path: CUDA graphs / compile on the draft loop, not a bigger GPU. → §5
5. **Why did FlashAttention/SDPA diverge?** Same math, different bf16 rounding; acceptance is a hard
   threshold, so a last-mantissa-bit difference can flip a token and cascade — measured ~0.2 BE gap
   from backend alone. → §6
6. **How did DDTE work, and why only ~14%?** It delays *where* the tree branches — commits to a
   single stem, splits only at the depth draft/target actually diverge. Gains are bounded by the
   draft-target capacity gap; can't distill/verify away a 13x parameter difference entirely. → §7
   (Research Methodology)
7. **Why verifier alignment matters.** Acceptance is a *product* of per-token min(1,q/p), not a sum
   of independent likelihoods — the objective that raises average calibration isn't automatically
   the one that raises accepted-tokens. → §7 (Research Methodology)
8. **What was the hardest negative result?** The objective that directly targeted the deployment
   verifier's exact quantity was the single worst performer tested — proof that matching the eval
   metric's form doesn't guarantee a trainable gradient. → §7 (Research Methodology)

---

## 1. High-Level Pitch & Contribution

**Q: Explain your project in two minutes, to someone who knows transformers but not speculative decoding.**

LLMs generate one token at a time, each requiring a full forward pass — that's the latency
bottleneck. Speculative decoding uses a small, cheap "draft" model to propose several tokens ahead,
then has the large "target" model verify all of them in one parallel forward pass, accepting
whichever prefix matches what the target would have generated itself. The key metric is **block
efficiency** (BE): accepted tokens per target call — a hardware-agnostic multiplier on decode
throughput. My project asked which training objective for the draft actually raises BE, tested on
Qwen 0.6B/8B and a harder Qwen 1.7B/32B pair, and separately whether a newer verification-time
technique (delayed tree-branching) helps *already-distilled* drafts, which its own paper never
tested.

**Q: What problem were you solving? Why does speculative decoding exist? Why not just use the larger model directly?**

Autoregressive decoding is sequential and memory-bandwidth-bound at batch size 1 — you pay a full
forward pass for one token, when the GPU has far more spare compute than that token needs.
Speculative decoding amortizes that fixed cost over multiple tokens per target call. And you *are*
using the larger model's output — the target's distribution is exactly preserved via rejection
sampling (§2), so this is a lossless acceleration technique, not an approximation. The question
isn't "target vs. draft," it's "how do I get target-quality output while paying for mostly
draft-model compute."

**Q: What was your personal contribution? What existed before you joined?**

Rahul developed the verifier implementations and guided the research. I built the training
infrastructure, experiment platform, and evaluation pipeline; implemented new distillation
objectives; extended two of his techniques with variations of my own (a loss family built on his
verifier codebase, and the DDTE branch-point derivation below); and ran the experimental analysis.

---

## 2. Speculative Decoding Fundamentals

**Q: Walk me through it step by step. Student 1B, teacher 70B, generating "The cat sat on the ...".**

1. Draft (1B) autoregressively proposes K tokens, e.g. "mat", "and", "purred" — cheap, sequential,
   small forward passes.
2. Target (70B) runs **one** forward pass over the whole draft sequence in parallel — verifying K
   tokens costs about the same as generating 1.
3. For each position, compute acceptance probability α = min(1, q_target(token)/p_draft(token)).
4. Accept tokens left-to-right while a Bernoulli(α) draw succeeds; at the first rejection, resample
   from a corrected residual `(q − p)⁺ / normalizer` — this is what makes the whole procedure
   **exact**, not approximate.
5. The target always emits at least one new token itself (the resample, or a bonus token if
   everything was accepted) — a block never produces zero new tokens.

**Q: Why is it mathematically exact? Why doesn't it change the model's distribution?**

The accept/reject + residual-resample step is constructed so the marginal distribution of the
emitted token, integrated over the draft's proposal, equals exactly q_target's — a form of
rejection sampling, proven in the original Leviathan et al. paper. That's why it's safe in
production: target-model quality guaranteed, none lost to distillation.

**Q: Why is rejection sampling needed? What if every draft token is rejected? What if acceptance is always 1?**

Rejection sampling is what preserves exactness — without correcting the residual, tokens the draft
over-proposes would be systematically over-represented. If every token is rejected, you fall back
to the target generating one token itself (BE = 1, the autoregressive floor) — never worse than
plain decoding. If acceptance is always 1, BE = K+1 every call (the ceiling for tree width K).

**Q: What determines the expected speedup? Why does acceptance rate matter more than perplexity?**

Expected accepted tokens per call = 1 + Σ_k ∏_{i≤k} α_i — a product, so dominated by the weakest
link near the front. Perplexity averages surprise over the whole vocabulary; acceptance is a
specific min(1,q/p) test at the exact sampled token. A model can have excellent perplexity while
being poorly calibrated at exactly the positions that matter for accept/reject — which is precisely
why some "more aligned on paper" objectives (see §7) turned out to be the worst performers.

---

## 3. Systems Architecture: Orchestration, Checkpointing, Multi-GPU

**This is the section to get exactly right — be precise, don't oversell.**

**Q: Describe the architecture. Walk me through your orchestration layer / experiment pipeline.**

Honest answer: **not** a distributed job scheduler (no Celery, Redis, or queue daemon — confirmed
absent from the whole repo). Bash scripts launch one process per GPU via GPU-indexed for-loops and
background jobs (`&`/`wait`), with **JSON/JSONL state files on disk** as the resumability mechanism:
- `scripts/eval_grid.sh:159-163` — `for gpu in "${!NAMES[@]}"; do run_gpu_eval "$gpu" "${NAMES[$gpu]}" & done; wait`, GPU assignment by array index → `--device cuda:$gpu`.
- Per-prompt eval resume state: `.state.jsonl` files (`eval_io.py:24-49`) — `eval.py` checks this before recomputing any (mode, K, L, checkpoint, dataset) cell.
- Training resume state: `checkpointing.py:36` writes `state.json` inside `ckpt_latest/` (step count, optimizer state, best BE); `checkpointing.py:38-48` reads it back on `--resume`.
- Sweep-level resumability: `if [[ -e "${out}/ckpt_best" ]]; then SKIP; fi` (`scripts/sweep_prefix_overlap.sh:145`).
- Multi-slot GPU sharing (packing >1 run per GPU) via modulo arithmetic across bash background jobs (`scripts/sweep_prefix_overlap.sh:178-187`), not a scheduler.
- Failure handling: `set -uo pipefail` (not `-e`, so one failed cell doesn't kill other GPUs' loops, `scripts/eval_grid.sh:78`); no automatic retry/requeue — a human re-runs the same script after a crash.

If asked directly "is there a real scheduler" — **no**, and don't imply otherwise.

**Q: Why wasn't Slurm enough? Why didn't you use a real scheduler?**

This didn't need multi-node scheduling, priority queues, or resource-limit enforcement — the actual
problem was single-node, a handful of GPUs, failure mode = "a process died mid-sweep, don't redo
finished work." A bash loop + JSON state file solves exactly that at a fraction of the setup cost;
Slurm would have solved problems I didn't have. Right-sized tooling, not under-engineering.

**Q: What exactly was checkpointed? How did crash recovery / resumability work?**

Training: `ckpt_latest` (resumable — step count, optimizer state via `state.json`) and `ckpt_best`
(val-best model only, no optimizer state — only these two kept per run to avoid blowing disk quota
across a multi-combo sweep). Eval: per-prompt results keyed by (mode, K, L, checkpoint, dataset) in
`.state.jsonl`, so a killed eval run resumes at the exact next incomplete prompt without reloading
models.

**Q: Why multiple GPUs — data parallel? Model parallel? NCCL?**

Neither, in training. Confirmed zero `torch.distributed`/DDP/FSDP/NCCL usage anywhere in `train.py`
or the rest of the repo. "Multi-GPU" here means **N independent single-GPU processes**, each pinned
via `--device cuda:N` or `CUDA_VISIBLE_DEVICES`, running different configs in parallel — not one
model sharded or replicated. Right call at this scale: drafts up to 1.7B fit comfortably on a single
80GB A100 with a full fine-tune, so DDP/FSDP's complexity wasn't needed. Don't imply distributed
training that didn't happen.

**Q: Why isolate vLLM in its own environment? Why separate environments in general?**

vLLM pins specific, often newer/incompatible versions of torch/transformers/CUDA libraries relative
to the main training venv — installing it into the same environment risks breaking the training
stack's pinned dependencies. A separate `venv-vllm` (`scripts/setup_vllm_env.sh`) avoids that;
`train.py` spawns the vLLM pass@k subprocess pointing at that separate interpreter.

---

## 4. GPU Memory & Quantization

**Q: Walk me through the memory-accounting story. Why did you move through LoRA → quantization → full fine-tune?**

Real hardware progression: 16GB Colab → Kaggle's 30GB → a 40GB A100 → a final 80GB A100. LoRA and
quantization were shed one at a time as headroom grew, ending at a full fine-tune with a
full-precision teacher and no quantization at all. Each swap had a specific, nameable memory reason,
not habit — narrate it as that sequence, not a list of techniques.

**Q: Why did LoRA help? Why not use it throughout?**

LoRA freezes the base weights and trains small low-rank adapter matrices instead, so the optimizer
state that dominates memory only needs to cover the adapter's parameters, not the full model — a
large cut precisely when GPU memory (not compute) was the binding constraint. `train.py`'s own
`USE_LORA` flag defaults to `False` ("full fine-tune is default"); once 80GB A100s were available,
there was no reason to keep the cheaper, lower-capacity option — full fine-tuning has strictly more
trainable capacity and no adapter-merge step.

**Q: Why did NF4 slow things down? Why 8-bit Adam instead?**

`--load_in_4bit` (`train.py:276-284`): bitsandbytes NF4 pays a dequantization cost on **every
single-token forward pass**. Flat losses call `teacher.generate()` for `max_new_tokens` sequential
steps, so this cost repeats — "~15-20x slower per step for flat losses than for tree losses (which
only need ~L sequential steps)." `--optim_8bit` (`train.py:285-292,607-613`) instead quantizes the
*draft's* optimizer state (int8 m/v instead of fp32, "roughly halves optimizer memory") — orthogonal
to teacher quantization, and the preferred fix wherever memory allowed, since it frees enough
headroom to keep the teacher in full bf16 and avoid NF4's slowdown entirely.

**Q: What exactly lives in optimizer state? How much memory does Adam actually use?**

AdamW keeps two moment buffers per trainable parameter (m: running mean of gradients, v: running
mean of squared gradients), fp32 by default. For a 0.6B model: "≈12GB, so the A100 40GB fits fine"
(`train.py:112-114` comment) — this is why optimizer-state size, not just weight size, is often the
real memory bottleneck.

**Q: Why bf16?**

Default dtype throughout (`--dtype bf16`, `README.md:569`) — wider dynamic range than fp16 (no
loss-scaling needed) and half the memory/bandwidth of fp32, negligible quality loss at this model
scale. Its low mantissa precision is also *why* the FlashAttention/SDPA divergence in §6 exists at
all.

---

## 5. Inference Serving: vLLM, Paged KV Cache, Continuous Batching

**Q: Explain paged KV cache. Explain continuous batching. Why are they faster? Why is fragmentation bad?**

Paged KV cache: instead of one large contiguous memory block per sequence's attention keys/values
(which fragments badly as variable-length sequences finish and free memory), it manages KV cache in
small fixed-size pages, like OS virtual memory — memory packs tightly regardless of sequence-length
variance, and a finished sequence's pages are immediately reusable, no defragmentation pass.
Continuous batching: instead of waiting for an entire fixed batch to finish before starting the next
(classic static batching, wasting GPU time on short sequences waiting for the longest), new
sequences are admitted the instant a slot frees up — GPU utilization stays high, no idling for
stragglers. Both are why vLLM's async pass@k backend generated tens of thousands of samples fast
enough for hundreds of checkpoint × dataset combinations; a naive HF `generate()` loop
(`PASSK_BACKEND=hf` fallback) is blocking and much slower at this scale.

**Q: Why tensor-parallel only for the 32B teacher? Was it actually used?**

The flag (`--tensor_parallel_size`, `scripts/passk_eval.py:34,140,156`) exists specifically because a
32B model can be too large for one GPU. Be precise if asked: **every real launcher script in this
repo runs it at the default of 1** — none override it. The 32B teacher's pass@k inference fit on a
single 80GB GPU in practice (inference-only, no optimizer state or multi-rollout activations
competing for memory the way training does), so sharding was available but never actually exercised.
Say this plainly if asked "did you shard it" — no.

---

## 6. Numerics: FlashAttention, SDPA, Determinism

**Q: Why do mathematically equivalent kernels disagree? Why does speculative decoding amplify tiny differences?**

FlashAttention and SDPA compute the same attention formula with different computation orders/kernel
fusions, which round differently in bf16 — usually invisible. But acceptance here is a **hard
threshold** (min(1,q/p)), so a last-decimal difference can flip whether a token is accepted, and one
early rejection cascades through the rest of a generation, discarding everything sampled after it.
Empirically: two otherwise-identical runs, different only in attention backend, diverged by ~0.2
block efficiency — large enough to look like a real result above the noise floor when it was purely
kernel-level rounding. Fix: pin one backend everywhere (`--force_attn sdpa` default) and log which
one ran.

**Q: Do you claim determinism? What about floating point more generally?**

No — this project doesn't claim bit-for-bit determinism (no fixed-algorithm cuDNN/cuBLAS
determinism flags were used). bf16's 7-bit mantissa makes kernel-to-kernel rounding differences
expected and normally harmless; what's unusual here is that the acceptance threshold turns an
invisible rounding difference into a visible, cascading, boolean flip. The actual fix was narrower
than true determinism — pin one backend so *comparisons* stay valid, not chase bit-exact
reproducibility.

**Q: What assumptions make block efficiency hardware-independent? How would you benchmark a production system?**

BE = accepted tokens / target call is a pure counting ratio — no wall-clock, FLOPs, or bandwidth term
in its definition, so it holds whether the target call takes 50ms on an A100 or 20ms on an H100. It
stops being the right proxy if the draft itself becomes the dominant cost, or if a serving stack
changes the relative cost of draft vs. target steps (e.g., heavy batching amortizing target calls
across requests). For production, I'd add exactly what this harness didn't need: wall-clock latency
and tokens/sec under a real serving stack (vLLM/continuous batching, not a blocking `generate()`
loop), measured at realistic concurrent request rates. BE stays the metric for *whether the
algorithm improved*; throughput/latency-under-load become the metric for *whether that improvement
survives contact with a real system* — this project's own dispatch-bound profiling is exactly the
evidence that those two can diverge.

**Q: Which profiler did you use? Which counters?**

`telemetry.py` — **pynvml + psutil**, a lightweight 1Hz background polling thread
(`_POLL_INTERVAL=1.0`), **not** Nsight, `torch.profiler`, `nvidia-smi` subprocesses, or CUPTI. Why
not those: NVML polling is cheap enough to run on *every eval cell* across hundreds without added
overhead, trading kernel-level granularity for always-on coverage — a deliberate tradeoff at this
scale. Columns logged: `gpu_sm_util_avg_pct`, `gpu_mem_util_avg_pct`, `gpu_vram_peak_mb`, PCIe/NVLink
throughput, plus wall-clock breakdown (`time_draft_s`, `time_target_s`, `time_verify_s`,
`time_cache_s`, instrumented via `time.perf_counter()` deltas in `main.py`).

**Q: How did you conclude CPU-GPU dispatch was the bottleneck? Show me the evidence.**

`results/per_checkpoint_sweeps_2026-07/jsd_mathhard_s123.csv`, traversal K=3 row:
`time_draft_s=766.2`, `time_target_s=108.7`, `gpu_sm_util_avg_pct=31.6`. The tiny 0.6B draft's
sequential single-token forward passes cost 7x more wall time than the 8B target's one parallel
verification pass, while SM utilization sits at only ~31.6% — nowhere near saturated. That
combination (small model dominating wall time, low utilization) is the signature of
dispatch/Python-overhead-bound execution, not compute-bound.

**Q: How would you optimize it now?**

One detailed `torch.profiler`/Nsight Systems trace to find exact per-call dispatch gaps, then target
the draft's `generate()` loop — CUDA graphs or `torch.compile` with static shapes to cut per-step
Python/launch overhead. (An actual `--compile` attempt already exists and is documented as broken:
`torch._inductor` cudagraph_trees hits an `AssertionError` from per-prompt `DynamicCache`
re-guarding, `scripts/eval_grid.sh:36-49` — a real fix needs to address that dynamic-cache
incompatibility first, e.g. `dynamic=True` without `mode='reduce-overhead'`, unvalidated here.)

---

## 7. PyTorch / HuggingFace Engineering

**Q: Custom autograd? Mixed precision? Gradient accumulation? DDP/FSDP? torch.compile? Gradient checkpointing?**

- **Custom autograd**: no — min(1,q/p) = exp(clamp(log q − log p, max=0)) is differentiable
  everywhere except a measure-zero kink; standard autograd handles it.
- **Mixed precision**: bf16 throughout (§4).
- **Gradient accumulation**: yes, `GRAD_ACCUM=8`.
- **DDP/FSDP**: no (§3) — single-GPU-per-job was sufficient at this model scale.
- **torch.compile**: attempted, currently broken (§6).
- **Gradient checkpointing**: not found in the repo — don't claim it; wasn't needed at this model
  scale (up to 1.7B student) on 80GB cards.

**Q: HF Trainer or custom loop? Why custom?**

Custom loop — `train.py` manages its own optimizer, scheduler, checkpointing, and early-stopping
directly, rather than `transformers.Trainer`. Gives direct control over per-objective loss routing
(flat vs. tree vs. enrichment vs. prefix-overlap need different forward-pass shapes), early-stopping
on a smoothed validation block-efficiency EMA rather than loss, and integrated pass@k
side-evaluation — control that's awkward to bolt onto Trainer's callback system for this many
divergent training modes.

---

## 8. Systems-Design Hypotheticals

**Q: Given 16-32 H100s, how would you redesign your platform?**

Keep the core design (independent single-GPU jobs + disk-based resumability) since it matched the
actual problem size, but invest the extra headroom in: (1) true multi-seed sweeps by default,
directly fixing the biggest current rigor gap; (2) larger K/tree-width sweeps that were previously
compute-limited; (3) actually exercising tensor-parallel sharding for a bigger teacher (e.g. testing
whether findings hold at an even wider capacity gap, like 70B+); (4) parallel multi-node evaluation
of the full verifier grid instead of one-node bash loops — at *that* scale, a real lightweight job
queue would finally pay for itself, unlike at the current single-node scale where it would have been
over-engineering.

**Q: 8 H100s, inference throughput is poor. How would you debug?**

Same playbook this project validated: check GPU utilization first (NVML polling, cheap and
always-on) — if low (like ~31.6% found here), the problem is dispatch/CPU overhead, fixed by
batching/kernel-launch reduction (continuous batching, CUDA graphs), not a bigger GPU. If
utilization is near 100%, check `gpu_mem_util_avg_pct` vs. SM util to distinguish compute-bound from
memory-bandwidth-bound. Only after that diagnosis would I reach for tensor/pipeline parallelism —
parallelizing a dispatch-bound workload across more GPUs just gives you more idle GPUs.

---

## 9. "Prove You Did It" — [PERSONALIZE THESE]

- **Draw the pipeline on a whiteboard**: use §3's honest architecture (bash orchestration + state
  files + W&B logging + eval harness), not an idealized distributed system.
- **Explain one experiment from memory / open a W&B run**: pick one you genuinely remember well —
  the enrichment sweep (K=1-6, sweet spot at K=4) or the DDTE branch-point derivation are both
  good, well-documented candidates, but only if you can walk through the real curves without notes.
- **What bug took the longest to fix?**: the CPU-contention-corrupting-eval-measurements bug (why
  eval needed one process per GPU) is real, documented, with a clear before/after.
- **If I gave you another student today, what would you tell them not to waste time on?**: a
  defensible answer is "don't chase objectives that directly target the eval metric's exact
  mathematical form assuming that guarantees trainability — check gradient allocation first," the
  project's most-repeated lesson.

---
---

# Research Methodology (for research-engineering / research-scientist-leaning interviewers)

The loss-function math, tree-loss mechanics, verifier theory, and negative-result reflection below
are separated from the systems content above — read this section if the interviewer steers toward
"why did you design the objective this way" rather than "how did you build the platform."

## R1. Loss Functions & Distillation Objectives

**Q: Why Forward KL instead of Reverse KL? Could Reverse KL work?**

Forward KL, KL(target‖student), is mode-covering — forces the student to place mass everywhere the
teacher does, which high acceptance rate needs (student's distribution must *cover* the teacher's
support). Reverse KL, KL(student‖target), is mode-seeking — the student collapses onto a few
high-confidence modes and ignores the teacher's tail, hurting acceptance on exactly the tokens where
the teacher assigns real but non-dominant probability. Both implemented
(`losses/flat.py:120-129`); forward-KL-family objectives (JSD, an FKL/RKL blend) are the strongest
baselines.

**Q: Why JSD? Why Total Variation / what does l1 encourage?**

JSD is symmetric and bounded (unlike KL, which can blow up near-zero-probability teacher-likely
tokens) — a practical training-stability advantage; selected as primary baseline after the sweep
(`forward_kl`, `reverse_kl`, `jsd`, `l1`) gave the broadest win across verifiers. There's no loss
literally named "TV" — `l1` is the implemented equivalent (`Σ|p_student − p_teacher|`, "Equivalent
up to a constant to 2·TV(p_s, p_t)," `losses/flat.py:80-90`), linear-in-error rather than log-error,
giving a more stable gradient when distributions are far apart.

**Q: What makes a loss "verifier-aligned"? Can two losses match perplexity but differ in acceptance?**

A verifier-aligned loss, in principle, has a gradient that directly increases the specific
product-of-min(1,q/p) quantity a verifier measures. But "aligned in form" and "trainable in
practice" are different: the objective literally targeting the deployment verifier's exact quantity
was the single worst performer tested (§R2), because gradient allocation (where the signal actually
lands, token-by-token and depth-by-depth) mattered more than formal resemblance to the metric. Two
losses can match perplexity while differing in acceptance because perplexity averages over the
whole vocabulary while acceptance depends on the specific min(1,q/p) value at the specific sampled
token.

**Q: Why not train directly for acceptance instead of likelihood?**

That's exactly what several of the ~34 objectives tried — the project's central negative result.
Every direct or REINFORCE-style "train for accepted-tokens directly" formulation either collapsed
(log-space tree losses, −0.4 to −0.6 BE at the harder pair) or was the single worst performer tested
(prefix-overlap's direct-verifier-targeting objective). Plain KL/JSD turned out to be a *more*
trainable proxy for acceptance than acceptance-shaped losses were, in every variant tested.

## R2. Tree Losses & Negative Results

**Q: What motivated tree losses? Why is token-level supervision insufficient?**

Token-level losses train the student to match the teacher's distribution at each position
independently, ignoring the sequential, product-structured nature of block acceptance — a token
late in the tree only matters if everything before it was accepted. Tree losses train on the
draft's own sampled tree structure so the loss reflects that dependency (21 entries in
`losses/tree.py:572-594`).

**Q: Did tree objectives help? Why did log-space reformulations fail? Why did REINFORCE fail?**

No — the project's central negative result. Every tree-loss variant matched flat JSD at best; the
log-space reformulations (`traversal_log`, `nss_log`, meant to fix vanishing gradients at depth)
collapsed catastrophically at 1.7B/32B: traversal Δ −0.4 to −0.6 at K1-K3, −0.22 to −0.39 at K4
(`project_report.md:18,29`). Mechanism: the log-space fix cures gradient *vanishing* but introduces
gradient *misallocation* — it amplifies gradient on low-α, deep, unlikely nodes, overfitting the
draft to tree tails it can't reach (`notes/tree_losses_research_note.md:4`). `bv_tree`/`gbv_tree`
separately collapse to zero gradient (single-token gate structure).

REINFORCE (`*_tree_pg`) used a whole-tree scalar reward r = E[BE] times the summed log-probability
of all sampled tokens (`notes/tree_losses_research_note.md:106-112`). Three structural problems: (1)
no depth credit assignment — a shallow decisive token and a deep never-accepted token get the same
advantage; (2) the global scalar baseline correlates with prompt difficulty, not which tokens were
responsible; (3) most fundamentally, REINFORCE score-function ascent *concentrates* q (mode-seeking),
while BE requires q to *spread toward* p (mode-covering) — so BE declined monotonically (4.81 → 4.0
across training). Mechanically correct, structurally misaligned — removed from the codebase (branch
`fix_bugs`, June 2026).

## R3. Verifiers & DDTE

**Q: Why compare nine verifiers? Which ones, what do they assume?**

`naive, nss, specinfer, spectr, khisti, max, bv, gbv, traversal` (`scripts/eval_grid.sh:85`). `naive`
= classic single-path SD (OT-based); `bv` = Block Verification (single-path, arXiv:2403.10444);
`nss`, `spectr` (arXiv:2310.15141), `specinfer` (arXiv:2305.09781) = multi-path, OT-based; `khisti` =
canonical decomposition (arXiv:2410.18234); `max` combines the others; `traversal`
(arXiv:2505.12398) and `gbv` (Rahul's own paper, arXiv:2602.16961) are multi-path but not OT-based.

**Q: Which verifier won, and why? How did DDTE work? Why doesn't everyone use it?**

Empirically `traversal` was strongest, and delayed tree branching (DDTE, Rahul's own recent paper,
arXiv:2602.16994) on top of it gave the largest single lift: up to ~0.8 additional accepted
tokens/call (~14% on the larger pair), transferring to *trained, distilled* drafts, which that paper
never tested. It works by delaying *where* the tree branches — a single stem, splitting into
candidates only at the depth draft/target actually diverge, instead of branching at token 1 —
changing tree *structure*, not any model's distribution, which is why it stacked additively with the
training-side win (enrichment) rather than competing with it. I wouldn't call it externally crowned
"SOTA" — it's Rahul's own newest work, empirically strongest in this project's data. It needs a
branch-point setting; I derived it analytically per-checkpoint from each model's own mean acceptance
depth rather than grid-searching it. Adoption lag is unsurprising for a 2026 paper.

## R4. Experimental Design & Methodology

**Q: Why 500+ experiments? What stayed fixed? How did you estimate noise? Why single seed, and does it matter?**

Scale came from systematically sweeping ~34 objectives × hyperparameters (LR, warmup, lr_min, weight
decay) × 2 model pairs × verifier × K × dataset — a genuine ablation grid, not 500 independent
ideas. Fixed across comparisons: dataset splits, seed (where single-seed, an explicit limitation),
eval prompt count (n=100). Noise floor estimated from near-duplicate configurations' spread (SE ≈
0.10-0.15 at n=100) — used as the default threshold for graying out inconclusive deltas (|Δ| <
0.15). Single-seed at 1.7B/32B is an acknowledged gap: several results "sign-flip across K" in a way
consistent with pure noise, so nothing at that pair is confidently conclusive without a second seed
— a real limitation I'd be upfront about, not defend.

## R5. Research Thinking & Reflection — [PERSONALIZE THESE]

Pick whichever is genuinely true for you and add the specifics only you know — don't recite these
as a script.

**Q: What surprised you most? Hardest negative result? Biggest mistake? Weakest assumption?**

Candidates, all real: (1) the objective most directly targeting the deployment verifier's exact
quantity was the single worst performer, twice (log-space tree loss, prefix-overlap's traversal
objective) — proof that formal alignment with the eval metric doesn't guarantee a trainable
gradient; (2) REINFORCE was mechanically correct RL but structurally mode-seeking when the goal
needed mode-covering — a lesson that an estimator's *classical correctness* doesn't guarantee its
*direction* matches what you need; (3) weakest assumption: single-seed 1.7B/32B results are treated
as if reliable when the noise floor (SE ≈ 0.10-0.15) means several deltas could be pure noise.
**[PERSONALIZE: your biggest mistake — an early result you misread, a config bug that cost a sweep,
or a claim you had to walk back after fair comparison.]**

## R6. Hardest Questions (research-lab-style)

**Q: Why is maximizing likelihood not equivalent to maximizing acceptance?**

Likelihood is a sum over independent token positions, weighted equally. Acceptance is a *product*
over a block (E[BE] = 1 + Σ_k ∏_{i≤k} α_i), so early positions have outsized leverage, and the
relevant quantity at each token is min(1,q/p), not log-likelihood. A model can maximize likelihood
while remaining poorly calibrated at exactly the early, decisive positions that dominate the
product — exactly what this project found when directly-acceptance-targeting objectives
underperformed plain likelihood-matching.

**Q: What properties must an objective have to be verifier-aligned?**

Matching the verifier's mathematical form is *not* sufficient — the gradient must be well-behaved
(not vanishing at depth, not concentrated on unreachable low-probability nodes, not mode-seeking
when the verifier needs mode-covering). The objectives that worked (JSD, then enrichment) weren't
the ones with the closest formal resemblance to the acceptance formula — they were the ones whose
gradient, in practice, spread the student's distribution to cover the teacher's.

**Q: Why did your best method win — better probability estimation, lower variance, or something else?**

DDTE won by exploiting tree *structure*, not per-token probability estimation — it changes *where*
the tree branches rather than any model's output distribution. A genuinely different lever from
every training-side objective tested, which is why it stacked additively with enrichment rather than
competing with it.

**Q: Which negative result most changed your understanding? If you had two more months, what would you pursue first?**

The prefix-overlap `traversal` objective — direct evidence that "optimize the literal thing you're
evaluated on" can be the *worst* choice if you don't separately verify the gradient is trainable.
Reframed the whole objective-search from "find the most metric-aligned formula" to "find the formula
whose gradient is well-behaved." First next step: a second (and third) seed on every 1.7B/32B result
before trusting any current sign-flipping deltas — a prerequisite for everything else being credible
at that pair. Then, since DDTE's win is structure-side and enrichment's is data-diversity-side rather
than loss-form-side, test whether those two axes generalize better than loss-form has, e.g.
enrichment at higher K or a third model-family pair, rather than proposing yet another
acceptance-shaped loss.
