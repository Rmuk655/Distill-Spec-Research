# DistillSpec Pipeline — Qwen3 / A100

Simplified, single-GPU research pipeline for verifier-aligned distillation of
speculative-decoding draft models.  Three scripts:

| Script           | What it does                                          |
| ---------------- | ----------------------------------------------------- |
| `train.py`       | Distil a Qwen3-0.6B draft against a Qwen3-8B teacher  |
| `eval.py`        | Block-efficiency + throughput on a held-out prompt set |
| `inference.py`   | Speculative decoding on a single user prompt          |

The verifier code under `verifiers/` is a verbatim copy of the reference
implementation at <https://github.com/Rmuk655/Distill-Spec-Research/tree/main/GBV>
(only the `__init__.py` is new — it lets the parent scripts import the verifier
files without modifying them).

---

## Setup (A100, one time)

```bash
# 1. clone and step in
git clone -b Pipeline https://github.com/Rmuk655/Distill-Spec-Research.git
cd Distill-Spec-Research

# 2. virtualenv + dependencies
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 3. download all 5 evaluation datasets + GSM8K train split
#    --n 1000 gives 1000 held-out eval prompts (default is 100)
python -m data_io.download --train --n 1000

# 3b. (optional) download MATH hard-level datasets for math probe training
#     Fetches EleutherAI/hendrycks_math (7 subjects, ~6532 level-4+5 problems total):
#       math_val.jsonl   —  200 problems  (during-training checkpoint selection)
#       math_eval.jsonl  — 1000 problems  (held-out final eval, same role as gsm8k_eval)
#       math_hard.jsonl  — remainder ~5332 problems  (training)
python -m data_io.download --datasets math_hard,math_val,math_eval

# 4. (optional) log in to Weights & Biases for training curves
wandb login
```

Runs on any CUDA GPU with bfloat16 support.

---

## Storage layout

```
Distill-Spec-Research/
├── train.py             # main trainer  — edit HARDCODED CONSTANTS block at top
├── eval.py              # block-efficiency eval + results.csv writer
├── inference.py         # single-prompt sanity check
├── losses/
│   ├── flat.py          # forward_kl, reverse_kl, jsd, l1
│   └── tree.py          # 11 tree losses (3 divergences + 8 verifier-aligned)
├── verifiers/           # VERBATIM copy of /GBV — do not edit unless syncing upstream
├── data_io/
│   ├── download.py      # fetch gsm8k / math_hard / math_val / alpaca / math500 / humaneval / mtbench / spec_bench
│   └── raw/             # downloaded JSONL files (gitignored)
├── scripts/
│   └── setup_a100.sh    # one-shot env setup wrapper
├── checkpoints/         # train.py writes here (gitignored)
├── results.csv          # eval.py appends one row per (mode × K × L × dataset)
└── requirements.txt
```

---

## Training

The training loop is **one file** (`train.py`) with all knobs at the top of
the file in a `HARDCODED CONSTANTS` block.  Edit those, or pass the few most
common ones on the command line.

```bash
# Flat baselines
python train.py --loss forward_kl
python train.py --loss reverse_kl
python train.py --loss jsd

# On-policy divergences (use the student's own draft tree)
python train.py --loss kl_tree
python train.py --loss rev_kl_tree
python train.py --loss jsd_tree

# Verifier-aligned tree losses (the paper contributions)
python train.py --loss bv_tree         --lr 1e-5
python train.py --loss gbv_tree        --lr 1e-5
python train.py --loss traversal_tree
python train.py --loss naive_tree
python train.py --loss nss_tree
python train.py --loss specinfer_tree
python train.py --loss spectr_tree
python train.py --loss khisti_tree

# Resume after a kill / crash
python train.py --loss kl_tree --resume

# Math domain probe — swap train/val datasets without touching anything else
python train.py --loss jsd        --train_dataset math_hard --val_dataset math_val --steps 2000
python train.py --loss naive_tree --train_dataset math_hard --val_dataset math_val --steps 2000
```

### Swapping the teacher (e.g. 32B)

The teacher defaults to `Qwen/Qwen3-8B`.  Override it with `--teacher` (a model id
or local path).  For a teacher too large to fit in bf16 on one GPU (e.g. 32B),
add `--load_in_4bit` to load it in 4-bit NF4 via bitsandbytes (the frozen teacher's
quantization loss is negligible for distillation):

```bash
# 32B teacher, draft unchanged — needs `pip install bitsandbytes accelerate`
python train.py --loss jsd --teacher Qwen/Qwen3-32B --load_in_4bit \
                --train_dataset math_hard --val_dataset math_val --steps 4000
```

A bigger teacher only helps when the bottleneck is teacher-side (the 8B teacher has
little left to teach at the contexts that matter).  If the bottleneck is the draft's
capacity, a larger teacher makes the gap *bigger*, not smaller — use `--diagnose`
and `val/forgetting` (below) to tell which regime you are in before scaling up.

### Enrichment training (`jsd_flat_enrich`)

Instead of training on the teacher's *greedy* rollout, enrichment training uses the teacher's *stochastic* speculative decoding loop as the training distribution.  The draft generates K tokens per step via the actual SD loop (teacher accepts/rejects), and the flat JSD loss is applied against the resulting sequence.  K controls how many on-policy steps are taken per training update: K=1 is one stochastic rollout token, K=3 gives three steps per update (richer signal, ~3× slower per step).

```bash
# K=1 enrichment — one stochastic rollout per step (8 K steps is sufficient;
#   enrich converges faster than 40K-step flat runs because the on-policy
#   distribution shifts faster)
python train.py --loss jsd_flat_enrich --K 1 --steps 8000 \
    --train_dataset math_hard --val_dataset math_val \
    --output checkpoints/jsd_flat_enrich_K1_mathhard_s123

# K=3 enrichment — three steps per update (richer multi-step signal)
python train.py --loss jsd_flat_enrich --K 3 --steps 4000 \
    --train_dataset math_hard --val_dataset math_val \
    --output checkpoints/jsd_flat_enrich_K3_mathhard_s123

# Second seed for reproducibility
python train.py --loss jsd_flat_enrich --K 1 --steps 8000 \
    --train_dataset math_hard --val_dataset math_val \
    --output checkpoints/jsd_flat_enrich_K1_mathhard_s456 --seed 456

# Warmer teacher temperature (more diverse paths — monitor train/path_diversity)
python train.py --loss jsd_flat_enrich --K 3 --steps 4000 \
    --train_dataset math_hard --val_dataset math_val \
    --teacher_temp 1.5 \
    --output checkpoints/jsd_flat_enrich_K3_mathhard_s123_ttemp1.5
```

`train/path_diversity` (logged to W&B and console as `pathdiv=`) is the key enrichment-specific signal: fraction of token positions where the K teacher rollouts disagree.  Values consistently above 0.15 mean the teacher is giving diverse training signal; below 0.05 means paths are collapsing (raise `--teacher_temp` or K).

### Math eval

After training, evaluate on the 1000-problem held-out set:

```bash
python eval.py --checkpoint checkpoints/<run>/ckpt_best --dataset math_eval
```

`math_val` (200 problems) is only for during-training checkpoint selection — never report it as a final number. `math_eval` (1000 problems) is the held-out set, analogous to `gsm8k_eval`.

### Evaluation methodology — iteration vs paper

- **Day-to-day iteration:** train and eval on `math_hard` / `math_eval`. Fast feedback; in-domain.
- **For the paper:** report on **Spec-Bench** — the field-standard speculative-decoding benchmark (480 prompts, 6 categories: multi-turn, translation, summarization, QA, math, RAG). It is *cross-domain* relative to math-only training, so it measures whether the distilled draft **transfers** beyond the training distribution — the generalization claim reviewers expect. Train on one distribution, eval on a different held-out one; never report numbers on the training distribution.

```bash
# fetch Spec-Bench (480 prompts, from the official hemingkx/Spec-Bench repo)
python -m data_io.download --datasets spec_bench

# eval a checkpoint on Spec-Bench (break down by the per-row "category" field for the paper table)
python eval.py --checkpoint checkpoints/<run>/ckpt_best --dataset spec_bench
```

### Combining a flat backbone with an acceptance-aligned tree loss

Two ways to mix a dense flat loss with a verifier-aligned tree loss.  Both reuse
`--aux_loss` (named by verifier, e.g. `naive_tree`, `khisti_tree`).

```bash
# (1) Additive — total = primary + aux_weight * aux.  Real per-level gradient
#     through q, with the natural (depth-decaying) survival weight.
python train.py --loss jsd --aux_loss naive_tree --aux_weight 0.1

# (3) Depth-weight — multiply the flat primary loss by a detached depth weight
#     w(d), d = E[tau_V] of the draft tree.  Curriculum reweighting only (no
#     acceptance gradient).  All forms below are normalised to E[w]~=1.

# (3a) Linear — w = d / EMA(d).  The researcher's literal "tree_depth * loss",
#      mean-normalised so there is no hidden LR change.
python train.py --loss jsd --aux_mode depth_weight --aux_loss khisti_tree --depth_linear

# (3b) Exponential — w = exp(depth_lambda*(d - EMA(d))).  Signed knob (below).
python train.py --loss jsd --aux_mode depth_weight --aux_loss khisti_tree --depth_lambda 0.5
```

`--depth_lambda` is a single signed knob: `>0` amplifies the loss on deep-tree
prompts, `<0` amplifies shallow, and **`--depth_lambda 0` is the control** — it
must reproduce a plain `--loss jsd` run (use it as a correctness check).
`--depth_linear` ignores `--depth_lambda` and uses `w = d / EMA(d)` instead.

> Both forms divide out the running mean of `d`, so `E[w]~=1` and the effective
> learning rate is unchanged — no need to drop `--lr` to compensate, and the
> normalisation self-adapts as the tree deepens during training.

Each combo gets its own checkpoint dir + W&B run name + tags, so `λ=0.5`,
`λ=-0.5`, and `lin` never overwrite each other:

| Flags | Checkpoint dir / W&B run |
|---|---|
| `--loss jsd --aux_loss naive_tree --aux_weight 0.1` | `jsd+naive_treex0.1` |
| `--loss jsd --aux_mode depth_weight --aux_loss khisti_tree --depth_linear` | `jsd+dw_khisti_tree_lin` |
| `--loss jsd --aux_mode depth_weight --aux_loss khisti_tree --depth_lambda 0.5` | `jsd+dw_khisti_tree_lam0.5` |

In `depth_weight` mode, `train/depth_w` (applied weight) and `train/depth_d`
(raw `E[tau_V]`) are logged to W&B — watch `depth_d` to see whether the draft
tree is actually getting deeper during training.

**Note — `--val_temp` (default 0.2):** val block-efficiency is decoded at a low,
near-deterministic temperature so the curve is readable.  The 0.8 training temp
made `val/block_eff` swing ±0.4 (pure sampling noise) and masked real effects.
`--val_temp` cannot be 0 (temperature divide).

What happens during training (per step):

1. Pick a prompt from the train dataset (`gsm8k_train.jsonl` by default; override with `--train_dataset`).
2. **Flat loss** path: teacher generates `MAX_NEW_TOKENS` tokens, student
   forwards on the same sequence with grad, loss = divergence(student_logits,
   teacher_logits) on the generated portion.
3. **Tree loss** path: sample K student draft paths (no grad), score every
   tree node under the teacher (no grad), then re-score under the student
   with grad → `q_probs_dict`.  Loss = `−E[τ_V](q, p, K, L)`.
4. Gradient accumulation over `GRAD_ACCUM` micro-steps, AdamW step, linear
   warmup → constant LR.
5. Every `VAL_EVERY` steps: decode block-efficiency on `gsm8k_val.jsonl`
   (verifier matched to the loss, at `--val_temp`).  If improved, save
   `ckpt_best/`.  W&B logs `val/block_eff`.
6. Every `SAVE_EVERY` steps: write `ckpt_latest/` + `state.json` so
   `--resume` works.

### W&B metrics logged during training

| Metric | When | What it tells you |
|---|---|---|
| `train/loss`, `train/lr`, `train/grad_norm` | every `LOG_EVERY` | standard training health |
| `val/block_eff` | every `VAL_EVERY` | the north-star metric (on the 25-prompt val set, `--val_temp` low to cut variance) |
| `val/best_block_eff` (summary) | on each new best | pinned best — the W&B *summary* now shows best, not the last value |
| `val/forgetting` | every `VAL_EVERY` | **backward-transfer / forgetting** (Lopez-Paz & Ranzato 2017): mean drop of each val prompt from its personal-best BE. 0 = no regression; large = the student learned prompts then lost them (signature of the teacher *confusing* a limited-capacity student). Reuses the **same 25 val prompts** — zero extra compute. |
| `train/path_diversity` | every `LOG_EVERY`, **flat-enrich K>1 only** | fraction of token positions where the K teacher rollouts disagree. `< 0.05` → enrichment paths are near-identical (teacher has nothing diverse to add — raise `--teacher_temp`); `> 0.15` → paths are diverse, so if BE still ties the baseline the bottleneck is the student, not the signal. |

`val/forgetting` and `train/path_diversity` are also echoed to the console
(`forget=…`, `pathdiv=…`).

`gsm8k_val.jsonl` and `gsm8k_eval.jsonl` use **non-overlapping** index ranges
of the GSM8K test split (`items[0:100]` for val, `items[200:200+n]` for eval).
With `--n 1000` the eval pool is `items[200:1200]` (within the 1319-prompt test
set), so the val-best checkpoint selection does not bias the reported eval numbers.

### LoRA toggle

Full fine-tuning is the default.  To switch to LoRA, set `USE_LORA = True`
at the top of `train.py` and rerun.  Adapter weights are still saved through
`save_pretrained()` so eval / inference can load them the same way.

---

## Evaluation

```bash
# Eval one checkpoint under one verifier mode
python eval.py --checkpoint checkpoints/kl_tree/ckpt_best --mode gbv

# Sweep all verifier modes for one checkpoint (comma-separated)
python eval.py --checkpoint checkpoints/jsd_flat_enrich_K1_mathhard_s123/ckpt_best \
               --modes traversal,bv,naive,specinfer,spectr,khisti,gbv,nss,max \
               --K 1 --L 8 --n 100 --dataset math_eval --device cuda:0

# Different K, L, or dataset
python eval.py --checkpoint checkpoints/gbv_tree/ckpt_best \
               --mode gbv --K 4 --L 8 --dataset math500

# Baseline (untrained Qwen3-0.6B)
python eval.py --checkpoint Qwen/Qwen3-0.6B --mode gbv

# Parallel 4-GPU sweep — pin each process to its own GPU
python eval.py --checkpoint checkpoints/jsd_mathhard_s123/ckpt_best \
    --modes traversal,bv,naive,specinfer,spectr,khisti,gbv,nss,max \
    --K 1 --L 8 --n 100 --dataset math_eval --device cuda:0 &
python eval.py --checkpoint checkpoints/jsd_flat_enrich_K1_mathhard_s123/ckpt_best \
    --modes traversal,bv,naive,specinfer,spectr,khisti,gbv,nss,max \
    --K 1 --L 8 --n 100 --dataset math_eval --device cuda:2 &
wait

# With diagnose — correlation check after the timed loop (adds ~10 min for n=100)
python eval.py --checkpoint checkpoints/jsd_flat_enrich_K1_mathhard_s123/ckpt_best \
    --modes traversal --K 1 --n 100 --dataset math_eval --diagnose --device cuda:0
```

Every eval cell appends one row to `results.csv`.  Key columns:

| Column group | Columns |
|---|---|
| Identity | `timestamp`, `checkpoint`, `dataset`, `mode`, `K`, `L` |
| Core metrics | `block_eff`, `throughput_tok_s`, `avg_tree_nodes` |
| Time breakdown | `time_draft_s`, `time_target_s`, `time_verify_s`, `time_cache_s`, `time_tokenizer_s` |
| GPU telemetry | `gpu_sm_util_avg_pct`, `gpu_mem_util_avg_pct` (HBM-bus %), `gpu_vram_peak_mb`, `gpu_pcie_tx/rx_avg_kbs`, `gpu_nvlink_tx/rx_avg_kbs`, `cpu_util_avg_pct` |
| Reproducibility | `device`, `seed`, `dtype`, `cpu_threads_used` |
| Machine spec | `machine_gpu`, `machine_gpu_count`, `machine_gpu_vram_gb`, `machine_driver`, `machine_cpu_physical_cores`, `machine_ram_gb`, `machine_os` |

Open it in pandas / Excel — one row per (checkpoint × mode × K × L × dataset).

### Reproducibility protocol — 1 eval per GPU

Our server has **4 GPUs and 96 CPU cores**.  To get comparable numbers across runs:

```bash
# Run each eval pinned to its own GPU — --device cuda:N is the only flag needed.
python eval.py --checkpoint ckpt_a --modes gbv --device cuda:0 &
python eval.py --checkpoint ckpt_b --modes gbv --device cuda:1 &
python eval.py --checkpoint ckpt_c --modes gbv --device cuda:2 &
python eval.py --checkpoint ckpt_d --modes gbv --device cuda:3 &
wait
```

**Rules — if you break any of these, your throughput numbers are not comparable:**

| Rule | Why |
|---|---|
| `--device cuda:N` | Pins PyTorch and pynvml telemetry to GPU N. `torch.cuda.set_device(N)` is called at startup so all ops go to that GPU. Run one process per GPU with different `--device` values to keep runs isolated. |
| `--seed 123` (default) | Fixes the RNG state before every mode sweep — same token sampling path |
| Prompts in file order, no shuffle | `load_prompts_jsonl()[:n]` is deterministic; never shuffle before eval |
| `--dtype bf16` (default) | Mixed precision changes numerics and throughput |
| Teacher loaded first, draft second | Already enforced in `load_models()` |
| `--cpu_threads` auto-detected | Default is `physical_cores // num_gpus` (96 / 4 = 24 on our server) — prevents CPU cache thrashing across 4 parallel evals |

**Interpreting GPU telemetry:**

- `gpu_sm_util_avg_pct` near 100 % → compute-bound (expected for large target model)
- `gpu_mem_util_avg_pct` near 100 % but SM low → HBM bandwidth-bound
- Both low → CPU / Python overhead dominates (check `time_draft_s` vs `time_target_s`)
- If throughput drops between runs while `gpu_vram_peak_mb` stays the same → HBM bandwidth contention, not VRAM pressure

**Note on L2 hit rate and exact HBM GB/s:** these require DCGM
(`sudo apt install datacenter-gpu-manager`).  The `gpu_mem_util_avg_pct` field
(from standard NVML) is the closest available proxy without DCGM.

---

## Diagnostics — is the loss even connected to the metric?

A standing question in this project is *why* every distillation loss tends to tie
the JSD baseline on block efficiency (BE).  Before tuning yet another loss, it is
worth checking whether the quantity the losses minimise (a teacher–draft
divergence) actually **predicts** BE at all.  That check is the `--diagnose` flag.

```bash
# Diagnostic mode — runs AFTER the timed eval; throughput is unaffected.
python eval.py --checkpoint checkpoints/<run>/ckpt_best \
               --dataset math_eval --modes traversal --diagnose
```

What it does, as a **separate pass after the timed loop**:

1. **Auto-reads the trained loss** from the checkpoint's `state.json`
   (`train_args.loss`).  All checkpoints saved after 2026-06-22 embed the full
   training arg set.  For older checkpoints, pass `--trained_loss jsd` or
   `--trained_loss fwdkl` to override.
2. For each prompt, computes the **trained divergence** (whichever loss the model
   was trained with — not hardcoded to JSD) and the other divergence as secondary,
   by rolling out the teacher greedily and forwarding both models.
3. Correlates divergence against BE using **Spearman ρ** as the primary statistic
   (rank-based, invariant to range compression) and Pearson r as secondary.
   Also tracks **σ(trained_loss)** alongside ρ to distinguish two failure modes:
   - **Case A — true signal loss:** σ(loss) stable, ρ drops → divergence has
     genuinely decoupled from BE; the objective is not the right lever.
   - **Case B — range restriction:** σ(loss) collapses (model uniformly
     better → narrower spread), ρ stable, Pearson r drops → the apparent
     weakening of r is a compression artefact, not signal loss.
4. Buckets prompts into **easy** (BE ≥ 0.75 L), **medium**, and **hard**
   (BE < 0.375 L) — thresholds are L-relative, not hardcoded absolute values.
5. Logs everything to a dedicated W&B run (`job_type=diagnose`).  Run name format:
   `diag_{training_run}_{ckpt_name}_{dataset}_{mode}` — includes the parent
   training-run directory so runs from different checkpoints are distinguishable.

**Reading the verdict:**

| Spearman ρ | σ(loss) | Meaning | Action |
|---|---|---|---|
| ρ ≈ 0  (p > 0.05) | any | Divergence does **not** predict BE rank → H0 not ruled out | Stop tuning divergence losses; the loss is not the lever |
| ρ < 0  (p < 0.05) | stable | Objective **is** connected to BE; rank order preserved | Keep tuning; the loss can move BE |
| ρ < 0  (p < 0.05) | collapsed | Range restriction: model improved, Pearson r artificially weak — ρ is the reliable signal | Likely fine; compare σ across checkpoints |
| ρ > 0 | — | Unexpected — verify divergence orientation before trusting | Sanity-check the implementation |

**W&B scalars** — all logged to `run.summary` (not `run.log`) so they appear as
numbers in the Overview panel rather than single-dot line charts:

| Key | Description |
|---|---|
| `diag/spearman_{loss}` | Spearman ρ — primary H0 signal |
| `diag/p_spearman_{loss}` | p-value for ρ (approximated via t-distribution) |
| `diag/pearson_{loss}` | Pearson r — secondary; shrinks under range compression |
| `diag/r2_{loss}` | R² for Pearson fit |
| `diag/std_{loss}`, `diag/mean_{loss}` | spread and mean of the trained divergence |
| `diag/mean_be`, `diag/std_be` | BE spread across all prompts |
| `diag/n_easy`, `diag/n_medium`, `diag/n_hard` | prompt counts by difficulty bucket |
| `diag/h0`, `diag/verdict` | plain-English H0 status and full interpretation |

**W&B tables** — logged via `run.log`:

| Key | Content |
|---|---|
| `diag/{loss}_vs_be` | scatter of trained divergence vs BE (trained loss only; secondary divergence omitted — different axis scale) |
| `diag/per_prompt` | 6-column table: `prompt_idx`, `prompt_preview`, `jsd`, `fwd_kl`, `block_eff`, `bucket`.  Filter by `bucket='hard'` in W&B UI to find struggling prompts. |
| `diag/bucket_summary` | per-bucket mean-BE and mean-divergence |

> **Throughput safety:** `--diagnose` is OFF by default.  The diagnostic pass runs
> strictly *after* the timed eval, so `throughput_tok_s` and all `time_*` columns
> are unaffected.  When the flag is off, eval does no extra work and never imports W&B.

### Decomposition stacked-bar + radar  (`--baseline_checkpoint`)

Add `--baseline_checkpoint` (typically `Qwen/Qwen3-0.6B`, the untrained base) to
get two additional plots that show **how much of the teacher–draft gap was closed**:

```bash
# requires: pip install matplotlib
python eval.py --checkpoint checkpoints/<run>/ckpt_best \
               --dataset math_eval --modes traversal --diagnose \
               --baseline_checkpoint Qwen/Qwen3-0.6B
```

What it computes (as a third pass, after `--diagnose`):

1. Loads the untrained base draft and runs `_prompt_divergence` on every prompt
   that had a valid diagnostic reading → produces **G₀** (baseline divergence per
   prompt, using the same divergence family the checkpoint was trained with).
2. From the `--diagnose` pass: **G** (trained divergence per prompt).
3. Learned = max(G₀ − G, 0);  Remaining = G.

**Stacked-bar** (two-panel figure, prompts sorted by G₀ easiest → hardest):

- Top panel: red = remaining JSD, green = gap closed, black step-line = G₀.  Tells you
  whether training reduced the gap uniformly or only on easy/hard prompts.
- Bottom panel: per-prompt BE.  Reading both panels together tells you whether
  high-divergence prompts also have low BE (they should if H0 is not the bottleneck).

**Radar** (polar chart):

- Splits prompts into 4 quartiles by G₀ (Q1=easy, Q4=hard), shows mean trained-BE in
  each quadrant, with an overall-mean reference ring.  Tells you where training helped
  most.

Both figures are logged as W&B images in a separate `job_type='decompose'` run.  Key
scalars logged: `decompose/mean_learned`, `decompose/mean_remaining`,
`decompose/frac_closed` (fraction of initial gap recovered), and
`decompose/be_q{1–4}_{easy,…,hard}`.

> **Interpretation:**  `frac_closed ≈ 0` → training changed the draft's distribution
> but the starting gap was already large (teacher is the bottleneck — try 32B).
> `frac_closed` high but BE still near baseline → the gap is closing but BE doesn't
> follow (H0 mismatch — the divergence metric is not the right lever).

---

## Inference (single prompt)

```bash
python inference.py --checkpoint checkpoints/gbv_tree/ckpt_best --mode gbv \
                    --prompt "What is 17 times 23?  Think step by step."
```

Prints the generated text plus block-efficiency / throughput for that one prompt.

---

## How K and L flow through the pipeline

Important distinction (subtle but matters for the paper):

* **train.py's `K` / `L`** — the draft tree size during training.  Affects
  three things depending on the loss:
  - *Flat losses (jsd, fwdkl, …):* K controls how many draft tokens the
    speculative decoding loop generates per training step.  Inert for the
    loss gradient itself, but **directly sets the distribution the draft
    learns from** — this is the core of enrichment training.  K=1 = one
    stochastic rollout step per update; K=3 = three steps, richer on-policy
    signal.
  - *Tree losses (bv_tree, gbv_tree, …):* K is additionally baked into the
    gradient via the α formula.  Higher K amplifies the verifier-alignment
    signal.
  - *Tree-depth experiments:* varying L (horizon) at fixed K changes how
    far ahead the tree looks; varying K at fixed L changes the branching
    factor.  Both are training-time hyperparameters and should be reported
    alongside eval K/L.
* **eval.py's `--K` / `--L`** — the draft tree size at inference time.
  Independent of training-time K/L; the two can differ.

For the cleanest "enrichment K scales" story, report the training K
alongside eval K so readers can distinguish in-distribution (train K = eval K)
from out-of-distribution generalisation (train K < eval K).
