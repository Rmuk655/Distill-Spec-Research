# DistillSpec Pipeline — Qwen3 / A100

Simplified, single-GPU research pipeline for verifier-aligned distillation of
speculative-decoding draft models.  Three scripts:

| Script           | What it does                                          |
| ---------------- | ----------------------------------------------------- |
| `train.py`       | Distil a draft model against a teacher (default Qwen3-0.6B/Qwen3-8B; override with `--draft`/`--teacher`) |
| `eval.py`        | Block-efficiency + throughput on a held-out prompt set |
| `inference.py`   | Speculative decoding on a single user prompt          |

See [`project_report.md`](project_report.md) for what's been tried and what beat the baseline; this file is CLI reference only.

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

# 3c. (optional) OlympiadBench — harder than math_hard, EVAL-ONLY (no val/train
#     split — never checkpoint-selected or trained on). NOT fetched by --train.
python -m data_io.download --datasets olympiad_eval

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

The training loop is **one file** (`train.py`). Every experiment family below
is a `--loss` choice plus a small set of family-specific flags. Full mechanism
writeups and results for each family live in `notes/*_research_note.md`
(pointer table in [`project_report.md`](project_report.md)); this section is
only the CLI reference.

### Model pair

| Flag | Default | Meaning |
|---|---|---|
| `--draft` | `None` → falls back to `config.DRAFT_MODEL` (`Qwen/Qwen3-0.6B`) | Draft model id or local checkpoint path (warm start) |
| `--teacher` | `config.TEACHER_MODEL` (`Qwen/Qwen3-8B`) | Teacher/target model id |
| `--load_in_4bit` | off | Load the teacher in 4-bit NF4 (bitsandbytes) — needed for teachers too large for bf16 on one GPU (e.g. 32B) |

Always pass both `--draft` and `--teacher` explicitly when using a non-default
pair — omitting `--draft` silently falls back to the 0.6B default. A bigger
teacher only helps if the bottleneck is teacher-side; if it's draft capacity, a
bigger teacher widens the gap. Use `--diagnose` (below) to tell which regime
you're in before scaling up.

```bash
python train.py --loss jsd --draft Qwen/Qwen3-1.7B --teacher Qwen/Qwen3-32B \
                --load_in_4bit --train_dataset math_hard --val_dataset math_val
```

### Flat baselines

| `--loss` | Notes |
|---|---|
| `jsd` | Primary baseline — strongest flat divergence across verifiers |
| `forward_kl` | Secondary reference |
| `reverse_kl` | Weakest; degrades at high K |
| `l1` | Competitive with JSD on some verifiers, not on mean |

```bash
python train.py --loss jsd
```

### Tree losses (on-policy draft tree)

| `--loss` | α formula | Form | Val verifier |
|---|---|---|---|
| `naive_tree` / `naive_tree_full` | naive | product / exact-survival product | naive |
| `traversal_tree` | naive (proxy) | product | traversal |
| `traversal_log` | naive (proxy) | **log-space** (no depth vanishing) | traversal |
| `naive_log` | naive | log-space | naive |
| `nss_tree` / `nss_log` | exact NSS | product / log-space | nss |
| `kl_tree`, `rev_kl_tree`, `jsd_tree` | generic divergence, no verifier | — | traversal |
| `bv_tree`, `gbv_tree` | BV/GBV | product | **broken — do not use** (gradient collapses to 0) |
| `specinfer_tree`/`_log`, `spectr_tree`/`_log`, `khisti_tree`/`_log` | resp. verifier | product / log-space | resp. |
| `op_naive_tree` / `op_naive_tree_full` | naive | off-policy (teacher path, not draft samples) | naive |

```bash
python train.py --loss traversal_log --K 3 --L 8
python train.py --loss bv_tree --lr 1e-5   # DO NOT USE for real runs — see note
```

`--K` = draft paths per step, `--L` = tree depth. Both are read by every loss
in this table (unlike flat losses, where they're inert).

### Enrichment (`jsd_flat_enrich`, `jsd_enrich`)

Trains on **fresh stochastic teacher rollouts** instead of a fixed teacher-context
sequence — the only family that beat flat JSD. `jsd_flat_enrich` is the flat
(non-tree) form; `--K` is the number of stochastic teacher paths averaged per step
(called M in the research note).

```bash
python train.py --loss jsd_flat_enrich --K 3 --steps 8000 \
    --train_dataset math_hard --val_dataset math_val \
    --output checkpoints/jsd_flat_enrich_K3_mathhard_s123

# Warmer teacher temperature (more diverse paths — monitor train/path_diversity)
python train.py --loss jsd_flat_enrich --K 3 --teacher_temp 1.5 \
    --train_dataset math_hard --val_dataset math_val
```

`train/path_diversity` (console `pathdiv=`): fraction of token positions where
the K teacher rollouts disagree. `>0.15` = diverse signal; `<0.05` = paths
collapsing (raise `--teacher_temp` or K).

### Prefix-overlap (`prefix_overlap`)

Rewards the student's log-probability of teacher-sampled prefixes — Rahul's PO
estimator (`prob`) plus three approximations designed to fix its practical
failure modes (see [`prefix_overlap_research_note.md`](notes/prefix_overlap_research_note.md)).

| Flag | Default | Meaning |
|---|---|---|
| `--prefix_objective` | `prob` | `prob` (exact, collapses in practice) / `logprob` (log-space) / `traversal` (K-aware reweight) / `nss` (masked CE) |
| `--prefix_M` | 4 | Teacher continuations per root (single-root mode) |
| `--prefix_root_spacing` | 0 | N > 0 → multi-root every N tokens (0 = single-root) |
| `--prefix_rollout_len` | `MAX_NEW_TOKENS` | Teacher rollout length |
| `--prefix_random_offset` | off | Randomise root position for unbiased coverage (Rahul §5) |
| `--prefix_aux` | `ce` | Secondary anchor term: `ce` or `jsd` |
| `--prefix_aux_weight` | 0.0 | λ for the secondary term |
| `--prefix_anneal_steps` | 0 | Anneal λ linearly from 1→0 over N steps (stable CE anchor early, pure PO gradient later) |
| `--prefix_min_root` | 0 | Minimum token index before placing a root |

```bash
# Warm-start from a JSD checkpoint (the mechanism only makes sense post-JSD)
python train.py --loss prefix_overlap --prefix_objective nss \
    --prefix_root_spacing 4 --L 8 --draft checkpoints/jsd/ckpt_best \
    --train_dataset math_hard --val_dataset math_val
```

### Combining a flat backbone with a tree aux loss

| Mode | Flags | Mechanism |
|---|---|---|
| Additive | `--aux_loss <tree_loss> --aux_weight λ` | `total = primary + λ·aux` — real per-level gradient through q |
| Depth-weight (linear) | `--aux_mode depth_weight --aux_loss <tree_loss> --depth_linear` | `w = d / EMA(d)` multiplies the flat loss; `d = E[τ_V]` of the draft tree |
| Depth-weight (exponential) | `--aux_mode depth_weight --aux_loss <tree_loss> --depth_lambda λ` | `w = exp(λ(d − EMA(d)))`; `λ=0` is a no-op control (must reproduce plain `jsd`) |

Both depth-weight forms normalise `E[w]≈1` — no hidden LR change. `train/depth_w`
(applied weight) and `train/depth_d` (raw `E[τ_V]`) are logged to W&B.

```bash
python train.py --loss jsd --aux_loss naive_tree --aux_weight 0.1
python train.py --loss jsd --aux_mode depth_weight --aux_loss naive_tree --depth_linear
```

### Early stopping / minimum run length

| Flag | Default | Meaning |
|---|---|---|
| `--early_stop_patience` | 15 | Stop if smoothed val BE hasn't improved for this many val checks (0 = disabled) |
| `--early_stop_min_delta` | 0.0 | Minimum improvement to reset the patience counter |
| `--min_steps` | 0 | Patience early-stop cannot fire before this step, regardless of patience |
| `--divergence_abort_frac` | 0.0 | Abort immediately (even below `--min_steps`) if smoothed val BE falls more than this fraction below the best seen (e.g. `0.20` = abort on a 20% collapse) |

```bash
# Always run the full 15K steps; only abort early on a genuine collapse
python train.py --loss jsd --steps 15000 --min_steps 15000 \
    --divergence_abort_frac 0.20 --early_stop_patience 15
```

### Other common flags

| Flag | Meaning |
|---|---|
| `--train_dataset` / `--val_dataset` | e.g. `gsm8k_train`/`gsm8k_val` or `math_hard`/`math_val` |
| `--val_temp` (default 0.2) | Val block-efficiency decode temperature — kept low so the curve is readable; training temp (0.8–1.0) makes val BE swing ±0.4 from sampling noise alone. Cannot be 0. |
| `--resume` | Resume from `ckpt_latest/` + `state.json` after a kill/crash |
| `--seed` | RNG seed |
| `--teacher_temp` / `--draft_temp` | Sampling temperature for teacher/draft rollouts (enrichment, tree losses) |
| `--no_wandb` | Disable W&B logging |

```bash
python train.py --loss kl_tree --resume
python train.py --loss jsd --train_dataset math_hard --val_dataset math_val --steps 2000
```

### What happens during training (per step)

1. Pick a prompt from the train dataset.
2. **Flat / enrichment loss:** teacher generates (greedily for flat, stochastically ×K for enrichment); student forwards on the same sequence with grad; loss = divergence on the generated portion.
3. **Tree loss:** sample K student draft paths (no grad), score every tree node under the teacher (no grad), re-score under the student with grad. Loss = `−E[τ_V](q, p, K, L)` (product or log-space).
4. **Prefix-overlap:** teacher samples M continuations from one or more root positions; loss rewards the student's log-probability of the sampled prefixes per `--prefix_objective`.
5. Gradient accumulation over `GRAD_ACCUM` micro-steps, AdamW step, linear warmup → cosine decay.
6. Every `VAL_EVERY` steps: decode block-efficiency on the val set (verifier matched to the loss, at `--val_temp`). If improved, save `ckpt_best/`.
7. Every `SAVE_EVERY` steps: write `ckpt_latest/` + `state.json` so `--resume` works.

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

# Non-default model pair — --teacher must match how the checkpoint was trained
# (defaults to config.TEACHER_MODEL / Qwen3-8B if omitted; wrong teacher here
# silently gives a meaningless block-efficiency number)
python eval.py --checkpoint checkpoints/qwen17b_qwen32b_run/ckpt_best \
    --teacher Qwen/Qwen3-32B --mode traversal --dataset math_eval

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
| Reproducibility | `device`, `seed`, `dtype`, `cpu_threads_used`, `teacher_model` |
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
