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

# Sweep all 8 verifier modes for one checkpoint
python eval.py --checkpoint checkpoints/kl_tree/ckpt_best \
               --modes naive,nss,specinfer,spectr,khisti,bv,gbv,traversal

# Different K, L, or dataset
python eval.py --checkpoint checkpoints/gbv_tree/ckpt_best \
               --mode gbv --K 4 --L 8 --dataset math500

# Baseline (untrained Qwen3-0.6B)
python eval.py --checkpoint Qwen/Qwen3-0.6B --mode gbv
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

1. For each prompt, compute the per-prompt training divergence — **JSD** and
   **forward-KL** between teacher and draft (the same flat objective training
   minimises: teacher greedily rolls out, both models forwarded, divergence
   averaged over generated tokens).
2. Correlate that divergence against the prompt's **block efficiency**.
3. Log two scatter plots (`diag/jsd_vs_be`, `diag/fwdkl_vs_be`), the correlations
   (`diag/corr_jsd_be`, `diag/corr_fwdkl_be`), and a plain-English **verdict** to a
   dedicated W&B run (`job_type=diagnose`).

**Reading the verdict:**

| Correlation | Meaning | Action |
|---|---|---|
| `\|r\| ≈ 0` (flat cloud) | Divergence does **not** predict BE → objective mismatch | Stop tuning divergence losses; the loss is not the lever |
| `r < 0` (lower divergence ⇒ higher BE) | Objective **is** connected to BE | Keep tuning the loss; it can move BE |
| `r > 0` | Unexpected — verify orientation before trusting | Sanity-check the implementation |

> **Throughput safety:** `--diagnose` is OFF by default and **must not** be used on
> a throughput-measurement run if you want to be cautious — but note the diagnostic
> pass runs strictly *after* the timed eval and uses timers internal to
> `speculative_decoding_loop`, so `throughput_tok_s` and all `time_*` columns are
> mathematically unaffected either way.  When the flag is off, eval does no extra
> work and never imports W&B.

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

* **train.py's `K` / `L`** — the size of the draft tree used to compute the
  tree loss.  Set in the HARDCODED CONSTANTS block at the top of `train.py`.
  Inert for flat losses; baked into the gradient for verifier-aligned tree
  losses (the α formula has K in it).
* **eval.py's `--K` / `--L`** — the draft tree size at inference time.
  Independent of training-time K/L.

For the cleanest "tree > flat as K grows" story, set the training K to the
same value you eval at, OR run the full train-K × eval-K cross-grid.

---

## Losses dropped from this pipeline

* `ebe` and `ebe_single` (flat off-policy block-efficiency surrogates) are
  intentionally excluded — they optimise α on the teacher's own rollout,
  which is the wrong distribution.  The on-policy versions live in
  `losses/tree.py` (`bv_tree`, `gbv_tree`, etc.).
* `ebe_tree` is also excluded — it's redundant with `naive_tree` for K=1
  and an approximation for K>1.  Use `naive_tree` directly if you want the
  on-policy naive-verifier-aligned loss.
