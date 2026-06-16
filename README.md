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

# 4. (optional) GPU telemetry — pynvml + psutil (pip, one time)
#    eval.py uses these to record SM utilization, HBM-bus %, PCIe throughput,
#    and CPU utilization in every results.csv row.  Degrades gracefully if absent.
pip install pynvml psutil

# 5. (optional) log in to Weights & Biases for training curves
wandb login
```

That's it.  No multi-tier config, no provider-specific bootstrap.
This pipeline targets A100 (40 GB) but runs on any CUDA GPU with bfloat16 support.

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
│   ├── download.py      # fetch gsm8k / alpaca / math500 / humaneval / mtbench
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
```

What happens during training (per step):

1. Pick a prompt from `gsm8k_train.jsonl`.
2. **Flat loss** path: teacher generates `MAX_NEW_TOKENS` tokens, student
   forwards on the same sequence with grad, loss = divergence(student_logits,
   teacher_logits) on the generated portion.
3. **Tree loss** path: sample K student draft paths (no grad), score every
   tree node under the teacher (no grad), then re-score under the student
   with grad → `q_probs_dict`.  Loss = `−E[τ_V](q, p, K, L)`.
4. Gradient accumulation over `GRAD_ACCUM` micro-steps, AdamW step, linear
   warmup → constant LR.
5. Every `VAL_EVERY` steps: forward-KL on `gsm8k_val.jsonl`.  If improved,
   save `ckpt_best/`.  W&B logs `val/loss`.
6. Every `SAVE_EVERY` steps: write `ckpt_latest/` + `state.json` so
   `--resume` works.

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
# Run each eval pinned to its own GPU.  Use CUDA_VISIBLE_DEVICES to isolate
# the process at the OS level so no other job can land on the same GPU.
CUDA_VISIBLE_DEVICES=0 python eval.py --checkpoint ckpt_a --modes gbv --device cuda:0 &
CUDA_VISIBLE_DEVICES=1 python eval.py --checkpoint ckpt_b --modes gbv --device cuda:1 &
CUDA_VISIBLE_DEVICES=2 python eval.py --checkpoint ckpt_c --modes gbv --device cuda:2 &
CUDA_VISIBLE_DEVICES=3 python eval.py --checkpoint ckpt_d --modes gbv --device cuda:3 &
wait
```

**Rules — if you break any of these, your throughput numbers are not comparable:**

| Rule | Why |
|---|---|
| `CUDA_VISIBLE_DEVICES=N --device cuda:N` | Prevents a second process from landing on the same GPU and sharing HBM bandwidth |
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
