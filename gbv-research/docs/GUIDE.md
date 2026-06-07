# SpecDist Experiment — Research Guide

**Primary result**: Qwen3-0.6B → Qwen3-8B (A100, GSM8K) — main paper result.  
**Cross-family result**: LLaMA-3.2-1B → LLaMA-3.2-3B (A100, GSM8K) — proves algorithm is family-agnostic.  
**Laptop smoke**: Qwen toy pair (crash-check) + LLaMA 1B→3B NF4 (directional trends).  
**CPU only**: GPT-2 (distilgpt2→gpt2-medium) — kept for WikiText-2 convergence verification; not a paper result.

**Goal**: Train the draft model with novel tree-aligned loss functions so it gets accepted more often by the target verifier (GBV, BV, traversal, etc.), speeding up generation without changing what the target produces.

> **Doc status — June 2026**: updated for LLaMA cross-family paper result, GPT-2 demoted to CPU-only convergence check, multi-family configs, `max_train_prompts` epoch-based training budget, A10 as Stage 1.5 exploration tier, BF16 vs NF4 convergence clarification, IITH A100 pricing, Modal credit change, W&B loss curve interpretation.

---

## Table of Contents

0. [The Three-Stage Experiment Sequence](#0-the-three-stage-experiment-sequence)
1. [Research Hypotheses](#1-research-hypotheses)
2. [Verifier Hierarchy and What We Need to Prove](#2-verifier-hierarchy-and-what-we-need-to-prove)
3. [Minimal Experiment Matrix for Paper](#3-minimal-experiment-matrix-for-paper)
4. [Key Metrics](#4-key-metrics)
5. [Pipeline Phases (by tier)](#5-pipeline-phases-by-tier)
6. [Training Losses](#6-training-losses)
7. [Dashboard](#7-dashboard)
8. [How to Run](#8-how-to-run)
   - 8a. [Checkpoint Progression Analysis](#8a-analysing-training-progress-with-checkpoints)
   - 8b. [Server / Colab / Modal Training](#8b-running-training-on-a-server-colab--modal)
   - 8c. [New Environment Setup (scripts/setup_download.py)](#8c-setting-up-a-new-environment-scriptssetup_downloadpy)
   - 8d. [Results Report (analyze_results.py)](#8d-generating-a-results-report-analyze_resultspy)
   - 8e. [EAGLE Benchmark](#8e-eagle-benchmark)
9. [Automatic Training Health Checks](#9-automatic-training-health-checks)
10. [Weights & Biases Integration](#10-weights--biases-integration)
11. [Quick Interpretation Cheatsheet](#11-quick-interpretation-cheatsheet)
12. [KL Divergence Comparison](#12-kl-divergence-comparison)
13. [Online Speculative Decoding (OSD)](#13-online-speculative-decoding-osd)

---

## Model Family Strategy

With A100 access, the model family choices are settled. Reference this table before choosing a config.

| Family | Pair | Dataset | Hardware | Role | Paper result? |
|--------|------|---------|----------|------|---------------|
| **Qwen3** | Qwen3-0.6B → Qwen3-8B | GSM8K (7473 train, 1319 test) | A100 40GB BF16 | **Primary paper result** | ✅ Yes |
| **LLaMA 3.2** | Llama-3.2-1B → Llama-3.2-3B | GSM8K (same benchmark) | A100 40GB BF16 | **Cross-family paper result** | ✅ Yes |
| **LLaMA 3.2** | Llama-3.2-1B → Llama-3.2-3B NF4 | GSM8K | Laptop 6GB | Directional trends (3× gap, instruction-tuned) | Directional only |
| **GPT-2** | distilgpt2 → gpt2-medium | WikiText-2 (in-distribution) | CPU server | Convergence verification only | ❌ No |
| **Qwen3** | Qwen2.5-0.5B → Qwen3-0.6B | — | Laptop | Code crash-check only | ❌ No |

**Why LLaMA and not GPT-2 for cross-family?**

| Criterion | GPT-2 (distilgpt2 → gpt2-medium) | LLaMA 3.2 (1B → 3B) |
|-----------|-----------------------------------|----------------------|
| Year | 2019 | 2024 |
| Instruction-tuned | ❌ No | ✅ Yes |
| Same benchmark (GSM8K) | ❌ OOD — needs WikiText-2 | ✅ In-distribution |
| Reviewer credibility | Low (legacy) | High (current SOTA family) |
| Architecture difference from Qwen | Minimal impact | Strong: GQA, SwiGLU, RoPE, different tokenizer |
| Purpose with A100 access | CPU-only backup (obsolete) | Cross-family paper result |

**Bottom line**: Now that A100 is available, GPT-2 is retired as a paper family. Run `laptop_llama` for laptop convergence checks, `a100_llama` for the cross-family paper numbers. GPT-2 configs remain for anyone wanting WikiText-2 CPU convergence verification.

---

## 0. The Three-Stage Experiment Sequence

The experiment runs in four hardware tiers, each with a specific purpose. **Never skip a tier**; each stage gates the next.

### Stage 0 — Laptop Smoke (code correctness)

**Hardware**: Any laptop with ≥ 4 GB VRAM.  
**Config**: `laptop_qwen` (Qwen toy pair) — fastest crash-test.  
**Models**: Qwen2.5-0.5B draft → Qwen3-0.6B target.  
**hw_tier tag in results.db**: `laptop`

**Purpose**: Verify every code path runs without crash or OOM. Results are **not paper-quality** (teacher ≈ draft size — zero distillation signal). Do not interpret numbers.

| Parameter | Value |
|---|---|
| Train steps | 100 per loss |
| Eval prompts | 5 (n=5) |
| Losses | ALL flat + tree losses |
| Verifiers | ALL: traversal, specinfer, gbv, naive, bv |
| K | 3 |
| Temperature | 0.6 |

```bash
# Fastest crash-test (~25 min, 10 steps per loss)
python orchestration/experiment.py --config laptop_qwen --smoke --yes

# Full code-path exerciser (~2-3 hr, 100 steps per loss)
python orchestration/experiment.py --config laptop_qwen --yes

# Alternate families (same laptop hardware, more signal):
python orchestration/experiment.py --config laptop_gpt2  --yes  # GPT-2, 4.3x size gap
python orchestration/experiment.py --config laptop_llama --yes  # LLaMA 1B→3B, 3x gap
```

**Decision gate**: if any step crashes or OOMs, fix it before proceeding. If all steps pass, proceed to Stage 0.5 or Stage 1.

---

### Stage 0.5 — CPU Convergence (GPT-2, ATS Cloud server)

**Hardware**: ATS Cloud — 128 GB RAM, CPU-only, no GPU.  
**Config**: `server_gpt2` (distilgpt2 → gpt2-medium).  
**hw_tier tag in results.db**: `cpu`

**Purpose**: Prove the distillation algorithm converges without GPU when T4 credits are exhausted. The GPT-2 family (4.3× size gap, vocab=50257) produces a real distillation signal on CPU in 1–2 days. This is **not** paper-quality (wrong family, too small) but proves the training loop and loss gradient are correct before committing GPU budget.

> **When to use this tier**: Kaggle/Colab credits are spent and you cannot access T4 until the next week's quota resets. Stage 0.5 keeps the research moving.

| Parameter | Value |
|---|---|
| Train steps | 1,000 per loss |
| Eval prompts | 5 (n=5, CPU-capped) |
| Losses | kl, rev_kl, jsd, l1 + all tree variants (ebe/online excluded) |
| Verifiers | alpha, bv, gbv, traversal, specinfer, naive |
| OMP threads | 16 (set in `bases/server.yaml`; tune to core count) |

```bash
# Sequential — experiment.py handles everything (~30-50 hr):
python orchestration/experiment.py --config server_gpt2 --yes

# Parallel — all losses at once using 128 GB RAM (~4-6 hr):
# (See server_gpt2.yaml Option B comment for the shell loop)
```

**Decision gate**: if loss curves decrease monotonically and acceptance rate > 0 at step 100, the algorithm is working. Proceed to Stage 1 (Qwen, 8B teacher) for research-quality trends.

---

### Stage 1 — Colab/Kaggle T4 (trend formation)

**Hardware**: Colab free T4 (15 GB) or Kaggle T4 (16 GB, 29 GB RAM).  
**Config**: `colab` (Qwen3-0.6B → Qwen3-4B BF16) or `kaggle` (Qwen3-0.6B → Qwen3-8B NF4).  
**Models**: Qwen3-0.6B draft → Qwen3-4B/8B teacher — **real distillation signal** (4–13× size gap).  
**Eval sets**: `gsm8k_30` (Phase 3 primary) + `alpaca_30`, `math500_30`, `humaneval`, `mtbench_80` (Phase 4 generalization) — all committed to repo, no download needed.  
**hw_tier tag in results.db**: `colab` / `kaggle`

**Purpose**: Directional signal — does EBE beat forward KL? Do the trends hold across domains? Alpha and block_eff are **directionally meaningful** at n=30. Timing/throughput are not reliable (quantized target on Kaggle changes latency).

| Parameter | Colab | Kaggle |
|---|---|---|
| Train steps | 500 | 1000 |
| Eval prompts | **n=30** per dataset | **n=30** per dataset |
| Primary eval | gsm8k_30 | gsm8k_30 |
| Generalization eval | alpaca_30, math500_30, humaneval | alpaca_30, math500_30, humaneval |
| Losses | all enabled in experiment.losses | all enabled |
| K | 3 | 3 |
| Temperature | 0.8 | 0.8 |

```bash
python orchestration/experiment.py --config colab --yes    # Colab
python orchestration/experiment.py --config kaggle --yes   # Kaggle
```

**Decision gate**: if EBE shows higher alpha/BE than forward_kl consistently across gsm8k and ≥1 secondary domain at n=30, proceed to Stage 1.5 or Stage 2.

---

### Stage 1.5 — A10 (stronger convergence, free access)

**Hardware**: NVIDIA A10 (24 GB VRAM) — free access where available.  
**Config**: `a10_qwen` (Qwen3-0.6B → Qwen3-8B **BF16**, no quantization).  
**Key advantage over T4**: A10's 24 GB fits Qwen3-8B in full BF16 — no NF4 quantization.

> **BF16 vs NF4 (NF4 = Kaggle T4 config):** Same convergence direction and loss rankings. BF16 gives ~5-10% cleaner gradients (less rounding noise in teacher logits), so trends appear slightly faster. NF4 Kaggle results reliably predict BF16 A10/A100 results — they are directionally interchangeable.

| Parameter | Value |
|---|---|
| Train steps | 2000 |
| max_train_prompts | 1000 — 2 full epochs, clear convergence |
| Teacher | Qwen3-8B **BF16** (exact, no quantization) |
| LoRA rank | r=8 (vs r=4 on colab_lite) |
| Speed | ~2-3 s/step → 2000 steps ≈ 80-100 min |
| Eval prompts | n=50 (trend direction) |

```bash
python orchestration/experiment.py --config a10_qwen --yes \
  --storage_root /path/to/storage --losses kl_tree
```

**Decision gate**: 2 epochs over 1000 prompts with 8B BF16 teacher should show clear monotonic loss decrease and loss ranking (e.g. `kl_tree` < `forward_kl` LoRA loss). If ranking is stable, promote to Stage 2 for paper numbers.

---

### Stage 2 — A100 (exploration → paper confirmation)

**Hardware**: A100 A100-SXM4-40GB (A100). Shared 8-GPU node — only GPU 0 is allocated to your job unless `CUDA_VISIBLE_DEVICES` is set by the cluster.

**Two A100 configs — run both for the paper:**

| Config | Models | Role |
|--------|--------|------|
| `a100_qwen` | Qwen3-0.6B → Qwen3-8B BF16 | **Primary result** — main paper table |
| `a100_llama` | LLaMA-3.2-1B → LLaMA-3.2-3B BF16 | **Cross-family result** — proves algorithm is family-agnostic |

> **LLaMA access**: HuggingFace gated model. Accept licence at  
> `https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct` then `huggingface-cli login`.

---

#### Stage 2A — Exploration (current default)

**Purpose**: Rank all 15 losses quickly (ebe/ebe_single excluded — off-policy, not the paper contribution). Pick top 4-5 winners before committing to full eval.

| Parameter | Value | Rationale |
|---|---|---|
| Train steps | 2000 | ~15-20 min/loss |
| grad_accum | 16 | 125 optimizer steps; lower variance than 4 |
| grad_clip | **5.0** (tree losses) | Tree losses have path-weight amplification → norms 30-100 are normal |
| max_train_prompts | null — full 7473 | Diversity > epochs for paper quality |
| val_every | **50** | 40 val checks → smooth trend curves (not 200 → only 10 points) |
| n_val (train) | **30 prompts** | SE≈0.031, 45s/check × 40 = 30 min total — doesn't dominate training |
| val generation | **greedy** (do_sample=False) | Deterministic reference — sampling caused 5% noise masking real trends |
| GSM8K eval (n) | **100** | SE≈0.017; ~40 min/eval |
| Verifier modes | alpha, bv, gbv, traversal (4 of 9) | ~40 min/eval vs 14h for all 9 |
| milestone_every | **200** | 10 checkpoints → smooth convergence curves for paper figures |
| **Total: 15 losses** | **~11 hours** (train + eval) | |

```bash
# Activate venv first (ALWAYS — running without venv gets killed by system Python 3.12):
source ~/.specdist_env

# Full exploration pipeline:
python deploy/aip_run.py --config a100_qwen
# OR one loss at a time (safer on shared cluster — 2h session limit):
python deploy/aip_run.py --config a100_qwen --losses kl_tree --train_only --no_smoke
```

**Decision gate**: rank by `BE/gbv` and `alpha/gsm8k` in W&B runs table. Top 4-5 → Stage 2B.

---

#### Stage 2B — Paper confirmation (flip two lines in YAML)

In `a100_qwen.yaml` `evaluation:` section:
```yaml
evaluation:
  n_prompts_gsm8k: 1319          # was 100
  modes: [alpha, naive, nss, specinfer, spectr, khisti, bv, gbv, traversal]  # all 9
```
Also override val settings for paper quality:
```yaml
dataset:
  val_dataset: gsm8k_100.jsonl   # was gsm8k_30
health:
  val_every: 100                  # was 50 — 20 checks with n=100 = 50 min total
```

Then rerun only winners:
```bash
bash deploy/rerun_loss.sh gbv_tree paper_run
bash deploy/rerun_loss.sh traversal_tree paper_run
```

| Parameter | Value |
|---|---|
| GSM8K eval (n) | **1319** (full test set) |
| Verifier modes | all 9 |
| Time per eval | ~14-17 hours |
| **Top 5 losses** | **~70-85 hours total** |

> Run Stage 2B only on 4-5 winners. Running all 15 losses at n=1319 = ~300 hours.

---

#### Convergence curves for paper figures

Paper requires alpha/BE vs training step (not just final numbers). Infrastructure is in place:

```bash
# After training finishes, sweep milestone checkpoints (n=20 prompts, ~5 min/checkpoint):
python deploy/eval_checkpoints.py \
    --config a100_qwen --loss gbv_tree \
    --storage_root "$GBV_DIR/db" \
    --n 20 --modes "alpha,gbv"

# Generates checkpoint_evals table in DB + W&B metrics:
#   checkpoint_eval/alpha_alpha  vs train_step
#   checkpoint_eval/gbv_be       vs train_step
```

With `milestone_every=200` (10 milestones over 2000 steps), each eval_checkpoints run takes ~50 min and produces 10 data points — enough for a smooth convergence figure.

---

#### Storage layout (A100 server)

See [deploy/A100_SETUP.md](../deploy/A100_SETUP.md) for the canonical layout. Summary:

| Data | Location | Size (typical) |
|------|----------|----------------|
| Checkpoints, merged models, `results.db`, logs | `$GBV_DIR/db/` | ~130 GB |
| HF model cache | `$HOME/specdist/hf_cache/` | ~18 GB |
| Python venv, W&B local files | `$HOME/specdist/` | ~13 GB |

> **Do NOT copy HF model files manually** — they download on first use and are cached in `hf_cache/`.

#### One-time setup

```bash
export WANDB_API_KEY="your-key-from-wandb.ai/authorize"   # REQUIRED before setup
git clone https://github.com/Rmuk655/Distill-Spec-Research.git ~/Distill-Spec-Research
bash ~/Distill-Spec-Research/gbv-research/deploy/aip_gpu_setup.sh a100_qwen
```

The setup script handles everything:
1. Sets storage root to `$GBV_DIR/db` (checkpoints + results.db)
2. Creates venv at `$HOME/specdist/venv`
3. Installs all ML deps (torch cu128, transformers 5.x, peft, bitsandbytes, wandb)
4. Downloads `gsm8k_train.jsonl` (7473 prompts) before offline flags are set
5. Writes `~/.specdist_env` with all env vars including `WANDB_API_KEY` and `STORAGE_ROOT`
6. Adds `source ~/.specdist_env` to `~/.bashrc` — persistent across sessions
7. Sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

> **Always `source ~/.specdist_env` before running** — or `aip_run.py` auto-detects system Python and re-executes itself under the venv.

---

#### Running the pipeline

```bash
source ~/.specdist_env   # if new terminal
cd ~/Distill-Spec-Research/gbv-research

# Full exploration (all 15 losses):
python deploy/aip_run.py --config a100_qwen

# Resume after SIGTERM (session expired):
python deploy/aip_run.py --config a100_qwen --resume

# Single-loss targeted run (safe to run in parallel — uses per-loss lock):
python orchestration/experiment.py --config a100_qwen --losses kl_tree --train_only --yes
python orchestration/experiment.py --config a100_qwen --losses gbv_tree --train_only --yes
# ↑ These two can run simultaneously without killing each other

# Eval-only for specific losses (one at a time — 2h session limit):
python orchestration/experiment.py --config a100_qwen --losses kl --eval_only --yes
python orchestration/experiment.py --config a100_qwen --losses rev_kl --eval_only --yes

# Restart a specific loss:
bash deploy/rerun_loss.sh kl_tree my_tag
```

**Monitor:**
```bash
tail -f "$GBV_DIR/db/logs/a100_qwen-q0.6b-q8b/pipeline_output.log"
# W&B: https://wandb.ai/rmukund16-indian-institute-of-technology-hyderabad/distillspec
# Key W&B columns: BE/gbv, BE/traversal, alpha/gsm8k
```

---

#### Shared-cluster operational notes (A100)

**Session time limit (~2h)**: each GPU session expires. Long evals get SIGTERMed.  
**Fix**: run ONE loss per session for eval. Training is fast (~20 min); eval is the bottleneck.

```bash
# Wrong — runs two evals in parallel, both get killed when session expires:
python experiment.py --config a100_qwen --losses kl,rev_kl --eval_only  # ← DON'T

# Right — one per session, each finishes in ~40 min:
python experiment.py --config a100_qwen --losses kl --eval_only --yes
```

**GPU allocation**: on the 8-GPU node, only GPU 0 is yours unless `CUDA_VISIBLE_DEVICES` is set.  
The scheduler auto-detects this and restricts to GPU 0.  
To use all 8 GPUs (dedicated machine only): `export SPECDIST_ALL_GPUS=1`

**Parallel targeted training** (safe): per-loss lock files prevent `--train_only` runs from killing each other. Running kl_tree and gbv_tree simultaneously is fine; their lock files are separate.

---

**Verified on A100-SXM4-40GB (June 2026):**

| Check | Status |
|---|---|
| 2 eval slots on 40GB | ✅ EVAL_VRAM_GB=19 → 2×19=38 GB |
| W&B from trainer subprocesses | ✅ WANDB_API_KEY in ~/.specdist_env |
| specInfer on transformers 5.x | ✅ bundled `algorithms/specInfer/` supports DynamicCache `.layers` API |
| draft-verify alpha fallback | ✅ used only when specInfer unavailable (CPU / import failure) |
| BE batch timeout | ✅ scales with n_prompts × combos (was hardcoded 2h) |
| tqdm verbosity | ✅ mininterval=60s (one line/min in logs) |
| HF loading bars | ✅ suppressed in trainer.py, evaluate.py, runner.py |
| task_score CPU fallback prevention | ✅ reuses preloaded alpha model — no second 8B load |
| task_score batch size | ✅ from YAML `evaluation.task_batch` (a100=32, laptop=4) |
| val loss determinism | ✅ greedy generation (was do_sample=True — 5% noise) |
| grad_accum from YAML | ✅ wired to train_hparams (was silently defaulting to 4) |
| Per-loss lock for --train_only | ✅ parallel single-loss runs coexist |
| GPU 0 only on shared cluster | ✅ auto-detected when CUDA_VISIBLE_DEVICES unset |
| Auto venv relaunch | ✅ aip_run.py re-execs under venv if run from system Python |
| .pyc cache cleared at startup | ✅ aip_run.py clears __pycache__ before launch |

**Known issues fixed (June 2026):**
- **grad_accum not reaching trainer**: was silently defaulting to 4 (125→500 optimizer steps). Fixed: wired to `train_hparams` dict.
- **task_score CPU fallback (~7 hours)**: loaded second copy of 8B model while alpha models were resident → OOM → CPU. Fixed: reuses preloaded alpha student. Now prints a loud `[WARNING]` banner if CPU fallback occurs.
- **specInfer DynamicCache on transformers 5.x**: `key_cache` removed; fixed via `.layers[].keys` cropping in bundled specInfer.
- **BE batch timeout (2h hardcoded)**: killed 14h eval at 9%. Fixed: scales with prompts × combos.
- **convergence alert miscalibrated**: fired at grad-step 200 (12 optimizer updates = peak LR). Fixed: fires at 30% of optimizer steps. Tree losses get threshold 50 (not 10) — path weights naturally amplify norms.
- **SIGTERM from using all 8 GPUs**: scheduler claimed all 8 A100s on shared node. Fixed: defaults to GPU 0 when CUDA_VISIBLE_DEVICES unset.
- **parallel runs killing each other**: --losses X --train_only used config-level lock → killed sibling loss runs. Fixed: per-loss lock files.

**Decision gate**: after Stage 2A, rank losses using the **loss × verifier matrix** from `python db/analyze_results.py` (not a single `gbv` column — best BE may be `rev_kl + traversal` while `jsd + bv` wins elsewhere). Losses that beat baseline BE under their best verifier are candidates for Stage 2B.

---

## 1. Research Hypotheses

The paper tests four hypotheses. Every experiment should be designed to confirm or refute one or more of them.

### H1 — Traversal verifier > SpecInfer (primary claim)

**Prediction**: traversal block efficiency > specinfer block efficiency, consistently across K=3 and K=5, across gsm8k and math500, at both T=0.6 and T=1.0.

**What "consistently" means**: traversal wins in ≥ 6/8 cells of the {K=3,5} × {gsm8k, math500} × {T=0.6,1.0} grid. A single win at K=5/T=0.6 is not sufficient.

**Rationale**: Traversal uses bottom-up acceptance which avoids wasting tree budget at shallow nodes (where p≈q and marginal gains are small). SpecInfer uses top-down prefix acceptance which discards valid non-prefix subtrees.

---

### H2 — EBE loss > forward KL (training claim)

**Prediction**: A draft trained with EBE loss achieves higher block efficiency under traversal verifier than a draft trained with forward KL, given the same number of training steps and learning rate.

**What to measure**: BE_traversal(EBE) − BE_traversal(KL) at K=3, K=5, both temperatures, both datasets.

**Rationale**: EBE directly optimises E[τ+1] = 1 + Σ_k Π_{i≤k} α_i, which is exactly block efficiency. Forward KL minimises KL at every token position independently without caring about the sequential acceptance product.

---

### H3 — Online OSD beats offline best

**Prediction**: Starting from the best offline-trained checkpoint (EBE, lr=1e-4), online OSD adaptation on incoming GSM8K queries raises traversal block efficiency by ≥0.15 tokens/call vs the offline checkpoint alone.

**Rationale**: offline training minimises KL at every position; online training targets only rejection positions as they appear in live inference — a signal directly tied to what the verifier rejects, not a proxy loss.

**Dataset dependence**: online results are distribution-matched to the adaptation dataset. Run separate online experiments for each eval dataset if you want cross-dataset claims.

**What would refute H3**: online eval_alpha at step 500 ≤ offline eval_alpha (indicates online training is not providing useful signal over the offline baseline).

---

### H4 — Improvements generalise across datasets

**Prediction**: EBE > KL ordering holds on both gsm8k and math500, at both temperatures. The gain on math500 is within 50% of the gain on gsm8k (shows the improvement is not GSM8K-specific).

**What to measure**: Phase 4 multi-dataset eval. Compare BE gain (%) over baseline for EBE vs KL, across all datasets.

---

### H5 — Tree losses beat their flat counterparts

**Prediction**: When trained on the student's own K-path draft tree (on-policy), tree-structured
losses outperform their flat equivalents under non-OT verifiers (bv, gbv, traversal).
Specifically:

- `kl_tree` BE > `forward_kl` BE under gbv/traversal
- `ebe_tree` BE > flat `ebe` BE under bv/gbv/traversal
- Among tree losses, the best loss matches its paired verifier: `bv_tree`→bv,
  `gbv_tree`→gbv, `traversal_tree`→traversal

**What "beats" means**: consistent improvement of ≥ 0.1 BE averaged over K=3 and K=5,
both temperatures.

**Rationale**: Flat losses train on teacher-generated tokens — the teacher's own draft, not
the student's.  At evaluation time the verifier scores the student's tokens.  This
off-policy mismatch means the gradient signal never touches the distribution the verifier
actually sees.  Tree losses train on the student's own K-path tree, so the training
distribution exactly matches the inference distribution.  In addition, verifier-specific
tree losses (bv_tree, gbv_tree, traversal_tree) optimise the exact acceptance integral that
each verifier computes, rather than a proxy divergence.

**Key ablation** (H5a): `ebe_tree` > flat `ebe` isolates the off-policy mismatch effect
alone (same formula, different training distribution).

**What would refute H5**: tree losses performing ≤ flat losses across ≥ 4 of 6
(K × temperature) cells, ruling out run-to-run variance.

---

### H6 — Online tree training beats flat online

**Prediction**: `online_kl_tree` adaptation (trains on a fresh K-path draft tree at each
update interval) achieves higher block efficiency than flat `online` adaptation under
gbv and traversal verifiers, with the gap widening over adaptation steps.

**Rationale**: Flat online training uses a replay buffer of
`(accepted+residual tokens, teacher_logits, wrong_mask)` tuples.  The buffer is
collected under the teacher's sampling, not the student's, so the on-policy correction
that tree training provides also applies to the online setting.  Additionally, training
on a structured tree allows the loss to see K simultaneous candidate paths per prompt
rather than a single linear rollout, providing a richer gradient signal per update.

**What to measure**: Steps 0–500 of `online_kl_tree` adaptation vs `online` adaptation,
tracked as rolling block efficiency under gbv at K=3, T=0.8.

**What would refute H6**: `online_kl_tree` showing no improvement over `online` at step
500 under gbv, after controlling for total compute (each tree update costs K× more than a
flat update).

---

## 2. Verifier Hierarchy and What We Need to Prove

The verifier is the algorithm that decides, given a draft tree of K paths each of length L, which prefix to accept.

### Expected ranking (Thomas et al. 2026, arXiv:2602.16994v1)

```
traversal (5.31) > spectr (4.61) ≈ specinfer (4.58) > bv (4.30) > nss (4.05)
```

For our paper, the relevant ordering is:

```
traversal ≥ gbv > specinfer > bv > naive
```

| Verifier | Role in paper | Run tier |
|---|---|---|
| `traversal` | **Primary claim** — must beat specinfer | colab + a100 |
| `gbv` | Secondary claim — shows GBV also benefits | colab + a100 |
| `specinfer` | **Key comparison** for H1 | colab + a100 |
| `bv` | Lower bound (single-path) | a100 only |
| `naive` | Lower bound (token-level) | laptop smoke only |
| `nss`, `spectr`, `khisti` | Research curiosity, exercise only | laptop smoke only |

**Why naive and bv must be included**: They serve as lower bounds. A paper that only shows EBE beats KL under traversal but doesn't also show the ordering traversal > bv > naive is incomplete — reviewers will ask.

### If traversal doesn't consistently beat specinfer

Do NOT quietly drop the comparison. Report it in a conditional framing:

> "Traversal outperforms specinfer when alpha ≥ X (well-trained draft); below that threshold both methods are equivalent."

This is itself a useful finding: it tells practitioners when traversal is worth the extra verification compute.

The threshold X is estimated from the data: plot BE_traversal − BE_specinfer vs alpha_mean, fit a piecewise linear model. If the crossover point is at alpha ≈ 0.6, report that.

---

## 3. Minimal Experiment Matrix for Paper

36 eval runs are required for the paper. Each row is one (model × dataset × verifier × K × temperature) combination.

| # | Model | Dataset | Verifier | K | T | Hypothesis |
|---|---|---|---|---|---|---|
| 1–4 | baseline | gsm8k | traversal, specinfer | 3, 5 | 0.6, 1.0 | H1 reference |
| 5–8 | kl1000 | gsm8k | traversal, specinfer | 3, 5 | 0.6, 1.0 | H1, H2 |
| 9–12 | ebe1000 | gsm8k | traversal, specinfer | 3, 5 | 0.6, 1.0 | **H1, H2** |
| 13–16 | baseline | math500 | traversal, specinfer | 3, 5 | 0.6, 1.0 | H4 reference |
| 17–20 | kl1000 | math500 | traversal, specinfer | 3, 5 | 0.6, 1.0 | H4 |
| 21–24 | ebe1000 | math500 | traversal, specinfer | 3, 5 | 0.6, 1.0 | **H4** |
| 25–28 | baseline | gsm8k | gbv, bv | 3, 5 | 0.6, 1.0 | H1 bounds |
| 29–32 | ebe1000 | gsm8k | gbv, bv | 3, 5 | 0.6, 1.0 | H1, H2 bounds |
| 33–34 | online | gsm8k | traversal | 3, 5 | 0.6 | **H3** |
| 35–36 | online | math500 | traversal | 3, 5 | 0.6 | H3, H4 |

Rows 9–12 and 21–24 are the core paper results. All 36 must be in the a100 tier.

---

## 4. Key Metrics

### Alpha (α) — Token Acceptance Rate
**Range**: 0 to 1. **Higher is better.**

Alpha is the per-token acceptance probability: given that the target has seen position *t*, how likely is it to accept the draft's proposal at position *t+1*?

```
α_t = min(1,  q(token_t) / p_draft(token_t))
```

where `q` is the target's probability and `p_draft` is the draft's probability for the actual token generated. When `p_draft ≥ q` the proposal is always accepted (α=1). When `p_draft < q` the proposal is accepted with probability `q/p_draft < 1`.

**Reported as**: mean ± 95% CI across all prompts and token positions.

**What to look for**: a trained model should have higher alpha than the baseline. An increase from 0.7 → 0.8 means the draft is much more frequently in agreement with the target.

---

### Block Efficiency (BE) — Tokens per Target Call
**Range**: 1 to K·L+1. **Higher is better.**

Block efficiency is the average number of tokens added to the output per target model forward pass. A value of 1.0 means every target call only adds one token (pure autoregressive, no speedup). A value of 3.0 means each target call adds three tokens on average.

```
BE = (total tokens generated) / (number of target calls)
```

With a draft tree of width K and depth L, the theoretical maximum is K·L+1 (all tokens accepted). In practice, BE depends on how aligned the draft is with the target.

**Why this matters more than alpha**: alpha is token-level; BE captures the sequential structure of a full speculative block. A draft that gets all early tokens right provides a higher BE than one that scatters its accuracy across positions.

---

### Perplexity (PPL)
**Range**: 1 to ∞. **Lower is better. Do not let it drift far from baseline.**

Perplexity measures how surprised the draft model is by the evaluation dataset. It is used as a sanity check: if training pushes PPL much higher than the baseline (e.g., baseline=9.1, trained=13.9), the model has degraded as a language model even if its acceptance rate went up.

**Healthy outcome**: PPL stays within ~20% of baseline after training.

---

### Throughput (tokens/sec) and Latency (ms/token)
Measured end-to-end for the speculative decoding loop. Throughput increases with BE because more tokens are produced per expensive target call.

**NOTE**: throughput is hardware-dependent. Only compare throughput numbers from the same tier (a100 vs a100, never colab vs a100). The dashboard HW TIER filter prevents mixing tiers in charts.

---

## 5. Pipeline Phases (by tier)

### Laptop smoke tier (--config laptop_qwen --smoke)

```bash
python orchestration/experiment.py --config laptop_qwen --yes --smoke
```

Phase 0: Merge pre-existing LoRA adapters.  
Phase 1: Quick eval on gsm8k — baseline + LR sweep (3 LRs × 6 eval steps).  
Phase 5: Laptop smoke — ALL verifiers (traversal, specinfer, gbv, naive, bv) + ALL losses (200 steps each on diverse50). Exercises every code path.

**State file**: `pipeline_state_laptop_smoke.json` (separate from the full pipeline so smoke "done" marks never block real training).

---

### Colab tier (--config colab or kaggle)

```bash
python orchestration/experiment.py --config colab --yes    # T4, Qwen3-4B BF16
python orchestration/experiment.py --config kaggle --yes   # T4 x2, Qwen3-8B NF4
```

Phase 0–1: same structure as laptop.  
Phase 2: Training KL + EBE on gsm8k, 1000 steps, **with `--load_in_4bit`** (target loaded in 4-bit NF4 for T4 VRAM).  
Phase 2b: Reverse KL + JSD variants.  
Phase 2c: Online OSD adaptation.  
Phase 3: Full eval on gsm8k (traversal, specinfer, gbv).  
Phase 4: Multi-dataset eval.

All results tagged `hw_tier=colab` in results.db. Alpha and BE are directionally meaningful; throughput is not reliable due to quantization.

---

### A100 tier (--config a100_qwen)

```bash
python orchestration/experiment.py --config a100_qwen --yes
```

Same phases as T4, but:
- Target loads in **full bf16** — NO `--load_in_4bit`.
- All results tagged `hw_tier=a100`.
- Phase 3 GSM8K: **n=1319** (full 1,319-problem test set — auto-downloaded on first run).
- Phase 4 secondary: **n=100** each for alpaca, math500, humaneval, mtbench.
- ALL 6 verifiers: alpha, bv, gbv, traversal, specinfer, naive.

The `evaluate.py --hw_tier a100` guard will error if it detects a quantized target, preventing accidental contamination of paper-quality data.

---

### Per-phase details

#### Phase 0 — Merge (3 steps, ~5 min each)

Pre-trained LoRA adapters (from an earlier LR sweep) are merged into a copy of the base draft model and saved to disk as a standalone model.

**What "merge" means**: LoRA training adds two small matrices A and B alongside every weight W. The merged model computes `W' = W + A·B` and saves W' — the adapters disappear, the model file is a normal HuggingFace model.

**Inputs**: `checkpoints/ebe_lr{X}/` (LoRA adapter)  
**Outputs**: `checkpoints/ebe_lr{X}_merged/` (full model)

---

#### Phase 1 — Quick Eval on GSM8K (6 steps, ~10–15 min each)

**Purpose**: Rapidly see whether the LR-sweep checkpoints improved over baseline on math word problems, before committing to a 5-hour training run.

| Step | Model evaluated | What it tests |
|---|---|---|
| `eval_baseline_gsm8k` | Raw draft (no training) | Reference point for all comparisons |
| `eval_kl200_gsm8k` | KL-distilled, 200 steps on diverse data | Sanity check: does KL distillation help? |
| `eval_ebe200_gsm8k` | EBE-distilled, 200 steps on diverse data | **Key early signal**: does the novel loss help? |
| `eval_ebe_lr1e5_gsm8k` | EBE, lr=1e-5 | Learning rate sensitivity |
| `eval_ebe_lr3e5_gsm8k` | EBE, lr=3e-5 | Learning rate sensitivity |
| `eval_ebe_lr1e4_gsm8k` | EBE, lr=1e-4 | Learning rate sensitivity |

---

#### Phase 2 — Full Training on GSM8K

**Purpose**: Train the draft model properly on real GSM8K data, for 1000 steps.

| Step | Action |
|---|---|
| `train_kl_gsm8k` | Train with KL distillation loss for 1000 steps |
| `merge_kl_gsm8k` | Merge the KL-1000 LoRA into a standalone model file |
| `train_ebe_gsm8k` | Train with EBE loss for 1000 steps |
| `merge_ebe_gsm8k` | Merge the EBE-1000 LoRA into a standalone model file |

**Colab vs A100**: The pipeline automatically adds `--load_in_4bit` for colab config and omits it for server config. Never manually add `--load_in_4bit` to server runs.

---

#### Phase 3 — Full Eval on GSM8K

| Step | Model | What it proves |
|---|---|---|
| `eval_kl1000_gsm8k` | KL-1000 trained draft | Baseline trained model performance |
| `eval_ebe1000_gsm8k` | EBE-1000 trained draft | **Core result**: does EBE beat KL after full training? |

---

#### Phase 4 — Multi-Dataset Eval

| Step | Datasets covered |
|---|---|
| `eval_baseline_all` | HumanEval, MATH-500, MTBench, Alpaca (diverse50) |
| `eval_kl1000_all` | Same 4 datasets |
| `eval_ebe1000_all` | Same 4 datasets |

**Purpose**: Show that the improvement is not specific to GSM8K.

---

## 6. Training Losses

Losses fall into two families: **flat** (train on teacher's linear rollout) and
**tree-structured** (train on the student's own K-path draft tree).

### Flat Losses

Flat losses operate on a sequence of `[T, V]` logit tensors produced by a single teacher
forward pass.  Because the training tokens come from the **teacher's** distribution,
flat losses are off-policy with respect to the student's inference-time distribution.

---

#### KL Distillation (baseline training method)

```
L_KL = KL( q_target ∥ p_draft ) = Σ_v  q(v) · log( q(v) / p_draft(v) )
```

Standard knowledge distillation. Minimizes the KL divergence from target to draft over the full vocabulary at every token position.

**Limitation**: treats every token position independently. Doesn't directly optimize the acceptance probability product that determines block efficiency.

---

#### EBE Loss — Block-Level Expected Block Efficiency (novel contribution)

```
L_EBE = -(1 + Σ_{k=1}^{T} Π_{i=1}^{k} α_i)  +  λ · KL(q ∥ p_draft)

where  α_i = min(1, q(token_i) / p_draft(token_i))
```

**What it optimises**: the expected number of tokens accepted in a speculative block, which is exactly `1 + Σ_k Π_{i≤k} α_i`. Minimizing the negative of this maximizes block efficiency directly.

**Gradient intuition**: for token *i* where `p_draft > q` (draft over-estimates), the gradient pushes `p_draft` *down* toward `q`. The cumulative product (`torch.cumprod`) amplifies this signal for early tokens — if token 1 is the bottleneck, fixing it benefits the entire block.

**KL regularizer** (λ=0.1): the EBE gradient vanishes when `p_draft ≤ q` (the token is already under-estimated, α=1). Without KL, under-estimated token probabilities drift unconstrained, causing perplexity collapse. The KL term provides gradient for those positions.

**Key difference from KL**: KL only pushes draft probabilities toward the target; it cannot push them away when the draft is over-confident. EBE explicitly corrects over-estimates, which is the root cause of low acceptance rates.

**Off-policy caveat**: flat EBE computes `α_i = min(1, q(t_i)/p(t_i))` where `t_i` is the
**teacher's** sampled token.  At inference, the verifier evaluates the **student's** draft
token.  This mismatch (H5) is why flat EBE numbers are sometimes disappointing — use
`ebe_tree` to eliminate it.

---

#### Other flat losses

| Loss | CLI name | Formula | Notes |
|---|---|---|---|
| Reverse KL | `reverse_kl` | KL(p_draft ∥ q_target) | Mode-seeking; collapses to target peaks |
| JSD | `jsd` | ½KL(p∥m)+½KL(q∥m) | Symmetric; bounded [0, log 2] |
| L1 | `l1` | Σ |p(v) − q(v)| | Dense, distribution-spread; simple baseline |
| EBE single | `ebe_single` | same as ebe, block_len=1 | Token-level EBE; no cumprod |

---

### Tree-Structured Losses

Tree losses train on a **K-path draft tree** sampled from the student's own distribution.
This makes the training data on-policy: the student's distribution at training time
exactly matches the student's distribution at inference time.

**Three-phase training loop** (see DESIGN.md §9 for full details):

```
Phase A  iid_draft(student, prompt, K=4, L=8, no_grad)
          → K independent paths of length L from the student

Phase B  target_tree_pass(teacher, q_paths, no_grad)
          → p_probs_dict: teacher logits for every tree node

Phase C  draft_tree_forward_with_grad(student, q_paths)
          → q_probs_dict: student logits WITH autograd
          → compute_tree_loss(...) → .backward()
```

Phase A is no-grad so Phase C's autograd graph is clean.

**Verifier compatibility**: tree losses are evaluated only on non-OT verifiers (`bv`,
`gbv`, `traversal`).  OT-based verifiers (`specinfer`, `naive`) do not have a differentiable
acceptance integral and cannot guide a tree-specific loss.

---

#### `kl_tree` — On-policy Forward KL (universal baseline)

```
L = Σ_{nodes} KL(p_teacher ∥ q_student)  =  −Σ p · log(q)
```

The same forward-KL formula as flat KL, but applied at every node of the student's own
draft tree.  This is the on-policy counterpart of the flat KL baseline: it provides a
strong numerically-stable tree training signal without any verifier assumptions.

Evaluated against: **bv, gbv, traversal** (all three non-OT verifiers).

---

#### `rev_kl_tree` — On-policy Reverse KL

```
L = Σ_{nodes} KL(q_student ∥ p_teacher)  =  Σ q · log(q/p)
```

Mode-seeking: the student concentrates mass on the teacher's high-probability tokens.
Qwen3 sets −∞ logits on forbidden tokens; all log computations clamp `p` to [1e-9, ∞)
before taking logs.

Evaluated against: **bv, gbv, traversal**.

---

#### `jsd_tree` — On-policy Symmetric JSD

```
m = ½(q + p)
L = Σ_{nodes}  ½·KL(q ∥ m)  +  ½·KL(p ∥ m)
```

Bounded [0, log 2] regardless of how peaked q or p are.  Numerically softer than either
KL direction: the mixture m always has non-negligible mass at any token that either model
assigns non-negligible mass to.

Evaluated against: **bv, gbv, traversal**.

---

#### `bv_tree` — BV Acceptance Integral Loss

Directly optimises the batch-verification (BV) acceptance integral at each tree node.
BV acceptance at a node = `min(1, p/q)` integrated over the token distribution.
Maximising this integral (minimising its negative) trains the student to increase the
probability that BV accepts the tokens it proposes.

**Learning rate:** The `h_i` block-acceptance denominator amplifies gradients relative to
`kl_tree` (stability clamps in `tree_losses.py` bound the worst cases, but a lower step
size still helps). Keep YAML `lr: 3e-5` for mixed sweeps; for dedicated `bv_tree` runs use
`--lr 1e-5`, e.g.
`python orchestration/experiment.py --config a100_qwen --losses bv_tree --lr 1e-5`.

Evaluated primarily against: **bv**.  Also evaluated against gbv/traversal in full runs.

---

#### `gbv_tree` — GBV Acceptance Integral Loss (with q-skew)

GBV extends BV with a query-skew reweighting that gives more weight to paths with higher
cumulative acceptance probability.  `gbv_tree` targets the resulting acceptance integral.

**Numerical stability note**: `gbv_tree` is stable only for `tree_K ≤ 4`.  At K=5+, the
q-skew term in `compute_skew` can underflow.  Always set `tree_K: 4` in YAML configs when
using `gbv_tree`.

**Learning rate:** Same gradient amplification as `bv_tree` — use `--lr 1e-5` for dedicated
`gbv_tree` training (see `bv_tree` above).

Evaluated primarily against: **gbv**.

---

#### `traversal_tree` — Traversal Leaf-Weight Product Loss

Traversal uses a leaf-weight product (bottom-up acceptance) that assigns each path a
weight equal to the product of acceptance probabilities along it.  `traversal_tree` targets
this exact weight product, making it the best-matched loss for the traversal verifier.

Evaluated primarily against: **traversal**.

---

#### `ebe_tree` — On-policy EBE Ablation

Same EBE formula as flat EBE (`1 + Σ cumprod(α)`), but `α_i = min(1, p[t_i]/q[t_i])`
where `t_i` is the **student's own draft token** at position i (not the teacher's token).

This loss is a direct ablation designed to answer one question:

> How much of flat EBE's underperformance is due to **off-policy training data**
> (teacher tokens) vs the **EBE formula itself**?

If `ebe_tree` >> flat `ebe`, the off-policy mismatch is the root cause.
If `ebe_tree` ≈ flat `ebe`, the EBE formula itself is the bottleneck.

Evaluated against: **bv** (smoke), **bv, gbv, traversal** (full runs).

---

#### Summary table

| Loss | Family | CLI name | Aligned verifier |
|---|---|---|---|
| Forward KL | flat | `forward_kl` | (generic) |
| Reverse KL | flat | `reverse_kl` | (generic) |
| JSD | flat | `jsd` | (generic) |
| L1 | flat | `l1` | (generic) |
| EBE | flat | `ebe` | (generic) |
| EBE single | flat | `ebe_single` | (generic) |
| On-policy KL | tree | `kl_tree` | (generic — all 8) |
| On-policy Reverse KL | tree | `rev_kl_tree` | (generic — all 8) |
| On-policy JSD | tree | `jsd_tree` | (generic — all 8) |
| BV integral | tree | `bv_tree` | **bv** |
| GBV integral | tree | `gbv_tree` | **gbv** (K ≤ 4 only) |
| Traversal leaf-weight | tree | `traversal_tree` | **traversal** |
| Naive (Chen/Leviathan) α | tree | `naive_tree` | **naive** |
| NSS α | tree | `nss_tree` | **nss** |
| SpecInfer α (K-iter) | tree | `specinfer_tree` | **specinfer** |
| SpecTr K-SEQ α (ρ detached) | tree | `spectr_tree` | **spectr** |
| Khisti canonical decomp α | tree | `khisti_tree` | **khisti** |
| On-policy EBE | tree | `ebe_tree` | (ablation — paired with bv) |
| Online KL (tree mode) | online+tree | `online_kl_tree` | (generic) |
| Online EBE (tree mode) | online+tree | `online_ebe_tree` | (generic) |

#### Verifier-aligned tree losses (added 2026-05)

The 5 OT-based tree losses (`naive_tree`, `nss_tree`, `specinfer_tree`,
`spectr_tree`, `khisti_tree`) are derived from each verifier's own
closed-form per-node acceptance probability α_V (defined in
`verifiers/tree.py` as `*_otlp_accept`).  The training objective is

  **L_V  =  −E[τ_V]  =  −Σᵢ Πⱼ₌₁ⁱ α_V(pⱼ, qⱼ, K)**

i.e. negative expected length of accepted prefix.  This is the same
scaffold as `bv_tree` (with BV block acceptance) and `traversal_tree`
(with leaf-weight product), but using the OT verifier's own per-node α.

**Loss-verifier alignment hypothesis** (paper main result, Phase 3 on
A100): a draft trained with L_V outperforms a draft trained with L_{V'}
when both are evaluated under V.  The 8×8 cross-pair table is the
empirical check — diagonal should beat off-diagonal cells.

**Assumptions documented in `tree_losses.py`**:

- `spectr_tree`: ρ (binary-search root) is detached.  Gradient flows
  through `min(p/ρ, q)`, not through ρ itself.  Future work: implement
  via implicit function theorem if results show spectr_tree under-
  performs other aligned losses.
- `khisti_tree`: LP solver is replaced by a smooth softmax-based
  importance reweighting (q_imp = q · softmax(K·log(p/q))).  Matches
  K=1 special case exactly; LP-based exact LB is future work.

---

## 7. Dashboard

Start with:
```bash
# Run from gbv-research/
python dashboard/training_dashboard.py    # opens http://127.0.0.1:5000
```

The dashboard auto-refreshes every 30 seconds. The status bar at the top shows live pipeline progress.

---

### HW Tier Filter (NEW)

At the top of the sidebar is a **HW TIER** filter with three chips:

| Chip | Color | Meaning |
|---|---|---|
| `laptop` | Blue (#4299e1) | Smoke/correctness runs only — exclude from paper charts |
| `colab` | Orange (#ed8936) | Trend-formation, quantized — directional only |
| `a100` | Green (#48bb78) | Paper-quality, bf16 — include in paper tables |

**Default**: all three selected.

**Typical workflow for paper tables**: deselect `laptop` and `colab`, keep only `a100`. All charts will then show only paper-quality numbers.

The filter applies to all charts simultaneously (alpha, BE vs K, mode comparison, etc.). Legacy runs inserted before the `hw_tier` column was added are treated as `laptop`.

---

### Sidebar Filters

Click chips to filter which runs are shown in all charts. Multiple chips within one category are OR'd; across categories they are AND'd.

Example: select `ebe1000` in Draft/Loss AND `gsm8k` in Dataset AND `a100` in HW TIER → shows only the EBE-1000 model on GSM8K on A100 hardware.

Click **Apply Filters** to update charts, **Clear All** to reset all filters including HW TIER back to all-selected.

---

### Status Bar (top of page)

| Element | Meaning |
|---|---|
| **Pulsing yellow badge** | Step currently running |
| **Green badge** | Step completed |
| **Red badge** | Step failed — check logs |
| **Progress bar** | N of total steps done |
| **GPU bar** | Live VRAM usage (orange → red as it fills) |
| **DB:** label | Most recently completed evaluation cell |
| **Steps button** | Expand/collapse per-step status panel with tooltips |
| **Logs button** | Show/hide live GBV subprocess output (`be_progress.log`) |

---

### Tab: Key Results

The three hypothesis-proving charts. Start here.

**Chart A — Loss × Verifier Heatmap**  
Rows = verifier modes, Columns = distillation models (baseline, kl200, ebe200, …). Color intensity = block efficiency. Darker red = higher BE.

*What to look for*: the EBE column should be darker than the KL column, especially for traversal and GBV modes.

**Chart B — BE Gain over Baseline (%)**  
Shows `(BE_model − BE_baseline) / BE_baseline × 100` for each trained model and verifier mode.

*What to look for*: EBE bars should be taller than KL bars. Negative bars mean the model is *worse* than the untrained baseline.

**Chart C — Dataset Robustness**  
Block efficiency across all four datasets (GSM8K, HumanEval, MATH-500, diverse50).

*What to look for*: consistent ordering across datasets (EBE > KL > baseline). If EBE wins only on GSM8K but loses on HumanEval, the model has overfit to the training distribution.

---

### Tab: Alpha

**Alpha by Condition (bar chart)**  
Mean token acceptance rate (α) per model, grouped by dataset. Error bars = 95% CI.

**Alpha Distribution (horizontal bars)**  
Same data, horizontal orientation, easier to compare models across many datasets.

---

### Tab: BE vs K

**Block Efficiency vs K (line chart)**  
X axis = K (3 or 5), Y axis = mean BE. One line per model.

*What to look for*: all lines should increase with K. The gap between EBE and baseline should widen as K increases.

**BE vs K — All Modes (subplots)**  
Same chart repeated for each verifier mode side by side, so you can see which mode benefits most from K.

---

### Tab: Mode Comparison

Grouped bar chart: X axis = verifier mode, Y axis = BE. One bar group per model.

*What to look for*: the ordering traversal ≥ gbv > specinfer > bv should hold (H1). EBE should gain more than KL (H2).

---

### Tab: Temperature

**Temperature Sensitivity (line chart)**  
X axis = sampling temperature (0.6, 1.0), Y axis = BE. One line per model.

*What to look for*: lines should decrease left-to-right (lower T = higher BE). The *relative ordering* of models should be preserved at both temperatures.

**Temperature Gain vs Baseline (line chart)**  
`BE_model − BE_baseline` at each temperature.

---

### Tab: Sensitivity

Fully configurable scatter/line chart. Choose any X axis (K, temperature, train steps, learning rate) and any Y axis (alpha, BE, throughput, latency), grouped by any categorical column.

---

### Tab: Per-Category

Heatmap of token acceptance rate broken down by prompt category within the `diverse50` dataset.

---

### Tab: Throughput

**Throughput by Condition (bar chart)**  
Tokens per second, grouped by dataset and verifier mode.

**IMPORTANT**: Only compare throughput within the same `hw_tier`. Use the HW TIER filter to select `a100` only before reading throughput charts.

---

### Tab: Training Curves

Loss plotted against training step. One line per model. Updated live during training.

*What to look for*:
- Both curves should decrease smoothly
- EBE loss starts large (the cumprod term can be large) but should converge
- If a curve spikes upward or oscillates, the learning rate is too high
- **Red banner** appears if val loss rises >10% above its minimum → early stopping recommended

---

### Tab: All Runs

Full table of every evaluation row in the database. Supports sorting and can be exported to CSV.

**Key columns**:
- `hw_tier`: which hardware tier produced this result (laptop / colab / a100)
- `draft_label`: which model was evaluated
- `mode`: verifier algorithm
- `K`: tree width
- `temperature`: sampling temperature
- `alpha_mean` / `alpha_ci95`: acceptance rate with confidence interval
- `block_eff`: block efficiency
- `throughput`: tokens per second
- `perplexity`: language model perplexity (sanity check)
- `experiment_tag`: tags multiple runs from the same experiment session

---

## 8. How to Run

### Training budget — how many steps do I need?

The key insight: you need enough **epochs** (passes over the training set), not raw step count.

```
epochs = steps / max_train_prompts

Visible trend   → 2 epochs minimum (with strong teacher signal)
Clear convergence → 5 epochs
Paper quality   → full dataset, no cap (diversity matters more than epochs)
```

Without `max_train_prompts`, most configs only see a fraction of one epoch:

| Without cap | With cap (default) |
|---|---|
| 500 steps / 7473 prompts = **6.7%** of 1 epoch | 500 steps / 250 prompts = **2.0 epochs** |
| Loss curve: flat/noisy, no visible trend | Loss curve: clear monotonic decrease |

**Per-config values (set in YAML, propagated automatically):**

| Config | Steps | max_train_prompts | Epochs | Purpose |
|---|---|---|---|---|
| `laptop_qwen` | 100 | 50 | 2 | crash-test: code runs without crash |
| `laptop_gpt2` | 500 | 100 | 5 | convergence proof: GPT-2 distillation works |
| `laptop_llama` | 500 | 100 | 5 | convergence proof: LLaMA 1B→3B works |
| `server_gpt2` | 1000 | 200 | 5 | CPU convergence proof |
| `colab` | 500 | 250 | 2 | T4 trend: 4B teacher, 2 epochs shows direction |
| `kaggle` | 1000 | 500 | 2 | T4 trend: 8B NF4 teacher, reliable ranking |
| `a10_qwen` | 2000 | 1000 | 2 | A10 convergence: 8B BF16, liberal epochs |
| `a100_qwen` | 2000 | none | 0.27 | paper: full 7473-prompt diversity, no cap |

**Why the A100 has no cap**: paper results need the model to have seen the full diversity of 7473 training prompts. With an 8B teacher, 27% of one epoch still produces a strong enough signal for the loss rankings to be reliable.

**Override on CLI** (e.g. to run a quick convergence check):
```bash
# 300 steps with max 100 prompts = 3 epochs — convergence check in ~5 min on A10
python orchestration/experiment.py --config a10_qwen --losses kl --train_steps 300 --yes
```
(The `max_train_prompts` from YAML still applies unless you change it.)

---

### Laptop smoke test (verify setup, ~25 min)
```bash
python orchestration/experiment.py --config laptop_qwen --smoke --yes
```

### Full laptop code-path run (~2-3 hr)
```bash
python orchestration/experiment.py --config laptop_qwen --yes
# GPT-2 alternative (meaningful convergence signal on CPU):
python orchestration/experiment.py --config laptop_gpt2  --yes
python orchestration/experiment.py --config laptop_llama --yes  # requires HF login
```

### CPU server convergence run (ATS Cloud, no GPU needed)
```bash
python orchestration/experiment.py --config server_gpt2 --yes   # ~16-27 hr total
```

### Run only specific losses (no config file change needed)
```bash
# CLI flag — runs only kl and jsd, skips everything else:
python orchestration/experiment.py --config laptop_qwen --losses kl,jsd --yes

# Via profile YAML — set experiment: losses: [kl, jsd] in the config
python orchestration/experiment.py --config profiles/online_only_laptop --yes
```

### A10 exploration runs (Stage 1.5)

A10 runs one loss at a time for fast iteration. Each run is ~80-100 min:

```bash
# One loss at a time — baseline runs once, --skip_existing resumes:
python orchestration/experiment.py --config a10_qwen --losses kl --yes \
  --storage_root /path/to/storage
python orchestration/experiment.py --config a10_qwen --losses kl_tree --yes \
  --storage_root /path/to/storage

# Smoke first (always):
python orchestration/experiment.py --config a10_qwen --smoke --yes
```

**Config `a10_qwen` key settings** (see `orchestration/configs/a10_qwen.yaml`):
- `steps: 2000`, `max_train_prompts: 1000` → 2 epochs with 8B BF16 teacher
- `load_in_4bit: false` → exact teacher distribution (vs Kaggle NF4)
- `lora_r: 8` → more capacity than T4 configs (r=4)
- `compile: true` → torch.compile on Linux GPU (~10-20% speedup)

**Reading W&B loss curves (what good looks like):**
- `train/loss_ema`: should decrease monotonically. Spiky raw `train/loss` is normal (high per-prompt KL variance); the EMA trend is the signal.
- `train/val_loss`: for tree losses (`kl_tree` etc.) this uses forward-KL as a proxy, NOT the tree objective — slight increase is expected and not alarming.
- `train/grad_norm`: initial spike then settling to 1-3 is healthy. Sustained >10 = LR too high.
- `gradients/lora_B_norm_mean`: should be small and stable (~0.03-0.08). Large spike + recovery is normal early-training behavior.

---

### A100 paper runs — 3-tier profile system

A100 runs use profiles, not raw `--config a100`. Run in order:

**Tier 1 — baseline losses** (train KL/JSD/L1/online, ~2-3 h):
```bash
python orchestration/experiment.py --config profiles/a100_baseline_losses --smoke --yes \
  --storage_root /content/drive/MyDrive/specdist
python orchestration/experiment.py --config profiles/a100_baseline_losses --yes \
  --storage_root /content/drive/MyDrive/specdist
```

**Tier 2 — verifier sweep** (eval-only on Tier 1 checkpoints, ~1-2 h):
```bash
python orchestration/experiment.py --config profiles/a100_verifier_sweep --yes \
  --storage_root /content/drive/MyDrive/specdist
```

**Tier 3 — new loss** (train new loss, compare vs baselines):
```bash
cp orchestration/configs/profiles/a100_new_loss_template.yaml \
   orchestration/configs/profiles/a100_my_loss.yaml
# edit: losses: [my_loss, kl, jsd, l1, online]
python orchestration/experiment.py --config profiles/a100_my_loss --yes \
  --storage_root /content/drive/MyDrive/specdist
```

### Eval-only (re-evaluate existing checkpoints, no training)
```bash
# Via YAML: set experiment: eval_only: true  (see a100_verifier_sweep.yaml)
# Via CLI flag:
python orchestration/experiment.py --config profiles/a100_baseline_losses --eval_only --yes \
  --storage_root /content/drive/MyDrive/specdist
```

### Check status without running
```bash
python orchestration/experiment.py --status
python orchestration/experiment.py --dry_run   # shows full plan with step states
```

### Resume after a crash
State is saved after every step. Just re-run the same command — already-done steps are skipped:
```bash
python orchestration/experiment.py --config profiles/a100_baseline_losses --yes \
  --storage_root /content/drive/MyDrive/specdist
```

### Jump to a specific step
```bash
python orchestration/experiment.py --config profiles/a100_baseline_losses --yes \
  --from eval_kl_gsm8k --storage_root /content/drive/MyDrive/specdist
```

### View dashboard
```bash
python dashboard/training_dashboard.py
```
Open http://127.0.0.1:5000 — use HW TIER filter to isolate a specific tier's results.

---

## 8a. Analysing Training Progress with Checkpoints

Training saves permanent numbered checkpoints every 200 steps (controlled by `--milestone_every`):
```
checkpoints/kl-run/
  ckpt_latest/          ← crash-safe, always overwritten
  ckpt_step_00200/      ← permanent snapshot at step 200
  ckpt_step_00400/      ← permanent snapshot at step 400
  ...
```

To answer "is training actually helping?", evaluate block efficiency at each checkpoint:

```bash
# Run from gbv-research/

# 1. Merge the LoRA adapter into a standalone model
python algorithms/distillspec_gbv/trainer.py --merge_only \
    --adapter db/checkpoints/kl-gsm8k/ckpt_step_00200 \
    --draft Qwen/Qwen3-0.6B

# 2. Run speculative decoding eval on the merged model
python orchestration/evaluate.py \
    --student db/checkpoints/kl-gsm8k/ckpt_step_00200_merged \
    --teacher Qwen/Qwen3-8B \
    --student_label kl_step200 \
    --datasets gsm8k --modes gbv,specinfer --K 3 --n 10 \
    --hw_tier a100

# 3. Repeat for each checkpoint, then view in dashboard
python dashboard/training_dashboard.py
```

The Mode Comparison tab will show a bar for each checkpoint. If BE increases step-by-step, training is working. If it flatlines or drops after a certain step, that is where to stop training.

**What rising validation loss means**: A red banner appears if val loss rises >10% above its minimum. This is the clearest early indicator that training is starting to memorise the training set rather than learning generalizable alignment.

---

## 8b. Running Training on a Server (Colab / Modal)

The existing `algorithms/distillspec_gbv/trainer.py` runs anywhere with no code changes needed. These are the setup steps for each environment.

---

### Google Colab (free T4, 15 GB VRAM)

**Preferred method**: open `deploy/colab_quickstart.ipynb` and run top-to-bottom. That's it.

For manual cell-by-cell control:

```python
# Cell 1: Mount Drive so checkpoints survive session restart
from google.colab import drive
drive.mount('/content/drive')

# Cell 2: Install dependencies
!pip install -q peft transformers accelerate bitsandbytes

# Cell 3: Clone repo and train
import os
os.environ["TRANSFORMERS_OFFLINE"] = "0"   # allow first-time download

# COLAB ONLY: --load_in_4bit loads the 8B target in 4-bit NF4 for T4 15GB
# DO NOT use --load_in_4bit on A100/server runs (bf16 full precision required)
!python algorithms/distillspec_gbv/trainer.py \
    --draft  Qwen/Qwen3-0.6B \
    --target Qwen/Qwen3-8B \
    --load_in_4bit \
    --loss forward_kl \
    --steps 1000 \
    --output /content/drive/MyDrive/specdist/checkpoints/kl-run \
    --dataset core/datasets/raw/gsm8k_train.jsonl \
    --milestone_every 200 \
    --val_every 50
```

**Notes:**
- `--load_in_4bit` is safe for colab only — the teacher is frozen during training.
- Set `--output` to a Google Drive path so checkpoints survive session death.
- After the first run, set `TRANSFORMERS_OFFLINE=1` to avoid re-downloading.
- When running `evaluate.py` on colab after training, pass `--hw_tier colab` to tag results correctly.

#### Validation frequency and compute cost

**Why `--val_every` defaults to 50 and not 10 (like `--log_every`):**

Training loss is free — it is already computed during the forward pass of every training step. Validation loss is expensive — it runs a completely separate inference cycle for each val prompt.

**Recommended settings per environment:**

| Environment | `--val_every` | `--val_split` | Rationale |
|---|---|---|---|
| Laptop (≤ 4 GB) | 50 (default) | 0.10 | Low overhead; 10 prompts cap keeps val ≤ 70 s |
| Colab T4 | 100 | 0.05 | Avoids 20% overhead eating into session time |
| A100 / H100 server | 50 | 0.10 | Cheap enough at this GPU speed |
| Quick debug run | 0 | — | Disable val entirely with `--val_every 0` |

---

### Modal.com (A100 on-demand, billed per second)

Modal requires a Python wrapper around the training call. `modal_train.py` in this directory is that wrapper.

```bash
# One-time setup
pip install modal
modal token new

# Download models into Modal's persistent volume (~16 GB, runs once)
modal run modal_train.py::download_models

# Train — KL baseline
modal run modal_train.py::train_kl

# Train — EBE novel loss
modal run modal_train.py::train_ebe

# Custom args (e.g. 2000 steps, lr=1e-4)
modal run modal_train.py::run --loss ebe --steps 2000 --lr 1e-4

# Download checkpoints from the volume to your laptop
modal volume get specdist-vol /checkpoints ./local_checkpoints
```

**Cost reference**:
- **IITH A100** (preferred): ₹80/GPU-hour (~$0.94/hr). A full 2000-step pipeline (train + eval) ≈ 4–6 hr ≈ ₹320–480 (~$4–6). Best value.
- **Modal A100-80GB**: ~$3–4 per 1000-step run. ⚠️ **Only $1 free credit available** (no longer $30). Effectively pay-per-use from the first run. Reserve for a single targeted ablation, not a full pipeline.
- **Lightning AI**: monthly free credits (A100 available); good for sustained exploration.
- **RunPod spot A100**: ~$1.5–2/hr; requires manual checkpoint management.

---

## 8c. Setting Up a New Environment (scripts/setup_download.py)

`scripts/setup_download.py` is a **one-time-per-machine** (or **one-time-per-session** on ephemeral compute) script that downloads all required model weights and datasets.

### When to run it

| Environment | When to run |
|---|---|
| **Your laptop** | Once. After that, `TRANSFORMERS_OFFLINE=1` is set automatically. |
| **Kaggle** | Every new session (kernel restarts clear the disk cache). |
| **Google Colab** | Every new runtime unless you save to Google Drive. |
| **Modal.com** | Once into the persistent volume via `modal run modal_train.py::download_models`. |

### Usage

```bash
# Laptop (0.5B draft + 0.6B target — default)
python scripts/setup_download.py --config laptop

# Server / Colab / Kaggle (0.6B draft + 8B target)
python core/datasets/downloader.py

# Check what will be downloaded without downloading
python core/datasets/downloader.py --datasets gsm8k --n 30
```

---

## 8d. Generating a Results Report (analyze_results.py)

`db/analyze_results.py` reads `results.db` and generates the **primary paper view**:
which **(trained model, verifier)** pair achieves the best block efficiency (BE).

### Why not a single BE column?

The DB stores **one row per eval cell**: `(draft_label, mode, dataset, K, temperature)`.
A full A100 Phase 3 eval saves ~9 BE rows per loss (one per verifier) plus one `alpha` row.
The quick A100 sanity-check snippet hardcodes `mode='gbv'` for BE — that is **not**
averaging across verifiers, but it also **hides** the best verifier per model.

Use `analyze_results.py` to answer: *"Does `rev_kl` win under `traversal` while `jsd`
wins under `bv`?"* — the loss–verifier alignment table.

### Usage

```bash
cd gbv-research/

# Full report (alpha + BE matrix + convergence + recommendations)
python db/analyze_results.py

# Save for the paper or a PR
python db/analyze_results.py --out report.md

# Block-efficiency section only (loss × verifier matrix)
python db/analyze_results.py --section be

# Limit compared drafts (baseline is always the anchor)
python db/analyze_results.py --section be --compare rev_kl,jsd,kl,l1

# Global best (model, verifier) pair only
python db/analyze_results.py --section recommend
```

On A100:

```bash
source ~/.specdist_env && cd "$GBV_DIR"
python db/analyze_results.py
```

### What it reports

| Section | Content |
| -------- | -------- |
| **Eval Conditions** | dataset, K, L, T, n, train_steps, lr, lora_r, hw_tier, experiment_tag per cell |
| **Paper Metrics** | α + **best BE (and which verifier)** + GSM8K EM; flags missing alpha rows |
| **Loss × Verifier matrix** | Full BE table; per-verifier ranking (bv / gbv / traversal) |
| **Hypothesis checks** | H1 traversal vs specinfer; H2 rev_kl vs kl; H4/H5 when data exists |
| **Data gaps** | Which losses have BE but no alpha; missing verifier modes |
| **Training health** | **Val loss only** — does *not* compare train loss across objectives |
| **Narrative** | α-vs-BE gap interpretation; novelty framing; next experiments |
| **Recommendations** | Best `(model, verifier)` with full eval conditions |

```bash
# Filter one session
python db/analyze_results.py --experiment_tag KrishnanRIITHServer --dataset gsm8k

# Single sections
python db/analyze_results.py --section metrics
python db/analyze_results.py --section hypotheses
python db/analyze_results.py --section gaps
```

**Removed / deprecated:** cross-objective train-loss "reduction %" table (KL vs JSD vs
rev_kl scales are incomparable; early grad-step snapshots are noisy). Use W&B val loss
or `--section training` for within-loss convergence.

A `—` cell means that `(model, verifier)` eval was never saved (partial run or skipped).

### Example interpretation (partial finalization run)

```text
| Mode      | K | baseline | jsd    | kl     | rev_kl |
| bv        | 3 |   3.6683 | 4.6960 | 4.4573 | 4.7799 |
| gbv       | 3 |   3.6229 | 4.4454 | 4.3035 | 4.0980 |
| traversal | 3 |   3.9605 |      — |      — | 4.9851 |

Recommendations:
- Best block efficiency: 4.9851 (rev_kl, traversal, K=3, gsm8k_eval)
```

Read as: **`rev_kl` trained draft + `traversal` verifier** is the current global BE
winner; **`jsd` peaks under `bv`** (4.70), not `gbv` (4.45). Ranking losses by
`gbv` alone would under-rank `jsd`.

### Quick snapshot vs full matrix

| Tool | Use when |
| ------ | -------- |
| `python db/analyze_results.py` | Choosing best **model + verifier**; paper tables |
| Per-tag Python snippet (see `deploy/A100_SETUP.md`) | Fast α / BE@gbv / GSM8K EM for one `experiment_tag` |
| W&B `eval_summary_table` artifact | Per-run pivot after each `evaluate.py` invocation |

`analyze_results.py` scans **all** rows in `results.db` (no `experiment_tag` filter).
Filter by tag in SQL or a custom script when comparing one session only.

### When to run it

- After each loss completes Phase 3 eval (incremental ranking)
- After Phase 4 (cross-dataset robustness section populates)
- Before Stage 2B / paper confirmation — to pick top losses **and** their best verifiers

---

## 8e. EAGLE Benchmark

EAGLE is a separate speculative-decoding method that trains a lightweight 1-layer head directly on the **target** model's hidden states. It provides a strong external baseline.

### Run on Colab / Modal / server only — not on laptop

The EAGLE head trains on the **target** model's hidden states. The paper's target is **Qwen3-8B** which requires 16+ GB VRAM.

| Environment | EAGLE? | Why |
|---|---|---|
| **Laptop** (0.6B target) | Do not run | 0.6B head is not a useful baseline |
| **Colab A100** (8B target) | Run here | Fits in VRAM |
| **Modal.com** (8B target) | Run here | Persistent volume survives runs |
| **In-house server** | Run here | Best option if available |

### Running Eagle as part of the pipeline

```bash
# Run full pipeline + EAGLE baseline — use --config server or colab (never laptop)
python orchestration/experiment.py --config a100_qwen --yes --eagle

# Run EAGLE phases only (Phases 0–4 already done)
python orchestration/experiment.py --config a100_qwen --yes --eagle --from eagle_gen
```

---

## 9. Automatic Training Health Checks

`trainer.py` runs five automated health checks as training proceeds.

### 9a. How Each Check Works

| Check | Trigger | Pass condition | What you see in logs |
|---|---|---|---|
| **1. Loss decreasing** | Every `--health_every` steps (default 100) | Last-10 avg < First-10 avg | `↓ IMPROVING` in health report |
| **2. No NaN in loss** | Every step, before backward | No NaN/Inf | `[HEALTH] ✓ none` in report |
| **3. PPL not collapsed** | Every `--ppl_check_every` steps (default 200) | PPL ≤ baseline × `--ppl_threshold` (default 1.25×) | `[HEALTH] PPL=X.XX  ratio=Y.YY× ✓ within threshold` |
| **4. Checkpoint files exist** | After each `--save_every` save | `adapter_config.json` present in `ckpt_latest/` | `[HEALTH] ✓ Checkpoint verified` |
| **5. Val loss not rising** | Every `--val_every` steps (default 50) | Val loss ≤ best × 1.10 | Health report row 3; red banner in dashboard |

### 9b. New Command-Line Flags

```bash
# NaN handling (default: stop and save checkpoint)
python algorithms/distillspec_gbv/trainer.py ... --nan_action stop   # abort on NaN (saves checkpoint first)
python algorithms/distillspec_gbv/trainer.py ... --nan_action skip   # discard step, continue
python algorithms/distillspec_gbv/trainer.py ... --nan_action warn   # log only, continue

# Health report frequency
python algorithms/distillspec_gbv/trainer.py ... --health_every 100  # print report every 100 steps

# Val loss frequency
python algorithms/distillspec_gbv/trainer.py ... --val_every 50      # check val loss every 50 train steps
python algorithms/distillspec_gbv/trainer.py ... --val_split 0.10    # hold out 10% of prompts for val

# PPL vs baseline
python algorithms/distillspec_gbv/trainer.py ... --ppl_check_every 200 --ppl_threshold 1.25

# Early stopping (recommended for unattended server runs)
python algorithms/distillspec_gbv/trainer.py ... --early_stop_patience 5
```

---

## 10. Weights & Biases Integration

### 10a. Does W&B help for training only, or also for inference / eval?

**Both**:

| Phase | What W&B tracks | Script |
|---|---|---|
| **Training** | Loss curves (train + val), PPL health, LR, GPU util, gradient histograms | `trainer.py` |
| **Eval / Inference** | Alpha, block efficiency, task score, throughput per eval cell | `evaluate.py` |
| **Hyperparameter sweep** | LR × LoRA-rank × loss-type sweep | `sweep_config.yaml` |

### 10b. Setup

```bash
pip install wandb
wandb login   # enter your API key from https://wandb.ai/authorize
```

W&B is **optional** — if not installed or not logged in, training and eval continue without it.

### 10c. Training Metrics

Every training run logs to the `distillspec` W&B project:

| W&B metric | Logged every | Description |
|---|---|---|
| `train/loss` | `--log_every` steps | Rolling average training loss |
| `train/lr` | `--log_every` steps | Learning rate |
| `train/peak_vram_mb` | `--log_every` steps | Peak CUDA memory |
| `train/steps_per_sec` | `--log_every` steps | Training throughput |
| `val/loss` | `--val_every` steps | Held-out validation loss |
| `train/ppl` | `--ppl_check_every` steps | Draft model perplexity |
| `train/ppl_ratio` | `--ppl_check_every` steps | PPL / baseline |

### 10d. Eval Metrics

Each eval cell logs one point to W&B:

| W&B metric | Available in mode | Description |
|---|---|---|
| `eval/alpha_mean` | alpha | Token acceptance rate |
| `eval/block_eff` | specinfer, gbv, traversal | Average tokens per verification call |
| `eval/task_score` | alpha + `--task_score` | GSM8K exact match / HumanEval pass@1 |

### 10e. Hyperparameter Sweeps

The sweep config at `orchestration/configs/sweep.yaml` runs a **Bayesian sweep** over lr, lora_r, and loss type.
Use `orchestration/run_sweep.py` rather than calling `wandb sweep` directly (it sets up the correct project and agent).

```bash
# Step 1 — register the sweep and run 20 trials
python orchestration/run_sweep.py --count 20

# Step 2 — add more parallel agents on other machines
wandb agent <entity>/distillspec/<sweep_id>
```

---

## 11. Quick Interpretation Cheatsheet

| Observation | Likely cause | Action |
|---|---|---|
| EBE alpha < KL alpha | EBE loss not converging, or LR too high | Check Training Curves tab; try lower LR |
| EBE PPL >> baseline PPL (e.g. 13 vs 9) | EBE loss pushing over-estimated tokens without KL regularizer | Ensure `kl_weight > 0` in training; re-run |
| BE gain is negative | Training made the draft *worse* | Check PPL; model may have collapsed |
| **Red banner in Training Curves tab** | **Val loss rose >10% above its minimum** | **Early stopping — the best checkpoint is the one just before the uptick** |
| Traversal doesn't beat specinfer | Draft alignment below threshold | Check alpha; plot traversal-specinfer vs alpha to find threshold |
| `NaN` in training loss | `alpha=0` in cumprod (target prob ≈ 0 for that token) | Should not happen after `clamp(min=1e-6)` fix; if recurs, lower LR |
| Charts show "No data yet" | Eval step not finished yet | Wait; check status bar for running step |
| Alpha high but throughput low | Verification overhead (traversal) dominates | Expected for large K on slow hardware; compare at K=3 |
| BE flat across K values | Draft too misaligned | EBE may not have converged; check LR and training steps |
| Colab BE > A100 BE | Quantization artifact or different n prompts | Use HW TIER filter; only compare within same tier |
| BE varies wildly across runs | Too few prompts (T4 uses n=30) | Use A100 config: n=1319 for GSM8K, n=100 for secondary datasets |

---

## 12. KL Divergence Comparison

DistillSpec (Zhou et al. 2023) Section 3.1 explicitly uses **forward KL** — KL(target ∥ student).

| Direction | Formula | Property | Expected Acceptance |
|---|---|---|---|
| Forward KL | KL(target ∥ student) | Mode-covering — student must cover all target modes | Highest (baseline) |
| Reverse KL | KL(student ∥ target) | Mode-seeking — student collapses to high-confidence tokens | Lower for tail tokens |
| JSD | ½KL(s∥m)+½KL(t∥m) | Symmetric blend, bounded [0, log 2] | Between fwd and rev |
| EBE | Expected block efficiency | Directly optimizes acceptance probability | Highest overall |

**Why forward KL is right for speculative decoding:** Acceptance rate = probability that draft token matches target's sample. To maximise this, the draft must assign nonzero probability to *every* token the target could produce — exactly what forward KL enforces.

**How to run all variants:**
The pipeline runs all three KL variants as Phase 2b steps:
```bash
python orchestration/experiment.py --config a100_qwen --yes --from train_rev_kl_gsm8k
```

Results appear in the dashboard's Training Curves and Key Results tabs, color-coded:
- Forward KL: orange
- Reverse KL: red
- JSD: green
- EBE: blue
- Online OSD: purple

---

## 13. Online Speculative Decoding (OSD)

**Paper:** Liu et al. 2023 — https://arxiv.org/abs/2310.07177

### How it improves over offline distillation

| Mode | Training data | When it adapts |
|---|---|---|
| Offline (DistillSpec) | Fixed dataset (GSM8K train) | Once, before deployment |
| Online (OSD) | Live inference requests | Continuously, while serving |

Offline distillation trains on a fixed dataset sampled before deployment. If the actual queries differ from the training distribution, the draft model's acceptance rate will be lower. **Online OSD eliminates this gap** by updating the draft model in real time as queries arrive.

### Algorithm

```
For each incoming prompt:
  1. Run speculative decode: draft proposes K tokens, target verifies
  2. Record full sequence + positions where draft was rejected
  3. Buffer (sequence, rejection_positions)

Every update_every prompts:
  4. Re-run BOTH models on buffer (target frozen, draft trainable)
  5. Compute forward KL loss ONLY at rejection positions
  6. Update draft weights
```

**Why only rejection positions?** Accepted positions already agree with the target — their gradient is near zero. Training on rejection positions focuses the update where the draft is wrong.

### Running online adaptation

```bash
# Run from gbv-research/

# Start from base draft (no prior training)
python algorithms/online_serve.py \
    --prompts core/datasets/raw/gsm8k_train.jsonl \
    --steps 500 --output db/checkpoints/online-gsm8k

# Start from a pre-trained offline checkpoint (recommended — faster convergence)
python algorithms/online_serve.py \
    --prompts core/datasets/raw/gsm8k_train.jsonl \
    --steps 500 \
    --adapter db/checkpoints/ebe-gsm8k \
    --output db/checkpoints/online-gsm8k
```

### Tests that prove improvement

#### 1. Wandb training curves (live, during the run)

| Metric | What it means | Expected trend |
|---|---|---|
| `online/alpha` | Rolling acceptance rate | Rises from ~0.73 → 0.80+ over 500 steps |
| `online/eval_alpha` | Acceptance rate on held-out prompts | Rises monotonically; end value > baseline |
| `online/loss` | Forward KL loss at rejection positions | Decreases (model correcting its errors) |

**Pass threshold (H3):** `eval_alpha` at step 500 > `eval_alpha` at step 0.  
A successful online run should reach ≥ 0.76 on GSM8K after 500 steps starting from baseline (~0.73).

#### 2. Ranked comparison (run after pipeline finishes Phase 3)

Use `orchestration/evaluate.py` directly to compare all checkpoints:

```bash
python orchestration/evaluate.py \
    --student db/checkpoints/online-gsm8k_merged \
    --teacher Qwen/Qwen3-0.6B \
    --datasets gsm8k --modes specinfer,traversal --K 3,5
```

This writes results to `db/results.db`; compare models in the dashboard.

#### 3. Dashboard comparison

After Phase 3/4 evals complete, open `http://127.0.0.1:5000`:
- **Key Results tab** → `online` bar should be taller than `baseline` bar in the alpha chart
- **Mode Comparison tab** → select `online-gsm8k` vs `ebe-gsm8k` to see the gap

### KL method for online mode

Online OSD defaults to `--kl_method forward_kl`, matching the paper.

```bash
python algorithms/online_serve.py --kl_method reverse_kl ...  # ablation
python algorithms/online_serve.py --kl_method jsd ...          # symmetric blend
```

### Online Tree Distillation (new)

Standard flat online training uses a **replay buffer** of `(token_ids, teacher_logits,
wrong_mask)` tuples collected from prior speculative decoding calls.  The training data
comes from the teacher's sampling — the same off-policy problem as flat offline losses.

**Online tree mode** replaces the buffer update with a fresh K-path tree update:

```
For each incoming prompt (serve loop — unchanged):
  1. Run speculative decoding as normal (linear, alpha measurement)
  2. No token goes into the buffer

Every update_every prompts (update loop):
  3. Re-run the CURRENT prompt through the three-phase tree loop:
     A. iid_draft(student, prompt, K, L, no_grad)  → q_paths
     B. target_tree_pass(teacher, q_paths, no_grad) → p_probs_dict
     C. draft_tree_forward_with_grad(student, q_paths) → q_probs_dict
     D. compute_tree_loss(loss_name, ...) → .backward() → step
```

Serving and tree-update phases are strictly interleaved, not concurrent.
No replay buffer is needed; the tree is always built from the current prompt.

**Online tree fixes the zero-gradient bug documented in online_serve.py lines 197–228**:
flat EBE online produces zero gradient when the draft over-estimates a position (α=1),
which is the majority of positions for a well-trained model.  Tree EBE and tree KL
always have non-zero gradient because the tree samples fresh on-policy paths every update.

**CLI flags** (added to `algorithms/online_serve.py`):
```bash
# Use tree KL update instead of flat buffer update
python algorithms/online_serve.py \
    --prompts core/datasets/raw/gsm8k_train.jsonl \
    --tree_loss kl_tree --tree_K 2 --tree_L 8 \
    --steps 500 --output db/checkpoints/online-kl-tree-gsm8k

# Use tree EBE update
python algorithms/online_serve.py \
    --prompts core/datasets/raw/gsm8k_train.jsonl \
    --tree_loss ebe_tree --tree_K 2 --tree_L 8 \
    --steps 500 --output db/checkpoints/online-ebe-tree-gsm8k
```

When `--tree_loss` is set, `--kl_method` is ignored.  All 7 tree loss names are accepted.
Keep `tree_K ≤ 4` if using `gbv_tree` (numerical stability constraint).

### In-pipeline integration

```
Phase 2  — train_kl_gsm8k, train_ebe_gsm8k              (offline flat, 1000 steps each)
Phase 2b — train_rev_kl_gsm8k, train_jsd_gsm8k          (flat KL variant comparison)
Phase 2c — online_adapt_gsm8k, online_ebe_adapt_gsm8k    (flat online, 500 prompts)
Phase 2d — train_kl_tree, train_bv_tree, train_gbv_tree, (offline tree, 1000 steps each)
           train_traversal_tree, train_ebe_tree,
           train_rev_kl_tree, train_jsd_tree
Phase 2e — online_kl_tree_adapt, online_ebe_tree_adapt   (online tree, 500 prompts)
Phase 3  — full eval on gsm8k (ALL variants)
Phase 4  — full eval on all datasets
```
