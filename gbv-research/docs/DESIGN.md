# SpecDist — Design Reference

**What this is**: A single-document description of what is implemented, how it maps to DistillSpec, how it simplifies the OSD codebase, what AdaSPEC adds (not implemented), and how the GBV multi-path verifiers work.

---

## 1. What We Built vs. What Existed

| | OSD (github) | DistillSpec (paper) | Our implementation |
|---|---|---|---|
| Training mode | **Online** (during serving) | **Offline** (before deployment) | **Offline** |
| Training trigger | Every N served requests | Fixed N steps on a dataset | Fixed N steps on a dataset |
| Draft architecture | Full model weights | Full model | **LoRA adapters only**, base frozen |
| Sample source | student / teacher / mix | **teacher only** (recommended) | **teacher only** |
| Loss function | forward KL, reverse KL, JSD | forward KL + EBE | forward KL + EBE (+ reverse KL, JSD, L1 as ablations) |
| Token masking | Only wrong positions (wrong_token_ids) | All generated positions | All generated positions |
| Verifier at training time | SD runs inside training loop | None — separate eval | None — separate eval |
| Eval framework | llamacpp-based | Separate pass | Separate `evaluate.py` → GBV subprocess |
| Model family | LLaMA/Vicuna | Generic | **Qwen2.5-0.5B → Qwen3-0.6B** (laptop) / **Qwen3-0.6B → Qwen3-8B** (server) |
| HF Trainer | Yes (DistillTrainer) | N/A | No — plain PyTorch loop |

---

## 2. DistillSpec Baseline — Implementation Mapping

### 2.1 Core claim of DistillSpec

> Aligning the draft model's distribution with the target model's distribution — before inference — raises the token acceptance rate α, which directly raises block efficiency BE.

Two design choices the paper identifies as critical:
1. Use **teacher-sampled continuations** (not student-sampled or pre-existing text)
2. Match the divergence function to the inference-time acceptance criterion

### 2.2 Training data

**Paper**: frozen target generates continuations; both models score the same sequence.

**Our code** (`algorithms/distillspec_gbv/trainer.py`):
```python
with torch.no_grad():
    full_ids = target_model.generate(
        prompt_ids, max_new_tokens=80, do_sample=True, temperature=0.8)
with torch.no_grad():
    teacher_logits = target_model(full_ids).logits[:, :-1, :].float()
draft_model.train()
student_logits = draft_model(full_ids).logits[:, :-1, :].float()
```

**Difference from OSD**: OSD's online mode (`distill_trainer.py:92`) runs actual speculative decoding inside the training step and only trains on `wrong_token_ids` — positions where the draft was rejected. DistillSpec (and our code) trains on every generated position.

### 2.3 Draft model

**Paper**: start from pretrained small model, fine-tune with KL distillation.

**Our code**: LoRA adapters on top of frozen base weights (r=8, α=16, targeting q/k/v/o projections). Only 1.08M of 495M parameters are trainable on Qwen2.5-0.5B.

**Why LoRA**: full fine-tuning on small model risks catastrophic forgetting of fluency; LoRA adapts the distribution while keeping the base decoding intact.

### 2.4 Loss functions

#### Forward KL — DistillSpec primary baseline
```
L_KL = KL(p_target || p_draft) = -Σ_v p_target(v) · log p_draft(v)
```
Mode-covering: draft learns to have mass wherever the target has mass. Recommended by DistillSpec as the default because it directly minimises the gap between target and draft distributions at every position.

**Code** (`algorithms/distillspec_gbv/losses/forward_kl.py`):
```python
def forward_kl(student_logits, teacher_logits, token_ids=None, **kw):
    log_s = F.log_softmax(student_logits, dim=-1)
    p_t   = F.softmax(teacher_logits,    dim=-1)
    return LossOutput(loss=-(p_t * log_s).sum(dim=-1).mean())
```

#### EBE — novel contribution
The loss directly optimises expected block efficiency over non-overlapping windows of `L=8` draft tokens:

```
L_EBE = -E[τ+1] = -avg_blocks( 1 + Σ_{k=1}^{L} Π_{i=1}^{k} α_i ) + λ·KL
```

where `α_i = min(1, q(t_i)/p(t_i))` and `λ≈0.1` is the KL regulariser weight.
The KL term provides gradient for already-accepted tokens (where the EBE gradient vanishes) and keeps language quality stable.

**Code** (`algorithms/distillspec_gbv/losses/ebe.py`):
```python
def ebe(student_logits, teacher_logits, token_ids, kl_weight=0.1, block_len=8, **kw):
    log_s = F.log_softmax(student_logits, dim=-1)
    log_t = F.log_softmax(teacher_logits, dim=-1).detach()
    log_p = log_s.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)  # [T]
    log_q = log_t.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)  # [T]
    alpha = torch.exp(torch.clamp(log_q - log_p, max=0.0)).clamp(min=1e-6)  # α ∈ (0,1]

    n_blocks, ebe_val = max(1, alpha.shape[0] // block_len), torch.zeros(1)
    for b in range(n_blocks):
        blk     = alpha[b * block_len : (b + 1) * block_len]
        ebe_val = ebe_val - (1.0 + torch.cumprod(blk, dim=0).sum())
    ebe_val = ebe_val / n_blocks   # scale-independent of sequence length

    loss = ebe_val + kl_weight * F.kl_div(log_s, log_t.exp(), reduction="batchmean")
    return LossOutput(loss=loss, accept_weight=alpha.mean().item())
```

**Why block-level**: partitioning the sequence into non-overlapping blocks of 8 tokens matches the inference-time draft window, avoids the "giant cumprod" gradient problem for long sequences, and makes the training scale directly comparable to the block efficiency evaluation metric.

### 2.5 Training loop

Per-step (`algorithms/distillspec_gbv/trainer.py`):
1. Sample a prompt (shuffle per epoch over the dataset)
2. Frozen target generates continuation (teacher-sampled)
3. Both models score the full sequence in one forward pass
4. Compute loss on generated portion only (not prompt tokens)
5. `loss.backward()` through draft LoRA params only
6. AdamW + gradient clip at 1.0

**Difference from OSD offline mode**: OSD's `offline_training_step` supports four sample sources (teacher, student, mix_request, mix_token) and uses HuggingFace Trainer with batch sizes and gradient accumulation. Ours is deliberately a single-sample-per-step loop — simpler, easier to reason about, and matches the DistillSpec description.

### 2.6 Evaluation — fully separate from training

**`evaluate.py`** is the evaluation entry point. It does not touch training code.

For each `(student, dataset, mode, K, temperature)` cell:
- `mode=alpha`: runs SpecInfer rejection sampling inline, measures α (token acceptance rate)
- `mode=specinfer/gbv/traversal/bv`: shells out to `GBV/main.py`, parses `"Block efficiency: X.XXX"`

Results go into `results.db` (SQLite). `dashboard/training_dashboard.py` reads from the DB and renders charts.

---

## 3. What OSD Does That We Deliberately Omit

### 3.1 Online mode

OSD's key contribution is *online* distillation: the draft model continues training while serving live traffic, using the actual query distribution. This achieves 0.1–0.65 higher α and 1.22×–3.06× latency reduction over a static draft.

We omit online mode because:
- DistillSpec asks: does offline distillation help at all? That's the baseline.
- Online adds complexity (buffer management, concurrent serving+training, query routing).
- It's a separate research contribution and not the paper we're implementing.

### 3.2 wrong_token_ids masking

In OSD's online step (`distill_trainer.py:164–168`), only positions where the draft was **rejected** during SD get gradient signal:
```python
mask = torch.ones_like(input_ids, dtype=torch.bool)
for i, data in enumerate(self.buffer):
    mask[i, wrong_token_ids] = False  # unmask only the wrong ones
```
This is the OSD insight: train harder on tokens the draft consistently gets wrong at inference time.

We train on **all** generated positions (no masking). This is what DistillSpec does — it trains on the full teacher-generated sequence, not conditioned on what SD rejected.

### 3.3 Buffer accumulation + batching

OSD accumulates multiple SD rollouts into a buffer, then does one batched gradient update. We do one update per step. No buffer, no batching complexity.

### 3.4 HuggingFace Trainer

OSD subclasses `transformers.Trainer` (a full training framework with checkpointing, evaluation callbacks, DDP, etc.). Our trainer is 50 lines of plain PyTorch. Appropriate for a research baseline that fits on one GPU.

### 3.5 Mix-token / mix-request sampling

OSD supports sampling where each token is drawn from either the teacher or student with probability `mix_ratio`. This interpolates between teacher-sampled and student-sampled distributions. We use teacher-only — the DistillSpec recommended setting.

---

## 4. AdaSPEC — What It Adds (Not Implemented)

AdaSPEC is a **different algorithm**, not a hyperparameter variant of DistillSpec. It requires a three-model system:

```
target (frozen)  →  reference model (KD-pretrained, frozen)  →  draft (trained)
```

### 4.1 Selective token filtering

For each token position t, compute the **loss gap**:
```
ΔL(t) = L_draft(t) − L_ref(t)
```
Tokens with larger ΔL are "easier" — the draft can improve on them more than the reference already did. Train only on the top-k% of easiest tokens (k=0.4, i.e. 40%):

```python
mask = (delta_L >= torch.quantile(delta_L, 1 - k)).float()
loss = (mask * kl_per_token).mean()
```

### 4.2 Why it matters vs. our EBE

EBE weights by **acceptance probability** (calibration signal). AdaSPEC weights by **learnability** (capacity signal). They are complementary: EBE says "train more on tokens that help acceptance", AdaSPEC says "train more on tokens the draft can actually learn". A natural extension would be combining both masks.

### 4.3 When to add it

Add AdaSPEC as a separate `--loss adaspec` mode in `algorithms/distillspec_gbv/trainer.py` after the KL/EBE baseline is solid. It requires pre-training a reference model first (one extra training run). Treating it as an ablation in the paper ("selective distillation") is the cleanest framing.

---

## 5. GBV — Multi-Path Verifier Algorithms

### 5.1 Background: vanilla SpecInfer

Standard speculative decoding (SpecInfer):
1. Draft proposes a **linear chain** of K tokens: `t_1, t_2, ..., t_K`
2. Target verifies all K+1 positions in a **single forward pass** (batched)
3. Accept tokens left-to-right using rejection sampling: `accept t_i iff U < min(1, p_t(t_i)/p_d(t_i))`
4. On first rejection, replace with target-sampled token and discard the rest

**Block efficiency** (BE) = expected tokens accepted per target call.

For SpecInfer with K draft tokens and acceptance rate α:
```
BE = (1 - α^(K+1)) / (1 - α)   (geometric series)
```

### 5.2 BV — Batch Verification

BV is essentially SpecInfer with the explicit observation that all K tokens can be verified in a single batched forward pass (exploiting that transformers can process variable-length sequences). It establishes the baseline that one target call can check K tokens.

### 5.3 GBV — Generalized Batch Verification

Instead of a single linear chain, the draft proposes a **tree** of candidate continuations. Each node in the tree is a token; each path from root to leaf is one possible continuation.

**Tree construction**: draft samples multiple completions by running the draft model with temperature > 0, or by keeping the top-B tokens at each step (beam-like). This gives a tree where each path has different tokens at divergence points.

**Single target forward pass over the tree**: the target scores all nodes simultaneously. Using attention masks that enforce causal structure per-path, all paths in the tree are verified in one call.

**Acceptance**: paths are accepted using the rejection sampling criterion applied to each node in order. The best accepted prefix across all paths is kept.

**Why this helps**: with K tokens in a linear chain, you get at most K+1 accepted tokens per call. With a tree, multiple paths can be partially accepted — you take the longest accepted prefix across all paths. This increases expected BE even for the same K total draft tokens.

**Assumptions GBV makes**:
1. Draft and target share identical vocabulary (required for the `p_d/p_t` ratio)
2. Target is always frozen at inference — no gradient through the verifier
3. Draft tokens are conditionally independent given the prefix (standard AR assumption)
4. The attention mask for tree verification is constructed so each node only attends to its own path's ancestors — not to sibling branches
5. Acceptance probabilities are well-defined: both `p_d(t)` and `p_t(t)` are non-zero for all sampled tokens (guaranteed if sampling with temperature > 0)

### 5.4 Traversal

Traversal is a specific tree-construction strategy for GBV. Rather than uniform beam expansion, it traverses the draft model's probability tree in a priority order (e.g., depth-first or best-first by cumulative log-probability). This produces trees that are more concentrated on high-probability paths, improving the expected length of accepted prefixes.

The acceptance algorithm is the same as GBV — the only difference is how the tree is built.

### 5.5 How we use GBV in evaluation

`evaluate.py:run_be()` calls `GBV/main.py` as a subprocess:
```python
cmd = [sys.executable, "GBV/main.py",
       "--p_model", teacher_path,    # frozen target
       "--q_model", student_path,    # draft (merged LoRA)
       "--mode", mode,               # specinfer | bv | gbv | traversal
       "--K", str(K),                # draft tokens per round
       "--L", str(L),                # tree depth (for gbv/traversal)
       "--data", data_path]
```
Parses `"Block efficiency: X.XXX"` from stdout and stores in `results.db`.

### 5.6 The paper's central interaction claim

The core hypothesis:
- KL-trained draft raises α uniformly → helps all verifiers roughly equally
- EBE-trained draft raises acceptance-weighted probability → disproportionately helps tree verifiers (GBV/traversal) because those methods benefit from multiple path-specific acceptance events
- GBV/traversal amplify a well-calibrated draft more than SpecInfer does, because they can exploit partial acceptance along multiple branches

This is the `Loss × Mode` interaction the viz dashboard's Key Results tab plots as a heatmap.

---

## 6. Experiment Matrix

→ See **GUIDE.md § 3** for the full training × verifier experiment matrix and evaluation conditions (datasets, temperatures, K values).

---

## 7. Pipeline Phases

`experiment.py --config laptop --yes` runs 18 steps across 5 phases:

| Phase | Steps | What it does |
|---|---|---|
| **0 — Merge** | 3 | Merge pre-existing LoRA adapters (kl200, ebe200, ebe_lr*) into full-weight models. **Auto-skipped** if `*_merged/config.json` already exists on disk. |
| **1 — Quick Eval** | 6 | Eval baseline + all pre-existing checkpoints on gsm8k (alpha + block-efficiency). Establishes the LR-sweep baseline before heavy training. |
| **2 — Training** | 4 | Train KL-1000 on gsm8k_train (1000 steps, ~5 h), merge it; then EBE-1000 similarly. These are the primary paper models. |
| **3 — Full Eval** | 2 | Eval the newly trained kl1000 + ebe1000 on gsm8k. Produces the core 3×3 interaction matrix for the paper. |
| **4 — Multi-Dataset** | 3 | Eval baseline + kl1000 + ebe1000 on humaneval, math500, mtbench, alpaca. Validates generalization beyond the training distribution. |

Resume at any time with `python experiment.py --config laptop --yes`. Steps with
a `done_check` file (training steps) are skipped if the output artifact exists.
Eval steps are skipped if the result is already in `results.db` (`--skip_existing`).

---

## 8. File Layout

→ For the current file map, see **PROJECT_CONTEXT.md § Codebase Architecture**.
