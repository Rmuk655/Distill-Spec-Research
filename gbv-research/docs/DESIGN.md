# SpecDist — Design Reference

**What this is**: A single-document description of what is implemented, how it maps to DistillSpec, how it simplifies the OSD codebase, what AdaSPEC adds (not implemented), and how the GBV multi-path verifiers work.

---

## 1. What We Built vs. What Existed

| | OSD (github) | DistillSpec (paper) | Our implementation |
|---|---|---|---|
| Training mode | **Online** (during serving) | **Offline** (before deployment) | **Offline** (primary) + **Online** (ablation) |
| Training trigger | Every N served requests | Fixed N steps on a dataset | Fixed N steps on a dataset |
| Draft architecture | Full model weights | Full model | **LoRA adapters only**, base frozen |
| Sample source | student / teacher / mix | **teacher only** (recommended) | **teacher only** (offline); student's own draft paths (online tree) |
| Loss function | forward KL, reverse KL, JSD | forward KL + EBE | forward KL + EBE (+ rev\_KL, JSD, L1, 7 tree losses as ablations) |
| Token masking | Only wrong positions (wrong_token_ids) | All generated positions | All generated positions |
| Verifier at training time | SD runs inside training loop | None — separate eval | None — separate eval |
| Eval framework | llamacpp-based | Separate pass | Separate `evaluate.py` → GBV subprocess |
| Model family | LLaMA/Vicuna | Generic | **Qwen2.5-0.5B → Qwen3-0.6B** (laptop) / **Qwen3-0.6B → Qwen3-8B** (server / A100) |
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

**We implement online mode as an ablation** (H6: does on-policy online tree training close the off-policy gap?). Four online variants run in the pipeline:

| Pipeline step | Loss | Architecture |
|---|---|---|
| `online_adapt_gsm8k` | `online_kl` | Flat KL — OSD-style replay buffer of `wrong_token_ids` |
| `online_ebe_adapt_gsm8k` | `online_ebe` | Flat EBE — same buffer, EBE loss on rejected positions |
| `online_kl_tree_adapt_gsm8k` | `online_kl_tree` | **Tree KL** — no buffer; fresh K-path tree from live prompt |
| `online_ebe_tree_adapt_gsm8k` | `online_ebe_tree` | **Tree EBE** — no buffer; fresh K-path tree from live prompt |

The flat online variants replicate OSD's buffer architecture (collect rollouts, replay-train on `wrong_token_ids`). The tree online variants discard the buffer entirely — see §7.5. The off-policy problem (§7.1) applies to the flat online variants just as it does to flat offline training; tree online eliminates it.

The **primary research questions** (H1–H5) are answered by offline distillation. Online mode is included to test H6 and to provide a fair comparison with OSD's key result. It is not the paper's main contribution.

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

## 7. Tree-Structured Distillation (H5 — On-Policy Training)

### 7.1 The Off-Policy Problem in Flat Losses

All flat losses (forward KL, EBE, L1, …) share the same training data source:

```python
with torch.no_grad():
    full_ids = target_model.generate(prompt_ids, ...)  # teacher tokens
teacher_logits = target_model(full_ids).logits
student_logits = draft_model(full_ids).logits          # scored on teacher tokens
loss = forward_kl(student_logits, teacher_logits, ...)
```

The tokens `full_ids` are drawn from the **teacher's** distribution.  At inference time,
the verifier scores the **student's** proposed tokens.  This off-policy mismatch means:

1. The student learns to have the right distribution conditional on teacher-generated
   prefixes, but at inference it conditions on its own prefix continuations.
2. For EBE specifically: `α_i = min(1, q(t_i)/p(t_i))` uses the teacher's token `t_i`.
   The verifier at inference evaluates `min(1, q(s_i)/p(s_i))` where `s_i` is the
   student's own token.  These two ratios can be very different.

### 7.2 Tree Training: Three-Phase Loop

Tree training eliminates the off-policy mismatch by building the training tree entirely
from the student's own samples:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  Phase A — Draft tree sampling (no_grad)                                      │
│                                                                                │
│  q_paths = iid_draft(student, prompt_ids, K=4, L=8)                          │
│    → Run student K times independently (temperature > 0)                     │
│    → Each run produces L new tokens → q_paths: list of K lists of L ints     │
│    → NO gradient here — we don't want autograd through Phase A               │
│                                                                                │
│  Why K independent runs?  → K i.i.d. paths give unbiased Monte Carlo          │
│  estimate of the expectation each loss minimises.  Beam search introduces     │
│  correlation; i.i.d. sampling does not.                                       │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│  Phase B — Teacher scoring (no_grad)                                          │
│                                                                                │
│  p_probs_dict = target_tree_pass(teacher, q_paths)                           │
│    → For each unique prefix in the q_paths tree, run one teacher forward pass │
│    → Return dict: { ",".join(path[:d]) → teacher_prob_tensor [V] }           │
│    → Detached: teacher is always frozen                                       │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│  Phase C — Student scoring WITH grad + loss                                   │
│                                                                                │
│  q_probs_dict = draft_tree_forward_with_grad(student, q_paths)               │
│    → Same prefix enumeration as Phase B, but WITH autograd enabled           │
│    → Return dict: { prefix → student_prob_tensor [V] }  (has .grad_fn)       │
│                                                                                │
│  loss = compute_tree_loss(loss_name, q_probs_dict, p_probs_dict, q_paths)    │
│  loss.backward()                                                              │
│  optimizer.step()                                                             │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Why Phase A must be no_grad**: if Phase A ran with gradient, Phase C's call to
`draft_tree_forward_with_grad` would create a second autograd graph through the same model
parameters, duplicating every computation in the graph.  Detaching Phase A's sampling
gives a fixed set of paths for Phase B/C to score — no graph doubling.

**Memory cost**: Phase B and Phase C each forward-pass the student/teacher once per unique
prefix in the tree.  For K=4, L=8, the tree has ≤ 32 unique prefixes (fewer when paths
share a prefix) plus the K × L leaf nodes.  Peak VRAM is approximately 2× a single
forward pass.

### 7.3 Loss Functions

Each tree loss iterates over all `(prefix, token)` pairs and computes a per-node
contribution.  All losses use the same node iteration:

```python
for path in q_paths:                        # K paths
    for depth in range(1, L + 1):           # L nodes per path
        prefix = ",".join(str(x) for x in path[:depth])
        token  = path[depth]
        p = p_probs_dict[prefix].detach()   # teacher [V]
        q = q_probs_dict[prefix]            # student [V], WITH grad
```

**`kl_tree`**: forward KL at each node.
```
loss_node = −Σ_v p(v) · log q(v)       (= KL(p ∥ q) + H(p), const wrt q)
```
Mode-covering.  Numerically stable for all K and L.  Universal baseline for tree training.

**`rev_kl_tree`**: reverse KL at each node.
```
loss_node = Σ_v q(v) · log(q(v)/p(v))  (= KL(q ∥ p))
```
Mode-seeking: draft concentrates mass on teacher's highest-probability tokens.  Clamps
`p.clamp(min=1e-9)` before log to handle Qwen3's −∞ forbidden-token logits.

**`jsd_tree`**: symmetric JSD at each node.
```
m = α·q + (1−α)·p,  α = 0.5
loss_node = ½·KL(q ∥ m) + ½·KL(p ∥ m)
```
Bounded [0, log 2].  The mixture m always has non-negligible mass at any token that
either model assigns non-negligible mass to, so log(q/m) is always finite.

**`bv_tree`**: BV acceptance integral loss.  Maximises `E_q[min(1, p/q)]` at each node
by minimising its negative.  Directly targets the BV verifier's acceptance probability.

**`gbv_tree`**: GBV acceptance integral with q-skew.  The q-skew term weights paths by
their cumulative acceptance probability, so the gradient concentrates on paths the GBV
verifier is most likely to accept.  **Numerically stable for `tree_K ≤ 4` only.**

**`traversal_tree`**: traversal leaf-weight product loss.  Traversal verifier scores each
path by the product of acceptance weights along it.  This loss targets that product
directly, making it the best-matched loss for the traversal verifier.

**`ebe_tree`**: on-policy EBE ablation.  Same formula as flat EBE:
```
loss = −(1 + Σ_{i=1}^{L} Π_{j≤i} α_j)  +  λ·KL_tree
where α_j = min(1, p[t_j] / q[t_j].clamp(min=1e-9))
```
but `t_j` is the **student's own token** at position j (from Phase A sampling).  This
isolates the off-policy mismatch: same formula as flat EBE, completely on-policy data.

### 7.3.1 Verifier-aligned OT tree losses (added 2026-05)

**`naive_tree`, `nss_tree`, `specinfer_tree`, `spectr_tree`, `khisti_tree`** — five new
tree losses, one per OT-based verifier.  Each uses the verifier's own closed-form
per-node acceptance probability α_V from `verifiers/tree.py` (the `*_otlp_accept`
methods) as the training target.

The unified scaffold is:

```
L_V  =  −E[τ_V]  =  −Σᵢ Πⱼ₌₁ⁱ α_V(pⱼ, qⱼ, K)
```

i.e. negative expected length of accepted prefix under verifier V along a path of L
nodes.  This is the same telescoping identity used by `bv_tree` (with BV block
acceptance) and `traversal_tree` (with leaf weight) — generalised to any verifier with
a differentiable α formula.

Per-verifier α (full math in `losses/tree_losses.py`):

| Loss | α formula (per node) | Source |
|---|---|---|
| `naive_tree` | Σ_v min(p[v], q[v]) + Σ_v relu(p−q)·(1−(1−q)^{K−1}) | Chen 2023; Leviathan 2023 |
| `nss_tree` | Σ_v p[v]·(1 − (1−q[v])^K) | Miao et al. 2024 |
| `specinfer_tree` | iterative K-iter rejection — see code | Miao 2024 (Alg. 2) |
| `spectr_tree` | p_acc + (1−p_acc)·Σ p_res·(1−(1−r)^K) | Sun 2023 (Thm. 1) |
| `khisti_tree` | LP-free surrogate: Σ min(p, q·softmax(K·log p/q)) | Khisti 2025 |

**Engineering assumptions, documented in code**:

- **SpecTr ρ-detach**: ρ is found by binary search; we detach it for the gradient pass
  and let q flow through `min(p/ρ, q)` only.  Sound first-order surrogate; ρ varies
  slowly with q across training steps.  Future: implicit function theorem.
- **Khisti LP-free**: replace the K−1 LP iterations with a softmax-based importance
  reweighting (matches K=1 exactly; monotone in K).  Future: differentiable LP.

**Loss-verifier alignment hypothesis** (the headline result Phase 3 tests on A100):

  *A draft trained with L_V outperforms a draft trained with L_{V'} when both are
  evaluated under verifier V*

is empirically verified by the **8×8 cross-pair matrix** in Phase 3 (see
`_tree_full_modes` in `experiment.py`).  Diagonal cells should beat off-diagonal
cells.

### 7.4 Verifier Compatibility

After 2026-05, every verifier has a paired tree loss:

```
non-OT:   bv, gbv, traversal
  ↑       paired with: bv_tree, gbv_tree, traversal_tree
  
OT-based: naive, nss, specinfer, spectr, khisti
  ↑       paired with: naive_tree, nss_tree, specinfer_tree, spectr_tree, khisti_tree
```

On A100, `_tree_full_modes = "naive,nss,specinfer,spectr,khisti,bv,gbv,traversal"`
(the full 8-verifier matrix).  On T4 it falls back to `"bv,gbv,traversal"` for
compute-budget reasons.  Generic divergence tree losses (`kl_tree`, `rev_kl_tree`,
`jsd_tree`) eval against all configured verifiers.

### 7.5 Online Tree Distillation

The flat online training loop collects a buffer of `(token_ids, teacher_logits,
wrong_mask)` from prior speculative decoding calls, then trains on the buffer.  The
buffer tokens come from the teacher — off-policy, same problem as flat offline.

**Online tree mode** (CLI: `--tree_loss kl_tree --tree_K 2 --tree_L 8`) discards the
buffer entirely.  At each update interval, the three-phase tree loop runs on the **current
prompt** just served:

```python
if args.tree_loss:
    # No buffer needed — build a fresh tree from the current prompt
    loss_val = _tree_online_update(
        draft_model, target_model, input_ids,
        optimizer, loss_name=args.tree_loss,
        tree_K=args.tree_K, tree_L=args.tree_L, temperature=args.temperature,
    )
```

This also fixes the **zero-gradient EBE bug** (documented in `online_serve.py` lines
197–228): flat EBE produces zero gradient when the draft over-estimates a position (α=1),
which is most positions for a well-trained model.  Tree EBE samples fresh paths on every
update, so it always hits some positions where the student's token is suboptimal.

Two online tree variants are in the pipeline:
- `online_kl_tree`: tree forward-KL update — numerically stable, universal
- `online_ebe_tree`: tree EBE update — directly optimises acceptance product

---

## 8. Pipeline Phases

The full pipeline (`experiment.py --config <cfg> --yes`) runs ~70 steps across 7 phases.
Resume any time — completed steps are skipped automatically (training: `done_check` file;
eval: `--skip_existing` against `results.db`).

### Track A — Flat Distillation Losses (Phases 1–5)

| Phase | Steps | What it does |
|---|---|---|
| **1 — Baseline** | 1 | `eval_baseline_gsm8k` — eval untrained draft with all verifiers on GSM8K. Pins the "no training" floor for every comparison. |
| **2 — Flat Training** | 14 | Train + merge 7 flat losses on GSM8K train split: `kl`, `ebe`, `rev_kl`, `jsd`, `l1`, `online_kl`, `online_ebe`. Each loss: `train_<loss>_gsm8k` → `merge_<loss>_gsm8k`. |
| **3 — GSM8K Eval** | 7 | Eval each flat-loss model on GSM8K with all verifiers (`specinfer`, `bv`, `gbv`, `traversal`, `alpha`). Produces the core Loss×Mode interaction matrix. |
| **4 — Multi-Dataset Eval** | 8 | Eval `baseline` + all 7 flat-loss models on `humaneval`, `math500`, `mtbench`, `alpaca`. Validates generalisation outside the training distribution. |
| **5 — EAGLE Benchmark** | 3 | `eagle_gen` → `eagle_train` → `eagle_eval`. External EAGLE baseline for block-efficiency comparison in the paper table. |

### Track B — Tree-Structured Losses (Phases 6–7)

| Phase | Steps | What it does |
|---|---|---|
| **6 — Tree Training + GSM8K** | 27 | Train + merge 9 tree losses: `kl_tree`, `rev_kl_tree`, `jsd_tree`, `bv_tree`, `gbv_tree`, `traversal_tree`, `ebe_tree`, `online_kl_tree`, `online_ebe_tree`. Then eval each on GSM8K with **non-OT verifiers only** (`bv`, `gbv`, `traversal`). |
| **7 — Tree Multi-Dataset** | 9 | Eval all 9 tree-loss models on `humaneval`, `math500`, `mtbench`, `alpaca` (same non-OT verifier subset). |

### Key design points

- **Tree losses use non-OT verifiers only.** `experiment.py` hardcodes `_TREE_NON_OT = "bv,gbv,traversal"` for all tree eval steps. OT verifiers (`specinfer`, `naive`) are out-of-distribution for tree-trained models.
- **Crash-safe state machine.** Every step writes `status: done` to `pipeline_state_<config>.json` before the next step starts. Kill the process at any point — re-run to resume exactly where it left off.
- **Profile-driven configuration.** Run a subset of steps via `--config profiles/a100_tree_losses --losses kl_tree,bv_tree`. See `orchestration/configs/profiles/README.md` for the full profile catalogue.

---

## 9. File Layout

→ For the current file map, see **PROJECT_CONTEXT.md § Codebase Architecture**.
