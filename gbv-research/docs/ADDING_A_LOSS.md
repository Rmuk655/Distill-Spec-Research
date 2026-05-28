# Adding a New Distillation Loss

All loss functions live in `algorithms/distillspec_gbv/losses/` as individual
Python modules. The registry is in `losses/__init__.py`. Adding a new loss is 3 steps.

---

## Step 1. Create the loss module

Add a new file `algorithms/distillspec_gbv/losses/my_loss.py`.

**Required function signature** (matches all existing losses):

```python
from __future__ import annotations
from typing import Optional
import torch
import torch.nn.functional as F
from .base import LossOutput


def my_loss(
    student_logits: torch.Tensor,   # float32  [T, V]  — raw (pre-softmax) logits
    teacher_logits: torch.Tensor,   # float32  [T, V]
    token_ids: Optional[torch.Tensor] = None,  # int64 [T] — generated token IDs
    **kwargs,                       # absorbs extra CLI args (kl_weight, alpha, etc.)
) -> LossOutput:
    """
    Returns:
        LossOutput.loss          — scalar tensor with gradient attached
        LossOutput.accept_weight — mean per-token α = min(1, q/p); set this if
                                   the loss computes acceptance weights, else leave None
        LossOutput.diagnostics   — optional dict with α statistics (EBE-family only);
                                   leave None for losses that don't compute α
    """
    p_s = F.softmax(student_logits.float(), dim=-1).clamp(min=1e-9)
    p_t = F.softmax(teacher_logits.float(), dim=-1).clamp(min=1e-9)
    loss = ...
    return LossOutput(loss=loss)
```

`LossOutput` is a dataclass from `losses/base.py`:

```python
@dataclass
class LossOutput:
    loss:          torch.Tensor           # scalar, grad attached
    accept_weight: Optional[float] = None # mean α = min(1, q/p); EBE-family sets this
    diagnostics:   Optional[dict]  = None # α stats: {"mean","std","min",
                                          #           "frac_lt_0.95","frac_lt_0.80"}
                                          # EBE-family only; other losses leave None
```

- **`loss`**: always required — the scalar you back-prop through.
- **`accept_weight`**: set to `alpha.mean().item()` if your loss computes per-token
  acceptance weights; leave `None` otherwise. The training loop uses this for logging
  and the dashboard displays it as the "mean α" curve.
- **`diagnostics`**: optional richer α breakdown. Only EBE-family losses populate this;
  leave `None` for KL / JSD / L1 style losses.

**Example — alpha divergence (generalises forward/reverse KL):**

```python
def alpha_divergence_loss(student_logits, teacher_logits, token_ids=None, alpha=0.5, **kw):
    p_s = F.softmax(student_logits.float(), dim=-1).clamp(min=1e-9)
    p_t = F.softmax(teacher_logits.float(), dim=-1).clamp(min=1e-9)
    inner = (p_t ** alpha) * (p_s ** (1.0 - alpha))
    loss  = (1.0 - inner.sum(dim=-1)).mean() / (alpha * (1.0 - alpha))
    return LossOutput(loss=loss)   # no accept_weight — this loss doesn't compute α
```

---

## Step 2. Register in LOSS_REGISTRY

Edit `algorithms/distillspec_gbv/losses/__init__.py` and add two lines:

```python
from .my_loss import my_loss   # ← import

LOSS_REGISTRY = {
    "forward_kl":  forward_kl,
    "reverse_kl":  reverse_kl,
    "jsd":         jsd,
    "l1":          l1,
    "ebe":         ebe,
    "ebe_single":  ebe_single,
    "my_loss":     my_loss,   # ← register
}
```

The `get_loss(name)` helper in `__init__.py` reads `LOSS_REGISTRY` and raises a
clear error for unknown names — no other registration needed.

Also add `"my_loss"` to the `choices=` list in `algorithms/distillspec_gbv/trainer.py`
so the CLI accepts it:

```python
parser.add_argument("--loss", default="forward_kl",
    choices=["forward_kl", "reverse_kl", "jsd", "l1", "ebe", "ebe_single", "my_loss"])
```

---

## Step 3. Add to pipeline

Open `orchestration/experiment.py` and add a **train + merge + eval** triple to
`build_steps()`, following the pattern of the existing losses. Copy the
`train_jsd_gsm8k` block and replace `jsd` / `jsd-gsm8k` with your loss name.

```python
# In build_steps() — Phase 2 Training block:
{
    "id": "train_my_loss_gsm8k",
    "group": "Phase 2 — Training",
    "desc": f"Train my_loss, {_steps} steps, gsm8k_train",
    "cmd": [
        sys.executable, _TRAIN_SCRIPT,
        "--loss", "my_loss",
        "--steps", str(_steps), "--lr", "3e-5",
        "--nan_action", "skip", "--early_stop_patience", "3",
        "--draft", draft, "--target", target,
        "--dataset", _data("gsm8k_train.jsonl"),
        "--output", _ckpt("my-gsm8k"),
    ],
    "done_check": os.path.join(_ckpt("my-gsm8k"), "adapter_model.safetensors"),
    "retryable": True,
},
{
    "id": "merge_my_loss_gsm8k",
    "group": "Phase 2 — Training",
    "desc": "Merge my-gsm8k LoRA",
    "cmd": [sys.executable, _TRAIN_SCRIPT, "--merge_only",
            "--adapter", _ckpt("my-gsm8k"), "--draft", draft],
    "done_check": os.path.join(_merged("my-gsm8k"), "config.json"),
},

# Phase 3 GSM8K Eval:
{
    "id": "eval_my_loss_gsm8k",
    "group": "Phase 3 — GSM8K Eval",
    "desc": "Eval my-gsm8k on gsm8k",
    "cmd": _ec(_merged("my-gsm8k"), "my_loss", datasets="gsm8k", task_score=True),
    "done_check": None,
},

# Phase 4 Multi-Dataset (add smoke_skip=smoke):
{
    "id": "eval_my_loss_all",
    "group": "Phase 4 — Multi-Dataset",
    "desc": "Eval my_loss on humaneval,math500,mtbench,alpaca",
    "cmd": _ec(_merged("my-gsm8k"), "my_loss",
               datasets="humaneval,math500,mtbench,alpaca", task_score=True),
    "done_check": None,
    "smoke_skip": smoke,
},
```

Also add the four step IDs to `clean_restart.py`'s `_ALL_STEP_IDS` list so
clean restarts reset them correctly.

---

---

## Adding a Tree-Structured Loss

Tree losses differ fundamentally from flat losses: they train on a **K-path draft tree sampled
from the student's own distribution** rather than on the teacher's linear rollout.  This
eliminates the off-policy mismatch that causes flat EBE numbers to be poor (see DESIGN.md §9
for the full theory).

### Why a separate code path?

Flat losses (`forward_kl`, `ebe`, …) operate on `[T, V]` logit tensors produced by a
single teacher forward pass.  Tree losses operate on **two dicts keyed by node prefix**:

```
q_probs_dict[prefix]  — student probabilities at that node  (with grad)
p_probs_dict[prefix]  — teacher probabilities at that node  (detached)
q_paths               — list of K full token paths (list of lists)
```

The pipeline uses three serial, no-overlap phases to avoid materialising a tree with
gradients through the draft model twice:

```
Phase A  iid_draft(draft_model, prompt, K, L, no_grad)
          → q_paths: K independent L-token paths (no gradient needed)

Phase B  target_tree_pass(target_model, q_paths, no_grad)
          → p_probs_dict: teacher logit dicts for every tree node

Phase C  draft_tree_forward_with_grad(draft_model, q_paths)
          → q_probs_dict: student logit dicts WITH gradient

          compute_tree_loss(loss_name, q_probs_dict, p_probs_dict, q_paths, L, K)
          → scalar loss → .backward() → optimizer.step()
```

Phase A is no-grad so Phase C's autograd graph stays clean.  Phase B is always no-grad
(target is frozen).

---

### Step 1. Write the tree loss function

Add your function to `algorithms/distillspec_gbv/losses/tree_losses.py`.

**Required signature**:

```python
def my_tree_loss(
    q_probs_dict: dict,   # {prefix: student prob tensor [V]}  WITH grad
    p_probs_dict: dict,   # {prefix: teacher prob tensor [V]}  detached
    q_paths:      list,   # [[tok, tok, ...], ...]  K paths of length L
    L: int,               # tree depth
    K: int,               # number of paths
    **kwargs,             # kl_weight, skew_alpha, …
) -> torch.Tensor:        # scalar loss WITH gradient
```

**Node iteration pattern** (same in all tree losses):

```python
for path in q_paths:
    for depth in range(1, L + 1):
        prefix = ",".join(str(x) for x in path[:depth])
        token  = path[depth]               # the token sampled at this node
        p = p_probs_dict[prefix].detach()  # teacher probs [V]
        q = q_probs_dict[prefix]           # student probs [V], WITH grad
        # ... compute your per-node loss contribution ...
```

**Numeric guards** (mandatory for Qwen3):

```python
# Qwen3 sets -inf logits on forbidden tokens; clamp before ratio computations
p = p_probs_dict[prefix].detach().clamp(min=0.0)
q = q_probs_dict[prefix].clamp(min=1e-9)
# For log quantities clamp log(q) from below:
log_q = torch.log(q)  # already guarded by clamp above
log_p = torch.log(p.clamp(min=1e-9))
```

**Reference: existing tree losses and their theory**

| Name | Formula | Behaviour |
|---|---|---|
| `kl_tree` | KL(p ∥ q) at each node | Mode-covering; universal baseline |
| `rev_kl_tree` | KL(q ∥ p) at each node | Mode-seeking; draft concentrates on target peaks |
| `jsd_tree` | ½KL(q∥m)+½KL(p∥m), m=½(q+p) | Symmetric; bounded [0, log 2]; numerically softer |
| `bv_tree` | −E[block accept] under BV verifier | Directly optimises BV acceptance integral |
| `gbv_tree` | −E[block accept] under GBV w/ q-skew | Optimises GBV acceptance; numerically stable ≤ K=4 |
| `traversal_tree` | −E[accept] under traversal leaf-weight product | Best match to traversal verifier |
| `ebe_tree` | −(1 + Σ cumprod(α)) on student's own paths | On-policy EBE; ablates off-policy mismatch in flat EBE |

---

### Step 2. Register in `TREE_LOSS_NAMES` and dispatch

In `algorithms/distillspec_gbv/losses/tree_losses.py`:

```python
# 1. Add to the frozenset
TREE_LOSS_NAMES = frozenset({
    "kl_tree", "rev_kl_tree", "jsd_tree",
    "bv_tree", "gbv_tree", "traversal_tree",
    "ebe_tree",
    "my_tree",   # ← add here
})

# 2. Add a branch in compute_tree_loss()
def compute_tree_loss(loss_name, q_probs_dict, p_probs_dict, q_paths, L, K, **kw):
    ...
    elif loss_name == "my_tree":
        return my_tree_loss(q_probs_dict, p_probs_dict, q_paths, L, K, **kw)
    ...
```

Also add `"my_tree"` to the `choices=` list in `algorithms/distillspec_gbv/trainer.py`
(look for the `# Tree-structured loss names` comment block).

---

### Step 3. Add to pipeline

In `orchestration/experiment.py`:

**3a. Add prefix tuple** to `_LOSS_STEP_PREFIXES`:
```python
"my_tree": ("train_my_tree_", "merge_my_tree_", "eval_my_tree_"),
```

**3b. Register smoke verifier pairing** in `_TREE_PAIRED`:
```python
# universal: evaluated against all 3 non-OT verifiers even in smoke
"my_tree":  _TREE_NON_OT,

# or targeted: only one verifier in smoke, all 3 in full
"my_tree":  "bv",  # smoke uses BV; full evals add GBV + traversal
```
The `_TREE_NON_OT` constant is `"bv,gbv,traversal"`.  OT-based verifiers (`specinfer`,
`naive`) are incompatible with tree losses and are never included.

**3c. Add train / merge / eval steps**:

```python
# Phase 2 — Training:
{
    "id":   "train_my_tree_gsm8k",
    "group": "Phase 2 — Training",
    "desc": f"Train my_tree, {_steps} steps, gsm8k_train",
    "cmd":  [sys.executable, _TRAIN_SCRIPT,
             "--loss", "my_tree", "--tree_K", str(_tree_K), "--tree_L", str(_tree_L),
             "--steps", str(_steps), "--lr", "3e-5",
             "--draft", draft, "--target", target,
             "--dataset", _data("gsm8k_train.jsonl"),
             "--output", _ckpt("my-tree-gsm8k")],
    "done_check": os.path.join(_ckpt("my-tree-gsm8k"), "adapter_model.safetensors"),
    "retryable": True,
},
{
    "id":   "merge_my_tree_gsm8k",
    "group": "Phase 2 — Training",
    "desc": "Merge my-tree-gsm8k LoRA",
    "cmd":  [sys.executable, _TRAIN_SCRIPT, "--merge_only",
             "--adapter", _ckpt("my-tree-gsm8k"), "--draft", draft],
    "done_check": os.path.join(_merged("my-tree-gsm8k"), "config.json"),
},

# Phase 3 — GSM8K Eval (non-OT verifiers only):
{
    "id":   "eval_my_tree_gsm8k",
    "group": "Phase 3 — GSM8K Eval",
    "desc": "Eval my-tree-gsm8k on gsm8k",
    "cmd":  _ec(_merged("my-tree-gsm8k"), "my_tree",
                modes=_TREE_PAIRED.get("my_tree", _TREE_NON_OT), task_score=True),
    "done_check": None,
    "smoke_skip": smoke and "my_tree" not in _TREE_PAIRED,
},

# Phase 4 — Multi-Dataset:
{
    "id":   "eval_my_tree_all",
    "group": "Phase 4 — Multi-Dataset",
    "desc": "Eval my_tree on humaneval,math500,mtbench,alpaca",
    "cmd":  _ec(_merged("my-tree-gsm8k"), "my_tree",
                datasets="humaneval,math500,mtbench,alpaca",
                modes=_TREE_NON_OT, task_score=True),
    "done_check": None,
    "smoke_skip": smoke,
},
```

Note: **do not** add `my_tree` to the flat-loss online training blocks.  Online tree
training uses a separate code path (`--tree_loss` in `online_serve.py`).

---

### Numerical checklist for tree losses

- [ ] No NaN/Inf in forward pass for K=2, L=8, V=32 (tiny vocab smoke test)
- [ ] `loss.backward()` succeeds; all student-path gradients are finite
- [ ] Loss is a scalar (not shape `[1]` or `[K]`)
- [ ] `p_probs_dict` entries are `.detach()`-ed before any ratio or log operation
- [ ] Qwen3 −∞ logit guard applied (clamp before log/ratio)
- [ ] Tested at K=1 (degenerate single-path edge case)

Quick smoke test (CPU, no GPU needed):

```bash
python -c "
import torch, sys
sys.path.insert(0, '.')
from algorithms.distillspec_gbv.losses.tree_losses import compute_tree_loss, TREE_LOSS_NAMES
V, L, K = 64, 4, 2
# Build synthetic K-path tree
q_paths = [[torch.randint(0, V, (1,)).item() for _ in range(L)] for _ in range(K)]
q_probs = {}; p_probs = {}
for path in q_paths:
    for d in range(1, L + 1):
        pref = ','.join(str(x) for x in path[:d])
        if pref not in q_probs:
            q_raw = torch.randn(V, requires_grad=True)
            p_raw = torch.randn(V).detach()
            q_probs[pref] = torch.softmax(q_raw, dim=-1)
            p_probs[pref] = torch.softmax(p_raw, dim=-1)
loss = compute_tree_loss('my_tree', q_probs, p_probs, q_paths, L, K)
assert loss.shape == torch.Size([]), f'Expected scalar, got {loss.shape}'
assert torch.isfinite(loss), f'NaN/Inf loss: {loss}'
loss.backward()
print(f'PASS  loss={loss.item():.4f}')
"
```

---

## Numerical checklist before merging

- [ ] No NaN/Inf in forward pass for random inputs (seeds 0–4)
- [ ] `loss.backward()` succeeds; all gradients are finite
- [ ] `loss.shape == torch.Size([])` — scalar, not shape `[1]`
- [ ] Works at `T = 1` (single-token edge case)
- [ ] If `accept_weight` is populated, it is in `[0, 1]`; otherwise it is `None`

Quick check (no GPU needed, run from `gbv-research/`):

```bash
python -c "
import torch, sys
sys.path.insert(0, '.')
from algorithms.distillspec_gbv.losses.my_loss import my_loss
from algorithms.distillspec_gbv.losses import LossOutput
V, T = 32, 10
s = torch.randn(T, V, requires_grad=True)
t = torch.randn(T, V)
ids = torch.randint(0, V, (T,))
out = my_loss(s, t, ids)
assert isinstance(out, LossOutput), f'Expected LossOutput, got {type(out)}'
assert out.loss.shape == torch.Size([]), f'Expected scalar, got {out.loss.shape}'
assert torch.isfinite(out.loss), f'NaN/Inf loss: {out.loss}'
out.loss.backward()
assert torch.isfinite(s.grad).all(), 'NaN gradient'
aw_str = f'{out.accept_weight:.4f}' if out.accept_weight is not None else 'None'
print(f'PASS  loss={out.loss.item():.4f}  accept_weight={aw_str}')
"
```

Also add a test in `tests/unit/test_losses.py` following the pattern of the
existing loss tests (see `TestForwardKL`, `TestEBE`, `TestEBESingle`, etc.).
