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
    token_ids: Optional[torch.Tensor] = None,  # int64 [T] — teacher-sampled IDs
    **kwargs,                       # absorbs extra CLI args (lr, weight, etc.)
) -> LossOutput:
    """
    Returns:
        LossOutput.loss     — scalar tensor with gradient attached
        LossOutput.aux      — float for logging (e.g. mean accept weight, 0.0 if unused)
    """
    p_s = F.softmax(student_logits.float(), dim=-1).clamp(min=1e-9)
    p_t = F.softmax(teacher_logits.float(), dim=-1).clamp(min=1e-9)
    loss = ...
    return LossOutput(loss=loss, aux=0.0)
```

`LossOutput` is a dataclass from `losses/base.py`:
```python
@dataclass
class LossOutput:
    loss: torch.Tensor    # scalar, grad attached
    aux:  float = 0.0     # logged as "aux_metric" in W&B
```

**Example — alpha divergence (generalises forward/reverse KL):**

```python
def alpha_divergence_loss(student_logits, teacher_logits, token_ids=None, alpha=0.5, **kw):
    p_s = F.softmax(student_logits.float(), dim=-1).clamp(min=1e-9)
    p_t = F.softmax(teacher_logits.float(), dim=-1).clamp(min=1e-9)
    inner = (p_t ** alpha) * (p_s ** (1.0 - alpha))
    loss  = (1.0 - inner.sum(dim=-1)).mean() / (alpha * (1.0 - alpha))
    return LossOutput(loss=loss, aux=0.0)
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
    "my_loss":     my_loss,   # ← register
}
```

The `get_loss(name)` helper in `__init__.py` reads `LOSS_REGISTRY` and raises a
clear error for unknown names — no other registration needed.

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

## Numerical checklist before merging

- [ ] No NaN/Inf in forward pass for random inputs (seeds 0–4)
- [ ] `loss.backward()` succeeds; all gradients are finite
- [ ] `loss.shape == torch.Size([])` — scalar, not shape `[1]`
- [ ] Works at `T = 1` (single-token edge case)
- [ ] `aux` is in `[0, 1]` if it represents a probability, else `0.0`

Quick check (no GPU needed, run from `gbv-research/`):

```bash
python -c "
import torch, torch.nn.functional as F, sys
sys.path.insert(0, '.')
from algorithms.distillspec_gbv.losses.my_loss import my_loss
V, T = 32, 10
s = torch.randn(T, V, requires_grad=True)
t = torch.randn(T, V)
ids = torch.randint(0, V, (T,))
out = my_loss(s, t, ids)
assert out.loss.shape == torch.Size([]), f'Expected scalar, got {out.loss.shape}'
assert torch.isfinite(out.loss), f'NaN/Inf loss: {out.loss}'
out.loss.backward()
assert torch.isfinite(s.grad).all(), 'NaN gradient'
print(f'PASS  loss={out.loss.item():.4f}  aux={out.aux:.4f}')
"
```

Also add a test in `tests/unit/test_losses.py` following the pattern of the
existing loss tests (see `test_forward_kl_basic`, `test_ebe_shape`, etc.).
