# Adding a New Distillation Loss

This guide shows how to add a new loss objective to the training framework.

## Step-by-step

### 1. Create a file in `capsules/distillation/losses/`

Name it after the loss (e.g. `alpha_divergence.py`):

```python
"""
Alpha-divergence loss  D_α(p_teacher ∥ p_student).

Generalises KL (α→1) and reverse-KL (α→0).
Useful ablation for studying the mode-covering/mode-seeking trade-off.
"""
from __future__ import annotations
from typing import Optional
import torch
import torch.nn.functional as F
from .base import LossOutput


def alpha_divergence(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    token_ids: Optional[torch.Tensor] = None,   # unused by this loss
    alpha: float = 0.5,
    **_kwargs,
) -> LossOutput:
    """
    D_α(p_t ∥ p_s) = (1 / (α*(1-α))) * (1 - Σ_v p_t[v]^α * p_s[v]^(1-α))

    Args:
        student_logits: Float32 [T, V]
        teacher_logits: Float32 [T, V]
        alpha:          Divergence parameter in (0, 1).
                        α→1: approaches KL(p_t ∥ p_s) (forward KL)
                        α→0: approaches KL(p_s ∥ p_t) (reverse KL)

    Returns:
        LossOutput with scalar loss.
    """
    p_s = F.softmax(student_logits, dim=-1).clamp(min=1e-9)
    p_t = F.softmax(teacher_logits, dim=-1).clamp(min=1e-9)

    # Hellinger-type inner product
    inner = (p_t ** alpha) * (p_s ** (1.0 - alpha))
    loss  = (1.0 - inner.sum(dim=-1)).mean() / (alpha * (1.0 - alpha))
    return LossOutput(loss=loss)
```

### 2. Register in `losses/__init__.py`

```python
from .alpha_divergence import alpha_divergence

LOSS_REGISTRY = {
    "forward_kl":       forward_kl,
    "reverse_kl":       reverse_kl,
    "jsd":              jsd,
    "l1":               l1,
    "ebe":              ebe,
    "alpha_divergence": alpha_divergence,   # ← add this
}
```

### 3. Add to trainer CLI choices

In `capsules/distillation/trainer.py`, update the `--loss` argument:

```python
p.add_argument("--loss", default="forward_kl",
               choices=["forward_kl", "reverse_kl", "jsd", "l1", "ebe",
                        "alpha_divergence"],   # ← add here
               ...)
```

### 4. Smoke test

```bash
python -m capsules.distillation.trainer \
    --loss alpha_divergence --steps 10 \
    --dataset capsules/datasets/raw/gsm8k_30.jsonl \
    --no_wandb
```

Verify: finite loss for all 10 steps, gradient flows (no NaN).

## Function signature contract

Every loss function **must** follow this signature exactly:

```python
def my_loss(
    student_logits: torch.Tensor,   # float32 [T, V]
    teacher_logits: torch.Tensor,   # float32 [T, V]
    token_ids: Optional[torch.Tensor] = None,  # int64 [T] (optional)
    **kwargs,                        # extra hyperparams
) -> LossOutput:
    ...
    return LossOutput(loss=scalar_tensor, accept_weight=optional_float)
```

- `student_logits` and `teacher_logits` are raw (pre-softmax) logits in float32.
- `token_ids` is only required by EBE; include it in signature with default=None.
- `**kwargs` absorbs unknown keyword args (e.g. `kl_weight`, `alpha`) passed from CLI.
- Return `LossOutput(loss=...)` where `loss` is a scalar tensor with gradient.
- Return `LossOutput(loss=..., accept_weight=float)` only if your loss naturally
  computes per-token acceptance probabilities (like EBE does).

## Numerical checklist

Before merging a new loss, verify:
- [ ] No NaN/Inf in forward pass for any typical input
- [ ] Gradient flows: `loss.backward()` succeeds without NaN in `grad`
- [ ] Valid range: loss values are in a sensible range for the model pair
- [ ] EOS / short sequence: works when `T = 1` (teacher generates 1 token)
