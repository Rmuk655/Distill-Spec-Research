# Adding a New Distillation Loss

All loss functions live in `OSD/train_qwen3.py` and are dispatched via
a `LOSS_REGISTRY` dict. Adding a new loss is 3 steps.

---

## Step 1. Implement the loss function

Add your function to `OSD/train_qwen3.py` alongside the existing losses
(`forward_kl_loss`, `ebe_loss`, `reverse_kl_loss`, `jsd_loss`, `l1_loss`).

**Required signature:**

```python
def my_loss(
    student_logits: torch.Tensor,   # float32  [T, V]  — raw (pre-softmax) logits
    teacher_logits: torch.Tensor,   # float32  [T, V]
    token_ids: torch.Tensor,        # int64    [T]     — sampled token IDs
    **kwargs,                       # absorbs extra CLI args (lr, weight, etc.)
) -> tuple[torch.Tensor, float]:
    """
    Returns:
        loss           — scalar tensor with gradient attached
        aux_metric     — float for logging (e.g. mean accept weight, 0.0 if unused)
    """
    ...
    return loss, aux_metric
```

Contract:
- `student_logits` and `teacher_logits` are **float32**, shape `[T, V]`
  (T = sequence length, V = vocab size = 151,936 for Qwen).
- `token_ids` are the teacher-sampled token IDs; needed for NLL-based losses (like EBE).
- `loss` must be a **scalar tensor** (shape `torch.Size([])`) with `.requires_grad = True`.
- `aux_metric` is any float you want logged to W&B per step (e.g. mean acceptance weight).
  Pass `0.0` if not applicable.
- All computations must stay on the **same device** as the input tensors.
- No side effects — the training loop calls this once per step.

**Example — alpha divergence (generalises forward/reverse KL):**

```python
def alpha_divergence_loss(student_logits, teacher_logits, token_ids, alpha=0.5, **kw):
    p_s = F.softmax(student_logits.float(), dim=-1).clamp(min=1e-9)
    p_t = F.softmax(teacher_logits.float(), dim=-1).clamp(min=1e-9)
    inner = (p_t ** alpha) * (p_s ** (1.0 - alpha))
    loss  = (1.0 - inner.sum(dim=-1)).mean() / (alpha * (1.0 - alpha))
    return loss, 0.0
```

---

## Step 2. Register in LOSS_REGISTRY

Find the `LOSS_REGISTRY` dict in `OSD/train_qwen3.py` and add your entry:

```python
LOSS_REGISTRY = {
    "forward_kl":       forward_kl_loss,
    "ebe":              ebe_loss,
    "reverse_kl":       reverse_kl_loss,
    "jsd":              jsd_loss,
    "l1":               l1_loss,
    "alpha_divergence": alpha_divergence_loss,   # ← add this line
}
```

---

## Step 3. Add to pipeline

Open `orchestration/pipeline.py` and add a train+merge pair to `build_steps()`,
following the exact same pattern as the existing losses (copy the `train_jsd_gsm8k` block):

```python
{
    "id": "train_alpha_gsm8k",
    "group": "Phase 2 — Training",
    "desc": f"Train alpha_divergence, {_steps} steps, gsm8k_train",
    "cmd": [
        sys.executable, _TRAIN_SCRIPT,
        "--loss", "alpha_divergence",
        "--steps", str(_steps), "--lr", "3e-5",
        "--nan_action", "skip", "--early_stop_patience", "3",
        "--draft", draft, "--target", target,
        "--dataset", _data("gsm8k_train.jsonl"),
        "--output", _ckpt("alpha-gsm8k"),
    ],
    "done_check": os.path.join(_ckpt("alpha-gsm8k"), "adapter_model.safetensors"),
    "retryable": True,
},
{
    "id": "merge_alpha_gsm8k",
    "group": "Phase 2 — Training",
    "desc": "Merge alpha-gsm8k LoRA",
    "cmd": [sys.executable, _TRAIN_SCRIPT, "--merge_only",
            "--adapter", _ckpt("alpha-gsm8k"), "--draft", draft],
    "done_check": os.path.join(_merged("alpha-gsm8k"), "config.json"),
},
```

Then add a Phase 3 eval step (copy `eval_jsd_gsm8k`):

```python
{
    "id": "eval_alpha_gsm8k",
    "group": "Phase 3 — GSM8K Eval",
    "desc": "Eval alpha-gsm8k on gsm8k",
    "cmd": _ec(_merged("alpha-gsm8k"), "alpha", datasets="gsm8k", task_score=True),
    "done_check": None,
},
```

And a Phase 4 multi-dataset eval step (copy `eval_jsd_all`):

```python
{
    "id": "eval_alpha_all",
    "group": "Phase 4 — Multi-Dataset",
    "desc": "Eval alpha on humaneval,math500,mtbench,alpaca",
    "cmd": _ec(_merged("alpha-gsm8k"), "alpha",
               datasets="humaneval,math500,mtbench,alpaca", task_score=True),
    "done_check": None,
    "smoke_skip": smoke,
},
```

Also add the step IDs to `clean_restart.py`'s `_ALL_STEP_IDS` list so clean
restarts reset them correctly.

Finally update `OSD/viz_server.py`'s `STEP_ORDER` and `PHASE_LABELS` dicts
to include the new step IDs so the dashboard shows them.

---

## Numerical checklist before merging

- [ ] No NaN/Inf in forward pass for a random input (seed 0–4)
- [ ] `loss.backward()` succeeds; `loss.grad` is finite for all parameters
- [ ] `loss.shape == torch.Size([])` — must be a 0-dim scalar, not shape `[1]`
- [ ] Works when `T = 1` (teacher generates a single token — edge case)
- [ ] `aux_metric` is in `[0, 1]` if it represents a probability (or `0.0`)

Quick check (no GPU needed):

```bash
cd OSD
python -c "
import torch, torch.nn.functional as F
from train_qwen3 import alpha_divergence_loss   # replace with your loss name
V, T = 32, 10
s = torch.randn(T, V, requires_grad=True)
t = torch.randn(T, V)
ids = torch.randint(0, V, (T,))
loss, aux = alpha_divergence_loss(s, t, ids)
assert loss.shape == torch.Size([]), f'Expected scalar, got {loss.shape}'
assert torch.isfinite(loss), f'NaN/Inf loss: {loss}'
loss.backward()
assert torch.isfinite(s.grad).all(), 'NaN gradient'
print(f'PASS  loss={loss.item():.4f}  aux={aux:.4f}')
"
```
