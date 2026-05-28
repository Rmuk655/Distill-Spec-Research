"""
Distillation loss registry.

Usage:
    from distillspec_gbv.losses import LOSS_REGISTRY, get_loss

    output = get_loss("forward_kl", student_logits, teacher_logits)
    output = get_loss("ebe", student_logits, teacher_logits,
                      token_ids=token_ids, kl_weight=0.1)

Every loss function has the uniform signature:
    fn(student_logits, teacher_logits, token_ids=None, **kwargs) -> LossOutput

To add a new loss:
    1. Create algorithms/distillspec_gbv/losses/<name>.py
    2. Implement a function with the above signature
    3. Register it in LOSS_REGISTRY below
    4. Pass --loss <name> to trainer.py
"""

from .base import LossOutput, compute_loss
from .forward_kl import forward_kl
from .reverse_kl import reverse_kl
from .jsd import jsd
from .l1 import l1
from .ebe import ebe
from .ebe_single import ebe_single

# Tree-structured losses (different interface — see tree_losses.py)
from .tree_losses import (
    kl_tree_loss,
    bv_tree_loss,
    bv_tree_loss_all_paths,
    gbv_tree_loss,
    traversal_tree_loss,
    compute_tree_loss,
    TREE_LOSS_NAMES,
)

# ── Flat-sequence registry (s_log [T,V], t_log [T,V] → LossOutput) ───────────
LOSS_REGISTRY: dict[str, callable] = {
    "forward_kl":  forward_kl,
    "reverse_kl":  reverse_kl,
    "jsd":         jsd,
    "l1":          l1,
    "ebe":         ebe,
    "ebe_single":  ebe_single,
}


def get_loss(name: str, *args, **kwargs) -> LossOutput:
    """
    Compute the named flat-sequence loss.  Convenience wrapper around compute_loss().

    For tree-structured losses (kl_tree, bv_tree, gbv_tree, traversal_tree),
    use compute_tree_loss() instead — they take a different set of arguments.

    Args:
        name:     Loss identifier (key in LOSS_REGISTRY).
        *args:    Positional args forwarded to the loss function.
        **kwargs: Keyword args forwarded (e.g. token_ids, kl_weight).

    Returns:
        LossOutput with .loss and optional .accept_weight.
    """
    if name in TREE_LOSS_NAMES:
        raise ValueError(
            f"'{name}' is a tree-structured loss — call compute_tree_loss() instead of get_loss()."
        )
    if name not in LOSS_REGISTRY:
        available = ", ".join(sorted(LOSS_REGISTRY))
        raise ValueError(
            f"Unknown loss '{name}'. Available flat losses: {available}. "
            f"Tree losses: {sorted(TREE_LOSS_NAMES)}."
        )
    return LOSS_REGISTRY[name](*args, **kwargs)


__all__ = [
    # Flat losses
    "LossOutput", "LOSS_REGISTRY", "compute_loss", "get_loss",
    "forward_kl", "reverse_kl", "jsd", "l1", "ebe", "ebe_single",
    # Tree losses
    "kl_tree_loss", "bv_tree_loss", "bv_tree_loss_all_paths",
    "gbv_tree_loss", "traversal_tree_loss",
    "compute_tree_loss", "TREE_LOSS_NAMES",
]
