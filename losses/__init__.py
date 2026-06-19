"""
losses/__init__.py — single entry point for all losses.

Two registries:
    FLAT_LOSSES   : name → fn(student_logits, teacher_logits)
    TREE_LOSSES   : name → fn(q_probs_dict, p_probs_dict, q_paths, L, K)

train.py looks at which registry --loss appears in and calls accordingly.
This keeps the training loop branch-free except for one if-statement.
"""
from .flat import FLAT_LOSSES
from .tree import TREE_LOSSES

ALL_LOSSES = {**FLAT_LOSSES, **TREE_LOSSES}

# Off-policy tree losses use the teacher's greedy path as the training sequence
# instead of draft samples.  They share the same loss formula as their on-policy
# counterparts but are routed through compute_offpolicy_tree_loss() in train.py.
OFFPOLICY_TREE_LOSSES = {"op_naive_tree", "op_naive_tree_full"}


def is_tree_loss(name: str) -> bool:
    """Return True if the loss is computed on a draft tree rather than flat logits."""
    return name in TREE_LOSSES


def is_offpolicy_tree_loss(name: str) -> bool:
    """Return True if the loss uses the teacher's path rather than draft samples."""
    return name in OFFPOLICY_TREE_LOSSES


def get_loss(name: str):
    """Look up a loss function by name.  Raises KeyError with a helpful message."""
    if name in FLAT_LOSSES:
        return FLAT_LOSSES[name]
    if name in TREE_LOSSES:
        return TREE_LOSSES[name]
    raise KeyError(
        f"Unknown loss '{name}'.  Available: "
        f"flat={sorted(FLAT_LOSSES)}, tree={sorted(TREE_LOSSES)}"
    )
