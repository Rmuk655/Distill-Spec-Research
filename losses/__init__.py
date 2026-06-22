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

# Enrichment losses build the tree from the TEACHER's own sampled branches
# (alternative continuations) and pull the draft toward them with JSD at every
# node.  Routed through compute_enrichment_loss() in train.py.  K=1 is the
# matched single-path control; K>1 is the enrichment ("teach additional paths").
ENRICHMENT_LOSSES = {"jsd_enrich"}

# Flat enrichment losses: same per-token JSD as jsd flat but the teacher
# samples K stochastic continuations (do_sample=True) instead of one greedy
# rollout.  K=1 isolates greedy-vs-stochastic; K>1 adds path diversity.
# Apples-to-apples with jsd flat: same L=128, only teacher sampling changes.
# Routed through compute_flat_enrich_loss() in train.py.
FLAT_ENRICH_LOSSES = {"jsd_flat_enrich"}


def is_tree_loss(name: str) -> bool:
    """Return True if the loss is computed on a draft tree rather than flat logits."""
    return name in TREE_LOSSES


def is_offpolicy_tree_loss(name: str) -> bool:
    """Return True if the loss uses the teacher's path rather than draft samples."""
    return name in OFFPOLICY_TREE_LOSSES


def is_enrichment_loss(name: str) -> bool:
    """Return True if the tree branches are sampled from the teacher, not the draft."""
    return name in ENRICHMENT_LOSSES


def is_flat_enrich_loss(name: str) -> bool:
    """Return True if the loss uses K stochastic teacher flat rollouts (not a tree)."""
    return name in FLAT_ENRICH_LOSSES


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

# Maps each tree loss to the verifier mode used for val block_eff measurement.
# Losses not listed fall through to "traversal" (best general-purpose verifier).
LOSS_TO_VERIFIER = {
    "naive_tree":         "naive",
    "naive_tree_full":    "naive",
    "op_naive_tree":      "naive",
    "op_naive_tree_full": "naive",
    "nss_tree":      "nss",
    "specinfer_tree":"specinfer",
    "spectr_tree":   "spectr",
    "khisti_tree":   "khisti",
    "bv_tree":       "bv",
    "gbv_tree":      "gbv",
    "traversal_tree":"traversal",
}