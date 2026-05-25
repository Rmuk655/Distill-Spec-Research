"""
EAGLE-2 — Li et al. 2024 (arXiv:2406.16858).  STUB — not yet implemented.

Core idea (extends EAGLE):
  Same EAGLE predictor architecture, but with a *dynamic* draft tree instead
  of a fixed tree.  At each decoding step:
    - Score each candidate node with its cumulative acceptance probability.
    - Expand nodes greedily in probability order up to a tree_budget cap.
    - Prune low-confidence branches early.

  This gives a variable-depth, variable-width tree per step, matching the
  target distribution much better than a fixed static tree.

How to implement:
  1. Start from algorithms/eagle/ — the predictor is identical.
  2. Replace the fixed tree builder in generate() with the dynamic-tree algorithm
     from EAGLE-2 Section 3 / Algorithm 1.
  3. No change to training (same regression loss as EAGLE).
  4. Uncomment the import in algorithms/__init__.py.

Reference: https://github.com/SafeAILab/EAGLE  (eagle2 branch)
"""

from .algorithm import Eagle2Algorithm

ALGORITHM = Eagle2Algorithm()

__all__ = ["ALGORITHM", "Eagle2Algorithm"]
