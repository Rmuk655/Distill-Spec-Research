"""
Sequoia — Chen et al. 2024 (arXiv:2402.12374).  STUB — not yet implemented.

Core idea:
  Speculative decoding with a *hardware-aware* draft tree.

  Standard speculative decoding uses a fixed-depth chain (K tokens).
  Tree-based methods (SpecInfer, EAGLE) use hand-tuned tree shapes.
  Sequoia *optimises* the tree shape offline for a specific hardware profile
  (memory bandwidth, compute throughput) and a specific draft model.

  At inference:
    - The optimal tree is pre-computed (offline) using dynamic programming
      over acceptance-rate statistics for the draft model.
    - At each decoding step, the draft model fills in the *same* pre-computed
      tree structure; target verifies in one pass.
    - No tree-shape decisions at inference time → very low overhead.

  No additional training needed beyond having any draft model.

How to implement:
  1. Read algorithms/sequoia/tree_optimizer.py — offline DP tree optimizer (stub).
  2. Implement SequoiaAlgorithm.setup()     — load optimal tree topology.
  3. Implement SequoiaAlgorithm.generate()  — fill tree + verify.
  4. Implement SequoiaAlgorithm.build_tree_topology() — offline optimizer wrapper.
  5. Uncomment the import in algorithms/__init__.py.

Reference: https://github.com/Infini-AI-Lab/Sequoia
"""

from .algorithm import SequoiaAlgorithm

ALGORITHM = SequoiaAlgorithm()

__all__ = ["ALGORITHM", "SequoiaAlgorithm"]
