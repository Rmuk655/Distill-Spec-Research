"""
EAGLE-3 — Li et al. 2025 (arXiv:2503.01840).  STUB — not yet implemented.

Core idea (extends EAGLE / EAGLE-2):
  EAGLE-3 introduces *intra-model alignment* — the predictor is trained using
  features from an *intermediate* layer of the target LM (not the last layer).
  This removes the need for a forward hook on the output layer and reduces the
  semantic gap between predictor input and target distribution.

  Key contributions vs. EAGLE-2:
    - Predictor reads from a mid-layer hidden state (e.g. layer 16 of a 32-layer LM)
      rather than the final layer output.  This is faster to extract during prefill.
    - Revised training: align predictor output to the target's intermediate features
      rather than final hidden states.
    - Further inference optimisations for long contexts.

How to implement:
  1. Read algorithms/eagle3/algorithm.py — EagleAlgorithm3 stub.
  2. Register a hook on target layer N//2 (configurable) instead of the last layer.
  3. Update training loss to regress onto intermediate features.
  4. Uncomment the import in algorithms/__init__.py.

Reference: https://arxiv.org/abs/2503.01840
"""

from .algorithm import Eagle3Algorithm

ALGORITHM = Eagle3Algorithm()

__all__ = ["ALGORITHM", "Eagle3Algorithm"]
