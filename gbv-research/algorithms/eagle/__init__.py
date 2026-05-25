"""
EAGLE — Li et al. 2024 (arXiv:2401.15077).  STUB — not yet implemented.

Core idea:
  Train a lightweight "EAGLE predictor" that takes the target LM's last-layer
  hidden state h_t and the current token embedding as input, and autoregressively
  predicts the next hidden state h_{t+1}.  Tokens are then sampled by applying the
  target's own LM head to the predicted hidden state — so no separate vocabulary
  projection is needed.

  At inference: one target forward pass hooks out h_t; the predictor runs K steps
  to produce K draft tokens; a token tree is verified in one more target pass.

How to implement:
  1. Read algorithms/eagle/predictor.py — EaglePredictor nn.Module (stub)
  2. Implement EagleAlgorithm.setup()  — load predictor, register hidden-state hook
  3. Implement EagleAlgorithm.generate() — hook h_t → K predictor steps → tree verify
  4. Implement EagleAlgorithm.train()  — regress predictor onto target hidden states
  5. Uncomment the import in algorithms/__init__.py

Reference: https://github.com/SafeAILab/EAGLE
"""

from .algorithm import EagleAlgorithm

ALGORITHM = EagleAlgorithm()

__all__ = ["ALGORITHM", "EagleAlgorithm"]
