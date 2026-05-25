"""
Medusa — Cai et al. 2024 (arXiv:2401.10774).  STUB — not yet implemented.

Core idea:
  Add K extra "Medusa heads" (small linear layers) directly on top of the
  target model.  Each head i predicts the token at position t+i+1 given
  the same hidden state that the target uses for position t.

  No separate draft model is needed — the heads live inside the target.
  At inference: one target forward pass generates K+1 candidates per step.

  Verification: build a draft tree from candidates, verify with target probs.

How to implement:
  1. Read algorithms/medusa/heads.py — MedusaHead nn.Module (stub)
  2. Implement MedusaAlgorithm.setup() — attach heads to the target model
  3. Implement MedusaAlgorithm.generate() — one forward pass + tree verify
  4. Implement MedusaAlgorithm.train() — SFT the heads on the target's own data
  5. Uncomment the import in algorithms/__init__.py

Reference: https://github.com/FasterDecoding/Medusa
"""

from .algorithm import MedusaAlgorithm

ALGORITHM = MedusaAlgorithm()

__all__ = ["ALGORITHM", "MedusaAlgorithm"]
