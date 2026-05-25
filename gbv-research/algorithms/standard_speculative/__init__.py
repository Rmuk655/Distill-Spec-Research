"""
Standard Speculative Decoding — Leviathan et al. 2022 (arXiv:2211.17192).

The original algorithm:
  - Draft:        Independent small model generates K tokens autoregressively
  - Verification: Token-level accept / reject sampling (no tree, K=1 path)
  - Training:     None — works with any pre-trained draft model

This is the baseline all other methods are compared against.
Block efficiency is 1 + α * (1 + α * ...) where α is the per-token
acceptance rate.  Tree methods (GBV, EAGLE-2) improve over this.
"""

from .algorithm import StandardSpecAlgorithm

ALGORITHM = StandardSpecAlgorithm()

__all__ = ["ALGORITHM", "StandardSpecAlgorithm"]
