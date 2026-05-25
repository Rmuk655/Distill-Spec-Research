"""
FastEagle — 2025 (arXiv:2509.20416).  STUB — not yet implemented.

Core idea (extends EAGLE / EAGLE-2 / EAGLE-3):
  FastEagle focuses on *inference-time* optimisations to reduce the wall-clock
  latency of EAGLE-style speculative decoding:
    - Batched draft-tree construction using parallelised predictor calls.
    - Sparse tree attention via custom CUDA kernels (avoids materialising the
      full tree attention matrix).
    - Adaptive draft depth: exit early when predictor confidence drops below
      a threshold, saving compute on unlikely continuations.

  The predictor training procedure is the same as EAGLE-3.

How to implement:
  1. Read algorithms/fasteagle/algorithm.py — FastEagleAlgorithm stub.
  2. Implement the batched draft-tree builder (GPU-parallelised).
  3. Integrate sparse tree attention (or use a reference CUDA kernel if available).
  4. Uncomment the import in algorithms/__init__.py.

Reference: https://arxiv.org/abs/2509.20416
"""

from .algorithm import FastEagleAlgorithm

ALGORITHM = FastEagleAlgorithm()

__all__ = ["ALGORITHM", "FastEagleAlgorithm"]
