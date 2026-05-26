"""
Verification algorithms package — evolved production version of GBV/.

Public API:
    Node           — draft tree node with OTLP solvers (tree.py)
    TreeVerifier   — dispatches all verifier modes (otlp_registry.py)
    speculative_decoding_loop / speculative_decoding_iter  — eval loop (runner.py)

CLI entry point (replaces GBV/main.py — Phase 2 switch):
    python -m algorithms.distillspec_gbv.verifiers.runner --help
"""

from .tree import Node
from .otlp_registry import TreeVerifier
from .runner import speculative_decoding_loop, speculative_decoding_iter

__all__ = [
    "Node",
    "TreeVerifier",
    "speculative_decoding_loop",
    "speculative_decoding_iter",
]
