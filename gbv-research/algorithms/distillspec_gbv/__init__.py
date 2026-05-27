"""
DistillSpec + GBV — our primary algorithm.

Paper base:
  Thomas et al., "Generalized Block Verification for Speculative Decoding"
  arXiv:2602.16994v1, 2026.

What this algorithm implements:
  - Draft:        Independent small model (Qwen2.5-0.5B or Qwen3-0.6B)
  - Training:     Knowledge distillation with pluggable losses:
                  forward_kl | reverse_kl | jsd | l1 | ebe (novel)
  - Verification: Tree-based OTLP solvers — gbv | traversal | specinfer |
                  naive | bv | nss | spectr | khisti

Directory layout:
  losses/          Pluggable loss functions (one file per objective)
  verifiers/       All OTLP verification algorithms + tree structures
  trainer.py       Training loop (model-family-agnostic via core/model_families)
  algorithm.py     SpecDecAlgorithm implementation (glues everything together)

Usage:
  from algorithms import get_algorithm
  algo = get_algorithm("gbv")
  algo.setup(target, tokenizer, device, draft_model=draft)
  result = algo.generate(prompt_ids, K=4, L=8, verifier="traversal")
"""

# GBVAlgorithm is available lazily — algorithm.py imports algorithms.base which
# requires the repo root in sys.path.  trainer.py imports only
# distillspec_gbv.losses (not this package directly) so we avoid the eager
# import to keep startup fast and path-agnostic.
__all__ = ["GBVAlgorithm", "ALGORITHM"]


def __getattr__(name: str):
    if name in ("GBVAlgorithm", "ALGORITHM"):
        from .algorithm import GBVAlgorithm as _GBVAlgorithm
        globals()["GBVAlgorithm"] = _GBVAlgorithm
        globals()["ALGORITHM"]    = _GBVAlgorithm()
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
