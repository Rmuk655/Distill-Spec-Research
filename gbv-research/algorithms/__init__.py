"""
Algorithm registry — one entry per paper.

Usage:
    from algorithms import get_algorithm, list_algorithms

    algo = get_algorithm("gbv")
    algo.setup(target_model, tokenizer, device, draft_model=draft)
    result = algo.generate(prompt_ids, K=4, L=8)

    print(list_algorithms())   # table of all registered algorithms

To add a new algorithm:
    See docs/ADDING_AN_ALGORITHM.md
"""

from .base import SpecDecAlgorithm, AlgorithmMeta, GenerationResult

# ── Import each algorithm implementation ─────────────────────────────────────
# Add a line here when you implement a new algorithm.
# Stub algorithms are imported as comments until they are implemented.

from .distillspec_gbv      import ALGORITHM as _gbv
from .standard_speculative import ALGORITHM as _std
# Uncomment each line below once the stub is fully implemented:
# from .medusa             import ALGORITHM as _medusa      # Cai et al. 2024
# from .eagle              import ALGORITHM as _eagle       # Li et al. 2024
# from .eagle2             import ALGORITHM as _eagle2      # Li et al. 2024
# from .eagle3             import ALGORITHM as _eagle3      # Li et al. 2025
# from .fasteagle          import ALGORITHM as _fasteagle   # 2025
# from .sequoia            import ALGORITHM as _sequoia     # Chen et al. 2024


# ── Registry ──────────────────────────────────────────────────────────────────
# Keys are the --algo CLI argument values used in eval_pipeline.py
ALGORITHM_REGISTRY: dict[str, SpecDecAlgorithm] = {
    "gbv":          _gbv,     # Thomas et al. 2026 — GBV / traversal / OTLP verifiers
    "standard":     _std,     # Leviathan et al. 2022 — token-level accept/reject
    # "medusa":     _medusa,  # Cai et al. 2024 — K heads on target, no draft model
    # "eagle":      _eagle,   # Li et al. 2024 — feature predictor, fixed tree
    # "eagle2":     _eagle2,  # Li et al. 2024 — feature predictor, dynamic tree
    # "eagle3":     _eagle3,  # Li et al. 2025 — mid-layer alignment
    # "fasteagle":  _fasteagle,# 2025 — batched tree + sparse attention
    # "sequoia":    _sequoia, # Chen et al. 2024 — hardware-aware DP tree, no training
}


def get_algorithm(name: str) -> SpecDecAlgorithm:
    """
    Retrieve a SpecDecAlgorithm instance by name.

    Args:
        name: Algorithm key (must be in ALGORITHM_REGISTRY).

    Raises:
        ValueError for unknown names.
    """
    if name not in ALGORITHM_REGISTRY:
        available = ", ".join(sorted(ALGORITHM_REGISTRY))
        raise ValueError(
            f"Unknown algorithm '{name}'. Available: {available}. "
            "See docs/ADDING_AN_ALGORITHM.md to add a new one."
        )
    return ALGORITHM_REGISTRY[name]


def list_algorithms() -> str:
    """Return a formatted table of all registered algorithms."""
    header = f"{'Name':<12} {'Paper':<45} {'arXiv':<15} {'Draft':<12} {'Training':<14}"
    rows   = [header, "-" * len(header)]
    for name, algo in sorted(ALGORITHM_REGISTRY.items()):
        m = algo.meta
        rows.append(
            f"{name:<12} {m.paper_title[:44]:<45} {m.arxiv_id:<15} "
            f"{m.draft_source.name:<12} {m.training.name:<14}"
        )
    # Also show stubs from known literature
    stubs = [
        ("medusa",    "Medusa: Simple LLM Inference (multi-head)",    "2401.10774"),
        ("eagle",     "EAGLE: Feature-based draft, fixed tree",       "2401.15077"),
        ("eagle2",    "EAGLE-2: Feature-based draft, dynamic tree",   "2406.16858"),
        ("eagle3",    "EAGLE-3: Mid-layer intra-model alignment",     "2503.01840"),
        ("fasteagle", "FastEagle: Batched tree + sparse attention",   "2509.20416"),
        ("sequoia",   "Sequoia: Hardware-aware DP tree (no train)",   "2402.12374"),
    ]
    rows.append("")
    rows.append("Stubs (not yet implemented — pull requests welcome):")
    for name, title, arxiv in stubs:
        rows.append(f"  {name:<12} {title:<45} arXiv:{arxiv}")
    return "\n".join(rows)


__all__ = [
    "SpecDecAlgorithm", "AlgorithmMeta", "GenerationResult",
    "ALGORITHM_REGISTRY", "get_algorithm", "list_algorithms",
]
