"""
Model family registry.

Usage:
    from capsules.distillation.model_families import get_family

    family = get_family("qwen")          # returns QwenFamily instance
    family = get_family("gemma")         # returns GemmaFamily instance

To add a new family:
    1. Create capsules/distillation/model_families/<name>.py
    2. Subclass ModelFamily, implement all abstract methods
    3. Add a module-level FAMILY = MyFamily() instance
    4. Register it below: FAMILY_REGISTRY["<name>"] = ...
"""

from .base import ModelFamily
from .gpt2 import FAMILY as _gpt2
from .llama import FAMILY as _llama
from .qwen import FAMILY as _qwen
from .gemma import FAMILY as _gemma

# ── Registry ──────────────────────────────────────────────────────────────────
# Keys must match the --model_family CLI argument value.
FAMILY_REGISTRY: dict[str, ModelFamily] = {
    "gpt2":  _gpt2,
    "llama": _llama,
    "qwen":  _qwen,
    "gemma": _gemma,
}


def get_family(name: str) -> ModelFamily:
    """
    Retrieve a ModelFamily instance by name.

    Args:
        name: Family identifier, e.g. 'qwen', 'gemma'.

    Returns:
        The corresponding ModelFamily instance.

    Raises:
        ValueError if the name is not registered.
    """
    if name not in FAMILY_REGISTRY:
        available = ", ".join(sorted(FAMILY_REGISTRY))
        raise ValueError(
            f"Unknown model family '{name}'. "
            f"Available: {available}. "
            "To add a new family see docs/ADDING_A_MODEL_FAMILY.md."
        )
    return FAMILY_REGISTRY[name]


__all__ = ["ModelFamily", "FAMILY_REGISTRY", "get_family"]
