"""
config.py — shared constants for train.py, eval.py, inference.py.

Source-of-truth files (main.py, verifiers/*.py) are not changed.
Edit here instead of hardcoding the same value in three places.
"""

# Model identifiers
DRAFT_MODEL            = "Qwen/Qwen3-0.6B"
TEACHER_MODEL          = "Qwen/Qwen3-8B"

# Speculative-decoding defaults (tree shape + generation length + temperature)
DEFAULT_K              = 3
DEFAULT_L              = 8
DEFAULT_MAX_NEW_TOKENS = 128
DEFAULT_TEMP           = 1.0
DEFAULT_DTYPE          = "bf16"
DEFAULT_SEED           = 123   # matches main.py default; used by train, eval, and inference

def block_eff(gen_tokens: int, target_calls: int) -> float:
    """Block efficiency = generated tokens / target model calls. Returns nan if no calls."""
    return gen_tokens / target_calls if target_calls > 0 else float("nan")


# BE difficulty thresholds (shared by diagnostics.py and train.py).
#   easy  : BE >= BE_EASY  — draft accepts most of the K-deep tree
#   medium: BE_HARD <= BE < BE_EASY
#   hard  : BE <  BE_HARD  — fewer than 3 tokens accepted per target call
BE_EASY = 6.0
BE_HARD = 3.0

# Verifier modes understood by speculative_decoding_loop (main.py)
VERIFIER_MODES = [
    "naive", "nss", "specinfer", "spectr", "khisti", "max",
    "bv", "gbv", "traversal",
]
