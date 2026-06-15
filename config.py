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
DEFAULT_SEED           = 123   # eval / inference default; train.py uses SEED=42

# Verifier modes understood by speculative_decoding_loop (main.py)
VERIFIER_MODES = [
    "naive", "nss", "specinfer", "spectr", "khisti",
    "bv", "gbv", "traversal",
]
