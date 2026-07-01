"""
config.py — shared constants for train.py, eval.py, inference.py.

Source-of-truth files (main.py, verifiers/*.py) are not changed.
Edit here instead of hardcoding the same value in three places.
"""

# Model identifiers (defaults; override per-run with train.py --draft / --teacher)
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


# BE difficulty threshold fractions of L (shared by diagnostics.py and train.py).
# Multiply by the actual L used at runtime — do NOT hardcode absolute values.
#   easy  : BE >= BE_EASY_FRAC * L  (0.875 × 8 = 7.0 at default L=8)
#   medium: BE_HARD_FRAC * L <= BE < BE_EASY_FRAC * L
#   hard  : BE <  BE_HARD_FRAC * L  (0.625 × 8 = 5.0 at default L=8)
# Rationale: trained models on math_hard cluster in 5-7 BE range.
# Old thresholds (6.0/3.0) left hard≈0 always; new thresholds spread 100 val prompts
# across all three buckets so you can see medium→easy and hard→medium transitions.
BE_EASY_FRAC = 0.875
BE_HARD_FRAC = 0.625

# Verifier modes understood by speculative_decoding_loop (main.py)
VERIFIER_MODES = [
    "naive", "nss", "specinfer", "spectr", "khisti", "max",
    "bv", "gbv", "traversal",
]
