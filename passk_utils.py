"""Pure, dependency-free helpers for pass@k accuracy grading.

Shared by `scripts/passk_eval.py` (standalone vLLM eval) and `train.py`
(in-training pass@k logged to W&B). No torch / vLLM imports here so it's
importable anywhere and unit-testable without a GPU.

Grading is a NORMALISED STRING MATCH on the last \\boxed{...} (MATH) /
"#### x" (GSM8K) / trailing number — good enough for a *stability diagnostic*
(pass@64 flat across training => coverage didn't collapse) but NOT rigorous
LaTeX equivalence. For a headline accuracy number, swap `is_correct` for
`math_verify`.
"""
import math
import re
from typing import Optional, List

__all__ = ["pass_at_k", "extract_answer", "extract_gold", "is_correct", "normalize"]

# ---- pass@k unbiased estimator (OpenAI HumanEval), numerically stable ----
def pass_at_k(n: int, c: int, k: int) -> float:
    """Prob that >=1 of k uniformly-drawn samples (from n) is correct, given c correct."""
    if k > n:
        return float("nan")          # can't estimate pass@k with fewer than k samples
    if n - c < k:
        return 1.0                    # even the worst draw contains a correct one
    prod = 1.0
    for i in range(n - c + 1, n + 1):
        prod *= (1.0 - k / i)
    return 1.0 - prod


# ---- answer extraction + grading ----
_BOXED = re.compile(r"\\boxed\{")

def _extract_boxed(text: str) -> Optional[str]:
    """Content of the LAST \\boxed{...}, brace-balanced."""
    starts = [m.end() for m in _BOXED.finditer(text)]
    if not starts:
        return None
    i = starts[-1]
    depth, out = 1, []
    while i < len(text) and depth > 0:
        ch = text[i]
        if ch == "{": depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0: break
        out.append(ch); i += 1
    return "".join(out).strip()

def extract_answer(text: str) -> Optional[str]:
    """For model PREDICTIONS (freeform): boxed -> #### -> last number."""
    b = _extract_boxed(text)
    if b is not None:
        return b
    m = re.search(r"####\s*(.+)", text)          # GSM8K format
    if m:
        return m.group(1).strip()
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)  # fallback: last number
    return nums[-1].replace(",", "") if nums else None

def extract_gold(text: str) -> Optional[str]:
    """For GOLD answers: boxed -> #### -> the raw string as-is (NOT last-number,
    which would mangle a bare-LaTeX gold like \\frac{1}{2} into '2')."""
    b = _extract_boxed(text)
    if b is not None:
        return b
    m = re.search(r"####\s*(.+)", text)
    if m:
        return m.group(1).strip()
    return text.strip() or None

def normalize(s: str) -> str:
    s = s.strip()
    for tok in ("\\left", "\\right", "\\!", "\\,", "\\ ", "$", " ", "\n"):
        s = s.replace(tok, "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    if s.endswith("."): s = s[:-1]
    return s

def is_correct(pred: Optional[str], gold: Optional[str]) -> bool:
    # ---- REPLACE with math_verify for a rigorous headline accuracy number ----
    if pred is None or gold is None:
        return False
    return normalize(pred) == normalize(gold)
