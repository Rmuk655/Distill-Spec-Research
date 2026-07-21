import os

_HERE     = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_HERE, "raw")

_KNOWN = {
    "gsm8k_train", "gsm8k_val", "gsm8k_eval",
    "alpaca", "math500", "humaneval", "mtbench",
    "math_train", "math_hard", "math_val", "math_eval",
    "olympiad_eval",   # eval-only — no olympiad_hard/olympiad_val split
    "spec_bench",
    "dapo_math_train",   # training-only — no dapo_val/dapo_eval split, see fetch_dapo_math
}


def get_path(name: str) -> str:
    if name in _KNOWN:
        return os.path.join(_DATA_DIR, f"{name}.jsonl")
    raise KeyError(f"Unknown dataset '{name}'. Known: {sorted(_KNOWN)}")
