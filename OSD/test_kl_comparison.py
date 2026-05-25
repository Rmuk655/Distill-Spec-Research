"""
test_kl_comparison.py — Compare all KL distillation variants + online OSD

Runs acceptance-rate evaluation across all trained models and prints a
ranked comparison table. Proves (or disproves) the hypothesis that:
  forward_kl >= reverse_kl ≈ jsd >= baseline  (in-distribution)
  online >= offline forward_kl >= baseline     (online continually adapts)

Usage:
    python test_kl_comparison.py [--n 20] [--dataset data/gsm8k_test.jsonl]
                                  [--K 3] [--T 0.6] [--max_tokens 128]

Expected improvement per variant:
  forward_kl : Mode-covering — assigns prob to all target tokens, best acceptance
  reverse_kl : Mode-seeking — collapses to draft's confident tokens, lower acceptance
  jsd        : Symmetric blend — should land between forward and reverse KL
  ebe        : Expected block efficiency loss — directly optimizes acceptance, best overall
  online     : Adapts to actual inference queries — should match or exceed offline forward_kl
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Model configs — map variant name to merged checkpoint path.
# Paths are relative to this file's parent directory (OSD/).
# Update these to match wherever your training runs saved checkpoints.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent

MODEL_CONFIGS: dict[str, dict] = {
    "baseline": {
        "q_model": "Qwen/Qwen2.5-0.5B",
        "description": "Untrained draft (Qwen2.5-0.5B) — reference point",
    },
    "forward_kl": {
        "q_model": str(_HERE.parent / "checkpoints" / "kl1000-gsm8k_merged"),
        "description": "Forward KL (KL(target‖draft)) — DistillSpec default, mode-covering",
    },
    "reverse_kl": {
        "q_model": str(_HERE.parent / "checkpoints" / "rev_kl1000-gsm8k_merged"),
        "description": "Reverse KL (KL(draft‖target)) — mode-seeking, collapses to top-1",
    },
    "jsd": {
        "q_model": str(_HERE.parent / "checkpoints" / "jsd1000-gsm8k_merged"),
        "description": "JSD — symmetric blend, bounded [0, log 2]",
    },
    "ebe": {
        "q_model": str(_HERE.parent / "checkpoints" / "ebe1000-gsm8k_merged"),
        "description": "EBE — directly optimizes block efficiency (block-level cumprod loss)",
    },
    "online": {
        "q_model": str(_HERE.parent / "checkpoints" / "online-gsm8k_merged"),
        "description": "Online OSD — continually adapts to live inference queries",
    },
}

# Default target model
_DEFAULT_TARGET = "Qwen/Qwen3-0.6B"

# Path to GBV/main.py relative to this file's parent
_GBV_MAIN = _HERE.parent / "GBV" / "main.py"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _checkpoint_exists(path: str) -> bool:
    """Return True if the checkpoint path is either a HF hub ID or an existing local dir."""
    if not path.startswith("/") and not path.startswith(".") and not os.sep in path:
        # Looks like a HuggingFace hub ID (e.g. "Qwen/Qwen2.5-0.5B")
        return True
    return Path(path).exists()


def run_gbv_eval(
    variant_name: str,
    q_model: str,
    p_model: str,
    data: str,
    K: int,
    T: float,
    max_tokens: int,
    mode: str = "gbv",
    seed: int = 42,
) -> dict:
    """
    Call GBV/main.py via subprocess for a single (variant, config) pair.
    Returns a dict with keys: block_eff, throughput, ms_per_token, returncode, stderr_tail.
    """
    cmd = [
        sys.executable,
        str(_GBV_MAIN),
        "--p_model", p_model,
        "--q_model", q_model,
        "--data", data,
        "--mode", mode,
        "--K", str(K),
        "--p_temp", str(T),
        "--max_new_tokens", str(max_tokens),
        "--seed", str(seed),
        "--device", "cpu",   # fall back to CPU so smoke tests work without a GPU
    ]

    print(f"  [RUN] {variant_name}: {' '.join(cmd[-6:])}")  # abbreviated log
    t0 = time.perf_counter()

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1-hour hard cap
        )
    except subprocess.TimeoutExpired:
        return {
            "block_eff": float("nan"),
            "throughput": float("nan"),
            "ms_per_token": float("nan"),
            "returncode": -1,
            "stderr_tail": "TIMEOUT",
            "wall_sec": time.perf_counter() - t0,
        }

    wall_sec = time.perf_counter() - t0
    combined = proc.stdout + "\n" + proc.stderr

    # Parse tagged output lines produced by GBV/main.py:
    #   Block efficiency (mode=gbv, K=3, T=0.6): 2.345678
    #   Throughput (tokens / second): 12.345678
    #   Walltime (ms / token): 81.000000
    block_eff   = _parse_float(combined, r"Block efficiency \(.*?\):\s*([\d.]+(?:nan)?)")
    throughput  = _parse_float(combined, r"Throughput \(tokens / second\):\s*([\d.]+(?:nan)?)")
    ms_per_token= _parse_float(combined, r"Walltime \(ms / token\):\s*([\d.]+(?:nan)?)")

    stderr_tail = "\n".join(proc.stderr.strip().splitlines()[-5:]) if proc.stderr.strip() else ""

    return {
        "block_eff": block_eff,
        "throughput": throughput,
        "ms_per_token": ms_per_token,
        "returncode": proc.returncode,
        "stderr_tail": stderr_tail,
        "wall_sec": wall_sec,
    }


def _parse_float(text: str, pattern: str) -> float:
    """Extract the first float captured by `pattern` from `text`, or NaN."""
    m = re.search(pattern, text)
    if not m:
        return float("nan")
    try:
        return float(m.group(1))
    except ValueError:
        return float("nan")


def _resolve_data_path(dataset: str) -> str:
    """
    Accept paths relative to OSD/, the parent directory, or absolute.
    Falls back to GBV/data/test.jsonl if the path does not exist.
    """
    p = Path(dataset)
    if p.is_absolute() and p.exists():
        return str(p)
    # Try relative to OSD/
    candidate = _HERE / dataset
    if candidate.exists():
        return str(candidate)
    # Try relative to parent (project root)
    candidate2 = _HERE.parent / dataset
    if candidate2.exists():
        return str(candidate2)
    # Fallback to GBV test set
    fallback = _HERE.parent / "GBV" / "data" / "test.jsonl"
    print(f"  [WARN] Dataset '{dataset}' not found; using fallback {fallback}")
    return str(fallback)


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------

def _print_table(rows: list[dict], baseline_be: float) -> None:
    """Print a human-readable ranked comparison table."""
    header = f"{'Rank':<5} {'Variant':<14} {'Block Eff':>10} {'vs Baseline':>12} {'Throughput':>12} {'ms/tok':>8}  Description"
    sep    = "-" * len(header)
    print()
    print(sep)
    print(header)
    print(sep)
    for i, row in enumerate(rows, 1):
        be   = row["block_eff"]
        diff = (be - baseline_be) / baseline_be * 100 if (baseline_be > 0 and not _is_nan(be)) else float("nan")
        diff_str = f"{diff:+.1f}%" if not _is_nan(diff) else "  n/a"
        be_str   = f"{be:.4f}" if not _is_nan(be) else "  n/a"
        tput_str = f"{row['throughput']:.2f}" if not _is_nan(row['throughput']) else "  n/a"
        ms_str   = f"{row['ms_per_token']:.1f}" if not _is_nan(row['ms_per_token']) else "  n/a"
        print(f"  {i:<4} {row['variant']:<14} {be_str:>10} {diff_str:>12} {tput_str:>12} {ms_str:>8}  {row['description']}")
    print(sep)
    print()


def _is_nan(v) -> bool:
    import math
    try:
        return math.isnan(v)
    except TypeError:
        return True


# ---------------------------------------------------------------------------
# Soft assertions
# ---------------------------------------------------------------------------

def _soft_assert(condition: bool, msg: str) -> None:
    """Print a WARNING if condition is False — does not raise."""
    if not condition:
        print(f"  [WARN] SOFT-ASSERT FAILED: {msg}")
    else:
        print(f"  [OK]   {msg}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Compare all KL distillation variants for speculative decoding acceptance rate."
    )
    ap.add_argument("--n",          type=int,   default=10,
                    help="Number of prompts to evaluate per variant (default: 10). "
                         "Use --n 5 for a quick smoke test.")
    ap.add_argument("--dataset",    type=str,   default="data/gsm8k_10.jsonl",
                    help="Path to JSONL prompt file (relative to OSD/ or absolute).")
    ap.add_argument("--K",          type=int,   default=3,
                    help="Draft tree width (default: 3).")
    ap.add_argument("--T",          type=float, default=0.6,
                    help="Target sampling temperature (default: 0.6).")
    ap.add_argument("--max_tokens", type=int,   default=128,
                    help="Max new tokens per prompt (default: 128).")
    ap.add_argument("--mode",       type=str,   default="gbv",
                    choices=["naive", "nss", "specinfer", "spectr", "bv", "gbv", "traversal"],
                    help="GBV verification algorithm (default: gbv).")
    ap.add_argument("--target",     type=str,   default=_DEFAULT_TARGET,
                    help=f"Target model HF ID or local path (default: {_DEFAULT_TARGET}).")
    ap.add_argument("--output",     type=str,   default="results_comparison.json",
                    help="Path to save JSON results (default: results_comparison.json).")
    ap.add_argument("--skip_missing", action="store_true",
                    help="Skip variants whose checkpoint path does not exist "
                         "(instead of running with a potentially wrong path).")
    ap.add_argument("--variants",   type=str,   default=None,
                    help="Comma-separated subset of variants to run, e.g. 'baseline,forward_kl,ebe'. "
                         "Defaults to all variants in MODEL_CONFIGS.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve dataset path — truncate to --n prompts by writing a temp slice if needed.
    data_path = _resolve_data_path(args.dataset)

    # If --n is smaller than the file's line count, write a truncated temp file.
    data_path = _maybe_truncate_dataset(data_path, args.n)

    # Decide which variants to run.
    if args.variants:
        selected = [v.strip() for v in args.variants.split(",")]
        configs = {k: v for k, v in MODEL_CONFIGS.items() if k in selected}
    else:
        configs = MODEL_CONFIGS

    print(f"\n=== KL Divergence Variant Comparison ===")
    print(f"  Target model : {args.target}")
    print(f"  Dataset      : {data_path}")
    print(f"  n prompts    : {args.n}")
    print(f"  K={args.K}, T={args.T}, max_tokens={args.max_tokens}, mode={args.mode}")
    print(f"  Variants     : {', '.join(configs.keys())}")
    print()

    results: list[dict] = []

    for variant, cfg in configs.items():
        q_model = cfg["q_model"]

        # Skip if checkpoint is missing and --skip_missing is set.
        if args.skip_missing and not _checkpoint_exists(q_model):
            print(f"  [SKIP] {variant}: checkpoint not found at '{q_model}'")
            results.append({
                "variant": variant,
                "description": cfg["description"],
                "q_model": q_model,
                "block_eff": float("nan"),
                "throughput": float("nan"),
                "ms_per_token": float("nan"),
                "returncode": -2,
                "skipped": True,
            })
            continue

        print(f"Running: {variant} ({cfg['description']})")
        r = run_gbv_eval(
            variant_name=variant,
            q_model=q_model,
            p_model=args.target,
            data=data_path,
            K=args.K,
            T=args.T,
            max_tokens=args.max_tokens,
            mode=args.mode,
        )

        if r["returncode"] != 0:
            print(f"  [ERROR] {variant} returned exit code {r['returncode']}")
            if r["stderr_tail"]:
                print(f"  stderr: {r['stderr_tail']}")

        result_row = {
            "variant": variant,
            "description": cfg["description"],
            "q_model": q_model,
            "block_eff": r["block_eff"],
            "throughput": r["throughput"],
            "ms_per_token": r["ms_per_token"],
            "returncode": r["returncode"],
            "wall_sec": r["wall_sec"],
            "skipped": False,
        }
        results.append(result_row)
        print(f"  -> block_eff={r['block_eff']:.4f}, throughput={r['throughput']:.2f} tok/s  "
              f"({r['wall_sec']:.1f}s wall)")

    # Sort by block efficiency descending (NaN sorts to the bottom).
    results_sorted = sorted(
        results,
        key=lambda x: x["block_eff"] if not _is_nan(x["block_eff"]) else -1.0,
        reverse=True,
    )

    # Find baseline block efficiency for relative comparisons.
    baseline_row = next((r for r in results if r["variant"] == "baseline"), None)
    baseline_be  = baseline_row["block_eff"] if baseline_row else float("nan")

    # Print ranked table.
    _print_table(results_sorted, baseline_be)

    # ---------------------------------------------------------------------------
    # Soft assertions — print warnings, do NOT raise / crash.
    # ---------------------------------------------------------------------------
    print("=== Soft Assertions ===")
    _IMPROVEMENT_THRESHOLD = 0.02   # 2% relative gain over baseline

    for target_variant in ("ebe", "forward_kl"):
        row = next((r for r in results if r["variant"] == target_variant), None)
        if row is None or _is_nan(row["block_eff"]) or _is_nan(baseline_be) or baseline_be <= 0:
            print(f"  [SKIP] Cannot check {target_variant} vs baseline — data unavailable")
            continue
        rel_gain = (row["block_eff"] - baseline_be) / baseline_be
        _soft_assert(
            rel_gain >= _IMPROVEMENT_THRESHOLD,
            f"{target_variant} block_eff ({row['block_eff']:.4f}) beats baseline "
            f"({baseline_be:.4f}) by >= 2% (actual: {rel_gain*100:+.2f}%)",
        )

    # Check forward_kl >= reverse_kl (mode-covering should win over mode-seeking).
    fwd_row = next((r for r in results if r["variant"] == "forward_kl"), None)
    rev_row = next((r for r in results if r["variant"] == "reverse_kl"), None)
    if fwd_row and rev_row and not _is_nan(fwd_row["block_eff"]) and not _is_nan(rev_row["block_eff"]):
        _soft_assert(
            fwd_row["block_eff"] >= rev_row["block_eff"],
            f"forward_kl ({fwd_row['block_eff']:.4f}) >= reverse_kl ({rev_row['block_eff']:.4f})",
        )

    # Check online >= baseline (online should never be worse after adaptation).
    online_row = next((r for r in results if r["variant"] == "online"), None)
    if online_row and not _is_nan(online_row["block_eff"]) and not _is_nan(baseline_be):
        _soft_assert(
            online_row["block_eff"] >= baseline_be,
            f"online ({online_row['block_eff']:.4f}) >= baseline ({baseline_be:.4f})",
        )

    print()

    # ---------------------------------------------------------------------------
    # Save results JSON for the viz server to pick up.
    # ---------------------------------------------------------------------------
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = _HERE / output_path

    save_payload = {
        "meta": {
            "target_model": args.target,
            "dataset": data_path,
            "n": args.n,
            "K": args.K,
            "T": args.T,
            "max_tokens": args.max_tokens,
            "mode": args.mode,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "results": results_sorted,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(save_payload, f, indent=2, default=str)

    print(f"Results saved to: {output_path}")
    print("Done.")


# ---------------------------------------------------------------------------
# Dataset truncation helper
# ---------------------------------------------------------------------------

def _maybe_truncate_dataset(data_path: str, n: int) -> str:
    """
    If the JSONL file has more lines than n, write a truncated copy to a temp
    file and return its path.  This avoids patching GBV/main.py.
    """
    p = Path(data_path)
    if not p.exists():
        return data_path

    with open(p, encoding="utf-8") as f:
        lines = [l for l in f if l.strip()]

    if len(lines) <= n:
        return data_path

    # Write truncated version alongside the original.
    tmp_path = p.parent / f"_tmp_trunc_{n}_{p.name}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.writelines(lines[:n])
    return str(tmp_path)


if __name__ == "__main__":
    main()
