#!/usr/bin/env python3
"""
Consolidate all eval results into one deduplicated CSV.
Sources (priority low→high):
  1. logs/*.state.jsonl  — BE computed from gen_tokens/target_calls
  2. res_*.csv / diag_*.csv — partial evals
  3. results.csv — full eval rows (wins on dedup)
Output: all_results.csv (same schema as results.csv)
"""

import csv
import glob
import json
import os
import re
from collections import defaultdict

LOG_DIR   = "logs"
CKPT_BASE = "/sensei-fs-3/users/rkrishna/checkpoints"

RESULTS_COLS = [
    "timestamp", "checkpoint", "dataset", "mode", "K", "L",
    "n_prompts", "skipped_prompts", "target_calls", "block_eff",
    "throughput_tok_s", "time_per_call_ms", "time_per_token_ms",
    "avg_tree_nodes", "total_gen_tokens", "total_time_s",
    "time_draft_s", "time_target_s", "time_verify_s", "time_cache_s",
    "time_tokenizer_s", "gpu_sm_util_avg_pct", "gpu_mem_util_avg_pct",
    "gpu_vram_peak_mb", "gpu_pcie_tx_avg_kbs", "gpu_pcie_rx_avg_kbs",
    "gpu_nvlink_tx_avg_kbs", "gpu_nvlink_rx_avg_kbs", "cpu_util_avg_pct",
    "torch_vram_alloc_mb", "torch_vram_reserv_mb", "device", "seed",
    "dtype", "cpu_threads_used", "machine_gpu", "machine_gpu_count",
    "machine_gpu_vram_gb", "machine_driver", "machine_cpu_logical_cores",
    "machine_cpu_physical_cores", "machine_ram_gb", "machine_os",
    "attn_backend", "training_wandb_url", "L1",
]


def stem_to_ckpt_path(stem):
    for suffix in ["_ckpt_best", "_ckpt_last"]:
        if stem.endswith(suffix):
            name = stem[: -len(suffix)]
            return f"{CKPT_BASE}/{name}/{suffix[1:]}"
    return f"{CKPT_BASE}/{stem}"


def parse_state_file(path):
    fname = os.path.basename(path)
    m = re.match(r"^(.+)\.([\w]+)_K(\d+)_L(\d+)\.(\w+)\.state\.jsonl$", fname)
    if not m:
        return None
    stem, mode, K, L, dataset = m.groups()
    K, L = int(K), int(L)

    try:
        rows = [json.loads(line) for line in open(path) if line.strip()]
    except Exception:
        return None
    if not rows:
        return None

    total_gen    = sum(r.get("gen_tokens", 0)     for r in rows)
    total_calls  = sum(r.get("target_calls", 0)   for r in rows)
    total_time   = sum(r.get("total_time", 0)      for r in rows)
    total_draft  = sum(r.get("time_draft", 0)      for r in rows)
    total_target = sum(r.get("time_target", 0)     for r in rows)
    total_verify = sum(r.get("time_verify", 0)     for r in rows)
    total_cache  = sum(r.get("time_cache", 0)      for r in rows)
    total_nodes  = sum(r.get("total_tree_nodes", 0) for r in rows)

    if total_calls == 0:
        return None

    be               = total_gen / total_calls
    throughput       = total_gen / total_time   if total_time  else 0
    time_per_call_ms = total_time / total_calls * 1000 if total_calls else 0
    time_per_tok_ms  = total_time / total_gen   * 1000 if total_gen   else 0
    avg_nodes        = total_nodes / total_calls if total_calls else 0

    row = {c: "" for c in RESULTS_COLS}
    row.update({
        "checkpoint":       stem_to_ckpt_path(stem),
        "dataset":          dataset,
        "mode":             mode,
        "K":                K,
        "L":                L,
        "n_prompts":        len(rows),
        "skipped_prompts":  0,
        "target_calls":     total_calls,
        "block_eff":        f"{be:.6f}",
        "throughput_tok_s": f"{throughput:.4f}",
        "time_per_call_ms": f"{time_per_call_ms:.3f}",
        "time_per_token_ms":f"{time_per_tok_ms:.3f}",
        "avg_tree_nodes":   f"{avg_nodes:.4f}",
        "total_gen_tokens": total_gen,
        "total_time_s":     f"{total_time:.2f}",
        "time_draft_s":     f"{total_draft:.3f}",
        "time_target_s":    f"{total_target:.3f}",
        "time_verify_s":    f"{total_verify:.3f}",
        "time_cache_s":     f"{total_cache:.3f}",
        "L1":               0,
    })
    return row


def dedup_key(row):
    return (
        str(row.get("checkpoint", "")),
        str(row.get("dataset", "")),
        str(row.get("mode", "")),
        str(row.get("K", "")),
        str(row.get("L", "")),
    )


def load_csv(path):
    try:
        return list(csv.DictReader(open(path)))
    except Exception:
        return []


def main():
    seen = {}  # dedup_key -> row; higher-priority source overwrites

    # Priority 1: state.jsonl (lowest — only BE, no GPU stats)
    for path in sorted(glob.glob(os.path.join(LOG_DIR, "*.state.jsonl"))):
        row = parse_state_file(path)
        if row is None:
            continue
        key = dedup_key(row)
        if key not in seen:
            seen[key] = row

    # Priority 2: res_*.csv and diag_*.csv
    for path in sorted(glob.glob("*.csv")):
        if path in ("all_results.csv", "results.csv"):
            continue
        for row in load_csv(path):
            if not row.get("block_eff"):
                continue
            key = dedup_key(row)
            seen[key] = row  # overwrites state.jsonl

    # Priority 3: results.csv (highest — full metadata)
    for row in load_csv("results.csv"):
        if not row.get("block_eff"):
            continue
        key = dedup_key(row)
        seen[key] = row

    # Collect any extra columns that appeared in CSVs
    all_cols = list(RESULTS_COLS)
    extra = []
    for row in seen.values():
        for k in row:
            if k not in all_cols and k not in extra:
                extra.append(k)
    all_cols += extra

    out_path = "all_results.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_cols, extrasaction="ignore")
        writer.writeheader()
        for row in sorted(
            seen.values(),
            key=lambda r: (
                r.get("checkpoint", ""),
                r.get("dataset", ""),
                r.get("mode", ""),
                str(r.get("K", "")),
                str(r.get("L", "")),
            ),
        ):
            writer.writerow(row)

    print(f"Written {len(seen)} rows → {out_path}")

    # ── Summary 1: best checkpoint per (mode, K, dataset) ──────────────────
    # Index: (mode, K, dataset) -> list of (be, ckpt_short)
    REPORT_MODES    = ["traversal", "nss", "bv", "gbv", "naive", "specinfer", "spectr", "khisti"]
    REPORT_KS       = ["1", "2", "3", "4"]
    REPORT_DATASETS = ["math_eval", "math500", "math_val"]

    # build lookup: (ckpt_short, mode, K, dataset) -> be
    cell = {}
    for row in seen.values():
        ckpt  = row.get("checkpoint", "")
        short = ckpt.split("/")[-2] if "/" in ckpt else ckpt
        mode  = row.get("mode", "")
        K     = str(row.get("K", ""))
        ds    = row.get("dataset", "")
        try:
            be = float(row.get("block_eff", ""))
        except (ValueError, TypeError):
            continue
        cell[(short, mode, K, ds)] = be

    print("\n\n── Best checkpoint per verifier / K (math_eval or math500) ──")
    for ds in REPORT_DATASETS:
        ds_rows = {k: v for k, v in cell.items() if k[3] == ds}
        if not ds_rows:
            continue
        print(f"\n  dataset={ds}")
        print(f"  {'mode':<12} {'K':>2}   {'BE':>6}   checkpoint")
        print(f"  {'-'*12} {'-'*2}   {'-'*6}   {'-'*50}")
        for mode in REPORT_MODES:
            for K in REPORT_KS:
                candidates = {ckpt: be for (ckpt, m, k, d), be in cell.items()
                              if m == mode and k == K and d == ds}
                if not candidates:
                    continue
                best_ckpt = max(candidates, key=candidates.__getitem__)
                best_be   = candidates[best_ckpt]
                print(f"  {mode:<12} {K:>2}   {best_be:>6.4f}   {best_ckpt}")

    # ── Summary 2: per-checkpoint table across modes at K=3 ────────────────
    by_ckpt = defaultdict(list)
    for row in seen.values():
        ckpt  = row.get("checkpoint", "")
        short = ckpt.split("/")[-2] if "/" in ckpt else ckpt
        by_ckpt[short].append(row)

    for ds in REPORT_DATASETS:
        has_any = any(
            r.get("dataset") == ds
            for rows in by_ckpt.values()
            for r in rows
        )
        if not has_any:
            continue
        modes_hdr = ["traversal", "nss", "bv", "naive"]
        col_w = 10
        hdr = f"{'checkpoint':<50}" + "".join(f"{m:>{col_w}}" for m in modes_hdr)
        print(f"\n\n── K=3, dataset={ds} ──")
        print(f"  {hdr}")
        print(f"  {'-'*50}" + "-" * (col_w * len(modes_hdr)))
        for ckpt, rows in sorted(by_ckpt.items()):
            vals = []
            for mode in modes_hdr:
                be = cell.get((ckpt, mode, "3", ds))
                vals.append(f"{be:.4f}" if be is not None else "—")
            print(f"  {ckpt:<50}" + "".join(f"{v:>{col_w}}" for v in vals))


if __name__ == "__main__":
    main()
