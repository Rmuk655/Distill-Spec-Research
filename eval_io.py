"""
eval_io.py — CSV output, prompt-level resume state, and result logging for eval.py.

Public API:
  _state_path(csv_path, mode, K, L, checkpoint, dataset) -> str
  _load_state(path) -> {prompt_idx: run_dict}
  append_csv_row(row, csv_path)
  log_result(stats, args, mode, gpu_monitor, csv_path, specs, cpu_threads_used, phys_gpu_idx)
  _resolve_training_url(checkpoint_path) -> str

CSV_COLUMNS and _FLOAT_FMT define the output schema — edit here to add/remove columns.
"""
from __future__ import annotations

import csv
import json
import os
from datetime import datetime

from config import DEFAULT_DTYPE, TEACHER_MODEL
from telemetry import GpuMonitor


def _state_path(csv_path: str, mode: str, K: int, L: int, checkpoint: str = "",
                dataset: str = "") -> str:
    logs_dir = os.path.join(os.path.dirname(os.path.abspath(csv_path)), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    ds_tag = f".{dataset}" if dataset else ""
    if checkpoint:
        parts = checkpoint.replace("\\", "/").rstrip("/").split("/")
        ckpt_tag = "_".join(parts[-2:]) if len(parts) >= 2 else parts[-1]
        filename = f"{ckpt_tag}.{mode}_K{K}_L{L}{ds_tag}.state.jsonl"
    else:
        filename = f"{mode}_K{K}_L{L}{ds_tag}.state.jsonl"
    return os.path.join(logs_dir, filename)


def _load_state(path: str) -> dict[int, dict]:
    """Return {prompt_idx: run_dict} from an existing state file, or {}."""
    done: dict[int, dict] = {}
    if not os.path.isfile(path):
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                done[r["prompt_idx"]] = r
    return done



CSV_COLUMNS = [
    "timestamp", "checkpoint", "dataset", "mode", "K", "L",
    # Core eval metrics
    "n_prompts", "skipped_prompts", "target_calls",
    "block_eff", "throughput_tok_s",
    "time_per_call_ms", "time_per_token_ms",
    "avg_tree_nodes", "total_gen_tokens", "total_time_s",
    # Time breakdown
    "time_draft_s", "time_target_s", "time_verify_s", "time_cache_s",
    "time_tokenizer_s",
    # GPU telemetry during eval (pynvml background thread, 1 Hz — ~0.001% overhead)
    "gpu_sm_util_avg_pct",    # SM utilization % (avg) — time SMs were active
    "gpu_mem_util_avg_pct",   # Memory-bus utilization % — closest proxy for HBM bandwidth
    "gpu_vram_peak_mb",       # Peak VRAM in use during eval (MB)
    "gpu_pcie_tx_avg_kbs",    # PCIe host→device throughput avg (KB/s)
    "gpu_pcie_rx_avg_kbs",    # PCIe device→host throughput avg (KB/s)
    "gpu_nvlink_tx_avg_kbs",  # NVLink TX avg KB/s (blank if GPU has no NVLink)
    "gpu_nvlink_rx_avg_kbs",  # NVLink RX avg KB/s (blank if GPU has no NVLink)
    "cpu_util_avg_pct",       # CPU utilization % averaged over eval (all cores)
    "torch_vram_alloc_mb",    # torch.cuda.memory_allocated at eval-end snapshot (MB)
    "torch_vram_reserv_mb",   # torch.cuda.memory_reserved at eval-end snapshot (MB)
    # Reproducibility fingerprint
    "device", "seed", "dtype", "cpu_threads_used",
    # Machine spec (one-time snapshot at startup — changes if server changes)
    "machine_gpu",            # GPU model name, e.g. "NVIDIA A100-SXM4-40GB"
    "machine_gpu_count",      # Total GPUs on the node
    "machine_gpu_vram_gb",    # VRAM per GPU (GB)
    "machine_driver",         # NVIDIA driver version
    "machine_cpu_logical_cores",
    "machine_cpu_physical_cores",
    "machine_ram_gb",
    "machine_os",
    # Attention backend (flash_attn | sdpa | eager) — throughput is order-of-magnitude
    # lower without flash_attn, so this must be logged for valid comparisons.
    "attn_backend",
    # Link back to the training run that produced this checkpoint
    "training_wandb_url",
    # Delayed-expansion config — 0 for all regular runs (backward compatible, last column)
    "L1",
    # Teacher/target model id used for this eval — needed once multiple draft-teacher
    # pairs are in play (e.g. Qwen3-1.7B/Qwen3-32B) so CSV rows are self-describing.
    "teacher_model",
]


def append_csv_row(row: dict, csv_path: str):
    """Append one row to csv_path, creating the file (with header) if needed."""
    new_file = not os.path.isfile(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        if new_file:
            w.writeheader()
        w.writerow(row)


# Format specs for float stats columns; integer columns pass through as-is.
_FLOAT_FMT: dict[str, str] = {
    "block_eff": ".6f", "throughput_tok_s": ".4f",
    "time_per_call_ms": ".3f", "time_per_token_ms": ".3f",
    "avg_tree_nodes": ".4f", "total_time_s": ".2f",
    "time_draft_s": ".3f", "time_target_s": ".3f",
    "time_verify_s": ".3f", "time_cache_s": ".3f",
    "time_tokenizer_s": ".3f",
    "gpu_sm_util_avg_pct": ".1f", "gpu_mem_util_avg_pct": ".1f",
    "gpu_vram_peak_mb": ".0f",
    "gpu_pcie_tx_avg_kbs": ".0f", "gpu_pcie_rx_avg_kbs": ".0f",
    "gpu_nvlink_tx_avg_kbs": ".0f", "gpu_nvlink_rx_avg_kbs": ".0f",
    "cpu_util_avg_pct": ".1f",
    "torch_vram_alloc_mb": ".1f", "torch_vram_reserv_mb": ".1f",
}


def _resolve_training_url(checkpoint_path: str) -> str:
    """Read the W&B training URL from wandb_run.json in the checkpoint's parent directory."""
    parent = os.path.dirname(os.path.abspath(checkpoint_path))
    meta = os.path.join(parent, "wandb_run.json")
    if os.path.isfile(meta):
        try:
            return json.load(open(meta, encoding="utf-8")).get("url", "")
        except Exception:
            pass
    return ""


def log_result(stats: dict, args, mode: str, gpu_monitor: GpuMonitor | None,
               csv_path: str, specs: dict | None = None,
               cpu_threads_used: int | None = None, phys_gpu_idx: int = 0):
    """Print a one-line summary, GPU telemetry, and append a CSV row."""
    skip_note = f"  ({stats['skipped_prompts']} skipped)" if stats.get("skipped_prompts") else ""
    print(f"\n  mode={mode:11s}  BE={stats['block_eff']:.4f}  "
          f"throughput={stats['throughput_tok_s']:.1f} tok/s  "
          f"avg_tree_nodes={stats['avg_tree_nodes']:.1f}{skip_note}")
    print(f"    time → draft={stats['time_draft_s']:.2f}s  "
          f"target={stats['time_target_s']:.2f}s  "
          f"verify={stats['time_verify_s']:.2f}s  "
          f"cache={stats['time_cache_s']:.2f}s  "
          f"tokenizer={stats['time_tokenizer_s']:.2f}s")
    if gpu_monitor is not None:
        gpu_monitor.print_summary()

    row: dict = {
        "timestamp":       datetime.utcnow().isoformat(timespec="seconds"),
        "checkpoint":      args.checkpoint, "dataset": args.dataset,
        "mode":            mode, "K": args.K, "L": args.L,
        "L1":              getattr(args, "L1", 0),
        "device":          f"cuda:{phys_gpu_idx}", "seed": args.seed, "dtype": DEFAULT_DTYPE,
        "cpu_threads_used": cpu_threads_used,
        "training_wandb_url": _resolve_training_url(args.checkpoint),
        "teacher_model":   getattr(args, "teacher", TEACHER_MODEL),
    }
    for k, v in stats.items():
        if isinstance(v, float) and k in _FLOAT_FMT:
            row[k] = format(v, _FLOAT_FMT[k])
        elif v is not None:
            row[k] = v
    if specs:
        row.update(specs)
    append_csv_row(row, csv_path)