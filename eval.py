"""
eval.py — block-efficiency + alpha evaluation for a trained draft model.

Thin wrapper around speculative_decoding_loop() from main.py (GBV source of truth).
This file owns: dataset selection, multi-mode sweep, per-mode seed reset, CSV logging,
GPU/CPU telemetry, and verifier-exception handling.
All speculative decoding logic lives in main.py — do not duplicate it here.

    block_efficiency  = total_generated_tokens / total_target_calls
    throughput        = total_generated_tokens / wall_time   (tok/s)
    avg_tree_nodes    = average tree size per target call

Dataset setup (run once before eval; default gsm8k_eval has 100 prompts):
    python -m data_io.download --datasets gsm8k --n 1000 --force

Usage:
    python eval.py --checkpoint checkpoints/kl_tree/ckpt_best  --mode gbv
    python eval.py --checkpoint checkpoints/gbv_tree/ckpt_best --mode gbv --K 3 --L 8
    python eval.py --checkpoint Qwen/Qwen3-0.6B --mode naive --dataset alpaca   # baseline
    python eval.py --checkpoint ckpts/best --mode gbv --modes naive,gbv,traversal  # sweep

    # one eval per GPU — pin to specific GPU so parallel runs don't fight:
    CUDA_VISIBLE_DEVICES=0 python eval.py --checkpoint ckpt_a --modes naive --device cuda:0 &
    CUDA_VISIBLE_DEVICES=1 python eval.py --checkpoint ckpt_b --modes gbv   --device cuda:1 &

    # parallel eval across modes (parallel-safe --output flag):
    python eval.py --checkpoint Qwen/Qwen3-0.6B --K 1 --n 1000 --modes naive     --output out_naive.csv &
    python eval.py --checkpoint Qwen/Qwen3-0.6B --K 1 --n 1000 --modes specinfer --output out_specinfer.csv &

Results print to stdout AND append one row to results.csv for later analysis.

GPU reproducibility protocol (1 eval per GPU):
    • Fix the GPU via --device cuda:N (or CUDA_VISIBLE_DEVICES=N --device cuda:0)
    • Fix seed via --seed (default 123) — reset before EVERY mode sweep
    • Prompts are always read in file order and sliced [:n]; do not shuffle
    • dtype is fixed to DEFAULT_DTYPE (bf16); do not mix precision across runs
    • CPU threads are capped via --cpu_threads to prevent inter-run variation
    • Both models load to the SAME device in a fixed order (teacher first, then draft)

GPU telemetry notes:
    • SM utilization and memory-bus utilization are polled via pynvml (1 s interval)
      in a background thread during each mode sweep.
    • HBM bandwidth proxy = memory-bus utilization % reported by the GPU driver;
      exact GB/s requires DCGM (install: sudo apt install datacenter-gpu-manager).
    • L2 cache hit rate requires DCGM; not available via standard NVML.
    • PCIe TX/RX throughput (KB/s) is read from NVML.
    • NVLink counters are read if NVLink is present; skipped otherwise.
    • CPU utilization is polled via psutil in the same background thread.
    Install optional deps once: pip install pynvml psutil
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import threading
import time
from datetime import datetime

import torch
from tqdm import tqdm

import verifiers  # noqa: F401 — sys.path injection
from util            import load_prompts_jsonl, set_seed, load_models
from main            import speculative_decoding_loop
from verifier_safe   import VerifierError

from data_io import get_path as dataset_path
from config  import (TEACHER_MODEL, DEFAULT_K, DEFAULT_L, DEFAULT_MAX_NEW_TOKENS,
                     DEFAULT_TEMP, DEFAULT_DTYPE, DEFAULT_SEED, VERIFIER_MODES, block_eff)

RESULTS_CSV = os.path.join(os.path.dirname(__file__), "results.csv")


# ═══════════════════════════════════════════════════════════════════════════
#  GPU / CPU telemetry (pynvml + psutil — gracefully degrades if missing)
# ═══════════════════════════════════════════════════════════════════════════

def _gpu_index_from_device(device_str: str) -> int:
    """Parse 'cuda:2' → 2.  Falls back to 0 for plain 'cuda' or 'cpu'."""
    if ":" in device_str:
        try:
            return int(device_str.split(":")[-1])
        except ValueError:
            pass
    return 0


class GpuMonitor:
    """
    Background daemon thread that polls GPU and CPU metrics at 1-second intervals.

    Overhead: each NVML call takes ~10 μs; at 1 Hz that is 0.001 % of wall time.
    The GIL is released during the C-extension NVML calls, so the main eval thread
    is NEVER blocked.  This does NOT measurably affect throughput measurements.

    Metrics collected (when pynvml / psutil are available):
        sm_util        : SM utilization % (time GPU SMs were active — compute proxy)
        mem_util       : memory-bus utilization % (fraction of time HBM was busy)
        vram_used_mb   : VRAM currently in use (MB)
        pcie_tx_kbs    : PCIe TX throughput (KB/s, host→device)
        pcie_rx_kbs    : PCIe RX throughput (KB/s, device→host)
        nvlink_tx_kbs  : NVLink TX counter, summed over all links (KB/s, if present)
        nvlink_rx_kbs  : NVLink RX counter, summed over all links (KB/s, if present)
        cpu_util       : CPU utilization % across all cores (via psutil)

    Metrics that require DCGM (not standard NVML):
        L2 cache hit rate  — sudo apt install datacenter-gpu-manager
        HBM bandwidth GB/s — mem_util % is the driver's closest proxy without DCGM
        SM occupancy       — requires DCGM or Nsight

    When running 1 eval per GPU the SM/HBM numbers still tell you whether:
        - SM utilization stays near 100% → compute-bound (good for A100)
        - mem_util near 100% but SM low → memory-bandwidth bound
        - Both low → CPU/Python overhead is the bottleneck

    If throughput changes while VRAM usage stays similar across checkpoints, the
    culprit is HBM bandwidth or cache contention, not capacity pressure.

    Usage:
        mon = GpuMonitor(device_idx=0)
        mon.start()
        ... eval loop ...
        mon.stop()
        summary = mon.summary()   # dict of avg/peak stats
    """

    _POLL_INTERVAL = 1.0  # seconds

    def __init__(self, device_idx: int = 0):
        self._dev = device_idx
        self._samples: list[dict] = []
        self._running = False
        self._thread: threading.Thread | None = None

        self._nvml = None
        self._handle = None
        self._nvlink_links: list[int] = []

        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
            # Probe NVLink links (A100 NVLink has up to 12 links).
            for link in range(12):
                try:
                    pynvml.nvmlDeviceGetNvLinkState(self._handle, link)
                    self._nvlink_links.append(link)
                except pynvml.NVMLError:
                    break
        except Exception as e:
            print(f"[GpuMonitor] pynvml unavailable — GPU telemetry disabled ({e})")

        self._psutil = None
        try:
            import psutil
            self._psutil = psutil
        except Exception:
            pass

    # ------------------------------------------------------------------
    def start(self):
        self._samples = []
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)

    def _poll_loop(self):
        while self._running:
            s = {}
            if self._nvml and self._handle:
                try:
                    util = self._nvml.nvmlDeviceGetUtilizationRates(self._handle)
                    s["sm_util"]    = util.gpu     # SM utilization %
                    s["mem_util"]   = util.memory  # memory-bus utilization %
                except Exception:
                    pass
                try:
                    mem = self._nvml.nvmlDeviceGetMemoryInfo(self._handle)
                    s["vram_used_mb"] = mem.used >> 20
                except Exception:
                    pass
                try:
                    tx = self._nvml.nvmlDeviceGetPcieThroughput(
                        self._handle, self._nvml.NVML_PCIE_UTIL_TX_BYTES)
                    rx = self._nvml.nvmlDeviceGetPcieThroughput(
                        self._handle, self._nvml.NVML_PCIE_UTIL_RX_BYTES)
                    s["pcie_tx_kbs"] = tx
                    s["pcie_rx_kbs"] = rx
                except Exception:
                    pass
                # NVLink — sum counters across all active links.
                if self._nvlink_links:
                    try:
                        tx_total, rx_total = 0, 0
                        for link in self._nvlink_links:
                            # Counter 0 = TX, 1 = RX (bytes, resets to 0 on overflow)
                            tx_total += self._nvml.nvmlDeviceGetNvLinkUtilizationCounter(
                                self._handle, link, 0)
                            rx_total += self._nvml.nvmlDeviceGetNvLinkUtilizationCounter(
                                self._handle, link, 1)
                        s["nvlink_tx_kbs"] = tx_total
                        s["nvlink_rx_kbs"] = rx_total
                    except Exception:
                        pass

            if self._psutil:
                try:
                    s["cpu_util"] = self._psutil.cpu_percent(interval=None)
                except Exception:
                    pass

            if s:
                self._samples.append(s)
            time.sleep(self._POLL_INTERVAL)

    # ------------------------------------------------------------------
    def summary(self) -> dict:
        """Aggregate stats over all collected samples."""
        if not self._samples:
            return {}

        def _avg(key):
            vals = [s[key] for s in self._samples if key in s]
            return round(sum(vals) / len(vals), 2) if vals else None

        def _peak(key):
            vals = [s[key] for s in self._samples if key in s]
            return max(vals) if vals else None

        out: dict = {}
        for k, fn in [
            ("gpu_sm_util_avg_pct",    lambda: _avg("sm_util")),
            ("gpu_mem_util_avg_pct",   lambda: _avg("mem_util")),   # HBM bus %
            ("gpu_vram_peak_mb",       lambda: _peak("vram_used_mb")),
            ("gpu_pcie_tx_avg_kbs",    lambda: _avg("pcie_tx_kbs")),
            ("gpu_pcie_rx_avg_kbs",    lambda: _avg("pcie_rx_kbs")),
            ("gpu_nvlink_tx_avg_kbs",  lambda: _avg("nvlink_tx_kbs")),
            ("gpu_nvlink_rx_avg_kbs",  lambda: _avg("nvlink_rx_kbs")),
            ("cpu_util_avg_pct",       lambda: _avg("cpu_util")),
        ]:
            v = fn()
            if v is not None:
                out[k] = v

        # Also include torch.cuda memory snapshot (always available).
        try:
            out["torch_vram_alloc_mb"]  = round(torch.cuda.memory_allocated(self._dev) / 1024**2, 1)
            out["torch_vram_reserv_mb"] = round(torch.cuda.memory_reserved(self._dev)  / 1024**2, 1)
        except Exception:
            pass

        return out

    def print_summary(self, prefix: str = ""):
        s = self.summary()
        if not s:
            print(f"  {prefix}[gpu] no telemetry collected (install pynvml, psutil)")
            return
        parts = []
        if "gpu_sm_util_avg_pct" in s:
            parts.append(f"SM={s['gpu_sm_util_avg_pct']:.0f}%")
        if "gpu_mem_util_avg_pct" in s:
            parts.append(f"HBM-bus={s['gpu_mem_util_avg_pct']:.0f}%")
        if "gpu_vram_peak_mb" in s:
            parts.append(f"VRAM-peak={s['gpu_vram_peak_mb']}MB")
        if "gpu_pcie_tx_avg_kbs" in s:
            parts.append(f"PCIe-TX={s['gpu_pcie_tx_avg_kbs']/1024:.1f}MB/s")
        if "gpu_pcie_rx_avg_kbs" in s:
            parts.append(f"PCIe-RX={s['gpu_pcie_rx_avg_kbs']/1024:.1f}MB/s")
        if "gpu_nvlink_tx_avg_kbs" in s:
            parts.append(f"NVLink-TX={s['gpu_nvlink_tx_avg_kbs']/1024:.1f}MB/s")
        if "cpu_util_avg_pct" in s:
            parts.append(f"CPU={s['cpu_util_avg_pct']:.0f}%")
        if "torch_vram_alloc_mb" in s:
            parts.append(f"torch-alloc={s['torch_vram_alloc_mb']}MB")
        print(f"  {prefix}[gpu] " + "  ".join(parts))


# ═══════════════════════════════════════════════════════════════════════════
#  Machine-spec snapshot (recorded once at startup, stored in every CSV row)
# ═══════════════════════════════════════════════════════════════════════════

def _machine_specs(gpu_idx: int = 0) -> dict:
    """
    Collect a one-time snapshot of the machine's hardware for reproducibility.

    Recorded in every CSV row so two runs can be compared even if the server
    changes.  All fields degrade gracefully when the relevant library is absent.
    """
    specs: dict = {}

    # GPU identity
    if torch.cuda.is_available():
        try:
            specs["machine_gpu"]       = torch.cuda.get_device_name(gpu_idx)
            specs["machine_gpu_count"] = torch.cuda.device_count()
            total_vram = torch.cuda.get_device_properties(gpu_idx).total_memory
            specs["machine_gpu_vram_gb"] = round(total_vram / 1024**3, 1)
        except Exception:
            pass
    else:
        specs["machine_gpu"] = "cpu"

    # NVIDIA driver version
    try:
        import pynvml
        pynvml.nvmlInit()
        specs["machine_driver"] = pynvml.nvmlSystemGetDriverVersion()
    except Exception:
        pass

    # CPU / RAM
    specs["machine_cpu_logical_cores"] = os.cpu_count() or 0
    try:
        import psutil
        specs["machine_cpu_physical_cores"] = psutil.cpu_count(logical=False) or 0
        specs["machine_ram_gb"] = round(psutil.virtual_memory().total / 1024**3, 1)
    except Exception:
        pass

    specs["machine_os"] = platform.platform(terse=True)

    return specs


def _auto_cpu_threads() -> int:
    """
    Compute a sensible default for torch CPU threads when running 1 eval per GPU.

    With N_gpus eval processes sharing C physical CPU cores, each process should
    use at most C // N_gpus threads so they don't thrash each other's CPU caches.
    For this workload (GPU-bound eval), the practical impact is small but it
    prevents the 4 × 96 = 384-thread scenario on a 4-GPU / 96-core server.

    Example (our server): 96 cores, 4 GPUs → 24 threads per eval process.
    """
    n_gpus = max(1, torch.cuda.device_count() if torch.cuda.is_available() else 1)
    try:
        import psutil
        n_cores = psutil.cpu_count(logical=False) or os.cpu_count() or 1
    except Exception:
        n_cores = os.cpu_count() or 1
    return max(1, n_cores // n_gpus)


# ═══════════════════════════════════════════════════════════════════════════
#  Resume state helpers — one JSONL per (mode, K, L), one line per prompt
# ═══════════════════════════════════════════════════════════════════════════

def _state_path(csv_path: str, mode: str, K: int, L: int, checkpoint: str = "") -> str:
    logs_dir = os.path.join(os.path.dirname(os.path.abspath(csv_path)), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    if checkpoint:
        parts = checkpoint.replace("\\", "/").rstrip("/").split("/")
        ckpt_tag = "_".join(parts[-2:]) if len(parts) >= 2 else parts[-1]
        filename = f"{ckpt_tag}.{mode}_K{K}_L{L}.state.jsonl"
    else:
        filename = f"{mode}_K{K}_L{L}.state.jsonl"
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


# ═══════════════════════════════════════════════════════════════════════════
#  Per-mode aggregate — calls speculative_decoding_loop (source of truth)
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_one_mode(p_model, q_model, tok, prompts, mode, K, L,
                      max_new_tokens, temp, state_path: str | None = None,
                      gpu_monitor: GpuMonitor | None = None):
    """Run the prompt set under one verifier mode and aggregate stats.

    Verifier exceptions (VerifierError from verifier_safe.py) are caught
    per-prompt: the failing prompt is skipped, its index is logged to
    verifier_errors.log, and the eval continues.  skipped_prompts count
    appears in the returned stats dict.

    Resumes from a prior partial run if state_path exists — already-completed
    prompts are skipped and their stats are loaded from disk so block_eff is
    computed correctly over the full n_prompts denominator.
    """
    done = _load_state(state_path) if state_path else {}
    if done:
        print(f"  [resume] {len(done)}/{len(prompts)} prompts already done "
              f"— skipping, loading from {state_path}")

    state_f = open(state_path, "a", encoding="utf-8") if state_path else None
    all_runs: list[dict] = []
    skipped = 0

    if gpu_monitor is not None:
        gpu_monitor.start()

    t_tokenizer_total = 0.0

    try:
        for i, prompt in enumerate(tqdm(prompts, desc=f"mode={mode} K={K} L={L}", ncols=80)):
            if i in done:
                all_runs.append(done[i])
                continue

            # Time the tokenizer separately so we can attribute its cost.
            _t0_tok = time.perf_counter()
            _ = tok.encode(prompt)  # warm the tokenizer; actual encode is inside the loop
            t_tokenizer_total += time.perf_counter() - _t0_tok

            p_model._spec_profile  = {"runs": []}
            p_model._spec_prompt_idx = i  # propagated to verifier_safe via _spec_debug_ctx

            try:
                speculative_decoding_loop(
                    p_model=p_model, q_model=q_model, tok=tok,
                    prompt=prompt, verification_algo=mode,
                    max_new_tokens=max_new_tokens, K=K, L=L,
                    p_temp=temp, q_temp=temp,
                )
            except VerifierError as ve:
                skipped += 1
                print(f"\n  [skip] prompt {i}: verifier raised → {ve}")
                print(f"         full context written to verifier_errors.log")
                continue

            run = p_model._spec_profile["runs"][0]
            run["prompt_idx"] = i
            all_runs.append(run)

            if state_f:
                state_f.write(json.dumps(run) + "\n")
                state_f.flush()
    finally:
        if gpu_monitor is not None:
            gpu_monitor.stop()
        if state_f:
            state_f.close()

    total_calls  = sum(r["target_calls"]     for r in all_runs)
    total_gen    = sum(r["gen_tokens"]       for r in all_runs)
    total_time   = sum(r["total_time"]       for r in all_runs)
    total_nodes  = sum(r["total_tree_nodes"] for r in all_runs)
    time_draft   = sum(r.get("time_draft",   0.0) for r in all_runs)
    time_target  = sum(r.get("time_target",  0.0) for r in all_runs)
    time_verify  = sum(r.get("time_verify",  0.0) for r in all_runs)
    time_cache   = sum(r.get("time_cache",   0.0) for r in all_runs)

    stats = {
        "n_prompts":         len(all_runs),
        "skipped_prompts":   skipped,
        "target_calls":      total_calls,
        "block_eff":         block_eff(total_gen, total_calls),
        "throughput_tok_s":  total_gen / total_time    if total_time  > 0 else float("nan"),
        "time_per_call_ms":  1000.0 * total_time / total_calls  if total_calls > 0 else float("nan"),
        "time_per_token_ms": 1000.0 * total_time / total_gen    if total_gen   > 0 else float("nan"),
        "avg_tree_nodes":    total_nodes / total_calls if total_calls > 0 else float("nan"),
        "total_gen_tokens":  total_gen,
        "total_time_s":      total_time,
        "time_draft_s":      time_draft,
        "time_target_s":     time_target,
        "time_verify_s":     time_verify,
        "time_cache_s":      time_cache,
        "time_tokenizer_s":  t_tokenizer_total,
    }

    # Merge GPU telemetry into stats dict.
    if gpu_monitor is not None:
        stats.update(gpu_monitor.summary())

    return stats


# ═══════════════════════════════════════════════════════════════════════════
#  CSV append (one row per eval cell — easy to grep / pandas later)
# ═══════════════════════════════════════════════════════════════════════════

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


def log_result(stats: dict, args, mode: str, gpu_monitor: GpuMonitor | None,
               csv_path: str, specs: dict | None = None,
               cpu_threads_used: int | None = None):
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
        "device":          args.device, "seed": args.seed, "dtype": DEFAULT_DTYPE,
        "cpu_threads_used": cpu_threads_used,
    }
    for k, v in stats.items():
        if isinstance(v, float) and k in _FLOAT_FMT:
            row[k] = format(v, _FLOAT_FMT[k])
        elif v is not None:
            row[k] = v
    if specs:
        row.update(specs)
    append_csv_row(row, csv_path)


# ═══════════════════════════════════════════════════════════════════════════
#  Argparse + main
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    ap = argparse.ArgumentParser(description="Block-efficiency eval (A100, Qwen3).")
    ap.add_argument("--checkpoint", required=True,
                    help="Path to draft checkpoint dir (or HF model id for baseline).")
    ap.add_argument("--mode",      default="gbv",   choices=VERIFIER_MODES,
                    help="Verifier mode (default gbv).  Override via --modes for a sweep.")
    ap.add_argument("--modes",     default=None,
                    help="Comma-separated verifier modes for a sweep, e.g. 'naive,gbv,traversal'.")
    ap.add_argument("--dataset",   default="gsm8k_eval",
                    help="Dataset name (gsm8k_eval, alpaca, math500, humaneval, mtbench).")
    ap.add_argument("--K",         type=int, default=DEFAULT_K)
    ap.add_argument("--L",         type=int, default=DEFAULT_L)
    ap.add_argument("--n",         type=int, default=100,
                    help="How many prompts from the dataset to evaluate.")
    ap.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    ap.add_argument("--temp",      type=float, default=DEFAULT_TEMP)
    ap.add_argument("--device",    default="cuda",
                    help="CUDA device to use, e.g. cuda:0 or cuda:1.  "
                         "For 1-eval-per-GPU runs use CUDA_VISIBLE_DEVICES=N --device cuda:0.  "
                         "Default 'cuda' auto-selects the GPU with most free memory.")
    ap.add_argument("--seed",      type=int, default=DEFAULT_SEED,
                    help="RNG seed (default 123).  Reset before EVERY mode to keep "
                         "prompt order and sampling identical across runs.")
    ap.add_argument("--cpu_threads", type=int, default=None,
                    help="Override the auto-computed PyTorch CPU thread count.  "
                         "Auto default: physical_cores // num_gpus (e.g. 96 cores / 4 GPUs = 24).  "
                         "When 4 evals run simultaneously (1 per GPU), each needs its own CPU "
                         "budget so the 4 processes don't compete for the same 96 cores.  "
                         "Within a single eval, prompts are always sequential; these threads "
                         "are for PyTorch intra-op parallelism (BLAS etc.), not prompt batching.")
    ap.add_argument("--output",    default=None,
                    help="Override output CSV path (default: results.csv next to eval.py). "
                         "Set to a unique path when running multiple parallel processes.")
    ap.add_argument("--no_gpu_monitor", action="store_true",
                    help="Disable the background GPU/CPU telemetry thread.  "
                         "The thread polls pynvml at 1 Hz (~10 μs per call, 0.001%% overhead) "
                         "so disabling it only makes sense if you are profiling at microsecond "
                         "resolution and want a completely clean baseline.")
    return ap.parse_args()


def main():
    args = parse_args()

    # ── Reproducibility: pin device, fix threads, fix seed ──────────────────
    gpu_idx = 0
    if args.device.startswith("cuda"):
        gpu_idx = _gpu_index_from_device(args.device)
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu_idx)

    # CPU thread budget: auto-detect unless user overrides.
    # With 4 simultaneous evals (1 per GPU) on a 4-GPU / 96-core server:
    #   auto = 96 physical cores / 4 GPUs = 24 threads per process
    # This prevents the 4 processes from each spawning 96 threads and trashing
    # each other's CPU L3 cache.  Within a single eval, prompts are sequential;
    # these threads are for PyTorch intra-op (BLAS) parallelism, not data loading.
    cpu_threads = args.cpu_threads if args.cpu_threads is not None else _auto_cpu_threads()
    torch.set_num_threads(cpu_threads)
    torch.set_num_interop_threads(cpu_threads)
    print(f"[init] CPU threads = {cpu_threads} "
          f"({'user override' if args.cpu_threads is not None else 'auto: cores/gpus'})")

    set_seed(args.seed)

    csv_path = args.output or RESULTS_CSV

    # Collect machine specs once — stored in every CSV row for reproducibility.
    specs = _machine_specs(gpu_idx)
    print(f"[hw]   GPU: {specs.get('machine_gpu', 'unknown')}  "
          f"({specs.get('machine_gpu_count', '?')} GPUs, "
          f"{specs.get('machine_gpu_vram_gb', '?')} GB VRAM each)  "
          f"CPU: {specs.get('machine_cpu_physical_cores', specs.get('machine_cpu_logical_cores', '?'))} physical cores  "
          f"RAM: {specs.get('machine_ram_gb', '?')} GB")

    # ── Load models (teacher first, then draft — always this order) ──────────
    print(f"[load] teacher={TEACHER_MODEL}")
    print(f"[load] draft={args.checkpoint}")
    print(f"[load] device={args.device}  dtype={DEFAULT_DTYPE}  seed={args.seed}")
    tok, p_model, q_model = load_models(TEACHER_MODEL, args.checkpoint,
                                        device=args.device, dtype=DEFAULT_DTYPE)

    # Log GPU memory after model load — both models share the same device.
    if torch.cuda.is_available():
        _alloc = torch.cuda.memory_allocated() / 1024**2
        _reserv = torch.cuda.memory_reserved() / 1024**2
        print(f"[gpu]  after model load — allocated={_alloc:.0f}MB  reserved={_reserv:.0f}MB")

    data_path = dataset_path(args.dataset)
    prompts   = load_prompts_jsonl(data_path)[:args.n]
    print(f"[data] {data_path} — {len(prompts)} prompts")

    modes = [m.strip() for m in args.modes.split(",")] if args.modes else [args.mode]

    print()
    print("=" * 78)
    print(f"  Eval  draft={args.checkpoint}  dataset={args.dataset}  "
          f"K={args.K} L={args.L} n={len(prompts)}")
    print(f"  device={args.device}  dtype={DEFAULT_DTYPE}  seed={args.seed}")
    print("=" * 78)

    gpu_idx = _gpu_index_from_device(args.device) if args.device.startswith("cuda") else 0
    for mode in modes:
        set_seed(args.seed)   # identical RNG state for every mode

        mon = GpuMonitor(device_idx=gpu_idx) if not args.no_gpu_monitor else None

        sp = _state_path(csv_path, mode, args.K, args.L, args.checkpoint)
        stats = evaluate_one_mode(
            p_model, q_model, tok, prompts, mode,
            K=args.K, L=args.L,
            max_new_tokens=args.max_new_tokens, temp=args.temp,
            state_path=sp, gpu_monitor=mon,
        )
        log_result(stats, args, mode, mon, csv_path,
                   specs=specs, cpu_threads_used=cpu_threads)

    print(f"\n[done] results appended to {csv_path}")


if __name__ == "__main__":
    main()
