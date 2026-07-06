"""
telemetry.py — optional GPU/CPU telemetry for eval runs (pynvml + psutil).

All code here degrades gracefully when pynvml / psutil are not installed.
Skip this file entirely if you do not need hardware monitoring.

Public API:
  _gpu_index_from_device("cuda:2") -> 2
  GpuMonitor(device_idx)  — 1-Hz background thread; call .start(), .stop(), .summary()
  _machine_specs(gpu_idx) -> dict  — one-shot hardware snapshot stored in every CSV row
  _model_attn_backend(*models) -> str  — selected HF backend from loaded model configs
  _auto_cpu_threads()     -> int   — physical_cores // num_gpus heuristic
"""
from __future__ import annotations

import json
import os
import platform
import threading
import time

import torch


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
    def start(self, sidecar_path: str | None = None):
        self._sidecar_path = sidecar_path
        self._samples = []
        if sidecar_path and os.path.exists(sidecar_path):
            try:
                with open(sidecar_path, encoding="utf-8") as _sf:
                    for _line in _sf:
                        _line = _line.strip()
                        if _line:
                            self._samples.append(json.loads(_line))
                print(f"  [gpu-resume] loaded {len(self._samples)} prior GPU samples"
                      f" from {os.path.basename(sidecar_path)}")
            except Exception as _e:
                print(f"  [gpu-resume] could not load GPU sidecar ({_e}) — starting fresh")
                self._samples = []
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        sidecar = getattr(self, "_sidecar_path", None)
        if sidecar and self._samples:
            try:
                with open(sidecar, "w", encoding="utf-8") as _sf:
                    for _s in self._samples:
                        _sf.write(json.dumps(_s) + "\n")
            except Exception as _e:
                print(f"  [gpu-resume] could not save GPU sidecar ({_e})")

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
                try:
                    # SM (shader) clock in MHz — drops under thermal throttle.
                    s["sm_clock_mhz"] = self._nvml.nvmlDeviceGetClockInfo(
                        self._handle, self._nvml.NVML_CLOCK_SM)
                    # Memory (HBM) clock in MHz.
                    s["mem_clock_mhz"] = self._nvml.nvmlDeviceGetClockInfo(
                        self._handle, self._nvml.NVML_CLOCK_MEM)
                except Exception:
                    pass
                try:
                    s["gpu_temp_c"] = self._nvml.nvmlDeviceGetTemperature(
                        self._handle, self._nvml.NVML_TEMPERATURE_GPU)
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
            ("gpu_sm_clock_avg_mhz",   lambda: _avg("sm_clock_mhz")),   # drops under throttle
            ("gpu_mem_clock_avg_mhz",  lambda: _avg("mem_clock_mhz")),
            ("gpu_temp_avg_c",         lambda: _avg("gpu_temp_c")),
            ("gpu_temp_peak_c",        lambda: _peak("gpu_temp_c")),
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
        if "gpu_sm_clock_avg_mhz" in s:
            parts.append(f"SM-clk={s['gpu_sm_clock_avg_mhz']:.0f}MHz")
        if "gpu_temp_avg_c" in s:
            parts.append(f"temp={s['gpu_temp_avg_c']:.0f}°C(peak={s.get('gpu_temp_peak_c','?')}°C)")
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


def _model_attn_backend(*models) -> str:
    """Return the attention backend selected by Transformers for loaded models."""
    impls: list[str] = []
    for model in models:
        cfg = getattr(model, "config", None)
        impl = getattr(cfg, "_attn_implementation", None)
        if impl and impl not in impls:
            impls.append(str(impl))

    try:
        import flash_attn
        flash_pkg = f"flash_attn-{flash_attn.__version__}"
    except ImportError:
        flash_pkg = ""

    if impls:
        backend = "+".join(impls)
        return f"{backend} ({flash_pkg})" if flash_pkg else backend
    return flash_pkg or "unknown"


def _auto_cpu_threads() -> int:
    """
    Compute a sensible default for torch CPU threads when running 1 eval per GPU.

    With N_gpus eval processes sharing C physical CPU cores, each process should
    use at most C // N_gpus threads so they don't thrash each other's CPU caches.
    For this workload (GPU-bound eval), the practical impact is small but it
    prevents the oversubscription scenario (e.g. 4 procs × 96 threads on a
    96-core box) when several evals run at once.

    Example: 96 cores, 4 GPUs → 24 threads per eval process.
    """
    n_gpus = max(1, torch.cuda.device_count() if torch.cuda.is_available() else 1)
    try:
        import psutil
        n_cores = psutil.cpu_count(logical=False) or os.cpu_count() or 1
    except Exception:
        n_cores = os.cpu_count() or 1
    return max(1, n_cores // n_gpus)
