"""
Hardware-aware parallel job scheduler for the SpecDist pipeline.

Detects GPU topology at runtime and dispatches pipeline steps to GPU slots,
maximising throughput without touching evaluate.py or trainer.py.

Design: each "slot" is a queue-entry GPU assignment. Steps are dispatched
to the slot with most free capacity via a thread pool. CUDA_VISIBLE_DEVICES is
injected into each subprocess's environment to pin it to the assigned GPU(s).

Parallel groups (topological ordering):
  group  0: baseline eval     — must run first, serial
  group  1: all training jobs — independent, parallel across GPUs
  group  2: LoRA merges       — serial, runs after all training completes
  group  3: all post-train evals — independent, parallel across GPUs
  group -1: always serial    — setup, PPL, admin steps

Note: groups 2 (merges) and -1 (admin) run through the existing run_step()
path in experiment.py so all state management, retry logic, and pipeline
logging is preserved exactly. Groups 1 and 3 are dispatched here in parallel.

VRAM-per-job estimates:
  Training : ~10.0 GB  (teacher NF4 4.5 + student 2.5 + optimizer 2.5 + buffer 0.5)
  Eval     :  ~6.5 GB  (teacher NF4 4.5 + draft 1.2 + runner overhead 0.8)

Slot formula per GPU: floor(gpu_vram_gb / vram_per_job_gb), capped at 3.
"""

import os
import queue
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from math import floor
from typing import Optional

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
_GBV_ROOT = os.path.dirname(HERE)   # gbv-research/

TRAIN_VRAM_GB = 10.0   # estimated VRAM per training job
EVAL_VRAM_GB  = 6.5    # estimated VRAM per eval job
MAX_SLOTS_PER_GPU = 3  # safety cap: never run more than 3 jobs per GPU

_stdout_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GPUSlot:
    """Represents one schedulable unit of GPU capacity."""
    device_ids: list   # GPU indices visible to this slot — usually [i]
    vram_gb: float     # VRAM available for this slot
    name: str          # Human-readable label, e.g. "Tesla T4 #0"


@dataclass
class HWProfile:
    """Describes detected hardware and derived slot pools."""
    slots:        list   # list[GPUSlot] — one per physical GPU
    train_slots:  list   # list[GPUSlot] — may have multiple entries per GPU
    eval_slots:   list   # list[GPUSlot] — may have more entries than train_slots
    profile_name: str    # e.g. "T4x2", "P100", "A100-40GB", "cpu"

    @classmethod
    def from_runtime(cls) -> "HWProfile":
        """Auto-detect GPU topology and build the appropriate HWProfile."""
        try:
            import torch
        except ImportError:
            return cls._cpu_profile()

        if not torch.cuda.is_available():
            return cls._cpu_profile()

        n_gpus = torch.cuda.device_count()
        if n_gpus == 0:
            return cls._cpu_profile()

        gpus = [(i, torch.cuda.get_device_properties(i)) for i in range(n_gpus)]

        all_slots:   list = []
        train_slots: list = []
        eval_slots:  list = []

        # ── Small multi-GPU (T4/V100-class, <20 GB each) → ONE spanning slot ──
        # A large teacher (e.g. Qwen3-8B) loaded in 4-bit NF4 has a transient
        # BF16 materialization spike (~16 GB) that does NOT fit a single 15 GB
        # T4 — it must spread across BOTH GPUs via device_map="auto".  Making
        # one slot per GPU (the default below) pins each parallel job to a
        # single T4 (CUDA_VISIBLE_DEVICES=i), so device_map sees only 15 GB and
        # OOMs at ~85% during materialization.  For small multi-GPU we therefore
        # build ONE slot covering ALL devices: jobs run sequentially, but each
        # uses both T4s (CUDA_VISIBLE_DEVICES="0,1") and the 8B load fits across
        # the combined ~30 GB.  A100-class GPUs (≥20 GB) each hold the full
        # model, so they keep the per-GPU parallel slots.
        _vrams = [props.total_memory / (1024 ** 3) for _, props in gpus]
        _small_multi = len(gpus) > 1 and all(v < 20 for v in _vrams)

        if _small_multi:
            _all_ids   = [i for i, _ in gpus]
            _total_vram = sum(_vrams)
            _span_name = "+".join(f"{props.name}#{i}" for i, props in gpus)
            for i, props in gpus:
                all_slots.append(GPUSlot(device_ids=[i],
                                         vram_gb=props.total_memory / (1024 ** 3),
                                         name=f"{props.name} #{i}"))
            # One spanning slot each for train and eval — both load the large
            # teacher and both need the combined VRAM for the load spike.
            train_slots.append(GPUSlot(device_ids=_all_ids, vram_gb=_total_vram,
                                       name=_span_name))
            eval_slots.append(GPUSlot(device_ids=_all_ids, vram_gb=_total_vram,
                                      name=_span_name))
        else:
            for i, props in gpus:
                vram    = props.total_memory / (1024 ** 3)
                gpu_name = f"{props.name} #{i}"

                n_train = max(1, min(MAX_SLOTS_PER_GPU, floor(vram / TRAIN_VRAM_GB)))
                n_eval  = max(1, min(MAX_SLOTS_PER_GPU, floor(vram / EVAL_VRAM_GB)))

                all_slots.append(GPUSlot(device_ids=[i], vram_gb=vram, name=gpu_name))
                for _ in range(n_train):
                    train_slots.append(GPUSlot(device_ids=[i], vram_gb=vram, name=gpu_name))
                for _ in range(n_eval):
                    eval_slots.append(GPUSlot(device_ids=[i], vram_gb=vram, name=gpu_name))

        profile_name = cls._detect_profile_name(gpus)
        return cls(
            slots=all_slots,
            train_slots=train_slots,
            eval_slots=eval_slots,
            profile_name=profile_name,
        )

    @staticmethod
    def _detect_profile_name(gpus: list) -> str:
        n = len(gpus)
        if n == 0:
            return "cpu"
        names = [props.name for _, props in gpus]
        vrams = [props.total_memory / (1024 ** 3) for _, props in gpus]
        total_vram = sum(vrams)
        base = names[0]
        if n == 1:
            if "P100" in base:
                return "P100"
            if "A100" in base:
                return "A100-80GB" if vrams[0] >= 70 else "A100-40GB"
            if "T4" in base:
                return "T4"
            return f"GPU-{total_vram:.0f}GB"
        # Multi-GPU
        if all("T4" in nm for nm in names):
            return f"T4x{n}"
        if all("A100" in nm for nm in names):
            return f"A100x{n}"
        return f"{n}GPU-{total_vram:.0f}GB"

    @classmethod
    def _cpu_profile(cls) -> "HWProfile":
        cpu_slot = GPUSlot(device_ids=[], vram_gb=0.0, name="CPU")
        return cls(
            slots=[cpu_slot],
            train_slots=[cpu_slot],
            eval_slots=[cpu_slot],
            profile_name="cpu",
        )


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class HWScheduler:
    """
    Dispatches pipeline steps to GPU slots, parallelising where safe.

    Usage (from experiment.py main loop):
        _scheduler = HWScheduler(HWProfile.from_runtime())
        results = _scheduler.run_parallel(train_steps, job_type="train")
        # results: {"train_kl_gsm8k": 0, "train_ebe_gsm8k": 0, ...}
    """

    def __init__(self, profile: HWProfile):
        self.profile = profile
        self._print_summary()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_parallel(self, steps: list, job_type: str = "eval",
                     on_complete=None) -> dict:
        """
        Run all steps in parallel across available GPU slots.

        Parameters
        ----------
        steps : list of step dicts
            Each dict must have at minimum:
              "cmd"       : list[str]   — the subprocess command
              "id" or "name" : str      — step identifier for the result dict
            Optional keys:
              "desc"      : str         — human-readable description
              "env_extra" : dict        — extra env vars injected into the process

        job_type : "train" | "eval"
            Selects which slot pool to use (train_slots vs eval_slots).

        on_complete : callable(step_id: str, rc: int) | None
            If provided, called immediately when each step finishes (before all
            steps complete).  Use this to update pipeline state incrementally so
            the dashboard shows green chips as each step finishes rather than
            waiting for the entire parallel group.

        Returns
        -------
        dict[str, int]
            Maps each step's id/name to its subprocess exit code.
        """
        if not steps:
            return {}

        slots = (self.profile.train_slots if job_type == "train"
                 else self.profile.eval_slots)

        # Slot queue acts as a counting semaphore: at most len(slots) concurrent jobs.
        slot_q: queue.Queue = queue.Queue()
        for s in slots:
            slot_q.put(s)

        results: dict = {}

        def _run_one(step: dict):
            step_id = step.get("id") or step.get("name", "unknown")
            slot: GPUSlot = slot_q.get()          # block until a slot is free
            try:
                rc = self._launch(step, slot)
                return step_id, rc
            finally:
                slot_q.put(slot)                  # always return slot to pool

        with ThreadPoolExecutor(max_workers=len(steps)) as exe:
            futures = {exe.submit(_run_one, s): s for s in steps}
            for fut in as_completed(futures):
                try:
                    step_id, rc = fut.result()
                    results[step_id] = rc
                except Exception as exc:
                    step = futures[fut]
                    step_id = step.get("id") or step.get("name", "unknown")
                    with _stdout_lock:
                        print(f"  [scheduler] ERROR in {step_id}: {exc}", flush=True)
                    rc = 1
                    results[step_id] = rc
                # Fire per-step callback immediately so state updates incrementally
                # and the dashboard shows green chips as each step finishes.
                if on_complete is not None:
                    try:
                        on_complete(step_id, rc)
                    except Exception as _cb_err:
                        with _stdout_lock:
                            print(f"  [scheduler] on_complete error for {step_id}: {_cb_err}",
                                  flush=True)

        return results

    def run_sequential(self, steps: list) -> dict:
        """
        Run steps one at a time on slot[0] (for serial admin phases).

        Returns dict[step_id, exit_code].  Exists so callers need not special-
        case single-GPU vs multi-GPU; the scheduler handles environment setup.
        """
        if not steps:
            return {}

        slots = self.profile.eval_slots or self.profile.train_slots
        slot = slots[0] if slots else GPUSlot(device_ids=[], vram_gb=0.0, name="CPU")

        results: dict = {}
        for step in steps:
            step_id = step.get("id") or step.get("name", "unknown")
            rc = self._launch(step, slot)
            results[step_id] = rc
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _launch(self, step: dict, slot: GPUSlot) -> int:
        """
        Spawn one subprocess pinned to `slot`, stream output, return exit code.
        Thread-safe: stdout lines are prefixed with the step id and printed
        under _stdout_lock so multi-threaded output stays readable.
        """
        step_id   = step.get("id") or step.get("name", "unknown")
        gpu_label = f"GPU{slot.device_ids[0]}" if slot.device_ids else "CPU"

        env = os.environ.copy()

        # GPU pinning: set CUDA_VISIBLE_DEVICES for this slot.
        if slot.device_ids:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(d) for d in slot.device_ids)
        else:
            env.setdefault("CUDA_VISIBLE_DEVICES", "")

        # Merge any per-step environment extras.
        env.update(step.get("env_extra") or {})

        # Standard subprocess environment — mirrors run_step() in experiment.py.
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        _existing_pypath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            _GBV_ROOT + os.pathsep + _existing_pypath
        ).rstrip(os.pathsep)
        # Propagate persistent storage paths (set by experiment.py on startup).
        for _ekey in ("SPECDIST_DB_PATH", "SPECDIST_LOGS_ROOT", "SPECDIST_STORAGE_ROOT"):
            if os.environ.get(_ekey):
                env[_ekey] = os.environ[_ekey]

        with _stdout_lock:
            desc = step.get("desc", step_id)
            print(f"\n  [scheduler] >> {step_id} -> {slot.name} ({gpu_label})")
            print(f"  [scheduler]   {desc}")
            sys.stdout.flush()

        t0 = time.time()

        try:
            proc = subprocess.Popen(
                step["cmd"],
                cwd=HERE,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                encoding="utf-8",
                errors="replace",
            )

            output_lines: list = []
            for line in iter(proc.stdout.readline, ""):
                output_lines.append(line)
                with _stdout_lock:
                    sys.stdout.write(f"  [{step_id}] {line}")
                    sys.stdout.flush()

            rc      = proc.wait()
            elapsed = time.time() - t0

        except Exception as exc:
            with _stdout_lock:
                print(f"  [scheduler] ✗ {step_id}: exception during launch: {exc}", flush=True)
            return 1

        # Append to the pipeline log so the dashboard can show parallel steps.
        self._append_to_log(step, step_id, slot, output_lines, rc, elapsed)

        status_sym = "✓" if rc == 0 else "✗"
        with _stdout_lock:
            print(f"  [scheduler] {status_sym} {step_id} — {elapsed/60:.1f} min  rc={rc}", flush=True)

        return rc

    @staticmethod
    def _append_to_log(step: dict, step_id: str, slot: GPUSlot,
                       output_lines: list, rc: int, elapsed: float):
        """Append this step's output to the pipeline log (best-effort)."""
        _log_dir  = os.environ.get(
            "SPECDIST_LOGS_ROOT",
            os.path.join(_GBV_ROOT, "db", "logs"),
        )
        _log_path = os.path.join(_log_dir, "pipeline_output.log")
        try:
            os.makedirs(_log_dir, exist_ok=True)
            with open(_log_path, "a", encoding="utf-8", errors="replace") as lf:
                lf.write(f"\n{'='*65}\n")
                lf.write(f"  STEP (parallel): {step.get('desc', step_id)}\n")
                lf.write(f"  GPU  : {slot.name}\n")
                lf.write(f"  CMD  : {' '.join(str(t) for t in step['cmd'])}\n")
                lf.write(f"  TIME : {elapsed/60:.1f} min  RC={rc}\n")
                lf.write(f"{'='*65}\n")
                lf.writelines(output_lines[-200:])   # last 200 lines per step
        except OSError:
            pass

    def _print_summary(self):
        """Print hardware summary on scheduler construction."""
        p   = self.profile
        n_t = len(p.train_slots)
        n_e = len(p.eval_slots)

        # Fall back to ASCII arrows on terminals that can't encode Unicode.
        _enc = getattr(sys.stdout, "encoding", None) or "ascii"
        try:
            "\u2192".encode(_enc)
            _arrow = "\u2192"
        except (UnicodeEncodeError, LookupError):
            _arrow = "->"

        print(f"\n[scheduler] Detected: {p.profile_name} -- "
              f"{n_t} train slot(s) / {n_e} eval slot(s)")
        if p.profile_name == "cpu":
            print(f"  CPU-only mode -- all steps run on CPU (slow but correct)")
        else:
            for slot in p.slots:
                vram  = slot.vram_gb
                n_ts  = max(1, min(MAX_SLOTS_PER_GPU, floor(vram / TRAIN_VRAM_GB)))
                n_es  = max(1, min(MAX_SLOTS_PER_GPU, floor(vram / EVAL_VRAM_GB)))
                print(f"  {slot.name}  {vram:.1f} GB "
                      f"{_arrow} {n_ts} train, {n_es} eval slot(s)")
        print()


# ---------------------------------------------------------------------------
# Quick smoke-test (python hw_scheduler.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    profile = HWProfile.from_runtime()
    sched   = HWScheduler(profile)
    print(f"Profile      : {profile.profile_name}")
    print(f"Train slots  : {len(profile.train_slots)}")
    print(f"Eval  slots  : {len(profile.eval_slots)}")
    print("hw_scheduler OK")
