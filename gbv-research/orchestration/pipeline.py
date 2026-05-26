"""
pipeline.py — master run script with crash-safe resume.

Runs the full SpecDist experiment pipeline in order:
  1. Merge pending LoRA checkpoints
  2. Train KL and EBE on real GSM8K data (overnight steps)
  3. Evaluate all models on all datasets with full hyperparameter sweep

State is persisted to pipeline_state.json after every step.
On restart, the script shows what's done and asks before resuming.
Eval steps always pass --skip_existing to run_all.py, so partial evals
are safe to re-run — they pick up from the last completed cell.

Usage:
    python pipeline.py                        # laptop config, interactive
    python pipeline.py --yes                  # auto-resume without prompts
    python pipeline.py --config server --yes  # Qwen3-0.6B -> Qwen3-8B
    python pipeline.py --restart              # force-restart from step 1
    python pipeline.py --from STEP_ID         # resume from a specific step
    python pipeline.py --dry_run              # print plan without running
    python pipeline.py --status               # print current status and exit
"""

import argparse
import atexit
import io
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime

# Windows cp1252 stdout can't encode Unicode arrows/checkmarks used by subprocesses.
# Reconfigure to UTF-8 with replacement so pipeline.py never dies on a stray character.
# Guard with __name__ == "__main__" so importing pipeline.py in tests doesn't break
# pytest's stdout capture (which holds open file handles that get invalidated by
# sys.stdout = io.TextIOWrapper(...) at module level).
def _reconfigure_stdout_for_windows():
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    if hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(
            sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)

HERE = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(HERE, "pipeline_state_laptop.json")  # overridden in main()

# ---------------------------------------------------------------------------
# Per-user WandB config (gitignored, never committed)
#
# Each researcher creates orchestration/wandb_config.json on their own machine:
#   {
#     "api_key": "wandb_v1_...",        # from wandb.ai/settings → API Keys
#     "entity":  "my-wandb-username",   # or team name
#     "project": "specdist-gbv"         # optional, defaults to "distillspec"
#   }
#
# On startup the pipeline sets WANDB_API_KEY / WANDB_ENTITY / WANDB_PROJECT
# env vars, which every subprocess (train + eval) inherits automatically.
# If the file is absent the pipeline falls back to whatever `wandb login`
# stored in ~/.netrc — so CI / Colab runs work without any file.
# ---------------------------------------------------------------------------

def _load_wandb_config():
    """Load wandb_config.json if present and apply as environment variables."""
    cfg_path = os.path.join(HERE, "wandb_config.json")
    if not os.path.exists(cfg_path):
        return  # fall back to cached wandb login (~/.netrc)
    try:
        with open(cfg_path, encoding="utf-8") as f:
            wc = json.load(f)
        if wc.get("api_key"):
            os.environ["WANDB_API_KEY"] = wc["api_key"]
        if wc.get("entity"):
            os.environ["WANDB_ENTITY"] = wc["entity"]
        if wc.get("project"):
            os.environ["WANDB_PROJECT"] = wc["project"]
        entity = wc.get("entity", "(from login)")
        project = wc.get("project", "distillspec")
        print(f"  [wandb] Loaded config: entity={entity}  project={project}")
    except Exception as e:
        print(f"  [wandb] Warning: could not read wandb_config.json: {e}")

# ---------------------------------------------------------------------------
# Zombie / concurrent-invocation prevention
#
# Root cause of the "same step ran 3 times" problem:
#   The user ran pipeline.py 3 times in the same terminal session.  Each time
#   the old pipeline.py process had been killed (OOM, Ctrl+C, power cycle) but
#   its child subprocess (run_all.py) was left running as an ORPHAN — it kept
#   holding GPU VRAM and the state file showed the step as "running".  On the
#   next restart, the old code fell through and re-ran the step concurrently
#   with any still-alive orphaned child.
#
# Fix:
#   1. A PID lock file (.pipeline_lock) tracks the pipeline PID + active child
#      PID.  On startup we check whether those PIDs are still alive and KILL
#      the whole process tree before proceeding.
#   2. run_step() uses Popen so we hold the child Popen handle and can
#      terminate it cleanly on SIGTERM/Ctrl-C.
#   3. PYTHONIOENCODING=utf-8 is injected into every subprocess environment
#      so Unicode characters in run_all.py output never crash on Windows cp1252.
# ---------------------------------------------------------------------------

_LOCK_FILE    = os.path.join(HERE, ".pipeline_lock")          # runtime artifact — stays in orchestration/
_GBV_RESEARCH_ROOT = os.path.dirname(HERE)                   # gbv-research/
_DB_LOGS      = os.path.join(_GBV_RESEARCH_ROOT, "db", "logs")
_PIPELINE_LOG = os.path.join(_DB_LOGS, "pipeline_output.log")  # live log visible in dashboard Logs panel
_child_popen  = None   # Popen handle for the currently running child step


def _pid_alive(pid: int) -> bool:
    """Return True if the process with pid is still running."""
    if sys.platform == "win32":
        try:
            import ctypes
            STILL_ACTIVE = 259
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            h = ctypes.windll.kernel32.OpenProcess(
                PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if not h:
                return False
            code = ctypes.c_ulong(0)
            ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code))
            ctypes.windll.kernel32.CloseHandle(h)
            return code.value == STILL_ACTIVE
        except Exception:
            return False
    else:
        try:
            os.kill(pid, 0)
            return True
        except (ProcessLookupError, PermissionError):
            return False


def _kill_tree(pid: int):
    """Kill a process and all its children (cross-platform)."""
    if sys.platform == "win32":
        # /T = kill entire process tree (children too — catches orphaned run_all.py)
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(pid)],
            capture_output=True, timeout=15,
        )
    else:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


def _write_lock(child_pid: int = None):
    """Write pipeline PID (+ optional child PID) to lock file."""
    with open(_LOCK_FILE, "w") as f:
        f.write(f"{os.getpid()}\n")
        if child_pid:
            f.write(f"{child_pid}\n")


def _release_lock():
    """Remove lock file on clean exit."""
    try:
        os.remove(_LOCK_FILE)
    except OSError:
        pass


atexit.register(_release_lock)


def _acquire_lock():
    """
    On startup: if a stale lock file exists, kill every PID listed in it
    (the previous pipeline.py AND its run_all.py child) before proceeding.
    This is what frees the GPU VRAM held by orphaned processes.
    """
    if not os.path.exists(_LOCK_FILE):
        _write_lock()
        return

    try:
        with open(_LOCK_FILE) as f:
            pids = [int(line.strip()) for line in f if line.strip().isdigit()]
    except Exception:
        pids = []

    alive = [p for p in pids if p != os.getpid() and _pid_alive(p)]
    if alive:
        print(f"\n  [CLEANUP] Found {len(alive)} stale pipeline process(es): {alive}")
        print(f"  Killing them now to free GPU VRAM and prevent duplicate runs...")
        for p in alive:
            _kill_tree(p)
            print(f"  Killed PID {p} (+ its children via /T)")
        time.sleep(1)   # brief pause so OS reclaims VRAM before we load models
        print()

    _write_lock()   # overwrite with our own PID

# ---------------------------------------------------------------------------
# Hardware configs — controls which model pair is used
# ---------------------------------------------------------------------------

CONFIGS = {
    "laptop": {
        "draft":  "Qwen/Qwen2.5-0.5B",
        "target": "Qwen/Qwen3-0.6B",
        "desc":   "Laptop/low-VRAM: Qwen2.5-0.5B draft -> Qwen3-0.6B target",
    },
    "server": {
        "draft":  "Qwen/Qwen3-0.6B",
        "target": "Qwen/Qwen3-8B",
        "desc":   "Server/A100: Qwen3-0.6B draft -> Qwen3-8B target",
        # A100/3090 (24 GB+): 8B in bfloat16 fits fine — no quantisation needed.
    },
    "colab": {
        "draft":  "Qwen/Qwen3-0.6B",
        "target": "Qwen/Qwen3-8B",
        "desc":   "Google Colab free T4 (15 GB): 8B teacher in 4-bit NF4 + 0.6B draft in bfloat16",
        # Free Colab T4 has 15 GB VRAM.  Qwen3-8B in bfloat16 = ~16 GB → OOM.
        # Loading the frozen teacher in 4-bit NF4 (bitsandbytes QLoRA) reduces it
        # to ~5 GB; total with draft + activations ≈ 8–9 GB → comfortable T4 fit.
        # The draft is trained in bfloat16 as normal — only the frozen teacher is quantised.
        "load_in_4bit": True,
    },
}

# ---------------------------------------------------------------------------
# Step definitions
# Each step has:
#   id          : unique key for state tracking
#   desc        : human-readable description
#   cmd         : command list (run from HERE directory)
#   done_check  : file that must exist for this step to be auto-skipped
#   group       : phase label for display
# ---------------------------------------------------------------------------

_GBV_RESEARCH = os.path.dirname(HERE)   # gbv-research/
_OSD_DIR = os.path.join(os.path.dirname(_GBV_RESEARCH), "OSD")   # OSD/ (sibling)
_GBV_SRC = os.path.join(os.path.dirname(_GBV_RESEARCH), "GBV")   # GBV/ (sibling)

# Absolute paths to battle-tested OSD scripts (do NOT modify these originals)
_TRAIN_SCRIPT  = os.path.join(_OSD_DIR, "train_qwen3.py")   # TODO: replace with algorithms.distillspec_gbv.trainer once import paths are fixed
_ONLINE_SCRIPT = os.path.join(_OSD_DIR, "online_serve.py")


def _ckpt(name):
    """
    Return the canonical checkpoint path, falling back to OSD/ if the
    gbv-research path doesn't exist yet (migration period).

    Priority:
      1. gbv-research/db/checkpoints/<name>  (canonical new location)
      2. OSD/checkpoints/<name>              (legacy — pre-refactor artifacts)

    New training always lands in gbv-research/db/checkpoints/ because
    _TRAIN_SCRIPT uses the --output path we supply.  Once a checkpoint is
    re-created there the OSD fallback is no longer needed.
    """
    gbv_path = os.path.join(_GBV_RESEARCH, "db", "checkpoints", name)
    if not os.path.exists(gbv_path):
        osd_path = os.path.join(_OSD_DIR, "checkpoints", name)
        if os.path.exists(osd_path):
            return osd_path
    return gbv_path

def _data(name): return os.path.join(_GBV_RESEARCH, "core", "datasets", "raw", name)
def _merged(name): return _ckpt(name + "_merged")


def _eval_cmd(student_path, label, teacher, datasets="gsm8k",
              modes="alpha,specinfer,gbv,traversal",
              Ks="3,5", temps="0.6,1.0", n=10, max_tokens=50, task_score=False,
              experiment_tag=None):
    """Eval command — always passes --skip_existing so restarts are safe.

    Defaults (laptop): n=10 prompts, max_tokens=50.  Run with n=30/max_tokens=100
    on Colab/server T4 for paper-quality results.
    """
    cmd = [
        sys.executable, os.path.join(HERE, "run_all.py"),
        "--student", student_path,
        "--teacher", teacher,
        "--student_label", label,
        "--datasets", datasets,
        "--modes", modes,
        "--K", Ks,
        "--temperature", temps,
        "--n", str(n),
        "--max_tokens", str(max_tokens),
        "--skip_existing",
        "--skip_fetch",
    ]
    if task_score:
        cmd.append("--task_score")
    if experiment_tag:
        cmd += ["--experiment_tag", experiment_tag]
    return cmd


def build_steps(draft, target, experiment_tag=None, smoke=False, eagle=False,
                load_in_4bit=False):
    """Build the STEPS list for a given draft/target model pair.

    Pipeline structure (same for both smoke and full — only numbers differ):

        Phase 1 — Baseline
            eval_baseline_gsm8k   (always first; establishes the untrained reference)

        Phase 2 — Training   (smoke: 50 steps · full: 1000 steps)
            forward_kl · ebe · reverse_kl · jsd · l1 · online
            Each loss: train → merge (sequential pairs)

        Phase 3 — GSM8K Eval   (smoke: n=5 · full: n=10, all 6 verifier modes)
            eval every trained model + baseline on gsm8k

        Phase 4 — Multi-dataset   (always skipped in smoke; slow)
            eval top models on humaneval, math500, mtbench, alpaca

    smoke=True:   50 steps/training · 50 online prompts · n=5 eval prompts ·
                  max_tokens=30 · K=3 · all 6 verifier modes · temp=0.6.
                  Exercises every loss + every verifier in ~30–40 min on laptop.
                  Purpose: catch crashes / NaN / shape errors before overnight run.
                  Uses a SEPARATE state file (pipeline_state_<config>_smoke.json)
                  so smoke "done" marks never block the real pipeline.

    smoke=False:  1000 steps/training · n=10 eval prompts · max_tokens=50 ·
                  K=3+5 · all 6 verifier modes · temps=0.6+1.0.
                  ~6–8 hrs total on laptop; use Colab/server for paper results.

    eagle=True:   Append Phase 5 — EAGLE Benchmark.  Only meaningful with
                  --config server/colab (Qwen3-8B target).
    """
    _steps         = 50    if smoke else 1000
    _online_steps  = 50    if smoke else 500
    _n             = 5     if smoke else 10
    _max_tok       = 30    if smoke else 50
    _Ks            = "3"   if smoke else "3,5"
    _temps         = "0.6" if smoke else "0.6,1.0"
    # All 6 verifier modes in both smoke and full — smoke is comprehensive by design.
    _modes = "alpha,bv,gbv,traversal,specinfer,naive"
    # 4-bit flag appended to every training/eval command when load_in_4bit=True.
    # Only set for --config colab (free T4, 15 GB).  Server/A100 loads bf16.
    _4bit = ["--load_in_4bit"] if load_in_4bit else []

    def _ec(student_path, label, datasets="gsm8k", task_score=False):
        """Shorthand: eval cmd with smoke-aware parameters."""
        cmd = _eval_cmd(student_path, label, target,
                        datasets=datasets, modes=_modes, Ks=_Ks, temps=_temps,
                        n=_n, max_tokens=_max_tok,
                        task_score=task_score, experiment_tag=experiment_tag)
        return cmd + _4bit  # append --load_in_4bit for colab config

    return [
        # -------------------------------------------------------------------
        # Phase 1 — Baseline
        # Run first every time so the untrained draft model's block efficiency
        # is in results.db before any trained model is evaluated.  This is the
        # comparison anchor for all Phase 3/4 results.
        # -------------------------------------------------------------------
        {
            "id": "eval_baseline_gsm8k",
            "group": "Phase 1 — Baseline",
            "desc": "Eval unmodified draft on gsm8k (all 6 verifier modes)",
            "cmd": _ec(draft, "baseline", datasets="gsm8k", task_score=True),
            "done_check": None,
        },

        # -------------------------------------------------------------------
        # Phase 2 — Training
        #
        # smoke: 50 steps each — just enough to hit the first checkpoint,
        #        trigger the health check, and verify no NaN / OOM / shape error.
        # full:  1000 steps each (~1 hr/loss on laptop, ~15 min on A100).
        #
        # Losses covered: forward_kl · ebe · reverse_kl · jsd · l1 · online
        # All use lr=3e-5.  EBE / revKL / JSD / L1 use --nan_action skip +
        # --early_stop_patience 3 as a safety net for unstable losses.
        # -------------------------------------------------------------------
        {
            "id": "train_kl_gsm8k",
            "group": "Phase 2 — Training",
            "desc": f"Train forward_kl, {_steps} steps, gsm8k_train",
            "cmd": [
                sys.executable, _TRAIN_SCRIPT,
                "--loss", "forward_kl",
                "--steps", str(_steps), "--lr", "3e-5",
                "--draft", draft, "--target", target,
                "--dataset", _data("gsm8k_train.jsonl"),
                "--output", _ckpt("kl-gsm8k"),
            ] + _4bit,
            "done_check": os.path.join(_ckpt("kl-gsm8k"), "adapter_model.safetensors"),
            "retryable": True,
        },
        {
            "id": "merge_kl_gsm8k",
            "group": "Phase 2 — Training",
            "desc": "Merge kl-gsm8k LoRA",
            "cmd": [sys.executable, _TRAIN_SCRIPT, "--merge_only",
                    "--adapter", _ckpt("kl-gsm8k"), "--draft", draft],
            "done_check": os.path.join(_merged("kl-gsm8k"), "config.json"),
        },
        {
            "id": "train_ebe_gsm8k",
            "group": "Phase 2 — Training",
            "desc": f"Train ebe, {_steps} steps, gsm8k_train",
            "cmd": [
                sys.executable, _TRAIN_SCRIPT,
                "--loss", "ebe",
                "--steps", str(_steps), "--lr", "3e-5",
                "--nan_action", "skip",
                "--early_stop_patience", "3",
                "--draft", draft, "--target", target,
                "--dataset", _data("gsm8k_train.jsonl"),
                "--output", _ckpt("ebe-gsm8k"),
            ] + _4bit,
            "done_check": os.path.join(_ckpt("ebe-gsm8k"), "adapter_model.safetensors"),
            "retryable": True,
        },
        {
            "id": "merge_ebe_gsm8k",
            "group": "Phase 2 — Training",
            "desc": "Merge ebe-gsm8k LoRA",
            "cmd": [sys.executable, _TRAIN_SCRIPT, "--merge_only",
                    "--adapter", _ckpt("ebe-gsm8k"), "--draft", draft],
            "done_check": os.path.join(_merged("ebe-gsm8k"), "config.json"),
        },
        {
            "id": "train_rev_kl_gsm8k",
            "group": "Phase 2 — Training",
            "desc": f"Train reverse_kl, {_steps} steps, gsm8k_train",
            "cmd": [
                sys.executable, _TRAIN_SCRIPT,
                "--loss", "reverse_kl",
                "--steps", str(_steps), "--lr", "3e-5",
                "--nan_action", "skip", "--early_stop_patience", "3",
                "--draft", draft, "--target", target,
                "--dataset", _data("gsm8k_train.jsonl"),
                "--output", _ckpt("rev_kl-gsm8k"),
            ] + _4bit,
            "done_check": os.path.join(_ckpt("rev_kl-gsm8k"), "adapter_model.safetensors"),
            "retryable": True,
        },
        {
            "id": "merge_rev_kl_gsm8k",
            "group": "Phase 2 — Training",
            "desc": "Merge rev_kl-gsm8k LoRA",
            "cmd": [sys.executable, _TRAIN_SCRIPT, "--merge_only",
                    "--adapter", _ckpt("rev_kl-gsm8k"), "--draft", draft],
            "done_check": os.path.join(_merged("rev_kl-gsm8k"), "config.json"),
        },
        {
            "id": "train_jsd_gsm8k",
            "group": "Phase 2 — Training",
            "desc": f"Train jsd, {_steps} steps, gsm8k_train",
            "cmd": [
                sys.executable, _TRAIN_SCRIPT,
                "--loss", "jsd",
                "--steps", str(_steps), "--lr", "3e-5",
                "--nan_action", "skip", "--early_stop_patience", "3",
                "--draft", draft, "--target", target,
                "--dataset", _data("gsm8k_train.jsonl"),
                "--output", _ckpt("jsd-gsm8k"),
            ] + _4bit,
            "done_check": os.path.join(_ckpt("jsd-gsm8k"), "adapter_model.safetensors"),
            "retryable": True,
        },
        {
            "id": "merge_jsd_gsm8k",
            "group": "Phase 2 — Training",
            "desc": "Merge jsd-gsm8k LoRA",
            "cmd": [sys.executable, _TRAIN_SCRIPT, "--merge_only",
                    "--adapter", _ckpt("jsd-gsm8k"), "--draft", draft],
            "done_check": os.path.join(_merged("jsd-gsm8k"), "config.json"),
        },
        {
            "id": "train_l1_gsm8k",
            "group": "Phase 2 — Training",
            "desc": f"Train l1, {_steps} steps, gsm8k_train",
            "cmd": [
                sys.executable, _TRAIN_SCRIPT,
                "--loss", "l1",
                "--steps", str(_steps), "--lr", "3e-5",
                "--nan_action", "skip", "--early_stop_patience", "3",
                "--draft", draft, "--target", target,
                "--dataset", _data("gsm8k_train.jsonl"),
                "--output", _ckpt("l1-gsm8k"),
            ] + _4bit,
            "done_check": os.path.join(_ckpt("l1-gsm8k"), "adapter_model.safetensors"),
            "retryable": True,
        },
        {
            "id": "merge_l1_gsm8k",
            "group": "Phase 2 — Training",
            "desc": "Merge l1-gsm8k LoRA",
            "cmd": [sys.executable, _TRAIN_SCRIPT, "--merge_only",
                    "--adapter", _ckpt("l1-gsm8k"), "--draft", draft],
            "done_check": os.path.join(_merged("l1-gsm8k"), "config.json"),
        },
        {
            "id": "online_adapt_gsm8k",
            "group": "Phase 2 — Training",
            "desc": f"Online OSD adaptation (forward_kl, K=4), {_online_steps} prompts, gsm8k",
            "cmd": [
                sys.executable, _ONLINE_SCRIPT,
                "--prompts", _data("gsm8k_train.jsonl"),
                "--draft", draft, "--target", target,
                "--output", _ckpt("online-gsm8k"),
                "--steps", str(_online_steps),
                "--update_every", "4",
                "--K", "4",
                "--kl_method", "forward_kl",
                "--lr", "3e-4",
                "--max_new_tokens", "128",
            ] + _4bit,
            "done_check": os.path.join(_ckpt("online-gsm8k"), "adapter_model.safetensors"),
            "retryable": True,
        },
        {
            "id": "merge_online_gsm8k",
            "group": "Phase 2 — Training",
            "desc": "Merge online-gsm8k LoRA",
            "cmd": [sys.executable, _TRAIN_SCRIPT, "--merge_only",
                    "--adapter", _ckpt("online-gsm8k"), "--draft", draft],
            "done_check": os.path.join(_merged("online-gsm8k"), "config.json"),
        },

        # -------------------------------------------------------------------
        # Phase 3 — GSM8K Eval
        # All 6 verifier modes (alpha · bv · gbv · traversal · specinfer · naive)
        # smoke: n=5 prompts, max_tokens=30, K=3, temp=0.6
        # full:  n=10 prompts, max_tokens=50, K=3+5, temps=0.6+1.0
        # task_score=True records answer-level accuracy alongside block efficiency.
        # -------------------------------------------------------------------
        {
            "id": "eval_kl_gsm8k",
            "group": "Phase 3 — GSM8K Eval",
            "desc": "Eval kl-gsm8k on gsm8k",
            "cmd": _ec(_merged("kl-gsm8k"), "kl", datasets="gsm8k", task_score=True),
            "done_check": None,
        },
        {
            "id": "eval_ebe_gsm8k",
            "group": "Phase 3 — GSM8K Eval",
            "desc": "Eval ebe-gsm8k on gsm8k",
            "cmd": _ec(_merged("ebe-gsm8k"), "ebe", datasets="gsm8k", task_score=True),
            "done_check": None,
        },
        {
            "id": "eval_rev_kl_gsm8k",
            "group": "Phase 3 — GSM8K Eval",
            "desc": "Eval rev_kl-gsm8k on gsm8k",
            "cmd": _ec(_merged("rev_kl-gsm8k"), "rev_kl", datasets="gsm8k", task_score=True),
            "done_check": None,
        },
        {
            "id": "eval_jsd_gsm8k",
            "group": "Phase 3 — GSM8K Eval",
            "desc": "Eval jsd-gsm8k on gsm8k",
            "cmd": _ec(_merged("jsd-gsm8k"), "jsd", datasets="gsm8k", task_score=True),
            "done_check": None,
        },
        {
            "id": "eval_l1_gsm8k",
            "group": "Phase 3 — GSM8K Eval",
            "desc": "Eval l1-gsm8k on gsm8k",
            "cmd": _ec(_merged("l1-gsm8k"), "l1", datasets="gsm8k", task_score=True),
            "done_check": None,
        },
        {
            "id": "eval_online_gsm8k",
            "group": "Phase 3 — GSM8K Eval",
            "desc": "Eval online-gsm8k on gsm8k",
            "cmd": _ec(_merged("online-gsm8k"), "online", datasets="gsm8k", task_score=True),
            "done_check": None,
        },

        # -------------------------------------------------------------------
        # Phase 4 — Multi-Dataset Eval
        # Always skipped in smoke (too slow; smoke already verified the eval path
        # in Phase 3).  In full mode, evaluates every trained model on four
        # additional datasets for the paper's cross-dataset table.
        # -------------------------------------------------------------------
        {
            "id": "eval_baseline_all",
            "group": "Phase 4 — Multi-Dataset",
            "desc": "Eval baseline on humaneval,math500,mtbench,alpaca",
            "cmd": _ec(draft, "baseline",
                       datasets="humaneval,math500,mtbench,alpaca", task_score=True),
            "done_check": None,
            "smoke_skip": smoke,
        },
        {
            "id": "eval_kl_all",
            "group": "Phase 4 — Multi-Dataset",
            "desc": "Eval kl on humaneval,math500,mtbench,alpaca",
            "cmd": _ec(_merged("kl-gsm8k"), "kl",
                       datasets="humaneval,math500,mtbench,alpaca", task_score=True),
            "done_check": None,
            "smoke_skip": smoke,
        },
        {
            "id": "eval_ebe_all",
            "group": "Phase 4 — Multi-Dataset",
            "desc": "Eval ebe on humaneval,math500,mtbench,alpaca",
            "cmd": _ec(_merged("ebe-gsm8k"), "ebe",
                       datasets="humaneval,math500,mtbench,alpaca", task_score=True),
            "done_check": None,
            "smoke_skip": smoke,
        },
        {
            "id": "eval_rev_kl_all",
            "group": "Phase 4 — Multi-Dataset",
            "desc": "Eval rev_kl on humaneval,math500,mtbench,alpaca",
            "cmd": _ec(_merged("rev_kl-gsm8k"), "rev_kl",
                       datasets="humaneval,math500,mtbench,alpaca", task_score=True),
            "done_check": None,
            "smoke_skip": smoke,
        },
        {
            "id": "eval_jsd_all",
            "group": "Phase 4 — Multi-Dataset",
            "desc": "Eval jsd on humaneval,math500,mtbench,alpaca",
            "cmd": _ec(_merged("jsd-gsm8k"), "jsd",
                       datasets="humaneval,math500,mtbench,alpaca", task_score=True),
            "done_check": None,
            "smoke_skip": smoke,
        },
        {
            "id": "eval_l1_all",
            "group": "Phase 4 — Multi-Dataset",
            "desc": "Eval l1 on humaneval,math500,mtbench,alpaca",
            "cmd": _ec(_merged("l1-gsm8k"), "l1",
                       datasets="humaneval,math500,mtbench,alpaca", task_score=True),
            "done_check": None,
            "smoke_skip": smoke,
        },
        {
            "id": "eval_online_all",
            "group": "Phase 4 — Multi-Dataset",
            "desc": "Eval online on humaneval,math500,mtbench,alpaca",
            "cmd": _ec(_merged("online-gsm8k"), "online",
                       datasets="humaneval,math500,mtbench,alpaca", task_score=True),
            "done_check": None,
            "smoke_skip": smoke,
        },

        # -------------------------------------------------------------------
        # Phase 5 (optional): EAGLE Benchmark — only included when --eagle
        #
        # EAGLE trains its own 1-layer head on the TARGET model's hidden states,
        # then measures block efficiency under the same speculative-decoding setup.
        # This gives a strong external baseline: if SpecDist-EBE can beat an EAGLE
        # head trained on the same hardware, it is a meaningful result.
        #
        # MUST be rerun for each new target model or compute environment:
        #   • New target (e.g. laptop→Colab/server: 0.6B→8B) → new head needed
        #   • Kaggle/Colab sessions are ephemeral — checkpoints/data/eagle_tmp lost
        #
        # Three phases:
        #   eagle_gen   — run target on 500 gsm8k prompts; save (h, ids) tensors
        #   eagle_train — train 1-layer EAGLE head for 2000 steps
        #   eagle_eval  — measure block efficiency; results saved to results.db
        # -------------------------------------------------------------------
        *([
            {
                "id": "eagle_gen",
                "group": "Phase 5 — EAGLE Benchmark",
                "desc": "EAGLE phase 1: generate hidden-state training data (500 prompts)",
                "cmd": [
                    sys.executable, os.path.join(HERE, "eagle_bench.py"), "gen",
                    "--base", target,
                    "--data", _data("gsm8k_train.jsonl"),
                    "--out",  _data("eagle_tmp"),
                    "--n",    "500",
                ],
                # done_check: first tensor file written by phase gen
                "done_check": os.path.join(_GBV_RESEARCH, "core", "datasets", "raw", "eagle_tmp", "00000.pt"),
            },
            {
                "id": "eagle_train",
                "group": "Phase 5 — EAGLE Benchmark",
                "desc": "EAGLE phase 2: train 1-layer head on target hidden states (2000 steps)",
                "cmd": [
                    sys.executable, os.path.join(HERE, "eagle_bench.py"), "train",
                    "--base",  target,
                    "--tmp",   _data("eagle_tmp"),
                    "--ckpt",  _ckpt("eagle-head"),
                    "--steps", "2000",
                ],
                "done_check": os.path.join(_ckpt("eagle-head"), "latest.pt"),
            },
            {
                "id": "eagle_eval",
                "group": "Phase 5 — EAGLE Benchmark",
                "desc": "EAGLE phase 3: evaluate block efficiency vs SpecDist baseline",
                "cmd": [
                    sys.executable, os.path.join(HERE, "eagle_bench.py"), "eval",
                    "--base", target,
                    "--ckpt", _ckpt("eagle-head"),
                    "--data", _data("gsm8k_30.jsonl"),
                    "--K",    "3",
                ],
                "done_check": None,  # writes to results.db; no single output file
            },
        ] if eagle else []),
    ]

# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

def load_state():
    if os.path.exists(STATE_FILE):
        # utf-8-sig strips the UTF-8 BOM that PowerShell's Set-Content -Encoding utf8
        # writes by default.  Without this, json.load() sees ﻿ before the opening
        # brace and raises "Expecting value: line 1 column 1 (char 0)".
        with open(STATE_FILE, encoding="utf-8-sig") as f:
            return json.load(f)
    return {"version": 1, "steps": {}}


def save_state(state):
    # Atomic write: .tmp → os.replace() prevents a corrupt state file if the
    # process is killed mid-write (Colab session death, Modal timeout, OOM kill).
    _tmp = STATE_FILE + ".tmp"
    with open(_tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    os.replace(_tmp, STATE_FILE)


def _osd_equivalent(path: str) -> str | None:
    """
    Return the OSD/checkpoints/ equivalent of a gbv-research/db/checkpoints/ path.
    Used to detect pre-existing artifacts from the original OSD/ pipeline run.
    E.g.:  gbv-research/db/checkpoints/kl200_merged/config.json
        -> OSD/checkpoints/kl200_merged/config.json
    Returns None if the path isn't under gbv-research/db/checkpoints/.
    """
    ckpt_marker = os.path.join("db", "checkpoints")
    idx = path.replace("\\", "/").find("db/checkpoints/")
    if idx < 0:
        return None
    rel = path.replace("\\", "/")[idx + len("db/checkpoints/"):]
    return os.path.join(_OSD_DIR, "checkpoints", rel.replace("/", os.sep))


def step_status(step, state):
    """
    Returns 'done' / 'pending' / 'failed' / 'running'.
    done_check file takes priority: if the output artifact exists, always 'done'.
    Also checks OSD/checkpoints/ as a fallback for legacy artifacts.
    """
    dc = step.get("done_check")
    if dc and os.path.exists(dc):
        return "done"
    # Check OSD legacy checkpoints as fallback
    if dc:
        osd_dc = _osd_equivalent(dc)
        if osd_dc and os.path.exists(osd_dc):
            return "done"
    recorded = state["steps"].get(step["id"], {}).get("status", "pending")
    return recorded


def mark_step(state, step_id, status, note=""):
    state["steps"][step_id] = {
        "status": status,
        "ts": datetime.utcnow().isoformat() + "Z",
        "note": note,
    }
    save_state(state)


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

TICK = "[OK]"
CROSS = "[FAIL]"
ARROW = ">>"
CLOCK = "[..]"

STATUS_ICON = {"done": TICK, "pending": " ", "failed": CROSS, "running": CLOCK}
STATUS_COLOR = {
    "done": "\033[32m", "pending": "\033[90m",
    "failed": "\033[31m", "running": "\033[33m",
}
RESET = "\033[0m"


def print_plan(steps, state, highlight_from=None):
    print()
    current_group = None
    for i, step in enumerate(steps):
        if step["group"] != current_group:
            current_group = step["group"]
            print(f"\n  {current_group}")
            print("  " + "-" * 50)
        status = step_status(step, state)
        icon = STATUS_ICON.get(status, " ")
        col = STATUS_COLOR.get(status, "")
        marker = f"{ARROW}" if highlight_from == step["id"] else " "
        print(f"  {marker} {col}{icon}{RESET} [{i+1:02d}] {step['desc']}")
    print()


def _yn(prompt):
    while True:
        r = input(prompt + " [y/n]: ").strip().lower()
        if r in ("y", "yes"):
            return True
        if r in ("n", "no"):
            return False


# ---------------------------------------------------------------------------
# Smoke preflight
# ---------------------------------------------------------------------------

_GBV_DIR = _GBV_SRC  # defined above: sibling GBV/ repo


def _cleanup_tmp(path):
    try:
        os.remove(path)
    except OSError:
        pass


def run_smoke_preflight(draft, target):
    """
    Run 2 gsm8k prompts through GBV/main.py directly — NO run_all.py, NO DB writes.

    Why bypass run_all.py?
      run_all.py's --skip_existing checks (student_label, dataset, mode, K, T)
      without experiment_tag.  If the preflight wrote "baseline/gsm8k/gbv/3/0.6"
      rows, the real eval would silently skip those combos.  By calling GBV/main.py
      directly we test model loading + speculative decoding without touching the DB.

    What it validates:
      • Both models load in bf16 without OOM
      • iid_draft() + target_tree_pass() run without assertion / shape errors
      • GBV verify() completes and produces a finite block-efficiency number
      • Everything is wired correctly end-to-end

    Returns True on success (rc=0), False on failure.
    Skips gracefully (returns True) if no gsm8k data file is present yet.
    """
    global _child_popen

    # ── Find the gsm8k data file ──────────────────────────────────────────
    _raw = os.path.join(_GBV_RESEARCH, "core", "datasets", "raw")
    data_candidates = [
        os.path.join(_raw, "gsm8k_30.jsonl"),
        os.path.join(_raw, "gsm8k_10.jsonl"),
        os.path.join(_raw, "gsm8k_50.jsonl"),
        os.path.join(_raw, "gsm8k_5.jsonl"),
    ]
    data_file = next((p for p in data_candidates if os.path.exists(p)), None)
    if not data_file:
        print(f"\n  [PREFLIGHT] No gsm8k data file found in data/ — skipping preflight.")
        print(f"  The first pipeline step will fetch/create it automatically.")
        print(f"  To always skip: python pipeline.py --no_smoke_first\n")
        return True   # non-fatal

    # ── Write a 2-prompt temp file ─────────────────────────────────────────
    with open(data_file, encoding="utf-8") as fh:
        two_prompts = [ln for ln in fh if ln.strip()][:2]
    tmp_path = os.path.join(_raw, "_smoke_preflight_2.jsonl")
    with open(tmp_path, "w", encoding="utf-8") as fh:
        fh.writelines(two_prompts)

    smoke_cmd = [
        sys.executable, "-u",
        os.path.join(_GBV_DIR, "main.py"),
        "--p_model",        target,
        "--q_model",        draft,
        "--data",           tmp_path,
        "--modes",          "gbv",
        "--Ks",             "3",
        "--p_temps",        "0.6",
        "--max_new_tokens", "30",
        "--dtype",          "bf16",
    ]

    print(f"\n{'='*65}")
    print(f"  PREFLIGHT SMOKE CHECK  (laptop auto-check before main pipeline)")
    print(f"  n=2 prompts · mode=gbv · K=3 · T=0.6 · max_tokens=30")
    print(f"  Runs GBV/main.py directly — zero DB writes, no skip_existing risk.")
    print(f"  Expected:  5-20 min on first run (CUDA kernel warm-up),")
    print(f"             1-3 min on subsequent runs (kernels already compiled).")
    print(f"  To skip:   python pipeline.py --no_smoke_first")
    print(f"{'='*65}")

    env = {**os.environ,
           "TRANSFORMERS_OFFLINE": "1",
           "HF_HUB_OFFLINE":       "1",
           "HF_DATASETS_OFFLINE":  "1",
           "PYTHONIOENCODING":     "utf-8",
           "PYTHONUNBUFFERED":     "1"}

    t0 = time.time()
    PARENT = os.path.dirname(HERE)
    proc = subprocess.Popen(smoke_cmd, cwd=PARENT, env=env)
    _child_popen = proc
    _write_lock(child_pid=proc.pid)

    try:
        rc = proc.wait(timeout=2400)          # 40-min safety cap
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        _child_popen = None
        _write_lock()
        _cleanup_tmp(tmp_path)
        print(f"\n  {CROSS} PREFLIGHT TIMED OUT (>40 min)")
        print(f"  Possible causes: OOM, frozen CUDA, very slow GPU.")
        print(f"  Re-run with:  python pipeline.py --no_smoke_first  to skip preflight.")
        return False
    except KeyboardInterrupt:
        if proc.poll() is None:
            _kill_tree(proc.pid)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        _child_popen = None
        _write_lock()
        _cleanup_tmp(tmp_path)
        raise

    _child_popen = None
    _write_lock()
    _cleanup_tmp(tmp_path)

    elapsed = time.time() - t0
    if rc == 0:
        print(f"\n  {TICK} Preflight passed in {elapsed/60:.1f} min"
              f"  — starting main pipeline")
        return True
    else:
        print(f"\n  {CROSS} Preflight FAILED (rc={rc}, {elapsed:.0f}s elapsed)")
        print(f"  Check the traceback above.  Common causes:")
        print(f"    OOM        → free GPU VRAM, or run on Colab/server")
        print(f"    Assertion  → GBV/util.py attention_type check failed (wrong model)")
        print(f"    Offline    → model not cached; run setup_download.py first")
        print(f"  Fix then re-run.  To skip preflight: --no_smoke_first")
        return False


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_step(step, state, dry_run=False):
    """
    Run one pipeline step as a child subprocess.

    Uses subprocess.Popen (not subprocess.run) so we hold the child handle
    and can:
      • Track its PID in the lock file → killed automatically on next startup
        if this process is orphaned (the root cause of the zombie problem).
      • Terminate it cleanly on KeyboardInterrupt (Ctrl-C) instead of leaving
        it running and holding GPU VRAM.
    """
    global _child_popen
    sid = step["id"]
    print(f"\n{'='*65}")
    print(f"  STEP: {step['desc']}")
    print(f"  CMD : {' '.join(step['cmd'][:6])}{'...' if len(step['cmd'])>6 else ''}")
    print(f"{'='*65}")

    if dry_run:
        print("  [dry_run] skipping execution")
        return True

    mark_step(state, sid, "running")
    t0 = time.time()

    env = os.environ.copy()
    # Force HF offline mode (models must already be cached)
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["HF_HUB_OFFLINE"]       = "1"
    env["HF_DATASETS_OFFLINE"]  = "1"
    # Force UTF-8 I/O in every child Python process — prevents UnicodeEncodeError
    # when run_all.py prints box-drawing characters on Windows cp1252 terminals.
    # Also needed for correct text handling on Kaggle (UTF-8 by default but explicit
    # is safer) and Colab (same).
    env["PYTHONIOENCODING"]  = "utf-8"
    env["PYTHONUNBUFFERED"]  = "1"   # flush every print() immediately — no more silent 5-min gaps

    # Open the pipeline log in append mode — one file for the whole pipeline run,
    # readable live from the dashboard Logs panel (http://127.0.0.1:5000/ → Logs button).
    _log_fh = open(_PIPELINE_LOG, "a", encoding="utf-8", errors="replace", buffering=1)
    _log_fh.write(
        f"\n{'='*65}\n"
        f"  STEP : {step['desc']}\n"
        f"  TIME : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"  CMD  : {' '.join(step['cmd'])}\n"
        f"{'='*65}\n"
    )
    _log_fh.flush()

    try:
        # Popen with PIPE so we can tee output to both the terminal and pipeline_output.log.
        # The dashboard Logs panel reads pipeline_output.log live via /api/log_tail.
        proc = subprocess.Popen(
            step["cmd"], cwd=HERE, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            encoding="utf-8", errors="replace",
        )
        _child_popen = proc
        _write_lock(child_pid=proc.pid)   # record child PID → killed on next startup if orphaned

        # Stream line-by-line → terminal AND log file simultaneously (tee)
        for _line in iter(proc.stdout.readline, ""):
            sys.stdout.write(_line)
            sys.stdout.flush()
            _log_fh.write(_line)
            _log_fh.flush()

        rc = proc.wait()
        _child_popen = None
        _write_lock()                     # remove child PID from lock (step finished)

        elapsed = time.time() - t0
        _done_msg = (f"\n  {TICK} Done in {elapsed/60:.1f} min\n"
                     if rc == 0
                     else f"\n  {CROSS} Failed (exit code {rc})\n")
        sys.stdout.write(_done_msg); sys.stdout.flush()
        _log_fh.write(_done_msg);   _log_fh.flush()

        if rc == 0:
            mark_step(state, sid, "done", f"elapsed={elapsed:.0f}s")
            return True
        else:
            mark_step(state, sid, "failed", f"rc={rc}")
            _hints = (
                "  Hint: check output above for [OOM] / [ERROR] / Traceback.\n"
                "  Common fixes:\n"
                "    OOM       -> run_all.py retries on CPU automatically;\n"
                "                 train_qwen3.py: add --no_lora or reduce --max_new_tokens\n"
                "    Offline   -> run setup_download.py first, or unset TRANSFORMERS_OFFLINE\n"
                "    Stale run -> python pipeline.py --status\n"
                "  Restart   -> python pipeline.py --config laptop --yes\n"
                "               (resets 'failed' training steps to 'pending' automatically)\n"
            )
            sys.stdout.write(_hints); sys.stdout.flush()
            _log_fh.write(_hints);   _log_fh.flush()
            # Write a self-contained per-step error snapshot for remote debugging.
            # On Colab/Modal you can download just this one file to see what failed.
            os.makedirs(_DB_LOGS, exist_ok=True)
            _err_path = os.path.join(_DB_LOGS, f"step_{sid}_error.log")
            try:
                with open(_PIPELINE_LOG, encoding="utf-8", errors="replace") as _plog:
                    _all_lines = _plog.readlines()
                with open(_err_path, "w", encoding="utf-8") as _ef:
                    _ef.write(f"STEP FAILURE SNAPSHOT — {sid}\n")
                    _ef.write(f"time={datetime.now().isoformat()}  rc={rc}\n")
                    _ef.write(f"cmd={' '.join(step['cmd'])}\n")
                    _ef.write("=" * 72 + "\n")
                    _ef.writelines(_all_lines[-120:])   # last 120 lines of pipeline log
                print(f"  [debug] Error snapshot → {_err_path}")
            except Exception:
                pass
            return False

    except KeyboardInterrupt:
        # Kill the child so it doesn't keep holding GPU VRAM as an orphan
        if _child_popen and _child_popen.poll() is None:
            _msg = f"\n  [Ctrl-C] Terminating child process PID {_child_popen.pid}...\n"
            sys.stdout.write(_msg); sys.stdout.flush()
            _log_fh.write(_msg);   _log_fh.flush()
            _kill_tree(_child_popen.pid)
            try:
                _child_popen.wait(timeout=10)
            except subprocess.TimeoutExpired:
                _child_popen.kill()
        _child_popen = None
        _write_lock()
        mark_step(state, sid, "failed", "interrupted by user")
        _int_msg = f"\n  [interrupted] Step {sid} marked as failed. Re-run to resume.\n"
        sys.stdout.write(_int_msg); sys.stdout.flush()
        _log_fh.write(_int_msg);   _log_fh.flush()
        raise

    finally:
        _log_fh.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _print_header(cfg, draft, target, args):
    """Print the startup header (config, GPU, smoke/eagle flags)."""
    print(f"  Config : {cfg['desc']}")
    print(f"  Draft  : {draft}")
    print(f"  Target : {target}")
    print(f"  State  : {os.path.basename(STATE_FILE)}")
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(0)
            print(f"  GPU    : {torch.cuda.get_device_name(0)}  "
                  f"{free/1024**3:.1f} GB free / {total/1024**3:.1f} GB total")
            if free / 1024**3 < 2.0:
                print("  [WARN] Less than 2 GB VRAM free. "
                      "run_all.py will fall back to CPU automatically on OOM.")
        else:
            print("  GPU    : None — all steps will run on CPU (slow but correct)")
    except Exception:
        print("  GPU    : (torch not available yet — checked per step)")
    if args.experiment_tag:
        print(f"  Tag    : {args.experiment_tag}")
    if args.smoke:
        print(f"  Mode   : SMOKE TEST  n=5 · max_tokens=30 · K=3 · modes=all-6-verifiers · temp=0.6")
        print(f"           (6 losses × 50 steps + 6 evals — exercises every code path; ~30-40 min)")
    if args.eagle:
        if args.config == "laptop":
            print(f"\n  [WARN] --eagle with --config laptop makes no sense for the paper.")
            print(f"         The EAGLE head trains on the TARGET model's hidden states.")
            print(f"         A head built on Qwen3-0.6B hidden states is NOT a useful")
            print(f"         comparison baseline.  Run with --config server or colab")
            print(f"         (Qwen3-8B target) on Colab/Kaggle/Modal instead.\n")
        print(f"  Eagle  : Phase 5 included — EAGLE head train+eval on {target}")
        print(f"           (rerun per Colab/Kaggle session — ephemeral disk loses checkpoints)")


def main():
    p = argparse.ArgumentParser(description="SpecDist pipeline with crash-safe resume")
    p.add_argument("--config", default="laptop", choices=list(CONFIGS.keys()),
                   help="Hardware config: laptop (0.5B->0.6B) or server/colab (0.6B->8B)")
    p.add_argument("--draft",  default=None,
                   help="Override draft model HF ID (overrides --config)")
    p.add_argument("--target", default=None,
                   help="Override target model HF ID (overrides --config)")
    p.add_argument("--yes", "-y", action="store_true",
                   help="Auto-resume without interactive prompts")
    p.add_argument("--restart", action="store_true",
                   help="Reset state and restart from step 1")
    p.add_argument("--from", dest="from_step", default=None,
                   help="Resume from a specific step ID (skips all prior steps)")
    p.add_argument("--dry_run", action="store_true",
                   help="Print plan without running anything")
    p.add_argument("--status", action="store_true",
                   help="Print current status and exit")
    p.add_argument("--experiment_tag", default=None,
                   help="Free-text label attached to every eval run in the DB for this "
                        "pipeline invocation, e.g. 'v2 EBE clipped accept weight'. "
                        "Visible in the viz dashboard as a filter. Each run still has "
                        "its own unique run_tag timestamp.")
    p.add_argument("--smoke", action="store_true",
                   help="Quick sanity-check mode: n=5 prompts, max_tokens=30, K=3, "
                        "modes=alpha+bv+gbv+traversal+specinfer+naive, temp=0.6.  "
                        "Exercises every loss + every verifier.  ~30-40 min total on laptop.")
    p.add_argument("--no_smoke_first", action="store_true",
                   help="Skip the automatic 2-prompt preflight smoke check that normally "
                        "runs before the first eval step on laptop config.  Use this if "
                        "you have already verified the pipeline is working and want to "
                        "skip the 5-20 min warm-up cost.")
    p.add_argument("--eagle", action="store_true",
                   help="Append Phase 5 — EAGLE Benchmark.  Only meaningful with "
                        "--config server or colab (Qwen3-8B target).  Do NOT use "
                        "with --config laptop: a head trained on 0.6B hidden states "
                        "is not a useful paper baseline.  Must be rerun for each new "
                        "target model or ephemeral compute session (Kaggle/Colab lose "
                        "checkpoints on session restart).  Adds ~3-6 hours on A100.")
    args = p.parse_args()

    cfg = CONFIGS[args.config]
    draft  = args.draft  or cfg["draft"]
    target = args.target or cfg["target"]

    # State file is config-scoped so laptop and server runs don't mix.
    # Smoke gets its OWN state file so smoke's "done" marks never prevent the
    # real pipeline from running Phase 2 training and Phase 3/4 evals.
    global STATE_FILE
    _smoke_tag = "_smoke" if args.smoke else ""
    STATE_FILE = os.path.join(HERE, f"pipeline_state_{args.config}{_smoke_tag}.json")

    # --status and --dry_run are read-only: build steps + print plan, then exit.
    # Do NOT acquire the lock (that kills any running pipeline process!).
    _load_4bit = cfg.get("load_in_4bit", False)
    if args.status or args.dry_run:
        STEPS = build_steps(draft, target, experiment_tag=args.experiment_tag,
                            smoke=args.smoke, eagle=args.eagle,
                            load_in_4bit=_load_4bit)
        _print_header(cfg, draft, target, args)
        state = load_state()
        for step in STEPS:                          # sync done_check files
            if step.get("done_check") and os.path.exists(step["done_check"]):
                if state["steps"].get(step["id"], {}).get("status") != "done":
                    mark_step(state, step["id"], "done", "auto-detected from done_check")
        print_plan(STEPS, state)
        if args.status:
            done    = sum(1 for s in STEPS if step_status(s, state) == "done")
            pending = sum(1 for s in STEPS if step_status(s, state) == "pending")
            failed  = sum(1 for s in STEPS if step_status(s, state) == "failed")
            print(f"  Summary: {done} done, {pending} pending, {failed} failed / {len(STEPS)} total\n")
        return

    # Load per-user WandB config (sets WANDB_API_KEY / WANDB_ENTITY / WANDB_PROJECT
    # env vars so all subprocesses inherit them; falls back to wandb login if absent).
    _load_wandb_config()

    # Acquire lock: kills any previously orphaned pipeline + child processes
    # before we load models, preventing GPU VRAM conflicts and duplicate runs.
    os.makedirs(_DB_LOGS, exist_ok=True)   # ensure db/logs/ exists before first log write
    _acquire_lock()

    STEPS = build_steps(draft, target, experiment_tag=args.experiment_tag,
                        smoke=args.smoke, eagle=args.eagle,
                        load_in_4bit=_load_4bit)

    _print_header(cfg, draft, target, args)

    state = load_state()

    # Sync done_check files → state
    for step in STEPS:
        if step.get("done_check") and os.path.exists(step["done_check"]):
            if state["steps"].get(step["id"], {}).get("status") != "done":
                mark_step(state, step["id"], "done", "auto-detected from done_check")

    # Auto-recover steps stuck in "running" — happens when the pipeline process
    # was killed (OOM, Ctrl-C, power cycle) before the step finished.
    # Eval steps always pass --skip_existing, so re-running them never inserts
    # duplicate DB rows.  Training steps have done_check files and are caught above.
    _recovered = []
    for step in STEPS:
        if state["steps"].get(step["id"], {}).get("status") == "running":
            mark_step(state, step["id"], "pending", "auto-recovered from interrupted run")
            _recovered.append(step["id"])
    if _recovered:
        print(f"\n  [RECOVER] {len(_recovered)} step(s) were left in 'running' state "
              f"(pipeline was interrupted). Reset to 'pending': "
              + ", ".join(_recovered))
        print(f"  Eval steps pass --skip_existing, so no duplicate DB entries "
              f"will be created on re-run.\n")

    if args.status or args.dry_run:
        print_plan(STEPS, state)
        if args.status:
            done = sum(1 for s in STEPS if step_status(s, state) == "done")
            pending = sum(1 for s in STEPS if step_status(s, state) == "pending")
            failed = sum(1 for s in STEPS if step_status(s, state) == "failed")
            print(f"  Summary: {done} done, {pending} pending, {failed} failed / {len(STEPS)} total\n")
            return

    if args.restart:
        if not args.yes and not _yn("Reset ALL state and restart from step 1?"):
            print("Aborted.")
            return
        state = {"version": 1, "steps": {}}
        save_state(state)
        print("State reset.")

    # Find start step
    start_id = args.from_step
    if start_id:
        ids = [s["id"] for s in STEPS]
        if start_id not in ids:
            print(f"Unknown step id '{start_id}'. Valid ids:\n  " + "\n  ".join(ids))
            sys.exit(1)

    # Check for any previously failed or interrupted steps
    failed_steps = [s for s in STEPS if step_status(s, state) == "failed"]
    pending_steps = [s for s in STEPS if step_status(s, state) == "pending"]

    if not args.yes and not args.restart and not start_id:
        print_plan(STEPS, state)
        done_count = sum(1 for s in STEPS if step_status(s, state) == "done")
        print(f"  {done_count}/{len(STEPS)} steps already done.")

        if failed_steps:
            print(f"  {CROSS} {len(failed_steps)} step(s) previously failed: "
                  + ", ".join(s["id"] for s in failed_steps))

        if done_count == len(STEPS):
            print("  All steps complete! Nothing to do.")
            return

        if not _yn(f"Resume pipeline ({len(pending_steps) + len(failed_steps)} steps to run)?"):
            print("Aborted. Re-run when ready.")
            return

    # ── Preflight smoke check (laptop config only) ────────────────────────
    # Runs 2 prompts through GBV/main.py directly before any pipeline step.
    # Purpose: catch model-load failures, OOM, assertion errors, and import
    # problems early — before the user waits hours for the real eval to start.
    # Bypasses run_all.py so zero DB rows are written → no skip_existing risk.
    # Skipped when: --smoke (user is already in lightweight mode), --no_smoke_first
    # (explicit opt-out), non-laptop config, or all eval steps already done.
    _has_pending_eval = any(
        s["id"].startswith("eval") and step_status(s, state) != "done"
        for s in STEPS
    )
    if (args.config == "laptop"
            and not args.no_smoke_first
            and not args.smoke
            and _has_pending_eval):
        if not run_smoke_preflight(draft, target):
            sys.exit(1)

    # Execute steps
    skip_until = start_id
    n_run = 0
    for step in STEPS:
        sid = step["id"]

        # --from: skip steps before the target
        if skip_until:
            if sid == skip_until:
                skip_until = None
            else:
                continue

        # In smoke mode, Phase 2/3/4 steps carry smoke_skip=True.
        # Print once per group, then skip without touching state — the real
        # pipeline (separate state file) will run them normally.
        if step.get("smoke_skip"):
            print(f"  [smoke] {sid} — skipped "
                  f"(training / post-train eval; run without --smoke for full pipeline)")
            continue

        status = step_status(step, state)
        if status == "done":
            print(f"  {TICK} [{sid}] already done — skipping")
            continue

        # Auto-skip steps whose input checkpoint doesn't exist (pre-pipeline artifacts).
        # This keeps clean-from-scratch runs non-blocking without manual state edits.
        skip_path = step.get("skip_if_missing")
        if skip_path and not os.path.exists(skip_path):
            print(f"  [skip] [{sid}] prerequisite missing ({os.path.basename(skip_path)}) — auto-skipped")
            mark_step(state, sid, "done", f"auto-skipped: {os.path.basename(skip_path)} not found")
            continue

        # Prompt before each step (unless --yes)
        if not args.yes and n_run > 0:
            if not _yn(f"\nContinue to: {step['desc']}?"):
                print("Paused. Re-run to continue from this step.")
                return

        # Training steps get up to 2 automatic retries — on Colab/Modal a transient
        # PermissionError, NFS hiccup, or BOM-corruption can cause a spurious rc=1
        # that resolves on the next attempt (checkpoint resumes cleanly).
        # Eval steps are already idempotent via --skip_existing, so no retry needed.
        _is_train_step = step.get("retryable", False)
        _max_attempts  = 3 if _is_train_step else 1
        success = False
        for _attempt in range(_max_attempts):
            if _attempt > 0:
                print(f"\n  [retry] Attempt {_attempt+1}/{_max_attempts} for '{sid}' "
                      f"(training resumes from ckpt_latest automatically)...")
                # Reset status so run_step() re-marks it running
                mark_step(state, sid, "pending", f"auto-retry attempt {_attempt+1}")
            success = run_step(step, state, dry_run=args.dry_run)
            if success:
                break
        n_run += 1

        if not success:
            print(f"\n{CROSS} Step '{sid}' failed. Fix the issue then re-run pipeline.py")
            print(f"   The pipeline will skip completed steps and retry from '{sid}'.")
            print(f"   Restart: python pipeline.py --config {args.config} --yes")
            sys.exit(1)

    print(f"\n{'='*65}")
    print(f"  {TICK} Pipeline complete! {n_run} step(s) executed.")
    print(f"  View results: python viz_server.py")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    _reconfigure_stdout_for_windows()
    main()
