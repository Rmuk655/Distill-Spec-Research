"""
experiment.py — master run script with crash-safe resume.

Runs the full SpecDist experiment pipeline in order:
  1. Merge pending LoRA checkpoints
  2. Train KL and EBE on real GSM8K data (overnight steps)
  3. Evaluate all models on all datasets with full hyperparameter sweep

State is persisted to pipeline_state.json after every step.
On restart, the script shows what's done and asks before resuming.
Eval steps always pass --skip_existing to evaluate.py, so partial evals
are safe to re-run — they pick up from the last completed cell.

Usage:
    python experiment.py                        # laptop config, interactive
    python experiment.py --yes                  # auto-resume without prompts
    python experiment.py --config server --yes  # Qwen3-0.6B -> Qwen3-8B
    python experiment.py --restart              # force-restart from step 1
    python experiment.py --from STEP_ID         # resume from a specific step
    python experiment.py --dry_run              # print plan without running
    python experiment.py --status               # print current status and exit
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
# Reconfigure to UTF-8 with replacement so experiment.py never dies on a stray character.
# Guard with __name__ == "__main__" so importing experiment.py in tests doesn't break
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
#   The user ran experiment.py 3 times in the same terminal session.  Each time
#   the old experiment.py process had been killed (OOM, Ctrl+C, power cycle) but
#   its child subprocess (evaluate.py) was left running as an ORPHAN — it kept
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
#      so Unicode characters in evaluate.py output never crash on Windows cp1252.
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
        # /T = kill entire process tree (children too — catches orphaned evaluate.py)
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
    (the previous experiment.py AND its evaluate.py child) before proceeding.
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
# YAML config loader — reads orchestration/configs/<name>.yaml for training
# hyperparameters that the hardcoded CONFIGS dict doesn't carry.
# Returns a flat dict; missing keys fall back to safe defaults.
# ---------------------------------------------------------------------------

def _load_config_yaml(config_name: str) -> dict:
    """Load orchestration/configs/{config_name}.yaml and return a flat hyperparams dict.

    YAML structure (nested) is flattened to a single-level dict so callers can
    do: h.get("lr", 3e-5) without knowing the nesting.

    Missing YAML file or missing keys → return defaults silently (never crash).
    """
    yaml_path = os.path.join(HERE, "configs", f"{config_name}.yaml")
    if not os.path.exists(yaml_path):
        return {}
    try:
        import yaml
        with open(yaml_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        training      = data.get("training", {})
        health        = data.get("health", {})
        checkpointing = data.get("checkpointing", {})
        hardware      = data.get("hardware", {})
        logging_cfg   = data.get("logging", {})
        models_cfg    = data.get("models", {})
        experiment_cfg = data.get("experiment", {})
        out = {
            "lr":                   training.get("lr", 3e-5),
            "lora_r":               training.get("lora_r", 8),
            "lora_alpha":           training.get("lora_alpha", 16),
            "teacher_temp":         training.get("teacher_temperature", 0.8),
            "max_new_tokens":       training.get("max_new_tokens", 80),
            "seed":                 training.get("seed", 42),
            "nan_action":           health.get("nan_action", "stop"),
            "early_stop_patience":  health.get("early_stop_patience", 0),
            # Checkpoint cadence — controls how much work is lost on crash/timeout.
            # These were previously read but never forwarded to trainer.py; the
            # trainer silently fell back to its own defaults (100/200/5).
            "save_every":           checkpointing.get("save_every", 100),
            "milestone_every":      checkpointing.get("milestone_every", 200),
            "max_checkpoints":      checkpointing.get("max_checkpoints", 5),
            # W&B wiring: project + group were parsed from YAML but never forwarded
            # to trainer.py.  Now passed via --wandb_project / --wandb_group so every
            # run lands in the right W&B project and group.
            "wandb_project":        logging_cfg.get("wandb_project", "distillspec"),
            "wandb_group":          logging_cfg.get("wandb_group", config_name),
            # run_label: short tag embedded in every W&B run name so runs from
            # different configs are distinguishable at a glance:
            #   "1.7B-T4-lite-kl_qwen_300steps"  vs  "4B-T4-kl_qwen_500steps"  etc.
            # Falls back to config_name if not set in YAML.
            "run_label":            logging_cfg.get("run_label", config_name),
        }
        # training.steps → train_steps (server=5000, colab=500, laptop=1000).
        # No default — None lets build_steps() apply the smoke/full default.
        if "steps" in training:
            out["train_steps"] = training["steps"]
        # checkpointing.storage_root → storage_root so Colab/Modal configs can
        # set persistent storage without a CLI flag every run.
        if checkpointing.get("storage_root"):
            out["storage_root"] = checkpointing["storage_root"]
        # hardware.compile → compile flag passed to trainer.py and online_serve.py.
        # server.yaml: compile: true (Linux/A100 — torch.compile gives 10-30% speedup).
        # colab.yaml:  compile: false (torch.compile unreliable in Colab environment).
        # laptop.yaml: not set   (Windows — scripts skip compile automatically).
        if "compile" in hardware:
            out["compile"] = hardware["compile"]
        # training.online_lr → LR for forward_kl online adapt (typically 10x offline LR).
        if "online_lr" in training:
            out["online_lr"] = training["online_lr"]
        # training.online_ebe_lr → LR for EBE online adapt (lower than online_lr).
        if "online_ebe_lr" in training:
            out["online_ebe_lr"] = training["online_ebe_lr"]
        # tree_training section — hyperparameters for tree-structured distillation losses.
        # Used when --loss is one of {kl_tree, bv_tree, gbv_tree, traversal_tree}.
        # trainer.py accepts --tree_K (number of i.i.d. draft paths) and
        # --tree_L (draft block depth).  Defaults match trainer.py built-in defaults.
        tree_cfg = data.get("tree_training", {})
        out["tree_K"] = tree_cfg.get("tree_K", 4)
        out["tree_L"] = tree_cfg.get("tree_L", 8)
        # experiment.losses → which loss variants to train/eval.
        # If set, only those losses run; all others are silently skipped.
        # Overridden by --losses CLI flag (CLI always wins).
        # Example: losses: [kl, jsd, l1]   (Rahul's A100 promote run)
        #          losses: [online]          (Mukund's online-only debug run)
        if "losses" in experiment_cfg:
            out["losses"] = experiment_cfg["losses"]   # list or null
        # experiment.seed_override → run a specific seed without editing training.seed
        if "seed_override" in experiment_cfg:
            out["seed"] = experiment_cfg["seed_override"]
        # experiment.eval_only → skip Phase 2 (train + merge); only run Phase 1/3/4 evals.
        # Use this when trained checkpoints already exist (e.g. from a previous run) and
        # you only want to measure verifier performance, not re-train.
        if "eval_only" in experiment_cfg:
            out["eval_only"] = bool(experiment_cfg["eval_only"])
        # models.draft / models.target — present only when the YAML sets them.
        # Consumed by main() when the config is a YAML-based profile (not a
        # legacy CONFIGS preset) so we know which model pair to load.
        if models_cfg.get("draft"):
            out["draft"] = models_cfg["draft"]
        if models_cfg.get("target"):
            out["target"] = models_cfg["target"]
        # hardware.load_in_4bit — forwarded so YAML profiles control 4-bit loading.
        if "load_in_4bit" in hardware:
            out["load_in_4bit"] = hardware["load_in_4bit"]
        # models.target → teacher_tag: short human-readable size label extracted from
        # the model name (e.g. "Qwen/Qwen3-4B" → "4B").  Used in run names if
        # run_label is not explicitly set.
        target = models_cfg.get("target", "")
        if target and "run_label" not in logging_cfg:
            import re as _re
            m = _re.search(r"(\d+\.?\d*[BbMm])", target)
            if m:
                size_tag = m.group(1).upper()
                out["run_label"] = f"{size_tag}-{config_name}"
        return out
    except Exception as exc:
        print(f"  [config] Warning: could not load {yaml_path}: {exc}")
        return {}

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

# Training / benchmark scripts
_TRAIN_SCRIPT  = os.path.join(_GBV_RESEARCH, "algorithms", "distillspec_gbv", "trainer.py")
_ONLINE_SCRIPT = os.path.join(_GBV_RESEARCH, "algorithms", "online_serve.py")
_EAGLE_SCRIPT  = os.path.join(_GBV_RESEARCH, "algorithms", "eagle_bench.py")


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
              experiment_tag=None, train_steps=0):
    """Eval command — always passes --skip_existing so restarts are safe.

    Defaults (laptop): n=10 prompts, max_tokens=50.  Run with n=30/max_tokens=100
    on Colab/server T4 for paper-quality results.

    train_steps is passed through to evaluate.py --train_steps so the DB row
    records how many training steps produced this checkpoint.  0 = baseline
    (no training).  Omit / leave 0 for the baseline eval; pass _steps or
    _online_steps for trained models.
    """
    cmd = [
        sys.executable, os.path.join(HERE, "evaluate.py"),
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
    if train_steps:
        cmd += ["--train_steps", str(train_steps)]
    if task_score:
        cmd.append("--task_score")
    if experiment_tag:
        cmd += ["--experiment_tag", experiment_tag]
    return cmd


# Loss name → step-ID prefixes.  Used to filter steps when --losses is given.
_LOSS_STEP_PREFIXES: dict = {
    "kl":               ("train_kl_",               "merge_kl_",               "eval_kl_"),
    "ebe":              ("train_ebe_",               "merge_ebe_",              "eval_ebe_"),
    "ebe_single":       ("train_ebe_single_",        "merge_ebe_single_",       "eval_ebe_single_"),
    "rev_kl":           ("train_rev_kl_",            "merge_rev_kl_",           "eval_rev_kl_"),
    "jsd":              ("train_jsd_",               "merge_jsd_",              "eval_jsd_"),
    "l1":               ("train_l1_",                "merge_l1_",               "eval_l1_"),
    # NOTE: online's merge/eval entries are exact step IDs (not short prefixes) to avoid
    # matching online_ebe / online_ebe_single steps.
    # "merge_online_"  would match "merge_online_ebe_gsm8k"  → ghost steps selected.
    # "eval_online_"   would match "eval_online_ebe_gsm8k"   → ghost steps selected.
    # online_adapt_ is safe: "online_ebe_adapt_" does NOT start with "online_adapt_".
    "online":           ("online_adapt_",            "merge_online_gsm8k",      "eval_online_gsm8k", "eval_online_all"),
    "online_ebe":       ("online_ebe_adapt_",        "merge_online_ebe_",       "eval_online_ebe_"),
    "online_ebe_single":("online_ebe_single_adapt_", "merge_online_ebe_single_","eval_online_ebe_single_"),
    # ── Tree-structured losses (offline, non-OT verifiers) ──────────────────
    # Divergence variants at each tree node (universal — work with any verifier):
    #   kl_tree       → forward KL(p ∥ q) — mode-covering
    #   rev_kl_tree   → reverse KL(q ∥ p) — mode-seeking
    #   jsd_tree      → symmetric JSD — bounded, stable
    # Verifier-specific surrogates (full-vocab integrals):
    #   bv_tree       → bv_verify (block acceptance integral)
    #   gbv_tree      → gbv_verify (bv with q_skew substituted)
    #   traversal_tree → traversal_verify (leaf-weight product)
    # On-policy EBE ablation (token-level, not full-vocab):
    #   ebe_tree      → isolates off-policy mismatch in flat EBE
    "kl_tree":        ("train_kl_tree_",    "merge_kl_tree_",    "eval_kl_tree_"),
    "rev_kl_tree":    ("train_rev_kl_tree_","merge_rev_kl_tree_","eval_rev_kl_tree_"),
    "jsd_tree":       ("train_jsd_tree_",   "merge_jsd_tree_",   "eval_jsd_tree_"),
    "bv_tree":        ("train_bv_tree_",    "merge_bv_tree_",    "eval_bv_tree_"),
    "gbv_tree":       ("train_gbv_tree_",   "merge_gbv_tree_",   "eval_gbv_tree_"),
    "traversal_tree": ("train_trav_tree_",  "merge_trav_tree_",  "eval_trav_tree_"),
    "ebe_tree":       ("train_ebe_tree_",   "merge_ebe_tree_",   "eval_ebe_tree_"),
    # ── Online tree distillation (online serving + on-policy tree training) ──
    # Builds a K-path draft tree at each update step on the served prompt.
    # No replay buffer. On-policy + prompt-distribution-matched.
    # Pairs naturally with standard speculative decoding (alpha/online) verifier.
    "online_kl_tree":  ("online_kl_tree_adapt_",  "merge_online_kl_tree_",  "eval_online_kl_tree_"),
    "online_ebe_tree": ("online_ebe_tree_adapt_",  "merge_online_ebe_tree_", "eval_online_ebe_tree_"),
}
ALL_LOSSES = list(_LOSS_STEP_PREFIXES.keys())


def build_steps(draft, target, experiment_tag=None, smoke=False, eagle=False,
                load_in_4bit=False, ckpt_root=None,
                train_hparams=None, losses_to_run=None):
    """Build the STEPS list for a given draft/target model pair.

    Pipeline structure (same for both smoke and full — only numbers differ):

        Phase 1 — Baseline
            eval_baseline_gsm8k   (always first; establishes the untrained reference)

        Phase 2 — Training   (smoke: 50 steps · full: 1000 steps)
            forward_kl · ebe · reverse_kl · jsd · l1 · online · online_ebe
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

    # Alpha evaluation requires both draft (0.6B BF16) and teacher (8B) in the
    # same evaluate.py process simultaneously.  On Colab (load_in_4bit=True) this
    # exhausts the ~12 GB system RAM and the Linux OOM killer fires with SIGKILL
    # before any Python exception handler can catch it.  The symptom is exit -9 at
    # ~7% of teacher weight loading (layer 2 of 28).
    #
    # Fix: exclude alpha on Colab.  Block-efficiency metrics (BE) run through
    # runner.py subprocesses that start fresh each time (small RAM footprint).
    # Alpha can be run with --config server (A100, full BF16, no 4-bit).
    if load_in_4bit:
        _modes = "bv,gbv,traversal,specinfer,naive"

    # 4-bit flag appended to every training/eval command when load_in_4bit=True.
    # Only set for --config colab (free T4, 15 GB).  Server/A100 loads bf16.
    _4bit = ["--load_in_4bit"] if load_in_4bit else []

    # ── Training hyperparameter args ───────────────────────────────────────────
    # Resolved from train_hparams dict (built in main() from YAML + CLI overrides).
    # Passed to every trainer.py invocation so the YAML / CLI fully controls
    # all hyperparameters without editing this file.
    _h = train_hparams or {}
    # Online adapt: max_new_tokens controls KV-cache size during speculative decode.
    # Smoke keeps this small (30).  Full run reads from YAML (laptop=80, server=128,
    # colab=64).  The KL computation is now memory-efficient (rejects-only selection
    # in _kl_at_positions) so the main remaining constraint is the KV cache.
    _online_max_tok = 30 if smoke else int(_h.get("max_new_tokens", 64))
    _train_hargs = [
        "--lr",              str(_h.get("lr", 3e-5)),
        "--lora_r",          str(_h.get("lora_r", 8)),
        "--lora_alpha",      str(_h.get("lora_alpha", 16)),
        "--teacher_temp",    str(_h.get("teacher_temp", 0.8)),
        "--max_new_tokens",  str(_h.get("max_new_tokens", 80)),
        # Checkpoint cadence — forwarded from YAML checkpointing: section.
        # Determines how much training is lost on Colab timeout / crash.
        "--save_every",      str(_h.get("save_every", 100)),
        "--milestone_every", str(_h.get("milestone_every", 200)),
        "--max_checkpoints", str(_h.get("max_checkpoints", 5)),
        # W&B routing — project, group, and run-label so every run lands in
        # the right W&B project/group and the run name encodes the teacher size.
        # Without these every run uses trainer.py's hard-coded defaults and all
        # configs produce identically-named, ungrouped W&B runs.
        "--wandb_project",   str(_h.get("wandb_project", "distillspec")),
        "--wandb_group",     str(_h.get("wandb_group", "")),
        "--run_label",       str(_h.get("run_label", "")),
    ]
    # Shared args passed to BOTH online adapt commands: lora_r/alpha must match
    # the offline training runs so all models have the same adapter capacity.
    _online_hargs = [
        "--lora_r",     str(_h.get("lora_r", 8)),
        "--lora_alpha", str(_h.get("lora_alpha", 16)),
    ]
    # --compile: passed to trainer.py and online_serve.py when YAML sets
    # hardware.compile: true (server/A100 Linux).  Both scripts skip compile
    # automatically on Windows and PyTorch < 2.0 so this is always safe to pass.
    _compile_flag = ["--compile"] if _h.get("compile") else []

    # Tree-loss extra args — only appended to tree-loss training commands.
    # tree_K: number of i.i.d. draft paths sampled per training step.
    # tree_L: draft block depth (how many tokens per path).
    # Both read from YAML tree_training: section; defaults match trainer.py.
    _tree_hargs = [
        "--tree_K", str(_h.get("tree_K", 4)),
        "--tree_L", str(_h.get("tree_L", 8)),
    ]

    # Override step count if train_steps was specified via CLI
    if _h.get("train_steps") is not None:
        _steps = _h["train_steps"]
        _online_steps = _h["train_steps"]

    # ── Checkpoint directory resolver ─────────────────────────────────────────
    # --ckpt_root overrides the default gbv-research/db/checkpoints/ location.
    # Use this for ephemeral compute (Colab, Modal, Kaggle) where local disk
    # is wiped on session death and you want checkpoints on persistent storage:
    #   Colab + Google Drive: --ckpt_root /content/drive/MyDrive/specdist/checkpoints
    #   Modal:                set via modal_train.py (volume at /vol/checkpoints/)
    #   Kaggle:               --ckpt_root /kaggle/working/specdist/checkpoints
    #
    # NOTE: _ckpt and _merged MUST be assigned in both branches of this if/else.
    # Python marks any name assigned anywhere inside a function as "local" for
    # the entire function body.  A one-sided `if ckpt_root: def _ckpt …` means
    # _ckpt is local-but-unbound when ckpt_root is None → UnboundLocalError.
    if ckpt_root:
        os.makedirs(ckpt_root, exist_ok=True)
        _ckpt   = lambda name: os.path.join(ckpt_root, name)           # noqa: E731
        _merged = lambda name: os.path.join(ckpt_root, name + "_merged")  # noqa: E731
        print(f"  [pipeline] Checkpoint root overridden: {ckpt_root}")
    else:
        # Fall back to the module-level helpers (gbv-research/db/checkpoints/ with
        # OSD legacy fallback).  globals()["_ckpt"] is used instead of a bare `_ckpt`
        # reference so that Python's local-variable detection doesn't complain.
        _ckpt   = globals()["_ckpt"]    # noqa: E731 — module-level default
        _merged = globals()["_merged"]  # noqa: E731

    # Labels that use online_steps (smaller budget, online distillation).
    _ONLINE_LABELS = {"online", "online_ebe", "online_ebe_single"}

    def _ec(student_path, label, datasets="gsm8k", task_score=False, modes=None):
        """Shorthand: eval cmd with smoke-aware parameters.

        Automatically infers train_steps from label:
          baseline           → 0    (no training)
          online / *         → _online_steps
          everything else    → _steps
        This keeps every DB row honest without touching each call site.

        modes: verifier mode string passed to evaluate.py --modes.
               Defaults to _modes (all 6 verifiers) when None.
               Tree-loss eval steps pass their paired mode(s) explicitly.
        """
        if label == "baseline":
            ts = 0
        elif label in _ONLINE_LABELS:
            ts = _online_steps
        else:
            ts = _steps
        _eval_modes = modes if modes is not None else _modes
        cmd = _eval_cmd(student_path, label, target,
                        datasets=datasets, modes=_eval_modes, Ks=_Ks, temps=_temps,
                        n=_n, max_tokens=_max_tok,
                        task_score=task_score, experiment_tag=experiment_tag,
                        train_steps=ts)
        return cmd + _4bit  # append --load_in_4bit for colab config

    # Tree-loss eval mode strategy
    # ─────────────────────────────────────────────────────────────────────────
    # Smoke: each tree-loss model is evaluated with ONLY its paired verifier.
    #   • Verifies the full code path (train → merge → eval) in minimal time.
    #   • One mode per model = fastest possible smoke check.
    #
    # Full: all three non-OT verifiers ("bv,gbv,traversal") for every tree model.
    #   • Gives the cross-matrix table: does gbv_tree also help traversal? etc.
    #   • OT-based verifiers (specinfer, naive, alpha) are deliberately excluded
    #     — tree losses don't target their per-token OTLP acceptance criterion.
    #
    # kl_tree is the universal on-policy baseline: always runs all 3 non-OT modes.
    # ─────────────────────────────────────────────────────────────────────────
    _TREE_NON_OT = "bv,gbv,traversal"   # full-run modes for all offline tree losses
    _TREE_PAIRED = {                     # smoke: just the naturally paired verifier
        # Divergence variants — universal baselines, test all 3 even in smoke
        "kl_tree":        _TREE_NON_OT,
        "rev_kl_tree":    _TREE_NON_OT,
        "jsd_tree":       _TREE_NON_OT,
        # Verifier-specific surrogates — paired with their target verifier in smoke
        "bv_tree":        "bv",
        "gbv_tree":       "gbv",
        "traversal_tree": "traversal",
        # On-policy EBE ablation — paired with bv in smoke (closest structural match)
        "ebe_tree":       "bv",
    }

    # ── Loss filter helper ─────────────────────────────────────────────────────
    def _is_loss_step(sid: str) -> bool:
        """True if this step ID belongs to a specific loss (train/merge/eval)."""
        return any(
            sid.startswith(pfx)
            for prefixes in _LOSS_STEP_PREFIXES.values()
            for pfx in prefixes
        )

    def _loss_selected(sid: str) -> bool:
        """True if this loss-specific step is for one of the selected losses."""
        selected = set(losses_to_run) if losses_to_run else set(ALL_LOSSES)
        return any(
            sid.startswith(pfx)
            for name in selected
            for pfx in _LOSS_STEP_PREFIXES.get(name, ())
        )

    # ── Build full step list then apply loss filter ────────────────────────
    _steps_list = [
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
        # Losses covered: forward_kl · ebe · reverse_kl · jsd · l1 · online · online_ebe
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
                "--steps", str(_steps),
                "--draft", draft, "--target", target,
                "--dataset", _data("gsm8k_train.jsonl"),
                "--output", _ckpt("kl-gsm8k"),
                *_train_hargs,
            ] + _4bit + _compile_flag,
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
                "--steps", str(_steps),
                "--nan_action", "skip",
                "--early_stop_patience", "3",
                "--draft", draft, "--target", target,
                "--dataset", _data("gsm8k_train.jsonl"),
                "--output", _ckpt("ebe-gsm8k"),
                *_train_hargs,
            ] + _4bit + _compile_flag,
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
        # -------------------------------------------------------------------
        # ebe_single: ablation of multi-token EBE.
        # Loss = −mean(α), no cumprod, no KL regulariser.  Ran to full --steps
        # (no early_stop_patience) so it gets the same budget as forward_kl.
        # Key question: if single-token EBE beats KL, the product structure in
        # multi-token EBE is the bottleneck; if not, EBE concept fails offline.
        # -------------------------------------------------------------------
        {
            "id": "train_ebe_single_gsm8k",
            "group": "Phase 2 — Training",
            "desc": f"Train ebe_single, {_steps} steps, gsm8k_train",
            "cmd": [
                sys.executable, _TRAIN_SCRIPT,
                "--loss", "ebe_single",
                "--steps", str(_steps),
                "--nan_action", "skip",
                "--draft", draft, "--target", target,
                "--dataset", _data("gsm8k_train.jsonl"),
                "--output", _ckpt("ebe_single-gsm8k"),
                *_train_hargs,
            ] + _4bit + _compile_flag,
            "done_check": os.path.join(_ckpt("ebe_single-gsm8k"), "adapter_model.safetensors"),
            "retryable": True,
        },
        {
            "id": "merge_ebe_single_gsm8k",
            "group": "Phase 2 — Training",
            "desc": "Merge ebe_single-gsm8k LoRA",
            "cmd": [sys.executable, _TRAIN_SCRIPT, "--merge_only",
                    "--adapter", _ckpt("ebe_single-gsm8k"), "--draft", draft],
            "done_check": os.path.join(_merged("ebe_single-gsm8k"), "config.json"),
        },
        {
            "id": "train_rev_kl_gsm8k",
            "group": "Phase 2 — Training",
            "desc": f"Train reverse_kl, {_steps} steps, gsm8k_train",
            "cmd": [
                sys.executable, _TRAIN_SCRIPT,
                "--loss", "reverse_kl",
                "--steps", str(_steps),
                "--nan_action", "skip", "--early_stop_patience", "3",
                "--draft", draft, "--target", target,
                "--dataset", _data("gsm8k_train.jsonl"),
                "--output", _ckpt("rev_kl-gsm8k"),
                *_train_hargs,
            ] + _4bit + _compile_flag,
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
                "--steps", str(_steps),
                "--nan_action", "skip", "--early_stop_patience", "3",
                "--draft", draft, "--target", target,
                "--dataset", _data("gsm8k_train.jsonl"),
                "--output", _ckpt("jsd-gsm8k"),
                *_train_hargs,
            ] + _4bit + _compile_flag,
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
                "--steps", str(_steps),
                "--nan_action", "skip", "--early_stop_patience", "3",
                "--draft", draft, "--target", target,
                "--dataset", _data("gsm8k_train.jsonl"),
                "--output", _ckpt("l1-gsm8k"),
                *_train_hargs,
            ] + _4bit + _compile_flag,
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
                "--lr", str(_h.get("online_lr", 3e-4)),
                # _online_max_tok: 30 smoke / YAML value full run (laptop=80, server=128, colab=64)
                "--max_new_tokens", str(_online_max_tok),
                *_online_hargs,   # --lora_r, --lora_alpha — must match offline training runs
            ] + _4bit + _compile_flag,
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
        # online_ebe: same setup as online (forward_kl) but uses --kl_method ebe.
        # Direct comparison: online_ebe vs online isolates whether block-level
        # EBE gradient weighting improves over flat KL at rejected positions.
        # Both see the same rejected tokens (real α < 1 events from live SD);
        # only the loss function differs.
        # -------------------------------------------------------------------
        {
            "id": "online_ebe_adapt_gsm8k",
            "group": "Phase 2 — Training",
            "desc": f"Online EBE adaptation (ebe, K=4), {_online_steps} prompts, gsm8k",
            "cmd": [
                sys.executable, _ONLINE_SCRIPT,
                "--prompts", _data("gsm8k_train.jsonl"),
                "--draft", draft, "--target", target,
                "--output", _ckpt("online-ebe-gsm8k"),
                "--steps", str(_online_steps),
                "--update_every", "4",
                "--K", "4",
                "--kl_method", "ebe",
                "--ebe_block_len", "4",    # match --K
                "--ebe_kl_weight", "0.1",
                # online_ebe uses a lower LR than forward_kl online (1e-4 vs 3e-4):
                # EBE's cumprod gradient is more volatile — same step size causes
                # mode collapse after ~100 steps.  Observed: eval_be peaked at step
                # 100 (3.302) then collapsed to 2.338 (below baseline 2.835) at step
                # 150 when run with lr=3e-4.  Use separate online_ebe_lr key so the
                # two online methods can be tuned independently.
                "--lr", str(_h.get("online_ebe_lr", 1e-4)),
                "--max_new_tokens", str(_online_max_tok),
                # milestone checkpoints every 50 steps: preserve the best model even
                # if it degrades later.  Forward_kl online is stable so doesn't need
                # this; EBE online can peak early and then collapse.
                "--milestone_every", "50",
                # early stopping: if eval_alpha worsens for 3 consecutive eval windows
                # (eval_every=50 steps by default), stop and keep ckpt_latest as-is.
                "--early_stop_patience", "3",
                *_online_hargs,   # --lora_r, --lora_alpha — must match offline training runs
            ] + _4bit + _compile_flag,
            "done_check": os.path.join(_ckpt("online-ebe-gsm8k"), "adapter_model.safetensors"),
            "retryable": True,
        },
        {
            "id": "merge_online_ebe_gsm8k",
            "group": "Phase 2 — Training",
            "desc": "Merge online-ebe-gsm8k LoRA",
            "cmd": [sys.executable, _TRAIN_SCRIPT, "--merge_only",
                    "--adapter", _ckpt("online-ebe-gsm8k"), "--draft", draft],
            "done_check": os.path.join(_merged("online-ebe-gsm8k"), "config.json"),
        },
        # -------------------------------------------------------------------
        # online_ebe_single: online ablation — single-token EBE at rejected positions.
        # Directly comparable to online_ebe (block-level) with the same budget.
        # Uses the same LR as online_ebe (lower than forward_kl online) since
        # gradient scale is similar (both O(α) per step).
        # -------------------------------------------------------------------
        {
            "id": "online_ebe_single_adapt_gsm8k",
            "group": "Phase 2 — Training",
            "desc": f"Online EBE-single adaptation (ebe_single, K=4), {_online_steps} prompts, gsm8k",
            "cmd": [
                sys.executable, _ONLINE_SCRIPT,
                "--prompts", _data("gsm8k_train.jsonl"),
                "--draft", draft, "--target", target,
                "--output", _ckpt("online-ebe-single-gsm8k"),
                "--steps", str(_online_steps),
                "--update_every", "4",
                "--K", "4",
                "--kl_method", "ebe_single",
                "--lr", str(_h.get("online_ebe_lr", 1e-4)),
                "--max_new_tokens", str(_online_max_tok),
                "--milestone_every", "50",
                "--early_stop_patience", "3",
                *_online_hargs,
            ] + _4bit + _compile_flag,
            "done_check": os.path.join(_ckpt("online-ebe-single-gsm8k"), "adapter_model.safetensors"),
            "retryable": True,
        },
        {
            "id": "merge_online_ebe_single_gsm8k",
            "group": "Phase 2 — Training",
            "desc": "Merge online-ebe-single-gsm8k LoRA",
            "cmd": [sys.executable, _TRAIN_SCRIPT, "--merge_only",
                    "--adapter", _ckpt("online-ebe-single-gsm8k"), "--draft", draft],
            "done_check": os.path.join(_merged("online-ebe-single-gsm8k"), "config.json"),
        },

        # -------------------------------------------------------------------
        # Tree-structured distillation losses — Phase 2 (Training)
        #
        # Unlike flat sequence losses (forward_kl, ebe, …) which train on the
        # teacher's linear rollout, tree losses train on the STUDENT's own
        # draft tree:
        #   1. iid_draft()  — sample K paths without grad
        #   2. target_tree_pass()  — score each node with teacher (no grad)
        #   3. draft_tree_forward_with_grad()  — re-run student WITH grad
        #   4. compute_tree_loss()  — verifier-specific differentiable surrogate
        #
        # Verifier ↔ loss pairing:
        #   kl_tree        → universal on-policy KL baseline; works with all verifiers
        #   bv_tree        → targets bv_verify block acceptance integral exactly
        #   gbv_tree       → targets gbv_verify (bv with q_skew substituted)
        #   traversal_tree → targets traversal_verify leaf-weight product
        #
        # OT-based verifiers (naive, nss, spectr, specinfer, khisti) are NOT
        # included here — those use per-token OTLP solvers that flat sequence
        # losses (forward_kl, ebe) already target correctly.
        #
        # All tree losses use --nan_action skip + --early_stop_patience 3 as
        # a safety net: the tree forward pass involves cumsum + clamp + relu
        # chains that can produce very small gradients under certain q_skew
        # fallbacks, and numerical instability (though rare) is possible.
        # -------------------------------------------------------------------
        {
            "id": "train_kl_tree_gsm8k",
            "group": "Phase 2 — Training",
            "desc": f"Train kl_tree (on-policy KL), {_steps} steps, gsm8k_train",
            "cmd": [
                sys.executable, _TRAIN_SCRIPT,
                "--loss", "kl_tree",
                "--steps", str(_steps),
                "--draft", draft, "--target", target,
                "--dataset", _data("gsm8k_train.jsonl"),
                "--output", _ckpt("kl_tree-gsm8k"),
                *_train_hargs, *_tree_hargs,
            ] + _4bit + _compile_flag,
            "done_check": os.path.join(_ckpt("kl_tree-gsm8k"), "adapter_model.safetensors"),
            "retryable": True,
        },
        {
            "id": "merge_kl_tree_gsm8k",
            "group": "Phase 2 — Training",
            "desc": "Merge kl_tree-gsm8k LoRA",
            "cmd": [sys.executable, _TRAIN_SCRIPT, "--merge_only",
                    "--adapter", _ckpt("kl_tree-gsm8k"), "--draft", draft],
            "done_check": os.path.join(_merged("kl_tree-gsm8k"), "config.json"),
        },
        {
            "id": "train_bv_tree_gsm8k",
            "group": "Phase 2 — Training",
            "desc": f"Train bv_tree (BV block acceptance), {_steps} steps, gsm8k_train",
            "cmd": [
                sys.executable, _TRAIN_SCRIPT,
                "--loss", "bv_tree",
                "--steps", str(_steps),
                "--nan_action", "skip", "--early_stop_patience", "3",
                "--draft", draft, "--target", target,
                "--dataset", _data("gsm8k_train.jsonl"),
                "--output", _ckpt("bv_tree-gsm8k"),
                *_train_hargs, *_tree_hargs,
            ] + _4bit + _compile_flag,
            "done_check": os.path.join(_ckpt("bv_tree-gsm8k"), "adapter_model.safetensors"),
            "retryable": True,
        },
        {
            "id": "merge_bv_tree_gsm8k",
            "group": "Phase 2 — Training",
            "desc": "Merge bv_tree-gsm8k LoRA",
            "cmd": [sys.executable, _TRAIN_SCRIPT, "--merge_only",
                    "--adapter", _ckpt("bv_tree-gsm8k"), "--draft", draft],
            "done_check": os.path.join(_merged("bv_tree-gsm8k"), "config.json"),
        },
        {
            "id": "train_gbv_tree_gsm8k",
            "group": "Phase 2 — Training",
            "desc": f"Train gbv_tree (GBV with q_skew), {_steps} steps, gsm8k_train",
            "cmd": [
                sys.executable, _TRAIN_SCRIPT,
                "--loss", "gbv_tree",
                "--steps", str(_steps),
                "--nan_action", "skip", "--early_stop_patience", "3",
                "--draft", draft, "--target", target,
                "--dataset", _data("gsm8k_train.jsonl"),
                "--output", _ckpt("gbv_tree-gsm8k"),
                *_train_hargs, *_tree_hargs,
            ] + _4bit + _compile_flag,
            "done_check": os.path.join(_ckpt("gbv_tree-gsm8k"), "adapter_model.safetensors"),
            "retryable": True,
        },
        {
            "id": "merge_gbv_tree_gsm8k",
            "group": "Phase 2 — Training",
            "desc": "Merge gbv_tree-gsm8k LoRA",
            "cmd": [sys.executable, _TRAIN_SCRIPT, "--merge_only",
                    "--adapter", _ckpt("gbv_tree-gsm8k"), "--draft", draft],
            "done_check": os.path.join(_merged("gbv_tree-gsm8k"), "config.json"),
        },
        {
            "id": "train_trav_tree_gsm8k",
            "group": "Phase 2 — Training",
            "desc": f"Train traversal_tree (leaf weight), {_steps} steps, gsm8k_train",
            "cmd": [
                sys.executable, _TRAIN_SCRIPT,
                "--loss", "traversal_tree",
                "--steps", str(_steps),
                "--nan_action", "skip", "--early_stop_patience", "3",
                "--draft", draft, "--target", target,
                "--dataset", _data("gsm8k_train.jsonl"),
                "--output", _ckpt("trav_tree-gsm8k"),
                *_train_hargs, *_tree_hargs,
            ] + _4bit + _compile_flag,
            "done_check": os.path.join(_ckpt("trav_tree-gsm8k"), "adapter_model.safetensors"),
            "retryable": True,
        },
        {
            "id": "merge_trav_tree_gsm8k",
            "group": "Phase 2 — Training",
            "desc": "Merge trav_tree-gsm8k LoRA",
            "cmd": [sys.executable, _TRAIN_SCRIPT, "--merge_only",
                    "--adapter", _ckpt("trav_tree-gsm8k"), "--draft", draft],
            "done_check": os.path.join(_merged("trav_tree-gsm8k"), "config.json"),
        },
        {
            "id": "train_ebe_tree_gsm8k",
            "group": "Phase 2 — Training",
            "desc": f"Train ebe_tree (on-policy EBE ablation), {_steps} steps, gsm8k_train",
            "cmd": [
                sys.executable, _TRAIN_SCRIPT,
                "--loss", "ebe_tree",
                "--steps", str(_steps),
                "--nan_action", "skip", "--early_stop_patience", "3",
                "--draft", draft, "--target", target,
                "--dataset", _data("gsm8k_train.jsonl"),
                "--output", _ckpt("ebe_tree-gsm8k"),
                *_train_hargs, *_tree_hargs,
            ] + _4bit + _compile_flag,
            "done_check": os.path.join(_ckpt("ebe_tree-gsm8k"), "adapter_model.safetensors"),
            "retryable": True,
        },
        {
            "id": "merge_ebe_tree_gsm8k",
            "group": "Phase 2 — Training",
            "desc": "Merge ebe_tree-gsm8k LoRA",
            "cmd": [sys.executable, _TRAIN_SCRIPT, "--merge_only",
                    "--adapter", _ckpt("ebe_tree-gsm8k"), "--draft", draft],
            "done_check": os.path.join(_merged("ebe_tree-gsm8k"), "config.json"),
        },

        # rev_kl_tree and jsd_tree — on-policy divergence variant ablations.
        # Same tree data as kl_tree; only the per-node divergence changes.
        # rev_kl_tree: mode-seeking (draft concentrates on target's peaks)
        # jsd_tree:    symmetric, bounded — more stable than forward/reverse KL
        {
            "id": "train_rev_kl_tree_gsm8k",
            "group": "Phase 2 — Training",
            "desc": f"Train rev_kl_tree (on-policy reverse KL), {_steps} steps, gsm8k_train",
            "cmd": [
                sys.executable, _TRAIN_SCRIPT,
                "--loss", "rev_kl_tree",
                "--steps", str(_steps),
                "--nan_action", "skip", "--early_stop_patience", "3",
                "--draft", draft, "--target", target,
                "--dataset", _data("gsm8k_train.jsonl"),
                "--output", _ckpt("rev_kl_tree-gsm8k"),
                *_train_hargs, *_tree_hargs,
            ] + _4bit + _compile_flag,
            "done_check": os.path.join(_ckpt("rev_kl_tree-gsm8k"), "adapter_model.safetensors"),
            "retryable": True,
        },
        {
            "id": "merge_rev_kl_tree_gsm8k",
            "group": "Phase 2 — Training",
            "desc": "Merge rev_kl_tree-gsm8k LoRA",
            "cmd": [sys.executable, _TRAIN_SCRIPT, "--merge_only",
                    "--adapter", _ckpt("rev_kl_tree-gsm8k"), "--draft", draft],
            "done_check": os.path.join(_merged("rev_kl_tree-gsm8k"), "config.json"),
        },
        {
            "id": "train_jsd_tree_gsm8k",
            "group": "Phase 2 — Training",
            "desc": f"Train jsd_tree (on-policy JSD), {_steps} steps, gsm8k_train",
            "cmd": [
                sys.executable, _TRAIN_SCRIPT,
                "--loss", "jsd_tree",
                "--steps", str(_steps),
                "--nan_action", "skip", "--early_stop_patience", "3",
                "--draft", draft, "--target", target,
                "--dataset", _data("gsm8k_train.jsonl"),
                "--output", _ckpt("jsd_tree-gsm8k"),
                *_train_hargs, *_tree_hargs,
            ] + _4bit + _compile_flag,
            "done_check": os.path.join(_ckpt("jsd_tree-gsm8k"), "adapter_model.safetensors"),
            "retryable": True,
        },
        {
            "id": "merge_jsd_tree_gsm8k",
            "group": "Phase 2 — Training",
            "desc": "Merge jsd_tree-gsm8k LoRA",
            "cmd": [sys.executable, _TRAIN_SCRIPT, "--merge_only",
                    "--adapter", _ckpt("jsd_tree-gsm8k"), "--draft", draft],
            "done_check": os.path.join(_merged("jsd_tree-gsm8k"), "config.json"),
        },

        # -------------------------------------------------------------------
        # Online tree distillation — on-policy tree updates during online serving.
        # Replaces the flat replay-buffer update with a K-path draft tree step on
        # each served prompt.  No rejected-position bias; no zero-gradient EBE bug.
        # Most natural for standard speculative decoding (alpha/online) verifier.
        # -------------------------------------------------------------------
        {
            "id": "online_kl_tree_adapt_gsm8k",
            "group": "Phase 2 — Training",
            "desc": f"Online tree adapt: kl_tree, tree_K={_h.get('tree_K', 4)}, {_online_steps} steps, gsm8k",
            "cmd": [
                sys.executable, _ONLINE_SCRIPT,
                "--prompts", _data("gsm8k_train.jsonl"),
                "--draft", draft, "--target", target,
                "--output", _ckpt("online-kl-tree-gsm8k"),
                "--steps", str(_online_steps),
                "--update_every", "4",
                "--K", "4",
                "--tree_loss", "kl_tree",
                "--tree_K", str(min(_h.get("tree_K", 4), 4)),  # max K=4 for safety
                "--tree_L", str(_h.get("tree_L", 8)),
                "--lr", str(_h.get("online_lr", 3e-4)),
                "--max_new_tokens", str(_online_max_tok),
                *_online_hargs,
            ] + _4bit + _compile_flag,
            "done_check": os.path.join(_ckpt("online-kl-tree-gsm8k"), "adapter_model.safetensors"),
            "retryable": True,
        },
        {
            "id": "merge_online_kl_tree_gsm8k",
            "group": "Phase 2 — Training",
            "desc": "Merge online-kl-tree-gsm8k LoRA",
            "cmd": [sys.executable, _TRAIN_SCRIPT, "--merge_only",
                    "--adapter", _ckpt("online-kl-tree-gsm8k"), "--draft", draft],
            "done_check": os.path.join(_merged("online-kl-tree-gsm8k"), "config.json"),
        },
        {
            "id": "online_ebe_tree_adapt_gsm8k",
            "group": "Phase 2 — Training",
            "desc": f"Online tree adapt: ebe_tree, tree_K={_h.get('tree_K', 4)}, {_online_steps} steps, gsm8k",
            "cmd": [
                sys.executable, _ONLINE_SCRIPT,
                "--prompts", _data("gsm8k_train.jsonl"),
                "--draft", draft, "--target", target,
                "--output", _ckpt("online-ebe-tree-gsm8k"),
                "--steps", str(_online_steps),
                "--update_every", "4",
                "--K", "4",
                "--tree_loss", "ebe_tree",
                "--tree_K", str(min(_h.get("tree_K", 4), 4)),
                "--tree_L", str(_h.get("tree_L", 8)),
                "--lr", str(_h.get("online_lr", 3e-4)),
                "--max_new_tokens", str(_online_max_tok),
                *_online_hargs,
            ] + _4bit + _compile_flag,
            "done_check": os.path.join(_ckpt("online-ebe-tree-gsm8k"), "adapter_model.safetensors"),
            "retryable": True,
        },
        {
            "id": "merge_online_ebe_tree_gsm8k",
            "group": "Phase 2 — Training",
            "desc": "Merge online-ebe-tree-gsm8k LoRA",
            "cmd": [sys.executable, _TRAIN_SCRIPT, "--merge_only",
                    "--adapter", _ckpt("online-ebe-tree-gsm8k"), "--draft", draft],
            "done_check": os.path.join(_merged("online-ebe-tree-gsm8k"), "config.json"),
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
            "requires": os.path.join(_merged("kl-gsm8k"), "config.json"),
        },
        {
            "id": "eval_ebe_gsm8k",
            "group": "Phase 3 — GSM8K Eval",
            "desc": "Eval ebe-gsm8k on gsm8k",
            "cmd": _ec(_merged("ebe-gsm8k"), "ebe", datasets="gsm8k", task_score=True),
            "done_check": None,
            "requires": os.path.join(_merged("ebe-gsm8k"), "config.json"),
        },
        {
            "id": "eval_rev_kl_gsm8k",
            "group": "Phase 3 — GSM8K Eval",
            "desc": "Eval rev_kl-gsm8k on gsm8k",
            "cmd": _ec(_merged("rev_kl-gsm8k"), "rev_kl", datasets="gsm8k", task_score=True),
            "done_check": None,
            "requires": os.path.join(_merged("rev_kl-gsm8k"), "config.json"),
        },
        {
            "id": "eval_jsd_gsm8k",
            "group": "Phase 3 — GSM8K Eval",
            "desc": "Eval jsd-gsm8k on gsm8k",
            "cmd": _ec(_merged("jsd-gsm8k"), "jsd", datasets="gsm8k", task_score=True),
            "done_check": None,
            "requires": os.path.join(_merged("jsd-gsm8k"), "config.json"),
        },
        {
            "id": "eval_l1_gsm8k",
            "group": "Phase 3 — GSM8K Eval",
            "desc": "Eval l1-gsm8k on gsm8k",
            "cmd": _ec(_merged("l1-gsm8k"), "l1", datasets="gsm8k", task_score=True),
            "done_check": None,
            "requires": os.path.join(_merged("l1-gsm8k"), "config.json"),
        },
        {
            "id": "eval_online_gsm8k",
            "group": "Phase 3 — GSM8K Eval",
            "desc": "Eval online-gsm8k on gsm8k",
            "cmd": _ec(_merged("online-gsm8k"), "online", datasets="gsm8k", task_score=True),
            "done_check": None,
            "requires": os.path.join(_merged("online-gsm8k"), "config.json"),
        },
        {
            "id": "eval_online_ebe_gsm8k",
            "group": "Phase 3 — GSM8K Eval",
            "desc": "Eval online-ebe-gsm8k on gsm8k",
            "cmd": _ec(_merged("online-ebe-gsm8k"), "online_ebe", datasets="gsm8k", task_score=True),
            "done_check": None,
            "requires": os.path.join(_merged("online-ebe-gsm8k"), "config.json"),
        },
        {
            "id": "eval_ebe_single_gsm8k",
            "group": "Phase 3 — GSM8K Eval",
            "desc": "Eval ebe_single-gsm8k on gsm8k",
            "cmd": _ec(_merged("ebe_single-gsm8k"), "ebe_single", datasets="gsm8k", task_score=True),
            "done_check": None,
            "requires": os.path.join(_merged("ebe_single-gsm8k"), "config.json"),
        },
        {
            "id": "eval_online_ebe_single_gsm8k",
            "group": "Phase 3 — GSM8K Eval",
            "desc": "Eval online-ebe-single-gsm8k on gsm8k",
            "cmd": _ec(_merged("online-ebe-single-gsm8k"), "online_ebe_single", datasets="gsm8k", task_score=True),
            "done_check": None,
            "requires": os.path.join(_merged("online-ebe-single-gsm8k"), "config.json"),
        },
        # Tree losses — Phase 3 evals.
        # Smoke: each model runs ONLY its paired verifier (fast code-path check).
        # Full:  all three non-OT verifiers (bv, gbv, traversal) for cross-matrix.
        # OT-based verifiers (specinfer, naive, alpha) are excluded — tree losses
        # don't target their per-token OTLP acceptance criterion.
        {
            "id": "eval_kl_tree_gsm8k",
            "group": "Phase 3 — GSM8K Eval",
            "desc": "Eval kl_tree-gsm8k on gsm8k [bv+gbv+traversal]",
            "cmd": _ec(_merged("kl_tree-gsm8k"), "kl_tree", datasets="gsm8k",
                       task_score=True,
                       modes=_TREE_PAIRED["kl_tree"]),   # bv,gbv,traversal (smoke+full)
            "done_check": None,
            "requires": os.path.join(_merged("kl_tree-gsm8k"), "config.json"),
        },
        {
            "id": "eval_bv_tree_gsm8k",
            "group": "Phase 3 — GSM8K Eval",
            "desc": "Eval bv_tree-gsm8k on gsm8k [bv | bv+gbv+traversal]",
            "cmd": _ec(_merged("bv_tree-gsm8k"), "bv_tree", datasets="gsm8k",
                       task_score=True,
                       modes=_TREE_PAIRED["bv_tree"] if smoke else _TREE_NON_OT),
            "done_check": None,
            "requires": os.path.join(_merged("bv_tree-gsm8k"), "config.json"),
        },
        {
            "id": "eval_gbv_tree_gsm8k",
            "group": "Phase 3 — GSM8K Eval",
            "desc": "Eval gbv_tree-gsm8k on gsm8k [gbv | bv+gbv+traversal]",
            "cmd": _ec(_merged("gbv_tree-gsm8k"), "gbv_tree", datasets="gsm8k",
                       task_score=True,
                       modes=_TREE_PAIRED["gbv_tree"] if smoke else _TREE_NON_OT),
            "done_check": None,
            "requires": os.path.join(_merged("gbv_tree-gsm8k"), "config.json"),
        },
        {
            "id": "eval_trav_tree_gsm8k",
            "group": "Phase 3 — GSM8K Eval",
            "desc": "Eval trav_tree-gsm8k on gsm8k [traversal | bv+gbv+traversal]",
            "cmd": _ec(_merged("trav_tree-gsm8k"), "traversal_tree", datasets="gsm8k",
                       task_score=True,
                       modes=_TREE_PAIRED["traversal_tree"] if smoke else _TREE_NON_OT),
            "done_check": None,
            "requires": os.path.join(_merged("trav_tree-gsm8k"), "config.json"),
        },
        {
            "id": "eval_ebe_tree_gsm8k",
            "group": "Phase 3 — GSM8K Eval",
            "desc": "Eval ebe_tree-gsm8k on gsm8k [bv | bv+gbv+traversal]",
            "cmd": _ec(_merged("ebe_tree-gsm8k"), "ebe_tree", datasets="gsm8k",
                       task_score=True,
                       modes=_TREE_PAIRED["ebe_tree"] if smoke else _TREE_NON_OT),
            "done_check": None,
            "requires": os.path.join(_merged("ebe_tree-gsm8k"), "config.json"),
        },
        {
            "id": "eval_rev_kl_tree_gsm8k",
            "group": "Phase 3 — GSM8K Eval",
            "desc": "Eval rev_kl_tree-gsm8k on gsm8k [bv+gbv+traversal]",
            "cmd": _ec(_merged("rev_kl_tree-gsm8k"), "rev_kl_tree", datasets="gsm8k",
                       task_score=True, modes=_TREE_NON_OT),
            "done_check": None,
            "requires": os.path.join(_merged("rev_kl_tree-gsm8k"), "config.json"),
        },
        {
            "id": "eval_jsd_tree_gsm8k",
            "group": "Phase 3 — GSM8K Eval",
            "desc": "Eval jsd_tree-gsm8k on gsm8k [bv+gbv+traversal]",
            "cmd": _ec(_merged("jsd_tree-gsm8k"), "jsd_tree", datasets="gsm8k",
                       task_score=True, modes=_TREE_NON_OT),
            "done_check": None,
            "requires": os.path.join(_merged("jsd_tree-gsm8k"), "config.json"),
        },
        # Online tree models are evaluated against all 6 verifier modes — they are
        # general draft models that benefit from the full eval table.
        {
            "id": "eval_online_kl_tree_gsm8k",
            "group": "Phase 3 — GSM8K Eval",
            "desc": "Eval online-kl-tree-gsm8k on gsm8k [all 6 verifiers]",
            "cmd": _ec(_merged("online-kl-tree-gsm8k"), "online_kl_tree",
                       datasets="gsm8k", task_score=True),
            "done_check": None,
            "requires": os.path.join(_merged("online-kl-tree-gsm8k"), "config.json"),
        },
        {
            "id": "eval_online_ebe_tree_gsm8k",
            "group": "Phase 3 — GSM8K Eval",
            "desc": "Eval online-ebe-tree-gsm8k on gsm8k [all 6 verifiers]",
            "cmd": _ec(_merged("online-ebe-tree-gsm8k"), "online_ebe_tree",
                       datasets="gsm8k", task_score=True),
            "done_check": None,
            "requires": os.path.join(_merged("online-ebe-tree-gsm8k"), "config.json"),
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
            "requires": os.path.join(_merged("kl-gsm8k"), "config.json"),
        },
        {
            "id": "eval_ebe_all",
            "group": "Phase 4 — Multi-Dataset",
            "desc": "Eval ebe on humaneval,math500,mtbench,alpaca",
            "cmd": _ec(_merged("ebe-gsm8k"), "ebe",
                       datasets="humaneval,math500,mtbench,alpaca", task_score=True),
            "done_check": None,
            "smoke_skip": smoke,
            "requires": os.path.join(_merged("ebe-gsm8k"), "config.json"),
        },
        {
            "id": "eval_rev_kl_all",
            "group": "Phase 4 — Multi-Dataset",
            "desc": "Eval rev_kl on humaneval,math500,mtbench,alpaca",
            "cmd": _ec(_merged("rev_kl-gsm8k"), "rev_kl",
                       datasets="humaneval,math500,mtbench,alpaca", task_score=True),
            "done_check": None,
            "smoke_skip": smoke,
            "requires": os.path.join(_merged("rev_kl-gsm8k"), "config.json"),
        },
        {
            "id": "eval_jsd_all",
            "group": "Phase 4 — Multi-Dataset",
            "desc": "Eval jsd on humaneval,math500,mtbench,alpaca",
            "cmd": _ec(_merged("jsd-gsm8k"), "jsd",
                       datasets="humaneval,math500,mtbench,alpaca", task_score=True),
            "done_check": None,
            "smoke_skip": smoke,
            "requires": os.path.join(_merged("jsd-gsm8k"), "config.json"),
        },
        {
            "id": "eval_l1_all",
            "group": "Phase 4 — Multi-Dataset",
            "desc": "Eval l1 on humaneval,math500,mtbench,alpaca",
            "cmd": _ec(_merged("l1-gsm8k"), "l1",
                       datasets="humaneval,math500,mtbench,alpaca", task_score=True),
            "done_check": None,
            "smoke_skip": smoke,
            "requires": os.path.join(_merged("l1-gsm8k"), "config.json"),
        },
        {
            "id": "eval_online_all",
            "group": "Phase 4 — Multi-Dataset",
            "desc": "Eval online on humaneval,math500,mtbench,alpaca",
            "cmd": _ec(_merged("online-gsm8k"), "online",
                       datasets="humaneval,math500,mtbench,alpaca", task_score=True),
            "done_check": None,
            "smoke_skip": smoke,
            "requires": os.path.join(_merged("online-gsm8k"), "config.json"),
        },
        {
            "id": "eval_online_ebe_all",
            "group": "Phase 4 — Multi-Dataset",
            "desc": "Eval online_ebe on humaneval,math500,mtbench,alpaca",
            "cmd": _ec(_merged("online-ebe-gsm8k"), "online_ebe",
                       datasets="humaneval,math500,mtbench,alpaca", task_score=True),
            "done_check": None,
            "smoke_skip": smoke,
            "requires": os.path.join(_merged("online-ebe-gsm8k"), "config.json"),
        },
        {
            "id": "eval_ebe_single_all",
            "group": "Phase 4 — Multi-Dataset",
            "desc": "Eval ebe_single on humaneval,math500,mtbench,alpaca",
            "cmd": _ec(_merged("ebe_single-gsm8k"), "ebe_single",
                       datasets="humaneval,math500,mtbench,alpaca", task_score=True),
            "done_check": None,
            "smoke_skip": smoke,
            "requires": os.path.join(_merged("ebe_single-gsm8k"), "config.json"),
        },
        {
            "id": "eval_online_ebe_single_all",
            "group": "Phase 4 — Multi-Dataset",
            "desc": "Eval online_ebe_single on humaneval,math500,mtbench,alpaca",
            "cmd": _ec(_merged("online-ebe-single-gsm8k"), "online_ebe_single",
                       datasets="humaneval,math500,mtbench,alpaca", task_score=True),
            "done_check": None,
            "smoke_skip": smoke,
            "requires": os.path.join(_merged("online-ebe-single-gsm8k"), "config.json"),
        },

        # Tree-loss multi-dataset eval — always uses all 3 non-OT verifiers so the
        # paper table shows bv/gbv/traversal cross-matrix for each tree loss.
        # kl_tree is the universal on-policy baseline in all columns.
        {
            "id": "eval_kl_tree_all",
            "group": "Phase 4 — Multi-Dataset",
            "desc": "Eval kl_tree on humaneval,math500,mtbench,alpaca [bv+gbv+traversal]",
            "cmd": _ec(_merged("kl_tree-gsm8k"), "kl_tree",
                       datasets="humaneval,math500,mtbench,alpaca", task_score=True,
                       modes=_TREE_NON_OT),
            "done_check": None,
            "smoke_skip": smoke,
            "requires": os.path.join(_merged("kl_tree-gsm8k"), "config.json"),
        },
        {
            "id": "eval_bv_tree_all",
            "group": "Phase 4 — Multi-Dataset",
            "desc": "Eval bv_tree on humaneval,math500,mtbench,alpaca [bv+gbv+traversal]",
            "cmd": _ec(_merged("bv_tree-gsm8k"), "bv_tree",
                       datasets="humaneval,math500,mtbench,alpaca", task_score=True,
                       modes=_TREE_NON_OT),
            "done_check": None,
            "smoke_skip": smoke,
            "requires": os.path.join(_merged("bv_tree-gsm8k"), "config.json"),
        },
        {
            "id": "eval_gbv_tree_all",
            "group": "Phase 4 — Multi-Dataset",
            "desc": "Eval gbv_tree on humaneval,math500,mtbench,alpaca [bv+gbv+traversal]",
            "cmd": _ec(_merged("gbv_tree-gsm8k"), "gbv_tree",
                       datasets="humaneval,math500,mtbench,alpaca", task_score=True,
                       modes=_TREE_NON_OT),
            "done_check": None,
            "smoke_skip": smoke,
            "requires": os.path.join(_merged("gbv_tree-gsm8k"), "config.json"),
        },
        {
            "id": "eval_trav_tree_all",
            "group": "Phase 4 — Multi-Dataset",
            "desc": "Eval trav_tree on humaneval,math500,mtbench,alpaca [bv+gbv+traversal]",
            "cmd": _ec(_merged("trav_tree-gsm8k"), "traversal_tree",
                       datasets="humaneval,math500,mtbench,alpaca", task_score=True,
                       modes=_TREE_NON_OT),
            "done_check": None,
            "smoke_skip": smoke,
            "requires": os.path.join(_merged("trav_tree-gsm8k"), "config.json"),
        },
        {
            "id": "eval_ebe_tree_all",
            "group": "Phase 4 — Multi-Dataset",
            "desc": "Eval ebe_tree on humaneval,math500,mtbench,alpaca [bv+gbv+traversal]",
            "cmd": _ec(_merged("ebe_tree-gsm8k"), "ebe_tree",
                       datasets="humaneval,math500,mtbench,alpaca", task_score=True,
                       modes=_TREE_NON_OT),
            "done_check": None,
            "smoke_skip": smoke,
            "requires": os.path.join(_merged("ebe_tree-gsm8k"), "config.json"),
        },
        {
            "id": "eval_rev_kl_tree_all",
            "group": "Phase 4 — Multi-Dataset",
            "desc": "Eval rev_kl_tree on humaneval,math500,mtbench,alpaca [bv+gbv+traversal]",
            "cmd": _ec(_merged("rev_kl_tree-gsm8k"), "rev_kl_tree",
                       datasets="humaneval,math500,mtbench,alpaca", task_score=True,
                       modes=_TREE_NON_OT),
            "done_check": None,
            "smoke_skip": smoke,
            "requires": os.path.join(_merged("rev_kl_tree-gsm8k"), "config.json"),
        },
        {
            "id": "eval_jsd_tree_all",
            "group": "Phase 4 — Multi-Dataset",
            "desc": "Eval jsd_tree on humaneval,math500,mtbench,alpaca [bv+gbv+traversal]",
            "cmd": _ec(_merged("jsd_tree-gsm8k"), "jsd_tree",
                       datasets="humaneval,math500,mtbench,alpaca", task_score=True,
                       modes=_TREE_NON_OT),
            "done_check": None,
            "smoke_skip": smoke,
            "requires": os.path.join(_merged("jsd_tree-gsm8k"), "config.json"),
        },
        {
            "id": "eval_online_kl_tree_all",
            "group": "Phase 4 — Multi-Dataset",
            "desc": "Eval online-kl-tree on humaneval,math500,mtbench,alpaca [all 6 verifiers]",
            "cmd": _ec(_merged("online-kl-tree-gsm8k"), "online_kl_tree",
                       datasets="humaneval,math500,mtbench,alpaca", task_score=True),
            "done_check": None,
            "smoke_skip": smoke,
            "requires": os.path.join(_merged("online-kl-tree-gsm8k"), "config.json"),
        },
        {
            "id": "eval_online_ebe_tree_all",
            "group": "Phase 4 — Multi-Dataset",
            "desc": "Eval online-ebe-tree on humaneval,math500,mtbench,alpaca [all 6 verifiers]",
            "cmd": _ec(_merged("online-ebe-tree-gsm8k"), "online_ebe_tree",
                       datasets="humaneval,math500,mtbench,alpaca", task_score=True),
            "done_check": None,
            "smoke_skip": smoke,
            "requires": os.path.join(_merged("online-ebe-tree-gsm8k"), "config.json"),
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
                    sys.executable, _EAGLE_SCRIPT, "gen",
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
                    sys.executable, _EAGLE_SCRIPT, "train",
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
                    sys.executable, _EAGLE_SCRIPT, "eval",
                    "--base", target,
                    "--ckpt", _ckpt("eagle-head"),
                    "--data", _data("gsm8k_30.jsonl"),
                    "--K",    "3",
                ],
                "done_check": None,  # writes to results.db; no single output file
            },
        ] if eagle else []),
    ]  # end _steps_list
    # Apply --losses filter: drop steps for unselected losses, keep all others.
    if losses_to_run is not None:
        _selected = set(losses_to_run)
        _steps_list = [
            s for s in _steps_list
            if not _is_loss_step(s["id"]) or _loss_selected(s["id"])
        ]
    return _steps_list

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
    # Retry loop: OneDrive on Windows holds a brief sync lock on newly-written
    # .tmp files, causing os.replace() to raise PermissionError (WinError 5).
    # Retry up to 5 times with exponential backoff; fall back to a direct
    # non-atomic write if all retries fail (still better than crashing).
    _tmp = STATE_FILE + ".tmp"
    with open(_tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    for _attempt in range(5):
        try:
            os.replace(_tmp, STATE_FILE)
            return
        except PermissionError as _pe:
            if _attempt == 4:
                # All retries exhausted — write directly (non-atomic fallback)
                print(f"  [state] WARNING: os.replace failed after 5 attempts "
                      f"({_pe}). Writing state file directly (non-atomic).")
                with open(STATE_FILE, "w", encoding="utf-8") as f:
                    json.dump(state, f, indent=2)
                try:
                    os.remove(_tmp)
                except OSError:
                    pass
            else:
                _delay = 0.3 * (2 ** _attempt)   # 0.3, 0.6, 1.2, 2.4 s
                print(f"  [state] PermissionError on state rename (OneDrive?), "
                      f"retry {_attempt + 1}/5 in {_delay:.1f}s…")
                time.sleep(_delay)


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

# ---------------------------------------------------------------------------
# Log Anomaly Scanner
# ---------------------------------------------------------------------------
# Runs automatically after every training step and again at pipeline end.
# Emits a clearly visible warning block when it finds anything suspicious.
# The full pipeline log is at _PIPELINE_LOG; per-step error snapshots are at
# db/logs/step_{sid}_error.log — both are scanned here.

_ANOMALY_PATTERNS = [
    # (regex_pattern, severity, label)
    (r"train_loss:\s*(nan|inf)",          "CRITICAL", "NaN/Inf loss"),
    (r"traceback \(most recent",          "CRITICAL", "Python traceback"),
    (r"cuda\s*(error|out of memory)",     "CRITICAL", "CUDA error"),
    (r"(sigkill|killed|exit code -9|exit code 247)", "CRITICAL", "OOM / killed"),
    (r"exit code 14[23]",                 "CRITICAL", "Killed (exit 143)"),
    (r"\[early stop\]",                   "WARNING",  "Early stopping triggered"),
    (r"ppl.*exceeded|ppl_threshold",      "WARNING",  "PPL threshold exceeded"),
    (r"(out of memory|oom)",              "WARNING",  "Out-of-memory hint"),
    (r"nan_count.*[1-9]",                 "WARNING",  "NaN count > 0"),
    (r"\[health\].*warn",                 "WARNING",  "Health check warning"),
    (r"loss.*spike|spike.*loss",          "WARNING",  "Loss spike"),
    (r"val loss.*not improved.*consecut", "INFO",     "Val not improving"),
    (r"no improvement.*streak",           "INFO",     "Improvement streak stalled"),
]

import re as _re_mod


def _scan_log_anomalies(log_path: str, tail_lines: int = 300) -> list[tuple[str, str, str]]:
    """
    Scan the tail of a log file for anomaly patterns.
    Returns a list of (severity, label, line) tuples, deduped.
    severity is one of: CRITICAL | WARNING | INFO
    """
    if not log_path or not os.path.exists(log_path):
        return []
    with open(log_path, encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()
    tail = lines[-tail_lines:]
    seen_labels: set[str] = set()
    results: list[tuple[str, str, str]] = []
    for raw in tail:
        lo = raw.lower().strip()
        for pattern, severity, label in _ANOMALY_PATTERNS:
            if label in seen_labels:
                continue
            if _re_mod.search(pattern, lo):
                results.append((severity, label, raw.rstrip()))
                seen_labels.add(label)
    # Sort: CRITICAL first, then WARNING, then INFO
    order = {"CRITICAL": 0, "WARNING": 1, "INFO": 2}
    results.sort(key=lambda x: order.get(x[0], 3))
    return results


def _print_anomaly_report(anomalies: list[tuple[str, str, str]],
                           step_id: str = "", header: str = "") -> None:
    """Pretty-print anomaly results. Noop if anomalies is empty."""
    if not anomalies:
        return
    criticals = [a for a in anomalies if a[0] == "CRITICAL"]
    warnings  = [a for a in anomalies if a[0] == "WARNING"]
    infos     = [a for a in anomalies if a[0] == "INFO"]

    sep = "!" * 65 if criticals else "-" * 65
    title = header or (f"Post-step health check: {step_id}" if step_id else "Pipeline health check")
    print(f"\n  {sep}")
    print(f"  ANOMALY REPORT — {title}")
    print(f"  {sep}")
    _COLOR = {"CRITICAL": "\033[91m", "WARNING": "\033[93m", "INFO": "\033[96m"}
    for sev, label, line in anomalies:
        color = _COLOR.get(sev, "")
        print(f"  {color}[{sev:8s}]{RESET} {label}")
        # Show the raw log line, truncated to 120 chars
        short = line[:120] + ("…" if len(line) > 120 else "")
        print(f"             {short}")
    print(f"  {sep}")
    if criticals:
        print(f"  \033[91m{len(criticals)} CRITICAL anomaly(s) — investigate before continuing.\033[0m")
    elif warnings:
        print(f"  \033[93m{len(warnings)} WARNING(s) — review before promoting to next tier.\033[0m")
    print()


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

# GBV/ is now reference-only — all eval logic uses algorithms/distillspec_gbv/verifiers/


def _cleanup_tmp(path):
    try:
        os.remove(path)
    except OSError:
        pass


def run_smoke_preflight(draft, target):
    """
    Run 2 gsm8k prompts through GBV/main.py directly — NO evaluate.py, NO DB writes.

    Why bypass evaluate.py?
      evaluate.py's --skip_existing checks (student_label, dataset, mode, K, T)
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
        print(f"  To always skip: python experiment.py --no_smoke_first\n")
        return True   # non-fatal

    # ── Write a 2-prompt temp file ─────────────────────────────────────────
    with open(data_file, encoding="utf-8") as fh:
        two_prompts = [ln for ln in fh if ln.strip()][:2]
    tmp_path = os.path.join(_raw, "_smoke_preflight_2.jsonl")
    with open(tmp_path, "w", encoding="utf-8") as fh:
        fh.writelines(two_prompts)

    _RUNNER = os.path.join(_GBV_RESEARCH, "algorithms", "distillspec_gbv", "verifiers", "runner.py")
    smoke_cmd = [
        sys.executable, "-u",
        _RUNNER,
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
    print(f"  Runs runner.py directly — zero DB writes, no skip_existing risk.")
    print(f"  Expected:  5-20 min on first run (CUDA kernel warm-up),")
    print(f"             1-3 min on subsequent runs (kernels already compiled).")
    print(f"  To skip:   python experiment.py --no_smoke_first")
    print(f"{'='*65}")

    env = {**os.environ,
           "TRANSFORMERS_OFFLINE": "1",
           "HF_HUB_OFFLINE":       "1",
           "HF_DATASETS_OFFLINE":  "1",
           "PYTHONIOENCODING":     "utf-8",
           "PYTHONUNBUFFERED":     "1"}

    t0 = time.time()
    PARENT = os.path.dirname(HERE)
    # stdout/stderr explicitly piped so the traceback appears in the pipeline
    # log even when experiment.py runs detached (DETACHED_PROCESS on Windows
    # does not reliably inherit file handles to grandchildren).
    proc = subprocess.Popen(
        smoke_cmd, cwd=PARENT, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        bufsize=1, universal_newlines=True,
    )
    _child_popen = proc
    _write_lock(child_pid=proc.pid)

    # Stream subprocess output directly to our stdout (which is the log file)
    if proc.stdout:
        for line in proc.stdout:
            print(line, end="", flush=True)
        proc.stdout.close()

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
        print(f"  Re-run with:  python experiment.py --no_smoke_first  to skip preflight.")
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
        print(f"    Offline    → model not cached; run scripts/setup_download.py first")
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
    # PYTHONPATH: ensure gbv-research/ is importable from every subprocess.
    # trainer.py does `from core.model_families import ...` and
    # `from algorithms.distillspec_gbv.losses import ...` — both require gbv-research/
    # to be on the Python path.  When run_step() launches a subprocess with
    # cwd=orchestration/, Python's sys.path[0] = the script's own directory
    # (e.g. algorithms/distillspec_gbv/), not gbv-research/ — so without this
    # PYTHONPATH fix those imports fail with ModuleNotFoundError on a fresh
    # Colab/Kaggle session (and on any machine where the user did not do
    # `pip install -e .` or manually set PYTHONPATH).
    _existing_pypath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        _GBV_RESEARCH_ROOT + os.pathsep + _existing_pypath
    ).rstrip(os.pathsep)
    # HF offline mode: DO NOT force offline here.
    # On a fresh Colab/Kaggle session models are not cached yet; forcing offline
    # would make the very first eval step fail with an HF connection error.
    # Each script manages its own offline preference:
    #   • evaluate.py  — never forces offline (downloads on first run, cache hit after)
    #   • trainer.py   — uses os.environ.setdefault("TRANSFORMERS_OFFLINE","1") so it
    #                    goes offline automatically once models are cached, but won't
    #                    block the first download if TRANSFORMERS_OFFLINE is not set.
    #   • online_serve.py — same as trainer.py
    # If you are running fully air-gapped, set TRANSFORMERS_OFFLINE=1 before launching
    # experiment.py and it will be inherited by all children via env = os.environ.copy().
    # Force UTF-8 I/O in every child Python process — prevents UnicodeEncodeError
    # when evaluate.py prints box-drawing characters on Windows cp1252 terminals.
    # Also needed for correct text handling on Kaggle (UTF-8 by default but explicit
    # is safer) and Colab (same).
    env["PYTHONIOENCODING"]  = "utf-8"
    env["PYTHONUNBUFFERED"]  = "1"   # flush every print() immediately — no more silent 5-min gaps
    # Propagate storage paths so every child (trainer, run_all, online_serve)
    # writes to the same DB and logs directory regardless of where it runs from.
    # These are already set on os.environ when --storage_root is active, so the
    # copy above already includes them — but set them explicitly here as insurance
    # (e.g., if a subprocess resets os.environ).
    for _ekey in ("SPECDIST_DB_PATH", "SPECDIST_LOGS_ROOT", "SPECDIST_STORAGE_ROOT"):
        if os.environ.get(_ekey):
            env[_ekey] = os.environ[_ekey]

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

        # Stream line-by-line → terminal AND log file simultaneously (tee).
        # INLINE CRITICAL detection: check every line as it arrives.
        # This fires immediately — no need to wait for step completion or pipeline end.
        # Only CRITICAL patterns are flagged here (NaN, OOM, traceback, SIGKILL).
        # WARNING/INFO patterns are caught in the post-step and end-of-pipeline sweeps.
        _inline_alerted: set[str] = set()
        for _line in iter(proc.stdout.readline, ""):
            sys.stdout.write(_line)
            sys.stdout.flush()
            _log_fh.write(_line)
            _log_fh.flush()
            # Check CRITICAL patterns on every line as it arrives
            _lo = _line.lower().strip()
            for _pat, _sev, _lbl in _ANOMALY_PATTERNS:
                if _sev == "CRITICAL" and _lbl not in _inline_alerted:
                    if _re_mod.search(_pat, _lo):
                        _inline_alerted.add(_lbl)
                        _alert = (f"\n  {'!'*60}\n"
                                  f"  LIVE ANOMALY [{_lbl}]\n"
                                  f"  {_line.rstrip()[:110]}\n"
                                  f"  {'!'*60}\n\n")
                        sys.stdout.write(_alert)
                        sys.stdout.flush()
                        _log_fh.write(_alert)
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
            # Post-step anomaly scan — only for training steps (they produce the big logs).
            # Eval steps are fast and their errors are obvious; training steps can
            # silently degrade (NaN loss, early stop, OOM-recovered) without a clear signal.
            if step.get("retryable", False):  # training steps set retryable=True
                _anomalies = _scan_log_anomalies(_PIPELINE_LOG, tail_lines=400)
                if _anomalies:
                    _print_anomaly_report(_anomalies, step_id=sid)
            return True
        else:
            mark_step(state, sid, "failed", f"rc={rc}")
            _hints = (
                "  Hint: check output above for [OOM] / [ERROR] / Traceback.\n"
                "  Common fixes:\n"
                "    OOM       -> evaluate.py retries on CPU automatically;\n"
                "                 trainer.py: add --no_lora or reduce --max_new_tokens\n"
                "    Offline   -> run scripts/setup_download.py first, or unset TRANSFORMERS_OFFLINE\n"
                "    Stale run -> python experiment.py --status\n"
                "  Restart   -> python experiment.py --config laptop --yes\n"
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
                      "evaluate.py will fall back to CPU automatically on OOM.")
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
    p.add_argument("--config", default="laptop",
                   help="Hardware config preset (laptop/server/colab) or a YAML profile name "
                        "relative to orchestration/configs/ "
                        "(e.g. profiles/colab_tree_losses, colab_lite, colab_a100).")
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
    p.add_argument("--storage_root", default=None,
                   help="Root directory for ALL persistent artifacts: checkpoints, "
                        "results DB, pipeline logs, and state file.  Prefer this over "
                        "--ckpt_root — it moves everything to one place so nothing is "
                        "silently left on ephemeral disk.  Sets SPECDIST_DB_PATH, "
                        "SPECDIST_LOGS_ROOT, and SPECDIST_STORAGE_ROOT env vars for "
                        "all child processes automatically.\n"
                        "  Colab :  /content/drive/MyDrive/specdist\n"
                        "  Modal :  /vol\n"
                        "  Kaggle:  /kaggle/working/specdist\n"
                        "  RunPod:  /workspace/specdist\n"
                        "  Local :  (omit — defaults to gbv-research/db/)")
    p.add_argument("--ckpt_root", default=None,
                   help="Persistent checkpoint directory only — use --storage_root "
                        "instead when you also want DB and logs on persistent storage. "
                        "Kept for backward compatibility.")
    # ── MLOps / sweep overrides ───────────────────────────────────────────────
    p.add_argument("--losses", default=None,
                   help="Comma-separated subset of losses to train/eval. "
                        "Default: run all (kl,ebe,ebe_single,rev_kl,jsd,l1,online,online_ebe,online_ebe_single). "
                        "Example: --losses ebe_single,online_ebe_single  runs only the EBE-single ablation. "
                        "Useful for running a single loss without the full pipeline.")
    p.add_argument("--eval_only", action="store_true",
                   help="Skip all Phase 2 training + merge steps. Only run Phase 1 baseline eval "
                        "and Phase 3/4 evals. Requires pre-built merged models in the checkpoint root. "
                        "Use after a baseline run to re-evaluate with different settings, or for the "
                        "verifier sweep profile. Overrides experiment.eval_only in YAML.")
    p.add_argument("--train_steps", type=int, default=None,
                   help="Override per-loss training step count. "
                        "Overrides the smoke (50) / full (1000) default. "
                        "Example: --train_steps 200 for a quick ablation run.")
    p.add_argument("--lr", type=float, default=None,
                   help="Learning rate for all training runs (overrides YAML config). "
                        "Default from YAML: laptop=3e-5, server=3e-5. "
                        "W&B sweep: set via sweep agent, not this flag directly.")
    p.add_argument("--lora_r", type=int, default=None,
                   help="LoRA rank for all training runs (overrides YAML config). "
                        "Default from YAML: laptop=8, server=16.")
    p.add_argument("--teacher_temp", type=float, default=None,
                   help="Teacher sampling temperature for all training runs "
                        "(overrides YAML config, default 0.8).")
    args = p.parse_args()

    # Load YAML config first — YAML profiles define models, hardware, training, etc.
    # Must happen before cfg is built so YAML models.draft/target are available.
    _yaml_cfg = _load_config_yaml(args.config)

    if args.config in CONFIGS:
        # Legacy preset (laptop / server / colab) — hardware baked into CONFIGS dict.
        # If the YAML also sets load_in_4bit, let it override the preset.
        cfg = dict(CONFIGS[args.config])
        if "load_in_4bit" in _yaml_cfg:
            cfg["load_in_4bit"] = _yaml_cfg["load_in_4bit"]
    else:
        # YAML-based profile (e.g. colab_lite, colab_a100, profiles/colab_tree_losses).
        # _load_config_yaml now exposes draft/target/load_in_4bit from the YAML.
        cfg = {
            "draft":        _yaml_cfg.get("draft",  "Qwen/Qwen3-0.6B"),
            "target":       _yaml_cfg.get("target", "Qwen/Qwen3-8B"),
            "load_in_4bit": _yaml_cfg.get("load_in_4bit", False),
            "desc":         f"YAML profile: {args.config}",
        }
    draft  = args.draft  or cfg["draft"]
    target = args.target or cfg["target"]

    # ── Storage root — single source of truth for all persistent paths ────────
    # --storage_root moves checkpoints, DB, logs, and state file to one directory
    # on persistent storage.  On ephemeral cloud (Colab/Kaggle/Modal) this MUST
    # point at persistent storage; on a local machine omit it entirely.
    #
    # Layout under storage_root:
    #   checkpoints/           trained LoRA adapters + merged models
    #   results.db             SQLite experiment database
    #   logs/                  pipeline_output.log + be_progress.log
    #   hf_cache/              (optional) HuggingFace model cache
    #   pipeline_state_*.json  crash-safe resume state
    #
    # Environment variables set for all child processes:
    #   SPECDIST_STORAGE_ROOT  the root itself (informational)
    #   SPECDIST_DB_PATH       full path to results.db
    #   SPECDIST_LOGS_ROOT     full path to logs/ directory
    global STATE_FILE, _DB_LOGS, _PIPELINE_LOG

    # Priority: CLI --storage_root > env var > YAML checkpointing.storage_root
    _storage_root = (args.storage_root
                     or os.environ.get("SPECDIST_STORAGE_ROOT")
                     or _yaml_cfg.get("storage_root"))

    if _storage_root:
        os.makedirs(_storage_root, exist_ok=True)
        # All persistent artifacts under the one root
        _effective_ckpt_root = args.ckpt_root or os.path.join(_storage_root, "checkpoints")
        _effective_db_path   = os.path.join(_storage_root, "results.db")
        _effective_logs_dir  = os.path.join(_storage_root, "logs")
        _effective_state_dir = _storage_root
        os.makedirs(_effective_logs_dir, exist_ok=True)
        # Propagate to all child processes via env vars
        os.environ["SPECDIST_STORAGE_ROOT"] = _storage_root
        os.environ["SPECDIST_DB_PATH"]       = _effective_db_path
        os.environ["SPECDIST_LOGS_ROOT"]     = _effective_logs_dir
        # Redirect module-level log paths so this process also writes there
        _DB_LOGS      = _effective_logs_dir
        _PIPELINE_LOG = os.path.join(_effective_logs_dir, "pipeline_output.log")
        print(f"  [storage] root       : {_storage_root}")
        print(f"  [storage] checkpoints: {_effective_ckpt_root}")
        print(f"  [storage] database   : {_effective_db_path}")
        print(f"  [storage] logs       : {_effective_logs_dir}")
    else:
        # Local dev defaults — existing behavior, nothing changes
        _effective_ckpt_root = args.ckpt_root   # may still be None (uses db/checkpoints/)
        _effective_db_path   = None             # results_db.py uses its own default
        _effective_logs_dir  = _DB_LOGS
        _effective_state_dir = HERE

    # Build training hyperparams dict — CLI overrides take priority over YAML values.
    train_hparams = {
        "lr":           args.lr           or _yaml_cfg.get("lr", 3e-5),
        "lora_r":       args.lora_r       or _yaml_cfg.get("lora_r", 8),
        "lora_alpha":                         _yaml_cfg.get("lora_alpha", 16),
        "teacher_temp": args.teacher_temp or _yaml_cfg.get("teacher_temp", 0.8),
        # CLI --train_steps > YAML training.steps > None (smoke/full default)
        "train_steps":  args.train_steps or _yaml_cfg.get("train_steps"),
        "max_new_tokens": _yaml_cfg.get("max_new_tokens", 80),
        # hardware.compile → passed to all training subprocesses as --compile.
        # False by default (safe on Windows).  server.yaml sets True for A100.
        "compile":        _yaml_cfg.get("compile", False),
        # online_lr: LR for forward_kl online adapt.  Defaults to 3e-4 (10x
        # offline LR) — online KL adaptation requires a larger step to move the draft
        # meaningfully within the short 500-prompt online budget.
        "online_lr":      _yaml_cfg.get("online_lr", 3e-4),
        # online_ebe_lr: separate LR for EBE online adapt.  Must be lower than
        # online_lr because EBE's cumprod gradient is more volatile.  Defaults to
        # 1e-4 (3x lower than online_lr).  Set in laptop/server/colab YAML if needed.
        "online_ebe_lr":  _yaml_cfg.get("online_ebe_lr", 1e-4),
    }

    # Parse --losses filter into a list; None = run all losses.
    # Loss filter: --losses CLI wins; fall back to experiment.losses in YAML; then all.
    _yaml_losses = cfg.get("losses")   # list or None (from experiment: losses: [...])
    _cli_losses  = (
        [l.strip() for l in args.losses.split(",") if l.strip()]
        if args.losses else None
    )
    _losses_to_run = _cli_losses or (_yaml_losses if isinstance(_yaml_losses, list) else None)
    if _losses_to_run:
        _invalid = [l for l in _losses_to_run if l not in ALL_LOSSES]
        if _invalid:
            p.error(f"--losses: unknown loss name(s): {_invalid}. "
                    f"Valid: {ALL_LOSSES}")
        _src = "CLI --losses" if _cli_losses else f"{args.config}.yaml experiment.losses"
        print(f"  [pipeline] Loss filter ({_src}): {_losses_to_run}")
        print(f"             Skipping: {[l for l in ALL_LOSSES if l not in _losses_to_run]}")

    # State file: config-scoped so laptop and server runs don't mix.
    # Smoke gets its OWN state file so smoke "done" marks never block the real run.
    # When --storage_root is set, state file lives there (survives cloud restarts).
    _smoke_tag = "_smoke" if args.smoke else ""
    STATE_FILE = os.path.join(
        _effective_state_dir, f"pipeline_state_{args.config}{_smoke_tag}.json"
    )

    # eval_only: CLI --eval_only wins over YAML experiment.eval_only.
    _eval_only = args.eval_only or cfg.get("eval_only", False)

    # --status and --dry_run are read-only: build steps + print plan, then exit.
    # Do NOT acquire the lock (that kills any running pipeline process!).
    _load_4bit = cfg.get("load_in_4bit", False)
    if args.status or args.dry_run:
        STEPS = build_steps(draft, target, experiment_tag=args.experiment_tag,
                            smoke=args.smoke, eagle=args.eagle,
                            load_in_4bit=_load_4bit,
                            ckpt_root=_effective_ckpt_root,
                            train_hparams=train_hparams,
                            losses_to_run=_losses_to_run)
        if _eval_only:
            _n_before = len(STEPS)
            STEPS = [s for s in STEPS if "Phase 2" not in s.get("group", "")]
            print(f"  [pipeline] eval_only — skipped {_n_before - len(STEPS)} Phase 2 "
                  f"train/merge steps (requires pre-built merged models)")
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
                        load_in_4bit=_load_4bit,
                        ckpt_root=_effective_ckpt_root,  # was args.ckpt_root — ignored --storage_root
                        train_hparams=train_hparams,
                        losses_to_run=_losses_to_run)

    if _eval_only:
        _n_before = len(STEPS)
        STEPS = [s for s in STEPS if "Phase 2" not in s.get("group", "")]
        print(f"  [pipeline] eval_only — skipped {_n_before - len(STEPS)} Phase 2 "
              f"train/merge steps (using pre-built merged models)")

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
    # Bypasses evaluate.py so zero DB rows are written → no skip_existing risk.
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

        # Hard prereq guard: eval steps cannot run until their train+merge steps finish.
        # Unlike skip_if_missing (silent auto-skip), this is a loud failure — the user
        # must explicitly run the corresponding train step before eval can proceed.
        requires = step.get("requires")
        if requires and not os.path.exists(requires):
            _req_model = os.path.basename(os.path.dirname(requires))
            _train_id  = "merge_" + _req_model.replace("-", "_")
            print(f"\n  {CROSS} [{sid}] BLOCKED — merged model not found.")
            print(f"         Expected : {requires}")
            print(f"         Fix      : run the train + merge steps for '{_req_model}' first.")
            print(f"         Hint     : re-run with --from {_train_id}")
            mark_step(state, sid, "failed", f"blocked: merged model missing: {_req_model}")
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
            print(f"\n{CROSS} Step '{sid}' failed. Fix the issue then re-run experiment.py")
            print(f"   The pipeline will skip completed steps and retry from '{sid}'.")
            print(f"   Restart: python experiment.py --config {args.config} --yes")
            sys.exit(1)

    # ── End-of-pipeline anomaly sweep ──────────────────────────────────────────
    # Scan the full log (not just tail) so we catch anything from any step.
    # This is the last thing the user sees — make it actionable.
    _final_anomalies = _scan_log_anomalies(_PIPELINE_LOG, tail_lines=2000)
    _print_anomaly_report(
        _final_anomalies,
        header=f"Full pipeline — {n_run} step(s) complete",
    )
    # Also write a persistent anomaly report next to the pipeline log.
    if _final_anomalies:
        _anom_path = os.path.join(_DB_LOGS, "anomaly_report.txt")
        try:
            with open(_anom_path, "w", encoding="utf-8") as _af:
                _af.write(f"Anomaly report — {datetime.now().isoformat()}\n")
                _af.write(f"Pipeline log   : {_PIPELINE_LOG}\n\n")
                for sev, label, line in _final_anomalies:
                    _af.write(f"[{sev:8s}] {label}\n  {line}\n\n")
            print(f"  Anomaly report saved: {_anom_path}")
        except OSError:
            pass

    print(f"\n{'='*65}")
    print(f"  {TICK} Pipeline complete! {n_run} step(s) executed.")
    print(f"  View results: python dashboard/training_dashboard.py")
    if _final_anomalies:
        print(f"  \033[93mReview anomaly report before promoting to next tier.\033[0m")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    _reconfigure_stdout_for_windows()
    main()
