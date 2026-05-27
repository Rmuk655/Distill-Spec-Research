"""
clean_restart.py — Wipe all pipeline outputs and restart from scratch.

Usage:
    python orchestration/clean_restart.py              # wipe + restart
    python orchestration/clean_restart.py --dry_run    # show what would be deleted
    python orchestration/clean_restart.py --no_restart # wipe only, don't relaunch
    python orchestration/clean_restart.py --config server  # use server config

What it wipes:
    db/checkpoints/   — all trained LoRA adapters and merged models
    db/logs/          — pipeline_output.log, be_progress.log, step error logs
    db/wandb/         — local WandB run dirs
    db/results.db     — evaluation results database
    OSD/checkpoints/  — legacy OSD checkpoint dir (same data, migration period)
    OSD/wandb/        — legacy OSD WandB run dirs
    orchestration/wandb/ — stale WandB runs written before db/wandb fix

What it keeps:
    db/.gitkeep, db/logs/.gitkeep   — directory markers
    orchestration/pipeline_state_*.json  — reset to all-pending (not deleted)
    All source code, configs, datasets

Pipeline state reset:
    All steps set to 'pending'. Steps with skip_if_missing will auto-skip
    if their prerequisite checkpoint doesn't exist (LR-sweep, pilot evals).
"""

import argparse
import json
import os
import shutil
import stat
import subprocess
import sys
import time


def _force_remove(path, desc=""):
    """Remove a file or directory tree, stripping read-only flags on Windows."""
    def _onerror(func, fpath, excinfo):
        # Strip read-only flag and retry once
        try:
            os.chmod(fpath, stat.S_IWRITE)
            func(fpath)
        except Exception as e2:
            print(f"  [warn] Could not remove {fpath}: {e2}")
    try:
        if os.path.isdir(path):
            shutil.rmtree(path, onerror=_onerror)
        else:
            try:
                os.remove(path)
            except PermissionError:
                os.chmod(path, stat.S_IWRITE)
                os.remove(path)
    except Exception as e:
        print(f"  [warn] Could not remove {path}: {e}")

_HERE         = os.path.dirname(os.path.abspath(__file__))
_GBV_RESEARCH = os.path.dirname(_HERE)
_SUMMER_DIR   = os.path.dirname(_GBV_RESEARCH)
_OSD_DIR      = os.path.join(_SUMMER_DIR, "OSD")


# ── Directories / files to wipe ────────────────────────────────────────────

def _wipe_targets():
    db = os.path.join(_GBV_RESEARCH, "db")
    return [
        # (path, keep_dir, description)
        (os.path.join(db, "checkpoints"),                True,  "db/checkpoints (trained models)"),
        (os.path.join(db, "logs"),                       True,  "db/logs (pipeline + step logs)"),
        (os.path.join(db, "wandb"),                      True,  "db/wandb (local WandB runs)"),
        (os.path.join(db, "results.db"),                 False, "db/results.db (eval results)"),
        (os.path.join(_OSD_DIR, "checkpoints"),          True,  "OSD/checkpoints (legacy)"),
        (os.path.join(_OSD_DIR, "wandb"),                False, "OSD/wandb (legacy WandB runs)"),
        (os.path.join(_OSD_DIR, "results.db"),           False, "OSD/results.db (legacy eval results)"),
        (os.path.join(_HERE, "wandb"),                   False, "orchestration/wandb (stale WandB)"),
    ]


# ── Kill running pipeline / model processes ────────────────────────────────

def _kill_pipeline_processes(dry_run=False):
    """Kill any Python processes that are part of this pipeline."""
    my_pid = os.getpid()
    try:
        result = subprocess.run(
            ["tasklist", "/FO", "CSV", "/NH"],
            capture_output=True, text=True, timeout=10
        )
        python_pids = []
        for line in result.stdout.splitlines():
            if "python.exe" in line.lower():
                parts = line.strip('"').split('","')
                if len(parts) >= 2:
                    try:
                        pid = int(parts[1])
                        if pid != my_pid:   # never kill ourselves
                            python_pids.append(pid)
                    except ValueError:
                        pass
    except Exception:
        print("  [warn] Could not list processes via tasklist — skipping process kill")
        return

    if not python_pids:
        print("  No Python processes found")
        return

    print(f"  Found {len(python_pids)} Python process(es): {python_pids}")
    if dry_run:
        print("  [dry_run] Would kill:", python_pids)
        return

    # Kill wandb background services first so they release file locks
    for svc in ["wandb-core", "wandb-xpu"]:
        subprocess.run(
            ["powershell", "-Command",
             f"Stop-Process -Name '{svc}' -Force -ErrorAction SilentlyContinue"],
            capture_output=True
        )

    # Kill Python processes (pipeline, models, dashboard)
    subprocess.run(
        ["powershell", "-Command",
         f"Stop-Process -Id {','.join(str(p) for p in python_pids)} -Force -ErrorAction SilentlyContinue"],
        capture_output=True
    )
    time.sleep(2)

    # Confirm
    result2 = subprocess.run(["tasklist"], capture_output=True, text=True)
    remaining = [p for p in python_pids if str(p) in result2.stdout]
    if remaining:
        print(f"  [warn] {len(remaining)} process(es) may still be running: {remaining}")
    else:
        print("  All pipeline processes terminated [ok]")


# ── Wipe files / directories ───────────────────────────────────────────────

def _wipe(dry_run=False):
    for path, keep_dir, desc in _wipe_targets():
        if not os.path.exists(path):
            print(f"  [skip] {desc} — not found")
            continue

        if dry_run:
            print(f"  [dry_run] Would wipe: {desc}")
            continue

        if os.path.isfile(path):
            _force_remove(path)
            print(f"  [del]  {desc} [ok]")
        else:
            # Wipe contents but optionally keep the directory itself
            for entry in os.listdir(path):
                if entry == ".gitkeep":
                    continue
                entry_path = os.path.join(path, entry)
                _force_remove(entry_path)
            if not keep_dir:
                try:
                    os.rmdir(path)
                except Exception:
                    pass  # not empty (gitkeep or leftover), leave it
            print(f"  [wipe] {desc} [ok]")

    # Recreate empty dirs with .gitkeep so git doesn't lose them
    for keep_path in [
        os.path.join(_GBV_RESEARCH, "db", "checkpoints"),
        os.path.join(_GBV_RESEARCH, "db", "logs"),
        os.path.join(_GBV_RESEARCH, "db", "wandb"),
        os.path.join(_OSD_DIR, "checkpoints"),
    ]:
        os.makedirs(keep_path, exist_ok=True)


# ── Reset pipeline state ───────────────────────────────────────────────────

_STATE_FILES = {
    "laptop":        os.path.join(_HERE, "pipeline_state_laptop.json"),
    "laptop_smoke":  os.path.join(_HERE, "pipeline_state_laptop_smoke.json"),
    "server":        os.path.join(_HERE, "pipeline_state_server.json"),
    "server_smoke":  os.path.join(_HERE, "pipeline_state_server_smoke.json"),
}

_ALL_STEP_IDS = [
    # Phase 1 — Baseline
    "eval_baseline_gsm8k",
    # Phase 2 — Training (6 losses)
    "train_kl_gsm8k",     "merge_kl_gsm8k",
    "train_ebe_gsm8k",    "merge_ebe_gsm8k",
    "train_rev_kl_gsm8k", "merge_rev_kl_gsm8k",
    "train_jsd_gsm8k",    "merge_jsd_gsm8k",
    "train_l1_gsm8k",     "merge_l1_gsm8k",
    "online_adapt_gsm8k", "merge_online_gsm8k",
    # Phase 3 — GSM8K Eval
    "eval_kl_gsm8k", "eval_ebe_gsm8k",
    "eval_rev_kl_gsm8k", "eval_jsd_gsm8k",
    "eval_l1_gsm8k", "eval_online_gsm8k",
    # Phase 4 — Multi-Dataset Eval
    "eval_baseline_all", "eval_kl_all",
    "eval_ebe_all", "eval_rev_kl_all",
    "eval_jsd_all", "eval_l1_all", "eval_online_all",
    # Phase 5 — EAGLE Benchmark (optional, --eagle only)
    "eagle_gen", "eagle_train", "eagle_eval",
]


def _reset_state(config, dry_run=False):
    state_path = _STATE_FILES.get(config)
    if not state_path:
        print(f"  [warn] Unknown config '{config}' — no state file to reset")
        return
    if dry_run:
        print(f"  [dry_run] Would reset {os.path.basename(state_path)} to all-pending")
        return

    fresh = {
        "version": 1,
        "config": config,
        "steps": {sid: {"status": "pending"} for sid in _ALL_STEP_IDS},
    }
    with open(state_path, "w") as f:
        json.dump(fresh, f, indent=2)
    print(f"  [reset] {os.path.basename(state_path)} -> all steps pending [ok]")


# ── Launch pipeline ────────────────────────────────────────────────────────

def _launch(config, dry_run=False):
    pipeline_script = os.path.join(_HERE, "experiment.py")
    log_path = os.path.join(_GBV_RESEARCH, "db", "logs", "pipeline_output.log")
    cmd = [sys.executable, pipeline_script, "--config", config, "--yes"]
    print(f"\n  Launching: {' '.join(os.path.basename(p) for p in cmd)}")
    print(f"  Log -> db/logs/pipeline_output.log")
    if dry_run:
        print("  [dry_run] Would launch pipeline")
        return
    with open(log_path, "w") as log_f:
        proc = subprocess.Popen(
            cmd,
            cwd=_GBV_RESEARCH,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.DETACHED_PROCESS if sys.platform == "win32" else 0,
        )
    print(f"  Pipeline started (PID {proc.pid}) [ok]")
    print(f"\n  Watch progress:  tail -f db/logs/pipeline_output.log")
    print(f"  Dashboard:       http://127.0.0.1:5000")
    print(f"  Status:          python orchestration/experiment.py --status")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Wipe all pipeline outputs and restart clean.")
    p.add_argument("--config", default="laptop", choices=["laptop", "server"],
                   help="Pipeline config (default: laptop)")
    p.add_argument("--dry_run", action="store_true",
                   help="Show what would be done, make no changes")
    p.add_argument("--no_restart", action="store_true",
                   help="Wipe only — do not relaunch the pipeline")
    args = p.parse_args()

    dry = args.dry_run
    tag = " [DRY RUN]" if dry else ""

    print(f"\n{'='*60}")
    print(f"  GBV Clean Restart{tag}")
    print(f"  Config: {args.config}")
    print(f"{'='*60}\n")

    print("1. Killing running pipeline processes...")
    _kill_pipeline_processes(dry_run=dry)

    print("\n2. Wiping outputs...")
    _wipe(dry_run=dry)

    print("\n3. Resetting pipeline state...")
    _reset_state(args.config, dry_run=dry)
    _reset_state(f"{args.config}_smoke", dry_run=dry)  # smoke state is kept separate

    if not args.no_restart:
        print("\n4. Launching pipeline...")
        _launch(args.config, dry_run=dry)

    print(f"\n{'='*60}")
    print(f"  Done{tag}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
