"""
clean_restart.py — Wipe all pipeline outputs and restart from scratch.

Usage:
    python orchestration/clean_restart.py                   # laptop config
    python orchestration/clean_restart.py --config kaggle   # any config
    python orchestration/clean_restart.py --config laptop_gpt2
    python orchestration/clean_restart.py --dry_run         # show what would be deleted
    python orchestration/clean_restart.py --no_restart      # wipe only, don't relaunch

What it wipes:
    db/checkpoints/   — all trained LoRA adapters and merged models
    db/logs/          — pipeline_output.log, be_progress.log, step error logs
    db/wandb/         — local WandB run dirs
    db/results.db     — evaluation results database
    OSD/checkpoints/  — legacy OSD checkpoint dir (migration period)
    OSD/wandb/        — legacy OSD WandB run dirs
    orchestration/wandb/ — stale WandB runs written before db/wandb fix

What it keeps:
    db/.gitkeep, db/logs/.gitkeep   — directory markers
    All source code, configs, datasets

Pipeline state reset:
    The state file for the chosen config (and its smoke variant) is reset to
    all-pending.  The step IDs are read from the existing state file when
    present; otherwise the file is simply deleted so experiment.py creates it
    fresh on the next run.

Accepts ANY --config value that experiment.py accepts (laptop, kaggle,
colab, a100, laptop_gpt2, laptop_llama, server_gpt2, profiles/*, ...).
"""

import argparse
import json
import os
import shutil
import stat
import subprocess
import sys
import time

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
                        if pid != my_pid:
                            python_pids.append(pid)
                    except ValueError:
                        pass
    except Exception:
        # Linux / non-Windows
        try:
            import psutil
            python_pids = [p.pid for p in psutil.process_iter(["pid", "name"])
                           if "python" in (p.info["name"] or "").lower()
                           and p.pid != my_pid]
        except ImportError:
            print("  [warn] Could not list processes — skipping process kill")
            return

    if not python_pids:
        print("  No Python processes found")
        return

    print(f"  Found {len(python_pids)} Python process(es): {python_pids}")
    if dry_run:
        print("  [dry_run] Would kill:", python_pids)
        return

    if sys.platform == "win32":
        # Kill wandb background services first so they release file locks
        for svc in ["wandb-core", "wandb-xpu"]:
            subprocess.run(
                ["powershell", "-Command",
                 f"Stop-Process -Name '{svc}' -Force -ErrorAction SilentlyContinue"],
                capture_output=True
            )
        subprocess.run(
            ["powershell", "-Command",
             f"Stop-Process -Id {','.join(str(p) for p in python_pids)} -Force "
             f"-ErrorAction SilentlyContinue"],
            capture_output=True
        )
    else:
        import signal
        for pid in python_pids:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    time.sleep(2)
    print("  All pipeline processes terminated [ok]")


# ── Wipe files / directories ───────────────────────────────────────────────

def _force_remove(path):
    def _onerror(func, fpath, excinfo):
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
            for entry in os.listdir(path):
                if entry == ".gitkeep":
                    continue
                _force_remove(os.path.join(path, entry))
            if not keep_dir:
                try:
                    os.rmdir(path)
                except Exception:
                    pass
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

def _state_path(config_slug: str) -> str:
    """Return the state file path for any config slug.

    Config slugs with slashes (e.g. 'profiles/kl_only') are flattened
    to underscores so the filename stays valid on all platforms.
    """
    slug = config_slug.replace("/", "_").replace("\\", "_")
    return os.path.join(_HERE, f"pipeline_state_{slug}.json")


def _reset_state(config_slug: str, dry_run=False):
    """Reset a state file to all-pending.

    If the state file already exists, reads its step IDs and resets them
    all to 'pending'.  If it doesn't exist, deletes nothing — experiment.py
    will create it fresh on the next run.
    """
    path = _state_path(config_slug)
    if dry_run:
        print(f"  [dry_run] Would reset {os.path.basename(path)} -> all steps pending")
        return
    if not os.path.exists(path):
        print(f"  [skip] {os.path.basename(path)} not found — will be created by experiment.py")
        return

    try:
        with open(path) as f:
            data = json.load(f)
        step_ids = list(data.get("steps", {}).keys())
        fresh = {
            "version": data.get("version", 1),
            "config": config_slug,
            "steps": {sid: {"status": "pending"} for sid in step_ids},
        }
    except Exception:
        # Corrupted state file — just remove it
        os.remove(path)
        print(f"  [del]   {os.path.basename(path)} (corrupted) -> will be recreated [ok]")
        return

    with open(path, "w") as f:
        json.dump(fresh, f, indent=2)
    print(f"  [reset] {os.path.basename(path)} -> all steps pending [ok]")


# ── Launch pipeline ────────────────────────────────────────────────────────

def _launch(config: str, extra_args: list, dry_run=False):
    pipeline_script = os.path.join(_HERE, "experiment.py")
    log_path = os.path.join(_GBV_RESEARCH, "db", "logs", "pipeline_output.log")
    cmd = [sys.executable, pipeline_script, "--config", config, "--yes"] + extra_args
    print(f"\n  Launching: {' '.join(os.path.basename(p) if os.sep in p else p for p in cmd)}")
    print(f"  Log -> db/logs/pipeline_output.log")
    if dry_run:
        print("  [dry_run] Would launch pipeline")
        return
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as log_f:
        kwargs = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.DETACHED_PROCESS
        proc = subprocess.Popen(
            cmd, cwd=_GBV_RESEARCH,
            stdout=log_f, stderr=subprocess.STDOUT,
            **kwargs
        )
    print(f"  Pipeline started (PID {proc.pid}) [ok]")
    print(f"\n  Watch progress:  tail -f db/logs/pipeline_output.log")
    print(f"  Dashboard:       http://127.0.0.1:5000")
    print(f"  Status:          python orchestration/experiment.py --status")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Wipe all pipeline outputs and restart clean.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python orchestration/clean_restart.py                    # laptop (default)
  python orchestration/clean_restart.py --config kaggle
  python orchestration/clean_restart.py --config laptop_gpt2
  python orchestration/clean_restart.py --config laptop_gpt2 --device cpu
  python orchestration/clean_restart.py --config profiles/kl_only
  python orchestration/clean_restart.py --no_restart       # wipe only
        """)
    p.add_argument("--config", default="laptop",
                   help="Pipeline config (any value accepted by experiment.py, "
                        "e.g. laptop, kaggle, laptop_gpt2, profiles/kl_only). "
                        "Default: laptop")
    p.add_argument("--dry_run", action="store_true",
                   help="Show what would be done, make no changes")
    p.add_argument("--no_restart", action="store_true",
                   help="Wipe only — do not relaunch the pipeline")
    # Pass-through args forwarded to experiment.py (e.g. --device cpu)
    p.add_argument("--device", default=None,
                   help="Forwarded to experiment.py (e.g. --device cpu)")
    p.add_argument("--losses", default=None,
                   help="Forwarded to experiment.py (e.g. --losses kl,bv_tree)")
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
    _reset_state(f"{args.config}_smoke", dry_run=dry)

    if not args.no_restart:
        print("\n4. Launching pipeline...")
        extra = []
        if args.device:
            extra += ["--device", args.device]
        if args.losses:
            extra += ["--losses", args.losses]
        _launch(args.config, extra, dry_run=dry)

    print(f"\n{'='*60}")
    print(f"  Done{tag}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
