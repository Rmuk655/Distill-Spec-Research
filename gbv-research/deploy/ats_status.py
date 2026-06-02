#!/usr/bin/env python3
"""
ATS Cloud Status Monitor
========================
Check the progress of parallel training jobs launched by ats_train.py.
Safe to run at any time — read-only, never touches running processes.

Usage
-----
  python deploy/ats_status.py              # snapshot of all losses
  python deploy/ats_status.py --watch      # refresh every 30 s (Ctrl+C to stop)
  python deploy/ats_status.py --tail kl   # live-tail one loss log
  python deploy/ats_status.py --log kl    # print last 40 lines of kl log
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime

HERE     = os.path.dirname(os.path.abspath(__file__))
ROOT     = os.path.dirname(HERE)
LOG_DIR  = os.path.join(ROOT, "db", "logs", "ats")
PID_FILE = os.path.join(ROOT, "db", "logs", "ats_pids.json")

STEP_RE  = re.compile(
    r"Step\s+(\d+)/(\d+)\s*\|\s*train_loss:\s*([\d.eE+\-]+)"
)
VAL_RE   = re.compile(
    r"Step\s+(\d+)/(\d+)\s*\|\s*val_loss:\s*([\d.eE+\-]+)"
)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False


def _parse_log(log_path: str):
    """Parse log file → latest step info."""
    if not os.path.exists(log_path):
        return None
    step, total, loss, val_loss = 0, 0, None, None
    try:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        for line in reversed(lines):
            if val_loss is None:
                vm = VAL_RE.search(line)
                if vm:
                    val_loss = float(vm.group(3))
            if step == 0:
                sm = STEP_RE.search(line)
                if sm:
                    step  = int(sm.group(1))
                    total = int(sm.group(2))
                    loss  = float(sm.group(3))
            if step > 0 and val_loss is not None:
                break
    except Exception:
        pass
    return {"step": step, "total": total, "loss": loss, "val_loss": val_loss,
            "lines": len(lines) if "lines" in dir() else 0}


def _bar(done: int, total: int, width: int = 20) -> str:
    if total == 0:
        return "?" * width
    filled = int(width * done / total)
    return "█" * filled + "░" * (width - filled)


def _elapsed(started_at: str) -> str:
    try:
        start = datetime.fromisoformat(started_at)
        delta = datetime.now() - start
        h, rem = divmod(int(delta.total_seconds()), 3600)
        m, s   = divmod(rem, 60)
        return f"{h}h {m:02d}m {s:02d}s"
    except Exception:
        return "?"


def show_status(pid_data: dict):
    """Print a progress table for all running losses."""
    family  = pid_data.get("family", "?")
    steps   = pid_data.get("steps", 0)
    started = pid_data.get("started_at", "")
    losses  = pid_data.get("losses", {})  # loss → pid

    alive, done, failed = [], [], []
    rows = []

    for loss, pid in losses.items():
        log_path = os.path.join(LOG_DIR, f"{family}_{loss}.log")
        info     = _parse_log(log_path) or {}
        cur_step = info.get("step", 0)
        tot_step = info.get("total", steps)
        cur_loss = info.get("loss")
        val_loss = info.get("val_loss")
        is_alive = _pid_alive(pid)

        if is_alive:
            alive.append(loss)
            state = "RUNNING"
        elif cur_step >= tot_step and tot_step > 0:
            done.append(loss)
            state = "DONE   "
        elif cur_step > 0:
            done.append(loss)
            state = "DONE   "
        else:
            failed.append(loss)
            state = "FAILED "

        pct   = (cur_step / tot_step * 100) if tot_step else 0
        bar   = _bar(cur_step, tot_step)
        loss_str = f"{cur_loss:.4f}" if cur_loss is not None else "   ?"
        val_str  = f"{val_loss:.4f}" if val_loss is not None else "   ?"
        rows.append((state, loss, bar, pct, cur_step, tot_step,
                     loss_str, val_str, pid))

    print()
    print(f"  ATS Parallel Training — {family.upper()}   elapsed: {_elapsed(started)}")
    print(f"  {len(alive)} running  |  {len(done)} done  |  {len(failed)} failed  "
          f"of {len(losses)} total")
    print()
    print(f"  {'STATUS':8s} {'LOSS':20s} {'PROGRESS':22s} "
          f"{'STEP':>10s} {'TRAIN_LOSS':>10s} {'VAL_LOSS':>10s}  PID")
    print("  " + "-" * 90)

    for state, loss, bar, pct, cur, tot, tloss, vloss, pid in rows:
        print(f"  {state:8s} {loss:20s} {bar} {pct:5.1f}%  "
              f"{cur:4d}/{tot:<4d}  {tloss:>10s}  {vloss:>10s}  {pid}")

    print()
    if alive:
        first = losses[alive[0]] if alive else 0
        sample_log = os.path.join(LOG_DIR, f"{family}_{alive[0]}.log")
        print(f"  Tail log: python deploy/ats_status.py --tail {alive[0]}")
    if done and not alive:
        print("  All losses complete. Next step:")
        print("    python orchestration/experiment.py --config server_gpt2 "
              "--eval_only --yes")


def main():
    ap = argparse.ArgumentParser(description="ATS Cloud training status monitor.")
    ap.add_argument("--watch",   action="store_true",
                    help="Refresh every 30 s (Ctrl+C to stop).")
    ap.add_argument("--interval", type=int, default=30,
                    help="Refresh interval in seconds (default: 30).")
    ap.add_argument("--tail",    default=None, metavar="LOSS",
                    help="Live-tail log for one loss (e.g. --tail kl).")
    ap.add_argument("--log",     default=None, metavar="LOSS",
                    help="Print last 40 lines of one loss log.")
    ap.add_argument("--family",  default=None,
                    help="Override family (auto-detected from PID file).")
    args = ap.parse_args()

    # ── Tail / log mode ───────────────────────────────────────────────────────
    if args.tail or args.log:
        loss    = args.tail or args.log
        family  = args.family or "gpt2"
        # Try to read family from PID file
        if os.path.exists(PID_FILE):
            with open(PID_FILE) as f:
                family = json.load(f).get("family", family)
        log_path = os.path.join(LOG_DIR, f"{family}_{loss}.log")
        if not os.path.exists(log_path):
            print(f"Log not found: {log_path}")
            sys.exit(1)
        if args.tail:
            print(f"Live-tailing {log_path}  (Ctrl+C to stop)")
            subprocess.run(["tail", "-f", log_path])
        else:
            subprocess.run(["tail", "-n", "40", log_path])
        return

    # ── Status mode ───────────────────────────────────────────────────────────
    if not os.path.exists(PID_FILE):
        print("No running jobs found. Start training with:")
        print("  python deploy/ats_train.py")
        sys.exit(0)

    with open(PID_FILE) as f:
        pid_data = json.load(f)

    if args.family:
        pid_data["family"] = args.family

    if args.watch:
        try:
            while True:
                os.system("clear" if os.name != "nt" else "cls")
                show_status(pid_data)
                print(f"  Refreshing every {args.interval}s ... (Ctrl+C to stop)")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nMonitor stopped.")
    else:
        show_status(pid_data)


if __name__ == "__main__":
    main()
