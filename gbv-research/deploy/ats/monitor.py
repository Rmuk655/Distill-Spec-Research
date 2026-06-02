#!/usr/bin/env python3
"""
monitor.py — Live progress monitor for parallel training (05_train_parallel.py).
Safe to run at any time from any SSH session — read-only.

Usage:
    python deploy/ats/monitor.py                    # snapshot table
    python deploy/ats/monitor.py --watch            # refresh every 30 s
    python deploy/ats/monitor.py --tail kl          # live-tail kl loss log
    python deploy/ats/monitor.py --tail kl --n 60  # last 60 lines
    python deploy/ats/monitor.py --summary          # just pass/fail counts
"""
import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta

ROOT     = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG_DIR  = os.path.join(ROOT, "db", "logs", "ats")
PID_FILE = os.path.join(LOG_DIR, "ats_pids.json")

STEP_RE  = re.compile(r"Step\s+(\d+)/(\d+)\s*\|\s*train_loss:\s*([\d.eE+\-nan]+)")
VAL_RE   = re.compile(r"Step\s+(\d+)/\d+\s*\|\s*val_loss:\s*([\d.eE+\-nan]+)")


def _pid_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False


def _parse_log(log_path):
    if not os.path.exists(log_path):
        return {}
    step = total = 0
    loss = val = None
    try:
        with open(log_path, errors="replace") as f:
            lines = f.readlines()
        n_lines = len(lines)
        for line in reversed(lines):
            if loss is None:
                m = STEP_RE.search(line)
                if m:
                    step, total = int(m.group(1)), int(m.group(2))
                    try:
                        loss = float(m.group(3))
                    except ValueError:
                        loss = float("nan")
            if val is None:
                m = VAL_RE.search(line)
                if m:
                    try:
                        val = float(m.group(2))
                    except ValueError:
                        pass
            if loss is not None and val is not None:
                break
    except Exception:
        n_lines = 0
    return {"step": step, "total": total, "loss": loss, "val": val,
            "lines": n_lines}


def _bar(done, total, w=18):
    if not total:
        return "?" * w
    f = int(w * done / total)
    return "█" * f + "░" * (w - f)


def _elapsed(iso):
    try:
        delta = datetime.now() - datetime.fromisoformat(iso)
        h, r  = divmod(int(delta.total_seconds()), 3600)
        m, s  = divmod(r, 60)
        return f"{h}h{m:02d}m"
    except Exception:
        return "?"


def _eta(step, total, elapsed_s):
    if step <= 0:
        return "?"
    rate = step / elapsed_s   # steps/s
    remaining = (total - step) / rate
    return str(timedelta(seconds=int(remaining)))[:-3]   # HH:MM


def show_table(pid_data, summary_only=False):
    family   = pid_data.get("family", "gpt2")
    steps    = pid_data.get("steps", 1000)
    started  = pid_data.get("started_at", "")
    log_pre  = pid_data.get("log_prefix", "train_")
    losses   = pid_data.get("losses", {})

    rows = []
    alive = done = failed = 0

    try:
        start_dt = datetime.fromisoformat(started)
        elapsed_s = (datetime.now() - start_dt).total_seconds()
    except Exception:
        elapsed_s = 1

    for loss, pid in losses.items():
        log_path = os.path.join(LOG_DIR, f"{log_pre}{loss}.log")
        info     = _parse_log(log_path)
        cur      = info.get("step", 0)
        tot      = info.get("total", steps)
        tloss    = info.get("loss")
        vloss    = info.get("val")
        is_alive = _pid_alive(pid)

        if is_alive:
            state = "RUN"
            alive += 1
        elif cur >= tot > 0 or (not is_alive and cur > 0):
            state = "DONE"
            done += 1
        else:
            state = "FAIL"
            failed += 1

        pct   = cur / tot * 100 if tot else 0
        bar   = _bar(cur, tot)
        tl_s  = f"{tloss:.4f}" if tloss and not (tloss != tloss) else "  ?"
        vl_s  = f"{vloss:.4f}" if vloss and not (vloss != vloss) else "  ?"
        eta   = _eta(cur, tot, elapsed_s) if is_alive and cur > 0 else ""
        rows.append((state, loss, bar, pct, cur, tot, tl_s, vl_s, eta, pid))

    if summary_only:
        print(f"  {alive} running / {done} done / {failed} failed  "
              f"of {len(losses)}   elapsed: {_elapsed(started)}")
        return

    print()
    print(f"  ATS Parallel Training — {family.upper()}   elapsed: {_elapsed(started)}")
    print(f"  {alive} running  ·  {done} done  ·  {failed} failed  of {len(losses)}")
    print()
    print(f"  {'ST':4s} {'LOSS':20s} {'PROGRESS':20s} {'%':>5s}  "
          f"{'STEP':>9s}  {'TRAIN':>8s}  {'VAL':>8s}  ETA     PID")
    print("  " + "─" * 95)
    for state, loss, bar, pct, cur, tot, tl, vl, eta, pid in rows:
        col = "\033[92m" if state == "DONE" else ("\033[91m" if state == "FAIL" else "")
        rst = "\033[0m"
        print(f"  {col}{state:4s}{rst} {loss:20s} {bar} {pct:5.1f}%  "
              f"{cur:4d}/{tot:<4d}  {tl:>8s}  {vl:>8s}  {eta:6s}  {pid}")
    print()

    if done == len(losses):
        print("  ✓  All losses complete!")
        print("  Next: python deploy/ats/06_eval_trained.py")
    elif alive > 0:
        print(f"  Tail a log: python deploy/ats/monitor.py --tail <loss>")
        print(f"  Available: {', '.join(l for l, *_ in rows if _[0] == 'RUN'[:1])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch",    action="store_true",
                    help="Auto-refresh every --interval seconds.")
    ap.add_argument("--interval", type=int, default=30)
    ap.add_argument("--tail",     metavar="LOSS", default=None,
                    help="Live-tail one loss log (e.g. --tail kl).")
    ap.add_argument("--n",        type=int, default=40,
                    help="Lines to tail with --tail (default 40).")
    ap.add_argument("--summary",  action="store_true",
                    help="Print one-line summary only.")
    ap.add_argument("--family",   default="gpt2")
    args = ap.parse_args()

    # ── Tail mode ─────────────────────────────────────────────────────────────
    if args.tail:
        family = args.family
        if os.path.exists(PID_FILE):
            family = json.load(open(PID_FILE)).get("family", family)
        log_path = os.path.join(LOG_DIR, f"train_{args.tail}.log")
        if not os.path.exists(log_path):
            print(f"Log not found: {log_path}")
            sys.exit(1)
        print(f"Tailing {log_path}  (Ctrl+C to stop)\n")
        # tail -f equivalent in Python
        with open(log_path, errors="replace") as f:
            # Print last --n lines first
            lines = f.readlines()
            for line in lines[-args.n:]:
                sys.stdout.write(line)
            sys.stdout.flush()
            try:
                while True:
                    line = f.readline()
                    if line:
                        sys.stdout.write(line)
                        sys.stdout.flush()
                    else:
                        time.sleep(0.5)
            except KeyboardInterrupt:
                pass
        return

    # ── Status mode ───────────────────────────────────────────────────────────
    if not os.path.exists(PID_FILE):
        print("No training jobs found (no PID file).")
        print("Start training: python deploy/ats/05_train_parallel.py")
        sys.exit(0)

    pid_data = json.load(open(PID_FILE))

    if args.watch:
        try:
            while True:
                os.system("clear")
                show_table(pid_data, args.summary)
                print(f"  Refreshing every {args.interval}s ... Ctrl+C to stop")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        show_table(pid_data, args.summary)


if __name__ == "__main__":
    main()
