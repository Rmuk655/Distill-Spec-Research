#!/usr/bin/env python
"""
One-row-per-run summary table over a directory of training runs, so you don't
have to open W&B for every run to tell which approach worked.

Covers BOTH naming conventions under a sweep root (e.g. .../prefix_overlap/):
  - po_prob_*   whose logs sit directly alongside the run dir (<run>.log)
  - jsd_*       whose logs live under <root>/logs/<run>.log instead

For each run dir found (anything containing a ckpt_best/ or ckpt_latest/ with
a state.json), reports:
  STATUS     finished (log has a "[done]" line) / CRASHED (Traceback near EOF,
             no "[done]") / running? (neither -- still going, or no log found)
  LOSS/OBJ   args.loss / args.prefix_objective, read straight from state.json's
             "train_args" (checkpointing.py's save_checkpoint() stores
             _serializable_args(args) there -- no need to regex flags out of
             log text or reconstruct them from the run's directory name)
  LR / WARMUP / LR_MIN / WD / TOPK / AUX / AUX_W / ANNEAL   the hyperparams
             actually used for this run, per the same state.json source
  BEST_BE / BEST_SM   best_val_block_eff / best_smoothed from the log's
             "[done]" summary line
  FINAL_BE   block_eff at the LAST "[val]" line, whether the run finished or
             not -- this is what catches "peaked early, then collapsed by the
             end" (e.g. topk20@lr1e-4: best=3.801, final=1.277)
  EASY/MED/HARD/FORGET   prompt-difficulty split + forgetting at the last val
             check -- easy=0 med=0 hard=100 is the smoking gun for total
             collapse
  COLLAPSED? heuristic: final_be < 70% of best_be => "yes"

Prefers ckpt_best/state.json for hyperparams (that's the checkpoint you'd
actually use); falls back to ckpt_latest/state.json if ckpt_best is missing
(e.g. a run that never improved past init).

USAGE:
    python scripts/summarize_sweep.py /sensei-fs-3/users/rkrishna/checkpoints/Qwen0.6B_Qwen8B/prefix_overlap
    # sort by best block_eff descending:
    python scripts/summarize_sweep.py <root> --sort best_be
"""
import argparse
import json
import os
import re
import sys

# train_args keys worth surfacing as columns -- add more here if you start a
# sweep dimension this doesn't cover yet.
HPARAM_COLS = [
    ("loss", "loss"),
    ("prefix_objective", "obj"),
    ("lr", "lr"),
    ("warmup_steps", "warmup"),
    ("lr_min_ratio", "lr_min"),
    ("weight_decay", "wd"),
    ("prefix_teacher_topk", "topk"),
    ("prefix_aux", "aux"),
    ("prefix_aux_weight", "aux_w"),
    ("prefix_anneal_steps", "anneal"),
    ("prefix_M", "M"),
    ("steps", "steps"),
]


def find_log(root: str, run_name: str):
    """Two different launch scripts, two different conventions:
      - one-off manual launches (nohup ... > <root>/<run>.log)
      - sweep_prefix_overlap.sh, which ALWAYS writes to
        <root>/logs/<run>.out (note: .out, not .log -- see run_job() there)
    Check all three combinations rather than assuming one."""
    candidates = [
        os.path.join(root, f"{run_name}.log"),
        os.path.join(root, "logs", f"{run_name}.log"),
        os.path.join(root, "logs", f"{run_name}.out"),
        os.path.join(root, f"{run_name}.out"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def load_train_args(run_dir: str):
    for ckpt in ("ckpt_best", "ckpt_latest"):
        state_path = os.path.join(run_dir, ckpt, "state.json")
        if os.path.isfile(state_path):
            try:
                with open(state_path, encoding="utf-8") as f:
                    state = json.load(f)
                return state.get("train_args", {})
            except (json.JSONDecodeError, OSError):
                continue
    return {}


def parse_log(log_path: str):
    with open(log_path, encoding="utf-8", errors="replace") as f:
        text = f.read()

    if "[done]" in text:
        status = "finished"
    elif re.search(r"^Traceback", text[-4000:], re.MULTILINE):
        status = "CRASHED"
    else:
        status = "running?"

    done_m = re.findall(
        r"best_val_block_eff = ([0-9.]+)\s+best_smoothed = ([0-9.]+)", text)
    best_be, best_sm = done_m[-1] if done_m else (None, None)

    val_lines = re.findall(
        r"\[val\] step=\d+\s+block_eff=([0-9.]+).*?forget=([0-9.]+)\s+"
        r"easy=(\d+)\s+med=(\d+)\s+hard=(\d+)", text)
    if val_lines:
        final_be, forget, easy, med, hard = val_lines[-1]
    else:
        final_be = forget = easy = med = hard = None

    return status, best_be, best_sm, final_be, easy, med, hard, forget


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="sweep root dir, e.g. .../prefix_overlap")
    ap.add_argument("--sort", default=None,
                    choices=["best_be", "best_sm", "final_be", "run"],
                    help="sort rows by this column (descending; default: run name)")
    args = ap.parse_args()

    run_dirs = sorted(
        d for d in os.listdir(args.root)
        if os.path.isdir(os.path.join(args.root, d)) and d != "logs"
        and (os.path.isdir(os.path.join(args.root, d, "ckpt_best"))
             or os.path.isdir(os.path.join(args.root, d, "ckpt_latest")))
    )
    if not run_dirs:
        sys.exit(f"No run dirs with ckpt_best/ckpt_latest found under {args.root}")

    rows = []
    for run_name in run_dirs:
        run_dir = os.path.join(args.root, run_name)
        train_args = load_train_args(run_dir)
        log_path = find_log(args.root, run_name)

        if log_path:
            status, best_be, best_sm, final_be, easy, med, hard, forget = parse_log(log_path)
        else:
            status, best_be, best_sm, final_be, easy, med, hard, forget = (
                "no log found", None, None, None, None, None, None, None)

        collapsed = "-"
        if best_be and final_be:
            collapsed = "yes" if float(final_be) < 0.7 * float(best_be) else "no"

        row = {"run": run_name, "status": status,
               "best_be": best_be, "best_sm": best_sm, "final_be": final_be,
               "easy": easy, "med": med, "hard": hard, "forget": forget,
               "collapsed": collapsed}
        for key, col in HPARAM_COLS:
            row[col] = train_args.get(key, "-")
        rows.append(row)

    if args.sort:
        sort_key = {"best_be": "best_be", "best_sm": "best_sm", "final_be": "final_be"}.get(args.sort, "run")
        if sort_key == "run":
            rows.sort(key=lambda r: r["run"])
        else:
            rows.sort(key=lambda r: float(r[sort_key]) if r[sort_key] else -1, reverse=True)
    else:
        rows.sort(key=lambda r: r["run"])

    cols = (["run", "status"] + [c for _, c in HPARAM_COLS]
            + ["best_be", "best_sm", "final_be", "easy", "med", "hard", "forget", "collapsed"])
    widths = {c: max(len(c), max((len(str(r.get(c, "-"))) for r in rows), default=0)) for c in cols}

    def fmt_row(vals):
        return "  ".join(str(vals.get(c, "-")).ljust(widths[c]) for c in cols)

    print(fmt_row({c: c.upper() for c in cols}))
    for r in rows:
        print(fmt_row(r))


if __name__ == "__main__":
    main()
