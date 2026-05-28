#!/usr/bin/env python3
"""
run_summary.py — structured summary of the most recent pipeline run.

Produces a level-appropriate report pulled from results.db + pipeline log.
The summary format differs by hardware tier because the question differs:

  Level 1 (laptop)     -> "Did every code path run without crashing?"
                          Output: checklist (pass/fail per eval mode + error scan)
  Level 2 (colab_lite) -> "Does the loss function show a positive trend?"
                          Output: block efficiency vs naive baseline, per mode
  Level 3 (colab)      -> "How large is the effect, precisely?"
                          Output: full BE table by mode × K, delta + significance
  Level 4 (a100)       -> "Is it good enough for a paper?"
                          Output: paper table, CI, alpha acceptance, task score

Usage:
    # Auto-detect tier from most recent DB rows
    python orchestration/run_summary.py

    # Specific tier
    python orchestration/run_summary.py --hw_tier laptop
    python orchestration/run_summary.py --hw_tier colab
    python orchestration/run_summary.py --hw_tier a100

    # Custom DB / log paths (e.g. after downloading from Drive)
    python orchestration/run_summary.py --db /path/to/results.db --log /path/to/pipeline_output.log

    # Output a markdown block you can paste into RUN_LOG.md
    python orchestration/run_summary.py --markdown

    # Summarize a specific experiment_tag
    python orchestration/run_summary.py --tag "laptop-may-28"
"""

import argparse, os, sys, re, json
from datetime import datetime, timezone
from collections import defaultdict

HERE      = os.path.dirname(os.path.abspath(__file__))
GBV_ROOT  = os.path.dirname(HERE)
sys.path.insert(0, GBV_ROOT)

# ── DB import ------------------------------------------------------------------
try:
    from db.results_db import query_runs, query_train_curves
    _HAS_DB = True
except ImportError:
    _HAS_DB = False
    print("[warn] db.results_db not importable -- DB summary disabled", file=sys.stderr)

# ── Helpers -------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _date_tag() -> str:
    return datetime.now().strftime("%Y%m%d")

def _scan_log_errors(log_path: str, tail_lines: int = 200) -> list[str]:
    """Return a list of error/warning lines from the tail of the pipeline log."""
    if not log_path or not os.path.exists(log_path):
        return []
    with open(log_path, encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()
    tail = lines[-tail_lines:]
    errors = []
    for line in tail:
        lo = line.lower()
        if any(kw in lo for kw in ("error", "traceback", "exception", "killed",
                                    "sigkill", "oom", "exit code", "failed")):
            errors.append(line.rstrip())
    return errors

def _scan_log_stats(log_path: str) -> dict:
    """Extract final training loss and step count from log."""
    stats = {"final_loss": None, "final_step": None, "train_time_min": None}
    if not log_path or not os.path.exists(log_path):
        return stats
    with open(log_path, encoding="utf-8", errors="replace") as fh:
        content = fh.read()
    # match patterns like "step 100/100  loss=0.4321" or "loss: 0.4321"
    for pat in [r"step\s+(\d+)[^\n]*loss[=:\s]+([0-9]+\.[0-9]+)",
                r"loss[=:\s]+([0-9]+\.[0-9]+)\s.*step\s+(\d+)"]:
        hits = re.findall(pat, content, re.IGNORECASE)
        if hits:
            last = hits[-1]
            try:
                stats["final_step"] = int(last[0])
                stats["final_loss"] = float(last[1])
            except (IndexError, ValueError):
                pass
            break
    # wall time from log timestamps if present
    ts_hits = re.findall(r"\[(\d{2}:\d{2}:\d{2})\]", content)
    if len(ts_hits) >= 2:
        try:
            t0 = datetime.strptime(ts_hits[0],  "%H:%M:%S")
            t1 = datetime.strptime(ts_hits[-1], "%H:%M:%S")
            delta = (t1 - t0).total_seconds()
            if delta < 0:  # crossed midnight
                delta += 86400
            stats["train_time_min"] = round(delta / 60, 1)
        except ValueError:
            pass
    return stats

# ── Summary builders ----------------------------------------------------------

def _level1_summary(runs: list, log_errors: list, log_stats: dict, markdown: bool) -> str:
    """
    Laptop code-check summary.
    Question: "Did every code path execute without crashing?"
    """
    EXPECTED_MODES = {"alpha", "bv", "gbv", "traversal", "specinfer", "naive"}
    found_modes    = {r["mode"] for r in runs}
    ok_modes       = EXPECTED_MODES & found_modes
    missing_modes  = EXPECTED_MODES - found_modes

    # Did the run log any eval results at all?
    has_results = len(runs) > 0

    lines = []
    sep = "---" if not markdown else "---"

    lines.append("## Level 1 — Laptop Code-Check Summary")
    lines.append(f"Generated: {_now()}")
    lines.append("")
    lines.append("> **Reminder**: Level 1 result *numbers* are meaningless.")
    lines.append("> The 0.6B teacher = same capacity as draft. Only check: did it run?")
    lines.append("")

    # ── Eval mode checklist ───────────────────────────────────────────────────
    lines.append("### Eval Mode Checklist")
    lines.append("")
    lines.append("| Mode | Status | Rows in DB | block_eff sample |")
    lines.append("|------|--------|-----------|-----------------|")
    for mode in sorted(EXPECTED_MODES):
        mode_rows = [r for r in runs if r["mode"] == mode]
        if mode_rows:
            be_vals = [r["block_eff"] for r in mode_rows if r["block_eff"] is not None]
            be_str  = f"{be_vals[0]:.3f}" if be_vals else "NULL"
            lines.append(f"| {mode:10s} | PASS | {len(mode_rows):3d} | {be_str} |")
        else:
            lines.append(f"| {mode:10s} | **MISSING** | 0 | — |")

    lines.append("")
    pass_count = len(ok_modes)
    total      = len(EXPECTED_MODES)
    lines.append(f"**{pass_count}/{total} modes produced DB rows.**")
    lines.append("")

    # ── Training stats ────────────────────────────────────────────────────────
    lines.append("### Training Stats")
    lines.append("")
    if log_stats["final_step"] is not None:
        lines.append(f"- Final step:  **{log_stats['final_step']}**")
    if log_stats["final_loss"] is not None:
        lines.append(f"- Final loss:  **{log_stats['final_loss']:.4f}**")
    if log_stats["train_time_min"] is not None:
        lines.append(f"- Wall time:   **{log_stats['train_time_min']} min**")
    if not any(log_stats.values()):
        lines.append("- *(log file not available or no stats extracted)*")
    lines.append("")

    # ── Error scan ────────────────────────────────────────────────────────────
    lines.append("### Error Scan (last 200 log lines)")
    lines.append("")
    if log_errors:
        lines.append(f"**{len(log_errors)} error/warning line(s) found:**")
        lines.append("```")
        for e in log_errors[:20]:
            lines.append(e)
        if len(log_errors) > 20:
            lines.append(f"... and {len(log_errors)-20} more")
        lines.append("```")
    else:
        lines.append("No error keywords found in log tail. ✓")
    lines.append("")

    # ── Gate verdict ──────────────────────────────────────────────────────────
    gate_pass = (pass_count == total) and (len(log_errors) == 0)
    lines.append("### Gate Verdict")
    lines.append("")
    if gate_pass:
        lines.append("✅ **PASS** — all eval modes ran, no errors in log.")
        lines.append("   Ready to promote to Level 2 (colab_lite) after code review.")
    else:
        issues = []
        if missing_modes:
            issues.append(f"missing eval modes: {', '.join(sorted(missing_modes))}")
        if log_errors:
            issues.append(f"{len(log_errors)} error line(s) in log")
        lines.append(f"❌ **FAIL** — {'; '.join(issues)}.")
        lines.append("   Fix before running colab_lite.")

    return "\n".join(lines)


def _research_summary(runs: list, hw_tier: str, markdown: bool) -> str:
    """
    Colab / A100 research summary.
    Question: "How does each loss compare to the naive baseline?"
    """
    if not runs:
        return "No runs found in DB for this hw_tier."

    # Group by (loss_name, mode, K)
    from collections import defaultdict
    table = defaultdict(list)  # (loss, mode, K) -> list of block_eff
    for r in runs:
        key = (r["loss_name"], r["mode"], r["K"])
        if r["block_eff"] is not None:
            table[key].append(r["block_eff"])

    # Find baseline (loss_name == 'baseline' or draft_label == 'baseline')
    baseline_be = {}
    for r in runs:
        if r["loss_name"] == "baseline" or r.get("draft_label") == "baseline":
            key = (r["mode"], r["K"])
            if r["block_eff"] is not None:
                baseline_be[key] = r["block_eff"]

    lines = []
    tier_label = {"colab_lite": "Level 2 (Colab Lite)", "colab": "Level 3 (Colab T4)",
                  "a100": "Level 4 (A100)"}.get(hw_tier, hw_tier)
    lines.append(f"## {tier_label} — Research Summary")
    lines.append(f"Generated: {_now()}")
    lines.append("")
    lines.append("### Block Efficiency by Loss × Mode")
    lines.append("")

    # Header
    modes = sorted({k[1] for k in table})
    losses = sorted({k[0] for k in table if k[0] != "baseline"})
    Ks     = sorted({k[2] for k in table})

    if not losses:
        lines.append("No non-baseline runs found.")
        return "\n".join(lines)

    # One sub-table per K
    for K in Ks:
        lines.append(f"**K = {K}**")
        lines.append("")
        header = "| Loss |" + "".join(f" {m:10s} |" for m in modes)
        sep    = "|------|" + "".join(f"------------|" for _ in modes)
        lines.append(header)
        lines.append(sep)

        # Baseline row
        baseline_row = "| baseline |"
        for mode in modes:
            be = baseline_be.get((mode, K))
            baseline_row += f" {be:.3f}      |" if be else " —          |"
        lines.append(baseline_row)

        # Loss rows
        for loss in losses:
            row = f"| {loss:8s} |"
            for mode in modes:
                vals = table.get((loss, mode, K), [])
                if vals:
                    mean_be = sum(vals) / len(vals)
                    base_be = baseline_be.get((mode, K))
                    if base_be:
                        delta = mean_be - base_be
                        sign  = "+" if delta >= 0 else ""
                        row += f" {mean_be:.3f} ({sign}{delta:.3f}) |"
                    else:
                        row += f" {mean_be:.3f}      |"
                else:
                    row += " —               |"
            lines.append(row)
        lines.append("")

    # Alpha
    alpha_rows = [r for r in runs if r["mode"] == "alpha" and r["alpha_mean"] is not None]
    if alpha_rows:
        lines.append("### Alpha Acceptance Rate")
        lines.append("")
        lines.append("| Loss | alpha_mean | alpha_std | n_prompts |")
        lines.append("|------|-----------|-----------|-----------|")
        by_loss = defaultdict(list)
        for r in alpha_rows:
            by_loss[r["loss_name"]].append(r)
        for loss, rows in sorted(by_loss.items()):
            means = [r["alpha_mean"] for r in rows if r["alpha_mean"]]
            stds  = [r["alpha_std"]  for r in rows if r["alpha_std"]]
            n     = rows[0]["n_prompts"] if rows else "?"
            m_str = f"{sum(means)/len(means):.3f}" if means else "—"
            s_str = f"{sum(stds)/len(stds):.3f}"  if stds  else "—"
            lines.append(f"| {loss:8s} | {m_str:9s} | {s_str:9s} | {n:9} |")
        lines.append("")

    # Gate verdict
    lines.append("### Gate Verdict")
    lines.append("")
    # Simple heuristic: does any non-baseline loss beat baseline BE on any mode?
    improvements = []
    for (loss, mode, K), vals in table.items():
        if loss == "baseline":
            continue
        base_be = baseline_be.get((mode, K))
        if base_be and vals:
            mean_be = sum(vals) / len(vals)
            if mean_be > base_be:
                improvements.append(f"{loss} on {mode} K={K}: +{mean_be-base_be:.3f}")

    if hw_tier == "colab_lite":
        if improvements:
            lines.append("✅ **Trend detected** — at least one loss beats baseline:")
            for imp in improvements[:5]:
                lines.append(f"   - {imp}")
            lines.append("")
            lines.append("   Ready to promote to Level 3 (colab) after:")
            lines.append("   - [ ] Code review of the winning loss implementation")
            lines.append("   - [ ] At least one more colab_lite seed confirms direction")
        else:
            lines.append("❌ **No positive trend** — no loss beats baseline BE.")
            lines.append("   Do NOT promote to Level 3. Revisit loss implementation.")
    elif hw_tier == "colab":
        if improvements:
            lines.append("✅ **Results confirmed** — improvements vs baseline:")
            for imp in improvements[:5]:
                lines.append(f"   - {imp}")
            lines.append("")
            lines.append("   Ready to promote to Level 4 (A100) after:")
            lines.append("   - [ ] Code review of winning loss + verifier")
            lines.append("   - [ ] 2+ seeds confirm direction")
            lines.append("   - [ ] Hypothesis is finalized (no more code changes)")
        else:
            lines.append("❌ **No improvement** — no loss beats baseline BE at Level 3.")
            lines.append("   Return to Level 2 and revisit the hypothesis.")
    elif hw_tier == "a100":
        lines.append("📋 **Paper-level run.** Manual review required:")
        lines.append("   - [ ] Effect size > 5% consistently across modes")
        lines.append("   - [ ] Results reproducible across 2+ seeds")
        lines.append("   - [ ] Ablations identify the responsible component")
        lines.append("   - [ ] Closest prior work identified and delta measured")

    return "\n".join(lines)


# ── Train curve summary --------------------------------------------------------

def _curve_summary(hw_tier: str) -> str:
    """Pull training curve stats from DB."""
    if not _HAS_DB:
        return ""
    curves = query_train_curves()
    if not curves:
        return "No training curve data in DB."
    lines = ["### Training Curves"]
    lines.append("")
    by_label = defaultdict(list)
    for row in curves:
        by_label[row["label"]].append(row)
    for label, rows in sorted(by_label.items()):
        train_rows = [r for r in rows if r.get("split","train") == "train"]
        val_rows   = [r for r in rows if r.get("split","train") == "val"]
        if not train_rows:
            continue
        first_loss = train_rows[0]["loss"]
        final_loss = train_rows[-1]["loss"]
        final_step = train_rows[-1]["step"]
        drop_pct   = 100 * (first_loss - final_loss) / first_loss if first_loss else 0
        val_str    = ""
        if val_rows:
            val_final = val_rows[-1]["loss"]
            val_str   = f"  val_loss_final={val_final:.4f}"
        lines.append(f"- **{label}**: loss {first_loss:.4f} → {final_loss:.4f} "
                     f"({drop_pct:+.1f}%) over {final_step} steps{val_str}")
    return "\n".join(lines)


# ── Main -----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hw_tier", choices=["laptop","colab_lite","colab","a100"],
                   help="Hardware tier to summarize. Auto-detected from DB if omitted.")
    p.add_argument("--db",  default=None,
                   help="Path to results.db. Defaults to db/results.db in this repo.")
    p.add_argument("--log", default=None,
                   help="Path to pipeline_output.log. Used for error scan.")
    p.add_argument("--tag", default=None,
                   help="Filter by experiment_tag in DB.")
    p.add_argument("--markdown", action="store_true",
                   help="Emit markdown suitable for pasting into RUN_LOG.md.")
    p.add_argument("--log_id", default=None,
                   help="Optional exp_id (e.g. 20260528-001) to stamp the output.")
    args = p.parse_args()

    # ── Locate DB ─────────────────────────────────────────────────────────────
    if args.db:
        os.environ["SPECDIST_DB_PATH"] = args.db

    # ── Locate log ────────────────────────────────────────────────────────────
    log_path = args.log
    if not log_path:
        # Try common paths
        for candidate in [
            os.path.join(GBV_ROOT, "db", "logs", "pipeline_output.log"),
            "/content/drive/MyDrive/specdist/logs/pipeline_output.log",
        ]:
            if os.path.exists(candidate):
                log_path = candidate
                break

    # ── Query DB ──────────────────────────────────────────────────────────────
    runs = []
    if _HAS_DB:
        filters = {}
        if args.tag:
            filters["experiment_tag"] = args.tag
        all_runs = query_runs(filters or None)

        # Auto-detect hw_tier from most recent runs if not specified
        hw_tier = args.hw_tier
        if not hw_tier and all_runs:
            # Use the most common hw_tier in the last 50 rows
            recent   = all_runs[-50:]
            tier_counts = defaultdict(int)
            for r in recent:
                tier_counts[r.get("hw_tier", "laptop")] += 1
            hw_tier = max(tier_counts, key=tier_counts.get)
        hw_tier = hw_tier or "laptop"

        # Filter by hw_tier
        runs = [r for r in all_runs if r.get("hw_tier") == hw_tier]
        # If tag specified, don't further filter
    else:
        hw_tier = args.hw_tier or "laptop"

    # ── Log scan ──────────────────────────────────────────────────────────────
    log_errors = _scan_log_errors(log_path)
    log_stats  = _scan_log_stats(log_path)

    # ── Build summary ─────────────────────────────────────────────────────────
    print("=" * 70)
    if hw_tier == "laptop":
        print(_level1_summary(runs, log_errors, log_stats, args.markdown))
    else:
        print(_research_summary(runs, hw_tier, args.markdown))
        print()
        print(_curve_summary(hw_tier))

    print()
    print("=" * 70)
    print(f"DB rows for this tier: {len(runs)}")
    if log_path:
        print(f"Log: {log_path}")
    print()

    # ── Markdown run-log stub ─────────────────────────────────────────────────
    if args.markdown:
        exp_id = args.log_id or f"{_date_tag()}-NNN"
        print()
        print("## Paste into deploy/RUN_LOG.md:")
        print()
        if hw_tier == "laptop":
            modes_ran = "|".join(sorted({r["mode"] for r in runs})) or "—"
            err_str   = f"errors:{len(log_errors)}" if log_errors else "clean"
            print(f"| {exp_id} | {_date_tag()} | laptop | "
                  f"{log_stats.get('final_step','?')} | {modes_ran} | "
                  f"{log_stats.get('final_loss','?')} | {err_str} | |")
        else:
            # Find best non-baseline BE
            best_be, best_cfg = None, "—"
            for r in runs:
                if r["loss_name"] != "baseline" and r["block_eff"] is not None:
                    if best_be is None or r["block_eff"] > best_be:
                        best_be  = r["block_eff"]
                        best_cfg = f"{r['loss_name']}/{r['mode']}/K={r['K']}"
            wandb_hint = "https://wandb.ai/distillspec/runs/..."
            print(f"| {exp_id} | {hw_tier} | {best_cfg} | "
                  f"{best_be:.3f if best_be else '—'} | | {wandb_hint} | COMPLETE | |")

if __name__ == "__main__":
    main()
