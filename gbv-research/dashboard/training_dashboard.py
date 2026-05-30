"""
training_dashboard.py — Live web dashboard for SpecDist training + evaluation results.

Shows:
  • Block efficiency heatmap (model × verifier) — updated as each eval cell lands
  • Training loss curves (step → loss, per model)
  • Per-prompt acceptance-rate breakdown
  • Live pipeline log tail (which step is running, ETA)
  • HW-tier filter (laptop / colab / a100) to compare results across machines

Start:
    python dashboard/training_dashboard.py                         # http://localhost:5000
    python dashboard/training_dashboard.py --port 8080
    python dashboard/training_dashboard.py --db /path/to/results.db        # custom DB
    python dashboard/training_dashboard.py --root ~/Downloads/kaggle-run/  # all files in one folder
    python dashboard/training_dashboard.py --root ~/Downloads/kaggle-run/ --port 8080

  --root DIR   sets --db DIR/results.db  +  --logs DIR  +  --state DIR at once.
               Download results.db, pipeline_output.log, be_progress.log, and
               pipeline_state_kaggle.json into one folder and use --root to see
               all results locally.

Endpoints:
    GET /                     HTML dashboard
    GET /api/runs             All run rows (JSON), supports ?col=val filters
    GET /api/dimensions       Distinct values per categorical column
    GET /api/per_prompt/<id>  Per-prompt data for one run
    GET /api/train_curves     Training loss curves
    GET /api/log_tail         Last N lines of pipeline_output.log or be_progress.log
"""

import glob, os, re, sys, json, argparse
_SERVER_HERE  = os.path.dirname(os.path.abspath(__file__))  # gbv-research/dashboard/
_GBV_RESEARCH = os.path.dirname(_SERVER_HERE)               # gbv-research/
# Add gbv-research/db/ so `import results_db` resolves correctly
sys.path.insert(0, os.path.join(_GBV_RESEARCH, "db"))
import results_db

HERE          = _SERVER_HERE
# Logs live in db/logs/ by default, but respect SPECDIST_LOGS_ROOT so that
# cloud runs (Colab/Kaggle/Modal) point at the persistent storage logs directory.
_DB_LOGS      = (os.environ.get("SPECDIST_LOGS_ROOT")
                 or os.path.join(_GBV_RESEARCH, "db", "logs"))
_BE_LOG       = os.path.join(_DB_LOGS, "be_progress.log")
_PIPELINE_LOG = os.path.join(_DB_LOGS, "pipeline_output.log")

try:
    from flask import Flask, jsonify, request, render_template_string, send_from_directory
except ImportError:
    print("Flask not installed. Run: pip install flask")
    sys.exit(1)

app = Flask(__name__)

_STATIC_DIR = os.path.join(_SERVER_HERE, "static")


@app.route("/static/<path:filename>")
def dashboard_static(filename):
    """Serve bundled static assets (Plotly, Bootstrap) from dashboard/static/."""
    return send_from_directory(_STATIC_DIR, filename)

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

@app.route("/api/runs")
def api_runs():
    filters = {k: v for k, v in request.args.items()
               if k in ("draft_label","loss_name","dataset","mode","K","temperature",
                        "train_steps","experiment_tag")}
    # cast numeric filters
    for k in ("K", "train_steps"):
        if k in filters:
            filters[k] = int(filters[k])
    for k in ("temperature",):
        if k in filters:
            filters[k] = float(filters[k])
    rows = results_db.query_runs(filters or None)
    return jsonify(rows)


@app.route("/api/dimensions")
def api_dimensions():
    cols = ["draft_label", "loss_name", "dataset", "mode", "K",
            "temperature", "train_steps", "experiment_tag"]
    dims = {c: results_db.distinct_values(c) for c in cols}
    # T=0.0 rows are perplexity-check placeholders (no real speculative-decoding run).
    # Strip them so they never appear in the Temperature filter or dropdowns.
    dims["temperature"] = [t for t in dims.get("temperature", []) if t != 0.0]
    # K=0 rows are perplexity-check runs (mode=perplexity, no speculative decoding).
    # Strip K=0 — users cannot meaningfully filter on it.
    dims["K"] = [k for k in dims.get("K", []) if k != 0]
    # Model family pairs — the primary "don't mix" grouping dimension
    dims["model_pairs"] = results_db.distinct_model_pairs()
    return jsonify(dims)


@app.route("/api/per_prompt/<int:run_id>")
def api_per_prompt(run_id):
    rows = results_db.query_per_prompt(run_id)
    return jsonify(rows)


@app.route("/api/train_curves")
def api_train_curves():
    label = request.args.get("label")
    rows = results_db.query_train_curves(label)
    return jsonify(rows)


@app.route("/api/pipeline_status")
def api_pipeline_status():
    """Read all pipeline_state_*.json files and return live progress."""
    # experiment.py writes state files to SPECDIST_STORAGE_ROOT when set
    # (Colab/cloud runs), otherwise they land in orchestration/.
    # start_dashboard() already exports SPECDIST_STORAGE_ROOT to this subprocess.
    _state_dir = (os.environ.get("SPECDIST_STORAGE_ROOT")
                  or os.path.join(_GBV_RESEARCH, "orchestration"))
    state_files = sorted(
        glob.glob(os.path.join(_state_dir, "pipeline_state_*.json")),
        key=os.path.getmtime, reverse=True,
    )
    if not state_files:
        return jsonify({"found": False})

    sf = state_files[0]
    try:
        with open(sf) as f:
            state = json.load(f)
    except Exception as e:
        return jsonify({"found": False, "error": str(e)})

    steps = state.get("steps", {})
    # ── Step order and phase labels ────────────────────────────────────────────
    # MUST stay in sync with experiment.py build_steps() step IDs and groups.
    # If you add a step there, add a matching entry here.
    STEP_ORDER = [
        # Phase 1 — Baseline
        "eval_baseline_gsm8k",
        # Phase 2 — Training (offline losses)
        "train_kl_gsm8k",              "merge_kl_gsm8k",
        "train_ebe_gsm8k",             "merge_ebe_gsm8k",
        "train_ebe_single_gsm8k",      "merge_ebe_single_gsm8k",
        "train_rev_kl_gsm8k",          "merge_rev_kl_gsm8k",
        "train_jsd_gsm8k",             "merge_jsd_gsm8k",
        "train_l1_gsm8k",              "merge_l1_gsm8k",
        # Phase 2 — Online training
        "online_adapt_gsm8k",              "merge_online_gsm8k",
        "online_ebe_adapt_gsm8k",          "merge_online_ebe_gsm8k",
        "online_ebe_single_adapt_gsm8k",   "merge_online_ebe_single_gsm8k",
        # Phase 3 — GSM8K Eval
        "eval_kl_gsm8k",
        "eval_ebe_gsm8k",
        "eval_ebe_single_gsm8k",
        "eval_rev_kl_gsm8k",
        "eval_jsd_gsm8k",
        "eval_l1_gsm8k",
        "eval_online_gsm8k",
        "eval_online_ebe_gsm8k",
        "eval_online_ebe_single_gsm8k",
        # Phase 4 — Multi-Dataset
        "eval_baseline_all",
        "eval_kl_all",
        "eval_ebe_all",
        "eval_ebe_single_all",
        "eval_rev_kl_all",
        "eval_jsd_all",
        "eval_l1_all",
        "eval_online_all",
        "eval_online_ebe_all",
        "eval_online_ebe_single_all",
        # Phase 5 — EAGLE benchmark
        "eagle_gen", "eagle_train", "eagle_eval",
        # Phase 6 — Tree-structured losses (train + merge, GSM8K eval)
        "train_kl_tree_gsm8k",       "merge_kl_tree_gsm8k",
        "train_bv_tree_gsm8k",       "merge_bv_tree_gsm8k",
        "train_gbv_tree_gsm8k",      "merge_gbv_tree_gsm8k",
        "train_trav_tree_gsm8k",     "merge_trav_tree_gsm8k",
        "train_ebe_tree_gsm8k",      "merge_ebe_tree_gsm8k",
        "train_rev_kl_tree_gsm8k",   "merge_rev_kl_tree_gsm8k",
        "train_jsd_tree_gsm8k",      "merge_jsd_tree_gsm8k",
        # Verifier-aligned OT tree losses (added 2026-05)
        "train_naive_tree_gsm8k",    "merge_naive_tree_gsm8k",
        "train_nss_tree_gsm8k",      "merge_nss_tree_gsm8k",
        "train_si_tree_gsm8k",       "merge_si_tree_gsm8k",
        "train_st_tree_gsm8k",       "merge_st_tree_gsm8k",
        "train_khisti_tree_gsm8k",   "merge_khisti_tree_gsm8k",
        "online_kl_tree_adapt_gsm8k",  "merge_online_kl_tree_gsm8k",
        "online_ebe_tree_adapt_gsm8k", "merge_online_ebe_tree_gsm8k",
        "eval_kl_tree_gsm8k",
        "eval_bv_tree_gsm8k",
        "eval_gbv_tree_gsm8k",
        "eval_trav_tree_gsm8k",
        "eval_ebe_tree_gsm8k",
        "eval_rev_kl_tree_gsm8k",
        "eval_jsd_tree_gsm8k",
        "eval_naive_tree_gsm8k",
        "eval_nss_tree_gsm8k",
        "eval_si_tree_gsm8k",
        "eval_st_tree_gsm8k",
        "eval_khisti_tree_gsm8k",
        "eval_online_kl_tree_gsm8k",
        "eval_online_ebe_tree_gsm8k",
        # Phase 7 — Tree losses multi-dataset eval
        "eval_kl_tree_all",
        "eval_bv_tree_all",
        "eval_gbv_tree_all",
        "eval_trav_tree_all",
        "eval_ebe_tree_all",
        "eval_rev_kl_tree_all",
        "eval_jsd_tree_all",
        "eval_naive_tree_all",
        "eval_nss_tree_all",
        "eval_si_tree_all",
        "eval_st_tree_all",
        "eval_khisti_tree_all",
        "eval_online_kl_tree_all",
        "eval_online_ebe_tree_all",
    ]
    PHASE_LABELS = {
        "eval_baseline_gsm8k":           "Ph 1 — Baseline",
        "train_kl_gsm8k":                "Ph 2 — Training",
        "merge_kl_gsm8k":                "Ph 2 — Training",
        "train_ebe_gsm8k":               "Ph 2 — Training",
        "merge_ebe_gsm8k":               "Ph 2 — Training",
        "train_ebe_single_gsm8k":        "Ph 2 — Training",
        "merge_ebe_single_gsm8k":        "Ph 2 — Training",
        "train_rev_kl_gsm8k":            "Ph 2 — Training",
        "merge_rev_kl_gsm8k":            "Ph 2 — Training",
        "train_jsd_gsm8k":               "Ph 2 — Training",
        "merge_jsd_gsm8k":               "Ph 2 — Training",
        "train_l1_gsm8k":                "Ph 2 — Training",
        "merge_l1_gsm8k":                "Ph 2 — Training",
        "online_adapt_gsm8k":            "Ph 2 — Training",
        "merge_online_gsm8k":            "Ph 2 — Training",
        "online_ebe_adapt_gsm8k":        "Ph 2 — Training",
        "merge_online_ebe_gsm8k":        "Ph 2 — Training",
        "online_ebe_single_adapt_gsm8k": "Ph 2 — Training",
        "merge_online_ebe_single_gsm8k": "Ph 2 — Training",
        "eval_kl_gsm8k":                 "Ph 3 — GSM8K Eval",
        "eval_ebe_gsm8k":                "Ph 3 — GSM8K Eval",
        "eval_ebe_single_gsm8k":         "Ph 3 — GSM8K Eval",
        "eval_rev_kl_gsm8k":             "Ph 3 — GSM8K Eval",
        "eval_jsd_gsm8k":                "Ph 3 — GSM8K Eval",
        "eval_l1_gsm8k":                 "Ph 3 — GSM8K Eval",
        "eval_online_gsm8k":             "Ph 3 — GSM8K Eval",
        "eval_online_ebe_gsm8k":         "Ph 3 — GSM8K Eval",
        "eval_online_ebe_single_gsm8k":  "Ph 3 — GSM8K Eval",
        "eval_baseline_all":             "Ph 4 — Multi-DS",
        "eval_kl_all":                   "Ph 4 — Multi-DS",
        "eval_ebe_all":                  "Ph 4 — Multi-DS",
        "eval_ebe_single_all":           "Ph 4 — Multi-DS",
        "eval_rev_kl_all":               "Ph 4 — Multi-DS",
        "eval_jsd_all":                  "Ph 4 — Multi-DS",
        "eval_l1_all":                   "Ph 4 — Multi-DS",
        "eval_online_all":               "Ph 4 — Multi-DS",
        "eval_online_ebe_all":           "Ph 4 — Multi-DS",
        "eval_online_ebe_single_all":    "Ph 4 — Multi-DS",
        "eagle_gen":                     "Ph 5 — EAGLE",
        "eagle_train":                   "Ph 5 — EAGLE",
        "eagle_eval":                    "Ph 5 — EAGLE",
        # Tree-structured losses
        "train_kl_tree_gsm8k":           "Ph 6 — Tree Training",
        "merge_kl_tree_gsm8k":           "Ph 6 — Tree Training",
        "train_bv_tree_gsm8k":           "Ph 6 — Tree Training",
        "merge_bv_tree_gsm8k":           "Ph 6 — Tree Training",
        "train_gbv_tree_gsm8k":          "Ph 6 — Tree Training",
        "merge_gbv_tree_gsm8k":          "Ph 6 — Tree Training",
        "train_trav_tree_gsm8k":         "Ph 6 — Tree Training",
        "merge_trav_tree_gsm8k":         "Ph 6 — Tree Training",
        "train_ebe_tree_gsm8k":          "Ph 6 — Tree Training",
        "merge_ebe_tree_gsm8k":          "Ph 6 — Tree Training",
        "train_rev_kl_tree_gsm8k":       "Ph 6 — Tree Training",
        "merge_rev_kl_tree_gsm8k":       "Ph 6 — Tree Training",
        "train_jsd_tree_gsm8k":          "Ph 6 — Tree Training",
        "merge_jsd_tree_gsm8k":          "Ph 6 — Tree Training",
        # Verifier-aligned OT tree losses (added 2026-05)
        "train_naive_tree_gsm8k":        "Ph 6 — Tree Training",
        "merge_naive_tree_gsm8k":        "Ph 6 — Tree Training",
        "train_nss_tree_gsm8k":          "Ph 6 — Tree Training",
        "merge_nss_tree_gsm8k":          "Ph 6 — Tree Training",
        "train_si_tree_gsm8k":           "Ph 6 — Tree Training",
        "merge_si_tree_gsm8k":           "Ph 6 — Tree Training",
        "train_st_tree_gsm8k":           "Ph 6 — Tree Training",
        "merge_st_tree_gsm8k":           "Ph 6 — Tree Training",
        "train_khisti_tree_gsm8k":       "Ph 6 — Tree Training",
        "merge_khisti_tree_gsm8k":       "Ph 6 — Tree Training",
        "online_kl_tree_adapt_gsm8k":    "Ph 6 — Tree Training",
        "merge_online_kl_tree_gsm8k":    "Ph 6 — Tree Training",
        "online_ebe_tree_adapt_gsm8k":   "Ph 6 — Tree Training",
        "merge_online_ebe_tree_gsm8k":   "Ph 6 — Tree Training",
        "eval_kl_tree_gsm8k":            "Ph 6 — Tree GSM8K Eval",
        "eval_bv_tree_gsm8k":            "Ph 6 — Tree GSM8K Eval",
        "eval_gbv_tree_gsm8k":           "Ph 6 — Tree GSM8K Eval",
        "eval_trav_tree_gsm8k":          "Ph 6 — Tree GSM8K Eval",
        "eval_ebe_tree_gsm8k":           "Ph 6 — Tree GSM8K Eval",
        "eval_rev_kl_tree_gsm8k":        "Ph 6 — Tree GSM8K Eval",
        "eval_jsd_tree_gsm8k":           "Ph 6 — Tree GSM8K Eval",
        "eval_naive_tree_gsm8k":         "Ph 6 — Tree GSM8K Eval",
        "eval_nss_tree_gsm8k":           "Ph 6 — Tree GSM8K Eval",
        "eval_si_tree_gsm8k":            "Ph 6 — Tree GSM8K Eval",
        "eval_st_tree_gsm8k":            "Ph 6 — Tree GSM8K Eval",
        "eval_khisti_tree_gsm8k":        "Ph 6 — Tree GSM8K Eval",
        "eval_online_kl_tree_gsm8k":     "Ph 6 — Tree GSM8K Eval",
        "eval_online_ebe_tree_gsm8k":    "Ph 6 — Tree GSM8K Eval",
        "eval_kl_tree_all":              "Ph 7 — Tree Multi-DS",
        "eval_bv_tree_all":              "Ph 7 — Tree Multi-DS",
        "eval_gbv_tree_all":             "Ph 7 — Tree Multi-DS",
        "eval_naive_tree_all":           "Ph 7 — Tree Multi-DS",
        "eval_nss_tree_all":             "Ph 7 — Tree Multi-DS",
        "eval_si_tree_all":              "Ph 7 — Tree Multi-DS",
        "eval_st_tree_all":              "Ph 7 — Tree Multi-DS",
        "eval_khisti_tree_all":          "Ph 7 — Tree Multi-DS",
        "eval_trav_tree_all":            "Ph 7 — Tree Multi-DS",
        "eval_ebe_tree_all":             "Ph 7 — Tree Multi-DS",
        "eval_rev_kl_tree_all":          "Ph 7 — Tree Multi-DS",
        "eval_jsd_tree_all":             "Ph 7 — Tree Multi-DS",
        "eval_online_kl_tree_all":       "Ph 7 — Tree Multi-DS",
        "eval_online_ebe_tree_all":      "Ph 7 — Tree Multi-DS",
    }

    # Training steps that stopped early (NaN / early-stop) but produced a partial
    # checkpoint are not real failures — pipeline can still merge + eval them.
    # Remap status from "failed" → "stopped" so badge shows orange not red.
    TRAIN_STEP_IDS = {
        "train_kl_gsm8k", "train_ebe_gsm8k", "train_ebe_single_gsm8k",
        "train_rev_kl_gsm8k", "train_jsd_gsm8k", "train_l1_gsm8k",
        "online_adapt_gsm8k", "online_ebe_adapt_gsm8k", "online_ebe_single_adapt_gsm8k",
        # Tree-structured losses
        "train_kl_tree_gsm8k", "train_bv_tree_gsm8k", "train_gbv_tree_gsm8k",
        "train_trav_tree_gsm8k", "train_ebe_tree_gsm8k", "train_rev_kl_tree_gsm8k",
        "train_jsd_tree_gsm8k",
        "online_kl_tree_adapt_gsm8k", "online_ebe_tree_adapt_gsm8k",
    }
    CKPT_DIR = os.path.join(HERE, "checkpoints")
    _CKPT_SUFFIXES = {
        "train_kl_gsm8k":                "kl-gsm8k",
        "train_ebe_gsm8k":               "ebe-gsm8k",
        "train_ebe_single_gsm8k":        "ebe_single-gsm8k",
        "train_rev_kl_gsm8k":            "rev_kl-gsm8k",
        "train_jsd_gsm8k":               "jsd-gsm8k",
        "train_l1_gsm8k":                "l1-gsm8k",
        "online_adapt_gsm8k":            "online-gsm8k",
        "online_ebe_adapt_gsm8k":        "online-ebe-gsm8k",
        "online_ebe_single_adapt_gsm8k": "online-ebe-single-gsm8k",
        # Tree-structured losses
        "train_kl_tree_gsm8k":           "kl_tree-gsm8k",
        "train_bv_tree_gsm8k":           "bv_tree-gsm8k",
        "train_gbv_tree_gsm8k":          "gbv_tree-gsm8k",
        "train_trav_tree_gsm8k":         "trav_tree-gsm8k",
        "train_ebe_tree_gsm8k":          "ebe_tree-gsm8k",
        "train_rev_kl_tree_gsm8k":       "rev_kl_tree-gsm8k",
        "train_jsd_tree_gsm8k":          "jsd_tree-gsm8k",
        "online_kl_tree_adapt_gsm8k":    "online_kl_tree-gsm8k",
        "online_ebe_tree_adapt_gsm8k":   "online_ebe_tree-gsm8k",
    }

    def _has_partial_output(sid):
        """True if a training step produced any checkpoint (ckpt_latest or final)."""
        suffix = _CKPT_SUFFIXES.get(sid)
        if not suffix:
            return False
        base = os.path.join(CKPT_DIR, suffix)
        return (os.path.exists(os.path.join(base, "ckpt_latest", "adapter_config.json"))
                or os.path.exists(os.path.join(base, "adapter_model.safetensors")))

    all_steps = []
    for sid in STEP_ORDER:
        info   = steps.get(sid, {})
        status = info.get("status", "pending")
        # Remap: failed training step with partial output → "stopped" (not a real error)
        if status == "failed" and sid in TRAIN_STEP_IDS and _has_partial_output(sid):
            status = "stopped"
        all_steps.append({
            "id":     sid,
            "status": status,
            "ts":     info.get("ts"),
            "note":   info.get("note", ""),
            "phase":  PHASE_LABELS.get(sid, ""),
        })

    n_done    = sum(1 for s in all_steps if s["status"] in ("done", "stopped"))
    n_running = [s["id"] for s in all_steps if s["status"] == "running"]
    n_failed  = [s["id"] for s in all_steps if s["status"] == "failed"]
    n_stopped = [s["id"] for s in all_steps if s["status"] == "stopped"]
    total     = len(STEP_ORDER)

    # Most recent DB entry (gives a sense of what just finished)
    try:
        latest_runs = results_db.query_runs()[-3:]
    except Exception:
        latest_runs = []

    # ── Parse be_progress.log for live GBV sub-step progress ────────────────
    # be_progress.log is written by run_be_batch() when a GBV subprocess is
    # active.  It contains tqdm progress bars and "Block efficiency (...)"
    # result lines.  We extract:
    #   current_combo  — which (mode, K, T) combo + prompt progress is active
    #   n_be_done      — how many combos have printed their result line so far
    #   n_be_total     — total combos expected in this batch (from log header)
    #
    # IMPORTANT: Only read be_progress.log when an EVAL step is running.
    # Training steps (online_serve.py, train_qwen3.py) never write to this
    # log — they use evaluate_metrics() which calls speculative_step() directly.
    # Reading a stale log during training steps shows confusing Phase 1 data.
    _running_step_ids = {s["id"] for s in all_steps if s["status"] == "running"}
    _an_eval_is_running = bool(_running_step_ids - TRAIN_STEP_IDS - {"eagle_gen", "eagle_train"})
    current_combo = None
    n_be_done     = 0
    n_be_total    = None
    try:
        if _an_eval_is_running and os.path.exists(_BE_LOG):
            with open(_BE_LOG, encoding="utf-8", errors="replace") as f:
                log_lines = f.readlines()

            # Count finished combos
            for line in log_lines:
                if re.search(r"Block efficiency \(mode=", line):
                    n_be_done += 1

            # Total combos from header line written by run_be_batch print:
            #   "loading models once for 12 combo(s) (3 mode(s) x 2 K x 2 temp(s)) ..."
            for line in log_lines[:10]:
                m = re.search(r"loading models once for (\d+) combo", line)
                if m:
                    n_be_total = int(m.group(1))
                    break

            # Current tqdm line (last one seen, walking backwards):
            # Format GBV tqdm outputs:  "mode=gbv K=3 T=0.6:  17%|█▊  |  5/30 [00:12<00:59,  4.2it/s]"
            for line in reversed(log_lines[-40:]):
                m = re.search(
                    r"(mode=\w+[^\:]*?):\s+(\d+)%.*?\|\s*(\d+)/(\d+)", line)
                if m:
                    pct  = m.group(2)
                    done = m.group(3)
                    tot  = m.group(4)
                    label = m.group(1).strip()
                    current_combo = f"{label}  {done}/{tot} prompts ({pct}%)"
                    break

            # If tqdm line not found but some combos done, show last completed result
            if current_combo is None and n_be_done > 0:
                for line in reversed(log_lines):
                    m = re.search(
                        r"Block efficiency \(mode=(\w+), K=(\d+), T=([\d.]+)\):\s*([\d.]+)", line)
                    if m:
                        current_combo = (f"Last: mode={m.group(1)} K={m.group(2)} "
                                         f"T={m.group(3)} BE={float(m.group(4)):.3f}  "
                                         f"({n_be_done}/{n_be_total or '?'} done)")
                        break
    except Exception:
        pass

    return jsonify({
        "found": True,
        "config": os.path.basename(sf).replace("pipeline_state_","").replace(".json",""),
        "steps": all_steps,
        "n_done": n_done,
        "n_total": total,
        "running": n_running,
        "failed": n_failed,
        "stopped": n_stopped,
        "pct": round(n_done / total * 100, 1),
        "current_combo": current_combo,
        "n_be_done": n_be_done,
        "n_be_total": n_be_total,
        "latest_db_runs": [
            {"draft_label": r["draft_label"], "dataset": r["dataset"],
             "mode": r["mode"], "K": r["K"], "ts": r["ts"],
             "alpha_mean": r.get("alpha_mean"), "block_eff": r.get("block_eff"),
             "perplexity": r.get("perplexity")}
            for r in latest_runs
        ],
    })


@app.route("/api/log_tail")
def api_log_tail():
    """Return last N lines from pipeline_output.log (training + eval) or be_progress.log (eval detail).

    ?source=pipeline  (default) — pipeline_output.log: all training steps, health checks,
                                   step boundaries, val loss lines — the main live view.
    ?source=eval                 — be_progress.log: per-prompt GBV progress during eval.
    """
    n      = min(int(request.args.get("lines", 80)), 500)
    source = request.args.get("source", "pipeline")
    log_path = _PIPELINE_LOG if source != "eval" else _BE_LOG

    if not os.path.exists(log_path):
        other = _BE_LOG if source != "eval" else _PIPELINE_LOG
        fallback_exists = os.path.exists(other)
        return jsonify({
            "lines": [], "total_lines": 0, "found": False,
            "source": source,
            "msg": (
                f"{os.path.basename(log_path)} not created yet — "
                f"{'run experiment.py first' if source != 'eval' else 'starts when the first GBV eval batch runs'}."
                + (f"  Try ?source={'eval' if source != 'eval' else 'pipeline'}" if fallback_exists else "")
            ),
        })
    try:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
        return jsonify({
            "lines": all_lines[-n:],
            "total_lines": len(all_lines),
            "found": True,
            "source": source,
            "log_file": os.path.basename(log_path),
        })
    except Exception as e:
        return jsonify({"lines": [], "total_lines": 0, "found": False,
                        "source": source, "error": str(e)})


@app.route("/api/gpu_status")
def api_gpu_status():
    """Return current GPU VRAM usage (reads live from NVML/PyTorch)."""
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(0)
            used = total - free
            return jsonify({
                "name": torch.cuda.get_device_name(0),
                "free_gb":  round(free  / 1024**3, 2),
                "total_gb": round(total / 1024**3, 2),
                "used_gb":  round(used  / 1024**3, 2),
                "pct_used": round(used  / total * 100, 1),
            })
    except Exception:
        pass
    return jsonify({"name": "CPU-only", "free_gb": None, "total_gb": None,
                    "used_gb": None, "pct_used": None})


# ---------------------------------------------------------------------------
# HTML Dashboard
# ---------------------------------------------------------------------------

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SpecDist Results Dashboard</title>
<link rel="stylesheet" href="/static/bootstrap.min.css">
<script src="/static/plotly.min.js"></script>
<style>
  body { background: #f8f9fa; font-size: 14px; }
  /* Pipeline status bar — sticky across the top */
  #pipeline-bar {
    position: sticky; top: 0; z-index: 1050;
    background: #1a1d21; color: #e0e0e0;
    padding: 5px 16px; font-size: 12px;
    display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
    border-bottom: 1px solid #333;
  }
  /* Log panel — monospace scrollable area just below the status bar */
  #log-panel {
    position: sticky; top: 36px; z-index: 1040;
    background: #0d1117; color: #a8ff78;
    font-family: monospace; font-size: 11px;
    padding: 6px 16px; max-height: 220px; overflow-y: auto;
    white-space: pre; border-bottom: 2px solid #2ea043;
    display: none;
  }
  #pipeline-bar .step-badge {
    padding: 2px 8px; border-radius: 10px; font-weight: 600; font-size: 11px;
  }
  .badge-running  { background: #ffc107; color: #000; animation: pulse 1.4s infinite; }
  .badge-done     { background: #198754; color: #fff; }
  .badge-failed   { background: #dc3545; color: #fff; }
  .badge-stopped  { background: #e67e22; color: #fff; }
  .badge-pending  { background: #6c757d; color: #fff; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.55} }
  #progress-wrap { flex: 1; min-width: 160px; max-width: 240px; }
  #progress-bar-inner {
    height: 6px; border-radius: 3px; background: #0d6efd;
    transition: width .6s ease;
  }
  #progress-track { height: 6px; background: #444; border-radius: 3px; }
  .gpu-bar-wrap { display:flex; align-items:center; gap:6px; }
  #gpu-bar-track { width:80px; height:6px; background:#444; border-radius:3px; }
  #gpu-bar-fill  { height:6px; border-radius:3px; background:#fd7e14;
                   transition: width .6s ease; }
  #refresh-countdown { color:#aaa; font-size:11px; }
  #sidebar { background: #fff; border-right: 1px solid #dee2e6; min-height: 100vh;
             padding: 1rem; overflow-y: auto; width: 220px; flex-shrink: 0; }
  #main { flex: 1; padding: 1.5rem; overflow-y: auto; }
  .filter-section { margin-bottom: 1rem; }
  .filter-section label { font-weight: 600; font-size: 12px; color: #555;
                          text-transform: uppercase; letter-spacing: 0.5px; }
  .chip { display: inline-block; padding: 2px 8px; margin: 2px;
          border-radius: 12px; background: #e9ecef; cursor: pointer;
          font-size: 12px; user-select: none; border: 1px solid #ced4da; }
  .chip.active { background: #0d6efd; color: #fff; border-color: #0d6efd; }
  .chart-card { background: #fff; border-radius: 8px; padding: 1rem;
                box-shadow: 0 1px 4px rgba(0,0,0,.08); margin-bottom: 1.5rem; }
  .chart-card h6 { color: #333; font-weight: 600; margin-bottom: .75rem; }
  .stat-box { background: #fff; border-radius: 8px; padding: 1rem;
              box-shadow: 0 1px 4px rgba(0,0,0,.08); text-align: center; }
  .stat-box .val { font-size: 1.6rem; font-weight: 700; color: #0d6efd; }
  .stat-box .lbl { font-size: 11px; color: #888; text-transform: uppercase; }
  .confidence-badge { font-size: 11px; padding: 2px 6px; border-radius: 4px; }
  #axis-controls select { font-size: 12px; }
  .tab-pane { padding-top: 1rem; }
</style>
</head>
<body>

<!-- ===== Pipeline Status Bar ===== -->
<div id="pipeline-bar">
  <span style="font-weight:700;color:#fff">SpecDist</span>

  <!-- Current step badge -->
  <span id="bar-current-step" style="color:#ccc">loading...</span>

  <!-- Phase label -->
  <span id="bar-phase" style="color:#aaa;font-size:11px"></span>

  <!-- Sub-step: current GBV combo + prompt progress -->
  <span id="bar-current-combo"
        style="color:#7ec8e3;font-size:11px;max-width:300px;
               overflow:hidden;white-space:nowrap;text-overflow:ellipsis"
        title="Current combo within the GBV batch (from be_progress.log)"></span>

  <!-- Progress: n/18 steps -->
  <div id="progress-wrap">
    <div id="progress-track">
      <div id="progress-bar-inner" style="width:0%"></div>
    </div>
    <span id="bar-progress-text" style="font-size:10px;color:#aaa"></span>
  </div>

  <!-- GPU memory -->
  <div class="gpu-bar-wrap">
    <span style="color:#aaa;font-size:11px">GPU</span>
    <div id="gpu-bar-track"><div id="gpu-bar-fill" style="width:0%"></div></div>
    <span id="bar-gpu-text" style="font-size:11px;color:#aaa">—</span>
  </div>

  <!-- Latest DB entry -->
  <span id="bar-latest"
        style="color:#9ecce8;font-size:11px;max-width:260px;
               overflow:hidden;white-space:nowrap;text-overflow:ellipsis"></span>

  <!-- Auto-refresh countdown -->
  <span id="refresh-countdown" class="ms-auto">&#8635; 30s</span>

  <!-- Toggle buttons -->
  <button class="btn btn-sm btn-outline-light py-0 px-2"
          style="font-size:11px" onclick="toggleStepPanel()">Steps</button>
  <button class="btn btn-sm py-0 px-2" id="btn-logs"
          style="font-size:11px;background:#1a3a1a;color:#a8ff78;border:1px solid #2ea043"
          onclick="toggleLogPanel()"
          title="Show/hide live GBV subprocess output (be_progress.log)">&#128196; Logs</button>
</div>

<!-- Step detail panel (hidden by default) -->
<div id="step-panel" style="display:none;flex-direction:column;background:#2b2f36;color:#e0e0e0;
     font-size:11px;padding:8px 16px;border-bottom:1px solid #444">
  <!-- populated by JS -->
</div>

<!-- Live log panel (hidden by default, toggled by "Logs" button) -->
<div id="log-panel"><!-- populated by JS --></div>

<div class="d-flex" style="height:calc(100vh - 36px)"  id="main-flex">

<!-- Sidebar -->
<div id="sidebar" style="height:100%;overflow-y:auto">
  <div class="fw-bold mb-3" style="font-size:15px">SpecDist Filters</div>

  <div class="filter-section">
    <label>Model Family</label>
    <div id="f-model_pairs"></div>
    <div style="font-size:11px;color:#ed8936;margin-top:4px;font-weight:600">
      Results are only comparable within the same model family.
    </div>
  </div>

  <!-- HW tier: secondary annotation only -->
  <div class="filter-section">
    <label style="color:#718096">HW Tier <small>(annotation only)</small></label>
    <div id="f-hw_tier">
      <span class="chip hw-tier-chip active" data-tier="laptop"
            style="border-color:#a0aec0" onclick="toggleTierChip(this)" title="smoke tests">laptop</span>
      <span class="chip hw-tier-chip active" data-tier="colab_lite"
            style="border-color:#9f7aea" onclick="toggleTierChip(this)" title="trend detection, T4">colab_lite</span>
      <span class="chip hw-tier-chip active" data-tier="colab"
            style="border-color:#ed8936" onclick="toggleTierChip(this)" title="publishable results, T4">colab</span>
      <span class="chip hw-tier-chip active" data-tier="a100"
            style="border-color:#48bb78" onclick="toggleTierChip(this)" title="paper quality, A100">a100</span>
    </div>
  </div>

  <div class="filter-section">
    <label>Distillation Method</label>
    <!-- draft_label = loss name used during training; loss_name is always identical, not shown separately -->
    <div id="f-draft_label"></div>
  </div>
  <!-- f-loss_name hidden: always same as draft_label in this pipeline -->
  <div id="f-loss_name" style="display:none"></div>
  <div class="filter-section">
    <label>Dataset</label>
    <div id="f-dataset"></div>
  </div>
  <div class="filter-section">
    <label>Verifier Mode</label>
    <div id="f-mode-tree"></div>
  </div>
  <!-- alpha/perplexity chips hidden — charts filter by mode internally;
       these row-type chips added no practical value for chart analysis.
       The divs are kept in the DOM so buildFilterChips() can still populate
       them (needed to keep chip-based mode filtering consistent). -->
  <div id="f-mode-scalar" style="display:none"></div>
  <div class="filter-section">
    <label>K (tree width)</label>
    <div id="f-K"></div>
  </div>
  <div class="filter-section">
    <label>Temperature</label>
    <div id="f-temperature"></div>
  </div>
  <div class="filter-section">
    <label>Train Steps</label>
    <div id="f-train_steps"></div>
  </div>
  <div class="filter-section">
    <label>Experiment Tag</label>
    <div id="f-experiment_tag"></div>
    <input type="text" id="etag-search" class="form-control form-control-sm mt-1"
           placeholder="search tags…" oninput="filterTagChips(this.value)">
  </div>

  <hr>
  <button class="btn btn-primary btn-sm w-100" onclick="applyFilters()">
    Apply Filters
  </button>
  <button class="btn btn-outline-secondary btn-sm w-100 mt-1" onclick="clearFilters()">
    Clear All
  </button>
  <hr>
  <div id="run-count" class="text-muted" style="font-size:11px"></div>
</div>

<!-- Main -->
<div id="main">
  <div class="d-flex align-items-center mb-3 gap-3">
    <h5 class="mb-0">SpecDist Results Dashboard</h5>
    <span class="badge bg-secondary" id="badge-runs">0 runs</span>
    <button class="btn btn-sm btn-outline-primary ms-auto"
            onclick="loadData(); updatePipelineStatus(); refreshLogPanel(); _countdown=30;">
      Refresh now
    </button>
  </div>

  <!-- Summary stats -->
  <div class="row g-2 mb-3" id="stat-row"></div>

  <!-- ===== 5 Research-Question Tabs ===== -->
  <ul class="nav nav-tabs" id="myTab">
    <li class="nav-item">
      <a class="nav-link active" data-bs-toggle="tab" href="#tab-training"
         title="Is training stable? Which loss converges best?">
        📈 Training
      </a>
    </li>
    <li class="nav-item">
      <a class="nav-link" data-bs-toggle="tab" href="#tab-results"
         title="Which model wins? Does distillation improve block efficiency?">
        🏆 Results
      </a>
    </li>
    <li class="nav-item">
      <a class="nav-link" data-bs-toggle="tab" href="#tab-robust"
         title="Is the improvement consistent across verifiers, K, temperatures, and datasets?">
        🔄 Robustness
      </a>
    </li>
    <li class="nav-item">
      <a class="nav-link" data-bs-toggle="tab" href="#tab-analysis"
         title="Understanding the mechanisms — alpha, throughput, hyperparameter sensitivity">
        🔬 Analysis
      </a>
    </li>
    <li class="nav-item">
      <a class="nav-link" data-bs-toggle="tab" href="#tab-data"
         title="All eval runs table + custom pivot cross-tabulation">
        📁 Data
      </a>
    </li>
  </ul>

  <div class="tab-content">

    <!-- ══════════════════════════════════════════════════════════════
         TAB 1: TRAINING
         Questions: Is training stable? Which loss converges best?
                    Is there overfitting? Does eval_be improve?
         ══════════════════════════════════════════════════════════════ -->
    <div class="tab-pane fade show active" id="tab-training">

      <!-- Context banner -->
      <div class="alert alert-secondary py-2 px-3 mb-3" style="font-size:12px;border-left:4px solid #6c757d">
        <strong>What to look for:</strong>
        One card per distillation method. Solid = train, dashed = val.
        Train loss should fall steadily; val should track it — if val rises while train falls that's overfitting (red badge).
        EBE/online loss values are negative (log-prob space) — lower is better. Online curves are shown in the Online Health chart below.
      </div>

      <!-- Overfitting banner (shown when triggered) -->
      <div id="val-redflag-banner"
           style="display:none;background:#dc3545;color:#fff;padding:8px 12px;
                  border-radius:6px;margin-bottom:10px;font-weight:600;font-size:0.9em">
      </div>

      <!-- Training + val loss — one card per distillation method -->
      <div id="chart-training-grid"
           style="display:grid;grid-template-columns:repeat(auto-fill,minmax(480px,1fr));gap:16px">
        <!-- populated by loadTrainingCurves() -->
      </div>

      <!-- Online health cards — one per online label, populated by renderOnlineHealth() -->
      <div id="chart-online-health-grid"
           style="display:grid;grid-template-columns:repeat(auto-fill,minmax(480px,1fr));gap:16px">
        <!-- populated by renderOnlineHealth() -->
      </div>

    </div>

    <!-- ══════════════════════════════════════════════════════════════
         TAB 2: RESULTS
         Questions: Which model has the best block efficiency?
                    Does distillation help vs untrained baseline?
                    Quality vs Speed tradeoff (Pareto)?
         ══════════════════════════════════════════════════════════════ -->
    <div class="tab-pane fade" id="tab-results">

      <div class="alert alert-secondary py-2 px-3 mb-3" style="font-size:12px;border-left:4px solid #0d6efd">
        <strong>Main paper table:</strong>
        The heatmap below is rows=verifier mode, cols=distillation loss — this is the core contribution table.
        Gain chart shows % improvement over the untrained baseline. Pareto shows whether speed gain comes at a quality cost.
      </div>

      <!-- Chart A: Loss × Verifier Heatmap — the paper's main result table -->
      <div class="chart-card">
        <h6>Block Efficiency — Loss &times; Verifier Heatmap
          <small class="text-muted ms-2">Paper Table 1: each cell = mean BE for that (loss, verifier) pair</small>
        </h6>
        <div class="d-flex gap-2 mb-2 flex-wrap">
          <label class="mb-0">K:</label>
          <select id="sel-k-hm" class="form-select form-select-sm" style="width:80px">
            <option>3</option><option>5</option><option>1</option>
          </select>
          <label class="mb-0 ms-2">Dataset:</label>
          <select id="sel-dataset-hm" class="form-select form-select-sm" style="width:140px"></select>
          <label class="mb-0 ms-2">Temperature:</label>
          <select id="sel-temp-hm" class="form-select form-select-sm" style="width:90px">
            <option value="">All</option>
          </select>
        </div>
        <div id="chart-interaction-hm" style="height:360px"></div>
      </div>

      <!-- Chart B: Gain over baseline (%) -->
      <div class="chart-card">
        <h6>Block Efficiency Gain over Untrained Baseline (%)
          <small class="text-muted ms-2">
            Positive = distillation helps. The main claim: EBE + tree-verifier &gt; KL + tree-verifier.
          </small>
        </h6>
        <div class="d-flex gap-2 mb-2 flex-wrap">
          <label class="mb-0">K:</label>
          <select id="sel-k-gain" class="form-select form-select-sm" style="width:80px">
            <option>3</option><option>5</option><option>1</option>
          </select>
          <label class="mb-0 ms-2">Dataset:</label>
          <select id="sel-dataset-gain" class="form-select form-select-sm" style="width:140px"></select>
          <label class="mb-0 ms-2">Temperature:</label>
          <select id="sel-temp-gain" class="form-select form-select-sm" style="width:90px">
            <option value="">All</option>
          </select>
        </div>
        <div id="chart-gain" style="height:360px"></div>
      </div>

      <!-- Pareto frontier: quality vs speed -->
      <div class="chart-card">
        <h6>Quality vs Speed — Pareto Frontier
          <small class="text-muted ms-2">
            X = task score (quality preservation) · Y = block efficiency (SD speedup).
            Top-right corner = best. Each point = one draft model.
            Goal: points above and to the right of the baseline diamond.
          </small>
        </h6>
        <div class="d-flex gap-2 mb-2 flex-wrap">
          <label class="mb-0">Mode:</label>
          <select id="eff-pareto-mode" class="form-select form-select-sm" style="width:140px">
            <option>traversal</option><option>gbv</option><option>specinfer</option>
          </select>
          <label class="mb-0 ms-2">K:</label>
          <select id="eff-pareto-k" class="form-select form-select-sm" style="width:80px">
            <option>3</option><option>5</option>
          </select>
        </div>
        <div id="chart-pareto" style="height:380px"></div>
      </div>

      <!-- Normalised % improvement (per verifier mode) -->
      <div class="chart-card">
        <h6>Relative BE Improvement over Baseline — per verifier mode
          <small class="text-muted ms-2">
            BE<sub>model</sub> / BE<sub>baseline</sub> − 1 (%).
            Green = better than no training. Red = regression.
          </small>
        </h6>
        <div class="d-flex gap-2 mb-2 flex-wrap">
          <label class="mb-0">K:</label>
          <select id="eff-norm-k" class="form-select form-select-sm" style="width:80px">
            <option>3</option><option>5</option><option>1</option>
          </select>
          <label class="mb-0 ms-2">Temperature:</label>
          <select id="eff-norm-t" class="form-select form-select-sm" style="width:90px">
            <option value="">All</option>
          </select>
        </div>
        <div id="chart-be-norm" style="height:360px"></div>
      </div>

    </div>

    <!-- ══════════════════════════════════════════════════════════════
         TAB 3: ROBUSTNESS
         Questions: Does the improvement hold across all verifiers?
                    Does it hold at different K (tree widths)?
                    Does it hold at different temperatures?
                    Does it generalise to other datasets? (Phase 4)
         ══════════════════════════════════════════════════════════════ -->
    <div class="tab-pane fade" id="tab-robust">

      <div class="alert alert-secondary py-2 px-3 mb-3" style="font-size:12px;border-left:4px solid #198754">
        <strong>Robustness checks:</strong>
        A result is only publishable if it holds across multiple settings.
        Check: same ranking across all 4 verifiers? Consistent across K=1,3,5?
        Stable at T=0.6 and T=1.0? Generalises beyond GSM8K (Phase 4)?
      </div>

      <!-- Acceptance Rate — derived from BE, no extra eval needed -->
      <div class="chart-card">
        <h6>Draft Token Acceptance Rate by Loss Type
          <small class="text-muted ms-2">α_eff = (BE−1)/K — fraction of draft slots accepted per round. Derived from existing BE data, no extra eval needed.</small>
        </h6>
        <div class="d-flex gap-2 mb-2 flex-wrap align-items-center">
          <label class="mb-0">Group by:</label>
          <select id="sel-arate-group" class="form-select form-select-sm" style="width:120px">
            <option value="mode">verifier mode</option>
            <option value="K">K (tree width)</option>
          </select>
          <label class="mb-0 ms-2">Show:</label>
          <select id="sel-arate-show" class="form-select form-select-sm" style="width:160px">
            <option value="abs">absolute α_eff</option>
            <option value="gain">gain vs baseline (%)</option>
          </select>
        </div>
        <div id="chart-accept-rate" style="height:400px"></div>
        <div id="chart-accept-rate-k" style="height:360px"></div>
      </div>

      <!-- Mode comparison -->
      <div class="chart-card">
        <h6>Block Efficiency by Verifier Mode — all loss types compared
          <small class="text-muted ms-2">If ranking is consistent across modes, the result is robust to verifier choice</small>
        </h6>
        <div class="d-flex gap-2 mb-2 flex-wrap">
          <label class="mb-0">K:</label>
          <select id="sel-k-mode" class="form-select form-select-sm" style="width:80px">
            <option>3</option><option>1</option><option>5</option>
          </select>
          <label class="mb-0 ms-2">Dataset:</label>
          <select id="sel-dataset-mode" class="form-select form-select-sm" style="width:140px"></select>
          <label class="mb-0 ms-2">Temperature:</label>
          <select id="sel-temp-mode" class="form-select form-select-sm" style="width:90px">
            <option value="">All</option>
          </select>
        </div>
        <div id="chart-mode" style="height:400px"></div>
      </div>

      <!-- BE vs K -->
      <div class="chart-card">
        <h6>Block Efficiency vs K (tree width) — does it scale as expected?
          <small class="text-muted ms-2">BE should rise with K — steeper rise = better draft model</small>
        </h6>
        <div class="d-flex gap-2 align-items-center mb-2 flex-wrap">
          <label class="mb-0">Mode:</label>
          <select id="sel-mode-be" class="form-select form-select-sm" style="width:140px">
            <option>specinfer</option><option>gbv</option>
            <option>traversal</option><option>bv</option>
          </select>
          <label class="mb-0 ms-2">Dataset:</label>
          <select id="sel-dataset-be" class="form-select form-select-sm" style="width:140px"></select>
          <label class="mb-0 ms-2">Temperature:</label>
          <select id="sel-temp-be" class="form-select form-select-sm" style="width:90px">
            <option value="">All</option>
          </select>
        </div>
        <div id="chart-be-k" style="height:380px"></div>
      </div>
      <div class="chart-card">
        <h6>BE vs K — all modes (subplots)</h6>
        <div id="chart-be-k-modes" style="height:440px"></div>
      </div>

      <!-- Temperature robustness -->
      <div class="chart-card">
        <h6>Temperature Robustness — BE at T=0.6 vs T=1.0
          <small class="text-muted ms-2">Lines should stay parallel — if they cross, the ranking depends on temperature</small>
        </h6>
        <div class="d-flex gap-2 mb-2 flex-wrap">
          <label class="mb-0">Mode:</label>
          <select id="sel-mode-templine" class="form-select form-select-sm" style="width:140px">
            <option>traversal</option><option>gbv</option><option>specinfer</option>
          </select>
          <label class="mb-0 ms-2">K:</label>
          <select id="sel-k-templine" class="form-select form-select-sm" style="width:80px">
            <option>3</option><option>5</option><option>1</option>
          </select>
          <label class="mb-0 ms-2">Dataset:</label>
          <select id="sel-dataset-templine" class="form-select form-select-sm" style="width:140px"></select>
        </div>
        <div id="chart-temp-line" style="height:380px"></div>
      </div>
      <div class="chart-card">
        <h6>Temperature Gain vs Baseline — distillation gain at each temperature
          <small class="text-muted ms-2">Positive at both temperatures = robust to sampling temperature</small>
        </h6>
        <div class="d-flex gap-2 mb-2 flex-wrap">
          <label class="mb-0">Mode:</label>
          <select id="sel-mode-tgain" class="form-select form-select-sm" style="width:140px">
            <option>traversal</option><option>gbv</option><option>specinfer</option>
          </select>
          <label class="mb-0 ms-2">K:</label>
          <select id="sel-k-tgain" class="form-select form-select-sm" style="width:80px">
            <option>3</option><option>5</option>
          </select>
          <label class="mb-0 ms-2">Dataset:</label>
          <select id="sel-dataset-tgain" class="form-select form-select-sm" style="width:140px"></select>
        </div>
        <div id="chart-temp-gain" style="height:360px"></div>
      </div>

      <!-- Dataset robustness (Phase 4) -->
      <div class="chart-card">
        <h6>Dataset Robustness — BE across all tasks (Phase 4: humaneval · math500 · mtbench · alpaca)
          <small class="text-muted ms-2">Populated after Phase 4 multi-dataset evals complete</small>
        </h6>
        <div class="d-flex gap-2 mb-2 flex-wrap">
          <label class="mb-0">Mode:</label>
          <select id="sel-mode-rob" class="form-select form-select-sm" style="width:140px">
            <option>traversal</option><option>gbv</option><option>specinfer</option>
          </select>
          <label class="mb-0 ms-2">K:</label>
          <select id="sel-k-rob" class="form-select form-select-sm" style="width:80px">
            <option>3</option><option>5</option>
          </select>
          <label class="mb-0 ms-2">Temperature:</label>
          <select id="sel-temp-rob" class="form-select form-select-sm" style="width:90px">
            <option value="">All</option>
          </select>
        </div>
        <div id="chart-robustness" style="height:360px"></div>
      </div>

    </div>

    <!-- ══════════════════════════════════════════════════════════════
         TAB 4: ANALYSIS
         Questions: Why does it work? What is the mechanism?
                    How much of BE gain comes from α vs tree bonus?
                    What are the latency/throughput numbers?
                    Which hyperparameters matter?
         ══════════════════════════════════════════════════════════════ -->
    <div class="tab-pane fade" id="tab-analysis">

      <div class="alert alert-secondary py-2 px-3 mb-3" style="font-size:12px;border-left:4px solid #fd7e14">
        <strong>Mechanism analysis:</strong>
        BE = f(α, K, verifier). The theory curve shows the sequential-SD ceiling for a given α.
        Points above the curve = additional gain from the tree structure.
        Alpha chart shows the raw acceptance rate shift from training.
        Throughput numbers translate BE into wall-clock speedup.
      </div>

      <!-- BE vs Alpha + theory curve -->
      <div class="chart-card">
        <h6>Block Efficiency vs α — measured points overlaid on sequential-SD theory
          <small class="text-muted ms-2">
            BE<sub>theory</sub>(α, K) = (1 − α<sup>K+1</sup>) / (1 − α).
            Points above the grey curve = tree bonus on top of α improvement.
          </small>
        </h6>
        <div class="d-flex gap-2 mb-2 flex-wrap">
          <label class="mb-0">K:</label>
          <select id="eff-k" class="form-select form-select-sm" style="width:80px">
            <option>3</option><option>4</option><option>5</option><option>1</option>
          </select>
          <label class="mb-0 ms-2">Mode:</label>
          <select id="eff-mode" class="form-select form-select-sm" style="width:140px">
            <option>traversal</option><option>gbv</option><option>specinfer</option><option>bv</option>
          </select>
        </div>
        <div id="chart-eff-decomp" style="height:400px"></div>
      </div>

      <!-- Alpha breakdown -->
      <div class="chart-card">
        <h6>Token Acceptance Rate (α) — how much does training shift the draft distribution?
          <small class="text-muted ms-2">Higher α = draft tokens accepted more often = higher BE ceiling</small>
        </h6>
        <div id="chart-alpha" style="height:380px"></div>
      </div>
      <div class="chart-card">
        <h6>α by Dataset — is the acceptance rate consistent across tasks?</h6>
        <div id="chart-alpha-dataset" style="height:320px"></div>
      </div>

      <!-- Per-category alpha -->
      <div class="chart-card">
        <h6>α by Category (diverse50) — which task types benefit most from distillation?</h6>
        <div id="chart-category" style="height:380px"></div>
      </div>

      <!-- Throughput & latency -->
      <div class="chart-card">
        <h6>Throughput (tok/s) — wall-clock speedup from speculative decoding + distillation</h6>
        <div id="chart-throughput" style="height:340px"></div>
      </div>
      <div class="chart-card">
        <h6>α vs Throughput — does higher acceptance directly translate to faster generation?</h6>
        <div id="chart-scatter" style="height:320px"></div>
      </div>

      <!-- Hyperparameter sensitivity -->
      <div class="chart-card">
        <h6>Hyperparameter Sensitivity — which parameters most affect the outcome?
          <small class="text-muted ms-2">Useful for ablation design: vary X, measure Y, group by Z</small>
        </h6>
        <div class="d-flex gap-3 mb-2 align-items-center flex-wrap">
          <div>
            <label class="form-label mb-0" style="font-size:12px">X axis</label>
            <select id="sens-x" class="form-select form-select-sm">
              <option value="K">K (tree width)</option>
              <option value="temperature">Temperature</option>
              <option value="train_steps">Train Steps</option>
              <option value="learning_rate">Learning Rate</option>
            </select>
          </div>
          <div>
            <label class="form-label mb-0" style="font-size:12px">Y axis</label>
            <select id="sens-y" class="form-select form-select-sm">
              <option value="block_eff">Block Efficiency</option>
              <option value="alpha_mean">Alpha</option>
              <option value="throughput">Throughput (tok/s)</option>
              <option value="ms_per_tok">Latency (ms/tok)</option>
            </select>
          </div>
          <div>
            <label class="form-label mb-0" style="font-size:12px">Group by</label>
            <select id="sens-group" class="form-select form-select-sm">
              <option value="draft_label">Distillation Method</option>
              <option value="mode">Verifier Mode</option>
              <option value="dataset">Dataset</option>
            </select>
          </div>
          <button class="btn btn-sm btn-primary mt-3" onclick="renderSensitivity()">Plot</button>
        </div>
        <div id="chart-sensitivity" style="height:380px"></div>
      </div>

    </div>

    <!-- ══════════════════════════════════════════════════════════════
         TAB 5: DATA
         All eval run rows + custom pivot cross-tabulations
         ══════════════════════════════════════════════════════════════ -->
    <div class="tab-pane fade" id="tab-data">

      <!-- All runs table -->
      <div class="chart-card">
        <div class="d-flex justify-content-between align-items-center mb-2">
          <h6 class="mb-0">All Eval Runs</h6>
          <button class="btn btn-sm btn-outline-secondary" onclick="exportCSV()">Export CSV</button>
        </div>
        <div class="table-responsive" style="max-height:480px;overflow-y:auto">
          <table class="table table-sm table-hover table-bordered" id="runs-table">
            <thead class="table-light sticky-top"></thead>
            <tbody></tbody>
          </table>
        </div>
      </div>

      <!-- Pivot builder -->
      <div class="chart-card">
        <h6 class="mb-3">Custom Pivot Report
          <small class="text-muted ms-2">
            Build any cross-tabulation from the current filtered runs.
            Default config reproduces the paper's main result table.
          </small>
        </h6>
        <div class="row g-2 align-items-end mb-3">
          <div class="col-auto">
            <label class="form-label mb-1" style="font-size:12px;font-weight:600">Rows</label>
            <select id="piv-row" class="form-select form-select-sm" style="width:140px">
              <option value="draft_label">Distillation Method</option>
              <option value="dataset">Dataset</option>
              <option value="mode">Verifier Mode</option>
              <option value="K">K (tree width)</option>
              <option value="temperature">Temperature</option>
              <option value="experiment_tag">Experiment Tag</option>
            </select>
          </div>
          <div class="col-auto">
            <label class="form-label mb-1" style="font-size:12px;font-weight:600">Columns</label>
            <select id="piv-col" class="form-select form-select-sm" style="width:140px">
              <option value="mode">Verifier Mode</option>
              <option value="dataset">Dataset</option>
              <option value="K">K (tree width)</option>
              <option value="draft_label">Distillation Method</option>
              <option value="temperature">Temperature</option>
              <option value="experiment_tag">Experiment Tag</option>
            </select>
          </div>
          <div class="col-auto">
            <label class="form-label mb-1" style="font-size:12px;font-weight:600">Value</label>
            <select id="piv-val" class="form-select form-select-sm" style="width:160px">
              <option value="block_eff">Block Efficiency</option>
              <option value="alpha_mean">Alpha (accept rate)</option>
              <option value="throughput">Throughput (tok/s)</option>
              <option value="ms_per_tok">Latency (ms/tok)</option>
              <option value="peak_vram_mb">Peak VRAM (MB)</option>
              <option value="alpha_ci95">Alpha CI-95</option>
            </select>
          </div>
          <div class="col-auto">
            <label class="form-label mb-1" style="font-size:12px;font-weight:600">Aggregate</label>
            <select id="piv-agg" class="form-select form-select-sm" style="width:100px">
              <option value="mean">Mean</option>
              <option value="max">Max</option>
              <option value="min">Min</option>
              <option value="count">Count</option>
            </select>
          </div>
          <div class="col-auto">
            <button class="btn btn-primary btn-sm" onclick="renderPivot()">Build</button>
            <button class="btn btn-outline-secondary btn-sm ms-1" onclick="exportPivotCSV()">CSV</button>
          </div>
        </div>
        <div id="pivot-info" class="text-muted mb-2" style="font-size:12px"></div>
        <div class="table-responsive">
          <table class="table table-sm table-bordered table-hover" id="pivot-table">
            <thead></thead><tbody></tbody>
          </table>
        </div>
        <p class="text-muted mt-2" style="font-size:11px">
          Default (Loss Type × Verifier Mode, Block Efficiency, Mean) reproduces the paper's main result table.
          Use Count to verify data coverage before reporting numbers.
        </p>
      </div>

    </div>

  </div><!-- tab-content -->
</div><!-- main -->
</div><!-- flex wrapper -->

<script src="/static/bootstrap.bundle.min.js"></script>
<script>
// ---- State ----
let ALL_RUNS = [];
let DIMS = {};
let ACTIVE_FILTERS = {};
let _refreshTimer = null;
let _countdown = 30;
let _stepPanelOpen = false;
let _logPanelOpen  = false;

// HW_TIER_FILTER: Set of selected tiers. Default = all four selected.
let HW_TIER_FILTER = new Set(['laptop', 'colab_lite', 'colab', 'a100']);

// MODEL_PAIR_FILTER: Set of "draft||target" composite keys.
// null = all pairs shown; a non-null Set restricts to selected pairs only.
// Key format: draft_path + "||" + target_path
let MODEL_PAIR_FILTER = null;

// ---- Step descriptions (shown as tooltips and in panel) ----
// MUST stay in sync with experiment.py build_steps() step IDs.
const STEP_DESC = {
  // Phase 1
  'eval_baseline_gsm8k':    'Ph 1 — Baseline: Untrained draft model — reference point all trained models are compared against',
  // Phase 2 — offline losses
  'train_kl_gsm8k':         'Ph 2 — Training: Train forward-KL distillation (DistillSpec baseline)',
  'merge_kl_gsm8k':         'Ph 2 — Training: Merge KL LoRA into base model',
  'train_ebe_gsm8k':        'Ph 2 — Training: Train offline EBE, multi-token cumprod (ablation — weak signal offline)',
  'merge_ebe_gsm8k':        'Ph 2 — Training: Merge EBE LoRA into base model',
  'train_ebe_single_gsm8k': 'Ph 2 — Training: Train EBE-single = −mean(α), no cumprod (ablation: tests if product structure is the bottleneck)',
  'merge_ebe_single_gsm8k': 'Ph 2 — Training: Merge EBE-single LoRA',
  'train_rev_kl_gsm8k':     'Ph 2 — Training: Train reverse-KL (mode-seeking ablation)',
  'merge_rev_kl_gsm8k':     'Ph 2 — Training: Merge rev-KL LoRA',
  'train_jsd_gsm8k':        'Ph 2 — Training: Train JSD (symmetric ablation)',
  'merge_jsd_gsm8k':        'Ph 2 — Training: Merge JSD LoRA',
  'train_l1_gsm8k':         'Ph 2 — Training: Train L1 (total-variation ablation)',
  'merge_l1_gsm8k':         'Ph 2 — Training: Merge L1 LoRA',
  // Phase 2 — online
  'online_adapt_gsm8k':     'Ph 2 — Training: Online forward-KL (OSD baseline — KL at rejected positions from live SD)',
  'merge_online_gsm8k':     'Ph 2 — Training: Merge online-KL LoRA',
  'online_ebe_adapt_gsm8k':        'Ph 2 — Training: Online EBE block-level (NOVEL — cumprod at live rejected positions)',
  'merge_online_ebe_gsm8k':        'Ph 2 — Training: Merge online-EBE LoRA',
  'online_ebe_single_adapt_gsm8k': 'Ph 2 — Training: Online EBE-single = −mean(α) at rejected positions (ablation vs block-level)',
  'merge_online_ebe_single_gsm8k': 'Ph 2 — Training: Merge online-EBE-single LoRA',
  // Phase 3
  'eval_kl_gsm8k':                 'Ph 3 — GSM8K Eval: Eval KL draft — DistillSpec baseline result',
  'eval_ebe_gsm8k':                'Ph 3 — GSM8K Eval: Eval offline-EBE (multi-token) draft',
  'eval_ebe_single_gsm8k':         'Ph 3 — GSM8K Eval: Eval offline EBE-single draft — ablation: does removing cumprod help?',
  'eval_rev_kl_gsm8k':             'Ph 3 — GSM8K Eval: Eval rev-KL draft',
  'eval_jsd_gsm8k':                'Ph 3 — GSM8K Eval: Eval JSD draft',
  'eval_l1_gsm8k':                 'Ph 3 — GSM8K Eval: Eval L1 draft',
  'eval_online_gsm8k':             'Ph 3 — GSM8K Eval: Eval online-KL draft (OSD baseline)',
  'eval_online_ebe_gsm8k':         'Ph 3 — GSM8K Eval: Eval online-EBE (block-level) — KEY result vs online-KL',
  'eval_online_ebe_single_gsm8k':  'Ph 3 — GSM8K Eval: Eval online-EBE-single — ablation: block-level vs single-token online',
  // Phase 4
  'eval_baseline_all':             'Ph 4 — Multi-DS: Baseline on HumanEval, MATH-500, MTBench, Alpaca',
  'eval_kl_all':                   'Ph 4 — Multi-DS: KL draft cross-domain',
  'eval_ebe_all':                  'Ph 4 — Multi-DS: Offline-EBE (multi-token) draft cross-domain',
  'eval_ebe_single_all':           'Ph 4 — Multi-DS: EBE-single draft cross-domain',
  'eval_rev_kl_all':               'Ph 4 — Multi-DS: Rev-KL draft cross-domain',
  'eval_jsd_all':                  'Ph 4 — Multi-DS: JSD draft cross-domain',
  'eval_l1_all':                   'Ph 4 — Multi-DS: L1 draft cross-domain',
  'eval_online_all':               'Ph 4 — Multi-DS: Online-KL draft cross-domain',
  'eval_online_ebe_all':           'Ph 4 — Multi-DS: Online-EBE (block-level) cross-domain',
  'eval_online_ebe_single_all':    'Ph 4 — Multi-DS: Online-EBE-single cross-domain',
  // Phase 5
  'eagle_gen':   'Ph 5 — EAGLE: Generate EAGLE training data from target model hidden states',
  'eagle_train': 'Ph 5 — EAGLE: Train 1-layer EAGLE head on hidden states',
  'eagle_eval':  'Ph 5 — EAGLE: Eval EAGLE draft — external baseline for block-efficiency comparison',
  // Phase 6 — Tree Training
  'train_kl_tree_gsm8k':       'Ph 6 — Tree Training: On-policy forward-KL at each tree node (universal baseline)',
  'merge_kl_tree_gsm8k':       'Ph 6 — Tree Training: Merge kl_tree LoRA',
  'train_bv_tree_gsm8k':       'Ph 6 — Tree Training: bv_tree — surrogate for bv_verify block-acceptance integral',
  'merge_bv_tree_gsm8k':       'Ph 6 — Tree Training: Merge bv_tree LoRA',
  'train_gbv_tree_gsm8k':      'Ph 6 — Tree Training: gbv_tree — surrogate for gbv_verify (bv + q_skew)',
  'merge_gbv_tree_gsm8k':      'Ph 6 — Tree Training: Merge gbv_tree LoRA',
  'train_trav_tree_gsm8k':     'Ph 6 — Tree Training: traversal_tree — leaf-weight surrogate for traversal_verify',
  'merge_trav_tree_gsm8k':     'Ph 6 — Tree Training: Merge traversal_tree LoRA',
  'train_ebe_tree_gsm8k':      'Ph 6 — Tree Training: ebe_tree — on-policy EBE ablation (off-policy mismatch test)',
  'merge_ebe_tree_gsm8k':      'Ph 6 — Tree Training: Merge ebe_tree LoRA',
  'train_rev_kl_tree_gsm8k':   'Ph 6 — Tree Training: rev_kl_tree — mode-seeking on-policy KL',
  'merge_rev_kl_tree_gsm8k':   'Ph 6 — Tree Training: Merge rev_kl_tree LoRA',
  'train_jsd_tree_gsm8k':      'Ph 6 — Tree Training: jsd_tree — symmetric JSD on tree nodes',
  'merge_jsd_tree_gsm8k':      'Ph 6 — Tree Training: Merge jsd_tree LoRA',
  // Verifier-aligned OT tree losses (added 2026-05)
  'train_naive_tree_gsm8k':    'Ph 6 — Tree Training: naive_tree — aligned with Chen/Leviathan naive OT verifier',
  'merge_naive_tree_gsm8k':    'Ph 6 — Tree Training: Merge naive_tree LoRA',
  'train_nss_tree_gsm8k':      'Ph 6 — Tree Training: nss_tree — aligned with NSS verifier (Σ p·(1−(1−q)^K))',
  'merge_nss_tree_gsm8k':      'Ph 6 — Tree Training: Merge nss_tree LoRA',
  'train_si_tree_gsm8k':       'Ph 6 — Tree Training: specinfer_tree — aligned with SpecInfer K-iter rejection α',
  'merge_si_tree_gsm8k':       'Ph 6 — Tree Training: Merge specinfer_tree LoRA',
  'train_st_tree_gsm8k':       'Ph 6 — Tree Training: spectr_tree — aligned with SpecTr K-SEQ α (ρ detached)',
  'merge_st_tree_gsm8k':       'Ph 6 — Tree Training: Merge spectr_tree LoRA',
  'train_khisti_tree_gsm8k':   'Ph 6 — Tree Training: khisti_tree — aligned with Khisti canonical decomp (LP-free surrogate)',
  'merge_khisti_tree_gsm8k':   'Ph 6 — Tree Training: Merge khisti_tree LoRA',
  'online_kl_tree_adapt_gsm8k':  'Ph 6 — Tree Training: Online adaptation with kl_tree loss (no replay buffer)',
  'merge_online_kl_tree_gsm8k':  'Ph 6 — Tree Training: Merge online_kl_tree LoRA',
  'online_ebe_tree_adapt_gsm8k': 'Ph 6 — Tree Training: Online adaptation with ebe_tree loss',
  'merge_online_ebe_tree_gsm8k': 'Ph 6 — Tree Training: Merge online_ebe_tree LoRA',
  'eval_kl_tree_gsm8k':        'Ph 6 — Tree GSM8K Eval: Eval kl_tree with bv+gbv+traversal verifiers',
  'eval_bv_tree_gsm8k':        'Ph 6 — Tree GSM8K Eval: Eval bv_tree — H5: does it beat kl on bv_verify?',
  'eval_gbv_tree_gsm8k':       'Ph 6 — Tree GSM8K Eval: Eval gbv_tree — H5: does it beat kl on gbv_verify?',
  'eval_trav_tree_gsm8k':      'Ph 6 — Tree GSM8K Eval: Eval traversal_tree — H5: does it beat kl on traversal?',
  'eval_ebe_tree_gsm8k':       'Ph 6 — Tree GSM8K Eval: Eval ebe_tree — on-policy EBE vs flat EBE',
  'eval_rev_kl_tree_gsm8k':    'Ph 6 — Tree GSM8K Eval: Eval rev_kl_tree',
  'eval_jsd_tree_gsm8k':       'Ph 6 — Tree GSM8K Eval: Eval jsd_tree',
  'eval_naive_tree_gsm8k':     'Ph 6 — Tree GSM8K Eval: Eval naive_tree — does it beat kl_tree on the naive verifier?',
  'eval_nss_tree_gsm8k':       'Ph 6 — Tree GSM8K Eval: Eval nss_tree — does it beat kl_tree on the nss verifier?',
  'eval_si_tree_gsm8k':        'Ph 6 — Tree GSM8K Eval: Eval specinfer_tree — does it beat kl_tree on specinfer?',
  'eval_st_tree_gsm8k':        'Ph 6 — Tree GSM8K Eval: Eval spectr_tree — does it beat kl_tree on spectr?',
  'eval_khisti_tree_gsm8k':    'Ph 6 — Tree GSM8K Eval: Eval khisti_tree — does it beat kl_tree on khisti?',
  'eval_online_kl_tree_gsm8k': 'Ph 6 — Tree GSM8K Eval: Eval online_kl_tree — H6: online tree vs flat online',
  'eval_online_ebe_tree_gsm8k':'Ph 6 — Tree GSM8K Eval: Eval online_ebe_tree',
  // Phase 7 — Tree Multi-DS
  'eval_kl_tree_all':          'Ph 7 — Tree Multi-DS: kl_tree cross-domain (HumanEval, MATH-500, MTBench, Alpaca)',
  'eval_bv_tree_all':          'Ph 7 — Tree Multi-DS: bv_tree cross-domain',
  'eval_gbv_tree_all':         'Ph 7 — Tree Multi-DS: gbv_tree cross-domain',
  'eval_trav_tree_all':        'Ph 7 — Tree Multi-DS: traversal_tree cross-domain',
  'eval_ebe_tree_all':         'Ph 7 — Tree Multi-DS: ebe_tree cross-domain',
  'eval_rev_kl_tree_all':      'Ph 7 — Tree Multi-DS: rev_kl_tree cross-domain',
  'eval_jsd_tree_all':         'Ph 7 — Tree Multi-DS: jsd_tree cross-domain',
  'eval_naive_tree_all':       'Ph 7 — Tree Multi-DS: naive_tree cross-domain',
  'eval_nss_tree_all':         'Ph 7 — Tree Multi-DS: nss_tree cross-domain',
  'eval_si_tree_all':          'Ph 7 — Tree Multi-DS: specinfer_tree cross-domain',
  'eval_st_tree_all':          'Ph 7 — Tree Multi-DS: spectr_tree cross-domain',
  'eval_khisti_tree_all':      'Ph 7 — Tree Multi-DS: khisti_tree cross-domain',
  'eval_online_kl_tree_all':   'Ph 7 — Tree Multi-DS: online_kl_tree cross-domain',
  'eval_online_ebe_tree_all':  'Ph 7 — Tree Multi-DS: online_ebe_tree cross-domain',
};

// Phases grouped for the step panel
// MUST stay in sync with experiment.py build_steps() step IDs.
const PHASE_GROUPS = [
  { label: 'Ph 1 — Baseline',  steps: ['eval_baseline_gsm8k'] },
  { label: 'Ph 2 — Training',  steps: [
      'train_kl_gsm8k','merge_kl_gsm8k',
      'train_ebe_gsm8k','merge_ebe_gsm8k',
      'train_ebe_single_gsm8k','merge_ebe_single_gsm8k',
      'train_rev_kl_gsm8k','merge_rev_kl_gsm8k',
      'train_jsd_gsm8k','merge_jsd_gsm8k',
      'train_l1_gsm8k','merge_l1_gsm8k',
      'online_adapt_gsm8k','merge_online_gsm8k',
      'online_ebe_adapt_gsm8k','merge_online_ebe_gsm8k',
      'online_ebe_single_adapt_gsm8k','merge_online_ebe_single_gsm8k',
  ]},
  { label: 'Ph 3 — GSM8K Eval', steps: [
      'eval_kl_gsm8k','eval_ebe_gsm8k','eval_ebe_single_gsm8k',
      'eval_rev_kl_gsm8k','eval_jsd_gsm8k','eval_l1_gsm8k',
      'eval_online_gsm8k','eval_online_ebe_gsm8k','eval_online_ebe_single_gsm8k',
  ]},
  { label: 'Ph 4 — Multi-DS',   steps: [
      'eval_baseline_all',
      'eval_kl_all','eval_ebe_all','eval_ebe_single_all','eval_rev_kl_all',
      'eval_jsd_all','eval_l1_all',
      'eval_online_all','eval_online_ebe_all','eval_online_ebe_single_all',
  ]},
  { label: 'Ph 5 — EAGLE',     steps: ['eagle_gen','eagle_train','eagle_eval'] },
  { label: 'Ph 6 — Tree Train', steps: [
      'train_kl_tree_gsm8k','merge_kl_tree_gsm8k',
      'train_bv_tree_gsm8k','merge_bv_tree_gsm8k',
      'train_gbv_tree_gsm8k','merge_gbv_tree_gsm8k',
      'train_trav_tree_gsm8k','merge_trav_tree_gsm8k',
      'train_ebe_tree_gsm8k','merge_ebe_tree_gsm8k',
      'train_rev_kl_tree_gsm8k','merge_rev_kl_tree_gsm8k',
      'train_jsd_tree_gsm8k','merge_jsd_tree_gsm8k',
      'train_naive_tree_gsm8k','merge_naive_tree_gsm8k',
      'train_nss_tree_gsm8k','merge_nss_tree_gsm8k',
      'train_si_tree_gsm8k','merge_si_tree_gsm8k',
      'train_st_tree_gsm8k','merge_st_tree_gsm8k',
      'train_khisti_tree_gsm8k','merge_khisti_tree_gsm8k',
      'online_kl_tree_adapt_gsm8k','merge_online_kl_tree_gsm8k',
      'online_ebe_tree_adapt_gsm8k','merge_online_ebe_tree_gsm8k',
  ]},
  { label: 'Ph 6 — Tree GSM8K Eval', steps: [
      'eval_kl_tree_gsm8k','eval_bv_tree_gsm8k','eval_gbv_tree_gsm8k',
      'eval_trav_tree_gsm8k','eval_ebe_tree_gsm8k','eval_rev_kl_tree_gsm8k',
      'eval_jsd_tree_gsm8k',
      'eval_naive_tree_gsm8k','eval_nss_tree_gsm8k','eval_si_tree_gsm8k',
      'eval_st_tree_gsm8k','eval_khisti_tree_gsm8k',
      'eval_online_kl_tree_gsm8k','eval_online_ebe_tree_gsm8k',
  ]},
  { label: 'Ph 7 — Tree Multi-DS', steps: [
      'eval_kl_tree_all','eval_bv_tree_all','eval_gbv_tree_all','eval_trav_tree_all',
      'eval_ebe_tree_all','eval_rev_kl_tree_all','eval_jsd_tree_all',
      'eval_naive_tree_all','eval_nss_tree_all','eval_si_tree_all',
      'eval_st_tree_all','eval_khisti_tree_all',
      'eval_online_kl_tree_all','eval_online_ebe_tree_all',
  ]},
];

// Show a "no data yet" placeholder inside a chart div
function showNoData(divId, msg) {
  const el = document.getElementById(divId);
  if (!el) return;
  try { Plotly.purge(divId); } catch(e) {}
  el.innerHTML = `<div style="display:flex;align-items:center;justify-content:center;
    height:100%;color:#bbb;font-size:13px;flex-direction:column;gap:10px">
    <span style="font-size:32px;opacity:.4">📊</span>
    <span style="font-style:italic">${msg || 'No data yet — waiting for pipeline results'}</span>
  </div>`;
}

const COLORS = {
  baseline:  '#6c757d',
  kl200:     '#fd7e14',
  ebe200:    '#0d6efd',
  kl:        '#fd7e14',    // forward KL — orange
  ebe:       '#0d6efd',    // EBE — blue
  rev_kl:    '#dc3545',    // reverse KL — red
  reverse_kl:'#dc3545',
  jsd:       '#198754',    // JSD — green
  online:    '#9c27b0',    // online OSD — purple
};
const MODE_SYMBOLS = { specinfer: 'circle', gbv: 'square', traversal: 'diamond', bv: 'triangle-up' };

// ---- Pipeline phase dropdown ----
let _activePhaseIdx = -1;
function togglePhase(pi) {
  const dropdown = document.getElementById('phase-dropdown');
  if (!dropdown) return;

  // Clicking the already-open phase closes it
  if (_activePhaseIdx === pi) {
    dropdown.style.display = 'none';
    _activePhaseIdx = -1;
    return;
  }
  _activePhaseIdx = pi;

  const statusMap = window._pipelineStatusMap || {};
  const ph = PHASE_GROUPS[pi];
  const badges = ph.steps.map(sid => {
    const st  = statusMap[sid] || 'pending';
    const col = STATUS_COLOR[st] || '#888';
    const dot = st === 'running'
      ? '<span style="animation:pulse 1s infinite;display:inline-block;margin-right:2px">&#9679;</span>' : '';
    const icon = st === 'done' ? '&#10003;&nbsp;' : st === 'failed' ? '&#10005;&nbsp;'
               : st === 'stopped' ? '&#9646;&nbsp;' : '';
    const desc = STEP_DESC[sid] || sid;
    return `<span title="${desc}"
      style="padding:2px 8px;border-radius:8px;cursor:default;
             background:${col}22;border:1px solid ${col}88;color:${col};
             font-size:10px;white-space:nowrap;display:inline-block">
      ${dot}${icon}${sid}</span>`;
  }).join('');

  dropdown.innerHTML = badges;
  dropdown.style.display = 'flex';
}

// ---- Init ----
async function init() {
  const resp = await fetch('/api/dimensions');
  DIMS = await resp.json();
  buildFilterChips();
  populateSelects();

  // Bootstrap tab change handlers (re-render charts when switching tabs)
  document.querySelectorAll('[data-bs-toggle="tab"]').forEach(tab => {
    tab.addEventListener('shown.bs.tab', e => {
      const target = e.target.getAttribute('href');
      if (target === '#tab-training') { loadTrainingCurves(); renderOnlineHealth(); }
      if (target === '#tab-results')  { renderKeyResults(); }
      if (target === '#tab-robust')   { renderAcceptanceRate(); renderModeComparison(); renderBeVsK(); renderTemperature(); renderRobustness(); }
      if (target === '#tab-analysis') { renderEfficiency(); renderAlpha(); renderCategory(); renderThroughput(); renderSensitivity(); }
      if (target === '#tab-data')     { renderTable(); renderPivot(); }
    });
  });

  await Promise.all([loadData(), updatePipelineStatus()]);

  // Auto-refresh every 30s — no interaction required
  // Polls pipeline state file + DB continuously so the dashboard
  // always shows the latest completed cells without manual refresh.
  startAutoRefresh(30);
}

// Tree-based verifiers vs scalar measurement modes
const _TREE_MODES   = new Set(['gbv','traversal','specinfer','bv','naive']);
const _SCALAR_MODES = new Set(['alpha','perplexity']);

function buildFilterChips() {
  // Model family combo chips — one chip per (draft, target) pair
  const pairsDiv = document.getElementById('f-model_pairs');
  if (pairsDiv) {
    const pairs = DIMS['model_pairs'] || [];
    pairsDiv.innerHTML = '';
    const sec = pairsDiv.closest('.filter-section');
    if (pairs.length <= 1) {
      if (sec) sec.style.display = 'none';  // only one family, no need to filter
    } else {
      if (sec) sec.style.display = '';
      pairs.forEach(p => {
        const chip = document.createElement('span');
        chip.className = 'chip model-pair-chip active';
        chip.textContent = p.label;
        chip.title = p.draft + '  ->  ' + p.target;
        chip.dataset.draft  = p.draft;
        chip.dataset.target = p.target;
        chip.onclick = () => toggleModelPairChip(chip);
        pairsDiv.appendChild(chip);
      });
      MODEL_PAIR_FILTER = null;  // all selected = no restriction
    }
  }

  // Standard columns (mode handled specially below)
  // loss_name always identical to draft_label — omit from chips
  const filterCols = ['draft_label','dataset','K','temperature','train_steps','experiment_tag'];
  filterCols.forEach(col => {
    const div = document.getElementById('f-' + col);
    if (!div) return;
    const vals = DIMS[col] || [];
    vals.forEach(val => {
      const chip = document.createElement('span');
      chip.className = 'chip';
      chip.textContent = val;
      chip.dataset.col = col;
      chip.dataset.val = val;
      chip.onclick = () => toggleChip(chip);
      div.appendChild(chip);
    });
    // Hide entire section when ≤1 distinct values (e.g. train_steps always "0")
    const section = div.closest('.filter-section');
    if (section && vals.length <= 1) section.style.display = 'none';
  });

  // Split mode into "Verifier Mode" (tree/sequential) and "Measurement" (alpha/perplexity)
  const modes = DIMS['mode'] || [];
  const treeModes   = modes.filter(m => _TREE_MODES.has(m));
  const scalarModes = modes.filter(m => _SCALAR_MODES.has(m));

  function _addModeChips(divId, modeList) {
    const div = document.getElementById(divId);
    if (!div) return;
    modeList.forEach(val => {
      const chip = document.createElement('span');
      chip.className = 'chip';
      chip.textContent = val;
      chip.dataset.col = 'mode';   // still filters on the 'mode' DB column
      chip.dataset.val = val;
      chip.onclick = () => toggleChip(chip);
      div.appendChild(chip);
    });
    const section = div.closest('.filter-section');
    if (section && modeList.length <= 1) section.style.display = 'none';
  }
  _addModeChips('f-mode-tree',   treeModes);
  _addModeChips('f-mode-scalar', scalarModes);
}

function toggleChip(chip) {
  chip.classList.toggle('active');
  applyFilters();   // instant re-render — no "Apply Filters" button press needed
}

function toggleTierChip(chip) {
  const tier = chip.dataset.tier;
  if (HW_TIER_FILTER.has(tier)) {
    if (HW_TIER_FILTER.size > 1) {
      HW_TIER_FILTER.delete(tier);
      chip.classList.remove('active');
    }
  } else {
    HW_TIER_FILTER.add(tier);
    chip.classList.add('active');
  }
  applyFilters();
}

function toggleModelPairChip(chip) {
  const key = chip.dataset.draft + '||' + chip.dataset.target;
  const allChips = document.querySelectorAll('.model-pair-chip');
  const activeKeys = new Set(
    [...allChips].filter(c => c.classList.contains('active'))
                 .map(c => c.dataset.draft + '||' + c.dataset.target)
  );
  if (activeKeys.has(key)) {
    if (activeKeys.size > 1) {   // keep at least one selected
      chip.classList.remove('active');
      activeKeys.delete(key);
      MODEL_PAIR_FILTER = activeKeys;
    }
  } else {
    chip.classList.add('active');
    activeKeys.add(key);
    // null = all pairs shown (no restriction needed when all are selected)
    MODEL_PAIR_FILTER = (activeKeys.size === allChips.length) ? null : activeKeys;
  }
  applyFilters();
}

function getActiveChips() {
  const filters = {};
  // Exclude hw-tier-chip and model-pair-chip — both handled by their own filter state.
  document.querySelectorAll('.chip.active:not(.hw-tier-chip):not(.model-pair-chip)').forEach(c => {
    const col = c.dataset.col, val = c.dataset.val;
    if (!col) return;   // guard: skip chips that lack data-col (shouldn't happen now)
    if (!filters[col]) filters[col] = [];
    filters[col].push(String(val));
  });
  return filters;
}

function clearFilters() {
  document.querySelectorAll('.chip.active').forEach(c => c.classList.remove('active'));
  const s = document.getElementById('etag-search');
  if (s) s.value = '';
  document.querySelectorAll('#f-experiment_tag .chip').forEach(c => c.style.display = '');
  // Reset model pair filter to all selected
  MODEL_PAIR_FILTER = null;
  document.querySelectorAll('.model-pair-chip').forEach(c => c.classList.add('active'));
  // Reset HW_TIER_FILTER to all tiers selected
  HW_TIER_FILTER = new Set(['laptop', 'colab_lite', 'colab', 'a100']);
  document.querySelectorAll('.hw-tier-chip').forEach(c => c.classList.add('active'));
  applyFilters();
}

// ---- Convenience: is a given mode a tree verifier or a scalar measurement? ----
function isMeasurementMode(mode) { return _SCALAR_MODES.has(mode); }
function isVerifierMode(mode)    { return _TREE_MODES.has(mode); }

function filterTagChips(q) {
  // Live-filter the experiment_tag chips by substring match
  document.querySelectorAll('#f-experiment_tag .chip').forEach(c => {
    c.style.display = q && !c.textContent.toLowerCase().includes(q.toLowerCase())
      ? 'none' : '';
  });
}

function populateSelects() {
  const datasets = DIMS.dataset || [];
  const datasetSelects = ['sel-dataset-be','sel-dataset-mode','sel-dataset-temp',
                          'sel-dataset-templine','sel-dataset-tdelta'];
  datasetSelects.forEach(id => {
    const sel = document.getElementById(id);
    if (!sel) return;
    // Preserve current selection if already set (e.g. user picked something earlier)
    const prev = sel.value;
    sel.innerHTML = '<option value="">All</option>'
      + datasets.map(d => `<option>${d}</option>`).join('');
    if (prev && datasets.includes(prev)) sel.value = prev;
  });

  // Populate temperature dropdowns
  const temps = (DIMS.temperature || []).map(String).sort();
  const tempSelects = ['sel-temp-be','sel-temp-mode','sel-temp-hm','sel-temp-gain','sel-temp-rob'];
  tempSelects.forEach(id => {
    const sel = document.getElementById(id);
    if (!sel) return;
    sel.innerHTML = '<option value="">All</option>'
      + temps.map(t => `<option value="${t}">${t}</option>`).join('');
  });

  // Populate dataset dropdowns for new charts
  const dsSelects = ['sel-dataset-hm','sel-dataset-gain','sel-dataset-templine',
                     'sel-dataset-tgain','sel-dataset-rob'];
  dsSelects.forEach(id => {
    const sel = document.getElementById(id);
    if (!sel) return;
    sel.innerHTML = '<option value="">All</option>'
      + datasets.map(d => `<option>${d}</option>`).join('');
  });
}

async function applyFilters() {
  await loadData();
}

async function loadData() {
  const activeChips = getActiveChips();
  // Build query string — if multiple values for a col, we filter client-side
  const resp = await fetch('/api/runs');
  let runs = await resp.json();

  // Client-side filter: model family (draft+target pair)
  if (MODEL_PAIR_FILTER !== null && MODEL_PAIR_FILTER.size > 0) {
    runs = runs.filter(r => {
      const key = (r.draft_path || '') + '||' + (r.target_path || '');
      return MODEL_PAIR_FILTER.has(key);
    });
  }

  // Client-side filter for hw_tier (annotation, uses HW_TIER_FILTER set)
  if (HW_TIER_FILTER.size > 0) {
    runs = runs.filter(r => HW_TIER_FILTER.has(r.hw_tier || 'laptop'));
  }

  // Client-side filter for multi-select chips
  Object.entries(activeChips).forEach(([col, vals]) => {
    if (vals.length > 0) {
      runs = runs.filter(r => vals.includes(String(r[col])));
    }
  });

  // Derive alpha_eff = (BE-1)/K for every BE row.
  // Exact derivation: gen_tokens = accepted_drafts + target_calls
  //   => mean_accepted_per_round = BE-1  => alpha_eff = (BE-1)/K
  // Fills the gap where GBV/main.py tracks gen_tokens/calls but not accepted_drafts separately.
  runs.forEach(r => {
    r.alpha_eff = (r.block_eff != null && r.K > 0)
      ? (r.block_eff - 1) / r.K
      : null;
  });

  ALL_RUNS = runs;
  document.getElementById('badge-runs').textContent = `${runs.length} runs`;
  document.getElementById('run-count').textContent = `${runs.length} run(s) match current filters`;

  renderSummaryStats();
  renderKeyResults();
  renderAlpha();
  renderAcceptanceRate();
  renderBeVsK();
  renderModeComparison();
  renderTemperature();
  renderThroughput();
  renderTable();
  loadTrainingCurves();
  renderOnlineHealth();
  renderCategory();
  // Only re-render heavy tabs when they're actually visible
  if (document.getElementById('tab-analysis')?.classList.contains('active')) renderEfficiency();
  if (document.getElementById('tab-data')?.classList.contains('active')) renderPivot();
}

// ---- Helpers ----
function groupBy(arr, key) {
  return arr.reduce((acc, x) => {
    const k = x[key];
    if (!acc[k]) acc[k] = [];
    acc[k].push(x);
    return acc;
  }, {});
}

function mean(arr) {
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function draftColor(label) {
  return COLORS[label] || COLORS[label?.split('_')[0]] || '#888';
}

// ---- Summary Stats ----
function renderSummaryStats() {
  const alphaRuns = ALL_RUNS.filter(r => r.mode === 'alpha' && r.alpha_mean != null);
  const beRuns    = ALL_RUNS.filter(r => r.block_eff != null);
  const pplRuns   = ALL_RUNS.filter(r => r.mode === 'perplexity' && r.perplexity != null);
  const labels    = [...new Set(ALL_RUNS.map(r => r.draft_label))];

  let html = '';

  // ── Total Runs — show the 3-way breakdown so the number is never mysterious ──
  const runBreakdown = [
    beRuns.length  ? `${beRuns.length} BE`    : null,
    pplRuns.length ? `${pplRuns.length} PPL`  : null,
    alphaRuns.length ? `${alphaRuns.length} α` : null,
  ].filter(Boolean).join(' + ');
  html += statBox('Total Runs', ALL_RUNS.length, runBreakdown);

  // ── α checks — honest about what they cover ──────────────────────────────────
  // (If only baseline was measured, say so — α_eff is now derived from all 160 BE rows)
  if (alphaRuns.length) {
    const alphaLabels    = [...new Set(alphaRuns.map(r => r.draft_label))];
    const isBaselineOnly = alphaLabels.length === 1 && alphaLabels[0] === 'baseline';
    const alphaSub = isBaselineOnly
      ? 'baseline calibration only — α_eff derived from all BE rows'
      : alphaLabels.join(', ');
    html += statBox('α checks', alphaRuns.length, alphaSub);
  }

  // ── PPL checks — quality-preservation sanity, separate from BE ───────────────
  if (pplRuns.length) {
    const pplModels = [...new Set(pplRuns.map(r => r.draft_label))];
    html += statBox('PPL checks', pplRuns.length, `${pplModels.length} models — quality sanity`);
  }

  html += statBox('BE Evals', beRuns.length, 'speedup metric — main result');

  // Best acceptance rate — prefer derived alpha_eff=(BE-1)/K from BE runs (covers all
  // trained models) over the raw alpha_mean rows (which are baseline-only right now).
  // alpha_eff = mean accepted draft tokens / K per round; exact derivation:
  //   gen_tokens = accepted_drafts + target_calls  →  accepted_drafts/call = BE−1  →  rate = (BE−1)/K
  if (beRuns.length) {
    // Group by draft_label, average alpha_eff across all conditions
    const byLabel = {};
    beRuns.forEach(r => {
      const ae = (r.block_eff - 1) / r.K;
      if (!byLabel[r.draft_label]) byLabel[r.draft_label] = [];
      byLabel[r.draft_label].push(ae);
    });
    let bestLabel = null, bestAe = -Infinity;
    Object.entries(byLabel).forEach(([lbl, vals]) => {
      const avg = vals.reduce((a,b) => a+b, 0) / vals.length;
      if (avg > bestAe) { bestAe = avg; bestLabel = lbl; }
    });
    const baselineAe = byLabel['baseline']
      ? byLabel['baseline'].reduce((a,b)=>a+b,0) / byLabel['baseline'].length : null;
    const gainStr = baselineAe != null && bestLabel !== 'baseline'
      ? `+${((bestAe - baselineAe) / baselineAe * 100).toFixed(0)}% vs baseline` : '';
    html += statBox('Best accept rate', bestAe.toFixed(3),
      `${bestLabel} · ${gainStr || '(BE-1)/K avg all modes+K'}`);
  } else if (alphaRuns.length) {
    // Fallback: only show raw alpha if no BE data at all, with a warning
    const best = alphaRuns.reduce((a,b) => (a.alpha_mean > b.alpha_mean ? a : b));
    const onlyLabels = [...new Set(alphaRuns.map(r => r.draft_label))].join(', ');
    html += statBox('Best α (measured)', best.alpha_mean.toFixed(4),
      `${best.draft_label} · only ${onlyLabels} measured`);
  }
  // Best BE — normalised by (L+1) so different K values are on the same scale.
  // Max possible BE is L+1, NOT K+1:
  //   K = tree width (# parallel draft paths) — affects hit probability, not max depth
  //   L = draft block length (# tokens per path) — each target call verifies up to L drafts
  //   In one target forward pass: accept ≤ L draft tokens + 1 residual = L+1 max tokens out.
  //   K has no effect on the ceiling; more K just raises the probability of hitting it.
  if (beRuns.length) {
    // Best by normalised efficiency BE/(L+1); fall back to K+1 if L absent (legacy rows)
    const _maxBE = r => (r.L != null ? r.L + 1 : r.K + 1);
    const bestNorm = beRuns.reduce((a,b) =>
      (a.block_eff/_maxBE(a) > b.block_eff/_maxBE(b) ? a : b));
    const normPct = (bestNorm.block_eff / _maxBE(bestNorm) * 100).toFixed(0);
    html += statBox('Best BE (norm)', `${bestNorm.block_eff.toFixed(3)} · ${normPct}%`,
      `${bestNorm.draft_label} ${bestNorm.mode} K=${bestNorm.K} L=${bestNorm.L ?? '?'} (of max L+1=${_maxBE(bestNorm)})`);

    // Also show best raw BE per K so both are visible
    [3, 5].forEach(k => {
      const kRuns = beRuns.filter(r => r.K === k);
      if (!kRuns.length) return;
      const bestK  = kRuns.reduce((a,b) => (a.block_eff > b.block_eff ? a : b));
      const maxPoss = _maxBE(bestK);
      const kEff   = (bestK.block_eff / maxPoss * 100).toFixed(0);
      html += statBox(`Best BE K=${k}`, bestK.block_eff.toFixed(3),
        `${bestK.draft_label} ${bestK.mode} · ${kEff}% of L+1=${maxPoss}`);
    });
  }

  document.getElementById('stat-row').innerHTML = html;
}

function statBox(label, val, sub) {
  return `<div class="col-auto">
    <div class="stat-box">
      <div class="val">${val}</div>
      <div class="lbl">${label}</div>
      ${sub ? `<div style="font-size:10px;color:#aaa">${sub}</div>` : ''}
    </div>
  </div>`;
}

// ---- Alpha Chart ----
function renderAlpha() {
  const runs = ALL_RUNS.filter(r => r.mode === 'alpha' && r.alpha_mean != null);
  if (!runs.length) { showNoData('chart-alpha', 'Alpha results pending — eval steps will populate this once the alpha cells complete'); showNoData('chart-alpha-dataset', 'Alpha by dataset pending'); return; }

  const byLabel = groupBy(runs, 'draft_label');
  const traces = Object.entries(byLabel).map(([label, rows]) => {
    const byDataset = groupBy(rows, 'dataset');
    const xs = Object.keys(byDataset);
    const ys = xs.map(d => mean(byDataset[d].map(r => r.alpha_mean)));
    const errs = xs.map(d => mean(byDataset[d].map(r => r.alpha_ci95 || 0)));
    return {
      type: 'bar', name: label, x: xs, y: ys,
      error_y: { type: 'data', array: errs, visible: true },
      marker: { color: draftColor(label) },
    };
  });

  Plotly.newPlot('chart-alpha', traces, {
    barmode: 'group',
    yaxis: { title: 'Alpha (acceptance rate)', range: [0, 1] },
    xaxis: { title: 'Dataset' },
    legend: { orientation: 'h', y: -0.2 },
    margin: { t: 20 },
  }, { responsive: true });

  // Per-dataset breakdown (if diverse50 has category data)
  renderAlphaDataset(runs);
}

function renderAlphaDataset(runs) {
  // Show alpha by dataset in a horizontal bar
  const byLabel = groupBy(runs, 'draft_label');
  const datasets = [...new Set(runs.map(r => r.dataset))];
  const traces = Object.entries(byLabel).map(([label, rows]) => ({
    type: 'bar', name: label, orientation: 'h',
    y: datasets,
    x: datasets.map(d => {
      const m = rows.filter(r => r.dataset === d).map(r => r.alpha_mean);
      return m.length ? mean(m) : null;
    }),
    marker: { color: draftColor(label) },
  }));
  Plotly.newPlot('chart-alpha-dataset', traces, {
    barmode: 'group',
    xaxis: { title: 'Alpha', range: [0, 1] },
    margin: { t: 10 },
  }, { responsive: true });
}

// ---- Block Efficiency vs K ----
function renderBeVsK() {
  const selMode = document.getElementById('sel-mode-be')?.value || 'specinfer';
  const selDataset = document.getElementById('sel-dataset-be')?.value || '';
  const selTemp = document.getElementById('sel-temp-be')?.value || '';
  let runs = ALL_RUNS.filter(r => r.mode === selMode && r.block_eff != null);
  if (selDataset) runs = runs.filter(r => r.dataset === selDataset);
  if (selTemp) runs = runs.filter(r => String(r.temperature) === selTemp);

  if (!runs.length) {
    const why = selDataset ? `no BE data for dataset "${selDataset}" — try "All"` : `no BE data yet for mode "${selMode}"`;
    showNoData('chart-be-k', why); showNoData('chart-be-k-modes', why); return;
  }

  const byLabel = groupBy(runs, 'draft_label');
  const traces = Object.entries(byLabel).map(([label, rows]) => {
    const byK = groupBy(rows, 'K');
    const Ks = Object.keys(byK).map(Number).sort((a,b)=>a-b);
    return {
      type: 'scatter', mode: 'lines+markers',
      name: label,
      x: Ks,
      y: Ks.map(k => mean(byK[k].map(r => r.block_eff))),
      marker: { color: draftColor(label), size: 8 },
      line: { color: draftColor(label) },
    };
  });

  Plotly.newPlot('chart-be-k', traces, {
    xaxis: { title: 'K (tree width)', dtick: 1 },
    yaxis: { title: 'Block Efficiency (tokens / target call)' },
    legend: { orientation: 'h', y: -0.2 },
    margin: { t: 10 },
  }, { responsive: true });

  // All modes subplot
  renderBeKModes();
}

function renderBeKModes() {
  const modes = [...new Set(ALL_RUNS.filter(r=>r.block_eff!=null).map(r=>r.mode))];
  if (!modes.length) return;

  const cols = Math.min(modes.length, 3);
  const rows_n = Math.ceil(modes.length / cols);
  const specs = [];
  for (let r = 0; r < rows_n; r++) {
    const row = [];
    for (let c = 0; c < cols; c++) row.push({type:'xy'});
    specs.push(row);
  }

  const labels = [...new Set(ALL_RUNS.map(r=>r.draft_label))];
  const traces = [];
  modes.forEach((mode, mi) => {
    const ri = Math.floor(mi/cols)+1, ci = (mi%cols)+1;
    labels.forEach(label => {
      const runs = ALL_RUNS.filter(r=>r.mode===mode && r.draft_label===label && r.block_eff!=null);
      if (!runs.length) return;
      const byK = groupBy(runs, 'K');
      const Ks = Object.keys(byK).map(Number).sort((a,b)=>a-b);
      traces.push({
        type:'scatter', mode:'lines+markers',
        name: label, legendgroup: label,
        showlegend: mi===0,
        xaxis: `x${mi+1}`, yaxis: `y${mi+1}`,
        x: Ks, y: Ks.map(k=>mean(byK[k].map(r=>r.block_eff))),
        marker: { color: draftColor(label), size:6 },
        line: { color: draftColor(label) },
      });
    });
  });

  const layout = {
    grid: { rows: rows_n, columns: cols, pattern: 'independent' },
    margin: { t: 30, b: 40 },
    height: 460,
  };
  modes.forEach((mode, mi) => {
    layout[`xaxis${mi+1}`] = { title: 'K' };
    layout[`yaxis${mi+1}`] = { title: `BE (${mode})` };
    layout[`annotations`] = layout[`annotations`] || [];
    layout[`annotations`].push({ text: mode, xref: `x${mi+1} domain`, yref: `y${mi+1} domain`,
                                  x: 0.5, y: 1.1, showarrow: false, font:{size:12, color:'#333'} });
  });

  Plotly.newPlot('chart-be-k-modes', traces, layout, { responsive: true });
}

document.getElementById('sel-mode-be')?.addEventListener('change', renderBeVsK);
document.getElementById('sel-dataset-be')?.addEventListener('change', renderBeVsK);

// ---- Mode Comparison ----
function renderModeComparison() {
  const selK = parseInt(document.getElementById('sel-k-mode')?.value || '3');
  const selDataset = document.getElementById('sel-dataset-mode')?.value || '';
  const selTemp = document.getElementById('sel-temp-mode')?.value || '';
  let runs = ALL_RUNS.filter(r => r.block_eff != null && r.K === selK);
  if (selDataset) runs = runs.filter(r => r.dataset === selDataset);
  if (selTemp) runs = runs.filter(r => String(r.temperature) === selTemp);
  if (!runs.length) {
    const why = selDataset ? `no BE data for dataset "${selDataset}" — try "All"` : `no BE data yet for K=${selK}`;
    showNoData('chart-mode', why); return;
  }

  const labels = [...new Set(runs.map(r=>r.draft_label))];
  const modes  = [...new Set(runs.map(r=>r.mode))];

  const traces = labels.map(label => ({
    type: 'bar', name: label,
    x: modes,
    y: modes.map(m => {
      const vals = runs.filter(r=>r.draft_label===label && r.mode===m).map(r=>r.block_eff);
      return vals.length ? mean(vals) : null;
    }),
    marker: { color: draftColor(label) },
  }));

  Plotly.newPlot('chart-mode', traces, {
    barmode: 'group',
    yaxis: { title: 'Block Efficiency' },
    xaxis: { title: 'Verifier Mode' },
    margin: { t: 10 },
  }, { responsive: true });
}

document.getElementById('sel-k-mode')?.addEventListener('change', renderModeComparison);
document.getElementById('sel-dataset-mode')?.addEventListener('change', renderModeComparison);
document.getElementById('sel-temp-mode')?.addEventListener('change', renderModeComparison);
document.getElementById('sel-temp-be')?.addEventListener('change', renderBeVsK);

// ---- Acceptance Rate (derived from BE) ----
// alpha_eff = (BE-1)/K: exact derivation — no new eval runs needed.
// Every BE row already encodes accepted_drafts implicitly:
//   gen_tokens = accepted_drafts + target_calls  =>  (BE-1) = mean_accepted/call  =>  /K = rate
function renderAcceptanceRate() {
  const aeRuns = ALL_RUNS.filter(r => r.alpha_eff != null);
  if (!aeRuns.length) {
    showNoData('chart-accept-rate', 'Acceptance rate needs BE evals — appears after first eval step');
    showNoData('chart-accept-rate-k', '');
    return;
  }

  const groupBy_ = document.getElementById('sel-arate-group')?.value || 'mode';
  const showMode = document.getElementById('sel-arate-show')?.value || 'abs';

  // Compute mean alpha_eff per (draft_label, groupKey)
  const labels    = [...new Set(aeRuns.map(r => r.draft_label))];
  const groupVals = [...new Set(aeRuns.map(r => String(r[groupBy_])))].sort();

  // Baseline alpha_eff averaged over all conditions (reference line)
  const blVals = aeRuns.filter(r => r.draft_label === 'baseline').map(r => r.alpha_eff);
  const blMean = blVals.length ? mean(blVals) : null;

  // ── Chart 1: grouped bar — x=groupVal, one bar series per draft_label ──
  const LABEL_ORDER = ['baseline','kl','ebe','rev_kl','jsd','l1','online','online_ebe'];
  const orderedLabels = [...LABEL_ORDER.filter(l => labels.includes(l)),
                         ...labels.filter(l => !LABEL_ORDER.includes(l))];

  const traces1 = orderedLabels.map(label => {
    const y = groupVals.map(gv => {
      const rows = aeRuns.filter(r => r.draft_label === label && String(r[groupBy_]) === gv);
      if (!rows.length) return null;
      const avg = mean(rows.map(r => r.alpha_eff));
      if (showMode === 'gain' && blMean != null) return (avg - blMean) / blMean * 100;
      return avg;
    });
    return {
      type: 'bar', name: label,
      x: groupVals, y,
      marker: { color: draftColor(label) },
      hovertemplate: showMode === 'gain'
        ? `${label}<br>${groupBy_}=%{x}<br>gain vs baseline: %{y:.1f}%<extra></extra>`
        : `${label}<br>${groupBy_}=%{x}<br>α_eff=%{y:.3f}<extra></extra>`,
    };
  });

  // Baseline reference line (only for absolute view)
  const shapes1 = (showMode === 'abs' && blMean != null) ? [{
    type: 'line', x0: -0.5, x1: groupVals.length - 0.5, y0: blMean, y1: blMean,
    xref: 'x', yref: 'y',
    line: { color: '#6c757d', width: 1.5, dash: 'dot' },
  }] : [];
  const annotations1 = (showMode === 'abs' && blMean != null) ? [{
    x: groupVals[groupVals.length - 1], y: blMean, xanchor: 'right', yanchor: 'bottom',
    text: `baseline ${blMean.toFixed(3)}`, showarrow: false,
    font: { size: 10, color: '#6c757d' },
  }] : [];

  const yTitle1 = showMode === 'gain'
    ? 'α_eff gain vs baseline (%)'
    : 'α_eff = (BE−1)/K  [0=all rejected, 1=all accepted]';

  Plotly.newPlot('chart-accept-rate', traces1, {
    barmode: 'group',
    xaxis: { title: groupBy_ === 'mode' ? 'Verifier Mode' : 'K (tree width)' },
    yaxis: { title: yTitle1, zeroline: showMode === 'gain' },
    shapes: shapes1,
    annotations: annotations1,
    legend: { orientation: 'h', y: -0.25 },
    margin: { t: 10, b: 80 },
  }, { responsive: true });

  // ── Chart 2: α_eff vs K lines — shows how acceptance rate falls with longer horizon ──
  const modes    = [...new Set(aeRuns.map(r => r.mode))].sort();
  const Ks       = [...new Set(aeRuns.map(r => r.K))].map(Number).sort((a,b)=>a-b);
  if (Ks.length < 2) { showNoData('chart-accept-rate-k', 'Need K=3 and K=5 data to show K scaling'); return; }

  const traces2 = orderedLabels.map(label => {
    const y = Ks.map(k => {
      const rows = aeRuns.filter(r => r.draft_label === label && r.K === k);
      return rows.length ? mean(rows.map(r => r.alpha_eff)) : null;
    });
    return {
      type: 'scatter', mode: 'lines+markers',
      name: label,
      x: Ks, y,
      marker: { color: draftColor(label), size: 9 },
      line: { color: draftColor(label), width: label === 'baseline' ? 1.5 : 2,
              dash: label === 'baseline' ? 'dot' : 'solid' },
      hovertemplate: `${label}<br>K=%{x}<br>α_eff=%{y:.3f}<extra></extra>`,
    };
  });

  Plotly.newPlot('chart-accept-rate-k', traces2, {
    xaxis: { title: 'K (tree width)', dtick: 1 },
    yaxis: { title: 'α_eff = (BE−1)/K  [higher = more draft tokens accepted per round]' },
    legend: { orientation: 'h', y: -0.25 },
    margin: { t: 10, b: 80 },
    annotations: [{
      xref: 'paper', yref: 'paper', x: 0.5, y: 1.03, xanchor: 'center',
      text: 'Slope = how fast acceptance rate drops as draft horizon lengthens. Steeper = draft model degrades faster.',
      showarrow: false, font: { size: 11, color: '#888' },
    }],
  }, { responsive: true });
}

['sel-arate-group','sel-arate-show'].forEach(id =>
  document.getElementById(id)?.addEventListener('change', renderAcceptanceRate));

// ---- Key Results charts ----
function renderKeyResults() {
  renderInteractionHeatmap();
  renderGainOverBaseline();
  renderRobustness();
  // Pareto and BE-norm live in the Results tab — render them here too
  renderPareto();
  renderBeNorm();
}

// Chart A: Loss × Verifier Heatmap
function renderInteractionHeatmap() {
  const K       = parseInt(document.getElementById('sel-k-hm')?.value || '3');
  const dataset = document.getElementById('sel-dataset-hm')?.value || '';
  const selTemp = document.getElementById('sel-temp-hm')?.value || '';

  let runs = ALL_RUNS.filter(r => r.block_eff != null && r.K === K);
  if (dataset) runs = runs.filter(r => r.dataset === dataset);
  if (selTemp) runs = runs.filter(r => String(r.temperature) === selTemp);
  if (!runs.length) { showNoData('chart-interaction-hm', 'Heatmap needs BE results for at least 2 models — appears after Phase 1 evals complete'); return; }

  const modes  = [...new Set(runs.map(r => r.mode))].sort();
  const labels = [...new Set(runs.map(r => r.draft_label))].sort();

  const z = modes.map(m =>
    labels.map(lbl => {
      const vals = runs.filter(r => r.mode === m && r.draft_label === lbl).map(r => r.block_eff);
      return vals.length ? mean(vals) : null;
    })
  );
  const text = z.map(row => row.map(v => v != null ? v.toFixed(3) : ''));

  Plotly.newPlot('chart-interaction-hm', [{
    type: 'heatmap',
    z, x: labels, y: modes,
    text, texttemplate: '%{text}',
    colorscale: [
      [0, '#fff5f0'], [0.5, '#fc8d59'], [1.0, '#b30000']
    ],
    showscale: true,
    colorbar: { title: 'Block Eff.' },
  }], {
    xaxis: { title: 'Distillation Loss / Model', tickangle: -20 },
    yaxis: { title: 'Verifier Mode' },
    margin: { t: 20, b: 90, l: 90 },
  }, { responsive: true });
}

// Chart B: BE gain over baseline (%)
function renderGainOverBaseline() {
  const K       = parseInt(document.getElementById('sel-k-gain')?.value || '3');
  const dataset = document.getElementById('sel-dataset-gain')?.value || '';
  const selTemp = document.getElementById('sel-temp-gain')?.value || '';

  let runs = ALL_RUNS.filter(r => r.block_eff != null && r.K === K);
  if (dataset) runs = runs.filter(r => r.dataset === dataset);
  if (selTemp) runs = runs.filter(r => String(r.temperature) === selTemp);

  // ── State 1: Phase 1 not yet done — no baseline at all ───────────────────
  const baselineRuns = runs.filter(r => r.draft_label === 'baseline');
  if (!baselineRuns.length) {
    showNoData('chart-gain',
      'Waiting for Phase 1 baseline eval (eval_baseline_gsm8k) to finish…');
    return;
  }

  // ── State 2: Baseline in, but no trained-model evals yet (Phase 3 pending) ─
  // Show the actual baseline BE numbers so the researcher has immediate signal.
  const nonBaseline = runs.filter(r => r.draft_label !== 'baseline');
  if (!nonBaseline.length) {
    const modes = [...new Set(baselineRuns.map(r => r.mode))].sort();
    const tableRows = modes.map(m => {
      const vals = baselineRuns.filter(r => r.mode === m).map(r => r.block_eff);
      const avg  = vals.length ? mean(vals).toFixed(3) : '—';
      return `<tr><td style="padding:3px 14px;font-weight:600">${m}</td>` +
             `<td style="padding:3px 10px;color:#4dabf7;font-variant-numeric:tabular-nums">${avg}</td></tr>`;
    }).join('');
    const el = document.getElementById('chart-gain');
    if (!el) return;
    try { Plotly.purge('chart-gain'); } catch(e) {}
    el.innerHTML =
      `<div style="display:flex;align-items:center;justify-content:center;` +
      `height:100%;flex-direction:column;gap:14px;color:#ccc">` +
      `<div style="font-size:13px;font-style:italic;color:#aaa">` +
      `✅ Baseline established — gain bars appear after Phase 3 trained-model evals land` +
      `</div>` +
      `<table style="font-size:13px;border-collapse:collapse">` +
      `<thead><tr>` +
      `<th style="padding:3px 14px;color:#777;font-weight:400;border-bottom:1px solid #444">Verifier mode</th>` +
      `<th style="padding:3px 10px;color:#777;font-weight:400;border-bottom:1px solid #444">Baseline BE (K=${K})</th>` +
      `</tr></thead><tbody>${tableRows}</tbody>` +
      `</table>` +
      `</div>`;
    return;
  }

  // ── State 3: Baseline + ≥1 trained model — render the gain bars ───────────

  function baselineBE(mode, temperature, ds) {
    const vals = baselineRuns.filter(r =>
      r.mode === mode &&
      (temperature == null || Math.abs(r.temperature - temperature) < 0.05) &&
      (ds == null || r.dataset === ds)
    ).map(r => r.block_eff);
    return vals.length ? mean(vals) : null;
  }

  const labels = [...new Set(nonBaseline.map(r => r.draft_label))].sort();
  const modes  = [...new Set(runs.map(r => r.mode))].sort();

  const traces = labels.map(lbl => {
    return {
      type: 'bar', name: lbl,
      x: modes,
      y: modes.map(m => {
        const trained = nonBaseline.filter(r => r.draft_label === lbl && r.mode === m).map(r => r.block_eff);
        const bl = baselineBE(m, null, dataset || null);
        if (!trained.length || bl == null || bl === 0) return null;
        return ((mean(trained) - bl) / bl) * 100;
      }),
      marker: { color: draftColor(lbl) },
    };
  });

  // Zero-line reference
  Plotly.newPlot('chart-gain', traces, {
    barmode: 'group',
    yaxis: { title: 'BE Gain over Baseline (%)', zeroline: true,
             zerolinecolor: '#333', zerolinewidth: 2 },
    xaxis: { title: 'Verifier Mode' },
    shapes: [{ type: 'line', x0: -0.5, x1: modes.length - 0.5, y0: 0, y1: 0,
               line: { color: '#333', width: 1.5, dash: 'dot' } }],
    legend: { orientation: 'h', y: -0.2 },
    margin: { t: 10, b: 60 },
  }, { responsive: true });
}

// Chart C: Robustness across datasets
function renderRobustness() {
  const mode    = document.getElementById('sel-mode-rob')?.value || 'traversal';
  const K       = parseInt(document.getElementById('sel-k-rob')?.value || '3');
  const selTemp = document.getElementById('sel-temp-rob')?.value || '';

  let runs = ALL_RUNS.filter(r => r.block_eff != null && r.mode === mode && r.K === K);
  if (selTemp) runs = runs.filter(r => String(r.temperature) === selTemp);
  if (!runs.length) { showNoData('chart-robustness', 'Dataset robustness needs Phase 4 multi-dataset evals — the last 3 steps of the pipeline'); return; }

  const datasets = [...new Set(runs.map(r => r.dataset))].sort();
  const labels   = [...new Set(runs.map(r => r.draft_label))].sort();

  const traces = labels.map(lbl => ({
    type: 'bar', name: lbl,
    x: datasets,
    y: datasets.map(ds => {
      const vals = runs.filter(r => r.draft_label === lbl && r.dataset === ds).map(r => r.block_eff);
      return vals.length ? mean(vals) : null;
    }),
    marker: { color: draftColor(lbl) },
  }));

  Plotly.newPlot('chart-robustness', traces, {
    barmode: 'group',
    yaxis: { title: 'Block Efficiency' },
    xaxis: { title: 'Dataset' },
    legend: { orientation: 'h', y: -0.2 },
    margin: { t: 10, b: 60 },
  }, { responsive: true });
}

['sel-k-hm','sel-dataset-hm','sel-temp-hm'].forEach(id =>
  document.getElementById(id)?.addEventListener('change', renderInteractionHeatmap));
['sel-k-gain','sel-dataset-gain','sel-temp-gain'].forEach(id =>
  document.getElementById(id)?.addEventListener('change', renderGainOverBaseline));
['sel-mode-rob','sel-k-rob','sel-temp-rob'].forEach(id =>
  document.getElementById(id)?.addEventListener('change', renderRobustness));

// ---- Temperature Robustness ----
function renderTemperature() {
  renderTempLine();
  renderTempGain();
}

function renderTempLine() {
  const mode    = document.getElementById('sel-mode-templine')?.value || 'traversal';
  const K       = parseInt(document.getElementById('sel-k-templine')?.value || '3');
  const dataset = document.getElementById('sel-dataset-templine')?.value || '';
  let runs = ALL_RUNS.filter(r => r.block_eff != null && r.mode === mode && r.K === K);
  if (dataset) runs = runs.filter(r => r.dataset === dataset);
  if (!runs.length) { showNoData('chart-temp-line', 'Temperature sensitivity needs BE results at T=0.6 and T=1.0 — appears after first eval step completes'); return; }

  const labels = [...new Set(runs.map(r => r.draft_label))];
  const traces = labels.map(lbl => {
    const lRuns = runs.filter(r => r.draft_label === lbl);
    const byT = groupBy(lRuns, 'temperature');
    const xs = Object.keys(byT).map(Number).sort((a,b)=>a-b);
    return {
      type: 'scatter', mode: 'lines+markers',
      name: lbl,
      x: xs,
      y: xs.map(t => mean(byT[t].map(r => r.block_eff))),
      marker: { color: draftColor(lbl), size: 10 },
      line: { color: draftColor(lbl) },
    };
  });

  Plotly.newPlot('chart-temp-line', traces, {
    xaxis: { title: 'Sampling Temperature' },
    yaxis: { title: 'Block Efficiency (tokens / target call)' },
    legend: { orientation: 'h', y: -0.2 },
    margin: { t: 10 },
  }, { responsive: true });
}

function renderTempGain() {
  const mode    = document.getElementById('sel-mode-tgain')?.value || 'traversal';
  const K       = parseInt(document.getElementById('sel-k-tgain')?.value || '3');
  const dataset = document.getElementById('sel-dataset-tgain')?.value || '';
  let runs = ALL_RUNS.filter(r => r.block_eff != null && r.mode === mode && r.K === K);
  if (dataset) runs = runs.filter(r => r.dataset === dataset);
  if (!runs.length) { showNoData('chart-temp-gain', 'Temperature gain needs BE results at multiple temperatures — appears after Phase 3 evals complete'); return; }

  const baselineRuns = runs.filter(r => r.draft_label === 'baseline');
  const nonBaseline  = runs.filter(r => r.draft_label !== 'baseline');
  if (!baselineRuns.length) { showNoData('chart-temp-gain', 'Temperature gain needs baseline + trained model BE — appears after Phase 1 evals complete'); return; }

  const labels = [...new Set(nonBaseline.map(r => r.draft_label))];
  const temps  = [...new Set(runs.map(r => r.temperature))].sort((a,b)=>a-b);

  const traces = labels.map(lbl => {
    const xs = temps;
    const ys = temps.map(t => {
      const trained = nonBaseline.filter(r => r.draft_label === lbl && Math.abs(r.temperature - t) < 0.05).map(r => r.block_eff);
      const bl      = baselineRuns.filter(r => Math.abs(r.temperature - t) < 0.05).map(r => r.block_eff);
      if (!trained.length || !bl.length) return null;
      return mean(trained) - mean(bl);
    });
    return {
      type: 'scatter', mode: 'lines+markers',
      name: lbl,
      x: xs, y: ys,
      marker: { color: draftColor(lbl), size: 10 },
      line: { color: draftColor(lbl) },
    };
  });

  Plotly.newPlot('chart-temp-gain', traces, {
    xaxis: { title: 'Sampling Temperature' },
    yaxis: { title: 'BE Gain over Baseline (absolute)', zeroline: true, zerolinecolor:'#999' },
    shapes: [{ type:'line', x0: Math.min(...temps)-0.05, x1: Math.max(...temps)+0.05,
               y0:0, y1:0, line:{color:'#999', width:1, dash:'dot'} }],
    legend: { orientation: 'h', y: -0.2 },
    margin: { t: 10 },
  }, { responsive: true });
}

['sel-mode-templine','sel-k-templine','sel-dataset-templine'].forEach(id =>
  document.getElementById(id)?.addEventListener('change', renderTempLine));
['sel-mode-tgain','sel-k-tgain','sel-dataset-tgain'].forEach(id =>
  document.getElementById(id)?.addEventListener('change', renderTempGain));

// ---- Sensitivity ----
function renderSensitivity() {
  const xCol = document.getElementById('sens-x').value;
  const yCol = document.getElementById('sens-y').value;
  const groupCol = document.getElementById('sens-group').value;

  const runs = ALL_RUNS.filter(r => r[xCol] != null && r[yCol] != null);
  if (!runs.length) { showNoData('chart-sensitivity', `No runs have both ${xCol} and ${yCol} — try different axes`); return; }

  const groups = groupBy(runs, groupCol);
  const traces = Object.entries(groups).map(([grp, rows]) => {
    const byX = groupBy(rows, xCol);
    const xs = Object.keys(byX).map(Number).sort((a,b)=>a-b);
    return {
      type: 'scatter', mode: 'lines+markers',
      name: grp,
      x: xs,
      y: xs.map(x => mean(byX[x].map(r => r[yCol]))),
      marker: { color: draftColor(grp), size: 8 },
    };
  });

  Plotly.newPlot('chart-sensitivity', traces, {
    xaxis: { title: xCol },
    yaxis: { title: yCol },
    margin: { t: 10 },
  }, { responsive: true });
}

// ---- Per-Category ----
async function renderCategory() {
  const alphaRuns = ALL_RUNS.filter(r => r.mode === 'alpha' && r.dataset === 'diverse50');
  if (!alphaRuns.length) {
    showNoData('chart-category', 'Per-category breakdown requires the diverse50 dataset alpha runs — these run in Phase 4 (eval_baseline_all and eval_kxxx_all steps)');
    return;
  }

  const cats = ['factual','coding','cs_concepts','ml_ai','math'];
  const labels = [...new Set(alphaRuns.map(r=>r.draft_label))];

  // Fetch per-prompt data for each relevant run
  const catData = {};
  for (const run of alphaRuns) {
    const pp = await fetch(`/api/per_prompt/${run.id}`).then(r=>r.json());
    if (!catData[run.draft_label]) catData[run.draft_label] = {};
    pp.forEach(row => {
      if (!row.category) return;
      if (!catData[run.draft_label][row.category]) catData[run.draft_label][row.category] = [];
      if (row.alpha != null) catData[run.draft_label][row.category].push(row.alpha);
    });
  }

  // Heatmap: rows=label, cols=category
  const z = labels.map(lbl =>
    cats.map(cat => {
      const vals = (catData[lbl]?.[cat]) || [];
      return vals.length ? mean(vals) : null;
    })
  );

  Plotly.newPlot('chart-category', [{
    type: 'heatmap',
    z, x: cats, y: labels,
    colorscale: 'Blues',
    text: z.map(row => row.map(v => v != null ? v.toFixed(3) : '')),
    texttemplate: '%{text}',
    showscale: true,
    zmin: 0.3, zmax: 0.8,
  }], {
    margin: { t: 10 },
    xaxis: { title: 'Category' },
    yaxis: { title: 'Draft Model' },
  }, { responsive: true });
}

// ---- Throughput ----
function renderThroughput() {
  const runs = ALL_RUNS.filter(r => r.throughput != null && r.throughput > 0);
  if (!runs.length) {
    showNoData('chart-throughput', 'Throughput data (tok/s) not yet available — runner.py will populate this when --measure_throughput is enabled');
    showNoData('chart-scatter', 'α vs throughput needs both alpha and throughput measurements');
    return;
  }

  const byLabel = groupBy(runs, 'draft_label');
  const tpTraces = Object.entries(byLabel).map(([label, rows]) => ({
    type: 'bar', name: label,
    x: rows.map(r => `${r.dataset} ${r.mode}`),
    y: rows.map(r => r.throughput),
    marker: { color: draftColor(label) },
  }));

  Plotly.newPlot('chart-throughput', tpTraces, {
    barmode: 'group',
    yaxis: { title: 'Throughput (tok/s)' },
    margin: { t: 10 },
  }, { responsive: true });

  // Scatter: alpha vs throughput
  const alphaRuns = ALL_RUNS.filter(r => r.alpha_mean != null && r.throughput != null);
  Plotly.newPlot('chart-scatter', [{
    type: 'scatter', mode: 'markers',
    x: alphaRuns.map(r => r.alpha_mean),
    y: alphaRuns.map(r => r.throughput),
    text: alphaRuns.map(r => `${r.draft_label} ${r.dataset}`),
    marker: {
      color: alphaRuns.map(r => draftColor(r.draft_label)),
      size: 8,
    },
  }], {
    xaxis: { title: 'Alpha' },
    yaxis: { title: 'Throughput (tok/s)' },
    margin: { t: 10 },
  }, { responsive: true });
}

// ---- Efficiency tab ----
function renderEfficiency() {
  // chart-eff-decomp lives in the Analysis tab
  renderBeDecomp();
  // chart-pareto and chart-be-norm live in the Results tab; renderKeyResults() handles them
}

// Card 1: α → BE theoretical curve + measured scatter
function renderBeDecomp() {
  const K    = parseInt(document.getElementById('eff-k')?.value || '3');
  const mode = document.getElementById('eff-mode')?.value || 'traversal';

  // Theoretical sequential-SD curve: BE(α) = (1 - α^(K+1)) / (1 - α)
  // (Each block: draft proposes K tokens; accepted up to first rejection.
  //  Expected accepted = sum_{k=0}^{K-1} α^k  = (1 - α^K)/(1-α).
  //  Plus the forced target token = +1.  So BE = 1 + (1-α^K)/(1-α) = (1-α^(K+1))/(1-α).)
  const alphaRange = Array.from({length: 100}, (_, i) => i / 100);
  const theoryCurve = alphaRange.map(a =>
    a >= 0.9999 ? K + 1 : (1 - Math.pow(a, K + 1)) / (1 - a)
  );

  const beRuns  = ALL_RUNS.filter(r => r.block_eff != null && r.mode === mode && r.K === K);
  const alpRuns = ALL_RUNS.filter(r => r.alpha_mean != null && r.mode === 'alpha');

  // Merge alpha + BE by (draft_label, dataset, temperature)
  const key = r => `${r.draft_label}||${r.dataset}||${r.temperature}`;
  const alpMap = {};
  alpRuns.forEach(r => { alpMap[key(r)] = r.alpha_mean; });

  const byLabel = {};
  beRuns.forEach(r => {
    const a = alpMap[key(r)];
    if (a == null) return;
    if (!byLabel[r.draft_label]) byLabel[r.draft_label] = {xs:[], ys:[], texts:[]};
    byLabel[r.draft_label].xs.push(a);
    byLabel[r.draft_label].ys.push(r.block_eff);
    byLabel[r.draft_label].texts.push(`${r.draft_label}<br>${r.dataset} T=${r.temperature}`);
  });

  const traces = [
    {
      type: 'scatter', mode: 'lines',
      name: `Sequential SD theory (K=${K})`,
      x: alphaRange, y: theoryCurve,
      line: { color: '#adb5bd', width: 2, dash: 'dot' },
      hovertemplate: 'α=%{x:.2f}<br>theory BE=%{y:.3f}<extra>sequential theory</extra>',
    },
    ...Object.entries(byLabel).map(([label, d]) => ({
      type: 'scatter', mode: 'markers',
      name: label,
      x: d.xs, y: d.ys, text: d.texts,
      marker: { color: draftColor(label), size: 11, symbol: 'circle',
                line: { width: 1.5, color: '#fff' } },
      hovertemplate: '%{text}<br>α=%{x:.3f}<br>BE=%{y:.3f}<extra>' + label + '</extra>',
    })),
  ];

  if (!Object.keys(byLabel).length) {
    showNoData('chart-eff-decomp', 'Needs both alpha and BE results — appears after Phase 3 evals complete');
    return;
  }

  Plotly.newPlot('chart-eff-decomp', traces, {
    xaxis: { title: 'Token Acceptance Rate (α)', range: [0, 1] },
    yaxis: { title: `Block Efficiency (tokens / target call) — ${mode} K=${K}` },
    legend: { orientation: 'h', y: -0.2 },
    margin: { t: 10, b: 70 },
    shapes: [{
      type: 'line', x0: 0, x1: 1, y0: 1, y1: 1,
      line: { color: '#dee2e6', width: 1 }
    }],
  }, { responsive: true });
}

// Card 2: Quality vs Speed (block_eff) Pareto frontier
// Quality metric: task_score if ≥3 labels have it; otherwise log(baseline_PPL/model_PPL) proxy
function renderPareto() {
  const mode = document.getElementById('eff-pareto-mode')?.value || 'traversal';
  const K    = parseInt(document.getElementById('eff-pareto-k')?.value || '3');

  const beRuns  = ALL_RUNS.filter(r => r.block_eff != null && r.mode === mode && r.K === K);
  const tsRuns  = ALL_RUNS.filter(r => r.task_score != null);

  // PPL fallback: rows where perplexity was measured (mode='perplexity', K=0)
  const pplRuns     = ALL_RUNS.filter(r => r.perplexity != null);
  const blPPLVals   = pplRuns.filter(r => r.draft_label === 'baseline').map(r => r.perplexity);
  const baselinePPL = blPPLVals.length ? mean(blPPLVals) : null;

  // Use task_score if ≥3 distinct labels have it; otherwise fall back to PPL proxy
  const tsLabels = new Set(tsRuns.map(r => r.draft_label));
  const usePPL   = tsLabels.size < 3;

  const labels = [...new Set(beRuns.map(r => r.draft_label))];
  const points = labels.map(lbl => {
    const be = beRuns.filter(r => r.draft_label === lbl).map(r => r.block_eff);
    if (!be.length) return null;
    let quality = null;
    if (!usePPL) {
      const ts = tsRuns.filter(r => r.draft_label === lbl).map(r => r.task_score);
      if (ts.length) quality = mean(ts);
    } else if (baselinePPL != null) {
      const lblPPL = pplRuns.filter(r => r.draft_label === lbl).map(r => r.perplexity);
      if (lblPPL.length) quality = Math.log(baselinePPL / mean(lblPPL)); // + = less degradation
    }
    return quality != null ? { label: lbl, be: mean(be), ts: quality } : null;
  }).filter(Boolean);

  if (!points.length) {
    showNoData('chart-pareto',
      'Pareto frontier needs BE evals + either task_score (--task_score flag) or perplexity data');
    return;
  }

  const xAxisTitle = usePPL
    ? 'Quality proxy  log(baseline_PPL / model_PPL)  — ↑ higher = less degradation vs baseline'
    : 'Task Score  — ↑ higher = less quality degradation';
  const hoverXFmt = usePPL
    ? 'quality proxy=%{x:.3f}' : 'task_score=%{x:.3f}';

  // Pareto-optimal points (top-right corner dominance: high quality AND high BE)
  const sorted = [...points].sort((a, b) => a.ts - b.ts);
  let paretoMaxBE = -Infinity;
  const paretoFront = sorted.filter(p => {
    if (p.be > paretoMaxBE) { paretoMaxBE = p.be; return true; }
    return false;
  });

  const traces = [
    // All points
    {
      type: 'scatter', mode: 'markers+text',
      name: 'Draft models',
      x: points.map(p => p.ts),
      y: points.map(p => p.be),
      text: points.map(p => p.label),
      textposition: 'top center',
      textfont: { size: 10 },
      marker: { color: points.map(p => draftColor(p.label)), size: 12,
                line: { width: 1.5, color: '#fff' } },
      hovertemplate: `%{text}<br>${hoverXFmt}<br>BE=%{y:.3f}<extra></extra>`,
    },
    // Pareto frontier line
    {
      type: 'scatter', mode: 'lines',
      name: 'Pareto frontier',
      x: paretoFront.map(p => p.ts),
      y: paretoFront.map(p => p.be),
      line: { color: '#0d6efd', width: 2, dash: 'dash' },
      hoverinfo: 'skip',
    },
  ];

  // Reference: baseline point (highlighted separately)
  const bl = points.find(p => p.label === 'baseline');
  if (bl) {
    traces.push({
      type: 'scatter', mode: 'markers',
      name: 'Baseline (no training)',
      x: [bl.ts], y: [bl.be],
      marker: { color: '#6c757d', size: 15, symbol: 'diamond',
                line: { width: 2, color: '#333' } },
      hovertemplate: `Baseline<br>${hoverXFmt.replace('%{x','%{x')}<br>BE=%{y:.3f}<extra></extra>`,
    });
  }

  const pplNote = usePPL
    ? ' <span style="font-size:11px;color:#888">(PPL proxy — run eval with --task_score for real quality scores)</span>'
    : '';
  document.getElementById('chart-pareto')?.previousElementSibling
    ?.querySelectorAll('.pareto-note').forEach(n => n.remove());

  Plotly.newPlot('chart-pareto', traces, {
    xaxis: { title: xAxisTitle },
    yaxis: { title: `Block Efficiency — ${mode} K=${K}` },
    legend: { orientation: 'h', y: -0.2 },
    margin: { t: 10, b: 70 },
    annotations: usePPL ? [{
      xref: 'paper', yref: 'paper', x: 0.5, y: 1.04, xanchor: 'center',
      text: '⚠ quality axis = PPL proxy (log baseline/model PPL) — run eval with --task_score for exact scores',
      showarrow: false, font: { size: 11, color: '#888' },
    }] : [],
  }, { responsive: true });
}

// Card 3: Normalised BE improvement over baseline (%)
function renderBeNorm() {
  const K    = parseInt(document.getElementById('eff-norm-k')?.value || '3');
  const selT = document.getElementById('eff-norm-t')?.value || '';

  let runs = ALL_RUNS.filter(r => r.block_eff != null && r.K === K);
  if (selT) runs = runs.filter(r => String(r.temperature) === selT);
  if (!runs.length) { showNoData('chart-be-norm', 'BE normalisation needs Phase 3 eval results'); return; }

  const baselineRuns = runs.filter(r => r.draft_label === 'baseline');
  if (!baselineRuns.length) { showNoData('chart-be-norm', 'Baseline BE results not yet available'); return; }

  const nonBL  = runs.filter(r => r.draft_label !== 'baseline');
  const labels = [...new Set(nonBL.map(r => r.draft_label))].sort();
  const modes  = [...new Set(runs.map(r => r.mode)).values()].filter(m => _TREE_MODES.has(m)).sort();

  const traces = labels.map(lbl => {
    const color = draftColor(lbl);
    const ys = modes.map(m => {
      const trained = nonBL.filter(r => r.draft_label === lbl && r.mode === m).map(r => r.block_eff);
      const bl      = baselineRuns.filter(r => r.mode === m).map(r => r.block_eff);
      if (!trained.length || !bl.length || mean(bl) === 0) return null;
      return ((mean(trained) - mean(bl)) / mean(bl)) * 100;
    });
    return {
      type: 'bar', name: lbl,
      x: modes, y: ys,
      marker: {
        color: ys.map(v => v == null ? '#eee' : v >= 0 ? color : '#dc3545'),
      },
      hovertemplate: '%{x}<br>%{y:.1f}%<extra>' + lbl + '</extra>',
    };
  });

  Plotly.newPlot('chart-be-norm', traces, {
    barmode: 'group',
    yaxis: { title: 'BE improvement over baseline (%)', zeroline: true,
             zerolinecolor: '#333', zerolinewidth: 2 },
    xaxis: { title: 'Verifier Mode' },
    shapes: [{ type: 'line', x0: -0.5, x1: modes.length - 0.5, y0: 0, y1: 0,
               line: { color: '#333', width: 1.5, dash: 'dot' } }],
    legend: { orientation: 'h', y: -0.2 },
    margin: { t: 10, b: 60 },
  }, { responsive: true });
}

['eff-k','eff-mode'].forEach(id =>
  document.getElementById(id)?.addEventListener('change', renderBeDecomp));
['eff-pareto-mode','eff-pareto-k'].forEach(id =>
  document.getElementById(id)?.addEventListener('change', renderPareto));
['eff-norm-k','eff-norm-t'].forEach(id =>
  document.getElementById(id)?.addEventListener('change', renderBeNorm));

// ---- Training Curves ----
// Train colors (solid lines) — one per label
const TRAIN_COLORS = ['#636efa','#ef553b','#00cc96','#ab63fa','#ffa15a','#19d3f3'];
// Val colors — warmer/brighter to stand out clearly from the paired train line
const VAL_COLORS   = ['#ff7f0e','#d62728','#2ca02c','#9467bd','#e377c2','#bcbd22'];

async function loadTrainingCurves() {
  const grid = document.getElementById('chart-training-grid');
  const curves = await fetch('/api/train_curves').then(r=>r.json());

  if (!curves.length) {
    grid.innerHTML = `<div class="chart-card" style="grid-column:1/-1">
      <p class="text-muted small mb-0">Training loss curves appear during Phase 2
      (train_kl_gsm8k and train_ebe_gsm8k steps).
      Train=solid line · Val=dashed line.
      If val rises while train falls → overfitting red flag.</p></div>`;
    return;
  }

  const byLabel = groupBy(curves, 'label');
  const globalMaxTsMs = Math.max(...curves.map(r => new Date(r.ts).getTime()));

  // Determine a stable sort order: baseline first, then by first-seen step
  const labelOrder = Object.keys(byLabel).sort((a, b) => {
    if (a === 'baseline') return -1;
    if (b === 'baseline') return 1;
    return a.localeCompare(b);
  });

  // Clear and rebuild grid — one card per label
  grid.innerHTML = '';
  let redFlagLabels = [];

  labelOrder.forEach((label, colorIdx) => {
    const rows = byLabel[label];
    // Skip online-* labels here — they get their own chart in renderOnlineHealth()
    if (label.includes('online')) return;

    const trainColor = TRAIN_COLORS[colorIdx % TRAIN_COLORS.length];
    const valColor   = VAL_COLORS  [colorIdx % VAL_COLORS.length];

    // De-duplicate train rows by step
    const trainMap = {};
    rows.filter(r => !r.split || r.split === 'train')
        .forEach(r => { if (!trainMap[r.step] || r.id > trainMap[r.step].id) trainMap[r.step] = r; });
    const trainRows = Object.values(trainMap).sort((a,b) => a.step - b.step);
    const valRows   = rows.filter(r => r.split === 'val').sort((a,b) => a.step - b.step);

    if (!trainRows.length) return;

    // Overfitting detection for this label
    const labelMaxTsMs = Math.max(...rows.map(r => new Date(r.ts).getTime()));
    const isCurrentRun  = labelMaxTsMs >= globalMaxTsMs - 2000;
    let overfit = false;
    if (valRows.length >= 2 && isCurrentRun) {
      const minVal  = Math.min(...valRows.map(r => r.loss));
      const lastVal = valRows[valRows.length - 1].loss;
      if (lastVal > minVal + 0.10 * Math.abs(minVal)) {
        redFlagLabels.push(label);
        overfit = true;
      }
    }

    // Divergence / convergence trend over training
    // All losses should go DOWN (lower = better for KL/JSD/L1/rev_kl;
    // more negative = better for EBE). If final > initial: diverging.
    let trendBadge = '';
    if (trainRows.length >= 5) {
      const firstLoss = trainRows[0].loss;
      const lastLoss  = trainRows[trainRows.length - 1].loss;
      const pctChange = (lastLoss - firstLoss) / Math.abs(firstLoss) * 100;
      if (pctChange > 5) {
        trendBadge = `<span class="badge bg-danger ms-2" style="font-size:0.7em">▲ diverging +${pctChange.toFixed(1)}%</span>`;
        redFlagLabels.push(label + ' (diverging)');
      } else if (pctChange < -10) {
        trendBadge = `<span class="badge bg-success ms-2" style="font-size:0.7em">▼ −${Math.abs(pctChange).toFixed(1)}%</span>`;
      }
    }

    // Build traces
    const traces = [];
    if (trainRows.length) {
      traces.push({
        type: 'scatter', mode: 'lines',
        name: 'train',
        x: trainRows.map(r => r.step),
        y: trainRows.map(r => r.loss),
        line: { color: trainColor, width: 2, dash: 'solid' },
        hovertemplate: 'step %{x}<br>train loss: %{y:.4f}<extra>train</extra>',
      });
    }
    if (valRows.length) {
      traces.push({
        type: 'scatter', mode: 'lines',
        name: 'val',
        x: valRows.map(r => r.step),
        y: valRows.map(r => r.loss),
        line: { color: valColor, width: 2.5, dash: 'dash' },
        hovertemplate: 'step %{x}<br><b>val loss: %{y:.4f}</b><extra>val</extra>',
      });
    }

    // Final step annotation
    const lastTrain = trainRows[trainRows.length - 1];
    const maxSteps  = Math.max(...trainRows.map(r => r.step));

    // Compute last val annotation if val exists
    let statusNote = '';
    if (valRows.length) {
      const lastValLoss = valRows[valRows.length - 1].loss;
      statusNote = `val ${lastValLoss.toFixed(3)}`;
    }

    // Create card element
    const chartId = `chart-tc-${label.replace(/[^a-z0-9]/gi, '_')}`;
    const overfitBadge = overfit
      ? `<span class="badge bg-danger ms-2" style="font-size:0.7em">⚠ overfitting</span>`
      : '';
    const card = document.createElement('div');
    card.className = 'chart-card';
    card.innerHTML = `
      <h6 style="text-transform:capitalize">${label}
        <small class="text-muted ms-2" style="font-size:0.75em">
          step ${lastTrain.step}/${maxSteps} · train ${lastTrain.loss.toFixed(3)}
          ${statusNote ? '· ' + statusNote : ''}
        </small>
        ${trendBadge}${overfitBadge}
      </h6>
      <div id="${chartId}" style="height:260px"></div>`;
    grid.appendChild(card);

    Plotly.newPlot(chartId, traces, {
      xaxis: { title: 'Step', tickfont: { size: 10 } },
      yaxis: { title: 'Loss  (↓ better)', tickfont: { size: 10 } },
      legend: { orientation: 'h', y: -0.3, font: { size: 11 } },
      margin: { t: 8, b: 60, l: 55, r: 10 },
      showlegend: true,
    }, { responsive: true });
  });

  // Red-flag banner (global)
  const banner = document.getElementById('val-redflag-banner');
  if (banner) {
    if (redFlagLabels.length) {
      banner.textContent = `⚠ Overfitting detected: val loss rising for [${redFlagLabels.join(', ')}]. Consider stopping early.`;
      banner.style.display = 'block';
    } else {
      banner.style.display = 'none';
    }
  }
}

// ---- Online Training Health ----
// One card per online label, same style as the distillation training curve cards.
//
// online_serve.py writes:
//   train split: the training loss (EBE or KL depending on the run)
//   val   split: rejection_rate = 1 - alpha  (stored as val_loss)
//
// For online_ebe the train loss IS block efficiency (EBE loss = −BE), so we
// negate it → both train and eval lines go UP as the model improves.
// For online (KL) the train loss is KL divergence (down = better); eval alpha
// goes up = better.  These can't share one y-axis so we use dual axes — but
// kept on ONE clear chart per label with no derived formulas.
async function renderOnlineHealth() {
  const grid = document.getElementById('chart-online-health-grid');
  const curves = await fetch('/api/train_curves').then(r => r.json());

  const allLabels = [...new Set(curves.map(r => r.label))];
  const onlineLabels = allLabels.filter(l => l.includes('online')).sort();

  if (!onlineLabels.length) {
    grid.innerHTML = `<div class="chart-card" style="grid-column:1/-1">
      <p class="text-muted small mb-0">Online training health appears during
      online_adapt_gsm8k / online_ebe_adapt_gsm8k steps.
      Shows train loss + eval acceptance rate at each checkpoint.</p></div>`;
    return;
  }

  grid.innerHTML = '';

  onlineLabels.forEach((label, colorIdx) => {
    const labelRows = curves.filter(r => r.label === label);
    // ebe_single: loss = −mean(α), negate → mean α goes up as training improves.
    // ebe (block-level): loss = −BE, negate → BE goes up as training improves.
    // Both benefit from negation; distinguish for axis labels only.
    const isEBE       = label.includes('ebe');
    const isEBESingle = label.includes('ebe-single') || label.includes('ebe_single');

    // De-duplicate train rows by step
    const trainMap = {};
    labelRows.filter(r => !r.split || r.split === 'train')
             .forEach(r => { if (!trainMap[r.step] || r.id > trainMap[r.step].id) trainMap[r.step] = r; });
    const trainRows = Object.values(trainMap).sort((a, b) => a.step - b.step);

    // Val rows: online_serve.py stores rejection_rate as val_loss → alpha = 1 - val_loss
    const valRows = labelRows.filter(r => r.split === 'val').sort((a, b) => a.step - b.step);

    if (!trainRows.length && !valRows.length) return;

    const trainColor = TRAIN_COLORS[colorIdx % TRAIN_COLORS.length];
    const valColor   = VAL_COLORS  [colorIdx % VAL_COLORS.length];

    const traces = [];

    if (trainRows.length) {
      if (isEBE) {
        // EBE loss = −BE (negative, lower = better as training progresses).
        // Plot raw loss directly — consistent with other training charts (down = better).
        const trainTraceName = isEBESingle ? 'train loss (= −mean α)' : 'EBE loss (= −BE)';
        const trainHover     = isEBESingle
          ? 'step %{x}<br><b>EBE-single loss: %{y:.3f}</b>  [= −mean α]<extra></extra>'
          : 'step %{x}<br><b>EBE loss: %{y:.2f}</b>  [= −BE, lower = better]<extra></extra>';
        traces.push({
          type: 'scatter', mode: 'lines',
          name: trainTraceName,
          x: trainRows.map(r => r.step),
          y: trainRows.map(r => r.loss),   // raw loss: negative, lower = better
          line: { color: trainColor, width: 1.8, dash: 'solid' },
          yaxis: 'y',
          hovertemplate: trainHover,
        });
      } else {
        // KL loss: lower = better — show on left axis, down = better
        traces.push({
          type: 'scatter', mode: 'lines',
          name: `train KL loss`,
          x: trainRows.map(r => r.step),
          y: trainRows.map(r => r.loss),
          line: { color: trainColor, width: 1.8, dash: 'solid' },
          yaxis: 'y',
          hovertemplate: 'step %{x}<br>train KL loss: %{y:.4f}<extra></extra>',
        });
      }
    }

    if (valRows.length) {
      // Rejection rate = 1 − α = val_loss (lower = better, consistent with train loss direction).
      const rejVals = valRows.map(r => Math.min(1, Math.max(0, r.loss)));

      // Both EBE and KL: show rejection rate so the eval line also trends down = better.
      traces.push({
        type: 'scatter', mode: 'lines+markers',
        name: `eval 1−α (rejection)`,
        x: valRows.map(r => r.step),
        y: rejVals,
        line: { color: valColor, width: 2.2, dash: 'dash' },
        marker: { color: valColor, size: 7, symbol: 'circle', line: { width: 1.5, color: '#fff' } },
        yaxis: 'y2',
        hovertemplate: 'step %{x}<br><b>eval 1−α: %{y:.3f}</b>  [rejection rate, lower = better]<extra></extra>',
      });
    }

    if (!traces.length) return;

    // Build header stats
    const lastTrain    = trainRows.length ? trainRows[trainRows.length - 1] : null;
    // Show α (acceptance rate) in the header — more interpretable than rejection rate.
    const lastAlpha    = valRows.length
      ? (1 - Math.min(1, Math.max(0, valRows[valRows.length - 1].loss))).toFixed(3)
      : null;
    // Header shows BE = −loss for EBE (intuitive); raw loss for KL.
    const lastTrainVal = lastTrain
      ? (isEBESingle ? (-lastTrain.loss).toFixed(3)
         : isEBE     ? (-lastTrain.loss).toFixed(2)
         :              lastTrain.loss.toFixed(3))
      : null;
    const trainLabel   = isEBESingle ? 'mean α' : (isEBE ? 'BE' : 'KL loss');
    const trainDir     = '↓ lower = better';   // both axes now trend downward = better

    const statsHtml = [
      lastTrain   ? `step ${lastTrain.step}` : '',
      lastTrainVal ? `Train ${trainLabel} ${lastTrainVal}` : '',
      lastAlpha   ? `Eval α ${lastAlpha}` : '',
    ].filter(Boolean).join(' · ');

    const yLeftTitle  = isEBESingle ? 'EBE-single loss = −mean(α)  (↓ better)'
                      : isEBE       ? 'EBE loss = −BE  (↓ better)'
                      :               'Train KL loss  (↓ better)';
    const yRightTitle = 'eval 1−α  (↓ better)';

    // Left axis range — raw loss (negative for EBE, positive for KL)
    const trainY  = trainRows.map(r => r.loss);
    const yMin    = Math.min(...trainY);
    const yMax    = Math.max(...trainY);
    const pad     = (yMax - yMin) * 0.15 || 0.5;

    const chartId = `chart-oh-${label.replace(/[^a-z0-9]/gi, '_')}`;
    const card = document.createElement('div');
    card.className = 'chart-card';
    card.innerHTML = `
      <h6 style="text-transform:capitalize">${label}
        <small class="text-muted ms-2" style="font-size:0.75em">${statsHtml}</small>
        <small class="text-muted ms-1" style="font-size:0.7em;font-style:italic">${trainDir}</small>
      </h6>
      <div id="${chartId}" style="height:260px"></div>`;
    grid.appendChild(card);

    Plotly.newPlot(chartId, traces, {
      xaxis: { title: 'Step', tickfont: { size: 10 } },
      yaxis: {
        title: yLeftTitle,
        range: [yMin - pad, yMax + pad],
        tickfont: { size: 10 }, titlefont: { size: 10 },
        gridcolor: '#e9ecef',
      },
      yaxis2: {
        title: yRightTitle,
        overlaying: 'y', side: 'right',
        range: [0, 1],
        tickfont: { size: 10, color: valColor },
        titlefont: { size: 10, color: valColor },
        showgrid: false,
      },
      legend: { orientation: 'h', y: -0.3, font: { size: 11 } },
      margin: { t: 8, b: 65, l: 55, r: 55 },
    }, { responsive: true });
  });

  // Note about missing online-KL data: the KL online run (online_adapt_gsm8k)
  // completed before results_db logging was added to online_serve.py.
  // Only the EBE online run was captured.
  const hasEBE = onlineLabels.some(l => l.includes('ebe'));
  // KL = any online label that is NOT ebe-flavoured (matches online-gsm8k but not ebe variants)
  const hasKL  = onlineLabels.some(l => !l.includes('ebe'));
  if (hasEBE && !hasKL) {
    // online-gsm8k (KL) run completed before DB logging was added to online_serve.py
    const note = document.createElement('div');
    note.className = 'chart-card';
    note.style.cssText = 'grid-column:1/-1;background:#fffbf0;border-left:3px solid #fd7e14';
    note.innerHTML = `
      <p class="mb-0 small text-muted">
        <strong>Online-KL-Gsm8k training curves not available.</strong>
        The <code>online_adapt_gsm8k</code> run completed before training-curve DB
        logging was added to online_serve.py. Block efficiency results will still
        appear once <code>eval_online_gsm8k</code> finishes. To capture curves,
        re-run only that step: mark it pending in pipeline_state_laptop.json.
      </p>`;
    grid.appendChild(note);
  }
}

// ---- Table ----
const TABLE_COLS = ['id','ts','experiment_tag','draft_label','loss_name','train_steps','learning_rate',
                    'dataset','mode','K','temperature','n_prompts',
                    'alpha_mean','alpha_ci95','block_eff','alpha_eff','throughput','ms_per_tok','peak_vram_mb'];

const TABLE_COL_LABELS = {
  alpha_eff: 'α_eff =(BE−1)/K',
  alpha_mean: 'α_mean (measured)',
  block_eff: 'block_eff',
};

function renderTable() {
  if (!ALL_RUNS.length) return;
  const thead = document.querySelector('#runs-table thead');
  const tbody = document.querySelector('#runs-table tbody');
  thead.innerHTML = `<tr>${TABLE_COLS.map(c => {
    const lbl = TABLE_COL_LABELS[c] || c;
    const tip = c === 'alpha_eff'
      ? 'title="Derived from BE: mean accepted draft tokens per round / K. Exact: (block_eff−1)/K. No extra eval needed."'
      : '';
    return `<th ${tip} style="${c==='alpha_eff'?'background:#fff3cd;cursor:help':''}">${lbl}</th>`;
  }).join('')}</tr>`;
  tbody.innerHTML = ALL_RUNS.map(row =>
    `<tr>${TABLE_COLS.map(c => {
      const v = row[c];
      if (v == null) return '<td style="color:#aaa">—</td>';
      if (c === 'alpha_eff') {
        const pct = (v * 100).toFixed(1);
        const col = v > 0.55 ? '#198754' : v > 0.45 ? '#856404' : '#dc3545';
        return `<td style="font-weight:600;color:${col}">${v.toFixed(3)} <small style="font-weight:400;color:#888">(${pct}%)</small></td>`;
      }
      if (typeof v === 'number') return `<td>${Math.abs(v) < 1 ? v.toFixed(4) : v.toFixed(2)}</td>`;
      return `<td>${v}</td>`;
    }).join('')}</tr>`
  ).join('');
}

function exportCSV() {
  const lines = [TABLE_COLS.join(',')];
  ALL_RUNS.forEach(row => {
    lines.push(TABLE_COLS.map(c => {
      const v = row[c] ?? '';
      return typeof v === 'string' && v.includes(',') ? `"${v}"` : v;
    }).join(','));
  });
  const blob = new Blob([lines.join('\n')], {type:'text/csv'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'specdist_results.csv';
  a.click();
}

// ---- Pivot Table ----
function renderPivot() {
  const rowKey = document.getElementById('piv-row')?.value;
  const colKey = document.getElementById('piv-col')?.value;
  const valKey = document.getElementById('piv-val')?.value;
  const aggFn  = document.getElementById('piv-agg')?.value;
  if (!rowKey || !colKey || !valKey || !aggFn) return;

  const thead = document.querySelector('#pivot-table thead');
  const tbody = document.querySelector('#pivot-table tbody');
  const info  = document.getElementById('pivot-info');

  const runs = ALL_RUNS.filter(r => r[valKey] != null);
  if (!runs.length) {
    if (info) info.textContent = 'No data matching current filters — run the pipeline to generate results.';
    if (thead) thead.innerHTML = '';
    if (tbody) tbody.innerHTML = '<tr><td class="text-muted" style="padding:1rem">No data yet.</td></tr>';
    return;
  }

  // Unique sorted row / col values
  const rowVals = [...new Set(runs.map(r => String(r[rowKey])))].sort();
  const colVals = [...new Set(runs.map(r => String(r[colKey])))].sort();

  // Bucket runs into pivot[rowVal][colVal] = [metric values]
  const pivot = {};
  rowVals.forEach(rv => { pivot[rv] = {}; colVals.forEach(cv => { pivot[rv][cv] = []; }); });
  runs.forEach(r => {
    const rv = String(r[rowKey]), cv = String(r[colKey]);
    if (pivot[rv]?.[cv] !== undefined) pivot[rv][cv].push(Number(r[valKey]));
  });

  // Aggregate
  function agg(vals) {
    if (!vals.length) return null;
    if (aggFn === 'mean')  return vals.reduce((a,b) => a+b, 0) / vals.length;
    if (aggFn === 'max')   return Math.max(...vals);
    if (aggFn === 'min')   return Math.min(...vals);
    if (aggFn === 'count') return vals.length;
    return null;
  }

  // Find max value for conditional heat-shading (skip for 'count')
  let allVals = [];
  rowVals.forEach(rv => colVals.forEach(cv => {
    const v = agg(pivot[rv][cv]);
    if (v != null) allVals.push(v);
  }));
  const maxVal = allVals.length ? Math.max(...allVals) : 1;
  const minVal = allVals.length ? Math.min(...allVals) : 0;

  function cellBg(v) {
    if (v == null || aggFn === 'count' || maxVal === minVal) return '';
    // Light blue → dark blue heat: 0% → max value
    const t = (v - minVal) / (maxVal - minVal);
    const r = Math.round(255 - t * 127);
    const g = Math.round(255 - t * 127);
    const b = 255;
    return `background:rgba(${r},${g},${b},${0.2 + t * 0.55})`;
  }

  // Render
  thead.innerHTML = `<tr>
    <th style="background:#f1f3f5;font-weight:700;min-width:100px">${rowKey} / ${colKey}</th>
    ${colVals.map(cv => `<th style="text-align:center;font-size:12px">${cv}</th>`).join('')}
  </tr>`;

  tbody.innerHTML = rowVals.map(rv => {
    const cells = colVals.map(cv => {
      const v = agg(pivot[rv][cv]);
      const disp = v == null ? '—'
                 : aggFn === 'count' ? v
                 : v.toFixed(3);
      const bg = cellBg(v);
      return `<td style="text-align:center;${bg}">${disp}</td>`;
    }).join('');
    return `<tr><td style="font-weight:600;background:#f8f9fa;font-size:12px">${rv}</td>${cells}</tr>`;
  }).join('');

  const valLabel = document.getElementById('piv-val')?.selectedOptions[0]?.text || valKey;
  if (info) info.textContent =
    `${aggFn} of ${valLabel} — ${rowVals.length} row(s) × ${colVals.length} col(s) — ${runs.length} run(s) in scope`;
}

function exportPivotCSV() {
  // Build pivot first in case it hasn't been rendered yet
  renderPivot();
  const thead = document.querySelector('#pivot-table thead');
  const tbody = document.querySelector('#pivot-table tbody');
  if (!thead || !tbody) return;

  const escCell = cell => {
    const t = cell.textContent.trim();
    return t.includes(',') || t.includes('"') ? `"${t.replace(/"/g,'""')}"` : t;
  };

  const rows = [];
  thead.querySelectorAll('tr').forEach(tr => {
    rows.push([...tr.querySelectorAll('th,td')].map(escCell).join(','));
  });
  tbody.querySelectorAll('tr').forEach(tr => {
    rows.push([...tr.querySelectorAll('th,td')].map(escCell).join(','));
  });

  const blob = new Blob([rows.join('\n')], {type:'text/csv'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'pivot_report.csv';
  a.click();
}

// ---- Bootstrap tab change handlers ----
// Primary handlers are registered inside init() (after DIMS loads and filter chips exist).
// This second registration is intentionally a no-op — kept as a hook for future per-tab
// work that must fire even before init() completes (currently nothing).
document.querySelectorAll('[data-bs-toggle="tab"]').forEach(tab => {
  tab.addEventListener('shown.bs.tab', e => {
    const target = e.target.getAttribute('href');
    // Additional per-tab side-effects can go here; chart rendering handled by init() block.
  });
});

// ---- Pipeline status bar ----
const STATUS_COLOR = {
  done: '#198754', running: '#ffc107', failed: '#dc3545',
  stopped: '#e67e22', pending: '#6c757d'
};

async function updatePipelineStatus() {
  try {
    const [ps, gs] = await Promise.all([
      fetch('/api/pipeline_status').then(r => r.json()),
      fetch('/api/gpu_status').then(r => r.json()),
    ]);

    // Current step
    const currentEl = document.getElementById('bar-current-step');
    const phaseEl   = document.getElementById('bar-phase');
    if (ps.found) {
      const running  = ps.steps.filter(s => s.status === 'running');
      const failed   = ps.steps.filter(s => s.status === 'failed');
      const stopped  = ps.steps.filter(s => s.status === 'stopped');
      if (running.length) {
        currentEl.innerHTML =
          `<span class="step-badge badge-running">${running[0].id}</span>`;
        phaseEl.textContent = running[0].phase || '';
      } else if (failed.length) {
        currentEl.innerHTML =
          `<span class="step-badge badge-failed">FAILED: ${failed[0].id}</span>`;
        phaseEl.textContent = '';
      } else if (stopped.length) {
        currentEl.innerHTML =
          `<span class="step-badge badge-stopped">STOPPED (partial): ${stopped[0].id}</span>`;
        phaseEl.textContent = 'Early stop — partial model saved, pipeline continues';
      } else if (ps.n_done === ps.n_total) {
        currentEl.innerHTML =
          `<span class="step-badge badge-done">Pipeline complete</span>`;
        phaseEl.textContent = 'All ${ps.n_total} steps done';
      } else {
        currentEl.innerHTML =
          `<span class="step-badge badge-pending">idle / paused</span>`;
        phaseEl.textContent = '';
      }

      // Progress bar
      document.getElementById('progress-bar-inner').style.width = ps.pct + '%';
      document.getElementById('bar-progress-text').textContent =
        `${ps.n_done}/${ps.n_total} steps (${ps.pct}%)`;

      // Step panel — single chip row + detached dropdown.
      // All phase chips stay on ONE line; clicking a chip toggles a dropdown
      // BELOW the row showing that phase's step badges. No layout shift.
      const panel = document.getElementById('step-panel');
      const statusMap = {};
      ps.steps.forEach(s => statusMap[s.id] = s.status);
      window._pipelineStatusMap = statusMap;   // shared with togglePhase()

      const chips = PHASE_GROUPS.map((ph, pi) => {
        const phDone    = ph.steps.filter(s => ['done','stopped'].includes(statusMap[s])).length;
        const phRunning = ph.steps.some(s => statusMap[s] === 'running');
        const phFailed  = ph.steps.some(s => statusMap[s] === 'failed');
        const col = phRunning ? '#ffc107' : phFailed ? '#dc3545'
                  : phDone === ph.steps.length ? '#198754' : '#6c757d';
        const icon = phRunning ? '&#9679;&nbsp;' : phFailed ? '&#10005;&nbsp;'
                   : phDone === ph.steps.length ? '&#10003;&nbsp;' : '';
        return `<button onclick="togglePhase(${pi})" data-pi="${pi}"
          style="cursor:pointer;padding:2px 9px;border-radius:12px;
                 border:1px solid ${col}66;background:${col}1a;color:${col};
                 font-size:10px;font-weight:700;white-space:nowrap;
                 margin:1px 3px 1px 0;outline:none"
        >${icon}${ph.label}&nbsp;<span style="opacity:.7">${phDone}/${ph.steps.length}</span></button>`;
      }).join('');

      panel.innerHTML =
        `<div style="display:flex;flex-wrap:nowrap;align-items:center;overflow-x:auto;padding-bottom:2px">${chips}</div>` +
        `<div id="phase-dropdown" style="display:none;flex-wrap:wrap;gap:3px;
           margin-top:4px;padding:4px 2px 2px 2px;
           border-top:1px solid #ffffff1a"></div>`;

      // Auto-open the first running / failed phase
      const autoIdx = PHASE_GROUPS.findIndex(ph =>
        ph.steps.some(s => (statusMap[s]==='running'||statusMap[s]==='failed')));
      if (autoIdx >= 0) togglePhase(autoIdx);

      // Current combo sub-progress (from be_progress.log parse)
      const comboEl = document.getElementById('bar-current-combo');
      if (ps.current_combo) {
        const beTotal = ps.n_be_total ? `/${ps.n_be_total}` : '';
        comboEl.textContent = `[BE ${ps.n_be_done}${beTotal}] ${ps.current_combo}`;
      } else if (running.length && ps.n_be_total == null) {
        comboEl.textContent = '';  // step running but no GBV batch started yet
      } else {
        comboEl.textContent = '';
      }

      // Latest DB entry
      if (ps.latest_db_runs && ps.latest_db_runs.length) {
        const r = ps.latest_db_runs[ps.latest_db_runs.length - 1];
        const val = r.alpha_mean != null ? `alpha=${r.alpha_mean.toFixed(3)}`
                  : r.block_eff  != null ? `BE=${r.block_eff.toFixed(3)}`
                  : r.perplexity != null ? `PPL=${r.perplexity.toFixed(2)}`
                  : '';
        document.getElementById('bar-latest').textContent =
          `DB: ${r.draft_label}/${r.dataset}/${r.mode}  ${val}`;
      }
    } else {
      currentEl.textContent = 'No pipeline state file found';
      phaseEl.textContent = '';
      document.getElementById('bar-current-combo').textContent = '';
    }

    // GPU bar
    if (gs.total_gb) {
      const pct = gs.pct_used || 0;
      document.getElementById('gpu-bar-fill').style.width = pct + '%';
      document.getElementById('gpu-bar-fill').style.background =
        pct > 85 ? '#dc3545' : pct > 65 ? '#ffc107' : '#fd7e14';
      document.getElementById('bar-gpu-text').textContent =
        `${gs.used_gb}/${gs.total_gb} GB`;
    }
  } catch (e) {
    document.getElementById('bar-current-step').textContent = 'status unavailable';
  }
}

function toggleStepPanel() {
  _stepPanelOpen = !_stepPanelOpen;
  document.getElementById('step-panel').style.display =
    _stepPanelOpen ? 'flex' : 'none';
}

let _logSource = 'pipeline';  // 'pipeline' = training log | 'eval' = GBV eval detail log

function toggleLogPanel() {
  _logPanelOpen = !_logPanelOpen;
  const panel = document.getElementById('log-panel');
  const btn   = document.getElementById('btn-logs');
  panel.style.display = _logPanelOpen ? 'block' : 'none';
  btn.style.background = _logPanelOpen ? '#2ea043' : '#1a3a1a';
  if (_logPanelOpen) refreshLogPanel();
}

async function refreshLogPanel() {
  if (!_logPanelOpen) return;
  try {
    const data = await fetch(`/api/log_tail?lines=100&source=${_logSource}`).then(r => r.json());
    const panel = document.getElementById('log-panel');

    // Source-toggle bar
    const toggleBar = `
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;font-size:10px;color:#aaa">
        <span>Showing:</span>
        <button onclick="_logSource='pipeline';refreshLogPanel()"
          style="padding:1px 7px;border-radius:3px;font-size:10px;cursor:pointer;
                 background:${_logSource==='pipeline'?'#2ea043':'#333'};
                 color:#fff;border:1px solid #555">
          Training log
        </button>
        <button onclick="_logSource='eval';refreshLogPanel()"
          style="padding:1px 7px;border-radius:3px;font-size:10px;cursor:pointer;
                 background:${_logSource==='eval'?'#2ea043':'#333'};
                 color:#fff;border:1px solid #555">
          Eval detail
        </button>
        <span style="margin-left:auto;color:#555">${data.log_file || ''} — ${data.total_lines||0} lines (last 100)</span>
      </div>`;

    if (data.found && data.lines.length) {
      // Color-code key lines
      const html = data.lines.map(line => {
        const esc = line.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
        // Training health checks
        if (/HEALTH CHECK|IMPROVING|WORSENING/i.test(line))
          return `<span style="color:#a8ff78;font-weight:bold">${esc}</span>`;
        if (/\[HEALTH\].*⚠|COLLAPSED|NaN\/Inf|MISSING|RISING/i.test(line))
          return `<span style="color:#ff6b6b;font-weight:bold">${esc}</span>`;
        if (/\[HEALTH\].*✓|Checkpoint verified|within threshold/i.test(line))
          return `<span style="color:#69db7c">${esc}</span>`;
        if (/\[RESUME\]/i.test(line))
          return `<span style="color:#74c0fc;font-weight:bold">${esc}</span>`;
        if (/val_loss:/i.test(line))
          return `<span style="color:#da77f2">${esc}</span>`;
        if (/train_loss:|Step\s+\d+/i.test(line))
          return `<span style="color:#e9ecef">${esc}</span>`;
        if (/Early stopping|PPL=|Baseline PPL/i.test(line))
          return `<span style="color:#ffd43b">${esc}</span>`;
        // Eval log lines
        if (/Block efficiency/i.test(line))
          return `<span style="color:#ffd700;font-weight:bold">${esc}</span>`;
        if (/\[WARN\]|\[OOM\]|\[ERROR\]/i.test(line))
          return `<span style="color:#ff6b6b">${esc}</span>`;
        if (/STEP :|={3,}/i.test(line))
          return `<span style="color:#868e96">${esc}</span>`;
        return esc;
      }).join('');
      panel.innerHTML = toggleBar + html;
      panel.scrollTop = panel.scrollHeight;
    } else {
      panel.innerHTML = toggleBar +
        `<span style="color:#666">${data.msg || 'Log empty — pipeline not started yet'}</span>`;
    }
  } catch (e) {
    const p = document.getElementById('log-panel');
    if (p) p.textContent = 'Error fetching log: ' + e;
  }
}

// ---- Auto-refresh ----
function startAutoRefresh(intervalSecs = 30) {
  if (_refreshTimer) clearInterval(_refreshTimer);
  _countdown = intervalSecs;

  // Countdown ticker (every second)
  const tick = setInterval(() => {
    _countdown--;
    const el = document.getElementById('refresh-countdown');
    if (el) el.textContent = `↻ ${_countdown}s`;
    if (_countdown <= 0) _countdown = intervalSecs;
  }, 1000);

  // Full refresh every intervalSecs
  _refreshTimer = setInterval(async () => {
    _countdown = intervalSecs;
    await Promise.all([loadData(), updatePipelineStatus()]);
    refreshLogPanel();   // no-op when panel is closed
  }, intervalSecs * 1000);
}

init();
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(_HTML)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--host", default="127.0.0.1",
                   help="Bind address. Use 0.0.0.0 for Colab/cloud port proxying.")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--db", metavar="PATH",
                   help="Path to a results.db SQLite file. Overrides SPECDIST_DB_PATH "
                        "and the default db/results.db location.")
    p.add_argument("--root", metavar="DIR",
                   help="Convenience: point DB, logs, and pipeline-state all at one "
                        "directory. Equivalent to setting --db DIR/results.db "
                        "--logs DIR --state DIR. Useful when you download Kaggle output "
                        "into a single local folder.")
    p.add_argument("--logs", metavar="DIR",
                   help="Directory that contains pipeline_output.log / be_progress.log. "
                        "Overrides SPECDIST_LOGS_ROOT.")
    p.add_argument("--state", metavar="DIR",
                   help="Directory that contains pipeline_state_*.json files. "
                        "Overrides SPECDIST_STORAGE_ROOT.")
    args = p.parse_args()

    # --root is a shorthand that sets DB, logs, and state all to the same folder.
    if args.root:
        root = os.path.abspath(args.root)
        if not args.db:
            args.db = os.path.join(root, "results.db")
        if not args.logs:
            args.logs = root
        if not args.state:
            args.state = root

    # Override DB path (CLI > env var > default)
    if args.db:
        results_db.DB_PATH = os.path.abspath(args.db)

    # Override log directory (_BE_LOG / _PIPELINE_LOG are module-level; no global needed)
    if args.logs:
        _logs_dir = os.path.abspath(args.logs)
        _BE_LOG       = os.path.join(_logs_dir, "be_progress.log")
        _PIPELINE_LOG = os.path.join(_logs_dir, "pipeline_output.log")

    # Override pipeline-state directory via env var (api_pipeline_status reads it)
    if args.state:
        os.environ["SPECDIST_STORAGE_ROOT"] = os.path.abspath(args.state)

    # Ensure DB exists / schema is initialised
    results_db._connect().close()

    print(f"\nSpecDist Dashboard  -> http://{args.host}:{args.port}/")
    print(f"Database            -> {results_db.DB_PATH}")
    print(f"Logs dir            -> {os.path.dirname(_PIPELINE_LOG)}")
    state_dir = (os.environ.get("SPECDIST_STORAGE_ROOT")
                 or os.path.join(_GBV_RESEARCH, "orchestration"))
    print(f"Pipeline state dir  -> {state_dir}\n")
    app.run(host=args.host, port=args.port, debug=args.debug)
