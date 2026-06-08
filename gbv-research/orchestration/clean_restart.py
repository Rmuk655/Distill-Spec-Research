"""
clean_restart.py — Wipe pipeline outputs for ONE config and restart from scratch.

Usage:
    python orchestration/clean_restart.py --config laptop_gpt2   # GPT-2 only
    python orchestration/clean_restart.py --config kaggle         # Kaggle Qwen only
    python orchestration/clean_restart.py --config laptop_qwen    # laptop Qwen only
    python orchestration/clean_restart.py --dry_run               # show what would be deleted
    python orchestration/clean_restart.py --no_restart            # wipe only, don't relaunch

Scope — only touches outputs belonging to the specified config's model pair:
    db/checkpoints/*-{pair_tag}*/    — checkpoints for THIS pair only (NOT other models)
    db/logs/{slug}-{pair_tag}/       — run logs for THIS pair only
    db/results.db rows               — DB rows where draft_path contains the pair tag
    pipeline_state_{slug}*.json      — state file(s) for this config

What it NEVER touches:
    Checkpoints from OTHER model pairs (e.g. restarting laptop_gpt2 keeps Qwen checkpoints)
    DB rows from OTHER models
    db/wandb — WandB run IDs have no pair-tag; manage via wandb.ai UI
    Source code, configs, datasets

Pair tag is derived from the YAML config's models.draft / models.target / load_in_4bit:
    laptop_gpt2  (distilgpt2→gpt2-medium)        → pair tag: dg2-g2m
    laptop_qwen  (Qwen2.5-0.5B→Qwen3-0.6B)       → pair tag: q0.5b-q0.6b
    kaggle       (Qwen3-0.6B→Qwen3-8B NF4)        → pair tag: q0.6b-q8bnf4
    a100_qwen    (Qwen3-0.6B→Qwen3-8B BF16)       → pair tag: q0.6b-q8b
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


# ── Pair-tag derivation ────────────────────────────────────────────────────

def _get_config_pair_info(config_name: str):
    """Derive the pair tag and draft model ID from a config YAML.

    Returns (pair_tag, draft_model_id).
    pair_tag encodes (draft, target, load_in_4bit) as a short string —
    the same tag that experiment.py appends to checkpoint dir names.

    Returns ("", "") if the YAML cannot be loaded or models.draft/target are missing.
    """
    try:
        import yaml, re as _re

        def _deep_merge(parent, child):
            result = dict(parent)
            for k, v in child.items():
                if k == "_base": continue
                if isinstance(v, dict) and isinstance(result.get(k), dict):
                    result[k] = _deep_merge(result[k], v)
                else:
                    result[k] = v
            return result

        def _load_raw(name, seen=None):
            if seen is None: seen = set()
            if name in seen: return {}
            seen = seen | {name}
            p = os.path.join(_HERE, "configs", f"{name}.yaml")
            if not os.path.exists(p): return {}
            with open(p, encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
            base = raw.pop("_base", None)
            if base:
                return _deep_merge(_load_raw(str(base), seen), raw)
            return raw

        raw = _load_raw(config_name)
        models = raw.get("models", {})
        hardware = raw.get("hardware", {})
        draft = models.get("draft", "")
        target = models.get("target", "")
        load_in_4bit = bool(hardware.get("load_in_4bit", False))

        if not draft or not target:
            return ""

        # Same _run_tag() logic as experiment.py
        _FAM = [("qwen","q"),("llama","l"),("gemma","g"),("mistral","m"),("phi","p"),("falcon","f")]
        def _short(mid):
            name = mid.split("/")[-1].lower()
            if "distilgpt2" in name or "distil-gpt2" in name: return "dg2"
            if "gpt2" in name or "gpt-2" in name:
                sz = "m" if "medium" in name else ("l" if "large" in name else ("xl" if "xl" in name else ""))
                return f"g2{sz}"
            sz_m = _re.search(r"(\d+\.?\d*)\s*([bm])", name, _re.I)
            size = ""
            if sz_m:
                num, unit = sz_m.group(1), sz_m.group(2).lower()
                if "." in num: num = num.rstrip("0").rstrip(".")
                size = f"{num}{unit}"
            for kw, pfx in _FAM:
                if kw in name: return pfx + size
            return (name.replace("-","")[:6] + size)[:10]

        d = _short(draft)
        t = _short(target) + ("nf4" if load_in_4bit else "")
        return f"{d}-{t}", draft   # (pair_tag, draft_model_id)
    except Exception:
        return "", ""


def _get_config_pair_tag(config_name: str) -> str:
    """Convenience wrapper — returns just the pair tag string."""
    tag, _ = _get_config_pair_info(config_name)
    return tag


# ── Directories / files to wipe ────────────────────────────────────────────

def _wipe_targets(config_slug=None, pair_tag="", db_root=None):
    """Return (path, keep_dir, description) tuples to wipe.

    Scoped by pair_tag so that restarting one model family never touches
    another family's checkpoints or eval results.

    config_slug: sanitised config name  (e.g. "laptop_gpt2")
    pair_tag:    model-pair identifier  (e.g. "dg2-g2m", "q0.5b-q0.6b")

    Config-specific restart (config_slug + pair_tag given):
      - db/checkpoints/*-{pair_tag}*   ← only THIS pair's checkpoints
      - db/logs/{slug}-{pair_tag}/     ← only THIS pair's logs
      - db/results.db rows             ← handled separately in _wipe_results_db_rows()
      - db/wandb                       ← skipped (WandB run IDs have no pair-tag; manage via UI)

    Full wipe (no config_slug) → wipes everything: all checkpoints, all logs, all results,
    WandB, and orchestration/wandb stale runs.
    """
    import glob as _glob
    # Use db_root if provided (e.g. storage_root from --storage_root flag),
    # otherwise default to gbv-research/db/
    db = db_root or os.path.join(_GBV_RESEARCH, "db")

    targets = []  # built up below based on scope

    if config_slug and pair_tag:
        slug = config_slug.replace("/", "_").replace("\\", "_")

        # ── Checkpoints: only dirs belonging to THIS pair tag ──────────────
        # Checkpoint dirs are named  <loss>-<dataset>-<pair_tag>/  and
        # <loss>-<dataset>-<pair_tag>_merged/  so we can safely glob for them.
        ckpt_base = os.path.join(db, "checkpoints")
        ckpt_dirs = (
            _glob.glob(os.path.join(ckpt_base, f"*-{pair_tag}"))
            + _glob.glob(os.path.join(ckpt_base, f"*-{pair_tag}_merged"))
            + _glob.glob(os.path.join(ckpt_base, "smoke", f"*-{pair_tag}"))
            + _glob.glob(os.path.join(ckpt_base, "smoke", f"*-{pair_tag}_merged"))
        )
        if ckpt_dirs:
            for d in sorted(ckpt_dirs):
                rel = os.path.relpath(d, _GBV_RESEARCH)
                targets.append((d, False, f"{rel}/"))
        else:
            print(f"  [info] No checkpoints found for pair tag '{pair_tag}'")

        # ── Logs: only this config's run subdirs ───────────────────────────
        log_subdirs = _glob.glob(os.path.join(db, "logs", f"{slug}-*"))
        for d in sorted(log_subdirs):
            targets.append((d, False, f"db/logs/{os.path.basename(d)}/ (run logs)"))

        # db/results.db rows are handled separately by _wipe_results_db_rows()
        # db/wandb: skipped — WandB run IDs have no pair-tag; user manages via wandb.ai

    else:
        # ── Full wipe: no config specified ────────────────────────────────
        # Wipe everything: all checkpoints, all logs, all results, WandB.
        targets = [
            (os.path.join(db, "checkpoints"), True,  "db/checkpoints (ALL trained models)"),
            (os.path.join(db, "logs"),         True,  "db/logs (ALL pipeline logs)"),
            (os.path.join(db, "wandb"),        True,  "db/wandb (ALL local WandB runs)"),
            (os.path.join(db, "results.db"),   False, "db/results.db (ALL eval results)"),
            (os.path.join(_HERE, "wandb"),     False, "orchestration/wandb (stale WandB)"),
        ]

    return targets


def _wipe_results_db_rows(pair_tag: str, draft_model: str = "", dry_run=False, storage_root=""):
    """Delete rows from results.db for this model pair.

    Two types of rows are cleaned:

    1. TRAINED model rows — draft_path contains the pair tag:
       e.g.  db/checkpoints/kl-gsm8k-dg2-g2m_merged  → LIKE '%dg2-g2m%'
       These are rows written when evaluating a LoRA-trained checkpoint.

    2. BASELINE rows — draft_path == draft_model_id (raw HF model name):
       e.g.  distilgpt2  or  Qwen/Qwen2.5-0.5B
       The baseline eval uses the untrained draft directly.  evaluate.py's
       _already_run() for baseline checks ONLY draft_label + dataset + mode
       (not the path), so old baseline rows survive a pair-tag-only DB wipe
       and cause all baseline cells to SKIP on the next run.
    """
    # Look for results.db in storage_root first (used when --storage_root is set),
    # then fall back to the default db/ directory inside the repo.
    db_candidates = []
    if storage_root and os.path.isdir(storage_root):
        db_candidates.append(os.path.join(storage_root, "results.db"))
    db_candidates.append(os.path.join(_GBV_RESEARCH, "db", "results.db"))
    db_path = next((p for p in db_candidates if os.path.exists(p)), None)
    if not db_path:
        print(f"  [skip] results.db not found — nothing to clear")
        return

    try:
        import sqlite3
        with sqlite3.connect(db_path) as conn:
            cur = conn.cursor()

            # ── Count rows to delete ──────────────────────────────────────────
            # 1. Trained model eval rows (pair tag in checkpoint path)
            cur.execute("SELECT COUNT(*) FROM runs WHERE draft_path LIKE ?",
                        (f"%{pair_tag}%",))
            n_trained = cur.fetchone()[0]

            # 2. Baseline eval rows (draft_path == raw HF model ID, no pair tag)
            n_baseline = 0
            if draft_model:
                cur.execute("SELECT COUNT(*) FROM runs WHERE draft_path = ?",
                            (draft_model,))
                n_baseline = cur.fetchone()[0]

            # 3. Training curves (label contains pair tag, e.g. "jsd-gsm8k-dg2-g2m")
            n_curves = 0
            try:
                cur.execute("SELECT COUNT(*) FROM train_curves WHERE label LIKE ?",
                            (f"%{pair_tag}%",))
                n_curves = cur.fetchone()[0]
            except sqlite3.OperationalError:
                pass   # table may not exist in old DBs

            n_total = n_trained + n_baseline + n_curves
            if n_total == 0:
                print(f"  [skip] db/results.db — no rows for pair '{pair_tag}'")
                return

            if dry_run:
                print(f"  [dry_run] Would delete: {n_trained} trained eval, "
                      f"{n_baseline} baseline eval, {n_curves} training curve row(s)")
                return

            # ── Delete ────────────────────────────────────────────────────────
            if n_trained:
                cur.execute("DELETE FROM runs WHERE draft_path LIKE ?",
                            (f"%{pair_tag}%",))
            if n_baseline and draft_model:
                cur.execute("DELETE FROM runs WHERE draft_path = ?",
                            (draft_model,))
            if n_curves:
                cur.execute("DELETE FROM train_curves WHERE label LIKE ?",
                            (f"%{pair_tag}%",))

            # Remove orphaned per_prompt rows
            try:
                cur.execute("DELETE FROM per_prompt WHERE run_id NOT IN "
                            "(SELECT id FROM runs)")
            except sqlite3.OperationalError:
                pass

            conn.commit()

        msg_parts = []
        if n_trained:  msg_parts.append(f"{n_trained} eval (trained)")
        if n_baseline: msg_parts.append(f"{n_baseline} eval (baseline '{draft_model}')")
        if n_curves:   msg_parts.append(f"{n_curves} training curves")
        print(f"  [clear] db/results.db — deleted {' + '.join(msg_parts)} [ok]")
    except Exception as e:
        print(f"  [warn] Could not clear results.db rows: {e}")


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


def _wipe(config_slug=None, pair_tag="", dry_run=False, db_root=None):
    """Wipe outputs for this config/pair.  db_root overrides the default db/ directory
    (use it to wipe storage_root artifacts when --storage_root was set)."""
    effective_db = db_root or os.path.join(_GBV_RESEARCH, "db")
    for path, keep_dir, desc in _wipe_targets(config_slug=config_slug, pair_tag=pair_tag,
                                               db_root=effective_db):
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
    ]:
        os.makedirs(keep_path, exist_ok=True)


# ── Reset pipeline state ───────────────────────────────────────────────────

def _state_glob(config_slug: str, storage_root: str = ""):
    """Return all state file paths matching this config slug.

    Looks in two places:
      1. orchestration/ (default, local dev)
      2. storage_root/ (when --storage_root is set, e.g. /home/krishnan/specdist)
         experiment.py writes state files there when --storage_root is passed.

    State files encode the model-pair tag:
        pipeline_state_{slug}-{pair_tag}.json
        pipeline_state_{slug}-{pair_tag}_smoke.json
    """
    import glob as _glob
    slug = config_slug.replace("/", "_").replace("\\", "_")

    def _patterns(directory):
        return [
            os.path.join(directory, f"pipeline_state_{slug}-*.json"),
            os.path.join(directory, f"pipeline_state_{slug}.json"),
        ]

    matches = []
    for pat in _patterns(_HERE):
        matches.extend(_glob.glob(pat))
    if storage_root and os.path.isdir(storage_root):
        for pat in _patterns(storage_root):
            matches.extend(_glob.glob(pat))

    return sorted(set(matches))


def _reset_state(config_slug: str, dry_run=False, storage_root=""):
    """Reset all state files for this config slug to all-pending.

    Finds every pipeline_state_{slug}*.json file (including pair-tag and
    smoke variants) and resets them.  experiment.py will re-create any
    missing files on the next run.
    """
    paths = _state_glob(config_slug, storage_root=storage_root)
    if not paths:
        slug = config_slug.replace("/", "_").replace("\\", "_")
        print(f"  [skip] pipeline_state_{slug}*.json not found — will be created by experiment.py")
        return

    for path in paths:
        fname = os.path.basename(path)
        if dry_run:
            print(f"  [dry_run] Would reset {fname} -> all steps pending")
            continue
        try:
            with open(path) as f:
                data = json.load(f)
            step_ids = list(data.get("steps", {}).keys())
            fresh = {
                "version": data.get("version", 1),
                "config":  config_slug,
                "steps":   {sid: {"status": "pending"} for sid in step_ids},
            }
        except Exception:
            # Corrupted state file — remove it; experiment.py recreates
            os.remove(path)
            print(f"  [del]   {fname} (corrupted) -> will be recreated [ok]")
            continue
        with open(path, "w") as f:
            json.dump(fresh, f, indent=2)
        print(f"  [reset] {fname} -> all steps pending [ok]")


# ── Launch pipeline ────────────────────────────────────────────────────────

def _launch(config: str, extra_args: list, dry_run=False):
    pipeline_script = os.path.join(_HERE, "experiment.py")
    slug = config.replace("/", "_").replace("\\", "_")
    cmd = [sys.executable, pipeline_script, "--config", config, "--yes"] + extra_args
    print(f"\n  Launching: {' '.join(os.path.basename(p) if os.sep in p else p for p in cmd)}")
    if dry_run:
        print("  [dry_run] Would launch pipeline")
        return
    os.makedirs(os.path.join(_GBV_RESEARCH, "db", "logs"), exist_ok=True)
    # experiment.py redirects its own stdout to  db/logs/{slug}-{pair_tag}/pipeline_output.log
    # as soon as it knows the run slug.  We do NOT redirect here — that would create a
    # separate startup log duplicating the pipeline log.  stdout → DEVNULL for the brief
    # pre-redirect output (just HF caching status, not needed externally).
    kwargs = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.DETACHED_PROCESS
    proc = subprocess.Popen(
        cmd, cwd=_GBV_RESEARCH,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        **kwargs
    )
    print(f"  Pipeline started (PID {proc.pid}) [ok]")
    print(f"\n  Log:         tail -f db/logs/{slug}-*/pipeline_output.log")
    print(f"               (dir appears 2-3 s after startup)")
    print(f"  Dashboard:   http://127.0.0.1:5000  (auto-shows most recent run)")
    print(f"  Status:      python orchestration/experiment.py --config {config} --status")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Wipe all pipeline outputs and restart clean.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python orchestration/clean_restart.py --config a100_qwen   # full wipe + restart
  python orchestration/clean_restart.py --config a100_qwen --no_restart   # wipe, don't restart
  python orchestration/clean_restart.py --config a100_qwen --reset_state_only  # ONLY reset state, keep checkpoints/DB

When to use what:
  Failed steps, want to retry without wiping data:
    → python deploy/aip_run.py --config a100_qwen --resume
    → (pipeline auto-retries failed steps, skips completed ones)
  OR: reset only the pipeline state (keep checkpoints + DB rows):
    → python orchestration/clean_restart.py --config a100_qwen --reset_state_only
  Full clean start (wipe everything):
    → python orchestration/clean_restart.py --config a100_qwen --no_restart
        """)
    p.add_argument("--config", default="laptop_qwen",
                   help="Pipeline config (any value accepted by experiment.py, "
                        "e.g. laptop_qwen, laptop_gpt2, laptop_llama, kaggle, "
                        "a100_qwen, server_gpt2, profiles/kl_only). "
                        "Default: laptop_qwen")
    p.add_argument("--storage_root", default=None,
                   help="Persistent storage root used when the pipeline ran "
                        "(e.g. /home/krishnan/specdist). When set, state files, "
                        "results.db, checkpoints, and logs are looked up there "
                        "instead of the default db/ directory inside the repo. "
                        "Must match the --storage_root passed to experiment.py.")
    p.add_argument("--dry_run", action="store_true",
                   help="Show what would be done, make no changes")
    p.add_argument("--no_restart", action="store_true",
                   help="Wipe only — do not relaunch the pipeline")
    p.add_argument("--reset_state_only", action="store_true",
                   help="Reset pipeline state file to all-pending WITHOUT wiping "
                        "checkpoints, results.db, or logs.  Use this to retry failed "
                        "steps without losing completed work.  Equivalent to just "
                        "re-running experiment.py --yes (which auto-retries failed).")
    # Pass-through args forwarded to experiment.py (e.g. --device cpu)
    p.add_argument("--device", default=None,
                   help="Forwarded to experiment.py (e.g. --device cpu)")
    p.add_argument("--losses", default=None,
                   help="Forwarded to experiment.py (e.g. --losses kl,bv_tree)")
    args = p.parse_args()
    # Also read STORAGE_ROOT from env (set by aip_gpu_setup.sh / ~/.specdist_env)
    if not args.storage_root:
        args.storage_root = os.environ.get("STORAGE_ROOT", "")

    dry = args.dry_run
    tag = " [DRY RUN]" if dry else ""

    # Derive pair tag AND draft model ID from YAML.
    # Both are needed: pair tag scopes trained-model rows; draft model ID
    # scopes baseline rows (which have draft_path=<hf_model_id>, no pair tag).
    _config_slug = args.config.replace("/", "_").replace("\\", "_")
    _pair_tag, _draft_model = _get_config_pair_info(args.config)
    _sr = args.storage_root or ""   # storage root (may be empty for local dev)

    print(f"\n{'='*60}")
    print(f"  GBV Clean Restart{tag}")
    print(f"  Config : {args.config}")
    if _sr:
        print(f"  Storage: {_sr}")
    if _pair_tag:
        print(f"  Pair   : {_pair_tag}  (trained rows + baseline for '{_draft_model}')")
    else:
        print(f"  Pair   : (unknown — full wipe)")
    print(f"{'='*60}\n")

    # ── State-only reset (no data wipe) ───────────────────────────────────────
    if args.reset_state_only:
        print("Mode: --reset_state_only — resetting pipeline state ONLY.")
        print("      Checkpoints, results.db, and logs are NOT touched.")
        print("      Use this to retry failed steps without losing completed work.")
        print("      Tip: 'python deploy/aip_run.py --resume' does the same thing.\n")
        print("3. Resetting pipeline state...")
        _reset_state(args.config, dry_run=dry, storage_root=_sr)
        print(f"\n{'='*60}")
        print(f"  Done{tag} — state reset, no data wiped")
        print(f"  Re-run: python deploy/aip_run.py --config {args.config} --resume")
        print(f"{'='*60}\n")
        return
    # ──────────────────────────────────────────────────────────────────────────

    print("1. Killing running pipeline processes...")
    _kill_pipeline_processes(dry_run=dry)

    print("\n2. Wiping outputs...")
    _wipe(config_slug=_config_slug, pair_tag=_pair_tag, dry_run=dry)
    # If storage_root was used, also wipe there (checkpoints, logs, etc.)
    if _sr:
        _wipe(config_slug=_config_slug, pair_tag=_pair_tag, dry_run=dry,
              db_root=_sr)

    print("\n2b. Clearing results.db rows for this model pair...")
    if _pair_tag:
        _wipe_results_db_rows(_pair_tag, _draft_model, dry_run=dry, storage_root=_sr)
    else:
        print("  [skip] No pair tag — results.db untouched (YAML could not be read)")

    print("\n3. Resetting pipeline state...")
    # _reset_state globs for pipeline_state_{slug}*.json so it catches both
    # pair-tag variants and smoke variants in both HERE and storage_root.
    _reset_state(args.config, dry_run=dry, storage_root=_sr)

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
