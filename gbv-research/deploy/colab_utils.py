"""
colab_utils.py — shared helpers for SpecDist Colab notebooks.

Loaded after git clone, so it is NOT available during the initial clone step.
The bootstrap cell inlines a minimal _secret() + clone block, then does:
    sys.path.insert(0, f"{GBV_DIR}/deploy")
    from colab_utils import install_deps, auth_wandb, auth_hf, check_gpu, run_pipeline
and delegates everything else here.
"""

import datetime
import glob
import json
import os
import subprocess
import sys
import threading
import time

# ---------------------------------------------------------------------------
# Default paths (ephemeral container)
# ---------------------------------------------------------------------------
REPO_URL   = "https://github.com/Rmuk655/Distill-Spec-Research.git"
REPO_DIR   = "/content/Distill-Spec-Research"
GBV_DIR    = f"{REPO_DIR}/gbv-research"
DRIVE_ROOT = "/content/drive/MyDrive/specdist"


# ---------------------------------------------------------------------------
# Secrets
# ---------------------------------------------------------------------------

def get_secret(name: str) -> str:
    """Read from Colab Secrets, then environment variables."""
    try:
        from google.colab import userdata  # type: ignore
        v = userdata.get(name)
        if v:
            return v
    except Exception:
        pass
    return os.environ.get(name, "")


def _auth_repo_url(base_url: str) -> str:
    """Inject GITHUB_TOKEN into a clone URL for private repos."""
    tok = get_secret("GITHUB_TOKEN")
    if not tok:
        print("⚠ GITHUB_TOKEN not set — clone will fail for a private repo.")
        print("  Add via: left sidebar → 🔑 Secrets → GITHUB_TOKEN")
        print("  Create a classic PAT (repo scope) at https://github.com/settings/tokens")
        return base_url
    # Both classic PATs (ghp_*) and fine-grained PATs (github_pat_*) work as
    # HTTP basic-auth passwords.  The x-access-token prefix is for GitHub App
    # installation tokens only — do NOT use it for personal access tokens.
    return base_url.replace("https://", f"https://{tok}@")


# ---------------------------------------------------------------------------
# Setup: install deps
# ---------------------------------------------------------------------------

def install_deps(gbv_dir: str = GBV_DIR) -> None:
    """pip-install requirements.txt + bitsandbytes + accelerate."""
    req = os.path.join(gbv_dir, "requirements.txt")
    # flash-attn is optional — may fail on older drivers
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q",
         "-r", req, "bitsandbytes", "accelerate", "flash-attn"],
        check=False,
    )
    # Core deps must succeed
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q",
         "-r", req, "bitsandbytes", "accelerate"],
        check=True,
    )
    print("[3/5] Dependencies installed")


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def auth_wandb() -> None:
    """Login to W&B from WANDB_API_KEY secret, or fall back to offline mode."""
    key = get_secret("WANDB_API_KEY")
    if key:
        os.environ["WANDB_API_KEY"] = key
        import wandb  # type: ignore
        wandb.login(key=key, relogin=True)
        print("[4/5] W&B authenticated")
    else:
        os.environ["WANDB_MODE"] = "offline"
        print("[4/5] W&B offline  (add WANDB_API_KEY via left sidebar → Secrets)")


def auth_hf() -> None:
    """Login to HuggingFace Hub from HF_TOKEN secret."""
    tok = get_secret("HF_TOKEN")
    if tok:
        os.environ["HF_TOKEN"] = tok
        os.environ["HUGGING_FACE_HUB_TOKEN"] = tok
        try:
            from huggingface_hub import login  # type: ignore
            login(token=tok, add_to_git_credential=False)
            print("     HuggingFace authenticated")
        except Exception:
            pass
    else:
        print("     HF_TOKEN not set (OK for public Qwen3 models)")


def keep_alive() -> None:
    """Inject a 45-second JS heartbeat to prevent Colab idle-timeout.

    Call this at the top of any long-running cell (Cell 7, Cell 8, etc.).
    The heartbeat fires a synthetic mousemove event and clicks the reconnect
    button if it appears, keeping the browser from triggering idle logout.

    Safe to call multiple times — the JS guard (window.__sd_ka) ensures only
    one interval is registered per page.
    """
    try:
        from IPython.display import display, Javascript  # type: ignore
        display(Javascript("""
(function(){
  if(window.__sd_ka)return;
  window.__sd_ka=setInterval(function(){
    document.dispatchEvent(new MouseEvent('mousemove',{bubbles:true}));
    var b=document.querySelector('[data-tooltip="Reconnect to runtime"]');
    if(b)b.click();
  },45000);
  console.log('[specdist] keep-alive on (45 s heartbeat)');
})();
"""))
    except Exception:
        pass  # not in a browser / no IPython — silently skip


def check_gpu(warn_below_gb: float = 12.0) -> None:
    """Print GPU name + free VRAM; warn if below warn_below_gb."""
    import torch  # type: ignore
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(0)
        gb = total / 1024 ** 3
        print(f"     GPU: {torch.cuda.get_device_name(0)}  "
              f"{free / 1024 ** 3:.1f} GB free / {gb:.1f} GB total")
        if gb < warn_below_gb:
            print(f"⚠ Only {gb:.1f} GB VRAM — colab config needs T4 (15 GB)")
    else:
        print("⚠ No GPU — Runtime → Change runtime type → T4 / A100 GPU")


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run_pipeline(
    config: str,
    drive_root: str = DRIVE_ROOT,
    gbv_dir: str = GBV_DIR,
    *,
    smoke: bool = False,
    losses: str = None,
    background: bool = False,
    extra_args: list = None,
):
    """
    Launch orchestration/experiment.py with the given config.

    Returns:
        subprocess.Popen if background=True, else subprocess.CompletedProcess.
    """
    os.chdir(gbv_dir)
    os.environ.update({
        "SPECDIST_STORAGE_ROOT": drive_root,
        "SPECDIST_DB_PATH":      os.path.join(drive_root, "results.db"),
        "SPECDIST_LOGS_ROOT":    os.path.join(drive_root, "logs"),
    })
    log_file = os.path.join(drive_root, "logs", "pipeline_output.log")

    cmd = [sys.executable, "orchestration/experiment.py",
           "--config", config, "--storage_root", drive_root, "--yes"]
    if smoke:
        cmd.append("--smoke")
    if losses:
        cmd += ["--losses", losses]
    if extra_args:
        cmd += extra_args

    mode = "SMOKE TEST" if smoke else f"FULL pipeline ({config})"
    print(f"{'[BG] ' if background else ''}{mode}")
    print(f"  Log → {log_file}")
    print(f"  DB  → {drive_root}/results.db")

    if background:
        log_fh = open(log_file, "a", buffering=1)
        proc = subprocess.Popen(cmd, cwd=gbv_dir,
                                stdout=log_fh, stderr=subprocess.STDOUT)

        def _tail():
            with open(log_file, "r", encoding="utf-8", errors="replace") as lf:
                lf.seek(0, 2)
                while proc.poll() is None:
                    line = lf.readline()
                    if line:
                        sys.stdout.write(line); sys.stdout.flush()
                    else:
                        time.sleep(0.4)
                for line in lf:
                    sys.stdout.write(line); sys.stdout.flush()

        threading.Thread(target=_tail, daemon=True).start()
        print(f"\n  PID {proc.pid} — call monitor() to track progress")
        print(f"  Stop: import os, signal; os.kill({proc.pid}, signal.SIGTERM)")
        return proc
    else:
        result = subprocess.run(cmd, cwd=gbv_dir)
        if result.returncode == 0:
            print("\n✓ Pipeline complete — call start_dashboard() to view results.")
        else:
            print(f"\n✗ Exit {result.returncode} — re-run to resume from last checkpoint.")
            print(f"  Full log: {log_file}")
        return result


# ---------------------------------------------------------------------------
# Profile inspector (reads YAML, prints key params before a run)
# ---------------------------------------------------------------------------

def show_profile(profile: str, gbv_dir: str = GBV_DIR) -> None:
    """Read a YAML profile and print key training params."""
    import yaml  # type: ignore
    yaml_path = os.path.join(
        gbv_dir, "orchestration", "configs",
        profile.replace("/", os.sep) + ".yaml",
    )
    try:
        with open(yaml_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        tr   = cfg.get("training", {})
        hw   = cfg.get("hardware", {})
        mdl  = cfg.get("models", {})
        tree = cfg.get("tree_training", {})
        ev   = cfg.get("evaluation", {})
        quant = "4-bit NF4" if hw.get("load_in_4bit") else "BF16"
        print(f"  Profile  : {profile}")
        print(f"  Teacher  : {mdl.get('target', '?')}  ({quant})")
        print(f"  Draft    : {mdl.get('draft', '?')}")
        print(f"  Steps    : {tr.get('steps', '?')}  "
              f"lr={tr.get('lr', '?')}  lora_r={tr.get('lora_r', '?')}")
        if tree:
            print(f"  Tree     : K={tree.get('tree_K', '?')}  "
                  f"L={tree.get('tree_L', '?')}")
        print(f"  Eval     : modes={ev.get('modes', '?')}  "
              f"n_prompts={ev.get('n_prompts', '?')}")
    except Exception as e:
        print(f"  Profile  : {profile}  (could not read YAML: {e})")


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

def start_dashboard(
    drive_root: str = DRIVE_ROOT,
    gbv_dir: str = GBV_DIR,
    port: int = 5000,
):
    """Start the Flask training dashboard and print its Colab proxy URL."""
    from google.colab.output import eval_js  # type: ignore
    db_path = os.path.join(drive_root, "results.db")
    if not os.path.exists(db_path):
        print(f"⚠ No results DB at {db_path}")
        print("  Run the pipeline first, then call start_dashboard().")
        return None
    os.environ["SPECDIST_DB_PATH"]   = db_path
    os.environ["SPECDIST_LOGS_ROOT"] = os.path.join(drive_root, "logs")

    proc_box = [None]

    def _run():
        proc_box[0] = subprocess.Popen(
            [sys.executable,
             os.path.join(gbv_dir, "dashboard", "training_dashboard.py"),
             "--host", "0.0.0.0", "--port", str(port)],
            env=os.environ.copy(),
        )
        proc_box[0].wait()

    threading.Thread(target=_run, daemon=True).start()
    time.sleep(3)
    url = eval_js(f"google.colab.kernel.proxyPort({port})")
    print(f"✓ Dashboard → {url}")
    print("  Auto-refreshes every 15 s. Stop: Runtime → Interrupt execution.")
    return proc_box[0]


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------

def monitor(
    drive_root: str = DRIVE_ROOT,
    config: str = "colab",
    log_tail: int = 60,
    auto_refresh: bool = False,
    refresh_secs: int = 20,
) -> None:
    """
    Print pipeline state + log tail.
    Set auto_refresh=True for a live updating view (interrupt cell to stop).
    """
    from IPython.display import clear_output  # type: ignore

    log_file   = os.path.join(drive_root, "logs", "pipeline_output.log")
    state_file = os.path.join(drive_root, f"pipeline_state_{config}.json")

    def _show():
        sep = "=" * 62
        print(sep); print("PIPELINE STATE"); print(sep)
        if os.path.exists(state_file):
            state  = json.load(open(state_file, encoding="utf-8"))
            steps  = state.get("steps", {})
            counts: dict = {}
            for sid, info in steps.items():
                s = info.get("status", "pending")
                counts[s] = counts.get(s, 0) + 1
                icon = {"done": "OK", "running": ">>",
                        "error": "!!", "pending": ".."}.get(s, "..")
                ts = (info.get("finished_at") or
                      info.get("started_at") or "")[:16]
                print(f"  [{icon}] {sid:45s}  {s:8s}  {ts}")
            print(f"\n  Done:{counts.get('done', 0)}  "
                  f"Running:{counts.get('running', 0)}  "
                  f"Pending:{counts.get('pending', 0)}  "
                  f"Error:{counts.get('error', 0)}")
        else:
            print(f"  State file not found: {state_file}")
            print("  Pipeline not started yet (or Drive not mounted).")

        print()
        print(sep); print(f"LOG (last {log_tail} lines)"); print(sep)
        if os.path.exists(log_file):
            lines = open(log_file, encoding="utf-8",
                         errors="replace").readlines()
            print("".join(lines[-log_tail:]))
            age = (datetime.datetime.now().timestamp() -
                   os.path.getmtime(log_file))
            status = ("ACTIVE" if age < 120
                      else f"STALE ({int(age // 60)} min ago)")
            print(f"[{len(lines)} lines | {status}]")
        else:
            print(f"  Not found: {log_file}")

        err_logs = sorted(
            glob.glob(os.path.join(drive_root, "logs", "step_*_error.log")))
        if err_logs:
            print()
            print(sep)
            print(f"ERRORS ({len(err_logs)} file(s))")
            print(sep)
            for ef in err_logs:
                txt = open(ef, encoding="utf-8",
                           errors="replace").read()
                print(f"\n--- {os.path.basename(ef)} ---")
                print(txt[-2000:] if len(txt) > 2000 else txt)

        if auto_refresh:
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            print(f"\n[{ts} | next in {refresh_secs}s | interrupt to stop]")

    if auto_refresh:
        print(f"Auto-refresh every {refresh_secs}s — interrupt cell to stop.")
        while True:
            clear_output(wait=True)
            _show()
            time.sleep(refresh_secs)
    else:
        _show()
