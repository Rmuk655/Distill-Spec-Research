"""
setup_env.py — one-file environment setup for SpecDist.

Works on both Windows (laptop) and Linux (lab server / A100).
Creates a virtual environment, installs all dependencies,
initialises the SQLite DB, and verifies the setup.

Usage:
    python setup_env.py                   # full setup
    python setup_env.py --no_venv         # skip venv creation (already inside one)
    python setup_env.py --cuda 12.1       # specify CUDA version (default: auto-detect)
    python setup_env.py --check           # only verify existing install

Requirements: Python 3.10+
"""

import sys, os, subprocess, platform, argparse, shutil

MIN_PYTHON = (3, 10)
VENV_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "venv")

# ---------------------------------------------------------------------------
# CUDA version -> PyTorch wheel index
# ---------------------------------------------------------------------------
TORCH_CUDA_WHEELS = {
    "12.4": "https://download.pytorch.org/whl/cu124",
    "12.1": "https://download.pytorch.org/whl/cu121",
    "11.8": "https://download.pytorch.org/whl/cu118",
    "cpu":  "https://download.pytorch.org/whl/cpu",
}


def _run(cmd: list, check=True, **kwargs) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, check=check, **kwargs)


def detect_cuda_version() -> str:
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=driver_version",
                                 "--format=csv,noheader"],
                                capture_output=True, text=True)
        if result.returncode == 0:
            # Try nvcc for CUDA version
            nvcc = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
            if nvcc.returncode == 0:
                for line in nvcc.stdout.splitlines():
                    if "release" in line.lower():
                        # e.g. "release 12.1, V12.1.66"
                        import re
                        m = re.search(r"release (\d+\.\d+)", line)
                        if m:
                            ver = m.group(1)
                            # normalise to supported key
                            major_minor = ver.rsplit(".", 1)[0] + "." + ver.split(".")[-1]
                            for key in TORCH_CUDA_WHEELS:
                                if major_minor.startswith(key.split(".")[0]):
                                    # pick closest supported version
                                    pass
                            # Return closest supported
                            if ver >= "12.4":
                                return "12.4"
                            elif ver >= "12.1":
                                return "12.1"
                            elif ver >= "11.8":
                                return "11.8"
            return "12.1"  # safe default if nvidia-smi found but nvcc not
    except FileNotFoundError:
        pass
    return "cpu"


def create_venv(venv_dir: str):
    if os.path.exists(venv_dir):
        print(f"  venv already exists at {venv_dir}, skipping creation")
        return
    print(f"\nCreating virtual environment at {venv_dir} ...")
    _run([sys.executable, "-m", "venv", venv_dir])


def get_pip(venv_dir: str) -> str:
    if platform.system() == "Windows":
        return os.path.join(venv_dir, "Scripts", "pip.exe")
    return os.path.join(venv_dir, "bin", "pip")


def get_python(venv_dir: str) -> str:
    if platform.system() == "Windows":
        return os.path.join(venv_dir, "Scripts", "python.exe")
    return os.path.join(venv_dir, "bin", "python")


def install_torch(pip: str, cuda_ver: str):
    wheel_url = TORCH_CUDA_WHEELS.get(cuda_ver, TORCH_CUDA_WHEELS["cpu"])
    print(f"\nInstalling PyTorch (CUDA {cuda_ver}) from {wheel_url} ...")
    _run([pip, "install",
          "torch", "torchvision", "torchaudio",
          "--index-url", wheel_url])


def install_requirements(pip: str, req_path: str):
    print("\nInstalling project requirements ...")
    _run([pip, "install", "-r", req_path])


def init_db(python: str):
    print("\nInitialising SQLite database ...")
    _gbv = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # gbv-research/
    script = os.path.join(_gbv, "db", "results_db.py")
    _run([python, script])


def patch_osd_inits():
    """Create missing __init__.py files in OSD submodule package directories.

    The upstream OSD repo omits ``__init__.py`` from several directories
    (e.g. distill/specInfer/).  A bare ``git submodule update --init`` leaves
    those directories without the marker files, causing ``ModuleNotFoundError``
    at runtime.  This function walks the OSD tree and creates the missing
    files.  It is idempotent — safe to call on every setup.
    """
    print("\nPatching OSD submodule __init__.py files ...")
    repo_dir = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    )
    osd_dir = os.path.join(repo_dir, "OSD")
    if not os.path.isdir(osd_dir):
        print("  [SKIP] OSD/ not found — run: git submodule update --init")
        return

    patched = []
    for dirpath, dirnames, filenames in os.walk(osd_dir):
        dirnames[:] = [
            d for d in dirnames
            if not d.startswith(".") and not d.startswith("__")
        ]
        if "__init__.py" in filenames:
            continue
        if any(f.endswith(".py") for f in filenames):
            init_path = os.path.join(dirpath, "__init__.py")
            with open(init_path, "w") as fh:
                fh.write("")
            patched.append(os.path.relpath(init_path, repo_dir))

    if patched:
        print(f"  Created {len(patched)} missing __init__.py file(s):")
        for p in patched:
            print(f"    + {p}")
    else:
        print("  All OSD package directories already have __init__.py.")


def fetch_datasets(python: str):
    print("\nFetching evaluation datasets (skip if already present) ...")
    _osd = os.path.normpath(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "OSD"))
    script = os.path.join(_osd, "fetch_datasets.py")
    _run([python, script, "--n", "30"])


def verify(python: str):
    print("\nVerifying installation ...")
    checks = [
        ("torch",          "import torch; print('torch', torch.__version__, '| CUDA:', torch.cuda.is_available())"),
        ("transformers",   "import transformers; print('transformers', transformers.__version__)"),
        ("peft",           "import peft; print('peft', peft.__version__)"),
        ("datasets",       "import datasets; print('datasets', datasets.__version__)"),
        ("flask",          "import flask; print('flask', flask.__version__)"),
        ("numpy",          "import numpy; print('numpy', numpy.__version__)"),
        ("results_db",     "import sys; sys.path.insert(0,'OSD'); import results_db; results_db._connect().close(); print('results_db OK')"),
        ("specInfer",      "import sys; sys.path.insert(0,'OSD/distill'); from specInfer.generator import Generator; print('specInfer OK')"),
    ]
    all_ok = True
    for name, code in checks:
        result = subprocess.run([python, "-c", code],
                                capture_output=True, text=True,
                                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if result.returncode == 0:
            print(f"  [OK] {result.stdout.strip()}")
        else:
            print(f"  [FAIL] {name}: {result.stderr.strip()[:120]}")
            all_ok = False
    return all_ok


def print_next_steps(python: str, venv_dir: str):
    is_win = platform.system() == "Windows"
    activate = (os.path.join(venv_dir, "Scripts", "activate")
                if is_win else f"source {os.path.join(venv_dir, 'bin', 'activate')}")
    print(f"""
{'='*60}
Setup complete! Next steps:

  1. Activate the environment:
       {activate}

  2. Run full evaluation (laptop — 0.6B models):
       python OSD/run_all.py \\
           --student Qwen/Qwen2.5-0.5B \\
           --teacher Qwen/Qwen3-0.6B

  3. Run full evaluation (server — 8B target):
       python OSD/run_all.py \\
           --student Qwen/Qwen2.5-0.5B \\
           --teacher Qwen/Qwen3-8B \\
           --n 50

  4. Launch the results dashboard:
       python OSD/viz_server.py
       # -> http://localhost:5000/

  5. Quick smoke test (2 min):
       python OSD/run_all.py \\
           --student Qwen/Qwen2.5-0.5B \\
           --teacher Qwen/Qwen3-0.6B \\
           --datasets diverse50 --modes alpha --K 1 --n 5
{'='*60}
""")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--no_venv",        action="store_true", help="Skip venv creation")
    p.add_argument("--no_torch",       action="store_true", help="Skip PyTorch install")
    p.add_argument("--no_datasets",    action="store_true", help="Skip dataset download")
    p.add_argument("--cuda",           default=None,        help="Override CUDA version, e.g. 12.1 | 11.8 | cpu")
    p.add_argument("--check",          action="store_true", help="Only verify, don't install")
    args = p.parse_args()

    # Python version check
    if sys.version_info < MIN_PYTHON:
        print(f"ERROR: Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ required, "
              f"found {sys.version_info.major}.{sys.version_info.minor}")
        sys.exit(1)

    _here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # gbv-research/
    req_path = os.path.join(_here, "requirements.txt")
    venv_dir = os.path.abspath(VENV_DIR)

    if args.check:
        python = get_python(venv_dir) if os.path.exists(venv_dir) else sys.executable
        ok = verify(python)
        sys.exit(0 if ok else 1)

    print(f"\nSpecDist Environment Setup")
    print(f"  Platform : {platform.system()} {platform.machine()}")
    print(f"  Python   : {sys.version.split()[0]}")

    # Detect CUDA
    cuda_ver = args.cuda or detect_cuda_version()
    print(f"  CUDA     : {cuda_ver}")

    # Create venv
    if not args.no_venv:
        create_venv(venv_dir)

    pip    = get_pip(venv_dir) if not args.no_venv else shutil.which("pip") or "pip"
    python = get_python(venv_dir) if not args.no_venv else sys.executable

    # Upgrade pip
    _run([python, "-m", "pip", "install", "--upgrade", "pip"])

    # Install PyTorch
    if not args.no_torch:
        install_torch(pip, cuda_ver)

    # Install requirements
    if os.path.exists(req_path):
        install_requirements(pip, req_path)
    else:
        print(f"\n[WARN] requirements.txt not found at {req_path}")

    # Patch OSD submodule: create missing __init__.py files so that
    # `from specInfer.generator import ...` and similar imports work for
    # every user after a plain `git submodule update --init`.
    patch_osd_inits()

    # Initialise DB
    init_db(python)

    # Fetch datasets
    if not args.no_datasets:
        fetch_datasets(python)

    # Verify
    ok = verify(python)

    if ok:
        print_next_steps(python, venv_dir)
    else:
        print("\n[WARN] Some checks failed — review output above")
        sys.exit(1)


if __name__ == "__main__":
    main()
