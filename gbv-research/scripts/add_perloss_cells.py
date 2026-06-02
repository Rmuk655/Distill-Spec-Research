"""Add per-loss and eval-only cells to kaggle.ipynb, colab_quickstart.ipynb, a100_quickstart.ipynb."""
import json, os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEPLOY = os.path.join(HERE, "deploy")

# ── New cells ─────────────────────────────────────────────────────────────────

def _train_one_cell(nb_id, drive_root, gbv_dir, config, platform_note=""):
    """Cell: train exactly one loss (replace LOSS each session)."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": f"{nb_id}-train-one",
        "metadata": {},
        "outputs": [],
        "source": [
            "# =============================================================================\n",
           f"# Cell 4 — Train ONE loss  ({platform_note})\n",
            "# Change LOSS below and re-run.  --skip_existing skips completed steps.\n",
            "# This is the standard per-session workflow on Kaggle:\n",
            "#   Session 1: LOSS = 'kl'        Session 2: LOSS = 'rev_kl' etc.\n",
            "# On persistent machines (A100): change LOSS each time you want the next loss.\n",
            "# =============================================================================\n",
            "import sys\n",
           f"sys.path.insert(0, '{gbv_dir}/deploy')\n",
            "from deploy_utils import run_pipeline\n",
            "\n",
           f"DRIVE_ROOT = '{drive_root}'\n",
           f"GBV_DIR    = '{gbv_dir}'\n",
            "\n",
            "# ── Edit this each session ───────────────────────────────────────────────────\n",
            "LOSS   = 'kl'      # kl | rev_kl | jsd | l1 | ebe | ebe_single\n",
            "#                   # kl_tree | bv_tree | gbv_tree | traversal_tree\n",
            "#                   # rev_kl_tree | jsd_tree | naive_tree | nss_tree\n",
            "#                   # specinfer_tree | spectr_tree | khisti_tree\n",
           f"CONFIG = '{config}'   # must match the CONFIG used in Cell 0\n",
            "SMOKE  = False         # True = 10-step crash check\n",
            "# ─────────────────────────────────────────────────────────────────────────────\n",
            "\n",
            "# Does: baseline eval (once, --skip_existing) → train LOSS → merge → eval GSM8K\n",
            "run_pipeline(CONFIG, DRIVE_ROOT, GBV_DIR, smoke=SMOKE, losses=LOSS, background=False)\n",
        ],
    }


def _eval_one_cell(nb_id, drive_root, gbv_dir, config):
    """Cell: eval one already-trained model without retraining."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": f"{nb_id}-eval-one",
        "metadata": {},
        "outputs": [],
        "source": [
            "# =============================================================================\n",
            "# Cell 5 — Eval ONE trained model  (no retraining)\n",
            "# Use when the checkpoint is already done but you want to re-run eval\n",
            "# (different K, different n_prompts, extra verifier modes, etc.)\n",
            "# Runs evaluate.py directly — bypasses the full pipeline orchestration.\n",
            "# =============================================================================\n",
            "import sys, os\n",
           f"sys.path.insert(0, '{gbv_dir}')\n",
           f"GBV_DIR    = '{gbv_dir}'\n",
           f"DRIVE_ROOT = '{drive_root}'\n",
            "\n",
            "# ── Edit these ───────────────────────────────────────────────────────────────\n",
            "LOSS    = 'kl'           # which trained model to evaluate\n",
           f"CONFIG  = '{config}'     # must match training config\n",
            "MODES   = 'alpha,bv,gbv,traversal,specinfer,naive'  # verifier modes\n",
            "K       = '3'            # draft paths\n",
            "TEMP    = '1.0'          # sampling temperature\n",
            "N       = '100'          # number of eval prompts\n",
            "DATASET = 'gsm8k'        # gsm8k | humaneval | math500 | mtbench | alpaca\n",
            "# ─────────────────────────────────────────────────────────────────────────────\n",
            "\n",
            "import subprocess\n",
            "ckpt = os.path.join(DRIVE_ROOT, 'checkpoints', f'{LOSS.replace(\"_\",\"_\")}-gsm8k_merged')\n",
            "if not os.path.isdir(ckpt):\n",
            "    ckpt = os.path.join(DRIVE_ROOT, 'checkpoints', f'{LOSS}-gsm8k_merged')\n",
            "    print(f'Trying: {ckpt}')\n",
            "if not os.path.isdir(ckpt):\n",
            "    raise FileNotFoundError(f'Merged checkpoint not found: {ckpt}\\n'\n",
            "                            'Run Cell 4 first to train + merge the model.')\n",
            "\n",
            "# Import the target model path from the running config\n",
            "import yaml\n",
            "cfg_path = os.path.join(GBV_DIR, 'orchestration', 'configs', f'{CONFIG}.yaml')\n",
            "target = yaml.safe_load(open(cfg_path))['models']['target']\n",
            "\n",
            "cmd = [\n",
            "    'python', os.path.join(GBV_DIR, 'orchestration', 'evaluate.py'),\n",
            "    '--student', ckpt,\n",
            "    '--teacher', target,\n",
            "    '--student_label', LOSS,\n",
            "    '--datasets', DATASET,\n",
            "    '--modes', MODES,\n",
            "    '--K', K, '--temperature', TEMP, '--n', N,\n",
            "    '--skip_fetch', '--skip_existing',\n",
            "    '--storage_root', DRIVE_ROOT,\n",
            "    '--loss_name', LOSS,\n",
            "]\n",
            "print('Running:', ' '.join(cmd[-10:]))\n",
            "result = subprocess.run(cmd, cwd=GBV_DIR)\n",
            "print('Done — refresh dashboard to see results.')\n",
        ],
    }


def _train_all_cell(nb_id, drive_root, gbv_dir, config, losses_list):
    """Cell: train all losses sequentially (one per kaggle session = impractical for all,
    but useful on persistent machines). Shows credit warning for Kaggle."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": f"{nb_id}-train-all",
        "metadata": {},
        "outputs": [],
        "source": [
            "# =============================================================================\n",
            "# Cell 6 — Train ALL losses sequentially\n",
            "# On Kaggle: realistically 1-2 losses fit in a session before T4 x2 credits run low.\n",
            "#             Use Cell 4 (one loss) per session instead.\n",
            "# On persistent machines (A100, ATS): this runs unattended overnight.\n",
            "# --skip_existing ensures completed losses are not re-run.\n",
            "# =============================================================================\n",
            "import sys, threading\n",
           f"sys.path.insert(0, '{gbv_dir}/deploy')\n",
            "from deploy_utils import run_pipeline\n",
            "\n",
           f"DRIVE_ROOT = '{drive_root}'\n",
           f"GBV_DIR    = '{gbv_dir}'\n",
            "\n",
            "# ── Edit if you want a subset ────────────────────────────────────────────────\n",
           f"CONFIG = '{config}'\n",
            "LOSSES = [\n",
        ] + [
            f"    '{l}',\n" for l in losses_list
        ] + [
            "]\n",
            "SMOKE = False\n",
            "# ─────────────────────────────────────────────────────────────────────────────\n",
            "\n",
            "print(f'Running {len(LOSSES)} losses sequentially. --skip_existing handles resume.')\n",
            "\n",
            "def _run_all():\n",
            "    for i, loss in enumerate(LOSSES):\n",
            "        print(f'\\n[{i+1}/{len(LOSSES)}] {loss}')\n",
            "        proc = run_pipeline(CONFIG, DRIVE_ROOT, GBV_DIR,\n",
            "                            smoke=SMOKE, losses=loss, background=True)\n",
            "        proc.wait()\n",
            "        if proc.returncode != 0:\n",
            "            print(f'  FAIL rc={proc.returncode} — re-run to resume.')\n",
            "            return\n",
            "    print('\\nAll losses done.')\n",
            "\n",
            "threading.Thread(target=_run_all, daemon=True).start()\n",
            "print('Running in background. Run Cell 2 (monitor) to watch progress.')\n",
        ],
    }


# ── Patch kaggle.ipynb ────────────────────────────────────────────────────────

path = os.path.join(DEPLOY, "kaggle.ipynb")
nb = json.load(open(path, encoding="utf-8"))

# Update title cell to list new cells
for c in nb["cells"]:
    if c.get("id") == "kaggle-title":
        src = "".join(c["source"] if isinstance(c["source"], list) else [c["source"]])
        old_table = "| Cell | What it does | When to run |"
        new_table = (
            "| Cell | What it does | When to run |\n"
            "|------|-------------|-------------|\n"
            "| 0. Bootstrap | Full pipeline (baseline + all losses) | Fresh session, no per-loss control |\n"
            "| 1. Resume | Re-attach after 9-hour limit, session timeout | After idle timeout |\n"
            "| 2b. Auto-backup | Checkpoint to Kaggle Dataset every N min | Optional alongside Cell 1 |\n"
            "| 2. Monitor | State + log tail | Any time |\n"
            "| 3. Save | Verify artifacts | Before session ends |\n"
            "| **4. Train ONE loss** | **Train one loss per session (Kaggle workflow)** | **Each session** |\n"
            "| 5. Eval ONE model | Re-run eval on a trained checkpoint | After Cell 4 completes |\n"
            "| 6. Train ALL losses | All losses sequentially (persistent machines only) | A100/ATS overnight |"
        )
        src = src.replace(
            "| Cell | What it does | When to run |",
            new_table,
            1
        )
        # Remove old rows if they existed
        c["source"] = [src]
    break

# Add new cells before the save cell
new_cells = [
    _train_one_cell("kaggle",
                    "/kaggle/working/specdist",
                    "/kaggle/working/Distill-Spec-Research/gbv-research",
                    "kaggle",
                    "Kaggle — change LOSS each session, ~1 loss per session"),
    _eval_one_cell("kaggle",
                   "/kaggle/working/specdist",
                   "/kaggle/working/Distill-Spec-Research/gbv-research",
                   "kaggle"),
    _train_all_cell("kaggle",
                    "/kaggle/working/specdist",
                    "/kaggle/working/Distill-Spec-Research/gbv-research",
                    "kaggle",
                    ["kl", "rev_kl", "jsd", "l1",
                     "kl_tree", "bv_tree", "gbv_tree", "traversal_tree",
                     "nss_tree", "specinfer_tree", "khisti_tree"]),
]

# Insert before the save cell (last cell)
save_idx = len(nb["cells"]) - 1
for i, cell in enumerate(new_cells):
    nb["cells"].insert(save_idx + i, cell)

json.dump(nb, open(path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(f"kaggle.ipynb: {len(nb['cells'])} cells (added 3)")

# ── Patch colab_quickstart.ipynb — add eval-only cell ──────────────────────

for fname, cfg, drive, gbv, nb_id in [
    ("colab_quickstart.ipynb", "colab",
     "/content/drive/MyDrive/specdist",
     "/content/Distill-Spec-Research/gbv-research",
     "colab"),
    ("a100_quickstart.ipynb", "a100",
     "/content/drive/MyDrive/specdist",
     "/content/Distill-Spec-Research/gbv-research",
     "a100"),
]:
    path = os.path.join(DEPLOY, fname)
    nb = json.load(open(path, encoding="utf-8"))

    # Check if eval-one cell already exists
    ids = [c.get("id", "") for c in nb["cells"]]
    if f"{nb_id}-eval-one" not in ids:
        # Add after the tree-all cell (last cell)
        nb["cells"].append(_eval_one_cell(nb_id, drive, gbv, cfg))
        print(f"{fname}: added eval-one cell ({len(nb['cells'])} total)")

    # Also add generic train-one cell if not already there (using --losses flag)
    if f"{nb_id}-train-one" not in ids:
        # Insert before the last cell
        nb["cells"].insert(-1, _train_one_cell(nb_id, drive, gbv, cfg,
            "Colab — one loss at a time, --skip_existing handles resume"))
        print(f"{fname}: added train-one cell")

    json.dump(nb, open(path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

print("Done.")
