"""Fix colab_quickstart.ipynb: correct the '4B = production' mislabel and add the
research-valid 8B-NF4 option (kaggle config, which is T4-generic, not Kaggle-only).
Run from gbv-research/:  python scripts/fix_colab_notebook.py
"""
import json, os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path = os.path.join(HERE, "deploy", "colab_quickstart.ipynb")
nb = json.load(open(path, encoding="utf-8"))

def _replace_in_cell(cell_id, replacements):
    for c in nb["cells"]:
        if c.get("id") == cell_id:
            src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
            for old, new in replacements:
                if old not in src:
                    print(f"  WARN: pattern not found in {cell_id}: {old[:50]!r}")
                src = src.replace(old, new)
            c["source"] = [src]
            return
    print(f"  WARN: cell {cell_id} not found")

# ── 1. Title cell — fix the config table + naming note ───────────────────────
_replace_in_cell("title-cell", [
    (
        "| CONFIG | Teacher | Steps | Time | Purpose |\n"
        "|--------|---------|-------|------|---------|\n"
        "| `colab_lite` | 1.7B (BF16) | 300 | ~25 min | Quick trend check — does the loss help? |\n"
        "| `colab` | 4B (BF16) | 500 | ~2-4 h | Production T4 results |",
        # ── replacement ──
        "| CONFIG | Teacher | Steps | Time | Research-valid? |\n"
        "|--------|---------|-------|------|-----------------|\n"
        "| `kaggle` | **8B (4-bit NF4)** | 1000 | ~5-8 h | **YES** — same 8B teacher as the A100 paper run; loss rankings transfer |\n"
        "| `colab` | 4B (BF16) | 500 | ~2-4 h | No — 4B ≠ paper 8B; dev / quota-fallback only |\n"
        "| `colab_lite` | 1.7B (BF16) | 300 | ~25 min | No — crash-check only |\n"
        "\n"
        "> **Config names describe the HARDWARE+teacher, not the platform.**\n"
        "> `kaggle` = \"T4 + 8B-NF4\" and runs on **any** T4 — Colab free, Kaggle, or Modal.\n"
        "> It is the research-valid setting on a free T4. Use it here on Colab.\n"
        "> Do **not** run the 8B teacher with `colab` (4B/BF16) — 8B in BF16 (~16 GB)\n"
        "> OOMs a 15 GB T4 and silently falls back to CPU. A startup guard now blocks this.",
    ),
    (
        "For A100 (8B teacher, 2000 steps) use `a100_quickstart.ipynb`.",
        "For A100 (8B teacher BF16, 2000 steps, full GSM8K) use `a100_quickstart.ipynb`.\n"
        "For the research-valid 8B run on a free T4, set `CONFIG = \"kaggle\"` below.",
    ),
])

# Also fix the cell-list line that says colab=4B "Production T4 results"
_replace_in_cell("title-cell", [
    ("| 7. Tree loss (single) | Train one tree loss | ~20-40 min |",
     "| 7. Tree loss (single) | Train one tree loss | ~20-40 min |"),  # no-op anchor (keep)
])

# ── 2. Bootstrap cell — fix CONFIG comment block + inline options ────────────
_replace_in_cell("cell-bootstrap", [
    (
        "# CONFIG options:\n"
        "#   colab_lite  — 1.7B teacher, 300 steps  (~25 min)  quick trend check\n"
        "#   colab       — 4B  teacher, 500 steps  (~2-4 h)   production T4 results",
        "# CONFIG options (all run on a free Colab T4):\n"
        "#   kaggle      — 8B teacher 4-bit NF4, 1000 steps  — RESEARCH-VALID (same 8B\n"
        "#                 as the A100 paper run; rankings transfer). Name is T4-generic,\n"
        "#                 NOT Kaggle-only — it means 'T4 + 8B-NF4' and works here on Colab.\n"
        "#   colab       — 4B teacher BF16, 500 steps  — dev/fallback only (4B ≠ paper 8B)\n"
        "#   colab_lite  — 1.7B teacher BF16, 300 steps — crash-check only",
    ),
    (
        'CONFIG      = "colab"        # colab_lite | colab',
        'CONFIG      = "kaggle"       # kaggle (8B NF4, research-valid) | colab (4B) | colab_lite (1.7B)',
    ),
])

# ── 3. Run cell — fix inline CONFIG comment ──────────────────────────────────
_replace_in_cell("cell-run", [
    (
        'CONFIG      = "colab"    # colab_lite | colab',
        'CONFIG      = "kaggle"   # kaggle (8B NF4, research-valid) | colab (4B) | colab_lite (1.7B)',
    ),
])

json.dump(nb, open(path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(f"Patched {path} ({len(nb['cells'])} cells)")
