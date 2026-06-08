"""Fix aip.ipynb: AIP is CPU-only, not A100."""
import json
from pathlib import Path

path = Path(__file__).resolve().parent.parent / "deploy" / "aip.ipynb"
nb = json.load(open(path, encoding="utf-8"))


def fix_src(src):
    # Fix CONFIG default
    src = src.replace(
        'CONFIG       = "a100"   # ← change this if your GPU has < 30 GB VRAM',
        'CONFIG       = "server_gpt2"   # AIP is CPU-only; GPT-2 convergence analysis only',
    )
    src = src.replace(
        'CONFIG       = "a100"   # must match Cell 0',
        'CONFIG       = "server_gpt2"   # must match Cell 0',
    )
    src = src.replace(
        'CONFIG       = "a100"   # must match Cell 0/1',
        'CONFIG       = "server_gpt2"   # must match Cell 0/1',
    )
    # Fix FAMILY in Cell 4
    src = src.replace(
        'FAMILY = "gpt2"    # ← change to "qwen" to run Qwen losses across GPUs',
        'FAMILY = "gpt2"   # AIP is CPU-only; only GPT-2 runs here',
    )
    return src


for cell in nb["cells"]:
    if isinstance(cell.get("source"), list):
        cell["source"] = [fix_src(s) for s in cell["source"]]
    elif isinstance(cell.get("source"), str):
        cell["source"] = fix_src(cell["source"])

# Update title cell
for cell in nb["cells"]:
    if cell.get("id") == "aip-title":
        src = "".join(cell["source"] if isinstance(cell["source"], list) else [cell["source"]])
        src = src.replace(
            "local storage persistent storage · 4-hour guaranteed session · up to 3 GPUs/user",
            "local storage persistent storage · 4-hour guaranteed session · 128 GB RAM · 2-socket CPU",
        )
        cell["source"] = [src]

json.dump(nb, open(path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("Done")
