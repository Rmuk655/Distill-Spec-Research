"""
setup_download.py — ONE-TIME download of all models and datasets.

Run this ONCE on a machine with internet access, then everything
runs fully offline. Works on Google Colab, remote GPU servers, or laptop.

Usage:
    python setup_download.py                      # default: laptop config
    python setup_download.py --config server      # Qwen3-0.6B draft, Qwen3-8B target
    python setup_download.py --config colab       # same as server
    python setup_download.py --models_only        # skip dataset download
    python setup_download.py --datasets_only      # skip model download

After this script completes, set TRANSFORMERS_OFFLINE=1 and HF_HUB_OFFLINE=1
(already set in .env and in every script) — no internet needed for any run.
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

CONFIGS = {
    "laptop": {
        "draft":  "Qwen/Qwen2.5-0.5B",
        "target": "Qwen/Qwen3-0.6B",
        "desc":   "Laptop (4–8 GB VRAM) — Qwen2.5-0.5B draft → Qwen3-0.6B target",
    },
    "server": {
        "draft":  "Qwen/Qwen3-0.6B",
        "target": "Qwen/Qwen3-8B",
        "desc":   "Server/Colab A100 (40+ GB VRAM) — Qwen3-0.6B draft → Qwen3-8B target",
    },
    "colab": {
        "draft":  "Qwen/Qwen3-0.6B",
        "target": "Qwen/Qwen3-8B",
        "desc":   "Google Colab A100 — Qwen3-0.6B draft → Qwen3-8B target",
    },
}


def download_models(draft_id, target_id):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    for model_id, role in [(target_id, "target (frozen)"), (draft_id, "draft (trainable)")]:
        print(f"\n[{role}] Downloading {model_id} ...")
        # Download tokenizer
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        print(f"  tokenizer: {len(tok)} tokens")
        # Download model weights (don't load into GPU — just cache)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="cpu",           # stays on CPU — just caches weights
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        n_params = sum(p.numel() for p in model.parameters()) / 1e9
        print(f"  params: {n_params:.2f}B  -> cached to ~/.cache/huggingface/hub/")
        del model
        import gc; gc.collect()

    print("\n[OK] Models cached. Future runs need no internet.")


def download_datasets(n_train=None):
    sys.path.insert(0, HERE)
    from fetch_datasets import (
        fetch_gsm8k, fetch_humaneval, fetch_math500, fetch_mtbench, fetch_alpaca,
        fetch_gsm8k_train, fetch_math_train, fetch_alpaca_train,
    )

    print("\n[datasets] Downloading eval datasets ...")
    fetch_gsm8k(n=30, force=False)
    print("  gsm8k_30.jsonl OK")
    fetch_humaneval(force=False)
    print("  humaneval.jsonl OK")
    fetch_math500(n=30, force=False)
    print("  math500_30.jsonl OK")
    fetch_mtbench(force=False)
    print("  mtbench_80.jsonl OK")
    fetch_alpaca(n=30, force=False)
    print("  alpaca_30.jsonl OK")

    print("\n[datasets] Downloading training datasets ...")
    fetch_gsm8k_train(n=n_train, force=False)
    print("  gsm8k_train.jsonl OK")
    fetch_alpaca_train(n=n_train, force=False)
    print("  alpaca_train.jsonl OK")

    print("\n[OK] All datasets saved to OSD/data/")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="laptop",
                   choices=list(CONFIGS.keys()),
                   help="Which model pair to download (default: laptop)")
    p.add_argument("--draft",  default=None,
                   help="Override draft model HF ID")
    p.add_argument("--target", default=None,
                   help="Override target model HF ID")
    p.add_argument("--models_only",   action="store_true")
    p.add_argument("--datasets_only", action="store_true")
    p.add_argument("--n_train", type=int, default=None,
                   help="Limit training dataset size (default: all)")
    args = p.parse_args()

    cfg = CONFIGS[args.config]
    draft_id  = args.draft  or cfg["draft"]
    target_id = args.target or cfg["target"]

    print("=" * 60)
    print(f"  Setup: {cfg['desc']}")
    print(f"  Draft : {draft_id}")
    print(f"  Target: {target_id}")
    print("=" * 60)

    if not args.datasets_only:
        download_models(draft_id, target_id)

    if not args.models_only:
        download_datasets(args.n_train)

    print("\n" + "=" * 60)
    print("  Download complete. To run the pipeline:")
    print(f"    python experiment.py --config {args.config} --yes")
    print("=" * 60)


if __name__ == "__main__":
    main()
