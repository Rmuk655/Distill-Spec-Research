"""
eval_all.py — unified evaluation runner.

Measures alpha (token acceptance rate) and block efficiency across:
  - draft models  : baseline | kl200 | ebe200 | custom path
  - datasets      : diverse50 | gsm8k | humaneval | math500 | mtbench | alpaca
  - verifier modes: alpha | specinfer | gbv | traversal | bv
  - K values      : any list of integers
  - temperatures  : any list of floats

All results are saved to results.db (SQLite) via results_db.py.

Usage examples:
  # Quick alpha check, one draft, one dataset
  python eval_all.py --drafts baseline --datasets diverse50 --modes alpha --K 1 --n 30

  # Full K-sweep, specinfer + gbv, baseline vs EBE
  python eval_all.py --drafts baseline,ebe200 --datasets diverse50 \\
                     --modes specinfer,gbv --K 1,3,5

  # Run everything (all drafts x all datasets x all modes x K=1,3,5)
  python eval_all.py --run_all --n 30
"""

import sys, os, json, time, argparse, subprocess, re
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "distill"))

import results_db
from fetch_datasets import get_dataset_path, fetch_all

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

TARGET_MODEL = "Qwen/Qwen3-0.6B"

DRAFT_REGISTRY = {
    "baseline": {
        "path": "Qwen/Qwen2.5-0.5B",
        "loss_name": "baseline",
        "train_steps": 0,
        "learning_rate": 0.0,
        "lora_rank": 0,
    },
    "kl200": {
        "path": os.path.join(_HERE, "checkpoints", "kl200_merged"),
        "loss_name": "kl",
        "train_steps": 200,
        "learning_rate": 3e-5,
        "lora_rank": 8,
    },
    "ebe200": {
        "path": os.path.join(_HERE, "checkpoints", "ebe200_merged"),
        "loss_name": "ebe",
        "train_steps": 200,
        "learning_rate": 3e-5,
        "lora_rank": 8,
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_prompts(dataset: str, n: int, data_dir=None) -> list:
    path = get_dataset_path(dataset, n)
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {dataset}. Run fetch_datasets.py first.")
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    if n:
        items = items[:n]
    return items


def _resolve_draft(label: str) -> dict:
    """Return registry entry for a label, or treat label as a raw path."""
    if label in DRAFT_REGISTRY:
        return dict(DRAFT_REGISTRY[label])
    # custom path — caller provides label as draft_label
    return {
        "path": label,
        "loss_name": "custom",
        "train_steps": 0,
        "learning_rate": 0.0,
        "lora_rank": 0,
    }


# ---------------------------------------------------------------------------
# Alpha evaluation (inline, no subprocess)
# ---------------------------------------------------------------------------

def _run_alpha(draft_label: str, draft_path: str, target_path: str,
               prompts: list, temperature: float,
               max_propose: int = 5, max_tokens: int = 30) -> dict:
    """
    Run speculative decoding alpha measurement.
    Returns dict with alpha_mean, alpha_std, alpha_ci95, throughput, ms_per_tok,
    peak_vram_mb, per_prompt list.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from specInfer.generator import Generator

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(target_path, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    draft_model = AutoModelForCausalLM.from_pretrained(
        draft_path, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device).eval()

    same = (draft_path == target_path)
    target_model = draft_model if same else AutoModelForCausalLM.from_pretrained(
        target_path, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device).eval()

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    generator = Generator(
        small_model=draft_model, large_model=target_model,
        tokenizer=tokenizer, max_propose_num=max_propose,
        is_encoder_decoder=False, use_cache=True,
    )

    alphas, per_prompt_rows = [], []
    total_tokens, total_time = 0, 0.0

    for i, item in enumerate(prompts):
        prompt = item["prompt"] if isinstance(item, dict) else item
        category = item.get("category") if isinstance(item, dict) else None
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.inference_mode():
            output = generator.generate(
                input_ids=input_ids,
                max_tokens=max_tokens,
                temperature=temperature,
                attention_mask=torch.ones_like(input_ids),
            )

        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        gen_tokens = (output.output[0].shape[-1] - input_ids.shape[-1]
                      if hasattr(output.output[0], "shape") else max_tokens)
        total_tokens += gen_tokens
        total_time += elapsed

        alpha = float(output.alpha_sum) / output.sample_steps if output.sample_steps > 0 else 0.0
        alphas.append(alpha)
        per_prompt_rows.append({
            "prompt_idx": i,
            "category": category,
            "alpha": alpha,
            "block_eff": None,
            "gen_tokens": gen_tokens,
            "wall_ms": elapsed * 1000,
        })

    alphas = np.array(alphas)
    n = len(alphas)
    peak_vram = torch.cuda.max_memory_allocated() / 1024**2 if device == "cuda" else 0

    # unload
    del draft_model
    if not same:
        del target_model
    if device == "cuda":
        torch.cuda.empty_cache()

    return {
        "alpha_mean": float(alphas.mean()),
        "alpha_std": float(alphas.std()),
        "alpha_ci95": float(1.96 * alphas.std() / np.sqrt(n)),
        "throughput": total_tokens / total_time if total_time > 0 else 0,
        "ms_per_tok": total_time / total_tokens * 1000 if total_tokens > 0 else 0,
        "peak_vram_mb": peak_vram,
        "per_prompt": per_prompt_rows,
    }


# ---------------------------------------------------------------------------
# Block efficiency evaluation (via GBV subprocess)
# ---------------------------------------------------------------------------

def _run_be_subprocess(draft_path: str, target_path: str,
                       mode: str, K: int, L: int,
                       data_path: str, max_new_tokens: int,
                       gbv_dir: str, python_exe: str) -> dict | None:
    """
    Call GBV main.py as subprocess and parse block efficiency from stdout.
    Returns dict or None on failure.
    """
    cmd = [
        python_exe,
        os.path.join(gbv_dir, "main.py"),
        "--p_model", target_path,
        "--q_model", draft_path,
        "--mode", mode,
        "--K", str(K),
        "--L", str(L),
        "--max_new_tokens", str(max_new_tokens),
        "--data", data_path,
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=1200,
            cwd=_PARENT,
        )
        out = result.stdout + result.stderr
        be_match = re.search(r"Block efficiency[^:]*:\s*([\d.]+)", out)
        if not be_match:
            print(f"    [WARN] No block efficiency in output:\n{out[:500]}")
            return None
        be = float(be_match.group(1))
        return {"block_eff": be, "raw_output": out}
    except subprocess.TimeoutExpired:
        print(f"    [WARN] Subprocess timed out for mode={mode} K={K}")
        return None
    except Exception as e:
        print(f"    [WARN] Subprocess error: {e}")
        return None


# ---------------------------------------------------------------------------
# Master evaluation function
# ---------------------------------------------------------------------------

def run_experiment(
    draft_label: str,
    dataset: str,
    mode: str,
    K: int,
    L: int = 5,
    temperature: float = 1.0,
    n_prompts: int = 30,
    max_tokens: int = 60,
    target_path: str = TARGET_MODEL,
    gbv_dir: str = None,
    python_exe: str = sys.executable,
    dry_run: bool = False,
    no_db: bool = False,
) -> dict:
    """Run one experiment cell and save to DB. Returns the run row dict."""

    info = _resolve_draft(draft_label)
    draft_path = info["path"]

    if gbv_dir is None:
        gbv_dir = os.path.join(_PARENT, "GBV")

    print(f"\n{'='*60}")
    print(f"  draft={draft_label}  dataset={dataset}  mode={mode}  K={K}  T={temperature}")
    print(f"{'='*60}")

    if dry_run:
        print("  [DRY RUN] skipping inference")
        return {}

    prompts = load_prompts(dataset, n_prompts)
    data_path = get_dataset_path(dataset, n_prompts)

    row = {
        "draft_label":   draft_label,
        "draft_path":    draft_path,
        "target_path":   target_path,
        "loss_name":     info["loss_name"],
        "train_steps":   info["train_steps"],
        "learning_rate": info["learning_rate"],
        "lora_rank":     info["lora_rank"],
        "dataset":       dataset,
        "n_prompts":     len(prompts),
        "mode":          mode,
        "K":             K,
        "L":             L,
        "temperature":   temperature,
    }

    if mode == "alpha":
        res = _run_alpha(draft_label, draft_path, target_path,
                         prompts, temperature, max_tokens=max_tokens)
        row.update({
            "alpha_mean":   res["alpha_mean"],
            "alpha_std":    res["alpha_std"],
            "alpha_ci95":   res["alpha_ci95"],
            "throughput":   res["throughput"],
            "ms_per_tok":   res["ms_per_tok"],
            "peak_vram_mb": res["peak_vram_mb"],
        })
        per_prompt_rows = res["per_prompt"]
        print(f"  alpha = {res['alpha_mean']:.4f} +/- {res['alpha_ci95']:.4f}  "
              f"throughput = {res['throughput']:.2f} tok/s")

    else:
        # GBV subprocess
        res = _run_be_subprocess(
            draft_path, target_path, mode, K, L,
            data_path, max_tokens, gbv_dir, python_exe,
        )
        if res is None:
            print("  [FAILED] no block efficiency result")
            return {}
        row["block_eff"] = res["block_eff"]
        per_prompt_rows = []
        print(f"  block_eff = {res['block_eff']:.4f}")

    if not no_db:
        run_id = results_db.insert_run(row)
        if per_prompt_rows:
            for r in per_prompt_rows:
                r["run_id"] = run_id
            results_db.insert_per_prompt_batch(per_prompt_rows)
        row["id"] = run_id
        print(f"  Saved to DB as run_id={run_id}")

    return row


# ---------------------------------------------------------------------------
# CLI / batch runner
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--drafts", default="baseline",
                   help="Comma-separated draft labels, e.g. baseline,kl200,ebe200")
    p.add_argument("--datasets", default="diverse50",
                   help="Comma-separated dataset names")
    p.add_argument("--modes", default="alpha",
                   help="Comma-separated modes: alpha,specinfer,gbv,traversal,bv")
    p.add_argument("--K", default="1", help="Comma-separated K values, e.g. 1,3,5")
    p.add_argument("--L", type=int, default=5)
    p.add_argument("--temperature", default="1.0",
                   help="Comma-separated temperatures, e.g. 0.6,1.0")
    p.add_argument("--n", type=int, default=30, help="Prompts per dataset")
    p.add_argument("--max_tokens", type=int, default=60)
    p.add_argument("--target", default=TARGET_MODEL)
    p.add_argument("--gbv_dir", default=None)
    p.add_argument("--fetch_datasets", action="store_true",
                   help="Download all datasets before running")
    p.add_argument("--run_all", action="store_true",
                   help="Run full matrix: all drafts x all datasets x all modes x K=1,3,5")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--no_db", action="store_true")
    args = p.parse_args()

    if args.fetch_datasets or args.run_all:
        fetch_all(n=args.n)

    if args.run_all:
        drafts = list(DRAFT_REGISTRY.keys())
        datasets = ["diverse50", "gsm8k", "humaneval", "math500", "mtbench", "alpaca"]
        modes = ["alpha", "specinfer", "gbv", "traversal"]
        Ks = [1, 3, 5]
        temps = [1.0]
    else:
        drafts = args.drafts.split(",")
        datasets = args.datasets.split(",")
        modes = args.modes.split(",")
        Ks = [int(k) for k in args.K.split(",")]
        temps = [float(t) for t in args.temperature.split(",")]

    python_exe = sys.executable

    for draft in drafts:
        for dataset in datasets:
            for mode in modes:
                for K in Ks:
                    for temp in temps:
                        # alpha mode doesn't use K for GBV but we still vary it
                        # for GBV modes K=1 is equivalent to standard specinfer
                        run_experiment(
                            draft_label=draft,
                            dataset=dataset,
                            mode=mode,
                            K=K,
                            L=args.L,
                            temperature=temp,
                            n_prompts=args.n,
                            max_tokens=args.max_tokens,
                            target_path=args.target,
                            gbv_dir=args.gbv_dir,
                            python_exe=python_exe,
                            dry_run=args.dry_run,
                            no_db=args.no_db,
                        )

    print("\n\nAll experiments complete.")


if __name__ == "__main__":
    main()
