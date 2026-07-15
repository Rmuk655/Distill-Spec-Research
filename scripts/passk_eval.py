#!/usr/bin/env python
"""
pass@k accuracy eval via vLLM — a SEPARATE inference/grading pipeline.

WHY THIS IS NEW: the rest of the repo measures *block efficiency* (spec-decoding
acceptance), and never grades answer correctness — `data_io/download.py` keeps a
gold `answer` field but the pipeline "only uses question as the eval prompt". This
script adds the accuracy side Rahul asked for: pass@k curves (k=1,2,4,8,16,32,64)
as a training-stability diagnostic (pass@64 stable => training didn't collapse the
model's coverage), measured (1) before training, (2) after each jsd run, (3) after
each prefix_overlap-prob run — on a FIXED prompt set each time.

USAGE (run on the cluster with a GPU; vLLM is not importable on the Windows dev box):

    Run this from a SEPARATE venv, not the main training venv -- vllm==0.25.1
    requires torch==2.11.0, which conflicts with requirements.txt's torch==2.12.0
    pin. Set it up once with `bash scripts/setup_vllm_env.sh`, then:
        source venv-vllm/bin/activate

    python scripts/passk_eval.py \
        --model Qwen/Qwen3-0.6B \
        --dataset data_io/raw/math_eval.jsonl \
        --n 64 --temp 0.8 --n_prompts 100 --seed 0 \
        --out results/passk.csv

    # baselines before any training:
    python scripts/passk_eval.py --model Qwen/Qwen3-0.6B  --dataset ... --n 64
    python scripts/passk_eval.py --model Qwen/Qwen3-8B    --dataset ... --n 64
    # after a training run, point --model at the checkpoint dir:
    python scripts/passk_eval.py --model $OUT/checkpoints/<run>/ckpt_best --dataset ... --n 64

MACHINE-TUNING FLAGS (these are the "parameters we need to tune for our machine"):
    --gpu_memory_utilization (default 0.9)   lower if you OOM sharing the GPU
    --tensor_parallel_size   (default 1)     >1 to shard a big teacher (e.g. 32B)
    --max_model_len          (default 4096)  cap context+gen to fit KV cache

NOTE ON GRADING: answer-matching here is a normalized string compare on the last
\\boxed{...} (MATH) / "#### x" (GSM8K) / trailing number. This is good enough for a
*stability diagnostic* but NOT rigorous LaTeX equivalence. For a headline accuracy
number, pip install `math_verify` and swap `is_correct` (in passk_utils.py) for its checker.
Theorem-proving rows (OlympiadBench proofs) have no short answer => skipped, counted
in `n_skipped`.

STATUS: written 2026-07-13, NOT yet run/tested (no vLLM+GPU on the dev box). Validate
one small run (--n 4 --n_prompts 5) on the cluster before trusting the numbers.
"""
import argparse, json, math, os, sys

# Shared, unit-tested grading helpers (repo root on sys.path when run as a script).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from passk_utils import pass_at_k, extract_answer, extract_gold, is_correct   # noqa: E402


def load_prompts(path: str, n_prompts: int):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows = rows[:n_prompts]
    out = []
    n_empty_prompt = 0
    for r in rows:
        prompt = r.get("prompt") or r.get("question") or r.get("problem")
        if not prompt:
            # vLLM raises "decoder prompt cannot be empty" on None/"" -- skip
            # here instead, since a row missing all three fields (or with all
            # of them blank) has nothing gradeable to generate from anyway.
            n_empty_prompt += 1
            continue
        gold_raw = r.get("answer") or r.get("solution") or ""
        gold = extract_gold(gold_raw) if gold_raw else None   # gold may be a full solution or bare answer
        out.append({"prompt": prompt, "gold": gold})
    if n_empty_prompt:
        print(f"[passk] WARNING: {n_empty_prompt}/{len(rows)} rows in {os.path.basename(path)} "
              f"had no usable prompt/question/problem field -- skipped. "
              f"Effective n_prompts = {len(out)}.")
    return out


def compute_passk_for_dataset(llm, dataset_path: str, n: int, temp: float, top_p: float,
                               max_tokens: int, n_prompts: int, k_values: list[int], seed: int):
    """Run one dataset's pass@k against an ALREADY-CONSTRUCTED vLLM engine.

    Split out from main() so a caller evaluating multiple datasets (e.g.
    scripts/passk_eval_and_log.py) can build the LLM engine ONCE and reuse it
    across datasets, instead of paying vLLM's ~1-2 min engine startup per dataset.

    Returns (rows_out, n_graded, n_skipped) where rows_out is [(k, pass_at_k), ...].
    """
    from vllm import SamplingParams

    data = load_prompts(dataset_path, n_prompts)
    sp = SamplingParams(n=n, temperature=temp, top_p=top_p, max_tokens=max_tokens, seed=seed)
    prompts = [d["prompt"] for d in data]
    outputs = llm.generate(prompts, sp)

    per_prompt_c, n_graded, n_skipped = [], 0, 0
    for d, o in zip(data, outputs):
        if d["gold"] is None:
            n_skipped += 1
            continue
        c = sum(1 for comp in o.outputs if is_correct(extract_answer(comp.text), d["gold"]))
        per_prompt_c.append(c)
        n_graded += 1

    rows_out = []
    for k in k_values:
        if k > n:
            continue
        vals = [pass_at_k(n, c, k) for c in per_prompt_c]
        vals = [v for v in vals if not math.isnan(v)]
        pak = sum(vals) / len(vals) if vals else float("nan")
        rows_out.append((k, pak))
    return rows_out, n_graded, n_skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF id or local checkpoint dir")
    ap.add_argument("--dataset", required=True, help="path to a {prompt, answer} jsonl")
    ap.add_argument("--n", type=int, default=64, help="samples per prompt (>= max k)")
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--n_prompts", type=int, default=100, help="FIXED set — keep constant across runs")
    ap.add_argument("--k_values", type=str, default="1,2,4,8,16,32,64")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--out", default=None, help="append per-k rows to this CSV")
    args = ap.parse_args()

    k_values = [int(x) for x in args.k_values.split(",")]
    if max(k_values) > args.n:
        sys.exit(f"--n ({args.n}) must be >= max k ({max(k_values)})")

    try:
        from vllm import LLM
    except ImportError:
        sys.exit("vLLM not installed. On the cluster: pip install vllm")

    llm = LLM(model=args.model, seed=args.seed,
              gpu_memory_utilization=args.gpu_memory_utilization,
              tensor_parallel_size=args.tensor_parallel_size,
              max_model_len=args.max_model_len)

    rows_out, n_graded, n_skipped = compute_passk_for_dataset(
        llm, args.dataset, args.n, args.temp, args.top_p, args.max_tokens,
        args.n_prompts, k_values, args.seed)

    if n_graded == 0:
        sys.exit("No gradeable prompts (all gold answers empty/None).")

    print(f"\nmodel={args.model}  dataset={os.path.basename(args.dataset)}  "
          f"n={args.n} temp={args.temp}  graded={n_graded} skipped={n_skipped}\n")
    print(f"{'k':>4}  {'pass@k':>8}")
    for k, pak in rows_out:
        print(f"{k:>4}  {pak:>8.4f}")

    if args.out:
        newfile = not os.path.exists(args.out)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "a", encoding="utf-8") as f:
            if newfile:
                f.write("model,dataset,n,temp,n_graded,n_skipped,k,pass_at_k\n")
            for k, pak in rows_out:
                f.write(f"{args.model},{os.path.basename(args.dataset)},{args.n},"
                        f"{args.temp},{n_graded},{n_skipped},{k},{pak:.6f}\n")
        print(f"\n[out] appended {len(rows_out)} rows to {args.out}")


if __name__ == "__main__":
    main()
