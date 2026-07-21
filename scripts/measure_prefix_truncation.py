#!/usr/bin/env python
"""
measure_prefix_truncation.py — quantify how often the tail-reuse multiroot
prefix-overlap loss (compute_prefix_overlap_multiroot_loss) would have hit the
M1 boundary-truncation bug: root windows tok_lp[r:r+L] silently shortened
because r+L > T (teacher rollout ended before the token budget).

Teacher-only: no student model, no backward pass, no GPU strictly required
(CPU works, just slower). Samples real prompts from the real training data
with the exact same teacher.generate() args train.py uses.

USAGE (remote box, inside the training venv)
  cd ~/Distill-Spec-Research
  python scripts/measure_prefix_truncation.py --n_prompts 200
  python scripts/measure_prefix_truncation.py --n_prompts 200 --rollout_len 128 --configs 8,16,32 --L 8
"""
import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from config import DRAFT_MODEL, TEACHER_MODEL, DEFAULT_MAX_NEW_TOKENS
from data_io import get_path as dataset_path
import verifiers  # noqa: F401  — side effect: adds verifiers/ to sys.path (see train.py)
from util import load_prompts_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="math_hard")
    ap.add_argument("--n_prompts", type=int, default=200)
    ap.add_argument("--rollout_len", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    ap.add_argument("--teacher_temp", type=float, default=1.0)
    ap.add_argument("--L", type=int, default=8)
    ap.add_argument("--configs", default="8,16,32", help="comma-separated N values to evaluate")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"[load] teacher={TEACHER_MODEL}")
    tok = AutoTokenizer.from_pretrained(TEACHER_MODEL)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    teacher = AutoModelForCausalLM.from_pretrained(TEACHER_MODEL, dtype=torch.bfloat16).to(dev).eval()

    prompts = load_prompts_jsonl(dataset_path(args.dataset))[: args.n_prompts]
    print(f"[data] {len(prompts)} prompts from {args.dataset}")

    Ts = []
    with torch.no_grad():
        for i, p in enumerate(prompts):
            ids = tok(p, return_tensors="pt").input_ids.to(dev)
            attn = torch.ones_like(ids)
            gen = teacher.generate(
                ids, attention_mask=attn, max_new_tokens=args.rollout_len,
                do_sample=True, temperature=args.teacher_temp,
                pad_token_id=teacher.config.eos_token_id, use_cache=True,
            )
            T = gen.shape[1] - ids.shape[1]
            Ts.append(T)
            if (i + 1) % 20 == 0:
                print(f"  ...{i+1}/{len(prompts)}")

    Ts = torch.tensor(Ts)
    full = (Ts == args.rollout_len).float().mean().item()
    print(f"\n[rollout lengths] n={len(Ts)}  mean={Ts.float().mean():.1f}  "
          f"min={Ts.min().item()}  max={Ts.max().item()}  "
          f"hit token budget (T==rollout_len)={full*100:.1f}%")

    print(f"\n{'N':>4} {'L':>4}  {'roots (all)':>12} {'roots truncated':>16} {'trunc rate':>10}")
    for N in [int(x) for x in args.configs.split(",")]:
        total_roots, trunc_roots = 0, 0
        for T in Ts.tolist():
            roots = list(range(0, max(T - 1, 0), N))
            for r in roots:
                total_roots += 1
                if r + args.L > T:
                    trunc_roots += 1
        rate = trunc_roots / total_roots if total_roots else 0.0
        print(f"{N:>4} {args.L:>4}  {total_roots:>12} {trunc_roots:>16} {rate*100:>9.2f}%")

    print("\nInterpretation: 'trunc rate' is the fraction of ALL roots across these "
          "sampled rollouts whose window would have been shortened by the M1 bug at "
          "that (N,L,rollout_len). Compare against the ~0.10-0.15 block_eff noise "
          "floor's practical relevance -- a low single-digit-% truncation rate on a "
          "few late roots per run is unlikely to move best_val_block_eff outside "
          "noise; a high rate would warrant re-running the affected sweep arms.")


if __name__ == "__main__":
    main()
