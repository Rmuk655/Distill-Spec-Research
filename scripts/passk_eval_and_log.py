#!/usr/bin/env python
"""
passk_eval_and_log.py — async vLLM pass@k eval that logs into an IN-PROGRESS
training run's W&B run, instead of writing a CSV.

WHY THIS EXISTS: train.py's in-training pass@k hook (_train_passk) uses plain
HF generate() in an unbatched Python loop -- ~1-2h per val check (see
grad-norm/passk cadence discussion in project history). vLLM does the same
computation in minutes via batched, paged-attention generation. train.py can
spawn THIS script as a non-blocking background subprocess at a val check
(--passk_backend vllm) instead of running the slow hook inline, so training
keeps going immediately while this process evaluates the just-saved checkpoint
in a SEPARATE venv (see scripts/setup_vllm_env.sh) and logs the result back
into the SAME W&B run via --wandb_run_id.

W&B CROSS-PROCESS LOGGING: wandb.init(id=..., resume="must") attaches to an
existing run from a different process; logging a single combined dict (not
one call per dataset) avoids the "second log() call at the same step is
silently dropped" gotcha in wandb>=0.15. train/step is logged explicitly
alongside the passk_* keys so it plots against the SAME custom x-axis
(train/step) the parent process already defined via _define_step_metric,
regardless of what the parent's own internal step counter is doing
concurrently.

CHECKPOINT RACE SAFETY: --model should point at an IMMUTABLE per-step snapshot
directory (e.g. <output>/passk_snapshots/step1200_routine/), NOT the mutable
ckpt_latest/ckpt_best that training keeps overwriting -- train.py is
responsible for creating that snapshot before spawning this process. Pass
--cleanup_model_dir to have this script delete its own snapshot dir after a
successful run (keeps disk usage bounded without the parent needing to track
completion).

USAGE (from venv-vllm, see scripts/setup_vllm_env.sh):
    python scripts/passk_eval_and_log.py \\
        --model /path/to/passk_snapshots/step1200_routine \\
        --datasets math_val --n 16 --n_prompts 100 \\
        --wandb_run_id abc123 --wandb_project distillspec-pipeline \\
        --train_step 1200 --cleanup_model_dir
"""
import argparse, os, shutil, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.passk_eval import compute_passk_for_dataset  # noqa: E402
from data_io import get_path as dataset_path              # noqa: E402 -- same resolver train.py uses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="local checkpoint SNAPSHOT dir (must be immutable)")
    ap.add_argument("--datasets", required=True, help="comma list, e.g. 'math_val' or 'math_val,math_eval'")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--n_prompts", type=int, default=100)
    ap.add_argument("--k_values", type=str, default="1,2,4,8,16,32,64")
    ap.add_argument("--seed", type=int, default=0)
    # Lower default than passk_eval.py's standalone 0.9 -- this process shares the
    # GPU with the actively-training model (optimizer state, activations, grads),
    # not a fully free GPU, so it must not grab most of the memory.
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.2)
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--wandb_run_id", required=True)
    ap.add_argument("--wandb_project", required=True)
    ap.add_argument("--train_step", type=int, required=True)
    ap.add_argument("--tier", default="routine", help="just a print/log tag (routine|full)")
    ap.add_argument("--cleanup_model_dir", action="store_true",
                     help="rmtree --model after a successful run (self-delete the snapshot)")
    args = ap.parse_args()

    k_values = [int(x) for x in args.k_values.split(",")]

    try:
        from vllm import LLM
    except ImportError:
        sys.exit("vLLM not installed in this venv. Run from venv-vllm (scripts/setup_vllm_env.sh).")

    import wandb
    run = wandb.init(id=args.wandb_run_id, project=args.wandb_project, resume="must")

    llm = LLM(model=args.model, seed=args.seed,
              gpu_memory_utilization=args.gpu_memory_utilization,
              tensor_parallel_size=args.tensor_parallel_size,
              max_model_len=args.max_model_len)

    payload = {"train/step": args.train_step}
    for ds in args.datasets.split(","):
        ds = ds.strip()
        rows_out, n_graded, n_skipped = compute_passk_for_dataset(
            llm, dataset_path(ds), args.n, args.temp, args.top_p, args.max_tokens,
            args.n_prompts, k_values, args.seed)
        print(f"[passk-async] {ds}: graded={n_graded} skipped={n_skipped} tier={args.tier} "
              f"step={args.train_step} " + "  ".join(f"k{k}={v:.3f}" for k, v in rows_out))
        for k, v in rows_out:
            payload[f"passk_{ds}/k{k}"] = v

    # Single log() call -- see module docstring on why this must not be split
    # into one call per dataset.
    run.log(payload)
    # Deliberately NOT calling run.finish() here: this process only ATTACHED to
    # the parent training process's still-running W&B run (resume="must"). An
    # explicit finish() would mark the whole shared run "finished" on the W&B
    # backend even though the parent is still actively training -- just let
    # this process exit naturally and let wandb's own atexit hook flush the
    # pending log() call without touching the run's overall state.

    if args.cleanup_model_dir:
        shutil.rmtree(args.model, ignore_errors=True)
        print(f"[passk-async] cleaned up snapshot dir {args.model}")


if __name__ == "__main__":
    main()
