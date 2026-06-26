"""
eval.py — block-efficiency + alpha evaluation for a trained draft model.

Thin wrapper around speculative_decoding_loop() from main.py (GBV source of truth).
This file owns: dataset selection, multi-mode sweep, per-mode seed reset, CSV logging,
GPU/CPU telemetry, and verifier-exception handling.
All speculative decoding logic lives in main.py — do not duplicate it here.

    block_efficiency  = total_generated_tokens / total_target_calls
    throughput        = total_generated_tokens / wall_time   (tok/s)
    avg_tree_nodes    = average tree size per target call

Dataset setup (run once before eval; default gsm8k_eval has 100 prompts):
    python -m data_io.download --datasets gsm8k --n 1000 --force

Usage:
    python eval.py --checkpoint checkpoints/kl_tree/ckpt_best  --mode gbv
    python eval.py --checkpoint checkpoints/gbv_tree/ckpt_best --mode gbv --K 3 --L 8
    python eval.py --checkpoint Qwen/Qwen3-0.6B --mode naive --dataset alpaca   # baseline
    python eval.py --checkpoint ckpts/best --mode gbv --modes naive,gbv,traversal  # sweep

    # one eval per GPU — --device cuda:N is the only control needed:
    python eval.py --checkpoint ckpt_a --modes naive --device cuda:0 &
    python eval.py --checkpoint ckpt_b --modes gbv   --device cuda:1 &

    # parallel eval across modes on separate GPUs (parallel-safe --output flag):
    python eval.py --checkpoint Qwen/Qwen3-0.6B --K 1 --n 1000 --modes naive     --device cuda:0 --output out_naive.csv &
    python eval.py --checkpoint Qwen/Qwen3-0.6B --K 1 --n 1000 --modes specinfer --device cuda:1 --output out_specinfer.csv &

Results print to stdout AND append one row to results.csv for later analysis.

GPU reproducibility protocol (1 eval per GPU):
    • Fix the GPU via --device cuda:N — this is the only flag needed
    • Fix seed via --seed (default 123) — reset before EVERY mode sweep
    • Prompts are always read in file order and sliced [:n]; do not shuffle
    • dtype is fixed to DEFAULT_DTYPE (bf16); do not mix precision across runs
    • CPU threads are capped via --cpu_threads to prevent inter-run variation
    • Both models load to the SAME device in a fixed order (teacher first, then draft)

GPU telemetry notes:
    • SM utilization and memory-bus utilization are polled via pynvml (1 s interval)
      in a background thread during each mode sweep.
    • HBM bandwidth proxy = memory-bus utilization % reported by the GPU driver;
      exact GB/s requires DCGM (install: sudo apt install datacenter-gpu-manager).
    • L2 cache hit rate requires DCGM; not available via standard NVML.
    • PCIe TX/RX throughput (KB/s) is read from NVML.
    • NVLink counters are read if NVLink is present; skipped otherwise.
    • CPU utilization is polled via psutil in the same background thread.
    Install optional deps once: pip install nvidia-ml-py psutil
"""
import argparse
import json
import os
import time

import torch
from tqdm import tqdm

import verifiers  # noqa: F401 — sys.path injection
from util            import load_prompts_jsonl, set_seed, load_models
from main            import speculative_decoding_loop
from delayed_draft   import delayed_speculative_decoding_loop
from verifier_safe   import VerifierError

from data_io import get_path as dataset_path
from config  import (TEACHER_MODEL, DEFAULT_K, DEFAULT_L, DEFAULT_MAX_NEW_TOKENS,
                     DEFAULT_TEMP, DEFAULT_DTYPE, DEFAULT_SEED, VERIFIER_MODES, block_eff)

from telemetry import (GpuMonitor, _gpu_index_from_device, _machine_specs,
                       _model_attn_backend, _auto_cpu_threads)
from eval_io import (_state_path, _load_state, CSV_COLUMNS, append_csv_row,
                     log_result, _resolve_training_url, _FLOAT_FMT)
from diagnostics import run_objective_be_diagnostic, run_decomposition_radar

RESULTS_CSV = os.path.join(os.path.dirname(__file__), "results.csv")


def evaluate_one_mode(p_model, q_model, tok, prompts, mode, K, L,
                      max_new_tokens, temp, state_path: str | None = None,
                      gpu_monitor: GpuMonitor | None = None,
                      warmup_n: int = 3, L1: int = 0, L1_adaptive: bool = False):
    """Run the prompt set under one verifier mode and aggregate stats.

    Verifier exceptions (VerifierError from verifier_safe.py) are caught
    per-prompt: the failing prompt is skipped, its index is logged to
    verifier_errors.log, and the eval continues.  skipped_prompts count
    appears in the returned stats dict.

    Resumes from a prior partial run if state_path exists — already-completed
    prompts are skipped and their stats are loaded from disk so block_eff is
    computed correctly over the full n_prompts denominator.
    """
    done = _load_state(state_path) if state_path else {}
    if done:
        print(f"  [resume] {len(done)}/{len(prompts)} prompts already done "
              f"— skipping, loading from {state_path}")

    state_f = open(state_path, "a", encoding="utf-8") if state_path else None
    all_runs: list[dict] = []
    skipped = 0

    if gpu_monitor is not None:
        _gpu_sidecar = (state_path.replace(".state.jsonl", ".gpu.jsonl")
                        if state_path and state_path.endswith(".state.jsonl")
                        else None)
        gpu_monitor.start(sidecar_path=_gpu_sidecar)

    t_tokenizer_total = 0.0

    # Prime CUDA kernels before the timed loop.  The first few prompts are slow
    # because flash-attention and other CUDA extensions JIT-compile on first use.
    _decoding_loop = (
        delayed_speculative_decoding_loop if L1 > 0 else speculative_decoding_loop
    )
    _loop_kwargs = dict(max_new_tokens=max_new_tokens, K=K, L=L, p_temp=temp, q_temp=temp)
    if L1 > 0:
        _loop_kwargs.update(L1=L1, L1_adaptive=L1_adaptive)

    if warmup_n > 0:
        print(f"  [warmup] running {warmup_n} prompt(s) to prime CUDA kernels ...")
        for wp in prompts[:warmup_n]:
            p_model._spec_profile = {"runs": []}
            try:
                _decoding_loop(
                    p_model=p_model, q_model=q_model, tok=tok,
                    prompt=wp, verification_algo=mode,
                    **_loop_kwargs,
                )
            except Exception:
                pass
        torch.cuda.synchronize()

    try:
        for i, prompt in enumerate(tqdm(prompts, desc=f"mode={mode} K={K} L={L}", ncols=80)):
            if i in done:
                all_runs.append(done[i])
                continue

            # Time the tokenizer separately so we can attribute its cost.
            _t0_tok = time.perf_counter()
            _ = tok.encode(prompt)  # warm the tokenizer; actual encode is inside the loop
            t_tokenizer_total += time.perf_counter() - _t0_tok

            p_model._spec_profile  = {"runs": []}
            p_model._spec_prompt_idx = i  # propagated to verifier_safe via _spec_debug_ctx

            try:
                _decoding_loop(
                    p_model=p_model, q_model=q_model, tok=tok,
                    prompt=prompt, verification_algo=mode,
                    **_loop_kwargs,
                )
            except VerifierError as ve:
                skipped += 1
                print(f"\n  [skip] prompt {i}: verifier raised → {ve}")
                print(f"         full context written to verifier_errors.log")
                continue

            run = p_model._spec_profile["runs"][0]
            run["prompt_idx"] = i
            all_runs.append(run)

            if state_f:
                state_f.write(json.dumps(run) + "\n")
                state_f.flush()
    finally:
        if gpu_monitor is not None:
            gpu_monitor.stop()
        if state_f:
            state_f.close()

    total_calls  = sum(r["target_calls"]     for r in all_runs)
    total_gen    = sum(r["gen_tokens"]       for r in all_runs)
    total_time   = sum(r["total_time"]       for r in all_runs)
    total_nodes  = sum(r["total_tree_nodes"] for r in all_runs)
    time_draft   = sum(r.get("time_draft",   0.0) for r in all_runs)
    time_target  = sum(r.get("time_target",  0.0) for r in all_runs)
    time_verify  = sum(r.get("time_verify",  0.0) for r in all_runs)
    time_cache   = sum(r.get("time_cache",   0.0) for r in all_runs)

    stats = {
        "n_prompts":         len(all_runs),
        "skipped_prompts":   skipped,
        "target_calls":      total_calls,
        "block_eff":         block_eff(total_gen, total_calls),
        "throughput_tok_s":  total_gen / total_time    if total_time  > 0 else float("nan"),
        "time_per_call_ms":  1000.0 * total_time / total_calls  if total_calls > 0 else float("nan"),
        "time_per_token_ms": 1000.0 * total_time / total_gen    if total_gen   > 0 else float("nan"),
        "avg_tree_nodes":    total_nodes / total_calls if total_calls > 0 else float("nan"),
        "total_gen_tokens":  total_gen,
        # per-prompt block_eff — used only by --diagnose; not a CSV column.
        "per_prompt_be": {r["prompt_idx"]: block_eff(r["gen_tokens"], r["target_calls"])
                          for r in all_runs if "prompt_idx" in r and r["target_calls"]},
        "total_time_s":      total_time,
        "time_draft_s":      time_draft,
        "time_target_s":     time_target,
        "time_verify_s":     time_verify,
        "time_cache_s":      time_cache,
        "time_tokenizer_s":  t_tokenizer_total,
    }

    # Merge GPU telemetry into stats dict.
    if gpu_monitor is not None:
        stats.update(gpu_monitor.summary())

    return stats



def parse_args():
    ap = argparse.ArgumentParser(description="Block-efficiency eval (A100, Qwen3).")
    ap.add_argument("--checkpoint", required=True,
                    help="Path to draft checkpoint dir (or HF model id for baseline).")
    ap.add_argument("--mode",      default="gbv",   choices=VERIFIER_MODES,
                    help="Verifier mode (default gbv).  Override via --modes for a sweep.")
    ap.add_argument("--modes",     default=None,
                    help="Comma-separated verifier modes for a sweep, e.g. 'naive,gbv,traversal'.")
    ap.add_argument("--dataset",   default="gsm8k_eval",
                    help="Dataset name (gsm8k_eval, alpaca, math500, humaneval, mtbench).")
    ap.add_argument("--K",         type=int, default=DEFAULT_K)
    ap.add_argument("--L",         type=int, default=DEFAULT_L)
    ap.add_argument("--n",         type=int, default=100,
                    help="How many prompts from the dataset to evaluate.")
    ap.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    ap.add_argument("--temp",      type=float, default=DEFAULT_TEMP)
    ap.add_argument("--device", default="cuda:0",
                    help="CUDA device to run on, e.g. --device cuda:0 or --device cuda:2. "
                         "This is the only flag needed to select a GPU — no environment "
                         "variables required.  Default: cuda:0.")
    ap.add_argument("--seed",      type=int, default=DEFAULT_SEED,
                    help="RNG seed (default 123).  Reset before EVERY mode to keep "
                         "prompt order and sampling identical across runs.")
    ap.add_argument("--cpu_threads", type=int, default=None,
                    help="Override the auto-computed PyTorch CPU thread count.  "
                         "Auto default: physical_cores // num_gpus (e.g. 96 cores / 4 GPUs = 24).  "
                         "When 4 evals run simultaneously (1 per GPU), each needs its own CPU "
                         "budget so the 4 processes don't compete for the same 96 cores.  "
                         "Within a single eval, prompts are always sequential; these threads "
                         "are for PyTorch intra-op parallelism (BLAS etc.), not prompt batching.")
    ap.add_argument("--output",    default=None,
                    help="Override output CSV path (default: results.csv next to eval.py). "
                         "Set to a unique path when running multiple parallel processes.")
    ap.add_argument("--L1",        type=int, default=0,
                    help="Delayed-expansion branch depth (0 = normal root branching, default). "
                         "When L1 > 0, the draft runs a single path for L1 steps then branches "
                         "into K i.i.d. paths for the remaining L-L1 steps.  Uses "
                         "delayed_speculative_decoding_loop from delayed_draft.py; "
                         "Rahul's inference_util.py is not modified.")
    ap.add_argument("--L1_adaptive", action="store_true",
                    help="Adapt L1 per-iteration using lagged teacher entropy from the previous "
                         "target pass (Option A — zero extra target forward pass cost). "
                         "--L1 becomes L1_max; iterations where teacher entropy > 1.5 nats "
                         "fall back to root branching (L1=0).  Only active when --L1 > 0.")
    ap.add_argument("--warmup_n",  type=int, default=3,
                    help="Prompts to run (untimed) before the timed loop to prime CUDA kernels. "
                         "Default 3.  Set 0 to skip (faster iteration, less accurate throughput).")

    ap.add_argument("--no_gpu_monitor", action="store_true",
                    help="Disable the background GPU/CPU telemetry thread.  "
                         "The thread polls pynvml at 1 Hz (~10 μs per call, 0.001%% overhead) "
                         "so disabling it only makes sense if you are profiling at microsecond "
                         "resolution and want a completely clean baseline.")
    ap.add_argument("--diagnose", action="store_true",
                    help="DIAGNOSTIC MODE (off by default — NOT for throughput runs). "
                         "After the timed eval, run a SEPARATE pass that computes, per "
                         "prompt, the training divergence (JSD and forward-KL between "
                         "teacher and draft) and correlates it with that prompt's block "
                         "efficiency.  Answers: does the objective the losses minimise "
                         "actually predict BE?  A flat correlation = objective mismatch "
                         "(minimising divergence won't move BE).  Opens a W&B run and logs "
                         "a scatter + correlation + verdict.  Adds forward passes, so it is "
                         "kept entirely outside the timed loop — throughput is unaffected.")
    ap.add_argument("--trained_loss", default=None,
                    help="Override for which divergence family was trained: 'jsd' or 'fwdkl'. "
                         "Normally auto-read from state.json inside the checkpoint directory "
                         "(written by train.py as 'loss': args.loss).  Only needed for old "
                         "checkpoints saved before this field was added.")
    ap.add_argument("--baseline_checkpoint", default=None,
                    help="Untrained base draft checkpoint for decomposition analysis "
                         "(e.g. 'Qwen/Qwen3-0.6B').  Only used together with --diagnose. "
                         "Loads the base draft, computes G₀ (baseline JSD per prompt), "
                         "then produces: (1) a stacked-bar showing learned vs remaining "
                         "gap per prompt; (2) a radar showing mean BE across 4 difficulty "
                         "quartiles defined by G₀.  Logs both as W&B images "
                         "(job_type='decompose').  Requires matplotlib.")
    return ap.parse_args()


def main():
    args = parse_args()

    # ── GPU selection — --device cuda:N is the single source of truth ────────
    # torch.cuda.set_device() pins all subsequent CUDA ops to GPU N.
    # The same index N is passed to pynvml so telemetry monitors the right GPU.
    torch_device = args.device
    phys_gpu_idx = _gpu_index_from_device(args.device)
    if torch.cuda.is_available():
        torch.cuda.set_device(phys_gpu_idx)

    # CPU thread budget: auto-detect unless user overrides.
    cpu_threads = args.cpu_threads if args.cpu_threads is not None else _auto_cpu_threads()
    torch.set_num_threads(cpu_threads)
    torch.set_num_interop_threads(cpu_threads)
    print(f"[init] device={args.device}  CPU threads={cpu_threads} "
          f"({'user override' if args.cpu_threads is not None else 'auto: cores/gpus'})")

    set_seed(args.seed)

    csv_path = args.output or RESULTS_CSV

    # Collect machine specs once — stored in every CSV row for reproducibility.
    specs = _machine_specs(phys_gpu_idx)
    print(f"[hw]   GPU: {specs.get('machine_gpu', 'unknown')}  "
          f"({specs.get('machine_gpu_count', '?')} GPUs, "
          f"{specs.get('machine_gpu_vram_gb', '?')} GB VRAM each)  "
          f"CPU: {specs.get('machine_cpu_physical_cores', specs.get('machine_cpu_logical_cores', '?'))} physical cores  "
          f"RAM: {specs.get('machine_ram_gb', '?')} GB")

    # ── Resolve data + modes BEFORE loading models, so a fully-cached run
    #    can short-circuit without paying the ~8B teacher model-load cost ────
    data_path = dataset_path(args.dataset)
    prompts   = load_prompts_jsonl(data_path)[:args.n]
    modes = [m.strip() for m in args.modes.split(",")] if args.modes else [args.mode]
    unknown = [m for m in modes if m not in VERIFIER_MODES]
    if unknown:
        raise SystemExit(f"Unknown verifier mode(s): {unknown}. Valid: {VERIFIER_MODES}")

    # A mode needs the models only if some prompt is still unfinished.
    mode_state: dict[str, tuple[str, bool]] = {}
    need_models = False
    for mode in modes:
        sp = _state_path(csv_path, mode, args.K, args.L, args.checkpoint, args.dataset)
        done = _load_state(sp)
        complete = bool(prompts) and all(i in done for i in range(len(prompts)))
        mode_state[mode] = (sp, complete)
        if not complete:
            need_models = True
    # --diagnose needs the models for its extra forward passes even when every
    # mode's block_eff is already cached.
    if args.diagnose:
        need_models = True

    # ── Load models (teacher first, then draft) — skipped entirely when every
    #    requested mode is already fully cached for this (K, L) ──────────────
    if need_models:
        print(f"[load] teacher={TEACHER_MODEL}")
        print(f"[load] draft={args.checkpoint}")
        print(f"[load] device={torch_device}  dtype={DEFAULT_DTYPE}  seed={args.seed}")
        tok, p_model, q_model = load_models(TEACHER_MODEL, args.checkpoint,
                                            device=torch_device, dtype=DEFAULT_DTYPE)
        specs["attn_backend"] = _model_attn_backend(p_model, q_model)
        print(f"[load] attention_backend={specs['attn_backend']}")
        # Log GPU memory after model load — both models share the same device.
        if torch.cuda.is_available():
            _alloc = torch.cuda.memory_allocated() / 1024**2
            _reserv = torch.cuda.memory_reserved() / 1024**2
            print(f"[gpu]  after model load — allocated={_alloc:.0f}MB  reserved={_reserv:.0f}MB")
    else:
        tok = p_model = q_model = None
        print(f"[load] all {len(modes)} mode(s) fully cached for "
              f"K={args.K} L={args.L} — skipping model load")

    print(f"[data] {data_path} — {len(prompts)} prompts")

    print()
    print("=" * 78)
    print(f"  Eval  draft={args.checkpoint}  dataset={args.dataset}  "
          f"K={args.K} L={args.L} n={len(prompts)}")
    print(f"  device={args.device}  dtype={DEFAULT_DTYPE}  seed={args.seed}")
    print("=" * 78)

    all_stats = {}
    for mode in modes:
        set_seed(args.seed)   # identical RNG state for every mode

        sp, complete = mode_state[mode]
        # Only monitor the GPU for modes that will actually run work.
        mon = (GpuMonitor(device_idx=phys_gpu_idx)
               if not complete and not args.no_gpu_monitor else None)

        stats = evaluate_one_mode(
            p_model, q_model, tok, prompts, mode,
            K=args.K, L=args.L,
            max_new_tokens=args.max_new_tokens, temp=args.temp,
            state_path=sp, gpu_monitor=mon,
            warmup_n=args.warmup_n,
            L1=args.L1, L1_adaptive=args.L1_adaptive,
        )
        all_stats[mode] = stats
        # Skip the CSV append for a run that was already fully cached at start
        # — re-appending an identical row only adds duplicates. A partial run
        # that *completes* during this invocation had complete=False, so it
        # still logs.
        if complete:
            print(f"  [cache] mode={mode} already complete — "
                  f"not re-appending to {os.path.basename(csv_path)} "
                  f"(BE={stats['block_eff']:.4f})")
        else:
            log_result(stats, args, mode, mon, csv_path,
                       specs=specs, cpu_threads_used=cpu_threads, phys_gpu_idx=phys_gpu_idx)

    print(f"\n[done] results appended to {csv_path}")

    # ── Diagnostic pass (--diagnose only) — strictly AFTER the timed eval, so it
    #    never contaminates throughput.  Correlates per-prompt training divergence
    #    with the first mode's per-prompt block efficiency. ────────────────────
    if args.diagnose:
        primary = modes[0]
        per_prompt_be = all_stats.get(primary, {}).get("per_prompt_be", {})
        # Resolve which loss was trained with.
        # Priority: explicit --trained_loss > state.json (written by train.py) > path heuristic.
        def _loss_to_divergence(loss_name: str) -> str:
            """Map full loss name (e.g. 'jsd_flat_enrich') to divergence family ('jsd'/'fwdkl')."""
            l = loss_name.lower()
            if "fwdkl" in l or ("kl" in l and "jsd" not in l):
                return "fwdkl"
            return "jsd"

        if args.trained_loss:
            trained_loss = args.trained_loss.lower()
        else:
            _state_file = os.path.join(args.checkpoint, "state.json")
            try:
                import json as _json
                _state = _json.load(open(_state_file))
                # train_args (full vars(args)) written by train.py >= this commit;
                # fall back to bare "loss" field from the previous commit.
                _saved_loss = (_state.get("train_args") or {}).get("loss") \
                              or _state.get("loss")
                if _saved_loss:
                    trained_loss = _loss_to_divergence(_saved_loss)
                    print(f"  [diagnose] checkpoint loss='{_saved_loss}' → primary divergence: {trained_loss}")
                else:
                    raise ValueError("no loss field in state.json")
            except Exception as _e:
                trained_loss = _loss_to_divergence(args.checkpoint)
                print(f"  [diagnose] state.json unreadable/missing loss ({type(_e).__name__}: {_e}); inferred from path: {trained_loss}")
        if not per_prompt_be:
            print("  [diagnose] no per-prompt BE available — skipping diagnostic.")
        else:
            diag = run_objective_be_diagnostic(p_model, q_model, tok, prompts,
                                               per_prompt_be, args, primary, torch_device,
                                               trained_loss=trained_loss)
            if args.baseline_checkpoint and diag is not None:
                jsd_xs, _fkl_xs, be_ys, valid_indices = diag
                if len(jsd_xs) >= 4:
                    run_decomposition_radar(
                        p_model, tok, prompts,
                        jsd_xs, be_ys, valid_indices,
                        args, primary, torch_device,
                    )
                else:
                    print(f"  [decompose] only {len(jsd_xs)} valid prompts — need ≥4, skipping.")


if __name__ == "__main__":
    main()
