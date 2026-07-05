#!/usr/bin/env bash
# eval_grid.sh — full post-training eval sweep for the Qwen3-1.7B/32B capacity study.
#
# One GPU per checkpoint (mirrors the training launch grid). Each GPU sequentially
# covers: 2 datasets (math_eval, olympiad_eval) x K=1..4 x all 9 verifier modes,
# plus DDTE-style delayed-expansion passes (--L1_adaptive, and fixed --L1 3/4/5)
# on traversal+specinfer at each K/dataset — delayed_draft.py's own docstring
# flags these two verifiers as the theoretically-motivated ones for L1.
#
# Resumable by construction: eval.py caches completed (mode,K,L,checkpoint,dataset)
# work per-prompt and skips it on re-run without reloading models. If this script
# (or one GPU's process) dies partway through, just re-run this exact script again
# — completed work is skipped, incomplete work continues from its cached state.
#
# Teacher is loaded in full bf16 (NOT the NF4 quantized dir used for training) —
# eval has no optimizer-state memory pressure, so use the deployment-accurate
# teacher for the reported numbers even though training used a 4-bit approximation.
#
# Usage:
#   Full grid (all 8 checkpoints, one per GPU):
#     nohup bash scripts/eval_grid.sh > $OUT/output/eval_grid_driver.out 2>&1 &
#     disown
#
#   Single checkpoint on one GPU:
#     bash scripts/eval_grid.sh <gpu> <name>
#     e.g.: bash scripts/eval_grid.sh 0 traversal_log_K3_L8_math_hard_s123
#
# Reusable across model pairs / boxes via env var overrides (all optional,
# default to the 1.7B/32B capacity-study values below):
#   OUT        — checkpoint+log root. Checkpoints expected at $OUT/checkpoints/<name>/ckpt_best
#   TEACHER    — HF id of the eval-time teacher
#   NAMES_CSV  — comma-separated checkpoint dir names (full-grid mode only;
#                overrides the default 8-checkpoint NAMES array below,
#                one GPU per entry, GPU index = array index)
#   DATASETS, KS, MODES, DELAYED_MODES, L1_VALUES, SEED — sweep parameters
#   COMPILE=1  — BROKEN, do not use (2026-07-05). Passes --compile to every
#                eval.py call: torch.compile(mode='reduce-overhead') on the
#                draft model only, meant to cut the kernel-launch overhead
#                that dominates wall time in iid_draft's sequential small-batch
#                calls (SM util ~25-28% uncompiled). In practice it crashes:
#                iid_draft builds a fresh DynamicCache per prompt, so the
#                cache's `is_initialized` bool re-guards/recompiles the draft
#                forward on every prompt; once torch._dynamo.config.
#                recompile_limit (8) is hit, torch._inductor's cudagraph_trees
#                bookkeeping desyncs and raises AssertionError in
#                dealloc_current_path_weakrefs — reproduced on 2+ independent
#                checkpoints. Defaults to off (0); leave it off until this is
#                fixed upstream (a non-cudagraph torch.compile(dynamic=True),
#                i.e. dropping mode='reduce-overhead', avoids this specific
#                crash but is untested here and gives a smaller speedup).
# e.g. (0.6B/8B box, 3 checkpoints, math_eval only, no delayed-L1 sweep):
#   OUT=/sensei-fs-3/users/rkrishna TEACHER=Qwen/Qwen3-8B DATASETS=math_eval \
#     NAMES_CSV=jsd_flat_enrich_K3_math_hard_s123,jsd_flat_enrich_K3_temp07_math_hard_s123,jsd_flat_enrich_K4_math_hard_s123 \
#     nohup bash scripts/eval_grid.sh > /sensei-fs-3/users/rkrishna/output/eval_grid_driver.out 2>&1 &
#
# Intermediate results: $OUT/logs/<name>.csv — one row per completed
# (checkpoint, dataset, mode, K, L) tuple written by eval.py as it finishes.
# Resumable: re-running skips already-completed rows.
set -uo pipefail   # NOT -e: one failed eval cell must not kill the other 7 GPUs' loops

REPO=/home/colligo/Distill-Spec-Research
OUT="${OUT:-/sensei-fs-3/users/rkrishna/Qwen32B-Qwen1.7B}"
TEACHER="${TEACHER:-Qwen/Qwen3-32B}"
MODES="${MODES:-naive,nss,specinfer,spectr,khisti,max,bv,gbv,traversal}"
DELAYED_MODES="${DELAYED_MODES:-traversal,specinfer}"   # the two verifiers L1 is theoretically motivated for
L1_VALUES="${L1_VALUES:-3 4 5}"
DATASETS="${DATASETS:-math_eval olympiad_eval}"
KS="${KS:-1 2 3 4}"
SEED="${SEED:-123}"
COMPILE_FLAG=""
[ "${COMPILE:-0}" = "1" ] && COMPILE_FLAG="--compile"

mkdir -p "$OUT/logs" "$OUT/output"

# GPU index -> checkpoint dir name (full-grid mode only). Override via
# NAMES_CSV=name1,name2,... ; defaults to the 1.7B/32B training launch grid.
if [ -n "${NAMES_CSV:-}" ]; then
    IFS=',' read -r -a NAMES <<< "$NAMES_CSV"
else
    NAMES=(
      jsd_math_hard_s123
      enrich_K3_math_hard_s123
      jsd_dw_lin_naive_K3_math_hard_s123
      traversal_log_K3_L8_math_hard_s123
      nss_log_K3_L8_math_hard_s123
      po_logprob_multiN4_L8_math_hard_s123
      po_nss_multiN4_L8_math_hard_s123
      po_traversal_multiN4_L8_math_hard_s123
    )
fi

run_gpu_eval() {
    local gpu="$1"
    local name="$2"
    local ckpt="$OUT/checkpoints/${name}/ckpt_best"
    local csv="$OUT/logs/${name}.csv"
    local out="$OUT/output/${name}_eval.out"

    if [ ! -d "$ckpt" ]; then
        echo "[eval_grid] GPU${gpu}: WARNING — checkpoint not found at ${ckpt}, skipping" >> "$out"
        return
    fi

    for dataset in $DATASETS; do
        for K in $KS; do
            echo "[eval_grid] GPU${gpu} ${name} dataset=${dataset} K=${K} modes-sweep" >> "$out"
            python "$REPO/eval.py" --checkpoint "$ckpt" --teacher "$TEACHER" \
                --modes "$MODES" --K "$K" --L 8 --n 100 --seed "$SEED" \
                --dataset "$dataset" --device "cuda:${gpu}" $COMPILE_FLAG \
                --output "$csv" >> "$out" 2>&1

            echo "[eval_grid] GPU${gpu} ${name} dataset=${dataset} K=${K} delayed (--L1_adaptive) ${DELAYED_MODES}" >> "$out"
            python "$REPO/eval.py" --checkpoint "$ckpt" --teacher "$TEACHER" \
                --modes "$DELAYED_MODES" --L1_adaptive --K "$K" --L 8 --n 100 --seed "$SEED" \
                --dataset "$dataset" --device "cuda:${gpu}" $COMPILE_FLAG \
                --output "$csv" >> "$out" 2>&1

            for L1 in $L1_VALUES; do
                echo "[eval_grid] GPU${gpu} ${name} dataset=${dataset} K=${K} delayed (--L1 ${L1}) ${DELAYED_MODES}" >> "$out"
                python "$REPO/eval.py" --checkpoint "$ckpt" --teacher "$TEACHER" \
                    --modes "$DELAYED_MODES" --L1 "$L1" --K "$K" --L 8 --n 100 --seed "$SEED" \
                    --dataset "$dataset" --device "cuda:${gpu}" $COMPILE_FLAG \
                    --output "$csv" >> "$out" 2>&1
            done
        done
    done
    echo "[eval_grid] GPU${gpu} ${name} DONE" >> "$out"
}

if [ $# -eq 2 ]; then
    # Single-checkpoint mode: bash eval_grid.sh <gpu> <name>
    run_gpu_eval "$1" "$2"
else
    # Full grid: one GPU per checkpoint
    for gpu in "${!NAMES[@]}"; do
        run_gpu_eval "$gpu" "${NAMES[$gpu]}" &
    done
    wait
    echo "[eval_grid] all ${#NAMES[@]} GPUs finished"
fi
