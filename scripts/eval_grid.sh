#!/usr/bin/env bash
# eval_grid.sh — full post-training eval sweep for the Qwen3-1.7B/32B capacity study.
#
# One GPU per checkpoint (mirrors the training launch grid). Each GPU sequentially
# covers: 2 datasets (math_eval, olympiad_eval) x K=1..4 x all 9 verifier modes,
# plus a delayed-expansion traversal pass (--L1_adaptive) at each K/dataset.
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
#   nohup bash scripts/eval_grid.sh > $OUT/output/eval_grid_driver.out 2>&1 &
#   disown
set -uo pipefail   # NOT -e: one failed eval cell must not kill the other 7 GPUs' loops

REPO=/home/colligo/Distill-Spec-Research
OUT=/sensei-fs-3/users/rkrishna/Qwen32B-Qwen1.7B
TEACHER=Qwen/Qwen3-32B
MODES="naive,nss,specinfer,spectr,khisti,max,bv,gbv,traversal"
DATASETS="math_eval olympiad_eval"
KS="1 2 3 4"
SEED=123

mkdir -p "$OUT/logs" "$OUT/output"

# GPU index -> checkpoint dir name (must match the training launch grid).
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
                --dataset "$dataset" --device "cuda:${gpu}" \
                --output "$csv" >> "$out" 2>&1

            echo "[eval_grid] GPU${gpu} ${name} dataset=${dataset} K=${K} delayed-traversal (--L1_adaptive)" >> "$out"
            python "$REPO/eval.py" --checkpoint "$ckpt" --teacher "$TEACHER" \
                --mode traversal --L1_adaptive --K "$K" --L 8 --n 100 --seed "$SEED" \
                --dataset "$dataset" --device "cuda:${gpu}" \
                --output "$csv" >> "$out" 2>&1
        done
    done
    echo "[eval_grid] GPU${gpu} ${name} DONE" >> "$out"
}

for gpu in "${!NAMES[@]}"; do
    run_gpu_eval "$gpu" "${NAMES[$gpu]}" &
done

wait
echo "[eval_grid] all 8 GPUs finished"
