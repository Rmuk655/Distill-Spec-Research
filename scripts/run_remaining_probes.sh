#!/usr/bin/env bash
# =============================================================================
# Runs the 6 remaining boundary-LR probes (prob below 1e-5, jsd below 1e-5,
# per Rahul's 2026-07-16 answers 3/4) queued across however many GPUs you
# give it -- each GPU processes its share of the queue sequentially, so you
# can fire this once and let everything finish overnight unattended.
#
# USAGE:
#   GPUS=0,1,2,3 bash scripts/run_remaining_probes.sh
#   DRY_RUN=1 GPUS=0,1,2,3 bash scripts/run_remaining_probes.sh   # preview only
# =============================================================================
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="/sensei-fs-3/users/rkrishna/checkpoints/Qwen0.6B_Qwen8B/prefix_overlap"
mkdir -p "$OUT"

IFS=',' read -r -a GPU_ARR <<< "${GPUS:-0}"
NGPU=${#GPU_ARR[@]}
DRY_RUN="${DRY_RUN:-0}"

# spec: "name|extra_args"
JOBS=(
  "po_prob_lr1e-6_wu20|--loss prefix_overlap --prefix_objective prob --lr 1e-6 --warmup_steps 125 --lr_min_ratio 0.1 --weight_decay 0.01"
  "po_prob_lr2e-6_wu20|--loss prefix_overlap --prefix_objective prob --lr 2e-6 --warmup_steps 125 --lr_min_ratio 0.1 --weight_decay 0.01"
  "po_prob_lr5e-6_wu20|--loss prefix_overlap --prefix_objective prob --lr 5e-6 --warmup_steps 125 --lr_min_ratio 0.1 --weight_decay 0.01"
  "po_prob_lr7e-6_wu20|--loss prefix_overlap --prefix_objective prob --lr 7e-6 --warmup_steps 125 --lr_min_ratio 0.1 --weight_decay 0.01"
  "jsd_lr3e-6_wu10_lrmin0.1_wd0.01|--loss jsd --lr 3e-6 --warmup_steps 62 --lr_min_ratio 0.1 --weight_decay 0.01"
  "jsd_lr1e-6_wu10_lrmin0.1_wd0.01|--loss jsd --lr 1e-6 --warmup_steps 62 --lr_min_ratio 0.1 --weight_decay 0.01"
)
echo "[remaining-probes] jobs=${#JOBS[@]}  gpus=${GPUS:-0}"

run_job() {
  local gpu="$1" spec="$2"
  IFS='|' read -r name extra <<< "$spec"
  local out="${OUT}/${name}" log="${OUT}/${name}.log"
  if [[ -e "${out}/ckpt_best" ]]; then echo "[gpu${gpu}] SKIP done: ${name}"; return; fi

  local cmd="python -u ${REPO}/train.py ${extra} \
    --draft Qwen/Qwen3-0.6B --teacher Qwen/Qwen3-8B \
    --train_dataset math_hard --val_dataset math_val \
    --steps 5000 --seed 123 \
    --passk_datasets math_val --passk_backend vllm --passk_samples 16 --passk_every_n_vals 5 \
    --output ${out}"

  echo "[gpu${gpu}] RUN: ${name}"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "         CUDA_VISIBLE_DEVICES=${gpu} ${cmd}" | tr -s ' '
  else
    CUDA_VISIBLE_DEVICES="${gpu}" bash -c "${cmd}" >> "${log}" 2>&1
  fi
}

gpu_worker() {
  local gi="$1" gpu="${GPU_ARR[$1]}" j
  for (( j=0; j<${#JOBS[@]}; j++ )); do
    (( j % NGPU == gi )) || continue
    run_job "${gpu}" "${JOBS[$j]}"
  done
  echo "[gpu${gpu}] worker done."
}
for (( gi=0; gi<NGPU; gi++ )); do gpu_worker "$gi" & done
wait
echo "[remaining-probes] all done."
