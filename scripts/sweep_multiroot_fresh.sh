#!/usr/bin/env bash
# =============================================================================
# STAGED (coordinate-descent) sweep for the doc-faithful multi-root prefix-
# overlap estimator (--prefix_multiroot_fresh). Sweeps N, random-offset, and M
# one axis at a time instead of the full N x offset x M cross product (4x2x3=24
# runs) -- lock in each winner before moving to the next axis, same pattern as
# scripts/sweep_prefix_overlap.sh.
#
# Cost note: --prefix_multiroot_fresh scales roughly num_roots x M teacher
# generations per step (num_roots = rollout_len / N). Larger N or larger M =
# more expensive; the N stage runs cheapest-first is not guaranteed, check
# early step-throughput on each new axis value before trusting a multi-day run.
#
# WORKFLOW
#   1) STAGE=N       GPUS=0,1,2,3 bash scripts/sweep_multiroot_fresh.sh   # sweep N, M=1, no offset
#      -> pick best N (check val block_eff vs noise floor, and vs the old
#         tail-reuse estimator's own N result)
#   2) STAGE=offset  LOCK_N=<winner> GPUS=... bash scripts/sweep_multiroot_fresh.sh
#      -> pick best offset (off/on)
#   3) STAGE=M       LOCK_N=<winner> LOCK_OFFSET=<winner> GPUS=... bash ...
#      -> pick best M
#
#   Already-completed (ckpt_best exists) combos are auto-skipped, so re-running
#   after a crash or across stages never repeats finished work.
#
#   DRY_RUN=1 STAGE=N ... bash ...   # print the plan, run nothing
# =============================================================================
set -uo pipefail

# ---- fixed paths / models ----------------------------------------------------
CKPT_ROOT="${CKPT_ROOT:-/home/colligo/local_ckpts/prefix_overlap}"
DRAFT="Qwen/Qwen3-0.6B"
TEACHER="Qwen/Qwen3-8B"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-python -u}"

# ---- fixed hyperparameters (matched to the plain prob baseline) --------------
STEPS="${STEPS:-5000}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
SEED="${SEED:-123}"
LR="${LR:-1e-5}"
WARMUP_PCT="${WARMUP_PCT:-20}"
LRMIN="${LRMIN:-0.1}"
WD="${WD:-0.01}"
ROLLOUT_LEN="${ROLLOUT_LEN:-128}"
TRAIN_DS="${TRAIN_DS:-math_hard}"
VAL_DS="${VAL_DS:-math_val}"
PASSK_BACKEND="${PASSK_BACKEND:-hf}"
PASSK_SAMPLES="${PASSK_SAMPLES:-16}"
PASSK_EVERY_N_VALS="${PASSK_EVERY_N_VALS:-5}"

WU_TOTAL=$(( STEPS / GRAD_ACCUM ))
WU_STEPS=$(( WU_TOTAL * WARMUP_PCT / 100 )); (( WU_STEPS < 1 )) && WU_STEPS=1

# ---- staged sweep: which axis + locked (winner) values for the others --------
STAGE="${STAGE:?set STAGE=N|offset|M}"
LOCK_N="${LOCK_N:-16}"
LOCK_OFFSET="${LOCK_OFFSET:-off}"       # off|on
LOCK_M="${LOCK_M:-1}"

# axis value lists -- adjust here if you want a wider/narrower grid
NS=(8 16 32 64)
OFFSETS=(off on)
MS=(1 4 8)

# ---- build the job list for THIS stage only ----------------------------------
JOBS=()
case "$STAGE" in
  N)      for v in "${NS[@]}";      do JOBS+=("${v}|${LOCK_OFFSET}|${LOCK_M}"); done ;;
  offset) for v in "${OFFSETS[@]}"; do JOBS+=("${LOCK_N}|${v}|${LOCK_M}");      done ;;
  M)      for v in "${MS[@]}";      do JOBS+=("${LOCK_N}|${LOCK_OFFSET}|${v}"); done ;;
  *) echo "bad STAGE='$STAGE' (want N|offset|M)"; exit 1 ;;
esac
echo "[sweep] STAGE=${STAGE}  jobs=${#JOBS[@]}  (locked: N=${LOCK_N} offset=${LOCK_OFFSET} M=${LOCK_M})"
mkdir -p "${CKPT_ROOT}"

# ---- one job -----------------------------------------------------------------
run_job() {   # $1 = gpu id, $2 = "N|offset|M"
  local gpu="$1" spec="$2"
  IFS='|' read -r n offset m <<< "$spec"
  local offset_flag=""; [[ "$offset" == "on" ]] && offset_flag="--prefix_random_offset"
  local offset_tag=""; [[ "$offset" == "on" ]] && offset_tag="_randoff"
  local name="po_prob_multiroot_N${n}_freshM${m}${offset_tag}_lr${LR}_wu${WARMUP_PCT}"
  local out="${CKPT_ROOT}/${name}" log="${CKPT_ROOT}/${name}.log"

  if [[ -e "${out}/ckpt_best" ]]; then echo "[gpu${gpu}] SKIP done: ${name}"; return; fi
  local resume=""; [[ -e "${out}/ckpt_latest" ]] && resume="--resume" && echo "[gpu${gpu}] RESUME: ${name}"

  local cmd="${PY} ${REPO}/train.py \
    --loss prefix_overlap --prefix_objective prob \
    --draft ${DRAFT} --teacher ${TEACHER} \
    --train_dataset ${TRAIN_DS} --val_dataset ${VAL_DS} \
    --steps ${STEPS} --seed ${SEED} \
    --lr ${LR} --warmup_steps ${WU_STEPS} --lr_min_ratio ${LRMIN} --weight_decay ${WD} \
    --prefix_multiroot_fresh --prefix_root_spacing ${n} --prefix_rollout_len ${ROLLOUT_LEN} \
    --prefix_multiroot_M ${m} ${offset_flag} \
    --passk_datasets ${VAL_DS} --passk_backend ${PASSK_BACKEND} \
    --passk_samples ${PASSK_SAMPLES} --passk_every_n_vals ${PASSK_EVERY_N_VALS} \
    --output ${out} ${resume}"

  echo "[gpu${gpu}] RUN: ${name}  (num_roots=$(( ROLLOUT_LEN / n )), cost ~$(( (ROLLOUT_LEN / n) * m ))x baseline)"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "         CUDA_VISIBLE_DEVICES=${gpu} ${cmd}" | tr -s ' '
  else
    # Blocking on purpose: this call must finish before gpu_worker's loop moves
    # to the next job on THIS gpu. Parallelism across GPUs comes from
    # backgrounding gpu_worker itself below, not from backgrounding this line.
    CUDA_VISIBLE_DEVICES="${gpu}" bash -c "${cmd}" >> "${log}" 2>&1
  fi
}

# ---- one worker queue per GPU (sequential within a GPU, parallel across) -----
IFS=',' read -r -a GPU_ARR <<< "${GPUS:-0}"
NGPU=${#GPU_ARR[@]}
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
echo "[sweep] STAGE=${STAGE} finished."
