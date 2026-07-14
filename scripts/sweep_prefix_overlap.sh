#!/usr/bin/env bash
# =============================================================================
# Prefix-overlap / JSD hyperparameter sweep (Qwen3-0.6B draft / Qwen3-8B teacher)
# Rahul's spec (2026-07): sweep LR x warmup% x cosine-floor x weight-decay for
# methods {jsd, prefix_overlap prob}, 5k steps, 0.6B/8B, with in-training pass@k.
#
# Multi-machine + multi-GPU: run the SAME command on each machine with a
# different MACHINE_IDX; jobs are sharded across machines (round-robin by global
# job index) and, within a machine, round-robined across local GPUS (one
# sequential worker queue per GPU). Resumable: a run whose ckpt_best exists is
# skipped; one with ckpt_latest is --resume'd.
#
# USAGE
#   # machine 0 of 3, using local GPUs 0-3:
#   OUT does not apply here — CKPT_ROOT is fixed below; just set the shard:
#   NUM_MACHINES=3 MACHINE_IDX=0 GPUS=0,1,2,3 bash scripts/sweep_prefix_overlap.sh
#   NUM_MACHINES=3 MACHINE_IDX=1 GPUS=0,1,2,3 bash scripts/sweep_prefix_overlap.sh   # on machine 1
#   NUM_MACHINES=3 MACHINE_IDX=2 GPUS=0,1     bash scripts/sweep_prefix_overlap.sh   # on machine 2
#
#   DRY_RUN=1 ... bash scripts/sweep_prefix_overlap.sh     # print the plan, run nothing
# =============================================================================
set -uo pipefail

# ---- fixed paths / models (edit here) ----------------------------------------
CKPT_ROOT="/sensei-fs-3/users/rkrishna/checkpoints/Qwen0.6B_Qwen8B/prefix_overlap"
DRAFT="Qwen/Qwen3-0.6B"
TEACHER="Qwen/Qwen3-8B"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root (this script is in scripts/)
PY="${PY:-python}"

# ---- run config --------------------------------------------------------------
STEPS="${STEPS:-5000}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"          # must match train.py's GRAD_ACCUM (for warmup% math)
SEED="${SEED:-123}"
TRAIN_DS="${TRAIN_DS:-math_hard}"
VAL_DS="${VAL_DS:-math_val}"

# in-training pass@k. NOTE: math_hard is the TRAIN set — pass@k there measures
# memorisation, so we use math_eval (held-out). Cost is high (samples x prompts x
# #datasets per val); for a 180-run sweep consider PASSK_DATASETS=math_eval only,
# or "" to disable during the broad sweep and run full pass@k on finalists.
PASSK_DATASETS="${PASSK_DATASETS:-math_eval,gsm8k_eval,olympiad_eval}"
PASSK_SAMPLES="${PASSK_SAMPLES:-64}"
PASSK_TEMP="${PASSK_TEMP:-0.8}"        # 0.8 (not val_temp 0.2) — pass@k needs diversity
PASSK_N_PROMPTS="${PASSK_N_PROMPTS:-100}"

# ---- sweep grid (edit / comment to stage; e.g. LR-first then expand) ---------
LRS=(1e-5 1e-4 3e-4 3e-3 1e-2)
WARMUP_PCTS=(5 10 20)
LR_MINS=(0.1 0.01)
WDS=(0.001 0.01 0.1)
METHODS=(jsd prob)                      # prob => prefix_overlap --prefix_objective prob

# ---- sharding ----------------------------------------------------------------
NUM_MACHINES="${NUM_MACHINES:-1}"
MACHINE_IDX="${MACHINE_IDX:-0}"
IFS=',' read -r -a GPU_ARR <<< "${GPUS:-0}"
NGPU=${#GPU_ARR[@]}
DRY_RUN="${DRY_RUN:-0}"

# ---- build the full job list -------------------------------------------------
WU_TOTAL=$(( STEPS / GRAD_ACCUM ))       # total optimizer steps (warmup% is of this)
JOBS=()
for m in "${METHODS[@]}"; do
  for lr in "${LRS[@]}"; do
    for pct in "${WARMUP_PCTS[@]}"; do
      for lrmin in "${LR_MINS[@]}"; do
        for wd in "${WDS[@]}"; do
          JOBS+=("${m}|${lr}|${pct}|${lrmin}|${wd}")
        done
      done
    done
  done
done
echo "[sweep] total jobs = ${#JOBS[@]}  (machines=${NUM_MACHINES} idx=${MACHINE_IDX} gpus=${GPUS:-0})"
echo "[sweep] ckpt root  = ${CKPT_ROOT}"
mkdir -p "${CKPT_ROOT}/logs"

# ---- one job -----------------------------------------------------------------
run_job() {   # $1 = gpu id, $2 = job spec "m|lr|pct|lrmin|wd"
  local gpu="$1" spec="$2"
  IFS='|' read -r m lr pct lrmin wd <<< "$spec"
  local wu=$(( WU_TOTAL * pct / 100 )); (( wu < 1 )) && wu=1
  local loss_args name
  if [[ "$m" == "prob" ]]; then
    loss_args="--loss prefix_overlap --prefix_objective prob"
    name="po_prob_lr${lr}_wu${pct}_lrmin${lrmin}_wd${wd}"
  else
    loss_args="--loss jsd"
    name="jsd_lr${lr}_wu${pct}_lrmin${lrmin}_wd${wd}"
  fi
  local out="${CKPT_ROOT}/${name}"
  local log="${CKPT_ROOT}/logs/${name}.out"

  if [[ -e "${out}/ckpt_best" ]]; then
    echo "[gpu${gpu}] SKIP (done): ${name}"; return
  fi
  local resume=""
  [[ -e "${out}/ckpt_latest" ]] && resume="--resume" && echo "[gpu${gpu}] RESUME: ${name}"

  local cmd="${PY} ${REPO}/train.py ${loss_args} \
    --draft ${DRAFT} --teacher ${TEACHER} \
    --train_dataset ${TRAIN_DS} --val_dataset ${VAL_DS} \
    --steps ${STEPS} --seed ${SEED} \
    --lr ${lr} --warmup_steps ${wu} --lr_min_ratio ${lrmin} --weight_decay ${wd} \
    --passk_datasets ${PASSK_DATASETS} --passk_samples ${PASSK_SAMPLES} \
    --passk_temp ${PASSK_TEMP} --passk_n_prompts ${PASSK_N_PROMPTS} \
    --output ${out} ${resume}"

  echo "[gpu${gpu}] RUN: ${name}"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "         CUDA_VISIBLE_DEVICES=${gpu} ${cmd}" | tr -s ' '
  else
    CUDA_VISIBLE_DEVICES="${gpu}" bash -c "${cmd}" >> "${log}" 2>&1
  fi
}

# ---- per-GPU worker: sequentially runs its slice of this machine's jobs -------
gpu_worker() {   # $1 = local gpu-array index
  local gi="$1" gpu="${GPU_ARR[$1]}"
  local j
  for (( j=0; j<${#JOBS[@]}; j++ )); do
    # machine shard, then GPU shard within the machine
    (( j % NUM_MACHINES == MACHINE_IDX )) || continue
    local local_idx=$(( (j - MACHINE_IDX) / NUM_MACHINES ))
    (( local_idx % NGPU == gi )) || continue
    run_job "${gpu}" "${JOBS[$j]}"
  done
  echo "[gpu${gpu}] worker done."
}

for (( gi=0; gi<NGPU; gi++ )); do
  gpu_worker "$gi" &
done
wait
echo "[sweep] machine ${MACHINE_IDX} finished all its jobs."
