#!/usr/bin/env bash
# =============================================================================
# STAGED (coordinate-descent) hyperparameter sweep — Qwen3-0.6B / Qwen3-8B.
# Instead of the full 180-run cross-product, sweep ONE axis at a time, lock in
# the winner, then sweep the next axis around it. ~20 runs total vs 180.
#
# Axes (Rahul 2026-07): LR, warmup%, cosine-floor (lr_min), weight-decay,
# for methods {jsd, prefix_overlap prob}, 5k steps, 0.6B/8B, in-training pass@k.
#
# WORKFLOW
#   1) STAGE=lr      GPUS=0,1,2,3 bash scripts/sweep_prefix_overlap.sh     # 5 LRs x2 methods
#      -> eval (block_eff + pass@k), pick the best LR, e.g. 3e-4
#   2) STAGE=warmup  LOCK_LR=3e-4 GPUS=... bash scripts/sweep_prefix_overlap.sh
#      -> pick best warmup%, e.g. 10
#   3) STAGE=lr_min  LOCK_LR=3e-4 LOCK_WARMUP=10 GPUS=... bash ...
#      -> pick best lr_min, e.g. 0.1
#   4) STAGE=wd      LOCK_LR=3e-4 LOCK_WARMUP=10 LOCK_LRMIN=0.1 GPUS=... bash ...
#
#   Overlapping points (e.g. the baseline config that recurs across stages) are
#   auto-skipped (ckpt_best exists), so no run is repeated across stages.
#
#   DRY_RUN=1 STAGE=lr ... bash ...   # print the plan, run nothing
#   Multi-machine: same command per machine with NUM_MACHINES + a distinct MACHINE_IDX.
# =============================================================================
set -uo pipefail

# ---- fixed paths / models ----------------------------------------------------
CKPT_ROOT="/sensei-fs-3/users/rkrishna/checkpoints/Qwen0.6B_Qwen8B/prefix_overlap"
DRAFT="Qwen/Qwen3-0.6B"
TEACHER="Qwen/Qwen3-8B"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-python -u}"   # -u = unbuffered stdout, so `tail -f` on the redirected log shows
                        # progress live instead of waiting for Python's block-buffer to fill
                        # (stdout defaults to full buffering, not line buffering, when it's
                        # redirected to a file rather than a TTY)

# ---- run config --------------------------------------------------------------
STEPS="${STEPS:-5000}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"          # must match train.py GRAD_ACCUM (for warmup% math)
SEED="${SEED:-123}"
TRAIN_DS="${TRAIN_DS:-math_hard}"
VAL_DS="${VAL_DS:-math_val}"
METHODS_STR="${METHODS:-jsd,prob}"     # comma list; prob => prefix_overlap --prefix_objective prob
IFS=',' read -r -a METHODS <<< "${METHODS_STR}"

# in-training pass@k: math_val ONLY (the dev set — same 100 prompts block-eff uses,
# so no test-peeking). The multi-dataset pass@k (math_eval/gsm8k_eval/olympiad_eval)
# is a SEPARATE post-training run via scripts/passk_eval.py, not during the sweep.
#
# TWO-TIER cadence (measured 2026-07-14: plain block-eff val ~15min, val+full
# pass@k(n=64,100prompts) ~55min under 2-jobs/GPU -> pass@k marginal ~40min):
#   routine tier: n=16 (pass@1/2/4/8/16), EVERY val check -- cheap
#   full tier:    n=64 (adds pass@32/64), every PASSK_FULL_EVERY_N_VALS-th val check
#                 (a val-check MULTIPLIER, not a step count -- always an exact
#                 multiple of whatever val_every is, nothing to misalign if it changes)
PASSK_DATASETS="${PASSK_DATASETS:-math_val}"
PASSK_SAMPLES="${PASSK_SAMPLES:-16}"                       # routine tier n
PASSK_TEMP="${PASSK_TEMP:-0.8}"
PASSK_N_PROMPTS="${PASSK_N_PROMPTS:-100}"                  # routine tier n_prompts
PASSK_MAX_NEW_TOKENS="${PASSK_MAX_NEW_TOKENS:-512}"        # must reach the answer; 128 (block-eff) truncates
PASSK_FULL_SAMPLES="${PASSK_FULL_SAMPLES:-64}"             # full tier n
PASSK_FULL_EVERY_N_VALS="${PASSK_FULL_EVERY_N_VALS:-4}"    # every 4th val check (= every 1600 steps @val_every=400)
PASSK_FULL_N_PROMPTS="${PASSK_FULL_N_PROMPTS:-100}"        # full tier n_prompts (same as routine by default)

# ---- staged sweep: which axis + locked (winner) values for the others --------
STAGE="${STAGE:?set STAGE=lr|warmup|lr_min|wd}"
LOCK_LR="${LOCK_LR:-1e-4}"              # baseline until stage 'lr' picks a winner
LOCK_WARMUP="${LOCK_WARMUP:-10}"       # percent
LOCK_LRMIN="${LOCK_LRMIN:-0.1}"
LOCK_WD="${LOCK_WD:-0.01}"

# axis value lists
LRS=(1e-5 1e-4 3e-4 3e-3 1e-2)
WARMUP_PCTS=(5 10 20)
LR_MINS=(0.1 0.01)
WDS=(0.001 0.01 0.1)

# ---- sharding ----------------------------------------------------------------
NUM_MACHINES="${NUM_MACHINES:-1}"
MACHINE_IDX="${MACHINE_IDX:-0}"
JOBS_PER_GPU="${JOBS_PER_GPU:-1}"   # >1 = pack multiple concurrent train.py per physical GPU.
                                    # UNMEASURED trade: val/verifier tree-construction is CPU-bound
                                    # (real idle-GPU-time to fill), but two jobs' matmuls also
                                    # contend for the same SMs, and VRAM headroom (~25-30GB/job est.,
                                    # unverified) could OOM. Pilot on ONE gpu (JOBS_PER_GPU=2 on a
                                    # single-GPU GPUS=... list) and watch nvidia-smi before trusting
                                    # it across the whole sweep.
IFS=',' read -r -a GPU_ARR <<< "${GPUS:-0}"
NGPU=${#GPU_ARR[@]}
TOTAL_SLOTS=$(( NGPU * JOBS_PER_GPU ))   # virtual worker slots; JOBS_PER_GPU of them share each physical GPU
DRY_RUN="${DRY_RUN:-0}"

# ---- build the job list for THIS stage only ----------------------------------
WU_TOTAL=$(( STEPS / GRAD_ACCUM ))
JOBS=()
for m in "${METHODS[@]}"; do
  case "$STAGE" in
    lr)      for v in "${LRS[@]}";         do JOBS+=("${m}|${v}|${LOCK_WARMUP}|${LOCK_LRMIN}|${LOCK_WD}"); done ;;
    warmup)  for v in "${WARMUP_PCTS[@]}"; do JOBS+=("${m}|${LOCK_LR}|${v}|${LOCK_LRMIN}|${LOCK_WD}");   done ;;
    lr_min)  for v in "${LR_MINS[@]}";     do JOBS+=("${m}|${LOCK_LR}|${LOCK_WARMUP}|${v}|${LOCK_WD}");   done ;;
    wd)      for v in "${WDS[@]}";         do JOBS+=("${m}|${LOCK_LR}|${LOCK_WARMUP}|${LOCK_LRMIN}|${v}"); done ;;
    *) echo "bad STAGE='$STAGE' (want lr|warmup|lr_min|wd)"; exit 1 ;;
  esac
done
echo "[sweep] STAGE=${STAGE}  methods=${METHODS_STR}  jobs=${#JOBS[@]}  (locked: lr=${LOCK_LR} wu=${LOCK_WARMUP}% lrmin=${LOCK_LRMIN} wd=${LOCK_WD})"
echo "[sweep] machines=${NUM_MACHINES} idx=${MACHINE_IDX} gpus=${GPUS:-0}  jobs_per_gpu=${JOBS_PER_GPU}  slots=${TOTAL_SLOTS}  ckpt_root=${CKPT_ROOT}"
mkdir -p "${CKPT_ROOT}/logs"

# ---- one job -----------------------------------------------------------------
run_job() {   # $1 = gpu id, $2 = "m|lr|pct|lrmin|wd"
  local gpu="$1" spec="$2"
  IFS='|' read -r m lr pct lrmin wd <<< "$spec"
  local wu=$(( WU_TOTAL * pct / 100 )); (( wu < 1 )) && wu=1
  local loss_args name
  if [[ "$m" == "prob" ]]; then
    loss_args="--loss prefix_overlap --prefix_objective prob"; name="po_prob"
  else
    loss_args="--loss jsd"; name="jsd"
  fi
  name="${name}_lr${lr}_wu${pct}_lrmin${lrmin}_wd${wd}"
  local out="${CKPT_ROOT}/${name}" log="${CKPT_ROOT}/logs/${name}.out"

  if [[ -e "${out}/ckpt_best" ]]; then echo "[gpu${gpu}] SKIP done: ${name}"; return; fi
  local resume=""; [[ -e "${out}/ckpt_latest" ]] && resume="--resume" && echo "[gpu${gpu}] RESUME: ${name}"

  local cmd="${PY} ${REPO}/train.py ${loss_args} \
    --draft ${DRAFT} --teacher ${TEACHER} \
    --train_dataset ${TRAIN_DS} --val_dataset ${VAL_DS} \
    --steps ${STEPS} --seed ${SEED} \
    --lr ${lr} --warmup_steps ${wu} --lr_min_ratio ${lrmin} --weight_decay ${wd} \
    --passk_datasets ${PASSK_DATASETS} --passk_samples ${PASSK_SAMPLES} \
    --passk_temp ${PASSK_TEMP} --passk_n_prompts ${PASSK_N_PROMPTS} \
    --passk_max_new_tokens ${PASSK_MAX_NEW_TOKENS} \
    --passk_full_samples ${PASSK_FULL_SAMPLES} --passk_full_every_n_vals ${PASSK_FULL_EVERY_N_VALS} \
    --passk_full_n_prompts ${PASSK_FULL_N_PROMPTS} \
    --output ${out} ${resume}"

  echo "[gpu${gpu}] RUN: ${name}"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "         CUDA_VISIBLE_DEVICES=${gpu} ${cmd}" | tr -s ' '
  else
    CUDA_VISIBLE_DEVICES="${gpu}" bash -c "${cmd}" >> "${log}" 2>&1
  fi
}

# ---- virtual-slot worker queue (machine shard, then slot shard) -------------
# TOTAL_SLOTS = NGPU * JOBS_PER_GPU independent sequential queues. Slot `s`
# always runs on physical GPU GPU_ARR[s / JOBS_PER_GPU] — with JOBS_PER_GPU=1
# (default) this is identical to the old one-worker-per-GPU behaviour; with
# JOBS_PER_GPU=2, slots {0,1} both pin to GPU_ARR[0] and run CONCURRENTLY on
# it (two separate train.py processes sharing that GPU), slots {2,3} to
# GPU_ARR[1], etc.
gpu_worker() {
  local slot="$1" gpu="${GPU_ARR[$(( slot / JOBS_PER_GPU ))]}" j
  for (( j=0; j<${#JOBS[@]}; j++ )); do
    (( j % NUM_MACHINES == MACHINE_IDX )) || continue
    local local_idx=$(( (j - MACHINE_IDX) / NUM_MACHINES ))
    (( local_idx % TOTAL_SLOTS == slot )) || continue
    run_job "${gpu}" "${JOBS[$j]}"
  done
  echo "[gpu${gpu} slot${slot}] worker done."
}
for (( slot=0; slot<TOTAL_SLOTS; slot++ )); do gpu_worker "$slot" & done
wait
echo "[sweep] STAGE=${STAGE} on machine ${MACHINE_IDX} finished."
