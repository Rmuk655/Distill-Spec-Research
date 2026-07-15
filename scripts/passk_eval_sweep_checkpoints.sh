#!/usr/bin/env bash
# =============================================================================
# Batch pass@k eval over every FINISHED run under a sweep's checkpoint root,
# on the 3 held-out eval sets (math_eval, gsm8k_eval, olympiad_eval) --
# Rahul's "pass@k after training, for every hyperparam choice" ask.
#
# "FINISHED" means ckpt_latest/state.json's saved step == --expected_steps
# (default 5000) -- runs that were killed early, are still in progress, or
# early-stopped before reaching the full schedule are SKIPPED, not evaluated.
# We don't want pass@k on a run that never got a fair shot at converging, and
# we already know from block_eff/loss alone which runs catastrophically
# collapsed -- this script doesn't re-litigate that, it only fills the actual
# gap: held-out-set pass@k for runs Rahul asked to see it on.
#
# Evaluates ckpt_best (the val-selected checkpoint) for each finished run, NOT
# ckpt_latest (ckpt_latest's step count is only used as the completion gate).
#
# USAGE (from venv-vllm, see scripts/setup_vllm_env.sh):
#   CKPT_ROOT=/sensei-fs-3/users/rkrishna/checkpoints/Qwen0.6B_Qwen8B/prefix_overlap \
#     GPUS=3 bash scripts/passk_eval_sweep_checkpoints.sh
#   DRY_RUN=1 CKPT_ROOT=... GPUS=3 bash scripts/passk_eval_sweep_checkpoints.sh   # preview only
#
# Resumable: skips (model, dataset) pairs already in the output CSV (matched
# on the ckpt dir path), so re-running after later sweep stages finish only
# evaluates the newly-completed runs, not everything again.
# =============================================================================
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-python -u}"
CKPT_ROOT="${CKPT_ROOT:?set CKPT_ROOT to the sweep's checkpoint root dir}"
EXPECTED_STEPS="${EXPECTED_STEPS:-5000}"
DATASETS_STR="${DATASETS:-math_eval,gsm8k_eval,olympiad_eval}"
IFS=',' read -r -a DATASETS <<< "${DATASETS_STR}"

N="${N:-64}"
N_PROMPTS="${N_PROMPTS:-100}"
PASSK_TEMP="${PASSK_TEMP:-0.8}"           # not TEMP -- collides with OS $TEMP on some shells
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
OUT_CSV="${OUT_CSV:-${REPO}/results/passk_sweep_checkpoints.csv}"
LOG_DIR="${REPO}/results/logs_passk_sweep_checkpoints"
mkdir -p "$(dirname "${OUT_CSV}")" "${LOG_DIR}"

IFS=',' read -r -a GPU_ARR <<< "${GPUS:-0}"
NGPU=${#GPU_ARR[@]}
DRY_RUN="${DRY_RUN:-0}"

# ---- discover finished runs -------------------------------------------------
FINISHED=()
for _run_dir in "${CKPT_ROOT}"/*/; do
    _run_name="$(basename "${_run_dir}")"
    _state="${_run_dir}ckpt_latest/state.json"
    _best="${_run_dir}ckpt_best"
    if [[ ! -f "${_state}" ]]; then
        echo "[passk-sweep] SKIP ${_run_name}: no ckpt_latest/state.json (never checkpointed)"
        continue
    fi
    if [[ ! -d "${_best}" ]]; then
        echo "[passk-sweep] SKIP ${_run_name}: no ckpt_best (never improved past init)"
        continue
    fi
    _step=$(python -c "import json,sys; print(json.load(open(sys.argv[1])).get('step', 0))" "${_state}" 2>/dev/null || echo 0)
    if [[ "${_step}" != "${EXPECTED_STEPS}" ]]; then
        echo "[passk-sweep] SKIP ${_run_name}: step=${_step} != ${EXPECTED_STEPS} (killed early / still running / early-stopped)"
        continue
    fi
    FINISHED+=("${_run_name}")
done
echo "[passk-sweep] ${#FINISHED[@]} finished run(s) out of $(ls -d "${CKPT_ROOT}"/*/ 2>/dev/null | wc -l) total under ${CKPT_ROOT}"

# ---- build (run, dataset) job list, skipping ones already in OUT_CSV -------
# passk_eval.py's CSV columns are model,dataset,n,temp,n_graded,n_skipped,k,pass_at_k
# -- "model" is the full ckpt_best path, already a unique key per run, and
# "dataset" is os.path.basename(--dataset) i.e. "<ds>.jsonl". No separate
# run-name column needed or written.
JOBS=()
for _run_name in "${FINISHED[@]}"; do
    _model_dir="${CKPT_ROOT}/${_run_name}/ckpt_best"
    for _ds in "${DATASETS[@]}"; do
        if [[ -f "${OUT_CSV}" ]] && grep -qF "${_model_dir},${_ds}.jsonl," "${OUT_CSV}"; then
            continue   # already evaluated in a prior invocation
        fi
        JOBS+=("${_run_name}|${_ds}")
    done
done
echo "[passk-sweep] jobs=${#JOBS[@]}  gpus=${GPUS:-0}  n=${N} n_prompts=${N_PROMPTS} temp=${PASSK_TEMP}"
echo "[passk-sweep] out=${OUT_CSV}"

run_job() {
    local gpu="$1" spec="$2"
    IFS='|' read -r run_name ds <<< "${spec}"
    local model_dir="${CKPT_ROOT}/${run_name}/ckpt_best"
    local log="${LOG_DIR}/${run_name}__${ds}.out"
    local cmd="${PY} ${REPO}/scripts/passk_eval.py \
        --model ${model_dir} --dataset ${REPO}/data_io/raw/${ds}.jsonl \
        --n ${N} --n_prompts ${N_PROMPTS} --temp ${PASSK_TEMP} \
        --gpu_memory_utilization ${GPU_MEM_UTIL} --max_model_len ${MAX_MODEL_LEN} \
        --out ${OUT_CSV}"
    echo "[gpu${gpu}] RUN: ${run_name} / ${ds}"
    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "         CUDA_VISIBLE_DEVICES=${gpu} ${cmd}" | tr -s ' '
    else
        if ! CUDA_VISIBLE_DEVICES="${gpu}" bash -c "${cmd}" >> "${log}" 2>&1; then
            echo "[gpu${gpu}] FAILED: ${run_name} / ${ds} — see ${log}"
        fi
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
for (( gi=0; gi<NGPU; gi++ )); do gpu_worker "${gi}" & done
wait
echo "[passk-sweep] all done. Results in ${OUT_CSV}"
