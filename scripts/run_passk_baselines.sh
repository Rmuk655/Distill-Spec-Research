#!/usr/bin/env bash
# =============================================================================
# Standalone vLLM pass@k BASELINE runner (Rahul's "before training" ask).
# Runs pass@k for {draft, teacher} x {math_eval, gsm8k_eval, olympiad_eval} =
# 6 jobs. Fully independent of train.py / the training sweep / any checkpoint
# state — safe to run on a DIFFERENT box.
#
# WHAT THE NEW BOX NEEDS
#   - This repo cloned (or at minimum scripts/passk_eval.py + passk_utils.py,
#     but cloning the repo is the easy path since it also gives you the
#     dataset-download tooling below).
#   - vLLM in a SEPARATE venv from training: `bash scripts/setup_vllm_env.sh`
#     then `source venv-vllm/bin/activate` before running this script.
#     (vllm==0.25.1 requires torch==2.11.0, which conflicts with
#     requirements.txt's torch==2.12.0 pin for the training venv — installing
#     vllm into that same venv silently downgrades torch and breaks it.)
#   - The 3 eval datasets present under data_io/raw/ — if not already there:
#       python -m data_io.download --datasets math_eval,gsm8k_eval,olympiad_eval
#
# USAGE
#   GPUS=0,1,2,3 bash scripts/run_passk_baselines.sh
#   DRY_RUN=1 GPUS=0,1,2,3 bash scripts/run_passk_baselines.sh   # preview only
#   # MODELS overrides the default 0.6B/8B pair, e.g. a single large model
#   # sequentially on one GPU (GPUS with one entry = one worker, jobs run
#   # one at a time -- important for a 32B-scale model's memory footprint):
#   MODELS=Qwen/Qwen3-32B GPUS=3 bash scripts/run_passk_baselines.sh
#
# Resumable: each (model,dataset) job writes a marker file on success; rerunning
# the same command skips anything already done instead of duplicating CSV rows.
# =============================================================================
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-python -u}"   # -u = unbuffered stdout, so `tail -f` on the redirected log shows
                        # progress live (see sweep_prefix_overlap.sh for why)
OUT_CSV="${OUT_CSV:-${REPO}/results/passk_baselines.csv}"
MARKER_DIR="${REPO}/results/.passk_baseline_done"
LOG_DIR="${REPO}/results/logs_passk_baselines"
mkdir -p "$(dirname "$OUT_CSV")" "$MARKER_DIR" "$LOG_DIR"

MODELS_STR="${MODELS:-Qwen/Qwen3-0.6B,Qwen/Qwen3-8B}"   # comma list; default draft+teacher (0.6B/8B pair)
IFS=',' read -r -a MODELS <<< "${MODELS_STR}"
DATASETS=(math_eval gsm8k_eval olympiad_eval)  # held-out eval sets (not math_hard/math_val)

# Same defaults as the in-training pass@k hook, for a fair before/after compare.
N="${N:-64}"
N_PROMPTS="${N_PROMPTS:-100}"
PASSK_TEMP="${PASSK_TEMP:-0.8}"   # NOT named TEMP -- collides with the pre-set OS $TEMP/$TMP on some shells
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"   # lower this first if you OOM on a smaller GPU
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"

IFS=',' read -r -a GPU_ARR <<< "${GPUS:-0}"
NGPU=${#GPU_ARR[@]}
DRY_RUN="${DRY_RUN:-0}"

JOBS=()
for m in "${MODELS[@]}"; do
  for d in "${DATASETS[@]}"; do
    JOBS+=("${m}|${d}")
  done
done
echo "[passk-baselines] jobs=${#JOBS[@]}  gpus=${GPUS:-0}  n=${N} n_prompts=${N_PROMPTS} temp=${PASSK_TEMP}"
echo "[passk-baselines] out=${OUT_CSV}"

run_job() {
  local gpu="$1" spec="$2"
  IFS='|' read -r model ds <<< "$spec"
  local safe_model="${model//\//_}"
  local marker="${MARKER_DIR}/${safe_model}__${ds}.done"
  local log="${LOG_DIR}/${safe_model}__${ds}.out"

  if [[ -e "$marker" ]]; then echo "[gpu${gpu}] SKIP done: ${model} / ${ds}"; return; fi

  local cmd="${PY} ${REPO}/scripts/passk_eval.py \
    --model ${model} --dataset ${REPO}/data_io/raw/${ds}.jsonl \
    --n ${N} --n_prompts ${N_PROMPTS} --temp ${PASSK_TEMP} \
    --gpu_memory_utilization ${GPU_MEM_UTIL} --max_model_len ${MAX_MODEL_LEN} \
    --out ${OUT_CSV}"

  echo "[gpu${gpu}] RUN: ${model} / ${ds}"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "         CUDA_VISIBLE_DEVICES=${gpu} ${cmd}" | tr -s ' '
  else
    if CUDA_VISIBLE_DEVICES="${gpu}" bash -c "${cmd}" >> "${log}" 2>&1; then
      touch "$marker"
    else
      echo "[gpu${gpu}] FAILED: ${model} / ${ds} — see ${log}"
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
for (( gi=0; gi<NGPU; gi++ )); do gpu_worker "$gi" & done
wait
echo "[passk-baselines] all done. Results in ${OUT_CSV}"
