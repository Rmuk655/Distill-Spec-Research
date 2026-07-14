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
#   - `pip install vllm` (not in requirements.txt — vLLM has its own heavy,
#     platform-specific deps; install it separately).
#   - The 3 eval datasets present under data_io/raw/ — if not already there:
#       python -m data_io.download --datasets math_eval,gsm8k_eval,olympiad_eval
#
# USAGE
#   GPUS=0,1,2,3 bash scripts/run_passk_baselines.sh
#   DRY_RUN=1 GPUS=0,1,2,3 bash scripts/run_passk_baselines.sh   # preview only
#
# Resumable: each (model,dataset) job writes a marker file on success; rerunning
# the same command skips anything already done instead of duplicating CSV rows.
# =============================================================================
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-python}"
OUT_CSV="${OUT_CSV:-${REPO}/results/passk_baselines.csv}"
MARKER_DIR="${REPO}/results/.passk_baseline_done"
LOG_DIR="${REPO}/results/logs_passk_baselines"
mkdir -p "$(dirname "$OUT_CSV")" "$MARKER_DIR" "$LOG_DIR"

MODELS=("Qwen/Qwen3-0.6B" "Qwen/Qwen3-8B")     # draft (untrained), teacher
DATASETS=(math_eval gsm8k_eval olympiad_eval)  # held-out eval sets (not math_hard/math_val)

# Same defaults as the in-training pass@k hook, for a fair before/after compare.
N="${N:-64}"
N_PROMPTS="${N_PROMPTS:-100}"
TEMP="${TEMP:-0.8}"
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
echo "[passk-baselines] jobs=${#JOBS[@]}  gpus=${GPUS:-0}  n=${N} n_prompts=${N_PROMPTS} temp=${TEMP}"
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
    --n ${N} --n_prompts ${N_PROMPTS} --temp ${TEMP} \
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
