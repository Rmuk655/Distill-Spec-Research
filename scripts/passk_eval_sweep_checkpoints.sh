#!/usr/bin/env bash
# =============================================================================
# Batch pass@k eval over every FINISHED run under a sweep's checkpoint root,
# on the 3 held-out eval sets (math_eval, gsm8k_eval, olympiad_eval) --
# Rahul's "pass@k after training, for every hyperparam choice" ask.
#
# "FINISHED" means ckpt_latest/state.json's saved step == that SAME run's own
# recorded --steps target (see the train_args.steps note below) -- runs that
# were killed early, are still in progress, or early-stopped before reaching
# their full schedule are SKIPPED, not evaluated. We don't want pass@k on a
# run that never got a fair shot at converging, and we already know from
# block_eff/loss alone which runs catastrophically collapsed -- this script
# doesn't re-litigate that, it only fills the actual gap: held-out-set pass@k
# for runs Rahul asked to see it on.
#
# Evaluates ckpt_best (the val-selected checkpoint) for each finished run, NOT
# ckpt_latest (ckpt_latest's step count is only used as the completion gate).
#
# USAGE (from venv-vllm, see scripts/setup_vllm_env.sh):
#   # mode 1: every run under a single sweep root (auto-glob CKPT_ROOT/*/)
#   CKPT_ROOT=$USER_HOME/checkpoints/Qwen0.6B_Qwen8B/prefix_overlap \
#     GPUS=3 bash scripts/passk_eval_sweep_checkpoints.sh
#   DRY_RUN=1 CKPT_ROOT=... GPUS=3 bash scripts/passk_eval_sweep_checkpoints.sh   # preview only
#   # set DRAFT_MODEL if the sweep isn't the default 0.6B/8B pair, e.g.:
#   DRAFT_MODEL=Qwen/Qwen3-1.7B CKPT_ROOT=... GPUS=3 bash scripts/passk_eval_sweep_checkpoints.sh
#
#   # mode 2: an explicit list of run dir names, scattered directly under
#   # CKPT_ROOT (e.g. older one-off experiments, not one common sweep root) --
#   # RUN_NAMES overrides the CKPT_ROOT/*/ auto-glob entirely.
#   CKPT_ROOT=$USER_HOME/checkpoints \
#     RUN_NAMES=po_mh_prob_ce_lr1e5,po_mh_prob_lr1e5,po_mh_prob_lr5e6,po_mh_prob_warm_anneal_lr1e5,po_mh_logprob_N2_lr1e5,jsd_mathhard_s123,jsd_flat_enrich_K3_mathhard_s123,jsd_flat_enrich_K4_math_hard_s123 \
#     GPUS=3 bash scripts/passk_eval_sweep_checkpoints.sh
#
# "FINISHED" is checked per-run against that run's OWN recorded --steps (read
# from ckpt_latest/state.json's train_args.steps), NOT a single hardcoded
# number -- historical one-off runs don't all share the same --steps target
# (e.g. po_mh_prob_warm_anneal_lr1e5 used --steps 15000, not 5000). Falls back
# to --expected_steps only for very old checkpoints saved before train_args
# was recorded in state.json.
#
# NOTE: the finished-check above is SKIPPED ENTIRELY in RUN_NAMES mode (mode
# 2) -- you already know which runs you want, and not every old run has a
# ckpt_latest to check (some only have ckpt_best + wandb_run.json). Only
# ckpt_best's existence is verified there. The finished-check still applies
# in auto-glob mode (mode 1), since you don't already know what's in there.
#
# Resumable: skips (model, dataset) pairs already in the output CSV (matched
# on the ckpt dir path), so re-running after later sweep stages finish only
# evaluates the newly-completed runs, not everything again.
# =============================================================================
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-python -u}"
CKPT_ROOT="${CKPT_ROOT:?set CKPT_ROOT to the sweep checkpoint root dir (or the shared parent dir in RUN_NAMES mode)}"
EXPECTED_STEPS="${EXPECTED_STEPS:-5000}"    # fallback only -- see train_args.steps note above
RUN_NAMES_STR="${RUN_NAMES:-}"              # if set, overrides the CKPT_ROOT/*/ auto-glob
DATASETS_STR="${DATASETS:-math_eval,gsm8k_eval,olympiad_eval}"
IFS=',' read -r -a DATASETS <<< "${DATASETS_STR}"

N="${N:-64}"
N_PROMPTS="${N_PROMPTS:-100}"
PASSK_TEMP="${PASSK_TEMP:-0.8}"           # not TEMP -- collides with OS $TEMP on some shells
# checkpointing.py's save_checkpoint() only writes model.safetensors/config.json/
# optim.pt/state.json -- no tokenizer files. Loading ckpt_best as --model with no
# --tokenizer makes vLLM fall back to a degenerate tokenizer that encodes every
# prompt to 0 tokens ("decoder prompt cannot be empty"), regardless of dataset
# content. The tokenizer never changes across training -- always re-derive it
# from the base draft model id.
DRAFT_MODEL="${DRAFT_MODEL:-Qwen/Qwen3-0.6B}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
OUT_CSV="${OUT_CSV:-${REPO}/results/passK/passk_sweep_checkpoints.csv}"
LOG_DIR="${REPO}/results/passK/logs_passk_sweep_checkpoints"
mkdir -p "$(dirname "${OUT_CSV}")" "${LOG_DIR}"

IFS=',' read -r -a GPU_ARR <<< "${GPUS:-0}"
NGPU=${#GPU_ARR[@]}
DRY_RUN="${DRY_RUN:-0}"

# ---- candidate run list: explicit RUN_NAMES, or auto-glob CKPT_ROOT/*/ -----
if [[ -n "${RUN_NAMES_STR}" ]]; then
    IFS=',' read -r -a CANDIDATES <<< "${RUN_NAMES_STR}"
else
    CANDIDATES=()
    for _run_dir in "${CKPT_ROOT}"/*/; do
        CANDIDATES+=("$(basename "${_run_dir}")")
    done
fi

# ---- filter to FINISHED runs (each checked against its OWN --steps) --------
# STAT_TIMEOUT guards against a stale NFS handle on CKPT_ROOT (shared network
# filesystem) hanging the ENTIRE discovery loop indefinitely on one bad path,
# with zero visible progress and nothing to tell you which run caused it.
# `stat`/`python` are real external processes `timeout` can actually kill --
# bash's own `[[ -f ]]`/`[[ -d ]]` builtins run in-process and can't be
# pre-empted this way, hence using `stat` here instead of `[[ ]]` directly.
#
# SKIPPED ENTIRELY in RUN_NAMES mode: you already know exactly which runs you
# want evaluated, so the "did this finish" safety net (which needs
# ckpt_latest/state.json -- not every old run has one, e.g. some only have
# ckpt_best + wandb_run.json) would just silently drop runs you explicitly
# asked for. Only checks ckpt_best actually exists (that's what gets
# evaluated) in this mode. The auto-glob mode (CKPT_ROOT/*/, no RUN_NAMES)
# keeps the full finished-check, since there you don't already know what's
# in there or whether it's done.
STAT_TIMEOUT="${STAT_TIMEOUT:-10}"
FINISHED=()
for _run_name in "${CANDIDATES[@]}"; do
    _run_dir="${CKPT_ROOT}/${_run_name}/"
    _best="${_run_dir}ckpt_best"
    if [[ -n "${RUN_NAMES_STR}" ]]; then
        if ! timeout "${STAT_TIMEOUT}" stat "${_best}" >/dev/null 2>&1; then
            echo "[passk-sweep] SKIP ${_run_name}: no ckpt_best, or timed out after ${STAT_TIMEOUT}s"
            continue
        fi
        FINISHED+=("${_run_name}")
        continue
    fi
    _state="${_run_dir}ckpt_latest/state.json"
    if ! timeout "${STAT_TIMEOUT}" stat "${_state}" >/dev/null 2>&1; then
        echo "[passk-sweep] SKIP ${_run_name}: no ckpt_latest/state.json, or ${_run_dir} timed out after ${STAT_TIMEOUT}s (stale NFS handle?)"
        continue
    fi
    if ! timeout "${STAT_TIMEOUT}" stat "${_best}" >/dev/null 2>&1; then
        echo "[passk-sweep] SKIP ${_run_name}: no ckpt_best (never improved past init), or timed out after ${STAT_TIMEOUT}s"
        continue
    fi
    read -r _step _target <<< "$(timeout "${STAT_TIMEOUT}" python -c '
import json, sys
s = json.load(open(sys.argv[1]))
print(s.get("step", 0), s.get("train_args", {}).get("steps", sys.argv[2]))
' "${_state}" "${EXPECTED_STEPS}" 2>/dev/null || echo "0 ${EXPECTED_STEPS}")"
    if [[ "${_step}" != "${_target}" ]]; then
        echo "[passk-sweep] SKIP ${_run_name}: step=${_step} != this run's own target ${_target} (killed early / still running / early-stopped)"
        continue
    fi
    FINISHED+=("${_run_name}")
done
echo "[passk-sweep] ${#FINISHED[@]} finished run(s) out of ${#CANDIDATES[@]} candidate(s)"

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
        --model ${model_dir} --tokenizer ${DRAFT_MODEL} \
        --dataset ${REPO}/data_io/raw/${ds}.jsonl \
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
