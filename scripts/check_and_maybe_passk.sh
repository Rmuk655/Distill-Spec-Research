#!/usr/bin/env bash
# =============================================================================
# Extracts best_val_block_eff from a finished training run's log and decides
# whether the result is interesting enough to spend GPU time on a pass@k eval
# for it -- compares against a REFERENCE value you supply (the closest
# existing comparable point), not an absolute bar.
#
# "Interesting" = new_be >= reference - MARGIN (default 0.02, i.e. beats OR
# comes within 0.02 BE of the reference -- close enough to be worth knowing
# pass@k for, given how tight this sweep's margins already are).
#
# USAGE:
#   bash scripts/check_and_maybe_passk.sh <run_name> <reference_be> [margin] [gpu]
#
#   # example: po_prob_lr2e-6_wu20 fills the gap between 1e-6(5.591)/3e-6(5.701)
#   bash scripts/check_and_maybe_passk.sh po_prob_lr2e-6_wu20 5.701 0.02 0
#
#   # example: the key combo test -- does anneal+aux0.5 beat 5.848 at lr=3e-6?
#   bash scripts/check_and_maybe_passk.sh po_prob_ceanneal3000_auxw0.5_lr3e-6_wu20 5.848 0.02 2
#
# If interesting: prints the exact passk_eval_sweep_checkpoints.sh RUN_NAMES
# command to launch on the given GPU (does NOT auto-launch -- review first).
# If not interesting: just reports the number and stops.
# =============================================================================
set -uo pipefail

RUN="${1:?usage: check_and_maybe_passk.sh <run_name> <reference_be> [margin] [gpu]}"
REF="${2:?need a reference best_be to compare against}"
MARGIN="${3:-0.02}"
GPU="${4:-0}"

CKPT_ROOT="$USER_HOME/checkpoints/Qwen0.6B_Qwen8B/prefix_overlap"
LOG="${CKPT_ROOT}/${RUN}.log"

if [[ ! -f "$LOG" ]]; then
    echo "[check] $LOG not found"
    exit 1
fi

DONE_LINE="$(grep '\[done\]' "$LOG" | tail -1)"
if [[ -z "$DONE_LINE" ]]; then
    echo "[check] $RUN: not finished yet (no [done] line). Still running -- rerun this once it completes."
    tail -3 "$LOG"
    exit 0
fi

BEST_BE="$(echo "$DONE_LINE" | grep -oP 'best_val_block_eff = \K[0-9.]+')"
echo "[check] $RUN: best_val_block_eff = ${BEST_BE}  (reference: ${REF}, margin: ${MARGIN})"

IS_INTERESTING="$(awk -v b="$BEST_BE" -v r="$REF" -v m="$MARGIN" 'BEGIN{print (b >= r - m) ? "yes" : "no"}')"

if [[ "$IS_INTERESTING" == "yes" ]]; then
    if awk -v b="$BEST_BE" -v r="$REF" 'BEGIN{exit !(b > r)}'; then
        echo "[check] >>> NEW BEST -- beats reference by $(awk -v b="$BEST_BE" -v r="$REF" 'BEGIN{printf "%.3f", b-r}')"
    else
        echo "[check] >>> within margin of reference (worth knowing pass@k)"
    fi
    echo
    echo "Run this to get pass@k for it:"
    echo
    echo "  source venv-vllm/bin/activate"
    echo "  CKPT_ROOT=${CKPT_ROOT} RUN_NAMES=${RUN} DRAFT_MODEL=Qwen/Qwen3-0.6B \\"
    echo "    GPUS=${GPU} OUT_CSV=results/passK/passk_hparam_sweep.csv \\"
    echo "    bash scripts/passk_eval_sweep_checkpoints.sh"
else
    echo "[check] below reference by more than ${MARGIN} -- not interesting enough to spend GPU time on pass@k."
fi
