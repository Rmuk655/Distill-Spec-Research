#!/usr/bin/env bash
# resume_crashed_runs.sh — resume runs killed by a disk-full crash, reading each
# run's OWN launch command from ckpt_latest/state.json's "cmd" field (train.py
# writes this on every checkpoint) instead of hand-reconstructing flags — avoids
# silently mismatching the LR schedule/warmup-step count on resume.
#
# Run this AFTER migrating checkpoints to the new root (scripts/migrate_ckpt_root.sh)
# and pointing CKPT_ROOT at the migrated location.
#
# Usage:
#   GPUS=0,1,2,3 CKPT_ROOT=$USER_HOME/checkpoints/Qwen0.6B_Qwen8B/prefix_overlap \
#     bash scripts/resume_crashed_runs.sh po_prob_ttemp0.5_lr1e-5_wu20 po_prob_ttemp0.7_lr1e-5_wu20 \
#          po_prob_gradclip10_lr1e-5_wu20 po_prob_M16_lr3e-6_wu20_continued
set -uo pipefail

CKPT_ROOT="${CKPT_ROOT:?set CKPT_ROOT to the migrated checkpoint root}"
REPO="${REPO:-$HOME/Distill-Spec-Research}"
IFS=',' read -r -a GPU_ARR <<< "${GPUS:-0}"

if [ "$#" -eq 0 ]; then
  echo "Usage: GPUS=0,1,2,3 CKPT_ROOT=... bash $0 <run_name> [<run_name> ...]"
  exit 1
fi

resume_one() {
  local gpu="$1" name="$2"
  local run_dir="${CKPT_ROOT}/${name}"
  local state="${run_dir}/ckpt_latest/state.json"

  if [ ! -f "$state" ]; then
    echo "[gpu${gpu}] SKIP ${name}: no ${state} found"
    return
  fi

  # Pull the exact argv this run was originally launched with, straight from its
  # own checkpoint metadata.
  local cmd_json
  cmd_json=$(python3 - "$state" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
cmd = d.get("cmd")
if not cmd:
    sys.exit(1)
print(json.dumps(cmd))
PYEOF
  ) || { echo "[gpu${gpu}] SKIP ${name}: state.json has no 'cmd' field"; return; }

  # Rebuild the arg list: point --output at the (possibly migrated) run_dir,
  # and add --resume if it isn't already present.
  local args
  args=$(python3 - "$cmd_json" "$run_dir" <<'PYEOF'
import json, sys, shlex
cmd = json.loads(sys.argv[1])
run_dir = sys.argv[2]
argv = cmd[1:]  # drop the script path itself (train.py)
out, i, seen_output = [], 0, False
while i < len(argv):
    a = argv[i]
    if a == "--output":
        out += ["--output", run_dir]
        seen_output = True
        i += 2
        continue
    out.append(a)
    i += 1
if not seen_output:
    out += ["--output", run_dir]
if "--resume" not in out:
    out.append("--resume")
print(" ".join(shlex.quote(x) for x in out))
PYEOF
  )

  local log="${run_dir}.log"
  echo "[gpu${gpu}] RESUME ${name}"
  echo "         CUDA_VISIBLE_DEVICES=${gpu} python -u ${REPO}/train.py ${args}" | tr -s ' '
  CUDA_VISIBLE_DEVICES="${gpu}" nohup python -u "${REPO}/train.py" ${args} >> "${log}" 2>&1 &
  disown
  echo "[gpu${gpu}] launched pid=$!"
}

i=0
for name in "$@"; do
  gpu="${GPU_ARR[$(( i % ${#GPU_ARR[@]} ))]}"
  resume_one "$gpu" "$name"
  i=$((i+1))
done

echo "[resume] done. tail -f each <run_dir>.log to confirm training actually resumed from a nonzero step."
