#!/usr/bin/env bash
# =============================================================================
# One-line-per-run summary table over a directory of training logs, so you
# don't have to open W&B for every run to tell which approach worked.
#
# Reads each <run>.log next to its checkpoint dir under CKPT_ROOT and reports:
#   STATUS    finished (has a "[done]" line) / CRASHED (Traceback near EOF,
#             no "[done]") / running? (neither -- still going, or log stale)
#   BEST_BE   best_val_block_eff from the "[done]" summary line
#   BEST_SM   best_smoothed from the "[done]" summary line
#   FINAL_BE  block_eff from the LAST "[val]" line (whether finished or not --
#             this is what catches the "peaked early, then collapsed by the
#             end" pattern seen repeatedly this sweep, e.g. topk20@lr1e-4:
#             best=3.801 but final=1.277)
#   EASY/MED/HARD  prompt-difficulty split at the last val check -- e.g.
#             easy=0 med=0 hard=100 is the smoking gun for total collapse
#   FORGET    forgetting metric at the last val check
#   COLLAPSED? heuristic flag: final_be < 70% of best_be => "yes"
#
# USAGE:
#   bash scripts/summarize_sweep.sh /sensei-fs-3/users/rkrishna/checkpoints/Qwen0.6B_Qwen8B/prefix_overlap
#   bash scripts/summarize_sweep.sh <dir> | sort -t$'\t' -k3 -rn   # sort by best_block_eff
# =============================================================================
set -uo pipefail

ROOT="${1:?usage: summarize_sweep.sh <checkpoint_root_dir_containing_*.log>}"

printf "%-50s %-10s %-9s %-9s %-9s %-6s %-6s %-6s %-8s %-10s\n" \
  "RUN" "STATUS" "BEST_BE" "BEST_SM" "FINAL_BE" "EASY" "MED" "HARD" "FORGET" "COLLAPSED?"

for log in "${ROOT}"/*.log; do
  [[ -f "$log" ]] || continue
  run_name="$(basename "$log" .log)"

  if grep -q '\[done\]' "$log"; then
    status="finished"
  elif tail -50 "$log" 2>/dev/null | grep -q '^Traceback'; then
    status="CRASHED"
  else
    status="running?"
  fi

  done_line="$(grep '\[done\]' "$log" | tail -1)"
  best_be="$(echo "$done_line" | grep -oP 'best_val_block_eff = \K[0-9.]+')"
  best_sm="$(echo "$done_line" | grep -oP 'best_smoothed = \K[0-9.]+')"

  # last [val] line -- works regardless of finished/crashed/running status
  last_val="$(grep '\[val\] step=' "$log" | tail -1)"
  final_be="$(echo "$last_val" | grep -oP 'block_eff=\K[0-9.]+')"
  easy="$(echo "$last_val"     | grep -oP 'easy=\K[0-9]+')"
  med="$(echo "$last_val"      | grep -oP 'med=\K[0-9]+')"
  hard="$(echo "$last_val"     | grep -oP 'hard=\K[0-9]+')"
  forget="$(echo "$last_val"   | grep -oP 'forget=\K[0-9.]+')"

  collapsed="-"
  if [[ -n "${best_be:-}" && -n "${final_be:-}" ]]; then
    collapsed="$(awk -v b="$best_be" -v f="$final_be" 'BEGIN{print (f < 0.7*b) ? "yes" : "no"}')"
  fi

  printf "%-50s %-10s %-9s %-9s %-9s %-6s %-6s %-6s %-8s %-10s\n" \
    "$run_name" "$status" "${best_be:--}" "${best_sm:--}" "${final_be:--}" \
    "${easy:--}" "${med:--}" "${hard:--}" "${forget:--}" "$collapsed"
done
