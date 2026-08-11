#!/usr/bin/env bash
# migrate_ckpt_root.sh — move a checkpoint root off a full filesystem onto one
# with more available space, without deleting anything until sizes are verified
# to match.
#
# Usage:
#   OLD_ROOT=$USER_HOME/checkpoints/Qwen0.6B_Qwen8B/prefix_overlap \
#   NEW_ROOT=$USER_HOME/checkpoints/Qwen0.6B_Qwen8B/prefix_overlap \
#     bash scripts/migrate_ckpt_root.sh
set -euo pipefail

OLD_ROOT="${OLD_ROOT:?set OLD_ROOT to the full checkpoint dir on the source filesystem}"
NEW_ROOT="${NEW_ROOT:?set NEW_ROOT to the destination dir on the target filesystem}"

echo "[migrate] rsyncing ${OLD_ROOT}/ -> ${NEW_ROOT}/ (large checkpoints, this can take a while)"
mkdir -p "$NEW_ROOT"
rsync -avh --progress "${OLD_ROOT}/" "${NEW_ROOT}/"

echo "[migrate] verifying byte counts match before anything gets deleted"
old_size=$(du -sb "$OLD_ROOT" | cut -f1)
new_size=$(du -sb "$NEW_ROOT" | cut -f1)
echo "  old: ${old_size} bytes   new: ${new_size} bytes"

if [ "$old_size" != "$new_size" ]; then
  echo "[migrate] SIZE MISMATCH — NOT deleting anything. Re-run rsync (it's incremental)" >&2
  echo "          or investigate before touching ${OLD_ROOT}." >&2
  exit 1
fi

echo "[migrate] sizes match. Old copy is NOT deleted automatically — review, then free space with:"
echo "  rm -rf ${OLD_ROOT}"
echo
echo "[migrate] point future launches at the new root:"
echo "  export CKPT_ROOT=$(dirname "$NEW_ROOT")"
