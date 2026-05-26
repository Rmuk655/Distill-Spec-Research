#!/bin/bash
# Merge and evaluate the three EBE LR-sweep checkpoints.
# Run from OSD/ directory: bash bash_scripts/eval_ebe_sweep.sh
#
# Checkpoints trained from Qwen2.5-0.5B with EBE loss, different LRs.
# Results land in results.db and DELIVERABLES.md.

set -e
cd "$(dirname "$0")/.."

TEACHER="Qwen/Qwen3-0.6B"
EVAL_DATA="data/gsm8k_30.jsonl"
TEMPS="0.6,1.0"
KS="3,5"

for LR in "1e-5" "3e-5" "1e-4"; do
    CKPT="checkpoints/ebe_lr${LR}"
    MERGED="${CKPT}_merged"
    LABEL="ebe_lr${LR}"

    echo ""
    echo "=== Merging ${CKPT} ==="
    if [ ! -f "${MERGED}/config.json" ]; then
        python train_qwen3.py --merge_only --adapter "${CKPT}"
        echo "  Merged → ${MERGED}"
    else
        echo "  Already merged, skipping"
    fi

    echo ""
    echo "=== Evaluating ${LABEL} ==="
    python run_all.py \
        --student "${MERGED}" \
        --teacher "${TEACHER}" \
        --student_label "${LABEL}" \
        --datasets gsm8k \
        --modes alpha,gbv,traversal \
        --K "${KS}" \
        --temperature "${TEMPS}" \
        --n 30 \
        --max_tokens 100 \
        --skip_existing
done

echo ""
echo "=== Summary of EBE LR sweep ==="
python -c "
import sys; sys.path.insert(0,'.')
import results_db
rows = results_db.query_runs()
ebe = [r for r in rows if r.get('draft_label','').startswith('ebe_lr')]
by_label = {}
for r in ebe:
    k = (r['draft_label'], r['mode'], r['K'], round(r['temperature'],1))
    val = r.get('block_eff') or r.get('alpha_mean')
    if val:
        by_label[k] = val
for k in sorted(by_label):
    print(f'  {k[0]:15s}  mode={k[1]:10s}  K={k[2]}  T={k[3]}  val={by_label[k]:.4f}')
"
