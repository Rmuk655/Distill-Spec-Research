#!/bin/bash
# EAGLE pipeline for Qwen3-0.6B on laptop
# Run from OSD/ directory: bash bash_scripts/run_eagle.sh
#
# Step 1: generate hidden states (~20 min for 500 prompts on laptop)
# Step 2: train EAGLE head (~30 min for 2000 steps)
# Step 3: eval block efficiency (comparable to GBV results in results.db)

set -e
cd "$(dirname "$0")/.."

BASE="Qwen/Qwen3-0.6B"
TRAIN_DATA="data/gsm8k_train.jsonl"
EVAL_DATA="data/gsm8k_30.jsonl"
TMP="data/eagle_tmp"
CKPT="checkpoints/eagle-qwen3-06b"

echo "=== EAGLE Phase 1: Generate hidden states ==="
python eagle_bench.py gen \
    --base "$BASE" \
    --data "$TRAIN_DATA" \
    --out "$TMP" \
    --n 500

echo ""
echo "=== EAGLE Phase 2: Train draft head ==="
python eagle_bench.py train \
    --base "$BASE" \
    --tmp "$TMP" \
    --ckpt "$CKPT" \
    --steps 2000 \
    --lr 3e-5

echo ""
echo "=== EAGLE Phase 3: Evaluate block efficiency ==="
for K in 3 5; do
    for TEMP in 0.6 1.0; do
        echo "--- K=$K  temperature=$TEMP ---"
        python eagle_bench.py eval \
            --base "$BASE" \
            --ckpt "$CKPT" \
            --data "$EVAL_DATA" \
            --K "$K" \
            --temperature "$TEMP" \
            --max_new 100
    done
done

echo ""
echo "=== EAGLE pipeline complete. Results in results.db ==="
