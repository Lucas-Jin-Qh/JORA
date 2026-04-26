#!/bin/bash
# P0 Experiments Launch Script
# Runs JORA-Diag 3 seeds vs Diag-only 3 seeds for rotation verdict
# Usage: bash scripts/run_p0_experiments.sh

set -e
cd /home/jqh/Workshop/JORA

export HF_HUB_OFFLINE=1
export HF_ENDPOINT=https://hf-mirror.com

TRAIN_SCRIPT="examples/sft/train.py"

echo "=============================================="
echo "P0 Experiments: JORA-Diag vs Diag-only (3 seeds)"
echo "=============================================="
echo ""

# GPU 1: JORA-Diag (s42, s2023, s7)
# GPU 2: Diag-only (s42, s2023, s7)
# Each pair (same seed) is launched together to share the model loading overhead

for SEED in 42 2023 7; do
    echo "[$(date)] Starting JORA-Diag s${SEED} (GPU 1) + Diag-only s${SEED} (GPU 2)"
    CUDA_VISIBLE_DEVICES=1 python $TRAIN_SCRIPT configs/run_diag_main_s${SEED}.json &
    PID1=$!
    CUDA_VISIBLE_DEVICES=2 python $TRAIN_SCRIPT configs/run_diag_no_rotation_s${SEED}.json &
    PID2=$!

    echo "  JORA-Diag s${SEED}: PID=$PID1"
    echo "  Diag-only s${SEED}: PID=$PID2"

    wait $PID1
    JORA_RET=$?
    wait $PID2
    DIAG_RET=$?

    if [ $JORA_RET -ne 0 ]; then
        echo "  [FAIL] JORA-Diag s${SEED} exited with code $JORA_RET"
    else
        echo "  [OK] JORA-Diag s${SEED} completed"
    fi

    if [ $DIAG_RET -ne 0 ]; then
        echo "  [FAIL] Diag-only s${SEED} exited with code $DIAG_RET"
    else
        echo "  [OK] Diag-only s${SEED} completed"
    fi
    echo ""
done

echo "=============================================="
echo "All P0 experiments complete"
echo "=============================================="

# Quick results extraction
echo ""
echo "--- Final Train Loss Summary ---"
for SEED in 42 2023 7; do
    for METHOD in "diag_main" "diag_no_rotation"; do
        LOG="results/run_${METHOD}_s${SEED}/trainer_state.json"
        if [ -f "$LOG" ]; then
            LAST_LOSS=$(python3 -c "
import json
with open('$LOG') as f:
    data = json.load(f)
history = data.get('log_history', [])
losses = [e['loss'] for e in history if 'loss' in e]
if losses:
    print(f's${SEED} {METHOD}: final_loss={losses[-1]:.4f}')
else:
    print(f's${SEED} {METHOD}: no loss recorded')
" 2>/dev/null || echo "s${SEED} ${METHOD}: parse error")
            echo "  $LAST_LOSS"
        else
            echo "  s${SEED} ${METHOD}: NO OUTPUT DIR ($LOG not found)"
        fi
    done
done
