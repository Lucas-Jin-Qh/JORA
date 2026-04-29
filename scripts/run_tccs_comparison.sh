#!/bin/bash
# Step 4 experiment: Compare Diag-Consecutive vs TC-CS-1S (attention-only, 1 epoch).
# Same seed (42), same S/k, same model/dataset — isolates pairing strategy effect.
# Usage: bash scripts/run_tccs_comparison.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIGS_DIR="$PROJECT_DIR/configs"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

GPU=0
conda_env="peft-jora"

run_one() {
  local config="$1"
  local name="${config%.json}"
  local logfile="$LOG_DIR/${name}.log"

  echo "[$(date '+%H:%M:%S')] === Starting $config on GPU $GPU ==="

  # Wait for GPU memory < 2GB
  while true; do
    mem_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')
    if [ "$mem_used" -lt 2000 ]; then
      break
    fi
    echo "  GPU $GPU busy (${mem_used} MiB), waiting 30s..."
    sleep 30
  done

  if [ -f "$logfile" ] && grep -q "DONE" "$logfile" 2>/dev/null; then
    echo "  SKIPPED (already done): $config"
    return 0
  fi

  # Kill any stale process
  pidfile="$LOG_DIR/${name}.pid"
  if [ -f "$pidfile" ]; then
    stale_pid=$(cat "$pidfile")
    kill -0 "$stale_pid" 2>/dev/null && { echo "  Killing stale PID $stale_pid"; kill "$stale_pid" || true; sleep 3; }
  fi

  (
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate "$conda_env"
    export PYTHONPATH="$PROJECT_DIR/src:$PYTHONPATH"
    export CUDA_VISIBLE_DEVICES="$GPU"
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1

    python -u "$PROJECT_DIR/examples/sft/train.py" "$CONFIGS_DIR/$config" \
      > "$logfile" 2>&1

    echo "[run_tccs_exp] DONE" >> "$logfile"
  ) &

  local pid=$!
  echo $pid > "$pidfile"
  echo "  Launched PID=$pid, log: $logfile"
}

# Run sequentially to isolate pairing strategy effect
for cfg in "run_diag_consecutive_s42.json" "run_tccs_1s_s42.json"; do
  run_one "$cfg"
  sleep 5
done

echo ""
echo "=== Both experiments launched ==="
echo "Monitor with:"
echo "  tail -f $LOG_DIR/run_diag_consecutive_s42.log"
echo "  tail -f $LOG_DIR/run_tccs_1s_s42.log"
echo ""
echo "When both are DONE, compare with:"
echo "  python -c \""
echo "    import json, os"
echo "    for name in ['run_diag_consecutive_s42', 'run_tccs_1s_s42']:"
echo "      log = '$LOG_DIR/' + name + '.log'"
echo "      for line in open(log):"
echo "        if 'train_loss' in line and 'DONE' not in line:"
echo "          print(name, line.strip())"
echo "  \""
