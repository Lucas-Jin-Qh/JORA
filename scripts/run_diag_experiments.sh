#!/bin/bash
# Diagnostic experiment runner: queues configs across available GPUs.
# Usage: bash scripts/run_diag_experiments.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$PROJECT_DIR/logs"
CONFIGS_DIR="$PROJECT_DIR/configs"
SOURCE_DIR="$PROJECT_DIR/examples/sft"

mkdir -p "$LOG_DIR"

# Experiments queue: each entry is "config_file,gpu"
# Priority order: 3ep pair first (longest), then ablation quick runs
declare -a EXPERIMENTS=(
  # 3-epoch pair: rotation verdict at longer training
  "run_diag_main_s42_3ep.json,0",
  "run_diag_no_rotation_s42_3ep.json,1",
  # Ablation: rotation-only and diag-only (1 epoch each)
  "run_rotation_only_s42.json,2",
  "run_diag_only_s42.json,2",
  # lr_theta sweep (1 epoch each, sequential on GPU2)
  "run_lr_theta_1x_s42.json,2",
  "run_lr_theta_5x_s42.json,2",
  "run_lr_theta_10x_s42.json,2",
)

run_experiment() {
  local config="$1"
  local gpu="$2"
  local name="${config%.json}"
  local logfile="$LOG_DIR/${name}.log"
  local pidfile="$LOG_DIR/${name}.pid"

  echo "[$(date '+%H:%M:%S')] Starting $config on GPU $gpu"

  # Check if already completed
  if [ -f "$logfile" ] && grep -q "\[run_jora_exp\] DONE" "$logfile" 2>/dev/null; then
    echo "  SKIPPED (already done): $config"
    return 0
  fi

  # Wait for GPU to be free
  while true; do
    mem_used=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F',' -v g="$gpu" 'NR==g+1 {print $2}' | tr -d ' ')
    if [ "$mem_used" -lt 1000 ]; then
      break
    fi
    echo "  GPU $gpu busy (${mem_used} MiB), waiting..."
    sleep 30
  done

  # Kill any stale process
  if [ -f "$pidfile" ]; then
    stale_pid=$(cat "$pidfile")
    if kill -0 "$stale_pid" 2>/dev/null; then
      echo "  WARNING: stale PID $stale_pid still running, killing..."
      kill "$stale_pid" 2>/dev/null || true
      sleep 3
    fi
  fi

  # Launch
  (
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate peft-jora
    CUDA_VISIBLE_DEVICES="$gpu" \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    python -u "$SOURCE_DIR/train.py" "$CONFIGS_DIR/$config" \
      > "$logfile" 2>&1
  ) &
  local pid=$!
  echo $pid > "$pidfile"
  echo "$pid" > "$LOG_DIR/.current_pid_gpu${gpu}"
  echo "  Launched PID=$pid on GPU $gpu, log: $logfile"
}

# Main loop: run all experiments in order, respecting GPU assignments
for entry in "${EXPERIMENTS[@]}"; do
  config="${entry%,*}"
  gpu="${entry#*,}"
  run_experiment "$config" "$gpu"
  # Small stagger to let process fork settle
  sleep 5
done

echo ""
echo "[$(date '+%H:%M:%S')] All experiments launched. Monitor with:"
echo "  tail -f logs/<config>.log"
echo "  nvidia-smi"
