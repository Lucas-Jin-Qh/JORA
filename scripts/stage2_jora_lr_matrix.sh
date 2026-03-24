#!/usr/bin/env bash
# Stage 2 LR Matrix: JORA-none with varying theta/core LR
# GPU0: theta/core = 1x/1x (lr_theta=0.05, lr_core=0.01)
# GPU1: theta/core = 5x/1x (lr_theta=0.25, lr_core=0.01)
# GPU2: theta/core = 10x/2x (lr_theta=0.5, lr_core=0.02)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/jqh/miniconda3/envs/peft-jora/bin/python}"
export PYTHONPATH="${REPO_ROOT}/src"

# Model and training config
MODEL_ID="${MODEL_ID:-facebook/opt-350m}"
DATASET_NAME="${DATASET_NAME:-yahma/alpaca-cleaned}"
MAX_STEPS="${MAX_STEPS:-1000}"
WORKDIR="${WORKDIR:-/tmp/jora-lr-matrix}"

# JORA params (using magnitude=none as per plan)
JORA_TARGETS="${JORA_TARGETS:-q_proj,k_proj,v_proj,out_proj}"
JORA_S_L=8
JORA_S_R=8
JORA_K=8
JORA_SELECTION="${JORA_SELECTION:-topk_ema}"
JORA_WARMUP="${JORA_WARMUP:-50}"
BASE_LR="1e-4"

# LR matrix configurations
# Format: name:lr_theta:lr_core:note
declare -a LR_CONFIGS=(
    "lr_1x_1x:0.05:0.01:基准"
    "lr_5x_1x:0.25:0.01:高theta"
    "lr_10x_2x:0.5:0.02:高theta高core"
)

echo "=== Stage 2: JORA LR Matrix (magnitude=none) ==="
echo "Model: $MODEL_ID"
echo "Steps: $MAX_STEPS"
echo "Base LR: $BASE_LR"
echo "Workdir: $WORKDIR"
echo ""

mkdir -p "$WORKDIR"

# Function to train JORA
train_jora() {
    local gpu_id="$1"
    local output_dir="$2"
    local lr_theta="$3"
    local lr_core="$4"
    local note="$5"
    local log_file="$6"
    
    if [ -d "$output_dir" ]; then
        echo "[GPU$gpu_id] Using existing: $output_dir"
        return 0
    fi
    
    echo "[GPU$gpu_id] Starting JORA ($note) lr_theta=$lr_theta lr_core=$lr_core"
    CUDA_VISIBLE_DEVICES=$gpu_id nohup "$PYTHON_BIN" examples/sft/train.py \
        --seed 42 \
        --model_name_or_path "$MODEL_ID" \
        --dataset_name "$DATASET_NAME" \
        --chat_template_format none \
        --add_special_tokens False \
        --append_concat_token False \
        --splits train \
        --torch_dtype bfloat16 \
        --bf16 True \
        --max_steps "$MAX_STEPS" \
        --logging_steps 100 \
        --save_steps 500 \
        --eval_strategy no \
        --report_to none \
        --output_dir "$output_dir" \
        --per_device_train_batch_size 8 \
        --gradient_accumulation_steps 1 \
        --learning_rate "$BASE_LR" \
        --max_length 512 \
        --dataset_text_field text \
        --use_cpu False \
        --gradient_checkpointing True \
        --use_reentrant False \
        --use_peft_jora True \
        --lora_target_modules "$JORA_TARGETS" \
        --jora_s_l "$JORA_S_L" \
        --jora_s_r "$JORA_S_R" \
        --jora_k "$JORA_K" \
        --jora_rotation_impl auto \
        --jora_selection_type "$JORA_SELECTION" \
        --jora_magnitude none \
        --jora_warmup_steps "$JORA_WARMUP" \
        --jora_lr_theta "$lr_theta" \
        --jora_lr_core "$lr_core" \
        --lr_scheduler_type cosine \
        --warmup_ratio 0.03 \
        > "$log_file" 2>&1 &
    
    echo "[GPU$gpu_id] Training started in background, log: $log_file"
    echo "[GPU$gpu_id] PID: $!"
}

# Launch training on 3 GPUs in parallel
echo "=========================================="
echo "Launching 3 JORA trainings in parallel"
echo "=========================================="
echo ""

for config in "${LR_CONFIGS[@]}"; do
    IFS=':' read -r name lr_theta lr_core note <<< "$config"
    
    gpu_idx=$(echo "$name" | grep -o '_[0-9]*x' | head -1 | tr -d 'x_' | head -1)
    case "$name" in
        lr_1x_1x) gpu_id=0 ;;
        lr_5x_1x) gpu_id=1 ;;
        lr_10x_2x) gpu_id=2 ;;
    esac
    
    output_dir="$WORKDIR/$name"
    log_file="$WORKDIR/${name}.log"
    
    # Export vars for env display
    echo "----------------------------------------"
    echo "Config: $name ($note)"
    echo "  GPU: $gpu_id"
    echo "  lr_theta: $lr_theta"
    echo "  lr_core: $lr_core"
    echo "  Output: $output_dir"
    echo "----------------------------------------"
    
    train_jora "$gpu_id" "$output_dir" "$lr_theta" "$lr_core" "$note" "$log_file"
done

echo ""
echo "=========================================="
echo "All 3 trainings launched!"
echo "=========================================="
echo ""
echo "Monitor with:"
echo "  tail -f $WORKDIR/lr_1x_1x.log"
echo "  tail -f $WORKDIR/lr_5x_1x.log"
echo "  tail -f $WORKDIR/lr_10x_2x.log"
echo ""
echo "Check GPU status:"
echo "  nvidia-smi"
