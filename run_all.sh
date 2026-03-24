#!/bin/bash

# =============================================================================
# 统一训练脚本 - 全部使用三卡并行训练 + DeepSpeed ZeRO-3
# 全部实验顺序执行，可一次性跑完
# =============================================================================

# 激活正确的 conda 环境
source /home/jqh/miniconda3/etc/profile.d/conda.sh
conda activate peft-jora

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Hugging Face 环境变量
export HF_HUB_DISABLE_TELEMETRY=1
export HF_ENDPOINT='https://hf-mirror.com'
export HF_DATASETS_CACHE='/home/jqh/Workshop/JORA/datasets'

# GPU 配置 - 全部3张卡
export CUDA_VISIBLE_DEVICES=0,1,2

# 训练参数 - 3卡并行 + DeepSpeed ZeRO-2
seeds=(42 1337 2026)
learning_rate=0.0002
num_epochs=3
batch_size=2
gradient_accumulation_steps=3  # 1×6×3 = 18 (接近之前的16)
num_gpus=3  # 3卡 + DeepSpeed
deepspeed_config="examples/sft/configs/deepspeed_config_3gpu_z2.yaml"
use_gradient_checkpointing=false  # DeepSpeed 与 gradient_checkpointing 不兼容

# 模型路径
llama2_path="/mnt/sda/jqh/pretrained_checkpoints/Llama-2-7b-hf/"
mistral_path="/mnt/sda/jqh/pretrained_checkpoints/Mistral-7B-v0.1/"

# 数据集
dataset_alpaca="yahma/alpaca-cleaned"
dataset_gsm8k="gsm8k:main"

output_base_dir="checkpoints"

echo "========================================"
echo "统一训练脚本 - 3卡并行"
echo "========================================"
echo "Batch Size: ${batch_size}"
echo "Num GPUs: ${num_gpus}"
echo "========================================"

total=39
current=0
success=0
failed=0

run_exp() {
    local model_path="$1"
    local dataset_name="$2"
    local config_file="$3"
    local output_dir="$4"
    local exp_name="$5"
    
    current=$((current + 1))
    
    echo ""
    echo "[${current}/${total}] ${exp_name}"
    
    # 检查是否已完成
    if [ -f "${output_dir}/adapter_model.safetensors" ] || [ -f "${output_dir}/pytorch_model.bin" ]; then
        echo "  ⏭️  已存在，跳过"
        return 0
    fi
    
    mkdir -p "${output_dir}"
    
    python train_with_config.py \
        --model_path "${model_path}" \
        --dataset_name "${dataset_name}" \
        --config "${config_file}" \
        --output_dir "${output_dir}" \
        --num_epochs ${num_epochs} \
        --batch_size ${batch_size} \
        --gradient_accumulation_steps ${gradient_accumulation_steps} \
        --learning_rate ${learning_rate} \
        --seed ${seed} \
        --num_gpus ${num_gpus} \
        --deepspeed "${deepspeed_config}" \
        --execute
    
    if [ $? -eq 0 ]; then
        echo "  ✅ 成功"
        success=$((success + 1))
    else
        echo "  ❌ 失败，删除输出目录"
        rm -rf "${output_dir}"
        failed=$((failed + 1))
    fi
    
    sleep 2
    python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
}

# ======== 阶段1: OFT Llama2-7B GSM8K (12个) ========
echo ""
echo ">>> 阶段1: OFT Llama2-7B GSM8K <<<"
ranks=(4 8 16 32)

for rank in "${ranks[@]}"; do
    config_file="config/oft/oft_llama2_7b_gsm8k_rank${rank}.json"
    
    if [ ! -f "${config_file}" ]; then
        echo "配置文件不存在: ${config_file}"
        continue
    fi
    
    for seed in "${seeds[@]}"; do
        output_dir="${output_base_dir}/oft/oft_llama2_7b_gsm8k_rank${rank}_seed${seed}"
        run_exp "${llama2_path}" "${dataset_gsm8k}" "${config_file}" "${output_dir}" "OFT Llama2-7B GSM8K rank${rank} seed${seed}"
    done
done

# ======== 阶段2: OFT Mistral-7B Alpaca (3个) ========
echo ""
echo ">>> 阶段2: OFT Mistral-7B Alpaca <<<"

config_file="config/oft/oft_mistral_7b_alpaca_rank4.json"

for seed in "${seeds[@]}"; do
    output_dir="${output_base_dir}/oft/oft_mistral_7b_alpaca_rank4_seed${seed}"
    run_exp "${mistral_path}" "${dataset_alpaca}" "${config_file}" "${output_dir}" "OFT Mistral-7B Alpaca rank4 seed${seed}"
done

# ======== 阶段3: BOFT Mistral-7B Alpaca (12个) ========
echo ""
echo ">>> 阶段3: BOFT Mistral-7B Alpaca <<<"
ranks=(4 8 16 32)

for rank in "${ranks[@]}"; do
    config_file="config/boft/boft_mistral_7b_alpaca_rank${rank}.json"
    
    if [ ! -f "${config_file}" ]; then
        echo "配置文件不存在: ${config_file}"
        continue
    fi
    
    for seed in "${seeds[@]}"; do
        output_dir="${output_base_dir}/boft/boft_mistral_7b_alpaca_rank${rank}_seed${seed}"
        run_exp "${mistral_path}" "${dataset_alpaca}" "${config_file}" "${output_dir}" "BOFT Mistral-7B Alpaca rank${rank} seed${seed}"
    done
done

# ======== 阶段4: BOFT Mistral-7B GSM8K (12个) ========
echo ""
echo ">>> 阶段4: BOFT Mistral-7B GSM8K <<<"

for rank in "${ranks[@]}"; do
    config_file="config/boft/boft_mistral_7b_gsm8k_rank${rank}.json"
    
    if [ ! -f "${config_file}" ]; then
        echo "配置文件不存在: ${config_file}"
        continue
    fi
    
    for seed in "${seeds[@]}"; do
        output_dir="${output_base_dir}/boft/boft_mistral_7b_gsm8k_rank${rank}_seed${seed}"
        run_exp "${mistral_path}" "${dataset_gsm8k}" "${config_file}" "${output_dir}" "BOFT Mistral-7B GSM8K rank${rank} seed${seed}"
    done
done

echo ""
echo "========================================"
echo "全部实验完成"
echo "成功: ${success} | 失败: ${failed}"
echo "========================================"
