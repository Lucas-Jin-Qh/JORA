#!/bin/bash

# =============================================================================
# BOFT 实验脚本 - 多 GPU 分布式训练
# 合并 gpu1.sh 和 gpu2.sh，支持多 GPU 分布式训练解决 OOM 问题
# 已完成: boft_llama2_7b_alpaca/gsm8k_rank4, rank8, rank16, rank32 (各3个seed)
# 待运行: 24 个配置 × 3 seeds = 72 个实验
# =============================================================================

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Hugging Face 相关环境变量配置
export HF_HUB_DISABLE_TELEMETRY=1
export HF_ENDPOINT='https://hf-mirror.com'
export HF_DATASETS_CACHE='/home/jqh/Workshop/JORA/datasets'

# GPU 配置 - 使用所有可用 GPU (通过 --num_gpus 参数控制)
# 默认使用 2 个 GPU，如果需要更多可以修改
NUM_GPUS=${1:-2}  # 可以通过命令行参数指定 GPU 数量

# 如果指定了 GPU 列表，使用该列表；否则使用 0,1,2,3
if [ -n "$2" ]; then
    export CUDA_VISIBLE_DEVICES=$2
    echo "使用 GPU: $2"
else
    # 默认使用前 N 个 GPU
    export CUDA_VISIBLE_DEVICES=0,1
    echo "使用 GPU: 0,1 (默认)"
fi

# 训练参数
seeds=(42 1337 2026)
learning_rate=0.0002
num_epochs=3
batch_size=1
gradient_accumulation_steps=8

# 模型和数据路径
declare -A model_paths
declare -A dataset_names

# Mistral-7B
model_paths["mistral_7b"]="/mnt/sda/jqh/pretrained_checkpoints/Mistral-7B-v0.1/"

# LLaMA2-7B
model_paths["llama2_7b"]="/mnt/sda/jqh/pretrained_checkpoints/Llama-2-7b-hf/"

# 数据集
dataset_names["alpaca"]="yahma/alpaca-cleaned"
dataset_names["gsm8k"]="gsm8k:main"

# BOFT 配置文件列表 - Alpaca 数据集 (仅未完成的)
# 格式: boft_{model}_{dataset}_{type}{rank/bf_size}
alpaca_configs=(
    # Mistral-7B Alpaca - 待运行 (全部)
    "boft_mistral_7b_alpaca_rank4"
    "boft_mistral_7b_alpaca_rank8"
    "boft_mistral_7b_alpaca_rank16"
    "boft_mistral_7b_alpaca_rank32"
    "boft_mistral_7b_alpaca_bf4"
    "boft_mistral_7b_alpaca_bf8"
    "boft_mistral_7b_alpaca_bf8_2"
    "boft_mistral_7b_alpaca_bf16_2"

    # LLaMA2-7B Alpaca - bf 配置待运行 (rank 已完成)
    "boft_llama2_7b_alpaca_bf4"
    "boft_llama2_7b_alpaca_bf8"
    "boft_llama2_7b_alpaca_bf8_2"
    "boft_llama2_7b_alpaca_bf16_2"
)

# BOFT 配置文件列表 - GSM8K 数据集 (仅未完成的)
gsm8k_configs=(
    # Mistral-7B GSM8K - 待运行 (全部)
    "boft_mistral_7b_gsm8k_rank4"
    "boft_mistral_7b_gsm8k_rank8"
    "boft_mistral_7b_gsm8k_rank16"
    "boft_mistral_7b_gsm8k_rank32"
    "boft_mistral_7b_gsm8k_bf4"
    "boft_mistral_7b_gsm8k_bf8"
    "boft_mistral_7b_gsm8k_bf8_2"
    "boft_mistral_7b_gsm8k_bf16_2"

    # LLaMA2-7B GSM8K - bf 配置待运行 (rank 已完成)
    "boft_llama2_7b_gsm8k_bf4"
    "boft_llama2_7b_gsm8k_bf8"
    "boft_llama2_7b_gsm8k_bf8_2"
    "boft_llama2_7b_gsm8k_bf16_2"
)

config_dir="config/boft"
output_base_dir="checkpoints/boft"

echo "========================================"
echo "BOFT Experiments - Multi-GPU Distributed"
echo "========================================"
echo ""
echo "GPU 数量: ${NUM_GPUS}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo ""
echo "已完成的实验 (已从列表中移除):"
echo "  - boft_llama2_7b_alpaca_rank4, rank8, rank16, rank32"
echo "  - boft_llama2_7b_gsm8k_rank4, rank8, rank16, rank32"
echo ""
echo "待运行配置: ${#alpaca_configs[@]} (Alpaca) + ${#gsm8k_configs[@]} (GSM8K) = $(( ${#alpaca_configs[@]} + ${#gsm8k_configs[@]} ))"
echo "待运行实验: $(( (${#alpaca_configs[@]} + ${#gsm8k_configs[@]}) * 3 )) 个 (3 seeds each)"
echo "Seeds: 42, 1337, 2026"
echo "Learning Rate: ${learning_rate}"
echo "Epochs: ${num_epochs}"
echo "Batch Size: ${batch_size}"
echo "Gradient Accumulation: ${gradient_accumulation_steps}"
echo "========================================"

# 计数器
total_configs=$((${#alpaca_configs[@]} + ${#gsm8k_configs[@]}))
current_config=0
success_count=0
fail_count=0
skipped_count=0

# 函数：运行单个实验
run_experiment() {
    local config_name=$1
    local model_key=$2
    local dataset_key=$3
    local seed=$4
    
    local model_path="${model_paths[$model_key]}"
    local dataset_name="${dataset_names[$dataset_key]}"
    
    current_config=$((current_config + 1))
    
    echo ""
    echo "========================================"
    echo "[${current_config}/${total_configs}] Config: ${config_name}"
    echo "Model/Dataset: ${model_key}/${dataset_key} | Seed: ${seed}"
    echo "========================================"
    
    # 检查配置文件
    config_file="${config_dir}/${config_name}.json"
    if [ ! -f "${config_file}" ]; then
        echo "⚠️  配置文件不存在，跳过: ${config_file}"
        skipped_count=$((skipped_count + 1))
        return 1
    fi
    
    echo "🚀 运行 ${config_name} seed=${seed} ..."
    
    output_dir="${output_base_dir}/${config_name}_seed${seed}"
    mkdir -p "${output_dir}"
    
    # 使用多 GPU 或单 GPU
    if [ "$NUM_GPUS" -gt 1 ]; then
        echo "🔧 使用 ${NUM_GPUS} GPU 分布式训练"
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
            --num_gpus ${NUM_GPUS} \
            --execute
    else
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
            --execute
    fi
    
    if [ $? -ne 0 ]; then
        echo "❌ 实验失败: ${config_name} seed=${seed}"
        fail_count=$((fail_count + 1))
        return 1
    fi
    
    echo "✅ 实验完成: ${config_name} seed=${seed}"
    success_count=$((success_count + 1))
    echo "----------------------------------------"
    
    # 清理显存
    sleep 2
    python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    
    return 0
}

# 遍历 Alpaca 配置
echo ""
echo "========== Alpaca 数据集实验 =========="

for config_name in "${alpaca_configs[@]}"; do
    # 从配置名称中提取模型名称
    # 格式: boft_mistral_7b_alpaca_rank4 -> mistral_7b
    model_key=$(echo "$config_name" | cut -d'_' -f2,3)
    dataset_key="alpaca"
    
    for seed in "${seeds[@]}"; do
        run_experiment "$config_name" "$model_key" "$dataset_key" "$seed" || true
    done
    
    echo "✅ 配置 ${config_name} 所有 seed 实验完成"
done

# 遍历 GSM8K 配置
echo ""
echo "========== GSM8K 数据集实验 =========="

for config_name in "${gsm8k_configs[@]}"; do
    # 从配置名称中提取模型名称
    # 格式: boft_mistral_7b_gsm8k_rank4 -> mistral_7b
    model_key=$(echo "$config_name" | cut -d'_' -f2,3)
    dataset_key="gsm8k"
    
    for seed in "${seeds[@]}"; do
        run_experiment "$config_name" "$model_key" "$dataset_key" "$seed" || true
    done
    
    echo "✅ 配置 ${config_name} 所有 seed 实验完成"
done

echo ""
echo "========================================"
echo "🎉 BOFT 多 GPU 实验完成!"
echo ""
echo "实验统计:"
echo "  - 总配置数: ${total_configs}"
echo "  - 成功实验: ${success_count}"
echo "  - 失败实验: ${fail_count}"
echo "  - 跳过配置: ${skipped_count}"
echo "  - GPU 数量: ${NUM_GPUS}"
echo "  - 输出目录: ${output_base_dir}"
echo "========================================"


