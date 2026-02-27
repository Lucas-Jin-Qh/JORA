#!/bin/bash

# =============================================================================
# BOFT 实验脚本 - GPU1
# 仅运行 Alpaca 数据集的 BOFT 实验
# 已完成: boft_llama2_7b_alpaca_rank4, rank8, rank16, rank32 (各3个seed)
# 待运行: 12 个配置 × 3 seeds = 36 个实验
# =============================================================================

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Hugging Face 相关环境变量配置
export HF_HUB_DISABLE_TELEMETRY=1
export HF_ENDPOINT='https://hf-mirror.com'
export HF_DATASETS_CACHE='/home/jqh/Workshop/JORA/datasets'

# GPU 配置 - 使用 GPU1
export CUDA_VISIBLE_DEVICES=1

# 训练参数
seeds=(42 1337 2026)
learning_rate=0.0002
num_epochs=3
batch_size=1
gradient_accumulation_steps=8

# 模型和数据路径
declare -A model_paths
declare -A dataset_names

# Mistral-7B + Alpaca
model_paths["mistral_7b_alpaca"]="/mnt/sda/jqh/pretrained_checkpoints/Mistral-7B-v0.1/"
dataset_names["mistral_7b_alpaca"]="yahma/alpaca-cleaned"

# BOFT 配置文件列表 - Alpaca 数据集 (仅未完成的)
# 格式: boft_{model}_{dataset}_{type}{rank/bf_size}
configs=(
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

config_dir="config/boft"
output_base_dir="checkpoints/boft"

echo "========================================"
echo "BOFT Experiments - Alpaca (GPU1)"
echo "========================================"
echo ""
echo "已完成的实验 (已从列表中移除):"
echo "  - boft_llama2_7b_alpaca_rank4 (seeds: 42, 1337, 2026)"
echo "  - boft_llama2_7b_alpaca_rank8 (seeds: 42, 1337, 2026)"
echo "  - boft_llama2_7b_alpaca_rank16 (seeds: 42, 1337, 2026)"
echo "  - boft_llama2_7b_alpaca_rank32 (seeds: 42, 1337, 2026)"
echo ""
echo "待运行配置: ${#configs[@]} (12 个)"
echo "待运行实验: $(( ${#configs[@]} * 3 )) 个 (3 seeds each)"
echo "Seeds: 42, 1337, 2026"
echo "Learning Rate: ${learning_rate}"
echo "Epochs: ${num_epochs}"
echo "Batch Size: ${batch_size}"
echo "Gradient Accumulation: ${gradient_accumulation_steps}"
echo "========================================"

# 计数器
total_configs=${#configs[@]}
current_config=0
success_count=0
fail_count=0
skipped_count=0

# 遍历所有配置
for config_name in "${configs[@]}"; do
    current_config=$((current_config + 1))

    # 从配置名称中提取模型名称和数据集名称
    # 格式: boft_mistral_7b_alpaca_rank4 -> mistral_7b_alpaca
    IFS='_' read -ra parts <<< "$config_name"
    model_key="${parts[1]}_${parts[2]}_${parts[3]}"  # mistral_7b_alpaca

    config_file="${config_dir}/${config_name}.json"
    model_path="${model_paths[$model_key]}"
    dataset_name="${dataset_names[$model_key]}"

    echo ""
    echo "========================================"
    echo "[${current_config}/${total_configs}] Config: ${config_name}"
    echo "Model/Dataset: ${model_key}"
    echo "========================================"

    # 检查文件是否存在
    if [ ! -f "${config_file}" ]; then
        echo "⚠️  配置文件不存在，跳过: ${config_file}"
        skipped_count=$((skipped_count + 1))
        continue
    fi

    # 遍历所有 seed
    for seed in "${seeds[@]}"; do
        echo ""
        echo "🚀 运行 ${config_name} seed=${seed} ..."

        output_dir="${output_base_dir}/${config_name}_seed${seed}"

        python train_with_config.py \
            --model_path "${model_path}" \
            --dataset_name "${dataset_name}" \
            --config "${config_file}" \
            --output_dir "${output_dir}" \
            --num_epochs ${num_epochs} \
            --batch_size ${batch_size} \
            --gradient_accumulation_steps ${gradient_accumulation_steps} \
            --learning_rate ${learning_rate} \
            --execute \
            --seed ${seed}

        if [ $? -ne 0 ]; then
            echo "❌ 实验失败: ${config_name} seed=${seed}"
            fail_count=$((fail_count + 1))
            echo "跳过此实验，继续下一个..."
            continue
        fi

        echo "✅ 实验完成: ${config_name} seed=${seed}"
        success_count=$((success_count + 1))
        echo "----------------------------------------"

        # 清理显存
        sleep 2
        python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    done

    echo "✅ 配置 ${config_name} 所有 seed 实验完成"
done

echo ""
echo "========================================"
echo "🎉 GPU1 BOFT Alpaca 实验完成!"
echo ""
echo "实验统计:"
echo "  - 总配置数: ${total_configs}"
echo "  - 成功实验: ${success_count}"
echo "  - 失败实验: ${fail_count}"
echo "  - 跳过配置: ${skipped_count}"
echo "  - 输出目录: ${output_base_dir}"
echo "========================================"
