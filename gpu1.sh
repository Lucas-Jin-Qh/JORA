#!/bin/bash

# =============================================================================
# BOFT 实验脚本 - GPU1
# 运行 config/boft/ 目录下所有 32 个配置
# 每个配置 3 个 seed 串行自动运行
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
gradient_accumulation_steps=8  # 保持等效 batch_size=8 (1*8)

# 模型和数据路径
declare -A model_paths
declare -A dataset_names

# LLaMA2-7B
model_paths["llama2_7b_gsm8k"]="/mnt/sda/jqh/pretrained_checkpoints/Llama-2-7b-hf/"
dataset_names["llama2_7b_gsm8k"]="gsm8k:main"
model_paths["llama2_7b_alpaca"]="/mnt/sda/jqh/pretrained_checkpoints/Llama-2-7b-hf/"
dataset_names["llama2_7b_alpaca"]="yahma/alpaca-cleaned"

# Mistral-7B
model_paths["mistral_7b_gsm8k"]="/mnt/sda/jqh/pretrained_checkpoints/Mistral-7B-v0.1/"
dataset_names["mistral_7b_gsm8k"]="gsm8k:main"
model_paths["mistral_7b_alpaca"]="/mnt/sda/jqh/pretrained_checkpoints/Mistral-7B-v0.1/"
dataset_names["mistral_7b_alpaca"]="yahma/alpaca-cleaned"

# 输出目录
output_base_dir="checkpoints/boft"

# BOFT 配置文件列表 (从 config/boft/ 目录读取)
configs=(
    "boft_llama2_7b_alpaca_rank4"
    "boft_llama2_7b_alpaca_rank8"
    "boft_llama2_7b_alpaca_rank16"
    "boft_llama2_7b_alpaca_rank32"
    "boft_llama2_7b_alpaca_bf4"
    "boft_llama2_7b_alpaca_bf8"
    "boft_llama2_7b_alpaca_bf8_2"
    "boft_llama2_7b_alpaca_bf16_2"
    "boft_llama2_7b_gsm8k_rank4"
    "boft_llama2_7b_gsm8k_rank8"
    "boft_llama2_7b_gsm8k_rank16"
    "boft_llama2_7b_gsm8k_rank32"
    "boft_llama2_7b_gsm8k_bf4"
    "boft_llama2_7b_gsm8k_bf8"
    "boft_llama2_7b_gsm8k_bf8_2"
    "boft_llama2_7b_gsm8k_bf16_2"
    "boft_mistral_7b_alpaca_rank4"
    "boft_mistral_7b_alpaca_rank8"
    "boft_mistral_7b_alpaca_rank16"
    "boft_mistral_7b_alpaca_rank32"
    "boft_mistral_7b_alpaca_bf4"
    "boft_mistral_7b_alpaca_bf8"
    "boft_mistral_7b_alpaca_bf8_2"
    "boft_mistral_7b_alpaca_bf16_2"
    "boft_mistral_7b_gsm8k_rank4"
    "boft_mistral_7b_gsm8k_rank8"
    "boft_mistral_7b_gsm8k_rank16"
    "boft_mistral_7b_gsm8k_rank32"
    "boft_mistral_7b_gsm8k_bf4"
    "boft_mistral_7b_gsm8k_bf8"
    "boft_mistral_7b_gsm8k_bf8_2"
    "boft_mistral_7b_gsm8k_bf16_2"
)

config_dir="config/boft"

echo "========================================"
echo "BOFT Experiments (GPU1)"
echo "Total Configs: ${#configs[@]}"
echo "Seeds: 42, 1337, 2026"
echo "Learning Rate: ${learning_rate}"
echo "Epochs: ${num_epochs}"
echo "Batch Size: ${batch_size}"
echo "========================================"

# 计数器
total_configs=${#configs[@]}
current_config=0
success_count=0
fail_count=0

# 遍历所有配置
for config_name in "${configs[@]}"; do
    current_config=$((current_config + 1))

    # 从配置名称中提取模型名称和数据集名称
    # 格式: boft_llama2_7b_gsm8k_rank4 -> llama2_7b_gsm8k
    IFS='_' read -ra parts <<< "$config_name"
    model_key="${parts[1]}_${parts[2]}_${parts[3]}"  # llama2_7b_gsm8k

    config_file="${config_dir}/${config_name}.json"
    model_path="${model_paths[$model_key]}"
    dataset_name="${dataset_names[$model_key]}"

    echo ""
    echo "========================================"
    echo "[${current_config}/${total_configs}] Config: ${config_name}"
    echo "Model/Dataset: ${model_key}"
    echo "========================================"

    # 遍历所有 seed
    for seed in "${seeds[@]}"; do
        echo ""
        echo "正在运行 ${config_name} seed=${seed} 的实验..."

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
            --execute --seed ${seed}

        if [ $? -ne 0 ]; then
            echo "❌ 实验失败: ${config_name} seed=${seed}"
            fail_count=$((fail_count + 1))
            echo "停止执行"
            exit 1
        fi

        echo "✅ 实验完成: ${config_name} seed=${seed}"
        echo "----------------------------------------"
    done

    success_count=$((success_count + 1))
    echo ""
    echo "✅ 配置 ${config_name} 所有 seed 实验完成"
done

echo ""
echo "========================================"
echo "🎉 GPU1 所有 BOFT 实验成功完成!"
echo ""
echo "实验统计:"
echo "  - 总配置数: ${total_configs}"
echo "  - 成功: ${success_count}"
echo "  - 失败: ${fail_count}"
echo "  - 总实验数: $((total_configs * ${#seeds[@]}))"
echo "  - 输出目录: ${output_base_dir}"
echo "========================================"
