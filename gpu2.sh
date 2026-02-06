#!/bin/bash

# Hugging Face 相关环境变量配置
export HF_HUB_DISABLE_TELEMETRY=1
export HF_ENDPOINT='https://hf-mirror.com'
export HF_DATASETS_CACHE='/home/jqh/Workshop/JORA/datasets'

# GPU 配置 - 使用 GPU2
export CUDA_VISIBLE_DEVICES=2

# BOFT 实验参数
block_sizes=(4 8 16 32)
seeds=(42 1337 2026)
learning_rate=0.0001
num_epochs=3
batch_size=2

# LLaMA2-7B + GSM8K
llama_model_path="/mnt/sda/jqh/pretrained_checkpoints/Llama-2-7b-hf/"
llama_dataset_name="gsm8k:main"

# Mistral-7B + GSM8K  
mistral_model_path="/mnt/sda/jqh/pretrained_checkpoints/Mistral-7B-v0.1/"
mistral_dataset_name="gsm8k:main"

# 输出目录
output_base_dir="checkpoints/boft"

echo "========================================"
echo "BOFT Experiments (GPU2: LLaMA2-7B & Mistral-7B, block_size 4, 8, 16, 32)"
echo "Dataset: GSM8K"
echo "Seeds: 42, 1337, 2026"
echo "Learning Rate: ${learning_rate}"
echo "Epochs: ${num_epochs}"
echo "Batch Size: ${batch_size}"
echo "========================================"

# =============================================================================
# LLaMA2-7B + GSM8K BOFT 实验
# =============================================================================
echo ""
echo "========================================"
echo "Starting LLaMA2-7B + GSM8K + BOFT experiments..."
echo "========================================"

for block_size in "${block_sizes[@]}"; do
    echo ""
    echo "----------------------------------------"
    echo "Training LLaMA2-7B + GSM8K + BOFT block_size=${block_size}"
    echo "----------------------------------------"
    
    for seed in "${seeds[@]}"; do
        echo ""
        echo "正在运行 LLaMA2-7B BOFT block_size=${block_size}, seed=${seed} 的实验..."
        
        python train_with_config.py \
            --model_path "${llama_model_path}" \
            --dataset_name "${llama_dataset_name}" \
            --config "config/boft/boft_llama2_7b_gsm8k_block${block_size}.json" \
            --output_dir "${output_base_dir}/llama2_7b_gsm8k_block${block_size}_seed${seed}" \
            --num_epochs ${num_epochs} \
            --batch_size ${batch_size} \
            --learning_rate ${learning_rate} \
            --execute --seed ${seed}

        if [ $? -ne 0 ]; then
            echo "❌ 实验失败: LLaMA2-7B BOFT block_size=${block_size}, seed=${seed}"
            echo "停止执行"
            exit 1
        fi

        echo "✅ 实验完成: LLaMA2-7B BOFT block_size=${block_size}, seed=${seed}"
        echo "----------------------------------------"
    done

    echo ""
    echo "LLaMA2-7B + GSM8K 所有 seed 对于 block_size=${block_size} 的实验完成"
    echo "========================================"
done

echo ""
echo "✅ LLaMA2-7B + GSM8K + BOFT 所有实验完成!"
echo ""

# =============================================================================
# Mistral-7B + GSM8K BOFT 实验
# =============================================================================
echo "========================================"
echo "Starting Mistral-7B + GSM8K + BOFT experiments..."
echo "========================================"

for block_size in "${block_sizes[@]}"; do
    echo ""
    echo "----------------------------------------"
    echo "Training Mistral-7B + GSM8K + BOFT block_size=${block_size}"
    echo "----------------------------------------"
    
    for seed in "${seeds[@]}"; do
        echo ""
        echo "正在运行 Mistral-7B BOFT block_size=${block_size}, seed=${seed} 的实验..."
        
        python train_with_config.py \
            --model_path "${mistral_model_path}" \
            --dataset_name "${mistral_dataset_name}" \
            --config "config/boft/boft_mistral_7b_gsm8k_block${block_size}.json" \
            --output_dir "${output_base_dir}/mistral_7b_gsm8k_block${block_size}_seed${seed}" \
            --num_epochs ${num_epochs} \
            --batch_size ${batch_size} \
            --learning_rate ${learning_rate} \
            --execute --seed ${seed}

        if [ $? -ne 0 ]; then
            echo "❌ 实验失败: Mistral-7B BOFT block_size=${block_size}, seed=${seed}"
            echo "停止执行"
            exit 1
        fi

        echo "✅ 实验完成: Mistral-7B BOFT block_size=${block_size}, seed=${seed}"
        echo "----------------------------------------"
    done

    echo ""
    echo "Mistral-7B + GSM8K 所有 seed 对于 block_size=${block_size} 的实验完成"
    echo "========================================"
done

echo ""
echo "========================================"
echo "🎉 GPU2 所有 BOFT 实验成功完成!"
echo ""
echo "实验总结:"
echo "  - 模型: LLaMA2-7B, Mistral-7B"
echo "  - 数据集: GSM8K"
echo "  - BOFT block_size: 4, 8, 16, 32"
echo "  - Seeds: 42, 1337, 2026"
echo "  - 输出目录: ${output_base_dir}"
echo "========================================"

