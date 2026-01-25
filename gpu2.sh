#!/bin/bash

# Hugging Face 相关环境变量配置
export HF_HUB_DISABLE_TELEMETRY=1
export HF_ENDPOINT='https://hf-mirror.com'
export HF_DATASETS_CACHE='/home/jqh/Workshop/JORA/datasets'

# GPU 配置 - 使用 GPU2
export CUDA_VISIBLE_DEVICES=2

# 定义参数数组 - GPU2负责GSM8K数据集的所有实验
ranks=(4 8 16 32)
seeds=(42 1337 2026)
learning_rate=0.0001

# LLaMA2-7B + GSM8K
llama_model_path="/mnt/sda/jqh/pretrained_checkpoints/Llama-2-7b-hf/"
llama_dataset_name="gsm8k:main"

# Mistral-7B + GSM8K
mistral_model_path="/mnt/sda/jqh/pretrained_checkpoints/Mistral-7B-v0.1/"
mistral_dataset_name="gsm8k:main"

echo "=== GSM8K Experiments (GPU2: LLaMA2-7B & Mistral-7B, rank 4, 8, 16, 32) DoRA ==="

# 先执行LLaMA2-7B + GSM8K
echo "Starting LLaMA2-7B + GSM8K experiments..."
for rank in "${ranks[@]}"; do
    echo "Training LLaMA2-7B + GSM8K + rank ${rank} (DoRA)..."
    echo "========================================"

    for seed in "${seeds[@]}"; do
        echo "正在运行 LLaMA2 rank=${rank}, seed=${seed} 的实验..."

        python train_with_config.py \
            --model_path "${llama_model_path}" \
            --dataset_name "${llama_dataset_name}" \
            --config "config/dora/dora_llama2_7b_gsm8k_rank${rank}.json" \
            --output_dir "checkpoints/dora_llama2_7b_gsm8k_rank${rank}_seed${seed}" \
            --num_epochs 3 \
            --batch_size 2 \
            --learning_rate ${learning_rate} \
            --execute --seed ${seed}

        if [ $? -ne 0 ]; then
            echo "实验 LLaMA2 rank=${rank}, seed=${seed} 失败，停止执行"
            exit 1
        fi

        echo "实验 LLaMA2 rank=${rank}, seed=${seed} 完成"
        echo "----------------------------------------"
    done

    echo "LLaMA2-7B + GSM8K 所有 rank=${rank} 的实验完成"
    echo "========================================"
done

echo "LLaMA2-7B + GSM8K 所有实验完成"
echo "========================================"

# 再执行Mistral-7B + GSM8K
echo "Starting Mistral-7B + GSM8K experiments..."
for rank in "${ranks[@]}"; do
    echo "Training Mistral-7B + GSM8K + rank ${rank} (DoRA)..."
    echo "========================================"

    for seed in "${seeds[@]}"; do
        echo "正在运行 Mistral rank=${rank}, seed=${seed} 的实验..."

        python train_with_config.py \
            --model_path "${mistral_model_path}" \
            --dataset_name "${mistral_dataset_name}" \
            --config "config/dora/dora_mistral_7b_gsm8k_rank${rank}.json" \
            --output_dir "checkpoints/dora_mistral_7b_gsm8k_rank${rank}_seed${seed}" \
            --num_epochs 3 \
            --batch_size 2 \
            --learning_rate ${learning_rate} \
            --execute --seed ${seed}
        
        if [ $? -ne 0 ]; then
            echo "实验 Mistral rank=${rank}, seed=${seed} 失败，停止执行"
            exit 1
        fi

        echo "实验 Mistral rank=${rank}, seed=${seed} 完成"
        echo "----------------------------------------"
    done

    echo "Mistral-7B + GSM8K 所有 rank=${rank} 的实验完成"
    echo "========================================"
done

echo "Mistral-7B + GSM8K 所有实验完成"
echo "========================================"
echo "GPU2所有GSM8K实验(LLaMA2 & Mistral)成功完成！"