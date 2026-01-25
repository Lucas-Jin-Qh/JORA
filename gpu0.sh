#!/bin/bash

# Hugging Face 相关环境变量配置
export HF_HUB_DISABLE_TELEMETRY=1
export HF_ENDPOINT='https://hf-mirror.com'
export HF_DATASETS_CACHE='/home/jqh/Workshop/JORA/datasets'

# GPU 配置
export CUDA_VISIBLE_DEVICES=0

# 定义参数数组
ranks=(4 8 16 32)
seeds=(42 1337 2026)

# 循环执行剩余实验（从 rank16_seed2026 开始）
for rank in "${ranks[@]}"; do
    echo "Training Llama-2-7B + alpaca-cleaned + rank ${rank} (DoRA)..."
    echo "========================================"

    for seed in "${seeds[@]}"; do
        # 跳过已完成的实验：rank4 和 rank8 的所有实验，以及 rank16 的 seed42 和 seed1337
        if [[ ($rank -eq 4) || ($rank -eq 8) || ($rank -eq 16 && ($seed -eq 42 || $seed -eq 1337)) ]]; then
            echo "跳过已完成的实验: rank=${rank}, seed=${seed}"
            continue
        fi

        echo "正在运行 rank=${rank}, seed=${seed} 的实验..."

        python train_with_config.py \
            --model_path "/mnt/sda/jqh/pretrained_checkpoints/Llama-2-7b-hf/" \
            --dataset_name "yahma/alpaca-cleaned" \
            --config "config/dora/dora_llama2_7b_alpaca_rank${rank}.json" \
            --output_dir "checkpoints/dora_llama2_7b_alpaca_rank${rank}_seed${seed}" \
            --num_epochs 3 \
            --batch_size 2 \
            --learning_rate 0.0002 \
            --execute --seed ${seed}
        
        if [ $? -ne 0 ]; then
            echo "实验 rank=${rank}, seed=${seed} 失败，停止执行"
            exit 1
        fi
        
        echo "实验 rank=${rank}, seed=${seed} 完成"
        echo "----------------------------------------"
    done
    
    echo "所有 rank=${rank} 的实验完成"
    echo "========================================"
done

echo "所有实验成功完成！"