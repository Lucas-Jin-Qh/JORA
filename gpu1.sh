#!/bin/bash

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Hugging Face 相关环境变量配置
export HF_HUB_DISABLE_TELEMETRY=1
export HF_ENDPOINT='https://hf-mirror.com'
export HF_DATASETS_CACHE='/home/jqh/Workshop/JORA/datasets'

# GPU 配置 - 使用 GPU1
export CUDA_VISIBLE_DEVICES=1

# 定义参数数组（完整流水线：4/8/16/32）
ranks=(4 8 16 32)
seeds=(42 1337 2026)
model_path="/mnt/sda/jqh/pretrained_checkpoints/Mistral-7B-v0.1/"
dataset_name="yahma/alpaca-cleaned"
learning_rate=0.0002



# 循环执行全部实验（按模型划分：Mistral -> GPU1）
for rank in "${ranks[@]}"; do
    echo "Training Mistral-7B + alpaca-cleaned + rank ${rank} (OFT)..."
    echo "========================================"

    for seed in "${seeds[@]}"; do
        output_dir="checkpoints/oft/oft_mistral_7b_alpaca_rank${rank}_seed${seed}"

        # 如果输出目录已包含主要权重文件，则跳过（支持断点续跑）
        if [ -f "${output_dir}/adapter_model.safetensors" ] || [ -f "${output_dir}/adapter_config.json" ]; then
            echo "已存在输出，跳过: ${output_dir}"
            continue
        fi

        echo "正在运行 rank=${rank}, seed=${seed} 的实验 -> 输出：${output_dir}"
        mkdir -p "${output_dir}"

        python train_with_config.py \
            --model_path "${model_path}" \
            --dataset_name "${dataset_name}" \
            --config "config/oft/oft_mistral_7b_alpaca_rank${rank}.json" \
            --output_dir "${output_dir}" \
            --num_epochs 3 \
            --batch_size 1 \
            --learning_rate ${learning_rate} \
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

echo "GPU1 所有 Mistral 实验成功完成！"