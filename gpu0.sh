#!/bin/bash

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Hugging Face 相关环境变量配置
export HF_HUB_DISABLE_TELEMETRY=1
export HF_ENDPOINT='https://hf-mirror.com'
export HF_DATASETS_CACHE='/home/jqh/Workshop/JORA/datasets'

# GPU 配置
export CUDA_VISIBLE_DEVICES=0

# 定义参数数组（完整流水线：4/8/16/32）
ranks=(4 8 16 32)
seeds=(42 1337 2026)

# 循环执行全部实验（按模型划分：Llama -> GPU0）
for rank in "${ranks[@]}"; do
    echo "Training Llama-2-7B + alpaca-cleaned + rank ${rank} (OFT)..."
    echo "========================================"

    for seed in "${seeds[@]}"; do
        output_dir="checkpoints/oft/oft_llama2_7b_alpaca_rank${rank}_seed${seed}"

        # 如果输出目录已包含主要权重文件，则跳过（支持断点续跑）
        if [ -f "${output_dir}/adapter_model.safetensors" ] || [ -f "${output_dir}/adapter_config.json" ]; then
            echo "已存在输出，跳过: ${output_dir}"
            continue
        fi

        echo "正在运行 rank=${rank}, seed=${seed} 的实验 -> 输出：${output_dir}"
        mkdir -p "${output_dir}"

        python train_with_config.py \
            --model_path "/mnt/sda/jqh/pretrained_checkpoints/Llama-2-7b-hf/" \
            --dataset_name "yahma/alpaca-cleaned" \
            --config "config/oft/oft_llama2_7b_alpaca_rank${rank}.json" \
            --output_dir "${output_dir}" \
            --num_epochs 3 \
            --batch_size 1 \
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

echo "所有 Llama 实验成功完成！"