#!/usr/bin/env python3

# 生成 LoRA 实验脚本

models = ['llama2_7b', 'mistral_7b']
datasets = ['alpaca', 'gsm8k']
ranks = [4, 8, 16, 32]
seeds = [42, 1337, 2026]

# 数据集到学习率的映射
lr_map = {
    'alpaca': 0.0002,
    'gsm8k': 0.0001
}

# 模型路径映射
model_paths = {
    'llama2_7b': '/mnt/sda/jqh/pretrained_checkpoints/Llama-2-7b-hf/',
    'mistral_7b': '/mnt/sda/jqh/pretrained_checkpoints/Mistral-7B-v0.1/'
}

# 数据集名称映射
dataset_names = {
    'alpaca': 'yahma/alpaca-cleaned',
    'gsm8k': 'gsm8k:main'
}

script_content = '''#!/bin/bash

# LoRA 实验训练脚本
# 包含所有配置的训练命令，每个配置3个seed

export HF_HUB_DISABLE_TELEMETRY=1
export HF_ENDPOINT='https://hf-mirror.com'
export HF_DATASETS_CACHE='/home/jqh/Workshop/JORA/datasets'
'''

for model in models:
    for dataset in datasets:
        script_content += f'''
# {model.replace('_', '-')} + {dataset} (lr={lr_map[dataset]})
echo "=== {model.replace('_', '-')} + {dataset} ==="
'''
        for rank in ranks:
            script_content += f'''
# rank {rank}
echo "Training {model.replace('_', '-')} + {dataset} + rank {rank}..."
'''
            for seed in seeds:
                script_content += f'''CUDA_VISIBLE_DEVICES=1 python train_with_config.py \\
    --model_path "{model_paths[model]}" \\
    --dataset_name "{dataset_names[dataset]}" \\
    --config "config/lora/lora_{model}_{dataset}_rank{rank}.json" \\
    --output_dir "checkpoints/lora_{model}_{dataset}_rank{rank}_seed{seed}" \\
    --num_epochs 3 \\
    --batch_size 2 \\
    --learning_rate {lr_map[dataset]} \\
    --seed {seed} \\
    --execute

'''

script_content += '''
echo "All LoRA experiments completed!"
'''

with open('scripts/run_lora_experiments.sh', 'w') as f:
    f.write(script_content)

print('LoRA experiment script generated successfully!')
