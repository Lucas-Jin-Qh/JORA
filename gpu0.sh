#!/bin/bash

# =============================================================================
# JORA 超参数扫描实验脚本 - GPU0
# 运行 jora_sweep 目录下所有配置
# 支持单配置和多实验（experiments数组）两种格式
# =============================================================================

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Hugging Face 相关环境变量配置
export HF_HUB_DISABLE_TELEMETRY=1
export HF_ENDPOINT='https://hf-mirror.com'
export HF_DATASETS_CACHE='/home/jqh/Workshop/JORA/datasets'

# GPU 配置 - 使用 GPU0
export CUDA_VISIBLE_DEVICES=0

# 训练参数
seeds=(42)
learning_rate=0.0002
num_epochs=3
batch_size=4
gradient_accumulation_steps=8

# JORA Sweep 配置文件列表
sweep_configs=(
    "main_config.json"
    "block_s_k_sweep.json"
    "block_size_sweep.json"
    "diag_baseline.json"
    "selection_refine.json"
    "magnitude_compare.json"
    "rotation_compare.json"
    "ablation_study.json"
    "lowrank_explore.json"
    "asymmetric_scan.json"
    "pairing_strategy_scan.json"
    "learning_rate_scan.json"
    "fine_block_size_scan.json"
    "all_linear_track.json"
    "temperature_annealing.json"
    "ema_grid_scan.json"
    "dataset_comparison.json"
    "model_comparison.json"
    "batch_lr_interaction.json"
    "advanced_ablation.json"
)

# JORA Sweep 配置目录
sweep_config_dir="/home/jqh/Workshop/JORA/config/jora_sweep"

# 临时配置文件目录
temp_config_dir="/home/jqh/Workshop/JORA/config/jora_sweep/temp_configs"
mkdir -p "${temp_config_dir}"

# 模型和数据本地路径
declare -A model_paths
declare -A dataset_names

# Mistral-7B (主实验模型)
model_paths["mistral_7b"]="/mnt/sda/jqh/pretrained_checkpoints/Mistral-7B-v0.1/"
dataset_names["mistral_7b_gsm8k"]="gsm8k:main"
dataset_names["mistral_7b_alpaca"]="yahma/alpaca-cleaned"

# LLaMA2-7B (辅助对比模型)
model_paths["llama2_7b"]="/mnt/sda/jqh/pretrained_checkpoints/Llama-2-7b-hf/"
dataset_names["llama2_7b_gsm8k"]="gsm8k:main"
dataset_names["llama2_7b_alpaca"]="yahma/alpaca-cleaned"

# 输出目录
output_base_dir="checkpoints/jora_sweep"

echo "========================================"
echo "JORA Hyperparameter Sweep (GPU0)"
echo "========================================"
echo ""
echo "配置目录: ${sweep_config_dir}"
echo "总配置数: ${#sweep_configs[@]}"
echo "Seeds: 42"
echo "Models: mistral_7b (主), llama2_7b (辅)"
echo "Datasets: alpaca (主), gsm8k (辅)"
echo ""
echo "训练参数:"
echo "  - Learning Rate: ${learning_rate}"
echo "  - Epochs: ${num_epochs}"
echo "  - Batch Size: ${batch_size}"
echo "  - Gradient Accumulation: ${gradient_accumulation_steps}"
echo ""
echo "输出目录: ${output_base_dir}"
echo "========================================"

# 创建输出目录
mkdir -p "${output_base_dir}"

# 清理临时配置文件
rm -rf "${temp_config_dir}"/*
echo "临时配置目录: ${temp_config_dir}"

# 统计变量
total_configs=${#sweep_configs[@]}
current_config=0
success_count=0
fail_count=0
skipped_count=0

# 函数：检测配置文件格式并生成实验列表
generate_experiments() {
    local config_path="$1"
    local config_file="$2"

    # 使用 Python 检测配置格式
    python3 -c "
import json
import sys

with open('$config_path', 'r') as f:
    config = json.load(f)

# 检查是否是单配置格式（有 'peft_type' 直接在顶层）
if 'peft_type' in config:
    print(f'SINGLE:{config.get(\"name\", \"$config_file\")}')
else:
    # 多实验格式 - 有 experiments 数组
    experiments = config.get('experiments', [])
    fixed_params = config.get('fixed_params', {})
    for i, exp in enumerate(experiments):
        merged = {**fixed_params, **exp}
        name = exp.get('name', f'exp{i}')
        print(f'MULTI:{name}', end='')
        if i < len(experiments) - 1:
            print()
    print()
"
}

# 函数：合并配置并生成临时文件
merge_and_create_temp_config() {
    local config_path="$1"
    local exp_name="$2"
    local temp_file="$3"

    python3 -c "
import json
import sys

exp_name = '$exp_name'

with open('$config_path', 'r') as f:
    config = json.load(f)

# 单配置格式
if 'peft_type' in config:
    merged = config
else:
    # 多实验格式 - 合并 fixed_params 和对应的 experiment
    experiments = config.get('experiments', [])
    fixed_params = config.get('fixed_params', {})

    merged = None
    for exp in experiments:
        if exp.get('name', '').startswith(exp_name.split('_')[-1]) or exp.get('name') == exp_name:
            merged = {**fixed_params, **exp}
            break

    if merged is None:
        print(f'Error: experiment {exp_name} not found in $config_path', file=sys.stderr)
        sys.exit(1)

# 输出合并后的配置
with open('$temp_file', 'w') as f:
    json.dump(merged, f, indent=2)

print(f'Created temp config: $temp_file')
"
}

# 遍历所有 JORA Sweep 配置
for config_file in "${sweep_configs[@]}"; do
    current_config=$((current_config + 1))

    config_path="${sweep_config_dir}/${config_file}"

    # 检查配置文件是否存在
    if [ ! -f "${config_path}" ]; then
        echo ""
        echo "⚠️  [${current_config}/${total_configs}] 配置文件不存在，跳过: ${config_path}"
        skipped_count=$((skipped_count + 1))
        continue
    fi

    config_name=$(basename "${config_file}" .json)

    echo ""
    echo "========================================"
    echo "[${current_config}/${total_configs}] 配置: ${config_file}"
    echo "========================================"

    # 检测配置格式并获取实验列表
    config_format=$(python3 -c "
import json
with open('${config_path}', 'r') as f:
    config = json.load(f)
print('SINGLE' if 'peft_type' in config else 'MULTI')
" 2>/dev/null)

    if [ "$config_format" = "SINGLE" ]; then
        # 单配置格式 - 直接使用
        experiments=("$config_name")
        echo "📄 格式: 单配置"
    else
        # 多实验格式 - 获取实验列表
        echo "📄 格式: 多实验 (experiments数组)"
        experiments=($(python3 -c "
import json
with open('${config_path}', 'r') as f:
    config = json.load(f)
for exp in config.get('experiments', []):
    print(exp.get('name', 'unknown'))
" 2>/dev/null))
    fi

    echo "🔬 实验数: ${#experiments[@]}"

    # 遍历每个实验
    for exp_name in "${experiments[@]}"; do
        # 为每个实验创建临时配置文件
        temp_config_file="${temp_config_dir}/${config_name}_${exp_name}.json"

        if [ "$config_format" = "MULTI" ]; then
            merge_and_create_temp_config "$config_path" "$exp_name" "$temp_config_file"
            actual_config="$temp_config_file"
        else
            actual_config="$config_path"
        fi

        # 遍历两个模型
        for model_key in "mistral_7b" "llama2_7b"; do
            model_path="${model_paths[$model_key]}"

            if [ ! -d "${model_path}" ]; then
                echo "⚠️  模型路径不存在，跳过: ${model_path}"
                continue
            fi

            # 遍历两个数据集 (alpaca 优先)
            for dataset_key in "alpaca" "gsm8k"; do
                dataset_name="${dataset_names[${model_key}_${dataset_key}]}"

                echo ""
                echo "----------------------------------------"
                echo "🔧 实验: ${exp_name} | 模型: ${model_key} | 数据集: ${dataset_key}"
                echo "----------------------------------------"

    # 遍历所有 seed
    for seed in "${seeds[@]}"; do
        echo ""
                    echo "🚀 运行: ${config_name}/${exp_name} | ${model_key} | ${dataset_key} | seed=${seed} ..."

                    output_dir="${output_base_dir}/${config_name}/${exp_name}/${model_key}/${dataset_key}/seed${seed}"
                    mkdir -p "${output_dir}"

        python train_with_config.py \
            --model_path "${model_path}" \
            --dataset_name "${dataset_name}" \
                        --config "${actual_config}" \
            --output_dir "${output_dir}" \
            --num_epochs ${num_epochs} \
            --batch_size ${batch_size} \
            --gradient_accumulation_steps ${gradient_accumulation_steps} \
            --learning_rate ${learning_rate} \
                        --seed ${seed} \
                        --execute

        if [ $? -ne 0 ]; then
                        echo "❌ 实验失败: ${config_name}/${exp_name} | ${model_key} | ${dataset_key} | seed=${seed}"
            fail_count=$((fail_count + 1))
                        echo "跳过此实验，继续下一个..."
                        continue 2  # 跳出 seed 和当前 experiment 循环
        fi

                    echo "✅ 实验完成: ${config_name}/${exp_name} | ${model_key} | ${dataset_key} | seed=${seed}"
                    success_count=$((success_count + 1))
        echo "----------------------------------------"

                    # 清理显存
                    sleep 2
                    python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
                done
            done
        done

        echo ""
        echo "✅ 实验 ${exp_name} 完成"
    done

    echo ""
    echo "✅ 配置 ${config_name} 所有实验完成"
done

echo ""
echo "========================================"
echo "🎉 GPU0 所有 JORA Sweep 实验完成!"
echo ""
echo "实验统计:"
echo "  - 总配置数: ${total_configs}"
echo "  - 成功实验: ${success_count}"
echo "  - 失败实验: ${fail_count}"
echo "  - 跳过配置: ${skipped_count}"
echo ""
echo "输出目录: ${output_base_dir}"
echo "========================================"

# 清理临时配置文件
rm -rf "${temp_config_dir}"
echo ""
echo "🧹 临时配置文件已清理"

# 生成结果汇总 CSV
echo ""
echo "📋 生成结果汇总..."

summary_file="${output_base_dir}/sweep_summary.csv"
echo "experiment_name,config_file,model,dataset,seed,status,trainable_params" > "${summary_file}"

find "${output_base_dir}" -name "train.log" -type f 2>/dev/null | while read log_file; do
    dir_path=$(dirname "${log_file}")
    seed=$(basename "${dir_path}")
    dataset=$(basename "$(dirname "${dir_path}")")
    model=$(basename "$(dirname "$(dirname "${dir_path}")")")
    exp_name=$(basename "$(dirname "$(dirname "$(dirname "${dir_path}")")")")
    config=$(basename "$(dirname "$(dirname "$(dirname "$(dirname "${dir_path}")")")")")
    trainable_params=$(grep -oP "trainable params: \K[0-9,]+" "${log_file}" 2>/dev/null | tr -d ',' || echo "N/A")
    echo "${config}_${exp_name}_${model}_${dataset}_${seed},${config},${exp_name},${model},${dataset},${seed},completed,${trainable_params}" >> "${summary_file}"
done

echo ""
echo "📊 结果汇总已保存至: ${summary_file}"
echo ""
echo "总计完成实验数: ${success_count}"
