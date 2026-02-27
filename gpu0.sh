#!/bin/bash

# =============================================================================
# JORA 超参数扫描实验脚本 - GPU0
# 仅运行 JORA Sweep 中未完成的 38 个实验
# 已完成: 73 个实验
# 待运行: 38 个实验 (10 + 3 + 3 + 7 + 6 + 2 + 1 + 1 + 1 + 4)
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

# 已完成的实验 (跳过)
declare -A completed_experiments
# block_s_k_sweep (10 个已完成)
completed_experiments["block_s_k_sweep/block-S128-k24"]=1
completed_experiments["block_s_k_sweep/block-S128-k32"]=1
completed_experiments["block_s_k_sweep/block-S32-k16"]=1
completed_experiments["block_s_k_sweep/block-S32-k8"]=1
completed_experiments["block_s_k_sweep/block-S64-k16"]=1
completed_experiments["block_s_k_sweep/block-S64-k24"]=1
completed_experiments["block_s_k_sweep/block-S64-k8"]=1
completed_experiments["block_s_k_sweep/block-S96-k16"]=1
completed_experiments["block_s_k_sweep/block-S96-k24"]=1
completed_experiments["block_s_k_sweep/block-S96-k32"]=1
# block_size_sweep (已完成 0 个)
# diag_baseline (已完成 0 个)
# selection_refine (已完成 0 个)
# magnitude_compare (已完成 0 个)
# rotation_compare (已完成 0 个)
# ablation_study (已完成 8 个)
completed_experiments["ablation_study/abl-core-diag"]=1
completed_experiments["ablation_study/abl-core-lowrank"]=1
completed_experiments["ablation_study/abl-mag-ecd"]=1
completed_experiments["ablation_study/abl-no_mag"]=1
completed_experiments["ablation_study/abl-no_rotation"]=1
completed_experiments["ablation_study/abl-no_selection"]=1
completed_experiments["ablation_study/abl-random_sel"]=1
completed_experiments["ablation_study/abl-single_left"]=1
# lowrank_explore (已完成 3 个)
completed_experiments["lowrank_explore/lowrank-r16"]=1
completed_experiments["lowrank_explore/lowrank-r4"]=1
completed_experiments["lowrank_explore/lowrank-r8-S64"]=1
# asymmetric_scan (已完成 6 个)
completed_experiments["asymmetric_scan/asym-S32-S64"]=1
completed_experiments["asymmetric_scan/asym-S48-S96"]=1
completed_experiments["asymmetric_scan/asym-S64-S32"]=1
completed_experiments["asymmetric_scan/asym-S64-S64-k16"]=1
completed_experiments["asymmetric_scan/asym-S64-S64-k8"]=1
completed_experiments["asymmetric_scan/asym-S96-S48"]=1
# pairing_strategy_scan (已完成 3 个)
completed_experiments["pairing_strategy_scan/pair-consecutive"]=1
completed_experiments["pairing_strategy_scan/pair-consecutive-S64"]=1
completed_experiments["pairing_strategy_scan/pair-high_low"]=1
# learning_rate_scan (已完成 8 个)
completed_experiments["learning_rate_scan/lr-0.00-0.01"]=1
completed_experiments["learning_rate_scan/lr-0.01-0.01"]=1
completed_experiments["learning_rate_scan/lr-0.02-0.01"]=1
completed_experiments["learning_rate_scan/lr-0.05-0.005"]=1
completed_experiments["learning_rate_scan/lr-0.05-0.01"]=1
completed_experiments["learning_rate_scan/lr-0.05-0.02"]=1
completed_experiments["learning_rate_scan/lr-0.10-0.01"]=1
completed_experiments["learning_rate_scan/lr-0.10-0.02"]=1
# fine_block_size_scan (已完成 5 个)
completed_experiments["fine_block_size_scan/block_size-1"]=1
completed_experiments["fine_block_size_scan/block_size-12"]=1
completed_experiments["fine_block_size_scan/block_size-3"]=1
completed_experiments["fine_block_size_scan/block_size-5"]=1
completed_experiments["fine_block_size_scan/block_size-6"]=1
# all_linear_track (已完成 0 个)
# temperature_annealing (已完成 6 个)
completed_experiments["temperature_annealing/temp-anneal-t10-1"]=1
completed_experiments["temperature_annealing/temp-anneal-t3-0.5"]=1
completed_experiments["temperature_annealing/temp-anneal-t5-1"]=1
completed_experiments["temperature_annealing/temp-fixed1.0"]=1
completed_experiments["temperature_annealing/temp-fixed-t5.0"]=1
completed_experiments["temperature_annealing/temp-no_anneal-t2.0"]=1
# ema_grid_scan (已完成 10 个)
completed_experiments["ema_grid_scan/ema-b0.90-u1"]=1
completed_experiments["ema_grid_scan/ema-b0.90-u16"]=1
completed_experiments["ema_grid_scan/ema-b0.90-u50"]=1
completed_experiments["ema_grid_scan/ema-b0.95-u1"]=1
completed_experiments["ema_grid_scan/ema-b0.95-u16"]=1
completed_experiments["ema_grid_scan/ema-b0.95-u50"]=1
completed_experiments["ema_grid_scan/ema-b0.98-u1"]=1
completed_experiments["ema_grid_scan/ema-b0.98-u100"]=1
completed_experiments["ema_grid_scan/ema-b0.98-u16"]=1
completed_experiments["ema_grid_scan/ema-b0.98-u50"]=1
# dataset_comparison (已完成 5 个)
completed_experiments["dataset_comparison/dataset-alpaca"]=1
completed_experiments["dataset_comparison/dataset-alpaca_cleaned"]=1
completed_experiments["dataset_comparison/dataset-gsm8k_main"]=1
completed_experiments["dataset_comparison/dataset-open_orca"]=1
completed_experiments["dataset_comparison/dataset-slimorca"]=1
# model_comparison (已完成 4 个)
completed_experiments["model_comparison/model-llama2-7b"]=1
completed_experiments["model_comparison/model-mistral-7b"]=1
completed_experiments["model_comparison/model-mistral-instruct"]=1
completed_experiments["model_comparison/model-qwen2-7b"]=1
# batch_lr_interaction (已完成 6 个)
completed_experiments["batch_lr_interaction/bs2-lr3e-4"]=1
completed_experiments["batch_lr_interaction/bs4-lr1e-4"]=1
completed_experiments["batch_lr_interaction/bs4-lr2e-4"]=1
completed_experiments["batch_lr_interaction/bs4-lr3e-4"]=1
completed_experiments["batch_lr_interaction/bs8-lr1e-4"]=1
completed_experiments["batch_lr_interaction/bs8-lr2e-4"]=1
# advanced_ablation (已完成 8 个)
completed_experiments["advanced_ablation/group-by-type"]=1
completed_experiments["advanced_ablation/group-size1"]=1
completed_experiments["advanced_ablation/group-size2"]=1
completed_experiments["advanced_ablation/group-size4"]=1
completed_experiments["advanced_ablation/gumbel-disable"]=1
completed_experiments["advanced_ablation/gumbel-enable-tau0.5"]=1
completed_experiments["advanced_ablation/gumbel-enable-tau1.0"]=1
completed_experiments["advanced_ablation/gumbel-enable-tau2.0"]=1
# main_config (已完成)
completed_experiments["main_config/main_config"]=1

# JORA Sweep 配置文件列表 (20 个配置，111 个实验)
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
echo "已完成的实验: 73 个"
echo "待运行的实验: 38 个"
echo "配置目录: ${sweep_config_dir}"
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

# 统计变量
success_count=0
fail_count=0
skipped_count=0

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
"
}

# 遍历所有 JORA Sweep 配置
for config_file in "${sweep_configs[@]}"; do
    config_path="${sweep_config_dir}/${config_file}"

    # 检查配置文件是否存在
    if [ ! -f "${config_path}" ]; then
        echo ""
        echo "⚠️  配置文件不存在，跳过: ${config_path}"
        continue
    fi

    config_name=$(basename "${config_file}" .json)

    echo ""
    echo "========================================"
    echo "配置: ${config_file}"
    echo "========================================"

    # 检测配置格式并获取实验列表
    config_format=$(python3 -c "
import json
with open('${config_path}', 'r') as f:
    config = json.load(f)
print('SINGLE' if 'peft_type' in config else 'MULTI')
" 2>/dev/null)

    if [ "$config_format" = "SINGLE" ]; then
        experiments=("$config_name")
        echo "📄 格式: 单配置"
    else
        echo "📄 格式: 多实验 (experiments数组)"
        IFS=$'\n' read -r -d '' -a experiments < <(python3 -c "
import json
with open('${config_path}', 'r') as f:
    config = json.load(f)
for exp in config.get('experiments', []):
    print(exp.get('name', 'unknown'))
" 2>/dev/null && printf '\0')
    fi

    echo "🔬 实验数: ${#experiments[@]}"

    # 遍历每个实验
    for exp_name in "${experiments[@]}"; do
        # 检查是否已完成
        exp_key="${config_name}/${exp_name}"
        if [ -n "${completed_experiments[$exp_key]}" ]; then
            echo "⏭️  已完成，跳过: ${exp_name}"
            skipped_count=$((skipped_count + 1))
            continue
        fi

        # 为每个实验创建临时配置文件
        temp_config_file="${temp_config_dir}/${config_name}_${exp_name}.json"

        if [ "$config_format" = "MULTI" ]; then
            merge_and_create_temp_config "$config_path" "$exp_name" "$temp_config_file"
            actual_config="$temp_config_file"
        else
            actual_config="$config_path"
        fi

        echo "🔄 运行: ${exp_name} ..."

        # 遍历两个模型 (mistral 优先)
        for model_key in "mistral_7b" "llama2_7b"; do
            model_path="${model_paths[$model_key]}"

            if [ ! -d "${model_path}" ]; then
                echo "⚠️  模型路径不存在，跳过: ${model_path}"
                continue
            fi

            # 遍历两个数据集 (alpaca 优先)
            for dataset_key in "alpaca" "gsm8k"; do
                dataset_name="${dataset_names[${model_key}_${dataset_key}]}"

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
                        continue 2
                    fi

                    echo "✅ 实验完成"
                    success_count=$((success_count + 1))

                    # 清理显存
                    sleep 2
                    python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
                done
            done
        done

        echo "✅ 实验 ${exp_name} 完成"
    done

    echo ""
    echo "✅ 配置 ${config_name} 所有实验完成"
done

# 清理临时配置文件
rm -rf "${temp_config_dir}"

echo ""
echo "========================================"
echo "🎉 GPU0 JORA Sweep 实验完成!"
echo ""
echo "实验统计:"
echo "  - 成功实验: ${success_count}"
echo "  - 失败实验: ${fail_count}"
echo "  - 跳过(已完成): ${skipped_count}"
echo "  - 输出目录: ${output_base_dir}"
echo "========================================"
