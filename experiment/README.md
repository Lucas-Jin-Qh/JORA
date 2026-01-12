# 实验结果整理系统使用说明

## 新的实验目录结构
experiment/
├── llama2_7b/
│   ├── lora/
│   │   ├── rank4/
│   │   │   └── mmlu_0shot.json        # 简洁文件名
│   │   └── rank8/
│   │       └── mmlu_0shot.json
│   └── jora/
│       ├── s16_k4_diag/
│       │   └── mmlu_0shot.json        # 配置文件通过文件夹体现
│       └── s32_k8_diag/
│           └── mmlu_0shot.json
└── mistral_7b/
    └── lora/
        └── rank4/
            └── mmlu_0shot.json

## 使用方法

### 1. 评测时指定实验目录
python evaluate_peft_model_custom.py \
    --base_model "/mnt/sda/jqh/pretrained_checkpoints/Llama-2-7b-hf/" \
    --checkpoint_dir "/home/jqh/Workshop/JORA/experiment/llama2_7b/lora/rank4_alpaca" \
    --tasks "mmlu" \
    --num_fewshot 0 \
    --batch_size "auto" \
    --max_seq_length 2048 \
    --gpu_id 0 \
    --adapter_type lora

### 2. 自动生成的文件结构
- 输入: /experiment/llama2_7b/lora/rank4_alpaca
- 输出: /experiment/llama2_7b/lora/rank4/mmlu_0shot.json

### 3. 配置文件解析
脚本会自动从adapter_config.json或目录名中提取配置信息：
- LoRA: rank4 → rank4/
- JORA: s16_k4_diag → s16_k4_diag/

