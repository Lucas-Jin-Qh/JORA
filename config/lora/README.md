# LoRA 实验配置文件

根据 TODO.md 实验设计生成的 LoRA 配置文件和训练命令。

## 📁 文件结构

### 配置文件 (16个)
命名规则: `lora_{model}_{dataset}_rank{rank}.json`

**模型**: `llama2_7b`, `mistral_7b`
**数据集**: `alpaca`, `gsm8k`
**Rank**: 4, 8, 16, 32

### 配置详情

- **目标模块**: `["q_proj","v_proj"]` (轨道A)
- **Alpha**: 2 × rank
- **Dropout**: 0.05
- **其他参数**: 参考 TODO.md 统一设置

## 🚀 训练命令

### 脚本位置
`scripts/run_lora_experiments.sh`

### 实验设置
- **每个配置**: 3个随机种子 (42, 1337, 2026)
- **总训练命令**: 16×3 = 48 个
- **学习率**:
  - alpaca-cleaned: 2e-4 (SFT-S)
  - gsm8k: 1e-4 (SFT-M)

### 运行方式
```bash
# 运行所有实验
bash scripts/run_lora_experiments.sh

# 或运行单个命令 (从脚本中复制)
CUDA_VISIBLE_DEVICES=1 python train_with_config.py \
    --model_path "/mnt/sda/jqh/pretrained_checkpoints/Llama-2-7b-hf/" \
    --dataset_name "yahma/alpaca-cleaned" \
    --config "config/lora/lora_llama2_7b_alpaca_rank4.json" \
    --output_dir "checkpoints/lora_llama2_7b_alpaca_rank4_seed42" \
    --num_epochs 3 \
    --batch_size 2 \
    --learning_rate 0.0002 \
    --execute --disable_wandb
```

## 📊 参数预算

| 配置 | 可训练参数 | 与 LoRA rank 对应关系 |
|------|-----------|----------------------|
| rank 4 | ~800K | 相当于 LoRA r≈4 |
| rank 8 | ~1.6M | 相当于 LoRA r≈8 |
| rank 16 | ~3.2M | 相当于 LoRA r≈16 |
| rank 32 | ~6.4M | 相当于 LoRA r≈32 |

## 🔍 注意事项

1. **轨道选择**: 使用轨道A `["q_proj","v_proj"]` 以确保公平对比
2. **参数匹配**: rank 选择与 TODO.md 中的 JORA 参数预算对应
3. **种子设置**: 每个配置使用不同随机种子确保结果可靠性
4. **输出目录**: 自动包含配置信息，便于后续分析

## 📈 预期结果

这些实验将为 JORA 方法提供 baseline 对比，帮助验证：
- JORA 的几何保持优势
- 不同模型架构对 PEFT 方法效果的影响
- 参数预算对性能的影响
