# PEFT配置文件说明

## 📁 配置文件目录

此目录包含各种PEFT方法的配置文件，用于不同的模型和任务。

## 🔧 LoRA Llama2-7B Rank4 配置

### 文件：`lora_llama2_7b_rank4.json`

**适用场景：** 微调Llama2-7B模型，使用LoRA方法，rank=4

### 参数说明：

| 参数 | 值 | 说明 |
|-----|-----|------|
| `peft_type` | `"LORA"` | PEFT方法类型 |
| `task_type` | `"CAUSAL_LM"` | 任务类型：因果语言模型 |
| `r` | `4` | LoRA秩（rank）- 较低的秩减少参数 |
| `target_modules` | `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]` | Llama模型的注意力层和FFN层 |
| `lora_alpha` | `8` | LoRA缩放因子（通常为rank的2倍） |
| `lora_dropout` | `0.1` | LoRA dropout概率 |
| `bias` | `"none"` | 不训练bias参数 |
| `inference_mode` | `true` | 推理模式优化 |

### 使用示例：

```python
from peft import PeftModel, LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

# 加载配置文件
with open('config/lora_llama2_7b_rank4.json', 'r') as f:
    config_dict = json.load(f)

# 创建LoRA配置
lora_config = LoraConfig(**config_dict)

# 加载基础模型
model = AutoModelForCausalLM.from_pretrained(
    "/mnt/sda/jqh/pretrained_checkpoints/Llama-2-7b-hf/",
    device_map="auto"
)

# 应用LoRA
model = PeftModel(model, lora_config)

# 查看可训练参数
model.print_trainable_parameters()
```

### 参数调优建议：

#### **Rank选择：**
- **Rank 4**: 最小的参数量，适合资源受限场景
- **Rank 8**: 平衡性能和效率的推荐选择
- **Rank 16**: 更好的性能，参数量适中

#### **Alpha设置：**
- 通常设置为rank的2倍：`alpha = 2 * rank`
- 可以根据任务调整：更难的任务可以使用更高的alpha

#### **Target Modules：**
- **注意力层**: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- **前馈网络**: `gate_proj`, `up_proj`, `down_proj`
- **嵌入层**: 可选择性添加 `embed_tokens`, `lm_head`

### 其他配置文件模板：

后续可以根据需要添加更多配置文件：
- 不同rank的LoRA配置
- 其他PEFT方法（BoFT, OFT, IA3等）
- 不同任务类型的配置

## 🚀 使用方法

### 方法1：使用训练包装脚本（推荐）

```bash
# 使用PEFT配置文件自动生成训练命令
python train_with_config.py \
    --model_path "/mnt/sda/jqh/pretrained_checkpoints/Llama-2-7b-hf/" \
    --dataset_name "yahma/alpaca-cleaned" \
    --config "config/lora_llama2_7b_rank4.json" \
    --output_dir "checkpoints/llama2_7b_lora_rank4_alpaca" \
    --num_epochs 3

# 直接执行训练（添加--execute参数）
python train_with_config.py \
    --model_path "/mnt/sda/jqh/pretrained_checkpoints/Llama-2-7b-hf/" \
    --dataset_name "yahma/alpaca-cleaned" \
    --config "config/lora_llama2_7b_rank4.json" \
    --output_dir "checkpoints/llama2_7b_lora_rank4_alpaca" \
    --num_epochs 3 \
    --execute
```

### 方法2：手动执行生成的命令

```bash
# 复制上述脚本生成的命令手动执行
python examples/sft/train.py \
    --seed 42 \
    --model_name_or_path "/mnt/sda/jqh/pretrained_checkpoints/Llama-2-7b-hf/" \
    --dataset_name "yahma/alpaca-cleaned" \
    --chat_template_format "none" \
    --add_special_tokens False \
    --append_concat_token False \
    --splits "train" \
    --max_length 2048 \
    --num_train_epochs 3 \
    --logging_steps 10 \
    --log_level "info" \
    --logging_strategy "steps" \
    --eval_strategy "no" \
    --save_strategy "epoch" \
    --bf16 True \
    --packing False \
    --learning_rate 0.0002 \
    --lr_scheduler_type "cosine" \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1.0 \
    --output_dir "checkpoints/llama2_7b_lora_rank4_alpaca" \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing True \
    --use_reentrant True \
    --dataset_text_field "text" \
    --use_peft_lora True \
    --lora_r 4 \
    --lora_alpha 8 \
    --lora_dropout 0.1 \
    --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
    --use_flash_attn True
```

## 📋 注意事项：

1. **模型兼容性**: 此配置专门针对Llama2-7B模型优化
2. **内存使用**: rank=4是最节省内存的设置
3. **性能平衡**: 在参数效率和任务性能间取得平衡
4. **扩展性**: 可以作为模板修改用于其他Llama模型
5. **配置文件**: 训练时使用命令行参数，配置文件主要用于保存和加载模型
