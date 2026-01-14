#!/usr/bin/env python3
"""
使用PEFT配置文件进行训练的包装脚本
从config目录读取PEFT配置文件，自动生成训练命令
"""

import json
import os
import argparse
from pathlib import Path


def load_peft_config(config_path: str) -> dict:
    """加载PEFT配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config


def create_training_command(
    model_path: str,
    dataset_name: str,
    peft_config_path: str,
    output_dir: str,
    num_epochs: int = 1,
    batch_size: int = 4,
    learning_rate: float = 0.01,
    max_length: int = 2048,
    seed: int = 42,
    use_4bit: bool = False,
    use_nested_quant: bool = False,
    push_to_hub: bool = False,
    eval_steps: int = None,
    eval_strategy: str = "no",
    torch_dtype: str = "auto"
) -> str:
    """从PEFT配置文件生成训练命令"""

    # 加载PEFT配置
    peft_config = load_peft_config(peft_config_path)
    peft_type = peft_config.get('peft_type', 'LORA')

    # Auto-detect optimal dtype for flash attention
    if torch_dtype == "auto" and peft_type == "JORA":
        torch_dtype = "bfloat16"  # JORA uses flash attention, needs efficient dtype

    # 构建基础命令
    cmd_parts = [
        "python examples/sft/train.py",
        f"--seed {seed}",
        f"--model_name_or_path {model_path}",
        f"--dataset_name {dataset_name}",
        "--chat_template_format none",
        "--add_special_tokens False",
        "--append_concat_token False",
        "--splits train",
        f"--torch_dtype {torch_dtype}",
        f"--num_train_epochs {num_epochs}",
        "--logging_steps 10",
        "--log_level info",
        "--logging_strategy steps",
        f"--eval_strategy {eval_strategy}",
        "--save_strategy epoch",  # 每epoch保存一次checkpoint
        "--save_total_limit 3"    # 只保留最新的3个checkpoint
    ]

    # 添加评估步骤（如果启用）
    if eval_strategy != "no" and eval_steps is not None:
        cmd_parts.append(f"--eval_steps {eval_steps}")

    cmd_parts.extend([
        "--packing False",
        f"--learning_rate {learning_rate}",
        "--lr_scheduler_type cosine",
        "--weight_decay 0.01",
        "--warmup_ratio 0.03",
        "--max_grad_norm 1.0",
        f"--output_dir {output_dir}",
        f"--per_device_train_batch_size {batch_size}",
        "--gradient_accumulation_steps 4",
        "--gradient_checkpointing",
        "--use_reentrant",
        "--dataset_text_field text"
    ])

    # 根据PEFT类型添加相应参数
    if peft_type == "JORA":
        cmd_parts.extend([
            "--use_peft_jora",
            f"--jora_s_l {peft_config['S_L']}",
            f"--jora_s_r {peft_config['S_R']}",
            f"--jora_k {peft_config['k']}",
            f"--jora_rotation_param {peft_config['rotation_param']}",
            f"--jora_selection_type {peft_config['selection']}",
            f"--jora_magnitude {peft_config['magnitude']}"
        ])

        # 添加可选的性能优化参数（如果配置文件中有）
        if 'update_interval' in peft_config:
            cmd_parts.append(f"--jora_update_interval {peft_config['update_interval']}")
        if 'ema_update_interval' in peft_config:
            cmd_parts.append(f"--jora_ema_update_interval {peft_config['ema_update_interval']}")
        if 'selection_group_size' in peft_config:
            cmd_parts.append(f"--jora_selection_group_size {peft_config['selection_group_size']}")
        if 'selection_group_by' in peft_config:
            cmd_parts.append(f"--jora_selection_group_by {peft_config['selection_group_by']}")
        # JORA使用LoRA的目标模块参数名
        if 'target_modules' in peft_config:
            cmd_parts.append(f"--lora_target_modules {','.join(peft_config['target_modules'])}")
    elif peft_type == "LORA":
        cmd_parts.extend([
            "--use_peft_lora",
            f"--lora_r {peft_config['r']}",
            f"--lora_alpha {peft_config['lora_alpha']}",
            f"--lora_dropout {peft_config['lora_dropout']}",
            f"--lora_target_modules {','.join(peft_config['target_modules'])}"
        ])

    # cmd_parts.append("--use_flash_attn True")  # 需要额外安装flash_attn包

    # 添加量化相关参数（如果启用）
    if use_4bit:
        cmd_parts.extend([
            "--use_4bit_quantization",
            "--use_nested_quant",
            "--bnb_4bit_compute_dtype bfloat16"
        ])

    # 添加Hub相关参数（如果启用）
    if push_to_hub:
        cmd_parts.extend([
            "--push_to_hub",
            "--hub_private_repo",
            "--hub_strategy every_save"
        ])

    return " \\\n    ".join(cmd_parts)


def main():
    parser = argparse.ArgumentParser(description="使用PEFT配置文件生成训练命令")
    parser.add_argument("--model_path", required=True, help="模型路径")
    parser.add_argument("--dataset_name", required=True, help="数据集名称")
    parser.add_argument("--config", required=True, help="PEFT配置文件路径")
    parser.add_argument("--output_dir", required=True, help="输出目录")
    parser.add_argument("--num_epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="学习率")
    parser.add_argument("--max_length", type=int, default=2048, help="最大序列长度")
    parser.add_argument("--use_4bit", action="store_true", help="启用4bit量化")
    parser.add_argument("--use_nested_quant", action="store_true", help="启用嵌套量化")
    parser.add_argument("--push_to_hub", action="store_true", help="推送结果到Hub")
    parser.add_argument("--torch_dtype", type=str, default="auto", help="模型精度(auto/float16/bfloat16/float32)")
    parser.add_argument("--execute", action="store_true", help="直接执行命令")

    args = parser.parse_args()

    # 生成命令
    command = create_training_command(
        model_path=args.model_path,
        dataset_name=args.dataset_name,
        peft_config_path=args.config,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        torch_dtype=args.torch_dtype,
        max_length=args.max_length,
        use_4bit=args.use_4bit,
        use_nested_quant=args.use_nested_quant,
        push_to_hub=args.push_to_hub
    )

    print("=== 生成的训练命令 ===")
    print(command)
    print("\n=== 配置文件信息 ===")

    # 显示配置文件内容
    config = load_peft_config(args.config)
    peft_type = config.get('peft_type', 'LORA')

    if peft_type == "JORA":
        print(f"JORA S_L: {config['S_L']}")
        print(f"JORA S_R: {config['S_R']}")
        print(f"JORA k: {config['k']}")
        print(f"JORA Rotation: {config['rotation_param']}")
        print(f"JORA Selection: {config['selection']}")
        print(f"JORA Magnitude: {config['magnitude']}")
    elif peft_type == "LORA":
        print(f"LoRA Rank: {config['r']}")
        print(f"LoRA Alpha: {config['lora_alpha']}")
        print(f"Dropout: {config['lora_dropout']}")

    print(f"Target Modules: {len(config['target_modules'])} 个")
    print(f"PEFT Type: {peft_type}")

    if args.execute:
        print("\n=== 开始执行训练 ===")
        os.system(command)
    else:
        print("\n=== 复制以上命令手动执行 ===")


if __name__ == "__main__":
    main()
