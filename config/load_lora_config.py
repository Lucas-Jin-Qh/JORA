#!/usr/bin/env python3
"""
LoRA配置加载和模型应用示例
演示如何使用配置文件加载LoRA并应用到Llama2-7B模型
"""

import json
import torch
from peft import PeftModel, LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_lora_config(config_path: str) -> LoraConfig:
    """从JSON配置文件加载LoRA配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)

    # 创建LoRA配置对象
    lora_config = LoraConfig(**config_dict)
    return lora_config


def load_model_with_lora(
    model_path: str,
    config_path: str,
    device_map: str = "auto"
) -> tuple:
    """加载模型并应用LoRA配置"""

    print(f"Loading LoRA config from: {config_path}")
    lora_config = load_lora_config(config_path)

    print(f"Loading base model from: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,  # 使用bf16节省显存
        device_map=device_map,
        trust_remote_code=True
    )

    print("Applying LoRA configuration...")
    model = PeftModel(model, lora_config)

    print("Model loaded successfully!")
    print("=" * 50)
    model.print_trainable_parameters()
    print("=" * 50)

    return model, lora_config


def main():
    """主函数示例"""

    # 配置路径
    MODEL_PATH = "/mnt/sda/jqh/pretrained_checkpoints/Llama-2-7b-hf/"
    CONFIG_PATH = "config/lora_llama2_7b_rank4.json"

    try:
        # 加载模型和配置
        model, lora_config = load_model_with_lora(MODEL_PATH, CONFIG_PATH)

        # 显示配置详情
        print("LoRA Configuration Details:")
        print(f"- Rank (r): {lora_config.r}")
        print(f"- Alpha: {lora_config.lora_alpha}")
        print(f"- Dropout: {lora_config.lora_dropout}")
        print(f"- Target Modules: {lora_config.target_modules}")
        print(f"- Task Type: {lora_config.task_type}")

        # 测试推理（可选）
        print("\nTesting inference...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

        # 确保tokenizer有pad_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        test_input = "Hello, I am a helpful AI assistant."
        inputs = tokenizer(test_input, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Input: {test_input}")
        print(f"Generated: {generated_text}")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
