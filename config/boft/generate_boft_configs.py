#!/usr/bin/env python3
"""
生成 BOFT 配置文件
与 OFT 的 rank 参数形成对应关系，便于对比实验
"""

import json
import os

# BOFT 配置模板
def create_boft_config(model: str, dataset: str, block_size: int, butterfly_factor: int = 1):
    """
    创建 BOFT 配置文件
    
    Args:
        model: 模型名称 (llama2_7b, mistral_7b)
        dataset: 数据集名称 (alpaca, gsm8k)
        block_size: BOFT block_size
        butterfly_factor: 蝴蝶因子数 (默认为1，即标准OFT)
    """
    # 隐藏层维度
    hidden_dim = 4096
    
    # 计算 block_num
    block_num = hidden_dim // block_size
    
    # 有效等价 block (用于与 OFT rank 对比)
    effective_block = block_size * butterfly_factor
    
    config = {
        "peft_type": "BOFT",
        "auto_mapping": None,
        "base_model_name_or_path": None,
        "revision": None,
        "task_type": "CAUSAL_LM",
        "inference_mode": False,
        "boft_n_butterfly_factor": butterfly_factor,
        "boft_dropout": 0.0,
        "target_modules": [
            "q_proj",
            "v_proj"
        ],
        "fan_in_fan_out": False,
        "bias": "none",
        "modules_to_save": None,
        "init_weights": True,
        "layers_to_transform": None,
        "layers_pattern": None,
        "layer_replication": None,
        "runtime_config": {
            "ephemeral_gpu_offload": False
        },
        "boft_block_size": block_size,
        "boft_block_num": block_num
    }
    
    return config, effective_block


# 生成与 OFT 对应的配置
def generate_oft_equivalent_configs():
    """生成与 OFT rank=4,8,16,32 对应的 BOFT 配置"""
    
    models = ["llama2_7b", "mistral_7b"]
    datasets = ["alpaca", "gsm8k"]
    
    # OFT 等价配置 (butterfly_factor=1)
    oft_equivalents = [
        {"oft_rank": 4, "block_size": 4, "butterfly_factor": 1},
        {"oft_rank": 8, "block_size": 8, "butterfly_factor": 1},
        {"oft_rank": 16, "block_size": 16, "butterfly_factor": 1},
        {"oft_rank": 32, "block_size": 32, "butterfly_factor": 1},
    ]
    
    # 扩展 BOFT 配置 (使用 butterfly_factor > 1)
    boft_extended = [
        {"name": "bf4", "block_size": 4, "butterfly_factor": 4},   # 有效块=16
        {"name": "bf8", "block_size": 4, "butterfly_factor": 8},   # 有效块=32
        {"name": "bf8_2", "block_size": 8, "butterfly_factor": 2}, # 有效块=16
        {"name": "bf16_2", "block_size": 8, "butterfly_factor": 4}, # 有效块=32
    ]
    
    output_dir = "config/boft"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("生成 BOFT 配置文件")
    print("=" * 70)
    
    # 生成 OFT 等价配置
    print("\n📁 生成 OFT 等价配置 (butterfly_factor=1):")
    for model in models:
        for dataset in datasets:
            for equiv in oft_equivalents:
                config, effective_block = create_boft_config(
                    model, dataset, 
                    equiv["block_size"], 
                    equiv["butterfly_factor"]
                )
                filename = f"boft_{model}_{dataset}_rank{equiv['oft_rank']}.json"
                filepath = os.path.join(output_dir, filename)
                with open(filepath, 'w') as f:
                    json.dump(config, f, indent=2)
                print(f"  ✅ {filename} (有效块大小: {effective_block})")
    
    # 生成扩展 BOFT 配置
    print("\n📁 生成扩展 BOFT 配置 (butterfly_factor > 1):")
    for model in models:
        for dataset in datasets:
            for ext in boft_extended:
                config, effective_block = create_boft_config(
                    model, dataset, 
                    ext["block_size"], 
                    ext["butterfly_factor"]
                )
                filename = f"boft_{model}_{dataset}_{ext['name']}.json"
                filepath = os.path.join(output_dir, filename)
                with open(filepath, 'w') as f:
                    json.dump(config, f, indent=2)
                print(f"  ✅ {filename} (有效块大小: {effective_block})")
    
    print("\n" + "=" * 70)
    print("配置文件生成完成!")
    print("=" * 70)


if __name__ == "__main__":
    generate_oft_equivalent_configs()

