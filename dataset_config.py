#!/usr/bin/env python3
"""
PEFT-JORA 数据集配置文件
配置已下载的数据集路径和加载方式
"""

import os
from datasets import load_dataset

# 数据集根目录
DATASET_ROOT = "/home/jqh/Workshop/JORA/datasets"

# HuggingFace 镜像源（可选，用于加速下载）
HF_ENDPOINT = "https://hf-mirror.com"

# 已下载的数据集配置
AVAILABLE_DATASETS = {
    # GLUE 基准测试集
    "glue_sst2": {
        "name": "glue",
        "config": "sst2",
        "splits": ["train", "validation", "test"],
        "description": "GLUE SST-2 情感分析任务"
    },

    # HellaSwag 常识推理
    "hellaswag": {
        "name": "hellaswag",
        "config": None,
        "splits": ["train", "validation", "test"],
        "description": "HellaSwag 常识推理任务"
    },

    # GSM8K 数学推理
    "gsm8k": {
        "name": "openai/gsm8k",
        "config": "main",
        "splits": ["train", "test"],
        "description": "GSM8K 数学推理任务"
    },

    # ARC 科学问答
    "arc_challenge": {
        "name": "ai2_arc",
        "config": "ARC-Challenge",
        "splits": ["train", "validation", "test"],
        "description": "ARC-Challenge 科学问答任务"
    },

    # MMLU 各学科子集
    "mmlu_biology": {
        "name": "cais/mmlu",
        "config": "college_biology",
        "splits": ["test", "validation", "dev"],
        "description": "MMLU 大学生物学"
    },

    "mmlu_chemistry": {
        "name": "cais/mmlu",
        "config": "college_chemistry",
        "splits": ["test", "validation", "dev"],
        "description": "MMLU 大学化学"
    },

    "mmlu_cs": {
        "name": "cais/mmlu",
        "config": "college_computer_science",
        "splits": ["test", "validation", "dev"],
        "description": "MMLU 大学计算机科学"
    },

    "mmlu_math": {
        "name": "cais/mmlu",
        "config": "college_mathematics",
        "splits": ["test", "validation", "dev"],
        "description": "MMLU 大学数学"
    },

    "mmlu_physics": {
        "name": "cais/mmlu",
        "config": "college_physics",
        "splits": ["test", "validation", "dev"],
        "description": "MMLU 大学物理"
    },

    "mmlu_ee": {
        "name": "cais/mmlu",
        "config": "electrical_engineering",
        "splits": ["test", "validation", "dev"],
        "description": "MMLU 电气工程"
    },

    "mmlu_ml": {
        "name": "cais/mmlu",
        "config": "machine_learning",
        "splits": ["test", "validation", "dev"],
        "description": "MMLU 机器学习"
    },

    # Alpaca 指令微调数据集
    "alpaca_cleaned": {
        "name": "yahma/alpaca-cleaned",
        "config": None,
        "splits": ["train"],
        "description": "Alpaca指令微调数据集（清理版）"
    },
}


def load_peft_dataset(dataset_key, split="train", cache_dir=DATASET_ROOT):
    """
    加载已下载的数据集

    Args:
        dataset_key (str): 数据集键名 (如 'glue_sst2', 'gsm8k')
        split (str): 数据分割 ('train', 'validation', 'test')
        cache_dir (str): 缓存目录

    Returns:
        Dataset: HuggingFace Dataset 对象
    """
    if dataset_key not in AVAILABLE_DATASETS:
        raise ValueError(f"未知数据集: {dataset_key}. 可用数据集: {list(AVAILABLE_DATASETS.keys())}")

    config = AVAILABLE_DATASETS[dataset_key]
    if split not in config["splits"]:
        raise ValueError(f"数据集 {dataset_key} 没有 {split} 分割. 可用分割: {config['splits']}")

    # 设置环境变量
    os.environ['HF_DATASETS_CACHE'] = cache_dir
    if HF_ENDPOINT:
        os.environ['HF_ENDPOINT'] = HF_ENDPOINT

    # 加载数据集
    if config["config"]:
        dataset = load_dataset(config["name"], config["config"], split=split, cache_dir=cache_dir)
    else:
        dataset = load_dataset(config["name"], split=split, cache_dir=cache_dir)

    print(f"✓ 加载数据集: {dataset_key} ({split}) - {len(dataset)} 条数据")
    return dataset


def list_available_datasets():
    """列出所有可用的数据集"""
    print("=== 可用的数据集 ===")
    for key, config in AVAILABLE_DATASETS.items():
        print(f"{key}: {config['description']}")
        print(f"  分割: {config['splits']}")
    print()


if __name__ == "__main__":
    # 示例用法
    list_available_datasets()

    # 测试加载一个数据集
    print("测试加载示例:")
    try:
        ds = load_peft_dataset("glue_sst2", "train")
        print(f"样例数据: {ds[0]}")
    except Exception as e:
        print(f"加载失败: {e}")
