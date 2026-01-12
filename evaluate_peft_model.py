#!/usr/bin/env python3
"""
评估PEFT模型的脚本
默认评估最新保存的checkpoint
"""

import os
import json
import argparse
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import lm_eval


def find_latest_checkpoint(checkpoint_dir: str) -> str:
    """查找最新的checkpoint"""
    if not os.path.exists(checkpoint_dir):
        raise ValueError(f"Checkpoint目录不存在: {checkpoint_dir}")

    # 查找所有checkpoint目录
    checkpoints = []
    for item in os.listdir(checkpoint_dir):
        if item.startswith("checkpoint-"):
            checkpoints.append(item)

    if not checkpoints:
        # 如果没有checkpoint目录，检查是否有adapter文件
        adapter_files = ["adapter_model.bin", "adapter_model.safetensors"]
        for file in adapter_files:
            if os.path.exists(os.path.join(checkpoint_dir, file)):
                return checkpoint_dir
        raise ValueError(f"在 {checkpoint_dir} 中没有找到checkpoint")

    # 按编号排序，找到最新的
    checkpoints.sort(key=lambda x: int(x.split("-")[1]))
    latest_checkpoint = os.path.join(checkpoint_dir, checkpoints[-1])
    print(f"使用最新checkpoint: {latest_checkpoint}")
    return latest_checkpoint


def evaluate_peft_model(
    base_model_path: str,
    checkpoint_dir: str,
    tasks: str = "mmlu",
    num_fewshot: int = 5,
    batch_size: int = 4,
    adapter_type: str = "auto"
):
    """评估PEFT模型"""

    # 生成实验结果输出路径的函数
    def generate_experiment_output_path(checkpoint_dir, tasks, num_fewshot, adapter_type):
        """生成实验结果输出路径"""
        import os

        # 检查是否是实验目录
        if 'experiment' in checkpoint_dir:
            # 解析checkpoint目录路径
            parts = checkpoint_dir.split('/')
            experiment_idx = -1
            for i, part in enumerate(parts):
                if part == 'experiment':
                    experiment_idx = i
                    break

            if experiment_idx >= 0 and len(parts) >= experiment_idx + 3:
                try:
                    base_experiment_dir = '/'.join(parts[:experiment_idx + 1])  # /path/to/experiment
                    model_name = parts[experiment_idx + 1]  # e.g., 'llama2_7b'
                    method_name = parts[experiment_idx + 2]  # e.g., 'lora', 'jora'

                    # 从adapter_config.json中提取配置信息来创建配置目录
                    config_str = ""
                    if adapter_type in ['lora', 'jora']:
                        try:
                            config_file = os.path.join(checkpoint_dir, 'adapter_config.json')
                            if os.path.exists(config_file):
                                with open(config_file, 'r') as f:
                                    adapter_config = json.load(f)

                                if adapter_type == 'lora':
                                    rank = adapter_config.get('r', adapter_config.get('rank', 'unknown'))
                                    config_str = f"rank{rank}"
                                elif adapter_type == 'jora':
                                    S_L = adapter_config.get('S_L', 'unknown')
                                    S_R = adapter_config.get('S_R', 'unknown')
                                    k = adapter_config.get('k', 'unknown')
                                    core = adapter_config.get('core', 'unknown')
                                    config_str = f"s{S_L}_k{k}_{core}"
                        except Exception as e:
                            print(f"Warning: Could not parse adapter config: {e}")
                            config_str = "unknown"

                    # 如果无法获取配置信息，尝试从目录名解析
                    if not config_str or config_str == "unknown":
                        checkpoint_basename = os.path.basename(checkpoint_dir)
                        # 从目录名提取配置信息
                        if adapter_type == 'lora' and 'rank' in checkpoint_basename:
                            # e.g., "rank4_alpaca" -> "rank4"
                            import re
                            rank_match = re.search(r'rank(\d+)', checkpoint_basename)
                            if rank_match:
                                config_str = f"rank{rank_match.group(1)}"
                        elif adapter_type == 'jora':
                            # e.g., "s16_k4_diag_alpaca" -> "s16_k4_diag"
                            # 移除常见的后缀
                            config_candidate = checkpoint_basename
                            for suffix in ['_alpaca', '_cleaned', '_chat']:
                                if config_candidate.endswith(suffix):
                                    config_candidate = config_candidate[:-len(suffix)]
                                    break
                            if config_candidate != checkpoint_basename:
                                config_str = config_candidate

                    # 如果还是无法确定，使用目录名
                    if not config_str:
                        config_str = os.path.basename(checkpoint_dir)

                    # 构建实验目录路径: experiment/{model}/{method}/{config}/
                    experiment_dir = f"{base_experiment_dir}/{model_name}/{method_name}/{config_str}"

                    # 确保目录存在
                    os.makedirs(experiment_dir, exist_ok=True)

                    # 生成简洁文件名
                    task_name = tasks.replace(',', '_')
                    filename = f"{task_name}_{num_fewshot}shot.json"
                    return f"{experiment_dir}/{filename}"

                except (IndexError, ValueError) as e:
                    print(f"Warning: Could not parse experiment path: {e}")
                    pass  # 回退到默认命名

        # 默认命名方式（保持兼容性）
        return f"{checkpoint_dir}/eval_results_{num_fewshot}shot.json"

    # 自动检测adapter类型
    if adapter_type == "auto":
        try:
            config_path = os.path.join(checkpoint_dir, "adapter_config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                detected_type = config.get("peft_type", "lora").lower()
                adapter_type = detected_type
            else:
                adapter_type = "lora"
        except Exception:
            adapter_type = "lora"

    print("=== PEFT模型评估 ===")
    print(f"基础模型: {base_model_path}")
    print(f"Checkpoint目录: {checkpoint_dir}")
    print(f"评估任务: {tasks}")
    print(f"Few-shot: {num_fewshot}")
    print(f"Batch size: {batch_size}")
    print(f"Adapter类型: {adapter_type}")
    print()

    # 查找最新checkpoint
    peft_model_path = find_latest_checkpoint(checkpoint_dir)

    # 设置设备
    device = torch.accelerator.current_accelerator().type if hasattr(torch, "accelerator") else "cuda"

    try:
        # 加载tokenizer
        print("加载tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)

        # 加载基础模型
        print("加载基础模型...")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

        # 加载PEFT权重
        print(f"加载{adapter_type.upper()}权重: {peft_model_path}")
        model = PeftModel.from_pretrained(model, peft_model_path)

        # 验证adapter类型
        if adapter_type == "jora":
            # 检查是否真的加载了JORA adapter
            has_jora = False
            jora_info = []

            for name, module in model.named_modules():
                # 检查JORA特有的属性或类名
                if 'jora' in str(type(module)).lower() or hasattr(module, 'cfg'):
                    if hasattr(module, 'cfg'):
                        cfg = module.cfg
                        if hasattr(cfg, 'S_L'):  # JORA特有的配置
                            has_jora = True
                            jora_info.append(f"S_L={cfg.S_L}, S_R={cfg.S_R}, k={cfg.k}")

            if has_jora:
                print(f"✅ 成功加载JORA adapter: {jora_info[:3]}...")  # 只显示前3个
            else:
                print("⚠️ 未检测到JORA adapter，但配置文件显示为JORA类型")
        elif adapter_type in ["lora", "roda", "oft", "boft"]:
            print(f"✅ 成功加载{adapter_type.upper()} adapter")

        # 确保tokenizer有pad_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print("开始评估...")

        # 使用lm_eval进行评估
        results = lm_eval.simple_evaluate(
            model=model,
            tasks=tasks,
            num_fewshot=num_fewshot,
            batch_size=batch_size,
            device=device,
            log_samples=False
        )

        print("\\n=== 评估结果 ===")
        for task, metrics in results["results"].items():
            print(f"\\n{task}:")
            for metric, value in metrics.items():
                if isinstance(value, float):
                    print(".4f")
                else:
                    print(f"  {metric}: {value}")

        # 保存结果 - 处理不可序列化的对象
        output_file = generate_experiment_output_path(
            checkpoint_dir, tasks, num_fewshot, adapter_type
        )

        # 递归清理结果字典中的不可序列化对象
        def clean_for_json(obj):
            if obj is None:
                return None
            elif isinstance(obj, (int, float, str, bool)):  # 基本JSON类型保持不变
                return obj
            elif isinstance(obj, dict):
                return {key: clean_for_json(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [clean_for_json(item) for item in obj]
            elif isinstance(obj, tuple):
                return [clean_for_json(item) for item in obj]  # Convert tuple to list
            elif hasattr(obj, 'item'):  # PyTorch tensors/scalars
                try:
                    return obj.item()  # For 0-dim tensors
                except RuntimeError:
                    return obj.tolist()  # For multi-dim tensors
            elif hasattr(obj, 'dtype'):  # PyTorch dtypes
                return str(obj.dtype)
            elif hasattr(obj, '__name__'):  # Functions, classes
                return obj.__name__
            elif hasattr(obj, '__class__'):  # Other complex objects
                return f"{obj.__class__.__module__}.{obj.__class__.__name__}"
            else:
                # Try to serialize directly, fallback to string representation
                try:
                    json.dumps(obj)
                    return obj
                except (TypeError, ValueError):
                    return str(obj)

        cleaned_results = clean_for_json(results)

        with open(output_file, 'w') as f:
            json.dump(cleaned_results, f, indent=2)
        print(f"\\n结果已保存到: {output_file}")

        return results

    except Exception as e:
        print(f"评估失败: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="评估PEFT模型")
    parser.add_argument("--base_model", required=True, help="基础模型路径")
    parser.add_argument("--checkpoint_dir", required=True, help="Checkpoint目录")
    parser.add_argument("--tasks", default="mmlu", help="评估任务")
    parser.add_argument("--num_fewshot", type=int, default=5, help="Few-shot数量")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--adapter_type", default="auto", choices=["auto", "lora", "jora", "roda", "oft", "boft"], help="Adapter类型")

    args = parser.parse_args()

    evaluate_peft_model(
        base_model_path=args.base_model,
        checkpoint_dir=args.checkpoint_dir,
        tasks=args.tasks,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        adapter_type=args.adapter_type
    )


if __name__ == "__main__":
    main()
