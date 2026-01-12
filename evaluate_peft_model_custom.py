#!/usr/bin/env python3
"""
自定义评估PEFT模型的脚本
支持自定义max_seq_length等参数
"""

import os
import json
import argparse
import warnings
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import lm_eval
from lm_eval.models.huggingface import HFLM


def evaluate_peft_model_custom(
    base_model_path: str,
    checkpoint_dir: str,
    tasks: str = "mmlu",
    num_fewshot: int = 5,
    batch_size: str = "auto",
    max_seq_length: int = 2048,
    gpu_id: int = 0,
    adapter_type: str = "auto"  # "auto", "lora", "jora"
):
    """评估PEFT模型"""

    print("=== PEFT模型评估 ===")
    print(f"基础模型: {base_model_path}")
    print(f"Checkpoint目录: {checkpoint_dir}")
    print(f"评估任务: {tasks}")
    print(f"Few-shot: {num_fewshot}")
    print(f"Batch size: {batch_size}")
    print(f"Max seq length: {max_seq_length}")
    print(f"GPU ID: {gpu_id}")
    print(f"Adapter类型: {adapter_type}")
    print()

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
        task_name = tasks.replace(',', '_')
        return f"{checkpoint_dir}/{task_name}_{num_fewshot}shot_results.json"

    # 设置环境变量指定GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = f"cuda:0"  # 由于设置了CUDA_VISIBLE_DEVICES，总是使用cuda:0

    # 抑制已知的无害警告
    warnings.filterwarnings("ignore", message=".*not a valid Python identifier.*")
    warnings.filterwarnings("ignore", message=".*torch_dtype.*is deprecated.*")

    try:
        # 自动检测adapter类型
        if adapter_type == "auto":
            try:
                config_path = os.path.join(checkpoint_dir, "adapter_config.json")
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    detected_type = config.get("peft_type", "lora").lower()
                    print(f"检测到adapter类型: {detected_type}")
                    adapter_type = detected_type
                else:
                    print("未找到adapter_config.json，默认使用lora类型")
                    adapter_type = "lora"
            except Exception as e:
                print(f"检测adapter类型失败: {e}，默认使用lora类型")
                adapter_type = "lora"

        # 加载tokenizer
        print("加载tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)

        # 加载基础模型
        print("加载基础模型...")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map={"": device} if torch.cuda.is_available() else "cpu",
            dtype=torch.bfloat16  # 修复deprecation warning
        )

        # 加载PEFT权重
        print(f"加载{adapter_type.upper()}权重: {checkpoint_dir}")
        model = PeftModel.from_pretrained(model, checkpoint_dir)

        # 验证adapter类型
        if adapter_type == "jora":
            # 检查是否真的加载了JORA adapter
            has_jora = False
            jora_info = []

            for name, module in model.named_modules():
                if hasattr(module, '_jora_layers') or 'jora' in str(type(module)).lower():
                    has_jora = True
                    if hasattr(module, 'cfg'):
                        cfg = module.cfg
                        jora_info.append(f"S_L={cfg.S_L}, S_R={cfg.S_R}, k={cfg.k}, core={cfg.core}")

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

        # 抑制lm_eval关于已初始化模型的警告
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*pretrained.*model kwarg is not of type.*")
            warnings.filterwarnings("ignore", message=".*Passed an already-initialized model.*")

            # 创建HFLM包装器，支持自定义参数
            hf_model = HFLM(
                pretrained=model,
                tokenizer=tokenizer,
                batch_size=batch_size,
                max_length=max_seq_length,
                device=device,
                dtype=torch.bfloat16
            )

            # 使用lm_eval进行评估
            results = lm_eval.simple_evaluate(
                model=hf_model,
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
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="自定义评估PEFT模型")
    parser.add_argument("--base_model", required=True, help="基础模型路径")
    parser.add_argument("--checkpoint_dir", required=True, help="Checkpoint目录")
    parser.add_argument("--tasks", default="mmlu", help="评估任务")
    parser.add_argument("--num_fewshot", type=int, default=5, help="Few-shot数量")
    parser.add_argument("--batch_size", default="auto", help="Batch size")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="最大序列长度")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
    parser.add_argument("--adapter_type", default="auto", choices=["auto", "lora", "jora", "roda", "oft", "boft"], help="Adapter类型")

    args = parser.parse_args()

    evaluate_peft_model_custom(
        base_model_path=args.base_model,
        checkpoint_dir=args.checkpoint_dir,
        tasks=args.tasks,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        gpu_id=args.gpu_id,
        adapter_type=args.adapter_type
    )


if __name__ == "__main__":
    main()
