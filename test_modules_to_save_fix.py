#!/usr/bin/env python3
"""
测试JORA modules_to_save兼容性修复
"""

import sys
import os
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from peft.tuners.jora.config import JoraConfig
from peft.tuners.jora.model import JoraModel


def test_modules_to_save_compatibility():
    """测试modules_to_save的兼容性修复"""
    print("🧪 测试JORA modules_to_save兼容性修复...")

    # 创建测试模型
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(10, 20)
            self.linear2 = nn.Linear(20, 30)
            self.lm_head = nn.Linear(30, 1000)  # 模拟lm_head

        def forward(self, x):
            x = self.linear1(x)
            x = self.linear2(x)
            x = self.lm_head(x)
            return x

    base_model = TestModel()

    # 创建JORA配置，包含modules_to_save
    config = JoraConfig(
        target_modules=["linear1", "linear2"],
        modules_to_save=["lm_head"]  # lm_head应该保持可训练
    )

    # 创建JORA模型
    jora_model = JoraModel(base_model, config, "test")

    print("  配置检查:")
    print(f"    target_modules: {config.target_modules}")
    print(f"    modules_to_save: {config.modules_to_save}")

    # 记录初始参数状态
    initial_requires_grad = {}
    for name, param in jora_model.named_parameters():
        initial_requires_grad[name] = param.requires_grad

    print("\n  初始参数状态检查:")
    lm_head_params = [name for name in initial_requires_grad.keys() if 'lm_head' in name]
    jora_params = [name for name in initial_requires_grad.keys() if any(prefix in name for prefix in ['theta_L', 'theta_R', 'core', 'ecd_log_mag'])]
    other_params = [name for name in initial_requires_grad.keys() if name not in lm_head_params + jora_params]

    print(f"    lm_head参数数量: {len(lm_head_params)}")
    print(f"    JORA参数数量: {len(jora_params)}")
    print(f"    基础模型参数数量: {len(other_params)}")

    # 调用_mark_only_adapters_as_trainable
    print("\n  调用_mark_only_adapters_as_trainable...")
    jora_model._mark_only_adapters_as_trainable(jora_model.model)

    # 检查参数状态
    final_requires_grad = {}
    for name, param in jora_model.named_parameters():
        final_requires_grad[name] = param.requires_grad

    print("\n  最终参数状态检查:")

    # 检查lm_head参数（应该保持可训练）
    lm_head_trainable = [name for name in lm_head_params if final_requires_grad[name]]
    lm_head_frozen = [name for name in lm_head_params if not final_requires_grad[name]]

    print(f"    lm_head可训练参数: {len(lm_head_trainable)}")
    print(f"    lm_head冻结参数: {len(lm_head_frozen)}")

    if lm_head_frozen:
        print(f"    ❌ 冻结的lm_head参数: {lm_head_frozen[:3]}...")  # 只显示前3个
        return False

    # 检查JORA参数（应该可训练）
    jora_trainable = [name for name in jora_params if final_requires_grad[name]]
    jora_frozen = [name for name in jora_params if not final_requires_grad[name]]

    print(f"    JORA可训练参数: {len(jora_trainable)}")
    print(f"    JORA冻结参数: {len(jora_frozen)}")

    if jora_frozen:
        print(f"    ❌ 冻结的JORA参数: {jora_frozen[:3]}...")  # 只显示前3个
        return False

    # 检查基础模型参数（应该冻结，除了modules_to_save中的）
    base_model_trainable = [name for name in other_params if final_requires_grad[name]]

    print(f"    基础模型可训练参数: {len(base_model_trainable)}")

    if base_model_trainable:
        print(f"    ❌ 基础模型中不应该可训练的参数: {base_model_trainable[:3]}...")  # 只显示前3个
        return False

    print("✅ modules_to_save兼容性修复测试通过!")
    print("    - lm_head参数正确保持可训练")
    print("    - JORA参数正确保持可训练")
    print("    - 其他参数正确冻结")

    return True


def test_without_modules_to_save():
    """测试不使用modules_to_save的情况"""
    print("\n🧪 测试不使用modules_to_save的情况...")

    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(10, 20)
            self.linear2 = nn.Linear(20, 30)

        def forward(self, x):
            x = self.linear1(x)
            x = self.linear2(x)
            return x

    base_model = TestModel()

    # 创建JORA配置，不包含modules_to_save
    config = JoraConfig(
        target_modules=["linear1", "linear2"]
    )

    # 创建JORA模型
    jora_model = JoraModel(base_model, config, "test")

    # 调用_mark_only_adapters_as_trainable
    jora_model._mark_only_adapters_as_trainable(jora_model.model)

    # 检查所有参数状态
    trainable_params = []
    frozen_params = []
    for name, param in jora_model.named_parameters():
        if param.requires_grad:
            trainable_params.append(name)
        else:
            frozen_params.append(name)

    # JORA参数应该可训练，其他参数应该冻结
    jora_trainable = [name for name in trainable_params if any(prefix in name for prefix in ['theta_L', 'theta_R', 'core', 'ecd_log_mag'])]
    non_jora_trainable = [name for name in trainable_params if name not in jora_trainable]

    print(f"  可训练参数总数: {len(trainable_params)}")
    print(f"  JORA可训练参数: {len(jora_trainable)}")
    print(f"  非JORA可训练参数: {len(non_jora_trainable)}")

    # 不使用modules_to_save时，所有非JORA参数都应该被冻结
    if len(non_jora_trainable) == 0:
        print("✅ 不使用modules_to_save时正确冻结所有非JORA参数")
        return True
    else:
        print("❌ 存在不应该可训练的参数")
        print(f"  不应该可训练的参数: {non_jora_trainable[:3]}...")
        return False


if __name__ == '__main__':
    success1 = test_modules_to_save_compatibility()
    success2 = test_without_modules_to_save()

    overall_success = success1 and success2
    print(f"\n🏆 总体结果: {'通过' if overall_success else '失败'}")
    sys.exit(0 if overall_success else 1)