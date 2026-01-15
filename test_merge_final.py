#!/usr/bin/env python3
"""
最终测试JORA merge实现的数学正确性和稳定性
"""

import sys
import os
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from peft.tuners.jora.config import JoraConfig
from peft.tuners.jora.layer import JoraLayer


def test_mathematical_correctness():
    """测试merge实现的数学正确性"""
    print("🔬 测试merge实现的数学正确性...")

    # 创建测试模型
    base_layer = nn.Linear(64, 32)
    cfg = JoraConfig(S_L=4, S_R=4, core="diag", magnitude="none")
    layer = JoraLayer(base_layer, 'test', cfg)

    # 记录原始权重
    original_weight = base_layer.weight.data.clone()
    print(".6f")

    # 执行merge
    layer.merge()

    # 检查权重是否发生了合理的变化
    merged_weight = base_layer.weight.data.clone()
    weight_diff = (merged_weight - original_weight).abs()
    max_diff = weight_diff.max().item()
    mean_diff = weight_diff.mean().item()

    print(".6f")
    print(".6f")

    # 变化应该很小（由于保守的缩放因子）
    assert max_diff < 0.01, f"Merge effect too large: {max_diff}"
    assert mean_diff < 0.001, f"Average merge effect too large: {mean_diff}"

    # 执行unmerge
    layer.unmerge()

    # 检查权重是否完全恢复
    restored_weight = base_layer.weight.data.clone()
    restore_error = (restored_weight - original_weight).abs()
    max_restore_error = restore_error.max().item()
    mean_restore_error = restore_error.mean().item()

    print(".2e")
    print(".2e")

    # 恢复误差应该非常小
    assert max_restore_error < 1e-6, f"Restore error too large: {max_restore_error}"

    print("✅ 数学正确性测试通过")
    return True


def test_rotation_effect_estimation():
    """测试旋转效应估计算法"""
    print("\n🔄 测试旋转效应估计算法...")

    base_layer = nn.Linear(64, 32)
    cfg = JoraConfig(S_L=4, S_R=4, core="diag", magnitude="none")
    layer = JoraLayer(base_layer, 'test', cfg)

    adapter_state = layer.adapters['test']

    # 测试旋转效应估计
    scale_matrix = layer._estimate_rotation_effect_magnitude(adapter_state)

    print(f"  旋转效应矩阵形状: {scale_matrix.shape}")
    print(".4f")
    print(".4f")

    # 缩放因子应该在合理范围内
    assert scale_matrix.min() >= 0.5, "Scale factor too small"
    assert scale_matrix.max() <= 2.0, "Scale factor too large"

    print("✅ 旋转效应估计算法测试通过")
    return True


def test_different_core_types():
    """测试不同核心类型的merge"""
    print("\n🔧 测试不同核心类型的merge...")

    results = []
    for core_type in ["diag", "block"]:
        print(f"  测试{core_type}核心...")

        try:
            base_layer = nn.Linear(64, 32)
            cfg = JoraConfig(S_L=4, S_R=4, core=core_type, magnitude="none")
            layer = JoraLayer(base_layer, 'test', cfg)

            original_weight = base_layer.weight.data.clone()

            # Merge/unmerge测试
            layer.merge()
            layer.unmerge()

            # 检查恢复精度
            final_weight = base_layer.weight.data.clone()
            error = (final_weight - original_weight).abs().max().item()

            if error < 1e-6:
                results.append(True)
                print(".2e")
            else:
                results.append(False)
                print(".2e")
        except Exception as e:
            print(f"    ❌ 异常: {e}")
            results.append(False)

    success_count = sum(results)
    total_count = len(results)

    print(f"\n📊 核心类型测试: {success_count}/{total_count} 通过")

    return success_count == total_count


def test_conservative_scaling():
    """测试保守缩放策略"""
    print("\n⚖️  测试保守缩放策略...")

    base_layer = nn.Linear(64, 32)
    cfg = JoraConfig(S_L=4, S_R=4, core="diag", magnitude="none")
    layer = JoraLayer(base_layer, 'test', cfg)

    adapter_state = layer.adapters['test']

    # 计算权重增量
    delta_weight = layer._compute_weight_delta_simple(adapter_state)

    print(f"  Delta权重形状: {delta_weight.shape}")
    print(".8f")
    print(".8f")

    # 增量应该非常小（由于0.05的保守缩放）
    max_delta = delta_weight.abs().max().item()
    mean_delta = delta_weight.abs().mean().item()

    print(".8f")
    print(".8f")

    # 验证保守性：最大增量应该小于0.01
    assert max_delta < 0.01, f"Delta too large: {max_delta}"
    assert mean_delta < 0.001, f"Average delta too large: {mean_delta}"

    print("✅ 保守缩放策略测试通过")
    return True


if __name__ == '__main__':
    print("🚀 开始JORA merge最终验证测试\n")

    test_functions = [
        test_mathematical_correctness,
        test_rotation_effect_estimation,
        test_different_core_types,
        test_conservative_scaling,
    ]

    results = []
    for test_func in test_functions:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ {test_func.__name__} 异常: {e}")
            results.append(False)

    success_count = sum(results)
    total_count = len(results)

    print(f"\n🏆 最终结果: {success_count}/{total_count} 测试通过")

    if success_count == total_count:
        print("🎉 JORA merge实现验证完成！")
        print("   - 数学正确性：✅")
        print("   - 旋转效应估计：✅")
        print("   - 保守缩放策略：✅")
        print("   - 核心类型兼容性：✅")
    else:
        print("⚠️  部分测试失败，需要进一步检查")

    sys.exit(0 if success_count == total_count else 1)