#!/usr/bin/env python3
"""
测试基于采样的JORA merge实现
"""

import sys
import os
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from peft.tuners.jora.config import JoraConfig
from peft.tuners.jora.layer import JoraLayer


def test_sampling_based_merge():
    """测试基于采样的merge方法"""
    print("🧪 测试基于采样的JORA merge方法...")

    # 创建测试模型
    base_layer = nn.Linear(64, 32)
    cfg = JoraConfig(S_L=4, S_R=4, core="diag", magnitude="none")
    layer = JoraLayer(base_layer, 'test', cfg)

    # 记录原始权重
    original_weight = base_layer.weight.data.clone()

    print("  原始权重范围: [{:.6f}, {:.6f}]".format(original_weight.min(), original_weight.max()))

    # 执行merge
    layer.merge()
    merged_weight = base_layer.weight.data.clone()

    # 计算变化
    weight_diff = (merged_weight - original_weight).abs()
    max_diff = weight_diff.max().item()
    mean_diff = weight_diff.mean().item()

    print("  Merge后权重范围: [{:.6f}, {:.6f}]".format(merged_weight.min(), merged_weight.max()))
    print("  最大权重变化: {:.6f}".format(max_diff))
    print("  平均权重变化: {:.6f}".format(mean_diff))

    # 执行unmerge
    layer.unmerge()
    restored_weight = base_layer.weight.data.clone()

    # 计算恢复误差
    restore_diff = (restored_weight - original_weight).abs()
    max_restore_error = restore_diff.max().item()
    mean_restore_error = restore_diff.mean().item()

    print("  Unmerge后权重范围: [{:.6f}, {:.6f}]".format(restored_weight.min(), restored_weight.max()))
    print("  最大恢复误差: {:.2e}".format(max_restore_error))
    print("  平均恢复误差: {:.2e}".format(mean_restore_error))

    # 验证一致性
    threshold = 1e-6
    if max_restore_error < threshold:
        print("✅ Merge/unmerge一致性良好")
        return True
    else:
        print("❌ Merge/unmerge一致性不足")
        return False


def test_merge_quality_assessment():
    """评估merge质量 - 比较merge前后模型输出"""
    print("\n🔍 评估merge质量...")

    # 创建测试模型
    base_layer = nn.Linear(64, 32)
    cfg = JoraConfig(S_L=4, S_R=4, core="diag", magnitude="none")
    layer = JoraLayer(base_layer, 'test', cfg)

    # 生成测试输入
    test_input = torch.randn(16, 64)

    # 记录原始JORA输出
    layer.eval()
    with torch.no_grad():
        original_output = layer(test_input)

    print("  原始输出范围: [{:.4f}, {:.4f}]".format(original_output.min(), original_output.max()))

    # 执行merge
    layer.merge()

    # 计算merge后输出
    with torch.no_grad():
        merged_output = layer(test_input)

    print("  Merge输出范围: [{:.4f}, {:.4f}]".format(merged_output.min(), merged_output.max()))

    # 计算输出差异
    output_diff = (merged_output - original_output).abs()
    max_output_diff = output_diff.max().item()
    mean_output_diff = output_diff.mean().item()
    rmse = torch.sqrt((output_diff ** 2).mean()).item()

    print("  最大输出差异: {:.6f}".format(max_output_diff))
    print("  平均输出差异: {:.6f}".format(mean_output_diff))
    print("  输出RMSE: {:.6f}".format(rmse))

    # 计算相对误差
    relative_diff = output_diff / (original_output.abs() + 1e-8)
    max_relative_diff = relative_diff.max().item()
    mean_relative_diff = relative_diff.mean().item()

    print("  最大相对误差: {:.4f}".format(max_relative_diff))
    print("  平均相对误差: {:.4f}".format(mean_relative_diff))

    # 对于基于采样的方法，我们期望相对误差在合理范围内
    # 由于这是数学近似，完美匹配是不可能的
    quality_threshold = 0.1  # 10%相对误差阈值

    if mean_relative_diff < quality_threshold:
        print("✅ Merge质量在可接受范围内")
        return True
    else:
        print("⚠️  Merge质量超出预期范围，可能需要调整")
        return False


if __name__ == '__main__':
    success1 = test_sampling_based_merge()
    success2 = test_merge_quality_assessment()

    overall_success = success1 and success2
    print(f"\n🏆 总体结果: {'通过' if overall_success else '失败'}")
    sys.exit(0 if overall_success else 1)