#!/usr/bin/env python3
"""
JORA 梯度诊断 - 深入分析梯度流动问题
"""

import torch
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from peft.tuners.jora.config import JoraConfig
from peft.tuners.jora.layer import _JoraAdapterState
import torch.nn as nn


def test_theta_gradients():
    """测试 theta 的梯度"""
    print("\n=== 诊断: theta 梯度问题 ===")

    linear = nn.Linear(8, 8)
    linear.weight.data.zero_()
    linear.bias.data.zero_()

    cfg = JoraConfig.paper_path(
        target_modules=[""],
        S_L=4,
        S_R=4,
        k=2,
    )
    cfg.core = "selective_diag"

    state = _JoraAdapterState(linear, cfg)

    # 检查初始状态
    print(f"\n1. 初始状态:")
    print(f"   theta_L 初始值: {state.theta_L.tolist()[:4]}")
    print(f"   theta_R 初始值: {state.theta_R.tolist()[:4]}")
    print(f"   num_pairs_L: {state.num_pairs_L.item()}")
    print(f"   num_pairs_R: {state.num_pairs_R.item()}")
    print(f"   pairs_L[:2]: {state.pairs_L[:2].tolist()}")

    # 初始化随机pairs
    state.init_random_pairs(n_pairs_L=2, n_pairs_R=2)
    print(f"\n2. 初始化后:")
    print(f"   num_pairs_L: {state.num_pairs_L.item()}")
    print(f"   num_pairs_R: {state.num_pairs_R.item()}")
    print(f"   pairs_L[:2]: {state.pairs_L[:2].tolist()}")
    print(f"   pairs_R[:2]: {state.pairs_R[:2].tolist()}")

    # 设置支持集
    support = torch.tensor([0, 1, 2, 3])
    state.core.set_support(support)
    print(f"\n3. 支持集:")
    print(f"   active_support_size: {state.core.active_support_size.item()}")
    print(f"   support_indices: {state.core.support_indices.tolist()}")

    # 设置非零theta
    with torch.no_grad():
        state.theta_L.uniform_(-0.1, 0.1)
        state.theta_R.uniform_(-0.1, 0.1)
    print(f"\n4. theta 值:")
    print(f"   theta_L: {state.theta_L.tolist()}")
    print(f"   theta_R: {state.theta_R.tolist()}")

    # 前向传播
    x = torch.randn(2, 8, requires_grad=True)
    delta = state.compute_delta(x)
    print(f"\n5. 前向结果:")
    print(f"   delta norm: {delta.norm().item():.4f}")

    # 反向传播
    loss = delta.sum()
    loss.backward()

    print(f"\n6. 梯度:")
    print(f"   x.grad: {x.grad.norm().item():.4f}" if x.grad is not None else "   x.grad: None")
    print(f"   theta_L.grad: {state.theta_L.grad.tolist() if state.theta_L.grad is not None else 'None'}")
    print(f"   theta_R.grad: {state.theta_R.grad.tolist() if state.theta_R.grad is not None else 'None'}")
    print(f"   core.delta.grad: {state.core.delta.grad.tolist() if state.core.delta.grad is not None else 'None'}")

    # 深入分析
    print(f"\n7. 梯度分析:")
    if state.theta_L.grad is not None:
        print(f"   theta_L grad norm: {state.theta_L.grad.norm().item():.6f}")
    if state.theta_R.grad is not None:
        print(f"   theta_R grad norm: {state.theta_R.grad.norm().item():.6f}")


def test_with_layer():
    """使用 JoraLayer 测试"""
    print("\n=== 使用 JoraLayer 测试 ===")

    from peft.tuners.jora.layer import JoraLayer

    linear = nn.Linear(8, 8)
    linear.weight.data.normal_()
    linear.bias.data.zero_()

    cfg = JoraConfig.paper_path(
        target_modules=[""],
        S_L=4,
        S_R=4,
        k=2,
    )
    cfg.core = "selective_diag"

    layer = JoraLayer(linear, "default", cfg)

    # 初始化pairs
    for adapter in layer.adapters.values():
        adapter.init_random_pairs(n_pairs_L=2, n_pairs_R=2)
        adapter.core.set_support(torch.tensor([0, 1, 2, 3]))

    # 设置非零theta
    for adapter in layer.adapters.values():
        with torch.no_grad():
            if adapter.theta_L is not None:
                adapter.theta_L.uniform_(-0.1, 0.1)
            if adapter.theta_R is not None:
                adapter.theta_R.uniform_(-0.1, 0.1)

    # 前向
    x = torch.randn(2, 8, requires_grad=True)
    out = layer(x)
    print(f"   out norm: {out.norm().item():.4f}")

    # 反向
    loss = out.sum()
    loss.backward()

    print(f"\n   x.grad: {x.grad.norm().item():.4f}" if x.grad is not None else "   x.grad: None")

    for name, adapter in layer.adapters.items():
        print(f"\n   Adapter '{name}':")
        print(f"   theta_L grad: {adapter.theta_L.grad.tolist() if adapter.theta_L.grad is not None else 'None'}")
        print(f"   theta_R grad: {adapter.theta_R.grad.tolist() if adapter.theta_R.grad is not None else 'None'}")
        print(f"   core.delta grad: {adapter.core.delta.grad.tolist() if adapter.core.delta.grad is not None else 'None'}")


def diagnose_rotation_path():
    """诊断旋转路径"""
    print("\n=== 诊断: 旋转路径 ===")

    linear = nn.Linear(8, 8)
    linear.weight.data.zero_()
    linear.bias.data.zero_()

    cfg = JoraConfig.paper_path(
        target_modules=[""],
        S_L=4,
        S_R=4,
        k=2,
    )
    cfg.core = "selective_diag"

    state = _JoraAdapterState(linear, cfg)

    # 初始化
    state.init_random_pairs(n_pairs_L=2, n_pairs_R=2)
    state.core.set_support(torch.tensor([0, 1, 2, 3]))

    with torch.no_grad():
        state.theta_L.uniform_(-0.1, 0.1)
        state.theta_R.uniform_(-0.1, 0.1)

    x = torch.randn(2, 8, requires_grad=True)

    # 手动追踪旋转
    print("\n1. 测试 apply_rotations:")

    # 左旋转
    x_rotated_left = state._apply_side_rotation(x, is_left_side=True)
    print(f"   x (original): {x[0, :4].tolist()}")
    print(f"   R_L^T @ x: {x_rotated_left[0, :4].tolist()}")

    if x_rotated_left.equal(x):
        print("   ⚠️  左旋转没有改变 x (is_left_side=True 时 negate_theta=True)")
        print("      原因: negate_theta=True 使 theta 变成 -theta")
        print("      对于小角度，-θ ≈ θ，所以看起来没变")

    # 右旋转
    x_rotated_right = state._apply_side_rotation(x, is_left_side=False)
    print(f"\n   R_R @ x: {x_rotated_right[0, :4].tolist()}")
    print(f"   变化量: {(x_rotated_right - x).abs().mean().item():.6f}")


def test_legacy_vs_paper_path():
    """测试 legacy path vs paper path"""
    print("\n=== Legacy Path vs Paper Path ===")

    linear = nn.Linear(8, 8)
    linear.weight.data.zero_()
    linear.bias.data.zero_()

    # Paper path (selective_diag)
    print("\nPaper path (core='selective_diag'):")
    cfg_paper = JoraConfig.paper_path(
        target_modules=[""],
        S_L=4,
        S_R=4,
        k=2,
    )
    cfg_paper.core = "selective_diag"

    state_paper = _JoraAdapterState(linear, cfg_paper)
    state_paper.init_random_pairs(n_pairs_L=2, n_pairs_R=2)
    state_paper.core.set_support(torch.tensor([0, 1, 2, 3]))

    with torch.no_grad():
        state_paper.theta_L.uniform_(-0.1, 0.1)
        state_paper.theta_R.uniform_(-0.1, 0.1)

    x = torch.randn(2, 8)
    delta_paper = state_paper.compute_delta(x)
    print(f"   delta norm: {delta_paper.norm().item():.6f}")

    # Legacy path (core='diag')
    print("\nLegacy path (core='diag'):")
    cfg_legacy = JoraConfig(
        target_modules=[""],
        S_L=4,
        S_R=4,
        k=2,
    )
    cfg_legacy.core = "diag"
    cfg_legacy.magnitude = "none"

    state_legacy = _JoraAdapterState(linear, cfg_legacy)
    state_legacy.init_random_pairs(n_pairs_L=2, n_pairs_R=2)

    with torch.no_grad():
        state_legacy.theta_L.uniform_(-0.1, 0.1)
        state_legacy.theta_R.uniform_(-0.1, 0.1)

    delta_legacy = state_legacy.compute_delta(x)
    print(f"   delta norm: {delta_legacy.norm().item():.6f}")

    # 比较
    print(f"\n比较:")
    print(f"   Paper path 使用 SelectiveDiagCore.apply_to_vector()")
    print(f"   Legacy path 使用 DiagCore.apply_to_vector()")
    print(f"   两者在支撑集外的行为不同！")


if __name__ == "__main__":
    test_theta_gradients()
    test_with_layer()
    diagnose_rotation_path()
    test_legacy_vs_paper_path()
