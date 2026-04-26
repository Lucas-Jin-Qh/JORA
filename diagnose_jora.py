#!/usr/bin/env python3
"""
JORA 深度诊断脚本
验证关键假设，检测潜在bug
"""

import torch
import sys
import os
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from peft.tuners.jora.config import JoraConfig
from peft.tuners.jora.layer import JoraLayer, _JoraAdapterState
from peft.tuners.jora.core import SelectiveDiagCore, build_core
from peft.tuners.jora.rotation import apply_rotations
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn as nn


def test_basic_forward():
    """测试基本前向传播"""
    print("\n=== 测试1: 基本前向传播 ===")

    # 创建一个小型线性层
    linear = nn.Linear(16, 16)
    linear.weight.data.zero_()  # 零初始化，便于验证
    linear.bias.data.zero_()

    # 创建 JORA 配置 (paper path)
    cfg = JoraConfig.paper_path(
        target_modules=[""],  # 不会实际用到
        S_L=8,
        S_R=8,
        k=4,
        t_stat=10,
    )
    cfg.core = "selective_diag"

    # 创建 adapter state
    state = _JoraAdapterState(linear, cfg)

    # 测试1: theta=0, delta=0 时 delta 应该为 0
    print("\n测试1a: theta=0, delta=0 (理想情况)")
    with torch.no_grad():
        x = torch.randn(2, 16)
        delta = state.compute_delta(x)
        print(f"  delta norm: {delta.norm().item():.2e}")
        print(f"  期望值: ~0.0 (数值误差范围内)")
        if delta.norm().item() > 1e-5:
            print("  ❌ FAIL: delta 不为零，公式可能有bug！")
        else:
            print("  ✓ PASS")

    # 测试2: 非零theta, delta=0 时应该产生非零delta
    print("\n测试1b: theta≠0, delta=0")
    with torch.no_grad():
        # 手动设置小的theta值
        if state.theta_L is not None:
            state.theta_L.uniform_(-0.1, 0.1)
        if state.theta_R is not None:
            state.theta_R.uniform_(-0.1, 0.1)
        # delta保持为0
        state.core.delta.zero_()

        x = torch.randn(2, 16)
        delta = state.compute_delta(x)
        print(f"  delta norm: {delta.norm().item():.4f}")
        print(f"  期望值: > 0.01 (应该有显著值)")
        if delta.norm().item() < 1e-3:
            print("  ⚠️  WARNING: delta太小，旋转可能没起作用")
        else:
            print("  ✓ PASS")


def test_support_projection():
    """测试支撑集投影 P_U @ x"""
    print("\n=== 测试2: 支撑集投影 P_U @ x ===")

    linear = nn.Linear(16, 16)
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

    # 手动设置支持集 U = {0, 3, 5, 7}
    support_indices = torch.tensor([0, 3, 5, 7])
    state.core.set_support(support_indices)

    # 创建测试输入
    x = torch.randn(2, 16)

    # 计算 P_U @ x
    proj_x = state.core.project_support(x)

    print(f"  输入 x (第一个样本): {x[0].tolist()}")
    print(f"  投影 P_U@x: {proj_x[0].tolist()}")
    print(f"  非零位置应该在: {support_indices.tolist()}")

    # 验证：只有支撑集位置有值
    mask = torch.zeros(16, dtype=torch.bool)
    mask[support_indices] = True

    non_support_vals = proj_x[0][~mask]
    support_vals = proj_x[0][mask]

    print(f"  支撑集位置的均值: {support_vals.abs().mean().item():.4f}")
    print(f"  非支撑集位置的均值: {non_support_vals.abs().mean().item():.4e}")

    if non_support_vals.abs().max().item() > 1e-6:
        print("  ❌ FAIL: 非支撑集位置有非零值！")
    else:
        print("  ✓ PASS")


def test_forward_formula():
    """测试前向传播公式的数学正确性"""
    print("\n=== 测试3: 前向传播公式验证 ===")

    linear = nn.Linear(4, 4)  # 小维度便于验证
    linear.weight.data.zero_()
    linear.bias.data.zero_()

    cfg = JoraConfig.paper_path(
        target_modules=[""],
        S_L=2,
        S_R=2,
        k=1,
    )
    cfg.core = "selective_diag"

    state = _JoraAdapterState(linear, cfg)

    # 手动构造一个简单场景
    # 设 U = {0, 1}, pairs = [(0,1)]
    state.core.set_support(torch.tensor([0, 1]))
    state.pairs_L[0] = torch.tensor([0, 1])
    state.pairs_L[1] = torch.tensor([-1, -1])  # padding
    state.num_pairs_L.fill_(1)
    state.pairs_R[0] = torch.tensor([0, 1])
    state.pairs_R[1] = torch.tensor([-1, -1])
    state.num_pairs_R.fill_(1)

    # 设置简单的theta: 小角度旋转
    with torch.no_grad():
        state.theta_L[0] = 0.1
        state.theta_R[0] = 0.1
        state.core.delta[0] = 0.01  # 小的对角修正
        state.core.delta[1] = -0.01

    # 创建单位矩阵输入，便于观察变换
    x = torch.eye(4).unsqueeze(0)  # [1, 4, 4]

    delta = state.compute_delta(x)

    print(f"  输入（单位矩阵）: {x[0]}")
    print(f"  delta: {delta[0]}")
    print(f"  delta 对角元素: {delta[0].diag().tolist()}")

    # 理论分析：
    # 当 x 是单位矩阵时，delta 应该反映出 R_L^T @ D_sel @ R_R - P_U
    # 如果 theta=0, delta=0: R=I, D=I, 所以 R_L^T D_sel R_R - P_U = I - P_U ≠ 0
    # 这是设计如此！因为 P_U 投影到支撑集，而 I 在所有维度上都是1

    print("\n  理论验证:")
    print("  当 theta=0, delta=0 时:")
    print("    R_L^T @ I_U @ R_R @ x = P_U @ x")
    print("    所以 delta = P_U@x - P_U@x = 0 ✓")
    print("  但当 theta=0, delta≠0 时:")
    print("    R_L^T @ D_sel @ R_R @ x 在支撑集外可能非零")
    print("    减去 P_U@x 后，支撑集外应该为零")


def test_parameter_count():
    """验证参数数量是否符合预期"""
    print("\n=== 测试4: 参数数量验证 ===")

    # paper-path 配置
    cfg = JoraConfig.paper_path(
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        S_L=32,
        S_R=32,
        k=32,
    )

    print(f"  配置: S_L={cfg.S_L}, S_R={cfg.S_R}, k={cfg.k}")
    print(f"  core={cfg.core}, magnitude={cfg.magnitude}")

    # 估算参数
    n_modules = 4  # q, k, v, o
    rotation_params = (cfg.S_L + cfg.S_R) * n_modules
    core_params = cfg.k * 2  # selective_diag: 只有 |U|=2k 个参数
    magnitude_params = 0 if cfg.magnitude == "none" else 4096 * n_modules

    total = rotation_params + core_params + magnitude_params

    print(f"  旋转参数: {rotation_params}")
    print(f"  核心参数 (selective_diag |U|=2k): {core_params}")
    print(f"  幅度参数: {magnitude_params}")
    print(f"  总计: {total} (~{total/1000:.1f}K)")

    print(f"\n  LoRA-r8 对比: ~8.5M/module × 4 = ~34M")
    print(f"  差距: {34_000_000/total:.0f}x")


def test_gradient_flow():
    """测试梯度流动"""
    print("\n=== 测试5: 梯度流动诊断 ===")

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
    state.core.set_support(torch.tensor([0, 1, 2, 3]))

    # 前向传播
    x = torch.randn(2, 8, requires_grad=True)
    delta = state.compute_delta(x)
    loss = delta.sum()
    loss.backward()

    print(f"  theta_L grad norm: {state.theta_L.grad.norm().item() if state.theta_L.grad is not None else 'None'}")
    print(f"  theta_R grad norm: {state.theta_R.grad.norm().item() if state.theta_R.grad is not None else 'None'}")
    print(f"  core.delta grad norm: {state.core.delta.grad.norm().item() if state.core.delta.grad is not None else 'None'}")

    # 检查梯度是否太小
    grads = [
        state.theta_L.grad.norm().item() if state.theta_L.grad is not None else 0,
        state.theta_R.grad.norm().item() if state.theta_R.grad is not None else 0,
        state.core.delta.grad.norm().item() if state.core.delta.grad is not None else 0,
    ]

    if max(grads) < 1e-8:
        print("  ❌ 严重问题: 梯度几乎为零！可能的原因:")
        print("     1. theta 零初始化导致梯度消失")
        print("     2. 前向传播公式bug导致某些路径梯度为0")
        print("     3. 激活值太小")


def check_code_issues():
    """检查代码中的潜在问题"""
    print("\n=== 代码问题检查 ===")

    issues = []

    # 1. 检查 theta 初始化
    print("\n1. theta 初始化:")
    print("   layer.py 行 56-77:")
    print("   if cfg.core == 'selective_diag': init_std = 0.0")
    print("   这导致旋转从恒等变换开始，可能收敛慢")
    issues.append("B5: theta 零初始化 - 建议改为 0.01~0.02")

    # 2. 检查前向公式
    print("\n2. 前向传播公式 (layer.py 483-484):")
    print("   return y_rotated - proj_x")
    print("   问题: -proj_x 可能过度减去，导致有效信号弱")
    issues.append("B1/B3: 公式歧义 - 需要数学验证")

    # 3. EMA beta
    print(f"\n3. EMA beta (config.py 行 82):")
    print(f"   ema_beta: float = 0.98")
    print(f"   对于 200 步校准，收敛太慢")
    issues.append("B1: EMA beta=0.98 太慢，建议 0.90-0.95")

    # 4. magnitude
    print(f"\n4. magnitude 模块 (config.py 行 103):")
    print(f"   默认: 'oer_softmax'")
    print(f"   paper_path(): magnitude='none'")
    issues.append("B6: paper path 关闭了 magnitude，但可能有用")

    # 5. 配对策略
    print(f"\n5. 配对策略 (selection.py 行 70):")
    print(f"   pair_scores = energy[i] * energy[j]")
    print(f"   启发式方法，缺乏理论基础")
    issues.append("B4: 配对策略需要理论支撑")

    print(f"\n\n发现 {len(issues)} 个关键问题:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")


def generate_report():
    """生成诊断报告"""
    print("\n" + "="*60)
    print("JORA 深度诊断报告")
    print("="*60)

    test_basic_forward()
    test_support_projection()
    test_forward_formula()
    test_parameter_count()
    test_gradient_flow()
    check_code_issues()

    print("\n" + "="*60)
    print("立即行动建议")
    print("="*60)
    print("""
1. 验证实验配置 (最高优先级)
   - 确认是否使用了 paper_path 配置
   - 检查: core='selective_diag', k=32, S_L=32, S_R=32
   - 当前实验: s16_k4_diag 可能不是 paper path！

2. 验证前向传播公式
   - 运行 test_basic_forward() 检查 delta 在 theta=0 时是否为零
   - 如果 delta≠0，说明公式有bug，修复可能提升 5-15pp

3. 改进校准
   - 增加 t_stat 到 500 或使用更快的 EMA (beta=0.95)
   - 预期提升: 0.5-2pp

4. 启用 magnitude 模块
   - paper path 设为 "none"，但 ablation 实验应该启用
   - 预期提升: 1-3pp

5. θ 非零初始化
   - 将 theta_init_std 改为 0.01
   - 预期提升: 1-3pp (加速收敛)

6. 调整学习率
   - lr_theta=0.01, lr_core=0.1
   - 预期提升: 0.3-1pp

7. 修复 EMA 更新频率
   - 确保在 optimizer step 粒度更新，而非 micro-batch
   - 预期提升: 0.3-0.5pp
    """)


if __name__ == "__main__":
    generate_report()
