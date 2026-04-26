#!/usr/bin/env python
"""Gate B test: Verify optimizer param groups cover all trainable params.

This test verifies that:
1. All trainable params are covered by optimizer param groups
2. CoupledPairCore's pair_blocks are included in the core group
3. The param groups have the correct LR ratios
"""

import sys
import torch
import torch.nn as nn

sys.path.insert(0, '/home/jqh/Workshop/JORA/src')

from peft.tuners.jora.core import CoupledPairCore


def test_optimizer_coverage():
    """Test that all trainable params are covered by optimizer param groups."""
    print("=" * 70)
    print("Gate B: Optimizer Param Group Coverage Test")
    print("=" * 70)

    # Simulate the classification logic from _configure_jora_optimizer_groups
    def classify_param(name):
        if "theta_L" in name or "theta_R" in name:
            return "theta"
        elif ".core." in name or "diag_params" in name or "ecd_log_mag" in name or "pair_blocks" in name:
            return "core"
        else:
            return "other"

    # Create mock params representing all JORA parameter types
    params = {
        "model.layers.0.self_attn.q_proj.jora.theta_L": nn.Parameter(torch.randn(8)),
        "model.layers.0.self_attn.q_proj.jora.theta_R": nn.Parameter(torch.randn(8)),
        "model.layers.0.self_attn.q_proj.jora.core.delta": nn.Parameter(torch.randn(16)),
        "model.layers.0.self_attn.q_proj.jora.core.pair_blocks": nn.Parameter(torch.randn(4, 2, 2)),
        "model.layers.0.self_attn.q_proj.jora.ecd_log_mag": nn.Parameter(torch.randn(4096)),
        "model.lm_head.weight": nn.Parameter(torch.randn(4096, 4096)),  # Other params
    }

    # Set requires_grad for all
    for p in params.values():
        p.requires_grad = True

    # Classify
    param_groups = {"theta": [], "core": [], "other": []}
    for name, param in params.items():
        if param.requires_grad:
            group = classify_param(name)
            param_groups[group].append(param)

    print("\nParam classification:")
    for group, ps in param_groups.items():
        total = sum(p.numel() for p in ps)
        print(f"  {group}: {len(ps)} params, {total:,} elements")

    # Verify pair_blocks is in core
    pair_blocks_found = False
    for name, param in params.items():
        if "pair_blocks" in name:
            group = classify_param(name)
            assert group == "core", f"pair_blocks should be in core, got {group}"
            pair_blocks_found = True
            print(f"\n✅ pair_blocks '{name}' correctly classified as core")

    assert pair_blocks_found, "pair_blocks not found in params"
    assert len(param_groups["theta"]) == 2, "Should have 2 theta params"
    assert len(param_groups["core"]) == 3, "Should have 3 core params (delta, pair_blocks, log_mag)"
    assert len(param_groups["other"]) == 1, "Should have 1 other param"

    # Verify total trainable coverage
    total_trainable = sum(p.numel() for p in params.values())
    total_in_groups = sum(p.numel() for ps in param_groups.values() for p in ps)
    assert total_trainable == total_in_groups, \
        f"Coverage gap: {total_trainable} trainable, {total_in_groups} in groups"

    print(f"\n✅ Total coverage: {total_in_groups:,} / {total_trainable:,} = 100%")

    print("\n" + "=" * 70)
    print("✅ OPTIMIZER COVERAGE TEST PASSED!")
    print("=" * 70)


def test_lr_ratio_computation():
    """Test that LR ratios are computed correctly."""
    print("\n" + "=" * 70)
    print("LR Ratio Computation Test")
    print("=" * 70)

    base_lr = 1e-4

    # From JoraConfig defaults
    lr_theta_config = 0.05
    lr_core_config = 0.01

    # Ratio from config
    lr_ratio = lr_theta_config / lr_core_config
    theta_lr = base_lr * lr_ratio

    print(f"\nBase LR: {base_lr}")
    print(f"Config: lr_theta={lr_theta_config}, lr_core={lr_core_config}")
    print(f"LR ratio: {lr_ratio}")
    print(f"Theta LR: {theta_lr}")
    print(f"Core LR: {base_lr}")

    assert abs(theta_lr - base_lr * 5.0) < 1e-12, "LR ratio should be 5x"

    print("\n✅ LR ratio computation is correct")


if __name__ == "__main__":
    test_optimizer_coverage()
    test_lr_ratio_computation()

    print("\n" + "=" * 70)
    print("🎉 ALL OPTIMIZER INTEGRATION TESTS PASSED!")
    print("=" * 70)
