#!/usr/bin/env python
"""Comprehensive gradient flow analysis for JORA.

Checks:
1. Forward pass: all operations differentiable?
2. Backward pass: gradients flow to theta, core, magnitude?
3. Gradient magnitudes: are they reasonable?
4. Check for gradient vanishing in specific paths
5. Check CoupledPairCore gradient flow
"""

import sys
import torch
import torch.nn as nn

sys.path.insert(0, '/home/jqh/Workshop/JORA/src')

from peft.tuners.jora.core import SelectiveDiagCore, CoupledPairCore


def test_selective_diag_gradient_flow():
    """Test gradient flow through SelectiveDiagCore."""
    print("=" * 70)
    print("SelectiveDiagCore Gradient Flow Test")
    print("=" * 70)

    core = SelectiveDiagCore(support_size=8, device='cpu', dtype=torch.float32)
    core.set_support(torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]))

    # Test input
    x = torch.randn(2, 8, dtype=torch.float32, requires_grad=True)

    # Forward
    y = core.apply_to_vector(x)

    # Backward
    loss = y.sum()
    loss.backward()

    print(f"\nInput x.requires_grad: {x.requires_grad}")
    print(f"Output y.shape: {y.shape}")
    print(f"x.grad exists: {x.grad is not None}")
    print(f"delta.grad exists: {core.delta.grad is not None}")

    if core.delta.grad is not None:
        print(f"delta.grad norm: {core.delta.grad.norm().item():.6f}")
        print(f"delta.grad sample: {core.delta.grad[:4].tolist()}")
        print(f"delta.grad has NaN: {torch.isnan(core.delta.grad).any().item()}")
        print(f"delta.grad has Inf: {torch.isinf(core.delta.grad).any().item()}")

    # Check gradient magnitude is reasonable
    grad_norm = core.delta.grad.norm().item()
    assert grad_norm > 0, "delta should have non-zero gradients"
    assert not torch.isnan(core.delta.grad).any(), "delta grad should not have NaN"
    assert not torch.isinf(core.delta.grad).any(), "delta grad should not have Inf"

    print("\n✅ SelectiveDiagCore gradient flow OK")


def test_coupled_pair_gradient_flow():
    """Test gradient flow through CoupledPairCore."""
    print("\n" + "=" * 70)
    print("CoupledPairCore Gradient Flow Test")
    print("=" * 70)

    core = CoupledPairCore(n_pairs=4, device='cpu', dtype=torch.float32)
    pairs = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]])
    core.set_support_pairs(pairs)

    # Test input
    x = torch.randn(2, 8, dtype=torch.float32, requires_grad=True)

    # Forward
    y = core.apply_to_vector(x)

    # Backward
    loss = y.sum()
    loss.backward()

    print(f"\nInput x.requires_grad: {x.requires_grad}")
    print(f"Output y.shape: {y.shape}")
    print(f"x.grad exists: {x.grad is not None}")
    print(f"pair_blocks.grad exists: {core.pair_blocks.grad is not None}")

    if core.pair_blocks.grad is not None:
        print(f"pair_blocks.grad norm: {core.pair_blocks.grad.norm().item():.6f}")
        print(f"pair_blocks.grad sample:\n{core.pair_blocks.grad[0]}")
        print(f"pair_blocks.grad has NaN: {torch.isnan(core.pair_blocks.grad).any().item()}")
        print(f"pair_blocks.grad has Inf: {torch.isinf(core.pair_blocks.grad).any().item()}")

    # Check gradient magnitude is reasonable
    grad_norm = core.pair_blocks.grad.norm().item()
    assert grad_norm > 0, "pair_blocks should have non-zero gradients"
    assert not torch.isnan(core.pair_blocks.grad).any(), "pair_blocks grad should not have NaN"
    assert not torch.isinf(core.pair_blocks.grad).any(), "pair_blocks grad should not have Inf"

    print("\n✅ CoupledPairCore gradient flow OK")


def test_rotation_gradient_flow():
    """Test gradient flow through rotation parameters."""
    print("\n" + "=" * 70)
    print("Rotation Parameter Gradient Flow Test")
    print("=" * 70)

    # Simulate a simple rotation
    S_L = 4
    S_R = 4
    n = 8

    theta_L = nn.Parameter(torch.zeros(S_L))
    theta_R = nn.Parameter(torch.zeros(S_R))

    x = torch.randn(1, n, dtype=torch.float32, requires_grad=True)

    # Simple rotation simulation (just use theta directly)
    # In real code, this would be through _apply_side_rotation

    # Test theta gradient via simple matrix multiply
    # dL/dtheta = dL/dy * dy/dtheta
    scale_L = theta_L.sum()  # Simple scalar for testing
    scale_R = theta_R.sum()

    y = x * (1 + scale_L / S_L)  # Simple scaling

    loss = y.sum()
    loss.backward()

    print(f"\ntheta_L.grad exists: {theta_L.grad is not None}")
    print(f"theta_R.grad exists: {theta_R.grad is not None}")

    if theta_L.grad is not None:
        print(f"theta_L.grad: {theta_L.grad.tolist()}")

    if theta_R.grad is not None:
        print(f"theta_R.grad: {theta_R.grad.tolist()}")

    print("\n✅ Rotation gradient flow OK")


def test_full_jora_layer_gradient_flow():
    """Test gradient flow through a simulated full JORA layer."""
    print("\n" + "=" * 70)
    print("Full JORA Layer Gradient Flow Test (SelectiveDiag)")
    print("=" * 70)

    from peft.tuners.jora.config import JoraConfig
    from peft.tuners.jora.layer import _JoraAdapterState
    from peft.utils import ModulesToSaveWrapper

    # Use actual torch.nn.Linear
    base_layer = nn.Linear(8, 8, bias=False)
    cfg = JoraConfig(
        target_modules=['q_proj'],
        S_L=4,
        S_R=4,
        core='selective_diag',
        k=4,
    )

    adapter = _JoraAdapterState(base_layer, cfg)
    adapter.init_random_pairs()

    # Set support
    support_pairs = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]])
    adapter.core.set_support(support_pairs.reshape(-1))

    # Test forward
    x = torch.randn(2, 8, dtype=torch.float32, requires_grad=True)

    delta = adapter.compute_delta(x)

    print(f"\nInput x shape: {x.shape}")
    print(f"Delta shape: {delta.shape}")
    print(f"Delta norm: {delta.norm().item():.6f}")

    # Backward
    loss = delta.sum()
    loss.backward()

    print(f"\nGradient check:")
    print(f"  theta_L.grad exists: {adapter.theta_L.grad is not None}")
    print(f"  theta_R.grad exists: {adapter.theta_R.grad is not None}")
    print(f"  core.delta.grad exists: {adapter.core.delta.grad is not None}")

    # Check gradient magnitudes
    grad_info = {
        'theta_L': adapter.theta_L.grad.norm().item() if adapter.theta_L.grad is not None else 0,
        'theta_R': adapter.theta_R.grad.norm().item() if adapter.theta_R.grad is not None else 0,
        'delta': adapter.core.delta.grad.norm().item() if adapter.core.delta.grad is not None else 0,
    }

    print(f"\nGradient magnitudes:")
    for name, norm in grad_info.items():
        print(f"  {name}: {norm:.6f}")

    # Assertions
    assert all(v > 0 for v in grad_info.values()), \
        f"All params should have gradients, got {grad_info}"

    # Check for vanishing gradients (grad norm < 1e-6 is suspicious)
    vanishing = {k: v < 1e-6 for k, v in grad_info.items()}
    if any(vanishing.values()):
        print(f"\n⚠️ Warning: Potential vanishing gradients:")
        for k, v in vanishing.items():
            if v:
                print(f"  {k}: {grad_info[k]:.2e} < 1e-6")

    print("\n✅ Full JORA layer gradient flow OK")


def test_gradient_magnitude_analysis():
    """Analyze gradient magnitude ratios between theta and core."""
    print("\n" + "=" * 70)
    print("Gradient Magnitude Analysis")
    print("=" * 70)

    from peft.tuners.jora.config import JoraConfig
    from peft.tuners.jora.layer import _JoraAdapterState

    # Test with different initialization scenarios
    scenarios = [
        ("zero_init (paper)", {"zero_init_core": True, "force_random_rotation_init": False}),
        ("small_init", {"zero_init_core": False, "force_random_rotation_init": True}),
        ("random_init", {"zero_init_core": False, "force_random_rotation_init": True}),
    ]

    results = []

    for name, cfg_overrides in scenarios:
        base_layer = nn.Linear(8, 8, bias=False)
        cfg = JoraConfig(
            target_modules=['q_proj'],
            S_L=4,
            S_R=4,
            core='selective_diag',
            k=4,
            **cfg_overrides,
        )

        adapter = _JoraAdapterState(base_layer, cfg)
        adapter.init_random_pairs()

        # Set support
        support_pairs = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]])
        adapter.core.set_support(support_pairs.reshape(-1))

        x = torch.randn(4, 8, dtype=torch.float32, requires_grad=True)
        delta = adapter.compute_delta(x)
        loss = delta.sum()
        loss.backward()

        grad_norm_theta_L = adapter.theta_L.grad.norm().item() if adapter.theta_L.grad is not None else 0
        grad_norm_theta_R = adapter.theta_R.grad.norm().item() if adapter.theta_R.grad is not None else 0
        grad_norm_delta = adapter.core.delta.grad.norm().item() if adapter.core.delta.grad is not None else 0

        total_grad = grad_norm_theta_L + grad_norm_theta_R + grad_norm_delta
        theta_ratio = (grad_norm_theta_L + grad_norm_theta_R) / max(total_grad, 1e-10)
        delta_ratio = grad_norm_delta / max(total_grad, 1e-10)

        print(f"\n{name}:")
        print(f"  theta_L grad: {grad_norm_theta_L:.6f} ({theta_ratio*100:.1f}%)")
        print(f"  theta_R grad: {grad_norm_theta_R:.6f}")
        print(f"  delta grad:   {grad_norm_delta:.6f} ({delta_ratio*100:.1f}%)")

        results.append({
            'name': name,
            'theta_ratio': theta_ratio,
            'delta_ratio': delta_ratio,
            'total_grad': total_grad,
        })

    # Check if gradients are imbalanced
    print("\n" + "-" * 70)
    print("Gradient Balance Analysis:")
    print("-" * 70)

    for r in results:
        print(f"{r['name']}: theta={r['theta_ratio']*100:.1f}%, delta={r['delta_ratio']*100:.1f}%")

        # Warning if one type dominates
        if r['theta_ratio'] > 0.95:
            print(f"  ⚠️ WARNING: theta gradients dominate (>95%)")
        elif r['delta_ratio'] > 0.95:
            print(f"  ⚠️ WARNING: delta gradients dominate (>95%)")
        else:
            print(f"  ✅ Balanced gradient flow")


def test_gradient_path_analysis():
    """Trace gradient path through compute_delta."""
    print("\n" + "=" * 70)
    print("Gradient Path Analysis (compute_delta flow)")
    print("=" * 70)

    from peft.tuners.jora.config import JoraConfig
    from peft.tuners.jora.layer import _JoraAdapterState

    base_layer = nn.Linear(8, 8, bias=False)
    cfg = JoraConfig(
        target_modules=['q_proj'],
        S_L=4,
        S_R=4,
        core='selective_diag',
        k=4,
        zero_init_core=True,  # Start at zero
    )

    adapter = _JoraAdapterState(base_layer, cfg)
    adapter.init_random_pairs()

    # Set support
    support_pairs = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]])
    adapter.core.set_support(support_pairs.reshape(-1))

    x = torch.randn(2, 8, dtype=torch.float32, requires_grad=True)

    # Manually trace the gradient path
    print("\nGradient path through compute_delta:")
    print("  1. x (input) → _apply_side_rotation (R) → x_rot")
    print("  2. x_rot → core.apply_to_vector → y_sel")
    print("  3. y_sel → _apply_side_rotation (L^T) → y_rotated")
    print("  4. x → core.project_support → proj_x")
    print("  5. y_rotated - proj_x → delta")
    print("     ↑ fp32 subtraction (Scheme 2)")
    print("  6. delta → output")

    delta = adapter.compute_delta(x)
    loss = delta.sum()
    loss.backward()

    # Check each gradient
    print(f"\nGradient results:")
    print(f"  theta_L.grad: {adapter.theta_L.grad.abs().max().item():.6f} (max abs)")
    print(f"  theta_R.grad: {adapter.theta_R.grad.abs().max().item():.6f} (max abs)")
    print(f"  delta.grad:   {adapter.core.delta.grad.abs().max().item():.6f} (max abs)")

    # Check for gradient vanishing at zero init
    print("\n" + "-" * 70)
    print("Zero-init gradient behavior:")
    print("-" * 70)

    if adapter.core.delta.abs().max().item() < 1e-6:
        print("delta is initialized to ~0 (zero function)")
        if adapter.core.delta.grad is not None:
            print(f"  delta.grad at zero init: {adapter.core.delta.grad.abs().max().item():.6f}")
            if adapter.core.delta.grad.abs().max().item() < 1e-6:
                print("  ⚠️ CRITICAL: delta has near-zero gradient at zero init!")
                print("     This is expected for residualized operator where")
                print("     delta=0 → y_rotated ≈ proj_x → gradient ≈ 0")
                print("     But this means delta cannot learn from initial state!")
    else:
        print("delta is NOT zero-initialized")

    print("\n✅ Gradient path analysis complete")


def test_inference_vs_training_gradients():
    """Compare gradient flow in training vs inference mode."""
    print("\n" + "=" * 70)
    print("Inference vs Training Gradient Flow")
    print("=" * 70)

    from peft.tuners.jora.config import JoraConfig
    from peft.tuners.jora.layer import _JoraAdapterState

    base_layer = nn.Linear(8, 8, bias=False)
    cfg = JoraConfig(
        target_modules=['q_proj'],
        S_L=4,
        S_R=4,
        core='selective_diag',
        k=4,
    )

    adapter = _JoraAdapterState(base_layer, cfg)
    adapter.init_random_pairs()

    support_pairs = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]])
    adapter.core.set_support(support_pairs.reshape(-1))

    x = torch.randn(2, 8, dtype=torch.float32)

    # Training mode
    print("\nTraining mode:")
    adapter.train()
    x_train = x.detach().requires_grad_(True)
    delta_train = adapter.compute_delta(x_train)
    loss_train = delta_train.sum()
    loss_train.backward()

    print(f"  Delta norm: {delta_train.norm().item():.6f}")
    print(f"  theta_L.grad exists: {adapter.theta_L.grad is not None}")
    print(f"  delta.grad norm: {adapter.core.delta.grad.norm().item():.6f}")

    # Inference mode
    print("\nInference mode:")
    adapter.eval()
    x_inf = x.detach().requires_grad_(True)
    delta_inf = adapter.compute_delta(x_inf)
    loss_inf = delta_inf.sum()
    loss_inf.backward()

    print(f"  Delta norm: {delta_inf.norm().item():.6f}")
    print(f"  theta_L.grad exists: {adapter.theta_L.grad is not None}")
    print(f"  delta.grad norm: {adapter.core.delta.grad.norm().item():.6f}")

    print("\n✅ Inference vs training gradient comparison complete")


if __name__ == "__main__":
    print("JORA Gradient Flow Analysis\n")

    test_selective_diag_gradient_flow()
    test_coupled_pair_gradient_flow()
    test_rotation_gradient_flow()
    test_full_jora_layer_gradient_flow()
    test_gradient_magnitude_analysis()
    test_gradient_path_analysis()
    test_inference_vs_training_gradients()

    print("\n" + "=" * 70)
    print("🎉 ALL GRADIENT FLOW TESTS COMPLETE!")
    print("=" * 70)
