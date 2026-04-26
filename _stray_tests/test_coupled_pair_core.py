#!/usr/bin/env python
"""Smoke test for JORA CoupledPairCore (Scheme 3).

Updated to match the corrected API where CoupledPairCore(n_pairs=k)
instead of CoupledPairCore(support_size=k).
"""

import sys
import torch
import torch.nn as nn

sys.path.insert(0, '/home/jqh/Workshop/JORA/src')

from peft.tuners.jora.core import CoupledPairCore, SelectiveDiagCore


def test_coupled_pair_creation():
    """Test that CoupledPairCore can be created."""
    print("=" * 60)
    print("CoupledPairCore Creation Test")
    print("=" * 60)

    # Test with n_pairs=4
    core = CoupledPairCore(n_pairs=4, device='cpu', dtype=torch.float32)
    print(f"\nCreated CoupledPairCore with n_pairs=4")
    print(f"  support_size: {core.support_size}")
    print(f"  n_pairs: {core.n_pairs}")
    print(f"  num_params: {core.num_params}")
    print(f"  pair_blocks shape: {core.pair_blocks.shape}")

    assert core.n_pairs == 4
    assert core.support_size == 8  # 2 * n_pairs
    assert core.num_params == 16   # n_pairs * 4

    print("\n✅ Creation test passed!")


def test_zero_init_identity():
    """Test that zero-initialized blocks act as identity."""
    print("\n" + "=" * 60)
    print("Zero-Init Identity Test")
    print("=" * 60)

    core = CoupledPairCore(n_pairs=2, device='cpu', dtype=torch.float32)
    pairs = torch.tensor([[0, 1], [2, 3]])
    core.set_support_pairs(pairs)

    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    y = core.apply_to_vector(x)

    print(f"\nInput x: {x.tolist()}")
    print(f"Output y (zero-init): {y.tolist()}")

    assert torch.allclose(y, x, atol=1e-6), \
        f"Zero-init should give identity, got {y.tolist()}"

    print("\n✅ Zero-init identity test passed!")


def test_coupled_transformation():
    """Test that non-zero blocks create coupled transformation."""
    print("\n" + "=" * 60)
    print("Coupled Transformation Test")
    print("=" * 60)

    core = CoupledPairCore(n_pairs=2, device='cpu', dtype=torch.float32)
    pairs = torch.tensor([[0, 1], [2, 3]])
    core.set_support_pairs(pairs)

    # Set off-diagonal: transform = [[1.5, 0.3], [0.2, 1.1]]
    core.pair_blocks.data.zero_()
    core.pair_blocks.data[0, 0, 0] = 0.5   # δ_ii
    core.pair_blocks.data[0, 0, 1] = 0.3   # δ_ij
    core.pair_blocks.data[0, 1, 0] = 0.2   # δ_ji
    core.pair_blocks.data[0, 1, 1] = 0.1   # δ_jj

    x = torch.tensor([[2.0, 3.0, 5.0, 7.0]])
    y = core.apply_to_vector(x)

    # With einsum, column 0 multiplies x0, column 1 multiplies x1
    # y0 = 1.5*x0 + 0.2*x1 = 3.0 + 0.6 = 3.6
    # y1 = 0.3*x0 + 1.1*x1 = 0.6 + 3.3 = 3.9
    print(f"\nInput x: {x.tolist()}")
    print(f"Output y: {y.tolist()}")

    assert torch.isclose(y[0, 0], torch.tensor(3.6), atol=1e-5)
    assert torch.isclose(y[0, 1], torch.tensor(3.9), atol=1e-5)

    print("\n✅ Coupled transformation test passed!")


def test_project_support():
    """Test that project_support returns identity on support."""
    print("\n" + "=" * 60)
    print("Project Support Test")
    print("=" * 60)

    core = CoupledPairCore(n_pairs=2, device='cpu', dtype=torch.float32)
    pairs = torch.tensor([[0, 1]])  # Only pair (0, 1)
    core.set_support_pairs(pairs)

    x = torch.tensor([[1.0, 2.0, 999.0, 999.0]])
    y = core.project_support(x)

    print(f"\nInput x: {x.tolist()}")
    print(f"Pairs: {pairs.tolist()}")
    print(f"Projected y: {y.tolist()}")

    expected = torch.tensor([[1.0, 2.0, 0.0, 0.0]])

    assert torch.allclose(y, expected, atol=1e-6)

    print("\n✅ Project support test passed!")


def test_vs_selective_diag():
    """Compare CoupledPairCore with SelectiveDiagCore for same k."""
    print("\n" + "=" * 60)
    print("Comparison: CoupledPairCore vs SelectiveDiagCore (same k)")
    print("=" * 60)

    k = 16

    diag_core = SelectiveDiagCore(support_size=2*k, device='cpu', dtype=torch.float32)
    coupled_core = CoupledPairCore(n_pairs=k, device='cpu', dtype=torch.float32)

    print(f"\nk = {k} (number of pairs)")

    print(f"\nSelectiveDiagCore:")
    print(f"  support_size = {2*k}")
    print(f"  num_params = {diag_core.num_params}")

    print(f"\nCoupledPairCore:")
    print(f"  n_pairs = {coupled_core.n_pairs}")
    print(f"  support_size = {coupled_core.support_size}")
    print(f"  num_params = {coupled_core.num_params}")

    # Note: for same k, CoupledPairCore has 2x params
    assert coupled_core.num_params == 2 * diag_core.num_params

    print("\n✅ Comparison test passed!")


def test_batched_input():
    """Test with batched input."""
    print("\n" + "=" * 60)
    print("Batched Input Test")
    print("=" * 60)

    core = CoupledPairCore(n_pairs=2, device='cpu', dtype=torch.float32)
    pairs = torch.tensor([[0, 1], [2, 3]])
    core.set_support_pairs(pairs)

    core.pair_blocks.data.zero_()
    core.pair_blocks.data[0, 0, 1] = 0.3
    core.pair_blocks.data[0, 1, 0] = 0.3

    x = torch.tensor([
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0],
    ])

    y = core.apply_to_vector(x)

    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {y.shape}")

    assert y.shape == x.shape

    # Verify coupling: y[batch, 0] should depend on x[batch, 1]
    # With transform [[1, 0.3], [0.3, 1]]
    # y0 = x0 + 0.3*x1
    for b in range(3):
        expected_0 = x[b, 0] + 0.3 * x[b, 1]
        assert torch.isclose(y[b, 0], expected_0, atol=1e-5), \
            f"Batch {b}: expected {expected_0}, got {y[b, 0]}"

    print("\n✅ Batched input test passed!")


def test_build_core():
    """Test that build_core can create CoupledPairCore."""
    print("\n" + "=" * 60)
    print("build_core Integration Test")
    print("=" * 60)

    from peft.tuners.jora.core import build_core
    from peft.tuners.jora.config import JoraConfig

    cfg = JoraConfig(
        target_modules=['q_proj'],
        core='coupled_pair',
        k=16,
    )

    core = build_core(
        core_type='coupled_pair',
        n=32,
        m=32,
        device='cpu',
        dtype=torch.float32,
        cfg=cfg,
    )

    print(f"\nBuilt CoupledPairCore via build_core:")
    print(f"  Type: {type(core).__name__}")
    print(f"  k from config: {cfg.k}")
    print(f"  n_pairs: {core.n_pairs}")
    print(f"  support_size: {core.support_size}")
    print(f"  num_params: {core.num_params}")

    assert isinstance(core, CoupledPairCore)
    assert core.n_pairs == cfg.k

    print("\n✅ build_core integration test passed!")


if __name__ == "__main__":
    print("Testing JORA CoupledPairCore (Scheme 3)...\n")

    test_coupled_pair_creation()
    test_zero_init_identity()
    test_coupled_transformation()
    test_project_support()
    test_vs_selective_diag()
    test_batched_input()
    test_build_core()

    print("\n" + "=" * 60)
    print("🎉 All CoupledPairCore tests passed!")
    print("=" * 60)
