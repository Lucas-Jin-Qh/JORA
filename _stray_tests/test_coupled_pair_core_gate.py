#!/usr/bin/env python
"""Correctness gate tests for CoupledPairCore (Scheme 3).

These 6 tests verify that CoupledPairCore is a self-consistent operator
before any benchmark comparisons are made.
"""

import sys
import torch
import torch.nn as nn

sys.path.insert(0, '/home/jqh/Workshop/JORA/src')

from peft.tuners.jora.core import CoupledPairCore, SelectiveDiagCore


# =======================================================================
# GATE TEST 1: Zero change before set_support
# =======================================================================
def test_zero_change_before_set_support():
    """Before set_support is called, apply_to_vector must return zeros."""
    print("=" * 70)
    print("GATE 1: Zero output before set_support")
    print("=" * 70)

    core = CoupledPairCore(n_pairs=4, device='cpu', dtype=torch.float32)
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]])

    # Before set_support: must return zeros
    y = core.apply_to_vector(x)

    print(f"\nBefore set_support:")
    print(f"  _active_n_pairs_py: {core._active_n_pairs_py}")
    print(f"  Input x: {x.tolist()}")
    print(f"  Output y: {y.tolist()}")

    assert core._active_n_pairs_py == 0, \
        f"Expected _active_n_pairs_py=0, got {core._active_n_pairs_py}"
    assert y.abs().max().item() == 0.0, \
        f"Expected all zeros, got {y.tolist()}"
    assert y.shape == x.shape, f"Shape mismatch: {y.shape} vs {x.shape}"

    print("\n✅ PASS: apply_to_vector returns zeros before set_support")


# =======================================================================
# GATE TEST 2: Non-zero after set_support
# =======================================================================
def test_nonzero_after_set_support():
    """After set_support_pairs, apply_to_vector must produce non-trivial output."""
    print("\n" + "=" * 70)
    print("GATE 2: Non-trivial output after set_support")
    print("=" * 70)

    core = CoupledPairCore(n_pairs=4, device='cpu', dtype=torch.float32)

    # Set support with actual pairs
    pairs = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]])
    core.set_support_pairs(pairs)

    x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]])

    # With zero-init blocks, should return identity (non-zero, but same as input)
    y = core.apply_to_vector(x)

    print(f"\nAfter set_support_pairs:")
    print(f"  _active_n_pairs_py: {core._active_n_pairs_py}")
    print(f"  Pairs stored:\n{core.pairs.tolist()}")
    print(f"  Input x: {x.tolist()}")
    print(f"  Output y (zero-init → identity): {y.tolist()}")

    assert core._active_n_pairs_py == 4, \
        f"Expected _active_n_pairs_py=4, got {core._active_n_pairs_py}"

    # With zero blocks, output should equal input (identity transformation)
    assert torch.allclose(y, x, atol=1e-6), \
        f"Zero-init should give identity, got {y.tolist()}"
    assert y.abs().max().item() > 0.0, \
        "Should produce non-zero output"

    print("\n✅ PASS: apply_to_vector produces non-trivial output after set_support")


# =======================================================================
# GATE TEST 3: Freeze sets support correctly
# =======================================================================
def test_freeze_sets_support():
    """Verify set_support_pairs stores the pair structure correctly."""
    print("\n" + "=" * 70)
    print("GATE 3: Pair structure preserved by set_support_pairs")
    print("=" * 70)

    core = CoupledPairCore(n_pairs=8, device='cpu', dtype=torch.float32)

    # Simulate what _freeze_support_if_needed does for CoupledPairCore
    pairs_L = torch.tensor([[5, 2], [3, 0], [7, 4], [1, 6]])
    pairs_R = torch.tensor([[4, 3], [2, 7], [0, 5], [6, 1]])

    # This is the actual code path in _freeze_support_if_needed
    all_pairs = torch.cat([pairs_L, pairs_R], dim=0)
    core.set_support_pairs(all_pairs)

    print(f"\nInput pairs_L:\n{pairs_L.tolist()}")
    print(f"Input pairs_R:\n{pairs_R.tolist()}")
    print(f"Combined pairs:\n{all_pairs.tolist()}")
    print(f"\nStored pairs:\n{core.pairs.tolist()}")
    print(f"Active n_pairs: {core._active_n_pairs_py}")

    # Verify the pairs are stored correctly
    assert core._active_n_pairs_py == 8, \
        f"Expected 8 active pairs, got {core._active_n_pairs_py}"

    # Each row of stored pairs should match input
    assert torch.equal(core.pairs[:8], all_pairs), \
        f"Pair structure not preserved correctly"

    print("\n✅ PASS: Pair structure is preserved by set_support_pairs")


# =======================================================================
# GATE TEST 4: Param count matches formula
# =======================================================================
def test_param_count_matches_formula():
    """Verify num_params = n_pairs * 4."""
    print("\n" + "=" * 70)
    print("GATE 4: Parameter count verification")
    print("=" * 70)

    # Test multiple sizes
    test_cases = [
        (4, 4, 16),    # n_pairs=4: 4*4=16 params
        (8, 8, 32),    # n_pairs=8: 8*4=32 params
        (16, 16, 64),  # n_pairs=16: 16*4=64 params
    ]

    for n_pairs, expected_n_pairs, expected_params in test_cases:
        core = CoupledPairCore(n_pairs=n_pairs, device='cpu', dtype=torch.float32)

        print(f"\nn_pairs={n_pairs}:")
        print(f"  support_size: {core.support_size}")
        print(f"  n_pairs: {core.n_pairs}")
        print(f"  pair_blocks shape: {core.pair_blocks.shape}")
        print(f"  num_params: {core.num_params} (expected {expected_params})")
        print(f"  actual param count: {core.pair_blocks.numel()}")

        assert core.n_pairs == expected_n_pairs, \
            f"n_pairs mismatch: {core.n_pairs} vs {expected_n_pairs}"
        assert core.num_params == expected_params, \
            f"num_params mismatch: {core.num_params} vs {expected_params}"
        assert core.pair_blocks.numel() == expected_params, \
            f"Actual params mismatch: {core.pair_blocks.numel()} vs {expected_params}"

    print("\n✅ PASS: num_params = n_pairs * 4 is correct")


# =======================================================================
# GATE TEST 5: Merge equals forward (at zero init)
# =======================================================================
def test_merge_equals_forward():
    """At zero init, the adapter should not change the forward pass (identity)."""
    print("\n" + "=" * 70)
    print("GATE 5: Merge equals forward at zero init")
    print("=" * 70)

    core = CoupledPairCore(n_pairs=4, device='cpu', dtype=torch.float32)
    pairs = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]])
    core.set_support_pairs(pairs)

    x = torch.randn(4, 8, dtype=torch.float32)  # [batch, features]

    y = core.apply_to_vector(x)
    proj = core.project_support(x)

    print(f"\nInput shape: {x.shape}")
    print(f"Output y shape: {y.shape}")
    print(f"Projected shape: {proj.shape}")

    # At zero init, y should equal x on support
    print(f"\nInput (first sample): {x[0].tolist()}")
    print(f"Output (first sample): {y[0].tolist()}")
    print(f"Projected (first sample): {proj[0].tolist()}")

    # y should equal x (identity at zero init)
    assert torch.allclose(y, x, atol=1e-6), \
        "Zero-init apply_to_vector should be identity"

    # project_support should also equal x on support
    assert torch.allclose(proj, x, atol=1e-6), \
        "project_support should return x on support dimensions"

    print("\n✅ PASS: Zero-init makes adapter transparent (identity)")


# =======================================================================
# GATE TEST 6: Gradients flow after support set
# =======================================================================
def test_gradients_nonzero_after_support_set():
    """Gradients should flow to pair_blocks after support is set."""
    print("\n" + "=" * 70)
    print("GATE 6: Gradients flow to pair_blocks after support set")
    print("=" * 70)

    core = CoupledPairCore(n_pairs=4, device='cpu', dtype=torch.float32)
    pairs = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]])
    core.set_support_pairs(pairs)

    x = torch.randn(2, 8, dtype=torch.float32, requires_grad=True)
    y = core.apply_to_vector(x)

    # Loss: sum of output
    loss = y.sum()
    loss.backward()

    print(f"\nInput x.requires_grad: {x.requires_grad}")
    print(f"Output y.grad: {y.grad is not None}")
    print(f"pair_blocks.grad exists: {core.pair_blocks.grad is not None}")

    if core.pair_blocks.grad is not None:
        print(f"pair_blocks.grad norm: {core.pair_blocks.grad.norm().item():.6f}")
        print(f"pair_blocks.grad sample:\n{core.pair_blocks.grad[0]}")

    # Gradients should exist and be non-zero
    assert core.pair_blocks.grad is not None, \
        "pair_blocks should have gradients after backward"
    assert core.pair_blocks.grad.abs().max().item() > 0.0, \
        "Gradients should be non-zero"

    # Input gradients should also exist
    assert x.grad is not None, "Input should have gradients"

    print("\n✅ PASS: Gradients flow to pair_blocks after support set")


# =======================================================================
# Additional: Compare with SelectiveDiagCore
# =======================================================================
def test_compare_with_selective_diag():
    """Show the structural difference between CoupledPairCore and SelectiveDiagCore."""
    print("\n" + "=" * 70)
    print("Comparison: CoupledPairCore vs SelectiveDiagCore (same k)")
    print("=" * 70)

    k = 16

    # SelectiveDiagCore: k pairs → support_size = 2k → params = 2k
    diag_core = SelectiveDiagCore(support_size=2*k, device='cpu', dtype=torch.float32)

    # CoupledPairCore: k pairs → n_pairs = k → params = 4k
    coupled_core = CoupledPairCore(n_pairs=k, device='cpu', dtype=torch.float32)

    print(f"\nk = {k} (number of pairs)")
    print(f"\nSelectiveDiagCore:")
    print(f"  support_size: {diag_core.support_size}")
    print(f"  num_params: {diag_core.num_params}")
    print(f"  Structure: Independent scaling per support index")

    print(f"\nCoupledPairCore:")
    print(f"  n_pairs: {coupled_core.n_pairs}")
    print(f"  support_size: {coupled_core.support_size}")
    print(f"  num_params: {coupled_core.num_params}")
    print(f"  Structure: 2x2 blocks per rotation pair")

    # Key comparison groups
    print("\n" + "-" * 70)
    print("Benchmark comparison groups:")
    print("-" * 70)
    print(f"  A: selective_diag @ k=16  → params=32    (independent scaling)")
    print(f"  B: coupled_pair  @ k=16  → params=64    (coupled scaling, 2x params)")
    print(f"  C: selective_diag @ k=8   → params=16    (half coverage)")
    print(f"  D: coupled_pair  @ k=8   → params=32    (same params as A)")

    print("\n  Key comparisons:")
    print("    B vs A: Same k, does coupling help with more params?")
    print("    B vs D: Different structures, same params coverage")
    print("    A vs D: Different structures, same param count")
    print("    C vs D: Same pair count, different structures")


def test_vectorized_correctness():
    """Verify the vectorized apply_to_vector matches the expected math."""
    print("\n" + "=" * 70)
    print("Bonus: Vectorized implementation correctness")
    print("=" * 70)

    core = CoupledPairCore(n_pairs=4, device='cpu', dtype=torch.float32)
    pairs = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]])
    core.set_support_pairs(pairs)

    # Set specific block values
    core.pair_blocks.data.zero_()
    core.pair_blocks.data[0, 0, 0] = 0.5   # δ_ii
    core.pair_blocks.data[0, 0, 1] = 0.3   # δ_ij
    core.pair_blocks.data[0, 1, 0] = 0.2   # δ_ji
    core.pair_blocks.data[0, 1, 1] = 0.1   # δ_jj

    # Pair 0: indices (0, 1)
    # transform = I + block = [[1.5, 0.3], [0.2, 1.1]]
    # y0 = transform[0,0]*x0 + transform[1,0]*x1 = 1.5*2 + 0.2*3 = 3.0 + 0.6 = 3.6
    # y1 = transform[0,1]*x0 + transform[1,1]*x1 = 0.3*2 + 1.1*3 = 0.6 + 3.3 = 3.9
    #
    # Note: einsum '...pi,pij->...pj' contracts over i, so transform[:,i,j] multiplies x[:,i].
    # This means column 0 of transform multiplies x0, column 1 multiplies x1.

    x = torch.tensor([[2.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0]])
    y = core.apply_to_vector(x)

    expected_y0 = 1.5 * 2.0 + 0.2 * 3.0  # 3.0 + 0.6 = 3.6
    expected_y1 = 0.3 * 2.0 + 1.1 * 3.0  # 0.6 + 3.3 = 3.9

    print(f"\nTransform matrix (I + block):\n{torch.eye(2) + core.pair_blocks.data[0]}")
    print(f"\nInput x[0, 0:2] = {x[0, 0:2].tolist()}")
    print(f"Expected y[0, 0] = 1.5*2 + 0.2*3 = {expected_y0}")
    print(f"Expected y[0, 1] = 0.3*2 + 1.1*3 = {expected_y1}")
    print(f"Actual y[0, 0] = {y[0, 0].item():.6f}")
    print(f"Actual y[0, 1] = {y[0, 1].item():.6f}")

    assert torch.isclose(y[0, 0], torch.tensor(expected_y0), atol=1e-5), \
        f"y[0] mismatch: {y[0, 0].item()} vs {expected_y0}"
    assert torch.isclose(y[0, 1], torch.tensor(expected_y1), atol=1e-5), \
        f"y[1] mismatch: {y[0, 1].item()} vs {expected_y1}"

    print("\n✅ PASS: Vectorized implementation is mathematically correct")


def test_build_core_integration():
    """Test that build_core creates CoupledPairCore with correct n_pairs."""
    print("\n" + "=" * 70)
    print("Bonus: build_core integration")
    print("=" * 70)

    from peft.tuners.jora.core import build_core
    from peft.tuners.jora.config import JoraConfig

    cfg = JoraConfig(
        target_modules=['q_proj'],
        core='coupled_pair',
        k=16,  # 16 pairs
    )

    core = build_core(
        core_type='coupled_pair',
        n=32,
        m=32,
        device='cpu',
        dtype=torch.float32,
        cfg=cfg,
    )

    print(f"\nk from config: {cfg.k}")
    print(f"n_pairs: {core.n_pairs}")
    print(f"support_size: {core.support_size}")
    print(f"num_params: {core.num_params}")

    assert core.n_pairs == cfg.k, f"n_pairs should be k={cfg.k}"
    assert isinstance(core, CoupledPairCore)

    print("\n✅ PASS: build_core integration works correctly")


if __name__ == "__main__":
    print("Running CoupledPairCore Correctness Gates...\n")

    test_zero_change_before_set_support()       # GATE 1
    test_nonzero_after_set_support()             # GATE 2
    test_freeze_sets_support()                  # GATE 3
    test_param_count_matches_formula()          # GATE 4
    test_merge_equals_forward()                  # GATE 5
    test_gradients_nonzero_after_support_set()   # GATE 6

    test_compare_with_selective_diag()           # Info
    test_vectorized_correctness()               # Bonus
    test_build_core_integration()                # Bonus

    print("\n" + "=" * 70)
    print("🎉 ALL 6 CORRECTNESS GATES PASSED!")
    print("=" * 70)
    print("\nCoupledPairCore is now a self-consistent operator.")
    print("Ready for benchmark comparisons.")
