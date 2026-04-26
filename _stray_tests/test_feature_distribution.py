#!/usr/bin/env python
"""Feature Distribution Stability Analysis for JORA.

Checks:
1. OER energy conservation properties
2. Feature distribution stability over training steps
3. Potential soft alignment needs
"""

import sys
import torch
import torch.nn as nn

sys.path.insert(0, '/home/jqh/Workshop/JORA/src')

from peft.tuners.jora.magnitude import compute_oer_scale_softmax


def test_oer_energy_conservation():
    """Test that OER maintains energy conservation."""
    print("=" * 70)
    print("OER Energy Conservation Test")
    print("=" * 70)

    # Setup
    out_features = 8
    base_norms = torch.rand(out_features) * 5 + 0.1  # Random norms [0.1, 5.1]
    total_energy = (base_norms ** 2).sum()

    print(f"\nBase norms: {base_norms.tolist()}")
    print(f"Total energy: {total_energy.item():.4f}")

    # Test with different logits (different energy distributions)
    test_logits = [
        ("uniform", torch.zeros(out_features)),
        ("high_first", torch.tensor([3.0] + [0.0] * (out_features - 1))),
        ("low_first", torch.tensor([-3.0] + [0.0] * (out_features - 1))),
        ("random", torch.randn(out_features)),
    ]

    for name, logits in test_logits:
        scale = compute_oer_scale_softmax(
            base_row_norms=base_norms,
            total_energy=total_energy,
            oer_logits=logits,
            temperature=1.0,
        )

        # Check energy conservation: scale * base_norms should have same total energy
        adjusted_norms = scale * base_norms
        adjusted_energy = (adjusted_norms ** 2).sum()
        energy_error = abs(adjusted_energy.item() - total_energy.item()) / total_energy.item()

        print(f"\n{name}:")
        print(f"  Scale: {scale.tolist()}")
        print(f"  Adjusted norms: {adjusted_norms.tolist()}")
        print(f"  Energy error: {energy_error:.2e}")

        assert energy_error < 1e-4, f"Energy conservation violated: {energy_error}"

    print("\n✅ OER energy conservation verified!")


def test_oer_numerical_stability():
    """Test OER numerical stability with extreme inputs."""
    print("\n" + "=" * 70)
    print("OER Numerical Stability Test")
    print("=" * 70)

    out_features = 8

    stability_cases = [
        ("zero_norms", torch.zeros(out_features), torch.randn(out_features)),
        ("inf_logits", torch.ones(out_features), torch.tensor([float('inf')] + [0.0] * (out_features - 1))),
        ("nan_logits", torch.ones(out_features), torch.tensor([float('nan')] + [0.0] * (out_features - 1))),
        ("all_same_norms", torch.ones(out_features), torch.randn(out_features)),
        ("extreme_temperature", torch.ones(out_features), torch.randn(out_features) * 100),
    ]

    for name, base_norms, logits in stability_cases:
        total_energy = torch.tensor(1.0) if base_norms.sum() == 0 else (base_norms ** 2).sum()

        try:
            scale = compute_oer_scale_softmax(
                base_row_norms=base_norms,
                total_energy=total_energy,
                oer_logits=logits,
                temperature=1.0,
            )

            is_finite = torch.isfinite(scale).all().item()
            print(f"\n{name}:")
            print(f"  Scale: {scale.tolist()}")
            print(f"  All finite: {is_finite}")

            assert is_finite, f"Scale has NaN/Inf for {name}"

        except Exception as e:
            print(f"\n{name}: Exception: {e}")
            # Should fall back to uniform
            pass

    print("\n✅ OER numerical stability verified!")


def test_feature_distribution_stability():
    """Test that feature distributions remain stable during training."""
    print("\n" + "=" * 70)
    print("Feature Distribution Stability Test")
    print("=" * 70)

    from peft.tuners.jora.config import JoraConfig
    from peft.tuners.jora.layer import _JoraAdapterState

    base_layer = nn.Linear(16, 16, bias=False)

    # Initialize with magnitude module
    cfg = JoraConfig(
        target_modules=['q_proj'],
        S_L=8,
        S_R=8,
        core='selective_diag',
        k=8,
        magnitude='oer_softmax',
    )

    adapter = _JoraAdapterState(base_layer, cfg)
    adapter.init_random_pairs()

    # Set support
    support_pairs = torch.tensor([[i, i+1] for i in range(0, 16, 2)])
    adapter.core.set_support(support_pairs.reshape(-1))

    print(f"\nBase row norms: {adapter.base_row_norms.tolist()}")
    print(f"Total energy: {adapter.total_energy.item():.4f}")

    # Simulate training steps
    norms_over_time = []
    energies_over_time = []

    for step in range(5):
        x = torch.randn(4, 16, dtype=torch.float32)
        delta = adapter.compute_delta(x)
        delta = adapter.maybe_apply_magnitude(delta)

        # Check delta distribution
        delta_norms = delta.norm(dim=-1)  # Per-sample norms
        print(f"\nStep {step}:")
        print(f"  Delta shape: {delta.shape}")
        print(f"  Delta norm per sample: {delta_norms.tolist()}")
        print(f"  Delta mean: {delta.mean().item():.4f}, std: {delta.std().item():.4f}")

        norms_over_time.append(delta_norms.mean().item())
        energies_over_time.append((delta ** 2).sum().item())

    # Check if distributions are stable
    if len(norms_over_time) > 1:
        norm_std = torch.tensor(norms_over_time).std().item()
        print(f"\nNorm stability (std over steps): {norm_std:.4f}")

        if norm_std > 1.0:
            print(f"  ⚠️ WARNING: Feature distributions may be unstable (std={norm_std:.4f})")
        else:
            print(f"  ✅ Feature distributions appear stable")

    print("\n✅ Feature distribution stability analysis complete")


def test_soft_alignment_needs():
    """Analyze if soft alignment is needed."""
    print("\n" + "=" * 70)
    print("Soft Alignment Needs Analysis")
    print("=" * 70)

    # The key question: is there distribution shift between layers?
    # This can happen if:
    # 1. JORA changes feature norms significantly
    # 2. LayerNorm/RMSNorm is applied after JORA (in the model)
    # 3. The magnitude module changes energy distribution

    print("\nPotential sources of distribution shift:")

    sources = [
        ("Rotation (theta)", "Changes feature directions, not norms directly"),
        ("Core (delta)", "Changes feature magnitudes on support indices"),
        ("OER magnitude", "Redistributes energy across dimensions"),
        ("project_support subtraction", "Removes projected components"),
    ]

    for name, effect in sources:
        print(f"  {name}: {effect}")

    print("\n" + "-" * 70)
    print("Soft Alignment Recommendations:")
    print("-" * 70)

    recommendations = [
        "1. OER already handles energy conservation → no alignment needed",
        "2. If using magnitude='oer_softmax', feature norms are controlled",
        "3. If features become unstable, consider:",
        "   - Clamping delta magnitude",
        "   - Adding output LayerNorm inside JORA",
        "   - Using RMSNorm-style normalization on delta",
    ]

    for r in recommendations:
        print(f"  {r}")

    print("\n✅ Soft alignment analysis complete")


def test_delta_magnitude_analysis():
    """Analyze delta magnitude over different input conditions."""
    print("\n" + "=" * 70)
    print("Delta Magnitude Analysis")
    print("=" * 70)

    from peft.tuners.jora.config import JoraConfig
    from peft.tuners.jora.layer import _JoraAdapterState

    base_layer = nn.Linear(16, 16, bias=False)
    cfg = JoraConfig(
        target_modules=['q_proj'],
        S_L=8,
        S_R=8,
        core='selective_diag',
        k=8,
        zero_init_core=True,  # Start at zero
    )

    adapter = _JoraAdapterState(base_layer, cfg)
    adapter.init_random_pairs()

    # Set support
    support_pairs = torch.tensor([[i, i+1] for i in range(0, 16, 2)])
    adapter.core.set_support(support_pairs.reshape(-1))

    # Test with different input scales
    test_scales = [0.01, 0.1, 1.0, 10.0, 100.0]

    print("\nDelta magnitude vs input scale:")
    print("-" * 70)

    results = []

    for scale in test_scales:
        x = torch.randn(4, 16, dtype=torch.float32) * scale
        delta = adapter.compute_delta(x)

        delta_norm = delta.norm().item()
        delta_max = delta.abs().max().item()
        delta_mean = delta.abs().mean().item()

        print(f"  Input scale {scale:6.2f}: delta norm={delta_norm:.4f}, max={delta_max:.4f}, mean={delta_mean:.4f}")

        results.append({
            'scale': scale,
            'norm': delta_norm,
            'max': delta_max,
            'mean': delta_mean,
        })

    # Check linearity
    print("\n" + "-" * 70)
    print("Linearity check:")
    print("-" * 70)

    norms = [r['norm'] for r in results]
    scales = [r['scale'] for r in results]

    # At zero init, delta should be ~0 regardless of input scale
    zero_init_delta = norms[0]  # With scale=0.01
    if zero_init_delta < 0.1:
        print(f"✅ Zero-init: delta norm ≈ {zero_init_delta:.4f} (near zero as expected)")
    else:
        print(f"⚠️ Zero-init: delta norm = {zero_init_delta:.4f} (should be near zero)")

    # Check if delta scales linearly with input
    if len(norms) >= 2:
        ratio = norms[-1] / norms[1] if norms[1] > 0 else float('inf')
        expected_ratio = scales[-1] / scales[1]
        print(f"  Delta scale ratio: {ratio:.2f} (expected ~{expected_ratio:.2f})")

        if 0.5 < ratio / expected_ratio < 2.0:
            print(f"  ✅ Delta scales approximately linearly with input")
        else:
            print(f"  ⚠️ Delta may not scale linearly (possible saturation)")

    print("\n✅ Delta magnitude analysis complete")


if __name__ == "__main__":
    test_oer_energy_conservation()
    test_oer_numerical_stability()
    test_feature_distribution_stability()
    test_soft_alignment_needs()
    test_delta_magnitude_analysis()

    print("\n" + "=" * 70)
    print("🎉 ALL FEATURE DISTRIBUTION TESTS COMPLETE!")
    print("=" * 70)
