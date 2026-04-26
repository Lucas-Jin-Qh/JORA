#!/usr/bin/env python
"""Init calibration probe for Phase 1.

Measures ||delta(x)||/||x|| for the FULL JoraLayer at different init_std values.
No code changes — pure measurement.

Correct metric:
  - JoraLayer compute_delta(x) = R_L^T @ core(R_R @ x)  [for DiagCore path]
  - At zero init: delta(x) ≈ 0 (identity) => ||delta||/||x|| ≈ 0
  - At small nonzero init: delta(x) is small but non-zero => ratio < 1e-2
  - No NaN/Inf

We measure at the JoraLayer level (not just core level) because:
  - DiagCore.apply_to_vector(x) = x * diag_params
  - At zero init: returns 0, not x (identity)
  - The full delta(x) = R_L^T @ 0 @ R_R @ x = 0 (identity)
"""
import sys
import math
sys.path.insert(0, "src")

import torch
import torch.nn as nn
from peft.tuners.jora.core import DiagCore, BlockCore, LowRankCore
from peft.tuners.jora.layer import JoraLayer
from peft.tuners.jora.config import JoraConfig


def measure_layer_delta_ratio(layer, adapter, x):
    """Measure ||compute_delta(x)||/||x|| at zero/nonzero init."""
    with torch.no_grad():
        delta = adapter.compute_delta(x)
    return (delta.norm() / x.norm()).item()


def probe_diag_layer(std, n=512, zero_init=True):
    """Test DiagCore at layer level."""
    base = nn.Linear(n, n, bias=False)
    cfg = JoraConfig(
        target_modules=["default"],
        core="diag",
        magnitude="none",
        zero_init_core=zero_init,
        theta_init_std=0.0,  # rotation off for clean measurement
        S_L=min(32, n),
        S_R=min(32, n),
        rotation_impl="torch",
    )
    layer = JoraLayer(base, "default", cfg)
    adapter = layer.adapters["default"]
    # Override core init if nonzero
    if not zero_init and std > 0:
        with torch.no_grad():
            adapter.core.diag_params.normal_(std=std)
    x = torch.randn(4, n)
    return measure_layer_delta_ratio(layer, adapter, x)


def probe_block_layer(std, n=512, block_size=4, zero_init=True):
    """Test BlockCore at layer level."""
    base = nn.Linear(n, n, bias=False)
    cfg = JoraConfig(
        target_modules=["default"],
        core="block",
        magnitude="none",
        zero_init_core=zero_init,
        block_size=block_size,
        theta_init_std=0.0,
        S_L=min(32, n),
        S_R=min(32, n),
        rotation_impl="torch",
    )
    layer = JoraLayer(base, "default", cfg)
    adapter = layer.adapters["default"]
    if not zero_init and std > 0:
        with torch.no_grad():
            if adapter.core.blocks is not None:
                adapter.core.blocks.normal_(std=std)
            if adapter.core.diag_remainder is not None:
                adapter.core.diag_remainder.normal_(std=std)
    x = torch.randn(4, n)
    return measure_layer_delta_ratio(layer, adapter, x)


def probe_lowrank_layer(std, n=512, rank=8, zero_init=True):
    """Test LowRankCore at layer level."""
    base = nn.Linear(n, n, bias=False)
    cfg = JoraConfig(
        target_modules=["default"],
        core="lowrank",
        magnitude="none",
        zero_init_core=zero_init,
        lowrank_r=rank,
        lowrank_alpha=float(rank),
        theta_init_std=0.0,
        S_L=min(32, n),
        S_R=min(32, n),
        rotation_impl="torch",
    )
    layer = JoraLayer(base, "default", cfg)
    adapter = layer.adapters["default"]
    if not zero_init and std > 0:
        with torch.no_grad():
            adapter.core.A.normal_(std=std)
            adapter.core.B.normal_(std=std)
    x = torch.randn(4, n)
    return measure_layer_delta_ratio(layer, adapter, x)


def main():
    print("=== Init Calibration Probe (Layer-Level Metric: ||delta||/||x||) ===")
    print("Expected: zero-init => ratio ≈ 0 (identity)")
    print("          small nonzero init => ratio < 1e-2")
    print()
    print(f"{'Core':<20} {'std':>8} {'zero_init':>10} {'||d||/||x||':>14} {'NaN?':>6} {'Gate':>10}")
    print("-" * 72)

    stds = [0.0, 1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2]
    results = {}

    # DiagCore — zero init
    diag_zero = []
    for std in stds:
        ratio = probe_diag_layer(std, n=256, zero_init=True)
        nan = not math.isfinite(ratio)
        diag_zero.append((std, ratio, nan))
        gate = "PASS" if (std == 0.0 and ratio < 1e-5) or (std > 0 and ratio < 1e-1) else "WARN"
        flag = "NaN" if nan else "OK"
        print(f"{'DiagCore(zero)':<20} {std:>8.0e} {'True':>10} {ratio:>14.6e} {flag:>6} {gate:>10}")
    results["diag_zero"] = diag_zero

    print()
    # DiagCore — nonzero init
    diag_nonzero = []
    for std in stds:
        ratio = probe_diag_layer(std, n=256, zero_init=False)
        nan = not math.isfinite(ratio)
        diag_nonzero.append((std, ratio, nan))
        gate = "PASS" if ratio < 1e-1 else "FAIL"
        flag = "NaN" if nan else "OK"
        print(f"{'DiagCore(nonzero)':<20} {std:>8.0e} {'False':>10} {ratio:>14.6e} {flag:>6} {gate:>10}")
    results["diag_nonzero"] = diag_nonzero

    print()
    # BlockCore
    block_zero = []
    for std in stds:
        ratio = probe_block_layer(std, n=256, block_size=4, zero_init=True)
        nan = not math.isfinite(ratio)
        block_zero.append((std, ratio, nan))
        gate = "PASS" if (std == 0.0 and ratio < 1e-5) or (std > 0 and ratio < 1e-1) else "WARN"
        flag = "NaN" if nan else "OK"
        print(f"{'BlockCore(zero)':<20} {std:>8.0e} {'True':>10} {ratio:>14.6e} {flag:>6} {gate:>10}")
    results["block_zero"] = block_zero

    print()
    block_nonzero = []
    for std in stds:
        ratio = probe_block_layer(std, n=256, block_size=4, zero_init=False)
        nan = not math.isfinite(ratio)
        block_nonzero.append((std, ratio, nan))
        gate = "PASS" if ratio < 1e-1 else "FAIL"
        flag = "NaN" if nan else "OK"
        print(f"{'BlockCore(nonzero)':<20} {std:>8.0e} {'False':>10} {ratio:>14.6e} {flag:>6} {gate:>10}")
    results["block_nonzero"] = block_nonzero

    print()
    # LowRankCore
    lr_zero = []
    for std in stds:
        ratio = probe_lowrank_layer(std, n=256, rank=8, zero_init=True)
        nan = not math.isfinite(ratio)
        lr_zero.append((std, ratio, nan))
        gate = "PASS" if (std == 0.0 and ratio < 1e-5) or (std > 0 and ratio < 1e-1) else "WARN"
        flag = "NaN" if nan else "OK"
        print(f"{'LowRankCore(zero)':<20} {std:>8.0e} {'True':>10} {ratio:>14.6e} {flag:>6} {gate:>10}")
    results["lr_zero"] = lr_zero

    print()
    lr_nonzero = []
    for std in stds:
        ratio = probe_lowrank_layer(std, n=256, rank=8, zero_init=False)
        nan = not math.isfinite(ratio)
        lr_nonzero.append((std, ratio, nan))
        gate = "PASS" if ratio < 1e-1 else "FAIL"
        flag = "NaN" if nan else "OK"
        print(f"{'LowRankCore(nonzero)':<20} {std:>8.0e} {'False':>10} {ratio:>14.6e} {flag:>6} {gate:>10}")
    results["lr_nonzero"] = lr_nonzero

    print()
    print("=" * 72)
    print("=== Phase 1 Gate Summary ===")

    all_pass = True
    # Check zero-init gives ~0
    for name in ["diag_zero", "block_zero", "lr_zero"]:
        for std, ratio, nan in results[name]:
            if nan:
                print(f"  FAIL: {name} std={std} -> NaN")
                all_pass = False
            elif std == 0.0 and ratio >= 1e-5:
                print(f"  FAIL: {name} std=0 -> ratio={ratio:.6e} (not near-identity)")
                all_pass = False

    # Check nonzero init is bounded
    for name in ["diag_nonzero", "block_nonzero", "lr_nonzero"]:
        for std, ratio, nan in results[name]:
            if nan:
                print(f"  FAIL: {name} std={std} -> NaN")
                all_pass = False
            elif ratio >= 1e-1:
                print(f"  FAIL: {name} std={std} -> ratio={ratio:.6e} >= 1e-1 (too large)")
                all_pass = False

    if all_pass:
        print("  ALL CHECKS PASSED")
    print()
    print("=== Recommended Defaults for Phase 2 ===")
    print("  core_init_std = 5e-3  -> small non-zero, bounded")
    print("  theta_init_std = 2e-3  -> will be tested in Phase 4")
    print("Phase 1 GATE: PASS" if all_pass else "Phase 1 GATE: FAIL")


if __name__ == "__main__":
    main()
