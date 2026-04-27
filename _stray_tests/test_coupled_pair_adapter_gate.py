#!/usr/bin/env python
"""Adapter-level gate tests for CoupledPairCore in JoraLayer.

These tests verify that JoraLayer + CoupledPairCore has correct adapter-level
semantics. Unlike the core-level tests (test_coupled_pair_core_gate.py),
these test the COMPLETE adapter path through compute_delta() and forward().

IMPORTANT: These gates test the FULL adapter, not just the core in isolation.
The key distinction from SelectiveDiagCore:

  SelectiveDiagCore (paper-exact): delta = R_L^T D_sel R_R x - P_U x
    → At theta=0, delta=0: zero-function-change at init ✓

  CoupledPairCore (legacy path): delta = R_L^T @ core(R_R @ x)  (no subtraction)
    → At theta=0, core=identity: delta = P_U @ x ≠ 0 at init
    → This is a KNOWN limitation of the legacy path.

These tests DOCUMENT this gap rather than hide it.
"""

import sys
import torch
import torch.nn as nn

sys.path.insert(0, '/home/jqh/Workshop/JORA/src')

from peft.tuners.jora.config import JoraConfig
from peft.tuners.jora.layer import JoraLayer


def _make_jora_layer_coupled_pair(n: int = 16, zero_theta: bool = True) -> JoraLayer:
    """Helper: create a JoraLayer with CoupledPairCore.

    Args:
        n: layer dimension
        zero_theta: if True, explicitly zero theta params after creation to test
                    zero-function-change. If False, theta uses config defaults.
    """
    base = nn.Linear(n, n)
    cfg = JoraConfig(
        target_modules=['q_proj'],
        core='coupled_pair',
        k=8,          # 8 pairs = 16 support indices
        S_L=n,        # same as n so rotation covers all dimensions
        S_R=n,
        warmup_steps=0,   # disable warmup gating
        selection='none',  # disable dynamic selection (use init pairs)
        theta_init_std=0.0,  # zero-init for CoupledPairCore (critical for test)
    )
    layer = JoraLayer(base, adapter_name='default', cfg=cfg)
    # Explicitly zero theta to test the zero-theta case.
    # CoupledPairCore STILL fails zero-function-change because P_U subtraction is missing.
    if zero_theta:
        st = layer.adapters['default']
        if st.theta_L is not None:
            st.theta_L.data.zero_()
        if st.theta_R is not None:
            st.theta_R.data.zero_()
    return layer


def _freeze_support_manual(layer: JoraLayer, pairs: torch.Tensor):
    """Manually set support on CoupledPairCore (bypassing warmup)."""
    st = layer.adapters['default']
    # Set pairs buffer directly
    st.pairs_L[:pairs.shape[0]] = pairs
    st.num_pairs_L.fill_(pairs.shape[0])
    st.pairs_R[:pairs.shape[0]] = pairs
    st.num_pairs_R.fill_(pairs.shape[0])
    # Freeze support
    st._freeze_support_if_needed()


# =======================================================================
# CP-1: Zero-function-change at init (CRITICAL — this gate may FAIL)
# =======================================================================
def test_cp1_zero_function_change_at_init():
    """CP-1: Full JoraLayer + CoupledPairCore forward must equal base forward at init.

    This is the MOST IMPORTANT gate for adapter correctness.

    For SelectiveDiagCore: this PASSES because delta = P_U x - P_U x = 0.
    For CoupledPairCore: this FAILS because delta = P_U x (no subtraction).
      The CoupledPairCore legacy path does NOT implement residualization.

    This test documents this known limitation. If you need zero-function-change,
    use SelectiveDiagCore instead.
    """
    print("=" * 70)
    print("CP-1: Zero-function-change at init")
    print("=" * 70)

    torch.manual_seed(42)
    n = 16
    layer = _make_jora_layer_coupled_pair(n=n)

    # Freeze support manually with 4 pairs
    pairs = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]], dtype=torch.long)
    _freeze_support_manual(layer, pairs)

    # Theta is now explicitly zeroed by _make_jora_layer_coupled_pair
    # Verify pair_blocks is zero-init
    st = layer.adapters['default']
    print(f"  theta_L abs max: {st.theta_L.abs().max().item():.2e}")
    print(f"  theta_R abs max: {st.theta_R.abs().max().item():.2e}")

    # Verify pair_blocks is zero-init
    print(f"  pair_blocks abs max: {st.core.pair_blocks.abs().max().item():.2e}")
    assert st.core.pair_blocks.abs().max().item() < 1e-8, "pair_blocks should be zero-init"

    # Compute delta
    x = torch.randn(2, n)
    layer.eval()
    with torch.no_grad():
        base_out = layer.base_layer(x)
        delta = st.compute_delta(x)  # compute_delta is on adapter state, not layer
        coupled_out = base_out + delta

    print(f"\n  Input x abs max: {x.abs().max().item():.6f}")
    print(f"  Base output abs max: {base_out.abs().max().item():.6f}")
    print(f"  Delta abs max: {delta.abs().max().item():.6f}")
    print(f"  Coupled output abs max: {coupled_out.abs().max().item():.6f}")

    # Check: is delta close to zero?
    delta_near_zero = delta.abs().max().item() < 1e-4
    forward_equals_base = torch.allclose(coupled_out, base_out, atol=1e-4)

    print(f"\n  Delta near zero: {delta_near_zero}")
    print(f"  Forward == Base: {forward_equals_base}")

    if forward_equals_base:
        print("\n  ✅ PASS: CoupledPairCore has zero-function-change at init")
        return True
    else:
        print("\n  ❌ FAIL: CoupledPairCore does NOT have zero-function-change at init")
        print("  This is a KNOWN limitation of the legacy adapter path.")
        print("  Expected: delta should be ~0")
        print(f"  Actual: delta abs max = {delta.abs().max().item():.6f}")
        print("")
        print("  Root cause: compute_delta for CoupledPairCore uses:")
        print("    delta = R_L^T @ core(R_R @ x)")
        print("  But SelectiveDiagCore (paper-exact) uses:")
        print("    delta = R_L^T @ D_sel @ R_R @ x - P_U @ x")
        print("  The subtraction is missing in CoupledPairCore.")
        print("")
        print("  CONCLUSION: CoupledPairCore cannot be used as main paper-path.")
        print("  Downgrade to 'exploratory variant' and use SelectiveDiagCore for paper.")
        return False


# =======================================================================
# CP-2: Non-zero merge equals forward (approx merge gate)
# =======================================================================
def test_cp2_nonzero_merge_equals_forward():
    """CP-2: Non-zero params — forward must equal merged-forward (approx gate).

    For SelectiveDiagCore: this PASSES (exact merge).
    For CoupledPairCore: this may PASS or FAIL depending on implementation.
      We test it to document the behavior.
    """
    print("\n" + "=" * 70)
    print("CP-2: Non-zero merge equals forward")
    print("=" * 70)

    torch.manual_seed(42)
    n = 16
    layer = _make_jora_layer_coupled_pair(n=n)

    # Freeze support with 4 pairs
    pairs = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]], dtype=torch.long)
    _freeze_support_manual(layer, pairs)

    st = layer.adapters['default']

    # Set non-zero pair_blocks
    torch.manual_seed(123)
    with torch.no_grad():
        st.core.pair_blocks[:] = 0.1 * torch.randn_like(st.core.pair_blocks)
    # Set non-zero theta
    if st.theta_L is not None:
        st.theta_L.data[:] = 0.05 * torch.randn_like(st.theta_L)
    if st.theta_R is not None:
        st.theta_R.data[:] = 0.05 * torch.randn_like(st.theta_R)

    print(f"  pair_blocks abs max: {st.core.pair_blocks.abs().max().item():.6f}")
    if st.theta_L is not None:
        print(f"  theta_L abs max: {st.theta_L.abs().max().item():.6f}")

    # Test: forward vs merged forward
    x = torch.randn(2, n)
    layer.eval()
    with torch.no_grad():
        # Normal forward
        out_normal = layer(x)

        # Merged forward (fold adapter into base)
        # NOTE: CoupledPairCore may not support merge() if it lacks core.forward()
        try:
            layer.merge()
            out_merged = layer(x)
            layer.unmerge()
            diff = (out_normal - out_merged).abs().max().item()
            close = diff < 1e-3
            merge_implemented = True
        except NotImplementedError as e:
            print(f"\n  merge() raises NotImplementedError: {e}")
            print("  CoupledPairCore does not implement core.forward() — merge not supported.")
            print("  This is expected for legacy-path cores.")
            close = False
            merge_implemented = False

    if merge_implemented:
        print(f"\n  Max difference between forward and merged: {diff:.6f}")
        print(f"  Close (atol=1e-3): {close}")

    if close:
        print("\n  ✅ PASS: CoupledPairCore merge is approximately correct")
        print("  Status: approx merge (not exact like SelectiveDiagCore)")
        return True
    else:
        print("\n  ❌ FAIL: CoupledPairCore merge does NOT match forward")
        print("  CoupledPairCore cannot support exact merge/deploy semantics.")
        print("  GATE DECISION: CouplingPairCore is BLOCKED from accuracy benchmarks.")
        print("  It may only be used as an operator probe / exploratory variant.")
        return False


# =======================================================================
# CP-3: Docstring documents legacy path
# =======================================================================
def test_cp3_docstring_documents_legacy_path():
    """CP-3: Verify CoupledPairCore docstring explicitly states it's legacy."""
    print("\n" + "=" * 70)
    print("CP-3: CoupledPairCore docstring documents legacy path")
    print("=" * 70)

    from peft.tuners.jora.core import CoupledPairCore

    docstring = CoupledPairCore.__doc__ or ""
    print(f"\n  Docstring excerpt:")
    lines = docstring.strip().split('\n')[:10]
    for line in lines:
        print(f"    {line}")

    # Check for warning keywords
    has_warning = any(keyword in docstring.lower()
                      for keyword in ['warning', 'legacy', 'not paper-path', 'exploratory'])
    has_legacy = 'legacy' in docstring.lower()
    has_exploratory = 'exploratory' in docstring.lower()

    print(f"\n  Contains 'legacy': {has_legacy}")
    print(f"  Contains 'exploratory': {has_exploratory}")
    print(f"  Contains warning keywords: {has_warning}")

    if has_legacy or has_exploratory:
        print("\n  ✅ PASS: Docstring acknowledges legacy/exploratory status")
        return True
    else:
        print("\n  ❌ FAIL: Docstring does NOT acknowledge legacy/exploratory status")
        print("  Add a WARNING to CoupledPairCore docstring stating it's NOT paper-path.")
        return False


# =======================================================================
# CP-4: Comparison group notes document correct comparison口径
# =======================================================================
def test_cp4_comparison_notes_are_correct():
    """CP-4: Verify build_core documentation specifies correct comparison groups.

    Correct groups:
      A: selective_diag @ k=16 → 32 params, paper-path, exact merge ← main claim
      B: coupled_pair  @ k=16 → 64 params, legacy-path, approx merge
      C: coupled_pair  @ k=8  → 32 params, legacy-path, approx merge

    A vs B is UNFAIR (different paths, different params)
    A vs C is FAIR (matched params, different structure)
    """
    print("\n" + "=" * 70)
    print("CP-4: Comparison group notes are correct")
    print("=" * 70)

    from peft.tuners.jora.core import build_core
    from peft.tuners.jora.config import JoraConfig

    # Check coupled_pair docstring in build_core
    import inspect
    source = inspect.getsource(build_core)

    # Check for "UNFAIR" or equivalent note about selective_diag vs coupled_pair
    has_unfair_note = (
        'not fair' in source.lower() or
        'unfair' in source.lower() or
        'different paths' in source.lower() or
        'NOT paper-path' in source or
        'not paper-path' in source.lower()
    )

    print(f"\n  build_core contains fairness notes: {has_unfair_note}")

    # Verify param counts
    cfg_sd = JoraConfig(target_modules=['q_proj'], core='selective_diag', k=16)
    cfg_cp16 = JoraConfig(target_modules=['q_proj'], core='coupled_pair', k=16)
    cfg_cp8 = JoraConfig(target_modules=['q_proj'], core='coupled_pair', k=8)

    n = 32
    core_sd = build_core('selective_diag', n, n, 'cpu', torch.float32, cfg_sd)
    core_cp16 = build_core('coupled_pair', n, n, 'cpu', torch.float32, cfg_cp16)
    core_cp8 = build_core('coupled_pair', n, n, 'cpu', torch.float32, cfg_cp8)

    print(f"\n  Param counts:")
    print(f"    selective_diag @ k=16: {core_sd.num_params} params")
    print(f"    coupled_pair  @ k=16: {core_cp16.num_params} params (2x selective_diag)")
    print(f"    coupled_pair  @ k=8:  {core_cp8.num_params} params (same as selective_diag @ k=16)")

    # A vs C is fair: same params
    assert core_sd.num_params == core_cp8.num_params, \
        f"sd@16 ({core_sd.num_params}) vs cp@8 ({core_cp8.num_params}) should be equal"

    # A vs B is unfair: different params AND different paths
    assert core_sd.num_params * 2 == core_cp16.num_params, \
        f"cp@16 ({core_cp16.num_params}) should be 2x sd@16 ({core_sd.num_params})"

    print(f"\n  Param count relationships are correct:")
    print(f"    selective_diag @ k=16 == coupled_pair @ k=8: {core_sd.num_params == core_cp8.num_params} ✓")
    print(f"    coupled_pair  @ k=16 == 2 * selective_diag @ k=16: {core_cp16.num_params == 2 * core_sd.num_params} ✓")

    if has_unfair_note:
        print("\n  ✅ PASS: Comparison notes are correct and documented")
        return True
    else:
        print("\n  ⚠️  PARTIAL: Param counts correct but build_core lacks comparison notes")
        print("  Add notes to build_core() explaining A vs B is UNFAIR, A vs C is FAIR.")
        return False


if __name__ == "__main__":
    print("Running CoupledPairCore Adapter-Level Gates...\n")

    results = {}
    results['CP-1'] = test_cp1_zero_function_change_at_init()
    results['CP-2'] = test_cp2_nonzero_merge_equals_forward()
    results['CP-3'] = test_cp3_docstring_documents_legacy_path()
    results['CP-4'] = test_cp4_comparison_notes_are_correct()

    print("\n" + "=" * 70)
    print("Summary: CoupledPairCore Adapter-Level Gates")
    print("=" * 70)
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name}: {status}")

    passed_count = sum(1 for v in results.values() if v)
    print(f"\n  Total: {passed_count}/{len(results)} passed")

    if results['CP-1']:
        print("\n  ✅ CoupledPairCore is adapter-level correct.")
        print("  It may be used as a paper-path variant.")
    else:
        print("\n  ❌ CoupledPairCore FAILS adapter-level zero-function-change gate.")
        print("  RECOMMENDATION: CoupledPairCore is NOT paper-path ready.")
        print("  - Do NOT use in main paper benchmarks")
        print("  - May be used as: exploratory operator probe only")
        print("  - Use SelectiveDiagCore as the main paper claim")
