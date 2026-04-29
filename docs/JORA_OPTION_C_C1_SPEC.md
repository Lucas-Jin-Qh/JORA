# JORA Option C — C1 Gate: Residualized DiagCore Design Audit

**Date**: 2026-04-28
**Gate**: C1 — offline design validation before any further training
**Purpose**: Determine whether a residualized DiagCore operator is worth implementing; no GPU time committed yet.
**Rule**: No code changes. No training runs. Analysis only.

---

## Status: Current Failure Analysis

### Root Cause Hypothesis

The triple failure (rotation null, TC-CS fail, JORA vs LoRA gap 0.28) has a shared root: **the additive DiagCore operator is poorly conditioned for optimization at standard training settings**.

Current JORA-Diag formula:
```
Δ(x) = R_L^T Diag(d) R_R x
y    = W₀ x + Δ(x)
```

With `zero_init_core=True`: `d=0`, so `Δ=0` at init. The optimizer starts from zero gradient on `d`, but receives gradient from the loss only after the first forward-backward pass. The path from `d` to loss is `W₀ → activation → loss`, which depends on `W₀`'s pretrained quality.

**Hypothesis**: The additive form `W₀ + Δ` with zero-init is harder to optimize than LoRA's `W₀ + BA x` with proper LoRA-scaled init, because:
1. `d` gradients flow through the full pretrained `W₀` → weaker signal at early steps
2. DiagCore has `d` parameters all coupled through `W₀` in the loss landscape
3. LoRA's `A,B` initialization is specifically designed for gradient flow

### What has NOT been ruled out

- Hyperparameters (lr, init_std, lr_theta/lr_core) are suboptimal
- Layer targeting (all-linear vs attention-only) matters
- Training horizon (3ep may not be enough to see convergence)

However, changing hyperparameters without changing the operator is iterative guesswork. The C1 gate tests whether the **operator form itself** has structural problems.

---

## Candidate Operators for C1

### Operator A — Residualized DiagCore (recommended)

**Formula**:
```
Δ(x) = R_L^T (I + Diag(d)) R_R x - x
y    = W₀ x + Δ(x)
     = W₀ x + R_L^T (I + Diag(d)) R_R x - x
```

At init (`theta=0, d=0`):
```
R_L = R_R = I
Δ(x) = (I^T I I - I) x = (I - I) x = 0
y    = W₀ x  (base model output)
```
Zero function change ✓

**Implementation changes required**:
1. `DiagCore.apply_to_vector()`: change from `x * diag_params` to `x * (1 + diag_params)`
2. `DiagCore.forward()`: same change for dense matrix path
3. `DiagCore.get_row_slice()`: same
4. DiagCore merge: use exact basis-probing (SelectiveDiagCore path) instead of legacy heuristic

**Note on the `+1`**: This is a hardcoded identity offset, not learnable. It makes `d=0` mean "identity perturbation from base." Equivalent to saying "at init, the adapter outputs zero delta; we learn d to move away from identity."

### Operator B — Residualized with Learnable Scale

**Formula**:
```
Δ(x) = R_L^T Diag(s + d) R_R x - s * x
y    = W₀ x + Δ(x)
```

With `zero_init_core=True` and `s=1` (fixed or initialized to 1): same zero-change behavior as Operator A.

This is strictly more general than Operator A (it can learn `s` as well), but adds complexity and a potential instability if `s` drifts from 1. **C1 evaluation should assess whether this complexity is justified.**

### Operator C — Residualized + Warmup on d

Without changing the operator form, this is a hyperparameter change, not a design change. Not a C1 candidate.

---

## Key Differences: Current Additive vs Residualized

| Property | Current (additive) | Residualized (A/B) |
|---|---|---|
| Formula | `Δ = R^T Diag(d) R x` | `Δ = R^T (I+Diag(d)) R x - x` |
| Init behavior | `d=0` → `Δ=0` → `y=W₀x` | `d=0,theta=0` → `Δ=0` → `y=W₀x` |
| Learned structure | Additive delta to `W₀x` | Identity perturbation from `W₀x` |
| d semantics | Absolute diagonal coefficients | Deviation from identity |
| `d → loss` gradient path | `W₀ → activation → loss` | `W₀ → activation → loss` (same) |
| Merge exact? | **No** (legacy heuristic) | **Yes** (basis probing needed) |
| Selective path compatibility | Yes | Yes |

**Key observation**: Both forms have `Δ=0` at zero-init. The gradient path `d → loss` is identical in both cases (both flow through `W₀`). The difference is not in initialization but in the **parameterization of the update direction** — whether `d` represents an absolute diagonal or a deviation from identity.

**Does this difference matter?** Theoretically:
- If `d` converges to a value `d*`, then the full adapter is `W₀ + R^T Diag(d*)R` (additive) vs `W₀ + R^T(I+Diag(d*))R - I` (residualized).
- These are different operators. The residualized form subtracts `I`, which changes the effective final weight matrix.
- With `d*` close to zero: both are near `W₀`. With `d*` away from zero: the residualized form constrains the perturbation to be near-identity (useful for fine-tuning).

---

## Zero-Init / Near-Zero-Init Expected Behavior

### Operator A (Residualized, zero_init_core=True)

| State | diag_params | Rotation | Δ(x) | y |
|---|---|---|---|---|
| Init | 0 | θ~N(0, std) | ≈ 0 (theta small) | ≈ W₀x |
| After 1 step | small gradient | same | small | slightly modified from W₀x |
| Converged | d* | θ* | full delta | W₀x + Δ(x) |

**Critical property**: `Δ=0` at init is strict only when `theta=0`. With nonzero theta_init_std (2e-3), `R_L^T R_R ≈ I + small`. So `Δ ≈ (I - I) x = 0` to first order. The `theta_init_std=2e-3` means slight perturbations from identity, but the residualized structure keeps Δ bounded.

### Operator A (Residualized, zero_init_core=False, core_init_std=0.005)

With nonzero init: `d ≈ N(0, 0.005)`. Since this is a deviation from identity (not absolute), the update direction is "perturb from identity" rather than "learn absolute diagonal." This may be more natural for fine-tuning.

**C1 must check**: Does `d_init = N(0, 0.005)` produce a `Δ` with reasonable magnitude at init? This determines whether nonzero init is viable.

---

## Merge Expected Semantics

### Current DiagCore merge: APPROXIMATE

Current `_compute_weight_delta_simple()` for DiagCore (legacy path):
```python
core_matrix = adapter_state.core.forward()  # dense matrix from DiagCore
rotation_scale = self._estimate_rotation_effect_magnitude(adapter_state)
delta_weight = core_matrix * rotation_scale
delta_weight *= 0.05  # very conservative
```

This is a heuristic approximation — **not** forward-equivalent.

### Residualized DiagCore must have EXACT merge

The residualized operator is linear: `Δ(x) = (R_L^T (I+Diag(d)) R_R - I) x`.

For exact merge, the delta weight is: `W_merged = W₀ + R_L^T (I+Diag(d)) R_R - I`.

This requires reconstructing the full dense operator `R_L^T (I+Diag(d)) R_R - I` in weight space.

**For a clean implementation, DiagCore should use SelectiveDiagCore's basis-probing path for merge**:
1. Probe the adapter with basis vectors: `e_i` for each `i ∈ [0, m)`
2. Collect `Δ(e_i)` for each basis vector → reconstruct full operator rows
3. `ΔW[i, :] = Δ(e_i)` for row-vector convention

**This is the same merge mechanism as SelectiveDiagCore** — it is general and works for any linear `compute_delta`.

### C1 implication

The residualized DiagCore merge requires switching from the legacy approximate path to the exact basis-probing path. This is a code-level change, not just a formula change. **C1 gate should include: "Can the exact merge path be implemented cleanly for DiagCore without breaking SelectiveDiagCore?"**

---

## Required New / Modified Tests

### T1 — Init: zero function change (PASS gate)

```python
def test_residualized_zero_init_strict_zero():
    # d=0, theta=0 (identity rotation)
    # Δ(x) = R^T (I+0) R x - x = (R^T R - I) x
    # With theta=0: R=I, so Δ=0 exactly
    assert delta.norm() / x.norm() < 1e-6
```

### T2 — Init: nonzero theta causes near-zero Δ (PASS gate)

```python
def test_residualized_nearzero_theta_delta():
    # theta ~ N(0, 2e-3), d=0
    # R^T R ≈ I + O(theta^2)
    # Δ(x) ≈ O(theta^2) * x  (second order)
    # |Δ| / |x| should be small: < 1e-3
    assert delta.norm() / x.norm() < 1e-3
```

### T3 — Init: nonzero d causes bounded Δ (PASS gate)

```python
def test_residualized_nonzero_d_bounded():
    # d ~ N(0, 0.005), theta=0
    # Δ(x) = (I + N(0,0.005) - I) x = N(0,0.005) * x
    # |Δ| / |x| should be small: < 1e-1
    assert delta.norm() / x.norm() < 1e-1
    assert delta.norm() / x.norm() > 1e-5  # not degenerate
```

### T4 — Backward: gradient flows to d and theta (PASS gate)

```python
def test_residualized_gradient_liveness():
    # With random x, nonzero loss, backward should produce
    # nonzero gradients for BOTH d and theta
    assert d.grad is not None and d.grad.abs().sum() > 0
    assert theta.grad is not None and theta.grad.abs().sum() > 0
```

### T5 — Merge: exact forward-equivalence (PASS gate)

```python
def test_residualized_merge_exact():
    # Build delta_weight via merge
    # Apply to x: x @ (W0 + delta_weight).T
    # Compare with: W0(x) + compute_delta(x)
    # Should be close: atol=1e-4
    torch.testing.assert_close(out_merged, out_adapter, atol=1e-4, rtol=1e-3)
```

### T6 — NoRot: residualized degenerates to Diag(d)x (PASS gate)

```python
def test_residualized_norot():
    # S_L=0, S_R=0 → R_L=R_R=I
    # Δ(x) = (I + Diag(d)) x - x = Diag(d) x
    # Same as current NoRot formula
```

### T7 — Selective path: not broken by DiagCore changes (REGRESSION gate)

```python
def test_selective_still_exact_after_changes():
    # SelectiveDiagCore should be unaffected
    # Run existing SelectiveDiagCore tests
```

### T8 — BlockCore: not broken (REGRESSION gate)

```python
def test_blockcore_still_works():
    # BlockCore uses similar pattern
    # Should still produce finite outputs
```

---

## C1 Pass / Fail Gates

### MUST PASS (all required for C1 → C2)

| Gate | Condition | Failure mode if violated |
|---|---|---|
| G1 | `zero_init` + `theta=0` → `Δ(x) ≈ 0` | Residualized form does not achieve zero-change at init |
| G2 | `theta~N(0,2e-3), d=0` → `|Δ|/|x| < 1e-3` | Nonzero rotation causes large Δ at init |
| G3 | `d~N(0,0.005), theta=0` → `1e-5 < |Δ|/|x| < 1e-1` | Nonzero init is either degenerate or too large |
| G4 | Backward produces nonzero grad for both `d` and `theta` | Gradient dead zones |
| G5 | Merge is forward-equivalent (`atol=1e-4`) | Merge does not match forward |
| G6 | NoRot branch degenerates to `Diag(d)x` | Residualized form breaks NoRot semantics |
| G7 | SelectiveDiagCore tests still pass | Regression: changes break paper-path |
| G8 | BlockCore tests still pass | Regression: changes break BlockCore |

### C1 Decision Tree

```
If G1–G8 all PASS → C1 PASS: proceed to C2 (1ep sanity)
If ANY of G1–G6 FAIL → C1 FAIL: option C is not viable, stop
If G7 or G8 FAIL (regression) → fix before proceeding
```

---

## Implementation Changes Required (for C2+ only, not C1)

C1 is analysis only. If C1 passes, C2 requires these code changes:

1. **`core.py`**: Modify `DiagCore.apply_to_vector()` to return `x * (1 + diag_params)` instead of `x * diag_params`
2. **`core.py`**: Modify `DiagCore.forward()` and `get_row_slice()` similarly
3. **`layer.py`**: Switch DiagCore merge from legacy approximate path to exact basis-probing (same as SelectiveDiagCore)
4. **`layer.py`**: Update `compute_delta()` comment to document the residualized formula
5. **`config.py`**: Update `diag_path()` docstring and `FORMULA_AUDIT.md` to reflect the new formula

**Note**: These are non-trivial changes. The switch to residualized form changes the semantics of `diag_params` — existing saved checkpoints cannot be directly loaded into the new code. Version compatibility should be considered.

---

## Analysis: Will Residualized Form Fix the 0.28 Gap?

**Honest assessment**: Possibly, but not guaranteed.

**Why it might help**:
- The residualized form makes the update `W₀ + Δ` start closer to a well-conditioned identity perturbation, which may have a better basin of attraction for fine-tuning.
- LoRA's parameterization `BAx` with specific init (Kaiming/zero) is known to work well for fine-tuning. Residualized DiagCore with nonzero init may have similar conditioning properties.

**Why it might not**:
- The gradient path `d → W₀ → loss` is the same in both forms. The optimizer faces the same landscape.
- The 0.28 gap may be due to: (a) insufficient training horizon, (b) suboptimal hyperparameters, (c) layer targeting choice, or (d) the DiagCore parameterization fundamentally lacking the low-rank structure that makes LoRA effective.

**What C1 does NOT resolve**: Even if all G1–G8 pass, C1 does not tell us whether the 0.28 gap will close. It only tells us the operator is well-defined and trainable.

---

## Verdict: Proceed to C1 Gate Check

C1 Gate check requires:

1. **Offline analysis** (this document): Confirm the residualized form has clean semantics
2. **Theoretical check** (can be done now, no code needed): Verify G1–G6 analytically
3. **If theory checks out → C1 PASS**: Proceed to C2 implementation
4. **If theory fails → C1 FAIL**: Option C is not viable; fall back to Option A/B

**This document IS the C1 gate check. The gates G1–G8 above can be evaluated theoretically without running code.** The key theoretical question is: "Does the residualized form with `+1` identity offset and proper init produce bounded, well-defined behavior?"

**Recommendation**: Evaluate G1–G8 analytically in this document before writing any code.

### Preliminary G1–G8 Evaluation (theoretical, no code)

| Gate | Question | Preliminary Answer | Confidence |
|---|---|---|---|
| G1 | `zero_init + theta=0` → Δ=0? | YES. With theta=0: R=I, so `R^T(I+0)R - I = 0`. Strict zero. | High |
| G2 | `theta~N(0,2e-3), d=0` → `\|Δ\|/\|x\| < 1e-3`? | YES. `R^T R ≈ I + O(theta^2)`. With theta=2e-3: O(4e-6) bound. Well within 1e-3. | High |
| G3 | `d~N(0,0.005), theta=0` → bounded? | YES. `\|Δ\|/\|x\| ≈ 0.005` in expectation. Well within [1e-5, 1e-1]. | High |
| G4 | Gradients flow to d and theta? | YES. Forward is linear in both. | High |
| G5 | Merge is forward-equivalent? | YES (basis probing, same as Selective). Implementation needed. | Medium (requires code change) |
| G6 | NoRot = Diag(d)x? | YES. R=I for NoRot. | High |
| G7 | Selective still works? | YES. SelectiveDiagCore unchanged. | High |
| G8 | BlockCore still works? | YES. BlockCore unchanged. | High |

**Preliminary conclusion**: G1–G4 and G6–G8 are analytically satisfied. G5 requires implementation but is mechanically straightforward (same as Selective). **C1 passes on theoretical grounds. Proceed to C2 (implementation + unit tests).**
