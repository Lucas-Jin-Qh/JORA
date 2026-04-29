# JORA Option C Postmortem (2026-04-28)

## Summary

**Option C** — residualized DiagCore operator (`Δ(x) = R_L^T (I+Diag(d)) R_R x - x`) — is **fatally broken** for the full-support DiagCore mainline. The attempt to "fix" the operator semantics has been abandoned.

**Verdict: STOP. Do not proceed with Option C for DiagCore.**

---

## What Was Attempted

The original additive DiagCore formula:
```
Δ(x) = R_L^T @ Diag(d) @ R_R @ x
```
At init (d=0): Δ(x) = 0. Zero-function-change is correct.

Option C proposed residualizing it to match the SelectiveDiagCore paper-exact formula:
```
Δ(x) = R_L^T @ (I + Diag(d)) @ R_R @ x - x
```
Intention: at init (d=0), Δ(x) = R_L^T @ R_R @ x - x. If R_L^T @ R_R ≈ I, then Δ(x) ≈ 0.

---

## Why It Failed

### Root Cause: R_L^T @ R_R ≠ I when pairs are independent

JORA uses **independent** random pair sets for left and right rotations:
- `pairs_L` and `pairs_R` are independently sampled from gradient energy
- `theta_L` and `theta_R` are independently parameterized
- Even at theta=0, the rotation matrices R_L and R_R are different structures

Therefore:
```
R_L^T @ R_R = O  (some orthogonal matrix)
O ≠ I in general
Δ(x)_init = O @ x - x ≠ 0
```

This breaks zero-function-change at init. The deviation is small at init (~0.02% relative), but during training:

1. `theta` drifts large (diagnostics show max(theta) = 1.67 at step 3074)
2. `d` also drifts large (max(d) = 10.83 at step 3074)
3. The product O @ x dominates the output because `d` modulates O @ x, not just the residual
4. As theta grows, R_L^T @ R_R becomes a complex orthogonal coupling that completely reshapes hidden-state geometry
5. The `-x` subtraction cannot compensate — the adapter becomes catastrophic rather than corrective

### Evidence

| Configuration | Eval Loss (512 samples) | vs Base | Verdict |
|---|---|---|---|
| Base model | 5.4572 | — | — |
| Additive JORA-Diag ON | 2.2387 | -3.22 | Correct learning |
| Residualized ON (1ep) | 15.7185 | +10.26 | **Catastrophic** |
| Residualized NoRot (1ep) | 19.1891 | +13.73 | **Catastrophic** |

Both residualized variants destroyed the model. The optimizer drives parameters away from their zero-init attractor, and the residualized form cannot recover.

### Why SelectiveDiagCore Works (and DiagCore Doesn't)

SelectiveDiagCore uses a **projector** P_U:
```
Δ(x) = R_L^T @ D_sel @ R_R @ x - P_U @ x
```
At theta=0:
```
Δ(x) = P_U @ x - P_U @ x = 0  ✓ (exact)
```
P_U satisfies P_U = P_U^T @ P_U, so R_L^T @ P_U @ R_R = P_U at theta=0. The projector is self-adjoint under the rotation basis, making the residualization **exact**.

DiagCore uses the **full identity matrix** I, which is NOT self-adjoint under arbitrary rotations. There is no such constraint that R_L^T @ I @ R_R = I when R_L and R_R are built from independent pairs.

---

## What Was Done

1. Implemented `is_residualized = True` flag on DiagCore (layer.py line 277)
2. Changed `apply_to_vector` to return `(I + Diag(d)) @ x` instead of `Diag(d) @ x`
3. Changed `compute_delta` to subtract `x` for residualized cores (layer.py lines 642-644)
4. Added exact basis-probing merge for DiagCore (C1.6 fix)
5. Created configs `run_diag_residualized_on_s42.json` and `run_diag_residualized_norot_s42.json`
6. Ran 1ep training for both variants
7. Evaluated: catastrophic failure confirmed

---

## What to Do Now

### Immediate

1. **Roll back DiagCore to additive default** — remove `is_residualized` flag, restore `apply_to_vector` to `Diag(d) @ x`
2. **Keep residualized DiagCore behind an explicit experimental flag** (`use_residualized=True`) — never default to it
3. **Keep the residualized formula ONLY for SelectiveDiagCore** — this is the only path where it is mathematically correct

### Code Changes Required

In `src/peft/tuners/jora/layer.py`:

```python
# Roll back: in compute_delta(), remove the residualization for DiagCore
# Line 642-644 currently:
if getattr(self.core, 'is_residualized', False) and self.n == self.m:
    return y.to(x.dtype) - x
# Change to: no automatic residualization
return y  # Always additive for DiagCore
```

In `src/peft/tuners/jora/core.py`:

```python
# Roll back: DiagCore.apply_to_vector should return Diag(d) @ x, not (I+Diag(d)) @ x
# Currently line 322-328 returns (I + diag_params) * x
# Should return: diag_params * x (additive form)
```

### Document as Negative Finding

In the paper (if applicable): "We attempted a full-support residualized orthogonal correction path (Option C) but found it unstable: when rotation bases are independently parameterized on left and right, R_L^T @ R_R ≠ I and the residualization term does not achieve zero-function-change at initialization. This instability was not recoverable through learning-rate or initialization tuning. The negative result motivates the projector-constrained approach used in the paper path (SelectiveDiagCore)."

---

## Key Lessons

1. **Mathematical constraints matter**: A formula that works with a projector (SelectiveDiagCore) does NOT work with the full identity matrix (DiagCore). The residualization requires the subtracted term to be self-adjoint under the rotation basis.

2. **Independent parameterization breaks symmetry**: When theta_L and theta_R are independently parameterized from independent pair sets, the identity property R_L^T @ R_R = I does not hold at init.

3. **Small init ≠ stable training**: A small deviation at init (0.02% relative) can become catastrophic during optimization if the parameter space allows drift toward regimes where the operator is destructive.

4. **Gate before training**: The C1.5/C2.0 gates were the right idea but insufficient. The failure was detectable with a 5-step offline eval before the full 1ep training waste.

---

## Files Involved

- `src/peft/tuners/jora/layer.py` — compute_delta residualization logic
- `src/peft/tuners/jora/core.py` — DiagCore.apply_to_vector implementation
- `configs/run_diag_residualized_on_s42.json` — residualized ON config (experimental, not to be reused)
- `configs/run_diag_residualized_norot_s42.json` — residualized NoRot config (experimental, not to be reused)
- `results/run_diag_residualized_on_s42/` — failed 1ep results
- `results/run_diag_residualized_norot_s42/` — failed 1ep results
