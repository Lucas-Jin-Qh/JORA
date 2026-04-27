# METHOD_WORDING_LOCK

Last updated: 2026-04-26

This document freezes the method wording that is currently allowed by `docs/FORMULA_AUDIT.md`.

## Allowed equations

### Current JORA-Diag
- `y = W_0 x + Δ(x)`
- `Δ(x) = R_L^⊤ Diag(d) R_R x`

### Current JORA-NoRot
- `y = W_0 x + Δ(x)`
- `Δ(x) = Diag(d) x`

### Current JORA-Selective
- `y = W_0 x + Δ(x)`
- `Δ(x) = R_L^⊤ D_sel R_R x - P_U x`
- `D_sel = P_U + Diag(δ)_U`

## Forbidden equations

### Forbidden for current JORA-Diag
- `Δ(x) = R_L^⊤ Diag(1+d) R_R x - x`
- any wording that treats current JORA-Diag as a residualized full-support operator

### Forbidden family-level unification
- any single equation that pretends JORA-Diag and JORA-Selective currently share the same residualized operator form

## Allowed merge wording

- `JORA supports weight-space merging, but merge semantics differ by variant.`
- `SelectiveDiagCore uses a forward-equivalent dense-operator construction for supported shapes.`
- `Current non-selective cores use a more conservative legacy merge path.`

## Forbidden merge wording

- `All current JORA variants are exactly merge-equivalent to forward.`
- `Current JORA-Diag has exact merge semantics identical to SelectiveDiagCore.`
- `Mergeability is uniform across the full JORA family.`

## Allowed zero-init wording

- `For additive variants such as current JORA-Diag and JORA-NoRot, zero_init_core=True gives zero function change because the additive operator is zero.`
- `For JORA-Selective, zero function change holds in the intended zero-init state with zero theta and zero support deltas.`
- `Zero-init behavior is variant-specific and should not be over-generalized.`

## Forbidden zero-init wording

- `All JORA variants share the same identity-centered initialization mechanism.`
- `Current JORA-Diag is initialized as I + Diag(d) with residual subtraction.`
- `Zero-init has the same mathematical meaning for selective and additive variants.`
