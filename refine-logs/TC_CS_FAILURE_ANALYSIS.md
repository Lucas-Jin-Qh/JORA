# TC-CS Failure Analysis

**Date**: 2026-04-28
**Status**: GATE FAIL — mechanism collapse confirmed
**Script**: `step47_tccs_failure_analysis.py`

---

## Verdict

TC-CS pair selection collapses to consecutive pairing through a **mathematical identity**, not an engineering bug. The gate criterion is not merely failed — it is failed by construction. No amount of pool-size tuning or score-formula tweaking on the current infrastructure can recover a meaningful differentiation.

---

## Evidence

### Evidence 1: 100% Pair Overlap (All 48 Layer-Module Combinations)

| Module | Mean Overlap | Min | Max |
|--------|-------------|-----|-----|
| q_proj | 100.0% | 100.0% | 100.0% |
| k_proj | 100.0% | 100.0% | 100.0% |
| v_proj | 100.0% | 100.0% | 100.0% |
| out_proj | 100.0% | 100.0% | 100.0% |

**Total**: 384/384 pairs identical = **100.0% overlap**
**Step 4.8 training delta**: 2.7e-6 (noise level)

Sample pairs (L0, q_proj, k=8):

| # | TC-CS pair | Consecutive pair | Same? |
|---|-----------|----------------|-------|
| 0 | (8, 454) | (8, 454) | YES |
| 1 | (257, 916) | (257, 916) | YES |
| 2 | (262, 308) | (262, 308) | YES |
| 3 | (30, 987) | (30, 987) | YES |
| 4 | (185, 779) | (185, 779) | YES |
| 5 | (356, 672) | (356, 672) | YES |
| 6 | (705, 899) | (705, 899) | YES |
| 7 | (47, 697) | (47, 697) | YES |

Not a single pair differs.

### Evidence 2: grad_col_ema is Exactly Rank-1

The outer product of `grad_col_ema` with itself is, by definition, a rank-1 matrix:

```
outer(gc)[i,j] = gc[i] * gc[j]
```

This means:

| Statistic | Value |
|-----------|-------|
| Rank-1 explained variance | **100.0%** (all 24 tested combos) |
| Correlation(corr\|outer(gc)\|, \|gc[i]*gc[j]\|) | **1.0000** (all 24 combos) |
| Energy top-64 share (mean) | 44.3% |
| Energy gap top-1 / top-64 (mean) | ~1.5x |

The outer product is exactly rank-1 because it is constructed from a single vector (`grad_col_ema`). There is no second dominant direction to create off-diagonal structure.

### Evidence 3: Candidate Pool Sensitivity — All Fail

Candidate pool size does not change the outcome. Smaller pools (top-32, top-16, top-8) still produce pairs that are identical to the consecutive selection, because the selection criterion within the pool is still `gc[i] * gc[j]` — the same product that drives the consecutive ordering.

Energy-rank difference within the top-64 pool:
- Mean rank difference between adjacent pool dims: 0.000 (std: 0.023)
- The pool is energy-sorted; adjacent indices in the pool have almost identical energy

---

## Root Cause: The Mathematical Identity

The TC-CS score formula uses `g_cov_ema` as a proxy for `E[x_i * x_j]`. But `g_cov_ema` is the EMA of the second moment `E[x_i^2]` — a diagonal-only quantity. The coupling score derivation goes:

```
score[i,j] = |E[x_i * x_j]| * sqrt(E[i] * E[j])
           = |g_cov_ema[i,j]| * sqrt(g_cov_ema[i,i] * g_cov_ema[j,j])
```

Since `g_cov_ema[i,j]` in practice equals `E[x_i * x_j]` estimated from the same activation stream, and `E[x_i * x_j]` is approximated by `E[x_i^2] * E[x_j^2]` under independence, the score collapses:

```
score[i,j] ≈ |E[x_i^2] * E[x_j^2]| * sqrt(E[x_i^2] * E[x_j^2])
           = E[x_i^2]^(3/2) * E[x_j^2]^(3/2)
           ∝ energy[i]^(3/2) * energy[j]^(3/2)
```

The exponent change does not alter the ordering — the score is a monotonic function of `energy[i] * energy[j]`. Therefore, the greedy selection produces the same pairs as energy-based consecutive selection.

The normalized correlation fix (Step 4.6) improves the situation on toy/random matrices (4.2% overlap), but on real activations where the centering barely changes the off-diagonal structure, the 100% collapse recurs.

---

## Why Step 4.6 Passed on Toy Data but Failed on Real Data

Step 4.6 used random activation matrices where dimensions have independent noise — the outer product is not rank-1, so the centering genuinely changes the ordering. On real OPT-350m activations with 100 steps of calibration, the activation statistics converge to the same rank-1-like structure, and centering provides negligible differentiation.

---

## What Would Be Needed to Fix TC-CS

A viable candidate pool redesign must satisfy:

1. **Genuine cross-dimension dependency signal**: The signal must measure something other than per-dimension magnitude. Options:
   - Fisher information matrix (FIM) off-diagonal entries — captures parameter sensitivity coupling
   - Gradient covariance: `E[∂L/∂h_i * ∂L/∂h_j]` — captures loss-surface coupling
   - Actual activation cross-covariance: `E[x_i * x_j]` with x centered — requires storing the full `(d, d)` outer product EMA, not just the diagonal
   - Sign correlation: `E[sign(x_i) * sign(x_j)]` — requires signed activation data

2. **Non-rank-1 signal**: The candidate signal must not be representable as a single outer product of a magnitude vector.

3. **Candidate pool independent of energy ranking**: The pool selection must not select by energy. Options:
   - Uniform random pool (pool of 50 random dims regardless of energy)
   - Fisher-information-ranked pool (not energy-ranked)
   - Per-layer-type pools (attention vs FFN treated differently)

Even if a redesign passes the overlap gate, the training benefit must then be demonstrated in an 8-step smoke test before escalating.

---

## Gate Outcome

| Criterion | Result |
|-----------|--------|
| Pair overlap < 80% on real activations | FAIL (100.0%) |
| Any pool size achieves < 80% overlap | FAIL (all >= 99%) |
| Any alternative score achieves < 80% overlap | FAIL (all >= 99%) |
| Training loss delta > noise | FAIL (2.7e-6) |

**Gate: FAIL.** All evidence points to a structural identity, not a tunable parameter.

---

## Decision

- **Rotation revival is paused** pending a fundamentally different candidate-pool/score design.
- **Mainline continues**: JORA-Diag (additive diagonal PEFT) as primary method; JORA-NoRot as the honest baseline.
- **TC-CS remains in the record** as "attempted rotation revival; failed mechanism gate due to pair collapse from rank-1 energy structure."
- **Next paper priorities**:
  1. Complete the 3-epoch JORA-Diag vs JORA-NoRot evaluation
  2. Run LoRA and DoRA comparable-budget baselines
  3. Evaluate on downstream tasks (MMLU, ARC-C, GSM8K)
  4. Write the JORA-Diag method section with additive-diagonal framing
