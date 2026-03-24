# Round 7 Review — GPT-5.4 xhigh

**ThreadId**: `019cfef1-9bb6-7570-ba01-7f2c18b7eb58`
**Date**: 2026-03-18
**Verdict**: REVISE
**Overall Score**: 8/10

## Scores

| Dimension | Score |
|-----------|-------|
| Problem Fidelity | 8 |
| Method Specificity | 7 |
| Contribution Quality | 8 |
| Frontier Leverage | 8 |
| Feasibility | 7 |
| Validation Focus | 9 |
| Venue Readiness | 7 |
| **Overall** | **8** |

## Anchor Check
**Preserved.** The proposal still attacks the original anchored problem: improving the LoRA accuracy-vs-parameter frontier with a principled rotation-based PEFT mechanism. The square-layer restriction and extreme-budget framing sharpen scope rather than changing the task.

## Assessment
- **Dominant contribution**: now sharp. Clearly one mechanism: one-shot adaptive allocation of sparse bilateral rotation support inside rotation-based PEFT.
- **Method size**: simple and no longer overbuilt.
- **Frontier leverage**: appropriate. No trendy add-ons needed.
- **More concrete/focused/current**: yes on all three. Materially better than earlier rounds.

## Remaining Action Items

### IMPORTANT
1. **Operator notation inconsistency**: `diag(1+δ)_U` is inconsistent across the proposal — in one place it behaves like `I_U` at init; in the system overview it is described differently. Fix: define once as `D_sel = P_U + diag(δ)_U` and use that notation everywhere.

2. **Warmup budget fairness underspecified**: During T_stat, backward passes collect `ema_out` but no trainable adapter exists yet. Must decide: is this a pre-training calibration pass (before optimizer updates) or does it consume training budget? Keep that treatment consistent across all baselines in comparisons.

3. **Residualized init is a correctness fix, not a headline novelty**: If the paper sells the init trick as a second contribution, it will read inflated. Keep it as a correctness property of the mechanism, not a claim.

4. **Keep empirical target honest**: `~12K within 2pp of LoRA-r2` is plausible as a stretchy primary claim, not a default assumption. Screening results must drive the final framing.

### MINOR
5. **Right-side is a proxy — do not oversell bilateral story**: The asymmetric bilateral framing is acceptable; do not dress it up as fully task-driven bilateral allocation.

## Simplification Opportunities
- Move statistics collection to a short calibration pass before optimizer updates begin. Removes mid-training switchover story; makes fairness cleaner.
- Replace `diag(1+δ)_U` with `D_sel = P_U + diag(δ)_U` everywhere.
- If bilateral allocation is not clearly stronger than a simpler variant, consider fallback to left-side adaptive only + fixed/random right-side support.

## Modernization Opportunities
NONE

## Drift Warning
NONE

<details>
<summary>Full Raw Review</summary>

**Problem Anchor**

`Preserved.` The proposal still attacks the original anchored problem: improving the LoRA accuracy-vs-parameter frontier with a principled rotation-based PEFT mechanism. The square-layer restriction and extreme-budget framing sharpen scope rather than changing the task.

**Scores**

1. `Problem Fidelity`: **8/10**
2. `Method Specificity`: **7/10**
3. `Contribution Quality`: **8/10**
4. `Frontier Leverage`: **8/10**
5. `Feasibility`: **7/10**
6. `Validation Focus`: **9/10**
7. `Venue Readiness`: **7/10**

**OVERALL SCORE**: **8/10**

**Assessment**

`Dominant contribution:` now sharp. The paper is clearly about one mechanism: one-shot adaptive allocation of sparse bilateral rotation support inside rotation-based PEFT.

`Method size:` simple and no longer overbuilt.

`Frontier leverage:` appropriate. This is modern enough already; no trendy add-ons are needed.

`More concrete / focused / current:` yes on all three. This is materially better than earlier rounds.

`Main remaining critiques:`
- The residualized operator is conceptually fixed, but the notation is still inconsistent. In one place `diag(1+δ)_U` behaves like `I_U` at init; in the system overview it is described as acting like identity outside `U`. Those are different operators. Define it once as `D_sel = P_U + diag(δ)_U` and use that notation everywhere.
- The warmup/integration point is still a little awkward. During `T_stat`, you are doing backward passes to collect `ema_out`, but no trainable adapter exists yet. That is fine operationally, but you should decide whether this is a short pre-training calibration pass or whether it consumes training budget. Right now that fairness point is underspecified.
- The right-side signal is still weak by design. That is acceptable, but it means the bilateral story is asymmetric. Do not oversell it as fully task-driven bilateral allocation.
- The residualized init is a correct implementation fix, not a headline novelty. If the paper starts selling the init trick as a second contribution, it will read inflated.
- The empirical bet is now much more honest, but still aggressive. `~12K` being within `2pp` of `LoRA-r2` is plausible as a stretchy primary claim, not as something the reader should assume will obviously happen.

**Verdict**

`REVISE`

**Drift Warning**

`NONE`

**Simplification Opportunities**

- Move the statistics collection to a short calibration pass before optimizer updates begin. That removes the mid-training switchover story and makes fairness cleaner.
- Replace the overloaded `diag(1+δ)_U` notation with `D_sel = P_U + diag(δ)_U`. It is simpler and avoids the current ambiguity.
- If bilateral allocation is not clearly stronger than a simpler variant, consider a fallback main method with left-side adaptive allocation and fixed/random right-side support.

**Modernization Opportunities**

`NONE`

**Remaining Action Items**

1. Clean up the operator definition so `D_sel`, `P_U`, and the zero-at-init property are mathematically unambiguous everywhere.
2. Specify whether warmup is a pre-training calibration phase or part of the training budget, and keep that treatment consistent across comparisons.
3. Keep the claim narrow: adaptive sparse support allocation inside rotation-based PEFT, not a broader adaptive-PEFT story.
4. Precommit the interpretation of `Diag-only-selected` exactly as written; that control now materially determines what the paper is actually about.
5. Treat `LoRA-r2 within 2pp` as the primary target and `match LoRA-r2` as stretch only. If screening misses that target, do not force the stronger claim.

</details>
