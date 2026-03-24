# Round 2 Review — GPT-5.4 xhigh

**ThreadId**: `019cfef1-9bb6-7570-ba01-7f2c18b7eb58`
**Date**: 2026-03-18
**Verdict**: REVISE
**Overall Score**: 7/10

## Scores

| Dimension | Score |
|-----------|-------|
| Problem Fidelity | 8 |
| Method Specificity | 6 |
| Contribution Quality | 7 |
| Frontier Leverage | 8 |
| Feasibility | 6 |
| Validation Focus | 8 |
| Venue Readiness | 7 |
| **Overall** | **7** |

## Anchor Check
**Preserved.** The revision still attacks the original bottleneck. The additive reframing fixes the earlier overclaim rather than changing the problem.

## Action Items by Priority

### CRITICAL
1. **Method Specificity — selection statistics underspecified**:
   - `ema_col[d_in]` is only input-side. K_L left slots need an output-side signal.
   - Actual update uses activation energy `x²`, but text claims "gradient energy" — mismatch.
   - `D_diag[min(K_L, K_R)]` is not shape-precise for rectangular layers.
   - Fix: define two separate statistics:
     - `ema_in[j] = EMA[(∂L/∂x_j)²]` or `EMA[x_j²]` (activation proxy)
     - `ema_out[i] = EMA[(∂L/∂δ_i)²]` (output-side gradient proxy)
   - Select top `2K_R` input dims, top `2K_L` output dims, pair deterministically.
   - Define exact ambient operator for non-square layers (rectangular diagonal core D_rect ∈ ℝ^{d_out×d_in} with active-index map).

### IMPORTANT
2. **Feasibility — rectangular module risk**: Bilateral slot alignment for non-square modules is high-risk.
   - Fix: implement shared codepath with switches (selection=fixed|EMA, freeze_after_burnin, oer=on|off). Validate on square attention projections first before expanding.

3. **OER framing still weak**: "Base output preserved at init" comes from zero DiagCore, not from OER. OER is a stabilizer, not a second novelty axis.
   - Fix: if OER stays, frame narrowly as "magnitude stabilizer preventing norm drift." Do not claim conservation as a novel property — the zero-delta init property is from DiagCore initialization, not OER per se.

4. **Pair-level novelty claim**: Current method is "adaptive dimension allocation + deterministic pairing" not "adaptive pair selection." Don't oversell.
   - Fix: reframe as "adaptive dimension allocation" throughout.

## Remaining Notes
- **Dominant contribution**: Now sharper — paper is mostly about one thing: adaptive allocation + freeze of sparse bilateral rotation budget.
- **Method size**: Simpler, close to adequately minimal. Only OER is still optional.
- **Modernization opportunity**: Replace EMA[x²] with EMA[g²] (first-order backward stat) for the "learning-signal concentration" claim to be technically aligned.

## Simplification Opportunities
- Drop OER from main method unless it gives clear, robust gain. Keep as appendix stabilizer.
- Stop claiming adaptive pair selection; reframe as adaptive dimension allocation with deterministic pairing.
- Restrict first main result to square attention projections if rectangular operator not made fully explicit.

## Modernization Opportunities
- Replace EMA[x²] with EMA[g²] if claiming task-driven allocation (optional).

## Drift Warning
NONE

<details>
<summary>Full Raw Review</summary>

**Problem Anchor**: Preserved.

Scores: Problem Fidelity 8, Method Specificity 6, Contribution Quality 7, Frontier Leverage 8, Feasibility 6, Validation Focus 8, Venue Readiness 7. Overall: 7/10.

Critical: Specify separate input-side/output-side selection statistics. Define exact pairing policy. Define D operator for rectangular layers, or narrow module scope.

Important: Validate on square projections first. OER framing — "base output preserved at init" comes from zero D, not OER. If OER stays, frame as stabilizer only.

Dominant contribution: sharper. One caveat: currently is "adaptive dimension allocation + deterministic pairing," not "adaptive pair selection."

Method size: Simpler, close to minimal. OER still optional.

Drift Warning: NONE. Verdict: REVISE.

</details>
