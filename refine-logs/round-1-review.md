# Round 1 Review — GPT-5.4 xhigh

**ThreadId**: `019cfef1-9bb6-7570-ba01-7f2c18b7eb58`
**Date**: 2026-03-18
**Verdict**: REVISE
**Overall Score**: 6/10

## Scores

| Dimension | Score |
|-----------|-------|
| Problem Fidelity | 6 |
| Method Specificity | 5 |
| Contribution Quality | 6 |
| Frontier Leverage | 8 |
| Feasibility | 6 |
| Validation Focus | 6 |
| Venue Readiness | 6 |
| **Overall** | **6** |

## Action Items by Priority

### CRITICAL
1. **Method Specificity — EMA selection underspecified**: Where do pair parameters live when pairs change? How often do you reseat them? Are left/right scores different?
   - Fix: use **slot-based design**: each layer has fixed left/right rotation slots storing `(i, j, theta)`, reseated every T steps from EMA row/column importance, then frozen after burn-in. Write the exact OER formula.

2. **Problem Fidelity — mechanism drift**: Deployed form `W_merged = W0 + Diag(scale) R_L^T D R_R` is a structured additive adapter, not a direct sparse orthogonal adaptation of W0. Weakens "geometry-preserving" thesis.
   - Fix: either make update act directly on W0, or explicitly reframe as structured sparse adapter and stop overclaiming orthogonal preservation.

### IMPORTANT
3. **Contribution Quality — too many parallel contributions**: DiagCore + BlockCore + LowRankCore + OER + EMA all feel co-equal; paper may read as "diagonal adapter in rotated basis."
   - Fix: make adaptive sparse bilateral rotations + diagonal core the whole method. Remove BlockCore/LowRankCore from main paper. Keep OER only if it gives clear, repeatable gain.

4. **Feasibility — schedule optimistic**: Risk is iteration time on selection stability and fair qGOFT comparison, not GPU-hours.
   - Fix: implement qGOFT as strict subcase of the same codepath (fixed pairs, no OER). Screen configs with one seed first.

5. **Validation Focus — "Pareto frontier" too strong for 1 JORA point**: Need 2+ budget points.
   - Fix: add one more JORA budget point. Minimal set: LoRA-r2, LoRA-r4, qGOFT, JORA-small, JORA-base, +/- selection, +/- OER.

6. **Venue Readiness — theory story weak**: Not yet sharp enough for top venue.
   - Fix: center on adaptive sparse pair allocation. OER = stabilization, not co-headline novelty. Replace init-only conservation pitch with mergeability + fixed-budget capacity concentration claim.

## Simplification Opportunities
- Delete BlockCore and LowRankCore. Ship only DiagCore.
- Freeze pair identities after burn-in instead of fully dynamic reselection throughout training.
- Drop OER from v1 unless it gives robust gain beyond the adaptive sparse rotation mechanism.

## Modernization Opportunities
- NONE

## Drift Warning
Partial mechanism drift: method is a structured additive adapter, not direct sparse orthogonal transformation of W0. Must present this honestly.

<details>
<summary>Full Raw Review</summary>

**Scorecard**

1. `Problem Fidelity`: **6/10**
Still aimed at the anchored PEFT bottleneck, but the current formulation learns an additive structured adapter `W0 + Delta`, not clearly a structure-preserving transformation of `W0` itself.

2. `Method Specificity`: **5/10**
The forward path is fairly concrete, but the key novelty is not: dynamic pair selection, parameter persistence under reselection, left/right scoring, and the exact OER normalization are not specified tightly enough.

3. `Contribution Quality`: **6/10**
There is one good idea here, but it is buried under too many co-equal pieces: adaptive sparse rotations, OER, multiple core variants, and a weak theory hook.

4. `Frontier Leverage`: **8/10**
This is already a sensible foundation-model-era PEFT proposal. It does not need extra LLM/RL/hypernetwork machinery.

5. `Feasibility`: **6/10**
The compute budget is plausible. The real risk is engineering/debugging time, especially for discrete reselection plus a fair qGOFT baseline.

6. `Validation Focus`: **6/10**
Mostly disciplined, but one operating point is not enough to justify "Pareto frontier" language, and a few proposed ablations are lower value than an extra budget point.

7. `Venue Readiness`: **6/10**
Promising, but not yet sharp. In current form it risks reading as "sparse qGOFT + extra scaler" rather than one inevitable mechanism paper.

**OVERALL SCORE**: **6/10**

**Drift Warning**: There is partial mechanism drift. The learned object is a structured additive adapter, not a direct sparse orthogonal transformation of the pretrained weights.

**Verdict**: `REVISE`

The core idea worth saving is adaptive sparse rotation allocation. The proposal becomes much stronger if you strip it down to that idea, pin the implementation interface precisely, and stop asking OER and core variants to carry equal novelty weight.

</details>
