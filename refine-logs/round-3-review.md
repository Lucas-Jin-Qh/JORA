# Round 3 Review — GPT-5.4 xhigh

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
| Validation Focus | 8 |
| Venue Readiness | 7 |
| **Overall** | **8** |

## Anchor Check
**Preserved.** The revision fixes the earlier drift. This is clearly a rotation-structured additive PEFT method aimed at the original bottleneck.

## Assessment
- **Dominant contribution**: Sharp. Paper is about one mechanism: one-time adaptive allocation of sparse bilateral rotation budget during burn-in, then frozen optimization.
- **Method size**: Much simpler, close to minimal. OER is no longer distorting the paper.
- **Frontier leverage**: Appropriate — modern PEFT paper, not trend-chasing.

## Remaining Action Items

### IMPORTANT
1. **Square-layer restriction**: Method is fully specified only for square layers. Rectangular extension not closed under stated operator. Restrict main paper to square-layer-only; rectangular to appendix/future work unless valid operator is defined.
2. **Right-side training signal weakness**: EMA[x²] is a cheap activation proxy, not a task-driven learning signal. Present honestly as "cheap activation proxy," not "gradient energy." Weakens the strongest version of the claim, but honest framing is acceptable.
3. **Novelty claim precision**: Not "the first adaptive PEFT selection method broadly" — "adaptive allocation of sparse rotation slots within rotation-based PEFT." Keep claim narrow.
4. **Deterministic pairing rule fully explicit**: Must be reproducible (greedy disjoint, exact tie-breaking rule).
5. **Seed coverage for Claims 2&3**: qGOFT and random-slots ablations need enough seeds to avoid single-seed artifact accusations.
6. **OER justification**: If retained, justify why inverse base-row-norm scaling is the right stabilizer for an additive adapter, not just a convenient heuristic. (Answer: it preserves the sum-of-squared-output-norms at init — that's the justification. Present it clearly.)

## Simplification Opportunities
- Make main paper square-layer-only; rectangular to appendix.
- OER appendix-only unless stable gain across seeds.
- If repeated reseating not clearly better than one-shot allocation, collapse burn-in to single allocation phase + immediate freeze.

## Modernization Opportunities
NONE

## Drift Warning
NONE

<details>
<summary>Full Raw Review</summary>

Overall 8/10. Dominant contribution: sharp (one mechanism: adaptive allocation of sparse bilateral rotation budget during burn-in, then frozen). Method size: close to minimal. Frontier leverage: appropriate.

Main remaining critiques:
- Square-only fully specified; rectangular extension not closed.
- EMA[x²] is activation proxy not task-driven signal — present honestly.
- Novelty is "adaptive sparse rotation-slot allocation in PEFT," not broadly "first adaptive PEFT selection."

Drift Warning: NONE. Verdict: REVISE.

Simplification: square-only main results; OER to appendix unless clear gain; single allocation phase if repeated reseating not clearly better.
Modernization: NONE.

</details>
