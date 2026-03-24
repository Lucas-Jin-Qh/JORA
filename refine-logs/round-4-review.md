# Round 4 Review — GPT-5.4 xhigh

**ThreadId**: `019cfef1-9bb6-7570-ba01-7f2c18b7eb58`
**Date**: 2026-03-18
**Verdict**: REVISE
**Overall Score**: 8/10

## Scores

| Dimension | Score |
|-----------|-------|
| Problem Fidelity | 8 |
| Method Specificity | 8 |
| Contribution Quality | 7 |
| Frontier Leverage | 8 |
| Feasibility | 8 |
| Validation Focus | 6 |
| Venue Readiness | 7 |
| **Overall** | **8** |

## Anchor Check
**Preserved.** The proposal still targets the original bottleneck: improving the LoRA accuracy-vs-parameter frontier with a principled rotation-based PEFT mechanism. The square-layer restriction sharpens the scope rather than changing the problem.

## Assessment
- **Dominant contribution**: sharper. The paper is now clearly about one mechanism: adaptive allocation of a sparse bilateral rotation budget during burn-in, then frozen optimization.
- **Method size**: simpler, but not fully minimal yet. The main remaining overbuild is the full-width `D_diag[d]`: it creates a real risk that the gains come from a diagonal additive core, with slot allocation acting as a secondary modifier rather than the main driver.
- **Frontier leverage**: appropriate. This no longer feels old-school or forced. It is a clean modern PEFT proposal without gratuitous LLM-era add-ons.

## OFT / BOFT Baseline Choice
**B) Secondary appendix references.**

The main novelty claim is about adaptive slot allocation within sparse rotation-based PEFT, and `qGOFT` is the closest baseline for isolating that question. `OFT` and `BOFT` are useful orthogonal-family breadth references, but they are not the decisive comparators for the core mechanism, and under this compute/scope budget they should not be main-claim blockers.

## Remaining Action Items

### IMPORTANT
1. **Diagonal-core ambiguity**: Add a `D_diag-only` control or constrain `D_diag` to selected dimensions only. Without this, the main claim is under-isolated.
2. **Faithful qGOFT baseline**: If `qGOFT` is instantiated as fixed random slots, that is not a faithful published static orthogonal baseline unless the paper really does that. Either implement published qGOFT faithfully or rename the shared-codepath baseline to something like `fixed-slot JORA`.
3. **Parameter-count fairness**: Report parameter counts including `D_diag` and optional `e`, since the diagonal core is central to the fairness story.
4. **Simplify burn-in**: Decide whether the simplest successful variant is `single-allocation` or `repeated burn-in reseating`, and make that the main method.
5. **OFT/BOFT scope**: If they run cleanly on the same `q_proj + o_proj` scope, include them in appendix with light seed coverage only.

## Simplification Opportunities
- Add a `Diag-only` baseline or constrain `D_diag` to selected dimensions only.
- If screening shows little benefit from repeated reseating, collapse burn-in to a single allocation at the end of burn-in.
- Keep `OER` in the main paper only if the 3-seed gain is stable; otherwise move it to appendix.

## Modernization Opportunities
NONE

## Drift Warning
NONE

<details>
<summary>Full Raw Review</summary>

**Problem Anchor**

`Preserved.` The proposal still targets the original bottleneck: improving the LoRA accuracy-vs-parameter frontier with a principled rotation-based PEFT mechanism. The square-layer restriction sharpens the scope rather than changing the problem.

**Scores**

1. `Problem Fidelity`: **8/10**
2. `Method Specificity`: **8/10**
3. `Contribution Quality`: **7/10**
4. `Frontier Leverage`: **8/10**
5. `Feasibility`: **8/10**
6. `Validation Focus`: **6/10**
7. `Venue Readiness`: **7/10**

**OVERALL SCORE**: **8/10**

**Assessment**

`Dominant contribution:` sharper. The paper is now clearly about one mechanism: adaptive allocation of a sparse bilateral rotation budget during burn-in, then frozen optimization.

`Method size:` simpler, but not fully minimal yet. The main remaining overbuild is the full-width `D_diag[d]`: it creates a real risk that the gains come from a diagonal additive core, with slot allocation acting as a secondary modifier rather than the main driver.

`Frontier leverage:` appropriate. This no longer feels old-school or forced. It is a clean modern PEFT proposal without gratuitous LLM-era add-ons.

`Main remaining critiques:`
- The biggest pseudo-novelty risk is `D_diag`. With a trainable full diagonal over all `d` dimensions, JORA may effectively be “diagonal adapter + sparse rotations,” not primarily “adaptive sparse rotation allocation.” You need to isolate that.
- The qGOFT comparison is still not quite safe if “qGOFT” is instantiated as fixed random slots. That is not the same as a faithful static orthogonal baseline unless qGOFT truly uses that schedule.
- The right-side signal is now honestly framed, which is good, but it is still a weak proxy. That is acceptable if the claim stays narrow.

**OFT / BOFT Baseline Choice**

`B) Secondary appendix references.`

Briefly: the main novelty claim is about adaptive slot allocation within sparse rotation-based PEFT, and `qGOFT` is the closest baseline for isolating that question. `OFT` and `BOFT` are useful orthogonal-family breadth references, but they are not the decisive comparators for the core mechanism, and under this compute/scope budget they should not be main-claim blockers.

**Verdict**

`REVISE`

**Drift Warning**

`NONE`

**Simplification Opportunities**

- Add a `Diag-only` baseline or constrain `D_diag` to selected dimensions only. Right now the diagonal core is the main source of ambiguity.
- If screening shows little benefit from repeated reseating, collapse burn-in to a single allocation at the end of burn-in.
- Keep `OER` in the main paper only if the 3-seed gain is stable; otherwise move it to appendix.

**Modernization Opportunities**

`NONE`

**Remaining Action Items**

1. Add a `D_diag-only` control. Without it, the main claim is under-isolated.
2. Make the `qGOFT` baseline faithful to the published static schedule, or rename the shared-codepath baseline to something like `fixed-slot JORA`.
3. Report parameter counts including `D_diag` and optional `e`, since the diagonal core is now central to the fairness story.
4. Decide whether the simplest successful variant is `single-allocation` or `repeated burn-in reseating`, and make that the main method.
5. If `OFT/BOFT` run cleanly on the same `q_proj + o_proj` scope, include them in appendix with light seed coverage only.

</details>
