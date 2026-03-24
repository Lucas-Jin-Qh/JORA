# Round 5 Review — GPT-5.4 xhigh

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
| Feasibility | 6 |
| Validation Focus | 8 |
| Venue Readiness | 7 |
| **Overall** | **8** |

## Anchor Check
**Preserved.** The proposal still attacks the original bottleneck: improving the LoRA accuracy-vs-parameter frontier with a principled rotation-based PEFT mechanism. The square-layer restriction and fixed-support simplification sharpen scope rather than changing the problem.

## Assessment
- **More concrete**: yes. The operator, warmup phase, allocation point, deterministic pairing rule, support definition, merge path, and baseline naming are now implementable.
- **More focused**: yes. The paper is now about one mechanism: one-time adaptive allocation of sparse rotation support inside rotation-based PEFT.
- **More current**: yes. This is an appropriate foundation-model-era PEFT proposal. It does not feel old-school or trend-forced.
- **Dominant contribution**: now sharp.
- **Method size**: much simpler, but the simplification may have gone slightly too far.
- **Frontier leverage**: appropriate.
- **Baseline scope**: keeping OFT/BOFT appendix-only is the right choice under this scope and budget.

## Remaining Action Items

### IMPORTANT
1. **Capacity/anchor misalignment**: Once the full-width diagonal is removed, `K=4` plus `d_sel[|U|]` gives a parameter count that appears far below the stated `~half LoRA-r4` regime. The method is cleaner, but it may now be too small to realistically target the operating point in the success condition. Either revise the success target or choose a larger operating point.
2. **Initialization bottleneck**: If `d_sel` is initialized to zero, then `θ_L` and `θ_R` get zero gradient at initialization because the whole path is multiplied by `D_sel`. Training starts as a selected-support diagonal adapter, and the rotation mechanism only becomes trainable after `d_sel` moves off zero. Use a small nonzero `d_sel` init or a short staged schedule.
3. **`Diag-only-selected` mandatory**: Keep it as mandatory, not optional. If it nearly matches JORA, the claim must narrow.
4. **Faithful qGOFT**: Keep faithful `qGOFT` reproduction strict. `fixed-slot JORA` is a diagnostic, not a substitute for the published baseline.
5. **Support stability evidence**: Report support-stability evidence from warmup so the one-shot allocation looks justified rather than arbitrary.

## Simplification Opportunities
- Drop `OER` entirely from the main proposal unless screening forces it back in.
- Remove repeated-reseating discussion from the core narrative altogether; keep only the one-shot allocation story.
- If you want the rotation mechanism to matter immediately, drop the exact-identity init requirement and use a tiny nonzero `d_sel` init.

## Modernization Opportunities
NONE

## Drift Warning
NONE

<details>
<summary>Full Raw Review</summary>

**Problem Anchor**

`Preserved.` The proposal still attacks the original bottleneck: improving the LoRA accuracy-vs-parameter frontier with a principled rotation-based PEFT mechanism. The square-layer restriction and fixed-support simplification sharpen scope rather than changing the problem.

**Scores**

1. `Problem Fidelity`: **8/10**
2. `Method Specificity`: **8/10**
3. `Contribution Quality`: **7/10**
4. `Frontier Leverage`: **8/10**
5. `Feasibility`: **6/10**
6. `Validation Focus`: **8/10**
7. `Venue Readiness`: **7/10**

**OVERALL SCORE**: **8/10**

**Assessment**

`More concrete:` yes. The operator, warmup phase, allocation point, deterministic pairing rule, support definition, merge path, and baseline naming are now implementable.

`More focused:` yes. The paper is now about one mechanism: one-time adaptive allocation of sparse rotation support inside rotation-based PEFT.

`More current:` yes. This is an appropriate foundation-model-era PEFT proposal. It does not feel old-school or trend-forced.

`Dominant contribution:` now sharp.

`Method size:` much simpler, but the simplification may have gone slightly too far.

`Frontier leverage:` appropriate.

`Baseline scope:` keeping `OFT/BOFT` appendix-only is the right choice under this scope and budget.

`Main remaining critiques:`
- The biggest remaining issue is **capacity/anchor misalignment**. Once the full-width diagonal is removed, `K=4` plus `d_sel[|U|]` gives a parameter count that appears far below the stated `~half LoRA-r4` regime. The method is cleaner, but it may now be too small to realistically target the operating point in the success condition.
- There is still a **training-signal/optimization issue**: if `d_sel` is initialized to zero, then `θ_L` and `θ_R` get zero gradient at initialization because the whole path is multiplied by `D_sel`. That means training starts as a selected-support diagonal adapter, and the rotation mechanism only becomes trainable after `d_sel` moves off zero.
- There is still a **pseudo-novelty risk** tied to `D_sel`. You handled the full-width ambiguity well, but the paper still needs `Diag-only-selected` to prove the gain is not mostly from support-selected diagonal capacity.
- The one-shot warmup signal is now honest, which is good, but it is still a **coarse heuristic**. The claim should remain narrow: early-statistic allocation heuristic inside rotation-based PEFT, not a strong statement about discovering true task subspaces.

**Verdict**

`REVISE`

**Drift Warning**

`NONE`

**Simplification Opportunities**

- Drop `OER` entirely from the main proposal unless screening forces it back in.
- Remove repeated-reseating discussion from the core narrative altogether; keep only the one-shot allocation story.
- If you want the rotation mechanism to matter immediately, drop the exact-identity init requirement and use a tiny nonzero `d_sel` init.

**Modernization Opportunities**

`NONE`

**Remaining Action Items**

1. Recompute the exact total parameter counts and reconcile them with the anchor. As written, `JORA-base` no longer appears to live near the stated `~half LoRA-r4` budget. Either revise the success target or choose a larger operating point.
2. Fix the initialization bottleneck. With `d_sel=0`, rotations do not learn at step 1. Use a small nonzero `d_sel` init or a short staged schedule.
3. Keep `Diag-only-selected` as mandatory, not optional. If it nearly matches JORA, the claim must narrow.
4. Keep faithful `qGOFT` reproduction strict. `fixed-slot JORA` is a diagnostic, not a substitute for the published baseline.
5. Report support-stability evidence from warmup so the one-shot allocation looks justified rather than arbitrary.

</details>
