# R007 ON/OFF 3EP Verdict

Last updated: 2026-04-26

This document records the matched 3-epoch ON/OFF comparison for the current additive JORA-Diag mechanism question.

## Scope
- `R006`: JORA-Diag ON, seed 42, 3 epochs
- `R007`: JORA-NoRot, seed 42, 3 epochs

These runs are interpreted under the current method contract in `docs/JORA_RESEARCH_CONTRACT.md`, which treats:
- current JORA-Diag as additive `Δ(x)=R_L^T Diag(d) R_R x`
- JORA-NoRot as additive `Δ(x)=Diag(d)x`

## Extracted final metrics

| Run ID | Variant | Rotation | Epochs | Total steps | Train loss | Mean token accuracy | Runtime (s) | Runtime (min) | Steps/sec | Sec/step |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R006 | JORA-Diag | ON | 3 | 9222 | 2.23681562 | 0.51470943 | 9473.4065 | 157.89 | 0.973 | 1.027 |
| R007 | JORA-NoRot | OFF | 3 | 9222 | 2.23736668 | 0.51497641 | 3314.9408 | 55.25 | 2.782 | 0.359 |

## Matched ON - NoRot deltas

Computed as `R006 - R007`.

| Metric | ON - NoRot |
|---|---:|
| Train loss | -0.00055106 |
| Mean token accuracy | -0.00026698 |
| Runtime (s) | +6158.4657 |
| Runtime (min) | +102.64 |
| Steps/sec | -1.809 |
| Sec/step | +0.668 |
| Total steps | 0 |

## Interpretation

### Loss
- Rotation ON has a slightly lower final train loss than NoRot:
  - `2.23681562` vs `2.23736668`
  - delta `-0.00055106`
- This difference is extremely small.

### Token accuracy
- Rotation ON has slightly lower mean token accuracy than NoRot:
  - `0.51470943` vs `0.51497641`
  - delta `-0.00026698`
  - roughly `-0.0267` percentage points
- This difference is also extremely small.

### Runtime
- Rotation ON is dramatically slower:
  - `9473.41s` vs `3314.94s`
  - about `2.86x` slower by total runtime
  - step time `1.027s` vs `0.359s`

## Verdict under the research contract

The current contract says rotation may remain in the main story only if matched evidence shows it adds value beyond diagonal scaling in a way that justifies its cost.

Based on this matched 3-epoch comparison:
- quality differences are tiny and mixed
  - ON is trivially better in train loss
  - ON is trivially worse in mean token accuracy
- runtime cost is large and clearly unfavorable to rotation

### Decision
**Rotation should be demoted from the main narrative.**

It should not currently be removed from the repository or from appendix/mechanism discussion, but it should no longer be treated as a validated primary contributor for the main paper story.

## Recommended narrative update

Safe current wording:
- `JORA-Diag` remains the implemented main method in code.
- The diagonal core is the main effective adaptation component.
- Sparse rotations are an optional basis reparameterization whose independent benefit is not supported by current matched 3-epoch evidence.
- `JORA-NoRot` should be treated as the claim-determining mechanism baseline.

Unsafe current wording:
- `rotation drives the gain`
- `rotation provides a clear optimization benefit`
- `rotation is worth its runtime overhead in the current setting`

## Bottom line

Matched 3-epoch evidence does **not** justify keeping rotation as a main claimed contributor.

Current recommendation:
- keep rotation in the codebase and method family as an optional mechanism or appendix topic
- demote it in the main paper narrative
- center the story on additive diagonal adaptation unless future matched final-task evaluation changes the verdict
