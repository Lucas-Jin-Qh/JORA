# Round 6 Review — GPT-5.4 xhigh

**ThreadId**: `019cfef1-9bb6-7570-ba01-7f2c18b7eb58`
**Date**: 2026-03-18
**Verdict**: REVISE
**Overall Score**: 7/10

## Scores

| Dimension | Score |
|-----------|-------|
| Problem Fidelity | 8 |
| Method Specificity | 6 |
| Contribution Quality | 8 |
| Frontier Leverage | 8 |
| Feasibility | 5 |
| Validation Focus | 9 |
| Venue Readiness | 6 |
| **Overall** | **7** |

## Anchor Check
**Preserved.** The proposal still targets the original anchored problem: a better LoRA accuracy-vs-parameter frontier via a principled rotation-based PEFT mechanism. The scope is now narrower and cleaner, not drifted.

## Assessment
- **Dominant contribution**: sharp. The paper is now clearly about one mechanism: one-shot adaptive allocation of sparse bilateral rotation support inside rotation-based PEFT.
- **Method size**: simple and no longer overbuilt.
- **Frontier leverage**: appropriate. No forced trendy machinery.
- **More concrete/focused/current**: yes, materially so. But two blocking mechanism issues remain.

## Remaining Action Items

### CRITICAL
1. **Init bug — additive function jump at allocation**: With `theta=0`, `d_sel=1`, the adapter `delta = R_L^T · D_sel · R_R · x` is NOT zero at allocation time — it injects `P_U · x` immediately, causing the model function to jump at `t = T_stat`. The gradient fix is real but incomplete: nonzero gradients are preserved, but the pretrained function is not preserved. Fix by residualizing the adapter:
   ```
   delta = (R_L^T · D_sel · R_R - P_U) · x
   ```
   or parameterize `D_sel = I_U + ΔD` and subtract `P_U · x`. This gives zero function change at allocation while keeping nonzero gradients for theta.

2. **Success condition / operating point mismatch**: The success condition says `~40K–100K` params, but the actual JORA operating points are `~3K / 12K / 24K`. Fix by either changing the success condition to the actual regime (`~3K–25K`), or adding a larger JORA variant that lives in the `40K–100K` range.

### IMPORTANT
3. **Feasibility risk — overly aggressive empirical bet**: `~12K` matching `LoRA-r2 ~1M` is speculative. Stage the claim: primary target = outperform `LoRA-r1` and be competitive with `LoRA-r2`; stretch goal = match or beat `LoRA-r2`.

4. **Precommit Diag-only-selected interpretation rule**: If `Diag-only-selected` is within error bars of JORA, the contribution narrows to selected-support diagonal adaptation. Precommit this interpretation rule in the paper.

### MINOR
5. **Right-side statistic is a proxy, not a true task signal**: Present the method honestly as "gradient-guided output allocation + activation-proxy input allocation," not fully task-driven bilateral selection.

## Simplification Opportunities
- Remove `JORA-large` unless needed for curve shape credibility. `small + base` may be enough.
- State only one success target in the main text (not both primary and stretch).
- Keep `DoRA`, `OFT`, `BOFT` strictly appendix-only.

## Modernization Opportunities
NONE

## Drift Warning
NONE

<details>
<summary>Full Raw Review</summary>

**Problem Anchor**

`Preserved.` The proposal still targets the original anchored problem: a better LoRA accuracy-vs-parameter frontier via a principled rotation-based PEFT mechanism. The scope is now narrower and cleaner, not drifted.

**Scores**

1. `Problem Fidelity`: **8/10**
2. `Method Specificity`: **6/10**
3. `Contribution Quality`: **8/10**
4. `Frontier Leverage`: **8/10**
5. `Feasibility`: **5/10**
6. `Validation Focus`: **9/10**
7. `Venue Readiness`: **6/10**

**OVERALL SCORE**: **7/10**

**Assessment**

`Dominant contribution:` sharp. The paper is now clearly about one mechanism: one-shot adaptive allocation of sparse bilateral rotation support inside rotation-based PEFT.

`Method size:` simple and no longer overbuilt.

`Frontier leverage:` appropriate. This is modern enough already; no forced trendy machinery.

`More concrete / focused / current:` yes, materially so. But two blocking mechanism issues remain.

**Main Remaining Critiques**

1. `Missing mechanism: the init bug is still not fixed.`
With
```text
out = base_out + delta
delta = R_L^T · D_sel · R_R · x
```
and `theta=0`, `d_sel=1`, the adapter is **not** zero at allocation time. It injects `P_U x` immediately, so the model function jumps at `t = T_stat`. Saying the path is "identity on U" fixes gradients, but it does not preserve the pretrained function.
Concrete fix: residualize the adapter:
```text
delta = (R_L^T · D_sel · R_R - P_U) x
```
or parameterize `D_sel = I_U + ΔD` and subtract `P_U x`. That gives zero function change at allocation while keeping nonzero gradients for `theta`.
Priority: `CRITICAL`

2. `Method/claim mismatch: the parameter regime is still internally inconsistent.`
The success condition says `~40K–100K` params, but the actual operating points are `~3K / 12K / 24K`. That mismatch remains.
Concrete fix: either:
- change the success condition to the actual regime (`~3K–25K`), or
- add a larger JORA point that actually lives in the `40K–100K` range.
Priority: `CRITICAL`

3. `Feasibility risk: the extreme-budget Pareto claim is still a very aggressive empirical bet.`
`~12K` matching `LoRA-r2 ~1M` is possible as a speculative upside claim, but it is not a plausible default target. At this scale, negative results are likely unless the chosen scope is unusually favorable.
Concrete fix: make the must-win claim weaker and staged:
- primary: outperform `LoRA-r1` and be competitive with `LoRA-r2`
- stretch: match or beat `LoRA-r2`
Priority: `IMPORTANT`

4. `Mechanism isolation is good, but the main ambiguity is not gone until the controls win the argument.`
`Diag-only-selected` is now the right mandatory control. If it nearly matches JORA, then the contribution is not sparse rotation allocation; it is selected-support diagonal adaptation.
Concrete fix: precommit the interpretation rule in the paper. If `Diag-only-selected` is within error bars of JORA, narrow the claim.
Priority: `IMPORTANT`

5. `Weak training signal remains on the right side.`
The right-side statistic is only an activation proxy. That is acceptable if the claim stays narrow, but it is still weaker than the left-side gradient signal and may make the bilateral story look asymmetric.
Concrete fix: present the method honestly as `gradient-guided output allocation + activation-proxy input allocation`, not as fully task-driven bilateral selection.
Priority: `MINOR`

**Drift Warning**

`NONE`

**Simplification Opportunities**

- Remove `JORA-large` unless you need it to make the curve shape credible. `small + base` may be enough for the main story.
- State only one success target in the main text: either "competitive with `LoRA-r2`" or "matches `LoRA-r2`" as stretch, not both.
- Keep `DoRA`, `OFT`, and `BOFT` strictly appendix-only.

**Modernization Opportunities**

`NONE`

**Remaining Action Items**

1. Fix the additive-init bug with a residualized zero-function-change parameterization.
2. Reconcile the success-condition numbers with the actual JORA operating points.
3. Downgrade the empirical target from "expected to match `LoRA-r2`" to a stretch goal unless screening supports it.
4. Keep `Diag-only-selected` mandatory and predefine how its outcome changes the paper claim.
5. Phrase the right-side statistic honestly as a proxy, not a true task signal.

**Verdict**

`REVISE`

</details>
