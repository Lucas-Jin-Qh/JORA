# Review Summary

**Problem**: How to give rotation an irreplaceable role in additive JORA-Diag, so that one-sided subspace-restricted coupling rotation outperforms NoRot?
**Initial Approach**: TC-CS JORA — Task-Conditioned Coupling Subspace Rotation, one-sided first
**Date**: 2026-04-27
**Rounds**: 3 / 5 MAX_ROUNDS
**Final Score**: 7.75 / 10
**Final Verdict**: Close to READY — coupling regime paragraph added (2026-04-27); ready for /experiment-plan

---

## Problem Anchor

- **Bottom-line problem**: Current additive JORA-Diag cannot justify rotation's role — matched 1ep and 3ep ON/OFF comparisons show near-zero quality difference while ON costs ~3× runtime. Rotation has no irreplaceable function.
- **Must-solve bottleneck**: Rotation acts as a global sparse basis reparameterization that diagonal scaling can absorb. Without a coupling-specific role, rotation cannot justify its computational cost.
- **Non-goals**: Not claiming "rotation drives the gain" without evidence. Not building a broad PEFT framework. Not rewriting into residualized full-support JORA. Not proving arbitrary sparse rotation helps by default.
- **Constraints**: Compatible with additive JORA-Diag code path. Compact enough for EMNLP narrative. Runtime cost must be justified.
- **Success condition**: A modified rotation design shows a clear, matched advantage over NoRot in a coupling-required regime, explainable as a necessary mechanism.

---

## Round-by-Round Resolution Log

| Round | Main Reviewer Concerns | What This Round Simplified / Modernized | Solved? | Remaining Risk |
|-------|------------------------|------------------------------------------|---------|----------------|
| 0 | Initial proposal: coupling proxy vague (`c_ij = |corr|`), unclear coupling vs importance distinction, unclear codebase mapping | Established the coupling subspace framing, stop rules, one-sided-first discipline | Partial | Proxy implementation unspecified |
| 1 | Coupling proxy ambiguous — requires O(d²) correlation matrix; Stage 1/2 terminology unnecessary; diagnostic framework clutters contributions | Connected to existing `single_sided`, `t_stat`, `pairs_freeze_after_warmup` flags; concretized proxy formula; dropped Stage 1/2 terminology | Partial | `cov_ij_ema` update path underspecified — backward hook sees no raw activations |
| 2 | `cov_ij_ema` update mechanism blocked by hook signature; gradient path (`grad_input`) rejected — weight gradient, not input gradient; `min` vs `sqrt` discontinuity | Pivoted to forward-pass accumulation of activation outer product EMA; simplified score formula; Fisher connection added; CoupledPairCore moved to footnote | Partial | Score of 7.75 — needs coupling regime specification to reach READY |

---

## Overall Evolution

- **How the method became more concrete**: From "use coupling-relevant selection" to a fully-specified activation outer product EMA (`g_cov_ema[i,j] = E[x_i * x_j]`) accumulated in the forward pass during calibration, combined with a pairing score of `|g_cov_ema[i,j]| * sqrt(grad_col_ema[i] * grad_col_ema[j])`.
- **How the dominant contribution became more focused**: Dropped the "diagnostic framework" supporting contribution. Moved CoupledPairCore to footnote. Main paper is now cleanly: TC-CS-1S (coupling subspace + one-sided + DiagCore) vs NoRot.
- **How unnecessary complexity was removed**: Dropped Stage 1/2/3 terminology in favor of existing `t_stat` + `pairs_freeze_after_warmup`. Dropped correlation normalization step. Reused `single_sided="right"` flag for one-sided restriction.
- **How modern technical leverage improved**: Added Fisher connection — `g_cov_ema[i,j] = E[x_i * x_j]` is the empirical Fisher for the task loss, making the coupling signal task-conditioned and grounded in information geometry rather than hand-picked statistics.
- **How drift was avoided**: The coupling subspace framing was never changed. The success condition (TC-CS-1S > NoRot) was never changed. Stop rules were never weakened.

---

## Final Status

- **Anchor status**: Preserved across all rounds.
- **Focus status**: Tight — one dominant contribution.
- **Modernity status**: Appropriately frontier-aware — empirical Fisher framing, no forced trendy components.
- **Strongest parts of final method**:
  1. Activation outer product EMA is task-conditioned by construction (forward pass only on training data)
  2. One-sided restriction via existing `single_sided="right"` — zero architectural changes
  3. Stop rules are falsifiable and actionable regardless of outcome
  4. Fisher connection positions the method in a principled geometric framework
- **Remaining weaknesses**:
  1. Coupling regime not specified at layer-type level (attention vs FFN) — one paragraph needed
  2. Outer product at d=4096 scale needs code optimization note for production use
  3. Score of 7.75, 1.25 points below READY threshold

---

## Score Evolution

| Round | Problem Fidelity | Method Specificity | Contribution Quality | Frontier Leverage | Feasibility | Validation Focus | Venue Readiness | Overall | Verdict |
|-------|-----------------|--------------------|----------------------|-------------------|-------------|------------------|-----------------|---------|---------|
| 0 | 8 | 6 | 7 | 5 | 7 | 8 | 7 | 6.80 | REVISE |
| 1 | 8 | 7 | 7.5 | 5.5 | 6 | 8 | 7.5 | 6.95 | REVISE |
| 2 | 8 | 8 | 8 | 7 | 8 | 8 | 8 | 7.75 | REVISE |
