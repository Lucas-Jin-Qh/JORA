# JORA Pivot Options

**Date**: 2026-04-28
**Trigger**: M1 results — JORA-Diag vs LoRA/DoRA baseline comparison completed. Triple failure confirmed.
**Purpose**: Adjudicate project direction before further resource commitment.

---

## Current State Summary

**Evidence accumulated**:

| Gate | Status | Evidence |
|---|---|---|
| M0 correctness (save/load/merge) | PASS | 27/27 tests |
| Rotation mechanism (ON vs NoRot) | FAIL | E1/E2: ~0 train_loss delta, 2.86× runtime cost |
| TC-CS coupling subspace | FAIL | E4–E8: 100% pair overlap, 2.7e-6 training delta |
| JORA vs LoRA quality | FAIL | E8: LoRA=1.95 vs JORA=2.24 (gap=0.28) |
| JORA vs DoRA quality | FAIL | E9: DoRA=1.95 vs JORA=2.24 (gap=0.28) |
| JORA vs LoRA runtime | FAIL | JORA=158 min vs LoRA=41 min (3.8× slower) |

**Triple failure**: Rotation contributes nothing, quality is far behind LoRA/DoRA, and runtime is worse despite fewer parameters.

**What has not been tried**: Different hyperparams for JORA (lr, init, epochs, lr_theta/lr_core), different layer targeting (attention-only vs all-linear), different model scales, longer training, etc.

---

## Decision Framework

Three options are evaluated against two criteria:
1. **Paper viability**: Can this produce a submission-ready paper?
2. **Resource efficiency**: How much additional GPU time is required before knowing the answer?

---

## Option A — Negative Result / Internal Report

**Description**: Accept that JORA-Diag (as currently implemented) does not support a competitive-method paper. Publish findings as a negative result documenting what does not work.

### Allowed Claims
- Diagonal JORA (with or without rotation) underperforms LoRA/DoRA on OPT-350m SFT at matched training setup.
- Sparse rotation provides no measurable benefit over diagonal-only adaptation in this setup.
- The TC-CS coupling subspace mechanism fails to differentiate from energy-based pairing on real activations.
- Additive diagonal PEFT (JORA-NoRot form) converges to ~2.24 train_loss vs LoRA's ~1.95 at 3ep.

### Required New Work
- **Minimal**: Write up the negative result narrative.
- Archive existing evidence as a technical report.
- No additional training runs needed.

### Expected Risk
- **Low effort, low reward**: Likely not publishable at a top venue as a standalone paper.
- Could be a workshop paper or blog post.

### Stop Condition
- If a workshop or blog outlet is acceptable, proceed with writing.
- If top-venue paper is required, do not pursue this option as the main path.

---

## Option B — Empirical Note: Diagonal Adaptation + Ablation Study

**Description**: Reposition the paper as an empirical note about what diagonal adaptation does and does not do, using JORA-NoRot as the primary method. Acknowledge the quality gap but focus on understanding the mechanism.

### Allowed Claims
- Diagonal PEFT adaptation converges to a different quality level than low-rank adaptation (LoRA/DoRA).
- Rotation provides no benefit in this configuration (with caveats about setup).
- The efficiency of diagonal adaptation is limited by its lack of expressive capacity.
- This is an exploratory study, not a competitive-method claim.

### Required New Work
- Write the paper framing the question as "what does diagonal adaptation learn vs low-rank?"
- Potentially add JORA-NoRot vs LoRA ablation at multiple param budgets.
- Acknowledge the quality gap explicitly rather than glossing over it.

### Expected Risk
- Medium effort, medium reward.
- Reviewers may question why this is a paper rather than an ablation in someone else's paper.
- Quality gap makes it hard to claim contribution.

### Stop Condition
- If the narrative can honestly center on "understanding diagonal adaptation" rather than "JORA beats LoRA", proceed.
- If the paper needs to claim competitive quality, do not pursue.

---

## Option C — Method Redesign (Major Pivot)

**Description**: Do not continue with the current additive DiagCore. Pivot to a fundamentally different operator before re-running expensive training. Only proceed if the core operator is redesigned.

### What Could Trigger Redesign

The root cause of failure is likely: **additive diagonal adaptation with zero init lacks the representational capacity to converge as fast as LoRA's low-rank updates**.

Hypotheses for redesign (each is speculative and requires validation before training):
1. **Residualized form**: Change from `Δ(x) = R^T Diag(d) R x` to `Δ(x) = R^T Diag(1+d) R x - x`. The current additive form makes the diagonal adapt relatively rather than absolutely.
2. **Non-zero init with lr scheduling**: Start from a stronger initialization (not zero) so the model can build on a non-trivial starting point.
3. **Attention-only targeting**: Instead of all-linear (including FFN), target only attention projections (q, k, v, out). This may be more comparable to LoRA's typical configuration.
4. **Layer-wise lr or warmup**: JORA's dual-LR (theta + core) may need warmup or layer-wise decay to stabilize.

### Required New Work
- **Step C1**: Validate redesign hypotheses offline (no training) — e.g., check that residualized form produces reasonable forward outputs at init.
- **Step C2**: If C1 passes, run a 1ep sanity with the new form, compare to LoRA r=1 1ep baseline (~40 min).
- **Step C3**: If C2 shows competitive 1ep loss, proceed to 3ep full comparison.
- **Step C4**: If C2 still shows large gap, stop. The operator is insufficient.

### Expected Risk
- High effort, high reward (if redesign works).
- Could still fail if the root cause is wrong.
- Requires careful offline validation before committing GPU hours.

### Stop Condition
- If redesign fails C1 (init validation), do not proceed to C2.
- If C2 (1ep sanity) shows gap > 0.05, do not proceed to C3.
- If C3 (3ep) does not close the gap with LoRA, pivot back to Option A/B.

---

## Decision

| | Option A | Option B | Option C |
|---|---|---|---|
| Paper viability | Low (workshop only) | Medium | High (if redesign works) |
| Additional GPU time | 0 | 0 | ~2–4 hours (C1 offline + C2 sanity) |
| Risk | None | Medium | High |
| Effort | Low | Medium | High |

**Recommended path**: **Option C with strict gates**.

- The method has not been properly tested: the current JORA-Diag uses zero init, additive form, all-linear targeting, and no warmup — all of which are questionable design choices.
- The failure may be fixable with a different operator design.
- But: no further training runs until the operator design is validated offline.

**Minimum viable Option C entry condition**: Offline validation (C1) shows that the redesigned operator produces reasonable outputs at initialization and is trainable without NaN.

---

## Immediate Next Steps (if Option C is chosen)

1. **[C1] Offline redesign validation**: Test whether residualized form + non-zero init produces valid outputs.
2. **[C1b] Attention-only targeting check**: Compare what `target_modules="q_proj,v_proj"` would give in terms of param count.
3. **[C2] 1ep sanity run**: If C1 passes, run one 1ep sanity with the new config (~40 min).
4. **[Decision gate after C2]**: If gap < 0.05 at 1ep, proceed to C3. Else stop.

---

## What NOT to Do

- Do not run large ablation matrices on the current failing setup.
- Do not write the paper as if the current evidence supports a competitive-method claim.
- Do not assume the quality gap is due to hyperparameters alone — the operator form may be fundamentally inadequate.
- Do not launch 3ep runs without passing the 1ep sanity gate.
