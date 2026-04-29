# Round 1 Review: TC-CS JORA

**Reviewer**: Codex (gpt-5.4)
**Date**: 2026-04-27
**Model reasoning effort**: xhigh

---

## Scores

| Dimension | Score |
|---|---|
| 1. Problem Fidelity | 8 |
| 2. Method Specificity | 6 |
| 3. Contribution Quality | 7 |
| 4. Frontier Leverage | 5 |
| 5. Feasibility | 7 |
| 6. Validation Focus | 8 |
| 7. Venue Readiness | 7 |
| **Overall (weighted)** | **6.8 / 10** |
| **Verdict** | **REVISE** |

---

## Round 1 Full Review

### Dimension 1: Problem Fidelity — 8/10

The method still attacks the right bottleneck. The anchor is honest: rotation has no irreplaceable function under matched ON/OFF comparison. The coupling subspace framing correctly reinterprets the failure as a role-assignment problem, not a parameter-count problem.

The stop rules are particularly well-designed. Stop Rule A (TC-CS-1S ≈ NoRot) correctly anchors the boundary between "rotation revival succeeds" and "rotation should be dropped entirely." This is honest research discipline.

**Deduction**: The proposal calls for coupling but never specifies the regime in which cross-dimension coupling is expected to matter. Is this about attention head mixing? FFN activation patterns? The proposal implies a coupling-relevant subspace exists in language adaptation without grounding it. This is a minor drift risk.

### Dimension 2: Method Specificity — 6/10 (CRITICAL GAP)

**The proposal has a significant specificity gap at the coupling relevance computation level.**

What is well-specified:
- Replacing `energy[i] * energy[j]` with `score_coupling(i,j) = c_ij * min(g_i, g_j)`
- One-sided restriction (`R_L = I`, `R_R^(S)` only)
- Subspace `S` derived from top-scoring pairs
- `m` as a hyperparameter

What is underspecified:
- **The `corr`/`E[a_i a_j]` proxy is ambiguous.** `|corr(activations_i, activations_j)|` requires tracking a full covariance matrix over `d` dimensions, which is `O(d²)` memory and computation. For OPT-350m (`d=1024` hidden), that's ~1M entries per layer per calibration step — potentially workable but nowhere near "minimal implementation." The proposal glosses over whether this lives in EMA buffers or requires a full batch-level accumulation.
- **The two-stage selection algorithm is vague.** "Take top-`m` dimensions that appear in the highest-scoring pairs" is underspecified: tie-breaking? overlapping pairs? dimension deduplication? The greedy disjointness requirement from the existing `select_top_k_pairs_gpu` already constrains this, but the proposal does not connect to it.
- **When and how `S` is computed is unclear.** Stage 1 says "~500-1000 steps, same data as main training." But the existing codebase already has a calibration mechanism: `t_stat` (number of calibration steps) + `pairs_freeze_after_warmup=True`. The proposal should explicitly say it reuses this infrastructure rather than proposing a new Stage 1 system.
- **`compute_delta()` changes are not discussed.** The diagram shows `R_L = I` but `compute_delta()` in `layer.py:488` applies both sides. The `single_sided` config flag (`config.py:132`) already supports `single_sided="right"`, which would skip `R_L` application. The proposal should explicitly reference this existing interface instead of implying a new code path.

**Priority: CRITICAL** — this is the core mechanism. If the coupling proxy is not precise, the entire hypothesis cannot be tested.

### Dimension 3: Contribution Quality — 7/10

The dominant contribution is clean: coupling-relevance-based subspace-restricted one-sided rotation. This is one mechanism-level change, not a parameter tweak. The parsimony is genuinely good — no new trainable components, no new losses, no auxiliary networks.

**Deduction**: The "optional supporting contribution" (diagnostic framework) is underspecified and could be confused as a second paper contribution. More importantly, the proposal acknowledges `CoupledPairCore` does not exist in it but the codebase already has a `CoupledPairCore` (`core.py:96-266`) that applies 2×2 coupled blocks on rotation pairs. The relationship between TC-CS and `CoupledPairCore` is never discussed. If TC-CS-1S restricts rotation to a coupling subspace and `CoupledPairCore` applies coupled 2×2 blocks to those pairs, these are complementary — but the proposal doesn't acknowledge the existing infrastructure.

### Dimension 4: Frontier Leverage — 5/10

This is the proposal's weakest dimension.

The approach is fundamentally a **hand-crafted statistics-based coupling metric** (EMA correlation, `c_ij * min(g_i, g_j)`). This is a pre-2022 technique. In 2026, the natural approach for identifying coupling-relevant dimensions in a language model would be:

- **Use the existing gradient EMA infrastructure more cleverly**: Instead of `c_ij = |corr(a_i, a_j)|`, use `c_ij = |g_i_grad * g_j_grad|` — the existing backward hook already tracks gradient EMAs, and correlated gradient signals directly indicate "these dimensions need to move together for the task."
- **Use Fisher Information or Jacobian-based metrics**: The literature on structural importance (e.g., Fisher leak, sensitivity-based pruning) already has principled metrics for "which dimensions are task-relevant and coupled."
- **Use a lightweight probe**: A single linear layer on top of frozen activations predicting a task label, then use the probe's gradient covariance as a coupling signal. This is the modern equivalent and is more principled than EMA-based co-activation tracking.

**The current `c_ij = |corr(activations_i, activations_j)|` is not a coupling signal — it's a raw activation correlation.** Dimensions can be correlated without being task-relevant, and task-relevant dimensions can be anti-correlated. The coupling signal should be `|corr(activations_i * task_gradient_i, activations_j * task_gradient_j)|` or equivalent.

### Dimension 5: Feasibility — 7/10

The implementation maps cleanly to existing modules — `selection.py`, `layer.py`, `config.py` — and no changes to `core.py`, `rotation.py`, or `model.py` are required. This is a genuine strength. The existing `single_sided` flag, `t_stat` calibration, and `pairs_freeze_after_warmup` mechanism are all directly reusable.

**Deduction**: The `corr` proxy is the feasibility risk. A full pairwise correlation matrix over `d` dimensions per layer requires `d(d-1)/2` float32 entries. For OPT-350m (d=1024), that's ~524K entries per layer. With ~32 JORA layers, that's ~17M entries total — manageable but non-trivial during calibration. The proposal should specify whether this is per-layer or shared, and whether it uses EMA updates (cheap) or batch accumulation (expensive).

### Dimension 6: Validation Focus — 8/10

Excellent. The validation matrix is minimal and sufficient:
- JORA-NoRot (decisive mechanism baseline)
- Current JORA-Diag (status-quo)
- TC-CS-1S (proposed)
- Coupling-random baseline (mechanism specificity check)

Four runs with a clear decision tree. The stop rules make the outcome actionable regardless of which direction the result goes.

**Deduction**: The proposal mentions "runtime overhead" as a critical metric but doesn't specify what overhead ratio would fail the "justifiable" bar. Should it be within 1.5× NoRot? 2×? Given that current JORA-Diag is already 3×, a TC-CS-1S that is 1.5× NoRot would still be a 2× improvement over current, which might be acceptable.

### Dimension 7: Venue Readiness — 7/10

The EMNLP narrative framing is appropriate — "language adaptation geometry," coupling subspace, one-sided restriction. The proposal is compact and mechanism-focused. The stop-rule discipline is venue-compatible (referees appreciate falsifiable claims).

**Deduction**: The proposal does not discuss related work positioning. The coupling subspace idea is adjacent to structured pruning (Magnav.jl, LoRAPruner) and Fisher-based importance metrics. A reviewer will ask about these connections. The proposal should include a brief related-work paragraph noting the distinction.

---

## Simplification Opportunities

1. **Drop the Stage 1/Stage 2 terminology.** The codebase already has `t_stat` calibration + `pairs_freeze_after_warmup`. Reuse these terms and configs. A new "Stage 1" system adds conceptual weight without new functionality.

2. **The "optional supporting contribution" (diagnostic framework) should be demoted or deleted.** It is vague and competes with the main mechanism. Move any diagnostic logging to the `callbacks.py` section as implementation detail, not a contribution.

3. **The calibration phase can reuse `grad_col_ema` for the correlation proxy.** Instead of a new `c_ij` buffer, compute `c_ij = |grad_col_ema[i] * grad_col_ema[j]|` (using the existing gradient EMA as a proxy for "which dimensions are task-relevant"). This is `O(d²)` but leverages existing infrastructure. This simplifies the proposal and avoids introducing a new statistic collection mechanism.

## Modernization Opportunities

1. **Replace raw activation correlation with gradient-guided coupling.** The proposal's `c_ij = |corr(a_i, a_j)|` is not task-conditioned — correlated activations may be irrelevant to the task. Replace with `c_ij = |g_i * g_j|` where `g_i` is the existing gradient EMA. This makes the coupling signal task-conditioned without any new infrastructure.

2. **Consider probe-based subspace identification.** Instead of hand-crafted statistics, use a lightweight linear probe on frozen activations (trained for 500 steps on the task data) and use the probe's gradient covariance as the coupling signal. This is the modern equivalent and is more principled than EMA-based co-activation tracking.

3. **The coupling signal should explicitly interact with the rotation parameterization.** The current `score_coupling = c_ij * min(g_i, g_j)` is a static pre-training signal. But Givens rotations inherently model 2D coupling — the proposal should discuss whether the coupling signal should directly inform the rotation angle parameterization (e.g., initial angle ∝ coupling strength), not just the support selection.

## Drift Warning

**MINOR DRIFT RISK**: The proposal asserts that "a regime that plausibly requires cross-dimension coupling" exists in language adaptation but never specifies which layer types, or which model scales. If the coupling subspace only helps in FFN layers (not attention) or only at certain model scales, the proposal's framing is too general. A reviewer will ask: "Why would attention projections benefit from rotation-based coupling but not from full LoRA?" The proposal should address this at least at the intuition level.

## Verdict

**REVISE**

The direction is promising and methodologically honest. The falsifiability discipline (stop rules, NoRot as decisive baseline) is exactly right for a top venue. However, the proposal has a **CRITICAL specificity gap at the coupling proxy level** — the core mechanism is described abstractly rather than concretely enough for implementation. Additionally, the **frontier leverage dimension is weak** — the approach uses pre-2022 hand-crafted statistics where modern approaches (probe-based, Fisher-based, or at minimum gradient-guided coupling) would be more principled and more defensible to reviewers.

**The single most impactful fix**: Specify the coupling proxy as `c_ij = |grad_col_ema[i] * grad_col_ema[j]|` (reusing existing gradient EMAs as a task-conditioning signal) and implement it in `selection.py` as a new `select_coupling_pairs_gpu()` function that replaces `select_top_k_pairs_gpu` under the new `subspace_selection="coupling"` config flag. This eliminates the need for new buffers, connects directly to existing infrastructure, and makes the signal task-conditioned.
