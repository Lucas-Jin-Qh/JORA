# Round 2 Review: TC-CS JORA

**Reviewer**: Codex (gpt-5.4) — Round 2 re-evaluation
**Date**: 2026-04-27
**Thread**: continued from round-1

---

## Scores

| Dimension | Round 0 | Round 1 | Round 2 | Change R1→R2 |
|---|---|---|---|---|
| 1. Problem Fidelity | 8 | 8 | 8 | 0 |
| 2. Method Specificity | 6 | 7 | 7.5 | +0.5 |
| 3. Contribution Quality | 7 | 7.5 | 7.5 | 0 |
| 4. Frontier Leverage | 5 | 5.5 | 5.5 | 0 |
| 5. Feasibility | 7 | 6 | 6.5 | +0.5 |
| 6. Validation Focus | 8 | 8 | 8 | 0 |
| 7. Venue Readiness | 7 | 7.5 | 7.5 | 0 |
| **Overall (weighted)** | **6.8** | **6.95** | **7.2** | **+0.25** |
| **Verdict** | **REVISE** | **REVISE** | **REVISE** | — |

---

## Round 2 Full Review

### Dimension 1: Problem Fidelity — 8/10 (unchanged)

Still solving the right problem. The coupling subspace framing remains the most honest interpretation of the current failure. The stop rules are discipline-maintaining.

### Dimension 2: Method Specificity — 7.5/10 (+0.5)

The coupling proxy formula is concrete. The code mapping table is accurate. The existing config flags (`single_sided`, `t_stat`, `pairs_freeze_after_warmup`) are correctly referenced.

**Remaining specificity gap: the `cov_ij_ema` update mechanism.**

The backward hook (`layer.py:644-692`) updates `grad_row_ema` using:
```python
g_sq = g.reshape(-1, st.n).float().pow(2).mean(dim=0)  # → (d,) tensor
st.grad_row_ema.lerp_(g_sq, 1.0 - beta)
```

This is the **per-dimension gradient second moment**, not the raw gradient. The raw activation `x` from the forward pass is **never available in the backward hook** — only `g = grad_output[0]` (output gradients) are available.

To compute `cov_ij_ema[i,j] = E[a_i * a_j]` requires the raw activation `x`, which would need a new forward-pass storage path. This was identified as the blocking issue.

**The Round 1 refinement proposed activation cross-covariance but did not specify this path.** The proposal implicitly assumed raw activation would be available, but it is not.

**The concrete fix**: Pivot from activation cross-covariance to **gradient outer product coupling**, which is task-conditioned by definition (gradients reflect the loss landscape) and uses the existing backward hook infrastructure directly.

### Dimension 3: Contribution Quality — 7.5/10 (unchanged)

One mechanism-level change. Coupling-relevance pairing + subspace restriction + one-sided rotation. The Phase 1/2 clarity is good. The contribution remains sharp.

**Remaining concern**: The `min(g_i, g_j)` vs `sqrt(g_i * g_j)` question in the pairing score is still unresolved. The `min` creates a discontinuity that could cause unstable pair selection near the threshold. This is a minor calibration concern.

### Dimension 4: Frontier Leverage — 5.5/10 (unchanged)

The gradient-based fallback options (Phase 2) show awareness. The task-conditioning argument (backward hook only fires on training steps) is valid.

**Still the weakest dimension.** A modern reviewer will ask: "Why gradient EMA outer product rather than Fisher Information?" The proposal should either address this explicitly or acknowledge that gradient EMA is an approximation of Fisher with lower memory/compute cost.

### Dimension 5: Feasibility — 6.5/10 (+0.5)

The gradient outer product pivot addresses the blocking issue identified in Round 1.

**New feasibility assessment for gradient outer product**:

```python
# In the backward hook (reuse existing infrastructure pattern):
g = grad_output[0].detach()  # shape: [..., out_features]
g_flat = g.reshape(-1, st.n).float()  # (B*L, d)
# Gradient coupling: outer product per step, accumulated via EMA
g_outer = g_flat.T @ g_flat / max(g_flat.size(0), 1)  # (d, d)
g_ij = g_outer.abs()  # |g_i * g_j| — coupling magnitude
st.g_cov_ema.lerp_(g_ij, 1.0 - beta)
```

This is **O((B*L) × d²)** FLOPs per step per layer — same compute as the activation outer product — but it uses the existing backward hook pattern without requiring new forward-pass storage.

Memory: `(d, d)` buffer per layer. For OPT-350m (d=1024): 1M float32 = 4 MB/layer × 24 layers = 96 MB. For LLaMA-7B (d=4096): 67 MB/layer × 32 layers = **2.1 GB**. Same as the activation approach, but with cleaner code plumbing.

The coupling score becomes:
```
c_ij = g_cov_ema[i,j] / (g_ii_ema[i] * g_jj_ema[j])^0.5
score_coupling(i,j) = c_ij * min(g_i, g_j)
```
where `g_ii_ema` is the diagonal of the gradient covariance EMA (already partially captured by `grad_row_ema`). Actually, we can simplify further:

```
# Diagonal of gradient outer product is just the gradient squared per dimension
g_diag = g_sq  # from existing hook, shape (d,)
# Off-diagonal: g_ij = E[g_i * g_j]
g_cov_ema[i,j] = E[g_i * g_j]
# Normalized: c_ij = |g_cov_ema[i,j]| / sqrt(g_row_ema[i] * g_row_ema[j])
# where g_row_ema is the per-dimension gradient second moment
```

The gradient outer product approach reuses more of the existing infrastructure (the `g_sq` computation is already there), making it the cleaner implementation.

**However, one concern remains**: The gradient covariance EMA accumulates `E[g_i * g_j]` where `g` is the output gradient of this layer. But for the coupling signal to be task-relevant, we want to know which input dimensions `i` and `j` of THIS layer need to couple. The output gradient tells us how the loss is sensitive to the output of this layer, but the coupling between input dimensions `i` and `j` should reflect how the function this layer computes benefits from coupling them. This is a subtle but important distinction.

- `grad_row_ema` = `E[g_j²]` = output gradient energy per output dimension (row)
- `grad_col_ema` = `E[x_i²]` = input activation energy per input dimension (col)
- `g_cov_ema[i,j]` = `E[g_i * g_j]` from the backward hook... but wait, `g` in the backward hook is the gradient w.r.t. the OUTPUT of this layer, not the gradient w.r.t. the INPUT. The backward hook fires on `grad_output[0]`, which has shape `[..., out_features]`.

So `g_cov_ema` from the backward hook captures output-side coupling, not input-side coupling. For input-side coupling (which is what `R_R` operates on), we'd need the gradient of the loss with respect to the INPUT `x`, which is `grad_input`, not `grad_output`. The backward hook receives both as arguments. But `grad_input` for a linear layer contains the gradient w.r.t. the weight, not w.r.t. the input — so this doesn't give us input-dimension coupling either.

**The coupling signal must come from the forward pass.** For a frozen layer, the forward pass input `x` is the representation produced by the previous layer. Which dimensions of `x` need to be coupled for the task? The natural signal for this is: (a) the activation of `x` itself (which dimensions co-activate during the task), or (b) the gradient of the loss w.r.t. `x` (which dimensions move together when the loss changes). Option (b) requires storing `grad_input` from the backward hook, which is available as `grad_input[0]` in the hook signature.

So the correct path for a gradient-based coupling signal is:
1. In the backward hook, capture `grad_input[0]` (gradient w.r.t. input x) — available via the hook's `grad_input` argument
2. Compute `g_in = grad_input[0].reshape(-1, d)` — shape `(B*L, d)` 
3. Accumulate `g_cov_ema[i,j] = E[g_in_i * g_in_j]` via EMA

This is task-conditioned (gradients reflect the loss), input-side (captures coupling at the input of this layer, where `R_R` acts), and uses the existing backward hook pattern. This is the right signal.

**Revised implementation in the backward hook:**
```python
def _backward_hook(self, module, grad_input, grad_output):
    # grad_output: gradient w.r.t. output [.., out_features]
    # grad_input: gradient w.r.t. input [.., in_features] — THIS IS THE SIGNAL
    g = grad_output[0].detach()
    g_sq = g.reshape(-1, st.n).float().pow(2).mean(dim=0)
    st.grad_row_ema.lerp_(g_sq, 1.0 - beta)  # existing

    # NEW: gradient coupling (input-side)
    if grad_input and grad_input[0] is not None:
        g_in = grad_input[0].detach().reshape(-1, st.m).float()
        g_in_cov = g_in.T @ g_in / max(g_in.size(0), 1)  # (d, d)
        st.g_cov_ema.lerp_(g_in_cov, 1.0 - beta)
```

**One more concern**: `grad_input` for a frozen layer with respect to its input — is this even non-None? For a `register_full_backward_hook` on an `nn.Linear`, `grad_input` is the gradient w.r.t. the input tensor, which exists if the input required a gradient. This should work in the typical fine-tuning setup. However, if `x` was detached or if the previous layer doesn't have gradients, `grad_input` could be None. The hook already has a guard for `grad_output[0] is None` — we need a similar guard for `grad_input`.

### Dimension 6: Validation Focus — 8/10 (unchanged)

The validation matrix is still minimal and sufficient. The coupling-random ablation remains the right mechanism check. The stop rules are discipline-maintaining.

### Dimension 7: Venue Readiness — 7.5/10 (unchanged)

The EMNLP framing is appropriate. The stop-rule discipline is venue-compatible. Still missing an explicit paragraph distinguishing this from Fisher-based pruning and attention-pattern coupling.

---

## Problem Anchor: PRESERVED

The pivot from activation to gradient coupling is within the same hypothesis: rotation should be restricted to dimensions that are coupled. The stop rules and success condition are unchanged.

## Dominant Contribution: STILL SHARP

One mechanism-level change. The gradient coupling signal is actually more principled than activation correlation — it directly measures which input dimensions move together when the task loss changes, which is the right coupling signal for a frozen layer.

## Method Simplicity: IMPROVED

The gradient outer product pivot is cleaner than activation storage. The backward hook already has the pattern for `(B*L, d)` → `(d,)` reductions. The `(d, d)` outer product is a natural extension. The `g_cov_ema` buffer reuses the same `lerp_` pattern as `grad_row_ema` and `grad_col_ema`.

## Frontier Leverage: STILL WEAK but DEFENSIBLE

Gradient covariance EMA is a principled signal. It is more task-relevant than activation correlation. A reviewer asking "why not Fisher?" gets: "gradient outer product EMA is an online approximation of Fisher information with O(d²) memory instead of O(d²) + per-sample storage, making it practical for online calibration without a separate Fisher computation pass."

---

## Drift Warning: NONE

The pivot from activation to gradient coupling stays within the coupling subspace hypothesis. This is the same mechanism with a cleaner signal.

---

## Simplification Opportunities

1. **The `min(g_i, g_j)` in the pairing score can be replaced by a soft version.** Using `sqrt(g_i * g_j + eps)` is symmetric and differentiable, avoiding the discontinuity at the threshold. This is a one-line change.

2. **The normalization of `c_ij` can be simplified.** Instead of dividing by `sqrt(g_row_ema[i] * g_row_ema[j])`, just use the unnormalized `g_cov_ema[i,j]` directly — the outer product already captures both the coupling strength and the magnitude. The `min(g_i, g_j)` signal (from `grad_col_ema`) provides the dimension-importance reweighting. The final score `score_coupling(i,j) = g_cov_ema[i,j] * min(grad_col_ema[i], grad_col_ema[j])` is sufficient without additional normalization. This simplifies the proposal from 3 buffers (`grad_row_ema`, `grad_col_ema`, `g_cov_ema`) to 2 (`grad_col_ema` for magnitude + `g_cov_ema` for coupling).

3. **Drop `CoupledPairCore` from the Phase 2 narrative** — keep it in a footnote. The main proposal should be TC-CS-1S vs NoRot.

---

## Modernization Opportunities

1. **Acknowledge the Fisher connection explicitly.** "Gradient covariance EMA is an online approximation of Fisher information" is a credible and brief statement that positions the method in modern context without adding complexity.

---

## Remaining Action Items

**Before READY — must resolve:**

1. **[CRITICAL] Implement the backward hook change.** Add `g_cov_ema` buffer to `_JoraAdapterState.__init__()`. Add gradient covariance update in `_backward_hook()` using `grad_input[0]` (not `grad_output`). Guard for `grad_input[0] is None`.

2. **[IMPORTANT] Simplify the pairing score.** Replace `score_coupling(i,j) = c_ij * min(g_i, g_j)` with `score_coupling(i,j) = g_cov_ema[i,j] * min(grad_col_ema[i], grad_col_ema[j])`. Remove the normalization step — it's unnecessary and complicates the pairing score.

3. **[MINOR] Add the Fisher connection statement** in the method section: "Gradient covariance EMA is an online approximation of Fisher information for the task loss, making it a principled task-conditioning signal."

---

## Verdict: **REVISE**

The gradient outer product pivot addresses the blocking implementation issue. The coupling signal is now cleaner, more principled, and uses the existing backward hook pattern without requiring new forward-pass storage. The proposal is close to READY — one more round addressing the hook implementation and the simplified pairing score should get it there.

**The score is 7.2/10, trending toward READY.** The remaining issues are implementation details, not conceptual gaps.
