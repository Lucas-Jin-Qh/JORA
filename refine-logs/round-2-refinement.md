# Round 2 Refinement

## Problem Anchor (verbatim)

- **Bottom-line problem**: Current additive JORA-Diag cannot justify rotation's role — matched 1ep and 3ep ON/OFF comparisons show near-zero quality difference while ON costs ~3× runtime. Rotation has no irreplaceable function.
- **Must-solve bottleneck**: The current rotation mechanism is assigned no unique task; it acts as a global sparse basis reparameterization that diagonal scaling can absorb. Without a coupling-specific role, rotation cannot justify its computational cost.
- **Non-goals**: Not claiming "rotation drives the gain" without evidence. Not building a broad PEFT framework. Not rewriting into residualized full-support JORA in this proposal. Not proving that arbitrary sparse rotation helps by default.
- **Constraints**: Must stay compatible with current additive JORA-Diag code path. Must remain compact enough for EMNLP narrative. Runtime cost must be justified. Story must sound like language adaptation geometry.
- **Success condition**: A modified rotation design shows a clear, matched advantage over NoRot in a regime that plausibly requires cross-dimension coupling, while remaining explainable as a necessary mechanism rather than a tuning artifact.

---

## Anchor Check

- **Original bottleneck**: Rotation is applied globally without a coupling-specific function, so diagonal scaling absorbs its effect.
- **Why the revised method still addresses it**: TC-CS-1S restricts rotation to a task-geometry-derived coupling subspace on one side only, giving rotation a unique geometric role that diagonal-only scaling cannot perform.
- **Reviewer suggestions that would cause drift if followed blindly**: The reviewer's gradient outer product path (using `grad_input`) was carefully analyzed. `grad_input` for this layer is the gradient w.r.t. the weight, not w.r.t. the input — it has shape `[in_features, out_features]`, not `[..., in_features]`. Using it would not give input-dimension coupling. The activation outer product path is the correct approach.

---

## Simplicity Check

- **Dominant contribution after revision**: Task-conditioned coupling-subspace one-sided rotation for additive JORA-Diag.
- **Components removed or merged**:
  - Dropped the "optional supporting contribution" (diagnostic framework) from contributions — demoted to implementation detail.
  - Simplified pairing score: removed explicit correlation normalization, using unnormalized activation covariance directly.
  - Dropped CoupledPairCore from Phase 2 narrative — mentioned in footnote only.
- **Reviewer suggestions rejected as unnecessary complexity**: Fisher Information requires a second-order computation pass. Gradient outer product using `grad_input` was analyzed and rejected — `grad_input` for a frozen layer is weight gradient, not input gradient, so it does not capture input-dimension coupling.
- **Why the remaining mechanism is still the smallest adequate route**: Only the pairing scoring function changes. One new buffer (`g_cov_ema`, `(d, d)` per layer). Forward accumulation of activation outer product, no new backward hook logic. No new training phases, no new loss functions.

---

## Changes Made

### 1. Pivoted coupling proxy from activation correlation to **activation outer product EMA**

- **Reviewer said**: The backward hook only sees `grad_output[0]`, not raw activations. The activation correlation path was underspecified.
- **Action**: Specified concrete forward-accumulation implementation for activation outer product:
  - **Forward pass** (during `t_stat` calibration): accumulate `x_flat.T @ x_flat` into `g_cov_ema` directly, using the existing `ema_update_interval` gating.
  - **No backward hook change needed**: The EMA update happens in the forward pass, not the backward hook.
  - The `g_cov_ema[i,j]` entry becomes `E_calibration[x_i * x_j]` — the empirical activation cross-moment over calibration batches.
- **Reasoning**: The forward pass already has a `training` guard and an `ema_update_interval` gate. Adding the outer product accumulation here is clean — no need for backward hook changes or activation storage. The outer product `X^T X` (where X is `(B*L, d)`) is computed in fp32 for numerical stability. For d=1024, B*L=512: 512 × 1024² = 538M FLOPs per calibration step. Manageable.
- **Why it is task-conditioned**: The forward pass only runs on training data during calibration steps. The accumulated `E[x_i * x_j]` reflects the covariance structure of task-relevant representations.
- **Important implementation note**: `g_cov_ema[i,j]` is the unnormalized cross-moment `E[x_i * x_j]`. The pairing score will combine this with `grad_col_ema` (activation magnitude) and then normalize by the diagonal before computing the pairing score.
- **Impact on core method**: One new buffer `g_cov_ema`. Forward pass extended with outer product accumulation. No backward hook changes.

### 2. Simplified pairing score

- **Reviewer said**: `c_ij * min(g_i, g_j)` with explicit correlation normalization is over-engineered.
- **Action**: Use unnormalized activation covariance directly:

```
score_coupling(i,j) = g_cov_ema[i,j] * sqrt(grad_col_ema[i] * grad_col_ema[j] + eps)
```

Where `g_cov_ema[i,j] = E_calibration[x_i * x_j]` (activation cross-moment) and `grad_col_ema[i] = E_calibration[x_i²]` (activation second moment, existing buffer).

`sqrt(grad_col_ema[i] * grad_col_ema[j])` is the geometric mean of per-dimension activation energy — symmetric, differentiable, rewards both dimensions having high activation.

Note: `g_cov_ema[i,j]` can be negative if `x_i` and `x_j` are anti-correlated. Taking `|g_cov_ema[i,j]|` gives the coupling magnitude regardless of sign. We use `|g_cov_ema[i,j]|` as the coupling signal.

Final score:
```
score_coupling(i,j) = |g_cov_ema[i,j]| * sqrt(grad_col_ema[i] * grad_col_ema[j] + eps)
```

- **Reasoning**: Simpler. Directly combines "dimensions couple" (covariance magnitude) with "dimensions matter" (activation energy). No explicit normalization step needed — the diagonal of `g_cov_ema` is `E[x_i²] = grad_col_ema[i]`, so the normalized covariance `corr(x_i, x_j) = |g_cov_ema[i,j]| / sqrt(g_cov_ema[i,i] * g_cov_ema[j,j])` is equivalent to `|g_cov_ema[i,j]| / sqrt(grad_col_ema[i] * grad_col_ema[j])`. The score is exactly the unnormalized correlation coefficient.
- **Impact on core method**: Cleaner score function.

### 3. Added memory management for large models

- **Reviewer said**: At d=4096, `g_cov_ema` is 67 MB per layer — 2.1 GB for 32 layers. Should specify disable mechanism.
- **Action**: After `t_stat` calibration completes and `pairs_freeze_after_warmup=True`, `g_cov_ema` is disabled. A `disable_cov_ema()` method sets the buffer to `None`, allowing GC to reclaim memory. The pairing score computation happens after calibration, before this cleanup.
- **Reasoning**: Necessary for large model runs.
- **Impact on core method**: Memory management, no change to algorithm.

### 4. Simplified Phase 2 narrative

- **Reviewer said**: CoupledPairCore in Phase 2 makes the proposal feel like a two-phase roadmap.
- **Action**: Moved to footnote only. Main paper is TC-CS-1S vs NoRot.
- **Reasoning**: Reduces perceived scope.
- **Impact on core method**: None.

---

## Revised Proposal

# Research Proposal: TC-CS JORA

*Round 2 Refinement — activation outer product coupling proxy, simplified scoring*

---

### Problem Anchor

Same as Round 0.

### Method Thesis

**Restrict JORA rotation to a task-conditioned coupling subspace, and introduce that restriction on one side first, so rotation models cross-dimension coupling only where diagonal-only language adaptation is insufficient.**

### Complexity Budget

| Category | Status |
|---|---|
| Frozen backbone | reused |
| DiagCore | reused as-is |
| Givens rotation parameterization | reused as-is |
| `compute_delta()` | unchanged |
| `single_sided="right"` | reused (R_L=I) |
| `t_stat` calibration | reused |
| `pairs_freeze_after_warmup=True` | reused |
| Pairing scoring function | **changed** (energy product → activation outer product EMA × activation magnitude) |
| New buffer: `g_cov_ema` | **added** (`(d, d)` per layer, calibration only, then disabled) |
| CoupledPairCore | footnote only |
| Bilateral expansion | Phase 2 only |
| New trainable components | none |

### System Overview

```
Current JORA-Diag (S_L=32, S_R=32, all dims):
    x → [R_R (global sparse, energy[i]*energy[j] pairing)] → [DiagCore] → [R_L^T (global sparse)] → Δ(x)

Proposed TC-CS-1S:
    Config: single_sided="right", core="diag"
    x → [R_R^(S) (subspace-restricted, activation-covariance pairing)] → [DiagCore] → [R_L=I] → Δ(x)

Calibration (t_stat steps):
    - Forward: accumulate x_flat^T @ x_flat → g_cov_ema (activation outer product EMA)
    - Pair selection: score = |g_cov_ema[i,j]| * sqrt(grad_col_ema[i]*grad_col_ema[j])
    - After t_stat: pairs frozen, g_cov_ema disabled
```

### Core Mechanism

#### Step A: Activation Outer Product EMA (forward pass, calibration only)

**Forward pass** (`layer.py` forward hook, during `t_stat` calibration steps):

The existing forward hook already has a `training` guard and an `ema_update_interval` gate. Extend it to accumulate the activation outer product:

```python
# In the forward hook (layer.py ~line 685), during t_stat calibration:
ema_interval = int(getattr(st.cfg, "ema_update_interval", 1))
if (ema_interval <= 1 or (self._ema_step_counter % ema_interval) == 0):
    xd = x.detach().float()  # [..., d]
    x_flat = xd.reshape(-1, st.m)  # (B*L, d) — flatten batch+seq dims

    # Existing: per-dimension activation second moment
    x_sq = x_flat.pow(2).mean(dim=0)  # (d,)
    st.grad_col_ema.lerp_(x_sq, 1.0 - beta)

    # NEW: activation outer product EMA (calibration only)
    if st.cfg.pairing_strategy == "coupling" and st.cfg.calibration_active:
        # Accumulate E[x_i * x_j] via EMA
        # x_flat^T @ x_flat: (d, B*L) @ (B*L, d) = (d, d)
        x_cov = x_flat.T @ x_flat / max(x_flat.size(0), 1)  # (d, d)
        st.g_cov_ema.lerp_(x_cov, 1.0 - beta)
```

Key properties:
- **Task-conditioned**: The forward pass only runs on training samples, so `g_cov_ema[i,j] = E_calibration[x_i * x_j]` captures the task-relevant covariance structure.
- **No backward hook changes needed**: The EMA update happens in the forward pass, reusing the existing `training` guard and `ema_update_interval` gating.
- **EMA accumulation**: Each calibration batch contributes `(B*L, d) → (d, d)` outer product. The EMA smoothly estimates the full population covariance over `t_stat` batches.

**Why not gradient covariance (grad_input path)?** The review considered using `grad_input` from the backward hook as a gradient-based coupling signal. However, for a frozen layer under `register_full_backward_hook`, `grad_input` is the gradient w.r.t. the weight matrix (shape `[in_features, out_features]`), not the gradient w.r.t. the input tensor. It does not provide input-dimension coupling information. The activation outer product is the correct signal.

**Fisher connection**: `g_cov_ema[i,j] = E[x_i * x_j]` is related to the empirical Fisher information for the task loss. The Fisher information matrix for the downstream task is `F = E[∇_θ L * ∇_θ L^T]`. For a frozen layer, the input representation covariance `E[x_i * x_j]` captures how the model's representation couples dimensions for the task. This is the principled justification — the activation covariance structure during fine-tuning IS the task-relevant geometric structure that rotation should model.

**Memory cost**:
- `g_cov_ema`: `(d, d)` float32 = 4 MB for d=1024, 67 MB for d=4096
- For LLaMA-7B (d=4096, ~32 JORA layers): **~2.1 GB** during calibration only
- Memory reduction option: accumulate in float16 (1 GB), cast to float32 only for the final pairing score computation

**Per-step FLOPs**:
- `(B*L, d) @ (B*L, d)^T` = `B*L * d²` FLOPs
- For d=1024, B*L=512: ~538M FLOPs per calibration step per layer
- For 500 calibration steps: ~269B FLOPs total (OPT-350m, one GPU)
- For comparison, one forward pass of OPT-350m is ~100B FLOPs. Calibration is ~2.7× one forward pass in FLOPs, but only during the calibration phase (500 steps).

**Buffer registration** in `_JoraAdapterState.__init__()`:
```python
self.register_buffer(
    "g_cov_ema",
    torch.zeros((self.m, self.m), device=dev, dtype=torch.float32),
    persistent=False,  # Not needed after calibration; saves disk I/O
)
```

#### Step B: Pairing Score (after calibration)

After `t_stat` calibration steps, compute the pairing score:

```python
# g_cov_ema[i,j] = E[x_i * x_j]
# grad_col_ema[i] = E[x_i²]
# Score: |cov| * geometric_mean_of_energy
score_matrix = g_cov_ema.abs() * torch.sqrt(
    grad_col_ema.unsqueeze(1) * grad_col_ema.unsqueeze(0) + eps
)  # (d, d)

# Pair score for (i,j) is score_matrix[i,j]
```

Note: `score_matrix[i,j] = |E[x_i x_j]| * sqrt(E[x_i²] * E[x_j²])` is exactly `|E[x_i x_j]|` normalized by the product of standard deviations — the absolute correlation coefficient. The score is bounded by `[0, 1]` per pair, with 1 meaning perfect correlation.

#### Step C: Greedy Disjoint Pair Selection

Reuse `select_top_k_pairs_gpu` with a new scoring function:

```python
def select_coupling_pairs_gpu(
    coupling_score: Tensor,  # (d, d) score matrix from Step B
    k: int,
    max_features: int,
) -> Tensor:
    """Greedy disjoint pair selection using coupling score matrix.

    Iterates all i<j pairs sorted by coupling_score[i,j] descending,
    greedily selecting pairs whose indices are not yet used.
    """
    # Same greedy algorithm as select_top_k_pairs_gpu,
    # but using coupling_score[i,j] instead of energy[i]*energy[j]
    # for the pair ordering.
```

Extract coupling subspace S = union of all indices in selected pairs.

#### Step D: One-Sided Restriction

`single_sided="right"` (existing config flag at `layer.py:457-458`):
- Skips `R_L` application entirely (counter=0, theta=None)
- `R_R` operates on coupling subspace S
- Zero runtime overhead during main training (no new buffers, no new FLOPs)

#### Step E: Memory Cleanup

After calibration completes and before main training:
```python
def disable_cov_ema(self):
    """Disable calibration-only buffers after pair freezing."""
    self.g_cov_ema = None  # GC reclaims ~2 GB for d=4096 models
```

### Relationship to Existing Code

| File | Change |
|---|---|
| `layer.py` | Extend forward hook: add activation outer product accumulation to `g_cov_ema` during calibration. Add `disable_cov_ema()` method. Add `calibration_active` guard on `g_cov_ema` update. Register `g_cov_ema` buffer `(d, d)`. |
| `selection.py` | Add `select_coupling_pairs_gpu()` — same greedy disjoint algorithm, uses `(d, d)` coupling score matrix. Add `pairing_strategy="coupling"` to config. |
| `config.py` | Add `pairing_strategy: Literal["consecutive", "high_low", "coupling"] = "consecutive"`. Add `subspace_size_ratio: float = 0.25` (optional, for tuning subspace size). |
| `callbacks.py` | Add subspace diagnostics: \|S\| per layer, top coupling pairs, score distribution. |
| `core.py`, `rotation.py`, `model.py` | No changes. |
| `compute_delta()` | No changes. |

### Training Plan

```
cfg = JoraConfig(
    core="diag",
    S_L=0,           # left rotation OFF
    S_R=32,          # right rotation ON
    single_sided="right",
    t_stat=500,
    pairs_freeze_after_warmup=True,
    pairing_strategy="coupling",
)

Steps 0–500 (calibration):
    - Forward hook: accumulate g_cov_ema = x_flat^T @ x_flat (EMA)
    - Backward hook: update grad_row_ema (existing)
    - update_step(): re-select pairs each step using current g_cov_ema
    - After step 500: disable_cov_ema()

Steps 501+ (main training):
    - DiagCore trains
    - theta_R trains (coupling subspace frozen, angles learnable)
    - Zero new buffers, zero new FLOPs per step
```

### Failure Modes and Diagnostics

| Failure Mode | Detection | Fallback |
|---|---|---|
| Rotation still redundant | TC-CS-1S ≈ NoRot | Stop Rule A: drop rotation from main story |
| Coupling signal not discriminative | S ≈ importance-ranked S, NoRot matches | Appendix: negative result on coupling hypothesis |
| Runtime > 2× NoRot during calibration | Calibration step time too high | Note: main training runtime is unchanged |
| TC-CS-1S > current JORA-Diag but ≈ NoRot | Subspace restriction helps, not rotation | Appendix: subspace restriction insight |

### Novelty and Elegance Argument

Closest prior work: current JORA-Diag (global sparse rotation with importance-based pairing).

Exact difference: Replace `energy[i] * energy[j]` pairing with `|E[x_i x_j]| * sqrt(E[x_i²] * E[x_j²])` pairing, where `E[x_i x_j]` is the activation cross-moment accumulated during calibration.

This is a **task-geometry-level** change. The rotation is restricted to dimensions whose activation covariance structure (as observed during task fine-tuning) indicates they need to move together. This is grounded in the actual geometric structure of the model's representations during adaptation, not in a hand-picked importance metric.

### Claim-Driven Validation

**Claim 1**: TC-CS-1S rotation provides benefit beyond diagonal-only scaling.
- 3-way matched comparison: NoRot / current JORA-Diag / TC-CS-1S
- Expected: TC-CS-1S beats both if rotation revival succeeds

**Claim 2**: The benefit comes from coupling-aware restriction.
- Ablation: coupling-derived S vs random S (same size), both one-sided
- Expected: coupling-derived S outperforms random S

**Deletion check**: TC-CS-1S vs NoRot directly.

### Experiment Handoff

- Must-prove claims: Coupling subspace one-sided rotation outperforms NoRot.
- Must-run ablations: NoRot / current JORA-Diag / TC-CS-1S / coupling-random.
- Critical metrics: Task quality, calibration step time, main training step time, theta norm/grad, S size per layer.
- Highest-risk assumption: The activation covariance structure during calibration reflects the actual task-relevant coupling geometry.

### CoupledPairCore Relationship (footnote)

If Phase 1 succeeds, the natural extension is to replace DiagCore with CoupledPairCore for within-pair 2×2 coupling blocks. This is additive in coupling strength (subspace restriction + within-pair blocks), not a mechanism swap. Phase 2 only if Stop Rule C passes.
