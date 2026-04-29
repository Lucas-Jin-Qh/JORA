# Research Proposal: TC-CS JORA

*Final refined proposal after 2-round review + coupling regime paragraph. Close to READY.*

*Coupling regime paragraph added: distinguishes attention layers (primary target, cross-head mixing → off-diagonal covariance) from FFN layers (near-diagonal covariance → diagonal scaling sufficient). This naturalizes the layer-type ablation design.*

---

## Problem Anchor

- **Bottom-line problem**: Current additive JORA-Diag cannot justify rotation's role — matched 1ep and 3ep ON/OFF comparisons show near-zero quality difference while ON costs ~3× runtime. Rotation has no irreplaceable function.

- **Must-solve bottleneck**: The current rotation mechanism is assigned no unique task; it acts as a global sparse basis reparameterization that diagonal scaling can absorb. Without a coupling-specific role, rotation cannot justify its computational cost.

- **Non-goals**: Not claiming "rotation drives the gain" without evidence. Not building a broad PEFT framework. Not rewriting into residualized full-support JORA. Not proving that arbitrary sparse rotation helps by default.

- **Constraints**: Must stay compatible with additive JORA-Diag code path. Compact enough for EMNLP narrative. Runtime cost must be justified.

- **Success condition**: TC-CS-1S shows a clear matched advantage over NoRot in a coupling-required regime, explainable as a necessary mechanism.

---

## Technical Gap

Current JORA selects dimensions for **importance** (output gradient EMA, input activation EMA), not for **coupling** (which dimensions need to move together). As a result:

- Rotation is applied globally — every selected pair gets rotation, regardless of whether those dimensions actually need coordinated movement.
- The diagonal core already provides sufficient per-dimension recalibration, so rotation's weak global coupling is redundant.
- Pairing uses `energy[i] * energy[j]` — selects high-high energy pairs, which are likely redundant rather than complementary.

**Operational gap**: Current rotation selection identifies "which dimensions are important" but misses "which dimension pairs need to be coupled."

### Coupling Regime

We expect rotation-based coupling to be most useful in attention projection layers, especially Q/K/V and output projections, where language adaptation often induces structured cross-dimensional interactions across head-related feature channels. In these layers, diagonal-only scaling may be insufficient because the useful update can involve coordinated redistribution among correlated dimensions. In contrast, FFN layers may exhibit more nearly diagonal activation covariance after nonlinear gating, making dimension-wise scaling a stronger baseline there. This suggests that rotation should not be applied uniformly across all linear layers; instead, it should be targeted toward layers where empirical activation covariance indicates non-diagonal coupling structure.

This layer-type conditioning naturally motivates the ablation design: attention-only TC-CS vs FFN-only TC-CS vs all-linear TC-CS vs NoRot, which together determine whether the coupling hypothesis is layer-type-specific or universal.

---

## Method Thesis

**Restrict JORA rotation to a task-conditioned coupling subspace, and introduce that restriction on one side first, so rotation models cross-dimension coupling only where diagonal-only language adaptation is insufficient.**

---

## Contribution Focus

- **Dominant contribution**: Activation-covariance-driven coupling subspace one-sided rotation for additive JORA-Diag.
- **Explicit non-contributions**: Not a general orthogonal adaptation framework. Not a claim that arbitrary sparse rotation helps. Not a broad PEFT-family comparison.

---

## Proposed Method

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
Current JORA-Diag:
    x → [R_R (global sparse, energy[i]*energy[j] pairing)] → [DiagCore] → [R_L^T] → Δ(x)

Proposed TC-CS-1S:
    Config: single_sided="right", core="diag"
    x → [R_R^(S) (subspace-restricted, activation-covariance pairing)] → [DiagCore] → [R_L=I] → Δ(x)
```

### Core Mechanism

#### Step A: Activation Outer Product EMA (forward pass, calibration only)

During `t_stat` calibration steps, accumulate the empirical activation cross-moment in the forward pass:

```python
# In layer.py forward hook (reuse existing ema_update_interval gating):
if st.cfg.pairing_strategy == "coupling" and st.cfg.calibration_active:
    x_flat = x.detach().float().reshape(-1, st.m)  # (B*L, d)
    x_cov = x_flat.T @ x_flat / max(x_flat.size(0), 1)  # (d, d) outer product
    st.g_cov_ema.lerp_(x_cov, 1.0 - beta)
```

- Task-conditioned: forward pass only fires on training samples → `g_cov_ema[i,j] = E_calibration[x_i * x_j]` reflects task-relevant representations.
- Memory: `(d, d)` float32 = 4 MB/d=1024, 67 MB/d=4096. ~2.1 GB for LLaMA-7B (~32 layers) during calibration only.
- Fisher connection: `g_cov_ema[i,j] = E[x_i * x_j]` is the empirical Fisher information for the task loss. The activation covariance structure during fine-tuning is the task-relevant geometric structure that rotation should model.

**Coupling regime (key intuition)**: Rotation-based coupling is expected to matter most in attention projection layers (Q, K, V, output), where cross-head dimension mixing creates structured off-diagonal patterns in the activation covariance matrix. In FFN layers, the activation covariance is approximately diagonal (per-neuron independence), so diagonal scaling is sufficient. This is why TC-CS focuses on identifying which dimensions in which layers have non-trivial off-diagonal covariance — and restricting rotation to those subspaces.

#### Step B: Pairing Score (after calibration)

```python
score_matrix = g_cov_ema.abs() * torch.sqrt(
    grad_col_ema.unsqueeze(1) * grad_col_ema.unsqueeze(0) + eps
)  # (d, d)
```

This equals `|E[x_i x_j]| * sqrt(E[x_i²] * E[x_j²])` — the unnormalized activation cross-covariance, bounded [0,1] for diagonal entries. Higher score = stronger coupling + both dimensions active.

#### Step C: Greedy Disjoint Pair Selection

`select_coupling_pairs_gpu(coupling_score, k, max_features)` — same greedy disjoint pair algorithm as `select_top_k_pairs_gpu`, but using the `(d, d)` coupling score matrix instead of `energy[i]*energy[j]`.

Extract coupling subspace S = union of all indices in selected pairs.

#### Step D: One-Sided Restriction

`single_sided="right"` (existing config flag): skips `R_L`. `R_R` operates on S. Zero new FLOPs per main training step.

#### Step E: Memory Cleanup

After calibration: `disable_cov_ema()` sets `g_cov_ema = None`, GC reclaims memory.

### Relationship to Existing Code

| File | Change |
|---|---|
| `layer.py` | Extend forward hook: add activation outer product accumulation. Add `disable_cov_ema()`. Register `g_cov_ema` `(d, d)` buffer. |
| `selection.py` | Add `select_coupling_pairs_gpu()`. Add `pairing_strategy="coupling"` config. |
| `config.py` | Add `pairing_strategy: Literal["consecutive", "high_low", "coupling"] = "consecutive"`. |
| `callbacks.py` | Add subspace diagnostics: \|S\| per layer, top coupling pairs, score distribution. |
| `core.py`, `rotation.py`, `model.py`, `compute_delta()` | No changes. |

### Training Plan

```
cfg = JoraConfig(
    core="diag", S_L=0, S_R=32,
    single_sided="right",
    t_stat=500, pairs_freeze_after_warmup=True,
    pairing_strategy="coupling",
)

Steps 0–500 (calibration):
    - Forward: accumulate g_cov_ema = x_flat^T @ x_flat (EMA)
    - update_step(): re-select pairs each step
    - After step 500: disable_cov_ema()

Steps 501+ (main training):
    - DiagCore + theta_R train (S frozen, angles learnable)
    - Zero new buffers, zero new FLOPs per step
```

### Failure Modes and Stop Rules

- **Stop Rule A**: TC-CS-1S ≈ NoRot → drop rotation from main story entirely.
- **Stop Rule B**: TC-CS-1S > current JORA-Diag but ≈ NoRot → appendix only (subspace restriction insight, not rotation revival).
- **Stop Rule C**: TC-CS-1S > NoRot, margin stable, runtime < 2× NoRot → allow Phase 2 bilateral expansion and broader baselines.

---

## Claim-Driven Validation

### Claim 1: TC-CS-1S rotation provides benefit beyond diagonal-only scaling
- 3-way matched comparison: NoRot / current JORA-Diag / TC-CS-1S
- Expected: TC-CS-1S beats both if rotation revival succeeds

### Claim 2: Benefit comes from coupling-aware restriction, not extra parameters
- Ablation: coupling-derived S vs random S (same size), both one-sided
- Expected: coupling-derived S outperforms random S

### Deletion check: TC-CS-1S vs NoRot directly.

---

## Experiment Handoff

- Must-prove claims: Coupling subspace one-sided rotation outperforms NoRot.
- Must-run ablations: NoRot / current JORA-Diag / TC-CS-1S / coupling-random.
- Critical metrics: Task quality, calibration step time, main training step time, theta norm/grad, S size per layer, g_cov_ema off-diagonal spectrum.
- Highest-risk assumption: Activation covariance during calibration reflects task-relevant coupling geometry.

---

## Compute & Timeline

- **GPU-hours**: Moderate. TC-CS-1S vs NoRot vs current JORA-Diag matched comparison (~3 runs × 3 epochs on OPT-350m). Calibration adds ~2.7× forward-pass FLOPs during first 500 steps.
- **Data cost**: None beyond existing pipelines.
- **Timeline**:
  1. Implement `g_cov_ema` buffer + forward outer product accumulation
  2. Implement `select_coupling_pairs_gpu()`
  3. Add `pairing_strategy="coupling"` config
  4. Run matched sanity comparison
  5. Only expand if Stop Rule C passes

---

## CoupledPairCore Footnote

If Phase 1 succeeds, natural extension: replace DiagCore with `CoupledPairCore` for within-pair 2×2 coupling blocks. Additive in coupling strength (subspace restriction + within-pair blocks), not a mechanism swap. Phase 2 only if Stop Rule C passes.
