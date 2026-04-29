# Round 1 Refinement

## Problem Anchor (verbatim from Round 0)

- **Bottom-line problem**: Current additive JORA-Diag cannot justify rotation's role — matched 1ep and 3ep ON/OFF comparisons show near-zero quality difference while ON costs ~3× runtime. Rotation has no irreplaceable function.
- **Must-solve bottleneck**: The current rotation mechanism is assigned no unique task; it acts as a global sparse basis reparameterization that diagonal scaling can absorb. Without a coupling-specific role, rotation cannot justify its computational cost.
- **Non-goals**: Not claiming "rotation drives the gain" without evidence. Not building a broad PEFT framework. Not rewriting into residualized full-support JORA in this proposal. Not proving that arbitrary sparse rotation helps by default.
- **Constraints**: Must stay compatible with current additive JORA-Diag code path. Must remain compact enough for EMNLP narrative. Runtime cost must be justified. Story must sound like language adaptation geometry.
- **Success condition**: A modified rotation design shows a clear, matched advantage over NoRot in a regime that plausibly requires cross-dimension coupling, while remaining explainable as a necessary mechanism rather than a tuning artifact.

---

## Anchor Check

- **Original bottleneck**: Rotation is applied globally without a coupling-specific function, so diagonal scaling absorbs its effect.
- **Why the revised method still addresses it**: TC-CS-1S restricts rotation to a coupling-derived subspace on one side only, giving rotation a unique geometric role that diagonal-only scaling cannot perform.
- **Reviewer suggestions that would cause drift if followed blindly**: The reviewer's suggestion to use a probe-based or Fisher-based approach would introduce a training phase outside the existing calibration pipeline. The coupling proxy should use existing EMA infrastructure to stay within the current code path boundaries. The reviewer's suggestion to address `CoupledPairCore` as a Phase 1 concern is rejected — CoupledPairCore is a separate core type with 2×2 blocks and is Phase 2+ only.

---

## Simplicity Check

- **Dominant contribution after revision**: Task-conditioned coupling-subspace one-sided rotation for additive JORA-Diag.
- **Components removed or merged**: Deleted the "optional supporting contribution" (diagnostic framework) from the contribution list — it was vague and competes with the main mechanism. Moved diagnostic logging to the implementation detail section. Dropped the "Stage 1/Stage 2" terminology in favor of reusing existing `t_stat` + `pairs_freeze_after_warmup` infrastructure.
- **Reviewer suggestions rejected as unnecessary complexity**: Probe-based subspace identification would require a separate training loop outside calibration. Fisher-based metrics would require Hessian approximations that are impractical for a v1. These are Phase 2+ options.
- **Why the remaining mechanism is still the smallest adequate route**: Only the pairing scoring function changes. No new trainable parameters. No new buffers beyond the cross-covariance EMA. No new training phases. Uses existing `t_stat` calibration, existing `pairs_freeze_after_warmup`, and existing `single_sided="right"` flag.

---

## Changes Made

### 1. Concretized the coupling proxy with existing infrastructure

- **Reviewer said**: The coupling proxy (`c_ij = |corr(a_i, a_j)|`) is ambiguous. O(d²) memory for full correlation matrix is not "minimal." The coupling signal should be task-conditioned.
- **Action**: Specified concrete implementation using the existing `grad_col_ema` buffer (which stores per-dimension activation magnitude EMAs) plus a new cross-covariance EMA buffer. The correlation proxy is:
  ```
  c_ij = |E[a_i * a_j]| / sqrt(E[a_i²] * E[a_j²])
  ```
  where `E[a_i²]` is already stored in `grad_col_ema` and `E[a_i * a_j]` is accumulated in a new `cov_ij_ema` buffer during calibration.
- **Reasoning**: Reuses existing EMA infrastructure. The signal is naturally task-conditioned because activations are collected only during training (backward-pass hook on training steps). The new buffer is O(d²) per layer — for OPT-350m (d=1024), ~524K float32 per layer, which is acceptable during calibration only.
- **Impact on core method**: The pairing score becomes:
  ```
  score_coupling(i,j) = c_ij * min(g_i, g_j)
  ```
  where `g_i` comes from `grad_col_ema`. This replaces `energy[i] * energy[j]`.

### 2. Explicitly connected to existing config flags and code paths

- **Reviewer said**: `single_sided` flag already exists in config.py. `t_stat` calibration already exists. The proposal implies new code paths that are unnecessary.
- **Action**: Added explicit references to `single_sided="right"` (for R_L=I), `t_stat` calibration, and `pairs_freeze_after_warmup=True`. Specified that no changes to `compute_delta()` are needed.
- **Reasoning**: Existing infrastructure covers the architectural changes. The only new code is in `selection.py` (new pairing scoring function) and `layer.py` (new `cov_ij_ema` buffer).
- **Impact on core method**: Zero. `compute_delta()` path is unchanged.

### 3. Added CoupledPairCore relationship clarification

- **Reviewer said**: The codebase already has `CoupledPairCore` (core.py:96-266) with 2×2 coupled blocks on rotation pairs. The proposal doesn't discuss its relationship to TC-CS.
- **Action**: Added explicit relationship note: TC-CS-1S with DiagCore and coupling-subspace restriction is Phase 1. If TC-CS-1S succeeds, TC-CS-2S (bilateral) with CoupledPairCore is the natural Phase 2 extension — coupling subspace restriction + within-pair 2×2 blocks gives a two-level coupling story.
- **Reasoning**: This clarifies the roadmap without adding complexity to Phase 1. The relationship is additive (more coupling structure, not a different mechanism).
- **Impact on core method**: None for Phase 1. Enables cleaner Phase 2 narrative.

### 4. Dropped the "diagnostic framework" supporting contribution

- **Reviewer said**: The optional supporting contribution (diagnostic framework) is vague and competes with the main mechanism.
- **Action**: Demoted to implementation detail in `callbacks.py`. It is not a paper contribution.
- **Reasoning**: Reduces perceived contribution sprawl. Diagnostics belong in the implementation, not the contribution list.
- **Impact on core method**: None.

### 5. Added runtime threshold to stop rules

- **Reviewer said**: "Runtime overhead" is mentioned as a failure condition but no threshold is specified.
- **Action**: Added explicit threshold: TC-CS-1S runtime must be within 2× NoRot to be considered "justifiable." Current JORA-Diag is ~3× NoRot.
- **Reasoning**: Gives Stop Rule A and C concrete decision boundaries.
- **Impact on core method**: None.

---

## Revised Proposal

# Research Proposal: TC-CS JORA

*Round 1 Refinement — addresses coupling proxy specificity gap and frontier leverage*

---

### Problem Anchor

Same as Round 0.

### Method Thesis

**Restrict JORA rotation to a task-conditioned coupling subspace, and introduce that restriction on one side first, so rotation models cross-dimension coupling only where diagonal-only language adaptation is insufficient.**

### Complexity Budget

| Category | Status |
|----------|--------|
| Frozen backbone | reused |
| DiagCore | reused as-is |
| Givens rotation parameterization | reused as-is |
| `compute_delta()` | unchanged |
| `single_sided="right"` | reused (R_L=I) |
| `t_stat` calibration | reused |
| `pairs_freeze_after_warmup=True` | reused |
| Pairing scoring function | **changed** (energy product → coupling correlation × min gradient) |
| New buffer: `cov_ij_ema` | **added** (O(d²) per layer, calibration only) |
| `CoupledPairCore` relationship | clarified as Phase 2 extension |
| Bilateral expansion | Phase 2 only |
| New trainable components | none |

### System Overview

```
Current JORA-Diag (S_L=32, S_R=32, all dims):
    x → [R_R (global sparse, energy[i]*energy[j] pairing)] → [DiagCore] → [R_L^T (global sparse)] → Δ(x)

Proposed TC-CS-1S:
    Config: single_sided="right", core="diag"
    x → [R_R^(S) (subspace-restricted, coupling-aware pairing)] → [DiagCore] → [R_L=I] → Δ(x)
    where S ⊂ {1,...,d}, |S| = m, R_R^(S) only pairs dimensions inside S.

Calibration: reuse t_stat steps, accumulate cov_ij_ema.
Train: freeze S, train DiagCore + one-sided theta_R.
```

### Core Mechanism

#### Coupling Relevance Score (concrete implementation)

**Step A: Existing EMA infrastructure**

The existing backward hook in `layer.py:644-692` updates two per-dimension EMA buffers per layer during training:

- `grad_col_ema[i] = E[a_i²]` — per-dimension activation second moment (input side)
- `grad_row_ema[j] = E[g_j²]` — per-dimension gradient second moment (output side)

**Step B: New cross-covariance EMA buffer (O(d²), calibration only)**

During `t_stat` calibration steps, accumulate:
```
cov_ij_ema[i,j] = E[a_i * a_j]   # for i ≤ j, symmetric
```
Update via EMA: `cov_ij_ema.lerp_(a_i * a_j, 1.0 - beta)` for each calibration batch.

Memory cost per layer: for d=1024, this is ~524K float32 entries (~2 MB per layer). For ~24 layers: ~48 MB. Acceptable during calibration only.

**Step C: Correlation proxy computation**

After calibration, compute per-dimension standard deviations from existing `grad_col_ema`:
```
std_i = sqrt(grad_col_ema[i] + eps)
```

Compute coupling correlation for each pair (i,j):
```
c_ij = |cov_ij_ema[i,j]| / (std_i * std_j)
```
This is the normalized cross-covariance — bounded in [-1,1], zero for uncorrelated dimensions.

**Step D: Combined pairing score**

```
score_coupling(i,j) = c_ij * min(g_i, g_j)
```
where `g_i = sqrt(grad_col_ema[i])` is the per-dimension activation magnitude (task-relevance proxy).

Why this is different from current `energy[i] * energy[j]`:
- Current: selects high-high energy pairs (likely redundant)
- Proposed: selects pairs with strong cross-covariance AND moderate activation energy (complementary, not redundant)

Why it is task-conditioned:
- Activations are collected only during training (backward hook only fires on training steps)
- Cross-covariance reflects which dimensions move together during task adaptation

**Step E: Subspace and pair selection**

1. Compute `score_coupling(i,j)` for all i<j.
2. Take top-K pairs by `score_coupling` with greedy disjoint enforcement (reuse existing `select_top_k_pairs_gpu` greedy logic, replacing `energy[i]*energy[j]` with `score_coupling(i,j)`).
3. Extract the coupling subspace S = union of all indices in top-K pairs.
4. Freeze S. During training, `R_R` only pairs dimensions inside S. `R_L = I` (via `single_sided="right"`).

#### Relationship to Existing Code

| File | Change |
|------|--------|
| `layer.py` | Add `cov_ij_ema` buffer (O(d²), calibration only). Update it in backward hook during `t_stat` steps. After calibration: compute `c_ij`, freeze pairs, disable buffer. |
| `selection.py` | Add `select_coupling_pairs_gpu()` function. Accepts `coupling_score` tensor (precomputed per pair) + `grad_col_ema` (per-dimension magnitude). Replaces `select_top_k_pairs_gpu` when `pairing_strategy="coupling"`. |
| `config.py` | Add `pairing_strategy: Literal["consecutive", "high_low", "coupling"] = "consecutive"`. Add `subspace_size_ratio: float = 0.25` (fraction of d for S size). |
| `callbacks.py` | Add subspace diagnostics: \|S\| per layer, c_ij distribution, top-10 coupling pairs. |
| `core.py`, `rotation.py`, `model.py` | No changes. |

**No changes to `compute_delta()`** — the one-sided restriction is handled entirely by `single_sided="right"` config flag (already implemented at `layer.py:457-458`).

#### Training Plan (concrete, reuses existing infrastructure)

```
Before training:
    cfg = JoraConfig(
        core="diag",
        S_L=0,          # left rotation OFF
        S_R=32,         # right rotation ON
        single_sided="right",  # R_L=I
        t_stat=500,     # calibration steps
        pairs_freeze_after_warmup=True,
        pairing_strategy="coupling",
        subspace_size_ratio=0.25,  # S = top 25% of dim by coupling score
    )

Calibration (first t_stat steps):
    - Backward hook updates grad_col_ema (existing) and cov_ij_ema (new)
    - update_step() re-selects pairs each step with coupling-aware scoring
    - After t_stat steps: pairs frozen, cov_ij_ema can be freed/disabled

Main training (remaining steps):
    - DiagCore trains normally
    - theta_R trains normally (coupling subspace frozen, angles learnable)
    - Single-sided rotation = localized coupling in task-conditioned S
```

#### Relationship to CoupledPairCore (Phase 2)

`CoupledPairCore` (core.py:96-266) applies 2×2 blocks on rotation pairs:
```
[y_i]   [1+δ_ii  δ_ij] [x_i]
[y_j] = [δ_ji   1+δ_jj] [x_j]
```

This is a Phase 2 extension:
- TC-CS-1S (Phase 1): coupling subspace restriction + DiagCore → within-subspace diagonal scaling
- TC-CS-2S (Phase 2): coupling subspace restriction + CoupledPairCore → within-subspace 2×2 coupling blocks

The Phase 1→2 progression is additive in coupling strength, not a mechanism swap. If Phase 1 is inconclusive, Phase 2 is not attempted.

### Novelty and Elegance Argument

Closest prior work: current JORA-Diag (global sparse rotation with importance-based pairing).

Exact difference: Replace `energy[i] * energy[j]` pairing with `c_ij * min(g_i, g_j)` pairing, where `c_ij` is the normalized activation cross-covariance. Restrict rotation to one side and coupling subspace.

The claim becomes falsifiable: if restricting rotation to a coupling-derived subspace on one side does not beat NoRot, the hypothesis is wrong. If it does, the coupling story is grounded in a specific geometric signal (cross-covariance) rather than magnitude.

### Failure Modes and Diagnostics (updated)

| Failure Mode | Detection | Fallback |
|---|---|---|
| Rotation still redundant | TC-CS-1S ≈ NoRot in quality | Stop Rule A: drop rotation from main story |
| Coupling proxy not discriminative | S is importance-heavy, NoRot matches | Revise coupling proxy (see Phase 2 options), not architecture |
| Runtime > 2× NoRot | Step time > 2× NoRot | Stop Rule C: runtime unjustifiable |
| TC-CS-1S > current JORA-Diag but ≈ NoRot | Subspace helps, rotation is not the mechanism | Appendix only — subspace restriction insight, not rotation revival |

**Phase 2 options if coupling proxy needs revision**:
- Use `g_ij = |grad_col_ema[i] * grad_col_ema[j]|` as a task-gradient-based proxy (O(d) memory, less precise but no new buffers)
- Use Fisher approximation: `F_ij = |∂L/∂h_i * ∂L/∂h_j|` (gradient outer product, task-conditioned naturally)

---

### Claim-Driven Validation Sketch (unchanged)

**Claim 1**: TC-CS-1S rotation provides benefit beyond diagonal-only scaling.
- 3-way matched comparison: NoRot / current JORA-Diag / TC-CS-1S
- Expected: TC-CS-1S beats both if rotation revival succeeds

**Claim 2**: The benefit comes from coupling-aware restriction.
- Ablation: coupling-random subspace vs coupling-relevance subspace
- Expected: coupling-relevance outperforms random, confirming mechanism specificity

**Deletion check**: TC-CS-1S vs NoRot directly.

---

### Experiment Handoff Inputs (unchanged)

- Must-prove claims: Coupling subspace one-sided rotation outperforms NoRot.
- Must-run ablations: NoRot / current JORA-Diag / TC-CS-1S / coupling-random.
- Critical metrics: Task quality, runtime (must be < 2× NoRot), theta norm/grad diagnostics, S size per layer, c_ij distribution.
- Highest-risk assumption: A coupling-relevant subspace exists in language adaptation and can be identified via activation cross-covariance from calibration data.
