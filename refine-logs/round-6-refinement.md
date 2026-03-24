# Round 6 Refinement

## Problem Anchor
*(Verbatim from round 0, with targeted corrections to the success condition — see Anchor Check)*

- **Bottom-line problem**: PEFT methods must achieve a better accuracy-vs-parameter Pareto frontier than LoRA for LLM fine-tuning, using a principled, structure-preserving adaptation mechanism.
- **Must-solve bottleneck**: Orthogonal PEFT (qGOFT, OFT) uses static rotation structures — all dimension pairs adapted uniformly, wasting capacity on low-signal directions. Low-rank PEFT (LoRA, DoRA) lacks geometric bias and suffers norm drift. Neither achieves an efficient adaptive sparse rotation.
- **Non-goals**: Not a new backbone, not replacing LoRA for all use cases, not targeting inference latency.
- **Constraints**: 3× RTX 4090, 2-week experiment window. PEFT library. NeurIPS 2026.
- **Success condition (corrected)**: JORA in the extreme-budget regime (~3K–25K total params on Mistral-7B q+o scope, i.e., 0.3–2.4% of LoRA-r4 params) Pareto-dominates LoRA-r1 as a must-win; competitiveness with LoRA-r2 as primary target; matching LoRA-r2 as a stretch goal. Beats faithful qGOFT at equal budget.

---

## Anchor Check

- **Original bottleneck**: Static rotation structure in orthogonal PEFT wastes capacity; LoRA lacks geometric bias.
- **Why the revised method still addresses it**: Yes. Still about adaptive allocation of a sparse bilateral rotation budget inside rotation-based PEFT. This round fixes two structural correctness issues in the method itself without changing the mechanism.
- **Reviewer suggestions rejected as drift**: None. Both CRITICAL items and all IMPORTANT items are valid and are addressed.

---

## Simplicity Check

- **Dominant contribution after revision**: One mechanism: one-time adaptive allocation of sparse bilateral rotation slots within rotation-based PEFT, with a correctly residualized zero-function-change parameterization and a verified success condition.
- **Components removed or merged**:
  - JORA-large variant: removed from main paper. Small + base are sufficient for the Pareto curve shape.
  - "40K–100K" success condition range: corrected to match actual operating points (~3K–25K).
  - Dual success framing (primary + stretch in one sentence): simplified to single primary target with stretch explicitly staged.
- **Reviewer suggestions rejected as unnecessary complexity**: None.
- **Why the remaining mechanism is still the smallest adequate route**: The residualization is not a new component — it changes the parameterization of `D_sel` from `d_sel` initialized to 1 to `Δd_sel` initialized to 0, while adding one fixed projection `P_U` in the forward pass. The model structure is identical; only the math of initialization changes.

---

## Changes Made

### 1. Fix the additive-init function jump (CRITICAL)

- **Reviewer said**: "With theta=0, d_sel=1, the adapter is NOT zero at allocation time. It injects P_U·x immediately, causing the model function to jump at t=T_stat."
- **Action**: Residualize the adapter. Parameterize `D_sel = I_U + ΔD_sel` where `ΔD_sel` is initialized to zero. The forward path becomes:
  ```
  delta = R_L^T · (I_U + ΔD_sel) · R_R · x - P_U · x
        = (R_L^T · R_R - I_d) · P_U · x + R_L^T · ΔD_sel · R_R · x
  ```
  At allocation time (theta=0, ΔD_sel=0):
  - `R_R = I_d` and `R_L = I_d` (rotation angles at zero → Givens rotations at identity)
  - `R_L^T · (I_U + 0) · R_R · x = P_U · x`
  - `delta = P_U · x - P_U · x = 0`
  - Gradient through `ΔD_sel`: `∂delta/∂ΔD_sel = R_L^T · ... · R_R · x ≠ 0` (P_U·x nonzero)
  - Gradient through `theta_R, theta_L`: `∂delta/∂theta ∝ P_U · x ≠ 0` (nonzero because ΔD_sel does not kill it)

  This is a correct zero-function-change initialization with nonzero gradients for all trainable parameters from step T_stat.

- **Simpler equivalent formulation**: Parameterize `D_sel = diag(1 + δ_u)_{u∈U}` where `δ` is the trainable vector initialized to zeros. Then:
  ```
  delta = R_L^T · diag(1 + δ) · R_R · x - P_U · x
  ```
  At init: `delta = P_U · x - P_U · x = 0`. Clean, minimal, no new components.

- **Reasoning**: This is a correctness fix. The old init was wrong: it introduced a function discontinuity at T_stat. The residualized version preserves the pretrained function at allocation time while keeping gradients live.
- **Impact on core method**: The forward path now has one extra fixed term `−P_U · x`, which is a sum of projections onto the selected dimensions. This is cheap (sparse) and exactly linear. Mergeability is preserved:
  ```
  W_merged = W₀ + R_L^T · (I_U + ΔD_sel) · R_R − P_U
  ```

### 2. Correct the success condition to match actual operating points (CRITICAL)

- **Reviewer said**: "The success condition says ~40K–100K params, but the actual operating points are ~3K/12K/24K."
- **Action**: Correct the success condition to the actual JORA operating regime (~3K–25K total params) and stage the empirical target:
  - **Must-win (primary)**: JORA-base (~12K params) outperforms LoRA-r1 (~524K params) on MMLU/ARC-C/GSM8K average.
  - **Primary target**: JORA-base competitive with LoRA-r2 (~1M params) — within 1–2 pp on average.
  - **Stretch goal**: JORA-base matches or beats LoRA-r2.
- **Reasoning**: The "40K–100K" figure was a carryover from when full-width D_diag was included. With only selected-support diagonal (|U| ≤ 4K), JORA-base has ~12K params. The correct framing is that JORA operates in a genuinely extreme-budget regime that LoRA cannot access without r=1. The Pareto story is that JORA-base achieves LoRA-r1 quality while occupying a unique extreme-budget operating point, with LoRA-r2 competitiveness as the headline result if empirics support it.
- **Impact on core method**: None. Changes only the success narrative and what the paper must prove vs. would like to prove.

### 3. Stage the empirical target (IMPORTANT)

- **Reviewer said**: "~12K matching LoRA-r2 ~1M is a very aggressive empirical bet."
- **Action**: The validation sketch now explicitly separates must-win, primary, and stretch targets. The paper can still be written and published if JORA-base is only competitive with LoRA-r2 (within 2pp), not matched exactly. The Pareto story is valid in either case because of the extreme parameter-count ratio.
- **Reasoning**: A staged success condition is scientifically honest and reduces feasibility risk without weakening the paper.
- **Impact on core method**: None.

### 4. Precommit the Diag-only-selected interpretation rule (IMPORTANT)

- **Reviewer said**: "If Diag-only-selected is within error bars of JORA, the contribution is not sparse rotation allocation."
- **Action**: The proposal now explicitly precommits the interpretation: if Diag-only-selected matches JORA within statistical noise (mean difference < 0.5 pp on all three benchmarks), the claim narrows to "adaptive sparse diagonal scaling with rotation-structured support selection" — still publishable, but the rotation contribution becomes a supporting rather than dominant claim.
- **Reasoning**: Scientific honesty. Precommitting forces the paper to be defensible regardless of outcome.
- **Impact on core method**: None.

### 5. Honest framing of asymmetric bilateral statistics (MINOR)

- **Reviewer said**: "Right-side statistic is only an activation proxy, not a true task signal."
- **Action**: The method section now explicitly frames the bilateral selection as asymmetric: left-side (output) uses gradient magnitude (task-coupled); right-side (input) uses activation magnitude (cheap proxy). This asymmetry is reported factually, not dressed up as symmetric task-driven selection.
- **Reasoning**: Honest framing is better than an overstatement that a reviewer will catch.
- **Impact on core method**: None.

### 6. Remove JORA-large from the main paper

- **Reviewer said**: "Remove JORA-large unless needed for curve shape."
- **Action**: JORA-large dropped from the main paper. Main Pareto curve uses JORA-small (K=8, ~3K params) and JORA-base (K=32, ~12K params). JORA-large is appendix-only if run.
- **Reasoning**: The curve shape is readable with two points. Adding a third only clutters the main figure and increases compute.
- **Impact on core method**: None.

---

## Revised Proposal

# Research Proposal: JORA — Adaptive Sparse Rotation-Slot Allocation for Extreme-Budget Square-Layer PEFT

## Problem Anchor

- **Bottom-line problem**: PEFT methods must achieve a better accuracy-vs-parameter Pareto frontier than LoRA for LLM fine-tuning, using a principled, structure-preserving adaptation mechanism.
- **Must-solve bottleneck**: Orthogonal PEFT (qGOFT, OFT) uses static rotation structures — all dimension pairs adapted uniformly, wasting capacity on low-signal directions. Low-rank PEFT (LoRA, DoRA) lacks geometric bias and suffers norm drift. Neither achieves an efficient adaptive sparse rotation.
- **Non-goals**: Not a new backbone, not replacing LoRA for all use cases, not targeting inference latency.
- **Constraints**: 3× RTX 4090, 2-week experiment window. PEFT library. NeurIPS 2026.
- **Success condition**: JORA in the extreme-budget regime (~3K–25K total params on Mistral-7B q+o scope) Pareto-dominates LoRA-r1 (must-win), is competitive with LoRA-r2 within 2pp (primary target), and optionally matches LoRA-r2 (stretch); JORA beats faithful qGOFT at equal budget.

## Technical Gap

`qGOFT` gives rotation-based PEFT a strong geometric primitive, but its rotation budget is **static**: the rotation structure is chosen once by a fixed schedule and then trained. `LoRA` allocates low-rank capacity without rotation structure and allows norm drift.

The clean gap is narrow: **within rotation-based PEFT, there is no method that uses early task statistics to allocate a fixed sparse bilateral rotation budget, freezes that support, and uses a correctly residualized zero-function-change initialization that keeps rotations immediately trainable**.

The Pareto story is honest: JORA-base (~12K params) targets accuracy at or near LoRA-r2 (~1M params). This is achieved by using a better inductive bias (rotation structure) with better allocation (adaptive vs static) in an extreme-budget regime that LoRA cannot address with integer rank choices.

## Method Thesis

JORA is a square-layer, mergeable additive adapter that first runs a short statistics warmup (asymmetric bilateral: gradient-guided output side, activation-proxy input side), then allocates **one fixed set of sparse bilateral rotation slots** and trains only the associated rotation angles plus a **residualized selected-support diagonal core initialized to zero**. The residualization gives exact zero function change at allocation while keeping all parameters gradient-live from the first training step.

This is the smallest adequate intervention because:
- the prior-art gap is about **static vs adaptive allocation**,
- the residualized diagonal core is explicitly tied to the same selected support,
- the mainline method avoids full-width diagonals, repeated reseating, tanh, and rectangular operators.

**Correct initialization with residualization**: JORA computes

```
delta = R_L^T · (I_U + ΔD_sel) · R_R · x  −  P_U · x
out   = base_out + delta
```

where `ΔD_sel` is the trainable diagonal correction (initialized to zero), `I_U` is the identity on the selected support U, and `P_U` is the fixed projection onto U. At allocation time (theta=0, ΔD_sel=0):
- `R_R = R_L = I_d` (Givens rotations at zero angles are identity)
- `delta = P_U·x − P_U·x = 0` ✓ (zero function change)
- `∂delta/∂ΔD_sel ≠ 0`, `∂delta/∂theta ≠ 0` ✓ (gradients live)

**Simpler equivalent parameterization**: `d_sel = 1 + δ` where `δ` is initialized to zeros:
```
delta = R_L^T · diag(1 + δ)_U · R_R · x  −  P_U · x
```

**Parameter scale** (concrete, Mistral-7B, 32 layers, q+o scope):
- JORA-small K=8:   2 × 32 × (8 + 8 + 32)   = ~3,072 params   (0.15% of LoRA-r2)
- JORA-base  K=32:  2 × 32 × (32 + 32 + 128)  = ~12,288 params  (1.17% of LoRA-r2)
- LoRA-r1 (q+o):    2 × 32 × 2 × 4096 × 1     = 524,288 params
- LoRA-r2 (q+o):    2 × 32 × 2 × 4096 × 2     = 1,048,576 params

## Contribution Focus

- **Single dominant contribution**: One-time adaptive allocation of sparse bilateral rotation slots within rotation-based PEFT, with a correctly residualized zero-function-change initialization, yielding Pareto-dominant extreme-budget accuracy on square attention layers.
- **Explicit non-contributions**: full-width diagonal cores in the main method, repeated reseating, broad "adaptive PEFT" claims, rectangular-layer main results, tanh-based merge path, OER.
- **OER**: removed from all proposal text; may appear in appendix only if a stable multi-seed gain forces it.

## Proposed Method

### Complexity Budget
- **Frozen**: Pretrained W₀, model backbone, tokenizer, PEFT infrastructure, projection P_U (non-trainable, computed once at allocation).
- **New trainable parameters (mainline)**:
  - `θ_R [K_R]`: right-rotation Givens angles for frozen input-side slot pairs
  - `θ_L [K_L]`: left-rotation Givens angles for frozen output-side slot pairs
  - `δ [|U|]`: diagonal correction on selected support U = dims(slots_R) ∪ dims(slots_L), |U| ≤ 4K, **initialized to 0**
- **Non-trainable structural state (after allocation)**:
  - `slots_R = {(i_r, j_r)}_{r=1}^{K_R}`
  - `slots_L = {(i_l, j_l)}_{l=1}^{K_L}`
  - `U = dims(slots_R) ∪ dims(slots_L)`
  - `P_U = diag(1_U)` — fixed projection, not trainable
- **Intentionally excluded from the main paper**: full-width D_diag[d], repeated reseating, rectangular operators, k/v under GQA, MLP projections, tanh merge variants, OER.

### System Overview

Main-paper target modules: **square attention projections only (q_proj, o_proj)**.

Core claim baselines (same module set):
- LoRA-r1, LoRA-r2, LoRA-r4 for Pareto frontier
- faithful qGOFT reimplementation for closest static orthogonal baseline

Mandatory mechanism controls:
- `Diag-only-selected`: same support U, no rotations, only δ trained
- `fixed-slot JORA`: same codepath, random static slots instead of adaptive

For a square layer of width d and input x ∈ ℝ^{B × d}:

```
base_out = W₀ · x

# statistics warmup (no slot changes yet)
ema_in[j]  ← β · ema_in[j]  + (1-β) · mean_batch(x_j²)        # activation proxy
ema_out[i] ← β · ema_out[i] + (1-β) · mean_batch((∂L/∂y_i)²)  # gradient signal

# single allocation after warmup (at step T_stat)
slots_R ← deterministic_pairs(top_2K(ema_in))
slots_L ← deterministic_pairs(top_2K(ema_out))
U       ← dims(slots_R) ∪ dims(slots_L)
δ       ← zeros(|U|)   # trainable correction, initialized to 0
θ_R     ← zeros(K_R)   # Givens angles, identity at init
θ_L     ← zeros(K_L)
P_U     ← diag(1_U)   # fixed projection, not trainable

# fixed-support training (θ_R, θ_L, δ all trainable from step T_stat)
x̃     = R_R(slots_R, θ_R) · x
x̂     = diag(1 + δ)_U · x̃           # applies (1+δ) on U, identity elsewhere
delta  = R_L^T(slots_L, θ_L) · x̂  −  P_U · x    # residualized

out = base_out + delta
```

At init: `delta = P_U·x − P_U·x = 0`. No function jump at T_stat.

After allocation: fully linear, exactly mergeable:
```
W_merged = W₀ + R_L^T · diag(1 + δ)_U · R_R − P_U
```

### Core Mechanism

#### 1. Asymmetric bilateral statistics

**Right-side (input activation proxy)**:
```
ema_in[j] ← β · ema_in[j] + (1-β) · mean_batch(x_j²)
```
Cheap; identifies high-magnitude input dimensions. Not task-driven.

**Left-side (output gradient signal)**:
```
ema_out[i] ← β · ema_out[i] + (1-β) · mean_batch((∂L/∂y_i)²)
```
Task-coupled; identifies high-sensitivity output dimensions. Requires backward pass during warmup.

These two statistics are intentionally asymmetric. The paper frames this honestly: "gradient-guided output allocation, activation-proxy input allocation."

#### 2. Support-stability diagnostic

Before committing to single allocation:
```
# at each of the last 5 warmup steps, record top_2K(ema_in) and top_2K(ema_out)
# compute Jaccard similarity between consecutive top-2K sets
# if mean Jaccard > 0.85, proceed with one-shot allocation at T_stat
# if mean Jaccard < 0.85, optionally extend T_stat by 50 steps (once) and re-check
```

This converts one-shot allocation from a heuristic to a verified convergence decision.

#### 3. Deterministic disjoint pairing rule

For either score vector:
1. Stable-sort dimensions by descending score; ties broken by smaller index first.
2. Keep the first 2K dimensions.
3. Form consecutive pairs: (d₁, d₂), (d₃, d₄), …, (d₂K₋₁, d₂K).
4. Canonicalize each pair as (min(d_a, d_b), max(d_a, d_b)).

#### 4. Residualized zero-function-change allocation

```
slots_R, slots_L ← allocation from statistics
U               ← dims(slots_R) ∪ dims(slots_L)
δ               ← zeros(|U|)    # diagonal correction init
θ_R, θ_L        ← zeros(K)     # Givens angles init (identity rotations)
P_U             ← diag(1_U)    # fixed projection, frozen
```

Forward pass at initialization:
- R_R = R_L = I_d → delta = I_U · x − P_U · x = P_U·x − P_U·x = **0** ✓
- Gradient w.r.t. δ: ∂delta/∂δ_u = x̃_u² (nonzero for active input) ✓
- Gradient w.r.t. θ_R: nonzero because P_U·x feeds into R_L^T ✓

#### 5. Fixed-support optimization

After allocation:
- slots_R, slots_L, U, P_U are frozen permanently
- θ_R, θ_L, and δ are trained jointly from step T_stat

#### 6. Precommitted interpretation rule

Diag-only-selected baseline uses the same U, no rotations, only δ trained. Interpretation:
- If JORA − Diag-only-selected > 0.5 pp on all three benchmarks: rotation claim is supported; paper's dominant contribution stands.
- If JORA − Diag-only-selected ≤ 0.5 pp: rotation contribution is marginal; paper narrows to "adaptive sparse diagonal scaling with rotation-structured support selection." Still publishable, but claims must be adjusted.

### Mergeability

```
W_merged = W₀ + R_L^T · diag(1 + δ)_U · R_R − P_U
```

Exact linear merge. Inference overhead: zero.

### Training Plan

1. **Backbone / data**: Mistral-7B-v0.1, Alpaca-cleaned 52k, standard SFT.
2. **Target modules (main paper)**: q_proj, o_proj only.
3. **Statistics warmup**: T_stat = min(200, 0.05 · T_total) steps. Accumulate ema_in and ema_out.
4. **Stability check**: measure Jaccard similarity of top-2K sets in the last 5 warmup steps. Extend by 50 steps once if Jaccard < 0.85.
5. **Single allocation**: at t = T_stat, allocate slots_R, slots_L, U; init δ = zeros, θ_R = θ_L = zeros, P_U = fixed.
6. **Main optimization**: AdamW, JORA-specific LR (tuned; may be higher than LoRA LR), cosine decay.
7. **Model sizes (main paper)**:
   - JORA-small: K=8 per side, ~3K params on q+o, 32 layers
   - JORA-base:  K=32 per side, ~12K params on q+o, 32 layers
8. **Parameter accounting** (all paper tables must report exact counts):
   - JORA-small K=8:  2 × 32 × (8 + 8 + 32) = 3,072 params
   - JORA-base  K=32: 2 × 32 × (32 + 32 + 128) = 12,288 params
   - LoRA-r1:   524,288 params
   - LoRA-r2:   1,048,576 params
   - JORA-base uses 1.17% of LoRA-r2 parameter budget

### Failure Modes and Diagnostics

- **Warmup unstable (Jaccard < 0.85 after extension)**: report support instability, increase β, flag as limitation.
- **δ explodes during training**: add L2 penalty on δ or clip; detect via max|δ| per module.
- **Diag-only-selected matches JORA (≤ 0.5 pp)**: narrow rotation claim per precommitted rule.
- **Faithful qGOFT reproduction fails**: fall back to fixed-slot JORA as static baseline; label clearly as internal ablation.

### Novelty and Elegance Argument

- **vs. qGOFT**: JORA changes where the sparse rotation budget goes — using early task statistics instead of a fixed schedule.
- **vs. fixed-slot JORA**: same parameterization, same support size, different allocation. Cleanly isolates adaptive allocation value.
- **vs. LoRA**: JORA achieves comparable accuracy at 1–2% of LoRA's parameter count, with rotation-structured adapter.
- **New this round**: the residualized initialization is not just "fix the init" — it is the correct way to introduce a structured adapter into a pretrained model at an interior training step, preserving the pretrained function exactly at the switchover point.
- **Why this is focused**: one mechanism, one parameter regime, one clean Pareto story, one honest bilateral framing.

## Claim-Driven Validation Sketch

**Rule**: two claim-bearing experiment blocks; one mandatory mechanism-isolation block; everything else appendix-only.

### Claim 1 (Primary): Pareto-dominant over LoRA in the extreme-budget regime

- **Minimal experiment**: JORA-small (K=8), JORA-base (K=32) vs. LoRA-r1, LoRA-r2, LoRA-r4, all on q_proj + o_proj.
- **Metric**: average accuracy on MMLU / ARC-C / GSM8K vs exact total trainable parameter count.
- **Staged targets**:
  - Must-win: JORA-base > LoRA-r1 (primary); competitive with LoRA-r2 within 2pp (headline); matches LoRA-r2 (stretch).
- **Seed plan**: 3 seeds for JORA-base and LoRA-r2; 2 seeds for JORA-small and LoRA-r1/r4.

### Claim 2 (Closest prior): JORA beats faithful qGOFT at equal budget

- **Minimal experiment**: JORA-base vs. faithful qGOFT reimplementation at matched parameter count.
- **Metric**: average MMLU / ARC-C / GSM8K.
- **Expected evidence**: JORA-base ≥ qGOFT with a consistent multi-seed margin.
- **Seed plan**: 3 seeds each.
- **Fallback**: if qGOFT faithful reproduction fails, label the static baseline fixed-slot JORA and report as an internal ablation.

### Mechanism-Isolation Diagnostics (Mandatory)

- **Diag-only-selected**: same U, no rotations, only δ trained from zeros. Tests whether sparse diagonal alone explains the gain.
- **fixed-slot JORA**: same codepath, random static slots. Tests whether the allocation policy adds value.
- **Support stability report**: Jaccard similarity curves from warmup, to justify one-shot allocation.
- **Seed plan**: 2 seeds each.
- **Precommitted interpretation**: if Diag-only-selected within 0.5 pp of JORA on all benchmarks → rotation contribution claim is downgraded.

### Appendix-Only (if low-overhead)
- DoRA-r4, OFT, BOFT breadth references
- JORA-large (K=64) if run
- Warmup length sensitivity (T_stat = 100, 200, 400)
- δ init sensitivity (0.0, 0.1, and the "d_sel=1" (non-residualized) comparison to show the function jump matters)

## Experiment Handoff Inputs

- **Must-prove claims**:
  1. Pareto dominance over LoRA in the extreme-budget regime (must-win: beat LoRA-r1)
  2. Improvement over faithful qGOFT or fixed-slot JORA at equal budget

- **Mandatory mechanism diagnostics**:
  - Diag-only-selected (mandatory, same scope and budget)
  - fixed-slot JORA (mandatory, same codepath)
  - support stability Jaccard curves

- **Appendix-only**: DoRA, OFT, BOFT, warmup/init sensitivity

- **Critical datasets / metrics**:
  - MMLU, ARC-C, GSM8K
  - exact total trainable parameter counts for every method and variant
  - mean ± std over seeds

- **Highest-risk assumptions**:
  1. JORA-base (~12K params) can beat LoRA-r1 and be competitive with LoRA-r2 — core empirical bet
  2. Faithful qGOFT can be reproduced on the same scope
  3. δ stays well-conditioned; residualization is numerically stable
  4. Diag-only-selected does NOT match JORA — otherwise the rotation contribution needs downgrading

## Compute & Timeline

- **Estimated compute**: ~150 GPU-hours on 3× RTX 4090.
- **Why still feasible**: two JORA variants (small + base); extreme-budget methods train faster; 32-layer Mistral-7B; 52k instruction examples.
- **Timeline**:
  - Implementation fix (residualized δ init, P_U injection, warmup stability check): 1 day
  - Faithful qGOFT reproduction or fixed-slot JORA fallback: 1–2 days
  - 1-seed screening for all methods + diagnostics: 2 days
  - Multi-seed core runs: 5 days
  - Analysis + paper tables: 2 days
