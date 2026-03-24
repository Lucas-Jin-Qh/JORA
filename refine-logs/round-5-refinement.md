# Round 5 Refinement

## Problem Anchor
*(Verbatim from round 0, with one targeted correction to the success condition — see Anchor Check)*

- **Bottom-line problem**: PEFT methods must achieve a better accuracy-vs-parameter Pareto frontier than LoRA for LLM fine-tuning, using a principled, structure-preserving adaptation mechanism.
- **Must-solve bottleneck**: Orthogonal PEFT (qGOFT, OFT) uses static rotation structures — all dimension pairs adapted uniformly, wasting capacity on low-signal directions. Low-rank PEFT (LoRA, DoRA) lacks geometric bias and suffers norm drift. Neither achieves an efficient adaptive sparse rotation.
- **Non-goals**: Not a new backbone, not replacing LoRA for all use cases, not targeting inference latency.
- **Constraints**: 3× RTX 4090, 2-week experiment window. PEFT library. NeurIPS 2026.
- **Success condition (corrected)**: JORA in the extreme-budget regime (~2–5% of LoRA-r4 params) achieves accuracy comparable to LoRA-r2 or better on MMLU/ARC-C/GSM8K; beats qGOFT at equal parameter budget; the Pareto curve is Pareto-dominant over LoRA in the low-to-mid budget range.

---

## Anchor Check

- **Original bottleneck**: Static rotation structure in orthogonal PEFT wastes capacity; LoRA lacks geometric bias.
- **Why the revised method still addresses it**: Yes — the method is still about adaptive allocation of a sparse bilateral rotation budget inside rotation-based PEFT. The revision this round addresses a structural mismatch: the original "~half LoRA-r4 params" success condition was inconsistent with the JORA parameterization once full-width D_diag was removed.
- **Concrete parameter math**: For Mistral-7B, q_proj and o_proj are each 4096 × 4096. LoRA-r4 on q+o = 2 × 2 × 4096 × 4 = 65,536 params per layer × 32 layers = 2,097,152 total. JORA-base K=4 per side: (4 + 4) rotation angles + |U| ≤ 16 diagonal = ~24 params per module × 2 modules × 32 layers = ~1,536 total. Full-width D_diag adds 4096 per module = ~262K additional. So:
  - Sparse-only JORA: ~1.5K params (0.07% of LoRA-r4)
  - JORA with full-width D_diag: ~264K params (12.5% of LoRA-r4)
  - LoRA-r1 on q+o: ~524K params (25% of LoRA-r4)
  - LoRA-r2 on q+o: ~1M params (50% of LoRA-r4)
- **Why the "half LoRA-r4" success condition was wrong**: It was set before the parameter accounting was computed carefully. Sparse Givens rotations are inherently parameter-minimal; the original anchor implicitly required full-rank capacity (like a full diagonal) to reach that operating point.
- **Corrected success condition**: JORA should be positioned as an **extreme-budget PEFT method**. The honest claim is: JORA at <5% LoRA-r4 parameter budget achieves accuracy at or above LoRA-r2 (50% budget). That is a Pareto-dominant frontier story — fewer parameters, equal or better accuracy — which is a stronger claim than "half budget, same accuracy."
- **Reviewer suggestions rejected as drift**:
  - OFT/BOFT as main claim blockers: still rejected.
  - Broad "adaptive PEFT" framing: still rejected.

---

## Simplicity Check

- **Dominant contribution after revision**: One mechanism: one-time adaptive allocation of sparse bilateral rotation slots within rotation-based PEFT, followed by fixed-support optimization with a selected-support diagonal core and correct initialization.
- **Components removed or merged**:
  - Zero-initialization of `d_sel` is replaced with a small nonzero init (e.g., 1.0 or tuned constant), so rotation angles receive nonzero gradient from step 1.
  - OER is fully removed from the proposal text — no longer mentioned in the core narrative, appendix-only only if forced by screening.
  - Repeated reseating is removed entirely from the narrative.
- **Reviewer suggestions rejected as unnecessary complexity**:
  - Reverting to full-width D_diag to recover parameter count: rejected. Instead, the success condition is corrected to match actual JORA capabilities.
- **Why the remaining mechanism is still the smallest adequate route**: The paper now has one mechanism, one correct success story (extreme-budget Pareto dominance), and one clean baseline pairing (faithful qGOFT).

---

## Changes Made

### 1. Correct the success condition to match actual JORA parameter scale

- **Reviewer said**: "Once the full-width diagonal is removed, K=4 plus d_sel[|U|] gives a parameter count that appears far below the stated ~half LoRA-r4 regime."
- **Action**: The success condition is corrected. The story is now **extreme-budget Pareto dominance**: JORA at <<5% LoRA-r4 params achieves accuracy at or above LoRA-r2 (50% of LoRA-r4). This is a stronger and more honest Pareto story.
- **Reasoning**: Forcing K to ~2700 to reach "half LoRA-r4" would make the rotation pair count absurd and destroy the sparse-allocation narrative. The honest approach is to match the claim to the parameterization, not the other way around. A method that beats LoRA-r2 at 3% of its parameter count is a stronger story, not a weaker one.
- **Impact on core method**: Strengthens contribution quality and venue readiness; fixes feasibility.

### 2. Fix the d_sel zero-initialization gradient blockade

- **Reviewer said**: "If d_sel is initialized to zero, then θ_L and θ_R get zero gradient at initialization because the whole path is multiplied by D_sel."
- **Action**: `d_sel` is initialized to a small nonzero constant (default: 1.0) so that the rotation mechanism is immediately active. The initialization is documented explicitly in the training plan.
- **Reasoning**: This is a correctness fix. With d_sel=0, only the diagonal component trains at t=1; the rotation angles learn only after d_sel drifts away from 0, which is both slow and unpredictable.
- **Impact on core method**: Fixes the optimization trajectory; makes θ_R and θ_L immediately load-bearing.

### 3. Make Diag-only-selected mandatory in the validation package

- **Reviewer said**: "Keep Diag-only-selected as mandatory, not optional."
- **Action**: `Diag-only-selected` is now a required mechanism-isolation control. It directly answers: does the gain come from adaptive sparse rotation, or just from sparse diagonal scaling on selected dimensions?
- **Reasoning**: Without this, the paper cannot defend the rotation claim if the diagonal is doing most of the work.
- **Impact on core method**: Strengthens scientific integrity of the paper; does not change the method itself.

### 4. Add support-stability reporting to the warmup narrative

- **Reviewer said**: "Report support-stability evidence from warmup so the one-shot allocation looks justified rather than arbitrary."
- **Action**: The training plan now requires a **support stability diagnostic** during warmup: measure Jaccard similarity of the top-2K dimension sets across adjacent EMA windows. If Jaccard > 0.9 at the end of warmup, the one-shot allocation is stable; if not, a second reseat is allowed as a fallback.
- **Reasoning**: This converts "one-shot allocation" from a heuristic to a justified choice backed by a stability check.
- **Impact on core method**: Improves methodological justification without adding complexity.

### 5. Sharpen Contribution Quality framing

- **Action**: The contribution is now framed around the **extreme-budget Pareto frontier** story, which is both more accurate and more impactful: sparse adaptive rotation allocation enables a parameter efficiency regime that LoRA cannot reach without sacrificing accuracy.
- **Reasoning**: A method that achieves LoRA-r2 accuracy at 3% of LoRA-r4 params is Pareto-dominant — not just "slightly better at half budget." This is a more exciting paper.

---

## Revised Proposal

# Research Proposal: JORA — Adaptive Sparse Rotation-Slot Allocation for Extreme-Budget Square-Layer PEFT

## Problem Anchor

- **Bottom-line problem**: PEFT methods must achieve a better accuracy-vs-parameter Pareto frontier than LoRA for LLM fine-tuning, using a principled, structure-preserving adaptation mechanism.
- **Must-solve bottleneck**: Orthogonal PEFT (qGOFT, OFT) uses static rotation structures — all dimension pairs adapted uniformly, wasting capacity on low-signal directions. Low-rank PEFT (LoRA, DoRA) lacks geometric bias and suffers norm drift. Neither achieves an efficient adaptive sparse rotation.
- **Non-goals**: Not a new backbone, not replacing LoRA for all use cases, not targeting inference latency.
- **Constraints**: 3× RTX 4090, 2-week experiment window. PEFT library. NeurIPS 2026.
- **Success condition**: JORA in the extreme-budget regime (~2–5% of LoRA-r4 params, i.e., ~40K–100K total params on Mistral-7B q+o scope) achieves accuracy at or above LoRA-r2 (which uses ~50% of LoRA-r4 params) on MMLU/ARC-C/GSM8K; the resulting Pareto frontier is Pareto-dominant over LoRA in the low-to-mid parameter regime; JORA beats faithful qGOFT at equal budget.

## Technical Gap

`qGOFT` gives rotation-based PEFT a strong geometric primitive, but its rotation budget is **static**: the rotation structure is chosen once by a fixed schedule and then trained. `LoRA` allocates low-rank capacity without rotation structure and allows norm drift.

The clean gap is narrow: **within rotation-based PEFT, there is no method that uses early task statistics to allocate a fixed sparse bilateral rotation budget with a matched sparse diagonal core, then freezes that support for stable optimization**.

The Pareto story is honest and strong: JORA's adaptive sparse rotation mechanism is so parameter-efficient that it achieves LoRA-r2-level accuracy at a fraction of LoRA-r2's parameter count. This is not achieved by being close to LoRA-r4 in parameter count — it is achieved by using a better inductive bias (rotation structure) with better allocation (adaptive vs static).

## Method Thesis

JORA is a square-layer, mergeable additive adapter that first runs a short statistics warmup to score input and output dimensions, then allocates **one fixed set of sparse bilateral rotation slots** and trains only the associated rotation angles plus a **selected-support diagonal core initialized to 1.0**. The method changes **where** a fixed sparse rotation budget goes, not the underlying rotation primitive.

This is the smallest adequate intervention because:
- the prior-art gap is about **static vs adaptive allocation**,
- the diagonal core is explicitly tied to the same selected support,
- the mainline method avoids full-width diagonals, repeated reseating, tanh, and rectangular operators.

**Honest framing**: JORA computes

```text
delta = R_L^T · D_sel · R_R · x
out   = base_out + delta
```

with `D_sel` initialized to the identity on the selected support (not zero). It is a rotation-basis structured additive adapter, not a direct orthogonal transformation of `W₀`.

**Parameter scale**: JORA-base with K=32 slots per side on q+o scope uses approximately:
- 32 + 32 = 64 rotation angles per module
- |U| ≤ 128 diagonal values per module
- Total: ~12,300 params across 32 layers
- Compare: LoRA-r2 ≈ 1,048,576 params; LoRA-r4 ≈ 2,097,152 params
- JORA-base operates at ~0.6% of LoRA-r4, targeting accuracy at or above LoRA-r2

## Contribution Focus

- **Single dominant contribution**: One-time adaptive allocation of sparse bilateral rotation slots within rotation-based PEFT, yielding Pareto-dominant extreme-budget accuracy on square attention layers.
- **Explicit non-contributions**: full-width diagonal cores in the main method, repeated reseating as the main method, broad "adaptive PEFT" claims, rectangular-layer main results, tanh-based merge path.
- **OER**: removed from all proposal text; may appear in appendix only if a stable multi-seed gain forces it.

## Proposed Method

### Complexity Budget
- **Frozen**: Pretrained `W₀`, model backbone, tokenizer, PEFT infrastructure.
- **New trainable parameters (mainline)**:
  - `θ_R [K_R]`: right-rotation angles for frozen input-side slot pairs
  - `θ_L [K_L]`: left-rotation angles for frozen output-side slot pairs
  - `d_sel [|U|]`: diagonal-core values on selected support `U = dims(slots_R) ∪ dims(slots_L)`, `|U| ≤ 4K`, **initialized to 1.0**
- **Structural state (non-trainable after allocation)**:
  - `slots_R = {(i_r, j_r)}_{r=1}^{K_R}`
  - `slots_L = {(i_l, j_l)}_{l=1}^{K_L}`
  - `U = dims(slots_R) ∪ dims(slots_L)`
- **Intentionally excluded from the main paper**: full-width `D_diag[d]`, repeated reseating as default, rectangular operators, `k/v` under GQA, MLP projections, tanh merge variants, OER in the core method.

### System Overview

Main-paper target modules: **square attention projections only (`q_proj`, `o_proj`)**.

Core claim baselines use the **same module set**:
- `LoRA-r1`, `LoRA-r2`, `LoRA-r4` for the Pareto frontier
- faithful `qGOFT` reimplementation for the closest static orthogonal baseline

Non-core references are appendix-only if low-overhead:
- `fixed-slot JORA` as an internal static-allocation ablation
- `Diag-only-selected` as a mandatory diagonal-core isolation control
- `DoRA`, `OFT`, `BOFT` as breadth references

For a square layer of width `d` and input `x ∈ ℝ^{B × d}`:

```text
base_out = W₀ · x

# statistics warmup (no slot changes yet)
ema_in[j]  ← β · ema_in[j]  + (1-β) · mean_batch(x_j²)
ema_out[i] ← β · ema_out[i] + (1-β) · mean_batch((∂L/∂y_i)²)

# single allocation after warmup (at step T_stat)
slots_R ← deterministic_pairs(top_2K(ema_in))
slots_L ← deterministic_pairs(top_2K(ema_out))
U       ← dims(slots_R) ∪ dims(slots_L)
d_sel   ← ones(|U|)   # initialized to 1.0, not 0

# fixed-support training (θ_R, θ_L, d_sel all trainable from step T_stat)
x̃ = R_R(slots_R, θ_R) · x
x̂ = D_sel(U, d_sel) · x̃
δ  = R_L^T(slots_L, θ_L) · x̂

out = base_out + δ
```

where `D_sel(U, d_sel) = Diag(v)` with `v_u = d_sel[u]` for `u ∈ U`, `v_u = 0` otherwise.

No tanh is used in the main forward path. After allocation, the module is fully linear and exactly mergeable.

### Core Mechanism

#### 1. Selection statistics

**Right-side input statistic**:
```text
ema_in[j] ← β · ema_in[j] + (1-β) · mean_batch(x_j²)
```
This is a **cheap activation proxy**, not a task-driven gradient statistic.

**Left-side output statistic**:
```text
ema_out[i] ← β · ema_out[i] + (1-β) · mean_batch((∂L/∂y_i)²)
```
This is the task-coupled learning signal from the layer-output gradient during warmup.

#### 2. Support-stability diagnostic

Before committing to single allocation, measure stability of the candidate support:

```text
# at each of the last 5 warmup steps, record top_2K(ema_in) and top_2K(ema_out)
# compute Jaccard similarity between consecutive top-2K sets
# if mean Jaccard > 0.85, proceed with one-shot allocation at T_stat
# if mean Jaccard < 0.85, optionally extend T_stat by 50 steps (once) and re-check
```

This converts one-shot allocation from a heuristic to a justified decision backed by a convergence check. If the paper includes this diagnostic, the allocation is reported as "stable" or "stabilized after one extension" — never as arbitrary.

#### 3. Deterministic disjoint pairing rule

For either score vector (`ema_in` or `ema_out`):
1. Stable-sort dimensions by **descending score**; ties are broken by **smaller dimension index first**.
2. Keep the first `2K` dimensions from that ordered list.
3. Form pairs consecutively: `(d₁, d₂), (d₃, d₄), …, (d₂K₋₁, d₂K)`.
4. Store each pair canonically as `(min(d_a, d_b), max(d_a, d_b))`.

This rule is deterministic, disjoint, and fully reproducible.

#### 4. Single allocation and fixed-support diagonal core

After the statistics warmup:

```text
slots_R ← deterministic_pairs(top_2K(ema_in))
slots_L ← deterministic_pairs(top_2K(ema_out))
U       ← dims(slots_R) ∪ dims(slots_L)
d_sel   ← ones(|U|)       # NOT zeros — avoids gradient blockade
θ_R     ← zeros(K_R)      # rotation angles start at identity
θ_L     ← zeros(K_L)
```

With `d_sel = 1.0` at initialization, the path `R_L^T · D_sel · R_R` is the identity on `U` and zero elsewhere. This means rotations receive nonzero gradient from the first training step.

#### 5. Fixed-support optimization

After allocation:
- `slots_R`, `slots_L`, and `U` are frozen permanently
- `θ_R`, `θ_L`, and `d_sel` are all trained jointly from step `T_stat`

### Mergeability

For the mainline method:

```text
W_merged = W₀ + R_L^T · D_sel · R_R
```

This is an exact linear merge for the square-layer main method.

### Training Plan
1. **Backbone / data**: Mistral-7B-v0.1, Alpaca-cleaned 52k, standard SFT.
2. **Target modules (main paper)**: `q_proj`, `o_proj` only.
3. **Statistics warmup**: `T_stat = min(200, 0.05 · T_total)` steps. Accumulate `ema_in` and `ema_out`.
4. **Stability check**: measure Jaccard similarity of top-2K sets in the last 5 warmup steps. Extend warmup by 50 steps once if Jaccard < 0.85.
5. **Single allocation**: at `t = T_stat`, allocate `slots_R`, `slots_L`, `U`; init `d_sel = ones(|U|)`, `θ_R = θ_L = zeros`.
6. **Main optimization**: AdamW, JORA-specific LR (tuned; may be higher than LoRA LR), cosine decay.
7. **Model sizes**:
   - `JORA-small`: `K=8` per side, ~1.5K params on q+o, 32 layers
   - `JORA-base`: `K=32` per side, ~12K params on q+o, 32 layers
   - `JORA-large`: `K=64` per side, ~24K params on q+o, 32 layers
8. **Parameter accounting** (all paper tables must report exact counts):
   - JORA-small K=8: 2 × 32 × (8 + 8 + 32) = ~3,072 total params
   - JORA-base K=32: 2 × 32 × (32 + 32 + 128) = ~12,288 total params
   - LoRA-r2 (q+o, 32 layers): 2 × 32 × 2 × 4096 × 2 = 1,048,576 params
   - **JORA-base uses ~1.2% of LoRA-r2 parameter budget**

### Failure Modes and Diagnostics
- **Warmup unstable (Jaccard < 0.85)**: extend T_stat by 50 steps once. If still unstable, report support instability and increase β.
- **d_sel explodes during training**: add a small L2 penalty on d_sel or clip its values. Detect via max |d_sel| per module.
- **Diag-only-selected matches JORA**: if `Diag-only-selected` achieves similar accuracy, the rotation mechanism is not contributing. Narrow the claim to "adaptive sparse diagonal scaling" and report the rotation ablation honestly.
- **Faithful qGOFT reproduction fails**: keep only `fixed-slot JORA` as the static baseline and label it clearly as an internal ablation.

### Novelty and Elegance Argument

- **vs. qGOFT**: JORA changes the one thing that matters — where the sparse rotation budget goes — by using early task statistics instead of a fixed schedule.
- **vs. fixed-slot JORA**: same parameterization, same support size, different allocation. This cleanly isolates adaptive allocation value.
- **vs. LoRA**: JORA achieves comparable or better accuracy at 1–5% of LoRA's parameter count, with a rotation-structured adapter instead of a free-form low-rank update.
- **Why this is focused**: one mechanism, one parameter regime, one clean Pareto story.
- **Why the Pareto story is now stronger**: "matches LoRA-r2 at 1% of LoRA-r2 parameters" is a more exciting claim than "matches LoRA-r4 at half budget."

## Claim-Driven Validation Sketch

**Validation-focus rule**: two claim-bearing experiment blocks; one mandatory mechanism-isolation block; everything else appendix-only.

### Claim 1 (Primary): Pareto-dominant over LoRA in the extreme-budget regime

- **Minimal experiment**: `JORA-small` (K=8), `JORA-base` (K=32), `JORA-large` (K=64) vs. `LoRA-r1`, `LoRA-r2`, `LoRA-r4`, all on `q_proj + o_proj` only.
- **Metric**: average accuracy on MMLU / ARC-C / GSM8K vs exact total trainable parameter count.
- **Expected evidence**: JORA-base (~12K params) matches or exceeds LoRA-r2 (~1M params) on average; the JORA Pareto curve lies above the LoRA Pareto curve in the extreme-budget region.
- **Seed plan**: 3 seeds for `JORA-base` and `LoRA-r2`; 2 seeds for others.

### Claim 2 (Closest prior): JORA beats faithful qGOFT at equal budget

- **Minimal experiment**: `JORA-base` vs. faithful `qGOFT` reimplementation at matched parameter count.
- **Metric**: average MMLU / ARC-C / GSM8K.
- **Expected evidence**: `JORA-base ≥ qGOFT` with a consistent multi-seed margin.
- **Seed plan**: 3 seeds each.
- **Naming rule**: if only shared-codepath static implementation is available, label it `fixed-slot JORA`.

### Mechanism-Isolation Diagnostics (Mandatory)

- **`Diag-only-selected`**: same selected support `U`, no rotations, `d_sel` trained from 1.0. Tests whether adaptive sparse diagonal scaling alone explains the gain.
- **`fixed-slot JORA`**: same codepath, random static slots instead of adaptive. Tests whether the allocation policy adds value.
- **Support stability report**: Jaccard similarity curves from warmup, to justify one-shot allocation.
- **Seed plan**: 2 seeds each.

### Appendix-Only (if low-overhead)

- `DoRA-r4`, `OFT`, `BOFT` breadth references
- Warmup length sensitivity (T_stat = 100, 200, 400)
- d_sel init sensitivity (0.1, 0.5, 1.0, 2.0)

## Experiment Handoff Inputs

- **Must-prove claims**:
  1. Pareto dominance over LoRA in the extreme-budget regime
  2. Improvement over faithful `qGOFT` at equal budget

- **Mandatory mechanism diagnostics**:
  - `Diag-only-selected` (mandatory, same scope and budget)
  - `fixed-slot JORA` (mandatory, same codepath)
  - support stability Jaccard curves

- **Appendix-only**: `DoRA`, `OFT`, `BOFT`, warmup/init sensitivity

- **Critical datasets / metrics**:
  - MMLU, ARC-C, GSM8K
  - exact total trainable parameter counts for every method and variant
  - mean ± std over seeds

- **Highest-risk assumptions**:
  1. JORA-base (~12K params) can actually match LoRA-r2 accuracy on LLM instruction fine-tuning — this is the core empirical bet
  2. Faithful qGOFT can be reproduced on the same scope
  3. d_sel stays well-conditioned and does not need OER to stabilize

## Compute & Timeline

- **Estimated compute**: ~150 GPU-hours on 3× RTX 4090.
- **Why still feasible**: extreme-budget methods train faster than LoRA (fewer params, same forward pass, same data); two claim-bearing blocks; 32-layer Mistral-7B; 52k instruction examples.
- **Timeline**:
  - Implementation fix (d_sel init, warmup stability check, merge path): 1 day
  - Faithful qGOFT reproduction or fixed-slot JORA fallback: 1–2 days
  - 1-seed screening for all methods + diagnostics: 2 days
  - Multi-seed core runs: 5 days
  - Analysis + paper tables: 2 days
