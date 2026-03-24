# Round 3 Refinement

## Problem Anchor
*(Verbatim from round 0)*

- **Bottom-line problem**: PEFT methods must achieve a better accuracy-vs-parameter Pareto frontier than LoRA for LLM fine-tuning, using a principled, structure-preserving adaptation mechanism.
- **Must-solve bottleneck**: Orthogonal PEFT (qGOFT, OFT) uses static rotation structures — all dimension pairs adapted uniformly, wasting capacity on low-signal directions. Low-rank PEFT (LoRA, DoRA) lacks geometric bias and suffers norm drift. Neither achieves an efficient adaptive sparse rotation.
- **Non-goals**: Not a new backbone, not replacing LoRA for all use cases, not targeting inference latency.
- **Constraints**: 3× RTX 4090, 2-week experiment window. PEFT library. NeurIPS 2026.
- **Success condition**: JORA at ~half LoRA-r4 params matches LoRA-r4 on MMLU/ARC-C/GSM8K; clearly beats LoRA-r2; beats qGOFT at equal budget.

---

## Anchor Check

- **Original bottleneck**: Static rotation structure in orthogonal PEFT wastes capacity; LoRA lacks geometric bias.
- **Why the revised method still addresses it**: Yes — the method is now explicitly framed as adaptive sparse rotation-slot allocation **within rotation-based PEFT**, with the main paper restricted to square attention projections where the operator is fully well-defined. This keeps the method aimed at qGOFT’s static-allocation weakness rather than drifting into a broader or different PEFT problem.
- **Reviewer suggestions rejected as drift**:
  - Broadening the novelty claim to “adaptive PEFT selection” was rejected; that would overstate the problem being solved.
  - Keeping rectangular layers in the main method was rejected; it adds scope without improving the core bottleneck story.

---

## Simplicity Check

- **Dominant contribution after revision**: One mechanism: adaptive allocation of sparse bilateral rotation slots during burn-in, then frozen optimization on square layers.
- **Components removed or merged**:
  - Rectangular-layer operator removed from the main proposal; moved to appendix/future work.
  - Novelty claim narrowed to rotation-based PEFT only.
  - Right-side statistic kept as a cheap activation proxy, not promoted to a stronger “task-driven gradient” claim.
  - OER kept as a stabilizer only; if 3-seed ablations do not show stable gain, it drops to appendix.
- **Reviewer suggestions rejected as unnecessary complexity**:
  - Introducing a new rectangular operator now.
  - Replacing the right-side activation proxy with a second backward-path statistic before establishing the square-layer main result.
- **Why the remaining mechanism is still the smallest adequate route**: Bilateral sparse rotations + diagonal core already provide the structure-preserving adapter; burn-in allocation + freeze is the minimal adaptive mechanism needed to beat static qGOFT. Restricting to square layers removes the main unresolved mathematical ambiguity without changing the core idea.

---

## Changes Made

### 1. Restrict the main method to square layers only
- **Reviewer said**: “Method is fully specified only for square layers. Restrict main paper to square-layer-only; rectangular to appendix/future work unless valid operator is defined.”
- **Action**: Main proposal now targets **square attention projections only** in the main paper — specifically `q_proj` and `o_proj` for Mistral-7B. `k_proj`, `v_proj`, and MLP projections are moved to appendix/future work.
- **Reasoning**: This removes the biggest remaining specification hole and forces all baselines onto the same clean module scope.
- **Impact on core method**: Improves feasibility and venue readiness without changing the central claim.

### 2. Make the right-side signal claim fully honest
- **Reviewer said**: “EMA[x²] is a cheap activation proxy, not a task-driven learning signal. Present honestly.”
- **Action**: The proposal now consistently calls `ema_in[j] = EMA[x_j²]` a **cheap activation proxy**. Only the left-side statistic is described as a gradient-based learning signal.
- **Reasoning**: The method remains practical under the stated compute budget, but the claim is no longer overstated.
- **Impact on core method**: Slightly narrows the strength of the story, but makes the method technically honest and more defensible.

### 3. Make the novelty and pairing rule fully precise
- **Reviewer said**: “Novelty should be narrow. Deterministic pairing rule must be fully explicit.”
- **Action**:
  - Novelty claim narrowed to **adaptive allocation of sparse rotation slots within rotation-based PEFT**.
  - Pairing rule now fully specified: stable descending sort by score with lower-index tie-break, take top `2K`, pair consecutive dimensions, store each pair in sorted `(min, max)` order.
- **Reasoning**: This removes pseudo-novelty risk and makes the implementation exactly reproducible.
- **Impact on core method**: Improves contribution quality and method specificity.

### 4. Strengthen validation and justify OER at the mechanism level
- **Reviewer said**: “qGOFT and random-slots need enough seeds; justify why inverse base-row-norm scaling is the right stabilizer.”
- **Action**:
  - Seed coverage increased for all claim-bearing comparisons: `JORA-base`, `LoRA-r4`, `qGOFT`, `JORA-random-slots`, and `JORA-no-OER` all get 3 seeds.
  - OER justification now states explicitly that inverse base-row-norm scaling equalizes each row’s initial additive perturbation budget and preserves the total sum of squared output-row norms at initialization.
- **Reasoning**: These were the last main “artifact” and “heuristic” vulnerabilities.
- **Impact on core method**: Validation becomes harder to dismiss; OER becomes a principled design choice instead of a convenient heuristic.

---

## Revised Proposal

# Research Proposal: JORA — Adaptive Sparse Rotation-Slot Allocation for Square-Layer PEFT

## Problem Anchor

- **Bottom-line problem**: PEFT methods must achieve a better accuracy-vs-parameter Pareto frontier than LoRA for LLM fine-tuning, using a principled, structure-preserving adaptation mechanism.
- **Must-solve bottleneck**: Orthogonal PEFT (qGOFT, OFT) uses static rotation structures — all dimension pairs adapted uniformly, wasting capacity on low-signal directions. Low-rank PEFT (LoRA, DoRA) lacks geometric bias and suffers norm drift. Neither achieves an efficient adaptive sparse rotation.
- **Non-goals**: Not a new backbone, not replacing LoRA for all use cases, not targeting inference latency.
- **Constraints**: 3× RTX 4090, 2-week experiment window. PEFT library. NeurIPS 2026.
- **Success condition**: JORA at ~half LoRA-r4 params matches LoRA-r4 on MMLU/ARC-C/GSM8K; clearly beats LoRA-r2; beats qGOFT at equal budget.

## Technical Gap

qGOFT gives rotation-based PEFT a strong geometric primitive, but its rotation budget is **static**: once dimension pairs are chosen, all selected pairs are treated uniformly throughout training. LoRA allocates low-rank capacity without rotation structure and allows norm drift.

The specific gap is therefore not “adaptive PEFT selection” in general. It is narrower and cleaner: **within rotation-based PEFT, there is no method that adaptively allocates a fixed sparse bilateral rotation budget during burn-in, then freezes the structure for stable optimization**.

To keep the operator mathematically closed and implementation-ready, the main paper should target **square attention projections only**. On Mistral-7B, that means `q_proj` and `o_proj` in the main result set; rectangular projections move to appendix/future work.

## Method Thesis

JORA is a square-layer, rotation-structured additive adapter: during burn-in, it uses a **cheap activation proxy** on input dimensions and a **gradient-energy signal** on output dimensions to allocate a fixed number of sparse bilateral rotation slots; after burn-in, slot identities are frozen and only the rotation angles, diagonal core, and optional magnitude stabilizer are trained.

This is the smallest adequate intervention because it changes only the **allocation of rotation capacity**, not the underlying PEFT primitive. The paper is therefore about one mechanism: **adaptive sparse rotation-slot allocation inside rotation-based PEFT**.

**Honest framing**: JORA computes `delta = R_L^T · D_diag · R_R · x` and uses `out = base_out + s ⊙ delta`. It is a rotation-basis structured additive adapter, not a direct orthogonal transformation of `W₀`.

## Contribution Focus

- **Single dominant contribution**: Adaptive allocation of sparse bilateral rotation slots within rotation-based PEFT — burn-in selection followed by frozen optimization on square layers.
- **Method design element** (not co-contribution): OER magnitude stabilization for additive adapters.
- **Explicit non-contributions**: Broad “adaptive PEFT selection” claims, rectangular-layer main results, dense orthogonal fine-tuning, BlockCore/LowRankCore, tanh-based merge path.

## Proposed Method

### Complexity Budget
- **Frozen**: Pretrained `W₀`, model backbone, tokenizer, PEFT infrastructure.
- **New trainable parameters**:
  - `θ_R [K_R]`: right-rotation angles for frozen input-side slot pairs
  - `θ_L [K_L]`: left-rotation angles for frozen output-side slot pairs
  - `D_diag [d]`: diagonal core for square layer dimension `d`
  - `e [d]`: optional OER logits
- **Structural state (non-trainable after burn-in)**:
  - `slots_R = {(i_r, j_r)}_{r=1}^{K_R}`
  - `slots_L = {(i_l, j_l)}_{l=1}^{K_L}`
- **Intentionally excluded from the main paper**: rectangular operators, k/v projections under GQA, MLP projections, BlockCore, LowRankCore, tanh merge variants.

### System Overview

Main-paper target modules: **square attention projections only (`q_proj`, `o_proj`)**.
Core claim baselines (`LoRA`, `DoRA`, `qGOFT`) use the **same module set** in the main comparison.
If their PEFT implementations are stable on the same scope, `OFT` and `BOFT` are added as **secondary orthogonal-family reference baselines** in the appendix, not as main claim blockers.

For a square layer of width `d` and input `x ∈ ℝ^{B × d}`:

```text
base_out = W₀ · x

x̃ = R_R(slots_R, θ_R) · x
x̂ = D_diag ⊙ x̃
δ  = R_L^T(slots_L, θ_L) · x̂

s  = oer_softmax(base_row_norms, e, τ)  ∈ ℝ^d
out = base_out + s ⊙ δ
```

No tanh is used in the main forward path. After burn-in, the module is fully linear and exactly mergeable.

### Core Mechanism

#### 1. Selection statistics

**Right-side input statistic**:
```text
ema_in[j] ← β · ema_in[j] + (1-β) · mean_batch(x_j²)
```
This is a **cheap activation proxy**, not a task-driven gradient statistic.

**Left-side output statistic**:
```text
ema_out[i] ← β · ema_out[i] + (1-β) · mean_batch((∂L/∂δ_i)²)
```
This is the task-coupled learning-signal statistic, computed from a backward hook on the additive delta output.

#### 2. Deterministic disjoint pairing rule

For either score vector (`ema_in` or `ema_out`):
1. Stable-sort dimensions by **descending score**; ties are broken by **smaller dimension index first**.
2. Keep the first `2K` dimensions from that ordered list.
3. Form pairs consecutively: `(d₁, d₂), (d₃, d₄), …, (d₂K₋₁, d₂K)`.
4. Store each pair canonically as `(min(d_a, d_b), max(d_a, d_b))`.

This rule is deterministic, disjoint, and fully reproducible.

#### 3. Burn-in allocation and freeze

```text
for step t in training:
    update ema_in every step from x
    update ema_out every step from ∂L/∂δ

    if t <= 0.10 · T_total and t % 50 == 0:
        slots_R ← deterministic_pairs(top_2K(ema_in))
        slots_L ← deterministic_pairs(top_2K(ema_out))
        reset θ only for slots whose identities changed

after step ceil(0.10 · T_total):
    freeze slots_R and slots_L permanently
    train only θ_R, θ_L, D_diag, and optional e
```

The core novelty is the **allocation-and-freeze pattern**, not the pairing heuristic by itself.

### Optional Supporting Component: OER Magnitude Stabilization

For output row `i`:
```text
s_i = softmax(e_i / τ) · E₀ / ||w_i^0||₂,
where E₀ = (Σ_i ||w_i^0||₂²)^{1/2}
```

Why this stabilizer is principled for an additive adapter:
- The adapter is added on top of a fixed base row `w_i^0`; without normalization, large-norm base rows would receive disproportionately large additive perturbations.
- Dividing by `||w_i^0||₂` converts the scale into a **relative perturbation budget** rather than an absolute one.
- With `e=0` at initialization,
  ```text
  s_i² ||w_i^0||₂² = E₀² / d²
  ```
  for every row `i`, so each row gets the same initial squared perturbation budget.
- Summing over rows preserves a fixed total squared output-row norm budget at initialization:
  ```text
  Σ_i s_i² ||w_i^0||₂² = E₀² / d
  ```

So OER is justified as a **row-budget equalizer** for additive adaptation, not as a separate novelty axis.

If 3-seed ablations do not show a stable gain, OER moves to appendix and the main paper falls back to the no-OER variant.

### Mergeability

After burn-in, with frozen slots and no tanh:
```text
W_merged = W₀ + Diag(s) · R_L^T · D_diag · R_R
```
This is an exact linear merge for the square-layer main method.

### Training Plan
1. **Backbone / data**: Mistral-7B-v0.1, Alpaca-cleaned 52k, standard SFT.
2. **Target modules (main paper)**: `q_proj`, `o_proj` only.
3. **Optimizer**: AdamW with JORA-specific LR (10–20× standard LoRA LR), cosine decay.
4. **Burn-in**: first 10% of steps; update EMAs every step; reseat every 50 steps.
5. **Post-burn-in**: freeze slot identities; continue training `θ_L`, `θ_R`, `D_diag`, and optional `e`.
6. **Model sizes**:
   - `JORA-small`: `K=2` per side, ~¼ LoRA-r4 parameter budget
   - `JORA-base`: `K=4` per side, ~½ LoRA-r4 parameter budget
7. **OER schedule**: `τ` annealed from 1.0 to 0.1 if OER is enabled.

### Failure Modes and Diagnostics
- **Slot collapse**: low diversity among selected dimensions after burn-in. Detect by unique-dimension count and slot entropy. Mitigation: increase EMA smoothing `β` or enlarge burn-in window slightly.
- **Noisy left-side gradient signal**: unstable `ema_out`. Detect via per-step variance spikes. Mitigation: gradient clipping on the hooked delta tensor and larger EMA smoothing.
- **OER saturation**: one or two rows dominate `s`. Detect via Gini coefficient of `s`. Mitigation: raise `τ` or drop OER from the main method.
- **Burn-in complexity not paying off**: repeated reseating gives no benefit over final allocation only. Detect in screening. Mitigation: collapse to a single allocation-at-burn-in-end variant and keep the simpler version.

### Novelty and Elegance Argument

- **vs. qGOFT**: same rotation primitive family, but JORA adaptively allocates sparse bilateral rotation slots during burn-in and then freezes them. That is the paper’s real mechanism-level difference.
- **vs. LoRA / DoRA**: JORA keeps a rotation-structured basis instead of low-rank free-form adaptation.
- **vs. broader adaptive PEFT claims**: JORA is **not** the first adaptive PEFT method broadly. The narrow novelty is adaptive slot allocation **inside rotation-based PEFT**.
- **Why this is focused**: the method modifies only one thing that qGOFT does not do — where the sparse rotation budget goes.

## Claim-Driven Validation Sketch

### Claim 1 (Primary): Better Pareto frontier than LoRA on the same square-layer module set
- **Minimal experiment**: `JORA-small`, `JORA-base` vs. `LoRA-r1/r2/r4` and `DoRA-r4`, all on `q_proj + o_proj` only.
- **Metric**: average accuracy on MMLU / ARC-C / GSM8K vs exact trainable parameter count.
- **Expected evidence**: `JORA-base` at ~½ `LoRA-r4` params matches `LoRA-r4`; `JORA-small` clearly beats `LoRA-r2` at comparable or lower budget.
- **Seed plan**: 3 seeds for `JORA-base` and `LoRA-r4`; 2 seeds for `JORA-small`, `LoRA-r2`, and `DoRA-r4`, expanded if frontier decisions hinge on them.

### Claim 2: JORA beats the closest static orthogonal baseline at equal budget
- **Minimal experiment**: `JORA-base` vs. `qGOFT` implemented as the same codepath with fixed random slots, no EMA allocation, no OER.
- **Metric**: average MMLU / ARC-C / GSM8K at matched trainable parameter count.
- **Expected evidence**: `JORA-base ≥ qGOFT` with a consistent multi-seed margin.
- **Seed plan**: 3 seeds each for `JORA-base` and `qGOFT`.
- **Secondary appendix references**: if implementation is straightforward on the same `q_proj + o_proj` scope, add `OFT` and `BOFT` as orthogonal-family reference baselines with 1–2 seed screening runs. They strengthen breadth-of-comparison, but they are not the decisive baseline for the main novelty claim because they do not isolate the static sparse-slot allocation question as directly as `qGOFT`.

### Claim 3 (Ablation): Allocation and OER contributions can be separated
- **Minimal experiment**: `JORA-base` vs. `JORA-random-slots` vs. `JORA-no-OER`.
- **Metric**: same 3-benchmark average plus per-seed variance.
- **Expected evidence**:
  - `JORA-base > JORA-random-slots` isolates the value of adaptive slot allocation.
  - `JORA-base ≥ JORA-no-OER` tests whether OER is worth keeping in the main method.
- **Seed plan**: 3 seeds each for `JORA-base`, `JORA-random-slots`, and `JORA-no-OER`.

## Experiment Handoff Inputs
- **Must-prove claims**: (1) Pareto gain vs LoRA on square-layer scope; (2) improvement over qGOFT at equal budget; (3) whether OER survives multi-seed ablation.
- **Must-run ablations**: fixed random slots, no-OER, optional single-allocation-only simplification.
- **Secondary comparison if low-overhead**: OFT and BOFT on the same `q_proj + o_proj` scope as appendix-level orthogonal-family references.
- **Critical datasets / metrics**: MMLU, ARC-C, GSM8K; exact trainable parameter counts; mean ± std over seeds.
- **Highest-risk assumptions**:
  1. `q_proj + o_proj` scope is enough to show the square-layer story clearly.
  2. The activation-proxy / gradient-proxy split is strong enough to beat qGOFT.
  3. OER provides a repeatable gain rather than a one-seed heuristic bump.
  4. OFT/BOFT can be run on the same scope without introducing a separate implementation-comparability problem.

## Compute & Timeline
- **Estimated compute**: ~240 GPU-hours on 3× RTX 4090.
- **Why still feasible**: Main-paper scope is now square attention projections only, which lowers implementation risk even with stronger seed coverage.
- **Timeline**:
  - square-scope code cleanup + merge-path verification: 2 days
  - qGOFT parity validation on shared codepath: 1 day
  - 1-seed screening for all methods + optional single-allocation simplification: 2 days
  - multi-seed core runs: 7 days
  - analysis + paper-facing tables/plots: 2 days
