# Round 4 Refinement

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
- **Why the revised method still addresses it**: Yes — the method is still about **adaptive allocation of a sparse rotation budget inside rotation-based PEFT**. The revision makes that story cleaner by removing the full-width diagonal ambiguity, simplifying allocation to a single claim-bearing event, and keeping the main scope on square `q_proj + o_proj` layers.
- **Reviewer suggestions rejected as drift**:
  - Treating `OFT`/`BOFT` as main-claim blockers was rejected. They are useful orthogonal-family breadth references, but the closest comparator for the core mechanism remains `qGOFT`.
  - Broadening the claim back toward “adaptive PEFT selection” was rejected. The paper is still about adaptive slot allocation **within rotation-based PEFT**, not adaptive PEFT broadly.

---

## Simplicity Check

- **Dominant contribution after revision**: One mechanism: **one-time adaptive allocation of sparse bilateral rotation slots, followed by fixed-support optimization with a diagonal core restricted to the same selected dimensions**.
- **Components removed or merged**:
  - Full-width `D_diag[d]` is removed from the main method and replaced by **selected-support `D_sel`** only on dimensions chosen by the slot allocator.
  - Repeated reseating is removed from the main method; the default method now uses **single allocation after a short statistics warmup**.
  - `OER` is demoted to an **appendix-only optional stabilizer** unless a stable multi-seed gain justifies promotion.
  - `DoRA`, `OFT`, and `BOFT` are moved out of the minimal claim-bearing package; they remain appendix breadth references only if low-overhead.
- **Reviewer suggestions rejected as unnecessary complexity**:
  - Keeping both repeated reseating and single-allocation as co-equal main variants.
  - Using a shared-codepath static-slot ablation and calling it `qGOFT` without a faithful reproduction of the published schedule.
- **Why the remaining mechanism is still the smallest adequate route**: The paper now changes only two things relative to static orthogonal PEFT: **where the sparse rotation budget is placed**, and **which coordinates are allowed to carry the diagonal additive core**. Everything else is frozen, linear, and mergeable.

---

## Changes Made

### 1. Remove the full-width diagonal ambiguity
- **Reviewer said**: “Add a `D_diag-only` control or constrain `D_diag` to selected dimensions only. Without this, the main claim is under-isolated.”
- **Action**: The main method now replaces full-width `D_diag[d]` with **`D_sel` supported only on the selected dimensions**:
  - `U = dims(slots_R) ∪ dims(slots_L)`
  - `D_sel(U, d_sel) = Diag(v)` where `v_u` is trainable only for `u ∈ U`, and `v_u = 0` otherwise.
  - A `Diag-only-selected` control is added to the validation package to directly test whether the diagonal support alone explains the gain.
- **Reasoning**: This ties the additive core to the same sparse support as the adaptive rotation mechanism, instead of giving the method a hidden full-width diagonal adapter.
- **Impact on core method**: Improves contribution quality and makes the mechanism read as “adaptive sparse rotation allocation” rather than “diagonal adapter + sparse rotations.”

### 2. Separate the faithful prior-art baseline from the internal static ablation
- **Reviewer said**: “If `qGOFT` is instantiated as fixed random slots, that is not a faithful published static orthogonal baseline unless the paper really does that.”
- **Action**:
  - `qGOFT` in the main comparison now means **a faithful reimplementation of the published static orthogonal baseline**, not merely a shared-codepath fixed-slot variant.
  - The shared-codepath static comparison is renamed **`fixed-slot JORA`** and is treated only as an internal ablation that isolates adaptive allocation.
- **Reasoning**: This avoids conflating prior art with an in-house simplification.
- **Impact on core method**: Improves venue readiness and makes the baseline story reviewer-safe.

### 3. Make the main method single-allocation, not repeated reseating
- **Reviewer said**: “Decide whether the simplest successful variant is `single-allocation` or `repeated burn-in reseating`, and make that the main method.”
- **Action**: The main paper now uses **a short statistics warmup followed by a single allocation event**. Repeated reseating is demoted to an appendix-only simplification check if it materially helps.
- **Reasoning**: The paper’s novelty is the adaptive allocation itself, not repeated topology churn. A single allocation is easier to implement, easier to analyze, and easier to validate.
- **Impact on core method**: Improves simplicity, feasibility, and validation focus.

### 4. Tighten the validation package around two claim-bearing blocks
- **Reviewer said**: Validation focus dropped in round 4.
- **Action**: The proposal now has **only two claim-bearing experiment blocks**:
  1. Pareto frontier vs `LoRA`
  2. Closest static orthogonal comparison vs faithful `qGOFT`

  Everything else is moved into one compact **mechanism-isolation diagnostic block**:
  - `fixed-slot JORA`
  - `Diag-only-selected`
  - optional `OER`
  - optional repeated reseating
  - appendix `OFT/BOFT/DoRA`
- **Reasoning**: This keeps the validation tied directly to the paper’s two real questions: “Is the frontier better than LoRA?” and “Does adaptive allocation beat static orthogonal PEFT?”
- **Impact on core method**: Improves validation focus without weakening the scientific story.

### 5. Make parameter fairness explicit
- **Reviewer said**: “Report parameter counts including `D_diag` and optional `e`.”
- **Action**: The revised proposal now explicitly requires exact trainable parameter counts for every method, including:
  - selected-support diagonal parameters `|U|`
  - rotation angles `K_R + K_L`
  - optional `e[d]` if `OER` is enabled
- **Reasoning**: The fairness story depends on counting all trainable parameters, especially after removing the full-width diagonal from the mainline method.
- **Impact on core method**: Strengthens the Pareto claim and preempts reviewer concerns about hidden capacity.

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

`qGOFT` gives rotation-based PEFT a strong geometric primitive, but its rotation budget is **static**: the rotation structure is chosen once by a fixed schedule and then trained. `LoRA` allocates low-rank capacity without rotation structure and allows norm drift.

The clean gap is therefore narrow: **within rotation-based PEFT, there is still no method that uses early task statistics to allocate a fixed sparse bilateral rotation budget, then freezes that sparse support for stable optimization**.

The round-4 review exposed one remaining ambiguity: if the diagonal core is full-width, the method risks becoming “diagonal adapter + sparse rotations,” which weakens the paper’s actual claim. So the main method should not use a full-width diagonal core. The smallest adequate route is:

1. **single adaptive slot allocation** from early statistics,
2. **fixed sparse rotation support** after allocation,
3. **a diagonal core restricted to the same selected support**.

This keeps the paper centered on adaptive sparse rotation allocation rather than on hidden full-width additive capacity.

## Method Thesis

JORA is a square-layer, mergeable additive adapter that first runs a short statistics warmup to score input and output dimensions, then allocates **one fixed set of sparse bilateral rotation slots** and trains only the associated rotation angles plus a **selected-support diagonal core**. The method changes **where** a fixed sparse rotation budget goes, not the underlying rotation primitive.

This is the smallest adequate intervention because:
- the prior-art gap is about **static vs adaptive allocation**,
- the diagonal core is now explicitly tied to the same selected support,
- the mainline method avoids repeated reseating, full-width diagonals, tanh, and rectangular operators.

**Honest framing**: JORA computes

```text
delta = R_L^T · D_sel · R_R · x
out   = base_out + delta
```

for the mainline method. It is a rotation-basis structured additive adapter, not a direct orthogonal transformation of `W₀`.

## Contribution Focus

- **Single dominant contribution**: One-time adaptive allocation of sparse bilateral rotation slots within rotation-based PEFT, followed by fixed-support optimization on square layers.
- **Appendix-only supporting design element**: `OER` magnitude stabilization, retained only if a stable multi-seed gain justifies it.
- **Explicit non-contributions**: full-width diagonal cores, repeated reseating as the main method, broad “adaptive PEFT” claims, rectangular-layer main results, tanh-based merge path, and treating `OFT/BOFT` as main-claim blockers.

## Proposed Method

### Complexity Budget
- **Frozen**: Pretrained `W₀`, model backbone, tokenizer, PEFT infrastructure.
- **New trainable parameters (mainline)**:
  - `θ_R [K_R]`: right-rotation angles for frozen input-side slot pairs
  - `θ_L [K_L]`: left-rotation angles for frozen output-side slot pairs
  - `d_sel [|U|]`: diagonal-core values only on the selected support `U = dims(slots_R) ∪ dims(slots_L)` with `|U| ≤ 4K`
- **Optional appendix-only trainable parameters**:
  - `e [d]`: `OER` logits if the stabilizer is retained
- **Structural state (non-trainable after allocation)**:
  - `slots_R = {(i_r, j_r)}_{r=1}^{K_R}`
  - `slots_L = {(i_l, j_l)}_{l=1}^{K_L}`
  - `U = dims(slots_R) ∪ dims(slots_L)`
- **Intentionally excluded from the main paper**: full-width `D_diag[d]`, repeated reseating as default, rectangular operators, `k/v` under GQA, MLP projections, tanh merge variants.

### System Overview

Main-paper target modules: **square attention projections only (`q_proj`, `o_proj`)**.

Core claim baselines use the **same module set**:
- `LoRA-r2`, `LoRA-r4` for the Pareto frontier
- faithful `qGOFT` reimplementation for the closest static orthogonal baseline

Non-core references are appendix-only if low-overhead:
- `fixed-slot JORA` as an internal static-allocation ablation
- `Diag-only-selected` as a diagonal-core control
- `DoRA`, `OFT`, `BOFT` as breadth references
- `OER` and repeated reseating only if screening justifies them

For a square layer of width `d` and input `x ∈ ℝ^{B × d}`:

```text
base_out = W₀ · x

# statistics warmup (no slot changes yet)
ema_in[j]  ← β · ema_in[j]  + (1-β) · mean_batch(x_j²)
ema_out[i] ← β · ema_out[i] + (1-β) · mean_batch((∂L/∂y_i)²)

# single allocation after warmup
slots_R ← deterministic_pairs(top_2K(ema_in))
slots_L ← deterministic_pairs(top_2K(ema_out))
U       ← dims(slots_R) ∪ dims(slots_L)

# fixed-support training
x̃ = R_R(slots_R, θ_R) · x
x̂ = D_sel(U, d_sel) · x̃
δ  = R_L^T(slots_L, θ_L) · x̂

out = base_out + δ
```

where `D_sel(U, d_sel) = Diag(v)` and

```text
v_u = trainable(d_sel[u])   if u ∈ U
v_u = 0                     otherwise
```

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
This is the task-coupled learning signal, computed from the layer-output gradient during the statistics warmup.

Why this formulation is simpler than the previous burn-in design:
- it avoids needing repeated reseating to expose a temporary delta path,
- it still measures which output dimensions carry the strongest task signal,
- once the adapter is allocated, `∂L/∂δ_i` and `∂L/∂y_i` coincide for the additive path.

#### 2. Deterministic disjoint pairing rule

For either score vector (`ema_in` or `ema_out`):
1. Stable-sort dimensions by **descending score**; ties are broken by **smaller dimension index first**.
2. Keep the first `2K` dimensions from that ordered list.
3. Form pairs consecutively: `(d₁, d₂), (d₃, d₄), …, (d₂K₋₁, d₂K)`.
4. Store each pair canonically as `(min(d_a, d_b), max(d_a, d_b))`.

This rule is deterministic, disjoint, and fully reproducible.

#### 3. Single allocation and fixed-support diagonal core

After the statistics warmup:

```text
slots_R ← deterministic_pairs(top_2K(ema_in))
slots_L ← deterministic_pairs(top_2K(ema_out))
U       ← dims(slots_R) ∪ dims(slots_L)
```

The diagonal core is then restricted to `U` only:

```text
D_sel(U, d_sel) = Diag(v),  v_u = d_u if u ∈ U else 0
```

So the method cannot fall back to a full-width diagonal adapter. The additive capacity is explicitly tied to the selected rotation support.

#### 4. Fixed-support optimization

After allocation:
- `slots_R`, `slots_L`, and `U` are frozen permanently
- only `θ_R`, `θ_L`, and `d_sel` are trained in the mainline method

Repeated reseating is **not** part of the main paper. It is an appendix-only simplification check if screening shows a clear benefit.

### Optional Supporting Component: Appendix-Only OER Stabilization

If a 3-seed screening run shows a stable gain, the paper may optionally include:

```text
s_i = softmax(e_i / τ) · E₀ / ||w_i^0||₂,
where E₀ = (Σ_i ||w_i^0||₂²)^{1/2}
```

and use

```text
out = base_out + s ⊙ δ
```

But `OER` is **not** part of the main novelty story and is not required for the core claim-bearing experiments. If the gain is weak or unstable, it stays in the appendix.

### Mergeability

For the mainline method:

```text
W_merged = W₀ + R_L^T · D_sel · R_R
```

This is an exact linear merge for the square-layer main method.

If `OER` is retained in the appendix variant:

```text
W_merged = W₀ + Diag(s) · R_L^T · D_sel · R_R
```

### Training Plan
1. **Backbone / data**: Mistral-7B-v0.1, Alpaca-cleaned 52k, standard SFT.
2. **Target modules (main paper)**: `q_proj`, `o_proj` only.
3. **Statistics warmup**: run a short calibration phase of `T_stat = min(200, 0.05 · T_total)` steps to accumulate `ema_in` and `ema_out`.
4. **Single allocation**: allocate `slots_R`, `slots_L`, and `U` once at `t = T_stat`; initialize `θ_R`, `θ_L`, and `d_sel` to zero.
5. **Main optimization**: train only `θ_R`, `θ_L`, and `d_sel` for the remaining steps.
6. **Model sizes**:
   - `JORA-small`: `K=2` per side
   - `JORA-base`: `K=4` per side
7. **Parameter accounting**:
   - per-layer `JORA-core`: `K_R + K_L + |U|`
   - per-layer `JORA+OER`: `K_R + K_L + |U| + d`
   - all paper tables report **exact total trainable parameter counts** across every adapted layer, including optional `e` if used.
8. **Appendix-only variants**: `OER`, repeated reseating, `DoRA`, `OFT`, `BOFT` only if they do not disrupt the core experiment package.

### Failure Modes and Diagnostics
- **Statistics warmup is too noisy**: selected supports vary strongly across adjacent warmup windows. Detect by Jaccard overlap of top-`2K` sets across checkpoints. Mitigation: slightly increase `T_stat` or test one appendix-only reseat.
- **Selected-support diagonal still explains most of the gain**: detect with `Diag-only-selected`. If it nearly matches full JORA, the claim must narrow accordingly.
- **Faithful qGOFT reproduction is not ready**: do not rename the shared-codepath static ablation as `qGOFT`; keep it as `fixed-slot JORA` and treat `OFT/BOFT` only as appendix breadth references.
- **OER gain is unstable**: keep `OER` out of the main paper and report only the simpler no-OER method.

### Novelty and Elegance Argument

- **vs. qGOFT**: JORA changes one thing that matters for the paper — it allocates a sparse bilateral rotation budget using early task statistics, then freezes that support. The closest published baseline remains static.
- **vs. fixed-slot JORA**: same parameterization, same sparse support size, different allocation policy. This isolates the value of adaptive allocation rather than conflating it with other implementation changes.
- **vs. LoRA**: JORA keeps a rotation-structured sparse adapter instead of a free-form low-rank update.
- **vs. OFT / BOFT**: they are useful orthogonal-family references, but they test fixed topology breadth, not the core static-vs-adaptive sparse-slot question. So they belong in the appendix unless they are nearly free to run.
- **Why this is focused**: the mainline paper now has one mechanism, one default method variant, one closest prior-art baseline, and one compact diagnostic block.

## Claim-Driven Validation Sketch

**Validation-focus rule for this round**: only **two experiment blocks are claim-bearing**. Everything else is diagnostic or appendix.

### Claim 1 (Primary): Better Pareto frontier than LoRA on the same square-layer module set
- **Minimal experiment**: `JORA-small`, `JORA-base` vs. `LoRA-r2`, `LoRA-r4`, all on `q_proj + o_proj` only.
- **Metric**: average accuracy on MMLU / ARC-C / GSM8K vs **exact trainable parameter count**.
- **Expected evidence**: `JORA-base` matches `LoRA-r4` at lower parameter count; `JORA-small` clearly beats `LoRA-r2` at comparable or lower budget.
- **Seed plan**: 3 seeds for `JORA-base` and `LoRA-r4`; 2 seeds for `JORA-small` and `LoRA-r2`, expanded only if frontier decisions hinge on them.

### Claim 2 (Closest prior): JORA beats the closest static orthogonal baseline at equal budget
- **Minimal experiment**: `JORA-base` vs. faithful `qGOFT` reimplementation on the same `q_proj + o_proj` scope, matched by exact trainable parameter count as closely as the published parameterization allows.
- **Metric**: average MMLU / ARC-C / GSM8K at matched budget.
- **Expected evidence**: `JORA-base ≥ qGOFT` with a consistent multi-seed margin.
- **Seed plan**: 3 seeds each for `JORA-base` and faithful `qGOFT`.
- **Important naming rule**: if only the shared-codepath static-slot implementation is available, label it `fixed-slot JORA`, not `qGOFT`.

### Mechanism-Isolation Diagnostics (Not Separate Paper-Level Claims)
- **`fixed-slot JORA`**: same codepath and same support size, but static slots. Purpose: isolate adaptive allocation from all other factors.
- **`Diag-only-selected`**: same selected support `U`, but no rotations. Purpose: test whether selected-dimension diagonal capacity alone explains the gain.
- **Optional appendix checks only if needed**:
  - repeated reseating vs single allocation
  - `OER` vs no `OER`
  - `DoRA`, `OFT`, `BOFT` breadth references
- **Seed plan**: 2 seeds each by default; expand only if the main claim interpretation depends on them.

## Experiment Handoff Inputs
- **Must-prove claims**:
  1. Pareto gain vs `LoRA` on square-layer scope
  2. Improvement over faithful `qGOFT` at equal budget
- **Must-run mechanism diagnostics**:
  - `fixed-slot JORA`
  - `Diag-only-selected`
- **Appendix-only if low-overhead**:
  - `DoRA-r4`
  - `OFT`
  - `BOFT`
  - repeated reseating
  - `OER`
- **Critical datasets / metrics**:
  - MMLU, ARC-C, GSM8K
  - exact trainable parameter counts including all selected-core parameters and optional `e`
  - mean ± std over seeds
- **Highest-risk assumptions**:
  1. faithful `qGOFT` can be reproduced cleanly on the same scope
  2. selected-support `D_sel` is strong enough without reverting to full-width diagonal capacity
  3. short statistics warmup is sufficient for stable allocation
  4. `OER` is not necessary for the mainline story

## Compute & Timeline
- **Estimated compute**: ~180 GPU-hours on 3× RTX 4090.
- **Why still feasible**: the main experiment package now has only two claim-bearing blocks, one compact diagnostic block, square-layer scope only, and a single-allocation mainline method.
- **Timeline**:
  - single-allocation implementation + merge-path verification: 2 days
  - faithful `qGOFT` parity validation: 1–2 days
  - 1-seed screening for main methods + diagnostics: 2 days
  - multi-seed core runs: 6 days
  - analysis + paper-facing tables/plots: 2 days
