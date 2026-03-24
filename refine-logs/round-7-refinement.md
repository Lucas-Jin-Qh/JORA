# Round 7 Refinement

## Problem Anchor
*(Verbatim from round 0, with targeted corrections — see Anchor Check)*

- **Bottom-line problem**: PEFT methods must achieve a better accuracy-vs-parameter Pareto frontier than LoRA for LLM fine-tuning, using a principled, structure-preserving adaptation mechanism.
- **Must-solve bottleneck**: Orthogonal PEFT (qGOFT, OFT) uses static rotation structures — all dimension pairs adapted uniformly, wasting capacity on low-signal directions. Low-rank PEFT (LoRA, DoRA) lacks geometric bias and suffers norm drift. Neither achieves an efficient adaptive sparse rotation.
- **Non-goals**: Not a new backbone, not replacing LoRA for all use cases, not targeting inference latency.
- **Constraints**: 3× RTX 4090, 2-week experiment window. PEFT library. NeurIPS 2026.
- **Success condition**: JORA in the extreme-budget regime (~3K–25K total params on Mistral-7B q+o scope) beats LoRA-r1 (must-win); is competitive with LoRA-r2 within 2pp average (primary); optionally matches LoRA-r2 (stretch). Beats faithful qGOFT at equal budget.

---

## Anchor Check

- **Original bottleneck**: Static rotation structure in orthogonal PEFT wastes capacity; LoRA lacks geometric bias.
- **Why the revised method still addresses it**: Yes. Unchanged. This round addresses notation inconsistency in the operator definition and warmup budget fairness — neither changes the mechanism.
- **Reviewer suggestions rejected as drift**: None. All action items are valid and addressed.

---

## Simplicity Check

- **Dominant contribution after revision**: One mechanism: one-time adaptive allocation of sparse bilateral rotation slots within rotation-based PEFT, with a correctly defined residualized zero-function-change operator and a clearly specified pre-training calibration warmup phase.
- **Components removed or merged**:
  - `diag(1+δ)_U` overloaded notation: replaced everywhere with canonical `D_sel = P_U + diag(δ)_U`.
  - Mid-training warmup ambiguity: resolved. Warmup is now a **pre-training calibration phase** (separate from the main optimizer budget), not part of the training steps counted in comparisons.
- **Reviewer suggestions that would add complexity**:
  - "Fallback to left-side only": not adopted. The bilateral structure is part of the method's identity. It is already framed honestly as asymmetric (gradient-guided left, activation-proxy right). Dropping right-side allocation would change the mechanism unnecessarily.
- **Why the remaining mechanism is still the smallest adequate route**: One operator, one warmup phase, one allocation, one training phase. No new modules.

---

## Changes Made

### 1. Unify operator notation to D_sel = P_U + diag(δ)_U (IMPORTANT)

- **Reviewer said**: "`diag(1+δ)_U` is inconsistent — in one place it acts like `I_U` at init; in another it acts differently. Define once as `D_sel = P_U + diag(δ)_U`."
- **Action**: Adopted. The canonical operator is now `D_sel = P_U + diag(δ)_U` where:
  - `P_U = diag(1_U)` is a fixed projection (ones on U, zeros elsewhere) — **non-trainable**
  - `diag(δ)_U` is the trainable diagonal correction (zeros on U at init, zeros elsewhere always)
  - `D_sel · x = P_U · x + diag(δ)_U · x`
  - At δ=0: `D_sel · x = P_U · x`
  - The full operator: `D_sel = P_U + diag(δ)_U`
- **Forward pass**:
  ```
  x̃    = R_R · x
  x̂    = D_sel · x̃  =  P_U · x̃  +  diag(δ)_U · x̃
  delta = R_L^T · x̂  −  P_U · x
  out   = base_out + delta
  ```
- **At init** (θ=0, δ=0): `R_R=R_L=I`, `D_sel=P_U`, so `delta = R_L^T · P_U · x − P_U · x = P_U·x − P_U·x = 0` ✓
- **Gradients at init**:
  - `∂delta/∂δ_u = [R_L^T]_u · [∂D_sel/∂δ_u · R_R · x] = x_u ≠ 0` for u∈U ✓
  - `∂delta/∂θ_R` involves `R_L^T · D_sel` evaluated at `R_L=I`, `D_sel=P_U`, giving `P_U · ∂R_R/∂θ · x ≠ 0` ✓
- **Mergeability**: `W_merged = W₀ + R_L^T · D_sel · R_R − P_U` — exact linear merge.
- **Reasoning**: The `D_sel = P_U + diag(δ)_U` form makes the zero-at-init and gradient-live properties self-evident from the definition, without requiring supplementary explanation.
- **Impact**: Method Specificity improves; the notation is now unambiguous throughout the proposal.

### 2. Specify warmup as pre-training calibration phase (IMPORTANT)

- **Reviewer said**: "Must decide whether warmup is a pre-training calibration pass or consumes training budget. Keep consistent across comparisons."
- **Action**: Warmup is defined as a **pre-training calibration phase**:
  - Duration: `T_stat = min(200, 0.05·T_total)` forward+backward passes over the training data, using the base model only (no adapter).
  - These passes are **not counted in the training budget** for comparison purposes.
  - All baselines (LoRA, qGOFT, fixed-slot JORA) receive the same total number of optimizer update steps; JORA additionally has a calibration phase before those steps begin.
  - This is reported transparently in the paper: "JORA uses T_stat calibration steps to allocate slot structure; all methods are then trained for the same N optimizer steps."
- **Reasoning**: This removes the mid-training switchover story entirely (cleaner narrative) and makes all comparisons fair on optimizer step count.
- **Impact**: Eliminates the fairness ambiguity; simplifies the training narrative.

### 3. Confirm residualized init as correctness property, not second contribution (IMPORTANT)

- **Reviewer said**: "Residualized init is a correct implementation fix, not a headline novelty. Do not sell it as a second contribution."
- **Action**: The residualized init is now framed entirely as a correctness property: "For the adapter to be introduced at an interior point without disrupting the pretrained function, the forward pass must be designed so that delta=0 at allocation time." This is one sentence in the method section, not a claim. The dominant contribution remains adaptive slot allocation.
- **Impact**: Contribution Quality is not diluted; Method Specificity improves.

### 4. Confirm empirical target framing (IMPORTANT)

- **Reviewer said**: "`~12K within 2pp of LoRA-r2` is plausible as a stretchy primary claim, not a default assumption."
- **Action**: The paper's success language is now:
  - Must-win: JORA-base beats LoRA-r1 (expected; this is the core Pareto claim)
  - Primary headline: JORA-base within 2pp of LoRA-r2 on average (stretchy; reported if achieved)
  - Stretch: JORA-base ≥ LoRA-r2 (aspirational; reported if achieved)
  - If only must-win is achieved: paper is still publishable as an extreme-budget Pareto method with an honest feasibility story.
- **Impact**: Feasibility score recovers; honest framing avoids reviewer overstatement objections.

---

## Revised Proposal

# Research Proposal: JORA — Adaptive Sparse Rotation-Slot Allocation for Extreme-Budget Square-Layer PEFT

## Problem Anchor

- **Bottom-line problem**: PEFT methods must achieve a better accuracy-vs-parameter Pareto frontier than LoRA for LLM fine-tuning, using a principled, structure-preserving adaptation mechanism.
- **Must-solve bottleneck**: Orthogonal PEFT (qGOFT, OFT) uses static rotation structures — all dimension pairs adapted uniformly, wasting capacity on low-signal directions. Low-rank PEFT (LoRA, DoRA) lacks geometric bias and suffers norm drift. Neither achieves an efficient adaptive sparse rotation.
- **Non-goals**: Not a new backbone, not replacing LoRA for all use cases, not targeting inference latency.
- **Constraints**: 3× RTX 4090, 2-week experiment window. PEFT library. NeurIPS 2026.
- **Success condition**: JORA in the extreme-budget regime (~3K–25K total params on Mistral-7B q+o scope) beats LoRA-r1 as a must-win; is competitive with LoRA-r2 within 2pp (primary headline if achieved); matches LoRA-r2 (stretch if achieved). Beats faithful qGOFT at equal budget.

## Technical Gap

`qGOFT` gives rotation-based PEFT a strong geometric primitive, but its rotation budget is **static**. `LoRA` allocates low-rank capacity without rotation structure. The gap: **within rotation-based PEFT, no method uses early task statistics to allocate a fixed sparse bilateral rotation budget and freezes that support for stable optimization with a zero-function-change initialization**.

## Method Thesis

JORA is a square-layer, mergeable additive adapter that runs a short pre-training calibration phase to score input and output dimensions, allocates **one fixed set of sparse bilateral rotation slots**, and then trains Givens angles plus a residualized selected-support diagonal correction. The allocation is one-time; support is frozen for the remainder of training.

**Core operator** — defined once, used everywhere:

```
D_sel = P_U + diag(δ)_U
```

where:
- `P_U = diag(1_U)` — fixed projection onto U (non-trainable, ones on U, zeros elsewhere)
- `diag(δ)_U` — trainable diagonal correction on U, **initialized to zeros** (zeros elsewhere always)

**Forward pass**:
```
x̃    = R_R(slots_R, θ_R) · x
x̂    = D_sel · x̃  =  P_U · x̃  +  diag(δ)_U · x̃
delta = R_L^T(slots_L, θ_L) · x̂  −  P_U · x        ← residualized
out   = base_out + delta
```

At init (θ=0, δ=0): `delta = P_U·x − P_U·x = 0` — zero function change at allocation. Gradients w.r.t. δ and θ are nonzero for active dimensions — all parameters gradient-live from step T_stat.

**Mergeability**: `W_merged = W₀ + R_L^T · D_sel · R_R − P_U` — exact, zero inference overhead.

**Parameter scale** (Mistral-7B, 32 layers, q+o scope):
- JORA-small K=8:  2 × 32 × (8 + 8 + 32)    =  3,072 params  (0.29% of LoRA-r1)
- JORA-base  K=32: 2 × 32 × (32 + 32 + 128)  = 12,288 params  (2.34% of LoRA-r1)
- LoRA-r1 (q+o):   524,288 params
- LoRA-r2 (q+o):   1,048,576 params

## Contribution Focus

- **Single dominant contribution**: One-time adaptive allocation of sparse bilateral rotation slots within rotation-based PEFT, yielding Pareto-dominant extreme-budget accuracy on square attention layers.
- **Explicit non-contributions**: full-width diagonal cores in main method, repeated reseating, broad "adaptive PEFT" claims, rectangular-layer main results, tanh merge path, OER.

## Proposed Method

### Complexity Budget
- **Frozen**: W₀, backbone, tokenizer, PEFT infrastructure; P_U (allocated once, then frozen).
- **New trainable parameters**:
  - `θ_R [K_R]`: Givens angles, right side, init 0
  - `θ_L [K_L]`: Givens angles, left side, init 0
  - `δ [|U|]`: diagonal correction on U, init 0
- **JORA-base (K=32)**: 32 + 32 + 128 = 192 params per module × 2 × 32 layers = 12,288 params total
- **Excluded**: full-width D_diag, repeated reseating, rectangular operators, k/v GQA, MLP, tanh, OER.

### System Overview

Target modules: **q_proj, o_proj only** (square, d=4096 on Mistral-7B).

```
# ── PRE-TRAINING CALIBRATION PHASE (T_stat steps, no optimizer updates) ──────
# Base model only; backward for gradient statistics.
for t in 1 … T_stat:
    forward(base_model, batch)
    backward(loss, retain_graph=True)
    ema_in[j]  ← β·ema_in[j]  + (1-β)·mean_batch(x_j²)       # activation proxy
    ema_out[i] ← β·ema_out[i] + (1-β)·mean_batch((∂L/∂y_i)²) # gradient signal

# ── STABILITY CHECK ───────────────────────────────────────────────────────────
if mean_jaccard(top_2K sets, last 5 steps) < 0.85:
    extend T_stat by 50 steps (once)

# ── SINGLE ALLOCATION ─────────────────────────────────────────────────────────
slots_R ← deterministic_pairs(top_2K(ema_in))
slots_L ← deterministic_pairs(top_2K(ema_out))
U       ← dims(slots_R) ∪ dims(slots_L)   # |U| ≤ 4K
P_U     ← diag(1_U)                        # fixed, non-trainable
δ       ← zeros(|U|)                        # trainable correction
θ_R, θ_L ← zeros(K)                        # Givens angles

# ── MAIN TRAINING (N optimizer steps, same as all baselines) ──────────────────
D_sel = P_U + diag(δ)_U
x̃     = R_R(slots_R, θ_R) · x
x̂     = D_sel · x̃
delta  = R_L^T(slots_L, θ_L) · x̂  −  P_U · x
out    = base_out + delta
```

All baselines train for exactly N optimizer update steps. JORA additionally uses T_stat calibration steps (reported transparently in the paper).

### Core Mechanism

#### Operator definition

`D_sel = P_U + diag(δ)_U`

- `P_U`: fixed projection, ones on selected dims U, zeros elsewhere.
- `diag(δ)_U`: trainable correction, zeros at init, nonzero after training.
- At δ=0: `D_sel = P_U` (pure projection onto U).
- At δ≠0: `D_sel · x_u = (1 + δ_u) · x_u` for u∈U; `D_sel · x_u = 0` for u∉U.

#### Zero-function-change at allocation

```
init: θ_R = θ_L = 0  →  R_R = R_L = I
      δ = 0           →  D_sel = P_U

Forward:
  x̃ = I · x = x
  x̂ = P_U · x
  R_L^T · x̂ = I · P_U · x = P_U · x
  delta = P_U·x − P_U·x = 0  ✓
```

#### Gradient-live at allocation

```
∂delta/∂δ_u = ∂/∂δ_u [R_L^T · D_sel · R_R · x]
             = [R_L^T]_u · [R_R · x]_u  |_{θ=0,δ=0}
             = x_u  (nonzero for active u∈U)  ✓

∂delta/∂θ_R  ∝  R_L^T · D_sel · ∂R_R/∂θ · x  |_{θ=0,δ=0}
              =  P_U · ∂R_R/∂θ · x  (nonzero for active input dims)  ✓
```

#### Asymmetric bilateral statistics (honest framing)

Left side (output) — **gradient signal** (task-coupled):
```
ema_out[i] ← β·ema_out[i] + (1-β)·mean_batch((∂L/∂y_i)²)
```

Right side (input) — **activation proxy** (cheap, no task coupling):
```
ema_in[j] ← β·ema_in[j] + (1-β)·mean_batch(x_j²)
```

The selection is intentionally asymmetric. The paper presents this as "gradient-guided output allocation + activation-proxy input allocation," not symmetric task-driven bilateral selection.

#### Deterministic disjoint pairing rule

1. Stable-sort dims by descending score; ties broken by smaller index first.
2. Take top 2K dims.
3. Pair consecutively: (d₁,d₂), …, (d_{2K-1},d_{2K}).
4. Canonicalize: (min(a,b), max(a,b)).

#### Precommitted Diag-only-selected interpretation rule

Diag-only-selected: same U, no rotations (R_R=R_L=I throughout), only δ trained.

- If JORA − Diag-only-selected > 0.5pp on all three benchmarks: rotation contribution supported; dominant contribution stands.
- If JORA − Diag-only-selected ≤ 0.5pp: rotation contribution marginal; paper narrows to "adaptive sparse diagonal scaling with rotation-structured support selection." Still publishable.

### Training Plan

1. **Backbone / data**: Mistral-7B-v0.1, Alpaca-cleaned 52k, standard SFT.
2. **Target modules**: q_proj, o_proj (32 layers each).
3. **Calibration phase**: T_stat = min(200, 0.05·T_total) forward+backward steps with base model only. Accumulate ema_in and ema_out. β=0.99. These steps are NOT counted in the training budget for comparisons.
4. **Stability check**: Jaccard similarity of top-2K sets over last 5 calibration steps. Extend by 50 if < 0.85.
5. **Allocation**: deterministic pairs; init δ=0, θ=0, P_U=fixed.
6. **Main training**: N optimizer steps (same as all baselines). AdamW, cosine decay, JORA-specific LR (tuned separately; expected to be higher than LoRA LR).
7. **Variants (main paper)**:
   - JORA-small: K=8, ~3K params
   - JORA-base:  K=32, ~12K params
8. **Parameter counts** (reported in every table):
   - JORA-small: 3,072 params
   - JORA-base:  12,288 params
   - LoRA-r1:    524,288 params
   - LoRA-r2:    1,048,576 params

### Failure Modes and Diagnostics

- **Warmup unstable (Jaccard < 0.85 after extension)**: report instability; increase β; flag as limitation.
- **δ explodes**: L2 penalty or clip; detect via max|δ|.
- **Diag-only-selected ≈ JORA**: narrow claim per precommitted rule.
- **qGOFT faithful reproduction fails**: use fixed-slot JORA as static baseline; label clearly.

### Novelty and Elegance Argument

- **vs. qGOFT**: same rotation primitive, adaptive vs. static allocation.
- **vs. fixed-slot JORA**: same codepath, same budget, adaptive vs. random allocation — isolates allocation value cleanly.
- **vs. LoRA**: rotation-structured adapter at 2–3% of LoRA-r1 parameter count.
- **Why the residualized init matters (correctness, not novelty)**: any adapter introduced at an interior training step requires delta=0 at the switchover point to avoid disrupting the pretrained function. The `D_sel = P_U + diag(δ)_U` formulation achieves this automatically.
- **Focused**: one mechanism, one parameter regime, one clear Pareto story.

## Claim-Driven Validation Sketch

### Claim 1 (Primary): JORA Pareto-dominates LoRA in the extreme-budget regime

- **Experiment**: JORA-small (K=8, ~3K) and JORA-base (K=32, ~12K) vs. LoRA-r1 (~524K), LoRA-r2 (~1M), LoRA-r4 (~2M), all on q_proj + o_proj.
- **Metric**: average MMLU / ARC-C / GSM8K accuracy vs. exact total trainable params.
- **Staged success**:
  - Must-win: JORA-base > LoRA-r1 average
  - Primary (if achieved): JORA-base within 2pp of LoRA-r2
  - Stretch (if achieved): JORA-base ≥ LoRA-r2
- **Seeds**: 3 seeds for JORA-base and LoRA-r2; 2 for others.

### Claim 2: JORA beats faithful qGOFT at equal budget

- **Experiment**: JORA-base vs. faithful qGOFT reimplementation at matched parameter count.
- **Metric**: average MMLU / ARC-C / GSM8K.
- **Fallback**: fixed-slot JORA as static baseline if qGOFT faithful reproduction fails.
- **Seeds**: 3 each.

### Mechanism-Isolation (Mandatory)

- **Diag-only-selected**: same U, no rotations, only δ. Precommitted interpretation rule applies.
- **fixed-slot JORA**: same codepath, random slots. Tests allocation policy value.
- **Support stability curves**: Jaccard over calibration phase. Justifies one-shot allocation.
- **Seeds**: 2 each.

### Appendix-Only

- DoRA, OFT, BOFT breadth
- JORA-large (K=64) if run
- Calibration length sensitivity (T_stat = 100, 200, 400)
- Non-residualized baseline (d_sel=1, old init) vs. residualized (δ=0) — shows the function-jump matters

## Experiment Handoff Inputs

- **Must-prove**: (1) JORA-base beats LoRA-r1; (2) JORA-base beats fixed-slot JORA or qGOFT at equal budget.
- **Mandatory diagnostics**: Diag-only-selected (precommitted rule), fixed-slot JORA, Jaccard curves.
- **Critical metrics**: MMLU, ARC-C, GSM8K; exact param counts; mean ± std.
- **Highest-risk assumptions**:
  1. JORA-base (~12K) beats LoRA-r1 (~524K) — core empirical bet
  2. qGOFT reproducible on same scope (or fixed-slot JORA fallback)
  3. δ well-conditioned; residualization numerically stable
  4. Diag-only-selected does NOT match JORA — else rotation claim is downgraded

## Compute & Timeline

- **Estimated compute**: ~150 GPU-hours, 3× RTX 4090.
- **Timeline**:
  - `D_sel = P_U + diag(δ)_U` implementation + calibration phase separation: 1 day
  - Faithful qGOFT or fixed-slot JORA fallback: 1–2 days
  - 1-seed screening (all methods + diagnostics): 2 days
  - Multi-seed core runs: 5 days
  - Analysis + paper tables: 2 days
