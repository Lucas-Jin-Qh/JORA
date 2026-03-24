# Round 1 Refinement

## Problem Anchor
*(Verbatim from round 0)*

- **Bottom-line problem**: PEFT methods must achieve a better accuracy-vs-parameter Pareto frontier than LoRA for LLM fine-tuning, using a principled, structure-preserving adaptation mechanism.
- **Must-solve bottleneck**: Current orthogonal PEFT methods (OFT, qGOFT) use static rotation structures — all dimensions adapted uniformly, capacity wasted on unimportant directions. Current low-rank PEFT (LoRA, DoRA) lacks geometric inductive bias and suffers from norm drift. Neither achieves an efficient sparse-vs-dense rotation tradeoff.
- **Non-goals**: Not a new backbone architecture, not replacing LoRA for all use cases, not targeting inference latency as the primary metric.
- **Constraints**: 3× RTX 4090 GPUs, 2-week experiment window. PEFT library. NeurIPS 2026.
- **Success condition**: JORA at ~half LoRA-r4 trainable parameter budget matches LoRA-r4 on MMLU/ARC-C/GSM8K benchmarks, clearly beats LoRA-r2, and beats qGOFT at equal budget.

---

## Anchor Check

- **Original bottleneck**: Static rotation structure in orthogonal PEFT methods wastes capacity; LoRA lacks geometric bias.
- **Does the revised method still address it?** Yes — the core mechanism (adaptive sparse bilateral rotations with slot-based pair selection) directly addresses the static-selection gap in qGOFT.
- **Reviewer suggestions rejected as drift**: None. The reviewer's reframe suggestion ("make update act directly on W0") was noted but not pursued — it would shift to a structurally different method (weight-space rotation vs. feature-space rotation). JORA's additive adapter form is honest and valid; we drop the overclaim of "direct orthogonal transformation of W0" and reframe accurately.

---

## Simplicity Check

- **Dominant contribution after revision**: Adaptive sparse bilateral rotation PEFT — slot-based EMA-guided dimension pair allocation with frozen slots after burn-in.
- **Components removed**: BlockCore and LowRankCore removed from main method. OER retained only as implicit stabilization component (not co-headline).
- **Reviewer suggestions rejected as unnecessary complexity**: Keeping OER (it provides clean initialization property and is already implemented; just not promoted as a parallel contribution).
- **Why remaining mechanism is smallest adequate route**: The final method is slot-based bilateral rotations + diagonal core + OER-stabilized magnitude. Three components, each with a clear geometric role and an ablation path.

---

## Changes Made

### 1. Reframe the geometric story (Critical — mechanism drift fix)
- **Reviewer said**: "Deployed form is a structured additive adapter, not a direct sparse orthogonal transformation of W0. Weakens geometry-preserving thesis."
- **Action**: Reframe explicitly. JORA implements a **rotation-basis additive adapter**: delta is computed in a rotated feature basis. The claim is not that W0 itself is orthogonally transformed, but that the delta lives in a rotation-structured subspace. This is honest and still geometrically meaningful. The "geometry-preserving" claim is softened to "geometry-structured."
- **Impact**: Removes overclaim; aligns paper claims with actual mechanism.

### 2. Specify slot-based EMA selection precisely (Critical — method specificity fix)
- **Reviewer said**: "EMA selection is underspecified — pair parameter persistence, reseat frequency, left/right scoring."
- **Action**: Specify exactly:
  - Each layer has **K_L left slots** and **K_R right slots** storing `(i, j, θ)` triplets.
  - Column energy EMA: `e_{col,i}[t] = β · e_{col,i}[t-1] + (1-β) · ||x_{:,i}||²` (right-side dimension importance).
  - Row energy EMA: computed from gradient signal via activation statistics (left-side importance).
  - Top-k greedy disjoint pair selection (already implemented: `select_top_k_pairs_gpu`).
  - **Slot reseat schedule**: reseat every T=50 steps during first 10% of training (burn-in), then freeze slots for the remainder (no more reseating). This prevents parameter discontinuity mid-training.
  - **After burn-in**: only θ values in the frozen slots are trained; pair identities (i, j) are fixed.
- **Impact**: Makes the selection mechanism precisely implementable; eliminates the "where do params go when pairs change" problem.

### 3. Simplify core variants (Important — contribution sprawl fix)
- **Reviewer said**: "Too many parallel contributions. DiagCore + BlockCore + LowRankCore feel co-equal."
- **Action**: Remove BlockCore and LowRankCore from main paper. Only DiagCore in main method. Other variants mentioned briefly in appendix as "configuration options."
- **Impact**: One core type, one story. The method is now: bilateral sparse Givens rotations (with slot-based EMA selection) + DiagCore + OER.

### 4. OER demoted from co-headline to stabilization (Important — contribution focus fix)
- **Reviewer said**: "Center on adaptive sparse pair allocation. OER = stabilization, not co-headline novelty."
- **Action**: OER is retained as an initialization-preserving magnitude calibration mechanism. In the paper, it is presented as "initialization-friendly magnitude stabilization" that prevents norm drift and yields the conservation property. It is NOT a parallel contribution — it is part of the method's design principles.
- **Impact**: Paper now has one dominant contribution (adaptive sparse bilateral rotation PEFT) and one supporting technical detail (OER conservation at init).

### 5. Add two JORA budget points for Pareto story (Important — validation fix)
- **Reviewer said**: "One operating point is not enough for Pareto frontier language."
- **Action**: Add JORA-small (S=8, k=2, DiagCore, ~¼ LoRA-r4 params) and JORA-base (S=16, k=4, DiagCore, ~½ LoRA-r4 params). Report both vs. LoRA-r1/r2/r4 to draw a genuine 2-point Pareto curve from JORA.
- **Impact**: Two JORA points plus LoRA-r1/r2/r4 establishes a genuine efficiency frontier comparison.

### 6. Replace init-only conservation claim with mergeability claim (Important — venue readiness fix)
- **Reviewer said**: "Replace init-only conservation pitch with mergeability + fixed-budget capacity concentration."
- **Action**: Lead with mergeability: after training, `W_merged = W₀ + Diag(s) · R_L^T · D_diag · R_R` (exact for DiagCore, post-burn-in with frozen slots). Conservation proposition moves to appendix as a technical note. Main paper highlights: (1) fully mergeable at inference, (2) slot-based EMA allocation concentrates fixed parameter budget on high-signal dimensions.
- **Impact**: Sharper venue story; mergeability is a concrete deployment advantage.

---

## Revised Proposal

# Research Proposal: JORA — Adaptive Sparse Bilateral Rotation Adapters for LLM Fine-Tuning

## Problem Anchor

- **Bottom-line problem**: PEFT methods must achieve a better accuracy-vs-parameter Pareto frontier than LoRA for LLM fine-tuning, using a principled, structure-preserving adaptation mechanism.
- **Must-solve bottleneck**: Orthogonal PEFT (qGOFT, OFT) uses static rotation structures — all dimension pairs adapted uniformly, wasting capacity on low-signal directions. Low-rank PEFT (LoRA, DoRA) lacks geometric bias and suffers norm drift. Neither achieves an efficient adaptive sparse rotation.
- **Non-goals**: Not a new backbone, not replacing LoRA for all use cases, not targeting inference latency.
- **Constraints**: 3× RTX 4090, 2-week experiment window. PEFT library. NeurIPS 2026.
- **Success condition**: JORA at ~half LoRA-r4 params matches LoRA-r4 on MMLU/ARC-C/GSM8K; clearly beats LoRA-r2; beats qGOFT at equal budget.

## Technical Gap

**Where qGOFT (ICML 2024) and LoRA fall short:**

qGOFT applies Givens rotations to weight matrix rows/columns but uses a **static, uniform** dimension pair schedule. It allocates rotation capacity equally across all selected pairs, regardless of which pairs receive the strongest gradient signal during training. This is analogous to using uniform attention over all tokens — effective but suboptimal when task-relevant signal concentrates on a few dimensions.

LoRA allocates capacity uniformly across r singular directions. No geometric structure; unconstrained scaling leads to norm drift (partially addressed by DoRA, but per-row independently).

**The gap**: No existing PEFT method implements **data-adaptive sparse rotation allocation** — slot-based selection of which dimension pairs to rotate, guided by online gradient energy statistics, with capacity concentrated where the learning signal is highest.

**Smallest adequate intervention**:
1. Bilateral sparse Givens rotations (left + right), with slot-based pair storage
2. EMA-guided burn-in selection: identify the highest-energy dimension pairs during the first 10% of training, then freeze slot assignments
3. DiagCore: element-wise scale between left and right rotations (captures magnitude coupling)
4. OER magnitude: initialization-preserving row-norm calibration (prevents norm drift at init; implicit competitive regularization during training)

## Method Thesis

- **One-sentence thesis**: JORA achieves parameter-efficient fine-tuning by implementing adaptive sparse bilateral Givens rotation adapters — slot-based EMA selection concentrates the rotation budget on the highest-signal dimension pairs, achieving a better Pareto frontier than LoRA and outperforming static-rotation qGOFT at equal budget.
- **Why smallest adequate intervention**: Each component addresses one specific gap — bilateral rotations add input-output coupling that unilateral rotations miss; EMA burn-in selection adds data-adaptivity that qGOFT lacks; DiagCore captures magnitude between rotation stages without over-parameterizing; OER prevents norm drift at initialization.
- **Honest framing**: JORA's delta lives in a rotation-structured subspace: `delta = R_L^T · D_diag · R_R · x`. The additive form `W_merged = W₀ + Diag(s) · R_L^T · D_diag · R_R` is a structured additive adapter, fully mergeable with the pretrained weight.

## Contribution Focus

- **Dominant contribution**: Slot-based EMA-guided adaptive sparse bilateral rotation adapters for PEFT — the first method that concentrates orthogonal rotation capacity on data-selected dimension pairs (burn-in selection + freeze), achieving better Pareto efficiency than both LoRA and static orthogonal PEFT (qGOFT).
- **Optional supporting contribution**: OER initialization preservation — softmax competitive magnitude calibration that makes JORA initialization-identical to the base model (zero delta at init), enabling stable training without extra warmup heuristics.
- **Explicit non-contributions**: Dense orthogonal fine-tuning, new backbone architectures, BlockCore/LowRankCore variants (available as config options, not main paper claims).

## Proposed Method

### Complexity Budget

- **Frozen / reused**: Pretrained weight W₀, PEFT library infrastructure (HuggingFace PEFT)
- **New trainable components** (2 + 1 structural):
  1. **Rotation angles** θ_L [K_L], θ_R [K_R]: per-slot Givens rotation angles (Cayley-parameterized), frozen pair indices (i, j)
  2. **DiagCore** d [min(S_L, S_R)]: element-wise scale in the rotated basis
  3. **OER logits** e [d_out]: softmax magnitude over output rows (trained jointly)
- **Intentionally excluded**: BlockCore, LowRankCore (appendix only), tanh in delta path (ablation only), continuous pair reselection after burn-in

### System Overview

```
Input x  ∈ ℝ^{B × d_in}
   │
   ├─► W₀ · x  (frozen)                           → base_out ∈ ℝ^{B × d_out}
   │
   └─► Delta path (trainable):
       x̃ = R_R · x          # right rotations on K_R frozen slot pairs, angles θ_R
       x̂ = D_diag ⊙ x̃       # element-wise scale (DiagCore)
       δ = R_L^T · x̂        # left rotations on K_L frozen slot pairs, angles θ_L

       s = oer_softmax(base_row_norms, e, τ)   # per-output-row scale ∈ ℝ^{d_out}
       out = base_out + s ⊙ δ                  # delta-only OER (no tanh)
```

### Slot-Based EMA Selection (Precise Specification)

**State per layer**:
- `slots_R`: K_R pairs `(i_r, j_r, θ_r)` — right-side rotation slots
- `slots_L`: K_L pairs `(i_l, j_l, θ_l)` — left-side rotation slots
- `ema_col`: column energy EMA vector, shape `[d_in]`
- `slot_frozen`: bool flag

**Column energy EMA update** (right-side, every `ema_update_interval` steps):
```
ema_col[t] = β · ema_col[t-1] + (1-β) · mean(x²)  [per column]
```

**Slot reseat** (only when `not slot_frozen`):
```
pairs = select_top_k_pairs_gpu(ema_col, k=K_R, max_features=d_in)
# greedy disjoint: pick pair (argmax, argmax2 disjoint), repeat K_R times
# update (i_r, j_r) for each slot; reset θ_r = 0 for reseated slots; keep θ_r for unchanged slots
```

**Burn-in schedule**:
- First `ceil(0.10 × total_steps)` steps: reseat every T=50 steps
- After burn-in: set `slot_frozen = True`; no more reseating; only θ values train

**Left-side selection**: uses the same EMA vector applied to rows of D_diag output (a single shared EMA buffer suffices for symmetric layers; can be separate if d_out ≠ d_in).

### OER Magnitude (Precise Specification)

```
s_i = softmax(e / τ)_i · total_energy / ||w_i^0||₂   [OER scale, per output row]
```
where `total_energy = (Σ_i ||w_i^0||₂²)^{1/2}`, and τ is an annealed temperature.

**Conservation at initialization** (zero_init_core → DiagCore d=0 → delta=0):
- `s_i = softmax(0) · total_energy / ||w_i^0||₂ = (1/d_out) · total_energy / ||w_i^0||₂`
- `Σ_i s_i² · ||w_i^0||₂² = (total_energy²/d_out²) · d_out = total_energy²/d_out` (a fixed constant)
- The sum of squared scaled norms is conserved at init — OER imposes a fixed row-energy budget.

This is presented as a Proposition in the appendix (technical note, not headline claim).

### Mergeability (Full Inference Integration)

After burn-in (frozen slots), JORA is fully linear and exactly mergeable:
```
W_merged = W₀ + Diag(s) · R_L^T · D_diag · R_R
```
where R_L, R_R are computed from the fixed slot pairs and trained angles θ_L, θ_R.

No tanh (removed from main forward path; tanh variant is an ablation).

### Training Plan

1. **Optimizer**: AdamW with JORA-specific LR (typically 10–20× standard LoRA LR), cosine decay
2. **Burn-in phase** (first 10% of steps): random or EMA-guided slot reseat every 50 steps
3. **Training phase** (remaining 90%): slots frozen; train only θ_L, θ_R, D_diag, OER logits e
4. **OER temperature**: anneal τ from 1.0 → 0.1 over training
5. **Data**: Alpaca-cleaned (52k), standard instruction-following SFT
6. **Model**: Mistral-7B-v0.1
7. **Target modules**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
8. **Configurations**: JORA-small (K=2 per side, S=8 → ~¼ LoRA-r4 params), JORA-base (K=4 per side, S=16 → ~½ LoRA-r4 params)

### Failure Modes and Diagnostics

- **Slot pair collapse** (all slots select the same pair): monitor slot diversity; fallback = increase EMA smoothing β or add random slot noise during burn-in
- **OER saturation** (all scale on one row): monitor Gini coefficient of scale; fallback = increase τ or clip scale
- **θ collapse** (all angles → 0): monitor angular magnitude; if < 1e-3 at end of burn-in, increase JORA-specific LR

### Novelty and Elegance Argument

The core idea — concentrating a fixed rotation budget on data-selected dimension pairs, identified during burn-in and then frozen — is **new relative to all prior work**:
- **qGOFT**: static pairs, no burn-in, no OER → JORA adds (1) EMA-guided burn-in selection and (2) OER magnitude
- **LoRA/DoRA**: no geometric structure, unconstrained scaling → JORA provides geometric inductive bias + competitive magnitude
- **OFT, BOFT**: dense orthogonal matrices (O(n²) params) → JORA uses O(K) sparse pairs post-burn-in
- **The burn-in + freeze pattern is the key insight**: it gives the method the ability to adapt its structure to the task during early training, then commit to a fixed structure for stable optimization — unlike methods that either use fixed structure from the start (qGOFT) or have no structure at all (LoRA).

## Claim-Driven Validation Sketch

### Claim 1 (Primary): JORA achieves better Pareto frontier than LoRA

- **Minimal experiment**: JORA-small and JORA-base vs. LoRA-r1/r2/r4/DoRA-r4 on Mistral-7B + Alpaca-cleaned → MMLU 5-shot, ARC-C 0-shot, GSM8K 4-shot
- **Expected**: JORA-base (~½ LoRA-r4 params) matches LoRA-r4; JORA-small (~¼ params) matches LoRA-r2; both sit on a better frontier than the LoRA curve
- **Metric**: Average accuracy across 3 benchmarks vs. exact trainable parameter count

### Claim 2 (Novelty): JORA outperforms static orthogonal baseline (qGOFT) at equal budget

- **Minimal experiment**: JORA-base vs. qGOFT-reimplemented (JORA codepath, fixed slots, no OER) at matched parameter count
- **Expected**: JORA-base ≥ qGOFT; the diff is attributable to EMA selection (confirmed by ablation)
- **Metric**: MMLU/ARC-C/GSM8K; parameters matched

### Claim 3 (Ablation): EMA selection and OER each contribute

- **Minimal experiment**: JORA-base vs. JORA-no-selection (random slots, frozen after burn-in) vs. JORA-no-OER (magnitude=none, uniform scale) vs. JORA-bare (no selection, no OER = qGOFT-equivalent)
- **Expected**: JORA-base > JORA-no-selection > JORA-bare; JORA-base > JORA-no-OER > JORA-bare; each component adds incremental gain
- **Metric**: MMLU/ARC-C/GSM8K

## Experiment Handoff Inputs

- **Must-prove claims**: Claim 1 (2-point Pareto), Claim 2 (vs qGOFT), Claim 3 (selection + OER ablation)
- **Must-run ablations**: no-selection, no-OER, full-output OER vs delta-only OER, tanh vs no-tanh
- **Critical datasets/metrics**: MMLU 5-shot, ARC-C 0-shot, GSM8K 4-shot; exact parameter counts
- **Highest-risk assumptions**: (1) qGOFT reimplementation fidelity; (2) JORA-base empirically matches LoRA-r4 at half budget; (3) EMA burn-in converges within 10% of training steps

## Compute & Timeline Estimate

- **Setup**: 3× RTX 4090 (24GB each), 2 weeks
- **Revised run count**:
  - LoRA-r1/r2/r4, DoRA-r4: 4 × 2 seeds = ~80 GPU-hours
  - qGOFT (JORA codepath, no EMA/OER): 2 seeds = ~20 GPU-hours
  - JORA-small: 3 seeds = ~30 GPU-hours
  - JORA-base: 3 seeds = ~30 GPU-hours
  - Ablations (no-sel, no-OER, full-out-OER, tanh): 4 × 1 seed = ~40 GPU-hours
  - **Total**: ~200 GPU-hours ≈ 3.5 days on 3 machines
- **Timeline**: Code fix + slot-freeze implementation (2 days) → qGOFT codepath (1 day) → screening run 1 seed (2 days) → full runs (7 days) → analysis + writeup (2 days)
