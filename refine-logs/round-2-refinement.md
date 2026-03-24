# Round 2 Refinement

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
- **Does the revised method still address it?** Yes — adaptive dimension allocation with burn-in freeze directly addresses static allocation in qGOFT.
- **Reviewer suggestions rejected as drift**: None this round. Reviewer drift warning was NONE.

---

## Simplicity Check

- **Dominant contribution after revision**: Adaptive sparse bilateral rotation adapter with burn-in dimension allocation + slot freeze. One mechanism, one story.
- **Components removed/demoted**:
  - OER: retained but demoted to "magnitude stabilizer" only, not a co-headline novelty. If ablations show it doesn't help, drop from main paper entirely.
  - Pair-level novelty claim: removed. Now framed as "adaptive dimension allocation with deterministic disjoint pairing."
- **Reviewer suggestions adopted**:
  - Separate input-side (EMA[g²]) and output-side selection statistics specified precisely.
  - D operator for rectangular layers defined explicitly.
  - Module scope restricted to square projections for primary results; rectangular extended as secondary.
- **Why remaining mechanism is smallest adequate route**: Bilateral rotation adapter (2 rotation stages + diagonal core) is the minimal structure-preserving PEFT; EMA burn-in + freeze is the minimal adaptive allocation mechanism.

---

## Changes Made

### 1. Specify two separate selection statistics (Critical fix)
- **Reviewer said**: "ema_col[d_in] is only input-side. Left slots need output-side signal. Actual update uses x² but text claims gradient energy."
- **Action**: Define precisely:
  - **Right-side (input)**: `ema_in[j] = β · ema_in[j] + (1-β) · mean_batch(x_j²)` — activation energy proxy per input dimension. No backward pass needed. Reframe as "activation energy," not "gradient energy."
  - **Left-side (output)**: `ema_out[i] = β · ema_out[i] + (1-β) · mean_batch((∂L/∂δ_i)²)` — gradient energy proxy per output dimension, computed via backward hook on the delta output tensor.
  - **Selection**: right slots: top `2K_R` input dimensions by `ema_in`, greedy disjoint pairing → K_R slot pairs. Left slots: top `2K_L` output dimensions by `ema_out`, greedy disjoint pairing → K_L slot pairs.
  - **Claim language**: "activation-energy-guided (right) and gradient-energy-guided (left) dimension allocation" — not "gradient energy" for both.
- **Impact**: Eliminates the gradient vs. activation mismatch. Both statistics are now precisely defined and implementable.

### 2. Define D operator for rectangular layers (Critical fix)
- **Reviewer said**: "D_diag[min(K_L, K_R)] is not shape-precise for rectangular layers."
- **Action**: Define explicitly:
  - For a layer with `d_out × d_in` weight (rectangular), the rotation pipeline operates on the active slot indices only:
    - R_R: applies K_R Givens rotations in the input space (d_in dimensions), output still d_in
    - D: a diagonal scale at the d_in level between R_R and R_L — specifically, D applies only to the active slot dimensions (K_R positions), others pass through with scale=1
    - R_L: applies K_L Givens rotations in the output space (d_out dimensions), input is d_in
    - Full delta shape: (d_out, d_in) — R_L^T acts on rows of (D · R_R · x) to rotate output dimensions
  - For the primary result set (q_proj, k_proj, v_proj, o_proj): these are all square in most transformer architectures (d_head × d_head). Gate/up/down projections may be rectangular. **Primary results on square projections only; rectangular extension noted.**
  - D_diag[d_in] — diagonal of length d_in, where only active slot indices have learned values, rest initialized to 1.0 and either frozen or also trained.
- **Impact**: Exact shape accounting for rectangular layers; primary results on square projections to avoid scope risk.

### 3. OER reframed as magnitude stabilizer (Important fix)
- **Reviewer said**: "Base output preserved at init comes from zero D, not from OER. OER should be framed as stabilizer, not novelty."
- **Action**:
  - Remove "OER initialization preservation" as a supporting contribution from the proposal.
  - The initialization property (output = base_out when D=0) is from zero_init_core, not OER.
  - OER is now described as: "row-norm magnitude stabilizer — softmax competitive redistribution prevents output row norms from diverging freely during training, providing implicit regularization stronger than per-row independent scaling."
  - OER stays in the main method if ablations confirm a consistent gain. If not, dropped to appendix.
  - Paper has **one contribution**: adaptive sparse bilateral rotation adapter with burn-in allocation. OER is a design choice, not a contribution.
- **Impact**: Contribution clarity massively improved. No false initialization claims.

### 4. Reframe "pair selection" as "dimension allocation" (Important fix)
- **Reviewer said**: "Currently is adaptive dimension allocation + deterministic pairing. Don't claim adaptive pair selection."
- **Action**: Replace all "pair selection" language with "dimension allocation with deterministic disjoint pairing." The novelty is in which dimensions are selected, not which pairs — since pairing is greedy/deterministic given the dimension scores.
- **Impact**: Honest framing; avoids overstating novelty at the pair level.

---

## Revised Proposal

# Research Proposal: JORA — Adaptive Sparse Bilateral Rotation Adapters for LLM Fine-Tuning

## Problem Anchor

- **Bottom-line problem**: PEFT methods must achieve a better accuracy-vs-parameter Pareto frontier than LoRA for LLM fine-tuning, using a principled, structure-preserving adaptation mechanism.
- **Must-solve bottleneck**: Orthogonal PEFT (qGOFT, OFT) uses static rotation structures — all dimension pairs adapted uniformly, wasting capacity on low-signal directions. Low-rank PEFT (LoRA, DoRA) lacks geometric bias and suffers norm drift. Neither achieves an efficient adaptive sparse rotation.
- **Non-goals**: Not a new backbone, not replacing LoRA for all use cases, not targeting inference latency.
- **Constraints**: 3× RTX 4090, 2-week window. PEFT library. NeurIPS 2026.
- **Success condition**: JORA at ~half LoRA-r4 params matches LoRA-r4 on MMLU/ARC-C/GSM8K; clearly beats LoRA-r2; beats qGOFT at equal budget.

## Technical Gap

qGOFT applies Givens rotations uniformly across all selected dimension pairs — static allocation, no gradient-guided prioritization. LoRA is unconstrained low-rank addition with norm drift.

The gap: **data-adaptive allocation of a fixed sparse bilateral rotation budget**. No existing PEFT method uses online statistics to identify which input and output dimensions carry the most learning signal, then concentrates the rotation capacity there and freezes the allocation for stable optimization.

## Method Thesis

JORA achieves PEFT by applying adaptive sparse bilateral Givens rotation adapters: during burn-in (first 10% of training), online activation/gradient energy statistics guide dimension allocation for right and left rotation slots; after burn-in, slot assignments are frozen and only rotation angles and DiagCore scale are optimized. This concentrates the fixed rotation budget on the highest-signal dimensions and makes the method fully mergeable after training.

**Honest framing**: JORA computes `delta = R_L^T · D_diag · R_R · x` and uses `out = base_out + s ⊙ delta` — a rotation-basis structured additive adapter.

## Contribution Focus

- **Single dominant contribution**: Adaptive sparse bilateral Givens rotation adapters with burn-in dimension allocation + slot freeze — the first PEFT method that concentrates orthogonal rotation capacity on data-selected dimensions (one-time EMA-guided allocation during burn-in, then frozen), achieving better Pareto efficiency than both LoRA and static orthogonal PEFT.
- **Method design element** (not co-contribution): OER magnitude stabilization — softmax competitive row-norm redistribution to prevent norm drift during training.
- **Explicit non-contributions**: Dense orthogonal fine-tuning, BlockCore/LowRankCore (appendix), tanh variant (ablation).

## Proposed Method

### Complexity Budget
- **Frozen**: Pretrained W₀, PEFT library infrastructure
- **New trainable** (within budget):
  - θ_R [K_R]: right rotation angles for K_R frozen slot pairs (input-side)
  - θ_L [K_L]: left rotation angles for K_L frozen slot pairs (output-side)
  - D_diag [d_in]: diagonal core scale (active at slot positions only)
  - e [d_out]: OER magnitude logits
- **Structural state** (not trainable after burn-in):
  - slots_R: K_R pairs (i_R, j_R) — frozen after burn-in
  - slots_L: K_L pairs (i_L, j_L) — frozen after burn-in

### Forward Pass

```
Input x ∈ ℝ^{B × d_in}:

base_out = W₀ · x   (frozen)

# Delta path:
x̃ = R_R(slots_R, θ_R) · x       # Givens rotations on K_R input dim pairs
x̂ = D_diag ⊙ x̃                  # element-wise scale (active at K_R positions)
δ = R_L^T(slots_L, θ_L) · x̂     # Givens rotations on K_L output dim pairs

s = oer_softmax(base_row_norms, e, τ)  ∈ ℝ^{d_out}
out = base_out + s ⊙ δ
```

For rectangular layers (d_out ≠ d_in): D_diag operates on d_in dimensions, with learned scale at slot-active positions (others = 1.0); R_L operates on d_out output dimensions. No shape mismatch.

### Selection Statistics (Precisely)

**Right-side (input dimensions)**:
```
ema_in[j] ← β · ema_in[j] + (1-β) · mean_batch(x_j²)
```
Activation energy proxy. No backward pass needed.

**Left-side (output dimensions)**:
```
ema_out[i] ← β · ema_out[i] + (1-β) · mean_batch((∂L/∂δ_i)²)
```
Gradient energy proxy. Computed via backward hook on delta output.

**Slot reseat** (when not slot_frozen):
```
# Right slots: select top 2K_R dims by ema_in, greedy disjoint pair → K_R pairs
slots_R = select_top_k_pairs_gpu(ema_in, k=K_R)
# Left slots: select top 2K_L dims by ema_out, greedy disjoint pair → K_L pairs
slots_L = select_top_k_pairs_gpu(ema_out, k=K_L)
# Reseated slots: reset θ=0; unchanged slots: keep θ
```

**Burn-in schedule**:
- Steps [0, 0.10 × T_total]: reseat every T_reseat=50 steps
- After burn-in: slot_frozen=True; never reseat again

Claim language: "activation-energy-guided (right) and gradient-energy-guided (left) dimension allocation with deterministic disjoint pairing" — not "pair selection."

### OER Magnitude

```
s_i = softmax(e_i / τ) · total_energy / ||w_i^0||₂
```
Softmax competition across output rows redistributes magnitude budget. Prevents individual row norms from growing unboundedly. Temperature τ anneals from 1.0 → 0.1 over training.

**Initialization property**: comes from zero_init_core (DiagCore initialized to 0 → delta=0 at init → out = base_out exactly). OER is orthogonal to this property.

### Mergeability

After burn-in (slots frozen, D_diag and θ values trained):
```
W_merged = W₀ + Diag(s) · R_L^T · D_diag · R_R
```
Exact merge (linear forward, no tanh). No JORA overhead at inference.

### Training Plan
1. AdamW, JORA-specific LR (10–20× standard), cosine decay
2. Burn-in (first 10%): reseat every 50 steps (right slots by ema_in, left slots by ema_out)
3. Main training (90%): frozen slots, train θ_L, θ_R, D_diag, e
4. OER temperature anneal: τ 1.0 → 0.1
5. Mistral-7B-v0.1, Alpaca-cleaned 52k, SFT
6. JORA-small (K=2/side, ~¼ LoRA-r4 params), JORA-base (K=4/side, ~½ LoRA-r4 params)
7. Primary module scope: q_proj, k_proj, v_proj, o_proj (square layers); gate/up/down as secondary

### Failure Modes and Diagnostics
- **Slot collapse** (all slots on same pair): monitor slot diversity at burn-in end; if entropy < log(K)/2, increase ema smoothing β
- **ema_out instability** (backward hook fails): fallback = use ema_in for both sides (symmetric)
- **OER saturation**: monitor Gini coefficient of s; if > 0.8 at mid-training, increase τ
- **Rectangular layer D shape bug**: validate delta shape matches expected (d_out, d_in) before main runs

### Novelty and Elegance Argument

- vs. qGOFT: same rotation primitive, but JORA adds data-adaptive allocation (ema_in/ema_out guided) + burn-in freeze + OER stabilization
- vs. LoRA: operates on orthogonal group in input/output spaces; geometric structure; competitive magnitude constraint
- vs. DoRA: OER is global zero-sum redistribution, not per-row independent scaling
- **The burn-in + freeze pattern** is the key new idea: the method adapts its structure to the task signal during early training (flexible), then commits to a fixed structure for stable optimization (efficient). This is structurally analogous to pruning during training, but for rotation subspace selection.

## Claim-Driven Validation Sketch

### Claim 1 (Primary): Better Pareto frontier than LoRA
- JORA-small and JORA-base vs. LoRA-r1/r2/r4/DoRA-r4 on Mistral-7B + Alpaca-cleaned
- Metric: avg MMLU/ARC-C/GSM8K vs. exact trainable param count
- Expected: 2-point JORA curve dominates LoRA curve at matched budgets
- 3 seeds for JORA-base + LoRA-r4 (variance estimate)

### Claim 2: JORA > qGOFT at equal budget
- JORA-base vs. qGOFT (same codepath: fixed random slots, no EMA, no OER)
- Expected: JORA-base ≥ qGOFT; ablation confirms EMA allocation contributes

### Claim 3 (Ablation): EMA allocation and OER each contribute
- JORA-base vs. JORA-random-slots (random frozen slots, with OER) vs. JORA-no-OER (EMA selection, no OER) vs. JORA-bare (random slots, no OER = qGOFT-equivalent)
- Expected: each component adds incremental gain; full JORA is best

## Experiment Handoff Inputs
- Must-prove: Claim 1 (2-point Pareto), Claim 2 (vs. qGOFT), Claim 3 (ablation decomposition)
- Critical: MMLU/ARC-C/GSM8K, exact param counts, 3 seeds for JORA-base + LoRA-r4
- Highest-risk: (1) qGOFT codepath fidelity; (2) JORA-base empirically matches LoRA-r4; (3) ema_out backward hook implementable without performance hit

## Compute & Timeline
- ~200 GPU-hours on 3× RTX 4090 ≈ 3.5 days compute
- Timeline: code fix + slot-freeze + ema_out hook (2d) → qGOFT codepath validation (1d) → screening 1-seed (2d) → full runs 3-seed (7d) → analysis (2d)
