# Research Proposal: JORA — Sparse Bilateral Givens Rotations with EMA Selection and Competitive Energy Redistribution for Parameter-Efficient Fine-Tuning

## Problem Anchor

- **Bottom-line problem**: PEFT methods must achieve a better accuracy-vs-parameter Pareto frontier than LoRA for LLM fine-tuning, using a principled, structure-preserving adaptation mechanism.
- **Must-solve bottleneck**: Current orthogonal PEFT methods (OFT, qGOFT) use static, dense rotation structures — they adapt all dimensions uniformly, wasting capacity on unimportant directions and lacking the ability to focus on task-relevant subspaces. Current low-rank PEFT (LoRA, DoRA) lacks geometric inductive bias and suffers from norm drift due to unconstrained scaling. Neither achieves an efficient sparse-vs-dense rotation tradeoff.
- **Non-goals**: Not a new backbone architecture, not replacing LoRA for all use cases, not targeting inference latency as the primary metric.
- **Constraints**: 3× RTX 4090 GPUs, 2-week experiment window. Must be implementable within PEFT library. Primary venue: NeurIPS 2026.
- **Success condition**: JORA at ~half LoRA-r4 trainable parameter budget matches LoRA-r4 on MMLU/ARC-C/GSM8K benchmarks, clearly beats LoRA-r2, and beats qGOFT at equal budget. Ablations must show that each component (sparse selection, OER) contributes.

---

## Technical Gap

**Where current methods fail:**

LoRA adds an unconstrained low-rank perturbation `ΔW = BA`. This is effective but ignores the geometric structure of the pretrained weight matrix. The free scaling in DoRA addresses per-row norm drift partially, but per-row independently — there is no global energy conservation constraint. Critically, LoRA's capacity is spread across all `r` directions uniformly (for fixed rank), with no mechanism to focus on task-relevant subspaces dynamically.

qGOFT (Ma et al., ICML 2024) is the closest prior: it applies Givens rotations (orthogonal updates) directly to weight matrix rows, providing geometric structure. However, qGOFT uses a **static, predetermined** rotation pattern — all selected dimensions are rotated with equal priority. There is no mechanism to:
1. **Dynamically identify which dimension pairs are most task-relevant** (data-driven selection)
2. **Competitively redistribute energy** across output dimensions (energy conservation)

This means qGOFT's capacity is still spread uniformly across selected pairs, rather than concentrating where the gradient signal is strongest.

**Why naive bigger systems are insufficient:**
- More LoRA rank: proportionally more parameters, no geometric structure
- Dense orthogonal matrices (OFT, full BOFT): O(n²) parameters, non-mergeable, costly
- qGOFT alone: static selection misses task-adaptive dimension pair prioritization

**Smallest adequate intervention:**
1. Bilateral sparse Givens rotations (left and right sides of weight matrix) — captures coupled input-output subspace rotation
2. EMA-driven dynamic top-k pair selection — concentrates capacity where gradient energy is highest
3. Competitive OER magnitude (softmax-based) — enforces zero-sum energy redistribution across output rows, preventing norm drift

**Core technical claim**: Sparse bilateral Givens rotations with EMA-based dynamic dimension selection and competitive energy redistribution (OER) achieves a better accuracy-vs-parameter Pareto frontier than LoRA, while extending qGOFT with learned, data-adaptive rotation pair selection.

---

## Method Thesis

- **One-sentence thesis**: JORA achieves parameter-efficient fine-tuning by applying sparse bilateral Givens rotations to pretrained weight matrices, with EMA-guided dynamic dimension pair selection and competitive output energy redistribution, forming a better Pareto frontier than LoRA-r4 at half the parameter budget.
- **Why this is the smallest adequate intervention**: Each component directly addresses one failure mode — bilateral rotations handle input-output coupling, EMA selection handles static-vs-dynamic pair choice, OER handles norm drift without over-parameterizing the magnitude space.
- **Why this route is timely**: Foundation model fine-tuning is parameter-constrained; geometric structure in weight updates is increasingly shown to improve stability and generalization (OFT → qGOFT → JORA progression). The EMA-guided sparsity mechanism mirrors attention-like selection in the parameter space, directly leveraging training signal geometry.

---

## Contribution Focus

- **Dominant contribution**: A sparse, bilateral, adaptive Givens rotation PEFT method that extends qGOFT with EMA-guided dynamic pair selection + competitive energy redistribution (OER), achieving a better accuracy-vs-params Pareto frontier than both LoRA and static orthogonal methods.
- **Optional supporting contribution**: Formal OER conservation proposition — softmax-based competitive energy redistribution preserves sum-of-squared-row-norms at initialization, providing implicit regularization stronger than per-row independent scaling (DoRA).
- **Explicit non-contributions**: New backbone architecture, inference-time optimizations, multi-modal applications.

---

## Proposed Method

### Complexity Budget

- **Frozen / reused**: Pretrained weight matrix W₀, PEFT library infrastructure (save/load, merge)
- **New trainable components** (2):
  1. Rotation parameters θ_L, θ_R (Cayley-parameterized Givens angles) for selected dimension pairs
  2. Core transformation matrix D (diagonal / block / low-rank) between left and right rotations
  3. OER logits e (softmax-based magnitude over output rows)
- **Intentionally excluded**: Full-rank orthogonal matrices, multi-step rotation composition, task-specific hypernetworks, adapter stacking

### System Overview

```
Input x [batch, in_features]
   │
   ├─► W₀x (frozen base linear)          = base_out
   │
   └─► Delta path (trainable):
       1. R_R(x):  apply right Givens rotations (S_R pairs, θ_R params)
       2. D · R_R(x): core transform (DiagCore: element-wise scale; BlockCore: block matrix; LowRankCore: UV)
       3. R_L^T(D · R_R(x)): apply left Givens rotations (S_L pairs, θ_L params)
       → delta = R_L^T · D · R_R · x

Output: base_out + scale * delta    [delta-only OER]
         where scale = OER(base_row_norms, e) = softmax competitive redistribution
```

### Core Mechanism

- **Input/output**: x ∈ ℝ^{B×d_in} → y ∈ ℝ^{B×d_out}
- **Forward pass (delta-only OER — the fixed version)**:
  ```
  delta = R_L^T · D · R_R · x       # geometric transformation
  scale = oer_softmax(base_row_norms, ecd_log_mag, temperature)  # per-row scale
  out = base_out + scale * delta     # magnitude scales ONLY the delta
  ```
- **Rotation parameterization**: Cayley transform maps unconstrained θ → bounded angle; rotation acts on selected dimension pairs (i, j): `[x_i, x_j] → [x_i cos θ - x_j sin θ, x_i sin θ + x_j cos θ]`
- **EMA selection**: At each training step, gradient energy per dimension pair `g_{ij} = E[||∂L/∂x_i||² + ||∂L/∂x_j||²]` is tracked via EMA; top-k pairs with highest EMA energy are selected for rotation
- **OER (Output Energy Redistribution)**: scale_i = C · softmax(e / τ)_i · ||w_i^0||₂ / Σ_j softmax(e/τ)_j · ||w_j^0||₂; conserves Σ scale_i² · ||w_i||² at initialization
- **Core transform D**: Default DiagCore (element-wise scale), max parameter efficiency; BlockCore / LowRankCore available for higher capacity
- **Why this is the main novelty**: qGOFT has bilateral rotations with static pair selection; JORA adds EMA-adaptive selection (focuses capacity) and OER competitive magnitude (prevents norm drift). Together they form a PEFT method with explicit learning-signal-guided structural adaptation.

### OER Conservation Proposition

At initialization (zero_init_core=True, uniform OER logits):
- scale_i = ||w_i^0||₂ / (Σ_j ||w_j^0||₂²)^{1/2} · constant
- Therefore Σ_i scale_i² · ||w_i^0||₂² = Σ_i ||w_i^0||₂² = total_energy (conserved)
- The OER induces a fixed row-energy budget in scale space by construction
- Frame as: **Proposition 1 (OER Row-Energy Conservation)**: at initialization, the OER mechanism preserves the total squared row-norm budget of the weight matrix

### Integration and Mergeability

**Delta-only OER is fully mergeable**:
```
W_merged = W₀ + Diag(scale) · R_L^T · D · R_R
```
(for DiagCore; approximate for BlockCore/LowRankCore — linear approximation captures main effects)

Merging eliminates all JORA runtime overhead; deployed model is identical to a modified linear layer.

**Note on tanh**: Current code includes `tanh` in delta computation (non-linear, breaks exact mergeability). Recommendation: drop tanh → `delta = R_L^T · D · R_R · x` → fully linear, fully mergeable. Run tanh variant as ablation only.

### Training Plan

1. **Optimizer**: AdamW, JORA-specific LR (typically 10× standard LR), cosine schedule
2. **EMA warmup**: First K steps (default K=50) use random pair selection; after warmup, switch to EMA top-k greedy disjoint pair selection
3. **OER temperature**: Anneal τ from τ_init=1.0 to τ_final=0.1 over training (concentrate energy allocation)
4. **Stage**: Single-stage joint training (rotations + core + OER logits trained jointly)
5. **Data**: Alpaca-cleaned (52k), standard SFT format
6. **Base model**: Mistral-7B-v0.1 (preferred; Llama-2-7B as fallback)
7. **Target modules**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

### Failure Modes and Diagnostics

- **EMA pair collapse** (all energy concentrates on one pair): monitor pair entropy during training; fallback = increase k or use random selection for burn-in
- **OER scale saturation** (all scale goes to one row): monitor scale Gini coefficient; fallback = increase temperature or clip scale
- **Rotation angle collapse** (θ → 0): monitor rotation angle magnitude; can indicate LR too low
- **Merge discrepancy**: after merging and unmerging, logit diff should be < 1e-4; test covered in test suite

### Novelty and Elegance Argument

- **vs. qGOFT**: JORA adds (a) bilateral vs. unilateral rotation, (b) EMA-adaptive vs. static pair selection, (c) OER competitive magnitude vs. no magnitude scaling. Each addition is individually motivated and ablated.
- **vs. LoRA**: JORA operates on the orthogonal group (geometry-preserving updates) rather than the space of low-rank perturbations. The two methods occupy different points in the "structure vs. flexibility" tradeoff space.
- **vs. DoRA**: OER is a global competitive mechanism (zero-sum redistribution), not per-row independent scaling. This provides implicit regularization that per-row DoRA lacks.
- **Elegance**: The core method is a 3-component pipeline (rotation → core → rotation) with 2 additional learned mechanisms (selection, OER), each with a direct geometric interpretation and a clear ablation path.

---

## Claim-Driven Validation Sketch

### Claim 1 (Primary): JORA achieves better Pareto frontier than LoRA

- **Minimal experiment**: JORA-full (DiagCore, S=16, k=4) at ~half LoRA-r4 params vs. LoRA-r2, LoRA-r4 on Mistral-7B + Alpaca-cleaned → MMLU 5-shot, ARC-C, GSM8K
- **Expected**: JORA matches LoRA-r4, clearly beats LoRA-r2, at ~50% of LoRA-r4 trainable params
- **Baselines**: LoRA-r1, LoRA-r2, LoRA-r4, DoRA-r4
- **Metric**: MMLU accuracy, ARC-C accuracy, GSM8K exact match; trainable parameter count (report exact)

### Claim 2 (Novelty): JORA extends qGOFT with adaptive selection and OER, improving on qGOFT at equal budget

- **Minimal experiment**: JORA-full vs. qGOFT (reimplemented, or BOFT+HRA as fallback) at matched parameter count, same training setup
- **Expected**: JORA ≥ qGOFT on all three benchmarks; ablation shows removing EMA selection hurts performance
- **Metric**: MMLU/ARC-C/GSM8K; parameter count matched

### Claim 3 (Ablation): EMA selection and OER each contribute independently

- **Minimal experiment**: JORA-full vs. JORA-no-selection (random pairs, fixed) vs. JORA-no-OER (magnitude=none) vs. JORA-no-selection-no-OER (qGOFT equivalent)
- **Expected**: Each ablation hurts performance; full JORA is best
- **Metric**: MMLU/ARC-C/GSM8K on Mistral-7B + Alpaca-cleaned

### Supporting Claim (OER correctness): delta-only OER preserves base output at initialization

- **Minimal experiment**: Unit test — with zero-initialized core, JORA output == base linear output (covered in `test_oer_softmax_initialization_is_identity`)
- **Already validated by test suite**

---

## Experiment Handoff Inputs

- **Must-prove claims**: Claim 1 (Pareto), Claim 2 (beats qGOFT), Claim 3 (ablation decomposition)
- **Must-run ablations**: no-selection, no-OER, full-output OER vs. delta-only OER, tanh vs. no-tanh
- **Critical datasets/metrics**: MMLU 5-shot, ARC-C 0-shot, GSM8K 4-shot; exact trainable parameter counts
- **Highest-risk assumptions**: (1) qGOFT reimplementation is fair; (2) JORA at half LoRA-r4 params actually matches — needs empirical verification; (3) EMA selection actually helps (may need task with sufficient training steps for EMA to converge)

---

## Compute & Timeline Estimate

- **Setup**: 3× RTX 4090 (24GB each), 2-week window
- **Per-run estimate**: Mistral-7B fine-tuning on 52k Alpaca, 3 epochs ≈ 8-12 GPU-hours (single GPU)
- **Total runs**:
  - LoRA-r1/r2/r4, DoRA-r4: 4 configs × 2 seeds = 8 runs × ~10h = ~80 GPU-hours
  - qGOFT or BOFT+HRA: 2 configs × 2 seeds = 4 runs × ~10h = ~40 GPU-hours
  - JORA-full: 3 seeds × ~10h = ~30 GPU-hours
  - JORA ablations (no-sel, no-OER, full-out-OER, tanh): 4 configs × 1 seed = 4 runs × ~10h = ~40 GPU-hours
  - **Total**: ~190 GPU-hours ≈ 64 GPU-hours/machine × 3 machines ≈ ~3 days of compute
- **Data cost**: Alpaca-cleaned is freely available; no annotation needed
- **Timeline**: Code fix (2 days) → qGOFT baseline (2 days) → main runs (7 days) → analysis + paper update (3 days) = 14 days total
