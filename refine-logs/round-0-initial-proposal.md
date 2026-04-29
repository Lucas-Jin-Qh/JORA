# Research Proposal: TC-CS JORA

*Date: 2026-04-27*
*Round: 0 — Initial Proposal*
*Author: User (JORA project), refined by Claude Code*

---

## Problem Anchor

- **Bottom-line problem**: Current additive JORA-Diag cannot justify rotation's role — matched 1ep and 3ep ON/OFF comparisons show near-zero quality difference while ON costs ~3× runtime. Rotation has no irreplaceable function.

- **Must-solve bottleneck**: The current rotation mechanism is assigned no unique task; it acts as a global sparse basis reparameterization that diagonal scaling can absorb. Without a coupling-specific role, rotation cannot justify its computational cost.

- **Non-goals**: Not claiming "rotation drives the gain" without evidence. Not building a broad PEFT framework. Not rewriting into residualized full-support JORA in this proposal. Not proving that arbitrary sparse rotation helps by default.

- **Constraints**: Must stay compatible with current additive JORA-Diag code path. Must remain compact enough for EMNLP narrative. Runtime cost must be justified. Story must sound like language adaptation geometry.

- **Success condition**: A modified rotation design shows a clear, matched advantage over NoRot in a regime that plausibly requires cross-dimension coupling, while remaining explainable as a necessary mechanism rather than a tuning artifact.

---

## Technical Gap

Current JORA selects dimensions for **importance** (via output gradient EMA and input activation EMA), but not for **coupling**. As a result:

- Rotation is applied too broadly — every selected pair gets rotation, regardless of whether those dimensions actually need coordinated movement.
- The diagonal core already provides sufficient per-dimension recalibration, so rotation's weak global coupling is redundant.
- Pairing uses `energy[i] * energy[j]` — this selects high-high pairs, which tend to be similar, not complementary.

The operational gap:

> Current rotation selection identifies "which dimensions are important" but misses "which dimension pairs need to be coupled." Without this distinction, rotation has no unique contribution.

---

## Method Thesis

### One-sentence thesis

**Restrict JORA rotation to a task-conditioned coupling subspace, and introduce that restriction on one side first, so rotation models cross-dimension coupling only where diagonal-only language adaptation is insufficient.**

### Why this is the smallest adequate intervention

- Keep DiagCore as the primary adaptation capacity (no change).
- Keep Givens rotation parameterization (no change).
- Change only: **selection target** — from importance ranking to coupling relevance.
- Change only: **rotation scope** — from global sparse to subspace-restricted.
- Change only: **rotation side** — from bilateral to one-sided first.

No new trainable components. No new loss functions. No auxiliary networks.

### Why one-sided first

If the hypothesis is "coupling subspace rotation helps," the minimal mechanism test is:

- NoRot: no coupling
- Current JORA-Diag: weak global bilateral coupling
- Proposed TC-CS: explicit localized coupling on one side only

If one-sided already shows a clear advantage, bilateral adds unnecessary complexity. If one-sided fails, we know the direction is wrong before escalating.

---

## Contribution Focus

- **Dominant contribution**: Task-conditioned coupling-subspace rotation for additive JORA-Diag, one-sided first.
- **Optional supporting contribution**: A diagnostic framework distinguishing coupling-relevant from importance-ranked dimensions in language adaptation.
- **Explicit non-contributions**: Not a general orthogonal adaptation framework. Not a claim that arbitrary sparse rotation helps. Not a broad PEFT-family comparison.

---

## Proposed Method

### Complexity Budget

| Category | Status |
|----------|--------|
| Frozen backbone | reused |
| DiagCore | reused as-is |
| Givens rotation parameterization | reused as-is |
| Selection target | **changed** (importance → coupling relevance) |
| Rotation scope | **changed** (global sparse → subspace-restricted) |
| Rotation side | **changed** (bilateral → one-sided first) |
| New trainable components | none |
| Auxiliary networks | none |
| New loss functions | none |
| LowRankCore / BlockCore | deliberately excluded from main story |
| Bilateral expansion | Phase 2 only |

### System Overview

```
Current JORA-Diag:
    x → [R_R (global sparse)] → [DiagCore] → [R_L^T (global sparse)] → Δ(x)
    where selection uses energy[i]*energy[j] (importance-based pairing)

Proposed TC-CS JORA:
    Stage 1 (calibration):
        collect per-dimension statistics → compute coupling relevance scores →
        derive coupling subspace S (|S| = m << d)
    Stage 2 (training):
        x → [R_R^(S) (subspace-restricted, one-sided)] → [DiagCore] → [R_L = I] → Δ(x)
    where:
        - R_R^(S) only pairs dimensions inside S
        - R_L is frozen/identity in v1
        - DiagCore unchanged
```

### Core Mechanism

#### Step A: Coupling Relevance Score (replaces importance ranking)

Instead of `energy[i] * energy[j]`, compute a coupling proxy:

For each dimension `i`, collect during calibration:

- `g_i`: output gradient EMA (existing)
- `a_i`: input activation EMA (existing)
- `c_ij`: co-activation proxy between dimensions `i` and `j`
  - Minimal implementation: `c_ij = |corr(activations_i, activations_j)|` over calibration batches
  - Or: `c_ij = |E[a_i a_j]|` as a lightweight second-order statistic

Define coupling relevance for a pair `(i, j)` as:

```
score_coupling(i,j) = c_ij * min(g_i, g_j)
```

This rewards pairs that:
- Are positively correlated in activation (need to move together)
- Have at least moderate gradient energy on both sides (not dead dimensions)

**Why this is different from current energy[i]*energy[j]**: that scores `g_i * g_j`, which selects high-high energy pairs that are likely redundant. Coupling relevance selects pairs where dimensions are correlated, which is the actual condition under which rotation provides information that diagonal scaling cannot.

#### Step B: Subspace Selection S

1. Compute coupling relevance scores for all candidate pairs.
2. Sort by `score_coupling` descending.
3. Take top-`m` dimensions that appear in the highest-scoring pairs as the coupling subspace `S`.
4. `m` is a hyperparameter (e.g., 25% of `S_R`, or determined by an elbow in the score distribution).

This replaces `top_k_pairs_gpu` with a **coupling-aware** two-stage selection:
- Stage 1: choose `S` via coupling relevance
- Stage 2: pair within `S` via coupling relevance (not energy product)

#### Step C: One-Sided Restriction

- `R_R` is restricted to act only on dimensions in `S`.
- `R_L = I` (identity, frozen) in v1.
- This gives `Δ(x) ≈ Diag(d) · (coupled projection of x)` — input-side coupling + diagonal calibration.

### Training Plan

```
Stage 0: Base model frozen (unchanged)

Stage 1: Calibration (~500-1000 steps, same data as main training)
    - Collect activation and gradient statistics
    - Compute coupling relevance scores
    - Derive coupling subspace S per layer
    - Freeze S topology

Stage 2: Main optimization
    - Train DiagCore + one-sided subspace-restricted rotation
    - Use split LR: lower LR for theta (rotation), higher for diag_params (core)
    - Log: theta norm, theta grad norm, subspace S per layer, coupling score distribution

Stage 3: Evaluation (bilateral expansion only if Stop Rule C passes)
```

### Failure Modes and Diagnostics

| Failure Mode | Detection | Fallback |
|---|---|---|
| Rotation still redundant | ON ≈ NoRot, theta grad weak | Drop rotation from main story |
| Subspace not coupling-relevant | S is importance-heavy, NoRot matches | Revise coupling proxy signal, not architecture |
| One-sided too weak but bilateral overcomplicates | Small unstable gains | Keep as negative-but-informative appendix |
| Rotation helps but runtime unjustifiable | Small quality gain, step time >> NoRot | Keep as informative ablation, not main claim |

### Stop Rules

- **Stop Rule A**: TC-CS-1S ≈ NoRot in quality → stop rotation revival主线 entirely.
- **Stop Rule B**: TC-CS-1S > current JORA-Diag but ≈ NoRot → subspace restriction helps but rotation itself not the mechanism → appendix only.
- **Stop Rule C**: TC-CS-1S > NoRot, margin stable, runtime justifiable → allow bilateral expansion and broader baselines.

### Novelty and Elegance Argument

Closest prior work: current JORA-Diag (global sparse rotation with importance selection).

Exact difference: Replace importance-based global sparse rotation with coupling-relevance-based subspace-restricted one-sided rotation.

This is a **mechanism-level** change, not a parameter count or learning rate tweak. The paper's core claim becomes falsifiable: if coupling subspace rotation does not beat NoRot, the hypothesis is wrong. If it does, the coupling story is grounded in a specific geometric intuition that is distinct from both "diagonal is enough" and "more parameters help."

---

## Claim-Driven Validation Sketch

### Claim 1: TC-CS-1S rotation provides benefit beyond diagonal-only scaling

- **Minimal experiment**: Matched comparison on OPT-350m (same protocol as existing 3ep run)
  1. JORA-NoRot (S_L=0, S_R=0)
  2. Current JORA-Diag (global sparse bilateral)
  3. TC-CS-1S (coupling subspace, one-sided)

- **Metric**: Final task quality (train loss, token accuracy). Runtime per step.

- **Expected evidence**: TC-CS-1S clearly beats both NoRot and current JORA-Diag in quality, with acceptable runtime overhead.

### Claim 2: The benefit comes from coupling-aware restriction, not extra parameters

- **Minimal experiment**: Compare coupling-random subspace vs coupling-relevance subspace, both one-sided, same subspace size.

- **Metric**: Final quality.

- **Expected evidence**: Coupling-relevance subspace outperforms random subspace of the same size, confirming the mechanism is coupling-specific.

### Deletion/Simplification Check

- Compare TC-CS-1S directly against NoRot with identical core budget.
- If NoRot matches, rotation should be abandoned entirely.

---

## Experiment Handoff Inputs

- **Must-prove claims**: Coupling subspace rotation can outperform diagonal-only when restricted to one-sided subspace action.
- **Must-run ablations**: NoRot / current JORA-Diag / TC-CS-1S / coupling-random baseline.
- **Critical metrics**: Task quality, runtime, theta norm/grad diagnostics, subspace stability diagnostics.
- **Highest-risk assumption**: A coupling-relevant subspace exists in language adaptation and can be identified cheaply from calibration statistics.

---

## Compute & Timeline Estimate

- **GPU-hours**: Moderate. TC-CS-1S vs NoRot vs current JORA-Diag matched comparison (~3 runs × 3 epochs on OPT-350m).
- **Data cost**: None beyond existing pipelines.
- **Timeline**:
  1. Implement coupling relevance scoring + subspace selection (layer.py, selection.py modifications)
  2. Implement one-sided restriction in rotation application
  3. Run calibration + 1 matched 3-epoch sanity comparison
  4. Only expand if Stop Rule C passes

---

## Relationship to Existing Code

The changes map cleanly to existing modules:

| File | Change |
|------|--------|
| `layer.py` | `compute_delta()` unchanged; add coupling subspace state to `_JoraAdapterState`; modify `update_step()` to use coupling-aware selection |
| `selection.py` | Replace `select_top_k_pairs_gpu` with coupling-relevance two-stage selection |
| `callbacks.py` | Add subspace diagnostics (S size, S stability, coupling score distribution) |
| `config.py` | Add `coupling_subspace` flag; add `subspace_selection` strategy parameter |

No changes to `core.py`, `rotation.py`, or `model.py` are required for v1.
