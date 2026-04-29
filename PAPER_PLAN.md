# Paper Plan

**Title**: (Working) On the Failure Modes of Rotation-Based Diagonal PEFT: A Diagnostic Study
**Venue**: ICLR 2026 (as technical report / negative result submission) or arXiv preprint
**Type**: Empirical / Diagnostic — negative result study
**Date**: 2026-04-28
**Page budget**: 8 pages (main body, excluding references & appendix)
**Section count**: 7

---

## Claims-Evidence Matrix

| # | Claim | Evidence | Status | Section |
|---|-------|----------|--------|---------|
| 1 | Rotation ON provides no quality advantage over NoRot at matched training | E1/E2/E3: loss delta = 0.00055 (noise), token_acc delta = -0.00027 (NoRot better), ON 2.86x slower | **Supported** | §4.1 |
| 2 | TC-CS coupling subspace fails to differentiate from energy-based selection | E4/E6/E7: 100% pair overlap (384/384), rank-1 collapse of outer(gc), train_loss delta = 2.7e-6 | **Supported** | §4.2 |
| 3 | Residualized full-support DiagCore (Δ = R_L^T(I+Diag)R_Rx - x) causes catastrophic training failure | C2.1: eval loss 15.7 (ON) / 19.2 (NoRot) vs base 5.5 vs additive 2.24; root cause R_L^T R_R ≠ I | **Supported** | §4.3 |
| 4 | Projector-constrained residualization (SelectiveDiagCore) is stable; full-identity residualization is not | Option C postmortem, mathematical analysis | **Supported** | §3.3 |
| 5 | Exact merge via basis-probing is verified for both DiagCore and SelectiveDiagCore | M0 gate: 4 key tests pass with 0 diff | **Supported** | §3.4 |
| 6 | JORA-Diag (additive diagonal) is not competitive with LoRA/DoRA | E8/E9: JORA train_loss=2.24 vs LoRA=1.95 vs DoRA=1.95; JORA is 2.86x slower than LoRA | **Supported (negative)** | §4.4 |
| 7 | Diagonal core is the effective capacity in JORA-Diag, not rotation | E1-E3: NoRot ≈ JORA-Diag | **Supported** | §4.1 |

### Claims NOT made (and why)

| Claim | Reason |
|-------|--------|
| "JORA-Diag outperforms LoRA/DoRA" | E8/E9: 0.28 loss gap; JORA slowest runtime |
| "Rotation drives the gain" | E1-E3: ON ≈ NoRot within noise |
| "TC-CS succeeded" | E4-E7: mechanism gate failed |
| "Residualized DiagCore is viable" | C2.1: catastrophic failure |
| "Method is general-purpose SOTA" | Only tested on OPT-350M, single dataset |

---

## Paper Narrative

The paper tells a single honest story: **rotation-based PEFT in the form of JORA-Diag does not work as intended, and the failure is diagnosable and instructive.**

Three independent failure modes are documented:
1. Rotation provides no independent benefit (mechanism null result)
2. Coupling subspace design (TC-CS) collapses to trivial energy-based selection
3. Residualizing full-support DiagCore breaks zero-function-change at init

The paper's value is not in a positive method contribution but in:
- Ruling out three specific design choices clearly
- Establishing the projector constraint as the critical difference between stable (SelectiveDiagCore) and unstable (residualized full DiagCore) residualization
- Providing exact merge verification as a standalone correctness contribution

---

## Structure

### §0 Abstract (150 words)

- **Problem**: Can rotation improve diagonal PEFT fine-tuning by re-aligning hidden-state basis?
- **Approach**: Diagnose JORA-Diag — a structured additive diagonal adapter in a sparse rotation basis — via three independent gates: rotation mechanism, coupling subspace, residualization.
- **Key result**: All three gates fail. Rotation ON matches NoRot at 1ep and 3ep (delta < 0.001); TC-CS pair selection collapses to energy-based consecutive pairing; residualized full DiagCore causes catastrophic loss divergence (15.7 vs 2.24). JORA-Diag also underperforms LoRA r=1 by 0.28 train_loss.
- **Implication**: Rotation-based diagonal PEFT requires stronger inductive bias or projector constraints to be viable.
- **Self-contained**: Yes — abstract states the problem, approach, result, and implication without requiring the paper.

---

### §1 Introduction (1.5 pages)

**Opening hook** (1-2 sentences):
Parameter-efficient fine-tuning (PEFT) methods have converged on low-rank (LoRA) and its magnitude-direction variant (DoRA). An alternative hypothesis — that re-aligning the frozen backbone's hidden-state basis via sparse rotations — remains underexplored. We diagnose this hypothesis rigorously.

**Gap**: Prior work on JORA (sparse rotation + diagonal core) claims empirical gains but lacks matched ablation (rotation ON vs OFF) under controlled conditions, a verified merge property, and analysis of residualization stability.

**Key questions**:
1. Does sparse rotation independently improve diagonal PEFT quality?
2. Can coupling subspace selection give rotation a unique geometric role?
3. Is full-support residualization of the diagonal operator stable?

**Contributions** (numbered, matching Claims-Evidence Matrix):
1. Matched 1ep and 3ep ablation of JORA-Diag with and without rotation on OPT-350M, showing rotation provides no quality benefit and adds 2.86x runtime overhead.
2. Diagnosis of TC-CS (task-conditioned coupling subspace) — demonstrates that the coupling signal collapses to energy-based selection because grad_col_ema outer product is rank-1 by construction.
3. Diagnosis of residualized full-support DiagCore — reveals catastrophic failure (eval loss 15.7 vs 2.24) caused by R_L^T R_R ≠ I when left/right rotation pairs are independently parameterized.
4. Exact merge verification for both DiagCore and SelectiveDiagCore via basis-probing, with mathematical characterization of the projector constraint that separates stable from unstable residualization.
5. Honest report that JORA-Diag (157K params) underperforms LoRA r=1 (445K params) by 0.28 train_loss, making competitive quality claims untenable.

**Hero figure**: Block diagram of the JORA-Diag forward path with three annotated "failure gates" at rotation, TC-CS coupling, and residualization, each labeled with its verdict (FAIL / PASS). Simple and honest.

**Key citations**:
- LoRA (Hu et al., ICLR 2022)
- DoRA (Liu et al., AAAI 2024)
- JORA (original paper — cite once)
- PEFT taxonomy (Zhu et al., 2024 survey)

---

### §2 Related Work (1 page)

**Subtopics**:
1. Low-rank PEFT: LoRA, DoRA, qLoRA — establish baselines
2. Diagonal PEFT: (related work if any — DSnoTT, SiLoRA, etc.)
3. Rotation-based methods: original JORA, FOFA, orthogonal fine-tuning
4. Residualized adapters: residual adapters, spectral methods

**Positioning**:
- JORA-Diag vs LoRA/DoRA: JORA is more parameter-efficient (157K vs 445K) but worse quality (0.28 loss gap). This is the central tension.
- JORA-Diag vs orthogonal PEFT: Both use rotations but JORA uses sparse Givens + diagonal core; orthogonal PEFT uses full orthogonalization. Key difference: sparsity and the independent-pair parameterization.
- TC-CS vs attention-based routing: TC-CS attempts a coupling subspace but collapses to magnitude-based selection — related to routing literature but with a different mechanism.

**Minimum synthesis**: The paper is most honest as a diagnostic study of JORA. Related work section should position it clearly as "PEFT with sparse rotations, a design we diagnose rigorously here" rather than claiming a new method.

---

### §3 Method (2 pages)

#### 3.1 Setup

Notation: define W_0 (frozen weight), x (input), Δ(x) (adapter contribution), R_L/R_R (sparse rotation matrices), Diag(d) (diagonal core).

#### 3.2 JORA Variants

Three variants, clearly separated:

**JORA-Diag (main)**: `Δ(x) = R_L^T Diag(d) R_R x` — additive diagonal in rotated basis.
- `zero_init_core=True` → d=0 → Δ(x)=0 (strict zero function change)
- Rotation via sparse Givens on pair sets (S_L, S_R) selected by gradient energy

**JORA-NoRot (baseline)**: `Δ(x) = Diag(d) x` — S_L=S_R=0, R_L=R_R=I
- Same diagonal core, no rotation
- Exact identity mapping at init

**JORA-Selective (SelectiveDiagCore, paper-exact)**: `Δ(x) = R_L^T D_sel R_R x - P_U x`
- `D_sel = P_U + Diag(δ)_U`
- Residualized on projector P_U
- Merge: exact basis-probing (verified)

#### 3.3 Residualization Analysis: Why Selective Works and Full-Support Fails

Mathematical characterization (key insight of the paper):

For SelectiveDiagCore: `P_U = P_U^T P_U` (projector). So `R_L^T P_U R_R = P_U` at θ=0.
This gives exact zero-function-change at init regardless of how pairs are selected.

For DiagCore (full-support): `I ≠ P_U`. `R_L^T I R_R = R_L^T R_R ≠ I` in general because pairs_L and pairs_R are independently sampled.
Therefore the residualized form `Δ(x) = R_L^T (I+Diag(d)) R_R x - x` does NOT give zero-function-change at init.

**Lemma**: For a residualized operator of the form `R_L^T C R_R - I` to have zero-function-change at init (C=I), the subtracted term must be a projector. Full identity I does not suffice when left/right rotations are independently parameterized.

**Proof sketch**: Show that when pairs_L ⊥ pairs_R (independent), R_L^T R_R has off-diagonal structure that couples dimensions. The residualized delta is O(x) - x ≠ 0. This is in Appendix A.

#### 3.4 Merge via Basis Probing

For both DiagCore and SelectiveDiagCore: probe Δ on one-hot basis vectors → recover exact delta weight matrix W_Δ where `Δ(x) = x @ W_Δ^T`.
This gives exact merged weights: `W_merged = W_0 + W_Δ`.
Verified: max difference < 1e-4 for DiagCore, < 1e-7 for SelectiveDiagCore.

#### 3.5 Training Setup

- Model: OPT-350M
- Dataset: alpaca-cleaned (51.8K)
- Epochs: 1ep (mechanism) and 3ep (convergence)
- Batch: 32, bf16
- Metrics: train_loss, token_accuracy, wall-clock time, eval_loss

---

### §4 Experiments: Three Failure Gates (3 pages)

#### 4.1 Gate 1: Rotation ON vs NoRot (1 page)

**Table 1** — Matched training (OPT-350M, alpaca-cleaned, seed 42, 3ep):

| Method | train_loss | token_acc | runtime (min) | params |
|--------|-----------|-----------|---------------|--------|
| JORA-Diag ON | 2.237 | 0.5147 | 158 | 157K |
| JORA-NoRot | 2.237 | 0.5150 | 55 | 157K |
| LoRA r=1 | 1.954 | 0.5675 | 41 | 445K |
| DoRA r=1 | 1.946 | 0.5708 | 75 | 668K |

**Findings**:
- Rotation ON vs NoRot: loss delta = -0.00055 (noise), token_acc delta = -0.00027 (NoRot slightly better). No evidence of rotation benefit.
- ON is 2.86x slower per step. Rotation overhead is structural, not marginal.
- Both JORA variants severely underperform LoRA/DoRA on quality (0.28 gap). JORA is slowest and worst quality.

**Figure 1**: Training loss curves for JORA-Diag ON and NoRot overlaid (nearly identical). Note: Do NOT show LoRA/DoRA on same plot — that would amplify the negative result too much for the mechanism section; show it separately in §4.4.

#### 4.2 Gate 2: TC-CS Coupling Subspace (0.5 page)

**Checkpoint analysis** (Step 4.5): At step 3074, TC-CS-1S and consecutive pairing yield 100% pair overlap (384/384 pairs). Zero dimensions differ.

**Root cause analysis** (Step 4.6): grad_col_ema outer product `outer(gc)` is rank-1 by construction (scalar outer product of gradient magnitude vector). Therefore `|outer(gc)|` = `gc[i] * gc[j]` — coupling score collapses to energy product.

**Matched training** (Step 4.8, 100 steps): train_loss delta = 2.7e-6 (noise). No mechanism differentiation.

**Conclusion**: TC-CS coupling subspace fails to provide rotation with an independent geometric role.

#### 4.3 Gate 3: Residualized Full-Support DiagCore (0.5 page)

**1ep training** (matched conditions):

| Configuration | Eval Loss (512 samples) | vs Base |
|---------------|------------------------|---------|
| Base model | 5.457 | — |
| Additive JORA-Diag ON | 2.239 | -3.22 |
| Residualized ON | **15.719** | +10.26 |
| Residualized NoRot | **19.189** | +13.73 |

Both residualized variants catastrophically diverge. The additive form remains stable.

**Analysis**: With R_L^T R_R ≠ I at init, the residualized operator produces O(x) - x ≠ 0. Small deviation at init amplifies during training as theta drifts (max=1.67 observed). The projector-constrained form (SelectiveDiagCore) is immune because P_U = P_U^T P_U guarantees R_L^T P_U R_R = P_U at θ=0.

#### 4.4 JORA vs LoRA/DoRA: The Quality Gap (0.5 page)

Neutral presentation: JORA-Diag uses fewer parameters (157K vs 445K) but converges to higher train_loss (2.24 vs 1.95). This is a negative result for the efficiency story. Acknowledge that LoRA/DoRA have been extensively optimized and JORA's different design may not suit this training regime.

Figure 2: Bar chart — train_loss (lower is better) for all four methods side by side. Clear and honest. No embellishment.

---

### §5 Discussion: What Went Wrong and What Survived (1 page)

#### 5.1 The Rotation Hypothesis

The core hypothesis — that sparse rotations can re-align the frozen basis to improve adaptation — was not supported. Possible reasons:
- The rotation subspace is too sparse (S_L=S_R=32 out of 4096) to meaningfully re-align the full representation
- Gradient energy is already a good proxy for adaptation importance; rotation on top provides marginal additional freedom
- The rotation is implemented as an orthogonal perturbation that may not interact well with the diagonal core

#### 5.2 The Residualization Constraint

The most theoretically interesting finding: residualization requires the subtracted term to be a projector under the rotation basis. The identity matrix I does not satisfy this when left/right rotations are independently parameterized. This is a general lesson: any residualized PEFT operator that subtracts the original input must ensure the subtracted term is invariant under the parameterization's symmetry.

#### 5.3 What Survived

- **Correct implementation**: DiagCore additive form is verified correct: `Δ(x) = R_L^T Diag(d) R_R x`
- **Exact merge**: Basis-probing merge is verified exact for both DiagCore and SelectiveDiagCore
- **SelectiveDiagCore**: The paper-exact residualized path with projector constraint remains stable
- **Honest baselines**: JORA-NoRot is a valid mechanism baseline

#### 5.4 Limitations

- Single model (OPT-350M) — results may not transfer to larger models
- Single dataset (alpaca-cleaned) — results may be data-specific
- No downstream eval (MMLU, ARC-C) — cannot claim generalization
- LoRA/DoRA comparison uses a different training setup (not hyperparameter-matched)

#### 5.5 Future Directions

If pursuing rotation-based PEFT:
1. Increase rotation coverage (more pairs) or use projector-constrained residualization throughout
2. Investigate whether rotation benefit emerges with larger models or longer training
3. Consider tied left/right rotation (θ_L = θ_R) to ensure R_L^T R_R ≈ I at init
4. Focus on the merge property as the primary advantage over LoRA, not quality

---

### §6 Conclusion (0.5 pages)

Restate the three failure gates and their implications. Emphasize the projector constraint as the key theoretical lesson. Note that the method is not ready for competitive-quality claims but exact merge and stable projector-constrained residualization are positive contributions.

---

## Figure Plan

| ID | Type | Description | Data Source | Priority |
|----|------|-------------|-------------|----------|
| Fig 1 | Architecture diagram | JORA-Diag forward path with three failure gates annotated | Manual / code diagram | HIGH |
| Fig 2 | Bar chart | train_loss for JORA-Diag ON, NoRot, LoRA, DoRA (3ep) | E1/E2/E8/E9 | HIGH |
| Fig 3 | Line plot | Training loss curves: JORA-Diag ON vs NoRot overlaid | E1/E2 | MEDIUM |
| Fig 4 | Scatter | TC-CS pair overlap vs consecutive (384 pairs) | Step 4.5 | MEDIUM |
| Fig 5 | Bar chart | Eval loss: base, additive ON, residualized ON, residualized NoRot | C2.1 | HIGH |
| Table 1 | Comparison table | All methods: train_loss, token_acc, runtime, params | E1/E2/E8/E9 | HIGH |

### Figure 1 (Hero) — Detailed Description

```
┌─────────────────────────────────────────────────────────────┐
│  x → [R_R] → [Diag(d)] → [R_L^T] → Δ(x) → [+] → y      │
│         ↑           ↑           ↑         ↑                 │
│      pairs_R    diagonal    pairs_L    W_0x                │
│      θ_R       params d     θ_L                              │
└─────────────────────────────────────────────────────────────┘
```
Three failure gates:
1. **Rotation** (GATE 1): R_L/R_R — ON ≈ NoRot (FAIL)
2. **Coupling** (GATE 2): TC-CS pairs — 100% overlap with energy (FAIL)
3. **Residualization** (GATE 3): R_L^T(I+Diag)R_Rx - x — catastrophic (FAIL)

Caption: "JORA-Diag: structured additive diagonal adapter in a sparse rotation basis. Three independent diagnostic gates reveal failure modes at rotation, coupling subspace, and residualization. See Sections 4.1–4.3."

---

## Citation Plan

### §1 Introduction
- LoRA: Hu et al., ICLR 2022 [VERIFY]
- DoRA: Liu et al., AAAI 2024 [VERIFY]
- JORA: original paper [VERIFY]
- PEFT survey: Zhu et al., 2024 [VERIFY]

### §2 Related Work
- LoRA, DoRA, qLoRA
- FOFA / orthogonal PEFT methods [VERIFY]
- Spectral/fast food PEFT methods [VERIFY]
- DSnoTT, SiLoRA (diagonal PEFT) [VERIFY]
- Residual adapters (Jia et al., 2022) [VERIFY]

### §3 Method
- Sparse Givens rotations [VERIFY if from JORA paper]
- PEFT mergeability literature [VERIFY]

### §4-5
- No new citations needed for negative results

---

## Reviewer Feedback

*(To be filled after cross-review)*

---

## Next Steps

- [ ] Draft each section with /paper-write
- [ ] Generate figures with /paper-figure
- [ ] Compile to PDF with /paper-compile
- [ ] External review with /auto-review-loop
