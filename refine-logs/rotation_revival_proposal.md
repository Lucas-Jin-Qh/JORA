# Rotation Revival Proposal v2

Last updated: 2026-04-28

> **Status: MECHANISM GATE FAILED** — TC-CS-1S failed the Step 4.8 mechanism gate. Pair overlap on real OPT-350m activations = 100% (384/384 pairs identical to consecutive selection). 100-step matched training loss delta = 2.7e-6 (noise). Candidate-pool redesign required before further training experiments. Rotation is demoted to optional/baseline pending redesign or abandonment.

> **Narrative update**: Mainline story = JORA-Diag as additive diagonal PEFT. TC-CS = "attempted rotation revival; failed current gate due to pair collapse on real activations."

## Problem Anchor

- **Bottom-line problem**:
  The current JORA project needs a technically honest way to determine whether rotation can become a meaningful contributor in language adaptation rather than remaining a decorative basis reparameterization.

- **Must-solve bottleneck**:
  In the current additive JORA-Diag implementation, the diagonal core already provides enough capacity that sparse global rotation does not show a clear independent benefit. Matched ON/OFF evidence indicates that rotation currently adds substantial runtime cost without a commensurate quality gain.

- **Observed failure point in the current pipeline**:
  Current JORA-Diag uses
  - `Δ(x) = R_L^T Diag(d) R_R x`
  while JORA-NoRot uses
  - `Δ(x) = Diag(d) x`
  Under matched 1-epoch and 3-epoch evidence, ON and OFF remain nearly identical in quality, while ON is much slower. This implies the current rotation module does not carry a necessary language-adaptation role.

- **Language-side interpretation of the failure**:
  The current evidence suggests that many of the observed gains can be explained by per-dimension calibration alone. If rotation is to matter in language adaptation, it must target a narrower phenomenon than generic basis mixing — namely, cross-dimension coupling that diagonal-only scaling cannot express well.

- **Why naive fixes are insufficient**:
  Small tricks such as nonzero theta init, theta/core split learning rate, tanh removal, and mild schedule tuning can improve training hygiene, but they do not give rotation a unique task-conditioned function. Without such a function, diagonal scaling can continue to absorb most useful adaptation.

- **Non-goals**:
  - Not rescuing rotation by stacking many new modules.
  - Not expanding into a broad JORA-family paper with multiple core types in the main story.
  - Not claiming universal superiority over LoRA or DoRA before the mechanism question is resolved.
  - Not rewriting the current additive code path into a new residualized full-support operator in this proposal.
  - Not proving that generic sparse rotation helps by default.

- **Constraints**:
  - The current mainline method is additive JORA-Diag, not residualized full-support JORA.
  - The proposal must remain compatible with the project’s current claim discipline.
  - The method must stay small and focused enough for a compact EMNLP-style narrative.
  - Runtime cost remains a serious constraint; any revived rotation mechanism must justify its extra cost.
  - The story must sound like language adaptation geometry, not an overbuilt PEFT framework.

- **Success condition**:
  Rotation should remain alive in the main story only if a modified rotation design shows a clear, matched advantage over NoRot in a language adaptation regime that plausibly requires cross-dimension coupling, while remaining explainable as a necessary mechanism rather than a tuning artifact.

---

## Technical Gap

The core issue is not that JORA lacks rotation, but that its current rotation is too weakly assigned and too globally scoped.

Current JORA-Diag lets the diagonal core and sparse rotations act together, but the diagonal core can already explain most of the observable adaptation effect. Because rotation operates over a sparse global basis without a task-conditioned coupling target, it behaves like a free reparameterization whose effect can be absorbed by the diagonal core.

For a language adaptation paper, this means the current rotation module fails at the level that matters most:
- it does not identify **where** coupling is needed
- it does not identify **which side** of the transformation needs coupling first
- it does not distinguish **important dimensions** from **coupling-relevant dimensions**

Therefore, the operational technical gap is:

> Current JORA selects dimensions for importance, but not for coupling. As a result, rotation is applied too broadly and too weakly to capture the specific cross-dimension interactions that diagonal-only language adaptation cannot express.

A viable revival route must make rotation responsible for a narrower and more linguistically meaningful role.

---

## Method Thesis

### One-sentence thesis

**Restrict JORA rotation to a task-conditioned coupling subspace, and introduce that restriction on one side first, so that rotation models cross-dimension interaction only where language-task adaptation appears to require more than diagonal scaling.**

### Why this is the smallest adequate intervention

The smallest plausible rescue is not a brand-new adapter family. It is a scoped redefinition of where and how rotation is allowed to act:
- keep the diagonal core as the main adaptation mechanism
- keep sparse Givens-style rotation as the parameterization primitive
- replace weak global sparse rotation with **task-conditioned coupling-subspace rotation**
- start with **one-sided rotation restriction first** before introducing bilateral complexity

### Why one-sided first

The current failure mode is not obviously symmetric. Before adding a bilateral subspace design, the method should first answer a simpler question:

> Is one-sided coupling already enough to make rotation useful?

This yields a much cleaner mechanism test:
- NoRot: no coupling
- current JORA-Diag: weak global coupling on both sides
- proposed JORA: explicit localized coupling on one side first

This is easier to interpret, cheaper to implement, and more compatible with a focused paper story.

---

## Contribution Focus

- **Dominant contribution**:
  A task-conditioned coupling-subspace rotation mechanism for additive JORA-Diag, beginning with one-sided rotation restriction.

- **Optional supporting contribution**:
  A diagnostic framework for distinguishing coupling-relevant dimensions from merely high-magnitude dimensions in language adaptation.

- **Explicit non-contributions**:
  - Not a general orthogonal adaptation framework.
  - Not a claim that arbitrary sparse rotation helps language tasks by default.
  - Not a claim that current JORA-Diag has already validated rotation.
  - Not a broad PEFT-family comparison paper.

---

## Proposed Method

### Complexity Budget

- **Frozen / reused backbone**:
  - Keep the frozen base model and existing additive JORA-Diag structure.
  - Reuse current DiagCore as the primary adaptation capacity.

- **New trainable components**:
  - None beyond the existing diagonal and rotation parameters.
  - The main change is in **rotation scope selection**, not in adding a new trainable branch.

- **Tempting additions intentionally not used**:
  - No LowRankCore in the main method.
  - No auxiliary router or gating network.
  - No bilateral subspace design in v2.
  - No new magnitude module.
  - No large optimization schedule stack.

### System Overview

Current JORA-Diag:
- `Δ(x) = R_L^T Diag(d) R_R x`

Proposed v2 method:
1. Run a short calibration phase.
2. Identify a **task-conditioned coupling subspace** rather than a generic importance-ranked support.
3. Restrict **one side** of rotation first — preferably the input-side rotation `R_R` — to act only inside that subspace.
4. Keep the diagonal core unchanged.
5. Keep the opposite rotation side fixed, minimal, or absent in the first version.

Informally:
- diagonal core = per-dimension language adaptation capacity
- one-sided rotation = localized coupling operator in a task-conditioned subspace

### Core Mechanism

#### Input / output
- Input: hidden activation `x` for a frozen linear layer.
- Output: additive update `Δ(x)` on top of `W_0 x`.

#### One-sided coupling-first design

Instead of asking both `R_L` and `R_R` to help, v2 first restricts attention to one side — the input-side transformation — because that side is easier to interpret as feature coupling before dimension-wise rescaling.

A conceptual form is:
- `Δ(x) = Diag(d) · R_R^(S) x` in rotated coordinates before mapping back into the existing additive wrapper

Operationally, within the current JORA architecture, this means:
- keep the additive JORA-Diag skeleton
- restrict the active rotation budget to a calibrated subspace on the input side first
- only introduce left-side subspace restriction later if one-sided coupling clearly works

The important point is not the exact algebraic rewrite, but the role assignment:
- **DiagCore handles calibration**
- **one-sided restricted rotation handles coupling**

#### Why this could let rotation matter

NoRot can scale dimensions independently, but it cannot couple them. Current global sparse rotation tries to couple everything weakly and ends up redundant. A task-conditioned coupling subspace gives rotation a narrower mission:
- act only where language-task adaptation seems to require coordinated movement across dimensions
- avoid wasting rotation capacity on dimensions where diagonal scaling is already sufficient

### Representation Design

The key representational move is to replace **importance selection** with **coupling selection**.

A practical first version should not require a full covariance-learning system. Instead, it should use a lightweight coupling-sensitive proxy during calibration.

Examples of acceptable signals:
- output-side gradient energy
- input-side activation energy
- a simple co-activation or co-gradient proxy used only to decide which dimensions belong to the coupling subspace

The goal is not to find the globally best subspace in a heavy statistical sense. The goal is to find a subspace where rotation is more likely to model real interactions than to act as decorative reparameterization.

### Training Plan

A minimal training recipe:
1. **Calibration stage**:
   - collect per-dimension task statistics
   - derive a coupling-sensitive subspace estimate
2. **Freeze topology**:
   - freeze the chosen subspace and the one-sided rotation support inside it
3. **Optimization stage**:
   - train the diagonal core together with one-sided subspace-restricted rotation
4. **Diagnostics**:
   - log theta norms, theta gradients, and subspace activity to verify rotation is actually used

### Failure Modes and Diagnostics

- **Failure mode 1**: Rotation still collapses to NoRot-equivalent behavior.
  - **How to detect**: matched ON/OFF performance remains equal while theta norms or gradients remain weak or unstable.
  - **Fallback**: conclude that diagonal scaling remains the true mechanism and drop rotation from the main story.

- **Failure mode 2**: The selected subspace is importance-heavy but not coupling-relevant.
  - **How to detect**: the selected dimensions are stable and high-energy, but NoRot still matches the method.
  - **Fallback**: revise the calibration signal rather than adding more trainable structure.

- **Failure mode 3**: One-sided restriction is too weak, but bilateral restriction would overcomplicate the story.
  - **How to detect**: small but unstable gains that disappear across seeds.
  - **Fallback**: stop at the one-sided negative result instead of escalating to a larger method immediately.

- **Failure mode 4**: Rotation helps slightly but not enough to justify runtime.
  - **How to detect**: tiny quality gains with clearly worse step time.
  - **Fallback**: keep the mechanism as a negative-but-informative appendix result rather than a mainline claim.

### Novelty and Elegance Argument

This proposal is more defensible than adding more local training tricks because it changes the role of rotation at the mechanism level.

The paper is not claiming:
- that more sparse rotations are always better
- or that orthogonal mixing is generically useful

Instead it claims:
- diagonal adaptation handles most language-task recalibration
- rotation only becomes useful when localized to a task-conditioned coupling subspace
- the smallest valid test of that claim is one-sided rotation restriction first

That is a cleaner, more EMNLP-compatible mechanism story than a broad PEFT framework narrative.

---

## Minimal Validation

### Claim 1

**Claim**: Task-conditioned one-sided coupling subspace rotation provides a real benefit beyond additive diagonal scaling.

- **Minimal experiment**:
  Compare three systems under the same backbone, data, horizon, and seed protocol:
  1. JORA-NoRot
  2. current additive JORA-Diag
  3. one-sided coupling-subspace JORA-Diag

- **Baselines / ablations**:
  - NoRot is the decisive mechanism baseline.
  - Current additive JORA-Diag is the status-quo rotation baseline.

- **Metric**:
  - primary: matched final task quality metric
  - secondary: train loss, token accuracy, runtime, step time

- **Expected evidence**:
  - one-sided coupling-subspace JORA should beat both NoRot and current JORA-Diag by a non-trivial margin if rotation is truly revived.

### Claim 2

**Claim**: The benefit of revived rotation comes from coupling-aware scope restriction rather than from generic extra complexity.

- **Minimal experiment**:
  Compare:
  1. one-sided coupling-subspace JORA
  2. current global sparse rotation JORA

- **Metric**:
  - final quality and runtime

- **Expected evidence**:
  - the coupling-subspace version should be more effective or more efficient than unconstrained sparse global rotation.

### Deletion / simplification check

**Question**: Can the same gain be recovered without rotation at all?

- **Minimal check**:
  Compare one-sided coupling-subspace JORA directly against NoRot with identical diagonal-core budget.

- **Interpretation**:
  - If NoRot still matches it, rotation should be abandoned.

---

## Experiment Handoff Inputs

- **Must-prove claims**:
  - rotation can help when localized to a task-conditioned coupling subspace
  - one-sided restricted rotation is enough to produce an interpretable gain, if a real gain exists

- **Must-run ablations**:
  - NoRot
  - current additive JORA-Diag
  - one-sided coupling-subspace JORA-Diag

- **Critical metrics**:
  - final task metric
  - runtime / step time
  - theta norm / grad diagnostics
  - subspace stability diagnostics

- **Highest-risk assumption**:
  - that a coupling-relevant subspace exists in the language adaptation setting and can be identified cheaply enough to justify restricted rotation

---

## Compute & Timeline Estimate

- **Estimated GPU-hours**:
  Moderate. The first pass remains compact because the method comparison only needs three core systems.

- **Data / annotation cost**:
  None beyond existing training/evaluation pipelines.

- **Timeline**:
  1. implement one-sided coupling-subspace restriction
  2. run one matched sanity comparison against NoRot and current JORA-Diag
  3. only expand if the result is clearly positive

---

## Final Recommendation

If rotation is to remain alive as a serious research direction in JORA, it should be revived through a **task-conditioned coupling role**, not another layer of local tuning tricks.

The most promising next step is:

> **Task-Conditioned Coupling Subspace Rotation for Additive JORA-Diag, with one-sided rotation first**

If this still fails to beat NoRot in a matched comparison, the project should stop treating rotation as a mainline contributor.
