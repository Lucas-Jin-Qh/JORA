# JORA Research Contract

Last updated: 2026-04-28 (M0 correctness gate PASS)

This document freezes the current JORA project into a claim-driven research contract. Its purpose is to prevent paper writing, experiment planning, and implementation work from drifting beyond what the code and evidence currently support.

## Problem Anchor

- **Bottom-line problem**: Build a parameter-efficient, structured adapter for frozen LLM linear layers that is simple, carefully merge-qualified in deployment wording, and competitive enough to justify a focused paper story.
- **Must-solve bottleneck**: The project must determine whether JORA's rotation mechanism contributes anything beyond diagonal adaptation, rather than assuming that sparse rotations are the main source of gains.
- **Non-goals**:
  - Not proving that all JORA family variants are paper-worthy at once.
  - Not claiming rotation is the primary contributor without matched evidence.
  - Not expanding into a large benchmark matrix before the main mechanism verdict is settled.
  - Not reframing the project around LoRA-beating alone.
- **Constraints**:
  - Current codebase contains multiple operator semantics.
  - Current strongest evidence is training-side, not yet a matched final evaluation package.
  - Runtime cost matters because rotation-on is already much slower.
  - The project should remain one-paper, one-main-method, one-main-mechanism baseline.
- **Success condition**: A future paper should be able to state the main method, its strongest allowed claims, its mandatory caveats, and the exact evidence required for each claim without overreach.

## 1. Main method definition

### Canonical main method
`JORA-Diag` is the current main method.

In the current codebase, `JORA-Diag` should be described as a **structured additive adapter** applied to a frozen linear layer:

- Base layer: `y = W_0 x`
- Adapted layer: `y = W_0 x + Δ_φ(x)`
- Current DiagCore instantiation (verified against code):
  - `Δ_φ(x) = R_L^T Diag(d) R_R x`
  - NOT residualized. There is no `− x` term. The base residual `W_0 x` handles the identity.

where:
- `Diag(d)` is the diagonal core and should be treated as the main adaptation capacity source.
- `R_L` and `R_R` are sparse Givens-parameterized orthogonal transforms.
- Rotation is currently interpreted as **optional basis reparameterization**, not a validated primary contributor.
- The diagonal core is **additive**: it outputs `d ⊙ (R_R x)`, not `(1+d) ⊙ (R_R x) − x`. There is no `−x` residualization in the DiagCore path.

### First-class mechanism baseline
`JORA-NoRot` is not a cosmetic ablation. It is a first-class mechanism baseline:

- `Δ_φ(x) = Diag(d) x`
- This is the DiagCore path with `S_L = S_R = 0` (rotation parameters are `None`, identity). Verified exact by `test_diag_core_unmerge_equals_original`.
- Zero-init: when `zero_init_core=True`, `d = 0` so `Δ(x) = 0` — the adapter is strictly identity at init.
- This baseline exists to test whether the diagonal core alone explains the observed adaptation behavior.

### Paper-path variant
`JORA-Selective` remains an important variant, but not the current mainline method claim.

It corresponds to the paper-exact residualized selective-support path:
- `Δ_φ(x) = R_L^T D_sel R_R x − P_U x`

where `D_sel = I_U + Diag(δ)` (only the support indices have learnable diagonal entries; the complement is the identity). The `− P_U x` term residualizes the identity, ensuring the output of `D_sel` and the residual projector both have zero gradient w.r.t. `P_U` at `δ = 0`.

- `SelectiveDiagCore` is **zero-function-change at init only when both `theta = 0` and `delta = 0`** — verified by `test_selective_zero_function_change_at_init`.
- Merge: `SelectiveDiagCore` merge is **exact by construction** via basis probing (probes the rotation basis to recover the adapter's delta in the unrotated space). It is the only variant with an exact merge path.
- This variant is structurally cleaner in theory, but it is not the current main empirical mainline.

## 2. Allowed claims

The following claims are currently allowed because they are either supported by code inspection, repository structure, or the reported evidence.

### Allowed claim A — Main method identity
- `JORA-Diag` is the current practical mainline method in this repository.
- It is best described as a structured additive diagonal adapter, optionally expressed in a sparse rotation basis.
- It is not the residualized full-support operator unless the implementation is refactored.

### Allowed claim B — Diagonal core is the main capacity hypothesis
- The diagonal core is the main capacity source in current additive JORA-Diag.
- Rotation should currently be treated as an optional basis reparameterization layered on top of the additive diagonal core.

### Allowed claim C — JORA-NoRot is claim-determining
- `JORA-NoRot` is a first-class mechanism baseline and must be included in any serious interpretation of JORA-Diag.
- If JORA-Diag does not beat JORA-NoRot under matched evaluation, the paper must acknowledge that diagonal adaptation is the main effective component.

### Allowed claim D — Rotation evidence is null at both 1-epoch and 3-epoch horizons
- At 1-epoch: JORA-Diag ON and NoRot have nearly identical train loss.
- At 3-epoch (seed s42, OPT-350m SFT): final train loss is 2.2368 (Diag) vs 2.2378 (NoRot) — delta ~0.001, essentially indistinguishable.
- Rotation-on has a much higher runtime cost, roughly 3x in the reported comparison.
- Therefore, current evidence does not support a claim that rotation is a meaningful independent contributor.

### Allowed claim E — Operator semantics are not yet unified
- The repository contains at least two JORA operator semantics:
  - paper-exact residualized selective path (`SelectiveDiagCore`, exact merge by basis probing)
  - legacy additive diag/block/lowrank path (no residualization, approximate merge)
- Any final paper Method section must respect this distinction unless the code is refactored and re-validated.

### Allowed claim F — JORA-Diag uses additive, not residualized, diagonal adaptation
- `DiagCore` forward: `Δ(x) = R_L^T Diag(d) R_R x`. The base residual `W_0 x` handles identity.
- There is no `−x` term in the DiagCore path. This is NOT the full-support residualized form `Δ(x) = R_L^T Diag(1+d) R_R x − x`.
- The `diag_path` config sets `zero_init_core=True`, so the diagonal core starts at zero, making the initial adapter output exactly zero (strict zero-function-change).

### Allowed claim G — The project is not yet ready for aggressive narrative expansion
- The responsible next step is claim discipline, not adding more loosely connected story lines.
- The project should be written as a narrowly scoped structured-adaptation paper until stronger evidence exists.

## 3. Forbidden claims

The following claims are forbidden unless new matched evidence or refactoring explicitly changes the contract.

### Forbidden claim A — “Rotation drives the gain”
Do not claim that sparse rotations are the primary reason JORA works.

### Forbidden claim B — “JORA-Diag is already validated over JORA-NoRot”
Do not imply that the mechanism question is settled.

### Forbidden claim C — “The main method is the paper-exact residualized full-support operator”
Do not describe current JORA-Diag as:
- `Δ(x) = R_L^⊤ Diag(1 + d) R_R x - x`

unless the implementation is actually changed to that form and all relevant checks are rerun.

### Forbidden claim D — “Exact mergeability is uniformly established for the whole family”
Do not flatten all JORA variants into one merge story:
- `SelectiveDiagCore` has a dedicated exact merge path via basis probing — it is the only variant with a verified exact merge.
- `DiagCore` (JORA-Diag) uses a conservative approximate merge — it cannot safely be claimed as exact.
- `BlockCore`, `LowRankCore` similarly use approximate merge.

### Forbidden claim E — “Rotation helps” based only on train loss parity or non-matched runs
Do not use incomplete, mismatched, or one-sided evidence to upgrade the rotation story.

### Forbidden claim F — “The paper is about beating LoRA”
This is not the current main scientific question.
The main question is whether the structured rotated-basis parameterization adds value beyond diagonal adaptation.

### Forbidden claim G — “All JORA family variants are equally central”
Do not let BlockCore, LowRankCore, and SelectiveDiagCore dilute the main method story.

## 4. Claim-to-evidence matrix

| Claim | Status | Minimum evidence required | Current evidence status | Allowed wording now |
|---|---|---|---|---|
| `JORA-Diag` is the main method | Allowed | Code/config repo structure | Supported by `AGENTS.md` and configs | “JORA-Diag is the current mainline method.” |
| JORA-Diag is a structured additive diagonal adapter | Allowed | Formula audit against current code | Supported by current `DiagCore` path inspection | “Current JORA-Diag uses a structured additive diagonal core, optionally expressed in a sparse rotation basis.” |
| DiagCore is the main capacity source | Allowed, but still interpretive | ON vs NoRot comparisons and operator inspection | Supported directionally by current evidence | “Current evidence suggests the diagonal core is the main effective capacity source.” |
| Rotation is optional basis reparameterization | Allowed | Code semantics + null/weak ON vs OFF evidence | Supported as conservative framing | “Rotation is currently treated as optional basis reparameterization.” |
| Rotation improves optimization or final task quality | Not allowed yet | Matched ON vs OFF final evaluation, ideally multi-seed or strong paired evidence | Not supported | “Unresolved; no positive claim allowed.” |
| JORA-Diag beats NoRot | Not allowed yet | Matched evaluation at the metric level that matters for the paper | Not supported | “Unresolved.” |
| Rotation is worth its runtime cost | Not allowed yet | Clear metric gain large enough to offset ~3x slowdown | Not supported | “Current evidence argues against this.” |
| Main method is exact residualized full-support JORA | Not allowed yet | Implementation refactor + save/load/merge/forward revalidation + rerun experiments | Not supported | “Current mainline is additive DiagCore, not residualized full-support.” |
| Exact mergeability holds uniformly | Not allowed yet | Variant-specific exact merge validation | Not supported | "Merge handling differs across variants." |
| SelectiveDiagCore has exact merge | Allowed | Basis-probing implementation + unit test | Supported by code + test_selective_merge_equals_forward | "SelectiveDiagCore supports exact merge via basis probing." |
| JORA-Diag forward is additive, not residualized | Allowed | Formula audit | Supported by code inspection | "JORA-Diag forward: Δ(x)=R_L^T Diag(d)R_R x; no residualization term." |
| Selective variant is the main paper path | Not allowed as mainline | Mainline empirical repositioning back to selective + new evidence | Not the current project direction | "Selective is a variant, not the current mainline." |
| TC-CS coupling subspace rotation provides meaningful differentiation | Not allowed | Pair overlap < 80% on real activations, training benefit demonstrated | Not supported — Step 4.7 offline diagnostic confirms 100% overlap (rank-1 structural collapse), 2.7e-6 training delta | "TC-CS mechanism gate failed; coupling signal collapses to energy-based ordering. Documented as negative result / future work." |

## 5. Required experiments before paper writing

These are the minimum experiments that should be considered required before drafting a serious paper narrative.

### Experiment block 1 — Matched JORA-Diag vs JORA-NoRot evaluation
**Purpose**: Resolve the main mechanism question.
**Status: SETTLED** — R006/R007 (seed 42, 3ep) complete. ON ≈ NoRot in quality; ON is 2.86× slower. Rotation verdict: demote to optional basis reparameterization.

### Experiment block 2 — Longer-horizon matched ON/OFF comparison
**Purpose**: Avoid over-interpreting 1-epoch null results if convergence behavior differs later.
**Status: SUPERSEDED** — 3ep evidence (E1/E2) is now available. 3ep verdict is the same as 1ep: null quality effect, negative runtime. No further rotation ON/OFF training runs are needed.

### Experiment block 3 — Formula audit / implementation contract
**Purpose**: Prevent paper-method mismatch.

**Status: COMPLETED** — `docs/FORMULA_AUDIT.md` has been produced.

Key frozen findings:
- JORA-Diag forward: `Δ(x) = R_L^T Diag(d) R_R x` (additive, NOT residualized)
- JORA-NoRot forward: `Δ(x) = Diag(d) x` (strict subset of JORA-Diag with no rotations)
- JORA-Selective forward: `Δ(x) = R_L^T D_sel R_R x − P_U x` (residualized selective path)
- SelectiveDiagCore merge: exact via basis probing (only variant with verified exact merge)
- DiagCore/BlockCore/LowRankCore merge: conservative approximate (NOT exact)
- Zero-function-change at init: Diag/NoRot/Block/LowRank with `zero_init_core=True` are strictly zero; Selective requires both theta=0 and delta=0

Required deliverable: (superseded by actual artifact)
- `docs/FORMULA_AUDIT.md` records all formulas, merge semantics, and required test gates.

### Experiment block 4 — Basic deployment sanity package
**Status: PASS (27/27 tests, 2026-04-28)**

Required checks:
- save/load roundtrip
- merge/unmerge sanity
- no obvious path mismatch between train-time and inference-time adapter application

**Minimum outcome**: Know exactly which variants can support strong merge/deployment wording.

**Results**:
- `SelectiveDiagCore` merge: exact via basis probing — only variant with verified exact merge.
- `DiagCore` (JORA-Diag mainline) merge: approximate via `_compute_weight_delta_simple` (0.05x scaling approximation) — NOT exact.
- `NoRot` merge/unmerge: exact.
- Magnitude variants (ecd_tanh, oer_softmax): survive save/load; unmerge has ~5-10% relative error.

**Allowed wording**: "JORA supports weight-space merging; merge semantics differ by variant. `SelectiveDiagCore` supports exact merge via basis probing. `DiagCore` (JORA-Diag mainline) uses a conservative approximate merge path."

**Forbidden wording**: "JORA-Diag has exact merge equivalence" (the mainline DiagCore path is approximate).

### Optional but high-value follow-up
These are useful after the mechanism verdict, not before it.
- theta-init ablation refinement
- theta learning-rate sweep interpretation
- BlockCore and LowRankCore as appendix family checks
- LoRA-r2 or other stronger baselines for broader context

## 6. Reviewer attack surface

This section lists the most likely reviewer attacks if the paper is written too aggressively.

### Attack 1 — “Your gains come from diagonal scaling, not rotation.”
Why it is dangerous:
- Current evidence already points in this direction.
- Reviewer can use your own NoRot baseline against you.

Required defense:
- Either matched evidence that ON > OFF,
- or a narrative that openly centers diagonal adaptation and demotes rotation.

### Attack 2 — “Your method section does not match your code.”
Why it is dangerous:
- Current repository has different operator semantics across variants.
- A polished but wrong formula will be fatal under close reading.

Required defense:
- A paper formula set that exactly matches the code path used for the main experiments.

### Attack 3 — “Why pay ~3x runtime for no gain?”
Why it is dangerous:
- This is already true under the current 1-epoch and 3-epoch evidence.

Required defense:
- Show later-task or later-training benefit,
- or stop making a strong pro-rotation claim.

### Attack 4 — “This is not a focused paper; it is a family of loosely related adapters.”
Why it is dangerous:
- Block, LowRank, Selective, NoRot, rotation-only, init sweeps, and pairing strategies can make the paper look diffuse.

Required defense:
- Freeze one main method (`JORA-Diag`), one mechanism baseline (`JORA-NoRot`), and one appendix-efficient variant (`JORA-Selective`).

### Attack 5 — “Mergeability is oversold.”
Why it is dangerous:
- Merge semantics differ across code paths.

Required defense:
- State precisely which variant supports which level of merge claim.

### Attack 6 — “This is a runtime-expensive reparameterization without a stable empirical advantage.”
Why it is dangerous:
- It combines the mechanism and efficiency objections into one.

Required defense:
- A stronger matched evaluation package,
- or a narrower paper story that emphasizes structured diagonal adaptation rather than rotation wins.

### Attack 7 — “You are asking the reader to infer claims that your results do not prove.”
Why it is dangerous:
- This is the default failure mode for overbuilt research narratives.

Required defense:
- Every sentence in the paper should be traceable to a row in the claim-to-evidence matrix.

## Operating rules for future work

- Every new experiment must state which claim in this contract it is meant to support or reject.
- No paper writing should begin before the main method formula is frozen.
- No strong rotation narrative is allowed unless matched evaluation changes the verdict.
- If JORA-NoRot remains equal or better, the paper must reposition around diagonal adaptation as the main effective mechanism.
- If the evidence stays mixed, the project should prefer a narrower but honest paper over a broader but fragile one.
- TC-CS is paused. It may not re-enter the active experiment pipeline without a fundamentally different candidate-pool/score design that can pass a read-only overlap gate on real activations before any training run is launched.

## Current bottom line

At the current state of the project (updated 2026-04-28):
- `JORA-Diag` is the main method.
- `JORA-NoRot` is the claim-determining baseline.
- `DiagCore` is the main effective capacity hypothesis.
- Rotation is currently only justified as optional basis reparameterization; matched 3ep evidence shows no quality advantage and 2.86× runtime cost.
- TC-CS rotation revival attempt: **failed mechanism gate** (Step 4.7 failure analysis, 2026-04-28). Pair overlap = 100%, training loss delta = 2.7e-6, rank-1 structural collapse confirmed. Requires fundamentally different candidate-pool/score design before reconsideration.
- **M0 correctness gate: PASS** (27/27 tests, 2026-04-28). save/load roundtrip and merge equivalence verified. Exact merge only for `SelectiveDiagCore`; DiagCore (JORA-Diag mainline) uses approximate merge — not exact.
- The paper is not allowed to claim more than the evidence supports.
