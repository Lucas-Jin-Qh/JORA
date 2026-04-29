# Experiment Plan

**Problem**: Determine whether task-conditioned one-sided coupling-subspace rotation can give additive JORA-Diag a real advantage over NoRot in a language adaptation regime where diagonal-only scaling is currently sufficient.
**Method Thesis**: TC-CS-1S replaces global importance-driven sparse rotation with one-sided task-conditioned coupling-subspace rotation, so that rotation is only used where empirical activation covariance suggests cross-dimension coupling beyond diagonal scaling.
**Date**: 2026-04-27

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|-------|-----------------|-----------------------------|---------------|
| C1. One-sided coupling-subspace rotation adds value beyond additive diagonal scaling | This is the core mechanism claim and the only reason to keep rotation alive | TC-CS-1S beats both NoRot and current additive JORA-Diag under matched settings | B1, B2 |
| C2. The benefit is layer-type specific and strongest in attention projections | This connects the method to a language-relevant coupling regime rather than a generic PEFT trick | Attention-only TC-CS-1S outperforms FFN-only and explains why all-linear should or should not be used | B2, B3 |
| A1. Anti-claim: gains come only from more parameters or retuning | Reviewers will default to this if the comparison is not tightly matched | Parameter-matched and protocol-matched comparison against NoRot and current JORA-Diag | B1, B2 |
| A2. Anti-claim: global sparse rotation is already enough, coupling restriction is decorative | This determines whether the new idea is real or just repackaged tuning | TC-CS-1S beats current additive JORA-Diag global rotation under matched budget | B2 |

## Paper Storyline
- **Main paper must prove**:
  - rotation can matter when localized to a task-conditioned coupling subspace
  - one-sided restriction is enough to produce an interpretable mechanism gain if a real gain exists
  - attention projections are the primary layer regime where coupling is useful
- **Appendix can support**:
  - all-linear TC-CS-1S context
  - FFN-only negative control details
  - NoRot robustness across additional seeds
  - current JORA-Diag vs TC-CS-1S diagnostics such as theta norms and subspace stability
- **Experiments intentionally cut**:
  - bilateral TC-CS-2S in the first paper pass
  - LowRankCore in the main story
  - broad benchmark shopping before mechanism validation
  - Selective JORA as a mainline comparison

## Experiment Blocks

### Block 1: Anchor Result — Can TC-CS-1S beat NoRot?
- **Claim tested**: C1.
- **Why this block exists**: If TC-CS-1S does not beat NoRot under matched conditions, the rotation revival story should stop.
- **Dataset / split / task**: Same primary language-adaptation setup already used by the project: OPT-350M on Alpaca-cleaned.
- **Compared systems**:
  - JORA-NoRot
  - current additive JORA-Diag
  - TC-CS-1S (attention-only first)
- **Metrics**:
  - Primary: final task quality metric used by the project for matched comparison
  - Secondary: final train loss, mean token accuracy, runtime, step time
- **Setup details**:
  - backbone: `facebook/opt-350m`
  - identical optimizer schedule, batch size, epochs, and seed protocol
  - preserve additive JORA-Diag forward path; only change calibration/pairing/scope selection
  - begin with seed `42`
- **Success criterion**:
  - TC-CS-1S must outperform NoRot by a non-trivial margin on the primary quality metric and also beat current additive JORA-Diag.
- **Failure interpretation**:
  - If TC-CS-1S ≈ NoRot, rotation should be abandoned as a mainline contributor.
- **Table / figure target**:
  - Main paper Table 1: matched NoRot vs current JORA-Diag vs TC-CS-1S.
- **Priority**: MUST-RUN.

### Block 2: Novelty Isolation — Is coupling-subspace restriction better than current global sparse rotation?
- **Claim tested**: C1 and A2.
- **Why this block exists**: The paper must show that the gain, if any, comes from coupling-aware restriction rather than from merely retrying rotation.
- **Dataset / split / task**: Same primary OPT-350M / Alpaca-cleaned setting.
- **Compared systems**:
  - current additive JORA-Diag (global sparse rotation)
  - TC-CS-1S attention-only
  - JORA-NoRot
- **Metrics**:
  - Primary: matched final task quality
  - Secondary: theta norm/grad diagnostics, runtime, selected-subspace statistics
- **Setup details**:
  - same parameter budget as closely as possible
  - same single-sided setting for the new method where applicable
  - same training horizon and seed
- **Success criterion**:
  - TC-CS-1S must exceed current additive JORA-Diag in the same budget regime, while also separating from NoRot.
- **Failure interpretation**:
  - If TC-CS-1S only beats current JORA-Diag but not NoRot, then global rotation was bad but rotation itself is still not necessary.
- **Table / figure target**:
  - Main paper ablation table or Figure 3 mechanism comparison.
- **Priority**: MUST-RUN.

### Block 3: Coupling Regime Check — Attention-only vs FFN-only vs All-linear
- **Claim tested**: C2.
- **Why this block exists**: The proposal explicitly claims that coupling is most relevant in attention projections, not uniformly across all linear layers.
- **Dataset / split / task**: Same primary language-adaptation setting.
- **Compared systems**:
  - TC-CS-1S attention-only
  - TC-CS-1S FFN-only
  - TC-CS-1S all-linear
  - JORA-NoRot anchor
- **Metrics**:
  - Primary: final task quality
  - Secondary: runtime, step time, subspace stability, per-layer diagnostic summaries
- **Setup details**:
  - use the same calibration rule and one-sided rotation design across layer scopes
  - keep the number of active rotated modules recorded explicitly
- **Success criterion**:
  - attention-only should be the strongest or most efficient positive regime if the coupling hypothesis is right.
- **Failure interpretation**:
  - If FFN-only matches or beats attention-only, the coupling-regime story is weak.
  - If all-linear is best but only marginally, the paper must justify whether the extra scope is worth the complexity.
- **Table / figure target**:
  - Main paper scope ablation table.
- **Priority**: MUST-RUN.

### Block 4: Simplicity Check — One-sided first is enough
- **Claim tested**: supports the minimality of the method.
- **Why this block exists**: The proposal argues that one-sided rotation should be the smallest adequate mechanism. The paper needs one compact check that it was right not to jump to bilateral complexity.
- **Dataset / split / task**: Same primary setting, but can be done on a reduced confirmation budget if compute is tight.
- **Compared systems**:
  - TC-CS-1S one-sided
  - current additive JORA-Diag two-sided baseline
- **Metrics**:
  - Primary: final task quality
  - Secondary: runtime and step time
- **Setup details**:
  - no new bilateral TC-CS-2S method in the first pass
  - use the already-implemented current JORA-Diag as the “broader/less targeted” comparator
- **Success criterion**:
  - one-sided TC-CS-1S should be competitive with or superior to the broader current two-sided global-rotation baseline.
- **Failure interpretation**:
  - If one-sided is uniformly weaker, the proposal may need bilateral expansion, but that should be deferred to phase 2 rather than rushed into the first paper.
- **Table / figure target**:
  - Main paper or appendix simplicity table.
- **Priority**: MUST-RUN.

### Block 5: Diagnostic and Failure Analysis
- **Claim tested**: supports interpretability of the mechanism, not a primary paper claim.
- **Why this block exists**: If the method wins or loses, the paper should still explain whether the coupling subspace is stable and whether theta is actually active.
- **Dataset / split / task**: Same primary setting.
- **Compared systems**:
  - TC-CS-1S attention-only
  - current additive JORA-Diag
- **Metrics**:
  - theta norms and grad norms
  - selected subspace overlap across seeds
  - calibration-time covariance magnitude summaries
- **Setup details**:
  - reuse existing diagnostic infrastructure where possible
- **Success criterion**:
  - diagnostics should reveal that TC-CS-1S uses rotation in a more targeted and interpretable way than current JORA-Diag.
- **Failure interpretation**:
  - unstable subspaces or dead theta behavior weaken the mechanism story even if quality moves slightly.
- **Table / figure target**:
  - Appendix diagnostic figure.
- **Priority**: NICE-TO-HAVE.

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|-----------|------|------|---------------|------|------|
| M0 | Sanity and instrumentation | config parsing, forward-pass covariance EMA, subspace freeze, save/load of new buffers if needed | If calibration stats are wrong or unstable, do not launch full comparisons | Low | Silent implementation bugs in calibration logic |
| M1 | Re-anchor baselines | reuse NoRot and current additive JORA-Diag matched references | If baseline numbers are not trusted, pause and normalize reporting | Low | Baseline drift |
| M2 | Main mechanism test | TC-CS-1S attention-only vs NoRot vs current JORA-Diag | If TC-CS-1S does not beat NoRot, stop the rotation revival story | Medium | Main claim fails early |
| M3 | Coupling regime ablation | attention-only vs FFN-only vs all-linear | If attention-only is not special, the language-coupling narrative weakens | Medium | Regime claim collapses |
| M4 | Simplicity confirmation | one-sided TC-CS-1S vs current broader global rotation | If one-sided loses badly, decide whether bilateral phase-2 is justified | Medium | Scope expansion pressure |
| M5 | Diagnostics and robustness | extra seeds + diagnostic summaries | Only run if M2 is promising | Medium | Spending compute on a dead story |

## Compute and Data Budget
- **Total estimated GPU-hours**:
  - M0: low
  - M1: low if existing baselines can be reused reliably
  - M2: moderate, primary must-run block
  - M3: moderate, three-way layer-scope comparison
  - M4: low–moderate
  - M5: optional moderate robustness budget
- **Data preparation needs**: reuse current Alpaca-cleaned training pipeline and existing matched comparison protocol.
- **Human evaluation needs**: none in the first experimental pass.
- **Biggest bottleneck**: the main risk is not scale, but whether TC-CS-1S can beat NoRot at all under matched conditions.

## Risks and Mitigations
- **Risk**: TC-CS-1S still matches NoRot.
  - **Mitigation**: stop the rotation revival mainline immediately and document the negative result.
- **Risk**: attention-only is not actually the coupling regime.
  - **Mitigation**: use Block 3 to directly test the layer-scope hypothesis instead of assuming it.
- **Risk**: covariance-based calibration is noisy or unstable.
  - **Mitigation**: add a minimal subspace stability diagnostic before broadening the experiment suite.
- **Risk**: quality gains are too small to justify runtime.
  - **Mitigation**: runtime is a first-class metric in every main block; a weak quality gain is not enough.
- **Risk**: the implementation changes too many things at once.
  - **Mitigation**: preserve additive `compute_delta()` and only change calibration, pairing, and one-sided scope.

## Final Checklist
- [ ] Main paper tables are covered
- [ ] Novelty is isolated against current additive JORA-Diag
- [ ] NoRot remains the decisive mechanism baseline
- [ ] Attention-only coupling regime is explicitly tested
- [ ] One-sided simplicity claim is defended
- [ ] Nice-to-have diagnostics are separated from must-run results
