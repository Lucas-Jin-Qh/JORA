# Experiment Plan

**Problem**: Determine whether rotation adds value beyond diagonal scaling in JORA-Diag, and only then assess whether JORA-Diag is competitive with tuned low-rank baselines at comparable or lower parameter count.
**Method Thesis**: Current JORA-Diag is a structured additive diagonal adapter with forward delta `Δ(x)=R_L^⊤ Diag(d) R_R x`; its diagonal core is the primary adaptation mechanism, while sparse rotations are treated as an optional basis reparameterization that must be justified by matched evidence.
**Date**: 2026-04-26

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|-------|-----------------|-----------------------------|---------------|
| C1. Rotation adds value beyond diagonal scaling, or does not | This is the main mechanism verdict for the current additive JORA-Diag implementation and controls the entire paper narrative | Matched ON vs NoRot comparison with the same model, data, horizon, metrics, and seeds | B1, B2 |
| C2. JORA-Diag is competitive with strong PEFT baselines at a similar or lower parameter budget | This determines whether the method is worth presenting as a main practical method | Comparison against tuned LoRA and DoRA under a fair parameter-budget regime after the mechanism story is frozen | B3 |
| A1. Anti-claim to rule out: any observed effect comes only from parameter count or unrelated training differences | Reviewers will default to this explanation if configs are not tightly matched | Parameter-matched and protocol-matched comparisons with explicit run metadata | B1, B3 |
| A2. Anti-claim to rule out: the paper story is overbuilt and rotation is decorative | This is the main simplicity risk | A clean NoRot baseline plus a compact appendix-only variant strategy | B2, B4 |

## Paper Storyline
- **Main paper must prove**:
  - Whether rotation adds anything beyond the current additive diagonal scaling mechanism.
  - Whether the final chosen additive JORA-Diag main method is competitive with tuned PEFT baselines under a fair budget.
- **Appendix can support**:
  - Selective JORA as a Pareto/efficiency operating point.
  - Extra theta-init and LR ablations if they help explain behavior.
  - BlockCore and LowRankCore only as family checks, not as mainline evidence.
- **Experiments intentionally cut from the main plan**:
  - LowRankCore mainline comparisons.
  - Large multi-benchmark sweeps before the mechanism verdict.
  - Broad pairing-strategy sweeps before the core ON/OFF question is settled.

## Experiment Blocks

### Block 1: Fast Mechanism Diagnostic — Rotation ON vs NoRot
- **Claim tested**: C1.
- **Why this block exists**: This is the highest-priority decision gate. The comparison is specifically between the current additive JORA-Diag operator and its NoRot additive baseline. If JORA-Diag does not beat NoRot, the rotation story must be demoted.
- **Dataset / split / task**: Alpaca-cleaned training task on the existing OPT-350M setup used in current repo runs.
- **Compared systems**:
  - JORA-Diag ON (`run_diag_main_*` style)
  - JORA-NoRot (`run_diag_no_rotation_*` style)
- **Metrics**:
  - Primary: final train loss
  - Secondary: mean token accuracy, runtime, trainable parameter count
- **Setup details**:
  - Backbone: `facebook/opt-350m`
  - Same dataset and preprocessing
  - Same batch size / grad accumulation / optimizer schedule
  - Same training horizon per pair
  - Start with seed `42` for a fast decision pass
  - Use existing config family before inventing new config shapes
- **Success criterion**:
  - If ON improves token accuracy by at least a practically non-trivial margin and/or improves loss consistently enough to offset runtime, rotation remains viable.
  - If ON ≈ OFF, the project must treat diagonal scaling as the main effective mechanism.
- **Failure interpretation**:
  - Negative or null result means rotation is not currently justified as a primary contributor.
- **Table / figure target**:
  - Main paper: a compact mechanism table `ON vs NoRot` with runtime.
- **Priority**: MUST-RUN.
- **Method wording lock for this block**:
  - interpret ON as additive `R_L^⊤ Diag(d) R_R x`
  - interpret OFF as additive `Diag(d)x`
  - do not analyze this block as residualized full-support JORA

### Block 2: Longer-Horizon Mechanism Verdict
- **Claim tested**: C1 under a less myopic horizon.
- **Why this block exists**: Current evidence suggests 1 epoch may be too early to judge convergence behavior.
- **Dataset / split / task**: Same Alpaca-cleaned / OPT-350M setup.
- **Compared systems**:
  - JORA-Diag ON, matched 3-epoch run
  - JORA-NoRot, matched 3-epoch run
- **Metrics**:
  - Primary: final token accuracy and final loss after full horizon
  - Secondary: epoch-wise learning curves, runtime, parameter count
- **Setup details**:
  - Use matched 3-epoch configs such as `run_diag_main_s42_3ep.json` and `run_diag_no_rotation_s42_3ep.json`
  - First run at seed `42`
  - Expand to 3 seeds only if the single-seed result remains ambiguous or promising
- **Success criterion**:
  - Rotation must show a stable quality advantage that is not swallowed by noise and is large enough to discuss despite runtime cost.
- **Failure interpretation**:
  - If ON remains equal or worse, the main paper should reposition around diagonal adaptation with NoRot as the effective mechanism.
- **Table / figure target**:
  - Main paper learning-curve figure or main appendix mechanism verdict plot.
- **Priority**: MUST-RUN.

### Block 3: Strong Baseline Check — Tuned LoRA / DoRA at Comparable Budget
- **Claim tested**: C2.
- **Why this block exists**: Even if JORA-Diag has a clean internal story, it still needs to be practically meaningful against strong PEFT baselines.
- **Dataset / split / task**: Same primary training setting first; only move to broader evaluation after the mechanism verdict is frozen.
- **Compared systems**:
  - Final chosen main JORA system after B1/B2 verdict
  - Tuned LoRA at comparable or slightly stronger budget
  - Tuned DoRA at comparable or slightly stronger budget if implementation is available and stable in this repo
- **Metrics**:
  - Primary: final task metric chosen for the paper's main table
  - Secondary: train loss, token accuracy, trainable params, runtime
- **Setup details**:
  - Use a fair budget band rather than exact identical parameter count when exact equality is impossible
  - Prefer one strong LoRA config and one strong DoRA config over many weak ranks
  - Seeds: start with `42`, expand to `3` only if the comparison is close enough to affect claims
- **Success criterion**:
  - JORA-Diag should either match or exceed tuned baselines at a lower or comparable trainable parameter count, or offer a cleaner mechanism story with competitive performance.
- **Failure interpretation**:
  - If tuned LoRA/DoRA clearly win, the paper must narrow its claim to structured diagonal adaptation rather than strong practical superiority.
- **Table / figure target**:
  - Main paper baseline table.
- **Priority**: MUST-RUN, but only after B1/B2 freeze the mechanism story.

### Block 4: Appendix Pareto Check — Selective JORA
- **Claim tested**: A2 supporting appendix claim.
- **Why this block exists**: Selective JORA is useful as a low-budget Pareto point, but it should not control the main narrative.
- **Dataset / split / task**: Same primary task setting, possibly reduced to a compact but fair run set.
- **Compared systems**:
  - JORA-Selective
  - Main JORA-Diag or NoRot anchor
  - One compact LoRA anchor if needed for Pareto context
- **Metrics**:
  - Primary: parameter count vs task quality
  - Secondary: runtime and stability
- **Setup details**:
  - Keep this in appendix-only budget
  - No need for broad multi-seed unless it becomes a surprising highlight
- **Success criterion**:
  - Show that Selective JORA provides an efficiency operating point worth mentioning.
- **Failure interpretation**:
  - If weak, cut from mainline discussion and keep only as an internal note.
- **Table / figure target**:
  - Appendix Pareto figure.
- **Priority**: NICE-TO-HAVE.

### Block 5: Contract Sanity Block — Formula / Merge / Save-Load Checks
- **Claim tested**: supports all writing claims indirectly.
- **Why this block exists**: Prevent paper-code mismatch and overclaiming around deployment.
- **Dataset / split / task**: Unit/integration validation, not a benchmark task.
- **Compared systems**:
  - JORA-Diag
  - JORA-NoRot
  - JORA-Selective
- **Metrics**:
  - Pass/fail on forward consistency, save/load roundtrip, merge/unmerge sanity
- **Setup details**:
  - Reuse existing tests and add a compact formula-audit artifact if missing
- **Success criterion**:
  - Freeze allowed wording for method equations and mergeability.
- **Failure interpretation**:
  - Paper writing should pause until the mismatch is resolved or explicitly bounded.
- **Table / figure target**:
  - Not a paper table; internal contract gate.
- **Priority**: MUST-RUN.

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|-----------|------|------|---------------|------|------|
| M0 | Contract sanity | save/load, merge, formula audit, existing JORA tests | If method wording is not frozen, do not write paper text | Low | Hidden semantic mismatch |
| M1 | Fast ON vs OFF mechanism pass | B1 seed-42 matched pair | If ON ≈ OFF, demote rotation immediately | Low–Medium | Over-reading noisy early results |
| M2 | Longer-horizon ON vs OFF verdict | B2 matched 3-epoch pair | If ON still ≈ OFF or worse, freeze NoRot-centered story | Medium | Costly if the story is already dead |
| M3 | Variance check only if needed | Selected 3-seed expansion for ON/OFF | Run only if the single-seed verdict is close enough to matter | Medium | Spending compute on a dead story |
| M4 | Strong baseline check | B3 tuned LoRA and DoRA comparisons | Only run after the mechanism story is frozen | Medium | Baseline sprawl |
| M5 | Appendix efficiency support | B4 Selective JORA Pareto point | Optional; do not delay mainline story | Low–Medium | Distracts from main claim |

## Compute and Data Budget
- **Total estimated GPU-hours**:
  - M0: negligible
  - M1: low, based on fast 1-epoch matched pair
  - M2: moderate, based on 3-epoch matched pair
  - M3: moderate if expanded to 3 seeds
  - M4: moderate, depending on tuned LoRA/DoRA replication
  - M5: optional appendix budget
- **Data preparation needs**: reuse current Alpaca-cleaned pipeline; avoid adding new data until the mechanism story is stable.
- **Human evaluation needs**: none in the main minimal plan.
- **Biggest bottleneck**: wasting compute on full baseline families before the ON/OFF mechanism verdict is frozen.

## Risks and Mitigations
- **Risk**: Rotation remains neutral and the main narrative collapses.
  - **Mitigation**: Reposition the paper around structured diagonal adaptation and make NoRot central.
- **Risk**: Longer-horizon behavior differs from 1 epoch.
  - **Mitigation**: Keep B2 as a required matched follow-up before final narrative lock.
- **Risk**: Strong baselines beat JORA-Diag cleanly.
  - **Mitigation**: Narrow the practical claim and emphasize structural/diagnostic contribution only if still worthwhile.
- **Risk**: Implementation semantics and paper formula drift apart.
  - **Mitigation**: M0 is a hard gate before paper writing.
- **Risk**: Experiment scope explodes.
  - **Mitigation**: Do not run LowRankCore or large family sweeps in the main plan.

## Final Checklist
- [ ] Main paper mechanism verdict is covered
- [ ] JORA-NoRot is treated as a first-class baseline
- [ ] Strong baseline comparison is deferred until after the mechanism decision
- [ ] LowRankCore is excluded from the main plan
- [ ] Selective JORA is appendix-only
- [ ] Formula / merge contract is frozen before paper writing
- [ ] Nice-to-have runs are separated from must-run runs
