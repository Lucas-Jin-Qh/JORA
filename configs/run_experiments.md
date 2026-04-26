# JORA Refactoring & Experiments Playbook

## Purpose

This document is a reliability-first execution playbook for refactoring JORA and running the core experiment ladder.
The goal is not to maximize speed; the goal is to make every step auditable, verifiable, and reversible.

Core rule:

> Do not start the next phase until the current phase has passed its validation gate.

---

## Guiding Principles

- One phase, one objective.
- Every code change must have an immediate local verification step.
- Every experiment phase must define:
  - entry condition
  - exact changes
  - validation gate
  - rollback / fallback action
- If a result contradicts the hypothesis, stop and update the plan before proceeding.
- Prefer the smallest test or run that can falsify the current assumption.

---

## High-Level Strategy

JORA is repositioned as a structured adaptation family in a learned rotated basis:

- `JORA-Diag`: main method
- `JORA-Block`: capacity expansion variant
- `JORA-LowRank`: bridge/control variant
- `SelectiveDiagCore`: efficiency appendix / operating point

The implementation and experiments should prove four things in order:

1. The refactored implementation is correct and stable.
2. Nonzero small initialization is safe and useful.
3. `JORA-Diag` is a viable main method.
4. Rotation contributes beyond plain diagonal scaling.

---

## Phase 0 — Baseline Audit

### Objective
Establish a trustworthy baseline before any modification.

### Tasks
- Record current behavior of:
  - `JoraConfig.paper_path()`
  - current theta initialization path
  - current core initialization path
  - current `compute_delta()` behavior for diag/block/lowrank/selective
- Run existing JORA tests.
- Save a short note with:
  - passing tests
  - known failures
  - current default configs used in previous experiments

### Validation Gate
Pass all existing relevant tests or clearly document current failures.

Suggested commands:

```bash
pytest tests/test_jora.py tests/test_jora_paper_path.py -q
```

### Rollback / Fallback
- If baseline tests already fail, stop refactoring.
- First isolate whether failures are unrelated legacy issues or current regressions-in-waiting.

### Deliverables
- Baseline test log
- Baseline config snapshot

---

## Phase 1 — Initialization Correctness Before Refactor

### Objective
Verify what “small nonzero init” should preserve.

### Hypothesis
A good initialization should be:
- near-identity / near-zero-update
- numerically stable
- symmetry-breaking enough to help optimization

### Tasks
- Add or run minimal probes for each core:
  - `DiagCore`
  - `BlockCore`
  - `LowRankCore`
  - `SelectiveDiagCore`
- For random input `x`, measure:
  - `||delta(x)|| / ||x||`
  - parameter norms after init
  - whether bf16/fp32 behavior is consistent enough

### Validation Gate
For each core under default init:
- output perturbation is small
- no NaN / Inf
- perturbation magnitude is repeatable across seeds

### Decision Rule
- If small nonzero init causes large perturbation, reduce init scale.
- Do not proceed to config refactor before init ranges are calibrated.

### Recommended Initial Sweep
- `theta_init_std`: `0`, `1e-3`, `2e-3`, `5e-3`
- `core_init_std`: `1e-3`, `5e-3`, `1e-2`

### Notes on Initialization
Use parameter-type-aware init instead of blindly applying Kaiming everywhere:
- rotation parameters (`theta`): small Gaussian only
- diagonal residual params: small Gaussian only
- block params: identity plus small Gaussian noise
- low-rank params: LoRA-style small init (`A` small, `B` zero or tiny)

Kaiming is acceptable only for auxiliary linear projections, not for raw rotation angles.

---

## Phase 2 — Config Refactor (Minimal Surface Area)

### Objective
Refactor config paths without changing semantics more than necessary.

### Entry Condition
Phase 1 has identified safe init ranges.

### Tasks
Modify `src/peft/tuners/jora/config.py` in the smallest safe order:

1. Add `core_init_std`
2. Add `diag_path()`
3. Add `selective_path()`
4. Only then consider changing `paper_path()` default from selective to diag

### Important Rule
Do **not** change `paper_path()` first.

Reason:
- it changes the semantic meaning of existing scripts
- it makes regression attribution harder

### Validation Gate After Each Substep
After each substep, run targeted checks:

```bash
pytest tests/test_jora.py tests/test_jora_paper_path.py -q
```

After adding factories, add targeted config tests before changing defaults.

### Rollback / Fallback
If changing `paper_path()` breaks old tests or old configs:
- keep `paper_path()` backward-compatible
- add a new `main_path()` / `diag_path()` for the new paper line
- postpone default flip until all experiment scripts are migrated

### Deliverables
- New config field
- New factory methods
- Explicit migration note on old vs new defaults

---

## Phase 3 — Core Initialization Unification

### Objective
Centralize initialization logic in `core.py`.

### Tasks
In `src/peft/tuners/jora/core.py`:
- add `initialize_core_params(...)`
- route `DiagCore`, `BlockCore`, `LowRankCore` through it
- keep `SelectiveDiagCore` behavior explicit and separate if needed

### Design Requirements
- initialization must remain core-specific
- function centralizes policy, not forces uniform math
- preserve identity-ish init for block / selective paths where required

### Validation Gate
- unit tests for parameter statistics
- sanity check that constructor behavior is deterministic under seed control
- no regression in existing JORA tests

### Suggested checks
- param count unchanged where expected
- init std approximately matches config
- zero-init and small-init both supported intentionally

### Rollback / Fallback
If the unified helper makes code less clear or breaks core-specific invariants:
- keep thin per-core wrappers
- centralize only shared utilities, not full policy

---

## Phase 4 — Layer Logic Changes

### Objective
Make layer behavior consistent with the new initialization and core semantics.

### Tasks
In `src/peft/tuners/jora/layer.py`:
- ensure theta init is fully config-driven
- review `compute_delta()` for diag/block/lowrank paths
- remove legacy `tanh` only if tests show it is unnecessary and not relied on elsewhere

### Caution
Do not combine these two changes blindly:
- nonzero init
- removal of `tanh`

This couples two sources of behavior change and makes regression analysis difficult.

### Recommended Order
1. Make theta init configurable
2. Validate behavior
3. Then remove `tanh` behind tests / ablation
4. Validate again

### Validation Gate
Must pass:
- shape / dtype checks
- merge/unmerge consistency
- small-init perturbation bounds
- no obvious explosion in tiny smoke runs

### Rollback / Fallback
If removing `tanh` harms stability:
- keep a flag or legacy branch temporarily
- compare metrics directly rather than forcing simplification prematurely

---

## Phase 5 — New Tests Before New Experiments

### Objective
Add method-specific tests before writing large experiment configs.

### New Test File
`tests/test_jora_diag_path.py`

### Required Coverage
- factory defaults for `diag_path()` and `selective_path()`
- initialization behavior:
  - zero init
  - small nonzero init
- `DiagCore` perturbation bounds at init
- `Diag-only` vs `rotation-enabled` wiring sanity
- merge/unmerge consistency for diag path
- family consistency checks that do not overclaim equivalence

### Validation Gate
All JORA-related tests pass together:

```bash
pytest tests/test_jora.py tests/test_jora_paper_path.py tests/test_jora_diag_path.py -q
```

### Rollback / Fallback
If tests are hard to write because interfaces are unclear, that is a design smell.
Refactor interfaces before adding more experiment surface area.

---

## Phase 6 — Experiment Configs (Only After Code Stabilizes)

### Objective
Create experiment configs only after the implementation path is trustworthy.

### Required Configs
- `configs/run_diag_main.json`
- `configs/run_diag_no_rotation.json`
- `configs/run_jora_block.json`
- `configs/run_jora_lowrank.json`
- `configs/run_lora_baseline.json`
- `configs/run_lora_r2_baseline.json`
- `configs/run_init_ablation.json`

### Config Review Checklist
Each config must clearly specify:
- model
- dataset
- targeted modules
- seed
- init settings
- learning rates
- whether rotation is active
- exact adapter type

### Validation Gate
Before long runs, every config must pass a tiny smoke run:
- tiny sample count
- tiny max steps
- single seed
- save logs and final config snapshot

### Rollback / Fallback
Any config that fails smoke testing is not allowed into the main queue.

---

## Phase 7 — Experiment Ladder (Strict Order)

### Objective
Run experiments in the smallest evidence-producing order.

### Stage 7.1 — Wiring & Stability
Run 1-seed tiny smoke tests for:
- `Diag-only`
- `JORA-Diag`
- `LoRA-r1`

Gate:
- no crashes
- no NaNs
- loss decreases at least slightly

### Stage 7.2 — Main Comparison, Single Seed
Run 1-seed full-ish experiments for:
- `JORA-Diag`
- `Diag-only`
- `LoRA-r1`
- `LoRA-r2`

Gate:
- results are numerically plausible
- no severe instability
- `JORA-Diag` is not obviously dominated

### Stage 7.3 — Main Comparison, 3 Seeds
Only if Stage 7.2 looks healthy.

Run:
- `JORA-Diag`
- `Diag-only`
- `LoRA-r1`
- `LoRA-r2`

Report:
- mean
- std
- parameter count
- throughput / VRAM if available

### Stage 7.4 — Initialization Ablation
Run limited ablation on:
- `theta_init_std`
- `core_init_std`

Goal:
- justify the chosen small nonzero init
- prove results are not a lucky initialization artifact

### Stage 7.5 — Family Expansion
Only after main claim is supported.

Run:
- `JORA-Block`
- `JORA-LowRank`

Questions answered:
- does local coupling help beyond diagonal?
- does rotation help low-rank cores too?

### Stage 7.6 — Extended Tasks / Models
Add:
- second model family
- reasoning tasks like GSM8K / MATH

Only do this after the main method is already credible on the primary setup.

---

## Required Stop Conditions

Stop and reassess if any of the following happen:

- `JORA-Diag` loses clearly to `Diag-only`
- nonzero init causes large unstable perturbations
- `paper_path()` migration breaks backward compatibility in multiple places
- `JORA-LowRank` wins everything and makes the novelty collapse into “LoRA + rotation”
- improvements disappear under 3 seeds

When a stop condition is triggered:
1. do not continue the queue
2. write a short failure analysis
3. update the hypothesis and next step

---

## Recommended Execution Rhythm For The Agent

For long-running work, the agent should operate in micro-cycles:

1. propose one concrete next step
2. implement only that step
3. run the smallest relevant validation
4. summarize result
5. decide whether to proceed or backtrack

Example:
- Step A: add `core_init_std`
- Validate: config tests
- Step B: add `diag_path()`
- Validate: config tests
- Step C: add initialization helper
- Validate: init tests

Do not batch unrelated edits and “hope the suite passes”.

---

## Suggested Progress Template

For each completed step, record:

- change made
- expected effect
- validation run
- result
- next action
- rollback needed? yes/no

Example:

```text
Step: Added `core_init_std` to `JoraConfig`
Expected: core-specific init scale becomes configurable
Validation: `pytest tests/test_jora.py -q`
Result: passed
Next: add `diag_path()` factory
Rollback: no
```

---

## Result Tables To Fill Later

### Table A — Main Comparison
- JORA-Diag
- Diag-only
- LoRA-r1
- LoRA-r2

Metrics:
- task score(s)
- mean ± std
- trainable params
- tokens/s
- peak VRAM

### Table B — Initialization Ablation
- theta init std
- core init std
- score
- stability notes

### Table C — Family Comparison
- JORA-Diag
- JORA-Block
- JORA-LowRank
- SelectiveDiagCore

### Table D — Rotation Contribution
- Diag-only vs JORA-Diag
- optional lowrank vs rotated-lowrank bridge

---

## Final Acceptance Criteria

The refactoring plan is only considered successful if all of the following hold:

- code path is simpler, not more fragile
- initialization behavior is explicit and tested
- main method has a reproducible config and smoke-tested run path
- primary claim is supported by 3-seed evidence
- the contribution of rotation is empirically isolated
- all major negative results are documented, not hidden

