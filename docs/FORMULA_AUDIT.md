# FORMULA_AUDIT

**Last updated: 2026-04-28**

**Option C (residualized DiagCore)**: CATASTROPHIC FAIL. DiagCore reverted to additive default (2026-04-28). See `docs/JORA_OPTION_C_POSTMORTEM.md`. The C1.5 residualized implementation and C1.6 exact-merge patch were both reverted; the C1.6 basis-probing merge fix was retained as it is a standalone improvement.

This document is the read-only correctness audit of the current JORA implementation.

Audit scope:
- `src/peft/tuners/jora/layer.py`
- `src/peft/tuners/jora/core.py`
- `src/peft/tuners/jora/rotation.py`
- `src/peft/tuners/jora/config.py`
- merge / save / load behavior in the current JORA layer + PEFT state flow
- JORA-related tests in `tests/test_jora.py`, `tests/test_jora_diag_path.py`, `tests/test_jora_paper_path.py`

This document answers one question above all others:

> What formulas does the current code actually implement, and what claims are therefore safe or unsafe?

---

## Executive verdict

The current repository does **not** implement a single unified JORA operator.

There are at least two distinct forward semantics:

1. **JORA-Diag / JORA-NoRot / Block / LowRank**
   - implemented as a **legacy additive path**
   - current DiagCore formula is:
     - `Δ(x) = R_L^⊤ Diag(d) R_R x`
   - current NoRot formula is:
     - `Δ(x) = Diag(d) x`
   - this is **not** the residualized full-support formula
     - `R_L^⊤ Diag(1+d) R_R x - x`

2. **JORA-Selective (`SelectiveDiagCore`)**
   - implemented as the **paper-exact residualized selective path**
   - formula is:
     - `Δ(x) = R_L^⊤ D_sel R_R x - P_U x`
   - where `D_sel = P_U + Diag(δ)_U`

This distinction is the current correctness gate for any method writeup.

---

## 1. Exact forward formula for JORA-Diag

### Code path
Relevant code:
- `src/peft/tuners/jora/layer.py:488`
- `src/peft/tuners/jora/layer.py:543`
- `src/peft/tuners/jora/core.py:269`
- `src/peft/tuners/jora/rotation.py:183`

### Actual implementation
For `core in {diag, block, lowrank}` the code path in `compute_delta()` is:
1. apply right rotation to input:
   - `x_rot = R_R x`
2. apply core operator:
   - `y_core = core.apply_to_vector(x_rot)`
3. apply left rotation transpose:
   - `y = R_L^⊤ y_core`
4. return `y`

So the current JORA-Diag delta is:

- `Δ_diag(x) = R_L^⊤ Diag(d) R_R x`

and the full adapted layer is:

- `y = W_0 x + R_L^⊤ Diag(d) R_R x`

### Important non-formula facts
- There is **no** `-x` subtraction in this path.
- There is **no** implicit `I + d` in `DiagCore`.
- `DiagCore.apply_to_vector()` multiplies by raw `diag_params`, not by `(1 + diag_params)`.
- Therefore the current JORA-Diag is an **additive diagonal operator in a rotated basis**, not a residualized identity-centered operator.

### Final verdict for JORA-Diag (pre-C1.5, additive form)
The correct current formula is:
- `Δ(x) = R_L^⊤ Diag(d) R_R x`

The following formula was **not** implemented before C1.5:
- `Δ(x) = R_L^⊤ Diag(1+d) R_R x - x`

---

## 1b. Exact forward formula for JORA-Diag (RESIDUALIZED, C1.5 — REVERTED 2026-04-28)

**Status**: REVERTED. This section is kept for historical record only.

The Option C residualized refactor was attempted but catastrophically failed (eval loss 15.7/19.2 vs base 5.5). DiagCore has been reverted to the additive form above. Do not cite the formulas in this section as current.

### Historical implementation (before revert)

The DiagCore `apply_to_vector()` was temporarily changed to:
```python
y_first = x_first + x_first * self.diag_params  # = x_first * (1 + d)
```

The `compute_delta()` path was temporarily changed to:
```python
x_rot = self._apply_side_rotation(x, is_left_side=False)
y_core = self.core.apply_to_vector(x_rot)  # (I + Diag(d)) @ x_rot
y = self._apply_side_rotation(y_core, is_left_side=True)
delta = y.to(x.dtype) - x
return delta
```

The intended formula was:
- `Δ_diag(x) = R_L^⊤ (I + Diag(d)) R_R x - x`

**Why it failed**: `R_L^T @ R_R ≠ I` when left/right rotation pairs are independently sampled. This caused the residualized operator to catastrophically reshape hidden-state geometry. See `docs/JORA_OPTION_C_POSTMORTEM.md`.

---

## 2. Exact forward formula for JORA-NoRot

### Code path
NoRot is achieved by disabling rotation slots, i.e. `S_L=0` and `S_R=0`.

Relevant code:
- `src/peft/tuners/jora/config.py` rotation counts
- `src/peft/tuners/jora/layer.py` side-rotation application logic
- tests around `S_L=0`, `S_R=0` in `tests/test_jora.py`

### Actual implementation
When `S_L=0` and `S_R=0`, no theta parameters are created and the side rotations reduce to identity.

So for DiagCore:
- `x_rot = x`
- `y_core = Diag(d) x`
- `y = y_core`

Thus:
- `Δ_norot(x) = Diag(d) x`

and the full adapted layer is:
- `y = W_0 x + Diag(d) x`

### Final verdict for JORA-NoRot
Yes, current NoRot is strictly consistent with:
- `Δ(x) = Diag(d) x`

This is the correct mechanism baseline for the current DiagCore path.

---

## 3. Exact forward formula for JORA-Selective

### Code path
Relevant code:
- `src/peft/tuners/jora/core.py:11`
- `src/peft/tuners/jora/core.py:66`
- `src/peft/tuners/jora/core.py:81`
- `src/peft/tuners/jora/layer.py:488`

### Actual implementation
`SelectiveDiagCore` stores support indices `U` and trainable support deltas `δ`.

Its internal operator is:
- `D_sel = P_U + Diag(δ)_U`

Operationally:
1. rotate input:
   - `x_rot = R_R x`
2. apply support-restricted scaled identity:
   - `y_sel = D_sel x_rot`
   - only active support coordinates are kept; outside support output is zero
3. rotate back on the left:
   - `y_rotated = R_L^⊤ y_sel`
4. subtract support projection in original input coordinates:
   - `proj_x = P_U x`
5. return:
   - `Δ_sel(x) = y_rotated - proj_x`

So the exact formula is:
- `Δ_sel(x) = R_L^⊤ D_sel R_R x - P_U x`

### Final verdict for JORA-Selective
Yes, current Selective is a residualized selective-support operator:
- `Δ(x) = R_L^⊤ D_sel R_R x - P_U x`

This is the cleanest paper-exact path in the current codebase.

---

## 4. Exact merge formula

### 4.1 General merge structure
Relevant code:
- `src/peft/tuners/jora/layer.py:722`
- `src/peft/tuners/jora/layer.py:809`
- `src/peft/tuners/jora/layer.py:867`
- `src/peft/tuners/jora/layer.py:1127`

The layer merge path adds a weight-space delta to the frozen base weight:
- `W_merged = W_0 + ΔW`

and unmerge subtracts the same stored/computed delta:
- `W_0 = W_merged - ΔW`

### 4.2 Merge formula for JORA-Diag / NoRot / Block / LowRank

**C1.6 fix (preserved after Option C revert)**: DiagCore now uses exact basis-probing merge, the same method as SelectiveDiagCore. Non-Selective cores are also covered by the same exact path in `_compute_weight_delta_simple`.

For DiagCore, the intent is a weight-space operator corresponding to the forward delta:
- `Δ(x) = R_L^⊤ C R_R x`

The basis-probing reconstruction in `_compute_weight_delta_simple` recovers the exact dense linear map by probing with one-hot basis vectors. This matches `compute_delta(x)` for all nonzero theta values.

### 4.3 Merge formula for JORA-Selective
For `SelectiveDiagCore`, merge is handled differently.

The code probes the adapter on basis vectors and builds the exact dense operator induced by the current forward path.

So for square layers:
- `ΔW_sel` is constructed such that
  - `x ↦ ΔW_sel x`
  - exactly matches `compute_delta(x)`

Thus, for Selective:
- `W_merged = W_0 + ΔW_sel`
- where `ΔW_sel` is the exact dense operator equivalent of
  - `R_L^⊤ D_sel R_R x - P_U x`

### Final verdict on merge equivalence
- **Selective**: merge is designed to be forward-equivalent by basis probing.
- **Diag / NoRot / Block / LowRank**: merge is not implemented as the same level of exact forward-equivalent reconstruction; it is a conservative approximation path.

So the answer to “merge 是否和 forward 等价” is:
- **Selective**: yes, by construction for supported shapes.
- **Current DiagCore mainline**: not guaranteed exact by the current merge implementation design.

---

## 5. Zero-init behavior for each variant

### 5.1 JORA-Diag zero-init behavior
Relevant code:
- `src/peft/tuners/jora/core.py:272`
- `src/peft/tuners/jora/layer.py:543`

If `zero_init_core=True`, then:
- `DiagCore.diag_params = 0`

So:
- `Diag(d) = 0`
- `Δ_diag(x) = R_L^⊤ 0 R_R x = 0`

Therefore the adapted layer becomes:
- `y = W_0 x`

This is zero function change regardless of theta values, because the core output is identically zero.

### Important nuance
This is a zero function change, but **not** an identity-centered parameterization.
It is zero because the additive adapter is zero.

### 5.2 JORA-NoRot zero-init behavior
If `S_L=0`, `S_R=0`, and `zero_init_core=True`, then:
- `Δ_norot(x) = Diag(d)x = 0`

Thus:
- `y = W_0 x`

So NoRot also has strict zero function change under zero core init.

### 5.3 JORA-Selective zero-init behavior
Relevant code comments explicitly describe this.

If support is set, `δ = 0`, and `theta = 0`, then:
- `R_L = R_R = I`
- `D_sel = P_U`
- `Δ_sel(x) = P_U x - P_U x = 0`

So Selective has zero function change at zero init **when theta is zero**.

### Important nuance for Selective
The code comments also state:
- at `δ = 0`, `D_sel = I_U`
- `R_L^⊤ I_U R_R x - P_U x` is not zero in general if theta is nonzero

So for Selective, strict zero function change requires the zero-init state with zero theta.

### 5.4 BlockCore zero-init behavior
Current BlockCore path is also additive.
If `zero_init_core=True`, blocks and remainder are zero, so:
- `Δ_block(x) = R_L^⊤ 0 R_R x = 0`

Thus zero function change holds.

### 5.5 LowRankCore zero-init behavior
The tests explicitly check that zero init preserves a zero operator without deadlocking gradients.
If `zero_init_core=True`, the current low-rank operator is initialized as zero in effect, so:
- `Δ_lowrank(x) = 0`

Thus zero function change holds.

### Final verdict on “zero-init 是否真的 zero function change”
- **JORA-Diag / NoRot / Block / LowRank**: yes, with `zero_init_core=True`, because the additive operator is zero.
- **JORA-Selective**: yes, in the zero-init state with theta also zero; that is the intended paper-path initialization.
- **JORA-Diag with `zero_init_core=False`**: no strict zero function change is guaranteed, because the additive core is nonzero at init.

---

## 6. Mismatch between current implementation and intended paper formula

This is the most important section in the audit.

### Mismatch A — JORA-Diag was attempted as residualized full-support (Option C) but failed

A full-support residualized refactor was attempted:
- Intended paper-clean formula: `Δ(x) = R_L^⊤ Diag(1+d) R_R x - x`
- Current JORA-Diag implementation: `Δ(x) = R_L^⊤ Diag(d) R_R x`
- **Outcome**: Catastrophic failure (loss 15.7/19.2 vs base 5.5). Option C reverted 2026-04-28.

Consequences:
- The current mainline is additive, not residualized
- `DiagCore` uses raw diagonal coefficients, not `(1 + d)`
- Paper text must not claim the residualized full-support formula

### Mismatch B — merge semantics differ across variants
- Selective has forward-equivalent merge by exact basis probing.
- DiagCore now also uses exact basis-probing (C1.6 fix, preserved after Option C revert).
- BlockCore and LowRankCore use the same basis-probing path.

Consequences:
- A uniform “exact mergeability” claim is now safe for DiagCore and SelectiveDiagCore.
- Deployment wording should still note that rectangular layers are not supported for Selective.

### Mismatch C — “zero-change init” means different things across variants
- Diag / NoRot / Block / LowRank: zero-change comes from a zero additive operator
- Selective: zero-change comes from residual cancellation under zero theta and zero delta

Consequences:
- the same phrase should not be used as if all variants share the same mechanism

### Mismatch D — intended unified family story does not yet hold in code
A unified family story would ideally align:
- full-support residualized JORA-Diag
- no-rotation full-support diagonal baseline
- selective residualized support variant

Current code does not fully realize that family-level unification.

---

## 7. Required code fixes, if any

These are not mandatory for running experiments, but they are mandatory if the paper wants a cleaner theory-to-code match.

### Required fix 1 — Residualized DiagCore (Option C) — REVERTED 2026-04-28

**Status**: REVERTED. Catastrophic failure (loss 15.7/19.2 vs base 5.5). The residualized refactor was entirely reverted; code default is now additive DiagCore. See `docs/JORA_OPTION_C_POSTMORTEM.md`.

### Required fix 2 — Make merge semantics explicit per variant
**Status**: DONE. C1.6 implemented exact basis-probing for all cores (DiagCore, BlockCore, LowRankCore, SelectiveDiagCore) in _compute_weight_delta_simple. The basis-probing path is the unified approach applied uniformly. SelectiveDiagCore and DiagCore are now on equal footing for exact mergeability.

### Required fix 3 — Freeze paper wording around zero-init
Current status:
- zero-init behavior differs mechanistically by variant

Required action:
- define variant-specific zero-init wording in the paper or docs
- do not use a single over-generalized “identity-preserving init” claim

### Required fix 4 — Produce and maintain a variant-level formula contract
Required action:
- keep this audit synchronized with code changes
- any change to `compute_delta()` or merge path should update the audit before new experiments are trusted

### Required fix 5 — If JORA-Diag remains mainline, add a stronger merge equivalence test
Current tests already probe some merge behavior, but the current mainline would benefit from a stricter exactness test or an explicit limitation note.

---

## 8. Tests that must pass before experiments are trusted

This section lists the most important tests or test classes that should be treated as trust gates.

### Must-pass existing JORA tests
- `tests/test_jora.py`
- `tests/test_jora_diag_path.py`
- `tests/test_jora_paper_path.py`

### Specific correctness checks already present and especially important

#### A. Save/load roundtrip must preserve adapter state
Relevant tests in `tests/test_jora.py`:
- JORA save/load roundtrip
- PEFT state dict roundtrip preserving frozen support and pair state

These are mandatory before trusting resumed training or comparison across runs.

#### B. NoRot wiring must be correct
Relevant tests:
- `S_L=0`, `S_R=0` behavior in `tests/test_jora.py`
- diag-path tests in `tests/test_jora_diag_path.py`

These are mandatory because JORA-NoRot is the key mechanism baseline.

#### C. Merge/unmerge sanity must pass
Relevant tests:
- merge consistency tests in `tests/test_jora_diag_path.py`

These are mandatory before trusting any deployment or merged-weight experiment.

#### D. Selective paper-path semantics must pass
Relevant tests:
- `tests/test_jora_paper_path.py`

These are mandatory because Selective is the only current paper-exact path.

#### E. Zero-init gradient liveness must pass
Relevant tests:
- low-rank zero-init gradient liveness in `tests/test_jora.py`
- diag-path zero-init related tests in `tests/test_jora_diag_path.py`

These are mandatory before interpreting negative results as mechanism failures rather than dead-parameter bugs.

### Additional trust gate strongly recommended
Before any new experiment batch, the following test command family should pass in the current environment:
- JORA broad tests
- diag-path focused tests
- paper-path focused tests

Practically, the trusted gate is:
- `tests/test_jora.py`
- `tests/test_jora_diag_path.py`
- `tests/test_jora_paper_path.py`

If any of these fail, experiment conclusions should be treated as provisional.

---

## Final answers to the critical questions

### Q1. JORA-Diag 当前 forward 到底是哪个？（After C1.5）

After the C1.5 residualized refactor:
- `Δ(x)=R_L^⊤ (I+Diag(d)) R_R x - x`

The old additive formula (`Δ=R_L^⊤ Diag(d) R_R x`) is **no longer current**.

### Q2. NoRot 是不是严格对应 `Diag(d)x`？
Yes.
Current NoRot is:
- `Δ(x)=Diag(d)x`

### Q3. Selective 是不是 residualized `D_U - P_U` 路径？
Yes.
Current Selective is:
- `Δ(x)=R_L^⊤ D_sel R_R x - P_U x`

### Q4. merge 是否和 forward 等价？
- **Selective**: yes, by exact basis probing construction for supported shapes.
- **DiagCore (current)**: yes, C1.6 exact basis-probing merge is preserved after Option C revert.

### Q5. zero-init 是否真的 zero function change？
- **Diag / NoRot / Block / LowRank with zero core init**: yes.
- **Selective with zero theta and zero delta**: yes.
- **Nonzero-initialized DiagCore mainline**: no strict zero function change guarantee.

---

## Bottom line

The current JORA codebase supports the following safe summary:

- `JORA-Diag` is a **structured additive diagonal adapter** in a sparse rotation basis.
- `JORA-NoRot` is exactly the diagonal-only baseline for that additive path.
- `JORA-Selective` is the residualized paper-exact selective-support variant.
- The current codebase does **not** justify writing one unified residualized formula for all variants.
- Option C (residualized full-support DiagCore) was attempted and catastrophically failed. The revert is complete.
- C1.6 exact basis-probing merge is preserved and applies uniformly to DiagCore and SelectiveDiagCore.
- This audit is the correctness gate that should be respected before experiments are interpreted or papers are written.
