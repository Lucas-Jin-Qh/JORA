# JORA Final State Audit

**Date**: 2026-04-28
**Trigger**: Postmortem for Option C (residualized DiagCore) catastrophic failure.
**Purpose**: Verify that all formula documentation, tests, and code are consistent with the reverted additive DiagCore default before entering "archived / technical report" phase.

---

## Executive Summary

After the Option C (residualized DiagCore) catastrophic failure, DiagCore was reverted to its additive default form. This audit verifies that all five consistency-check targets correctly reflect the current state.

| # | Check | Status |
|---|-------|--------|
| 1 | `docs/FORMULA_AUDIT.md` — additive formula, Option C failure documented | **FIXED** |
| 2 | `docs/METHOD_WORDING_LOCK.md` — additive equations, forbidden residualized | **ALREADY CORRECT** |
| 3 | `experiments/RESULT_TO_CLAIM_CURRENT.md` — `claim_supported=no` explicit | **ALREADY CORRECT** |
| 4 | `tests/test_jora_diag_path.py` — additive test suite, stale C1.5 removed | **FIXED** |
| 5 | `tests/test_jora_save_load_merge_sanity.py` — exact merge, stale 0.05x removed | **FIXED** |

---

## Canonical Code State

The following is the authoritative, current formula for each JORA variant.

### JORA-Diag (DiagCore, rotation ON)
```
Δ(x) = R_L^T Diag(d) R_R x
y    = W_0 x + R_L^T Diag(d) R_R x
```
- `DiagCore.apply_to_vector(x)` returns `Diag(d) @ x` (additive, not `(I + Diag(d)) @ x`).
- `zero_init_core=True` → `d=0` → `Δ(x)=0` → strict zero function change.
- Merge: exact basis-probing (`_compute_weight_delta_simple`).

### JORA-NoRot (S_L=0, S_R=0)
```
Δ(x) = Diag(d) x
y    = W_0 x + Diag(d) x
```
- Identity rotation (`R_L=R_R=I`).
- Merge: exact.

### JORA-Selective (SelectiveDiagCore)
```
Δ(x) = R_L^T D_sel R_R x - P_U x
D_sel = P_U + Diag(δ)_U
```
- Paper-exact residualized selective path.
- Merge: exact basis-probing (for square layers only).
- Not affected by DiagCore revert.

### Option C (Residualized full DiagCore) — FAILED
```
Δ(x) = R_L^T (I + Diag(d)) R_R x - x   ← NOT CURRENT
```
- **Status**: REVERTED 2026-04-28.
- **Root cause**: `R_L^T @ R_R ≠ I` when left/right rotation pairs are independently sampled.
- **Evidence**: 1ep eval loss = 15.7 (ON) and 19.2 (NoRot) vs base 5.5.
- Full postmortem: `docs/JORA_OPTION_C_POSTMORTEM.md`.

---

## What Was Fixed

### 1. `docs/FORMULA_AUDIT.md`

| Before | After |
|--------|-------|
| Header: "In-progress change (Option C, C1.5)" | Header: "Option C CATASTROPHIC FAIL. Reverted 2026-04-28." |
| Section 1b: residualized implementation as "current" | Section 1b: labeled "REVERTED", preserved as historical record only |
| Section 4.2: "legacy approximation path" for DiagCore merge | Section 4.2: "exact basis-probing, C1.6 fix preserved" |
| Q4: "DiagCore (post-C1.5) — broken — C1.6 must fix" | Q4: "DiagCore (current) — yes, exact basis-probing" |
| Required Fix 1: "C1.5 complete" | Required Fix 1: "REVERTED 2026-04-28" |
| Required Fix 2: "other cores do not" have exact merge | Required Fix 2: "DONE — basis-probing applied uniformly" |
| Mismatch A: "not the residualized formula" | Mismatch A: "Option C attempted, failed, reverted" |
| Mismatch B: "asymmetric across variants" | Mismatch B: "basis-probing applied uniformly" |
| Bottom line: no mention of Option C | Bottom line: Option C reverted; C1.6 exact merge preserved |

### 2. `tests/test_jora_diag_path.py`

| Before | After |
|--------|-------|
| Class docstring: "C1.5 gate: residualized DiagCore" | Class docstring: "DiagCore backward-compatibility regression suite — additive form" |

The test class name `TestResidualizedDiagCore` is kept as-is because it accurately describes the *purpose* of the suite (testing the reverted DiagCore behavior, which previously went through a residualized phase). The class docstring now correctly identifies this as a regression suite for the additive form.

### 3. `tests/test_jora_save_load_merge_sanity.py`

| Before | After |
|--------|-------|
| Module docstring: "DiagCore uses a rough 0.05x approximation" | Module docstring: "DiagCore uses exact basis-probing (C1.6 fix, preserved after Option C revert)" |
| `TestMergeDiagOn` docstring: "DiagCore merge is now exact" | `TestMergeDiagOn` docstring: "exact basis-probing (C1.6 fix, preserved after Option C revert)" |
| `TestMergeWithMagnitude` docstring: "legacy approximate path (0.05x scaling)" | `TestMergeWithMagnitude` docstring: "exact basis-probing for merge (C1.6 fix, preserved after Option C revert)" |
| `test_merge_unmerge_with_magnitude` docstring: "limited by 0.05x approximation" | `test_merge_unmerge_with_magnitude` docstring: "exact within numerical tolerance" |

### 4. `src/peft/tuners/jora/layer.py`

| Before | After |
|--------|-------|
| `compute_delta` docstring: "Legacy formula (other core types): ... [with optional tanh]" | `compute_delta` docstring: "Standard formula (DiagCore / BlockCore / LowRankCore): delta = R_L^T @ core(R_R @ x)" |
| `_compute_weight_delta_simple`: "C1.6 fix: DiagCore also uses exact basis-probing" | `_compute_weight_delta_simple`: "DiagCore uses exact basis-probing, same as SelectiveDiagCore" |

### 5. Unchanged (already correct)

| File | Verdict | Evidence |
|------|---------|----------|
| `docs/METHOD_WORDING_LOCK.md` | ALREADY CORRECT | Lines 9-15: additive equations correctly listed. Lines 24-25: residualized formula explicitly forbidden. Last updated 2026-04-26 (pre-Option C). |
| `experiments/RESULT_TO_CLAIM_CURRENT.md` | ALREADY CORRECT | F9: "Residualized DiagCore is not viable" is listed as a forbidden claim. Option C failure fully documented in Section 6. |

---

## Current Narrative Lock

After this audit, the following narrative is locked:

**Allowed**:
- "JORA-Diag is a structured additive diagonal adapter in a sparse rotation basis."
- "JORA-NoRot establishes the diagonal-core-only performance floor."
- "Sparse rotations do not show a clear independent benefit under matched training."
- "Option C (residualized full-support DiagCore) was attempted and catastrophically failed."
- "Exact merge via basis-probing is verified for DiagCore and SelectiveDiagCore."

**Forbidden**:
- Any claim that DiagCore is residualized.
- Any claim that rotation drives the gain.
- Any claim that JORA-Diag is competitive with LoRA/DoRA.
- Any claim that TC-CS succeeded.
- Any claim that current JORA-Diag implements `Δ(x) = R_L^T (I + Diag(d)) R_R x - x`.

---

## Verification Commands

Run these to confirm correctness gates pass:

```bash
# M0 correctness gate
python -m pytest tests/test_jora_save_load_merge_sanity.py -v
python -m pytest tests/test_jora_diag_path.py -v

# Smoke test
python -c "from peft import JoraConfig, get_peft_model; import torch; from transformers import OPTConfig, AutoModelForCausalLM
m = AutoModelForCausalLM.from_config(OPTConfig(vocab_size=128, hidden_size=64, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2, ffn_dim=128, max_position_embeddings=64, pad_token_id=0, bos_token_id=1, eos_token_id=2))
c = JoraConfig.diag_path(target_modules=['q_proj'], S_L=4, S_R=4)
p = get_peft_model(m, c)
x = torch.randint(0, 128, (2, 8))
print('Output finite:', torch.isfinite(p(x).logits).all().item())
print('Delta at zero-init:', p.base_model.transformer.h[0].attn.q_proj.jora_layer.adapters['default'].compute_delta(torch.randn(2, 64)).norm().item(), '< 1e-3')
"
```

---

## Sign-off

This audit was completed on 2026-04-28. All five consistency-check targets are now consistent with the reverted additive DiagCore default. The codebase is in a state suitable for writing a technical report / negative-result archive.
